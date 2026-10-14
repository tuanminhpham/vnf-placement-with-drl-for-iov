from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .models import DEFAULT_DELAY_CALIBRATION, NODE_KIND_TO_FLOAT, RSU, IoVScenario


DEFAULT_INVALID_PENALTY = 50.0
DEFAULT_COST_WEIGHT = 1.0
DEFAULT_DELAY_WEIGHT = 0.15
OBSERVATION_SCHEMA_VERSION = 9
DEFAULT_REWARD_CONFIG = {
    "invalid_penalty": DEFAULT_INVALID_PENALTY,
    "cost_weight": DEFAULT_COST_WEIGHT,
    "delay_weight": DEFAULT_DELAY_WEIGHT,
}


@dataclass(frozen=True)
# kết quả ktra 1 action : xem có hợp lệ không, lý do lỗi, cost, delay, path
class PlacementCheck:
    feasible: bool
    reason: str
    compute_cost: float = 0.0
    migration_cost: float = 0.0
    delay: float = 0.0
    cpu_need: float = 0.0
    path_to_vnf: Tuple[int, ...] = ()
    path_to_destination: Tuple[int, ...] = ()


@dataclass(frozen=True)
class PlacementAllocation:
    node_id: int
    cpu_need: float
    bandwidth: float
    compute_cost: float
    migration_cost: float
    delay: float
    path_to_vnf: Tuple[int, ...]
    path_to_destination: Tuple[int, ...]

# khởi tạo môi trường, tính số node/request, observation size, action space, shortest paths
class DPVEnv(gym.Env): 
    """Gymnasium environment for Admission-Aware Sequential VNF Placement with Deep Reinforcement Learning for Internet of Vehicles."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: IoVScenario,
        invalid_penalty: float = DEFAULT_INVALID_PENALTY,
        cost_weight: float = DEFAULT_COST_WEIGHT,
        delay_weight: float = DEFAULT_DELAY_WEIGHT,
        compute_action_mask: bool = True,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.invalid_penalty = float(invalid_penalty)
        self.cost_weight = float(cost_weight)
        self.delay_weight = float(delay_weight)
        self.compute_action_mask = bool(compute_action_mask)
        delay_calibration = (scenario.metadata or {}).get("delay_calibration", {})
        self.wireless_access_base_delay_ms = float(
            delay_calibration.get(
                "wireless_access_base_delay_ms",
                DEFAULT_DELAY_CALIBRATION["wireless_access_base_delay_ms"],
            )
        )

        self.node_count = scenario.node_count
        self.request_count = scenario.request_count
        self.max_chain_len = scenario.max_chain_len
        self.tasks: List[Tuple[int, int, int]] = []
        self._path_cache: Dict[Tuple[int, int, float], Dict[int, int]] = {}
        self._action_check_cache: Dict[Tuple[int, bool, bool], PlacementCheck] = {}

        self.node_cpu = np.asarray([node.cpu_capacity for node in scenario.nodes], dtype=np.float32)
        self.node_cost = np.asarray([node.cpu_cost for node in scenario.nodes], dtype=np.float32)
        self.node_delay = np.asarray([node.routing_delay for node in scenario.nodes], dtype=np.float32)
        self.node_kind = [node.kind for node in scenario.nodes]
        self.adjacency = [
            np.flatnonzero(scenario.links[node_id] > 0).astype(np.int64).tolist()
            for node_id in range(self.node_count)
        ]
        self.rsu_node_ids = [node.node_id for node in scenario.nodes if node.kind == RSU]
        # Keep every topology node in the raw action space so the unmasked
        # ablation is genuinely unmasked.  The strict mask removes transit and
        # other infeasible actions before the policy selects an action.
        self.placement_node_ids = [node.node_id for node in scenario.nodes]
        self.node_x = np.asarray(
            [0.0 if node.x is None else node.x for node in scenario.nodes],
            dtype=np.float32,
        )
        self.node_y = np.asarray(
            [0.0 if node.y is None else node.y for node in scenario.nodes],
            dtype=np.float32,
        )
        self.has_vehicle_coordinates = bool(self.rsu_node_ids) and all(
            scenario.nodes[node_id].x is not None
            and scenario.nodes[node_id].y is not None
            for node_id in self.rsu_node_ids
        )
        self.max_cpu = float(max(np.max(self.node_cpu), 1.0))
        self.max_cost = float(max(np.max(self.node_cost), 1.0))
        self.max_delay = float(max(np.max(self.node_delay), 1.0))
        self.max_bandwidth = float(max(np.max(scenario.bandwidth), 1.0))
        self.max_access_capacity = float(max(np.max(scenario.access_capacity), 1.0))
        self.max_link_capacity = float(max(np.max(scenario.links), 1.0))
        self.max_delay_budget = float(max(req.delay_budget for req in scenario.requests))
        self.max_cpu_per_traffic = float(
            max(function.cpu_per_traffic for function in scenario.functions)
        )
        active_task_counts = [
            sum(
                len(request.sfc)
                for request_id, request in enumerate(scenario.requests)
                if bool(scenario.active_requests[time_slot, request_id])
            )
            for time_slot in range(scenario.horizon)
        ]
        self.max_tasks_per_slot = float(max(active_task_counts, default=1))
        self.max_active_requests_per_slot = float(
            max(np.max(np.sum(scenario.active_requests, axis=1)), 1)
        )
        processing_delays = [
            delay
            for function in scenario.functions
            for delay in function.processing_delay.values()
        ]
        self.max_processing_delay = float(max(max(processing_delays, default=1.0), 1.0))
        self.request_feature_size = int(scenario.request_features.shape[2])
        self.slot_feature_size = int(scenario.slot_features.shape[1])
        self.request_feature_max = np.maximum(
            np.max(np.abs(scenario.request_features), axis=(0, 1)), 1.0
        )
        self.slot_feature_max = np.maximum(
            np.max(np.abs(scenario.slot_features), axis=0), 1.0
        )

        self.core_feature_size = 14
        self.node_feature_count = 7
        self.node_feature_size = self.node_feature_count * self.node_count
        self.link_feature_size = self.node_count * self.node_count
        self.observation_size = (
            self.core_feature_size
            + self.request_feature_size
            + self.slot_feature_size
            + self.node_feature_size
            + self.link_feature_size
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.observation_size,),
            dtype=np.float32,
        )
        # The paper defines one action per placement node.  Rejection is an
        # environment outcome after an infeasible placement, not a policy action.
        self.action_space = spaces.Discrete(len(self.placement_node_ids))
        self.reset()

    # bắt đầu episode mới, reset tài nguyên CPU/link, delay, placement
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.task_index = 0
        self.done = False
        self.total_compute_cost = 0.0
        self.total_migration_cost = 0.0
        self.total_delay = 0.0
        self.total_rejected_delay_penalty = 0.0
        self.violations = 0
        self.rejected_requests = 0
        self.policy_rejections = 0
        self.infeasible_rejections = 0
        self.violation_rejections = 0
        self.rejection_reason_counts: Dict[str, int] = {}
        self.remaining_cpu = self.node_cpu.copy()
        self.remaining_links = self.scenario.links.copy()
        self._path_cache.clear()
        self._action_check_cache.clear()
        self.active_allocations: Dict[int, List[PlacementAllocation]] = {}
        self.active_release_slots: Dict[int, int] = {}
        self.active_request_placements = np.full(
            (self.request_count, self.max_chain_len), -1, dtype=np.int64
        )
        self.accepted_request_ids: set[int] = set()
        self.rejected_request_ids: set[int] = set()
        self.accepted_request_events: set[Tuple[int, int]] = set()
        self.rejected_request_events: set[Tuple[int, int]] = set()
        self.placements_by_time: List[np.ndarray] = []
        self.rsu_assignments_by_time: List[np.ndarray] = []
        self.previous_placement = np.full(
            (self.request_count, self.max_chain_len), -1, dtype=np.int64
        )
        self._reset_slot_state()
        self._skip_empty_slots()
        return self._observation(), self._info("reset")

    # hàm step(action) : agents sẽ chọn 1 node để đặt VNF hiện tại
    def step(self, action: int):
        if self.done:
            return self._observation(), 0.0, True, False, self._info("done")

        check = self.evaluate_action(int(action)) # gọi evaluate_action()
        request_id, chain_pos, function_id = self.current_task()
        request = self.scenario.requests[request_id]
        request_boundary = chain_pos == len(request.sfc) - 1

        # Nếu sai ràng buộc thì rollback request hiện tại và phạt tức thời
        if not check.feasible:  
            return self.reject_current_request(check.reason, violation=True)
        # current_task(): trả về VNF hiện tại cần đặt: (request_id, chain_pos, function_id)

        bandwidth = self._bandwidth(request_id)
        node_id = self.action_to_node(int(action))
        self._apply_allocation(request_id, chain_pos, node_id, check, bandwidth)
        if request_boundary:
            self._commit_request(request_id)

        reward = -(
            self.cost_weight * check.compute_cost
            + self.delay_weight * check.delay
        )

        self._advance()
        self._action_check_cache.clear()
        return (
            self._observation(),
            float(reward),
            self.done,
            False,
            self._info(
                "ok",
                transition_terminal=request_boundary or self.done,
                request_boundary=request_boundary,
            ),
        )

    def current_task(self) -> Tuple[int, int, int]:
        return self.tasks[self.task_index]
    # action_masks(): tạo mask node hợp lệ theo ràng buộc hiện tại.
    def action_masks(
        self,
        enforce_delay: bool = True,
        enforce_future: bool = False,
    ) -> np.ndarray:
        if self.done:
            return np.zeros(self.action_space.n, dtype=bool)

        return np.asarray(
            [
                self.evaluate_action(
                    action,
                    enforce_delay=enforce_delay,
                    enforce_future=enforce_future,
                ).feasible
                for action in range(self.action_space.n)
            ],
            dtype=bool,
        )

    def action_to_node(self, action: int) -> int:
        if action < 0 or action >= self.action_space.n:
            raise IndexError(action)
        return int(self.placement_node_ids[action])
    # evaluate_action(action): kiểm tra node tương ứng với action có đặt được VNF không
    # nó sẽ phải ktra :
    # - đúng loại RSU/cloud
    # - đủ CPU
    # - có path từ RSU hiện tại tới node đặt VNF
    # - link đủ bandwidth
    # - nếu là VNF cuối thì có path quay về RSU đích
    # - delay gồm cả vehicle->RSU ở VNF đầu và RSU->vehicle ở VNF cuối
    # - tổng delay không vượt quá delta_r
    def evaluate_action(
        self,
        action: int,
        enforce_delay: bool = True,
        enforce_future: bool = False,
    ) -> PlacementCheck:
        cache_key = (int(action), bool(enforce_delay), bool(enforce_future))
        cached = self._action_check_cache.get(cache_key)
        if cached is not None:
            return cached
        check = self._evaluate_action_uncached(action, enforce_delay, enforce_future)
        self._action_check_cache[cache_key] = check
        return check

    def _evaluate_action_uncached(
        self,
        action: int,
        enforce_delay: bool = True,
        enforce_future: bool = False,
    ) -> PlacementCheck:
        if self.done:
            return PlacementCheck(False, "episode_finished")
        if action < 0 or action >= self.action_space.n:
            return PlacementCheck(False, "node_out_of_range")

        node_id = self.action_to_node(action)

        request_id, chain_pos, function_id = self.current_task()
        check = self._evaluate_action_state(
            node_id=node_id,
            request_id=request_id,
            chain_pos=chain_pos,
            function_id=function_id,
            anchor=int(self.current_anchor[request_id]),
            current_delay=float(self.current_delay[request_id]),
            remaining_cpu=self.remaining_cpu,
            remaining_links=self.remaining_links,
            enforce_delay=enforce_delay,
        )
        if not check.feasible or not enforce_future:
            return check

        if not self._has_feasible_completion_after(
            request_id=request_id,
            chain_pos=chain_pos,
            node_id=node_id,
            check=check,
            enforce_delay=enforce_delay,
        ):
            return PlacementCheck(
                False,
                "future_feasibility_exhausted",
                compute_cost=check.compute_cost,
                migration_cost=check.migration_cost,
                delay=check.delay,
                cpu_need=check.cpu_need,
                path_to_vnf=check.path_to_vnf,
                path_to_destination=check.path_to_destination,
            )
        return check

    def _evaluate_action_state(
        self,
        node_id: int,
        request_id: int,
        chain_pos: int,
        function_id: int,
        anchor: int,
        current_delay: float,
        remaining_cpu: np.ndarray,
        remaining_links: np.ndarray,
        enforce_delay: bool,
    ) -> PlacementCheck:
        request = self.scenario.requests[request_id]
        function = self.scenario.function_by_id(function_id)
        bandwidth = self._bandwidth(request_id)
        cpu_need = bandwidth * function.cpu_per_traffic

        if int(self.current_rsu_assignment[request_id]) < 0:
            return PlacementCheck(False, "rsu_not_assigned", cpu_need=cpu_need)
        if (
            chain_pos == 0
            and self._available_access_capacity(request_id) + 1e-9 < bandwidth
        ):
            return PlacementCheck(
                False,
                "access_link_capacity_exceeded",
                cpu_need=cpu_need,
            )

        if self.node_kind[node_id] != function.kind:
            return PlacementCheck(False, "wrong_node_type", cpu_need=cpu_need)
        if remaining_cpu[node_id] + 1e-9 < cpu_need:
            return PlacementCheck(False, "node_capacity_exceeded", cpu_need=cpu_need)

        path_to_vnf = self._path(anchor, node_id, remaining_links, bandwidth)
        if not path_to_vnf:
            return PlacementCheck(False, "no_path_to_vnf", cpu_need=cpu_need)

        link_copy = remaining_links.copy()
        ok, route_delay = self._try_path(link_copy, path_to_vnf, bandwidth)
        if not ok:
            return PlacementCheck(
                False,
                "link_capacity_exceeded_to_vnf",
                cpu_need=cpu_need,
                path_to_vnf=tuple(path_to_vnf),
            )

        delay = (
            route_delay
            + float(self.node_delay[node_id])
            + float(function.processing_delay[node_id])
        )
        if chain_pos == 0:
            delay += self._vehicle_access_delay(request_id, anchor)
        path_to_destination: Sequence[int] = ()
        if chain_pos == len(request.sfc) - 1:
            destination_rsu = int(self.current_rsu_assignment[request_id])
            path_to_destination = self._path(
                node_id,
                destination_rsu,
                link_copy,
                bandwidth,
            )
            if not path_to_destination:
                return PlacementCheck(
                    False,
                    "no_path_to_destination",
                    cpu_need=cpu_need,
                    path_to_vnf=tuple(path_to_vnf),
                )
            ok, return_delay = self._try_path(link_copy, path_to_destination, bandwidth)
            if not ok:
                return PlacementCheck(
                    False,
                    "link_capacity_exceeded_to_destination",
                    cpu_need=cpu_need,
                    path_to_vnf=tuple(path_to_vnf),
                    path_to_destination=tuple(path_to_destination),
                )
            delay += return_delay
            delay += self._vehicle_access_delay(request_id, destination_rsu)

        if enforce_delay and current_delay + delay > request.delay_budget + 1e-9:
            return PlacementCheck(
                False,
                "delay_budget_exceeded",
                cpu_need=cpu_need,
                delay=delay,
                path_to_vnf=tuple(path_to_vnf),
                path_to_destination=tuple(path_to_destination),
            )

        compute_cost = float(self.node_cost[node_id] * cpu_need)
        migration_cost = self._migration_cost(request_id, chain_pos, node_id, bandwidth)
        return PlacementCheck(
            True,
            "ok",
            compute_cost=compute_cost,
            migration_cost=migration_cost,
            delay=delay,
            cpu_need=cpu_need,
            path_to_vnf=tuple(path_to_vnf),
            path_to_destination=tuple(path_to_destination),
        )

    def _has_feasible_completion_after(
        self,
        request_id: int,
        chain_pos: int,
        node_id: int,
        check: PlacementCheck,
        enforce_delay: bool,
    ) -> bool:
        request = self.scenario.requests[request_id]
        next_chain_pos = chain_pos + 1
        if next_chain_pos >= len(request.sfc):
            return True

        bandwidth = self._bandwidth(request_id)
        remaining_cpu = self.remaining_cpu.copy()
        remaining_links = self.remaining_links.copy()
        self._apply_allocation_to_state(
            remaining_cpu,
            remaining_links,
            node_id,
            check,
            bandwidth,
        )
        return self._has_feasible_completion(
            request_id=request_id,
            chain_pos=next_chain_pos,
            anchor=node_id,
            current_delay=float(self.current_delay[request_id]) + check.delay,
            remaining_cpu=remaining_cpu,
            remaining_links=remaining_links,
            enforce_delay=enforce_delay,
        )

    def _has_feasible_completion(
        self,
        request_id: int,
        chain_pos: int,
        anchor: int,
        current_delay: float,
        remaining_cpu: np.ndarray,
        remaining_links: np.ndarray,
        enforce_delay: bool,
    ) -> bool:
        request = self.scenario.requests[request_id]
        if chain_pos >= len(request.sfc):
            return True

        function_id = request.sfc[chain_pos]
        bandwidth = self._bandwidth(request_id)
        for candidate in self.placement_node_ids:
            check = self._evaluate_action_state(
                node_id=candidate,
                request_id=request_id,
                chain_pos=chain_pos,
                function_id=function_id,
                anchor=anchor,
                current_delay=current_delay,
                remaining_cpu=remaining_cpu,
                remaining_links=remaining_links,
                enforce_delay=enforce_delay,
            )
            if not check.feasible:
                continue
            if chain_pos == len(request.sfc) - 1:
                return True

            next_cpu = remaining_cpu.copy()
            next_links = remaining_links.copy()
            self._apply_allocation_to_state(
                next_cpu,
                next_links,
                candidate,
                check,
                bandwidth,
            )
            if self._has_feasible_completion(
                request_id=request_id,
                chain_pos=chain_pos + 1,
                anchor=candidate,
                current_delay=current_delay + check.delay,
                remaining_cpu=next_cpu,
                remaining_links=next_links,
                enforce_delay=enforce_delay,
            ):
                return True
        return False

    # placement_report(): xuất kết quả cuối: số slot hoàn thành, cost, delay, violations, placement
    def placement_report(self) -> Dict[str, object]:
        active_request_events = int(np.sum(self.scenario.active_requests))
        accepted_request_events = len(self.accepted_request_events)
        rejected_request_events = len(self.rejected_request_events)
        penalized_total_delay = self.total_delay + self.total_rejected_delay_penalty
        accepted_denominator = max(accepted_request_events, 1)
        active_denominator = max(active_request_events, 1)
        fully_accepted_sessions = 0
        for request_id in range(self.request_count):
            active_count = int(np.sum(self.scenario.active_requests[:, request_id]))
            accepted_count = sum(
                1 for _, event_request_id in self.accepted_request_events
                if event_request_id == request_id
            )
            if active_count > 0 and accepted_count == active_count:
                fully_accepted_sessions += 1
        return {
            "time_slots_completed": len(self.placements_by_time),
            "horizon": self.scenario.horizon,
            "request_sessions": self.request_count,
            "active_request_events": active_request_events,
            "accepted_request_events": accepted_request_events,
            "accepted_request_sessions": fully_accepted_sessions,
            "sessions_with_acceptance": len(self.accepted_request_ids),
            "rejected_request_events": rejected_request_events,
            "rejected_request_sessions": len(self.rejected_request_ids),
            "violations": self.violations,
            "policy_rejections": self.policy_rejections,
            "infeasible_rejections": self.infeasible_rejections,
            "violation_rejections": self.violation_rejections,
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "total_compute_cost": self.total_compute_cost,
            "total_migration_cost": self.total_migration_cost,
            "served_total_delay": self.total_delay,
            "total_delay": self.total_delay,
            "rejected_delay_penalty": self.total_rejected_delay_penalty,
            "penalized_total_delay": penalized_total_delay,
            "avg_served_delay_per_accepted_request": self.total_delay
            / accepted_denominator,
            "avg_penalized_delay_per_active_request": penalized_total_delay
            / active_denominator,
            "acceptance_ratio_pct": 100.0
            * accepted_request_events
            / active_denominator,
            "rejected_requests": len(self.rejected_request_ids),
            "placements": [placement.tolist() for placement in self.placements_by_time],
            "rsu_assignments": [
                assignment.tolist() for assignment in self.rsu_assignments_by_time
            ],
        }

    # _build_tasks(): biến tất cả SFC thành danh sách các bước đặt VNF
    def _build_tasks(self, time_slot: int) -> List[Tuple[int, int, int]]:
        tasks: List[Tuple[int, int, int]] = []
        for request_id, request in enumerate(self.scenario.requests):
            if not bool(self.scenario.active_requests[time_slot, request_id]):
                continue
            if (time_slot, request_id) in self.rejected_request_events:
                continue
            for chain_pos, function_id in enumerate(request.sfc):
                tasks.append((request_id, chain_pos, function_id))
        return tasks

    # _reset_slot_state(): tạo một snapshot tài nguyên sạch cho placement tại slot t.
    def _reset_slot_state(self) -> None:
        self._release_completed_resources()
        self._release_resources_for_reallocation()
        self.current_delay = np.zeros(self.request_count, dtype=np.float32)
        self.current_rsu_assignment = self._compute_slot_rsu_assignment()
        self.current_anchor = self.current_rsu_assignment.copy()
        self.current_placement = self.active_request_placements.copy()
        self.remaining_access_capacity = self._build_slot_access_capacity()
        self.current_access_reservations: Dict[
            int, Tuple[Tuple[str, int], float]
        ] = {}
        self.current_allocations: List[List[PlacementAllocation]] = [
            [] for _ in range(self.request_count)
        ]
        self.prepared_requests: set[int] = set()
        self.tasks = self._build_tasks(self.t)
        self.task_index = 0
        self._build_task_load_cache()
        self._prepare_current_request()

    def _build_task_load_cache(self) -> None:
        task_count = len(self.tasks)
        self.remaining_task_cpu_load = np.zeros(task_count + 1, dtype=np.float32)
        self.remaining_task_bandwidth_load = np.zeros(task_count + 1, dtype=np.float32)
        self.remaining_request_count = np.zeros(task_count + 1, dtype=np.float32)
        max_cpu_need = max(self.max_bandwidth * self.max_cpu_per_traffic, 1.0)

        for index in range(task_count - 1, -1, -1):
            request_id, _, function_id = self.tasks[index]
            bandwidth = self._bandwidth(request_id)
            function = self.scenario.function_by_id(function_id)
            self.remaining_task_cpu_load[index] = (
                self.remaining_task_cpu_load[index + 1]
                + bandwidth * function.cpu_per_traffic / max_cpu_need
            )
            self.remaining_task_bandwidth_load[index] = (
                self.remaining_task_bandwidth_load[index + 1]
                + bandwidth / self.max_bandwidth
            )
            request_boundary = (
                index == task_count - 1
                or request_id != self.tasks[index + 1][0]
            )
            self.remaining_request_count[index] = (
                self.remaining_request_count[index + 1] + float(request_boundary)
            )

    def _skip_empty_slots(self) -> None:
        while not self.tasks and self.t < self.scenario.horizon:
            self.placements_by_time.append(self.current_placement.copy())
            self.rsu_assignments_by_time.append(self.current_rsu_assignment.copy())
            self.previous_placement = self.current_placement.copy()
            self.t += 1
            if self.t >= self.scenario.horizon:
                self.done = True
                return
            self._reset_slot_state()

    # _advance(): chuyển sang VNF tiếp theo hoặc time slot tiếp theo
    def _advance(self) -> None:
        self.task_index += 1
        if self.task_index < len(self.tasks):
            self._prepare_current_request()
            return

        self._finish_slot()

    def _finish_slot(self) -> None:
        self.active_request_placements = self.current_placement.copy()
        self.placements_by_time.append(self.current_placement.copy())
        self.rsu_assignments_by_time.append(self.current_rsu_assignment.copy())
        # Migration at slot t+1 is defined relative to slot t. Rejected VNFs
        # therefore have no previous host instead of retaining an older host.
        self.previous_placement = self.current_placement.copy()
        self.t += 1
        if self.t >= self.scenario.horizon:
            self.done = True
            return
        self._reset_slot_state()
        self._skip_empty_slots()

    def _prepare_current_request(self) -> None:
        if not self.tasks or self.task_index >= len(self.tasks):
            return
        request_id = self.tasks[self.task_index][0]
        if request_id in self.prepared_requests:
            return

        self._release_active_request_resources(request_id)
        self.current_placement[request_id, :] = -1
        self.current_anchor[request_id] = int(self.current_rsu_assignment[request_id])
        self.current_delay[request_id] = 0.0
        self.current_allocations[request_id].clear()
        self.prepared_requests.add(request_id)

    # _observation(): tạo state vector cho DQN
    def _observation(self) -> np.ndarray:
        if self.done:
            return np.zeros(self.observation_size, dtype=np.float32)

        request_id, chain_pos, function_id = self.current_task()
        request = self.scenario.requests[request_id]
        function = self.scenario.function_by_id(function_id)
        anchor = int(self.current_anchor[request_id])
        bandwidth = self._bandwidth(request_id)
        cpu_need = bandwidth * function.cpu_per_traffic
        chain_len = max(len(request.sfc), 1)
        function_kind = NODE_KIND_TO_FLOAT[function.kind]
        remaining_delay = max(request.delay_budget - float(self.current_delay[request_id]), 0.0)
        remaining_task_count = len(self.tasks) - self.task_index
        request_features = self.scenario.request_features[self.t, request_id]
        if self.request_feature_size:
            request_features = request_features / self.request_feature_max
        slot_features = self.scenario.slot_features[self.t]
        if self.slot_feature_size:
            slot_features = slot_features / self.slot_feature_max

        core = np.asarray(
            [
                max(chain_pos, 0) / max(chain_len - 1, 1),
                max(anchor, 0) / max(self.node_count - 1, 1),
                bandwidth / self.max_bandwidth,
                self._available_access_capacity(request_id) / self.max_access_capacity,
                cpu_need / max(self.max_bandwidth * self.max_cpu_per_traffic, 1.0),
                self.current_delay[request_id] / self.max_delay_budget,
                remaining_delay / self.max_delay_budget,
                request.delay_budget / self.max_delay_budget,
                function_kind,
                self.t / max(self.scenario.horizon - 1, 1),
                remaining_task_count / self.max_tasks_per_slot,
                self.remaining_request_count[self.task_index]
                / self.max_active_requests_per_slot,
                self.remaining_task_cpu_load[self.task_index]
                / self.max_tasks_per_slot,
                self.remaining_task_bandwidth_load[self.task_index]
                / self.max_tasks_per_slot,
            ],
            dtype=np.float32,
        )
        core = np.concatenate(
            [
                core,
                np.clip(request_features.astype(np.float32), 0.0, 1.0),
                np.clip(slot_features.astype(np.float32), 0.0, 1.0),
            ]
        )

        hop_distance = np.full(self.node_count, self.node_count, dtype=np.float32)
        if anchor >= 0:
            for node_id in range(self.node_count):
                path = self._path(anchor, node_id, self.remaining_links, bandwidth)
                hop_distance[node_id] = len(path) - 1 if path else self.node_count

        processing_delay = np.ones(self.node_count, dtype=np.float32)
        for node_id, delay in function.processing_delay.items():
            processing_delay[node_id] = delay / self.max_processing_delay

        previous_node = int(self.previous_placement[request_id, chain_pos])
        previous_placement = np.asarray(
            [1.0 if node_id == previous_node else 0.0 for node_id in range(self.node_count)],
            dtype=np.float32,
        )
        target_kind = function.kind
        node_features = np.concatenate(
            [
                self.remaining_cpu / np.maximum(self.node_cpu, 1.0),
                self.node_cost / self.max_cost,
                self.node_delay / self.max_delay,
                np.asarray(
                    [1.0 if kind == target_kind else 0.0 for kind in self.node_kind],
                    dtype=np.float32,
                ),
                np.clip(
                    hop_distance / max(self.node_count - 1, 1),
                    0.0,
                    1.0,
                ),
                np.clip(processing_delay, 0.0, 1.0),
                previous_placement,
            ]
        )
        link_features = np.clip(
            self.remaining_links / self.max_link_capacity,
            0.0,
            1.0,
        ).reshape(-1)
        observation = np.concatenate([core, node_features, link_features])
        return np.clip(observation, 0.0, 1.0).astype(np.float32)

    def _info(
        self,
        reason: str,
        transition_terminal: bool = False,
        request_boundary: bool = False,
    ) -> Dict[str, object]:
        task_type = "done"
        if not self.done:
            task_type = "place_vnf"
        return {
            "reason": reason,
            "action_mask": (
                self.action_masks()
                if self.compute_action_mask and not self.done
                else np.ones(self.action_space.n, dtype=bool)
                if not self.done
                else np.zeros(self.action_space.n, dtype=bool)
            ),
            "placement_node_ids": self.placement_node_ids,
            "time_slot": self.t,
            "task_index": self.task_index,
            "task_type": task_type,
            "transition_terminal": bool(transition_terminal),
            "request_boundary": bool(request_boundary),
        }

    def _bandwidth(self, request_id: int) -> float:
        return float(self.scenario.bandwidth[self.t, request_id])

    def _access_link_key(self, request_id: int) -> Tuple[str, int]:
        request = self.scenario.requests[request_id]
        vehicle_id = request.vehicle_id or f"request:{request_id}"
        return str(vehicle_id), int(self.current_rsu_assignment[request_id])

    def _build_slot_access_capacity(self) -> Dict[Tuple[str, int], float]:
        capacities: Dict[Tuple[str, int], float] = {}
        for request_id in range(self.request_count):
            if not bool(self.scenario.active_requests[self.t, request_id]):
                continue
            key = self._access_link_key(request_id)
            capacity = max(
                float(self.scenario.access_capacity[self.t, request_id]),
                0.0,
            )
            capacities[key] = min(capacities.get(key, capacity), capacity)
        return capacities

    def _available_access_capacity(self, request_id: int) -> float:
        return float(self.remaining_access_capacity.get(self._access_link_key(request_id), 0.0))

    def _reserve_access_capacity(self, request_id: int, bandwidth: float) -> None:
        key = self._access_link_key(request_id)
        self.remaining_access_capacity[key] -= bandwidth
        self.current_access_reservations[request_id] = (key, bandwidth)

    def _release_current_access_capacity(self, request_id: int) -> None:
        reservation = self.current_access_reservations.pop(request_id, None)
        if reservation is None:
            return
        key, bandwidth = reservation
        self.remaining_access_capacity[key] += bandwidth

    # _migration_cost(): tính chi phí migration nếu VNF đổi node so với time slot trước
    def _migration_cost(
        self, request_id: int, chain_pos: int, node_id: int, bandwidth: float
    ) -> float:
        previous_node = int(self.previous_placement[request_id, chain_pos])
        if previous_node < 0 or previous_node == node_id:
            return 0.0
        path = self._path(previous_node, node_id, self.remaining_links, bandwidth)
        hops = len(path) - 1 if path else self.node_count
        function_id = self.scenario.requests[request_id].sfc[chain_pos]
        function = self.scenario.function_by_id(function_id)
        return float(hops * bandwidth * function.cpu_per_traffic)

    def _apply_allocation(
        self,
        request_id: int,
        chain_pos: int,
        node_id: int,
        check: PlacementCheck,
        bandwidth: float,
    ) -> None:
        if chain_pos == 0:
            self._reserve_access_capacity(request_id, bandwidth)
        self._reserve_path(check.path_to_vnf, bandwidth)
        self._reserve_path(check.path_to_destination, bandwidth)
        self.remaining_cpu[node_id] -= check.cpu_need
        self.current_delay[request_id] += check.delay
        self.current_anchor[request_id] = node_id
        self.current_placement[request_id, chain_pos] = node_id
        self.total_compute_cost += check.compute_cost
        self.total_migration_cost += check.migration_cost
        self.total_delay += check.delay
        self.current_allocations[request_id].append(
            PlacementAllocation(
                node_id=node_id,
                cpu_need=check.cpu_need,
                bandwidth=bandwidth,
                compute_cost=check.compute_cost,
                migration_cost=check.migration_cost,
                delay=check.delay,
                path_to_vnf=check.path_to_vnf,
                path_to_destination=check.path_to_destination,
            )
        )

    def _apply_allocation_to_state(
        self,
        remaining_cpu: np.ndarray,
        remaining_links: np.ndarray,
        node_id: int,
        check: PlacementCheck,
        bandwidth: float,
    ) -> None:
        remaining_cpu[node_id] -= check.cpu_need
        self._reserve_path_in_state(remaining_links, check.path_to_vnf, bandwidth)
        self._reserve_path_in_state(
            remaining_links,
            check.path_to_destination,
            bandwidth,
        )

    def _rollback_request(self, request_id: int) -> None:
        for allocation in reversed(self.current_allocations[request_id]):
            self.remaining_cpu[allocation.node_id] += allocation.cpu_need
            self._release_path(allocation.path_to_vnf, allocation.bandwidth)
            self._release_path(allocation.path_to_destination, allocation.bandwidth)
            self.total_compute_cost -= allocation.compute_cost
            self.total_migration_cost -= allocation.migration_cost
            self.total_delay -= allocation.delay

        self.current_allocations[request_id].clear()
        self._release_current_access_capacity(request_id)
        self.current_delay[request_id] = 0.0
        self.current_anchor[request_id] = int(self.current_rsu_assignment[request_id])
        self.current_placement[request_id, :] = -1

    def _skip_current_request(self, request_id: int) -> None:
        while self.task_index < len(self.tasks) and self.tasks[self.task_index][0] == request_id:
            self.task_index += 1
        if self.task_index >= len(self.tasks):
            self._finish_slot()
        else:
            self._prepare_current_request()

    def reject_current_request(
        self,
        reason: str = "no_valid_action",
        violation: bool = True,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        if self.done:
            return self._observation(), 0.0, True, False, self._info("done")

        request_id, _, _ = self.current_task()
        request = self.scenario.requests[request_id]
        self.violations += int(violation)
        if violation:
            self.violation_rejections += 1
        if reason in {"no_valid_action", "no_feasible_action"}:
            self.infeasible_rejections += 1
        else:
            self.policy_rejections += int(reason == "policy_reject")
        self.rejected_requests += 1
        self.rejection_reason_counts[reason] = self.rejection_reason_counts.get(reason, 0) + 1
        self.rejected_request_ids.add(request_id)
        self.rejected_request_events.add((self.t, request_id))
        self.total_rejected_delay_penalty += float(request.delay_budget)
        self._rollback_request(request_id)
        self._skip_current_request(request_id)
        self._action_check_cache.clear()
        return (
            self._observation(),
            -self.invalid_penalty,
            self.done,
            False,
            self._info(
                reason,
                transition_terminal=True,
                request_boundary=True,
            ),
        )

    def _vehicle_position(self, request_id: int) -> Tuple[float, float]:
        position = self.scenario.vehicle_positions[self.t, request_id]
        return float(position[0]), float(position[1])

    def _vehicle_rsu_distance(self, request_id: int, node_id: int) -> float:
        if self.node_kind[node_id] != RSU:
            return float("inf")
        if not self.has_vehicle_coordinates:
            request = self.scenario.requests[request_id]
            if (
                request.distance_to_rsu_m is not None
                and (
                    request.source_rsu_id is None
                    or int(request.source_rsu_id) == int(node_id)
                )
            ):
                return float(request.distance_to_rsu_m)
            return 0.0
        vehicle_x, vehicle_y = self._vehicle_position(request_id)
        dx = vehicle_x - float(self.node_x[node_id])
        dy = vehicle_y - float(self.node_y[node_id])
        return float(np.sqrt(dx * dx + dy * dy))

    def _vehicle_rsu_distance_ratio(self, request_id: int, node_id: int) -> float:
        distance = self._vehicle_rsu_distance(request_id, node_id)
        if not np.isfinite(distance) or distance <= 0.0:
            return 0.0
        coverage = self.scenario.nodes[node_id].coverage_radius
        if coverage is None or coverage <= 0.0:
            return 1.0
        return float(np.clip(distance / float(coverage), 0.0, 1.0))

    def _vehicle_access_delay(self, request_id: int, rsu_node_id: int) -> float:
        distance_ratio = self._vehicle_rsu_distance_ratio(request_id, rsu_node_id)
        if distance_ratio <= 0.0:
            return 0.0
        congestion_multiplier = 1.0
        if self.slot_feature_size > 0:
            congestion_multiplier = max(
                float(self.scenario.slot_features[self.t][0]),
                0.1,
            )
        base_delay = self.wireless_access_base_delay_ms * congestion_multiplier
        return float(base_delay * distance_ratio)

    def _compute_slot_rsu_assignment(self) -> np.ndarray:
        assignment = self.scenario.association[self.t].astype(np.int64).copy()
        if not self.has_vehicle_coordinates:
            return assignment

        for request_id in range(self.request_count):
            if bool(self.scenario.active_requests[self.t, request_id]):
                assignment[request_id] = self._nearest_rsu_for_request(request_id)
        return assignment

    def _nearest_rsu_for_request(self, request_id: int) -> int:
        nearest = min(
            self.rsu_node_ids,
            key=lambda node_id: (self._vehicle_rsu_distance(request_id, node_id), node_id),
        )
        distance = self._vehicle_rsu_distance(request_id, nearest)
        coverage = self.scenario.nodes[nearest].coverage_radius
        if coverage is not None and coverage > 0.0 and distance > float(coverage) + 1e-9:
            return -1
        return nearest

    def _path(
        self,
        src: int,
        dst: int,
        link_state: np.ndarray,
        bandwidth: float,
    ) -> List[int]:
        """Find the lowest-delay path that has enough residual bandwidth."""

        src = int(src)
        dst = int(dst)
        if src == dst:
            return [src]

        cache_key = (self.t, src, round(float(bandwidth), 6))
        use_cache = link_state is self.remaining_links
        parent = self._path_cache.get(cache_key) if use_cache else None
        if parent is None:
            parent = self._shortest_path_tree(src, link_state, bandwidth)
            if use_cache:
                self._path_cache[cache_key] = parent
        if dst not in parent:
            return []

        path = [dst]
        current = dst
        while current != src:
            current = parent[current]
            path.append(current)
        return list(reversed(path))

    def _shortest_path_tree(
        self,
        src: int,
        link_state: np.ndarray,
        bandwidth: float,
    ) -> Dict[int, int]:
        best = {src: 0.0}
        parent = {src: src}
        queue = [(0.0, 0, src)]
        slot_network_delay = self._slot_network_delay()
        while queue:
            distance, hops, current = heapq.heappop(queue)
            if distance > best.get(current, float("inf")) + 1e-12:
                continue

            for neighbor in self.adjacency[current]:
                capacity = float(self.scenario.links[current, neighbor])
                residual = float(link_state[current, neighbor])
                if capacity <= 0.0 or residual + 1e-9 < bandwidth:
                    continue
                utilization = 1.0 - np.clip(residual / capacity, 0.0, 1.0)
                edge_cost = (
                    float(self.scenario.link_latency[current, neighbor])
                    + float(utilization * slot_network_delay)
                )
                candidate = distance + max(edge_cost, 1e-9)
                if candidate + 1e-12 >= best.get(neighbor, float("inf")):
                    continue
                best[neighbor] = candidate
                parent[neighbor] = current
                heapq.heappush(
                    queue,
                    (candidate, hops + 1, neighbor),
                )
        return parent

    # _try_path(): thử reserve bandwidth trên path mà chưa sửa trạng thái thật
    def _try_path(
        self, link_state: np.ndarray, path: Sequence[int], bandwidth: float
    ) -> Tuple[bool, float]:
        delay = 0.0
        slot_network_delay = self._slot_network_delay()
        for src, dst in zip(path, path[1:]):
            capacity = float(self.scenario.links[src, dst])
            if capacity <= 0 or link_state[src, dst] + 1e-9 < bandwidth:
                return False, delay
            used_before = max(capacity - float(link_state[src, dst]), 0.0)
            link_state[src, dst] -= bandwidth
            delay += float(self.scenario.link_latency[src, dst])
            delay += float(
                ((used_before + bandwidth) / max(capacity, 1e-9))
                * slot_network_delay
            )
        return True, delay

    # _reserve_path(): trừ bandwidth thật sau khi action hợp lệ
    def _reserve_path(self, path: Sequence[int], bandwidth: float) -> None:
        for src, dst in zip(path, path[1:]):
            self.remaining_links[src, dst] -= bandwidth
        self._path_cache.clear()

    def _reserve_path_in_state(
        self,
        link_state: np.ndarray,
        path: Sequence[int],
        bandwidth: float,
    ) -> None:
        for src, dst in zip(path, path[1:]):
            link_state[src, dst] -= bandwidth

    def _release_path(self, path: Sequence[int], bandwidth: float) -> None:
        for src, dst in zip(path, path[1:]):
            self.remaining_links[src, dst] += bandwidth
        self._path_cache.clear()

    def _commit_request(self, request_id: int) -> None:
        allocations = list(self.current_allocations[request_id])
        if not allocations:
            return
        self.active_allocations[request_id] = allocations
        self.active_release_slots[request_id] = self._request_release_slot(request_id)
        self.active_request_placements[request_id] = self.current_placement[request_id]
        self.accepted_request_ids.add(request_id)
        self.accepted_request_events.add((self.t, request_id))
        self.current_allocations[request_id] = []

    def _request_release_slot(self, request_id: int) -> int:
        release_slot = self.t + 1
        while (
            release_slot < self.scenario.horizon
            and bool(self.scenario.active_requests[release_slot, request_id])
        ):
            release_slot += 1
        return release_slot

    def _release_completed_resources(self) -> None:
        for request_id, release_slot in list(self.active_release_slots.items()):
            if self.t < release_slot:
                continue
            self._release_active_request_resources(request_id)

    def _release_resources_for_reallocation(self) -> None:
        for request_id in np.flatnonzero(self.scenario.active_requests[self.t]):
            self._release_active_request_resources(int(request_id))

    def _release_active_request_resources(self, request_id: int) -> None:
        for allocation in self.active_allocations.pop(request_id, []):
            self.remaining_cpu[allocation.node_id] += allocation.cpu_need
            self._release_path(allocation.path_to_vnf, allocation.bandwidth)
            self._release_path(allocation.path_to_destination, allocation.bandwidth)
        self.active_release_slots.pop(request_id, None)
        self.active_request_placements[request_id, :] = -1

    def _request_event_count(self, request_ids: set[int]) -> int:
        if not request_ids:
            return 0
        ids = list(request_ids)
        return int(np.sum(self.scenario.active_requests[:, ids]))

    def _slot_network_delay(self) -> float:
        if self.slot_feature_size == 0:
            return 1.0
        slot_features = self.scenario.slot_features[self.t]
        congestion_multiplier = float(slot_features[0]) if len(slot_features) > 0 else 1.0
        avg_comm_delay_ms = float(slot_features[1]) if len(slot_features) > 1 else 0.0
        busy_ratio_pct = float(slot_features[2]) if len(slot_features) > 2 else 0.0
        packet_loss_pct = float(slot_features[5]) if len(slot_features) > 5 else 0.0
        return (
            max(congestion_multiplier, 0.1)
            + avg_comm_delay_ms / 100.0
            + busy_ratio_pct / 100.0
            + packet_loss_pct / 100.0
        )
