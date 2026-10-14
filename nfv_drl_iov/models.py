from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


RSU = "rsu"
CLOUD = "cloud"
TRANSIT = "transit"
NODE_KIND_TO_FLOAT = {RSU: 0.0, CLOUD: 1.0, TRANSIT: 2.0}
DEFAULT_DELAY_CALIBRATION = {
    "version": "v2",
    "wireless_access_base_delay_ms": 5.0,
    "rsu_cloud_latency_ms": 5.0,
    "rsu_routing_delay_ms": 0.3,
    "cloud_routing_delay_ms": 0.75,
    "processing_delay_ms_by_function": {
        "firewall": 1.0,
        "compressor": 1.5,
        "route_optimizer": 2.0,
        "decoder": 2.5,
        "detector": 3.0,
    },
    "service_delay_budget_ms": {
        "safety": 50.0,
        "navigation": 100.0,
        "video": 150.0,
    },
}


@dataclass(frozen=True)
class NodeSpec:
    node_id: int
    kind: str
    cpu_capacity: float
    cpu_cost: float
    routing_delay: float
    name: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    coverage_radius: Optional[float] = None


@dataclass(frozen=True)
class FunctionSpec:
    function_id: int
    kind: str
    cpu_per_traffic: float
    processing_delay: Dict[int, float]
    name: Optional[str] = None


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    sfc: List[int]
    delay_budget: float
    name: Optional[str] = None
    vehicle_id: Optional[str] = None
    request_type: Optional[str] = None
    source_time_slot: Optional[int] = None
    source_rsu_id: Optional[int] = None
    vehicle_x: Optional[float] = None
    vehicle_y: Optional[float] = None
    speed_mps: Optional[float] = None
    distance_to_rsu_m: Optional[float] = None
    duration_slots: Optional[int] = None


@dataclass
class IoVScenario:
    nodes: List[NodeSpec]
    links: np.ndarray
    functions: List[FunctionSpec]
    requests: List[RequestSpec]
    bandwidth: np.ndarray
    association: np.ndarray
    access_capacity: Optional[np.ndarray] = None
    link_latency: Optional[np.ndarray] = None
    active_requests: Optional[np.ndarray] = None
    request_features: Optional[np.ndarray] = None
    slot_features: Optional[np.ndarray] = None
    vehicle_positions: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.links = np.asarray(self.links, dtype=np.float32)
        self.bandwidth = np.asarray(self.bandwidth, dtype=np.float32)
        self.association = np.asarray(self.association, dtype=np.int64)
        if self.access_capacity is None:
            self.access_capacity = np.maximum(self.bandwidth, 1.0) * 10.0
        else:
            self.access_capacity = np.asarray(self.access_capacity, dtype=np.float32)
        if self.link_latency is None:
            self.link_latency = np.zeros_like(self.links, dtype=np.float32)
        else:
            self.link_latency = np.asarray(self.link_latency, dtype=np.float32)
        if self.active_requests is None:
            self.active_requests = self.bandwidth > 0
        else:
            self.active_requests = np.asarray(self.active_requests, dtype=bool)
        if self.request_features is None:
            self.request_features = np.zeros(
                (*self.bandwidth.shape, 0), dtype=np.float32
            )
        else:
            self.request_features = np.asarray(self.request_features, dtype=np.float32)
        if self.slot_features is None:
            self.slot_features = np.zeros((self.bandwidth.shape[0], 0), dtype=np.float32)
        else:
            self.slot_features = np.asarray(self.slot_features, dtype=np.float32)
        if self.vehicle_positions is None:
            self.vehicle_positions = self._build_default_vehicle_positions()
        else:
            self.vehicle_positions = np.asarray(self.vehicle_positions, dtype=np.float32)
        if self.metadata is None:
            self.metadata = {}
        self.validate()

    @property
    def horizon(self) -> int:
        return int(self.bandwidth.shape[0])

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def max_chain_len(self) -> int:
        return max(len(req.sfc) for req in self.requests)

    def validate(self) -> None:
        if self.links.shape != (len(self.nodes), len(self.nodes)):
            raise ValueError("links must be a square node_count x node_count matrix")
        if self.link_latency.shape != self.links.shape:
            raise ValueError("link_latency must have the same shape as links")
        if self.bandwidth.ndim != 2:
            raise ValueError("bandwidth must have shape [time][request]")
        if self.association.shape != self.bandwidth.shape:
            raise ValueError("association must have the same shape as bandwidth")
        if self.access_capacity.shape != self.bandwidth.shape:
            raise ValueError("access_capacity must have the same shape as bandwidth")
        if self.active_requests.shape != self.bandwidth.shape:
            raise ValueError("active_requests must have the same shape as bandwidth")
        if self.request_features.shape[:2] != self.bandwidth.shape:
            raise ValueError("request_features must start with shape [time][request]")
        if self.slot_features.shape[0] != self.bandwidth.shape[0]:
            raise ValueError("slot_features must have one row per time slot")
        if self.vehicle_positions.shape != (*self.bandwidth.shape, 2):
            raise ValueError("vehicle_positions must have shape [time][request][2]")
        if self.bandwidth.shape[1] != len(self.requests):
            raise ValueError("bandwidth request count does not match requests")

        node_ids = {node.node_id for node in self.nodes}
        if node_ids != set(range(len(self.nodes))):
            raise ValueError("node ids must be contiguous and zero-based")

        function_ids = {func.function_id for func in self.functions}
        for request in self.requests:
            if not request.sfc:
                raise ValueError(f"request {request.request_id} has an empty SFC")
            for function_id in request.sfc:
                if function_id not in function_ids:
                    raise ValueError(f"unknown function id {function_id}")

        rsu_nodes = {node.node_id for node in self.nodes if node.kind == RSU}
        for node_id in self.association.reshape(-1):
            if int(node_id) not in rsu_nodes:
                raise ValueError("each association value must be an RSU node id")

    def function_by_id(self, function_id: int) -> FunctionSpec:
        for function in self.functions:
            if function.function_id == function_id:
                return function
        raise KeyError(function_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "links": self.links.tolist(),
            "functions": [
                {
                    "function_id": func.function_id,
                    "kind": func.kind,
                    "cpu_per_traffic": func.cpu_per_traffic,
                    "processing_delay": {
                        str(node_id): delay
                        for node_id, delay in func.processing_delay.items()
                    },
                    "name": func.name,
                }
                for func in self.functions
            ],
            "requests": [asdict(request) for request in self.requests],
            "bandwidth": self.bandwidth.tolist(),
            "association": self.association.tolist(),
            "access_capacity": self.access_capacity.tolist(),
            "link_latency": self.link_latency.tolist(),
            "active_requests": self.active_requests.astype(int).tolist(),
            "request_features": self.request_features.tolist(),
            "slot_features": self.slot_features.tolist(),
            "vehicle_positions": self.vehicle_positions.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IoVScenario":
        if "nodes" not in data and "rsus" in data and "service_requests" in data:
            return scenario_from_processed_dataset(data)
        return cls(
            nodes=[NodeSpec(**node) for node in data["nodes"]],
            links=np.asarray(data["links"], dtype=np.float32),
            functions=[
                FunctionSpec(
                    function_id=int(function["function_id"]),
                    kind=function["kind"],
                    cpu_per_traffic=float(function["cpu_per_traffic"]),
                    processing_delay={
                        int(node_id): float(delay)
                        for node_id, delay in function["processing_delay"].items()
                    },
                    name=function.get("name"),
                )
                for function in data["functions"]
            ],
            requests=[RequestSpec(**request) for request in data["requests"]],
            bandwidth=np.asarray(data["bandwidth"], dtype=np.float32),
            association=np.asarray(data["association"], dtype=np.int64),
            access_capacity=np.asarray(data.get("access_capacity", []), dtype=np.float32)
            if "access_capacity" in data
            else None,
            link_latency=np.asarray(data.get("link_latency", []), dtype=np.float32)
            if "link_latency" in data
            else None,
            active_requests=np.asarray(data.get("active_requests", []), dtype=bool)
            if "active_requests" in data
            else None,
            request_features=np.asarray(data.get("request_features", []), dtype=np.float32)
            if "request_features" in data
            else None,
            slot_features=np.asarray(data.get("slot_features", []), dtype=np.float32)
            if "slot_features" in data
            else None,
            vehicle_positions=np.asarray(data.get("vehicle_positions", []), dtype=np.float32)
            if "vehicle_positions" in data
            else None,
            metadata=data.get("metadata", {}),
        )

    def _build_default_vehicle_positions(self) -> np.ndarray:
        positions = np.zeros((*self.bandwidth.shape, 2), dtype=np.float32)
        for request_id, request in enumerate(self.requests):
            if request.vehicle_x is not None and request.vehicle_y is not None:
                positions[:, request_id, 0] = float(request.vehicle_x)
                positions[:, request_id, 1] = float(request.vehicle_y)
                continue

            for slot in range(self.bandwidth.shape[0]):
                node = self.nodes[int(self.association[slot, request_id])]
                positions[slot, request_id, 0] = float(node.x or 0.0)
                positions[slot, request_id, 1] = float(node.y or 0.0)
        return positions


def save_scenario(scenario: IoVScenario, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(scenario.to_dict(), fout, indent=2)


def load_scenario(path: str | Path) -> IoVScenario:
    with Path(path).open("r", encoding="utf-8") as fin:
        return IoVScenario.from_dict(json.load(fin))


def scenario_from_processed_dataset(data: Dict[str, Any]) -> IoVScenario:
    rsus = data["rsus"]
    clouds = data["cloud_nodes"]
    transit_nodes = data.get("cloud_transit_nodes", [])
    service_requests = data["service_requests"]
    delay_calibration = _delay_calibration(data)
    slot_count = int(data.get("metadata", {}).get("slot_count", 0))
    if slot_count <= 0:
        slot_count = 1 + max(int(row["time_slot"]) for row in service_requests)

    rsu_to_node = {row["rsu_id"]: idx for idx, row in enumerate(rsus)}
    cloud_to_node = {
        row["cloud_id"]: len(rsus) + idx for idx, row in enumerate(clouds)
    }
    transit_to_node = {
        row["transit_id"]: len(rsus) + len(clouds) + idx
        for idx, row in enumerate(transit_nodes)
    }
    topology_to_node = {**cloud_to_node, **transit_to_node}

    nodes: List[NodeSpec] = []
    for idx, row in enumerate(rsus):
        degree = float(row.get("degree", 1.0))
        nodes.append(
            NodeSpec(
                node_id=idx,
                kind=RSU,
                cpu_capacity=90.0 + 5.0 * degree,
                cpu_cost=1.0,
                routing_delay=float(delay_calibration["rsu_routing_delay_ms"]),
                name=row["rsu_id"],
                x=float(row["x"]),
                y=float(row["y"]),
                coverage_radius=float(row.get("coverage_radius_m", 500.0)),
            )
        )
    for idx, row in enumerate(clouds):
        nodes.append(
            NodeSpec(
                node_id=len(rsus) + idx,
                kind=CLOUD,
                cpu_capacity=320.0 + 20.0 * float(row.get("degree", 1.0)),
                cpu_cost=2.2,
                routing_delay=float(delay_calibration["cloud_routing_delay_ms"]),
                name=row["cloud_id"],
            )
        )
    for idx, row in enumerate(transit_nodes):
        nodes.append(
            NodeSpec(
                node_id=len(rsus) + len(clouds) + idx,
                kind=TRANSIT,
                cpu_capacity=0.0,
                cpu_cost=0.0,
                routing_delay=float(row.get("routing_delay_ms", 0.05)),
                name=row["transit_id"],
            )
        )

    node_count = len(nodes)
    avg_throughput = _mean_profile(data, "throughput_mbps", default=50.0)
    avg_fog_latency = _mean_profile(data, "latency_ms", default=40.0)
    links = np.zeros((node_count, node_count), dtype=np.float32)
    link_latency = np.zeros((node_count, node_count), dtype=np.float32)

    _add_rsu_mesh_links(rsus, rsu_to_node, links, link_latency, avg_throughput)

    for row in data.get("rsu_cloud_links", []):
        src = rsu_to_node[row["source"]]
        dst = topology_to_node[row["target"]]
        capacity = float(row.get("capacity_mbps", avg_throughput))
        latency = float(delay_calibration["rsu_cloud_latency_ms"])
        _set_bidirectional_link(links, link_latency, src, dst, capacity, latency)

    for row in data.get("cloud_links", []):
        src = topology_to_node[row["source"]]
        dst = topology_to_node[row["target"]]
        _set_bidirectional_link(
            links,
            link_latency,
            src,
            dst,
            float(row.get("capacity_mbps", max(avg_throughput * 2.0, 100.0))),
            float(
                row.get(
                    "latency_ms",
                    row.get("delay_ms", max(avg_fog_latency * 0.25, 3.0)),
                )
            ),
        )

    function_specs = _function_specs_from_requests(service_requests)
    functions = _build_processed_functions(function_specs, nodes, delay_calibration)
    function_id_by_name = {func.name: func.function_id for func in functions}

    congestion_by_slot = _rows_by_slot(data.get("congestion_profile", []))
    fog_by_slot = _rows_by_slot(data.get("fog_network_profile", []))
    summary_by_slot_rsu = {
        (int(row["time_slot"]), row["rsu_id"]): row
        for row in data.get("vehicle_rsu_summary", [])
    }
    session_max_gap_slots = int(
        data.get("metadata", {}).get("session_max_gap_slots", 1)
    )
    grouped_service_requests = _group_service_request_sessions(
        service_requests,
        max_gap_slots=session_max_gap_slots,
    )
    requests: List[RequestSpec] = []
    bandwidth = np.zeros((slot_count, len(grouped_service_requests)), dtype=np.float32)
    association = np.zeros((slot_count, len(grouped_service_requests)), dtype=np.int64)
    access_capacity = np.zeros(
        (slot_count, len(grouped_service_requests)),
        dtype=np.float32,
    )
    active_requests = np.zeros((slot_count, len(grouped_service_requests)), dtype=bool)
    request_features = np.zeros(
        (slot_count, len(grouped_service_requests), 5),
        dtype=np.float32,
    )
    vehicle_positions = np.zeros(
        (slot_count, len(grouped_service_requests), 2),
        dtype=np.float32,
    )

    for req_idx, rows in enumerate(grouped_service_requests):
        rows = sorted(rows, key=lambda item: int(item["time_slot"]))
        first_row = rows[0]
        first_slot = int(first_row["time_slot"])
        first_rsu_node = rsu_to_node[first_row["rsu_id"]]
        chain = first_row["service_chain"]
        sfc = [function_id_by_name[name] for name in chain]
        budget_by_type = delay_calibration["service_delay_budget_ms"]
        delay_budget = float(
            budget_by_type.get(
                first_row["request_type"],
                min(float(row["delay_budget_ms"]) for row in rows),
            )
        )
        requests.append(
            RequestSpec(
                request_id=req_idx,
                sfc=sfc,
                delay_budget=delay_budget,
                name=first_row["request_id"],
                vehicle_id=first_row["vehicle_id"],
                request_type=first_row["request_type"],
                source_time_slot=first_slot,
                source_rsu_id=first_rsu_node,
                vehicle_x=float(first_row["vehicle_x"]),
                vehicle_y=float(first_row["vehicle_y"]),
                speed_mps=float(first_row["speed_mps"]),
                distance_to_rsu_m=float(first_row["distance_to_rsu_m"]),
                duration_slots=first_row.get("duration_slots"),
            )
        )
        association[:, req_idx] = first_rsu_node

        rows_by_slot = {
            int(row["time_slot"]): row
            for row in rows
            if 0 <= int(row["time_slot"]) < slot_count
        }
        if not rows_by_slot:
            continue

        # R(t) contains requests explicitly generated at slot t.  Do not
        # manufacture extra active slots from service-class durations: the
        # paper defines neither such durations nor repeated placement events.
        for slot, row in rows_by_slot.items():
            # A generated request already carries the vehicle position and RSU
            # association at its decision instant.  A later sample from the
            # same coarse one-minute slot must not overwrite that source state.
            mobility_row = row
            rsu_node = rsu_to_node[mobility_row["rsu_id"]]
            congestion = congestion_by_slot.get(slot, {})
            multiplier = float(congestion.get("congestion_multiplier", 1.0))
            bandwidth[slot, req_idx] = float(row["bandwidth_mbps"]) * multiplier
            access_capacity[slot, req_idx] = max(
                float(fog_by_slot.get(slot, {}).get("throughput_mbps", avg_throughput)),
                0.0,
            )
            association[slot, req_idx] = rsu_node
            active_requests[slot, req_idx] = True
            vehicle_positions[slot, req_idx] = np.asarray(
                [
                    float(mobility_row["vehicle_x"]),
                    float(mobility_row["vehicle_y"]),
                ],
                dtype=np.float32,
            )
            coverage = max(
                float(rsus[rsu_node].get("coverage_radius_m", 500.0)),
                1.0,
            )
            summary = summary_by_slot_rsu.get((slot, mobility_row["rsu_id"]), {})
            request_features[slot, req_idx] = np.asarray(
                [
                    float(mobility_row["speed_mps"]),
                    float(mobility_row["distance_to_rsu_m"]) / coverage,
                    float(summary.get("covered_vehicle_count", 0.0)),
                    float(congestion.get("packet_loss_pct", 0.0)),
                    float(fog_by_slot.get(slot, {}).get("link_utilization_pct", 0.0)),
                ],
                dtype=np.float32,
            )

    slot_features = _build_slot_features(data, slot_count)
    metadata = dict(data.get("metadata", {}))
    metadata["delay_calibration"] = delay_calibration
    metadata.update(
        {
            "source_schema": "processed_iov_dataset",
            "resource_lifecycle": "per_slot_request_events",
            "request_feature_names": [
                "speed_mps",
                "distance_to_rsu_ratio",
                "covered_vehicle_count",
                "packet_loss_pct",
                "link_utilization_pct",
            ],
            "slot_feature_names": [
                "congestion_multiplier",
                "avg_comm_delay_ms",
                "channel_busy_ratio_pct",
                "fog_latency_ms",
                "fog_throughput_mbps",
                "fog_packet_loss_pct",
                "fog_link_utilization_pct",
            ],
            "function_name_by_id": {
                str(func.function_id): func.name for func in functions
            },
            "raw_service_request_count": len(service_requests),
            "request_session_count": len(requests),
            "request_grouping": "request_id",
            "request_session_max_gap_slots": session_max_gap_slots,
            "cloud_compute_node_count": len(clouds),
            "cloud_transit_node_count": len(transit_nodes),
        }
    )

    return IoVScenario(
        nodes=nodes,
        links=links,
        functions=functions,
        requests=requests,
        bandwidth=bandwidth,
        association=association,
        access_capacity=access_capacity,
        link_latency=link_latency,
        active_requests=active_requests,
        request_features=request_features,
        slot_features=slot_features,
        vehicle_positions=vehicle_positions,
        metadata=metadata,
    )


def _group_service_request_sessions(
    service_requests: List[Dict[str, Any]],
    max_gap_slots: int = 1,
) -> List[List[Dict[str, Any]]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in service_requests:
        key = (row["request_id"],)
        grouped[key].append(row)

    sessions: List[List[Dict[str, Any]]] = []
    max_gap_slots = max(int(max_gap_slots), 0)
    for rows in grouped.values():
        rows = sorted(rows, key=lambda item: int(item["time_slot"]))
        current_session: List[Dict[str, Any]] = []
        previous_slot: Optional[int] = None
        for row in rows:
            slot = int(row["time_slot"])
            if (
                current_session
                and previous_slot is not None
                and slot - previous_slot > max_gap_slots
            ):
                sessions.append(current_session)
                current_session = []
            current_session.append(row)
            previous_slot = slot
        if current_session:
            sessions.append(current_session)

    return sorted(
        sessions,
        key=lambda rows: (
            min(int(row["time_slot"]) for row in rows),
            rows[0].get("vehicle_id", ""),
            rows[0].get("request_type", ""),
        ),
    )


def _mean_profile(data: Dict[str, Any], key: str, default: float) -> float:
    rows = data.get("fog_network_profile", [])
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else float(default)


def _rows_by_slot(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(row["time_slot"]): row for row in rows if "time_slot" in row}


def _set_bidirectional_link(
    links: np.ndarray,
    link_latency: np.ndarray,
    src: int,
    dst: int,
    capacity: float,
    latency: float,
) -> None:
    links[src, dst] = links[dst, src] = float(capacity)
    link_latency[src, dst] = link_latency[dst, src] = float(latency)


def _add_rsu_mesh_links(
    rsus: List[Dict[str, Any]],
    rsu_to_node: Dict[str, int],
    links: np.ndarray,
    link_latency: np.ndarray,
    avg_throughput: float,
) -> None:
    for src in rsus:
        distances = []
        for dst in rsus:
            if src["rsu_id"] == dst["rsu_id"]:
                continue
            distance = _euclidean_m(src, dst)
            distances.append((distance, dst))
        for distance, dst in sorted(distances, key=lambda item: item[0])[:2]:
            latency = 1.0 + distance / 500.0
            capacity = max(avg_throughput * 0.8, 20.0)
            _set_bidirectional_link(
                links,
                link_latency,
                rsu_to_node[src["rsu_id"]],
                rsu_to_node[dst["rsu_id"]],
                capacity,
                latency,
            )


def _euclidean_m(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    return math.sqrt(
        (float(left["x"]) - float(right["x"])) ** 2
        + (float(left["y"]) - float(right["y"])) ** 2
    )


def _function_specs_from_requests(requests: List[Dict[str, Any]]) -> Dict[str, str]:
    specs: Dict[str, str] = {}
    for row in requests:
        for name in row["service_chain"]:
            if name in {"decoder", "detector"}:
                specs[name] = CLOUD
            else:
                specs[name] = RSU
    return specs


def _build_processed_functions(
    specs: Dict[str, str],
    nodes: List[NodeSpec],
    delay_calibration: Dict[str, Any],
) -> List[FunctionSpec]:
    cpu_by_name = {
        "firewall": 0.8,
        "compressor": 1.1,
        "route_optimizer": 1.0,
        "decoder": 2.0,
        "detector": 2.4,
    }
    functions: List[FunctionSpec] = []
    processing_delay_by_name = delay_calibration["processing_delay_ms_by_function"]
    for function_id, name in enumerate(sorted(specs)):
        kind = specs[name]
        processing_delay = {}
        for node in nodes:
            if node.kind != kind:
                continue
            processing_delay[node.node_id] = float(processing_delay_by_name[name])
        functions.append(
            FunctionSpec(
                function_id=function_id,
                kind=kind,
                cpu_per_traffic=cpu_by_name.get(name, 1.0),
                processing_delay=processing_delay,
                name=name,
            )
        )
    return functions


def _delay_calibration(data: Dict[str, Any]) -> Dict[str, Any]:
    configured = data.get("metadata", {}).get("delay_calibration", {})
    calibration = {
        **DEFAULT_DELAY_CALIBRATION,
        **configured,
        "processing_delay_ms_by_function": {
            **DEFAULT_DELAY_CALIBRATION["processing_delay_ms_by_function"],
            **configured.get("processing_delay_ms_by_function", {}),
        },
        "service_delay_budget_ms": {
            **DEFAULT_DELAY_CALIBRATION["service_delay_budget_ms"],
            **configured.get("service_delay_budget_ms", {}),
        },
    }
    return calibration


def _build_slot_features(data: Dict[str, Any], slot_count: int) -> np.ndarray:
    congestion_by_slot = _rows_by_slot(data.get("congestion_profile", []))
    fog_by_slot = _rows_by_slot(data.get("fog_network_profile", []))
    features = np.zeros((slot_count, 7), dtype=np.float32)
    for slot in range(slot_count):
        congestion = congestion_by_slot.get(slot, {})
        fog = fog_by_slot.get(slot, {})
        features[slot] = np.asarray(
            [
                float(congestion.get("congestion_multiplier", 1.0)),
                float(congestion.get("avg_comm_delay_ms", 0.0)),
                float(congestion.get("channel_busy_ratio_pct", 0.0)),
                float(fog.get("latency_ms", 0.0)),
                float(fog.get("throughput_mbps", 0.0)),
                float(fog.get("packet_loss_pct", 0.0)),
                float(fog.get("link_utilization_pct", 0.0)),
            ],
            dtype=np.float32,
        )
    return features
