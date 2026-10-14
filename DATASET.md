# Evaluation dataset

Each JSON file in `data/` contains a complete simulation scenario, including the network, service requests, mobility and network-condition features.

## Network parameters

| Parameter | Fat-Tree | BCube |
|---|---:|---:|
| Topology | k=6 | n=4, k=2 |
| RSUs | 40 | 40 |
| Cloud compute nodes | 54 | 64 |
| Transit switches | 45 | 48 |
| Total nodes | 139 | 152 |
| Internal cloud links | 162 | 192 |

RSU and cloud compute capacities are 20 and 50, respectively. Transit switches do not host VNFs. Internal-cloud links have a capacity of 60 Mbps, and RSU-to-cloud links have a capacity of 40 Mbps. RSU compute costs range from 0.8 to 1.2; cloud compute costs range from 1.8 to 2.6. Delay values are in milliseconds, and a time slot represents 60 seconds.

## Workloads

The source scenario contains 160 request sessions. Nested subsets of 10, 20, ..., 100 requests were selected with seed 42. Each subset was then modified so that approximately 75% of requests use Firewall–Decoder–Detector–Compressor–Route Optimizer, represented by `[3, 1, 2, 0, 4]`.

The numbers of five-VNF chains are 8, 15, 23, 30, 38, 45, 53, 60, 68 and 75, respectively. Chain assignments are made per workload, so the transformed chain of a selected request need not remain identical across workload sizes.

## Data fields

- `nodes`, `links`, `link_latency`: network resources and link delays.
- `functions`: function types, CPU demand and processing delay.
- `requests`: service chains and delay budgets.
- `bandwidth`, `active_requests`: request demand and activity by time slot.
- `association`, `access_capacity`, `vehicle_positions`: mobility and access conditions.
- `request_features`, `slot_features`: observation features used by the policies.
- `metadata`: calibration and workload parameters.

The scenario loader is defined in `nfv_drl_iov/models.py`. A request-event is one request decision in a time slot; a request session may span several slots.

## Data sources

- **LuST mobility:** Lara Codeca, Raphael Frank, Sebastien Faye and Thomas Engel, *Luxembourg SUMO Traffic (LuST) Scenario: Traffic Demand Evaluation*, IEEE Intelligent Transportation Systems Magazine 9(2), 2017, pp. 52–63. The LuST 2.0 license is included in `third_party/LuST_LICENSE.md`.
- **SUMO:** https://eclipse.dev/sumo/
- **VANET Traffic Congestion Dataset:** https://www.kaggle.com/datasets/ucimachinelearning/vanet-traffic-congestion-dataset
- **Vehicle Network Dataset Kaggle:** the project uses the file `fog_vanet_dataset.csv`; its local source distribution does not include a publisher URL or license notice.

Fat-Tree and BCube networks are generated topologies. The package contains processed simulator inputs, not the original raw-data archives. The included LuST license applies to that source distribution and does not license the other datasets or the authors' code.
