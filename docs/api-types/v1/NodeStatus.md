###NodeStatus###

---
* addresses: 
  * **_type_**: [][NodeAddress](NodeAddress.md)
  * **_description_**: list of addresses reachable to the node; see http://releases.k8s.io/HEAD/docs/node.md#node-addresses
* capacity: 
  * **_type_**: any
  * **_description_**: compute resource capacity of the node; see http://releases.k8s.io/HEAD/docs/compute_resources.md
* conditions: 
  * **_type_**: [][NodeCondition](NodeCondition.md)
  * **_description_**: list of node conditions observed; see http://releases.k8s.io/HEAD/docs/node.md#node-condition
* nodeInfo: 
  * **_type_**: [NodeSystemInfo](NodeSystemInfo.md)
  * **_description_**: set of ids/uuids to uniquely identify the node; see http://releases.k8s.io/HEAD/docs/node.md#node-info
* phase: 
  * **_type_**: string
  * **_description_**: most recently observed lifecycle phase of the node; see http://releases.k8s.io/HEAD/docs/node.md#node-phase
