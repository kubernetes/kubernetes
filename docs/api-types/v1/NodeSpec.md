###NodeSpec###

---
* externalID: 
  * **_type_**: string
  * **_description_**: deprecated. External ID assigned to the node by some machine database (e.g. a cloud provider). Defaults to node name when empty.
* podCIDR: 
  * **_type_**: string
  * **_description_**: pod IP range assigned to the node
* providerID: 
  * **_type_**: string
  * **_description_**: ID of the node assigned by the cloud provider in the format: <ProviderName>://<ProviderSpecificNodeID>
* unschedulable: 
  * **_type_**: boolean
  * **_description_**: disable pod scheduling on the node; see http://releases.k8s.io/HEAD/docs/node.md#manual-node-administration
