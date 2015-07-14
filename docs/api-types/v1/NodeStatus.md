<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
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


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/NodeStatus.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
