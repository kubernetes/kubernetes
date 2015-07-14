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
###NodeSystemInfo###

---
* bootID: 
  * **_type_**: string
  * **_description_**: boot id is the boot-id reported by the node
* containerRuntimeVersion: 
  * **_type_**: string
  * **_description_**: Container runtime version reported by the node through runtime remote API (e.g. docker://1.5.0)
* kernelVersion: 
  * **_type_**: string
  * **_description_**: Kernel version reported by the node from 'uname -r' (e.g. 3.16.0-0.bpo.4-amd64)
* kubeProxyVersion: 
  * **_type_**: string
  * **_description_**: Kube-proxy version reported by the node
* kubeletVersion: 
  * **_type_**: string
  * **_description_**: Kubelet version reported by the node
* machineID: 
  * **_type_**: string
  * **_description_**: machine-id reported by the node
* osImage: 
  * **_type_**: string
  * **_description_**: OS image used reported by the node from /etc/os-release (e.g. Debian GNU/Linux 7 (wheezy))
* systemUUID: 
  * **_type_**: string
  * **_description_**: system-uuid reported by the node


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/NodeSystemInfo.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
