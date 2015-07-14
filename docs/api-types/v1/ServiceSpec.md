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
###ServiceSpec###

---
* clusterIP: 
  * **_type_**: string
  * **_description_**: IP address of the service; usually assigned by the system; if specified, it will be allocated to the service if unused or else creation of the service will fail; cannot be updated; 'None' can be specified for a headless service when proxying is not required; see http://releases.k8s.io/HEAD/docs/services.md#virtual-ips-and-service-proxies
* deprecatedPublicIPs: 
  * **_type_**: []string
  * **_description_**: deprecated. externally visible IPs (e.g. load balancers) that should be proxied to this service
* ports: 
  * **_type_**: [][ServicePort](ServicePort.md)
  * **_description_**: ports exposed by the service; see http://releases.k8s.io/HEAD/docs/services.md#virtual-ips-and-service-proxies
* selector: 
  * **_type_**: any
  * **_description_**: label keys and values that must match in order to receive traffic for this service; if empty, all pods are selected, if not specified, endpoints must be manually specified; see http://releases.k8s.io/HEAD/docs/services.md#overview
* sessionAffinity: 
  * **_type_**: string
  * **_description_**: enable client IP based session affinity; must be ClientIP or None; defaults to None; see http://releases.k8s.io/HEAD/docs/services.md#virtual-ips-and-service-proxies
* type: 
  * **_type_**: string
  * **_description_**: type of this service; must be ClusterIP, NodePort, or LoadBalancer; defaults to ClusterIP; see http://releases.k8s.io/HEAD/docs/services.md#external-services


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/ServiceSpec.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
