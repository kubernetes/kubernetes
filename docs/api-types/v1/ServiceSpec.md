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
