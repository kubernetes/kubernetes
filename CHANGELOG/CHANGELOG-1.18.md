<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.18.0-alpha.1](#v1180-alpha1)
  - [Downloads for v1.18.0-alpha.1](#downloads-for-v1180-alpha1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.17.0](#changelog-since-v1170)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.18.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-alpha.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes.tar.gz) | `0c4904efc7f4f1436119c91dc1b6c93b3bd9c7490362a394bff10099c18e1e7600c4f6e2fcbaeb2d342a36c4b20692715cf7aa8ada6dfac369f44cc9292529d7`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-src.tar.gz) | `0a50fc6816c730ca5ae4c4f26d5ad7b049607d29f6a782a4e5b4b05ac50e016486e269dafcc6a163bd15e1a192780a9a987f1bb959696993641c603ed1e841c8`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `c6d75f7f3f20bef17fc7564a619b54e6f4a673d041b7c9ec93663763a1cc8dd16aecd7a2af70e8d54825a0eecb9762cf2edfdade840604c9a32ecd9cc2d5ac3c`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `ca1f19db289933beace6daee6fc30af19b0e260634ef6e89f773464a05e24551c791be58b67da7a7e2a863e28b7cbcc7b24b6b9bf467113c26da76ac8f54fdb6`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `af2e673653eb39c3f24a54efc68e1055f9258bdf6cf8fea42faf42c05abefc2da853f42faac3b166c37e2a7533020b8993b98c0d6d80a5b66f39e91d8ae0a3fb`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `9009032c3f94ac8a78c1322a28e16644ce3b20989eb762685a1819148aed6e883ca8e1200e5ec37ec0853f115c67e09b5d697d6cf5d4c45f653788a2d3a2f84f`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `afba9595b37a3f2eead6e3418573f7ce093b55467dce4da0b8de860028576b96b837a2fd942f9c276e965da694e31fbd523eeb39aefb902d7e7a2f169344d271`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `04fc3b2fe3f271807f0bc6c61be52456f26a1af904964400be819b7914519edc72cbab9afab2bb2e2ba1a108963079367cedfb253c9364c0175d1fcc64d52f5c`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `04c7edab874b33175ff7bebfff5b3a032bc6eb088fcd7387ffcd5b3fa71395ca8c5f9427b7ddb496e92087dfdb09eaf14a46e9513071d3bd73df76c182922d38`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `499287dbbc33399a37b9f3b35e0124ff20b17b6619f25a207ee9c606ef261af61fa0c328dde18c7ce2d3dfb2eea2376623bc3425d16bc8515932a68b44f8bede`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `cf84aeddf00f126fb13c0436b116dd0464a625659e44c84bf863517db0406afb4eefd86807e7543c4f96006d275772fbf66214ae7d582db5865c84ac3545b3e6`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `69f20558ccd5cd6dbaccf29307210db4e687af21f6d71f68c69d3a39766862686ac1333ab8a5012010ca5c5e3c11676b45e498e3d4c38773da7d24bcefc46d95`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `3f29df2ce904a0f10db4c1d7a425a36f420867b595da3fa158ae430bfead90def2f2139f51425b349faa8a9303dcf20ea01657cb6ea28eb6ad64f5bb32ce2ed1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `4a21073b2273d721fbf062c254840be5c8471a010bcc0c731b101729e36e61f637cb7fcb521a22e8d24808510242f4fff8a6ca40f10e9acd849c2a47bf135f27`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `7f1cb6d721bedc90e28b16f99bea7e59f5ad6267c31ef39c14d34db6ad6aad87ee51d2acdd01b6903307c1c00b58ff6b785a03d5a491cc3f8a4df9a1d76d406c`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `8f2b552030b5274b1c2c7c166eacd5a14b0c6ca0f23042f4c52efe87e22a167ba4460dcd66615a5ecd26d9e88336be1fb555548392e70efe59070dd2c314da98`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `8d9f2c96f66edafb7c8b3aa90960d29b41471743842aede6b47b3b2e61f4306fb6fc60b9ebc18820c547ee200bfedfe254c1cde962d447c791097dd30e79abdb`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `84194cb081d1502f8ca68143569f9707d96f1a28fcf0c574ebd203321463a8b605f67bb2a365eaffb14fbeb8d55c8d3fa17431780b242fb9cba3a14426a0cd4a`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `0091e108ab94fd8683b89c597c4fdc2fbf4920b007cfcd5297072c44bc3a230dfe5ceed16473e15c3e6cf5edab866d7004b53edab95be0400cc60e009eee0d9d`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `b7e85682cc2848a35d52fd6f01c247f039ee1b5dd03345713821ea10a7fa9939b944f91087baae95eaa0665d11857c1b81c454f720add077287b091f9f19e5d3`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `cd1f0849e9c62b5d2c93ff0cebf58843e178d8a88317f45f76de0db5ae020b8027e9503a5fccc96445184e0d77ecdf6f57787176ac31dbcbd01323cd0a190cbb`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `e1e697a34424c75d75415b613b81c8af5f64384226c5152d869f12fd7db1a3e25724975b73fa3d89e56e4bf78d5fd07e68a709ba8566f53691ba6a88addc79ea`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `c725a19a4013c74e22383ad3fb4cb799b3e161c4318fdad066daf806730a89bc3be3ff0f75678d02b3cbe52b2ef0c411c0639968e200b9df470be40bb2c015cc`

## Changelog since v1.17.0

### Action Required

* action required ([#85363](https://github.com/kubernetes/kubernetes/pull/85363), [@immutableT](https://github.com/immutableT))
    * 1. Currently, if users were to explicitly specify CacheSize of 0 for KMS provider, they would end-up with a provider that caches up to 1000 keys. This PR changes this behavior.
    * Post this PR, when users supply 0 for CacheSize this will result in a validation error.
    * 2. CacheSize type was changed from int32 to *int32. This allows defaulting logic to differentiate between cases where users explicitly supplied 0 vs. not supplied any value.
    * 3. KMS Provider's endpoint (path to Unix socket) is now validated when the EncryptionConfiguration files is loaded. This used to be handled by the GRPCService.

### Other notable changes

* fix: azure data disk should use same key as os disk by default ([#86351](https://github.com/kubernetes/kubernetes/pull/86351), [@andyzhangx](https://github.com/andyzhangx))
* New flag `--show-hidden-metrics-for-version` in kube-proxy can be used to show all hidden metrics that deprecated in the previous minor release. ([#85279](https://github.com/kubernetes/kubernetes/pull/85279), [@RainbowMango](https://github.com/RainbowMango))
* Remove cluster-monitoring addon ([#85512](https://github.com/kubernetes/kubernetes/pull/85512), [@serathius](https://github.com/serathius))
* Changed core_pattern on COS nodes to be an absolute path. ([#86329](https://github.com/kubernetes/kubernetes/pull/86329), [@mml](https://github.com/mml))
* Track mount operations as uncertain if operation fails with non-final error ([#82492](https://github.com/kubernetes/kubernetes/pull/82492), [@gnufied](https://github.com/gnufied))
* add kube-proxy flags --ipvs-tcp-timeout, --ipvs-tcpfin-timeout, --ipvs-udp-timeout to configure IPVS connection timeouts. ([#85517](https://github.com/kubernetes/kubernetes/pull/85517), [@andrewsykim](https://github.com/andrewsykim))
* The sample-apiserver aggregated conformance test has updated to use the Kubernetes v1.17.0 sample apiserver ([#84735](https://github.com/kubernetes/kubernetes/pull/84735), [@liggitt](https://github.com/liggitt))
* The underlying format of the `CPUManager` state file has changed. Upgrades should be seamless, but any third-party tools that rely on reading the previous format need to be updated. ([#84462](https://github.com/kubernetes/kubernetes/pull/84462), [@klueska](https://github.com/klueska))
* kubernetes will try to acquire the iptables lock every 100 msec during 5 seconds instead of every second. This specially useful for environments using kube-proxy in iptables mode with a high churn rate of services. ([#85771](https://github.com/kubernetes/kubernetes/pull/85771), [@aojea](https://github.com/aojea))
* Fixed a panic in the kubelet cleaning up pod volumes ([#86277](https://github.com/kubernetes/kubernetes/pull/86277), [@tedyu](https://github.com/tedyu))
* azure cloud provider cache TTL is configurable, list of the azure cloud provider is as following: ([#86266](https://github.com/kubernetes/kubernetes/pull/86266), [@zqingqing1](https://github.com/zqingqing1))
    * - "availabilitySetNodesCacheTTLInSeconds"
    * - "vmssCacheTTLInSeconds"
    * - "vmssVirtualMachinesCacheTTLInSeconds"
    * - "vmCacheTTLInSeconds"
    * - "loadBalancerCacheTTLInSeconds"
    * - "nsgCacheTTLInSeconds"
    * - "routeTableCacheTTLInSeconds"
* Fixes kube-proxy when EndpointSlice feature gate is enabled on Windows. ([#86016](https://github.com/kubernetes/kubernetes/pull/86016), [@robscott](https://github.com/robscott))
* Fixes wrong validation result of NetworkPolicy PolicyTypes ([#85747](https://github.com/kubernetes/kubernetes/pull/85747), [@tnqn](https://github.com/tnqn))
* Fixes an issue with kubelet-reported pod status on deleted/recreated pods. ([#86320](https://github.com/kubernetes/kubernetes/pull/86320), [@liggitt](https://github.com/liggitt))
* kube-apiserver no longer serves the following deprecated APIs: ([#85903](https://github.com/kubernetes/kubernetes/pull/85903), [@liggitt](https://github.com/liggitt))
        * All resources under `apps/v1beta1` and `apps/v1beta2` - use `apps/v1` instead
        * `daemonsets`, `deployments`, `replicasets` resources under `extensions/v1beta1` - use `apps/v1` instead
        * `networkpolicies` resources under `extensions/v1beta1` - use `networking.k8s.io/v1` instead
        * `podsecuritypolicies` resources under `extensions/v1beta1` - use `policy/v1beta1` instead
* kubeadm: fix potential panic when executing "kubeadm reset" with a corrupted kubelet.conf file ([#86216](https://github.com/kubernetes/kubernetes/pull/86216), [@neolit123](https://github.com/neolit123))
* Fix a bug in port-forward: named port not working with service ([#85511](https://github.com/kubernetes/kubernetes/pull/85511), [@oke-py](https://github.com/oke-py))
* kube-proxy no longer modifies shared EndpointSlices. ([#86092](https://github.com/kubernetes/kubernetes/pull/86092), [@robscott](https://github.com/robscott))
* allow for configuration of CoreDNS replica count ([#85837](https://github.com/kubernetes/kubernetes/pull/85837), [@pickledrick](https://github.com/pickledrick))
* Fixed a regression where the kubelet would fail to update the ready status of pods. ([#84951](https://github.com/kubernetes/kubernetes/pull/84951), [@tedyu](https://github.com/tedyu))
* Resolves performance regression in client-go discovery clients constructed using `NewDiscoveryClientForConfig` or `NewDiscoveryClientForConfigOrDie`. ([#86168](https://github.com/kubernetes/kubernetes/pull/86168), [@liggitt](https://github.com/liggitt))
* Make error message and service event message more clear ([#86078](https://github.com/kubernetes/kubernetes/pull/86078), [@feiskyer](https://github.com/feiskyer))
* e2e-test-framework: add e2e test namespace dump if all tests succeed but the cleanup fails. ([#85542](https://github.com/kubernetes/kubernetes/pull/85542), [@schrodit](https://github.com/schrodit))
* SafeSysctlWhitelist: add net.ipv4.ping_group_range ([#85463](https://github.com/kubernetes/kubernetes/pull/85463), [@AkihiroSuda](https://github.com/AkihiroSuda))
* kubelet: the metric process_start_time_seconds be marked as with the ALPHA stability level. ([#85446](https://github.com/kubernetes/kubernetes/pull/85446), [@RainbowMango](https://github.com/RainbowMango))
* API request throttling (due to a high rate of requests) is now reported in the kubelet (and other component) logs by default.  The messages are of the form ([#80649](https://github.com/kubernetes/kubernetes/pull/80649), [@RobertKrawitz](https://github.com/RobertKrawitz))
    * Throttling request took 1.50705208s, request: GET:<URL>
    * The presence of large numbers of these messages, particularly with long delay times, may indicate to the administrator the need to tune the cluster accordingly.
* Fix API Server potential memory leak issue in processing watch request. ([#85410](https://github.com/kubernetes/kubernetes/pull/85410), [@answer1991](https://github.com/answer1991))
* Verify kubelet & kube-proxy can recover after being killed on Windows nodes ([#84886](https://github.com/kubernetes/kubernetes/pull/84886), [@YangLu1031](https://github.com/YangLu1031))
* Fixed an issue that the scheduler only returns the first failure reason. ([#86022](https://github.com/kubernetes/kubernetes/pull/86022), [@Huang-Wei](https://github.com/Huang-Wei))
* kubectl/drain: add skip-wait-for-delete-timeout option. ([#85577](https://github.com/kubernetes/kubernetes/pull/85577), [@michaelgugino](https://github.com/michaelgugino))
    * If pod DeletionTimestamp older than N seconds, skip waiting for the pod.  Seconds must be greater than 0 to skip.
* Following metrics have been turned off: ([#83841](https://github.com/kubernetes/kubernetes/pull/83841), [@RainbowMango](https://github.com/RainbowMango))
    * - kubelet_pod_worker_latency_microseconds
    * - kubelet_pod_start_latency_microseconds
    * - kubelet_cgroup_manager_latency_microseconds
    * - kubelet_pod_worker_start_latency_microseconds
    * - kubelet_pleg_relist_latency_microseconds
    * - kubelet_pleg_relist_interval_microseconds
    * - kubelet_eviction_stats_age_microseconds
    * - kubelet_runtime_operations
    * - kubelet_runtime_operations_latency_microseconds
    * - kubelet_runtime_operations_errors
    * - kubelet_device_plugin_registration_count
    * - kubelet_device_plugin_alloc_latency_microseconds
    * - kubelet_docker_operations
    * - kubelet_docker_operations_latency_microseconds
    * - kubelet_docker_operations_errors
    * - kubelet_docker_operations_timeout
    * - network_plugin_operations_latency_microseconds
* - Renamed Kubelet metric certificate_manager_server_expiration_seconds to certificate_manager_server_ttl_seconds and changed to report the second until expiration at read time rather than absolute time of expiry. ([#85874](https://github.com/kubernetes/kubernetes/pull/85874), [@sambdavidson](https://github.com/sambdavidson))
    * - Improved accuracy of Kubelet metric rest_client_exec_plugin_ttl_seconds.
* Bind metadata-agent containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83363](https://github.com/kubernetes/kubernetes/pull/83363), [@wawa0210](https://github.com/wawa0210))
* Bind metrics-server containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83362](https://github.com/kubernetes/kubernetes/pull/83362), [@wawa0210](https://github.com/wawa0210))
* During initialization phase (preflight), kubeadm now verifies the presence of the conntrack executable ([#85857](https://github.com/kubernetes/kubernetes/pull/85857), [@hnanni](https://github.com/hnanni))
* VMSS cache is added so that less chances of VMSS GET throttling ([#85885](https://github.com/kubernetes/kubernetes/pull/85885), [@nilo19](https://github.com/nilo19))
* Update go-winio module version from 0.4.11 to 0.4.14 ([#85739](https://github.com/kubernetes/kubernetes/pull/85739), [@wawa0210](https://github.com/wawa0210))
* Fix LoadBalancer rule checking so that no unexpected LoadBalancer updates are made ([#85990](https://github.com/kubernetes/kubernetes/pull/85990), [@feiskyer](https://github.com/feiskyer))
* kubectl drain node --dry-run will list pods that would be evicted or deleted ([#82660](https://github.com/kubernetes/kubernetes/pull/82660), [@sallyom](https://github.com/sallyom))
* Windows nodes on GCE can use TPM-based authentication to the master. ([#85466](https://github.com/kubernetes/kubernetes/pull/85466), [@pjh](https://github.com/pjh))
* kubectl/drain: add disable-eviction option. ([#85571](https://github.com/kubernetes/kubernetes/pull/85571), [@michaelgugino](https://github.com/michaelgugino))
    * Force drain to use delete, even if eviction is supported. This will bypass checking PodDisruptionBudgets, and should be used with caution.
* kubeadm now errors out whenever a not supported component config version is supplied for the kubelet and kube-proxy ([#85639](https://github.com/kubernetes/kubernetes/pull/85639), [@rosti](https://github.com/rosti))
* Fixed issue with addon-resizer using deprecated extensions APIs ([#85793](https://github.com/kubernetes/kubernetes/pull/85793), [@bskiba](https://github.com/bskiba))
* Includes FSType when describing CSI persistent volumes. ([#85293](https://github.com/kubernetes/kubernetes/pull/85293), [@huffmanca](https://github.com/huffmanca))
* kubelet now exports a "server_expiration_renew_failure" and "client_expiration_renew_failure" metric counter if the certificate rotations cannot be performed. ([#84614](https://github.com/kubernetes/kubernetes/pull/84614), [@rphillips](https://github.com/rphillips))
* kubeadm: don't write the kubelet environment file on "upgrade apply" ([#85412](https://github.com/kubernetes/kubernetes/pull/85412), [@boluisa](https://github.com/boluisa))
* fix azure file AuthorizationFailure ([#85475](https://github.com/kubernetes/kubernetes/pull/85475), [@andyzhangx](https://github.com/andyzhangx))
* Resolved regression in admission, authentication, and authorization webhook performance in v1.17.0-rc.1 ([#85810](https://github.com/kubernetes/kubernetes/pull/85810), [@liggitt](https://github.com/liggitt))
* kubeadm: uses the apiserver AdvertiseAddress IP family to choose the etcd endpoint IP family for non external etcd clusters ([#85745](https://github.com/kubernetes/kubernetes/pull/85745), [@aojea](https://github.com/aojea))
* kubeadm: Forward cluster name to the controller-manager arguments ([#85817](https://github.com/kubernetes/kubernetes/pull/85817), [@ereslibre](https://github.com/ereslibre))
* Fixed "requested device X but found Y" attach error on AWS. ([#85675](https://github.com/kubernetes/kubernetes/pull/85675), [@jsafrane](https://github.com/jsafrane))
* addons: elasticsearch discovery supports IPv6 ([#85543](https://github.com/kubernetes/kubernetes/pull/85543), [@SataQiu](https://github.com/SataQiu))
* kubeadm: retry `kubeadm-config` ConfigMap creation or mutation if the apiserver is not responding. This will improve resiliency when joining new control plane nodes. ([#85763](https://github.com/kubernetes/kubernetes/pull/85763), [@ereslibre](https://github.com/ereslibre))
* Update Cluster Autoscaler to 1.17.0; changelog: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.17.0 ([#85610](https://github.com/kubernetes/kubernetes/pull/85610), [@losipiuk](https://github.com/losipiuk))
* Filter published OpenAPI schema by making nullable, required fields non-required in order to avoid kubectl to wrongly reject null values. ([#85722](https://github.com/kubernetes/kubernetes/pull/85722), [@sttts](https://github.com/sttts))
* kubectl set resources will no longer return an error if passed an empty change for a resource.  ([#85490](https://github.com/kubernetes/kubernetes/pull/85490), [@sallyom](https://github.com/sallyom))
    * kubectl set subject will no longer return an error if passed an empty change for a resource.  
* kube-apiserver: fixed a conflict error encountered attempting to delete a pod with gracePeriodSeconds=0 and a resourceVersion precondition ([#85516](https://github.com/kubernetes/kubernetes/pull/85516), [@michaelgugino](https://github.com/michaelgugino))
* kubeadm: add a upgrade health check that deploys a Job ([#81319](https://github.com/kubernetes/kubernetes/pull/81319), [@neolit123](https://github.com/neolit123))
* kubeadm: make sure images are pre-pulled even if a tag did not change but their contents changed ([#85603](https://github.com/kubernetes/kubernetes/pull/85603), [@bart0sh](https://github.com/bart0sh))
* kube-apiserver: Fixes a bug that hidden metrics can not be enabled by the command-line option `--show-hidden-metrics-for-version`. ([#85444](https://github.com/kubernetes/kubernetes/pull/85444), [@RainbowMango](https://github.com/RainbowMango))
* kubeadm now supports automatic calculations of dual-stack node cidr masks to kube-controller-manager.  ([#85609](https://github.com/kubernetes/kubernetes/pull/85609), [@Arvinderpal](https://github.com/Arvinderpal))
* Fix bug where EndpointSlice controller would attempt to modify shared objects. ([#85368](https://github.com/kubernetes/kubernetes/pull/85368), [@robscott](https://github.com/robscott))
* Use context to check client closed instead of http.CloseNotifier in processing watch request which will reduce 1 goroutine for each request if proto is HTTP/2.x . ([#85408](https://github.com/kubernetes/kubernetes/pull/85408), [@answer1991](https://github.com/answer1991))
* kubeadm: reset raises warnings if it cannot delete folders ([#85265](https://github.com/kubernetes/kubernetes/pull/85265), [@SataQiu](https://github.com/SataQiu))
* Wait for kubelet & kube-proxy to be ready on Windows node within 10s ([#85228](https://github.com/kubernetes/kubernetes/pull/85228), [@YangLu1031](https://github.com/YangLu1031))

