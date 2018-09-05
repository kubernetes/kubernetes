<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.12.0-beta.1](#v1120-beta1)
  - [Downloads for v1.12.0-beta.1](#downloads-for-v1120-beta1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.12.0-alpha.1](#changelog-since-v1120-alpha1)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
- [v1.12.0-alpha.1](#v1120-alpha1)
  - [Downloads for v1.12.0-alpha.1](#downloads-for-v1120-alpha1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.11.0](#changelog-since-v1110)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-1)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.12.0-beta.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.12/examples)

## Downloads for v1.12.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes.tar.gz) | `caa332b14a6ea9d24710e3b015a91b62c04cab14bed14c49077e08bd82b8f4c1`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-src.tar.gz) | `821bdea3a52a348306fa8226bcfffa67b375cf1dd80e4be343ce0b38dd20a9a0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `58323c0a81afe53dd0dda1c6eb513caa4c82514fb6c7f0a327242e573ce80490`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `28e9344ede16890ea7848c261e461ded89c3bb2dd5b08446da04b071b48f0b02`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-386.tar.gz) | `a9eece5e0994d2ad5e07152d88787a8b5e9efcdf78983a5bafe3699e5274a9da`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `9a67750cc4243335f0c2eb89db1c4b54b0a8af08c59e2041636d0a3e946546bf`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `bbd2644f843917a3de517a53c90b327502b577fe533a9ad3da4fe6bc437c4a02`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `630946f49ef18dd43c004d99dccd9ae76390281f54740d7335c042f6f006324b`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `1d4e5cd83faf4cae8e16667576492fcd48a72f69e8fd89d599a8b555a41e90d6`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `9cefdcf21a62075b5238fda8ef2db08f81b0541ebce0e67353af1dded9e53483`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-windows-386.tar.gz) | `8b0085606ff38bded362bbe4826b5c8ee5199a33d5cbbc1b9b58f1336648ad5b`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `f44a3ec55dc7d926e681c33b5f7830c6d1cb165e24e349e426c1089b2d05a1df`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `1bf7364aa168fc251768bc850d66fef1d93f324f0ec85f6dce74080627599b70`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `dadc94fc0564cfa98add5287763bbe9c33bf8ba3eebad95fb2258c33fe8c5df3`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `2e6c8a7810705594f191b33476bf4c8fca8cebb364f0855dfea577b01fca7b7e`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `ced4a0a4e03639378eff0d3b8bfb832f5fb96be8df3e0befbdbd71373a323130`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `7e1a3fac2115c15b5baa0db04c7f319fbaaca92aa4c4588ecf62fb19812465a8`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `81d2e2f4cd3254dd345c1e921b12bff62eb96e7551336c44fb0da5407bf5fe5f`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `b14734a20190aca2b2af9cee59549d285be4f0c38faf89c5308c94534110edc1`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `ad0a81ecf6ef8346b7aa98a8d02a4f3853d0a5439d149a14b1ac2307b763b2ad`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `8e6d72837fe19afd055786c8731bd555fe082e107195c956c6985e56a03d504f`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `0fc7d55fb2750b29c0bbc36da050c8bf14508b1aa40e38e3b7f6cf311b464827`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `09bf133156b9bc474d272bf16e765b143439959a1f007283c477e7999f2b4d6a`

## Changelog since v1.12.0-alpha.1

### Action Required

* Move volume dynamic provisioning scheduling to beta (ACTION REQUIRED: The DynamicProvisioningScheduling alpha feature gate has been removed. The VolumeScheduling beta feature gate is still required for this feature) ([#67432](https://github.com/kubernetes/kubernetes/pull/67432), [@lichuqiang](https://github.com/lichuqiang))

### Other notable changes

* Not split nodes when searching for nodes but doing it all at once. ([#67555](https://github.com/kubernetes/kubernetes/pull/67555), [@wgliang](https://github.com/wgliang))
* Deprecate kubectl run generators, except for run-pod/v1 ([#68132](https://github.com/kubernetes/kubernetes/pull/68132), [@soltysh](https://github.com/soltysh))
* Using the Horizontal Pod Autoscaler with metrics from Heapster is now deprecated. ([#68089](https://github.com/kubernetes/kubernetes/pull/68089), [@DirectXMan12](https://github.com/DirectXMan12))
* Support both directory and block device for local volume plugin FileSystem VolumeMode  ([#63011](https://github.com/kubernetes/kubernetes/pull/63011), [@NickrenREN](https://github.com/NickrenREN))
* Add CSI volume attributes for kubectl describe pv. ([#65074](https://github.com/kubernetes/kubernetes/pull/65074), [@wgliang](https://github.com/wgliang))
* `kubectl rollout status` now works for unlimited timeouts. ([#67817](https://github.com/kubernetes/kubernetes/pull/67817), [@tnozicka](https://github.com/tnozicka))
* Fix panic when processing Azure HTTP response. ([#68210](https://github.com/kubernetes/kubernetes/pull/68210), [@feiskyer](https://github.com/feiskyer))
* add mixed protocol support for azure load balancer ([#67986](https://github.com/kubernetes/kubernetes/pull/67986), [@andyzhangx](https://github.com/andyzhangx))
* Replace scale down forbidden window with scale down stabilization window. Rather than waiting a fixed period of time between scale downs HPA now scales down to the highest recommendation it during the scale down stabilization window. ([#68122](https://github.com/kubernetes/kubernetes/pull/68122), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* Adding validation to kube-scheduler at the API level ([#66799](https://github.com/kubernetes/kubernetes/pull/66799), [@noqcks](https://github.com/noqcks))
* Improve performance of Pod affinity/anti-affinity in the scheduler ([#67788](https://github.com/kubernetes/kubernetes/pull/67788), [@ahmad-diaa](https://github.com/ahmad-diaa))
* kubeadm: fix air-gapped support and also allow some kubeadm commands to work without an available networking interface ([#67397](https://github.com/kubernetes/kubernetes/pull/67397), [@neolit123](https://github.com/neolit123))
* Increase Horizontal Pod Autoscaler default update interval (30s -> 15s). It will improve HPA reaction time for metric changes. ([#68021](https://github.com/kubernetes/kubernetes/pull/68021), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* Increase scrape frequency of metrics-server to 30s ([#68127](https://github.com/kubernetes/kubernetes/pull/68127), [@serathius](https://github.com/serathius))
* Add new `--server-dry-run` flag to `kubectl apply` so that the request will be sent to the server with the dry-run flag (alpha), which means that changes won't be persisted. ([#68069](https://github.com/kubernetes/kubernetes/pull/68069), [@apelisse](https://github.com/apelisse))
* kubelet v1beta1 external ComponentConfig types are now available in the `k8s.io/kubelet` repo ([#67263](https://github.com/kubernetes/kubernetes/pull/67263), [@luxas](https://github.com/luxas))
* Adds a kubelet parameter and config option to change CFS quota period from the default 100ms to some other value between 1µs and 1s. This was done to improve response latencies for workloads running in clusters with guaranteed and burstable QoS classes.   ([#63437](https://github.com/kubernetes/kubernetes/pull/63437), [@szuecs](https://github.com/szuecs))
* Enable secure serving on port 10258 to cloud-controller-manager (configurable via `--secure-port`). Delegated authentication and authorization have to be configured like for aggregated API servers. ([#67069](https://github.com/kubernetes/kubernetes/pull/67069), [@sttts](https://github.com/sttts))
* Support extra `--prune-whitelist` resources in kube-addon-manager. ([#67743](https://github.com/kubernetes/kubernetes/pull/67743), [@Random-Liu](https://github.com/Random-Liu))
* Upon receiving a LIST request with expired continue token, the apiserver now returns a continue token together with the 410 "the from parameter is too old " error. If the client does not care about getting a list from a consistent snapshot, the client can use this token to continue listing from the next key, but the returned chunk will be from the latest snapshot. ([#67284](https://github.com/kubernetes/kubernetes/pull/67284), [@caesarxuchao](https://github.com/caesarxuchao))
* Role, ClusterRole and their bindings for cloud-provider is put under system namespace. Their addonmanager mode switches to EnsureExists. ([#67224](https://github.com/kubernetes/kubernetes/pull/67224), [@grayluck](https://github.com/grayluck))
* Mount propagation has promoted to GA. The `MountPropagation` feature gate is deprecated and will be removed in 1.13. ([#67255](https://github.com/kubernetes/kubernetes/pull/67255), [@bertinatto](https://github.com/bertinatto))
* Introduce CSI Cluster Registration mechanism to ease CSI plugin discovery and allow CSI drivers to customize Kubernetes' interaction with them. ([#67803](https://github.com/kubernetes/kubernetes/pull/67803), [@saad-ali](https://github.com/saad-ali))
* Adds the commands `kubeadm alpha phases renew <cert-name>` ([#67910](https://github.com/kubernetes/kubernetes/pull/67910), [@liztio](https://github.com/liztio))
* ProcMount added to SecurityContext and AllowedProcMounts added to PodSecurityPolicy to allow paths in the container's /proc to not be masked. ([#64283](https://github.com/kubernetes/kubernetes/pull/64283), [@jessfraz](https://github.com/jessfraz))
* support cross resource group for azure file ([#68117](https://github.com/kubernetes/kubernetes/pull/68117), [@andyzhangx](https://github.com/andyzhangx))
* Port 31337 will be used by fluentd ([#68051](https://github.com/kubernetes/kubernetes/pull/68051), [@Szetty](https://github.com/Szetty))
* Improve CPU sample sanitization in HPA by taking metric's freshness into account. ([#68068](https://github.com/kubernetes/kubernetes/pull/68068), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* CoreDNS is now v1.2.2 for Kubernetes 1.12 ([#68076](https://github.com/kubernetes/kubernetes/pull/68076), [@rajansandeep](https://github.com/rajansandeep))
* Enable secure serving on port 10257 to kube-controller-manager (configurable via `--secure-port`). Delegated authentication and authorization have to be configured like for aggregated API servers. ([#64149](https://github.com/kubernetes/kubernetes/pull/64149), [@sttts](https://github.com/sttts))
* Update metrics-server to v0.3.0. ([#68077](https://github.com/kubernetes/kubernetes/pull/68077), [@DirectXMan12](https://github.com/DirectXMan12))
* TokenRequest and TokenRequestProjection are now beta features. To enable these feature, the API server needs to be started with the following flags: ([#67349](https://github.com/kubernetes/kubernetes/pull/67349), [@mikedanese](https://github.com/mikedanese))
        * --service-account-issuer
        * --service-account-signing-key-file
        * --service-account-api-audiences
* Don't let aggregated apiservers fail to launch if the external-apiserver-authentication configmap is not found in the cluster. ([#67836](https://github.com/kubernetes/kubernetes/pull/67836), [@sttts](https://github.com/sttts))
* Promote AdvancedAuditing to GA, replacing the previous (legacy) audit logging mechanisms. ([#65862](https://github.com/kubernetes/kubernetes/pull/65862), [@loburm](https://github.com/loburm))
* Azure cloud provider now supports unmanaged nodes (such as on-prem) that are labeled with `kubernetes.azure.com/managed=false` and `alpha.service-controller.kubernetes.io/exclude-balancer=true` ([#67984](https://github.com/kubernetes/kubernetes/pull/67984), [@feiskyer](https://github.com/feiskyer))
* `kubectl get apiservice` now shows the target service and whether the service is available ([#67747](https://github.com/kubernetes/kubernetes/pull/67747), [@smarterclayton](https://github.com/smarterclayton))
* Openstack supports now node shutdown taint. Taint is added when instance is shutdown in openstack. ([#67982](https://github.com/kubernetes/kubernetes/pull/67982), [@zetaab](https://github.com/zetaab))
* Return apiserver panics as 500 errors instead terminating the apiserver process. ([#68001](https://github.com/kubernetes/kubernetes/pull/68001), [@sttts](https://github.com/sttts))
* Fix VMWare VM freezing bug by reverting [#51066](https://github.com/kubernetes/kubernetes/pull/51066) ([#67825](https://github.com/kubernetes/kubernetes/pull/67825), [@nikopen](https://github.com/nikopen))
* Make CoreDNS be the default DNS server in kube-up (instead of kube-dns formerly).  ([#67569](https://github.com/kubernetes/kubernetes/pull/67569), [@fturib](https://github.com/fturib))
    * It is still possible to deploy kube-dns by setting CLUSTER_DNS_CORE_DNS=false.
* Added support to restore a volume from a volume snapshot data source.  ([#67087](https://github.com/kubernetes/kubernetes/pull/67087), [@xing-yang](https://github.com/xing-yang))
* fixes the errors/warnings in fluentd configuration ([#67947](https://github.com/kubernetes/kubernetes/pull/67947), [@saravanan30erd](https://github.com/saravanan30erd))
* Stop counting soft-deleted pods for scaling purposes in HPA controller to avoid soft-deleted pods incorrectly affecting scale up replica count calculation. ([#67067](https://github.com/kubernetes/kubernetes/pull/67067), [@moonek](https://github.com/moonek))
* delegated authn/z: optionally opt-out of mandatory authn/authz kubeconfig ([#67545](https://github.com/kubernetes/kubernetes/pull/67545), [@sttts](https://github.com/sttts))
* kubeadm: Control plane images (etcd, kube-apiserver, kube-proxy, etc.) don't use arch suffixes. Arch suffixes are kept for kube-dns only. ([#66960](https://github.com/kubernetes/kubernetes/pull/66960), [@rosti](https://github.com/rosti))
* Adds sample-cli-plugin staging repository ([#67938](https://github.com/kubernetes/kubernetes/pull/67938), [@soltysh](https://github.com/soltysh))
* adjusted http/2 buffer sizes for apiservers to prevent starvation issues between concurrent streams ([#67902](https://github.com/kubernetes/kubernetes/pull/67902), [@liggitt](https://github.com/liggitt))
* SCTP is now supported as additional protocol (alpha) alongside TCP and UDP in Pod, Service, Endpoint, and NetworkPolicy.   ([#64973](https://github.com/kubernetes/kubernetes/pull/64973), [@janosi](https://github.com/janosi))
* Always create configmaps/extensions-apiserver-authentication from kube-apiserver. ([#67694](https://github.com/kubernetes/kubernetes/pull/67694), [@sttts](https://github.com/sttts))
* kube-proxy v1beta1 external ComponentConfig types are now available in the `k8s.io/kube-proxy` repo ([#67688](https://github.com/kubernetes/kubernetes/pull/67688), [@Lion-Wei](https://github.com/Lion-Wei))
* Apply unreachable taint to a node when it lost network connection. ([#67734](https://github.com/kubernetes/kubernetes/pull/67734), [@Huang-Wei](https://github.com/Huang-Wei))
* Allow ImageReview backend to return annotations to be added to the created pod. ([#64597](https://github.com/kubernetes/kubernetes/pull/64597), [@wteiken](https://github.com/wteiken))
* Bump ip-masq-agent to v2.1.1 ([#67916](https://github.com/kubernetes/kubernetes/pull/67916), [@MrHohn](https://github.com/MrHohn))
    * - Update debian-iptables image for CVEs.
    * - Change chain name to IP-MASQ to be compatible with the
    * pre-injected masquerade rules.
* AllowedTopologies field inside StorageClass is now validated against set and map semantics. Specifically, there cannot be duplicate TopologySelectorTerms, MatchLabelExpressions keys, and TopologySelectorLabelRequirement Values. ([#66843](https://github.com/kubernetes/kubernetes/pull/66843), [@verult](https://github.com/verult))
* Introduces autoscaling/v2beta2 and custom_metrics/v1beta2, which implement metric selectors for Object and Pods metrics, as well as allowing AverageValue targets on Objects, similar to External metrics. ([#64097](https://github.com/kubernetes/kubernetes/pull/64097), [@damemi](https://github.com/damemi))
* The cloudstack cloud provider now reports a `Hostname` address type for nodes based on the `local-hostname` metadata key. ([#67719](https://github.com/kubernetes/kubernetes/pull/67719), [@liggitt](https://github.com/liggitt))
* kubeadm: --cri-socket now defaults to tcp://localhost:2375 when running on Windows ([#67447](https://github.com/kubernetes/kubernetes/pull/67447), [@benmoss](https://github.com/benmoss))
* kubeadm: The kubeadm configuration now support definition of more than one control plane instances with their own APIEndpoint. The APIEndpoint for the "bootstrap" control plane instance should be defined using `InitConfiguration.APIEndpoint`, while the APIEndpoints for additional control plane instances should be added using `JoinConfiguration.APIEndpoint`.   ([#67832](https://github.com/kubernetes/kubernetes/pull/67832), [@fabriziopandini](https://github.com/fabriziopandini))
* Enable dynamic azure disk volume limits ([#67772](https://github.com/kubernetes/kubernetes/pull/67772), [@andyzhangx](https://github.com/andyzhangx))
* kubelet: Users can now enable the alpha NodeLease feature gate to have the Kubelet create and periodically renew a Lease in the kube-node-lease namespace. The lease duration defaults to 40s, and can be configured via the kubelet.config.k8s.io/v1beta1.KubeletConfiguration's NodeLeaseDurationSeconds field. ([#66257](https://github.com/kubernetes/kubernetes/pull/66257), [@mtaufen](https://github.com/mtaufen))
* latent controller caches no longer cause repeating deletion messages for deleted pods ([#67826](https://github.com/kubernetes/kubernetes/pull/67826), [@deads2k](https://github.com/deads2k))
* API paging is now enabled for custom resource definitions, custom resources and APIService objects ([#67861](https://github.com/kubernetes/kubernetes/pull/67861), [@liggitt](https://github.com/liggitt))
* kubeadm: ControlPlaneEndpoint was moved from the API config struct to ClusterConfiguration ([#67830](https://github.com/kubernetes/kubernetes/pull/67830), [@fabriziopandini](https://github.com/fabriziopandini))
* kubeadm - feature-gates HighAvailability, SelfHosting, CertsInSecrets are now deprecated and can't be used anymore for new clusters. Update of cluster using above feature-gates flag is not supported ([#67786](https://github.com/kubernetes/kubernetes/pull/67786), [@fabriziopandini](https://github.com/fabriziopandini))
* Replace scale up forbidden window with disregarding CPU samples collected when pod was initializing. ([#67252](https://github.com/kubernetes/kubernetes/pull/67252), [@jbartosik](https://github.com/jbartosik))
* Moving KubeSchedulerConfiguration from ComponentConfig API types to staging repos ([#66916](https://github.com/kubernetes/kubernetes/pull/66916), [@dixudx](https://github.com/dixudx))
* Improved error message when checking the rollout status of StatefulSet with OnDelete strategy type ([#66983](https://github.com/kubernetes/kubernetes/pull/66983), [@mortent](https://github.com/mortent))
* RuntimeClass is a new API resource for defining different classes of runtimes that may be used to run containers in the cluster. Pods can select a RunitmeClass to use via the RuntimeClassName field. This feature is in alpha, and the RuntimeClass feature gate must be enabled in order to use it. ([#67737](https://github.com/kubernetes/kubernetes/pull/67737), [@tallclair](https://github.com/tallclair))
* Remove rescheduler since scheduling DS pods by default scheduler is moving to beta. ([#67687](https://github.com/kubernetes/kubernetes/pull/67687), [@Lion-Wei](https://github.com/Lion-Wei))
* Turn on PodReadinessGate by default ([#67406](https://github.com/kubernetes/kubernetes/pull/67406), [@freehan](https://github.com/freehan))
* Speed up kubelet start time by executing an immediate runtime and node status update when the Kubelet sees that it has a CIDR. ([#67031](https://github.com/kubernetes/kubernetes/pull/67031), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* The OpenStack cloud provider now reports a `Hostname` address type for nodes ([#67748](https://github.com/kubernetes/kubernetes/pull/67748), [@FengyunPan2](https://github.com/FengyunPan2))
* The aws cloud provider now reports a `Hostname` address type for nodes based on the `local-hostname` metadata key. ([#67715](https://github.com/kubernetes/kubernetes/pull/67715), [@liggitt](https://github.com/liggitt))
* Azure cloud provider now supports cross resource group nodes that are labeled with `kubernetes.azure.com/resource-group=<rg-name>` and `alpha.service-controller.kubernetes.io/exclude-balancer=true` ([#67604](https://github.com/kubernetes/kubernetes/pull/67604), [@feiskyer](https://github.com/feiskyer))
* Reduce API calls for Azure instance metadata. ([#67478](https://github.com/kubernetes/kubernetes/pull/67478), [@feiskyer](https://github.com/feiskyer))
* `kubectl create secret tls` can now read certificate and key files from process substitution arguments ([#67713](https://github.com/kubernetes/kubernetes/pull/67713), [@liggitt](https://github.com/liggitt))
* change default value of kind for azure disk ([#67483](https://github.com/kubernetes/kubernetes/pull/67483), [@andyzhangx](https://github.com/andyzhangx))
* To address the possibility dry-run requests overwhelming admission webhooks that rely on side effects and a reconciliation mechanism, a new field is being added to admissionregistration.k8s.io/v1beta1.ValidatingWebhookConfiguration and admissionregistration.k8s.io/v1beta1.MutatingWebhookConfiguration so that webhooks can explicitly register as having dry-run support. If a dry-run request is made on a resource that triggers a non dry-run supporting webhook, the request will be completely rejected, with "400: Bad Request". Additionally, a new field is being added to the admission.k8s.io/v1beta1.AdmissionReview API object, exposing to webhooks whether or not the request being reviewed is a dry-run. ([#66936](https://github.com/kubernetes/kubernetes/pull/66936), [@jennybuckley](https://github.com/jennybuckley))
* Kubeadm ha upgrade ([#66973](https://github.com/kubernetes/kubernetes/pull/66973), [@fabriziopandini](https://github.com/fabriziopandini))
* kubeadm: InitConfiguration now consists of two structs: InitConfiguration and ClusterConfiguration ([#67441](https://github.com/kubernetes/kubernetes/pull/67441), [@rosti](https://github.com/rosti))
* Updated Cluster Autoscaler version to 1.3.2-beta.2. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.2-beta.2 ([#67697](https://github.com/kubernetes/kubernetes/pull/67697), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* cpumanager: rollback state if updateContainerCPUSet failed ([#67430](https://github.com/kubernetes/kubernetes/pull/67430), [@choury](https://github.com/choury))
* [CRI] Adds a "runtime_handler" field to RunPodSandboxRequest, for selecting the runtime configuration to run the sandbox with (alpha feature). ([#67518](https://github.com/kubernetes/kubernetes/pull/67518), [@tallclair](https://github.com/tallclair))
* Create cli-runtime staging repository ([#67658](https://github.com/kubernetes/kubernetes/pull/67658), [@soltysh](https://github.com/soltysh))
* Headless Services with no ports defined will now create Endpoints correctly, and appear in DNS. ([#67622](https://github.com/kubernetes/kubernetes/pull/67622), [@thockin](https://github.com/thockin))
* Kubernetes juju charms will now use CSI for ceph. ([#66523](https://github.com/kubernetes/kubernetes/pull/66523), [@hyperbolic2346](https://github.com/hyperbolic2346))
* kubeadm:  Fix panic when node annotation is nil ([#67648](https://github.com/kubernetes/kubernetes/pull/67648), [@xlgao-zju](https://github.com/xlgao-zju))
* Prevent `resourceVersion` updates for custom resources on no-op writes. ([#67562](https://github.com/kubernetes/kubernetes/pull/67562), [@nikhita](https://github.com/nikhita))
* Fail container start if its requested device plugin resource hasn't registered after Kubelet restart. ([#67145](https://github.com/kubernetes/kubernetes/pull/67145), [@jiayingz](https://github.com/jiayingz))
* Use sync.map to scale ecache better ([#66862](https://github.com/kubernetes/kubernetes/pull/66862), [@resouer](https://github.com/resouer))
* DaemonSet: Fix bug- daemonset didn't create pod after node have enough resource ([#67337](https://github.com/kubernetes/kubernetes/pull/67337), [@linyouchong](https://github.com/linyouchong))
* updates kibana to 6.3.2  ([#67582](https://github.com/kubernetes/kubernetes/pull/67582), [@monotek](https://github.com/monotek))
* fixes json logging in fluentd-elasticsearch image by downgrading fluent-plugin-kubernetes_metadata_filter plugin to version 2.0.0 ([#67544](https://github.com/kubernetes/kubernetes/pull/67544), [@monotek](https://github.com/monotek))
* add --dns-loop-detect option to dnsmasq run by kube-dns ([#67302](https://github.com/kubernetes/kubernetes/pull/67302), [@dixudx](https://github.com/dixudx))
* Switched certificate data replacement from "REDACTED" to "DATA+OMITTED" ([#66023](https://github.com/kubernetes/kubernetes/pull/66023), [@ibrasho](https://github.com/ibrasho))
* improve performance of anti-affinity predicate of default scheduler. ([#66948](https://github.com/kubernetes/kubernetes/pull/66948), [@mohamed-mehany](https://github.com/mohamed-mehany))
* Fixed a bug that was blocking extensible error handling when serializing API responses error out. Previously, serialization failures always resulted in the status code of the original response being returned. Now, the following behavior occurs: ([#67041](https://github.com/kubernetes/kubernetes/pull/67041), [@tristanburgess](https://github.com/tristanburgess))
    *    - If the serialization type is application/vnd.kubernetes.protobuf, and protobuf marshaling is not implemented for the requested API resource type, a '406 Not Acceptable is returned'.
    *    - If the serialization type is 'application/json':
    *         - If serialization fails, and the original status code was an failure (e.g. 4xx or 5xx), the original status code will be returned.
    *         - If serialization fails, and the original status code was not a failure (e.g. 2xx), the status code of the serialization failure will be returned. By default, this is '500 Internal Server Error', because JSON serialization is our default, and not supposed to be implemented on a type-by-type basis.
* Add a feature to the scheduler to score fewer than all nodes in every scheduling cycle. This can improve performance of the scheduler in large clusters. ([#66733](https://github.com/kubernetes/kubernetes/pull/66733), [@bsalamat](https://github.com/bsalamat))
* kube-controller-manager can now start the quota controller when discovery results can only be partially determined. ([#67433](https://github.com/kubernetes/kubernetes/pull/67433), [@deads2k](https://github.com/deads2k))
* The plugin mechanism functionality now closely follows the git plugin design ([#66876](https://github.com/kubernetes/kubernetes/pull/66876), [@juanvallejo](https://github.com/juanvallejo))
* GCE: decrease cpu requests on master node, to allow more components to fit on one core machine. ([#67504](https://github.com/kubernetes/kubernetes/pull/67504), [@loburm](https://github.com/loburm))
* PVC may not be synced to controller local cache in time if PV is bound by external PV binder (e.g. kube-scheduler), double check if PVC is not found to prevent reclaiming PV wrongly. ([#67062](https://github.com/kubernetes/kubernetes/pull/67062), [@cofyc](https://github.com/cofyc))
* add more storage account sku support for azure disk ([#67528](https://github.com/kubernetes/kubernetes/pull/67528), [@andyzhangx](https://github.com/andyzhangx))
* updates es-image to elasticsearch 6.3.2 ([#67484](https://github.com/kubernetes/kubernetes/pull/67484), [@monotek](https://github.com/monotek))
* Bump GLBC version to 1.2.3 ([#66793](https://github.com/kubernetes/kubernetes/pull/66793), [@freehan](https://github.com/freehan))
* kube-apiserver: fixes error creating system priority classes when starting multiple apiservers simultaneously ([#67372](https://github.com/kubernetes/kubernetes/pull/67372), [@tanshanshan](https://github.com/tanshanshan))
* kubectl patch now respects --local ([#67399](https://github.com/kubernetes/kubernetes/pull/67399), [@deads2k](https://github.com/deads2k))
* Defaults for file audit logging backend in batch mode changed: ([#67223](https://github.com/kubernetes/kubernetes/pull/67223), [@tallclair](https://github.com/tallclair))
    * - Logs are written 1 at a time (no batching)
    * - Only a single writer process (lock contention)
* Forget rate limit when CRD establish controller successfully updated CRD condition ([#67370](https://github.com/kubernetes/kubernetes/pull/67370), [@yue9944882](https://github.com/yue9944882))
* updates fluentd in fluentd-elasticsearch to version 1.2.4 ([#67434](https://github.com/kubernetes/kubernetes/pull/67434), [@monotek](https://github.com/monotek))
        * also updates activesupport, fluent-plugin-elasticsearch & oj gems
* The dockershim now sets the "bandwidth" and "ipRanges" CNI capabilities (dynamic parameters). Plugin authors and administrators can now take advantage of this by updating their CNI configuration file. For more information, see the [CNI docs](https://github.com/containernetworking/cni/blob/master/CONVENTIONS.md#dynamic-plugin-specific-fields-capabilities--runtime-configuration) ([#64445](https://github.com/kubernetes/kubernetes/pull/64445), [@squeed](https://github.com/squeed))
* Expose `/debug/flags/v` to allow kubelet dynamically set glog logging level.  If want to change glog level to 3, you only have to send a PUT request like `curl -X PUT http://127.0.0.1:8080/debug/flags/v -d "3"`. ([#64601](https://github.com/kubernetes/kubernetes/pull/64601), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Fix an issue that pods using hostNetwork keep increasing. ([#67456](https://github.com/kubernetes/kubernetes/pull/67456), [@Huang-Wei](https://github.com/Huang-Wei))
* DaemonSet controller is now using backoff algorithm to avoid hot loops fighting with kubelet on pod recreation when a particular DaemonSet is misconfigured. ([#65309](https://github.com/kubernetes/kubernetes/pull/65309), [@tnozicka](https://github.com/tnozicka))
* Add node affinity for Azure unzoned managed disks ([#67229](https://github.com/kubernetes/kubernetes/pull/67229), [@feiskyer](https://github.com/feiskyer))
* Attacher/Detacher refactor for local storage ([#66884](https://github.com/kubernetes/kubernetes/pull/66884), [@NickrenREN](https://github.com/NickrenREN))
* Update debian-iptables and hyperkube-base images to include CVE fixes. ([#67365](https://github.com/kubernetes/kubernetes/pull/67365), [@ixdy](https://github.com/ixdy))
* Fix an issue where filesystems are not unmounted when a backend is not reachable and returns EIO. ([#67097](https://github.com/kubernetes/kubernetes/pull/67097), [@chakri-nelluri](https://github.com/chakri-nelluri))
* Update Cluster Autoscaler version to 1.3.2-beta.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.2-beta.1 ([#67396](https://github.com/kubernetes/kubernetes/pull/67396), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* Remove unused binary and container image for kube-aggregator. The functionality is already integrated into the kube-apiserver. ([#67157](https://github.com/kubernetes/kubernetes/pull/67157), [@dims](https://github.com/dims))
* Avoid creating new controller revisions for statefulsets when cache is stale ([#67039](https://github.com/kubernetes/kubernetes/pull/67039), [@mortent](https://github.com/mortent))
* Revert [#63905](https://github.com/kubernetes/kubernetes/pull/63905): Setup dns servers and search domains for Windows Pods. DNS for Windows containers will be set by CNI plugins. ([#66587](https://github.com/kubernetes/kubernetes/pull/66587), [@feiskyer](https://github.com/feiskyer))
* attachdetach controller attaches volumes immediately when Pod's PVCs are bound ([#66863](https://github.com/kubernetes/kubernetes/pull/66863), [@cofyc](https://github.com/cofyc))
* The check for unsupported plugins during volume resize has been moved from the admission controller to the two controllers that handle volume resize. ([#66780](https://github.com/kubernetes/kubernetes/pull/66780), [@kangarlou](https://github.com/kangarlou))
* Fix kubelet to not leak goroutines/intofiy watchers on an inactive connection if it's closed ([#67285](https://github.com/kubernetes/kubernetes/pull/67285), [@yujuhong](https://github.com/yujuhong))
* fix azure disk create failure due to sdk upgrade ([#67236](https://github.com/kubernetes/kubernetes/pull/67236), [@andyzhangx](https://github.com/andyzhangx))
* Kubeadm join --control-plane main workflow ([#66873](https://github.com/kubernetes/kubernetes/pull/66873), [@fabriziopandini](https://github.com/fabriziopandini))
* Dynamic provisions that create iSCSI PVs can ensure that multipath is used by specifying 2 or more target portals in the PV, which will cause kubelet to wait up to 10 seconds for the multipath device. PVs with just one portal continue to work as before, with kubelet not waiting for the multipath device and just using the first disk it finds. ([#67140](https://github.com/kubernetes/kubernetes/pull/67140), [@bswartz](https://github.com/bswartz))
* kubectl: recreating resources for immutable fields when force is applied ([#66602](https://github.com/kubernetes/kubernetes/pull/66602), [@dixudx](https://github.com/dixudx))
* Remove deprecated --interactive flag from kubectl logs. ([#65420](https://github.com/kubernetes/kubernetes/pull/65420), [@jsoref](https://github.com/jsoref))
* kubeadm uses audit policy v1 instead of v1beta1 ([#67176](https://github.com/kubernetes/kubernetes/pull/67176), [@charrywanganthony](https://github.com/charrywanganthony))
* kubeadm: make sure pre-pulled kube-proxy image and the one specified in its daemon set manifest are the same ([#67131](https://github.com/kubernetes/kubernetes/pull/67131), [@rosti](https://github.com/rosti))
* Graduate Resource Quota ScopeSelectors to beta, and enable it by default. ([#67077](https://github.com/kubernetes/kubernetes/pull/67077), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Decrease the amount of time it takes to modify kubeconfig files with large amounts of contexts ([#67093](https://github.com/kubernetes/kubernetes/pull/67093), [@juanvallejo](https://github.com/juanvallejo))
* Fixes issue when updating a DaemonSet causes a hash collision. ([#66476](https://github.com/kubernetes/kubernetes/pull/66476), [@mortent](https://github.com/mortent))
* fix cluster-info dump error ([#66652](https://github.com/kubernetes/kubernetes/pull/66652), [@charrywanganthony](https://github.com/charrywanganthony))
* The PodShareProcessNamespace feature to configure PID namespace sharing within a pod has been promoted to beta. ([#66507](https://github.com/kubernetes/kubernetes/pull/66507), [@verb](https://github.com/verb))
* `kubectl create {clusterrole,role}`'s `--resources` flag supports asterisk to specify all resources. ([#62945](https://github.com/kubernetes/kubernetes/pull/62945), [@nak3](https://github.com/nak3))
* Bump up version number of debian-base, debian-hyperkube-base and debian-iptables.  ([#67026](https://github.com/kubernetes/kubernetes/pull/67026), [@satyasm](https://github.com/satyasm))
    * Also updates dependencies of users of debian-base. 
    * debian-base version 0.3.1 is already available.
* DynamicProvisioningScheduling and VolumeScheduling is now supported for Azure managed disks. Feature gates DynamicProvisioningScheduling and VolumeScheduling should be enabled before using this feature. ([#67121](https://github.com/kubernetes/kubernetes/pull/67121), [@feiskyer](https://github.com/feiskyer))
* kube-apiserver now includes all registered API groups in discovery, including registered extension API group/versions for unavailable extension API servers. ([#66932](https://github.com/kubernetes/kubernetes/pull/66932), [@nilebox](https://github.com/nilebox))
* Allows extension API server to dynamically discover the requestheader CA certificate when the core API server doesn't use certificate based authentication for it's clients ([#66394](https://github.com/kubernetes/kubernetes/pull/66394), [@rtripat](https://github.com/rtripat))
* audit.k8s.io api group is upgraded from v1beta1 to v1. ([#65891](https://github.com/kubernetes/kubernetes/pull/65891), [@CaoShuFeng](https://github.com/CaoShuFeng))
    * Deprecated element metav1.ObjectMeta and Timestamp are removed from audit Events in v1 version.
    * Default value of option --audit-webhook-version and --audit-log-version will be changed from `audit.k8s.io/v1beta1` to `audit.k8s.io/v1` in release 1.13
* scope AWS LoadBalancer security group ICMP rules to spec.loadBalancerSourceRanges ([#63572](https://github.com/kubernetes/kubernetes/pull/63572), [@haz-mat](https://github.com/haz-mat))
* Add NoSchedule/NoExecute tolerations to ip-masq-agent, ensuring it to be scheduled in all nodes except master. ([#66260](https://github.com/kubernetes/kubernetes/pull/66260), [@tanshanshan](https://github.com/tanshanshan))
* The flag `--skip-preflight-checks` of kubeadm has been removed. Please use `--ignore-preflight-errors` instead. ([#62727](https://github.com/kubernetes/kubernetes/pull/62727), [@xiangpengzhao](https://github.com/xiangpengzhao))
* The watch API endpoints prefixed with `/watch` are deprecated and will be removed in a future release. These standard method for watching resources (supported since v1.0) is to use the list API endpoints with a `?watch=true` parameter. All client-go clients have used the parameter method since v1.6.0. ([#65147](https://github.com/kubernetes/kubernetes/pull/65147), [@liggitt](https://github.com/liggitt))
* Bump Heapster to v1.6.0-beta.1 ([#67074](https://github.com/kubernetes/kubernetes/pull/67074), [@kawych](https://github.com/kawych))
* kube-apiserver: setting a `dryRun` query parameter on a CONNECT request will now cause the request to be rejected, consistent with behavior of other mutating API requests. Examples of CONNECT APIs are the `nodes/proxy`, `services/proxy`, `pods/proxy`, `pods/exec`, and `pods/attach` subresources. Note that this prevents sending a `dryRun` parameter to backends via `{nodes,services,pods}/proxy` subresources. ([#66083](https://github.com/kubernetes/kubernetes/pull/66083), [@jennybuckley](https://github.com/jennybuckley))
* In clusters where the DryRun feature is enabled, dry-run requests will go through the normal admission chain. Because of this, ImagePolicyWebhook authors should especially make sure that their webhooks do not rely on side effects. ([#66391](https://github.com/kubernetes/kubernetes/pull/66391), [@jennybuckley](https://github.com/jennybuckley))
* Metadata Agent Improvements ([#66485](https://github.com/kubernetes/kubernetes/pull/66485), [@bmoyles0117](https://github.com/bmoyles0117))
    * Bump metadata agent version to 0.2-0.0.21-1.
    * Expand the metadata agent's access to all API groups.
    * Remove metadata agent config maps in favor of command line flags.
    * Update the metadata agent's liveness probe to a new /healthz handler.
    * Logging Agent Improvements
    * Bump logging agent version to 0.2-1.5.33-1-k8s-1.
    * Appropriately set log severity for k8s_container.
    * Fix detect exceptions plugin to analyze message field instead of log field.
    * Fix detect exceptions plugin to analyze streams based on local resource id.
    * Disable the metadata agent for monitored resource construction in logging.
    * Disable timestamp adjustment in logs to optimize performance.
    * Reduce logging agent buffer chunk limit to 512k to optimize performance.
* kubectl: the wait command now prints an error message and exits with the code 1, if there is no resources matching selectors ([#66692](https://github.com/kubernetes/kubernetes/pull/66692), [@m1kola](https://github.com/m1kola))
* Quota admission configuration api graduated to v1beta1 ([#66156](https://github.com/kubernetes/kubernetes/pull/66156), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Unit tests for scopes and scope selectors in the quota spec ([#66351](https://github.com/kubernetes/kubernetes/pull/66351), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Print kube-apiserver --help flag help in sections. ([#64517](https://github.com/kubernetes/kubernetes/pull/64517), [@sttts](https://github.com/sttts))
* Azure managed disks now support availability zones and new parameters `zoned`, `zone` and `zones` are added for AzureDisk storage class. ([#66553](https://github.com/kubernetes/kubernetes/pull/66553), [@feiskyer](https://github.com/feiskyer))
* nodes: improve handling of erroneous host names ([#64815](https://github.com/kubernetes/kubernetes/pull/64815), [@dixudx](https://github.com/dixudx))
* remove deprecated shorthand flag `-c` from `kubectl version (--client)` ([#66817](https://github.com/kubernetes/kubernetes/pull/66817), [@charrywanganthony](https://github.com/charrywanganthony))
* Added etcd_object_count metrics for CustomResources. ([#65983](https://github.com/kubernetes/kubernetes/pull/65983), [@sttts](https://github.com/sttts))
* Handle newlines for `command`, `args`, `env`, and `annotations` in `kubectl describe` wrapping ([#66841](https://github.com/kubernetes/kubernetes/pull/66841), [@smarterclayton](https://github.com/smarterclayton))
* Fix pod launch by kubelet when --cgroups-per-qos=false and --cgroup-driver="systemd" ([#66617](https://github.com/kubernetes/kubernetes/pull/66617), [@pravisankar](https://github.com/pravisankar))
* kubelet: fix nil pointer dereference while enforce-node-allocatable flag is not config properly ([#66190](https://github.com/kubernetes/kubernetes/pull/66190), [@linyouchong](https://github.com/linyouchong))
* Fix a bug on GCE that /etc/crictl.yaml is not generated when crictl is preloaded. ([#66877](https://github.com/kubernetes/kubernetes/pull/66877), [@Random-Liu](https://github.com/Random-Liu))
* This fix prevents a GCE PD volume from being mounted if the udev device link is stale and tries to correct the link. ([#66832](https://github.com/kubernetes/kubernetes/pull/66832), [@msau42](https://github.com/msau42))



# v1.12.0-alpha.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.12.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes.tar.gz) | `603345769f5e2306e5c22db928aa1cbedc6af63f387ab7a8818cb0111292133f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-src.tar.gz) | `f8fb4610cee20195381e54bfd163fbaeae228d68986817b685948b8957f324d0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `e081c275601bcaa45d906a976d35902256f836bb60caa738a2fd8719ff3e1048`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `2dd222a267ac247dce4dfc52aff313f20c427b4351f7410aadebe8569ede3139`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `46b16d6b0429163da67b06242772c3c6c5ab9da6deda5306e63d21be04b4811d`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `8b8bf0a8a4568559d3762a72c1095ab37785fc8bbbb290aaff3a34341a24d7eb`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `d71dc60e087746b2832e66170053816dc8ed42e95efe0769ed926a6e044175ef`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `e9091bbfb997d1603dfd17ba9f145ca7dacf304f04d10230e056f8a12ce44445`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `fc6c0985ccbd806add497f2557000f7e90f3176427250e019a40e8acf7c42282`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `b8c64b318d702f6e8be76330fd5da9b87e2e4e31e904ea7e00c0cd6412ab2bcf`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `cb96e353eb5d400756a93c8d16321d0fac87d6a4f8ad89fda42858f8e4d85e9d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `003284f983cafc6fd0ce1205c03d47e638a999def1ef4e1e77bfb9149e5f598b`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `d9c282cd02c8c3fdbeb2f46abd0ddd257a8449e94be3beed2514c6e30a335a87`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `613390ba73f4236feb10bb4f70cbf96e504cf8d598da0180efc887d316b8bc5e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `1dd417f59d17c3583c6b4a3989d24c57e4989eb7b6ab9f2aa10c4cbf9bf5c11b`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `44e9e6424ed3a5a91f5adefa456b2b71c0c5d3b01be9f60f5c8c0f958815ffc1`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `3118d9c955f9a50f86ebba324894f06dbf7c1cb8f9bc5bdf6a95caf2a6678805`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `6b4d363d190e0ce6f4e41d19a0ac350b39cad7859bc442166a1da9124d1a82bb`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `c80ac005c228217b871bf3e9de032044659db3aa048cc95b101820e31d62264c`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `d8b84e7cc6ff5d0e26b045de37bdd40ca8809c303b601d8604902e5957d98621`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `b0a667c5c905e6e724fba95d44797fb52afb564aedd1c25cbd4e632e152843e9`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `78e7dbb82543ea6ac70767ed63c92823726adb6257f6b70b5911843d18288df7`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `1a3e11cc3f1a0297de2b894a43eb56ede5fbd5cdc43e4da7e61171f5c1f3ef60`

## Changelog since v1.11.0

### Action Required

* action required: the API server and client-go libraries have been fixed to support additional non-alpha-numeric characters in UserInfo "extra" data keys. Both should be updated in order to properly support extra data containing "/" characters or other characters disallowed in HTTP headers. ([#65799](https://github.com/kubernetes/kubernetes/pull/65799), [@dekkagaijin](https://github.com/dekkagaijin))
* [action required] The `NodeConfiguration` kind in the kubeadm v1alpha2 API has been renamed `JoinConfiguration` in v1alpha3 ([#65951](https://github.com/kubernetes/kubernetes/pull/65951), [@luxas](https://github.com/luxas))
* ACTION REQUIRED: Removes defaulting of CSI file system type to ext4. All the production drivers listed under https://kubernetes-csi.github.io/docs/Drivers.html were inspected and should not be impacted after this change. If you are using a driver not in that list, please test the drivers on an updated test cluster first. ``` ([#65499](https://github.com/kubernetes/kubernetes/pull/65499), [@krunaljain](https://github.com/krunaljain))
* [action required] The `MasterConfiguration` kind in the kubeadm v1alpha2 API has been renamed `InitConfiguration` in v1alpha3 ([#65945](https://github.com/kubernetes/kubernetes/pull/65945), [@luxas](https://github.com/luxas))
* [action required] The formerly publicly-available cAdvisor web UI that the kubelet started using `--cadvisor-port` is now entirely removed in 1.12. The recommended way to run cAdvisor if you still need it, is via a DaemonSet. ([#65707](https://github.com/kubernetes/kubernetes/pull/65707), [@dims](https://github.com/dims))
* Cluster Autoscaler version updated to 1.3.1-beta.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.1-beta.1 ([#65857](https://github.com/kubernetes/kubernetes/pull/65857), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
    * Default value for expendable pod priority cutoff in GCP deployment of Cluster Autoscaler changed from 0 to -10.
    * action required: users deploying workloads with priority lower than 0 may want to use priority lower than -10 to avoid triggering scale-up.
* [action required] kubeadm: The `v1alpha1` config API has been removed. ([#65628](https://github.com/kubernetes/kubernetes/pull/65628), [@luxas](https://github.com/luxas))
    * Please convert your `v1alpha1` configuration files to `v1alpha2` using the
    * `kubeadm config migrate` command of kubeadm v1.11.x
* kube-apiserver: the `Priority` admission plugin is now enabled by default when using `--enable-admission-plugins`. If using `--admission-control` to fully specify the set of admission plugins, the `Priority` admission plugin should be added if using the `PodPriority` feature, which is enabled by default in 1.11. ([#65739](https://github.com/kubernetes/kubernetes/pull/65739), [@liggitt](https://github.com/liggitt))
* The `system-node-critical` and `system-cluster-critical` priority classes are now limited to the `kube-system` namespace by the `PodPriority` admission plugin. ([#65593](https://github.com/kubernetes/kubernetes/pull/65593), [@bsalamat](https://github.com/bsalamat))
* kubernetes-worker juju charm: Added support for setting the --enable-ssl-chain-completion option on the ingress proxy.  "action required": if your installation relies on supplying incomplete certificate chains and using OCSP to fill them in, you must set "ingress-ssl-chain-completion" to "true" in your juju configuration. ([#63845](https://github.com/kubernetes/kubernetes/pull/63845), [@paulgear](https://github.com/paulgear))

### Other notable changes

* admin RBAC role now aggregates edit and view.  edit RBAC role now aggregates view.  ([#66684](https://github.com/kubernetes/kubernetes/pull/66684), [@deads2k](https://github.com/deads2k))
* Speed up HPA reaction to metric changes by removing scale up forbidden window. ([#66615](https://github.com/kubernetes/kubernetes/pull/66615), [@jbartosik](https://github.com/jbartosik))
    * Scale up forbidden window was protecting HPA against making decision to scale up based on metrics gathered during pod initialisation (which may be invalid, for example pod may be using a lot of CPU despite not doing any "actual" work).
    * To avoid that negative effect only use per pod metrics from pods that are:
    * - ready (so metrics about them should be valid), or
    * - unready but creation and last readiness change timestamps are apart more than 10s (pods that have formerly been ready and so metrics are in at least some cases (pod becoming unready because of overload) very useful).
* The `kubectl patch` command no longer exits with exit code 1 when a redundant patch results in a no-op ([#66725](https://github.com/kubernetes/kubernetes/pull/66725), [@juanvallejo](https://github.com/juanvallejo))
* Improved the output of `kubectl get events` to prioritize showing the message, and move some fields to `-o wide`. ([#66643](https://github.com/kubernetes/kubernetes/pull/66643), [@smarterclayton](https://github.com/smarterclayton))
* Added CPU Manager state validation in case of changed CPU topology. ([#66718](https://github.com/kubernetes/kubernetes/pull/66718), [@ipuustin](https://github.com/ipuustin))
* Make EBS volume expansion faster ([#66728](https://github.com/kubernetes/kubernetes/pull/66728), [@gnufied](https://github.com/gnufied))
* Kubelet serving certificate bootstrapping and rotation has been promoted to beta status. ([#66726](https://github.com/kubernetes/kubernetes/pull/66726), [@liggitt](https://github.com/liggitt))
* Flag --pod (-p shorthand) of kubectl exec command marked as deprecated ([#66558](https://github.com/kubernetes/kubernetes/pull/66558), [@quasoft](https://github.com/quasoft))
* Fixed an issue which prevented `gcloud` from working on GCE when metadata concealment was enabled. ([#66630](https://github.com/kubernetes/kubernetes/pull/66630), [@dekkagaijin](https://github.com/dekkagaijin))
* Azure Go SDK has been upgraded to v19.0.0 and VirtualMachineScaleSetVM now supports availability zones. ([#66648](https://github.com/kubernetes/kubernetes/pull/66648), [@feiskyer](https://github.com/feiskyer))
* kubeadm now can join the cluster with pre-existing client certificate if provided ([#66482](https://github.com/kubernetes/kubernetes/pull/66482), [@dixudx](https://github.com/dixudx))
* If `TaintNodesByCondition` enabled, taint node with `TaintNodeUnschedulable` when ([#63955](https://github.com/kubernetes/kubernetes/pull/63955), [@k82cn](https://github.com/k82cn))
    * initializing node to avoid race condition.
* kubeadm: remove misleading error message regarding image pulling ([#66658](https://github.com/kubernetes/kubernetes/pull/66658), [@dixudx](https://github.com/dixudx))
* Fix Stackdriver integration based on node annotation container.googleapis.com/instance_id. ([#66676](https://github.com/kubernetes/kubernetes/pull/66676), [@kawych](https://github.com/kawych))
* Fix kubelet startup failure when using ExecPlugin in kubeconfig ([#66395](https://github.com/kubernetes/kubernetes/pull/66395), [@awly](https://github.com/awly))
* When attaching iSCSI volumes, kubelet now scans only the specific ([#63176](https://github.com/kubernetes/kubernetes/pull/63176), [@bswartz](https://github.com/bswartz))
    * LUNs being attached, and also deletes them after detaching. This avoids
    * dangling references to LUNs that no longer exist, which used to be the
    * cause of random I/O errors/timeouts in kernel logs, slowdowns during
    * block-device related operations, and very rare cases of data corruption.
* kubeadm: Pull sidecar and dnsmasq-nanny images when using kube-dns ([#66499](https://github.com/kubernetes/kubernetes/pull/66499), [@rosti](https://github.com/rosti))
* Extender preemption should respect IsInterested() ([#66291](https://github.com/kubernetes/kubernetes/pull/66291), [@resouer](https://github.com/resouer))
* Properly autopopulate OpenAPI version field without needing other OpenAPI fields present in generic API server code. ([#66411](https://github.com/kubernetes/kubernetes/pull/66411), [@DirectXMan12](https://github.com/DirectXMan12))
* renamed command line option  --cri-socket-path of the kubeadm subcommand "kubeadm config images pull" to --cri-socket to be consistent with the rest of kubeadm subcommands. ([#66382](https://github.com/kubernetes/kubernetes/pull/66382), [@bart0sh](https://github.com/bart0sh))
* The --docker-disable-shared-pid kubelet flag has been removed. PID namespace sharing can instead be enable per-pod using the ShareProcessNamespace option. ([#66506](https://github.com/kubernetes/kubernetes/pull/66506), [@verb](https://github.com/verb))
* Add support for using User Assigned MSI (https://docs.microsoft.com/en-us/azure/active-directory/managed-service-identity/overview) with Kubernetes cluster on Azure. ([#66180](https://github.com/kubernetes/kubernetes/pull/66180), [@kkmsft](https://github.com/kkmsft))
* fix acr could not be listed in sp issue ([#66429](https://github.com/kubernetes/kubernetes/pull/66429), [@andyzhangx](https://github.com/andyzhangx))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63665](https://github.com/kubernetes/kubernetes/pull/63665), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* Fix volume limit for EBS on m5 and c5 instance types ([#66397](https://github.com/kubernetes/kubernetes/pull/66397), [@gnufied](https://github.com/gnufied))
* Extend TLS timeouts to work around slow arm64 math/big ([#66264](https://github.com/kubernetes/kubernetes/pull/66264), [@joejulian](https://github.com/joejulian))
* kubeadm: stop setting UID in the kubelet ConfigMap ([#66341](https://github.com/kubernetes/kubernetes/pull/66341), [@runiq](https://github.com/runiq))
* kubectl: fixes a panic displaying pods with nominatedNodeName set ([#66406](https://github.com/kubernetes/kubernetes/pull/66406), [@liggitt](https://github.com/liggitt))
* Update crictl to v1.11.1. ([#66152](https://github.com/kubernetes/kubernetes/pull/66152), [@Random-Liu](https://github.com/Random-Liu))
* fixes a panic when using a mutating webhook admission plugin with a DELETE operation ([#66425](https://github.com/kubernetes/kubernetes/pull/66425), [@liggitt](https://github.com/liggitt))
* GCE: Fixes loadbalancer creation and deletion issues appearing in 1.10.5. ([#66400](https://github.com/kubernetes/kubernetes/pull/66400), [@nicksardo](https://github.com/nicksardo))
* Azure nodes with availability zone now will have label `failure-domain.beta.kubernetes.io/zone=<region>-<zoneID>`. ([#66242](https://github.com/kubernetes/kubernetes/pull/66242), [@feiskyer](https://github.com/feiskyer))
* Re-design equivalence class cache to two level cache ([#65714](https://github.com/kubernetes/kubernetes/pull/65714), [@resouer](https://github.com/resouer))
* Checks CREATE admission for create-on-update requests instead of UPDATE admission ([#65572](https://github.com/kubernetes/kubernetes/pull/65572), [@yue9944882](https://github.com/yue9944882))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63666](https://github.com/kubernetes/kubernetes/pull/63666), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* Fixed a panic in the node status update logic when existing node has nil labels. ([#66307](https://github.com/kubernetes/kubernetes/pull/66307), [@guoshimin](https://github.com/guoshimin))
* Bump Ingress-gce version to 1.2.0 ([#65641](https://github.com/kubernetes/kubernetes/pull/65641), [@freehan](https://github.com/freehan))
* Bump event-exporter to 0.2.2 to pick up security fixes. ([#66157](https://github.com/kubernetes/kubernetes/pull/66157), [@loburm](https://github.com/loburm))
* Allow ScaleIO volumes to be provisioned without having to first manually create /dev/disk/by-id path on each kubernetes node (if not already present) ([#66174](https://github.com/kubernetes/kubernetes/pull/66174), [@ddebroy](https://github.com/ddebroy))
* fix rollout status for statefulsets ([#62943](https://github.com/kubernetes/kubernetes/pull/62943), [@faraazkhan](https://github.com/faraazkhan))
* Fix for resourcepool-path configuration in the vsphere.conf file. ([#66261](https://github.com/kubernetes/kubernetes/pull/66261), [@divyenpatel](https://github.com/divyenpatel))
* OpenAPI spec and documentation reflect 202 Accepted response path for delete request ([#63418](https://github.com/kubernetes/kubernetes/pull/63418), [@roycaihw](https://github.com/roycaihw))
* fixes a validation error that could prevent updates to StatefulSet objects containing non-normalized resource requests ([#66165](https://github.com/kubernetes/kubernetes/pull/66165), [@liggitt](https://github.com/liggitt))
* Fix validation for HealthzBindAddress in kube-proxy when --healthz-port is set to 0 ([#66138](https://github.com/kubernetes/kubernetes/pull/66138), [@wsong](https://github.com/wsong))
* kubeadm: use an HTTP request timeout when fetching the latest version of Kubernetes from dl.k8s.io ([#65676](https://github.com/kubernetes/kubernetes/pull/65676), [@dkoshkin](https://github.com/dkoshkin))
* Support configuring the Azure load balancer idle connection timeout for services ([#66045](https://github.com/kubernetes/kubernetes/pull/66045), [@cpuguy83](https://github.com/cpuguy83))
* `kubectl config set-context` can now set attributes of the current context, like the current namespace, by passing `--current` instead of a specific context name ([#66140](https://github.com/kubernetes/kubernetes/pull/66140), [@liggitt](https://github.com/liggitt))
* The alpha `Initializers` admission plugin is no longer enabled by default. This matches the off-by-default behavior of the alpha API which drives initializer behavior. ([#66039](https://github.com/kubernetes/kubernetes/pull/66039), [@liggitt](https://github.com/liggitt))
* kubeadm: Default component configs are printable via kubeadm config print-default ([#66074](https://github.com/kubernetes/kubernetes/pull/66074), [@rosti](https://github.com/rosti))
* prevents infinite CLI wait on delete when item is recreated ([#66136](https://github.com/kubernetes/kubernetes/pull/66136), [@deads2k](https://github.com/deads2k))
* Preserve vmUUID when renewing nodeinfo in vSphere cloud provider ([#66007](https://github.com/kubernetes/kubernetes/pull/66007), [@w-leads](https://github.com/w-leads))
* Cluster Autoscaler version updated to 1.3.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.1 ([#66122](https://github.com/kubernetes/kubernetes/pull/66122), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* Expose docker registry config for addons used in Juju deployments ([#66092](https://github.com/kubernetes/kubernetes/pull/66092), [@kwmonroe](https://github.com/kwmonroe))
* kubelets that specify `--cloud-provider` now only report addresses in Node status as determined by the cloud provider ([#65594](https://github.com/kubernetes/kubernetes/pull/65594), [@liggitt](https://github.com/liggitt))
        * kubelet serving certificate rotation now reacts to changes in reported node addresses, and will request certificates for addresses set by an external cloud provider
* Fix the bug where image garbage collection is disabled by mistake. ([#66051](https://github.com/kubernetes/kubernetes/pull/66051), [@jiaxuanzhou](https://github.com/jiaxuanzhou))
* fixes an issue with multi-line annotations injected via downward API files getting scrambled ([#65992](https://github.com/kubernetes/kubernetes/pull/65992), [@liggitt](https://github.com/liggitt))
* kubeadm: run kube-proxy on non-master tainted nodes ([#65931](https://github.com/kubernetes/kubernetes/pull/65931), [@neolit123](https://github.com/neolit123))
* "kubectl delete" no longer waits for dependent objects to be deleted when removing parent resources ([#65908](https://github.com/kubernetes/kubernetes/pull/65908), [@juanvallejo](https://github.com/juanvallejo))
* Introduce a new flag `--keepalive` for kubectl proxy to allow setting keep-alive period for long-running request. ([#63793](https://github.com/kubernetes/kubernetes/pull/63793), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* If Openstack LoadBalancer is not defined in cloud config, the loadbalancer is not initialized any more in openstack. All setups must have some setting under that section ([#65781](https://github.com/kubernetes/kubernetes/pull/65781), [@zetaab](https://github.com/zetaab))
* Re-adds `pkg/generated/bindata.go` to the repository to allow some parts of k8s.io/kubernetes to be go-vendorable. ([#65985](https://github.com/kubernetes/kubernetes/pull/65985), [@ixdy](https://github.com/ixdy))
* Fix a bug that preempting a pod may block forever. ([#65987](https://github.com/kubernetes/kubernetes/pull/65987), [@Random-Liu](https://github.com/Random-Liu))
* Fix flexvolume in containarized kubelets ([#65549](https://github.com/kubernetes/kubernetes/pull/65549), [@gnufied](https://github.com/gnufied))
* Add volume mode filed to constructed volume spec for CSI plugin ([#65456](https://github.com/kubernetes/kubernetes/pull/65456), [@wenlxie](https://github.com/wenlxie))
* Fix an issue with dropped audit logs, when truncating and batch backends enabled at the same time. ([#65823](https://github.com/kubernetes/kubernetes/pull/65823), [@loburm](https://github.com/loburm))
* Support traffic shaping for CNI network driver ([#63194](https://github.com/kubernetes/kubernetes/pull/63194), [@m1093782566](https://github.com/m1093782566))
* kubeadm: Use separate YAML documents for the kubelet and kube-proxy ComponentConfigs ([#65787](https://github.com/kubernetes/kubernetes/pull/65787), [@luxas](https://github.com/luxas))
* kubeadm: Fix pause image to not use architecture, as it is a manifest list ([#65920](https://github.com/kubernetes/kubernetes/pull/65920), [@dims](https://github.com/dims))
* kubeadm: print required flags when running kubeadm upgrade plan ([#65802](https://github.com/kubernetes/kubernetes/pull/65802), [@xlgao-zju](https://github.com/xlgao-zju))
* Fix `RunAsGroup` which doesn't work since 1.10. ([#65926](https://github.com/kubernetes/kubernetes/pull/65926), [@Random-Liu](https://github.com/Random-Liu))
* Running `kubectl describe pvc` now shows which pods are mounted to the pvc being described with the `Mounted By` field ([#65837](https://github.com/kubernetes/kubernetes/pull/65837), [@clandry94](https://github.com/clandry94))
* fix azure storage account creation failure ([#65846](https://github.com/kubernetes/kubernetes/pull/65846), [@andyzhangx](https://github.com/andyzhangx))
* Allow kube- and cloud-controller-manager to listen on ports up to 65535. ([#65860](https://github.com/kubernetes/kubernetes/pull/65860), [@sttts](https://github.com/sttts))
* Allow kube-scheduler to listen on ports up to 65535. ([#65833](https://github.com/kubernetes/kubernetes/pull/65833), [@sttts](https://github.com/sttts))
* kubeadm: Remove usage of `PersistentVolumeLabel` ([#65827](https://github.com/kubernetes/kubernetes/pull/65827), [@xlgao-zju](https://github.com/xlgao-zju))
* kubeadm: Add a `v1alpha3` API. ([#65629](https://github.com/kubernetes/kubernetes/pull/65629), [@luxas](https://github.com/luxas))
* Update to use go1.10.3 ([#65726](https://github.com/kubernetes/kubernetes/pull/65726), [@ixdy](https://github.com/ixdy))
* LimitRange and Endpoints resources can be created via an update API call if the object does not already exist. When this occurs, an authorization check is now made to ensure the user making the API call is authorized to create the object. In previous releases, only an update authorization check was performed. ([#65150](https://github.com/kubernetes/kubernetes/pull/65150), [@jennybuckley](https://github.com/jennybuckley))
* Fix 'kubectl cp' with no arguments causes a panic ([#65482](https://github.com/kubernetes/kubernetes/pull/65482), [@wgliang](https://github.com/wgliang))
* bazel deb package bugfix: The kubeadm deb package now reloads the kubelet after installation ([#65554](https://github.com/kubernetes/kubernetes/pull/65554), [@rdodev](https://github.com/rdodev))
* fix smb mount issue ([#65751](https://github.com/kubernetes/kubernetes/pull/65751), [@andyzhangx](https://github.com/andyzhangx))
* More fields are allowed at the root of the CRD validation schema when the status subresource is enabled. ([#65357](https://github.com/kubernetes/kubernetes/pull/65357), [@nikhita](https://github.com/nikhita))
* Reload systemd config files before starting kubelet. ([#65702](https://github.com/kubernetes/kubernetes/pull/65702), [@mborsz](https://github.com/mborsz))
* Unix: support ZFS as a valid graph driver for Docker ([#65635](https://github.com/kubernetes/kubernetes/pull/65635), [@neolit123](https://github.com/neolit123))
* Fix controller-manager crashes when flex plugin is removed from flex plugin directory ([#65536](https://github.com/kubernetes/kubernetes/pull/65536), [@gnufied](https://github.com/gnufied))
* Enable etcdv3 client prometheus metics ([#64741](https://github.com/kubernetes/kubernetes/pull/64741), [@wgliang](https://github.com/wgliang))
* skip nodes that have a primary NIC in a 'Failed' provisioningState ([#65412](https://github.com/kubernetes/kubernetes/pull/65412), [@yastij](https://github.com/yastij))
* kubeadm: remove redundant flags settings for kubelet ([#64682](https://github.com/kubernetes/kubernetes/pull/64682), [@dixudx](https://github.com/dixudx))
* Fixes the wrong elasticsearch node counter ([#65627](https://github.com/kubernetes/kubernetes/pull/65627), [@IvanovOleg](https://github.com/IvanovOleg))
* - Can configure the vsphere cloud provider with a trusted Root-CA ([#64758](https://github.com/kubernetes/kubernetes/pull/64758), [@mariantalla](https://github.com/mariantalla))
* Add Ubuntu 18.04 (Bionic) series to Juju charms ([#65644](https://github.com/kubernetes/kubernetes/pull/65644), [@tvansteenburgh](https://github.com/tvansteenburgh))
* Fix local volume directory can't be deleted because of volumeMode error ([#65310](https://github.com/kubernetes/kubernetes/pull/65310), [@wenlxie](https://github.com/wenlxie))
* kubectl: --use-openapi-print-columns is deprecated in favor of --server-print ([#65601](https://github.com/kubernetes/kubernetes/pull/65601), [@liggitt](https://github.com/liggitt))
* Add prometheus scrape port to CoreDNS service ([#65589](https://github.com/kubernetes/kubernetes/pull/65589), [@rajansandeep](https://github.com/rajansandeep))
* fixes an out of range panic in the NoExecuteTaintManager controller when running a non-64-bit build ([#65596](https://github.com/kubernetes/kubernetes/pull/65596), [@liggitt](https://github.com/liggitt))
* kubectl: fixes a regression with --use-openapi-print-columns that would not print object contents ([#65600](https://github.com/kubernetes/kubernetes/pull/65600), [@liggitt](https://github.com/liggitt))
* Hostnames are now converted to lowercase before being used for node lookups in the kubernetes-worker charm. ([#65487](https://github.com/kubernetes/kubernetes/pull/65487), [@dshcherb](https://github.com/dshcherb))
* N/A ([#64660](https://github.com/kubernetes/kubernetes/pull/64660), [@figo](https://github.com/figo))
* bugfix: Do not print feature gates in the generic apiserver code for glog level 0 ([#65584](https://github.com/kubernetes/kubernetes/pull/65584), [@neolit123](https://github.com/neolit123))
* Add metrics for PVC in-use ([#64527](https://github.com/kubernetes/kubernetes/pull/64527), [@gnufied](https://github.com/gnufied))
* Fixed exception detection in fluentd-gcp plugin. ([#65361](https://github.com/kubernetes/kubernetes/pull/65361), [@xperimental](https://github.com/xperimental))
* api-machinery utility functions `SetTransportDefaults` and `DialerFor` once again respect custom Dial functions set on transports ([#65547](https://github.com/kubernetes/kubernetes/pull/65547), [@liggitt](https://github.com/liggitt))
* Improve the display of jobs in `kubectl get` and `kubectl describe` to emphasize progress and duration. ([#65463](https://github.com/kubernetes/kubernetes/pull/65463), [@smarterclayton](https://github.com/smarterclayton))
* kubectl convert previous created a list inside of a list.  Now it is only wrapped once. ([#65489](https://github.com/kubernetes/kubernetes/pull/65489), [@deads2k](https://github.com/deads2k))
* fix azure disk creation issue when specifying external resource group ([#65516](https://github.com/kubernetes/kubernetes/pull/65516), [@andyzhangx](https://github.com/andyzhangx))
* fixes a regression in kube-scheduler to properly load client connection information from a `--config` file that references a kubeconfig file ([#65507](https://github.com/kubernetes/kubernetes/pull/65507), [@liggitt](https://github.com/liggitt))
* Fixed cleanup of CSI metadata files. ([#65323](https://github.com/kubernetes/kubernetes/pull/65323), [@jsafrane](https://github.com/jsafrane))
* Update Rescheduler's manifest to use version 0.4.0. ([#65454](https://github.com/kubernetes/kubernetes/pull/65454), [@bsalamat](https://github.com/bsalamat))
* On COS, NPD creates a node condition for frequent occurrences of unregister_netdevice ([#65342](https://github.com/kubernetes/kubernetes/pull/65342), [@dashpole](https://github.com/dashpole))
* Properly manage security groups for loadbalancer services on OpenStack. ([#65373](https://github.com/kubernetes/kubernetes/pull/65373), [@multi-io](https://github.com/multi-io))
* Add user-agent to audit-logging. ([#64812](https://github.com/kubernetes/kubernetes/pull/64812), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* kubeadm: notify the user of manifest upgrade timeouts ([#65164](https://github.com/kubernetes/kubernetes/pull/65164), [@xlgao-zju](https://github.com/xlgao-zju))
* Fixes incompatibility with custom scheduler extender configurations specifying `bindVerb` ([#65424](https://github.com/kubernetes/kubernetes/pull/65424), [@liggitt](https://github.com/liggitt))
* Using `kubectl describe` on CRDs that use underscores will be prettier. ([#65391](https://github.com/kubernetes/kubernetes/pull/65391), [@smarterclayton](https://github.com/smarterclayton))
* Improve scheduler's performance by eliminating sorting of nodes by their score. ([#65396](https://github.com/kubernetes/kubernetes/pull/65396), [@bsalamat](https://github.com/bsalamat))
* Add more conditions to the list of predicate failures that won't be resolved by preemption. ([#64995](https://github.com/kubernetes/kubernetes/pull/64995), [@bsalamat](https://github.com/bsalamat))
* Allow access to ClusterIP from the host network namespace when kube-proxy is started in IPVS mode without either masqueradeAll or clusterCIDR flags ([#65388](https://github.com/kubernetes/kubernetes/pull/65388), [@lbernail](https://github.com/lbernail))
* User can now use `sudo crictl` on GCE cluster. ([#65389](https://github.com/kubernetes/kubernetes/pull/65389), [@Random-Liu](https://github.com/Random-Liu))
* Tolerate missing watch permission when deleting a resource ([#65370](https://github.com/kubernetes/kubernetes/pull/65370), [@deads2k](https://github.com/deads2k))
* Prevents a `kubectl delete` hang when deleting controller managed lists ([#65367](https://github.com/kubernetes/kubernetes/pull/65367), [@deads2k](https://github.com/deads2k))
* fixes a memory leak in the kube-controller-manager observed when large numbers of pods with tolerations are created/deleted ([#65339](https://github.com/kubernetes/kubernetes/pull/65339), [@liggitt](https://github.com/liggitt))
* checkLimitsForResolvConf for the  pod create and update events instead of checking period ([#64860](https://github.com/kubernetes/kubernetes/pull/64860), [@wgliang](https://github.com/wgliang))
* Fix concurrent map access panic ([#65334](https://github.com/kubernetes/kubernetes/pull/65334), [@dashpole](https://github.com/dashpole))
    * Don't watch .mount cgroups to reduce number of inotify watches
    * Fix NVML initialization race condition
    * Fix brtfs disk metrics when using a subdirectory of a subvolume
* Change Azure ARM Rate limiting error message. ([#65292](https://github.com/kubernetes/kubernetes/pull/65292), [@wgliang](https://github.com/wgliang))
* AWS now checks for validity of ecryption key when creating encrypted volumes. Dynamic provisioning of encrypted volume may get slower due to these checks. ([#65223](https://github.com/kubernetes/kubernetes/pull/65223), [@jsafrane](https://github.com/jsafrane))
* Report accurate status for kubernetes-master and -worker charms. ([#65187](https://github.com/kubernetes/kubernetes/pull/65187), [@kwmonroe](https://github.com/kwmonroe))
* Fixed issue 63608, which is that under rare circumstances the ResourceQuota admission controller could lose track of an request in progress and time out after waiting 10 seconds for a decision to be made. ([#64598](https://github.com/kubernetes/kubernetes/pull/64598), [@MikeSpreitzer](https://github.com/MikeSpreitzer))
* In the vSphere cloud provider the `Global.vm-uuid` configuration option is not deprecated anymore, it can be used to overwrite the VMUUID on the controller-manager ([#65152](https://github.com/kubernetes/kubernetes/pull/65152), [@alvaroaleman](https://github.com/alvaroaleman))
* fluentd-gcp grace termination period increased to 60s. ([#65084](https://github.com/kubernetes/kubernetes/pull/65084), [@x13n](https://github.com/x13n))
* Pass cluster_location argument to Heapster ([#65176](https://github.com/kubernetes/kubernetes/pull/65176), [@kawych](https://github.com/kawych))
* Fix a scalability issue where high rates of event writes degraded etcd performance. ([#64539](https://github.com/kubernetes/kubernetes/pull/64539), [@ccding](https://github.com/ccding))
* Corrected a mistake in the documentation for wait.PollImmediate(...) ([#65026](https://github.com/kubernetes/kubernetes/pull/65026), [@spew](https://github.com/spew))
* Split 'scheduling_latency_seconds' metric into finer steps (predicate, priority, premption) ([#65306](https://github.com/kubernetes/kubernetes/pull/65306), [@shyamjvs](https://github.com/shyamjvs))
* Etcd health checks by the apiserver now ensure the apiserver can connect to and exercise the etcd API ([#65027](https://github.com/kubernetes/kubernetes/pull/65027), [@liggitt](https://github.com/liggitt))
* Add e2e regression tests for the kubelet being secure ([#64140](https://github.com/kubernetes/kubernetes/pull/64140), [@dixudx](https://github.com/dixudx))
* set EnableHTTPSTrafficOnly in azure storage account creation ([#64957](https://github.com/kubernetes/kubernetes/pull/64957), [@andyzhangx](https://github.com/andyzhangx))
* Fixes an issue where Portworx PVCs remain in pending state when created using a StorageClass with empty parameters ([#64895](https://github.com/kubernetes/kubernetes/pull/64895), [@harsh-px](https://github.com/harsh-px))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63662](https://github.com/kubernetes/kubernetes/pull/63662), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63661](https://github.com/kubernetes/kubernetes/pull/63661), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63660](https://github.com/kubernetes/kubernetes/pull/63660), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* Updated default image for nginx ingress in CDK to match current Kubernetes docs. ([#64285](https://github.com/kubernetes/kubernetes/pull/64285), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Added block volume support to Cinder volume plugin. ([#64879](https://github.com/kubernetes/kubernetes/pull/64879), [@bertinatto](https://github.com/bertinatto))
* fixed incorrect OpenAPI schema for CustomResourceDefinition objects ([#65256](https://github.com/kubernetes/kubernetes/pull/65256), [@liggitt](https://github.com/liggitt))
* ignore not found file error when watching manifests ([#64880](https://github.com/kubernetes/kubernetes/pull/64880), [@dixudx](https://github.com/dixudx))
* add port-forward examples for sevice ([#64773](https://github.com/kubernetes/kubernetes/pull/64773), [@MasayaAoyama](https://github.com/MasayaAoyama))
* Fix issues for block device not mapped to container. ([#64555](https://github.com/kubernetes/kubernetes/pull/64555), [@wenlxie](https://github.com/wenlxie))
* Update crictl on GCE to v1.11.0. ([#65254](https://github.com/kubernetes/kubernetes/pull/65254), [@Random-Liu](https://github.com/Random-Liu))
* Fixes missing nodes lines when kubectl top nodes ([#64389](https://github.com/kubernetes/kubernetes/pull/64389), [@yue9944882](https://github.com/yue9944882))
* keep pod state consistent when scheduler cache UpdatePod ([#64692](https://github.com/kubernetes/kubernetes/pull/64692), [@adohe](https://github.com/adohe))
* add external resource group support for azure disk ([#64427](https://github.com/kubernetes/kubernetes/pull/64427), [@andyzhangx](https://github.com/andyzhangx))
* Increase the gRPC max message size to 16MB in the remote container runtime. ([#64672](https://github.com/kubernetes/kubernetes/pull/64672), [@mcluseau](https://github.com/mcluseau))
* The new default value for the --allow-privileged parameter of the Kubernetes-worker charm has been set to true based on changes which went into the Kubernetes 1.10 release. Before this change the default value was set to false. If you're installing Canonical Kubernetes you should expect this value to now be true by default and you should now look to use PSP (pod security policies).  ([#64104](https://github.com/kubernetes/kubernetes/pull/64104), [@CalvinHartwell](https://github.com/CalvinHartwell))
* The --remove-extra-subjects and --remove-extra-permissions flags have been enabled for kubectl auth reconcile ([#64541](https://github.com/kubernetes/kubernetes/pull/64541), [@mrogers950](https://github.com/mrogers950))
* Fix kubectl drain --timeout option when eviction is used. ([#64378](https://github.com/kubernetes/kubernetes/pull/64378), [@wrdls](https://github.com/wrdls))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63659](https://github.com/kubernetes/kubernetes/pull/63659), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.

