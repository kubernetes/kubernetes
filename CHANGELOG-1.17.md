<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.17.0-alpha.1](#v1170-alpha1)
  - [Downloads for v1.17.0-alpha.1](#downloads-for-v1170-alpha1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.16.0](#changelog-since-v1160)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.17.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.17.0-alpha.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes.tar.gz) | `40985964b5f4b1e1eb448a8ca61ae5fe05b76cf4e97a4a6b0df0f7933071239ed8c3a6753d8ed8ba0c963694c0f94cce2b5976ddcc0386018cdc66337d80d006`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-src.tar.gz) | `475dfeb8544804dcc206f2284205fb1ee0bcb73169419be5e548ff91ffe6a35cea7e94039af562baee15933bef3afaa7ff10185e40926c7baa60d5936bcc9c1b`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `3f894661ed9b6ed3e75b6882e6c3e4858325f3b7c5c83cb8f7f632a8c8f30dd96a7dd277e4676a8a2ab598fe68da6473f414b494c63bfb4ed386a20dad7ae11a`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `f3070d79b0835fdc0791bbc31a334d8b46bf1bbb02f389c871b31063417598d17dd464532df420f2fa0dbbbb9f8cc0730a7ca4e19af09f873e0777d1e296f20c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `64e8961fa32a18a780e40b753772c6794c90a6dd5834388fd67564bb36f5301ea82377893f51e7c7c7247f91ca813e59f5f293522a166341339c2e5d34ac3f28`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `d7ba0f5f4c879d8dcd4404a7c18768190f82985425ab394ddc832ee71c407d0ac517181a24fd5ca2ebfd948c6fa63d095a43c30cf195c9b9637e1a762a2d8d2f`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `36fc47ee9530ee8a89d64d4be6b78b09831d0838a01b63d2a824a9e7dd0c2127ef1b49f539d16ba1248fbf40a7eb507b968b18c59080e7b80a7a573138218e36`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `a0a8fba0f4424f0a1cb7bad21244f47f98ba717165eaa49558c2612e1949a1b34027e23ccbd44959b391b6d9f82046c5bc07eb7d773603b678bbc0e5bf54502c`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `eaae9ed0cc8c17f27cff31d92c95c11343b9f383de27e335c83bfdf236e6da6ab55a9d89b3e0b087be159d6b64de21827ca19c861ecfb6471b394ea3720bcb61`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `994cf2dc42d20d36956a51b98dde31a00eae3bd853f7be4fbc32f48fec7b323a47ea5d841f31d2ca41036d27fbfaa3be4f2286654711245accf01c3be81f540c`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `68ebe4abea5a174eb189caea567e24e87cca57e7fbc9f8ec344aafbaf48c892d52d179fef67f9825be0eb93f5577f7573873b946e688de78c442c798a5b426bc`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `f29cd3caf5b40622366eae87e8abb47bea507f275257279587b507a00a858de87bcfa56894ae8cd6ba754688fd5cdf093ce6c4e0d0fd1e21ca487a3a8a9fd9f9`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `93e560e8572c6a593583d20a35185b93d04c950e6b1980a7b40ca5798958d184724ddebd1fa9377cfe87be4d11169bdba2a9f7fa192690f9edae04779aaf93a4`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `fe2af93336280e1251f97afecbdfb7416fd9dd7d08b3e5539abeea8ccaf7114cac399e832fa52359d2bc63ec9f8703ae3bca340db85f9b764816f4c36e4eefee`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `efc32c8477efda554d8e82d6e52728f58e706d3d53d1072099b3297c310632e8adece6030427154287d5525e74158c0b44a33421b3dd0ffb57372d63768e82ec`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `bda4fce6f5be7d0102ff265e0ba10e4dab776caeba1cebdf57db9891a34b4974fa57ac014aa6eca2dcfc1b64e9f67c8168e18026ae30c72ba61205d180f6e8ff`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `655c7157176f4f972c80877d73b0e390aaff455a5dcd046b469eb0f07d18ea1aaef447f90127be699e74518072ea1605652798fa431eb6ac7ee4e4fd85676362`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `1ec25c0350973ed06f657f2b613eb07308a9a4f0af7e72ebc5730c3c8d39ce3139a567acc7a224bebbe4e3496633b6053082b7172e2ce76b228c4b697f03f3d1`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `c65ac3db834596bcb9e81ffa5b94493244385073a232e9f7853759bce2b96a8199f79081d2f00a1b5502d53dc1e82a89afa97ffdb83994f67ebc261de9fb62b9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `0de8af66269db1ef7513f92811ec52a780abb3c9c49c0a4de9337eb987119bb583d03327c55353b4375d233e1a07a382cc91bdbf9477cf66e3f9e7fb0090499e`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `adb43c68cd5d1d52f254a14d80bb66667bfc8b367176ff2ed242184cf0b5accd3206bcbd42dec3f132bf1a230193812ae3e7a0c48f68634cb5f67538385e142a`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `1c834cfc06b9ba4a6da3bca2d504b734c935436546bc9304c7933e256dba849d665d34e82f48180f3975a907d37fec5ffb929923352ff63e1d3ff84143eea65b`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.17.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `6fc54fd17ebb65a6bd3d4efe93a713cc2aaea54599ddd3d73d01e93d6484087271b3ca65ed7a5861090356224140776a9606c10873b6b106bc9a6634c25b1677`

## Changelog since v1.16.0

### Action Required

* The deprecated feature gates `GCERegionalPersistentDisk`, `EnableAggregatedDiscoveryTimeout` and `PersistentLocalVolumes` are now unconditionally enabled and can no longer be specified in component invocations. ([#82472](https://github.com/kubernetes/kubernetes/pull/82472), [@draveness](https://github.com/draveness))
* ACTION REQUIRED: ([#81668](https://github.com/kubernetes/kubernetes/pull/81668), [@darshanime](https://github.com/darshanime))
    * Deprecate the default service IP CIDR. The previous default was `10.0.0.0/24` which will be removed in 6 months/2 releases. Cluster admins must specify their own desired value, by using `--service-cluster-ip-range` on kube-apiserver.
* Remove deprecated "include-uninitialized" flag. action required ([#80337](https://github.com/kubernetes/kubernetes/pull/80337), [@draveness](https://github.com/draveness))

### Other notable changes

* Bump version of event-exporter to 0.3.1, to switch it to protobuf. ([#83396](https://github.com/kubernetes/kubernetes/pull/83396), [@loburm](https://github.com/loburm))
* kubeadm: use the --service-cluster-ip-range flag to init or use the ServiceSubnet field in the kubeadm config to pass a comma separated list of Service CIDRs.  ([#82473](https://github.com/kubernetes/kubernetes/pull/82473), [@Arvinderpal](https://github.com/Arvinderpal))
* Remove MaxPriority in the scheduler API, please use MaxNodeScore or MaxExtenderPriority instead. ([#83386](https://github.com/kubernetes/kubernetes/pull/83386), [@draveness](https://github.com/draveness))
* Fixes a goroutine leak in kube-apiserver when a request times out. ([#83333](https://github.com/kubernetes/kubernetes/pull/83333), [@lavalamp](https://github.com/lavalamp))
* Some scheduler extender API fields are moved from `pkg/scheduler/api` to `pkg/scheduler/apis/extender/v1`. ([#83262](https://github.com/kubernetes/kubernetes/pull/83262), [@Huang-Wei](https://github.com/Huang-Wei))
* Fix aggressive VM calls for Azure VMSS ([#83102](https://github.com/kubernetes/kubernetes/pull/83102), [@feiskyer](https://github.com/feiskyer))
* Update Azure load balancer to prevent orphaned public IP addresses ([#82890](https://github.com/kubernetes/kubernetes/pull/82890), [@chewong](https://github.com/chewong))
* Use online nodes instead of possible nodes when discovering available NUMA nodes ([#83196](https://github.com/kubernetes/kubernetes/pull/83196), [@zouyee](https://github.com/zouyee))
* Fix typos in `certificates.k8s.io/v1beta1` KeyUsage constant names: `UsageContentCommittment` becomes `UsageContentCommitment` and `UsageNetscapSGC` becomes `UsageNetscapeSGC`. ([#82511](https://github.com/kubernetes/kubernetes/pull/82511), [@abursavich](https://github.com/abursavich))
* Fixes the bug in informer-gen that it produces incorrect code if a type has nonNamespaced tag set. ([#80458](https://github.com/kubernetes/kubernetes/pull/80458), [@tatsuhiro-t](https://github.com/tatsuhiro-t))
* Update to go 1.12.10 ([#83139](https://github.com/kubernetes/kubernetes/pull/83139), [@cblecker](https://github.com/cblecker))
* Update crictl to v1.16.1. ([#82856](https://github.com/kubernetes/kubernetes/pull/82856), [@Random-Liu](https://github.com/Random-Liu))
* Reduces the number of calls made to the Azure API when requesting the instance view of a virtual machine scale set node. ([#82496](https://github.com/kubernetes/kubernetes/pull/82496), [@hasheddan](https://github.com/hasheddan))
* Consolidate ScoreWithNormalizePlugin into the ScorePlugin interface ([#83042](https://github.com/kubernetes/kubernetes/pull/83042), [@draveness](https://github.com/draveness))
* On AWS nodes with multiple network interfaces, kubelet should now more reliably report the same primary node IP. ([#80747](https://github.com/kubernetes/kubernetes/pull/80747), [@danwinship](https://github.com/danwinship))
* Fixes kube-proxy bug accessing self nodeip:port on windows ([#83027](https://github.com/kubernetes/kubernetes/pull/83027), [@liggitt](https://github.com/liggitt))
* Resolves bottleneck in internal API server communication that can cause increased goroutines and degrade API Server performance ([#80465](https://github.com/kubernetes/kubernetes/pull/80465), [@answer1991](https://github.com/answer1991))
* The deprecated mondo `kubernetes-test` tarball is no longer built. Users running Kubernetes e2e tests should use the `kubernetes-test-portable` and `kubernetes-test-{OS}-{ARCH}` tarballs instead. ([#83093](https://github.com/kubernetes/kubernetes/pull/83093), [@ixdy](https://github.com/ixdy))
* Improved performance of kube-proxy with EndpointSlice enabled with more efficient sorting.  ([#83035](https://github.com/kubernetes/kubernetes/pull/83035), [@robscott](https://github.com/robscott))
* New APIs to allow adding/removing pods from pre-calculated prefilter state in the scheduling framework ([#82912](https://github.com/kubernetes/kubernetes/pull/82912), [@ahg-g](https://github.com/ahg-g))
* Conformance tests may now include disruptive tests. If you are running tests against a live cluster, consider skipping those tests tagged as `Disruptive` to avoid non-test workloads being impacted. Be aware, skipping any conformance tests (even disruptive ones) will make the results ineligible for consideration for the CNCF Certified Kubernetes program. ([#82664](https://github.com/kubernetes/kubernetes/pull/82664), [@johnSchnake](https://github.com/johnSchnake))
* Resolves regression generating informers for packages whose names contain `.` characters ([#82410](https://github.com/kubernetes/kubernetes/pull/82410), [@nikhita](https://github.com/nikhita))
* Added metrics 'authentication_latency_seconds' that can be used to understand the latency of authentication. ([#82409](https://github.com/kubernetes/kubernetes/pull/82409), [@RainbowMango](https://github.com/RainbowMango))
* kube-dns add-on:  ([#82347](https://github.com/kubernetes/kubernetes/pull/82347), [@pjbgf](https://github.com/pjbgf))
    * - All containers are now being executed under more restrictive privileges. 
    * - Most of the containers now run as non-root user and has the root filesystem set as read-only. 
    * - The remaining container running as root only has the minimum Linux capabilities it requires to run. 
    * - Privilege escalation has been disabled for all containers.
* k8s dockerconfigjson secrets are now compatible with docker config desktop authentication credentials files ([#82148](https://github.com/kubernetes/kubernetes/pull/82148), [@bbourbie](https://github.com/bbourbie))
* Use ipv4 in wincat port forward. ([#83036](https://github.com/kubernetes/kubernetes/pull/83036), [@liyanhui1228](https://github.com/liyanhui1228))
* Added Clone method to the scheduling framework's PluginContext and ContextData. ([#82951](https://github.com/kubernetes/kubernetes/pull/82951), [@ahg-g](https://github.com/ahg-g))
* Bump metrics-server to v0.3.5 ([#83015](https://github.com/kubernetes/kubernetes/pull/83015), [@olagacek](https://github.com/olagacek))
* dashboard: disable the dashboard Deployment on non-Linux nodes. This step is required to support Windows worker nodes. ([#82975](https://github.com/kubernetes/kubernetes/pull/82975), [@wawa0210](https://github.com/wawa0210))
* Fix possible fd leak and closing of dirs when using openstack ([#82873](https://github.com/kubernetes/kubernetes/pull/82873), [@odinuge](https://github.com/odinuge))
* PersistentVolumeLabel admission plugin, responsible for labeling `PersistentVolumes` with topology labels, now does not overwrite existing labels on PVs that were dynamically provisioned. It trusts the  dynamic provisioning that it provided the correct labels to the `PersistentVolume`, saving one potentially expensive cloud API call. `PersistentVolumes` created manually by users are labelled by the admission plugin in the same way as before. ([#82830](https://github.com/kubernetes/kubernetes/pull/82830), [@jsafrane](https://github.com/jsafrane))
* Fixes a panic in kube-controller-manager cleaning up bootstrap tokens ([#82887](https://github.com/kubernetes/kubernetes/pull/82887), [@tedyu](https://github.com/tedyu))
* Fixed a scheduler panic when using PodAffinity. ([#82841](https://github.com/kubernetes/kubernetes/pull/82841), [@Huang-Wei](https://github.com/Huang-Wei))
* Modified the scheduling framework's Filter API. ([#82842](https://github.com/kubernetes/kubernetes/pull/82842), [@ahg-g](https://github.com/ahg-g))
* Fix panic in kubelet when running IPv4/IPv6 dual-stack mode with a CNI plugin ([#82508](https://github.com/kubernetes/kubernetes/pull/82508), [@aanm](https://github.com/aanm))
* Kubernetes no longer monitors firewalld. On systems using firewalld for firewall ([#81517](https://github.com/kubernetes/kubernetes/pull/81517), [@danwinship](https://github.com/danwinship))
    * maintenance, kube-proxy will take slightly longer to recover from disruptive
    * firewalld operations that delete kube-proxy's iptables rules.
* Added cloud operation count metrics to azure cloud controller manager. ([#82574](https://github.com/kubernetes/kubernetes/pull/82574), [@kkmsft](https://github.com/kkmsft))
* Report non-confusing error for negative storage size in PVC spec. ([#82759](https://github.com/kubernetes/kubernetes/pull/82759), [@sttts](https://github.com/sttts))
* When registering with a 1.17+ API server, MutatingWebhookConfiguration and ValidatingWebhookConfiguration objects can now request that only `v1` AdmissionReview requests be sent to them. Previously, webhooks were required to support receiving `v1beta1` AdmissionReview requests as well for compatibility with API servers <= 1.15. ([#82707](https://github.com/kubernetes/kubernetes/pull/82707), [@liggitt](https://github.com/liggitt))
        * When registering with a 1.17+ API server, a CustomResourceDefinition conversion webhook can now request that only `v1` ConversionReview requests be sent to them. Previously, conversion webhooks were required to support receiving `v1beta1` ConversionReview requests as well for compatibility with API servers <= 1.15.
* Resolves issue with /readyz and /livez not including etcd and kms health checks ([#82713](https://github.com/kubernetes/kubernetes/pull/82713), [@logicalhan](https://github.com/logicalhan))
* fix: azure disk detach failure if node not exists ([#82640](https://github.com/kubernetes/kubernetes/pull/82640), [@andyzhangx](https://github.com/andyzhangx))
* Single static pod files and pod files from http endpoints cannot be larger than 10 MB. HTTP probe payloads are now truncated to 10KB. ([#82669](https://github.com/kubernetes/kubernetes/pull/82669), [@rphillips](https://github.com/rphillips))
* Restores compatibility with <=1.15.x custom resources by not publishing OpenAPI for non-structural custom resource definitions ([#82653](https://github.com/kubernetes/kubernetes/pull/82653), [@liggitt](https://github.com/liggitt))
* Take the context as the first argument of Schedule. ([#82119](https://github.com/kubernetes/kubernetes/pull/82119), [@wgliang](https://github.com/wgliang))
* Fixes regression in logging spurious stack traces when proxied connections are closed by the backend ([#82588](https://github.com/kubernetes/kubernetes/pull/82588), [@liggitt](https://github.com/liggitt))
* Correct a reference to a not/no longer used kustomize subcommand in the documentation ([#82535](https://github.com/kubernetes/kubernetes/pull/82535), [@demobox](https://github.com/demobox))
* Limit the body length of exec readiness/liveness probes. remote CRIs and Docker shim read a max of 16MB output of which the exec probe itself inspects 10kb. ([#82514](https://github.com/kubernetes/kubernetes/pull/82514), [@dims](https://github.com/dims))
* fixed an issue that the correct PluginConfig.Args is not passed to the corresponding PluginFactory in kube-scheduler when multiple PluginConfig items are defined. ([#82483](https://github.com/kubernetes/kubernetes/pull/82483), [@everpeace](https://github.com/everpeace))
* Adding TerminationGracePeriodSeconds to the test framework API ([#82170](https://github.com/kubernetes/kubernetes/pull/82170), [@vivekbagade](https://github.com/vivekbagade))
* /test/e2e/framework: Adds a flag "non-blocking-taints" which allows tests to run in environments with tainted nodes. String value should be a comma-separated list. ([#81043](https://github.com/kubernetes/kubernetes/pull/81043), [@johnSchnake](https://github.com/johnSchnake))

