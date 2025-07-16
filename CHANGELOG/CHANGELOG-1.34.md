<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.34.0-beta.0](#v1340-beta0)
  - [Downloads for v1.34.0-beta.0](#downloads-for-v1340-beta0)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.34.0-alpha.3](#changelog-since-v1340-alpha3)
  - [Changes by Kind](#changes-by-kind)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.34.0-alpha.3](#v1340-alpha3)
  - [Downloads for v1.34.0-alpha.3](#downloads-for-v1340-alpha3)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.34.0-alpha.2](#changelog-since-v1340-alpha2)
  - [Changes by Kind](#changes-by-kind-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.34.0-alpha.2](#v1340-alpha2)
  - [Downloads for v1.34.0-alpha.2](#downloads-for-v1340-alpha2)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.34.0-alpha.1](#changelog-since-v1340-alpha1)
  - [Changes by Kind](#changes-by-kind-2)
    - [Deprecation](#deprecation)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.34.0-alpha.1](#v1340-alpha1)
  - [Downloads for v1.34.0-alpha.1](#downloads-for-v1340-alpha1)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.33.0](#changelog-since-v1330)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-3)
    - [Feature](#feature-3)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)

<!-- END MUNGE: GENERATED_TOC -->

# v1.34.0-beta.0


## Downloads for v1.34.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes.tar.gz) | e1a1cf79f95354bae349afa992f72cf8cb23aa9a016f67599de1f0c31572a00cd84f541163d0da3205ecfe421901a88dc2c9012cec91d45fa2f094d524059f92
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-src.tar.gz) | 7e2c9837dd9be43df835d999024d516d52d211ee7e65f995da8e6c45442c8a8b6e5bc3e13a9279fc401c58b3ad1ba2b0b37abba3719e0605dfb5cb5c752d7df7

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | 1a3944812f26c37de6418f84d14e97366a1d2e268d8d61619f98f92778f3f3a9e30e4fd092ea0963ee19524284815803511e3d143c9f1b7df77f06728eddcefd
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | 01bcf3e380e9b18e7db316c0a7968b9293ff0cee6bd6395f8b3a8fcfbd9bc660b3016cfa636498c28d35a0e8a221f56303bd34b136d044df2356f3085aa4e613
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-linux-386.tar.gz) | 847526c7c2d2559f16ad1f6172d07590b4f35051a7bcf741c98067ace09fc92c52241f74a8c1d7ad1f4b713b26d8abc7059b47d97f4a8d9afc87d465b837dfd4
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | 260d78b743af5e7a6563cf26df7a4a4e75987f1bce96de3cec020d47f1a2586a39f3058cc1668a0b77266bb131490c74c55eaf669766918c8379e3c9818abebe
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-linux-arm.tar.gz) | f4dcc3597f2e005b51c4f3fc8323e119582fd00626ddaea6f2602810fd64fb65d1c1a795519d458b2c74ef5bd52467e6cd77b01972e858bb97d12f4ef2c81839
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | 4cc18be405d27f797ccd93b2f3ae0fe985450a0cf6f35e023c91e4a116b8443e32ba99e07bbc93c8dc4d9739c5adbb888cbc16ba457e362975e907057d0f38c1
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | 06eca6eb5dc82304566fc7194f1ae6f002a70dd031357608bbf65e9449840dcb55b37b1c61ff13e40f0eb95a0456bb6e5d692b14f806cd7e694ef71cb720bfb1
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | ed6db8acb534c557e3619628b78c1de5abcb31bda04e418296acc4fde54e23bba1ee42b4db9daefdf5622b09e3c9d4916461b85da10058d822251ac3da2eebca
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-windows-386.tar.gz) | 0302f1dea8c321f254b9aeb87882c82b28a4be74b4718f73840769e06c21a4a240d285ec89d94657522e49bd7550eda44a8e7312d83198c4b4f60990609beaae
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | 28dcf914521f31ed11d258fe1ff516eac9f7e1ed317bc55a816a2bca2ef41ce18140c296ea0c22e1a3808f82979ce8970e91951a982c33dd18e3fedb840ca4ad
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | ed50434e96f2fd80abaf3b9fa6befa96f829c086ac6b87d0d9f6ce9d6d3e10a22eb17928902b42b95ad4709a936e791d189b338af46fbe91d5391fde7c1f2904

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | 2862b8ed25f52542558fe48a6a584b02644a731decec878cfa0cee00173476f354d70a04efb84d084b87fe303291092d06e795e42e13a40c339699982a90044a
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | 1c00a6559f4f6c6190fe2265fb88cad4ac448eb3223dbd809976e3c85139d04b9cc02b4a9b80e9b42a2e4ee4a7a03a7a303ced49bc9673bff7be7cde7bb5f7a5
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | 7a998922d3fff36914ee690a5937d7b592f1916f68f7a31311065b25e7035cd38572df062e90680d56299b93be278c2fa24a370547270c07add274cf4a420d2f
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | 555b5690e99d0470ea7ca1bc4aebfda68a1126859962876db897b3024d5d7e352a3beeae4f2f3cba28a0d1b3c6edcf7094395492ff36fbc7d2d7a1e87ebb5fca

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 4b029d2f1022c4fd84ad1afaeeff9ae4fd80593c90f3f30a633df04bde68fac182c72bd906575b779eff01cc2e7d18884d9b5b0a3259a02e3131976a4339d1e1
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | c65b44be119997d321d13d6f9d08e42b1576fb9515cbf646c730f72e4e80a47afa1e59ea55cf8a8de1aa93a9db586ecb7101b2f59633460f4a4381ded987051b
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 837442a3311c2382b417e2d8cbf9638f9abc22f8584519becd44e9a161ef2cecee686a76977391f2c20b0477d5417d657ec29b9f0ab81e059a64f9566065f37b
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | 3c8232cd07d8869258cc4a7793fee524ec26847d32c4c6efe966946b81df6e36450acbfcbe199296b2ad79201875d00e7a8af8ceacc2c9681fdae9b4a11c2c0e
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 768c4cd582f4b708451d5f3fdacf048de7550251e468a9e255f1c5180602d7abca5f86f22a16089309e35c0f5eee18c9133cebe24830461e3471bc180efc3769

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.34.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.34.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.34.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.34.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.34.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.34.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.34.0-alpha.3

## Changes by Kind

### API Change

- Added `tokenAttributes.cacheType` field to v1 credential provider config. This field is required to be set to either ServiceAccount or Token when configuring a provider that uses service account to fetch registry credentials. ([#132617](https://github.com/kubernetes/kubernetes/pull/132617), [@aramase](https://github.com/aramase)) [SIG Auth, Node and Testing]
- JWT authenticators specified via the `AuthenticationConfiguration.jwt` array can now optionally specify either the `controlplane` or `cluster` egress selector by setting the `issuer.egressSelectorType` field.  When unset, the prior behavior of using no egress selector is retained.  The StructuredAuthenticationConfigurationEgressSelector beta feature (default on) must be enabled to use this functionality. ([#132768](https://github.com/kubernetes/kubernetes/pull/132768), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Promoted the `KubeletTracing` feature gate to GA. ([#132341](https://github.com/kubernetes/kubernetes/pull/132341), [@dashpole](https://github.com/dashpole)) [SIG Instrumentation and Node]
- Replaces boolPtrFn helper functions with the "k8s.io/utils/ptr" implementation. ([#132907](https://github.com/kubernetes/kubernetes/pull/132907), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Architecture]
- Simplied validation error message for invalid fields by removing redundant field name. ([#132513](https://github.com/kubernetes/kubernetes/pull/132513), [@xiaoweim](https://github.com/xiaoweim)) [SIG API Machinery, Apps, Auth, Node and Scheduling]
- The `AuthorizeWithSelectors` and `AuthorizeNodeWithSelectors` feature gates are promoted to stable and locked on. ([#132656](https://github.com/kubernetes/kubernetes/pull/132656), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth and Testing]

### Feature

- Add DetectCacheInconsistency feature gate that allows apiserver to periodically compare consistency between cache and etcd. Inconsistency is reported to `apiserver_storage_consistency_checks_total` metric and results in cache snapshots being purged. ([#132884](https://github.com/kubernetes/kubernetes/pull/132884), [@serathius](https://github.com/serathius)) [SIG API Machinery, Instrumentation and Testing]
- Add SizeBasedListCostEstimate feature gate, enabled by default, changing method of assigning APF seats to LIST request. Assign one seat per 100KB of data loaded to memory at once to handle LIST request. ([#132932](https://github.com/kubernetes/kubernetes/pull/132932), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Add warning on use of alpha metrics with emulated versions. ([#132276](https://github.com/kubernetes/kubernetes/pull/132276), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Architecture]
- Compact snapshots in watch cache based on etcd compaction ([#132876](https://github.com/kubernetes/kubernetes/pull/132876), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- Graduate `ConsistentListFromCache` to GA ([#132645](https://github.com/kubernetes/kubernetes/pull/132645), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Kubeadm: started using a named port 'probe-port' for all probes in the static pod manifests for kube-apiserver, kube-controller-manager, kube-scheduler and etc. If you have previously patched the port values in probes with kubeadm patches, you must now also patch the named port value in the pod container under 'ports'. ([#132776](https://github.com/kubernetes/kubernetes/pull/132776), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes is now built using Go 1.24.5 ([#132896](https://github.com/kubernetes/kubernetes/pull/132896), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- New PreBindPreFlight function is added to PreBindPlugin interface. In-tree PreBind plugins now implement PreBindPreFlight function. ([#132391](https://github.com/kubernetes/kubernetes/pull/132391), [@sanposhiho](https://github.com/sanposhiho)) [SIG Node, Scheduling, Storage and Testing]
- Prioritize resize requests by priorityClass and qos class when there is not enough room on the node to accept all the resize requests. ([#132342](https://github.com/kubernetes/kubernetes/pull/132342), [@natasha41575](https://github.com/natasha41575)) [SIG Node and Testing]
- Promote Ordered Namespace Deletion to Conformance ([#132219](https://github.com/kubernetes/kubernetes/pull/132219), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, Architecture and Testing]

### Bug or Regression

- CLI: `kubectl get job` now displays the SuccessCriteriaMet status for the listed jobs. ([#132832](https://github.com/kubernetes/kubernetes/pull/132832), [@Goend](https://github.com/Goend)) [SIG Apps and CLI]
- Change the node-local podresources API endpoint to only consider of active pods. Because this fix changes a long-established behavior, users observing a regressions can use the KubeletPodResourcesListUseActivePods feature gate (default on) to restore the old behavior. Please file an issue if you encounter problems and have to use the Feature Gate. ([#132028](https://github.com/kubernetes/kubernetes/pull/132028), [@ffromani](https://github.com/ffromani)) [SIG Node and Testing]
- Fix kubelet token cache returning stale tokens when service accounts are recreated with the same name. The token cache is now UID-aware and the new `TokenRequestServiceAccountUIDValidation` feature gate (Beta, enabled by default) validates the TokenRequest UID when set matches the service account UID. ([#132803](https://github.com/kubernetes/kubernetes/pull/132803), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth, Node and Testing]
- Fixed a bug that caused duplicate validation when updating a DaemonSet. ([#132548](https://github.com/kubernetes/kubernetes/pull/132548), [@gavinkflam](https://github.com/gavinkflam)) [SIG Apps]
- Kube-proxy nftables now reject/drop traffic to service with no endpoints from filter chains at priority 0 (NF_IP_PRI_FILTER) ([#132456](https://github.com/kubernetes/kubernetes/pull/132456), [@aroradaman](https://github.com/aroradaman)) [SIG Network]
- When both InPlacePodVerticalScaling and PodObservedGenerationTracking feature gates are set, fix the `observedGeneration` field exposed in the pod resize conditions to more accurately reflect which pod generation is associated with the condition. ([#131157](https://github.com/kubernetes/kubernetes/pull/131157), [@natasha41575](https://github.com/natasha41575)) [SIG Node]
- Windows kube-proxy: ensures that Windows kube-proxy aligns with Linux behavior and correctly honors the EndpointSlice-provided port for internal traffic routing. ([#132647](https://github.com/kubernetes/kubernetes/pull/132647), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]

### Other (Cleanup or Flake)

- Kubeadm: instead of passing the etcd flag --experimental-initial-corrupt-check, set the InitialCorruptCheck=true etcd feature gate, and instead of passing the --experimental-watch-progress-notify-interval flag, pass its graduated variant --watch-progress-notify-interval. ([#132838](https://github.com/kubernetes/kubernetes/pull/132838), [@AwesomePatrol](https://github.com/AwesomePatrol)) [SIG Cluster Lifecycle]
- Masked off access to Linux thermal interrupt info in `/proc` and `/sys`. ([#131018](https://github.com/kubernetes/kubernetes/pull/131018), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- NONW ([#132890](https://github.com/kubernetes/kubernetes/pull/132890), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- Promoted two EndpointSlice tests to conformance, to require that service
  proxy implementations are based on EndpointSlices rather than Endpoints. ([#132019](https://github.com/kubernetes/kubernetes/pull/132019), [@danwinship](https://github.com/danwinship)) [SIG Architecture, Network and Testing]
- Reduced excessive logging from the volume binding scheduler plugin by lowering verbosity of high-frequency messages from V(4) to V(5). ([#132840](https://github.com/kubernetes/kubernetes/pull/132840), [@ppmechlinski](https://github.com/ppmechlinski)) [SIG Autoscaling, Scheduling and Storage]

## Dependencies

### Added
- sigs.k8s.io/structured-merge-diff/v6: v6.2.0

### Changed
- k8s.io/kube-openapi: d90c4fd → f3f2b99

### Removed
- sigs.k8s.io/structured-merge-diff/v4: v4.7.0



# v1.34.0-alpha.3


## Downloads for v1.34.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes.tar.gz) | ec76c311b4aa0bcc97d4a83e6586a14081129343721bf844f0907ec2e14cad1ba4d0db04b667de963043c0fd4b410f7fe90788c10f070fa3b8ad0aa340e2dc5f
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-src.tar.gz) | b99cf04b86438285c24872e6ec2fdc03998a95b88e502e457ea03fba01beb870ef34e57055f7a14a016ae102906a1ed32ea20ddada31c9c1fa467c47b203d1f9

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | 5f2b298b4f1c27e06e79258b2ac840a36f70d46bd95b776d01bee89c1821b8a1138556224d4c23b8e582ca1676e0125bda8ebc93e8db0a92ede240efa169b01f
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | 719d4d81d85cf7f73e6f461e17c1559768f4e084753d0b210603e920b2ee6d687350e7ba5ae0bfa160630c02159e2900936ebde03104050f2eb6906b24573694
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 07ec8bd3d5308431bb4cf17dc8937ad13b95a2aab35fa0389479776228cc0f47756d1791d2371a66fa8f045d1894ac6d0dec4e42a3f96e443ea961cd2e7477ee
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | 5c4613fa4b8de852147a24e7c80894f1588e93023cff4bfe58725e2b141f5417662bcf837272c41eeaf8a91382eca3e6015b27cca099e516ba2e08214521279a
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | 1204c3f108b5e83a081b31af13d9e3185f0ff3c9547213ecbd854293b89661f5060c2b10a2c73d8bb6d5099438287dedbcdf88f33de6bc95a060e2d634d80652
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | 763d07a3a3f69b42047686e81e6cc137f9ef4b7ab2f50cc7bfbb26b8a15e011549501893e5cb9b77de0008cc77631fd8f37d113c0f0ee6a17e6435da06c269a2
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | ad4f45f8402014da35a3ff1daa5cfaafedd2d7e579bff1a0a87f49647e2d8488eb2b791776de3e4d5ac25631f13ec9c0bc64e7e0edd9f9049619659b9ebf9305
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | aba923f458d8f8d4c27b0144f40cac186663884447a5a67c20d2fb59d0c9ebd83e8d7480555365562e97c88992d28ffdd20378ce18412465a0356b2c20fa5dff
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-windows-386.tar.gz) | b5b1a854e0298f2f401627ecec5bd61fad8ccb7f77a42b5c34e6c0dc8f4f5e7485d13525f2775b160b15277d591d46c1633acddea60dd2b20949794f0f80a4e5
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 75b60210f1e6994abc4da9fceb23290438be81c094e27b2025c96addde4cbc034348dde0d50098db22aeceed3e3d6dd855d8d740d36e0f16932ca4ae537542d0
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | f997fa3ba6081273b46e6a71a98fcee06c0df36e045fd43eb38454b28dcf3863e8e5f053cc14057f9cfd53267ae611477f5410a2305d56b7a60a88f4c0cb36bd

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 3313f4746bdfaf7bd86bd72d035c552ae800426f5546eb23b83bdb3178e378d3aa5c4a59bc2b5ce5d97755432879812a293f582ca1dec3733e95adc5a5c07524
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | add5d69f2d48656649d1712c476a9de99ef2fcf4473d982b78b834e5ad544cf947a9fc35324552cd38e3824ea96194a81d76bc99111bfc725b9aa9212da8e88d
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | f2038f7382e660e8c97c4efa05adcb3d785ccd550597a50aa9d98e04e9bf1f29b5ac0c5d3d686f01870d64949dc43cf83e176c718fddc20f84ed38bd44f8ba54
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 077708405b4b22ebeaee8feeddbe1134374008129cc0cc40830434fa762f549f5950c1c2f76b72ddfab2ccc972e3325cb360a13acdbe54729a1eae0324b60b08

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | 9c0a3e76311789bfbaa3d8e27c289e5b5ab142ab0269dd5922016d2e3e8be6ddffd60ba1a57d6deb2571e4826bec1aab81c98a973d633b263ac316275ee01251
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | 1289b5e39164eaac2acce143dbb389341966e06b8b0261eb0eb4eb774848dd34f35b78d22b56d1613d5a6801868881415af62aa15ff8b1cbedbec9cad0567591
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 235d060ddd4c3da0b58fdb64a6a25b03690b7eb9c2888201b86d269f29d2a4e70000cea9711dff472626e78b56e228b9ffd16ab89442c54991de44bc8c3d9344
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | 8e9ca919e77e0ff226e92c036ac35ab284aa07fdd2e01b01bb4352d04acd567f40edc4caf934092cb3c1b04ac4ea281ef6681b6589d9e43ea25ae038381e95ad
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | 421cf6c68c5e0603f67dd44ac0d4d02c6feb3e60e9608268cdc4b498718c99b2cef4842fb7df60feaa99dc18d650164a0fcc31fbf896653ea4b6d126e0683d14

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.34.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.34.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.34.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.34.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.34.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.34.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.34.0-alpha.2

## Changes by Kind

### API Change

- DRA: the v1alpha4 kubelet gRPC API (added in 1.31, superseded in 1.32) is no longer supported. DRA drivers using the helper package from Kubernetes  >= 1.32 use the v1beta1 API and continue to be supported. ([#132574](https://github.com/kubernetes/kubernetes/pull/132574), [@pohly](https://github.com/pohly)) [SIG Node]
- Deprecate StreamingConnectionIdleTimeout field of the kubelet config. ([#131992](https://github.com/kubernetes/kubernetes/pull/131992), [@lalitc375](https://github.com/lalitc375)) [SIG Node]
- Removed deprecated gogo protocol definitions from `k8s.io/cri-api` in favor of `google.golang.org/protobuf`. ([#128653](https://github.com/kubernetes/kubernetes/pull/128653), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery, Auth, Instrumentation, Node and Testing]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the apiextensions-apiserver apiextensions. ([#132723](https://github.com/kubernetes/kubernetes/pull/132723), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the component-base. ([#132754](https://github.com/kubernetes/kubernetes/pull/132754), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery, Architecture, Instrumentation and Scheduling]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the kube-aggregator apiregistration. ([#132701](https://github.com/kubernetes/kubernetes/pull/132701), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaces Boolean-pointer-helper functions with the "k8s.io/utils/ptr" implementations. ([#132794](https://github.com/kubernetes/kubernetes/pull/132794), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery, Auth, CLI, Node and Testing]
- Replaces deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the apiserver (1/2). ([#132751](https://github.com/kubernetes/kubernetes/pull/132751), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery and Auth]
- Simplied validation error message for required fields by removing redundant messages. ([#132472](https://github.com/kubernetes/kubernetes/pull/132472), [@xiaoweim](https://github.com/xiaoweim)) [SIG API Machinery, Apps, Architecture, Auth, Cloud Provider, Network, Node and Storage]

### Feature

- Add configurable flags to kube-apiserver for coordinated leader election. ([#132433](https://github.com/kubernetes/kubernetes/pull/132433), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Testing]
- Add support for --cpu, --memory flag to kubectl autoscale, start deprecating --cpu-precent. ([#129373](https://github.com/kubernetes/kubernetes/pull/129373), [@googs1025](https://github.com/googs1025)) [SIG CLI]
- Added SizeBasedListCostEstimate feature gate that allows apiserver to estimate sizes of objects to calculate cost of LIST requests ([#132355](https://github.com/kubernetes/kubernetes/pull/132355), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- DRA kubelet: the kubelet now also cleans up ResourceSlices in some additional failure scenarios (driver gets removed forcibly or crashes and does not restart). ([#132058](https://github.com/kubernetes/kubernetes/pull/132058), [@pohly](https://github.com/pohly)) [SIG Node and Testing]
- Graduate `StreamingCollectionEncodingToJSON` and `StreamingCollectionEncodingToProtobuf` to GA ([#132648](https://github.com/kubernetes/kubernetes/pull/132648), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Kubeadm: graduated the kubeadm specific feature gate WaitForAllControlPlaneComponents to GA. The feature gate is now locked to always enabled and on node initialization kubeadm will perform a health check for all control plane components and not only the kube-apiserver. ([#132594](https://github.com/kubernetes/kubernetes/pull/132594), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Static pods that reference API objects are now denied admission by the kubelet so that static pods would not be silently running even after the mirror pod creation fails. ([#131837](https://github.com/kubernetes/kubernetes/pull/131837), [@sreeram-venkitesh](https://github.com/sreeram-venkitesh)) [SIG Auth, Node and Testing]
- The new `dra_resource_claims_in_use` kubelet metrics informs about active ResourceClaims, overall and by driver. ([#131641](https://github.com/kubernetes/kubernetes/pull/131641), [@pohly](https://github.com/pohly)) [SIG Architecture, Instrumentation, Node and Testing]
- When `RelaxedServiceNameValidation` feature gate is enabled, the 
  names of new Services names are validation with `NameIsDNSLabel()`,
  relaxing the  pre-existing validation. ([#132339](https://github.com/kubernetes/kubernetes/pull/132339), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Apps, Network and Testing]

### Failing Test

- Fixed e2e test "[Driver: csi-hostpath] [Testpattern: Dynamic PV (filesystem volmode)] volumeLimits should support volume limits]" not to leak Pods and namespaces. ([#132674](https://github.com/kubernetes/kubernetes/pull/132674), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]

### Bug or Regression

- Add podSpec validation for create StatefulSet ([#131790](https://github.com/kubernetes/kubernetes/pull/131790), [@chengjoey](https://github.com/chengjoey)) [SIG Apps, Etcd and Testing]
- Clarify help message of --ignore-not-found flag. Support --ignore-not-found in `watch` operation. ([#132542](https://github.com/kubernetes/kubernetes/pull/132542), [@gemmahou](https://github.com/gemmahou)) [SIG CLI]
- DRA drivers: the resource slice controller sometimes didn't react properly when kubelet or someone else deleted a recently created ResourceSlice. It incorrectly assumed that the ResourceSlice still exists and didn't recreate it. ([#132683](https://github.com/kubernetes/kubernetes/pull/132683), [@pohly](https://github.com/pohly)) [SIG Apps, Node and Testing]
- Ensure objects are transformed prior to storage in SharedInformers if a transformer is provided and `WatchList` is activated ([#131799](https://github.com/kubernetes/kubernetes/pull/131799), [@valerian-roche](https://github.com/valerian-roche)) [SIG API Machinery]
- Fix validation for Job with suspend=true, and completions=0 to set the Complete condition. ([#132614](https://github.com/kubernetes/kubernetes/pull/132614), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- Fixed a bug that fails to create a replica set when a deployment name is too long. ([#132560](https://github.com/kubernetes/kubernetes/pull/132560), [@hdp617](https://github.com/hdp617)) [SIG API Machinery and Apps]
- Fixed the bug when swap related metrics were not available in `/metrics/resource` endpoint. ([#132065](https://github.com/kubernetes/kubernetes/pull/132065), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node and Testing]
- Fixed the problem of validation error when specifying resource requirements at the container level for a resource not supported at the pod level. It implicitly interpreted the pod-level value as 0. ([#132551](https://github.com/kubernetes/kubernetes/pull/132551), [@chao-liang](https://github.com/chao-liang)) [SIG Apps]
- HPA status now displays memory metrics using Ki ([#132351](https://github.com/kubernetes/kubernetes/pull/132351), [@googs1025](https://github.com/googs1025)) [SIG Apps and Autoscaling]
- Removed defunct `make vet` target, please use `make lint` instead ([#132509](https://github.com/kubernetes/kubernetes/pull/132509), [@yongruilin](https://github.com/yongruilin)) [SIG Testing]
- Statefulset now respects minReadySeconds ([#130909](https://github.com/kubernetes/kubernetes/pull/130909), [@Edwinhr716](https://github.com/Edwinhr716)) [SIG Apps]

### Other (Cleanup or Flake)

- Removed deprecated gogo protocol definitions from `k8s.io/externaljwt` in favor of `google.golang.org/protobuf`. ([#132772](https://github.com/kubernetes/kubernetes/pull/132772), [@saschagrunert](https://github.com/saschagrunert)) [SIG Auth]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for ./test/e2e and ./test/utils. ([#132763](https://github.com/kubernetes/kubernetes/pull/132763), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Autoscaling and Testing]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for ./test/e2e. ([#132764](https://github.com/kubernetes/kubernetes/pull/132764), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Auth, Network, Node, Storage and Testing]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for ./test/e2e. ([#132765](https://github.com/kubernetes/kubernetes/pull/132765), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery, Apps, CLI and Testing]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for ./test/integration ([#132762](https://github.com/kubernetes/kubernetes/pull/132762), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Testing]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for apiextensions apiservers validation tests. ([#132726](https://github.com/kubernetes/kubernetes/pull/132726), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for apiextensions-apiserver pkg/controller. ([#132724](https://github.com/kubernetes/kubernetes/pull/132724), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for apiextensions-apiserver pkg/registry. ([#132725](https://github.com/kubernetes/kubernetes/pull/132725), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for pkg/apis (1/2). ([#132778](https://github.com/kubernetes/kubernetes/pull/132778), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Apps and Network]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for pkg/apis (2/2). ([#132779](https://github.com/kubernetes/kubernetes/pull/132779), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Apps, Auth and Storage]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for pkg/controller (1/2). ([#132781](https://github.com/kubernetes/kubernetes/pull/132781), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery, Apps and Network]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for pkg/controller (2/2). ([#132784](https://github.com/kubernetes/kubernetes/pull/132784), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery, Apps, Network, Node and Storage]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for pod-security-admission tests. ([#132741](https://github.com/kubernetes/kubernetes/pull/132741), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Auth]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the apiextensions-apiservers integration tests. ([#132721](https://github.com/kubernetes/kubernetes/pull/132721), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the cli-runtime. ([#132750](https://github.com/kubernetes/kubernetes/pull/132750), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG CLI and Release]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the cloud-provider. ([#132720](https://github.com/kubernetes/kubernetes/pull/132720), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Cloud Provider and Network]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the components-helper of the apimachinery. ([#132413](https://github.com/kubernetes/kubernetes/pull/132413), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the controller-manager. ([#132753](https://github.com/kubernetes/kubernetes/pull/132753), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery and Cloud Provider]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the csr. ([#132699](https://github.com/kubernetes/kubernetes/pull/132699), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery and Auth]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the e2e_node. ([#132755](https://github.com/kubernetes/kubernetes/pull/132755), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Node and Testing]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the kubeapiserver. ([#132529](https://github.com/kubernetes/kubernetes/pull/132529), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery and Architecture]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the pkg/security and plugin/pkg. ([#132777](https://github.com/kubernetes/kubernetes/pull/132777), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Auth, Node and Release]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the pod-security-admission admissiontests. ([#132742](https://github.com/kubernetes/kubernetes/pull/132742), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Auth]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the pod-security-admission policy. ([#132743](https://github.com/kubernetes/kubernetes/pull/132743), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Auth]
- Replaced deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the reflector. ([#132698](https://github.com/kubernetes/kubernetes/pull/132698), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery]
- Replaces deprecated package 'k8s.io/utils/pointer' with 'k8s.io/utils/ptr' for the apiserver (2/2). ([#132752](https://github.com/kubernetes/kubernetes/pull/132752), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery and Auth]
- Replaces toPtr helper functions with the "k8s.io/utils/ptr" implementations. ([#132806](https://github.com/kubernetes/kubernetes/pull/132806), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Apps, Testing and Windows]
- Types: ClusterEvent, ActionType, EventResource, ClusterEventWithHint, QueueingHint and QueueingHintFn moved from pkg/scheduler/framework to k8s.io/kube-scheduler/framework. ([#132190](https://github.com/kubernetes/kubernetes/pull/132190), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Node, Scheduling, Storage and Testing]
- Types: Code and Status moved from pkg/scheduler/framework to staging repo.
  Users should update import path for these types from "k8s.io/kubernetes/pkg/scheduler/framework" to "k8s.io/kube-scheduler/framework" ([#132087](https://github.com/kubernetes/kubernetes/pull/132087), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Node, Scheduling, Storage and Testing]
- Update etcd version to v3.6.1 ([#132284](https://github.com/kubernetes/kubernetes/pull/132284), [@ArkaSaha30](https://github.com/ArkaSaha30)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]

## Dependencies

### Added
- go.yaml.in/yaml/v2: v2.4.2
- go.yaml.in/yaml/v3: v3.0.4

### Changed
- github.com/emicklei/go-restful/v3: [v3.11.0 → v3.12.2](https://github.com/emicklei/go-restful/compare/v3.11.0...v3.12.2)
- github.com/google/gnostic-models: [v0.6.9 → v0.7.0](https://github.com/google/gnostic-models/compare/v0.6.9...v0.7.0)
- k8s.io/kube-openapi: 8b98d1e → d90c4fd
- sigs.k8s.io/json: 9aa6b5e → cfa47c3
- sigs.k8s.io/yaml: v1.4.0 → v1.5.0

### Removed
_Nothing has changed._



# v1.34.0-alpha.2


## Downloads for v1.34.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes.tar.gz) | 566db0b881557117fd7038bb5f25c46c727d2cc6a7cf3de0afc720eeeecfce947ae0e1b5162173a3ebfb915cfcc2c05fe8ab61db4551ac882a2756ad333d6337
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-src.tar.gz) | 3ccccf95776d0639455cead6d74a04e1af8f244915c583213b70f688ffd0cb291752da48589134eac5392ff1f6fb5046803d1e35f70475bcf74ded85c587df49

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | 058f6b47787adabfbb191ef80888633cddf5e2e36be6bb113da7db2c239c2691ad5467d381b09ca78bf9c54397a7eb0d54f2025ba7314c504eee4537787982b1
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 1e22f2b5c699e991daa282aaa1475d37e1614e4d90022dadc205b64c988c5050005a2347d0e93c9b0804c0db1fd0eb1f8eb4f86a0811638ccd9324cda95265fa
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-linux-386.tar.gz) | dacff605a6be45b4844b5120e420aedeea422297de1c9d5b5bc5926cc730efdc13f9881c75cb346159cb8a4e0a4364070299ffcc41494dbdd8ece6f698238658
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 38f5c80ad4cf1c8e422d5ac54cf6e5ea93425bd4fe4dd8d9ac011734e2b187769f74da749240bea1cc3a850ea6530dcbc27979af8fc9d86b3ec3299362c54e03
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | e8e116b2603e961d6090da8755d61c895a5ec7e9b6bf0bfc52a6a2b45c2111c73f7c30496dfdc624778c9ce74aa116206c0b3adcc41b046d06d8301a55218679
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | ece12bddbae26f6d63e39482985a43429b768d82bc6c1b523724c134d98f52ae41c64f66d267d53400566bf0428021228c9cf9b0b663399ab27c08304bbe193f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 4346923ea8eae6e51c07fa53a6a6f72d75ca6a50db5ae255c9902f4bd7af0a1cde359f9d6c2a84253c74e4d32f32ee81abe8b1dfaabb0206f871c57ba1eacb73
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | e7f93440b0497bab07db1f5bf70be61148abe555a8aa83712201128056a0e53c0273a7269ba92c65af0095bb9e69d3bfe85359720969bca1399d21e0b04b1264
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 5a43893f34cac36608a7817c1116c43b71411ef75d71188886672941db7d8080efcb94e183b0beadc852b36b12986eb356bdde4c4a7729e284e214ba8cf43fea
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 52f3514824ab0a152eaf588722f56e6d12366ecf8479e1cd11f0e878ed7c9b0b5ec528cdb7dd0f03273eef704adeaa3cce3918e89bd7a4c15480130aa5c6b5f3
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 5f31dfc54626f31feff6373a7282cf624779a79b2178f0d7ff4e977652c5f8bc2b2c64de1b6db22eda9c563a4980b3b72b134ff2a1743a5b196ab3eeb6f5e452

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | b13750ef0157384cf353ef6a4471fd17706c3bf3bd7bed2c84efc57f8863f3c7306a09813d8788fcd97d0e7e0929f4c136e2ac047c30fef2c45d4fb3d0bbe8ff
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 84d004c4df9c46a280abfca2af02d5601d07fa8e1355b4ebfc2dcc069829804650a1097f97254c4f4ad0423b7f4828c76dfd2a56348aa1339f5518bbb9257c8e
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | f1cb4b333fc9bc696a3a75b4b0a846fe9f207c79fcfac438f2d3e3a709d23039c3f1507ea6e03dff2b5a4ef737052c613d354efe3b034384544ebf99551be7ea
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | ded6953d4d2b04f589a24f5e6e21aa3630d9a12f5562d8c8e6301660b1fa04782523500d74bb5399a8cb0d6102546bd1000591c3dd8464d98a3d3399576a20e7

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | e4e4e2ac9acb4d36aded75ee4e841947b8eec2e66b08d11b01b662c5372be51cd746b9a87248a03be45c49de9aa53a31904b38a3e1253f0aaecc5e5b774cb4d7
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 40318c61e6b060c18d2cd80143d83bab9178275f099e65b82161334eb9970ec9151b581654151799532564c92c4a3730abeb00a9e88ddcfddd66ed69d09c9921
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 778ca04559776b3e03537e13ff9f2136ae7fdb71c2a130e9734761cb2ad278f1af9e7285a132f2f77f49566f0c302ef8ca7f3694f84e48c93f5585236718cd8e
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | b6b0865359b6c767b233374918263d05cfd8fb52130b67ca7db2dcc01df119efbff89041a6310dbed77c96aa8faeb5a50bd1c184ccd9e5441eb09c1cb6df8e03
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 9374a3ec4ea2d417a63fef4809e982b9cc62f98ef67cd0bcabb7b673f1efaa82c66724bf605e658d0905cf9ee61f61f1ab00a077a330ee1e4197c06c582d9a37

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.34.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.34.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.34.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.34.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.34.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.34.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.34.0-alpha.1

## Changes by Kind

### Deprecation

- Apimachinery: deprecated MessageCountMap and CreateAggregateFromMessageCountMap ([#132376](https://github.com/kubernetes/kubernetes/pull/132376), [@tico88612](https://github.com/tico88612)) [SIG API Machinery]

### API Change

- Add a `runtime.ApplyConfiguration` interface that is implemented by all generated applyconfigs ([#132194](https://github.com/kubernetes/kubernetes/pull/132194), [@alvaroaleman](https://github.com/alvaroaleman)) [SIG API Machinery and Instrumentation]
- Added omitempty and opt tag to the API v1beta2 AdminAccess type in the DeviceRequestAllocationResult struct. ([#132338](https://github.com/kubernetes/kubernetes/pull/132338), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Auth]
- Introduces OpenAPI format support for `k8s-short-name` and `k8s-long-name`. ([#132504](https://github.com/kubernetes/kubernetes/pull/132504), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- Promoted Job Pod Replacement Policy to general availability. The `JobPodReplacementPolicy` feature gate is now locked to true, and will be removed in a future release of Kubernetes. ([#132173](https://github.com/kubernetes/kubernetes/pull/132173), [@dejanzele](https://github.com/dejanzele)) [SIG Apps and Testing]
- This PR corrects that documentation, making it clear to users that podSelector is optional and describes its default behavior. ([#131354](https://github.com/kubernetes/kubernetes/pull/131354), [@tomoish](https://github.com/tomoish)) [SIG Network]

### Feature

- Added a delay to node updates after kubelet startup. A random offset, based on the configured `nodeStatusReportFrequency`, helps spread the traffic and load (due to node status updates) more evenly over time. The initial status update can be up to 50% earlier or 50% later than the regular schedule. ([#130919](https://github.com/kubernetes/kubernetes/pull/130919), [@mengqiy](https://github.com/mengqiy)) [SIG Node]
- Included namespace in the output of the kubectl delete for clearer identification of resources. ([#126619](https://github.com/kubernetes/kubernetes/pull/126619), [@totegamma](https://github.com/totegamma)) [SIG CLI]
- Kube-apiserver: each unique set of etcd server overrides specified with `--etcd-servers-overrides` now surface health checks named `etcd-override-<index>` and `etcd-override-readiness-<index>`. These checks are still excluded by `?exclude=etcd` and `?exclude=etcd-readiness` directives. ([#129438](https://github.com/kubernetes/kubernetes/pull/129438), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery and Testing]

### Bug or Regression

- Fix regression introduced in 1.33  - where some Paginated LIST calls are falling back to etcd instead of serving from cache. ([#132244](https://github.com/kubernetes/kubernetes/pull/132244), [@hakuna-matatah](https://github.com/hakuna-matatah)) [SIG API Machinery]
- Fixed API response for StorageClassList queries and returns a graceful error message, if the provided ResourceVersion is too large. ([#132374](https://github.com/kubernetes/kubernetes/pull/132374), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG API Machinery and Etcd]
- Fixed an issue which allowed Custom Resources to be created with Server-Side Apply even when its CustomResourceDefinition was terminating. ([#132467](https://github.com/kubernetes/kubernetes/pull/132467), [@sdowell](https://github.com/sdowell)) [SIG API Machinery]
- Removed the deprecated flag '--wait-interval' for the ip6tables-legacy-restore binary. ([#132352](https://github.com/kubernetes/kubernetes/pull/132352), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Network]

### Other (Cleanup or Flake)

- Conntrack reconciler now considers service's target port during cleanup of stale flow entries. ([#130542](https://github.com/kubernetes/kubernetes/pull/130542), [@aroradaman](https://github.com/aroradaman)) [SIG Network]
- Job controller uses controller UID index for pod lookups. ([#132305](https://github.com/kubernetes/kubernetes/pull/132305), [@xigang](https://github.com/xigang)) [SIG Apps]
- Removed the deprecated `--register-schedulable` command line argument from the kubelet. ([#122384](https://github.com/kubernetes/kubernetes/pull/122384), [@carlory](https://github.com/carlory)) [SIG Cloud Provider, Node and Scalability]
- Removes the `kubernetes.io/initial-events-list-blueprint` annotation from the synthetic "Bookmark" event for the watch stream requests. ([#132326](https://github.com/kubernetes/kubernetes/pull/132326), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]

## Dependencies

### Added
- go.yaml.in/yaml/v2: v2.4.2
- go.yaml.in/yaml/v3: v3.0.3

### Changed
- k8s.io/kube-openapi: c8a335a → 8b98d1e
- sigs.k8s.io/yaml: v1.4.0 → v1.5.0

### Removed
_Nothing has changed._



# v1.34.0-alpha.1


## Downloads for v1.34.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes.tar.gz) | 4125206915e9f0cd7bffd77021f210901bade4747d84855c8210922c82e2085628a05b81cef137e347b16a05828f99ac2a27a8f8f19a14397011031454736ea0
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-src.tar.gz) | c1dfe0a1df556adcad5881a7960da5348feacc23894188b94eb75be0b156912ab8680b94e2579a96d9d71bff74b1c813b8592de6926fba8e5a030a88d8b4b208

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 22c4d1031297ea1833b3cd3e6805008c34b66f932ead3818db3eb2663a71510a8cdb53a05852991d54e354800ee97a2aad4afc31726d956f38c674929ce10778
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 6be320d2075d8a7835751c019556059ff2fca704d0bbeeff181248492d8ed6fcc2d6d6b68c509e4453431100b06a20268e61b9e434b638a78ebfad68e7c41276
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-386.tar.gz) | e63ac6b7127591068626a3d7caf0e1bae6390106f6c93efae34b18e38af257f1521635eb2adf76c40ad0f0d9a5397947bbb0215087d4d2e87ce6f253b6aec1a4
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 12dc8dc4997b71038c377bfd9869610110cebb20afcb051e85c86832f75bc8e7eabbb08b5caa00423c5f8df68210ad5ca140a61d4a8e9ad8640f648250205752
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 0a7f8df6abfe9971f778add6771135d7079c245b18dd941eacf1230f75f461e7d8302142584aa4d60062c8cfd4e021f21ae5aa428d82b5fbe3697bda0e5854ff
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | b1442640ac1e45268e9916d0c51e711b7640fd2594ecad05a0d990c19db2e0dcde53cc90fb13588a2b926e25c831f62bf5461fa9c8e6a03a83573cc1c3791903
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | e5a028da7fcb24aee85d010741c864fa4e5a3d6c87223b5c397686107a53dd2801a8c75cf9e1046ab28c97b06a5457aa6b3e4f809cd46cbe4858f78b2cb6a4df
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 4d3fce13d8f29e801c4d7355f83ded4d2e4abcc0b788f09d616ef7f89bd04e9d92d0b32e6e365118e618b32020d8b43e4cbd59a82262cc787b98f42e7df4ddbc
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 3bbe15f8856cab69c727b02766024e1bb430add8ad18216929a96d7731d255c5d5bb6b678a4d4e7a021f2e976633b69c0516c2260dcc0bee7d2447f64bd52fe8
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 1833d8b09d5524df91120115667f897df47ad66edb57d2570e022234794c4d0d09212fca9b0b64e21ccc8ce6dcd41080bf9198c81583949cb8001c749f25e8a0
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | c0819674e11923b38d2df7cb9955929247a5b0752c93fc5215300da3514c592348cbe649a5c6fd6ac63500c6d68cf61a2733c099788164547e3f7738afe78ecf

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | acd0b0b6723789780fd536894a965001056e94e92e2070edacdb53d2d879f56a90cc2c1ad0ff6d634ed74ef4debcefa01eee9f675cc4c70063da6cc52cc140d3
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 31321659424b4847ec456ae507486efe57c8e903c2bc450df65ffc3bc90011ba050e8351ab32133943dfebd9d6e8ad47f2546a7cdc47e424cdaf0dc7247e08c3
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | fe81aa313be46ed5cc91507e58bc165e98722921d33473c29d382dceb948b1ffc0437d74825277a7da487f9390dec64f6a70617b05e0441c106fa87af737b90c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 69a54f40e7a8684a6a1606f0463266d83af615f70a55d750031d82601c8070f4f9161048018c78e0859faa631ec9984fc20af3bc17240c8fc9394c6cbffacaf9

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 797a5df349e571330e8090bd78f024d659d0d46e8a7352210b80ac594ef50dc2f3866240b75f7c0d2e08fa526388d0dfdcb91b4686f01b547c860a2d0a9846a7
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 552a114facbd42c655574953186ba15a91c061b3db9ad25e665892c355347bf841e1bf716f8e28a16f1f1b37492911103212ec452bf5e663f8fcf26fae3ccc6a
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 7f08bad1921127fdceba7deb58d305e0b599de7ab588da936ff753ab4c6410b5db0634d71094e97ee1baeaccc491370c88268f6a540eedb556c90fb1ce350eda
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 4d1ac168b4591bf5ed7773d87eb47e64eb322adb6fd22b89f4f79c9849aee70188f0fa04a18775feff6f9baf95277499c56cd471a56240a87f9810c82434ba35
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 896e508aa1c0bb3249c01554aea0ea25d65c4d9740772f8c053ded411b89a34a1c1e954e62fad10a1366cb0a9534af9b3d4e0a46acd956b47eb801e900dfcbe6

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.33.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - For metrics `apiserver_cache_list_fetched_objects_total`, `apiserver_cache_list_returned_objects_total`, `apiserver_cache_list_total` replace `resource_prefix` label with API `group` and `resource` labels.
  For metrics `etcd_request_duration_seconds`, `etcd_requests_total` and `etcd_request_errors_total` replace `type` label with API `resource` and `group` label.
  For metric `apiserver_selfrequest_total` add a API `group` label.
  For metrics `apiserver_watch_events_sizes` and `apiserver_watch_events_total` replace API `kind` label with `resource` label.
  For metrics `apiserver_request_body_size_bytes`, `apiserver_storage_events_received_total`, `apiserver_storage_list_evaluated_objects_total`, `apiserver_storage_list_fetched_objects_total`, `apiserver_storage_list_returned_objects_total`, `apiserver_storage_list_total`, `apiserver_watch_cache_events_dispatched_total`, `apiserver_watch_cache_events_received_total`, `apiserver_watch_cache_initializations_total`, `apiserver_watch_cache_resource_version`, `watch_cache_capacity`, `apiserver_init_events_total`, `apiserver_terminated_watchers_total`, `watch_cache_capacity_increase_total`, `watch_cache_capacity_decrease_total`, `apiserver_watch_cache_read_wait_seconds`, `apiserver_watch_cache_consistent_read_total`, `apiserver_storage_consistency_checks_total`, `etcd_bookmark_counts`, `storage_decode_errors_total` extract the API group from `resource` label and put it in new `group` label. ([#131845](https://github.com/kubernetes/kubernetes/pull/131845), [@serathius](https://github.com/serathius)) [SIG API Machinery, Etcd, Instrumentation and Testing]
  - Kubelet:  removed the deprecated flag `--cloud-config` from the command line. ([#130161](https://github.com/kubernetes/kubernetes/pull/130161), [@carlory](https://github.com/carlory)) [SIG Cloud Provider, Node and Scalability]
  - Scheduling Framework exposes NodeInfos to the PreFilterPlugins.
  The PreFilterPlugins need to accept the NodeInfo list from the arguments. ([#130720](https://github.com/kubernetes/kubernetes/pull/130720), [@saintube](https://github.com/saintube)) [SIG Node, Scheduling, Storage and Testing]
 
## Changes by Kind

### Deprecation

- Deprecate preferences field in kubeconfig in favor of kuberc ([#131741](https://github.com/kubernetes/kubernetes/pull/131741), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI, Cluster Lifecycle and Testing]
- Kubeadm: consistently print an 'error: ' prefix before errors. ([#132080](https://github.com/kubernetes/kubernetes/pull/132080), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: only expose non-deprecated klog flags, 'v' and 'vmodule', to align with KEP https://features.k8s.io/2845 ([#131647](https://github.com/kubernetes/kubernetes/pull/131647), [@carsontham](https://github.com/carsontham)) [SIG Cluster Lifecycle]
- [cloud-provider] respect the "exclude-from-external-load-balancers=false" label ([#131085](https://github.com/kubernetes/kubernetes/pull/131085), [@kayrus](https://github.com/kayrus)) [SIG Cloud Provider and Network]

### API Change

- #### Additional documentation e.g., KEPs (Kubernetes Enhancement Proposals), usage docs, etc.:
  
  <!--
  This section can be blank if this pull request does not require a release note.
  
  When adding links which point to resources within git repositories, like
  KEPs or supporting documentation, please reference a specific commit and avoid
  linking directly to the master branch. This ensures that links reference a
  specific point in time, rather than a document that may change over time.
  
  See here for guidance on getting permanent links to files: https://help.github.com/en/articles/getting-permanent-links-to-files
  
  Please use the following format for linking documentation:
  - [KEP]: <link>
  - [Usage]: <link>
  - [Other doc]: <link>
  --> ([#131996](https://github.com/kubernetes/kubernetes/pull/131996), [@ritazh](https://github.com/ritazh)) [SIG Node and Testing]
- DRA API: resource.k8s.io/v1alpha3 now only contains DeviceTaintRule. All other types got removed because they became obsolete when introducing the v1beta1 API in 1.32.
  before updating a cluster where resourceclaims, resourceclaimtemplates, deviceclasses, or resourceslices might have been stored using Kubernetes < 1.32, delete all of those resources before updating and recreate them as needed while running Kubernetes >= 1.32. ([#132000](https://github.com/kubernetes/kubernetes/pull/132000), [@pohly](https://github.com/pohly)) [SIG Etcd, Node, Scheduling and Testing]
- Extends the nodeports scheduling plugin to consider hostPorts used by restartable init containers. ([#132040](https://github.com/kubernetes/kubernetes/pull/132040), [@avrittrohwer](https://github.com/avrittrohwer)) [SIG Scheduling and Testing]
- Kube-apiserver: Caching of authorization webhook decisions for authorized and unauthorized requests can now be disabled in the `--authorization-config` file by setting the new fields `cacheAuthorizedRequests` or `cacheUnauthorizedRequests` to `false` explicitly. See https://kubernetes.io/docs/reference/access-authn-authz/authorization/#using-configuration-file-for-authorization for more details. ([#129237](https://github.com/kubernetes/kubernetes/pull/129237), [@rfranzke](https://github.com/rfranzke)) [SIG API Machinery and Auth]
- Kube-apiserver: Promoted the `StructuredAuthenticationConfiguration` feature gate to GA. ([#131916](https://github.com/kubernetes/kubernetes/pull/131916), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-apiserver: the AuthenticationConfiguration type accepted in `--authentication-config` files has been promoted to `apiserver.config.k8s.io/v1`. ([#131752](https://github.com/kubernetes/kubernetes/pull/131752), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-log-runner: rotating log output into a new file when reaching a certain file size can be requested via the new `-log-file-size` parameter. `-log-file-age` enables automatical removal of old output files.  Periodic flushing can be requested through ` -flush-interval`. ([#127667](https://github.com/kubernetes/kubernetes/pull/127667), [@zylxjtu](https://github.com/zylxjtu)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Release, Scheduling, Storage, Testing and Windows]
- Kubectl: graduated `kuberc` support to beta. A `kuberc` configuration file provides a mechanism for customizing kubectl behavior (separate from kubeconfig, which configured cluster access across different clients). ([#131818](https://github.com/kubernetes/kubernetes/pull/131818), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Promote the RelaxedEnvironmentVariableValidation feature gate to GA and lock it in the default enabled state. ([#132054](https://github.com/kubernetes/kubernetes/pull/132054), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Apps, Architecture, Node and Testing]
- Remove inaccurate statement about requiring ports from pod spec hostNetwork field ([#130994](https://github.com/kubernetes/kubernetes/pull/130994), [@BenTheElder](https://github.com/BenTheElder)) [SIG Network and Node]
- TBD ([#131318](https://github.com/kubernetes/kubernetes/pull/131318), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Architecture, Auth, Etcd, Network and Testing]
- The validation of `replicas` field in the ReplicationController `/scale` subresource has been migrated to declarative validation.
  If the `DeclarativeValidation` feature gate is enabled, mismatches with existing validation are reported via metrics.
  If the `DeclarativeValidationTakeover` feature gate is enabled, declarative validation is the primary source of errors for migrated fields. ([#131664](https://github.com/kubernetes/kubernetes/pull/131664), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Apps]
- The validation-gen code generator generates validation code that supports validation ratcheting. ([#132236](https://github.com/kubernetes/kubernetes/pull/132236), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps, Auth and Node]
- Update etcd version to v3.6.0 ([#131501](https://github.com/kubernetes/kubernetes/pull/131501), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- When the IsDNS1123SubdomainWithUnderscore function returns an error, it will return the correct regex information dns1123SubdomainFmtWithUnderscore. ([#132034](https://github.com/kubernetes/kubernetes/pull/132034), [@ChosenFoam](https://github.com/ChosenFoam)) [SIG Network]
- Zero-value `metadata.creationTimestamp` values are now omitted and no longer serialize an explicit `null` in JSON, YAML, and CBOR output ([#130989](https://github.com/kubernetes/kubernetes/pull/130989), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling, Storage and Testing]

### Feature

- Add a flag to `kubectl version` that detects whether a client/server version mismatch is outside the officially supported range. ([#127365](https://github.com/kubernetes/kubernetes/pull/127365), [@omerap12](https://github.com/omerap12)) [SIG CLI]
- Add support for CEL expressions with escaped names in structured authentication config.  Using `[` for accessing claims or user data is preferred when names contain characters that would need to be escaped.  CEL optionals via `?` can be used in places where `has` cannot be used, i.e. `claims[?"kubernetes.io"]` or `user.extra[?"domain.io/foo"]`. ([#131574](https://github.com/kubernetes/kubernetes/pull/131574), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]
- Added Traffic Distribution field to `kubectl describe service` output ([#131491](https://github.com/kubernetes/kubernetes/pull/131491), [@tchap](https://github.com/tchap)) [SIG CLI]
- Added a `--show-swap` option to `kubectl top` subcommands ([#129458](https://github.com/kubernetes/kubernetes/pull/129458), [@iholder101](https://github.com/iholder101)) [SIG CLI]
- Added alpha metrics for compatibility versioning ([#131842](https://github.com/kubernetes/kubernetes/pull/131842), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Architecture, Instrumentation and Scheduling]
- Enabling completion for aliases defined in kuberc ([#131586](https://github.com/kubernetes/kubernetes/pull/131586), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Graduate ResilientWatchCacheInitialization to GA ([#131979](https://github.com/kubernetes/kubernetes/pull/131979), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Graduate configurable endpoints for anonymous authentication using the authentication configuration file to stable. ([#131654](https://github.com/kubernetes/kubernetes/pull/131654), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG API Machinery and Testing]
- Graduated relaxed DNS search string validation to GA. For the Pod API, `.spec.dnsConfig.searches`
  now allows an underscore (`_`) where a dash (`-`) would be allowed, and it allows search strings be a single dot `.`. ([#132036](https://github.com/kubernetes/kubernetes/pull/132036), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network and Testing]
- Graduated scheduler `QueueingHint` support to GA (general availability) ([#131973](https://github.com/kubernetes/kubernetes/pull/131973), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Kube-apiserver: Promoted `ExternalServiceAccountTokenSigner` feature to beta, which enables external signing of service account tokens and fetching of public verifying keys, by enabling the beta `ExternalServiceAccountTokenSigner` feature gate and specifying `--service-account-signing-endpoint`. The flag value can either be the location of a Unix domain socket on a filesystem, or be prefixed with an @ symbol and name a Unix domain socket in the abstract socket namespace. ([#131300](https://github.com/kubernetes/kubernetes/pull/131300), [@HarshalNeelkamal](https://github.com/HarshalNeelkamal)) [SIG API Machinery, Auth and Testing]
- Kube-controller-manager events to support contextual logging. ([#128351](https://github.com/kubernetes/kubernetes/pull/128351), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery]
- Kube-proxy: Check if IPv6 is available on Linux before using it ([#131265](https://github.com/kubernetes/kubernetes/pull/131265), [@rikatz](https://github.com/rikatz)) [SIG Network]
- Kubeadm: add support for ECDSA-P384 as an encryption algorithm type in v1beta4. ([#131677](https://github.com/kubernetes/kubernetes/pull/131677), [@lalitc375](https://github.com/lalitc375)) [SIG Cluster Lifecycle]
- Kubeadm: fixed issue where etcd member promotion fails with an error saying the member was already promoted ([#130782](https://github.com/kubernetes/kubernetes/pull/130782), [@BernardMC](https://github.com/BernardMC)) [SIG Cluster Lifecycle]
- Kubeadm: graduated the `NodeLocalCRISocket` feature gate to beta and enabed it by default. When its enabled, kubeadm will:
    1. Generate a `/var/lib/kubelet/instance-config.yaml` file to customize the `containerRuntimeEndpoint` field in per-node kubelet configurations.
    2. Remove the `kubeadm.alpha.kubernetes.io/cri-socket` annotation from nodes during upgrade operations.
    3. Remove the `--container-runtime-endpoint` flag from the `/var/lib/kubelet/kubeadm-flags.env` file during upgrades. ([#131981](https://github.com/kubernetes/kubernetes/pull/131981), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Kubeadm: switched the validation check for Linux kernel version to throw warnings instead of errors. ([#131919](https://github.com/kubernetes/kubernetes/pull/131919), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
- Kubelet: the `--image-credential-provider-config` flag previously only accepted an individual file, but can now specify a directory path as well; when a directory is specified, all .json/.yaml/.yml files in the directory are loaded and merged in lexicographical order. ([#131658](https://github.com/kubernetes/kubernetes/pull/131658), [@dims](https://github.com/dims)) [SIG Auth and Node]
- Kubernetes api-server now merges selectors built from matchLabelKeys into the labelSelector of topologySpreadConstraints, 
  aligning Pod Topology Spread with the approach used by Inter-Pod Affinity.
  
  To avoid breaking existing pods that use matchLabelKeys, the current scheduler behavior will be preserved until it is removed in v1.34. 
  Therefore, do not upgrade your scheduler directly from v1.32 to v1.34. 
  Instead, upgrade step-by-step (from v1.32 to v1.33, then to v1.34), 
  ensuring that any pods created at v1.32 with matchLabelKeys are either removed or already scheduled by the time you reach v1.34.
  
  If you maintain controllers that previously relied on matchLabelKeys (for instance, to simulate scheduling), 
  you likely no longer need to handle matchLabelKeys directly. Instead, you can just rely on the labelSelector field going forward.
  
  Additionally, a new feature gate `MatchLabelKeysInPodTopologySpreadSelectorMerge`, which is enabled by default, has been 
  added to control this behavior. ([#129874](https://github.com/kubernetes/kubernetes/pull/129874), [@mochizuki875](https://github.com/mochizuki875)) [SIG Apps, Node, Scheduling and Testing]
- Kubernetes is now built using Go 1.24.3 ([#131934](https://github.com/kubernetes/kubernetes/pull/131934), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built using Go 1.24.4 ([#132222](https://github.com/kubernetes/kubernetes/pull/132222), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- LeaseLocks can now have custom Labels that different holders will overwrite when they become the holder of the underlying lease. ([#131632](https://github.com/kubernetes/kubernetes/pull/131632), [@DerekFrank](https://github.com/DerekFrank)) [SIG API Machinery]
- Non-scheduling related errors (e.g., network errors) don't lengthen the Pod scheduling backoff time. ([#128748](https://github.com/kubernetes/kubernetes/pull/128748), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Promote feature OrderedNamespaceDeletion to GA. ([#131514](https://github.com/kubernetes/kubernetes/pull/131514), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]
- Removed "endpoint-controller" and "workload-leader-election" FlowSchemas from the default APF configuration.
  
  migrate the lock type used in the leader election in your workloads from configmapsleases/endpointsleases to leases. ([#131215](https://github.com/kubernetes/kubernetes/pull/131215), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery, Apps, Network, Scalability and Scheduling]
- The PreferSameTrafficDistribution feature gate is now enabled by default,
  enabling the `PreferSameNode` traffic distribution value for Services. ([#132127](https://github.com/kubernetes/kubernetes/pull/132127), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]
- Updated the built in `system:monitoring` role with permission to access kubelet metrics endpoints. ([#132178](https://github.com/kubernetes/kubernetes/pull/132178), [@gavinkflam](https://github.com/gavinkflam)) [SIG Auth]

### Failing Test

- Kube-apiserver: The --service-account-signing-endpoint flag now only validates the format of abstract socket names ([#131509](https://github.com/kubernetes/kubernetes/pull/131509), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Auth]

### Bug or Regression

- Check for newer resize fields when deciding recovery feature's status in kubelet ([#131418](https://github.com/kubernetes/kubernetes/pull/131418), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- DRA: ResourceClaims requesting a fixed number of devices with `adminAccess` will no longer be allocated the same device multiple times. ([#131299](https://github.com/kubernetes/kubernetes/pull/131299), [@nojnhuh](https://github.com/nojnhuh)) [SIG Node]
- Disable reading of disk geometry before calling expansion for ext and xfs filesystems ([#131568](https://github.com/kubernetes/kubernetes/pull/131568), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Do not expand PVCs annotated with node-expand-not-required ([#131907](https://github.com/kubernetes/kubernetes/pull/131907), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Etcd, Node, Storage and Testing]
- Do not expand volume on the node, if controller expansion is finished ([#131868](https://github.com/kubernetes/kubernetes/pull/131868), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Do not log error event when waiting for expansion on the kubelet ([#131408](https://github.com/kubernetes/kubernetes/pull/131408), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Do not remove CSI json file if volume is already mounted on subsequent errors ([#131311](https://github.com/kubernetes/kubernetes/pull/131311), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Fix ReplicationController reconciliation when the DeploymentReplicaSetTerminatingReplicas feature gate is enabled ([#131822](https://github.com/kubernetes/kubernetes/pull/131822), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- Fix a bug causing unexpected delay of creating pods for newly created jobs ([#132109](https://github.com/kubernetes/kubernetes/pull/132109), [@linxiulei](https://github.com/linxiulei)) [SIG Apps and Testing]
- Fix a bug in Job controller which could result in creating unnecessary Pods for a Job which is already
  recognized as finished (successful or failed). ([#130333](https://github.com/kubernetes/kubernetes/pull/130333), [@kmala](https://github.com/kmala)) [SIG Apps and Testing]
- Fix the allocatedResourceStatuses Field name mismatch in PVC status validation ([#131213](https://github.com/kubernetes/kubernetes/pull/131213), [@carlory](https://github.com/carlory)) [SIG Apps]
- Fixed a bug in CEL's common.UnstructuredToVal where `==` evaluates to false for identical objects when a field is present but the value is null.  This bug does not impact the Kubernetes API. ([#131559](https://github.com/kubernetes/kubernetes/pull/131559), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fixed a bug that caused duplicate validation when updating a ReplicaSet. ([#131873](https://github.com/kubernetes/kubernetes/pull/131873), [@gavinkflam](https://github.com/gavinkflam)) [SIG Apps]
- Fixed a panic issue related to kubectl revision history kubernetes/kubectl#1724 ([#130503](https://github.com/kubernetes/kubernetes/pull/130503), [@tahacodes](https://github.com/tahacodes)) [SIG CLI]
- Fixed a possible deadlock in the watch client that could happen if the watch was not stopped. ([#131266](https://github.com/kubernetes/kubernetes/pull/131266), [@karlkfi](https://github.com/karlkfi)) [SIG API Machinery]
- Fixed an incorrect reference to `JoinConfigurationKind` in the error message when no ResetConfiguration is found during `kubeadm reset` with the `--config` flag. ([#132258](https://github.com/kubernetes/kubernetes/pull/132258), [@J3m3](https://github.com/J3m3)) [SIG Cluster Lifecycle]
- Fixed an issue where `insufficientResources` was logged as a pointer during pod preemption, making logs more readable. ([#132183](https://github.com/kubernetes/kubernetes/pull/132183), [@chrisy-x](https://github.com/chrisy-x)) [SIG Node]
- Fixed incorrect behavior for AllocationMode: All in ResourceClaim when used in subrequests. ([#131660](https://github.com/kubernetes/kubernetes/pull/131660), [@mortent](https://github.com/mortent)) [SIG Node]
- Fixed misleading response codes in admission control metrics. ([#132165](https://github.com/kubernetes/kubernetes/pull/132165), [@gavinkflam](https://github.com/gavinkflam)) [SIG API Machinery, Architecture and Instrumentation]
- Fixes an issue where Windows kube-proxy's ModifyLoadBalancer API updates did not match HNS state in version 15.4. ModifyLoadBalancer policy is supported from Kubernetes 1.31+. ([#131506](https://github.com/kubernetes/kubernetes/pull/131506), [@princepereira](https://github.com/princepereira)) [SIG Windows]
- HPA controller will no longer emit a 'FailedRescale' event if a scale operation initially fails due to a conflict but succeeds after a retry; a 'SuccessfulRescale' event will be emitted instead. A 'FailedRescale' event is still emitted if retries are exhausted. ([#132007](https://github.com/kubernetes/kubernetes/pull/132007), [@AumPatel1](https://github.com/AumPatel1)) [SIG Apps and Autoscaling]
- Improve error message when a pod with user namespaces is created and the runtime doesn't support user namespaces. ([#131623](https://github.com/kubernetes/kubernetes/pull/131623), [@rata](https://github.com/rata)) [SIG Node]
- Kube-apiserver: Fixes OIDC discovery document publishing when external service account token signing is enabled ([#131493](https://github.com/kubernetes/kubernetes/pull/131493), [@hoskeri](https://github.com/hoskeri)) [SIG API Machinery, Auth and Testing]
- Kube-apiserver: cronjob objects now default empty `spec.jobTemplate.spec.podFailurePolicy.rules[*].onPodConditions[*].status` fields as documented, avoiding validation failures during write requests. ([#131525](https://github.com/kubernetes/kubernetes/pull/131525), [@carlory](https://github.com/carlory)) [SIG Apps]
- Kube-proxy:  Remove iptables cli wait interval flag ([#131961](https://github.com/kubernetes/kubernetes/pull/131961), [@cyclinder](https://github.com/cyclinder)) [SIG Network]
- Kube-scheduler: in Kubernetes 1.33, the number of devices that can be allocated per ResourceClaim was accidentally reduced to 16. Now the supported number of devices per ResourceClaim is 32 again. ([#131662](https://github.com/kubernetes/kubernetes/pull/131662), [@mortent](https://github.com/mortent)) [SIG Node]
- Kubelet: close a loophole where static pods could reference arbitrary ResourceClaims. The pods created by the kubelet then don't run due to a sanity check, but such references shouldn't be allowed regardless. ([#131844](https://github.com/kubernetes/kubernetes/pull/131844), [@pohly](https://github.com/pohly)) [SIG Apps, Auth and Node]
- Kubelet: fix a bug where the unexpected NodeResizeError condition was in PVC status when the csi driver does not support node volume expansion and the pvc has the ReadWriteMany access mode. ([#131495](https://github.com/kubernetes/kubernetes/pull/131495), [@carlory](https://github.com/carlory)) [SIG Storage]
- Reduce 5s delay of tainting `node.kubernetes.io/unreachable:NoExecute` when a Node becomes unreachable ([#120816](https://github.com/kubernetes/kubernetes/pull/120816), [@tnqn](https://github.com/tnqn)) [SIG Apps and Node]
- Skip pod backoff completely when PodMaxBackoffDuration kube-scheduler option is set to zero and SchedulerPopFromBackoffQ feature gate is enabled. ([#131965](https://github.com/kubernetes/kubernetes/pull/131965), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- The shorthand for --output flag in kubectl explain was accidentally deleted, but has been added back. ([#131962](https://github.com/kubernetes/kubernetes/pull/131962), [@superbrothers](https://github.com/superbrothers)) [SIG CLI]
- `kubectl create|delete|get|replace --raw` commands now honor server root paths specified in the kubeconfig file. ([#131165](https://github.com/kubernetes/kubernetes/pull/131165), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Added a warning to `kubectl attach`, notifying / reminding users that commands and output are available via the `log` subresource of that Pod. ([#127183](https://github.com/kubernetes/kubernetes/pull/127183), [@mochizuki875](https://github.com/mochizuki875)) [SIG Auth, CLI, Node and Security]
- Bump cel-go dependency to v0.25.0. The changeset is available at: https://github.com/google/cel-go/compare/v0.23.2...v0.25.0 ([#131444](https://github.com/kubernetes/kubernetes/pull/131444), [@erdii](https://github.com/erdii)) [SIG API Machinery, Auth, Cloud Provider and Node]
- Bump kube dns to v1.26.4 ([#132012](https://github.com/kubernetes/kubernetes/pull/132012), [@pacoxu](https://github.com/pacoxu)) [SIG Cloud Provider]
- By default the binaries like kube-apiserver are built with "grpcnotrace" tag enabled. Please use DBG flag if you want to enable golang tracing. ([#132210](https://github.com/kubernetes/kubernetes/pull/132210), [@dims](https://github.com/dims)) [SIG Architecture]
- Changed apiserver to treat failures decoding a mutating webhook patch as failures to call the webhook so they trigger the webhook failurePolicy and count against metrics like `webhook_fail_open_count` ([#131627](https://github.com/kubernetes/kubernetes/pull/131627), [@dims](https://github.com/dims)) [SIG API Machinery]
- DRA kubelet: logging now uses `driverName` like the rest of the Kubernetes components, instead of `pluginName`. ([#132096](https://github.com/kubernetes/kubernetes/pull/132096), [@pohly](https://github.com/pohly)) [SIG Node and Testing]
- DRA kubelet: recovery from mistakes like scheduling a pod onto a node with the required driver not running is a bit simpler now because the kubelet does not block pod deletion unnecessarily. ([#131968](https://github.com/kubernetes/kubernetes/pull/131968), [@pohly](https://github.com/pohly)) [SIG Node and Testing]
- Fixed some missing white spaces in the flag descriptions and logs. ([#131562](https://github.com/kubernetes/kubernetes/pull/131562), [@logica0419](https://github.com/logica0419)) [SIG Network]
- Hack/update-codegen.sh now automatically ensures goimports and protoc ([#131459](https://github.com/kubernetes/kubernetes/pull/131459), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery]
- Kube-apiserver: removed the deprecated `apiserver_encryption_config_controller_automatic_reload_success_total` and `apiserver_encryption_config_controller_automatic_reload_failure_total` metrics in favor of `apiserver_encryption_config_controller_automatic_reloads_total`. ([#132238](https://github.com/kubernetes/kubernetes/pull/132238), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-scheduler: removed the deprecated scheduler_scheduler_cache_size metric  in favor of scheduler_cache_size ([#131425](https://github.com/kubernetes/kubernetes/pull/131425), [@carlory](https://github.com/carlory)) [SIG Scheduling]
- Kubeadm: fixed missing space when printing the warning about pause image mismatch. ([#131563](https://github.com/kubernetes/kubernetes/pull/131563), [@logica0419](https://github.com/logica0419)) [SIG Cluster Lifecycle]
- Kubeadm: made the coredns deployment manifest use named ports consistently for the liveness and readiness probes. ([#131587](https://github.com/kubernetes/kubernetes/pull/131587), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl interactive delete: treat empty newline input as N ([#132251](https://github.com/kubernetes/kubernetes/pull/132251), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Migrate pkg/kubelet/status to contextual logging ([#130852](https://github.com/kubernetes/kubernetes/pull/130852), [@Chulong-Li](https://github.com/Chulong-Li)) [SIG Node]
- Promote `apiserver_authentication_config_controller_automatic_reloads_total` and `apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds` metrics to BETA. ([#131798](https://github.com/kubernetes/kubernetes/pull/131798), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Instrumentation]
- Promote `apiserver_authorization_config_controller_automatic_reloads_total` and `apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds` metrics to BETA. ([#131768](https://github.com/kubernetes/kubernetes/pull/131768), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Instrumentation]
- Promoted the `SeparateTaintEvictionController` feature gate to GA; it is now enabled unconditionally. ([#122634](https://github.com/kubernetes/kubernetes/pull/122634), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Node and Testing]
- Removed generally available feature-gate `PodDisruptionConditions`. ([#129501](https://github.com/kubernetes/kubernetes/pull/129501), [@carlory](https://github.com/carlory)) [SIG Apps]
- Removes support for API streaming from the `List() method` of the dynamic client. ([#132229](https://github.com/kubernetes/kubernetes/pull/132229), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery, CLI and Testing]
- Removes support for API streaming from the `List() method` of the metadata client. ([#132149](https://github.com/kubernetes/kubernetes/pull/132149), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Removes support for API streaming from the `List() method` of the typed client. ([#132257](https://github.com/kubernetes/kubernetes/pull/132257), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Removes support for API streaming from the rest client. ([#132285](https://github.com/kubernetes/kubernetes/pull/132285), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Types: CycleState, StateData, StateKey and ErrNotFound moved from pkg/scheduler/framework to k8s.io/kube-scheduler/framework.
  Type CycleState that is passed to each plugin in scheduler framework is changed to the new interface CycleState (in k8s.io/kube-scheduler/framework) ([#131887](https://github.com/kubernetes/kubernetes/pull/131887), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Node, Scheduling, Storage and Testing]
- Updated CNI plugins to v1.7.1 ([#131602](https://github.com/kubernetes/kubernetes/pull/131602), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.33.0. ([#131406](https://github.com/kubernetes/kubernetes/pull/131406), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Upgrade CoreDNS to v1.12.1 ([#131151](https://github.com/kubernetes/kubernetes/pull/131151), [@yashsingh74](https://github.com/yashsingh74)) [SIG Cloud Provider and Cluster Lifecycle]

## Dependencies

### Added
- buf.build/gen/go/bufbuild/protovalidate/protocolbuffers/go: 63bb56e
- github.com/GoogleCloudPlatform/opentelemetry-operations-go/detectors/gcp: [v1.26.0](https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/tree/detectors/gcp/v1.26.0)
- github.com/bufbuild/protovalidate-go: [v0.9.1](https://github.com/bufbuild/protovalidate-go/tree/v0.9.1)
- github.com/envoyproxy/go-control-plane/envoy: [v1.32.4](https://github.com/envoyproxy/go-control-plane/tree/envoy/v1.32.4)
- github.com/envoyproxy/go-control-plane/ratelimit: [v0.1.0](https://github.com/envoyproxy/go-control-plane/tree/ratelimit/v0.1.0)
- github.com/go-jose/go-jose/v4: [v4.0.4](https://github.com/go-jose/go-jose/tree/v4.0.4)
- github.com/golang-jwt/jwt/v5: [v5.2.2](https://github.com/golang-jwt/jwt/tree/v5.2.2)
- github.com/grpc-ecosystem/go-grpc-middleware/providers/prometheus: [v1.0.1](https://github.com/grpc-ecosystem/go-grpc-middleware/tree/providers/prometheus/v1.0.1)
- github.com/grpc-ecosystem/go-grpc-middleware/v2: [v2.3.0](https://github.com/grpc-ecosystem/go-grpc-middleware/tree/v2.3.0)
- github.com/spiffe/go-spiffe/v2: [v2.5.0](https://github.com/spiffe/go-spiffe/tree/v2.5.0)
- github.com/zeebo/errs: [v1.4.0](https://github.com/zeebo/errs/tree/v1.4.0)
- go.etcd.io/raft/v3: v3.6.0
- go.opentelemetry.io/contrib/detectors/gcp: v1.34.0
- go.opentelemetry.io/otel/sdk/metric: v1.34.0

### Changed
- cel.dev/expr: v0.19.1 → v0.23.1
- cloud.google.com/go/compute/metadata: v0.5.0 → v0.6.0
- github.com/Microsoft/hnslib: [v0.0.8 → v0.1.1](https://github.com/Microsoft/hnslib/compare/v0.0.8...v0.1.1)
- github.com/cncf/xds/go: [b4127c9 → 2f00578](https://github.com/cncf/xds/compare/b4127c9...2f00578)
- github.com/coredns/corefile-migration: [v1.0.25 → v1.0.26](https://github.com/coredns/corefile-migration/compare/v1.0.25...v1.0.26)
- github.com/cpuguy83/go-md2man/v2: [v2.0.4 → v2.0.6](https://github.com/cpuguy83/go-md2man/compare/v2.0.4...v2.0.6)
- github.com/envoyproxy/go-control-plane: [v0.13.0 → v0.13.4](https://github.com/envoyproxy/go-control-plane/compare/v0.13.0...v0.13.4)
- github.com/envoyproxy/protoc-gen-validate: [v1.1.0 → v1.2.1](https://github.com/envoyproxy/protoc-gen-validate/compare/v1.1.0...v1.2.1)
- github.com/fsnotify/fsnotify: [v1.7.0 → v1.9.0](https://github.com/fsnotify/fsnotify/compare/v1.7.0...v1.9.0)
- github.com/fxamacker/cbor/v2: [v2.7.0 → v2.8.0](https://github.com/fxamacker/cbor/compare/v2.7.0...v2.8.0)
- github.com/golang/glog: [v1.2.2 → v1.2.4](https://github.com/golang/glog/compare/v1.2.2...v1.2.4)
- github.com/google/cel-go: [v0.23.2 → v0.25.0](https://github.com/google/cel-go/compare/v0.23.2...v0.25.0)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.24.0 → v2.26.3](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.24.0...v2.26.3)
- github.com/ishidawataru/sctp: [7ff4192 → ae8eb7f](https://github.com/ishidawataru/sctp/compare/7ff4192...ae8eb7f)
- github.com/jonboulle/clockwork: [v0.4.0 → v0.5.0](https://github.com/jonboulle/clockwork/compare/v0.4.0...v0.5.0)
- github.com/modern-go/reflect2: [v1.0.2 → 35a7c28](https://github.com/modern-go/reflect2/compare/v1.0.2...35a7c28)
- github.com/spf13/cobra: [v1.8.1 → v1.9.1](https://github.com/spf13/cobra/compare/v1.8.1...v1.9.1)
- github.com/spf13/pflag: [v1.0.5 → v1.0.6](https://github.com/spf13/pflag/compare/v1.0.5...v1.0.6)
- github.com/vishvananda/netlink: [62fb240 → v1.3.1](https://github.com/vishvananda/netlink/compare/62fb240...v1.3.1)
- github.com/vishvananda/netns: [v0.0.4 → v0.0.5](https://github.com/vishvananda/netns/compare/v0.0.4...v0.0.5)
- go.etcd.io/bbolt: v1.3.11 → v1.4.0
- go.etcd.io/etcd/api/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/client/pkg/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/client/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/pkg/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/server/v3: v3.5.21 → v3.6.1
- go.etcd.io/gofail: v0.1.0 → v0.2.0
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.42.0 → v0.44.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.58.0 → v0.60.0
- go.opentelemetry.io/contrib/propagators/b3: v1.17.0 → v1.19.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.33.0 → v1.34.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.33.0 → v1.34.0
- go.opentelemetry.io/otel/metric: v1.33.0 → v1.35.0
- go.opentelemetry.io/otel/sdk: v1.33.0 → v1.34.0
- go.opentelemetry.io/otel/trace: v1.33.0 → v1.35.0
- go.opentelemetry.io/otel: v1.33.0 → v1.35.0
- go.opentelemetry.io/proto/otlp: v1.4.0 → v1.5.0
- google.golang.org/genproto/googleapis/api: e6fa225 → a0af3ef
- google.golang.org/genproto/googleapis/rpc: e6fa225 → a0af3ef
- google.golang.org/grpc: v1.68.1 → v1.72.1
- k8s.io/gengo/v2: 1244d31 → 85fd79d
- k8s.io/system-validators: v1.9.1 → v1.10.1
- k8s.io/utils: 3ea5e8c → 4c0f3b2
- sigs.k8s.io/structured-merge-diff/v4: v4.6.0 → v4.7.0

### Removed
- cloud.google.com/go/accessapproval: v1.7.4
- cloud.google.com/go/accesscontextmanager: v1.8.4
- cloud.google.com/go/aiplatform: v1.58.0
- cloud.google.com/go/analytics: v0.22.0
- cloud.google.com/go/apigateway: v1.6.4
- cloud.google.com/go/apigeeconnect: v1.6.4
- cloud.google.com/go/apigeeregistry: v0.8.2
- cloud.google.com/go/appengine: v1.8.4
- cloud.google.com/go/area120: v0.8.4
- cloud.google.com/go/artifactregistry: v1.14.6
- cloud.google.com/go/asset: v1.17.0
- cloud.google.com/go/assuredworkloads: v1.11.4
- cloud.google.com/go/automl: v1.13.4
- cloud.google.com/go/baremetalsolution: v1.2.3
- cloud.google.com/go/batch: v1.7.0
- cloud.google.com/go/beyondcorp: v1.0.3
- cloud.google.com/go/bigquery: v1.58.0
- cloud.google.com/go/billing: v1.18.0
- cloud.google.com/go/binaryauthorization: v1.8.0
- cloud.google.com/go/certificatemanager: v1.7.4
- cloud.google.com/go/channel: v1.17.4
- cloud.google.com/go/cloudbuild: v1.15.0
- cloud.google.com/go/clouddms: v1.7.3
- cloud.google.com/go/cloudtasks: v1.12.4
- cloud.google.com/go/compute: v1.23.3
- cloud.google.com/go/contactcenterinsights: v1.12.1
- cloud.google.com/go/container: v1.29.0
- cloud.google.com/go/containeranalysis: v0.11.3
- cloud.google.com/go/datacatalog: v1.19.2
- cloud.google.com/go/dataflow: v0.9.4
- cloud.google.com/go/dataform: v0.9.1
- cloud.google.com/go/datafusion: v1.7.4
- cloud.google.com/go/datalabeling: v0.8.4
- cloud.google.com/go/dataplex: v1.14.0
- cloud.google.com/go/dataproc/v2: v2.3.0
- cloud.google.com/go/dataqna: v0.8.4
- cloud.google.com/go/datastore: v1.15.0
- cloud.google.com/go/datastream: v1.10.3
- cloud.google.com/go/deploy: v1.17.0
- cloud.google.com/go/dialogflow: v1.48.1
- cloud.google.com/go/dlp: v1.11.1
- cloud.google.com/go/documentai: v1.23.7
- cloud.google.com/go/domains: v0.9.4
- cloud.google.com/go/edgecontainer: v1.1.4
- cloud.google.com/go/errorreporting: v0.3.0
- cloud.google.com/go/essentialcontacts: v1.6.5
- cloud.google.com/go/eventarc: v1.13.3
- cloud.google.com/go/filestore: v1.8.0
- cloud.google.com/go/firestore: v1.14.0
- cloud.google.com/go/functions: v1.15.4
- cloud.google.com/go/gkebackup: v1.3.4
- cloud.google.com/go/gkeconnect: v0.8.4
- cloud.google.com/go/gkehub: v0.14.4
- cloud.google.com/go/gkemulticloud: v1.1.0
- cloud.google.com/go/gsuiteaddons: v1.6.4
- cloud.google.com/go/iam: v1.1.5
- cloud.google.com/go/iap: v1.9.3
- cloud.google.com/go/ids: v1.4.4
- cloud.google.com/go/iot: v1.7.4
- cloud.google.com/go/kms: v1.15.5
- cloud.google.com/go/language: v1.12.2
- cloud.google.com/go/lifesciences: v0.9.4
- cloud.google.com/go/logging: v1.9.0
- cloud.google.com/go/longrunning: v0.5.4
- cloud.google.com/go/managedidentities: v1.6.4
- cloud.google.com/go/maps: v1.6.3
- cloud.google.com/go/mediatranslation: v0.8.4
- cloud.google.com/go/memcache: v1.10.4
- cloud.google.com/go/metastore: v1.13.3
- cloud.google.com/go/monitoring: v1.17.0
- cloud.google.com/go/networkconnectivity: v1.14.3
- cloud.google.com/go/networkmanagement: v1.9.3
- cloud.google.com/go/networksecurity: v0.9.4
- cloud.google.com/go/notebooks: v1.11.2
- cloud.google.com/go/optimization: v1.6.2
- cloud.google.com/go/orchestration: v1.8.4
- cloud.google.com/go/orgpolicy: v1.12.0
- cloud.google.com/go/osconfig: v1.12.4
- cloud.google.com/go/oslogin: v1.13.0
- cloud.google.com/go/phishingprotection: v0.8.4
- cloud.google.com/go/policytroubleshooter: v1.10.2
- cloud.google.com/go/privatecatalog: v0.9.4
- cloud.google.com/go/pubsub: v1.34.0
- cloud.google.com/go/pubsublite: v1.8.1
- cloud.google.com/go/recaptchaenterprise/v2: v2.9.0
- cloud.google.com/go/recommendationengine: v0.8.4
- cloud.google.com/go/recommender: v1.12.0
- cloud.google.com/go/redis: v1.14.1
- cloud.google.com/go/resourcemanager: v1.9.4
- cloud.google.com/go/resourcesettings: v1.6.4
- cloud.google.com/go/retail: v1.14.4
- cloud.google.com/go/run: v1.3.3
- cloud.google.com/go/scheduler: v1.10.5
- cloud.google.com/go/secretmanager: v1.11.4
- cloud.google.com/go/security: v1.15.4
- cloud.google.com/go/securitycenter: v1.24.3
- cloud.google.com/go/servicedirectory: v1.11.3
- cloud.google.com/go/shell: v1.7.4
- cloud.google.com/go/spanner: v1.55.0
- cloud.google.com/go/speech: v1.21.0
- cloud.google.com/go/storagetransfer: v1.10.3
- cloud.google.com/go/talent: v1.6.5
- cloud.google.com/go/texttospeech: v1.7.4
- cloud.google.com/go/tpu: v1.6.4
- cloud.google.com/go/trace: v1.10.4
- cloud.google.com/go/translate: v1.10.0
- cloud.google.com/go/video: v1.20.3
- cloud.google.com/go/videointelligence: v1.11.4
- cloud.google.com/go/vision/v2: v2.7.5
- cloud.google.com/go/vmmigration: v1.7.4
- cloud.google.com/go/vmwareengine: v1.0.3
- cloud.google.com/go/vpcaccess: v1.7.4
- cloud.google.com/go/webrisk: v1.9.4
- cloud.google.com/go/websecurityscanner: v1.6.4
- cloud.google.com/go/workflows: v1.12.3
- cloud.google.com/go: v0.112.0
- github.com/BurntSushi/toml: [v0.3.1](https://github.com/BurntSushi/toml/tree/v0.3.1)
- github.com/census-instrumentation/opencensus-proto: [v0.4.1](https://github.com/census-instrumentation/opencensus-proto/tree/v0.4.1)
- github.com/client9/misspell: [v0.3.4](https://github.com/client9/misspell/tree/v0.3.4)
- github.com/cncf/udpa/go: [269d4d4](https://github.com/cncf/udpa/tree/269d4d4)
- github.com/ghodss/yaml: [v1.0.0](https://github.com/ghodss/yaml/tree/v1.0.0)
- github.com/go-kit/kit: [v0.9.0](https://github.com/go-kit/kit/tree/v0.9.0)
- github.com/go-logfmt/logfmt: [v0.4.0](https://github.com/go-logfmt/logfmt/tree/v0.4.0)
- github.com/go-stack/stack: [v1.8.0](https://github.com/go-stack/stack/tree/v1.8.0)
- github.com/golang-jwt/jwt/v4: [v4.5.2](https://github.com/golang-jwt/jwt/tree/v4.5.2)
- github.com/golang/mock: [v1.1.1](https://github.com/golang/mock/tree/v1.1.1)
- github.com/grpc-ecosystem/grpc-gateway: [v1.16.0](https://github.com/grpc-ecosystem/grpc-gateway/tree/v1.16.0)
- github.com/konsorten/go-windows-terminal-sequences: [v1.0.1](https://github.com/konsorten/go-windows-terminal-sequences/tree/v1.0.1)
- github.com/kr/logfmt: [b84e30a](https://github.com/kr/logfmt/tree/b84e30a)
- github.com/opentracing/opentracing-go: [v1.1.0](https://github.com/opentracing/opentracing-go/tree/v1.1.0)
- go.etcd.io/etcd/client/v2: v2.305.21
- go.etcd.io/etcd/raft/v3: v3.5.21
- go.uber.org/atomic: v1.7.0
- golang.org/x/lint: d0100b6
- google.golang.org/appengine: v1.4.0
- google.golang.org/genproto: ef43131
- honnef.co/go/tools: ea95bdf