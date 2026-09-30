<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.38.0-alpha.1](#v1380-alpha1)
  - [Downloads for v1.38.0-alpha.1](#downloads-for-v1380-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.37.0](#changelog-since-v1370)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Dependency](#dependency)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Documentation](#documentation)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)

<!-- END MUNGE: GENERATED_TOC -->

# v1.38.0-alpha.1


## Downloads for v1.38.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes.tar.gz) | dca191621a9d10f7289339266abaaac97e43401f1b681bda6ef6429ca4b1fc5bce9d4aa0d9c5a96128a0fbcd43f9b07f5128e24add6c36e0efe841113fa5f7b3
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-src.tar.gz) | 58ae09f7c3e5e50b9d130b11299e2e364b0c1e3d6bdbddcf5dd04d99799f8321d4ccaeb4d1b17c3b29b7da83214c9f54397c266badfb9f1f21643d3c533b6946

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 10b3ca27c8cf3f08d077cc74c3e1b81a78d908a3966cd7afa3271ae85e8005cf7042e4a13931284d0b2adf18c78b2380977624cdfd685e52a0c935b77d5a447f
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | c1c29fc1b29be17678b2caa21467991d9f476a5ecdd6b62bfa6215c0d40c9367c1ac43f7d7cc4af02dbac8d11d09b818beb077311519dd9655e67077f11d165a
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 74d979c23ebac98b8414d9a625fbb9fe56a8766bf8e4242af3bef8753609ae46243b07b523d01d4903d20193058993ce1261aac8290aaea45a04b78ee92b2462
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 3f54eaa3c9592b2e141224690118ad67a2249e7e3dd45d35dcc596f64d128fb1c2a9ef9481a805881a4c57c5d9d50ca86e6265e68ea5c3a147c97fb09b5725b6
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 1ea11c633ae90d6430deb6529fded74dd9d777aaa5fd9d51361db73115af080491273c3a3e8183bd4f1d6caa5ad8415daa6565cf916f5ee7ea4d82eb7e99faea
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | a56ceda02e2217b7b06cf9be8e7a001e7098ab0ec970c8f3775fdcc03242f6746f17946f7a5e4fa86a15f76f7d6ac445d7057f3dacf83c29c33a4b903993d633
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | fb726ab5717b2d3219a40b21e1c4c1cbfc3b1042440239dd9f0263fa13453148c39400e1d78c8d210f9487b1b9bf665bff2a02b06eb6d4d4cf2595d393e3ef0e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 32ee44042a6786a236f08ceb10a93bb92ca7da118be44f940f742e5797f86aa06bd9d7815bfe9d6f4f35443120fa33693fc0e5b96482687cdf33f0048519434d
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 039071b84a2c2f6928cf7727fa50ba2d00968af157f1b84e6e18e76be0c50e3681d1936f50320f682a85f4d98d91bcfbfb762023dacce2a2d141ef376d3372c2
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 3074c39f87c196f0cadc80affb5c5bf691ff64a30cc2adaf7f1293aca752676e1ea6914e77dc38678c0a06cb1e31964dbd5adfbbd10f54b84fa3da8d6a6b7f8e
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | d0297867e2ebc40b3844ba98a0776c7dba99651fd5b47691c39557207f9b904d2024fe848095af0700442348f2614807c08a47330869df5385f8e9d928c8125d

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | a83cb6f2d9754d833e6c2550c1a163fd15b04e24a59a85190631d4e73af597db6319c8d29250465a9c71a889e7d13364555b26e5622b5f639a0ca35cda0b15df
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | cccb58bbece05401f7dcf2e251e5978adba4d8dc46419df38b5e17914f25f8c724add6498da21cdfe097fd3ae76a6627f53790fa2714639ea162578e71a94a06
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 243d94899bea2f2cd7acd1d817232a014a7ba9078772f4cc4c3bf7bf6ad41cf5cffa467dbb4ade78bb4f8f0ad6f14fdcdbb753fcefc9ac266827dbfede27d1bd
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | b183ebce372e40d3e2be6376ac90a2fbf4c9c6d279b9e8d9d4aa70a8891316263ee62ac47d1b8e96ead6f893082ea1ecd52073e5e6e6fe07248492836cf1b5da

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 62f07fa7d2a93f771cf494875691364d0ea5dfd1ad6e788e4ca23848a3f9675908a32054ec8354b9e7773cdbd9863f60d82e02264ab0c29154a0d7eaedfea87b
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 1a75974e2597e1461f3bd58e7ace4318f58462ed898f86629e7ea5d7af34db3aa7f76b318d1f0b2b010cc25e053e6db0c12efb50c4dded5bbe874f92ac8ebdd0
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | fc72dd13365a5e0a0685efbfbe37077b8ea09270d319d3ba68c023c1e2f001781946e2b75704af69d2eadcd11e87867bf4b8fd6d3fd912072f0e6bd3fe0ac0b3
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | defed2d9015f644facf3b608c41e949a60f09dfa5c04817925d3786ec6aa3563b583b5bc400c0b1fd80d9c7097e03ee1786664eb0bf033737d1bf52ba2db199c
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.38.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 0c2edc8d2a0d7e60dddedc2b3fd43e91b09abba1f27d4055cc6ca7463c58ff3addd496070ce25426883e54be7c750dc21bab45a8e89bfa1b56494322f71a0039

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.38.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.38.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.38.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.38.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.38.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.38.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.37.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Out-of-tree kube-scheduler plugins using the `NominatedPodsForNode` method must now pass `logger klog.Logger` as the first argument. ([#142118](https://github.com/kubernetes/kubernetes/pull/142118), [@jdzikowski](https://github.com/jdzikowski)) [SIG Scheduling and Testing]
  - Removed the `PodSchedulingReadiness` feature gate from `kube-scheduler` and related components as the feature is permanently enabled. ([#141741](https://github.com/kubernetes/kubernetes/pull/141741), [@wasylkowski-a](https://github.com/wasylkowski-a)) [SIG Scheduling]
  - Removed the `SeparateTaintEvictionController` feature gate from `kube-controller-manager`. Remove this gate from existing `--feature-gates` configuration before upgrading. Taint-based eviction is now permanently implemented by the standalone taint eviction controller, which can still be disabled using `--controllers=-taint-eviction-controller`. ([#141789](https://github.com/kubernetes/kubernetes/pull/141789), [@wasylkowski-a](https://github.com/wasylkowski-a)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
 
## Changes by Kind

### Dependency

- Bump cadvisor to v0.60.6. ([#142272](https://github.com/kubernetes/kubernetes/pull/142272), [@cmosetick](https://github.com/cmosetick)) [SIG Node]
- Kubernetes is now built with Go 1.27.1 ([#141663](https://github.com/kubernetes/kubernetes/pull/141663), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- Updated google.golang.org/grpc to v1.82.2 to address CVE-2026-84445. ([#141976](https://github.com/kubernetes/kubernetes/pull/141976), [@harshitgupta31415](https://github.com/harshitgupta31415)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Network, Node and Scheduling]
- Updated google.golang.org/grpc to v1.84.0 to pick up the fix for CVE-2026-84304. ([#142410](https://github.com/kubernetes/kubernetes/pull/142410), [@dims](https://github.com/dims)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling and Storage]

### API Change

- Added resource.MaxMilliQuantity(), which returns the largest quantity whose MilliValue and ScaledValue(resource.Milli) do not overflow an int64. For a non-negative quantity, compare it against the returned bound with Quantity.Cmp before calling MilliValue or ScaledValue(resource.Milli). Comparing Quantity.Value against resource.MaxMilliValue for this is unreliable, because Value itself overflows for a quantity larger than int64. ([#140674](https://github.com/kubernetes/kubernetes/pull/140674), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- Adds (Beta) support for requesting pod certificates to be signed using ML-DSA algorithms. This updates both PodCertificateRequest and the `projected` volume type. ([#142108](https://github.com/kubernetes/kubernetes/pull/142108), [@everettraven](https://github.com/everettraven)) [SIG API Machinery, Apps, Auth, Autoscaling, Node and Scheduling]
- Adds support for using PKCS#10 signing requests signed with ML-DSA keys in CertificateSigningRequest, gated by the CertificateSigningRequestMLDSA feature-gate. ([#142196](https://github.com/kubernetes/kubernetes/pull/142196), [@everettraven](https://github.com/everettraven)) [SIG API Machinery, Apps and Auth]
- Allow CompositePodGroup.Spec.SchedulingPolicy.Gang.MinGroupCount to be mutable, and automatically re-enqueue pods in the scheduler when MinGroupCount decreases. ([#141023](https://github.com/kubernetes/kubernetes/pull/141023), [@jdzikowski](https://github.com/jdzikowski)) [SIG API Machinery, Scheduling and Testing]
- CRD validation failures for `minItems` and `minProperties` now report `Too few: <count>: must have at least N items` instead of the misleading `Invalid value: <count>`. ([#141641](https://github.com/kubernetes/kubernetes/pull/141641), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery]
- Clearing `node.spec.podCIDRs` on update now reports "field cannot be cleared once set" instead of "node updates may not change podCIDR except from \"\" to valid". ([#141260](https://github.com/kubernetes/kubernetes/pull/141260), [@krishhna24](https://github.com/krishhna24)) [SIG API Machinery, Apps and Testing]
- Clients will now receive resource versions on delete when deleting objects. This resource version is either the actual delete or a version when the object was observed to be deleted at. ([#138724](https://github.com/kubernetes/kubernetes/pull/138724), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth, Etcd, Node, Scheduling and Testing]
- Exposed PlacementFeasible as an extension point in kube-scheduler, allowing for out-of-tree PodGroup placement feasibility evaluation. ([#141314](https://github.com/kubernetes/kubernetes/pull/141314), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Kube-scheduler: TAS placement selection now respects NominatedNodeName. This does not yet apply to PodGroups belonging to a CompositePodGroup; see https://github.com/kubernetes/kubernetes/issues/140863. ([#139472](https://github.com/kubernetes/kubernetes/pull/139472), [@mboersma](https://github.com/mboersma)) [SIG API Machinery, Scheduling and Testing]
- Kubelet: KubeletConfiguration now supports selecting ECDSA, RSA, or ML-DSA key algorithms for client and server certificate requests. The default remains ECDSA-P256. ([#142046](https://github.com/kubernetes/kubernetes/pull/142046), [@rphillips](https://github.com/rphillips)) [SIG API Machinery, Auth, Node and Scheduling]
- Preemption-capable scheduler extenders must now handle candidates with empty victim lists. ([#135486](https://github.com/kubernetes/kubernetes/pull/135486), [@FouoF](https://github.com/FouoF)) [SIG Scheduling]
- Removed locked GA feature gates `CustomResourceFieldSelectors` and `CRDValidationRatcheting`. ([#141207](https://github.com/kubernetes/kubernetes/pull/141207), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Testing]
- Resource.Quantity: calling String or encoding a quantity is now guaranteed to not mutate the instance. Previously it did in some cases. If you know that an instance will need the string representation multiple times, use CacheString to pre-populate the value at a time when the caller owns the instance and can safely mutate it. This is not necessary for instances coming from ParseQuantity, only for instances created differently or modified through math operations. ([#142277](https://github.com/kubernetes/kubernetes/pull/142277), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- The OpenAPI schema for ValidatingAdmissionPolicyBinding and MutatingAdmissionPolicyBinding now correctly marks `spec.paramRef.parameterNotFoundAction` as required, matching long-standing apiserver validation. Client-side validators (e.g. kubeconform) can now catch the missing field before apply. ([#141527](https://github.com/kubernetes/kubernetes/pull/141527), [@grosser](https://github.com/grosser)) [SIG API Machinery]
- The `DynamicResourceAllocation` feature gate, which has been locked to default on since v1.35, has been removed. Dynamic resource allocation is now unconditionally enabled. ([#141946](https://github.com/kubernetes/kubernetes/pull/141946), [@troychiu](https://github.com/troychiu)) [SIG API Machinery, Apps, Auth, Node, Scheduling and Testing]
- The `endpoint_slice_controller_changes` metric is deprecated in favor of `endpoint_slice_controller_changes_total`. Both metrics are emitted during the deprecation period. ([#140447](https://github.com/kubernetes/kubernetes/pull/140447), [@cipheraxat](https://github.com/cipheraxat)) [SIG Architecture, Instrumentation and Network]
- The apidiscovery.k8s.io/v2beta1 types and the AggregatedDiscoveryRemoveBetaType feature gate are removed, including the k8s.io/api/apidiscovery/v2beta1 Go package. ([#141208](https://github.com/kubernetes/kubernetes/pull/141208), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- The deprecated and unused `metrics.k8s.io/v1alpha1` API has been removed. Clients still using it should migrate to `metrics.k8s.io/v1` or `metrics.k8s.io/v1beta1`. ([#140862](https://github.com/kubernetes/kubernetes/pull/140862), [@dalaoqi](https://github.com/dalaoqi)) [SIG Instrumentation]
- The documentation of `Node.status.volumesAttached[].devicePath` now states the platform-specific semantics: on Linux it is the host block-device node, on Windows it carries the CSI VolumeID. ([#141861](https://github.com/kubernetes/kubernetes/pull/141861), [@abhinav-phi](https://github.com/abhinav-phi)) [SIG API Machinery, Apps, CLI, Storage and Windows]
- Updated the authorization API group to improve OpenAPI schema correctness for fields being optional or required. ([#139728](https://github.com/kubernetes/kubernetes/pull/139728), [@krishhna24](https://github.com/krishhna24)) [SIG Auth]
- Updated the batch API group to improve OpenAPI schema correctness for fields being optional or required. ([#140010](https://github.com/kubernetes/kubernetes/pull/140010), [@krishhna24](https://github.com/krishhna24)) [SIG Apps]
- `resource.Quantity` gains `AsScaledInt64` and `AsMilliInt64`, which report int64 overflow with a boolean. `Value`, `MilliValue` and `ScaledValue` now saturate to `math.MinInt64` or `math.MaxInt64` on overflow instead of returning a wrapped value. A kubelet CPU reservation no longer wraps to a negative value when the configured quantity exceeds that range. ([#141305](https://github.com/kubernetes/kubernetes/pull/141305), [@thc1006](https://github.com/thc1006)) [SIG API Machinery and Node]

### Feature

- Add a library for generating unique names for workloads and podgroups with builder library ([#141557](https://github.com/kubernetes/kubernetes/pull/141557), [@kannon92](https://github.com/kannon92)) [SIG Apps and Scheduling]
- Added auto complete support to kubectl explain. ([#140058](https://github.com/kubernetes/kubernetes/pull/140058), [@oscrx](https://github.com/oscrx)) [SIG CLI]
- Added the `cacher_queue_latency` stage to the `apiserver_watch_events_dispatch_duration_seconds` metric to measure watch event queuing latency in the cacher. ([#141895](https://github.com/kubernetes/kubernetes/pull/141895), [@richabanker](https://github.com/richabanker)) [SIG API Machinery and Instrumentation]
- Added the `watcher_queue_latency` stage to the `apiserver_watch_events_dispatch_duration_seconds` metric to measure watch event queuing latency in individual watch buffers. ([#141896](https://github.com/kubernetes/kubernetes/pull/141896), [@richabanker](https://github.com/richabanker)) [SIG API Machinery and Instrumentation]
- Client-go: add support for ML-DSA keys from the crypto/mldsa package of Go 1.27 ([#142038](https://github.com/kubernetes/kubernetes/pull/142038), [@neolit123](https://github.com/neolit123)) [SIG API Machinery and Auth]
- Conformance test `Pods, completes the lifecycle of a Pod and the PodStatus` now cover resetting the `PodReadyToStartContainers` condition ([#141460](https://github.com/kubernetes/kubernetes/pull/141460), [@Priyankasaggu11929](https://github.com/Priyankasaggu11929)) [SIG Node and Testing]
- DRA: the helper  code for ResourceSlice publishing now validates that drivers use the shorter attribute and capacity names without the driver prefix. DRA driver authors can opt out of this check. ([#142218](https://github.com/kubernetes/kubernetes/pull/142218), [@pohly](https://github.com/pohly)) [SIG Node]
- DRADeviceTaints and DRADeviceTaintRules feature gates are now locked to true (GA) in 1.38. Users with these gates explicitly set to false in --feature-gates should remove those entries as they will be ignored. Both gates will be removed in 1.41. ([#140256](https://github.com/kubernetes/kubernetes/pull/140256), [@SamWangEng](https://github.com/SamWangEng)) [SIG Apps and Node]
- Extends the scheduler's queue metrics to support `CompositePodGroup` ([#141800](https://github.com/kubernetes/kubernetes/pull/141800), [@dom4ha](https://github.com/dom4ha)) [SIG Instrumentation and Scheduling]
- Graduated container _env files_ to GA. The `EnvFiles` feature gate is now locked to enabled. ([#139115](https://github.com/kubernetes/kubernetes/pull/139115), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- KEP-4858 "IP/CIDR Validation improvements" is now GA, and the stricter rules
  for validating IP and CIDR values in Kubernetes API types are now always in
  effect. ([#141979](https://github.com/kubernetes/kubernetes/pull/141979), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Apps, Network and Testing]
- Kube-apiserver: Requests to unknown `/apis/...` paths and final delegation 404s now return a JSON `Status` object with `Content-Type: application/json` instead of a plain-text "404 page not found" body. ([#138219](https://github.com/kubernetes/kubernetes/pull/138219), [@yashsingh74](https://github.com/yashsingh74)) [SIG API Machinery and Testing]
- Kube-apiserver: added the alpha `ManagedFieldsOptOut` feature gate (off by default). When it's enabled, clients can omit `metadata.managedFields` from responses by adding `drop=metadata.managedFields` to the `Accept` header, e.g. `Accept: application/json;drop=metadata.managedFields`. This works for all verbs including watch, for built-in and custom resources, in every supported media type. ([#139561](https://github.com/kubernetes/kubernetes/pull/139561), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Testing]
- Kube-controller-manager: promoted metric 'volume_operation_errors_total' from ALPHA to BETA stability. ([#141536](https://github.com/kubernetes/kubernetes/pull/141536), [@obarisk](https://github.com/obarisk)) [SIG Apps, Instrumentation and Storage]
- Kubeadm: add support for the ML-DSA encryption algorithm. "ML-DSA-44",  "ML-DSA-65" and "ML-DSA-87" are now allowed values for ClusterConfiguration.EncryptionAlgorithm for new clusters using the v1beta4 and the still disabled (WIP) v1 API. Note that while kubeadm is adding this support, support in other components like kube-apiserver and etcd might be added a bit later and until then a kubeadm default setup with ML-DSA might fail. ([#142037](https://github.com/kubernetes/kubernetes/pull/142037), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl debug: The sysadmin profile now auto-detects Windows nodes and configures debug pods as Windows Host Process Containers with NT AUTHORITY\SYSTEM access. ([#138182](https://github.com/kubernetes/kubernetes/pull/138182), [@rzlink](https://github.com/rzlink)) [SIG CLI and Windows]
- Kubernetes is now built using Go 1.27.1 ([#142176](https://github.com/kubernetes/kubernetes/pull/142176), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Moves node authorization processing outside of listener callback, preventing lock contention. ([#140803](https://github.com/kubernetes/kubernetes/pull/140803), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Auth]
- Return resource version in status on pod binding ([#142327](https://github.com/kubernetes/kubernetes/pull/142327), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Node]
- Return resource version in status on pod eviction ([#142197](https://github.com/kubernetes/kubernetes/pull/142197), [@michaelasp](https://github.com/michaelasp)) [SIG Node]
- Validation-gen: `+k8s:minimum` and `+k8s:maximum` support `time.Duration` fields with quoted Go duration strings, such as `+k8s:minimum="1s"`. Integer payloads on `time.Duration` fields are now rejected. ([#142368](https://github.com/kubernetes/kubernetes/pull/142368), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery]
- `SelfSignedCertKeyOptions` in `k8s.io/client-go/util/cert` accepts a `KeyGenerator`, allowing self-signed certificates to be generated with a key algorithm other than RSA. The default remains a 2048-bit RSA key. ([#142350](https://github.com/kubernetes/kubernetes/pull/142350), [@antcybersec](https://github.com/antcybersec)) [SIG API Machinery and Auth]
- `kubectl` now prints a warning when creating or updating a `DeviceTaintRule` whose `deviceSelector`
  is present but empty, since that selector matches every device from every driver in the cluster.
  Scripts using `kubectl ... --warnings-as-errors` against an intentional "select all" rule will see a
  non-zero exit code where they previously didn't, even though the object is still created
  successfully — the same behavior `PodSecurity` `warn`-level violations already have. ([#141831](https://github.com/kubernetes/kubernetes/pull/141831), [@yogeshbendre](https://github.com/yogeshbendre))
- `validation-gen` now accepts a `--validation-extensions-file` flag, taking a YAML file that declares project-specific `+k8s:format=<name>` values backed by a regular expression. ([#142116](https://github.com/kubernetes/kubernetes/pull/142116), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery]

### Documentation

- Kube-scheduler metrics that carry a `profile` label now document that label in their help text. ([#140800](https://github.com/kubernetes/kubernetes/pull/140800), [@alimaazamat](https://github.com/alimaazamat)) [SIG Instrumentation and Scheduling]
- Kubectl top pod: add example for filtering metrics by node with --field-selector ([#141656](https://github.com/kubernetes/kubernetes/pull/141656), [@Pratik-Redhat-Tech](https://github.com/Pratik-Redhat-Tech)) [SIG CLI]
- The /flagz plain-text response now notes that it reports command-line flag values (or defaults), not the effective runtime configuration, and points to /configz where available. ([#140233](https://github.com/kubernetes/kubernetes/pull/140233), [@ashishpatel26](https://github.com/ashishpatel26)) [SIG API Machinery and Instrumentation]

### Bug or Regression

- Both `distribute-cpus-across-cores=true` and `align-by-socket=true` options were enabled, even when a single socket had sufficient capacity. ([#141238](https://github.com/kubernetes/kubernetes/pull/141238), [@Chunxia202410](https://github.com/Chunxia202410)) [SIG Node]
- CEL cost estimation for CRD validation rules using URL accessor functions (getScheme, getHostname, getHost, getPort, getEscapedPath, getQuery) now propagates the result size bound from the input URL. Previously, chaining a URL accessor into a string operation like matches() treated the intermediate result as unbounded, causing legitimate validation rules to be rejected for exceeding the cost budget. ([#141984](https://github.com/kubernetes/kubernetes/pull/141984), [@jubittajohn](https://github.com/jubittajohn)) [SIG API Machinery]
- CPUManager now preserves TopologyManager's multi-NUMA affinity when using the
  distribute-cpus-across-numa policy option, ensuring CPU allocation stays
  aligned with device topology (e.g., GPUs, NICs) across NUMA nodes. ([#139456](https://github.com/kubernetes/kubernetes/pull/139456), [@yong-jie-gong](https://github.com/yong-jie-gong)) [SIG Node]
- Client-go: a non-nil NegotiatedSerializer in rest.Config is now used when constructing clientsets instead of being unconditionally overwritten ([#140317](https://github.com/kubernetes/kubernetes/pull/140317), [@npinaeva](https://github.com/npinaeva)) [SIG API Machinery, Auth, CLI and Instrumentation]
- Client-go: restrict CA key encipherment usage to RSA keys. Key encipherment usage is no longer included for non RSA keys in self signed CA certificates generated from client-go ([#142273](https://github.com/kubernetes/kubernetes/pull/142273), [@adoi](https://github.com/adoi)) [SIG API Machinery, Apps, Auth, Node and Testing]
- Conntrack entries for a deleted UDP Service's frontend IPs are now cleared on deletion instead of waiting to expire. ([#141371](https://github.com/kubernetes/kubernetes/pull/141371), [@kiarashazarnia](https://github.com/kiarashazarnia)) [SIG Network and Testing]
- DRA: Fixed scheduling failures for `AllocationMode: All` requests when a resource pool on a candidate node is incomplete or invalid, allowing the scheduler to continue trying other nodes. ([#141898](https://github.com/kubernetes/kubernetes/pull/141898), [@divyanshuprakas-h](https://github.com/divyanshuprakas-h)) [SIG Node]
- Decoding a JSON or CBOR null into a `resource.Quantity` that had already been serialized left its cached text in place, so the quantity read as zero while `String`, JSON, CBOR and unstructured conversion kept returning the previous value. The cached text is now cleared with the rest of the value. ([#141980](https://github.com/kubernetes/kubernetes/pull/141980), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- Device-plugin: skip DRA resources in GetDeviceRunContainerOptions ([#141426](https://github.com/kubernetes/kubernetes/pull/141426), [@jackfrancis](https://github.com/jackfrancis)) [SIG Node]
- EndpointSliceMirroring: Ignore NotFound Errors When Deleting EndpointSlices ([#141608](https://github.com/kubernetes/kubernetes/pull/141608), [@weizhoublue](https://github.com/weizhoublue)) [SIG Apps and Network]
- Ensure job controller retries removing tracking finalizer, unblocking pod removal. ([#141329](https://github.com/kubernetes/kubernetes/pull/141329), [@249043822](https://github.com/249043822)) [SIG Apps]
- Fix a bug in scheduler where a PodGroup and CompositePodGroup sharing the same name may cause lookup collisions. ([#141864](https://github.com/kubernetes/kubernetes/pull/141864), [@iomarsayed](https://github.com/iomarsayed)) [SIG Scheduling]
- Fix drifts of CPUManager gauge metrics: `CPUManagerExclusiveCPUsAllocationCount`, `CPUManagerSharedPoolSizeMilliCores`, `CPUManagerAllocationPerNUMA` from true state. (#141262, @lukaszwojciechowski) ([#141261](https://github.com/kubernetes/kubernetes/pull/141261), [@lukaszwojciechowski](https://github.com/lukaszwojciechowski)) [SIG Node]
- Fix incorrect ExtendedResourceCache mappings when multiple DeviceClasses use the same extended resource name, which could cause stale or incorrect resource-to-DeviceClass mappings after DeviceClass updates or deletion. ([#141110](https://github.com/kubernetes/kubernetes/pull/141110), [@anshulchikhale30-p](https://github.com/anshulchikhale30-p)) [SIG Node]
- Fix kubectl top pod --sum column alignment when --no-headers flag is specified. ([#140874](https://github.com/kubernetes/kubernetes/pull/140874), [@0xff-dev](https://github.com/0xff-dev)) [SIG CLI]
- Fix: Keep Zero-Exponent BinarySI Values on Integer Serialization Path ([#141817](https://github.com/kubernetes/kubernetes/pull/141817), [@weizhoublue](https://github.com/weizhoublue)) [SIG API Machinery]
- Fix: silently dropped DNS resolution latency metric in restclient ([#141493](https://github.com/kubernetes/kubernetes/pull/141493), [@weizhoublue](https://github.com/weizhoublue)) [SIG Architecture and Instrumentation]
- Fixed CPUManager's `distribute-cpus-across-cores` policy option to select CPUs using CPU topology instead of relying on logical CPU numbering, improving CPU distribution across physical cores on systems where sibling threads have contiguous CPU IDs. ([#140766](https://github.com/kubernetes/kubernetes/pull/140766), [@OchukoWH](https://github.com/OchukoWH)) [SIG Node]
- Fixed CompositePodGroup validation in the scheduling cycle to ensure that child CompositePodGroups and PodGroups match the root group's priority and preemptionPolicy. ([#141930](https://github.com/kubernetes/kubernetes/pull/141930), [@macsko](https://github.com/macsko)) [SIG Node and Scheduling]
- Fixed DRA kubelet plugin helper rolling updates for drivers with long valid names by shortening automatic DRA service socket paths when needed. ([#142003](https://github.com/kubernetes/kubernetes/pull/142003), [@ryo-whaletech](https://github.com/ryo-whaletech)) [SIG Node]
- Fixed Memory Manager pod-scope topology hint restoration for multi-container Pods after kubelet restart. ([#141070](https://github.com/kubernetes/kubernetes/pull/141070), [@AutuSnow](https://github.com/AutuSnow)) [SIG Node]
- Fixed StatefulSet scale-down becoming blocked with Parallel Pod management when a failed scale-up leaves missing desired Pods and ResourceQuota prevents their creation. ([#141057](https://github.com/kubernetes/kubernetes/pull/141057), [@suii2210](https://github.com/suii2210)) [SIG Apps]
- Fixed `FailedToCreateEndpoint`/`FailedToUpdateEndpoint` events from the endpoints controller being misfiled into the `default` namespace instead of the service's namespace. ([#141509](https://github.com/kubernetes/kubernetes/pull/141509), [@inspirit941](https://github.com/inspirit941)) [SIG Apps and Network]
- Fixed `PersistentVolumeClaim` `status.capacity` accepting a negative quantity whose `Value()` overflows (for example `-9.5Gi` or `-1e30`), and stopped a positive capacity whose `Value()` wraps negative (for example `9223372036854775808`) from being wrongly rejected. ([#141169](https://github.com/kubernetes/kubernetes/pull/141169), [@semx](https://github.com/semx)) [SIG API Machinery, Apps and Storage]
- Fixed `kubectl top pod --sum` printing the separator and totals rows misaligned by one column when `--show-swap` was also specified. ([#141621](https://github.com/kubernetes/kubernetes/pull/141621), [@0xff-dev](https://github.com/0xff-dev)) [SIG CLI]
- Fixed `pv_collector_total_pv_count` metric reporting `plugin_name="N/A"` for CSI-provisioned PersistentVolumes. The label now correctly shows `kubernetes.io/csi:<driver-name>` (e.g. `kubernetes.io/csi:ebs.csi.aws.com`). ([#139356](https://github.com/kubernetes/kubernetes/pull/139356), [@dfajmon](https://github.com/dfajmon)) [SIG API Machinery, Apps and Storage]
- Fixed `resource.Quantity.AsApproximateFloat64` returning NaN instead of 0 for a zero-valued quantity with a large scale. ([#141937](https://github.com/kubernetes/kubernetes/pull/141937), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fixed `resource.Quantity.String()` (and `MarshalJSON`/`ToUnstructured`) silently dropping the magnitude for `DecimalSI` quantities larger than 10^18 (for example `1000E`), and for `BinarySI` quantities grown past `Ei` via arithmetic (for example 2^70), which corrupted the value and broke round-tripping through String/JSON/YAML. Such quantities now serialize to an exact base-10 representation: decimal exponent notation where trailing zeros allow it (for example `1e21`), otherwise the full decimal integer (for example 2^70 becomes `1180591620717411303424`). ([#140459](https://github.com/kubernetes/kubernetes/pull/140459), [@semx](https://github.com/semx)) [SIG API Machinery]
- Fixed `resource.Quantity.Value()` and `MilliValue()` to round negative values away from zero, as documented. Previously, small negative quantities could round toward positive infinity (e.g. `ParseQuantity("-.484785E-7466").Value()` returned `1` instead of `-1`). ([#138510](https://github.com/kubernetes/kubernetes/pull/138510), [@qflen](https://github.com/qflen)) [SIG API Machinery]
- Fixed a DRA scheduling bug where a device with a live shared allocation could also be allocated exclusively after a ResourceSlice changed it to disallow multiple allocations. ([#140798](https://github.com/kubernetes/kubernetes/pull/140798), [@thc1006](https://github.com/thc1006)) [SIG Node]
- Fixed a bug in `kubectl create ingress` where malformed trailing data in `--rule` passed validation and silently dropped TLS configuration. ([#140926](https://github.com/kubernetes/kubernetes/pull/140926), [@bhuvan-somisetty](https://github.com/bhuvan-somisetty)) [SIG API Machinery and CLI]
- Fixed a bug in the PodTopologySpread scheduler plugin where a topology spread constraint with an empty (non-nil) `labelSelector` (`labelSelector: {}`) was silently ignored instead of matching all pods in the namespace. ([#141340](https://github.com/kubernetes/kubernetes/pull/141340), [@0xff-dev](https://github.com/0xff-dev)) [SIG Scheduling]
- Fixed a bug in the `HorizontalPodAutoscaler`: on amd64, an extremely large metric value could scale a workload down to `minReplicas` instead of up, and an extreme Percent scaling policy could block scaling up entirely. Replica calculations are now capped at the `int32` maximum, so the HPA scales up toward `maxReplicas`, matching the existing behavior on all other architectures. ([#141397](https://github.com/kubernetes/kubernetes/pull/141397), [@ChihweiLHBird](https://github.com/ChihweiLHBird)) [SIG Apps and Autoscaling]
- Fixed a bug in the attach-detach controller that could delay volume detach by up to one node-status update interval (~5min) after a pod is deleted, when the volume attach was uncertain or kubelet had not yet reported the volume in `Node.Status.VolumesInUse`. ([#141385](https://github.com/kubernetes/kubernetes/pull/141385), [@huww98](https://github.com/huww98)) [SIG Apps and Storage]
- Fixed a bug where Kubelet device manager may assign the same device ID to multiple pods simultaneously. ([#137534](https://github.com/kubernetes/kubernetes/pull/137534), [@trozet](https://github.com/trozet)) [SIG Node and Testing]
- Fixed a bug where `NoExecute` device taints (`DeviceTaintRule`) did not evict pods using DRA-backed extended resources (`pod.Status.ExtendedResourceClaimStatus`). ([#142289](https://github.com/kubernetes/kubernetes/pull/142289), [@troychiu](https://github.com/troychiu)) [SIG Apps, Auth, Node and Scheduling]
- Fixed a bug where `resource.ParseQuantity` failed to use the fast integer path for values near `math.MaxInt64` (19-digit quantities), causing `AsInt64()` to incorrectly return false for valid int64 values. ([#141938](https://github.com/kubernetes/kubernetes/pull/141938), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fixed a bug where a DeviceTaintRule with no `spec.deviceSelector` incorrectly matched every device cluster-wide and made all DRA devices unschedulable for `NoSchedule` or `NoExecute` taints. The rule now correctly matches no devices as documented. Use an explicit empty selector (`deviceSelector: {}`) to target all devices. ([#141648](https://github.com/kubernetes/kubernetes/pull/141648), [@marwan562](https://github.com/marwan562)) [SIG Node]
- Fixed a bug where a pod could be permanently lost from the scheduling
  queue when a concurrent Pop picked it up before Done() was called in
  AddUnschedulablePodIfNotPresent. ([#139450](https://github.com/kubernetes/kubernetes/pull/139450), [@rsd-darshan](https://github.com/rsd-darshan)) [SIG Scheduling]
- Fixed a bug where binding a PersistentVolumeClaim to a PersistentVolume by name could misjudge whether the volume was large enough when the capacity or the request was larger than a 64-bit integer. ([#141618](https://github.com/kubernetes/kubernetes/pull/141618), [@thc1006](https://github.com/thc1006)) [SIG Apps and Storage]
- Fixed a bug where kubelet logged `--manifest-url-header` credential values (e.g., Authorization tokens) to runtime logs at startup. Only header key names are now logged. ([#140220](https://github.com/kubernetes/kubernetes/pull/140220), [@rithwik-01](https://github.com/rithwik-01)) [SIG Node]
- Fixed a bug where pods requesting extended resources backed by DRA failed to schedule when the DRA devices also contributed node allocatable resources (with the alpha DRANodeAllocatableResources feature gate enabled). ([#141556](https://github.com/kubernetes/kubernetes/pull/141556), [@harche](https://github.com/harche)) [SIG Apps, Node, Scheduling and Testing]
- Fixed a bug where the kubelet could hang during Pod admission when the CPU manager `distribute-cpus-across-numa` and `full-pcpus-only` policy options were both enabled and per-NUMA CPU availability was not a multiple of the hardware thread count per core. ([#140730](https://github.com/kubernetes/kubernetes/pull/140730), [@kiarashazarnia](https://github.com/kiarashazarnia)) [SIG Node]
- Fixed a kubelet panic when validating an in-place decrease of a container memory limit while pod-level memory usage stats are unavailable. ([#141100](https://github.com/kubernetes/kubernetes/pull/141100), [@tancheng33](https://github.com/tancheng33)) [SIG Node]
- Fixed a panic in DRA drivers that use `k8s.io/dynamic-resource-allocation/client` to `Get` a ResourceClaim and then call `UpdateStatus` on it, once the client has selected `resource.k8s.io/v1beta2`. That selection happens when a driver built with the v1-based helper runs against a Kubernetes 1.33 API server, and also when an object-level `NotFound` makes the client fall back from v1 to v1beta2. ([#141106](https://github.com/kubernetes/kubernetes/pull/141106), [@thc1006](https://github.com/thc1006)) [SIG Node]
- Fixed a panic in the Windows kube-proxy (winkernel) that could terminate kube-proxy when
  an HNS load balancer without port mappings was enumerated. ([#141489](https://github.com/kubernetes/kubernetes/pull/141489), [@prince-melvin](https://github.com/prince-melvin)) [SIG Network and Windows]
- Fixed a performance degradation in scheduling queue, when PodGroups are waiting for quorum and large quantity of events arrive to scheduler. ([#141577](https://github.com/kubernetes/kubernetes/pull/141577), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- Fixed a race in client-go MutationCache indexed lookups that could temporarily hide a recently created replacement object. ([#141736](https://github.com/kubernetes/kubernetes/pull/141736), [@jackfrancis](https://github.com/jackfrancis)) [SIG API Machinery]
- Fixed a rare race in the PersistentVolume controller that could leave a PV stuck in the `Bound` phase and never released after concurrent updates. ([#140691](https://github.com/kubernetes/kubernetes/pull/140691), [@huww98](https://github.com/huww98)) [SIG Apps, Scheduling and Storage]
- Fixed a regression in 1.38 where resource quantities written by a 1.37 or older apiserver with a decimal exponent outside the int32 range, such as `1e4294967296`, could not be decoded, which failed reads of the whole collection. ([#142395](https://github.com/kubernetes/kubernetes/pull/142395), [@antcybersec](https://github.com/antcybersec)) [SIG API Machinery]
- Fixed a zero-division overflow/underflow issue in the HPA replica calculator when the replica count is 0. ([#140958](https://github.com/kubernetes/kubernetes/pull/140958), [@KengoA](https://github.com/KengoA)) [SIG Apps and Autoscaling]
- Fixed an issue where HorizontalPodAutoscaler incorrectly included pod overhead in the request calculation for pods using pod-level resources. ([#142154](https://github.com/kubernetes/kubernetes/pull/142154), [@sophieliu15](https://github.com/sophieliu15)) [SIG Apps and Autoscaling]
- Fixed an issue where a Job recreated with the same name as a recently deleted Job could remain unreconciled because it inherited pending Pod expectations from the previous Job. ([#142247](https://github.com/kubernetes/kubernetes/pull/142247), [@6democratickim9](https://github.com/6democratickim9)) [SIG Apps and Testing]
- Fixed an issue where a container that was recreated while the kubelet could not reach the API server was reported Ready after a kubelet restart, before its own readiness probe had run. ([#141487](https://github.com/kubernetes/kubernetes/pull/141487), [@toVersus](https://github.com/toVersus)) [SIG Node]
- Fixed an issue where liveness and readiness probes could run before the startup probe succeeded after a container restart. ([#141342](https://github.com/kubernetes/kubernetes/pull/141342), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node and Testing]
- Fixed an issue where the in-tree iSCSI detach of a block volume could fail permanently when the /dev/disk/by-path link was already gone (e.g. the iSCSI session was lost before teardown), leaving the volume stuck in node.status.volumesInUse and blocking attach on other nodes. ([#141863](https://github.com/kubernetes/kubernetes/pull/141863), [@abhinav-phi](https://github.com/abhinav-phi)) [SIG Storage]
- Fixed apiserver_watch_cache_events_dispatched_total to not count bookmark events. ([#141333](https://github.com/kubernetes/kubernetes/pull/141333), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Fixed false fractional-byte warnings for integer resource quantities larger than the int64 range. ([#142131](https://github.com/kubernetes/kubernetes/pull/142131), [@6democratickim9](https://github.com/6democratickim9)) [SIG API Machinery]
- Fixed kubectl describe service printing "+ 0 more..." after the endpoint list when exactly three ready endpoints are followed by not-ready endpoints. ([#142214](https://github.com/kubernetes/kubernetes/pull/142214), [@reckless-sherixx](https://github.com/reckless-sherixx)) [SIG CLI]
- Fixed logic around PlacementFeasible status propagation so that the unschedulable plugins for a pod are correctly stored. PodGroup statuses were improved to be more readable. ([#141253](https://github.com/kubernetes/kubernetes/pull/141253), [@macsko](https://github.com/macsko)) [SIG Node, Scheduling and Testing]
- Fixed preemption algorithm in kube-scheduler so that it no longer ignores a PodDisruptionBudget with an empty selector. ([#141785](https://github.com/kubernetes/kubernetes/pull/141785), [@weizhoublue](https://github.com/weizhoublue)) [SIG Scheduling]
- Fixed quantity parsing to reject decimal exponents that are too large to represent as an internal scale, instead of silently narrowing them to an unrelated value (for example `1e4294967297` no longer parses as `1e1`). ([#141203](https://github.com/kubernetes/kubernetes/pull/141203), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- Fixed rare out-of-tree DRA rolling-update failures with 108-byte Unix socket paths. ([#141815](https://github.com/kubernetes/kubernetes/pull/141815), [@weizhoublue](https://github.com/weizhoublue)) [SIG Node]
- Fixed the DRA ResourceSlice controller to preserve `TimeAdded` for every unchanged device taint when updating a ResourceSlice. ([#141107](https://github.com/kubernetes/kubernetes/pull/141107), [@thc1006](https://github.com/thc1006)) [SIG Node]
- Fixed the NodeRestriction admission plugin so the request-serviceaccounts-token-audience authorization check is still consulted when resolving a pod's CSI volume sources fails (e.g. a referenced CSIDriver, PVC, or PV is not found). This restores the authorizer bypass on the error path when the ServiceAccountNodeAudienceRestriction feature is enabled. ([#140607](https://github.com/kubernetes/kubernetes/pull/140607), [@aramase](https://github.com/aramase)) [SIG Auth and Testing]
- Fixed the kubelet CPU manager so that pods with pod-level (QoS-class) CPU allocations are no longer re-allocated from scratch after a kubelet restart. The checkpointed pod-level CPU set and container assignments are now restored as-is, preventing running pods from being rejected or silently migrated to different CPUs, which previously leaked exclusive CPUs. (#140989, @lukaszwojciechowski) ([#141041](https://github.com/kubernetes/kubernetes/pull/141041), [@lukaszwojciechowski](https://github.com/lukaszwojciechowski)) [SIG Node and Testing]
- Fixed the kubelet rejecting a negative CSI volume expansion size. Sizes whose int64 conversion overflowed were previously accepted because the sign was lost. ([#141181](https://github.com/kubernetes/kubernetes/pull/141181), [@ashvinctrl](https://github.com/ashvinctrl)) [SIG Node and Storage]
- Fixes a panic in `kubectl auth reconcile` if an error was encountered reconciling a rolebinding or clusterrolebinding ([#141035](https://github.com/kubernetes/kubernetes/pull/141035), [@liggitt](https://github.com/liggitt)) [SIG Auth and CLI]
- Integer resource quantities (such as extended resources) that are whole numbers large enough to overflow an int64 milli projection are no longer incorrectly rejected as "must be an integer". ([#141349](https://github.com/kubernetes/kubernetes/pull/141349), [@thc1006](https://github.com/thc1006)) [SIG Apps]
- Kube-proxy no longer tries to set the `nf_conntrack_max` sysctl if its
  value is already higher than the value kube-proxy wants it to be. This
  fixes a regression in 1.36 on systems where kube-proxy does not have
  permission to modify the sysctls, since the new maximum value for
  `nf_conntrack_max` could cause kube-proxy to now want a different
  (lower) value than it had before. ([#142035](https://github.com/kubernetes/kubernetes/pull/142035), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kube-scheduler no longer logs the extender TLS private key (`tlsConfig.keyData`) when logging its component configuration at verbosity >=2. ([#140519](https://github.com/kubernetes/kubernetes/pull/140519), [@UditDewan](https://github.com/UditDewan)) [SIG Scheduling]
- Kube-scheduler: a ResourceClaim whose device request counts add up past the int range no longer crashes the scheduler or skips the `firstAvailable` fallback in the DRA allocator. A request over the per-claim device limit is rejected with an error naming it, and an oversized `firstAvailable` alternative falls through to a smaller one. ([#141307](https://github.com/kubernetes/kubernetes/pull/141307), [@thc1006](https://github.com/thc1006)) [SIG Node]
- Kubeadm: fixed a preflight check on OpenRC systems that reported a service as enabled whenever `rc-update show default` mentioned its name anywhere, including as part of another service's name. ([#141981](https://github.com/kubernetes/kubernetes/pull/141981), [@thc1006](https://github.com/thc1006)) [SIG Cluster Lifecycle]
- Kubeadm: no longer persists a node-specific systemd-resolved resolvConf path in the cluster-wide kubelet-config ConfigMap during init. ([#141337](https://github.com/kubernetes/kubernetes/pull/141337), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Kubectl describe node: per-pod resource percentages no longer print -9223372036854775808% on nodes that have not reported allocatable resources, and the "Allocated resources" totals now use the same resource accounting as the per-pod rows. ([#142215](https://github.com/kubernetes/kubernetes/pull/142215), [@reckless-sherixx](https://github.com/reckless-sherixx)) [SIG CLI]
- Kubectl explain no longer prints the KIND/VERSION header to stdout when the command fails because of an invalid field path; the error is still written to stderr and the exit code is unchanged. ([#141534](https://github.com/kubernetes/kubernetes/pull/141534), [@ManafovM](https://github.com/ManafovM)) [SIG CLI]
- Kubectl get --sort-by now treats resources that both lack the selected sort field as equal. ([#141613](https://github.com/kubernetes/kubernetes/pull/141613), [@DevaanshPathak](https://github.com/DevaanshPathak)) [SIG CLI]
- Kubectl: `kubectl label` and `kubectl annotate` now return an error if modification arguments are passed alongside the `--list` flag. ([#141589](https://github.com/kubernetes/kubernetes/pull/141589), [@Roaimkhan](https://github.com/Roaimkhan)) [SIG CLI]
- Kubectl: defaults are now matched by full command path instead of the last path segment. ([#140230](https://github.com/kubernetes/kubernetes/pull/140230), [@Mujib-Ahasan](https://github.com/Mujib-Ahasan)) [SIG CLI]
- Kubelet: a Pod with an unknown or empty `seccompProfile.type` now fails closed, the container fails to start with `CreateContainerConfigError` instead of silently running as `Unconfined`. Both values are already rejected by API validation on Pod create and update, and by the kubelet's own validation for static Pods, so no Pod accepted by a supported API server is affected. This only changes behavior for Pods that reached the kubelet without passing API validation, or for a kubelet older than an API server that supports a seccomp profile type the kubelet does not know. The error message for a `Localhost` profile without `localhostProfile` no longer ends with a trailing period. ([#141958](https://github.com/kubernetes/kubernetes/pull/141958), [@saschagrunert](https://github.com/saschagrunert)) [SIG Auth and Node]
- Kubelet: fixed a panic in the DRA device-health goroutine when the on-disk health checkpoint file decoded to a nil map — either the top-level map (`null`) or a driver's inner `Devices` map (e.g. `{"driverA":{"Devices":null}}`). Both cases are now handled on load, and the next health update proceeds normally instead of panicking. ([#140904](https://github.com/kubernetes/kubernetes/pull/140904), [@bart0sh](https://github.com/bart0sh)) [SIG Node]
- LimitRanger no longer re-validates the resource requests and limits a pod resize leaves unchanged. Only the values a resize changes are checked against the LimitRange, so an existing pod is no longer rejected on resize by a constraint it already violates. ([#142171](https://github.com/kubernetes/kubernetes/pull/142171), [@semx](https://github.com/semx)) [SIG API Machinery and Node]
- LimitRanger no longer rejects an update of a PersistentVolumeClaim because of a request that the update leaves unchanged, even when that request is outside the LimitRange minimum or maximum. A changed, added or removed request is checked against both the minimum and the maximum, as on create, so a stored request outside the range is rejected as soon as an update changes it. ([#142170](https://github.com/kubernetes/kubernetes/pull/142170), [@thc1006](https://github.com/thc1006)) [SIG API Machinery and Storage]
- LimitRanger now compares resource quantities exactly. A LimitRange min, max, or limit/request ratio is enforced for quantities large enough to overflow an int64 projection instead of being silently skipped, and a ratio constraint no longer rejects an equal request and limit above 2^63. Requests and limits within one milli-unit of a bound, such as a request of 0.9999 against a minimum of 1, are now rejected where the previous projection rounded them onto the bound. This applies to pod creation and to every PVC create and update, so a PVC already holding such a value has to have its request corrected before it can be updated. Where the stored request sits above a maximum rather than below a minimum, correcting it would mean decreasing the request, which PVC validation refuses; those claims need the LimitRange widened instead. ([#141348](https://github.com/kubernetes/kubernetes/pull/141348), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- Resource.Quantity.Cmp and CmpInt64 now return the correct result when the two quantities' scales differ by more than an int32 can represent. ([#142013](https://github.com/kubernetes/kubernetes/pull/142013), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- ResourceQuota updates no longer fail validation on a `spec.hard`, `status.hard` or `status.used` value that the object already holds, so a quota holding a value an older release accepted can still be updated and its status still synced. ([#142190](https://github.com/kubernetes/kubernetes/pull/142190), [@thc1006](https://github.com/thc1006)) [SIG API Machinery and Apps]
- RuntimeClass admission plugin will admit Pods with an empty `spec.runtimeClassName`. ([#140108](https://github.com/kubernetes/kubernetes/pull/140108), [@soltysh](https://github.com/soltysh)) [SIG Node]
- The EndpointSlice controller no longer retries reconciliation when an EndpointSlice selected for deletion has already been removed. ([#141304](https://github.com/kubernetes/kubernetes/pull/141304), [@jfremy-openai](https://github.com/jfremy-openai)) [SIG Network]
- When a Service and its corresponding Pods are rapidly deleted and recreated, ensure the associated EndpointSlice remains correct and up-to-date. ([#141593](https://github.com/kubernetes/kubernetes/pull/141593), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Apps, Network and Testing]
- `resource.Quantity`: for int64-backed quantities whose scales, and for subtraction whose scale alignment, fit the decimal backend's scale range, `Neg()` of `math.MinInt64` and `Sub()` with a `math.MinInt64` subtrahend now promote to the decimal backend instead of overflowing while negating the int64 operand. The exact value is preserved (for example, `-math.MinInt64` becomes `2^63`) and `AsInt64()` returns `ok=false` for the decimal-backed result. Unsupported extreme-scale combinations retain their pre-existing behavior. The unchecked `Value()`, `MilliValue()`, and `ScaledValue()` overflow behavior is unchanged and tracked separately. ([#141264](https://github.com/kubernetes/kubernetes/pull/141264), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Apiserver_request_total now counts requests to the aggregated /openapi/v2 endpoint under subresource="openapi/v2". ([#140888](https://github.com/kubernetes/kubernetes/pull/140888), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Bump crictl (cri-tools) to v1.37.0 ([#141740](https://github.com/kubernetes/kubernetes/pull/141740), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Bump etcd SDK to [v3.7.1 ](https://github.com/etcd-io/etcd/issues/22156) ([#141033](https://github.com/kubernetes/kubernetes/pull/141033), [@humblec](https://github.com/humblec)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]
- Kubeadm: added a preflight warning when `getent` is not found in PATH. kubelet uses it to resolve the `kubelet` account when setting up user namespace ID mappings. ([#141968](https://github.com/kubernetes/kubernetes/pull/141968), [@thc1006](https://github.com/thc1006)) [SIG Cluster Lifecycle]
- Kubelet is built by default without using CGO across all platforms. If you need CGO support, please use KUBE_CGO_OVERRIDES env variable when building kubernetes. ([#135870](https://github.com/kubernetes/kubernetes/pull/135870), [@dims](https://github.com/dims)) [SIG Architecture, Node and Testing]
- Promote `apiserver_flowcontrol_priority_level_seat_utilization` to BETA ([#141161](https://github.com/kubernetes/kubernetes/pull/141161), [@LiomSV](https://github.com/LiomSV)) [SIG API Machinery and Instrumentation]
- Removed Permit extension point implementation from GangScheduling plugin. Similar functionality is already covered by the PlacementFeasible, making the Permit redundant. ([#141182](https://github.com/kubernetes/kubernetes/pull/141182), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Removed the `KubeletTracing` feature gate, which has been GA and locked to default since v1.34. ([#141714](https://github.com/kubernetes/kubernetes/pull/141714), [@dashpole](https://github.com/dashpole)) [SIG Instrumentation and Node]
- Removed the deprecated `WindowsHostNetwork` feature gate (KEP-3503, withdrawn). The gate no longer guarded any code paths. ([#142018](https://github.com/kubernetes/kubernetes/pull/142018), [@Adarsh-Me](https://github.com/Adarsh-Me)) [SIG Node and Windows]
- Removed the generally available feature gate `ServiceAccountTokenPodNodeInfo`, which was locked and enabled since 1.32. ([#135338](https://github.com/kubernetes/kubernetes/pull/135338), [@carlory](https://github.com/carlory)) [SIG Auth, Node and Testing]
- Rename the label "cache_to_watcher" in apiserver_watch_events_dispatch_duration_seconds ALPHA metric to "watcher_to_client_handler" ([#141175](https://github.com/kubernetes/kubernetes/pull/141175), [@richabanker](https://github.com/richabanker)) [SIG API Machinery and Instrumentation]
- Updated the default etcd version to 3.7.1 ([#141032](https://github.com/kubernetes/kubernetes/pull/141032), [@humblec](https://github.com/humblec)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Updates the bundled CoreDNS version to v1.14.7. ([#141462](https://github.com/kubernetes/kubernetes/pull/141462), [@bzsuni](https://github.com/bzsuni)) [SIG Cloud Provider and Cluster Lifecycle]

## Dependencies

### Added
- cloud.google.com/go/auth: [v0.20.0](https://github.com/googleapis/google-cloud-go/commit/9886dcff5240f37ff21e26c7462325f3c1cfafc8)
- github.com/anishathalye/porcupine: [v1.3.0](https://github.com/anishathalye/porcupine/commit/55508eb201b314c218d7e8412c3ea4b9499a5f53)
- github.com/coreos/go-oidc/v3: [v3.21.0](https://github.com/coreos/go-oidc/commit/c914bd380327a5a3a81403774d1a5d5b73772ce7)
- github.com/google/s2a-go: [v0.1.9](https://github.com/google/s2a-go/commit/b293be1aa7a6e6e4565f9967c093dd412253b267)
- github.com/googleapis/enterprise-certificate-proxy: [v0.3.15](https://github.com/googleapis/enterprise-certificate-proxy/commit/6964da16bfdce018ff1041134b2508988e0d994d)
- github.com/googleapis/gax-go/v2: [v2.22.0](https://github.com/googleapis/gax-go/commit/1cdb1c1716745c77ec8538923d1c5b9ff0c1a02c)
- github.com/peterbourgon/diskv/v3: [v3.0.1](https://github.com/peterbourgon/diskv/tree/v3.0.1)
- google.golang.org/api: [v0.278.0](https://github.com/googleapis/google-api-go-client/commit/07c758daacbc24e32753c3f1b537c7f6cce626f0)
- sigs.k8s.io/structured-merge-diff/v7: [v7.0.0](https://github.com/kubernetes-sigs/structured-merge-diff/commit/8d11054be605e4018886201fb2bce0651c053592)

### Changed
- cel.dev/expr: [v0.25.1 → v0.25.2](https://github.com/google/cel-spec/compare/v0.25.1...v0.25.2)
- github.com/GoogleCloudPlatform/opentelemetry-operations-go/detectors/gcp: [v1.32.0 → v1.34.0](https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/compare/v1.32.0...v1.34.0)
- github.com/chai2010/gettext-go: [v1.0.2 → v1.0.3](https://github.com/chai2010/gettext-go/compare/v1.0.2...v1.0.3)
- github.com/container-storage-interface/spec: [cd9e7ad → v1.13.0](https://github.com/container-storage-interface/spec/compare/cd9e7ad1ae0915cabcad179f2b8a660c0cb6eb9f...v1.13.0)
- github.com/containerd/containerd/api: [v1.11.1 → v1.12.0](https://github.com/containerd/containerd/compare/api/v1.11.1...api/v1.12.0)
- github.com/containerd/log: [v0.1.0 → v0.2.0](https://github.com/containerd/log/compare/v0.1.0...v0.2.0)
- github.com/containerd/ttrpc: [v1.2.9 → v1.2.10](https://github.com/containerd/ttrpc/compare/v1.2.9...v1.2.10)
- github.com/coredns/caddy: [v1.1.1 → v1.1.4](https://github.com/coredns/caddy/compare/v1.1.1...v1.1.4)
- github.com/coredns/corefile-migration: [v1.0.34 → v1.0.35](https://github.com/coredns/corefile-migration/compare/v1.0.34...v1.0.35)
- github.com/felixge/httpsnoop: [v1.0.4 → v1.1.0](https://github.com/felixge/httpsnoop/compare/v1.0.4...v1.1.0)
- github.com/fxamacker/cbor/v2: [v2.9.1 → v2.9.4](https://github.com/fxamacker/cbor/compare/v2.9.1...v2.9.4)
- github.com/go-jose/go-jose/v4: [v4.1.4 → v4.1.5](https://github.com/go-jose/go-jose/compare/v4.1.4...v4.1.5)
- github.com/go-logr/logr: [v1.4.3 → v1.4.4](https://github.com/go-logr/logr/compare/v1.4.3...v1.4.4)
- github.com/google/cadvisor/lib: [v0.60.5 → v0.60.6](https://github.com/google/cadvisor/compare/lib/v0.60.5...lib/v0.60.6)
- github.com/mdlayher/socket: [v0.6.1 → v0.7.0](https://github.com/mdlayher/socket/compare/v0.6.1...v0.7.0)
- github.com/moby/sys/userns: [v0.1.0 → v0.2.1](https://github.com/moby/sys/compare/userns/v0.1.0...userns/v0.2.1)
- github.com/modern-go/reflect2: [35a7c28 → v1.0.2](https://github.com/modern-go/reflect2/compare/35a7c28...v1.0.2)
- github.com/onsi/ginkgo/v2: [v2.32.0 → v2.33.0](https://github.com/onsi/ginkgo/compare/v2.32.0...v2.33.0)
- github.com/onsi/gomega: [v1.40.0 → v1.44.0](https://github.com/onsi/gomega/compare/v1.40.0...v1.44.0)
- github.com/opencontainers/cgroups: [v0.0.7 → v0.1.0](https://github.com/opencontainers/cgroups/compare/v0.0.7...v0.1.0)
- github.com/sirupsen/logrus: [v1.9.4 → v1.10.2](https://github.com/sirupsen/logrus/compare/v1.9.4...v1.10.2)
- github.com/spiffe/go-spiffe/v2: [v2.6.0 → v2.8.1](https://github.com/spiffe/go-spiffe/compare/v2.6.0...v2.8.1)
- github.com/stretchr/testify: [v1.11.1 → v1.12.1](https://github.com/stretchr/testify/compare/v1.11.1...v1.12.1)
- go.etcd.io/etcd/api/v3: [v3.7.0 → v3.7.2](https://github.com/etcd-io/etcd/compare/api/v3.7.0...api/v3.7.2)
- go.etcd.io/etcd/client/pkg/v3: [v3.7.0 → v3.7.2](https://github.com/etcd-io/etcd/compare/client/pkg/v3.7.0...client/pkg/v3.7.2)
- go.etcd.io/etcd/client/v3: [v3.7.0 → v3.7.2](https://github.com/etcd-io/etcd/compare/client/v3.7.0...client/v3.7.2)
- go.etcd.io/etcd/pkg/v3: [v3.7.0 → v3.7.2](https://github.com/etcd-io/etcd/compare/pkg/v3.7.0...pkg/v3.7.2)
- go.etcd.io/etcd/server/v3: [v3.7.0 → v3.7.2](https://github.com/etcd-io/etcd/compare/server/v3.7.0...server/v3.7.2)
- go.opentelemetry.io/contrib/detectors/gcp: [v1.43.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/detectors/gcp/v1.43.0...detectors/gcp/v1.44.0)
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: [v0.68.0 → v0.71.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/instrumentation/google.golang.org/grpc/otelgrpc/v0.68.0...instrumentation/google.golang.org/grpc/otelgrpc/v0.71.0)
- go.opentelemetry.io/otel: [v1.44.0 → v1.46.0](https://github.com/open-telemetry/opentelemetry-go/compare/v1.44.0...v1.46.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: [v1.44.0 → v1.45.0](https://github.com/open-telemetry/opentelemetry-go/compare/exporters/otlp/otlptrace/v1.44.0...exporters/otlp/otlptrace/v1.45.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: [v1.44.0 → v1.45.0](https://github.com/open-telemetry/opentelemetry-go/compare/exporters/otlp/otlptrace/otlptracegrpc/v1.44.0...exporters/otlp/otlptrace/otlptracegrpc/v1.45.0)
- go.opentelemetry.io/otel/metric: [v1.44.0 → v1.46.0](https://github.com/open-telemetry/opentelemetry-go/compare/metric/v1.44.0...metric/v1.46.0)
- go.opentelemetry.io/otel/sdk: [v1.44.0 → v1.46.0](https://github.com/open-telemetry/opentelemetry-go/compare/sdk/v1.44.0...sdk/v1.46.0)
- go.opentelemetry.io/otel/sdk/metric: [v1.44.0 → v1.46.0](https://github.com/open-telemetry/opentelemetry-go/compare/sdk/metric/v1.44.0...sdk/metric/v1.46.0)
- go.opentelemetry.io/otel/trace: [v1.44.0 → v1.46.0](https://github.com/open-telemetry/opentelemetry-go/compare/trace/v1.44.0...trace/v1.46.0)
- go.opentelemetry.io/proto/otlp: [v1.10.0 → v1.11.0](https://github.com/open-telemetry/opentelemetry-proto-go/compare/otlp/v1.10.0...otlp/v1.11.0)
- go.yaml.in/yaml/v3: [v3.0.4 → v3.0.5](https://github.com/yaml/go-yaml/compare/v3.0.4...v3.0.5)
- golang.org/x/crypto: [v0.54.0 → v0.57.0](https://go.googlesource.com/crypto/+/cdce021fa6c7d9c7eb2743bfbe551f0a98fd5d62^1..3f62bf119e84c6e35e8518a2958089ade622d1a3/)
- golang.org/x/exp: [746e56f → 85c1c22](https://go.googlesource.com/exp/+/746e56fc9e2fafde18176275ce0b96b06ac53955^1..85c1c2202aba13ab812be55279024d8594344c1f/)
- golang.org/x/mod: [v0.37.0 → v0.41.0](https://go.googlesource.com/mod/+/deb1dfcdb7c7fd98fb5afddc3e95dd36d5880874^1..d0a27b2d4a48460806692bf5c87fc157c3c65292/)
- golang.org/x/net: [v0.57.0 → v0.59.0](https://go.googlesource.com/net/+/b8f09f6f062ceb4531b7af4bd17a5c8fe9c4b2b5^1..540d04cfe5028e2655754591a4d3e08c586809f2/)
- golang.org/x/oauth2: [v0.36.0 → v0.37.0](https://go.googlesource.com/oauth2/+/4d954e69a88d9e1ccb8439f8d5b6cbef230c4ef9^1..c624b89dadc3221560b7345c090bbe69e90808ee/)
- golang.org/x/sync: [v0.22.0 → v0.23.0](https://go.googlesource.com/sync/+/1eb64d4bc0cde6da1bb8ebc7f178bb577508e5d0^1..f75267d8412fc1dfd12b343644a7ea46e4d9c85d/)
- golang.org/x/sys: [v0.47.0 → v0.48.0](https://go.googlesource.com/sys/+/9e7e939dcafac07e8ab4cffa6e5fc74908413f00^1..613e2570718ecde85c04e69ebd5585c3881c442c/)
- golang.org/x/telemetry: [59b4966 → 4bcc4b2](https://go.googlesource.com/telemetry/+/59b4966ccb57499277814ee2272936a2c01cfbcd^1..4bcc4b2ee518b727c591c4c8efd19bfaca66d618/)
- golang.org/x/term: [v0.45.0 → v0.46.0](https://go.googlesource.com/term/+/9f69229da31ca6a34b522f59dbe07cad5ea21587^1..6226200ed12cba417a9d9e799c2a7179d3fc0e27/)
- golang.org/x/text: [v0.40.0 → v0.42.0](https://go.googlesource.com/text/+/724af9c35838492dcaacc1ac51a8a0187c994c54^1..fafe4a06967e06550e69ee42787d9902845d2a3f/)
- golang.org/x/time: [v0.15.0 → v0.16.0](https://go.googlesource.com/time/+/812b343c8714c317b0dad633efa6d103e554c006^1..fb013b3d305a26f5ef5350a6eaffc7c87a200383/)
- golang.org/x/tools: [v0.47.0 → v0.50.0](https://go.googlesource.com/tools/+/fbf9f2e2c8124fbe1877f5ed2857111038d9fe12^1..265dd1a6ecf0ee85548c7a8d1787d25fc5675e06/)
- google.golang.org/genproto/googleapis/api: [3dc84a4 → 6ac0973](https://github.com/googleapis/go-genproto/compare/3dc84a4a5aaa87331e10f51e22e90d961f986894...6ac0973c030de548f6a485902facf1e906a58fbf)
- google.golang.org/genproto/googleapis/rpc: [3dc84a4 → da73d73](https://github.com/googleapis/go-genproto/compare/3dc84a4a5aaa87331e10f51e22e90d961f986894...da73d73af1c5183531788b6e26a2d1ce2c9faab7)
- google.golang.org/grpc: [v1.82.1 → v1.84.0](https://github.com/grpc/grpc-go/compare/v1.82.1...v1.84.0)
- google.golang.org/protobuf: [f2248ac → v1.36.12](https://go.googlesource.com/protobuf/+/f2248ac996afc39b3df0777cdcc269f6ade50b07^1..cdd4c5f7406e82462949c7a65defa9f3029c162d/)
- k8s.io/kube-openapi: [d427ff9 → c4db2bd](https://github.com/kubernetes/kube-openapi/compare/d427ff9ee9ad05f5da435abbb7c5929cb713ac56...c4db2bdfbfe686300282ac9c6c7c654f70625e81)
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: [v0.36.0 → v0.37.0](https://github.com/kubernetes-sigs/apiserver-network-proxy/compare/konnectivity-client/v0.36.0...konnectivity-client/v0.37.0)
- tags.cncf.io/container-device-interface/specs-go: [v1.1.0 → v1.1.1](https://github.com/cncf-tags/container-device-interface/compare/specs-go/v1.1.0...specs-go/v1.1.1)

### Removed
- github.com/coreos/go-oidc: [v2.5.0](https://github.com/coreos/go-oidc/commit/153fc73f601ff388edee90ce864c564ed5195695)
- github.com/google/gofuzz: [v1.0.0](https://github.com/google/gofuzz/tree/v1.0.0)
- github.com/kr/pty: [v1.1.1](https://github.com/kr/pty/tree/v1.1.1)
- github.com/peterbourgon/diskv: [v2.0.1](https://github.com/peterbourgon/diskv/tree/v2.0.1)
- github.com/pkg/diff: [20ebb0f](https://github.com/pkg/diff/tree/20ebb0f)
- github.com/pquerna/cachecontrol: [v0.1.0](https://github.com/pquerna/cachecontrol/tree/v0.1.0)
- go.opentelemetry.io/otel/metric/x: [v0.66.0](https://github.com/open-telemetry/opentelemetry-go/commit/b62d92831b2dd142f5a0cc89c828270274196877)
- gopkg.in/go-jose/go-jose.v2: v2.6.3
- sigs.k8s.io/structured-merge-diff/v6: [v6.4.2](https://github.com/kubernetes-sigs/structured-merge-diff/commit/15d075473efece0367330fd20a7cc2aad2eeb998)