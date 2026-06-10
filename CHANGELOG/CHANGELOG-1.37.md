<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.37.0-alpha.1](#v1370-alpha1)
  - [Downloads for v1.37.0-alpha.1](#downloads-for-v1370-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.36.0](#changelog-since-v1360)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Dependency](#dependency)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)

<!-- END MUNGE: GENERATED_TOC -->

# v1.37.0-alpha.1


## Downloads for v1.37.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes.tar.gz) | 9fd5423640e5935366e023dd22c62e9fca12405937c2411e81cc3ca2fc28255a43984bb736a2c4cc30189791d30a79b62e82aa5a80b065ce5b32171a95942a61
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-src.tar.gz) | e142c1afc4ff99f21bc928354d3eca1508c8ed34cc98952bf75b2ab6171da83864e15218a4669bc555964cbcbddc759ba12ac7cb989a2afa00312e03e56f9fab

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | adcd35338556cfb80dee5a093d769a89d551609dd8b1d8f5c075bdcf1abd956f13fb1a90b262f47bd99c2b5d8b89db018dbaa818881021d44fef7e072898f09c
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 02e2a54db5dc24b8661e44030e09695078a64a615ca94862edbd222a4c5955c1fa2ae9df6d54329d93b3e329720d6f097f9f32dee0118e612df225ee16d67755
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 9cc9c69c6d76a6afb214a72a5fb06995ed91f6026cc592e64aa94eff3c76a119fc486ddb2ce6a2ca6ab4adf17d48ceb89aae4cd801c572b9128c9e968041e753
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 6979e554e427dd993b32ca218cfc727f6bccbdc50981251626ab9b1a72c22b270467d27ebcfa7cb6ece7b41d6584e42086886f76a74be3c5cd79b0f4cb5c1c4c
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 6c925eb12c287a5cc8860d639176f814255f91bb3f28e4c526137474231e47cbfa2e67176ead7d783a77633b104cf976a523d07f9b784470555c0b5e1c53adc6
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 37a440504570ccdf8a247b6e148c5b4c249697bd4f42546746de5fadd11bab393a56802c92c363d77e621604721560282d1f482a463e546102e41d2a032c95ac
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 08fe5ef61e41ac33c39492e527ebf292e8758fc319e2a3dc8535d454de16161a1c8e77c1c988f44d5b0f5ccab14f97a7587ebaa633ea794e39ac17e18f8288cc
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 5c1a0e0a94da2a70c67c2a0e3ecc94c1c914d5bbcc49f81b62e9c3141aa3661e59e8e73f908738674cbbbbdee3fea88636f4e1bf742309db86bc12c916744e66
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 7bc268f0352c671e7c30aadcc0af3976960cb0f43efcafd3966ebb99de236a88c53d04deed9fb831b891440319e280239ab9bcc1d9eecb27d5f665ccdf8f143e
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | cd44efdb02873ea13792931621882bb4c8834839874e2aa6cb97293995212a3fa09237dffb014b9510dc09b8638d71e259d44830bdd8b52315c33f5db6c798ce
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | e377ffb371bb61de9e93d8f3f4048768a1ff28bad8682f80cc520e477e1a43e01b5d0c076fdaf8d44ed464cc0dfb9a97f3ecf33aed2fd3b062221278a9ebe2c4

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | e88c3f6312ae9567706c0c9ea4558c32ba1e86523d07d1d12d5d8460aa759db44f84c7949806a1c2cddde8cc482d99cace4aca88685e20a5ddfac32f9de76386
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | adf4468c0bb18b2d2ea315dbebf55eb2ed17cd1eb951cafc180b141f0a6fe9302f6c446d760aedc177fb9c14e9cdaf7fd59f8fed9b2511fb713aa33d28c6f847
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 26e52f7ce483fbef715162a98ff7506fdd6ee659c2dabfd0a3600e63e1b0a31762451093c3ed2683e50287ba82b057d19388f061e2f06a92a5cc4398ac62f128
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | d19ab25c6914aa6dd2b74398df9d455d37c40784520394399c57a8416a811fc16d4becc048168def681f3c964089ef0ce2de8074d9029829156d9e344ae4feca

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 6e27d1b85181f3642a8c1dcf39243ce6a6c54913529ddf4a0a42884c233da462d3cc0db7bc223272889c2281f7a4d2be69c6327fe49bf9827331373168e818f1
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | f1a6bc663c3226bba06b1a78580e27e1de2f311832e94c30213933b97cbb8ad66a75e6707942cf4aaa8b30c4d2313a799c52b019468783ebaca37874c04bb207
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 7e27887af5ef9c20d761ba43a2ec593569962f5aa8a2d854d73cf8ed88b94833f841359a622861817c2759fad4c1ae00bb54c55964168a588da29458aa3530ab
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 01013b8b020c46562269e1137aaecbfb6171468a08e3ba38238cdb1df0ad46bac8f3e138fb6c8d3c14b81041cce99e409d09cd54a932fb04036cba70673b85fe
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.37.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 6ae104549e49dff89004b252ad0712c137f33b51ffbc56ff0b3ad3147ba6608c8d76ed1e08fb7372a12b14c7ba75d88801e45e9013b433149805a17ca61f7f66

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.37.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.37.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.37.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.37.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.37.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.37.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.36.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - When eventRecordQPS in kubelet configuration file is set to 0, there will be no limit enforced. The bug when 0 value was treated as a default value, while the field description was saying "unlimited" is fixed.
  if your kubelet configuration setting eventRecordQPS to 0 and you want to preserve the previous behavior, please change the value to 50. Keeping it as 0 will make it undefined. ([#117119](https://github.com/kubernetes/kubernetes/pull/117119), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery, Auth and Node]
 
## Changes by Kind

### Dependency

- Updated the default etcd version to 3.7.0-rc.0 ([#139427](https://github.com/kubernetes/kubernetes/pull/139427), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Updates the etcd client library to v3.6.10 ([#138393](https://github.com/kubernetes/kubernetes/pull/138393), [@humblec](https://github.com/humblec)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling and Storage]

### Deprecation

- Deprecated the ignored `--filename`/`-f` flag on `kubectl run`. ([#138671](https://github.com/kubernetes/kubernetes/pull/138671), [@Suknna](https://github.com/Suknna)) [SIG CLI]
- Kubeadm: added a (delayed) warning that kube-proxy's 'ipvs' mode is deprecated since v1.35 and users on newer Linux kernels should be using the 'nftables' mode instead, which became GA in 1.33. For older kernel versions, users can use 'iptables', which is still the default. ([#139067](https://github.com/kubernetes/kubernetes/pull/139067), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- The deprecated `DeclarativeValidationTakeover` feature gate is now locked to its default value and can no longer be set. ([#139212](https://github.com/kubernetes/kubernetes/pull/139212), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery]

### API Change

- API Go types switched the json tag for inlined TypeMeta fields from `",inline"` to simply `""`. `inline` was not a recognized json serializer option and did not modify marshal or unmarshal behavior. ([#138260](https://github.com/kubernetes/kubernetes/pull/138260), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Converts the `DisruptionMode` enum field to struct to support future extensibility.
  Promotes the `scheduling.k8s.io` API group from `v1alpha2` to `v1alpha3` and drops `v1alpha2` entirely.
  Remember to remove all `v1alpha2` objects from the api-server while performing the cluster update. ([#138572](https://github.com/kubernetes/kubernetes/pull/138572), [@dom4ha](https://github.com/dom4ha)) [SIG API Machinery, Apps, CLI, Etcd, Node, Scheduling and Testing]
- DRA extended resource feature is promoted to GA in 1.37 ([#138488](https://github.com/kubernetes/kubernetes/pull/138488), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Fixes a 1.34+ regression handling containers with environment values set from Secret API objects containing binary non-utf8 data. ([#139168](https://github.com/kubernetes/kubernetes/pull/139168), [@liggitt](https://github.com/liggitt)) [SIG Architecture, Node and Testing]
- HorizontalPodAutoscaler conditions allow optionally including the `observedGeneration` at the time the condition was recorded ([#138653](https://github.com/kubernetes/kubernetes/pull/138653), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG API Machinery, Apps, Autoscaling and Testing]
- Improved CEL error messages in Dynamic Resource Allocation to provide guidance when accessing non-existent device attributes. Error messages now link to documentation on handling optional fields using orValue() and has(). ([#136709](https://github.com/kubernetes/kubernetes/pull/136709), [@gzb1128](https://github.com/gzb1128)) [SIG API Machinery, Node and Scheduling]
- Promoted kubelet volume metrics (`storage_operation_duration_seconds`, `volume_operation_total_seconds`) from Alpha to Beta stability, providing stronger API and label stability guarantees for metric consumers. ([#136189](https://github.com/kubernetes/kubernetes/pull/136189), [@bhope](https://github.com/bhope)) [SIG Instrumentation and Storage]
- Removed the generally available feature gate `AnyVolumeDataSource`, which was locked and enabled since 1.33. ([#135336](https://github.com/kubernetes/kubernetes/pull/135336), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Storage and Testing]
- Removed the unused `PodStatusResult` type from the Kubernetes API. This type had no REST endpoint and has been unused since 2015. ([#136271](https://github.com/kubernetes/kubernetes/pull/136271), [@adityasharmawork](https://github.com/adityasharmawork)) [SIG API Machinery, Apps, Node and Testing]
- The change is for developers building against cri-api. Enum keys of Signal are now prefixed with `SIGNAL_` in api.proto definition to avoid conflicts with C++ macroses. The wire format is unchanged. ([#139251](https://github.com/kubernetes/kubernetes/pull/139251), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Apps, Node and Testing]

### Feature

- Add new KubeProxyIPVS feature gate in preparation of deactivating and then removing the ipvs mode of kube-proxy. ([#139397](https://github.com/kubernetes/kubernetes/pull/139397), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network]
- Added Prometheus metrics for Windows kube-proxy (winkernel) load balancer operation failures: `kubeproxy_sync_proxy_rules_winkernel_lb_create_failures_total`, `kubeproxy_sync_proxy_rules_winkernel_lb_update_failures_total`, and `kubeproxy_sync_proxy_rules_winkernel_lb_delete_failures_total`. Each metric includes `ip_family`, `lb_type`, and `error` labels for fine-grained failure observability. ([#137767](https://github.com/kubernetes/kubernetes/pull/137767), [@princepereira](https://github.com/princepereira)) [SIG Instrumentation, Network and Windows]
- Added ServiceName, PodManagementPolicy, and PersistentVolumeClaimRetentionPolicy to `kubectl describe statefulset` output. ([#137547](https://github.com/kubernetes/kubernetes/pull/137547), [@kfess](https://github.com/kfess)) [SIG CLI]
- Added `AnnotatedEventf` method to the new events API (`EventRecorder` and `EventRecorderLogger` interfaces in `client-go/tools/events`), enabling callers to attach custom annotations to events at creation time. ([#138103](https://github.com/kubernetes/kubernetes/pull/138103), [@adri1197](https://github.com/adri1197)) [SIG API Machinery and Node]
- Added `net.ipv4.tcp_slow_start_after_idle` and `net.ipv4.tcp_notsent_lowat` to the allowed safe sysctls list. ([#138389](https://github.com/kubernetes/kubernetes/pull/138389), [@gheffern](https://github.com/gheffern)) [SIG Auth, Network and Node]
- Added an alpha feature gate, ConsistentListFromCacheSkipTimeoutFallback. When enabled, kube-apiserver returns HTTP 429 for consistent LIST requests that cannot be served from watch cache within the timeout window, instead of falling back to storage. ([#138701](https://github.com/kubernetes/kubernetes/pull/138701), [@yedou37](https://github.com/yedou37)) [SIG API Machinery]
- Added metric `apiserver_watch_cache_initialization_duration_seconds` recording the duration of the most recent watch cache initialization, labeled by group and resource. ([#138767](https://github.com/kubernetes/kubernetes/pull/138767), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Instrumentation]
- Added scheduler extension point "PlacementFeasible" to allow for early termination of PodGroup scheduling cycle. This extension point is used by the GangScheduling plugin to stop evaluating pods once minCount becomes unsatisfiable. ([#138643](https://github.com/kubernetes/kubernetes/pull/138643), [@brejman](https://github.com/brejman)) [SIG Scheduling and Testing]
- Added structured `CauseType` values to PodDisruptionBudget-related eviction `Forbidden` errors in the eviction API, allowing clients to programmatically distinguish PDB invalid-state errors from other forbidden errors without string-matching on the message. ([#138003](https://github.com/kubernetes/kubernetes/pull/138003), [@shady0503](https://github.com/shady0503)) [SIG Apps, Auth and Node]
- Added support for testing invariant metrics in integration tests. ([#137883](https://github.com/kubernetes/kubernetes/pull/137883), [@lalitc375](https://github.com/lalitc375)) [SIG API Machinery, Auth and Testing]
- Added the `EtcdRangeStream` beta feature gate. The watch cache initializes by streaming objects from etcd in a single `RangeStream` RPC instead of paginated `Range` requests. ([#136915](https://github.com/kubernetes/kubernetes/pull/136915), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Adds the `+k8s:dependentRequired("siblingJSONName")` declarative validation tag. When the tagged field is set, the named sibling must also be set ([#139164](https://github.com/kubernetes/kubernetes/pull/139164), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery]
- After successful pod group preemption, the pods from pod group will have their Nominated Node Name set, similarly to pod preemption ([#138967](https://github.com/kubernetes/kubernetes/pull/138967), [@antekjb](https://github.com/antekjb)) [SIG Scheduling and Testing]
- Apply --field-selector to pod metrics when invoking kubectl top pod ([#139107](https://github.com/kubernetes/kubernetes/pull/139107), [@Mujib-Ahasan](https://github.com/Mujib-Ahasan)) [SIG CLI]
- Binding API calls in kube-scheduler are now retried when a transient error occurs. ([#138855](https://github.com/kubernetes/kubernetes/pull/138855), [@antekjb](https://github.com/antekjb)) [SIG Scheduling]
- Bump coredns to 1.14.3 ([#138536](https://github.com/kubernetes/kubernetes/pull/138536), [@yashsingh74](https://github.com/yashsingh74)) [SIG Cloud Provider and Cluster Lifecycle]
- Changed the `PatchPodStatus` API in the scheduler framework to accept a slice of Pod conditions (`[]*v1.PodCondition`) instead of a single condition (`*v1.PodCondition`). This allows scheduler plugins to update multiple Pod conditions in a single API call, preventing newer calls from overwriting older ones when multiple conditions need to be updated concurrently. ([#135160](https://github.com/kubernetes/kubernetes/pull/135160), [@KunWuLuan](https://github.com/KunWuLuan)) [SIG Scheduling]
- Ensure stale cache does not impact the marking of nodes as unhealthy by checking with a live get ([#138698](https://github.com/kubernetes/kubernetes/pull/138698), [@michaelasp](https://github.com/michaelasp)) [SIG Apps, Auth and Node]
- Errors coming from pod group preemption are now prefixed with `pod group preemption:` message. ([#139218](https://github.com/kubernetes/kubernetes/pull/139218), [@Argh4k](https://github.com/Argh4k)) [SIG Scheduling]
- Functions and structs that take in authorizer.Authorizer might now choose to accept only a smaller interface, authorizer.UnconditionalAuthorizer, in case only the receiver only needs to perform unconditional authorization requests and wants to signal this in the code for clarity. Any authorizer implementation must still implement the full authorizer.Authorizer interface. ([#138801](https://github.com/kubernetes/kubernetes/pull/138801), [@luxas](https://github.com/luxas)) [SIG API Machinery, Auth, Node, Scheduling and Testing]
- Graduate WatchCacheInitializationPostStartHook to GA ([#139452](https://github.com/kubernetes/kubernetes/pull/139452), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Kube-controller-manager: The HPA controller now defers syncing an HPA object when the controller has not yet observed HPA status writes from the last time the object was synced. ([#139025](https://github.com/kubernetes/kubernetes/pull/139025), [@omerap12](https://github.com/omerap12)) [SIG Apps and Autoscaling]
- Kube-scheduler now supports PodGroups in its scheduling queue. The active, backoff, and unschedulable queues have been abstracted to store `QueuedEntityInfo` (handling either individual pods or pod groups). ([#138567](https://github.com/kubernetes/kubernetes/pull/138567), [@macsko](https://github.com/macsko)) [SIG Instrumentation, Scheduling and Testing]
- Kube-scheduler: Added `PlacementCycleState` to the scheduling framework, providing per-placement state to `PlacementScore` plugins under the alpha `TopologyAwareWorkloadScheduling` feature gate. ([#138274](https://github.com/kubernetes/kubernetes/pull/138274), [@wtravO](https://github.com/wtravO)) [SIG Scheduling]
- Kubeadm: add the "kubeproxydaemonset" patch target to allow patching the kube-proxy DaemonSet during "kubeadm init" and "kubeadm upgrade", consistent with the existing "corednsdeployment" patch target. ([#138090](https://github.com/kubernetes/kubernetes/pull/138090), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: during preflight, instead of running the "Port-xx" checks for kube-apiserver, kube-scheduler, kube-controller-manager and etcm using a "net.Listen()" call without an address (which instructs the operating system to bind to all available unicast and anycast IP addresses for a given port), pass an address which is configured in the kubeadm config for the respective components either using the "localAPIEndpoint.address" field or using the "--bind-address" extraArgs override. ([#138250](https://github.com/kubernetes/kubernetes/pull/138250), [@lentzi90](https://github.com/lentzi90)) [SIG Cluster Lifecycle]
- Kubeadm: removed the NodeLocalCRISocket feature gate which graduated to GA and was locked to enabled by default in a previous release. ([#138645](https://github.com/kubernetes/kubernetes/pull/138645), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: the preflight check `ContainerRuntimeVersion` validates if the installed container runtime supports the `RuntimeConfig` gRPC method. For older kubelet versions than 1.38, it will return a preflight warning. ([#139122](https://github.com/kubernetes/kubernetes/pull/139122), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubectl now sets its path in the KUBECTL_PATH environment variable when executing a plugin. ([#138694](https://github.com/kubernetes/kubernetes/pull/138694), [@brianpursley](https://github.com/brianpursley)) [SIG CLI and Testing]
- Kubelet: defer the configurations flags (and the related fallback behavior) deprecation removal timeline from 1.37 to 1.38 to align with containerd v1.7 support ([#139121](https://github.com/kubernetes/kubernetes/pull/139121), [@carlory](https://github.com/carlory)) [SIG Node and Testing]
- Kubernetes is now built using Go 1.26.4 ([#139584](https://github.com/kubernetes/kubernetes/pull/139584), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with Go 1.26.3 ([#138864](https://github.com/kubernetes/kubernetes/pull/138864), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Kubernetes is now built with Go 1.26.4 ([#139479](https://github.com/kubernetes/kubernetes/pull/139479), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Made it possible for authorizers to return conditional decisions in addition to unconditional (Allow/Deny/NoOpinion). ([#137204](https://github.com/kubernetes/kubernetes/pull/137204), [@luxas](https://github.com/luxas)) [SIG API Machinery, Auth, Node, Scheduling and Testing]
- Optimized CEL admission policy evaluation by adopting a lazy zero-allocation reflection-based utility for object traversal, significantly reducing CPU usage and garbage collection overhead during request processing. ([#138771](https://github.com/kubernetes/kubernetes/pull/138771), [@lalitc375](https://github.com/lalitc375)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Network, Node, Scheduling and Storage]
- Promote `serviceaccount_legacy_tokens_total`, `serviceaccount_stale_tokens_total`, `serviceaccount_valid_tokens_total` to beta ([#137072](https://github.com/kubernetes/kubernetes/pull/137072), [@tico88612](https://github.com/tico88612)) [SIG Auth, Instrumentation and Testing]
- Promote the apiserver webhook `apiserver_webhooks_x509_missing_san_total` and `apiserver_webhooks_x509_insecure_sha1_total` metrics to BETA and update their documentation. ([#136894](https://github.com/kubernetes/kubernetes/pull/136894), [@LoginovIlia](https://github.com/LoginovIlia)) [SIG API Machinery, Instrumentation and Testing]
- The MaxUnavailableStatefulSet feature is now enabled by default. ([#139466](https://github.com/kubernetes/kubernetes/pull/139466), [@soltysh](https://github.com/soltysh)) [SIG Apps]
- The `apiserver_storage_list_*` metrics now include `storage` and `index` labels to distinguish the storage backend and lookup path used to serve LIST requests. ([#139125](https://github.com/kubernetes/kubernetes/pull/139125), [@yedou37](https://github.com/yedou37)) [SIG API Machinery, Etcd and Instrumentation]
- The scheduler now avoids redundant preemption attempts during PodGroup scheduling when terminating victim pods are already present on the nominated nodes. ([#138710](https://github.com/kubernetes/kubernetes/pull/138710), [@mm4tt](https://github.com/mm4tt)) [SIG Scheduling]
- Three different subtypes of the cluster event resource "Pod" are being added: "AssignedPod", "UnscheduledPod", "TargetPod". Plugins can and are expected to register to specific pod events for better performance. ([#135905](https://github.com/kubernetes/kubernetes/pull/135905), [@iomarsayed](https://github.com/iomarsayed)) [SIG Node, Scheduling, Storage and Testing]
- Updated cri-tools to v1.36.0. ([#138613](https://github.com/kubernetes/kubernetes/pull/138613), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- Workload-aware preemption now preempts victims so that as many as possible of the preemptor pods can be scheduled. ([#138757](https://github.com/kubernetes/kubernetes/pull/138757), [@jdzikowski](https://github.com/jdzikowski)) [SIG Scheduling and Testing]
- `kubectl get crd` now displays additional columns—GROUP, SCOPE, VERSIONS, and CREATED AT—alongside NAME.  
  
  This provides at-a-glance visibility into the API group, scope (Cluster‑ or Namespaced), served versions (comma‑separated), and exact creation timestamp of each CustomResourceDefinition. ([#131599](https://github.com/kubernetes/kubernetes/pull/131599), [@jaehanbyun](https://github.com/jaehanbyun)) [SIG API Machinery]

### Documentation

- Fixed a nil pointer dereference panic in client-go event recorder when processing events with nil fields.
  
    Additional documentation e.g., KEPs (Kubernetes Enhancements Proposals), usage docs, etc.:
  
    N/A
  
    /sig api-machinery
    /area client-libraries
    /priority important-soon ([#135925](https://github.com/kubernetes/kubernetes/pull/135925), [@jianzhangbjz](https://github.com/jianzhangbjz)) [SIG API Machinery]
- Update Japanese translation for kubectl ([#131176](https://github.com/kubernetes/kubernetes/pull/131176), [@yude](https://github.com/yude)) [SIG CLI and Testing]

### Failing Test

- Fixed a bug, where nomination of gated pod wasn't preventing lower priority pods from scheduling on the nominated space. ([#139057](https://github.com/kubernetes/kubernetes/pull/139057), [@macsko](https://github.com/macsko)) [SIG Scheduling]

### Bug or Regression

- Avoid costly comparisons during selinux metric emission. ([#138981](https://github.com/kubernetes/kubernetes/pull/138981), [@gnufied](https://github.com/gnufied)) [SIG Apps and Storage]
- Client-go: RetryWatcher now logs 410 Gone (resource expired) errors at debug verbosity (V(4)) instead of ERROR level during watch establishment. ([#138295](https://github.com/kubernetes/kubernetes/pull/138295), [@kencochrane](https://github.com/kencochrane)) [SIG API Machinery]
- DRA metadata read helper (KEP-5304): on decode skip only if version is unknown, return error if object/file is malformed ([#138530](https://github.com/kubernetes/kubernetes/pull/138530), [@alaypatel07](https://github.com/alaypatel07)) [SIG Node]
- Exposes the error reason when invalid service CIDRs are configured ([#139182](https://github.com/kubernetes/kubernetes/pull/139182), [@PseudoResonance](https://github.com/PseudoResonance)) [SIG Network]
- Fix a regression in kubernetes v1.35, where with a Parallel pod management policy unavailable pods from an older revision were incorrectly counted towards maxUnavailable budget. ([#137666](https://github.com/kubernetes/kubernetes/pull/137666), [@soltysh](https://github.com/soltysh)) [SIG Apps]
- Fix apiserver to create metadata fields for create-via-update and created-via-apply requests like they are for create requests. UID and resourceVersion preconditions are still honored. ([#138908](https://github.com/kubernetes/kubernetes/pull/138908), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]
- Fix duplicated mount arguments in log string output from MakeMountArgsSensitiveWithMountFlags ([#138098](https://github.com/kubernetes/kubernetes/pull/138098), [@jeffbearer](https://github.com/jeffbearer)) [SIG Storage]
- Fix issue with stateful set controller skip metrics not being properly registered. ([#138451](https://github.com/kubernetes/kubernetes/pull/138451), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps and Testing]
- Fix nil pointer dereference in Windows memory eviction threshold notifier when GetPerformanceInfo() fails. ([#138727](https://github.com/kubernetes/kubernetes/pull/138727), [@rzlink](https://github.com/rzlink)) [SIG Node]
- Fix regression in kubectl resource printing on bigger data sets (100+ rows) ([#138550](https://github.com/kubernetes/kubernetes/pull/138550), [@rawkode](https://github.com/rawkode)) [SIG CLI]
- Fixed VolumeAttachment validation to report the correct maximum message size (1024 bytes) in error messages. ([#136436](https://github.com/kubernetes/kubernetes/pull/136436), [@Okabe-Junya](https://github.com/Okabe-Junya)) [SIG Storage]
- Fixed a bug in ImageLocality scoring where image volumes could receive a higher score than equivalent regular container images. ([#138951](https://github.com/kubernetes/kubernetes/pull/138951), [@sujoshua](https://github.com/sujoshua)) [SIG Scheduling]
- Fixed a bug when the GenericWorkload feature gate is enabled that could prevent Pods in the same PodGroup sharing the same ResourceClaim from successfully scheduling. ([#139418](https://github.com/kubernetes/kubernetes/pull/139418), [@nojnhuh](https://github.com/nojnhuh)) [SIG Node and Scheduling]
- Fixed a bug where Pod `.status.resourceClaimStatuses` could flap between partial lists of claims, when multiple claims were used in the pod. ([#138408](https://github.com/kubernetes/kubernetes/pull/138408), [@johnbelamaric](https://github.com/johnbelamaric)) [SIG Apps and Node]
- Fixed a bug where Pods that share multi-node claims and also have per-node claims can get stuck in Pending. ([#139017](https://github.com/kubernetes/kubernetes/pull/139017), [@johnbelamaric](https://github.com/johnbelamaric)) [SIG Node and Scheduling]
- Fixed a bug where StatefulSet with OnDelete update strategy never updated
  Status.CurrentRevision to match Status.UpdateRevision after all pods were
  recreated with the new revision. ([#136833](https://github.com/kubernetes/kubernetes/pull/136833), [@zhijun42](https://github.com/zhijun42)) [SIG Apps]
- Fixed a bug where `kubectl drain --disable-eviction --dry-run=server` hangs indefinitely. ([#137543](https://github.com/kubernetes/kubernetes/pull/137543), [@kfess](https://github.com/kfess)) [SIG CLI and Testing]
- Fixed a bug where disabling the MemoryQoS feature gate did not clear per-container memory.high cgroup values, causing containers to remain throttled at stale limits. ([#139377](https://github.com/kubernetes/kubernetes/pull/139377), [@sohankunkerkar](https://github.com/sohankunkerkar)) [SIG Node and Testing]
- Fixed a bug where kubelet would generate an event once per second for every image volume in a pod. ([#138655](https://github.com/kubernetes/kubernetes/pull/138655), [@mdbooth](https://github.com/mdbooth)) [SIG Node]
- Fixed a bug where pods with multiple subPath volume mounts on Windows would get stuck in Terminating state because file handles from subPath preparation were leaked, preventing volume cleanup. ([#138367](https://github.com/kubernetes/kubernetes/pull/138367), [@timmy-wright](https://github.com/timmy-wright)) [SIG Node, Testing and Windows]
- Fixed a bug where the kubelet did not enforce per-container ephemeral-storage limits on restartable init containers (sidecar containers), allowing them to exceed their declared limit without triggering pod eviction. ([#138462](https://github.com/kubernetes/kubernetes/pull/138462), [@shachartal](https://github.com/shachartal)) [SIG Node and Testing]
- Fixed a kube-proxy IPVS-mode performance bug where `syncProxyRules` could take tens of seconds in clusters with many Services because `GetAllLocalAddressesExcept` issued one full netlink address dump per interface. The function now issues a single dump per address family, reducing `syncProxyRules` latency by orders of magnitude on large clusters. ([#138927](https://github.com/kubernetes/kubernetes/pull/138927), [@ytcisme](https://github.com/ytcisme)) [SIG Network]
- Fixed a kube-scheduler panic when a DRA ResourceClaim using `allocationMode: All` selects a device that consumes shared counters. ([#138885](https://github.com/kubernetes/kubernetes/pull/138885), [@takonomura](https://github.com/takonomura)) [SIG Node]
- Fixed a kubelet panic in image pull credential verification when maxParallelImagePulls is configured above 31. ([#138937](https://github.com/kubernetes/kubernetes/pull/138937), [@RajvardhanPatil07](https://github.com/RajvardhanPatil07)) [SIG Node]
- Fixed a panic in the endpoint controller when processing services with empty IPFamilies field (pre-dual-stack services that were never spec-updated). ([#138736](https://github.com/kubernetes/kubernetes/pull/138736), [@rahulbabu95](https://github.com/rahulbabu95)) [SIG Apps and Network]
- Fixed a race condition in preemption, where a preemptor pod could get stuck in unschedulable state. ([#139162](https://github.com/kubernetes/kubernetes/pull/139162), [@brejman](https://github.com/brejman)) [SIG Scheduling and Testing]
- Fixed a regression in 1.36 where modifications to scheduling directives (nodeSelector, tolerations, node affinity) on suspended Jobs were rejected if the JobSuspended condition had not yet been set by the job controller. ([#139287](https://github.com/kubernetes/kubernetes/pull/139287), [@kannon92](https://github.com/kannon92)) [SIG Apps and Testing]
- Fixed a regression where kubelet did not clear stale cgroup v2 memory.min and memory.low values when the MemoryQoS feature gate was disabled after being previously enabled. ([#138903](https://github.com/kubernetes/kubernetes/pull/138903), [@sohankunkerkar](https://github.com/sohankunkerkar)) [SIG Node and Testing]
- Fixed an issue in the CronJob controller where it failed to adopt existing Jobs by erroneously using the empty namespace from the JobTemplate. ([#136920](https://github.com/kubernetes/kubernetes/pull/136920), [@ysam12345](https://github.com/ysam12345)) [SIG Apps]
- Fixed an issue where kubelet would delete the CSI mount directory when
  a periodic NodePublishVolume call (triggered by
  CSIDriver.spec.requiresRepublish=true) returned an error, leaving the
  pod with stale volume contents that subsequent successful republishes
  could not repair. ([#139045](https://github.com/kubernetes/kubernetes/pull/139045), [@aramase](https://github.com/aramase)) [SIG Storage]
- Fixed build for test/images/glibc-dns-testing ([#138877](https://github.com/kubernetes/kubernetes/pull/138877), [@BenTheElder](https://github.com/BenTheElder)) [SIG Network and Testing]
- Fixed duplicate logs when trying to attach to a pod fails. ([#139091](https://github.com/kubernetes/kubernetes/pull/139091), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Fixed incorrect error message formatting in the HPA controller when object metric retrieval fails. Error messages now correctly display the metric name, object kind, namespace, object name, and the underlying error. Also improved error wrapping across the HPA controller to use %w instead of %v, enabling proper error chain inspection. ([#139029](https://github.com/kubernetes/kubernetes/pull/139029), [@Fedosin](https://github.com/Fedosin)) [SIG Apps and Autoscaling]
- Fixed kubectl get storageclass to show only the effective default StorageClass as "(default)" when multiple StorageClasses have the default annotation. ([#135964](https://github.com/kubernetes/kubernetes/pull/135964), [@jaehanbyun](https://github.com/jaehanbyun)) [SIG CLI and Storage]
- Fixed kubelet failure starting on ZFS due to missing cadvisor plugin. ([#138587](https://github.com/kubernetes/kubernetes/pull/138587), [@BenTheElder](https://github.com/BenTheElder)) [SIG Node]
- Fixed queue hint for inter-pod anti-affinity in case there are multiple terms, which might have caused delays in scheduling. ([#139161](https://github.com/kubernetes/kubernetes/pull/139161), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fixed stale remote HNS endpoint cleanup on Windows when a pod IP is reused across nodes in L2Bridge networks, preventing DNS timeouts caused by traffic being routed to the wrong node. ([#138000](https://github.com/kubernetes/kubernetes/pull/138000), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fixed the inconsistency between opportunistic batching and PodGroups that made the batching hints always infeasible during PodGroup scheduling cycle. ([#138754](https://github.com/kubernetes/kubernetes/pull/138754), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Fixed the wrong cause of the UnexpectedJob event/warning by checking the owner reference of the job correctly in the cron job controller. ([#133313](https://github.com/kubernetes/kubernetes/pull/133313), [@kei01234kei](https://github.com/kei01234kei)) [SIG Apps]
- Generate `metadata.generation` and `status.observedGeneration` fields in HorizontalPodAutoscaler resources ([#138228](https://github.com/kubernetes/kubernetes/pull/138228), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG API Machinery, Apps, Autoscaling and Testing]
- HPA controller now reconciles newly created and spec-changed HPAs immediately instead of waiting for the full resync period (default 15s). ([#138294](https://github.com/kubernetes/kubernetes/pull/138294), [@Fedosin](https://github.com/Fedosin)) [SIG Apps and Autoscaling]
- Image volume validation now rejects empty `image.reference` fields in Pod templates (Deployment, StatefulSet, DaemonSet, Job, etc.). ([#135989](https://github.com/kubernetes/kubernetes/pull/135989), [@Okabe-Junya](https://github.com/Okabe-Junya)) [SIG Apps and Node]
- Improve error reporting when invoking kubectl exec ([#138214](https://github.com/kubernetes/kubernetes/pull/138214), [@hunshcn](https://github.com/hunshcn)) [SIG CLI and Testing]
- Kube-apiserver now validates the `--advertise-address` IP when using `--endpoint-reconciler-type` `master-count` or `lease` to ensure the specified IP address can be persisted to an `Endpoints` API object successfully. ([#138102](https://github.com/kubernetes/kubernetes/pull/138102), [@kairosci](https://github.com/kairosci)) [SIG API Machinery]
- Kube-proxy does not perform full-sync operations when operation in large cluster mode (more than 1000 endpoints) ([#138571](https://github.com/kubernetes/kubernetes/pull/138571), [@aojea](https://github.com/aojea)) [SIG Network]
- Kube-proxy now truncates nftables comments to the kernel's 128-byte limit before programming service maps, avoiding sync failures for long Service names. ([#139516](https://github.com/kubernetes/kubernetes/pull/139516), [@Vinayak9769](https://github.com/Vinayak9769)) [SIG Network]
- Kubeadm: during 'kubeadm init', if the default 'admin.conf' and 'super-admin.conf' paths are used, load the files, but construct in memory kubeconfigs that point to the InitConfiguration.localAPIEndpoint instead of the ClusterConfiguration.controlPlaneEndpoint. This would resolve issues with delayed load balancers which are provisioned only after the first kube-apiserver instance starts. ([#138449](https://github.com/kubernetes/kubernetes/pull/138449), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix MemberPromote to skip the etcd promote API call when the member is already a voting member, avoiding unnecessary retries and timeout. ([#138390](https://github.com/kubernetes/kubernetes/pull/138390), [@wgkingk](https://github.com/wgkingk)) [SIG Cluster Lifecycle]
- Kubeadm: fixed a panic in kubeadm PKI key loading when the private key type and public key type mismatch. ([#138939](https://github.com/kubernetes/kubernetes/pull/138939), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: fixed kubeadm init phase certs --dry-run to correctly copy existing CA files. ([#139339](https://github.com/kubernetes/kubernetes/pull/139339), [@ErikJiang](https://github.com/ErikJiang)) [SIG Cluster Lifecycle]
- Kubeadm: kubeadm join now returns a clear error message when the TLS bootstrap kubeconfig has a current-context that does not appear in the contexts list, instead of panicking with a nil pointer dereference. ([#138853](https://github.com/kubernetes/kubernetes/pull/138853), [@alexmchughdev](https://github.com/alexmchughdev)) [SIG Cluster Lifecycle]
- Kubeadm: skip LocalAPIEndpoint defaulting on 'kubeadm join' for worker nodes. ([#138692](https://github.com/kubernetes/kubernetes/pull/138692), [@clwluvw](https://github.com/clwluvw)) [SIG Cluster Lifecycle]
- Kubeadm: use a dedicated ClusterRole 'system:kubelet-api-admin' for the kube-apiserver kubelet client. ([#138957](https://github.com/kubernetes/kubernetes/pull/138957), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: when checking the etcd cluster status use a quorum approach, instead of considering the health of all members. This would allow the check to not fail if there are sufficient healthy voting members. ([#138403](https://github.com/kubernetes/kubernetes/pull/138403), [@ahrtr](https://github.com/ahrtr)) [SIG Cluster Lifecycle]
- Kubeadm: when fetching cluster-info over HTTPS during discovery, the HTTP response status code is now checked, so a non-200 response produces a clear error instead of a confusing kubeconfig parse failure. ([#138852](https://github.com/kubernetes/kubernetes/pull/138852), [@alexmchughdev](https://github.com/alexmchughdev)) [SIG Cluster Lifecycle]
- Kubectl get now errors when --label-columns is used with custom-columns output. ([#138094](https://github.com/kubernetes/kubernetes/pull/138094), [@ahmadmaha02](https://github.com/ahmadmaha02)) [SIG CLI]
- Kubelet now enforces explicit HTTP method restrictions for logs-related endpoints. Read-only kubelet server endpoints reject non-GET methods with 405. NodeLogQuery explicitly allows only GET and POST and rejects other methods with 405. ([#138088](https://github.com/kubernetes/kubernetes/pull/138088), [@amritansh1502](https://github.com/amritansh1502)) [SIG Node]
- Kubelet now recovers from corrupted subpath mount points (e.g. stale NFS file handle) during container restart instead of leaving the pod stuck in CreateContainerConfigError. ([#138856](https://github.com/kubernetes/kubernetes/pull/138856), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]
- Kubelet: set cgroup v2 memory.high for BestEffort containers when MemoryQoS is enabled (per KEP-2570). ([#138139](https://github.com/kubernetes/kubernetes/pull/138139), [@amritansh1502](https://github.com/amritansh1502)) [SIG Node]
- Kubelet: the eviction manager's monitoring goroutine now exits promptly when the kubelet's context is cancelled, fixing a goroutine leak on shutdown. ([#138854](https://github.com/kubernetes/kubernetes/pull/138854), [@alexmchughdev](https://github.com/alexmchughdev)) [SIG Node]
- Remove [alpha] admission plugin that validates PodGroup resources reference an existing Workload and match the declared PodGroupTemplate spec. ([#139008](https://github.com/kubernetes/kubernetes/pull/139008), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Etcd, Scheduling and Testing]
- Removed an edge case that could allow malformed object deletion to bypass admission and graceful deletion of well-formed objects. ([#137582](https://github.com/kubernetes/kubernetes/pull/137582), [@benluddy](https://github.com/benluddy)) [SIG API Machinery, Etcd and Testing]
- This fixes a bug related to pods that were removed from the active or backoff queues before scheduling. Previously, the metrics associated with these removed pods were not adjusted; this PR introduces a fix that allows us to decrease metrics for such pods. ([#138482](https://github.com/kubernetes/kubernetes/pull/138482), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling]
- Use stable curl download for windows busybox testing image ([#138879](https://github.com/kubernetes/kubernetes/pull/138879), [@BenTheElder](https://github.com/BenTheElder)) [SIG Testing and Windows]

### Other (Cleanup or Flake)

- Client-go will request v2 for aggregated discovery and not fall back to v2beta1 ([#138271](https://github.com/kubernetes/kubernetes/pull/138271), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Deprecated MultiLock, UnknownLeader, and ConcatRawRecord in client-go leader election resourcelock package. ([#138070](https://github.com/kubernetes/kubernetes/pull/138070), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Fixed a theoretic issue where nodes might have been denied access to synthesized ResourceClaims for pods using extended resources (e.g. nvidia.com/gpu), causing containers to get stuck in ContainerCreating. Not observed in practice. ([#138792](https://github.com/kubernetes/kubernetes/pull/138792), [@dims](https://github.com/dims)) [SIG Auth and Node]
- Kube-apiserver enable-logs-handler, deprecated in 1.15, is no-longer marked deprecated. It remains off-by-default. ([#138915](https://github.com/kubernetes/kubernetes/pull/138915), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery]
- Kube-controller-manager and kube-scheduler now both expose "dynamic_resource_allocation_resourceclaim_creates_total" as metric for number of ResourceClaims created, replacing differently names metrics in each component. The kube-controller-manager metric "resource_claims" gets moved to the same "dynamic_resource_allocation" sub-system. ([#138542](https://github.com/kubernetes/kubernetes/pull/138542), [@pohly](https://github.com/pohly)) [SIG Apps, Instrumentation, Node, Release, Scheduling and Testing]
- Kubeadm: removed the v1beta3 API which was deprecated since v1.31. The 1.35 kubeadm binary can be used to migrate to v1beta4 by using the command 'kubeadm config migrate'. Additionally, removed the PublicKeysECDSA kubeadm specific feature gate which was only kept for backwards compatibility with v1beta3. The support for ECDSA keys was added as part of the v1beta4 field ClusterConfiguration.EncryptionAlgorithm. Added a placeholder v1 API, that is a copy of v1beta4 and is flagged as experimental and cannot be used yet. ([#136016](https://github.com/kubernetes/kubernetes/pull/136016), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: updated the supported etcd version to v3.6.10 for supported control plane versions v1.34, v1.35, and v1.36 ([#138392](https://github.com/kubernetes/kubernetes/pull/138392), [@humblec](https://github.com/humblec)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Kubeadm: updated the supported etcd version to v3.6.11 for supported control plane versions v1.34, v1.35, and v1.36 ([#138746](https://github.com/kubernetes/kubernetes/pull/138746), [@humblec](https://github.com/humblec)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Promote `apiserver_watch_events_total` and `apiserver_watch_events_sizes` to BETA ([#137116](https://github.com/kubernetes/kubernetes/pull/137116), [@tico88612](https://github.com/tico88612)) [SIG API Machinery, Instrumentation and Testing]
- Remove RelaxedDNSSearchValidation feature gate ([#139217](https://github.com/kubernetes/kubernetes/pull/139217), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Apps, Node and Testing]
- Removed locked GA feature gates `RetryGenerateName`, `BtreeWatchCache`, `OrderedNamespaceDeletion`, `StreamingCollectionEncodingToJSON`, `StreamingCollectionEncodingToProtobuf`, `APIServerTracing`, `ResilientWatchCacheInitialization`, and `ConsistentListFromCache`. ([#138907](https://github.com/kubernetes/kubernetes/pull/138907), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Apps, Etcd and Node]
- Removed the `--concurrent-service-syncs` kube-controller-manager flag (no-op since v1.31). ([#138002](https://github.com/kubernetes/kubernetes/pull/138002), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Removes the `KubeletMinVersion` label from the DRA e2e test covering multiple `ResourceClaims`. ([#138001](https://github.com/kubernetes/kubernetes/pull/138001), [@rogowski-piotr](https://github.com/rogowski-piotr)) [SIG Node and Testing]
- Switch StorageVersionMigration to use merge patch over SSA ([#138874](https://github.com/kubernetes/kubernetes/pull/138874), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps and Auth]
- The SidecarContainers feature gate, unconditionally enabled since 1.33, is removed. ([#137755](https://github.com/kubernetes/kubernetes/pull/137755), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Apps, Node, Scheduling and Testing]
- The deprecated ALPHA metrics `apiserver_cache_list_total`, `apiserver_cache_list_fetched_objects_total`, and `apiserver_cache_list_returned_objects_total` are no longer exposed by default.  
  Should migrate to the unified `apiserver_storage_list_*` metrics with `storage="watchcache"` label. ([#139154](https://github.com/kubernetes/kubernetes/pull/139154), [@yedou37](https://github.com/yedou37)) [SIG API Machinery and Instrumentation]
- The no-op `DefaultWatchCacheSize` field of `k8s.io/apiserver/pkg/server/options.EtcdOptions` is now removed. ([#134151](https://github.com/kubernetes/kubernetes/pull/134151), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- Updates the etcd client library to v3.6.11 ([#138747](https://github.com/kubernetes/kubernetes/pull/138747), [@humblec](https://github.com/humblec)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]

## Dependencies

### Added
- github.com/aclements/go-moremath: [f10218a](https://github.com/aclements/go-moremath/tree/f10218a)
- github.com/go-openapi/swag/cmdutils: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/conv: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/fileutils: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/jsonname: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/jsonutils: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/jsonutils/fixtures_test: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/loading: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/mangling: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/netutils: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/stringutils: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/typeutils: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/swag/yamlutils: [v0.25.4](https://github.com/go-openapi/swag/commit/73525ad4f9d84ce7e4b15796f0d41a6cb32cc3cd)
- github.com/go-openapi/testify/enable/yaml/v2: [v2.0.2](https://github.com/go-openapi/testify/commit/43fcbc6c768e560ad65ed0be848fb306fee0c312)
- github.com/go-openapi/testify/v2: [v2.0.2](https://github.com/go-openapi/testify/commit/43fcbc6c768e560ad65ed0be848fb306fee0c312)
- go.opentelemetry.io/otel/metric/x: [v0.66.0](https://github.com/open-telemetry/opentelemetry-go/commit/b62d92831b2dd142f5a0cc89c828270274196877)
- golang.org/x/perf: [2f7363a](https://go.googlesource.com/perf/+/2f7363a06fe1e84314f47158b693ef982b0c2255)

### Changed
- github.com/Azure/go-ansiterm: [306776e → faa5f7b](https://github.com/Azure/go-ansiterm/compare/306776e...faa5f7b)
- github.com/GoogleCloudPlatform/opentelemetry-operations-go/detectors/gcp: [v1.30.0 → v1.31.0](https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/compare/v1.30.0...v1.31.0)
- github.com/Microsoft/hnslib: [v0.1.2 → v0.1.3](https://github.com/Microsoft/hnslib/compare/v0.1.2...v0.1.3)
- github.com/antlr4-go/antlr/v4: [v4.13.0 → v4.13.1](https://github.com/antlr4-go/antlr/compare/v4.13.0...v4.13.1)
- github.com/cncf/xds/go: [ee656c7 → dba9d58](https://github.com/cncf/xds/compare/ee656c7534f5d7dc23d44dd611689568f72017a6...dba9d589def2cd10099a3a64887d859188c2f57a)
- github.com/containerd/containerd/api: [v1.10.0 → v1.11.0](https://github.com/containerd/containerd/compare/api/v1.10.0...api/v1.11.0)
- github.com/containerd/ttrpc: [v1.2.7 → v1.2.8](https://github.com/containerd/ttrpc/compare/v1.2.7...v1.2.8)
- github.com/containerd/typeurl/v2: [v2.2.3 → v2.3.0](https://github.com/containerd/typeurl/compare/v2.2.3...v2.3.0)
- github.com/envoyproxy/go-control-plane/envoy: [v1.36.0 → v1.37.0](https://github.com/envoyproxy/go-control-plane/compare/envoy/v1.36.0...envoy/v1.37.0)
- github.com/envoyproxy/protoc-gen-validate: [v1.3.0 → v1.3.3](https://github.com/envoyproxy/protoc-gen-validate/compare/v1.3.0...v1.3.3)
- github.com/fxamacker/cbor/v2: [v2.9.0 → v2.9.1](https://github.com/fxamacker/cbor/compare/v2.9.0...v2.9.1)
- github.com/go-jose/go-jose/v4: [v4.1.3 → v4.1.4](https://github.com/go-jose/go-jose/compare/v4.1.3...v4.1.4)
- github.com/go-openapi/jsonpointer: [v0.21.0 → v0.22.4](https://github.com/go-openapi/jsonpointer/compare/v0.21.0...v0.22.4)
- github.com/go-openapi/jsonreference: [v0.20.2 → v0.21.4](https://github.com/go-openapi/jsonreference/compare/v0.20.2...v0.21.4)
- github.com/go-openapi/swag: [v0.23.0 → v0.25.4](https://github.com/go-openapi/swag/compare/v0.23.0...v0.25.4)
- github.com/golang-jwt/jwt/v5: [v5.3.0 → v5.3.1](https://github.com/golang-jwt/jwt/compare/v5.3.0...v5.3.1)
- github.com/google/cadvisor: [v0.56.2 → v0.57.0](https://github.com/google/cadvisor/compare/v0.56.2...v0.57.0)
- github.com/google/cel-go: [v0.26.0 → v0.27.0](https://github.com/google/cel-go/compare/v0.26.0...v0.27.0)
- github.com/google/pprof: [294ebfa → 545e8a4](https://github.com/google/pprof/compare/294ebfa9ad836ed3d00d43d54ea599339e403110...545e8a4df9364095d66e521b8f515f7af961e653)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.27.7 → v2.29.0](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.27.7...v2.29.0)
- github.com/moby/moby/api: [v1.52.0 → v1.54.1](https://github.com/moby/moby/compare/1408c9ca4f4f0717b2d885fc87ae0ff000a91c40...api/v1.54.1)
- github.com/moby/moby/client: [v0.2.1 → v0.4.0](https://github.com/moby/moby/compare/client/v0.2.1...client/v0.4.0)
- github.com/moby/term: [v0.5.0 → v0.5.2](https://github.com/moby/term/compare/main...v0.5.2)
- github.com/onsi/ginkgo/v2: [v2.28.1 → v2.28.3](https://github.com/onsi/ginkgo/compare/v2.28.1...v2.28.3)
- github.com/onsi/gomega: [v1.39.1 → v1.40.0](https://github.com/onsi/gomega/compare/v1.39.1...v1.40.0)
- github.com/sirupsen/logrus: [v1.9.3 → v1.9.4](https://github.com/sirupsen/logrus/compare/v1.9.3...v1.9.4)
- github.com/spf13/pflag: [v1.0.9 → v1.0.10](https://github.com/spf13/pflag/compare/v1.0.9...v1.0.10)
- github.com/stretchr/objx: [v0.5.2 → v0.5.3](https://github.com/stretchr/objx/compare/v0.5.2...v0.5.3)
- go.etcd.io/bbolt: [v1.4.3 → v1.5.0-rc.0](https://github.com/etcd-io/bbolt/compare/v1.4.3...v1.5.0-rc.0)
- go.etcd.io/etcd/api/v3: [v3.6.8 → v3.7.0-rc.0](https://github.com/etcd-io/etcd/compare/api/v3.6.8...api/v3.7.0-rc.0)
- go.etcd.io/etcd/client/pkg/v3: [v3.6.8 → v3.7.0-rc.0](https://github.com/etcd-io/etcd/compare/client/pkg/v3.6.8...client/pkg/v3.7.0-rc.0)
- go.etcd.io/etcd/client/v3: [v3.6.8 → v3.7.0-rc.0](https://github.com/etcd-io/etcd/compare/client/v3.6.8...client/v3.7.0-rc.0)
- go.etcd.io/etcd/pkg/v3: [v3.6.8 → v3.7.0-rc.0](https://github.com/etcd-io/etcd/compare/pkg/v3.6.8...pkg/v3.7.0-rc.0)
- go.etcd.io/etcd/server/v3: [v3.6.8 → v3.7.0-rc.0](https://github.com/etcd-io/etcd/compare/server/v3.6.8...server/v3.7.0-rc.0)
- go.etcd.io/raft/v3: [v3.6.0 → v3.7.0-rc.1](https://github.com/etcd-io/raft/compare/v3.6.0...v3.7.0-rc.1)
- go.opentelemetry.io/contrib/detectors/gcp: [v1.39.0 → v1.42.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/detectors/gcp/v1.39.0...detectors/gcp/v1.42.0)
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: [v0.65.0 → v0.68.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/instrumentation/github.com/emicklei/go-restful/otelrestful/v0.65.0...instrumentation/github.com/emicklei/go-restful/otelrestful/v0.68.0)
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: [v0.65.0 → v0.68.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/instrumentation/google.golang.org/grpc/otelgrpc/v0.65.0...instrumentation/google.golang.org/grpc/otelgrpc/v0.68.0)
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: [v0.65.0 → v0.68.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/instrumentation/net/http/otelhttp/v0.65.0...instrumentation/net/http/otelhttp/v0.68.0)
- go.opentelemetry.io/contrib/propagators/b3: [v1.40.0 → v1.43.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/propagators/b3/v1.40.0...propagators/b3/v1.43.0)
- go.opentelemetry.io/otel: [v1.41.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go/compare/v1.41.0...v1.44.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: [v1.40.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go/compare/exporters/otlp/otlptrace/v1.40.0...exporters/otlp/otlptrace/v1.44.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: [v1.40.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go/compare/exporters/otlp/otlptrace/otlptracegrpc/v1.40.0...exporters/otlp/otlptrace/otlptracegrpc/v1.44.0)
- go.opentelemetry.io/otel/exporters/stdout/stdouttrace: [v1.40.0 → v1.43.0](https://github.com/open-telemetry/opentelemetry-go/compare/exporters/stdout/stdouttrace/v1.40.0...exporters/stdout/stdouttrace/v1.43.0)
- go.opentelemetry.io/otel/metric: [v1.41.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go/compare/metric/v1.41.0...metric/v1.44.0)
- go.opentelemetry.io/otel/sdk: [v1.40.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go/compare/sdk/v1.40.0...sdk/v1.44.0)
- go.opentelemetry.io/otel/sdk/metric: [v1.40.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go/compare/sdk/metric/v1.40.0...sdk/metric/v1.44.0)
- go.opentelemetry.io/otel/trace: [v1.41.0 → v1.44.0](https://github.com/open-telemetry/opentelemetry-go/compare/trace/v1.41.0...trace/v1.44.0)
- go.opentelemetry.io/proto/otlp: [v1.9.0 → v1.10.0](https://github.com/open-telemetry/opentelemetry-proto-go/compare/otlp/v1.9.0...otlp/v1.10.0)
- go.yaml.in/yaml/v2: [v2.4.3 → v2.4.4](https://github.com/yaml/go-yaml/compare/v2.4.3...v2.4.4)
- golang.org/x/crypto: [v0.47.0 → v0.52.0](https://go.googlesource.com/crypto/+/506e022208b864bc3c9c4a416fe56be75d10ad24^1..a1c0d9929856c8aba2b31f079340f00578eda803/)
- golang.org/x/exp: [944ab1f → 746e56f](https://go.googlesource.com/exp/+/944ab1f22d936eefb8f6260ecd2053101d8d7b2a^1..746e56fc9e2fafde18176275ce0b96b06ac53955/)
- golang.org/x/mod: [v0.32.0 → v0.35.0](https://go.googlesource.com/mod/+/4c04067938546e62fc0572259a68a6912726bcdd^1..03901d351deb5bd95deb90714fb75bf8e232cb22/)
- golang.org/x/net: [v0.49.0 → 42abb85](https://go.googlesource.com/net/+/d977772e17ccaa1903b2af736f6405ab3a9f05cc^1..42abb857022cb79baacfa240bcf48588aa80bbee/)
- golang.org/x/oauth2: [v0.34.0 → v0.36.0](https://go.googlesource.com/oauth2/+/acc38155b7f6f36aefcb58faff6f36d314dd915c^1..4d954e69a88d9e1ccb8439f8d5b6cbef230c4ef9/)
- golang.org/x/sync: [v0.19.0 → v0.20.0](https://go.googlesource.com/sync/+/2a180e22fddcc336475e72aa950be958c1b68d33^1..ec11c4a93de22cde2abe2bf74d70791033c2464c/)
- golang.org/x/sys: [v0.40.0 → v0.45.0](https://go.googlesource.com/sys/+/2f442297556c884f9b52fc6ef7280083f4d65023^1..397d5f80920585bc27433d878aba498d062f81e1/)
- golang.org/x/telemetry: [bd525da → be6f6cb](https://go.googlesource.com/telemetry/+/bd525da824e2505db9e8ac44025316bf6f43a6f6^1..be6f6cb8b1fafcb1c710a0666a79acac61139b7b/)
- golang.org/x/term: [v0.39.0 → v0.43.0](https://go.googlesource.com/term/+/a7e5b0437ffa3159709172efbe396bc546550e23^1..3c3e4855f7d2eb06c3e48933554add9ec6b599b5/)
- golang.org/x/text: [v0.33.0 → v0.37.0](https://go.googlesource.com/text/+/536231a9abc69feaab8d726b5ec75ee8d3620829^1..3ef517e623a4bfc08d6457f87d73afda7af7d8e1/)
- golang.org/x/time: [v0.14.0 → v0.15.0](https://go.googlesource.com/time/+/2b4e43900c03fd6b77109b7b2b6d77583f48bc1c^1..812b343c8714c317b0dad633efa6d103e554c006/)
- golang.org/x/tools: [v0.41.0 → v0.44.0](https://go.googlesource.com/tools/+/2ad2b30edf98d0e3b67a7b3e8f6d1d6e41c963c3^1..3dd188df80fd3563559f02e4eeb10ba1043cce55/)
- gonum.org/v1/gonum: [v0.16.0 → v0.17.0](https://github.com/gonum/gonum/compare/v0.16.0...v0.17.0)
- google.golang.org/genproto/googleapis/api: [8636f87 → 3dc84a4](https://github.com/googleapis/go-genproto/compare/8636f8732409467ddc8453f81f4429397739bb17...3dc84a4a5aaa87331e10f51e22e90d961f986894)
- google.golang.org/genproto/googleapis/rpc: [8636f87 → 3dc84a4](https://github.com/googleapis/go-genproto/compare/8636f8732409467ddc8453f81f4429397739bb17...3dc84a4a5aaa87331e10f51e22e90d961f986894)
- google.golang.org/grpc: [v1.79.3 → v1.81.1](https://github.com/grpc/grpc-go/compare/v1.79.3...v1.81.1)
- k8s.io/gengo/v2: [ec3ebc5 → 25e2208](https://github.com/kubernetes/gengo/compare/ec3ebc5fd46b84f44dfb135e9684c6567791dd8e...25e2208e0dc371a827289e7faced19a2dbcd480b)
- k8s.io/kube-openapi: [43fb72c → bbf5c55](https://github.com/kubernetes/kube-openapi/compare/43fb72c5454a03ed83388cf20c070499ee359af8...bbf5c557728870a14cee51b4271cfd51b58ef2f8)
- sigs.k8s.io/structured-merge-diff/v6: [v6.3.2 → v6.4.0](https://github.com/kubernetes-sigs/structured-merge-diff/compare/v6.3.2...v6.4.0)

### Removed
- github.com/cenkalti/backoff/v4: [v4.3.0](https://github.com/cenkalti/backoff/commit/720b78985a65c0452fd37bb155c7cac4157a7c45)
- github.com/golang/groupcache: [41bb18b](https://github.com/golang/groupcache/tree/41bb18b)
- github.com/grpc-ecosystem/go-grpc-middleware: [v1.3.0](https://github.com/grpc-ecosystem/go-grpc-middleware/tree/v1.3.0)