<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.20.0-alpha.2](#v1200-alpha2)
  - [Downloads for v1.20.0-alpha.2](#downloads-for-v1200-alpha2)
    - [Source Code](#source-code)
    - [Client binaries](#client-binaries)
    - [Server binaries](#server-binaries)
    - [Node binaries](#node-binaries)
  - [Changelog since v1.20.0-alpha.1](#changelog-since-v1200-alpha1)
  - [Changes by Kind](#changes-by-kind)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.20.0-alpha.1](#v1200-alpha1)
  - [Downloads for v1.20.0-alpha.1](#downloads-for-v1200-alpha1)
    - [Source Code](#source-code-1)
    - [Client binaries](#client-binaries-1)
    - [Server binaries](#server-binaries-1)
    - [Node binaries](#node-binaries-1)
  - [Changelog since v1.20.0-alpha.0](#changelog-since-v1200-alpha0)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)

<!-- END MUNGE: GENERATED_TOC -->

# v1.20.0-alpha.2


## Downloads for v1.20.0-alpha.2

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes.tar.gz) | 45089a4d26d56a5d613ecbea64e356869ac738eca3cc71d16b74ea8ae1b4527bcc32f1dc35ff7aa8927e138083c7936603faf063121d965a2f0f8ba28fa128d8
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-src.tar.gz) | 646edd890d6df5858b90aaf68cc6e1b4589b8db09396ae921b5c400f2188234999e6c9633906692add08c6e8b4b09f12b2099132b0a7533443fb2a01cfc2bf81

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | c136273883e24a2a50b5093b9654f01cdfe57b97461d34885af4a68c2c4d108c07583c02b1cdf7f57f82e91306e542ce8f3bddb12fcce72b744458bc4796f8eb
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 6ec59f1ed30569fa64ddb2d0de32b1ae04cda4ffe13f339050a7c9d7c63d425ee6f6d963dcf82c17281c4474da3eaf32c08117669052872a8c81bdce2c8a5415
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 7b40a4c087e2ea7f8d055f297fcd39a3f1cb6c866e7a3981a9408c3c3eb5363c648613491aad11bc7d44d5530b20832f8f96f6ceff43deede911fb74aafad35f
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | cda9955feebea5acb8f2b5b87895d24894bbbbde47041453b1f926ebdf47a258ce0496aa27d06bcbf365b5615ce68a20d659b64410c54227216726e2ee432fca
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | f65bd9241c7eb88a4886a285330f732448570aea4ededaebeabcf70d17ea185f51bf8a7218f146ee09fb1adceca7ee71fb3c3683834f2c415163add820fba96e
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 1e377599af100a81d027d9199365fb8208d443a8e0a97affff1a79dc18796e14b78cb53d6e245c1c1e8defd0e050e37bf5f2a23c8a3ff45a6d18d03619709bf5
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 1cdee81478246aa7e7b80ae4efc7f070a5b058083ae278f59fad088b75a8052761b0e15ab261a6e667ddafd6a69fb424fc307072ed47941cad89a85af7aee93d
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-windows-386.tar.gz) | d8774167c87b6844c348aa15e92d5033c528d6ab9e95d08a7cb22da68bafd8e46d442cf57a5f6affad62f674c10ae6947d524b94108b5e450ca78f92656d63c0
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | f664b47d8daa6036f8154c1dc1f881bfe683bf57c39d9b491de3848c03d051c50c6644d681baf7f9685eae45f9ce62e4c6dfea2853763cfe8256a61bdd59d894

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | d6fcb4600be0beb9de222a8da64c35fe22798a0da82d41401d34d0f0fc7e2817512169524c281423d8f4a007cd77452d966317d5a1b67d2717a05ff346e8aa7d
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | 022a76cf10801f8afbabb509572479b68fdb4e683526fa0799cdbd9bab4d3f6ecb76d1d63d0eafee93e3edf6c12892d84b9c771ef2325663b95347728fa3d6c0
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 0679aadd60bbf6f607e5befad74b5267eb2d4c1b55985cc25a97e0f4c5efb7acbb3ede91bfa6a5a5713dae4d7a302f6faaf678fd6b359284c33d9a6aca2a08bb
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 9f2cfeed543b515eafb60d9765a3afff4f3d323c0a5c8a0d75e3de25985b2627817bfcbe59a9a61d969e026e2b861adb974a09eae75b58372ed736ceaaed2a82
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 937258704d7b9dcd91f35f2d34ee9dd38c18d9d4e867408c05281bfbbb919ad012c95880bee84d2674761aa44cc617fb2fae1124cf63b689289286d6eac1c407

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 076165d745d47879de68f4404eaf432920884be48277eb409e84bf2c61759633bf3575f46b0995f1fc693023d76c0921ed22a01432e756d7f8d9e246a243b126
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 1ff2e2e3e43af41118cdfb70c778e15035bbb1aca833ffd2db83c4bcd44f55693e956deb9e65017ebf3c553f2820ad5cd05f5baa33f3d63f3e00ed980ea4dfed
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | b232c7359b8c635126899beee76998078eec7a1ef6758d92bcdebe8013b0b1e4d7b33ecbf35e3f82824fe29493400845257e70ed63c1635bfa36c8b3b4969f6f
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 51d415a068f554840f4c78d11a4fedebd7cb03c686b0ec864509b24f7a8667ebf54bb0a25debcf2b70f38be1e345e743f520695b11806539a55a3620ce21946f
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | b51c082d8af358233a088b632cf2f6c8cfe5421471c27f5dc9ba4839ae6ea75df25d84298f2042770097554c01742bb7686694b331ad9bafc93c86317b867728
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 91b9d26620a2dde67a0edead0039814efccbdfd54594dda3597aaced6d89140dc92612ed0727bc21d63468efeef77c845e640153b09e39d8b736062e6eee0c76

## Changelog since v1.20.0-alpha.1

## Changes by Kind

### Deprecation

- Action-required: kubeadm: graduate the "kubeadm alpha certs" command to a parent command "kubeadm certs". The command "kubeadm alpha certs" is deprecated and will be removed in a future release. Please migrate. ([#94938](https://github.com/kubernetes/kubernetes/pull/94938), [@yagonobre](https://github.com/yagonobre)) [SIG Cluster Lifecycle]
- Action-required: kubeadm: remove the deprecated feature --experimental-kustomize from kubeadm commands. The feature was replaced with --experimental-patches in 1.19. To migrate see the --help description for the --experimental-patches flag. ([#94871](https://github.com/kubernetes/kubernetes/pull/94871), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: deprecate self-hosting support. The experimental command "kubeadm alpha self-hosting" is now deprecated and will be removed in a future release. ([#95125](https://github.com/kubernetes/kubernetes/pull/95125), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Removes deprecated scheduler metrics DeprecatedSchedulingDuration, DeprecatedSchedulingAlgorithmPredicateEvaluationSecondsDuration, DeprecatedSchedulingAlgorithmPriorityEvaluationSecondsDuration ([#94884](https://github.com/kubernetes/kubernetes/pull/94884), [@arghya88](https://github.com/arghya88)) [SIG Instrumentation and Scheduling]
- Scheduler alpha metrics binding_duration_seconds and scheduling_algorithm_preemption_evaluation_seconds are deprecated, Both of those metrics are now covered as part of framework_extension_point_duration_seconds, the former as a PostFilter the latter and a Bind plugin. The plan is to remove both in 1.21 ([#95001](https://github.com/kubernetes/kubernetes/pull/95001), [@arghya88](https://github.com/arghya88)) [SIG Instrumentation and Scheduling]

### API Change

- GPU metrics provided by kubelet are now disabled by default ([#95184](https://github.com/kubernetes/kubernetes/pull/95184), [@RenaudWasTaken](https://github.com/RenaudWasTaken)) [SIG Node]
- New parameter `defaultingType` for `PodTopologySpread` plugin allows to use k8s defined or user provided default constraints ([#95048](https://github.com/kubernetes/kubernetes/pull/95048), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Server Side Apply now treats LabelSelector fields as atomic (meaning the entire selector is managed by a single writer and updated together), since they contain interrelated and inseparable fields that do not merge in intuitive ways. ([#93901](https://github.com/kubernetes/kubernetes/pull/93901), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Storage and Testing]
- Status of v1beta1 CRDs without "preserveUnknownFields:false" will show violation "spec.preserveUnknownFields: Invalid value: true: must be false" ([#93078](https://github.com/kubernetes/kubernetes/pull/93078), [@vareti](https://github.com/vareti)) [SIG API Machinery]

### Feature

- Added `get-users` and `delete-user` to the `kubectl config` subcommand ([#89840](https://github.com/kubernetes/kubernetes/pull/89840), [@eddiezane](https://github.com/eddiezane)) [SIG CLI]
- Added counter metric "apiserver_request_self" to count API server self-requests with labels for verb, resource, and subresource. ([#94288](https://github.com/kubernetes/kubernetes/pull/94288), [@LogicalShark](https://github.com/LogicalShark)) [SIG API Machinery, Auth, Instrumentation and Scheduling]
- Added new k8s.io/component-helpers repository providing shared helper code for (core) components. ([#92507](https://github.com/kubernetes/kubernetes/pull/92507), [@ingvagabund](https://github.com/ingvagabund)) [SIG Apps, Node, Release and Scheduling]
- Adds `create ingress` command to `kubectl` ([#78153](https://github.com/kubernetes/kubernetes/pull/78153), [@amimof](https://github.com/amimof)) [SIG CLI and Network]
- Allow configuring AWS LoadBalancer health check protocol via service annotations ([#94546](https://github.com/kubernetes/kubernetes/pull/94546), [@kishorj](https://github.com/kishorj)) [SIG Cloud Provider]
- Azure: Support multiple services sharing one IP address ([#94991](https://github.com/kubernetes/kubernetes/pull/94991), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Ephemeral containers now apply the same API defaults as initContainers and containers ([#94896](https://github.com/kubernetes/kubernetes/pull/94896), [@wawa0210](https://github.com/wawa0210)) [SIG Apps and CLI]
- In dual-stack bare-metal clusters, you can now pass dual-stack IPs to `kubelet --node-ip`.
  eg: `kubelet --node-ip 10.1.0.5,fd01::0005`. This is not yet supported for non-bare-metal
  clusters.
  
  In dual-stack clusters where nodes have dual-stack addresses, hostNetwork pods
  will now get dual-stack PodIPs. ([#95239](https://github.com/kubernetes/kubernetes/pull/95239), [@danwinship](https://github.com/danwinship)) [SIG Network and Node]
- Introduces a new GCE specific cluster creation variable KUBE_PROXY_DISABLE. When set to true, this will skip over the creation of kube-proxy (whether the daemonset or static pod). This can be used to control the lifecycle of kube-proxy separately from the lifecycle of the nodes. ([#91977](https://github.com/kubernetes/kubernetes/pull/91977), [@varunmar](https://github.com/varunmar)) [SIG Cloud Provider]
- Kubeadm: do not throw errors if the current system time is outside of the NotBefore and NotAfter bounds of a loaded certificate. Print warnings instead. ([#94504](https://github.com/kubernetes/kubernetes/pull/94504), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: make the command "kubeadm alpha kubeconfig user" accept a "--config" flag and remove the following flags:
  - apiserver-advertise-address / apiserver-bind-port: use either localAPIEndpoint from InitConfiguration or controlPlaneEndpoint from ClusterConfiguration.
  - cluster-name: use clusterName from ClusterConfiguration
  - cert-dir: use certificatesDir from ClusterConfiguration ([#94879](https://github.com/kubernetes/kubernetes/pull/94879), [@knight42](https://github.com/knight42)) [SIG Cluster Lifecycle]
- Kubectl rollout history sts/sts-name --revision=some-revision will start showing the detailed view of  the sts on that specified revision ([#86506](https://github.com/kubernetes/kubernetes/pull/86506), [@dineshba](https://github.com/dineshba)) [SIG CLI]
- Scheduling Framework: expose Run[Pre]ScorePlugins functions to PreemptionHandle which can be used in PostFilter extention point. ([#93534](https://github.com/kubernetes/kubernetes/pull/93534), [@everpeace](https://github.com/everpeace)) [SIG Scheduling and Testing]
- Send gce node startup scripts logs to console and journal ([#95311](https://github.com/kubernetes/kubernetes/pull/95311), [@karan](https://github.com/karan)) [SIG Cloud Provider and Node]
- Support kubectl delete orphan/foreground/background options ([#93384](https://github.com/kubernetes/kubernetes/pull/93384), [@zhouya0](https://github.com/zhouya0)) [SIG CLI and Testing]

### Bug or Regression

- Change the mount way from systemd to normal mount except ceph and glusterfs intree-volume. ([#94916](https://github.com/kubernetes/kubernetes/pull/94916), [@smileusd](https://github.com/smileusd)) [SIG Apps, Cloud Provider, Network, Node, Storage and Testing]
- Cloud node controller: handle empty providerID from getProviderID ([#95342](https://github.com/kubernetes/kubernetes/pull/95342), [@nicolehanjing](https://github.com/nicolehanjing)) [SIG Cloud Provider]
- Fix a bug where the endpoint slice controller was not mirroring the parent service labels to its corresponding endpoint slices ([#94443](https://github.com/kubernetes/kubernetes/pull/94443), [@aojea](https://github.com/aojea)) [SIG Apps and Network]
- Fix azure disk attach failure for disk size bigger than 4TB ([#95463](https://github.com/kubernetes/kubernetes/pull/95463), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix azure disk data loss issue on Windows when unmount disk ([#95456](https://github.com/kubernetes/kubernetes/pull/95456), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix detach azure disk issue when vm not exist ([#95177](https://github.com/kubernetes/kubernetes/pull/95177), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix network_programming_latency metric reporting for Endpoints/EndpointSlice deletions, where we don't have correct timestamp ([#95363](https://github.com/kubernetes/kubernetes/pull/95363), [@wojtek-t](https://github.com/wojtek-t)) [SIG Network and Scalability]
- Fix scheduler cache snapshot when a Node is deleted before its Pods ([#95130](https://github.com/kubernetes/kubernetes/pull/95130), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Fix vsphere detach failure for static PVs ([#95447](https://github.com/kubernetes/kubernetes/pull/95447), [@gnufied](https://github.com/gnufied)) [SIG Cloud Provider and Storage]
- Fixed a bug that prevents the use of ephemeral containers in the presence of a validating admission webhook. ([#94685](https://github.com/kubernetes/kubernetes/pull/94685), [@verb](https://github.com/verb)) [SIG Node and Testing]
- Gracefully delete nodes when their parent scale set went missing ([#95289](https://github.com/kubernetes/kubernetes/pull/95289), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- In dual-stack clusters, kubelet will now set up both IPv4 and IPv6 iptables rules, which may
  fix some problems, eg with HostPorts. ([#94474](https://github.com/kubernetes/kubernetes/pull/94474), [@danwinship](https://github.com/danwinship)) [SIG Network and Node]
- Kubeadm: for Docker as the container runtime, make the "kubeadm reset" command stop containers before removing them ([#94586](https://github.com/kubernetes/kubernetes/pull/94586), [@BedivereZero](https://github.com/BedivereZero)) [SIG Cluster Lifecycle]
- Kubeadm: warn but do not error out on missing "ca.key" files for root CA, front-proxy CA and etcd CA, during "kubeadm join --control-plane" if the user has provided all certificates, keys and kubeconfig files which require signing with the given CA keys. ([#94988](https://github.com/kubernetes/kubernetes/pull/94988), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Port mapping allows to map the same `containerPort` to multiple `hostPort` without naming the mapping explicitly. ([#94494](https://github.com/kubernetes/kubernetes/pull/94494), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Network and Node]
- Warn instead of fail when creating Roles and ClusterRoles with custom verbs via kubectl ([#92492](https://github.com/kubernetes/kubernetes/pull/92492), [@eddiezane](https://github.com/eddiezane)) [SIG CLI]

### Other (Cleanup or Flake)

- Added fine grained debugging to the intra-pod conformance test for helping easily resolve networking issues for nodes that might be unhealthy when running conformance or sonobuoy tests. ([#93837](https://github.com/kubernetes/kubernetes/pull/93837), [@jayunit100](https://github.com/jayunit100)) [SIG Network and Testing]
- AdmissionReview objects sent for the creation of Namespace API objects now populate the `namespace` attribute consistently (previously the `namespace` attribute was empty for Namespace creation via POST requests, and populated for Namespace creation via server-side-apply PATCH requests) ([#95012](https://github.com/kubernetes/kubernetes/pull/95012), [@nodo](https://github.com/nodo)) [SIG API Machinery and Testing]
- Client-go header logging (at verbosity levels >= 9) now masks `Authorization` header contents ([#95316](https://github.com/kubernetes/kubernetes/pull/95316), [@sfowl](https://github.com/sfowl)) [SIG API Machinery]
- Enhance log information of verifyRunAsNonRoot, add pod, container information ([#94911](https://github.com/kubernetes/kubernetes/pull/94911), [@wawa0210](https://github.com/wawa0210)) [SIG Node]
- Errors from staticcheck:                                                                                                                                      
  vendor/k8s.io/client-go/discovery/cached/memory/memcache_test.go:94:2: this value of g is never used (SA4006) ([#95098](https://github.com/kubernetes/kubernetes/pull/95098), [@phunziker](https://github.com/phunziker)) [SIG API Machinery]
- Kubeadm: update the default pause image version to 1.4.0 on Windows. With this update the image supports Windows versions 1809 (2019LTS), 1903, 1909, 2004 ([#95419](https://github.com/kubernetes/kubernetes/pull/95419), [@jsturtevant](https://github.com/jsturtevant)) [SIG Cluster Lifecycle and Windows]
- Masks ceph RBD adminSecrets in logs when logLevel >= 4 ([#95245](https://github.com/kubernetes/kubernetes/pull/95245), [@sfowl](https://github.com/sfowl)) [SIG Storage]
- No ([#95412](https://github.com/kubernetes/kubernetes/pull/95412), [@saikat-royc](https://github.com/saikat-royc)) [SIG Cloud Provider]
- Remove offensive words from kubectl cluster-info command ([#95202](https://github.com/kubernetes/kubernetes/pull/95202), [@rikatz](https://github.com/rikatz)) [SIG Architecture, CLI and Testing]
- The following new metrics are available.
  - network_plugin_operations_total
  - network_plugin_operations_errors_total ([#93066](https://github.com/kubernetes/kubernetes/pull/93066), [@AnishShah](https://github.com/AnishShah)) [SIG Instrumentation, Network and Node]
- Vsphere: improve logging message on node cache refresh event ([#95236](https://github.com/kubernetes/kubernetes/pull/95236), [@andrewsykim](https://github.com/andrewsykim)) [SIG Cloud Provider]
- `kubectl api-resources` now prints the API version (as 'API group/version', same as output of `kubectl api-versions`). The column APIGROUP is now APIVERSION ([#95253](https://github.com/kubernetes/kubernetes/pull/95253), [@sallyom](https://github.com/sallyom)) [SIG CLI]

## Dependencies

### Added
- github.com/jmespath/go-jmespath/internal/testify: [v1.5.1](https://github.com/jmespath/go-jmespath/internal/testify/tree/v1.5.1)

### Changed
- github.com/aws/aws-sdk-go: [v1.28.2 → v1.35.5](https://github.com/aws/aws-sdk-go/compare/v1.28.2...v1.35.5)
- github.com/jmespath/go-jmespath: [c2b33e8 → v0.4.0](https://github.com/jmespath/go-jmespath/compare/c2b33e8...v0.4.0)
- k8s.io/kube-openapi: 6aeccd4 → 8b50664
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.9 → v0.0.12
- sigs.k8s.io/structured-merge-diff/v4: v4.0.1 → b3cf1e8

### Removed
_Nothing has changed._



# v1.20.0-alpha.1


## Downloads for v1.20.0-alpha.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes.tar.gz) | e7daed6502ea07816274f2371f96fe1a446d0d7917df4454b722d9eb3b5ff6163bfbbd5b92dfe7a0c1d07328b8c09c4ae966e482310d6b36de8813aaf87380b5
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-src.tar.gz) | e91213a0919647a1215d4691a63b12d89a3e74055463a8ebd71dc1a4cabf4006b3660881067af0189960c8dab74f4a7faf86f594df69021901213ee5b56550ea

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 1f3add5f826fa989820d715ca38e8864b66f30b59c1abeacbb4bfb96b4e9c694eac6b3f4c1c81e0ee3451082d44828cb7515315d91ad68116959a5efbdaef1e1
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-linux-386.tar.gz) | c62acdc8993b0a950d4b0ce0b45473bf96373d501ce61c88adf4007afb15c1d53da8d53b778a7eccac6c1624f7fdda322be9f3a8bc2d80aaad7b4237c39f5eaf
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 1203ababfe00f9bc5be5c059324c17160a96530c1379a152db33564bbe644ccdb94b30eea15a0655bd652efb17895a46c31bbba19d4f5f473c2a0ff62f6e551f
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 31860088596e12d739c7aed94556c2d1e217971699b950c8417a3cea1bed4e78c9ff1717b9f3943354b75b4641d4b906cd910890dbf4278287c0d224837d9a7d
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 8d469f37fe20d6e15b5debc13cce4c22e8b7a4f6a4ac787006b96507a85ce761f63b28140d692c54b5f7deb08697f8d5ddb9bbfa8f5ac0d9241fc7de3a3fe3cd
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 0d62ee1729cd5884946b6c73701ad3a570fa4d642190ca0fe5c1db0fb0cba9da3ac86a948788d915b9432d28ab8cc499e28aadc64530b7d549ee752a6ed93ec1
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 0fc0420e134ec0b8e0ab2654e1e102cebec47b48179703f1e1b79d51ee0d6da55a4e7304d8773d3cf830341ac2fe3cede1e6b0460fd88f7595534e0730422d5a
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 3fb53b5260f4888c77c0e4ff602bbcf6bf38c364d2769850afe2b8d8e8b95f7024807c15e2b0d5603e787c46af8ac53492be9e88c530f578b8a389e3bd50c099
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 2f44c93463d6b5244ce0c82f147e7f32ec2233d0e29c64c3c5759e23533aebd12671bf63e986c0861e9736f9b5259bb8d138574a7c8c8efc822e35cd637416c0

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | ae82d14b1214e4100f0cc2c988308b3e1edd040a65267d0eddb9082409f79644e55387889e3c0904a12c710f91206e9383edf510990bee8c9ea2e297b6472551
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | 9a2a5828b7d1ddb16cc19d573e99a4af642f84129408e6203eeeb0558e7b8db77f3269593b5770b6a976fe9df4a64240ed27ad05a4bd43719e55fce1db0abf58
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | ed700dd226c999354ce05b73927388d36d08474c15333ae689427de15de27c84feb6b23c463afd9dd81993315f31eb8265938cfc7ecf6f750247aa42b9b33fa9
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | abb7a9d726538be3ccf5057a0c63ff9732b616e213c6ebb81363f0c49f1e168ce8068b870061ad7cba7ba1d49252f94cf00a5f68cec0f38dc8fce4e24edc5ca6
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 3a51888af1bfdd2d5b0101d173ee589c1f39240e4428165f5f85c610344db219625faa42f00a49a83ce943fb079be873b1a114a62003fae2f328f9bf9d1227a4

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | d0f28e3c38ca59a7ff1bfecb48a1ce97116520355d9286afdca1200d346c10018f5bbdf890f130a388654635a2e83e908b263ed45f8a88defca52a7c1d0a7984
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | ed9d3f13028beb3be39bce980c966f82c4b39dc73beaae38cc075fea5be30b0309e555cb2af8196014f2cc9f0df823354213c314b4d6545ff6e30dd2d00ec90e
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | ad5b3268db365dcdded9a9a4bffc90c7df0f844000349accdf2b8fb5f1081e553de9b9e9fb25d5e8a4ef7252d51fa94ef94d36d2ab31d157854e164136f662c2
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | c4de2524e513996def5eeba7b83f7b406f17eaf89d4d557833a93bd035348c81fa9375dcd5c27cfcc55d73995449fc8ee504be1b3bd7b9f108b0b2f153cb05ae
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 9157b44e3e7bd5478af9f72014e54d1afa5cd19b984b4cd8b348b312c385016bb77f29db47f44aea08b58abf47d8a396b92a2d0e03f2fe8acdd30f4f9466cbdb
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.20.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 8b40a43c5e6447379ad2ee8aac06e8028555e1b370a995f6001018a62411abe5fbbca6060b3d1682c5cadc07a27d49edd3204e797af46368800d55f4ca8aa1de

## Changelog since v1.20.0-alpha.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Azure blob disk feature(`kind`: `Shared`, `Dedicated`) has been deprecated, you should use `kind`: `Managed` in `kubernetes.io/azure-disk` storage class. ([#92905](https://github.com/kubernetes/kubernetes/pull/92905), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
  - CVE-2020-8559 (Medium): Privilege escalation from compromised node to cluster. See https://github.com/kubernetes/kubernetes/issues/92914 for more details.
  The API Server will no longer proxy non-101 responses for upgrade requests. This could break proxied backends (such as an extension API server) that respond to upgrade requests with a non-101 response code. ([#92941](https://github.com/kubernetes/kubernetes/pull/92941), [@tallclair](https://github.com/tallclair)) [SIG API Machinery]
 
## Changes by Kind

### Deprecation

- Kube-apiserver: the componentstatus API is deprecated. This API provided status of etcd, kube-scheduler, and kube-controller-manager components, but only worked when those components were local to the API server, and when kube-scheduler and kube-controller-manager exposed unsecured health endpoints. Instead of this API, etcd health is included in the kube-apiserver health check and kube-scheduler/kube-controller-manager health checks can be made directly against those components' health endpoints. ([#93570](https://github.com/kubernetes/kubernetes/pull/93570), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps and Cluster Lifecycle]
- Kubeadm: deprecate the "kubeadm alpha kubelet config enable-dynamic" command. To continue using the feature please defer to the guide for "Dynamic Kubelet Configuration" at k8s.io. ([#92881](https://github.com/kubernetes/kubernetes/pull/92881), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated "kubeadm alpha kubelet config enable-dynamic" command. To continue using the feature please defer to the guide for "Dynamic Kubelet Configuration" at k8s.io. This change also removes the parent command "kubeadm alpha kubelet" as there are no more sub-commands under it for the time being. ([#94668](https://github.com/kubernetes/kubernetes/pull/94668), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated --kubelet-config flag for the command "kubeadm upgrade node" ([#94869](https://github.com/kubernetes/kubernetes/pull/94869), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubelet's deprecated endpoint `metrics/resource/v1alpha1` has been removed, please adopt to `metrics/resource`. ([#94272](https://github.com/kubernetes/kubernetes/pull/94272), [@RainbowMango](https://github.com/RainbowMango)) [SIG Instrumentation and Node]
- The v1alpha1 PodPreset API and admission plugin has been removed with no built-in replacement. Admission webhooks can be used to modify pods on creation. ([#94090](https://github.com/kubernetes/kubernetes/pull/94090), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Apps, CLI, Cloud Provider, Scalability and Testing]

### API Change

- A new `nofuzz` go build tag now disables gofuzz support. Release binaries enable this. ([#92491](https://github.com/kubernetes/kubernetes/pull/92491), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery]
- A new alpha-level field, `SupportsFsGroup`, has been introduced for CSIDrivers to allow them to specify whether they support volume ownership and permission modifications. The `CSIVolumeSupportFSGroup` feature gate must be enabled to allow this field to be used. ([#92001](https://github.com/kubernetes/kubernetes/pull/92001), [@huffmanca](https://github.com/huffmanca)) [SIG API Machinery, CLI and Storage]
- Added pod version skew strategy for seccomp profile to synchronize the deprecated annotations with the new API Server fields. Please see the corresponding section [in the KEP](https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/20190717-seccomp-ga.md&#35;version-skew-strategy) for more detailed explanations. ([#91408](https://github.com/kubernetes/kubernetes/pull/91408), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps, Auth, CLI and Node]
- Adds the ability to disable Accelerator/GPU metrics collected by Kubelet ([#91930](https://github.com/kubernetes/kubernetes/pull/91930), [@RenaudWasTaken](https://github.com/RenaudWasTaken)) [SIG Node]
- Custom Endpoints are now mirrored to EndpointSlices by a new EndpointSliceMirroring controller. ([#91637](https://github.com/kubernetes/kubernetes/pull/91637), [@robscott](https://github.com/robscott)) [SIG API Machinery, Apps, Auth, Cloud Provider, Instrumentation, Network and Testing]
- External facing API podresources is now available under k8s.io/kubelet/pkg/apis/ ([#92632](https://github.com/kubernetes/kubernetes/pull/92632), [@RenaudWasTaken](https://github.com/RenaudWasTaken)) [SIG Node and Testing]
- Fix conversions for custom metrics. ([#94481](https://github.com/kubernetes/kubernetes/pull/94481), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Instrumentation]
- Generic ephemeral volumes, a new alpha feature under the `GenericEphemeralVolume` feature gate, provide a more flexible alternative to `EmptyDir` volumes: as with `EmptyDir`, volumes are created and deleted for each pod automatically by Kubernetes. But because the normal provisioning process is used (`PersistentVolumeClaim`), storage can be provided by third-party storage vendors and all of the usual volume features work. Volumes don't need to be empt; for example, restoring from snapshot is supported. ([#92784](https://github.com/kubernetes/kubernetes/pull/92784), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Instrumentation, Node, Scheduling, Storage and Testing]
- Kube-controller-manager: volume plugins can be restricted from contacting local and loopback addresses by setting `--volume-host-allow-local-loopback=false`, or from contacting specific CIDR ranges by setting `--volume-host-cidr-denylist` (for example, `--volume-host-cidr-denylist=127.0.0.1/28,feed::/16`) ([#91785](https://github.com/kubernetes/kubernetes/pull/91785), [@mattcary](https://github.com/mattcary)) [SIG API Machinery, Apps, Auth, CLI, Network, Node, Storage and Testing]
- Kubernetes is now built with golang 1.15.0-rc.1.
  - The deprecated, legacy behavior of treating the CommonName field on X.509 serving certificates as a host name when no Subject Alternative Names are present is now disabled by default. It can be temporarily re-enabled by adding the value x509ignoreCN=0 to the GODEBUG environment variable. ([#93264](https://github.com/kubernetes/kubernetes/pull/93264), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Storage and Testing]
- Migrate scheduler, controller-manager and cloud-controller-manager to use LeaseLock ([#94603](https://github.com/kubernetes/kubernetes/pull/94603), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Apps, Cloud Provider and Scheduling]
- Modify DNS-1123 error messages to indicate that RFC 1123 is not followed exactly ([#94182](https://github.com/kubernetes/kubernetes/pull/94182), [@mattfenwick](https://github.com/mattfenwick)) [SIG API Machinery, Apps, Auth, Network and Node]
- The ServiceAccountIssuerDiscovery feature gate is now Beta and enabled by default. ([#91921](https://github.com/kubernetes/kubernetes/pull/91921), [@mtaufen](https://github.com/mtaufen)) [SIG Auth]
- The kube-controller-manager managed signers can now have distinct signing certificates and keys.  See the help about `--cluster-signing-[signer-name]-{cert,key}-file`.  `--cluster-signing-{cert,key}-file` is still the default. ([#90822](https://github.com/kubernetes/kubernetes/pull/90822), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Apps and Auth]
- When creating a networking.k8s.io/v1 Ingress API object, `spec.tls[*].secretName` values are required to pass validation rules for Secret API object names. ([#93929](https://github.com/kubernetes/kubernetes/pull/93929), [@liggitt](https://github.com/liggitt)) [SIG Network]
- WinOverlay feature graduated to beta ([#94807](https://github.com/kubernetes/kubernetes/pull/94807), [@ksubrmnn](https://github.com/ksubrmnn)) [SIG Windows]

### Feature

- ACTION REQUIRED : In CoreDNS v1.7.0, [metrics names have been changed](https://github.com/coredns/coredns/blob/master/notes/coredns-1.7.0.md&#35;metric-changes) which will be backward incompatible with existing reporting formulas that use the old metrics' names. Adjust your formulas to the new names before upgrading. 
  
  Kubeadm now includes CoreDNS version v1.7.0. Some of the major changes include:
  -  Fixed a bug that could cause CoreDNS to stop updating service records.
  -  Fixed a bug in the forward plugin where only the first upstream server is always selected no matter which policy is set.
  -  Remove already deprecated options `resyncperiod` and `upstream` in the Kubernetes plugin.
  -  Includes Prometheus metrics name changes (to bring them in line with standard Prometheus metrics naming convention). They will be backward incompatible with existing reporting formulas that use the old metrics' names.
  -  The federation plugin (allows for v1 Kubernetes federation) has been removed.
  More details are available in https://coredns.io/2020/06/15/coredns-1.7.0-release/ ([#92651](https://github.com/kubernetes/kubernetes/pull/92651), [@rajansandeep](https://github.com/rajansandeep)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Add metrics for azure service operations (route and loadbalancer). ([#94124](https://github.com/kubernetes/kubernetes/pull/94124), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider and Instrumentation]
- Add network rule support in Azure account creation ([#94239](https://github.com/kubernetes/kubernetes/pull/94239), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Add tags support for Azure File Driver ([#92825](https://github.com/kubernetes/kubernetes/pull/92825), [@ZeroMagic](https://github.com/ZeroMagic)) [SIG Cloud Provider and Storage]
- Added kube-apiserver metrics: apiserver_current_inflight_request_measures and, when API Priority and Fairness is enable, windowed_request_stats. ([#91177](https://github.com/kubernetes/kubernetes/pull/91177), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery, Instrumentation and Testing]
- Audit events for API requests to deprecated API versions now include a `"k8s.io/deprecated": "true"` audit annotation. If a target removal release is identified, the audit event includes a `"k8s.io/removal-release": "<majorVersion>.<minorVersion>"` audit annotation as well. ([#92842](https://github.com/kubernetes/kubernetes/pull/92842), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Instrumentation]
- Cloud node-controller use InstancesV2 ([#91319](https://github.com/kubernetes/kubernetes/pull/91319), [@gongguan](https://github.com/gongguan)) [SIG Apps, Cloud Provider, Scalability and Storage]
- Kubeadm: Add a preflight check that the control-plane node has at least 1700MB of RAM ([#93275](https://github.com/kubernetes/kubernetes/pull/93275), [@xlgao-zju](https://github.com/xlgao-zju)) [SIG Cluster Lifecycle]
- Kubeadm: add the "--cluster-name" flag to the "kubeadm alpha kubeconfig user" to allow configuring the cluster name in the generated kubeconfig file ([#93992](https://github.com/kubernetes/kubernetes/pull/93992), [@prabhu43](https://github.com/prabhu43)) [SIG Cluster Lifecycle]
- Kubeadm: add the "--kubeconfig" flag to the "kubeadm init phase upload-certs" command to allow users to pass a custom location for a kubeconfig file. ([#94765](https://github.com/kubernetes/kubernetes/pull/94765), [@zhanw15](https://github.com/zhanw15)) [SIG Cluster Lifecycle]
- Kubeadm: deprecate the "--csr-only" and "--csr-dir" flags of the "kubeadm init phase certs" subcommands. Please use "kubeadm alpha certs generate-csr" instead. This new command allows you to generate new private keys and certificate signing requests for all the control-plane components, so that the certificates can be signed by an external CA. ([#92183](https://github.com/kubernetes/kubernetes/pull/92183), [@wallrj](https://github.com/wallrj)) [SIG Cluster Lifecycle]
- Kubeadm: make etcd pod request 100m CPU, 100Mi memory and 100Mi ephemeral_storage by default ([#94479](https://github.com/kubernetes/kubernetes/pull/94479), [@knight42](https://github.com/knight42)) [SIG Cluster Lifecycle]
- Kubemark now supports both real and hollow nodes in a single cluster. ([#93201](https://github.com/kubernetes/kubernetes/pull/93201), [@ellistarn](https://github.com/ellistarn)) [SIG Scalability]
- Kubernetes is now built using go1.15.2
  - build: Update to k/repo-infra@v0.1.1 (supports go1.15.2)
  - build: Use go-runner:buster-v2.0.1 (built using go1.15.1)
  - bazel: Replace --features with Starlark build settings flag
  - hack/lib/util.sh: some bash cleanups
    
    - switched one spot to use kube::logging
    - make kube::util::find-binary return an error when it doesn't find
      anything so that hack scripts fail fast instead of with '' binary not
      found errors.
    - this required deleting some genfeddoc stuff. the binary no longer
      exists in k/k repo since we removed federation/, and I don't see it
      in https://github.com/kubernetes-sigs/kubefed/ either. I'm assuming
      that it's gone for good now.
  
  - bazel: output go_binary rule directly from go_binary_conditional_pure
    
    From: @mikedanese:
    Instead of aliasing. Aliases are annoying in a number of ways. This is
    specifically bugging me now because they make the action graph harder to
    analyze programmatically. By using aliases here, we would need to handle
    potentially aliased go_binary targets and dereference to the effective
    target.
  
    The comment references an issue with `pure = select(...)` which appears
    to be resolved considering this now builds.
  
  - make kube::util::find-binary not dependent on bazel-out/ structure
  
    Implement an aspect that outputs go_build_mode metadata for go binaries,
    and use that during binary selection. ([#94449](https://github.com/kubernetes/kubernetes/pull/94449), [@justaugustus](https://github.com/justaugustus)) [SIG Architecture, CLI, Cluster Lifecycle, Node, Release and Testing]
- Only update Azure data disks when attach/detach ([#94265](https://github.com/kubernetes/kubernetes/pull/94265), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Promote SupportNodePidsLimit to GA to provide node to pod pid isolation
  Promote SupportPodPidsLimit to GA to provide ability to limit pids per pod ([#94140](https://github.com/kubernetes/kubernetes/pull/94140), [@derekwaynecarr](https://github.com/derekwaynecarr)) [SIG Node and Testing]
- Rename pod_preemption_metrics to preemption_metrics. ([#93256](https://github.com/kubernetes/kubernetes/pull/93256), [@ahg-g](https://github.com/ahg-g)) [SIG Instrumentation and Scheduling]
- Server-side apply behavior has been regularized in the case where a field is removed from the applied configuration. Removed fields which have no other owners are deleted from the live object, or reset to their default value if they have one. Safe ownership transfers, such as the transfer of a `replicas` field from a user to an HPA without resetting to the default value are documented in [Transferring Ownership](https://kubernetes.io/docs/reference/using-api/api-concepts/&#35;transferring-ownership) ([#92661](https://github.com/kubernetes/kubernetes/pull/92661), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Testing]
- Set CSIMigrationvSphere feature gates to beta.
  Users should enable CSIMigration + CSIMigrationvSphere features and install the vSphere CSI Driver (https://github.com/kubernetes-sigs/vsphere-csi-driver) to move workload from the in-tree vSphere plugin "kubernetes.io/vsphere-volume" to vSphere CSI Driver.
  
  Requires: vSphere vCenter/ESXi Version: 7.0u1, HW Version: VM version 15 ([#92816](https://github.com/kubernetes/kubernetes/pull/92816), [@divyenpatel](https://github.com/divyenpatel)) [SIG Cloud Provider and Storage]
- Support [service.beta.kubernetes.io/azure-pip-ip-tags] annotations to allow customers to specify ip-tags to influence public-ip creation in Azure [Tag1=Value1, Tag2=Value2, etc.] ([#94114](https://github.com/kubernetes/kubernetes/pull/94114), [@MarcPow](https://github.com/MarcPow)) [SIG Cloud Provider]
- Support a smooth upgrade from client-side apply to server-side apply without conflicts, as well as support the corresponding downgrade. ([#90187](https://github.com/kubernetes/kubernetes/pull/90187), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG API Machinery and Testing]
- Trace output in apiserver logs is more organized and comprehensive. Traces are nested, and for all non-long running request endpoints, the entire filter chain is instrumented (e.g. authentication check is included). ([#88936](https://github.com/kubernetes/kubernetes/pull/88936), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Scheduling]
- `kubectl alpha debug` now supports debugging nodes by creating a debugging container running in the node's host namespaces. ([#92310](https://github.com/kubernetes/kubernetes/pull/92310), [@verb](https://github.com/verb)) [SIG CLI]

### Documentation

- Kubelet: remove alpha warnings for CNI flags. ([#94508](https://github.com/kubernetes/kubernetes/pull/94508), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network and Node]

### Failing Test

- Kube-proxy iptables min-sync-period defaults to 1 sec. Previously, it was 0. ([#92836](https://github.com/kubernetes/kubernetes/pull/92836), [@aojea](https://github.com/aojea)) [SIG Network]

### Bug or Regression

- A panic in the apiserver caused by the `informer-sync` health checker is now fixed. ([#93600](https://github.com/kubernetes/kubernetes/pull/93600), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- Add kubectl wait  --ignore-not-found flag ([#90969](https://github.com/kubernetes/kubernetes/pull/90969), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Adding fix to the statefulset controller to wait for pvc deletion before creating pods. ([#93457](https://github.com/kubernetes/kubernetes/pull/93457), [@ymmt2005](https://github.com/ymmt2005)) [SIG Apps]
- Azure ARM client: don't segfault on empty response and http error ([#94078](https://github.com/kubernetes/kubernetes/pull/94078), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- Azure: fix a bug that kube-controller-manager would panic if wrong Azure VMSS name is configured ([#94306](https://github.com/kubernetes/kubernetes/pull/94306), [@knight42](https://github.com/knight42)) [SIG Cloud Provider]
- Azure: per VMSS VMSS VMs cache to prevent throttling on clusters having many attached VMSS ([#93107](https://github.com/kubernetes/kubernetes/pull/93107), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- Both apiserver_request_duration_seconds metrics and RequestReceivedTimestamp field of an audit event take 
  into account the time a request spends in the apiserver request filters. ([#94903](https://github.com/kubernetes/kubernetes/pull/94903), [@tkashem](https://github.com/tkashem)) [SIG API Machinery, Auth and Instrumentation]
- Build/lib/release: Explicitly use '--platform' in building server images
  
  When we switched to go-runner for building the apiserver,
  controller-manager, and scheduler server components, we no longer
  reference the individual architectures in the image names, specifically
  in the 'FROM' directive of the server image Dockerfiles.
  
  As a result, server images for non-amd64 images copy in the go-runner
  amd64 binary instead of the go-runner that matches that architecture.
  
  This commit explicitly sets the '--platform=linux/${arch}' to ensure
  we're pulling the correct go-runner arch from the manifest list.
  
  Before:
  `FROM ${base_image}`
  
  After:
  `FROM --platform=linux/${arch} ${base_image}` ([#94552](https://github.com/kubernetes/kubernetes/pull/94552), [@justaugustus](https://github.com/justaugustus)) [SIG Release]
- CSIDriver object can be deployed during volume attachment. ([#93710](https://github.com/kubernetes/kubernetes/pull/93710), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG Apps, Node, Storage and Testing]
- CVE-2020-8557 (Medium): Node-local denial of service via container /etc/hosts file. See https://github.com/kubernetes/kubernetes/issues/93032 for more details. ([#92916](https://github.com/kubernetes/kubernetes/pull/92916), [@joelsmith](https://github.com/joelsmith)) [SIG Node]
- Do not add nodes labeled with kubernetes.azure.com/managed=false to backend pool of load balancer. ([#93034](https://github.com/kubernetes/kubernetes/pull/93034), [@matthias50](https://github.com/matthias50)) [SIG Cloud Provider]
- Do not fail sorting empty elements. ([#94666](https://github.com/kubernetes/kubernetes/pull/94666), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- Do not retry volume expansion if CSI driver returns FailedPrecondition error ([#92986](https://github.com/kubernetes/kubernetes/pull/92986), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Dockershim security: pod sandbox now always run with `no-new-privileges` and `runtime/default` seccomp profile
  dockershim seccomp: custom profiles can now have smaller seccomp profiles when set at pod level ([#90948](https://github.com/kubernetes/kubernetes/pull/90948), [@pjbgf](https://github.com/pjbgf)) [SIG Node]
- Dual-stack: make nodeipam compatible with existing single-stack clusters when dual-stack feature gate become enabled by default ([#90439](https://github.com/kubernetes/kubernetes/pull/90439), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery]
- Endpoint controller requeues service after an endpoint deletion event occurs to confirm that deleted endpoints are undesired to mitigate the effects of an out of sync endpoint cache. ([#93030](https://github.com/kubernetes/kubernetes/pull/93030), [@swetharepakula](https://github.com/swetharepakula)) [SIG Apps and Network]
- EndpointSlice controllers now return immediately if they encounter an error creating, updating, or deleting resources. ([#93908](https://github.com/kubernetes/kubernetes/pull/93908), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- EndpointSliceMirroring controller now copies labels from Endpoints to EndpointSlices. ([#93442](https://github.com/kubernetes/kubernetes/pull/93442), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- EndpointSliceMirroring controller now mirrors Endpoints that do not have a Service associated with them. ([#94171](https://github.com/kubernetes/kubernetes/pull/94171), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- Ensure backoff step is set to 1 for Azure armclient. ([#94180](https://github.com/kubernetes/kubernetes/pull/94180), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Ensure getPrimaryInterfaceID not panic when network interfaces for Azure VMSS are null ([#94355](https://github.com/kubernetes/kubernetes/pull/94355), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Eviction requests for pods that have a non-zero DeletionTimestamp will always succeed ([#91342](https://github.com/kubernetes/kubernetes/pull/91342), [@michaelgugino](https://github.com/michaelgugino)) [SIG Apps]
- Extended DSR loadbalancer feature in winkernel kube-proxy to HNS versions 9.3-9.max, 10.2+ ([#93080](https://github.com/kubernetes/kubernetes/pull/93080), [@elweb9858](https://github.com/elweb9858)) [SIG Network]
- Fix HandleCrash order ([#93108](https://github.com/kubernetes/kubernetes/pull/93108), [@lixiaobing1](https://github.com/lixiaobing1)) [SIG API Machinery]
- Fix a concurrent map writes error in kubelet ([#93773](https://github.com/kubernetes/kubernetes/pull/93773), [@knight42](https://github.com/knight42)) [SIG Node]
- Fix a regression where kubeadm bails out with a fatal error when an optional version command line argument is supplied to the "kubeadm upgrade plan" command ([#94421](https://github.com/kubernetes/kubernetes/pull/94421), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Fix azure file migration panic ([#94853](https://github.com/kubernetes/kubernetes/pull/94853), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix bug where loadbalancer deletion gets stuck because of missing resource group &#35;75198 ([#93962](https://github.com/kubernetes/kubernetes/pull/93962), [@phiphi282](https://github.com/phiphi282)) [SIG Cloud Provider]
- Fix calling AttachDisk on a previously attached EBS volume ([#93567](https://github.com/kubernetes/kubernetes/pull/93567), [@gnufied](https://github.com/gnufied)) [SIG Cloud Provider, Storage and Testing]
- Fix detection of image filesystem, disk metrics for devicemapper, detection of OOM Kills on 5.0+ linux kernels. ([#92919](https://github.com/kubernetes/kubernetes/pull/92919), [@dashpole](https://github.com/dashpole)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Node]
- Fix etcd_object_counts metric reported by kube-apiserver ([#94773](https://github.com/kubernetes/kubernetes/pull/94773), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fix incorrectly reported verbs for kube-apiserver metrics for CRD objects ([#93523](https://github.com/kubernetes/kubernetes/pull/93523), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Instrumentation]
- Fix instance not found issues when an Azure Node is recreated in a short time ([#93316](https://github.com/kubernetes/kubernetes/pull/93316), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix kube-apiserver /readyz to contain "informer-sync" check ensuring that internal informers are synced. ([#93670](https://github.com/kubernetes/kubernetes/pull/93670), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Testing]
- Fix kubectl SchemaError on CRDs with schema using x-kubernetes-preserve-unknown-fields on array types. ([#94888](https://github.com/kubernetes/kubernetes/pull/94888), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Fix memory leak in EndpointSliceTracker for EndpointSliceMirroring controller. ([#93441](https://github.com/kubernetes/kubernetes/pull/93441), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Fix missing csi annotations on node during parallel csinode update. ([#94389](https://github.com/kubernetes/kubernetes/pull/94389), [@pacoxu](https://github.com/pacoxu)) [SIG Storage]
- Fix the `cloudprovider_azure_api_request_duration_seconds` metric buckets to correctly capture the latency metrics. Previously, the majority of the calls would fall in the "+Inf" bucket. ([#94873](https://github.com/kubernetes/kubernetes/pull/94873), [@marwanad](https://github.com/marwanad)) [SIG Cloud Provider and Instrumentation]
- Fix: azure disk resize error if source does not exist ([#93011](https://github.com/kubernetes/kubernetes/pull/93011), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: detach azure disk broken on Azure Stack ([#94885](https://github.com/kubernetes/kubernetes/pull/94885), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: determine the correct ip config based on ip family ([#93043](https://github.com/kubernetes/kubernetes/pull/93043), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Fix: initial delay in mounting azure disk & file ([#93052](https://github.com/kubernetes/kubernetes/pull/93052), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: use sensitiveOptions on Windows mount ([#94126](https://github.com/kubernetes/kubernetes/pull/94126), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fixed Ceph RBD volume expansion when no ceph.conf exists ([#92027](https://github.com/kubernetes/kubernetes/pull/92027), [@juliantaylor](https://github.com/juliantaylor)) [SIG Storage]
- Fixed a bug where improper storage and comparison of endpoints led to excessive API traffic from the endpoints controller ([#94112](https://github.com/kubernetes/kubernetes/pull/94112), [@damemi](https://github.com/damemi)) [SIG Apps, Network and Testing]
- Fixed a bug whereby the allocation of reusable CPUs and devices was not being honored when the TopologyManager was enabled ([#93189](https://github.com/kubernetes/kubernetes/pull/93189), [@klueska](https://github.com/klueska)) [SIG Node]
- Fixed a panic in kubectl debug when pod has multiple init containers or ephemeral containers ([#94580](https://github.com/kubernetes/kubernetes/pull/94580), [@kiyoshim55](https://github.com/kiyoshim55)) [SIG CLI]
- Fixed a regression that sometimes prevented `kubectl portforward` to work when TCP and UDP services were configured on the same port ([#94728](https://github.com/kubernetes/kubernetes/pull/94728), [@amorenoz](https://github.com/amorenoz)) [SIG CLI]
- Fixed bug in reflector that couldn't recover from "Too large resource version" errors with API servers 1.17.0-1.18.5 ([#94316](https://github.com/kubernetes/kubernetes/pull/94316), [@janeczku](https://github.com/janeczku)) [SIG API Machinery]
- Fixed bug where kubectl top pod output is not sorted when --sort-by and --containers flags are used together ([#93692](https://github.com/kubernetes/kubernetes/pull/93692), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Fixed kubelet creating extra sandbox for pods with RestartPolicyOnFailure after all containers succeeded ([#92614](https://github.com/kubernetes/kubernetes/pull/92614), [@tnqn](https://github.com/tnqn)) [SIG Node and Testing]
- Fixed memory leak in endpointSliceTracker ([#92838](https://github.com/kubernetes/kubernetes/pull/92838), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Fixed node data lost in kube-scheduler for clusters with imbalance on number of nodes across zones ([#93355](https://github.com/kubernetes/kubernetes/pull/93355), [@maelk](https://github.com/maelk)) [SIG Scheduling]
- Fixed the EndpointSliceController to correctly create endpoints for IPv6-only pods.
  
  Fixed the EndpointController to allow IPv6 headless services, if the IPv6DualStack
  feature gate is enabled, by specifying `ipFamily: IPv6` on the service. (This already
  worked with the EndpointSliceController.) ([#91399](https://github.com/kubernetes/kubernetes/pull/91399), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]
- Fixes a bug evicting pods after a taint with a limited tolerationSeconds toleration is removed from a node ([#93722](https://github.com/kubernetes/kubernetes/pull/93722), [@liggitt](https://github.com/liggitt)) [SIG Apps and Node]
- Fixes a bug where EndpointSlices would not be recreated after rapid Service recreation. ([#94730](https://github.com/kubernetes/kubernetes/pull/94730), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- Fixes a race condition in kubelet pod handling ([#94751](https://github.com/kubernetes/kubernetes/pull/94751), [@auxten](https://github.com/auxten)) [SIG Node]
- Fixes an issue proxying to ipv6 pods without specifying a port ([#94834](https://github.com/kubernetes/kubernetes/pull/94834), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Network]
- Fixes an issue that can result in namespaced custom resources being orphaned when their namespace is deleted, if the CRD defining the custom resource is removed concurrently with namespaces being deleted, then recreated. ([#93790](https://github.com/kubernetes/kubernetes/pull/93790), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Apps]
- Ignore root user check when windows pod starts ([#92355](https://github.com/kubernetes/kubernetes/pull/92355), [@wawa0210](https://github.com/wawa0210)) [SIG Node and Windows]
- Increased maximum IOPS of AWS EBS io1 volumes to 64,000 (current AWS maximum). ([#90014](https://github.com/kubernetes/kubernetes/pull/90014), [@jacobmarble](https://github.com/jacobmarble)) [SIG Cloud Provider and Storage]
- K8s.io/apimachinery: runtime.DefaultUnstructuredConverter.FromUnstructured now handles converting integer fields to typed float values ([#93250](https://github.com/kubernetes/kubernetes/pull/93250), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-aggregator certificates are dynamically loaded on change from disk ([#92791](https://github.com/kubernetes/kubernetes/pull/92791), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kube-apiserver: fixed a bug returning inconsistent results from list requests which set a field or label selector and set a paging limit ([#94002](https://github.com/kubernetes/kubernetes/pull/94002), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Kube-apiserver: jsonpath expressions with consecutive recursive descent operators are no longer evaluated for custom resource printer columns ([#93408](https://github.com/kubernetes/kubernetes/pull/93408), [@joelsmith](https://github.com/joelsmith)) [SIG API Machinery]
- Kube-proxy now trims extra spaces found in loadBalancerSourceRanges to match Service validation. ([#94107](https://github.com/kubernetes/kubernetes/pull/94107), [@robscott](https://github.com/robscott)) [SIG Network]
- Kube-up now includes CoreDNS version v1.7.0. Some of the major changes include:
  -  Fixed a bug that could cause CoreDNS to stop updating service records.
  -  Fixed a bug in the forward plugin where only the first upstream server is always selected no matter which policy is set.
  -  Remove already deprecated options `resyncperiod` and `upstream` in the Kubernetes plugin.
  -  Includes Prometheus metrics name changes (to bring them in line with standard Prometheus metrics naming convention). They will be backward incompatible with existing reporting formulas that use the old metrics' names.
  -  The federation plugin (allows for v1 Kubernetes federation) has been removed.
  More details are available in https://coredns.io/2020/06/15/coredns-1.7.0-release/ ([#92718](https://github.com/kubernetes/kubernetes/pull/92718), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cloud Provider]
- Kubeadm now makes sure the etcd manifest is regenerated upon upgrade even when no etcd version change takes place ([#94395](https://github.com/kubernetes/kubernetes/pull/94395), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: avoid a panic when determining if the running version of CoreDNS is supported during upgrades ([#94299](https://github.com/kubernetes/kubernetes/pull/94299), [@zouyee](https://github.com/zouyee)) [SIG Cluster Lifecycle]
- Kubeadm: ensure "kubeadm reset" does not unmount the root "/var/lib/kubelet" directory if it is mounted by the user ([#93702](https://github.com/kubernetes/kubernetes/pull/93702), [@thtanaka](https://github.com/thtanaka)) [SIG Cluster Lifecycle]
- Kubeadm: ensure the etcd data directory is created with 0700 permissions during control-plane init and join ([#94102](https://github.com/kubernetes/kubernetes/pull/94102), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that kubeadm tries to call 'docker info' even if the CRI socket was for another CR ([#94555](https://github.com/kubernetes/kubernetes/pull/94555), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: make the kubeconfig files for the kube-controller-manager and kube-scheduler use the LocalAPIEndpoint instead of the ControlPlaneEndpoint. This makes kubeadm clusters more reseliant to version skew problems during immutable upgrades: https://kubernetes.io/docs/setup/release/version-skew-policy/&#35;kube-controller-manager-kube-scheduler-and-cloud-controller-manager ([#94398](https://github.com/kubernetes/kubernetes/pull/94398), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: relax the validation of kubeconfig server URLs. Allow the user to define custom kubeconfig server URLs without erroring out during validation of existing kubeconfig files (e.g. when using external CA mode). ([#94816](https://github.com/kubernetes/kubernetes/pull/94816), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove duplicate DNS names and IP addresses from generated certificates ([#92753](https://github.com/kubernetes/kubernetes/pull/92753), [@QianChenglong](https://github.com/QianChenglong)) [SIG Cluster Lifecycle]
- Kubelet: assume that swap is disabled when `/proc/swaps` does not exist ([#93931](https://github.com/kubernetes/kubernetes/pull/93931), [@SataQiu](https://github.com/SataQiu)) [SIG Node]
- Kubelet: fix race condition in pluginWatcher ([#93622](https://github.com/kubernetes/kubernetes/pull/93622), [@knight42](https://github.com/knight42)) [SIG Node]
- Kuberuntime security: pod sandbox now always runs with `runtime/default` seccomp profile
  kuberuntime seccomp: custom profiles can now have smaller seccomp profiles when set at pod level ([#90949](https://github.com/kubernetes/kubernetes/pull/90949), [@pjbgf](https://github.com/pjbgf)) [SIG Node]
- NONE ([#71269](https://github.com/kubernetes/kubernetes/pull/71269), [@DeliangFan](https://github.com/DeliangFan)) [SIG Node]
- New Azure instance types do now have correct max data disk count information. ([#94340](https://github.com/kubernetes/kubernetes/pull/94340), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Cloud Provider and Storage]
- Pods with invalid Affinity/AntiAffinity LabelSelectors will now fail scheduling when these plugins are enabled ([#93660](https://github.com/kubernetes/kubernetes/pull/93660), [@damemi](https://github.com/damemi)) [SIG Scheduling]
- Require feature flag CustomCPUCFSQuotaPeriod if setting a non-default cpuCFSQuotaPeriod in kubelet config. ([#94687](https://github.com/kubernetes/kubernetes/pull/94687), [@karan](https://github.com/karan)) [SIG Node]
- Reverted devicemanager for Windows node added in 1.19rc1. ([#93263](https://github.com/kubernetes/kubernetes/pull/93263), [@liggitt](https://github.com/liggitt)) [SIG Node and Windows]
- Scheduler bugfix: Scheduler doesn't lose pod information when nodes are quickly recreated. This could happen when nodes are restarted or quickly recreated reusing a nodename. ([#93938](https://github.com/kubernetes/kubernetes/pull/93938), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scalability, Scheduling and Testing]
- The EndpointSlice controller now waits for EndpointSlice and Node caches to be synced before starting. ([#94086](https://github.com/kubernetes/kubernetes/pull/94086), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- The `/debug/api_priority_and_fairness/dump_requests` path at an apiserver will no longer return a phantom line for each exempt priority level. ([#93406](https://github.com/kubernetes/kubernetes/pull/93406), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery]
- The kubelet recognizes the --containerd-namespace flag to configure the namespace used by cadvisor. ([#87054](https://github.com/kubernetes/kubernetes/pull/87054), [@changyaowei](https://github.com/changyaowei)) [SIG Node]
- The terminationGracePeriodSeconds from pod spec is respected for the mirror pod. ([#92442](https://github.com/kubernetes/kubernetes/pull/92442), [@tedyu](https://github.com/tedyu)) [SIG Node and Testing]
- Update Calico to v3.15.2 ([#94241](https://github.com/kubernetes/kubernetes/pull/94241), [@lmm](https://github.com/lmm)) [SIG Cloud Provider]
- Update default etcd server version to 3.4.13 ([#94287](https://github.com/kubernetes/kubernetes/pull/94287), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Updated Cluster Autoscaler to 1.19.0; ([#93577](https://github.com/kubernetes/kubernetes/pull/93577), [@vivekbagade](https://github.com/vivekbagade)) [SIG Autoscaling and Cloud Provider]
- Use NLB Subnet CIDRs instead of VPC CIDRs in Health Check SG Rules ([#93515](https://github.com/kubernetes/kubernetes/pull/93515), [@t0rr3sp3dr0](https://github.com/t0rr3sp3dr0)) [SIG Cloud Provider]
- Users will see increase in time for deletion of pods and also guarantee that removal of pod from api server  would mean deletion of all the resources from container runtime. ([#92817](https://github.com/kubernetes/kubernetes/pull/92817), [@kmala](https://github.com/kmala)) [SIG Node]
- Very large patches may now be specified to `kubectl patch` with the `--patch-file` flag instead of including them directly on the command line. The `--patch` and `--patch-file` flags are mutually exclusive. ([#93548](https://github.com/kubernetes/kubernetes/pull/93548), [@smarterclayton](https://github.com/smarterclayton)) [SIG CLI]
- When creating a networking.k8s.io/v1 Ingress API object, `spec.rules[*].http` values are now validated consistently when the `host` field contains a wildcard. ([#93954](https://github.com/kubernetes/kubernetes/pull/93954), [@Miciah](https://github.com/Miciah)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Storage and Testing]

### Other (Cleanup or Flake)

- --cache-dir sets cache directory for both http and discovery, defaults to $HOME/.kube/cache ([#92910](https://github.com/kubernetes/kubernetes/pull/92910), [@soltysh](https://github.com/soltysh)) [SIG API Machinery and CLI]
- Adds a bootstrapping ClusterRole, ClusterRoleBinding and group for /metrics, /livez/*, /readyz/*, & /healthz/- endpoints. ([#93311](https://github.com/kubernetes/kubernetes/pull/93311), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery, Auth, Cloud Provider and Instrumentation]
- Base-images: Update to debian-iptables:buster-v1.3.0
  - Uses iptables 1.8.5
  - base-images: Update to debian-base:buster-v1.2.0
  - cluster/images/etcd: Build etcd:3.4.13-1 image
    - Uses debian-base:buster-v1.2.0 ([#94733](https://github.com/kubernetes/kubernetes/pull/94733), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Release and Testing]
- Build: Update to debian-base@v2.1.2 and debian-iptables@v12.1.1 ([#93667](https://github.com/kubernetes/kubernetes/pull/93667), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Release and Testing]
- Build: Update to debian-base@v2.1.3 and debian-iptables@v12.1.2 ([#93916](https://github.com/kubernetes/kubernetes/pull/93916), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Release and Testing]
- Build: Update to go-runner:buster-v2.0.0 ([#94167](https://github.com/kubernetes/kubernetes/pull/94167), [@justaugustus](https://github.com/justaugustus)) [SIG Release]
- Fix kubelet to properly log when a container is started. Before, sometimes the log said that a container is dead and was restarted when it was started for the first time. This only happened when using pods with initContainers and regular containers. ([#91469](https://github.com/kubernetes/kubernetes/pull/91469), [@rata](https://github.com/rata)) [SIG Node]
- Fix: license issue in blob disk feature ([#92824](https://github.com/kubernetes/kubernetes/pull/92824), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fixes the flooding warning messages about setting volume ownership for configmap/secret volumes ([#92878](https://github.com/kubernetes/kubernetes/pull/92878), [@jvanz](https://github.com/jvanz)) [SIG Instrumentation, Node and Storage]
- Fixes the message about no auth for metrics in scheduler. ([#94035](https://github.com/kubernetes/kubernetes/pull/94035), [@zhouya0](https://github.com/zhouya0)) [SIG Scheduling]
- Kube-up: defaults to limiting critical pods to the kube-system namespace to match behavior prior to 1.17 ([#93121](https://github.com/kubernetes/kubernetes/pull/93121), [@liggitt](https://github.com/liggitt)) [SIG Cloud Provider and Scheduling]
- Kubeadm: Separate argument key/value in log msg ([#94016](https://github.com/kubernetes/kubernetes/pull/94016), [@mrueg](https://github.com/mrueg)) [SIG Cluster Lifecycle]
- Kubeadm: remove support for the "ci/k8s-master" version label. This label has been removed in the Kubernetes CI release process and would no longer work in kubeadm. You can use the "ci/latest" version label instead. See kubernetes/test-infra&#35;18517 ([#93626](https://github.com/kubernetes/kubernetes/pull/93626), [@vikkyomkar](https://github.com/vikkyomkar)) [SIG Cluster Lifecycle]
- Kubeadm: remove the CoreDNS check for known image digests when applying the addon ([#94506](https://github.com/kubernetes/kubernetes/pull/94506), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes is now built with go1.15.0 ([#93939](https://github.com/kubernetes/kubernetes/pull/93939), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Kubernetes is now built with go1.15.0-rc.2 ([#93827](https://github.com/kubernetes/kubernetes/pull/93827), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Release and Testing]
- Lock ExternalPolicyForExternalIP to default, this feature gate will be removed in 1.22. ([#94581](https://github.com/kubernetes/kubernetes/pull/94581), [@knabben](https://github.com/knabben)) [SIG Network]
- Service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset is removed.  All Standard load balancers will always enable tcp resets. ([#94297](https://github.com/kubernetes/kubernetes/pull/94297), [@MarcPow](https://github.com/MarcPow)) [SIG Cloud Provider]
- Stop propagating SelfLink (deprecated in 1.16) in kube-apiserver ([#94397](https://github.com/kubernetes/kubernetes/pull/94397), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Testing]
- Strip unnecessary security contexts on Windows ([#93475](https://github.com/kubernetes/kubernetes/pull/93475), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Node, Testing and Windows]
- To ensure the code be strong,  add unit test for GetAddressAndDialer ([#93180](https://github.com/kubernetes/kubernetes/pull/93180), [@FreeZhang61](https://github.com/FreeZhang61)) [SIG Node]
- Update CNI plugins to v0.8.7 ([#94367](https://github.com/kubernetes/kubernetes/pull/94367), [@justaugustus](https://github.com/justaugustus)) [SIG Cloud Provider, Network, Node, Release and Testing]
- Update Golang to v1.14.5
  - Update repo-infra to 0.0.7 (to support go1.14.5 and go1.13.13)
    - Includes:
      - bazelbuild/bazel-toolchains@3.3.2
      - bazelbuild/rules_go@v0.22.7 ([#93088](https://github.com/kubernetes/kubernetes/pull/93088), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Update Golang to v1.14.6
  - Update repo-infra to 0.0.8 (to support go1.14.6 and go1.13.14)
    - Includes:
      - bazelbuild/bazel-toolchains@3.4.0
      - bazelbuild/rules_go@v0.22.8 ([#93198](https://github.com/kubernetes/kubernetes/pull/93198), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Update cri-tools to [v1.19.0](https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.19.0) ([#94307](https://github.com/kubernetes/kubernetes/pull/94307), [@xmudrii](https://github.com/xmudrii)) [SIG Cloud Provider]
- Update default etcd server version to 3.4.9 ([#92349](https://github.com/kubernetes/kubernetes/pull/92349), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Update etcd client side to v3.4.13 ([#94259](https://github.com/kubernetes/kubernetes/pull/94259), [@jingyih](https://github.com/jingyih)) [SIG API Machinery and Cloud Provider]
- `kubectl get ingress` now prefers the `networking.k8s.io/v1` over `extensions/v1beta1` (deprecated since v1.14). To explicitly request the deprecated version, use `kubectl get ingress.v1beta1.extensions`. ([#94309](https://github.com/kubernetes/kubernetes/pull/94309), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and CLI]

## Dependencies

### Added
- github.com/Azure/go-autorest: [v14.2.0+incompatible](https://github.com/Azure/go-autorest/tree/v14.2.0)
- github.com/fvbommel/sortorder: [v1.0.1](https://github.com/fvbommel/sortorder/tree/v1.0.1)
- github.com/yuin/goldmark: [v1.1.27](https://github.com/yuin/goldmark/tree/v1.1.27)
- sigs.k8s.io/structured-merge-diff/v4: v4.0.1

### Changed
- github.com/Azure/go-autorest/autorest/adal: [v0.8.2 → v0.9.0](https://github.com/Azure/go-autorest/autorest/adal/compare/v0.8.2...v0.9.0)
- github.com/Azure/go-autorest/autorest/date: [v0.2.0 → v0.3.0](https://github.com/Azure/go-autorest/autorest/date/compare/v0.2.0...v0.3.0)
- github.com/Azure/go-autorest/autorest/mocks: [v0.3.0 → v0.4.0](https://github.com/Azure/go-autorest/autorest/mocks/compare/v0.3.0...v0.4.0)
- github.com/Azure/go-autorest/autorest: [v0.9.6 → v0.11.1](https://github.com/Azure/go-autorest/autorest/compare/v0.9.6...v0.11.1)
- github.com/Azure/go-autorest/logger: [v0.1.0 → v0.2.0](https://github.com/Azure/go-autorest/logger/compare/v0.1.0...v0.2.0)
- github.com/Azure/go-autorest/tracing: [v0.5.0 → v0.6.0](https://github.com/Azure/go-autorest/tracing/compare/v0.5.0...v0.6.0)
- github.com/Microsoft/hcsshim: [v0.8.9 → 5eafd15](https://github.com/Microsoft/hcsshim/compare/v0.8.9...5eafd15)
- github.com/cilium/ebpf: [9f1617e → 1c8d4c9](https://github.com/cilium/ebpf/compare/9f1617e...1c8d4c9)
- github.com/containerd/cgroups: [bf292b2 → 0dbf7f0](https://github.com/containerd/cgroups/compare/bf292b2...0dbf7f0)
- github.com/coredns/corefile-migration: [v1.0.8 → v1.0.10](https://github.com/coredns/corefile-migration/compare/v1.0.8...v1.0.10)
- github.com/evanphx/json-patch: [e83c0a1 → v4.9.0+incompatible](https://github.com/evanphx/json-patch/compare/e83c0a1...v4.9.0)
- github.com/google/cadvisor: [8450c56 → v0.37.0](https://github.com/google/cadvisor/compare/8450c56...v0.37.0)
- github.com/json-iterator/go: [v1.1.9 → v1.1.10](https://github.com/json-iterator/go/compare/v1.1.9...v1.1.10)
- github.com/opencontainers/go-digest: [v1.0.0-rc1 → v1.0.0](https://github.com/opencontainers/go-digest/compare/v1.0.0-rc1...v1.0.0)
- github.com/opencontainers/runc: [1b94395 → 819fcc6](https://github.com/opencontainers/runc/compare/1b94395...819fcc6)
- github.com/prometheus/client_golang: [v1.6.0 → v1.7.1](https://github.com/prometheus/client_golang/compare/v1.6.0...v1.7.1)
- github.com/prometheus/common: [v0.9.1 → v0.10.0](https://github.com/prometheus/common/compare/v0.9.1...v0.10.0)
- github.com/prometheus/procfs: [v0.0.11 → v0.1.3](https://github.com/prometheus/procfs/compare/v0.0.11...v0.1.3)
- github.com/rubiojr/go-vhd: [0bfd3b3 → 02e2102](https://github.com/rubiojr/go-vhd/compare/0bfd3b3...02e2102)
- github.com/storageos/go-api: [343b3ef → v2.2.0+incompatible](https://github.com/storageos/go-api/compare/343b3ef...v2.2.0)
- github.com/urfave/cli: [v1.22.1 → v1.22.2](https://github.com/urfave/cli/compare/v1.22.1...v1.22.2)
- go.etcd.io/etcd: 54ba958 → dd1b699
- golang.org/x/crypto: bac4c82 → 75b2880
- golang.org/x/mod: v0.1.0 → v0.3.0
- golang.org/x/net: d3edc99 → ab34263
- golang.org/x/tools: c00d67e → c1934b7
- k8s.io/kube-openapi: 656914f → 6aeccd4
- k8s.io/system-validators: v1.1.2 → v1.2.0
- k8s.io/utils: 6e3d28b → d5654de

### Removed
- github.com/godbus/dbus: [ade71ed](https://github.com/godbus/dbus/tree/ade71ed)
- github.com/xlab/handysort: [fb3537e](https://github.com/xlab/handysort/tree/fb3537e)
- sigs.k8s.io/structured-merge-diff/v3: v3.0.0
- vbom.ml/util: db5cfe1