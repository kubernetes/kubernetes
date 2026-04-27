<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.36.0](#v1360)
  - [Downloads for v1.36.0](#downloads-for-v1360)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.35.0](#changelog-since-v1350)
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
- [v1.36.0-rc.1](#v1360-rc1)
  - [Downloads for v1.36.0-rc.1](#downloads-for-v1360-rc1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.36.0-rc.0](#changelog-since-v1360-rc0)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.36.0-rc.0](#v1360-rc0)
  - [Downloads for v1.36.0-rc.0](#downloads-for-v1360-rc0)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.36.0-beta.0](#changelog-since-v1360-beta0)
  - [Changes by Kind](#changes-by-kind-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.36.0-beta.0](#v1360-beta0)
  - [Downloads for v1.36.0-beta.0](#downloads-for-v1360-beta0)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.36.0-alpha.2](#changelog-since-v1360-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-2)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Documentation](#documentation-1)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.36.0-alpha.2](#v1360-alpha2)
  - [Downloads for v1.36.0-alpha.2](#downloads-for-v1360-alpha2)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
    - [Container Images](#container-images-4)
  - [Changelog since v1.36.0-alpha.1](#changelog-since-v1360-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-3)
    - [Dependency](#dependency-1)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-3)
    - [Feature](#feature-3)
    - [Failing Test](#failing-test-2)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)
- [v1.36.0-alpha.1](#v1360-alpha1)
  - [Downloads for v1.36.0-alpha.1](#downloads-for-v1360-alpha1)
    - [Source Code](#source-code-5)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
    - [Container Images](#container-images-5)
  - [Changelog since v1.35.0](#changelog-since-v1350-1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-3)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-3)
  - [Changes by Kind](#changes-by-kind-4)
    - [Dependency](#dependency-2)
    - [API Change](#api-change-4)
    - [Feature](#feature-4)
    - [Failing Test](#failing-test-3)
    - [Bug or Regression](#bug-or-regression-4)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-5)
    - [Added](#added-5)
    - [Changed](#changed-5)
    - [Removed](#removed-5)

<!-- END MUNGE: GENERATED_TOC -->

# v1.36.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.36.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes.tar.gz) | `3c9b9225c75080950fdb53fdeb326606133eeb5efbc8ecdd7514c290f1aaf8fa247a6f6f5b34beb87658a0ba5533c1f3cc7a8c680fc30785775ad73702374834`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-src.tar.gz) | `0b2c28c5b9f58c3ac6286e892f0bfeb1dbf8bcb9a76dc6128c080ab4f39d861af9b0ae50bf4a819d6ec376b4e8d2a55122d51d5fb26a69dc8af29bcd2406fb48`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-darwin-amd64.tar.gz) | `4be41ab0e38d809fad076b2c811124d93a6b48d696843f5bd85e5c77e27a112ee526f95de82c57d9f59060b27a7d2e036d6989246510f9772862b9f2ca87c5ed`
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-darwin-arm64.tar.gz) | `57b90606b066b6073362ee06ff8b00198abe2590ae89216f001a7d3bcd339dc8ede6616ad498eeaa65328a39834bc71e8c18944df0e90fc89139ac8d7290cfe9`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-linux-386.tar.gz) | `d1964452b0276b83933c1642743bb0058212bb88b41a5601e446ea49bb06fa1bb682d2cba4afd72896faf860a61e68494cb159db51c4c2ec5928de533f6cc9b9`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-linux-amd64.tar.gz) | `cac4ee270f7a5ca8e96f2b86f1b822bdc66168253b253f4838caf5bd16b8e314ae307c7ba718f32e9543d502f5d0c703bd3358449718c6956436969e125011cf`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-linux-arm.tar.gz) | `76117398c77401cb62303f765c6e42f93bff42d3f04d6501b282e14013fdce1ed57c743ed049995575c1326817c57db57523fdcca3ea3a7b7d58fe586d491bf1`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-linux-arm64.tar.gz) | `d669cc342059d88cf93db37d2bd41b444e352a9af64cc14767d77321e8e9bcdbdce886d605bf09b5dad09500a9d7b10023e3f07539915df4f175b56e0de8f5bb`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-linux-ppc64le.tar.gz) | `4245f7ec5bbc53b4b375c855110295d3e9640833e916ce83cdc9f610047442a705c17a641d8590c250d3c511771f478beab0f19d0258211d9fd1f97f7f00fcf3`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-linux-s390x.tar.gz) | `538b0b193767272ada79b832ec994d2172a7b88933c62711a117f6b0476902d7f4c771400e7468020dca8cb968d5ee220627a5249db2f16efb27bfa29e0570fc`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-windows-386.tar.gz) | `b97a5b7bcc96b42648fe9ea639742d05dc699a6394d3da246bfc72c810b650cb440f96319f0064e5479cd885aced8310640d5c8fcf6256fe5e88c0de93d27e9d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-windows-amd64.tar.gz) | `d1d3ca9de4c5917538b0865aa28a1fa9b2c7cad46921ab85f661025f5ddc277755cda46d441359d2d5717b908d9bc3fd7fe2aae22d95deeb172b1fdd49b0c9b3`
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-client-windows-arm64.tar.gz) | `4cbe90820c58892bf4327634dcfdca64c7db36cf09e2beae2f417248abe05fa01f8ae7a2edec08fa2ea28a368bc6e3bba7431be30a138bfaf78ba4e433bfa463`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-server-linux-amd64.tar.gz) | `1c64da92575451c2c7ff97c79b772e603995f8e76da1371a6b0746aaa27b65dce81c4d734cbb50f40e71486a8e08df36f14e12974f97d4d29e41f23a172a6a25`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-server-linux-arm64.tar.gz) | `7f95e451baedd9368a2fa637afa84c9ed1b958736540ca27a379e5e292e1a10c5d9a29539f833047574bf5d3b6f907f32dee5735acecc11aa4686f078108dcf8`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-server-linux-ppc64le.tar.gz) | `4d3f6b70ccb785d3264acf6b3cf0565560bb3c2ba9db85a06fa10c020ee248318ee009f9cc5b6d7171bda7ec2f96cf0f9a1a57ba857a4fa5331022c9122692d9`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-server-linux-s390x.tar.gz) | `7c12c4c89522c449fcd96837c27659edef53245b7a39e4802a7c00c6497624eff15b2d53ec7a70328176681967fa7e22bbf5c4b0f3fc32996daaf696ca54bc25`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-node-linux-amd64.tar.gz) | `8bba5bd0cb77997ad965739caff3c8839fd16f284cf28a0e46c44538ce0b6fd83c5f3c608dd86c4831c3de9306bbfd02ce26bbe9a426754b6253efa1d5cc030f`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-node-linux-arm64.tar.gz) | `10a4c3660e6e19fae998b04345dbe42711e61745bf374405c33e8604335b12ff55ce7b68f8a0d67d1f7ce4519f706fbd8d80781b7128369fe047c38915e8d189`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-node-linux-ppc64le.tar.gz) | `9eb7a4df4b518df4b846bd12ec25b50908b610b823f4d781094e0667e47d607209acbace2747955a373366a68b77f32170e2de71f0b2ee8a87723bac30c41d95`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-node-linux-s390x.tar.gz) | `492a71b292953ee5ccb603bf132a99fdbf6ae6fa8ae6c02139887b77ae6e1d7d77e65429bf37bb46eee33bf32cee3fe8ead03f74cd7b0f1f5a7c7126ba839800`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0/kubernetes-node-windows-amd64.tar.gz) | `a4f2bdb613da646877aacae6a2c39cbb18d74f164973fa033b5f042f1ed8a4de0285e0d99f303d8fdd89cecaa341c6e29361d7677a6bde5bbfc956dd156ac55c`

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.
name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0

## Urgent Upgrade Notes 

### (No, really, you MUST read this before you upgrade)

- ACTION REQUIRED:
  kube-controller-manager: Renamed metric `volume_operation_total_errors` to `volume_operation_errors_total`. If you are using custom monitoring dashboards or alerting rules based on the `volume_operation_total_errors` metric, update them to use the new `volume_operation_errors_total` metric. ([#136399](https://github.com/kubernetes/kubernetes/pull/136399), [@tico88612](https://github.com/tico88612)) [SIG Apps, Instrumentation, Storage and Testing]
- Added support for running PreBind plugins in parallel in the scheduler framework to improve binding latency.
  ACTION REQUIRED: Plugins can opt-in to parallel execution by returning `AllowParallel: true` from the `PreBindPreFlight` method. PreBind plugin implementations need to be updated to return `PreBindPreFlightResult` from the `PreBindPreFlight` method; returning nil retains the existing sequential behavior. ([#135393](https://github.com/kubernetes/kubernetes/pull/135393), [@tosi3k](https://github.com/tosi3k)) [SIG Node, Scheduling, Storage and Testing]
 
## Changes by Kind

### Dependency

- Fixed a bug where pod lifecycle hooks could run for their full duration when pods are terminated. ([#136598](https://github.com/kubernetes/kubernetes/pull/136598), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]
- Updated etcd client library to `v3.6.8`. ([#137225](https://github.com/kubernetes/kubernetes/pull/137225), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Auth, Cloud Provider, Cluster Lifecycle, Etcd, Node, Scheduling and Testing]

### Deprecation

- Added warnings and deprecation for Service `.spec.externalIPs`. ([#137293](https://github.com/kubernetes/kubernetes/pull/137293), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Apps, Network and Windows]
- Direct access to the `Raw` field of `metav1.FieldsV1` is deprecated. Code that constructs or reads `FieldsV1` should migrate to the new `NewFieldsV1(string)`, `GetRawBytes()`, `GetRawString()`, and `SetRawBytes()` accessor methods. ([#137304](https://github.com/kubernetes/kubernetes/pull/137304), [@aaron-prindle](https://github.com/aaron-prindle)) [SIG API Machinery, Apps and Testing]
- Disabled `git-repo` volume plugin by default, with no option to turn it back on. ([#136400](https://github.com/kubernetes/kubernetes/pull/136400), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Storage]
- Renamed `AllowlistEntry.Name` to `AllowlistEntry.Command` in the credential plugin allowlist. ([#137272](https://github.com/kubernetes/kubernetes/pull/137272), [@pmengelbert](https://github.com/pmengelbert)) [SIG API Machinery, Auth, CLI and Testing]

### API Change

- ACTION REQUIRED: DRA (Dynamic Resource Allocation) drivers and controllers now require granular RBAC permissions to update ResourceClaim statuses when the `DRAResourceClaimGranularStatusAuthorization` feature gate is enabled (beta in `v1.36`). Schedulers and controllers must be granted `update`/`patch` on `resourceclaims/binding`. DRA drivers must be granted `associated-node:update` or `arbitrary-node:update` (or patch equivalents) on `resourceclaims/driver`, restricted by their specific `resourceNames`. ([#134947](https://github.com/kubernetes/kubernetes/pull/134947), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Auth, Instrumentation, Node, Scheduling and Testing]
- ACTION REQUIRED: Removed the integrated support for flex-volumes in kubeadm. Users were advised to migrate away from flex-volumes as recommended by SIG Storage since `v1.22`. If `kubeadm` users wish to continue using the feature, they need a custom image for the KCM that is not based on distroless, pass the KCM flag `--flex-volume-plugin-dir`, and mount the directory `/usr/libexec/kubernetes/kubelet-plugins/volume/exec` in the KCM static pod using `kubeadm`'s `extraVolumes` mechanism before upgrading to `v1.36`. Previously, `kubeadm` automatically did the mounting if the user passed the flag. ([#136423](https://github.com/kubernetes/kubernetes/pull/136423), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- ACTION REQUIRED: Renamed metric `etcd_bookmark_counts` to `etcd_bookmark_total`. If you are using custom monitoring dashboards or alerting rules based on the `etcd_bookmark_counts` metric, update them to use the new `etcd_bookmark_total` metric. ([#136483](https://github.com/kubernetes/kubernetes/pull/136483), [@petern48](https://github.com/petern48)) [SIG API Machinery, Etcd, Instrumentation and Testing]
- Added SchedulingConstraints to express topology-aware scheduling (TAS) constraints for PodGroup scheduling behind the `TopologyAwareWorkloadScheduling` feature gate. Added the TopologyPlacement plugin implementing the PlacementGenerate extension point to take constraints into consideration during PodGroup scheduling. ([#137271](https://github.com/kubernetes/kubernetes/pull/137271), [@brejman](https://github.com/brejman)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Added `DisruptionMode`, `PriorityClassName`, and `Priority` fields to the Workload and PodGroup APIs to support workload-aware preemption when the `WorkloadAwarePreemption` feature gate is enabled. ([#136589](https://github.com/kubernetes/kubernetes/pull/136589), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Added `ImageVolumeWithDigest` which includes the digest of image volumes in the container status. ([#132807](https://github.com/kubernetes/kubernetes/pull/132807), [@iholder101](https://github.com/iholder101)) [SIG API Machinery, Apps, Node and Testing]
- Added `MemoryReservationPolicy` cgroup v2 MemoryQoS support to KubeletConfiguration for `memory.min` protection. ([#137584](https://github.com/kubernetes/kubernetes/pull/137584), [@QiWang19](https://github.com/QiWang19)) [SIG Node and Storage]
- Added `spec.stubPKCS10Request` to the Pod Certificates beta API to improve compatibility with existing certificate authority implementations that expect a PKCS#10 certificate signing request. `spec.pkixPublicKey` and `spec.proofOfPossession` were deprecated in favor of this field. ([#136729](https://github.com/kubernetes/kubernetes/pull/136729), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Auth, Node and Testing]
- Added a deletion protection mechanism for PodGroup objects. ([#137641](https://github.com/kubernetes/kubernetes/pull/137641), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Auth, Scheduling and Storage]
- Added alpha support (behind the `PersistentVolumeClaimUnusedSinceTime` feature gate) for tracking PersistentVolumeClaim unused status via a new `Unused` condition on PersistentVolumeClaimStatus. When enabled, the PVC protection controller sets `Unused=True` with a `lastTransitionTime` when no non-terminal Pods reference the PersistentVolumeClaim. ([#137862](https://github.com/kubernetes/kubernetes/pull/137862), [@gnufied](https://github.com/gnufied)) [SIG Apps, Auth, Storage and Testing]
- Added alpha support for manifest-based admission control configuration (KEP-5793). When the `ManifestBasedAdmissionControlConfig` feature gate is enabled, admission webhooks and CEL-based policies can be loaded from static manifest files on disk via the `staticManifestsDir` field in `AdmissionConfiguration`. These policies are active from API server startup, survive `etcd` unavailability, and can protect API-based admission resources from modification. ([#137346](https://github.com/kubernetes/kubernetes/pull/137346), [@aramase](https://github.com/aramase)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage, Testing and Windows]
- Added an admission plugin that validates PodGroup resources reference an existing Workload and match the declared PodGroupTemplate spec. ([#137464](https://github.com/kubernetes/kubernetes/pull/137464), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Added list-type support for attributes in DRA (KEP-5491). The `DRAListTypeAttributes` feature gate (disabled by default) activates the following enhancements:
  - DRA drivers can use list-type fields (`bools`/`ints`/`strings`/`versions`) for device attributes in ResourceSlice. The number of attribute values, including scalars and lists, per single device is limited to 48.
  - The `matchAttribute`/`distinctAttribute` constraints in ResourceClaim now work on both scalar and list attributes. The `matchAttribute` constraint matches when the intersection of all list values among candidate devices is non-empty. The `distinctAttribute` constraint (behind the `ConsumableCapacity` feature gate) matches when all list values among candidate devices are pairwise disjoint. Scalar values are implicitly treated as a singleton set.
  - Added a new CEL function `.includes` that works on both scalar and list attributes to test inclusion (e.g., `device.attributes["dra.example.com"].model.includes("model-a")`), supporting migration when a DRA driver changes an attribute value type from scalar to list or vice versa. ([#137190](https://github.com/kubernetes/kubernetes/pull/137190), [@everpeace](https://github.com/everpeace)) [SIG API Machinery, Node, Scheduling and Testing]
- Added new `concurrent-node-status-updates` flag that is split from the `concurrent-node-syncs` flag. ([#136716](https://github.com/kubernetes/kubernetes/pull/136716), [@yonizxz](https://github.com/yonizxz)) [SIG Cloud Provider]
- Added opt-in alpha support in the kubeletplugin framework for DRA drivers to publish DRA Device metadata in Pod CDI mounts. ([#137086](https://github.com/kubernetes/kubernetes/pull/137086), [@alaypatel07](https://github.com/alaypatel07)) [SIG Apps, Network, Node and Testing]
- Added opt-in scheduling behavior for CSI volumes. ([#137343](https://github.com/kubernetes/kubernetes/pull/137343), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Scheduling and Storage]
- Added placement-based PodGroup scheduling algorithm to the scheduler. Its use is guarded by the `TopologyAwareWorkloadScheduling` feature gate. ([#136944](https://github.com/kubernetes/kubernetes/pull/136944), [@brejman](https://github.com/brejman)) [SIG Scheduling and Testing]
- Added stability-based lifecycle for declarative validation (Alpha/Beta/Stable). Scheduling Workload `v1alpha1` now uses explicit declarative enforcement. ([#136793](https://github.com/kubernetes/kubernetes/pull/136793), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Scheduling]
- Added the PlacementGenerate extension point to the scheduler. It is used to generate placements for placement-based PodGroup scheduling. Its use is guarded by the `TopologyAwareWorkloadScheduling` feature gate. ([#137083](https://github.com/kubernetes/kubernetes/pull/137083), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Added the PlacementScore extension point to the scheduler for scoring placements in placement-based PodGroup scheduling, guarded by the `TopologyAwareWorkloadScheduling` feature gate. Deprecated `MinNodeScore` and `MaxNodeScore` in favor of `MinScore` and `MaxScore`. ([#137201](https://github.com/kubernetes/kubernetes/pull/137201), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Added the ResourcePoolStatusRequest API (`v1alpha1`) for querying DRA resource pool availability. External schedulers can discover available devices across pools before submitting workloads. Requires the `DRAResourcePoolStatus` feature gate (alpha). ([#137028](https://github.com/kubernetes/kubernetes/pull/137028), [@nmn3m](https://github.com/nmn3m)) [SIG API Machinery, Apps, Auth, Etcd, Instrumentation, Node, Scheduling, Storage and Testing]
- Added the `--concurrent-resourceclaim-syncs` flag to `kube-controller-manager` to configure `ResourceClaim` reconcile concurrency. ([#134701](https://github.com/kubernetes/kubernetes/pull/134701), [@anson627](https://github.com/anson627)) [SIG API Machinery, Apps, Node and Testing]
- Added the `--tls-curve-preferences` flag for configuring TLS key exchange mechanism. ([#137115](https://github.com/kubernetes/kubernetes/pull/137115), [@damdo](https://github.com/damdo)) [SIG API Machinery, Architecture, CLI, Cloud Provider, Node and Testing]
- Added the `PodGroupPodsCount` scheduler plugin to support workload-aware scheduling by prioritizing placements with higher Pod counts within a group. ([#137488](https://github.com/kubernetes/kubernetes/pull/137488), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling and Testing]
- Added the `tlsServerName` field to `EgressSelectorConfiguration` `TLSConfig` to allow overriding the server name used for TLS certificate verification. ([#136640](https://github.com/kubernetes/kubernetes/pull/136640), [@kennangaibel](https://github.com/kennangaibel)) [SIG API Machinery, Apps, Auth, Storage and Testing]
- Added the alpha `DRANativeResources` feature, which includes a new `ResourceSlice.Spec.Devices[*].NativeResourceMappings` field for DRA drivers to declare how device resources map to native Kubernetes resources (e.g., cpu, memory), changes in the DynamicResources plugin and the scheduler framework to correctly account for native resources requested through resource claims, and `kubelet` admission handler validation for native resource DRA requests along with standard requests in the Pod spec. ([#136725](https://github.com/kubernetes/kubernetes/pull/136725), [@pravk03](https://github.com/pravk03)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Added topology-aware scheduling (TAS) logic to the PodGroup scheduling cycle behind the `TopologyAwareWorkloadScheduling` feature gate, supporting scheduling of PodGroups on nodes with matching topology domains. ([#137489](https://github.com/kubernetes/kubernetes/pull/137489), [@brejman](https://github.com/brejman)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Added validation to prevent negative duration values for `imageMinimumGCAge`. ([#135997](https://github.com/kubernetes/kubernetes/pull/135997), [@ngopalak-redhat](https://github.com/ngopalak-redhat)) [SIG API Machinery and Node]
- Changed deprecated `sets.String` with `sets.Set[string]` in apiserver admission subsystem. This is a **breaking change** for consumers of the `NewLifecycle` function. ([#134044](https://github.com/kubernetes/kubernetes/pull/134044), [@mcallzbl](https://github.com/mcallzbl)) [SIG API Machinery and Auth]
- Clarified documentation and comments to indicate that the `cpuCFSQuotaPeriod` kubelet config field requires the `CustomCPUCFSQuotaPeriod` feature gate when using non-default values. No functional changes introduced. ([#133845](https://github.com/kubernetes/kubernetes/pull/133845), [@rbiamru](https://github.com/rbiamru)) [SIG Node and Release]
- Corrected OpenAPI schema union validation for the `PodGroupPolicy` struct in `scheduling.k8s.io/v1alpha1`. ([#136424](https://github.com/kubernetes/kubernetes/pull/136424), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery and Scheduling]
- DRA `DeviceTaintRules`: the `TimeAdded` field of the taint is now automatically updated when changing the effect. ([#137167](https://github.com/kubernetes/kubernetes/pull/137167), [@pohly](https://github.com/pohly)) [SIG API Machinery, Node and Testing]
- DRA: Added a `spec.resourceClaims` field to PodGroup resources for referencing ResourceClaims and ResourceClaimTemplates. Claims made by a PodGroup are reserved for the entire PodGroup instead of individual Pods, supporting more than 256 Pods sharing a single ResourceClaim. ResourceClaimTemplates referenced by a PodGroup's claim replicate into a ResourceClaim specific to that PodGroup, shared by all of the group's Pods. ([#136989](https://github.com/kubernetes/kubernetes/pull/136989), [@nojnhuh](https://github.com/nojnhuh)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- DRA: Graduated Device Binding Conditions (KEP #5007) to beta, enabled by default in `v1.36`. ([#137795](https://github.com/kubernetes/kubernetes/pull/137795), [@ttsuuubasa](https://github.com/ttsuuubasa)) [SIG API Machinery, Node, Scheduling and Testing]
- DRA: Graduated device taints and tolerations (KEP #5055) to beta. Support for DeviceTaints in ResourceSlices is on by default. Support for `DeviceTaintRules` depends on enabling `resource.k8s.io/v1beta2` and the `DeviceTaintRules` feature gate. ([#137170](https://github.com/kubernetes/kubernetes/pull/137170), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Cluster Lifecycle, Etcd, Node, Scheduling and Testing]
- Extended `NodeResourcesFit` to implement the PlacementScore extension point. The usage of the PlacementScore extension point is guarded by the `TopologyAwareWorkloadScheduling` feature gate. ([#136652](https://github.com/kubernetes/kubernetes/pull/136652), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fixed `fake.NewClientset()` to work properly with correct schema. ([#131068](https://github.com/kubernetes/kubernetes/pull/131068), [@soltysh](https://github.com/soltysh)) [SIG API Machinery]
- Fixed a few log calls that did not properly format their parameters. ([#137108](https://github.com/kubernetes/kubernetes/pull/137108), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Cluster Lifecycle, Network, Node, Scheduling and Testing]
- Fixed a potential nil pointer dereference in the scheduler's `NodeResourcesFitArgs` validation when using `RequestedToCapacityRatio` scoring strategy. ([#132120](https://github.com/kubernetes/kubernetes/pull/132120), [@flpanbin](https://github.com/flpanbin)) [SIG Scheduling]
- Fixed an issue in `kube-apiserver`, allowing it to recover from an established connection to an incorrect server that never returns the expected response during APIService availability checks. ([#137157](https://github.com/kubernetes/kubernetes/pull/137157), [@bsalamat](https://github.com/bsalamat)) [SIG API Machinery]
- For Pod resizes requested on nodes where the resize request exceeds the node's allocatable capacity or the node is running an OS that does not support resize, the request fails in admission rather than being marked as Infeasible in the Pod status later. ([#136043](https://github.com/kubernetes/kubernetes/pull/136043), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Node, Release, Scheduling, Storage and Testing]
- Generated `fake.NewClientset` which replaces the deprecated `NewSimpleClientset` for `kube-aggregator` and `sample-apiserver`. ([#136537](https://github.com/kubernetes/kubernetes/pull/136537), [@soltysh](https://github.com/soltysh)) [SIG API Machinery]
- Graduated metric `apiserver_storage_events_received_total` to beta. ([#136314](https://github.com/kubernetes/kubernetes/pull/136314), [@petern48](https://github.com/petern48)) [SIG API Machinery, Etcd, Instrumentation and Testing]
- Graduated the `ImageVolume` feature to stable. ([#136711](https://github.com/kubernetes/kubernetes/pull/136711), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps, Architecture, Node and Testing]
- Graduated the `InPlacePodLevelResourcesVerticalScaling` feature gate to beta, enabled by default. Pod-level CPU and memory resources can be resized in place for Pods with pod-level resources configured. ([#137684](https://github.com/kubernetes/kubernetes/pull/137684), [@ndixita](https://github.com/ndixita)) [SIG API Machinery, Apps, Autoscaling, Node, Release, Scheduling and Testing]
- Graduated the `UserNamespacesSupport` feature gate to GA. ([#136792](https://github.com/kubernetes/kubernetes/pull/136792), [@rata](https://github.com/rata)) [SIG API Machinery, Apps, CLI, Node, Storage and Testing]
- Graduated the `config.k8s.io/flagz` API to `v1beta1`. ([#137174](https://github.com/kubernetes/kubernetes/pull/137174), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation, Node, Scheduling and Testing]
- Graduated the `config.k8s.io/statusz` API to `v1beta1`. ([#137173](https://github.com/kubernetes/kubernetes/pull/137173), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation, Scheduling and Testing]
- HPA: Improved scaling to and from zero when the `HPAScaleToZero` feature gate is enabled. ([#135118](https://github.com/kubernetes/kubernetes/pull/135118), [@johanneswuerbach](https://github.com/johanneswuerbach)) [SIG Apps, Autoscaling and Testing]
- Integrated Workload and PodGroup APIs with the Job controllers to support gang-scheduling. ([#137032](https://github.com/kubernetes/kubernetes/pull/137032), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Instrumentation, Node, Scheduling and Testing]
- Introduced `scheduling.k8s.io/v1alpha2` Workload and PodGroup API to express workload-level scheduling requirements and let `kube-scheduler` act on those. Removed `scheduling.k8s.io/v1alpha1` Workload API. ([#136976](https://github.com/kubernetes/kubernetes/pull/136976), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling, Storage and Testing]
- Kube-apiserver: The `--audit-policy-file` config file now supports specifying `group: "*"` in resource rules to match all API groups. ([#135262](https://github.com/kubernetes/kubernetes/pull/135262), [@cmuuss](https://github.com/cmuuss)) [SIG API Machinery, Auth and Testing]
- Kube-controller-manager: Added ALPHA gauge metric `informer_queued_items` for informer queue length, published as `informer_queued_items{name=kube-controller-manager,group=<group>,resource=<resource>,version=<version>} <count>`. ([#135782](https://github.com/kubernetes/kubernetes/pull/135782), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Kubelet: Added tiered cgroup v2 memory protection for MemoryQoS: `memory.min` for Guaranteed pods and `memory.low` for Burstable pods, with node-level metrics and rollback reconciliation (KEP-2570). ([#137719](https://github.com/kubernetes/kubernetes/pull/137719), [@sohankunkerkar](https://github.com/sohankunkerkar)) [SIG Node, Storage and Testing]
- Locked the `VolumeAttributesClass` feature gate to `true` and updated the preferred storage version to `storage.k8s.io/v1`. ([#134556](https://github.com/kubernetes/kubernetes/pull/134556), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Etcd, Network, Node, Scheduling, Storage and Testing]
- Marked the `endpoints` field as optional in the OpenAPI spec for `discovery.k8s.io/v1` EndpointSlice. This matches server behavior and resolves validation issues. ([#136111](https://github.com/kubernetes/kubernetes/pull/136111), [@aojea](https://github.com/aojea)) [SIG Network]
- Promoted `DRAPrioritizedList` to GA. ([#136924](https://github.com/kubernetes/kubernetes/pull/136924), [@troychiu](https://github.com/troychiu)) [SIG Apps, Architecture, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Network, Node, Release, Scheduling, Storage and Testing]
- Promoted `NodeDeclaredFeatures` to beta. ([#136042](https://github.com/kubernetes/kubernetes/pull/136042), [@pravk03](https://github.com/pravk03)) [SIG API Machinery, Apps, Cluster Lifecycle, Instrumentation, Node, Scheduling, Storage and Testing]
- Promoted `SnapshotMetadataService` to `v1beta1`. Removed support for the `v1alpha1` version. ([#137564](https://github.com/kubernetes/kubernetes/pull/137564), [@iPraveenParihar](https://github.com/iPraveenParihar)) [SIG Storage and Testing]
- Promoted mutable CSI node allocatable count to GA. The `MutableCSINodeAllocatableCount` feature gate is locked to enabled. ([#136230](https://github.com/kubernetes/kubernetes/pull/136230), [@torredil](https://github.com/torredil)) [SIG API Machinery and Storage]
- Promoted several EndpointSlice metrics from alpha to beta stability. ([#136368](https://github.com/kubernetes/kubernetes/pull/136368), [@bhope](https://github.com/bhope)) [SIG Instrumentation and Network]
- Promoted several component-base metrics (`kubernetes_build_info`, `rest_client_requests_total`, `rest_client_request_duration_seconds`, `running_managed_controllers`) from Alpha to Beta stability, providing stronger API and label stability guarantees for consumers. ([#136154](https://github.com/kubernetes/kubernetes/pull/136154), [@bhope](https://github.com/bhope)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scalability, Scheduling, Storage and Testing]
- Promoted several scheduler metrics (`scheduler_goroutines`, `scheduler_permit_wait_duration_seconds`, `scheduler_plugin_evaluation_total`, `scheduler_plugin_execution_duration_seconds`, `scheduler_scheduling_algorithm_duration_seconds`, `scheduler_unschedulable_pods`) from alpha to beta stability, providing stronger API and label stability guarantees for metric consumers. ([#136155](https://github.com/kubernetes/kubernetes/pull/136155), [@bhope](https://github.com/bhope)) [SIG Instrumentation and Scheduling]
- Promoted the DRA extended resource feature to beta in `v1.36`. ([#135048](https://github.com/kubernetes/kubernetes/pull/135048), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Architecture, Auth, Network, Node, Scheduling and Testing]
- Promoted the `ConstrainedImpersonation` feature to beta, enabled by default. ([#137609](https://github.com/kubernetes/kubernetes/pull/137609), [@enj](https://github.com/enj)) [SIG API Machinery and Testing]
- Promoted the `DRAAdminAccess` feature gate to GA. ([#137373](https://github.com/kubernetes/kubernetes/pull/137373), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth, Node, Scheduling and Testing]
- Promoted the `MutatingAdmissionPolicy` to GA (v1) in Kubernetes `v1.36`. The feature is now enabled by default. ([#136039](https://github.com/kubernetes/kubernetes/pull/136039), [@lalitc375](https://github.com/lalitc375)) [SIG API Machinery, Architecture, Etcd and Testing]
- Promoted the `NodeLogQuery` feature gate to GA. ([#137544](https://github.com/kubernetes/kubernetes/pull/137544), [@jrvaldes](https://github.com/jrvaldes)) [SIG Node and Windows]
- Promoted the `ProcMountType` feature to GA. ([#137454](https://github.com/kubernetes/kubernetes/pull/137454), [@haircommander](https://github.com/haircommander)) [SIG API Machinery, Apps, Auth, CLI, Node, Storage and Testing]
- Promoted the `watch_list_duration_seconds` metric from ALPHA to BETA. ([#136086](https://github.com/kubernetes/kubernetes/pull/136086), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation, Node and Testing]
- Promoted two Job controller metrics from alpha to beta stability, providing stronger API and label stability guarantees for metric consumers. ([#136367](https://github.com/kubernetes/kubernetes/pull/136367), [@bhope](https://github.com/bhope)) [SIG Apps and Instrumentation]
- Promoted workqueue metrics from ALPHA to BETA. ([#135522](https://github.com/kubernetes/kubernetes/pull/135522), [@petern48](https://github.com/petern48)) [SIG Architecture, Instrumentation and Testing]
- Removed CustomResourceDefinition stored versions from status upon StorageVersionMigrator migration. ([#135297](https://github.com/kubernetes/kubernetes/pull/135297), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth and Testing]
- Removed the in-tree Portworx volume plugin, completing the migration to CSI.
  Removed the GA `CSIMigrationPortworx` feature gate (locked since `v1.33`) and alpha `InTreePluginPortworxUnregister` feature gate, with all operations now redirected to CSI. ([#135322](https://github.com/kubernetes/kubernetes/pull/135322), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Auth, Node, Scalability, Scheduling, Storage and Testing]
- Removed the temporary build-tagged `ProtoMessage()` marker method implementations from Kubernetes REST API types in `k8s.io/api`, which had incorrectly identified them as standard `v1` proto messages. Protobuf serialization of Kubernetes API types should use [k8s.io/apimachinery/pkg/runtime/serializer/protobuf](https://pkg.go.dev/k8s.io/apimachinery/pkg/runtime/serializer/protobuf). ([#137084](https://github.com/kubernetes/kubernetes/pull/137084), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, Node, Scheduling and Storage]
- Slow requests that use impersonation can be tracked via the `apiserver.latency.k8s.io/impersonation` audit event annotation when the `ConstrainedImpersonation` feature is enabled. ([#137523](https://github.com/kubernetes/kubernetes/pull/137523), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- The `DRAConsumableCapacity` feature gate is enabled by default. ([#136611](https://github.com/kubernetes/kubernetes/pull/136611), [@sunya-ch](https://github.com/sunya-ch)) [SIG API Machinery, Cluster Lifecycle, Node, Scheduling and Testing]
- The `StrictIPCIDRValidation` feature gate in `kube-apiserver` is enabled by default, meaning that API fields no longer allow IP or CIDR values with extraneous leading "0"s (e.g., `010.000.000.005` rather than `10.0.0.5`) or CIDR subnet/mask values with ambiguous semantics (e.g., `192.168.0.5/24` rather than `192.168.0.0/24` or `192.168.0.5/32`). ([#137053](https://github.com/kubernetes/kubernetes/pull/137053), [@danwinship](https://github.com/danwinship)) [SIG Network and Testing]
- The `kube-scheduler` now updates PodGroup status with a `PodGroupScheduled` condition reflecting whether the group was successfully scheduled or is unschedulable. ([#137611](https://github.com/kubernetes/kubernetes/pull/137611), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Scheduling and Testing]
- Updated API comments to reflect the stable state of Dynamic Resource Allocation (DRA). ([#136441](https://github.com/kubernetes/kubernetes/pull/136441), [@kannon92](https://github.com/kannon92)) [SIG API Machinery]
- Updated API server internal API group to improve openapi schema correctness for fields being optional or required. ([#134675](https://github.com/kubernetes/kubernetes/pull/134675), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery, Apps, Auth, Node and Storage]
- Updated the `/configz` endpoint of `kubelet`, `kube-scheduler`, cloud controller manager, and `kube-proxy` to serialize the `APIVersion` and `Kind` fields and use public types instead of internal. ([#136044](https://github.com/kubernetes/kubernetes/pull/136044), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Network, Node, Scheduling and Testing]

### Feature

- Added ALPHA counter metric `scheduler_pod_scheduled_after_flush_total` to track pods successfully scheduled after timeout flush from the `unschedulablePods` queue. ([#135126](https://github.com/kubernetes/kubernetes/pull/135126), [@mrvarmazyar](https://github.com/mrvarmazyar)) [SIG Scheduling]
- Added `ARCH` column in the `kubectl get node -o wide` output. ([#132402](https://github.com/kubernetes/kubernetes/pull/132402), [@astraw99](https://github.com/astraw99)) [SIG CLI]
- Added `apiserver_peer_proxy_errors_total` and `apiserver_peer_discovery_sync_errors_total` alpha metrics to apiserver to track errors encountered in peer proxying and peer discovery. ([#137065](https://github.com/kubernetes/kubernetes/pull/137065), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Added `kubectl explain -r` flag as a shorthand for `--recursive`. ([#135283](https://github.com/kubernetes/kubernetes/pull/135283), [@laervn](https://github.com/laervn)) [SIG CLI]
- Added `kubelet_metrics_provider` metric to help users identify where kubelet's metrics are coming from. ([#136952](https://github.com/kubernetes/kubernetes/pull/136952), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Node]
- Added a `PodGroup` scheduling cycle to `kube-scheduler`'s main scheduling loop, enabling all pods within a `PodGroup` to be scheduled within a single cycle. ([#136618](https://github.com/kubernetes/kubernetes/pull/136618), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- Added a `show-secret` flag to the diff command to explicitly allow secret values to be displayed during the diff operation. ([#137019](https://github.com/kubernetes/kubernetes/pull/137019), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Added a new gRPC service to the `kubelet` that provides information about Pods running on the node. ([#134627](https://github.com/kubernetes/kubernetes/pull/134627), [@briansonnenberg](https://github.com/briansonnenberg)) [SIG Node and Testing]
- Added a warning when `kubectl rollout undo` is used on resources managed with `kubectl apply` to prevent unexpected behavior from annotation mismatch. ([#137064](https://github.com/kubernetes/kubernetes/pull/137064), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Added alpha counter metric `route_controller_route_sync_total` to Cloud Controller Manager to track route syncs with cloud providers. This metric is in alpha stage. ([#136539](https://github.com/kubernetes/kubernetes/pull/136539), [@lukasmetzner](https://github.com/lukasmetzner)) [SIG API Machinery, Cloud Provider and Instrumentation]
- Added alpha metrics tracking the resource version the cache layer of an informer is at. ([#137419](https://github.com/kubernetes/kubernetes/pull/137419), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Added an alpha `informer_processing_latency_seconds` histogram metric to measure event handler execution time in RealFIFO. ([#137101](https://github.com/kubernetes/kubernetes/pull/137101), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Added metrics for constrained impersonation: `apiserver_impersonation_attempts_total`, `apiserver_impersonation_attempts_duration_seconds`, `apiserver_impersonation_authorization_attempts_total`, and `apiserver_impersonation_authorization_attempts_duration_seconds` (labels: mode, decision). ([#137374](https://github.com/kubernetes/kubernetes/pull/137374), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Added missing flags to `webhook serving` options for `k8s.io/cloud-provider`. ([#136816](https://github.com/kubernetes/kubernetes/pull/136816), [@damdo](https://github.com/damdo)) [SIG Cloud Provider]
- Added multiple conditions support to the `kubectl wait` command. ([#136855](https://github.com/kubernetes/kubernetes/pull/136855), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Added new RuntimeService streaming RPCs (`StreamPodSandboxes`, `StreamContainers`, `StreamContainerStats`, `StreamPodSandboxStats`, `StreamPodSandboxMetrics`) and new ImageService streaming RPC (`StreamImages`). ([#136987](https://github.com/kubernetes/kubernetes/pull/136987), [@bitoku](https://github.com/bitoku)) [SIG Cluster Lifecycle, Node and Testing]
- Added support for in-place Pod resize of running non-sidecar initContainers. ([#137352](https://github.com/kubernetes/kubernetes/pull/137352), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Apps, Autoscaling, Node, Scheduling, Storage and Testing]
- Added support for the CRI (and NRI) to block Pod-level resizes. ([#137555](https://github.com/kubernetes/kubernetes/pull/137555), [@natasha41575](https://github.com/natasha41575)) [SIG Node]
- Added support for unknown (non-pod) references in `ResourceClaim` `status.reservedFor`. The controller now gracefully skips these entries instead of halting sync, ensuring stale pod references can still be cleaned up. ([#136450](https://github.com/kubernetes/kubernetes/pull/136450), [@MohammedSaalif](https://github.com/MohammedSaalif)) [SIG Apps and Node]
- Added the `ControllerManagerReleaseLeaderElectionLockOnCancel` feature gate to gate leader election lock release on exit for `kube-controller-manager`. ([#136279](https://github.com/kubernetes/kubernetes/pull/136279), [@tchap](https://github.com/tchap)) [SIG API Machinery and Cloud Provider]
- Added the `ExtendWebSocketsToKubelet` feature gate (beta, default true in `v1.36`). When enabled, the API server proxies WebSocket `exec/attach/portforward` requests directly to the `kubelet` rather than translating or tunneling them at the API server. The `kubelet` handles WebSocket-to-SPDY stream translation (`exec/attach`) and WebSocket tunneling (portforward) using the same handlers previously used at the API server. The `kubelet` advertises support for this feature to the API server via the `NodeDeclaredFeatures` mechanism; the API server only proxies directly to a `kubelet` that has advertised support. Two new alpha metrics track routing decisions and WebSocket streaming volume: `apiserver_websocket_streaming_requests_total` (labels: subresource, proxy_type) and `kubelet_streaming_websocket_requests_total` (label: subresource). ([#136256](https://github.com/kubernetes/kubernetes/pull/136256), [@seans3](https://github.com/seans3)) [SIG API Machinery, Autoscaling, Node, Scheduling and Testing]
- Added the `UserNamespacesHostNetwork` runtime handler and integrated the `UserNamespacesHostNetworkSupport` feature gate with the `NodeDeclaredFeatures` feature gate. The `UserNamespacesHostNetworkSupport` feature gate only takes effect when the container runtime's `UserNamespacesHostNetwork` runtime handler returns true and the `NodeDeclaredFeatures` feature gate is enabled. ([#135828](https://github.com/kubernetes/kubernetes/pull/135828), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Autoscaling, Node, Scheduling and Testing]
- Added the `appProtocol` field to `kubectl describe service` output. ([#135744](https://github.com/kubernetes/kubernetes/pull/135744), [@ali-a-a](https://github.com/ali-a-a)) [SIG CLI]
- Added the `timezone` field to the `kubectl describe` CronJob output. ([#136663](https://github.com/kubernetes/kubernetes/pull/136663), [@kfess](https://github.com/kfess)) [SIG CLI]
- Added the ability for the StatefulSet controller to read its own Pod and PVC writes. ([#137254](https://github.com/kubernetes/kubernetes/pull/137254), [@michaelasp](https://github.com/michaelasp)) [SIG Apps]
- Added the ability for the `ReplicaSet` controller to read its own writes, preventing spurious reconciliation loops while the cache catches up to recent updates. ([#137212](https://github.com/kubernetes/kubernetes/pull/137212), [@michaelasp](https://github.com/michaelasp)) [SIG Apps]
- Added the metric `terminated_containers_total` to track the number of failed or succeeded containers, broken down by exit code. ([#137453](https://github.com/kubernetes/kubernetes/pull/137453), [@rawsocket](https://github.com/rawsocket)) [SIG Instrumentation, Node and Testing]
- Added tracing for WatchList requests. ([#137202](https://github.com/kubernetes/kubernetes/pull/137202), [@serathius](https://github.com/serathius)) [SIG API Machinery and Testing]
- Added two scheduler metrics for Device Binding Conditions, covering allocation attempts and PreBind duration with status and driver labels. ([#137284](https://github.com/kubernetes/kubernetes/pull/137284), [@ttsuuubasa](https://github.com/ttsuuubasa)) [SIG Node and Scheduling]
- Added write and read permissions for workloads to the `admin` cluster role, write permissions to the `edit` cluster role, and read permissions to the `view` cluster role. ([#135418](https://github.com/kubernetes/kubernetes/pull/135418), [@carlory](https://github.com/carlory)) [SIG Auth]
- Aligned the `scheduler_preemption_victims` metric definition between asynchronous and synchronous preemption modes. The metric now consistently reports the number of pods chosen as victims across both modes. ([#135955](https://github.com/kubernetes/kubernetes/pull/135955), [@utam0k](https://github.com/utam0k)) [SIG Scheduling]
- CRI API: Added the `image_id` field to the `PullImageResponse` message, serving as a unique identifier for the image on the node as returned by the container runtimes. ([#137217](https://github.com/kubernetes/kubernetes/pull/137217), [@stlaz](https://github.com/stlaz)) [SIG Node]
- Changed the default debug profile from `legacy` to `general`. The `legacy` profile is planned to be removed in `v1.39`. ([#135874](https://github.com/kubernetes/kubernetes/pull/135874), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI and Testing]
- Client-go: Default informer behavior now updates store state with all the objects in a list or relist before calling handler `OnDelete`, `OnAdd`, or `OnUpdate` methods for individual items which were deleted, added, or removed. This ensures that the store state which can be inspected by handlers corresponds to a set of objects that existed at a particular resource version on the server. This behavior is guarded by the `AtomicFIFO` feature gate, which is enabled by default in `v1.36` but can be disabled if needed to temporarily regain the previous behavior. ([#135462](https://github.com/kubernetes/kubernetes/pull/135462), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Client-go: Improved informer resync processing to reduce contention on store locks between incoming events and handler updates, which may result in observable timing differences of handler invocations. This behavior is guarded by the `AtomicFIFO` feature gate, which is enabled by default in `v1.36` but can be disabled if needed to temporarily regain the previous behavior. ([#136008](https://github.com/kubernetes/kubernetes/pull/136008), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Client-go: Informers can now enqueue new watch events while already-queued events are being processed. This avoids dropping watches during a burst of incoming events due to contention on slow processing. This behavior is controlled by the `UnlockWhileProcessing` client-go feature gate, which is enabled by default. ([#136264](https://github.com/kubernetes/kubernetes/pull/136264), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Scheduling]
- Client-go: informer stores now keep track the resourceVersion they are synced to (via add/update/delete events, or replace calls, or bookmark events), and provide a `LastStoreSyncResourceVersion` method to obtain this resource version. This method can return `""` if the store has not been synced to yet, and depends on the `AtomicFIFO` feature being enabled. ([#134827](https://github.com/kubernetes/kubernetes/pull/134827), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Testing]
- CustomResourceDefinition (CRD) validation now strictly enforces ranges for numeric formats (`int32`, `int64`, `float`, `double`) when specified in the schema. Existing objects with out-of-range values are preserved via validation ratcheting. ([#136582](https://github.com/kubernetes/kubernetes/pull/136582), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- DRA ResourceSlice controller: Added optional `ReconcilePoolWithName` to allow per-pool reconciliation without setting `NodeName` on slices, so the scheduler can use `NodeSelector` or `allNodes` for node-owned, cluster-visible resources (e.g. network-shared devices). "All nodes" is no longer the default. When publishing devices for the entire cluster, it must be set explicitly. ([#137365](https://github.com/kubernetes/kubernetes/pull/137365), [@yaroslavborbat](https://github.com/yaroslavborbat)) [SIG Node and Testing]
- Enabled Prometheus native histogram support in `kube-apiserver` when the feature gate is enabled. Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160). ([#136763](https://github.com/kubernetes/kubernetes/pull/136763), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Cloud Provider, Instrumentation, Network, Node, Scheduling and Testing]
- Enabled Prometheus native histogram support in `kube-controller-manager` when the feature gate is enabled. Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160). ([#137779](https://github.com/kubernetes/kubernetes/pull/137779), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation and Testing]
- Enabled Prometheus native histogram support in `kube-proxy` when the feature gate is enabled. Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160). ([#137781](https://github.com/kubernetes/kubernetes/pull/137781), [@richabanker](https://github.com/richabanker)) [SIG Network]
- Enabled Prometheus native histogram support in `kube-scheduler` when the feature gate is enabled. Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160). ([#137466](https://github.com/kubernetes/kubernetes/pull/137466), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation, Scheduling and Testing]
- Enabled Prometheus native histogram support in `kubelet` when the feature gate is enabled. Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160). ([#137780](https://github.com/kubernetes/kubernetes/pull/137780), [@richabanker](https://github.com/richabanker)) [SIG Node]
- Enabled the Topology, CPU, and Memory managers to recognize and act upon `pod.spec.resources`, enabling two flexible resource management models. Both models support `guaranteed` Pods that contain a mix of containers that may be eligible to receive exclusive resource allocation or be part of the Pod-allocated shared resource pool. ([#134768](https://github.com/kubernetes/kubernetes/pull/134768), [@KevinTMtz](https://github.com/KevinTMtz)) [SIG Node and Testing]
- Enabled the `WatchCacheInitializationPostStartHook` feature gate by default. ([#135777](https://github.com/kubernetes/kubernetes/pull/135777), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Enabled workload-aware preemption for PodGroups when the `WorkloadAwarePreemption` feature gate is active. When PodGroup scheduling fails to find placement for a PodGroup, workload-aware preemption runs for the entire group instead of running default preemption for each individual Pod. ([#137606](https://github.com/kubernetes/kubernetes/pull/137606), [@Argh4k](https://github.com/Argh4k)) [SIG Apps, Node, Scheduling, Storage and Testing]
- Ensured single-container Pod can restart quickly with the `RestartAllContainers` action. ([#136966](https://github.com/kubernetes/kubernetes/pull/136966), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node and Testing]
- Fixed missing field conversions (`BindsToNode`, `BindingConditions`, `BindingFailureConditions`, `AllowMultipleAllocations`, `Capacity`) in DRA API `v1beta1` hand-written conversion code. ([#137240](https://github.com/kubernetes/kubernetes/pull/137240), [@yykkibbb](https://github.com/yykkibbb)) [SIG Node]
- Graduated `ComponentFlagz` to beta. ([#137386](https://github.com/kubernetes/kubernetes/pull/137386), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Auth, Instrumentation, Node and Testing]
- Graduated `ComponentStatusz` to beta. ([#137384](https://github.com/kubernetes/kubernetes/pull/137384), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Auth and Instrumentation]
- Graduated fine-grained kubelet API authorization to stable. ([#136116](https://github.com/kubernetes/kubernetes/pull/136116), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Node]
- Graduated the `KubeletPSI` feature to GA, enabled by default. The `kubelet` exposes Linux cgroup Pressure Stall Information (PSI) metrics, providing deeper visibility into system and Pod-level resource contention (CPU, Memory, and I/O) via the `kubelet` Summary API. ([#136548](https://github.com/kubernetes/kubernetes/pull/136548), [@mariafromano-25](https://github.com/mariafromano-25)) [SIG Node]
- Improved preemption behavior so that pods preempted during the `PreBind` phase are now re-queued into the backoff queue instead of being deleted via the API server, enabling more graceful handling of preemption during binding. ([#135502](https://github.com/kubernetes/kubernetes/pull/135502), [@Argh4k](https://github.com/Argh4k)) [SIG Scheduling and Testing]
- Instrumented `/flagz` and `/statusz` endpoints with apiserver request metrics (`apiserver_request_total`, `apiserver_request_duration_seconds`), with group and version labels reflecting the content-negotiated API version. ([#137021](https://github.com/kubernetes/kubernetes/pull/137021), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Instrumentation]
- Introduced index-based naming in the ResourceSlice controller and ensured ResourceSlices and pools are sorted lexicographically before allocation, allowing users to control allocation priority. ([#136641](https://github.com/kubernetes/kubernetes/pull/136641), [@troychiu](https://github.com/troychiu)) [SIG Node and Testing]
- Introduced new staging modules `k8s.io/streaming` and `k8s.io/cri-streaming` for Kubernetes streaming transport and CRI streaming server code. `k8s.io/apimachinery/pkg/util/httpstream` (including `spdy` and `wsstream`) remains available as a deprecated compatibility wrapper backed by `k8s.io/streaming`. The extracted SPDY roundtripper preserves CIDR matching in `NO_PROXY`/`no_proxy`. ([#137298](https://github.com/kubernetes/kubernetes/pull/137298), [@dims](https://github.com/dims)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Kube-apiserver: Graduated the `UnknownVersionInteroperabilityProxy` feature gate to beta, enabled by default. The `--peer-ca-file` flag is required to turn on the proxy. ([#137172](https://github.com/kubernetes/kubernetes/pull/137172), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Kube-apiserver: Promoted the `ExternalServiceAccountTokenSigner` feature gate to GA. ([#136118](https://github.com/kubernetes/kubernetes/pull/136118), [@HarshalNeelkamal](https://github.com/HarshalNeelkamal)) [SIG API Machinery and Auth]
- Kube-controller-manager: The daemonset controller now defers syncing a DaemonSet object when the controller has not yet observed daemonset or pod writes from the last time the object was synced. This prevents spurious creation of duplicate pods for nodes when the controller's cache is stale. When a sync is deferred for this reason, a `daemonset_controller_stale_sync_skips_total` metric is incremented and a message is logged by the daemonset controller. This behavior can be temporarily disabled by setting the `StaleControllerConsistencyDaemonSet` feature gate to false. ([#134937](https://github.com/kubernetes/kubernetes/pull/134937), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Kube-controller-manager: The job controller now defers syncing a Job object when the controller has not yet observed job or pod writes from the last time the object was synced. This prevents spurious creation of duplicate pods for jobs when the controller's cache is stale. When a sync is deferred for this reason, a `job_controller_stale_sync_skips_total` metric is incremented and a message is logged by the job controller. This behavior can be temporarily disabled by setting the `StaleControllerConsistencyJob` feature gate to false. ([#137210](https://github.com/kubernetes/kubernetes/pull/137210), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Apps]
- Kubeadm: Added the `--allow-deprecated-api` flag to `kubeadm config validate`. By default the command prints a warning for deprecated APIs unless the flag is passed. Additionally, added missing support for `v1beta4` `UpgradeConfiguration` to `kubeadm config migrate` and `kubeadm config validate` commands. ([#135148](https://github.com/kubernetes/kubernetes/pull/135148), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: Changed Node object patching behavior to retry on unknown (non-allowlisted) API errors within the polling duration instead of exiting early. ([#135776](https://github.com/kubernetes/kubernetes/pull/135776), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: Increased the timeout of the `kubeadm upgrade` `CreateJob` preflight check to 1 minute. This allows Windows worker nodes to have more time to run the preflight check. The check uses the `pause` image, so if you are experiencing slow pull times, you can either pre-pull the image on the worker using `kubeadm config images pull --kubernetes-version TARGET` or skip the preflight check with `--ignore-preflight-errors`. ([#136273](https://github.com/kubernetes/kubernetes/pull/136273), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: Promoted the `NodeLocalCRISocket` feature gate to GA and locked it to enabled. ([#135742](https://github.com/kubernetes/kubernetes/pull/135742), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Kubeadm: Removed the `ControlPlaneKubeletLocalMode` feature gate, which graduated to GA in `v1.35` and was locked to enabled. ([#135773](https://github.com/kubernetes/kubernetes/pull/135773), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: The preflight check `ContainerRuntimeVersion` validates if the installed container runtime supports the `RuntimeConfig` gRPC method. For older kubelet versions than `v1.37`, it will return a preflight warning. ([#136898](https://github.com/kubernetes/kubernetes/pull/136898), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: When using `--v=1` or higher log verbosity, prints information about the CA certificate used for discovery when using `kubeadm join`. ([#137102](https://github.com/kubernetes/kubernetes/pull/137102), [@sivchari](https://github.com/sivchari)) [SIG Cluster Lifecycle]
- Kubelet: Deferred the removal of deprecated kubelet configuration flags (and their related fallback behavior) from version 1.36 to 1.37, aligning with the end of containerd v1.7 support. ([#136846](https://github.com/kubernetes/kubernetes/pull/136846), [@carlory](https://github.com/carlory)) [SIG Node and Testing]
- Kubelet: If the `--client-ca-file` is updated while `kubelet` is running, the updated root certificates are correctly used to advertise accepted authorities to TLS clients connecting to the `kubelet` endpoints. This behavior is guarded by the `ReloadKubeletClientCAFile` feature gate, which is enabled by default. ([#136762](https://github.com/kubernetes/kubernetes/pull/136762), [@HarshalNeelkamal](https://github.com/HarshalNeelkamal)) [SIG API Machinery, Auth, Node and Testing]
- Kubernetes is now built using Go `v1.26.2`. ([#138299](https://github.com/kubernetes/kubernetes/pull/138299), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Kubernetes is now built using Go `v1.26.0`. ([#137080](https://github.com/kubernetes/kubernetes/pull/137080), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with Go `v1.25.6`. ([#136257](https://github.com/kubernetes/kubernetes/pull/136257), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Kubernetes is now built with Go `v1.25.6`. ([#136465](https://github.com/kubernetes/kubernetes/pull/136465), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with Go `v1.25.7`. ([#136750](https://github.com/kubernetes/kubernetes/pull/136750), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Kubernetes is now built with Go `v1.25.7`. ([#136982](https://github.com/kubernetes/kubernetes/pull/136982), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Preserved the logs of restarted containers for containers restarted by the `RestartAllContainers` feature. ([#136963](https://github.com/kubernetes/kubernetes/pull/136963), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node]
- Promoted `DRAPartitionableDevices` to beta. ([#137350](https://github.com/kubernetes/kubernetes/pull/137350), [@mortent](https://github.com/mortent)) [SIG Node, Scheduling and Testing]
- Promoted `kubectl kuberc` commands to beta. ([#136643](https://github.com/kubernetes/kubernetes/pull/136643), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Promoted the `CSIServiceAccountTokenSecrets` feature gate to GA. ([#136596](https://github.com/kubernetes/kubernetes/pull/136596), [@aramase](https://github.com/aramase)) [SIG Auth and Storage]
- Promoted the `KubeletPodResourcesDynamicResources` and `KubeletPodResourcesGet` feature gates to GA. ([#136728](https://github.com/kubernetes/kubernetes/pull/136728), [@guptaNswati](https://github.com/guptaNswati)) [SIG Node and Testing]
- Promoted the `RelaxedServiceNameValidation` feature gate to beta and enabled it by default.
  Service names are now validated with `NameIsDNSLabel()`, relaxing the pre-existing validation. ([#136389](https://github.com/kubernetes/kubernetes/pull/136389), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network]
- Promoted the `RestartAllContainersOnContainerExits` feature gate to beta, enabled by default. ([#136681](https://github.com/kubernetes/kubernetes/pull/136681), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node and Testing]
- Reduced the needs of the setcap build image for `kube-apiserver` by no longer requiring that image to contain a shell (`sh` or `dash` or `bash`). ([#136633](https://github.com/kubernetes/kubernetes/pull/136633), [@addyess](https://github.com/addyess)) [SIG Release]
- Reverted the addition of the `image_id` field to the CRI API `PullImageResponse` message. ([#137574](https://github.com/kubernetes/kubernetes/pull/137574), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node]
- Server images now use `staging/src/k8s.io/component-base/logs/kube-log-runner` instead of `go-runner`; full compatibility is maintained (including the same `/go-runner` executable path). In the future Kubernetes will use base-images without go-runner. ([#136954](https://github.com/kubernetes/kubernetes/pull/136954), [@BenTheElder](https://github.com/BenTheElder)) [SIG Instrumentation and Release]
- Updated CoreDNS to `v1.14.2`. ([#137605](https://github.com/kubernetes/kubernetes/pull/137605), [@pacoxu](https://github.com/pacoxu)) [SIG Cloud Provider and Cluster Lifecycle]
- Updated `kubectl describe node` to list aggregated ResourceSlices when the ResourceSlice API is present, detailing slice name, driver, and pool. ([#131744](https://github.com/kubernetes/kubernetes/pull/131744), [@ArangoGutierrez](https://github.com/ArangoGutierrez)) [SIG CLI]
- Updated `kubectl explain` to display an EXTERNAL DOCS section when a schema or field includes an `externalDocs` section. This appears after the DESCRIPTION block for top-level resources and after the field description for individual fields. The section is omitted in short mode and when `externalDocs` is absent. ([#136988](https://github.com/kubernetes/kubernetes/pull/136988), [@pedjak](https://github.com/pedjak)) [SIG CLI]
- Updated `kubectl get ingressclass` to display a `(default)` marker for the default IngressClass. ([#134422](https://github.com/kubernetes/kubernetes/pull/134422), [@jaehanbyun](https://github.com/jaehanbyun)) [SIG CLI and Network]
- Updated `kubectl kuberc set` with options for setting `credentialPluginPolicy` and `credentialPluginAllowlist`. ([#137300](https://github.com/kubernetes/kubernetes/pull/137300), [@pmengelbert](https://github.com/pmengelbert)) [SIG CLI]
- Updated cAdvisor to `v0.55.0` in vendor dependencies. ([#135829](https://github.com/kubernetes/kubernetes/pull/135829), [@dims](https://github.com/dims)) [SIG Node]
- Updated feature gate `MutablePodResourcesForSuspendedJobs` and `MutableSchedulingDirectivesForSuspendedJobs` to be enabled by default. ([#135965](https://github.com/kubernetes/kubernetes/pull/135965), [@kannon92](https://github.com/kannon92)) [SIG Apps and Testing]
- Updated node performance e2e tests to use PyTorch Wide-Deep workload instead of TensorFlow. ([#136397](https://github.com/kubernetes/kubernetes/pull/136397), [@dims](https://github.com/dims)) [SIG Testing]
- Updated node performance e2e tests to use PyTorch Wide-Deep workload instead of TensorFlow. ([#136398](https://github.com/kubernetes/kubernetes/pull/136398), [@dims](https://github.com/dims)) [SIG Node and Testing]
- Updated the `ImageLocality` scheduler plugin to consider `ImageVolume` images when scoring nodes for Pod scheduling. ([#130231](https://github.com/kubernetes/kubernetes/pull/130231), [@Barakmor1](https://github.com/Barakmor1)) [SIG Scheduling]
- When `kubectl exec` or `kubectl logs` are run with a specified container name, and no container with that name is found, `kubectl` lists the names of containers that would be valid to specify. ([#136973](https://github.com/kubernetes/kubernetes/pull/136973), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]

### Documentation

- Added metric component and endpoint to generated metric reference documentation. ([#136360](https://github.com/kubernetes/kubernetes/pull/136360), [@skl](https://github.com/skl)) [SIG Instrumentation and Testing]

### Failing Test

- Kubelet: Fixed device plugin test failures after kubelet restart. ([#135485](https://github.com/kubernetes/kubernetes/pull/135485), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node and Testing]
- The `PLEGOnDemandRelist` feature flag is kept at beta level, but switched off by default. ([#137909](https://github.com/kubernetes/kubernetes/pull/137909), [@dims](https://github.com/dims)) [SIG Node]

### Bug or Regression

- Added the `--detach-keys` flag to `kubectl attach` and `kubectl run`, allowing detach without terminating the container. ([#134997](https://github.com/kubernetes/kubernetes/pull/134997), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG API Machinery and CLI]
- Capped `nf_conntrack_max` to 1,048,576 to prevent excessive memory consumption on high-core machines when using automatic calculation. ([#137002](https://github.com/kubernetes/kubernetes/pull/137002), [@kairosci](https://github.com/kairosci)) [SIG Apps and Network]
- Changed some error logs to info logs with verbosity level in `controller/resourcequota` and `controller/garbagecollector`. ([#136040](https://github.com/kubernetes/kubernetes/pull/136040), [@petern48](https://github.com/petern48)) [SIG API Machinery and Apps]
- Changed the `nodeGetCapabilities` method of `csiDriverClient` to return `NewUncertainProgressError` when receiving a non-final gRPC error. This resolves residual global mount paths during rapid pod creation-deletion cycles. ([#135930](https://github.com/kubernetes/kubernetes/pull/135930), [@249043822](https://github.com/249043822)) [SIG Node and Storage]
- Changed the behavior of default scheduler preemption plugin when preempting Pods that are in `WaitOnPermit` phase. They are now moved to the scheduler backoff queue instead of being marked as unschedulable. ([#135719](https://github.com/kubernetes/kubernetes/pull/135719), [@Argh4k](https://github.com/Argh4k)) [SIG Scheduling and Testing]
- Changed the runtime handlers list returned by the CRI runtime to be sorted, preventing unnecessary Node object updates when the order changes. ([#135358](https://github.com/kubernetes/kubernetes/pull/135358), [@harche](https://github.com/harche)) [SIG Node]
- Client-go: Fixed an unlikely deadlock during informer startup. ([#136509](https://github.com/kubernetes/kubernetes/pull/136509), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- CustomResourceDefinitions: Fixed server-side apply field ownership tracking so that metadata ownership is correctly tracked for writes to the `/status` subresource.
  Custom Resources: Fixed server-side apply field ownership to not update metadata from the `/status` subresource since these writes are wiped for custom resources. ([#137689](https://github.com/kubernetes/kubernetes/pull/137689), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Network and Testing]
- DRA BindingConditions: Fixed a panic in the scheduler when the `DRABindingConditions` feature was enabled and the same claim was reused among different Pods while deallocation happened in parallel. ([#137371](https://github.com/kubernetes/kubernetes/pull/137371), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Disabled `SchedulerAsyncAPICalls` feature gate due to performance issues caused by API client throttling. ([#135903](https://github.com/kubernetes/kubernetes/pull/135903), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Disallowed setting a resize restart policy of `RestartContainer` on non-sidecar initContainers, as the resize of such containers has never been supported. ([#137458](https://github.com/kubernetes/kubernetes/pull/137458), [@natasha41575](https://github.com/natasha41575)) [SIG Apps, Node and Testing]
- Explicitly wrote `memory.min=0` for QoS cgroups when the calculated requests are zero. ([#137637](https://github.com/kubernetes/kubernetes/pull/137637), [@QiWang19](https://github.com/QiWang19)) [SIG Node]
- Fixed SELinux warning controller to not emit events for completed Pods (Succeeded and Failed states). ([#135629](https://github.com/kubernetes/kubernetes/pull/135629), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Storage and Testing]
- Fixed StatefulSets to always count `.status.availableReplicas` at the correct time without delay, resulting in faster StatefulSet rollout progress. ([#135428](https://github.com/kubernetes/kubernetes/pull/135428), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- Fixed `DRA manager` not initializing sharedID from cache when `DRAConsumableCapacity` is enabled. ([#136734](https://github.com/kubernetes/kubernetes/pull/136734), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node and Scheduling]
- Fixed `PodCertificateRequest` OwnerReference using incorrect apiVersion "core/v1" instead of "v1", which prevented garbage collection of `PodCertificateRequests` when their owning Pod was deleted. ([#137008](https://github.com/kubernetes/kubernetes/pull/137008), [@srhppr](https://github.com/srhppr)) [SIG Auth and Node]
- Fixed `ReadWriteOncePod` preemption e2e test to run as serial, preventing it from causing other random e2e tests to flake. ([#135623](https://github.com/kubernetes/kubernetes/pull/135623), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]
- Fixed `container_swap_usage_bytes` in the `/metrics/resource` endpoint to correctly report container-level swap usage instead of always reporting 0. The root cause was missing logic in `addCadvisorContainerCPUAndMemoryStats` to propagate swap stats from cadvisor to the container stats object. ([#137098](https://github.com/kubernetes/kubernetes/pull/137098), [@yuanwang04](https://github.com/yuanwang04)) [SIG Apps, Node and Testing]
- Fixed `event_handling_duration_seconds`, `preemption_goroutines_duration_seconds`, `run_podsandbox_duration_seconds`, and `store_schedule_results_duration_seconds` metrics incorrectly recording near-zero latency values instead of actual durations, caused by premature evaluation of `SinceInSeconds(startTime)` in a deferred call. ([#135749](https://github.com/kubernetes/kubernetes/pull/135749), [@novahe](https://github.com/novahe)) [SIG Architecture, Instrumentation, Node and Scheduling]
- Fixed `kube-apiserver` startup failure during upgrade when `MultiCIDRServiceAllocator` is enabled and the cluster has a large number of namespaces. The IP address repair controller retries on Forbidden errors from admission plugins that are not yet ready. ([#137147](https://github.com/kubernetes/kubernetes/pull/137147), [@haojiwu](https://github.com/haojiwu)) [SIG Testing]
- Fixed `kube-proxy` log spam when all of a Service's endpoints were unready. ([#136743](https://github.com/kubernetes/kubernetes/pull/136743), [@ansilh](https://github.com/ansilh)) [SIG Network]
- Fixed `kubectl delete` to properly handle deletion of multiple StatefulSet pods and exit normally. ([#135563](https://github.com/kubernetes/kubernetes/pull/135563), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG CLI, Network and Node]
- Fixed `kubectl describe node` to correctly display resource requests and limits for Pods using Pod-level resources. ([#137394](https://github.com/kubernetes/kubernetes/pull/137394), [@Nikateen](https://github.com/Nikateen)) [SIG CLI]
- Fixed `kubectl describe` to correctly recognize uppercase acronyms as a single element when displaying Custom Resource field names. ([#135683](https://github.com/kubernetes/kubernetes/pull/135683), [@uozalp](https://github.com/uozalp)) [SIG CLI]
- Fixed `kubectl label` output message to display `modified` when labels are both added and removed. ([#134849](https://github.com/kubernetes/kubernetes/pull/134849), [@tchap](https://github.com/tchap)) [SIG CLI]
- Fixed `kubectl logs -f` to wait for containers to start instead of failing immediately when pods are in ContainerCreating or PodInitializing states. ([#136411](https://github.com/kubernetes/kubernetes/pull/136411), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Fixed a `v1.29` regression in the `apiserver_watch_events_sizes` metric to report total outgoing watch traffic again. ([#135367](https://github.com/kubernetes/kubernetes/pull/135367), [@mborsz](https://github.com/mborsz)) [SIG API Machinery]
- Fixed a `v1.34` regression in `ipvs` and `winkernel` `kube-proxy` backends. These backends now revert to their `pre-v1.34` behavior of regularly rechecking all rules even when no Services or EndpointSlices change. ([#135631](https://github.com/kubernetes/kubernetes/pull/135631), [@danwinship](https://github.com/danwinship)) [SIG Network and Windows]
- Fixed a `v1.34` regression when starting pods with environment variables containing a value with `$` followed by a multi-byte character. ([#136325](https://github.com/kubernetes/kubernetes/pull/136325), [@AutuSnow](https://github.com/AutuSnow)) [SIG Architecture]
- Fixed a `v1.35` regression in StatefulSet parallel Pod management by disabling the `MaxUnavailableStatefulSet` feature by default. ([#137904](https://github.com/kubernetes/kubernetes/pull/137904), [@soltysh](https://github.com/soltysh)) [SIG Apps]
- Fixed a bug causing clients to error out when decoding large CBOR encoded lists. ([#135340](https://github.com/kubernetes/kubernetes/pull/135340), [@ricardomaraschini](https://github.com/ricardomaraschini)) [SIG API Machinery]
- Fixed a bug in `DeepEqualWithNilDifferentFromEmpty` where empty slices and maps were incorrectly considered equal to non-empty ones due to using OR (`||`) instead of AND (`&&`) logic. This could cause managed fields timestamps to not update when the only change was adding or removing all elements from a list or map. ([#135636](https://github.com/kubernetes/kubernetes/pull/135636), [@mikecook](https://github.com/mikecook)) [SIG API Machinery]
- Fixed a bug in the `dra_operations_duration_seconds` metric where the `is_error` label was recording inverted values. Error operations now correctly report `is_error=true`, and successful operations report `is_error=false`. ([#135227](https://github.com/kubernetes/kubernetes/pull/135227), [@hime](https://github.com/hime)) [SIG Node]
- Fixed a bug preventing Pods sharing ResourceClaims from being scheduled with GangScheduling. ([#137647](https://github.com/kubernetes/kubernetes/pull/137647), [@nojnhuh](https://github.com/nojnhuh)) [SIG Node, Scheduling and Testing]
- Fixed a bug that caused `EndpointSlice` churn for headless services with no ports defined. ([#136502](https://github.com/kubernetes/kubernetes/pull/136502), [@tzneal](https://github.com/tzneal)) [SIG Network]
- Fixed a bug where `kubectl apply --dry-run=client` would only output server state instead of merged manifest values when the resource already exists. ([#135513](https://github.com/kubernetes/kubernetes/pull/135513), [@grandeit](https://github.com/grandeit)) [SIG CLI]
- Fixed a bug where `kubectl plugin list` failed to detect overshadowed plugins on Windows. ([#136689](https://github.com/kubernetes/kubernetes/pull/136689), [@kfess](https://github.com/kfess)) [SIG CLI]
- Fixed a bug where the Gated pods metric was not updated when a Pod transitioned from Unschedulable to Gated during an update. ([#135368](https://github.com/kubernetes/kubernetes/pull/135368), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling]
- Fixed a bug where the `scheduler_unschedulable_pods` metric could be artificially inflated (leak) when a pod fails `PreEnqueue` plugins after being previously marked unschedulable. ([#135981](https://github.com/kubernetes/kubernetes/pull/135981), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling]
- Fixed a bug where users could not update HPAv2 resources that use object metrics with `averageValue` via the v1 HPA API. ([#137856](https://github.com/kubernetes/kubernetes/pull/137856), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Autoscaling]
- Fixed a bug where, after a `kubelet` restart, regular containers in a Pod with a sidecar (initContainer with `restartPolicy`: Always) and a `startupProbe` failed to restart after crashing. Affected Pods remained stuck with `RestartCount: 0` indefinitely. ([#137146](https://github.com/kubernetes/kubernetes/pull/137146), [@george-angel](https://github.com/george-angel)) [SIG Node and Testing]
- Fixed a data race in the `PopulateRefs` function in `k8s.io/apiserver/pkg/cel/openapi/resolver` where concurrent goroutines could simultaneously modify shared pointer fields from a shallow-copied schema struct. ([#136802](https://github.com/kubernetes/kubernetes/pull/136802), [@pohly](https://github.com/pohly)) [SIG API Machinery, Node and Testing]
- Fixed a kubelet device manager bug where topology hint computation enumerated O(2^n) NUMA node combinations using all machine NUMA nodes. On systems with many NUMA nodes that carry no devices (e.g. NVIDIA GB200 with 36 NUMA nodes), this caused kubelet to stall indefinitely during pod admission. The device manager now restricts iteration to NUMA nodes that actually host devices for the requested resource, reducing the search space to O(2^k) where k is typically 1–2. ([#138244](https://github.com/kubernetes/kubernetes/pull/138244), [@fanzhangio](https://github.com/fanzhangio)) [SIG Node]
- Fixed a loophole that allowed users to work around DRA extended resource quota set by system administrators. ([#135434](https://github.com/kubernetes/kubernetes/pull/135434), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Fixed a race condition in CEL admission policy compilation that could cause `kube-apiserver` to crash with a `concurrent map read and map write` error under high load. ([#135759](https://github.com/kubernetes/kubernetes/pull/135759), [@Abhigyan-Shekhar](https://github.com/Abhigyan-Shekhar)) [SIG API Machinery and CLI]
- Fixed a race condition in Dynamic Resource Allocation (DRA) where the same device could be allocated twice for different `ResourceClaims` when scheduling many pods very rapidly. Depending on whether DRA drivers check for this during `NodePrepareResources` (they should, but not all may implement this properly), the second pod using the same device could fail to start until the first one is done or (worse) run in parallel. ([#136269](https://github.com/kubernetes/kubernetes/pull/136269), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Fixed an issue in the Windows `kube-proxy` (winkernel) where IPv4 and IPv6 Service load balancers could be incorrectly shared, causing broken dual-stack Service behavior. The `kube-proxy` now tracks load balancers per IP family, enabling correct support for `PreferDualStack` and `RequireDualStack` Services on Windows nodes. ([#136241](https://github.com/kubernetes/kubernetes/pull/136241), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fixed an issue where zero-valued PSI (Pressure Stall Information) metrics were emitted by the `kubelet` when the OS does not support PSI, even if the `KubeletPSI` feature gate was enabled. ([#137326](https://github.com/kubernetes/kubernetes/pull/137326), [@amritansh1502](https://github.com/amritansh1502)) [SIG Node]
- Fixed container restart policy validation error message to correctly show available actions when the `RestartAllContainersOnContainerExits` feature gate is enabled. ([#137369](https://github.com/kubernetes/kubernetes/pull/137369), [@kfess](https://github.com/kfess)) [SIG Apps]
- Fixed erroneously reporting a pod-level resize in progress on Pod creation when the `InPlacePodLevelResourcesVerticalScaling` feature gate is enabled. ([#138049](https://github.com/kubernetes/kubernetes/pull/138049), [@ndixita](https://github.com/ndixita)) [SIG Node and Testing]
- Fixed feature gates `ChangeContainerStatusOnKubeletRestart` and `StatefulSetSemanticRevisionComparison` to be visible in `--help` output across different components. ([#135515](https://github.com/kubernetes/kubernetes/pull/135515), [@dims](https://github.com/dims)) [SIG Architecture]
- Fixed goroutine hot-loop in client-go `StartEventWatcher` when the event broadcaster shuts down before the cancellation context fires. ([#137398](https://github.com/kubernetes/kubernetes/pull/137398), [@Rajneesh180](https://github.com/Rajneesh180)) [SIG API Machinery]
- Fixed how image names are compared to the values from `preloadedImagesVerificationAllowlist` in the `kubelet`'s configuration. Previously, the use of "familiar" image names (e.g. "alpine") from a Pod did not properly match the same name in `preloadedImagesVerificationAllowlist` in the `kubelet`'s configuration. ([#137629](https://github.com/kubernetes/kubernetes/pull/137629), [@stlaz](https://github.com/stlaz)) [SIG Auth, Node and Testing]
- Fixed incorrect behavior when using AllocationModeAll with DRA PrioritizedList that prevented the allocator from successfully allocating a claim even when devices were available. ([#137347](https://github.com/kubernetes/kubernetes/pull/137347), [@mortent](https://github.com/mortent)) [SIG Node]
- Fixed informer-gen to generate SetTransform calls that correctly override per-informer transforms. ([#137473](https://github.com/kubernetes/kubernetes/pull/137473), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Scheduling]
- Fixed issues in server side apply and client-go's `Extract{TypeName}()` and `Extract{TypeName}From()` functions where empty arrays and maps were incorrectly treated as absent, and atomic elements from associative lists were incorrectly duplicated. ([#135391](https://github.com/kubernetes/kubernetes/pull/135391), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Network, Node, Scheduling and Storage]
- Fixed kubeadm to skip appending the client URL of etcd learner members to `c.Endpoints`, since learners do not serve client traffic. ([#137251](https://github.com/kubernetes/kubernetes/pull/137251), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Fixed link file ownership of projected serviceAccountToken. ([#137332](https://github.com/kubernetes/kubernetes/pull/137332), [@gavinkflam](https://github.com/gavinkflam)) [SIG Storage]
- Fixed log verbosity for non-error messages in the SELinux warning controller so they are no longer logged at error level. ([#136050](https://github.com/kubernetes/kubernetes/pull/136050), [@ShaanveerS](https://github.com/ShaanveerS)) [SIG Apps and Storage]
- Fixed log verbosity for non-error messages in the storage version migrator so they are no longer logged at error level. ([#136046](https://github.com/kubernetes/kubernetes/pull/136046), [@Tanner-Gladson](https://github.com/Tanner-Gladson)) [SIG API Machinery and Apps]
- Fixed queue hint for certain plugins on change to pods with nominated nodes. ([#135392](https://github.com/kubernetes/kubernetes/pull/135392), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fixed queue hint for inter-pod anti-affinity in case deleted pod's anti-affinity matched the pending pod, which might have caused delays in scheduling. ([#135325](https://github.com/kubernetes/kubernetes/pull/135325), [@brejman](https://github.com/brejman)) [SIG Scheduling and Testing]
- Fixed queue hint for the `interpodaffinity` plugin in case target pod labels change. ([#135394](https://github.com/kubernetes/kubernetes/pull/135394), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fixed redundant SSH command executions in the `etcd` failure e2e test. ([#137001](https://github.com/kubernetes/kubernetes/pull/137001), [@kairosci](https://github.com/kairosci)) [SIG API Machinery and Testing]
- Fixed running of DRA e2e tests in air-gapped clusters or with test images in private registries. ([#138318](https://github.com/kubernetes/kubernetes/pull/138318), [@jsafrane](https://github.com/jsafrane)) [SIG Node and Testing]
- Fixed static pod status displaying `Init:0/1` when unable to retrieve init container status from container runtime. ([#131317](https://github.com/kubernetes/kubernetes/pull/131317), [@bitoku](https://github.com/bitoku)) [SIG Node and Testing]
- Fixed the `lastTerminationStatus` to match the `RestartAllContainers` action if the container was restarted this way. ([#136964](https://github.com/kubernetes/kubernetes/pull/136964), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node]
- Fixed the total Pod resources computation. ([#137683](https://github.com/kubernetes/kubernetes/pull/137683), [@ndixita](https://github.com/ndixita)) [SIG CLI and Node]
- Fixed unsupported `Table` object detection to cover all List and Watch operations, preventing the reflector from incorrectly processing resources returned in `Table` format. ([#136937](https://github.com/kubernetes/kubernetes/pull/136937), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Fixed validation error messages for `restartPolicyRules` and `exitCodes.values` to report "items" instead of "bytes". ([#137136](https://github.com/kubernetes/kubernetes/pull/137136), [@kfess](https://github.com/kfess)) [SIG Apps]
- Improved CPU usage in the `nftables` mode of `kube-proxy` when loading very large rulesets. ([#135800](https://github.com/kubernetes/kubernetes/pull/135800), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Improved `DRA scheduling` performance by splitting ResourceSlice entries into shared and onNode categories, reducing Filter stage latency by ~50% in large clusters. ([#136588](https://github.com/kubernetes/kubernetes/pull/136588), [@abel-von](https://github.com/abel-von)) [SIG API Machinery, Apps, Auth, Node and Scheduling]
- Improved a misleading error message when updating `batch.Job`'s `status.startTime`. The error for unsuspended Jobs correctly indicates the field is immutable once set, instead of incorrectly referring to the action as a "removal". ([#136585](https://github.com/kubernetes/kubernetes/pull/136585), [@zhzhuang-zju](https://github.com/zhzhuang-zju)) [SIG Apps]
- Kube-apiserver: Fixed request latency annotation `apiserver.latency.k8s.io/total` in the audit log when request took more than `500ms`. ([#135685](https://github.com/kubernetes/kubernetes/pull/135685), [@chaochn47](https://github.com/chaochn47)) [SIG API Machinery]
- Kube-apiserver: Fixed the log verbosity level in the unsafe delete authorization check that was incorrectly using Error level instead of Info level. ([#136229](https://github.com/kubernetes/kubernetes/pull/136229), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- Kube-apiserver: Liveness probes will now fail when the loopback client certificate expires. ([#136477](https://github.com/kubernetes/kubernetes/pull/136477), [@everettraven](https://github.com/everettraven)) [SIG API Machinery and Testing]
- Kube-apiserver: Setting `--audit-log-maxsize=0` now disables audit log rotation (the default remains `100 MB`). To avoid outages due to filling disks with ever-growing audit logs, `--audit-log-maxage` now defaults to `366 (1 year)` and `--audit-log-maxbackup` now defaults to `100`. If retention of all rotated logs is desired, age and count-based pruning can be disabled by explicitly specifying `--audit-log-maxage=0` and `--audit-log-maxbackup=0`. ([#136478](https://github.com/kubernetes/kubernetes/pull/136478), [@kairosci](https://github.com/kairosci)) [SIG API Machinery]
- Kube-controller-manager: Fixed `VolumeAttachment` cleanup when CSI's `attachRequired` switches from true to false. ([#129664](https://github.com/kubernetes/kubernetes/pull/129664), [@hkttty2009](https://github.com/hkttty2009)) [SIG Storage and Testing]
- Kube-proxy now correctly handles the case where a pod IP gets assigned to
  a newly-created pod when the pod that previously had that IP has been
  terminated but is not yet fully deleted. ([#135593](https://github.com/kubernetes/kubernetes/pull/135593), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kube-proxy: Fixed nftables mode to work on systems with `nft` `v1.1.3`. ([#137501](https://github.com/kubernetes/kubernetes/pull/137501), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kubeadm: Changed `kubeadm join` to wait for the etcd learner member to start before promoting it. ([#136014](https://github.com/kubernetes/kubernetes/pull/136014), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: Fixed a bug where `kubeadm upgrade` failed if the content of the `/var/lib/kubelet/kubeadm-flags.env` file was `KUBELET_KUBEADM_ARGS=""`. ([#136127](https://github.com/kubernetes/kubernetes/pull/136127), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: Ignored EINVAL when unmounting `/var/lib/kubelet` peer mounts during reset. ([#137494](https://github.com/kubernetes/kubernetes/pull/137494), [@fuweid](https://github.com/fuweid)) [SIG Cluster Lifecycle]
- Kubeadm: When applying user-provided overrides using `extraArgs`, the resulting list of arguments is no longer sorted alphanumerically. Only default arguments are sorted, while overrides preserve their order. This allows finer control for flags where order matters, such as `--service-account-issuer` for `kube-apiserver`. ([#135400](https://github.com/kubernetes/kubernetes/pull/135400), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl: Fixed `kyaml` output of `kubectl get ... --output-watch-events -o kyaml`. ([#136110](https://github.com/kubernetes/kubernetes/pull/136110), [@liggitt](https://github.com/liggitt)) [SIG CLI]
- Kubectl: Fixed a panic in `kubectl exec` when the terminal size queue delegate is uninitialized. ([#135918](https://github.com/kubernetes/kubernetes/pull/135918), [@MarcosDaNight](https://github.com/MarcosDaNight)) [SIG CLI]
- Kubectl: Fixed a panic when processing pods with nil resource requests but populated container status resources. ([#136534](https://github.com/kubernetes/kubernetes/pull/136534), [@dmaizel](https://github.com/dmaizel)) [SIG CLI]
- Kubectl: Fixed an issue where `kubectl run -i/-it` would miss container output written before the attach connection was established. ([#136010](https://github.com/kubernetes/kubernetes/pull/136010), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Kubelet: Fixed Dynamic Resource Allocation (DRA) to correctly handle multiple `ResourceClaims` even if one is already prepared. ([#135919](https://github.com/kubernetes/kubernetes/pull/135919), [@rogowski-piotr](https://github.com/rogowski-piotr)) [SIG Node and Testing]
- Kubelet: Fixed a data race in pod allocated resources. ([#136226](https://github.com/kubernetes/kubernetes/pull/136226), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Kubelet: Fixed a data race in the container manager. ([#136206](https://github.com/kubernetes/kubernetes/pull/136206), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Kubelet: Fixed a data race in the status manager. ([#136205](https://github.com/kubernetes/kubernetes/pull/136205), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Kubelet: Fixed a data race in the volume manager's `WaitForAllPodsUnmount` that could cause errors to be lost during concurrent pod unmount operations. ([#135794](https://github.com/kubernetes/kubernetes/pull/135794), [@AutuSnow](https://github.com/AutuSnow)) [SIG Node and Storage]
- Kubelet: Fixed a nil pointer dereference when handling pod updates of mirror pods with the `NodeDeclaredFeatures` feature gate enabled. ([#136037](https://github.com/kubernetes/kubernetes/pull/136037), [@pravk03](https://github.com/pravk03)) [SIG Node]
- Kubelet: Fixed logging to properly respect verbosity levels. Previously, some debug/info messages using `V().Error()` would always be printed regardless of the configured log verbosity. ([#136028](https://github.com/kubernetes/kubernetes/pull/136028), [@thc1006](https://github.com/thc1006)) [SIG Node]
- Kubelet: Fixed preservation of DRA `NodeAllocatableResourceClaimStatuses` in PodStatus. ([#138030](https://github.com/kubernetes/kubernetes/pull/138030), [@askervin](https://github.com/askervin)) [SIG Node]
- Kubelet: Fixed reloading of server certificate files when they are changed on disk and kubelet is dialed by IP address instead of DNS/hostname. ([#133654](https://github.com/kubernetes/kubernetes/pull/133654), [@kwohlfahrt](https://github.com/kwohlfahrt)) [SIG API Machinery, Auth, Node and Testing]
- Kubelet: Relisted Pods on-demand for lower latency operations. Guarded by the beta feature gate `PLEGOnDemandRelist`. ([#137362](https://github.com/kubernetes/kubernetes/pull/137362), [@tallclair](https://github.com/tallclair)) [SIG Node]
- Kubelet: The plugin manager now properly handles plugin registration failures by removing failed plugins from the actual state and retrying with exponential backoff (initial delay `500ms`, doubling each failure up to `~2 minutes` maximum) to protect against broken plugins causing denial of service while still allowing recovery from transient failures. ([#133335](https://github.com/kubernetes/kubernetes/pull/133335), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Storage and Testing]
- Kubernetes is now built using Go `v1.26.1`. ([#137474](https://github.com/kubernetes/kubernetes/pull/137474), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release and Testing]
- Optimized kube-proxy conntrack cleanup logic, reducing the time complexity of deleting stale UDP entries. This significantly improves performance when there are many stale connections to clean up. ([#135511](https://github.com/kubernetes/kubernetes/pull/135511), [@aojea](https://github.com/aojea)) [SIG Network]
- Previously, when trying to allocate devices through DRA for a node timed out, scheduling would proceed with another node if any had the necessary resources. This potentially hid that a node was ignored. Worse, if scheduling was slow overall, the Pod was incorrectly moved to "unschedulable" and only retried after a periodic sweep. Timeouts are now errors that are always visible as Pod scheduling failures and get retried with per-Pod exponential backoff. ([#137607](https://github.com/kubernetes/kubernetes/pull/137607), [@0xMH](https://github.com/0xMH)) [SIG Node, Scheduling and Testing]
- Reflected the expected replica count in the output of the `kubectl scale` command. ([#136945](https://github.com/kubernetes/kubernetes/pull/136945), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Removed `GuaranteedQoSPodCPUResize` from node declared features. ([#136759](https://github.com/kubernetes/kubernetes/pull/136759), [@pravk03](https://github.com/pravk03)) [SIG Node and Testing]
- Removed `container_cpu_load_average_10s`, `container_cpu_load_d_average_10s`, and `cpu_tasks_state` metrics from being reported by cadvisor. This is done because the values were always 0, because a flag was not enabled in the kubelet. ([#134981](https://github.com/kubernetes/kubernetes/pull/134981), [@haircommander](https://github.com/haircommander)) [SIG Node and Testing]
- The `k8s.io/client-go/transport` package automatically reloads certificate authority roots from disk when they are supplied via a file path. This functionality is enabled by default and can be disabled via the `ClientsAllowCARotation` feature gate. ([#132922](https://github.com/kubernetes/kubernetes/pull/132922), [@yt2985](https://github.com/yt2985)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Instrumentation, Network, Node, Release, Scheduling and Testing]
- The `k8s.io/client-go/transport` package garbage collects TLS cache entries and client certificate rotation goroutines when a transport is no longer used. This functionality is enabled by default and can be controlled via the `ClientsAllowTLSCacheGC` feature gate. Binaries embedding `k8s.io/client-go` but not wiring the feature gates can disable it by setting the `KUBE_FEATURE_ClientsAllowTLSCacheGC=false` environment variable. When the feature is disabled, the TLS cache can grow indefinitely and client certificate rotation goroutines are leaked. The new `rest_client_transport_cert_rotation_gc_calls_total{}` and `rest_client_transport_cache_gc_calls_total{result: deleted/skipped}` counter metrics can be used with the preexisting `rest_client_transport_*` metrics to help with debugging. ([#136355](https://github.com/kubernetes/kubernetes/pull/136355), [@enj](https://github.com/enj)) [SIG API Machinery, Architecture, Auth, Instrumentation, Node and Testing]
- The `kubelet_pod_start_sli_duration_seconds_bucket` metric matches Pod startup latency SLI/SLO documentation. ([#131950](https://github.com/kubernetes/kubernetes/pull/131950), [@alimaazamat](https://github.com/alimaazamat)) [SIG Node]
- The `kubelet` sets the `PodReadyToStartContainers` condition immediately after sandbox creation rather than after image pull, reducing the time to condition True. ([#134660](https://github.com/kubernetes/kubernetes/pull/134660), [@Priyankasaggu11929](https://github.com/Priyankasaggu11929)) [SIG Apps, Node and Testing]
- The garbage collector correctly handles objects deleted externally, preventing spurious error logs. ([#136817](https://github.com/kubernetes/kubernetes/pull/136817), [@kairosci](https://github.com/kairosci)) [SIG API Machinery, Apps and Testing]
- Updated `NodeResourcesBalancedAllocation` scoring algorithm to align with the documentation. The score will now take into consideration both balance with and without the requested pod. Previous algorithm only considered balance with the requested pod. This can change the scheduling decisions in some cases. ([#135573](https://github.com/kubernetes/kubernetes/pull/135573), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Updated the CDI spec for discoverable metadata to `v0.5.0`. ([#138035](https://github.com/kubernetes/kubernetes/pull/138035), [@alaypatel07](https://github.com/alaypatel07)) [SIG Node]
- Updated the pause image to `v3.10.2`. ([#138199](https://github.com/kubernetes/kubernetes/pull/138199), [@neolit123](https://github.com/neolit123)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Scheduling and Testing]
- Validation messages for a Pod's `status.resourceClaimStatuses[].resourceClaimName` refer correctly to the `resourceClaimName` field instead of the `name` field. ([#137321](https://github.com/kubernetes/kubernetes/pull/137321), [@nojnhuh](https://github.com/nojnhuh)) [SIG Apps]
- Writes to the ServiceCIDR main resource ignore status field changes in the request, consistent with all other Kubernetes APIs. The `ServiceCIDRStatusFieldWiping` feature gate can be disabled to restore the previous behavior; it will be locked to enabled in a future release. ([#137715](https://github.com/kubernetes/kubernetes/pull/137715), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Network and Testing]

### Other (Cleanup or Flake)

- Added `audit-id` to the "Starting watch" log line. ([#136084](https://github.com/kubernetes/kubernetes/pull/136084), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Added explicit logging when `WatchList` requests complete their initial listing phase. ([#136085](https://github.com/kubernetes/kubernetes/pull/136085), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Added group, version, and resource labels to the existing alpha metric `apiserver_rerouted_request_total`. ([#137063](https://github.com/kubernetes/kubernetes/pull/137063), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Added missing tests for client-go metrics. ([#136052](https://github.com/kubernetes/kubernetes/pull/136052), [@sreeram-venkitesh](https://github.com/sreeram-venkitesh)) [SIG Architecture and Instrumentation]
- Client-go: Fake client-go (i.e., anything using `k8s.io/client-go/testing`) now supports separate List+Watch calls with checking of `ResourceVersion` in the Watch call. This closes a race condition where creating an object directly after an informer cache has synced (List call completed) and before the Watch call completed would cause that object to not be sent to the informer. A visible side-effect of adding that support is that List metadata contains a `ResourceVersion` (starting at 1 for the empty set, incremented by one for each add/update) and that Watch may return objects where it previously did not.
  Note that this List+Watch is not to be confused with the `ListWatch` feature, which uses a single call. That feature is still not supported by fake client-go. ([#136143](https://github.com/kubernetes/kubernetes/pull/136143), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth and CLI]
- Client-go: Fixed an issue where Reflector could get confused about the resource version it should use to restart a watch while receiving synthetic ADDED events at the beginning of a watch from `resourceVersion` 0 or empty string (`""`). ([#136583](https://github.com/kubernetes/kubernetes/pull/136583), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Deprecated the `SeparateCacheWatchRPC` feature gate. It is now locked to its default value (false) and can no longer be overridden. The feature gate will be removed in a future release. ([#135808](https://github.com/kubernetes/kubernetes/pull/135808), [@tico88612](https://github.com/tico88612)) [SIG API Machinery]
- Enabled YAML support for `/statusz` and `/flagz` endpoints. ([#135309](https://github.com/kubernetes/kubernetes/pull/135309), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation and Testing]
- Fixed DRA device taint eviction controller to avoid confusing intermediate status messages by delaying status updates after pod eviction until the informer cache is updated. ([#135611](https://github.com/kubernetes/kubernetes/pull/135611), [@Karthik-K-N](https://github.com/Karthik-K-N)) [SIG Apps and Scheduling]
- For performance reasons, `kubectl describe` defaults to showing related events only when describing a single object. Passing `--show-events` explicitly when describing multiple objects or fuzzy matching on prefix still shows related events if desired. ([#137145](https://github.com/kubernetes/kubernetes/pull/137145), [@mark-liu](https://github.com/mark-liu)) [SIG CLI]
- Improved stability by sorting containers by create time and ID in `kubeGenericRuntimeManager.GetPods()` and `GetPod()`. ([#137566](https://github.com/kubernetes/kubernetes/pull/137566), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Node]
- Kubeadm: Removed the cleanup of the `--pod-infra-container-image` kubelet flag from `/var/lib/kubelet/kubeadm-flags.env` on upgrade. This cleanup was necessary when upgrading to `v1.35`. ([#135807](https://github.com/kubernetes/kubernetes/pull/135807), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: Removed usage of the deprecated `etcd` flags `--experimental-initial-corrupt-check` and `--experimental-watch-progress-notify-interval` if the `etcd` version is < `v3.6.0`. In this version of kubeadm, `etcd` < `v3.6.0` is no longer supported in terms of the Kubernetes / `etcd` version mapping. These deprecated flags have been replaced by `--feature-gates=InitialCorruptCheck=true` and `--watch-progress-notify-interval`. ([#135701](https://github.com/kubernetes/kubernetes/pull/135701), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubelet: Fixed admission to correctly handle DRA-backed extended resources, allowing Pods to be admitted even when these resources are not present in the node's allocatable capacity. ([#135725](https://github.com/kubernetes/kubernetes/pull/135725), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Scheduling and Testing]
- Kubernetes is now built using Go `v1.26.2`. ([#138261](https://github.com/kubernetes/kubernetes/pull/138261), [@dims](https://github.com/dims)) [SIG Architecture and Testing]
- Locked the `DisableNodeKubeProxyVersion` feature gate to enabled by default. ([#136673](https://github.com/kubernetes/kubernetes/pull/136673), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG CLI and Network]
- Promoted HPA metrics `reconciliations_total`, `reconciliation_duration_seconds`, `metric_computation_total`, and `metric_computation_duration_seconds` to beta. ([#136178](https://github.com/kubernetes/kubernetes/pull/136178), [@omerap12](https://github.com/omerap12)) [SIG Apps, Autoscaling and Instrumentation]
- Promoted `InOrderInformers` to GA via the usage of `RealFIFO`. This means that `DeltaFIFO` will gradually be deprecated in favor of `RealFIFO` in internal implementations. ([#136601](https://github.com/kubernetes/kubernetes/pull/136601), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Promoted `SELinuxChangePolicy` and `SELinuxMountReadWriteOncePod` to GA; they are enabled unconditionally. ([#136912](https://github.com/kubernetes/kubernetes/pull/136912), [@dfajmon](https://github.com/dfajmon)) [SIG Apps, Storage and Testing]
- Reduced get PV request from KCM pv-controller for CSI volumes. ([#134290](https://github.com/kubernetes/kubernetes/pull/134290), [@huww98](https://github.com/huww98)) [SIG Apps and Storage]
- Removed `v1alpha1` `WebhookAdmissionConfiguration`. It was deprecated in `v1.17` in favor of `apiserver.config.k8s.io/v1`. ([#137379](https://github.com/kubernetes/kubernetes/pull/137379), [@aramase](https://github.com/aramase)) [SIG API Machinery and Testing]
- Removed event listing behavior when describing a deleted Pod from file using `kubectl describe -f`, ensuring consistent NotFound error handling across all resource types. ([#135281](https://github.com/kubernetes/kubernetes/pull/135281), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Removed misleading `SuggestFor` entries from `kubectl wait` so that it is no longer suggested when users type `kubectl list` or `kubectl ps`. ([#137266](https://github.com/kubernetes/kubernetes/pull/137266), [@kfess](https://github.com/kfess)) [SIG CLI and Testing]
- Removed the `WatchFromStorageWithoutResourceVersion` feature gate in `v1.36`. ([#136066](https://github.com/kubernetes/kubernetes/pull/136066), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Removed the cri-client helper method `NewLogOptions`; `LogOptions` must be constructed directly. This eliminates the unwanted dependency from cri-client to apimachinery. ([#137827](https://github.com/kubernetes/kubernetes/pull/137827), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node and Release]
- Removed the dead `--bounding-dirs` flag and `BoundingDirs` field from deepcopy-gen. ([#137348](https://github.com/kubernetes/kubernetes/pull/137348), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Removed the generally available feature gate `HonorPVReclaimPolicy`, which was locked and enabled since `v1.33`. ([#135335](https://github.com/kubernetes/kubernetes/pull/135335), [@carlory](https://github.com/carlory)) [SIG Apps and Storage]
- Renamed `PodGroupInfo` to `PodGroupState`, which may break custom scheduler plugins that use `Handle.WorkloadManager`. ([#136344](https://github.com/kubernetes/kubernetes/pull/136344), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Reverted graduation of `maxLength` property. ([#137274](https://github.com/kubernetes/kubernetes/pull/137274), [@lalitc375](https://github.com/lalitc375)) [SIG API Machinery]
- The "Failed to update lease optimistically" log message may not be shown to users anymore, depending on the log level they have set. ([#137753](https://github.com/kubernetes/kubernetes/pull/137753), [@adamkasztenny](https://github.com/adamkasztenny)) [SIG API Machinery]
- The `GetPCIeRootAttributeByPCIBusID` helper accepts a `fs.ReadLinkFS` optional argument to be filesystem-independent. ([#137220](https://github.com/kubernetes/kubernetes/pull/137220), [@ffromani](https://github.com/ffromani)) [SIG Node]
- The cri-api client accepts a context instead of a logger on initialization. ([#137248](https://github.com/kubernetes/kubernetes/pull/137248), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Cluster Lifecycle, Node and Testing]
- Truncated the watch cache RV metric to 15 digits to ensure precision. ([#137615](https://github.com/kubernetes/kubernetes/pull/137615), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Instrumentation]
- Updated `cri-tools` to `v1.35.0`. ([#135694](https://github.com/kubernetes/kubernetes/pull/135694), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- Updated etcd client library to `v3.6.6`. ([#135331](https://github.com/kubernetes/kubernetes/pull/135331), [@yashsingh74](https://github.com/yashsingh74)) [SIG API Machinery, Auth, Cloud Provider, Etcd, Node and Scheduling]
- Updated etcd client library to `v3.6.7`. ([#136407](https://github.com/kubernetes/kubernetes/pull/136407), [@ivanvc](https://github.com/ivanvc)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]
- Updated etcd images to `v3.6.8`. ([#137107](https://github.com/kubernetes/kubernetes/pull/137107), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Updated kube-dns to `v1.26.7`. ([#134394](https://github.com/kubernetes/kubernetes/pull/134394), [@toredash](https://github.com/toredash)) [SIG Cloud Provider]
- Updated kustomize dependency to `v5.8.1`. ([#136892](https://github.com/kubernetes/kubernetes/pull/136892), [@koba1t](https://github.com/koba1t)) [SIG Architecture and CLI]

## Dependencies

### Added
- buf.build/go/protovalidate: [v0.12.0](https://github.com/bufbuild/protovalidate-go/commit/8fa707802a9ac7e6fa9860b23c0b09b76344cf29)
- github.com/cenkalti/backoff/v5: [v5.0.3](https://github.com/cenkalti/backoff/commit/7cad66a637c4ffff09d0795608116ddcc7eb1769)
- github.com/moby/moby/api: [v1.52.0](https://github.com/moby/moby/commit/1408c9ca4f4f0717b2d885fc87ae0ff000a91c40)
- github.com/moby/moby/client: [v0.2.1](https://github.com/moby/moby/commit/b6f067c0cfe38401c7fa7678efbbe383ebe134fb)
- go.opentelemetry.io/otel/exporters/stdout/stdouttrace: [v1.40.0](https://github.com/open-telemetry/opentelemetry-go/commit/a3a5317c5caed1656fb5b301b66dfeb3c4c944e0)
- gonum.org/v1/gonum: [v0.16.0](https://github.com/gonum/gonum/commit/7826ba4358a76646a8bd1f852201c4fed2c81c38)

### Changed
- buf.build/gen/go/bufbuild/protovalidate/protocolbuffers/go: 63bb56e → 8976f5b
- cel.dev/expr: [v0.24.0 → v0.25.1](https://github.com/google/cel-spec/compare/v0.24.0...v0.25.1)
- cloud.google.com/go/compute/metadata: [v0.7.0 → v0.9.0](https://github.com/googleapis/google-cloud-go/compare/compute/metadata/v0.7.0...compute/metadata/v0.9.0)
- cyphar.com/go-pathrs: [v0.2.1 → v0.2.2](https://github.com/cyphar/libpathrs.git/compare/go-pathrs/v0.2.1...go-pathrs/v0.2.2)
- github.com/GoogleCloudPlatform/opentelemetry-operations-go/detectors/gcp: [v1.26.0 → v1.30.0](https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/compare/v1.26.0...v1.30.0)
- github.com/Microsoft/hnslib: [v0.1.1 → v0.1.2](https://github.com/Microsoft/hnslib/compare/v0.1.1...v0.1.2)
- github.com/alecthomas/units: [b94a6e3 → 0f3dac3](https://github.com/alecthomas/units/compare/b94a6e3...0f3dac3)
- github.com/cncf/xds/go: [2f00578 → ee656c7](https://github.com/cncf/xds/compare/2f005788dc42b92dee41c8ad934450dc4746f027...ee656c7534f5d7dc23d44dd611689568f72017a6)
- github.com/containerd/containerd/api: [v1.9.0 → v1.10.0](https://github.com/containerd/containerd/compare/api/v1.9.0...api/v1.10.0)
- github.com/coredns/corefile-migration: [v1.0.29 → v1.0.31](https://github.com/coredns/corefile-migration/compare/v1.0.29...v1.0.31)
- github.com/coreos/go-oidc: [v2.3.0 → v2.5.0](https://github.com/coreos/go-oidc/compare/v2.3.0...v2.5.0)
- github.com/coreos/go-systemd/v22: [v22.5.0 → v22.7.0](https://github.com/coreos/go-systemd/compare/v22.5.0...v22.7.0)
- github.com/cyphar/filepath-securejoin: [v0.6.0 → v0.6.1](https://github.com/cyphar/filepath-securejoin/compare/v0.6.0...v0.6.1)
- github.com/davecgh/go-spew: [v1.1.1 → d8f796a](https://github.com/davecgh/go-spew/compare/v1.1.1...d8f796a)
- github.com/docker/go-connections: [v0.5.0 → v0.6.0](https://github.com/docker/go-connections/compare/v0.5.0...v0.6.0)
- github.com/emicklei/go-restful/v3: [v3.12.2 → v3.13.0](https://github.com/emicklei/go-restful/compare/v3.12.2...v3.13.0)
- github.com/envoyproxy/go-control-plane: [v0.13.4 → v0.14.0](https://github.com/envoyproxy/go-control-plane/compare/v0.13.4...v0.14.0)
- github.com/envoyproxy/go-control-plane/envoy: [v1.32.4 → v1.36.0](https://github.com/envoyproxy/go-control-plane/compare/envoy/v1.32.4...envoy/v1.36.0)
- github.com/envoyproxy/protoc-gen-validate: [v1.2.1 → v1.3.0](https://github.com/envoyproxy/protoc-gen-validate/compare/v1.2.1...v1.3.0)
- github.com/go-jose/go-jose/v4: [v4.0.4 → v4.1.3](https://github.com/go-jose/go-jose/compare/v4.0.4...v4.1.3)
- github.com/godbus/dbus/v5: [v5.1.0 → v5.2.2](https://github.com/godbus/dbus/compare/v5.1.0...v5.2.2)
- github.com/golang-jwt/jwt/v5: [v5.2.2 → v5.3.0](https://github.com/golang-jwt/jwt/compare/v5.2.2...v5.3.0)
- github.com/golang/glog: [v1.2.4 → v1.2.5](https://github.com/golang/glog/compare/v1.2.4...v1.2.5)
- github.com/google/cadvisor: [v0.53.0 → v0.56.2](https://github.com/google/cadvisor/compare/v0.53.0...v0.56.2)
- github.com/google/pprof: [27863c8 → 294ebfa](https://github.com/google/pprof/compare/27863c87afa6df68fb88a574f81b47d6fb7fbf29...294ebfa9ad836ed3d00d43d54ea599339e403110)
- github.com/grpc-ecosystem/go-grpc-middleware/providers/prometheus: [v1.0.1 → v1.1.0](https://github.com/grpc-ecosystem/go-grpc-middleware/compare/providers/prometheus/v1.0.1...providers/prometheus/v1.1.0)
- github.com/grpc-ecosystem/go-grpc-middleware/v2: [v2.3.0 → v2.3.3](https://github.com/grpc-ecosystem/go-grpc-middleware/compare/v2.3.0...v2.3.3)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.26.3 → v2.27.7](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.26.3...v2.27.7)
- github.com/ianlancetaylor/demangle: [bd984b5 → f615e6b](https://github.com/ianlancetaylor/demangle/compare/bd984b5ce465cae93d894caa85838021209b9a57...f615e6bd150ba7c2231ba4be6104d64396a3ea50)
- github.com/moby/spdystream: [v0.5.0 → v0.5.1](https://github.com/moby/spdystream/compare/v0.5.0...v0.5.1)
- github.com/onsi/ginkgo/v2: [v2.27.2 → v2.28.1](https://github.com/onsi/ginkgo/compare/v2.27.2...v2.28.1)
- github.com/onsi/gomega: [v1.38.2 → v1.39.1](https://github.com/onsi/gomega/compare/v1.38.2...v1.39.1)
- github.com/opencontainers/cgroups: [v0.0.3 → v0.0.6](https://github.com/opencontainers/cgroups/compare/v0.0.3...v0.0.6)
- github.com/opencontainers/runc: [v1.3.0 → v1.4.0](https://github.com/opencontainers/runc/compare/v1.3.0...v1.4.0)
- github.com/opencontainers/runtime-spec: [v1.2.1 → v1.3.0](https://github.com/opencontainers/runtime-spec/compare/v1.2.1...v1.3.0)
- github.com/opencontainers/selinux: [v1.13.0 → v1.13.1](https://github.com/opencontainers/selinux/compare/v1.13.0...v1.13.1)
- github.com/pmezard/go-difflib: [v1.0.0 → 5d4384e](https://github.com/pmezard/go-difflib/compare/v1.0.0...5d4384e)
- github.com/prometheus/common: [v0.66.1 → v0.67.5](https://github.com/prometheus/common/compare/v0.66.1...v0.67.5)
- github.com/prometheus/procfs: [v0.16.1 → v0.19.2](https://github.com/prometheus/procfs/compare/v0.16.1...v0.19.2)
- github.com/sergi/go-diff: [v1.2.0 → v1.4.0](https://github.com/sergi/go-diff/compare/v1.2.0...v1.4.0)
- github.com/spf13/cobra: [v1.10.0 → v1.10.2](https://github.com/spf13/cobra/compare/v1.10.0...v1.10.2)
- github.com/spiffe/go-spiffe/v2: [v2.5.0 → v2.6.0](https://github.com/spiffe/go-spiffe/compare/v2.5.0...v2.6.0)
- go.etcd.io/etcd/api/v3: [v3.6.5 → v3.6.8](https://github.com/etcd-io/etcd/compare/api/v3.6.5...api/v3.6.8)
- go.etcd.io/etcd/client/pkg/v3: [v3.6.5 → v3.6.8](https://github.com/etcd-io/etcd/compare/client/pkg/v3.6.5...client/pkg/v3.6.8)
- go.etcd.io/etcd/client/v3: [v3.6.5 → v3.6.8](https://github.com/etcd-io/etcd/compare/client/v3.6.5...client/v3.6.8)
- go.etcd.io/etcd/pkg/v3: [v3.6.5 → v3.6.8](https://github.com/etcd-io/etcd/compare/pkg/v3.6.5...pkg/v3.6.8)
- go.etcd.io/etcd/server/v3: [v3.6.5 → v3.6.8](https://github.com/etcd-io/etcd/compare/server/v3.6.5...server/v3.6.8)
- go.opentelemetry.io/auto/sdk: [v1.1.0 → v1.2.1](https://github.com/open-telemetry/opentelemetry-go-instrumentation/compare/sdk/v1.1.0...sdk/v1.2.1)
- go.opentelemetry.io/contrib/detectors/gcp: [v1.34.0 → v1.39.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/detectors/gcp/v1.34.0...detectors/gcp/v1.39.0)
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: [v0.44.0 → v0.65.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/instrumentation/github.com/emicklei/go-restful/otelrestful/v0.44.0...instrumentation/github.com/emicklei/go-restful/otelrestful/v0.65.0)
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: [v0.60.0 → v0.65.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/instrumentation/google.golang.org/grpc/otelgrpc/v0.60.0...instrumentation/google.golang.org/grpc/otelgrpc/v0.65.0)
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: [v0.61.0 → v0.65.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/instrumentation/net/http/otelhttp/v0.61.0...instrumentation/net/http/otelhttp/v0.65.0)
- go.opentelemetry.io/contrib/propagators/b3: [v1.19.0 → v1.40.0](https://github.com/open-telemetry/opentelemetry-go-contrib/compare/propagators/b3/v1.19.0...propagators/b3/v1.40.0)
- go.opentelemetry.io/otel: [v1.36.0 → v1.41.0](https://github.com/open-telemetry/opentelemetry-go/compare/a85ae98dcedc0761078518a715dea53e519b4846...v1.41.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: [v1.34.0 → v1.40.0](https://github.com/open-telemetry/opentelemetry-go/compare/exporters/otlp/otlptrace/v1.34.0...exporters/otlp/otlptrace/v1.40.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: [v1.34.0 → v1.40.0](https://github.com/open-telemetry/opentelemetry-go/compare/exporters/otlp/otlptrace/otlptracegrpc/v1.34.0...exporters/otlp/otlptrace/otlptracegrpc/v1.40.0)
- go.opentelemetry.io/otel/metric: [v1.36.0 → v1.41.0](https://github.com/open-telemetry/opentelemetry-go/compare/metric/v1.36.0...metric/v1.41.0)
- go.opentelemetry.io/otel/sdk: [v1.36.0 → v1.40.0](https://github.com/open-telemetry/opentelemetry-go/compare/sdk/v1.36.0...sdk/v1.40.0)
- go.opentelemetry.io/otel/sdk/metric: [v1.36.0 → v1.40.0](https://github.com/open-telemetry/opentelemetry-go/compare/sdk/metric/v1.36.0...sdk/metric/v1.40.0)
- go.opentelemetry.io/otel/trace: [v1.36.0 → v1.41.0](https://github.com/open-telemetry/opentelemetry-go/compare/trace/v1.36.0...trace/v1.41.0)
- go.opentelemetry.io/proto/otlp: [v1.5.0 → v1.9.0](https://github.com/open-telemetry/opentelemetry-proto-go/compare/otlp/v1.5.0...otlp/v1.9.0)
- go.uber.org/zap: [v1.27.0 → v1.27.1](https://github.com/uber-go/zap/compare/v1.27.0...v1.27.1)
- golang.org/x/crypto: [v0.45.0 → v0.47.0](https://go.googlesource.com/crypto/+/4e0068c0098be10d7025c99ab7c50ce454c1f0f9^1..506e022208b864bc3c9c4a416fe56be75d10ad24/)
- golang.org/x/exp: [8a7402a → 944ab1f](https://go.googlesource.com/exp/+/8a7402abbf56ed11a2540c1d8beb569bd29e22d1^1..944ab1f22d936eefb8f6260ecd2053101d8d7b2a/)
- golang.org/x/mod: [v0.29.0 → v0.32.0](https://go.googlesource.com/mod/+/bba3e065a67271df90253c78c98f2cea7f572948^1..4c04067938546e62fc0572259a68a6912726bcdd/)
- golang.org/x/net: [v0.47.0 → v0.49.0](https://go.googlesource.com/net/+/9a296438e54dff851a45667aa645a97003b44db5^1..d977772e17ccaa1903b2af736f6405ab3a9f05cc/)
- golang.org/x/oauth2: [v0.30.0 → v0.34.0](https://go.googlesource.com/oauth2/+/cf1431934151b3a93e0b3286eb6798ca08ea3770^1..acc38155b7f6f36aefcb58faff6f36d314dd915c/)
- golang.org/x/sync: [v0.18.0 → v0.19.0](https://go.googlesource.com/sync/+/1966f539bbd7664efd5bb7462ae94d9db67f4502^1..2a180e22fddcc336475e72aa950be958c1b68d33/)
- golang.org/x/sys: [v0.38.0 → v0.40.0](https://go.googlesource.com/sys/+/15129aafc3056028aa2694528ac20373f8cd34e4^1..2f442297556c884f9b52fc6ef7280083f4d65023/)
- golang.org/x/telemetry: [078029d → bd525da](https://go.googlesource.com/telemetry/+/078029d740a8681fc932f4e25e60ff1a037f5eca^1..bd525da824e2505db9e8ac44025316bf6f43a6f6/)
- golang.org/x/term: [v0.37.0 → v0.39.0](https://go.googlesource.com/term/+/1231d5465be98a7c5f01140358c142d365d4fbb6^1..a7e5b0437ffa3159709172efbe396bc546550e23/)
- golang.org/x/text: [v0.31.0 → v0.33.0](https://go.googlesource.com/text/+/e7ff6b3572e1a83c072ef150c985f86603986e1b^1..536231a9abc69feaab8d726b5ec75ee8d3620829/)
- golang.org/x/time: [v0.9.0 → v0.14.0](https://go.googlesource.com/time/+/1ce61fe87e0e5dd90752d2b6c5972f9b6918e77c^1..2b4e43900c03fd6b77109b7b2b6d77583f48bc1c/)
- golang.org/x/tools: [v0.38.0 → v0.41.0](https://go.googlesource.com/tools/+/a22b5e8a9b8d2234e1e960ec2473e4011f012a6b^1..2ad2b30edf98d0e3b67a7b3e8f6d1d6e41c963c3/)
- google.golang.org/genproto/googleapis/api: [a0af3ef → 8636f87](https://github.com/googleapis/go-genproto/compare/a0af3efb3deb0aa5253d43c55b96e303c64cc06b...8636f8732409467ddc8453f81f4429397739bb17)
- google.golang.org/genproto/googleapis/rpc: [200df99 → 8636f87](https://github.com/googleapis/go-genproto/compare/200df99c418ae1eac9aa6d0268db9c22c1715c0c...8636f8732409467ddc8453f81f4429397739bb17)
- google.golang.org/grpc: [v1.72.2 → v1.79.3](https://github.com/grpc/grpc-go/compare/v1.72.2...v1.79.3)
- google.golang.org/protobuf: [v1.36.8 → f2248ac](https://go.googlesource.com/protobuf/+/0833cf304e6344e895e819f769afa28107fe8892^1..f2248ac996afc39b3df0777cdcc269f6ade50b07/)
- k8s.io/klog/v2: [v2.130.1 → v2.140.0](https://github.com/kubernetes/klog/compare/v2.130.1...main)
- k8s.io/kube-openapi: [589584f → 43fb72c](https://github.com/kubernetes/kube-openapi/compare/589584f1c912f4367fe8954f649a59a98b912da5...43fb72c5454a03ed83388cf20c070499ee359af8)
- k8s.io/utils: [bc988d5 → b8788ab](https://github.com/kubernetes/utils/compare/bc988d571ff40eb17793769e9c1b71ecf8ee9c0f...b8788abfbbc27cab6c8732274b5c2ae213868854)
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: [v0.31.2 → v0.34.0](https://github.com/kubernetes-sigs/apiserver-network-proxy/compare/konnectivity-client/v0.31.2...konnectivity-client/v0.34.0)
- sigs.k8s.io/knftables: [v0.0.17 → v0.0.21](https://github.com/kubernetes-sigs/knftables/compare/v0.0.17...v0.0.21)
- sigs.k8s.io/kustomize/api: [v0.20.1 → v0.21.1](https://github.com/kubernetes-sigs/kustomize/compare/api/v0.20.1...api/v0.21.1)
- sigs.k8s.io/kustomize/cmd/config: [v0.20.1 → v0.21.1](https://github.com/kubernetes-sigs/kustomize/compare/cmd/config/v0.20.1...cmd/config/v0.21.1)
- sigs.k8s.io/kustomize/kustomize/v5: [v5.7.1 → v5.8.1](https://github.com/kubernetes-sigs/kustomize/compare/kustomize/v5.7.1...kustomize/v5.8.1)
- sigs.k8s.io/kustomize/kyaml: [v0.20.1 → v0.21.1](https://github.com/kubernetes-sigs/kustomize/compare/kyaml/v0.20.1...kyaml/v0.21.1)
- sigs.k8s.io/structured-merge-diff/v6: [v6.3.0 → v6.3.2](https://github.com/kubernetes-sigs/structured-merge-diff/compare/v6.3.0...v6.3.2)

### Removed
- github.com/armon/circbuf: [5111143](https://github.com/armon/circbuf/tree/5111143)
- github.com/bufbuild/protovalidate-go: [v0.9.1](https://github.com/bufbuild/protovalidate-go/commit/107b51bb93ef18ea769ece164b12e844315ae6b1)
- github.com/docker/docker: [v28.2.2](https://github.com/docker/docker/commit/45873be4ae3f5488c9498b3d9f17deaddaf609f4)
- github.com/gregjones/httpcache: [901d907](https://github.com/gregjones/httpcache/tree/901d907)
- github.com/grpc-ecosystem/go-grpc-prometheus: [v1.2.0](https://github.com/grpc-ecosystem/go-grpc-prometheus/tree/v1.2.0)
- github.com/karrick/godirwalk: [v1.17.0](https://github.com/karrick/godirwalk/tree/v1.17.0)
- github.com/libopenstorage/openstorage: [v1.0.0](https://github.com/libopenstorage/openstorage/tree/v1.0.0)
- github.com/moby/sys/atomicwriter: [v0.1.0](https://github.com/moby/sys/commit/4a75548218baa36bdbaaed1371a3e8a9cdfcffa0)
- github.com/mohae/deepcopy: [c48cc78](https://github.com/mohae/deepcopy/tree/c48cc78)
- github.com/morikuni/aec: [v1.0.0](https://github.com/morikuni/aec/tree/v1.0.0)
- github.com/mrunalp/fileutils: [v0.5.1](https://github.com/mrunalp/fileutils/commit/7363e975f9cfb558be601bece0df81714c3c9084)
- github.com/pkg/errors: [v0.9.1](https://github.com/pkg/errors/tree/v0.9.1)
- github.com/zeebo/errs: [v1.4.0](https://github.com/zeebo/errs/commit/13450ab50383c5857164d564544fa0bef3ff689d)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp: [v1.27.0](https://github.com/open-telemetry/opentelemetry-go/commit/5661ff0ded32cf1b83f1147dae96ca403c198504)
- go.uber.org/automaxprocs: [v1.6.0](https://github.com/uber-go/automaxprocs/commit/1ea14c35ce47a73089b824e504d1c92eeb61a5a6)
- gotest.tools/v3: v3.0.2



# v1.36.0-rc.1


## Downloads for v1.36.0-rc.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes.tar.gz) | c134ae3f6d7b8276d2f2020ea0890a70f1ce136ee9460f4b12fa6719f6f4e8909fe9fbf85a52b4b87f6d2032d7044d57f1928f48060f251dbba644deb6649a1f
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-src.tar.gz) | 627e4211a1578935ad9635c2400f302ac2a08e56d8cdb0427dd2fa016f4a45f2d58aeaa0c1b089deb9abb2c3036d29b6ee919ada57a1606fabef8c513c618717

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | 876b2a5db62a21aa525e3b9766c9ad5a89b0b718fc36f35a0aa976de8d0890e482bf1cc9e07b805544d8d20aff6ef3869b5492afb877cac87f35cea169cf817f
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-darwin-arm64.tar.gz) | bd9f7e60992fd4508ae74722cfe7a7aa8e9805c46ed0dec6b4347c0c278eb97e672c4fb894007af9c59673514a9b0645e17196c492edaafb072381697ea45ac9
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-linux-386.tar.gz) | cef8f4aa41dd953b9a6065b1e761c59b20d35cdf13a95a2bcffc6ae9fac77c14b833ee054429ade778c8032dce372ff4234c69cb463b072a4494b628e25400a1
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | 0c23410d581b626d4f36dd291101ce81b64d15da95692693b16b05ffffb42b0f77c8e85b32631fa19e3c9d2109e267d775740f0e50dc29694093ef32be3bada1
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-linux-arm.tar.gz) | 02b27b095c060baa4034588353c4f777eb4d0defc6a536ebe62e9a72a93987ecb5cbbe9181dc00a5c6fbeac7232e7770cf04655a46376b9b63bb5229eeeb63fd
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | 6777fc534be81b04deaf37051ff82359e894e71c2ec97656850ac1973375f1ec2d61e538e574b9d95d1548fa4c2e9e584fd0225f79930caba9f7d981ae0f8026
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | 661dc1a997a9eae839b9f7835d062115b327590dd241ceb72bf32fe56e6bbc82614ff7b100f72056a694e2f4fd6a86d4396fc65a43858aafabbea825f925f274
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | b9b1c9e6cc9b6d1778ab5ac290c5541dfd57877d8fed80f6a7a3b01108cb8296e038908fb881801d6cd5204906ee68672cf6ea2d4766e64034f5615dc3fd116b
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-windows-386.tar.gz) | 9f0c3d7f7057b0c61075db496041b92808e9039f9fbbbf4e8acb90c2e535700d1bdc11bdd78effd3a27b9b2593998a40016e7bd806311ff72256cc3d5e4c759c
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | 9d62b1d9e3c0c869907fb10cb850cd4ef855570f3991c08e779369fe4c732e06c36dafa2bfea0589bac4852308a74f9f4f4de495d30ccf80ea4b886538a4541f
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-client-windows-arm64.tar.gz) | e4f8a2dc09c59132cca3ffe96c9da2d2e862343382fbcb64cbb77b61e0ba02898b1f78b546a833c081307d0959190b39991b8b898313bb7ee3a7bcce65aea8de

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | 2a746d8485428451ea98b4a6bc87caff3d5a16380081b5dca8e4a945b398c415770b2ade78e9c356d4b1d8289b8eb04d155fd75c8a22bfd8e4a8eb8b68243174
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | 0f8e304d016ec0b976a05df060ea6e6c750b2c0d0990f5572a2e2b285015a08aa4892ba1046822d34d5620f84b73d655ca32e054eb05d78a47c13f567d7a643f
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | d0b58696236107a80cd2d907316ce771d626dc7c3427b8d7f61191d506f68aa7bd0cd47aa2708f6e229bf8722a834bf4af9b1ed886e972e466bd306dba3f259f
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | 4bced69b2ff0b8231b9085729d8da1f59e3311c0013eee67ae742da79c9d8cff0921fca1f4fb08d9ed2d2d78b059b026ebddac7f36471b244d9dfdd952018c72

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | 5adae9badcb1bc7dda5bc75a7fc41ead727e62a9744c5545a4c43a8e2e77127720f83212ad0b7a244c7c7f9b82abb2e2e3c2c8e18a4739320591ec1c388eb7d4
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | cba5683285a0e74830d4334e35b1fa6495b553a03917af40c795157b57a96a7165e502b3dfac37d8cd619619c871eac1055c501615c2e762fbb7c1d9a824dae0
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | 1352a58e7e1aff9809cb1f875058ed9a1f5b80a8aeaa4c5b38cc91316c1fe0c3846c3868d3b8c70912a3a736d433197ca76a6a656f4b58ecc34ef8a773843042
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | be83680c9c73bf2c9c2f90d27fe7efd9eb26074ebf4004391bb0650125c354bcc71c42b5b964ed1d725d34a7be60d3a676d9fac5efedd6fc157653a2c6cd32b5
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | 45fdaa8d3020589e197ea37532c2428fecb71b0c4a7661d627b24ee52cacf9323b51387188e848f85f25f630b97ed86f0318b5708adcddfd5a3cdaa9e46e0908

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.36.0-rc.0

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/moby/spdystream: [v0.5.0 → v0.5.1](https://github.com/moby/spdystream/compare/v0.5.0...v0.5.1)
- go.opentelemetry.io/otel: [v1.40.0 → v1.41.0](https://github.com/open-telemetry/opentelemetry-go/compare/v1.40.0...v1.41.0)
- go.opentelemetry.io/otel/metric: [v1.40.0 → v1.41.0](https://github.com/open-telemetry/opentelemetry-go/compare/metric/v1.40.0...metric/v1.41.0)
- go.opentelemetry.io/otel/trace: [v1.40.0 → v1.41.0](https://github.com/open-telemetry/opentelemetry-go/compare/trace/v1.40.0...trace/v1.41.0)

### Removed
_Nothing has changed._



# v1.36.0-rc.0


## Downloads for v1.36.0-rc.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes.tar.gz) | 5b28aae440bdb013eb1497bc357a4b27eb51d275b6da72a9f8aec845169ecbc47ed8c82f091b3b62ede51765e587e652040e39f78da4dd5f560be88f3c3cc52b
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-src.tar.gz) | 84e1f0f1e3e57e4b19cb0971d0d980b47cbd2b512e0a0a1477bf88f9d0ff4b786a714e391276fc7b6c4ec119fb8311272b4838644be0bb54c898d31c6640e8b5

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | d7f5c47f327bee440c506a37876538eb232900b1d01c922ca3e3f3ee5a7cf73a35e1c9bcf0c045f9341ee41c04bdac1728f691b470c0ce6302704000347bbad1
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | fac20e24b8f9f4fba2c22a6acc193d2fc561542233c7707328a87a1ddbf0d2a476503487c055a21b3661df4cda7e96e271734192c65d1d8badd275cfabbb6ec8
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-linux-386.tar.gz) | 64edce3331e69b6abb4cb6b0668125c511ee82e2e98978af44e541a3218d09d832f44cec05ea22fc8fd141587b15915153cd92f4664b6da6b2f56891f6b9af3e
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | e15cea3fe11262d903b549c26c36936232b08cfa231573a1c7aeaa64553ff2099534bae8e92c99a304ab7edc945e692f95da77ed68b4e4896a99377d00bcde66
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-linux-arm.tar.gz) | eb93e29a907f19f4fabd77f0268c2ebb278a332aa182b9e8e3d56b9fff2a28442dee888689f87ca25b3111ee7dc87d9961d5bd0d1867d812edbeff867380996d
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | 3d184176e56da806368a36b2c8ee33e7f695970498132ed39ddf62d0733440bf1198833441ec05923b1b7d0d50da01896a5253e6be38f950775eb6e82022302f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | e985ea5dd3a5d61bcb60e209d7e520c6102a0c7117e23c17ad5dc1a34538a5c73cd5e1abe335aba122e3e6abebaa98b4a1d98912f0f9e63efa9f824594ba54db
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | a44617572f6099efab0133cc2c0b4e9dcfedca6e9d4381036ab2b2a58909250b2062d14f2ef1c06fd814ecf1f6979784f9f5303983865ce490817d8f88abe021
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-windows-386.tar.gz) | da1df34ff592947383f02c34df15ad2d68c09eb43de670eb6c59525ce2e35dd24578645e8c114314c19fc3c3f1fe6dc05d141db513b3326a3d76d08b23f92e74
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | 37e27266d6a671c0d5dd2add6e3e441715f627419f4765a055020aa19d303997d5ae05da56b88a3a7de5b2d85be4b5b49ee776ad16349f458a94eabe3d4dca38
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | 40f0e5a3fce2e5650d7a00ca1e55f49f1ac00c0a1fd91f79719102f0eac4bfb164f3deb94d3b6ac39c7f2596bdab8c95d43b76a68b81b07d9e000e6a3bf9532e

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | 134342bbe78abb9b02ae3ebe10b9e5de72d2ae9084f33c767d0045c04e740f5c50c40fb9bba00ac2afcb96afcd507b8745c26ec02a40ab0b6a2ca938b9797577
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | 51a034a8e544db735f803237d6e363f156b36793fea613029c6db005d9fede5147f623f03732b3bf55d8674cd5faf2f6c675f2f0998c1fbb836548fb581402a5
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | 5e9a38e2fd29e23240dd4f82f94504258310c5262b59d2b7dd24ddde6e61282042ac7727fdde7fd7c137ca3a63a7a6b8ae38278a839dc825129dc53069d82ceb
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | e70e9b0cf26a81c5d8030609f8d3aa91e569353d688fc74f6f63244453651f2f38139af92880b68393cc110c70d25fd0f686e06d55c6873bbb4e0f572dd964c2

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | a716ef2572eedde30999ec7ab1a720dc749afa7c4440457b846f868634f1faa6d36b88d80838575cf41bd5eac0b229c2606f32898347bb56342a7ff854dd6852
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | 69ae5167206cc88b125a8f53e618e4f452130a940ea5fbd1e79bf0c6ce82a624a468529cb4ac4d56cc644826fea3bf13a42c99cc43d54fd42acfa98eb04e0897
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | d9f292776a53c2db0d7a518eb6b3844d2eca971528e04ee30ac17340d7748eca9a0aa57edceef15267d254dd8a04fcc2a63485f8fda8610cb90a6c53b88779bd
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | 3f564f9340a52714669e73cff2c5b7a86971bd252ee095e0739e71792d42577d76a4df2fe669f18377cc5ed3a7dc577439af54e4d0033d2d5539978730a85ded
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | b00af83413f84e2f585a74aa217af4deadf39b792479cc6e47b9b4e7ae1e2d4975c094d385d405a6a47de602153a7df7be9fa663d4734f3fdb6fb513b76a5c26

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.36.0-beta.0

## Changes by Kind

### API Change

- Add ResourcePoolStatusRequest API (v1alpha1) for querying DRA resource pool availability. This enables users and external schedulers to discover available devices across pools before submitting workloads. Requires the DRAResourcePoolStatus feature gate (alpha). ([#137028](https://github.com/kubernetes/kubernetes/pull/137028), [@nmn3m](https://github.com/nmn3m)) [SIG API Machinery, Apps, Auth, Etcd, Instrumentation, Node, Scheduling, Storage and Testing]
- Added `DisruptionMode`, `PriorityClassName` and `Priority` fields to Workload and PodGroup APIs to support workload-aware preemption when `WorkloadAwarePreemption` feature gate is enabled. ([#136589](https://github.com/kubernetes/kubernetes/pull/136589), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- DRA (Dynamic Resource Allocation) drivers and controllers now require granular RBAC permissions to update ResourceClaim statuses when the `DRAResourceClaimGranularStatusAuthorization` feature gate is enabled (Beta in 1.36). Schedulers and controllers must be granted `update`/`patch` on `resourceclaims/binding`. DRA drivers must be granted `associated-node:update` or `arbitrary-node:update` (or patch equivalents) on `resourceclaims/driver`, restricted by their specific `resourceNames`. ([#134947](https://github.com/kubernetes/kubernetes/pull/134947), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Auth, Instrumentation, Node, Scheduling and Testing]
- DRA: PodGroup resources can now make requests with ResourceClaims through a `spec.resourceClaims` field which can refer to ResourceClaims and ResourceClaimTemplates. Claims made by a PodGroup are reserved for the entire PodGroup instead of individual Pods, allowing more than 256 Pods to share a single ResourceClaim. ResourceClaimTemplates referenced by a PodGroup's claim will replicate into a ResourceClaim specific to that PodGroup able to be shared by all of the group's Pods. ([#136989](https://github.com/kubernetes/kubernetes/pull/136989), [@nojnhuh](https://github.com/nojnhuh)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Graduate InPlacePodLevelResourcesVerticalScaling feature to beta and have it on by default. This feature allows resizing the CPU and memory resources at pod-level for pods with pod-level resources set and enabled. ([#137684](https://github.com/kubernetes/kubernetes/pull/137684), [@ndixita](https://github.com/ndixita)) [SIG API Machinery, Apps, Autoscaling, Node, Release, Scheduling and Testing]
- Promote NodeLogQuery to GA. ([#137544](https://github.com/kubernetes/kubernetes/pull/137544), [@jrvaldes](https://github.com/jrvaldes)) [SIG Node and Windows]
- [Alpha] Introduce List Types for Attributes in DRA (KEP-5491). 
  
  The `DRAListTypeAttributes` feature gate(false by default) can activate below enhancements.
  
  For DRA drivers, it can enable list-type fields(`bools/ints/strings/versions`) for device attributes in `ResourceSlice`. Please remember that the number of attribute values, including scalars and lists, per single device is limited to 48. 
  
  For DRA users, this feature enhances the semantics of `matchAttribute`/`distinctAttribute` constraint in `ResourceClaim` to work on both scalar and list attributes. The `matchAttribute` constraint now matches when the intersection (as a set) of all the list values among candidate devices is non-empty. The `distinctAttribute` constraint, which is behind the `ConsumableCapacity` feature gate, matches when all the list values (as a set) among candidate devices are pairwise disjoint. In both constraints, scalar values are implicitly treated as a singleton set. 
  
  And, a new CEL function `.includes`　is introduced. The function can work on both scalar and list attributes to test inclusion(e.g., `device.attributes["dra.example.com"].model.includes("model-a")`). This can support smooth migration for CEL expression in DRA resources when a DRA driver changes the attribute value type from scalar to list, or vice versa. ([#137190](https://github.com/kubernetes/kubernetes/pull/137190), [@everpeace](https://github.com/everpeace)) [SIG API Machinery, Node, Scheduling and Testing]

### Feature

- Adds the `UserNamespacesHostNetwork` runtime handler and integrates the `UserNamespacesHostNetworkSupport` feature gate with the `NodeDeclaredFeatures` feature gate. The `UserNamespacesHostNetworkSupport` feature gate only takes effect when the container runtime's `UserNamespacesHostNetwork` runtime handler returns true and the `NodeDeclaredFeatures` feature gate is enabled. ([#135828](https://github.com/kubernetes/kubernetes/pull/135828), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Autoscaling, Node, Scheduling and Testing]
- When WorkloadAwarePreemption Feature Gate is enabled, and the Pod Group scheduling fails to find a place for the Pod Group, instead of running default preemption for each pod from the pod group, the workload aware preemption will be run for the whole group. ([#137606](https://github.com/kubernetes/kubernetes/pull/137606), [@Argh4k](https://github.com/Argh4k)) [SIG Apps, Node, Scheduling, Storage and Testing]

### Bug or Regression

- Fix erroneously reporting a pod-level resize in progress on pod creation when InPlacePodLevelResourcesVerticalScaling is enabled. ([#138049](https://github.com/kubernetes/kubernetes/pull/138049), [@ndixita](https://github.com/ndixita)) [SIG Node and Testing]
- Fixed kubelet to preserve DRA NodeAllocatableResourceClaimStatuses in Pod.Status. ([#138030](https://github.com/kubernetes/kubernetes/pull/138030), [@askervin](https://github.com/askervin)) [SIG Node]
- Fixes a 1.35 regression in StatefulSet Parallel pod management by disabling the MaxUnavailableStatefulSet feature by default. ([#137904](https://github.com/kubernetes/kubernetes/pull/137904), [@soltysh](https://github.com/soltysh)) [SIG Apps]
- Kep-5304: bump cdi spec for discoverable metadata to 0.5.0 ([#138035](https://github.com/kubernetes/kubernetes/pull/138035), [@alaypatel07](https://github.com/alaypatel07)) [SIG Node]
- Update the version of the pause image to 3.10.2. Track all pause image version dependencies under the 'registry.k8s.io/pause' item in build/dependencies.yaml. ([#138199](https://github.com/kubernetes/kubernetes/pull/138199), [@neolit123](https://github.com/neolit123)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Scheduling and Testing]

### Other (Cleanup or Flake)

- Golang version has been updated to 1.26.2 ([#138261](https://github.com/kubernetes/kubernetes/pull/138261), [@dims](https://github.com/dims)) [SIG Architecture and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.36.0-beta.0


## Downloads for v1.36.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes.tar.gz) | 7e4c5eb75fdcbddb19c94139ead1b7d16bbc9332f319006312441db15cf2d5562625a42cc1a14aad93338a1e48e6c00e2abfe23071626d2a80959ce71c82fc07
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-src.tar.gz) | a97a579bf0b56b408908d1ab58bb75c821e05bb26b301afb223302314a569ebed865b019118ece4a94e598564d21d36ad1f452ea5f64837e3ebbca67b11c1201

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | 942e91cc2e59e2b3c6d14d0e3547c349b5fdf573f76861baa78a650871bbddf80c5503b8ab42a7c34481fd5750781722274aae1210cddbb82cccc5ebb38c96f1
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | 3d985d2d528017b806cd6073ff4dee53faf3ee8b6a25c994287b82a4b09be6ed0e0ab122c4659b08d61fb0a0603bad00dd6c34f5d08f7d3a9fa7ed810619efd0
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-linux-386.tar.gz) | 4631d3edc7dc0fe4dc2cfa6b0b520d167d372b60ea386dcea8c2b59b2e4c37bcf8fff93ef77813a5842dac7d45926efdc78a69fcaafe8aa488dab2d36eb36f37
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | bc41228fd7e01c94c88b6f6798ae4a37834098880bc00109a5e54bd1138a9b827804426003dfedf92d418c8a1b79bc39443518b1cefbcac616c211528287e26b
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-linux-arm.tar.gz) | b89b8174eba4fc43560c35f37abb599b326a38963bdf1a4925c1eacbf1090c859ce6fb664475ab953859ea3c9f7e1d7bceaeadb2a9f42d2772efba7dc521917c
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | 4693d9992d1b2148e6cd93eafba1ef03a73a72f1b81c610be6f6597225bc31a7fe60775f04c1b89b7dc669eba16087762b14c21ae7d8fd405e1a9a24bd2b838b
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | 166ab0173f7e80f4d584fe2cf61693b3bb0896840d587e7c2b5fd670db5498f8facfefd2ea49c91c50ab53304c9bc430fa29e87420abb552355e1ed9068c7e95
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | 0b3c66b59cd4935ff56b493a3aae0ddf441f89ab4e52a45edbfbd0cd882af317ddf85e40cb914681fbb89a55ad3707884d0c3046f8df2a9045702ff62df06bf2
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-windows-386.tar.gz) | cb4083fed4d5e3140f7dd5730e7dabf681ea11999f0b6e946a6f840380355257f849b35600f29e2d87fc056036e41827fe5596991882eedc4fd9a5c3addf8d97
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | 9b858b63cea22adfe924f5a583ed398c0c613a436c4f9db625d13ca52f1dd79c49daf7b6d981c2be1d90218385194822908dfdf70e41bc55b58c5ff8c6a32e00
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | a7982d04ee2c2a804e575c20121036ac5fbfcb82f1c473525635ad7bb816055ec6f223b4c7612b7b87d185816368d2e690a1c3ed47336cc2eebbf6372f0576de

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | 826ce2b809ae92eddacdb7c1dc75108024112457a0c1d48a23809d282707d13292d5fa40502ef7b6f44a9d68821eb556f59af05498a47427fd337634018ed576
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | 3f70adbb9b1389670b0d77c1caab3b2b83253caa90e4641fc4604a0d559f286a1b083d0111db7c0bc057e25f6184f2b1fed9a206ad21c29b7c0a03869c8cbaf2
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | a092665ee24cdda2800f31be942704340bdd541590d7d752e0ba847fadbcf6eb047954f890476f7b926c49984b3d9cb1ef675eedb76163d3b155ab4f5430fc3d
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | acfb69d1a089bd2bd9250d37ebc705eaf80a3b831b80756cabe90f9a8d92816344a12859ec49e6f6a74e6ec0b2a39d382be7f3061f0ffad3f9d1fdb82e25d5f0

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | cb8e8ec77f7e4ea9e8db50cb41339b1c1026b3565b0a47a6873156cec58e495897b0a767885838d9f085c1f6b69d40eb2e3b1babc5f9c85c235a13ddd9399670
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | cea68a83e7b202e273721be4fd708fcd45d4d3e06f1d8b6da5d6282672e11282d0cb162a76595ae4b95381a27afd3eebfe15595b18539dac720658f67e9ab549
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | f1e73ab63bef0a79f8c04a4e123f99d6fcc5efd576766bb84dde9988979a87a7d4aba18e29e3696396cd1fca2281345b353fcd470cfbdfae27a122bdb9270ecd
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | 61f255a0e29fb87b10bf3aa0cd0d48921cf3491662acf39ed455e8322d76034129808a6855a95e451a69960991ee3641d11a1a487bc897b0f0097c4d5eb9bce6
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 8e30f607ba4c196ae4fb422fa632d497d940bb82c34f3aef4ee13477070cf932119eb2968065d571bbd1a714f77ca6a6a238013fabfb6f4d580921553fd8abab

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.36.0-alpha.2

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Kube-controller-manager: renamed metric 'volume_operation_total_errors' to 'volume_operation_errors_total'. If you are using custom monitoring dashboards or alerting rules based on the 'volume_operation_total_errors' metric, please update them to use the new 'volume_operation_errors_total' metric. ([#136399](https://github.com/kubernetes/kubernetes/pull/136399), [@tico88612](https://github.com/tico88612)) [SIG Apps, Instrumentation, Storage and Testing]
 
## Changes by Kind

### Deprecation

- Add warnings and deprecation for Service.spec.externalIPs ([#137293](https://github.com/kubernetes/kubernetes/pull/137293), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Apps, Network and Windows]
- Direct access to the `Raw` field of `metav1.FieldsV1` is deprecated. Code that constructs or reads `FieldsV1` should migrate to the new `NewFieldsV1(string)`, `GetRawBytes()`, `GetRawString()`, and `SetRawBytes()` accessor methods. ([#137304](https://github.com/kubernetes/kubernetes/pull/137304), [@aaron-prindle](https://github.com/aaron-prindle)) [SIG API Machinery, Apps and Testing]
- Rename `AllowlistEntry.Name` to `AllowlistEntry.Command` in the credential plugin allowlist ([#137272](https://github.com/kubernetes/kubernetes/pull/137272), [@pmengelbert](https://github.com/pmengelbert)) [SIG API Machinery, Auth, CLI and Testing]

### API Change

- A few log calls which did not properly format their parameters were fixed. ([#137108](https://github.com/kubernetes/kubernetes/pull/137108), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Cluster Lifecycle, Network, Node, Scheduling and Testing]
- Add --tls-curve-preferences flag for configuring TLS key exchange mechanism ([#137115](https://github.com/kubernetes/kubernetes/pull/137115), [@damdo](https://github.com/damdo)) [SIG API Machinery, Architecture, CLI, Cloud Provider, Node and Testing]
- Add a deletion protection mechanism for PodGroup objects. ([#137641](https://github.com/kubernetes/kubernetes/pull/137641), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Auth, Scheduling and Storage]
- Add admission plugin that validates PodGroup resources reference an existing Workload and match the declared PodGroupTemplate spec. ([#137464](https://github.com/kubernetes/kubernetes/pull/137464), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Add alpha support for manifest-based admission control configuration (KEP-5793). When the `ManifestBasedAdmissionControlConfig` feature gate is enabled, admission webhooks and CEL-based policies can be loaded from static manifest files on disk via the `staticManifestsDir` field in `AdmissionConfiguration`. These policies are active from API server startup, survive etcd unavailability, and can protect API-based admission resources from modification. ([#137346](https://github.com/kubernetes/kubernetes/pull/137346), [@aramase](https://github.com/aramase)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage, Testing and Windows]
- Add opt-in alpha support in kubeletplugin framework for DRA drivers to publish DRA Device metadata in pod CDI mounts ([#137086](https://github.com/kubernetes/kubernetes/pull/137086), [@alaypatel07](https://github.com/alaypatel07)) [SIG Apps, Network, Node and Testing]
- Add tlsServerName field to EgressSelectorConfiguration TLSConfig to allow overriding the server name used for TLS certificate verification ([#136640](https://github.com/kubernetes/kubernetes/pull/136640), [@kennangaibel](https://github.com/kennangaibel)) [SIG API Machinery, Apps, Auth, Storage and Testing]
- Added MemoryReservationPolicy cgroup v2 MemoryQoS support to KubeletConfiguration for memory.min protection. ([#137584](https://github.com/kubernetes/kubernetes/pull/137584), [@QiWang19](https://github.com/QiWang19)) [SIG Node and Storage]
- Added PlacementGenerate extension point to the scheduler. It's used to generate placements for placement-based pod group scheduling. Its use is guarded by the TopologyAwareWorkloadScheduling feature gate. ([#137083](https://github.com/kubernetes/kubernetes/pull/137083), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Added PlacementScore extension point to the scheduler. It's used to score placements in order to choose the best one for placement-based pod group scheduling. Its use is guarded by the TopologyAwareWorkloadScheduling feature gate.
  
  Deprecated MinNodeScore and MaxNodeScore in favor of MinScore and MaxScore. ([#137201](https://github.com/kubernetes/kubernetes/pull/137201), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Added SchedulingConstraints to express TAS constraints for pod group scheduling behind TopologyAwareWorkloadScheduling feature gate.
  
  Added TopologyPlacement plugin implementing PlacementGenerate extension point to take the constraints into consideration during pod group scheduling. The usage of this plugin is guarded by the TopologyAwareWorkloadScheduling feature gate. ([#137271](https://github.com/kubernetes/kubernetes/pull/137271), [@brejman](https://github.com/brejman)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Added TAS logic to the pod group scheduling cycle behind TopologyAwareWorkloadScheduling feature gate. This feature supports scheduling pod groups on nodes with matching topology domains. ([#137489](https://github.com/kubernetes/kubernetes/pull/137489), [@brejman](https://github.com/brejman)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling and Testing]
- Added `PodGroupPodsCount` scheduler plugin to support workload-aware scheduling by prioritizing placements with higher pod counts within a group. ([#137488](https://github.com/kubernetes/kubernetes/pull/137488), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling and Testing]
- Added alpha support (behind `PersistentVolumeClaimUnusedSinceTime` feature gate) for tracking PVC unused status via a new `Unused` condition on PersistentVolumeClaimStatus. When enabled, the PVC protection controller sets `Unused=True` with a `lastTransitionTime` when no non-terminal Pods reference the PVC, enabling external automation to identify and manage unused storage. ([#137862](https://github.com/kubernetes/kubernetes/pull/137862), [@gnufied](https://github.com/gnufied)) [SIG Apps, Auth, Storage and Testing]
- Added placement-based pod group scheduling algorithm to scheduler. Its use is guarded by the TopologyAwareWorkloadScheduling feature gate. ([#136944](https://github.com/kubernetes/kubernetes/pull/136944), [@brejman](https://github.com/brejman)) [SIG Scheduling and Testing]
- Allow users to opt-in to scheduling behaviour for CSI volume ([#137343](https://github.com/kubernetes/kubernetes/pull/137343), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Scheduling and Storage]
- Config.k8s.io.flagz API is graduated to v1beta1 ([#137174](https://github.com/kubernetes/kubernetes/pull/137174), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation, Node, Scheduling and Testing]
- Config.k8s.io.statusz API is graduated to v1beta1 ([#137173](https://github.com/kubernetes/kubernetes/pull/137173), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation, Scheduling and Testing]
- DRA DeviceTaintRules: the TimeAdded of the taint is not only added automatically, it now also gets updated automatically when changing the effect. ([#137167](https://github.com/kubernetes/kubernetes/pull/137167), [@pohly](https://github.com/pohly)) [SIG API Machinery, Node and Testing]
- DRA extended resource feature is promoted to beta in 1.36 ([#135048](https://github.com/kubernetes/kubernetes/pull/135048), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Architecture, Auth, Network, Node, Scheduling and Testing]
- DRA: graduate Device Binding Conditions (KEP #5007) to beta. The feature is now enabled by default in v1.36. ([#137795](https://github.com/kubernetes/kubernetes/pull/137795), [@ttsuuubasa](https://github.com/ttsuuubasa)) [SIG API Machinery, Node, Scheduling and Testing]
- DRA: graduated device taints and tolerations (KEP #5055) to beta. Support for DeviceTaints in ResourceSlices is on by default. Support for DeviceTaintRules depends on enabling resource.k8s.io/v1beta2 and the DeviceTaintRules feature gate. ([#137170](https://github.com/kubernetes/kubernetes/pull/137170), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Cluster Lifecycle, Etcd, Node, Scheduling and Testing]
- DRAConsumableCapacity is enabled by default. ([#136611](https://github.com/kubernetes/kubernetes/pull/136611), [@sunya-ch](https://github.com/sunya-ch)) [SIG API Machinery, Cluster Lifecycle, Node, Scheduling and Testing]
- Extended NodeResourcesFit to implement the PlacementScore extension point. The usage of the PlacementScore extension point is guarded by the TopologyAwareWorkloadScheduling feature gate. ([#136652](https://github.com/kubernetes/kubernetes/pull/136652), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Feature gate UserNamespacesSupport is now GA. ([#136792](https://github.com/kubernetes/kubernetes/pull/136792), [@rata](https://github.com/rata)) [SIG API Machinery, Apps, CLI, Node, Storage and Testing]
- For pod resizes requested on nodes where the resize request exceeds the node's allocatable capacity or the node is running an OS that does not support resize, the request will now fail in admission rather than be marked as Infeasible in the pod status later. ([#136043](https://github.com/kubernetes/kubernetes/pull/136043), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Node, Release, Scheduling, Storage and Testing]
- Graduate metric 'apiserver_storage_events_received_total' to BETA ([#136314](https://github.com/kubernetes/kubernetes/pull/136314), [@petern48](https://github.com/petern48)) [SIG API Machinery, Etcd, Instrumentation and Testing]
- Graduated `ImageVolume` feature to stable. ([#136711](https://github.com/kubernetes/kubernetes/pull/136711), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps, Architecture, Node and Testing]
- HPA: Improved scaling to and from zero with enabled HPAScaleToZero feature gate. ([#135118](https://github.com/kubernetes/kubernetes/pull/135118), [@johanneswuerbach](https://github.com/johanneswuerbach)) [SIG Apps, Autoscaling and Testing]
- Integrate Workload and PodGroup APIs with the Job controllers to support gang-scheduling. ([#137032](https://github.com/kubernetes/kubernetes/pull/137032), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Instrumentation, Node, Scheduling and Testing]
- Introduced scheduling.k8s.io/v1alpha2 Workload and PodGroup API to allow for expressing workload-level scheduling requirements and let kube-scheduler act on those. Removed scheduling.k8s.io/v1alpha1 Workload API. ([#136976](https://github.com/kubernetes/kubernetes/pull/136976), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Etcd, Node, Scheduling, Storage and Testing]
- Kube-scheduler now updates PodGroup status with a `PodGroupScheduled` condition reflecting whether the group was successfully scheduled or is unschedulable. ([#137611](https://github.com/kubernetes/kubernetes/pull/137611), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Scheduling and Testing]
- Promote DRAPrioritizedList to GA ([#136924](https://github.com/kubernetes/kubernetes/pull/136924), [@troychiu](https://github.com/troychiu)) [SIG Apps, Architecture, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Network, Node, Release, Scheduling, Storage and Testing]
- Promote ProcMountType feature to GA ([#137454](https://github.com/kubernetes/kubernetes/pull/137454), [@haircommander](https://github.com/haircommander)) [SIG API Machinery, Apps, Auth, CLI, Node, Storage and Testing]
- Promote `NodeDeclaredFeatures` to beta. ([#136042](https://github.com/kubernetes/kubernetes/pull/136042), [@pravk03](https://github.com/pravk03)) [SIG API Machinery, Apps, Cluster Lifecycle, Instrumentation, Node, Scheduling, Storage and Testing]
- Promoted mutable CSI node allocatable count to GA. The `MutableCSINodeAllocatableCount` feature gate is now locked to enabled. ([#136230](https://github.com/kubernetes/kubernetes/pull/136230), [@torredil](https://github.com/torredil)) [SIG API Machinery and Storage]
- Promoted several endpointslice metrics from Alpha to Beta stability. ([#136368](https://github.com/kubernetes/kubernetes/pull/136368), [@bhope](https://github.com/bhope)) [SIG Instrumentation and Network]
- Promoted several scheduler metrics (`scheduler_goroutines`, `scheduler_permit_wait_duration_seconds`, `scheduler_plugin_evaluation_total`, `scheduler_plugin_execution_duration_seconds`, `scheduler_scheduling_algorithm_duration_seconds`, `scheduler_unschedulable_pods`) from Alpha to Beta stability, providing stronger API and label stability guarantees for metric consumers. ([#136155](https://github.com/kubernetes/kubernetes/pull/136155), [@bhope](https://github.com/bhope)) [SIG Instrumentation and Scheduling]
- Promoted the `DRAAdminAccess` feature gate to GA. ([#137373](https://github.com/kubernetes/kubernetes/pull/137373), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth, Node, Scheduling and Testing]
- Promoted two Job controller metrics from Alpha to Beta stability, providing stronger API and label stability guarantees for metric consumers. ([#136367](https://github.com/kubernetes/kubernetes/pull/136367), [@bhope](https://github.com/bhope)) [SIG Apps and Instrumentation]
- Remove CRD stored versions from status upon SVM migration ([#135297](https://github.com/kubernetes/kubernetes/pull/135297), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth and Testing]
- Renamed metric 'etcd_bookmark_counts' to 'etcd_bookmark_total'. If you are using custom monitoring dashboards or alerting rules based on the 'etcd_bookmark_counts' metric, please update them to use the new 'etcd_bookmark_total' metric. ([#136483](https://github.com/kubernetes/kubernetes/pull/136483), [@petern48](https://github.com/petern48)) [SIG API Machinery, Etcd, Instrumentation and Testing]
- Slow requests that use impersonation can now be tracked via the `apiserver.latency.k8s.io/impersonation` audit event annotation when the ConstrainedImpersonation feature is enabled. ([#137523](https://github.com/kubernetes/kubernetes/pull/137523), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- The /configz endpoint of kubelet, scheduler, cloud controller manager, and kube-proxy serializes the APIVersion and Kind fields as well as using public types instead of internal. ([#136044](https://github.com/kubernetes/kubernetes/pull/136044), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Network, Node, Scheduling and Testing]
- The ConstrainedImpersonation feature is now beta and enabled by default. ([#137609](https://github.com/kubernetes/kubernetes/pull/137609), [@enj](https://github.com/enj)) [SIG API Machinery and Testing]
- The `StrictIPCIDRValidation` feature gate to kube-apiserver is now
  enabled by default, meaning that API fields no longer allow IP or CIDR
  values with extraneous leading "0"s (e.g., `010.000.000.005` rather than
  `10.0.0.5`) or CIDR subnet/mask values with ambiguous semantics (e.g.,
  `192.168.0.5/24` rather than `192.168.0.0/24` or `192.168.0.5/32`). ([#137053](https://github.com/kubernetes/kubernetes/pull/137053), [@danwinship](https://github.com/danwinship)) [SIG Network and Testing]
- This change adds a new alpha feature DRANativeResources, which includes:
   - A new ResourceSlice.Spec.Devices[*].NativeResourceMappings field for DRA drivers to declare how device resources map to native Kubernetes resources (e.g., cpu, memory).
   - Changes in the DynamicResources plugin and the scheduler framework to correctly account for native resources requested through resource claims.
   - Kubelet's admission handler validates if the node can fulfill native resource DRA requests along with standard requests in the pod spec
   ```
  
  #### Additional documentation e.g., KEPs (Kubernetes Enhancement Proposals), usage docs, etc.:
  
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
  --> ([#136725](https://github.com/kubernetes/kubernetes/pull/136725), [@pravk03](https://github.com/pravk03)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- `SnapshotMetadataService` is now available in v1beta1 version. The support for the v1alpha1 version have been removed. ([#137564](https://github.com/kubernetes/kubernetes/pull/137564), [@iPraveenParihar](https://github.com/iPraveenParihar)) [SIG Storage and Testing]

### Feature

- A new gRPC service is added to the Kubelet that provides information about pods running on the node. ([#134627](https://github.com/kubernetes/kubernetes/pull/134627), [@briansonnenberg](https://github.com/briansonnenberg)) [SIG Node and Testing]
- Add alpha metrics tracking the resource version the cache layer of an informer is at. ([#137419](https://github.com/kubernetes/kubernetes/pull/137419), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Add the `timezone` field to the cronjob describe output. ([#136663](https://github.com/kubernetes/kubernetes/pull/136663), [@kfess](https://github.com/kfess)) [SIG CLI]
- Add the ability for statefulset controller to read its own pod and pvc writes ([#137254](https://github.com/kubernetes/kubernetes/pull/137254), [@michaelasp](https://github.com/michaelasp)) [SIG Apps]
- Add tracing for WatchList requests ([#137202](https://github.com/kubernetes/kubernetes/pull/137202), [@serathius](https://github.com/serathius)) [SIG API Machinery and Testing]
- Added ControllerManagerReleaseLeaderElectionLockOnCancel feature gate to gate leader election lock release on exit for kube-controller-manager ([#136279](https://github.com/kubernetes/kubernetes/pull/136279), [@tchap](https://github.com/tchap)) [SIG API Machinery and Cloud Provider]
- Added New RuntimeService streaming RPCs (`StreamPodSandboxes`, `StreamContainers`, `StreamContainerStats`, `StreamPodSandboxStats`, `StreamPodSandboxMetrics`) and New ImageService streaming RPC (`StreamImages`). ([#136987](https://github.com/kubernetes/kubernetes/pull/136987), [@bitoku](https://github.com/bitoku)) [SIG Cluster Lifecycle, Node and Testing]
- Added the metric terminated_containers_total to track the number of containers failed or succeeded broken down by exit code ([#137453](https://github.com/kubernetes/kubernetes/pull/137453), [@rawsocket](https://github.com/rawsocket)) [SIG Instrumentation, Node and Testing]
- Added two scheduler metrics for Device Binding Conditions, covering allocation attempts and PreBind duration with status and driver labels. ([#137284](https://github.com/kubernetes/kubernetes/pull/137284), [@ttsuuubasa](https://github.com/ttsuuubasa)) [SIG Node and Scheduling]
- Added warning when kubectl rollout undo is used on resources managed with kubectl apply to prevent unexpected behavior from annotation mismatch ([#137064](https://github.com/kubernetes/kubernetes/pull/137064), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Adding multiple conditions support to kubectl wait command. ([#136855](https://github.com/kubernetes/kubernetes/pull/136855), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Adds metrics for constrained impersonation as described in https://kep.k8s.io/5284
  
  apiserver_impersonation_attempts_total{mode, decision}
  apiserver_impersonation_attempts_duration_seconds{mode, decision}
  apiserver_impersonation_authorization_attempts_total{mode, decision}
  apiserver_impersonation_authorization_attempts_duration_seconds{mode, decision} ([#137374](https://github.com/kubernetes/kubernetes/pull/137374), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Adds the ExtendWebSocketsToKubelet feature gate (Beta, default true in v1.36). When enabled, the API server proxies WebSocket exec/attach/portforward requests directly to the kubelet rather than translating or tunneling them at the API server. The kubelet now handles WebSocket-to-SPDY stream translation (exec/attach) and WebSocket tunneling (portforward) using the same handlers previously used at the API server. The kubelet advertises support for this feature to the API server via the NodeDeclaredFeatures mechanism; the API server only proxies directly to a kubelet that has advertised support. Two new ALPHA metrics are added to track routing decisions and WebSocket streaming volume: apiserver_websocket_streaming_requests_total (labels: subresource, proxy_type) and kubelet_streaming_websocket_requests_total (label: subresource). ([#136256](https://github.com/kubernetes/kubernetes/pull/136256), [@seans3](https://github.com/seans3)) [SIG API Machinery, Autoscaling, Node, Scheduling and Testing]
- Allow the CRI (and NRI) to block pod-level resizes. ([#137555](https://github.com/kubernetes/kubernetes/pull/137555), [@natasha41575](https://github.com/natasha41575)) [SIG Node]
- Bump coredns to 1.14.2 ([#137605](https://github.com/kubernetes/kubernetes/pull/137605), [@pacoxu](https://github.com/pacoxu)) [SIG Cloud Provider and Cluster Lifecycle]
- CRI API: A new field is added to the PullImageResponse message - `image_id`. This field serves as a unique identifier for the image on the node as returned by the container runtimes. ([#137217](https://github.com/kubernetes/kubernetes/pull/137217), [@stlaz](https://github.com/stlaz)) [SIG Node]
- DRA ResourceSlice controller: new optional `ReconcilePoolWithName` allows per-pool reconciliation without setting NodeName on slices, so the scheduler can use NodeSelector or allNodes for node-owned, cluster-visible resources (e.g. network-shared devices).  "All nodes" is no longer the default.  When publishing devices for the entire cluster, it *must* be set explicitly. ([#137365](https://github.com/kubernetes/kubernetes/pull/137365), [@yaroslavborbat](https://github.com/yaroslavborbat)) [SIG Node and Testing]
- Enable the feature gate `RestartAllContainersOnContainerExits` by default. The RestartAllContainersOnContainerExits feature is promoted to beta. ([#136681](https://github.com/kubernetes/kubernetes/pull/136681), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node and Testing]
- Enables Prometheus native histogram support in apiserver when feature gate is enabled.
  Histograms are exposed in both classic and native formats using
  exponential bucket configuration (factor=1.1, max buckets=160) ([#136763](https://github.com/kubernetes/kubernetes/pull/136763), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Cloud Provider, Instrumentation, Network, Node, Scheduling and Testing]
- Enables Prometheus native histogram support in kube-controller-manager when feature gate is enabled.
  Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160) ([#137779](https://github.com/kubernetes/kubernetes/pull/137779), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation and Testing]
- Enables Prometheus native histogram support in kube-proxy when feature gate is enabled.
  Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160) ([#137781](https://github.com/kubernetes/kubernetes/pull/137781), [@richabanker](https://github.com/richabanker)) [SIG Network]
- Enables Prometheus native histogram support in kubelet when feature gate is enabled.
  Histograms are exposed in both classic and native formats using exponential bucket configuration (factor=1.1, max buckets=160) ([#137780](https://github.com/kubernetes/kubernetes/pull/137780), [@richabanker](https://github.com/richabanker)) [SIG Node]
- Enables Prometheus native histogram support in scheduler when feature gate is enabled.
  Histograms are exposed in both classic and native formats using
  exponential bucket configuration (factor=1.1, max buckets=160) ([#137466](https://github.com/kubernetes/kubernetes/pull/137466), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation, Scheduling and Testing]
- Ensures single-container pod can restart quickly with RestartAllContainers action. ([#136966](https://github.com/kubernetes/kubernetes/pull/136966), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node and Testing]
- Fix missing field conversions (BindsToNode, BindingConditions, BindingFailureConditions, AllowMultipleAllocations, Capacity) in DRA API v1beta1 hand-written conversion code ([#137240](https://github.com/kubernetes/kubernetes/pull/137240), [@yykkibbb](https://github.com/yykkibbb)) [SIG Node]
- Graduate ComponentFlagz to beta ([#137386](https://github.com/kubernetes/kubernetes/pull/137386), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Auth, Instrumentation, Node and Testing]
- Graduate ComponentStatusz to beta ([#137384](https://github.com/kubernetes/kubernetes/pull/137384), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Auth and Instrumentation]
- Instrument /flagz and /statusz endpoints with apiserver request metrics (apiserver_request_total, apiserver_request_duration_seconds), with group and version labels reflecting the content-negotiated API version. ([#137021](https://github.com/kubernetes/kubernetes/pull/137021), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Instrumentation]
- Introduce index-based naming in ResourceSlice controller and ensure ResourceSlices and pools are sorted lexicographically before allocation, allowing users to control allocation priority. ([#136641](https://github.com/kubernetes/kubernetes/pull/136641), [@troychiu](https://github.com/troychiu)) [SIG Node and Testing]
- Introduces new staging modules `k8s.io/streaming` and `k8s.io/cri-streaming` for Kubernetes streaming transport and CRI streaming server code.
  
  `k8s.io/apimachinery/pkg/util/httpstream` (including `spdy` and `wsstream`) remains available as a deprecated compatibility wrapper backed by `k8s.io/streaming`.
  
  The extracted SPDY roundtripper preserves CIDR matching in `NO_PROXY`/`no_proxy`. ([#137298](https://github.com/kubernetes/kubernetes/pull/137298), [@dims](https://github.com/dims)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Kube-apiserver: The UnknownVersionInteroperabilityProxy feature gate graduates to beta and enabled by default. The `--peer-ca-file` flag is required to turn on the proxy. ([#137172](https://github.com/kubernetes/kubernetes/pull/137172), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Kubeadm: when using '--v=1' or higher log verbosity, print information of the CA certificate used for discovery  when using 'kubeadm join'. ([#137102](https://github.com/kubernetes/kubernetes/pull/137102), [@sivchari](https://github.com/sivchari)) [SIG Cluster Lifecycle]
- Kubectl explain: when a schema or field includes an externalDocs section, it is now displayed as:
  
  
      EXTERNAL DOCS:
          <description>
          URL: <url>
  
  
  This appears after the DESCRIPTION block for top-level resources and
  after the field description for individual fields. The section is
  omitted in short mode and when `externalDocs` is absent. ([#136988](https://github.com/kubernetes/kubernetes/pull/136988), [@pedjak](https://github.com/pedjak)) [SIG CLI]
- Kubectl: `kubectl describe node` now lists aggregated **ResourceSlices** when the `ResourceSlice` API is present, detailing slice name, driver, and pool. ([#131744](https://github.com/kubernetes/kubernetes/pull/131744), [@ArangoGutierrez](https://github.com/ArangoGutierrez)) [SIG CLI]
- Kubelet: if the `--client-ca-file` is updated while kubelet is running, the updated root certificates are now correctly used to advertise accepted authorities to TLS clients connecting to the kubelet endpoints. This behavior is guarded by the `ReloadKubeletClientCAFile` feature gate, which is enabled by default. ([#136762](https://github.com/kubernetes/kubernetes/pull/136762), [@HarshalNeelkamal](https://github.com/HarshalNeelkamal)) [SIG API Machinery, Auth, Node and Testing]
- Kubernetes is now built using Go 1.26.0 ([#137080](https://github.com/kubernetes/kubernetes/pull/137080), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Preserve the logs of restarted containers for containers restarted by feature RestartAllContainers. ([#136963](https://github.com/kubernetes/kubernetes/pull/136963), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node]
- Promote DRAPartitionableDevices to beta ([#137350](https://github.com/kubernetes/kubernetes/pull/137350), [@mortent](https://github.com/mortent)) [SIG Node, Scheduling and Testing]
- Promoted the `KubeletPodResourcesDynamicResources` and `KubeletPodResourcesGet` feature gates to GA. ([#136728](https://github.com/kubernetes/kubernetes/pull/136728), [@guptaNswati](https://github.com/guptaNswati)) [SIG Node and Testing]
- REVERT: CRI API: A new field is added to the PullImageResponse message - `image_id`. This field serves as a unique identifier for the image on the node as returned by the container runtimes. ([#137574](https://github.com/kubernetes/kubernetes/pull/137574), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node]
- Reduces the needs of the setcap build image for kube-apiserver by no longer requiring that image to contain a shell (`sh` or `dash` or `bash`). ([#136633](https://github.com/kubernetes/kubernetes/pull/136633), [@addyess](https://github.com/addyess)) [SIG Release]
- Server images now use `staging/src/k8s.io/component-base/logs/kube-log-runner` instead of `go-runner`, full compatability is maintained (including the same `/go-runner` executable path).
  
  In the future Kubernetes will use base-images without go-runner. ([#136954](https://github.com/kubernetes/kubernetes/pull/136954), [@BenTheElder](https://github.com/BenTheElder)) [SIG Instrumentation and Release]
- Support in-place pod resize of running non-sidecar initContainers. ([#137352](https://github.com/kubernetes/kubernetes/pull/137352), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Apps, Autoscaling, Node, Scheduling, Storage and Testing]
- The KubeletPSI feature has graduated to General Availability (GA) and continues to be enabled by default. This feature allows the Kubelet to expose Linux cgroup Pressure Stall Information (PSI) metrics, providing deeper visibility into system and pod-level resource contention (CPU, Memory, and I/O) via the Kubelet Summary API. ([#136548](https://github.com/kubernetes/kubernetes/pull/136548), [@mariafromano-25](https://github.com/mariafromano-25)) [SIG Node]
- This change allows the Topology, CPU, and Memory managers to recognize and act upon
  `pod.spec.resources`, enabling two flexible resource management models. Both models
  support `guaranteed` pods that contain a mix of containers that may be eligible to receive
  exclusive resource allocation or be part of the pod-allocated shared resource pool. ([#134768](https://github.com/kubernetes/kubernetes/pull/134768), [@KevinTMtz](https://github.com/KevinTMtz)) [SIG Node and Testing]
- Update `kubectl kuberc set` with options for setting `credentialPluginPolicy` and `credentialPluginAllowlist` ([#137300](https://github.com/kubernetes/kubernetes/pull/137300), [@pmengelbert](https://github.com/pmengelbert)) [SIG CLI]
- When `kubectl exec` or `kubectl logs` are run with a specified container name, and no container with that name is found, `kubectl` now lists the names of containers that would be valid to specify. ([#136973](https://github.com/kubernetes/kubernetes/pull/136973), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]

### Documentation

- Add metric component and endpoint to generated metric reference documentation. ([#136360](https://github.com/kubernetes/kubernetes/pull/136360), [@skl](https://github.com/skl)) [SIG Instrumentation and Testing]

### Failing Test

- PLEGOnDemandRelist feature flag is kept a Beta level, but switched off by default. ([#137909](https://github.com/kubernetes/kubernetes/pull/137909), [@dims](https://github.com/dims)) [SIG Node]

### Bug or Regression

- Add `--detach-keys` flag to `kubectl attach` and `kubectl run`, allowing detach without terminating the container ([#134997](https://github.com/kubernetes/kubernetes/pull/134997), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG API Machinery and CLI]
- Capped nf_conntrack_max to 1,048,576 to prevent excessive memory consumption on high-core machines when using automatic calculation. ([#137002](https://github.com/kubernetes/kubernetes/pull/137002), [@kairosci](https://github.com/kairosci)) [SIG Apps and Network]
- CustomResourceDefinitions: Fixed server-side apply field ownership tracking so that metadata ownership is correctly tracked for writes to the /status subresource.
  Custom Resources: Fixed server-side apply field ownership to NOT be updates to metadata from the /status subresource since these writes are wiped for custom resources. ([#137689](https://github.com/kubernetes/kubernetes/pull/137689), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Network and Testing]
- DRA BindingConditions: when the DRABindingConditions feature was enabled, reusing the same claim among different pods may have caused a panic in the scheduler when deallocation happens in parallel (a rare race condition). ([#137371](https://github.com/kubernetes/kubernetes/pull/137371), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Disallow setting a resize restart policy of `RestartContainer` on non-sidecar initContainers, as the resize of such containers has never been supported. ([#137458](https://github.com/kubernetes/kubernetes/pull/137458), [@natasha41575](https://github.com/natasha41575)) [SIG Apps, Node and Testing]
- Explicitly writes memory.min=0 for QoS cgroups when the calculated requests are zero ([#137637](https://github.com/kubernetes/kubernetes/pull/137637), [@QiWang19](https://github.com/QiWang19)) [SIG Node]
- Fix apiserver startup failure during upgrade when MultiCIDRServiceAllocator is enabled and the cluster has a large number of namespaces. The IP address repair controller now retries on Forbidden errors from admission plugins that are not yet ready. ([#137147](https://github.com/kubernetes/kubernetes/pull/137147), [@haojiwu](https://github.com/haojiwu)) [SIG Testing]
- Fix bug where users can't update HPAv2 resources that use object metrics with averageValue via the v1 HPA API ([#137856](https://github.com/kubernetes/kubernetes/pull/137856), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Autoscaling]
- Fix container restart policy validation error message to correctly show available actions when RestartAllContainersOnContainerExits feature gate is enabled ([#137369](https://github.com/kubernetes/kubernetes/pull/137369), [@kfess](https://github.com/kfess)) [SIG Apps]
- Fix goroutine hot-loop in client-go StartEventWatcher when the event broadcaster shuts down before the cancellation context fires. ([#137398](https://github.com/kubernetes/kubernetes/pull/137398), [@Rajneesh180](https://github.com/Rajneesh180)) [SIG API Machinery]
- Fix informer-gen to generate SetTransform calls that correctly override per-informer transforms. ([#137473](https://github.com/kubernetes/kubernetes/pull/137473), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Scheduling]
- Fix link file ownership of projected serviceAccountToken. ([#137332](https://github.com/kubernetes/kubernetes/pull/137332), [@gavinkflam](https://github.com/gavinkflam)) [SIG Storage]
- Fixed a bug preventing Pods sharing ResourceClaims from being scheduled with GangScheduling. ([#137647](https://github.com/kubernetes/kubernetes/pull/137647), [@nojnhuh](https://github.com/nojnhuh)) [SIG Node, Scheduling and Testing]
- Fixed a bug where, after a kubelet restart, regular containers in a pod with a
  sidecar (initContainer with restartPolicy: Always) and a startupProbe failed
  to restart after crashing. Affected pods remained stuck with RestartCount: 0
  indefinitely. ([#137146](https://github.com/kubernetes/kubernetes/pull/137146), [@george-angel](https://github.com/george-angel)) [SIG Node and Testing]
- Fixed an issue where zero-valued PSI (Pressure Stall Information) metrics were emitted by the kubelet when the OS does not support PSI, even if the KubeletPSI feature gate was enabled. ([#137326](https://github.com/kubernetes/kubernetes/pull/137326), [@amritansh1502](https://github.com/amritansh1502)) [SIG Node]
- Fixed how image names are compared to the values from `preloadedImagesVerificationAllowlist` in Kubelet's configuration. Previously, the use of "familiar" image names (e.g. "alpine") from a Pod wouldn't properly match the same name in `preloadedImagesVerificationAllowlist` in Kubelet's configuration. ([#137629](https://github.com/kubernetes/kubernetes/pull/137629), [@stlaz](https://github.com/stlaz)) [SIG Auth, Node and Testing]
- Fixed kubectl describe node to correctly display resource requests and limits for pods using pod-level resources. ([#137394](https://github.com/kubernetes/kubernetes/pull/137394), [@Nikateen](https://github.com/Nikateen)) [SIG CLI]
- Fixed redundant SSH command executions in the etcd failure e2e test. ([#137001](https://github.com/kubernetes/kubernetes/pull/137001), [@kairosci](https://github.com/kairosci)) [SIG API Machinery and Testing]
- Fixed the lastTerminationStatus to match RestartAllContainers action if the container was restarted this way. ([#136964](https://github.com/kubernetes/kubernetes/pull/136964), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node]
- Fixed validation error messages for restartPolicyRules and exitCodes.values to report "items" instead of "bytes" ([#137136](https://github.com/kubernetes/kubernetes/pull/137136), [@kfess](https://github.com/kfess)) [SIG Apps]
- Fixes incorrect behavior when using AllocationModeAll with DRA PrioritizedList that prevented the allocator from successfully allocating a claim even when devices were available. ([#137347](https://github.com/kubernetes/kubernetes/pull/137347), [@mortent](https://github.com/mortent)) [SIG Node]
- Fixes kube-proxy's nftables mode to work on systems with nft 1.1.3. ([#137501](https://github.com/kubernetes/kubernetes/pull/137501), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Fixes the total pod resources computation ([#137683](https://github.com/kubernetes/kubernetes/pull/137683), [@ndixita](https://github.com/ndixita)) [SIG CLI and Node]
- Garbage collector now correctly handles objects deleted externally, preventing spurious error logs. ([#136817](https://github.com/kubernetes/kubernetes/pull/136817), [@kairosci](https://github.com/kairosci)) [SIG API Machinery, Apps and Testing]
- Improved a misleading error message when updating `batch.Job`'s `status.startTime`. The error for unsuspended jobs now correctly indicates the field is immutable once set, instead of incorrectly referring to the action as a "removal". ([#136585](https://github.com/kubernetes/kubernetes/pull/136585), [@zhzhuang-zju](https://github.com/zhzhuang-zju)) [SIG Apps]
- K8s.io/client-go/transport now automatically reloads certificate authority roots from disk when they are supplied via a file path.  This functionality is enabled by default and can be disabled via the ClientsAllowCARotation feature gate. ([#132922](https://github.com/kubernetes/kubernetes/pull/132922), [@yt2985](https://github.com/yt2985)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Instrumentation, Network, Node, Release, Scheduling and Testing]
- K8s.io/client-go/transport now garbage collects TLS cache entries and client certificate rotation go routines when a transport is no longer used.  This functionality is enabled by default and can be controlled via the ClientsAllowTLSCacheGC feature gate.  Binaries embedding k8s.io/client-go, but not wiring the feature gates can disable it by setting the KUBE_FEATURE_ClientsAllowTLSCacheGC=false environment variable.  When the feature is disabled, the TLS cache can grow indefinitely and client certificate rotation go routines are leaked.  The new rest_client_transport_cert_rotation_gc_calls_total{} and rest_client_transport_cache_gc_calls_total{result: deleted/skipped} counter metrics can be used with the preexisting rest_client_transport_* metrics to help with debugging. ([#136355](https://github.com/kubernetes/kubernetes/pull/136355), [@enj](https://github.com/enj)) [SIG API Machinery, Architecture, Auth, Instrumentation, Node and Testing]
- Kubeadm: ignore EINVAL when unmounting /var/lib/kubelet peer mounts during reset ([#137494](https://github.com/kubernetes/kubernetes/pull/137494), [@fuweid](https://github.com/fuweid)) [SIG Cluster Lifecycle]
- Kubelet now sets `PodReadyToStartContainers` condition immediately after sandbox creation rather than after image pull, reducing the time to condition True. ([#134660](https://github.com/kubernetes/kubernetes/pull/134660), [@Priyankasaggu11929](https://github.com/Priyankasaggu11929)) [SIG Apps, Node and Testing]
- Kubelet: relist pods on-demand for lower latency operations. Guarded by the new beta feature gate "PLEGOnDemandRelist". ([#137362](https://github.com/kubernetes/kubernetes/pull/137362), [@tallclair](https://github.com/tallclair)) [SIG Node]
- Kubelet_pod_start_sli_duration_seconds_bucket metric now matches pod startup latency SLI/SLO documentation. ([#131950](https://github.com/kubernetes/kubernetes/pull/131950), [@alimaazamat](https://github.com/alimaazamat)) [SIG Node]
- Kubernetes is now built using Go 1.26.1 ([#137474](https://github.com/kubernetes/kubernetes/pull/137474), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release and Testing]
- Previously, when trying to allocate devices through DRA for a node timed out, scheduling would proceed with another node if any had the necessary resources. This potentially hid that a node was ignored. Worse, if scheduling was slow overall, the pod was incorrectly moved to "unschedulable" and only retried after a periodic sweep. Now timeouts are errors that are always visible as pod scheduling failures and get retried with per-pod exponential backoff. ([#137607](https://github.com/kubernetes/kubernetes/pull/137607), [@0xMH](https://github.com/0xMH)) [SIG Node, Scheduling and Testing]
- Reflecting the expected replica count to the output of kubectl scale command ([#136945](https://github.com/kubernetes/kubernetes/pull/136945), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Remove `GuaranteedQoSPodCPUResize` from node declared features. ([#136759](https://github.com/kubernetes/kubernetes/pull/136759), [@pravk03](https://github.com/pravk03)) [SIG Node and Testing]
- Validation messages for a Pod's `status.resourceClaimStatuses[].resourceClaimName` now refer correctly to the `resourceClaimName` field instead of the `name` field. ([#137321](https://github.com/kubernetes/kubernetes/pull/137321), [@nojnhuh](https://github.com/nojnhuh)) [SIG Apps]
- Writes to the ServiceCIDR main resource now ignore status field changes in the request, consistent with all other Kubernetes APIs.
  The ServiceCIDRStatusFieldWiping feature gate can be disabled to restore the previous behavior; it will be locked to enabled in a future release. ([#137715](https://github.com/kubernetes/kubernetes/pull/137715), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Network and Testing]

### Other (Cleanup or Flake)

- Cri-client helper method NewLogOptions was removed and LogOptions must be constructed directly. This eliminates the unwanted depdendency from cri-client to apimachinery. ([#137827](https://github.com/kubernetes/kubernetes/pull/137827), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node and Release]
- For performance reasons, `kubectl describe` now defaults to showing related events only when describing a single object. Passing `--show-events` explicitly when describing multiple objects or fuzzy matching on prefix will still show related events if desired. ([#137145](https://github.com/kubernetes/kubernetes/pull/137145), [@mark-liu](https://github.com/mark-liu)) [SIG CLI]
- Improve stability by sorting containers by create time and ID in kubeGenericRuntimeManager.GetPods() and GetPod() ([#137566](https://github.com/kubernetes/kubernetes/pull/137566), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Node]
- Promote HPA metrics: reconciliations_total,reconciliation_duration_seconds,metric_computation_total,metric_computation_duration_seconds to beta ([#136178](https://github.com/kubernetes/kubernetes/pull/136178), [@omerap12](https://github.com/omerap12)) [SIG Apps, Autoscaling and Instrumentation]
- Promote `SELinuxChangePolicy` & `SELinuxMountReadWriteOncePod` to GA; it is now enabled unconditionally. ([#136912](https://github.com/kubernetes/kubernetes/pull/136912), [@dfajmon](https://github.com/dfajmon)) [SIG Apps, Storage and Testing]
- Reduced get PV request from KCM pv-controller for CSI volumes ([#134290](https://github.com/kubernetes/kubernetes/pull/134290), [@huww98](https://github.com/huww98)) [SIG Apps and Storage]
- Removed misleading `SuggestFor` entries from `kubectl wait` so that it is no longer suggested when users type `kubectl list` or `kubectl ps` ([#137266](https://github.com/kubernetes/kubernetes/pull/137266), [@kfess](https://github.com/kfess)) [SIG CLI and Testing]
- Removes the dead `--bounding-dirs` flag and `BoundingDirs` field from deepcopy-gen. ([#137348](https://github.com/kubernetes/kubernetes/pull/137348), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- The "Failed to update lease optimistically" log message may not be shown to users anymore, depending on the log level they have set. ([#137753](https://github.com/kubernetes/kubernetes/pull/137753), [@adamkasztenny](https://github.com/adamkasztenny)) [SIG API Machinery]
- The GetPCIeRootAttributeByPCIBusID helper now accepts a `fs.ReadLinkFS` optional argument to be filesystem-independenent ([#137220](https://github.com/kubernetes/kubernetes/pull/137220), [@ffromani](https://github.com/ffromani)) [SIG Node]
- The cri-api client is now accepts a context and do not accept logger on iniitalization. ([#137248](https://github.com/kubernetes/kubernetes/pull/137248), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Cluster Lifecycle, Node and Testing]
- Truncates the watch cache RV metric to 15 digits to ensure precision ([#137615](https://github.com/kubernetes/kubernetes/pull/137615), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Instrumentation]
- `v1alpha1` WebhookAdmissionConfiguration has been removed. It was deprecated in v1.17 in favor of `apiserver.config.k8s.io/v1`. ([#137379](https://github.com/kubernetes/kubernetes/pull/137379), [@aramase](https://github.com/aramase)) [SIG API Machinery and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
- cel.dev/expr: v0.24.0 → v0.25.1
- github.com/cncf/xds/go: [0feb691 → ee656c7](https://github.com/cncf/xds/compare/0feb691...ee656c7)
- github.com/coredns/corefile-migration: [v1.0.30 → v1.0.31](https://github.com/coredns/corefile-migration/compare/v1.0.30...v1.0.31)
- github.com/envoyproxy/go-control-plane/envoy: [v1.35.0 → v1.36.0](https://github.com/envoyproxy/go-control-plane/compare/envoy/v1.35.0...envoy/v1.36.0)
- github.com/envoyproxy/go-control-plane: [75eaa19 → v0.14.0](https://github.com/envoyproxy/go-control-plane/compare/75eaa19...v0.14.0)
- github.com/envoyproxy/protoc-gen-validate: [v1.2.1 → v1.3.0](https://github.com/envoyproxy/protoc-gen-validate/compare/v1.2.1...v1.3.0)
- github.com/google/cadvisor: [v0.56.0 → v0.56.2](https://github.com/google/cadvisor/compare/v0.56.0...v0.56.2)
- github.com/google/pprof: [27863c8 → 294ebfa](https://github.com/google/pprof/compare/27863c8...294ebfa)
- github.com/ianlancetaylor/demangle: [bd984b5 → f615e6b](https://github.com/ianlancetaylor/demangle/compare/bd984b5...f615e6b)
- github.com/onsi/ginkgo/v2: [v2.27.4 → v2.28.1](https://github.com/onsi/ginkgo/compare/v2.27.4...v2.28.1)
- github.com/onsi/gomega: [v1.39.0 → v1.39.1](https://github.com/onsi/gomega/compare/v1.39.0...v1.39.1)
- github.com/spf13/cobra: [v1.10.0 → v1.10.2](https://github.com/spf13/cobra/compare/v1.10.0...v1.10.2)
- go.opentelemetry.io/contrib/detectors/gcp: v1.38.0 → v1.39.0
- golang.org/x/telemetry: 8fff8a5 → bd525da
- golang.org/x/tools: v0.40.0 → v0.41.0
- google.golang.org/grpc: v1.78.0 → v1.79.3
- google.golang.org/protobuf: v1.36.11 → f2248ac
- k8s.io/klog/v2: v2.130.1 → v2.140.0
- k8s.io/kube-openapi: a19766b → 43fb72c
- sigs.k8s.io/knftables: v0.0.17 → v0.0.21

### Removed
- go.uber.org/automaxprocs: v1.6.0



# v1.36.0-alpha.2


## Downloads for v1.36.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes.tar.gz) | 7aa0ed3b03a1574ec0db00d47381e3c76610be89f5ceb52145add9f01533d72833b9594499e0a9af43c41df38d4c448906689cd5662532f32728e11f0f0b39d7
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-src.tar.gz) | a6858468f30d207b375dd2278d8653c69211b364136b212e84ee563bba6ca2bfe89d5ecefda3bc88671ee0c901f02d0f50eefe1e12b7cc58a842fad90811bd23

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | 2dde9c67ba2d165e46d4e09c886b8b17563f3767f6fcb5ed98883dfe5bd2ecd7f70af94b0d71678a36916f8d6a6ba1cd3b2c9b4e8b7084e94dfef075291e32c5
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 70178cdf6431041bcd6bcd55e67ba978ec5fe53a47ded53d6fa932e65af8742a47099de5c20fb44da279e957abf57f54281b4ce8f7f7e638810c89851751c7fd
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 5ca76f2388552d2a4d601a19c0120d8db65f8f151ac2223a69cf426523de7d2d9532104d5464b15a588cb3c283203069f5347edd5a3ac912d459d7c703467b81
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | f9901ed9e5a1d9b8638a2ac84e65e7986c81d2d72d086aa14cf298397d72a290e6b22b644cb74aa61ff0181df6e3ca38103863289679d4ae00195827a3333c93
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | a347d318571485aa21325ba4de7dd4afa29bffa4b8060ed04a9ef3afe7cea4f12c520e6eebc1b96e991147f0123096a1a745f905b687baa6981d1f8d16971452
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 2e493202a0023af3cff5f5e6d19307791a994eac931f18fe33fe5016861d5b17690b75aece91c02b5e022d4e00a824e85cb7fdc43ca45e238cea16d202ef7ec6
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 9feebb6cca23158f299085945e1ff6db5c1742a72d157f89d837811f84db5f9cb9b5db042f2898840a86839621b89066a1ccc925cd2a981af64bbf513d94e229
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | c8a42085199cf548119fd2af703c84cc1d2fda61086b50e09d03ea18cd0aaf28063d21eb258b64c30884022c6ab81b78bfd6d848fc4bf04b410d2972eb23266f
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-windows-386.tar.gz) | ccb9320af60720149913f39a88d4abdf5c9d53d3ddd396a61504288576e6f2f27ef21cb26a23e27e34b4c56c95eb1daa54ed389460c51660074a92a3c34eee23
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | c068c174870aca6710c688088b46bee4e852ea97be7e8c02aad10bef81d1a3e6485b50ff41bee11121cce357f74cf54d67f65df04087f590ff6259bd03eb2d11
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 9247e597dab4fb45facc3c32d4b2772e7bb5c9814a1f5ae64b5fdba5469784957df154a7b7aedc99e79e4fd81e07f7fe2b0eb492e17492fba497e94e5e501a8f

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | ef279ca406ed3e939972353287b9ad769df0915749c5ffe61f096b6f75ab7604ff03177affc8c66a2dd688fe8b3fb6c1a7e8deabb4985fa7f40517f483fc5983
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | a3414c851a8c0cf42abfb1ba2e17fb40b31b4d2cd9a4224ed8ecfc951b127a829f4ab14fa2cf68dbabfbe19aa6212f09c207d25518832b13e9f81986415a1a88
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 841d2a9dd87ab1232d79611c855c1dd2d9aca11afdd4d790692e38894ef6fd37b382c68aacaea32daf71b39f196bea2448d860ce9ed70b03766d83895779a68b
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | ec89bea22f65a612eb0e52b03c6d277c35f5f2e2cbbc4cd50510f013b4d7ca6702f47017dcb417e1cc257017b1755ae02f6c21fac0d44881be2814aed0130127

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 37fca1f367e1b78035fd95f1bc547fca89018fc42f450e3660d6f108890e6f319a1e94bf52584178558101375846baf7721a59e88ef3a22f8a493381bbc2a6e1
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 6ddbfbe217414c874d2367cdba0943d9a8ead4c98bb4e2ccaec64cdb2a0218bae46f108b4fb3ba949f78707cc47477fd0b97dc2291a83d5c08832201a185f595
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 6e1707b097daae5e2a68ffd068245034e056ee49d96a70bfce430de3b752ddc148027a9b94155d77da270df617d9450b59e02749fc06a41622e48c9505f00553
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 12674c621376b790dc2421a113ce3ca226af8d649a65d465ddf0ce29f21ecea39f74283b39ced2f8b894595b86f0c40bfcaea37e9d7b44e95d1f8cdb0f01e6b8
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 25e2cd21678e8051716172074a8cc9efeb53e8ff275a15f9d0e801616804b9cbb1a5b16378e057b905dad75ce0e94e9837934343374bef87d4bf9843c8b3ef76

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.36.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Extended PostFilterResult with a list of Pods that the PostFilter plugin preempted.
  PostFilter plugin implementations need to be updated to return the preempted Pods (if any) in the PostFilterResult from the PostFilter method; returning nil means the plugin did not preempt any Pods at all. ([#136254](https://github.com/kubernetes/kubernetes/pull/136254), [@tosi3k](https://github.com/tosi3k)) [SIG Scheduling]
  - Kubeadm: removed the integrated support for flex-volumes in kubeadm. Users were advised to migrate away from flex-volumes as recommended by SIG Storage, since 1.22. If kubeadm users wish to continue using the feature, they would need a custom image for the KCM that is not based on distroless, pass the KCM flag `--flex-volume-plugin-dir` and mount the directory `/usr/libexec/kubernetes/kubelet-plugins/volume/exec` in the KCM static pod using kubeadm's `extraVolumes` mechanism before upgrading to 1.36. Up until now, kubeadm automatically did the mounting if the user passed the flag. ([#136423](https://github.com/kubernetes/kubernetes/pull/136423), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
 
## Changes by Kind

### Dependency

- Updates the etcd client library to v3.6.8 ([#137225](https://github.com/kubernetes/kubernetes/pull/137225), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Auth, Cloud Provider, Cluster Lifecycle, Etcd, Node, Scheduling and Testing]

### Deprecation

- Disabled git-repo volume plugin by default, with no option to turn it back on. ([#136400](https://github.com/kubernetes/kubernetes/pull/136400), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Storage]

### API Change

- Add a new concurrent-node-status-updates flag that is split from the concurrent-node-syncs flag ([#136716](https://github.com/kubernetes/kubernetes/pull/136716), [@yonizxz](https://github.com/yonizxz)) [SIG Cloud Provider]
- Fixed an issue in kube-apiserver, allowing it to recover from an established connection to an incorrect server that never returns the expected response during APIService availability checks. ([#137157](https://github.com/kubernetes/kubernetes/pull/137157), [@bsalamat](https://github.com/bsalamat)) [SIG API Machinery]
- Graduated MutatingAdmissionPolicy to GA (v1) in Kubernetes 1.36. The feature is now enabled by default. ([#136039](https://github.com/kubernetes/kubernetes/pull/136039), [@lalitc375](https://github.com/lalitc375)) [SIG API Machinery, Architecture, Etcd and Testing]
- Introduced stability-based lifecycle for declarative validation (Alpha/Beta/Stable). Scheduling Workload v1alpha1 now uses explicit declarative enforcement. ([#136793](https://github.com/kubernetes/kubernetes/pull/136793), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Scheduling]
- K8s.io/api: REST API types no longer implement marker `ProtoMessage()` methods, identifying them as standard v1 proto messages. Protobuf serialization of Kubernetes API types should use [k8s.io/apimachinery/pkg/runtime/serializer/protobuf](https://pkg.go.dev/k8s.io/apimachinery/pkg/runtime/serializer/protobuf). See [KEP-5589](https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/5589-gogo-dependency) for more details. ([#137084](https://github.com/kubernetes/kubernetes/pull/137084), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, Node, Scheduling and Storage]
- Pod Certificates (beta) now includes a PKCS#10 certificate signing request for wider compatibility with existing certificate authority software. ([#136729](https://github.com/kubernetes/kubernetes/pull/136729), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Auth, Node and Testing]
- Promoted several component-base metrics (`kubernetes_build_info`, `rest_client_requests_total`, `rest_client_request_duration_seconds`, `running_managed_controllers`) from Alpha to Beta stability, providing stronger API and label stability guarantees for consumers. ([#136154](https://github.com/kubernetes/kubernetes/pull/136154), [@bhope](https://github.com/bhope)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scalability, Scheduling, Storage and Testing]
- Replace deprecated sets.String with sets.Set[string] in apiserver admission subsystem. This is a breaking change for consumers of the NewLifecycle function. ([#134044](https://github.com/kubernetes/kubernetes/pull/134044), [@mcallzbl](https://github.com/mcallzbl)) [SIG API Machinery and Auth]
- Updates API server internal API group to improve openapi schema correctness for fields being optional or required ([#134675](https://github.com/kubernetes/kubernetes/pull/134675), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery, Apps, Auth, Node and Storage]

### Feature

- Add an alpha informer_processing_latency_seconds histogram metric to measure event handler execution time in RealFIFO. ([#137101](https://github.com/kubernetes/kubernetes/pull/137101), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Add the kubelet_metrics_provider metric to help users identify where kubelet's metrics are coming from. ([#136952](https://github.com/kubernetes/kubernetes/pull/136952), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Node]
- Added a show-secret flag to the diff command to explicitly allow secret values to be displayed during the diff operation. ([#137019](https://github.com/kubernetes/kubernetes/pull/137019), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Adds alpha metrics `apiserver_peer_proxy_errors_total` and `apiserver_peer_discovery_sync_errors_total` to apiserver to track errors encountered in peer proxying and peer discovery ([#137065](https://github.com/kubernetes/kubernetes/pull/137065), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Client-go: informer stores now keep track the resourceVersion they are synced to (via add/update/delete events, or replace calls, or bookmark events), and provide a `LastStoreSyncResourceVersion` method to obtain this resource version. This method can return `""` if the store has not been synced to yet, and depends on the AtomicFIFO feature being enabled. ([#134827](https://github.com/kubernetes/kubernetes/pull/134827), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Testing]
- Introduced PodGroup scheduling cycle in kube-scheduler to schedule entire PodGroup in one cycle. ([#136618](https://github.com/kubernetes/kubernetes/pull/136618), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- K8s.io/cloud-provider: Adds missing TLS flags to webhook serving options ([#136816](https://github.com/kubernetes/kubernetes/pull/136816), [@damdo](https://github.com/damdo)) [SIG Cloud Provider]
- Kube-controller-manager: The daemonset controller now defers syncing a DaemonSet object when the controller has not yet observed daemonset or pod writes from the last time the object was synced. This prevents spurious creation of duplicate pods for nodes when the controller's cache is stale. When a sync is deferred for this reason, a `daemonset_controller_stale_sync_skips_total` metric is incremented and a message is logged by the daemonset controller. This behavior can be temporarily disabled by setting the `StaleControllerConsistencyDaemonSet` feature gate to false. ([#134937](https://github.com/kubernetes/kubernetes/pull/134937), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Kube-controller-manager: The job controller now defers syncing a Job object when the controller has not yet observed job or pod writes from the last time the object was synced. This prevents spurious creation of duplicate pods for jobs when the controller's cache is stale. When a sync is deferred for this reason, a `job_controller_stale_sync_skips_total` metric is incremented and a message is logged by the job controller. This behavior can be temporarily disabled by setting the `StaleControllerConsistencyJob` feature gate to false. ([#137210](https://github.com/kubernetes/kubernetes/pull/137210), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Apps]
- Kubeadm: the preflight check `ContainerRuntimeVersion` validates if the installed container runtime supports the `RuntimeConfig` gRPC method. For older kubelet versions than 1.37, it will return a preflight warning. ([#136898](https://github.com/kubernetes/kubernetes/pull/136898), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubelet: defer the configurations flags (and the related fallback behavior) deprecation removal timeline from 1.36 to 1.37 to align with containerd v1.7 support ([#136846](https://github.com/kubernetes/kubernetes/pull/136846), [@carlory](https://github.com/carlory)) [SIG Node and Testing]
- Kubernetes is now built using Go 1.25.7 ([#136982](https://github.com/kubernetes/kubernetes/pull/136982), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Pods that are preempted when in PreBind phase will go back to backoff queue instead of being completely deleted from apisever. ([#135502](https://github.com/kubernetes/kubernetes/pull/135502), [@Argh4k](https://github.com/Argh4k)) [SIG Scheduling and Testing]
- Prevent the replicaset controller from spuriously reconciling while its own writes have not been read. ([#137212](https://github.com/kubernetes/kubernetes/pull/137212), [@michaelasp](https://github.com/michaelasp)) [SIG Apps]
- Updated feature gate `MutablePodResourcesForSuspendedJobs` and `MutableSchedulingDirectivesForSuspendedJobs` to be enabled by default. ([#135965](https://github.com/kubernetes/kubernetes/pull/135965), [@kannon92](https://github.com/kannon92)) [SIG Apps and Testing]

### Failing Test

- (reverts #136796, so ignore the release note from there...) ([#137169](https://github.com/kubernetes/kubernetes/pull/137169), [@danwinship](https://github.com/danwinship)) [SIG Network]

### Bug or Regression

- Extend unsupported Table object detection from watchlist only to all List and Watch operations. This prevents the reflector from processing resources returned in Table format which it cannot properly handle ([#136937](https://github.com/kubernetes/kubernetes/pull/136937), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Fix PodCertificateRequest OwnerReference using incorrect apiVersion "core/v1" instead of "v1", which prevented garbage collection of PodCertificateRequests when their owning Pod was deleted. ([#137008](https://github.com/kubernetes/kubernetes/pull/137008), [@srhppr](https://github.com/srhppr)) [SIG Auth and Node]
- Fix a bug where the DRA manager did not initialize sharedID from cache when DRAConsumableCapacity is enabled. ([#136734](https://github.com/kubernetes/kubernetes/pull/136734), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node and Scheduling]
- Fixed /metrics/resource container_swap_usage_bytes to report the correct container swap usage ([#137098](https://github.com/kubernetes/kubernetes/pull/137098), [@yuanwang04](https://github.com/yuanwang04)) [SIG Apps, Node and Testing]
- Fixed a bug where `kubectl plugin list` failed to detect overshadowed plugins on Windows. ([#136689](https://github.com/kubernetes/kubernetes/pull/136689), [@kfess](https://github.com/kfess)) [SIG CLI]
- Fixed a bug where the event_handling_duration_seconds/preemption_goroutines_duration_seconds/run_podsandbox_duration_seconds/store_schedule_results_duration_seconds metric was recording
  near-zero latency values instead of actual value. ([#135749](https://github.com/kubernetes/kubernetes/pull/135749), [@novahe](https://github.com/novahe)) [SIG Architecture, Instrumentation, Node and Scheduling]
- Fixed a data race in k8s.io/apiserver/pkg/cel/openapi/resolver with (probably) no real-world impact. ([#136802](https://github.com/kubernetes/kubernetes/pull/136802), [@pohly](https://github.com/pohly)) [SIG API Machinery, Node and Testing]
- Fixed kubectl logs -f to wait for containers to start instead of failing 
  immediately when pods are in ContainerCreating or PodInitializing states ([#136411](https://github.com/kubernetes/kubernetes/pull/136411), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Fixes kube-proxy log spam when all of a Service's endpoints were unready. ([#136743](https://github.com/kubernetes/kubernetes/pull/136743), [@ansilh](https://github.com/ansilh)) [SIG Network]
- Fixes kube-proxy's nftables mode to work on systems with nft 1.1.3. ([#136796](https://github.com/kubernetes/kubernetes/pull/136796), [@kairosci](https://github.com/kairosci)) [SIG API Machinery, Auth and Network]
- Kubeadm: do not add learner member to etcd client endpoints ([#137251](https://github.com/kubernetes/kubernetes/pull/137251), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- The metrics `container_cpu_load_average_10s`, `container_cpu_load_d_average_10s`, and `cpu_tasks_state` have been dropped from the reported metrics by cadvisor. This is done because the values were always 0, because a flag was not enabled in the kubelet. ([#134981](https://github.com/kubernetes/kubernetes/pull/134981), [@haircommander](https://github.com/haircommander)) [SIG Node and Testing]

### Other (Cleanup or Flake)

- Apiserver_rerouted_request_total metric will expose labels for group, version and resource. ([#137063](https://github.com/kubernetes/kubernetes/pull/137063), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Removed the generally available feature gate `HonorPVReclaimPolicy`, which was locked and enabled since 1.33. ([#135335](https://github.com/kubernetes/kubernetes/pull/135335), [@carlory](https://github.com/carlory)) [SIG Apps and Storage]
- The deprecated SeparateCacheWatchRPC feature gate is now locked to its default value (false) and can no longer be overridden. The feature gate will be removed in a future release. ([#135808](https://github.com/kubernetes/kubernetes/pull/135808), [@tico88612](https://github.com/tico88612)) [SIG API Machinery]
- Update etcd images to v3.6.8 ([#137107](https://github.com/kubernetes/kubernetes/pull/137107), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Upgrades functionality of `kubectl kustomize` as described at https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.8.1 ([#136892](https://github.com/kubernetes/kubernetes/pull/136892), [@koba1t](https://github.com/koba1t)) [SIG Architecture and CLI]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.27.4 → v2.27.7](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.27.4...v2.27.7)
- github.com/sergi/go-diff: [v1.2.0 → v1.4.0](https://github.com/sergi/go-diff/compare/v1.2.0...v1.4.0)
- go.etcd.io/etcd/api/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/client/pkg/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/client/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/pkg/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/server/v3: v3.6.7 → v3.6.8
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.64.0 → v0.65.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.63.0 → v0.65.0
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.64.0 → v0.65.0
- go.opentelemetry.io/contrib/propagators/b3: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/exporters/stdout/stdouttrace: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/metric: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/sdk/metric: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/sdk: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/trace: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel: v1.39.0 → v1.40.0
- google.golang.org/genproto/googleapis/api: 99fd39f → 8636f87
- google.golang.org/genproto/googleapis/rpc: 99fd39f → 8636f87
- k8s.io/utils: 914a6e7 → b8788ab
- sigs.k8s.io/kustomize/api: v0.20.1 → v0.21.1
- sigs.k8s.io/kustomize/cmd/config: v0.20.1 → v0.21.1
- sigs.k8s.io/kustomize/kustomize/v5: v5.7.1 → v5.8.1
- sigs.k8s.io/kustomize/kyaml: v0.20.1 → v0.21.1
- sigs.k8s.io/structured-merge-diff/v6: v6.3.1 → v6.3.2

### Removed
- github.com/pkg/errors: [v0.9.1](https://github.com/pkg/errors/tree/v0.9.1)



# v1.36.0-alpha.1


## Downloads for v1.36.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes.tar.gz) | 79ec354722859240b1d715c0da181e5a2a0fd354984ae9d511d58e7c09ec5cf7e54421db38a6d96b409b9a08b1c1b9dc13e7df20e7ab21299837d00cf72f162c
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-src.tar.gz) | f274bff791b16bb4de10730aabbfc027220f45c44d2dc8d1a8b575cc86421ec01fb106bcb2f3cb137145e64396ca37f2ec689932395162dcae5d3b6b65fc97ec

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 6fc2c7b184ee6435c0e7179dbe8ff63549631d9a5eb28262b10596a6f26e245ab2cf16402a6466e37b81a42760f811808796d1e83dd205125c0e64e1330772bd
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | e56eb183ba431d530b6cbd83ee94c1c398f3f4969cdee247092738a5cbe2b567d705788f95adec2f13cf17ebd791165903a0f1fcac9fcbf36ed65b9a00f38ac3
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 1a65eb81a4cf1631fa6fb102f2dfcffd29732bb96479c0432c9780f2dcb4600f8c85991b0f68f038ae963348d39291c74a855a71705093ecc09218ee4ed5271c
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | a1824ed2091dac2289c99c67e504b01b0176657675496752136a630ecf57347e0a5578a21c3bd74d0baad995d1e99ba0c78b5dfc2a61316227171c25a216111b
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 9e368482df69b6990c917d6df0f3589851d72bcd7304226970eae32898baeb76e5955e93bda577d72231ca2342562ab91dfecb19dd5479f91352a4577f8f7d92
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 7894d5868aa7888a26648ea338fbe63031e2b3ce0919a337e69cba002c369a0bbe6971ed7add1fd5e2284ebc696f9bb9c0180c298fe349140482b2e93a51d72a
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | ae379c93762b86b8ddd1f1076b2e37c2866991e5a5700eb08ff5b65aaeca552764c6bb0506376b502e67f1734f5655b59ca4aa751b572667a9522065822874f1
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 316d1093cd109d91b55c5ca18a8aa2d0e04feb29bb688b3d52018818329519c8fa6b226de8505de3ced920c064b8f150dccb2829d2bb2a4bed594e4d663377cd
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 48f7ca49e081393474c644c31a7bae8810dfc7673c2f1800207960ea14e73616e1b7717d312e4787e8b5179c5a46a256cdbaa80e01cdf37cf999f370a17060e1
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | d70670136a91a5f81f7b030f258881917371102e958382d3dfef425d5d76718cc242d7ee86dbaaaa5705e019b4178e323a29cc0d4ace6d803550ea8d412189e2
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 99c08b44870989a9629573f110317215e4ec177a25298e28f497deb75867838085b987a100c8ee678ccf1345e4cef075d159c5cb9198a3a342a92733309820d5

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | 9208265b86d2d7ebf8dd8f771a586de571aa07ab54c1d7428deb8803dbf63f2e396e30b103cac4da8ddf791be8b66dfce90e428906ec2d59e485953a6b6e1b7b
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 704ce893ee7b239194aae37b608b175940efbf60072622a661d866221f6fdba9c6b6c4ca11008bf2be1cf2c70cb9f6e5aece9052efa2468c1e3b73510c35d2f0
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 99cc1cf4169b4c8a00b7e7e4f49c98703e0191dd0aa0e4f641a0b35a92e7d5df14587d5b56f571fa4d00290d806bbf916cb3d934537883f9364a04d66d0ba958
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | ea9ac2489aa2b8d9a0bd8a8a12958b754edd0c9227aaee23dadf7fc7711cab5033edcb85dc545e6bd8cb78336e073b2cfb8ed8655932b91a4d8d22c3eb3dfb90

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 1cff3dd843e8ffbe2728967c3520bb000a551573fc9049fb4d3e6734d76a9ea72a2e54a0eca18bf68b0f023106714a877fb01b2e942720b707f738df1709361d
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 55a65f51bc9d25ca2993bf2d277e1ceadc585d8af90e3f92b9b6fd682b7fa78f8fd0bb4bba127143825c70227d8815b037159feb8a319c0b4fb406e3b4dfa913
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | e67d17e2d38716e6d20a777d360454f0a0e04d8f5dc94f8888c823df55ddb96973e819229e043d0e9a7f25027f4f64583f13443d633f568b62237e1fcceae8b4
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 807376145bb55d7b534523c18fac4f61ddb8d20e0646c2152de28cd547f1ef82787f01e8b43a7ea1128a9c76c7ee14183e28629032396ff124629c1a6c6c9ab4
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 1f204db26af19933504f50292513ea3e238f57ac9bc8c4faad8a71271cd2c2920515ad0267ed87618ac4d6f510f111fe5a2feeebe084425b0f369905f577cac8

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Added support for running PreBind plugins in parallel in the scheduler framework to improve the binding latency.
  Plugins can now opt-in to parallel execution by returning `AllowParallel: true` from the PreBindPreFlight method. PreBind plugin implementations need to be updated to return the PreBindPreFlightResult from the PreBindPreFlight method; returning nil retains the existing sequential behavior. ([#135393](https://github.com/kubernetes/kubernetes/pull/135393), [@tosi3k](https://github.com/tosi3k)) [SIG Node, Scheduling, Storage and Testing]
 
## Changes by Kind

### Dependency

- Fix a bug where pod lifecycle hooks could run for their full duration when pods are terminated. ([#136598](https://github.com/kubernetes/kubernetes/pull/136598), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]

### API Change

- Add --concurrent-resourceclaim-syncs to configure kube-controller-manager resource claim reconcile concurrency ([#134701](https://github.com/kubernetes/kubernetes/pull/134701), [@anson627](https://github.com/anson627)) [SIG API Machinery, Apps, Node and Testing]
- Added negative duration validation for imageMinimumGCAge ([#135997](https://github.com/kubernetes/kubernetes/pull/135997), [@ngopalak-redhat](https://github.com/ngopalak-redhat)) [SIG API Machinery and Node]
- Clarified documentation and comments to indicate that the `cpuCFSQuotaPeriod` kubelet config field requires the `CustomCPUCFSQuotaPeriod` feature gate when using non-default values. No functional changes introduced. ([#133845](https://github.com/kubernetes/kubernetes/pull/133845), [@rbiamru](https://github.com/rbiamru)) [SIG Node and Release]
- Correct openapi schema union validation for the PodGroupPolicy struct in scheduling v1alpha1 ([#136424](https://github.com/kubernetes/kubernetes/pull/136424), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery and Scheduling]
- Fixed a potential nil pointer dereference in the scheduler's NodeResourcesFitArgs validation when using RequestedToCapacityRatio scoring strategy ([#132120](https://github.com/kubernetes/kubernetes/pull/132120), [@flpanbin](https://github.com/flpanbin)) [SIG Scheduling]
- Fixes `fake.NewClientset()` to work properly with correct schema. ([#131068](https://github.com/kubernetes/kubernetes/pull/131068), [@soltysh](https://github.com/soltysh)) [SIG API Machinery]
- Generate fake.NewClientset which replace the deprecated NewSimpleClientset, for kube-aggregator and sample-apiserver ([#136537](https://github.com/kubernetes/kubernetes/pull/136537), [@soltysh](https://github.com/soltysh)) [SIG API Machinery]
- Graduate watch_list_duration_seconds from ALPHA to BETA ([#136086](https://github.com/kubernetes/kubernetes/pull/136086), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation, Node and Testing]
- Kube-apiserver: the `--audit-policy-file` config file now supports specifying `group: "*"` in resources rules to match all API groups ([#135262](https://github.com/kubernetes/kubernetes/pull/135262), [@cmuuss](https://github.com/cmuuss)) [SIG API Machinery, Auth and Testing]
- Kube-controller-manager: alpha gauge metrics for informer queue length are now published as `informer_queued_items{name=kube-controller-manager,group=<group>,resource=<resource>,version=<version>} <count>` ([#135782](https://github.com/kubernetes/kubernetes/pull/135782), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Locked the feature-gate VolumeAttributesClass to default (true) and bump VolumeAttributesClass preferred storage version to `storage.k8s.io/v1` ([#134556](https://github.com/kubernetes/kubernetes/pull/134556), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Etcd, Network, Node, Scheduling, Storage and Testing]
- Promote workqueue metrics from ALPHA to BETA ([#135522](https://github.com/kubernetes/kubernetes/pull/135522), [@petern48](https://github.com/petern48)) [SIG Architecture, Instrumentation and Testing]
- Removed the generally available feature gate `CSIMigrationPortworx`, which was locked and enabled since 1.33.
  - Removed alpha feature gate `InTreePluginPortworxUnregister`
  - Removed Portworx volume plugin from in-tree plugins because all operations are redirected to CSI. ([#135322](https://github.com/kubernetes/kubernetes/pull/135322), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Auth, Node, Scalability, Scheduling, Storage and Testing]
- The ImageVolumeWithDigest is added which adds the digest of image volumes to the container's status. ([#132807](https://github.com/kubernetes/kubernetes/pull/132807), [@iholder101](https://github.com/iholder101)) [SIG API Machinery, Apps, Node and Testing]
- The `endpoints` field in discovery.k8s.io/v1 EndpointSlice is now correctly defined as optional in the OpenAPI specification, matching the server's behavior. ([#136111](https://github.com/kubernetes/kubernetes/pull/136111), [@aojea](https://github.com/aojea)) [SIG Network]
- Update API comments to reflect that stable state of Dynamic Resource Allocation ([#136441](https://github.com/kubernetes/kubernetes/pull/136441), [@kannon92](https://github.com/kannon92)) [SIG API Machinery]

### Feature

- Add architecture to the kernel version column in the `kubectl get node -owide` output. ([#132402](https://github.com/kubernetes/kubernetes/pull/132402), [@astraw99](https://github.com/astraw99)) [SIG CLI]
- Add the `appProtocol` field to the service describe output. ([#135744](https://github.com/kubernetes/kubernetes/pull/135744), [@ali-a-a](https://github.com/ali-a-a)) [SIG CLI]
- Add write and read permissions for workloads to the admin cluster role. Add write permissions for workloads to the edit cluster role. Add read permissions for workloads to the view cluster role. ([#135418](https://github.com/kubernetes/kubernetes/pull/135418), [@carlory](https://github.com/carlory)) [SIG Auth]
- Added ALPHA metric `scheduler_pod_scheduled_after_flush_total` to count pods scheduled after being flushed from unschedulablePods due to timeout ([#135126](https://github.com/kubernetes/kubernetes/pull/135126), [@mrvarmazyar](https://github.com/mrvarmazyar)) [SIG Scheduling]
- Added kubectl explain -r flag as a shorthand for --recursive ([#135283](https://github.com/kubernetes/kubernetes/pull/135283), [@laervn](https://github.com/laervn)) [SIG CLI]
- Align the meaning of victim metrics between async preemption and sync preemption. The definition has been standardized to refer to the number of Pods chosen as victims. ([#135955](https://github.com/kubernetes/kubernetes/pull/135955), [@utam0k](https://github.com/utam0k)) [SIG Scheduling]
- CRD validation now strictly enforces ranges for numeric formats (int32, int64, float, double) when specified in the schema. Existing objects with out-of-range values are preserved via validation ratcheting ([#136582](https://github.com/kubernetes/kubernetes/pull/136582), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- Change the default debug profile from `legacy` to `general`. `legacy` profile is planned to be removed in v1.39. ([#135874](https://github.com/kubernetes/kubernetes/pull/135874), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI and Testing]
- Client-go informers can now enqueue new watch events while already-queued events are being processed. This avoids dropping watches during a burst of incoming events due to contention on slow processing. This behavior is controlled by the UnlockWhileProcessing client-go feature gate, which is enabled by default. ([#136264](https://github.com/kubernetes/kubernetes/pull/136264), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Scheduling]
- Client-go: Informer resync processing improved handling of Resync handling. This reduces contention on store locks between incoming events and handler updates, which may result in observable timing differences of handler invocations. This behavior is guarded by an AtomicFIFO feature gate. This gate is enabled by default in 1.36, but can be disabled if needed to temporarily regain the previous behavior. ([#136008](https://github.com/kubernetes/kubernetes/pull/136008), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Client-go: default informer behavior now updates store state with all the objects in a list or relist, before calling handler OnDelete/OnAdd/OnUpdate methods for individual items which were deleted/added/removed. This ensures that the store state which can be inspected by handlers actually corresponds to a set of objects that existed at a particular resource version on the server. This behavior is guarded by an AtomicFIFO feature gate. This gate is enabled by default in 1.36, but can be disabled if needed to temporarily regain the previous behavior. ([#135462](https://github.com/kubernetes/kubernetes/pull/135462), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Cloud Controller Manager now exports the counter metric `route_controller_route_sync_total`, which increments each time routes are synced with the cloud provider. This metric is in alpha stage. ([#136539](https://github.com/kubernetes/kubernetes/pull/136539), [@lukasmetzner](https://github.com/lukasmetzner)) [SIG API Machinery, Cloud Provider and Instrumentation]
- Enable WatchCacheInitializationPostStartHook by default ([#135777](https://github.com/kubernetes/kubernetes/pull/135777), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Graduated fine-grained kubelet API authorization to stable. ([#136116](https://github.com/kubernetes/kubernetes/pull/136116), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Node]
- ImageLocality plugin: consider ImageVolume images when scoring nodes for pod scheduling. ([#130231](https://github.com/kubernetes/kubernetes/pull/130231), [@Barakmor1](https://github.com/Barakmor1)) [SIG Scheduling]
- Kube-apiserver: Promoted `ExternalServiceAccountTokenSigner` feature to GA. ([#136118](https://github.com/kubernetes/kubernetes/pull/136118), [@HarshalNeelkamal](https://github.com/HarshalNeelkamal)) [SIG API Machinery and Auth]
- Kubeadm: Upgraded the `NodeLocalCRISocket` feature gating to GA and locked it to be enabled. ([#135742](https://github.com/kubernetes/kubernetes/pull/135742), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Kubeadm: added the flag --allow-deprecated-api to 'kubeadm config validate'. By default the command will print a warning for a deprecated API unless the flag is passed. Additionally, added missing support for v1beta4 UpgradeConfiguration to 'kubeadm config migrate|validate' commands. ([#135148](https://github.com/kubernetes/kubernetes/pull/135148), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: bumped the timeout of the `kubeadm upgrade` `CreateJob` preflight check to 1 minute. This allows Windows worker nodes to have more time to run the preflight check. It uses the `pause` image, so if you are experiencing slow pull times, you can either pre-pull the new pause on the work using `kubeadm config images pull --kubernetes-version TARGET` or skip the preflight check with `--ignore-preflight-errors`. ([#136273](https://github.com/kubernetes/kubernetes/pull/136273), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: removed the kubeadm specific feature gate ControlPlaneKubeletLocalMode which became GA in 1.35 and was locked to enabled. ([#135773](https://github.com/kubernetes/kubernetes/pull/135773), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: when patching a Node object do not exit early on unknown (non-allowlisted) API errors. Instead, always retry within the duration of the polling for getting and patching a Node object. ([#135776](https://github.com/kubernetes/kubernetes/pull/135776), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl get ingressclass now displays (default) marker for default IngressClass ([#134422](https://github.com/kubernetes/kubernetes/pull/134422), [@jaehanbyun](https://github.com/jaehanbyun)) [SIG CLI and Network]
- Kubernetes is now built using Go 1.25.6 ([#136465](https://github.com/kubernetes/kubernetes/pull/136465), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with Go 1.25.6 ([#136257](https://github.com/kubernetes/kubernetes/pull/136257), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Kubernetes is now built with Go 1.25.7 ([#136750](https://github.com/kubernetes/kubernetes/pull/136750), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Promote Relaxed validation for Services names to beta (enabled by default)
  
  Promote `RelaxedServiceNameValidation` feature to beta (enabled by default)
  The names of new Services names are validation with `NameIsDNSLabel()`,
  relaxing the pre-existing validation. ([#136389](https://github.com/kubernetes/kubernetes/pull/136389), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network]
- Promoted the `CSIServiceAccountTokenSecrets` feature gate to GA. ([#136596](https://github.com/kubernetes/kubernetes/pull/136596), [@aramase](https://github.com/aramase)) [SIG Auth and Storage]
- Promoting kubectl kuberc commands to beta ([#136643](https://github.com/kubernetes/kubernetes/pull/136643), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- The ResourceClaim controller now correctly handles unknown (non-pod) references in the status.reservedFor field by skipping them instead of halting the sync process. ([#136450](https://github.com/kubernetes/kubernetes/pull/136450), [@MohammedSaalif](https://github.com/MohammedSaalif)) [SIG Apps and Node]
- Update to latest cAdvisor 0.55.0 in our vendor dependencies ([#135829](https://github.com/kubernetes/kubernetes/pull/135829), [@dims](https://github.com/dims)) [SIG Node]
- Using pytorch based e2e integration test instead of tensorflow in some node e2e CI tests. ([#136397](https://github.com/kubernetes/kubernetes/pull/136397), [@dims](https://github.com/dims)) [SIG Testing]
- Using pytorch based e2e integration test instead of tensorflow in some node e2e CI tests. ([#136398](https://github.com/kubernetes/kubernetes/pull/136398), [@dims](https://github.com/dims)) [SIG Node and Testing]

### Failing Test

- Fixed device plugin test failures after kubelet restart. ([#135485](https://github.com/kubernetes/kubernetes/pull/135485), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node and Testing]

### Bug or Regression

- Added extra check to prevent users to work around DRA extended resource quota set by system admin ([#135434](https://github.com/kubernetes/kubernetes/pull/135434), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Aligned `kubectl label` output message to include 'modified' when labels are both added and removed ([#134849](https://github.com/kubernetes/kubernetes/pull/134849), [@tchap](https://github.com/tchap)) [SIG CLI]
- Apiserver liveness probes will now fail when the loopback client certificate expires. ([#136477](https://github.com/kubernetes/kubernetes/pull/136477), [@everettraven](https://github.com/everettraven)) [SIG API Machinery and Testing]
- Changed the behavior of default scheduler preemption plugin when preempting pods that are in "WaitOnPermit" phase. They are now moved to the scheduler backoff queue instead of being marked as unschedulable. ([#135719](https://github.com/kubernetes/kubernetes/pull/135719), [@Argh4k](https://github.com/Argh4k)) [SIG Scheduling and Testing]
- Changes some instances of error logs to info logs with verbosity level inside of controller/resourcequota and controller/garbagecollector ([#136040](https://github.com/kubernetes/kubernetes/pull/136040), [@petern48](https://github.com/petern48)) [SIG API Machinery and Apps]
- Changes the nodeGetCapabilities method of csiDriverClient returning NewUncertainProgressError while received a non final GRPC error ([#135930](https://github.com/kubernetes/kubernetes/pull/135930), [@249043822](https://github.com/249043822)) [SIG Node and Storage]
- Client-go informers: fix an unlikely deadlock during informer startup. ([#136509](https://github.com/kubernetes/kubernetes/pull/136509), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- DRA: when scheduling many pods very rapidly, sometimes the same device was allocated twice for different ResourceClaims due races between data processing in different goroutines. Depending on whether DRA drivers check for this during NodePrepareResources (they should, but maybe not all implement this properly), the second pod using the same device then failed to start until the first one is done or (worse) ran in parallel. ([#136269](https://github.com/kubernetes/kubernetes/pull/136269), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Disabled `SchedulerAsyncAPICalls` feature gate due to performance issues caused by API client throttling. ([#135903](https://github.com/kubernetes/kubernetes/pull/135903), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Ensures a couple of feature gates - ChangeContainerStatusOnKubeletRestart and StatefulSetSemanticRevisionComparison are visible from the "--help" in different components ([#135515](https://github.com/kubernetes/kubernetes/pull/135515), [@dims](https://github.com/dims)) [SIG Architecture]
- Fix a nil pointer dereference in Kubelet when handling pod updates of mirror pods with the NodeDeclaredFeatures feature gate enabled. ([#136037](https://github.com/kubernetes/kubernetes/pull/136037), [@pravk03](https://github.com/pravk03)) [SIG Node]
- Fix apiserver request latency annotation in the audit log when request took more than 500ms ([#135685](https://github.com/kubernetes/kubernetes/pull/135685), [@chaochn47](https://github.com/chaochn47)) [SIG API Machinery]
- Fix data race in kubelet container manager. ([#136206](https://github.com/kubernetes/kubernetes/pull/136206), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Fix data race in kubelet pod allocated resources. ([#136226](https://github.com/kubernetes/kubernetes/pull/136226), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Fix data race in kubelet status manager. ([#136205](https://github.com/kubernetes/kubernetes/pull/136205), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Fix issues where server side apply patches operations incorrectly treat empty arrays and maps as absent.
  Fix issue where client-go's `Extract{TypeName}()` and `Extract{TypeName}From() functions incorrectly treat empty arrays and maps as absent.
  Fix issue where client-go's `Extract{TypeName}()` and `Extract{TypeName}From() functions would incorrectly duplicate atomic elements from associative lists. ([#135391](https://github.com/kubernetes/kubernetes/pull/135391), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Network, Node, Scheduling and Storage]
- Fix log verbosity level in apiserver's unsafe delete authorization check that was incorrectly using Error level instead of Info level ([#136229](https://github.com/kubernetes/kubernetes/pull/136229), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- Fix queue hint for the interpodaffinity plugin in case target pod labels change ([#135394](https://github.com/kubernetes/kubernetes/pull/135394), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fix static pod status is always Init:0/1 if unable to get init container status from container runtime. ([#131317](https://github.com/kubernetes/kubernetes/pull/131317), [@bitoku](https://github.com/bitoku)) [SIG Node and Testing]
- Fix the log verbosity level of some non-error logs that were incorrectly logged at error level ([#136046](https://github.com/kubernetes/kubernetes/pull/136046), [@Tanner-Gladson](https://github.com/Tanner-Gladson)) [SIG API Machinery and Apps]
- Fix the log verbosity level of some non-error logs that were incorrectly logged at error level ([#136050](https://github.com/kubernetes/kubernetes/pull/136050), [@ShaanveerS](https://github.com/ShaanveerS)) [SIG Apps and Storage]
- Fixed SELinux warning controller not to emit events for completed pods. ([#135629](https://github.com/kubernetes/kubernetes/pull/135629), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Storage and Testing]
- Fixed a bug causing clients to error out when decoding large CBOR encoded lists. ([#135340](https://github.com/kubernetes/kubernetes/pull/135340), [@ricardomaraschini](https://github.com/ricardomaraschini)) [SIG API Machinery]
- Fixed a bug in DeepEqualWithNilDifferentFromEmpty where empty slices/maps were incorrectly considered equal to non-empty ones due to using OR (||) instead of AND (&&) logic. This could cause managed fields timestamps to not update when the only change was adding or removing all elements from a list or map. ([#135636](https://github.com/kubernetes/kubernetes/pull/135636), [@mikecook](https://github.com/mikecook)) [SIG API Machinery]
- Fixed a bug in the `dra_operations_duration_seconds` metric where the `is_error` label was recording inverted values. Error operations now correctly report `is_error="true"`, and successful operations report `is_error="false"`. ([#135227](https://github.com/kubernetes/kubernetes/pull/135227), [@hime](https://github.com/hime)) [SIG Node]
- Fixed a bug that caused endpoint slice churn for headless services with no ports defined (#133474) ([#136502](https://github.com/kubernetes/kubernetes/pull/136502), [@tzneal](https://github.com/tzneal)) [SIG Network]
- Fixed a bug where `kubectl apply --dry-run=client` would only output server state instead of merged manifest values when the resource already exists. ([#135513](https://github.com/kubernetes/kubernetes/pull/135513), [@grandeit](https://github.com/grandeit)) [SIG CLI]
- Fixed a bug where the Gated pods metric was not updated when a Pod transitioned from Unschedulable to Gated during an update. ([#135368](https://github.com/kubernetes/kubernetes/pull/135368), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling]
- Fixed a bug where the `scheduler_unschedulable_pods` metric could be artificially inflated (leak) when a pod fails `PreEnqueue` plugins after being previously marked unschedulable. ([#135981](https://github.com/kubernetes/kubernetes/pull/135981), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling]
- Fixed a panic in `kubectl exec` when the terminal size queue delegate is uninitialized. ([#135918](https://github.com/kubernetes/kubernetes/pull/135918), [@MarcosDaNight](https://github.com/MarcosDaNight)) [SIG CLI]
- Fixed a panic in kubectl when processing pods with nil resource requests but populated container status resources. ([#136534](https://github.com/kubernetes/kubernetes/pull/136534), [@dmaizel](https://github.com/dmaizel)) [SIG CLI]
- Fixed a race condition in the CEL compiler that could occur when initializing composited policies concurrently. 
  
  ### Description
  Fixes a fatal crash (concurrent map read/write) in `NewCompositedCompilerFromTemplate`.
  
  The `NewCompositedCompilerFromTemplate` function previously performed a shallow copy of `CompositionEnv`, sharing the `MapType` pointer across all compilers. Under high concurrency, this caused a race condition when `FindStructFieldType` (reader) and `AddField` (writer) accessed `MapType.Fields` simultaneously, leading to an APIServer panic.
  
  This change implements a deep copy of the `Fields` map for each composition environment, ensuring thread safety.
  
  ### Issue
  Fixes #135757 ([#135759](https://github.com/kubernetes/kubernetes/pull/135759), [@Abhigyan-Shekhar](https://github.com/Abhigyan-Shekhar)) [SIG API Machinery and CLI]
- Fixed an issue in the Windows kube-proxy (winkernel) where IPv4 and IPv6 Service load balancers could be incorrectly shared, causing broken dual-stack Service behavior. The kube-proxy now tracks load balancers per IP family, enabling correct support for PreferDualStack and RequireDualStack Services on Windows nodes. ([#136241](https://github.com/kubernetes/kubernetes/pull/136241), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fixed issue where `kubectl run -i/-it` would miss container output written before the attach connection was established. ([#136010](https://github.com/kubernetes/kubernetes/pull/136010), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Fixed kubelet logging to properly respect verbosity levels. Previously, some debug/info messages using V().Error() would always be printed regardless of the configured log verbosity. ([#136028](https://github.com/kubernetes/kubernetes/pull/136028), [@thc1006](https://github.com/thc1006)) [SIG Node]
- Fixed queue hint for certain plugins on change to pods with nominated nodes ([#135392](https://github.com/kubernetes/kubernetes/pull/135392), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fixed queue hint for inter-pod anti-affinity in case deleted pod's anti-affinity matched the pending pod, which might have caused delays in scheduling. ([#135325](https://github.com/kubernetes/kubernetes/pull/135325), [@brejman](https://github.com/brejman)) [SIG Scheduling and Testing]
- Fixed volumeattachment cleanup in kube-controller-manager when CSI's attachRequired switches from true to false ([#129664](https://github.com/kubernetes/kubernetes/pull/129664), [@hkttty2009](https://github.com/hkttty2009)) [SIG Storage and Testing]
- Fixes a 1.29 regression in the apiserver_watch_events_sizes metric to report total outgoing watch traffic again ([#135367](https://github.com/kubernetes/kubernetes/pull/135367), [@mborsz](https://github.com/mborsz)) [SIG API Machinery]
- Fixes a 1.34 regression starting pods with environment variables with a value containing `$` followed by a multi-byte character ([#136325](https://github.com/kubernetes/kubernetes/pull/136325), [@AutuSnow](https://github.com/AutuSnow)) [SIG Architecture]
- Fixes a 1.34+ regression in ipvs and winkernel kube-proxy backends; these are now reverted back to their
  pre-1.34 behavior of regularly rechecking all of their rules even when no
  Services or EndpointSlices change. ([#135631](https://github.com/kubernetes/kubernetes/pull/135631), [@danwinship](https://github.com/danwinship)) [SIG Network and Windows]
- Fixes kube-proxy log spam when all of a Service's endpoints were unready. ([#136743](https://github.com/kubernetes/kubernetes/pull/136743), [@ansilh](https://github.com/ansilh)) [SIG Network]
- Kube-apiserver: setting `--audit-log-maxsize=0` now disables audit log rotation (the default remains `100` MB). In order to avoid outages due to filling disks with ever-growing audit logs, `--audit-log-maxage` now defaults to 366 (1 year) and `--audit-log-maxbackup` now defaults to 100. If retention of all rotated logs is desired, age and count-based pruning can be disabled by explicitly specifying `--audit-log-maxage=0` and `--audit-log-maxbackup=0`. ([#136478](https://github.com/kubernetes/kubernetes/pull/136478), [@kairosci](https://github.com/kairosci)) [SIG API Machinery]
- Kube-proxy now correctly handles the case where a pod IP gets assigned to
  a newly-created pod when the pod that previously had that IP has been
  terminated but is not yet fully deleted. ([#135593](https://github.com/kubernetes/kubernetes/pull/135593), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kubeadm: fix a bug where kubeadm upgrade is failed if the content of the `kubeadm-flags.env` file is `KUBELET_KUBEADM_ARGS=""` ([#136127](https://github.com/kubernetes/kubernetes/pull/136127), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: waiting for etcd learner member to be started before promoting during 'kubeadm join' ([#136014](https://github.com/kubernetes/kubernetes/pull/136014), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: when applying the overrides provided by the user using "extraArgs", do not sort the resulted list of arguments alpha-numerically. Instead, only sort the list of default arguments and keep the list of overrides unsorted. This allows finer control for flags which have an order that matters, such as, "--service-account-issuer" for kube-apiserver. ([#135400](https://github.com/kubernetes/kubernetes/pull/135400), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl: fixes kyaml output of `kubectl get ... --output-watch-events -o kyaml` ([#136110](https://github.com/kubernetes/kubernetes/pull/136110), [@liggitt](https://github.com/liggitt)) [SIG CLI]
- Kubelet(dra): correctly handles multiple ResourceClaims even if one is already prepared ([#135919](https://github.com/kubernetes/kubernetes/pull/135919), [@rogowski-piotr](https://github.com/rogowski-piotr)) [SIG Node and Testing]
- Kubelet: fix data race in volume manager's WaitForAllPodsUnmount that could cause errors to be lost during concurrent pod unmount operations. ([#135794](https://github.com/kubernetes/kubernetes/pull/135794), [@AutuSnow](https://github.com/AutuSnow)) [SIG Node and Storage]
- Kubelet: fixed reloading of kubelet server certificate files when they are changed on disk, and kubelet is dialed by IP address instead of DNS/hostname ([#133654](https://github.com/kubernetes/kubernetes/pull/133654), [@kwohlfahrt](https://github.com/kwohlfahrt)) [SIG API Machinery, Auth, Node and Testing]
- Optimized kube-proxy conntrack cleanup logic, reducing the time complexity of deleting stale UDP entries. This significantly improves performance when there are many stale connections to clean up. ([#135511](https://github.com/kubernetes/kubernetes/pull/135511), [@aojea](https://github.com/aojea)) [SIG Network]
- ReadWriteOncePod preemption e2e test no longer causes other random e2e tests to flake randomly. ([#135623](https://github.com/kubernetes/kubernetes/pull/135623), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]
- Sort runtime handlers list coming from the CRI runtime ([#135358](https://github.com/kubernetes/kubernetes/pull/135358), [@harche](https://github.com/harche)) [SIG Node]
- StatefulSets should always count `.status.availableReplicas` at the correct time without a delay. This results in faster progress of StatefulSet rollout. ([#135428](https://github.com/kubernetes/kubernetes/pull/135428), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- The kubelet plugin manager now properly handles plugin registration failures by removing failed plugins from the actual state and retrying with exponential backoff (initial delay 500ms, doubling each failure up to ~2 minutes maximum) to protect against broken plugins causing denial of service while still allowing recovery from transient failures. ([#133335](https://github.com/kubernetes/kubernetes/pull/133335), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Storage and Testing]
- The nftables mode of kube-proxy now uses less CPU when loading
  very large rulesets. ([#135800](https://github.com/kubernetes/kubernetes/pull/135800), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Updated `NodeResourcesBalancedAllocation` scoring algorithm to align with the documentation. The score will now take into consideration both balance with and without the requested pod. Previous algorithm only considered balance with the requested pod. This can change the scheduling decisions in some cases. ([#135573](https://github.com/kubernetes/kubernetes/pull/135573), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- When use kubectl command to delete multiple sts pods, the kubectl command deletes pods and exits normally. ([#135563](https://github.com/kubernetes/kubernetes/pull/135563), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG CLI, Network and Node]

### Other (Cleanup or Flake)

- Added missing tests for client-go metrics ([#136052](https://github.com/kubernetes/kubernetes/pull/136052), [@sreeram-venkitesh](https://github.com/sreeram-venkitesh)) [SIG Architecture and Instrumentation]
- Adds audit-id to 'Starting watch' log line ([#136084](https://github.com/kubernetes/kubernetes/pull/136084), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Adds explicit logging when WatchList requests complete their initial listing phase. ([#136085](https://github.com/kubernetes/kubernetes/pull/136085), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Client-go: Reflector no longer gets confused about the resource version it should use to restart a watch while receiving synthetic ADDED events at the beginning of a watch from resourceVersion "0" or "". ([#136583](https://github.com/kubernetes/kubernetes/pull/136583), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Client-go: fake client-go (i.e. anything using k8s.io/client-go/testing) now supports separate List+Watch calls  with checking of ResourceVersion in the Watch call. This closes a race condition where creating an object directly after an informer cache has synced (= List call completed) and before the Watch call completed would cause that object to not be sent to the informer. A visible side-effect of adding that support is that List meta data contains a ResourceVersion (starting at "1" for the empty set, incremented by one  for each add/update) and that Watch may return objects where it previously didn't.
  
  Note that this List+Watch is not to be confused with the ListWatch feature, which uses a single call. That feature is still not supported by fake client-go. ([#136143](https://github.com/kubernetes/kubernetes/pull/136143), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth and CLI]
- DRA device taint eviction: the controller might have reported "1 pod needs to be evicted in 1 namespace. 1 pod evicted since starting the controller." when only a single pod is involved, depending on timing (pod evicted, informer cache not updated yet). It would eventually arrive at the correct "1 pod evicted since starting the controller.", but now it tries harder to avoid the confusing intermediate state by delaying the status update after eviction. ([#135611](https://github.com/kubernetes/kubernetes/pull/135611), [@Karthik-K-N](https://github.com/Karthik-K-N)) [SIG Apps and Scheduling]
- DRA: Fixed Kubelet admission to correctly handle DRA-backed extended resources, allowing pods to be admitted even when these resources are not present in the node's allocatable capacity. ([#135725](https://github.com/kubernetes/kubernetes/pull/135725), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Scheduling and Testing]
- Enables YAML support for statusz and flagz. ([#135309](https://github.com/kubernetes/kubernetes/pull/135309), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation and Testing]
- Kubeadm: removed the cleanup of the "--pod-infra-container-image" kubelet flag from the "/var/lib/kubelet/kubeadm-flags.env" on upgrade. This cleanup was necessary when upgrading to 1.35. ([#135807](https://github.com/kubernetes/kubernetes/pull/135807), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: removed usage of the deprecated flags '--experimental-initial-corrupt-check' and '--experimental-watch-progress-notify-interval' if the etcd version is < 3.6.0. In this version of kubeadm, etcd < 3.6.0 is no longer supported in terms of the k8s / etcd version mapping. These deprecated flags have been replaced by '--feature-gates=InitialCorruptCheck=true' and '--watch-progress-notify-interval'. ([#135701](https://github.com/kubernetes/kubernetes/pull/135701), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Lock the `DisableNodeKubeProxyVersion` feature gate to be enabled by default. ([#136673](https://github.com/kubernetes/kubernetes/pull/136673), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG CLI and Network]
- Remove WatchFromStorageWithoutResourceVersion feature gate ([#136066](https://github.com/kubernetes/kubernetes/pull/136066), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Remove event listing behavior when describing a removed pod from file. ([#135281](https://github.com/kubernetes/kubernetes/pull/135281), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Renamed PodGroupInfo to PodGroupState, which can break custom scheduler plugins that use Handle.WorkloadManager ([#136344](https://github.com/kubernetes/kubernetes/pull/136344), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Set InOrderInformers to GA via the usage of RealFIFO, this means that DeltaFIFO will gradually be deprecated in favor of RealFIFO in internal implementations. ([#136601](https://github.com/kubernetes/kubernetes/pull/136601), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Updated cri-tools to v1.35.0. ([#135694](https://github.com/kubernetes/kubernetes/pull/135694), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- Updates the etcd client library to v3.6.6 ([#135331](https://github.com/kubernetes/kubernetes/pull/135331), [@yashsingh74](https://github.com/yashsingh74)) [SIG API Machinery, Auth, Cloud Provider, Etcd, Node and Scheduling]
- Updates the etcd client library to v3.6.7 ([#136407](https://github.com/kubernetes/kubernetes/pull/136407), [@ivanvc](https://github.com/ivanvc)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]

## Dependencies

### Added
- buf.build/go/protovalidate: v0.12.0
- github.com/cenkalti/backoff/v5: [v5.0.3](https://github.com/cenkalti/backoff/tree/v5.0.3)
- github.com/moby/moby/api: [v1.52.0](https://github.com/moby/moby/tree/api/v1.52.0)
- github.com/moby/moby/client: [v0.2.1](https://github.com/moby/moby/tree/client/v0.2.1)
- go.opentelemetry.io/otel/exporters/stdout/stdouttrace: v1.39.0
- gonum.org/v1/gonum: v0.16.0

### Changed
- buf.build/gen/go/bufbuild/protovalidate/protocolbuffers/go: 63bb56e → 8976f5b
- cloud.google.com/go/compute/metadata: v0.7.0 → v0.9.0
- cyphar.com/go-pathrs: v0.2.1 → v0.2.2
- github.com/GoogleCloudPlatform/opentelemetry-operations-go/detectors/gcp: [v1.26.0 → v1.30.0](https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/compare/detectors/gcp/v1.26.0...detectors/gcp/v1.30.0)
- github.com/Microsoft/hnslib: [v0.1.1 → v0.1.2](https://github.com/Microsoft/hnslib/compare/v0.1.1...v0.1.2)
- github.com/alecthomas/units: [b94a6e3 → 0f3dac3](https://github.com/alecthomas/units/compare/b94a6e3...0f3dac3)
- github.com/cncf/xds/go: [2f00578 → 0feb691](https://github.com/cncf/xds/compare/2f00578...0feb691)
- github.com/containerd/containerd/api: [v1.9.0 → v1.10.0](https://github.com/containerd/containerd/compare/api/v1.9.0...api/v1.10.0)
- github.com/coredns/corefile-migration: [v1.0.29 → v1.0.30](https://github.com/coredns/corefile-migration/compare/v1.0.29...v1.0.30)
- github.com/coreos/go-oidc: [v2.3.0+incompatible → v2.5.0+incompatible](https://github.com/coreos/go-oidc/compare/v2.3.0...v2.5.0)
- github.com/coreos/go-systemd/v22: [v22.5.0 → v22.7.0](https://github.com/coreos/go-systemd/compare/v22.5.0...v22.7.0)
- github.com/cyphar/filepath-securejoin: [v0.6.0 → v0.6.1](https://github.com/cyphar/filepath-securejoin/compare/v0.6.0...v0.6.1)
- github.com/davecgh/go-spew: [v1.1.1 → d8f796a](https://github.com/davecgh/go-spew/compare/v1.1.1...d8f796a)
- github.com/docker/go-connections: [v0.5.0 → v0.6.0](https://github.com/docker/go-connections/compare/v0.5.0...v0.6.0)
- github.com/emicklei/go-restful/v3: [v3.12.2 → v3.13.0](https://github.com/emicklei/go-restful/compare/v3.12.2...v3.13.0)
- github.com/envoyproxy/go-control-plane/envoy: [v1.32.4 → v1.35.0](https://github.com/envoyproxy/go-control-plane/compare/envoy/v1.32.4...envoy/v1.35.0)
- github.com/envoyproxy/go-control-plane: [v0.13.4 → 75eaa19](https://github.com/envoyproxy/go-control-plane/compare/v0.13.4...75eaa19)
- github.com/go-jose/go-jose/v4: [v4.0.4 → v4.1.3](https://github.com/go-jose/go-jose/compare/v4.0.4...v4.1.3)
- github.com/godbus/dbus/v5: [v5.1.0 → v5.2.2](https://github.com/godbus/dbus/compare/v5.1.0...v5.2.2)
- github.com/golang-jwt/jwt/v5: [v5.2.2 → v5.3.0](https://github.com/golang-jwt/jwt/compare/v5.2.2...v5.3.0)
- github.com/golang/glog: [v1.2.4 → v1.2.5](https://github.com/golang/glog/compare/v1.2.4...v1.2.5)
- github.com/google/cadvisor: [v0.53.0 → v0.56.0](https://github.com/google/cadvisor/compare/v0.53.0...v0.56.0)
- github.com/grpc-ecosystem/go-grpc-middleware/providers/prometheus: [v1.0.1 → v1.1.0](https://github.com/grpc-ecosystem/go-grpc-middleware/compare/providers/prometheus/v1.0.1...providers/prometheus/v1.1.0)
- github.com/grpc-ecosystem/go-grpc-middleware/v2: [v2.3.0 → v2.3.3](https://github.com/grpc-ecosystem/go-grpc-middleware/compare/v2.3.0...v2.3.3)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.26.3 → v2.27.4](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.26.3...v2.27.4)
- github.com/onsi/ginkgo/v2: [v2.27.2 → v2.27.4](https://github.com/onsi/ginkgo/compare/v2.27.2...v2.27.4)
- github.com/onsi/gomega: [v1.38.2 → v1.39.0](https://github.com/onsi/gomega/compare/v1.38.2...v1.39.0)
- github.com/opencontainers/cgroups: [v0.0.3 → v0.0.6](https://github.com/opencontainers/cgroups/compare/v0.0.3...v0.0.6)
- github.com/opencontainers/runc: [v1.3.0 → v1.4.0](https://github.com/opencontainers/runc/compare/v1.3.0...v1.4.0)
- github.com/opencontainers/runtime-spec: [v1.2.1 → v1.3.0](https://github.com/opencontainers/runtime-spec/compare/v1.2.1...v1.3.0)
- github.com/opencontainers/selinux: [v1.13.0 → v1.13.1](https://github.com/opencontainers/selinux/compare/v1.13.0...v1.13.1)
- github.com/pmezard/go-difflib: [v1.0.0 → 5d4384e](https://github.com/pmezard/go-difflib/compare/v1.0.0...5d4384e)
- github.com/prometheus/common: [v0.66.1 → v0.67.5](https://github.com/prometheus/common/compare/v0.66.1...v0.67.5)
- github.com/prometheus/procfs: [v0.16.1 → v0.19.2](https://github.com/prometheus/procfs/compare/v0.16.1...v0.19.2)
- github.com/spiffe/go-spiffe/v2: [v2.5.0 → v2.6.0](https://github.com/spiffe/go-spiffe/compare/v2.5.0...v2.6.0)
- go.etcd.io/etcd/api/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/client/pkg/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/client/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/pkg/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/server/v3: v3.6.5 → v3.6.7
- go.opentelemetry.io/auto/sdk: v1.1.0 → v1.2.1
- go.opentelemetry.io/contrib/detectors/gcp: v1.34.0 → v1.38.0
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.44.0 → v0.64.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.60.0 → v0.63.0
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.61.0 → v0.64.0
- go.opentelemetry.io/contrib/propagators/b3: v1.19.0 → v1.39.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.34.0 → v1.39.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.34.0 → v1.39.0
- go.opentelemetry.io/otel/metric: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel/sdk/metric: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel/sdk: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel/trace: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel: v1.36.0 → v1.39.0
- go.opentelemetry.io/proto/otlp: v1.5.0 → v1.9.0
- go.uber.org/zap: v1.27.0 → v1.27.1
- golang.org/x/crypto: v0.45.0 → v0.47.0
- golang.org/x/exp: 8a7402a → 944ab1f
- golang.org/x/mod: v0.29.0 → v0.32.0
- golang.org/x/net: v0.47.0 → v0.49.0
- golang.org/x/oauth2: v0.30.0 → v0.34.0
- golang.org/x/sync: v0.18.0 → v0.19.0
- golang.org/x/sys: v0.38.0 → v0.40.0
- golang.org/x/telemetry: 078029d → 8fff8a5
- golang.org/x/term: v0.37.0 → v0.39.0
- golang.org/x/text: v0.31.0 → v0.33.0
- golang.org/x/time: v0.9.0 → v0.14.0
- golang.org/x/tools: v0.38.0 → v0.40.0
- google.golang.org/genproto/googleapis/api: a0af3ef → 99fd39f
- google.golang.org/genproto/googleapis/rpc: 200df99 → 99fd39f
- google.golang.org/grpc: v1.72.2 → v1.78.0
- google.golang.org/protobuf: v1.36.8 → v1.36.11
- k8s.io/kube-openapi: 589584f → a19766b
- k8s.io/utils: bc988d5 → 914a6e7
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.31.2 → v0.34.0
- sigs.k8s.io/structured-merge-diff/v6: v6.3.0 → v6.3.1

### Removed
- github.com/armon/circbuf: [5111143](https://github.com/armon/circbuf/tree/5111143)
- github.com/bufbuild/protovalidate-go: [v0.9.1](https://github.com/bufbuild/protovalidate-go/tree/v0.9.1)
- github.com/docker/docker: [v28.2.2+incompatible](https://github.com/docker/docker/tree/v28.2.2)
- github.com/gregjones/httpcache: [901d907](https://github.com/gregjones/httpcache/tree/901d907)
- github.com/grpc-ecosystem/go-grpc-prometheus: [v1.2.0](https://github.com/grpc-ecosystem/go-grpc-prometheus/tree/v1.2.0)
- github.com/karrick/godirwalk: [v1.17.0](https://github.com/karrick/godirwalk/tree/v1.17.0)
- github.com/libopenstorage/openstorage: [v1.0.0](https://github.com/libopenstorage/openstorage/tree/v1.0.0)
- github.com/moby/sys/atomicwriter: [v0.1.0](https://github.com/moby/sys/tree/atomicwriter/v0.1.0)
- github.com/mohae/deepcopy: [c48cc78](https://github.com/mohae/deepcopy/tree/c48cc78)
- github.com/morikuni/aec: [v1.0.0](https://github.com/morikuni/aec/tree/v1.0.0)
- github.com/mrunalp/fileutils: [v0.5.1](https://github.com/mrunalp/fileutils/tree/v0.5.1)
- github.com/zeebo/errs: [v1.4.0](https://github.com/zeebo/errs/tree/v1.4.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp: v1.27.0
- gotest.tools/v3: v3.0.2