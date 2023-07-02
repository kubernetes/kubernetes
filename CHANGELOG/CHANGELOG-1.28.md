<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.28.0-alpha.3](#v1280-alpha3)
  - [Downloads for v1.28.0-alpha.3](#downloads-for-v1280-alpha3)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.28.0-alpha.2](#changelog-since-v1280-alpha2)
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
- [v1.28.0-alpha.2](#v1280-alpha2)
  - [Downloads for v1.28.0-alpha.2](#downloads-for-v1280-alpha2)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.28.0-alpha.1](#changelog-since-v1280-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.28.0-alpha.1](#v1280-alpha1)
  - [Downloads for v1.28.0-alpha.1](#downloads-for-v1280-alpha1)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.27.0](#changelog-since-v1270)
  - [Changes by Kind](#changes-by-kind-2)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-2)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)

<!-- END MUNGE: GENERATED_TOC -->

# v1.28.0-alpha.3


## Downloads for v1.28.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes.tar.gz) | 1479e19873837e09f08f4c14d4b7587b1bc40d0b0d3214637311b63068301d34a63663f5d13b8ec62c81095a30eef1e8589633c630fe613eb825eb4afa0ddeb9
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-src.tar.gz) | 1f5d3486f15574220d3c5a5d7c7a2b7412347418650deeda326eb513ba2bb43c197e089999756eee09ac4b55dcafc18eef96c6e15e20fcc9b91183a35f224cc1

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | 4e5139d5a5a98343ded5b506451c3b9b052803a2e2ebb4e2328e17edbbde56dde749407ef8fc816283a1a1b1f80939a76b7c64a09a9496d4448ec47fe34cd95b
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | c58fdf32d3f1b411e10a7ede56115020c7a4c50912a899f06cf94d2f06a5e24a21df1deda6eb0e87f70d88afec46186f64dd18bb1b26b94b24b01059aef88b2c
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 26d3d2663de49063d02c682557dce616f5630b69a1ca219024ccd3405081193732bee42d24d9e5f5cf9fcd214da50defd64c141775a6aea372dbe3e9793e4547
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | f9c29ea075c4b842c39587e911b83e2c798d6d69ed5f4601ce17ce3c33bb9db8f511b3d2149dfe3d86576763e92e0b6eb6d3aa1b12bc04868fb6c8a013a2fdd1
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | be984144ebe8c2f3156b3d13aaebd362543399f4c91f8f18b1936999327468ccb5af7d95723280000b627e2f81fb119d6e175cc305391ffd83b3b632eb3c30df
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | bfc44ed472351dee434984e49be2cc9959df2f98048beeb1ad85a50bfe833a5396bb3f5c7f81ced2048de269bd60f6d4cf55944c0d44d288fb858b5552354389
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | d7ee2633c172442a04d317d4d935b55aa892131f716308b78d0f7b3cb5d2bfa9069f7b55ec837e4ff9a260717313e7fc4c7ca4a931f9f2b36768a9c54593ba45
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 4621ed7a77990430cbb03a14f85780f059a408a567c26c14eb7c7127300f4e1ac1b2013206f00a92e35da9f1142909745262ad43a8026a9ebe31cf423dbb3b90
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 6526de9f4ccd6f036152d73ebbf3a2b626ca7291c85d5b46977e8a8643ff80616665f4bd8b009310cc887f7d5e1827045a0b2489dd19621338e727e18cca4097
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 784e8d07b1db0331f454faac34b279bffba16c11f98ba25aa54f64eb6a0d9d55f0e8221828aa4e6e4d0d6bfb66a3c22a0d8a66d6287b37a33890a960d25fc54e
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | 523d2809d4ff4616ffd94e0868ac15ee10f87bee65760741b09ba2a04626e3ce226a5fc44a1bddb32835403ee374a6c12afa74ec0d2b9441304a11d923f43cee

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 5c2716137752b514fe7a8b23291cc7a661984deaf642d111e65690e928ecfbb6b5168b1f043847f3d4689cafbbdf5bb6fb5c97d2fd6f5f83c5b6cc701a4f388f
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 814e4ec6b46960e90ccce6ef1c234666f372b2b41583ead530b4163b829c4bb9001df822daf55493e54ad749076ab5391ebde7261d73f4e524f6cace402d49c3
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | 8a7749282ba4d3df877097dc07843112e815879ee911379d02ed33d5aab59cb60ffd27127ebd94879bef45d9534470d6aa7e48d71ecb455e3055c2dbb169fcdb
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 031a55294a06e5773d3c277dcab6b1e32c5e6409448015282275cb62e5bafb6a10845f0de3d37d501067f3f4176f48d04cafdd4eb96f6f7e30c7274ce3adaec6

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | 488d426c09f92a6f30d77352af7154b9fda8758a946566b31d855a4c30f82ccdb071d40708f96e78f9bf5ca961be27fa67ecc833101818e43da19bc967c9eb28
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | e5f6914bd6e393915ea814401c09f4b0accc3454087091e60b01ea7c87aa64edd1a0af954cd1a90a3dc83c5559772d1ba5a8ff04134638e911d972947e3e94a0
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 459453ad83c9cdd2568e43ed988ec2635b5413b435e587e8a6394feb7d49c550622c491f337c48f90ad32cc02b7ef47e24069f09737817a1c37f1a8930d6c5c2
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | e5a6ff5bb58275720c18b6181ffb9d135dcfaf05a229f87787538775e0280543db43e7218168f0a2de8a74acca6899dccd1cce88780b086ce6a96fb3e0368870
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | 7cc51923e34cfeb00681e7c2d26dff5c442b39d0000f11ad30ec5c843d80847a0353b3b4e6e5a228ad5cae7f33dfc0d4c38bf4b843b98b8115d28c3dc683dd9e

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.28.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.28.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.28.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.28.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.28.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.28.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.28.0-alpha.2

## Changes by Kind

### Deprecation

- Kube-controller-manager deprecate `--volume-host-cidr-denylist` and `--volume-host-allow-local-loopback` flags. ([#118128](https://github.com/kubernetes/kubernetes/pull/118128), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Network, Node, Storage and Testing]
- Kubelet: The `--azure-container-registry-config` flag has been deprecated and will be removed in a future release, please use `--image-credential-provider-config` and `--image-credential-provider-bin-dir` to setup acr credential provider instead. ([#118596](https://github.com/kubernetes/kubernetes/pull/118596), [@SataQiu](https://github.com/SataQiu)) [SIG Node]

### API Change

- ACTION_REQUIRED
  When an Indexed Job has a number of completions higher than 10^5 and parallelism higher than 10^4, and a big number of Indexes fail, Kubernetes might not be able to track the termination of the Job. Kubernetes now emits a warning, at Job creation, when the Job manifest exceeds both of these limits. ([#118420](https://github.com/kubernetes/kubernetes/pull/118420), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps]
- Expose rest.DefaultServerUrlFor function ([#118055](https://github.com/kubernetes/kubernetes/pull/118055), [@timofurrer](https://github.com/timofurrer)) [SIG API Machinery]
- If using cgroups v2, then the cgroup aware OOM killer will be enabled for container cgroups via  `memory.oom.group` .  This causes processes within the cgroup to be treated as a unit and killed simultaneously in the event of an OOM kill on any process in the cgroup. ([#117793](https://github.com/kubernetes/kubernetes/pull/117793), [@tzneal](https://github.com/tzneal)) [SIG Apps, Node and Testing]
- Update the comment about the feature-gate level for PodFailurePolicy from alpha to beta ([#118278](https://github.com/kubernetes/kubernetes/pull/118278), [@mimowo](https://github.com/mimowo)) [SIG Apps]

### Feature

- Add '--concurrent-cron-job-syncs' flag for kube-controller-manager to set the number of workers for cron job controller ([#117550](https://github.com/kubernetes/kubernetes/pull/117550), [@borgerli](https://github.com/borgerli)) [SIG Apps]
- Client-go: make generated CA certificates valid 1 hour in the past (NewSelfSignedCACert). Applies to CA certificates and other certificates generated by kubeadm. ([#118631](https://github.com/kubernetes/kubernetes/pull/118631), [@champtar](https://github.com/champtar)) [SIG Auth]
- Fixes the alpha `CloudDualStackNodeIPs` feature. ([#118329](https://github.com/kubernetes/kubernetes/pull/118329), [@danwinship](https://github.com/danwinship)) [SIG Network and Node]
- Kubelet: un-deprecate --provider-id flag ([#116530](https://github.com/kubernetes/kubernetes/pull/116530), [@pacoxu](https://github.com/pacoxu)) [SIG Node]
- Migrated the Job controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116910](https://github.com/kubernetes/kubernetes/pull/116910), [@fatsheep9146](https://github.com/fatsheep9146)) [SIG API Machinery, Apps and Testing]
- Rename PodHasNetwork to PodReadyToStartContainers ([#117702](https://github.com/kubernetes/kubernetes/pull/117702), [@kannon92](https://github.com/kannon92)) [SIG Node and Testing]

### Bug or Regression

- CI job `ci-kubernetes-node-arm64-ubuntu-serial` will test node e2e on arm64, `use-dockerized-build` and `target-build-arch` are required to run this job. ([#118567](https://github.com/kubernetes/kubernetes/pull/118567), [@chendave](https://github.com/chendave)) [SIG Node and Testing]
- Fix Cronjob status.lastSuccessfulTime not populated by a manually triggered job ([#118530](https://github.com/kubernetes/kubernetes/pull/118530), [@carlory](https://github.com/carlory)) [SIG CLI]
- Fix component status calling etcd health endpoint over http which exposed kubernetes to the risk of complete watch starvation and is inconsistent with other etcd probing done by kube-apiserver. ([#118460](https://github.com/kubernetes/kubernetes/pull/118460), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Fixed the preStop hook will block the pod termination grace period ([#115835](https://github.com/kubernetes/kubernetes/pull/115835), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node and Testing]
- Users will no longer see an error for failed events caused due to terminating namespace. ([#114849](https://github.com/kubernetes/kubernetes/pull/114849), [@padlar](https://github.com/padlar)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Kube-proxy: remove log warning about not using config file ([#118115](https://github.com/kubernetes/kubernetes/pull/118115), [@TommyStarK](https://github.com/TommyStarK)) [SIG Network]
- Make Job controller batching of syncJob invocations enabled unconditionally (it was conditional on JobReadyPods feature before).
  
  Also, Job controller's constants for default backoff and maximal backoff are lowered down to 1s (from 10s) and 1min (from 6min), respectively. These constants are used to determine the backoff delay for the next Job controller sync in case of a request failure. ([#118615](https://github.com/kubernetes/kubernetes/pull/118615), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- Migrated the interpodaffinity scheduler plugin to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116635](https://github.com/kubernetes/kubernetes/pull/116635), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]

## Dependencies

### Added
- github.com/alecthomas/kingpin/v2: [v2.3.1](https://github.com/alecthomas/kingpin/v2/tree/v2.3.1)
- github.com/xhit/go-str2duration: [v1.2.0](https://github.com/xhit/go-str2duration/tree/v1.2.0)

### Changed
- github.com/alecthomas/units: [f65c72e → b94a6e3](https://github.com/alecthomas/units/compare/f65c72e...b94a6e3)
- github.com/go-kit/log: [v0.2.0 → v0.2.1](https://github.com/go-kit/log/compare/v0.2.0...v0.2.1)
- github.com/kr/pretty: [v0.3.0 → v0.3.1](https://github.com/kr/pretty/compare/v0.3.0...v0.3.1)
- github.com/matttproud/golang_protobuf_extensions: [v1.0.2 → v1.0.4](https://github.com/matttproud/golang_protobuf_extensions/compare/v1.0.2...v1.0.4)
- github.com/prometheus/client_golang: [v1.14.0 → v1.16.0](https://github.com/prometheus/client_golang/compare/v1.14.0...v1.16.0)
- github.com/prometheus/common: [v0.37.0 → v0.42.0](https://github.com/prometheus/common/compare/v0.37.0...v0.42.0)
- github.com/prometheus/procfs: [v0.8.0 → v0.10.1](https://github.com/prometheus/procfs/compare/v0.8.0...v0.10.1)
- golang.org/x/sync: v0.1.0 → v0.2.0

### Removed
_Nothing has changed._



# v1.28.0-alpha.2


## Downloads for v1.28.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes.tar.gz) | 3ef803c3df0a722e4fdaa045fa7aca8c39892916b0788e0c3216747397fe4e6e8fc02483c7e1a7280ed9f2716324c067cd806037dbf9b635e268f8b62db43841
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-src.tar.gz) | c7594dfdefc92fabd6ffeef08fb98e6686c6bbf8dc952197e6629838e645c83d4b7d375e078dc3f61f6cdbd425ca68eb88f0ffdbb1e3d2e034af9efeb8f7f34b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | 66e80a10b94d81ca08bd3eb94168afa429c3bc2b036e80bd55e4d0c93b4eb9e645d575ff520e6961368d2559e9bec786fbf115d0d902af7a3b0eae43213fb9e2
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 0654d9ec7234a042d08db30cf01b69f66889451219a57a9e8a1dda2e11e7767e0ea1a171c2f16957bc6fedf1777bea455a03f453b9b06046d1588824ddb72627
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-linux-386.tar.gz) | cdb3fdaacad12e8910f7b3dc26cd4bb684fcfbf8e9d0e594a1dda3da8ca804a9f69d0d030893b15cdd4c0daf613b3efdf8c2be54aaa8406764aaa17550df2fb8
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | ceb64baa175b6444f719311f76b5de7cc1cfcad13650d3a3dc50fa5fbb6f5c724aa7fc63aa4df27fc1e3b58c419a3d1b3aaa75fb415c3709bd4f4c7253b4d99a
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | c9a9f307fe69ce45a1f31ff8f68a158c08de202676952fc243cba4bca4f66f020fc68c7f36191d731ac536f9ff96a0e8c75a7ecc2884eac59468b92af7f72c5c
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | b5ee31bfb316559a9bec2bf7c5c2e555d1cfc07a307e1bfd2f4022b3a9988f2b6ae2e1189079a81dac0d3cc46d8982e25eafb2aaad5500be1c872313427e3544
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | d91e713944e21cd7729933fcb13118a3aae1a59e9809f7c8c23c047a55ceb9428459758b27d1496be3909d5097cdf5bcdc9089f6dbbfca6154a763ede0fc38d1
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | b34d9111b967d295092a57c9838f97400f526c9ad8086c562e090b3e92d171159dcece2576dfd0345c59c3adab06f5b01ff51a840e5997764b5bdc15df44c00f
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 82dc590e72c866e561617aed045b4377b90e4eb42574e998c71fe287202acf37ff0a3f1d55545d3465b2410108e0b4a4ab921a0fd749a99efc36723fe945c1f6
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | a7aba1527252e942bb3a88df21a0c7a6c9f9fdbfbb1e5b2e6e28c431a413fbb51d1e50682f05a353fc6f997ed7b5e7b229f08aea0d12b92631c1b8152b029fce
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 728574089c01b253a1e81da19680ffaf58a88780653583d1af35ad68ee50c05cbbc21a7ca1b33c293d6db0eaa1de04988b8540467f3b4dfa22e48d5ca7a93604

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | fe373323bd6940fc1801d04f3fd1f574a85d103ce6526f58bc894cf0bbbbe387a26f4a4f248249aec36f88bc15c1eceb67b85af9dc876b139bad245b0551f219
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 4e76b003da0dedc9b457cb47fbcb4e8f719144aced497b6ebaaf9c515c8125a72e72c3a059757de3cdfc177dd4e3b18368ba47fd2fefb9cb2e4bc6a23c73f802
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | c17f57f13d1393d4cf32a4ac0c122b6ebe400d88a26b3bd2eae9c6b4b4ed9f26a23ab99032847fb758bc4cbfa09a4dc7b843603d81659d0b79d42953efeca15b
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 36ce4946d3879c1e4b5e6e928be8da0eb9f146c96633d1bcb7d6541fa158f5a176d8174fba113d17bacff0db2debcedad73aed2d4054e3cc65bbf29256e62942

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 34274b82a809fea6bdbfc1602dc66929efa71da0b3e510880a7b498253aa379a5ddc1fb1875666d2e6b5bbdd6473a7471a6b80e46526d57358ae9f97ec0e1904
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 610dc57e34222170c57b8c31ae15392cb960c2bff750f57acf0925d1597b54326c2f78b610c0f79f9762ba67a3af077f985b0ed8ef275e0d81794775b6448d58
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 2b65b45735680dab34e462f45c34c64c809bd8e80b2a45297446b1489dc974a54f8cd11364192e090e98edd987b4d7fb81b81d439987b1131d64a943d94ebb8f
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | b3cba42fa9532bb9c7d80e31b1be8cb904bad1d32d6f3c651a690be1435d7f8d610bb203c89c420e96997a11a51b3e46f05fe626d2818fcc1bd3cef7d5a15b92
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 0b00eac88f2220dbb7f6243f9d4de41789bf3f38c14eec1c44bd48b9a067a43ef4cc9c468dc5869598d9f5ba06828b8bc5b7ea5eaa42faabaf584a8bdcb9ea29

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.28.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.28.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.28.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.28.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.28.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.28.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.28.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - CephFS volume plugin ( `kubernetes.io/cephfs`) has been deprecated in this release and will be removed in a subsequent release. Alternative is to use CephFS CSI driver (https://github.com/ceph/ceph-csi/) in your Kubernetes Cluster. ([#118143](https://github.com/kubernetes/kubernetes/pull/118143), [@humblec](https://github.com/humblec)) [SIG Storage]
 
## Changes by Kind

### Feature

- Introduce support for CEL optionals (see [CEL spec proposal 246](https://github.com/google/cel-spec/wiki/proposal-246)).
  This feature will not be fully enabled until a future Kubernetes release (likely to be v1.29), but is added in v1.28 to enable
  safe rollback on downgrade. ([#118339](https://github.com/kubernetes/kubernetes/pull/118339), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Auth, Cloud Provider and Testing]
- Kubernetes is now built with Go 1.20.5 ([#118507](https://github.com/kubernetes/kubernetes/pull/118507), [@jeremyrickard](https://github.com/jeremyrickard)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Storage and Testing]
- Promote ServiceNodePortStaticSubrange to beta and it will be enabled by default ([#117877](https://github.com/kubernetes/kubernetes/pull/117877), [@xuzhenglun](https://github.com/xuzhenglun)) [SIG Network]
- The `ExpandedDNSConfig` feature has graduated to GA. 'ExpandedDNSConfig' feature was locked to default value and will be removed in v1.30. If you were setting this feature gate explicitly, please remove it now. ([#116741](https://github.com/kubernetes/kubernetes/pull/116741), [@gjkim42](https://github.com/gjkim42)) [SIG Apps, Network and Node]
- The helping message of commands which have sub-commands is now clearer and more instructive. It will show the full command instead of 'kubectl <command> --help ...'
  
  Changed 'kubectl create secret --help' description. There will be a short introduction to the three secret types and clearer guidance on how to use the command. ([#117930](https://github.com/kubernetes/kubernetes/pull/117930), [@LronDC](https://github.com/LronDC)) [SIG CLI and Testing]
- Updated distroless I-tables to use registry.k8s.io/build-image/distroless-iptables:v0.2.5 ([#118541](https://github.com/kubernetes/kubernetes/pull/118541), [@jeremyrickard](https://github.com/jeremyrickard)) [SIG Testing]

### Bug or Regression

- Compute the backoff delay more accurately for deleted pods ([#118413](https://github.com/kubernetes/kubernetes/pull/118413), [@mimowo](https://github.com/mimowo)) [SIG Apps]
- Ensure Job status updates are batched by 1s. This fixes an unlikely scenario when a sequence of immediately 
  completing pods could trigger a sequence of non-batched Job status updates. ([#118470](https://github.com/kubernetes/kubernetes/pull/118470), [@mimowo](https://github.com/mimowo)) [SIG Apps]
- Fix a race condition in kube-proxy when using LocalModeNodeCIDR to avoid dropping Services traffic if the object node is recreated when kube-proxy is starting ([#118499](https://github.com/kubernetes/kubernetes/pull/118499), [@aojea](https://github.com/aojea)) [SIG Network]
- Fixed a race condition between `Run()` and `SetTransform()` and `SetWatchErrorHandler()` in shared informers. ([#117870](https://github.com/kubernetes/kubernetes/pull/117870), [@howardjohn](https://github.com/howardjohn)) [SIG API Machinery]
- Fixes bug where explain was not properly respecting jsonpaths ([#115694](https://github.com/kubernetes/kubernetes/pull/115694), [@mpuckett159](https://github.com/mpuckett159)) [SIG CLI]
- Kubelet: print sorted volumes message in events ([#117079](https://github.com/kubernetes/kubernetes/pull/117079), [@qingwave](https://github.com/qingwave)) [SIG Node]

### Other (Cleanup or Flake)

- E2e framework: the `node-role.kubernetes.io/master` taint has been removed from the default value of `--non-blocking-taints` flag. You may need to set `--non-blocking-taints` explicitly if the cluster to be tested has nodes with the deprecated `node-role.kubernetes.io/master` taint. ([#118510](https://github.com/kubernetes/kubernetes/pull/118510), [@SataQiu](https://github.com/SataQiu)) [SIG Testing]
- Kube-apiserver adds two new alpha metrics `conversion_webhook_request_total` and `conversion_webhook_duration_seconds` that allow users to monitor requests to CRD conversion webhooks, split by result, and failure_type (In case of failure). ([#118292](https://github.com/kubernetes/kubernetes/pull/118292), [@cchapla](https://github.com/cchapla)) [SIG API Machinery, Architecture and Instrumentation]
- Moved `k8s.io/kubernetes/pkg/kubelet/cri/streaming` package to `k8s.io/kubelet/pkg/cri/streaming`. ([#118253](https://github.com/kubernetes/kubernetes/pull/118253), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node, Release and Security]
- OpenAPI proto deserializations should use gnostic-models instead of the gnostic library ([#118384](https://github.com/kubernetes/kubernetes/pull/118384), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Instrumentation, Node, Storage and Testing]
- [KCCM] drop filtering nodes for the providerID when syncing load balancers, but have changes to the field trigger a re-sync of load balancers. This should ensure that cloud providers which don't specify providerID, can still use the service controller implementation to provision load balancers. ([#117602](https://github.com/kubernetes/kubernetes/pull/117602), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider and Network]

## Dependencies

### Added
- github.com/antlr/antlr4/runtime/Go/antlr/v4: [8188dc5](https://github.com/antlr/antlr4/runtime/Go/antlr/v4/tree/8188dc5)
- github.com/google/gnostic-models: [v0.6.8](https://github.com/google/gnostic-models/tree/v0.6.8)

### Changed
- github.com/dustin/go-humanize: [v1.0.0 → v1.0.1](https://github.com/dustin/go-humanize/compare/v1.0.0...v1.0.1)
- github.com/evanphx/json-patch: [v4.12.0+incompatible → v5.6.0+incompatible](https://github.com/evanphx/json-patch/compare/v4.12.0...v5.6.0)
- github.com/go-openapi/jsonreference: [v0.20.1 → v0.20.2](https://github.com/go-openapi/jsonreference/compare/v0.20.1...v0.20.2)
- github.com/google/cel-go: [v0.12.6 → v0.16.0](https://github.com/google/cel-go/compare/v0.12.6...v0.16.0)
- github.com/mitchellh/mapstructure: [v1.4.1 → v1.1.2](https://github.com/mitchellh/mapstructure/compare/v1.4.1...v1.1.2)
- go.starlark.net: 8dd3e2e → a134d8f
- golang.org/x/exp: 6cc2880 → a9213ee
- golang.org/x/sys: v0.7.0 → v0.8.0
- k8s.io/kube-openapi: 7828149 → 7562a10
- sigs.k8s.io/kustomize/api: v0.13.2 → 6ce0bf3
- sigs.k8s.io/kustomize/cmd/config: v0.11.1 → v0.11.2
- sigs.k8s.io/kustomize/kustomize/v5: v5.0.1 → 6ce0bf3
- sigs.k8s.io/kustomize/kyaml: v0.14.1 → 6ce0bf3

### Removed
- github.com/antlr/antlr4/runtime/Go/antlr: [v1.4.10](https://github.com/antlr/antlr4/runtime/Go/antlr/tree/v1.4.10)
- github.com/docopt/docopt-go: [ee0de3b](https://github.com/docopt/docopt-go/tree/ee0de3b)
- github.com/google/gnostic: [v0.5.7-v3refs](https://github.com/google/gnostic/tree/v0.5.7-v3refs)



# v1.28.0-alpha.1


## Downloads for v1.28.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes.tar.gz) | 65d841f778b00a04a13f3e722753704d4164f8590c2b0aca9cbb9bf85822be5343205ead8c71f9502d8b22fc84d80804fed5edc665662b0405bb0efa65fec808
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-src.tar.gz) | 82fbe3f389b922cc635a896fa6c3e8cc342e4ca70003ca5491c7b3eb2e38065349e270da9c0deb0e541271978ade247ff3a420806a51d035a5a850262e41baa9

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | c5be770467a8617221021255a22a970a72ccee3672b1973fb31c65b1de02767d014a8e9058f710f0d9b402f2b056fd17ed216cb1d6126f9738efb16f88e184c0
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | a194b07e23b8cee142080361394e0db7f3fb0488c16eeef3059dfb178f4cef6e124ad31c511a516058b8f82a6ab0f0194183714016ebd88e3060368528405e2c
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-linux-386.tar.gz) | ada349bd3f76b5572467a8fad504c26a223eeb50ad7677287b39db434adb5a59d2ceadd1922712f99878153f20fa8b0cd2b30a16e8e178a41c6ac747b55ee79c
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | c729d419e53a006996f5e583e0fa9a541ea7d2df7dc875dae729c63cd8222f10121908750c48ff34942fcbdf6456ed977bef86c4b979202fab120de0a7a42fc9
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 6bf4a115b4f4b7b21d193fe44f99c5b019e9f2097e831bd44958de6e63bd8068a70a9cfa535dc18dca23c0c4461195e8a62c8f1cd9faff7f5bb3c7b1b13ad604
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 536101d9f50bf71e66e35781e0ca729156227405225986198276a43d2cf32aa2cbae32f0743bcb967701309ea3bd19e9e9f6150e532a2d251440f18ca8afbd16
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | a8dd8c0aaa7dce825f982edbff1ecd57671643e2725390c60b43450118abf2dd3594f306af6cbbd2df1aa146a0b21d0576c1b6e8e1dd2b50190702d1e879ad3a
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | b6ab35eb6c55536f91c4c0ae32b8db3462426fea11a4cce3e06581129995b42c4acdd16674e357d92280dae5ab9f50bcb6b8d5052d65c0a06b9c21fbb646e830
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 362c2f7f7327775a75b0c6cc2e3e372475d7d9291ad5f7c224632e037fe181b149d6def98dbd034d8ba73d3bac335a7788fbaa08df924e05c9ed9844fa75135f
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 75297a5c9f7d8f39f640d97bf4ece9a78b2226103d6b66865dcf6752375bf76b9d3e3d4b13efb291275621e7b1e4858eaa36f469ac73495bba43dfca2b900085
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | e79cddab0abb31ef7f17855d9b14799fc7a66247c3aa71eed01231d40cb5caa7dad08082904fd18cc126cef1d3a7c2f42b8a8994e7ab40271eb0d8baa1a42f74

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | 64b5c5e1502fbe6a21ff6cde999408ff83f1d3b1088fbe05d720f90e5f0a9193b5ba1b1aaaee65e6ec1354e63e60d29c55a90535f79624f4526dea96295ad48d
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 26519f8406e2900b00a22d4e03260701ded84ddba0730f25a794f5b4bfcba452ab1c321f32fe30a7e2bf748fc93cf05fe81b2fdec7fa86af1e9f882428179f85
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | c66df63d33607d8a3f2ae57ca80e4134b423bd8448ee3ecd72936f0c5973d027ab27f92481fc83e41b4b929cdae4be3865477e59f316dc102e19aa79e52afe6b
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 5a6c30cdf7f24b2ab906cf1a27f07bb7e5fafef100942b33320c2e8445b7934c2663ae7b7cc47f8aec173c1788ace9576144df357bef83e3d7a42e827f1a7c94

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 95ce88f26c3809f268e8b83122dc4d0685e7b31f44dedad3b1360edd76c921e2a6e0c9077c136fea078299f4451280fbf49c9f956fc30339db752e5aa0e73367
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | bf36de0876bab1b08e1268dd5602d5af46e99a9939e8befcb9d6fea91d04fc67438d136ae28503c3342dcff63e9849b2ca81b00c29627a9a477fcaed5e4f3443
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 649b49fe2319a9fd149d08665bdbe3c825f21bb96d4695dbb4fadad367e027f000272326217194f8319cb074ee6f15dc9b6bf4c0ff4dfcda08003680b39faebf
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 2b0c9466e9d42576d1bae61b2141e41521cfb0ae2c13ff3b59ea8abec124a44601c76a3e9e0a6283b6c74e9fee27d420b131238811f4dd4bdee789247b44941c
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.28.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | a26243c3e7bab5180b5ff44139dfcecb6975326fdc6dec9b71f5dfccd89889710bcfadcde5c5a0c9ef03378396729e9b2763b38d6b67840239cb144981b98317

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.28.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.28.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.28.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.28.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.28.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.28.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.27.0

## Changes by Kind

### Deprecation

- Remove tracking annotation from validation and defaulting ([#117633](https://github.com/kubernetes/kubernetes/pull/117633), [@kannon92](https://github.com/kannon92)) [SIG Apps]
- Remove withdrawn feature NetworkPolicyStatus ([#115843](https://github.com/kubernetes/kubernetes/pull/115843), [@rikatz](https://github.com/rikatz)) [SIG API Machinery, Apps, Architecture, Network and Testing]

### API Change

- Added a warning that TLS 1.3 ciphers are not configurable. ([#115399](https://github.com/kubernetes/kubernetes/pull/115399), [@3u13r](https://github.com/3u13r)) [SIG API Machinery and Node]
- Added error handling for seccomp localhost configurations that do not properly set a localhostProfile ([#117020](https://github.com/kubernetes/kubernetes/pull/117020), [@cji](https://github.com/cji)) [SIG API Machinery and Node]
- Added new config option `delayCacheUntilActive` to `KubeSchedulerConfiguration` that can provide a tradeoff between memory efficiency and scheduling speed when their leadership is updated in `kube-scheduler` ([#115754](https://github.com/kubernetes/kubernetes/pull/115754), [@linxiulei](https://github.com/linxiulei)) [SIG API Machinery and Scheduling]
- Client-go: Improved memory use of reflector caches when watching large numbers of objects which do not change frequently ([#113362](https://github.com/kubernetes/kubernetes/pull/113362), [@sxllwx](https://github.com/sxllwx)) [SIG API Machinery]
- Kube-controller-manager: The `LegacyServiceAccountTokenCleanUp` feature gate is now available as alpha (off by default). When enabled, the `legacy-service-account-token-cleaner` controller loop removes service account token secrets that have not been used in the time specified by `--legacy-service-account-token-clean-up-period` (defaulting to one year), **and are** referenced from the `.secrets` list of a ServiceAccount object, **and are not** referenced from pods. ([#115554](https://github.com/kubernetes/kubernetes/pull/115554), [@yt2985](https://github.com/yt2985)) [SIG API Machinery, Apps, Auth, Release and Testing]
- Kube-scheduler component config (KubeSchedulerConfiguration) kubescheduler.config.k8s.io/v1beta2 is removed in v1.28. Migrate kube-scheduler configuration files to kubescheduler.config.k8s.io/v1. ([#117649](https://github.com/kubernetes/kubernetes/pull/117649), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery, Scheduling and Testing]
- NodeVolumeLimits implement the PreFilter extension point for skipping the Filter phase if the Pod doesn't use volumes with limits. ([#115398](https://github.com/kubernetes/kubernetes/pull/115398), [@tangwz](https://github.com/tangwz)) [SIG Scheduling]
- Pods which set `hostNetwork: true` and declare ports get the `hostPort` field set automatically.  Previously this would happen in the PodTemplate of a Deployment, DaemonSet or other workload API.  Now `hostPort` will only be set when an actual Pod is being created.  If this presents a problem, setting the feature gate "DefaultHostNetworkHostPortsInWorkloads" to true will revert this behavior.  Please file a kubernetes bug if you need to do this. ([#117696](https://github.com/kubernetes/kubernetes/pull/117696), [@thockin](https://github.com/thockin)) [SIG Apps]
- Removing WindowsHostProcessContainers feature-gate ([#117570](https://github.com/kubernetes/kubernetes/pull/117570), [@marosset](https://github.com/marosset)) [SIG API Machinery, Apps, Auth, Node and Windows]
- Revised the comment about the feature-gate level for PodFailurePolicy from alpha to beta ([#117802](https://github.com/kubernetes/kubernetes/pull/117802), [@kerthcet](https://github.com/kerthcet)) [SIG API Machinery and Apps]
- The `SelfSubjectReview` API is promoted to `authentication.k8s.io/v1` and the `kubectl auth whoami` command is GA. ([#117713](https://github.com/kubernetes/kubernetes/pull/117713), [@nabokihms](https://github.com/nabokihms)) [SIG API Machinery, Architecture, Auth, CLI and Testing]

### Feature

- Add '--concurrent-job-syncs' flag for kube-controller-manager to set the number of job controller workers ([#117138](https://github.com/kubernetes/kubernetes/pull/117138), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery and CLI]
- Add DisruptionTarget condition to the pod preempted by Kubelet to make room for a critical pod ([#117586](https://github.com/kubernetes/kubernetes/pull/117586), [@mimowo](https://github.com/mimowo)) [SIG Node and Testing]
- Added a container image for `kubectl` at `registry.k8s.io/kubectl` across the same architectures as other images (linux/amd64 linux/arm64 linux/s390x linux/ppc64le) ([#116672](https://github.com/kubernetes/kubernetes/pull/116672), [@dims](https://github.com/dims)) [SIG Architecture and Release]
- Added support for pod `hostNetwork` field selector ([#110477](https://github.com/kubernetes/kubernetes/pull/110477), [@halfcrazy](https://github.com/halfcrazy)) [SIG Apps and Node]
- Apiserver adds two new metrics `etcd_requests_total` and `etcd_request_errors_total` that allow users to monitor requests to etcd storage, split by operation and resource type. ([#117222](https://github.com/kubernetes/kubernetes/pull/117222), [@iyear](https://github.com/iyear)) [SIG API Machinery]
- Bump metrics-server to v0.6.3. ([#117120](https://github.com/kubernetes/kubernetes/pull/117120), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Cloud Provider and Instrumentation]
- Client-go exposes two new metrics to monitor the client-go logic that
  generate http.Transports for the clients.
  
  - rest_client_transport_cache_entries is a gauge metric
  with the number of existin entries in the internal cache
  
  - rest_client_transport_create_calls_total is a counter
  that increments each time a new transport is created, storing
  the result of the operation needed to generate it: hit, miss
  or uncacheable ([#117295](https://github.com/kubernetes/kubernetes/pull/117295), [@aojea](https://github.com/aojea)) [SIG API Machinery, Architecture, Instrumentation, Network, Node and Testing]
- External credential provider plugins now have their standard error output logged by kubelet upon failures. ([#117448](https://github.com/kubernetes/kubernetes/pull/117448), [@cartermckinnon](https://github.com/cartermckinnon)) [SIG Node]
- Graduated the `LegacyServiceAccountTokenTracking` feature gate to GA. The usage of auto-generated secret-based service account token now produces warnings, and relevant Secrets are labeled with a last-used timestamp (label key `kubernetes.io/legacy-token-last-used`). ([#117591](https://github.com/kubernetes/kubernetes/pull/117591), [@zshihang](https://github.com/zshihang)) [SIG API Machinery, Auth and Testing]
- Klog text output now uses JSON as encoding for structs, maps and slices. ([#117687](https://github.com/kubernetes/kubernetes/pull/117687), [@pohly](https://github.com/pohly)) [SIG Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node and Storage]
- Kube-proxy handles Terminating EndpointSlices conditions and enables zero downtime deployments for Services with ExternalTrafficPolicy=Local author: @andrewsykim ([#117718](https://github.com/kubernetes/kubernetes/pull/117718), [@aojea](https://github.com/aojea)) [SIG Network, Testing and Windows]
- Kube-proxy in iptables mode now has separate `sync_full_proxy_rules_duration_seconds`
  and `sync_partial_proxy_rules_duration_seconds` (in addition to the existing
  `sync_proxy_rules_duration_seconds`), to give better information about how long
  each sync type is taking, rather than only giving a weighted average of the two
  sync types together. ([#117787](https://github.com/kubernetes/kubernetes/pull/117787), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kubeadm: add `--feature-gates` flag for `kubeadm upgrade node` ([#118316](https://github.com/kubernetes/kubernetes/pull/118316), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: add a new "kubeadm config validate" command that can be used to validate any input config file. Use the --config flag to pass a config file to it. See the command --help screen for more information. As a result of adding this new command, enhance the validation capabilities of the existing "kubeadm config migrate" command. For both commands unknown APIs or fields will throw errors. ([#118013](https://github.com/kubernetes/kubernetes/pull/118013), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Go 1.20.4 ([#117744](https://github.com/kubernetes/kubernetes/pull/117744), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Metric `scheduler_scheduler_goroutines` is removed. Use `scheduler_goroutines` instead. ([#117727](https://github.com/kubernetes/kubernetes/pull/117727), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Migrated `pkg/scheduler/framework/preemption` to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116835](https://github.com/kubernetes/kubernetes/pull/116835), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]
- Migrated `pod-security-admission` to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#114471](https://github.com/kubernetes/kubernetes/pull/114471), [@Namanl2001](https://github.com/Namanl2001)) [SIG Apps and Auth]
- Migrated the noderesources scheduler plugin to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116748](https://github.com/kubernetes/kubernetes/pull/116748), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]
- Migrated the podtopologyspread scheduler plugins to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116797](https://github.com/kubernetes/kubernetes/pull/116797), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]
- Set metrics-server's metric-resolution to 15s ([#117121](https://github.com/kubernetes/kubernetes/pull/117121), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Cloud Provider and Instrumentation]
- SubjectAccessReview requests sent to webhook authorizers now default `spec.resourceAttributes.version` to `*` if unset. ([#116937](https://github.com/kubernetes/kubernetes/pull/116937), [@AxeZhan](https://github.com/AxeZhan)) [SIG Apps and Auth]
- Support specifying a custom retry period for cloud load-balancer operations ([#94021](https://github.com/kubernetes/kubernetes/pull/94021), [@timoreimann](https://github.com/timoreimann)) [SIG API Machinery, Cloud Provider and Network]
- The Kubernetes apiserver now emits a warning message for Pods with a null labelSelector in podAffinity or topologySpreadConstraints. The null labelSelector means "match none". Using it in podAffinity or topologySpreadConstraint could lead to unintended behavior. ([#117025](https://github.com/kubernetes/kubernetes/pull/117025), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- The scheduler skips the InterPodAffinity Score plugin when nothing to do with the Pod.
  It will affect some metrics values related to the InterPodAffinity Score plugin. ([#117794](https://github.com/kubernetes/kubernetes/pull/117794), [@utam0k](https://github.com/utam0k)) [SIG Scheduling]
- The scheduler skips the PodTopologySpread Filter plugin if no spread constraints.
  It will affect some metrics values related to the PodTopologySpread Filter plugin. ([#117683](https://github.com/kubernetes/kubernetes/pull/117683), [@utam0k](https://github.com/utam0k)) [SIG Scheduling]
- The short names vwc and mwc were introduced for the resources validatingwebhookconfigurations and mutatingwebhookconfigurations. ([#117535](https://github.com/kubernetes/kubernetes/pull/117535), [@hysyeah](https://github.com/hysyeah)) [SIG API Machinery]
- Update etcd image to 3.5.9-0 ([#117999](https://github.com/kubernetes/kubernetes/pull/117999), [@kkkkun](https://github.com/kkkkun)) [SIG API Machinery]
- Update the scheduler interface and cache methods to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116849](https://github.com/kubernetes/kubernetes/pull/116849), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps, Instrumentation, Scheduling and Testing]
- Updated distroless iptables to use released image `registry.k8s.io/build-image/distroless-iptables:v0.2.4` ([#117746](https://github.com/kubernetes/kubernetes/pull/117746), [@xmudrii](https://github.com/xmudrii)) [SIG Testing]
- `--version=v1.X.Y...` can now be used to set the prerelease and buildID portions of the version reported by components ([#117688](https://github.com/kubernetes/kubernetes/pull/117688), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Architecture and Release]

### Documentation

- Enhanced clarity in error messaging when waiting for volume creation ([#118262](https://github.com/kubernetes/kubernetes/pull/118262), [@torredil](https://github.com/torredil)) [SIG Apps and Storage]

### Failing Test

- Allow Azure Disk e2es to use newer topology labels if available from nodes ([#117216](https://github.com/kubernetes/kubernetes/pull/117216), [@gnufied](https://github.com/gnufied)) [SIG Storage and Testing]
- Fix nil pointer in test AfterEach volumeperf.go for sidecar release ([#117368](https://github.com/kubernetes/kubernetes/pull/117368), [@sunnylovestiramisu](https://github.com/sunnylovestiramisu)) [SIG Storage and Testing]

### Bug or Regression

- CVE-2023-27561 CVE-2023-25809 CVE-2023-28642: Bump fix runc v1.1.4 -> v1.1.5 ([#117095](https://github.com/kubernetes/kubernetes/pull/117095), [@PushkarJ](https://github.com/PushkarJ)) [SIG Architecture, Node and Security]
- Code blocks in kubectl {$COMMAND}--help will move right by 3 indentation. ([#118029](https://github.com/kubernetes/kubernetes/pull/118029), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- During device plugin allocation, resources requested by the pod can only be allocated if the device plugin has registered itself to kubelet AND healthy devices are present on the node to be allocated. If these conditions are not sattsfied, the pod would fail with `UnexpectedAdmissionError` error. ([#116376](https://github.com/kubernetes/kubernetes/pull/116376), [@swatisehgal](https://github.com/swatisehgal)) [SIG Node and Testing]
- Fix Topology Aware Hints not working when the `topology.kubernetes.io/zone` label is added after Node creation ([#117245](https://github.com/kubernetes/kubernetes/pull/117245), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Fix a data race in TopologyCache when `AddHints` and `SetNodes` are called concurrently ([#117249](https://github.com/kubernetes/kubernetes/pull/117249), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Fix bug where `listOfStrings.join()` in CEL expressions resulted in an unexpected internal error. ([#117593](https://github.com/kubernetes/kubernetes/pull/117593), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fix incorrect calculation for ResourceQuota with PriorityClass as its scope. ([#117677](https://github.com/kubernetes/kubernetes/pull/117677), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG API Machinery]
- Fix performance regression in scheduler caused by frequent metric lookup on critical code path. ([#117594](https://github.com/kubernetes/kubernetes/pull/117594), [@tosi3k](https://github.com/tosi3k)) [SIG Scheduling]
- Fix restricted debug profile. ([#117543](https://github.com/kubernetes/kubernetes/pull/117543), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI and Testing]
- Fix: After a Node is down and take some time to get back to up again, the mount point of the evicted Pods cannot be cleaned up successfully. (#111933) Meanwhile Kubelet will print the log `Orphaned pod "xxx" found, but error not a directory occurred when trying to remove the volumes dir` every 2 seconds. (#105536) ([#116134](https://github.com/kubernetes/kubernetes/pull/116134), [@cvvz](https://github.com/cvvz)) [SIG Node and Storage]
- Fix: the volume is not detached after the pod and PVC objects are deleted ([#116138](https://github.com/kubernetes/kubernetes/pull/116138), [@cvvz](https://github.com/cvvz)) [SIG Storage]
- Fixed a bug that unintentionally overrides your custom Accept headers in http (live-/readiness)-probes if the header is in lower casing ([#114606](https://github.com/kubernetes/kubernetes/pull/114606), [@tuunit](https://github.com/tuunit)) [SIG Network and Node]
- Fixed a bug where pv recycler failed to scrub volume with too many files in the directory due to hitting ARG_MAX limit with rm command (#117189). ([#117283](https://github.com/kubernetes/kubernetes/pull/117283), [@defo89](https://github.com/defo89)) [SIG Cloud Provider and Storage]
- Fixed a memory leak in the Kubernetes API server that occurs during APIService processing. ([#117258](https://github.com/kubernetes/kubernetes/pull/117258), [@enj](https://github.com/enj)) [SIG API Machinery]
- Fixed an issue where the API server did not send impersonated UID to authentication webhooks. ([#116681](https://github.com/kubernetes/kubernetes/pull/116681), [@stlaz](https://github.com/stlaz)) [SIG API Machinery and Auth]
- Fixed bug to correctly report `ErrRegistryUnavailable` on pulling container images for remote CRI runtimes. ([#117612](https://github.com/kubernetes/kubernetes/pull/117612), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Fixed bug where using the $deleteFromPrimitiveList directive in a strategic merge patch of certain fields would remove the other values from the list instead of the values specified. ([#110472](https://github.com/kubernetes/kubernetes/pull/110472), [@brianpursley](https://github.com/brianpursley)) [SIG API Machinery]
- Fixed issue where kubectl-convert would fail when encountering resources that could not be converted to the specified api version. New behavior is to warn the user of the failed conversions and continue to convert the remaining resources. ([#117002](https://github.com/kubernetes/kubernetes/pull/117002), [@gxwilkerson33](https://github.com/gxwilkerson33)) [SIG CLI and Testing]
- Fixed issue where there was no response or error from kubectl rollout status when there were no resources of specified kind. ([#117884](https://github.com/kubernetes/kubernetes/pull/117884), [@gxwilkerson33](https://github.com/gxwilkerson33)) [SIG CLI]
- Fixed vSphere cloud provider not to skip detach volumes from nodes at kube-controller-startup. ([#117243](https://github.com/kubernetes/kubernetes/pull/117243), [@jsafrane](https://github.com/jsafrane)) [SIG Cloud Provider]
- Fixes a bug at kube-apiserver start where APIService objects for custom resources could be deleted and recreated. ([#118104](https://github.com/kubernetes/kubernetes/pull/118104), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Fixes a race condition serving OpenAPI content ([#117705](https://github.com/kubernetes/kubernetes/pull/117705), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Instrumentation and Node]
- Fixes a regression in 1.27.0 that resulted in "missing metadata in converted object" errors when modifying objects for multi-version custom resource definitions with a conversion strategy of `None`. ([#117301](https://github.com/kubernetes/kubernetes/pull/117301), [@ncdc](https://github.com/ncdc)) [SIG API Machinery]
- Fixes a regression in kubectl and client-go discovery when configured with a server URL other than the root of a server. ([#117495](https://github.com/kubernetes/kubernetes/pull/117495), [@ardaguclu](https://github.com/ardaguclu)) [SIG API Machinery]
- Fixes bug that caused a resource to include patch directives when using strategic merge patch against a non-existent field ([#117568](https://github.com/kubernetes/kubernetes/pull/117568), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery and Testing]
- Fixes creationTimestamp: null causing unnecessary writes to etcd ([#116865](https://github.com/kubernetes/kubernetes/pull/116865), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery and Testing]
- If `kubeadm reset` finds no etcd member ID for the peer it removes during the `remove-etcd-member` phase, it continues immediately to other phases, instead of retrying the phase for up to 3 minutes before continuing. ([#117724](https://github.com/kubernetes/kubernetes/pull/117724), [@dlipovetsky](https://github.com/dlipovetsky)) [SIG Cluster Lifecycle]
- Improved exponential backoff in Reflector, significantly reducing the load on Kubernetes apiserver in case of throttling of requests. ([#118132](https://github.com/kubernetes/kubernetes/pull/118132), [@marseel](https://github.com/marseel)) [SIG API Machinery and Scalability]
- Known issue: fixed that the PreEnqueue plugins aren't executed for Pods proceeding to activeQ through backoffQ. ([#117194](https://github.com/kubernetes/kubernetes/pull/117194), [@sanposhiho](https://github.com/sanposhiho)) [SIG Release and Scheduling]
- Kube-apiserver always removes its endpoint from kubernetes service during graceful shutdown (even if it's the only/last one) ([#116685](https://github.com/kubernetes/kubernetes/pull/116685), [@czybjtu](https://github.com/czybjtu)) [SIG API Machinery]
- Kubeadm: crictl pull should use `-i` to set the image service endpoint ([#117835](https://github.com/kubernetes/kubernetes/pull/117835), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where file copy(backup) could not be executed correctly on Windows platform during upgrade ([#117861](https://github.com/kubernetes/kubernetes/pull/117861), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where the static pod changes detection logic is inconsistent with kubelet ([#118069](https://github.com/kubernetes/kubernetes/pull/118069), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: speedup init by 0s or 20s. kubelet-start phase is now after etcd and control-plane phases, removing a race condition between kubelet looking for static pod manifests and kubeadm writing them. ([#117984](https://github.com/kubernetes/kubernetes/pull/117984), [@champtar](https://github.com/champtar)) [SIG Cluster Lifecycle]
- Kubeadm: throw warnings instead of errors for deprecated feature gates ([#118270](https://github.com/kubernetes/kubernetes/pull/118270), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubectl events --for will also support fully qualified names such as replicasets.apps, etc. ([#117034](https://github.com/kubernetes/kubernetes/pull/117034), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Kubelet now skips pod resource checks when the request is zero. ([#116408](https://github.com/kubernetes/kubernetes/pull/116408), [@ChenLingPeng](https://github.com/ChenLingPeng)) [SIG Scheduling]
- Kubelet terminates pods correctly upon restart, fixing an issue where pods may have not been fully terminated if the kubelet was restarted during pod termination. ([#117019](https://github.com/kubernetes/kubernetes/pull/117019), [@bobbypage](https://github.com/bobbypage)) [SIG Node and Testing]
- Kubelet will ensure /etc/hosts file is mode 0644 regardless of umask. ([#113209](https://github.com/kubernetes/kubernetes/pull/113209), [@luozhiwenn](https://github.com/luozhiwenn)) [SIG Node]
- Number of errors reported to the metric `storage_operation_duration_seconds_count` for emptyDir decreased significantly because previously one error was reported for each projected volume created. ([#117022](https://github.com/kubernetes/kubernetes/pull/117022), [@mpatlasov](https://github.com/mpatlasov)) [SIG Storage]
- Pod termination will be faster when the pod has a missing volume reference. ([#117412](https://github.com/kubernetes/kubernetes/pull/117412), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- Recording timing traces had a race condition. Impact in practice was probably low. ([#117139](https://github.com/kubernetes/kubernetes/pull/117139), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node and Storage]
- Removed leading zeros from the etcd member ID in kubeadm log messages. ([#117919](https://github.com/kubernetes/kubernetes/pull/117919), [@dlipovetsky](https://github.com/dlipovetsky)) [SIG Cluster Lifecycle]
- Resolves a spurious "Unknown discovery response content-type" error in client-go discovery requests by tolerating extra content-type parameters in API responses ([#117571](https://github.com/kubernetes/kubernetes/pull/117571), [@seans3](https://github.com/seans3)) [SIG API Machinery]
- Reverted NewVolumeManagerReconstruction and SELinuxMountReadWriteOncePod feature gates to disabled by default to resolve a regression of volume reconstruction on kubelet/node restart ([#117751](https://github.com/kubernetes/kubernetes/pull/117751), [@liggitt](https://github.com/liggitt)) [SIG Storage]
- Setting a mirror pod's phase to Succeeded or Failed can prevent the corresponding static pod from restarting due mutation of a Kubelet cache. ([#116482](https://github.com/kubernetes/kubernetes/pull/116482), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Show a warning when `volume.beta.kubernetes.io/storage-class` annotation is used in pv or pvc ([#117036](https://github.com/kubernetes/kubernetes/pull/117036), [@haoruan](https://github.com/haoruan)) [SIG Storage]
- Static pods were taking extra time to be restarted after being updated.  Static pods that are waiting to restart were not correctly counted in `kubelet_working_pods`. ([#116995](https://github.com/kubernetes/kubernetes/pull/116995), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- This PR adds additional validation for endpoint ip configuration while iterating through queried endpoint list. ([#116749](https://github.com/kubernetes/kubernetes/pull/116749), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Update etcd version to 3.5.8 ([#117335](https://github.com/kubernetes/kubernetes/pull/117335), [@kkkkun](https://github.com/kkkkun)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Updated static pods are restarted 2s faster by correcting a safe but non-optimal ordering bug. ([#116690](https://github.com/kubernetes/kubernetes/pull/116690), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- [KCCM] service controller: change the cloud controller manager to make `providerID` a predicate when synchronizing nodes. This change allows load balancer integrations to ensure that  the `providerID` is set when configuring
  load balancers and targets. ([#117388](https://github.com/kubernetes/kubernetes/pull/117388), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider and Network]

### Other (Cleanup or Flake)

- A v2-level info log will be added, which will output the details of the pod being preempted, including victim and preemptor ([#117214](https://github.com/kubernetes/kubernetes/pull/117214), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Scheduling]
- Allow container runtimes to use `ErrSignatureValidationFailed` as possible image pull failure. ([#117717](https://github.com/kubernetes/kubernetes/pull/117717), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Deprecate genericclioptions.IOStreams and use genericiooptions.IOStreams ([#117102](https://github.com/kubernetes/kubernetes/pull/117102), [@ardaguclu](https://github.com/ardaguclu)) [SIG Auth, CLI and Release]
- Enables the node-local kubelet podresources API endpoint on windows, alongside unix. ([#115133](https://github.com/kubernetes/kubernetes/pull/115133), [@ffromani](https://github.com/ffromani)) [SIG Cloud Provider, Node, Testing and Windows]
- Fixed dra e2e image build on non-amd64 architectures ([#117912](https://github.com/kubernetes/kubernetes/pull/117912), [@bart0sh](https://github.com/bart0sh)) [SIG Node and Testing]
- Kube-apiserver adds two new metrics `authorization_attempts_total` and `authorization_duration_seconds` that allow users to monitor requests to authorization webhooks, split by result. ([#117211](https://github.com/kubernetes/kubernetes/pull/117211), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery, Auth and Instrumentation]
- Kubeadm: introduce a new feature gate UpgradeAddonsBeforeControlPlane to fix a kube-proxy skew policy misalignment. Its default value is `false`. Upgrade of the CoreDNS and kube-proxy addons will now trigger after all the control plane instances have been upgraded, unless the fearure gate is set to true. This feature gate will be removed in a future release. ([#117660](https://github.com/kubernetes/kubernetes/pull/117660), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Marked the feature gate `ExperimentalHostUserNamespaceDefaulting` as deprecated.
  Enabling the feature gate already had no effect; the deprecation allows for removing the feature gate in a future release. ([#116723](https://github.com/kubernetes/kubernetes/pull/116723), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node]
- Migrated `pkg/scheduler/framework/runtime` to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116842](https://github.com/kubernetes/kubernetes/pull/116842), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]
- Migrated the volumezone scheduler plugin to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116829](https://github.com/kubernetes/kubernetes/pull/116829), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]
- Projects which use k8s.io/code-generator and invoke `generate-groups` or `generate-internal-groups.sh` have a new, simpler script (`kube_codegen.sh`) they can use.  The old scripts are deprecated but remain intact. ([#117262](https://github.com/kubernetes/kubernetes/pull/117262), [@thockin](https://github.com/thockin)) [SIG API Machinery and Instrumentation]
- Remove GAed feature gate DelegateFSGroupToCSIDriver ([#117655](https://github.com/kubernetes/kubernetes/pull/117655), [@carlory](https://github.com/carlory)) [SIG Storage]
- Remove GAed feature gate DevicePlugins ([#117656](https://github.com/kubernetes/kubernetes/pull/117656), [@carlory](https://github.com/carlory)) [SIG Node]
- Remove GAed feature gate KubeletCredentialProviders ([#116901](https://github.com/kubernetes/kubernetes/pull/116901), [@pacoxu](https://github.com/pacoxu)) [SIG Cloud Provider, Node and Testing]
- Remove GAed feature gates: MixedProtocolLBService, ServiceInternalTrafficPolicy, ServiceIPStaticSubrange, and EndpointSliceTerminatingCondition ([#117237](https://github.com/kubernetes/kubernetes/pull/117237), [@yulng](https://github.com/yulng)) [SIG Network]
- Removed the deprecated `azureFile` in-tree storage plugin ([#118236](https://github.com/kubernetes/kubernetes/pull/118236), [@andyzhangx](https://github.com/andyzhangx)) [SIG API Machinery, Cloud Provider, Node and Storage]
- Structured logging of NamespacedName was inconsistent with klog.KObj. Now both use lower case field names and namespace is optional. ([#117238](https://github.com/kubernetes/kubernetes/pull/117238), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture and Instrumentation]
- The `generate_groups.sh` and `generate_internal_groups.sh` scripts from the k8s.io/code-generator repo are deprecated (but still work) in favor of `kube_codegen.sh` in that same repo.  Projects which use the old scripts are encouraged to look at adopting the new one. ([#117897](https://github.com/kubernetes/kubernetes/pull/117897), [@thockin](https://github.com/thockin)) [SIG API Machinery]
- The feature gate CSIStorageCapacity have been removed and must no longer be referenced in `--feature-gates` flags ([#118018](https://github.com/kubernetes/kubernetes/pull/118018), [@humblec](https://github.com/humblec)) [SIG Storage]
- The feature gates `DisableAcceleratorUsageMetrics` and `PodSecurity` that graduated to GA and were unconditionally enabled have been removed in v1.28 ([#114068](https://github.com/kubernetes/kubernetes/pull/114068), [@cyclinder](https://github.com/cyclinder)) [SIG API Machinery, Node, Scheduling and Storage]
- The kubelet podresources endpoint is GA and always enabled ([#116525](https://github.com/kubernetes/kubernetes/pull/116525), [@ffromani](https://github.com/ffromani)) [SIG Node]
- Updated Cluster Autosaler to version 1.26.1 ([#116526](https://github.com/kubernetes/kubernetes/pull/116526), [@pacoxu](https://github.com/pacoxu)) [SIG Autoscaling and Cloud Provider]
- Updated cri-tools to v1.26.1. ([#116649](https://github.com/kubernetes/kubernetes/pull/116649), [@saschagrunert](https://github.com/saschagrunert)) [SIG Architecture and Release]
- Updated cri-tools to v1.27.0 ([#117545](https://github.com/kubernetes/kubernetes/pull/117545), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- When retrieving event resources, the reportingController and reportingInstance fields in the event will contain values. ([#116506](https://github.com/kubernetes/kubernetes/pull/116506), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery and Instrumentation]

## Dependencies

### Added
- cloud.google.com/go/accessapproval: v1.6.0
- cloud.google.com/go/accesscontextmanager: v1.7.0
- cloud.google.com/go/aiplatform: v1.37.0
- cloud.google.com/go/analytics: v0.19.0
- cloud.google.com/go/apigateway: v1.5.0
- cloud.google.com/go/apigeeconnect: v1.5.0
- cloud.google.com/go/apigeeregistry: v0.6.0
- cloud.google.com/go/appengine: v1.7.1
- cloud.google.com/go/area120: v0.7.1
- cloud.google.com/go/artifactregistry: v1.13.0
- cloud.google.com/go/asset: v1.13.0
- cloud.google.com/go/assuredworkloads: v1.10.0
- cloud.google.com/go/automl: v1.12.0
- cloud.google.com/go/baremetalsolution: v0.5.0
- cloud.google.com/go/batch: v0.7.0
- cloud.google.com/go/beyondcorp: v0.5.0
- cloud.google.com/go/billing: v1.13.0
- cloud.google.com/go/binaryauthorization: v1.5.0
- cloud.google.com/go/certificatemanager: v1.6.0
- cloud.google.com/go/channel: v1.12.0
- cloud.google.com/go/cloudbuild: v1.9.0
- cloud.google.com/go/clouddms: v1.5.0
- cloud.google.com/go/cloudtasks: v1.10.0
- cloud.google.com/go/compute/metadata: v0.2.3
- cloud.google.com/go/compute: v1.19.0
- cloud.google.com/go/contactcenterinsights: v1.6.0
- cloud.google.com/go/container: v1.15.0
- cloud.google.com/go/containeranalysis: v0.9.0
- cloud.google.com/go/datacatalog: v1.13.0
- cloud.google.com/go/dataflow: v0.8.0
- cloud.google.com/go/dataform: v0.7.0
- cloud.google.com/go/datafusion: v1.6.0
- cloud.google.com/go/datalabeling: v0.7.0
- cloud.google.com/go/dataplex: v1.6.0
- cloud.google.com/go/dataproc: v1.12.0
- cloud.google.com/go/dataqna: v0.7.0
- cloud.google.com/go/datastream: v1.7.0
- cloud.google.com/go/deploy: v1.8.0
- cloud.google.com/go/dialogflow: v1.32.0
- cloud.google.com/go/dlp: v1.9.0
- cloud.google.com/go/documentai: v1.18.0
- cloud.google.com/go/domains: v0.8.0
- cloud.google.com/go/edgecontainer: v1.0.0
- cloud.google.com/go/errorreporting: v0.3.0
- cloud.google.com/go/essentialcontacts: v1.5.0
- cloud.google.com/go/eventarc: v1.11.0
- cloud.google.com/go/filestore: v1.6.0
- cloud.google.com/go/functions: v1.13.0
- cloud.google.com/go/gaming: v1.9.0
- cloud.google.com/go/gkebackup: v0.4.0
- cloud.google.com/go/gkeconnect: v0.7.0
- cloud.google.com/go/gkehub: v0.12.0
- cloud.google.com/go/gkemulticloud: v0.5.0
- cloud.google.com/go/gsuiteaddons: v1.5.0
- cloud.google.com/go/iam: v0.13.0
- cloud.google.com/go/iap: v1.7.1
- cloud.google.com/go/ids: v1.3.0
- cloud.google.com/go/iot: v1.6.0
- cloud.google.com/go/kms: v1.10.1
- cloud.google.com/go/language: v1.9.0
- cloud.google.com/go/lifesciences: v0.8.0
- cloud.google.com/go/logging: v1.7.0
- cloud.google.com/go/longrunning: v0.4.1
- cloud.google.com/go/managedidentities: v1.5.0
- cloud.google.com/go/maps: v0.7.0
- cloud.google.com/go/mediatranslation: v0.7.0
- cloud.google.com/go/memcache: v1.9.0
- cloud.google.com/go/metastore: v1.10.0
- cloud.google.com/go/monitoring: v1.13.0
- cloud.google.com/go/networkconnectivity: v1.11.0
- cloud.google.com/go/networkmanagement: v1.6.0
- cloud.google.com/go/networksecurity: v0.8.0
- cloud.google.com/go/notebooks: v1.8.0
- cloud.google.com/go/optimization: v1.3.1
- cloud.google.com/go/orchestration: v1.6.0
- cloud.google.com/go/orgpolicy: v1.10.0
- cloud.google.com/go/osconfig: v1.11.0
- cloud.google.com/go/oslogin: v1.9.0
- cloud.google.com/go/phishingprotection: v0.7.0
- cloud.google.com/go/policytroubleshooter: v1.6.0
- cloud.google.com/go/privatecatalog: v0.8.0
- cloud.google.com/go/pubsublite: v1.7.0
- cloud.google.com/go/recaptchaenterprise/v2: v2.7.0
- cloud.google.com/go/recommendationengine: v0.7.0
- cloud.google.com/go/recommender: v1.9.0
- cloud.google.com/go/redis: v1.11.0
- cloud.google.com/go/resourcemanager: v1.7.0
- cloud.google.com/go/resourcesettings: v1.5.0
- cloud.google.com/go/retail: v1.12.0
- cloud.google.com/go/run: v0.9.0
- cloud.google.com/go/scheduler: v1.9.0
- cloud.google.com/go/secretmanager: v1.10.0
- cloud.google.com/go/security: v1.13.0
- cloud.google.com/go/securitycenter: v1.19.0
- cloud.google.com/go/servicedirectory: v1.9.0
- cloud.google.com/go/shell: v1.6.0
- cloud.google.com/go/spanner: v1.45.0
- cloud.google.com/go/speech: v1.15.0
- cloud.google.com/go/storagetransfer: v1.8.0
- cloud.google.com/go/talent: v1.5.0
- cloud.google.com/go/texttospeech: v1.6.0
- cloud.google.com/go/tpu: v1.5.0
- cloud.google.com/go/trace: v1.9.0
- cloud.google.com/go/translate: v1.7.0
- cloud.google.com/go/video: v1.15.0
- cloud.google.com/go/videointelligence: v1.10.0
- cloud.google.com/go/vision/v2: v2.7.0
- cloud.google.com/go/vmmigration: v1.6.0
- cloud.google.com/go/vmwareengine: v0.3.0
- cloud.google.com/go/vpcaccess: v1.6.0
- cloud.google.com/go/webrisk: v1.8.0
- cloud.google.com/go/websecurityscanner: v1.5.0
- cloud.google.com/go/workflows: v1.10.0
- github.com/googleapis/enterprise-certificate-proxy: [v0.2.3](https://github.com/googleapis/enterprise-certificate-proxy/tree/v0.2.3)
- go.etcd.io/gofail: v0.1.0
- google.golang.org/genproto/googleapis/api: dd9d682
- google.golang.org/genproto/googleapis/rpc: 28d5490

### Changed
- cloud.google.com/go/bigquery: v1.8.0 → v1.50.0
- cloud.google.com/go/datastore: v1.1.0 → v1.11.0
- cloud.google.com/go/firestore: v1.1.0 → v1.9.0
- cloud.google.com/go/pubsub: v1.3.1 → v1.30.0
- cloud.google.com/go: v0.97.0 → v0.110.0
- github.com/Azure/azure-sdk-for-go: [v55.0.0+incompatible → v68.0.0+incompatible](https://github.com/Azure/azure-sdk-for-go/compare/v55.0.0...v68.0.0)
- github.com/Azure/go-autorest/autorest/adal: [v0.9.20 → v0.9.23](https://github.com/Azure/go-autorest/autorest/adal/compare/v0.9.20...v0.9.23)
- github.com/Azure/go-autorest/autorest/validation: [v0.1.0 → v0.3.1](https://github.com/Azure/go-autorest/autorest/validation/compare/v0.1.0...v0.3.1)
- github.com/Azure/go-autorest/autorest: [v0.11.27 → v0.11.29](https://github.com/Azure/go-autorest/autorest/compare/v0.11.27...v0.11.29)
- github.com/Microsoft/go-winio: [v0.4.17 → v0.6.0](https://github.com/Microsoft/go-winio/compare/v0.4.17...v0.6.0)
- github.com/cenkalti/backoff/v4: [v4.1.3 → v4.2.1](https://github.com/cenkalti/backoff/v4/compare/v4.1.3...v4.2.1)
- github.com/census-instrumentation/opencensus-proto: [v0.2.1 → v0.4.1](https://github.com/census-instrumentation/opencensus-proto/compare/v0.2.1...v0.4.1)
- github.com/cespare/xxhash/v2: [v2.1.2 → v2.2.0](https://github.com/cespare/xxhash/v2/compare/v2.1.2...v2.2.0)
- github.com/cilium/ebpf: [v0.7.0 → v0.9.1](https://github.com/cilium/ebpf/compare/v0.7.0...v0.9.1)
- github.com/cncf/udpa/go: [04548b0 → c52dc94](https://github.com/cncf/udpa/go/compare/04548b0...c52dc94)
- github.com/cncf/xds/go: [cb28da3 → 06c439d](https://github.com/cncf/xds/go/compare/cb28da3...06c439d)
- github.com/cockroachdb/datadriven: [bf6692d → v1.0.2](https://github.com/cockroachdb/datadriven/compare/bf6692d...v1.0.2)
- github.com/container-storage-interface/spec: [v1.7.0 → v1.8.0](https://github.com/container-storage-interface/spec/compare/v1.7.0...v1.8.0)
- github.com/containerd/cgroups: [v1.0.1 → v1.1.0](https://github.com/containerd/cgroups/compare/v1.0.1...v1.1.0)
- github.com/containerd/ttrpc: [v1.1.0 → v1.2.2](https://github.com/containerd/ttrpc/compare/v1.1.0...v1.2.2)
- github.com/coredns/caddy: [v1.1.0 → v1.1.1](https://github.com/coredns/caddy/compare/v1.1.0...v1.1.1)
- github.com/coreos/go-oidc: [v2.1.0+incompatible → v2.2.1+incompatible](https://github.com/coreos/go-oidc/compare/v2.1.0...v2.2.1)
- github.com/coreos/go-semver: [v0.3.0 → v0.3.1](https://github.com/coreos/go-semver/compare/v0.3.0...v0.3.1)
- github.com/coreos/go-systemd/v22: [v22.4.0 → v22.5.0](https://github.com/coreos/go-systemd/v22/compare/v22.4.0...v22.5.0)
- github.com/docker/distribution: [v2.8.1+incompatible → v2.8.2+incompatible](https://github.com/docker/distribution/compare/v2.8.1...v2.8.2)
- github.com/envoyproxy/go-control-plane: [49ff273 → v0.10.3](https://github.com/envoyproxy/go-control-plane/compare/49ff273...v0.10.3)
- github.com/envoyproxy/protoc-gen-validate: [v0.1.0 → v0.9.1](https://github.com/envoyproxy/protoc-gen-validate/compare/v0.1.0...v0.9.1)
- github.com/frankban/quicktest: [v1.11.3 → v1.14.0](https://github.com/frankban/quicktest/compare/v1.11.3...v1.14.0)
- github.com/fvbommel/sortorder: [v1.0.1 → v1.1.0](https://github.com/fvbommel/sortorder/compare/v1.0.1...v1.1.0)
- github.com/go-logr/logr: [v1.2.3 → v1.2.4](https://github.com/go-logr/logr/compare/v1.2.3...v1.2.4)
- github.com/go-task/slim-sprig: [348f09d → 52ccab3](https://github.com/go-task/slim-sprig/compare/348f09d...52ccab3)
- github.com/gofrs/uuid: [v4.0.0+incompatible → v4.4.0+incompatible](https://github.com/gofrs/uuid/compare/v4.0.0...v4.4.0)
- github.com/golang-jwt/jwt/v4: [v4.4.2 → v4.5.0](https://github.com/golang-jwt/jwt/v4/compare/v4.4.2...v4.5.0)
- github.com/google/gofuzz: [v1.1.0 → v1.2.0](https://github.com/google/gofuzz/compare/v1.1.0...v1.2.0)
- github.com/googleapis/gax-go/v2: [v2.1.1 → v2.7.1](https://github.com/googleapis/gax-go/v2/compare/v2.1.1...v2.7.1)
- github.com/inconshreveable/mousetrap: [v1.0.1 → v1.1.0](https://github.com/inconshreveable/mousetrap/compare/v1.0.1...v1.1.0)
- github.com/mitchellh/go-wordwrap: [v1.0.0 → v1.0.1](https://github.com/mitchellh/go-wordwrap/compare/v1.0.0...v1.0.1)
- github.com/onsi/ginkgo/v2: [v2.9.1 → v2.9.4](https://github.com/onsi/ginkgo/v2/compare/v2.9.1...v2.9.4)
- github.com/onsi/gomega: [v1.27.4 → v1.27.6](https://github.com/onsi/gomega/compare/v1.27.4...v1.27.6)
- github.com/opencontainers/runc: [v1.1.4 → v1.1.7](https://github.com/opencontainers/runc/compare/v1.1.4...v1.1.7)
- github.com/rogpeppe/go-internal: [v1.10.0 → v1.6.1](https://github.com/rogpeppe/go-internal/compare/v1.10.0...v1.6.1)
- github.com/seccomp/libseccomp-golang: [f33da4d → v0.10.0](https://github.com/seccomp/libseccomp-golang/compare/f33da4d...v0.10.0)
- github.com/spf13/cobra: [v1.6.0 → v1.7.0](https://github.com/spf13/cobra/compare/v1.6.0...v1.7.0)
- github.com/stretchr/testify: [v1.8.1 → v1.8.2](https://github.com/stretchr/testify/compare/v1.8.1...v1.8.2)
- github.com/vishvananda/netns: [v0.0.2 → v0.0.4](https://github.com/vishvananda/netns/compare/v0.0.2...v0.0.4)
- github.com/xlab/treeprint: [v1.1.0 → v1.2.0](https://github.com/xlab/treeprint/compare/v1.1.0...v1.2.0)
- go.etcd.io/bbolt: v1.3.6 → v1.3.7
- go.etcd.io/etcd/api/v3: v3.5.7 → v3.5.9
- go.etcd.io/etcd/client/pkg/v3: v3.5.7 → v3.5.9
- go.etcd.io/etcd/client/v2: v2.305.7 → v2.305.9
- go.etcd.io/etcd/client/v3: v3.5.7 → v3.5.9
- go.etcd.io/etcd/pkg/v3: v3.5.7 → v3.5.9
- go.etcd.io/etcd/raft/v3: v3.5.7 → v3.5.9
- go.etcd.io/etcd/server/v3: v3.5.7 → v3.5.9
- go.opencensus.io: v0.23.0 → v0.24.0
- go.uber.org/atomic: v1.7.0 → v1.10.0
- go.uber.org/multierr: v1.6.0 → v1.11.0
- golang.org/x/crypto: v0.1.0 → v0.6.0
- golang.org/x/mod: v0.9.0 → v0.10.0
- golang.org/x/net: v0.8.0 → v0.9.0
- golang.org/x/oauth2: ee48083 → v0.6.0
- golang.org/x/sys: v0.6.0 → v0.7.0
- golang.org/x/term: v0.6.0 → v0.7.0
- golang.org/x/text: v0.8.0 → v0.9.0
- golang.org/x/time: 90d013b → v0.3.0
- golang.org/x/tools: v0.7.0 → v0.8.0
- google.golang.org/api: v0.60.0 → v0.114.0
- google.golang.org/genproto: c8bf987 → 0005af6
- google.golang.org/grpc: v1.51.0 → v1.54.0
- google.golang.org/protobuf: v1.28.1 → v1.30.0
- gopkg.in/gcfg.v1: v1.2.0 → v1.2.3
- gopkg.in/natefinch/lumberjack.v2: v2.0.0 → v2.2.1
- gopkg.in/warnings.v0: v0.1.1 → v0.1.2
- k8s.io/klog/v2: v2.90.1 → v2.100.1
- k8s.io/kube-openapi: 15aac26 → 7828149
- k8s.io/utils: a36077c → d93618c
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.1.1 → v0.1.2

### Removed
- github.com/certifi/gocertifi: [2c3bb06](https://github.com/certifi/gocertifi/tree/2c3bb06)
- github.com/cockroachdb/errors: [v1.2.4](https://github.com/cockroachdb/errors/tree/v1.2.4)
- github.com/cockroachdb/logtags: [eb05cc2](https://github.com/cockroachdb/logtags/tree/eb05cc2)
- github.com/getsentry/raven-go: [v0.2.0](https://github.com/getsentry/raven-go/tree/v0.2.0)