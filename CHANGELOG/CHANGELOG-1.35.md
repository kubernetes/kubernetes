<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.35.0](#v1350)
  - [Downloads for v1.35.0](#downloads-for-v1350)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.34.0](#changelog-since-v1340)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Documentation](#documentation)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.35.0-rc.1](#v1350-rc1)
  - [Downloads for v1.35.0-rc.1](#downloads-for-v1350-rc1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.35.0-rc.0](#changelog-since-v1350-rc0)
  - [Changes by Kind](#changes-by-kind-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.35.0-rc.0](#v1350-rc0)
  - [Downloads for v1.35.0-rc.0](#downloads-for-v1350-rc0)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.35.0-beta.0](#changelog-since-v1350-beta0)
  - [Changes by Kind](#changes-by-kind-2)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.35.0-beta.0](#v1350-beta0)
  - [Downloads for v1.35.0-beta.0](#downloads-for-v1350-beta0)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.35.0-alpha.3](#changelog-since-v1350-alpha3)
  - [Changes by Kind](#changes-by-kind-3)
    - [API Change](#api-change-1)
    - [Feature](#feature-3)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.35.0-alpha.3](#v1350-alpha3)
  - [Downloads for v1.35.0-alpha.3](#downloads-for-v1350-alpha3)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
    - [Container Images](#container-images-4)
  - [Changelog since v1.35.0-alpha.2](#changelog-since-v1350-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-4)
    - [API Change](#api-change-2)
    - [Feature](#feature-4)
    - [Bug or Regression](#bug-or-regression-4)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)
- [v1.35.0-alpha.2](#v1350-alpha2)
  - [Downloads for v1.35.0-alpha.2](#downloads-for-v1350-alpha2)
    - [Source Code](#source-code-5)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
    - [Container Images](#container-images-5)
  - [Changelog since v1.35.0-alpha.1](#changelog-since-v1350-alpha1)
  - [Changes by Kind](#changes-by-kind-5)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-3)
    - [Feature](#feature-5)
    - [Documentation](#documentation-1)
    - [Bug or Regression](#bug-or-regression-5)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-5)
    - [Added](#added-5)
    - [Changed](#changed-5)
    - [Removed](#removed-5)
- [v1.35.0-alpha.1](#v1350-alpha1)
  - [Downloads for v1.35.0-alpha.1](#downloads-for-v1350-alpha1)
    - [Source Code](#source-code-6)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
    - [Container Images](#container-images-6)
  - [Changelog since v1.34.0](#changelog-since-v1340-1)
  - [Changes by Kind](#changes-by-kind-6)
    - [API Change](#api-change-4)
    - [Feature](#feature-6)
    - [Bug or Regression](#bug-or-regression-6)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-5)
  - [Dependencies](#dependencies-6)
    - [Added](#added-6)
    - [Changed](#changed-6)
    - [Removed](#removed-6)

<!-- END MUNGE: GENERATED_TOC -->

# v1.35.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.35.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes.tar.gz) | `478ae8101675fa873a3ad84c81c91604e70bdb947e3379564907916c8a3a1d4a0b7d2077e1d2701f18f2509a6fce0997d93a441ef6d1a17a2e90fdffdd4c13ec`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-src.tar.gz) | `dc9fc72736999bc40fdf28a7668c8e183effe135893c98f0773b0a50fe018c2f49156026c490f201def57645bf6172c81e07c1c6cb2d80bfb6b246c94fb4c5aa`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-darwin-amd64.tar.gz) | `e7d510566442afd96dd3759764b573719469bb0ef00086d536bd7af0b8af29ddf150e6ece5ae95856daaaf7f2454f45755ac300648c692508e445aca7a8bd0de`
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-darwin-arm64.tar.gz) | `cd3b216a5418ef2eb00aeb74bf0ebae34c41aa16419bd5bbe5cbb5d394570a38f54c88294aaa5bd7c27ef28c4f1aee2b5658beb4cd025258b6bbd522e8d499bc`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-linux-386.tar.gz) | `50250aefecc03afe5a6b1be8dffbd58efb4814fed2aae299ac3bbd3b32a40b47697897bafcc36f31f226c5fd2b185cb970e64674aa9ee60412e122128487598d`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-linux-amd64.tar.gz) | `a1469924896411ab3365628b301d2bbacaf235908cea47308498c9c351a17462ab4154928ef6f91cee849ff52600e394f2abe70f5165371ccfe6638446699d2c`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-linux-arm.tar.gz) | `df921ad2702a8bc90b8797d97e5ddba5d7d077d18f3b9e53a4594a432f628f52842ee5e26f70c16a82b4decf7c72cba1d04c43163c85026f9b0610fbde63e183`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-linux-arm64.tar.gz) | `0b332e13c9bb52093f57c4f2ae4ab103bc7f51e4c5dad2859300e7ece09ef303a9345ed3aea4d050b287f52dd8ed8d7cf9185c9e40ea5cc900c8d34e63eec83d`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-linux-ppc64le.tar.gz) | `07789dc2ec7e8439774d88437f0b1ee35d6b60a8bd23055b93dcf1461de5ae69aba0e0e99a0202892f6c70217388646e1592b087f048bb57e5ab10b1b0dfa956`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-linux-s390x.tar.gz) | `6563b8d452d29e7f155563294478e39dba7311dd086cf9fb0bc62c94a139b7f5d81a5716880d8072cd864948988e68f2dcd607a8ec79e339224ed5f4bcd48dc9`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-windows-386.tar.gz) | `522f96799bdaacdd1d10ab4c3a58d8fd86e45e6326c3b6538cc079ca951c28916bd1c8c9bb1d98f6257be0ba1ed91e97614407fe11a1c4bbea2c2052ba0feca7`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-windows-amd64.tar.gz) | `149145263071c8e1a4d73efe4d1c868286e7cea37629f1c076d2f2683e6b63fb3387d867f3283c9950a3b5b830f005019fa03874e4d53dfa9ad489aaaa9f535b`
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-client-windows-arm64.tar.gz) | `2cffd56e01eaf24ace819cf9f4ef94187185978c8fa1192fd9d47236824ccfe745fe649d38c4351a016e0406bcfd1944178cb93af67b5e69015c04ab2ca5bf7c`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-server-linux-amd64.tar.gz) | `23af53c49de841a0d5c19d9525d820cecc9d55367c132296a5f381d051438bf06dcddff3d0236df8ba6011a6aa5d0ffc31960d277c7f53a0ad98e66d6f8d6a0a`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-server-linux-arm64.tar.gz) | `fd245273c6ace20abc893f868d678c4a24c0dbe7d5340087f852d245e59329e66f79afce489dc1b396908d2f005b132eca8d15a7664508fe923627bb2eddee18`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-server-linux-ppc64le.tar.gz) | `68c48db8537c0470d2245740b8cdf3225efafc48a96646e369137e35931bd43324caf1394ee4b31774b0f43d44e6a4eaa5976186248a114d0e0feb2cb8953edc`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-server-linux-s390x.tar.gz) | `dd71c4b5ab213452d41059772de3b0db2c71fc6f958280694b2c1b20151bded5b6beb1b03a40dc683ce2d587e9a8bbf3bf486b3965064945803af4f10557558e`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-node-linux-amd64.tar.gz) | `179278fecb65d246443f58cef00ca2f2a9d0ac6fbdb310994f0ac7fca249f7bdc1c79ea7f3e5455c1e2d2460f5447d006bfa579f97b502ee7034b2a1927f934a`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-node-linux-arm64.tar.gz) | `01178703c84e0f671770e53024e3cc53f540c0cf93b0804d35884a777c3e3bc44c44d62b6fd25204348986fa589969a9255c0ef04235a0bb9d5560b09867aa0b`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-node-linux-ppc64le.tar.gz) | `05d1ae963d5c4a382d380cb4f4cdfa924fa8a311953b5eaefe66b8696cebf14bffb13bda8ea784ca5fa1dd073c82ee148faa9a50911449cefad16fe2e800d7c1`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-node-linux-s390x.tar.gz) | `b7501e91153d062c7c545ef9900faf9b29826b6ff5ec5320f6a799d3d3b479f6ae79092909a1905e055b72dd540a9c8fb02b2d0655f6957cd0b4b7b2e9c18909`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0/kubernetes-node-windows-amd64.tar.gz) | `f54c606e8ecc29b4ba4ef4570f679352f66cbae1f1bd4f49db5e18227b00ed0e6d8dd47422390fd2a3b87d837cf39dae58a260208096169a3aabef9e874c7586`

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.
name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.34.0

## Urgent Upgrade Notes 

### (No, really, you MUST read this before you upgrade)

- ACTION REQUIRED:
  
  Removed the `--pod-infra-container-image` flag from `kubelet` command line. For `non-kubeadm` clusters, users must manually remove this flag from their `kubelet` configuration to prevent startup failures before upgrading `kubelet`. For `kubeadm` clusters, if users pass extra arguments to the `kubelet` like `--pod-infra-container-image`, it will be written to the `kubelet` env file during the `init` phase. `kubeadm` does not remove it during the `init` or `join` phase, so users must manually remove it from `extraArgs` in the `kubelet` configuration file. ([#133779](https://github.com/kubernetes/kubernetes/pull/133779), [@carlory](https://github.com/carlory))
 - ACTION REQUIRED:
  
  vendor: Updated `k8s.io/system-validators` to `v1.12.1`. The cgroups validator now throws an error instead of a warning if cgroups v1 is detected on the host and the provided KubeletVersion is `v1.35` or newer.
  
  kubeadm: Started using `k8s.io/system-validators` `v1.12.1` in `kubeadm` `v1.35`. During `kubeadm init`, `kubeadm join`, and `kubeadm upgrade`, the SystemVerification preflight check throws an error if cgroups v1 is detected and the detected `kubelet` version is `v1.35` or newer. For older versions of `kubelet`, a preflight warning is displayed.
  
  To allow cgroups v1 with `kubeadm` and `kubelet` version `v1.35` or newer, you must:
  - Ignore the error from the SystemVerification preflight check by `kubeadm`.
  - Edit the `kube-system/kubelet-config` ConfigMap and add the `failCgroupV1: false` field before upgrading. ([#134744](https://github.com/kubernetes/kubernetes/pull/134744), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
 
## Changes by Kind

### Deprecation

- ACTION REQUIRED: `failCgroupV1` will be set to true from 1.35. 
  This means that nodes will not start on a cgroup v1 by default. This puts cgroup v1 into a deprecated state. ([#134298](https://github.com/kubernetes/kubernetes/pull/134298), [@kannon92](https://github.com/kannon92))
- Marked `ipvs` mode in kube-proxy as deprecated, which will be removed in a future version of Kubernetes. Users are encouraged to migrate to `nftables`. ([#134539](https://github.com/kubernetes/kubernetes/pull/134539), [@adrianmoisey](https://github.com/adrianmoisey))

### API Change

- Added `ObservedGeneration` to CustomResourceDefinition conditions. ([#134984](https://github.com/kubernetes/kubernetes/pull/134984), [@michaelasp](https://github.com/michaelasp))
- Added `WithOrigin` within `apis/core/validation` with adjusted tests. ([#132825](https://github.com/kubernetes/kubernetes/pull/132825), [@PatrickLaabs](https://github.com/PatrickLaabs))
- Added scoring for the prioritized list feature so nodes that best satisfy the highest-ranked subrequests were chosen. ([#134711](https://github.com/kubernetes/kubernetes/pull/134711), [@mortent](https://github.com/mortent)) [SIG Node, Scheduling and Testing]
- Added the `--min-compatibility-version` flag to `kube-apiserver`, `kube-controller-manager`, and `kube-scheduler`. ([#133980](https://github.com/kubernetes/kubernetes/pull/133980), [@siyuanfoundation](https://github.com/siyuanfoundation)) [SIG API Machinery, Architecture, Cluster Lifecycle, Etcd, Scheduling and Testing]
- Added the `StorageVersionMigration` `v1beta1` API and removed the `v1alpha1` API.
  
  ACTION REQUIRED: The `v1alpha1` API is no longer supported. Users must remove any `v1alpha1` resources before upgrading. ([#134784](https://github.com/kubernetes/kubernetes/pull/134784), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth, Etcd and Testing]
- Added validation to ensure `log-flush-frequency` is a positive value, returning an error instead of causing a panic. ([#133540](https://github.com/kubernetes/kubernetes/pull/133540), [@BenTheElder](https://github.com/BenTheElder)) [SIG Architecture, Instrumentation, Network and Node]
- All containers are restarted when a source container in a restart policy rule exits. This alpha feature is gated behind `RestartAllContainersOnContainerExit`. ([#134345](https://github.com/kubernetes/kubernetes/pull/134345), [@yuanwang04](https://github.com/yuanwang04)) [SIG Apps, Node and Testing]
- CSI drivers can now opt in to receive service account tokens via the secrets field instead of volume context by setting `spec.serviceAccountTokenInSecrets: true` in the CSIDriver object. This prevents tokens from being exposed in logs and other outputs. The feature is gated by the `CSIServiceAccountTokenSecrets` feature gate (beta in `v1.35`). ([#134826](https://github.com/kubernetes/kubernetes/pull/134826), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth, Storage and Testing]
- Changed kuberc configuration schema. Two new optional fields added to kuberc configuration, `credPluginPolicy` and `credPluginAllowlist`. This is documented in [KEP-3104](https://github.com/kubernetes/enhancements/blob/master/keps/sig-cli/3104-introduce-kuberc/README.md#allowlist-design-details) and documentation is added to the website by [kubernetes/website#52877](https://github.com/kubernetes/website/pull/52877) ([#134870](https://github.com/kubernetes/kubernetes/pull/134870), [@pmengelbert](https://github.com/pmengelbert)) [SIG API Machinery, Architecture, Auth, CLI, Instrumentation and Testing]
- DRA device taints: `DeviceTaintRule` status provides information about the rule, including whether Pods still need to be evicted (`EvictionInProgress` condition). The newly added `None` effect can be used to preview what a `DeviceTaintRule` would do if it used the `NoExecute` effect and to taint devices (`device health`) without immediately affecting scheduling or running Pods. ([#134152](https://github.com/kubernetes/kubernetes/pull/134152), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Node, Release, Scheduling and Testing]
- DRA: The `DynamicResourceAllocation` feature gate for the core functionality (GA in `v1.34`) has now been locked to enabled-by-default and cannot be disabled anymore. ([#134452](https://github.com/kubernetes/kubernetes/pull/134452), [@pohly](https://github.com/pohly)) [SIG Auth, Node, Scheduling and Testing]
- Enabled `kubectl get -o kyaml` by default. To disable it, set `KUBECTL_KYAML=false`. ([#133327](https://github.com/kubernetes/kubernetes/pull/133327), [@thockin](https://github.com/thockin))
- Enabled in-place resizing of pod-level resources.  
  - Added `Resources` in `PodStatus` to capture resources set in the pod-level cgroup.  
  - Added `AllocatedResources` in `PodStatus` to capture resources requested in the `PodSpec`. ([#132919](https://github.com/kubernetes/kubernetes/pull/132919), [@ndixita](https://github.com/ndixita)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Instrumentation, Node, Scheduling and Testing]
- Enabled the `NominatedNodeNameForExpectation` feature in kube-scheduler by default.
  - Enabled the `ClearingNominatedNodeNameAfterBinding` feature in kube-apiserver by default. ([#135103](https://github.com/kubernetes/kubernetes/pull/135103), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Enhanced discovery responses to merge API groups and resources from all peer apiservers when the `UnknownVersionInteroperabilityProxy` feature is enabled. ([#133648](https://github.com/kubernetes/kubernetes/pull/133648), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Auth, Cloud Provider, Node, Scheduling and Testing]
- Extended `core/v1` `Toleration` to support numeric comparison operators (`Gt`,`Lt`). ([#134665](https://github.com/kubernetes/kubernetes/pull/134665), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Node, Scheduling, Testing and Windows]
- Feature gate dependencies are now explicit, and validated at startup. A feature can no longer be enabled if it depends on a disabled feature. In particular, this means that `AllAlpha=true` will no longer work without enabling disabled-by-default beta features that are depended on (either with `AllBeta=true` or explicitly enumerating the disabled dependencies). ([#133697](https://github.com/kubernetes/kubernetes/pull/133697), [@tallclair](https://github.com/tallclair)) [SIG API Machinery, Architecture, Cluster Lifecycle and Node]
- Generated OpenAPI model packages for API types into `zz_generated.model_name.go` files, accessible via the `OpenAPIModelName()` function. This allows API authors to declare desired OpenAPI model packages instead of relying on the Go package path of API types. ([#131755](https://github.com/kubernetes/kubernetes/pull/131755), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Implemented constrained impersonation as described in [KEP-5284](https://kep.k8s.io/5284). ([#134803](https://github.com/kubernetes/kubernetes/pull/134803), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Introduced a new declarative validation tag `+k8s:customUnique` to control listmap uniqueness. ([#134279](https://github.com/kubernetes/kubernetes/pull/134279), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Auth]
- Introduced a structured and versioned `v1alpha1` response for the `statusz` endpoint. ([#134313](https://github.com/kubernetes/kubernetes/pull/134313), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation, Network, Node, Scheduling and Testing]
- Introduced a structured and versioned `v1alpha1` response format for the `flagz` endpoint. ([#134995](https://github.com/kubernetes/kubernetes/pull/134995), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Architecture, Instrumentation, Network, Node, Scheduling and Testing]
- Introduced the GangScheduling kube-scheduler plugin to support "all-or-nothing" scheduling using the `scheduling.k8s.io/v1alpha1` Workload API. ([#134722](https://github.com/kubernetes/kubernetes/pull/134722), [@macsko](https://github.com/macsko)) [SIG API Machinery, Apps, Auth, CLI, Etcd, Scheduling and Testing]
- Introduced the Node Declared Features capability (alpha), which includes:
  - A new `Node.Status.DeclaredFeatures` field for publishing node-specific features.
  - A `component-helpers` library for feature registration and inference.
  - A `NodeDeclaredFeatures` scheduler plugin to match pods with nodes that provide required features.
  - A `NodeDeclaredFeatureValidator` admission plugin to validate pod updates against a node's declared features. ([#133389](https://github.com/kubernetes/kubernetes/pull/133389), [@pravk03](https://github.com/pravk03)) [SIG API Machinery, Apps, Node, Release, Scheduling and Testing]
- Introduced the `scheduling.k8s.io/v1alpha1` Workload API to express workload-level scheduling requirements and allow the kube-scheduler to act on them. ([#134564](https://github.com/kubernetes/kubernetes/pull/134564), [@macsko](https://github.com/macsko)) [SIG API Machinery, Apps, CLI, Etcd, Scheduling and Testing]
- Introduced the alpha `MutableSchedulingDirectivesForSuspendedJobs` feature gate (disabled by default), which allows mutating a Job's scheduling directives while the Job is suspended. 
  It also updates the Job controller to clears the `status.startTime` field for suspended Jobs. ([#135104](https://github.com/kubernetes/kubernetes/pull/135104), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- Kube-apiserver: Fixed a `v1.34` regression in `CustomResourceDefinition` handling that incorrectly warned about unrecognized formats on number and integer properties. ([#133896](https://github.com/kubernetes/kubernetes/pull/133896), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Contributor Experience, Network, Node and Scheduling]
- Kube-apiserver: Fixed a possible panic validating a custom resource whose `CustomResourceDefinition` indicates a status subresource exists, but which does not define a `status` property in the `openAPIV3Schema`. ([#133721](https://github.com/kubernetes/kubernetes/pull/133721), [@fusida](https://github.com/fusida)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Kubernetes API Go types removed runtime use of the `github.com/gogo/protobuf` library, and are no longer registered into the global gogo type registry. Kubernetes API Go types were not suitable for use with the `google.golang.org/protobuf` library, and no longer implement `ProtoMessage()` by default to avoid accidental incompatible use. If removal of these marker methods impacts your use, it can be re-enabled for one more release with a `kubernetes_protomessage_one_more_release` build tag, but will be removed in `v1.36`. ([#134256](https://github.com/kubernetes/kubernetes/pull/134256), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- Made node affinity in Persistent Volume mutable. ([#134339](https://github.com/kubernetes/kubernetes/pull/134339), [@huww98](https://github.com/huww98)) [SIG API Machinery, Apps and Node]
- Moved the `ImagePullIntent` and `ImagePulledRecord` objects used by the kubelet to track image pulls to the `v1beta1` API version. ([#132579](https://github.com/kubernetes/kubernetes/pull/132579), [@stlaz](https://github.com/stlaz)) [SIG Auth and Node]
- Pod resize now only allows CPU and memory resources; other resource types are forbidden. ([#135084](https://github.com/kubernetes/kubernetes/pull/135084), [@tallclair](https://github.com/tallclair)) [SIG Apps, Node and Testing]
- Prevented Pods from being scheduled onto nodes that lack the required CSI driver. ([#135012](https://github.com/kubernetes/kubernetes/pull/135012), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Scheduling, Storage and Testing]
- Promoted HPA configurable tolerance to beta. The `HPAConfigurableTolerance` feature gate has now been enabled by default. ([#133128](https://github.com/kubernetes/kubernetes/pull/133128), [@jm-franc](https://github.com/jm-franc)) [SIG API Machinery and Autoscaling]
- Promoted ReplicaSet and Deployment `.status.terminatingReplicas` tracking to beta. The `DeploymentReplicaSetTerminatingReplicas` feature gate is now enabled by default. ([#133087](https://github.com/kubernetes/kubernetes/pull/133087), [@atiratree](https://github.com/atiratree)) [SIG API Machinery, Apps and Testing]
- Promoted `PodObservedGenerationTracking` to GA. ([#134948](https://github.com/kubernetes/kubernetes/pull/134948), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Promoted the `JobManagedBy` feature to general availability. The `JobManagedBy` feature gate was locked to `true` and will be removed in a future Kubernetes release. ([#135080](https://github.com/kubernetes/kubernetes/pull/135080), [@dejanzele](https://github.com/dejanzele)) [SIG API Machinery, Apps and Testing]
- Promoted the `MaxUnavailableStatefulSet` feature to beta and enabling it by default. ([#133153](https://github.com/kubernetes/kubernetes/pull/133153), [@helayoty](https://github.com/helayoty)) [SIG API Machinery and Apps]
- Removed the `StrictCostEnforcementForVAP` and `StrictCostEnforcementForWebhooks` feature gates, which were locked since `v1.32`. ([#134994](https://github.com/kubernetes/kubernetes/pull/134994), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, Node and Testing]
- Scheduler: Added the `bindingTimeout` argument to the DynamicResources plugin configuration, allowing customization of the wait duration in `PreBind` for device binding conditions.
  Defaults to 10 minutes when `DRADeviceBindingConditions` and `DRAResourceClaimDeviceStatus` are both enabled. ([#134905](https://github.com/kubernetes/kubernetes/pull/134905), [@fj-naji](https://github.com/fj-naji)) [SIG Node and Scheduling]
- The DRA device taints and toleration feature received a separate feature gate, `DRADeviceTaintRules`, which controlled support for `DeviceTaintRules`. This allowed disabling it while keeping `DRADeviceTaints` enabled so that tainting via `ResourceSlices` continued to work. ([#135068](https://github.com/kubernetes/kubernetes/pull/135068), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Node, Scheduling and Testing]
- The Pod Certificates feature moved to beta. The `PodCertificateRequest` feature gate is set disabled by default. To use the feature, users must enable the certificates API groups in `v1beta1` and enable the `PodCertificateRequest` feature gate. The `UserAnnotations` field was added to the `PodCertificateProjection` API and the corresponding `UnverifiedUserAnnotations` field was added to the `PodCertificateRequest` API. ([#134624](https://github.com/kubernetes/kubernetes/pull/134624), [@yt2985](https://github.com/yt2985)) [SIG API Machinery, Apps, Auth, Etcd, Instrumentation, Node and Testing]
- The `KubeletEnsureSecretPulledImages` feature was promoted to Beta and enabled by default. ([#135228](https://github.com/kubernetes/kubernetes/pull/135228), [@aramase](https://github.com/aramase)) [SIG Auth, Node and Testing]
- The `PreferSameZone` and `PreferSameNode` values for the Service
  `trafficDistribution` field graduated to general availability. The
  `PreferClose` value is now deprecated in favor of the more explicit
  `PreferSameZone`. ([#134457](https://github.com/kubernetes/kubernetes/pull/134457), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Apps, Network and Testing]
- Updated `ResourceQuota` to count device class requests within a `ResourceClaim` as two additional quotas when the `DRAExtendedResource` feature is enabled:
  - `requests.deviceclass.resource.k8s.io/<deviceclass>` is charged based on the worst-case number of devices requested.
  - Device classes mapping to an extended resource now consume `requests.<extended resource name>`. ([#134210](https://github.com/kubernetes/kubernetes/pull/134210), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Updated storage version for `MutatingAdmissionPolicy` to `v1beta1`. ([#133715](https://github.com/kubernetes/kubernetes/pull/133715), [@cici37](https://github.com/cici37)) [SIG API Machinery, Etcd and Testing]
- Updated the Partitionable Devices feature to support referencing counter sets across ResourceSlices within the same resource pool. Devices from incomplete pools were no longer considered for allocation. This change introduced backwards-incompatible updates to the alpha feature, requiring any ResourceSlices using it to be removed before upgrading or downgrading between v1.34 and v1.35. ([#134189](https://github.com/kubernetes/kubernetes/pull/134189), [@mortent](https://github.com/mortent)) [SIG API Machinery, Node, Scheduling and Testing]
- Upgraded the `PodObservedGenerationTracking` feature to beta in `v1.34` and removed the alpha version description from the OpenAPI specification. ([#133883](https://github.com/kubernetes/kubernetes/pull/133883), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))

### Feature

- Added `k8s-short-name` and `k8s-long-name` format validation tags to enforce DNS label and DNS subdomain compliance. ([#133894](https://github.com/kubernetes/kubernetes/pull/133894), [@lalitc375](https://github.com/lalitc375))
- Added `kubectl kuberc view` and `kubectl kuberc set` commands to perform operations against the `kuberc` file. ([#135003](https://github.com/kubernetes/kubernetes/pull/135003), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Added `kubelet` stress test for pod cleanup when rejection due to `VolumeAttachmentLimitExceeded`. ([#133357](https://github.com/kubernetes/kubernetes/pull/133357), [@torredil](https://github.com/torredil)) [SIG Node and Storage]
- Added `paths` section to kubelet `statusz` endpoint. ([#133239](https://github.com/kubernetes/kubernetes/pull/133239), [@Peac36](https://github.com/Peac36))
- Added a `source` label to the `resourceclaim_controller_resource_claims` metric.
  Added the `scheduler_resourceclaim_creates_total` metric for `DRAExtendedResource`. ([#134523](https://github.com/kubernetes/kubernetes/pull/134523), [@bitoku](https://github.com/bitoku)) [SIG Apps, Instrumentation, Node and Scheduling]
- Added a counter metric `kubelet_image_manager_ensure_image_requests_total{present_locally, pull_policy, pull_required}` that exposes details about `kubelet` ensuring an image exists on the node. ([#132644](https://github.com/kubernetes/kubernetes/pull/132644), [@stlaz](https://github.com/stlaz)) [SIG Auth and Node]
- Added additional event emissions during Pod resizing to provide clearer visibility when a Pod’s resize status changes. ([#134825](https://github.com/kubernetes/kubernetes/pull/134825), [@natasha41575](https://github.com/natasha41575))
- Added configurable per-device health check timeouts to the DRA health monitoring API. ([#135147](https://github.com/kubernetes/kubernetes/pull/135147), [@harche](https://github.com/harche)) [SIG Node]
- Added metrics for the `MaxUnavailable` feature in `StatefulSet`. ([#130951](https://github.com/kubernetes/kubernetes/pull/130951), [@Edwinhr716](https://github.com/Edwinhr716)) [SIG Apps and Instrumentation]
- Added paths section to scheduler `statusz` endpoint. ([#132606](https://github.com/kubernetes/kubernetes/pull/132606), [@Peac36](https://github.com/Peac36)) [SIG API Machinery, Architecture, Instrumentation, Network, Node, Scheduling and Testing]
- Added remote runtime and image `Close()` method to be able to close the connection. ([#133211](https://github.com/kubernetes/kubernetes/pull/133211), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Added support for tracing in `kubectl` with the `--profile=trace` flag. ([#134709](https://github.com/kubernetes/kubernetes/pull/134709), [@tchap](https://github.com/tchap))
- Added support for validating UUID format. ([#133948](https://github.com/kubernetes/kubernetes/pull/133948), [@lalitc375](https://github.com/lalitc375))
- Added the `-n` flag as a shorthand for `--namespace` in the `kubectl config set-context` command. ([#134384](https://github.com/kubernetes/kubernetes/pull/134384), [@tchap](https://github.com/tchap)) [SIG CLI and Testing]
- Added the `ChangeContainerStatusOnKubeletRestart` feature gate, which defaults to disabled. When the feature gate is disabled, `kubelet` does not change the Pod status upon restart, and Pods do not re-run startup probes after the `kubelet` restarts. ([#134746](https://github.com/kubernetes/kubernetes/pull/134746), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node and Testing]
- Added the `CloudControllerManagerWatchBasedRoutesReconciliation` feature gate. ([#131220](https://github.com/kubernetes/kubernetes/pull/131220), [@lukasmetzner](https://github.com/lukasmetzner)) [SIG API Machinery and Cloud Provider]
- Added the `UserNamespacesHostNetworkSupport` feature gate. This gate is disabled by default, and when enabled, allowed `hostNetwork` pods to use user namespaces. ([#134893](https://github.com/kubernetes/kubernetes/pull/134893), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Apps, Node and Testing]
- After fixing regressions detected in `v1.34`, the `SchedulerAsyncAPICalls` feature gate was re-enabled by default. ([#135059](https://github.com/kubernetes/kubernetes/pull/135059), [@macsko](https://github.com/macsko))
- Changed `WaitForNamedCacheSync` to `WaitForNamedCacheSyncWithContext`. ([#133904](https://github.com/kubernetes/kubernetes/pull/133904), [@aditigupta96](https://github.com/aditigupta96)) [SIG API Machinery, Apps, Auth and Network]
- DRA: the resource.k8s.io API now uses the v1 API version (introduced in 1.34) as default storage version. Downgrading to 1.33 is not supported. ([#133876](https://github.com/kubernetes/kubernetes/pull/133876), [@kei01234kei](https://github.com/kei01234kei)) [SIG API Machinery, Etcd and Testing]
- Enabled the `MutableCSINodeAllocatableCount` feature gate by default in beta. ([#134647](https://github.com/kubernetes/kubernetes/pull/134647), [@torredil](https://github.com/torredil))
- Enabled the `WatchListClient` feature gate. ([#134180](https://github.com/kubernetes/kubernetes/pull/134180), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery, Apps, Auth, CLI, Instrumentation, Node and Testing]
- Enabled the feature gate `ContainerRestartRules` by default. The `ContainerRestartRules` feature has been promoted to beta. Fixed a bug in this feature that caused probes to continue to run even if the container has terminated and is not restartable. ([#134631](https://github.com/kubernetes/kubernetes/pull/134631), [@yuanwang04](https://github.com/yuanwang04))
- Graduated the `PodTopologyLabelsAdmission` feature gate to Beta and enabled it by default.
  Pods now receive `topology.kubernetes.io/zone` and `topology.kubernetes.io/region` labels automatically when their assigned Node has these labels. ([#135158](https://github.com/kubernetes/kubernetes/pull/135158), [@andrewsykim](https://github.com/andrewsykim))
- Graduated the fine-grained supplemental groups policy (KEP-3619) to GA. ([#135088](https://github.com/kubernetes/kubernetes/pull/135088), [@everpeace](https://github.com/everpeace)) [SIG Node and Testing]
- Graduated the image volume source feature to Beta and enabled it by default. ([#135195](https://github.com/kubernetes/kubernetes/pull/135195), [@haircommander](https://github.com/haircommander)) [SIG Apps, Instrumentation, Node and Testing]
- Implemented opportunistic batching (KEP-5598) to optimize scheduling for pods with identical scheduling requirements. ([#135231](https://github.com/kubernetes/kubernetes/pull/135231), [@bwsalmon](https://github.com/bwsalmon)) [SIG Node, Scheduling, Storage and Testing]
- Implemented scoring for DRA-backed extended resources. ([#134058](https://github.com/kubernetes/kubernetes/pull/134058), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Scheduling and Testing]
- Improved throughput in the `real-FIFO` queue used by `informers` and `controllers` by adding batch handling for processing watch events. ([#132240](https://github.com/kubernetes/kubernetes/pull/132240), [@yue9944882](https://github.com/yue9944882)) [SIG API Machinery, Scheduling and Storage]
- Introduced end-to-end tests to verify component invariant metrics across the entire test suite. ([#133394](https://github.com/kubernetes/kubernetes/pull/133394), [@BenTheElder](https://github.com/BenTheElder))
- Introduced new kubelet metrics for the Ensure Secret Pulled Images KEP, including:
      - `kubelet_imagemanager_ondisk_pullintents` for tracking pull intent records on disk
      - `kubelet_imagemanager_ondisk_pulledrecords` for tracking pulled image records on disk
      - `kubelet_imagemanager_image_mustpull_checks_total{result}` for counting image must-pull verification checks. ([#132812](https://github.com/kubernetes/kubernetes/pull/132812), [@stlaz](https://github.com/stlaz)) [SIG Auth and Node]
- Introduced the `--as-user-extra` persistent flag in `kubectl`, which allows passing extra arguments during impersonation. ([#134378](https://github.com/kubernetes/kubernetes/pull/134378), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- K8s.io/apimachinery: Introduced a helper function to compare `resourceVersion` strings between two objects of the same resource. ([#134330](https://github.com/kubernetes/kubernetes/pull/134330), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- KEP-5440: Enabled support for resizing resources while a Job is suspended. This feature is alpha. ([#132441](https://github.com/kubernetes/kubernetes/pull/132441), [@kannon92](https://github.com/kannon92)) [SIG Apps and Testing]
- Kube-apiserver: Made the subresources `pods/exec`, `pods/attach`, and `pods/portforward` require `create` permission for both SPDY and Websocket API requests. Previously, SPDY requests required `create` permission, but Websocket requests only required `get` permission. This change is gated by the `AuthorizePodWebsocketUpgradeCreatePermission` feature-gate, which is enabled by default.
  
  Before upgrading to 1.35, ensure any custom ClusterRoles and Roles intended to grant `pods/exec`, `pods/attach`, or `pods/portforward` permission include the `create` verb. ([#134577](https://github.com/kubernetes/kubernetes/pull/134577), [@seans3](https://github.com/seans3)) [SIG API Machinery, Auth, Node and Testing]
- Kubeadm: Added error printing during retries related to the `WaitForAllControlPlaneComponents` functionality at verbosity level 5. ([#134433](https://github.com/kubernetes/kubernetes/pull/134433), [@neolit123](https://github.com/neolit123))
- Kubeadm: Added the `HTTPEndpoints` field to `ClusterConfiguration.Etcd.ExternalEtcd` to configure HTTP endpoints for etcd communication in v1beta4. This separates HTTP traffic (e.g., `/metrics`, `/health`) from gRPC traffic, improving access control. Mirrors etcd’s `--listen-client-http-urls` behavior; if not set, the `Endpoints` field handles both traffic types. ([#134890](https://github.com/kubernetes/kubernetes/pull/134890), [@SataQiu](https://github.com/SataQiu))
- Kubeadm: Graduated the kubeadm-specific feature gate `ControlPlaneKubeletLocalMode` to GA and locked it to enabled by default. To opt out, patch the `server` field in `/etc/kubernetes/kubelet.conf`. Deprecated the subphase of `kubeadm join phase control-plane-join` called `etcd`, which is now hidden and replaced by subphase with identical functionality `etcd-join`. The `etcd` subphase will be removed in a future release. The subphase `kubelet-wait-bootstrap` of `kubeadm join` is no longer experimental and will now always run. ([#134106](https://github.com/kubernetes/kubernetes/pull/134106), [@neolit123](https://github.com/neolit123))
- Kubernetes is now built using Go 1.25.1 ([#134095](https://github.com/kubernetes/kubernetes/pull/134095), [@dims](https://github.com/dims)) [SIG Release and Testing]
- Kubernetes is now built using Go 1.25.4 ([#135492](https://github.com/kubernetes/kubernetes/pull/135492), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes now uses Go Language Version 1.25, including https://go.dev/blog/container-aware-gomaxprocs ([#134120](https://github.com/kubernetes/kubernetes/pull/134120), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling and Storage]
- Locked down the `AllowOverwriteTerminationGracePeriodSeconds` feature gate. ([#133792](https://github.com/kubernetes/kubernetes/pull/133792), [@HirazawaUi](https://github.com/HirazawaUi))
- Locked the (generally available) feature gate `ExecProbeTimeout` to true. ([#134635](https://github.com/kubernetes/kubernetes/pull/134635), [@vivzbansal](https://github.com/vivzbansal)) [SIG Node and Testing]
- Metrics: Excluded `dryRun` requests from `apiserver_request_sli_duration_seconds`. ([#131092](https://github.com/kubernetes/kubernetes/pull/131092), [@aldudko](https://github.com/aldudko)) [SIG API Machinery and Instrumentation]
- Migrated validation in `resource.k8s.io` to declarative validation.
  When the `DeclarativeValidation` feature gate is enabled, mismatches with existing validation are reported via metrics.
  when `DeclarativeValidationTakeover` feature gate is enabled, declarative validation becomes the primary source of errors for migrated fields. ([#134072](https://github.com/kubernetes/kubernetes/pull/134072), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps and Auth]
- Moved the Pod Certificates feature to beta. Added `UserAnnotations` to the `PodCertificateProjection` API and `UnverifiedUserAnnotations` to the `PodCertificateRequest` API. The `PodCertificateRequest` feature gate remains disabled by default and requires enabling the v1beta1 certificates API groups. ([#134790](https://github.com/kubernetes/kubernetes/pull/134790), [@yt2985](https://github.com/yt2985)) [SIG Auth, Instrumentation and Testing]
- Promoted `ImageGCMaximumAge` to stable. ([#134736](https://github.com/kubernetes/kubernetes/pull/134736), [@haircommander](https://github.com/haircommander)) [SIG Node and Testing]
- Promoted `InPlacePodVerticalScaling` to GA. ([#134949](https://github.com/kubernetes/kubernetes/pull/134949), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Node and Scheduling]
- Promoted `kubectl` command headers to stable. ([#134777](https://github.com/kubernetes/kubernetes/pull/134777), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Promoted the `EnvFiles` feature gate to beta and is enabled by default. Additionally, the syntax specification for environment variables has been restricted to a subset of POSIX shell syntax (all variable values must be wrapped in single quotes). ([#134414](https://github.com/kubernetes/kubernetes/pull/134414), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node and Testing]
- Promoted the `HostnameOverride` feature gate to beta and enabled it by default. ([#134729](https://github.com/kubernetes/kubernetes/pull/134729), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Network and Node]
- Promoted the `KubeletCrashLoopBackOffMax` feature gate to beta and enabled it by default. ([#135044](https://github.com/kubernetes/kubernetes/pull/135044), [@hankfreund](https://github.com/hankfreund))
- Selected a single device class deterministically when multiple device classes were available for an extended resource. ([#135037](https://github.com/kubernetes/kubernetes/pull/135037), [@yliaog](https://github.com/yliaog)) [SIG Node, Scheduling and Testing]
- The JWT authenticator in `kube-apiserver` now reports the following metrics when the `StructuredAuthenticationConfiguration` feature gate is enabled:
  - `apiserver_authentication_jwt_authenticator_jwks_fetch_last_timestamp_seconds`
  - `apiserver_authentication_jwt_authenticator_jwks_fetch_last_key_set_info`. ([#123642](https://github.com/kubernetes/kubernetes/pull/123642), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- The scheduler now clears the `nominatedNodeName` field for Pods upon scheduling or binding failure. External components, such as Cluster Autoscaler and Karpenter, should not overwrite this field. ([#135007](https://github.com/kubernetes/kubernetes/pull/135007), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Scheduling and Testing]
- Updated `applyconfiguration-gen` to generate extract functions for all subresources. ([#132665](https://github.com/kubernetes/kubernetes/pull/132665), [@mrIncompetent](https://github.com/mrIncompetent))
- Updated `applyconfiguration-gen` to preserve struct and field comments from source types in the generated code. ([#132663](https://github.com/kubernetes/kubernetes/pull/132663), [@mrIncompetent](https://github.com/mrIncompetent))
- Updated `kubectl describe pods` to include the involved object’s `fieldPath` (e.g., container name) in event messages, providing better context for debugging multi-container Pods. Note: This changes the previous message format for events that include a `fieldPath`. ([#133627](https://github.com/kubernetes/kubernetes/pull/133627), [@itzPranshul](https://github.com/itzPranshul))
- Updated sandbox ordering to use by attempt count or creation time. ([#130551](https://github.com/kubernetes/kubernetes/pull/130551), [@yylt](https://github.com/yylt))
- Updated the Kubernetes build to use Go `1.25.4`. ([#135187](https://github.com/kubernetes/kubernetes/pull/135187), [@BenTheElder](https://github.com/BenTheElder))
- Updated underlying images and dependencies to be compatible with Go version`1.25.3`. ([#134611](https://github.com/kubernetes/kubernetes/pull/134611), [@cpanato](https://github.com/cpanato)) [SIG Architecture, Cloud Provider, Etcd, Release, Storage and Testing]
- `kubeadm`: Added a preflight check `ContainerRuntimeVersion` to validate if the installed container runtime supports the `RuntimeConfig` gRPC method. If unsupported, `kubeadm` prints a warning message.
  
  Starting with Kubernetes `v1.36`, `kubelet` might refuse to start if the CRI runtime does not support this feature. More information can be found at the [Kubernetes blog](https://kubernetes.io/blog/2025/09/12/kubernetes-v1-34-cri-cgroup-driver-lookup-now-ga/). ([#134906](https://github.com/kubernetes/kubernetes/pull/134906), [@carlory](https://github.com/carlory))

- Kubernetes is now built using Go `1.25.5`. ([#135609](https://github.com/kubernetes/kubernetes/pull/135609), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]

### Documentation

- Promoted the `--chunk-size` flag to stable. The kubectl `describe`, `get`, `drain`, and `events` commands can use `--chunk-size` flag to set chunk size. ([#134481](https://github.com/kubernetes/kubernetes/pull/134481), [@soltysh](https://github.com/soltysh))

### Bug or Regression

- Added support for Pods to reference the same `PersistentVolumeClaim` across multiple volumes. ([#122140](https://github.com/kubernetes/kubernetes/pull/122140), [@huww98](https://github.com/huww98)) [SIG Node, Storage and Testing]
- Added support for the `ShareID` field of the `DRAConsumableCapacity` feature in the Kubelet Plugin API. ([#134520](https://github.com/kubernetes/kubernetes/pull/134520), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node and Testing]
- Added the correct error when eviction is blocked due to the failSafe mechanism of the `DisruptionController`. ([#133097](https://github.com/kubernetes/kubernetes/pull/133097), [@kei01234kei](https://github.com/kei01234kei)) [SIG Apps and Node]
- Changed `kubectl exec` syntax to require `--` before the command. The form `kubectl exec [POD] [COMMAND]` is no longer supported; use `kubectl exec [POD] -- [COMMAND]` instead. ([#133841](https://github.com/kubernetes/kubernetes/pull/133841), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- DRA API: Fixed the `tolerations` field in exact and sub requests to drop properly when the `DRADeviceTaints` API is disabled. ([#132927](https://github.com/kubernetes/kubernetes/pull/132927), [@pohly](https://github.com/pohly))
- DRA Device Taints: Fixed toleration of `NoExecute`. Prior to this enhancement, tolerating a `NoExecute` did not work because the scheduler did not inform the eviction controller about the toleration, so the scheduled pod got evicted almost immediately. ([#134479](https://github.com/kubernetes/kubernetes/pull/134479), [@pohly](https://github.com/pohly)) [SIG Apps, Node, Scheduling and Testing]
- Deprecated metrics will be hidden as per the metrics deprecation policy. https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecating-a-metric . ([#133436](https://github.com/kubernetes/kubernetes/pull/133436), [@richabanker](https://github.com/richabanker)) [SIG Architecture, Instrumentation and Network]
- Disabled the `SchedulerAsyncAPICalls` feature gate to mitigate a bug where its interaction with asynchronous preemption could degrade `kube-scheduler` performance, especially under high `kube-apiserver` load. ([#134400](https://github.com/kubernetes/kubernetes/pull/134400), [@macsko](https://github.com/macsko))
- Dropped `DeviceBindingConditions` fields when the `DRADeviceBindingConditions` feature gate is not enabled and not in use. ([#134964](https://github.com/kubernetes/kubernetes/pull/134964), [@sunya-ch](https://github.com/sunya-ch))
- Extended resources requested by initContainers which are allocated using an automatic ResourceClaim now match the behavior of legacy device plugins, reusing the same resources requested by later sidecar initContainers or regular containers when possible, to minimize the total number of devices requested by the pod. ([#134882](https://github.com/kubernetes/kubernetes/pull/134882), [@yliaog](https://github.com/yliaog)) [SIG Apps, CLI, Node, Scheduling and Testing]
- Fixed SELinux warning controller not emitting events on some SELinux label conflicts. ([#133425](https://github.com/kubernetes/kubernetes/pull/133425), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Storage and Testing]
- Fixed `replicaCount` calculation exceeding max `int32`. ([#126979](https://github.com/kubernetes/kubernetes/pull/126979), [@omerap12](https://github.com/omerap12)) [SIG Apps and Autoscaling]
- Fixed a Windows kube-proxy (winkernel) issue where stale `RemoteEndpoints`
  remained when a Deployment was referenced by multiple Services due to premature
  clearing of the `terminatedEndpoints` map. ([#135146](https://github.com/kubernetes/kubernetes/pull/135146), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fixed a bug in `ValidatingAdmissionPolicy` where schemas with `additionalProperties: true` could cause the kube-controller-manager to crash with a nil pointer exception. ([#135155](https://github.com/kubernetes/kubernetes/pull/135155), [@jpbetz](https://github.com/jpbetz))
- Fixed a bug in `kube-proxy` `nftables` mode (GA as of `v1.33`) which fails to determine if traffic originates from a local source on the node. The issue was caused by using the wrong meta `iif` instead of `iifname` for name based matches. ([#134024](https://github.com/kubernetes/kubernetes/pull/134024), [@jack4it](https://github.com/jack4it))
- Fixed a bug in `kube-scheduler` where pending pod preemption caused preemptor pods to be retried more frequently. ([#134245](https://github.com/kubernetes/kubernetes/pull/134245), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- Fixed a bug that caused apiservers to send an inappropriate Content-Type request header to authorization, token authentication, imagepolicy admission, and audit webhooks when the alpha client-go feature gate "ClientsPreferCBOR" is enabled. ([#132960](https://github.com/kubernetes/kubernetes/pull/132960), [@benluddy](https://github.com/benluddy)) [SIG API Machinery and Node]
- Fixed a bug that caused duplicate validation when updating `PersistentVolumeClaims`, `VolumeAttachments` and `VolumeAttributesClasses`. ([#132549](https://github.com/kubernetes/kubernetes/pull/132549), [@gavinkflam](https://github.com/gavinkflam))
- Fixed a bug that caused duplicate validation when updating `Role` and `RoleBinding` resources. ([#132550](https://github.com/kubernetes/kubernetes/pull/132550), [@gavinkflam](https://github.com/gavinkflam))
- Fixed a bug that prevented allocating the same device that was previously consuming the `CounterSet` when both `DRAConsumableCapacity` and `DRAPartitionableDevices` were enabled. ([#134103](https://github.com/kubernetes/kubernetes/pull/134103), [@sunya-ch](https://github.com/sunya-ch))
- Fixed a bug that prevents scheduling the next pod when using the `DRAConsumableCapacity` feature. ([#133706](https://github.com/kubernetes/kubernetes/pull/133706), [@sunya-ch](https://github.com/sunya-ch))
- Fixed a bug to prevent segmentation fault from occurring when updating deeply nested JSON fields. ([#134381](https://github.com/kubernetes/kubernetes/pull/134381), [@kon-angelo](https://github.com/kon-angelo)) [SIG API Machinery and CLI]
- Fixed a bug where 64-bit IPv6 `ServiceCIDRs` allocated addresses outside the subnet range. ([#134193](https://github.com/kubernetes/kubernetes/pull/134193), [@hoskeri](https://github.com/hoskeri))
- Fixed a bug where Job status updates fail after resuming a Job that was previously started and suspended.
  The error message was: `status.startTime: Required value: startTime cannot be removed for unsuspended job`. ([#134769](https://github.com/kubernetes/kubernetes/pull/134769), [@dejanzele](https://github.com/dejanzele)) [SIG Apps and Testing]
- Fixed a bug where `AllocationMode: All` would not succeed if a resource pool contained `ResourceSlices` that were not targeting the current node. ([#134466](https://github.com/kubernetes/kubernetes/pull/134466), [@mortent](https://github.com/mortent))
- Fixed a bug where a deleted Pod in the binding phase continued to occupy space on the node in `kube-scheduler`. ([#134157](https://github.com/kubernetes/kubernetes/pull/134157), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- Fixed a bug where high latency `kube-apiserver` caused scheduling throughput degradation. ([#134154](https://github.com/kubernetes/kubernetes/pull/134154), [@macsko](https://github.com/macsko))
- Fixed a bug where the health of a DRA resource was not reported in the Pod status if the resource claim was generated from a template or used a different local name in the Pod spec. ([#134875](https://github.com/kubernetes/kubernetes/pull/134875), [@Jpsassine](https://github.com/Jpsassine)) [SIG Node and Testing]
- Fixed a long-standing issue where `kubelet` rejected Pods with `NodeAffinityFailed` due to a stale informer cache. ([#134445](https://github.com/kubernetes/kubernetes/pull/134445), [@natasha41575](https://github.com/natasha41575))
- Fixed a panic in `kubectl api-resources` that occurred when the Discovery Client failed. ([#134833](https://github.com/kubernetes/kubernetes/pull/134833), [@rikatz](https://github.com/rikatz))
- Fixed a possible data race during metrics registration. ([#134390](https://github.com/kubernetes/kubernetes/pull/134390), [@liggitt](https://github.com/liggitt)) [SIG Architecture and Instrumentation]
- Fixed a spurious `namespace not found` error in default `v1.30+` configurations when using `ValidatingAdmissionPolicy` or `MutatingAdmissionPolicy` to intercept namespaced objects in newly-created namespaces. ([#135359](https://github.com/kubernetes/kubernetes/pull/135359), [@liggitt](https://github.com/liggitt))
- Fixed a startup probe race condition that caused main containers to remain stuck in "Initializing" state when sidecar containers with startup probes had failed initially but succeeded on restart in pods with `restartPolicy=Never`. ([#133072](https://github.com/kubernetes/kubernetes/pull/133072), [@AadiDev005](https://github.com/AadiDev005)) [SIG Node and Testing]
- Fixed an issue in asynchronous preemption: Scheduler now checks if preemption is ongoing for a Pod before initiating new preemption calls. ([#134730](https://github.com/kubernetes/kubernetes/pull/134730), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Scheduling and Testing]
- Fixed an issue that prevented restart policies and restart rules from being applied to static pods. ([#135031](https://github.com/kubernetes/kubernetes/pull/135031), [@yuanwang04](https://github.com/yuanwang04))
- Fixed an issue where requests for a config `FromClass` in the `ResourceClaim` status were not referenced. ([#134793](https://github.com/kubernetes/kubernetes/pull/134793), [@LionelJouin](https://github.com/LionelJouin))
- Fixed an issue where the `kubelet` `/configz` endpoint reported an incorrect value for `kubeletconfig.cgroupDriver` when the cgroup driver setting was received from the container runtime. ([#134743](https://github.com/kubernetes/kubernetes/pull/134743), [@marquiz](https://github.com/marquiz))
- Fixed an issue where the default `serviceCIDR` controller did not log events because the event broadcaster was shutdown during initialization. ([#133338](https://github.com/kubernetes/kubernetes/pull/133338), [@aojea](https://github.com/aojea))
- Fixed an issue with setting `distinctAttribute=nil` when the `DRAConsumableCapacity` feature gate is disabled. ([#134962](https://github.com/kubernetes/kubernetes/pull/134962), [@sunya-ch](https://github.com/sunya-ch))
- Fixed broken shell completion for API resources. ([#133771](https://github.com/kubernetes/kubernetes/pull/133771), [@marckhouzam](https://github.com/marckhouzam))
- Fixed incorrect behavior of preemptor pod when preemption of the victim takes long to complete. The preemptor pod should not be circling in scheduling cycles until preemption is finished. ([#134294](https://github.com/kubernetes/kubernetes/pull/134294), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Scheduling and Testing]
- Fixed missing `kubelet_volume_stats_*` metrics. ([#133890](https://github.com/kubernetes/kubernetes/pull/133890), [@huww98](https://github.com/huww98)) [SIG Instrumentation and Node]
- Fixed occasional schedule delays when a static `PersistentVolume` is created. ([#133929](https://github.com/kubernetes/kubernetes/pull/133929), [@huww98](https://github.com/huww98)) [SIG Scheduling and Storage]
- Fixed resource claims deallocation for extended resource when Pod completes. ([#134312](https://github.com/kubernetes/kubernetes/pull/134312), [@alaypatel07](https://github.com/alaypatel07)) [SIG Apps, Node and Testing]
- Fixed the kubelet to honor the `userNamespaces.idsPerPod` configuration, which was previously ignored. ([#133373](https://github.com/kubernetes/kubernetes/pull/133373), [@AkihiroSuda](https://github.com/AkihiroSuda)) [SIG Node and Testing]
- Fixed the replacement tag in APIs so it no longer acted as a selector for storage version. ([#135197](https://github.com/kubernetes/kubernetes/pull/135197), [@Jefftree](https://github.com/Jefftree))
- Fixed validation error when `ConfigFlags` includes `CertFile` and/or `KeyFile` while the original configuration also contains `CertFileData` and/or `KeyFileData`. ([#133917](https://github.com/kubernetes/kubernetes/pull/133917), [@n2h9](https://github.com/n2h9)) [SIG API Machinery and CLI]
- Improved performance of `Endpoint` and `EndpointSlice` controllers when there are a large number of services in a single namespace by making pod-to-service lookup asynchronous. ([#134739](https://github.com/kubernetes/kubernetes/pull/134739), [@shyamjvs](https://github.com/shyamjvs)) [SIG Apps and Network]
- Improved the `FreeDiskSpaceFailed` warning event to provide more actionable details when image garbage collection fails to free enough disk space. Example: `Insufficient free disk space on the node's image filesystem (95.0% of 10.0 GiB used). Failed to free sufficient space by deleting unused images. Consider resizing the disk or deleting unused files.`. ([#132578](https://github.com/kubernetes/kubernetes/pull/132578), [@drigz](https://github.com/drigz))
- Introduced support for using an implicit extended resource name derived from the device class (`deviceclass.resource.kubernetes.io/<device-class-name>`) to request DRA devices matching that class. ([#133363](https://github.com/kubernetes/kubernetes/pull/133363), [@yliaog](https://github.com/yliaog)) [SIG Node, Scheduling and Testing]
- Kube-apiserver: Fixed a `v1.34` regression with spurious "Error getting keys" log messages. ([#133817](https://github.com/kubernetes/kubernetes/pull/133817), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- Kube-apiserver: Fixed a possible `v1.34` performance regression calculating object size statistics for resources not served from the watch cache, typically only `Events`. ([#133873](https://github.com/kubernetes/kubernetes/pull/133873), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- Kube-apiserver: Improved validation error messages for custom resources with CEL validation rules to include the value that failed validation. ([#132798](https://github.com/kubernetes/kubernetes/pull/132798), [@cbandy](https://github.com/cbandy))
- Kube-apiserver: Made sure that when `--requestheader-client-ca-file` and `--client-ca-file` contain overlapping certificates, `--requestheader-allowed-names` must be specified so that regular client certificates cannot set authenticating proxy headers for arbitrary users. ([#131411](https://github.com/kubernetes/kubernetes/pull/131411), [@ballista01](https://github.com/ballista01)) [SIG API Machinery, Auth and Security]
- Kube-apiserver: Resolved an issue causing unnecessary warning log messages about enabled alpha APIs during API server startup. ([#135327](https://github.com/kubernetes/kubernetes/pull/135327), [@michaelasp](https://github.com/michaelasp))
- Kube-controller-manager: Fixed a possible data race in the garbage collection controller. ([#134379](https://github.com/kubernetes/kubernetes/pull/134379), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Apps]
- Kube-controller-manager: Resolved potential issues handling pods with incorrect uids in their `ownerReference`. ([#134654](https://github.com/kubernetes/kubernetes/pull/134654), [@liggitt](https://github.com/liggitt))
- Kubeadm: Added missing cluster-info context validation to prevent panics when the user has a malformed kubeconfig in the cluster-info ConfigMap that excludes a valid current context. ([#134715](https://github.com/kubernetes/kubernetes/pull/134715), [@neolit123](https://github.com/neolit123))
- Kubeadm: Ensured waiting for `apiserver` uses a local client that doesn't reach to the control plane endpoint and instead reaches directly to the local API server endpoint. ([#134265](https://github.com/kubernetes/kubernetes/pull/134265), [@neolit123](https://github.com/neolit123))
- Kubeadm: Fixed `KUBEADM_UPGRADE_DRYRUN_DIR` not honored in upgrade phase when writing kubelet config files. ([#134007](https://github.com/kubernetes/kubernetes/pull/134007), [@carlory](https://github.com/carlory))
- Kubeadm: Fixed a bug where `ClusterConfiguration.APIServer.TimeoutForControlPlane` from `v1beta3` was not respected in newer kubeadm versions where `v1beta4` is the default. ([#133513](https://github.com/kubernetes/kubernetes/pull/133513), [@tom1299](https://github.com/tom1299))
- Kubeadm: Fixed a bug where the node registration information for a given node was not fetched correctly during `kubeadm upgrade node` and the node name can end up being incorrect in cases where the node name is not the same as the host name. ([#134319](https://github.com/kubernetes/kubernetes/pull/134319), [@neolit123](https://github.com/neolit123))
- Kubeadm: Fixed a preflight check that could fail hostname construction in IPv6 setups. ([#134588](https://github.com/kubernetes/kubernetes/pull/134588), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, Cloud Provider, Cluster Lifecycle and Testing]
- Kubelet: Fixed a concurrent map write error when creating a pod with an empty volume while the `LocalStorageCapacityIsolationFSQuotaMonitoring` feature gate is enabled. ([#135174](https://github.com/kubernetes/kubernetes/pull/135174), [@carlory](https://github.com/carlory))
- Kubelet: Fixed an internal deadlock that caused the connection to a DRA driver to become unusable after being idle for 30 minutes. ([#133926](https://github.com/kubernetes/kubernetes/pull/133926), [@pohly](https://github.com/pohly))
- Made legacy watch calls (`ResourceVersion` = 0 or unset) that generate init-events weigh higher in `API Priority and Fairness (APF)` seat usage. Properly accounting for their cost protects the API server from CPU overload. Users might see increased throttling of such calls as a result. ([#134601](https://github.com/kubernetes/kubernetes/pull/134601), [@shyamjvs](https://github.com/shyamjvs))
- Namespace is now included in the `--dry-run=client` output for `HorizontalPodAutoscaler (HPA)` objects. ([#134263](https://github.com/kubernetes/kubernetes/pull/134263), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Populated `involvedObject.apiVersion` on Events created for Nodes and Pods. ([#134545](https://github.com/kubernetes/kubernetes/pull/134545), [@novahe](https://github.com/novahe)) [SIG Cloud Provider, Network, Node, Scalability and Testing]
- Promoted VAC API test to conformance. ([#133615](https://github.com/kubernetes/kubernetes/pull/133615), [@carlory](https://github.com/carlory)) [SIG Architecture, Storage and Testing]
- Removed `BlockOwnerDeletion` from `ResourceClaim` created from `ResourceClaimTemplate` and from `extendedResourceClaim` created by the `scheduler`. ([#134956](https://github.com/kubernetes/kubernetes/pull/134956), [@yliaog](https://github.com/yliaog)) [SIG Apps, Node and Scheduling]
- Removed an incorrect `SessionAffinity` warning that appeared when a headless service was created or updated. ([#134054](https://github.com/kubernetes/kubernetes/pull/134054), [@Peac36](https://github.com/Peac36))
- Slow container runtime initialization no longer causes the System WatchDog to kill the kubelet. The Device Manager was treated as unhealthy until it began listening on its port. ([#135153](https://github.com/kubernetes/kubernetes/pull/135153), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- Typed workqueue now cleans up goroutines before shutting down ([#135072](https://github.com/kubernetes/kubernetes/pull/135072), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Updated `kubectl scale` to return a consistent error message when a specified resource is not found. Previously, it returned: `error: no objects passed to scale <GroupResource> "<ResourceName>" not found`. It now matches the format used by other commands (e.g., `kubectl get`): `Error from server (NotFound): <GroupResource> "<ResourceName>" not found`. ([#134017](https://github.com/kubernetes/kubernetes/pull/134017), [@mochizuki875](https://github.com/mochizuki875))
- `kube-controller-manager`: Fixed a `v1.34` regression that triggered a spurious rollout of existing StatefulSets when upgrading the control plane from `v1.33` to `v1.34`. This fix is guarded by the `StatefulSetSemanticRevisionComparison` feature gate, which is enabled by default. ([#135017](https://github.com/kubernetes/kubernetes/pull/135017), [@liggitt](https://github.com/liggitt))
- `kube-scheduler`: Pod statuses no longer include specific taint keys or values when scheduling fails due to untolerated taints. ([#134740](https://github.com/kubernetes/kubernetes/pull/134740), [@hoskeri](https://github.com/hoskeri))
- Fixes a bug where `MutatingAdmissionPolicy` would fail to apply to objects with duplicate list items (like env vars). ([#135560](https://github.com/kubernetes/kubernetes/pull/135560), [@lalitc375](https://github.com/lalitc375) [SIG API Machinery]
- K8s.io/client-go: Fixes a regression in 1.34+ which prevented informers from using configured Transformer functions. ([#135580](https://github.com/kubernetes/kubernetes/pull/135580), [@serathius](https://github.com/serathius) [SIG API Machinery]

### Other (Cleanup or Flake)

- Added the `Step` field to the testing framework to allow volume expansion in configurable step sizes for tests. ([#134760](https://github.com/kubernetes/kubernetes/pull/134760), [@Rishita-Golla](https://github.com/Rishita-Golla)) [SIG Storage and Testing]
- Bumped addon manager to use `kubectl` version `v1.32.2`. ([#130548](https://github.com/kubernetes/kubernetes/pull/130548), [@Jefftree](https://github.com/Jefftree)) [SIG Cloud Provider, Scalability and Testing]
- Dropped support for `certificates/v1beta1` `CertificateSigningRequest` in `kubectl`. ([#134782](https://github.com/kubernetes/kubernetes/pull/134782), [@scaliby](https://github.com/scaliby))
- Dropped support for `discovery/v1beta1` `EndpointSlice` in `kubectl`. ([#134913](https://github.com/kubernetes/kubernetes/pull/134913), [@scaliby](https://github.com/scaliby))
- Dropped support for `networking/v1beta1` `Ingress` in `kubectl`. ([#135108](https://github.com/kubernetes/kubernetes/pull/135108), [@scaliby](https://github.com/scaliby))
- Dropped support for `networking/v1beta1` `Ingress` in `kubectl`. ([#135176](https://github.com/kubernetes/kubernetes/pull/135176), [@scaliby](https://github.com/scaliby))
- Dropped support for `policy/v1beta1` PodDisruptionBudget in kubectl. ([#134685](https://github.com/kubernetes/kubernetes/pull/134685), [@scaliby](https://github.com/scaliby))
- Eliminated and prevented future use of the `md5` algorithm in favor of more appropriate hashing algorithms. ([#133511](https://github.com/kubernetes/kubernetes/pull/133511), [@BenTheElder](https://github.com/BenTheElder)) [SIG Apps, Architecture, CLI, Cluster Lifecycle, Network, Node, Security, Storage and Testing]
- Fixed `nfacct` test cases on s390x. ([#133603](https://github.com/kubernetes/kubernetes/pull/133603), [@saisindhuri91](https://github.com/saisindhuri91))
- Fixed formatting of various Go API deprecations for `GoDoc` and `pkgsite`, and enabled a linter to detect misformatted deprecations. ([#133571](https://github.com/kubernetes/kubernetes/pull/133571), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, Architecture, CLI, Instrumentation and Testing]
- Improved HPA performance when using container-specific resource metrics by optimizing container lookup logic to exit early once the target container is found, reducing unnecessary iterations through all containers in a pod. ([#133415](https://github.com/kubernetes/kubernetes/pull/133415), [@AadiDev005](https://github.com/AadiDev005)) [SIG Apps and Autoscaling]
- Increased the coverage to 89.8%. ([#132607](https://github.com/kubernetes/kubernetes/pull/132607), [@ylink-lfs](https://github.com/ylink-lfs))
- Kube-apiserver: Fixed an issue where passing invalid `DeleteOptions` incorrectly returned a 500 status instead of 400. ([#133358](https://github.com/kubernetes/kubernetes/pull/133358), [@ostrain](https://github.com/ostrain))
- Kubeadm: Updated the supported `etcd` version to `v3.5.23` for supported control plane versions `v1.31`, `v1.32`, and `v1.33`. ([#134692](https://github.com/kubernetes/kubernetes/pull/134692), [@joshjms](https://github.com/joshjms)) [SIG Cluster Lifecycle and Etcd]
- Kubeadm: stopped applying the `--pod-infra-container-image` flag for the kubelet. The flag has been deprecated and no longer served a purpose in the kubelet as the logic was migrated to CRI (Container Runtime Interface). During upgrade, kubeadm will attempt to remove the flag from the file `/var/lib/kubelet/kubeadm-flags.env`. ([#133778](https://github.com/kubernetes/kubernetes/pull/133778), [@carlory](https://github.com/carlory)) [SIG Cloud Provider and Cluster Lifecycle]
- Migrated the `CPUManager` to contextual logging. ([#125912](https://github.com/kubernetes/kubernetes/pull/125912), [@ffromani](https://github.com/ffromani))
- Moved Types in `k/k/pkg/scheduler/framework`:
  `Handle`,
  `Plugin`,
  `PreEnqueuePlugin`, `QueueSortPlugin`, `EnqueueExtensions`, `PreFilterExtensions`, `PreFilterPlugin`, `FilterPlugin`, `PostFilterPlugin`, `PreScorePlugin`, `ScorePlugin`, `ReservePlugin`, `PreBindPlugin`, `PostBindPlugin`, `PermitPlugin`, `BindPlugin`,
  `PodActivator`, `PodNominator`, `PluginsRunner`,
  `LessFunc`, `ScoreExtensions`, `NodeToStatusReader`, `NodeScoreList`, `NodeScore`, `NodePluginScores`, `PluginScore`, `NominatingMode`, `NominatingInfo`, `WaitingPod`, `PreFilterResult`, `PostFilterResult`,
  `Extender`,
  `NodeInfoLister`, `StorageInfoLister`, `SharedLister`, `ResourceSliceLister`, `DeviceClassLister`, `ResourceClaimTracker`, `SharedDRAManager`
  
  to package `k8s.io/kube-scheduler/framework`. Users should update import paths. The interfaces don't change.
  
  Type `Parallelizer` in `k/k/pkg/scheduler/framework/parallelism` has been split into interface `Parallelizer` (in `k8s.io/kube-scheduler/framework`) and `struct Parallelizer` (location unchanged in k/k). Plugin developers should update the import path to staging repo. ([#133172](https://github.com/kubernetes/kubernetes/pull/133172), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Node, Release, Scheduling, Storage and Testing]
- Moved the CPU Manager static policy option `strict-cpu-reservation` to the GA version. ([#134388](https://github.com/kubernetes/kubernetes/pull/134388), [@psasnal](https://github.com/psasnal))
- Promoted the Topology Manager policy option `max-allowable-numa-nodes` to GA version. ([#134614](https://github.com/kubernetes/kubernetes/pull/134614), [@ffromani](https://github.com/ffromani))
- Reduced event spam during volume operation errors in the Portworx in-tree driver. ([#135081](https://github.com/kubernetes/kubernetes/pull/135081), [@gohilankit](https://github.com/gohilankit))
- Removed `rsync` as a dependency to build Kubernetes. ([#134656](https://github.com/kubernetes/kubernetes/pull/134656), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release and Testing]
- Removed container name from messages for container created and started events. ([#134043](https://github.com/kubernetes/kubernetes/pull/134043), [@HirazawaUi](https://github.com/HirazawaUi))
- Removed deprecated gogo protocol definitions from `k8s.io/kubelet/pkg/apis/dra` in favor of `google.golang.org/protobuf`. ([#133026](https://github.com/kubernetes/kubernetes/pull/133026), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery and Node]
- Removed general available feature-gate `SizeMemoryBackedVolumes`. ([#133720](https://github.com/kubernetes/kubernetes/pull/133720), [@carlory](https://github.com/carlory)) [SIG Node, Storage and Testing]
- Removed the `ComponentSLIs` feature gate, as it was promoted to stable in the Kubernetes `v1.32` release. ([#133742](https://github.com/kubernetes/kubernetes/pull/133742), [@carlory](https://github.com/carlory)) [SIG Architecture and Instrumentation]
- Removed the `KUBECTL_OPENAPIV3_PATCH` environment variable, as aggregated discovery has been stable since `v1.30`. ([#134130](https://github.com/kubernetes/kubernetes/pull/134130), [@ardaguclu](https://github.com/ardaguclu))
- Removed the `UserNamespacesPodSecurityStandards` feature gate. The minimum supported Kubernetes version for `kubelet` is now `v1.31`, so the gate is no longer needed. ([#132157](https://github.com/kubernetes/kubernetes/pull/132157), [@haircommander](https://github.com/haircommander)) [SIG Auth, Node and Testing]
- Removed the `VolumeAttributesClass` resource from the `storage.k8s.io/v1alpha1` API in `v1.35`. ([#134625](https://github.com/kubernetes/kubernetes/pull/134625), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Etcd, Storage and Testing]
- Specified the deprecated version of `apiserver_storage_objects` metric in metrics docs. ([#134028](https://github.com/kubernetes/kubernetes/pull/134028), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Etcd and Instrumentation]
- Substantially simplified building Kubernetes by making the process run a pre-built container image directly without running `rsyncd`. ([#134510](https://github.com/kubernetes/kubernetes/pull/134510), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release and Testing]
- Tests: Switched to https://go.dev/doc/go1.25#container-aware-gomaxprocs from `go.uber.org/automaxprocs`. ([#133492](https://github.com/kubernetes/kubernetes/pull/133492), [@BenTheElder](https://github.com/BenTheElder))
- The `AggregatedDiscoveryRemoveBetaType` feature gate was deprecated and locked to `true`. ([#134230](https://github.com/kubernetes/kubernetes/pull/134230), [@Jefftree](https://github.com/Jefftree))
- The `SystemdWatchdog` feature gate has been locked to default and will be removed in future release. The systemd watchdog functionality in `kubelet` can be enabled via systemd without any feature gate configuration. See the [systemd watchdog documentation](https://kubernetes.io/docs/reference/node/systemd-watchdog/) for more information. ([#134691](https://github.com/kubernetes/kubernetes/pull/134691), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- Updated CNI plugins to v1.8.0. ([#133837](https://github.com/kubernetes/kubernetes/pull/133837), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated `etcd` to `v3.6.5`. ([#134251](https://github.com/kubernetes/kubernetes/pull/134251), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Updated `kubectl auth reconcile` to retry reconciliation when a conflict error occurs. ([#133323](https://github.com/kubernetes/kubernetes/pull/133323), [@liggitt](https://github.com/liggitt)) [SIG Auth and CLI]
- Updated `kubectl get` and `kubectl describe` human-readable output to no longer show counts for referenced tokens and secrets. ([#117160](https://github.com/kubernetes/kubernetes/pull/117160), [@liggitt](https://github.com/liggitt)) [SIG CLI and Testing]
- Updated cri-tools to v1.34.0. ([#133636](https://github.com/kubernetes/kubernetes/pull/133636), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Updated the Go version of Kubernetes to `1.25.3`. ([#134598](https://github.com/kubernetes/kubernetes/pull/134598), [@BenTheElder](https://github.com/BenTheElder))
- Updated the `/statusz` page for `kube-proxy` to include a list of exposed endpoints, making debugging and introspection easier. ([#133190](https://github.com/kubernetes/kubernetes/pull/133190), [@aman4433](https://github.com/aman4433)) [SIG Network and Node]
- Updated the `kubectl wait` command description by removing the `Experimental` prefix, as the command has been stable for a long time. ([#133731](https://github.com/kubernetes/kubernetes/pull/133731), [@ardaguclu](https://github.com/ardaguclu))
- Updated the etcd client library to `v3.6.5`. ([#134780](https://github.com/kubernetes/kubernetes/pull/134780), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- Updated the short description of the `kubectl wait` command by removing the `Experimental` prefix, as the command has been stable for a long time. ([#133907](https://github.com/kubernetes/kubernetes/pull/133907), [@ardaguclu](https://github.com/ardaguclu))
- Upgraded CoreDNS to v1.12.4. ([#133968](https://github.com/kubernetes/kubernetes/pull/133968), [@yashsingh74](https://github.com/yashsingh74)) [SIG Cloud Provider and Cluster Lifecycle]
- Upgraded `CoreDNS` to `v1.12.3`. ([#132288](https://github.com/kubernetes/kubernetes/pull/132288), [@thevilledev](https://github.com/thevilledev)) [SIG Cloud Provider and Cluster Lifecycle]
- `kubeadm`: Removed the `WaitForAllControlPlaneComponents` feature gate, which graduated to GA in `v1.34` and was locked to enabled by default. ([#134781](https://github.com/kubernetes/kubernetes/pull/134781), [@neolit123](https://github.com/neolit123))
- `kubeadm`: Updated the supported etcd version to `v3.5.24` for control plane versions `v1.32`, `v1.33`, and `v1.34`. ([#134779](https://github.com/kubernetes/kubernetes/pull/134779), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- `etcd: Update etcd to `v3.6.6`. (#135271, @bzsuni) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Fix a bug in the kube-apiserver where a malformed Service without name can cause high CPU usage. The bug is present on the new Cluster IP allocators enabled with the feature MultiCIDRServiceAllocator (enabled by default since 1.33)


## Dependencies

### Added
- cyphar.com/go-pathrs: v0.2.1
- github.com/Masterminds/semver/v3: [v3.4.0](https://github.com/Masterminds/semver/tree/v3.4.0)
- github.com/gkampitakis/ciinfo: [v0.3.2](https://github.com/gkampitakis/ciinfo/tree/v0.3.2)
- github.com/gkampitakis/go-diff: [v1.3.2](https://github.com/gkampitakis/go-diff/tree/v1.3.2)
- github.com/gkampitakis/go-snaps: [v0.5.15](https://github.com/gkampitakis/go-snaps/tree/v0.5.15)
- github.com/goccy/go-yaml: [v1.18.0](https://github.com/goccy/go-yaml/tree/v1.18.0)
- github.com/joshdk/go-junit: [v1.0.0](https://github.com/joshdk/go-junit/tree/v1.0.0)
- github.com/maruel/natural: [v1.1.1](https://github.com/maruel/natural/tree/v1.1.1)
- github.com/mfridman/tparse: [v0.18.0](https://github.com/mfridman/tparse/tree/v0.18.0)
- github.com/moby/sys/atomicwriter: [v0.1.0](https://github.com/moby/sys/tree/atomicwriter/v0.1.0)
- github.com/tidwall/gjson: [v1.18.0](https://github.com/tidwall/gjson/tree/v1.18.0)
- github.com/tidwall/match: [v1.1.1](https://github.com/tidwall/match/tree/v1.1.1)
- github.com/tidwall/pretty: [v1.2.1](https://github.com/tidwall/pretty/tree/v1.2.1)
- github.com/tidwall/sjson: [v1.2.5](https://github.com/tidwall/sjson/tree/v1.2.5)
- go.uber.org/automaxprocs: v1.6.0
- golang.org/x/tools/go/expect: v0.1.1-deprecated
- golang.org/x/tools/go/packages/packagestest: v0.1.1-deprecated

### Changed
- cloud.google.com/go/compute/metadata: v0.6.0 → v0.7.0
- github.com/aws/aws-sdk-go-v2/config: [v1.27.24 → v1.29.14](https://github.com/aws/aws-sdk-go-v2/compare/config/v1.27.24...config/v1.29.14)
- github.com/aws/aws-sdk-go-v2/credentials: [v1.17.24 → v1.17.67](https://github.com/aws/aws-sdk-go-v2/compare/credentials/v1.17.24...credentials/v1.17.67)
- github.com/aws/aws-sdk-go-v2/feature/ec2/imds: [v1.16.9 → v1.16.30](https://github.com/aws/aws-sdk-go-v2/compare/feature/ec2/imds/v1.16.9...feature/ec2/imds/v1.16.30)
- github.com/aws/aws-sdk-go-v2/internal/configsources: [v1.3.13 → v1.3.34](https://github.com/aws/aws-sdk-go-v2/compare/internal/configsources/v1.3.13...internal/configsources/v1.3.34)
- github.com/aws/aws-sdk-go-v2/internal/endpoints/v2: [v2.6.13 → v2.6.34](https://github.com/aws/aws-sdk-go-v2/compare/internal/endpoints/v2/v2.6.13...internal/endpoints/v2/v2.6.34)
- github.com/aws/aws-sdk-go-v2/internal/ini: [v1.8.0 → v1.8.3](https://github.com/aws/aws-sdk-go-v2/compare/internal/ini/v1.8.0...internal/ini/v1.8.3)
- github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding: [v1.11.3 → v1.12.3](https://github.com/aws/aws-sdk-go-v2/compare/service/internal/accept-encoding/v1.11.3...service/internal/accept-encoding/v1.12.3)
- github.com/aws/aws-sdk-go-v2/service/internal/presigned-url: [v1.11.15 → v1.12.15](https://github.com/aws/aws-sdk-go-v2/compare/service/internal/presigned-url/v1.11.15...service/internal/presigned-url/v1.12.15)
- github.com/aws/aws-sdk-go-v2/service/sso: [v1.22.1 → v1.25.3](https://github.com/aws/aws-sdk-go-v2/compare/service/sso/v1.22.1...service/sso/v1.25.3)
- github.com/aws/aws-sdk-go-v2/service/ssooidc: [v1.26.2 → v1.30.1](https://github.com/aws/aws-sdk-go-v2/compare/service/ssooidc/v1.26.2...service/ssooidc/v1.30.1)
- github.com/aws/aws-sdk-go-v2/service/sts: [v1.30.1 → v1.33.19](https://github.com/aws/aws-sdk-go-v2/compare/service/sts/v1.30.1...service/sts/v1.33.19)
- github.com/aws/aws-sdk-go-v2: [v1.30.1 → v1.36.3](https://github.com/aws/aws-sdk-go-v2/compare/v1.30.1...v1.36.3)
- github.com/aws/smithy-go: [v1.20.3 → v1.22.3](https://github.com/aws/smithy-go/compare/v1.20.3...v1.22.3)
- github.com/containerd/containerd/api: [v1.8.0 → v1.9.0](https://github.com/containerd/containerd/compare/api/v1.8.0...api/v1.9.0)
- github.com/containerd/ttrpc: [v1.2.6 → v1.2.7](https://github.com/containerd/ttrpc/compare/v1.2.6...v1.2.7)
- github.com/containerd/typeurl/v2: [v2.2.2 → v2.2.3](https://github.com/containerd/typeurl/compare/v2.2.2...v2.2.3)
- github.com/coredns/corefile-migration: [v1.0.26 → v1.0.29](https://github.com/coredns/corefile-migration/compare/v1.0.26...v1.0.29)
- github.com/cyphar/filepath-securejoin: [v0.4.1 → v0.6.0](https://github.com/cyphar/filepath-securejoin/compare/v0.4.1...v0.6.0)
- github.com/docker/docker: [v26.1.4+incompatible → v28.2.2+incompatible](https://github.com/docker/docker/compare/v26.1.4...v28.2.2)
- github.com/go-logr/logr: [v1.4.2 → v1.4.3](https://github.com/go-logr/logr/compare/v1.4.2...v1.4.3)
- github.com/google/cadvisor: [v0.52.1 → v0.53.0](https://github.com/google/cadvisor/compare/v0.52.1...v0.53.0)
- github.com/google/pprof: [d1b30fe → 27863c8](https://github.com/google/pprof/compare/d1b30fe...27863c8)
- github.com/onsi/ginkgo/v2: [v2.21.0 → v2.27.2](https://github.com/onsi/ginkgo/compare/v2.21.0...v2.27.2)
- github.com/onsi/gomega: [v1.35.1 → v1.38.2](https://github.com/onsi/gomega/compare/v1.35.1...v1.38.2)
- github.com/opencontainers/cgroups: [v0.0.1 → v0.0.3](https://github.com/opencontainers/cgroups/compare/v0.0.1...v0.0.3)
- github.com/opencontainers/runc: [v1.2.5 → v1.3.0](https://github.com/opencontainers/runc/compare/v1.2.5...v1.3.0)
- github.com/opencontainers/runtime-spec: [v1.2.0 → v1.2.1](https://github.com/opencontainers/runtime-spec/compare/v1.2.0...v1.2.1)
- github.com/opencontainers/selinux: [v1.11.1 → v1.13.0](https://github.com/opencontainers/selinux/compare/v1.11.1...v1.13.0)
- github.com/prometheus/client_golang: [v1.22.0 → v1.23.2](https://github.com/prometheus/client_golang/compare/v1.22.0...v1.23.2)
- github.com/prometheus/client_model: [v0.6.1 → v0.6.2](https://github.com/prometheus/client_model/compare/v0.6.1...v0.6.2)
- github.com/prometheus/common: [v0.62.0 → v0.66.1](https://github.com/prometheus/common/compare/v0.62.0...v0.66.1)
- github.com/prometheus/procfs: [v0.15.1 → v0.16.1](https://github.com/prometheus/procfs/compare/v0.15.1...v0.16.1)
- github.com/rogpeppe/go-internal: [v1.13.1 → v1.14.1](https://github.com/rogpeppe/go-internal/compare/v1.13.1...v1.14.1)
- github.com/spf13/cobra: [v1.9.1 → v1.10.0](https://github.com/spf13/cobra/compare/v1.9.1...v1.10.0)
- github.com/spf13/pflag: [v1.0.6 → v1.0.9](https://github.com/spf13/pflag/compare/v1.0.6...v1.0.9)
- github.com/stretchr/testify: [v1.10.0 → v1.11.1](https://github.com/stretchr/testify/compare/v1.10.0...v1.11.1)
- go.etcd.io/bbolt: v1.4.2 → v1.4.3
- go.etcd.io/etcd/api/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/client/pkg/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/client/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/pkg/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/server/v3: v3.6.4 → v3.6.5
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.58.0 → v0.61.0
- go.opentelemetry.io/otel/metric: v1.35.0 → v1.36.0
- go.opentelemetry.io/otel/sdk/metric: v1.34.0 → v1.36.0
- go.opentelemetry.io/otel/sdk: v1.34.0 → v1.36.0
- go.opentelemetry.io/otel/trace: v1.35.0 → v1.36.0
- go.opentelemetry.io/otel: v1.35.0 → v1.36.0
- go.yaml.in/yaml/v2: v2.4.2 → v2.4.3
- golang.org/x/crypto: v0.36.0 → v0.45.0
- golang.org/x/mod: v0.21.0 → v0.29.0
- golang.org/x/net: v0.38.0 → v0.47.0
- golang.org/x/oauth2: v0.27.0 → v0.30.0
- golang.org/x/sync: v0.12.0 → v0.18.0
- golang.org/x/sys: v0.31.0 → v0.38.0
- golang.org/x/telemetry: bda5523 → 078029d
- golang.org/x/term: v0.30.0 → v0.37.0
- golang.org/x/text: v0.23.0 → v0.31.0
- golang.org/x/tools: v0.26.0 → v0.38.0
- google.golang.org/genproto/googleapis/rpc: a0af3ef → 200df99
- google.golang.org/grpc: v1.72.1 → v1.72.2
- google.golang.org/protobuf: v1.36.5 → v1.36.8
- gopkg.in/evanphx/json-patch.v4: v4.12.0 → v4.13.0
- k8s.io/gengo/v2: 85fd79d → ec3ebc5
- k8s.io/kube-openapi: f3f2b99 → 589584f
- k8s.io/system-validators: v1.10.1 → v1.12.1
- k8s.io/utils: 4c0f3b2 → bc988d5
- sigs.k8s.io/json: cfa47c3 → 2d32026

### Removed
- gopkg.in/yaml.v2: v2.4.0



# v1.35.0-rc.1


## Downloads for v1.35.0-rc.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes.tar.gz) | 6b08badb1402c5a4d5e83e14bc64e464c7bd8bbdd9473ea1b501b7bf04ac9f53d2ac23a5f70761cc03024bd046c329253d2914af9225530755bcbc06d7459616
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-src.tar.gz) | 35d2827a2bb7b01162c506d7a15392c72c6537f9f1570ade160bc286ad9a409e0830d32e7ea5ae8191bb6435859826d2d5ec56255a29fe279aee3517239cb9a6

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | 85f9f154296b0579444ee9ff43f74ad616ef52e453782da8dd10f5150d1fb6f1d71151b7525fe5784142f930fb9fe9bfbc395277b5a088d3bc892c9415e15611
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-darwin-arm64.tar.gz) | 2811a1b3901c9a82cb042e4b4c4bc4d75ea0d894b42e7f7c63b25b5df8d26b5ff2bfd0ee819b9cb8e946a2bac90ac4dfb649c692f054fb290e08516966384d0b
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-linux-386.tar.gz) | aed99699ba9f635d1e073061ab89b7ee80744cb56a4aa078ad6afc5670483db456fcc6a544bf34b95a4a3c8f25938898b093d7d279120063688f0406efaf37f9
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | 607724204e2b3265f25d12cda5397c4943e8df4ae66efe2f5cf2582b7aa1c9fc9a60c61a193aef8ae1ee5b3d261146f5cc9956de873d82864979ef19c70c48c7
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-linux-arm.tar.gz) | 378770b5529ce39fbdbb1d39c4a966ad96c03523411e5f830ac2e86e67e23518de534f148838903561de5a7db6f852f532ad7a0799514b72892fea0362f3c635
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | 357f66e108d651d0a77d61f9f0da286f5c8d312d6f320174dab42cd6aae69ecdf24e081f5bdcbc6313d2688c2585009348766f7a0eed58f23f8eda898d3e7081
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | ac9d343138f01a1d6fa986ef684938686e7ac656f8b8b1b5c1a75894e2c98f7f36754eef15a95b20c0812c5a62d775a285c4e015b5eca9bc4e937b9f44b7a537
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | b12631a0bb8c2f9e28ca855302a57ab4bffb2db4105668cf16635193cada26bd0aa63f80428835339f42eea13839d1990f171276697eb00f9cbe5d8edae434f0
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-windows-386.tar.gz) | 5b1849523011be7f569f17f42250a98a2513caa7652c3055712bace9ba440ab4e38d4f5bceb6e89158f4f3008a836fcee765ef9af818ba6cb07296ef8467cf02
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | fb95a54326565bcc3458fa0a876d5f457e5218eafa349f9cee3db93982c30482b8fe57d0f43a2be89e4816382a5efc97b95fae069bc2ed8f19b6ac253e46790a
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-client-windows-arm64.tar.gz) | 465a7fe0e87dd3dc0b22d3881bbe4c33e706ca53a512a981afd49bccdd5d638a818cabb84dbc5796b54ce97d045bbaf466c0cbf46f809bb5a31d64117386a0c7

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | c6728d534c85e11a58dea5ca830541efde07f390586ca77f7f59308179b677f3f3f28492e1c26314adf79e800b7690a1b0d5ac5670d18282064bc4fbb4093b05
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | 6b9eb6d36179bef49b8f4f276f1ffe03552876e46c9477129b6e9a690c38e7343e0430e22fbbfaccab83c3879282e6994cdaca47a709267617a84e3143b4598e
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | 946b9b5d817437258a77e6b3fcf674c0c9adb0a5b4cc2aa3f68f8364c7916016d6811a9596221801f570018f3976ae456eb01a6ad871fb5461daebfd1c2ff589
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | 5d5c22a1149d5c741b5c37710a97a84b2f9f45ccc9b8457408233a6a7c5fc20c3a2bbf92b68f135630308991cdc463559cdad6131fa6c9e0ad5cfdca7933cdc2

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | 269725e2afd028e6eb8a3d12605789f6a5715a0bf9cf477df689bdc9f9b164f0fd32c3007b16ffee17c5234f6fff17ff9096396892e71155074a3499e9480088
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | 37472ba5b63177bcb9430b761f1d4df459771cab2bf9448f69fa5abe0dc4f2e0a26bd6b9c8d747d37e05e8e0263736ca4150a624cecf03938aacd4e98929d03d
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | 25dce33f916a55d27f825902fc7776735a5ddfd0e86555b193d431eab54781d52fff9c41dafcf5a845e79f0153d515b014cb8a5c21179c82e96321851109195c
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | 16e22350e2871ea0d42668bb9c1487b2c5a6e6b436c78d1f986f68bc0312d340ab0285aca7f619322e3238cf9ee90f5425e49dc018e53069e5b0cd092bdd28a2
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | 1b4aab8eb20bd974c7585ba55a5caeb43174afbef738e1bb99fd51fd228b800c943c4a8f0204ef935c733b15cab52ae0ae2a9a7da9299387be272932b1c54c9e

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0-rc.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0-rc.0

## Changes by Kind

### Feature

- Kubernetes is now built using Go 1.25.5 ([#135609](https://github.com/kubernetes/kubernetes/pull/135609), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]

### Bug or Regression

- Fix a bug in the kube-apiserver where a malformed Service without name can cause high CPU usage. The bug is present on the new Cluster IP allocators enabled with the feature MultiCIDRServiceAllocator (enabled by default since 1.33) ([#135499](https://github.com/kubernetes/kubernetes/pull/135499), [@aojea](https://github.com/aojea)) [SIG Testing]
- Fixes a bug where MutatingAdmissionPolicy would fail to apply to objects with duplicate list items (like env vars). ([#135560](https://github.com/kubernetes/kubernetes/pull/135560), [@lalitc375](https://github.com/lalitc375)) [SIG API Machinery]
- K8s.io/client-go: Fixes a regression in 1.34+ which prevented informers from using configured Transformer functions ([#135580](https://github.com/kubernetes/kubernetes/pull/135580), [@serathius](https://github.com/serathius)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Etcd: Update etcd to v3.6.6 ([#135271](https://github.com/kubernetes/kubernetes/pull/135271), [@bzsuni](https://github.com/bzsuni)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
- golang.org/x/crypto: v0.41.0 → v0.45.0
- golang.org/x/mod: v0.28.0 → v0.29.0
- golang.org/x/net: v0.43.0 → v0.47.0
- golang.org/x/sync: v0.17.0 → v0.18.0
- golang.org/x/sys: v0.37.0 → v0.38.0
- golang.org/x/telemetry: 1a19826 → 078029d
- golang.org/x/term: v0.36.0 → v0.37.0
- golang.org/x/text: v0.29.0 → v0.31.0
- golang.org/x/tools: v0.36.0 → v0.38.0

### Removed
_Nothing has changed._



# v1.35.0-rc.0


## Downloads for v1.35.0-rc.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes.tar.gz) | ef5c549823abd93a260ef90bd282362015e8e7e4f079bb59acc3104191bed88eed98638726866ef9d5e56849cbe58b641e7ebaf0d247c0088deaae81141b4007
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-src.tar.gz) | 906dbf942d446b9ffaa56dc31024dd496c609599a5391c81891ed2223c60017eec879de2d68d8ac5003e05c03cdaf5584186b3303291ce81885ec1875e24748c

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | f771e62019fc1cba671015aca0efaf24492598d8200330672e61525a0cd56d406ce53bde56ed1bbdfac4b1ac09e8b4820ce9d612904cf9169f075010a81f3548
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | 74c1b281351df048797db693cb0fe01dc4646d7884b1588e80184fb41ef83bb6016d095e0266eb7993fc2e75d54df6bcf4b8909f6c1e0a7245afb457a6f17516
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-linux-386.tar.gz) | ca6b1747442056c10cbc838ffaca3ace45d7fc773183d526aaec728bb322370f2cc05b2b7304dc19434b40d1fb3b39f7e4eb1930cc9952ac27abccf260968737
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | c9e5e3aa9df4e8c2f1253e4f72cdea94a8deae601b5df0b2cbdac4babdd2d2ac85191c908453bca40a2a0b44247dbb18ef1b5a7751c16660f9487391ed8dacad
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-linux-arm.tar.gz) | 167490337b040849d953de5aa32e3c5491f4d6225fc0c5c86ff6ef9820f61e189f91da26831ee2d910749b38331ec1a73ff820bc29497939935e1f1b3462bf9a
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | 21d7091f25a112e6f940bc1a047294a2c6cd03a8b52c3cfb7ac024460d21ad5ab0a5aea448ec705ed3c5f0619d512a132051eea492e7d7ae008716abbab51a24
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | b4a1b9c03f7de8e0e2c89b1c32fcd4347a287eb62fe21bc8b4d6abafc3ae30559173bd070b5ddcb94038bc7d980e09d946a131ef0191ce2febbadffd1621143e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | 737916148885ade333cbd5c7f927d32e30bb6f6bfe78db8c3c00c05c82d0a6ea5255a6c85b256acb14dc2b322e79e0af6bfa4f2ee055625da0b7ba29f99c847b
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-windows-386.tar.gz) | aa6b19469d3268235012a4a6a2f3d6ee3d56734b083133c213fa7064afef6f3caecadf9a714b6905ba69c22f0103eb44135e51a00907e4ddd1856c14b988ea58
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | 7d7c7edc9e67fecdb0d2a96992933acf2e4d68680ea20349af3feabf449072130de99a9aad6e03ca47806c1e10415de4fb69e56e8f73293fde96eb27bf02a71b
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | c5302ca43bbf8ba1f2f4d89f561edb53c4a6021d084593524a20418dff20c6f4f0052ad85a230961b208a40ca89d4daa64852e9730ce2350932e4d9e369f1621

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | 5aa98c85cefbdc12c7816add8c5eb40438e39779fc96cd0606b21ab89c4d0af354168572f39787cc3f3ef471862d543b1804d433f76f52191490498aa4670255
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | 0942d2aee9638b32a976dc623269332d1f82feb0dc601e009fe97a8179cd23704fdd3ccecb3ecb51cea1abf57bee89d60707b1e69679580e1534ab4222820b01
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | 76a9351112c6f40752f3c2526bcc3fbb100f7e577ca7746644087c386f4e38f01f12b6306a329ec3bcbfc33ce244db95fd3035cc2c37dc32920181f26b7cf78b
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | bb70ad37cd30db82afa9dbd2b6a221f7ed43d6ec54c8a22ff852c5ac5364f2abc7c30db26640d42ba3e265dff70607f5dde86075fe841fe13500c811000ba0bc

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | dcb41e49aed2c0c686c283a721a509d1d72f8cda4ad15216cdf6f65b48b7ba803a88a22aa2dc2bb35c4ffcc329f9184638423e3b0e0e5edd0142ed05922617d1
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | e91bcc56eea9d07aa65b2e855d9b15c6b480c05bc9581cd3aed740d3fe712b064cd9a9dd1baea27e0678747179bd76752ccdf510cc4855764f38b78cf2d9ebdf
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | 7870ee543c2a131451eaaa7b62e7292045e5ee0d4cd592c385a81170d1787835883d58326adabe201df9f920d2bedc8a1c59714b541dabe4f8ee4b9491bbdd23
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | d8265447f63f2466b0d890b3f37918b70cadec0a3e0682f5410414d5456403e992dc859129447b02980b544ddb43cb6feef421d95d29ec5c2bd3fba8811884ca
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | 8cbb1c7fd68dc474fbb88baaf4f169d9adf3ee177a963be3927c51bc08ab19ca5f1cb6a20fc45b52eba8c95b8abdf884d9f140dfacf4b64bfc5f3bf3540cd481

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0-beta.0

## Changes by Kind

### Feature

- Kubernetes is now built using Go 1.25.4 ([#135492](https://github.com/kubernetes/kubernetes/pull/135492), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]

### Bug or Regression

- Fixes a spurious "namespace not found" error possible in default configurations in 1.30+ when using ValidatingAdmissionPolicy or MutatingAdmissionPolicy to intercept namespaced objects in newly-created namespaces ([#135359](https://github.com/kubernetes/kubernetes/pull/135359), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-apiserver: Fixes spurious warning log messages about enabled alpha APIs while starting API server ([#135327](https://github.com/kubernetes/kubernetes/pull/135327), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.35.0-beta.0


## Downloads for v1.35.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes.tar.gz) | 17fae05597b73bf8ed2c14bfbc7d863e6ca470877be12a510cb354bcaf4fa5f9b15b3702e45d231efe9f4865f687bf8d1ace312b4e0a15442a14c9997f1caa07
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-src.tar.gz) | a51fcd8dbe8097f1890931435bdeaf9c1aa31ae3c55ae6abeb504aa881c3e125ecb72af7518a9d3d38ffe67fbcffc6f1dd9e1e456218856ba2f25ec4d466f339

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | 50e6712a9d2a35d782ac0ddb22eb0799fdceca6c434c5ebe446e9f49bf9b7612cd3af3f31af211d8364d128d8b75f87ba9e61466aa7c6552416c50c14b08dd78
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | c181381e2554d20b5ffbe024b33b8593800491bccceb98eaacc25fbc9ef44be4c360ec59603b857b730670c6cdcfe8d8428b790e1413d402f699c6579877ed4a
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-linux-386.tar.gz) | 26fcb99525560328c9ab1e856741e3eb6aecbb3fc8e9adf72daaeb0f6c57058e61989f696ad866a0f9b1a43098914933bed142077754e0bd82bd2054965fe44a
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | d6118e683ea4a64b1812a0eb0374678879a0b0b37868bd8b43517e5961a46b36f9addfbdbd84aa87e6dd4510afa644102140ec727bef3143788e2c97f23cc318
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-linux-arm.tar.gz) | cafd385adecb9ed43201df1c5206f127177fbef0345972a5492c62eb4304c8666eafc1bffa9fb54f530d4258d8f7aaded83f08efaab5391bf92ac50a4d53122f
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | bb7d2281b2b9f02ae61a9607d932873a2dfd1ed551c79209a55feba88219093c8594c03c3915e87f7d847c6e59cf75625123dce8f367c34bb16d4e0ad5681f22
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | 198e2102eb0b24e6c6b406ceb8cce6803154c3e68bef5ace7583683cd49d09b6a8ecbd2ca8f1d105a6454b8550b1d6132e65e618dd0ac49c9d36494af24e3ecb
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | 6e81675b8b523aa1df9f4599548f92bc990a2dbe6ab4f6a4039477e237f2e6286a19e2786ba436bada30969cd5b2f3fd2fb8bc16a41943fcf6865f8b6b690f67
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-windows-386.tar.gz) | 0a68cd18169b4f269766aaff195b07fef44418f6e108ce47f6ce445407a28c1ff27cde4ef015bb01affd2309044a5ef86392132112681edfd5fe2e1632e99862
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | d64bba50e7878fb1bc89dd31e9c4a2796364d2226fcfb14964ca2783d18fe482824b6cd0a4b842076998db1794c6a5e33d78fd3bd2d5ae696ab438d4c7207114
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | c1820e5be65918d6278ed8897059c5d87d8faa8640a5a7bab7f28822d2af19f1dbba650069a469b8eb955353d9c47a7f3098e9c7598f7f1272d08e4edba5bff8

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | f57c7fb934e0261f71fa7f2e219730cc977367bd015a0572d4446f28c9a70e89f641d029d206d27238c7fa27ba166a6c1e81e129b583e8c1555513d5360fabf0
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | 1d5b399f921da76ba0f88c9c28a64880a7be32c017fefd961939c3538e0361cd08f1d7bb38b09982600c5d09711a89c13ee757363d3f10101ee0b1775c85f97e
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | 1b6dbe8765ad8f740e699b842447d513b3b4e0a685db8380a3f38fc89efd3b962b0ec3026e6df55ac4c3cadc0120577d34e4cf5c10311fd1d2bcde9bc47ba844
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | dd0213e41f26158f3cb9a589ca68211d4d68ca2cc9346726361f609b896b4b8975274dd53d3e7561c1f61178bd9aef69c156a435b530def48c215e2d03d18414

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 7c43a88e1b86871d5f76d3d3fed4c458aedcb7d41f3dd944f08a24087b6e0703b2ce8c4d34ee2625de18d75b5bfbc36f5ab6da5158e69ba929cd1cf9f0a205cc
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | 0fbca961eead65de9401ff6211792f27f845003bb5d91655299b3d0c05dc805eca0ade4e0caf784d561734e1ba15ac1112efd92a439953835730a866c92f2205
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | c9bbfcb37f32d267e00067b7334b5a961e875be5776a941ef84a2b2a9d8ee7f2bd97998a789b2180809a9869dbbfa52ef3b5c154447207a83c1de4fa07ed7b05
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | 32b416cc48008e5c1b34af1205a880f8c43a194c264451222bb6281ffa894e4267d2ebdd49e4b08731a44274f5bd7c7ff54733dfd6aaa5df89ac57c12bc50073
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 90eb6f5b268eacadeba1be60f2765740c77a65b3f2b357cec8814a225c26bb8a60341b1c515c0867abf119f3522c7d90281a5490ed7c8926c2ecb7c8ec0b4fe9

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0-alpha.3

## Changes by Kind

### API Change

- Add scoring for the prioritized list feature so that the node that can satisfy the best ranked subrequests are chosen. ([#134711](https://github.com/kubernetes/kubernetes/pull/134711), [@mortent](https://github.com/mortent)) [SIG Node, Scheduling and Testing]
- Allows restart all containers when the source container exits with a matching restart policy rule. This is an alpha feature behind feature gate RestartAllContainersOnContainerExit. ([#134345](https://github.com/kubernetes/kubernetes/pull/134345), [@yuanwang04](https://github.com/yuanwang04)) [SIG Apps, Node and Testing]
- Changed kuberc configuration schema. Two new optional fields added to kuberc configuration, `credPluginPolicy` and `credPluginAllowlist`. This is documented in [KEP-3104](https://github.com/kubernetes/enhancements/blob/master/keps/sig-cli/3104-introduce-kuberc/README.md#allowlist-design-details) and documentation is added to the website by [kubernetes/website#52877](https://github.com/kubernetes/website/pull/52877) ([#134870](https://github.com/kubernetes/kubernetes/pull/134870), [@pmengelbert](https://github.com/pmengelbert)) [SIG API Machinery, Architecture, Auth, CLI, Instrumentation and Testing]
- Enhanced discovery response to support merged API groups/resources from all peer apiservers when UnknownVersionInteroperabilityProxy feature is enabled ([#133648](https://github.com/kubernetes/kubernetes/pull/133648), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Auth, Cloud Provider, Node, Scheduling and Testing]
- Extend `core/v1 Toleration` to support numeric comparison operators (`Gt`, `Lt`). ([#134665](https://github.com/kubernetes/kubernetes/pull/134665), [@helayoty](https://github.com/helayoty)) [SIG API Machinery, Apps, Node, Scheduling, Testing and Windows]
- Features: NominatedNodeNameForExpectation in kube-scheduler and CleaeringNominatedNodeNameAfterBinding in kube-apiserver are now enabled by default. ([#135103](https://github.com/kubernetes/kubernetes/pull/135103), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Implement changes to prevent pod scheduling to a node without CSI driver ([#135012](https://github.com/kubernetes/kubernetes/pull/135012), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Scheduling, Storage and Testing]
- Introduce scheduling.k8s.io/v1alpha1 Workload API to allow for expressing workload-level scheduling requirements and let kube-scheduler act on those. ([#134564](https://github.com/kubernetes/kubernetes/pull/134564), [@macsko](https://github.com/macsko)) [SIG API Machinery, Apps, CLI, Etcd, Scheduling and Testing]
- Introduce the alpha MutableSchedulingDirectivesForSuspendedJobs feature gate (disabled by default) which:
  1. allows to mutate Job's scheduling directives for suspended Jobs
  2. makes the Job controller to clear the status.startTime field for suspended Jobs ([#135104](https://github.com/kubernetes/kubernetes/pull/135104), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- Introduced GangScheduling kube-scheduler plugin to enable "all-or-nothing" scheduling. Workload API in scheduling.k8s.io/v1alpha1 is used to express the desired policy. ([#134722](https://github.com/kubernetes/kubernetes/pull/134722), [@macsko](https://github.com/macsko)) [SIG API Machinery, Apps, Auth, CLI, Etcd, Scheduling and Testing]
- PV node affinity is now mutable. ([#134339](https://github.com/kubernetes/kubernetes/pull/134339), [@huww98](https://github.com/huww98)) [SIG API Machinery, Apps and Node]
- ResourceQuota now counts device class requests within a ResourceClaim object as consuming two additional quotas when the DRAExtendedResource feature is enabled:
  - `requests.deviceclass.resource.k8s.io/<deviceclass>` with a quantity equal to the worst case count of devices requested
  - requests for device classes that map to an extended resource consume `requests.<extended resource name>` ([#134210](https://github.com/kubernetes/kubernetes/pull/134210), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- The DRA device taints and toleration feature now has a separate feature gate, DRADeviceTaintRules, which controls whether support for DeviceTaintRules is enabled. It is possible to disable that and keep DRADeviceTaints enabled, in which case tainting by DRA drivers through ResourceSlices continues to work. ([#135068](https://github.com/kubernetes/kubernetes/pull/135068), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Node, Scheduling and Testing]
- The ImagePullIntent and ImagePulledRecord objects used by kubelet to store information about image pulls have been moved to the v1beta1 API version. ([#132579](https://github.com/kubernetes/kubernetes/pull/132579), [@stlaz](https://github.com/stlaz)) [SIG Auth and Node]
- The KubeletEnsureSecretPulledImages feature is now beta and enabled by default. ([#135228](https://github.com/kubernetes/kubernetes/pull/135228), [@aramase](https://github.com/aramase)) [SIG Auth, Node and Testing]
- This change adds a new alpha feature Node Declared Features, which includes:
  - A new `Node.Status.DeclaredFeatures` field for Kubelet to publish node-specific features.
  - A library in `component-helpers` for feature registration and inference.
  - A scheduler plugin (`NodeDeclaredFeatures`) scheduler plugin to match pods with nodes that provide their required features.
  - An admission plugin (`NodeDeclaredFeatureValidator`) to validate pod updates against a node's declared features. ([#133389](https://github.com/kubernetes/kubernetes/pull/133389), [@pravk03](https://github.com/pravk03)) [SIG API Machinery, Apps, Node, Release, Scheduling and Testing]
- This change allows In Place Resize of Pod Level Resources 
  - Add Resources in PodStatus to capture resources set at pod-level cgroup
  - Add AllocatedResources in PodStatus to capture resources requested in the PodSpec ([#132919](https://github.com/kubernetes/kubernetes/pull/132919), [@ndixita](https://github.com/ndixita)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Instrumentation, Node, Scheduling and Testing]
- Updates to the Partitionable Devices feature which allows for referencing counter sets across different ResourceSlices within the same resource pool.
  
  Devices from incomplete pools are no longer considered for allocation.
  
  This contains backwards incompatible changes to the Partitionable Devices alpha feature, so any ResourceSlices that uses the feature should be removed prior to upgrading or downgrading between 1.34 and 1.35. ([#134189](https://github.com/kubernetes/kubernetes/pull/134189), [@mortent](https://github.com/mortent)) [SIG API Machinery, Node, Scheduling and Testing]

### Feature

- Add cloud-controller-manager feature gate CloudControllerManagerWatchBasedRoutesReconciliation ([#131220](https://github.com/kubernetes/kubernetes/pull/131220), [@lukasmetzner](https://github.com/lukasmetzner)) [SIG API Machinery and Cloud Provider]
- Add the `UserNamespacesHostNetworkSupport` feature gate. The feature gate defaults to disabled. When the feature gate is enabled, will allow `hostNetwork` pods to use `user namespace`. ([#134893](https://github.com/kubernetes/kubernetes/pull/134893), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Apps, Node and Testing]
- Added a new `source` label in `resourceclaim_controller_resource_claims`.
  Added a new metrics for DRAExtendedResource `scheduler_resourceclaim_creates_total`. ([#134523](https://github.com/kubernetes/kubernetes/pull/134523), [@bitoku](https://github.com/bitoku)) [SIG Apps, Instrumentation, Node and Scheduling]
- Added configurable per-device health check timeouts to the DRA health monitoring API. ([#135147](https://github.com/kubernetes/kubernetes/pull/135147), [@harche](https://github.com/harche)) [SIG Node]
- Bump ImageGCMaximumAge to stable ([#134736](https://github.com/kubernetes/kubernetes/pull/134736), [@haircommander](https://github.com/haircommander)) [SIG Node and Testing]
- Enables the `WatchListClient` feature gate. ([#134180](https://github.com/kubernetes/kubernetes/pull/134180), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery, Apps, Auth, CLI, Instrumentation, Node and Testing]
- Graduate PodTopologyLabelsAdmission feature gate to Beta and on by default.
  
  Pods will now have labels `topology.kubernetes.io/zone` and `topology.kubernetes.io/region` by default if the assigned Node has these labels. ([#135158](https://github.com/kubernetes/kubernetes/pull/135158), [@andrewsykim](https://github.com/andrewsykim)) [SIG Node]
- Graduate image volume source to on by default Beta ([#135195](https://github.com/kubernetes/kubernetes/pull/135195), [@haircommander](https://github.com/haircommander)) [SIG Apps, Instrumentation, Node and Testing]
- Implement scoring for DRA-backed extended resources ([#134058](https://github.com/kubernetes/kubernetes/pull/134058), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Scheduling and Testing]
- KEP-3619: fined-grained supplemental groups policy is graduated to GA. ([#135088](https://github.com/kubernetes/kubernetes/pull/135088), [@everpeace](https://github.com/everpeace)) [SIG Node and Testing]
- KEP-5440: Allow for resizing of resources while job is suspended.  This feature is alpha. ([#132441](https://github.com/kubernetes/kubernetes/pull/132441), [@kannon92](https://github.com/kannon92)) [SIG Apps and Testing]
- KEP-5598 opportunistic batching is implemented to optimize scheduling for pods that have the same scheduling requirements. ([#135231](https://github.com/kubernetes/kubernetes/pull/135231), [@bwsalmon](https://github.com/bwsalmon)) [SIG Node, Scheduling, Storage and Testing]
- Kubeadm: Add `HTTPEndpoints` field to `ClusterConfiguration.Etcd.ExternalEtcd` that can be used to configure the HTTP endpoints for etcd communication in v1beta4. This field is used to separate the HTTP traffic (such as /metrics and /health endpoints) from the gRPC traffic handled by Endpoints. This separation allows for better access control, as HTTP endpoints can be exposed without exposing the primary gRPC interface. Corresponds to etcd's `--listen-client-http-urls` configuration. If not provided, Endpoints will be used for both gRPC and HTTP traffic. ([#134890](https://github.com/kubernetes/kubernetes/pull/134890), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubernetes is now built with go 1.25.4 ([#135187](https://github.com/kubernetes/kubernetes/pull/135187), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- New metrics are introduced related to Ensure Secret Pulled Images KEP:
      - kubelet_imagemanager_ondisk_pullintents - the number of pull intent records currently kept on disk
      - kubelet_imagemanager_ondisk_pulledrecords - the number of image pulled records currently kept on disk
      - kubelet_imagemanager_image_mustpull_checks_total{result} - the number for how many times an image was checked against the pull records and the results of those checks ([#132812](https://github.com/kubernetes/kubernetes/pull/132812), [@stlaz](https://github.com/stlaz)) [SIG Auth and Node]
- Pick one device class deterministically for extended resource when there are more than one ([#135037](https://github.com/kubernetes/kubernetes/pull/135037), [@yliaog](https://github.com/yliaog)) [SIG Node, Scheduling and Testing]
- Promoted the `EnvFiles` feature gate to beta and is enabled by default. Additionally, the syntax specification for environment variables has been restricted to a subset of POSIX shell syntax (all variable values must be wrapped in single quotes). ([#134414](https://github.com/kubernetes/kubernetes/pull/134414), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node and Testing]
- Promoted the `KubeletCrashLoopBackOffMax` feature gate to beta, it is now enabled by default. ([#135044](https://github.com/kubernetes/kubernetes/pull/135044), [@hankfreund](https://github.com/hankfreund)) [SIG Node]
- The Pod Certificates feature is moving to beta. The PodCertificateRequest feature gate is still set false by default. To use the feature, users will need to enable the certificates API groups in v1beta1 and enable the feature gate PodCertificateRequest. A new field UserAnnotations is added to the PodCertificateProjection API and the corresponding UnverifiedUserAnnotations is added to the PodCertificateRequest API. ([#134790](https://github.com/kubernetes/kubernetes/pull/134790), [@yt2985](https://github.com/yt2985)) [SIG Auth, Instrumentation and Testing]
- When resizing pods, more events will be emitted when the pod's resize status changes. ([#134825](https://github.com/kubernetes/kubernetes/pull/134825), [@natasha41575](https://github.com/natasha41575)) [SIG Node]

### Bug or Regression

- Extended resources requested by initContainers which are allocated using an automatic ResourceClaim now match the behavior of legacy device plugins, reusing the same resources requested by later sidecar initContainers or regular containers when possible, to minimize the total number of devices requested by the pod. ([#134882](https://github.com/kubernetes/kubernetes/pull/134882), [@yliaog](https://github.com/yliaog)) [SIG Apps, CLI, Node, Scheduling and Testing]
- Fix Windows kube-proxy (winkernel) issue where stale RemoteEndpoints remained
  when a Deployment was referenced by multiple Services due to premature clearing
  of the terminatedEndpoints map. ([#135146](https://github.com/kubernetes/kubernetes/pull/135146), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fix bug in ValidatingAdmissionPolicy where a object schema with additionalProperties:true would crash the kube-controller-manager with a nil pointer exception. ([#135155](https://github.com/kubernetes/kubernetes/pull/135155), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fixes an issue that disallowed restart policies and restart rules on static pods. ([#135031](https://github.com/kubernetes/kubernetes/pull/135031), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node]
- Fixes the replacement tag in APIs to not be a selector for storage version ([#135197](https://github.com/kubernetes/kubernetes/pull/135197), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Kube-apiserver: Fixes spurious warning log messages about enabled alpha APIs while starting API server ([#135327](https://github.com/kubernetes/kubernetes/pull/135327), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Kubelet: fix concurrent map write error when creating a pod with empty volume when the LocalStorageCapacityIsolationFSQuotaMonitoring feature-gate is enabled ([#135174](https://github.com/kubernetes/kubernetes/pull/135174), [@carlory](https://github.com/carlory)) [SIG Storage]
- Support ShareID of DRAConsumableCapacity feature in the Kubelet Plugin API ([#134520](https://github.com/kubernetes/kubernetes/pull/134520), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node and Testing]
- The slow initialization of container runtime will not cause System WatchDog to kill kubelet. Device Manager is not considered healthy before it attempted to start listening on the port. ([#135153](https://github.com/kubernetes/kubernetes/pull/135153), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node]
- Typed workqueue now cleans up goroutines before shutting down ([#135072](https://github.com/kubernetes/kubernetes/pull/135072), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]

### Other (Cleanup or Flake)

- AggregatedDiscoveryRemoveBetaType feature gate is deprecated and locked to True ([#134230](https://github.com/kubernetes/kubernetes/pull/134230), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Dropped support for networking/v1beta1 Ingress in kubectl ([#135176](https://github.com/kubernetes/kubernetes/pull/135176), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Dropped support for networking/v1beta1 IngressClass in kubectl ([#135108](https://github.com/kubernetes/kubernetes/pull/135108), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Upgrade CoreDNS to v1.12.4 ([#133968](https://github.com/kubernetes/kubernetes/pull/133968), [@yashsingh74](https://github.com/yashsingh74)) [SIG Cloud Provider and Cluster Lifecycle]

## Dependencies

### Added
- cyphar.com/go-pathrs: v0.2.1

### Changed
- github.com/coredns/corefile-migration: [v1.0.27 → v1.0.29](https://github.com/coredns/corefile-migration/compare/v1.0.27...v1.0.29)
- github.com/cyphar/filepath-securejoin: [v0.4.1 → v0.6.0](https://github.com/cyphar/filepath-securejoin/compare/v0.4.1...v0.6.0)
- github.com/opencontainers/selinux: [v1.11.1 → v1.13.0](https://github.com/opencontainers/selinux/compare/v1.11.1...v1.13.0)

### Removed
_Nothing has changed._



# v1.35.0-alpha.3


## Downloads for v1.35.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes.tar.gz) | 054e77631e6a17dcb1589e14aaf215672c054a3315de0e72fad066d5f4392ff09288dc0ead2e9667c65c3c7c770d81206abb94eaf2615b1ef0cc99fbf3a5c793
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-src.tar.gz) | fe30a5b352bb1656d7306aec0f491fde6f874af7d749fa31fe75ac5035c98d3c63d95db1b0c0024b30c55eadf7b60a1c3513a343eff2d6b0793147112940c82b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | ca86f0ff39c9ee9ddf75674369cac952652afb3d36c11d8b761d00e9a6f9827adda24d87db6d936ab4ff54cd3d65afcc1e8b77868bc8054837d36cc9725a0fe8
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | b5a6772bfd7fd59ad18d0ccd6cece28d316613c1364607bdcb6389b2be1e911297b8ea3fb4b0ced7c38e66be36bf3f42898e4a5fade67add6a29cc5caec0f449
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 8ecf519056385911fcec30039c8c3bf8537726c35ad9637602444dc6f1c5cc4f34fd2b924641b64b5a94b81935deff3a1445bc161fa3c3887a26b6a572e5a126
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | 499d946c3baf4bf55cc12ae0166ebbd3ae2c0c383d0f0cabca18cdc843b101e4fe0a972117f01d59e6eb61056471bdf5ff7b1c124e42298a4d758c01f8d888dc
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | fc240f0e23fad7578330cfb65ee271b3dfee099fd4fea3df6e5bd6cd5c50d8d398915d3c1dd735593b96ed1f3d30a800dd3ee6b1553c32bbc46428823ff68d6d
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | ee4a49a4c55d9fce0cddf8150fa506df2c498abb257bb87c773260c26dc32fcb14be97630a27a22dd7406207778c7111f95751dc80b5dfabdf0755757b9c7082
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | 758eab71ab6435a689ac081bad27270967b6f8a09532b2dbc1c45b16eb8cc9ee24d317c4c8adf4345c569e89a1edeea8fb6f1bf97f7f84604fc17a7459f9a59f
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 13947dfc8a67de7805e2e0818452d287079f9f382c8e36e8501b0871c5083f3eef1ac0461ca3570abeb39f84391b75843236a98f56f17a75f01a3d88cbfc6998
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 813f670f33f20dfbec2dfd53136831f1117b5d172fa381fb1f69348d9f2e1cdda5eff2f807529924d1751d31011f3ba0a9dfd2e395114f8c289cbc3a262a207b
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 6cfafe20404a2d6d8d7f9ed923eabc59360ba16454db8602de7aaaf3f40af7ff0429f54c3a34fd8c94d4a2e83bffeab29de5eed78f45f3cbe4027a8ae23a25c9
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | dc4f018a9d7182c32f82727e42624f6b5883e944a730855ab0dd9ab9e8a5eea0766c5214ce5bf63c3bafc795d28bc335a94c9e50f1c4c80887c944780bb7811a

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 741bc1b0cb536ae82284b299fbb27c466e7ce3b54ba879a40631c5c00d822bce76dbb927d51ccb50383f22e115c4f0a5d22d8157cd9ea69da797f2fae2229b50
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 26d073a93c26511aa3ec2e47954193175a87426d6f489370cbc5d2cbc636e98785a8c065d3cee1e3fcd52f4ee2b37e3137ade65739704b4aa3582c41d9e69341
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | 2ce67d08040129bc1d290faf63573f7e1881f2ec7eaf02a4a27cbd48285fc315ff336d245c63f6bb8dd5b2e82821beed731bf9e9f807a4d5a0fadac355413183
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 6b12151c9ab895a9c51f7e17b067d165b511f8c7e32c5ee2cb9924087314bcacd74826e1b18bccd1e06b85a9a3c26e151c38fed9a4f40794777bd06f68cb3e95

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | f13943abe46974a701c1de6a20f76a2ade96db4795de7f6615680c3a360d602d5efca1d062c206f5154e3c3f504c0e51fda10e96ed31e20b3bd3d711be3600f8
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | b7b664dde1dea0469dcab0a8f30032c583210d008621580930feb4a56353f9d51b732643fb41600febae3da3f2f17617c7914487539e4d7be8b4942c52219c85
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 38cc958f6bc855b9fb6da6dbe1dd4eda874916865b030b929eab5f4110fa9554d7531757992e54ad912ca41d7eee6f01e6a299132c023199d7751491ae5456da
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | c32b6d753f1c76bfcac7c37d5986d243cb5b7ad6bd01596b84d4262250ba31d875005302e8b92f7b8bac9ad30a85a6a60f99609fbbcceb0df1d08dfad8539488
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | c5dfdcf501f39003ff818fa66c90e3874d9db21afc74ce9d6fe20de6f074d755ee0a90e18e47ffebda0da5685e9b42e8385f6e7e3d518b08a911c019686257d9

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0-alpha.2

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - ACTION REQUIRED
  
  vendor: updated k8s.io/system-validators to v1.12.1. The cgroups validator will now throw an error instead of a warning if cgroups v1 is detected on the host and the provided KubeletVersion is 1.35 or newer.
  
  kubeadm: started using k8s.io/system-validators v1.12.1 in kubeadm 1.35. During `kubeadm init`, `kubeadm join` and `kubeadm upgrade`, the SystemVerification preflight check will throw an error if cgroups v1 is detected and if the detected kubelet version is 1.35 or newer. For older versions of kubelet, there will be just a preflight warning.
  
  To allow cgroups v1 with kubeadm and kubelet version 1.35 or newer, you must:
  - Ignore the error from the SystemVerifcation preflight check by kubeadm.
  - Edit the kube-system/kubelet-config ConfigMap and add the `failCgroupV1: false` field, before upgrading. ([#134744](https://github.com/kubernetes/kubernetes/pull/134744), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
  - Removed the `--pod-infra-container-image` flag from kubelet's command line. For non-kubeadm clusters, users must manually remove this flag from their kubelet configuration to prevent startup failures before they upgrade kubelet. ([#133779](https://github.com/kubernetes/kubernetes/pull/133779), [@carlory](https://github.com/carlory)) [SIG Node]
 
## Changes by Kind

### API Change

- Add ObservedGeneration to CustomResourceDefinition Conditions. ([#134984](https://github.com/kubernetes/kubernetes/pull/134984), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Add StorageVersionMigration v1beta1 api and remove the v1alpha API. 
  
  Any use of the v1alpha1 api is no longer supported and 
  users must remove any v1alpha1 resources prior to upgrade. ([#134784](https://github.com/kubernetes/kubernetes/pull/134784), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth, Etcd and Testing]
- CSI drivers can now opt-in to receive service account tokens via the secrets field instead of volume context by setting `spec.serviceAccountTokenInSecrets: true` in the CSIDriver object. This prevents tokens from being exposed in logs and other outputs. The feature is gated by the `CSIServiceAccountTokenSecrets` feature gate (Beta in v1.35). ([#134826](https://github.com/kubernetes/kubernetes/pull/134826), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth, Storage and Testing]
- DRA device taints: DeviceTaintRule status provided information about the rule, in particular whether pods still need to be evicted ("EvictionInProgress" condition). The new "None" effect can be used to preview what a DeviceTaintRule would do if it used the "NoExecute" effect and to taint devices ("device health") without immediately affecting scheduling or running pods. ([#134152](https://github.com/kubernetes/kubernetes/pull/134152), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Node, Release, Scheduling and Testing]
- DRA: the DynamicResourceAllocation feature gate for the core functionality (GA in 1.34) is now locked to enabled-by-default and thus cannot be disabled anymore. ([#134452](https://github.com/kubernetes/kubernetes/pull/134452), [@pohly](https://github.com/pohly)) [SIG Auth, Node, Scheduling and Testing]
- Forbid adding resources other than CPU & memory on pod resize. ([#135084](https://github.com/kubernetes/kubernetes/pull/135084), [@tallclair](https://github.com/tallclair)) [SIG Apps, Node and Testing]
- Implement constrained impersonation as described in https://kep.k8s.io/5284 ([#134803](https://github.com/kubernetes/kubernetes/pull/134803), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Introduces a structured and versioned v1alpha1 response for flagz ([#134995](https://github.com/kubernetes/kubernetes/pull/134995), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Architecture, Instrumentation, Network, Node, Scheduling and Testing]
- Introduces a structured and versioned v1alpha1 response for statusz ([#134313](https://github.com/kubernetes/kubernetes/pull/134313), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation, Network, Node, Scheduling and Testing]
- New `--min-compatibility-version` flag for apiserver, kcm and kube scheduler ([#133980](https://github.com/kubernetes/kubernetes/pull/133980), [@siyuanfoundation](https://github.com/siyuanfoundation)) [SIG API Machinery, Architecture, Cluster Lifecycle, Etcd, Scheduling and Testing]
- Promote PodObservedGenerationTracking to GA. ([#134948](https://github.com/kubernetes/kubernetes/pull/134948), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Promoted Job Managed By to general availability. The `JobManagedBy` feature gate is now locked to true, and will be removed in a future release of Kubernetes. ([#135080](https://github.com/kubernetes/kubernetes/pull/135080), [@dejanzele](https://github.com/dejanzele)) [SIG API Machinery, Apps and Testing]
- Promoted ReplicaSet and Deployment `.status.terminatingReplicas` tracking to beta. The `DeploymentReplicaSetTerminatingReplicas` feature gate is now enabled by default. ([#133087](https://github.com/kubernetes/kubernetes/pull/133087), [@atiratree](https://github.com/atiratree)) [SIG API Machinery, Apps and Testing]
- Scheduler: added a new `bindingTimeout` argument to the DynamicResources plugin configuration.
  This allows customizing the wait duration in PreBind for device binding conditions.
  Defaults to 10 minutes when DRADeviceBindingConditions and DRAResourceClaimDeviceStatus are both enabled. ([#134905](https://github.com/kubernetes/kubernetes/pull/134905), [@fj-naji](https://github.com/fj-naji)) [SIG Node and Scheduling]
- The Pod Certificates feature is moving to beta. The PodCertificateRequest feature gate is still set false by default. To use the feature, users will need to enable the certificates API groups in v1beta1 and enable the feature gate PodCertificateRequest. A new field UserAnnotations is added to the PodCertificateProjection API and the corresponding UnverifiedUserAnnotations is added to the PodCertificateRequest API. ([#134624](https://github.com/kubernetes/kubernetes/pull/134624), [@yt2985](https://github.com/yt2985)) [SIG API Machinery, Apps, Auth, Etcd, Instrumentation, Node and Testing]
- The StrictCostEnforcementForVAP and StrictCostEnforcementForWebhooks feature gates, locked on since 1.32, have been removed ([#134994](https://github.com/kubernetes/kubernetes/pull/134994), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, Node and Testing]
- The `PreferSameZone` and `PreferSameNode` values for Service's
  `trafficDistribution` field are now GA. The old value `PreferClose` is now
  deprecated in favor of the more-explicit `PreferSameZone`. ([#134457](https://github.com/kubernetes/kubernetes/pull/134457), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Apps, Network and Testing]

### Feature

- Add the `ChangeContainerStatusOnKubeletRestart` feature gate. The feature gate defaults to disabled. When the feature gate is disabled, the kubelet does not change the pod status upon restart, and pods will not re-run startup probes after kubelet restart. ([#134746](https://github.com/kubernetes/kubernetes/pull/134746), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node and Testing]
- Added a new `source` label in `resourceclaim_controller_resource_claims`.
  Added a new metrics for DRAExtendedResource `scheduler_resourceclaim_creates_total`. ([#134523](https://github.com/kubernetes/kubernetes/pull/134523), [@bitoku](https://github.com/bitoku)) [SIG Apps, Instrumentation, Node and Scheduling]
- Added support for tracing in kubectl with --profile=trace ([#134709](https://github.com/kubernetes/kubernetes/pull/134709), [@tchap](https://github.com/tchap)) [SIG CLI]
- Adding new kuberc view/set commands in kubectl to perform operations against kuberc file ([#135003](https://github.com/kubernetes/kubernetes/pull/135003), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Enable MutableCSINodeAllocatableCount by default. ([#134647](https://github.com/kubernetes/kubernetes/pull/134647), [@torredil](https://github.com/torredil)) [SIG Storage]
- Improved throughput in the real-FIFO queue used by informer/controllers by adding batch handling for processing watch events. ([#132240](https://github.com/kubernetes/kubernetes/pull/132240), [@yue9944882](https://github.com/yue9944882)) [SIG API Machinery, Scheduling and Storage]
- Introducing new flag --as-user-extra persistent flag in kubectl that can be used to pass extra arguments during the impersonation ([#134378](https://github.com/kubernetes/kubernetes/pull/134378), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Kube-apiserver: JWT authenticator now report the following metrics:
  - apiserver_authentication_jwt_authenticator_jwks_fetch_last_timestamp_seconds
  - apiserver_authentication_jwt_authenticator_jwks_fetch_last_key_set_info
  
  when StructuredAuthenticationConfiguration feature is enabled. ([#123642](https://github.com/kubernetes/kubernetes/pull/123642), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kubeadm: added a new preflight check `ContainerRuntimeVersion ` to validate if the installed container runtime supports the RuntimeConfig gRPC method. If the container runtime does not support the RuntimeConfig gRPC method, kubeadm will print a warning message. 
  
  Once Kubernetes 1.36 is released, the kubelet might refuse to start if the CRI runtime does not support this feature. More information can be found in https://kubernetes.io/blog/2025/09/12/kubernetes-v1-34-cri-cgroup-driver-lookup-now-ga/. ([#134906](https://github.com/kubernetes/kubernetes/pull/134906), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- New counter metric exposing details about kubelet ensuring an image exists on the node is added - `kubelet_image_manager_ensure_image_requests_total{present_locally, pull_policy, pull_required}` ([#132644](https://github.com/kubernetes/kubernetes/pull/132644), [@stlaz](https://github.com/stlaz)) [SIG Auth and Node]
- Promote InPlacePodVerticalScaling to GA. ([#134949](https://github.com/kubernetes/kubernetes/pull/134949), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Node and Scheduling]
- Promote Relaxed validation for Services names to beta (enabled by default)
  
  Promote `RelaxedServiceNameValidation` feature to beta (enabled by default)
  The  names of new Services names are validation with `NameIsDNSLabel()`,
  relaxing the  pre-existing validation. ([#134493](https://github.com/kubernetes/kubernetes/pull/134493), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network]
- Promote kubectl command headers to stable ([#134777](https://github.com/kubernetes/kubernetes/pull/134777), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- The SchedulerAsyncAPICalls feature gate has been re-enabled by default after fixing regressions detected in v1.34. ([#135059](https://github.com/kubernetes/kubernetes/pull/135059), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- The scheduler clears the `nominatedNodeName` field for Pods upon scheduling or binding failure. External components, such as Cluster Autoscaler and Karpenter, should not overwrite this field. ([#135007](https://github.com/kubernetes/kubernetes/pull/135007), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Scheduling and Testing]

### Bug or Regression

- BlockOwnerDeletion is removed from resource claims created from resource claim templates, and extended resource claims created by scheduler ([#134956](https://github.com/kubernetes/kubernetes/pull/134956), [@yliaog](https://github.com/yliaog)) [SIG Apps, Node and Scheduling]
- Drop DeviceBindingConditions fields if the DRADeviceBindingConditions is not enabled and not in-use ([#134964](https://github.com/kubernetes/kubernetes/pull/134964), [@sunya-ch](https://github.com/sunya-ch))
- Fix a very old issue where kubelet rejects pods with NodeAffinityFailed due to a stale informer cache. ([#134445](https://github.com/kubernetes/kubernetes/pull/134445), [@natasha41575](https://github.com/natasha41575)) [SIG Node]
- Fix issue in asynchronous preemption: Scheduler checks if preemption is ongoing for a pod before initiating new preemption calls ([#134730](https://github.com/kubernetes/kubernetes/pull/134730), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Scheduling and Testing]
- Fix panic on kubectl api-resources ([#134833](https://github.com/kubernetes/kubernetes/pull/134833), [@rikatz](https://github.com/rikatz)) [SIG CLI]
- Fix setting distinctAttribute=nil when DRAConsumableCapacity is disabled ([#134962](https://github.com/kubernetes/kubernetes/pull/134962), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node]
- Fix the bug which could result in Job status updates failing with the error:
  status.startTime: Required value: startTime cannot be removed for unsuspended job
  The error could be raised after a Job is resumed, if started and suspended previously. ([#134769](https://github.com/kubernetes/kubernetes/pull/134769), [@dejanzele](https://github.com/dejanzele)) [SIG Apps and Testing]
- Fix: The requests for a config FromClass in the status of a ResourceClaim were not referenced. ([#134793](https://github.com/kubernetes/kubernetes/pull/134793), [@LionelJouin](https://github.com/LionelJouin)) [SIG Node]
- Fixed a bug that caused a deleted pod staying in the binding phase to occupy space on the node in the kube-scheduler. ([#134157](https://github.com/kubernetes/kubernetes/pull/134157), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- Fixed a bug that prevent allocating the same device that was previously consuming the CounterSet when enabling both DRAConsumableCapacity and DRAPartitionableDevices. ([#134103](https://github.com/kubernetes/kubernetes/pull/134103), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node]
- Fixed a bug where the health of a DRA resource was not reported in the Pod status if the resource claim was generated from a template or used a different local name in the pod spec. ([#134875](https://github.com/kubernetes/kubernetes/pull/134875), [@Jpsassine](https://github.com/Jpsassine)) [SIG Node and Testing]
- Fixes an issue where the kubelet /configz endpoint reported incorrect value for kubeletconfig.cgroupDriver when the cgroup driver setting is received from the container runtime. ([#134743](https://github.com/kubernetes/kubernetes/pull/134743), [@marquiz](https://github.com/marquiz)) [SIG Node]
- Fixes bug where AllocationMode: All would not succeed if a resource pool contained ResourceSlices that wasn't targeting the current node. ([#134466](https://github.com/kubernetes/kubernetes/pull/134466), [@mortent](https://github.com/mortent)) [SIG Node]
- Kube-controller-manager: Fixes a 1.34 regression, which triggered a spurious rollout of existing statefulsets when upgrading the control plane from 1.33 → 1.34. This fix is guarded by a `StatefulSetSemanticRevisionComparison` feature gate, which is enabled by default. ([#135017](https://github.com/kubernetes/kubernetes/pull/135017), [@liggitt](https://github.com/liggitt)) [SIG Apps]
- Kube-scheduler: Pod statuses no longer include specific taint keys or values when scheduling fails because of untolerated taints ([#134740](https://github.com/kubernetes/kubernetes/pull/134740), [@hoskeri](https://github.com/hoskeri)) [SIG Scheduling]
- Namespace is added to the output of dry-run=client of HPA object ([#134263](https://github.com/kubernetes/kubernetes/pull/134263), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]

### Other (Cleanup or Flake)

- Added a new filed `Step` in the testing framework to allow volume expansion in configurable step sizes for tests. ([#134760](https://github.com/kubernetes/kubernetes/pull/134760), [@Rishita-Golla](https://github.com/Rishita-Golla)) [SIG Storage and Testing]
- Dropped support for certificates/v1beta1 CertificateSigningRequest in kubectl ([#134782](https://github.com/kubernetes/kubernetes/pull/134782), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Dropped support for discovery/v1beta1 EndpointSlice in kubectl ([#134913](https://github.com/kubernetes/kubernetes/pull/134913), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Dropped support for networking/v1beta1 IngressClass in kubectl ([#135108](https://github.com/kubernetes/kubernetes/pull/135108), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Eliminate use of md5 and prevent future use of md5 in favor of more appropriate hashing algorithms. ([#133511](https://github.com/kubernetes/kubernetes/pull/133511), [@BenTheElder](https://github.com/BenTheElder)) [SIG Apps, Architecture, CLI, Cluster Lifecycle, Network, Node, Security, Storage and Testing]
- Kubeadm: removed the kubeadm-specific feature gate WaitForAllControlPlaneComponents which graduated to GA in 1.34 and was locked to enabled by default. ([#134781](https://github.com/kubernetes/kubernetes/pull/134781), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: updated the supported etcd version to v3.5.24 for supported control plane versions v1.32, v1.33, and v1.34. ([#134779](https://github.com/kubernetes/kubernetes/pull/134779), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Migrate the cpumanager to contextual logging ([#125912](https://github.com/kubernetes/kubernetes/pull/125912), [@ffromani](https://github.com/ffromani)) [SIG Node]
- Removed the `UserNamespacesPodSecurityStandards` feature gate. The minimum supported Kubernetes version for a kubelet is now v1.31, so the gate is not needed. ([#132157](https://github.com/kubernetes/kubernetes/pull/132157), [@haircommander](https://github.com/haircommander)) [SIG Auth, Node and Testing]
- The FeatureGate SystemdWatchdog is locked to default and will be removed. The Systemd Watchdog functionality in kubelet can be turned on via Systemd without any feature gate set up. See https://kubernetes.io/docs/reference/node/systemd-watchdog/ for information. ([#134691](https://github.com/kubernetes/kubernetes/pull/134691), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node]
- Updates the etcd client library to v3.6.5 ([#134780](https://github.com/kubernetes/kubernetes/pull/134780), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]

## Dependencies

### Added
- github.com/Masterminds/semver/v3: [v3.4.0](https://github.com/Masterminds/semver/tree/v3.4.0)
- github.com/gkampitakis/ciinfo: [v0.3.2](https://github.com/gkampitakis/ciinfo/tree/v0.3.2)
- github.com/gkampitakis/go-diff: [v1.3.2](https://github.com/gkampitakis/go-diff/tree/v1.3.2)
- github.com/gkampitakis/go-snaps: [v0.5.15](https://github.com/gkampitakis/go-snaps/tree/v0.5.15)
- github.com/goccy/go-yaml: [v1.18.0](https://github.com/goccy/go-yaml/tree/v1.18.0)
- github.com/joshdk/go-junit: [v1.0.0](https://github.com/joshdk/go-junit/tree/v1.0.0)
- github.com/maruel/natural: [v1.1.1](https://github.com/maruel/natural/tree/v1.1.1)
- github.com/mfridman/tparse: [v0.18.0](https://github.com/mfridman/tparse/tree/v0.18.0)
- github.com/tidwall/gjson: [v1.18.0](https://github.com/tidwall/gjson/tree/v1.18.0)
- github.com/tidwall/match: [v1.1.1](https://github.com/tidwall/match/tree/v1.1.1)
- github.com/tidwall/pretty: [v1.2.1](https://github.com/tidwall/pretty/tree/v1.2.1)
- github.com/tidwall/sjson: [v1.2.5](https://github.com/tidwall/sjson/tree/v1.2.5)
- go.uber.org/automaxprocs: v1.6.0

### Changed
- github.com/google/pprof: [d1b30fe → 27863c8](https://github.com/google/pprof/compare/d1b30fe...27863c8)
- github.com/onsi/ginkgo/v2: [v2.21.0 → v2.27.2](https://github.com/onsi/ginkgo/compare/v2.21.0...v2.27.2)
- github.com/onsi/gomega: [v1.35.1 → v1.38.2](https://github.com/onsi/gomega/compare/v1.35.1...v1.38.2)
- github.com/rogpeppe/go-internal: [v1.13.1 → v1.14.1](https://github.com/rogpeppe/go-internal/compare/v1.13.1...v1.14.1)
- go.etcd.io/bbolt: v1.4.2 → v1.4.3
- go.etcd.io/etcd/api/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/client/pkg/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/client/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/pkg/v3: v3.6.4 → v3.6.5
- go.etcd.io/etcd/server/v3: v3.6.4 → v3.6.5
- go.yaml.in/yaml/v2: v2.4.2 → v2.4.3
- golang.org/x/mod: v0.27.0 → v0.28.0
- golang.org/x/sync: v0.16.0 → v0.17.0
- golang.org/x/sys: v0.35.0 → v0.37.0
- golang.org/x/term: v0.34.0 → v0.36.0
- golang.org/x/text: v0.28.0 → v0.29.0
- k8s.io/system-validators: v1.11.1 → v1.12.1
- k8s.io/utils: 4c0f3b2 → bc988d5

### Removed
_Nothing has changed._



# v1.35.0-alpha.2


## Downloads for v1.35.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes.tar.gz) | acba342356249738a81bf6bc6de95e4a30097fdd0ebe956b8cd8a2b0715e3161930f7408bd3b1ca1e05c07de4359485cf887b278987366efef3caf9024e80c6d
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-src.tar.gz) | 6e9f58180f53e57ae6b462d4ab3a13f7cafc9bb9802f8af3254e9f3c78b9883103972dced5dd0796c9c8e4176fd8557754981a63fc4b5eb4fb0d07838027ac70

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | cb54b9aa876b327915048fa3d9a152abcde442d60cee750566339335b19c668f1d440f1dd79409137e7ee5d7e32e2d3c6e8b3fcaf7f4932b19508b483e3d4172
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 600f2922a818c9c750269695b9158892fcfdd1dd1311701033f93b396689c7d4625c24880598ea36ca3d1ff76be53dcdff911a96d8f337ec93847e340639a92b
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-linux-386.tar.gz) | bcc1d2c3b5577b22636b7c9aa515fb9944e586d5ae657e066e204388992bb1e9c94dd54ecc7feaaafe46c89943e5500366a26dac11ee2eb32ea3106daf1da51b
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 4250b2063c70cd69b49a50a4a416a9bd5a4e7734ed8b9ccc1081ed12e23c30018c2be9dc377100eb14823bab26aa33670e92d7ba38588a2a0ca011c3d63ecbf5
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 203825398afd6c697ac2fc13b126d7419b1c108362e6bb8a27eddef57e2845dd02735e4c48a5c2aa813f9e0ce24ee97ae94360cf50a9197fba53ba3ac736a50e
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | b6669df8c4e096d7ca435bfa481823b74e131907433fc7b7dbf6e6a699f2905a60c98e3c23c9321462ae3afdd707ffea3acf473a13905e63203cebefa80028c2
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 37b0ce0d3dfa8dcd2222c63b6572e32ad1a7f07d4164de886b3eca04d4c655a3cc07786090eb24cc20f0bf641cae2efba7ab3c3cd2da5536575571db31aa89da
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 3d7054c4b8d18501b535b0cd070bab316b7393bcb575fc869e2fde7190044b15a42e32dbea6aee64aef933ec1d8c7c11581c61bcb4829e710b26971b133180c3
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-windows-386.tar.gz) | ae011d1aa7b41160d50b9cd9bc4fe2890bbc2ce2f2b6c63695ae20f36e93cbf189c32deafc0d99c46532917ba291f40965cd4038edcb5bb3a27cd66974dba539
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | bcff5d410cea98ab7e83a66c545b4322d17055ed0b3c7acb110a757e6f0ee55aadfc0174c8c641511ac832024af5b2660f4e2be5c3076a12e0b862aa55a1d02f
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 4fb60b2747b500f1139f590e436318fabdd692fc7d2de27be9667c1e5f9af3a6a67796fcd3c69b92e225a04ae92292d715c4c5e1a1437f1723e5bd16d30e5c59

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | b29aaf01ad35edf7d24ac2a1d493c28a65941fd9f490bbcaeecfc418b1e26060f90e1677353ace6229ae1b8416f5080e116fcfb90732a7aab094761d9f1dadbe
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | fc8aeaec77c22d1cb777d9d626f6cbdc0bd178a29d1c125305592d4b40680c51d30d6ffeeb5754abd029c56ed2f49462a85799939f7ffc343f4816da3b9a2d20
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 61a70fa842e8afff5bdf3ab45a85b0bae183eb0e3910c440ca21520d3f03e0ee66ffbdd8b335b0d9ccfd2c77fb1a82e0f1480a267da6e6c10255c87465b12965
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 9275068613ddfaa163bbfaed5ae0c69dc2ca2031b3f42f990ed42995b14e7d5ed1bd5d49d3c3b7e95f7024a4434cb3518e05240b82d508f2be2c6d4971d3ab43

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 2200656cfe27817ec8bfc67564fba75c0afb582c75b2ce37734dff1c2757d142d45a24695c7e898b4663362f4058ca0ae8399ee485883833498cac9867caccdc
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | fe087958b7b7b7197132473508deffa90a740fffc2bf7a06c9a7c7df029394fd27a307efc6bb8003c6f95d9013f57ed577ec4a777881c44acba26a1ebc918ae5
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 621a7d6f7f3fcc382922f0912a5dd3f9587ec15992c65be806a18e4b3254895d42dc78ac2b1aab10a16dee1227ca315c1b5b35c27b29946c1337d548b799ddc7
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 16a9074fba6db7ef45b38ed5ea05ae9cd47a6388b01cfa551377de5bc1b720df3507b8bda974c1220d8a82394902192a4013754026f2d71732bb480743862c05
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 6a893a6a4ad7f664fea7536f22bd27d4f874e7568658f5863e156c290b4afee2ef566797c6e3dae86b4d706f219b8169da5e03ef61e565b7a4ba2123a6b43c5c

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0-alpha.1

## Changes by Kind

### Deprecation

- FailCgroupV1 will be set to true from 1.35. 
  This means that nodes will not start on a cgroup v1 in our default behavior. 
  This is putting cgroup v1 into a deprecated state. ([#134298](https://github.com/kubernetes/kubernetes/pull/134298), [@kannon92](https://github.com/kannon92)) [SIG Node]
- Mark ipvs mode in kube-proxy as deprecated. ipvs mode in kube-proxy is deprecated and will be removed in a future version of Kubernetes. Users are encouraged to move to nftables. ([#134539](https://github.com/kubernetes/kubernetes/pull/134539), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network]

### API Change

- Kube-apiserver: fix a possible panic validating a custom resource whose CustomResourceDefinition indicates a status subresource exists, but which does not define a `status` property in the `openAPIV3Schema` ([#133721](https://github.com/kubernetes/kubernetes/pull/133721), [@fusida](https://github.com/fusida)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Kubernetes API Go types removed runtime use of the github.com/gogo/protobuf library, and are no longer registered into the global gogo type registry. Kubernetes API Go types were not suitable for use with the google.golang.org/protobuf library, and no longer implement `ProtoMessage()` by default to avoid accidental incompatible use. If removal of these marker methods impacts your use, it can be re-enabled for one more release with a `kubernetes_protomessage_one_more_release` build tag, but will be removed in 1.36. ([#134256](https://github.com/kubernetes/kubernetes/pull/134256), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- Promoted HPA configurable tolerance to beta. The `HPAConfigurableTolerance` feature gate is now enabled by default. ([#133128](https://github.com/kubernetes/kubernetes/pull/133128), [@jm-franc](https://github.com/jm-franc)) [SIG API Machinery and Autoscaling]
- The MaxUnavailableStatefulSet feature is now beta and enabled by default. ([#133153](https://github.com/kubernetes/kubernetes/pull/133153), [@helayoty](https://github.com/helayoty)) [SIG API Machinery and Apps]

### Feature

- Enable the feature gate `ContainerRestartRules` by default. The ContainerRestartRules feature is promoted to beta. Fixing a bug in this feature that caused probes continue to run even if the container has terminated and is not restartable. ([#134631](https://github.com/kubernetes/kubernetes/pull/134631), [@yuanwang04](https://github.com/yuanwang04)) [SIG Node]
- Kube-apiserver: the subresources `pods/exec`, `pods/attach`, and `pods/portforward` now require `create` permission for both SPDY and Websocket API requests. Previously, SPDY requests required `create` permission, but Websocket requests only required `get` permission. This change is gated by the `AuthorizePodWebsocketUpgradeCreatePermission` feature-gate, which is enabled by default.
  
  Before upgrading to 1.35, ensure any custom ClusterRoles and Roles intended to grant `pods/exec`, `pods/attach`, or `pods/portforward` permission include the `create` verb. ([#134577](https://github.com/kubernetes/kubernetes/pull/134577), [@seans3](https://github.com/seans3)) [SIG API Machinery, Auth, Node and Testing]
- Kubeadm: print the errors during retires related to the WaitForAllControlPlaneComponents functionality at verbosity level 5. ([#134433](https://github.com/kubernetes/kubernetes/pull/134433), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes is now built using Go 1.25.3 ([#134611](https://github.com/kubernetes/kubernetes/pull/134611), [@cpanato](https://github.com/cpanato)) [SIG Architecture, Cloud Provider, Etcd, Release, Storage and Testing]
- Locked the (generally available) feature gate `ExecProbeTimeout` to true. ([#134635](https://github.com/kubernetes/kubernetes/pull/134635), [@vivzbansal](https://github.com/vivzbansal)) [SIG Node and Testing]
- Promoted the `HostnameOverride` feature gate to beta and is enabled by default. ([#134729](https://github.com/kubernetes/kubernetes/pull/134729), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Network and Node]

### Documentation

- Kubectl describe, get, drain and events have the ability to set chunk-size using --chunk-size flag, which is now officially stable. ([#134481](https://github.com/kubernetes/kubernetes/pull/134481), [@soltysh](https://github.com/soltysh)) [SIG CLI]

### Bug or Regression

- DRA API: the "tolerations" field in exact and sub requests now gets dropped properly when the DRADeviceTaints API is disabled. ([#132927](https://github.com/kubernetes/kubernetes/pull/132927), [@pohly](https://github.com/pohly))
- DRA Device Taints: tolerating a NoExecute did not work because the scheduler did not inform the eviction controller about the toleration, so the scheduled pod got evicted almost immediately. ([#134479](https://github.com/kubernetes/kubernetes/pull/134479), [@pohly](https://github.com/pohly)) [SIG Apps, Node, Scheduling and Testing]
- Endpoints/endpointslice controllers perform much better when there are a large number of services in a single namespace ([#134739](https://github.com/kubernetes/kubernetes/pull/134739), [@shyamjvs](https://github.com/shyamjvs)) [SIG Apps and Network]
- Fixed a bug that prevents schedule next pod when using DRAConsumableCapacity feature. (#133705, @sunya-ch) ([#133706](https://github.com/kubernetes/kubernetes/pull/133706), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node]
- Fixed a bug where 64 bit IPv6 ServiceCIDRs allocated addresses outside the subnet range. ([#134193](https://github.com/kubernetes/kubernetes/pull/134193), [@hoskeri](https://github.com/hoskeri)) [SIG Network]
- Fixed a startup probe race condition that caused main containers to remain stuck in "Initializing" state when sidecar containers with startup probes failed initially but succeeded on restart in pods with restartPolicy=Never. ([#133072](https://github.com/kubernetes/kubernetes/pull/133072), [@AadiDev005](https://github.com/AadiDev005)) [SIG Node and Testing]
- Kube-apiserver: when --requestheader-client-ca-file and --client-ca-file contain overlapping certificates, --requestheader-allowed-names must be specified to ensure regular client certificates cannot set authenticating proxy headers for arbitrary users ([#131411](https://github.com/kubernetes/kubernetes/pull/131411), [@ballista01](https://github.com/ballista01)) [SIG API Machinery, Auth and Security]
- Kube-controller-manager: Resolves potential issues handling pods with incorrect uids in their ownerReference ([#134654](https://github.com/kubernetes/kubernetes/pull/134654), [@liggitt](https://github.com/liggitt)) [SIG Apps]
- Kubeadm: avoid panicing if the user has malformed the kubeconfig in the cluster-info config map to not include a valid current context. Include proper validation at the appropriate locations and throw errors instead. ([#134715](https://github.com/kubernetes/kubernetes/pull/134715), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fixes a preflight check that can fail hostname construction in IPV6 setups ([#134588](https://github.com/kubernetes/kubernetes/pull/134588), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, Cloud Provider, Cluster Lifecycle and Testing]
- Legacy watch calls (RV = 0 or unset) that generate init-events weigh higher in APF seat usage now. Properly accounting for their cost protects the API server from CPU overload. Users might see increased throttling of such calls as a result. ([#134601](https://github.com/kubernetes/kubernetes/pull/134601), [@shyamjvs](https://github.com/shyamjvs)) [SIG API Machinery]
- Prevent a segfault occurring when updating deeply nested JSON fields ([#134381](https://github.com/kubernetes/kubernetes/pull/134381), [@kon-angelo](https://github.com/kon-angelo)) [SIG API Machinery and CLI]
- The kubelet now honors the configuration userNamespaces.idsPerPod. Before it was ignored. ([#133373](https://github.com/kubernetes/kubernetes/pull/133373), [@AkihiroSuda](https://github.com/AkihiroSuda)) [SIG Node and Testing]

### Other (Cleanup or Flake)

- Building Kubernetes is now implemented by running a pre-built container image directly, without running rsyncd, and is substantially simplified. ([#134510](https://github.com/kubernetes/kubernetes/pull/134510), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release and Testing]
- CPU Manager static policy option `strict-cpu-reservation` moved to the GA version ([#134388](https://github.com/kubernetes/kubernetes/pull/134388), [@psasnal](https://github.com/psasnal)) [SIG Node]
- Dropped support for policy/v1beta1 PodDisruptionBudget in kubectl ([#134685](https://github.com/kubernetes/kubernetes/pull/134685), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Kubeadm: stoped applying the --pod-infra-container-image flag for the kubelet. The flag has been deprecated and no longer served a purpose in the kubelet as the logic was migrated to CRI. During upgrade, kubeadm will attempt to remove the flag from the file /var/lib/kubelet/kubeadm-flags.env. ([#133778](https://github.com/kubernetes/kubernetes/pull/133778), [@carlory](https://github.com/carlory)) [SIG Cloud Provider and Cluster Lifecycle]
- Kubeadm: updated the supported etcd version to v3.5.23 for supported control plane versions v1.31, v1.32, and v1.33. ([#134692](https://github.com/kubernetes/kubernetes/pull/134692), [@joshjms](https://github.com/joshjms)) [SIG Cluster Lifecycle and Etcd]
- Kubeadm: updated the supported etcd version to v3.5.24 for supported control plane versions v1.32, v1.33, and v1.34. ([#134779](https://github.com/kubernetes/kubernetes/pull/134779), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Kubernetes is now built with go 1.25.3 ([#134598](https://github.com/kubernetes/kubernetes/pull/134598), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Promote the Topology Manager policy option max-allowable-numa-nodes to GA ([#134614](https://github.com/kubernetes/kubernetes/pull/134614), [@ffromani](https://github.com/ffromani)) [SIG Node]
- Rsync is no longer required to build kubernetes. ([#134656](https://github.com/kubernetes/kubernetes/pull/134656), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release and Testing]
- The storage.k8s.io/v1alpha1 VolumeAttributesClass API is no longer served in 1.35 ([#134625](https://github.com/kubernetes/kubernetes/pull/134625), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Etcd, Storage and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.35.0-alpha.1


## Downloads for v1.35.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes.tar.gz) | 1d6fb6a4c7f82fe04e56757b733c3fc4aac652f8c2113e79ddce83b6cbe0179404147b35ddbc18e1b60eb802acb3f6d884599fd573f3d16f0558ef7ddfb8aae2
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-src.tar.gz) | 364788bac4d405ac6180fe3cb7e3d847e7960fcb0532146b105270aeac2624ade2ff87370c5aa8f768eda07fd28e5e75f73afbdf9cc1b786827a0e123bdea561

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 1ba40849b104851d922bce32dc9306004e9b95cfadeff9ecfb65f779892009f9a70878b8efe96159088b1ad8c700bf19e58d68416dfdff7853660e6074dd3752
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 0c939895ad2d53f57e9137774eed99cbfbfa5f15d4276f5f55c4ea40b922a5f37ab375a065fdd330f5a1ddf452896f2a13621075b049f382ab42a65ea1085dac
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-386.tar.gz) | e98a9b2d5f1c8bec552be6353b623a8f12078befd968a662a933907d0ff72b0164fa8b38e4cbd4aae6191e28aedc19d996ec99298053d7bafc458f91580a7cfa
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 34ab1e9edf70c84fe58a223e91b0bf679e5d1273a2b6503a18a61a4bea79231948efe098f84e39f83dbfa2c5271aad8e9819aac104d6c3360c97e5c348b15be7
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 554f8240597e7eb8470c8e2b4bca33c06a0a91746831ef93f76b344f5c3d6226d4ef26cb59f127d1436ac091b7b79395320fe8a9b2acc512afd601989c138d4a
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 62c750654e898622aa87b2d81d4b0cbaf36614899f37181c2e3a6aa645d2270c4dedb7d6e7b974059a716ad144a5615407e6a9a4c03f761a512d37fdda796e50
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | bbe22979c4e300675dfa955ce9855b7b33b29119a9f78e58bc1b088dba2b8dedbf0b068092d42cd98dbc10cda1da317eae6414b91587593678ec76780ff575df
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | b8360d0bf930149d360da2a95549b35cb7e14932ae8507d99e34d93729ae645bab203dfba325c74db13204e09e5ee032f887cdd67badfbf3bc8a08d71ccf9c3d
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-windows-386.tar.gz) | cc935a74f30dcd1eaaeadc8f2353a9742ebc4a36b133342c6402b065750f4028a1a392bd5f7ea51533c2d799ff2bdb3d0f21493e7fabacb27289e58f011bb229
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 119742103985be0cd4296b85aa2c713cdc510b9a9412706fdf88ca1c703f69338146efc5cf37168ab56e74576ad561ce37c3f500d29b63002139d11544b1b7cf
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 67f5fe6b14aa4c49acb6f706ec4a9e43b87f1e19555579895452183e0b2d2be2202f8d48622208ac5ef6e0fb9050d99bb7e1ed9e4e31e8fcac7c0b5e44787c39

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | a1c935db625766b02113087068fa1087a6b74e9f57ad72cc1d5d85e830c0569b9257746013053ee8dc89404940458c3bba00064666978ddff4df9f3cae0ae066
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 5f87f8f46719af413fa864b7d91d5b95fa86adf27df63cf904470b15e844ebda4c802d7ec7cb4006b4e5a3780903d0436eb57a7e2f2b79b74cf5e7e8f65496b1
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 9f2e9476ae3b95919c991dd0438b31eeded7c7f686948ef2d6227311dabc952e74f61d3638f224eb18e63a4fe4955b55a9e358032477b989800541617a8b5f6a
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 043311de9dad81d3774decc3aeeac48833d2d234a15ce4a2062fd9af778879afde5e2eb9fdec5f3641725f630e7c3bd845a348ad713b54275e77293941d4c8d0

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 865b4ee818cd53bc91001f658243e6b6fd9464f17ef8dc0cc739586689f39998e5c47630df439ad43553d6830ae3a7375cc780c3a5e49f1422d35d77194efc35
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | fc7df94bc328817d20c59e1ab1371634bf3849141ad982c9d403b136d009c3ed9ee3f7a20659a0f93af7179e13e2c0835b80fefc677b8840f7bcbde0dabb4483
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 8ec718436680766d9026b56ece5bff7a6bac9f63da0edf33843a7a6c255e1a1d22aecf260c1d5fc394c1e2f581931c5ae0acad80a0141fc8d4d7730bf04566b5
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | fff0562e3a89b4f9444ce83bb8cfc860bcd5178a2c1ba0f1404d87556f483d48241aa3f927f2534d811b1c554958c167f1e5199fa916ebfd9a754cef2f761139
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 7774299a2b581a4ab2514d8dc95c1d0dff0652aa0518efbffa9ac39f5c510cb83d8a41a70e9b4e78f0b179e5a806402310dac161ff3a2398b97755938c225586

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.34.0

## Changes by Kind

### API Change

- Added WithOrigin within apis/core/validation with adjusted tests ([#132825](https://github.com/kubernetes/kubernetes/pull/132825), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Apps]
- Component-base: validate that log-flush-frequency is positive and return an error instead of panic-ing ([#133540](https://github.com/kubernetes/kubernetes/pull/133540), [@BenTheElder](https://github.com/BenTheElder)) [SIG Architecture, Instrumentation, Network and Node]
- Feature gate dependencies are now explicit, and validated at startup. A feature can no longer be enabled if it depends on a disabled feature. In particular, this means that `AllAlpha=true` will no longer work without enabling disabled-by-default beta features that are depended on (either with `AllBeta=true` or explicitly enumerating the disabled dependencies). ([#133697](https://github.com/kubernetes/kubernetes/pull/133697), [@tallclair](https://github.com/tallclair)) [SIG API Machinery, Architecture, Cluster Lifecycle and Node]
- In version 1.34, the PodObservedGenerationTracking feature has been upgraded to beta, and the description of the alpha version in the openapi has been removed. ([#133883](https://github.com/kubernetes/kubernetes/pull/133883), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Apps]
- Introduce a new declarative validation tag +k8s:customUnique to control listmap uniqueness ([#134279](https://github.com/kubernetes/kubernetes/pull/134279), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Auth]
- Kube-apiserver: Fixed a 1.34 regression in CustomResourceDefinition handling that incorrectly warned about unrecognized formats on number and integer properties ([#133896](https://github.com/kubernetes/kubernetes/pull/133896), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Contributor Experience, Network, Node and Scheduling]
- OpenAPI model packages of API types are generated into `zz_generated.model_name.go` files and are accessible using the `OpenAPIModelName()` function.  This allows API authors to declare the desired OpenAPI model packages instead of using the go package path of API types. ([#131755](https://github.com/kubernetes/kubernetes/pull/131755), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Support for `kubectl get -o kyaml` is now on by default.  To disable it, set `KUBECTL_KYAML=false`. ([#133327](https://github.com/kubernetes/kubernetes/pull/133327), [@thockin](https://github.com/thockin)) [SIG CLI]
- The storage version for MutatingAdmissionPolicy is updated to v1beta1. ([#133715](https://github.com/kubernetes/kubernetes/pull/133715), [@cici37](https://github.com/cici37)) [SIG API Machinery, Etcd and Testing]

### Feature

- Add paths section to kubelet statusz endpoint ([#133239](https://github.com/kubernetes/kubernetes/pull/133239), [@Peac36](https://github.com/Peac36)) [SIG Node]
- Add paths section to scheduler statusz endpoint ([#132606](https://github.com/kubernetes/kubernetes/pull/132606), [@Peac36](https://github.com/Peac36)) [SIG API Machinery, Architecture, Instrumentation, Network, Node, Scheduling and Testing]
- Added kubectl config set-context -n flag as a shorthand for --namespace ([#134384](https://github.com/kubernetes/kubernetes/pull/134384), [@tchap](https://github.com/tchap)) [SIG CLI and Testing]
- Added remote runtime and image `Close()` method to be able to close the connection. ([#133211](https://github.com/kubernetes/kubernetes/pull/133211), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Adds metric for Maxunavailable feature ([#130951](https://github.com/kubernetes/kubernetes/pull/130951), [@Edwinhr716](https://github.com/Edwinhr716)) [SIG Apps and Instrumentation]
- Applyconfiguration-gen now generates extract functions for all subresources ([#132665](https://github.com/kubernetes/kubernetes/pull/132665), [@mrIncompetent](https://github.com/mrIncompetent)) [SIG API Machinery]
- Applyconfiguration-gen now preserves struct and field comments from source types in generated code ([#132663](https://github.com/kubernetes/kubernetes/pull/132663), [@mrIncompetent](https://github.com/mrIncompetent)) [SIG API Machinery]
- DRA: the resource.k8s.io API now uses the v1 API version (introduced in 1.34) as default storage version. Downgrading to 1.33 is not supported. ([#133876](https://github.com/kubernetes/kubernetes/pull/133876), [@kei01234kei](https://github.com/kei01234kei)) [SIG API Machinery, Etcd and Testing]
- Events:
    Type     Reason   Age                 From               Message
    ----     ------   ----                ----               -------
    Warning  Failed   7m11s (x2 over 7m33s) kubelet          spec.containers{nginx}: Failed to pull image "nginx": failed to pull and unpack image... ([#133627](https://github.com/kubernetes/kubernetes/pull/133627), [@itzPranshul](https://github.com/itzPranshul)) [SIG CLI]
- Introduces e2e tests that check component invariant metrics across the entire suite run. ([#133394](https://github.com/kubernetes/kubernetes/pull/133394), [@BenTheElder](https://github.com/BenTheElder)) [SIG Testing]
- K8s.io/apimachinery: Introduce a helper function to compare resourceVersion strings from two objects of the same resource ([#134330](https://github.com/kubernetes/kubernetes/pull/134330), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Kubeadm: graduate the kubeadm specific feature gate ControlPlaneKubeletLocalMode to GA and lock it to enabled by default. To opt-out manually from this desired default behavior you must patch the "server" field in the  /etc/kubernetes/kubelet.conf file. The subphase of "kubeadm join phase control-plane-join" called "etcd" is now deprecated, hidden and replaced by the subphase with identical functionality "etcd-join". "etcd" will be removed in a follow-up release. The subphase "kubelet-wait-bootstrap" of "kubeadm join" is no longer experimental and will always run. ([#134106](https://github.com/kubernetes/kubernetes/pull/134106), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes is now built using Go 1.25.1 ([#134095](https://github.com/kubernetes/kubernetes/pull/134095), [@dims](https://github.com/dims)) [SIG Release and Testing]
- Kubernetes now uses Go Language Version 1.25, including https://go.dev/blog/container-aware-gomaxprocs ([#134120](https://github.com/kubernetes/kubernetes/pull/134120), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling and Storage]
- Lock down the `AllowOverwriteTerminationGracePeriodSeconds` feature gate. ([#133792](https://github.com/kubernetes/kubernetes/pull/133792), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Metrics: exclude dryRun requests from apiserver_request_sli_duration_seconds ([#131092](https://github.com/kubernetes/kubernetes/pull/131092), [@aldudko](https://github.com/aldudko)) [SIG API Machinery and Instrumentation]
- The validation in the resouce.k8s.io has been migrated to declarative validation.
  If the `DeclarativeValidation` feature gate is enabled, mismatches with existing validation are reported via metrics.
  If the `DeclarativeValidationTakeover` feature gate is enabled, declarative validation is the primary source of errors for migrated fields. ([#134072](https://github.com/kubernetes/kubernetes/pull/134072), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps and Auth]

### Bug or Regression

- Added the correct error when eviction is blocked due to the failSafe mechanism of the DisruptionController. ([#133097](https://github.com/kubernetes/kubernetes/pull/133097), [@kei01234kei](https://github.com/kei01234kei)) [SIG Apps and Node]
- Bugfix: the default serviceCIDR controller was not logging events because the event broadcaster was shutdown during its initialization. ([#133338](https://github.com/kubernetes/kubernetes/pull/133338), [@aojea](https://github.com/aojea)) [SIG Network]
- Deprecated metrics will be hidden as per the metrics deprecation policy https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecating-a-metric ([#133436](https://github.com/kubernetes/kubernetes/pull/133436), [@richabanker](https://github.com/richabanker)) [SIG Architecture, Instrumentation and Network]
- Fix incorrect behavior of preemptor pod when preemption of the victim takes long to complete. The preemptor pod should not be circling in scheduling cycles until preemption is finished. ([#134294](https://github.com/kubernetes/kubernetes/pull/134294), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Scheduling and Testing]
- Fix missing kubelet_volume_stats_* metrics ([#133890](https://github.com/kubernetes/kubernetes/pull/133890), [@huww98](https://github.com/huww98)) [SIG Instrumentation and Node]
- Fix occasional schedule delay when the static PV is created ([#133929](https://github.com/kubernetes/kubernetes/pull/133929), [@huww98](https://github.com/huww98)) [SIG Scheduling and Storage]
- Fix resource claims deallocation for extended resource when pod is completed ([#134312](https://github.com/kubernetes/kubernetes/pull/134312), [@alaypatel07](https://github.com/alaypatel07)) [SIG Apps, Node and Testing]
- Fixed SELinux warning controller not emitting events on some SELinux label conflicts. ([#133425](https://github.com/kubernetes/kubernetes/pull/133425), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Storage and Testing]
- Fixed a bug in kube-proxy nftables mode (GA as of 1.33) that fails to determine if traffic originates from a local source on the node. The issue was caused by using the wrong meta `iif` instead of `iifname` for name based matches. ([#134024](https://github.com/kubernetes/kubernetes/pull/134024), [@jack4it](https://github.com/jack4it)) [SIG Network]
- Fixed a bug in kube-scheduler where pending pod preemption caused preemptor pods to be retried more frequently. ([#134245](https://github.com/kubernetes/kubernetes/pull/134245), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- Fixed a bug that caused apiservers to send an inappropriate Content-Type request header to authorization, token authentication, imagepolicy admission, and audit webhooks when the alpha client-go feature gate "ClientsPreferCBOR" is enabled. ([#132960](https://github.com/kubernetes/kubernetes/pull/132960), [@benluddy](https://github.com/benluddy)) [SIG API Machinery and Node]
- Fixed a bug that caused duplicate validation when updating PersistentVolumeClaims, VolumeAttachments and VolumeAttributesClasses. ([#132549](https://github.com/kubernetes/kubernetes/pull/132549), [@gavinkflam](https://github.com/gavinkflam)) [SIG Storage]
- Fixed a bug that caused duplicate validation when updating role and role binding resources. ([#132550](https://github.com/kubernetes/kubernetes/pull/132550), [@gavinkflam](https://github.com/gavinkflam)) [SIG Auth]
- Fixed a bug where high latency kube-apiserver caused scheduling throughput degradation. ([#134154](https://github.com/kubernetes/kubernetes/pull/134154), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Fixed broken shell completion for api resources. ([#133771](https://github.com/kubernetes/kubernetes/pull/133771), [@marckhouzam](https://github.com/marckhouzam)) [SIG CLI]
- Fixed validation error when ConfigFlags has CertFile and (or) KeyFile and original config also contains CertFileData and (or) KeyFileData. ([#133917](https://github.com/kubernetes/kubernetes/pull/133917), [@n2h9](https://github.com/n2h9)) [SIG API Machinery and CLI]
- Fixes a possible data race during metrics registration ([#134390](https://github.com/kubernetes/kubernetes/pull/134390), [@liggitt](https://github.com/liggitt)) [SIG Architecture and Instrumentation]
- Implicit extended resource name derived from device class (deviceclass.resource.kubernetes.io/<device-class-name>) can be used to request DRA devices matching the device class. ([#133363](https://github.com/kubernetes/kubernetes/pull/133363), [@yliaog](https://github.com/yliaog)) [SIG Node, Scheduling and Testing]
- Kube-apiserver: Fixes a 1.34 regression with spurious "Error getting keys" log messages ([#133817](https://github.com/kubernetes/kubernetes/pull/133817), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- Kube-apiserver: Fixes a possible 1.34 performance regression calculating object size statistics for resources not served from the watch cache, typically only Events ([#133873](https://github.com/kubernetes/kubernetes/pull/133873), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- Kube-apiserver: improve the validation error message shown for custom resources with CEL validation rules to include the value that failed validation ([#132798](https://github.com/kubernetes/kubernetes/pull/132798), [@cbandy](https://github.com/cbandy)) [SIG API Machinery]
- Kube-controller-manager: Fixes a possible data race in the garbage collection controller ([#134379](https://github.com/kubernetes/kubernetes/pull/134379), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Apps]
- Kubeadm: ensured waiting for apiserver uses a local client that doesn't reach to the control plane endpoint and instead reaches directly to the local API server endpoint. ([#134265](https://github.com/kubernetes/kubernetes/pull/134265), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix KUBEADM_UPGRADE_DRYRUN_DIR not honored in upgrade phase when writing kubelet config files ([#134007](https://github.com/kubernetes/kubernetes/pull/134007), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: fixed a bug where the node registration information for a given node was not fetched correctly during "kubeadm upgrade node" and the node name can end up being incorrect in cases where the node name is not the same as the host name. ([#134319](https://github.com/kubernetes/kubernetes/pull/134319), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fixed bug where v1beta3's ClusterConfiguration.APIServer.TimeoutForControlPlane is not respected in newer versions of kubeadm where v1beta4 is the default. ([#133513](https://github.com/kubernetes/kubernetes/pull/133513), [@tom1299](https://github.com/tom1299)) [SIG Cluster Lifecycle]
- Kubelet: the connection to a DRA driver became unusable because of an internal deadlock when a connection was idle for 30 minutes. ([#133926](https://github.com/kubernetes/kubernetes/pull/133926), [@pohly](https://github.com/pohly)) [SIG Node]
- Pod can have multiple volumes reference the same PVC ([#122140](https://github.com/kubernetes/kubernetes/pull/122140), [@huww98](https://github.com/huww98)) [SIG Node, Storage and Testing]
- Previously, `kubectl scale` returned the error message `error: no objects passed to scale <GroupResource> "<ResourceName>" not found` when the specified resource did not exist. 
  For consistency with other commands(e.g. `kubectl get`), it has been changed to just return `Error from server (NotFound): <GroupResource> "<ResourceName>" not found`. ([#134017](https://github.com/kubernetes/kubernetes/pull/134017), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI]
- Promote VAC API test to conformance ([#133615](https://github.com/kubernetes/kubernetes/pull/133615), [@carlory](https://github.com/carlory)) [SIG Architecture, Storage and Testing]
- Remove incorrectly printed warning for SessionAffinity whenever a headless service is creater or updated ([#134054](https://github.com/kubernetes/kubernetes/pull/134054), [@Peac36](https://github.com/Peac36)) [SIG Network]
- The SchedulerAsyncAPICalls feature gate has been disabled to mitigate a bug where its interaction with asynchronous preemption in could degrade kube-scheduler performance, particularly under high kube-apiserver load. ([#134400](https://github.com/kubernetes/kubernetes/pull/134400), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- When image garbage collection is unable to free enough disk space, the FreeDiskSpaceFailed warning event is now more actionable. Example: `Insufficient free disk space on the node's image filesystem (95.0% of 10.0 GiB used). Failed to free sufficient space by deleting unused images. Consider resizing the disk or deleting unused files.` ([#132578](https://github.com/kubernetes/kubernetes/pull/132578), [@drigz](https://github.com/drigz)) [SIG Node]

### Other (Cleanup or Flake)

- Bump addon manager to use kubectl v1.32.2 ([#130548](https://github.com/kubernetes/kubernetes/pull/130548), [@Jefftree](https://github.com/Jefftree)) [SIG Cloud Provider, Scalability and Testing]
- Dropping the experimental prefix from kubectl wait command's short description, since kubectl wait command has been stable for a long time. ([#133907](https://github.com/kubernetes/kubernetes/pull/133907), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Fix formatting of assorted go API deprecations for godoc / pkgsite and enable a linter to help catch mis-formatted deprecations ([#133571](https://github.com/kubernetes/kubernetes/pull/133571), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, Architecture, CLI, Instrumentation and Testing]
- Improved HPA performance when using container-specific resource metrics by optimizing container lookup logic to exit early once the target container is found, reducing unnecessary iterations through all containers in a pod. ([#133415](https://github.com/kubernetes/kubernetes/pull/133415), [@AadiDev005](https://github.com/AadiDev005)) [SIG Apps and Autoscaling]
- Kube-apiserver: Fixes an issue where passing invalid DeleteOptions incorrectly returned status 500 rather than 400. ([#133358](https://github.com/kubernetes/kubernetes/pull/133358), [@ostrain](https://github.com/ostrain)) [SIG API Machinery]
- Kubeadm: removed the `RootlessControlPlane` feature gate. User Namespaces will serve as its replacement. ([#134178](https://github.com/kubernetes/kubernetes/pull/134178), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Remove container name from messages for container created and started events. ([#134043](https://github.com/kubernetes/kubernetes/pull/134043), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Removed deprecated gogo protocol definitions from `k8s.io/kubelet/pkg/apis/dra` in favor of `google.golang.org/protobuf`. ([#133026](https://github.com/kubernetes/kubernetes/pull/133026), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery and Node]
- Removed general available feature-gate SizeMemoryBackedVolumes ([#133720](https://github.com/kubernetes/kubernetes/pull/133720), [@carlory](https://github.com/carlory)) [SIG Node, Storage and Testing]
- Removed the `ComponentSLIs` feature gate, which had been promoted to stable as part of the Kubernetes 1.32 release. ([#133742](https://github.com/kubernetes/kubernetes/pull/133742), [@carlory](https://github.com/carlory)) [SIG Architecture and Instrumentation]
- Removing Experimental prefix from the description of kubectl wait to emphasize that it is stable. ([#133731](https://github.com/kubernetes/kubernetes/pull/133731), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Removing the KUBECTL_OPENAPIV3_PATCH environment variable entirely, since aggregated discovery has been stable from 1.30. ([#134130](https://github.com/kubernetes/kubernetes/pull/134130), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Specifies the deprecated version of apiserver_storage_objects metric in metrics docs ([#134028](https://github.com/kubernetes/kubernetes/pull/134028), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Etcd and Instrumentation]
- Tests: switch to https://go.dev/doc/go1.25#container-aware-gomaxprocs from go.uber.org/automaxprocs ([#133492](https://github.com/kubernetes/kubernetes/pull/133492), [@BenTheElder](https://github.com/BenTheElder)) [SIG Testing]
- The `/statusz` page for `kube-proxy` now includes a list of exposed endpoints, making it easier to debug and introspect. ([#133190](https://github.com/kubernetes/kubernetes/pull/133190), [@aman4433](https://github.com/aman4433)) [SIG Network and Node]
- Types in k/k/pkg/scheduler/framework:
  Handle,
  Plugin,
  PreEnqueuePlugin, QueueSortPlugin, EnqueueExtensions, PreFilterExtensions, PreFilterPlugin, FilterPlugin, PostFilterPlugin, PreScorePlugin, ScorePlugin, ReservePlugin, PreBindPlugin, PostBindPlugin, PermitPlugin, BindPlugin,
  PodActivator, PodNominator, PluginsRunner,
  LessFunc, ScoreExtensions, NodeToStatusReader, NodeScoreList, NodeScore, NodePluginScores, PluginScore, NominatingMode, NominatingInfo, WaitingPod, PreFilterResult, PostFilterResult,
  Extender,
  NodeInfoLister, StorageInfoLister, SharedLister, ResourceSliceLister, DeviceClassLister, ResourceClaimTracker, SharedDRAManager
  
  are moved to package k8s.io/kube-scheduler/framework . Users should update import paths. The interfaces don't change.
  
  Type Parallelizer in k/k/pkg/scheduler/framework/parallelism is split into interface Parallelizer (in k8s.io/kube-scheduler/framework) and struct Parallelizer (location unchanged in k/k). Plugin developers should update the import path to staging repo. ([#133172](https://github.com/kubernetes/kubernetes/pull/133172), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Node, Release, Scheduling, Storage and Testing]
- Updated CNI plugins to v1.8.0. ([#133837](https://github.com/kubernetes/kubernetes/pull/133837), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.34.0. ([#133636](https://github.com/kubernetes/kubernetes/pull/133636), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Updated etcd to v3.6.5. ([#134251](https://github.com/kubernetes/kubernetes/pull/134251), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Upgrade CoreDNS to v1.12.3 ([#132288](https://github.com/kubernetes/kubernetes/pull/132288), [@thevilledev](https://github.com/thevilledev)) [SIG Cloud Provider and Cluster Lifecycle]
- `kubectl auth reconcile` now re-attempts reconciliation if it encounters a conflict error ([#133323](https://github.com/kubernetes/kubernetes/pull/133323), [@liggitt](https://github.com/liggitt)) [SIG Auth and CLI]
- `kubectl get` and `kubectl describe` human-readable output no longer includes counts for referenced tokens and secrets ([#117160](https://github.com/kubernetes/kubernetes/pull/117160), [@liggitt](https://github.com/liggitt)) [SIG CLI and Testing]

## Dependencies

### Added
- github.com/moby/sys/atomicwriter: [v0.1.0](https://github.com/moby/sys/tree/atomicwriter/v0.1.0)
- golang.org/x/tools/go/expect: v0.1.1-deprecated
- golang.org/x/tools/go/packages/packagestest: v0.1.1-deprecated

### Changed
- cloud.google.com/go/compute/metadata: v0.6.0 → v0.7.0
- github.com/aws/aws-sdk-go-v2/config: [v1.27.24 → v1.29.14](https://github.com/aws/aws-sdk-go-v2/compare/config/v1.27.24...config/v1.29.14)
- github.com/aws/aws-sdk-go-v2/credentials: [v1.17.24 → v1.17.67](https://github.com/aws/aws-sdk-go-v2/compare/credentials/v1.17.24...credentials/v1.17.67)
- github.com/aws/aws-sdk-go-v2/feature/ec2/imds: [v1.16.9 → v1.16.30](https://github.com/aws/aws-sdk-go-v2/compare/feature/ec2/imds/v1.16.9...feature/ec2/imds/v1.16.30)
- github.com/aws/aws-sdk-go-v2/internal/configsources: [v1.3.13 → v1.3.34](https://github.com/aws/aws-sdk-go-v2/compare/internal/configsources/v1.3.13...internal/configsources/v1.3.34)
- github.com/aws/aws-sdk-go-v2/internal/endpoints/v2: [v2.6.13 → v2.6.34](https://github.com/aws/aws-sdk-go-v2/compare/internal/endpoints/v2/v2.6.13...internal/endpoints/v2/v2.6.34)
- github.com/aws/aws-sdk-go-v2/internal/ini: [v1.8.0 → v1.8.3](https://github.com/aws/aws-sdk-go-v2/compare/internal/ini/v1.8.0...internal/ini/v1.8.3)
- github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding: [v1.11.3 → v1.12.3](https://github.com/aws/aws-sdk-go-v2/compare/service/internal/accept-encoding/v1.11.3...service/internal/accept-encoding/v1.12.3)
- github.com/aws/aws-sdk-go-v2/service/internal/presigned-url: [v1.11.15 → v1.12.15](https://github.com/aws/aws-sdk-go-v2/compare/service/internal/presigned-url/v1.11.15...service/internal/presigned-url/v1.12.15)
- github.com/aws/aws-sdk-go-v2/service/sso: [v1.22.1 → v1.25.3](https://github.com/aws/aws-sdk-go-v2/compare/service/sso/v1.22.1...service/sso/v1.25.3)
- github.com/aws/aws-sdk-go-v2/service/ssooidc: [v1.26.2 → v1.30.1](https://github.com/aws/aws-sdk-go-v2/compare/service/ssooidc/v1.26.2...service/ssooidc/v1.30.1)
- github.com/aws/aws-sdk-go-v2/service/sts: [v1.30.1 → v1.33.19](https://github.com/aws/aws-sdk-go-v2/compare/service/sts/v1.30.1...service/sts/v1.33.19)
- github.com/aws/aws-sdk-go-v2: [v1.30.1 → v1.36.3](https://github.com/aws/aws-sdk-go-v2/compare/v1.30.1...v1.36.3)
- github.com/aws/smithy-go: [v1.20.3 → v1.22.3](https://github.com/aws/smithy-go/compare/v1.20.3...v1.22.3)
- github.com/containerd/containerd/api: [v1.8.0 → v1.9.0](https://github.com/containerd/containerd/compare/api/v1.8.0...api/v1.9.0)
- github.com/containerd/ttrpc: [v1.2.6 → v1.2.7](https://github.com/containerd/ttrpc/compare/v1.2.6...v1.2.7)
- github.com/containerd/typeurl/v2: [v2.2.2 → v2.2.3](https://github.com/containerd/typeurl/compare/v2.2.2...v2.2.3)
- github.com/coredns/corefile-migration: [v1.0.26 → v1.0.27](https://github.com/coredns/corefile-migration/compare/v1.0.26...v1.0.27)
- github.com/docker/docker: [v26.1.4+incompatible → v28.2.2+incompatible](https://github.com/docker/docker/compare/v26.1.4...v28.2.2)
- github.com/go-logr/logr: [v1.4.2 → v1.4.3](https://github.com/go-logr/logr/compare/v1.4.2...v1.4.3)
- github.com/google/cadvisor: [v0.52.1 → v0.53.0](https://github.com/google/cadvisor/compare/v0.52.1...v0.53.0)
- github.com/opencontainers/cgroups: [v0.0.1 → v0.0.3](https://github.com/opencontainers/cgroups/compare/v0.0.1...v0.0.3)
- github.com/opencontainers/runc: [v1.2.5 → v1.3.0](https://github.com/opencontainers/runc/compare/v1.2.5...v1.3.0)
- github.com/opencontainers/runtime-spec: [v1.2.0 → v1.2.1](https://github.com/opencontainers/runtime-spec/compare/v1.2.0...v1.2.1)
- github.com/prometheus/client_golang: [v1.22.0 → v1.23.2](https://github.com/prometheus/client_golang/compare/v1.22.0...v1.23.2)
- github.com/prometheus/client_model: [v0.6.1 → v0.6.2](https://github.com/prometheus/client_model/compare/v0.6.1...v0.6.2)
- github.com/prometheus/common: [v0.62.0 → v0.66.1](https://github.com/prometheus/common/compare/v0.62.0...v0.66.1)
- github.com/prometheus/procfs: [v0.15.1 → v0.16.1](https://github.com/prometheus/procfs/compare/v0.15.1...v0.16.1)
- github.com/spf13/cobra: [v1.9.1 → v1.10.0](https://github.com/spf13/cobra/compare/v1.9.1...v1.10.0)
- github.com/spf13/pflag: [v1.0.6 → v1.0.9](https://github.com/spf13/pflag/compare/v1.0.6...v1.0.9)
- github.com/stretchr/testify: [v1.10.0 → v1.11.1](https://github.com/stretchr/testify/compare/v1.10.0...v1.11.1)
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.58.0 → v0.61.0
- go.opentelemetry.io/otel/metric: v1.35.0 → v1.36.0
- go.opentelemetry.io/otel/sdk/metric: v1.34.0 → v1.36.0
- go.opentelemetry.io/otel/sdk: v1.34.0 → v1.36.0
- go.opentelemetry.io/otel/trace: v1.35.0 → v1.36.0
- go.opentelemetry.io/otel: v1.35.0 → v1.36.0
- golang.org/x/crypto: v0.36.0 → v0.41.0
- golang.org/x/mod: v0.21.0 → v0.27.0
- golang.org/x/net: v0.38.0 → v0.43.0
- golang.org/x/oauth2: v0.27.0 → v0.30.0
- golang.org/x/sync: v0.12.0 → v0.16.0
- golang.org/x/sys: v0.31.0 → v0.35.0
- golang.org/x/telemetry: bda5523 → 1a19826
- golang.org/x/term: v0.30.0 → v0.34.0
- golang.org/x/text: v0.23.0 → v0.28.0
- golang.org/x/tools: v0.26.0 → v0.36.0
- google.golang.org/genproto/googleapis/rpc: a0af3ef → 200df99
- google.golang.org/grpc: v1.72.1 → v1.72.2
- google.golang.org/protobuf: v1.36.5 → v1.36.8
- gopkg.in/evanphx/json-patch.v4: v4.12.0 → v4.13.0
- k8s.io/gengo/v2: 85fd79d → ec3ebc5
- k8s.io/kube-openapi: f3f2b99 → 589584f
- k8s.io/system-validators: v1.10.1 → v1.11.1
- sigs.k8s.io/json: cfa47c3 → 2d32026

### Removed
- gopkg.in/yaml.v2: v2.4.0