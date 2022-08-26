<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.25.0](#v1250)
  - [Downloads for v1.25.0](#downloads-for-v1250)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.24.0](#changelog-since-v1240)
  - [What's New (Major Themes)](#whats-new-major-themes)
    - [PodSecurityPolicy is Removed, Pod Security Admission graduates to Stable](#podsecuritypolicy-is-removed-pod-security-admission-graduates-to-stable)
    - [Ephemeral Containers Graduate to Stable](#ephemeral-containers-graduate-to-stable)
    - [Support for cgroups v2 Graduates to Stable](#support-for-cgroups-v2-graduates-to-stable)
    - [Windows support improved](#windows-support-improved)
    - [Moved container registry service from k8s.gcr.io to registry.k8s.io](#moved-container-registry-service-from-k8sgcrio-to-registryk8sio)
    - [Promoted SeccompDefault to Beta](#promoted-seccompdefault-to-beta)
    - [Promoted endPort in Network Policy to Stable](#promoted-endport-in-network-policy-to-stable)
    - [Promoted Local Ephemeral Storage Capacity Isolation to Stable](#promoted-local-ephemeral-storage-capacity-isolation-to-stable)
    - [Promoted core CSI Migration to Stable](#promoted-core-csi-migration-to-stable)
    - [Promoted CSI Ephemeral Volume to Stable](#promoted-csi-ephemeral-volume-to-stable)
    - [Promoted CRD Validation Expression Language to Beta](#promoted-crd-validation-expression-language-to-beta)
    - [Promoted Server Side Unknown Field Validation to Beta](#promoted-server-side-unknown-field-validation-to-beta)
    - [Kube-proxy images are now based in distroless](#kube-proxy-images-are-now-based-in-distroless)
    - [Introduced KMS v2](#introduced-kms-v2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
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
- [v1.25.0-rc.1](#v1250-rc1)
  - [Downloads for v1.25.0-rc.1](#downloads-for-v1250-rc1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.25.0-rc.0](#changelog-since-v1250-rc0)
  - [Changes by Kind](#changes-by-kind-1)
    - [Documentation](#documentation-1)
    - [Bug or Regression](#bug-or-regression-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.25.0-rc.0](#v1250-rc0)
  - [Downloads for v1.25.0-rc.0](#downloads-for-v1250-rc0)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.25.0-beta.0](#changelog-since-v1250-beta0)
  - [Changes by Kind](#changes-by-kind-2)
    - [API Change](#api-change-1)
    - [Bug or Regression](#bug-or-regression-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.25.0-beta.0](#v1250-beta0)
  - [Downloads for v1.25.0-beta.0](#downloads-for-v1250-beta0)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.25.0-alpha.3](#changelog-since-v1250-alpha3)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-2)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.25.0-alpha.3](#v1250-alpha3)
  - [Downloads for v1.25.0-alpha.3](#downloads-for-v1250-alpha3)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
    - [Container Images](#container-images-4)
  - [Changelog since v1.25.0-alpha.2](#changelog-since-v1250-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-4)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-3)
    - [Feature](#feature-2)
    - [Documentation](#documentation-2)
    - [Bug or Regression](#bug-or-regression-4)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)
- [v1.25.0-alpha.2](#v1250-alpha2)
  - [Downloads for v1.25.0-alpha.2](#downloads-for-v1250-alpha2)
    - [Source Code](#source-code-5)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
    - [Container Images](#container-images-5)
  - [Changelog since v1.25.0-alpha.1](#changelog-since-v1250-alpha1)
  - [Changes by Kind](#changes-by-kind-5)
    - [API Change](#api-change-4)
    - [Feature](#feature-3)
    - [Documentation](#documentation-3)
    - [Bug or Regression](#bug-or-regression-5)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-5)
    - [Added](#added-5)
    - [Changed](#changed-5)
    - [Removed](#removed-5)
- [v1.25.0-alpha.1](#v1250-alpha1)
  - [Downloads for v1.25.0-alpha.1](#downloads-for-v1250-alpha1)
    - [Source Code](#source-code-6)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
    - [Container Images](#container-images-6)
  - [Changelog since v1.24.0](#changelog-since-v1240-1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-3)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-3)
  - [Changes by Kind](#changes-by-kind-6)
    - [Deprecation](#deprecation-3)
    - [API Change](#api-change-5)
    - [Feature](#feature-4)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-6)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-6)
    - [Added](#added-6)
    - [Changed](#changed-6)
    - [Removed](#removed-6)

<!-- END MUNGE: GENERATED_TOC -->

# v1.25.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.25.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes.tar.gz) | `2bff2da02f6197fbde3e3a378dd8a95415edcef2e5f95a9e1399ec8369a592dac461dbb7402cded1ba93ace22c87ee050ea02a7c1c2cabaa97609352302d5d0c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-src.tar.gz) | `20810dbbc1ee7ec06348b09b1d0dcef456eab600b3c6fe594d426a9c7b6fbab69712594d6f97c63db371a41a509f17bb7c4675b8e28a25dd84abb9bea35fc8ad`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-darwin-amd64.tar.gz) | `4960a153e5cda0b30e7e437895fc8c08f80978d9f3189f6408e04446fa9777dafacb3cddbac5b09c8de0a19459893150a691b28f95cfa9f5c87b2926fe442df5`
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-darwin-arm64.tar.gz) | `17972e1f0ad64113b9adf5f8e9c6231463048c0f04fde35fb614f59281127630ae475a7ec91e31cb03ebe444828e47e6d32ed05f000b6514bad646d6b9f5469c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-linux-386.tar.gz) | `8d5a35a0a8e71ec59be0b720ff6fd80c172bcacae5f9a5f58bfc1ac66b4e877640d8626f23852ba2459dc1ec783347ae3300017e0919d59b16ae1011ac50c5d1`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-linux-amd64.tar.gz) | `34a7e9a496fff31a3afa6f5f7245212d051de3c2966e42a662040bde8a733c1cf55ce2e50227813fd29c6db758687a453a7df66b6c32f7f2c93959280c4e130a`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-linux-arm.tar.gz) | `f5defcd92f99562c455f44bf62b478d88d00ed3cce662bdd8bed8b880bae31e6e534fb044844b4664c543bed713332fd1734415e6da320f752984157e3fa32c9`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-linux-arm64.tar.gz) | `a01731aa3e7e8e560c92cf0b3d7ed9ce9964bfed88fcd055c881a0513c10be33e11b2c0539d4f52c17bfd605d540be5c6824a2f9f182d97a7a255a9a25b64607`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-linux-ppc64le.tar.gz) | `b022a28c25b0fefb1abbcaaa71d12f9edd14cf76d291ef3f8d13a8fc5cdec2edef3aee96c851e0dde7b8c7c1f5f38c3cc631e0bce15ccbc6f30bac45895ac7f4`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-linux-s390x.tar.gz) | `b4781d09fd1360097104fb6cd33e3f4de9ce571a13e329f26ce55505e9ddbbd2ca20794bb80a58eaa23ae63860376ae51187de76c99db9cbce984773ddf03c34`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-windows-386.tar.gz) | `5f6f71b7cb88dc18930d1864df21061f5b73a6969425701e945239323a0d7e5eb182c1607d3ddc26cbaf4dab0dc4bc59c7fbffe8bd66f5fd1edae075220d1c15`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-windows-amd64.tar.gz) | `cfcbdc13cfd17e8b38baa29d7638ae086b97d5784352d0e8bc92c6f6f34ce2e885ef1371d255d4b48c9a73cfdea8168d1d6cc7ac0b1b3bdc18a09970066d59ca`
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-client-windows-arm64.tar.gz) | `55cc2d36764d8f496ca973d82fd1c60e93d227a8107fd0649c228f479b70dc05cbe23708fd8e4b820fd3b191786bbe054446d31f7d1b9fecff173d0f554d5a69`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-server-linux-amd64.tar.gz) | `21614040cd3cc5a8ee0668bd91383427427e7796ee3de0f9fe6b4b5d9becb830141bb1ed3ba5376815385baa2675595e97765209f14bf68653bcf6fdfb070f3d`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-server-linux-arm.tar.gz) | `44ada6a0a6ba77aebdad52680c973f98bed495bdf7ac7cc6abbcb7e5b3fd5476961e0db6b8384b128c81ffd0b70d3a020f248062acfcd8b8b85fa295b1f8cf69`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-server-linux-arm64.tar.gz) | `c76ca3dc152a51c08d147a9ad9594a70071e548638d5065bf1aaaad63c3553c8af8d8dcb885e6e4ed724d944a5283fd000315d2622b7ef7a2df4b0b783ab0256`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-server-linux-ppc64le.tar.gz) | `491084954f951a3f3f61642e9006195f5dac84b4ccc29e407f85a3bb7aba8ea79c2b4deda69507bd516c1feff47ba9887309c26d6e6c4d2270cf5e0a8f0d5cac`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-server-linux-s390x.tar.gz) | `0e89499c004e5faaf69deb917c1df2eb3c2da36024ff3dcd10e5c3f7e7ef7c2393c109bb8ee8bd30756bc4024cacc05ea5f0131a5bc630a3513aada182a69949`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-node-linux-amd64.tar.gz) | `46b4fd437824b1178dd570bc3a1a12aa5549482793f329d194b72a45daaa0bb8b990d7ee98f2a3d9a643ae113973f1f303cdcb6fdf8c56f439bf13acc7460728`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-node-linux-arm.tar.gz) | `f4338883d811369be6d7a2b658e3b46c06ddef0b584293c3e44bb5f961805d6659d2a73e62c470b80029938f8793f1b11c093e4b99431e419bd61b0e9ed5c133`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-node-linux-arm64.tar.gz) | `c2b174bb22485c0efb6a812ed78d84afa161d635f365d913ea08b3db8daae50ae15087a6d5fefce95c5363d90414079d8c5ff476bbfab9547c87befdf23bbcce`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-node-linux-ppc64le.tar.gz) | `7985f22fb43d3af183e94581dd7547b8817ad27b2c8b6304de60d28ce165341ab5dcd576f6c61b999c73ebf6a35d5cd5e328253160959f5112b94e4d17f41364`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-node-linux-s390x.tar.gz) | `7d6622e3128c3ad46ab286bdbb445f158b697f7624ff03f1905d4b62a91601d0f33880b4b0cfa93b435a9c786070f20ef86da877690c100bfa7460a8e8ca3cd5`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0/kubernetes-node-windows-amd64.tar.gz) | `213954569d1e0b682805342964597a03d3b16a98756c9123d18dd8ceda9d4cddeeac500d5bdcf8cb27c136b2bfa5947fb5c75fdec7533a153199a855162d742e`

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.
name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.25.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.25.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.25.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.25.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.25.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0

## What's New (Major Themes)

### PodSecurityPolicy is Removed, Pod Security Admission graduates to Stable

PodSecurityPolicy was initially [deprecated in v1.21](/blog/2021/04/06/podsecuritypolicy-deprecation-past-present-and-future/), and with the release of v1.25, it has been removed. The updates required to improve its usability would have introduced breaking changes, so it became necessary to remove it in favor of a more friendly replacement. That replacement is [Pod Security Admission](/docs/concepts/security/pod-security-admission/), which graduates to Stable with this release. If you are currently relying on PodSecurityPolicy, please follow the instructions for [migration to Pod Security Admission](/docs/tasks/configure-pod-container/migrate-from-psp/).

### Ephemeral Containers Graduate to Stable

[Ephemeral Containers](/docs/concepts/workloads/pods/ephemeral-containers/) are containers that exist for only a limited time within an existing pod. This is particularly useful for troubleshooting when you need to examine another container but cannot use `kubectl exec` because that container has crashed or its image lacks debugging utilities. Ephemeral containers graduated to Beta in Kubernetes v1.23, and with this release, the feature graduates to Stable.

### Support for cgroups v2 Graduates to Stable

It has been more than two years since the Linux kernel cgroups v2 API was declared stable. With some distributions now defaulting to this API, Kubernetes must support it to continue operating on those distributions. cgroups v2 offers several improvements over cgroups v1, for more information see the [cgroups v2](https://kubernetes.io/docs/concepts/architecture/cgroups/) documentation. While cgroups v1 will continue to be supported, this enhancement puts Kubernetes to be ready for eventual deprecation and replacement in favor of v2.


### Windows support improved

- [Performance dashboards](http://perf-dash.k8s.io/#/?jobname=soak-tests-capz-windows-2019) added support for Windows
- [Unit tests](https://github.com/kubernetes/kubernetes/issues/51540) added support for Windows
- [Conformance tests](https://github.com/kubernetes/kubernetes/pull/108592) added support for Windows
- New repository created for [Windows Operational Readiness](https://github.com/kubernetes-sigs/windows-operational-readiness)

### Moved container registry service from k8s.gcr.io to registry.k8s.io

[Moving container registry from k8s.gcr.io to registry.k8s.io](https://github.com/kubernetes/kubernetes/pull/109938) got merged. For more details, see the [wiki page](https://github.com/kubernetes/k8s.io/wiki/New-Registry-url-for-Kubernetes-\(registry.k8s.io\)), [announcement](https://groups.google.com/a/kubernetes.io/g/dev/c/DYZYNQ_A6_c/m/oD9_Q8Q9AAAJ) was sent to the kubernetes development mailing list.

### Promoted SeccompDefault to Beta

SeccompDefault promoted to beta, see the tutorial [Restrict a Container's Syscalls with seccomp](https://kubernetes.io/docs/tutorials/security/seccomp/#enable-the-use-of-runtimedefault-as-the-default-seccomp-profile-for-all-workloads) for more details.

### Promoted endPort in Network Policy to Stable

Promoted `endPort` in [Network Policy](https://kubernetes.io/docs/concepts/services-networking/network-policies/#targeting-a-range-of-ports) to GA. Network Policy providers that support `endPort` field now can use it to specify a range of ports to apply a Network Policy. Previously, each Network Policy could only target a single port.

Please be aware that `endPort` field **MUST BE SUPPORTED** by the Network Policy provider. If your provider does not support `endPort`, and this field is specified in a Network Policy, the Network Policy will be created covering only the port field (single port).

### Promoted Local Ephemeral Storage Capacity Isolation to Stable
The [Local Ephemeral Storage Capacity Isolation](https://github.com/kubernetes/enhancements/tree/master/keps/sig-storage/361-local-ephemeral-storage-isolation) feature moved to GA. This was introduced as alpha in 1.8, moved to beta in 1.10, and it is now a stable feature. It provides support for capacity isolation of local ephemeral storage between pods, such as `EmptyDir`, so that a pod can be hard limited in its consumption of shared resources by evicting Pods if its consumption of local ephemeral storage exceeds that limit.

### Promoted core CSI Migration to Stable

[CSI Migration](https://kubernetes.io/blog/2021/12/10/storage-in-tree-to-csi-migration-status-update/#quick-recap-what-is-csi-migration-and-why-migrate) is an ongoing effort that SIG Storage has been working on for a few releases. The goal is to move in-tree volume plugins to out-of-tree CSI drivers and eventually remove the in-tree volume plugins. The [core CSI Migration](https://github.com/kubernetes/enhancements/tree/master/keps/sig-storage/625-csi-migration) feature moved to GA. CSI Migration for GCE PD and AWS EBS also moved to GA. CSI Migration for vSphere remains in beta (but is on by default). CSI Migration for Portworx moved to Beta (but is off-by-default).


### Promoted CSI Ephemeral Volume to Stable

The [CSI Ephemeral Volume](https://github.com/kubernetes/enhancements/tree/master/keps/sig-storage/596-csi-inline-volumes) feature allows CSI volumes to be specified directly in the pod specification for ephemeral use cases. They can be used to inject arbitrary states, such as configuration, secrets, identity, variables or similar information, directly inside pods using a mounted volume. This was initially introduced in 1.15 as an alpha feature, and it moved to GA. This feature is used by some CSI drivers such as the [secret-store CSI driver](https://github.com/kubernetes-sigs/secrets-store-csi-driver).

### Promoted CRD Validation Expression Language to Beta

[CRD Validation Expression Language](https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/2876-crd-validation-expression-language/README.mdv) is promoted to beta, which makes it possible to declare how custom resources are validated using the [Common Expression Language (CEL)](https://github.com/google/cel-spec). Please see the [validation rules](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/#validation-rules) guide.

### Promoted Server Side Unknown Field Validation to Beta

Promoted the `ServerSideFieldValidation` feature gate to beta (on by default). This allows optionally triggering schema validation on the API server that errors when unknown fields are detected. This allows the removal of client-side validation from kubectl while maintaining the same core functionality of erroring out on requests that contain unknown or invalid fields.


###  Introduced KMS v2

Introduce KMS v2alpha1 API to add performance, rotation, and observability improvements. Encrypt data at rest (ie Kubernetes `Secrets`) with DEK using AES-GCM instead of AES-CBC for kms data encryption. No user action is required. Reads with AES-GCM and AES-CBC will continue to be allowed. See the guide [Using a KMS provider for data encryption](https://kubernetes.io/docs/tasks/administer-cluster/kms-provider/) for more information.

### Kube-proxy images are now based on distroless images

In previous releases, kube-proxy container images were built using Debian as the base image. Starting with this release, the images are now built using [distroless](https://github.com/GoogleContainerTools/distroless). This change reduced image size by almost 50% and decreased the number of installed packages and files to only those strictly required for kube-proxy to do its job.

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

- Deprecated beta APIs scheduled for removal in 1.25 are no longer served. See https://kubernetes.io/docs/reference/using-api/deprecation-guide/#v1-25 for more information. ([#108797](https://github.com/kubernetes/kubernetes/pull/108797), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Instrumentation and Testing]
 - Encrypted data with DEK using AES-GCM instead of AES-CBC for kms data encryption. No user action required. Reads with AES-GCM and AES-CBC will continue to be allowed. ([#111119](https://github.com/kubernetes/kubernetes/pull/111119), [@aramase](https://github.com/aramase))
 - End-to-end testing has been migrated from Ginkgo v1 to v2.

  When running test/e2e via the Ginkgo CLI, the v2 CLI must be used and `-timeout=24h` (or some other, suitable value) must be passed because the default timeout was reduced from 24h to 1h. When running it via `go test`, the corresponding `-args` parameter is `-ginkgo.timeout=24h`. To build the CLI in the Kubernetes repo, use `make all WHAT=github.com/onsi/ginkgo/v2/ginkgo`.
  Ginkgo V2 doesn't accept go test's `-parallel` flags to parallelize Ginkgo specs, please switch to use `ginkgo -p` or `ginkgo -procs=N` instead. ([#109111](https://github.com/kubernetes/kubernetes/pull/109111), [@chendave](https://github.com/chendave)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage, Testing and Windows]
 - No action required; No API/CLI changed; Add new Windows Image Support ([#110333](https://github.com/kubernetes/kubernetes/pull/110333), [@liurupeng](https://github.com/liurupeng)) [SIG Cloud Provider and Windows]
 - The intree volume plugin flocker support was completely removed from Kubernetes. ([#111618](https://github.com/kubernetes/kubernetes/pull/111618), [@Jiawei0227](https://github.com/Jiawei0227))
 - The intree volume plugin quobyte support has been completely removed from Kubernetes. ([#111619](https://github.com/kubernetes/kubernetes/pull/111619), [@Jiawei0227](https://github.com/Jiawei0227))
 - The intree volume plugin storageos support has been completely removed from Kubernetes. ([#111620](https://github.com/kubernetes/kubernetes/pull/111620), [@Jiawei0227](https://github.com/Jiawei0227))
 - There is a new OCI image registry (`registry.k8s.io`) that can be used to pull Kubernetes images. The old registry (`k8s.gcr.io`) will continue to be supported for the foreseeable future, but the new name should perform better because it frontends equivalent mirrors in other clouds.  Please point your clusters to the new registry going forward. \n\nAdmission/Policy integrations that have an allowlist of registries need to include `registry.k8s.io` alongside `k8s.gcr.io`.\nAir-gapped environments and image garbage-collection configurations will need to update to pre-pull and preserve required images under `registry.k8s.io` as well as `k8s.gcr.io`. ([#109938](https://github.com/kubernetes/kubernetes/pull/109938), [@dims](https://github.com/dims))

## Changes by Kind

### Deprecation

- API server's deprecated `--service-account-api-audiences` flag was removed.  Use `--api-audiences` instead. ([#108624](https://github.com/kubernetes/kubernetes/pull/108624), [@ialidzhikov](https://github.com/ialidzhikov))
- Ginkgo.Measure has been deprecated in Ginkgo V2, switch to use gomega/gmeasure instead ([#111065](https://github.com/kubernetes/kubernetes/pull/111065), [@chendave](https://github.com/chendave)) [SIG Autoscaling and Testing]
- Kube-controller-manager: Removed flags  `deleting-pods-qps`, `deleting-pods-burst`, and `register-retry-count`. ([#109612](https://github.com/kubernetes/kubernetes/pull/109612), [@pandaamanda](https://github.com/pandaamanda))
- Kubeadm: during "upgrade apply/diff/node", in case the `ClusterConfiguration.imageRepository` stored in the "kubeadm-config" `ConfigMap` contains the legacy "k8s.gcr.io" repository, modify it to the new default "registry.k8s.io". Reflect the change in the in-cluster `ConfigMap` only during "upgrade apply". ([#110343](https://github.com/kubernetes/kubernetes/pull/110343), [@neolit123](https://github.com/neolit123))
- Kubeadm: graduated the kubeadm specific feature gate `UnversionedKubeletConfigMap` to GA and locked it to `true` by default. The kubelet related ConfigMap and RBAC rules are now locked to have a simplified naming `*kubelet-config` instead of the legacy naming `*kubelet-config-x.yy`, where `x.yy` was the version of the control plane. If you have previously used the old naming format with `UnversionedKubeletConfigMap=false`, you must manually copy the config map from `kube-system/kubelet-config-x.yy` to `kube-system/kubelet-config` before upgrading to `v1.25`. ([#110327](https://github.com/kubernetes/kubernetes/pull/110327), [@neolit123](https://github.com/neolit123))
- Kubeadm: stop applying the `node-role.kubernetes.io/master:NoSchedule` taint to control plane nodes for new clusters. Remove the taint from existing control plane nodes during "kubeadm upgrade apply" ([#110095](https://github.com/kubernetes/kubernetes/pull/110095), [@neolit123](https://github.com/neolit123))
- Support for the alpha seccomp annotations `seccomp.security.alpha.kubernetes.io/pod` and `container.seccomp.security.alpha.kubernetes.io`, deprecated since v1.19, was partially removed. Kubelets no longer support the annotations, use of the annotations in static pods is no longer supported, and the seccomp annotations are no longer auto-populated when pods with seccomp fields are created. Auto-population of the seccomp fields from the annotations is planned to be removed in 1.27. Pods  should use the corresponding pod or container `securityContext.seccompProfile` field instead. ([#109819](https://github.com/kubernetes/kubernetes/pull/109819), [@saschagrunert](https://github.com/saschagrunert))
- The `gcp` and `azure` auth plugins have been removed from client-go and kubectl. See https://github.com/Azure/kubelogin and https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke ([#110013](https://github.com/kubernetes/kubernetes/pull/110013), [@enj](https://github.com/enj))
- The beta `PodSecurityPolicy` admission plugin, deprecated since 1.21, is removed. Follow the instructions at https://kubernetes.io/docs/tasks/configure-pod-container/migrate-from-psp/ to migrate to the built-in PodSecurity admission plugin (or to another third-party  policy webhook) prior to upgrading to v1.25. ([#109798](https://github.com/kubernetes/kubernetes/pull/109798), [@liggitt](https://github.com/liggitt))
- VSphere releases less than 7.0u2 are not supported for in-tree vSphere volume  as of Kubernetes v1.25. Please consider upgrading vSphere (both ESXi and vCenter)  to 7.0u2 or above. ([#111255](https://github.com/kubernetes/kubernetes/pull/111255), [@divyenpatel](https://github.com/divyenpatel))
- Windows winkernel kube-proxy no longer supports Windows HNS v1 APIs. ([#110957](https://github.com/kubernetes/kubernetes/pull/110957), [@papagalu](https://github.com/papagalu))

### API Change

- Add `NodeInclusionPolicy` to `TopologySpreadConstraints` in PodSpec. ([#108492](https://github.com/kubernetes/kubernetes/pull/108492), [@kerthcet](https://github.com/kerthcet))
- Added KMS v2alpha1 support. ([#111126](https://github.com/kubernetes/kubernetes/pull/111126), [@aramase](https://github.com/aramase))
- Added a deprecated warning for node beta label usage in PV/SC/RC and CSI Storage Capacity. ([#108554](https://github.com/kubernetes/kubernetes/pull/108554), [@pacoxu](https://github.com/pacoxu))
- Added a new feature gate `CheckpointRestore` to enable support to checkpoint containers. If enabled it is possible to checkpoint a container using the newly kubelet API (/checkpoint/{podNamespace}/{podName}/{containerName}). ([#104907](https://github.com/kubernetes/kubernetes/pull/104907), [@adrianreber](https://github.com/adrianreber)) [SIG Node and Testing]
- Added alpha support for user namespaces in pods phase 1 (KEP 127, feature gate: UserNamespacesStatelessPodsSupport) ([#111090](https://github.com/kubernetes/kubernetes/pull/111090), [@rata](https://github.com/rata))
- As of v1.25, the PodSecurity `restricted` level no longer requires pods that set .spec.os.name="windows" to also set Linux-specific securityContext fields. If a 1.25+ cluster has unsupported [out-of-skew](https://kubernetes.io/releases/version-skew-policy/#kubelet) nodes prior to v1.23 and wants to ensure namespaces enforcing the `restricted` policy continue to require Linux-specific securityContext fields on all pods, ensure a version of the `restricted` prior to v1.25 is selected by labeling the namespace (for example, `pod-security.kubernetes.io/enforce-version: v1.24`) ([#105919](https://github.com/kubernetes/kubernetes/pull/105919), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- Changed ownership semantics of PersistentVolume's spec.claimRef from `atomic` to `granular`. ([#110495](https://github.com/kubernetes/kubernetes/pull/110495), [@alexzielenski](https://github.com/alexzielenski))
- Extended ContainerStatus CRI API to allow runtime response with container resource requests and limits that are in effect.
  - UpdateContainerResources CRI API now supports both Linux and Windows. ([#111645](https://github.com/kubernetes/kubernetes/pull/111645), [@vinaykul](https://github.com/vinaykul))
- For v1.25, Kubernetes will be using Golang 1.19, In this PR the version is updated to 1.19rc2 as GA is not yet available. ([#111254](https://github.com/kubernetes/kubernetes/pull/111254), [@dims](https://github.com/dims))
- Introduced NodeIPAM support for multiple ClusterCIDRs ([#2593](https://github.com/kubernetes/enhancements/issues/2593)) as an alpha feature.
  Set feature gate `MultiCIDRRangeAllocator=true`, determines whether the `MultiCIDRRangeAllocator` controller can be used, while the kube-controller-manager flag below will pick the active controller.
  Enabled the `MultiCIDRRangeAllocator` by setting `--cidr-allocator-type=MultiCIDRRangeAllocator` flag in kube-controller-manager. ([#109090](https://github.com/kubernetes/kubernetes/pull/109090), [@sarveshr7](https://github.com/sarveshr7))
- Introduced PodHasNetwork condition for pods. ([#111358](https://github.com/kubernetes/kubernetes/pull/111358), [@ddebroy](https://github.com/ddebroy))
- Introduced support for handling pod failures with respect to the configured pod failure policy rules. ([#111113](https://github.com/kubernetes/kubernetes/pull/111113), [@mimowo](https://github.com/mimowo))
- Introduction of the `DisruptionTarget` pod condition type. Its `reason` field indicates the reason for pod termination:
  - PreemptionByKubeScheduler (Pod preempted by kube-scheduler)
  - DeletionByTaintManager (Pod deleted by taint manager due to NoExecute taint)
  - EvictionByEvictionAPI (Pod evicted by Eviction API)
  - DeletionByPodGC (an orphaned Pod deleted by PodGC) ([#110959](https://github.com/kubernetes/kubernetes/pull/110959), [@mimowo](https://github.com/mimowo))
- Kube-Scheduler ComponentConfig is graduated to GA, `kubescheduler.config.k8s.io/v1` is available now.
  Plugin `SelectorSpread` is removed in v1. ([#110534](https://github.com/kubernetes/kubernetes/pull/110534), [@kerthcet](https://github.com/kerthcet))
- Local Storage Capacity Isolation feature is GA in 1.25 release. For systems (rootless) that cannot check root file system, please use kubelet config --local-storage-capacity-isolation=false to disable this feature. Once disabled, pod cannot set local ephemeral storage request/limit, and emptyDir sizeLimit niether. ([#111513](https://github.com/kubernetes/kubernetes/pull/111513), [@jingxu97](https://github.com/jingxu97))
- Make PodSpec.Ports' description clearer on how this information is only informational and how it can be incorrect. ([#110564](https://github.com/kubernetes/kubernetes/pull/110564), [@j4m3s-s](https://github.com/j4m3s-s)) [SIG API Machinery, Network and Node]
- On compatible systems, a mounter's Unmount implementation is changed to not return an error when the specified target can be detected as not a mount point. On Linux, the behavior of detecting a mount point depends on `umount` command is validated when the mounter is created. Additionally, mount point checks will be skipped in CleanupMountPoint/CleanupMountWithForce if the mounter's Unmount having the changed behavior of not returning error when target is not a mount point. ([#109676](https://github.com/kubernetes/kubernetes/pull/109676), [@cartermckinnon](https://github.com/cartermckinnon)) [SIG Storage]
- PersistentVolumeClaim objects are no longer left with storage class set to `nil` forever, but will be updated retroactively once any StorageClass is set or created as default. ([#111467](https://github.com/kubernetes/kubernetes/pull/111467), [@RomanBednar](https://github.com/RomanBednar))
- Promote StatefulSet minReadySeconds to GA. This means `--feature-gates=StatefulSetMinReadySeconds=true` are not needed on kube-apiserver and kube-controller-manager binaries and they'll be removed soon following policy at https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecation ([#110896](https://github.com/kubernetes/kubernetes/pull/110896), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG API Machinery, Apps and Testing]
- Promoted CronJob's TimeZone support to beta. ([#111435](https://github.com/kubernetes/kubernetes/pull/111435), [@soltysh](https://github.com/soltysh))
- Promoted DaemonSet MaxSurge to GA. This means `--feature-gates=DaemonSetUpdateSurge=true` are not needed on kube-apiserver and kube-controller-manager binaries and they'll be removed soon following policy at https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecation . ([#111194](https://github.com/kubernetes/kubernetes/pull/111194), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- Scheduler: included supported ScoringStrategyType list in error message for NodeResourcesFit plugin ([#111206](https://github.com/kubernetes/kubernetes/pull/111206), [@SataQiu](https://github.com/SataQiu))
- The Go API for logging configuration in `k8s.io/component-base` was moved to `k8s.io/component-base/logs/api/v1`. The configuration file format and command line flags are the same as before. ([#105797](https://github.com/kubernetes/kubernetes/pull/105797), [@pohly](https://github.com/pohly))
- The Pod `spec.podOS` field is promoted to GA. The `IdentifyPodOS` feature gate unconditionally enabled, and will no longer be accepted as a `--feature-gates` parameter in 1.27. ([#111229](https://github.com/kubernetes/kubernetes/pull/111229), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- The PodTopologySpread is respected after rolling upgrades. ([#111441](https://github.com/kubernetes/kubernetes/pull/111441), [@denkensk](https://github.com/denkensk))
- The `CSIInlineVolume` feature has moved from beta to GA. ([#111258](https://github.com/kubernetes/kubernetes/pull/111258), [@dobsonj](https://github.com/dobsonj))
- The `PodSecurity` admission plugin has graduated to GA and is enabled by default. The admission configuration version has been promoted to `pod-security.admission.config.k8s.io/v1`. ([#110459](https://github.com/kubernetes/kubernetes/pull/110459), [@wangyysde](https://github.com/wangyysde))
- The `endPort` field in Network Policy is now promoted to GA

  Network Policy providers that support `endPort` field now can use it to specify a range of ports to apply a Network Policy.

  Previously, each Network Policy could only target a single port.

  Please be aware that `endPort` field MUST BE SUPPORTED by the Network Policy provider. In case your provider does not support `endPort` and this field is specified in a Network Policy, the Network Policy will be created covering only the port field (single port). ([#110868](https://github.com/kubernetes/kubernetes/pull/110868), [@rikatz](https://github.com/rikatz))
- The `metadata.clusterName` field is completely removed. This should not have any user-visible impact. ([#109602](https://github.com/kubernetes/kubernetes/pull/109602), [@lavalamp](https://github.com/lavalamp))
- The `minDomains` field in Pod Topology Spread is graduated to beta ([#110388](https://github.com/kubernetes/kubernetes/pull/110388), [@sanposhiho](https://github.com/sanposhiho)) [SIG API Machinery and Apps]
- The command line flag `enable-taint-manager` for kube-controller-manager is deprecated and will be removed in 1.26. The feature that it supports, taint based eviction, is enabled by default and will continue to be implicitly enabled when the flag is removed. ([#111411](https://github.com/kubernetes/kubernetes/pull/111411), [@alculquicondor](https://github.com/alculquicondor))
- This release added support for `NodeExpandSecret` for CSI driver client which enables the CSI drivers to make use of this secret while performing node expansion operation based on the user request. Previously there was no secret  provided as part of the `nodeexpansion` call, thus CSI drivers did not make use of the same while expanding the volume at the node side. ([#105963](https://github.com/kubernetes/kubernetes/pull/105963), [@zhucan](https://github.com/zhucan))
- [Ephemeral Containers](https://kubernetes.io/docs/concepts/workloads/pods/ephemeral-containers/) are now generally available (GA). The `EphemeralContainers` feature gate is always enabled and should be removed from `--feature-gates` flag on the kube-apiserver and the kubelet command lines. The `EphemeralContainers` feature gate is [deprecated and scheduled for removal](https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecation)  in a future release. ([#111402](https://github.com/kubernetes/kubernetes/pull/111402), [@verb](https://github.com/verb))

### Feature

- Added KMS `v2alpha1` API. ([#110201](https://github.com/kubernetes/kubernetes/pull/110201), [@aramase](https://github.com/aramase))
- Added Service Account field in the output of `kubectl describe pod` command. ([#111192](https://github.com/kubernetes/kubernetes/pull/111192), [@aufarg](https://github.com/aufarg))
- Added a new `align-by-socket` policy option to cpu manager `static` policy.  When enabled CPU's to be aligned at socket boundary rather than NUMA boundary. ([#111278](https://github.com/kubernetes/kubernetes/pull/111278), [@arpitsardhana](https://github.com/arpitsardhana))
- Added container probe duration metrics. ([#104484](https://github.com/kubernetes/kubernetes/pull/104484), [@jackfrancis](https://github.com/jackfrancis))
- Added new flags into alpha events such as --output, --types, --no-headers. ([#110007](https://github.com/kubernetes/kubernetes/pull/110007), [@ardaguclu](https://github.com/ardaguclu))
- Added sum feature to `kubectl top pod` ([#105100](https://github.com/kubernetes/kubernetes/pull/105100), [@lauchokyip](https://github.com/lauchokyip))
- Added the `Apply` and `ApplyStatus` methods to the dynamic `ResourceInterface` ([#109443](https://github.com/kubernetes/kubernetes/pull/109443), [@kevindelgado](https://github.com/kevindelgado))
- Feature gate `CSIMigration` was locked to enabled. `CSIMigration` is GA now. The feature gate will be removed in `v1.27`. ([#110410](https://github.com/kubernetes/kubernetes/pull/110410), [@Jiawei0227](https://github.com/Jiawei0227))
- Feature gate `ProbeTerminationGracePeriod` is enabled by default. ([#108541](https://github.com/kubernetes/kubernetes/pull/108541), [@kerthcet](https://github.com/kerthcet))
- Ginkgo: when e2e tests are invoked through ginkgo-e2e.sh, the default now is to use color escape sequences only when connected to a terminal. `GINKGO_NO_COLOR=y/n` can be used to override that default. ([#111633](https://github.com/kubernetes/kubernetes/pull/111633), [@pohly](https://github.com/pohly))
- Graduated SeccompDefault to `beta`. The Kubelet feature gate is now enabled by default and the configuration/CLI flag still defaults to `false`. ([#110805](https://github.com/kubernetes/kubernetes/pull/110805), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node and Testing]
- Graduated ServerSideFieldValidation to `beta`. Schema validation is performed server-side and requests will receive warnings for any invalid/unknown fields by default. ([#110178](https://github.com/kubernetes/kubernetes/pull/110178), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- Graduated `CustomResourceValidationExpressions` to `beta`. The `CustomResourceValidationExpressions` feature gate is now enabled by default. ([#111524](https://github.com/kubernetes/kubernetes/pull/111524), [@cici37](https://github.com/cici37))
- Graduated `ServiceIPStaticSubrange` feature to Beta (disabled by default). ([#110419](https://github.com/kubernetes/kubernetes/pull/110419), [@aojea](https://github.com/aojea))
- If a Pod has a DisruptionTarget condition with status=True for more than 2 minutes without getting a DeletionTimestamp, the control plane resets it to status=False. ([#111475](https://github.com/kubernetes/kubernetes/pull/111475), [@alculquicondor](https://github.com/alculquicondor))
- In "large" clusters, kube-proxy in iptables mode will now sometimes
  leave unused rules in iptables for a while (up to `--iptables-sync-period`)
  before deleting them. This improves performance by not requiring it to
  check for stale rules on every sync. (In smaller clusters, it will still
  remove unused rules immediately once they are no longer used.)

  (The threshold for "large" used here is currently "1000 endpoints" but
  this is subject to change.) ([#110334](https://github.com/kubernetes/kubernetes/pull/110334), [@danwinship](https://github.com/danwinship))
- Kube-up now includes CoreDNS version v1.9.3. ([#110488](https://github.com/kubernetes/kubernetes/pull/110488), [@mzaian](https://github.com/mzaian))
- Kubeadm: Added support for additional authentication strategies in `kubeadm join` with discovery/kubeconfig file: client-go authentication plugins (`exec`), `tokenFile`, and `authProvider.` ([#110553](https://github.com/kubernetes/kubernetes/pull/110553), [@tallaxes](https://github.com/tallaxes))
- Kubeadm: Update CoreDNS to v1.9.3. ([#110489](https://github.com/kubernetes/kubernetes/pull/110489), [@pacoxu](https://github.com/pacoxu))
- Kubeadm: added support for the flag `--print-manifest` to the addon phases `kube-proxy` and `coredns` of `kubeadm init phase addon`. If this flag is `usedkubeadm` will not apply a given addon and instead print to the terminal the API objects that will be applied. ([#109995](https://github.com/kubernetes/kubernetes/pull/109995), [@wangyysde](https://github.com/wangyysde))
- Kubeadm: enhanced the "patches" functionality to be able to patch kubelet config files containing `v1beta1.KubeletConfiguration`. The new patch target is called `kubeletconfiguration` (e.g. patch file `kubeletconfiguration+json.json`).This makes it possible to apply node specific KubeletConfiguration options during `init`, `join` and `upgrade`, while the main `KubeletConfiguration` that is passed to `init` as part of the `--config` file can still act as the global stored in the cluster `KubeletConfiguration`. ([#110405](https://github.com/kubernetes/kubernetes/pull/110405), [@neolit123](https://github.com/neolit123))
- Kubeadm: make sure the etcd static pod startup probe uses /health?serializable=false while the liveness probe uses /health?serializable=true&exclude=NOSPACE. The NOSPACE exclusion would allow administrators to address space issues one member at a time. ([#110744](https://github.com/kubernetes/kubernetes/pull/110744), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: modified the etcd static Pod `liveness` and `readiness` probes to use a new etcd `v3.5.3+` HTTP(s) health check endpoint `/health?serializable=true` that allows to track the health of individual etcd members and not fail all members if a single member is not healthy in the etcd cluster. ([#110072](https://github.com/kubernetes/kubernetes/pull/110072), [@neolit123](https://github.com/neolit123))
- Kubeadm: support experimental JSON/YAML output for `kubeadm upgrade plan` with the `--output` flag. ([#108447](https://github.com/kubernetes/kubernetes/pull/108447), [@pacoxu](https://github.com/pacoxu))
- Kubeadm: the preferred pod anti-affinity for CoreDNS is now enabled by default. ([#110593](https://github.com/kubernetes/kubernetes/pull/110593), [@SataQiu](https://github.com/SataQiu))
- Kubectl: support multiple resources for kubectl rollout status. ([#108777](https://github.com/kubernetes/kubernetes/pull/108777), [@pjo256](https://github.com/pjo256))
- Kubernetes is now built with Golang 1.18.2. ([#110043](https://github.com/kubernetes/kubernetes/pull/110043), [@cpanato](https://github.com/cpanato))
- Kubernetes is now built with Golang 1.18.3 ([#110421](https://github.com/kubernetes/kubernetes/pull/110421), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with Golang 1.19.0. ([#111679](https://github.com/kubernetes/kubernetes/pull/111679), [@puerco](https://github.com/puerco))
- Lock CSIMigrationAzureDisk feature gate to default. ([#110491](https://github.com/kubernetes/kubernetes/pull/110491), [@andyzhangx](https://github.com/andyzhangx))
- Metric `running_managed_controllers` is enabled for Cloud Node Lifecycle controller. ([#111033](https://github.com/kubernetes/kubernetes/pull/111033), [@jprzychodzen](https://github.com/jprzychodzen))
- Metric `running_managed_controllers` is enabled for Node IPAM controller in KCM. ([#111466](https://github.com/kubernetes/kubernetes/pull/111466), [@jprzychodzen](https://github.com/jprzychodzen))
- Metric `running_managed_controllers` is enabled for Route,Service and Cloud Node controllers in KCM and CCM. ([#111462](https://github.com/kubernetes/kubernetes/pull/111462), [@jprzychodzen](https://github.com/jprzychodzen))
- New `KUBECACHEDIR` environment variable was introduced to override default discovery cache directory which is `$HOME/.kube/cache`. ([#109479](https://github.com/kubernetes/kubernetes/pull/109479), [@ardaguclu](https://github.com/ardaguclu))
- Pod SecurityContext and PodSecurityPolicy supports slash as sysctl separator. ([#106834](https://github.com/kubernetes/kubernetes/pull/106834), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps, Architecture, Auth, Node, Security and Testing]
- Promoted LocalStorageCapacityIsolationFSQuotaMonitoring to beta. ([#107329](https://github.com/kubernetes/kubernetes/pull/107329), [@pacoxu](https://github.com/pacoxu))
- Promoted the `CSIMigrationPortworx` feature gate to Beta. ([#110411](https://github.com/kubernetes/kubernetes/pull/110411), [@trierra](https://github.com/trierra))
- Return a warning when applying a `pod-security.kubernetes.io` label to a PodSecurity-exempted namespace.
  Stop including the `pod-security.kubernetes.io/exempt=namespace` audit annotation on namespace requests. ([#109680](https://github.com/kubernetes/kubernetes/pull/109680), [@tallclair](https://github.com/tallclair))
- The  new flag `etcd-ready-timeout` has been added. It configures a timeout of an additional etcd check performed as part of readyz check. ([#111399](https://github.com/kubernetes/kubernetes/pull/111399), [@Argh4k](https://github.com/Argh4k))
- The TopologySpreadConstraints will be shown in describe command for pods, deployments, daemonsets, etc. ([#109563](https://github.com/kubernetes/kubernetes/pull/109563), [@ardaguclu](https://github.com/ardaguclu))
- The `kubectl diff` changed to ignore managed fields by default, and a new --show-managed-fields flag has been added to allow you to include managed fields in the diff. ([#111319](https://github.com/kubernetes/kubernetes/pull/111319), [@brianpursley](https://github.com/brianpursley))
- The beta feature `ServiceIPStaticSubrange` is now enabled by default. ([#110703](https://github.com/kubernetes/kubernetes/pull/110703), [@aojea](https://github.com/aojea))
- Updated base image for Windows pause container images to one built on Windows machines to address limitations of building Windows container images on Linux machines. ([#110379](https://github.com/kubernetes/kubernetes/pull/110379), [@marosset](https://github.com/marosset))
- Updated cAdvisor to v0.45.0. ([#111647](https://github.com/kubernetes/kubernetes/pull/111647), [@bobbypage](https://github.com/bobbypage))
- Updated debian-base, debian-iptables, and setcap images:
  - debian-base:bullseye-v1.3.0
  - debian-iptables:bullseye-v1.4.0
  - setcap:bullseye-v1.3.0 ([#110558](https://github.com/kubernetes/kubernetes/pull/110558), [@wespanther](https://github.com/wespanther))
- When using the OpenStack legacy cloud provider, kubelet and KCM will ignore unknown configuration directives rather than failing to start. ([#109709](https://github.com/kubernetes/kubernetes/pull/109709), [@mdbooth](https://github.com/mdbooth))
- ` JobTrackingWithFinalizers` enabled by default. This feature allows to keep track of the Job progress without relying on Pods staying in the apiserver.
   ([#110948](https://github.com/kubernetes/kubernetes/pull/110948), [@alculquicondor](https://github.com/alculquicondor))
- `CSIMigrationAWS` upgraded to GA and locked to true.
   ([#111479](https://github.com/kubernetes/kubernetes/pull/111479), [@wongma7](https://github.com/wongma7))
- `CSIMigrationGCE` upgraded to GA and locked to true.
   ([#111301](https://github.com/kubernetes/kubernetes/pull/111301), [@mattcary](https://github.com/mattcary))
- `CSIMigrationvSphere` feature is now enabled by default.
   ([#103523](https://github.com/kubernetes/kubernetes/pull/103523), [@divyenpatel](https://github.com/divyenpatel))
- `MaxUnavailable` for `StatefulSets`, allows faster `RollingUpdate` by taking down more than 1 pod at a time.
  The number of pods you want to take down during a `RollingUpdate` is configurable using `maxUnavailable` parameter.
   ([#109251](https://github.com/kubernetes/kubernetes/pull/109251), [@krmayankk](https://github.com/krmayankk))
- The `gcp` and `azure` auth plugins have been restored to client-go and kubectl until https://issue.k8s.io/111911 is resolved in supported kubectl minor versions. ([#111918](https://github.com/kubernetes/kubernetes/pull/111918), [@liggitt](https://github.com/liggitt))

### Documentation

- EndpointSlices with Pod referencing Nodes that don't exist couldn't be created or updated. The behavior on the EndpointSlice controller has been modified to update the EndpointSlice without the Pods that reference non-existing Nodes and keep retrying until all Pods reference existing Nodes. However, if `service.Spec.PublishNotReadyAddresses` is set, all the Pods are published without retrying. Fixed EndpointSlices metrics to reflect correctly the number of desired EndpointSlices when no endpoints are present. ([#110639](https://github.com/kubernetes/kubernetes/pull/110639), [@aojea](https://github.com/aojea))
- Optimization of kubectl Chinese translation ([#110538](https://github.com/kubernetes/kubernetes/pull/110538), [@hwdef](https://github.com/hwdef)) [SIG CLI]

### Failing Test

- E2e tests: fixed bug in the e2e image `agnhost:2.38` which hangs instead of exiting if a `SIGTERM` signal is received and the `shutdown-delay` option is `0`. ([#110214](https://github.com/kubernetes/kubernetes/pull/110214), [@aojea](https://github.com/aojea))
- In-tree GCE PD test cases no longer run in Kubernetes testing harness anymore (side effect of switching on CSI migration in 1.22). Please switch on the environment variable `ENABLE_STORAGE_GCE_PD_DRIVER` to `yes` if you need to run these tests. ([#109541](https://github.com/kubernetes/kubernetes/pull/109541), [@dims](https://github.com/dims))

### Bug or Regression

- A bug was fixed where a job sync was not retried in case of a transient ResourceQuota conflict. ([#111026](https://github.com/kubernetes/kubernetes/pull/111026), [@alculquicondor](https://github.com/alculquicondor))
- A change of a failed job condition status to `False` does not result in duplicate  conditions. ([#110292](https://github.com/kubernetes/kubernetes/pull/110292), [@mimowo](https://github.com/mimowo))
- Added error message "dry-run can not be used when --force is set" when dry-run and force flags are set in replace command. ([#110326](https://github.com/kubernetes/kubernetes/pull/110326), [@ardaguclu](https://github.com/ardaguclu))
- Added the latest GCE pinhole firewall feature, which introduces `destination-ranges` in the ingress `firewall-rules`. It restricts access to the backend IPs by allowing traffic through `ILB` or `NetLB` only. This change does **NOT** change the existing `ILB` or `NetLB` behavior. ([#109510](https://github.com/kubernetes/kubernetes/pull/109510), [@sugangli](https://github.com/sugangli))
- Allow expansion of ephemeral volumes ([#109987](https://github.com/kubernetes/kubernetes/pull/109987), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Apiserver: fixed audit of loading more than one webhooks. ([#110145](https://github.com/kubernetes/kubernetes/pull/110145), [@sxllwx](https://github.com/sxllwx))
- Bug fix in test/e2e/framework  Framework.RecordFlakeIfError ([#111048](https://github.com/kubernetes/kubernetes/pull/111048), [@alingse](https://github.com/alingse)) [SIG Testing]
- Client-go: fixed an error in the fake client when creating API requests are submitted to subresources like `pods/eviction`. ([#110425](https://github.com/kubernetes/kubernetes/pull/110425), [@LY-today](https://github.com/LY-today))
- Do not raise an error when setting a label with the same value, ignore it. ([#105936](https://github.com/kubernetes/kubernetes/pull/105936), [@zigarn](https://github.com/zigarn))
- Do not report terminated container metrics ([#110950](https://github.com/kubernetes/kubernetes/pull/110950), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Node]
- EndpointSlices marked for deletion are now ignored during reconciliation. ([#109624](https://github.com/kubernetes/kubernetes/pull/109624), [@aryan9600](https://github.com/aryan9600))
- Etcd: Update to v3.5.4 ([#110033](https://github.com/kubernetes/kubernetes/pull/110033), [@mk46](https://github.com/mk46)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Faster mount detection for linux kernel 5.10+ using openat2 speeding up pod churn rates. On Kernel versions less 5.10, it will fallback to using the original way of detecting mount points i.e by parsing /proc/mounts. ([#109217](https://github.com/kubernetes/kubernetes/pull/109217), [@manugupt1](https://github.com/manugupt1))
- FibreChannel volume plugin may match the wrong device and wrong associated devicemapper parent. This may cause a disater that pods attach wrong disks. ([#110719](https://github.com/kubernetes/kubernetes/pull/110719), [@xakdwch](https://github.com/xakdwch))
- Fix a bug where CRI implementations that use cAdvisor stats provider (CRI-O) don't evict pods when their logs exceed ephemeral storage limit. ([#108115](https://github.com/kubernetes/kubernetes/pull/108115), [@haircommander](https://github.com/haircommander)) [SIG Node]
- Fix a bug where metrics are not recorded during Preemption(PostFilter). ([#108727](https://github.com/kubernetes/kubernetes/pull/108727), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Fix a data race in authentication between AuthenticatedGroupAdder and cached token authenticator. ([#109969](https://github.com/kubernetes/kubernetes/pull/109969), [@sttts](https://github.com/sttts)) [SIG API Machinery and Auth]
- Fix bug that prevented the job controller from enforcing activeDeadlineSeconds when set. ([#110294](https://github.com/kubernetes/kubernetes/pull/110294), [@harshanarayana](https://github.com/harshanarayana))
- Fix for volume reconstruction of CSI ephemeral volumes ([#108997](https://github.com/kubernetes/kubernetes/pull/108997), [@dobsonj](https://github.com/dobsonj)) [SIG Node, Storage and Testing]
- Fix incorrectly report scope for request_duration_seconds and request_slo_duration_seconds metrics for POST custom resources API calls. ([#110009](https://github.com/kubernetes/kubernetes/pull/110009), [@azylinski](https://github.com/azylinski)) [SIG Instrumentation]
- Fix printing resources with int64 fields ([#110408](https://github.com/kubernetes/kubernetes/pull/110408), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fix spurious kube-apiserver log warnings related to openapi v3 merging when creating or modifying CustomResourceDefinition objects ([#109880](https://github.com/kubernetes/kubernetes/pull/109880), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Testing]
- Fix the bug that reported incorrect metrics for the cluster IP allocator. ([#110027](https://github.com/kubernetes/kubernetes/pull/110027), [@tksm](https://github.com/tksm))
- Fixed JobTrackingWithFinalizers when a pod succeeds after the job is considered failed, which led to API conflicts that blocked finishing the job. ([#111646](https://github.com/kubernetes/kubernetes/pull/111646), [@alculquicondor](https://github.com/alculquicondor))
- Fixed `JobTrackingWithFinalizers` that:
  - was declaring a job finished before counting all the created pods in the status
  - was leaving pods with finalizers, blocking pod and job deletions
   `JobTrackingWithFinalizers` is still disabled by default. ([#109486](https://github.com/kubernetes/kubernetes/pull/109486), [@alculquicondor](https://github.com/alculquicondor))
- Fixed `NeedResize` build failure on Windows. ([#109721](https://github.com/kubernetes/kubernetes/pull/109721), [@andyzhangx](https://github.com/andyzhangx))
- Fixed a bug in `kubectl` that caused the wrong result length when using `--chunk-size` and `--selector` together. ([#110652](https://github.com/kubernetes/kubernetes/pull/110652), [@Abirdcfly](https://github.com/Abirdcfly))
- Fixed a bug involving Services of type `LoadBalancer` with multiple IPs and a `LoadBalancerSourceRanges` that overlaps the node IP. ([#109826](https://github.com/kubernetes/kubernetes/pull/109826), [@danwinship](https://github.com/danwinship))
- Fixed a bug which could have allowed an improperly annotated LoadBalancer service to become active. ([#109601](https://github.com/kubernetes/kubernetes/pull/109601), [@mdbooth](https://github.com/mdbooth))
- Fixed a kubelet issue that could result in invalid pod status updates to be sent to the api-server where pods would be reported in a terminal phase but also report a ready condition of true in some cases. ([#110256](https://github.com/kubernetes/kubernetes/pull/110256), [@bobbypage](https://github.com/bobbypage))
- Fixed an issue on Windows nodes where `HostProcess` containers may not be created as expected. ([#110140](https://github.com/kubernetes/kubernetes/pull/110140), [@marosset](https://github.com/marosset))
- Fixed bug where CSI migration doesn't count inline volumes for attach limit. ([#107787](https://github.com/kubernetes/kubernetes/pull/107787), [@Jiawei0227](https://github.com/Jiawei0227))
- Fixed error "dbus: connection closed by user" after dbus daemon restarts. ([#110496](https://github.com/kubernetes/kubernetes/pull/110496), [@kolyshkin](https://github.com/kolyshkin))
- Fixed image pull failure when `IMDS` is unavailable in kubelet startup. ([#110523](https://github.com/kubernetes/kubernetes/pull/110523), [@andyzhangx](https://github.com/andyzhangx))
- Fixed memory leak in the job controller related to `JobTrackingWithFinalizers`. ([#111721](https://github.com/kubernetes/kubernetes/pull/111721), [@alculquicondor](https://github.com/alculquicondor))
- Fixed mounting of iSCSI volumes over IPv6 networks. ([#110688](https://github.com/kubernetes/kubernetes/pull/110688), [@jsafrane](https://github.com/jsafrane))
- Fixed performance issue when creating large objects using SSA with fully unspecified schemas ( preserveUnknownFields ). ([#111557](https://github.com/kubernetes/kubernetes/pull/111557), [@alexzielenski](https://github.com/alexzielenski))
- Fixed s.RuntimeCgroups error condition and Fixed possible wrong log print. ([#110648](https://github.com/kubernetes/kubernetes/pull/110648), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- Fixed scheduling of **CronJob** with `@every X` schedules. ([#109250](https://github.com/kubernetes/kubernetes/pull/109250), [@d-honeybadger](https://github.com/d-honeybadger))
- Fixed strict server-side field validation treating `metadata` fields as unknown fields. ([#109268](https://github.com/kubernetes/kubernetes/pull/109268), [@liggitt](https://github.com/liggitt))
- Fixed the GCE firewall update when the destination IPs are changing so that firewalls reflect the IP updates of the LBs. ([#111186](https://github.com/kubernetes/kubernetes/pull/111186), [@sugangli](https://github.com/sugangli))
- Fixed the bug that a `ServiceIPStaticSubrange` enabled cluster assigns duplicate IP addresses when the dynamic block is exhausted. ([#109928](https://github.com/kubernetes/kubernetes/pull/109928), [@tksm](https://github.com/tksm))
- For scheduler plugin developers: the scheduler framework's shared PodInformer is now initialized with empty indexers. This enables scheduler plugins to add their extra indexers. Note that only non-conflict indexers are allowed to be added. ([#110663](https://github.com/kubernetes/kubernetes/pull/110663), [@fromanirh](https://github.com/fromanirh)) [SIG Scheduling]
- If the parent directory of the file specified in the `--audit-log-path` argument does not exist, Kubernetes now creates it. ([#110813](https://github.com/kubernetes/kubernetes/pull/110813), [@vpnachev](https://github.com/vpnachev)) [SIG Auth]
- Informer/reflector callers can now catch and unwrap specific API errors by type. ([#110076](https://github.com/kubernetes/kubernetes/pull/110076), [@karlkfi](https://github.com/karlkfi))
- Kube-apiserver: Get, GetList and Watch requests that should be served by the apiserver cacher during shutdown will be rejected to avoid a deadlock situation leaving requests hanging. ([#108414](https://github.com/kubernetes/kubernetes/pull/108414), [@aojea](https://github.com/aojea))
- Kubeadm: Fixed duplicate `unix://` prefix in node annotation. ([#110656](https://github.com/kubernetes/kubernetes/pull/110656), [@pacoxu](https://github.com/pacoxu))
- Kubeadm: a bug was fixed due to which configurable `KubernetesVersion` was not being respected during kubeadm join. ([#110791](https://github.com/kubernetes/kubernetes/pull/110791), [@SataQiu](https://github.com/SataQiu))
- Kubeadm: enabled the --experimental-watch-progress-notify-interval flag for etcd and set it to 5s. The flag specifies an interval at which etcd sends watch data to the kube-apiserver. ([#111383](https://github.com/kubernetes/kubernetes/pull/111383), [@p0lyn0mial](https://github.com/p0lyn0mial))
- Kubeadm: now sets the host `OS` environment variables when executing `crictl` during image pulls. This fixed a bug where `*PROXY` environment variables did not affect `crictl` internet connectivity. ([#110134](https://github.com/kubernetes/kubernetes/pull/110134), [@mk46](https://github.com/mk46))
- Kubeadm: only taint control plane nodes when the legacy "master" taint is present. This avoids the bug where "kubeadm upgrade" will re-taint a control plane  node with the new "control plane" taint even if the user explicitly untainted the node. ([#109840](https://github.com/kubernetes/kubernetes/pull/109840), [@neolit123](https://github.com/neolit123))
- Kubeadm: respect user specified image repository when using Kubernetes ci version ([#111017](https://github.com/kubernetes/kubernetes/pull/111017), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: support retry mechanism for removing container in reset phase ([#110837](https://github.com/kubernetes/kubernetes/pull/110837), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet: added log for volume metric collection taking too long ([#107490](https://github.com/kubernetes/kubernetes/pull/107490), [@pacoxu](https://github.com/pacoxu))
- Kubelet: added retry of checking Unix domain sockets on Windows nodes for the plugin registration mechanism. ([#110075](https://github.com/kubernetes/kubernetes/pull/110075), [@luckerby](https://github.com/luckerby))
- Kubelet: added validation for labels provided with --node-labels. Malformed labels will result in errors. ([#109263](https://github.com/kubernetes/kubernetes/pull/109263), [@FeLvi-zzz](https://github.com/FeLvi-zzz))
- Kubelet: wait for node allocatable ephemeral-storage data. ([#101882](https://github.com/kubernetes/kubernetes/pull/101882), [@jackfrancis](https://github.com/jackfrancis))
- Kubernetes now correctly handles "search ." in the host's resolv.conf file by preserving the "." entry in the "resolv.conf" that the kubelet writes to pods. ([#109441](https://github.com/kubernetes/kubernetes/pull/109441), [@Miciah](https://github.com/Miciah)) [SIG Network and Node]
- Made usage of key encipherment optional in API validation. ([#111061](https://github.com/kubernetes/kubernetes/pull/111061), [@pacoxu](https://github.com/pacoxu))
- ManagedFields time is correctly updated when the value of a managed field is modified. ([#110058](https://github.com/kubernetes/kubernetes/pull/110058), [@glebiller](https://github.com/glebiller))
- OpenAPI will no longer duplicate these schemas:
  - `io.k8s.apimachinery.pkg.apis.meta.v1.DeleteOptions_v2`
  - `io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta_v2`
  - `io.k8s.apimachinery.pkg.apis.meta.v1.OwnerReference_v2`
  - `io.k8s.apimachinery.pkg.apis.meta.v1.StatusDetails_v2`
  - `io.k8s.apimachinery.pkg.apis.meta.v1.Status_v2` ([#110179](https://github.com/kubernetes/kubernetes/pull/110179), [@Jefftree](https://github.com/Jefftree))
- Panics while calling validating admission webhook are caught and honor the fail open or fail closed setting. ([#108746](https://github.com/kubernetes/kubernetes/pull/108746), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Pods now post their `readiness` during termination. ([#110191](https://github.com/kubernetes/kubernetes/pull/110191), [@rphillips](https://github.com/rphillips))
- Reduced duration to sync proxy rules on Windows `kube-proxy` when using `kernelspace` mode. ([#109124](https://github.com/kubernetes/kubernetes/pull/109124), [@daschott](https://github.com/daschott))
- Reduced the number of cloud API calls and service downtime caused by excessive re-configurations of cluster LBs with externalTrafficPolicy=Local when node readiness changes (https://github.com/kubernetes/kubernetes/issues/111539). The service controller (in cloud-controller-manager) will avoid resyncing nodes which are transitioning between `Ready` / `NotReady` (only for for ETP=Local Services).  The LBs used for these services will solely rely on the health check probe defined by the `healthCheckNodePort` to determine if a particular node is to be used for traffic load balancing. ([#109706](https://github.com/kubernetes/kubernetes/pull/109706), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu))
- Removed the recently re-introduced schedulability predicate ([#109706](https://github.com/kubernetes/kubernetes/pull/109706)) as to not have unschedulable nodes removed from load balancers back-end pools. ([#111691](https://github.com/kubernetes/kubernetes/pull/111691), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu))
- Removed unused flags from `kubectl run` command. ([#110668](https://github.com/kubernetes/kubernetes/pull/110668), [@brianpursley](https://github.com/brianpursley))
- Run kubelet, when there is an error exit, print the error log. ([#110691](https://github.com/kubernetes/kubernetes/pull/110691), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- The `priority_level_request_utilization` metric histogram is adjusted so that for the cases where `phase=waiting` the denominator is the cumulative capacity of all of the priority level's queues.
   The `read_vs_write_current_requests` metric histogram is adjusted, in the case of using API Priority and Fairness instead of max-in-flight, to divide by the relevant limit: sum of queue capacities for waiting requests, sum of seat limits for executing requests. ([#110164](https://github.com/kubernetes/kubernetes/pull/110164), [@MikeSpreitzer](https://github.com/MikeSpreitzer))
- The commands `kubeadm certs renew` and `kubeadm certs check-expiration` now honor the `cert-dir` flag on a running Kubernetes cluster. ([#110709](https://github.com/kubernetes/kubernetes/pull/110709), [@chendave](https://github.com/chendave))
- The kube-proxy `sync_proxy_rules_no_endpoints_total` metric now only counts local-traffic-policy services which have remote endpoints but not local endpoints. ([#109782](https://github.com/kubernetes/kubernetes/pull/109782), [@danwinship](https://github.com/danwinship))
- The namespace editors and admins can now create leases.coordination.k8s.io and should use this type for leaderelection instead of configmaps. ([#111472](https://github.com/kubernetes/kubernetes/pull/111472), [@deads2k](https://github.com/deads2k))
- The node annotation alpha.kubernetes.io/provided-node-ip is no longer set ONLY when `--cloud-provider=external`.  Now, it is set on kubelet startup if the `--cloud-provider` flag is set at all, including the deprecated in-tree providers. ([#109794](https://github.com/kubernetes/kubernetes/pull/109794), [@mdbooth](https://github.com/mdbooth)) [SIG Network and Node]
- The pod phase lifecycle guarantees that terminal Pods, those whose states are `Unready` or `Succeeded`, can not regress and will have all container stopped. Hence, terminal Pods will never be reachable and should not publish their IP addresses on the `Endpoints` or `EndpointSlices`, independently of the Service `TolerateUnready` option. ([#110255](https://github.com/kubernetes/kubernetes/pull/110255), [@robscott](https://github.com/robscott))
- Unmounted volumes correctly for reconstructed volumes even if mount operation fails after kubelet restart. ([#110670](https://github.com/kubernetes/kubernetes/pull/110670), [@gnufied](https://github.com/gnufied))
- Updated max azure data disk count map with new VM types. ([#111406](https://github.com/kubernetes/kubernetes/pull/111406), [@bennerv](https://github.com/bennerv))
- Updated to cAdvisor v0.44.1 to fix an issue where metrics generated by kubelet for pod network stats were empty in some cases. ([#109658](https://github.com/kubernetes/kubernetes/pull/109658), [@bobbypage](https://github.com/bobbypage))
- Upgraded Azure/go-autorest/autorest to v0.11.27. ([#110371](https://github.com/kubernetes/kubernetes/pull/110371), [@andyzhangx](https://github.com/andyzhangx))
- Upgraded functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv4.5.7. ([#111606](https://github.com/kubernetes/kubernetes/pull/111606), [@natasha41575](https://github.com/natasha41575))
- Use checksums instead of fsyncs to ensure discovery cache integrity ([#110851](https://github.com/kubernetes/kubernetes/pull/110851), [@negz](https://github.com/negz)) [SIG API Machinery]
- UserName check for 'ContainerAdministrator' is now case-insensitive if runAsNonRoot is set to true on Windows. ([#111009](https://github.com/kubernetes/kubernetes/pull/111009), [@marosset](https://github.com/marosset))
- Volumes are no longer detached from healthy nodes after 6 minutes timeout. 6 minute force-detach timeout is used only for unhealthy nodes (`node.status.conditions["Ready"]!= true`). ([#110721](https://github.com/kubernetes/kubernetes/pull/110721), [@jsafrane](https://github.com/jsafrane))
- When metrics are counted, discard the wrong container StartTime metrics ([#110880](https://github.com/kubernetes/kubernetes/pull/110880), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Instrumentation and Node]
- Windows kubelet plugin Watcher now working as intended. ([#111439](https://github.com/kubernetes/kubernetes/pull/111439), [@claudiubelu](https://github.com/claudiubelu))
- [aws] Fixed a bug which reduces the number of unnecessary calls to STS in the event of assume role failures in the legacy cloud provider ([#110706](https://github.com/kubernetes/kubernetes/pull/110706), [@prateekgogia](https://github.com/prateekgogia)) [SIG Cloud Provider]
- `pod.Spec.RuntimeClassName` field is now available in kubectl describe command. ([#110914](https://github.com/kubernetes/kubernetes/pull/110914), [@yeahdongcn](https://github.com/yeahdongcn))

### Other (Cleanup or Flake)

- Add missing powershell option to kubectl completion command short description. ([#109773](https://github.com/kubernetes/kubernetes/pull/109773), [@danielhelfand](https://github.com/danielhelfand))
- Added e2e test flag to specify which volume drivers should be installed. This  deprecated the ENABLE_STORAGE_GCE_PD_DRIVER environment variable. ([#111481](https://github.com/kubernetes/kubernetes/pull/111481), [@mattcary](https://github.com/mattcary))
- Changed PV framework delete timeout to 5 minutes as documented. ([#109764](https://github.com/kubernetes/kubernetes/pull/109764), [@saikat-royc](https://github.com/saikat-royc))
- Default burst limit for the discovery client set to 300. ([#109141](https://github.com/kubernetes/kubernetes/pull/109141), [@ulucinar](https://github.com/ulucinar))
- Deleted the `apimachinery/clock` package. Please use `k8s.io/utils/clock` package instead. ([#109752](https://github.com/kubernetes/kubernetes/pull/109752), [@MadhavJivrajani](https://github.com/MadhavJivrajani))
- Feature gates that graduated to GA in 1.23 or earlier and were unconditionally enabled have been removed: CSIServiceAccountToken, ConfigurableFSGroupPolicy, EndpointSlice, EndpointSliceNodeName, EndpointSliceProxying, GenericEphemeralVolume, IPv6DualStack, IngressClassNamespacedParams, StorageObjectInUseProtection, TTLAfterFinished, VolumeSubpath, WindowsEndpointSliceProxying. ([#109435](https://github.com/kubernetes/kubernetes/pull/109435), [@pohly](https://github.com/pohly))
- For Linux, `kube-proxy` uses a new “distroless” container image, instead of an image based on Debian. ([#111060](https://github.com/kubernetes/kubernetes/pull/111060), [@aojea](https://github.com/aojea))
- For resources built into an apiserver, the server now logs at `-v=3` whether it is using watch caching. ([#109175](https://github.com/kubernetes/kubernetes/pull/109175), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery]
- GlusterFS provisioner (`kubernetes.io/glusterfs`) has been deprecated in this release. ([#111485](https://github.com/kubernetes/kubernetes/pull/111485), [@humblec](https://github.com/humblec))
- Improved `kubectl run` and `kubectl debug` error messages upon attaching failures. ([#110764](https://github.com/kubernetes/kubernetes/pull/110764), [@soltysh](https://github.com/soltysh))
- In the event that more than one IngressClass is designated "default", the DefaultIngressClass admission controller will choose one rather than fail. ([#110974](https://github.com/kubernetes/kubernetes/pull/110974), [@kidddddddddddddddddddddd](https://github.com/kidddddddddddddddddddddd)) [SIG Network]
- Kube-proxy: The "userspace" proxy-mode is deprecated on Linux and Windows, despite being the default on Windows.  As of v1.26, the default mode for Windows will change to 'kernelspace'. ([#110762](https://github.com/kubernetes/kubernetes/pull/110762), [@pandaamanda](https://github.com/pandaamanda)) [SIG Network]
- Kubeadm: perform additional dockershim cleanup. Treat all container runtimes as remote by using the flag "--container-runtime=remote", given dockershim was removed in 1.24 and given kubeadm 1.25 supports a kubelet version of 1.24 and 1.25. The flag "--network-plugin" will no longer be used for new clusters. Stop cleaning up the following dockershim related directories on "kubeadm reset": "/var/lib/dockershim", "/var/runkubernetes", "/var/lib/cni" ([#110022](https://github.com/kubernetes/kubernetes/pull/110022), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubelet's deprecated `--experimental-kernel-memcg-notification` flag is now removed.  Use `--kernel-memcg-notification` instead. ([#109388](https://github.com/kubernetes/kubernetes/pull/109388), [@ialidzhikov](https://github.com/ialidzhikov))
- Kubelet: Silenced flag output on errors. ([#110728](https://github.com/kubernetes/kubernetes/pull/110728), [@howardjohn](https://github.com/howardjohn))
- Kubernetes binaries are now built-in `module` mode instead of `GOPATH` mode. ([#109464](https://github.com/kubernetes/kubernetes/pull/109464), [@liggitt](https://github.com/liggitt))
- Removed branch `release-1.20` from prom bot due to EOL. ([#110748](https://github.com/kubernetes/kubernetes/pull/110748), [@cpanato](https://github.com/cpanato))
- Removed deprecated kubectl.kubernetes.io/default-logs-container support ([#109254](https://github.com/kubernetes/kubernetes/pull/109254), [@pacoxu](https://github.com/pacoxu))
- Renamed `apiserver_watch_cache_watch_cache_initializations_total` to `apiserver_watch_cache_initializations_total` ([#109579](https://github.com/kubernetes/kubernetes/pull/109579), [@logicalhan](https://github.com/logicalhan))
- Shell completion is now provided for the "--subresource" flag. ([#109070](https://github.com/kubernetes/kubernetes/pull/109070), [@marckhouzam](https://github.com/marckhouzam))
- Some apiserver metrics were changed, as follows.
  - `priority_level_seat_count_samples` is replaced with `priority_level_seat_utilization`, which samples every nanosecond rather than every millisecond; the old metric conveyed utilization despite its name.
  - `priority_level_seat_count_watermarks` is removed.
  - `priority_level_request_count_samples` is replaced with `priority_level_request_utilization`, which samples every nanosecond rather than every millisecond; the old metric conveyed utilization despite its name.
  - `priority_level_request_count_watermarks` is removed.
  - `read_vs_write_request_count_samples` is replaced with `read_vs_write_current_requests`, which samples every nanosecond rather than every second; the new metric, like the old one, measures utilization when the max-in-flight filter is used and number of requests when the API Priority and Fairness filter is used.
  - `read_vs_write_request_count_watermarks` is removed. ([#110104](https://github.com/kubernetes/kubernetes/pull/110104), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery, Instrumentation and Testing]
- The kube-controller-manager's deprecated `--experimental-cluster-signing-duration` flag is now removed. Adapt your machinery to use the `--cluster-signing-duration` flag that is available since v1.19. ([#108476](https://github.com/kubernetes/kubernetes/pull/108476), [@ialidzhikov](https://github.com/ialidzhikov))
- The kube-scheduler ComponentConfig v1beta2 is deprecated in v1.25. ([#111547](https://github.com/kubernetes/kubernetes/pull/111547), [@kerthcet](https://github.com/kerthcet))
- The kubelet no longer supports collecting accelerator metrics through cAdvisor. The feature gate `DisableAcceleratorUsageMetrics` is now GA and cannot be disabled. ([#110940](https://github.com/kubernetes/kubernetes/pull/110940), [@pacoxu](https://github.com/pacoxu))
- Updated cri-tools to [v1.24.2(https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.24.2). ([#109813](https://github.com/kubernetes/kubernetes/pull/109813), [@saschagrunert](https://github.com/saschagrunert))
- `apiserver_dropped_requests` is dropped from this release since `apiserver_request_total` can now be used to track dropped requests. `etcd_object_counts` is also removed in favor of `apiserver_storage_objects`. `apiserver_registered_watchers` is also removed in favor of `apiserver_longrunning_requests`. ([#110337](https://github.com/kubernetes/kubernetes/pull/110337), [@logicalhan](https://github.com/logicalhan))
- `apiserver_longrunning_gauge` was removed from the codebase. Please use `apiserver_longrunning_requests`
  instead.
   ([#110310](https://github.com/kubernetes/kubernetes/pull/110310), [@logicalhan](https://github.com/logicalhan))


## Dependencies

### Added
- github.com/emicklei/go-restful/v3: [v3.8.0](https://github.com/emicklei/go-restful/v3/tree/v3.8.0)
- github.com/go-task/slim-sprig: [348f09d](https://github.com/go-task/slim-sprig/tree/348f09d)
- github.com/gogo/googleapis: [v1.4.1](https://github.com/gogo/googleapis/tree/v1.4.1)
- github.com/golang-jwt/jwt/v4: [v4.2.0](https://github.com/golang-jwt/jwt/v4/tree/v4.2.0)
- github.com/golang/snappy: [v0.0.3](https://github.com/golang/snappy/tree/v0.0.3)
- github.com/golangplus/bytes: [v1.0.0](https://github.com/golangplus/bytes/tree/v1.0.0)
- github.com/golangplus/fmt: [v1.0.0](https://github.com/golangplus/fmt/tree/v1.0.0)
- github.com/onsi/ginkgo/v2: [v2.1.4](https://github.com/onsi/ginkgo/v2/tree/v2.1.4)
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.20.0
- go.opentelemetry.io/contrib/propagators: v0.20.0
- google.golang.org/grpc/cmd/protoc-gen-go-grpc: v1.1.0

### Changed
- bitbucket.org/bertimus9/systemstat: 0eeff89 → v0.5.0
- cloud.google.com/go: v0.81.0 → v0.97.0
- github.com/Azure/go-autorest/autorest/adal: [v0.9.13 → v0.9.20](https://github.com/Azure/go-autorest/autorest/adal/compare/v0.9.13...v0.9.20)
- github.com/Azure/go-autorest/autorest/mocks: [v0.4.1 → v0.4.2](https://github.com/Azure/go-autorest/autorest/mocks/compare/v0.4.1...v0.4.2)
- github.com/Azure/go-autorest/autorest: [v0.11.18 → v0.11.27](https://github.com/Azure/go-autorest/autorest/compare/v0.11.18...v0.11.27)
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [ea6160c → f118173](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/ea6160c...f118173)
- github.com/MakeNowJust/heredoc: [bb23615 → v1.0.0](https://github.com/MakeNowJust/heredoc/compare/bb23615...v1.0.0)
- github.com/antlr/antlr4/runtime/Go/antlr: [b48c857 → f25a4f6](https://github.com/antlr/antlr4/runtime/Go/antlr/compare/b48c857...f25a4f6)
- github.com/chai2010/gettext-go: [c6fed77 → v1.0.2](https://github.com/chai2010/gettext-go/compare/c6fed77...v1.0.2)
- github.com/cncf/udpa/go: [5459f2c → 04548b0](https://github.com/cncf/udpa/go/compare/5459f2c...04548b0)
- github.com/cncf/xds/go: [fbca930 → cb28da3](https://github.com/cncf/xds/go/compare/fbca930...cb28da3)
- github.com/container-storage-interface/spec: [v1.5.0 → v1.6.0](https://github.com/container-storage-interface/spec/compare/v1.5.0...v1.6.0)
- github.com/containerd/containerd: [v1.4.12 → v1.4.9](https://github.com/containerd/containerd/compare/v1.4.12...v1.4.9)
- github.com/coredns/corefile-migration: [v1.0.14 → v1.0.17](https://github.com/coredns/corefile-migration/compare/v1.0.14...v1.0.17)
- github.com/daviddengcn/go-colortext: [511bcaf → v1.0.0](https://github.com/daviddengcn/go-colortext/compare/511bcaf...v1.0.0)
- github.com/docker/docker: [v20.10.12+incompatible → v20.10.17+incompatible](https://github.com/docker/docker/compare/v20.10.12...v20.10.17)
- github.com/envoyproxy/go-control-plane: [63b5d3c → 49ff273](https://github.com/envoyproxy/go-control-plane/compare/63b5d3c...49ff273)
- github.com/go-logr/logr: [v1.2.0 → v1.2.3](https://github.com/go-logr/logr/compare/v1.2.0...v1.2.3)
- github.com/go-logr/zapr: [v1.2.0 → v1.2.3](https://github.com/go-logr/zapr/compare/v1.2.0...v1.2.3)
- github.com/golangplus/testing: [af21d9c → v1.0.0](https://github.com/golangplus/testing/compare/af21d9c...v1.0.0)
- github.com/google/cadvisor: [v0.44.1 → v0.45.0](https://github.com/google/cadvisor/compare/v0.44.1...v0.45.0)
- github.com/google/cel-go: [v0.10.1 → v0.12.4](https://github.com/google/cel-go/compare/v0.10.1...v0.12.4)
- github.com/google/go-cmp: [v0.5.5 → v0.5.6](https://github.com/google/go-cmp/compare/v0.5.5...v0.5.6)
- github.com/google/martian/v3: [v3.1.0 → v3.2.1](https://github.com/google/martian/v3/compare/v3.1.0...v3.2.1)
- github.com/google/pprof: [cbba55b → 94a9f03](https://github.com/google/pprof/compare/cbba55b...94a9f03)
- github.com/googleapis/gax-go/v2: [v2.0.5 → v2.1.1](https://github.com/googleapis/gax-go/v2/compare/v2.0.5...v2.1.1)
- github.com/imdario/mergo: [v0.3.5 → v0.3.6](https://github.com/imdario/mergo/compare/v0.3.5...v0.3.6)
- github.com/matttproud/golang_protobuf_extensions: [c182aff → v1.0.1](https://github.com/matttproud/golang_protobuf_extensions/compare/c182aff...v1.0.1)
- github.com/onsi/gomega: [v1.10.1 → v1.19.0](https://github.com/onsi/gomega/compare/v1.10.1...v1.19.0)
- github.com/opencontainers/runc: [v1.1.1 → v1.1.3](https://github.com/opencontainers/runc/compare/v1.1.1...v1.1.3)
- github.com/pquerna/cachecontrol: [0dec1b3 → v0.1.0](https://github.com/pquerna/cachecontrol/compare/0dec1b3...v0.1.0)
- github.com/seccomp/libseccomp-golang: [3879420 → f33da4d](https://github.com/seccomp/libseccomp-golang/compare/3879420...f33da4d)
- github.com/xlab/treeprint: [a009c39 → v1.1.0](https://github.com/xlab/treeprint/compare/a009c39...v1.1.0)
- github.com/yuin/goldmark: [v1.4.1 → v1.4.13](https://github.com/yuin/goldmark/compare/v1.4.1...v1.4.13)
- go.etcd.io/etcd/api/v3: v3.5.1 → v3.5.4
- go.etcd.io/etcd/client/pkg/v3: v3.5.1 → v3.5.4
- go.etcd.io/etcd/client/v2: v2.305.0 → v2.305.4
- go.etcd.io/etcd/client/v3: v3.5.1 → v3.5.4
- go.etcd.io/etcd/pkg/v3: v3.5.0 → v3.5.4
- go.etcd.io/etcd/raft/v3: v3.5.0 → v3.5.4
- go.etcd.io/etcd/server/v3: v3.5.0 → v3.5.4
- golang.org/x/crypto: 8634188 → 3147a52
- golang.org/x/mod: 9b9b3d8 → 86c51ed
- golang.org/x/net: cd36cc0 → a158d28
- golang.org/x/sync: 036812b → 886fb93
- golang.org/x/sys: 3681064 → 8c9f86f
- golang.org/x/tools: 897bd77 → v0.1.12
- google.golang.org/api: v0.46.0 → v0.60.0
- google.golang.org/genproto: 42d7afd → c8bf987
- google.golang.org/grpc: v1.40.0 → v1.47.0
- google.golang.org/protobuf: v1.27.1 → v1.28.0
- gopkg.in/yaml.v3: 496545a → v3.0.1
- k8s.io/klog/v2: v2.60.1 → v2.70.1
- k8s.io/kube-openapi: 3ee0da9 → 67bda5d
- k8s.io/utils: 3a6ce19 → ee6ede2
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.30 → v0.0.32
- sigs.k8s.io/json: 9f7c6b3 → f223a00
- sigs.k8s.io/kustomize/api: v0.11.4 → v0.12.1
- sigs.k8s.io/kustomize/cmd/config: v0.10.6 → v0.10.9
- sigs.k8s.io/kustomize/kustomize/v4: v4.5.4 → v4.5.7
- sigs.k8s.io/kustomize/kyaml: v0.13.6 → v0.13.9
- sigs.k8s.io/structured-merge-diff/v4: v4.2.1 → v4.2.3

### Removed
- github.com/OneOfOne/xxhash: [v1.2.2](https://github.com/OneOfOne/xxhash/tree/v1.2.2)
- github.com/cespare/xxhash: [v1.1.0](https://github.com/cespare/xxhash/tree/v1.1.0)
- github.com/clusterhq/flocker-go: [2b8b725](https://github.com/clusterhq/flocker-go/tree/2b8b725)
- github.com/emicklei/go-restful: [v2.9.5+incompatible](https://github.com/emicklei/go-restful/tree/v2.9.5)
- github.com/google/cel-spec: [v0.6.0](https://github.com/google/cel-spec/tree/v0.6.0)
- github.com/jstemmer/go-junit-report: [v0.9.1](https://github.com/jstemmer/go-junit-report/tree/v0.9.1)
- github.com/nxadm/tail: [v1.4.4](https://github.com/nxadm/tail/tree/v1.4.4)
- github.com/onsi/ginkgo: [v1.14.0](https://github.com/onsi/ginkgo/tree/v1.14.0)
- github.com/quobyte/api: [v0.1.8](https://github.com/quobyte/api/tree/v0.1.8)
- github.com/spaolacci/murmur3: [f09979e](https://github.com/spaolacci/murmur3/tree/f09979e)
- github.com/storageos/go-api: [v2.2.0+incompatible](https://github.com/storageos/go-api/tree/v2.2.0)
- gopkg.in/tomb.v1: dd63297



# v1.25.0-rc.1


## Downloads for v1.25.0-rc.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes.tar.gz) | e14dae73addde178b3cfe2d282a261e4ff09ba0015094911a33a5c0657ac7480bb1eeb96e6c48b773a4acbe4379c20372c8dec5e41840b4f886a143aef24fb44
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-src.tar.gz) | f1e9237aa0c169a3763a33ec17cb2c52adc7401c413c157aff34ca37168612e7be7adc6d181d81081b0d5577e3036c91e211273c2b13799306d20e4ae1df9427

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | a2b1b1eabbd3cc9edd8965d99b018906e1b9228eef8af502c985237598d34705c1e85b36c7c79eb7084a4269a22d08b8d38d05126a41e4aeed4e4cd023e7f630
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-darwin-arm64.tar.gz) | acf9af8669789858160a479273a820b6e28b2f245d5d5904b1b2ab48c62984b1c4c1ebec5ac728878406443daf66a82a1a29d30916a5aaea2d3b0a7fdd2b40cc
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-linux-386.tar.gz) | 225898328977467b8158af2bca4335efe74e00416eb2f4003361086c37a0bc8d2490642afd9a2e374ea462ad6bf9a760340975fe84d0daed81b4fb8faf280b81
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | 6148209b0088f8491f17d31375420bb5c027e115dfae5e4366a6bb5d7ce1416e36b58f23723e75c91ff311891d132fde11bf387766825ad1ecc5186007a1c0cf
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-linux-arm.tar.gz) | 630308287cdd40f53ce809807cfb60daa657ebe898426db47e7641f608621786ae4af6bae5505dda717bc9a7bdda4950d6a95c1100393183a8bb07bc8fe12383
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | 8bc549474541955bd4da1acea34908b4ea958e002aa135251e97868b759e0aa29e9494bd07375eb38cfa07c973f54f884c711c19f96eaa490a4aceb0065ab3c1
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | b46310a3a224db894dfe93d85ac7107a2ab66f80a57860d3615df6a6af8fca6728549c0aaafe2b7b07f22c3bf7187f72b630527687e4d1a5fae79b51fe653a24
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | c0d1ec17577e9dd0c45b28f89a49af8cd17f3618d5ef5428c28ac96e842e7eacc84828bbac4209f4e3bec463cdc963b11f7f1f016e5032c94d85f98a64630d30
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-windows-386.tar.gz) | 99f85b4d5bd693fa8fe52471d2e970adb5ed7236c9793fa91d9b39739f2062d3dd9bd370fa3e810eb13a690df7314d1c31dd501585499251ab169323569115f9
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | 2ea4a4cf1c17006f7bd2309fca52dbe219d0e73f1c103dcd2ac0dfdefc650b80d2bb12ce0a6375f6894d6bfca44ef2718ac72244e5b6e4c432db0f198b4e34ad
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-client-windows-arm64.tar.gz) | c27593a4d834e3ecf3fd5ddb5aebad152af5c2bd30fd4c09b946e258498330745114963221ee98ae2b32953c5c50898f1328e38c64ca442a4c3fc07f1668220b

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | 46ad89bff41efcc9a6f98a16b268ef467037c4a735eddefe98794c5c652b417efe0087d84d1461809a75f443721412bbea8a0191d7bc542a0762a8bc6ef67190
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-server-linux-arm.tar.gz) | c79ee0a2de9b9558baf631f7b1918f46e07fb0c5dd21861700c151666013a4a9ded300d0b552de5efe9cd977abec0dda4514b903092ad376ecd5fb2a8ea1173f
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | 4cc9f720cc391c6bbb20d6de7f5c47e457c7d7ea42672bf27080ed2e673b52fd5c2266f766318d6e2bb08b1f733c03e983ef2a440bd481d741c9ab07029860f8
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | 50889dac23696d2f8242aa1059d5bbcfbe440fade700c7f2fac1148b19a0754d48f76672bd0c0e7030fe88e29aa06de4c22f36d00f8fca9989d45d24337db82f
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | a9493cfa2ba2fe482eba8e83d524cd0b3218ae438157163daf740dc100dd7a32249ff19fd87ae753f133b1858412938739c108b4d34279d46e53268e63713694

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | 590a658c556b7269efbcc802e3495b3e9e3216d3ad22c113131d46c4680e8fdda8b6eb56fce088fda18947861d89889019d101c4d27246a30608a27f05a32fa6
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-node-linux-arm.tar.gz) | e09c753808a87f772d418d750489c0fd2c53cedf0f874d5672fcebba0cd3e549f2f14ed463ba9c398b41e46525f513c883c825880356a670679cf05a383c01a7
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | 3db05d3150fcf061dbcfdb3ddeca69a2fd905660ad85446ccf8e68210592cda0c64b4324251058354e99f4b7747ae212bf3b5b0db1943310dba0d7bbe6c5f034
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | dd05bc3a5c2209489af484c799e7a731bbbe1104574f186ab8a9be58fd74c70cad8c1ca097d00e2348ad62610061713ae42f2cfdd356f2c1f83b50161385c0fc
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | c4d461cc4e442d8a033df8916ee20b58323b20e2a25f04d50ea2a7c3586b9737eac63317cc9175b5dc8308299e42e3b3d86d3aa1ab755d17515bdb44612453c3
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | a9b75fd62986abce44a5ad888e35ff7353cdaf91c28092c6c117c599ede0c6c43bbaf720b1a6f189e9f85c1dfa6371d88f6c9930fd46ff5d9547a764b5854cde

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.25.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.25.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.25.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.25.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.25.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.25.0-rc.0

## Changes by Kind

### Documentation

- Clarify that the node-port range (kube-apiserver config) must not overlap the nodes' ephemeral range. ([#111697](https://github.com/kubernetes/kubernetes/pull/111697), [@thockin](https://github.com/thockin)) [SIG Network]

### Bug or Regression

- Fix memory leak on kube-scheduler preemption ([#111773](https://github.com/kubernetes/kubernetes/pull/111773), [@amewayne](https://github.com/amewayne)) [SIG Scheduling]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.25.0-rc.0


## Downloads for v1.25.0-rc.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes.tar.gz) | 88f1261569ccde08beb4f7d5d79a9750168fc32b6bf4975ebf9a225ea1d57d6de21f5321944de0d83f572af93fa11265584a09049b6fa594866daa8f99102a6a
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-src.tar.gz) | bed0ce425fff0ff75608a043ed3a9a9b89ab640d08620b96976c733df84be6c161d8c586b9021e0cbd7ce88fba749e417befe7cc4042dcc7a4a4d6a9d6c40d26

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | c937bb03c08b081f1889e50a0a2eb4631ab7e1ceca44ed244d221959468dfbdb60c0169624aa6d04e1775cc91ea2f448a213d2e0addd6525548a88735a970379
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | c32a0de11f0ccf44dd495d01d4d174d361124fe527ab9dc13b210994ecfa19d1f23e61053d920ee1d60bf80c7d4bb197c5dcbb03e1aed2231d53fca7aa9e876a
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-linux-386.tar.gz) | 32f64a3d8b7547e94b934e58d83ac566f36232822458e84cd07de0f6aeb15b0169d74413fdf2456a0b0fe17d0a600e203e4fffdfd8869e9d6ca3b704c0e41e48
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | a76fb8e22ca249914e47a8b523a9eb201b17014b52a7ae335386087f510a00899ea76d4f7cd4f3ce1cfbee8f8b9c9a23862fdcd6da71c6a6269f663e5e813182
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-linux-arm.tar.gz) | f8f96ebd7727a99a923542f1f212615beb83556de93784f641e56021cf1ccac3bbe8a656389111cc95a55d9fa85b711e5a8ac50d1c169c41ae32f72c28c801d7
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | 4de505812c10612f3cf420e87c6e712f46a208f2a64ce778c6f33291b390f7a4e4754913f402196456c8fef17f67ba44fa1bd14171e820d2e1a47d086c86e0bc
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | 10d2a289d54ba75788ba15cd1de5ebc3dea89a0a1066003526b635c7260da9aab22b7b46794f341bca80c5e42bfe4681211513be0c02d8f42fb9db5a36f8ff90
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | 1254fa747d2f0a92f21136a64b2efcd634c549a66f0fecf4a7f4bf5be2ee1986c5cfea4b1cbe6ff04970891e4a2e0f9abb2462cdaf0eb275397977a1e4d06877
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-windows-386.tar.gz) | 85467d33b8d1a628e1b1db44e0e87116d82d3d00b06b51a0754475fe7f03ffc9dd1dba1bbbc4dea62a28d6d762ca557c88a29926015c2109f30655710d5fc746
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | fa51f5e14ac2d4582be4eedb1e68c8e6ceeac57ac0bf25fadc0276c7b6584cde6f41bf1495944606b5493debd1fb787d511d342f02ab5f5fe1ffce2d40f488ff
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | 8f3f81a7ceaa1f6167f85046dbcb3091a09dbfa3b49fc82e8d5ca08283b077fd5031203929b4a212ec62c02a72409e0b947fcc3b21e7430b3b14407654a97c90

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | 8e9695a09dfeca990afc5401e89e89722ba0dbf8c4c9aef40552596b9c5d6dd45bf53d4c1ecea821c44b48db9110668049034f462c38e4b0a48dbeb5cfdecff9
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-server-linux-arm.tar.gz) | b05066b5004d7a827f27bb2f38a111babef9f43f8d4b6b112a793c373b175085f3053756ae548c51d42b0d32f7fce463010d14bb9a652b326843406538f8e2aa
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | 49e39c509dd49566eb14d52f9047743f3bb96d86fb9747fa465f2ece3311fdfa70e601fbe9015b103be8748db3474216bcd7a33cc82a4dd984094aedccfde02e
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | cf9b707cdb69bd45eff015b67379d42a7049884c484a740e36e533ada1f22f0f28128605a6149f33933b8b8b68754da01545b4c2c55d5d6956d5cf993ecbf615
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | bbbd35f8b436aed6b76b9d60a5efec2865af776b331db67b45a96eb4c898fab51713952a5339242514e80f26b5581961f5cd4d1829f5c7022074716577031589

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | 0a25201b921171661a084c5aa51c61b1922f7e90593d747760588bf7104fb52f4fa796b75aac86ea9ec91c1cb2c379f9dd5b343bfdff6c5f2bb4fe8b099be02e
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-node-linux-arm.tar.gz) | 66a15466074a0813a3d04bf3cb1d4e44209b6a90b258fc3b4ef84ced4d7b030036dd03ce19273ae67aa8903775431b85bdae32710b1a11d8651c2f6191d9e864
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | a7a80e0958b6620c052f65e0e491396ceb0c3ad121499a76fa0b76e0df43b5a7a29a1fc31b1aabb7d856efc2c7df4c746a79e0e6ed1019a3e42cde877a19f2fb
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | be06991362911c1c08d7ab5e138fd7d05a80b481c9f7fb9dbc68ad85b205e44811f37a3d6ea5cf3368b95b2a878b8b13096bb76491d20afa03216a7ca7e2f42d
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | 261c4ab9cde5e3b1f51db396ad1c737fc2cc165437df7c19afeae4548f57822a82326a039767dbe36582ec1bedc08bf9d0f3d58726d7d81df04834d46aac239f
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | 114e3630bb6f5140a3097c6c81a6d62280a7892fdc792f22ad9417d80a19932ed0b719ad56718e8e9f08c80c6149e9cf21fe20ba65db4f460b1d63db0abc2e24

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.25.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.25.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.25.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.25.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.25.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.25.0-beta.0

## Changes by Kind

### API Change

- Introduces support for handling pod failures with respect to the configured pod failure policy rules ([#111113](https://github.com/kubernetes/kubernetes/pull/111113), [@mimowo](https://github.com/mimowo)) [SIG API Machinery, Apps, Auth, Scheduling and Testing]
- NodeIPAM support for multiple ClusterCIDRs (https://github.com/kubernetes/enhancements/issues/2593) introduced as an alpha feature.

  Setting feature gate MultiCIDRRangeAllocator=true, determines whether the MultiCIDRRangeAllocator controller can be used, while the kube-controller-manager flag below will pick the active controller.

  Enable the MultiCIDRRangeAllocator by setting --cidr-allocator-type=MultiCIDRRangeAllocator flag in kube-controller-manager. ([#109090](https://github.com/kubernetes/kubernetes/pull/109090), [@sarveshr7](https://github.com/sarveshr7)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Instrumentation, Network and Testing]
- The CSIInlineVolume feature has moved from beta to GA. ([#111258](https://github.com/kubernetes/kubernetes/pull/111258), [@dobsonj](https://github.com/dobsonj)) [SIG API Machinery, Apps, Auth, Instrumentation, Storage and Testing]

### Bug or Regression

- Fix memory leak in the job controller related to JobTrackingWithFinalizers ([#111721](https://github.com/kubernetes/kubernetes/pull/111721), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps]
- Remove the recently re-introduced schedulability predicate (by PR: https://github.com/kubernetes/kubernetes/pull/109706) as to not have unschedulable nodes removed from load balancers back-end pools. ([#111691](https://github.com/kubernetes/kubernetes/pull/111691), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider and Network]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.25.0-beta.0


## Downloads for v1.25.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes.tar.gz) | f1f8548098b679784aeda6e2453d34a3b2e1670a066b9984acce6790d61bd8733f5c5a7875e48c379f4b4a6a28130a807f93a847d8ac776b3fb3d1dec167be9a
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-src.tar.gz) | c338733b41387cce6dd40ebef9ff3bd35e796cd2635e75ea51b8e1d944672d4abdbf6ed14c0bfab070fc19259c66f7ba426858b79c51c9bc74a23a31076def6b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | 11723387623bbae84f76fc01f2a9fa1612b13238576b205073fd38512ce9aa5c8356a1c072a7fd8f27b271f3fe6441f8b7a2ea19e02f9d15503b3e76a1c65f6a
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | 9600383719091ab9a18fd871a8b7349db7b3b1162e54c202fb725d8e21365e9ff9f109f4eed68db87d28482768af325ad17996382a537b3988527278756997a9
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-linux-386.tar.gz) | b04908a39ef653e913e090bc8c46eac528272598f4e173a5898355caf027e38614a1fe7ad9c3f307a28e1e7718bb38274f463cc09d8020fc01594df2186bc9b3
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | 7dfbb41cd1cd43db8b63cefa0aed33754a3274f350037bdf484593365942dcad3112436cfd7a83fb760b7e81c98dfcc66a7a7d2c36de59b4a80a9049775bae81
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-linux-arm.tar.gz) | 4e4ded957a7dd17f3b19544eb564cdbd7ea0018a77d1c7106403607f93e41896e90abb3caedb2c7d4372326c370490034380af741948fb86e5ea6a1a71648008
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | de312ef5789512f27bcaf78e201e69f239caeb897b57183e237d34f90a886f2ad11bd143658f5a0a0e6866cea05fb099808a5498d529bb609ec2247b3165d849
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | a1cfae2dd27aa42646d5f21ddea50749d41bac4575b2003bae08e22567a90fb091d76660148aa0354feb6e49764c5995a1f6c2f967d73c5504ed9cc9188f44fd
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | 15aa3ac2ce68e5485b066b458f5cde48d09326683c85c19459ecdf3d0d135b5f44818db7c359f69ec2bd7da0049871fadd70ab2216314c433bf46a015370f5f5
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-windows-386.tar.gz) | 7ce798115a4c405682d98d7482af08a6c22f20121187745883c5209d91bad7a4faac044c3e4c1501cdff112a1342867f4b6d2186a89eb9465a6ec7359983a6d4
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | 68e3675cbedb69c69b9fb2bced9ad453d4cc11ed465cd0a193d7bbb0c8ea9448ca981ec47b971d2d4de5ce0245b1d0e4f9f1ac2ace2647b1ef7bce35edf525a3
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | 5e655248bdefe4b37010df99abe80c76fddf6e299cdb88c5cb93d5f1d8f55d2cfa0fbeba3d419929e7ecbef5a7b22d096c737f3493fe2ec748d228257ac63880

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | 509d38d4c7a8b513f414c1e64889f8ea1573fcd9a0920bd955b870b234c794fb8d2a8347447b7940323d4a43b1691a05f22178c80401975d0cf1fc1b31d71086
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-server-linux-arm.tar.gz) | 3d7896f8cde58d63243f3a704a291cc9135e01147448b93165d8c826d3429fc799a3e497614091545b87a8389a658b4dc2cc4e2b91f38166ef35d52a8139f17c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | 4ce21bc8b68eebc9d7d52d63af7e4cfb641f0912417c601f5bbef1957c6aee70c729bde33cd3d9a4d12dd804f2f3fd45a6fee88c7e45d7eed23cbb6ac2aa1839
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | 4438d15cc91c606e0c66a4d49ade0ee98ae9b3a4440a007dc017542ab52da2dbe73ff9fd4ccb7fe97cf9d44219b237e4f2887f6ae333d431170f2dcd51409879
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | ebad4c0aeacd63763b5ca7b059b373e909283bde5464b2727283e6b76edda7ebfd0de56a93c9c1543fd7206a0b12b0a8c1ef7e121188d7201118db85074b919c

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 9e7fe60d78e2a9df8dcd091c1f8bee1ebd59dee19714224b851895964bc1e3d47efff3d38ff3bf3e0077ca3af263d9adca0c5cbb8fa2d968090bac2e7097f745
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-node-linux-arm.tar.gz) | 56727507022103d2d4bfdb1043a7d340227346b641390dd9c288f796553733716a148ea41c5cb4ec5c52e45e210acf30f2be60e73ceedabeb97f09f280653e80
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | 3731fc563fcb6d7bd7a998b5e0a538effaf4843ae6d8a3b1fde666e564aa6961e3664791e5615ba831c7adcd10bb64ab9455bff57577f96b42e6470ca16ff8d9
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 6e4bfaa2e5c599928930ca85981a27b338ef4366d34a089d3a807d51cb83c9be5d3af54be16efacce102b7d9b9e146e5d6f0b0c0987dea35f374f2f1b8e0c68c
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | efe12a8e4c15e3688afd9f13e1824f6bcf1fc3ffbc05da0321a5eb82534085d90832ade189c41ea44b8b8ae8c3d8eb85ac450f21287a7e1b91090780fe1f3bbc
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 0f494940e778ca8f043b7a917097a3fa719d7a3cdd555e23d722c329b0a0d7293eafc54833868d56db3cb6e5ed8a73462af6a2c00e61fff77fcfbdb14217c7aa

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.25.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.25.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.25.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.25.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.25.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.25.0-alpha.3

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Encrypt data with DEK using AES-GCM instead of AES-CBC for kms data encryption. No user action required. Reads with AES-GCM and AES-CBC will continue to be allowed. ([#111119](https://github.com/kubernetes/kubernetes/pull/111119), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
  - Intree volume plugin flocker support is been completely removed from Kubernetes. ([#111618](https://github.com/kubernetes/kubernetes/pull/111618), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG API Machinery, Node, Scalability and Storage]
  - Intree volume plugin quobyte support is been completely removed from Kubernetes. ([#111619](https://github.com/kubernetes/kubernetes/pull/111619), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG API Machinery, Node, Scalability and Storage]
  - Intree volume plugin storageos support is been completely removed from Kubernetes. ([#111620](https://github.com/kubernetes/kubernetes/pull/111620), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG API Machinery, Node, Scalability and Storage]

## Changes by Kind

### Deprecation

- API server's deprecated `--service-account-api-audiences` flag is now removed.  Use `--api-audiences` instead. ([#108624](https://github.com/kubernetes/kubernetes/pull/108624), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Auth]
- Support for the alpha seccomp annotations `seccomp.security.alpha.kubernetes.io/pod` and `container.seccomp.security.alpha.kubernetes.io`, deprecated since v1.19, has been partially removed. Kubelets no longer support the annotations, use of the annotations in static pods is no longer supported, and the seccomp annotations are no longer auto-populated when pods with seccomp fields are created. Auto-population of the seccomp fields from the annotations is planned to be removed in 1.27. Pods should use the corresponding pod or container `securityContext.seccompProfile` field instead. ([#109819](https://github.com/kubernetes/kubernetes/pull/109819), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps, Auth, Node and Testing]
- VSphere releases less than 7.0u2 are not supported for in-tree vSphere volume as of Kubernetes v1.25. Please consider upgrading vSphere (both ESXi and vCenter)  to 7.0u2 or above. ([#111255](https://github.com/kubernetes/kubernetes/pull/111255), [@divyenpatel](https://github.com/divyenpatel)) [SIG Cloud Provider]
- Windows winkernel Kube-proxy no longer supports Windows HNS v1 APIs ([#110957](https://github.com/kubernetes/kubernetes/pull/110957), [@papagalu](https://github.com/papagalu)) [SIG Network and Windows]

### API Change

- Added alpha support for user namespaces in pods phase 1 (KEP 127, feature gate: UserNamespacesSupport) ([#111090](https://github.com/kubernetes/kubernetes/pull/111090), [@rata](https://github.com/rata)) [SIG Apps, Auth, Network, Node, Storage and Testing]
- Adds KMS v2alpha1 support ([#111126](https://github.com/kubernetes/kubernetes/pull/111126), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth, Instrumentation and Testing]
- As of v1.25, the PodSecurity `restricted` level no longer requires pods that set .spec.os.name="windows" to also set Linux-specific securityContext fields. If a 1.25+ cluster has unsupported [out-of-skew](https://kubernetes.io/releases/version-skew-policy/#kubelet) nodes prior to v1.23 and wants to ensure namespaces enforcing the `restricted` policy continue to require Linux-specific securityContext fields on all pods, ensure a version of the `restricted` prior to v1.25 is selected by labeling the namespace (for example, `pod-security.kubernetes.io/enforce-version: v1.24`) ([#105919](https://github.com/kubernetes/kubernetes/pull/105919), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG API Machinery, Apps, Auth, Testing and Windows]
- Changes ownership semantics of PersistentVolume's spec.claimRef from `atomic` to `granular`. ([#110495](https://github.com/kubernetes/kubernetes/pull/110495), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Instrumentation and Testing]
- Extends ContainerStatus CRI API to allow runtime response with container resource requests and limits that are in effect.
  - UpdateContainerResources CRI API now supports both Linux and Windows.

  For details, see KEPs below. ([#111645](https://github.com/kubernetes/kubernetes/pull/111645), [@vinaykul](https://github.com/vinaykul)) [SIG Node]
- For v1.25, Kubernetes will be using golang 1.19, In this PR we update to 1.19rc2 as GA is not yet available. ([#111254](https://github.com/kubernetes/kubernetes/pull/111254), [@dims](https://github.com/dims)) [SIG Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Introduce PodHasNetwork condition for pods ([#111358](https://github.com/kubernetes/kubernetes/pull/111358), [@ddebroy](https://github.com/ddebroy)) [SIG Apps, Node and Testing]
- Introduction of the `DisruptionTarget` pod condition type. Its `reason` field indicates the reason for pod termination:
  - PreemptionByKubeScheduler (Pod preempted by kube-scheduler)
  - DeletionByTaintManager (Pod deleted by taint manager due to NoExecute taint)
  - EvictionByEvictionAPI (Pod evicted by Eviction API)
  - DeletionByPodGC (an orphaned Pod deleted by PodGC) ([#110959](https://github.com/kubernetes/kubernetes/pull/110959), [@mimowo](https://github.com/mimowo)) [SIG Apps, Auth, Node, Scheduling and Testing]
- Kube-Scheduler ComponentConfig is graduated to GA, `kubescheduler.config.k8s.io/v1` is available now.
  Plugin `SelectorSpread` is removed in v1. ([#110534](https://github.com/kubernetes/kubernetes/pull/110534), [@kerthcet](https://github.com/kerthcet)) [SIG API Machinery, Scheduling and Testing]
- Local Storage Capacity Isolation feature is GA in 1.25 release. For systems (rootless) that cannot check root file system, please use kubelet config --local-storage-capacity-isolation=false to disable this feature. Once disabled, pod cannot set local ephemeral storage request/limit, and emptyDir sizeLimit niether. ([#111513](https://github.com/kubernetes/kubernetes/pull/111513), [@jingxu97](https://github.com/jingxu97)) [SIG API Machinery, Node, Scalability and Scheduling]
- PersistentVolumeClaim objects are no longer left with storage class set to `nil` forever, but will be updated retroactively once any StorageClass is set or created as default. ([#111467](https://github.com/kubernetes/kubernetes/pull/111467), [@RomanBednar](https://github.com/RomanBednar)) [SIG Apps, Storage and Testing]
- Promote CronJob's TimeZone support to beta ([#111435](https://github.com/kubernetes/kubernetes/pull/111435), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, Apps and Testing]
- Promote DaemonSet MaxSurge to GA. This means `--feature-gates=DaemonSetUpdateSurge=true` are not needed on kube-apiserver and kube-controller-manager binaries and they'll be removed soon following policy at https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecation ([#111194](https://github.com/kubernetes/kubernetes/pull/111194), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps]
- Respect PodTopologySpread after rolling upgrades ([#111441](https://github.com/kubernetes/kubernetes/pull/111441), [@denkensk](https://github.com/denkensk)) [SIG API Machinery, Apps, Scheduling and Testing]
- Scheduler: include supported ScoringStrategyType list in error message for NodeResourcesFit plugin ([#111206](https://github.com/kubernetes/kubernetes/pull/111206), [@SataQiu](https://github.com/SataQiu)) [SIG Scheduling]
- The Pod `spec.podOS` field is promoted to GA. The `IdentifyPodOS` feature gate unconditionally enabled, and will no longer be accepted as a `--feature-gates` parameter in 1.27. ([#111229](https://github.com/kubernetes/kubernetes/pull/111229), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG API Machinery, Apps and Windows]
- The command line flag `enable-taint-manager` for kube-controller-manager is deprecated and will be removed in 1.26.
  The feature that it supports, taint based eviction, is enabled by default and will continue to be implicitly enabled when the flag is removed. ([#111411](https://github.com/kubernetes/kubernetes/pull/111411), [@alculquicondor](https://github.com/alculquicondor)) [SIG API Machinery]
- [Ephemeral Containers](https://kubernetes.io/docs/concepts/workloads/pods/ephemeral-containers/) are now generally available. The `EphemeralContainers` feature gate is always enabled and should be removed from `--feature-gates` flag on the kube-apiserver and the kubelet command lines. The `EphemeralContainers` feature gate is [deprecated and scheduled for removal](https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecation) in a future release. ([#111402](https://github.com/kubernetes/kubernetes/pull/111402), [@verb](https://github.com/verb)) [SIG API Machinery, Apps, Node, Storage and Testing]

### Feature

- A new flag `etcd-ready-timeout` has been added. It configures a timeout of an additional etcd check performed as part of readyz check. ([#111399](https://github.com/kubernetes/kubernetes/pull/111399), [@Argh4k](https://github.com/Argh4k)) [SIG API Machinery]
- Add a new `align-by-socket` policy option to cpu manager `static` policy. When enabled CPU's to be aligned at socket boundary rather than NUMA boundary. ([#111278](https://github.com/kubernetes/kubernetes/pull/111278), [@arpitsardhana](https://github.com/arpitsardhana)) [SIG Node]
- Add container probe duration metrics ([#104484](https://github.com/kubernetes/kubernetes/pull/104484), [@jackfrancis](https://github.com/jackfrancis)) [SIG Instrumentation and Node]
- Added Service Account field in the output of `kubectl describe pod` command. ([#111192](https://github.com/kubernetes/kubernetes/pull/111192), [@aufarg](https://github.com/aufarg)) [SIG CLI]
- Adds new flags into alpha events such as --output, --types, --no-headers ([#110007](https://github.com/kubernetes/kubernetes/pull/110007), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- CSIMigrationAWS upgraded to GA and locked to true. ([#111479](https://github.com/kubernetes/kubernetes/pull/111479), [@wongma7](https://github.com/wongma7)) [SIG Apps, Scheduling and Storage]
- CSIMigrationGCE upgraded to GA and locked to true. ([#111301](https://github.com/kubernetes/kubernetes/pull/111301), [@mattcary](https://github.com/mattcary)) [SIG Apps, Node, Scheduling and Storage]
- Feature gate `ProbeTerminationGracePeriod` is enabled by default. ([#108541](https://github.com/kubernetes/kubernetes/pull/108541), [@kerthcet](https://github.com/kerthcet)) [SIG Node]
- Ginkgo: when e2e tests are invoked through ginkgo-e2e.sh, the default now is to use color escape sequences only when connected to a terminal. `GINKGO_NO_COLOR=y/n` can be used to override that default. ([#111633](https://github.com/kubernetes/kubernetes/pull/111633), [@pohly](https://github.com/pohly)) [SIG Testing]
- Graduated `CustomResourceValidationExpressions` to `beta`. The `CustomResourceValidationExpressions` feature gate is now enabled by default. ([#111524](https://github.com/kubernetes/kubernetes/pull/111524), [@cici37](https://github.com/cici37)) [SIG API Machinery]
- If a Pod has a DisruptionTarget condition with status=True for more than 2 minutes without getting a DeletionTimestamp, the control plane resets it to status=False ([#111475](https://github.com/kubernetes/kubernetes/pull/111475), [@alculquicondor](https://github.com/alculquicondor)) [SIG API Machinery, Apps, Node and Testing]
- Kubectl diff changed to ignore managed fields by default, and a new --show-managed-fields flag has been added to allow you to include managed fields in the diff ([#111319](https://github.com/kubernetes/kubernetes/pull/111319), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Kubernetes is now built with go 1.19.0 ([#111679](https://github.com/kubernetes/kubernetes/pull/111679), [@puerco](https://github.com/puerco)) [SIG Release and Testing]
- Metric `running_managed_controllers` is enabled for Cloud Node Lifecycle controller ([#111033](https://github.com/kubernetes/kubernetes/pull/111033), [@jprzychodzen](https://github.com/jprzychodzen)) [SIG Apps, Cloud Provider and Network]
- Metric `running_managed_controllers` is enabled for Node IPAM controller in KCM ([#111466](https://github.com/kubernetes/kubernetes/pull/111466), [@jprzychodzen](https://github.com/jprzychodzen)) [SIG API Machinery, Apps, Cloud Provider and Network]
- Metric `running_managed_controllers` is enabled for Route,Service and Cloud Node controllers in KCM and CCM ([#111462](https://github.com/kubernetes/kubernetes/pull/111462), [@jprzychodzen](https://github.com/jprzychodzen)) [SIG Cloud Provider, Network and Testing]
- New flag `--disable-compression-for-client-ips` can be used to control client address ranges for which traffic shouldn't be compressed. ([#111507](https://github.com/kubernetes/kubernetes/pull/111507), [@mborsz](https://github.com/mborsz)) [SIG API Machinery]
- Promote LocalStorageCapacityIsolationFSQuotaMonitoring to beta ([#107329](https://github.com/kubernetes/kubernetes/pull/107329), [@pacoxu](https://github.com/pacoxu)) [SIG Node and Testing]
- Update cAdvisor to v0.45.0 ([#111647](https://github.com/kubernetes/kubernetes/pull/111647), [@bobbypage](https://github.com/bobbypage)) [SIG Node]

### Bug or Regression

- Faster mount detection for linux kernel 5.10+ using openat2 speeding up pod churn rates. On Kernel versions less 5.10, it will fallback to using the original way of detecting mount points i.e by parsing /proc/mounts. ([#109217](https://github.com/kubernetes/kubernetes/pull/109217), [@manugupt1](https://github.com/manugupt1)) [SIG Cloud Provider and Storage]
- Fix JobTrackingWithFinalizers when a pod succeeds after the job is considered failed, which led to API conflicts that blocked finishing the job. ([#111646](https://github.com/kubernetes/kubernetes/pull/111646), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Testing]
- Fix performance issue when creating large objects using SSA with fully unspecified schemas (preserveUnknownFields). ([#111557](https://github.com/kubernetes/kubernetes/pull/111557), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Fix s.RuntimeCgroups error condition and Fix possible wrong log print ([#110648](https://github.com/kubernetes/kubernetes/pull/110648), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Node]
- Fixed mounting of iSCSI volumes over IPv6 networks. ([#110688](https://github.com/kubernetes/kubernetes/pull/110688), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixes a bug which could have allowed an improperly annotated LoadBalancer service to become active. ([#109601](https://github.com/kubernetes/kubernetes/pull/109601), [@mdbooth](https://github.com/mdbooth)) [SIG Cloud Provider and Network]
- Kubeadm: enable the --experimental-watch-progress-notify-interval flag for etcd and set it to 5s. The flag specifies an interval at which etcd sends watch data to the kube-apiserver. ([#111383](https://github.com/kubernetes/kubernetes/pull/111383), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG Cluster Lifecycle]
- Kubelet: add log for volume metric collection taking too long ([#107490](https://github.com/kubernetes/kubernetes/pull/107490), [@pacoxu](https://github.com/pacoxu)) [SIG Node and Storage]
- Kubelet: add validation for labels provided with --node-labels. Malformed labels will result in errors. ([#109263](https://github.com/kubernetes/kubernetes/pull/109263), [@FeLvi-zzz](https://github.com/FeLvi-zzz)) [SIG Node]
- Make usage of key encipherment optional in API validation ([#111061](https://github.com/kubernetes/kubernetes/pull/111061), [@pacoxu](https://github.com/pacoxu)) [SIG Apps, Auth and Node]
- Namespace editors and admins can now create leases.coordination.k8s.io and should use this type for leaderelection instead of configmaps. ([#111472](https://github.com/kubernetes/kubernetes/pull/111472), [@deads2k](https://github.com/deads2k)) [SIG API Machinery and Auth]
- Print pod.Spec.RuntimeClassName in kubectl describe ([#110914](https://github.com/kubernetes/kubernetes/pull/110914), [@yeahdongcn](https://github.com/yeahdongcn)) [SIG CLI]
- Reduce the number of cloud API calls and service downtime caused by excessive re-configurations of cluster LBs with externalTrafficPolicy=Local when node readiness changes (https://github.com/kubernetes/kubernetes/issues/111539). The service controller (in cloud-controller-manager) will avoid resyncing nodes which are transitioning between `Ready` / `NotReady` (only for for ETP=Local Services). The LBs used for these services will solely rely on the health check probe defined by the `healthCheckNodePort` to determine if a particular node is to be used for traffic load balancing. ([#109706](https://github.com/kubernetes/kubernetes/pull/109706), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG API Machinery, Cloud Provider, Network and Testing]
- Remove the recently re-introduced schedulability predicate (by PR: https://github.com/kubernetes/kubernetes/pull/109706) as to not have unschedulable nodes removed from load balancers back-end pools. ([#111691](https://github.com/kubernetes/kubernetes/pull/111691), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider and Network]
- The `priority_level_request_utilization` metric histogram is adjusted so that for the cases where `phase=waiting` the denominator is the cumulative capacity of all of the priority level's queues.
  The `read_vs_write_current_requests` metric histogram is adjusted, in the case of using API Priority and Fairness instead of max-in-flight, to divide by the relevant limit: sum of queue capacities for waiting requests, sum of seat limits for executing requests. ([#110164](https://github.com/kubernetes/kubernetes/pull/110164), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery, Instrumentation and Testing]
- This change fixes the gce firewall update when the destination IPs are changing so that firewalls reflect the IP updates of the LBs. ([#111186](https://github.com/kubernetes/kubernetes/pull/111186), [@sugangli](https://github.com/sugangli)) [SIG Cloud Provider]
- Unmount volumes correctly for reconstructed volumes even if mount operation fails after kubelet restart ([#110670](https://github.com/kubernetes/kubernetes/pull/110670), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Update max azure data disk count map with new VM types ([#111406](https://github.com/kubernetes/kubernetes/pull/111406), [@bennerv](https://github.com/bennerv)) [SIG Cloud Provider and Storage]
- Upgrades functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv4.5.7 ([#111606](https://github.com/kubernetes/kubernetes/pull/111606), [@natasha41575](https://github.com/natasha41575)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider and Instrumentation]
- UserName check for 'ContainerAdministrator' is now case-insensitive if runAsNonRoot is set to true on Windows. ([#111009](https://github.com/kubernetes/kubernetes/pull/111009), [@marosset](https://github.com/marosset)) [SIG Node, Testing and Windows]
- Windows kubelet plugin Watcher now working as intended. ([#111439](https://github.com/kubernetes/kubernetes/pull/111439), [@claudiubelu](https://github.com/claudiubelu)) [SIG Node, Testing and Windows]

### Other (Cleanup or Flake)

- Add e2e test flag to specify which volume drivers should be installed. This deprecates the ENABLE_STORAGE_GCE_PD_DRIVER environment variable. ([#111481](https://github.com/kubernetes/kubernetes/pull/111481), [@mattcary](https://github.com/mattcary)) [SIG Storage and Testing]
- Default burst limit for the discovery client is now 300. ([#109141](https://github.com/kubernetes/kubernetes/pull/109141), [@ulucinar](https://github.com/ulucinar)) [SIG API Machinery and CLI]
- For Linux, `kube-proxy` uses a new “distroless” container image, instead of an image based on Debian. ([#111060](https://github.com/kubernetes/kubernetes/pull/111060), [@aojea](https://github.com/aojea)) [SIG Network, Release and Testing]
- GlusterFS provisioner (`kubernetes.io/glusterfs`) has been deprecated in this release. ([#111485](https://github.com/kubernetes/kubernetes/pull/111485), [@humblec](https://github.com/humblec)) [SIG Storage]
- Kube-scheduler ComponentConfig v1beta2 is deprecated in v1.25. ([#111547](https://github.com/kubernetes/kubernetes/pull/111547), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Shell completion is now provided for the "--subresource" flag. ([#109070](https://github.com/kubernetes/kubernetes/pull/109070), [@marckhouzam](https://github.com/marckhouzam)) [SIG CLI]
- The kubelet no longer supports collecting accelerator metrics through cAdvisor. The feature gate `DisableAcceleratorUsageMetrics` is now GA and cannot be disabled. ([#110940](https://github.com/kubernetes/kubernetes/pull/110940), [@pacoxu](https://github.com/pacoxu)) [SIG Node]

## Dependencies

### Added
- github.com/gogo/googleapis: [v1.4.1](https://github.com/gogo/googleapis/tree/v1.4.1)
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.20.0
- go.opentelemetry.io/contrib/propagators: v0.20.0

### Changed
- github.com/containerd/containerd: [v1.4.12 → v1.4.9](https://github.com/containerd/containerd/compare/v1.4.12...v1.4.9)
- github.com/docker/docker: [v20.10.12+incompatible → v20.10.17+incompatible](https://github.com/docker/docker/compare/v20.10.12...v20.10.17)
- github.com/google/cadvisor: [v0.44.1 → v0.45.0](https://github.com/google/cadvisor/compare/v0.44.1...v0.45.0)
- github.com/google/cel-go: [v0.12.3 → v0.12.4](https://github.com/google/cel-go/compare/v0.12.3...v0.12.4)
- github.com/imdario/mergo: [v0.3.5 → v0.3.6](https://github.com/imdario/mergo/compare/v0.3.5...v0.3.6)
- github.com/matttproud/golang_protobuf_extensions: [c182aff → v1.0.1](https://github.com/matttproud/golang_protobuf_extensions/compare/c182aff...v1.0.1)
- github.com/xlab/treeprint: [a009c39 → v1.1.0](https://github.com/xlab/treeprint/compare/a009c39...v1.1.0)
- github.com/yuin/goldmark: [v1.4.1 → v1.4.13](https://github.com/yuin/goldmark/compare/v1.4.1...v1.4.13)
- golang.org/x/mod: 9b9b3d8 → 86c51ed
- golang.org/x/net: 27dd868 → a158d28
- golang.org/x/sync: 036812b → 886fb93
- golang.org/x/sys: a9b59b0 → 8c9f86f
- golang.org/x/tools: v0.1.10 → v0.1.12
- k8s.io/kube-openapi: 011e075 → 67bda5d
- k8s.io/utils: 3a6ce19 → ee6ede2
- sigs.k8s.io/kustomize/api: v0.11.4 → v0.12.1
- sigs.k8s.io/kustomize/cmd/config: v0.10.6 → v0.10.9
- sigs.k8s.io/kustomize/kustomize/v4: v4.5.4 → v4.5.7
- sigs.k8s.io/kustomize/kyaml: v0.13.6 → v0.13.9
- sigs.k8s.io/structured-merge-diff/v4: v4.2.1 → v4.2.3

### Removed
- github.com/clusterhq/flocker-go: [2b8b725](https://github.com/clusterhq/flocker-go/tree/2b8b725)
- github.com/quobyte/api: [v0.1.8](https://github.com/quobyte/api/tree/v0.1.8)
- github.com/storageos/go-api: [v2.2.0+incompatible](https://github.com/storageos/go-api/tree/v2.2.0)



# v1.25.0-alpha.3


## Downloads for v1.25.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes.tar.gz) | 7f4c79b7ed811df512afca3b8cd9cac2e7119b8ea3f2a03c858d9482e9b788e85fa0f78e60e913e3fc6ba30c5c398730461f5d9753839cf7acc8108089c7c9d7
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-src.tar.gz) | b6ccd63be2677633774f98d182a3acd4828135dd43bb8cd532f722373f8271d00295b401a0344c6b263c9c44fcdd1f815263e0ac31e60f0e55b38435a8cde7bc

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | c1cc390b4aa00bcb367a746af4ea6dd884662559254367868fd57afd79c45b51dc3cf62861e980b91059a998bf13a8deba247b068ab7df179cc3206af4e731fa
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | b5fbc71abe517ab45486708081b66938f6643c2d85b3630a13008338ba65f0be56d3304e00d013ed49c67edd39a60cae1b045bd9165c8c1bd562e1bc6df09803
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-linux-386.tar.gz) | f981b4421493d3588c4f8fcb4dc95821d7218cc10a62fc68264dad2d640e8e2ece4a4647f5cf96bf9337719c9a803c72255b6d5fbbc5533c719e9e59ccb50de6
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | ec2d0dda384702d5c826f08a71d6527d79565a4f81dbd56f5482ffde889e6d5d9a6cd26824212af1e91f2ba84f062b6926cf795edc3dac1174e085a14be4ced5
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | 7125fa0b199de63d1ffe56d5749e8a37cf4d61e993a6fa3c6cc47f1e3d120b949eec3c2901df0875a579f3b4fa142925918169a6a1dfa098932c57b91d5cf53b
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | ee24154a7f95cb9c73c1eb53ccd3dc297492afd319b3581682a5050d825cf52c01753ae1fc18cb8bc6c5fa22ba019642edafaccf87b7077320ac8976a0b66655
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | b1218f27c5f3183622d7ec909c416df2bdf7a178005dccf4c84f24340140f9221fe6c284e7947ff4d59cc652d3489758f406fc27e60593f3aaa1c5ac692556d9
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 4aa77ffe79235a5c3dfffd797a567535c2b23a47fd10840bed5db8f3ad623bc1977045fd13e14875f8cddc59b9d146babe5904ed4e2a9e4d2920894dca6e7d15
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 5c3c42e37f8d6cb53d8abd8a8af7b8da8dad8151f8182428714acc0809e1a0d478488f9d106f5c6acc1ec0489ee22c63230f65f0677796605c3e0bde352440c9
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 1c55a3c1a73d519a4958e2ec79a56bbe28b27bada3af605efcd60a3a643244b24d7eaf2932046a811c11aa98a32de460613b012bbb7adfe0d13aa8a9a9113acc
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | 855779f6a7b5d75c60ba21401b2c9a94672dd9a99250bb95103a871e59232e4b409e3330d0ef946899b077f6c43f2bc3eb32c22f3fcc731b6db34247d3278fee

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 1a25a5475380658ad56bd49e7db241cec648179df8627a1975d996c0656629a272427f944a587a817b28e2690a5723fcf00127084ef84505d13cb64d7a4ccc61
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | ad9e57e6d46d2b0840df0e67c9162524175667c05abc30338c0fec06ecc130ca616346c1c54195c08732950d428c51f5b0898eb4f09f64d94407dfb432daf028
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | ba04737cf996a2ada5db399df37b736f4511670e4d2537effab97f8f5b774b624ecce3fdcffa42d4edb5e084f62ba0fe526dfabc42b137c29570bc6516d6053a
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | d5f6ab04e5017132e3f76251b6a91791b5b62b8a897ffddada6e4bb7172ad771c72472e63bdfb296240ef8794d57d5c7382dc13af4d6d49a4e716b6215f693e8
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 9878847bcff5c4154eb2fc85f6962a0be8c37a126b23a1351eb2bc65bb22ddabce3f32473f35e91463e464a1189f5caaa764c1019e4217bfd86223b878f4ae69

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | e6f58c79b02d4a6a6d65be9d890cd6cdfeb61a200439f4640f7c3529c38524bf19ba5769514ebeee95a3fc1da2c67049ce563f8cc39635bf025930e74319c944
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | ead6acf178ea4e74dfc64a06f1c2deb88bf940af44e50117706a0b11ba86cf9fce66346bc5a22f857c10994d3e7145eb89d74f5e8c35133a03cc6f18630c9439
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | a253334f379c9df9cb81bc9ce68c1034b091c765e29c80e34d7c0eb7cd8bf743f66e004f45853ecc9de9a5aa5c757736bc6baa09fcd571297edd32a44e719baf
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 44d54ce977e3e97abf5020de3957903a1c06caa4f034ce9c6af3b975bb6ea2559f33bce870e684712815f27f70c5cad793c965ae645689a9ca0bad6360a93f9b
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | 02739821cc67e7146e89646f357475a362ca75a42e266c8c430582743126ee78db22dbfde9a3a51e92a28ca1d35dcdb79dd5a89383a68918b489a8feaf3d01e8
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | 0a2c21b73203531d4772d178a6b1a642147a02171d0563fadc8041e2f8972f6fa2d475e44105b8f17d58f04fffd7eecb3e1f2d0cc5dd8c9ebe581e1b539c0912

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.25.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.25.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.25.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.25.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.25.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.25.0-alpha.2

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - End-to-end testing has been migrated from Ginkgo v1 to v2.

  When running test/e2e via the Ginkgo CLI, the v2 CLI must be used and `-timeout=24h` (or some other, suitable value) must be passed because the default timeout was reduced from 24h to 1h. When running it via `go test`, the corresponding `-args` parameter is `-ginkgo.timeout=24h`. To build the CLI in the Kubernetes repo, use `make all WHAT=github.com/onsi/ginkgo/v2/ginkgo`.
  Ginkgo V2 doesn't accept go test's `-parallel` flags to parallelize Ginkgo specs, please switch to use `ginkgo -p` or `ginkgo -procs=N` instead. ([#109111](https://github.com/kubernetes/kubernetes/pull/109111), [@chendave](https://github.com/chendave)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage, Testing and Windows]

## Changes by Kind

### Deprecation

- Ginkgo.Measure has been deprecated in Ginkgo V2, switch to use gomega/gmeasure instead ([#111065](https://github.com/kubernetes/kubernetes/pull/111065), [@chendave](https://github.com/chendave)) [SIG Autoscaling and Testing]

### API Change

- Added a new feature gate `CheckpointRestore` to enable support to checkpoint containers. If enabled it is possible to checkpoint a container using the newly kubelet API (/checkpoint/{podNamespace}/{podName}/{containerName}). ([#104907](https://github.com/kubernetes/kubernetes/pull/104907), [@adrianreber](https://github.com/adrianreber)) [SIG Node and Testing]
- EndPort field in Network Policy is now promoted to GA

  Network Policy providers that support endPort field now can use it to specify a range of ports to apply a Network Policy.

  Previously, each Network Policy could only target a single port.

  Please be aware that endPort field MUST BE SUPPORTED by the Network Policy provider. In case your provider does not support endPort and this field is specified in a Network Policy, the Network Policy will be created covering only the port field (single port). ([#110868](https://github.com/kubernetes/kubernetes/pull/110868), [@rikatz](https://github.com/rikatz)) [SIG API Machinery, Network and Testing]
- Make PodSpec.Ports' description clearer on how this information is only informational and how it can be incorrect. ([#110564](https://github.com/kubernetes/kubernetes/pull/110564), [@j4m3s-s](https://github.com/j4m3s-s)) [SIG API Machinery, Network and Node]
- On compatible systems, a mounter's Unmount implementation is changed to not return an error when the specified target can be detected as not a mount point. On Linux, the behavior of detecting a mount point depends on `umount` command is validated when the mounter is created. Additionally, mount point checks will be skipped in CleanupMountPoint/CleanupMountWithForce if the mounter's Unmount having the changed behavior of not returning error when target is not a mount point. ([#109676](https://github.com/kubernetes/kubernetes/pull/109676), [@cartermckinnon](https://github.com/cartermckinnon)) [SIG Storage]
- Promote StatefulSet minReadySeconds to GA. This means `--feature-gates=StatefulSetMinReadySeconds=true` are not needed on kube-apiserver and kube-controller-manager binaries and they'll be removed soon following policy at https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecation ([#110896](https://github.com/kubernetes/kubernetes/pull/110896), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG API Machinery, Apps and Testing]
- The Pod `spec.podOS` field is promoted to GA. The `IdentifyPodOS` feature gate unconditionally enabled, and will no longer be accepted as a `--feature-gates` parameter in 1.27. ([#111229](https://github.com/kubernetes/kubernetes/pull/111229), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG API Machinery, Apps and Windows]
- The `minDomains` field in Pod Topology Spread is graduated to beta ([#110388](https://github.com/kubernetes/kubernetes/pull/110388), [@sanposhiho](https://github.com/sanposhiho)) [SIG API Machinery and Apps]

### Feature

- Enable the beta feature ServiceIPStaticSubrange by default ([#110703](https://github.com/kubernetes/kubernetes/pull/110703), [@aojea](https://github.com/aojea)) [SIG Network]
- Enabling CSIMigrationvSphere feature by default. ([#103523](https://github.com/kubernetes/kubernetes/pull/103523), [@divyenpatel](https://github.com/divyenpatel)) [SIG Cloud Provider and Storage]
- Graduated SeccompDefault to `beta`. The Kubelet feature gate is now enabled by default and the configuration/CLI flag still defaults to `false`. ([#110805](https://github.com/kubernetes/kubernetes/pull/110805), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node and Testing]
- Graduated ServerSideFieldValidation to `beta`. Schema validation is performed server-side and requests will receive warnings for any invalid/unknown fields by default. ([#110178](https://github.com/kubernetes/kubernetes/pull/110178), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- In "large" clusters, kube-proxy in iptables mode will now sometimes
  leave unused rules in iptables for a while (up to `--iptables-sync-period`)
  before deleting them. This improves performance by not requiring it to
  check for stale rules on every sync. (In smaller clusters, it will still
  remove unused rules immediately once they are no longer used.)

  (The threshold for "large" used here is currently "1000 endpoints" but
  this is subject to change.) ([#110334](https://github.com/kubernetes/kubernetes/pull/110334), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Introduce new KUBECACHEDIR environment variable to override default discovery cache directory which is $HOME/.kube/cache ([#109479](https://github.com/kubernetes/kubernetes/pull/109479), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- JobTrackingWithFinalizers enabled by default. This feature allows to keep track of the Job progress without relying on Pods staying in the apiserver. ([#110948](https://github.com/kubernetes/kubernetes/pull/110948), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps]
- Kubeadm: make sure the etcd static pod startup probe uses /health?serializable=false while the liveness probe uses /health?serializable=true&exclude=NOSPACE. The NOSPACE exclusion would allow administrators to address space issues one member at a time. ([#110744](https://github.com/kubernetes/kubernetes/pull/110744), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Pod SecurityContext and PodSecurityPolicy supports slash as sysctl separator. ([#106834](https://github.com/kubernetes/kubernetes/pull/106834), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps, Architecture, Auth, Node, Security and Testing]

### Documentation

- Optimization of kubectl Chinese translation ([#110538](https://github.com/kubernetes/kubernetes/pull/110538), [@hwdef](https://github.com/hwdef)) [SIG CLI]

### Bug or Regression

- Adds error message "dry-run can not be used when --force is set" when dry-run and force flags are set in replace command. ([#110326](https://github.com/kubernetes/kubernetes/pull/110326), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Bug fix in test/e2e/framework  Framework.RecordFlakeIfError ([#111048](https://github.com/kubernetes/kubernetes/pull/111048), [@alingse](https://github.com/alingse)) [SIG Testing]
- Do not report terminated container metrics ([#110950](https://github.com/kubernetes/kubernetes/pull/110950), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Node]
- Fix bug where a job sync is not retried when there is a transient ResourceQuota conflict ([#111026](https://github.com/kubernetes/kubernetes/pull/111026), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps]
- Fixes scheduling of cronjobs with @every X schedules. ([#109250](https://github.com/kubernetes/kubernetes/pull/109250), [@d-honeybadger](https://github.com/d-honeybadger)) [SIG Apps]
- For scheduler plugin developers: the scheduler framework's shared PodInformer is now initialized with empty indexers. This enables scheduler plugins to add their extra indexers. Note that only non-conflict indexers are allowed to be added. ([#110663](https://github.com/kubernetes/kubernetes/pull/110663), [@fromanirh](https://github.com/fromanirh)) [SIG Scheduling]
- If the parent directory of the file specified in the `--audit-log-path` argument does not exist, Kubernetes now creates it. ([#110813](https://github.com/kubernetes/kubernetes/pull/110813), [@vpnachev](https://github.com/vpnachev)) [SIG Auth]
- Kubeadm: fix the bug that configurable KubernetesVersion not respected during kubeadm join ([#110791](https://github.com/kubernetes/kubernetes/pull/110791), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: respect user specified image repository when using Kubernetes ci version ([#111017](https://github.com/kubernetes/kubernetes/pull/111017), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: support retry mechanism for removing container in reset phase ([#110837](https://github.com/kubernetes/kubernetes/pull/110837), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Run kubelet, when there is an error exit, print the error log ([#110691](https://github.com/kubernetes/kubernetes/pull/110691), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Node]
- The node annotation alpha.kubernetes.io/provided-node-ip is no longer set ONLY when `--cloud-provider=external`.  Now, it is set on kubelet startup if the `--cloud-provider` flag is set at all, including the deprecated in-tree providers. ([#109794](https://github.com/kubernetes/kubernetes/pull/109794), [@mdbooth](https://github.com/mdbooth)) [SIG Network and Node]
- When metrics are counted, discard the wrong container StartTime metrics ([#110880](https://github.com/kubernetes/kubernetes/pull/110880), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Instrumentation and Node]
- [aws] Fixed a bug which reduces the number of unnecessary calls to STS in the event of assume role failures in the legacy cloud provider ([#110706](https://github.com/kubernetes/kubernetes/pull/110706), [@prateekgogia](https://github.com/prateekgogia)) [SIG Cloud Provider]

### Other (Cleanup or Flake)

- In the event that more than one IngressClass is designated "default", the DefaultIngressClass admission controller will choose one rather than fail. ([#110974](https://github.com/kubernetes/kubernetes/pull/110974), [@kidddddddddddddddddddddd](https://github.com/kidddddddddddddddddddddd)) [SIG Network]
- Kube-proxy: The "userspace" proxy-mode is deprecated on Linux and Windows, despite being the default on Windows.  As of v1.26, the default mode for Windows will change to 'kernelspace'. ([#110762](https://github.com/kubernetes/kubernetes/pull/110762), [@pandaamanda](https://github.com/pandaamanda)) [SIG Network]
- Some apiserver metrics were changed, as follows.
  - `priority_level_seat_count_samples` is replaced with `priority_level_seat_utilization`, which samples every nanosecond rather than every millisecond; the old metric conveyed utilization despite its name.
  - `priority_level_seat_count_watermarks` is removed.
  - `priority_level_request_count_samples` is replaced with `priority_level_request_utilization`, which samples every nanosecond rather than every millisecond; the old metric conveyed utilization despite its name.
  - `priority_level_request_count_watermarks` is removed.
  - `read_vs_write_request_count_samples` is replaced with `read_vs_write_current_requests`, which samples every nanosecond rather than every second; the new metric, like the old one, measures utilization when the max-in-flight filter is used and number of requests when the API Priority and Fairness filter is used.
  - `read_vs_write_request_count_watermarks` is removed. ([#110104](https://github.com/kubernetes/kubernetes/pull/110104), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery, Instrumentation and Testing]

## Dependencies

### Added
- github.com/go-task/slim-sprig: [348f09d](https://github.com/go-task/slim-sprig/tree/348f09d)
- github.com/google/pprof: [94a9f03](https://github.com/google/pprof/tree/94a9f03)
- github.com/ianlancetaylor/demangle: [28f6c0f](https://github.com/ianlancetaylor/demangle/tree/28f6c0f)
- github.com/onsi/ginkgo/v2: [v2.1.4](https://github.com/onsi/ginkgo/v2/tree/v2.1.4)

### Changed
- github.com/antlr/antlr4/runtime/Go/antlr: [ad29539 → f25a4f6](https://github.com/antlr/antlr4/runtime/Go/antlr/compare/ad29539...f25a4f6)
- github.com/google/cel-go: [v0.11.2 → v0.12.3](https://github.com/google/cel-go/compare/v0.11.2...v0.12.3)
- github.com/onsi/gomega: [v1.10.1 → v1.19.0](https://github.com/onsi/gomega/compare/v1.10.1...v1.19.0)
- golang.org/x/net: cd36cc0 → 27dd868
- golang.org/x/sys: 3681064 → a9b59b0
- golang.org/x/tools: 897bd77 → v0.1.10
- google.golang.org/genproto: 1973136 → c8bf987
- google.golang.org/protobuf: v1.27.1 → v1.28.0
- k8s.io/klog/v2: v2.70.0 → v2.70.1
- k8s.io/kube-openapi: 31174f5 → 011e075
- sigs.k8s.io/json: 9f7c6b3 → f223a00

### Removed
- github.com/nxadm/tail: [v1.4.4](https://github.com/nxadm/tail/tree/v1.4.4)
- github.com/onsi/ginkgo: [v1.14.0](https://github.com/onsi/ginkgo/tree/v1.14.0)
- gopkg.in/tomb.v1: dd63297



# v1.25.0-alpha.2


## Downloads for v1.25.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes.tar.gz) | 72eb69d6fa1af801e98f207711ff30a2f77c95e71174386976d43e4704f8ff4a88e6e04c1a15d4b9806fc9ada321e39c74bed751049bb45a31f6fdbd85acde42
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-src.tar.gz) | 94dfca305c9bae36ff6984b06f335720db5dc6df41852012c8284424fbf021814461de0b376c3eed850f65e353b544ca66b0fc2d86c3b46dd6701c5380600e5b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | d71f74e1d9d3fbc59e25df8c43b6af360bcdf3ac3597411092e22359ecf6fc7f1091f01996865603ca0d65bfe1764bbc2cfa1aa94b5f9fad4b457986a880d42d
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 824a641a183a6a88a71a5d1cffe1c4db40a59e455dae0586bb379d71095fb9011c0a6c4a1338189c3c7615cec8b71c355d3ef18ff15bb72c82ba08027403b0be
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-linux-386.tar.gz) | b6bfb3d57f5842bb30cdf6b00442a43cb555dff741575a7b0ee43ffa8661b23db8e540a8cf2fd47c9ccf9454e0be593a4a06bc0a308d1861cc58cf3604739626
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | a71cd1bdf181a0a8fe0484aee393761b4fc50dbf6b71a589e2f3322fe9452302675a2d8deb3b087e9cc7e7495e5c7c3b7f2f70e065c65c83cf23561b8d32d97f
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 6cec180b1b623277b7ff9a680714f005ec4a606ab3297ac8159ed463cf268aa5668a4a8c5cddd63512869c46258193b9d0bedd7bae8a1e355c6ea84affbd9a7d
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 9a6d51a396a96f5beb49327fae9aaec5e391fbbb44db4698f9b25fce1882b3d921b8b65e10b10f4b840f14e5e6c210f4a318c52c627e327adc9742dc5d909d6c
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 533b2aedea645803cb282f84fbd8c38ee3146edaf33ff86dad85e9b9602a33f4ec19eceaf60cc3d4c02ef811a4a35c1c8cedad4d702c8c57f1a8c64b3d1c2674
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | fa83bc04fb0802638929fd4174342ec8ac567b996d5b44f6e2ce309d7084bc496de448b8f62479e4b1630890e7851afd3cecdce9926ce1c7e931fef5c1dfc063
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 70c7cbcc0337eeb0e31eabf9b2309a82240e17858637c01e24af24830b971ac58aa3b8d6e2628b9eee29a075131c4037b3a18e33cf214ae8ca10b0341e782b2d
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | e04d4e605536a1a2aef33e699e22c49649e3bcf1330aae57c511914cbdf6ca5c1c7056a4a2aa7109900b58ef836ccf91572adbeddd274a75d8136cf8300a4e5a
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | cbc2075880daae0aed15d665e3bab50311a23616ab6b562bf5873754a83623ae01c8886c588fa36982eca4bc1d6d25f5e9bcaa9384b67dfbe376ed24bf1ba887

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | 0feb8e90c5defdb639154f4e5e87e4ec1ad2e3fdd2eb9f667df5e6ea4112fe0ab64d4c194ccb6bba812bdce0b41aed2f04e72b3fd34b08fbca9e92feec26015f
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | e1104e286e710367e3254cd546f31f639c8ce2e1539e6d08a1b41e3af051535bca990922af29e2a9d63f6328bb19630ed7eb945176ff0e9d8c6dea4c1a540f13
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | d08a78a9791b6797cabd415c7b9a996e1dba5e0eef9e7eab9b09110018091fdec6e5ee7cb692960f5ef0805d3e4181f49507f09ba1ef3dfeb18adb58e09e1820
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 23116faf8df942c1d7da7ebf02dcc78740a7ede7d9ec72473e0390ab7203cfc3fd4cd5bc65ff818825b924782c7cd9df742e1fad177f862e32b15a1a4d7d0e0c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 4853481452158658c3eecc3107f9c95df408b6ec99d655c6167f66374939d4dc030883376d1777b4420024473ffb7365d3a9c6f652032fb908377aa9648d61d5

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 710ac7c343259319329a63accb4295f81d7ae63a104412c0d909449492c6091038d2ffc677169100da678759db02a9d79c57b340753e8e30fb3e2b5675d21ebb
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 372ca94ece9ac5ff862abb7fdb6732f23d7f665e1208e38e7aac642bcbeeb95801230872f48012502d1bbe9d7b0e0a22ff7b019a54c0a7e34b0deb153ca3f9ce
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | de75500809eb3ad7ee2772d0315e3e7aa697c8a1888c6eb57479770f8c92e9778088e0c970127462cc8e7bdd22016c38344cff0854100707e942128c47fa30cf
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 1b4f54ec0037859a8410bbfecabbbc8de656576d121af35450182e7ec9708bc57cbaaa4fd36d79ddf368ebf74935e6f368365d1a9d0d93ec60982977c9900dca
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | efbfae7e592c9edb6830dbfafb0e1c4feadaee1d2c7211946789b4aac0f3fd56c3b1abb4f7c30617e8d0e51cef1f1df2b3cc582ecc543351c0918efc8e3fe6e9
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 290649bd96432aa9e2042fae7a8412ef7e0b6c16dae9551b9320ced7c5a1c805cf112df1a5ede327e5474cd32ff04f9841eb2e84dec992c24882d5d71c178064

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.25.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.25.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.25.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.25.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.25.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.25.0-alpha.1

## Changes by Kind

### API Change

- The Go API for logging configuration in k8s.io/component-base was moved to k8s.io/component-base/logs/api/v1. The configuration file format and command line flags are the same as before. ([#105797](https://github.com/kubernetes/kubernetes/pull/105797), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Cluster Lifecycle, Instrumentation, Node, Scheduling and Testing]
- The PodSecurity admission plugin has graduated to GA and is enabled by default. The admission configuration version has been promoted to `pod-security.admission.config.k8s.io/v1`. ([#110459](https://github.com/kubernetes/kubernetes/pull/110459), [@wangyysde](https://github.com/wangyysde)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Storage and Testing]

### Feature

- Adds KMS v2alpha1 API ([#110201](https://github.com/kubernetes/kubernetes/pull/110201), [@aramase](https://github.com/aramase)) [SIG API Machinery and Auth]
- Feature gate `CSIMigration` is locked to enabled. CSIMigration is GA now. The feature gate will be removed in 1.27 ([#110410](https://github.com/kubernetes/kubernetes/pull/110410), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG Apps, Auth, Scheduling, Storage and Testing]
- Kubeadm: the preferred pod anti-affinity for CoreDNS is now enabled by default ([#110593](https://github.com/kubernetes/kubernetes/pull/110593), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- These changes promote the CSIMigrationPortworx feature gate to Beta ([#110411](https://github.com/kubernetes/kubernetes/pull/110411), [@trierra](https://github.com/trierra)) [SIG Storage]
- Updating base image for Windows pause container images to one built on Windows machines to address limitations of building Windows container images on Linux machines. ([#110379](https://github.com/kubernetes/kubernetes/pull/110379), [@marosset](https://github.com/marosset)) [SIG Windows]

### Documentation

- EndpointSlices with Pod referencing Nodes that doesn't exist couldn't be created or updated.
  The behavior on the EndpointSlice controller has been modified to update the EndpointSlice without the Pods that reference non-existing Nodes, and keep retrying until all Pods reference existing Nodes.
  However, if service.Spec.PublishNotReadyAddresses is set, all the Pods are published without retrying.
  Fixed EndpointSlices metrics to reflect correctly the number of desired EndpointSlices when no endpoints are present. ([#110639](https://github.com/kubernetes/kubernetes/pull/110639), [@aojea](https://github.com/aojea)) [SIG Apps and Network]

### Bug or Regression

- Client-go: fixed an error in the fake client when submitting create API requests to subresources like pods/eviction ([#110425](https://github.com/kubernetes/kubernetes/pull/110425), [@LY-today](https://github.com/LY-today)) [SIG API Machinery]
- FibreChannel volume plugin may match the wrong device and wrong associated devicemapper parent.This may cause a disater that pods attach wrong disks. ([#110719](https://github.com/kubernetes/kubernetes/pull/110719), [@xakdwch](https://github.com/xakdwch)) [SIG Storage]
- Fix "dbus: connection closed by user" error after dbus daemon restart. ([#110496](https://github.com/kubernetes/kubernetes/pull/110496), [@kolyshkin](https://github.com/kolyshkin)) [SIG Node]
- Fix a bug that caused the wrong result length when using --chunk-size and --selector together ([#110652](https://github.com/kubernetes/kubernetes/pull/110652), [@Abirdcfly](https://github.com/Abirdcfly)) [SIG API Machinery and Testing]
- Fixes scheduling of cronjobs with @every X schedules. ([#109250](https://github.com/kubernetes/kubernetes/pull/109250), [@d-honeybadger](https://github.com/d-honeybadger)) [SIG Apps]
- Fixing issue on Windows nodes where HostProcess containers may not be created as expected. ([#110140](https://github.com/kubernetes/kubernetes/pull/110140), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Kubeadm: handle dup `unix://` prefix in node annotation ([#110656](https://github.com/kubernetes/kubernetes/pull/110656), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubelet: add retry of checking Unix domain sockets on Windows nodes for the plugin registration mechanism ([#110075](https://github.com/kubernetes/kubernetes/pull/110075), [@luckerby](https://github.com/luckerby)) [SIG Node and Windows]
- Removed unused flags from kubectl run command ([#110668](https://github.com/kubernetes/kubernetes/pull/110668), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- This change picks up the latest GCE pinhole firewall feature, which introduces destination-ranges in the ingress firewall-rules. It restricts the access to the backend IPs via allowing traffic via allowing traffic through ILB or NetLB IP only. This change does NOT change the existing ILB or NetLB behavior. ([#109510](https://github.com/kubernetes/kubernetes/pull/109510), [@sugangli](https://github.com/sugangli)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- Volumes are no longer detached from healthy nodes after 6 minutes timeout. 6 minute force-detach timeout is used only for unhealthy nodes (`node.status.conditions["Ready"] != true`). ([#110721](https://github.com/kubernetes/kubernetes/pull/110721), [@jsafrane](https://github.com/jsafrane)) [SIG Apps]
- `kubeadm certs renew` and   `kubeadm certs check-expiration`  now honor the `cert-dir` on a working kubernetes cluster. ([#110709](https://github.com/kubernetes/kubernetes/pull/110709), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]

### Other (Cleanup or Flake)

- Improve kubectl run and debug attach problems error ([#110764](https://github.com/kubernetes/kubernetes/pull/110764), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- Kubelet: silence flag output on errors ([#110728](https://github.com/kubernetes/kubernetes/pull/110728), [@howardjohn](https://github.com/howardjohn)) [SIG Node]
- Remove release-1.20 from prom bot due to eol ([#110748](https://github.com/kubernetes/kubernetes/pull/110748), [@cpanato](https://github.com/cpanato)) [SIG Release]

## Dependencies

### Added
- github.com/golang/snappy: [v0.0.3](https://github.com/golang/snappy/tree/v0.0.3)
- google.golang.org/grpc/cmd/protoc-gen-go-grpc: v1.1.0

### Changed
- cloud.google.com/go: v0.81.0 → v0.97.0
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [ea6160c → f118173](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/ea6160c...f118173)
- github.com/google/martian/v3: [v3.1.0 → v3.2.1](https://github.com/google/martian/v3/compare/v3.1.0...v3.2.1)
- github.com/googleapis/gax-go/v2: [v2.0.5 → v2.1.1](https://github.com/googleapis/gax-go/v2/compare/v2.0.5...v2.1.1)
- github.com/opencontainers/runc: [v1.1.1 → v1.1.3](https://github.com/opencontainers/runc/compare/v1.1.1...v1.1.3)
- github.com/seccomp/libseccomp-golang: [3879420 → f33da4d](https://github.com/seccomp/libseccomp-golang/compare/3879420...f33da4d)
- google.golang.org/api: v0.46.0 → v0.60.0
- k8s.io/klog/v2: v2.60.1 → v2.70.0
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.30 → v0.0.32

### Removed
- github.com/google/pprof: [cbba55b](https://github.com/google/pprof/tree/cbba55b)
- github.com/ianlancetaylor/demangle: [28f6c0f](https://github.com/ianlancetaylor/demangle/tree/28f6c0f)
- github.com/jstemmer/go-junit-report: [v0.9.1](https://github.com/jstemmer/go-junit-report/tree/v0.9.1)



# v1.25.0-alpha.1


## Downloads for v1.25.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes.tar.gz) | 01aa5755e4d58e2f5e449af62342155e784973f504f831b52aed13fa941075ea06e8fbcadca2445ac570318e7dd9f1042e0447bb74d6042e447dc87dd472b3fc
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-src.tar.gz) | e62170247fb7c50f52274a6e86fde244766cf1cf86bab2913905e9063d03ce5a3882042c755291c766f66b4f4ab630e126d1a9446e31392f77c90a398af57570

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | bb9408704de0f2adac81031d347cfa229a6aef413102a116193f50bf690eac8443bb97cfe59044a1857f7859b462f98fef5c9b7db52f9895d70d399e0d381f19
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 0d8aafe99969241019685348bd40c9a5e252110121a9c19c6008874c54c4e6c62eb9470cc55e2eab300bebbaa78696539989a8a23d1e958a222aeabadad9e740
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 67db15a269e252c8945b40e049137a8e7575128f9890ffb121f8dd72b1a79f55aa92aa9e66d5299ef05dcd30bdf1856878f0373ba8c33c46a161cad696dd1c0a
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 58079330c0ebb41806782a7675bf37e8e6466d37ec50d75466b6a2633918d36326863c55a258f0cec8134f3a3abaaf66ab51be70cdb981499ed6f411d58d04fd
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | afc3b8b30ee5a4baef2b8bf69603884dc488ecd16ab790cdb94859abcbfee0103aff858f7e5778856a7265c54d1eef9df392b0f10ecd26c929a2fe89dcb292e5
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | ef1417a70c0a3a428d966e7711f244a2553d85a330898c461a826082533358e0f4a220fd3ccf9295554a6e284fd89d2ebfc37b30cc69d323bf16ce4b9f5e02a4
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 5aec975459b3141f7dbb38c86c86fb89ee75470b77095280d2c4e6f5e9f8a4c3b5ef323cd6e4a4403c90d725276621c2d85e749090ce93648f238f289de83ceb
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 91dbe24885b0fad1bef4c0d60ac4c36ba78669e912ec51f7cfb9d4cc33e6f1ece4ee832a96eedb4d117d7defeaffa539e50b42433caf5145543463deb26146fb
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-windows-386.tar.gz) | b9afed1db794c6df8a325c2831643965a7f4c0633b1a8e73432a08bc21a63f57d98a67115c5d8ede46e8b3ea002c1c2ef7d16d62e53530eb9cff92471f06e840
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 95f80902ecae4fac9aa2ef863c63bef752b2554a2e0b07cebd49db37d0ce06b5aafcad779793852634f92e1f0d6a3da4dec48387a361a4ede3f9bd77ee2b70ff
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | dee0a8658117f90c138e4bcc5dcba7bc201dcac3a993806eaeacc4ed1ba4a4b2aed0eed395169f372010bcde1966b5ba60734729247f7dc7610421ebf70746ba

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | 144c23ff1718b3635cb227c5bda29e7f57a63dc80b9aa5120046deb8cff44946a7ca1844303f6fcaebe3d9b9c91f41f57ec96cb407862322fb783068819ab917
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | b989cacd95bce88d70a2f17200f5d50e06c66e184bb1ab72ee94e65174a1cb8acbbbfb29dc22e85a0893cae31b903ef93ecc62aa7425d8fcbb35c365b588e72c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 0a2ce7c6f016e52149cbe5df5f35db47fcd0c4de5dd04e8054305e3f6ad4554719d658af7cf6ddb40e8e3420bfbc32bbfc87d9f209a9e59813a17f51c4b0581d
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 63bb9ec0e88a6560cb9a415ee8d5656e5bfd9f63a8388f7abd960b72b6d58e0ba664f7aef7aaa6d436aafcc595a19766b85b1af9d076cf3985786888bdd8a258
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 8f99b8c0133771d0326b7c11ba65009c245f40fe71b1d7737343860e022d4be3c032df8dedf728184d017014c17df93df971622ced800788045fd234b50f36b4

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 02f764c2f83992f0820d0186898da449616c4f6ce8a771990ac1f17e277ae369bfeadd24aa0ce2405c6386ff308556437e4f968401b2ef4fb6f344a0a6e60ebd
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | 6d87f6554d8051be21c268c9d847c1b66cf7785a9cb781b44494f013b9f1afb1018ad2ce54cedb62b33e93517cd630f3168ab63a6b03e7badde70e300bab9a9a
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 9d8d34cb09a73db6c3dabd04975fddbf7387db0f0a241c285fb7995c7256fc5ca37285b680f0f978438e5ca92451f163e1e4c90642c82101d415aa40b06afa2f
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 6c5d27306f65ab4eab19f0b39cadd5adb33a3dc3ef602cf4c1e7afd51ac250dea8fef58f748bdbd651d0d77806442b214d098b8a40910627b8a359e482a2706a
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | c8d45846d19e8179969f339bfb4cd13c6d952990b30d301bbb345ac17c4ddaab38c22f798986af6ded6aad96e8415c92c86f2f8fb7fe1bda8b0aba6216758112
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.25.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | dd1607634a718d790662420cbaa30af6d7c89b6f9ae64cef5ee224e42c32fe318bf6e1b2180894723fc890c315c4943004dd16c59f94e6daaa09e4435aed1e07

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.25.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.25.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.25.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.25.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.25.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Deprecated beta APIs scheduled for removal in 1.25 are no longer served. See https://kubernetes.io/docs/reference/using-api/deprecation-guide/#v1-25 for more information. ([#108797](https://github.com/kubernetes/kubernetes/pull/108797), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Instrumentation and Testing]
  - No action required; No API/CLI changed; Add new Windows Image Support ([#110333](https://github.com/kubernetes/kubernetes/pull/110333), [@liurupeng](https://github.com/liurupeng)) [SIG Cloud Provider and Windows]
  - There is a new OCI image registry (registry.k8s.io) that can be used to pull kubernetes images. The old registry (k8s.gcr.io) will continue to be supported for the foreseeable future, but the new name should perform better because it frontends equivalent mirrors in other clouds.  Please point your clusters to the new registry going forward.

  Admission/Policy integrations that have an allowlist of registries need to include "registry.k8s.io" alongside "k8s.gcr.io".
  Air-gapped environments and image garbage-collection configurations will need to update to pre-pull and preserve required images under "registry.k8s.io" as well as "k8s.gcr.io". ([#109938](https://github.com/kubernetes/kubernetes/pull/109938), [@dims](https://github.com/dims)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, K8s Infra, Node, Release, Scalability, Storage and Testing]

## Changes by Kind

### Deprecation

- Kube-controller-manager:  'deleting-pods-qps'  'deleting-pods-burst'  'register-retry-count' flags are removed. ([#109612](https://github.com/kubernetes/kubernetes/pull/109612), [@pandaamanda](https://github.com/pandaamanda)) [SIG API Machinery]
- Kubeadm: during "upgrade apply/diff/node", in case the "ClusterConfiguration.imageRepository" stored in the "kubeadm-config" ConfigMap contains the legacy "k8s.gcr.io" repository, modify it to the new default "registry.k8s.io". Reflect the change in the in-cluster ConfigMap only during "upgrade apply". ([#110343](https://github.com/kubernetes/kubernetes/pull/110343), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: graduate the kubeadm specific feature gate UnversionedKubeletConfigMap to GA and lock it to "true" by default. The kubelet related ConfigMap and RBAC rules are now locked to have a simplified naming "*kubelet-config" instead of the legacy naming "*kubelet-config-x.yy", where "x.yy" was the version of the control plane. If you have previously used the old naming format with UnversionedKubeletConfigMap=false, you must manually copy the config map from kube-system/kubelet-config-x.yy to kube-system/kubelet-config before upgrading to 1.25. ([#110327](https://github.com/kubernetes/kubernetes/pull/110327), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: stop applying the "node-role.kubernetes.io/master:NoSchedule" taint to control plane nodes for new clusters. Remove the taint from existing control plane nodes during "kubeadm upgrade apply" ([#110095](https://github.com/kubernetes/kubernetes/pull/110095), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- The `gcp` and `azure` auth plugins have been removed from client-go and kubectl. See https://github.com/Azure/kubelogin and https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke for details about the cloud-specific replacements. ([#110013](https://github.com/kubernetes/kubernetes/pull/110013), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]
- The beta `PodSecurityPolicy` admission plugin, deprecated since 1.21, is removed. Follow the instructions at https://kubernetes.io/docs/tasks/configure-pod-container/migrate-from-psp/ to migrate to the built-in PodSecurity admission plugin (or to another third-party policy webhook) prior to upgrading to v1.25. ([#109798](https://github.com/kubernetes/kubernetes/pull/109798), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Auth, Cloud Provider, Instrumentation, Node, Security, Storage and Testing]

### API Change

- Introduce NodeInclusionPolicies to specify nodeAffinity/nodeTaint strategy when calculating pod topology spread skew. ([#108492](https://github.com/kubernetes/kubernetes/pull/108492), [@kerthcet](https://github.com/kerthcet)) [SIG API Machinery, Apps, Scheduling and Testing]
- The `metadata.clusterName` field is completely removed. This should not have any user-visible impact. ([#109602](https://github.com/kubernetes/kubernetes/pull/109602), [@lavalamp](https://github.com/lavalamp)) [SIG API Machinery, Apps, Auth and Testing]
- This release add support for NodeExpandSecret for CSI driver client which enables the CSI drivers to make use of this secret while performing node expansion operation based on the user request. Previously there was no secret  provided as part of the nodeexpansion call, thus CSI drivers were not make use of the same while expanding the volume at node side. ([#105963](https://github.com/kubernetes/kubernetes/pull/105963), [@zhucan](https://github.com/zhucan)) [SIG API Machinery, Apps and Storage]

### Feature

- Added sum feature to `kubectl top pod` ([#105100](https://github.com/kubernetes/kubernetes/pull/105100), [@lauchokyip](https://github.com/lauchokyip)) [SIG CLI]
- Adds the `Apply` and `ApplyStatus` methods to the dynamic `ResourceInterface` ([#109443](https://github.com/kubernetes/kubernetes/pull/109443), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery and Testing]
- Graduate ServiceIPStaticSubrange feature to beta (disabled by default) ([#110419](https://github.com/kubernetes/kubernetes/pull/110419), [@aojea](https://github.com/aojea)) [SIG Network]
- Kube-up now includes CoreDNS version v1.9.3 ([#110488](https://github.com/kubernetes/kubernetes/pull/110488), [@mzaian](https://github.com/mzaian)) [SIG Cloud Provider]
- Kubeadm: Added support for additional authentication strategies in `kubeadm join` with discovery/kubeconfig file: client-go authentication plugins (`exec`), `tokenFile`, and `authProvider` ([#110553](https://github.com/kubernetes/kubernetes/pull/110553), [@tallaxes](https://github.com/tallaxes)) [SIG Cluster Lifecycle]
- Kubeadm: add support for the flag "--print-manifest" to the addon phases "kube-proxy" and "coredns" of "kubeadm init phase addon". If this flag is used kubeadm will not apply a given addon and instead print to the terminal the API objects that will be applied. ([#109995](https://github.com/kubernetes/kubernetes/pull/109995), [@wangyysde](https://github.com/wangyysde)) [SIG Cluster Lifecycle]
- Kubeadm: enhance the "patches" functionality to be able to patch kubelet config files containing v1beta1.KubeletConfiguration. The new patch target is called "kubeletconfiguration" (e.g. patch file "kubeletconfiguration+json.json"). This makes it possible to apply node specific KubeletConfiguration options during "init", "join" and "upgrade", while the main KubeletConfiguration that is passed to "init" as part of the "--config" file can still act as the global / stored in the cluster KubeletConfiguration. ([#110405](https://github.com/kubernetes/kubernetes/pull/110405), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: modify the etcd static Pod liveness and readyness probes to use a new etcd 3.5.3+ HTTP(s) health check endpoint "/health?serializable=true" that allows to track the health of individual etcd members and not fail all members if a single member is not healthy in the etcd cluster. ([#110072](https://github.com/kubernetes/kubernetes/pull/110072), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: support experimental JSON/YAML output for "kubeadm upgrade plan" with the "--output" flag ([#108447](https://github.com/kubernetes/kubernetes/pull/108447), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: update CoreDNS to v1.9.3. ([#110489](https://github.com/kubernetes/kubernetes/pull/110489), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubectl: support multiple resources for kubectl rollout status ([#108777](https://github.com/kubernetes/kubernetes/pull/108777), [@pjo256](https://github.com/pjo256)) [SIG CLI and Testing]
- Kubernetes is now built with Golang 1.18.2 ([#110043](https://github.com/kubernetes/kubernetes/pull/110043), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with Golang 1.18.3 ([#110421](https://github.com/kubernetes/kubernetes/pull/110421), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Lock CSIMigrationAzureDisk feature gate to default ([#110491](https://github.com/kubernetes/kubernetes/pull/110491), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- MaxUnavailable for StatefulSets, allows faster RollingUpdate by taking down more than 1 pod at a time. The number of pods you want to take down during a RollingUpdate is configurable using maxUnavailable parameter. ([#109251](https://github.com/kubernetes/kubernetes/pull/109251), [@krmayankk](https://github.com/krmayankk)) [SIG Apps and CLI]
- Return a warning when applying a pod-security.kubernetes.io label to a PodSecurity-exempted namespace.
  Stop including the pod-security.kubernetes.io/exempt=namespace audit annotation on namespace requests. ([#109680](https://github.com/kubernetes/kubernetes/pull/109680), [@tallclair](https://github.com/tallclair)) [SIG Auth]
- TopologySpreadConstraints will be shown in describe command for pods, deployments, daemonsets, etc. ([#109563](https://github.com/kubernetes/kubernetes/pull/109563), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Updat debian-base, debian-iptables, and setcap images:
  - debian-base:bullseye-v1.3.0
  - debian-iptables:bullseye-v1.4.0
  - setcap:bullseye-v1.3.0 ([#110558](https://github.com/kubernetes/kubernetes/pull/110558), [@wespanther](https://github.com/wespanther)) [SIG Architecture, Release and Testing]
- When using the OpenStack legacy cloud provider, kubelet and KCM will ignore unknown configuration directives rather than failing to start. ([#109709](https://github.com/kubernetes/kubernetes/pull/109709), [@mdbooth](https://github.com/mdbooth)) [SIG Cloud Provider]

### Failing Test

- E2e tests: the e2e image, agnhost:2.38, has a bug and it hangs instead of exiting if a SIGTERM signal is received and the shutdown-delay option is 0` ([#110214](https://github.com/kubernetes/kubernetes/pull/110214), [@aojea](https://github.com/aojea)) [SIG Testing]

### Bug or Regression

- Allow expansion of ephemeral volumes ([#109987](https://github.com/kubernetes/kubernetes/pull/109987), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Apiserver: fix audit of loading more than one webhooks ([#110145](https://github.com/kubernetes/kubernetes/pull/110145), [@sxllwx](https://github.com/sxllwx)) [SIG API Machinery and Auth]
- Do not raise an error when setting a label with the same value, just ignore it. ([#105936](https://github.com/kubernetes/kubernetes/pull/105936), [@zigarn](https://github.com/zigarn)) [SIG CLI]
- EndpointSlices marked for deletion are now ignored during reconciliation. ([#109624](https://github.com/kubernetes/kubernetes/pull/109624), [@aryan9600](https://github.com/aryan9600)) [SIG Apps and Network]
- Etcd: Update to v3.5.4 ([#110033](https://github.com/kubernetes/kubernetes/pull/110033), [@mk46](https://github.com/mk46)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Fix JobTrackingWithFinalizers that:
  - was declaring a job finished before counting all the created pods in the status
  - was leaving pods with finalizers, blocking pod and job deletions

  JobTrackingWithFinalizers is still disabled by default. ([#109486](https://github.com/kubernetes/kubernetes/pull/109486), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Testing]
- Fix a bug where CRI implementations that use cAdvisor stats provider (CRI-O) don't evict pods when their logs exceed ephemeral storage limit. ([#108115](https://github.com/kubernetes/kubernetes/pull/108115), [@haircommander](https://github.com/haircommander)) [SIG Node]
- Fix a bug where CSI migration doesn't count inline volumes for attach limit. ([#107787](https://github.com/kubernetes/kubernetes/pull/107787), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG Scheduling and Storage]
- Fix a bug where metrics are not recorded during Preemption(PostFilter). ([#108727](https://github.com/kubernetes/kubernetes/pull/108727), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Fix a data race in authentication between AuthenticatedGroupAdder and cached token authenticator. ([#109969](https://github.com/kubernetes/kubernetes/pull/109969), [@sttts](https://github.com/sttts)) [SIG API Machinery and Auth]
- Fix bug that prevented informer/reflector callers from unwrapping and catching specific API errors by type. ([#110076](https://github.com/kubernetes/kubernetes/pull/110076), [@karlkfi](https://github.com/karlkfi)) [SIG API Machinery]
- Fix bug that prevented the job controller from enforcing activeDeadlineSeconds when set ([#110294](https://github.com/kubernetes/kubernetes/pull/110294), [@harshanarayana](https://github.com/harshanarayana)) [SIG Apps and Scheduling]
- Fix for volume reconstruction of CSI ephemeral volumes ([#108997](https://github.com/kubernetes/kubernetes/pull/108997), [@dobsonj](https://github.com/dobsonj)) [SIG Node, Storage and Testing]
- Fix image pulling failure when IMDS is unavailable in kubelet startup ([#110523](https://github.com/kubernetes/kubernetes/pull/110523), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix incorrectly report scope for request_duration_seconds and request_slo_duration_seconds metrics for POST custom resources API calls. ([#110009](https://github.com/kubernetes/kubernetes/pull/110009), [@azylinski](https://github.com/azylinski)) [SIG Instrumentation]
- Fix printing resources with int64 fields ([#110408](https://github.com/kubernetes/kubernetes/pull/110408), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fix spurious kube-apiserver log warnings related to openapi v3 merging when creating or modifying CustomResourceDefinition objects ([#109880](https://github.com/kubernetes/kubernetes/pull/109880), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Testing]
- Fix the bug that a ServiceIPStaticSubrange enabled cluster assigns duplicate IP addresses when the dynamic block is exhausted. ([#109928](https://github.com/kubernetes/kubernetes/pull/109928), [@tksm](https://github.com/tksm)) [SIG Network]
- Fix the bug that the metrics for the cluster IP allocator are incorrectly reported. ([#110027](https://github.com/kubernetes/kubernetes/pull/110027), [@tksm](https://github.com/tksm)) [SIG Instrumentation]
- Fixed a kubelet issue that could result in invalid pod status updates to be sent to the api-server where pods would be reported in a terminal phase but also report a ready condition of true in some cases. ([#110256](https://github.com/kubernetes/kubernetes/pull/110256), [@bobbypage](https://github.com/bobbypage)) [SIG Node and Testing]
- Fixed a long-standing but very obscure bug involving Services of type LoadBalancer with multiple IPs and a LoadBalancerSourceRanges that overlaps the node IP. ([#109826](https://github.com/kubernetes/kubernetes/pull/109826), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Fixes strict server-side field validation treating metadata fields as unknown fields ([#109268](https://github.com/kubernetes/kubernetes/pull/109268), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Kube-apiserver: Get, GetList and Watch requests that should be served by the apiserver cacher during shutdown will be rejected to avoid a deadlock situation leaving requests hanging. ([#108414](https://github.com/kubernetes/kubernetes/pull/108414), [@aojea](https://github.com/aojea)) [SIG API Machinery]
- Kubeadm: only taint control plane nodes when the legacy "master" taint is present. This avoids a bug where "kubeadm upgrade" will re-taint a control plane node with the new "control plane" taint even if the user explicitly untainted the node. ([#109840](https://github.com/kubernetes/kubernetes/pull/109840), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: pass the host OS environment variables when executing "crictl" during image pulls. This fixes a bug where *PROXY environment variables did not affect crictl's internet connectivity. ([#110134](https://github.com/kubernetes/kubernetes/pull/110134), [@mk46](https://github.com/mk46)) [SIG Cluster Lifecycle]
- Kubelet: wait for node allocatable ephemeral-storage data ([#101882](https://github.com/kubernetes/kubernetes/pull/101882), [@jackfrancis](https://github.com/jackfrancis)) [SIG Node and Storage]
- Kubernetes now correctly handles "search ." in the host's resolv.conf file by preserving the "." entry in the "resolv.conf" that the kubelet writes to pods. ([#109441](https://github.com/kubernetes/kubernetes/pull/109441), [@Miciah](https://github.com/Miciah)) [SIG Network and Node]
- ManagedFields time is correctly updated when the value of a managed field is modified. ([#110058](https://github.com/kubernetes/kubernetes/pull/110058), [@glebiller](https://github.com/glebiller)) [SIG API Machinery]
- Manual change of a failed job condition status to False does not result in duplicate conditions ([#110292](https://github.com/kubernetes/kubernetes/pull/110292), [@mimowo](https://github.com/mimowo)) [SIG Apps]
- OpenAPI will no longer duplicate these schemas:
  - io.k8s.apimachinery.pkg.apis.meta.v1.DeleteOptions_v2
  - io.k8s.apimachinery.pkg.apis.meta.v1.ObjectMeta_v2
  - io.k8s.apimachinery.pkg.apis.meta.v1.OwnerReference_v2
  - io.k8s.apimachinery.pkg.apis.meta.v1.StatusDetails_v2
  - io.k8s.apimachinery.pkg.apis.meta.v1.Status_v2 ([#110179](https://github.com/kubernetes/kubernetes/pull/110179), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Testing]
- Panics while calling validating admission webhook are caught and honor the fail open or fail closed setting. ([#108746](https://github.com/kubernetes/kubernetes/pull/108746), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Pods will now post their readiness during termination. ([#110191](https://github.com/kubernetes/kubernetes/pull/110191), [@rphillips](https://github.com/rphillips)) [SIG Network, Node and Testing]
- Reduced time taken to sync proxy rules on Windows kube-proxy with kernelspace mode ([#109124](https://github.com/kubernetes/kubernetes/pull/109124), [@daschott](https://github.com/daschott)) [SIG Network, Release and Windows]
- The kube-proxy `sync_proxy_rules_no_endpoints_total` metric now only counts local-traffic-policy services which have remote endpoints but not local endpoints. ([#109782](https://github.com/kubernetes/kubernetes/pull/109782), [@danwinship](https://github.com/danwinship)) [SIG Network]
- The pod phase lifecycle guarantees that terminal Pods, those whose states are Unready or Succeeded, can not regress and will have all container stopped. Hence, terminal Pods will never be reachable and should not publish their IP addresses on the Endpoints or EndpointSlices, independently of the Service TolerateUnready option. ([#110255](https://github.com/kubernetes/kubernetes/pull/110255), [@robscott](https://github.com/robscott)) [SIG Apps, Network, Node and Testing]
- Upgrade Azure/go-autorest/autorest to v0.11.27 ([#110371](https://github.com/kubernetes/kubernetes/pull/110371), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]

### Other (Cleanup or Flake)

- Add missing powershell option to kubectl completion command short description ([#109773](https://github.com/kubernetes/kubernetes/pull/109773), [@danielhelfand](https://github.com/danielhelfand)) [SIG CLI]
- Apimachinery/clock: This deletes the apimachinery/clock package. Please use k8s.io/utils/clock instead. ([#109752](https://github.com/kubernetes/kubernetes/pull/109752), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG API Machinery]
- Apiserver_longrunning_gauge is removed from the codebase. Please use apiserver_longrunning_requests instead. ([#110310](https://github.com/kubernetes/kubernetes/pull/110310), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Instrumentation]
- Feature gates that graduated to GA in 1.23 or earlier and were unconditionally enabled have been removed: CSIServiceAccountToken, ConfigurableFSGroupPolicy, EndpointSlice, EndpointSliceNodeName, EndpointSliceProxying, GenericEphemeralVolume, IPv6DualStack, IngressClassNamespacedParams, StorageObjectInUseProtection, TTLAfterFinished, VolumeSubpath, WindowsEndpointSliceProxying ([#109435](https://github.com/kubernetes/kubernetes/pull/109435), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture and Cloud Provider]
- For resources built into an apiserver, the server now logs at `-v=3` whether it is using watch caching. ([#109175](https://github.com/kubernetes/kubernetes/pull/109175), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery]
- Honor the framework delete timeout for pv ([#109764](https://github.com/kubernetes/kubernetes/pull/109764), [@saikat-royc](https://github.com/saikat-royc)) [SIG Storage and Testing]
- Kube-controller-manager's deprecated `--experimental-cluster-signing-duration` flag is now removed. Adapt your machinery to use the `--cluster-signing-duration` flag that is available since v1.19. ([#108476](https://github.com/kubernetes/kubernetes/pull/108476), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Auth]
- Kubeadm: perform additional dockershim cleanup. Treat all container runtimes as remote by using the flag "--container-runtime=remote", given dockershim was removed in 1.24 and given kubeadm 1.25 supports a kubelet version of 1.24 and 1.25. The flag "--network-plugin" will no longer be used for new clusters. Stop cleaning up the following dockershim related directories on "kubeadm reset": "/var/lib/dockershim", "/var/runkubernetes", "/var/lib/cni" ([#110022](https://github.com/kubernetes/kubernetes/pull/110022), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubelet's deprecated `--experimental-kernel-memcg-notification` flag is now removed.  Use `--kernel-memcg-notification` instead. ([#109388](https://github.com/kubernetes/kubernetes/pull/109388), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Node]
- Kubernetes binaries are now built in module mode instead of GOPATH mode ([#109464](https://github.com/kubernetes/kubernetes/pull/109464), [@liggitt](https://github.com/liggitt)) [SIG Architecture, Node and Testing]
- Remove deprecated kubectl.kubernetes.io/default-logs-container support ([#109254](https://github.com/kubernetes/kubernetes/pull/109254), [@pacoxu](https://github.com/pacoxu)) [SIG CLI]
- Rename apiserver_watch_cache_watch_cache_initializations_total to apiserver_watch_cache_initializations_total ([#109579](https://github.com/kubernetes/kubernetes/pull/109579), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Instrumentation]
- TBD ([#109277](https://github.com/kubernetes/kubernetes/pull/109277), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG Architecture and Instrumentation]
- Updated cri-tools to [v1.24.2(https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.24.2) ([#109813](https://github.com/kubernetes/kubernetes/pull/109813), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Release]
- `apiserver_dropped_requests` is dropped from this release since `apiserver_request_total` can now be used to track dropped requests. `etcd_object_counts` is also removed in favor of `apiserver_storage_objects`. `apiserver_registered_watchers` is also removed in favor of `apiserver_longrunning_requests`. ([#110337](https://github.com/kubernetes/kubernetes/pull/110337), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Instrumentation]

## Dependencies

### Added
- github.com/emicklei/go-restful/v3: [v3.8.0](https://github.com/emicklei/go-restful/v3/tree/v3.8.0)
- github.com/golang-jwt/jwt/v4: [v4.2.0](https://github.com/golang-jwt/jwt/v4/tree/v4.2.0)
- github.com/golangplus/bytes: [v1.0.0](https://github.com/golangplus/bytes/tree/v1.0.0)
- github.com/golangplus/fmt: [v1.0.0](https://github.com/golangplus/fmt/tree/v1.0.0)

### Changed
- bitbucket.org/bertimus9/systemstat: 0eeff89 → v0.5.0
- github.com/Azure/go-autorest/autorest/adal: [v0.9.13 → v0.9.20](https://github.com/Azure/go-autorest/autorest/adal/compare/v0.9.13...v0.9.20)
- github.com/Azure/go-autorest/autorest/mocks: [v0.4.1 → v0.4.2](https://github.com/Azure/go-autorest/autorest/mocks/compare/v0.4.1...v0.4.2)
- github.com/Azure/go-autorest/autorest: [v0.11.18 → v0.11.27](https://github.com/Azure/go-autorest/autorest/compare/v0.11.18...v0.11.27)
- github.com/MakeNowJust/heredoc: [bb23615 → v1.0.0](https://github.com/MakeNowJust/heredoc/compare/bb23615...v1.0.0)
- github.com/antlr/antlr4/runtime/Go/antlr: [b48c857 → ad29539](https://github.com/antlr/antlr4/runtime/Go/antlr/compare/b48c857...ad29539)
- github.com/chai2010/gettext-go: [c6fed77 → v1.0.2](https://github.com/chai2010/gettext-go/compare/c6fed77...v1.0.2)
- github.com/cncf/udpa/go: [5459f2c → 04548b0](https://github.com/cncf/udpa/go/compare/5459f2c...04548b0)
- github.com/cncf/xds/go: [fbca930 → cb28da3](https://github.com/cncf/xds/go/compare/fbca930...cb28da3)
- github.com/container-storage-interface/spec: [v1.5.0 → v1.6.0](https://github.com/container-storage-interface/spec/compare/v1.5.0...v1.6.0)
- github.com/coredns/corefile-migration: [v1.0.14 → v1.0.17](https://github.com/coredns/corefile-migration/compare/v1.0.14...v1.0.17)
- github.com/daviddengcn/go-colortext: [511bcaf → v1.0.0](https://github.com/daviddengcn/go-colortext/compare/511bcaf...v1.0.0)
- github.com/envoyproxy/go-control-plane: [63b5d3c → 49ff273](https://github.com/envoyproxy/go-control-plane/compare/63b5d3c...49ff273)
- github.com/go-logr/logr: [v1.2.0 → v1.2.3](https://github.com/go-logr/logr/compare/v1.2.0...v1.2.3)
- github.com/go-logr/zapr: [v1.2.0 → v1.2.3](https://github.com/go-logr/zapr/compare/v1.2.0...v1.2.3)
- github.com/golangplus/testing: [af21d9c → v1.0.0](https://github.com/golangplus/testing/compare/af21d9c...v1.0.0)
- github.com/google/cel-go: [v0.10.1 → v0.11.2](https://github.com/google/cel-go/compare/v0.10.1...v0.11.2)
- github.com/google/go-cmp: [v0.5.5 → v0.5.6](https://github.com/google/go-cmp/compare/v0.5.5...v0.5.6)
- github.com/pquerna/cachecontrol: [0dec1b3 → v0.1.0](https://github.com/pquerna/cachecontrol/compare/0dec1b3...v0.1.0)
- go.etcd.io/etcd/api/v3: v3.5.1 → v3.5.4
- go.etcd.io/etcd/client/pkg/v3: v3.5.1 → v3.5.4
- go.etcd.io/etcd/client/v2: v2.305.0 → v2.305.4
- go.etcd.io/etcd/client/v3: v3.5.1 → v3.5.4
- go.etcd.io/etcd/pkg/v3: v3.5.0 → v3.5.4
- go.etcd.io/etcd/raft/v3: v3.5.0 → v3.5.4
- go.etcd.io/etcd/server/v3: v3.5.0 → v3.5.4
- golang.org/x/crypto: 8634188 → 3147a52
- google.golang.org/genproto: 42d7afd → 1973136
- google.golang.org/grpc: v1.40.0 → v1.47.0
- gopkg.in/yaml.v3: 496545a → v3.0.1
- k8s.io/kube-openapi: 3ee0da9 → 31174f5

### Removed
- github.com/OneOfOne/xxhash: [v1.2.2](https://github.com/OneOfOne/xxhash/tree/v1.2.2)
- github.com/cespare/xxhash: [v1.1.0](https://github.com/cespare/xxhash/tree/v1.1.0)
- github.com/emicklei/go-restful: [v2.9.5+incompatible](https://github.com/emicklei/go-restful/tree/v2.9.5)
- github.com/google/cel-spec: [v0.6.0](https://github.com/google/cel-spec/tree/v0.6.0)
- github.com/spaolacci/murmur3: [f09979e](https://github.com/spaolacci/murmur3/tree/f09979e)