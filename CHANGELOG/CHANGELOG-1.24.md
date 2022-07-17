<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.24.3](#v1243)
  - [Downloads for v1.24.3](#downloads-for-v1243)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.24.2](#changelog-since-v1242)
  - [Changes by Kind](#changes-by-kind)
    - [Bug or Regression](#bug-or-regression)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.24.2](#v1242)
  - [Downloads for v1.24.2](#downloads-for-v1242)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.24.1](#changelog-since-v1241)
  - [Changes by Kind](#changes-by-kind-1)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.24.1](#v1241)
  - [Downloads for v1.24.1](#downloads-for-v1241)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.24.0](#changelog-since-v1240)
  - [Changes by Kind](#changes-by-kind-2)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.24.0](#v1240)
  - [Downloads for v1.24.0](#downloads-for-v1240)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.23.0](#changelog-since-v1230)
  - [Major Themes](#major-themes)
    - [Dockershim Removed from kubelet](#dockershim-removed-from-kubelet)
    - [Beta APIs Off by Default](#beta-apis-off-by-default)
    - [Signing Release Artifacts](#signing-release-artifacts)
    - [OpenAPI v3](#openapi-v3)
    - [Storage Capacity and Volume Expansion Are Generally Available](#storage-capacity-and-volume-expansion-are-generally-available)
    - [NonPreemptingPriority to Stable](#nonpreemptingpriority-to-stable)
    - [Storage Plugin Migration](#storage-plugin-migration)
    - [gRPC Probes Graduate to Beta](#grpc-probes-graduate-to-beta)
    - [Kubelet Credential Provider Graduates to Beta](#kubelet-credential-provider-graduates-to-beta)
    - [Contextual Logging in Alpha](#contextual-logging-in-alpha)
    - [Avoiding Collisions in IP allocation to Services](#avoiding-collisions-in-ip-allocation-to-services)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
    - [Uncategorized](#uncategorized)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.24.0-rc.1](#v1240-rc1)
  - [Downloads for v1.24.0-rc.1](#downloads-for-v1240-rc1)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
    - [Container Images](#container-images-4)
  - [Changelog since v1.24.0-rc.0](#changelog-since-v1240-rc0)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)
- [v1.24.0-rc.0](#v1240-rc0)
  - [Downloads for v1.24.0-rc.0](#downloads-for-v1240-rc0)
    - [Source Code](#source-code-5)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
    - [Container Images](#container-images-5)
  - [Changelog since v1.24.0-beta.0](#changelog-since-v1240-beta0)
  - [Changes by Kind](#changes-by-kind-4)
    - [API Change](#api-change-1)
    - [Feature](#feature-3)
    - [Bug or Regression](#bug-or-regression-4)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-5)
    - [Added](#added-5)
    - [Changed](#changed-5)
    - [Removed](#removed-5)
- [v1.24.0-beta.0](#v1240-beta0)
  - [Downloads for v1.24.0-beta.0](#downloads-for-v1240-beta0)
    - [Source Code](#source-code-6)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
    - [Container Images](#container-images-6)
  - [Changelog since v1.24.0-alpha.4](#changelog-since-v1240-alpha4)
  - [Changes by Kind](#changes-by-kind-5)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-2)
    - [Feature](#feature-4)
    - [Bug or Regression](#bug-or-regression-5)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
    - [Uncategorized](#uncategorized-1)
  - [Dependencies](#dependencies-6)
    - [Added](#added-6)
    - [Changed](#changed-6)
    - [Removed](#removed-6)
- [v1.24.0-alpha.4](#v1240-alpha4)
  - [Downloads for v1.24.0-alpha.4](#downloads-for-v1240-alpha4)
    - [Source Code](#source-code-7)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
    - [Container Images](#container-images-7)
  - [Changelog since v1.24.0-alpha.3](#changelog-since-v1240-alpha3)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-6)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-3)
    - [Feature](#feature-5)
    - [Bug or Regression](#bug-or-regression-6)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-7)
    - [Added](#added-7)
    - [Changed](#changed-7)
    - [Removed](#removed-7)
- [v1.24.0-alpha.3](#v1240-alpha3)
  - [Downloads for v1.24.0-alpha.3](#downloads-for-v1240-alpha3)
    - [Source Code](#source-code-8)
    - [Client Binaries](#client-binaries-8)
    - [Server Binaries](#server-binaries-8)
    - [Node Binaries](#node-binaries-8)
    - [Container Images](#container-images-8)
  - [Changelog since v1.24.0-alpha.2](#changelog-since-v1240-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-7)
    - [Deprecation](#deprecation-3)
    - [API Change](#api-change-4)
    - [Feature](#feature-6)
    - [Bug or Regression](#bug-or-regression-7)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-8)
    - [Added](#added-8)
    - [Changed](#changed-8)
    - [Removed](#removed-8)
- [v1.24.0-alpha.2](#v1240-alpha2)
  - [Downloads for v1.24.0-alpha.2](#downloads-for-v1240-alpha2)
    - [Source Code](#source-code-9)
    - [Client Binaries](#client-binaries-9)
    - [Server Binaries](#server-binaries-9)
    - [Node Binaries](#node-binaries-9)
    - [Container Images](#container-images-9)
  - [Changelog since v1.24.0-alpha.1](#changelog-since-v1240-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-3)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-3)
  - [Changes by Kind](#changes-by-kind-8)
    - [Deprecation](#deprecation-4)
    - [API Change](#api-change-5)
    - [Feature](#feature-7)
    - [Bug or Regression](#bug-or-regression-8)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-5)
  - [Dependencies](#dependencies-9)
    - [Added](#added-9)
    - [Changed](#changed-9)
    - [Removed](#removed-9)
- [v1.24.0-alpha.1](#v1240-alpha1)
  - [Downloads for v1.24.0-alpha.1](#downloads-for-v1240-alpha1)
    - [Source Code](#source-code-10)
    - [Client Binaries](#client-binaries-10)
    - [Server Binaries](#server-binaries-10)
    - [Node Binaries](#node-binaries-10)
  - [Changelog since v1.23.0](#changelog-since-v1230-1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-4)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-4)
  - [Changes by Kind](#changes-by-kind-9)
    - [Feature](#feature-8)
    - [Bug or Regression](#bug-or-regression-9)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-6)
  - [Dependencies](#dependencies-10)
    - [Added](#added-10)
    - [Changed](#changed-10)
    - [Removed](#removed-10)

<!-- END MUNGE: GENERATED_TOC -->

# v1.24.3


## Downloads for v1.24.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes.tar.gz) | 65ce2f76a6954ebb341a85ffe5acecb13e5ea43f578f36b191395ecfcfa2b1b2383866b9ff19a15f44cfbd7bbd58991667b3e072066edcefb36f072fa05f54c0
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-src.tar.gz) | 300e8ae55bd7c491557a31d9b55e62a38310f78f30a5512a58b017c422bb8af295d703df95b33a7672ccaeec714f5e92e1d7a36a31d6d733250d388926bea1a1

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-darwin-amd64.tar.gz) | 64c0863ef5de961110d07dd42d16284ab641c0d895b9705014f1db568386f698820f12c379cd8114b79de3922441589e8f4e09427fdec724459eacd3d2019b81
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-darwin-arm64.tar.gz) | f8c095529e3027c7e0108debd9471ce604a435b33bc4a24fb0a29db12c3268b348c3d7f086e759da20d378f41a463695344fb6093a122ae225e3a5b1b2a0b6f8
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-linux-386.tar.gz) | 13a464bdb73d93de135622221eb8b36d1e337b816940e9e7c4fe46b7a2960b3185ac69edd48f2fd9774b4ec49f52ae44909c8f0dbd7acbfa3465ee2f6f353dc9
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-linux-amd64.tar.gz) | 7411fef7bd0398b0986e874845aba93108e0e579965601970f74e1f5132402a983eb65c732c25e822d1c2eb92e7f14206802ec052a7d2b8e70a7cbe8888c1caa
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-linux-arm.tar.gz) | 25ec4d84e86f967008572c31205472014dc414298ee5f265f37e5584c21d82cd6683a5673ee8ed9a0dbfde151cf40dd71d3f59bf7adee280c260a888b9158342
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-linux-arm64.tar.gz) | 6c04e56bad70826133ec04fa9fcf14b020998860a5562eb47dbf748f356158fa4712f118f63a334a5ebdd49ba09eaed0ee995da7e7e8a87d5ccf5f107b2edec6
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-linux-ppc64le.tar.gz) | 2e4939e7d85a85b9c67b6ad40aa17267f94f8c8ecbbc9bbc03c238d8b9da908763cb35787418841be3c4b0e3d2f5b690a6f6757e5f9d36d381f458f07dce43c6
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-linux-s390x.tar.gz) | 23ff90938005e4d5c34812e18c83b177420d638a78a7a615053fef4ebe677cfab0bc3d5e99aefea58e653db8c96713d47b86fe64d8e39324a97cf910653e2e7e
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-windows-386.tar.gz) | fb2ff05cc0099bb338f4aec0eebaf8f8236ee948c438b0e6018850e0a90f71fe753d7d9f284467712f03ee40b31fd3ee9f31fdc642c6db63bd1b861df37834cf
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-windows-amd64.tar.gz) | 6a7f1c089938aa66c0e4a251288f94b39340e43f87259cd3109184dfd85ef8179f79d393eaffae657910497b1b0cc8bed6a319d6c8a53db7bf9fd33ead0932b9
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-client-windows-arm64.tar.gz) | 8f3bb3e8a97d9377fd9d06a59f72be9ab5f4db297dc64fdda800407c1ee195b0fedccc526c021527d987243c18b8adbd9b3088d6ed5927a939ea4ffc9e0e0d55

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-server-linux-amd64.tar.gz) | 2f1bfef848b9b588c219bebfb61c59afd7dc7d292c733a1106e5dca51d250b9db95a92d5f9d642275c8af653700d5e7ef45412bd0aa14dd2d6a72069611f6ac4
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-server-linux-arm.tar.gz) | 598922f30f6daa6e07b10eb973f515d45a64c8e7b8f27956cd46ebf7c351deac73deab599fd9bc4fbdaea8c85d547046c7b96a3a05671c5589bc2e561c01b25e
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-server-linux-arm64.tar.gz) | 44e1dd3c936a6de42c8a6ea6fb76aeb7876b6f7a76fc88e2ff4dc0c693e0916610c5c7949a842803130d050760c11edb8a0aae2d08ac6d29b6b2b1db3d42414a
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-server-linux-ppc64le.tar.gz) | 341b36a4fa403259fb834c16dd66cbd958aad0a8aaf8c5dcea76625b67edb17ec59cacd25ca1537cf4cd64aaab8cbf7829673369436f0c9f6f3f1062be652ac1
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-server-linux-s390x.tar.gz) | e0feb8422d0a41fe4682e6a708ff1f422d6c003d15d52163282714b2030352801c15f6fff98b9c76d3b255268c8ef82c31d488882ca72c1e92786fa0f7830b3e

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-node-linux-amd64.tar.gz) | 1c80891c02d2ad691e701e1427f72e7c7e7fa3a8d84537cc90b683bea750cebee64535d56a4d5356aa4415f66e5ec015c0ccf98e71ff16fac4f74de32ebafea6
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-node-linux-arm.tar.gz) | 589ce7abd6104a73058c22e7180cc25dc1fe8649ecf1581c41e0899d59ca1cf5dac0f1871214743a0ce5dce049665e879fbf8e817472eb1154b9c3b19476dcbe
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-node-linux-arm64.tar.gz) | 912c1fe512bffb31758caf0f78c1ce676b949e78400c8aac7feae0ef253c29f12e2517a7ff7f6fcb250cb6d9d4898ccbd23aebf63d7ac43d492ff426f11250ee
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-node-linux-ppc64le.tar.gz) | 9628ff5acdb0d1fd9a2dff2831b158e78252e802b7db148dbf71d2a85a34647b2ac5194bf0100f3733f87482d3bb005278c39eb80d1208717369b82969c0da64
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-node-linux-s390x.tar.gz) | dfd3606d587f635b363f59b48c86074fbe5315c77a91e0ce1b4b15d7700c2e46ba89a3e7a7bab6f43fade596b6f70647a70e4045bafa50890640415ca88e1e9a
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.3/kubernetes-node-windows-amd64.tar.gz) | 9fb40c932719b25df7c25f9a683a3a0855a05feb157df8c2b9498b7f3d9a89a4be17c4b4291b37643475496ef4e03c6ba2a3466695d7f3b3abe1138b4ce0dc75

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.2

## Changes by Kind

### Bug or Regression

- Fix a bug on endpointslices tests comparing the wrong metrics ([#110920](https://github.com/kubernetes/kubernetes/pull/110920), [@jluhrsen](https://github.com/jluhrsen)) [SIG Apps and Network]
- Fix a bug that caused the wrong result length when using --chunk-size and --selector together ([#110735](https://github.com/kubernetes/kubernetes/pull/110735), [@Abirdcfly](https://github.com/Abirdcfly)) [SIG API Machinery and Testing]
- Fix bug that prevented the job controller from enforcing activeDeadlineSeconds when set ([#110544](https://github.com/kubernetes/kubernetes/pull/110544), [@harshanarayana](https://github.com/harshanarayana)) [SIG Apps]
- Fix image pulling failure when IMDS is unavailable in kubelet startup ([#110523](https://github.com/kubernetes/kubernetes/pull/110523), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix printing resources with int64 fields ([#110572](https://github.com/kubernetes/kubernetes/pull/110572), [@sanchezl](https://github.com/sanchezl)) [SIG API Machinery]
- Fix unnecessary recreation of placeholder EndpointSlice ([#110732](https://github.com/kubernetes/kubernetes/pull/110732), [@jluhrsen](https://github.com/jluhrsen)) [SIG Apps and Network]
- Fixed a regression introduced in 1.24.0 where Azure load balancers were not kept up to date with the state of cluster nodes. In particular, nodes that are not in the ready state and are not newly created (i.e. not having the `node.cloudprovider.kubernetes.io/uninitialized` taint) now get removed from Azure load balancers. ([#109931](https://github.com/kubernetes/kubernetes/pull/109931), [@ricky-rav](https://github.com/ricky-rav)) [SIG Cloud Provider]
- Kubeadm: fix error adding extra prefix unix:// to CRI endpoints that were missing URL scheme ([#110634](https://github.com/kubernetes/kubernetes/pull/110634), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that configurable KubernetesVersion not respected during kubeadm join ([#111021](https://github.com/kubernetes/kubernetes/pull/111021), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.24.2


## Downloads for v1.24.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes.tar.gz) | fade8dd6c1f80c0e38b0d9eacf27eb98d5b4397b87b6a0271e785675277abc98c99b3cfa55abf8d5c9de19edf47174e5cc83f1d28c214ab27f4fd0b540f25bb5
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-src.tar.gz) | 9655cc036272fe5305e008ff0e86dfc66e593c74ed4631f798fa2a24ad960c4a22bfce9384385c748971e21d67e113e12a271744f4e3a9d6c5fc11ea775619c6

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-darwin-amd64.tar.gz) | 0d1112448ca967c6d1c0fc8e8c45f4879d2025ffcd641f057b47cd04d0e4cd725c4431f7985cfb526bac2fd9c6747348125ebe82dce1541db5561844bb23185e
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-darwin-arm64.tar.gz) | 274809ebbfe99a2e32ff51c6897ee9907bd78a54437c0c0fe28a3ce36122bd965d202c59f4a8b2e229495adec1682d2e66f31376444579136c215a0408139725
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-linux-386.tar.gz) | c78c29d3cc012043d464389be317693954b0e5b0489b64a2a665b3c28163c89c183d3a69f194d3028409d78d1e3d24a46fe7bda390f560abb79fb35f236a920b
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-linux-amd64.tar.gz) | 3a70b1ec679f2ea17887f74579d5e566774239d40dfac69b1ea10036a39674fdf36ec3351c906d1a28f1c375b86fcd0a84d0a359cce81c1acd10743fb3553221
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-linux-arm.tar.gz) | 0ca579d5a80bb329d83556029fd71051b3d07a015d3e4a1359f58d8e95c7823ae32e0889d27325a343c5a05e588b9bd2310e8309c268c336689bff6997b89ac4
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-linux-arm64.tar.gz) | 194780370da50d4bbc88bf919c9f9655a52226f6d95bf0c406e690dfc997c15543d426241d533449ec4fa585c2f6ff3d1404a4ea9932fb7a589a4fba27f71b8e
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-linux-ppc64le.tar.gz) | 17ad8fc4270a0a0c7d31de3663dd3dd39274e54acfa835ffbda7c76a94b20ff28a2f5ed2c2dd4828f27974994d30410a1167e65abe49793a42cbedb7b778bd68
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-linux-s390x.tar.gz) | 09dd184eb37f2ee98c55694b7ea41fea04d8f89c6f0be0e2288129d51f6ebafa34d020358ab09b2cd88d1e605d3105a70a4311b65c276e435bb6a7767d40fbaf
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-windows-386.tar.gz) | 4fda4137d7714776600d75e65d400a91fff7a69bca01f55a56592de1bb6f13f7c3ff555a833928cf677dea1c16919d235e8c409a3c719fe16e696fb16631a0ae
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-windows-amd64.tar.gz) | 48d768bbc5811c2e7e876aa90fdbfc17450fbe5b18012a8f510a5bf671f55215cb5bdce56e5e99443c3b0d9d8d1c75d903897d378a3ebfb91cd9d78fb6d2af72
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-client-windows-arm64.tar.gz) | 364b791de6c9e12aabb7abe2f0aed94408478b289cf0afd6715794989bc0eb9d2449ae35f5b801d40cb2661d53abaa397d0508758a063b7b9c646067c085440d

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-server-linux-amd64.tar.gz) | 76ad6c96ee8904c428ec3067e5e5935c8309d92ccc900bb4559fc65ce674b8f460479389cd4d8f8f9b1e999477b961a63bc8b1ff495d7f71a33959d670ade2a6
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-server-linux-arm.tar.gz) | a706082338e4fb7b6480d2b85a5d441f3d42ab24d63203b1a6cbf26d5d646face3eadfff9c42d1a217be99109570fd43bffae5e7a25660696e58682128aa5105
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-server-linux-arm64.tar.gz) | fcbffb31858ed899b569df2a60664370a7161bbb25894e62b8a1625e73d607a534b2c75d6fca53d346586f4653cc0542837868b62f17a4aa1dc191ad871315f8
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-server-linux-ppc64le.tar.gz) | 8241dd044914f72dcdd9472e6c2f41ab8d700136e4b4aadff035a801b9e85b250ceab2750627181f8bdde52d1dd0aa1972e16a818f84cdd889888ec22185eb4c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-server-linux-s390x.tar.gz) | 1a55064a5a172ca8889898679df4d7e707328a0d2220ab2b6b8905532a27a0207d442f9c43674d73b60a32a99088da18ab48ccdb020f26b12983eaed8310ab47

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-node-linux-amd64.tar.gz) | 054961bdf6a2e3023c49f2148b520ae39ab5dda134e2666c5b01bbb78ac24c8803233a092db0ababd5c5c13d3e00b50822045051f56019fe8181e53b42e4e7c6
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-node-linux-arm.tar.gz) | 5251ff61fa1e43b9762dac59c410c61379fc3cb5f99253ed8181ac99e5aeb63499d58b3b4ef3544cddc72fc1b1500716e104ede7ddd8c7d4a1b008faad1026df
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-node-linux-arm64.tar.gz) | 05ea2768e4438ff57395016072267a4b920c0e55d963f7d5dd35bafe64a2f76944a4c56378a05a47e6989e1bfad3bc56ec5641af09545bc13cc73b3c42a56b41
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-node-linux-ppc64le.tar.gz) | 98a62bf68e4a3121b5d61d0411adb792d8a4c37ec9b0ede050bea7e7a020d0a0fc2339fd14f184b8486ddf95baee53987f7a5cf3334b17df5378dbfac419d440
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-node-linux-s390x.tar.gz) | 39a1a8321ee3d211e693afe451af09db3f03d01a5bc24d17548dd9c2b0c0e859500851dbeab87a1152e02963b18392a7f3de1d5783a81ccd7f201bb3e87abf69
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.2/kubernetes-node-windows-amd64.tar.gz) | 02612c705b45864682b7bcc9795070e6aee7ceea20da520ada0187ef73c1a73d68cde626ef1d623570e3415d373c3b408abae3c7a0cfaa0b2a7cdd2c421eb4b1

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.1

## Changes by Kind

### Feature

- Kubernetes is now built with Golang 1.18.3 ([#110422](https://github.com/kubernetes/kubernetes/pull/110422), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]

### Bug or Regression

- EndpointSlices marked for deletion are now ignored during reconciliation. ([#110484](https://github.com/kubernetes/kubernetes/pull/110484), [@aryan9600](https://github.com/aryan9600)) [SIG Apps and Network]
- Fixed a kubelet issue that could result in invalid pod status updates to be sent to the api-server where pods would be reported in a terminal phase but also report a ready condition of true in some cases. ([#110479](https://github.com/kubernetes/kubernetes/pull/110479), [@bobbypage](https://github.com/bobbypage)) [SIG Node and Testing]
- Pods will now post their readiness during termination. ([#110416](https://github.com/kubernetes/kubernetes/pull/110416), [@aojea](https://github.com/aojea)) [SIG Network, Node and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.24.1


## Downloads for v1.24.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes.tar.gz) | 51a10087465c18b067657b452b6d1302b7203cf66b1011d6603281d5f59bdcdf6f3eb43c9220a5f43492560360bc1c297c0618873409e5b221bcba10242f0c21
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-src.tar.gz) | ba82208abe240a71f9111a02782d1f2675709b36427a9669aa8aa25fb7bce3ae04a84922a2d0f0fa915259c5c77736e6997a502280165a405394226e5df66aea

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-darwin-amd64.tar.gz) | debebe3f34580fb96ec5ccdb54f3786b315d4eb2d743e8a77559e4c5eff2be3d19ae18287a76acd6465434bf831e419996ab75baeca18636611516fa1d504270
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-darwin-arm64.tar.gz) | ef574b92d920066e8281f1bf3c66904bb3e5974ee843ef66b97a42cee8021158b5c0110541960f89bb6af0b437d8b5642baa87111cc7528cb3eec0abd8c3dca0
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-linux-386.tar.gz) | acae789d48f69d6e879574bea8ca30a5d403abad39d9ba82c4b9285a14b5600bba1f988f86d5c6435ef1c1848cd6c64daef0408694f75a8bd5d275844ca74161
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-linux-amd64.tar.gz) | d57a83c01257932c4cb197da03ec232dd968a3f9ba7e0df33929340030a59d4d69b68c45ed6b0f73ec13dd91e594d07f961937d08a8223f9be5701d7abc148d5
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-linux-arm.tar.gz) | e43daa9e2e538d542aed1f0fba9e483bf09414eee1f0b0f22b2b03f91e96a0942c8ad90072ea80a1f669fd5d8cbde7749eea98919c05bb29a5f44f5397fa7367
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-linux-arm64.tar.gz) | 98414b9874097f54d03a118ecb52e2e5f5897ce77790245238d68a9fd5838e47a8051976cba1547db6f3548daa5686af1ea7eeacab86c6a888157273c1a5a0b5
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-linux-ppc64le.tar.gz) | 45f1d14ada9c1f07fcb871f55cb8ca08e3cee2583cc902636cd3741f4317187184bd832e78e246ce2dc8df86b9bec85431207e82cbb9f688930004b2ede8406e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-linux-s390x.tar.gz) | 7ff43d31c12b035b494f296b1edcb6c31b8ba9a21744072d420a9f61ccb10deb7beed63f993fd1fda8970f726597f101e60255d6fa63070d6a154c69a178d16b
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-windows-386.tar.gz) | 0ab4e2fd57ef61bfac39a012ef2174323a3c5d2b6d59b6ecbb0de7b1094dcedbd70e74dde879928d2625e419060a2eb2c4251d7ffd3773e197456ebbd87b48b3
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-windows-amd64.tar.gz) | 1e50249d6c23c029867a33d0ab31830f17567eb50dd6f3412b8c172c5f137e2763c780b40d1c05e39447a5c5c5b953faecea352e6611c310973a8a36791c714d
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-client-windows-arm64.tar.gz) | fe1fde250b5085392ba2517c270318dda26862429fdac85a67b274b4746f64120d0c28606b45de0ccbd4a25c43521010835138ba4d262429639df7cf5899d842

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-server-linux-amd64.tar.gz) | 823e661ea9aa68789da6aa8bf5687411ddcc9cb1ccf4028805e092cea8b7c572995accfdc148d0dfe475576eda0b95a852182256a86efeaee0f1a8f0cd0efc42
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-server-linux-arm.tar.gz) | 9b30af3dd07e32fe50804196ef639d30db72542441d6871daddeb79b401099fcb52237a456e001b22b78eefc274adc199138012597de1c040557073967b1643a
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-server-linux-arm64.tar.gz) | ea2724f331afab4084e0e8dcaa74101a7ba4e245ebb2792823a88ae604fab5feb7a5e2ee93340822a829d3541757f9370be2657fca757737b2cea1a3e898bbc8
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-server-linux-ppc64le.tar.gz) | 95a1bc6825f826fafad74b279cab45591c2e05996dc6707749757d432e4d75ff5d4eb0ce0cd3e3ed534fce63d8c0bd5cb686670d27ddc32f0f4aa7f7b92db58a
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-server-linux-s390x.tar.gz) | 352e117b67419d3b3762467c1a1d9670c8b2a5bfd86683856171606c4f28d17cff6eb21532183765d9e96343a9c1220c29e624ca1da903e9b646689662ab6f3d

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-node-linux-amd64.tar.gz) | 889f98608ed08de02f970e14cba562f94ed4762b8968ea8df272413ee8a9b8c8cf589920a420af019cd0844218794279dd6d8ea73ae2d1e7d886b0c6d609d55f
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-node-linux-arm.tar.gz) | e21223c31ccd29d25653fff3754c02a98e924e495057b73db645ca7fd3c9aa1110f99f8da211f81066175a939600284dbe16b0dd633a376402d632211105e9eb
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-node-linux-arm64.tar.gz) | 84e563ce6573e585c5556d9fb7d71fac6a1b6d35b4e34d5f60ee7ba840ebce12e9b0dca3a693b2e4bcde683adb5b9a2c3c0a9d74ef4b0404fdb8c7380b243e90
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-node-linux-ppc64le.tar.gz) | 4a50e534c4247250da882f1343a5e313e9d7e77cf238d7cb541b716f48fbf2135d883200ce5b25176cbe9b975e3196984b914c8619878e84616714741bc25bb6
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-node-linux-s390x.tar.gz) | 75961a347e19b0adcf930ebbe4353a0b553cfe6b4d1823f6da8527493e3f3ead0471620bd094a6429e6fe93b8d8b4d178ddc5557f017b48894a170a7d53f9724
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.1/kubernetes-node-windows-amd64.tar.gz) | 4ce432964a4c63d7770c59f398e4827c45907495be899220c730ecb4bdc839a3bf133965b9b6c84a41d06dd0311ee921d45a73ab1eb6880151b9ecd27a880dab

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0

## Changes by Kind

### Feature

- Kubernetes is now built with Golang 1.18.2 ([#110044](https://github.com/kubernetes/kubernetes/pull/110044), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]

### Bug or Regression

- Fix JobTrackingWithFinalizers that:
  - was declaring a job finished before counting all the created pods in the status
  - was leaving pods with finalizers, blocking pod and job deletions
  
  JobTrackingWithFinalizers is still disabled by default. ([#109486](https://github.com/kubernetes/kubernetes/pull/109486), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Testing]
- Kubeadm: only taint control plane nodes when the legacy "master" taint is present. This avoids a bug where "kubeadm upgrade" will re-taint a control plane node with the new "control plane" taint even if the user explicitly untainted the node. ([#109841](https://github.com/kubernetes/kubernetes/pull/109841), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.24.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.24.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes.tar.gz) | `ef3014768ea305a97865ceef486e704083af78841079cad5c589c02711ac12769f5e949cefa2188c80d28a30884a559befc2239e2adfecdca972d1d211a0cb4f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-src.tar.gz) | `973ce8ba840125da3d1f45205fd53e82bf5c6736517ff1f27d3ecfb951d413e8db30878194177a76bd73912829a9db3e5948437234f23b269b5e05fd0b29ab75`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-darwin-amd64.tar.gz) | `acb682468b459bfc51fbc823aa5cef7ed6dbd7441bb3f00e1842c7eb59d15a9c0d0c941700f518e8826d0b3e68d54c9607c94186d624d04402d97ba4b716d384`
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-darwin-arm64.tar.gz) | `066e93a872ea0cef77ab734d7a37ec0ada797da31cd004b29326fc75d4849067ce8a58f11807a4b45643794db4cffa1091820af8c3c476ff9d49c7ce517129b1`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-linux-386.tar.gz) | `61aba7c6fef077ddf94e6f63958fd4f1c6062460f47098979e8bedf3b1081478f5905cb72e5a6edcb28f46f8e78965dbf4dbc666be49d2a7ad025b39e74e7369`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-linux-amd64.tar.gz) | `291a4d24a400666ec0e3d91fbaae0605de438abf4878ca81e4e8e923beefa8a434ade3a98e8538f654dfe7d32c2038a1610ca6863efe91d2649ea11c28a20a49`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-linux-arm.tar.gz) | `ac37de6c47ec1e8d8be6e60ef8ca03cd66b6f212b671d619bfa38fe62182c830550d5d6ebe80f4363d2007160ad177c27e7eaab29c2dd5642c4c91f30ce0a347`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-linux-arm64.tar.gz) | `643b95ff27f275ee7f8999676c141d9a6199d1a60f19ab425be57e6170abf66a5dd4c3712e10db6e11b104e1e41f1fc2b2366507c477b080061cc0ae1fcd788e`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-linux-ppc64le.tar.gz) | `2ce8193c228bb8703d2b42c0e354c489a8c2cf73acfa317c8720f7b1deda68f71ff19c27260ac300d5ab1a70958d082fd4146d09be09505694e65c00b23f103b`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-linux-s390x.tar.gz) | `c4124123942178f3371ace92b21e284afd0982d6e2cf8f43db0dc9266a35f5d771d84d829ffa64780213ac7375d0f100463fc990529b28c269527061958098c4`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-windows-386.tar.gz) | `9973e72958a27b11a02f74147355f8cc4525d283fc148f7a39c2863bb66601839dbfdb27e0da6477dd0afba7afd6a01138ada0d708dda89518d942eb58d1f44c`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-windows-amd64.tar.gz) | `6075bd48040a710395ea6df57f3379984291d90f7d2f75f08d3f24e46abc5b716445dc1d7c26cd877439f213f4efea8e2c03fc362a67db86c2c44326bcc6ff43`
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-client-windows-arm64.tar.gz) | `17a7d93c0245cdd959845f9c7be95f1f172cae242af09bff03cae161abe828ff2b49cb014b847fff97bdfffd899e78f1a6a0d6c75fd7540c7de5fc9ab7321cef`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-server-linux-amd64.tar.gz) | `43a3e68bed60252b588493d07ed85eaa35ff3fec7f9440096fe9af284925f040467d1b31a8948e3035e4738bb689ad6d6fb9208fe77c16b053874d020a3fabd3`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-server-linux-arm.tar.gz) | `a8ed49f4a6c57b6e0d4a3dc8705fb5d59c8b77e1cc67564bc3825782922bcc2cf431ed762b97f1fd05b4e63d1bf71a3d43f698aae49db4e670b6a7e99384db0c`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-server-linux-arm64.tar.gz) | `dfcf3c4e751b9c174dbc667a87b0f561cb9a0ff4c0503439ca57d4e904db775f19be39605a8f553f9fee6af4e6256fea3eedc71a9cd401aa25d836d722b0f695`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-server-linux-ppc64le.tar.gz) | `e5d91705c5969a2483314cdb3f80e6b828987036f5fbcf269cce83cbf62b8d73210ec3b469c6e0667432f2f874309622768ecc3df851c9711c4fa51dcaf489e4`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-server-linux-s390x.tar.gz) | `1fcbf0e575752cff6a11c8518658454237953227588c7c73efca1a036c73d30246ce57f0b89095b02ccd536b37be2427ddde918763cf63e6a1e6248f38f41689`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-node-linux-amd64.tar.gz) | `b6b36973c45986e61d3ab4440b551b145d6279801e388b08f83c7f0369984bd55979504209d3cc70409ff70b200923c680f00302d3410c973c92f657157a7510`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-node-linux-arm.tar.gz) | `dd57bb241b0468ac78d5eb64770e99a9ad14b2cba345cd762e9552e4e58ea50a92eb037ac6e250983fc5bede37459faee7b9c97eccf7eb3dcc2587eb6f280bae`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-node-linux-arm64.tar.gz) | `b8f339d796644e1bac5e33bb47d14460c012411f28e531a61f3d7b4db939a4722bc8ecafb8193b290b9b4e896b4e52d8332a8e4b010f327b896616614b8695fb`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-node-linux-ppc64le.tar.gz) | `df52fed23afa5bebaaea2fc18482fc718e6e6643d931416cb62f1efbc8bf2e9ba82f6d3372f82b18c8d0562378dfa82aae621e13cb7a1a80a769d843f9279e1e`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-node-linux-s390x.tar.gz) | `dc79e44a9a2879d1dea8c22b2b21965c7a0279e54d8c54af899eecb4f14da0b56eb52e11812e663ed9edc94410b2771f38118bf2936aa15bf66011ee04fad03a`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0/kubernetes-node-windows-amd64.tar.gz) | `4275d4c6ab19433398adf8122923b0dd0a8e1432f966a38f40c9547255e69c952b951e76273b8f3220c777504905480a457a7f0b1a1ed64769f9db0cc9f1b716`

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.
name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.23.0

## Major Themes

### Dockershim Removed from kubelet

After its deprecation in v1.20, the dockershim component has been removed from the kubelet.
From v1.24 onwards, you will need to either use one of the other [supported runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/) (such as containerd or CRI-O)
or use cri-dockerd if you are relying on Docker Engine as your container runtime.
For more information about ensuring your cluster is ready for this removal, please
see [this guide](https://kubernetes.io/blog/2022/03/31/ready-for-dockershim-removal/).

### Beta APIs Off by Default

[New beta APIs will not be enabled in clusters by default](https://github.com/kubernetes/enhancements/issues/3136).
Existing beta APIs and new versions of existing beta APIs, will continue to be enabled by default.

### Signing Release Artifacts

Release artifacts are [signed](https://github.com/kubernetes/enhancements/issues/3031) using [cosign](https://github.com/sigstore/cosign)
signatures
and there is experimental support for [verifying image signatures](https://kubernetes.io/docs/tasks/administer-cluster/verify-signed-images/).
Signing and verification of release artifacts is part of [increasing software supply chain security for the Kubernetes release process](https://github.com/kubernetes/enhancements/issues/3027).

### OpenAPI v3

Kubernetes 1.24 offers beta support for publishing its APIs in the [OpenAPI v3 format](https://github.com/kubernetes/enhancements/issues/2896).

### Storage Capacity and Volume Expansion Are Generally Available

[Storage capacity tracking](https://github.com/kubernetes/enhancements/issues/1472)
supports exposing currently available storage capacity via [CSIStorageCapacity objects](https://kubernetes.io/docs/concepts/storage/storage-capacity/#api)
and enhances scheduling of pods that use CSI volumes with late binding.

[Volume expansion](https://github.com/kubernetes/enhancements/issues/284) adds support
for resizing existing persistent volumes.

### NonPreemptingPriority to Stable

This feature adds [a new option to PriorityClasses](https://github.com/kubernetes/enhancements/issues/902),
which can enable or disable pod preemption.

### Storage Plugin Migration

There is work under way to [migrate the internals of in-tree storage plugins](https://github.com/kubernetes/enhancements/issues/625) to call out to CSI Plugins,
while maintaining the original API.
The [Azure Disk](https://github.com/kubernetes/enhancements/issues/1490)
and [OpenStack Cinder](https://github.com/kubernetes/enhancements/issues/1489) plugins
have both been migrated.

### gRPC Probes Graduate to Beta

With Kubernetes 1.24, the [gRPC probes functionality](https://github.com/kubernetes/enhancements/issues/2727)
has entered beta and is available by default. You can now [configure startup, liveness, and readiness probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#configure-probes) for your gRPC app
natively within Kubernetes, without exposing an HTTP endpoint or
using an extra executable.

### Kubelet Credential Provider Graduates to Beta

Originally released as Alpha in Kubernetes 1.20, the kubelet's support for
[image credential providers](https://kubernetes.io/docs/tasks/kubelet-credential-provider/kubelet-credential-provider/)
has now graduated to Beta.
This allows the kubelet to dynamically retrieve credentials for a container image registry
using exec plugins, rather than storing credentials on the node's filesystem.

### Contextual Logging in Alpha

Kubernetes 1.24 has introduced [contextual logging](https://github.com/kubernetes/enhancements/issues/3077)
that enables the caller of a function to control all aspects of logging (output formatting, verbosity, additional values and names).

### Avoiding Collisions in IP allocation to Services

Kubernetes 1.24 introduced a new opt-in feature that allows you to
[soft-reserve a range for static IP address assignments](https://kubernetes.io/docs/concepts/services-networking/service/#service-ip-static-sub-range)
to Services.
With the manual enablement of this feature, the cluster will prefer automatic assignment from
the pool of Service IP addresses thereby reducing the risk of collision.

A Service `ClusterIP` can be assigned:

* dynamically, which means the cluster will automatically pick a free IP within the configured Service IP range.
* statically, which means the user will set one IP within the configured Service IP range.

Service `ClusterIP` are unique, hence, trying to create a Service with a `ClusterIP` that has already been allocated will return an error.

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

- Docker runtime support using dockershim in the kubelet is now completely removed in 1.24. The kubelet used to have a module called dockershim, which implements CRI support for Docker, and it has seen maintenance issues in the Kubernetes community. From 1.24 onwards, please move to a container runtime that is a full-fledged implementation of CRI (v1alpha1 or v1 compliant) as they become available. ([#97252](https://github.com/kubernetes/kubernetes/pull/97252), [@dims](https://github.com/dims))
- Fixed bug with leads to Node goes `Not-ready` state when credentials for vCenter stored in a secret and Zones feature is in use. Zone labels setup moved to KCM component, kubelet skips this step during startup in such case. If credentials stored in cloud-provider config file as plaintext current behaviour does not change and no action required. For proper functioning `kube-system:vsphere-legacy-cloud-provider` should be allowed to update node object if vCenter credentials stored in secret and Zone feature used. ([#101028](https://github.com/kubernetes/kubernetes/pull/101028), [@lobziik](https://github.com/lobziik))
- The `LegacyServiceAccountTokenNoAutoGeneration` feature gate is beta, and enabled by default. When enabled, Secret API objects containing service account tokens are no longer auto-generated for every ServiceAccount. Use the [TokenRequest](https://kubernetes.io/docs/reference/kubernetes-api/authentication-resources/token-request-v1/) API to acquire service account tokens, or if a non-expiring token is required, create a Secret API object for the token controller to populate with a service account token by following this [guide](https://kubernetes.io/docs/concepts/configuration/secret/#service-account-token-secrets). ([#108309](https://github.com/kubernetes/kubernetes/pull/108309), [@zshihang](https://github.com/zshihang))
- The calculations for Pod topology spread skew now exclude nodes that
  don't match the node affinity/selector. This may lead to unschedulable pods if you previously had pods
  matching the spreading selector on those excluded nodes (not matching the node affinity/selector),
  especially when the `topologyKey` is not node-level. Revisit the node affinity and/or pod selector in the
  topology spread constraints to avoid this scenario. ([#107009](https://github.com/kubernetes/kubernetes/pull/107009), [@kerthcet](https://github.com/kerthcet))
- Remove the deprecated flag `--experimental-check-node-capabilities-before-mount`. With CSI now GA, there is a better alternative. Remove any use of  `--experimental-check-node-capabilities-before-mount` from your kubelet scripts or manifests. ([#104732](https://github.com/kubernetes/kubernetes/pull/104732), [@mengjiao-liu](https://github.com/mengjiao-liu))
- `kubeadm.k8s.io/v1beta2` has been deprecated and will be removed in a future release, possibly in 3 releases (one year). You should start using `kubeadm.k8s.io/v1beta3` for new clusters. To migrate your old configuration files on disk you can use the `kubeadm config migrate` command. ([#107013](https://github.com/kubernetes/kubernetes/pull/107013), [@pacoxu](https://github.com/pacoxu))
- Kubeadm: default the kubeadm configuration to the containerd socket (Unix: `unix:///var/run/containerd/containerd.sock`, Windows: `npipe:////./pipe/containerd-containerd`) instead of the one for Docker. If the `Init|JoinConfiguration.nodeRegistration.criSocket` field is empty during cluster creation and multiple sockets are found on the host always throw an error and ask the user to specify which one to use by setting the value in the field. Make sure you update any kubeadm configuration files on disk, to not include the dockershim socket unless you are still using kubelet version < 1.24 with kubeadm >= 1.24. Remove the DockerValidor and ServiceCheck for the `docker` service from kubeadm preflight. Docker is no longer special cased during host validation and ideally this task should be done in the now external cri-dockerd project where the importance of the compatibility matters. Use `crictl` for all communication with CRI sockets for actions like pulling images and obtaining a list of running containers instead of using the docker CLI in the case of Docker. ([#107317](https://github.com/kubernetes/kubernetes/pull/107317), [@neolit123](https://github.com/neolit123))
- The feature gate was mentioned as `csiMigrationRBD` where it should have been `CSIMigrationRBD` to be in parity with other migration plugins. This release correct the same and keep it as `CSIMigrationRBD`.
    users who have configured this feature gate as `csiMigrationRBD` has to reconfigure the same to `CSIMigrationRBD` from this release. ([#107554](https://github.com/kubernetes/kubernetes/pull/107554), [@humblec](https://github.com/humblec))
- The experimental dynamic log sanitization feature has been deprecated and removed in the 1.24 release. The feature is no longer available for use. ([#107207](https://github.com/kubernetes/kubernetes/pull/107207), [@ehashman](https://github.com/ehashman))
- Kubeadm: apply `second stage` of the plan to migrate kubeadm away from the usage of the word `master` in labels and taints. For new clusters, the label `node-role.kubernetes.io/master` will no longer be added to control plane nodes, only the label `node-role.kubernetes.io/control-plane` will be added. For clusters that are being upgraded to 1.24 with `kubeadm upgrade apply`, the command will remove the label `node-role.kubernetes.io/master` from existing control plane nodes. For new clusters, both the old taint `node-role.kubernetes.io/master:NoSchedule` and new taint `node-role.kubernetes.io/control-plane:NoSchedule` will be added to control plane nodes. In release 1.20 (`first stage`), a release note instructed to preemptively tolerate the new taint. For clusters that are being upgraded to 1.24 with `kubeadm upgrade apply`, the command will add the new taint `node-role.kubernetes.io/control-plane:NoSchedule` to existing control plane nodes. Please adapt your infrastructure to these changes. In 1.25 the old taint `node-role.kubernetes.io/master:NoSchedule` will be removed. ([#107533](https://github.com/kubernetes/kubernetes/pull/107533), [@neolit123](https://github.com/neolit123))

## Changes by Kind

### Deprecation

- Deprecated `Service.Spec.LoadBalancerIP`. This field was under-specified and its meaning varies across implementations.  As of Kubernetes v1.24, users are encouraged to use implementation-specific annotations when available.  This field may be removed in a future API version. ([#107235](https://github.com/kubernetes/kubernetes/pull/107235), [@uablrek](https://github.com/uablrek))
- Kube-apiserver: the `--master-count` flag and `--endpoint-reconciler-type=master-count` reconciler are deprecated in favor of the lease reconciler ([#108062](https://github.com/kubernetes/kubernetes/pull/108062), [@aojea](https://github.com/aojea))
- Kube-apiserver: the insecure address flags `--address`, `--insecure-bind-address`, `--port` and `--insecure-port` (inert since 1.20) are removed ([#106859](https://github.com/kubernetes/kubernetes/pull/106859), [@knight42](https://github.com/knight42))
- Kubeadm: graduated the `UnversionedKubeletConfigMap` feature gate to Beta and enabled the feature by default. This implies that 1) for new clusters kubeadm will start using the `kube-system/kubelet-config` naming scheme for the kubelet ConfigMap and RBAC rules, instead of the legacy `kubelet-config-x.yy` naming. 2) during upgrade, kubeadm will only write the new scheme ConfigMap and RBAC objects. To disable the feature you can pass `UnversionedKubeletConfigMap: false` in the kubeadm config for new clusters. For upgrade on existing clusters you can also override the behavior by patching the ClusterConfiguration object in `kube-system/kubeadm-config`. More details in the associated KEP. ([#108027](https://github.com/kubernetes/kubernetes/pull/108027), [@neolit123](https://github.com/neolit123))
- Remove `tolerate-unready-endpoints` annotation in Service deprecated from 1.11, use `Service.spec.publishNotReadyAddresses` instead. ([#108020](https://github.com/kubernetes/kubernetes/pull/108020), [@tossmilestone](https://github.com/tossmilestone))
- Remove deprecated feature gates `ValidateProxyRedirects` and `StreamingProxyRedirects` ([#106830](https://github.com/kubernetes/kubernetes/pull/106830), [@pacoxu](https://github.com/pacoxu))
- Remove insecure serving configuration from cloud-provider package, which is consumed by cloud-controller-managers. ([#108953](https://github.com/kubernetes/kubernetes/pull/108953), [@nckturner](https://github.com/nckturner))
- The `--pod-infra-container-image` kubelet flag is deprecated and will be removed in future releases ([#108045](https://github.com/kubernetes/kubernetes/pull/108045), [@hakman](https://github.com/hakman))
- The `client.authentication.k8s.io/v1alpha1` ExecCredential has been removed. If you are using a client-go credential plugin that relies on the v1alpha1 API please contact the distributor of your plugin for instructions on how to migrate to the v1 API. ([#108616](https://github.com/kubernetes/kubernetes/pull/108616), [@margocrawf](https://github.com/margocrawf))
- The `node.k8s.io/v1alpha1` RuntimeClass API is no longer served. Use the `node.k8s.io/v1` API version, available since v1.20 ([#103061](https://github.com/kubernetes/kubernetes/pull/103061), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- The cluster addon for dashboard was removed. To install dashboard, see [here](https://github.com/kubernetes/dashboard/blob/master/docs/user/README.md). ([#107481](https://github.com/kubernetes/kubernetes/pull/107481), [@shu-mutou](https://github.com/shu-mutou))
- The in-tree Azure plugin has been deprecated. The Azure kubelogin plugin serves as an out-of-tree replacement via the kubectl/client-go credential plugin mechanism.  Users will now see a warning in the logs regarding this deprecation. ([#107904](https://github.com/kubernetes/kubernetes/pull/107904), [@sabbey37](https://github.com/sabbey37))
- The insecure address flags `--address` and `--port` in kube-controller-manager have had no effect since v1.20 and are removed in v1.24. ([#106860](https://github.com/kubernetes/kubernetes/pull/106860), [@knight42](https://github.com/knight42))
- The metadata.clusterName field is deprecated. This field has always been unwritable and always blank, but its presence is confusing, so we will remove it next release. Out of an abundance of caution, this release we have merely changed the name in the go struct to ensure any accidental client uses are found before complete removal. ([#108717](https://github.com/kubernetes/kubernetes/pull/108717), [@lavalamp](https://github.com/lavalamp))
- VSphere releases less than 7.0u2 are deprecated as of v1.24. Please consider upgrading vSphere to 7.0u2 or above. vSphere CSI Driver requires minimum vSphere 7.0u2.

  General Support for vSphere 6.7 will end on October 15, 2022. vSphere 6.7 Update 3 is deprecated in Kubernetes v1.24.  Customers are recommended to upgrade vSphere (both ESXi and vCenter) to 7.0u2 or above.  vSphere CSI Driver 2.2.3 and higher supports CSI Migration.

  Support for these deprecations will be available till October 15, 2022. ([#109089](https://github.com/kubernetes/kubernetes/pull/109089), [@deepakkinni](https://github.com/deepakkinni))

### API Change

- Add 2 new options for kube-proxy running in winkernel mode. `--forward-healthcheck-vip`, if specified as true, health check traffic whose destination is service VIP will be forwarded to kube-proxy's healthcheck service. `--root-hnsendpoint-name` specifies the name of the hns endpoint for the root network namespace. This option enables the pass-through load balancers like Google's GCLB to correctly health check the backend services. Without this change, the health check packets is dropped, and Windows node will be considered to be unhealthy by those load balancers. ([#99287](https://github.com/kubernetes/kubernetes/pull/99287), [@anfernee](https://github.com/anfernee))
- Added CEL runtime cost calculation into CustomerResource validation. CustomerResource validation will fail if runtime cost exceeds the budget. ([#108482](https://github.com/kubernetes/kubernetes/pull/108482), [@cici37](https://github.com/cici37))
- Added a new metric `webhook_fail_open_count` to monitor webhooks that fail to open. ([#107171](https://github.com/kubernetes/kubernetes/pull/107171), [@ltagliamonte-dd](https://github.com/ltagliamonte-dd))
- Adds a new Status subresource in Network Policy objects ([#107963](https://github.com/kubernetes/kubernetes/pull/107963), [@rikatz](https://github.com/rikatz))
- Adds support for `InterfaceNamePrefix` and `BridgeInterface` as arguments to `--detect-local-mode` option and also introduces a new optional `--pod-interface-name-prefix` and `--pod-bridge-interface` flags to kube-proxy. ([#95400](https://github.com/kubernetes/kubernetes/pull/95400), [@tssurya](https://github.com/tssurya))
- CEL CRD validation expressions may now reference existing object state using the identifier `oldSelf`. ([#108073](https://github.com/kubernetes/kubernetes/pull/108073), [@benluddy](https://github.com/benluddy))
- CRD deep copies should no longer contain shallow copies of `JSONSchemaProps.XValidations`. ([#107956](https://github.com/kubernetes/kubernetes/pull/107956), [@benluddy](https://github.com/benluddy))
- CRD writes will generate validation errors if a CEL validation rule references the identifier `oldSelf` on a part of the schema that does not support it. ([#108013](https://github.com/kubernetes/kubernetes/pull/108013), [@benluddy](https://github.com/benluddy))
- CSIStorageCapacity.storage.k8s.io: The v1beta1 version of this API is deprecated in favor of v1, and will be removed in v1.27. If a CSI driver supports storage capacity tracking, then it must get deployed with a release of external-provisioner that supports the v1 API. ([#108445](https://github.com/kubernetes/kubernetes/pull/108445), [@pohly](https://github.com/pohly))
- Custom resource requests with `fieldValidation=Strict` consistently require `apiVersion` and `kind`, matching non-strict requests ([#109019](https://github.com/kubernetes/kubernetes/pull/109019), [@liggitt](https://github.com/liggitt))
- Feature of `DefaultPodTopologySpread` is graduated to GA ([#108278](https://github.com/kubernetes/kubernetes/pull/108278), [@kerthcet](https://github.com/kerthcet))
- Feature of `NonPreemptingPriority` is graduated to GA ([#107432](https://github.com/kubernetes/kubernetes/pull/107432), [@denkensk](https://github.com/denkensk))
- Feature of `PodOverhead` is graduated to GA ([#108441](https://github.com/kubernetes/kubernetes/pull/108441), [@pacoxu](https://github.com/pacoxu))
- Fixed OpenAPI serialization of the x-kubernetes-validations field ([#107970](https://github.com/kubernetes/kubernetes/pull/107970), [@liggitt](https://github.com/liggitt))
- Fixed failed flushing logs in defer function when kubelet cmd exit 1. ([#104774](https://github.com/kubernetes/kubernetes/pull/104774), [@kerthcet](https://github.com/kerthcet))
- Fixes a regression in v1beta1 PodDisruptionBudget handling of `strategic merge patch`-type API requests for the `selector` field. Prior to 1.21, these requests would merge `matchLabels` content and replace `matchExpressions` content. In 1.21, patch requests touching the `selector` field started replacing the entire selector. This is consistent with server-side apply and the v1 PodDisruptionBudget behavior, but should not have been changed for v1beta1. ([#108138](https://github.com/kubernetes/kubernetes/pull/108138), [@liggitt](https://github.com/liggitt))
- Improve kubectl's user help commands readability ([#104736](https://github.com/kubernetes/kubernetes/pull/104736), [@lauchokyip](https://github.com/lauchokyip))
- Indexed Jobs graduated to stable. ([#107395](https://github.com/kubernetes/kubernetes/pull/107395), [@alculquicondor](https://github.com/alculquicondor))
- Introduce a v1alpha1 networking API for ClusterCIDRConfig ([#108290](https://github.com/kubernetes/kubernetes/pull/108290), [@sarveshr7](https://github.com/sarveshr7))
- Introduction of a new "sync_proxy_rules_no_local_endpoints_total" proxy metric. This metric represents the number of services with no internal endpoints. The "traffic_policy" label will contain both "internal" or "external". ([#108930](https://github.com/kubernetes/kubernetes/pull/108930), [@MaxRenaud](https://github.com/MaxRenaud))
- JobReadyPods graduates to Beta and it's enabled by default. ([#107476](https://github.com/kubernetes/kubernetes/pull/107476), [@alculquicondor](https://github.com/alculquicondor))
- Kube-apiserver: `--audit-log-version` and `--audit-webhook-version` now only support the default value of `audit.k8s.io/v1`. The v1alpha1 and v1beta1 audit log versions, deprecated since 1.13, have been removed. ([#108092](https://github.com/kubernetes/kubernetes/pull/108092), [@carlory](https://github.com/carlory))
- Kube-apiserver: the `metadata.selfLink` field can no longer be populated by kube-apiserver; it was deprecated in 1.16 and has not been populated by default since 1.20+. ([#107527](https://github.com/kubernetes/kubernetes/pull/107527), [@wojtek-t](https://github.com/wojtek-t))
- Kubelet external Credential Provider feature is moved to Beta. Credential Provider Plugin and Credential Provider Config API's updated from v1alpha1 to v1beta1 with no API changes. ([#108847](https://github.com/kubernetes/kubernetes/pull/108847), [@adisky](https://github.com/adisky))
- Make STS available replicas optional again. ([#109241](https://github.com/kubernetes/kubernetes/pull/109241), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- MaxUnavailable for StatefulSets, allows faster RollingUpdate by taking down more than 1 pod at a time. The number of pods you want to take down during a RollingUpdate is configurable using maxUnavailable parameter. ([#82162](https://github.com/kubernetes/kubernetes/pull/82162), [@krmayankk](https://github.com/krmayankk))
- Non-graceful node shutdown handling is enabled for stateful workload failovers ([#108486](https://github.com/kubernetes/kubernetes/pull/108486), [@sonasingh46](https://github.com/sonasingh46))
- Omit enum declarations from the static openapi file captured at https://git.k8s.io/kubernetes/api/openapi-spec. This file is used to generate API clients, and use of enums in those generated clients (rather than strings) can break forward compatibility with additional future values in those fields. See https://issue.k8s.io/109177 for details. ([#109178](https://github.com/kubernetes/kubernetes/pull/109178), [@liggitt](https://github.com/liggitt))
- OpenAPI V3 is turned on by default ([#109031](https://github.com/kubernetes/kubernetes/pull/109031), [@Jefftree](https://github.com/Jefftree))
- Pod affinity namespace selector and cross-namespace quota graduated to GA. The feature gate `PodAffinityNamespaceSelector` is locked and will be removed in 1.26. ([#108136](https://github.com/kubernetes/kubernetes/pull/108136), [@ahg-g](https://github.com/ahg-g))
- Promote IdentifyPodOS feature to beta. ([#107859](https://github.com/kubernetes/kubernetes/pull/107859), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- Remove a v1alpha1 networking API for ClusterCIDRConfig ([#109436](https://github.com/kubernetes/kubernetes/pull/109436), [@JamesLaverack](https://github.com/JamesLaverack))
- Renamed metrics `evictions_number` to `evictions_total` and mark it as stable. The original `evictions_number` metrics name is marked as "Deprecated" and has been removed in kubernetes 1.23 . ([#106366](https://github.com/kubernetes/kubernetes/pull/106366), [@cyclinder](https://github.com/cyclinder))
- Skip x-kubernetes-validations rules if having fundamental error against the OpenAPIv3 schema. ([#108859](https://github.com/kubernetes/kubernetes/pull/108859), [@cici37](https://github.com/cici37))
- Support for gRPC probes is now in beta. GRPCContainerProbe feature gate is enabled by default. ([#108522](https://github.com/kubernetes/kubernetes/pull/108522), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- Suspend job to GA. The feature gate `SuspendJob` is locked and will be removed in 1.26. ([#108129](https://github.com/kubernetes/kubernetes/pull/108129), [@ahg-g](https://github.com/ahg-g))
- The AnyVolumeDataSource feature is now beta, and the feature gate is enabled by default. In order to provide user feedback on PVCs with data sources, deployers must install the VolumePopulators CRD and the data-source-validator controller. ([#108736](https://github.com/kubernetes/kubernetes/pull/108736), [@bswartz](https://github.com/bswartz))
- The CertificateSigningRequest `spec.expirationSeconds` API field has graduated to GA. The `CSRDuration` feature gate for the field is now unconditionally enabled and will be removed in 1.26. ([#108782](https://github.com/kubernetes/kubernetes/pull/108782), [@cfryanr](https://github.com/cfryanr))
- The `ServerSideFieldValidation` feature has graduated to beta and is now enabled by default. Kubectl 1.24 and newer will use server-side validation instead of client-side validation when writing to API servers with the feature enabled. ([#108889](https://github.com/kubernetes/kubernetes/pull/108889), [@kevindelgado](https://github.com/kevindelgado))
- The `ServiceLBNodePortControl` feature has graduated to GA. The feature gate will be removed in 1.26. ([#107027](https://github.com/kubernetes/kubernetes/pull/107027), [@uablrek](https://github.com/uablrek))
- The deprecated kube-controller-manager flag '--deployment-controller-sync-period' has been removed, it is not used by the deployment controller. ([#107178](https://github.com/kubernetes/kubernetes/pull/107178), [@SataQiu](https://github.com/SataQiu))
- The feature `DynamicKubeletConfig` has been removed from the kubelet. ([#106932](https://github.com/kubernetes/kubernetes/pull/106932), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- The infrastructure for contextual logging is complete (feature gate implemented, JSON backend ready). ([#108995](https://github.com/kubernetes/kubernetes/pull/108995), [@pohly](https://github.com/pohly))
- This adds an optional `timeZone` field as part of the CronJob spec to support running cron jobs in a specific time zone. ([#108032](https://github.com/kubernetes/kubernetes/pull/108032), [@deejross](https://github.com/deejross))
- Updated the default API priority-and-fairness config to avoid endpoint/configmaps operations from controller-manager to all match leader-election priority level. ([#106725](https://github.com/kubernetes/kubernetes/pull/106725), [@wojtek-t](https://github.com/wojtek-t))
- `topologySpreadConstraints` includes `minDomains` field to limit the minimum number of topology domains. ([#107674](https://github.com/kubernetes/kubernetes/pull/107674), [@sanposhiho](https://github.com/sanposhiho))

### Feature

- A new Priority and Fairness metric 'apiserver_flowcontrol_work_estimate_seats_samples' has been added that tracks the estimated seats associated with a request. ([#106628](https://github.com/kubernetes/kubernetes/pull/106628), [@tkashem](https://github.com/tkashem))
- Add a deprecated cmd flag for the time interval between flushing pods from unschedulable queue to active queue or backoff queue. ([#108017](https://github.com/kubernetes/kubernetes/pull/108017), [@denkensk](https://github.com/denkensk))
- Add one metrics(`kubelet_volume_stats_health_abnormal`) of volume health state to kubelet ([#105585](https://github.com/kubernetes/kubernetes/pull/105585), [@fengzixu](https://github.com/fengzixu))
- Add the metric `container_oom_events_total` to kubelet's cAdvisor metric endpoint. ([#108004](https://github.com/kubernetes/kubernetes/pull/108004), [@jonkerj](https://github.com/jonkerj))
- Added `SetTransform` to `SharedInformer` to allow users to transform objects before they are stored. ([#107507](https://github.com/kubernetes/kubernetes/pull/107507), [@alexzielenski](https://github.com/alexzielenski))
- Added a `proxy-url` flag into `kubectl config set-cluster`. ([#105566](https://github.com/kubernetes/kubernetes/pull/105566), [@ardaguclu](https://github.com/ardaguclu))
- Added a metric for measuring end-to-end volume mount timing. ([#107006](https://github.com/kubernetes/kubernetes/pull/107006), [@gnufied](https://github.com/gnufied))
- Added a new Priority and Fairness metric `apiserver_flowcontrol_request_dispatch_no_accommodation_total` to track the number of times a request dispatch attempt results in a no-accommodation status due to lack of available seats. ([#106629](https://github.com/kubernetes/kubernetes/pull/106629), [@tkashem](https://github.com/tkashem))
- Added a path `/header?key=` to `agnhost netexec` allowing one to view what the header value is of the incoming request.

  Ex:

  ```$ curl -H "X-Forwarded-For: something" 172.17.0.2:8080/header?key=X-Forwarded-For
  something``` ([#107796](https://github.com/kubernetes/kubernetes/pull/107796), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu))
- Added completion for `kubectl config set-context`. ([#106739](https://github.com/kubernetes/kubernetes/pull/106739), [@kebe7jun](https://github.com/kebe7jun))
- Added field `add_ambient_capabilities` to the Capabilities message in the CRI-API. ([#104620](https://github.com/kubernetes/kubernetes/pull/104620), [@vinayakankugoyal](https://github.com/vinayakankugoyal))
- Added label selector flag to all `kubectl rollout` commands. ([#99758](https://github.com/kubernetes/kubernetes/pull/99758), [@aramperes](https://github.com/aramperes))
- Added more message for no PodSandbox container. ([#107116](https://github.com/kubernetes/kubernetes/pull/107116), [@yxxhero](https://github.com/yxxhero))
- Added prune flag into `diff` command to simulate `apply --prune`. ([#105164](https://github.com/kubernetes/kubernetes/pull/105164), [@ardaguclu](https://github.com/ardaguclu))
- Added support for `btrfs` resizing ([#108561](https://github.com/kubernetes/kubernetes/pull/108561), [@RomanBednar](https://github.com/RomanBednar))
- Added support for kubectl commands (`kubectl exec` and `kubectl port-forward`) via a SOCKS5 proxy. ([#105632](https://github.com/kubernetes/kubernetes/pull/105632), [@xens](https://github.com/xens))
- Adds `OpenAPIV3SchemaInterface` to `DiscoveryClient` and its variants for fetching OpenAPI v3 schema documents. ([#108992](https://github.com/kubernetes/kubernetes/pull/108992), [@alexzielenski](https://github.com/alexzielenski))
- Allow kubectl to manage resources by filename patterns without the shell expanding it first ([#102265](https://github.com/kubernetes/kubernetes/pull/102265), [@danielrodriguez](https://github.com/danielrodriguez))
- An alpha flag `--subresource` is added to get, patch, edit replace kubectl commands to fetch and update status and scale subresources. ([#99556](https://github.com/kubernetes/kubernetes/pull/99556), [@nikhita](https://github.com/nikhita))
- Apiextensions_openapi_v3_regeneration_count metric (alpha) will be emitted for OpenAPI V3. ([#109128](https://github.com/kubernetes/kubernetes/pull/109128), [@Jefftree](https://github.com/Jefftree))
- Apply ProxyTerminatingEndpoints to all traffic policies (external, internal, cluster, local). ([#108691](https://github.com/kubernetes/kubernetes/pull/108691), [@andrewsykim](https://github.com/andrewsykim))
- CEL regex patterns in x-kubernetes-valiation rules are compiled when CRDs are created/updated if the pattern is provided as a string constant in the expression. Any regex compile errors are reported as a CRD create/update validation error. ([#108617](https://github.com/kubernetes/kubernetes/pull/108617), [@jpbetz](https://github.com/jpbetz))
- CRD `x-kubernetes-validations` rules now support the CEL functions: `isSorted`, `sum`, `min`, `max`, `indexOf`, `lastIndexOf`, `find` and `findAll`. ([#108312](https://github.com/kubernetes/kubernetes/pull/108312), [@jpbetz](https://github.com/jpbetz))
- Changes the kubectl `--validate` flag from a bool to a string that accepts the values {true, strict, warn, false, ignore}
  - true/strict - perform validation and error the request on any invalid fields in the ojbect. It will attempt to perform server-side validation if it is enabled on the apiserver, otherwise it will fall back to client-side validation.
  - warn - perform server-side validation and warn on any invalid fields (but ultimately let the request succeed by dropping any invalid fields from the object). If validation is not available on the server, perform no validation.
  - false/ignore - perform no validation, silently dropping invalid fields from the object. ([#108350](https://github.com/kubernetes/kubernetes/pull/108350), [@kevindelgado](https://github.com/kevindelgado))
- Client-go metrics: change bucket distribution for `rest_client_request_duration_seconds` and `rest_client_rate_limiter_duration_seconds` from [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512] to [0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0}] ([#106911](https://github.com/kubernetes/kubernetes/pull/106911), [@aojea](https://github.com/aojea))
- Client-go: add new histogram metric to record the size of the requests and responses. ([#108296](https://github.com/kubernetes/kubernetes/pull/108296), [@aojea](https://github.com/aojea))
- CycleState is now optimized for "write once and read many times". ([#108724](https://github.com/kubernetes/kubernetes/pull/108724), [@sanposhiho](https://github.com/sanposhiho))
- Enabled beta feature HonorPVReclaimPolicy by default. ([#109035](https://github.com/kubernetes/kubernetes/pull/109035), [@deepakkinni](https://github.com/deepakkinni))
- Env var for additional cli flags used in the csi-proxy binary when a Windows nodepool is created with `kube-up.sh` ([#107806](https://github.com/kubernetes/kubernetes/pull/107806), [@mauriciopoppe](https://github.com/mauriciopoppe))
- Feature of `PreferNominatedNode` is graduated to GA. ([#106619](https://github.com/kubernetes/kubernetes/pull/106619), [@chendave](https://github.com/chendave))
- In text format, log messages that previously used quoting to prevent multi-line output (for example, text="some \"quotation\", a\nline break") will now be printed with more readable multi-line output without the escape sequences. ([#107103](https://github.com/kubernetes/kubernetes/pull/107103), [@pohly](https://github.com/pohly))
- Increase default value of discovery cache TTL for kubectl to 6 hours. ([#107141](https://github.com/kubernetes/kubernetes/pull/107141), [@mk46](https://github.com/mk46))
- Introduce policy to allow the HPA to consume the `external.metrics.k8s.io` API group. ([#104244](https://github.com/kubernetes/kubernetes/pull/104244), [@dgrisonnet](https://github.com/dgrisonnet))
- Kube-apiserver: Subresources such as `status` and `scale` now support tabular output content types. ([#103516](https://github.com/kubernetes/kubernetes/pull/103516), [@ykakarap](https://github.com/ykakarap))
- Kube-apiserver: when merging lists, Server Side Apply now prefers the order of the submitted request instead of the existing persisted object. ([#107565](https://github.com/kubernetes/kubernetes/pull/107565), [@jiahuif](https://github.com/jiahuif))
- Kubeadm: added support for dry running `kubeadm reset`. The new flag `kubeadm reset --dry-run` is similar to the existing flag for `kubeadm init/join/upgrade` and allows you to see what changes would be applied. ([#107512](https://github.com/kubernetes/kubernetes/pull/107512), [@SataQiu](https://github.com/SataQiu))
- Kubeadm: added the flag `--experimental-initial-corrupt-check` to etcd static Pod manifests to ensure etcd member data consistency ([#109074](https://github.com/kubernetes/kubernetes/pull/109074), [@neolit123](https://github.com/neolit123))
- Kubeadm: better surface errors during `kubeadm upgrade` when waiting for the kubelet to restart static pods on control plane nodes ([#108315](https://github.com/kubernetes/kubernetes/pull/108315), [@Monokaix](https://github.com/Monokaix))
- Kubeadm: improve the strict parsing of user YAML/JSON configuration files. Next to printing warnings for unknown and duplicate fields (current state), also print warnings for fields with incorrect case sensitivity - e.g. `controlPlaneEndpoint` (valid), `ControlPlaneEndpoint` (invalid). Instead of only printing warnings during `init` and `join` also print warnings when downloading the ClusterConfiguration, KubeletConfiguration or KubeProxyConfiguration objects from the cluster. This can be useful if the user has patched these objects in their respective ConfigMaps with mistakes. ([#107725](https://github.com/kubernetes/kubernetes/pull/107725), [@neolit123](https://github.com/neolit123))
- Kubectl now supports shell completion for the <type>/<name> format for specifying resources.
  kubectl now provides shell completion for container names following the `--container/-c` flag of the `exec` command.
  kubectl's shell completion now suggests resource types for commands that only apply to pods. ([#108493](https://github.com/kubernetes/kubernetes/pull/108493), [@marckhouzam](https://github.com/marckhouzam))
- Kubelet: add `kubelet_volume_metric_collection_duration_seconds` metrics for volume disk usage calculation duration ([#107201](https://github.com/kubernetes/kubernetes/pull/107201), [@pacoxu](https://github.com/pacoxu))
- Kubelet: the following dockershim related flags are also removed along with dockershim `--experimental-dockershim-root-directory`, `--docker-endpoint`, `--image-pull-progress-deadline`, `--network-plugin`, `--cni-conf-dir`, `--cni-bin-dir`, `--cni-cache-dir`, `--network-plugin-mtu`. ([#106907](https://github.com/kubernetes/kubernetes/pull/106907), [@cyclinder](https://github.com/cyclinder))
- Kubernetes 1.24 bumped version of golang it is compiled with to go1.18, which introduced significant changes to its garbage collection algorithm. As a result, we observed an increase in memory usage for kube-apiserver in larger an heavily loaded clusters up to ~25% (with the benefit of API call latencies drop by up to 10x on 99th percentiles). If the memory increase is not acceptable for you you can mitigate by setting GOGC env variable (for our tests using GOGC=63 brings memory usage back to original value, although the exact value may depend on usage patterns on your cluster). ([#108870](https://github.com/kubernetes/kubernetes/pull/108870), [@dims](https://github.com/dims))
- Kubernetes 1.24 is built with go1.18, which will no longer validate certificates signed with a SHA-1 hash algorithm by default. See https://golang.org/doc/go1.18#sha1 for more details. If you are using certificates like this in admission or conversion ([#109024](https://github.com/kubernetes/kubernetes/pull/109024), [@stlaz](https://github.com/stlaz))
- Kubernetes in now built with go1.18rc1 ([#107105](https://github.com/kubernetes/kubernetes/pull/107105), [@justaugustus](https://github.com/justaugustus))
- Kubernetes is now built with Golang 1.17.4 ([#106833](https://github.com/kubernetes/kubernetes/pull/106833), [@cpanato](https://github.com/cpanato))
- Kubernetes is now built with Golang 1.17.5. ([#106956](https://github.com/kubernetes/kubernetes/pull/106956), [@cpanato](https://github.com/cpanato))
- Kubernetes is now built with Golang 1.17.6. ([#107612](https://github.com/kubernetes/kubernetes/pull/107612), [@palnabarun](https://github.com/palnabarun))
- Kubernetes is now built with Golang 1.17.7 ([#108091](https://github.com/kubernetes/kubernetes/pull/108091), [@xmudrii](https://github.com/xmudrii))
- Kubernetes is now built with Golang 1.18.1 ([#109461](https://github.com/kubernetes/kubernetes/pull/109461), [@cpanato](https://github.com/cpanato))
- Leader Migration is now GA. All new configuration files onwards should use version v1. ([#109072](https://github.com/kubernetes/kubernetes/pull/109072), [@jiahuif](https://github.com/jiahuif))
- Mark AzureDisk CSI migration as GA ([#107681](https://github.com/kubernetes/kubernetes/pull/107681), [@andyzhangx](https://github.com/andyzhangx))
- Move volume expansion feature to GA ([#108929](https://github.com/kubernetes/kubernetes/pull/108929), [@gnufied](https://github.com/gnufied))
- Moving MixedProtocolLBService from alpha to beta ([#109213](https://github.com/kubernetes/kubernetes/pull/109213), [@bridgetkromhout](https://github.com/bridgetkromhout))
- New "field_validation_request_duration_seconds" metric, measures how long requests take, indicating the value of the fieldValidation query parameter and whether or not server-side field validation is enabled on the apiserver ([#109120](https://github.com/kubernetes/kubernetes/pull/109120), [@kevindelgado](https://github.com/kevindelgado))
- New feature gate, ServiceIPStaticSubrange, to enable the new strategy in the Service IP allocators, so the IP range is subdivided and dynamic allocated ClusterIP addresses for Services are allocated preferently from the upper range. ([#106792](https://github.com/kubernetes/kubernetes/pull/106792), [@aojea](https://github.com/aojea))
- OpenAPI definitions served by kube-apiserver now include enum types by default. ([#108898](https://github.com/kubernetes/kubernetes/pull/108898), [@jiahuif](https://github.com/jiahuif))
- OpenStack Cinder CSI migration is now GA and switched on by default, Cinder CSI driver must be installed on clusters on OpenStack for Cinder volumes to work (has been since v1.21). ([#107462](https://github.com/kubernetes/kubernetes/pull/107462), [@dims](https://github.com/dims))
- PreFilter extension in the scheduler framework now returns not only status but also PreFilterResult ([#108648](https://github.com/kubernetes/kubernetes/pull/108648), [@ahg-g](https://github.com/ahg-g))
- Promoted graceful shutdown based on pod priority to beta ([#107986](https://github.com/kubernetes/kubernetes/pull/107986), [@wzshiming](https://github.com/wzshiming))
- Removed feature gate `SetHostnameAsFQDN`. ([#108038](https://github.com/kubernetes/kubernetes/pull/108038), [@mengjiao-liu](https://github.com/mengjiao-liu))
- Removed kube-scheduler insecure flags. You can use `--bind-address` and `--secure-port` instead. ([#106865](https://github.com/kubernetes/kubernetes/pull/106865), [@jonyhy96](https://github.com/jonyhy96))
- Removed the `ImmutableEphemeralVolumes` feature gate. ([#107152](https://github.com/kubernetes/kubernetes/pull/107152), [@mengjiao-liu](https://github.com/mengjiao-liu))
- Set `PodMaxUnschedulableQDuration` as 5 min. ([#108761](https://github.com/kubernetes/kubernetes/pull/108761), [@denkensk](https://github.com/denkensk))
- Support in-tree PV deletion protection finalizer. ([#108400](https://github.com/kubernetes/kubernetes/pull/108400), [@deepakkinni](https://github.com/deepakkinni))
- The `.spec.loadBalancerClass` field for Services is now generally available. ([#107979](https://github.com/kubernetes/kubernetes/pull/107979), [@XudongLiuHarold](https://github.com/XudongLiuHarold))
- The `NamespaceDefaultLabelName` feature gate, GA since v1.22, is now removed. ([#106838](https://github.com/kubernetes/kubernetes/pull/106838), [@mengjiao-liu](https://github.com/mengjiao-liu))
- The `kubectl logs` will now warn and default to the first container in a pod. This new behavior brings it in line with `kubectl exec`. ([#105964](https://github.com/kubernetes/kubernetes/pull/105964), [@kidlj](https://github.com/kidlj))
- The `v1` version of `LeaderMigrationConfiguration` supports only `leases` API for leader election. To use formerly supported mechanisms, please continue using `v1beta1`. ([#108016](https://github.com/kubernetes/kubernetes/pull/108016), [@jiahuif](https://github.com/jiahuif))
- The kubelet now creates an iptables chain named `KUBE-IPTABLES-HINT` in
  the `mangle` table. Containerized components that need to modify iptables
  rules in the host network namespace can use the existence of this chain
  to more-reliably determine whether the system is using iptables-legacy or
  iptables-nft. ([#109059](https://github.com/kubernetes/kubernetes/pull/109059), [@danwinship](https://github.com/danwinship))
- The output of `kubectl describe ingress` now includes an IngressClass name if available. ([#107921](https://github.com/kubernetes/kubernetes/pull/107921), [@mpuckett159](https://github.com/mpuckett159))
- The scheduler prints info logs when the extender returned an error. (`--v>5`) ([#107974](https://github.com/kubernetes/kubernetes/pull/107974), [@sanposhiho](https://github.com/sanposhiho))
- The script `cluster/gce/gci/configure.sh` now supports downloading `crictl` on ARM64 nodes ([#108034](https://github.com/kubernetes/kubernetes/pull/108034), [@tstapler](https://github.com/tstapler))
- Turn on `CSIMigrationAzureFile` by default on 1.24 ([#105070](https://github.com/kubernetes/kubernetes/pull/105070), [@andyzhangx](https://github.com/andyzhangx))
- Update the k8s.io/system-validators library to v1.7.0 ([#108988](https://github.com/kubernetes/kubernetes/pull/108988), [@neolit123](https://github.com/neolit123))
- Updated golang.org/x/net to v0.0.0-20211209124913-491a49abca63. ([#106949](https://github.com/kubernetes/kubernetes/pull/106949), [@cpanato](https://github.com/cpanato))
- Updates `kubectl kustomize` and `kubectl apply -k` to Kustomize v4.5.4 ([#108994](https://github.com/kubernetes/kubernetes/pull/108994), [@KnVerey](https://github.com/KnVerey))
- When invoked with `-list-images`, the `e2e.test` binary now also lists the images that might be needed for storage tests. ([#108458](https://github.com/kubernetes/kubernetes/pull/108458), [@pohly](https://github.com/pohly))
- `kubectl config delete-user` now supports completion ([#107142](https://github.com/kubernetes/kubernetes/pull/107142), [@dimbleby](https://github.com/dimbleby))
- `kubectl create token` can now be used to request a service account token, and permission to request service account tokens is added to the `edit` and `admin` RBAC roles ([#107880](https://github.com/kubernetes/kubernetes/pull/107880), [@liggitt](https://github.com/liggitt))
- `kubectl version` now includes information on the embedded version of Kustomize ([#108817](https://github.com/kubernetes/kubernetes/pull/108817), [@KnVerey](https://github.com/KnVerey))

### Bug or Regression

- A node IP provided to kublet via `--node-ip` will now be preferred for when determining the node's primary IP and using the external cloud provider (CCM). ([#107750](https://github.com/kubernetes/kubernetes/pull/107750), [@stephenfin](https://github.com/stephenfin))
- A static pod that is rapidly updated was failing to start until the Kubelet was restarted. ([#107900](https://github.com/kubernetes/kubernetes/pull/107900), [@smarterclayton](https://github.com/smarterclayton))
- Add one metrics(`kubelet_volume_stats_health_abnormal`) of volume health state to kubelet ([#108758](https://github.com/kubernetes/kubernetes/pull/108758), [@fengzixu](https://github.com/fengzixu))
- Added a new label `type` to `apiserver_flowcontrol_request_execution_seconds` metric - it has the following values: - 'regular': indicates that it is a non long running request - 'watch': indicates that it is a watch request. ([#105517](https://github.com/kubernetes/kubernetes/pull/105517), [@tkashem](https://github.com/tkashem))
- Added a test to guarantee that conformance clusters require at least 2 untainted nodes. ([#106313](https://github.com/kubernetes/kubernetes/pull/106313), [@aojea](https://github.com/aojea))
- Adds PV deletion protection finalizer only when PV reclaimPolicy is Delete for dynamically provisioned volumes. ([#109205](https://github.com/kubernetes/kubernetes/pull/109205), [@deepakkinni](https://github.com/deepakkinni))
- Allowed attached volumes to be mounted quicker by skipping exponential backoff when checking for reported-in-use volumes. ([#106853](https://github.com/kubernetes/kubernetes/pull/106853), [@gnufied](https://github.com/gnufied))
- Alowed useful inclusion of `-args $prog_args` in KUBE_TEST_ARGS, when doing `make test-integration`. ([#107516](https://github.com/kubernetes/kubernetes/pull/107516), [@MikeSpreitzer](https://github.com/MikeSpreitzer))
- An inefficient lock in EndpointSlice controller metrics cache has been reworked. Network programming latency may be significantly reduced in certain scenarios, especially in clusters with a large number of Services. ([#107091](https://github.com/kubernetes/kubernetes/pull/107091), [@robscott](https://github.com/robscott))
- Apiserver will now reject connection attempts to `0.0.0.0/::` when handling a proxy subresource request. ([#107402](https://github.com/kubernetes/kubernetes/pull/107402), [@anguslees](https://github.com/anguslees))
- Bug: client-go clientset was not defaulting to the user agent, and was using the default golang agent for all the requests. ([#108772](https://github.com/kubernetes/kubernetes/pull/108772), [@aojea](https://github.com/aojea))
- Bump `sigs.k8s.io/apiserver-network-proxy/konnectivity-client@v0.0.30` to fix a goroutine leak in kube-apiserver when using egress selctor with the gRPC mode. ([#108437](https://github.com/kubernetes/kubernetes/pull/108437), [@andrewsykim](https://github.com/andrewsykim))
- CEL validation failure returns object type instead of object. ([#107090](https://github.com/kubernetes/kubernetes/pull/107090), [@cici37](https://github.com/cici37))
- CRI-API: IPs returned by `PodSandboxNetworkStatus`` are ignored by the kubelet for host-network pods. ([#106715](https://github.com/kubernetes/kubernetes/pull/106715), [@aojea](https://github.com/aojea))
- Call `NodeExpand` on all nodes in case of RWX volumes ([#108693](https://github.com/kubernetes/kubernetes/pull/108693), [@gnufied](https://github.com/gnufied))
- Changed node staging path for CSI driver to use a PV agnostic path. Nodes must be drained before updating the kubelet with this change. ([#107065](https://github.com/kubernetes/kubernetes/pull/107065), [@saikat-royc](https://github.com/saikat-royc))
- Client-go: fixed the paged list calls with `ResourceVersionMatch` set would fail once paging is kicked in. ([#107311](https://github.com/kubernetes/kubernetes/pull/107311), [@fasaxc](https://github.com/fasaxc))
- Correct event registration for multiple scheduler plugins; this fixes a potential significant delay in re-queueing unschedulable pods. ([#109442](https://github.com/kubernetes/kubernetes/pull/109442), [@ahg-g](https://github.com/ahg-g))
- Etcd: Update to v3.5.3 ([#109471](https://github.com/kubernetes/kubernetes/pull/109471), [@justaugustus](https://github.com/justaugustus))
- Existing InTree AzureFile PVs which don't have a secret namespace defined will now work properly after enabling CSI migration - the namespace will be obtained from ClaimRef. ([#108000](https://github.com/kubernetes/kubernetes/pull/108000), [@RomanBednar](https://github.com/RomanBednar))
- Failure to start a container cannot accidentally result in the pod being considered "Succeeded" in the presence of deletion. ([#107845](https://github.com/kubernetes/kubernetes/pull/107845), [@smarterclayton](https://github.com/smarterclayton))
- Fix a race in the timeout handler that could lead to kube-apiserver crashes ([#108455](https://github.com/kubernetes/kubernetes/pull/108455), [@Argh4k](https://github.com/Argh4k))
- Fix container creation errors for pods with cpu requests bigger than 256 cpus ([#106570](https://github.com/kubernetes/kubernetes/pull/106570), [@odinuge](https://github.com/odinuge))
- Fix issue where the job controller might not remove the job tracking finalizer from pods when deleting a job, or when the pod is orphan ([#108752](https://github.com/kubernetes/kubernetes/pull/108752), [@alculquicondor](https://github.com/alculquicondor))
- Fix libct/cg/fs2: fixed GetStats for unsupported hugetlb error on Raspbian Bullseye ([#106912](https://github.com/kubernetes/kubernetes/pull/106912), [@Letme](https://github.com/Letme))
- Fix the bug that the outdated services may be sent to the cloud provider ([#107631](https://github.com/kubernetes/kubernetes/pull/107631), [@lzhecheng](https://github.com/lzhecheng))
- Fix the overestimated cost of delegated API requests in kube-apiserver API priority & fairness ([#109188](https://github.com/kubernetes/kubernetes/pull/109188), [@wojtek-t](https://github.com/wojtek-t))
- Fix to allow `fsGroup` to be applied for CSI Inline Volumes ([#108662](https://github.com/kubernetes/kubernetes/pull/108662), [@dobsonj](https://github.com/dobsonj))
- Fixed CSI migration of Azure Disk in-tree StorageClasses with topology requirements in Azure regions that do not have availability zones. ([#109154](https://github.com/kubernetes/kubernetes/pull/109154), [@jsafrane](https://github.com/jsafrane))
- Fixed `--retries` functionality for negative values in `kubectl cp` ([#108748](https://github.com/kubernetes/kubernetes/pull/108748), [@atiratree](https://github.com/atiratree))
- Fixed `azureDisk` parameter lowercase translation issue. ([#107429](https://github.com/kubernetes/kubernetes/pull/107429), [@andyzhangx](https://github.com/andyzhangx))
- Fixed `azureFile` `volumeID` collision issue in CSI migration. ([#107575](https://github.com/kubernetes/kubernetes/pull/107575), [@andyzhangx](https://github.com/andyzhangx))
- Fixed a bug in attachdetach controller that didn't properly handle kube-apiserver errors leading to stuck attachments/detachments. ([#108167](https://github.com/kubernetes/kubernetes/pull/108167), [@jfremy](https://github.com/jfremy))
- Fixed a bug that a pod's `.status.nominatedNodeName` is not cleared properly, and thus over-occupied system resources. ([#106816](https://github.com/kubernetes/kubernetes/pull/106816), [@Huang-Wei](https://github.com/Huang-Wei))
- Fixed a bug that caused credentials in an exec plugin to override the static certificates set in a kubeconfig. ([#107410](https://github.com/kubernetes/kubernetes/pull/107410), [@margocrawf](https://github.com/margocrawf))
- Fixed a bug that could cause panic when a `/healthz` request times out. ([#107034](https://github.com/kubernetes/kubernetes/pull/107034), [@benluddy](https://github.com/benluddy))
- Fixed a bug that out-of-tree plugin is misplaced when using scheduler v1beta3 config ([#108613](https://github.com/kubernetes/kubernetes/pull/108613), [@Huang-Wei](https://github.com/Huang-Wei))
- Fixed a bug where a partial `EndpointSlice` update could cause node name information to be dropped from endpoints that were not updated. ([#108198](https://github.com/kubernetes/kubernetes/pull/108198), [@liggitt](https://github.com/liggitt))
- Fixed a bug where unwanted fields were being returned from a `create --dry-run`: uid and, if generateName was used, name. ([#107088](https://github.com/kubernetes/kubernetes/pull/107088), [@joejulian](https://github.com/joejulian))
- Fixed a bug where vSphere client connections where not being closed during testing. Leaked vSphere client sessions were causing resource exhaustion during automated testing. ([#107337](https://github.com/kubernetes/kubernetes/pull/107337), [@derek-pryor](https://github.com/derek-pryor))
- Fixed a panic when using invalid output format in `kubectl create secret` command. ([#107221](https://github.com/kubernetes/kubernetes/pull/107221), [@rikatz](https://github.com/rikatz))
- Fixed a rare race condition handling requests that timeout. ([#107452](https://github.com/kubernetes/kubernetes/pull/107452), [@liggitt](https://github.com/liggitt))
- Fixed a regression in 1.23 that incorrectly pruned data from array items of a custom resource that set `x-kubernetes-preserve-unknown-fields: true`. ([#107688](https://github.com/kubernetes/kubernetes/pull/107688), [@liggitt](https://github.com/liggitt))
- Fixed a regression in 1.23 where update requests to previously persisted `Service` objects that have not been modified since 1.19 can be rejected with an incorrect `spec.clusterIPs: Required value` error. ([#107847](https://github.com/kubernetes/kubernetes/pull/107847), [@thockin](https://github.com/thockin))
- Fixed a regression that could incorrectly reject pods with `OutOfCpu` errors if they were rapidly scheduled after other pods were reported as complete in the API. The Kubelet now waits to report the phase of a pod as terminal in the API until all running containers are guaranteed to have stopped and no new containers can be started.  Short-lived pods may take slightly longer (~1s) to report Succeeded or Failed after this change. ([#108366](https://github.com/kubernetes/kubernetes/pull/108366), [@smarterclayton](https://github.com/smarterclayton))
- Fixed bug in `TopologyManager` for ensuring aligned allocations on machines with more than 2 NUMA nodes ([#108052](https://github.com/kubernetes/kubernetes/pull/108052), [@klueska](https://github.com/klueska))
- Fixed bug in error messaging for basic-auth and ssh secret validations. ([#106179](https://github.com/kubernetes/kubernetes/pull/106179), [@vivek-koppuru](https://github.com/vivek-koppuru))
- Fixed detaching CSI volumes from nodes when a CSI driver name has prefix "csi-". ([#107025](https://github.com/kubernetes/kubernetes/pull/107025), [@jsafrane](https://github.com/jsafrane))
- Fixed duplicate port opening in kube-proxy when `--nodeport-addresses` is empty. ([#107413](https://github.com/kubernetes/kubernetes/pull/107413), [@tnqn](https://github.com/tnqn))
- Fixed handling of objects with invalid selectors. ([#107559](https://github.com/kubernetes/kubernetes/pull/107559), [@liggitt](https://github.com/liggitt))
- Fixed indexer bug that resulted in incorrect index updates if number of index values for a given object was changing during update ([#109137](https://github.com/kubernetes/kubernetes/pull/109137), [@wojtek-t](https://github.com/wojtek-t))
- Fixed kubectl bug where bash completions don't work if `--context` flag is specified with a value that contains a colon. ([#107439](https://github.com/kubernetes/kubernetes/pull/107439), [@brianpursley](https://github.com/brianpursley))
- Fixed performance regression in JSON logging caused by syncing stdout every time error was logged. ([#107035](https://github.com/kubernetes/kubernetes/pull/107035), [@serathius](https://github.com/serathius))
- Fixed regression in CPUManager that it will release exclusive CPUs in app containers inherited from init containers when the init containers were removed. ([#104837](https://github.com/kubernetes/kubernetes/pull/104837), [@eggiter](https://github.com/eggiter))
- Fixed static pod add and removes restarts in certain cases. ([#107695](https://github.com/kubernetes/kubernetes/pull/107695), [@rphillips](https://github.com/rphillips))
- Fixed: deleted a non-existent Azure disk issue. ([#107406](https://github.com/kubernetes/kubernetes/pull/107406), [@andyzhangx](https://github.com/andyzhangx))
- Fixed: do not return early in the node informer when there is no change of the topology label. ([#108149](https://github.com/kubernetes/kubernetes/pull/108149), [@nilo19](https://github.com/nilo19))
- Fixed: removed outdated ipv4 route when the corresponding node is deleted. ([#106164](https://github.com/kubernetes/kubernetes/pull/106164), [@nilo19](https://github.com/nilo19))
- Fixes bug in CronJob Controller V2 where it would lose track of jobs upon job template labels change. ([#107997](https://github.com/kubernetes/kubernetes/pull/107997), [@d-honeybadger](https://github.com/d-honeybadger))
- If drainer has nil for Ctx or Client it will error with `RunCordonOrUncordon`. ([#105297](https://github.com/kubernetes/kubernetes/pull/105297), [@jackfrancis](https://github.com/jackfrancis))
- Improved handling of unmount failures when device may be in-use by another container/process. ([#107789](https://github.com/kubernetes/kubernetes/pull/107789), [@gnufied](https://github.com/gnufied))
- Improved logging when volume times out waiting for attach/detach. ([#108628](https://github.com/kubernetes/kubernetes/pull/108628), [@RomanBednar](https://github.com/RomanBednar))
- Improved the rounding of `PodTopologySpread` scores to offer better scoring when spreading a low number of pods. ([#107384](https://github.com/kubernetes/kubernetes/pull/107384), [@sanposhiho](https://github.com/sanposhiho))
- Increase Azure ACR credential provider timeout ([#108209](https://github.com/kubernetes/kubernetes/pull/108209), [@andyzhangx](https://github.com/andyzhangx))
- Kube-apiserver: Server Side Apply merge order is reverted to match v1.22 behavior until http://issue.k8s.io/104641 is resolved. ([#106660](https://github.com/kubernetes/kubernetes/pull/106660), [@liggitt](https://github.com/liggitt))
- Kube-apiserver: ensures the namespace of objects sent to admission webhooks matches the request namespace. Previously, objects without a namespace set would have the request namespace populated after mutating admission, and objects with a namespace that did not match the request namespace would be rejected after admission. ([#94637](https://github.com/kubernetes/kubernetes/pull/94637), [@liggitt](https://github.com/liggitt))
- Kube-apiserver: removed `apf_fd` from server logs which could contain data identifying the requesting user ([#108631](https://github.com/kubernetes/kubernetes/pull/108631), [@jupblb](https://github.com/jupblb))
- Kube-proxy in iptables mode now only logs the full iptables input at `-v=9` rather than `-v=5`. ([#108224](https://github.com/kubernetes/kubernetes/pull/108224), [@danwinship](https://github.com/danwinship))
- Kube-proxy will no longer hold service node ports open on the node. Users are still advised not to run any listener on node ports range used by kube-proxy. ([#108496](https://github.com/kubernetes/kubernetes/pull/108496), [@khenidak](https://github.com/khenidak))
- Kubeadm: allow the `certs check-expiration` command to not require the existence of the cluster CA key (ca.key file) when checking the expiration of managed certificates in kubeconfig files. ([#106854](https://github.com/kubernetes/kubernetes/pull/106854), [@neolit123](https://github.com/neolit123))
- Kubeadm: during execution of the `certs check-expiration` command, treat the etcd CA as external if there is a missing etcd CA key file (etcd/ca.key) and perform the proper validation on certificates signed by the etcd CA. Additionally, make sure that the CA for all entries in the output table is included - for both certificates on disk and in kubeconfig files. ([#106891](https://github.com/kubernetes/kubernetes/pull/106891), [@neolit123](https://github.com/neolit123))
- Kubeadm: fixed a bug related to a warning printed if the `KubeletConfiguration` `resolvConf` field value does not match `/run/systemd/resolve/resolv.conf` ([#107785](https://github.com/kubernetes/kubernetes/pull/107785), [@chendave](https://github.com/chendave))
- Kubeadm: fixed a bug when using `kubeadm init --dry-run` with certificate authority files (`ca.key` / `ca.crt`) present in `/etc/kubernetes/pki`) ([#108410](https://github.com/kubernetes/kubernetes/pull/108410), [@Haleygo](https://github.com/Haleygo))
- Kubeadm: fixed a bug where Windows nodes fail to join an IPv6 cluster due to preflight errors ([#108769](https://github.com/kubernetes/kubernetes/pull/108769), [@SataQiu](https://github.com/SataQiu))
- Kubeadm: fixed the bug that `kubeadm certs generate-csr` command does not remove duplicated SANs ([#107982](https://github.com/kubernetes/kubernetes/pull/107982), [@SataQiu](https://github.com/SataQiu))
- Kubelet now checks "NoExecute" taint/toleration before accepting pods, except for static pods. ([#101218](https://github.com/kubernetes/kubernetes/pull/101218), [@gjkim42](https://github.com/gjkim42))
- Metrics Server image bumped to v0.5.2 ([#106492](https://github.com/kubernetes/kubernetes/pull/106492), [@serathius](https://github.com/serathius))
- Modified command line errors (for example, `kubectl list` -> `unknown command`) that were printed as log message with escaped line breaks instead of a multi-line plain text, making the error hard to read. ([#107044](https://github.com/kubernetes/kubernetes/pull/107044), [@pohly](https://github.com/pohly))
- Modified log messages that were logged with `"v":0` in JSON output although they were debug messages with a higher verbosity. ([#106978](https://github.com/kubernetes/kubernetes/pull/106978), [@pohly](https://github.com/pohly))
- No ([#107769](https://github.com/kubernetes/kubernetes/pull/107769), [@liurupeng](https://github.com/liurupeng)) [SIG Cloud Provider and Windows]
- NodeRestriction admission: nodes are now allowed to update PersistentVolumeClaim status fields `resizeStatus` and `allocatedResources` when the `RecoverVolumeExpansionFailure` feature is enabled. ([#107686](https://github.com/kubernetes/kubernetes/pull/107686), [@gnufied](https://github.com/gnufied))
- Only extend token lifetimes when `--service-account-extend-token-expiration` is true and the requested token audiences are empty or exactly match all values for `--api-audiences`. ([#105954](https://github.com/kubernetes/kubernetes/pull/105954), [@jyotimahapatra](https://github.com/jyotimahapatra))
- Prevent kube-scheduler from nominating a Pod that was already scheduled to a node ([#109245](https://github.com/kubernetes/kubernetes/pull/109245), [@alculquicondor](https://github.com/alculquicondor))
- Prevent unnecessary `Endpoints` and `EndpointSlice` updates caused by `Pod ResourceVersion` change ([#108078](https://github.com/kubernetes/kubernetes/pull/108078), [@tnqn](https://github.com/tnqn))
- Print `<default>` as the value in case kubectl describe ingress shows `default-backend:80` when no default backend is present ([#108506](https://github.com/kubernetes/kubernetes/pull/108506), [@jlsong01](https://github.com/jlsong01))
- Publishing kube-proxy metrics for Windows kernel-mode ([#106581](https://github.com/kubernetes/kubernetes/pull/106581), [@knabben](https://github.com/knabben))
- Re-adds response status and headers on verbose kubectl responses ([#108505](https://github.com/kubernetes/kubernetes/pull/108505), [@rikatz](https://github.com/rikatz))
- Record requests rejected with 429 in the apiserver_request_total metric ([#108927](https://github.com/kubernetes/kubernetes/pull/108927), [@wojtek-t](https://github.com/wojtek-t))
- Removed validation if AppArmor profiles are loaded on the local node. This should be handled by the container runtime. ([#97966](https://github.com/kubernetes/kubernetes/pull/97966), [@saschagrunert](https://github.com/saschagrunert))
- Replace the url label of `rest_client_request_duration_seconds` and `rest_client_rate_limiter_duration_seconds` metrics with a host label to prevent cardinality explosions and keep only the useful information. This is a breaking change required for security reasons. ([#106539](https://github.com/kubernetes/kubernetes/pull/106539), [@dgrisonnet](https://github.com/dgrisonnet))
- Restored `NumPDBViolations` info of nodes, when `HTTPExtender ProcessPreemption`. This info will be used in subsequent filtering steps - `pickOneNodeForPreemption` ([#105853](https://github.com/kubernetes/kubernetes/pull/105853), [@caden2016](https://github.com/caden2016))
- Reverted graceful node shutdown to match 1.21 behavior of setting pods that have not yet successfully completed to "Failed" phase if the GracefulNodeShutdown feature is enabled in kubelet. The GracefulNodeShutdown feature is beta and must be explicitly configured via kubelet config to be enabled in 1.21+. This changes 1.22 and 1.23 behavior on node shutdown to match 1.21. If you do not want pods to be marked terminated on node shutdown in 1.22 and 1.23, disable the GracefulNodeShutdown feature. ([#106901](https://github.com/kubernetes/kubernetes/pull/106901), [@bobbypage](https://github.com/bobbypage))
- Reverts the CRI API version surfaced by dockershim to v1alpha2 ([#106803](https://github.com/kubernetes/kubernetes/pull/106803), [@saschagrunert](https://github.com/saschagrunert))
- Services with "internalTrafficPolicy: Local" now behave more like
  "externalTrafficPolicy: Local". Also, "internalTrafficPolicy: Local,
  externalTrafficPolicy: Cluster" is now implemented correctly. ([#106497](https://github.com/kubernetes/kubernetes/pull/106497), [@danwinship](https://github.com/danwinship))
- Sets JobTrackingWithFinalizers, a beta feature, as disabled by default, due to unresolved bug https://github.com/kubernetes/kubernetes/issues/109485 ([#109487](https://github.com/kubernetes/kubernetes/pull/109487), [@alculquicondor](https://github.com/alculquicondor))
- Skip re-allocate logic if pod is already removed to avoid panic ([#108831](https://github.com/kubernetes/kubernetes/pull/108831), [@waynepeking348](https://github.com/waynepeking348))
- The Service field `spec.internalTrafficPolicy` is no longer defaulted for Services when the type is `ExternalName`. The field is also dropped on read when the Service type is `ExternalName`. ([#104846](https://github.com/kubernetes/kubernetes/pull/104846), [@andrewsykim](https://github.com/andrewsykim))
- The `ServerSideFieldValidation` feature has been reverted to alpha for 1.24. ([#109271](https://github.com/kubernetes/kubernetes/pull/109271), [@liggitt](https://github.com/liggitt))
- The `TopologyAwareHints` feature gate is now enabled by default. This will allow users to opt-in to Topology Aware Hints by setting the `service.kubernetes.io/topology-aware-hints` on a Service. This will not affect any Services without that annotation set. ([#108747](https://github.com/kubernetes/kubernetes/pull/108747), [@robscott](https://github.com/robscott))
- The deprecated flag `--really-crash-for-testing` was removed. ([#101719](https://github.com/kubernetes/kubernetes/pull/101719), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- The kubelet no longer forcefully closes active connections on heartbeat failures, using the HTTP2 health check mechanism to detect broken connections. Users can force the previous behavior of the kubelet by setting the environment variable DISABLE_HTTP2. ([#108107](https://github.com/kubernetes/kubernetes/pull/108107), [@aojea](https://github.com/aojea))
- This code change fixes the bug that UDP services would trigger unnecessary LoadBalancer updates. The root cause is that a field not working for non-TCP protocols is considered.
  ref: https://github.com/kubernetes-sigs/cloud-provider-azure/pull/1090 ([#107981](https://github.com/kubernetes/kubernetes/pull/107981), [@lzhecheng](https://github.com/lzhecheng))
- Topology translation of in-tree vSphere volume to vSphere CSI. ([#108611](https://github.com/kubernetes/kubernetes/pull/108611), [@divyenpatel](https://github.com/divyenpatel))
- Updating kubelet permissions check for Windows nodes to see if process is elevated instead of checking if process owner is in Administrators group ([#108146](https://github.com/kubernetes/kubernetes/pull/108146), [@marosset](https://github.com/marosset))
- `apiserver`, if configured to reconcile the `kubernetes.default` service endpoints, checks if the configured Service IP range matches the apiserver public address IP family, and fails to start if not. ([#106721](https://github.com/kubernetes/kubernetes/pull/106721), [@aojea](https://github.com/aojea))
- `kubectl version` now fails when given extra arguments. ([#107967](https://github.com/kubernetes/kubernetes/pull/107967), [@jlsong01](https://github.com/jlsong01))

### Other (Cleanup or Flake)

- '`build/dependencies.yaml`: remove the dependency on Docker. With the dockershim removal, core Kubernetes no longer
  has to track the latest validated version of Docker.' ([#107607](https://github.com/kubernetes/kubernetes/pull/107607), [@neolit123](https://github.com/neolit123))
- API server's deprecated `--experimental-encryption-provider-config` flag is now removed. Adapt your machinery to use the `--encryption-provider-config` flag that is available since v1.13. ([#108423](https://github.com/kubernetes/kubernetes/pull/108423), [@ialidzhikov](https://github.com/ialidzhikov))
- API server's deprecated `--target-ram-mb` flag is now removed. ([#108457](https://github.com/kubernetes/kubernetes/pull/108457), [@ialidzhikov](https://github.com/ialidzhikov))
- Added PreemptionPolicy in PriorityClass describe ([#108701](https://github.com/kubernetes/kubernetes/pull/108701), [@denkensk](https://github.com/denkensk))
- Added an e2e test to verify that the cluster is not vulnerable to CVE-2021-29923 when using Services with IPs with leading zeros, note that this test is a necessary but not sufficient condition, all the components in the clusters that consume IPs addresses from the APIs MUST interpret them as decimal or discard them. ([#107552](https://github.com/kubernetes/kubernetes/pull/107552), [@aojea](https://github.com/aojea))
- Added an example for the `kubectl plugin list` command. ([#106600](https://github.com/kubernetes/kubernetes/pull/106600), [@bergerhoffer](https://github.com/bergerhoffer))
- Added details about preemption in the event for scheduling failed. ([#107775](https://github.com/kubernetes/kubernetes/pull/107775), [@denkensk](https://github.com/denkensk))
- Allow KUBE_TEST_REPO_LIST to be a remote url ([#108429](https://github.com/kubernetes/kubernetes/pull/108429), [@dims](https://github.com/dims))
- Client-go: if resetting the body fails before a retry, an error is now surfaced to the user. ([#109050](https://github.com/kubernetes/kubernetes/pull/109050), [@MadhavJivrajani](https://github.com/MadhavJivrajani))
- Deprecate apiserver_dropped_requests_total metric. The same data can be read from apiserver_request_terminations_total metric. ([#109018](https://github.com/kubernetes/kubernetes/pull/109018), [@wojtek-t](https://github.com/wojtek-t))
- Deprecated types in `k8s.io/apimachinery/util/clock`. Please use `k8s.io/utils/clock` instead. ([#106850](https://github.com/kubernetes/kubernetes/pull/106850), [@MadhavJivrajani](https://github.com/MadhavJivrajani))
- E2e tests wait for `kube-root-ca.crt` to be populated in namespaces for use with projected service account tokens, reducing delays starting those test pods and errors in the logs. ([#107763](https://github.com/kubernetes/kubernetes/pull/107763), [@smarterclayton](https://github.com/smarterclayton))
- Endpoints and EndpointSlice controllers no longer populate [resourceVersion of targetRef](https://kubernetes.io/docs/reference/kubernetes-api/common-definitions/object-reference/#ObjectReference) in Endpoints and EndpointSlices ([#108450](https://github.com/kubernetes/kubernetes/pull/108450), [@tnqn](https://github.com/tnqn))
- Fixed default config flags for `NewDefaultKubectlCommand`. ([#107131](https://github.com/kubernetes/kubernetes/pull/107131), [@jonnylangefeld](https://github.com/jonnylangefeld))
- Fixed documentation typo in cloud-provider. ([#106445](https://github.com/kubernetes/kubernetes/pull/106445), [@majst01](https://github.com/majst01))
- Fixed spelling of implemented in pkg/proxy/apis/config/types.go line 206 ([#106453](https://github.com/kubernetes/kubernetes/pull/106453), [@davidleitw](https://github.com/davidleitw))
- Improve error message when applying CRDs before the CRD exists in a cluster ([#107363](https://github.com/kubernetes/kubernetes/pull/107363), [@eddiezane](https://github.com/eddiezane))
- Improved algorithm for selecting `best` non-preferred hint in the TopologyManager ([#108154](https://github.com/kubernetes/kubernetes/pull/108154), [@klueska](https://github.com/klueska))
- Kube-proxy doesn't set the sysctl `net.ipv4.conf.all.route_localnet=1` if no IPv4 loopback address is selected by the `nodePortAddresses` configuration parameter. ([#107684](https://github.com/kubernetes/kubernetes/pull/107684), [@aojea](https://github.com/aojea))
- Kubeadm: all warning messages are printed to stderr instead of stdout. ([#107467](https://github.com/kubernetes/kubernetes/pull/107467), [@SataQiu](https://github.com/SataQiu))
- Kubeadm: handled the removal of dockershim related flags for new kubeadm clusters. If kubelet <1.24 is on the host, kubeadm >=1.24 can continue using the built-in dockershim in the kubelet if the user passes the `{Init|Join}Configuration.nodeRegistration.criSocket` value in the kubeadm configuration to be equal to `unix:///var/run/dockershim.sock` on Unix or `npipe:////./pipe/dockershim` on Windows. If kubelet version >=1.24 is on the host, kubeadm >=1.24 will treat all container runtimes as "remote" using the kubelet flags `--container-runtime=remote --container-runtime-endpoint=scheme://some/path`. The special management for kubelet <1.24 will be removed in kubeadm 1.25. ([#106973](https://github.com/kubernetes/kubernetes/pull/106973), [@neolit123](https://github.com/neolit123))
- Kubeadm: make sure that `kubeadm init/join` always use a URL scheme (unix:// on Linux and npipe:// on Windows) when passing a value to the `--container-runtime-endpoint` kubelet flag. This flag's value is taken from the kubeadm configuration `criSocket` field or the `--cri-socket` CLI flag. Automatically add a missing URL scheme to the user configuration in memory, but warn them that they should also update their configuration on disk manually. During `kubeadm upgrade apply/node` mutate the `/var/lib/kubelet/kubeadm-flags.env` file on disk and the `kubeadm.alpha.kubernetes.io/cri-socket` annotation Node object if needed. These automatic actions are temporary and will be removed in a future release. In the future the kubelet may not support CRI endpoints without an URL scheme. ([#107295](https://github.com/kubernetes/kubernetes/pull/107295), [@neolit123](https://github.com/neolit123))
- Kubeadm: remove the `IPv6DualStack` feature gate. The feature has been GA and locked to enabled since 1.23. ([#106648](https://github.com/kubernetes/kubernetes/pull/106648), [@calvin0327](https://github.com/calvin0327))
- Kubeadm: removed the deprecated `output/v1alpha1` API used for machine readable output by some kubeadm commands. In 1.23 kubeadm started using the newer version `output/v1alpha2` for the same purpose. ([#107468](https://github.com/kubernetes/kubernetes/pull/107468), [@neolit123](https://github.com/neolit123))
- Kubeadm: removed the restriction that the `ca.crt` can only contain one certificate. If there is more than one certificate in the `ca.crt` file, kubeadm will pick the first one by default. ([#107327](https://github.com/kubernetes/kubernetes/pull/107327), [@SataQiu](https://github.com/SataQiu))
- Kubectl stack traces now only print at verbose `-v=99` and not `-v=6` ([#108053](https://github.com/kubernetes/kubernetes/pull/108053), [@eddiezane](https://github.com/eddiezane))
- Kubectl: restored `--dry-run`, `--dry-run=true`, and `--dry-run=false` for compatibility with pre-1.23 invocations. ([#107003](https://github.com/kubernetes/kubernetes/pull/107003), [@julianvmodesto](https://github.com/julianvmodesto))
- Kubelet config validation error messages are updated. ([#105360](https://github.com/kubernetes/kubernetes/pull/105360), [@shuheiktgw](https://github.com/shuheiktgw))
- Kubernetes e2e framework will use the url `invalid.registry.k8s.io/invalid` instead `invalid.com/invalid` for test that use an invalid registry. ([#107455](https://github.com/kubernetes/kubernetes/pull/107455), [@aojea](https://github.com/aojea))
- Marked kubelet `--container-runtime-endpoint` and `--image-service-endpoint` CLI flags as stable. ([#106954](https://github.com/kubernetes/kubernetes/pull/106954), [@saschagrunert](https://github.com/saschagrunert))
- Migrate `volume/csi/csi-client.go` logs to structured logging. ([#99441](https://github.com/kubernetes/kubernetes/pull/99441), [@CKchen0726](https://github.com/CKchen0726))
- Migrate statefulset files to structured logging ([#106109](https://github.com/kubernetes/kubernetes/pull/106109), [@h4ghhh](https://github.com/h4ghhh))
- Refactor kubelet command line for enabling features and "drop `RuntimeClass` feature gate" if present. Note that this feature has been on by default since 1.14 and was GA'ed in 1.20. ([#106882](https://github.com/kubernetes/kubernetes/pull/106882), [@cyclinder](https://github.com/cyclinder))
- Remove deprecated `--serviceaccount`, `--hostport`, `--requests` and `--limits` from kubectl run. ([#108820](https://github.com/kubernetes/kubernetes/pull/108820), [@mozillazg](https://github.com/mozillazg))
- Remove support for `node-expansion` between `node-stage` and `node-publish` ([#108614](https://github.com/kubernetes/kubernetes/pull/108614), [@gnufied](https://github.com/gnufied))
- Removed deprecated `generator` and `container-port` flags ([#106824](https://github.com/kubernetes/kubernetes/pull/106824), [@lauchokyip](https://github.com/lauchokyip))
- Removed kubelet `--non-masquerade-cidr` deprecated CLI flag ([#107096](https://github.com/kubernetes/kubernetes/pull/107096), [@hakman](https://github.com/hakman))
- Rename unschedulableQ to unschedulablePods ([#108919](https://github.com/kubernetes/kubernetes/pull/108919), [@denkensk](https://github.com/denkensk))
- SPDY transport in client-go will no longer follow redirects. ([#108531](https://github.com/kubernetes/kubernetes/pull/108531), [@tallclair](https://github.com/tallclair))
- ServerResources was deprecated in February 2019 (https://github.com/kubernetes/kubernetes/commit/618050e) and now it's being removed and ServerGroupsAndResources is suggested to be used instead ([#107180](https://github.com/kubernetes/kubernetes/pull/107180), [@ardaguclu](https://github.com/ardaguclu))
- The API server's deprecated `--deserialization-cache-size` flag is now removed. ([#108448](https://github.com/kubernetes/kubernetes/pull/108448), [@ialidzhikov](https://github.com/ialidzhikov))
- The `--container-runtime` kubelet flag is deprecated and will be removed in future releases. ([#107094](https://github.com/kubernetes/kubernetes/pull/107094), [@adisky](https://github.com/adisky))
- The `WarningHeaders` feature gate that is GA since v1.22 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#108394](https://github.com/kubernetes/kubernetes/pull/108394), [@ialidzhikov](https://github.com/ialidzhikov))
- The `e2e.test` binary supports a new `--kubelet-root` parameter to override the default `/var/lib/kubelet` path. CSI storage tests use this. ([#108253](https://github.com/kubernetes/kubernetes/pull/108253), [@pohly](https://github.com/pohly))
- The fluentd-elasticsearch addon is no longer included in the cluster directory. It is available from https://github.com/kubernetes-sigs/instrumentation-addons/tree/master/fluentd-elasticsearch. ([#107553](https://github.com/kubernetes/kubernetes/pull/107553), [@liggitt](https://github.com/liggitt))
- The scheduler framework option `runAllFilters` is removed. ([#108829](https://github.com/kubernetes/kubernetes/pull/108829), [@kerthcet](https://github.com/kerthcet))
- Updated cri-tools to [v1.23.0](https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.23.0). ([#107604](https://github.com/kubernetes/kubernetes/pull/107604), [@saschagrunert](https://github.com/saschagrunert))
- Updated runc to 1.1.0 and updated cadvisor to 0.44.0 ([#109029](https://github.com/kubernetes/kubernetes/pull/109029), [@ehashman](https://github.com/ehashman))
- Updated runc to 1.1.1 ([#109104](https://github.com/kubernetes/kubernetes/pull/109104), [@kolyshkin](https://github.com/kolyshkin))
- Updated the error message to not use the `--max-resource-write-bytes` & `--json-patch-max-copy-bytes` string. ([#106875](https://github.com/kubernetes/kubernetes/pull/106875), [@warmchang](https://github.com/warmchang))
- Users who look at iptables dumps will see some changes in the naming and structure of rules. ([#109060](https://github.com/kubernetes/kubernetes/pull/109060), [@thockin](https://github.com/thockin))
- Windows Pause no longer has support for SAC releases 1903, 1909, 2004. Windows image support is now Ltcs 2019 (1809), 20H2, LTSC 2022 ([#107056](https://github.com/kubernetes/kubernetes/pull/107056), [@jsturtevant](https://github.com/jsturtevant))
- [k8s.io/utils/clock]: IntervalClock is now deprecated in favour of SimpleIntervalClock ([#108059](https://github.com/kubernetes/kubernetes/pull/108059), [@RaghavRoy145](https://github.com/RaghavRoy145))
- `kube-addon-manager` image version is bumped to 9.1.6 ([#108341](https://github.com/kubernetes/kubernetes/pull/108341), [@zshihang](https://github.com/zshihang))
- Add SourceVolumeMode field to VolumeSnapshotContents. Documentation for this alpha feature is pending. ([#665](https://github.com/kubernetes-csi/external-snapshotter/pull/665), [@RaunakShah](https://github.com/RaunakShah))
- Update snapshotter module to v6 and client module to v5. Documentation for this alpha feature is pending. ([#670],(https://github.com/kubernetes-csi/external-snapshotter/pull/670), [@RaunakShah](https://github.com/RaunakShah))

### Uncategorized

- Deprecate kubectl version long output, will be replaced with kubectl version ` --short`. Users requiring full output should use `--output=yaml|json` instead. ([#108987](https://github.com/kubernetes/kubernetes/pull/108987), [@soltysh](https://github.com/soltysh))


## Dependencies

### Added
- github.com/armon/go-socks5: [e753329](https://github.com/armon/go-socks5/tree/e753329)
- github.com/blang/semver/v4: [v4.0.0](https://github.com/blang/semver/v4/tree/v4.0.0)
- github.com/google/gnostic: [v0.5.7-v3refs](https://github.com/google/gnostic/tree/v0.5.7-v3refs)

### Changed
- github.com/cespare/xxhash/v2: [v2.1.1 → v2.1.2](https://github.com/cespare/xxhash/v2/compare/v2.1.1...v2.1.2)
- github.com/checkpoint-restore/go-criu/v5: [v5.0.0 → v5.3.0](https://github.com/checkpoint-restore/go-criu/v5/compare/v5.0.0...v5.3.0)
- github.com/cilium/ebpf: [v0.6.2 → v0.7.0](https://github.com/cilium/ebpf/compare/v0.6.2...v0.7.0)
- github.com/containerd/console: [v1.0.2 → v1.0.3](https://github.com/containerd/console/compare/v1.0.2...v1.0.3)
- github.com/containerd/containerd: [v1.4.11 → v1.4.12](https://github.com/containerd/containerd/compare/v1.4.11...v1.4.12)
- github.com/cpuguy83/go-md2man/v2: [v2.0.0 → v2.0.1](https://github.com/cpuguy83/go-md2man/v2/compare/v2.0.0...v2.0.1)
- github.com/cyphar/filepath-securejoin: [v0.2.2 → v0.2.3](https://github.com/cyphar/filepath-securejoin/compare/v0.2.2...v0.2.3)
- github.com/docker/distribution: [v2.7.1+incompatible → v2.8.1+incompatible](https://github.com/docker/distribution/compare/v2.7.1...v2.8.1)
- github.com/docker/docker: [v20.10.7+incompatible → v20.10.12+incompatible](https://github.com/docker/docker/compare/v20.10.7...v20.10.12)
- github.com/godbus/dbus/v5: [v5.0.4 → v5.0.6](https://github.com/godbus/dbus/v5/compare/v5.0.4...v5.0.6)
- github.com/golang/mock: [v1.5.0 → v1.6.0](https://github.com/golang/mock/compare/v1.5.0...v1.6.0)
- github.com/google/cadvisor: [v0.43.0 → v0.44.1](https://github.com/google/cadvisor/compare/v0.43.0...v0.44.1)
- github.com/google/cel-go: [v0.9.0 → v0.10.1](https://github.com/google/cel-go/compare/v0.9.0...v0.10.1)
- github.com/moby/sys/mountinfo: [v0.4.1 → v0.6.0](https://github.com/moby/sys/mountinfo/compare/v0.4.1...v0.6.0)
- github.com/moby/term: [9d4ed18 → 3f7ff69](https://github.com/moby/term/compare/9d4ed18...3f7ff69)
- github.com/opencontainers/image-spec: [v1.0.1 → v1.0.2](https://github.com/opencontainers/image-spec/compare/v1.0.1...v1.0.2)
- github.com/opencontainers/runc: [v1.0.2 → v1.1.1](https://github.com/opencontainers/runc/compare/v1.0.2...v1.1.1)
- github.com/opencontainers/selinux: [v1.8.2 → v1.10.0](https://github.com/opencontainers/selinux/compare/v1.8.2...v1.10.0)
- github.com/prometheus/client_golang: [v1.11.0 → v1.12.1](https://github.com/prometheus/client_golang/compare/v1.11.0...v1.12.1)
- github.com/prometheus/common: [v0.28.0 → v0.32.1](https://github.com/prometheus/common/compare/v0.28.0...v0.32.1)
- github.com/prometheus/procfs: [v0.6.0 → v0.7.3](https://github.com/prometheus/procfs/compare/v0.6.0...v0.7.3)
- github.com/russross/blackfriday/v2: [v2.0.1 → v2.1.0](https://github.com/russross/blackfriday/v2/compare/v2.0.1...v2.1.0)
- github.com/seccomp/libseccomp-golang: [v0.9.1 → 3879420](https://github.com/seccomp/libseccomp-golang/compare/v0.9.1...3879420)
- github.com/spf13/cobra: [v1.2.1 → v1.4.0](https://github.com/spf13/cobra/compare/v1.2.1...v1.4.0)
- github.com/yuin/goldmark: [v1.4.0 → v1.4.1](https://github.com/yuin/goldmark/compare/v1.4.0...v1.4.1)
- go.etcd.io/etcd/api/v3: v3.5.0 → v3.5.1
- go.etcd.io/etcd/client/pkg/v3: v3.5.0 → v3.5.1
- go.etcd.io/etcd/client/v3: v3.5.0 → v3.5.1
- golang.org/x/crypto: 32db794 → 8634188
- golang.org/x/mod: v0.4.2 → 9b9b3d8
- golang.org/x/net: e898025 → cd36cc0
- golang.org/x/oauth2: 2bc19b1 → d3ed0bb
- golang.org/x/sys: f4d4317 → 3681064
- golang.org/x/term: 6886f2d → 03fcf44
- golang.org/x/time: 1f47c86 → 90d013b
- golang.org/x/tools: d4cc65f → 897bd77
- google.golang.org/genproto: fe13028 → 42d7afd
- k8s.io/gengo: 485abfe → c02415c
- k8s.io/klog/v2: v2.30.0 → v2.60.1
- k8s.io/kube-openapi: e816edb → 3ee0da9
- k8s.io/system-validators: v1.6.0 → v1.7.0
- k8s.io/utils: cb0fa31 → 3a6ce19
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.25 → v0.0.30
- sigs.k8s.io/json: c049b76 → 9f7c6b3
- sigs.k8s.io/kustomize/api: v0.10.1 → v0.11.4
- sigs.k8s.io/kustomize/cmd/config: v0.10.2 → v0.10.6
- sigs.k8s.io/kustomize/kustomize/v4: v4.4.1 → v4.5.4
- sigs.k8s.io/kustomize/kyaml: v0.13.0 → v0.13.6
- sigs.k8s.io/structured-merge-diff/v4: v4.1.2 → v4.2.1

### Removed
- cloud.google.com/go/firestore: v1.1.0
- github.com/armon/go-metrics: [f0300d1](https://github.com/armon/go-metrics/tree/f0300d1)
- github.com/armon/go-radix: [7fddfc3](https://github.com/armon/go-radix/tree/7fddfc3)
- github.com/bgentry/speakeasy: [v0.1.0](https://github.com/bgentry/speakeasy/tree/v0.1.0)
- github.com/bits-and-blooms/bitset: [v1.2.0](https://github.com/bits-and-blooms/bitset/tree/v1.2.0)
- github.com/bketelsen/crypt: [v0.0.4](https://github.com/bketelsen/crypt/tree/v0.0.4)
- github.com/containernetworking/cni: [v0.8.1](https://github.com/containernetworking/cni/tree/v0.8.1)
- github.com/fatih/color: [v1.7.0](https://github.com/fatih/color/tree/v1.7.0)
- github.com/googleapis/gnostic: [v0.5.5](https://github.com/googleapis/gnostic/tree/v0.5.5)
- github.com/hashicorp/consul/api: [v1.1.0](https://github.com/hashicorp/consul/api/tree/v1.1.0)
- github.com/hashicorp/consul/sdk: [v0.1.1](https://github.com/hashicorp/consul/sdk/tree/v0.1.1)
- github.com/hashicorp/errwrap: [v1.0.0](https://github.com/hashicorp/errwrap/tree/v1.0.0)
- github.com/hashicorp/go-cleanhttp: [v0.5.1](https://github.com/hashicorp/go-cleanhttp/tree/v0.5.1)
- github.com/hashicorp/go-immutable-radix: [v1.0.0](https://github.com/hashicorp/go-immutable-radix/tree/v1.0.0)
- github.com/hashicorp/go-msgpack: [v0.5.3](https://github.com/hashicorp/go-msgpack/tree/v0.5.3)
- github.com/hashicorp/go-multierror: [v1.0.0](https://github.com/hashicorp/go-multierror/tree/v1.0.0)
- github.com/hashicorp/go-rootcerts: [v1.0.0](https://github.com/hashicorp/go-rootcerts/tree/v1.0.0)
- github.com/hashicorp/go-sockaddr: [v1.0.0](https://github.com/hashicorp/go-sockaddr/tree/v1.0.0)
- github.com/hashicorp/go-syslog: [v1.0.0](https://github.com/hashicorp/go-syslog/tree/v1.0.0)
- github.com/hashicorp/go-uuid: [v1.0.1](https://github.com/hashicorp/go-uuid/tree/v1.0.1)
- github.com/hashicorp/go.net: [v0.0.1](https://github.com/hashicorp/go.net/tree/v0.0.1)
- github.com/hashicorp/golang-lru: [v0.5.0](https://github.com/hashicorp/golang-lru/tree/v0.5.0)
- github.com/hashicorp/hcl: [v1.0.0](https://github.com/hashicorp/hcl/tree/v1.0.0)
- github.com/hashicorp/logutils: [v1.0.0](https://github.com/hashicorp/logutils/tree/v1.0.0)
- github.com/hashicorp/mdns: [v1.0.0](https://github.com/hashicorp/mdns/tree/v1.0.0)
- github.com/hashicorp/memberlist: [v0.1.3](https://github.com/hashicorp/memberlist/tree/v0.1.3)
- github.com/hashicorp/serf: [v0.8.2](https://github.com/hashicorp/serf/tree/v0.8.2)
- github.com/magiconair/properties: [v1.8.5](https://github.com/magiconair/properties/tree/v1.8.5)
- github.com/mattn/go-colorable: [v0.0.9](https://github.com/mattn/go-colorable/tree/v0.0.9)
- github.com/mattn/go-isatty: [v0.0.3](https://github.com/mattn/go-isatty/tree/v0.0.3)
- github.com/miekg/dns: [v1.0.14](https://github.com/miekg/dns/tree/v1.0.14)
- github.com/mitchellh/cli: [v1.0.0](https://github.com/mitchellh/cli/tree/v1.0.0)
- github.com/mitchellh/go-homedir: [v1.0.0](https://github.com/mitchellh/go-homedir/tree/v1.0.0)
- github.com/mitchellh/go-testing-interface: [v1.0.0](https://github.com/mitchellh/go-testing-interface/tree/v1.0.0)
- github.com/mitchellh/gox: [v0.4.0](https://github.com/mitchellh/gox/tree/v0.4.0)
- github.com/mitchellh/iochan: [v1.0.0](https://github.com/mitchellh/iochan/tree/v1.0.0)
- github.com/pascaldekloe/goe: [57f6aae](https://github.com/pascaldekloe/goe/tree/57f6aae)
- github.com/pelletier/go-toml: [v1.9.3](https://github.com/pelletier/go-toml/tree/v1.9.3)
- github.com/posener/complete: [v1.1.1](https://github.com/posener/complete/tree/v1.1.1)
- github.com/ryanuber/columnize: [9b3edd6](https://github.com/ryanuber/columnize/tree/9b3edd6)
- github.com/sean-/seed: [e2103e2](https://github.com/sean-/seed/tree/e2103e2)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/spf13/cast: [v1.3.1](https://github.com/spf13/cast/tree/v1.3.1)
- github.com/spf13/jwalterweatherman: [v1.1.0](https://github.com/spf13/jwalterweatherman/tree/v1.1.0)
- github.com/spf13/viper: [v1.8.1](https://github.com/spf13/viper/tree/v1.8.1)
- github.com/subosito/gotenv: [v1.2.0](https://github.com/subosito/gotenv/tree/v1.2.0)
- gopkg.in/ini.v1: v1.62.0



# v1.24.0-rc.1


## Downloads for v1.24.0-rc.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes.tar.gz) | 7915e218b651f9fd7cfa7ab2125e2e72eb92844dc2ab2c1a6c3068dd0db298eea8c7c209a9c97a4b8c55bc34f7365032db2f67bfd873e298afec51083572e39d
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-src.tar.gz) | 14497e555b3366560d179fc09c5b8ff57b04573902bb0e1e760e6cf280aab82d6f4b04fcb5c0d400b7425646077d40a7654faff21b44c48c98ceab08d0e7afe0

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | 52fba6e9b2ff27b673485661e0c93d163b3317dc034990fc7460cd301ceb28a453616fc5ce8bba6dc9f8031d4f0966976bebaedf96243bcafa5db0d8ba4f8e70
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-darwin-arm64.tar.gz) | 4fc2807ada2aa2987f6b84520ca5febbf78d65bea6b58851fdddbf417a59539a0276bcea157e1ae81be0fb87832fea8dbc78b6fdb07fda71920aacddb3705e67
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-linux-386.tar.gz) | 347f98f20c5a835aa32cc47960b217ad9b7e4b4254d2479769ae22dc2491b030bc8c1ec406ab502311e978fcd6bb5f53edc1ff8b41b6e0faab9b7cb435d49ece
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | f860dfc209864639791f2133752209ca08f4f7c7d8f65c031c5668160cedfc32a0729e105bf85462dcd8e809c637ce250f433abc3159effcd89e2e20252c1a97
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-linux-arm.tar.gz) | f2a819b9e4e35a8741977c41488f8ea75e135d6ea664325259c0e01766c8f2c147682f20f2ba5462de89d3162120babe9c7d2c10e14b0d1fc1e63ad58f784a6d
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | bf28d10570c1cdc7d7901c0a9d5df7258711df39771e1bd2a267ef4dfe54fe9310746492ce49652a9bed92831c99532edf08784316cbdf76c17f2c6a0851c780
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | 3fb29a2a4e0deb5c2e8660ea1f0097e5fc89f283be1f7159b4de80fabc3d76a458add2b50d76e5c432318e44b83f3d7b8fa2527c4f74595bc30031ad6deba431
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | 157f07f9e410cd2f996ede0b2ce4357b6a55e11f2307401618a855f318bab3d805c6af40d35367dc6b458611debcf011aba883f19dd6fcbff0f46d5e7d8a038b
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-windows-386.tar.gz) | 9199f7ec5780b6110d0f6209f676c18ab1b38d0bfb8e39e93262b5bbb7e72f6fa4235d55a19e455fdf625dbe3c5687004d002969d1981f209d433d0ba41fda21
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | 3bc4af50477bebb7d58a82bf1181ad3b5551acae46f5300c7c4c2c434f9a6118d41c346bbe1fd17dc9cdf47a9a042e33f7fab4721c94e95fea847f814988b575
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-client-windows-arm64.tar.gz) | de4af15c041f6b4e3f9841e8a95fd0497ae1e8717fcea26f6015fd7881fabfc4437941713d3c68cfc55d9bd88d6f98cb7e29037fab864d8effe9f3123e12a6d3

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | 9e3bd58bab7202e54dfeadfd7c8289cf5aef82dcb4e045db4a0a5dce77f47a9362170cf92372c938ba60b51256c33cb409a44ede25b61f076939e1a29738f859
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-server-linux-arm.tar.gz) | 4ab694514d281baf47a61ca76ab08aa8f840d521f6e8d54a842c6692bab7e83e898487bd5cf6bf4aef1589cbd4ad9bf37117d32cdffa5a83a5463b4b3f4d82d6
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | a20f0950ef6e61d5ca93e0c2c12c09c474de6160be7655f0739e28338872c22814543e6d8f9406e9b3494afcc085273a66d624e9526352895d59d50995e56d21
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | 212916a0ae56b4f9c48b57d2deeaf4582391cef468961f552296261680098779d7cebf9aaf25bfa68a75f97aee45a1d46cb820f300982cbf9ac0d5aff11ad156
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | 046639871d5a720fdf03365b9da9af934e0b18ca45afe092887d5d84a5bf925850baeafcf437537ab6ac66bdc1a87bb7fd9f82b7649e664c6f6bd9c38114a552

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | e1460cb62d5794b8e736de0c30ef86b7b0aa3c544d2603f3395f8eaffc7718c4c47881c217a515b218ae69a8af52fbfa1699b233634309f96f7f997e35b43244
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-node-linux-arm.tar.gz) | b662a8be7e9c7545cba7e7cfa689474b544382f8df86a10929234c86d32be9ca843eb01dd35faec367172afb828dc6ac3044483c5852c84ffd5281d86e7fc11f
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | 5e890f3ffddc473f8f20afe9fa8747ef091a60bf4ccddd4d87f94e99c51374bffaf976cab2bc71deb6d8142b967d1cfef50fe96c100d2b4aaecb86d903189311
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | 12f2f16b57abfdaea2c35652c68d60833150a835f11d50a787898d7f178b1bf258c26f14414bbb165e3782e30687e0f373ae82d85e888c701a53d66916a6d432
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | 5d15a3227aed35d3aa5c7292b44e482ea9c66311d2fa9473c618a64a7c4ebac3ef285b85f2db57e0d1575152dd310efac28b895c7f24191ee297c8cba98c9e65
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | c09d31a38358fc6e74f6ee77d31e91ebb0f7ec2f8251e4a4c805a7312561d52c30edc68344ac618b73fb3e0326886b639200d52dfd94b4459d8ff9560b9c8a29

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0-rc.0

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.24.0-rc.0


## Downloads for v1.24.0-rc.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes.tar.gz) | 315c7878ec565acc7a6d8a36af3b435d9862bc33e79f5c8f9a09d41e2f3b754e1cb4555095609929646ec24bf3dc4ed16bdc3ec68bd1f5be8ed40abf4b88ddc7
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-src.tar.gz) | 8cb327285c7bdf09173a40ddcdb6396c3165147c8e789dd693542656e7bce481d2bf13672d4764e364f6bc5bdb9441f2b680f2a7664b924194e56813323f0c37

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | be1715cfe2f364a16b6e066647245c825f68a4cc913c793485fa6f23722e4b8097e594b2ea6ab771ccaf045db14206a0322f6f98321e6cd95aeeaa7f79f80cc1
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | 52c1327f59b57cc8dbab85c1bc28c63224a7f3ade3b9d9d49896879b933b835a3a715c474993f26b7deba758b606b28633bf350c03abacae3925d84e43d07aac
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-linux-386.tar.gz) | 7618c924b36d40354125c6cc1105ed65683f1d3e8595146c1299f42705bca8cbc4c5b3d371ad1f7441af3eb8ee091ba1d5019dcde8f3008776541c8d321f64c6
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | 494cbd874f549afd489c589e2d06d9f14dfff74c139bbf30e085f0951cc42c1bce526072b0c3e0b8728809568f9d73afd7b18f9d6b6d4a25503015b23bf14296
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-linux-arm.tar.gz) | 38d308d5854e4dadba617850e3af4fa3be4337baa6d7ba8a73158d154899f1fdeea577d8248b4bcb21896f04475b5a230957a0c39cf3b917f288449b6a6a2d00
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | 8c7c1de2a791472466db1abb5540b5cd4a9d87c8e03278c0be5c38ff02eca36e8c1672ce622dacf3aa392fd7b4ed79277f8eca2d77ebbda986ee2f7e49453ff4
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | ab1166538fadf54943425309ab5b158919f43211dad075b5b2f423706caab7e5e91df0fa1cb8214d40288c3a6adb90cf9b2b104f26f69c98195a63f743d5c055
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | b15d7f4890c6526c2a1927e9030847561a11348ec84a40589c07676c64a9fcbbd62cd5d87f1a3957207eefafff450564d115c4f05c020419d4bab7b0d5798798
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-windows-386.tar.gz) | a8114d927cf57d568c9f4e945df6413b5a38e650fe44e9f61031ac0ded376ed35432622cab4212069c4025a4dead7eb8d3f91a475f9791129e818302df8085a1
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | 85031db6f486c5585ef75d96b2458aa3023242f3d0a2bc0d82d9d28342fd7dfc27efda902eb3ff655dd819df87eb95461d504e1ecf80def09629623b08bbab0c
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | 0550650fcfd9efb7a28d3683e8036084da38119f35f7718c75fcc8a5821b8db1a5ae42b026a06caa9f89b303c7d4bc06ca81ca1868d2f178f9541fc8ceb3dfe1

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | cf657293f200f07a653202c3b1825fab03ceed13fa3dcda5b03c9f6b1e71a977eee4b8ce0cebd3f257da17c3362802659975b9c583ae4271ca5df003aef616c3
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-server-linux-arm.tar.gz) | 8d69b6543858995471521d4a72837d58f498b871581cefb9e79a28317a65c2263be9b05fa5f54fa9044cc017cbbb2e5c650a6aa199be531ad0b822c30ba46b30
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | 48faa785514ed0c141a9faa4e8e392227bbfbdc5cf19e7539c0061faf181e4c0b68c9afdb40d1bca3c5d3a92d877060f777956f837cfe540219a8807fb4624cb
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | a711bea03b751129f79e1f93e5eae2e5e41ade170e19172d396a598e62bb6f459cdcd4585a14e855cee9eb42b200c4283b88fa99825dc3e0c9bb3b01de773bf8
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | 78ba05c54ef5a33614cd8ae51a25073cc8584fdbd966220803d76b2232af9c58eaf69c4c2f83bc9d8c5ddb747b33d17e7820e9b60957b5cc51f470b19dcd4f88

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | 1f7118bc1596dbcca0e88588908092975d01fcd33e804dd168ea2b494f2fcbd90f65a332192713a6012fe506619e14a0dcb22207074e264c63f7f4ceea20e78a
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-node-linux-arm.tar.gz) | cf5bc666aa406a2f260594b6643cb5cc5555794e51f0a05fb467ea0ae47ae6d502c5d74696568179622e78043107bd703666344b48244f75eb1414d9f4876126
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | efdba80a901ef623274ac4a1f80932675be40f87603b0baea77e0d1a040bd18bafc30c5326080c24479db4b7f694db5841bab761ac77adc530087ef9fdb6aa17
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | a3254d5b500e8dd34003241c875af79ac118a348d5dfe7f4007927e2c01a8925e8e89ed6a9a0e08c4d3c364b5ede67f378a1531cddabfd3f12e876c3ff4c3263
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | 063edc211d87c0e7811e20d22e8b2413137914b4bd4c635460f8d9aa31ad7c6acd411cb8a6b0c374d81abd0a82b8066522266c6a5a591fbc6e9f767455a24be4
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | 214954da8586b452ba1b819cca24b39acdc38c9209d023d97d5a26031ae7de9091487e965f4a46722f3913093c44b6de521e11aa536e66409e1b48131890148c

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0-beta.0

## Changes by Kind

### API Change

- Introduce a v1alpha1 networking API for ClusterCIDRConfig ([#108290](https://github.com/kubernetes/kubernetes/pull/108290), [@sarveshr7](https://github.com/sarveshr7)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Instrumentation, Network and Testing]
- Introduction of a new "sync_proxy_rules_no_local_endpoints_total" proxy metric. This metric represents the number of services with no internal endpoints. The "traffic_policy" label will contain both "internal" or "external". ([#108930](https://github.com/kubernetes/kubernetes/pull/108930), [@MaxRenaud](https://github.com/MaxRenaud)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Instrumentation, Network, Node, Release, Scheduling, Storage, Testing and Windows]
- Make STS available replicas optional again, ([#109241](https://github.com/kubernetes/kubernetes/pull/109241), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG API Machinery and Apps]
- Omit enum declarations from the static openapi file captured at https://git.k8s.io/kubernetes/api/openapi-spec. This file is used to generate API clients, and use of enums in those generated clients (rather than strings) can break forward compatibility with additional future values in those fields. See https://issue.k8s.io/109177 for details. ([#109178](https://github.com/kubernetes/kubernetes/pull/109178), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Auth]
- Remove a v1alpha1 networking API for ClusterCIDRConfig ([#109436](https://github.com/kubernetes/kubernetes/pull/109436), [@JamesLaverack](https://github.com/JamesLaverack)) [SIG API Machinery, Apps, Auth, CLI, Network and Testing]
- The deprecated kube-controller-manager flag '--deployment-controller-sync-period' has been removed, it is not used by the deployment controller. ([#107178](https://github.com/kubernetes/kubernetes/pull/107178), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery and Apps]

### Feature

- Kubernetes is now built with Golang 1.18.1 ([#109461](https://github.com/kubernetes/kubernetes/pull/109461), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Moving MixedProtocolLBService from alpha to beta ([#109213](https://github.com/kubernetes/kubernetes/pull/109213), [@bridgetkromhout](https://github.com/bridgetkromhout)) [SIG Network]
- The `v1` version of `LeaderMigrationConfiguration` supports only `leases` API for leader election. To use formerly supported mechanisms, please continue using `v1beta1`. ([#108016](https://github.com/kubernetes/kubernetes/pull/108016), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery and Cloud Provider]

### Bug or Regression

- Adds PV deletion protection finalizer only when PV reclaimPolicy is Delete for dynamically provisioned volumes. ([#109205](https://github.com/kubernetes/kubernetes/pull/109205), [@deepakkinni](https://github.com/deepakkinni)) [SIG Apps and Storage]
- Correct event registration for multiple scheduler plugins; this fixes a potential significant delay in re-queueing unschedulable pods. ([#109442](https://github.com/kubernetes/kubernetes/pull/109442), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling and Testing]
- Etcd: Update to v3.5.3 ([#109471](https://github.com/kubernetes/kubernetes/pull/109471), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Fix the bug that the outdated services may be sent to the cloud provider ([#107631](https://github.com/kubernetes/kubernetes/pull/107631), [@lzhecheng](https://github.com/lzhecheng)) [SIG Cloud Provider and Network]
- Fix the overestimated cost of delegated API requests in kube-apiserver API priority&fairness ([#109188](https://github.com/kubernetes/kubernetes/pull/109188), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Fixed CSI migration of Azure Disk in-tree StorageClasses with topology requirements in Azure regions that do not have availability zones. ([#109154](https://github.com/kubernetes/kubernetes/pull/109154), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Prevent kube-scheduler from nominating a Pod that was already scheduled to a node ([#109245](https://github.com/kubernetes/kubernetes/pull/109245), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Sets JobTrackingWithFinalizers, beta feature, as disabled by default, due to unresolved bug https://github.com/kubernetes/kubernetes/issues/109485 ([#109487](https://github.com/kubernetes/kubernetes/pull/109487), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Testing]
- The `ServerSideFieldValidation` feature has been reverted to alpha for 1.24. ([#109271](https://github.com/kubernetes/kubernetes/pull/109271), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, CLI and Testing]

### Other (Cleanup or Flake)

- Client-go: if resetting the body fails before a retry, an error is now surfaced to the user. ([#109050](https://github.com/kubernetes/kubernetes/pull/109050), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG API Machinery]
- Users who look at iptables dumps will see some changes in the naming and structure of rules. ([#109060](https://github.com/kubernetes/kubernetes/pull/109060), [@thockin](https://github.com/thockin)) [SIG Network and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.24.0-beta.0


## Downloads for v1.24.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes.tar.gz) | fee35c2c970d740f4d1cd06ab8f661a025d03639e30ea9d88c711a6e5292396499fd57519297669e6643a56a80ae5770786f7bea105b5c5d5fc5b7835fa00a3b
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-src.tar.gz) | 79a8ebfe8d822e8e4b5fa888d37b078ac8b19641692364058a274dc63a3dd0f7fc6ad2dcbca72c4e8bf72fead3a89e0feb207dc7459d0e9cb6b28e1cd4b2e532

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | ee6f396270db71e7a74545a2868705985a59edc400951f4c368e4471f152b1dd2456de26dcdc187d066e1e0747ddbcf9e4eb4737c5f03dc9b38ddf48d2c15aa4
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | dd08e3148f184410a865356754b1b1be21afecb3b671aff52b2b7d037da22cde8ec4cc5e53c01f418e281f00e8c978bb22c59a1c03c5ed79505824d9c00c29ab
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-linux-386.tar.gz) | bc10a3b0b81a0dde4fb1c1d9dfe5c6ec81122c71bd215b1da629a94993fed2e55e8367f7e916557a69bcba08cccec2301500ebf62e6dbb34b2c200cac545ce73
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | f8ce5bd528b0d31b98164d03b2d1906dce61747e9d64632064000fe872384d6abca66b3dfe8023ec492ae8b2589d8d79db06be56951e48ff52cc9f9871035408
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-linux-arm.tar.gz) | 044005bed2812d80bd90bcd5bf015082af54f4d02218f137d304d0d2b2933bac904bdc5d63695d786e9bb04d4115157ea28be676c2da064c48ca16fe803fb6bd
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | 886604d1f21486fb78a8053c43e58c5753a058d6f67c554c072cf8d6c061cd007a1648bb013956901a913a80d90c336c0990b1d9bf810663ad0b1dd03f4855d9
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | f6c84131ee7a6f1561b2000073841387846d1c457d489d2d6f86dbafdeba2b131ee7bec460e321b8857ea60fb5706b02988f612d40c3d73638410c46db1e2976
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | a6fce005a85893bb9618e8e7f51f052625b42098fe30561b32031edf2f5b815ac951249651ede1c74cf8ce7585baf15f37b651b6ef7d84ffb77b0f85736803a6
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-windows-386.tar.gz) | 62aaf0a58ca5182a22e4eae4fcb4f69a8bc415d28a54dce6386784cf396ffa61f95f9c0232d876b7b81a49138c6fdf0ed95e4d817169fff4f20b463e40cbea7d
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | 658bf802631693d14fcf25546af4e1d7040e535d4915d6fde3e6ba84e404221142a26b41b2c91579374340317c6b9545da67d40170d099797d4359e6899209f7
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | 7f7ee26073c736ca0ffe09f169d51169ecb753eecc4969d369460d7b867ec1a0f17a2606b908d457d5f6c56befc82fec56dc7702d0a59fcb7c76d4bb680bfa83

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | 579e5574f0dc7ac8f2c628474dddf2c61146a696c12344bebeb7dd2ba615f81b08c4710a61ada7151f7146ce0e7c86e2a3c4aa5a3ae5cf8e701289a23faa8e30
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-server-linux-arm.tar.gz) | 22d3da15193ad6fa9ad4261ab7eba6465a6b7bf444bc30b2aad71c3e40da2b3540c44d27556b8eb1ec94782fcaeec9bb0c6f0aeea8494ccefa9640ca7b750599
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | 15d643d90c0d64303946a7493ac48f8331933f59418b4dc0e2bd7ebbf3f7979f8f1915b82e05eb8d070dbb88581c4a8e7e11f06128c0305b591178c6db735ad3
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | e89646d4056cd82fb0b84b3e44dc93878f92800113220d2b4354c37d99ef82b4b62eb370406ef503565031b9370a135ee97931673950b6f7b94b8bedc4e068bb
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | 321a31f3a8e4aea274943d5cfb0fd41dd40e785cf7d90efd80e4fe9dc45f8e1138685a0637538f7c115954500730e6ef0f03c0aaada982b1474aea5c91f91a50

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 26c94e60a56ae3c56f3882745cb2cab76d1ffdd9a5a7539491e181431040f5a471e31bf30b496c7bb972bc9df1d01d92ae1d1e82e4012fca8dbb528b81e08817
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-node-linux-arm.tar.gz) | 7112ff9f692f8f8e65211d86ce35581c2a48cd849982112018d6f9d8bea1cd49211333571de0958a86b7b14940b9704e724765116ae26c157283b6506502c0f2
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | 58df66e532c1a3f668cf414bd11da72926cc95e28491528b4f618710b16e577d557d89f43a3c4891ca53ec415ed13b5695d9701ae3694d18437eea66603a2e29
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 0a58004da31b1407e2299a74664d94500fd2afe726dd76713f0bfbd36713621d23ac450b8c3cee4528a16e32b3b78c9b269b3d213045d06e0dd4746fc3404e82
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | ac5895babb4b19e301c65bcf03913408ee887f217efe47710bbed9e885e4191171e182fc394610f5d7bb98e4cf2adcdc6246d5053b6e8902eb287f7be64a5e55
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | ee512b1144c33bd8b503ce6de43fe22543988cfd7665d2048588c7367bdacda31761fe4c93b3c1c13a6e3e78286b31560410e1a5fe93bcc852853a1af4bb48e0

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0-alpha.4

## Changes by Kind

### Deprecation

- Remove insecure serving configuration from cloud-provider package, which is consumed by cloud-controller-managers. ([#108953](https://github.com/kubernetes/kubernetes/pull/108953), [@nckturner](https://github.com/nckturner)) [SIG Cloud Provider and Testing]
- The metadata.clusterName field is deprecated. This field has always been unwritable and always blank, but its presence is confusing, so we will remove it next release. Out of an abundance of caution, this release we have merely changed the name in the go struct to ensure any accidental client uses are found before complete removal. ([#108717](https://github.com/kubernetes/kubernetes/pull/108717), [@lavalamp](https://github.com/lavalamp)) [SIG API Machinery, Apps, Auth, Scheduling and Testing]
- VSphere releases less than 7.0u2 are deprecated as of v1.24. Please consider upgrading vSphere to 7.0u2 or above. vSphere CSI Driver requires minimum vSphere 7.0u2.
  
  General Support for vSphere 6.7 will end on October 15, 2022. vSphere 6.7 Update 3 is deprecated in Kubernetes v1.24.  Customers are recommended to upgrade vSphere (both ESXi and vCenter) to 7.0u2 or above.  vSphere CSI Driver 2.2.3 and higher supports CSI Migration.
  
  Support for these deprecations will be available till October 15, 2022. ([#109089](https://github.com/kubernetes/kubernetes/pull/109089), [@deepakkinni](https://github.com/deepakkinni)) [SIG Cloud Provider]

### API Change

- Adds a new Status subresource in Network Policy objects ([#107963](https://github.com/kubernetes/kubernetes/pull/107963), [@rikatz](https://github.com/rikatz)) [SIG API Machinery, Apps, Network and Testing]
- Adds support for "InterfaceNamePrefix" and "BridgeInterface" as arguments to --detect-local-mode option and also introduces a new optional `--pod-interface-name-prefix` and `--pod-bridge-interface` flags to kube-proxy. ([#95400](https://github.com/kubernetes/kubernetes/pull/95400), [@tssurya](https://github.com/tssurya)) [SIG API Machinery and Network]
- CEL CRD validation expressions may now reference existing object state using the identifier `oldSelf`. ([#108073](https://github.com/kubernetes/kubernetes/pull/108073), [@benluddy](https://github.com/benluddy)) [SIG API Machinery and Testing]
- CSIStorageCapacity.storage.k8s.io: The v1beta1 version of this API is deprecated in favor of v1, and will be removed in v1.27. If a CSI driver supports storage capacity tracking, then it must get deployed with a release of external-provisioner that supports the v1 API. ([#108445](https://github.com/kubernetes/kubernetes/pull/108445), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, Scheduling, Storage and Testing]
- Custom resource requests with fieldValidation=Strict consistently require apiVersion and kind, matching non-strict requests ([#109019](https://github.com/kubernetes/kubernetes/pull/109019), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Improve kubectl's user help commands readability ([#104736](https://github.com/kubernetes/kubernetes/pull/104736), [@lauchokyip](https://github.com/lauchokyip)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Contributor Experience, Instrumentation, Network, Node, Release, Scalability, Scheduling, Security, Storage, Testing and Windows]
- Indexed Jobs graduates to stable ([#107395](https://github.com/kubernetes/kubernetes/pull/107395), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps, Architecture and Testing]
- Introduce a v1alpha1 networking API for ClusterCIDRConfig ([#108290](https://github.com/kubernetes/kubernetes/pull/108290), [@sarveshr7](https://github.com/sarveshr7)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Instrumentation, Network and Testing]
- JobReadyPods graduates to Beta and it's enabled by default. ([#107476](https://github.com/kubernetes/kubernetes/pull/107476), [@alculquicondor](https://github.com/alculquicondor)) [SIG API Machinery, Apps and Testing]
- Kubelet external Credential Provider feature is moved to Beta.  Credential Provider Plugin and Credential Provider Config API's updated from v1alpha1 to v1beta1 with no API changes. ([#108847](https://github.com/kubernetes/kubernetes/pull/108847), [@adisky](https://github.com/adisky)) [SIG API Machinery and Node]
- MaxUnavailable for StatefulSets, allows faster RollingUpdate by taking down more than 1 pod at a time. The number of pods you want to take down during a RollingUpdate is configurable using maxUnavailable parameter. ([#82162](https://github.com/kubernetes/kubernetes/pull/82162), [@krmayankk](https://github.com/krmayankk)) [SIG API Machinery and Apps]
- Non graceful node shutdown handling. ([#108486](https://github.com/kubernetes/kubernetes/pull/108486), [@sonasingh46](https://github.com/sonasingh46)) [SIG Apps, Node and Storage]
- OpenAPI V3 is turned on by default ([#109031](https://github.com/kubernetes/kubernetes/pull/109031), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Promote IdentifyPodOS feature to beta. ([#107859](https://github.com/kubernetes/kubernetes/pull/107859), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG API Machinery, Apps, Node, Testing and Windows]
- Skip x-kubernetes-validations rules if having fundamental error against OpenAPIv3 schema. ([#108859](https://github.com/kubernetes/kubernetes/pull/108859), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]
- Support for gRPC probes is now in beta. GRPCContainerProbe feature gate is enabled by default. ([#108522](https://github.com/kubernetes/kubernetes/pull/108522), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Apps, Node and Testing]
- The AnyVolumeDataSource feature is now beta, and the feature gate is enabled by default. In order to provide user feedback on PVCs with data sources, deployers must install the VolumePopulators CRD and the data-source-validator controller. ([#108736](https://github.com/kubernetes/kubernetes/pull/108736), [@bswartz](https://github.com/bswartz)) [SIG Apps, Storage and Testing]
- The `ServerSideFieldValidation` feature has graduated to beta and is now enabled by default. Kubectl 1.24 and newer will use server-side validation instead of client-side validation when writing to API servers with the feature enabled. ([#108889](https://github.com/kubernetes/kubernetes/pull/108889), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery, Architecture, CLI and Testing]
- The infrastructure for contextual logging is complete (feature gate implemented, JSON backend ready). ([#108995](https://github.com/kubernetes/kubernetes/pull/108995), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Testing]
- This adds an optional `timeZone` field as part of the CronJob spec to support running cron jobs in a specific time zone. ([#108032](https://github.com/kubernetes/kubernetes/pull/108032), [@deejross](https://github.com/deejross)) [SIG API Machinery and Apps]

### Feature

- **Additional documentation e.g., KEPs (Kubernetes Enhancement Proposals), usage docs, etc.**:
  
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
  --> ([#109024](https://github.com/kubernetes/kubernetes/pull/109024), [@stlaz](https://github.com/stlaz)) [SIG API Machinery and Instrumentation]
- Adds `OpenAPIV3SchemaInterface` to `DiscoveryClient` and its variants for fetching OpenAPI v3 schema documents. ([#108992](https://github.com/kubernetes/kubernetes/pull/108992), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery, Architecture, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Allow kubectl to manage resources by filename patterns without the shell expanding it first ([#102265](https://github.com/kubernetes/kubernetes/pull/102265), [@danielrodriguez](https://github.com/danielrodriguez)) [SIG CLI]
- An alpha flag --subresource is added to get, patch, edit replace kubectl commands to fetch and update status and scale subresources. ([#99556](https://github.com/kubernetes/kubernetes/pull/99556), [@nikhita](https://github.com/nikhita)) [SIG API Machinery, CLI and Testing]
- Apiextensions_openapi_v3_regeneration_count metric (alpha) will be emitted for OpenAPI V3. ([#109128](https://github.com/kubernetes/kubernetes/pull/109128), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Instrumentation]
- Apply ProxyTerminatingEndpoints to all traffic policies (external, internal, cluster, local). ([#108691](https://github.com/kubernetes/kubernetes/pull/108691), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network and Testing]
- CEL regex patterns in x-kubernetes-valiation rules are compiled when CRDs are created/updated if the pattern is provided as a string constant in the expression. Any regex compile errors are reported as a CRD create/update validation error. ([#108617](https://github.com/kubernetes/kubernetes/pull/108617), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- Changes the kubectl --validate flag from a bool to a string that accepts the values {true, strict, warn, false, ignore}
  - true/strict - perform validation and error the request on any invalid fields in the ojbect. It will attempt to perform server-side validation if it is enabled on the apiserver, otherwise it will fall back to client-side validation.
  - warn - perform server-side validation and warn on any invalid fields (but ultimately let the request succeed by dropping any invalid fields from the object). If validation is not available on the server, perform no validation.
  - false/ignore - perform no validation, silently dropping invalid fields from the object. ([#108350](https://github.com/kubernetes/kubernetes/pull/108350), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery, CLI, Node and Testing]
- CycleState is now optimized for "write once and read many times". ([#108724](https://github.com/kubernetes/kubernetes/pull/108724), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Enable beta feature HonorPVReclaimPolicy by default. ([#109035](https://github.com/kubernetes/kubernetes/pull/109035), [@deepakkinni](https://github.com/deepakkinni)) [SIG Apps and Storage]
- Kube-apiserver: Subresources such as 'status' and 'scale' now support tabular output content types. ([#103516](https://github.com/kubernetes/kubernetes/pull/103516), [@ykakarap](https://github.com/ykakarap)) [SIG API Machinery, Auth and Testing]
- Kubeadm: add the flag "--experimental-initial-corrupt-check" to etcd static Pod manifests to ensure etcd member data consistency ([#109074](https://github.com/kubernetes/kubernetes/pull/109074), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl now supports shell completion for the <type>/<name> format for specifying resources.
  
  kubectl now provides shell completion for container names following the --container/-c flag of the exec command.
  
  kubectl's shell completion now suggests resource types for commands that only apply to pods. ([#108493](https://github.com/kubernetes/kubernetes/pull/108493), [@marckhouzam](https://github.com/marckhouzam)) [SIG CLI]
- Kubelet now creates an iptables chain named `KUBE-IPTABLES-HINT` in
  the `mangle` table. Containerized components that need to modify iptables
  rules in the host network namespace can use the existence of this chain
  to more-reliably determine whether the system is using iptables-legacy or
  iptables-nft. ([#109059](https://github.com/kubernetes/kubernetes/pull/109059), [@danwinship](https://github.com/danwinship)) [SIG Network and Node]
- Kubernetes 1.24 bumped version of golang it is compiled with to go1.18, which introduced significant changes to its garbage collection algorithm. As a result, we observed an increase in memory usage for kube-apiserver in larger an heavily loaded clusters up to ~25% (with the benefit of API call latencies drop by up to 10x on 99th percentiles). If the memory increase is not acceptable for you you can mitigate by setting GOGC env variable (for our tests using GOGC=63 brings memory usage back to original value, although the exact value may depend on usage patterns on your cluster). ([#108870](https://github.com/kubernetes/kubernetes/pull/108870), [@dims](https://github.com/dims)) [SIG Architecture, Release and Testing]
- Leader Migration is now GA. All new configuration files onwards should use version v1. ([#109072](https://github.com/kubernetes/kubernetes/pull/109072), [@jiahuif](https://github.com/jiahuif)) [SIG Cloud Provider]
- Mark AzureDisk CSI migration as GA ([#107681](https://github.com/kubernetes/kubernetes/pull/107681), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Move volume expansion feature to GA ([#108929](https://github.com/kubernetes/kubernetes/pull/108929), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Apps, Auth, Node, Storage and Testing]
- New "field_validation_request_duration_seconds" metric, measures how long requests take, indicating the value of the fieldValidation query parameter and whether or not server-side field validation is enabled on the apiserver ([#109120](https://github.com/kubernetes/kubernetes/pull/109120), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery and Instrumentation]
- New feature gate, ServiceIPStaticSubrange, to enable the new strategy in the Service IP allocators, so the IP range is subdivided and dynamic allocated ClusterIP addresses for Services are allocated preferently from the upper range. ([#106792](https://github.com/kubernetes/kubernetes/pull/106792), [@aojea](https://github.com/aojea)) [SIG Instrumentation]
- OpenAPI definitions served by kube-api-server now include enum types by default. ([#108898](https://github.com/kubernetes/kubernetes/pull/108898), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery]
- Promote graceful shutdown based on pod priority to beta ([#107986](https://github.com/kubernetes/kubernetes/pull/107986), [@wzshiming](https://github.com/wzshiming)) [SIG Instrumentation, Node and Testing]
- Update the k8s.io/system-validators library to v1.7.0 ([#108988](https://github.com/kubernetes/kubernetes/pull/108988), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Updates `kubectl kustomize` and `kubectl apply -k` to Kustomize v4.5.4 ([#108994](https://github.com/kubernetes/kubernetes/pull/108994), [@KnVerey](https://github.com/KnVerey)) [SIG CLI]
- `kubectl version` now includes information on the embedded version of Kustomize ([#108817](https://github.com/kubernetes/kubernetes/pull/108817), [@KnVerey](https://github.com/KnVerey)) [SIG CLI and Testing]

### Bug or Regression

- A node IP provided to kublet via `--node-ip` will now be preferred for
  when determining the node's primary IP and using the external cloud provider
  (CCM). ([#107750](https://github.com/kubernetes/kubernetes/pull/107750), [@stephenfin](https://github.com/stephenfin)) [SIG Cloud Provider and Node]
- Add one metrics(`kubelet_volume_stats_health_abnormal`) of volume health state to kubelet ([#108758](https://github.com/kubernetes/kubernetes/pull/108758), [@fengzixu](https://github.com/fengzixu)) [SIG Instrumentation, Node, Storage and Testing]
- CEL validation failure returns object type instead of object. ([#107090](https://github.com/kubernetes/kubernetes/pull/107090), [@cici37](https://github.com/cici37)) [SIG API Machinery]
- Call NodeExpand on all nodes in case of RWX volumes ([#108693](https://github.com/kubernetes/kubernetes/pull/108693), [@gnufied](https://github.com/gnufied)) [SIG Apps, Node and Storage]
- Failure to start a container cannot accidentally result in the pod being considered "Succeeded" in the presence of deletion. ([#107845](https://github.com/kubernetes/kubernetes/pull/107845), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Fix --retries functionality for negative values in kubectl cp ([#108748](https://github.com/kubernetes/kubernetes/pull/108748), [@atiratree](https://github.com/atiratree)) [SIG CLI]
- Fix a bug that out-of-tree plugin is misplaced when using scheduler v1beta3 config ([#108613](https://github.com/kubernetes/kubernetes/pull/108613), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fix a race in timeout handler that could lead to kube-apiserver crashes ([#108455](https://github.com/kubernetes/kubernetes/pull/108455), [@Argh4k](https://github.com/Argh4k)) [SIG API Machinery]
- Fix indexer bug that resulted in incorrect index updates if number of index values for a given object was changing during update ([#109137](https://github.com/kubernetes/kubernetes/pull/109137), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Fix issue where the job controller might not remove the job tracking finalizer from pods when deleting a job, or when the pod is orphan ([#108752](https://github.com/kubernetes/kubernetes/pull/108752), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Testing]
- Kubelet now checks "NoExecute" taint/toleration before accepting pods, except for static pods. ([#101218](https://github.com/kubernetes/kubernetes/pull/101218), [@gjkim42](https://github.com/gjkim42)) [SIG Node]
- Re-adds response status and headers on verbose kubectl responses ([#108505](https://github.com/kubernetes/kubernetes/pull/108505), [@rikatz](https://github.com/rikatz)) [SIG API Machinery and CLI]
- Record requests rejected with 429 in the apiserver_request_total metric ([#108927](https://github.com/kubernetes/kubernetes/pull/108927), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Instrumentation]
- Services with "internalTrafficPolicy: Local" now behave more like
  "externalTrafficPolicy: Local". Also, "internalTrafficPolicy: Local,
  externalTrafficPolicy: Cluster" is now implemented correctly. ([#106497](https://github.com/kubernetes/kubernetes/pull/106497), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Skip re-allocate logic if pod is already removed to avoid panic ([#108831](https://github.com/kubernetes/kubernetes/pull/108831), [@waynepeking348](https://github.com/waynepeking348)) [SIG Node]
- Updating kubelet permissions check for Windows nodes to see if process is elevated instead of checking if process owner is in Administrators group ([#108146](https://github.com/kubernetes/kubernetes/pull/108146), [@marosset](https://github.com/marosset)) [SIG Node and Windows]

### Other (Cleanup or Flake)

- Add PreemptionPolicy in PriorityClass describe ([#108701](https://github.com/kubernetes/kubernetes/pull/108701), [@denkensk](https://github.com/denkensk)) [SIG CLI and Scheduling]
- Deprecate apiserver_dropped_requests_total metric. The same data can be read from apiserver_request_terminations_total metric. ([#109018](https://github.com/kubernetes/kubernetes/pull/109018), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Instrumentation]
- Migrate statefulset files to structured logging ([#106109](https://github.com/kubernetes/kubernetes/pull/106109), [@h4ghhh](https://github.com/h4ghhh)) [SIG Apps and Instrumentation]
- Remove deprecated `--serviceaccount`, `--hostport`, `--requests` and `--limits` from kubectl run. ([#108820](https://github.com/kubernetes/kubernetes/pull/108820), [@mozillazg](https://github.com/mozillazg)) [SIG CLI]
- Remove deprecated generator and container-port flags ([#106824](https://github.com/kubernetes/kubernetes/pull/106824), [@lauchokyip](https://github.com/lauchokyip)) [SIG CLI]
- Rename unschedulableQ to unschedulablePods ([#108919](https://github.com/kubernetes/kubernetes/pull/108919), [@denkensk](https://github.com/denkensk)) [SIG Instrumentation, Scheduling and Testing]
- SPDY transport in client-go will no longer follow redirects. ([#108531](https://github.com/kubernetes/kubernetes/pull/108531), [@tallclair](https://github.com/tallclair)) [SIG API Machinery and Node]
- ServerResources was deprecated in February 2019 (https://github.com/kubernetes/kubernetes/commit/618050e) and now it's being removed and ServerGroupsAndResources is suggested to be used instead ([#107180](https://github.com/kubernetes/kubernetes/pull/107180), [@ardaguclu](https://github.com/ardaguclu)) [SIG API Machinery, Apps and CLI]
- Update runc to 1.1.0
  Update cadvisor to 0.44.0 ([#109029](https://github.com/kubernetes/kubernetes/pull/109029), [@ehashman](https://github.com/ehashman)) [SIG CLI, Node and Testing]
- Update runc to 1.1.1 ([#109104](https://github.com/kubernetes/kubernetes/pull/109104), [@kolyshkin](https://github.com/kolyshkin)) [SIG Node]
- Users who look at iptables dumps will see some changes in the naming and structure of rules. ([#109060](https://github.com/kubernetes/kubernetes/pull/109060), [@thockin](https://github.com/thockin)) [SIG Network and Testing]

### Uncategorized

- Deprecate kubectl version long output, will be replaced with kubectl version --short. Users requiring full output should use --output=yaml|json instead. ([#108987](https://github.com/kubernetes/kubernetes/pull/108987), [@soltysh](https://github.com/soltysh)) [SIG CLI]

## Dependencies

### Added
- github.com/blang/semver/v4: [v4.0.0](https://github.com/blang/semver/v4/tree/v4.0.0)

### Changed
- github.com/checkpoint-restore/go-criu/v5: [v5.0.0 → v5.3.0](https://github.com/checkpoint-restore/go-criu/v5/compare/v5.0.0...v5.3.0)
- github.com/cilium/ebpf: [v0.6.2 → v0.7.0](https://github.com/cilium/ebpf/compare/v0.6.2...v0.7.0)
- github.com/containerd/console: [v1.0.2 → v1.0.3](https://github.com/containerd/console/compare/v1.0.2...v1.0.3)
- github.com/containerd/containerd: [v1.4.11 → v1.4.12](https://github.com/containerd/containerd/compare/v1.4.11...v1.4.12)
- github.com/cyphar/filepath-securejoin: [v0.2.2 → v0.2.3](https://github.com/cyphar/filepath-securejoin/compare/v0.2.2...v0.2.3)
- github.com/docker/distribution: [v2.7.1+incompatible → v2.8.1+incompatible](https://github.com/docker/distribution/compare/v2.7.1...v2.8.1)
- github.com/docker/docker: [v20.10.7+incompatible → v20.10.12+incompatible](https://github.com/docker/docker/compare/v20.10.7...v20.10.12)
- github.com/godbus/dbus/v5: [v5.0.4 → v5.0.6](https://github.com/godbus/dbus/v5/compare/v5.0.4...v5.0.6)
- github.com/golang/mock: [v1.5.0 → v1.6.0](https://github.com/golang/mock/compare/v1.5.0...v1.6.0)
- github.com/google/cadvisor: [v0.43.0 → v0.44.0](https://github.com/google/cadvisor/compare/v0.43.0...v0.44.0)
- github.com/moby/sys/mountinfo: [v0.4.1 → v0.6.0](https://github.com/moby/sys/mountinfo/compare/v0.4.1...v0.6.0)
- github.com/opencontainers/image-spec: [v1.0.1 → v1.0.2](https://github.com/opencontainers/image-spec/compare/v1.0.1...v1.0.2)
- github.com/opencontainers/runc: [v1.0.3 → v1.1.1](https://github.com/opencontainers/runc/compare/v1.0.3...v1.1.1)
- github.com/opencontainers/selinux: [v1.8.2 → v1.10.0](https://github.com/opencontainers/selinux/compare/v1.8.2...v1.10.0)
- github.com/seccomp/libseccomp-golang: [v0.9.1 → 3879420](https://github.com/seccomp/libseccomp-golang/compare/v0.9.1...3879420)
- go.etcd.io/etcd/api/v3: v3.5.0 → v3.5.1
- go.etcd.io/etcd/client/pkg/v3: v3.5.0 → v3.5.1
- go.etcd.io/etcd/client/v3: v3.5.0 → v3.5.1
- k8s.io/klog/v2: v2.40.1 → v2.60.1
- k8s.io/kube-openapi: ddc6692 → 3ee0da9
- k8s.io/system-validators: v1.6.0 → v1.7.0
- sigs.k8s.io/kustomize/api: v0.10.1 → v0.11.4
- sigs.k8s.io/kustomize/cmd/config: v0.10.2 → v0.10.6
- sigs.k8s.io/kustomize/kustomize/v4: v4.4.1 → v4.5.4
- sigs.k8s.io/kustomize/kyaml: v0.13.0 → v0.13.6

### Removed
- github.com/bits-and-blooms/bitset: [v1.2.0](https://github.com/bits-and-blooms/bitset/tree/v1.2.0)



# v1.24.0-alpha.4


## Downloads for v1.24.0-alpha.4



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes.tar.gz) | 951531e83aed1aaaf6df424e195a913aa7c6faf9aae4f4b55970b37bc223727201088011f5ed35b988aca36e30b8cea75f6a666721b2c52d672c6c8406d3d9c4
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-src.tar.gz) | c715efaa416a7fe208188bf01f40c34e559cf1c2ed6b153eb843563398ec05b1b4574219bb9a4a548e5f726c30d0739753c7a8086837c0ffeee8f2053a6c463b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-darwin-amd64.tar.gz) | dd7d18c1babbc2fcbe481f8e41335cadcd9274b27f05c2d3ded19e820e9b8cc55be72eb2cc404afeea8107c503731cd62b0684533582bfb05b3b58b74d8c091e
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-darwin-arm64.tar.gz) | 0ce731610736b3ed26ac0aa9f193cd76adee1e7d34e4dcfb233dbcb83fcd8620e13371ec9583629bb8a404506c779e8f51415756049849c11c230cec475cfbe4
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-386.tar.gz) | bc5e071407dd994bfa45788e4aa395d23e0a3c0431703804db910dc76eea9cff2ff3d02046e4ead8b04c7bec0d148cdd1332f9951a4b546a32b6f3ca2c8e839d
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-amd64.tar.gz) | 2e52b5d5b7852f1d61a7d03bbcf2d20967846f3295501b32014ed99db0694868a5f67e8ea835d58bd6835d1dcba9bdfba4418f10669a71a8859f7768037fe4c9
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-arm.tar.gz) | 6210c9e5a0327b483fa243b88be0f9afeec36c435c0e001bc25360204ea32ebddf98d4dfdf42b93cad683665ad7976706214abfc84f479c0e47f26d971a9752b
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-arm64.tar.gz) | acba30ad585a11e1a875660556118fb449a2e2e92c19d647c030323bbb3f265face715bda90e67458fbd3272fe2c23abad5dfd712874a5c1a232017ca8747984
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-ppc64le.tar.gz) | 9336988dd0933424f70772b21d17a8c798965abbb77722dec58b3a92da8e6ab2f2dcf8702def7f0d4498b9f76a90dcb6af316650569328cc2f988c015ba9c9b7
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-s390x.tar.gz) | d073aa8ad2ee476b12ae1fabcc762d514a01dab18e3e0afb27396e4d2d77bc9091858146c30c8906e9e577411386c009fa591b2f667765dabf5df04c57330f4b
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-windows-386.tar.gz) | 4096d1de90320c4ad68abf7458eb3d57e6e4e8603a430e0109d4b3e1086a8784f284adc567bd99b063e928e7090d60d17491e528416a7e99c39a00aae7679e9e
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-windows-amd64.tar.gz) | e86cc5e6817f5defbc344958ca88c5d0272d65449bf9bedf95a4588ba188d1c1b870c71513fdbf8159d8bce9b7d3026c37f18c42f60649225d542ab6c545f842
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-windows-arm64.tar.gz) | bcac6d5daeb604857feff355a6e999ae7f64d748a4c9ecb4393282b6a6ad488bbb43f770de86e173bd2299b18cc98bb9f1690664dc8857753a2f625cc82b15a2

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-amd64.tar.gz) | d511b8ab8d3ecb70f35f472a25de8d3b301d384e347f5aedcaeb286e82f264c8715bd87ac9dbf8474c431c6d10290d49b6b2e92cba9bbfe8b6f0af8f11e434e3
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-arm.tar.gz) | 2b2f4f596675ac3871f0f8ba12f619e550395c3ac40cf3c26b8b5aab4f0a9c0e5b30398bcb9aaf5c5e717ba0eb53e317f01d9f6ff4c37c8a8f2bd644a864b43c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-arm64.tar.gz) | 4da262bb6112ae5a8bf0b65659a8a15fda17fdeab4935f137e7680bef03e6262b7db57cea2db6c1a143ef43ca3346d845bba3e1fa97ca0dba7e3769d333310b4
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-ppc64le.tar.gz) | e64aa2f04c46deef946a88a87449dd46f6cbc5ee7cf4662ba6f455877c4bdc3527edeb8eb861e56fe458d8e49d933db5c4a485b587252ee6f55b7751f01c62cd
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-s390x.tar.gz) | 75a4ad3e091709ff2e8ac04f7a7af2999527322a9e857a41985413080a01c5989208648765617ee8bfbc00b462831f3dea5fc512228bda4fcebb45055417c2f7

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-amd64.tar.gz) | 1593bd6ced1aa42e1834a6e049d257a298c7964c706ac07f57d087271e5a4d13866fadc36661affa078bc64f91a7058666fe85879d1d339015e3637669c4c8c6
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-arm.tar.gz) | 883a6b24e134d825330baf1805d563f94300fa060b3c978da9138bba174059b604e28c115437673b0411fbb38e3a7949b99a7089da716ef9a482386dc0b45ca4
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-arm64.tar.gz) | d317105d7ef00696cc27660a871f181a319fcf2d4c1f19a88091f8c7d2bbfbd0a9a4ac8f0fa1e28e432df6ea1215848e23e002eb7f7f51b1f01b67d4acaaaa5a
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-ppc64le.tar.gz) | 01bd08aabd58aee0db6b1e376e76d741d2ebd592995407a47cb315f73c2a5b311a540ae620eb4471eb2aea74c13162107f299efe98c15e5ec6948d9d3e1cd378
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-s390x.tar.gz) | 81cd8e22010cab4cabefb4e2a879a61dd1d5d057ec15bb3b98d5aa5e79820abf0a650cb7c667d7fb38afd9c851a5043df8a148e303a77da2ed14a69a7747c18e
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-windows-amd64.tar.gz) | 033527ea64ddc4d4f166ebee9e171958fd8b25f1f515c736fe820ae1e7c763b1a79e402e2f8446a216582dda23564dd810bed064d35e8652d27bc51dca224e10

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0-alpha.3

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - The `LegacyServiceAccountTokenNoAutoGeneration` feature gate is beta, and enabled by default. When enabled, Secret API objects containing service account tokens are no longer auto-generated for every ServiceAccount. Use the [TokenRequest](https://kubernetes.io/docs/reference/kubernetes-api/authentication-resources/token-request-v1/) API to acquire service account tokens, or if a non-expiring token is required, create a Secret API object for the token controller to populate with a service account token by following this [guide](https://kubernetes.io/docs/concepts/configuration/secret/#service-account-token-secrets). ([#108309](https://github.com/kubernetes/kubernetes/pull/108309), [@zshihang](https://github.com/zshihang)) [SIG API Machinery, Apps, Auth and Testing]
 
## Changes by Kind

### Deprecation

- --pod-infra-container-image kubelet flag is deprecated and will be removed in future releases ([#108045](https://github.com/kubernetes/kubernetes/pull/108045), [@hakman](https://github.com/hakman)) [SIG Node]
- Client.authentication.k8s.io/v1alpha1 ExecCredential has been removed. If you are using a client-go credential plugin that relies on the v1alpha1 API please contact the distributor of your plugin for instructions on how to migrate to the v1 API. ([#108616](https://github.com/kubernetes/kubernetes/pull/108616), [@margocrawf](https://github.com/margocrawf)) [SIG API Machinery and Auth]
- Remove deprecated feature gates ValidateProxyRedirects and StreamingProxyRedirects ([#106830](https://github.com/kubernetes/kubernetes/pull/106830), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery]
- The node.k8s.io/v1alpha1 RuntimeClass API is no longer served. Use the node.k8s.io/v1 API version, available since v1.20 ([#103061](https://github.com/kubernetes/kubernetes/pull/103061), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, CLI, Node and Testing]

### API Change

- Add 2 new options for kube-proxy running in winkernel mode. 
  `--forward-healthcheck-vip`, if specified as true, health check traffic whose destination is service VIP will be forwarded to kube-proxy's healthcheck service. `--root-hnsendpoint-name` specifies the name of the hns endpoint for the root network namespace.
  This option enables the pass-through load balancers like Google's GCLB to correctly health check the backend services. Without this change, the health check packets is dropped, and Windows node will be considered to be unhealthy by those load balancers. ([#99287](https://github.com/kubernetes/kubernetes/pull/99287), [@anfernee](https://github.com/anfernee)) [SIG API Machinery, Cloud Provider, Network, Testing and Windows]
- Added CEL runtime cost calculation into CustomerResource validation. CustomerResource validation will fail if runtime cost exceeds the budget. ([#108482](https://github.com/kubernetes/kubernetes/pull/108482), [@cici37](https://github.com/cici37)) [SIG API Machinery]
- CRD writes will generate validation errors if a CEL validation rule references the identifier "oldSelf" on a part of the schema that does not support it. ([#108013](https://github.com/kubernetes/kubernetes/pull/108013), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Feature of `DefaultPodTopologySpread` is graduated to GA ([#108278](https://github.com/kubernetes/kubernetes/pull/108278), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Feature of `PodOverhead` is graduated to GA ([#108441](https://github.com/kubernetes/kubernetes/pull/108441), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery, Apps, Node and Scheduling]
- Fixes a regression in v1beta1 PodDisruptionBudget handling of "strategic merge patch"-type API requests for the `selector` field. Prior to 1.21, these requests would merge `matchLabels` content and replace `matchExpressions` content. In 1.21, patch requests touching the `selector` field started replacing the entire selector. This is consistent with server-side apply and the v1 PodDisruptionBudget behavior, but should not have been changed for v1beta1. ([#108138](https://github.com/kubernetes/kubernetes/pull/108138), [@liggitt](https://github.com/liggitt)) [SIG Apps, Auth and Testing]
- Kube-apiserver: --audit-log-version and --audit-webhook-version now only support the default value of audit.k8s.io/v1. The v1alpha1 and v1beta1 audit log versions, deprecated since 1.13, have been removed. ([#108092](https://github.com/kubernetes/kubernetes/pull/108092), [@carlory](https://github.com/carlory)) [SIG API Machinery, Auth and Testing]
- Pod-affinity namespace selector and cross-namespace quota graduated to GA. The feature gate PodAffinityNamespaceSelector is locked and will be removed in 1.26. ([#108136](https://github.com/kubernetes/kubernetes/pull/108136), [@ahg-g](https://github.com/ahg-g)) [SIG API Machinery, Apps, Scheduling and Testing]
- Suspend job to GA. The feature gate SuspendJob is locked and will be removed in 1.26. ([#108129](https://github.com/kubernetes/kubernetes/pull/108129), [@ahg-g](https://github.com/ahg-g)) [SIG Apps and Testing]
- The CertificateSigningRequest `spec.expirationSeconds` API field has graduated to GA. The `CSRDuration` feature gate for the field is now unconditionally enabled and will be removed in 1.26. ([#108782](https://github.com/kubernetes/kubernetes/pull/108782), [@cfryanr](https://github.com/cfryanr)) [SIG API Machinery, Apps, Auth, Instrumentation and Testing]
- TopologySpreadConstraints includes minDomains field to limit the minimum number of topology domains. ([#107674](https://github.com/kubernetes/kubernetes/pull/107674), [@sanposhiho](https://github.com/sanposhiho)) [SIG API Machinery, Apps and Scheduling]

### Feature

- Add a deprecated cmd flag for the time interval between flushing pods from unschedualbeQ to activeQ or backoffQ. ([#108017](https://github.com/kubernetes/kubernetes/pull/108017), [@denkensk](https://github.com/denkensk)) [SIG Scheduling]
- Add one metrics(`kubelet_volume_stats_health_abnormal`) of volume health state to kubelet ([#105585](https://github.com/kubernetes/kubernetes/pull/105585), [@fengzixu](https://github.com/fengzixu)) [SIG Instrumentation, Node, Storage and Testing]
- Add the metric `container_oom_events_total` to kubelet's cAdvisor metric endpoint. ([#108004](https://github.com/kubernetes/kubernetes/pull/108004), [@jonkerj](https://github.com/jonkerj)) [SIG Node]
- Added support for btrfs resizing ([#108561](https://github.com/kubernetes/kubernetes/pull/108561), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]
- CRD x-kubernetes-validations rules now support the CEL functions: isSorted, sum, min, max, indexOf, lastIndexOf, find and findAll. ([#108312](https://github.com/kubernetes/kubernetes/pull/108312), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Client-go metrics: change bucket distribution for rest_client_request_duration_seconds and rest_client_rate_limiter_duration_seconds from [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512] to [0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0}] ([#106911](https://github.com/kubernetes/kubernetes/pull/106911), [@aojea](https://github.com/aojea)) [SIG API Machinery, Architecture and Instrumentation]
- Client-go: add new histogram metric to record the size of the requests and responses. ([#108296](https://github.com/kubernetes/kubernetes/pull/108296), [@aojea](https://github.com/aojea)) [SIG API Machinery, Architecture and Instrumentation]
- Cluster/gce/gci/configure.sh now supports downloading crictl on ARM64 nodes ([#108034](https://github.com/kubernetes/kubernetes/pull/108034), [@tstapler](https://github.com/tstapler)) [SIG Cloud Provider]
- Env var for additional cli flags used in the csi-proxy binary when a Windows nodepool is created with kube-up.sh ([#107806](https://github.com/kubernetes/kubernetes/pull/107806), [@mauriciopoppe](https://github.com/mauriciopoppe)) [SIG Cloud Provider and Windows]
- Increase default value of discovery cache TTL for kubectl to 6 hours. ([#107141](https://github.com/kubernetes/kubernetes/pull/107141), [@mk46](https://github.com/mk46)) [SIG CLI]
- Introduce policy to allow the HPA to consume the external.metrics.k8s.io API group. ([#104244](https://github.com/kubernetes/kubernetes/pull/104244), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Auth, Autoscaling and Instrumentation]
- Kubeadm: apply "second stage" of the plan to migrate kubeadm away from the usage of the word "master" in labels and taints. For new clusters, the label "node-role.kubernetes.io/master" will no longer be added to control plane nodes, only the label "node-role.kubernetes.io/control-plane" will be added. For clusters that are being upgraded to 1.24 with "kubeadm upgrade apply", the command will remove the label "node-role.kubernetes.io/master" from existing control plane nodes. For new clusters, both the old taint "node-role.kubernetes.io/master:NoSchedule" and new taint "node-role.kubernetes.io/control-plane:NoSchedule" will be added to control plane nodes. In release 1.20 ("first stage"), a release note instructed to preemptively tolerate the new taint. For clusters that are being upgraded to 1.24 with "kubeadm upgrade apply", the command will add the new taint "node-role.kubernetes.io/control-plane:NoSchedule" to existing control plane nodes. Please adapt your infrastructure to these changes. In 1.25 the old taint "node-role.kubernetes.io/master:NoSchedule" will be removed. ([#107533](https://github.com/kubernetes/kubernetes/pull/107533), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: better surface errors during "kubeadm upgrade" when waiting for the kubelet to restart static pods on control plane nodes ([#108315](https://github.com/kubernetes/kubernetes/pull/108315), [@Monokaix](https://github.com/Monokaix)) [SIG Cluster Lifecycle]
- Kubeadm: improve the strict parsing of user YAML/JSON configuration files. Next to printing warnings for unknown and duplicate fields (current state), also print warnings for fields with incorrect case sensitivity - e.g. "controlPlaneEndpoint" (valid), "ControlPlaneEndpoint" (invalid). Instead of only printing warnings during "init" and "join" also print warnings when downloading the ClusterConfiguration, KubeletConfiguration or KubeProxyConfiguration objects from the cluster. This can be useful if the user has patched these objects in their respective ConfigMaps with mistakes. ([#107725](https://github.com/kubernetes/kubernetes/pull/107725), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubelet: add kubelet_volume_metric_collection_duration_seconds metrics for volume disk usage calculation duration ([#107201](https://github.com/kubernetes/kubernetes/pull/107201), [@pacoxu](https://github.com/pacoxu)) [SIG Instrumentation, Node and Storage]
- Kubernetes in now built with go1.18rc1 ([#107105](https://github.com/kubernetes/kubernetes/pull/107105), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Release, Storage and Testing]
- No ([#108432](https://github.com/kubernetes/kubernetes/pull/108432), [@iXinqi](https://github.com/iXinqi)) [SIG Testing and Windows]
- PreFilter extension in the scheduler framework now returns not only status but also PreFilterResult ([#108648](https://github.com/kubernetes/kubernetes/pull/108648), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling, Storage and Testing]
- Remove the deprecated flag `--experimental-check-node-capabilities-before-mount`. With CSI now GA, there is a better alternative. Remove any use of  `--experimental-check-node-capabilities-before-mount` from your kubelet scripts or manifests. ([#104732](https://github.com/kubernetes/kubernetes/pull/104732), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps, Cloud Provider, Node and Storage]
- Set PodMaxUnschedulableQDuration as 5 min. ([#108761](https://github.com/kubernetes/kubernetes/pull/108761), [@denkensk](https://github.com/denkensk)) [SIG Scheduling]
- Support in-tree PV deletion protection finalizer. ([#108400](https://github.com/kubernetes/kubernetes/pull/108400), [@deepakkinni](https://github.com/deepakkinni)) [SIG Apps and Storage]
- The .spec.loadBalancerClass field for Services is now generally available. ([#107979](https://github.com/kubernetes/kubernetes/pull/107979), [@XudongLiuHarold](https://github.com/XudongLiuHarold)) [SIG Cloud Provider, Network and Testing]
- Turn on CSIMigrationAzureFile by default on 1.24 ([#105070](https://github.com/kubernetes/kubernetes/pull/105070), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- When invoked with `-list-images`, the e2e.test binary now also lists the images that might be needed for storage tests. ([#108458](https://github.com/kubernetes/kubernetes/pull/108458), [@pohly](https://github.com/pohly)) [SIG Testing]

### Bug or Regression

- <kubectl version> fails for unexpected extra arguments ([#107967](https://github.com/kubernetes/kubernetes/pull/107967), [@jlsong01](https://github.com/jlsong01)) [SIG CLI]
- Bug: client-go clientset was not defaulting the user agent, using the default golang agent for all the requests. ([#108772](https://github.com/kubernetes/kubernetes/pull/108772), [@aojea](https://github.com/aojea)) [SIG API Machinery and Instrumentation]
- Bump sigs.k8s.io/apiserver-network-proxy/konnectivity-client@v0.0.30 to fix a goroutine leak in kube-apiserver when using egress selctor with the gRPC mode. ([#108437](https://github.com/kubernetes/kubernetes/pull/108437), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, Auth and Cloud Provider]
- Existing InTree AzureFile PVs which don't have a secret namespace defined will now work properly after enabling CSI migration - the namespace will be obtained from ClaimRef. ([#108000](https://github.com/kubernetes/kubernetes/pull/108000), [@RomanBednar](https://github.com/RomanBednar)) [SIG Cloud Provider and Storage]
- Fix a bug in attachdetach controller that didn't properly handle kube-apiserver errors leading to stuck attachments/detachments. ([#108167](https://github.com/kubernetes/kubernetes/pull/108167), [@jfremy](https://github.com/jfremy)) [SIG Apps]
- Fix a bug that out-of-tree plugin is misplaced when using scheduler v1beta3 config ([#108613](https://github.com/kubernetes/kubernetes/pull/108613), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fix container creation errors for pods with cpu requests bigger than 256 cpus ([#106570](https://github.com/kubernetes/kubernetes/pull/106570), [@odinuge](https://github.com/odinuge)) [SIG Node]
- Fix to allow fsGroup to be applied for CSI Inline Volumes ([#108662](https://github.com/kubernetes/kubernetes/pull/108662), [@dobsonj](https://github.com/dobsonj)) [SIG Storage]
- Fix: do not return early in the node informer when there is no change of the topology label. ([#108149](https://github.com/kubernetes/kubernetes/pull/108149), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fixed a bug that caused credentials in an exec plugin to override the static certificates set in a kubeconfig. ([#107410](https://github.com/kubernetes/kubernetes/pull/107410), [@margocrawf](https://github.com/margocrawf)) [SIG API Machinery, Auth and Testing]
- Fixed a regression that could incorrectly reject pods with OutOfCpu errors if they were rapidly scheduled after other pods were reported as complete in the API. The Kubelet now waits to report the phase of a pod as terminal in the API until all running containers are guaranteed to have stopped and no new containers can be started.  Short-lived pods may take slightly longer (~1s) to report Succeeded or Failed after this change. ([#108366](https://github.com/kubernetes/kubernetes/pull/108366), [@smarterclayton](https://github.com/smarterclayton)) [SIG Apps, Node and Testing]
- Fixes a bug where a partial EndpointSlice update could cause node name information to be dropped from endpoints that were not updated. ([#108198](https://github.com/kubernetes/kubernetes/pull/108198), [@liggitt](https://github.com/liggitt)) [SIG Network]
- Fixes bug in CronJob Controller V2 where it would lose track of jobs upon job template labels change. ([#107997](https://github.com/kubernetes/kubernetes/pull/107997), [@d-honeybadger](https://github.com/d-honeybadger)) [SIG Apps]
- Improved logging when volume times out waiting for attach/detach. ([#108628](https://github.com/kubernetes/kubernetes/pull/108628), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]
- Increase Azure ACR credential provider timeout ([#108209](https://github.com/kubernetes/kubernetes/pull/108209), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Kube-apiserver: ensures the namespace of objects sent to admission webhooks matches the request namespace. Previously, objects without a namespace set would have the request namespace populated after mutating admission, and objects with a namespace that did not match the request namespace would be rejected after admission. ([#94637](https://github.com/kubernetes/kubernetes/pull/94637), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Kube-apiserver: removed apf_fd from server logs which could contain data identifying the requesting user ([#108631](https://github.com/kubernetes/kubernetes/pull/108631), [@jupblb](https://github.com/jupblb)) [SIG API Machinery]
- Kube-proxy in iptables mode now only logs the full iptables input at -v=9 rather than -v=5. ([#108224](https://github.com/kubernetes/kubernetes/pull/108224), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kube-proxy will no longer hold service node ports open on the node. Users are still advised not to run any listener on node ports range used by kube-proxy. ([#108496](https://github.com/kubernetes/kubernetes/pull/108496), [@khenidak](https://github.com/khenidak)) [SIG Network]
- Kubeadm: fix a bug when using "kubeadm init --dry-run" with certificate authority files (ca.key / ca.crt) present in /etc/kubernetes/pki) ([#108410](https://github.com/kubernetes/kubernetes/pull/108410), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where Windows nodes fail to join an IPv6 cluster due to preflight errors ([#108769](https://github.com/kubernetes/kubernetes/pull/108769), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet don't forcefully close active connections on heartbeat failures, using the http2 health check mechanism to detect broken connections. Users can force the previous behavior of the kubelet by setting the environment variable DISABLE_HTTP2. ([#108107](https://github.com/kubernetes/kubernetes/pull/108107), [@aojea](https://github.com/aojea)) [SIG API Machinery and Node]
- Prevent unnecessary Endpoints and EndpointSlice updates caused by Pod ResourceVersion change ([#108078](https://github.com/kubernetes/kubernetes/pull/108078), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Print <default> as the value in case kubectl describe ingress shows default-backend:80 when no default backend is present ([#108506](https://github.com/kubernetes/kubernetes/pull/108506), [@jlsong01](https://github.com/jlsong01)) [SIG CLI]
- Replace the url label of rest_client_request_duration_seconds and rest_client_rate_limiter_duration_seconds metrics with a host label to prevent cardinality explosions and keep only the useful information. This is a breaking change required for security reasons. ([#106539](https://github.com/kubernetes/kubernetes/pull/106539), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Instrumentation]
- The `TopologyAwareHints` feature gate is now enabled by default. This will allow users to opt-in to Topology Aware Hints by setting the `service.kubernetes.io/topology-aware-hints` on a Service. This will not affect any Services without that annotation set. ([#108747](https://github.com/kubernetes/kubernetes/pull/108747), [@robscott](https://github.com/robscott)) [SIG Network]
- This code change fixes the bug that UDP services would trigger unnecessary LoadBalancer updates. The root cause is that a field not working for non-TCP protocols is considered.
  ref: https://github.com/kubernetes-sigs/cloud-provider-azure/pull/1090 ([#107981](https://github.com/kubernetes/kubernetes/pull/107981), [@lzhecheng](https://github.com/lzhecheng)) [SIG Cloud Provider]
- Topology translation of in-tree vSphere volume to vSphere CSI. ([#108611](https://github.com/kubernetes/kubernetes/pull/108611), [@divyenpatel](https://github.com/divyenpatel)) [SIG Storage]

### Other (Cleanup or Flake)

- API server's deprecated `--deserialization-cache-size` flag is now removed. ([#108448](https://github.com/kubernetes/kubernetes/pull/108448), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- API server's deprecated `--experimental-encryption-provider-config` flag is now removed. Adapt your machinery to use the `--encryption-provider-config` flag that is available since v1.13. ([#108423](https://github.com/kubernetes/kubernetes/pull/108423), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- API server's deprecated `--target-ram-mb` flag is now removed. ([#108457](https://github.com/kubernetes/kubernetes/pull/108457), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery, Cloud Provider, Scalability and Testing]
- Endpoints and EndpointSlice controllers no longer populate [resourceVersion of targetRef](https://kubernetes.io/docs/reference/kubernetes-api/common-definitions/object-reference/#ObjectReference) in Endpoints and EndpointSlices ([#108450](https://github.com/kubernetes/kubernetes/pull/108450), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Improve error message when applying CRDs before the CRD exists in a cluster ([#107363](https://github.com/kubernetes/kubernetes/pull/107363), [@eddiezane](https://github.com/eddiezane)) [SIG CLI]
- Improved algorithm for selecting "best" non-preferred hint in the TopologyManager ([#108154](https://github.com/kubernetes/kubernetes/pull/108154), [@klueska](https://github.com/klueska)) [SIG Node]
- Kube-proxy doesn't set the sysctl net.ipv4.conf.all.route_localnet=1 if no IPv4 loopback address is selected by the nodePortAddresses configuration parameter. ([#107684](https://github.com/kubernetes/kubernetes/pull/107684), [@aojea](https://github.com/aojea)) [SIG Network]
- Remove support for node-expansion between node-stage and node-publish ([#108614](https://github.com/kubernetes/kubernetes/pull/108614), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- The `WarningHeaders` feature gate that is GA since v1.22 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#108394](https://github.com/kubernetes/kubernetes/pull/108394), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- The e2e.test binary supports a new `--kubelet-root` parameter to override the default `/var/lib/kubelet` path. CSI storage tests use this. ([#108253](https://github.com/kubernetes/kubernetes/pull/108253), [@pohly](https://github.com/pohly)) [SIG Node, Storage and Testing]
- The scheduler framework option `runAllFilters` is removed. ([#108829](https://github.com/kubernetes/kubernetes/pull/108829), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Windows Pause no longer has support for SAC releases 1903, 1909, 2004. Windows image support is now Ltcs 2019 (1809), 20H2, LTSC 2022 ([#107056](https://github.com/kubernetes/kubernetes/pull/107056), [@jsturtevant](https://github.com/jsturtevant)) [SIG Windows]
- `kube-addon-manager` image version is bumped to 9.1.6 ([#108341](https://github.com/kubernetes/kubernetes/pull/108341), [@zshihang](https://github.com/zshihang)) [SIG Cloud Provider, Scalability and Testing]

## Dependencies

### Added
- github.com/google/gnostic: [v0.5.7-v3refs](https://github.com/google/gnostic/tree/v0.5.7-v3refs)

### Changed
- github.com/cpuguy83/go-md2man/v2: [v2.0.0 → v2.0.1](https://github.com/cpuguy83/go-md2man/v2/compare/v2.0.0...v2.0.1)
- github.com/google/cel-go: [v0.9.0 → v0.10.1](https://github.com/google/cel-go/compare/v0.9.0...v0.10.1)
- github.com/prometheus/client_golang: [v1.12.0 → v1.12.1](https://github.com/prometheus/client_golang/compare/v1.12.0...v1.12.1)
- github.com/russross/blackfriday/v2: [v2.0.1 → v2.1.0](https://github.com/russross/blackfriday/v2/compare/v2.0.1...v2.1.0)
- github.com/spf13/cobra: [v1.2.1 → v1.4.0](https://github.com/spf13/cobra/compare/v1.2.1...v1.4.0)
- golang.org/x/crypto: 32db794 → 8634188
- golang.org/x/mod: v0.5.1 → 9b9b3d8
- golang.org/x/net: 491a49a → cd36cc0
- golang.org/x/oauth2: 2bc19b1 → d3ed0bb
- golang.org/x/sys: da31bd3 → 3681064
- golang.org/x/term: 6886f2d → 03fcf44
- golang.org/x/time: 1f47c86 → 90d013b
- golang.org/x/tools: v0.1.8 → 897bd77
- google.golang.org/genproto: fe13028 → 42d7afd
- k8s.io/kube-openapi: e816edb → ddc6692
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.27 → v0.0.30

### Removed
- cloud.google.com/go/firestore: v1.1.0
- github.com/armon/go-metrics: [f0300d1](https://github.com/armon/go-metrics/tree/f0300d1)
- github.com/armon/go-radix: [7fddfc3](https://github.com/armon/go-radix/tree/7fddfc3)
- github.com/bgentry/speakeasy: [v0.1.0](https://github.com/bgentry/speakeasy/tree/v0.1.0)
- github.com/bketelsen/crypt: [v0.0.4](https://github.com/bketelsen/crypt/tree/v0.0.4)
- github.com/fatih/color: [v1.7.0](https://github.com/fatih/color/tree/v1.7.0)
- github.com/googleapis/gnostic: [v0.5.5](https://github.com/googleapis/gnostic/tree/v0.5.5)
- github.com/hashicorp/consul/api: [v1.1.0](https://github.com/hashicorp/consul/api/tree/v1.1.0)
- github.com/hashicorp/consul/sdk: [v0.1.1](https://github.com/hashicorp/consul/sdk/tree/v0.1.1)
- github.com/hashicorp/errwrap: [v1.0.0](https://github.com/hashicorp/errwrap/tree/v1.0.0)
- github.com/hashicorp/go-cleanhttp: [v0.5.1](https://github.com/hashicorp/go-cleanhttp/tree/v0.5.1)
- github.com/hashicorp/go-immutable-radix: [v1.0.0](https://github.com/hashicorp/go-immutable-radix/tree/v1.0.0)
- github.com/hashicorp/go-msgpack: [v0.5.3](https://github.com/hashicorp/go-msgpack/tree/v0.5.3)
- github.com/hashicorp/go-multierror: [v1.0.0](https://github.com/hashicorp/go-multierror/tree/v1.0.0)
- github.com/hashicorp/go-rootcerts: [v1.0.0](https://github.com/hashicorp/go-rootcerts/tree/v1.0.0)
- github.com/hashicorp/go-sockaddr: [v1.0.0](https://github.com/hashicorp/go-sockaddr/tree/v1.0.0)
- github.com/hashicorp/go-syslog: [v1.0.0](https://github.com/hashicorp/go-syslog/tree/v1.0.0)
- github.com/hashicorp/go-uuid: [v1.0.1](https://github.com/hashicorp/go-uuid/tree/v1.0.1)
- github.com/hashicorp/go.net: [v0.0.1](https://github.com/hashicorp/go.net/tree/v0.0.1)
- github.com/hashicorp/golang-lru: [v0.5.0](https://github.com/hashicorp/golang-lru/tree/v0.5.0)
- github.com/hashicorp/hcl: [v1.0.0](https://github.com/hashicorp/hcl/tree/v1.0.0)
- github.com/hashicorp/logutils: [v1.0.0](https://github.com/hashicorp/logutils/tree/v1.0.0)
- github.com/hashicorp/mdns: [v1.0.0](https://github.com/hashicorp/mdns/tree/v1.0.0)
- github.com/hashicorp/memberlist: [v0.1.3](https://github.com/hashicorp/memberlist/tree/v0.1.3)
- github.com/hashicorp/serf: [v0.8.2](https://github.com/hashicorp/serf/tree/v0.8.2)
- github.com/magiconair/properties: [v1.8.5](https://github.com/magiconair/properties/tree/v1.8.5)
- github.com/mattn/go-colorable: [v0.0.9](https://github.com/mattn/go-colorable/tree/v0.0.9)
- github.com/mattn/go-isatty: [v0.0.3](https://github.com/mattn/go-isatty/tree/v0.0.3)
- github.com/miekg/dns: [v1.0.14](https://github.com/miekg/dns/tree/v1.0.14)
- github.com/mitchellh/cli: [v1.0.0](https://github.com/mitchellh/cli/tree/v1.0.0)
- github.com/mitchellh/go-homedir: [v1.0.0](https://github.com/mitchellh/go-homedir/tree/v1.0.0)
- github.com/mitchellh/go-testing-interface: [v1.0.0](https://github.com/mitchellh/go-testing-interface/tree/v1.0.0)
- github.com/mitchellh/gox: [v0.4.0](https://github.com/mitchellh/gox/tree/v0.4.0)
- github.com/mitchellh/iochan: [v1.0.0](https://github.com/mitchellh/iochan/tree/v1.0.0)
- github.com/pascaldekloe/goe: [57f6aae](https://github.com/pascaldekloe/goe/tree/57f6aae)
- github.com/pelletier/go-toml: [v1.9.3](https://github.com/pelletier/go-toml/tree/v1.9.3)
- github.com/posener/complete: [v1.1.1](https://github.com/posener/complete/tree/v1.1.1)
- github.com/ryanuber/columnize: [9b3edd6](https://github.com/ryanuber/columnize/tree/9b3edd6)
- github.com/sean-/seed: [e2103e2](https://github.com/sean-/seed/tree/e2103e2)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/spf13/cast: [v1.3.1](https://github.com/spf13/cast/tree/v1.3.1)
- github.com/spf13/jwalterweatherman: [v1.1.0](https://github.com/spf13/jwalterweatherman/tree/v1.1.0)
- github.com/spf13/viper: [v1.8.1](https://github.com/spf13/viper/tree/v1.8.1)
- github.com/subosito/gotenv: [v1.2.0](https://github.com/subosito/gotenv/tree/v1.2.0)
- gopkg.in/ini.v1: v1.62.0



# v1.24.0-alpha.3


## Downloads for v1.24.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes.tar.gz) | 9de65cc28a4a641c4bbfd386706cf61a8c36d8928623ab6cac28ac378d77aed458e14e8782703d58182a2aab31a71ea5806bb7013f7ddb8657fdc350790f03cd
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-src.tar.gz) | 7f76982cbc1eb84883b86ac2b3b017b02dd970cd3a57883306ec0cbf7dfac9009ee368ca93cec43261dd186d1b75bcde4fe61a6eb4b804361b23b2c8ef2755df

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | ff98f55de0e1db549135811e801afe85e7e4b2216d4dc543adc9e6039f997ad9f2ecbd06daf3deb14d86afb736a6a114ddde760925b647be1f6ea6fb1d9b5153
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | 8582dcb6cbf56eca07cbfe98f06adf0f6e0c2827783d6a8c3fbc7e7e5b69f4b5781643beae37fe0868bde8915aa45c991db550e6fec619a7bec94eb38ed4c3b5
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 15a56b02915397970c90615ad921e9e3125e39b8ee8a857c22c855957fee8b2de540eb51b2be4f3f8138bee1bf5ed7dfe0b8f1171df9ff25b6789c5ba420dc3c
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | a16a6762713dd8aaf5a8f5cab4f9be949ac04ce7bd1e182858c47a1769f43d5c99faebee228e8543c237444a3ffb55bf1b6c5272a370f545c137b699c12cdce6
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | ff9432a0b3538e5170276ced1f4878570a2a2ddb5c59c25d735eed09dde492bc2a2bbde5cccbcd835cbae94f4912f6b5c1b82168e81b4e106204ebe342c4e866
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | bbc22c098152f1a0e2532e3d7e700116a31fe59212ca8d47f9f95d3f3c997c1d8e8a1b75619bc73feb9f2c4953733d7e5b67ee34e8b5781b3b8d9865075f4b7f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | a6046b9726268f0557ab8c72fe5274ea5554afdf7b6b2aa77d41a207ecb5460e42d5ba68de3e359318956af5c390e46b3471418b413d6ee54169262245f8fe46
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | c5701618eba6bec591445f2bd192301178fa6b0b8c5359d81faea92a393611a5fe697c2dea241769b7c55f2a2dd5fbf1361ee5f0edefa6582ec88ce0f248b1b4
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 87f8d99bacbf846b3cf0e6ccc72e99c38363ef606103ea924330a49f1a9d46c3043c8eda397246246d0fac0fe1204944628bf5add48b1a51a618ba4523df4673
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 74ba2c6770113dc91f18072a1df15f368ea23312e418a016e25f93545a5e234d1e133210f0e136642d2a4e23574a015625ca434234b085d1f1a16821ba2dc266
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | f72d3d0948e0669b23227cc952d20e40054088df52c12c0f84ff9b2c6714e558f7d2349fc8b78758d57bd251e1f2ea50296a8a0f22fbea762b048c547eb78e26

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | f68d4691dfa7916c8e566c1ca2dfb11b0c675223a95cef8b90586ead93217713122ae402f1276246a4c1daa45f3545ee9fd600a7f97fdc056a23b5e66acb9921
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | babf77bfb1061120dd525b25bf89b7bb0b81892dddfd70a1693ddbe642e3066104dd54938e1bfff86069b66cf5af6e0698236cc25c62daa1187548255c512bb4
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 0b8846b01d20fada109fef957b5670ee2da9cb334090475ecf9de508beae912a330cb446d77a8308782ab77ed37784838dd8b287c55d5a54a64588992fa9c385
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | a6d3c3b261aa0abcfb5ecb8c94cec75ca01da839100c6ec0456a5b0e4e118b4c603954b5094e434da25489528a4cf346675e04d4c213202053f6b5ab431bad67
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 10cd559a40cac8e35d89976703a839a6095bf7a52e2cf1c090007e43bf3b487a6eba19e86d066f4a76922acd3217ec4b62dfbb2f1ccb3b422bc84ffdf844b161

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | 32702b81a4f0a19c9ef28978880b150284d12c766c25d89b83808bff44ec99ebf629585ec9a5ee4cec72ecb27e8282802578e39c586b4bb53e5caf02f80012a0
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | 2fca5344d8debe5a37676793992db19a1ef296d3d0d14682f7a8f023a5ee5c5a1691bbd585483b277ea81ed2487d2260da9cd869b55116d6c2606a630c2a29d7
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | 424ea735c4bde3e3e70ae5e6517ec4c9ca46628b8f8e74eba5775b85faf31c9e3d9c6fb876f2965b86e596f1268741f75851d95ec91a3c9eff4ef0b772a3e3b6
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 517aff6677ac101bb09b2f2aba7d96d8901ae4b0ed6f7ba33bd8ab33447babd7eaf2a9482d0ebdc1b3f235c7fa9e6b4437fd19ccc6518889265e78fcdc8a6d27
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | dcc61fcc5ac44a6e25f53f2538e41bd23e8c34c2b481c3bb8be6caf42ab74b0363f9ddce16c41c5083a70ff4fe1c569d26dc6fcd71b06ad6203ddc3fd7c1be0f
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | 95afe157524f285127fe2e808ec785fb0aa3205dd0b1f34233d35c0eabd8efbcba6b9cd98d69b66464cdb1b178d28c33fd091d95fdcbdfe5a8d6af2df7d44363

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
k8s.gcr.io/conformance:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-apiserver:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-controller-manager:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-proxy:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-scheduler:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x

## Changelog since v1.24.0-alpha.2

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Fixed bug with leads to Node goes "Not-ready" state when credentials for vCenter stored in a secret and Zones feature is in use.
  Zone labels setup moved to KCM component, kubelet skips this step during startup in such case. If credentials stored in cloud-provider config file as plaintext current behaviour does not change and no action required.
  
  For proper functioning "kube-system:vsphere-legacy-cloud-provider" should be allowed to update node object if vCenter credentials stored in secret and Zone feature used. ([#101028](https://github.com/kubernetes/kubernetes/pull/101028), [@lobziik](https://github.com/lobziik)) [SIG Cloud Provider]
 
## Changes by Kind

### Deprecation

- Cluster addon for dashboard was removed. To install dashboard, see [here](https://github.com/kubernetes/dashboard/blob/master/docs/user/README.md). ([#107481](https://github.com/kubernetes/kubernetes/pull/107481), [@shu-mutou](https://github.com/shu-mutou)) [SIG Cloud Provider and Testing]
- Kube-apiserver: the --master-count flag and --endpoint-reconciler-type=master-count reconciler are deprecated in favor of the lease reconciler ([#108062](https://github.com/kubernetes/kubernetes/pull/108062), [@aojea](https://github.com/aojea)) [SIG API Machinery]
- Kubeadm: graduate the UnversionedKubeletConfigMap feature gate to Beta and enable the feature by default. This implies that 1) for new clusters kubeadm will start using the "kube-system/kubelet-config" naming scheme for the kubelet ConfigMap and RBAC rules, instead of the legacy "kubelet-config-x.yy" naming. 2) during upgrade, kubeadm will only write the new scheme ConfigMap and RBAC objects. To disable the feature you can pass "UnversionedKubeletConfigMap: false" in the kubeadm config for new clusters. For upgrade on existing clusters you can also override the behavior by patching the ClusterConfiguration object in "kube-system/kubeadm-config". More details in the associated KEP. ([#108027](https://github.com/kubernetes/kubernetes/pull/108027), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- Remove `tolerate-unready-endpoints` annotation in Service deprecated from 1.11, use `Service.spec.publishNotReadyAddresses` instead. ([#108020](https://github.com/kubernetes/kubernetes/pull/108020), [@tossmilestone](https://github.com/tossmilestone)) [SIG Apps and Network]
- The in-tree azure plugin has been deprecated.  The https://github.com/Azure/kubelogin serves as an out-of-tree replacement via the kubectl/client-go credential plugin mechanism.  Users will now see a warning in the logs regarding this deprecation. ([#107904](https://github.com/kubernetes/kubernetes/pull/107904), [@sabbey37](https://github.com/sabbey37)) [SIG Auth]

### API Change

- CRD deep copies should no longer contain shallow copies of JSONSchemaProps.XValidations. ([#107956](https://github.com/kubernetes/kubernetes/pull/107956), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Feature of `NonPreemptingPriority` is graduated  to GA ([#107432](https://github.com/kubernetes/kubernetes/pull/107432), [@denkensk](https://github.com/denkensk)) [SIG Apps, Scheduling and Testing]
- Fix OpenAPI serialization of the x-kubernetes-validations field ([#107970](https://github.com/kubernetes/kubernetes/pull/107970), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-apiserver: the `metadata.selfLink` field can no longer be populated by kube-apiserver; it was deprecated in 1.16 and has not been populated by default in 1.20+. ([#107527](https://github.com/kubernetes/kubernetes/pull/107527), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Apps, Auth, Autoscaling, CLI, Cloud Provider, Network, Scheduling, Storage and Testing]

### Feature

- Kubernetes is now built with Golang 1.17.7 ([#108091](https://github.com/kubernetes/kubernetes/pull/108091), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Remove feature gate `SetHostnameAsFQDN`. ([#108038](https://github.com/kubernetes/kubernetes/pull/108038), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Node]
- The output of `kubectl describe ingress` now includes an IngressClass name if available ([#107921](https://github.com/kubernetes/kubernetes/pull/107921), [@mpuckett159](https://github.com/mpuckett159)) [SIG CLI]
- The scheduler prints info logs when the extender returned an error. (--v>5) ([#107974](https://github.com/kubernetes/kubernetes/pull/107974), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- `kubectl config delete-user` now supports completion ([#107142](https://github.com/kubernetes/kubernetes/pull/107142), [@dimbleby](https://github.com/dimbleby)) [SIG CLI]
- `kubectl create token` can now be used to request a service account token, and permission to request service account tokens is added to the `edit` and `admin` RBAC roles ([#107880](https://github.com/kubernetes/kubernetes/pull/107880), [@liggitt](https://github.com/liggitt)) [SIG Auth, CLI and Testing]

### Bug or Regression

- A static pod that is rapidly updated was failing to start until the Kubelet was restarted. ([#107900](https://github.com/kubernetes/kubernetes/pull/107900), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- CRI-API: IPs returned by PodSandboxNetworkStatus are ignored by the kubelet for host-network pods. ([#106715](https://github.com/kubernetes/kubernetes/pull/106715), [@aojea](https://github.com/aojea)) [SIG Node]
- Fixes bug in TopologyManager for ensuring aligned allocations on machines with more than 2 NUMA nodes ([#108052](https://github.com/kubernetes/kubernetes/pull/108052), [@klueska](https://github.com/klueska)) [SIG Node]
- Kubeadm: fix a bug related to a warning printed if the KubeletConfiguration "resolvConf" field value does not match "/run/systemd/resolve/resolv.conf" ([#107785](https://github.com/kubernetes/kubernetes/pull/107785), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that 'kubeadm certs generate-csr' command does not remove duplicated SANs ([#107982](https://github.com/kubernetes/kubernetes/pull/107982), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- RunCordonOrUncordon error if drainer has nil Ctx or Client ([#105297](https://github.com/kubernetes/kubernetes/pull/105297), [@jackfrancis](https://github.com/jackfrancis)) [SIG CLI]

### Other (Cleanup or Flake)

- Added an e2e test to verify that the cluster is not vulnerable to CVE-2021-29923 when using Services with IPs with leading zeros, note that this test is a necessary but not sufficient condition, all the components in the clusters that consume IPs addresses from the APIs MUST interpret them as decimal or discard them. ([#107552](https://github.com/kubernetes/kubernetes/pull/107552), [@aojea](https://github.com/aojea)) [SIG Network and Testing]
- Kubectl stack traces will now only print at -v=99 and not -v=6 ([#108053](https://github.com/kubernetes/kubernetes/pull/108053), [@eddiezane](https://github.com/eddiezane)) [SIG CLI]
- Remove kubelet `--non-masquerade-cidr` deprecated CLI flag ([#107096](https://github.com/kubernetes/kubernetes/pull/107096), [@hakman](https://github.com/hakman)) [SIG Cloud Provider and Node]
- [k8s.io/utils/clock]: IntervalClock is now deprecated in favour of SimpleIntervalClock ([#108059](https://github.com/kubernetes/kubernetes/pull/108059), [@RaghavRoy145](https://github.com/RaghavRoy145)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]

## Dependencies

### Added
_Nothing has changed._

### Changed
- k8s.io/utils: 7d6a63d → 3a6ce19

### Removed
_Nothing has changed._



# v1.24.0-alpha.2


## Downloads for v1.24.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes.tar.gz) | bd3257bbae848869e20696e4570f29d61d78187d710c99fa01c5602e4edcf818f8129a68d80e83e51cc4b1010eea8e61691a9439c6c72607b5e1b6e32cd2a60e
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-src.tar.gz) | d8235197b71248ffa5fcbabbdab11c208f9d55f58db498e038e7464c0caf99bfddfa8d34e8af46ca3f908d865d6836786c0030afce15a3d1ff5f4d1cdfc69929

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | a17876d27eb72590893ddf440f9c0fd18137a6e5f9dc57b34a8a9057fffd6b6a5356bca92adf888e3e223b0aa58f47dc08594fbdb6d0e1934d86fdd167b7aca9
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 9f513e665ebf86d795933d55ba7b2d9e183761d6ff36e04626cb2e597ba4af9a840dbf995466a4c4d4ee89f9a9b0cbfa9217ce69bf7d6d66d65989e02e04ae73
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 511b2147da305368cc24c372f052aeaf1f2aa7bf7fdfdb4fc81a6b3163cda4bc8392b0392610799f0bd96500daeae98aa39f657ca37811fed326a21e2d43f218
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 94514f56a6fab4887ea44405ee59020115cfb53d7c1a4d2464fa3ceb804c3d141d4c1d090e7d7652d9514950ea7f52f96b1f59a560359673aa0bb7dffe307198
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 4a3a0c2fa1875caf5c0715b67a8b0e375362e02cd9be88439c32a853a73eff26b419da58772ab1b13ecaed0480a6f7d6d85681d71b096cf941eb9d45e137e157
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 831b837ce1159bdd0e4b7a238d0bbf998b24495cf335fdf960b789fbe255ebee75a7f3d6e9831782b0967bc04323abced69b1384411fbfd637b06d7483a24053
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 09cd6c441ee4b57261966e8c93d45374b577818f79ccaae3042a17bb06203ad41a1a0d046d28382782f2cd8a49a0677fbdfa600783cd61ee36a2cc8ae9ce9e7e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | fe29b548df2d6016a98b4409ff783001be70945875f036d7a799445ef60a1493fd52618c8136cbb6a089d98703148076a09286701e2918400cf1a3ed77aac953
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 3ca9c008c79575525b1b758240e24466a60f9a34c5c12895bc0d8f79ff6b5ab057f3be1d1a7bb561084092cf18d9d46d80698fc0691947fc86b63ec1a4c0decf
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | ee32c78eeae2c8db9f8fc4bac02b5c5a0b9eb29612bfac71f0c9c48f83fd03c31aa2b459a41f0a06087dafbb71cd8c109e797bd243fe72f32db05a584e03f697
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | debcc893f4c4be2ef034e056b126ef5b7c0f60a0a7d43117e2271850ded56dcc9eb103cc764337506bcd5d4bc22a87611c1501c17dfbfe89c62185588a6356ee

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | ec33945179f1ea5ca6334cf761247c975e5d22b1bf9b415dae9903aef67443c94894794f1e2ae932421c847cc7388c6c31307d2c7dd8b28aa8c2f39483f83de6
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | 77a7a799e675ee4fc5371768cbca36f624e2419611740393fd850c0f2506cd4926a0d31d8ed754c06bb1b1852cd53b073a21b7b6a03c6059efe64316a1d39f69
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 6d198e0edf891f4b161d2f0df8945113264b31bafa153ca2a22f4cf0043a2810e2f1687d41e9f7fd2351704d2c720c45ab8cd235ee452897f8322a233e65c435
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 4c1e2a4f076297f4684f8c4781b5cfca685423a3b0b7e761b74d8e35860546437901f8a896a182c2ae6fe69dfd5f2a32468a8fffb2ca08a7c80be6afb444617c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | b3df4a87f3d3d9a2e83d7136bb8ebf6c2a625b893812e1c01bf6e7424e41b8c5c0373912b80e0a309a36936181439b3b5700dc97451fcf86fb7d983d15e8d284

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 20aa621176d9f09cb4e32e4e56eaa933953871d877d4d9a55963f73290e3acce3773446c32e69624f15483c29fb5c05166d0ddc4e413cc5d9dd27f93109b86f1
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 6c7e549b50ba0a1d1ac6371bf72e3833f92187848fae3d75a52f7087336d2e85f976dbf8104ac01109177a8478d82efe12de905db7a88b1b7a11d4f05649e02c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 99f25bc10d2f9139d92e3fb12186854da8b987fd3f060b5b7a906bc27345b93e3cde23b07527f42c3ffc34288e2dc87d957aa73e91cfa4c5f2a0f43bfb00037b
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | aa68d2de162cf75f58fb97b2a65a7d1963a9a2483dae565846da44a335696733aa10e0982badebe4fc9048716cd0a85aef32ba9cf9f22d244f69f2adfe60bc12
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | d64ed933b24b7d897194bc70954a42d8217e9e4bc5f0bf797cad3fa54a16b63b5b5a1731886d4bde9b5e80306481686620b027caa0ef413925bb446f6d0a96a9
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | f3af562d8b4f3b17d039b56bec041166bcf9dfa831b3bbbc1ea67864dae00093e564ce7605855d57158f6ab3aaa7b847bebb91948563b439388c028617184429

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
k8s.gcr.io/kube-apiserver:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-controller-manager:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-proxy:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-scheduler:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/conformance:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x

## Changelog since v1.24.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Docker runtime support using dockshim in the kubelet is now completely removed in 1.24. The kubelet used to have a a module called "dockershim" which implements CRI support for Docker and it has seen maintenance issues in the Kubernetes community. From 1.24 onwards, please move to a container runtime that is a full-fledged implementation of CRI (v1alpha1 or v1 compliant) as they become available. ([#97252](https://github.com/kubernetes/kubernetes/pull/97252), [@dims](https://github.com/dims)) [SIG Cloud Provider, Instrumentation, Network, Node and Testing]
  - The calculations for Pod topology spread skew now excludes nodes that
  don't match the node affinity/selector. This may lead to unschedulable pods if you previously had pods
  matching the spreading selector on those excluded nodes (not matching the node affinity/selector),
  especially when the topologyKey is not node-level. Revisit the node affinity and/or pod selector in the
  topology spread constraints to avoid this scenario. ([#107009](https://github.com/kubernetes/kubernetes/pull/107009), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
 
## Changes by Kind

### Deprecation

- "kubeadm.k8s.io/v1beta2" has been deprecated and will be removed in a future release, possibly in 3 releases (one year). You should start using "kubeadm.k8s.io/v1beta3" for new clusters. To migrate your old configuration files on disk you can use the "kubeadm config migrate" command. ([#107013](https://github.com/kubernetes/kubernetes/pull/107013), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Deprecate Service.Spec.LoadBalancerIP. This field was under-specified and its meaning varies across implementations.  As of Kubernetes v1.24, users are encouraged to use implementation-specific annotations when available.  This field may be removed in a future API version. ([#107235](https://github.com/kubernetes/kubernetes/pull/107235), [@uablrek](https://github.com/uablrek)) [SIG Apps and Network]
- Kube-apiserver: the insecure address flags `--address`, `--insecure-bind-address`, `--port` and `--insecure-port` (inert since 1.20) are removed ([#106859](https://github.com/kubernetes/kubernetes/pull/106859), [@knight42](https://github.com/knight42)) [SIG API Machinery, Cloud Provider and Cluster Lifecycle]
- The experimental dynamic log sanitization feature has been deprecated and removed in the 1.24 release. The feature is no longer available for use. ([#107207](https://github.com/kubernetes/kubernetes/pull/107207), [@ehashman](https://github.com/ehashman)) [SIG Instrumentation, Scheduling and Security]
- The insecure address flags `--address` and `--port` in kube-controller-manager have been no effect since v1.20 and is removed in v1.24. ([#106860](https://github.com/kubernetes/kubernetes/pull/106860), [@knight42](https://github.com/knight42)) [SIG API Machinery, Node and Testing]

### API Change

- Add a new metric `webhook_fail_open_count` to monitor webhooks that fail open ([#107171](https://github.com/kubernetes/kubernetes/pull/107171), [@ltagliamonte-dd](https://github.com/ltagliamonte-dd)) [SIG API Machinery and Instrumentation]
- Fix failed flushing logs in defer function when kubelet cmd exit 1. ([#104774](https://github.com/kubernetes/kubernetes/pull/104774), [@kerthcet](https://github.com/kerthcet)) [SIG Node and Scheduling]
- Rename metrics `evictions_number` to `evictions_total` and mark it as stable. The original `evictions_number` metrics name is marked as "Deprecated" and will be removed in kubernetes 1.23 ([#106366](https://github.com/kubernetes/kubernetes/pull/106366), [@cyclinder](https://github.com/cyclinder)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage, Testing and Windows]
- The `ServiceLBNodePortControl` feature graduates to GA. The feature gate will be removed in 1.26. ([#107027](https://github.com/kubernetes/kubernetes/pull/107027), [@uablrek](https://github.com/uablrek)) [SIG Network and Testing]
- The feature DynamicKubeletConfig is removed from the kubelet. ([#106932](https://github.com/kubernetes/kubernetes/pull/106932), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Apps, Auth, Instrumentation, Node and Testing]
- Update default API priority-and-fairness config to avoid endpoint/configmaps operations from controller-manager to all match leader-election priority level. ([#106725](https://github.com/kubernetes/kubernetes/pull/106725), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]

### Feature

- A new Priority and Fairness metric 'apiserver_flowcontrol_work_estimate_seats_samples' has been 
  added that tracks the estimated seats associated with a request ([#106628](https://github.com/kubernetes/kubernetes/pull/106628), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Instrumentation]
- Add completion for `kubectl config set-context`. ([#106739](https://github.com/kubernetes/kubernetes/pull/106739), [@kebe7jun](https://github.com/kebe7jun)) [SIG CLI]
- Add metric for measuring end-to-end volume mount timing ([#107006](https://github.com/kubernetes/kubernetes/pull/107006), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Add more message for no PodSandbox container ([#107116](https://github.com/kubernetes/kubernetes/pull/107116), [@yxxhero](https://github.com/yxxhero)) [SIG Node]
- Added field add_ambient_capabilities to the Capabilities message in the CRI-API. ([#104620](https://github.com/kubernetes/kubernetes/pull/104620), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Node]
- Added label selector flag to all "kubectl rollout" commands ([#99758](https://github.com/kubernetes/kubernetes/pull/99758), [@aramperes](https://github.com/aramperes)) [SIG CLI]
- Added prune flag into diff command to simulate `apply --prune` ([#105164](https://github.com/kubernetes/kubernetes/pull/105164), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Adds SetTransform to SharedInformer to allow users to transform objects before they are stored. ([#107507](https://github.com/kubernetes/kubernetes/pull/107507), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- Adds proxy-url flag into kubectl config set-cluster ([#105566](https://github.com/kubernetes/kubernetes/pull/105566), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Adds support for kubectl commands (`kubectl exec` and `kubectl port-forward`) via a SOCKS5 proxy. ([#105632](https://github.com/kubernetes/kubernetes/pull/105632), [@xens](https://github.com/xens)) [SIG API Machinery, Architecture, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Feature of `PreferNominatedNode` is graduated  to GA ([#106619](https://github.com/kubernetes/kubernetes/pull/106619), [@chendave](https://github.com/chendave)) [SIG Scheduling and Testing]
- In text format, log messages that previously used quoting to prevent multi-line output (for example, text="some \"quotation\", a\nline break") will now be printed with more readable multi-line output without the escape sequences. ([#107103](https://github.com/kubernetes/kubernetes/pull/107103), [@pohly](https://github.com/pohly)) [SIG Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Kube-apiserver: when merging lists, Server Side Apply now prefers the order of the submitted request instead of the existing persisted object ([#107565](https://github.com/kubernetes/kubernetes/pull/107565), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- Kube-scheduler remove insecure flags. You can use --bind-address and --secure-port instead. ([#106865](https://github.com/kubernetes/kubernetes/pull/106865), [@jonyhy96](https://github.com/jonyhy96)) [SIG Scheduling]
- Kubeadm: add support for dry running "kubeadm reset". The new flag "kubeadm reset --dry-run" is similar to the existing flag for "kubeadm init/join/upgrade" and allows you to see what changes would be applied. ([#107512](https://github.com/kubernetes/kubernetes/pull/107512), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: default the kubeadm configuration to the containerd socket (Unix: unix:///var/run/containerd/containerd.sock, Windows: "npipe:////./pipe/containerd-containerd") instead of the one for Docker. If the "Init|JoinConfiguration.nodeRegistration.criSocket" field is empty during cluster creation and multiple sockets are found on the host always throw an error and ask the user to specify which one to use by setting the value in the field. Make sure you update any kubeadm configuration files on disk, to not include the dockershim socket unless you are still using kubelet version < 1.24 with kubeadm >= 1.24.
  
  Remove the DockerValidor and ServiceCheck for the "docker" service from kubeadm preflight. Docker is no longer special cased during host validation and ideally this task should be done in the now external cri-dockerd project where the importance of the compatibility matters.
  
  Use crictl for all communication with CRI sockets for actions like pulling images and obtaining a list of running containers instead of using the docker CLI in the case of Docker. ([#107317](https://github.com/kubernetes/kubernetes/pull/107317), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl logs will now warn and default to the first container in a pod. This new behavior brings it in line with kubectl exec. ([#105964](https://github.com/kubernetes/kubernetes/pull/105964), [@kidlj](https://github.com/kidlj)) [SIG CLI]
- Kubelet: following dockershim related flags are also removed along with dockershim 
  --experimental-dockershim-root-directory, --docker-endpoint, --image-pull-progress-deadline, --network-plugin, 
  --cni-conf-dir,--cni-bin-dir, --cni-cache-dir, --network-plugin-mtu ([#106907](https://github.com/kubernetes/kubernetes/pull/106907), [@cyclinder](https://github.com/cyclinder)) [SIG Cloud Provider, Node and Testing]
- Kubernetes is now built with Golang 1.17.5 ([#106956](https://github.com/kubernetes/kubernetes/pull/106956), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- Kubernetes is now built with Golang 1.17.6 ([#107612](https://github.com/kubernetes/kubernetes/pull/107612), [@palnabarun](https://github.com/palnabarun)) [SIG Release and Testing]
- OpenStack Cinder CSI migration is now GA and switched on by default, Cinder CSI driver must be installed on clusters on OpenStack for Cinder volumes to work (has been since v1.21). ([#107462](https://github.com/kubernetes/kubernetes/pull/107462), [@dims](https://github.com/dims)) [SIG Scheduling and Storage]
- Remove feature gate `ImmutableEphemeralVolumes`. ([#107152](https://github.com/kubernetes/kubernetes/pull/107152), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Node and Storage]
- This adds a path `/header?key=` to `agnhost netexec` allowing one to view what the header value is of the incoming request.
  
  Ex:
  
  $ curl -H "X-Forwarded-For: something" 172.17.0.2:8080/header?key=X-Forwarded-For
  something ([#107796](https://github.com/kubernetes/kubernetes/pull/107796), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Testing]
- Update golang.org/x/net to v0.0.0-20211209124913-491a49abca63 ([#106949](https://github.com/kubernetes/kubernetes/pull/106949), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- We have added a new Priority and Fairness metric apiserver_flowcontrol_request_dispatch_no_accommodation_total' 
  to track the number of times a request dispatch attempt results in a no-accommodation status due to lack of available seats ([#106629](https://github.com/kubernetes/kubernetes/pull/106629), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Instrumentation]

### Bug or Regression

- A new label `type` has been added to `apiserver_flowcontrol_request_execution_seconds` metric - it has the following values: 
  - 'regular': indicates that it is a non long running request
  - 'watch': indicates that it is a watch request ([#105517](https://github.com/kubernetes/kubernetes/pull/105517), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Instrumentation]
- Add a test to guarantee that conformance clusters require at least 2 untainted nodes ([#106313](https://github.com/kubernetes/kubernetes/pull/106313), [@aojea](https://github.com/aojea)) [SIG Architecture and Testing]
- Allow attached volumes to be mounted quicker by skipping exp. backoff when checking for reported-in-use volumes ([#106853](https://github.com/kubernetes/kubernetes/pull/106853), [@gnufied](https://github.com/gnufied)) [SIG Apps, Node and Storage]
- An inefficient lock in EndpointSlice controller metrics cache has been reworked. Network programming latency may be significantly reduced in certain scenarios, especially in clusters with a large number of Services. ([#107091](https://github.com/kubernetes/kubernetes/pull/107091), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Scalability]
- Apiserver will now reject connection attempts to 0.0.0.0/:: when handling a proxy subresource request ([#107402](https://github.com/kubernetes/kubernetes/pull/107402), [@anguslees](https://github.com/anguslees)) [SIG Network]
- Apiserver, if configured to reconcile the kubernetes.default service endpoints, checks if the configured Service IP range matches the apiserver public address IP family, and fails to start if not. ([#106721](https://github.com/kubernetes/kubernetes/pull/106721), [@aojea](https://github.com/aojea)) [SIG API Machinery and Testing]
- Change node staging path for csi driver to use a PV agnostic path. Nodes must be drained before updating the kubelet with this change. ([#107065](https://github.com/kubernetes/kubernetes/pull/107065), [@saikat-royc](https://github.com/saikat-royc)) [SIG Storage and Testing]
- Client-go: fix that paged list calls with ResourceVersionMatch set would fail once paging kicked in. ([#107311](https://github.com/kubernetes/kubernetes/pull/107311), [@fasaxc](https://github.com/fasaxc)) [SIG API Machinery]
- Fix Azurefile volumeid collision issue in csi migration ([#107575](https://github.com/kubernetes/kubernetes/pull/107575), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix a panic when using invalid output format in kubectl create secret command ([#107221](https://github.com/kubernetes/kubernetes/pull/107221), [@rikatz](https://github.com/rikatz)) [SIG CLI]
- Fix libct/cg/fs2: fix GetStats for unsupported hugetlb error on Raspbian Bullseye ([#106912](https://github.com/kubernetes/kubernetes/pull/106912), [@Letme](https://github.com/Letme)) [SIG Node]
- Fix performance regression in JSON logging caused by syncing stdout every time error was logged. ([#107035](https://github.com/kubernetes/kubernetes/pull/107035), [@serathius](https://github.com/serathius)) [SIG Instrumentation and Scalability]
- Fix: azuredisk parameter lowercase translation issue ([#107429](https://github.com/kubernetes/kubernetes/pull/107429), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: delete non existing Azure disk issue ([#107406](https://github.com/kubernetes/kubernetes/pull/107406), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: remove outdated ipv4 route when the corresponding node is deleted ([#106164](https://github.com/kubernetes/kubernetes/pull/106164), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fixed a bug that a pod's .status.nominatedNodeName is not cleared properly, and thus over-occupied system resources. ([#106816](https://github.com/kubernetes/kubernetes/pull/106816), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fixed a bug that could cause a panic when a /healthz request times out. ([#107034](https://github.com/kubernetes/kubernetes/pull/107034), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Fixed a bug where vSphere client connections where not being closed during testing. Leaked vSphere client sessions were causing resource exhaustion during automated testing. ([#107337](https://github.com/kubernetes/kubernetes/pull/107337), [@derek-pryor](https://github.com/derek-pryor)) [SIG Storage and Testing]
- Fixed detaching CSI volumes from nodes when a CSI driver name has prefix "csi-". ([#107025](https://github.com/kubernetes/kubernetes/pull/107025), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixed duplicate port opening in kube-proxy when "--nodeport-addresses" is empty ([#107413](https://github.com/kubernetes/kubernetes/pull/107413), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Fixed kubectl bug where bash completions don't work if --context flag is specified with a value that contains a colon ([#107439](https://github.com/kubernetes/kubernetes/pull/107439), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Fixes a bug where unwanted fields were being returned from a create dry-run: uid and, if generateName was used, name. ([#107088](https://github.com/kubernetes/kubernetes/pull/107088), [@joejulian](https://github.com/joejulian)) [SIG API Machinery and Testing]
- Fixes a rare race condition handling requests that timeout ([#107452](https://github.com/kubernetes/kubernetes/pull/107452), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Fixes a regression in 1.23 that incorrectly pruned data from array items of a custom resource that set `x-kubernetes-preserve-unknown-fields: true` ([#107688](https://github.com/kubernetes/kubernetes/pull/107688), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Fixes a regression in 1.23 where update requests to previously persisted `Service` objects that have not been modified since 1.19 can be rejected with an incorrect `spec.clusterIPs: Required value` error ([#107847](https://github.com/kubernetes/kubernetes/pull/107847), [@thockin](https://github.com/thockin)) [SIG API Machinery, Network and Testing]
- Fixes handling of objects with invalid selectors ([#107559](https://github.com/kubernetes/kubernetes/pull/107559), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Scheduling and Storage]
- Fixes regression in CPUManager that it will release exclusive CPUs in app containers inherited from init containers when the init containers were removed. ([#104837](https://github.com/kubernetes/kubernetes/pull/104837), [@eggiter](https://github.com/eggiter)) [SIG Node]
- Fixes static pod add and removes restarts in certain cases. ([#107695](https://github.com/kubernetes/kubernetes/pull/107695), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Improve handling of unmount failures when device may be in-use by another container/process ([#107789](https://github.com/kubernetes/kubernetes/pull/107789), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Improve rounding of PodTopologySpread scores to offer better scoring when spreading a low number of pods. ([#107384](https://github.com/kubernetes/kubernetes/pull/107384), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Kubeadm: during execution of the "check expiration" command, treat the etcd CA as external if there is a missing etcd CA key file (etcd/ca.key) and perform the proper validation on certificates signed by the etcd CA. Additionally, make sure that the CA for all entries in the output table is included - for both certificates on disk and in kubeconfig files. ([#106891](https://github.com/kubernetes/kubernetes/pull/106891), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- No ([#107769](https://github.com/kubernetes/kubernetes/pull/107769), [@liurupeng](https://github.com/liurupeng)) [SIG Cloud Provider and Windows]
- NodeRestriction admission: nodes are now allowed to update PersistentVolumeClaim status fields `resizeStatus` and `allocatedResources` when the `RecoverVolumeExpansionFailure` feature is enabled ([#107686](https://github.com/kubernetes/kubernetes/pull/107686), [@gnufied](https://github.com/gnufied)) [SIG Auth and Storage]
- Only extend token lifetimes when --service-account-extend-token-expiration is true and the requested token audiences are empty or exactly match all values for --api-audiences ([#105954](https://github.com/kubernetes/kubernetes/pull/105954), [@jyotimahapatra](https://github.com/jyotimahapatra)) [SIG Auth and Testing]
- Removed validation if AppArmor profiles are loaded on the local node. This should be handled by the
  container runtime. ([#97966](https://github.com/kubernetes/kubernetes/pull/97966), [@saschagrunert](https://github.com/saschagrunert)) [SIG Auth, Node and Security]
- Restore NumPDBViolations info of nodes, when HTTPExtender ProcessPreemption. This info will be used in subsequent filtering steps - pickOneNodeForPreemption ([#105853](https://github.com/kubernetes/kubernetes/pull/105853), [@caden2016](https://github.com/caden2016)) [SIG Scheduling]
- Reverts graceful node shutdown to match 1.21 behavior of setting pods that have not yet successfully completed to "Failed" phase if the GracefulNodeShutdown feature is enabled in kubelet. The GracefulNodeShutdown feature is beta and must be explicitly configured via kubelet config to be enabled in 1.21+. This changes 1.22 and 1.23 behavior on node shutdown to match 1.21. If you do not want pods to be marked terminated on node shutdown in 1.22 and 1.23, disable the GracefulNodeShutdown feature. ([#106901](https://github.com/kubernetes/kubernetes/pull/106901), [@bobbypage](https://github.com/bobbypage)) [SIG Node and Testing]
- Some command line errors (for example, "kubectl list" -> "unknown command") were printed as log message with escaped line breaks instead of a multi-line plain text, which made the error harder to read. ([#107044](https://github.com/kubernetes/kubernetes/pull/107044), [@pohly](https://github.com/pohly)) [SIG CLI and Testing]
- Some log messages were logged with `"v":0` in JSON output although they are debug messages with a higher verbosity. ([#106978](https://github.com/kubernetes/kubernetes/pull/106978), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Scheduling and Storage]
- The Service field spec.internalTrafficPolicy is no longer defaulted for Services when the type is ExternalName. The field is also dropped on read when the Service type is ExternalName. ([#104846](https://github.com/kubernetes/kubernetes/pull/104846), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
- The feature gate was mentioned as `csiMigrationRBD` where it should have been `CSIMigrationRBD` to be in parity with other migration plugins. This release correct the same and keep it as `CSIMigrationRBD`.
  
  users who have configured this feature gate as `csiMigrationRBD` has to reconfigure the same to `CSIMigrationRBD` from this release. ([#107554](https://github.com/kubernetes/kubernetes/pull/107554), [@humblec](https://github.com/humblec)) [SIG Storage]
- When doing `make test-integration`, you can now usefully include `-args $prog_args` in KUBE_TEST_ARGS. ([#107516](https://github.com/kubernetes/kubernetes/pull/107516), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG Testing]

### Other (Cleanup or Flake)

- --container-runtime kubelet flag is deprecated and will be removed in future releases ([#107094](https://github.com/kubernetes/kubernetes/pull/107094), [@adisky](https://github.com/adisky)) [SIG Node]
- Add details about preemption in the event for scheduling failed ([#107775](https://github.com/kubernetes/kubernetes/pull/107775), [@denkensk](https://github.com/denkensk)) [SIG Scheduling]
- Build/dependencies.yaml: remove the dependency on Docker. With the dockershim removal, core Kubernetes no longer
  has to track the latest validated version of Docker. ([#107607](https://github.com/kubernetes/kubernetes/pull/107607), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
- Correct the error message to not use the "--max-resource-write-bytes" & "--json-patch-max-copy-bytes" string. ([#106875](https://github.com/kubernetes/kubernetes/pull/106875), [@warmchang](https://github.com/warmchang)) [SIG API Machinery]
- E2e tests wait for kube-root-ca.crt to be populated in namespaces for use with projected service account tokens, reducing delays starting those test pods and errors in the logs. ([#107763](https://github.com/kubernetes/kubernetes/pull/107763), [@smarterclayton](https://github.com/smarterclayton)) [SIG Testing]
- Fix documentation typo in cloud-provider ([#106445](https://github.com/kubernetes/kubernetes/pull/106445), [@majst01](https://github.com/majst01)) [SIG Cloud Provider]
- Fix spelling of implemented in pkg/proxy/apis/config/types.go line 206 ([#106453](https://github.com/kubernetes/kubernetes/pull/106453), [@davidleitw](https://github.com/davidleitw)) [SIG Network]
- Kubeadm: all warning messages are printed to stderr instead of stdout. ([#107467](https://github.com/kubernetes/kubernetes/pull/107467), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: handle the removal of dockershim related flags for new kubeadm clusters. If kubelet <1.24 is on the host, kubeadm >=1.24 can continue using the built-in dockershim in the kubelet if the user passes the "{Init|Join}Configuration.nodeRegistration.criSocket" value in the kubeadm configuration to be equal to "unix:///var/run/dockershim.sock" on Unix or "npipe:////./pipe/dockershim" on Windows. If kubelet version >=1.24 is on the host, kubeadm >=1.24 will treat all container runtimes as "remote" using the kubelet flags "--container-runtime=remote --container-runtime-endpoint=scheme://some/path". The special management for kubelet <1.24 will be removed in kubeadm 1.25. ([#106973](https://github.com/kubernetes/kubernetes/pull/106973), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: make sure that "kubeadm init/join" always use a URL scheme (unix:// on Linux and npipe:// on Windows) when passing a value to the "--container-runtime-endpoint" kubelet flag. This flag's value is taken from the kubeadm configuration "criSocket" field or the "--cri-socket" CLI flag. Automatically add a missing URL scheme to the user configuration in memory, but warn them that they should also update their configuration on disk manually. During "kubeadm upgrade apply/node" mutate the "/var/lib/kubelet/kubeadm-flags.env" file on disk and the "kubeadm.alpha.kubernetes.io/cri-socket" annotation Node object if needed. These automatic actions are temporary and will be removed in a future release. In the future the kubelet may not support CRI endpoints without an URL scheme. ([#107295](https://github.com/kubernetes/kubernetes/pull/107295), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the IPv6DualStack feature gate. The feature has been GA and locked to enabled since 1.23. ([#106648](https://github.com/kubernetes/kubernetes/pull/106648), [@calvin0327](https://github.com/calvin0327)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: remove the deprecated output/v1alpha1 API used for machine readable output by some kubeadm commands. In 1.23 kubeadm started using the newer version output/v1alpha2 for the same purpose. ([#107468](https://github.com/kubernetes/kubernetes/pull/107468), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the restriction that the ca.crt can only contain one certificate. If there is more than one certificate in the ca.crt file, kubeadm will pick the first one by default. ([#107327](https://github.com/kubernetes/kubernetes/pull/107327), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl: restores `--dry-run`, `--dry-run=true`, and `--dry-run=false` for compatibility with pre-1.23 invocations. ([#107003](https://github.com/kubernetes/kubernetes/pull/107003), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Kubernetes e2e framework will use the url "invalid.registry.k8s.io/invalid" instead "invalid.com/invalid" for test that use an invalid registry. ([#107455](https://github.com/kubernetes/kubernetes/pull/107455), [@aojea](https://github.com/aojea)) [SIG Testing]
- Mark kubelet `--container-runtime-endpoint` and `--image-service-endpoint` CLI flags as stable ([#106954](https://github.com/kubernetes/kubernetes/pull/106954), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Migrate volume/csi/csi-client.go logs to structured logging ([#99441](https://github.com/kubernetes/kubernetes/pull/99441), [@CKchen0726](https://github.com/CKchen0726)) [SIG Storage]
- Please check your kubelet command line for enabling features and drop "RuntimeClass" if present. Note that this feature has been on by default since 1.14 and was GA'ed in 1.20. ([#106882](https://github.com/kubernetes/kubernetes/pull/106882), [@cyclinder](https://github.com/cyclinder)) [SIG Node]
- The fluentd-elasticsearch addon is no longer included in the cluster directory. It is available from https://github.com/kubernetes-sigs/instrumentation-addons/tree/master/fluentd-elasticsearch ([#107553](https://github.com/kubernetes/kubernetes/pull/107553), [@liggitt](https://github.com/liggitt)) [SIG Cloud Provider and Instrumentation]
- This PR deprecates types in `k8s.io/apimachinery/util/clock`. Please use `k8s.io/utils/clock` instead. ([#106850](https://github.com/kubernetes/kubernetes/pull/106850), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Updated cri-tools to [v1.23.0](https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.23.0) ([#107604](https://github.com/kubernetes/kubernetes/pull/107604), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Release]

## Dependencies

### Added
- github.com/armon/go-socks5: [e753329](https://github.com/armon/go-socks5/tree/e753329)

### Changed
- github.com/cespare/xxhash/v2: [v2.1.1 → v2.1.2](https://github.com/cespare/xxhash/v2/compare/v2.1.1...v2.1.2)
- github.com/moby/term: [9d4ed18 → 3f7ff69](https://github.com/moby/term/compare/9d4ed18...3f7ff69)
- github.com/opencontainers/runc: [v1.0.2 → v1.0.3](https://github.com/opencontainers/runc/compare/v1.0.2...v1.0.3)
- github.com/prometheus/client_golang: [v1.11.0 → v1.12.0](https://github.com/prometheus/client_golang/compare/v1.11.0...v1.12.0)
- github.com/prometheus/common: [v0.28.0 → v0.32.1](https://github.com/prometheus/common/compare/v0.28.0...v0.32.1)
- github.com/prometheus/procfs: [v0.6.0 → v0.7.3](https://github.com/prometheus/procfs/compare/v0.6.0...v0.7.3)
- github.com/yuin/goldmark: [v1.4.0 → v1.4.1](https://github.com/yuin/goldmark/compare/v1.4.0...v1.4.1)
- golang.org/x/mod: v0.4.2 → v0.5.1
- golang.org/x/net: e898025 → 491a49a
- golang.org/x/sys: f4d4317 → da31bd3
- golang.org/x/tools: d4cc65f → v0.1.8
- k8s.io/gengo: 485abfe → c02415c
- k8s.io/klog/v2: v2.30.0 → v2.40.1
- k8s.io/utils: cb0fa31 → 7d6a63d
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.25 → v0.0.27
- sigs.k8s.io/json: c049b76 → 9f7c6b3
- sigs.k8s.io/structured-merge-diff/v4: v4.1.2 → v4.2.1

### Removed
_Nothing has changed._



# v1.24.0-alpha.1


## Downloads for v1.24.0-alpha.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes.tar.gz) | 966bcdcaadb18787bab26852602a56dc973d785d7d9620c9ca870eba7133d93b2aaebf369ce52ae9b49160a4cd0101f7356a080b34c4a9a3a6ed2ff82ffd6400
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-src.tar.gz) | 107fe6bfba5ff79b28ab28a3652b6a3d03fe5a667217e3e6e8aabe391b95ddc8109e62b72239d0f66b31c99a8c0d7efb4a74ea49337c0986a53e4628cd4c45e2

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 70cc548677446b9e523c00b76b928ab7af0685bae57b4e52eb9916fd929d540a05596505cd1e198bdf41f85cebc38ddbde95d5214bfba0de1d24593ea1a047a7
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | fdfa4ee47ea5fa40782cf1c719a1ae2bb33a491209e53761f3368fa409f81d0dfeceafa10fa4659032a1fc1a5ff2c1959cba575c8a6bbfa151abadec01c180ab
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-386.tar.gz) | e7dbad9054cd7b2e7b212cb6403d8727470564b967e95f53e8ff1648f6fe7f63cee22fb1622fb4b278ad911f67c3488f8446e145f44e7e0befe85bba9c94ea11
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 28e9c8e79dc87dc701c87195589a5a38da7563f0c05ad1c0d40a1f545ef51ec4f6973b02e970bf74167a7534c5b788c5b01a94df570032306d561c2b3f7bbde4
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 57f3ad5670e3a52a6f988a6c0177f530ec9cf1841829b5ee439dad57243898ddd53b89988873b60bd6128cff430b4ff24244f48edbcec4ecb1885f7d5cd09bb8
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 99272cdc6adddf2f15b18070e8a174591192c27d419d80ce6f03f584e283c7626dea8b494c1f3b6b3607e94c6ccfeba678713e6041a23a5833880938bd356906
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | dc638a62b53f15038554aea9a652b3688c7f9843f61d183d7984f20195d1d4183baa923ce0c17ccd0fbae98192be97ccc8f2bd32fa1b774d32160196f6c2debc
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | e330d88076c4bd788a17da59281a23fe31076c8c5409df688091dd8688f4f94028db06f3f6dd777ab019184e4287487db76599eeb6647ee8fb545fd1e54b0dd9
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 0b594d163496eadc8f1643e4d383b0fc96f820c47ec649b0d843cba7b43eb0df050c4fb7b6a23e3f5b2696629d2ba9725d0b59a9e3256e6fdda470eb9a726424
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 6c6656618e461a0c398cfc4fd69b5b2aa959c8ef6a25ec23e62e5504e5bd5c72572d6a5dbe795a469a85a330fb5ca3d86aece447c0fbf8067f8ef7d8592359c2
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 8394dec41b013f3869b32ee17ad82e55201f77573a84037c21511f732c851f6297dfd7c145fc9b65e1d0aa8cecca6dd04027bef36942af9fa140260e48851aad

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | b781f1aa2ebdb89c0c2b35cba35c5c000cf8e6f87c71cc5cd9ac5938081d6914fb325a4a902e060a16ba31ada136f8d0d8dbbf2a27eb1c426428cda3e8166580
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | 0b92d1a3020c8128ea6dc337ce2fffb5dc8bf2500a02467434e90ad3025a699fea4eaca837bc9eea291d87b8adbc2b2814d9ab078ed49ecbabb47c42d9b910cf
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | e0804c2fa12d6c356a2dd32c26df3ae2b389ac21f5ea426abe1d3f99e0460d4096ad0a42bdf96fd1d4392874afa5fe16f5796a075f99c3690340fce5533377b3
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 2a520b5ea04d00c3c6f54f4ddb75b6e6ffa3c472d4951e51674b103187c8f129e20a5b1c22b0b3ce64281ae9fbf192069ad849af5ce4d2f1cdc394269c983b55
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 4993090e12df0cb1a3a9abea52e1f6bc5efefe7202d81ec36646b02799200c7128721bffb940d88d763effbcb094d159a18aabad476b39b1fcae461dfec1967e

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 4e8e4cea3ee4f2dfe12ad5d2361ac43dd1d961aa1bf0e5f9cedbe18ef37eae76bed6f9643ef4d771a5eef70ffb65e49e9dd917591dbd0ec0de243df85c20e86a
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | 10823d27fe41b45ea61a75974657d1a178af0ff2535dc8fb4aaf18ae69ddac73375be124294428e723f775fcd7a01b394d65aa623875393f5dcf8c60e51b2709
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 5e3ff263b9b236c78af8fec2372e42ffaa5518a95b086ebe7cd133d0553581ccdba52048297614913ef5f9580a2c2a978ac99152c4cf8871bbab9986c61efb96
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 2bcddf4c9ad002cc8314ff528ae8d91cc3e83123ab8666ae92ea15e024469a01ff0ae18558f521489ec1e0e07f268f5e2324943243bb8fdf3f927205843f057d
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | cf1f4dddbc37d77b41e5bc2cea7c4086d1a45dc018a9b8a2cd91764c70c4818c4deb2d5be451720c47ee373bec2e84f9aba64b99b3363cf98534acf340cf03e3
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 5f8ded94c3833e3748eab6e45192aa49ec50adf6eb7eca57f9342d96c8592d7a33860c794a522276212b988d751bdbc07ff345eb024fc2a29488cca25ea6ddef

## Changelog since v1.23.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Docker runtime support using dockshim in the kubelet is now completely removed in 1.24. The kubelet used to have a a module called "dockershim" which implements CRI support for Docker and it has seen maintenance issues in the Kubernetes community. From 1.24 onwards, please move to a container runtime that is a full-fledged implementation of CRI (v1alpha1 or v1 compliant) as they become available. ([#97252](https://github.com/kubernetes/kubernetes/pull/97252), [@dims](https://github.com/dims)) [SIG Cloud Provider, Instrumentation, Network, Node and Testing]
 
## Changes by Kind

### Feature

- Kubernetes is now built with Golang 1.17.4 ([#106833](https://github.com/kubernetes/kubernetes/pull/106833), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- The `NamespaceDefaultLabelName` feature gate, GA since v1.22, is now removed. ([#106838](https://github.com/kubernetes/kubernetes/pull/106838), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps and Node]

### Bug or Regression

- Address a bug in rbd migration translation plugin ([#106878](https://github.com/kubernetes/kubernetes/pull/106878), [@humblec](https://github.com/humblec)) [SIG Storage]
- Fix bug in error messaging for basic-auth and ssh secret validations. ([#106179](https://github.com/kubernetes/kubernetes/pull/106179), [@vivek-koppuru](https://github.com/vivek-koppuru)) [SIG Apps and Auth]
- Kubeadm: allow the "certs check-expiration" command to not require the existence of the cluster CA key (ca.key file) when checking the expiration of managed certificates in kubeconfig files. ([#106854](https://github.com/kubernetes/kubernetes/pull/106854), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Publishing kube-proxy metrics for Windows kernel-mode ([#106581](https://github.com/kubernetes/kubernetes/pull/106581), [@knabben](https://github.com/knabben)) [SIG Instrumentation, Network and Windows]
- The deprecated flag `--really-crash-for-testing` is removed. ([#101719](https://github.com/kubernetes/kubernetes/pull/101719), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Network, Node and Testing]
- [Metrics Server] Bump image to v0.5.2 ([#106492](https://github.com/kubernetes/kubernetes/pull/106492), [@serathius](https://github.com/serathius)) [SIG Cloud Provider and Instrumentation]

### Other (Cleanup or Flake)

- Added an example for the kubectl plugin list command. ([#106600](https://github.com/kubernetes/kubernetes/pull/106600), [@bergerhoffer](https://github.com/bergerhoffer)) [SIG CLI]
- Kubelet config validation error messages are updated ([#105360](https://github.com/kubernetes/kubernetes/pull/105360), [@shuheiktgw](https://github.com/shuheiktgw)) [SIG Node]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
- github.com/containernetworking/cni: [v0.8.1](https://github.com/containernetworking/cni/tree/v0.8.1)