<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.24.0-alpha.1](#v1240-alpha1)
  - [Downloads for v1.24.0-alpha.1](#downloads-for-v1240-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.23.0](#changelog-since-v1230)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)

<!-- END MUNGE: GENERATED_TOC -->

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