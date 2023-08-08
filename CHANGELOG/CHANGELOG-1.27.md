<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.27.3](#v1273)
  - [Downloads for v1.27.3](#downloads-for-v1273)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.27.2](#changelog-since-v1272)
  - [Important Security Information](#important-security-information)
    - [CVE-2023-2728: Bypassing enforce mountable secrets policy imposed by the ServiceAccount admission plugin](#cve-2023-2728-bypassing-enforce-mountable-secrets-policy-imposed-by-the-serviceaccount-admission-plugin)
  - [Changes by Kind](#changes-by-kind)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.27.2](#v1272)
  - [Downloads for v1.27.2](#downloads-for-v1272)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.27.1](#changelog-since-v1271)
  - [Changes by Kind](#changes-by-kind-1)
    - [API Change](#api-change)
    - [Feature](#feature-1)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.27.1](#v1271)
  - [Downloads for v1.27.1](#downloads-for-v1271)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.27.0](#changelog-since-v1270)
  - [Changes by Kind](#changes-by-kind-2)
    - [Bug or Regression](#bug-or-regression-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.27.0](#v1270)
  - [Downloads for v1.27.0](#downloads-for-v1270)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.26.0](#changelog-since-v1260)
  - [Known Issues](#known-issues)
    - [The PreEnqueue extension point doesn't work for Pods going to activeQ through backoffQ](#the-preenqueue-extension-point-doesnt-work-for-pods-going-to-activeq-through-backoffq)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation)
    - [API Change](#api-change-1)
    - [Feature](#feature-2)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.27.0-rc.1](#v1270-rc1)
  - [Downloads for v1.27.0-rc.1](#downloads-for-v1270-rc1)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
    - [Container Images](#container-images-4)
  - [Changelog since v1.27.0-rc.0](#changelog-since-v1270-rc0)
  - [Changes by Kind](#changes-by-kind-4)
    - [Feature](#feature-3)
    - [Bug or Regression](#bug-or-regression-4)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)
- [v1.27.0-rc.0](#v1270-rc0)
  - [Downloads for v1.27.0-rc.0](#downloads-for-v1270-rc0)
    - [Source Code](#source-code-5)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
    - [Container Images](#container-images-5)
  - [Changelog since v1.27.0-beta.0](#changelog-since-v1270-beta0)
  - [Changes by Kind](#changes-by-kind-5)
    - [API Change](#api-change-2)
    - [Feature](#feature-4)
    - [Bug or Regression](#bug-or-regression-5)
  - [Dependencies](#dependencies-5)
    - [Added](#added-5)
    - [Changed](#changed-5)
    - [Removed](#removed-5)
- [v1.27.0-beta.0](#v1270-beta0)
  - [Downloads for v1.27.0-beta.0](#downloads-for-v1270-beta0)
    - [Source Code](#source-code-6)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
    - [Container Images](#container-images-6)
  - [Changelog since v1.27.0-alpha.3](#changelog-since-v1270-alpha3)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-6)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-3)
    - [Feature](#feature-5)
    - [Documentation](#documentation-1)
    - [Failing Test](#failing-test-2)
    - [Bug or Regression](#bug-or-regression-6)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-6)
    - [Added](#added-6)
    - [Changed](#changed-6)
    - [Removed](#removed-6)
- [v1.27.0-alpha.3](#v1270-alpha3)
  - [Downloads for v1.27.0-alpha.3](#downloads-for-v1270-alpha3)
    - [Source Code](#source-code-7)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
    - [Container Images](#container-images-7)
  - [Changelog since v1.27.0-alpha.2](#changelog-since-v1270-alpha2)
  - [Changes by Kind](#changes-by-kind-7)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-4)
    - [Feature](#feature-6)
    - [Documentation](#documentation-2)
    - [Failing Test](#failing-test-3)
    - [Bug or Regression](#bug-or-regression-7)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-7)
    - [Added](#added-7)
    - [Changed](#changed-7)
    - [Removed](#removed-7)
- [v1.27.0-alpha.2](#v1270-alpha2)
  - [Downloads for v1.27.0-alpha.2](#downloads-for-v1270-alpha2)
    - [Source Code](#source-code-8)
    - [Client Binaries](#client-binaries-8)
    - [Server Binaries](#server-binaries-8)
    - [Node Binaries](#node-binaries-8)
    - [Container Images](#container-images-8)
  - [Changelog since v1.27.0-alpha.1](#changelog-since-v1270-alpha1)
  - [Changes by Kind](#changes-by-kind-8)
    - [API Change](#api-change-5)
    - [Feature](#feature-7)
    - [Bug or Regression](#bug-or-regression-8)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-8)
    - [Added](#added-8)
    - [Changed](#changed-8)
    - [Removed](#removed-8)
- [v1.27.0-alpha.1](#v1270-alpha1)
  - [Downloads for v1.27.0-alpha.1](#downloads-for-v1270-alpha1)
    - [Source Code](#source-code-9)
    - [Client Binaries](#client-binaries-9)
    - [Server Binaries](#server-binaries-9)
    - [Node Binaries](#node-binaries-9)
    - [Container Images](#container-images-9)
  - [Changelog since v1.26.0](#changelog-since-v1260-1)
  - [Changes by Kind](#changes-by-kind-9)
    - [Deprecation](#deprecation-3)
    - [API Change](#api-change-6)
    - [Feature](#feature-8)
    - [Documentation](#documentation-3)
    - [Failing Test](#failing-test-4)
    - [Bug or Regression](#bug-or-regression-9)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-5)
  - [Dependencies](#dependencies-9)
    - [Added](#added-9)
    - [Changed](#changed-9)
    - [Removed](#removed-9)

<!-- END MUNGE: GENERATED_TOC -->

# v1.27.3


## Downloads for v1.27.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes.tar.gz) | dd9174340ea3773db4c8a19d0a958fa22046863f270e31add0b7fad07ec285c6f2d09d335121697f6cd879c91589204b73b6631100c95c027bddbdef391fe5e0
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-src.tar.gz) | b3ecffc433df2dcaea9287e45da74f23c784f18bf22a33b65a196563077b01ba61b73f5761c16ae723ae821f27644f98c22be69ea16b8b14deb8b6e35d35b018

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-darwin-amd64.tar.gz) | fb8f708135406f1ba9116d2fecd6fadcc2e5be7e56c78e0620840352e236178835e4e162b76c8b545faf02700347f01792c2b9c83c6e3819a020e3ba0aef7afe
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-darwin-arm64.tar.gz) | 1c5b9ba8d9e41b61d519ede27c7d65320ababa7ac922f4b425553922cb319d825e0ca8a4bb38440408c8f99395c981f682887e7a0d8e20217feafafc2cbfa75c
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-linux-386.tar.gz) | 6ccc8cce4c26eacd27499b20a0017b4059aa133438b26f0576828d57c1810095f5fae67ca023047d44cdc9535605cf95a084af509138087616cd8d56a7e5c2fd
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-linux-amd64.tar.gz) | 14339c478f3c2d169d2e9fc387a62b32ba86be6cc9d9fc73f50bfe3a2225bbefd5463a09f665155ec36b04e52676831e7b010a1fcb9158e7f10a55dde393502d
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-linux-arm.tar.gz) | d40738d863188768089cf14fe31c060f578182283f8d4070e279cdabd9eccaf5a8e33f20025281e394de95a76ddc4a9835a6d1eac2b131d64f137a8a5d4cd667
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-linux-arm64.tar.gz) | 79aa1ea8d763c5258d43499a55ed4c88cf9aec36353246c5de797bc399023ffcb110c4b69da419b085a259b8b7253de2956480e6734a9ce1edd674821756cb54
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-linux-ppc64le.tar.gz) | c9e89f6032853874dce84df467c9f8ade7800af2c7b29fb070628e16350ba49b086520903508c711f6970b3fe37a3e88065be705d8950c024c006929abec6459
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-linux-s390x.tar.gz) | 927f54417e2e177ba811b08db45677a7a5693c99da4eaa268fc2026d68769d49bb036c1b4dc2d376a2080f17f207399d9a1e83bcf17c747647bd86aa3e1bab77
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-windows-386.tar.gz) | 78e6a307237f66386856a217ed82a63e2d1f9a1b674fe1f5fd26d6169193490d0aaeb8597f2ac9ea876ff984ca51e4f6810cb6587aff3d6af4eaf1a4b8590330
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-windows-amd64.tar.gz) | 83252ef0c7491a637fc45b696c35cdafd3bd168608c1cb6fa0cc380b20dd68e456a63615d5b42074c3a8a5e1f8cc988e746793b598b5e658f1d598f8da29b2d4
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-client-windows-arm64.tar.gz) | c4f364d16a3fdf3885d2176beb058b85f8f1f603d25c0e87b2c97c122543c7f1f0f6c1d7377a145a3cf893ae3af864779f555f0bdfddfa8d6cf2a5d442a57e1f

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-server-linux-amd64.tar.gz) | 4493a9c8e64579f0904c5dddb5967397f7ad9f8ec1d16d34eaa9692a716170f8392d414ee8d2adce0effdfc875bce5e70567df51d538d520eb4d284ff538ed3c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-server-linux-arm64.tar.gz) | 06f0440bf4fdfe19a27d4ee5e2ae71df9d8697b37422133a487b15ddb3861fde4e705367374daa1b726adea432d4e23cab3477741583dd5f9fdc7ddc4633081b
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-server-linux-ppc64le.tar.gz) | 6886265f0839ecd83988dbe535d0c15613f828697b2a2948c79fcb5eabc7f4177b08d9e58a93c80d6f8fb2f463688ab043a4d0bf354f4bc410785d2d2e5b94a4
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-server-linux-s390x.tar.gz) | a7a595f914393f776809555f22831d7c5ca91d2de117b69f392ae6f7f3611199074abd373a973b5d4d5aaddfcce60c17ba9ecb8eeeeabfe0a4ccbc7c6f0735c7

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-node-linux-amd64.tar.gz) | 5950f42effc3b48697142cc6012de0921a575252e1eb860eb89941be7f22ef9005d590fd825f0b2e9025583a7912ecae10f3a848932d16981afc9a36dac4d337
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-node-linux-arm64.tar.gz) | 9dedf1d677a9a57d8ec28d09ae70ea9bf5deb4869a33e15391978347e7f1e9d51ac4fcd82e0411803670d84d161104575fd84e4ea90458c877b9247c748d12b4
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-node-linux-ppc64le.tar.gz) | 54c1018641d2ffb075fa6c3674a0c69a8509332f13db569c937c649c4c678db418512266d6373aca9892776fc090e060a51c1f6e54e4ab8018f44c3a6bfad4d4
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-node-linux-s390x.tar.gz) | 3d0e771b812c8aff468cced3bb29619cc15ee2f2bd5980c8bb5880391e0076eab54bfea6bdd77953ed929462024349ba5fc07edad85126ae5a6e81c399e8b333
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.3/kubernetes-node-windows-amd64.tar.gz) | 3614c4df2074714a74ce392f3f14ff91b15d97a7759cc33755c0c1f0b96e728cef04d8e55068911a64611a46cc40f0d25bee2ac974e4cc78b7428b2f1e1f5ecc

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.2

## Important Security Information

This release contains changes that address the following vulnerabilities:

### CVE-2023-2728: Bypassing enforce mountable secrets policy imposed by the ServiceAccount admission plugin

A security issue was discovered in Kubernetes where users may be able to launch containers that bypass the mountable secrets policy enforced by the ServiceAccount admission plugin when using ephemeral containers. The policy ensures pods running with a service account may only reference secrets specified in the service account's secrets field. Kubernetes clusters are only affected if the ServiceAccount admission plugin and the `kubernetes.io/enforce-mountable-secrets` annotation are used together with ephemeral containers.

**Note**: This only impacts the cluster if the ServiceAccount admission plugin is used (most cluster should have this on by default as recommended in https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/#serviceaccount), the `kubernetes.io/enforce-mountable-secrets` annotation is used by a service account (this annotation is not added by default), and Pods are using ephemeral containers.

**Affected Versions**:
  - kube-apiserver v1.27.0 - v1.27.2
  - kube-apiserver v1.26.0 - v1.26.5
  - kube-apiserver v1.25.0 - v1.25.10
  - kube-apiserver <= v1.24.14

**Fixed Versions**:
  - kube-apiserver v1.27.3
  - kube-apiserver v1.26.6
  - kube-apiserver v1.25.11
  - kube-apiserver v1.24.15


**CVSS Rating:** Medium (6.5) [CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:U/C:H/I:H/A:N](https://www.first.org/cvss/calculator/3.1#CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:U/C:H/I:H/A:N)

## Changes by Kind

### Feature

- Kubernetes is now built with Go 1.20.5 ([#118553](https://github.com/kubernetes/kubernetes/pull/118553), [@puerco](https://github.com/puerco)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Release, Storage and Testing]

### Bug or Regression

- Add DisruptionTarget condition to the pod preempted by Kubelet to make room for a critical pod ([#118219](https://github.com/kubernetes/kubernetes/pull/118219), [@mimowo](https://github.com/mimowo)) [SIG Node and Testing]
- Fixes a bug at kube-apiserver start where APIService objects for custom resources could be deleted and recreated. ([#118104](https://github.com/kubernetes/kubernetes/pull/118104), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- If `kubeadm reset` finds no etcd member ID for the peer it removes during the `remove-etcd-member` phase, it continues immediately to other phases, instead of retrying the phase for up to 3 minutes before continuing. ([#117948](https://github.com/kubernetes/kubernetes/pull/117948), [@dlipovetsky](https://github.com/dlipovetsky)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where the static pod changes detection logic is inconsistent with kubelet ([#118069](https://github.com/kubernetes/kubernetes/pull/118069), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: fix etc version support for Kubernetes v1.27 ([#118307](https://github.com/kubernetes/kubernetes/pull/118307), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.27.2


## Downloads for v1.27.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes.tar.gz) | c46c9c1c4cdb0b1532630ce0e01295c7185f725e494d4fd190bae0540283c679b1c8b0a1ad1f0f5d320ddbf439e5fdd6f925700080cdf810158e1f41b8c5d9c9
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-src.tar.gz) | b9be38a5506071a362864661b369c71b0a02e66df0a77a2afc68040fa9634751a189e3c6c94771aee3e17b50a73228ad992a08f31cf4b322ebd7003e7676d381

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-darwin-amd64.tar.gz) | 7e4c0a207e505f6966999e0efb293c0e885d5975ad02f8e534b60ab1a94e0fdbcefd72c18fd536bb23e9356735098dd3fb85c6034e3ec07879f9511deb254f1b
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-darwin-arm64.tar.gz) | 41a51f588a9c19d0377921b66c21d7b406244aca4fbd32c0d7dcdd9b1cf80712c8f26a1134c2ae1d57612025b943e87a7bb55ee01c22838c0deb2754cf4a43cd
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-linux-386.tar.gz) | 729310d48d34fd21805869f849492227d3a74d1feafa4969d2aa5e0336c85d51f379865eda7b20c92b2f5122094884de5947a715f2fc9b6cb32e8a4e79dcd16c
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-linux-amd64.tar.gz) | cc34cffb3ec65a1b29dc3998341c8317dd1bf34d45b230a2379b1676d4d9a600cb662cde7caa7c8253e4cf2320d40b9581f97c0a04ac81037643b4fd105c6103
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-linux-arm.tar.gz) | b85927b9ff2f5871dac6814800c0fda43c4e69c27dd5c9d5cb3c73551c147ad2501675ffc5775cb70af4b643e88e784d034c3eec714de66c7ddf163a2ae4f500
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-linux-arm64.tar.gz) | cdf09ad3150c702e84c22158e95f164cffcbfa5e06af65e33dafada0d0d00fd6c160f41eb72d7966b0659e49705f7197d92dc3dc7153cf907fcdf318071138bc
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-linux-ppc64le.tar.gz) | 7219c79d43cc57a0866c854183dbed2629866e4ce081b62eef6c3034094bf0d3143e9e2eb7cf819a2f49bd98566cf4ac56cc9f2989f4c49906a71e2df68767e4
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-linux-s390x.tar.gz) | 9a5c0d13732ebf2d69f714ee953cc57f1c7ca2a27cb26333336a0da225414a96976f6a383cbe89aee80bcdd47a59dc17784acce377f656a6f959ed65f638a82d
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-windows-386.tar.gz) | 7bad7610f5a000cf40f68451d4cf94395d43907271aa98132cb6a52eae541e25cce7f40b5dfb1b45c79da5bbf54ce49cfbcd04f819635e16d91e03c63b48b8f4
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-windows-amd64.tar.gz) | 3d1c4f023867e8289d19dba43d702d40f3a8d8583e2c436e447af43127da9b0e90b5ca4ac055c3256c92d8fcaaa3734f0a83039480b35a012aed86ecd377da59
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-client-windows-arm64.tar.gz) | 0b08b36d4869b6b1de0314bb365ae45f85719297088f12b723a535a61f7b2c648969c12a4e1ecbd29a6deb804551815ed21c3b8ae9ed6813aa26d625723a273e

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-server-linux-amd64.tar.gz) | 53f1533aa8f493ebbdfb07ba59eaf971cf865c60d1ac9a5ad9f61e6d5f670e9d86e0dc70c6d3057953da2968d100de8d8bf50d5863ad2decb69c397aa6f185b9
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-server-linux-arm64.tar.gz) | 66466de2b1b5ad7ce09fba95da00d3451ae13b28b89755a64cc4e18e1254c5dfed290df5f8509f312396679bcc90eec98eb84e333ea9bd206ecb8bf00eeeba71
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-server-linux-ppc64le.tar.gz) | 283dd8c6391d62b1f11102ce3a252d78b1dd3268dd2b8c5f08276c9c764ced6f0f8e8056b5d302045c464efc063a81d815e6fc3f804b997770b40bc1b2a89f8c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-server-linux-s390x.tar.gz) | ea30de775e794eb738a3c10c730c0e291ae1460fdaba984d4eb00bf52b552f192608e8213b382ac8161f1f13486ddf13b7f20e4f5838a1e38f08da9288c01a3c

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-node-linux-amd64.tar.gz) | 2a05c3ebcec8bce9ca2a7c835617f0b85dbf11a07d39c5b002b5389a45465807c437dd22d975c60e680ee34f3bc00e460d848927a8fa2c1543ec97fb66f50477
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-node-linux-arm64.tar.gz) | 9d6c45fb54c01ac9a106d356d8c9ed1c6564f4b86bb1ba55ced628bbbf5c4fb1f78d68a15c3055680c6dff6c85a32726300d09b7a85bdcb7ba263a434b148826
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-node-linux-ppc64le.tar.gz) | b029fe744619f9649e42c80ca2f0bb14ae72c934d4687100ee7f041cbcd72cc5567ae01acef9e2c7b5c579228b2af9a419bdbf2af64420754a5f6049cdd391bb
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-node-linux-s390x.tar.gz) | d3319d9a4a205cd1fa9da590407fe2be3b149c50e77422ff2b2f1e803c24e0d496fdc89d16c658fbef7a0bc59d1b0e295dfa5354ce3c3c5d9a6749e60e1580ee
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.2/kubernetes-node-windows-amd64.tar.gz) | a745fe1b46ec6c3bd27e72c2774a01fd53c27361de6ad7281c2adeaf88ab59ec725d60dc99a4de5f3579428ef4e923860a85143f14b51590ce43bfdee7a36a10

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.1

## Changes by Kind

### API Change

- Added error handling for seccomp localhost configurations that do not properly set a localhostProfile ([#117020](https://github.com/kubernetes/kubernetes/pull/117020), [@cji](https://github.com/cji)) [SIG API Machinery and Node]
- Fixed an issue where kubelet does not set case-insensitive headers for http probes. (#117182, @dddddai) ([#117324](https://github.com/kubernetes/kubernetes/pull/117324), [@dddddai](https://github.com/dddddai)) [SIG API Machinery, Apps and Node]
- Revised the comment about the feature-gate level for PodFailurePolicy from alpha to beta ([#117815](https://github.com/kubernetes/kubernetes/pull/117815), [@kerthcet](https://github.com/kerthcet)) [SIG Apps]

### Feature

- Kubernetes is now built with Go 1.20.4 ([#117773](https://github.com/kubernetes/kubernetes/pull/117773), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]

### Failing Test

- Allow Azure Disk e2es to use newer topology labels if available from nodes ([#117216](https://github.com/kubernetes/kubernetes/pull/117216), [@gnufied](https://github.com/gnufied)) [SIG Storage and Testing]

### Bug or Regression

- CVE-2023-27561 CVE-2023-25809 CVE-2023-28642: Bump fix runc v1.1.4 -> v1.1.5 ([#117242](https://github.com/kubernetes/kubernetes/pull/117242), [@haircommander](https://github.com/haircommander)) [SIG Node]
- During device plugin allocation, resources requested by the pod can only be allocated if the device plugin has registered itself to kubelet AND healthy devices are present on the node to be allocated. If these conditions are not sattsfied, the pod would fail with `UnexpectedAdmissionError` error. ([#117719](https://github.com/kubernetes/kubernetes/pull/117719), [@swatisehgal](https://github.com/swatisehgal)) [SIG Node and Testing]
- Fallback from OpenAPI V3 to V2 when the OpenAPI V3 document is invalid or incomplete. ([#117980](https://github.com/kubernetes/kubernetes/pull/117980), [@seans3](https://github.com/seans3)) [SIG CLI]
- Fix bug where `listOfStrings.join()` in CEL expressions resulted in an unexpected internal error. ([#117596](https://github.com/kubernetes/kubernetes/pull/117596), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fix incorrect calculation for ResourceQuota with PriorityClass as its scope. ([#117825](https://github.com/kubernetes/kubernetes/pull/117825), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG API Machinery]
- Fix performance regression in scheduler caused by frequent metric lookup on critical code path. ([#117617](https://github.com/kubernetes/kubernetes/pull/117617), [@tosi3k](https://github.com/tosi3k)) [SIG Scheduling]
- Fix: the volume is not detached after the pod and PVC objects are deleted ([#117236](https://github.com/kubernetes/kubernetes/pull/117236), [@cvvz](https://github.com/cvvz)) [SIG Storage]
- Fixed a memory leak in the Kubernetes API server that occurs during APIService processing. ([#117310](https://github.com/kubernetes/kubernetes/pull/117310), [@enj](https://github.com/enj)) [SIG API Machinery]
- Fixes a race condition serving OpenAPI content ([#117708](https://github.com/kubernetes/kubernetes/pull/117708), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Instrumentation and Node]
- Fixes a regression in kubectl and client-go discovery when configured with a server URL other than the root of a server. ([#117685](https://github.com/kubernetes/kubernetes/pull/117685), [@ardaguclu](https://github.com/ardaguclu)) [SIG API Machinery]
- Fixes bug where an incomplete OpenAPI V3 document can cause a nil-pointer crash.
  Ensures fallback to OpenAPI V2 endpoint for errors retrieving OpenAPI V3 document. ([#117918](https://github.com/kubernetes/kubernetes/pull/117918), [@seans3](https://github.com/seans3)) [SIG CLI]
- Kubeadm: fix a bug where file copy(backup) could not be executed correctly on Windows platform during upgrade ([#117861](https://github.com/kubernetes/kubernetes/pull/117861), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet terminates pods correctly upon restart, fixing an issue where pods may have not been fully terminated if the kubelet was restarted during pod termination. ([#117433](https://github.com/kubernetes/kubernetes/pull/117433), [@bobbypage](https://github.com/bobbypage)) [SIG Node and Testing]
- Number of errors reported to the metric `storage_operation_duration_seconds_count` for emptyDir decreased significantly because previously one error was reported for each projected volume created. ([#117022](https://github.com/kubernetes/kubernetes/pull/117022), [@mpatlasov](https://github.com/mpatlasov)) [SIG Storage]
- Resolves a spurious "Unknown discovery response content-type" error in client-go discovery requests by tolerating extra content-type parameters in API responses ([#117637](https://github.com/kubernetes/kubernetes/pull/117637), [@seans3](https://github.com/seans3)) [SIG API Machinery]
- Reverted NewVolumeManagerReconstruction and SELinuxMountReadWriteOncePod feature gates to disabled by default to resolve a regression of volume reconstruction on kubelet/node restart ([#117752](https://github.com/kubernetes/kubernetes/pull/117752), [@liggitt](https://github.com/liggitt)) [SIG Storage]
- Static pods were taking extra time to be restarted after being updated.  Static pods that are waiting to restart were not correctly counted in `kubelet_working_pods`. ([#116995](https://github.com/kubernetes/kubernetes/pull/116995), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- [KCCM] service controller: change the cloud controller manager to make `providerID` a predicate when synchronizing nodes. This change allows load balancer integrations to ensure that  the `providerID` is set when configuring
  load balancers and targets. ([#117450](https://github.com/kubernetes/kubernetes/pull/117450), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider and Network]

### Other (Cleanup or Flake)

- A v2-level info log will be added, which will output the details of the pod being preempted, including victim and preemptor ([#117214](https://github.com/kubernetes/kubernetes/pull/117214), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Scheduling]
- Structured logging of NamespacedName was inconsistent with klog.KObj. Now both use lower case field names and namespace is optional. ([#117238](https://github.com/kubernetes/kubernetes/pull/117238), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture and Instrumentation]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/opencontainers/runc: [v1.1.4 → v1.1.6](https://github.com/opencontainers/runc/compare/v1.1.4...v1.1.6)
- k8s.io/kube-openapi: 15aac26 → 8b0f38b
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.1.1 → v0.1.2

### Removed
_Nothing has changed._



# v1.27.1


## Downloads for v1.27.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes.tar.gz) | 01e0cc312fa04847174bf63eb888b0247ee92788801b77d642480226b112be6e3989169b9a51209187878ac6dd9c7475fd6550f16dc34b8bc52ae2f0d82a8830
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-src.tar.gz) | e5b55fe1e6262c60cf14229ac6dabc12b6108c7c75a454aa53919fa779bd9566412bcb31122405faa7326a19b3506a2e21904225bae1d63b23c0479ef909324d

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-darwin-amd64.tar.gz) | a85b973c88f5684d6ea73753b8c6b99b3e64e0cd2507db24b2bd7c25e22b55cf725c6e7ee80795884a34135434bb19b676e3572d587a13d2400ba7ddc5ae4935
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-darwin-arm64.tar.gz) | 83aa7154c08bbe92c6850e1c9669f6a074019ec42253b45cf0a29d58526ce7f3c3845f9504266c01cceef5013f3977eed6bd5c7a4926ee3433194c8ea8984e6f
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-linux-386.tar.gz) | 0723e0b8a1fdc6aea6028bc2f4e25addd20dca83060c12b8b7869dbd49e9eb852264957383dcf6069b542d4db7a2e3a5c5408091d58709d676afa899df25bce2
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-linux-amd64.tar.gz) | 98caa662a63d7f9ba36761caaf997be4d214ea2b921a4387965a67d168b52ea29ae9185de20192f7b4b9169a887beb19d22e5776ff0bb0b68907e177b11a8043
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-linux-arm.tar.gz) | 773171a17712112850332effaf64c901b9eace6b52c003eed79032178bbbbc2b9f7ed6725bf95533e574511c70e75cd03ff1b3a1e22bcc3871d50887f04b7121
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-linux-arm64.tar.gz) | 9dd09ca0738b1b7e287271cf16f5527d3a1abc7878e4259dc4208f74e82e3db16e5fed07a3dca75550ca531be0c504f6942841ab65e67687a94648e1b0e382a5
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-linux-ppc64le.tar.gz) | 976499635c9b7507350edeebff6d997bbff8c9f66aecea55d549eeab51b89bc3e415a703fc49a37f284f8180055d1d36d0722932d7d19729aabb6ed39c08b8ff
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-linux-s390x.tar.gz) | ad642b631eab5a68d8c88670f7880d6cafa2fdfb04e6e69e50a7c264b309ed2b04ff63b045f5133db9c3aecd99e7ac556577a74d57a64b705f3c6367d998c323
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-windows-386.tar.gz) | bc3a685dd07085119672a6ccabc2dd832a990cf50304eea8396771006ef0bcb1fc8e08b6db9d2f455dc64c5f3e9c48fdb2ff19ec7802f35257742465fbc083a7
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-windows-amd64.tar.gz) | 55d2a9e5ca066166592d778697d657eb04d828ffd6e15b177a399b44574dcae972d109e83bc2e33f0effe679e61538d5b1b4fea66ec1d812e678a9b1c50494fc
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-client-windows-arm64.tar.gz) | adc10b48334f2a4fae6d7ace0e1485b8a874f98187b3b5491ca056b25f9096b58a0e3e37f096c4e9045bd7c56aa7efb4d2544570153864288095ddf6d0515172

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-server-linux-amd64.tar.gz) | 0752a63510a6d0ae06d8e24e42faa1641e093c25061f60b4c8355f735788ddd4bd25ae2a47a796064b6eb2ea15f0d451852fddc3c1b82b0e1afd279df700cea2
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-server-linux-arm64.tar.gz) | 63d086447a7abcd37a7d4e8d36609dc1a8217e6a7d2f45efd6bf35444321bf96f181d621d7a12224e55034fae1ece62a6c3109204dfd3b40577afb7c61a3ead7
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-server-linux-ppc64le.tar.gz) | 0b477fb3af66976f7aafba0b97c089e5dda8c21fc314ba3fb2d54ed1ff453389af8d3aba6bbe14fdd8db02f1a309bff2291abbdbdd64b2174878bec6997a889b
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-server-linux-s390x.tar.gz) | bf198f3099663ead089c42221f60d4dee53c6ac0cefab37e316d2b6449a46a1708426782aff946f6cb735ab7141f61df1e33144115845aa277c45b1ea8741a6c

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-node-linux-amd64.tar.gz) | 484993a2478aa7d734fe711789f661b74152a96eff540b883e8603a92d91b89d870c4bfd0b482e72fd0ab2a95b7766725a0e8627676e99d051db0bf7c97d5791
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-node-linux-arm64.tar.gz) | 1d0d9a393b513f32e35fd609fbdd61179402421674a51ceb25b6516f2d5a5d596a451a285a038e4fe31e175727b75e30ff2e9bb368a7aa82b363018634b55a6d
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-node-linux-ppc64le.tar.gz) | 00c40568cc8e255b557f9a69f704f393d9ad8cfce75aaadc4df5485551106064d62fa58a6bb8a2bacd4cc363c09424eeb0acc156fe7172df057df46f4cc57109
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-node-linux-s390x.tar.gz) | 0615a6be0ea7ca1c90418690633013872f5f56404a6eecfedb677276b2653e22f600ad3863c80b592ff7ff2107fc4ce407ab526793464abcb0f93b65d671f6dc
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.1/kubernetes-node-windows-amd64.tar.gz) | 87ab85b2d784124c945c4b92ff037ecdc12800176c22d7c06d38285386d32b3fc4fea0b00d6dcdd3db77f4b67123aaeb1dc62cca6c9ff381c0d2155ee4c1ec4e

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.0

## Changes by Kind

### Bug or Regression

- Fixes a regression in 1.27.0 that resulted in "missing metadata in converted object" errors when modifying objects for multi-version custom resource definitions with a conversion strategy of `None`. ([#117305](https://github.com/kubernetes/kubernetes/pull/117305), [@ncdc](https://github.com/ncdc)) [SIG API Machinery]
- Known issue: fixed that the PreEnqueue plugins aren't executed for Pods proceeding to activeQ through backoffQ. ([#117194](https://github.com/kubernetes/kubernetes/pull/117194), [@sanposhiho](https://github.com/sanposhiho)) [SIG Release and Scheduling]
- Setting a mirror pod's phase to Succeeded or Failed can prevent the corresponding static pod from restarting due mutation of a Kubelet cache. ([#116482](https://github.com/kubernetes/kubernetes/pull/116482), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.27.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.27.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes.tar.gz) | `78dbb72f270ab70d0ad70d2da6727eed64bdc54a11892fd6c2157882865f93ab41fedf5fced2f3e71dc0eda5679d06884c262a7960277face4510eed30a3678e`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-src.tar.gz) | `4080d2452ff4fd316a823c1c495e7e9a39d364e24225020a91bf0bc0289c3ef90ade746ef5a05172d6e355af9014cbddf144ca71839ec65fc57f3eaf553fb7ab`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-darwin-amd64.tar.gz) | `faa0e340f1829ba694326c6ff71f8527249af03d8d78f784289be4122b6ceb0829fa70ee1eab25f64bbb9f5972ae30f3cfdfefe617ce3360b2897d4f6259bd81`
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-darwin-arm64.tar.gz) | `9c4fe911e41ab9c355d39b21d77372bd2a070cc376fdfceac362eb6cc3e8f616754cc61593ea140030a81961b40fa6344b7628d7a4edf7e6dcdef29711bbd064`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-linux-386.tar.gz) | `ba522302624ac7b3a9e5c1a5c80857bdde4c47b44394dbfa8da597ee07b2e1975409e8eac514516329826f593fa82d143a03185ef3c30a97cb1f8011ffb96060`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-linux-amd64.tar.gz) | `3ea3b4a866815cc08f1897771d63bf4e4f75b481e1d70417e34581d079a58b647b077382a264224acb52e6a76474d6e92efd22a0d4f7fdfde0c244006beef76a`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-linux-arm.tar.gz) | `5fd69b567ab835b35b8156c66eec02ee109f731acf7d68250b05a1f43a56458be68654f95107cd28859b4b8e73d5f64c78aca2f4b1dc74fff3ca8d942c60d2db`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-linux-arm64.tar.gz) | `f20e579ab71b1cdace22bec0a11314ec44534f0e7040a436c63eb18a47d839e070e5134917ef2b531fe7b8bfee12133fa14de4dac7c0ac7798b4d9fa5679f193`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-linux-ppc64le.tar.gz) | `c56a2d021b1a99fde0871bbe8e71427b8c4f03847e2bf6cbf526a71f6d7d1060481bd0f00d7dce2bd8afa1c969e02422ac1a2283ab58facd3db43f0713c10212`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-linux-s390x.tar.gz) | `4ad879e2ab2b952cc0fdfcd738b6264db60b72174057947737ab07f40dd0c4c727fb042c24323be3accaf8fbc320973821c915fd1bb3c4ea8a22eb16c03ce4a3`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-windows-386.tar.gz) | `befab85193ce017c647b391606d45d3626e71bf7ea6bbca7f955985e0f505a9c8ca27898ee41c4f3124b7a3788b4a4eab602994415b24b8b0bfc154b938c547e`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-windows-amd64.tar.gz) | `0fbba06f00713c32c74d9b62733dfb83a597e3a33ee62bfb3a93de7cd883c460a0c56f25cd1577dd7923ef73312788d9b805020297fcf784722783ac1890253f`
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-client-windows-arm64.tar.gz) | `b7edbb25dbbf5b0bd9839f93d43f08262cf5f6e138599c034da0ff402c763a0cff18c1e9b42631d250389ff6b865dec4aa35b577fac75a51e65c825ec8efe234`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-server-linux-amd64.tar.gz) | `9726ba173084adade1c1b0de014ccedc5dc5317a80076cbf20d15fdcd6296dd1e9efcf1b1349456757a5c186fa52293f60411397cb6c79765adff335391add9e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-server-linux-arm64.tar.gz) | `657726cd4ae93a9696371717a280689af76c488586c49273086bef4e712228025c6e179c2a5c93b8a33640ac42347dd821053485659f9383dbb1b3e2a17f022c`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-server-linux-ppc64le.tar.gz) | `2ff2464453ca8ca2e9e4a024ad730c12fa506379b4a7bd749431fe64ddb13c2dccea05c37dba119799940eac2dc57635e9d70b908d1786a3cdc031a5b70504a5`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-server-linux-s390x.tar.gz) | `44ce8faa8710832593b656e3b053207e05def556ad821b8e08e0c2f33b73f280a455fbef933ea70e9efbe8a085ef7deb47139d4e9af43417d8242029a2b60c35`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-node-linux-amd64.tar.gz) | `812f5adfafe778200558678af6510f9f315f75b46f7bb4482e92b57d1bed08c4b7f236850bf8e4dcac7018879736d614fc482e3641da06c6f8d0554af4f4ef45`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-node-linux-arm64.tar.gz) | `a5e353205a93ebaade50dfd652ee5623b28ee4f6fd8ca949fb2303d708468026ba66c10b70f1761f4099706baad8959993a9ec0053259b94b5f4793aeda27adb`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-node-linux-ppc64le.tar.gz) | `b5dbcf8131bad7ef897c64ac482599ac3bedc99e5c211d189e0566543a13c89d0812d7a7b1e4e9655d8d884ee24dc553616c96cf74df19f4d2cce0ea552015ce`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-node-linux-s390x.tar.gz) | `81662d6e14a7500bf2714ae3c0b9070031ea5ef2628c84aaf2a8fe96fed07a52c3677babc74247160cb71ce1fc77b728f549f2c18dfc7dc6a65dfadb7ec17cd7`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0/kubernetes-node-windows-amd64.tar.gz) | `a8b1a53ba6ee416fb9939961d8290ae1f5e0c21117f1cd6cebbc9ba01cafa235730a2887fd91b92552d07eec78aeb65aefac92899eaf9b2f4f195c61f20d05d7`

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.
name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.26.0

## Known Issues

### The PreEnqueue extension point doesn't work for Pods going to activeQ through backoffQ

In v1.26.0, we've found the bug that the PreEnqueue extension point doesn't work for Pods going to activeQ through backoffQ. 
It doesn't affect any of the vanilla Kubernetes behavior, but, may break custom PreEnqueue plugins.

The cause PR is [reverted](https://github.com/kubernetes/kubernetes/pull/117194) by v1.26.1.

## Urgent Upgrade Notes 

### (No, really, you MUST read this before you upgrade)

- 'The `IPv6DualStack` feature gate for external cloud providers was removed.
  (The feature became GA in 1.23 and the gate was removed for all other
  components several releases ago.) If you were still manually
  enabling it you must stop now.' ([#116255](https://github.com/kubernetes/kubernetes/pull/116255), [@danwinship](https://github.com/danwinship))
- Give terminal phase correctly to all pods that will not be restarted.

  In particular, assign Failed phase to pods which are deleted while pending. Also, assign a terminal
  phase (Succeeded or Failed, depending on the exit statuses of the pod containers) to pods which
  are deleted while running.
  
  This fixes the issue for jobs using pod failure policy (with JobPodFailurePolicy and PodDisruptionConditions
  feature gates enabled) that their pods could get stuck in the pending phase when deleted.

  Users who maintain controllers which relied on the fact that pods with RestartPolicy=Always
  never enter the Succeeded phase may need to adapt their controllers. This is because as a consequence of
  the change pods which use RestartPolicy=Always may end up in the Succeeded phase in two scenarios: pod
  deletion and graceful node shutdown. ([#115331](https://github.com/kubernetes/kubernetes/pull/115331), [@mimowo](https://github.com/mimowo)) [SIG Cloud Provider, Node and Testing]
- The in-tree cloud provider for AWS (and the EBS storage plugin) has now been removed. Please use the external cloud provider and CSI driver from https://github.com/kubernetes/cloud-provider-aws instead. ([#115838](https://github.com/kubernetes/kubernetes/pull/115838), [@torredil](https://github.com/torredil)) [SIG API Machinery, Apps, Architecture, Auth, CLI , Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Release, Scheduling, Storage, and Testing]

## Changes by Kind

### Deprecation

- Added a [warning](https://k8s.io/blog/2020/09/03/warnings/) response when handling requests that set the deprecated `spec.externalID` field for a Node. ([#115944](https://github.com/kubernetes/kubernetes/pull/115944), [@SataQiu](https://github.com/SataQiu)) [SIG Node]
- Added warnings to the Services API. Kubernetes now warns for Services in the case of:
  - IPv4 addresses with leading zeros
  - IPv6 address in non-canonical format (RFC 5952) ([#114505](https://github.com/kubernetes/kubernetes/pull/114505), [@aojea](https://github.com/aojea))
- Support for the alpha seccomp annotations `seccomp.security.alpha.kubernetes.io/pod` and `container.seccomp.security.alpha.kubernetes.io` were deprecated since v1.19, now have been completely removed. The seccomp fields are no longer auto-populated when pods with seccomp annotations are created. Pods should use the corresponding pod or container `securityContext.seccompProfile` field instead. ([#114947](https://github.com/kubernetes/kubernetes/pull/114947), [@saschagrunert](https://github.com/saschagrunert))
- The `SecurityContextDeny` admission plugin is going deprecated and will be removed in future versions. ([#115879](https://github.com/kubernetes/kubernetes/pull/115879), [@mtardy](https://github.com/mtardy))

### API Change

- A fix in the `resource.k8s.io/v1alpha1/ResourceClaim` API avoids harmless (?) ".status.reservedFor: element 0: associative list without keys has an element that's a map type" errors in the apiserver. Validation now rejects the incorrect reuse of the same UID in different entries. ([#115354](https://github.com/kubernetes/kubernetes/pull/115354), [@pohly](https://github.com/pohly))
- A terminating pod on a node that is not caused by preemption no longer prevents `kube-scheduler` from preempting pods on that node
  - Rename `PreemptionByKubeScheduler` to `PreemptionByScheduler` ([#114623](https://github.com/kubernetes/kubernetes/pull/114623), [@Huang-Wei](https://github.com/Huang-Wei))
- API: resource.k8s.io/v1alpha1.PodScheduling was renamed to resource.k8s.io/v1alpha2.PodSchedulingContext. ([#116556](https://github.com/kubernetes/kubernetes/pull/116556), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Node, Scheduling and Testing]
- Added CEL runtime cost calculation into ValidatingAdmissionPolicy, matching the evaluation cost
  restrictions that already apply to CustomResourceDefinition.
  If rule evaluation uses more compute than the limit, the API server aborts the evaluation and the
  admission check that was being performed is aborted; the `failurePolicy` for the ValidatingAdmissionPolicy
  determines the outcome. ([#115747](https://github.com/kubernetes/kubernetes/pull/115747), [@cici37](https://github.com/cici37))
- Added `auditAnnotations` to `ValidatingAdmissionPolicy`, enabling CEL to be used to add audit annotations to request audit events.
  Added `validationActions` to `ValidatingAdmissionPolicyBinding`, enabling validation failures to be handled by any combination of the warn, audit and deny enforcement actions. ([#115973](https://github.com/kubernetes/kubernetes/pull/115973), [@jpbetz](https://github.com/jpbetz))
- Added `messageExpression` field to `ValidationRule`. ([#115969](https://github.com/kubernetes/kubernetes/pull/115969), [@DangerOnTheRanger](https://github.com/DangerOnTheRanger))
- Added `messageExpression` to `ValidatingAdmissionPolicy`, to set custom failure message via CEL expression. ([#116397](https://github.com/kubernetes/kubernetes/pull/116397), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery]
- Added a new IPAddress object kind
  - Added a new ClusterIP allocator. The new allocator removes previous Service CIDR block size limitations for IPv4, and limits IPv6 size to a /64 ([#115075](https://github.com/kubernetes/kubernetes/pull/115075), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Auth, CLI, Cluster Lifecycle, Network and Testing]
- Added a new alpha API: ClusterTrustBundle (`certificates.k8s.io/v1alpha1`).
  A ClusterTrustBundle may be used to distribute [X.509](https://www.itu.int/rec/T-REC-X.509) trust anchors to workloads within the cluster. ([#113218](https://github.com/kubernetes/kubernetes/pull/113218), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Auth and Testing]
- Added authorization check support to the CEL expressions of ValidatingAdmissionPolicy via a `authorizer`
  variable with expressions. The new variable provides a builder that allows expressions such `authorizer.group('').resource('pods').check('create').allowed()`. ([#116054](https://github.com/kubernetes/kubernetes/pull/116054), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]
- Added matchConditions field to ValidatingAdmissionPolicy and enabled support for CEL based custom match criteria. ([#116350](https://github.com/kubernetes/kubernetes/pull/116350), [@maxsmythe](https://github.com/maxsmythe))
- Added new option to the `InterPodAffinity` scheduler plugin to ignore existing
  pods` preferred inter-pod affinities if the incoming pod has no preferred inter-pod
  affinities. This option can be used as an optimization for higher scheduling throughput
  (at the cost of an occasional pod being scheduled non-optimally/violating existing
  pods preferred inter-pod affinities). To enable this scheduler option, set the
  `InterPodAffinity` scheduler plugin arg `ignorePreferredTermsOfExistingPods: true` ([#114393](https://github.com/kubernetes/kubernetes/pull/114393), [@danielvegamyhre](https://github.com/danielvegamyhre))
- Added the `MatchConditions` field to `ValidatingWebhookConfiguration` and `MutatingWebhookConfiguration` for the v1beta and v1 apis. 
  
  The `AdmissionWebhookMatchConditions` featuregate is now in Alpha ([#116261](https://github.com/kubernetes/kubernetes/pull/116261), [@ivelichkovich](https://github.com/ivelichkovich)) [SIG API Machinery and Testing]
- Added validation to ensure that if `service.kubernetes.io/topology-aware-hints` and `service.kubernetes.io/topology-mode` annotations are both set, they are set to the same value.Also Added deprecation warning if `service.kubernetes.io/topology-aware-hints` annotation is used. ([#116612](https://github.com/kubernetes/kubernetes/pull/116612), [@robscott](https://github.com/robscott))
- Added warnings about workload resources (Pods, ReplicaSets, Deployments, Jobs, CronJobs, or ReplicationControllers) whose names are not valid DNS labels. ([#114412](https://github.com/kubernetes/kubernetes/pull/114412), [@thockin](https://github.com/thockin))
- Adds feature gate `NodeLogQuery` which provides cluster administrators with a streaming view of logs using kubectl without them having to implement a client side reader or logging into the node. ([#96120](https://github.com/kubernetes/kubernetes/pull/96120), [@LorbusChris](https://github.com/LorbusChris))
- Api: validation of a `PodSpec` now rejects invalid `ResourceClaim` and `ResourceClaimTemplate` names. For a pod, the name generated for the `ResourceClaim` when using a template also must be valid. ([#116576](https://github.com/kubernetes/kubernetes/pull/116576), [@pohly](https://github.com/pohly))
- Bump default API QPS limits for Kubelet. ([#116121](https://github.com/kubernetes/kubernetes/pull/116121), [@wojtek-t](https://github.com/wojtek-t))
- Enabled the `StatefulSetStartOrdinal` feature gate in beta ([#115260](https://github.com/kubernetes/kubernetes/pull/115260), [@pwschuurman](https://github.com/pwschuurman))
- Enabled usage of `kube-proxy`, `kube-scheduler` and `kubelet` HTTP APIs for changing the logging
   verbosity at runtime for JSON output. ([#114609](https://github.com/kubernetes/kubernetes/pull/114609), [@pohly](https://github.com/pohly))
- Encryption of API Server at rest configuration now allows the use of wildcards in the list of resources.  For example, *.* can be used to encrypt all resources, including all current and future custom resources. ([#115149](https://github.com/kubernetes/kubernetes/pull/115149), [@nilekhc](https://github.com/nilekhc))
- Extended the kubelet's PodResources API to include resources allocated in `ResourceClaims` via `DynamicResourceAllocation`. Additionally, added a new `Get()` method to query a specific pod for its resources. ([#115847](https://github.com/kubernetes/kubernetes/pull/115847), [@moshe010](https://github.com/moshe010)) [SIG Node]
- Forbid to set matchLabelKeys when labelSelector is not set in topologySpreadConstraints ([#116535](https://github.com/kubernetes/kubernetes/pull/116535), [@denkensk](https://github.com/denkensk))
- GCE does not support LoadBalancer Services with ports with different protocols (TCP and UDP) ([#115966](https://github.com/kubernetes/kubernetes/pull/115966), [@aojea](https://github.com/aojea)) [SIG Apps and Cloud Provider]
- GRPC probes are now a GA feature. `GRPCContainerProbe` feature gate was locked to default value and will be removed in v1.29. If you were setting this feature gate explicitly, please remove it now. ([#116233](https://github.com/kubernetes/kubernetes/pull/116233), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- Graduated `Kubelet Topology Manager` to GA. ([#116093](https://github.com/kubernetes/kubernetes/pull/116093), [@swatisehgal](https://github.com/swatisehgal))
- Graduated `KubeletTracing` to beta, which means that the feature gate is now enabled by default. ([#115750](https://github.com/kubernetes/kubernetes/pull/115750), [@saschagrunert](https://github.com/saschagrunert))
- Graduated seccomp profile defaulting to GA.
  
  Set the kubelet `--seccomp-default` flag or `seccompDefault` kubelet configuration field to `true` to make pods on that node default to using the `RuntimeDefault` seccomp profile.
  
  Enabling seccomp for your workload can have a negative performance impact depending on the kernel and container runtime version in use.
  
  Guidance for identifying and mitigating those issues is outlined in the Kubernetes [seccomp tutorial](https://k8s.io/docs/tutorials/security/seccomp). ([#115719](https://github.com/kubernetes/kubernetes/pull/115719), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery, Node, Storage and Testing]
- Graduated the container resource metrics feature on `HPA` to beta. ([#116046](https://github.com/kubernetes/kubernetes/pull/116046), [@sanposhiho](https://github.com/sanposhiho))
- Implemented API streaming for the `watch-cache`
  
  When `sendInitialEvents` `ListOption` is set together with `watch=true`, it begins the watch stream with synthetic init events followed by a synthetic "Bookmark" after which the server continues streaming events. ([#110960](https://github.com/kubernetes/kubernetes/pull/110960), [@p0lyn0mial](https://github.com/p0lyn0mial))
- Introduced API for streaming.
  
  Added `SendInitialEvents` field to the `ListOptions`. When the new option is set together with `watch=true`, it begins the watch stream with synthetic init events followed by a synthetic "Bookmark" after which the server continues streaming events. ([#115402](https://github.com/kubernetes/kubernetes/pull/115402), [@p0lyn0mial](https://github.com/p0lyn0mial))
- Introduced a breaking change to the `resource.k8s.io` API in its `AllocationResult` struct. This change allows a kubelet plugin for the `DynamicResourceAllocation` feature to service allocations from multiple resource driver controllers. ([#116332](https://github.com/kubernetes/kubernetes/pull/116332), [@klueska](https://github.com/klueska))
- Introduces new alpha functionality to the reflector, allowing user to enable API streaming.
  
  To activate this feature, users can set the `ENABLE_CLIENT_GO_WATCH_LIST_ALPHA` environmental variable.
  It is important to note that the server must support streaming for this feature to function properly.
  If streaming is not supported by the server, the reflector will revert to the previous method
  of obtaining data through LIST/WATCH semantics. ([#110772](https://github.com/kubernetes/kubernetes/pull/110772), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- K8s.io/client-go/tools/record.EventBroadcaster: after Shutdown() is called, the broadcaster now gives up immediately after a failure to write an event to a sink. Previously it tried multiple times for 12 seconds in a goroutine. ([#115514](https://github.com/kubernetes/kubernetes/pull/115514), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- K8s.io/component-base/logs: usage of the pflag values in a normal Go flag set led to panics when printing the help message ([#114680](https://github.com/kubernetes/kubernetes/pull/114680), [@pohly](https://github.com/pohly)) [SIG Instrumentation]
- Kubeadm: explicitly set `priority` for static pods with `priorityClassName: system-node-critical` ([#114338](https://github.com/kubernetes/kubernetes/pull/114338), [@champtar](https://github.com/champtar)) [SIG Cluster Lifecycle]
- Kubelet: a "maxParallelImagePulls" field can now be specified in the kubelet configuration file to control how many image pulls the kubelet can perform in parallel. ([#115220](https://github.com/kubernetes/kubernetes/pull/115220), [@ruiwen-zhao](https://github.com/ruiwen-zhao)) [SIG API Machinery, Node and Scalability]
- Kubelet: changed `MemoryThrottlingFactor` default value to `0.9` and formulas to calculate `memory.high` ([#115371](https://github.com/kubernetes/kubernetes/pull/115371), [@pacoxu](https://github.com/pacoxu))
- Kubernetes components that perform leader election now only support using `Leases` for this. ([#114055](https://github.com/kubernetes/kubernetes/pull/114055), [@aimuz](https://github.com/aimuz))
- Migrated the `DaemonSet` controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging) ([#113622](https://github.com/kubernetes/kubernetes/pull/113622), [@249043822](https://github.com/249043822))
- New `service.kubernetes.io/topology-mode` annotation has been introduced as a replacement for the `service.kubernetes.io/topology-aware-hints` annotation.
  - `service.kubernetes.io/topology-aware-hints` annotation has been deprecated.
  - kube-proxy now accepts any value that is not "disabled" for these annotations, enabling custom implementation-specific and/or future built-in heuristics to be used. ([#116522](https://github.com/kubernetes/kubernetes/pull/116522), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- Pods owned by a Job now uses the labels `batch.kubernetes.io/job-name` and `batch.kubernetes.io/controller-uid`.
  The legacy labels `job-name` and `controller-uid` are still added for compatibility. ([#114930](https://github.com/kubernetes/kubernetes/pull/114930), [@kannon92](https://github.com/kannon92))
- Promoted `CronJobTimeZone` feature to GA ([#115904](https://github.com/kubernetes/kubernetes/pull/115904), [@soltysh](https://github.com/soltysh))
- Promoted `SelfSubjectReview` to Beta ([#116274](https://github.com/kubernetes/kubernetes/pull/116274), [@nabokihms](https://github.com/nabokihms)) [SIG API Machinery, Auth, CLI and Testing]
- Relaxed API validation to allow pod node selector to be mutable for gated pods (additions only, no deletions or mutations). ([#116161](https://github.com/kubernetes/kubernetes/pull/116161), [@danielvegamyhre](https://github.com/danielvegamyhre))
- Remove `kubernetes.io/grpc` standard appProtocol ([#116866](https://github.com/kubernetes/kubernetes/pull/116866), [@LiorLieberman](https://github.com/LiorLieberman)) [SIG API Machinery and Apps]
- Remove deprecated `--enable-taint-manager` and `--pod-eviction-timeout` CLI ([#115840](https://github.com/kubernetes/kubernetes/pull/115840), [@atosatto](https://github.com/atosatto))
- Removed support for the `v1alpha1` kubeletplugin API of `DynamicResourceManagement`. All plugins must be updated to `v1alpha2` in order to function properly. ([#116558](https://github.com/kubernetes/kubernetes/pull/116558), [@klueska](https://github.com/klueska))
- The API server now re-uses data encryption keys while the kms v2 plugin key ID is stable.  Data encryption keys are still randomly generated on server start but an atomic counter is used to prevent nonce collisions. ([#116155](https://github.com/kubernetes/kubernetes/pull/116155), [@enj](https://github.com/enj))
- The PodDisruptionBudget `spec.unhealthyPodEvictionPolicy` field has graduated to beta and is enabled by default. On servers with the feature enabled, this field may be set to `AlwaysAllow` to always allow unhealthy pods covered by the PodDisruptionBudget to be evicted. ([#115363](https://github.com/kubernetes/kubernetes/pull/115363), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps, Auth and Node]
- The `DownwardAPIHugePages` kubelet feature graduated to stable / GA. ([#115721](https://github.com/kubernetes/kubernetes/pull/115721), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps and Node]
- The following feature gates for volume expansion GA features have now been removed and must no longer be referenced in `--feature-gates` flags: `ExpandCSIVolumes`, `ExpandInUsePersistentVolumes`, `ExpandPersistentVolumes` ([#113942](https://github.com/kubernetes/kubernetes/pull/113942), [@mengjiao-liu](https://github.com/mengjiao-liu))
- The list-type of the alpha `resourceClaims` field introduced to `Pods` in `1.26.0` was modified from `set` to `map`, resolving an incompatibility with use of this schema in `CustomResourceDefinitions` and with server-side apply. ([#114585](https://github.com/kubernetes/kubernetes/pull/114585), [@JoelSpeed](https://github.com/JoelSpeed))
- Updated API reference for Requests, specifying they must not exceed limits ([#115434](https://github.com/kubernetes/kubernetes/pull/115434), [@ehashman](https://github.com/ehashman))
- Updated `KMSv2` to beta ([#115123](https://github.com/kubernetes/kubernetes/pull/115123), [@aramase](https://github.com/aramase))
- Updated: Redefine AppProtocol field description and add new standard values ([#115433](https://github.com/kubernetes/kubernetes/pull/115433), [@LiorLieberman](https://github.com/LiorLieberman)) [SIG API Machinery, Apps and Network]
- `/metrics/slis` is now available for control plane components allowing you to scrape health check metrics. ([#114997](https://github.com/kubernetes/kubernetes/pull/114997), [@Richabanker](https://github.com/Richabanker))
- `APIServerTracing` feature gate is now enabled by default. Tracing in the API
  Server is still disabled by default, and requires a config file to enable. ([#116144](https://github.com/kubernetes/kubernetes/pull/116144), [@dashpole](https://github.com/dashpole))
- `NodeResourceFit` and `NodeResourcesBalancedAllocation` implement the `PreScore`
  extension point for a more performant calculation. ([#115655](https://github.com/kubernetes/kubernetes/pull/115655), [@tangwz](https://github.com/tangwz))
- `PodSchedulingReadiness` is graduated to beta. ([#115815](https://github.com/kubernetes/kubernetes/pull/115815), [@Huang-Wei](https://github.com/Huang-Wei))
- `PodSpec.Container.Resources` became mutable for CPU and memory resource types.
  - `PodSpec.Container.ResizePolicy` (new object) gives users control over how their containers are resized.
  - `PodStatus.Resize` status describes the state of a requested Pod resize.
  - `PodStatus.ResourcesAllocated` describes node resources allocated to Pod.
  - `PodStatus.Resources` describes node resources applied to running containers by CRI.
  - `UpdateContainerResources` CRI API now supports both Linux and Windows. ([#102884](https://github.com/kubernetes/kubernetes/pull/102884), [@vinaykul](https://github.com/vinaykul))
- `SELinuxMountReadWriteOncePod` graduated to Beta. ([#116425](https://github.com/kubernetes/kubernetes/pull/116425), [@jsafrane](https://github.com/jsafrane))
- `StatefulSetAutoDeletePVC` feature gate promoted to beta. ([#116501](https://github.com/kubernetes/kubernetes/pull/116501), [@mattcary](https://github.com/mattcary))
- `StatefulSet` names must be DNS labels, rather than subdomains. Any `StatefulSet`
  which took advantage of subdomain validation (by having dots in the name) can't
  possibly have worked, because we eventually set `pod.spec.hostname` from the `StatefulSetName`,
  and that is validated as a DNS label. ([#114172](https://github.com/kubernetes/kubernetes/pull/114172), [@thockin](https://github.com/thockin))
- `ValidatingAdmissionPolicy` now provides a status field that contains results of type checking the validation expression.
  The type checking is fully informational, and the behavior of the policy is unchanged. ([#115668](https://github.com/kubernetes/kubernetes/pull/115668), [@jiahuif](https://github.com/jiahuif))
- `cacheSize` field in `EncryptionConfiguration` is not supported for KMSv2 provider ([#113121](https://github.com/kubernetes/kubernetes/pull/113121), [@aramase](https://github.com/aramase))
- `k8s.io/component-base/logs` now also supports adding command line flags to a `flag.FlagSet`. ([#114731](https://github.com/kubernetes/kubernetes/pull/114731), [@pohly](https://github.com/pohly))
- `kubelet`: migrated `--container-runtime-endpoint` and `--image-service-endpoint`
  to kubelet config ([#112136](https://github.com/kubernetes/kubernetes/pull/112136), [@pacoxu](https://github.com/pacoxu))
- `resource.k8s.io/v1alpha1` was replaced with `resource.k8s.io/v1alpha2`. Before
  upgrading a cluster, all objects in resource.k8s.io/v1alpha1 (ResourceClaim, ResourceClaimTemplate,
  ResourceClass, PodScheduling) must be deleted. The changes are internal, so
  YAML files which create pods and resource claims don't need changes except for
  the newer `apiVersion`. ([#116299](https://github.com/kubernetes/kubernetes/pull/116299), [@pohly](https://github.com/pohly))
- `volumes`: `resource.claims` is now cleared for PVC specs during create or update of a pod spec with inline PVC template or of a PVC because it has no effect. ([#115928](https://github.com/kubernetes/kubernetes/pull/115928), [@pohly](https://github.com/pohly))

### Feature

- A new client side metric `rest_client_request_retries_total` has been added that tracks the number of retries sent to the server, partitioned by status code, verb and host ([#108396](https://github.com/kubernetes/kubernetes/pull/108396), [@tkashem](https://github.com/tkashem))
- A new feature was enabled to improve the performance of the iptables mode of `kube-proxy` in large clusters. No action was required, however:
  
  1. If you experienced problems with Services not syncing to iptables correctly, you can disable the feature by passing `--feature-gates=MinimizeIPTablesRestore=false` to kube-proxy (and file a bug if this fixes it). (This might also be detected by seeing the value of kube-proxy's `sync_proxy_rules_iptables_partial_restore_failures_total` metric rising.)
  2. If you were previously overriding the kube-proxy configuration for performance reasons, this may no longer be necessary. See https://kubernetes.io/docs/reference/networking/virtual-ips/#optimizing-iptables-mode-performance. ([#115138](https://github.com/kubernetes/kubernetes/pull/115138), [@danwinship](https://github.com/danwinship))
- API validation relaxed allowing Indexed Jobs to be scaled up/down by changing parallelism and completions in tandem, such that parallelism == completions. ([#115236](https://github.com/kubernetes/kubernetes/pull/115236), [@danielvegamyhre](https://github.com/danielvegamyhre)) [SIG Apps and Testing]
- Added "general", "baseline", and "restricted" debugging profiles for kubectl debug. ([#114280](https://github.com/kubernetes/kubernetes/pull/114280), [@sding3](https://github.com/sding3)) [SIG CLI]
- Added "netadmin" debugging profiles for kubectl debug. ([#115712](https://github.com/kubernetes/kubernetes/pull/115712), [@wedaly](https://github.com/wedaly)) [SIG CLI]
- Added `--output plaintext-openapiv2` argument to kubectl explain to use old openapiv2 `explain` implementation. ([#115480](https://github.com/kubernetes/kubernetes/pull/115480), [@alexzielenski](https://github.com/alexzielenski))
- Added `NewVolumeManagerReconstruction` feature gate and enabled it by default to enable updated discovery of mounted volumes during kubelet startup. Please watch for kubelet getting stuck at startup and / or not unmounting volumes from deleted Pods and report any issues in this area. ([#115268](https://github.com/kubernetes/kubernetes/pull/115268), [@jsafrane](https://github.com/jsafrane))
- Added `kubelet` Topology Manager metrics to track admission requests processed and occured admission errors. ([#115137](https://github.com/kubernetes/kubernetes/pull/115137), [@swatisehgal](https://github.com/swatisehgal))
- Added apiserver_envelope_encryption_invalid_key_id_from_status_total to measure number of times an invalid keyID is returned by the Status RPC call. ([#115846](https://github.com/kubernetes/kubernetes/pull/115846), [@ritazh](https://github.com/ritazh)) [SIG API Machinery and Auth]
- Added apiserver_envelope_encryption_kms_operations_latency_seconds metric to measure the KMSv2 grpc calls latency. ([#115649](https://github.com/kubernetes/kubernetes/pull/115649), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Added e2e test to node expand volume with secret ([#115451](https://github.com/kubernetes/kubernetes/pull/115451), [@zhucan](https://github.com/zhucan))
- Added e2e tests for kubectl `--subresource` for beta graduation ([#116590](https://github.com/kubernetes/kubernetes/pull/116590), [@MadhavJivrajani](https://github.com/MadhavJivrajani))
- Added kubelet Topology Manager metric to measure topology manager admission latency. ([#115590](https://github.com/kubernetes/kubernetes/pull/115590), [@swatisehgal](https://github.com/swatisehgal))
- Added logging-format option to CCMs based on `k8s.io/cloud-provider` ([#108984](https://github.com/kubernetes/kubernetes/pull/108984), [@LittleFox94](https://github.com/LittleFox94))
- Added metrics for volume reconstruction during kubelet startup. ([#115965](https://github.com/kubernetes/kubernetes/pull/115965), [@jsafrane](https://github.com/jsafrane)) [SIG Node and Storage]
- Added new -f flag into debug command to be used passing pod or node files instead explicit names. ([#111453](https://github.com/kubernetes/kubernetes/pull/111453), [@ardaguclu](https://github.com/ardaguclu))
- Added new feature gate `ServiceNodePortStaticSubrange`, to enable the new strategy in the `NodePort` Service port allocators, so the node port range is subdivided and dynamic allocated `NodePort` port for Services are allocated preferentially from the upper range. ([#114418](https://github.com/kubernetes/kubernetes/pull/114418), [@xuzhenglun](https://github.com/xuzhenglun))
- Added scheduler preemption support for pods using `ReadWriteOncePod` PVCs ([#114051](https://github.com/kubernetes/kubernetes/pull/114051), [@chrishenzie](https://github.com/chrishenzie))
- Added the `applyconfiguration` generator to the code-generator script that generates server-side apply configuration and client APIs ([#114987](https://github.com/kubernetes/kubernetes/pull/114987), [@astefanutti](https://github.com/astefanutti))
- Added the ability to host webhooks in the cloud controller manager. ([#108838](https://github.com/kubernetes/kubernetes/pull/108838), [@nckturner](https://github.com/nckturner))
- Apiserver_storage_transformation_operations_total metric has been updated to include labels transformer_prefix and status. ([#115394](https://github.com/kubernetes/kubernetes/pull/115394), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth, Instrumentation and Testing]
- By enabling the `UserNamespacesStatelessPodsSupport` feature gate in kubelet, you can now run a stateless pod in a separate user namespace ([#116377](https://github.com/kubernetes/kubernetes/pull/116377), [@giuseppe](https://github.com/giuseppe)) [SIG Apps, Node and Storage]
- By enabling the alpha `CloudNodeIPs` feature gate in kubelet and the cloud
  provider, you can now specify a dual-stack `--node-ip` value (when using an
  external cloud provider that supports that functionality). ([#116305](https://github.com/kubernetes/kubernetes/pull/116305), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Cloud Provider, Network and Node]
- Changed kubectl `--subresource` flag to beta ([#116595](https://github.com/kubernetes/kubernetes/pull/116595), [@MadhavJivrajani](https://github.com/MadhavJivrajani))
- Changed metrics for aggregated discovery to publish new time series (alpha). ([#115630](https://github.com/kubernetes/kubernetes/pull/115630), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Testing]
- Dynamic Resource Allocation framework can be used for network devices ([#114364](https://github.com/kubernetes/kubernetes/pull/114364), [@bart0sh](https://github.com/bart0sh)) [SIG Node]
- Enable external plugins can be used as subcommands for kubectl create command if subcommand does not exist as builtin only when KUBECTL_ENABLE_CMD_SHADOW environment variable is exported. ([#116293](https://github.com/kubernetes/kubernetes/pull/116293), [@ardaguclu](https://github.com/ardaguclu))
- GRPC probes now set a linger option of 1s to improve the TIME-WAIT state. ([#115321](https://github.com/kubernetes/kubernetes/pull/115321), [@rphillips](https://github.com/rphillips)) [SIG Network and Node]
- Graduated CRI Events driven Pod LifeCycle Event Generator (Evented PLEG) to Beta ([#115967](https://github.com/kubernetes/kubernetes/pull/115967), [@harche](https://github.com/harche))
- Graduated `matchLabelKeys` in `podTopologySpread` to Beta ([#116291](https://github.com/kubernetes/kubernetes/pull/116291), [@denkensk](https://github.com/denkensk))
- Graduated the `CSINodeExpandSecret` feature to Beta. This feature facilitates passing secrets to CSI driver as part of Node Expansion CSI operation. ([#115621](https://github.com/kubernetes/kubernetes/pull/115621), [@humblec](https://github.com/humblec))
- Graduated the `LegacyServiceAccountTokenTracking` feature gate to Beta. The usage of auto-generated secret-based service account token now produces warnings by default, and relevant Secrets are labeled with a last-used timestamp (label key `kubernetes.io/legacy-token-last-used`). ([#114523](https://github.com/kubernetes/kubernetes/pull/114523), [@zshihang](https://github.com/zshihang)) [SIG API Machinery and Auth]
- HPA controller exposes the following metrics from the kube-controller-manager. 
  - `metric_computation_duration_seconds`: The time(seconds) that the HPA controller takes to calculate one metric.
  - `metric_computation_total`: Number of metric computations. ([#116326](https://github.com/kubernetes/kubernetes/pull/116326), [@sanposhiho](https://github.com/sanposhiho)) [SIG Apps, Autoscaling and Instrumentation]
- HPA controller starts to expose metrics from the kube-controller-manager.\n- `reconciliations_total`: Number of reconciliation of HPA controller. \n- `reconciliation_duration_seconds`: The time(seconds) that the HPA controller takes to reconcile once. ([#116010](https://github.com/kubernetes/kubernetes/pull/116010), [@sanposhiho](https://github.com/sanposhiho))
- Kube-up now includes `CoreDNS` version `v1.9.3` ([#114279](https://github.com/kubernetes/kubernetes/pull/114279), [@pacoxu](https://github.com/pacoxu))
- Kubeadm: added the experimental (alpha) feature gate `EtcdLearnerMode` that allows etcd members to be joined as learner and only then promoted as voting members ([#113318](https://github.com/kubernetes/kubernetes/pull/113318), [@pacoxu](https://github.com/pacoxu))
- Kubectl will now display `SeccompProfile` for pods, containers and ephemeral containers, if values were set. ([#113284](https://github.com/kubernetes/kubernetes/pull/113284), [@williamyeh](https://github.com/williamyeh))
- Kubectl: added e2e test for default container annotation ([#115046](https://github.com/kubernetes/kubernetes/pull/115046), [@pacoxu](https://github.com/pacoxu))
- Kubelet TCP and HTTP probes are now more effective using networking resources:
  conntrack entries, sockets. This is achieved by reducing the `TIME-WAIT` state
  of the connection to 1 second, instead of the defaults 60 seconds. This allows
  kubelet to free the socket, and free conntrack entry and ephemeral port associated. ([#115143](https://github.com/kubernetes/kubernetes/pull/115143), [@aojea](https://github.com/aojea))
- Kubelet allows pods to use the `net.ipv4.ip_local_reserved_ports` sysctl by default and the minimal kernel version is 3.16; Pod Security admission allows this sysctl in v1.27+ versions of the baseline and restricted policies. ([#115374](https://github.com/kubernetes/kubernetes/pull/115374), [@pacoxu](https://github.com/pacoxu)) [SIG Auth, Network and Node]
- Kubelet config file will be backed up to `/etc/kubernetes/tmp/` folder with `kubeadm-kubelet-config` append with a random suffix as the filename ([#114695](https://github.com/kubernetes/kubernetes/pull/114695), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Go `1.19.5` ([#115010](https://github.com/kubernetes/kubernetes/pull/115010), [@cpanato](https://github.com/cpanato))
- Kubernetes is now built with go 1.20 ([#114502](https://github.com/kubernetes/kubernetes/pull/114502), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with go 1.20.1 ([#115828](https://github.com/kubernetes/kubernetes/pull/115828), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with go 1.20.2 ([#116404](https://github.com/kubernetes/kubernetes/pull/116404), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Locked `CSIMigrationvSphere` feature gate. ([#116610](https://github.com/kubernetes/kubernetes/pull/116610), [@xing-yang](https://github.com/xing-yang))
- Made `apiextensions-apiserver` binary linking static (also affects the deb and rpm packages). ([#114226](https://github.com/kubernetes/kubernetes/pull/114226), [@saschagrunert](https://github.com/saschagrunert))
- Made `kube-aggregator` binary linking static (also affects the deb and rpm packages). ([#114227](https://github.com/kubernetes/kubernetes/pull/114227), [@saschagrunert](https://github.com/saschagrunert))
- Made `kubectl-convert` binary linking static (also affects the deb and rpm packages). ([#114228](https://github.com/kubernetes/kubernetes/pull/114228), [@saschagrunert](https://github.com/saschagrunert))
- Migrated controller helper functions to use contextual logging. ([#115049](https://github.com/kubernetes/kubernetes/pull/115049), [@fatsheep9146](https://github.com/fatsheep9146))
- Migrated the ResourceQuota controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113315](https://github.com/kubernetes/kubernetes/pull/113315), [@ncdc](https://github.com/ncdc)) [SIG API Machinery, Apps and Testing]
- Migrated the StatefulSet controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging) ([#113840](https://github.com/kubernetes/kubernetes/pull/113840), [@249043822](https://github.com/249043822))
- Migrated the `ClusterRole` aggregation controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113910](https://github.com/kubernetes/kubernetes/pull/113910), [@mengjiao-liu](https://github.com/mengjiao-liu))
- Migrated the `Deployment` controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging) ([#113525](https://github.com/kubernetes/kubernetes/pull/113525), [@249043822](https://github.com/249043822))
- Migrated the `ReplicaSet` controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#114871](https://github.com/kubernetes/kubernetes/pull/114871), [@Namanl2001](https://github.com/Namanl2001))
- Migrated the bootstrap signer controller and the token cleaner controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113464](https://github.com/kubernetes/kubernetes/pull/113464), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery, Apps and Instrumentation]
- Migrated the defaultbinder scheduler plugin to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116571](https://github.com/kubernetes/kubernetes/pull/116571), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]
- Migrated the main kube-controller-manager binary to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116529](https://github.com/kubernetes/kubernetes/pull/116529), [@pohly](https://github.com/pohly))
- Migrated the namespace controller (within `kube-controller-manager`) to support [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113443](https://github.com/kubernetes/kubernetes/pull/113443), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- Migrated the service-account controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#114918](https://github.com/kubernetes/kubernetes/pull/114918), [@Namanl2001](https://github.com/Namanl2001)) [SIG API Machinery, Apps, Auth, Instrumentation and Testing]
- Migrated the volume attach/detach controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging).
  Migrated the `PersistentVolumeClaim` protection controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging).
  Migrated the `PersistentVolume` protection controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113584](https://github.com/kubernetes/kubernetes/pull/113584), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- Migrated the “TTL after finished” controller (within `kube-controller-manager`)to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113916](https://github.com/kubernetes/kubernetes/pull/113916), [@songxiao-wang87](https://github.com/songxiao-wang87))
- Migrated the `cronjob` controller (within `kube-controller-manager`)to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging) ([#113428](https://github.com/kubernetes/kubernetes/pull/113428), [@mengjiao-liu](https://github.com/mengjiao-liu))
- New plugin_evaluation_total is added to the scheduler.This metric counts how many times the specific plugin affects the scheduling result. The metric does not get incremented when the plugin has nothing to do with an incoming Pod. ([#115082](https://github.com/kubernetes/kubernetes/pull/115082), [@sanposhiho](https://github.com/sanposhiho))
- Node `ipam` controller now exposes metrics `cidrset_cidrs_max_total` and `multicidrset_cidrs_max_total` with information about the max number of CIDRs that can be allocated. ([#112260](https://github.com/kubernetes/kubernetes/pull/112260), [@aryan9600](https://github.com/aryan9600))
- Performance improvements in `klog` ([#115277](https://github.com/kubernetes/kubernetes/pull/115277), [@pohly](https://github.com/pohly))
- Pod template `schedulingGates` are now mutable for Jobs that are suspended and have never been started ([#115940](https://github.com/kubernetes/kubernetes/pull/115940), [@ahg-g](https://github.com/ahg-g)) [SIG Apps]
- Pods which have an invalid negative `spec.terminationGracePeriodSeconds` value will now be treated as having a `terminationGracePeriodSeconds` of `1` ([#115606](https://github.com/kubernetes/kubernetes/pull/115606), [@wzshiming](https://github.com/wzshiming))
- Profiling can now be served on a unix-domain socket by using the `--profiling-path` option (when profiling is enabled) for security purposes. ([#114191](https://github.com/kubernetes/kubernetes/pull/114191), [@apelisse](https://github.com/apelisse)) [SIG API Machinery]
- Promote aggregated discovery endpoint to beta and it will be enabled by default ([#116108](https://github.com/kubernetes/kubernetes/pull/116108), [@Jefftree](https://github.com/Jefftree))
- Promoted `OpenAPIV3` to GA ([#116235](https://github.com/kubernetes/kubernetes/pull/116235), [@Jefftree](https://github.com/Jefftree))
- Promoted `whoami` kubectl command. ([#116510](https://github.com/kubernetes/kubernetes/pull/116510), [@nabokihms](https://github.com/nabokihms))
- Scheduler no longer runs the plugin's `Filter` method when its `PreFilter` method returned a Skip status.
  In other words, your `PreFilter`/`Filter` plugin can return a Skip status in `PreFilter` if the plugin does nothing in Filter for that Pod.
  Scheduler skips `NodeAffinity` Filter plugin when `NodeAffinity` Filter plugin has nothing to do with a Pod.
  It may affect some metrics values related to the `NodeAffinity` Filter plugin. ([#114125](https://github.com/kubernetes/kubernetes/pull/114125), [@sanposhiho](https://github.com/sanposhiho))
- Scheduler now skips `InterPodAffinity` Filter plugin when `InterPodAffinity` Filter plugin has nothing to do with a Pod.
  It may affect some metrics values related to the `InterPodAffinity` Filter plugin. ([#114889](https://github.com/kubernetes/kubernetes/pull/114889), [@sanposhiho](https://github.com/sanposhiho))
- Scheduler volumebinding: leveraged `PreFilterResult` to reduce down to only
  eligible node(s) for pod with bound claim(s) to local `PersistentVolume(s)` ([#109877](https://github.com/kubernetes/kubernetes/pull/109877), [@yibozhuang](https://github.com/yibozhuang))
- Scheduling cycle now terminates immediately when any scheduler plugin returns an 
  `unschedulableAndUnresolvable` status in `PostFilter`. ([#114699](https://github.com/kubernetes/kubernetes/pull/114699), [@kerthcet](https://github.com/kerthcet))
- Since Kubernetes v1.5, `kubectl apply` has had an alpha-stage `--prune` flag to support deleting previously applied objects that have been removed from the input manifest. This feature has remained in alpha ever since due to performance and correctness issues inherent in its design. This PR exposes a second, independent pruning alpha powered by a new standard named `ApplySets`. An `ApplySet` is a server-side object (by default, a Secret; ConfigMaps are also allowed) that kubectl can use to accurately and efficiently track set membership across `apply` operations. The format used for `ApplySet` is set out in  [KEP 3659](https://github.com/kubernetes/enhancements/issues/3659) as a low-level specification.  Other tools in the ecosystem can also build on this specification for improved interoperability.  To try the ApplySet-based pruning alpha, set `KUBECTL_APPLYSET=true` and use the flags `--prune --applyset=secret-name` with `kubectl apply`. ([#116205](https://github.com/kubernetes/kubernetes/pull/116205), [@justinsb](https://github.com/justinsb))
- Switched kubectl explain to use OpenAPIV3 information published by the server. OpenAPIV2 backend can  still be used with the `--output plaintext-openapiv2` argument ([#116390](https://github.com/kubernetes/kubernetes/pull/116390), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery, CLI and Testing]
- The Pod API field `.spec.schedulingGates[*].name` now requires qualified names (like `example.com/mygate`), matching validation for names of `.spec.readinessGates[*].name`. Any uses of the alpha scheduling gate feature prior to 1.27 that do not match that validation must be renamed or deleted before upgrading to 1.27. ([#115821](https://github.com/kubernetes/kubernetes/pull/115821), [@lianghao208](https://github.com/lianghao208)) [SIG Apps and Scheduling]
- The Scheduler did not run the plugin Score method when its PreScore method returned a Skip status. In other words, the PreScore/Score plugin could return a Skip status in PreScore if the plugin did nothing in Score for that Pod. ([#115652](https://github.com/kubernetes/kubernetes/pull/115652), [@AxeZhan](https://github.com/AxeZhan))
- The `AdvancedAuditing` feature gate was locked to _true_ in v1.27, and will be removed completely in v1.28 ([#115163](https://github.com/kubernetes/kubernetes/pull/115163), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery]
- The `JobMutableNodeSchedulingDirectives` feature gate has graduated to GA. ([#116116](https://github.com/kubernetes/kubernetes/pull/116116), [@ahg-g](https://github.com/ahg-g)) [SIG Apps, Scheduling and Testing]
- The `ReadWriteOncePod` feature gate has been graduated to beta. ([#114494](https://github.com/kubernetes/kubernetes/pull/114494), [@chrishenzie](https://github.com/chrishenzie))
- The bug which caused the status of Indexed Jobs to only update when new indexes were completed was fixed. Now, completed indexes are updated even if the `.status.completedIndexes` values are outside the `[0, .spec.completions> range`. ([#115349](https://github.com/kubernetes/kubernetes/pull/115349), [@danielvegamyhre](https://github.com/danielvegamyhre))
- The go version defined in `.go-version` is now fetched when invoking test, build, and code generation targets if the current go version does not match it. Set $FORCE_HOST_GO=y while testing or building to skip this behavior, or set $GO_VERSION to override the selected go version. ([#115377](https://github.com/kubernetes/kubernetes/pull/115377), [@liggitt](https://github.com/liggitt)) [SIG Testing]
- The job controller back-off logic is now decoupled from workqueue. In case of parallelism > 1, if there are multiple new failures in a reconciliation cycle, all the failures are taken into account to compute the back-off. Previously, the back-off kicked in for all types of failures; with this change, only pod failures are taken into account. If the back-off limits exceeds, the job is marked as failed immediately; before this change, the job is marked as failed in the next back-off. ([#114768](https://github.com/kubernetes/kubernetes/pull/114768), [@sathyanarays](https://github.com/sathyanarays)) [SIG Apps and Testing]
- The mount-utils mounter now provides an option to limit the number of concurrent format operations. ([#115379](https://github.com/kubernetes/kubernetes/pull/115379), [@artemvmin](https://github.com/artemvmin)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node and Storage]
- The scheduler's metric `plugin_execution_duration_seconds` now records `PreEnqueue` plugins execution seconds. ([#116201](https://github.com/kubernetes/kubernetes/pull/116201), [@sanposhiho](https://github.com/sanposhiho))
- Two changes to the `/debug/api_priority_and_fairness/dump_priority_levels` endpoint of API Priority and Fairness: added total number of dispatched, timed-out, rejected and cancelled requests; output now sorted by `PriorityLevelName`. ([#112393](https://github.com/kubernetes/kubernetes/pull/112393), [@borgerli](https://github.com/borgerli))
- Unlocked the `CSIMigrationvSphere` feature gate.
  The change allow users to continue using the in-tree vSphere driver,pending a vSphere
  CSI driver release that has with GA support for Windows, XFS, and raw block access. ([#116342](https://github.com/kubernetes/kubernetes/pull/116342), [@msau42](https://github.com/msau42)) [SIG Storage]
- Updated `cAdvisor` to `v0.47.0` ([#114883](https://github.com/kubernetes/kubernetes/pull/114883), [@bobbypage](https://github.com/bobbypage))
- Updated `kube-apiserver` SLO/SLI latency metrics to exclude priority & fairness queue wait times ([#116420](https://github.com/kubernetes/kubernetes/pull/116420), [@andrewsykim](https://github.com/andrewsykim))
- Updated distroless iptables to use released image `registry.k8s.io/build-image/distroless-iptables:v0.2.2`
  - Updated setcap to use released image `registry.k8s.io/build-image/setcap:bullseye-v1.4.2` ([#116509](https://github.com/kubernetes/kubernetes/pull/116509), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- Updated distroless iptables to use released image `registry.k8s.io/distroless-iptables:v0.2.1` ([#115905](https://github.com/kubernetes/kubernetes/pull/115905), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- Upgrades functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.0.0 and https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.0.1. 
  
  This is a new major release of kustomize, so there are a few backwards-incompatible changes, most of which are rare use cases, bug fixes with side effects, or things that have been deprecated for multiple releases already:
  
  - https://github.com/kubernetes-sigs/kustomize/pull/4911: Drop support for a very old, legacy style of patches. patches used to be allowed to be used as an alias for patchesStrategicMerge in kustomize v3. You now have to use patchesStrategicMerge explicitly, or update to the new syntax supported by patches. See examples in the PR description of https://github.com/kubernetes-sigs/kustomize/pull/4911.
  - https://github.com/kubernetes-sigs/kustomize/issues/4731: Remove a potential build-time side-effect in ConfigMapGenerator and SecretGenerator, which loaded values from the local environment under some circumstances, breaking kustomize build's side-effect-free promise. While this behavior was never intended, we deprecated it and are announcing it as a breaking change since it existed for a long time. See also the Eschewed Features documentation.
  - https://github.com/kubernetes-sigs/kustomize/pull/4985: If you previously included .git in an AWS or Azure URL, we will no longer automatically remove that suffix. You may need to add an extra / to replace the .git for the URL to properly resolve.
  - https://github.com/kubernetes-sigs/kustomize/pull/4954: Drop support for using gh: as a host (e.g. gh:kubernetes-sigs/kustomize). We were unable to find any usage of or basis for this and believe it may have been targeting a custom gitconfig shorthand syntax. ([#116598](https://github.com/kubernetes/kubernetes/pull/116598), [@natasha41575](https://github.com/natasha41575)) [SIG CLI]
- When an unsupported PodDisruptionBudget configuration is found, an event and log will be emitted to inform users of the misconfiguration. ([#115861](https://github.com/kubernetes/kubernetes/pull/115861), [@JayKayy](https://github.com/JayKayy)) [SIG Apps]
- [E2E] Pods spawned by E2E tests can now pull images from the private registry using the new --e2e-docker-config-file flag ([#114625](https://github.com/kubernetes/kubernetes/pull/114625), [@Divya063](https://github.com/Divya063)) [SIG Node and Testing]
- [alpha: kubectl apply --prune --applyset] Enabled certain custom resources (CRs) to be used as `ApplySet` parent objects. To enable this for a given CR, apply the label `applyset.kubernetes.io/is-parent-type: true` to the CustomResourceDefinition (CRD) that defines it. ([#116353](https://github.com/kubernetes/kubernetes/pull/116353), [@KnVerey](https://github.com/KnVerey))
- `Kubelet` no longer creates certain legacy iptables rules by default.
  It is possible that this will cause problems with some third-party components
  that improperly depended on those rules. If this affects you, you can run
  `kubelet` with `--feature-gates=IPTablesOwnershipCleanup=false`, but a bug should also be filed against the third-party component. ([#114472](https://github.com/kubernetes/kubernetes/pull/114472), [@danwinship](https://github.com/danwinship))
- `MinDomainsInPodTopologySpread` feature gate is enabled by default as a
  Beta feature in 1.27. ([#114445](https://github.com/kubernetes/kubernetes/pull/114445), [@mengjiao-liu](https://github.com/mengjiao-liu))
- `Secret` of `kubernetes.io/tls` type now verifies that the private key matches the cert ([#113581](https://github.com/kubernetes/kubernetes/pull/113581), [@aimuz](https://github.com/aimuz))
- `StorageVersionGC` (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113986](https://github.com/kubernetes/kubernetes/pull/113986), [@songxiao-wang87](https://github.com/songxiao-wang87))
- `client-go`: `sharedInformerFactory` now waits for goroutines during shutdown for metadatainformer and dynamicinformer. ([#114434](https://github.com/kubernetes/kubernetes/pull/114434), [@howardjohn](https://github.com/howardjohn))
- `kube-proxy` now accepts the `ContextualLogging`, `LoggingAlphaOptions`,
  `LoggingBetaOptions` ([#115233](https://github.com/kubernetes/kubernetes/pull/115233), [@pohly](https://github.com/pohly))
- `kube-scheduler`: Optimized implementation of null `labelSelector` in topology spreading. ([#116607](https://github.com/kubernetes/kubernetes/pull/116607), [@alculquicondor](https://github.com/alculquicondor))
- `kubeadm`: now shows a warning message when detecting that the sandbox image of the container runtime is inconsistent with that used by kubeadm ([#115610](https://github.com/kubernetes/kubernetes/pull/115610), [@SataQiu](https://github.com/SataQiu))
- `kubectl` now uses `HorizontalPodAutoscaler` `v2` by default. ([#114886](https://github.com/kubernetes/kubernetes/pull/114886), [@a7i](https://github.com/a7i))
- Kubernetes is now built with Go 1.20.3 ([#117125](https://github.com/kubernetes/kubernetes/pull/117125), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Updated distroless iptables to use released image `registry.k8s.io/build-image/distroless-iptables:v0.2.3` ([#117126](https://github.com/kubernetes/kubernetes/pull/117126), [@xmudrii](https://github.com/xmudrii)) [SIG Testing]

### Documentation

- Documented the reason field in CRI API to ensure it equals `OOMKilled` for the containers terminated by OOM killer ([#112977](https://github.com/kubernetes/kubernetes/pull/112977), [@mimowo](https://github.com/mimowo))
- Error message for Pods with requests exceeding limits will have a limit value printed. ([#112925](https://github.com/kubernetes/kubernetes/pull/112925), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev))
- The change affects the following CLI command:
  
  kubectl create rolebinding -h ([#107124](https://github.com/kubernetes/kubernetes/pull/107124), [@ptux](https://github.com/ptux)) [SIG CLI]

### Failing Test

- Deflaked a preemption test that may patch Nodes incorrectly. ([#114350](https://github.com/kubernetes/kubernetes/pull/114350), [@Huang-Wei](https://github.com/Huang-Wei))
- Fixed panic in vSphere e2e tests. ([#115863](https://github.com/kubernetes/kubernetes/pull/115863), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]
- Setting the Kubelet config option `--resolv-conf=Host` on Windows will now result in Kubelet applying the Pod DNS Policies as intended. ([#110566](https://github.com/kubernetes/kubernetes/pull/110566), [@claudiubelu](https://github.com/claudiubelu))

### Bug or Regression

- Added (dry run) and (server dry run) suffixes to `kubectl scale` command when `dry-run` is passed ([#114252](https://github.com/kubernetes/kubernetes/pull/114252), [@ardaguclu](https://github.com/ardaguclu))
- Applied configurations can be generated for types with `non-builtin` map fields ([#114920](https://github.com/kubernetes/kubernetes/pull/114920), [@astefanutti](https://github.com/astefanutti))
- Changed the error message of `kubectl rollout restart` when subsequent `kubectl rollout restart` commands are executed within a second ([#113040](https://github.com/kubernetes/kubernetes/pull/113040), [@ardaguclu](https://github.com/ardaguclu))
- Changed the error message to `cannot exec into multiple objects at a time` when file passed to `kubectl exec` contains multiple resources ([#114249](https://github.com/kubernetes/kubernetes/pull/114249), [@ardaguclu](https://github.com/ardaguclu))
- Client-go: fixed potential data races retrying requests using a custom `io.Reader` body; with this fix, only requests with no body or with `string` / `[]byte` / `runtime.Object` bodies can be retried ([#113933](https://github.com/kubernetes/kubernetes/pull/113933), [@liggitt](https://github.com/liggitt))
- Describing the CRs will now hide `.metadata.managedFields` ([#114584](https://github.com/kubernetes/kubernetes/pull/114584), [@soltysh](https://github.com/soltysh))
- Discovery document will correctly return the resources for aggregated apiservers that do not implement aggregated disovery ([#115770](https://github.com/kubernetes/kubernetes/pull/115770), [@Jefftree](https://github.com/Jefftree))
- Excluded preemptor pod metadata in the event message ([#114923](https://github.com/kubernetes/kubernetes/pull/114923), [@mimowo](https://github.com/mimowo))
- Expanded the partial fix for https://github.com/kubernetes/kubernetes/issues/111539
  which was already started in https://github.com/kubernetes/kubernetes/pull/109706
  Specifically, we will now reduce the amount of syncs for `ETP=local` services even
  further in the CCM and avoid re-configuring LBs to an even greater extent. ([#111658](https://github.com/kubernetes/kubernetes/pull/111658), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu))
- File content check for IPV4 is now enabled by default, and the check of IPV4 or IPV6 is done for `kubeadm init` or `kubeadm join` only in case the user intends to create a cluster to support that kind of IP address family ([#115420](https://github.com/kubernetes/kubernetes/pull/115420), [@chendave](https://github.com/chendave))
- Fixed CSI `PersistentVolumes` to allow Secrets names longer than 63 characters. ([#114776](https://github.com/kubernetes/kubernetes/pull/114776), [@jsafrane](https://github.com/jsafrane))
- Fixed Route controller to update routes when NodeIP changes ([#108095](https://github.com/kubernetes/kubernetes/pull/108095), [@lzhecheng](https://github.com/lzhecheng))
- Fixed `DaemonSet` to update the status even if it fails to create a pod. ([#113787](https://github.com/kubernetes/kubernetes/pull/113787), [@gjkim42](https://github.com/gjkim42))
- Fixed `SELinux` label for host path volumes created by host path provisioner ([#112021](https://github.com/kubernetes/kubernetes/pull/112021), [@mrunalp](https://github.com/mrunalp))
- Fixed `StatefulSetAutoDeletePVC` feature when `OwnerReferencesPermissionEnforcement` admission plugin is enabled. ([#114116](https://github.com/kubernetes/kubernetes/pull/114116), [@jsafrane](https://github.com/jsafrane))
- Fixed a bug on the `EndpointSlice` mirroring controller that generated multiple slices in some cases for custom endpoints in non canonical format. ([#114155](https://github.com/kubernetes/kubernetes/pull/114155), [@aojea](https://github.com/aojea))
- Fixed a bug that caused the `apiserver` to panic when trying to allocate a Service with a dynamic `ClusterIP` and was configured with Service CIDRs with a /28 mask for IPv4 and a /124 mask for IPv6 ([#115322](https://github.com/kubernetes/kubernetes/pull/115322), [@aojea](https://github.com/aojea))
- Fixed a bug where Kubernetes would apply a default StorageClass to a PersistentVolumeClaim,
  even when the deprecated annotation `volume.beta.kubernetes.io/storage-class` was set. ([#116089](https://github.com/kubernetes/kubernetes/pull/116089), [@cvvz](https://github.com/cvvz)) [SIG Apps and Storage]
- Fixed a bug where `events/v1` `Events` with similar event type and reporting instance were not aggregated by `client-go`. ([#112365](https://github.com/kubernetes/kubernetes/pull/112365), [@dgrisonnet](https://github.com/dgrisonnet))
- Fixed a bug where when emitting similar Events consecutively, some were rejected by the apiserver. ([#114237](https://github.com/kubernetes/kubernetes/pull/114237), [@dgrisonnet](https://github.com/dgrisonnet))
- Fixed a data race when emitting similar Events consecutively ([#114236](https://github.com/kubernetes/kubernetes/pull/114236), [@dgrisonnet](https://github.com/dgrisonnet))
- Fixed a log line in scheduler that inaccurately implies that volume binding has finalized ([#116018](https://github.com/kubernetes/kubernetes/pull/116018), [@TommyStarK](https://github.com/TommyStarK))
- Fixed a rare race condition in `kube-apiserver` that could lead to missing events when a watch API request was created at the same time `kube-apiserver` was re-initializing its internal watch. ([#116172](https://github.com/kubernetes/kubernetes/pull/116172), [@wojtek-t](https://github.com/wojtek-t))
- Fixed a regression in the pod binding subresource to honor the `metadata.uid` precondition.
  This allows kube-scheduler to ensure it is assigns node names to the same instances of pods it made scheduling decisions for. ([#116550](https://github.com/kubernetes/kubernetes/pull/116550), [@alculquicondor](https://github.com/alculquicondor))
- Fixed a regression that the scheduler always goes through all Filter plugins. ([#114518](https://github.com/kubernetes/kubernetes/pull/114518), [@Huang-Wei](https://github.com/Huang-Wei))
- Fixed an EndpointSlice Controller hashing bug that could cause EndpointSlices to incorrectly handle Pods with duplicate IP addresses. For example this could happen when a new Pod reused an IP that was also assigned to a Pod in a completed state. ([#115907](https://github.com/kubernetes/kubernetes/pull/115907), [@qinqon](https://github.com/qinqon)) [SIG Apps and Network]
- Fixed an issue where a CSI migrated volume may be prematurely detached when the CSI driver is not running on the node.
  If CSI migration is enabled on the node, even the csi-driver is not up and ready, we will still add this volume to DSW. ([#115464](https://github.com/kubernetes/kubernetes/pull/115464), [@sunnylovestiramisu](https://github.com/sunnylovestiramisu))
- Fixed an issue where failed pods associated with a job with `parallelism = 1` are recreated by the job controller honoring exponential backoff delay again. However, for jobs with `parallelism > 1`, pods might be created without exponential backoff delay. ([#114516](https://github.com/kubernetes/kubernetes/pull/114516), [@nikhita](https://github.com/nikhita))
- Fixed an issue with Winkernel Proxier - ClusterIP Loadbalancers missing if the `ExternalTrafficPolicy` is set to Local and the available endpoints are all `remoteEndpoints`. ([#115919](https://github.com/kubernetes/kubernetes/pull/115919), [@princepereira](https://github.com/princepereira))
- Fixed bug in CRD Validation Rules (beta) and `ValidatingAdmissionPolicy` (alpha) where all admission requests could result in `internal error: runtime error: index out of range [3] with length 3 evaluating rule: <rule name>` under certain circumstances. ([#114857](https://github.com/kubernetes/kubernetes/pull/114857), [@jpbetz](https://github.com/jpbetz))
- Fixed bug in beta aggregated discovery endpoint which caused CRD discovery information to be temporarily missing when an Aggregated APIService with the same GroupVersion is deleted (and vice versa). ([#116770](https://github.com/kubernetes/kubernetes/pull/116770), [@alexzielenski](https://github.com/alexzielenski))
- Fixed bug in reflector that couldn't recover from `Too large resource version` errors with API servers before 1.17.0. ([#115093](https://github.com/kubernetes/kubernetes/pull/115093), [@xuzhenglun](https://github.com/xuzhenglun))
- Fixed clearing of rate-limiter for the queue of checks for cleaning stale pod disruption conditions. The bug could result in the PDB synchronization updates firing too often or the pod disruption cleanups taking too long to happen. ([#114770](https://github.com/kubernetes/kubernetes/pull/114770), [@mimowo](https://github.com/mimowo))
- Fixed data race in `kube-scheduler` when preemption races with a Pod update. ([#116395](https://github.com/kubernetes/kubernetes/pull/116395), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Fixed file permission issues that happened during update of `Secret`/`ConfigMap`/`projected volume` when `fsGroup` is used. The problem caused a race condition where application gets intermittent permission denied error when reading files that were just updated, before the correct permissions were applied. ([#114464](https://github.com/kubernetes/kubernetes/pull/114464), [@tsaarni](https://github.com/tsaarni))
- Fixed incorrect watch events when a watch is initialized simultanously with a reinitializing watchcache. ([#116436](https://github.com/kubernetes/kubernetes/pull/116436), [@wojtek-t](https://github.com/wojtek-t))
- Fixed issue in `Winkernel` Proxier - Unexpected active TCP connection drops while horizontally scaling the endpoints for a LoadBalancer Service with Internal Traffic Policy: `Local` ([#113742](https://github.com/kubernetes/kubernetes/pull/113742), [@princepereira](https://github.com/princepereira))
- Fixed issue on Windows when calculating cpu limits on nodes with more than 64 logical processors ([#114231](https://github.com/kubernetes/kubernetes/pull/114231), [@mweibel](https://github.com/mweibel))
- Fixed issue with Winkernel Proxier - IPV6 load balancer policies were missing when service was configured with `ipFamilyPolicy`: `RequireDualStack` ([#115503](https://github.com/kubernetes/kubernetes/pull/115503), [@princepereira](https://github.com/princepereira))
- Fixed issue with Winkernel Proxier - IPV6 load balancer policies were missing when service was configured with `ipFamilyPolicy`: `RequireDualStack` ([#115577](https://github.com/kubernetes/kubernetes/pull/115577), [@princepereira](https://github.com/princepereira))
- Fixed issue with `Winkernel Proxier` - No ingress load balancer rules with endpoints to support load balancing when all the endpoints are terminating. ([#113776](https://github.com/kubernetes/kubernetes/pull/113776), [@princepereira](https://github.com/princepereira))
- Fixed missing delete events on informer re-lists to ensure all delete events were correctly emitted and using the latest known object state, so that all event handlers and stores always reflect the actual apiserver state as best as possible ([#115620](https://github.com/kubernetes/kubernetes/pull/115620), [@odinuge](https://github.com/odinuge))
- Fixed nil pointer error in `NodeVolumeLimits` csi logging ([#115179](https://github.com/kubernetes/kubernetes/pull/115179), [@sunnylovestiramisu](https://github.com/sunnylovestiramisu))
- Fixed panic validating custom resource definition schemas that set `multipleOf` to 0 ([#114869](https://github.com/kubernetes/kubernetes/pull/114869), [@liggitt](https://github.com/liggitt))
- Fixed performance regression in scheduler caused by frequent metric lookup on critical code path. ([#116428](https://github.com/kubernetes/kubernetes/pull/116428), [@mborsz](https://github.com/mborsz)) [SIG Scheduling]
- Fixed stuck apiserver if an aggregated apiservice returned `304 Not Modified` for aggregated discovery information ([#114459](https://github.com/kubernetes/kubernetes/pull/114459), [@alexzielenski](https://github.com/alexzielenski))
- Fixed the problem Pod terminating stuck because of trying to umount not actual mounted dir. ([#115769](https://github.com/kubernetes/kubernetes/pull/115769), [@mochizuki875](https://github.com/mochizuki875))
- Fixed the regression that introduced 34s timeout for DELETECOLLECTION calls ([#115341](https://github.com/kubernetes/kubernetes/pull/115341), [@tkashem](https://github.com/tkashem))
- Fixed two regressions introduced by the `PodDisruptionConditions` feature (on by default in 1.26):
  - pod eviction API calls returned spurious precondition errors and required a second evict API call to succeed
  - dry-run eviction API calls persisted a DisruptionTarget condition into the pod being evicted ([#116554](https://github.com/kubernetes/kubernetes/pull/116554), [@atiratree](https://github.com/atiratree))
- Fixes #115825. Kube-proxy will now include the `healthz` state in its response to the LB HC as to avoid indicating to the LB that it should use the node in question when Kube-proxy is not healthy. ([#111661](https://github.com/kubernetes/kubernetes/pull/111661), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Network]
- Flag `--concurrent-node-syncs` has been added to cloud node controller which defines how many workers in parallel will be initialising and synchronising nodes. ([#113104](https://github.com/kubernetes/kubernetes/pull/113104), [@pawbana](https://github.com/pawbana)) [SIG API Machinery, Cloud Provider and Scalability]
- Force deleted pods may fail to terminate until the kubelet is restarted when the container runtime returns an error during termination. We have strengthened testing for runtime failures and now perform a more rigorous reconciliation to ensure static pods (especially those that use fixed UIDs) are restarted.  As a side effect of these changes static pods will be restarted with lower latency than before (2s vs 4s, on average) and rapid updates to pod configuration should take effect sooner.
  
  A new metric `kubelet_known_pods` has been added at ALPHA stability to report the number of pods a Kubelet is tracking in a number of internal states.  Operators may use the metrics to track an excess of pods in the orphaned state that may not be completing. ([#113145](https://github.com/kubernetes/kubernetes/pull/113145), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery, Auth, Cloud Provider, Node and Testing]
- From now on, the HPA controller will return an error for the container resource metrics when the feature gate `HPAContainerMetrics` is disabled. As a result, HPA with a container resource metric performs no scale-down and performs only. ([#116043](https://github.com/kubernetes/kubernetes/pull/116043), [@sanposhiho](https://github.com/sanposhiho))
- IPVS: Any ipvs scheduler can now be configured. If a un-usable scheduler is configured `kube-proxy` will re-start and the logs must be checked (same as before but different log printouts). ([#114878](https://github.com/kubernetes/kubernetes/pull/114878), [@uablrek](https://github.com/uablrek))
- If a user attempts to add an ephemeral container to a static pod, they will now get a visible validation error. ([#114086](https://github.com/kubernetes/kubernetes/pull/114086), [@xmcqueen](https://github.com/xmcqueen))
- Ingress with `ingressClass` annotation and `IngressClassName` both set can be created now. ([#115447](https://github.com/kubernetes/kubernetes/pull/115447), [@AxeZhan](https://github.com/AxeZhan))
- Kube-apiserver: errors decoding objects in etcd are now recorded in an `apiserver_storage_decode_errors_total` counter metric ([#114376](https://github.com/kubernetes/kubernetes/pull/114376), [@baomingwang](https://github.com/baomingwang)) [SIG API Machinery and Instrumentation]
- Kube-apiserver: regular expressions specified with the `--cors-allowed-origins` option are now validated to match the entire `hostname` inside the `Origin` header of the request and 
  must contain '^' or the '//' prefix to anchor to the start, and '$' or the port separator ':' to anchor to 
  the end. ([#112809](https://github.com/kubernetes/kubernetes/pull/112809), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Kube-apiserver: removed N^2 behavior loading webhook configurations. ([#114794](https://github.com/kubernetes/kubernetes/pull/114794), [@lavalamp](https://github.com/lavalamp)) [SIG API Machinery, Architecture, CLI, Cloud Provider and Node]
- Kubeadm: fixed an etcd learner-mode bug by preparing an etcd static pod manifest before promoting ([#115038](https://github.com/kubernetes/kubernetes/pull/115038), [@tobiasgiese](https://github.com/tobiasgiese))
- Kubeadm: fixed the bug where `kubeadm` always does CRI detection even if it is not required by a phase subcommand ([#114455](https://github.com/kubernetes/kubernetes/pull/114455), [@SataQiu](https://github.com/SataQiu))
- Kubeadm: improved retries when updating node information, in case `kube-apiserver` is temporarily unavailable ([#114176](https://github.com/kubernetes/kubernetes/pull/114176), [@QuantumEnergyE](https://github.com/QuantumEnergyE))
- Kubeadm`: modified `--config` flag from required to optional for `kubeadm kubeconfig user` command ([#116074](https://github.com/kubernetes/kubernetes/pull/116074), [@SataQiu](https://github.com/SataQiu))
- Kubectl: enabled usage of label selector for filtering out resources when pruning for kubectl diff ([#114863](https://github.com/kubernetes/kubernetes/pull/114863), [@danlenar](https://github.com/danlenar))
- Kubelet startup now fails CRI connection if service or image endpoint is throwing any error ([#115102](https://github.com/kubernetes/kubernetes/pull/115102), [@saschagrunert](https://github.com/saschagrunert))
- Kubelet: fix recording issue when pulling image did finish ([#114904](https://github.com/kubernetes/kubernetes/pull/114904), [@TommyStarK](https://github.com/TommyStarK)) [SIG Node]
- Kubelet`: fixed a bug in `kubelet` that stopped rendering the `ConfigMaps` when `fsquota` monitoring is enabled ([#112624](https://github.com/kubernetes/kubernetes/pull/112624), [@pacoxu](https://github.com/pacoxu))
- Messages of `DisruptionTarget` condition now excludes preemptor pod metadata ([#114914](https://github.com/kubernetes/kubernetes/pull/114914), [@mimowo](https://github.com/mimowo))
- Optimized `LoadBalancer` creation with the help of attribute Internal Traffic Policy: `Local` ([#114407](https://github.com/kubernetes/kubernetes/pull/114407), [@princepereira](https://github.com/princepereira))
- PVCs will automatically be recreated if they are missing for a pending Pod. ([#113270](https://github.com/kubernetes/kubernetes/pull/113270), [@rrangith](https://github.com/rrangith)) [SIG Apps and Testing]
- PersistentVolume API objects which set NodeAffinities using beta Kubernetes labels for OS, architecture, zone, region, and instance type may now be modified to use the stable Kubernetes labels. ([#115391](https://github.com/kubernetes/kubernetes/pull/115391), [@haoruan](https://github.com/haoruan))
- Potentially breaking change - Updating the polling interval for Windows stats collection from 1 second to 10 seconds ([#116546](https://github.com/kubernetes/kubernetes/pull/116546), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Relaxed API validation for usage `key encipherment` and `kubelet` uses requested usages accordingly ([#111660](https://github.com/kubernetes/kubernetes/pull/111660), [@pacoxu](https://github.com/pacoxu))
- Removed scheduler names from preemption event messages. ([#114980](https://github.com/kubernetes/kubernetes/pull/114980), [@mimowo](https://github.com/mimowo))
- Shared informers now correctly propagate whether they are synced or not. Individual informer handlers may now check if they are synced or not (new `HasSynced` method). Library support is added to assist controllers in tracking whether their own work is completed for items in the initial list (`AsyncTracker`). ([#113985](https://github.com/kubernetes/kubernetes/pull/113985), [@lavalamp](https://github.com/lavalamp))
- The Kubernetes API server now correctly detects and closes existing TLS connections when its client certificate file for kubelet authentication has been rotated. ([#115315](https://github.com/kubernetes/kubernetes/pull/115315), [@enj](https://github.com/enj)) [SIG API Machinery, Auth, Node and Testing]
- Total test spec is now available by `ProgressReporter`, it will be reported before test suite got executed. ([#114417](https://github.com/kubernetes/kubernetes/pull/114417), [@chendave](https://github.com/chendave))
- Updated the Event series starting count when emitting isomorphic events from 1 to 2. ([#112334](https://github.com/kubernetes/kubernetes/pull/112334), [@dgrisonnet](https://github.com/dgrisonnet))
- When GCing pods, `kube-controller-manager` will delete Evicted pods first. ([#116167](https://github.com/kubernetes/kubernetes/pull/116167), [@borgerli](https://github.com/borgerli))
- When describing deployments, `OldReplicaSets` now always shows all replicasets controlled the deployment, not just those that still have replicas available. ([#113083](https://github.com/kubernetes/kubernetes/pull/113083), [@llorllale](https://github.com/llorllale)) [SIG CLI]
- Windows CPU usage node stats are now correctly calculated for nodes with multiple Processor Groups. ([#110864](https://github.com/kubernetes/kubernetes/pull/110864), [@claudiubelu](https://github.com/claudiubelu)) [SIG Node, Testing and Windows]
- `LabelSelectors` specified in `topologySpreadConstraints` were validated to ensure that pods are scheduled as expected. Existing pods with invalid `LabelSelectors` could be updated, but new pods were required to specify valid `LabelSelectors`. ([#111802](https://github.com/kubernetes/kubernetes/pull/111802), [@maaoBit](https://github.com/maaoBit))
- `PodGC` for pods which are in terminal phase now do not add the `DisruptionTarget` condition. ([#115056](https://github.com/kubernetes/kubernetes/pull/115056), [@mimowo](https://github.com/mimowo))
- `Service` of type `ExternalName` do not create an `Endpoint` anymore. ([#114814](https://github.com/kubernetes/kubernetes/pull/114814), [@panslava](https://github.com/panslava))
- `cacher`: If `ResourceVersion` is unset, the watch is now served from the underlying storage as documented. ([#115096](https://github.com/kubernetes/kubernetes/pull/115096), [@MadhavJivrajani](https://github.com/MadhavJivrajani))
- `client-go`: fixed the wait time for trying to acquire the leader lease ([#114872](https://github.com/kubernetes/kubernetes/pull/114872), [@Iceber](https://github.com/Iceber))
- `etcd`: Updated to `v3.5.7` ([#115310](https://github.com/kubernetes/kubernetes/pull/115310), [@mzaian](https://github.com/mzaian))
- `golang.org/x/net` updated to `v0.7.0` to fix CVE-2022-41723 ([#115786](https://github.com/kubernetes/kubernetes/pull/115786), [@liggitt](https://github.com/liggitt))
- `kube-controller-manager` will not run nodeipam controller when allocator type
  is `CloudAllocator` and the cloud provider is not enabled. ([#114596](https://github.com/kubernetes/kubernetes/pull/114596), [@andrewsykim](https://github.com/andrewsykim))
- `kube-controller-manager`: fixed a bug that the `kubeconfig` field of `kubecontrollermanager.config.k8s.io` configuration is not populated correctly ([#116219](https://github.com/kubernetes/kubernetes/pull/116219), [@SataQiu](https://github.com/SataQiu))
- `kube-proxy` with `--proxy-mode=ipvs` can be used with statically linked kernels.
  The reseved IPv4 range `TEST-NET-2` in `rfc5737` MUST NOT be used for `ClusterIP` or `loadBalancerIP` since address `198.51.100.0` is used for probing. ([#114669](https://github.com/kubernetes/kubernetes/pull/114669), [@uablrek](https://github.com/uablrek))
- `kubeadm`: fixed a bug where the uploaded kubelet configuration in `kube-system/kubelet-config` `ConfigMap` does not respect user patch ([#115575](https://github.com/kubernetes/kubernetes/pull/115575), [@SataQiu](https://github.com/SataQiu))
- `kubeadm`: now respects user provided `kubeconfig` during discovery process ([#113998](https://github.com/kubernetes/kubernetes/pull/113998), [@SataQiu](https://github.com/SataQiu))
- `kubectl port-forward` now exits with exit code 1 when remote connection is
  lost ([#114460](https://github.com/kubernetes/kubernetes/pull/114460), [@brianpursley](https://github.com/brianpursley))
- `nodeName` being set along with non-empty `schedulingGates` is now enforced. ([#115569](https://github.com/kubernetes/kubernetes/pull/115569), [@Huang-Wei](https://github.com/Huang-Wei))
- `node_stage_path` is now set whenever available for expansion during mount ([#115346](https://github.com/kubernetes/kubernetes/pull/115346), [@gnufied](https://github.com/gnufied))
- `statefulset` status will now be consistent on API errors ([#113834](https://github.com/kubernetes/kubernetes/pull/113834), [@atiratree](https://github.com/atiratree))
- `tryUnmount` now respects `mounter.withSafeNotMountedBehavior` ([#114736](https://github.com/kubernetes/kubernetes/pull/114736), [@andyzhangx](https://github.com/andyzhangx))
- The encryption response from KMS v2 plugins is now validated earlier at DEK generation time instead of waiting until an encryption is performed. ([#116877](https://github.com/kubernetes/kubernetes/pull/116877), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]
- Recreate DaemonSet pods completed with Succeeded phase ([#117073](https://github.com/kubernetes/kubernetes/pull/117073), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]


### Other (Cleanup or Flake)

- Added basic Denial Of Service prevention for the the node-local kubelet `podresource` API ([#116459](https://github.com/kubernetes/kubernetes/pull/116459), [@ffromani](https://github.com/ffromani)) [SIG Node and Testing]
- Callers of `wait.ExponentialBackoffWithContext` now must pass a `ConditionWithContextFunc` to be consistent with the signature and avoid creating a duplicate context. If your condition does not need a context you can use the `ConditionFunc.WithContext()` helper to ignore the context, or use `ExponentialBackoff` directly. ([#115113](https://github.com/kubernetes/kubernetes/pull/115113), [@smarterclayton](https://github.com/smarterclayton))
- Changed docs for `--contention-profiling` flag to reflect it performed block profiling ([#114490](https://github.com/kubernetes/kubernetes/pull/114490), [@MadhavJivrajani](https://github.com/MadhavJivrajani))
- E2e framework: added `--report-complete-ginkgo` and `--report-complete-junit` parameters. They work like `ginkgo --json-report <report dir>/ginkgo/report.json --junit-report <report dir>/ginkgo/report.xml`. ([#115678](https://github.com/kubernetes/kubernetes/pull/115678), [@pohly](https://github.com/pohly)) [SIG Testing]
- Fixed incorrect log information in the `iptables` utility. ([#110723](https://github.com/kubernetes/kubernetes/pull/110723), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- Improved FormatMap: Improves performance by about 4x, or nearly 2x in the worst case ([#112661](https://github.com/kubernetes/kubernetes/pull/112661), [@aimuz](https://github.com/aimuz)) [SIG Node]
- Improved misleading message, in case of no metrics received for the `HPA` controlled pods. ([#114740](https://github.com/kubernetes/kubernetes/pull/114740), [@kushagra98](https://github.com/kushagra98))
- Introduced new metrics removing the redundant subsystem in kube-apiserver pod logs metrics and deprecate the original ones:
  - kube_apiserver_pod_logs_pods_logs_backend_tls_failure_total becomes kube_apiserver_pod_logs_backend_tls_failure_total
  - kube_apiserver_pod_logs_pods_logs_insecure_backend_total becomes kube_apiserver_pod_logs_insecure_backend_total ([#114497](https://github.com/kubernetes/kubernetes/pull/114497), [@dgrisonnet](https://github.com/dgrisonnet))
- Kubeadm: removed the deprecated `v1beta2` API. kubeadm 1.26's `config migrate`
  command can be used to migrate a `v1beta2` configuration file to `v1beta3` ([#114540](https://github.com/kubernetes/kubernetes/pull/114540), [@pacoxu](https://github.com/pacoxu))
- Kubelet: remove deprecated flag `--container-runtime` ([#114017](https://github.com/kubernetes/kubernetes/pull/114017), [@calvin0327](https://github.com/calvin0327)) [SIG Cloud Provider and Node]
- Kubelet: the deprecated `--master-service-namespace` flag is removed in v1.27 ([#116015](https://github.com/kubernetes/kubernetes/pull/116015), [@SataQiu](https://github.com/SataQiu))
- Linux/arm will not ship in Kubernetes 1.27 as we are running into issues with building artifacts using golang 1.20.2 (please see issue #116492) ([#115742](https://github.com/kubernetes/kubernetes/pull/115742), [@dims](https://github.com/dims)) [SIG Architecture, Release and Testing]
- Migrated `pkg/controller/nodeipam/ipam/cloud_cidr_allocator.go, pkg/controller/nodeipam/ipam/multi_cidr_range_allocator.go pkg/controller/nodeipam/ipam/range_allocator.go pkg/controller/nodelifecycle/node_lifecycle_controller.go` to structured logging ([#112670](https://github.com/kubernetes/kubernetes/pull/112670), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- Migrated the Kubernetes object garbage collector (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113471](https://github.com/kubernetes/kubernetes/pull/113471), [@ncdc](https://github.com/ncdc))
- Migrated the ttlafterfinished controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#115332](https://github.com/kubernetes/kubernetes/pull/115332), [@obaranov1](https://github.com/obaranov1)) [SIG Apps]
- Migrated the “sample-controller” controller to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113879](https://github.com/kubernetes/kubernetes/pull/113879), [@pchan](https://github.com/pchan)) [SIG API Machinery and Instrumentation]
- Promoted pod resource `limit/request` metrics to stable. ([#115454](https://github.com/kubernetes/kubernetes/pull/115454), [@dgrisonnet](https://github.com/dgrisonnet))
- Removed AWS kubelet credential provider. Please use the external kubelet credential provider binary named `ecr-credential-provider` instead. ([#116329](https://github.com/kubernetes/kubernetes/pull/116329), [@dims](https://github.com/dims)) [SIG Node, Storage and Testing]
- Removed Azure disk in-tree storage plugin ([#116301](https://github.com/kubernetes/kubernetes/pull/116301), [@andyzhangx](https://github.com/andyzhangx))
- Removed flag `master-service-namespace` from `api-server` arguments ([#114446](https://github.com/kubernetes/kubernetes/pull/114446), [@lengrongfu](https://github.com/lengrongfu))
- Removed the following deprecated metrics:
  - node_collector_evictions_number replaced by node_collector_evictions_total
  - scheduler_e2e_scheduling_duration_seconds replaced by scheduler_scheduling_attempt_duration_seconds ([#115209](https://github.com/kubernetes/kubernetes/pull/115209), [@dgrisonnet](https://github.com/dgrisonnet))
- Removed unused rule for `nodes/spec` from `ClusterRole` `system:kubelet-api-admin` ([#113267](https://github.com/kubernetes/kubernetes/pull/113267), [@hoskeri](https://github.com/hoskeri))
- Renamed API server identity Lease labels to use the key `apiserver.kubernetes.io/identity` ([#114586](https://github.com/kubernetes/kubernetes/pull/114586), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, Apps, Cloud Provider and Testing]
- Storage.k8s.io/v1beta1 API version of CSIStorageCapacity will no longer be served ([#116523](https://github.com/kubernetes/kubernetes/pull/116523), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery]
- The `CSIMigrationAzureFile` feature gate (for the feature which graduated to GA in v1.26) is now unconditionally enabled and will be removed in v1.28. ([#114953](https://github.com/kubernetes/kubernetes/pull/114953), [@enj](https://github.com/enj))
- The `ControllerManagerLeaderMigration` feature, GA since `1.24`, is now unconditionally enabled and the feature gate option has been removed. ([#113534](https://github.com/kubernetes/kubernetes/pull/113534), [@pacoxu](https://github.com/pacoxu))
- The `WaitFor` and `WaitForWithContext` functions in the wait package have now been marked private. Callers should use the equivalent `Poll*` method with a zero duration interval. ([#115116](https://github.com/kubernetes/kubernetes/pull/115116), [@smarterclayton](https://github.com/smarterclayton))
- The `wait.Poll*` and `wait.ExponentialBackoff*` functions have been deprecated and will be removed in a future release.  Callers should switch to using `wait.PollUntilContextCancel`, `wait.PollUntilContextTimeout`, or `wait.ExponentialBackoffWithContext` as appropriate.
  
  `PollWithContext(Cancel|Deadline)` will no longer return `ErrWaitTimeout` - use the `Interrupted(error) bool` helper to replace checks for `err == ErrWaitTimeout`, or compare specifically to context errors as needed. A future release will make the `ErrWaitTimeout` error private and callers must use `Interrupted()` instead. If you are returning `ErrWaitTimeout` from your own methods, switch to creating a location specific `cause err` and pass it to the new method `wait.ErrorInterrupted(cause) error` which will ensure `Interrupted()` returns true for your loop. 
  
  The `wait.NewExponentialBackoffManager` and `wait.NewJitteringBackoffManager` functions have been marked as deprecated.  Callers should switch to using the `Backoff{...}.DelayWithReset(clock, resetInterval)` method and must set the `Steps` field when using `Factor`. As a short term change, callers may use the `Timer()` method on the `BackoffManager` until the backoff managers are deprecated and removed. Please see the godoc of the deprecated functions for examples of how to replace usage of this function. ([#107826](https://github.com/kubernetes/kubernetes/pull/107826), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery, Auth, Cloud Provider, Storage and Testing]
- The feature gates `CSIInlineVolume`, `CSIMigration`, `DaemonSetUpdateSurge`, `EphemeralContainers`, `IdentifyPodOS`, `LocalStorageCapacityIsolation`, `NetworkPolicyEndPort` and `StatefulSetMinReadySeconds` that graduated to GA in v1.25 and were unconditionally enabled have been removed in v1.27 ([#114410](https://github.com/kubernetes/kubernetes/pull/114410), [@SataQiu](https://github.com/SataQiu)) [SIG Node]
- Upgraded `coredns` to `v1.10.1` ([#115603](https://github.com/kubernetes/kubernetes/pull/115603), [@pacoxu](https://github.com/pacoxu))
- Upgraded `go-jose` to `v2.6.0` ([#115893](https://github.com/kubernetes/kubernetes/pull/115893), [@mgoltzsche](https://github.com/mgoltzsche))
- [KCCM - service controller]: enabled connection draining for terminating pods upon node downscale by the cluster autoscaler. This is done by not reacting to the taint used by the cluster autoscaler to indicate that the node is going away soon, thus keeping the node referenced by the load balancer until the VM has been completely deleted. ([#115204](https://github.com/kubernetes/kubernetes/pull/115204), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu))
- `apiserver_admission_webhook_admission_duration_seconds` buckets have been expanded, 25s is now the largest bucket size to match the webhook default timeout. ([#115802](https://github.com/kubernetes/kubernetes/pull/115802), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Instrumentation]
- `wait.ContextForChannel()` now implements the context.Context interface and
  does not return a cancellation function. ([#115140](https://github.com/kubernetes/kubernetes/pull/115140), [@smarterclayton](https://github.com/smarterclayton))

## Dependencies

### Added
- github.com/a8m/tree: [10a5fd5](https://github.com/a8m/tree/tree/10a5fd5)
- github.com/dougm/pretty: [2ee9d74](https://github.com/dougm/pretty/tree/2ee9d74)
- github.com/rasky/go-xdr: [4930550](https://github.com/rasky/go-xdr/tree/4930550)
- github.com/vmware/vmw-guestinfo: [25eff15](https://github.com/vmware/vmw-guestinfo/tree/25eff15)
- sigs.k8s.io/kustomize/kustomize/v5: v5.0.1

### Changed
- github.com/Microsoft/hcsshim: [v0.8.22 → v0.8.25](https://github.com/Microsoft/hcsshim/compare/v0.8.22...v0.8.25)
- github.com/aws/aws-sdk-go: [v1.44.116 → v1.35.24](https://github.com/aws/aws-sdk-go/compare/v1.44.116...v1.35.24)
- github.com/coredns/corefile-migration: [v1.0.17 → v1.0.20](https://github.com/coredns/corefile-migration/compare/v1.0.17...v1.0.20)
- github.com/coreos/go-systemd/v22: [v22.3.2 → v22.4.0](https://github.com/coreos/go-systemd/v22/compare/v22.3.2...v22.4.0)
- github.com/creack/pty: [v1.1.11 → v1.1.18](https://github.com/creack/pty/compare/v1.1.11...v1.1.18)
- github.com/docker/docker: [v20.10.18+incompatible → v20.10.21+incompatible](https://github.com/docker/docker/compare/v20.10.18...v20.10.21)
- github.com/go-errors/errors: [v1.0.1 → v1.4.2](https://github.com/go-errors/errors/compare/v1.0.1...v1.4.2)
- github.com/go-openapi/jsonpointer: [v0.19.5 → v0.19.6](https://github.com/go-openapi/jsonpointer/compare/v0.19.5...v0.19.6)
- github.com/go-openapi/jsonreference: [v0.20.0 → v0.20.1](https://github.com/go-openapi/jsonreference/compare/v0.20.0...v0.20.1)
- github.com/go-openapi/swag: [v0.19.14 → v0.22.3](https://github.com/go-openapi/swag/compare/v0.19.14...v0.22.3)
- github.com/golang-jwt/jwt/v4: [v4.2.0 → v4.4.2](https://github.com/golang-jwt/jwt/v4/compare/v4.2.0...v4.4.2)
- github.com/golang/protobuf: [v1.5.2 → v1.5.3](https://github.com/golang/protobuf/compare/v1.5.2...v1.5.3)
- github.com/google/cadvisor: [v0.46.0 → v0.47.1](https://github.com/google/cadvisor/compare/v0.46.0...v0.47.1)
- github.com/google/cel-go: [v0.12.5 → v0.12.6](https://github.com/google/cel-go/compare/v0.12.5...v0.12.6)
- github.com/google/uuid: [v1.1.2 → v1.3.0](https://github.com/google/uuid/compare/v1.1.2...v1.3.0)
- github.com/kr/pretty: [v0.2.1 → v0.3.0](https://github.com/kr/pretty/compare/v0.2.1...v0.3.0)
- github.com/mailru/easyjson: [v0.7.6 → v0.7.7](https://github.com/mailru/easyjson/compare/v0.7.6...v0.7.7)
- github.com/moby/ipvs: [v1.0.1 → v1.1.0](https://github.com/moby/ipvs/compare/v1.0.1...v1.1.0)
- github.com/moby/term: [39b0c02 → 1aeaba8](https://github.com/moby/term/compare/39b0c02...1aeaba8)
- github.com/onsi/ginkgo/v2: [v2.4.0 → v2.9.1](https://github.com/onsi/ginkgo/v2/compare/v2.4.0...v2.9.1)
- github.com/onsi/gomega: [v1.23.0 → v1.27.4](https://github.com/onsi/gomega/compare/v1.23.0...v1.27.4)
- github.com/opencontainers/runtime-spec: [1c3f411 → 494a5a6](https://github.com/opencontainers/runtime-spec/compare/1c3f411...494a5a6)
- github.com/rogpeppe/go-internal: [v1.3.0 → v1.10.0](https://github.com/rogpeppe/go-internal/compare/v1.3.0...v1.10.0)
- github.com/sirupsen/logrus: [v1.8.1 → v1.9.0](https://github.com/sirupsen/logrus/compare/v1.8.1...v1.9.0)
- github.com/stretchr/objx: [v0.4.0 → v0.5.0](https://github.com/stretchr/objx/compare/v0.4.0...v0.5.0)
- github.com/stretchr/testify: [v1.8.0 → v1.8.1](https://github.com/stretchr/testify/compare/v1.8.0...v1.8.1)
- github.com/tmc/grpc-websocket-proxy: [e5319fd → 673ab2c](https://github.com/tmc/grpc-websocket-proxy/compare/e5319fd...673ab2c)
- github.com/vishvananda/netns: [db3c7e5 → v0.0.2](https://github.com/vishvananda/netns/compare/db3c7e5...v0.0.2)
- github.com/vmware/govmomi: [v0.20.3 → v0.30.0](https://github.com/vmware/govmomi/compare/v0.20.3...v0.30.0)
- go.etcd.io/etcd/api/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/client/pkg/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/client/v2: v2.305.5 → v2.305.7
- go.etcd.io/etcd/client/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/pkg/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/raft/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/server/v3: v3.5.5 → v3.5.7
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.35.0 → v0.35.1
- go.uber.org/goleak: v1.2.0 → v1.2.1
- golang.org/x/mod: v0.6.0 → v0.9.0
- golang.org/x/net: 1e63c2f → v0.8.0
- golang.org/x/sync: 886fb93 → v0.1.0
- golang.org/x/sys: v0.3.0 → v0.6.0
- golang.org/x/term: v0.3.0 → v0.6.0
- golang.org/x/text: v0.5.0 → v0.8.0
- golang.org/x/tools: v0.2.0 → v0.7.0
- golang.org/x/xerrors: 5ec99f8 → 04be3eb
- google.golang.org/grpc: v1.49.0 → v1.51.0
- gopkg.in/check.v1: 8fa4692 → 10cb982
- gopkg.in/square/go-jose.v2: v2.2.2 → v2.6.0
- k8s.io/klog/v2: v2.80.1 → v2.90.1
- k8s.io/kube-openapi: 172d655 → 15aac26
- k8s.io/utils: 1a15be2 → a36077c
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.33 → v0.1.1
- sigs.k8s.io/json: f223a00 → bc3834c
- sigs.k8s.io/kustomize/api: v0.12.1 → v0.13.2
- sigs.k8s.io/kustomize/cmd/config: v0.10.9 → v0.11.1
- sigs.k8s.io/kustomize/kyaml: v0.13.9 → v0.14.1

### Removed
- github.com/PuerkitoBio/purell: [v1.1.1](https://github.com/PuerkitoBio/purell/tree/v1.1.1)
- github.com/PuerkitoBio/urlesc: [de5bf2a](https://github.com/PuerkitoBio/urlesc/tree/de5bf2a)
- github.com/elazarl/goproxy: [947c36d](https://github.com/elazarl/goproxy/tree/947c36d)
- github.com/form3tech-oss/jwt-go: [v3.2.3+incompatible](https://github.com/form3tech-oss/jwt-go/tree/v3.2.3)
- github.com/mattn/go-runewidth: [v0.0.7](https://github.com/mattn/go-runewidth/tree/v0.0.7)
- github.com/mindprince/gonvml: [9ebdce4](https://github.com/mindprince/gonvml/tree/9ebdce4)
- github.com/niemeyer/pretty: [a10e7ca](https://github.com/niemeyer/pretty/tree/a10e7ca)
- github.com/olekukonko/tablewriter: [v0.0.4](https://github.com/olekukonko/tablewriter/tree/v0.0.4)
- sigs.k8s.io/kustomize/kustomize/v4: v4.5.7



# v1.27.0-rc.1


## Downloads for v1.27.0-rc.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes.tar.gz) | f6a57401347cb6c6329f4334d0f5b9408125784c13cac8c69288e49fac9fcf057ba9b38340e170c9ee24840bfd9c6e63df5706760837e321efc2ce4da795d6cb
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-src.tar.gz) | 62cca03a925930f58083070e3877df2d3de0fc5a2a96ce4079931fab77c77f10cdd739d1ac9f64c16c3dac107f075c6f112d8ed063e3f466d662e55271487e10

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | 3cf168843bd207f0c277465560c900e09487a3ff5eca6877afc5ca947bf164b7fc20f33c5379783ddf55380ca8370a44e400b37216229496a89309242a6f9bbf
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-darwin-arm64.tar.gz) | 1b6c7eab5bb7cfe437400049c1d5bb7320828f975c63f3ac07d9e0ef943799f5fc7e0cd283aa6486da5ada075367447963b1346c96ebdb49632535e2f90dd664
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-linux-386.tar.gz) | 7d4e84fd2fa4a06890c9a5a8fc2af2e01c1f3516eb8d164ef4c97a554536f845993f096828469e845392d39576baa7ba5fa125dabeb17c049f8556ff07d941e6
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | f300c80e465f32b3c365e9c036c02aaaa3e713fa9e7318ce9e426406e07be6f3e6166664d9d940e2c3f073c3044f2d677f32a14e87a1e40acd973625634baf59
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-linux-arm.tar.gz) | 5152fddf3a01b83c89070403d37c43b46daad2ccf566f484f9f9eaa776b3d022672d902dbef1edd7b8f4241c035208f83d7049c2cf6b76ef1da8d7ffb41d86b3
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | 955c4019d6e52475bb6cf1df7d3db70ea6dbe226b54e145260f4470c53a0c12128e747ac0fae9843b2ed9d598eb3fbe964df2999a999f89726abd63feadb7ccf
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | feed26863b79c6a67c3ab7842e11663fa4861982666b451ff13d89c7659482ed08c174b2d70517ef542ba17423e5aff172e30806e17e473815c05d5f6d1c431e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | 6d3704b1edc07ff244b85e9a387f283524cc9975a527a23c567af0554ebe61f564d10068fe6af5635d71ed2d9e01ca1dc79716177c449493cbbd32903b4c11fd
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-windows-386.tar.gz) | 4abff8190b3e843b603fae1b6236815c7f15510735ed25f24ffe2ed31a0e904495431b9314366a042eadd1312a3c2e7d97ab4748ac0b67bdb0d0c3b0322a4b36
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | f903c4b8397954a11c10463e1fde15c95805fc048988e66c5d59cba5155e244c12cea98e0502365705c3cae138d251a1a87c2bb29da0e1762e84d08ff225ccc4
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-client-windows-arm64.tar.gz) | 4ac1bcba49774c9eba9a562ebc007bbe5963575c9e55917128171b955cc95bc2447352668fe768130b3013fbc425dfc86ed17e6767d59441e3566a456d6356c7

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | 3c726e0583497813ba93953e17ec93805d18129401ad3e851e44c74d4a2ecb45c2340bf82b53e28c4e480222e7cf85fa4270506103233fbad7f38ac751b16c2d
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | 17ef3d22f86328c2faea3df3b476f5b949e0bab2964f878e2b7b61965381ce3dfe43702aea2cfb9ed191e4e3d41ebbe780eccaa57b943d019cabb2adddafe458
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | d35ba5854c853d095f7588372677f1da2a5b3ef5fde846b8cce2925e7ff4fca21b9f3c9afc8eb82a973027a35918887f2d838cf5c3ec29fb2c7af576a2493efb
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | 79568358644e993126aaaf14a4126de5abae4ba2cbf59f264de57045189b53c1ead4ce8bdc75592164537ed6b836261feab1f2f5aca07f218ba5c2db1c80318c

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | dd1feed73c4d0ea7be79338500ae413dc5f55a69da0e2a27cf8e44d14f49a514b262c69b6069d1cd709564fac6e030cef581204b3d48e5208b9967a9d42b69af
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | 23421b4f6c4b21168302fa22ddf193f9d2db6a582edfded4cbf88dfa646b1f5f65817165f49bf251cb5c32aaf51a92e425cd92051745f890bfd314a82e292573
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | 9149c0bd4765484351e703d074e1554ca0735bed13bc7da3ba511ba8b6e547337f2c2c904e6db8256306b8089018ac66857a212bcb555034487111c07170c5b6
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | 1fc2f78314fc7781f5296ebe82cce905eb841bb551919261571f549f3d39f53dd8b9546f060f31209e129ff9a4dafbb50ebfc68741525b01d82639ef8a48e12c
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | 3ab964ee2fa017e96adf606ce8e8540bc3b79323141548247415e5865dd7447dca4146706cf21a1e36512d6f7950ed866351f9e6427e27c20b27118f81bf5d6c

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.0-rc.0

## Changes by Kind

### Feature

- Kubernetes is now built with Go 1.20.3 ([#117125](https://github.com/kubernetes/kubernetes/pull/117125), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Updated distroless iptables to use released image `registry.k8s.io/build-image/distroless-iptables:v0.2.3` ([#117126](https://github.com/kubernetes/kubernetes/pull/117126), [@xmudrii](https://github.com/xmudrii)) [SIG Testing]

### Bug or Regression

- Recreate DaemonSet pods completed with Succeeded phase ([#117073](https://github.com/kubernetes/kubernetes/pull/117073), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- The encryption response from KMS v2 plugins is now validated earlier at DEK generation time instead of waiting until an encryption is performed. ([#116877](https://github.com/kubernetes/kubernetes/pull/116877), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/rogpeppe/go-internal: [v1.9.0 → v1.10.0](https://github.com/rogpeppe/go-internal/compare/v1.9.0...v1.10.0)

### Removed
_Nothing has changed._



# v1.27.0-rc.0


## Downloads for v1.27.0-rc.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes.tar.gz) | 00c1377aacf2540f9dd92538e95e4d676bb77839edf59645dec6be96d4988d64b79f0a2f4a3e604a42a8c06710a88efc86d7c4bdab8d11269aadbbeaa8f02cc0
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-src.tar.gz) | e39d9fe4d1426ad35db1593c75d9bcc1e64178fac46a4c759aeb24cf37e061e1e559ab2fe8d3c4f7f66e813a11aa730aefc06649ba5ff3e8a7ac5b4db79db278

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | 811e5f52ee5f000bbac5ef5f45a7266da96ec56054056e793fab1b39dfcea2c872b6c464d163f9ff445e928750ac7fe04539b38751646c06ec204c10968d3114
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | 4a1b6ee903132f23d153369bdf97b40eeb7d111c67f60fa6367909c0bb6dd82ec7ad017ced4cb381bdbe272db514f5e6f12576b8f4b5f384a4cbd544a39268f5
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-linux-386.tar.gz) | a422191ad2118c8f1debd73834d5a963d1992441c4c0a917ebfd64818f59038d1971589b9a6f8ba7252949b5bfde62dbfab60da8783502733550c5f65cfae592
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | aca2b4ff673b381da5a979266395b4eea61c48ba59b6ae81555da21394fd535bb4b408ba584586a1336c02bf0706a90ed53ee9b6d0712519612d60b1c1f7b59c
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-linux-arm.tar.gz) | dd5d9d5adc928e114a9a35d1f845fa98b2b236a3937f056adce5c7280f91d7024f013329e3a9a17329f1247c42a1af7c011d37bfb7830937af2cf14babec3dc4
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | 0dff54bcb39c82a7142dea53ccec384c12637b9b3f261d67338d7ab1508a32897baa29657ffeeaf9bf8e65d11feb25b83349749671c7fb2a38e1a44491f50716
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | 28538ade0567bfd90f3b3975ba4e1e5c986fc2a0e7e02d4cd5f93b22a9f7ec9a0b82226938a0512a1b95149e5d801c16b46c9181aff33cf4b02ec958fefdcd73
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | 39cbbed42e2e53955e0b76306046f7ad06a88a800902c76584fe9a8a2741349c1cd74ebf4e9d0c7711e5721dd2b028df3f882eeee7242596c747b39f7ddadf87
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-windows-386.tar.gz) | 6f5947e9d4760bb00393d9bd8a2a5d389306cb51e0cc46012becaaa2ebbc4f6d0b63d239a7b99f24136ec6ce50ea51032659a68daedd79b8a494acc2cc966e09
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | 5436de39bcb3edbc7db16e8628e161a0c1fea9cf501ec6969622f3980c2b009f62137f9a60bff77b33751d3188f24812b13bafe4b397e341cd3e8979459f9972
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | 9390bc41779539e2475e992969a6f41eb68f82082d75abb793a8c9b832b86f4de42be6e313fb963b9e50ec9fd3265d9b8c8b223f7999d823a74062baaf536406

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | 652e12dd953d3dbd0fc11c02d145cfe608c41639077a056cc8ecef8d427a73c937d5fc341b8d7e4adb9a68bb6babc3165315106fe554a7ea961f1e1a45cf9566
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | c2d93520b9a7e6554207477527f408698d1b086c9ee6c6c5716f42e007367b15d1d137c723768d0d8b55cc4f1b32e5f2c57bb05b589c3a2ffe71af9c5626e303
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | f388a0b93b722814e51c0de14dd332fbe4517549a717237fa3d2d75503d5f0db7d73db154e07bc958e40ca4bea11f88312b5c8b334b77d9494637b0da4046b2e
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | 2ff2af56a4306b807e6792524489c4e2e04b2e9b661b767fe9efa34ede151ff6281b30d2a86332ea9ad69ef3031d96aa072bd099a893ad8db0be81c5c1215d98

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | 2fbe286b501a1d84bad54c035fd5daecff6268b21adec003ad90b5c6c813964d00853e8014ea8d4fc9e748586801f3231ad5fa405f213c1fc15b5b0e1819eda8
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | 1531ac12836e354269f2c0df254593d37f5860352a408c931594735212aa3504fd0ba21bee6ac8df0081d9502d307f093d6932536bd6a48124884e8925a2a76d
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | 87e3599e471a33e922235d6c54eb76c477100c04b185d897896b2942396b609492e851f5537de46b08973732e7518ccdbf5fe88c015b1479bf899e6e8c2549b4
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | 4c67645f9658809cf4357f94c0a8363a9ef4535ee07e7212565443f5e35a3037d87293a57e4b5fdc7cdca4ef946f8f0e1db09f4db87d747230da21df5f59bd8b
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | e5bd3034027e38bfe9ffd81f85a3d4c5f5a3e07542d5d718aa4e89d4d4662530316ac7c3b17bbaeb514ecdcc553117688c58899d067b89ffc0e8f3a282e119c2

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.0-beta.0

## Changes by Kind

### API Change

- Added a new alpha API: ClusterTrustBundle (`certificates.k8s.io/v1alpha1`).
  A ClusterTrustBundle may be used to distribute [X.509](https://www.itu.int/rec/T-REC-X.509) trust anchors to workloads within the cluster. ([#113218](https://github.com/kubernetes/kubernetes/pull/113218), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Auth and Testing]
- Remove `kubernetes.io/grpc` standard appProtocol ([#116866](https://github.com/kubernetes/kubernetes/pull/116866), [@LiorLieberman](https://github.com/LiorLieberman)) [SIG API Machinery and Apps]

### Feature

- Give terminal phase correctly to all pods that will not be restarted. 
  
  In particular, assign Failed phase to pods which are deleted while pending. Also, assign a terminal 
  phase (Succeeded or Failed, depending on the exit statuses of the pod containers) to pods which
  are deleted while running.
  
  This fixes the issue for jobs using pod failure policy (with JobPodFailurePolicy and PodDisruptionConditions 
  feature gates enabled) that their pods could get stuck in the pending phase when deleted. ([#115331](https://github.com/kubernetes/kubernetes/pull/115331), [@mimowo](https://github.com/mimowo)) [SIG Cloud Provider, Node and Testing]

### Bug or Regression

- Fixed two regressions introduced by the PodDisruptionConditions feature (on by default in 1.26):
  - pod eviction API calls returned spurious precondition errors and required a second evict API call to succeed
  - dry-run eviction API calls persisted a DisruptionTarget condition into the pod being evicted ([#116554](https://github.com/kubernetes/kubernetes/pull/116554), [@atiratree](https://github.com/atiratree)) [SIG API Machinery and Testing]
- Fixes a regression in the pod binding subresource to honor the `metadata.uid` precondition.
  This allows kube-scheduler to ensure it is assigns node names to the same instances of pods it made scheduling decisions for. ([#116550](https://github.com/kubernetes/kubernetes/pull/116550), [@alculquicondor](https://github.com/alculquicondor)) [SIG API Machinery and Testing]
- Fixes bug in beta aggregated discovery endpoint which caused CRD discovery information to be temporarily missing when an Aggregated APIService with the same GroupVersion is deleted (and vice versa). ([#116770](https://github.com/kubernetes/kubernetes/pull/116770), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.27.0-beta.0


## Downloads for v1.27.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes.tar.gz) | a648cbc81d762e1b37f673871906ebe7f3b871f0a3c527d0dcfb5d20a9f4eff519354155d6a2cec8deabc2f0e9db8bb4b6ac2215597a11caad396e9d31461944
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-src.tar.gz) | 2cb02e63a58590dc65962f42a6be484b804595adbecb1bcbfaf94186004bb3f9e0000aa8be9e1fb270de89733ea3baa0853211673e8c2f76d6be436782bba5dd

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | 957d1abe4282ae6bba75732b83f858b5c3a61de4148c947862bbc90f0ecf290a3cd94eb267da2127bb2ff28237a50c0b913c261c014e06580a766f69e4b45d5b
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | 5827723ec6bc6f0d96cd20046bd736a3045f168cbf78a9064645f0e94653f3e751bcca6d18836aa038cb726ab991a48b1451fcc00bd0e751eb0af30d7bf002aa
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-linux-386.tar.gz) | bdacf9b42269238e97b6301a975c4accd7363a05a63a35305d0d74916c138c70985491ac9d13a152d0b10609f265aede4a910ebed61bdf1b8a37264773dffd3b
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | e139daa8df28d13ad8625c819ba94e6e4dd7805c89dd2a0bba6ce478a2bc7d9b52a3fccc18de08c13dca1b98c693d50d37599e8a3b34b7a1f39401098dea2df5
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-linux-arm.tar.gz) | 3f669851c6317d67bbcae591056ee9cfda6e9bca3eeac02cc41eae35db3448e745e123ab75da8b9dbb546172b07d625bf821da3b0a1b6420d41140eb7b96b474
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | c8f394650db292a117e1db5a76775541087ab0da9b3d43041d50f3126ef47a0dcb65ebbe61d8be9bdb67adce1c43d5f7a695ff0b9909c8c9461d6937ebe9160f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | 83b51c787f57b698584c3c585a772470819260008808a2102a9e765ef1458d9bb536aeb3e2587d391c6efb06d56326f1c8b47f12ab98069d1605ef210ecd6e8c
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | d79766f56263a78549d7e2bc8f93977d8730435beeb7fe9413686d09ac6a6edc8a868621023623656782272e518fa7955275ab0d4aecb8a71cb4ba544dd5f77d
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-windows-386.tar.gz) | 399741ba92a59c0c3640f4d4d0c961b63bd24ba8a5ce036f4a82dcd040a0d2873e7e3237af10da1b2982af5ca6ae8edb2a4d023db3af87dfae6c90528a487de3
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | 3eba7adbb6c7c386d04bdddcd6d66ca7f5799789680c7fbb9216a0520884264dc5fdb35a0417d03d77955097c6341a30e3e07d077266c2ed2f96d1765f344e39
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | 158475196f75764dd115e187a5fb27894367a8a2ddad755e3d542e5f225fe9bad476f592c0b7fad2a3dded4638ccec2a1f717eec4d04c8e510334a3a410e0541

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | fb9caa627e77d1bd39b11106dd95c9dd008c5d418234636a0beddd48e59c980d4924ed3006133e20d2ac0715a4353d14a90f7ebc5345804f24160a13efb7a2b5
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | 34f61cfeba8adf7fd3dd83599e34ed36d5942a41904f0430a7b8a5078d306283a4dd7eec40716c8aa6f4ff87dea1faa588fff66a2c388aac8c7b461a64366c33
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | c37a226fa7b6d35b32420c13e67482820f4b23cd9dc9c23820d8f3024bf969d2acc96dd31267a964a73e3a4a61a046c778ab3443598b111eccbf20a682b93f40
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | b2f29641f5756bb77b048cd336997e89ae50236fb32a7b425c348fab1f077534facce6c90ad9650dd2db5b708bff1ddabb478e29fc69f32b59e5ded247665840

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 305ee41682bb222e040134e75aefeda6cad1f81f4af761c514bb5d66fe83d42dd993c0a118c178a9e8abd6d2ae3fdb7b70c0509f1134f032c2ef2ef2bc103d81
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | 6bc84fb35f278742734ac0c6265d6f2d654a7d57d65e98d597ba4c438b7ea20033e0431515f120fbcbf2fb6e99d3f50d4b4ecfc88e3705d08fc949b7f42c3776
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 638ad423ddbc52179320fe497f775d50c210745044aca9cea00c674dc1e710e979b7fca564811ccae99b801582e075194b09a00548f789740e0e6c4791309bdc
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | 4874d3e34145c19973aa130c3f2c4eb5b01991142eb9bbf7391378bb6f83179a163659c80b3e45526cf334f7c63868502381afce18205ab92c521f4c911e3179
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 5d3e9e88577e5be11d56e65d76cec6ab931811f106fd1683551d9b2514ec8edf21f39c6512adc3ce901862f015b28237fc1774b0ccfaf771f106237a2ed599c6

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.0-alpha.3

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

  - The `IPv6DualStack` feature gate for external cloud providers was removed.
  (The feature became GA in 1.23 and the gate was removed for all other
  components several releases ago.) If you were still manually
  enabling it you must stop now. ([#116255](https://github.com/kubernetes/kubernetes/pull/116255), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Cloud Provider and Network]
 
## Changes by Kind

### Deprecation

- The alpha SecurityContextDeny admission plugin is deprecated and now requires enabling the alpha `SecurityContextDeny` feature gate to use. It will be removed in a future version. ([#115879](https://github.com/kubernetes/kubernetes/pull/115879), [@mtardy](https://github.com/mtardy)) [SIG Auth]

### API Change

- API: resource.k8s.io/v1alpha1.PodScheduling was renamed to resource.k8s.io/v1alpha2.PodSchedulingContext. ([#116556](https://github.com/kubernetes/kubernetes/pull/116556), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Node, Scheduling and Testing]
- APIServerTracing feature gate is now enabled by default. Tracing in the API Server is still disabled by default, and requires a config file to enable. ([#116144](https://github.com/kubernetes/kubernetes/pull/116144), [@dashpole](https://github.com/dashpole)) [SIG API Machinery and Testing]
- Added CEL runtime cost calculation into ValidatingAdmissionPolicy, matching the evaluation cost
  restrictions that already apply to CustomResourceDefinition.
  If rule evaluation uses more compute than the limit, the API server aborts the evaluation and the
  admission check that was being performed is aborted; the `failurePolicy` for the ValidatingAdmissionPolicy
  determines the outcome. ([#115747](https://github.com/kubernetes/kubernetes/pull/115747), [@cici37](https://github.com/cici37)) [SIG API Machinery]
- Added `messageExpression` to `ValidatingAdmissionPolicy`, to set custom failure message via CEL expression. ([#116397](https://github.com/kubernetes/kubernetes/pull/116397), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery]
- Added a new IPAddress object kind
  - Added a new ClusterIP allocator. The new allocator removes previous Service CIDR block size limitations for IPv4, and limits IPv6 size to a /64 ([#115075](https://github.com/kubernetes/kubernetes/pull/115075), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Auth, CLI, Cluster Lifecycle, Network and Testing]
- Added a new alpha API: ClusterTrustBundle (`certificates.k8s.io/v1alpha1`).
  A ClusterTrustBundle may be used to distribute [X.509](https://www.itu.int/rec/T-REC-X.509) trust anchors to workloads within the cluster. ([#113218](https://github.com/kubernetes/kubernetes/pull/113218), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Auth and Testing]
- Added authorization check support to the CEL expressions of ValidatingAdmissionPolicy via a `authorizer`
  variable with expressions. The new variable provides a builder that allows expressions such `authorizer.group('').resource('pods').check('create').allowed()`. ([#116054](https://github.com/kubernetes/kubernetes/pull/116054), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]
- Added matchConditions field to ValidatingAdmissionPolicy, enabled support for CEL based custom match criteria. ([#116350](https://github.com/kubernetes/kubernetes/pull/116350), [@maxsmythe](https://github.com/maxsmythe)) [SIG API Machinery and Testing]
- Added messageExpression field to ValidationRule. (#115969, @DangerOnTheRanger) ([#115969](https://github.com/kubernetes/kubernetes/pull/115969), [@DangerOnTheRanger](https://github.com/DangerOnTheRanger)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Instrumentation, Node and Testing]
- Added the `MatchConditions` field to `ValidatingWebhookConfiguration` and `MutatingWebhookConfiguration` for the v1beta and v1 apis. 
  
  The `AdmissionWebhookMatchConditions` featuregate is now in Alpha ([#116261](https://github.com/kubernetes/kubernetes/pull/116261), [@ivelichkovich](https://github.com/ivelichkovich)) [SIG API Machinery and Testing]
- Added validation to ensure that if `service.kubernetes.io/topology-aware-hints` and `service.kubernetes.io/topology-mode` annotations are both set, they are set to the same value.
  - Added deprecation warning if `service.kubernetes.io/topology-aware-hints` annotation is used. ([#116612](https://github.com/kubernetes/kubernetes/pull/116612), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- Adds auditAnnotations to ValidatingAdmissionPolicy, enabling CEL to be used to add audit annotations to request audit events.
  Adds validationActions to ValidatingAdmissionPolicyBinding, enabling validation failures to be handled by any combination of the warn, audit and deny enforcement actions. ([#115973](https://github.com/kubernetes/kubernetes/pull/115973), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]
- Adds feature gate `NodeLogQuery` which provides cluster administrators with a streaming view of logs using kubectl without them having to implement a client side reader or logging into the node. ([#96120](https://github.com/kubernetes/kubernetes/pull/96120), [@LorbusChris](https://github.com/LorbusChris)) [SIG API Machinery, Apps, CLI, Node, Testing and Windows]
- Api: validation of a PodSpec now rejects invalid ResourceClaim and ResourceClaimTemplate names. For a pod, the name generated for the ResourceClaim when using a template also must be valid. ([#116576](https://github.com/kubernetes/kubernetes/pull/116576), [@pohly](https://github.com/pohly)) [SIG Apps]
- Bump default API QPS limits for Kubelet. ([#116121](https://github.com/kubernetes/kubernetes/pull/116121), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Node]
- Enable the "StatefulSetStartOrdinal" feature gate in beta ([#115260](https://github.com/kubernetes/kubernetes/pull/115260), [@pwschuurman](https://github.com/pwschuurman)) [SIG API Machinery and Apps]
- Extended the kubelet's PodResources API to include resources allocated in `ResourceClaims` via `DynamicResourceAllocation`. Additionally, added a new `Get()` method to query a specific pod for its resources. ([#115847](https://github.com/kubernetes/kubernetes/pull/115847), [@moshe010](https://github.com/moshe010)) [SIG Node]
- Forbid to set matchLabelKeys when labelSelector isn’t set in topologySpreadConstraints ([#116535](https://github.com/kubernetes/kubernetes/pull/116535), [@denkensk](https://github.com/denkensk)) [SIG API Machinery, Apps and Scheduling]
- GCE does not support LoadBalancer Services with ports with different protocols (TCP and UDP) ([#115966](https://github.com/kubernetes/kubernetes/pull/115966), [@aojea](https://github.com/aojea)) [SIG Apps and Cloud Provider]
- GRPC probes are now a GA feature. GRPCContainerProbe feature gate was locked to default value and will be removed in v1.29. If you were setting this feature gate explicitly, please remove it now. ([#116233](https://github.com/kubernetes/kubernetes/pull/116233), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Apps and Node]
- Graduate Kubelet Topology Manager to GA. ([#116093](https://github.com/kubernetes/kubernetes/pull/116093), [@swatisehgal](https://github.com/swatisehgal)) [SIG API Machinery, Node and Testing]
- Graduate `KubeletTracing` to beta, which means that the feature gate is now enabled by default. ([#115750](https://github.com/kubernetes/kubernetes/pull/115750), [@saschagrunert](https://github.com/saschagrunert)) [SIG Instrumentation and Node]
- Graduate the container resource metrics feature on HPA to beta. ([#116046](https://github.com/kubernetes/kubernetes/pull/116046), [@sanposhiho](https://github.com/sanposhiho)) [SIG Autoscaling]
- Introduced a breaking change to the `resource.k8s.io` API in its `AllocationResult` struct. This change allows a kubelet plugin for the `DynamicResourceAllocation` feature to service allocations from multiple resource driver controllers. ([#116332](https://github.com/kubernetes/kubernetes/pull/116332), [@klueska](https://github.com/klueska)) [SIG API Machinery, Apps, CLI, Node, Scheduling and Testing]
- Introduces new alpha functionality to the reflector, allowing user to enable API streaming.
  
  To activate this feature, users can set the `ENABLE_CLIENT_GO_WATCH_LIST_ALPHA` environmental variable.
  It is important to note that the server must support streaming for this feature to function properly.
  If streaming is not supported by the server, the reflector will revert to the previous method
  of obtaining data through LIST/WATCH semantics. ([#110772](https://github.com/kubernetes/kubernetes/pull/110772), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kubelet: change MemoryThrottlingFactor default value to 0.9 and formulas to calculate memory.high ([#115371](https://github.com/kubernetes/kubernetes/pull/115371), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery, Apps and Node]
- Migrated the DaemonSet controller (within `kube-controller-manager) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging) ([#113622](https://github.com/kubernetes/kubernetes/pull/113622), [@249043822](https://github.com/249043822)) [SIG API Machinery, Apps, Instrumentation and Testing]
- New `service.kubernetes.io/topology-mode` annotation has been introduced as a replacement for the `service.kubernetes.io/topology-aware-hints` annotation.
  - `service.kubernetes.io/topology-aware-hints` annotation has been deprecated.
  - kube-proxy now accepts any value that is not "disabled" for these annotations, enabling custom implementation-specific and/or future built-in heuristics to be used. ([#116522](https://github.com/kubernetes/kubernetes/pull/116522), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- NodeResourceFit and NodeResourcesBalancedAllocation implement the PreScore extension point for a more performant calculation. ([#115655](https://github.com/kubernetes/kubernetes/pull/115655), [@tangwz](https://github.com/tangwz)) [SIG Scheduling]
- Pods owned by a Job will now use the labels `batch.kubernetes.io/job-name` and `batch.kubernetes.io/controller-uid`.
  The legacy labels `job-name` and `controller-uid` are still added for compatibility. ([#114930](https://github.com/kubernetes/kubernetes/pull/114930), [@kannon92](https://github.com/kannon92)) [SIG Apps]
- Promote CronJobTimeZone feature to GA ([#115904](https://github.com/kubernetes/kubernetes/pull/115904), [@soltysh](https://github.com/soltysh)) [SIG API Machinery and Apps]
- Promoted `SelfSubjectReview` to Beta ([#116274](https://github.com/kubernetes/kubernetes/pull/116274), [@nabokihms](https://github.com/nabokihms)) [SIG API Machinery, Auth, CLI and Testing]
- Relax API validation to allow pod node selector to be mutable for gated pods (additions only, no deletions or mutations). ([#116161](https://github.com/kubernetes/kubernetes/pull/116161), [@danielvegamyhre](https://github.com/danielvegamyhre)) [SIG Apps, Scheduling and Testing]
- Remove deprecated `--enable-taint-manager` and `--pod-eviction-timeout` CLI flags ([#115840](https://github.com/kubernetes/kubernetes/pull/115840), [@atosatto](https://github.com/atosatto)) [SIG API Machinery, Apps, Node and Testing]
- Resource.k8s.io/v1alpha1 was replaced with resource.k8s.io/v1alpha2. Before upgrading a cluster, all objects in resource.k8s.io/v1alpha1 (ResourceClaim, ResourceClaimTemplate, ResourceClass, PodScheduling) must be deleted. The changes will be internal, so YAML files which create pods and resource claims don't need changes except for the newer `apiVersion`. ([#116299](https://github.com/kubernetes/kubernetes/pull/116299), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, CLI, Node, Scheduling and Testing]
- SELinuxMountReadWriteOncePod graduated to Beta. ([#116425](https://github.com/kubernetes/kubernetes/pull/116425), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]
- StatefulSetAutoDeletePVC feature gate promoted to beta. ([#116501](https://github.com/kubernetes/kubernetes/pull/116501), [@mattcary](https://github.com/mattcary)) [SIG Apps, Auth and Testing]
- The API server now re-uses data encryption keys while the kms v2 plugin's key ID is stable.  Data encryption keys are still randomly generated on server start but an atomic counter is used to prevent nonce collisions. ([#116155](https://github.com/kubernetes/kubernetes/pull/116155), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- The API server's encryption at rest configuration now allows the use of wildcards in the list of resources.  For example, '*.*' can be used to encrypt all resources, including all current and future custom resources. ([#115149](https://github.com/kubernetes/kubernetes/pull/115149), [@nilekhc](https://github.com/nilekhc)) [SIG API Machinery, Auth and Testing]
- Update KMSv2 to beta ([#115123](https://github.com/kubernetes/kubernetes/pull/115123), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Updated: Redefine AppProtocol field description and add new standard values ([#115433](https://github.com/kubernetes/kubernetes/pull/115433), [@LiorLieberman](https://github.com/LiorLieberman)) [SIG API Machinery, Apps and Network]
- ValidatingAdmissionPolicy now provides a status field that contains results of type checking the validation expression.
  The type checking is fully informational, and the behavior of the policy is unchanged. ([#115668](https://github.com/kubernetes/kubernetes/pull/115668), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery, Auth, Cloud Provider and Testing]
- We have removed support for the v1alpha1 kubeletplugin API of DynamicResourceManagement. All plugins must update to v1alpha2 in order to function properly going forward. ([#116558](https://github.com/kubernetes/kubernetes/pull/116558), [@klueska](https://github.com/klueska)) [SIG API Machinery, Apps, CLI, Node, Scheduling and Testing]

### Feature

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
  --> ([#113428](https://github.com/kubernetes/kubernetes/pull/113428), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery, Apps, Instrumentation and Testing]
- Add e2e test to node expand volume with secret ([#115451](https://github.com/kubernetes/kubernetes/pull/115451), [@zhucan](https://github.com/zhucan)) [SIG Storage and Testing]
- Added NewVolumeManagerReconstruction feature gate and enable it by default to enable updated discovery of mounted volumes during kubelet startup. Please watch for kubelet getting stuck at startup and / or not unmounting volumes from deleted Pods and report any issues in this area. ([#115268](https://github.com/kubernetes/kubernetes/pull/115268), [@jsafrane](https://github.com/jsafrane)) [SIG Node and Storage]
- Added metrics for volume reconstruction during kubelet startup. ([#115965](https://github.com/kubernetes/kubernetes/pull/115965), [@jsafrane](https://github.com/jsafrane)) [SIG Node and Storage]
- Added the ability to host webhooks in the cloud controller manager. ([#108838](https://github.com/kubernetes/kubernetes/pull/108838), [@nckturner](https://github.com/nckturner)) [SIG API Machinery, Cloud Provider and Testing]
- Adding e2e tests for kubectl --subresource for beta graduation ([#116590](https://github.com/kubernetes/kubernetes/pull/116590), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG CLI and Testing]
- Adds --output plaintext-openapiv2 argument to kubectl explain to use old openapiv2 `explain` implementation. ([#115480](https://github.com/kubernetes/kubernetes/pull/115480), [@alexzielenski](https://github.com/alexzielenski)) [SIG Architecture, Auth, CLI, Cloud Provider and Node]
- By enabling the `UserNamespacesStatelessPodsSupport` feature gate in kubelet, you can now run a stateless pod in a separate user namespace ([#116377](https://github.com/kubernetes/kubernetes/pull/116377), [@giuseppe](https://github.com/giuseppe)) [SIG Apps, Node and Storage]
- By enabling the alpha `CloudNodeIPs` feature gate in kubelet and the cloud
  provider, you can now specify a dual-stack `--node-ip` value (when using an
  external cloud provider that supports that functionality). ([#116305](https://github.com/kubernetes/kubernetes/pull/116305), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Cloud Provider, Network and Node]
- Change kubectl --subresource flag to beta ([#116595](https://github.com/kubernetes/kubernetes/pull/116595), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG CLI]
- Changed metrics for aggregated discovery to publish new time series (alpha). ([#115630](https://github.com/kubernetes/kubernetes/pull/115630), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Testing]
- Graduate CRI Events driven Pod LifeCycle Event Generator (Evented PLEG) to Beta ([#115967](https://github.com/kubernetes/kubernetes/pull/115967), [@harche](https://github.com/harche)) [SIG Node]
- Graduated `matchLabelKeys` in `podTopologySpread` to Beta ([#116291](https://github.com/kubernetes/kubernetes/pull/116291), [@denkensk](https://github.com/denkensk)) [SIG Scheduling]
- Graduates the CSINodeExpandSecret feature to Beta. This feature facilitates passing secrets to CSI driver as part of Node Expansion CSI operation. ([#115621](https://github.com/kubernetes/kubernetes/pull/115621), [@humblec](https://github.com/humblec)) [SIG Storage]
- HPA controller exposes the following metrics from the kube-controller-manager.
  - `metric_computation_duration_seconds`: Number of metric computations. 
  - `metric_computation_total`: The time(seconds) that the HPA controller takes to calculate one metric. ([#116326](https://github.com/kubernetes/kubernetes/pull/116326), [@sanposhiho](https://github.com/sanposhiho)) [SIG Apps, Autoscaling and Instrumentation]
- HPA controller starts to expose metrics from the kube-controller-manager.
  - `reconciliations_total`: Number of reconciliation of HPA controller. 
  - `reconciliation_duration_seconds`: The time(seconds) that the HPA controller takes to reconcile once. ([#116010](https://github.com/kubernetes/kubernetes/pull/116010), [@sanposhiho](https://github.com/sanposhiho)) [SIG Apps, Autoscaling and Instrumentation]
- Kube-scheduler: Optimized implementation of null labelSelector in topology spreading. ([#116607](https://github.com/kubernetes/kubernetes/pull/116607), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Kubelet allows pods to use the `net.ipv4.ip_local_reserved_ports` sysctl by default and the minimal kernel version is 3.16; Pod Security admission allows this sysctl in v1.27+ versions of the baseline and restricted policies. ([#115374](https://github.com/kubernetes/kubernetes/pull/115374), [@pacoxu](https://github.com/pacoxu)) [SIG Auth, Network and Node]
- Kubernetes is now built with go 1.20.2 ([#116404](https://github.com/kubernetes/kubernetes/pull/116404), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Locks CSIMigrationvSphere feature gate. ([#116610](https://github.com/kubernetes/kubernetes/pull/116610), [@xing-yang](https://github.com/xing-yang)) [SIG Storage]
- Make `apiextensions-apiserver` binary linking static (also affects the deb and rpm packages). ([#114226](https://github.com/kubernetes/kubernetes/pull/114226), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery and Release]
- Make `kube-aggregator` binary linking static (also affects the deb and rpm packages). ([#114227](https://github.com/kubernetes/kubernetes/pull/114227), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery and Release]
- Migrated controller helper functions to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#115049](https://github.com/kubernetes/kubernetes/pull/115049), [@fatsheep9146](https://github.com/fatsheep9146)) [SIG Apps]
- Migrated the ClusterRole aggregation controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113910](https://github.com/kubernetes/kubernetes/pull/113910), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery, Apps and Instrumentation]
- Migrated the Deployment controller (within `kube-controller-manager) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging) ([#113525](https://github.com/kubernetes/kubernetes/pull/113525), [@249043822](https://github.com/249043822)) [SIG API Machinery, Apps, Instrumentation and Testing]
- Migrated the StatefulSet controller (within `kube-controller-manager) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging) ([#113840](https://github.com/kubernetes/kubernetes/pull/113840), [@249043822](https://github.com/249043822)) [SIG API Machinery, Apps, Instrumentation and Testing]
- Migrated the bootstrap signer controller and the token cleaner controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113464](https://github.com/kubernetes/kubernetes/pull/113464), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery, Apps and Instrumentation]
- Migrated the defaultbinder scheduler plugin to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116571](https://github.com/kubernetes/kubernetes/pull/116571), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Scheduling]
- Migrated the main kube-controller-manager binary to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116529](https://github.com/kubernetes/kubernetes/pull/116529), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth and Node]
- Migrated the replicaset controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#114871](https://github.com/kubernetes/kubernetes/pull/114871), [@Namanl2001](https://github.com/Namanl2001)) [SIG API Machinery, Apps, Instrumentation and Testing]
- Migrated the service-account controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#114918](https://github.com/kubernetes/kubernetes/pull/114918), [@Namanl2001](https://github.com/Namanl2001)) [SIG API Machinery, Apps, Auth, Instrumentation and Testing]
- Migrated the volume attach/detach controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging).
  Migrated the PersistentVolumeClaim protection controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging).
  Migrated the PersistentVolume protection controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113584](https://github.com/kubernetes/kubernetes/pull/113584), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG API Machinery, Apps, Instrumentation, Node, Scheduling, Storage and Testing]
- Migrated the “TTL after finished” controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113916](https://github.com/kubernetes/kubernetes/pull/113916), [@songxiao-wang87](https://github.com/songxiao-wang87)) [SIG API Machinery, Apps, Instrumentation and Testing]
- New "plugin_evaluation_total" is added to the scheduler. 
  This metric counts how many times the specific plugin affects the scheduling result. The metric doesn't get incremented when the plugin has nothing to do with an incoming Pod. ([#115082](https://github.com/kubernetes/kubernetes/pull/115082), [@sanposhiho](https://github.com/sanposhiho)) [SIG Instrumentation and Scheduling]
- Promote `whoami` kubectl command. ([#116510](https://github.com/kubernetes/kubernetes/pull/116510), [@nabokihms](https://github.com/nabokihms)) [SIG Auth and CLI]
- Promote aggregated discovery endpoint to beta and it will be enabled by default ([#116108](https://github.com/kubernetes/kubernetes/pull/116108), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Promoted `OpenAPIV3` to GA ([#116235](https://github.com/kubernetes/kubernetes/pull/116235), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- StorageVersionGC (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113986](https://github.com/kubernetes/kubernetes/pull/113986), [@songxiao-wang87](https://github.com/songxiao-wang87)) [SIG API Machinery, Apps and Testing]
- Switched kubectl explain to use OpenAPIV3 information published by the server. OpenAPIV2 backend can  still be used with the `--output plaintext-openapiv2` argument ([#116390](https://github.com/kubernetes/kubernetes/pull/116390), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery, CLI and Testing]
- The job controller back-off logic is now decoupled from workqueue. In case of parallelism > 1, if there are multiple new failures in a reconciliation cycle, all the failures are taken into account to compute the back-off. Previously, the back-off kicked in for all types of failures; with this change, only pod failures are taken into account. If the back-off limits exceeds, the job is marked as failed immediately; before this change, the job is marked as failed in the next back-off. ([#114768](https://github.com/kubernetes/kubernetes/pull/114768), [@sathyanarays](https://github.com/sathyanarays)) [SIG Apps and Testing]
- The scheduler's metric "plugin_execution_duration_seconds" now records PreEnqueue plugins execution seconds. ([#116201](https://github.com/kubernetes/kubernetes/pull/116201), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Unlocked the `CSIMigrationvSphere` feature gate.
  The change allow users to continue using the in-tree vSphere driver,pending a vSphere
  CSI driver release that has with GA support for Windows, XFS, and raw block access. ([#116342](https://github.com/kubernetes/kubernetes/pull/116342), [@msau42](https://github.com/msau42)) [SIG Storage]
- Update kube-apiserver SLO/SLI latency metrics to exclude priority & fairness queue wait times ([#116420](https://github.com/kubernetes/kubernetes/pull/116420), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery]
- Updated distroless iptables to use released image `registry.k8s.io/build-image/distroless-iptables:v0.2.2`
  - Updated setcap to use released image `registry.k8s.io/build-image/setcap:bullseye-v1.4.2` ([#116509](https://github.com/kubernetes/kubernetes/pull/116509), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- Upgrades functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.0.0 and https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.0.1. 
  
  This is a new major release of kustomize, so there are a few backwards-incompatible changes, most of which are rare use cases, bug fixes with side effects, or things that have been deprecated for multiple releases already:
  
  - https://github.com/kubernetes-sigs/kustomize/pull/4911: Drop support for a very old, legacy style of patches. patches used to be allowed to be used as an alias for patchesStrategicMerge in kustomize v3. You now have to use patchesStrategicMerge explicitly, or update to the new syntax supported by patches. See examples in the PR description of https://github.com/kubernetes-sigs/kustomize/pull/4911.
  - https://github.com/kubernetes-sigs/kustomize/issues/4731: Remove a potential build-time side-effect in ConfigMapGenerator and SecretGenerator, which loaded values from the local environment under some circumstances, breaking kustomize build's side-effect-free promise. While this behavior was never intended, we deprecated it and are announcing it as a breaking change since it existed for a long time. See also the Eschewed Features documentation.
  - https://github.com/kubernetes-sigs/kustomize/pull/4985: If you previously included .git in an AWS or Azure URL, we will no longer automatically remove that suffix. You may need to add an extra / to replace the .git for the URL to properly resolve.
  - https://github.com/kubernetes-sigs/kustomize/pull/4954: Drop support for using gh: as a host (e.g. gh:kubernetes-sigs/kustomize). We were unable to find any usage of or basis for this and believe it may have been targeting a custom gitconfig shorthand syntax. ([#116598](https://github.com/kubernetes/kubernetes/pull/116598), [@natasha41575](https://github.com/natasha41575)) [SIG CLI]
- When an unsupported PodDisruptionBudget configuration is found, an event and log will be emitted to inform users of the misconfiguration. ([#115861](https://github.com/kubernetes/kubernetes/pull/115861), [@JayKayy](https://github.com/JayKayy)) [SIG Apps]
- [alpha: kubectl apply --prune --applyset] Enables certain custom resources (CRs) to be used as ApplySet parent objects. To enable this for a given CR, apply the label `applyset.k8s.io/is-parent-type: true` to the CustomResourceDefinition (CRD) that defines it . ([#116353](https://github.com/kubernetes/kubernetes/pull/116353), [@KnVerey](https://github.com/KnVerey)) [SIG CLI]

### Documentation

- The change affects the following CLI command:
  
  kubectl create rolebinding -h ([#107124](https://github.com/kubernetes/kubernetes/pull/107124), [@ptux](https://github.com/ptux)) [SIG CLI]

### Failing Test

- Setting the Kubelet config option ``--resolv-conf=Host`` on Windows will now result in Kubelet applying the Pod DNS Policies as intended. ([#110566](https://github.com/kubernetes/kubernetes/pull/110566), [@claudiubelu](https://github.com/claudiubelu)) [SIG Network, Node, Testing and Windows]

### Bug or Regression

- Expands the partial fix for https://github.com/kubernetes/kubernetes/issues/111539 which was already started in https://github.com/kubernetes/kubernetes/pull/109706 Specifically, we will now reduce the amount of syncs for ETP=local services even further in the CCM and avoid re-configuring LBs to an even greater extent. ([#111658](https://github.com/kubernetes/kubernetes/pull/111658), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider and Network]
- Fix the problem Pod terminating stuck because of trying to umount not actual mounted dir. ([#115769](https://github.com/kubernetes/kubernetes/pull/115769), [@mochizuki875](https://github.com/mochizuki875)) [SIG Node and Storage]
- Fixed a rare race condition in kube-apiserver that could lead to missing events when a watch API request was created at the same time kube-apiserver was re-initializing its internal watch. ([#116172](https://github.com/kubernetes/kubernetes/pull/116172), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Fixed data race in `kube-scheduler` when preemption races with a Pod update. ([#116395](https://github.com/kubernetes/kubernetes/pull/116395), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Fixed incorrect watch events when a watch is initialized simultanously with a reinitializing watchcache. ([#116436](https://github.com/kubernetes/kubernetes/pull/116436), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Fixed performance regression in scheduler caused by frequent metric lookup on critical code path. ([#116428](https://github.com/kubernetes/kubernetes/pull/116428), [@mborsz](https://github.com/mborsz)) [SIG Scheduling]
- Fixes #115825. Kube-proxy will now include the `healthz` state in its response to the LB HC as to avoid indicating to the LB that it should use the node in question when Kube-proxy is not healthy. ([#111661](https://github.com/kubernetes/kubernetes/pull/111661), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Network]
- Force deleted pods may fail to terminate until the kubelet is restarted when the container runtime returns an error during termination. We have strengthened testing for runtime failures and now perform a more rigorous reconciliation to ensure static pods (especially those that use fixed UIDs) are restarted.  As a side effect of these changes static pods will be restarted with lower latency than before (2s vs 4s, on average) and rapid updates to pod configuration should take effect sooner.
  
  A new metric `kubelet_known_pods` has been added at ALPHA stability to report the number of pods a Kubelet is tracking in a number of internal states.  Operators may use the metrics to track an excess of pods in the orphaned state that may not be completing. ([#113145](https://github.com/kubernetes/kubernetes/pull/113145), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery, Auth, Cloud Provider, Node and Testing]
- From now on, the HPA controller will return an error for the container resource metrics when the feature gate "HPAContainerMetrics" is disabled. As a result, HPA with a container resource metric performs no scale-down and performs only scale-up based on other metrics. ([#116043](https://github.com/kubernetes/kubernetes/pull/116043), [@sanposhiho](https://github.com/sanposhiho)) [SIG API Machinery, Apps and Autoscaling]
- Ingress with ingressClass annotation and IngressClassName both set can be created now. ([#115447](https://github.com/kubernetes/kubernetes/pull/115447), [@kidddddddddddddddddddddd](https://github.com/kidddddddddddddddddddddd)) [SIG Network]
- Kube-controller-manager: fix a bug that the "kubeconfig" field of "kubecontrollermanager.config.k8s.io" configuration is not populated correctly ([#116219](https://github.com/kubernetes/kubernetes/pull/116219), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery and Cloud Provider]
- Kubelet: fix recording issue when pulling image did finish ([#114904](https://github.com/kubernetes/kubernetes/pull/114904), [@TommyStarK](https://github.com/TommyStarK)) [SIG Node]
- PVCs will automatically be recreated if they are missing for a pending Pod. ([#113270](https://github.com/kubernetes/kubernetes/pull/113270), [@rrangith](https://github.com/rrangith)) [SIG Apps and Testing]
- PersistentVolume API objects which set NodeAffinities using beta Kubernetes labels for OS, architecture, zone, region, and instance type may now be modified to use the stable Kubernetes labels. ([#115391](https://github.com/kubernetes/kubernetes/pull/115391), [@haoruan](https://github.com/haoruan)) [SIG Apps and Storage]
- Potentially breaking change - Updating the polling interval for Windows stats collection from 1 second to 10 seconds ([#116546](https://github.com/kubernetes/kubernetes/pull/116546), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Update the Event series starting count when emitting isomorphic events from 1 to 2. ([#112334](https://github.com/kubernetes/kubernetes/pull/112334), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery and Testing]
- When GCing pods, kube-controller-manager will delete Evicted pods first. ([#116167](https://github.com/kubernetes/kubernetes/pull/116167), [@borgerli](https://github.com/borgerli)) [SIG Apps]
- Windows CPU usage node stats are now correctly calculated for nodes with multiple Processor Groups. ([#110864](https://github.com/kubernetes/kubernetes/pull/110864), [@claudiubelu](https://github.com/claudiubelu)) [SIG Node, Testing and Windows]

### Other (Cleanup or Flake)

- Added basic Denial Of Service prevention for the the node-local kubelet `podresource` API ([#116459](https://github.com/kubernetes/kubernetes/pull/116459), [@ffromani](https://github.com/ffromani)) [SIG Node and Testing]
- Introduce new metrics removing the redundant subsystem in kube-apiserver pod logs metrics and deprecate the original ones:
  - kube_apiserver_pod_logs_pods_logs_backend_tls_failure_total becomes kube_apiserver_pod_logs_backend_tls_failure_total
  - kube_apiserver_pod_logs_pods_logs_insecure_backend_total becomes kube_apiserver_pod_logs_insecure_backend_total ([#114497](https://github.com/kubernetes/kubernetes/pull/114497), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery]
- Kubelet: remove deprecated flag `--container-runtime` ([#114017](https://github.com/kubernetes/kubernetes/pull/114017), [@calvin0327](https://github.com/calvin0327)) [SIG Cloud Provider and Node]
- Kubelet: the deprecated `--master-service-namespace` flag is removed in v1.27 ([#116015](https://github.com/kubernetes/kubernetes/pull/116015), [@SataQiu](https://github.com/SataQiu)) [SIG Node]
- Linux/arm will not ship in Kubernetes 1.27 as we are running into issues with building artifacts using golang 1.20.2 (please see issue #116492) ([#115742](https://github.com/kubernetes/kubernetes/pull/115742), [@dims](https://github.com/dims)) [SIG Architecture, Release and Testing]
- Migrate `pkg/controller/nodeipam/ipam/cloud_cidr_allocator.go, pkg/controller/nodeipam/ipam/multi_cidr_range_allocator.go pkg/controller/nodeipam/ipam/range_allocator.go pkg/controller/nodelifecycle/node_lifecycle_controller.go` to structured logging ([#112670](https://github.com/kubernetes/kubernetes/pull/112670), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG API Machinery, Apps, Architecture, Cloud Provider, Instrumentation, Network and Testing]
- Migrated the Kubernetes object garbage collector (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113471](https://github.com/kubernetes/kubernetes/pull/113471), [@ncdc](https://github.com/ncdc)) [SIG API Machinery, Apps and Testing]
- Migrated the ttlafterfinished controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#115332](https://github.com/kubernetes/kubernetes/pull/115332), [@obaranov1](https://github.com/obaranov1)) [SIG Apps]
- Migrated the “sample-controller” controller to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113879](https://github.com/kubernetes/kubernetes/pull/113879), [@pchan](https://github.com/pchan)) [SIG API Machinery and Instrumentation]
- Remove Azure disk in-tree storage plugin ([#116301](https://github.com/kubernetes/kubernetes/pull/116301), [@andyzhangx](https://github.com/andyzhangx)) [SIG API Machinery, Cloud Provider, Node, Scheduling, Storage and Testing]
- Remove the following deprecated metrics:
  - node_collector_evictions_number replaced by node_collector_evictions_total
  - scheduler_e2e_scheduling_duration_seconds replaced by scheduler_scheduling_attempt_duration_seconds ([#115209](https://github.com/kubernetes/kubernetes/pull/115209), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Apps and Scheduling]
- Removed AWS kubelet credential provider. Please use the external kubelet credential provider binary named `ecr-credential-provider` instead. ([#116329](https://github.com/kubernetes/kubernetes/pull/116329), [@dims](https://github.com/dims)) [SIG Node, Storage and Testing]
- Storage.k8s.io/v1beta1 API version of CSIStorageCapacity will no longer be served ([#116523](https://github.com/kubernetes/kubernetes/pull/116523), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery]
- The `wait.Poll*` and `wait.ExponentialBackoff*` functions have been deprecated and will be removed in a future release.  Callers should switch to using `wait.PollUntilContextCancel`, `wait.PollUntilContextTimeout`, or `wait.ExponentialBackoffWithContext` as appropriate.
  
  `PollWithContext(Cancel|Deadline)` will no longer return `ErrWaitTimeout` - use the `Interrupted(error) bool` helper to replace checks for `err == ErrWaitTimeout`, or compare specifically to context errors as needed. A future release will make the `ErrWaitTimeout` error private and callers must use `Interrupted()` instead. If you are returning `ErrWaitTimeout` from your own methods, switch to creating a location specific `cause err` and pass it to the new method `wait.ErrorInterrupted(cause) error` which will ensure `Interrupted()` returns true for your loop. 
  
  The `wait.NewExponentialBackoffManager` and `wait.NewJitteringBackoffManager` functions have been marked as deprecated.  Callers should switch to using the `Backoff{...}.DelayWithReset(clock, resetInterval)` method and must set the `Steps` field when using `Factor`. As a short term change, callers may use the `Timer()` method on the `BackoffManager` until the backoff managers are deprecated and removed. Please see the godoc of the deprecated functions for examples of how to replace usage of this function. ([#107826](https://github.com/kubernetes/kubernetes/pull/107826), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery, Auth, Cloud Provider, Storage and Testing]
- Upgrade coredns to v1.10.1 ([#115603](https://github.com/kubernetes/kubernetes/pull/115603), [@pacoxu](https://github.com/pacoxu)) [SIG Cloud Provider and Cluster Lifecycle]
- [KCCM - service controller]: enable connection draining for terminating pods upon node downscale by the cluster autoscaler. This is done by not reacting to the taint used by the cluster autoscaler to indicate that the node is going away soon, thus keeping the node referenced by the load balancer until the VM has been completely deleted. ([#115204](https://github.com/kubernetes/kubernetes/pull/115204), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG API Machinery, Cloud Provider, Instrumentation and Network]

## Dependencies

### Added
- sigs.k8s.io/kustomize/kustomize/v5: v5.0.1

### Changed
- github.com/aws/aws-sdk-go: [v1.44.147 → v1.35.24](https://github.com/aws/aws-sdk-go/compare/v1.44.147...v1.35.24)
- github.com/coreos/go-systemd/v22: [v22.3.2 → v22.4.0](https://github.com/coreos/go-systemd/v22/compare/v22.3.2...v22.4.0)
- github.com/go-errors/errors: [v1.0.1 → v1.4.2](https://github.com/go-errors/errors/compare/v1.0.1...v1.4.2)
- github.com/golang/protobuf: [v1.5.2 → v1.5.3](https://github.com/golang/protobuf/compare/v1.5.2...v1.5.3)
- github.com/onsi/ginkgo/v2: [v2.7.0 → v2.9.1](https://github.com/onsi/ginkgo/v2/compare/v2.7.0...v2.9.1)
- github.com/onsi/gomega: [v1.26.0 → v1.27.4](https://github.com/onsi/gomega/compare/v1.26.0...v1.27.4)
- golang.org/x/mod: v0.7.0 → v0.9.0
- golang.org/x/net: v0.7.0 → v0.8.0
- golang.org/x/sys: v0.5.0 → v0.6.0
- golang.org/x/term: v0.5.0 → v0.6.0
- golang.org/x/text: v0.7.0 → v0.8.0
- golang.org/x/tools: v0.4.0 → v0.7.0
- k8s.io/kube-openapi: 1cb3ae2 → 15aac26
- sigs.k8s.io/json: f223a00 → bc3834c
- sigs.k8s.io/kustomize/api: v0.12.1 → v0.13.2
- sigs.k8s.io/kustomize/cmd/config: v0.10.9 → v0.11.1
- sigs.k8s.io/kustomize/kyaml: v0.13.9 → v0.14.1

### Removed
- github.com/PuerkitoBio/purell: [v1.1.1](https://github.com/PuerkitoBio/purell/tree/v1.1.1)
- github.com/PuerkitoBio/urlesc: [de5bf2a](https://github.com/PuerkitoBio/urlesc/tree/de5bf2a)
- github.com/mattn/go-runewidth: [v0.0.7](https://github.com/mattn/go-runewidth/tree/v0.0.7)
- github.com/niemeyer/pretty: [a10e7ca](https://github.com/niemeyer/pretty/tree/a10e7ca)
- github.com/olekukonko/tablewriter: [v0.0.4](https://github.com/olekukonko/tablewriter/tree/v0.0.4)
- sigs.k8s.io/kustomize/kustomize/v4: v4.5.7



# v1.27.0-alpha.3


## Downloads for v1.27.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes.tar.gz) | 86bdc8dfcb5ce47ef6f57917a9deed3dbd800669411e494f565bcb0fc6caf2982026d995205cac614e608be2cf240804668fc9f90579bef0872ee5e5ef33f4d8
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-src.tar.gz) | c8395e5693aa148b0b326477b78d1067ff4368f34c755c003938fd88a777ae2303b102d2da240e762cf40cf171cc7a70746a4ddee4c6a35a16bd0eb9265877af

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | 677ee66b49d137335a16ec3acc7fef33bcf1cee094edc0370487ac8060d728ed009c92b3438dbeb675c113320991ea9bf703579dbaf929883a82874222534760
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | d1dbdc0d3f9ad5772aa33276a097f0848f445072c4a3cad1902c3fa015952aa0b78656d144a792beccbfe1bf7b78c0257b4507edcda00fb960fd2249b2cab5fd
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 8f85dfe2f157921b6b3250c744549c2ed4ccd491279b16eb0eb45402ad8f72e73a368eeaa46a4cc309fc5079c333b6f810d5904f87bb282b07b1d0abb0e22586
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | a058c582c63da65fd9a4b95ea80e67367d0a98e5181c8733f6fe1023c356a85c87594c1ade300eb00f3ef3ef6a32f13c44eb8394f862f512217b4d437b6dbb63
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | 403e8083abc8ee509e636d0b58a013b35c7b36f10fb5e75cccbefbd6bc9a2760dfbd1ad40845b7dc0f5d78d6c93d4a6527e119a4e5f74f5dfa71281bda6a64bb
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | a4defcf0bb8684cfe49352016abf1c720c36cc2cec8a599f987b164669ce67f581b23134a4194f4a2fb58ac489fd5b7c99ad5539efffa5a7ac5aaa2a31c79f65
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | fdf33f56239537311b1839f22de6db9a60f515bdcfadd5e470f5c459ab1a1d1fcdd67e27abdd4d6fcc5b356a8188794e7605d9633dd9364a4cccc01bce027357
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 8475beee121129cb7bf68763ba2cba816f9ad0daec055b564074d9364674b254dc5b368f5acf648ae32b00f0f925b7d8c0403c9ca15754fabf635c272e26a64a
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 7a03a2918722fb94e4a2cee827f6abf13184479274e9c246573cc5515bd9eb4cae46c62988bf9072ae0572086d4689986b78ee5349fcefa60a62e20f17c28d45
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 52e8d2da2ec5e3f51f79a2f946c49f57ab0c22321d721914a91656100a721f7278baf419d6aa629c9e8247be60177cc8cb077499b692159a6496506b56e8e17a
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | efa2d76d57b6e3d9eb897a6e8d9812d67d4d57b9368754e22efeaa4a031d34e1ba4e8db0272b5f2ae9ee6d5615ab078201bc35fa4762f1e32c6b67c8fb7b6c8d

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | a1602f5be2abcbc67c763533c7352ac067fe9f2e778b3b2eeb72ee39c885e299edb53f01889420199111cd53465b463290097b8a4f8879c91b912ffbf1aa7dde
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | e129be174e855f48a5de6cf07731c9897edb360c9df1f842915323a8abf2bf704be1c275df04c93a878edbb51ea3316eeb37779ba3ebf3dc19887c97c4b5bc23
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | f890256c2e5b4096ee944c095df2ff5fd38056ed09deabcbe7501f6b9b5665267e775d1db081f94735dba4e9423f0bebca4b275a2b38389119513bbe2d2b53c2
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | 878ebfa012b184c93505f62e30ada071420bd199ef22d4690c28dff6cbc48de6048978d41d7a6f518d26564261c9da4b7fefce8f0da1e5d274ac871696bb7b93
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 217bcec3012f3aefa612178e9842779825cf8d50dc7e1d6a84239206f70976d939c1089d177851f3657a1f7f763ebe6a31b3ff3d99c1120105036cf37650900c

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | c15b9a152ceed976aff0b64c2d601985db4e58785b1a756fac6e4c83be27c70e96a277692448d75fccc69b82a5d2a872fa2ab3b5fe4b4145eacbab3d0fa03086
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | 6c0f7d3b97532aacd7aaba931bf4fb858cf49559bf51ea0e154efc4ee26de8d01f7f39f017462e136afe5faaa8d86ee49e954c3a9051751ce8dfa8312cd95f06
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | fb2934282018539abb5ba0fde67225d5131d08c5798ed54dc93081739a803210c9aaf73412c56af7aa9999b497aac073be42ec3a7677255e5d4e39783330bd86
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 84078c93f7661c5aedf276964cc5866e1eba9b8cbc9e1c40b9e5bfe76ad115a632f4f591b177dc5fa926cb65a66c6847fb243b82ad1f880796d1d7cfe061b498
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | dff5a35a352e5a17abad8eaa51b9447ca0a8a52df4b22e325ac7acf70efe3d0fe9bef958e8bb5a1e90d43c4586551d456e2dcf42cd01074605c07d69a52fc5b7
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | a2da87f8a7bd25d6efc8c3b7af79e268809e743ce36c187cdc0af3ff423d54f95a5bd03b50494aea8b0c394ca716e5c4225e7e2d49f40f08db6cd302d894eb3b

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.0-alpha.2

## Changes by Kind

### Deprecation

- Added a [warning](https://k8s.io/blog/2020/09/03/warnings/) response when handling requests that set the deprecated `spec.externalID` field for a Node. ([#115944](https://github.com/kubernetes/kubernetes/pull/115944), [@SataQiu](https://github.com/SataQiu)) [SIG Node]

### API Change

- Graduated seccomp profile defaulting to GA.
  
  Set the kubelet `--seccomp-default` flag or `seccompDefault` kubelet configuration field to `true` to make pods on that node default to using the `RuntimeDefault` seccomp profile.
  
  Enabling seccomp for your workload can have a negative performance impact depending on the kernel and container runtime version in use.
  
  Guidance for identifying and mitigating those issues is outlined in the Kubernetes [seccomp tutorial](https://k8s.io/docs/tutorials/security/seccomp). ([#115719](https://github.com/kubernetes/kubernetes/pull/115719), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery, Node, Storage and Testing]
- Implements API for streaming for the watch-cache
  
  When sendInitialEvents ListOption is set together with watch=true, it begins the watch stream with synthetic init events followed by a synthetic "Bookmark" after which the server continues streaming events. ([#110960](https://github.com/kubernetes/kubernetes/pull/110960), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Introduce API for streaming.
  
  Add SendInitialEvents field to the ListOptions. When the new option is set together with watch=true, it begins the watch stream with synthetic init events followed by a synthetic "Bookmark" after which the server continues streaming events. ([#115402](https://github.com/kubernetes/kubernetes/pull/115402), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kubelet: a "maxParallelImagePulls" field can now be specified in the kubelet configuration file to control how many image pulls the kubelet can perform in parallel. ([#115220](https://github.com/kubernetes/kubernetes/pull/115220), [@ruiwen-zhao](https://github.com/ruiwen-zhao)) [SIG API Machinery, Node and Scalability]
- PodSchedulingReadiness is graduated to beta. ([#115815](https://github.com/kubernetes/kubernetes/pull/115815), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG API Machinery, Apps, Scheduling and Testing]
- In-place resize feature for Kubernetes Pods
  - Changed the Pod API so that the `resources` defined for containers are mutable for `cpu` and `memory` resource types.
  - Added `resizePolicy` for containers in a pod to allow users control over how their containers are resized.
  - Added `allocatedResources` field to container status in pod status that describes the node resources allocated to a pod.
  - Added `resources` field to container status that reports actual resources applied to running containers.
  - Added `resize` field to pod status that describes the state of a requested pod resize.

  For details, see KEPs below. ([#102884](https://github.com/kubernetes/kubernetes/pull/102884), [@vinaykul](https://github.com/vinaykul)) [SIG API Machinery, Apps, Instrumentation, Node, Scheduling and Testing]
- The PodDisruptionBudget `spec.unhealthyPodEvictionPolicy` field has graduated to beta and is enabled by default. On servers with the feature enabled, this field may be set to `AlwaysAllow` to always allow unhealthy pods covered by the PodDisruptionBudget to be evicted. ([#115363](https://github.com/kubernetes/kubernetes/pull/115363), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps, Auth and Node]
- The `DownwardAPIHugePages` kubelet feature graduated to stable / GA. ([#115721](https://github.com/kubernetes/kubernetes/pull/115721), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps and Node]
- Volumes: `resource.claims` gets cleared for PVC specs during create or update of a pod spec with inline PVC template or of a PVC because it has no effect. ([#115928](https://github.com/kubernetes/kubernetes/pull/115928), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps and Storage]

### Feature

- API validation relaxed allowing Indexed Jobs to be scaled up/down by changing parallelism and completions in tandem, such that parallelism == completions. ([#115236](https://github.com/kubernetes/kubernetes/pull/115236), [@danielvegamyhre](https://github.com/danielvegamyhre)) [SIG Apps and Testing]
- Add kubelet Topology Manager metric to measure topology manager admission latency. ([#115590](https://github.com/kubernetes/kubernetes/pull/115590), [@swatisehgal](https://github.com/swatisehgal)) [SIG Node and Testing]
- Added "netadmin" debugging profiles for kubectl debug. ([#115712](https://github.com/kubernetes/kubernetes/pull/115712), [@wedaly](https://github.com/wedaly)) [SIG CLI]
- Added apiserver_envelope_encryption_invalid_key_id_from_status_total to measure number of times an invalid keyID is returned by the Status RPC call. ([#115846](https://github.com/kubernetes/kubernetes/pull/115846), [@ritazh](https://github.com/ritazh)) [SIG API Machinery and Auth]
- Apiserver_storage_transformation_operations_total metric has been updated to include labels transformer_prefix and status. ([#115394](https://github.com/kubernetes/kubernetes/pull/115394), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth, Instrumentation and Testing]
- Client-go: metadatainformer and dynamicinformer `SharedInformerFactory`s now supports waiting for goroutines during shutdown ([#114434](https://github.com/kubernetes/kubernetes/pull/114434), [@howardjohn](https://github.com/howardjohn)) [SIG API Machinery]
- Graduate the `ReadWriteOncePod` feature gate to beta ([#114494](https://github.com/kubernetes/kubernetes/pull/114494), [@chrishenzie](https://github.com/chrishenzie)) [SIG Scheduling, Storage and Testing]
- Kubeadm: show a warning message when detecting that the sandbox image of the container runtime is inconsistent with that used by kubeadm ([#115610](https://github.com/kubernetes/kubernetes/pull/115610), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubernetes is now built with go 1.20.1 ([#115828](https://github.com/kubernetes/kubernetes/pull/115828), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Performance improvements in klog ([#115277](https://github.com/kubernetes/kubernetes/pull/115277), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Storage and Testing]
- Pod template `schedulingGates` are now mutable for Jobs that are suspended and have never been started ([#115940](https://github.com/kubernetes/kubernetes/pull/115940), [@ahg-g](https://github.com/ahg-g)) [SIG Apps]
- Pods which have an invalid negative `spec.terminationGracePeriodSeconds` value will be treated as having terminationGracePeriodSeconds of 1 ([#115606](https://github.com/kubernetes/kubernetes/pull/115606), [@wzshiming](https://github.com/wzshiming)) [SIG Apps, Node and Testing]
- The Pod API field `.spec.schedulingGates[*].name` now requires qualified names (like `example.com/mygate`), matching validation for names of `.spec.readinessGates[*].name`. Any uses of the alpha scheduling gate feature prior to 1.27 that do not match that validation must be renamed or deleted before upgrading to 1.27. ([#115821](https://github.com/kubernetes/kubernetes/pull/115821), [@lianghao208](https://github.com/lianghao208)) [SIG Apps and Scheduling]
- The `JobMutableNodeSchedulingDirectives` feature gate has graduated to GA. ([#116116](https://github.com/kubernetes/kubernetes/pull/116116), [@ahg-g](https://github.com/ahg-g)) [SIG Apps, Scheduling and Testing]
- Two changes to the /debug/api_priority_and_fairness/dump_priority_levels endpoint of API Priority and Fairness: add total number of dispatched, timed-out, rejected and cancelled requests; output now sorted by PriorityLevelName. ([#112393](https://github.com/kubernetes/kubernetes/pull/112393), [@borgerli](https://github.com/borgerli)) [SIG API Machinery]
- Updated distroless iptables to use released image `registry.k8s.io/distroless-iptables:v0.2.1` ([#115905](https://github.com/kubernetes/kubernetes/pull/115905), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- [E2E] Pods spawned by E2E tests can now pull images from the private registry using the new --e2e-docker-config-file flag ([#114625](https://github.com/kubernetes/kubernetes/pull/114625), [@Divya063](https://github.com/Divya063)) [SIG Node and Testing]

### Documentation

- Document the reason field in CRI API to ensure it equals OOMKilled for the containers terminated by OOM killer ([#112977](https://github.com/kubernetes/kubernetes/pull/112977), [@mimowo](https://github.com/mimowo)) [SIG Node]

### Failing Test

- Fixed panic in vSphere e2e tests. ([#115863](https://github.com/kubernetes/kubernetes/pull/115863), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]

### Bug or Regression

- Cacher: If RV is unset, the watch is now served from the underlying storage as documented. ([#115096](https://github.com/kubernetes/kubernetes/pull/115096), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG API Machinery]
- Client-go: fix the wait time for trying to acquire the leader lease ([#114872](https://github.com/kubernetes/kubernetes/pull/114872), [@Iceber](https://github.com/Iceber)) [SIG API Machinery]
- File content check for IPV4 is not enabled by default, and the check of IPV4 or IPV6 is done for `kubeadm init` or `kubeadm join` only in case the user intends to create a cluster to support that kind of IP address family ([#115420](https://github.com/kubernetes/kubernetes/pull/115420), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle and Network]
- Fix log line in scheduler that inaccurately implies that volume binding has finalized ([#116018](https://github.com/kubernetes/kubernetes/pull/116018), [@TommyStarK](https://github.com/TommyStarK)) [SIG Scheduling and Storage]
- Fix missing delete events on informer re-lists to ensure all delete events are correctly emitted and using the latest known object state, so that all event handlers and stores always reflect the actual apiserver state as best as possible ([#115620](https://github.com/kubernetes/kubernetes/pull/115620), [@odinuge](https://github.com/odinuge)) [SIG API Machinery]
- Fixed a bug where Kubernetes would apply a default StorageClass to a PersistentVolumeClaim,
  even when the deprecated annotation `volume.beta.kubernetes.io/storage-class` was set. ([#116089](https://github.com/kubernetes/kubernetes/pull/116089), [@cvvz](https://github.com/cvvz)) [SIG Apps and Storage]
- Fixed an EndpointSlice Controller hashing bug that could cause EndpointSlices to incorrectly handle Pods with duplicate IP addresses. For example this could happen when a new Pod reused an IP that was also assigned to a Pod in a completed state. ([#115907](https://github.com/kubernetes/kubernetes/pull/115907), [@qinqon](https://github.com/qinqon)) [SIG Apps and Network]
- Fixing issue with Winkernel Proxier - ClusterIP Loadbalancers are missing if the ExternalTrafficPolicy is set to Local and the available endpoints are all remoteEndpoints. ([#115919](https://github.com/kubernetes/kubernetes/pull/115919), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Golang.org/x/net updates to v0.7.0 to fix CVE-2022-41723 ([#115786](https://github.com/kubernetes/kubernetes/pull/115786), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- Kubeadm: fix a bug where the uploaded kubelet configuration in `kube-system/kubelet-config` ConfigMap does not respect user patch ([#115575](https://github.com/kubernetes/kubernetes/pull/115575), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: modify '--config' flag from required to optional for 'kubeadm kubeconfig user' command ([#116074](https://github.com/kubernetes/kubernetes/pull/116074), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Yes, discovery document will correctly return the resources for aggregated apiservers that do not implement aggregated disovery ([#115770](https://github.com/kubernetes/kubernetes/pull/115770), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Improved FormatMap: Improves performance by about 4x, or nearly 2x in the worst case ([#112661](https://github.com/kubernetes/kubernetes/pull/112661), [@aimuz](https://github.com/aimuz)) [SIG Node]
- Upgrade go-jose to v2.6.0 ([#115893](https://github.com/kubernetes/kubernetes/pull/115893), [@mgoltzsche](https://github.com/mgoltzsche)) [SIG API Machinery, Auth, Cluster Lifecycle and Testing]
- `apiserver_admission_webhook_admission_duration_seconds` buckets have been expanded, 25s is now the largest bucket size to match the webhook default timeout. ([#115802](https://github.com/kubernetes/kubernetes/pull/115802), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Instrumentation]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/coredns/corefile-migration: [v1.0.18 → v1.0.20](https://github.com/coredns/corefile-migration/compare/v1.0.18...v1.0.20)
- github.com/golang-jwt/jwt/v4: [v4.2.0 → v4.4.2](https://github.com/golang-jwt/jwt/v4/compare/v4.2.0...v4.4.2)
- go.etcd.io/etcd/api/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/client/pkg/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/client/v2: v2.305.5 → v2.305.7
- go.etcd.io/etcd/client/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/pkg/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/raft/v3: v3.5.5 → v3.5.7
- go.etcd.io/etcd/server/v3: v3.5.5 → v3.5.7
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.35.0 → v0.35.1
- golang.org/x/net: v0.5.0 → v0.7.0
- golang.org/x/sys: v0.4.0 → v0.5.0
- golang.org/x/term: v0.4.0 → v0.5.0
- golang.org/x/text: v0.6.0 → v0.7.0
- gopkg.in/square/go-jose.v2: v2.2.2 → v2.6.0
- k8s.io/klog/v2: v2.80.1 → v2.90.1

### Removed
- github.com/form3tech-oss/jwt-go: [v3.2.3+incompatible](https://github.com/form3tech-oss/jwt-go/tree/v3.2.3)



# v1.27.0-alpha.2


## Downloads for v1.27.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes.tar.gz) | 5420d881db6412c1c1e55044aea61f310ef42d7809d4d90113b2a80ae0d1446f3e7988a8205100c476a313182a0c8b2d1605ad3000eee3b45fec4034d17f2ac2
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-src.tar.gz) | 3b9693bd03ed7f5aee3257a167e431b9de4c576f8843f1441f81cbcdfc6be607c84ca703bd2e7ca4bb5f3b9dee9fbb8645cdf49c1921d796e1a3f027c8f23162

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | ce9875156a7c80452dc3303177b0a137cfc6ae398b66a32b1436768ab77771b000287dba0702510da239c056a697e624416f6126a6205c3c65e78ff6d7d4635b
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | bc0791295f926f285f18163bef7faf893162918d75e8de0aa46704d2ac665bbff641a7332d5a3d112d93dd5e14087f8e7333e39b4cc44ab71330e059b0abe4bb
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 76abdb1dbb8886c554628ba634449c42f0c61eb47e168d1cb7bb1eabb0354b37474738879955805165e52a4cbbe39c57cabe63a2b8dbbaed00e14a0da5a7419f
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 5ced29d3f8411f34ba9dbd115aab7e45d542b34f7feebba6bfe8dcd394abd9fe127daa6c36460c2d8b35ca6386729a3e644b23b5631fe4d81ae3ae0cf1297e67
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | fff2ce7f24f9fa6c5240d69d81a0c363742b17963265ce744e3366d05d149bce005cfe96fb1dd20b7c5faceed481225da0715dee8b2743ee3ff21391c742a1a0
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 54da883e07f1a6e6bb9ca29ca4b5bedb2d24485cd07c8ba03da90b063a07e01271d0ad3b58d20fc3370a40486134b7b6144ad2d18049d7e3a38600ad14d84f8f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 50be6728b20612ea3d422e3346150c4cece1cd42356446cf8fd2f9164a40ac997188d840536dc45deab5acf12143233d36b76d9ef12165bb0884024f1725f28c
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | fa76f8655266fb9b64c450185c175f6854ac1d569140f6d62c383d829111091b79a2f266929ca10f642a989e9ada066988845666621fe13c75cbcfa971f5aa0d
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 67c36d790cd5de91e0241cee3800fcdb49db3f3a9e91e087937686367149d7b06c489e62919ef6fdcb8ec29974ead4a64bab0eb3278f404188cf8fffe89baba9
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 49e73490b58576237627cc8015f84eda36aa3af02b8e80b251b07d294fe161e90815ee4149f3b8605fad7c43b278f7b0f631ae3d51fa344ad326abcd480d781d
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 27ac0573663d5e45585b205c84cb0e5a7f16282654e30445ad4115a148c20b548d0c3520e45327a4443768c83b68c96e8754cdfb34f17fa0ebf875e4eec2eb48

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | 0a2fa2de60af23f722a27479ee0721551561a6bf947ec66c9548b0d14410745de2db3f69c6536768768ffeec4f6afe3af3bd336aeccef67391c4cdaca4a427a6
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | c5ab1da7a7e19acebdba7107c27954e522d33c245ea04556347b601e2cc0f40595b9ca5159661b134a090d5505a76d967da3161b3a409b2a4d9c0f36e1b4d7b1
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 3a5cad9ae0f4a1086a238e2fb44f59733361d9dea206390c73825daf25dc8b333fce166f5c5e6c0e1ca3be80b303afb1b6b6c8e9dc13666446c2a70b5b7bc1cb
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 411be709cde53aa27bca30e7d5ab7523f4dea192c85a1aa810985b23a41cdd00c6969e9b9614a193618be94d346900c2f8e9211c95927a12398142268db4ce5e
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 0ac65f78a5cad3506649ca40912f328b2af747ae6367f0ca16ba741af20aeecd73c27ea216921ab7a28da8a61b58f0760e9211c65953e70304fec4b940a39440

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 3a0559b2305136a15cd43104ce728f7651b4fcde13db69f565d66e117ad7f8f30a017d3ea6be92811e4ab880273033c089688675559912bfb6d2aa2c92d60225
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | ac7597cfab9eb93dd9c0f1cd088dd08d120991bc94a718fa89ddd9b8fa12a9f6b9987eaaee66b8aafbb055c836c289ca7ca415b57f61bd8f9159045025026100
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 395d65f26b4f482cd1d8be49b846ce80f536ca825ae8ce25d10fe746d95e4297c31512247d22caefe632d2236a33616e2650ed385811ced24c3e6338a5eda36d
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 7f92e5fdfba981ac80b71fdc00e84b4eb661604861f5602e5fa489f13a10ac699e6e5795cc3654ff95e1b5f7fd51df7773bd5ae511ace9b861a87b6fb1465cc7
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 75ec78e900a4df4819c899893fe98fe32b6fa8ae000318dcfed8972d356cc1c5e0a3875885681375c080b0770c377164c03c100a6c45b7d025363e174a00af00
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | d7630730d547414bdb2b245e1b444a5949cecead751d6c243db72e8f20782ba85e1c06dfab499c13e1b529a74a5d4acef4a9b1a6ca571faf41d3253b1bf74773

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.27.0-alpha.1

## Changes by Kind

### API Change

- A fix in the resource.k8s.io/v1alpha1/ResourceClaim API avoids harmless (?) ".status.reservedFor: element 0: associative list without keys has an element that's a map type" errors in the apiserver. Validation now rejects the incorrect reuse of the same UID in different entries. ([#115354](https://github.com/kubernetes/kubernetes/pull/115354), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- CacheSize field in EncryptionConfiguration is not supported for KMSv2 provider ([#113121](https://github.com/kubernetes/kubernetes/pull/113121), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- K8s.io/client-go/tools/record.EventBroadcaster: after Shutdown() is called, the broadcaster now gives up immediately after a failure to write an event to a sink. Previously it tried multiple times for 12 seconds in a goroutine. ([#115514](https://github.com/kubernetes/kubernetes/pull/115514), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- K8s.io/component-base/logs now also supports adding command line flags to a flag.FlagSet. ([#114731](https://github.com/kubernetes/kubernetes/pull/114731), [@pohly](https://github.com/pohly)) [SIG Architecture]
- Update API reference for Requests, specifying they must not exceed limits ([#115434](https://github.com/kubernetes/kubernetes/pull/115434), [@ehashman](https://github.com/ehashman)) [SIG Architecture, Docs and Node]
- `/metrics/slis` is made available for control plane components allowing you to scrape health check metrics. ([#114997](https://github.com/kubernetes/kubernetes/pull/114997), [@Richabanker](https://github.com/Richabanker)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]

### Feature

- A new client side metric `rest_client_request_retries_total` has been added that tracks 
  the number of retries sent to the server, partitioned by status code, verb, and host ([#108396](https://github.com/kubernetes/kubernetes/pull/108396), [@tkashem](https://github.com/tkashem)) [SIG API Machinery, Architecture and Instrumentation]
- A new feature has been enabled to improve the performance of the iptables mode of kube-proxy in large clusters. You do not need to take any action, however:
  
  1. If you experience problems with Services not syncing to iptables correctly, you can disable the feature by passing `--feature-gates=MinimizeIPTablesRestore=false` to kube-proxy (and file a bug if this fixes it). (This might also be detected by seeing the value of kube-proxy's `sync_proxy_rules_iptables_partial_restore_failures_total` metric rising.)
  2. If you were previously overriding the kube-proxy configuration for performance reasons, this may no longer be necessary. See https://kubernetes.io/docs/reference/networking/virtual-ips/#optimizing-iptables-mode-performance. ([#115138](https://github.com/kubernetes/kubernetes/pull/115138), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Add kubelet Topology Manager metrics to track admission requests processed by it and occured admission errors. ([#115137](https://github.com/kubernetes/kubernetes/pull/115137), [@swatisehgal](https://github.com/swatisehgal)) [SIG Node and Testing]
- Add logging-format option to CCMs based on k8s.io/cloud-provider ([#108984](https://github.com/kubernetes/kubernetes/pull/108984), [@LittleFox94](https://github.com/LittleFox94)) [SIG Cloud Provider and Instrumentation]
- Add new -f flag into debug command to be used passing pod or node files instead explicit names. ([#111453](https://github.com/kubernetes/kubernetes/pull/111453), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Added "general", "baseline", and "restricted" debugging profiles for kubectl debug. ([#114280](https://github.com/kubernetes/kubernetes/pull/114280), [@sding3](https://github.com/sding3)) [SIG CLI]
- Added apiserver_envelope_encryption_kms_operations_latency_seconds metric to measure the KMSv2 grpc calls latency. ([#115649](https://github.com/kubernetes/kubernetes/pull/115649), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Adds scheduler preemption support for pods using `ReadWriteOncePod` PVCs ([#114051](https://github.com/kubernetes/kubernetes/pull/114051), [@chrishenzie](https://github.com/chrishenzie)) [SIG Scheduling, Storage and Testing]
- Adds the applyconfiguration generator to the code-generator script that generates server-side apply configuration and client APIs ([#114987](https://github.com/kubernetes/kubernetes/pull/114987), [@astefanutti](https://github.com/astefanutti)) [SIG API Machinery]
- Dynamic Resource Allocation framework can be used for network devices ([#114364](https://github.com/kubernetes/kubernetes/pull/114364), [@bart0sh](https://github.com/bart0sh)) [SIG Node]
- Fixed bug which caused the status of Indexed Jobs to only be updated when there are newly completed indexes. The completed indexes are now updated if the .status.completedIndexes has values outside of the [0, .spec.completions> range ([#115349](https://github.com/kubernetes/kubernetes/pull/115349), [@danielvegamyhre](https://github.com/danielvegamyhre)) [SIG Apps]
- GRPC probes now set a linger option of 1s to improve the TIME-WAIT state. ([#115321](https://github.com/kubernetes/kubernetes/pull/115321), [@rphillips](https://github.com/rphillips)) [SIG Network and Node]
- Kubelet config file will be backed up to `/etc/kubernetes/tmp/` folder with `kubeadm-kubelet-config` append with a random suffix as the filename ([#114695](https://github.com/kubernetes/kubernetes/pull/114695), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubelet no longer creates certain legacy iptables rules by default.
  It is possible that this will cause problems with some third-party components
  that improperly depended on those rules. If this affects you, you can run
  kubelet with `--feature-gates=IPTablesOwnershipCleanup=false`, but you should
  also file a bug against the third-party component. ([#114472](https://github.com/kubernetes/kubernetes/pull/114472), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kubernetes is now built with go 1.20 ([#114502](https://github.com/kubernetes/kubernetes/pull/114502), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Migrated the ResourceQuota controller (within `kube-controller-manager`) to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#113315](https://github.com/kubernetes/kubernetes/pull/113315), [@ncdc](https://github.com/ncdc)) [SIG API Machinery, Apps and Testing]
- New feature gate, ServiceNodePortStaticSubrange, to enable the new strategy in the NodePort Service port allocators, so the node port range is subdivided and dynamic allocated NodePort port for Services are allocated preferentially from the upper range. ([#114418](https://github.com/kubernetes/kubernetes/pull/114418), [@xuzhenglun](https://github.com/xuzhenglun)) [SIG Network]
- Scheduler doesn't run plugin's Score method when its PreScore method returned a Skip status. In other words, your PreScore/Score plugin can return a Skip status in PreScore if the plugin does nothing in Score for that Pod. ([#115652](https://github.com/kubernetes/kubernetes/pull/115652), [@kidddddddddddddddddddddd](https://github.com/kidddddddddddddddddddddd)) [SIG Scheduling]
- The go version defined in `.go-version` is now fetched when invoking test, build, and code generation targets if the current go version does not match it. Set $FORCE_HOST_GO=y while testing or building to skip this behavior, or set $GO_VERSION to override the selected go version. ([#115377](https://github.com/kubernetes/kubernetes/pull/115377), [@liggitt](https://github.com/liggitt)) [SIG Testing]
- The mount-utils mounter now provides an option to limit the number of concurrent format operations. ([#115379](https://github.com/kubernetes/kubernetes/pull/115379), [@artemvmin](https://github.com/artemvmin)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node and Storage]

### Bug or Regression

- Apply configurations can be generated for types with non-builtin map fields ([#114920](https://github.com/kubernetes/kubernetes/pull/114920), [@astefanutti](https://github.com/astefanutti)) [SIG API Machinery]
- Enforce nodeName cannot be set along with non-empty schedulingGates ([#115569](https://github.com/kubernetes/kubernetes/pull/115569), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Apps and Scheduling]
- Etcd: Update to v3.5.7 ([#115310](https://github.com/kubernetes/kubernetes/pull/115310), [@mzaian](https://github.com/mzaian)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Fix a bug that caused to panic the apiserver when trying to allocate a Service with a dynamic ClusterIP and it has been configured with Service CIDRs with a /28 mask for IPv4 and a /124 mask for IPv6 ([#115322](https://github.com/kubernetes/kubernetes/pull/115322), [@aojea](https://github.com/aojea)) [SIG Testing]
- Fix an issue where a CSI migrated volume may be prematurely detached when the CSI driver is not running on the node.
  If CSI migration is enabled on the node, even the csi-driver is not up and ready, we will still add this volume to DSW. ([#115464](https://github.com/kubernetes/kubernetes/pull/115464), [@sunnylovestiramisu](https://github.com/sunnylovestiramisu)) [SIG Apps and Storage]
- Fix nil pointer error in nodevolumelimits csi logging ([#115179](https://github.com/kubernetes/kubernetes/pull/115179), [@sunnylovestiramisu](https://github.com/sunnylovestiramisu)) [SIG Scheduling]
- Fix the regression that introduced 34s timeout for DELETECOLLECTION calls ([#115341](https://github.com/kubernetes/kubernetes/pull/115341), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fixing issue with Winkernel Proxier - IPV6 load balancer policies are missing when service is configured with ipFamilyPolicy: RequireDualStack ([#115503](https://github.com/kubernetes/kubernetes/pull/115503), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fixing issue with Winkernel Proxier - IPV6 load balancer policies are missing when service is configured with ipFamilyPolicy: RequireDualStack ([#115577](https://github.com/kubernetes/kubernetes/pull/115577), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Flag `workerCount` has been added to cloud node controller which defines how many workers will be synchronizing nodes. ([#113104](https://github.com/kubernetes/kubernetes/pull/113104), [@pawbana](https://github.com/pawbana)) [SIG API Machinery, Cloud Provider and Scalability]
- Kube-apiserver: errors decoding objects in etcd are now recorded in an `apiserver_storage_decode_errors_total` counter metric ([#114376](https://github.com/kubernetes/kubernetes/pull/114376), [@baomingwang](https://github.com/baomingwang)) [SIG API Machinery and Instrumentation]
- Kube-apiserver: regular expressions specified with the `--cors-allowed-origins` option are now validated to match the entire `hostname` inside the `Origin` header of the request and 
  must contain '^' or the '//' prefix to anchor to the start, and '$' or the port separator ':' to anchor to 
  the end. ([#112809](https://github.com/kubernetes/kubernetes/pull/112809), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Kubeadm: fix an etcd learner-mode bug by preparing an etcd static pod manifest before promoting ([#115038](https://github.com/kubernetes/kubernetes/pull/115038), [@tobiasgiese](https://github.com/tobiasgiese)) [SIG Cluster Lifecycle]
- Kubelet: fix a bug of stoping rendering configmap when enabling fsquota monitoring ([#112624](https://github.com/kubernetes/kubernetes/pull/112624), [@pacoxu](https://github.com/pacoxu)) [SIG Node and Storage]
- Set device stage path whenever available for expansion during mount ([#115346](https://github.com/kubernetes/kubernetes/pull/115346), [@gnufied](https://github.com/gnufied)) [SIG Storage and Testing]
- The Kubernetes API server now correctly detects and closes existing TLS connections when its client certificate file for kubelet authentication has been rotated. ([#115315](https://github.com/kubernetes/kubernetes/pull/115315), [@enj](https://github.com/enj)) [SIG API Machinery, Auth, Node and Testing]

### Other (Cleanup or Flake)

- Changes docs for --contention-profiling flag to reflect it performs block profiling ([#114490](https://github.com/kubernetes/kubernetes/pull/114490), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG API Machinery, Cloud Provider, Docs, Node and Scheduling]
- E2e framework: added `--report-complete-ginkgo` and `--report-complete-junit` parameters. They work like `ginkgo --json-report <report dir>/ginkgo/report.json --junit-report <report dir>/ginkgo/report.xml`. ([#115678](https://github.com/kubernetes/kubernetes/pull/115678), [@pohly](https://github.com/pohly)) [SIG Testing]
- Promote pod resource limit/request metrics to stable. ([#115454](https://github.com/kubernetes/kubernetes/pull/115454), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Instrumentation and Scheduling]
- The `ControllerManagerLeaderMigration ` feature, GA since 1.24, is unconditionally enabled and the feature gate option has been removed. ([#113534](https://github.com/kubernetes/kubernetes/pull/113534), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery and Cloud Provider]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/onsi/gomega: [v1.24.2 → v1.26.0](https://github.com/onsi/gomega/compare/v1.24.2...v1.26.0)
- go.uber.org/goleak: v1.2.0 → v1.2.1
- golang.org/x/net: v0.4.0 → v0.5.0
- golang.org/x/sys: v0.3.0 → v0.4.0
- golang.org/x/term: v0.3.0 → v0.4.0
- golang.org/x/text: v0.5.0 → v0.6.0
- k8s.io/kube-openapi: 3758b55 → 1cb3ae2
- k8s.io/utils: 1a15be2 → a36077c

### Removed
_Nothing has changed._



# v1.27.0-alpha.1


## Downloads for v1.27.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes.tar.gz) | 36ddbb7f1cdf386cd6857d891029b7244dd13aa346a78ba2fa2146b866751802989fc5c5c8a8675f80b72b12816ae94b2b00fdeb01421ef15aad8ae87c06c512
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-src.tar.gz) | d5be5bdf5734c89338b2fd52942b70f8d57d831659235a0ea91c6fb10d74637c2a2aeed602bdafb4da01071764f754ddec9d37abf8aeb8eaa11636e405e93b04

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | d9b5f7ec09b64a6e0d270c077fdcd4835b6e16f39373fbf1a2e2526f80aa4df25f41a7e1dab1c33c4ec3f3c430b1cd42b08944d4c39d8ac78c4b23db83d6411b
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | e1a0f33d0610dbac940db0cad930bf4530f90a9c1702e534bb8c5524077af82d6da2f63befd3ba331ddfe84710ece101b9dd93f308ef727605646cc0d0847292
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 1e49e5f94a3bd14c6c5681ea8ad003948f7824566dc4d8ff299aed0a3e650a2b0674ade00fc951131fe34c210b8b7832c1a1fd5c290ee8c9e42bac89efdc8f26
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | ddb000a6a1604a5cb95bbe296366eebd9e0b9b4be250b0d22302c697294840c216641f287dc7212b49f9f121549172590a1f1e285b3d87cee32bcbd95a7d19b1
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 40c242c7bae26da4948fe97daf256dc48d99edc91a535ff9f4516e214ff50cfbbbc985be8e4361a97bfd54dc8b406e6f4a68b1054396b01a157d54a6bd82c7e3
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 54954248f4aa1d2977fe92a552cf7e0298c94adac8bf0720c697cfce654a1fbdf01cd56219e0dd064bab7b07a4cd125c257b61cd4d69af03ec550aec31cb26ac
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | d58709ea83b2a82ea44ae402a574ddc97177494bac90b9c6de30caf3fbcc79697addd0bb8245a62bbaf2fd35483415293659703cbed573bbe11a86a57814f8a3
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 08ab82af97bb2ccf0b90d5de23960ce2634a42eaafcc50725ce345e3df61e082acc2e733c3aae159463645cbad28a8536371850899403eaeb7a040ed221ba861
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 047fdddf4c1095240eb1296a1d79e9dbb90325d13e9ed94b3532ecfa843dbd089b4b5aa3d6a51e4265c1ecdb2f08b73cd05f309b8bf4c585b213fa605a0bc74d
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 1bd72cfd091f452b4d32496a1f1b0c78231ae6cf9696ca029a340761aaeba08e27003d9fcb5942a63b2feddb2562b287b0d8c49fb0d92f2e50b65a947591b105
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 07daf9e21a7741908e358f0a9ae30164229344c1653c435fb1c6029932431e5bb985842ebcc97330a6ca5aff57584488a17fe01b234de6286f6753faaa4d31f3

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | d426504774023346a994ca709ae48daf47baf0e2f680a2237cc1b8b5b2ba7d4a41be9d0f80af4f1c097348b01a3e1a968b3bbea5cb5937d17e0ef23db6a36b27
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | d4b4eaa2012b9ce5f3fa111cd1a61ef98aec238905f134636470bab2661d4c14464dfc8223bd74456a8848e3c7fe879d14e1eba6c4f2376f9042893077fcb068
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | bc9b1388a93c65f6b603271e8a1a1d622698fd54ce26d3743e888efd5f61cd538229161006a2ab5afcbda8782f87646c9da02a5e7cc93ff52e48616a7d56cb33
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 69972b9c95b50b7566660b44547d974133e6f2f18737a93c2f4a6ca5833400473c0917c0c44b1a27336990214c1f188c75e39701d91c567b8ff7d8ccd3496243
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 1867d483df955260b9cf13bfe43451c2f906c1854399a396e0d9e2fac33fc4343d3c79039891c1b943cf4707312cedcffb4d0cedf7a69271020f88c50d230d5d

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 132c74737e8fd525c8a4bcedbf7ad783917a42a485e740586933dceef76177c6043682654f76f88670933fea0a3066adfcfdd08a9b6e07b3a6ad23e903c1b4a5
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | 7d7716457ff136344e8338ad52e819346de13495d79a5c615a8bfc9d45a00fa659d60630aa2025d8bc4570080d77d1e298e47a8934d47c0c3fb5b70b043ef2dd
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | df5b9d5ae99d8001b25448bf6148b9cf3a47ce8b4d1f7b1f52ab3dd8ad0ee6958045a382a561d3a617991a2255712f2e6ed0c8d6fb843edbdd0284054a238d74
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 9c0fc5952d8dcf07cb0225c252112fead50a4920d978f174a17a881d92c8b34ed162ec3a806bb451a990b81e8294f39755c93acbfd6147748c5dea9bb8ad38f2
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 0b203edad11a1350d3f3b6f16391450f2ca7bb19424a9eb398f69f8ef2195c821d3c72105248889cfb6efd317a3012d94c9564cb6b5794ba050fcc0902364eaa
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.27.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 68b2d73b7bf98445498e46188d39fde06acfee946ac9075f064f39bc7ea658cfcb82498ffe06bc2a2272ad5c105cac21d4929de47038e0ce95e07e3cd9d519ad

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.27.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.27.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.27.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.27.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.27.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.26.0

## Changes by Kind

### Deprecation

- Add warnings to the Services API. Kubernetes now warns for Services in the case of:
  - IPv4 addresses with leading zeros
  - IPv6 address in non-canonical format (RFC 5952) ([#114505](https://github.com/kubernetes/kubernetes/pull/114505), [@aojea](https://github.com/aojea)) [SIG Network]
- Support for the alpha seccomp annotations `seccomp.security.alpha.kubernetes.io/pod` and `container.seccomp.security.alpha.kubernetes.io`, deprecated since v1.19, has been completely removed. The seccomp fields are no longer auto-populated when pods with seccomp annotations are created. Pods should use the corresponding pod or container `securityContext.seccompProfile` field instead. ([#114947](https://github.com/kubernetes/kubernetes/pull/114947), [@saschagrunert](https://github.com/saschagrunert))

### API Change

- A terminating pod on a node that is not caused by preemption won't prevent kube-scheduler from preempting pods on that node
  - Rename 'PreemptionByKubeScheduler' to 'PreemptionByScheduler' ([#114623](https://github.com/kubernetes/kubernetes/pull/114623), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- Added new option to the InterPodAffinity scheduler plugin to ignore existing pods` preferred inter-pod affinities if the incoming pod has no preferred inter-pod affinities. This option can be used as an optimization for higher scheduling throughput (at the cost of an occasional pod being scheduled non-optimally/violating existing pods' preferred inter-pod affinities). To enable this scheduler option, set the InterPodAffinity scheduler plugin arg "ignorePreferredTermsOfExistingPods: true". ([#114393](https://github.com/kubernetes/kubernetes/pull/114393), [@danielvegamyhre](https://github.com/danielvegamyhre)) [SIG API Machinery and Scheduling]
- Added warnings about workload resources (Pods, ReplicaSets, Deployments, Jobs, CronJobs, or ReplicationControllers) whose names are not valid DNS labels. ([#114412](https://github.com/kubernetes/kubernetes/pull/114412), [@thockin](https://github.com/thockin)) [SIG API Machinery and Apps]
- K8s.io/component-base/logs: usage of the pflag values in a normal Go flag set led to panics when printing the help message ([#114680](https://github.com/kubernetes/kubernetes/pull/114680), [@pohly](https://github.com/pohly)) [SIG Instrumentation]
- Kube-proxy, kube-scheduler and kubelet have HTTP APIs for changing the logging verbosity at runtime. This now also works for JSON output. ([#114609](https://github.com/kubernetes/kubernetes/pull/114609), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Cloud Provider, Instrumentation and Testing]
- Kubeadm: explicitly set `priority` for static pods with `priorityClassName: system-node-critical` ([#114338](https://github.com/kubernetes/kubernetes/pull/114338), [@champtar](https://github.com/champtar)) [SIG Cluster Lifecycle]
- Kubelet: migrate "--container-runtime-endpoint" and "--image-service-endpoint" to kubelet config ([#112136](https://github.com/kubernetes/kubernetes/pull/112136), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery, Node and Scalability]
- Kubernetes components that perform leader election now only support using Leases for this. ([#114055](https://github.com/kubernetes/kubernetes/pull/114055), [@aimuz](https://github.com/aimuz)) [SIG API Machinery, Cloud Provider and Scheduling]
- StatefulSet names must be DNS labels, rather than subdomains.  Any StatefulSet which took advantage of subdomain validation (by having dots in the name) can't possibly have worked, because we eventually set `pod.spec.hostname` from the StatefulSetName, and that is validated as a DNS label. ([#114172](https://github.com/kubernetes/kubernetes/pull/114172), [@thockin](https://github.com/thockin)) [SIG Apps]
- The following feature gates for volume expansion GA features have been removed and must no longer be referenced in `--feature-gates` flags: ExpandCSIVolumes, ExpandInUsePersistentVolumes, ExpandPersistentVolumes ([#113942](https://github.com/kubernetes/kubernetes/pull/113942), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery, Apps and Testing]
- The list-type of the alpha resourceClaims field introduced to Pods in 1.26.0 was modified from "set" to "map", resolving an incompatibility with use of this schema in CustomResourceDefinitions and with server-side apply. ([#114585](https://github.com/kubernetes/kubernetes/pull/114585), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery]

### Feature

- Graduated the `LegacyServiceAccountTokenTracking` feature gate to Beta. The usage of auto-generated secret-based service account token now produces warnings by default, and relevant Secrets are labeled with a last-used timestamp (label key `kubernetes.io/legacy-token-last-used`). ([#114523](https://github.com/kubernetes/kubernetes/pull/114523), [@zshihang](https://github.com/zshihang)) [SIG API Machinery and Auth]
- Kube-proxy accepts the ContextualLogging, LoggingAlphaOptions, LoggingBetaOptions feature gates. ([#115233](https://github.com/kubernetes/kubernetes/pull/115233), [@pohly](https://github.com/pohly)) [SIG Instrumentation and Network]
- Kube-up now includes CoreDNS version v1.9.3 ([#114279](https://github.com/kubernetes/kubernetes/pull/114279), [@pacoxu](https://github.com/pacoxu)) [SIG Cloud Provider and Cluster Lifecycle]
- Kubeadm: add the experimental (alpha) feature gate EtcdLearnerMode that allows etcd members to be joined as learner and only then promoted as voting members ([#113318](https://github.com/kubernetes/kubernetes/pull/113318), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubectl will display SeccompProfile for pods, containers and ephemeral containers, if values were set. ([#113284](https://github.com/kubernetes/kubernetes/pull/113284), [@williamyeh](https://github.com/williamyeh)) [SIG CLI and Security]
- Kubectl: add e2e test for default container annotation ([#115046](https://github.com/kubernetes/kubernetes/pull/115046), [@pacoxu](https://github.com/pacoxu)) [SIG Architecture, CLI and Testing]
- Kubelet TCP and HTTP probes are more effective using networking resources: conntrack entries, sockets, ... 
  This is achieved by reducing the TIME-WAIT state of the connection to 1 second, instead of the defaults 60 seconds. This allows kubelet to free the socket, and free conntrack entry and ephemeral port associated. ([#115143](https://github.com/kubernetes/kubernetes/pull/115143), [@aojea](https://github.com/aojea)) [SIG Network and Node]
- Kubernetes is now built with Go 1.19.5 ([#115010](https://github.com/kubernetes/kubernetes/pull/115010), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Make `kubectl-convert` binary linking static (also affects the deb and rpm packages). ([#114228](https://github.com/kubernetes/kubernetes/pull/114228), [@saschagrunert](https://github.com/saschagrunert)) [SIG Release]
- New metrics `cidrset_cidrs_max_total` and `multicidrset_cidrs_max_total` expose the max number of CIDRs that can be allocated. ([#112260](https://github.com/kubernetes/kubernetes/pull/112260), [@aryan9600](https://github.com/aryan9600)) [SIG Apps, Instrumentation and Network]
- Profiling can now be served on a unix-domain socket by using the `--profiling-path` option (when profiling is enabled) for security purposes. ([#114191](https://github.com/kubernetes/kubernetes/pull/114191), [@apelisse](https://github.com/apelisse)) [SIG API Machinery]
- Scheduler doesn't run plugin's Filter method when its PreFilter method returned a Skip status.
  In other words, your PreFilter/Filter plugin can return a Skip status in PreFilter if the plugin does nothing in Filter for that Pod.
  Scheduler skips NodeAffinity Filter plugin when NodeAffinity Filter plugin has nothing to do with a Pod.
  It may affect some metrics values related to the NodeAffinity Filter plugin. ([#114125](https://github.com/kubernetes/kubernetes/pull/114125), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling, Storage and Testing]
- Scheduler skips InterPodAffinity Filter plugin when InterPodAffinity Filter plugin has nothing to do with a Pod.
  It may affect some metrics values related to the InterPodAffinity Filter plugin. ([#114889](https://github.com/kubernetes/kubernetes/pull/114889), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Scheduler volumebinding: leverage PreFilterResult to reduce down to only eligible node(s) for pod with bound claim(s) to local PersistentVolume(s) ([#109877](https://github.com/kubernetes/kubernetes/pull/109877), [@yibozhuang](https://github.com/yibozhuang)) [SIG Scheduling, Storage and Testing]
- The MinDomainsInPodTopologySpread feature gate is enabled by default as a Beta feature in 1.27. ([#114445](https://github.com/kubernetes/kubernetes/pull/114445), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Scheduling]
- The `AdvancedAuditing` feature gate was locked to _true_ in v1.27, and will be removed completely in v1.28 ([#115163](https://github.com/kubernetes/kubernetes/pull/115163), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery]
- Updated cAdvisor to v0.47.0 ([#114883](https://github.com/kubernetes/kubernetes/pull/114883), [@bobbypage](https://github.com/bobbypage)) [SIG Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- Use HorizontalPodAutoscaler v2 for kubectl ([#114886](https://github.com/kubernetes/kubernetes/pull/114886), [@a7i](https://github.com/a7i)) [SIG CLI]
- Verify that the key matches the cert ([#113581](https://github.com/kubernetes/kubernetes/pull/113581), [@aimuz](https://github.com/aimuz)) [SIG Apps]
- When any scheduler plugin returns an `unschedulableAndUnresolvable` status
  in `PostFilter`, the scheduling cycle terminates immediately for that Pod. ([#114699](https://github.com/kubernetes/kubernetes/pull/114699), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling and Testing]

### Documentation

- Error message for Pods with requests exceeding limits will have a limit value printed. ([#112925](https://github.com/kubernetes/kubernetes/pull/112925), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Apps and Node]

### Failing Test

- Deflake a preemption test that may patch Nodes incorrectly. ([#114350](https://github.com/kubernetes/kubernetes/pull/114350), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]

### Bug or Regression

- Adding (dry run) and (server dry run) suffixes to kubectl scale command when dry-run is passed ([#114252](https://github.com/kubernetes/kubernetes/pull/114252), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Change the error message to "cannot exec into multiple objects at a time" when file passed to kubectl exec contains multiple resources ([#114249](https://github.com/kubernetes/kubernetes/pull/114249), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Changing the error message of kubectl rollout restart when subsequent kubectl rollout restart commands are executed within a second ([#113040](https://github.com/kubernetes/kubernetes/pull/113040), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Client-go: fixes potential data races retrying requests using a custom io.Reader body; with this fix, only requests with no body or with string / []byte / runtime.Object bodies can be retried ([#113933](https://github.com/kubernetes/kubernetes/pull/113933), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Do not add DisruptionTarget condition by PodGC for pods which are in terminal phase ([#115056](https://github.com/kubernetes/kubernetes/pull/115056), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- Do not include preemptor pod metadata in the event message ([#114923](https://github.com/kubernetes/kubernetes/pull/114923), [@mimowo](https://github.com/mimowo)) [SIG Scheduling]
- Do not include preemptor pod metadata in the message of DisruptionTarget condition ([#114914](https://github.com/kubernetes/kubernetes/pull/114914), [@mimowo](https://github.com/mimowo)) [SIG Scheduling]
- Do not include scheduler name in the preemption event message ([#114980](https://github.com/kubernetes/kubernetes/pull/114980), [@mimowo](https://github.com/mimowo)) [SIG Scheduling]
- Don't create endpoints for Service of type ExternalName. ([#114814](https://github.com/kubernetes/kubernetes/pull/114814), [@panslava](https://github.com/panslava)) [SIG Apps, Network and Testing]
- Fail CRI connection if service or image endpoint is throwing any error on kubelet startup. ([#115102](https://github.com/kubernetes/kubernetes/pull/115102), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Failed pods associated with a job with `parallelism = 1` are recreated by the job controller honoring exponential backoff delay again. However, for jobs with `parallelism > 1`, pods might be created without exponential backoff delay. ([#114516](https://github.com/kubernetes/kubernetes/pull/114516), [@nikhita](https://github.com/nikhita)) [SIG Apps and Testing]
- Fix SELinux label for host path volumes created by host path provisioner ([#112021](https://github.com/kubernetes/kubernetes/pull/112021), [@mrunalp](https://github.com/mrunalp)) [SIG Node and Storage]
- Fix a bug on the endpointslice mirroring controller that generated multiple slices in some cases for custom endpoints in non canonical format ([#114155](https://github.com/kubernetes/kubernetes/pull/114155), [@aojea](https://github.com/aojea)) [SIG Apps, Network and Testing]
- Fix a bug where events/v1 Events with similar event type and reporting instance were not aggregated by client-go. ([#112365](https://github.com/kubernetes/kubernetes/pull/112365), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery and Instrumentation]
- Fix a bug where when emitting similar Events consecutively, some were rejected by the apiserver. ([#114237](https://github.com/kubernetes/kubernetes/pull/114237), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery]
- Fix a data race when emitting similar Events consecutively ([#114236](https://github.com/kubernetes/kubernetes/pull/114236), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery]
- Fix a regression that the scheduler always goes through all Filter plugins. ([#114518](https://github.com/kubernetes/kubernetes/pull/114518), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- Fix bug in CRD Validation Rules (beta) and ValidatingAdmissionPolicy (alpha) where all admission requests could result in `internal error: runtime error: index out of range [3] with length 3 evaluating rule: <rule name>` under certain circumstances. ([#114857](https://github.com/kubernetes/kubernetes/pull/114857), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Auth and Cloud Provider]
- Fix clearing of rate-limiter for the queue of checks for cleaning stale pod disruption conditions. 
  The bug could result in the PDB synchronization updates firing too often or the pod disruption cleanups taking too long to happen. ([#114770](https://github.com/kubernetes/kubernetes/pull/114770), [@mimowo](https://github.com/mimowo)) [SIG Apps]
- Fix: Route controller should update routes with NodeIP changed ([#108095](https://github.com/kubernetes/kubernetes/pull/108095), [@lzhecheng](https://github.com/lzhecheng)) [SIG Cloud Provider and Network]
- Fixed CSI PersistentVolumes to allow Secrets names longer than 63 characters. ([#114776](https://github.com/kubernetes/kubernetes/pull/114776), [@jsafrane](https://github.com/jsafrane)) [SIG Apps]
- Fixed DaemonSet to update the status even if it fails to create a pod. ([#113787](https://github.com/kubernetes/kubernetes/pull/113787), [@gjkim42](https://github.com/gjkim42)) [SIG Apps and Testing]
- Fixed StatefulSetAutoDeletePVC feature when OwnerReferencesPermissionEnforcement admission plugin is enabled. ([#114116](https://github.com/kubernetes/kubernetes/pull/114116), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Auth and Storage]
- Fixed bug in reflector that couldn't recover from "Too large resource version" errors with API servers before 1.17.0 ([#115093](https://github.com/kubernetes/kubernetes/pull/115093), [@xuzhenglun](https://github.com/xuzhenglun)) [SIG API Machinery]
- Fixed file permission issues that happened during update of Secret/ConfigMap/projected volume when fsGroup is used. The problem caused a race condition where application gets intermittent permission denied error when reading files that were just updated, before the correct permissions were applied. ([#114464](https://github.com/kubernetes/kubernetes/pull/114464), [@tsaarni](https://github.com/tsaarni)) [SIG Storage]
- Fixes panic validating custom resource definition schemas that set `multipleOf` to 0 ([#114869](https://github.com/kubernetes/kubernetes/pull/114869), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- Fixes stuck apiserver if an aggregated apiservice returned 304 Not Modified for aggregated discovery information ([#114459](https://github.com/kubernetes/kubernetes/pull/114459), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- Fixing issue in Winkernel Proxier - Unexpected active TCP connection drops while horizontally scaling the endpoints for a LoadBalancer Service with Internal Traffic Policy: Local ([#113742](https://github.com/kubernetes/kubernetes/pull/113742), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fixing issue on Windows when calculating cpu limits on nodes with more than 64 logical processors ([#114231](https://github.com/kubernetes/kubernetes/pull/114231), [@mweibel](https://github.com/mweibel)) [SIG Node and Windows]
- Fixing issue with Winkernel Proxier - No ingress load balancer rules with endpoints to support load balancing when all the endpoints are terminating. ([#113776](https://github.com/kubernetes/kubernetes/pull/113776), [@princepereira](https://github.com/princepereira)) [SIG Network, Testing and Windows]
- Hide .metadata.managedFields when describing CRs ([#114584](https://github.com/kubernetes/kubernetes/pull/114584), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- IPVS: Any ipvs scheduler can now be configured. If a un-usable scheduler is configured `kube-proxy` will re-start and the logs must be checked (same as before but different log printouts). ([#114878](https://github.com/kubernetes/kubernetes/pull/114878), [@uablrek](https://github.com/uablrek)) [SIG Network]
- If a user attempts to add an ephemeral container to a static pod, they will get a visible validation error. ([#114086](https://github.com/kubernetes/kubernetes/pull/114086), [@xmcqueen](https://github.com/xmcqueen)) [SIG Apps and Node]
- Kube-apiserver: removed N^2 behavior loading webhook configurations. ([#114794](https://github.com/kubernetes/kubernetes/pull/114794), [@lavalamp](https://github.com/lavalamp)) [SIG API Machinery, Architecture, CLI, Cloud Provider and Node]
- Kube-controller-manager will not run nodeipam controller when allocator type is CloudAllocator and the cloud provider is not enabled. ([#114596](https://github.com/kubernetes/kubernetes/pull/114596), [@andrewsykim](https://github.com/andrewsykim)) [SIG Cloud Provider]
- Kube-proxy with proxy-mode=ipvs can be used with statically linked kernels.
  The reseved IPv4 range TEST-NET-2 in rfc5737 MUST NOT be used for ClusterIP or loadBalancerIP since address 198.51.100.0 is used for probing. ([#114669](https://github.com/kubernetes/kubernetes/pull/114669), [@uablrek](https://github.com/uablrek)) [SIG Network]
- Kubeadm: fix the bug that kubeadm always do CRI detection even if it is not required by a phase subcommand ([#114455](https://github.com/kubernetes/kubernetes/pull/114455), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: improve retries when updating node information, in case kube-apiserver is temporarily unavailable ([#114176](https://github.com/kubernetes/kubernetes/pull/114176), [@QuantumEnergyE](https://github.com/QuantumEnergyE)) [SIG Cluster Lifecycle]
- Kubeadm: respect user provided kubeconfig during discovery process ([#113998](https://github.com/kubernetes/kubernetes/pull/113998), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl port-forward now exits with exit code 1 when remote connection is lost ([#114460](https://github.com/kubernetes/kubernetes/pull/114460), [@brianpursley](https://github.com/brianpursley)) [SIG API Machinery]
- Kubectl: use label selector for filtering out resources when pruning for kubectl diff ([#114863](https://github.com/kubernetes/kubernetes/pull/114863), [@danlenar](https://github.com/danlenar)) [SIG CLI and Testing]
- LabelSelectors specified in topologySpreadConstraints are now validated to ensure that pod is scheduled as expected. Existing pods with invalid LabelSelectors can be updated, but new pods are required to specify valid LabelSelectors. ([#111802](https://github.com/kubernetes/kubernetes/pull/111802), [@maaoBit](https://github.com/maaoBit)) [SIG Apps]
- Optimizing loadbalancer creation with the help of attribute Internal Traffic Policy: Local ([#114407](https://github.com/kubernetes/kubernetes/pull/114407), [@princepereira](https://github.com/princepereira)) [SIG Network]
- Relax API validation for usage key encipherment and kubelet uses requested usages accordingly ([#111660](https://github.com/kubernetes/kubernetes/pull/111660), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery, Apps, Auth and Node]
- Shared informers now correctly propagate whether they are synced or not. Individual informer handlers may now check if they are synced or not (new HasSynced method). Library support is added to assist controllers in tracking whether their own work is completed for items in the initial list (AsyncTracker). ([#113985](https://github.com/kubernetes/kubernetes/pull/113985), [@lavalamp](https://github.com/lavalamp)) [SIG API Machinery, Apps, Auth, Network, Node and Testing]
- Statefulset status will be consistent on API errors ([#113834](https://github.com/kubernetes/kubernetes/pull/113834), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- Total test spec is now available by `ProgressReporter`, it will be reported before test suite got executed. ([#114417](https://github.com/kubernetes/kubernetes/pull/114417), [@chendave](https://github.com/chendave)) [SIG Architecture, Auth, CLI, Cloud Provider, Instrumentation, Node and Testing]
- TryUnmount should respect `mounter.withSafeNotMountedBehavior` ([#114736](https://github.com/kubernetes/kubernetes/pull/114736), [@andyzhangx](https://github.com/andyzhangx)) [SIG Storage]
- When describing deployments, `OldReplicaSets` now always shows all replicasets controlled the deployment, not just those that still have replicas available. ([#113083](https://github.com/kubernetes/kubernetes/pull/113083), [@llorllale](https://github.com/llorllale)) [SIG CLI]

### Other (Cleanup or Flake)

- Callers of wait.ExponentialBackoffWithContext must pass a ConditionWithContextFunc to be consistent with the signature and avoid creating a duplicate context. If your condition does not need a context you can use the `ConditionFunc.WithContext()` helper to ignore the context, or use ExponentialBackoff directly. ([#115113](https://github.com/kubernetes/kubernetes/pull/115113), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery, Storage and Testing]
- Fix incorrect log information ([#110723](https://github.com/kubernetes/kubernetes/pull/110723), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Network]
- Improved misleading message, in case of no metrics received for the HPA controlled pods. ([#114740](https://github.com/kubernetes/kubernetes/pull/114740), [@kushagra98](https://github.com/kushagra98)) [SIG Apps and Autoscaling]
- Kubeadm: remove the deprecated v1beta2 API. kubeadm 1.26's "config migrate" command can be used to migrate a v1beta2 configuration file to v1beta3. ([#114540](https://github.com/kubernetes/kubernetes/pull/114540), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Remove unused rule for `nodes/spec` from `ClusterRole system:kubelet-api-admin` ([#113267](https://github.com/kubernetes/kubernetes/pull/113267), [@hoskeri](https://github.com/hoskeri)) [SIG Auth and Cloud Provider]
- Renamed API server identity Lease labels to use the key `apiserver.kubernetes.io/identity` ([#114586](https://github.com/kubernetes/kubernetes/pull/114586), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, Apps, Cloud Provider and Testing]
- The CSIMigrationAzureFile feature gate (for the feature which graduated to GA in v1.26) is now unconditionally enabled and will be removed in v1.28. ([#114953](https://github.com/kubernetes/kubernetes/pull/114953), [@enj](https://github.com/enj)) [SIG Storage]
- The WaitFor and WaitForWithContext functions in the wait package have been marked private. Callers should use the equivalent Poll* method with a zero duration interval. ([#115116](https://github.com/kubernetes/kubernetes/pull/115116), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery]
- The feature gates `CSIInlineVolume`, `CSIMigration`, `DaemonSetUpdateSurge`, `EphemeralContainers`, `IdentifyPodOS`, `LocalStorageCapacityIsolation`, `NetworkPolicyEndPort` and `StatefulSetMinReadySeconds` that graduated to GA in v1.25 and were unconditionally enabled have been removed in v1.27 ([#114410](https://github.com/kubernetes/kubernetes/pull/114410), [@SataQiu](https://github.com/SataQiu)) [SIG Node]
- This flag `master-service-namespace` will be removed in v1.27. ([#114446](https://github.com/kubernetes/kubernetes/pull/114446), [@lengrongfu](https://github.com/lengrongfu)) [SIG API Machinery]
- Wait.ContextForChannel() now implements the context.Context interface and does not return a cancellation function. ([#115140](https://github.com/kubernetes/kubernetes/pull/115140), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery and Cloud Provider]

## Dependencies

### Added
- github.com/a8m/tree: [10a5fd5](https://github.com/a8m/tree/tree/10a5fd5)
- github.com/dougm/pretty: [2ee9d74](https://github.com/dougm/pretty/tree/2ee9d74)
- github.com/rasky/go-xdr: [4930550](https://github.com/rasky/go-xdr/tree/4930550)
- github.com/vmware/vmw-guestinfo: [25eff15](https://github.com/vmware/vmw-guestinfo/tree/25eff15)

### Changed
- github.com/Microsoft/hcsshim: [v0.8.22 → v0.8.25](https://github.com/Microsoft/hcsshim/compare/v0.8.22...v0.8.25)
- github.com/aws/aws-sdk-go: [v1.44.116 → v1.44.147](https://github.com/aws/aws-sdk-go/compare/v1.44.116...v1.44.147)
- github.com/coredns/corefile-migration: [v1.0.17 → v1.0.18](https://github.com/coredns/corefile-migration/compare/v1.0.17...v1.0.18)
- github.com/creack/pty: [v1.1.11 → v1.1.18](https://github.com/creack/pty/compare/v1.1.11...v1.1.18)
- github.com/docker/docker: [v20.10.18+incompatible → v20.10.21+incompatible](https://github.com/docker/docker/compare/v20.10.18...v20.10.21)
- github.com/go-openapi/jsonpointer: [v0.19.5 → v0.19.6](https://github.com/go-openapi/jsonpointer/compare/v0.19.5...v0.19.6)
- github.com/go-openapi/jsonreference: [v0.20.0 → v0.20.1](https://github.com/go-openapi/jsonreference/compare/v0.20.0...v0.20.1)
- github.com/go-openapi/swag: [v0.19.14 → v0.22.3](https://github.com/go-openapi/swag/compare/v0.19.14...v0.22.3)
- github.com/google/cadvisor: [v0.46.0 → v0.47.1](https://github.com/google/cadvisor/compare/v0.46.0...v0.47.1)
- github.com/google/cel-go: [v0.12.5 → v0.12.6](https://github.com/google/cel-go/compare/v0.12.5...v0.12.6)
- github.com/google/uuid: [v1.1.2 → v1.3.0](https://github.com/google/uuid/compare/v1.1.2...v1.3.0)
- github.com/kr/pretty: [v0.2.1 → v0.3.0](https://github.com/kr/pretty/compare/v0.2.1...v0.3.0)
- github.com/mailru/easyjson: [v0.7.6 → v0.7.7](https://github.com/mailru/easyjson/compare/v0.7.6...v0.7.7)
- github.com/moby/ipvs: [v1.0.1 → v1.1.0](https://github.com/moby/ipvs/compare/v1.0.1...v1.1.0)
- github.com/moby/term: [39b0c02 → 1aeaba8](https://github.com/moby/term/compare/39b0c02...1aeaba8)
- github.com/onsi/ginkgo/v2: [v2.4.0 → v2.7.0](https://github.com/onsi/ginkgo/v2/compare/v2.4.0...v2.7.0)
- github.com/onsi/gomega: [v1.23.0 → v1.24.2](https://github.com/onsi/gomega/compare/v1.23.0...v1.24.2)
- github.com/opencontainers/runtime-spec: [1c3f411 → 494a5a6](https://github.com/opencontainers/runtime-spec/compare/1c3f411...494a5a6)
- github.com/rogpeppe/go-internal: [v1.3.0 → v1.9.0](https://github.com/rogpeppe/go-internal/compare/v1.3.0...v1.9.0)
- github.com/sirupsen/logrus: [v1.8.1 → v1.9.0](https://github.com/sirupsen/logrus/compare/v1.8.1...v1.9.0)
- github.com/stretchr/objx: [v0.4.0 → v0.5.0](https://github.com/stretchr/objx/compare/v0.4.0...v0.5.0)
- github.com/stretchr/testify: [v1.8.0 → v1.8.1](https://github.com/stretchr/testify/compare/v1.8.0...v1.8.1)
- github.com/tmc/grpc-websocket-proxy: [e5319fd → 673ab2c](https://github.com/tmc/grpc-websocket-proxy/compare/e5319fd...673ab2c)
- github.com/vishvananda/netns: [db3c7e5 → v0.0.2](https://github.com/vishvananda/netns/compare/db3c7e5...v0.0.2)
- github.com/vmware/govmomi: [v0.20.3 → v0.30.0](https://github.com/vmware/govmomi/compare/v0.20.3...v0.30.0)
- golang.org/x/mod: v0.6.0 → v0.7.0
- golang.org/x/net: 1e63c2f → v0.4.0
- golang.org/x/sync: 886fb93 → v0.1.0
- golang.org/x/tools: v0.2.0 → v0.4.0
- golang.org/x/xerrors: 5ec99f8 → 04be3eb
- google.golang.org/grpc: v1.49.0 → v1.51.0
- gopkg.in/check.v1: 8fa4692 → 10cb982
- k8s.io/kube-openapi: 172d655 → 3758b55
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.33 → v0.1.1

### Removed
- github.com/elazarl/goproxy: [947c36d](https://github.com/elazarl/goproxy/tree/947c36d)
- github.com/mindprince/gonvml: [9ebdce4](https://github.com/mindprince/gonvml/tree/9ebdce4)