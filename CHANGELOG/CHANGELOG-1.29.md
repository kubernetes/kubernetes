<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.29.0-alpha.2](#v1290-alpha2)
  - [Downloads for v1.29.0-alpha.2](#downloads-for-v1290-alpha2)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.29.0-alpha.1](#changelog-since-v1290-alpha1)
  - [Changes by Kind](#changes-by-kind)
    - [Feature](#feature)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.29.0-alpha.1](#v1290-alpha1)
  - [Downloads for v1.29.0-alpha.1](#downloads-for-v1290-alpha1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.28.0](#changelog-since-v1280)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature-1)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)

<!-- END MUNGE: GENERATED_TOC -->

# v1.29.0-alpha.2


## Downloads for v1.29.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes.tar.gz) | 138f47b2c53030e171d368d382c911048ce5d8387450e5e6717f09ac8cf6289b6c878046912130d58d7814509bbc45dbc19d6ee4f24404321ea18b24ebab2a36
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-src.tar.gz) | 73ab06309d6f6cbcb8a417c068367b670a04dcbe90574a7906201dd70b9c322cd052818114b746a4d61b7bce6115ae547eaafc955c41053898a315c968db2f36

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | c9604fbb9e848a4b3dc85ee2836f74b4ccd321e4c72d22b2d4558eb0f0c3833bff35d0c36602c13c5c5c79e9233fda874bfa85433291ab3484cf61c9012ee515
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | fed42ecbfc20b5f63ac48bbb9b73abc4b72aca76ac8bdd51b9ea6af053b1fc6a8e63b5e11f9d14c4814f03b49531da2536f1342cda2da03514c44ccf05c311b0
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 93c61229d7b07a476296b5b800c853c8e984101d5077fc19a195673f7543e7d2eb2599311c1846c91ef1f7ae29c3e05b6f41b873e92a3429563e3d83900050da
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 4260b49733f6b0967c504e2246b455b2348b487e84f7a019fda8b4a87d43d27a03e7ed55b505764c14f2079c4c3d71c68d77f981b604e13e7210680f45ee66e3
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 4e837fd2f55cbb5f93cdf60235511a85635485962f00e0378a95a7ff846eb86b7bf053203ab353b294131b2e2663d0e783dae79c18601d4d66f98a6e5152e96e
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 6f3954d2adc289879984d18c2605110a7d5f0a5f6366233c25adf3a742f8dc1183e8a4d4747de8077af1045a259b150e0e86b27e10d683aa8decdc760ac6279b
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 741b76827ff9e810e490d8698eb7620826a16e978e5c7744a1fa0e65124690cfc9601e7f1c8f50e77f25185ba3176789ddcb7d5caaddde66436c31658bacde1d
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 0c635883e2f9caca03bcf3b42ba0b479f44c8cc2a3d5dd425b0fee278f3e884bef0e897fe51cbf00bb0bc061371805d9f9cbccf839477671f92e078c04728735
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-windows-386.tar.gz) | ebddbb358fd2d817908069eb66744dc62cae56ad470b1e36c6ebd0d2284e79ae5b9a5f8a86fef365f30b34e14093827ad736814241014f597e2ac88788102cf4
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 01a451a809cd45e7916a3e982e2b94d372accab9dfe20667e95c10d56f9194b997721c0c219ff7ff97828b6466108eec6e57dcb33e3e3b0c5f770af1514a9f1a
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 473ba648ffde41fd5b63374cc1595eb43b873808c6b0cc5e939628937f3f7fb36dba4b7c7c8ef03408d557442094ec22e12c03f40be137f9cc99761b4cc1a1f8

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | c3f7abcee3fdcf6f311b5de0bfe037318e646641c1ce311950d920252623cca285d1f1cef0e2d936c0f981edc1c725897a42aa9e03b77fe5f76f1090665d967f
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 17614842df6bb528434b8b063b1d1c3efc8e4eff9cbc182f049d811f68e08514026fbb616199a3dee97e62ce2fd1eda0b9778d8e74040e645c482cfe6a18a8b4
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 2f818035ef199a7745e24d2ce86abf6c52e351d7922885e264c5d07db3e0f21048c32db85f3044e01443abd87a45f92df52fda44e8df05000754b03f34132f2f
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 96a34c768f347f23c46f990a8f6ddf3127b13f7a183453b92eb7bc27ce896767f31b38317a6ae5a11f2d4b459ec9564385f8abe61082a4165928edfee0c9765e

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 66845cf86e32c19be9d339417a4772b9bcf51b2bf4d1ef5acc2e9eb006bbd19b3c036aa3721b3d8fe08b6fb82284ba25a6ecb5eb7b84f657cc968224d028f22c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 98902ee33242f9e78091433115804d54eafde24903a3515f0300f60c0273c7c0494666c221ce418d79e715f8ecf654f0edabc5b69765da26f83a812e963b5afb
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 82f1213b5942c5c1576afadb4b066dfa1427c7709adf6ba636b9a52dfdb1b20f62b1cc0436b265e714fbee08c71d8786295d2439c10cc05bd58b2ab2a87611d4
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 7cb8cb65195c5dd63329d02907cdbb0f5473066606c108f4516570f449623f93b1ca822d5a00fad063ec8630e956fa53a0ab530a8487bccb01810943847d4942
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 1222e2d7dbaf7920e1ba927231cc7e275641cf0939be1520632353df6219bbcb3b49515d084e7f2320a2ff59b2de9fee252d8f5e9c48d7509f1174c6cb357b66

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.29.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.29.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.29.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.29.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.29.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.29.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.29.0-alpha.1

## Changes by Kind

### Feature

- Adds `apiserver_watch_list_duration_seconds` metrics. Which will measure response latency distribution in seconds for watch list requests broken by group, version, resource and scope ([#120490](https://github.com/kubernetes/kubernetes/pull/120490), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Instrumentation]
- Allow-list of metric labels can be configured by supplying a manifest using the --allow-metric-labels-manifest flag ([#118299](https://github.com/kubernetes/kubernetes/pull/118299), [@rexagod](https://github.com/rexagod)) [SIG Architecture and Instrumentation]
- Bump distroless-iptables to 0.3.3 based on Go 1.21.2 ([#121073](https://github.com/kubernetes/kubernetes/pull/121073), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- Implements API for streaming for the etcd store implementation
  
  When sendInitialEvents ListOption is set together with watch=true, it begins the watch stream with synthetic init events followed by a synthetic "Bookmark" after which the server continues streaming events. ([#119557](https://github.com/kubernetes/kubernetes/pull/119557), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kubelet, when using cloud provider external, initializes temporary the node addresses using the --node-ip flag values if set, until the cloud provider overrides it. ([#121028](https://github.com/kubernetes/kubernetes/pull/121028), [@aojea](https://github.com/aojea)) [SIG Cloud Provider and Node]
- Kubernetes is now built with Go 1.21.2 ([#121021](https://github.com/kubernetes/kubernetes/pull/121021), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Migrated the volumebinding scheduler plugins to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116803](https://github.com/kubernetes/kubernetes/pull/116803), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation, Scheduling and Storage]
- The kube-apiserver exposes four new metrics to inform about errors on the clusterIP and nodePort allocation logic ([#120843](https://github.com/kubernetes/kubernetes/pull/120843), [@aojea](https://github.com/aojea)) [SIG Instrumentation and Network]

### Failing Test

- K8s.io/dynamic-resource-allocation: DRA drivers updating to this release are compatible with Kubernetes 1.27 and 1.28. ([#120868](https://github.com/kubernetes/kubernetes/pull/120868), [@pohly](https://github.com/pohly)) [SIG Node]

### Bug or Regression

- Cluster-bootstrap: improve the security of the functions responsible for generation and validation of bootstrap tokens ([#120400](https://github.com/kubernetes/kubernetes/pull/120400), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Security]
- Do not fail volume attach or publish operation at kubelet if target path directory already exists on the node. ([#119735](https://github.com/kubernetes/kubernetes/pull/119735), [@akankshapanse](https://github.com/akankshapanse)) [SIG Storage]
- Fix regression with adding aggregated apiservices panicking and affected health check introduced in release v1.28.0 ([#120814](https://github.com/kubernetes/kubernetes/pull/120814), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and Testing]
- Fixed a bug where containers would not start on cgroupv2 systems where swap is disabled. ([#120784](https://github.com/kubernetes/kubernetes/pull/120784), [@elezar](https://github.com/elezar)) [SIG Node]
- Fixed a regression in kube-proxy where it might refuse to start if given
  single-stack IPv6 configuration options on a node that has both IPv4 and
  IPv6 IPs. ([#121008](https://github.com/kubernetes/kubernetes/pull/121008), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Fixed attaching volumes after detach errors. Now volumes that failed to detach are not treated as attached, Kubernetes will make sure they are fully attached before they can be used by pods. ([#120595](https://github.com/kubernetes/kubernetes/pull/120595), [@jsafrane](https://github.com/jsafrane)) [SIG Apps and Storage]
- Fixes a regression (CLIENTSET_PKG: unbound variable) when invoking deprecated generate-groups.sh script ([#120877](https://github.com/kubernetes/kubernetes/pull/120877), [@soltysh](https://github.com/soltysh)) [SIG API Machinery]
- K8s.io/dynamic-resource-allocation/controller: UnsuitableNodes did not handle a mix of allocated and unallocated claims correctly. ([#120338](https://github.com/kubernetes/kubernetes/pull/120338), [@pohly](https://github.com/pohly)) [SIG Node]
- K8s.io/dynamic-resource-allocation: handle a selected node which isn't listed as potential node ([#120871](https://github.com/kubernetes/kubernetes/pull/120871), [@pohly](https://github.com/pohly)) [SIG Node]
- Kubeadm: fix the bug that kubeadm always do CRI detection when --config is passed even if it is not required by the subcommand ([#120828](https://github.com/kubernetes/kubernetes/pull/120828), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]

### Other (Cleanup or Flake)

- Client-go: k8s.io/client-go/tools events and record packages have new APIs for specifying a context and logger ([#120729](https://github.com/kubernetes/kubernetes/pull/120729), [@pohly](https://github.com/pohly)) [SIG API Machinery and Instrumentation]
- Deprecated the `--cloud-provider` and `--cloud-config` CLI parameters in kube-apiserver.
  These parameters will be removed in a future release. ([#120903](https://github.com/kubernetes/kubernetes/pull/120903), [@dims](https://github.com/dims)) [SIG API Machinery]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/emicklei/go-restful/v3: [v3.9.0 → v3.11.0](https://github.com/emicklei/go-restful/v3/compare/v3.9.0...v3.11.0)
- github.com/onsi/ginkgo/v2: [v2.9.4 → v2.13.0](https://github.com/onsi/ginkgo/v2/compare/v2.9.4...v2.13.0)
- github.com/onsi/gomega: [v1.27.6 → v1.28.0](https://github.com/onsi/gomega/compare/v1.27.6...v1.28.0)
- golang.org/x/crypto: v0.11.0 → v0.12.0
- golang.org/x/mod: v0.10.0 → v0.12.0
- golang.org/x/net: v0.13.0 → v0.14.0
- golang.org/x/sync: v0.2.0 → v0.3.0
- golang.org/x/sys: v0.10.0 → v0.12.0
- golang.org/x/term: v0.10.0 → v0.11.0
- golang.org/x/text: v0.11.0 → v0.12.0
- golang.org/x/tools: v0.8.0 → v0.12.0

### Removed
_Nothing has changed._



# v1.29.0-alpha.1


## Downloads for v1.29.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes.tar.gz) | 107062e8da7c416206f18b4376e9e0c2ca97b37c720a047f2bc6cf8a1bdc2b41e84defd0a29794d9562f3957932c0786a5647450b41d2850a9b328826bb3248d
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-src.tar.gz) | 8182774faa5547f496642fdad7e2617a4d07d75af8ddf85fb8246087ddffab596528ffde29500adc9945d4e263fce766927ed81396a11f88876b3fa76628a371

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | ac9a08cd98af5eb27f8dde895510db536098dd52ee89682e7f103c793cb99cddcd992e3a349d526854caaa27970aa1ef964db4cc27d1009576fb604bf0c1cdf1
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 28744076618dcd7eca4175726d7f3ac67fe94f08f1b6ca4373b134a6402c0f5203f1146d79a211443c751b2f2825df3507166fc3c5e40a55d545c3e5d2a48e56
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 0207a2571b6d0e6e55f36af9d2ed27f31eacfb23f2f54dd2eb8fbc38ef5b033edb24fb9a5ece7e7020fd921a9c841fff435512d12421bfa13294cc9c297eb877
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 57fc39ba259ae61b88c23fd136904395abc23c44f4b4db3e2922827ec7e6def92bc77364de3e2f6b54b27bb4b5e42e9cf4d1c0aa6d12c4a5a17788d9f996d9ad
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 53a54d3fbda46162139a90616d708727c23d3aae0a2618197df5ac443ac3d49980a62034e3f2514f1a1622e4ce5f6e821d2124a61a9e63ce6d29268b33292949
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | ee3ca4626c802168db71ad55c1d8b45c03ec774c146dd6da245e5bb26bf7fd6728a477f1ad0c5094967a0423f94e35e4458c6716f3abe005e8fc55ae354174cf
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 60cd35076dd4afb9005349003031fa9f1802a2a120fbbe842d6fd061a1bca39baabcbb18fb4b6610a5ca626fc64e1d780c7aadb203d674697905489187a415ce
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 68fdd0fc35dfd6fae0d25d7834270c94b16ae860fccc4253e7c347ce165d10cadc190e8b320fd2c4afd508afc6c10f246b8a5f0148ca1b1d56f7b2843cc39d30
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 0c5d3dbfaaffa81726945510c972cc15895ea87bcd43b798675465fdadaa4d2d9597cb4fc6baee9ee719c919d1f46a9390c15cb0da60250f41eb4fcc3337b337
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 2e519867cbc793ea1c1e45f040de81b49c70b9b42fac072ac5cac36e8de71f0dddd0c64354631bcb2b3af36a0f377333c0cd885c2df36ef8cd7e6c8fd5628aa4
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 1a80cad80c1c9f753a38e6c951b771b0df820455141f40ba44e227f6acc81b59454f8dbff12e83c61bf647eaa1ff98944930969a99c96a087a35921f4e6ac968

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | c74a3f7bdd16095fb366b4313e50984f2ee7cb99c77ad2bcccea066756ce6e0fc45f4528b79c8cb7e6370430ee2d03fa6bc10ca87a59d8684a59e1ebd3524afd
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | b6844b5769fd5687525dcedca42c7bb036f6acad65d3de3c8cda46dbbe0ac23c289fdb7fbf15f1c37184498d6a1fb018e41e1c97ded4581f045ad2039e3ddec2
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | a15eb2db4821454974920a987bb1e73bc4ee638b845b07f35cab55dcf482c142d3cdaed347bfa0452d5311b3d9152463a3dae1d176b6101ed081ec594e0d526c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 60e24d8b4902821b436b5adebd6594ef0db79802d64787a1424aa6536873e2d749dfc6ebc2eb81db3240c925500a3e927ee7385188f866c28123736459e19b7b

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 44832c7b90c88e7ca70737bad8d50ee8ba434ee7a94940f9d45beda9e9aadc7e2c973b65fcb986216229796a5807dae2470dbcf1ade5c075d86011eefe21509b
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | a13862d9bae0ff358377afc60f5222490a8e6bb7197d4a7d568edd4f150348f7a3dc7342129cd2d5c5353d2d43349b97c854df3e8886a8d52aedb95c634e3b5a
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 57348f82bb4db8c230d8dffdef513ed75d7b267b226a5d15b3deb9783f8ed56fe40f8ce018ab34c28f9f8210b2e41b0f55d185dcdbaf912dd57e2ea78f8d3c53
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 2013eb4746e818cf336e0fee37650df98c19876030397803abce9531730eb0b95e6284f5a2abdd2b97090a67d07fd7a9c74c84fc7b4b83f0bce04a6dc9ad2555
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 3a4d63e2117cdbebc655e674bb017e246c263e893fc0ca3e8dc0091d6d9f96c9f0756c0fa8b45ba461502ae432f908ea922c21378b82ff3990b271f42eedc138

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.29.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.29.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.29.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.29.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.29.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.29.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.28.0

## Changes by Kind

### Deprecation

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
  --> ([#119495](https://github.com/kubernetes/kubernetes/pull/119495), [@bzsuni](https://github.com/bzsuni)) [SIG API Machinery]

### API Change

- Added a new `ipMode` field to the `.status` of Services where `type` is set to `LoadBalancer`.
  The new field is behind the `LoadBalancerIPMode` feature gate. ([#119937](https://github.com/kubernetes/kubernetes/pull/119937), [@RyanAoh](https://github.com/RyanAoh)) [SIG API Machinery, Apps, Cloud Provider, Network and Testing]
- Fixed a bug where CEL expressions in CRD validation rules would incorrectly compute a high estimated cost for functions that return strings, lists or maps.
  The incorrect cost was evident when the result of a function was used in subsequent operations. ([#119800](https://github.com/kubernetes/kubernetes/pull/119800), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Auth and Cloud Provider]
- Go API: the ResourceRequirements struct needs to be replaced with VolumeResourceRequirements for use with volumes. ([#118653](https://github.com/kubernetes/kubernetes/pull/118653), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Node, Scheduling, Storage and Testing]
- Kube-apiserver: adds --authentication-config flag for reading AuthenticationConfiguration files. --authentication-config flag is mutually exclusive with the existing --oidc-* flags. ([#119142](https://github.com/kubernetes/kubernetes/pull/119142), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-scheduler component config (KubeSchedulerConfiguration) kubescheduler.config.k8s.io/v1beta3 is removed in v1.29. Migrate kube-scheduler configuration files to kubescheduler.config.k8s.io/v1. ([#119994](https://github.com/kubernetes/kubernetes/pull/119994), [@SataQiu](https://github.com/SataQiu)) [SIG Scheduling and Testing]
- Mark the onPodConditions field as optional in Job's pod failure policy. ([#120204](https://github.com/kubernetes/kubernetes/pull/120204), [@mimowo](https://github.com/mimowo)) [SIG API Machinery and Apps]
- Retry NodeStageVolume calls if CSI node driver is not running ([#120330](https://github.com/kubernetes/kubernetes/pull/120330), [@rohitssingh](https://github.com/rohitssingh)) [SIG Apps, Storage and Testing]
- The kube-scheduler `selectorSpread` plugin has been removed, please use the `podTopologySpread` plugin instead. ([#117720](https://github.com/kubernetes/kubernetes/pull/117720), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]

### Feature

- --sync-frequency will not affect the update interval of volumes that use ConfigMaps or Secrets when the configMapAndSecretChangeDetectionStrategy is set to Cache. The update interval is only affected by node.alpha.kubernetes.io/ttl node annotation." ([#120255](https://github.com/kubernetes/kubernetes/pull/120255), [@likakuli](https://github.com/likakuli)) [SIG Node]
- Add a new scheduler metric, `pod_scheduling_sli_duration_seconds`, and start the deprecation for `pod_scheduling_duration_seconds`. ([#119049](https://github.com/kubernetes/kubernetes/pull/119049), [@helayoty](https://github.com/helayoty)) [SIG Instrumentation, Scheduling and Testing]
- Added apiserver_envelope_encryption_dek_cache_filled to measure number of records in data encryption key(DEK) cache. ([#119878](https://github.com/kubernetes/kubernetes/pull/119878), [@ritazh](https://github.com/ritazh)) [SIG API Machinery and Auth]
- Added kubectl node drain helper callbacks `OnPodDeletionOrEvictionStarted` and `OnPodDeletionOrEvictionFailed`; people extending `kubectl` can use these new callbacks for more granularity.
  - Deprecated the `OnPodDeletedOrEvicted` node drain helper callback. ([#117502](https://github.com/kubernetes/kubernetes/pull/117502), [@adilGhaffarDev](https://github.com/adilGhaffarDev)) [SIG CLI]
- Adding apiserver identity to the following metrics: 
  apiserver_envelope_encryption_key_id_hash_total, apiserver_envelope_encryption_key_id_hash_last_timestamp_seconds, apiserver_envelope_encryption_key_id_hash_status_last_timestamp_seconds, apiserver_encryption_config_controller_automatic_reload_failures_total, apiserver_encryption_config_controller_automatic_reload_success_total, apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds
  
  Fix bug to surface events for the following metrics: apiserver_encryption_config_controller_automatic_reload_failures_total, apiserver_encryption_config_controller_automatic_reload_last_timestamp_seconds, apiserver_encryption_config_controller_automatic_reload_success_total ([#120438](https://github.com/kubernetes/kubernetes/pull/120438), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth, Instrumentation and Testing]
- Bump distroless-iptables to 0.3.2 based on Go 1.21.1 ([#120527](https://github.com/kubernetes/kubernetes/pull/120527), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- Changed `kubectl help` to display basic details for subcommands from plugins ([#116752](https://github.com/kubernetes/kubernetes/pull/116752), [@xvzf](https://github.com/xvzf)) [SIG CLI]
- Changed the `KMSv2KDF` feature gate to be enabled by default. ([#120433](https://github.com/kubernetes/kubernetes/pull/120433), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Graduated the following kubelet resource metrics to **general availability**:
  - `container_cpu_usage_seconds_total`
  - `container_memory_working_set_bytes`
  - `container_start_time_seconds`
  - `node_cpu_usage_seconds_total`
  - `node_memory_working_set_bytes`
  - `pod_cpu_usage_seconds_total`
  - `pod_memory_working_set_bytes`
  - `resource_scrape_error`
  
  Deprecated (renamed) `scrape_error` in favor of `resource_scrape_error` ([#116897](https://github.com/kubernetes/kubernetes/pull/116897), [@Richabanker](https://github.com/Richabanker)) [SIG Architecture, Instrumentation, Node and Testing]
- Graduation API List chunking (aka pagination) feature to stable ([#119503](https://github.com/kubernetes/kubernetes/pull/119503), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Cloud Provider and Testing]
- Implements API for streaming for the etcd store implementation
  
  When sendInitialEvents ListOption is set together with watch=true, it begins the watch stream with synthetic init events followed by a synthetic "Bookmark" after which the server continues streaming events. ([#119557](https://github.com/kubernetes/kubernetes/pull/119557), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Improve memory usage of kube-scheduler by dropping the `.metadata.managedFields` field that kube-scheduler doesn't require. ([#119556](https://github.com/kubernetes/kubernetes/pull/119556), [@linxiulei](https://github.com/linxiulei)) [SIG Scheduling]
- In a scheduler with Permit plugins, when a Pod is rejected during WaitOnPermit, the scheduler records the plugin.
  The scheduler will use the record to honor cluster events and queueing hints registered for the plugin, to inform whether to retry the pod. ([#119785](https://github.com/kubernetes/kubernetes/pull/119785), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- In tree cloud providers are now switched off by default. Please use DisableCloudProviders and DisableKubeletCloudCredentialProvider feature flags if you still need this functionality. ([#117503](https://github.com/kubernetes/kubernetes/pull/117503), [@dims](https://github.com/dims)) [SIG API Machinery, Cloud Provider and Testing]
- Introduce new apiserver metric apiserver_flowcontrol_current_inqueue_seats. This metric is analogous to `apiserver_flowcontrol_current_inqueue_requests` but tracks totals seats as each request can take more than 1 seat. ([#119385](https://github.com/kubernetes/kubernetes/pull/119385), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery]
- Kube-proxy don't panic on exit when the Node object changes its PodCIDR ([#120375](https://github.com/kubernetes/kubernetes/pull/120375), [@pegasas](https://github.com/pegasas)) [SIG Network]
- Kube-proxy will only install the DROP rules for invalid conntrack states if the nf_conntrack_tcp_be_liberal is not set. ([#120412](https://github.com/kubernetes/kubernetes/pull/120412), [@aojea](https://github.com/aojea)) [SIG Network]
- Kubeadm: add validation to verify that the CertificateKey is a valid hex encoded AES key ([#120064](https://github.com/kubernetes/kubernetes/pull/120064), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: promoted feature gate `EtcdLearnerMode` to beta. Learner mode for joining etcd members is now enabled by default. ([#120228](https://github.com/kubernetes/kubernetes/pull/120228), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubelet exposes latency metrics of different stages of the node startup. ([#118568](https://github.com/kubernetes/kubernetes/pull/118568), [@qiutongs](https://github.com/qiutongs)) [SIG Instrumentation, Node and Scalability]
- Kubernetes is now built with Go 1.21.1 ([#120493](https://github.com/kubernetes/kubernetes/pull/120493), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with go 1.21.0 ([#118996](https://github.com/kubernetes/kubernetes/pull/118996), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- List the pods using <PVC> as an ephemeral storage volume in "Used by:" part of the output of `kubectl describe pvc <PVC>` command. ([#120427](https://github.com/kubernetes/kubernetes/pull/120427), [@MaGaroo](https://github.com/MaGaroo)) [SIG CLI]
- Migrated the nodevolumelimits scheduler plugin to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#116884](https://github.com/kubernetes/kubernetes/pull/116884), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation, Node, Scheduling, Storage and Testing]
- Promote ServiceNodePortStaticSubrange to stable and lock to default ([#120233](https://github.com/kubernetes/kubernetes/pull/120233), [@xuzhenglun](https://github.com/xuzhenglun)) [SIG Network]
- QueueingHint got error in its returning value. If QueueingHint returns error, the scheduler logs the error and treats the event as QueueAfterBackoff so that the Pod wouldn't be stuck in the unschedulable pod pool. ([#119290](https://github.com/kubernetes/kubernetes/pull/119290), [@carlory](https://github.com/carlory)) [SIG Node, Scheduling and Testing]
- Remove /livez livezchecks for KMS v1 and v2 to ensure KMS health does not cause kube-apiserver restart. KMS health checks are still in place as a healthz and readiness checks. ([#120583](https://github.com/kubernetes/kubernetes/pull/120583), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth and Testing]
- The CloudDualStackNodeIPs feature is now beta, meaning that when using
  an external cloud provider that has been updated to support the feature,
  you can pass comma-separated dual-stack `--node-ips` to kubelet and have
  the cloud provider take both IPs into account. ([#120275](https://github.com/kubernetes/kubernetes/pull/120275), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Cloud Provider and Network]
- The Dockerfile for the kubectl image has been updated with the addition of a specific base image and essential utilities (bash and jq). ([#119592](https://github.com/kubernetes/kubernetes/pull/119592), [@rayandas](https://github.com/rayandas)) [SIG CLI, Node, Release and Testing]
- Use of secret-based service account tokens now adds an `authentication.k8s.io/legacy-token-autogenerated-secret` or `authentication.k8s.io/legacy-token-manual-secret` audit annotation containing the name of the secret used. ([#118598](https://github.com/kubernetes/kubernetes/pull/118598), [@yuanchen8911](https://github.com/yuanchen8911)) [SIG Auth, Instrumentation and Testing]
- Volume_zone plugin will consider beta labels as GA labels during the scheduling process.Therefore, if the values of the labels are the same, PVs with beta labels can also be scheduled to nodes with GA labels. ([#118923](https://github.com/kubernetes/kubernetes/pull/118923), [@AxeZhan](https://github.com/AxeZhan)) [SIG Scheduling]

### Documentation

- Added descriptions and examples for the situation of using kubectl rollout restart without specifying a particular deployment. ([#120118](https://github.com/kubernetes/kubernetes/pull/120118), [@Ithrael](https://github.com/Ithrael)) [SIG CLI]

### Failing Test

- DRA: when the scheduler has to deallocate a claim after a node became unsuitable for a pod, it might have needed more attempts than really necessary. ([#120428](https://github.com/kubernetes/kubernetes/pull/120428), [@pohly](https://github.com/pohly)) [SIG Node and Scheduling]
- E2e framework: retrying after intermittent apiserver failures was fixed in WaitForPodsResponding ([#120559](https://github.com/kubernetes/kubernetes/pull/120559), [@pohly](https://github.com/pohly)) [SIG Testing]
- KCM specific args can be passed with `/cluster` script, without affecting CCM. New variable name: `KUBE_CONTROLLER_MANAGER_TEST_ARGS`. ([#120524](https://github.com/kubernetes/kubernetes/pull/120524), [@jprzychodzen](https://github.com/jprzychodzen)) [SIG Cloud Provider]
- This contains the modified windows kubeproxy testcases with mock implementation ([#120105](https://github.com/kubernetes/kubernetes/pull/120105), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]

### Bug or Regression

- Added a redundant process to remove tracking finalizers from Pods that belong to Jobs. The process kicks in after the control plane marks a Job as finished ([#119944](https://github.com/kubernetes/kubernetes/pull/119944), [@Sharpz7](https://github.com/Sharpz7)) [SIG Apps]
- Allow specifying ExternalTrafficPolicy for Services with ExternalIPs. ([#119150](https://github.com/kubernetes/kubernetes/pull/119150), [@tnqn](https://github.com/tnqn)) [SIG API Machinery, Apps, CLI, Cloud Provider, Network, Release and Testing]
- Exclude nodes from daemonset rolling update if the scheduling constraints are not met. This eliminates the problem of rolling update stuck of daemonset with tolerations. ([#119317](https://github.com/kubernetes/kubernetes/pull/119317), [@mochizuki875](https://github.com/mochizuki875)) [SIG Apps and Testing]
- Fix OpenAPI v3 not being cleaned up after deleting APIServices ([#120108](https://github.com/kubernetes/kubernetes/pull/120108), [@tnqn](https://github.com/tnqn)) [SIG API Machinery and Testing]
- Fix a 1.28 regression in scheduler: a pod with concurrent events could incorrectly get moved to the unschedulable queue where it could got stuck until the next periodic purging after 5 minutes if there was no other event for it. ([#120413](https://github.com/kubernetes/kubernetes/pull/120413), [@pohly](https://github.com/pohly)) [SIG Scheduling]
- Fix a bug in cronjob controller where already created jobs may be missing from the status. ([#120649](https://github.com/kubernetes/kubernetes/pull/120649), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps]
- Fix a concurrent map access in TopologyCache's `HasPopulatedHints` method. ([#118189](https://github.com/kubernetes/kubernetes/pull/118189), [@Miciah](https://github.com/Miciah)) [SIG Apps and Network]
- Fix kubectl events doesn't filter events by GroupVersion for resource with full name. ([#120119](https://github.com/kubernetes/kubernetes/pull/120119), [@Ithrael](https://github.com/Ithrael)) [SIG CLI and Testing]
- Fixed CEL estimated cost of `replace()` to handle a zero length replacement string correctly.
  Previously this would cause the estimated cost to be higher than it should be. ([#120097](https://github.com/kubernetes/kubernetes/pull/120097), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fixed a 1.26 regression scheduling bug by ensuring that preemption is skipped when a PreFilter plugin returns `UnschedulableAndUnresolvable` ([#119778](https://github.com/kubernetes/kubernetes/pull/119778), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Fixed a 1.27 scheduling regression that PostFilter plugin may not function if previous PreFilter plugins return Skip ([#119769](https://github.com/kubernetes/kubernetes/pull/119769), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fixed a 1.28 regression around restarting init containers in the right order relative to normal containers ([#120281](https://github.com/kubernetes/kubernetes/pull/120281), [@gjkim42](https://github.com/gjkim42)) [SIG Node and Testing]
- Fixed a regression in default 1.27 configurations in kube-apiserver: fixed the AggregatedDiscoveryEndpoint feature (beta in 1.27+) to successfully fetch discovery information from aggregated API servers that do not check `Accept` headers when serving the `/apis` endpoint ([#119870](https://github.com/kubernetes/kubernetes/pull/119870), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Fixed an issue where a CronJob could fail to clean up Jobs when the ResourceQuota for Jobs had been reached. ([#119776](https://github.com/kubernetes/kubernetes/pull/119776), [@ASverdlov](https://github.com/ASverdlov)) [SIG Apps]
- Fixes a 1.28 regression handling negative index json patches ([#120327](https://github.com/kubernetes/kubernetes/pull/120327), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node and Storage]
- Fixes a bug where Services using finalizers may hold onto ClusterIP and/or NodePort allocated resources for longer than expected if the finalizer is removed using the status subresource ([#120623](https://github.com/kubernetes/kubernetes/pull/120623), [@aojea](https://github.com/aojea)) [SIG Network and Testing]
- Fixes an issue where StatefulSet might not restart a pod after eviction or node failure. ([#120398](https://github.com/kubernetes/kubernetes/pull/120398), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska)) [SIG Apps]
- Fixes an issue with the garbagecollection controller registering duplicate event handlers if discovery requests fail. ([#117992](https://github.com/kubernetes/kubernetes/pull/117992), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Apps]
- Fixes the bug when images pinned by the container runtime can be garbage collected by kubelet ([#119986](https://github.com/kubernetes/kubernetes/pull/119986), [@ruiwen-zhao](https://github.com/ruiwen-zhao)) [SIG Node]
- Fixing issue with incremental id generation for loadbalancer and endpoint in Kubeproxy mock test framework. ([#120723](https://github.com/kubernetes/kubernetes/pull/120723), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- If a watch with the `progressNotify` option set is to be created, and the registry hasn't provided a `newFunc`, return an error. ([#120212](https://github.com/kubernetes/kubernetes/pull/120212), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Improved handling of jsonpath expressions for kubectl wait --for. It is now possible to use simple filter expressions which match on a field's content. ([#118748](https://github.com/kubernetes/kubernetes/pull/118748), [@andreaskaris](https://github.com/andreaskaris)) [SIG CLI and Testing]
- Incorporating feedback on PR #119341 ([#120087](https://github.com/kubernetes/kubernetes/pull/120087), [@divyasri537](https://github.com/divyasri537)) [SIG API Machinery]
- Kubeadm: Use universal deserializer to decode static pod. ([#120549](https://github.com/kubernetes/kubernetes/pull/120549), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: fix nil pointer when etcd member is already removed ([#119753](https://github.com/kubernetes/kubernetes/pull/119753), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that `--image-repository` flag is missing for some init phase sub-commands ([#120072](https://github.com/kubernetes/kubernetes/pull/120072), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: improve the logic that checks whether a systemd service exists. ([#120514](https://github.com/kubernetes/kubernetes/pull/120514), [@fengxsong](https://github.com/fengxsong)) [SIG Cluster Lifecycle]
- Kubeadm: print the default component configs for `reset` and `join` is now not supported ([#119346](https://github.com/kubernetes/kubernetes/pull/119346), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubeadm: remove 'system:masters' organization from etcd/healthcheck-client certificate. ([#119859](https://github.com/kubernetes/kubernetes/pull/119859), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl prune v2: Switch annotation from `contains-group-resources` to `contains-group-kinds`,
  because this is what we defined in the KEP and is clearer to end-users.  Although the functionality is
  in alpha, we will recognize the prior annotation; this migration support will be removed in beta/GA. ([#118942](https://github.com/kubernetes/kubernetes/pull/118942), [@justinsb](https://github.com/justinsb)) [SIG CLI]
- Kubectl will not print events if --show-events=false argument is passed to describe PVC subcommand. ([#120380](https://github.com/kubernetes/kubernetes/pull/120380), [@MaGaroo](https://github.com/MaGaroo)) [SIG CLI]
- More accurate requeueing in scheduling queue for Pods rejected by the temporal failure (e.g., temporal failure on kube-apiserver.) ([#119105](https://github.com/kubernetes/kubernetes/pull/119105), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- No-op and GC related updates to cluster trust bundles no longer require attest authorization when the ClusterTrustBundleAttest plugin is enabled. ([#120779](https://github.com/kubernetes/kubernetes/pull/120779), [@enj](https://github.com/enj)) [SIG Auth]
- Reintroduce resourcequota.NewMonitor constructor for other consumers ([#120777](https://github.com/kubernetes/kubernetes/pull/120777), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- Scheduler: Fix field apiVersion is missing from events reported from taint manager ([#114095](https://github.com/kubernetes/kubernetes/pull/114095), [@aimuz](https://github.com/aimuz)) [SIG Apps, Node and Scheduling]
- Service Controller: update load balancer hosts after node's ProviderID is updated ([#120492](https://github.com/kubernetes/kubernetes/pull/120492), [@cezarygerard](https://github.com/cezarygerard)) [SIG Cloud Provider and Network]
- Setting the `status.loadBalancer` of a Service whose `spec.type` is not `"LoadBalancer"` was previously allowed, but any update to the `metadata` or `spec` would wipe that field. Setting this field is no longer permitted unless `spec.type` is  `"LoadBalancer"`.  In the very unlikely event that this has unexpected impact, you can enable the `AllowServiceLBStatusOnNonLB` feature gate, which will restore the previous behavior.  If you do need to set this, please file an issue with the Kubernetes project to help contributors understand why you need it. ([#119789](https://github.com/kubernetes/kubernetes/pull/119789), [@thockin](https://github.com/thockin)) [SIG Apps and Testing]
- Sometimes, the scheduler incorrectly placed a pod in the "unschedulable" queue instead of the "backoff" queue. This happened when some plugin previously declared the pod as "unschedulable" and then in a later attempt encounters some other error. Scheduling of that pod then got delayed by up to five minutes, after which periodic flushing moved the pod back into the "active" queue. ([#120334](https://github.com/kubernetes/kubernetes/pull/120334), [@pohly](https://github.com/pohly)) [SIG Scheduling]
- The `--bind-address` parameter in kube-proxy is misleading, no port is opened with this address. Instead it is translated internally to "nodeIP". The nodeIPs for both families are now taken from the Node object if `--bind-address` is unspecified or set to the "any" address (0.0.0.0 or ::). It is recommended to leave `--bind-address` unspecified, and in particular avoid to set it to localhost (127.0.0.1 or ::1) ([#119525](https://github.com/kubernetes/kubernetes/pull/119525), [@uablrek](https://github.com/uablrek)) [SIG Network and Scalability]

### Other (Cleanup or Flake)

- Add context to "caches populated" log messages. ([#119796](https://github.com/kubernetes/kubernetes/pull/119796), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Add download the cni binary for the corresponding arch in local-up-cluster.sh ([#120312](https://github.com/kubernetes/kubernetes/pull/120312), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Network and Node]
- Changes behavior of kube-proxy by allowing to set sysctl values lower than the existing one. ([#120448](https://github.com/kubernetes/kubernetes/pull/120448), [@aroradaman](https://github.com/aroradaman)) [SIG Network]
- Clean up kube-apiserver http logs for impersonated requests. ([#119795](https://github.com/kubernetes/kubernetes/pull/119795), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Dynamic resource allocation: avoid creating a new gRPC connection for every call of prepare/unprepare resource(s) ([#118619](https://github.com/kubernetes/kubernetes/pull/118619), [@TommyStarK](https://github.com/TommyStarK)) [SIG Node]
- Fixes an issue where the vsphere cloud provider will not trust a certificate if:
  - The issuer of the certificate is unknown (x509.UnknownAuthorityError)
  - The requested name does not match the set of authorized names (x509.HostnameError)
  - The error surfaced after attempting a connection contains one of the substrings: "certificate is not trusted" or "certificate signed by unknown authority" ([#120736](https://github.com/kubernetes/kubernetes/pull/120736), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG Architecture and Cloud Provider]
- Fixes bug where Adding GroupVersion log line is constantly repeated without any group version changes ([#119825](https://github.com/kubernetes/kubernetes/pull/119825), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Generated ResourceClaim names are now more readable because of an additional hyphen before the random suffix (`<pod name>-<claim name>-<random suffix>` ). ([#120336](https://github.com/kubernetes/kubernetes/pull/120336), [@pohly](https://github.com/pohly)) [SIG Apps and Node]
- Improve memory usage of kube-controller-manager by dropping the `.metadata.managedFields` field that kube-controller-manager doesn't require. ([#118455](https://github.com/kubernetes/kubernetes/pull/118455), [@linxiulei](https://github.com/linxiulei)) [SIG API Machinery and Cloud Provider]
- Kubeadm: remove 'system:masters' organization from apiserver-etcd-client certificate ([#120521](https://github.com/kubernetes/kubernetes/pull/120521), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: updated warning message when swap space is detected. When swap is active on Linux, kubeadm explains that swap is supported for cgroup v2 only and is beta but disabled by default. ([#120198](https://github.com/kubernetes/kubernetes/pull/120198), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Makefile and scripts now respect GOTOOLCHAIN and otherwise ensure ./.go-version is used ([#120279](https://github.com/kubernetes/kubernetes/pull/120279), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Optimized NodeUnschedulable Filter to avoid unnecessary calculations ([#119399](https://github.com/kubernetes/kubernetes/pull/119399), [@wackxu](https://github.com/wackxu)) [SIG Scheduling]
- Previously, the pod name and namespace were eliminated in the event log message. This PR attempts to add the preemptor pod UID in the preemption event message logs for easier debugging and safer transparency. ([#119971](https://github.com/kubernetes/kubernetes/pull/119971), [@kwakubiney](https://github.com/kwakubiney)) [SIG Scheduling]
- Promote to conformance a test that verify that Services only forward traffic on the port and protocol specified. ([#120069](https://github.com/kubernetes/kubernetes/pull/120069), [@aojea](https://github.com/aojea)) [SIG Architecture, Network and Testing]
- Remove ephemeral container legacy server support for the server versions prior to 1.22 ([#119537](https://github.com/kubernetes/kubernetes/pull/119537), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Scheduler: handling of unschedulable pods because a ResourceClass is missing is a bit more efficient and no longer relies on periodic retries ([#120213](https://github.com/kubernetes/kubernetes/pull/120213), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Set the resolution for the job_controller_job_sync_duration_seconds metric from 4ms to 1min ([#120577](https://github.com/kubernetes/kubernetes/pull/120577), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Instrumentation]
- Statefulset should wait for new replicas in tests when removing .start.ordinal ([#119761](https://github.com/kubernetes/kubernetes/pull/119761), [@soltysh](https://github.com/soltysh)) [SIG Apps and Testing]
- The `horizontalpodautoscaling` and `clusterrole-aggregation` controllers now assume the `autoscaling/v1` and `rbac.authorization.k8s.io/v1` APIs are available. If you disable those APIs and do not want to run those controllers, exclude them by passing `--controllers=-horizontalpodautoscaling` or `--controllers=-clusterrole-aggregation` to `kube-controller-manager`. ([#117977](https://github.com/kubernetes/kubernetes/pull/117977), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Cloud Provider]
- The metrics controlled by the ComponentSLIs feature-gate and served at /metrics/slis are now GA and unconditionally enabled. The feature-gate will be removed in 1.31. ([#120574](https://github.com/kubernetes/kubernetes/pull/120574), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery, Architecture, Cloud Provider, Instrumentation, Network, Node and Scheduling]
- Updated CNI plugins to v1.3.0. ([#119969](https://github.com/kubernetes/kubernetes/pull/119969), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.28.0. ([#119933](https://github.com/kubernetes/kubernetes/pull/119933), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Updated distroless-iptables to use registry.k8s.io/build-image/distroless-iptables:v0.3.1 ([#120352](https://github.com/kubernetes/kubernetes/pull/120352), [@saschagrunert](https://github.com/saschagrunert)) [SIG Release and Testing]
- Upgrade coredns to v1.11.1 ([#120116](https://github.com/kubernetes/kubernetes/pull/120116), [@tukwila](https://github.com/tukwila)) [SIG Cloud Provider and Cluster Lifecycle]
- ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding objects are persisted in etcd using the v1beta1 version. Remove alpha objects or disable the alpha ValidatingAdmissionPolicy feature in a 1.27 server before upgrading to a 1.28 server with the beta feature and API enabled. ([#120018](https://github.com/kubernetes/kubernetes/pull/120018), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Yes, kubectl will not support the "/swagger-2.0.0.pb-v1" endpoint that has been long deprecated ([#119410](https://github.com/kubernetes/kubernetes/pull/119410), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]

## Dependencies

### Added
- github.com/distribution/reference: [v0.5.0](https://github.com/distribution/reference/tree/v0.5.0)

### Changed
- github.com/coredns/corefile-migration: [v1.0.20 → v1.0.21](https://github.com/coredns/corefile-migration/compare/v1.0.20...v1.0.21)
- github.com/docker/distribution: [v2.8.2+incompatible → v2.8.1+incompatible](https://github.com/docker/distribution/compare/v2.8.2...v2.8.1)
- github.com/evanphx/json-patch: [v5.6.0+incompatible → v4.12.0+incompatible](https://github.com/evanphx/json-patch/compare/v5.6.0...v4.12.0)
- github.com/google/cel-go: [v0.16.0 → v0.17.6](https://github.com/google/cel-go/compare/v0.16.0...v0.17.6)
- github.com/gorilla/websocket: [v1.4.2 → v1.5.0](https://github.com/gorilla/websocket/compare/v1.4.2...v1.5.0)
- github.com/opencontainers/runc: [v1.1.7 → v1.1.9](https://github.com/opencontainers/runc/compare/v1.1.7...v1.1.9)
- github.com/opencontainers/selinux: [v1.10.0 → v1.11.0](https://github.com/opencontainers/selinux/compare/v1.10.0...v1.11.0)
- github.com/vmware/govmomi: [v0.30.0 → v0.30.6](https://github.com/vmware/govmomi/compare/v0.30.0...v0.30.6)
- google.golang.org/protobuf: v1.30.0 → v1.31.0
- k8s.io/gengo: c0856e2 → 9cce18d
- k8s.io/kube-openapi: 2695361 → d090da1
- k8s.io/utils: d93618c → 3b25d92
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.1.2 → v0.28.0
- sigs.k8s.io/structured-merge-diff/v4: v4.2.3 → v4.3.0

### Removed
_Nothing has changed._