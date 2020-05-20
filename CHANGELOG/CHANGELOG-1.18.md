<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.18.3](#v1183)
  - [Downloads for v1.18.3](#downloads-for-v1183)
    - [Source Code](#source-code)
    - [Client binaries](#client-binaries)
    - [Server binaries](#server-binaries)
    - [Node binaries](#node-binaries)
  - [Changelog since v1.18.2](#changelog-since-v1182)
  - [Changes by Kind](#changes-by-kind)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.18.2](#v1182)
  - [Downloads for v1.18.2](#downloads-for-v1182)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.18.1](#changelog-since-v1181)
  - [Changes by Kind](#changes-by-kind-1)
    - [Bug or Regression](#bug-or-regression-1)
- [v1.18.1](#v1181)
  - [Downloads for v1.18.1](#downloads-for-v1181)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.18.0](#changelog-since-v1180)
  - [Changes by Kind](#changes-by-kind-2)
    - [Feature](#feature)
    - [Other (Bug, Cleanup or Flake)](#other-bug-cleanup-or-flake)
- [v1.18.0](#v1180)
  - [Downloads for v1.18.0](#downloads-for-v1180)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.17.0](#changelog-since-v1170)
  - [What’s New (Major Themes)](#what’s-new-major-themes)
    - [Kubernetes Topology Manager Moves to Beta - Align Up!](#kubernetes-topology-manager-moves-to-beta---align-up)
    - [Serverside Apply - Beta 2](#serverside-apply---beta-2)
    - [Extending Ingress with and replacing a deprecated annotation with IngressClass](#extending-ingress-with-and-replacing-a-deprecated-annotation-with-ingressclass)
    - [SIG CLI introduces kubectl debug](#sig-cli-introduces-kubectl-debug)
    - [Introducing Windows CSI support alpha for Kubernetes](#introducing-windows-csi-support-alpha-for-kubernetes)
    - [Other notable announcements](#other-notable-announcements)
  - [Known Issues](#known-issues)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
      - [kube-apiserver:](#kube-apiserver:)
      - [kubelet:](#kubelet:)
      - [kubectl:](#kubectl:)
      - [client-go:](#client-go:)
    - [Changes by Kind](#changes-by-kind-3)
      - [Deprecation](#deprecation)
        - [kube-apiserver:](#kube-apiserver:-1)
        - [kube-controller-manager:](#kube-controller-manager:)
        - [kubelet:](#kubelet:-1)
        - [kube-proxy:](#kube-proxy:)
        - [kubeadm:](#kubeadm:)
        - [kubectl:](#kubectl:-1)
        - [add-ons:](#add-ons:)
        - [kube-scheduler:](#kube-scheduler:)
        - [Other deprecations:](#other-deprecations:)
      - [API Change](#api-change)
        - [New API types/versions:](#new-api-types/versions:)
        - [New API fields:](#new-api-fields:)
        - [Other API changes:](#other-api-changes:)
        - [Configuration file changes:](#configuration-file-changes:)
        - [kube-apiserver:](#kube-apiserver:-2)
        - [kube-scheduler:](#kube-scheduler:-1)
        - [kube-proxy:](#kube-proxy:-1)
        - [Features graduated to beta:](#features-graduated-to-beta:)
        - [Features graduated to GA:](#features-graduated-to-ga:)
      - [Feature](#feature-1)
        - [Metrics:](#metrics:)
      - [Other (Bug, Cleanup or Flake)](#other-bug-cleanup-or-flake-1)
    - [Dependencies](#dependencies-1)
- [v1.18.0-rc.1](#v1180-rc1)
  - [Downloads for v1.18.0-rc.1](#downloads-for-v1180-rc1)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.18.0-beta.2](#changelog-since-v1180-beta2)
  - [Changes by Kind](#changes-by-kind-4)
    - [API Change](#api-change-1)
    - [Other (Bug, Cleanup or Flake)](#other-bug-cleanup-or-flake-2)
- [v1.18.0-beta.2](#v1180-beta2)
  - [Downloads for v1.18.0-beta.2](#downloads-for-v1180-beta2)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.18.0-beta.1](#changelog-since-v1180-beta1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-5)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Documentation](#documentation)
    - [Other (Bug, Cleanup or Flake)](#other-bug-cleanup-or-flake-3)
- [v1.18.0-beta.1](#v1180-beta1)
  - [Downloads for v1.18.0-beta.1](#downloads-for-v1180-beta1)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.18.0-beta.0](#changelog-since-v1180-beta0)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-6)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-3)
- [v1.18.0-alpha.1](#v1180-alpha1)
  - [Downloads for v1.18.0-alpha.1](#downloads-for-v1180-alpha1)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
  - [Changelog since v1.17.0](#changelog-since-v1170-1)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)

<!-- END MUNGE: GENERATED_TOC -->

# v1.18.3


## Downloads for v1.18.3

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes.tar.gz) | 7d511c960f766f76bc087c00d706dc78ed403f661ea62ea6a2e84b9a0498826c0186f8705d18e1101ce148eecf7046f3e96e0a64aff7698a0976414a56056d4d
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-src.tar.gz) | 93b83acf5d15cab94e1d2866b2613d1aeed67c00a9eed064988c3bc4c700e34bd854fffac730c5ed6d8e138f15ce7750c17952f33bd4918771c7da358fbf5b53

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-darwin-386.tar.gz) | cf6a22a453b88de6be0e09ad67e8bb3e364b702a86c9ba911540d4f4b2ae0872d3802f814cd471ef4ec0b2822b071489d59e21ece01f38fa567d3afd8718158f
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-darwin-amd64.tar.gz) | bd3c15726d44d083f48dbc1af0f55f2d2d0c82ad020ed583bf05460a4fc9073bdd0395c2e0e4bfc74b705f4427bfe76bee0c77eb4aea83247b6627fc15d8d3f9
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-linux-386.tar.gz) | 6cb3d18086e275b78c3019a51de5795f5d112482ea6dc99a58e3985269e948e054a5e38457f837add606b83d86fb1c5b9044903f9dea964d9ea84c564a8ca6e5
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-linux-amd64.tar.gz) | cfcb89a706eb8ddc7aa8225e3f0eb76a0d973faa1c82b1bec0a457cd8b44b7bd5c154b7ed1f7cdabfdee84af8a33fd7fff83970a6ee5a97d661a806b69da968b
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-linux-arm.tar.gz) | adbb0383ab50358e479438831168b6f3187a7cafcad84e8c22ff2ec52300be643bf535ffbe0c6ffc971120afb56703c206a64459db7b485ee67dd05d84393f5e
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-linux-arm64.tar.gz) | 32f5e6cc5a811f941faaa92667d236bb08bc245a103a2ab555569a5bf1bfd1926f30ce08635273fbea414f92c8764f8077b3e3be5ea52e90be4ed3ded9253452
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-linux-ppc64le.tar.gz) | b0b2ade932e17aa4b88b147fcb6aeace81c65e795202e27c780e270a86f50fd7f863f41558b0fc37ece1196fa39047baf47568416ee6faed7946593ad35fba03
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-linux-s390x.tar.gz) | db49113c3e5d727d6c66b17a0a5a3f7d383b7179630fb5680d278f34b35e4f3d5a1086489858a90a5417dafc42222d23ff2aacba7b7d571d3b2d9ced460877de
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-windows-386.tar.gz) | d2a8e6f6e93a3ce6af473372de1c52e039d14a443d93537001e1bc5e7b237768a25a9a1d891a40d50142c23df1e7d94f767d11adf949439e77328c7aafbc7a23
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-client-windows-amd64.tar.gz) | 5f2739b862fbbab9f847b61f9373021b92c4d9188ff7f534125dc48d2d1e6ed51bd7bc5c987f26db3bf1d1f05a4c0f4c0ff7a0836a4ca14e56d2b12f406f1a72

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-server-linux-amd64.tar.gz) | 5561483d796b124b8fe0e1cf5778ea03fec1e244ebc29f4b1b6c5ac93ab6bd10808d05da81b5f26426d51a3784c93ddf8b375ff971a78aebfd0ec7acac161365
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-server-linux-arm.tar.gz) | 5e0e026fb93ac5452d1ddeb5fc016dbd44276d2340088ef59916bdd264b43ab02889beb645b6b8d9e5ee02bf2fe827327478b158f80b152c073b9ecaaba6fd0e
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-server-linux-arm64.tar.gz) | eb0b72f79e9d0c717995f7e52d24646daa8cbaf0e1502c0b27a15acebcfa5d61495bee49cc6ba4eb4ed8d4e1d9aae29170a86c70240da56f8fb9360dabb9ec7b
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-server-linux-ppc64le.tar.gz) | 52ef224a68d3ea50f320ca43b2ec98fedc07431b05db6fb00556b870bb8a533aa1ceb5f5ad90ea3d41f611045014a387ae84de5df8c51c6fbf8fa1e0d8c07d31
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-server-linux-s390x.tar.gz) | eb4581d2419734c4835ebd2a91a40fa7e1180c8b8ff4088c9d1995c11787b7d6b7cb26cedfdce8c53affc648b106fe852ed53f165142f1089c27f9368570025d

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-node-linux-amd64.tar.gz) | 1027a6fccadd320f123894dc624db31539333deef0b3d51b4bd3efc9214f2d74a0a50c53dc28e5801426d3c9556f82f4e06042c824151fea9112df795976d158
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-node-linux-arm.tar.gz) | 9dfd609692372660152c6fe6d08be082b0d20d4c70546d722ce5aa5565cc6d810bb0cdac6b98d9a9374382b9dcf8819e4e1d263481791e044cfef5d3980d0b13
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-node-linux-arm64.tar.gz) | d0b0a8ac1f448df7c3bb2254000e0ca8567fafe1fd4e680c75a6c8d40dcc9d4b9ae34648532aef50a953be99d41d62464d704b516c33745fc169337684e7433b
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-node-linux-ppc64le.tar.gz) | b5f3a0e3b2b26d3ac5b8be808355233787c1d3663268da88096351b39b6ff6e58befff97dab109b6324ad050fe2a361ac6b35dbdab7f2dd85dc013d881baecf9
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-node-linux-s390x.tar.gz) | 52c9cc8a09c5d5c9150dc023b759ebe77632121a54bf0936b96f4148b7e421964f39bdad4fcd1002f453c012a03d928888cc04520a10a96ea45d798a7cd2ca7b
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.3/kubernetes-node-windows-amd64.tar.gz) | f83802db06a86edd9ade8e737ed0e8a11ebeeaa69e102f9550b90f0d8a724e7864f6531f7f500ff70d4202168a774c5b328fea1ec0a95b9e275a0233a852f04f

## Changelog since v1.18.2

## Changes by Kind

### Bug or Regression
 - An issue preventing GCP cloud-controller-manager running out-of-cluster to initialize new Nodes is now fixed. ([#90057](https://github.com/kubernetes/kubernetes/pull/90057), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Apps and Cloud Provider]
 - Avoid unnecessary scheduling churn when annotations are updated while Pods are being scheduled. ([#90373](https://github.com/kubernetes/kubernetes/pull/90373), [@fabiokung](https://github.com/fabiokung)) [SIG Scheduling]
 - Base-images: Update to kube-cross:v1.13.9-5 ([#90964](https://github.com/kubernetes/kubernetes/pull/90964), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
 - CSINode initialization does not crash kubelet on startup when APIServer is not reachable or kubelet has not the right credentials yet. ([#89589](https://github.com/kubernetes/kubernetes/pull/89589), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
 - Fix IPVS compatiblity issue with older kernels (< 3.18) where the netlink address family attribute is not set ([#90678](https://github.com/kubernetes/kubernetes/pull/90678), [@andrewsykim](https://github.com/andrewsykim)) [SIG Cluster Lifecycle, Network and Testing]
 - Fix flaws in Azure CSI translation ([#90324](https://github.com/kubernetes/kubernetes/pull/90324), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
 - Fix: Init containers are now considered for the calculation of resource requests when scheduling ([#90378](https://github.com/kubernetes/kubernetes/pull/90378), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
 - Fix: azure disk dangling attach issue which would cause API throttling ([#90749](https://github.com/kubernetes/kubernetes/pull/90749), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
 - Fix: support removal of nodes backed by deleted non VMSS instances on Azure ([#91184](https://github.com/kubernetes/kubernetes/pull/91184), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
 - Fixed a 1.18 regression in wait.Forever that skips the backoff period on the first repeat ([#90476](https://github.com/kubernetes/kubernetes/pull/90476), [@zhan849](https://github.com/zhan849)) [SIG API Machinery]
 - Fixed a regression running kubectl commands with  --local or --dry-run flags when no kubeconfig file is present ([#90243](https://github.com/kubernetes/kubernetes/pull/90243), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI and Testing]
 - Fixes a bug defining a default value for a replicas field in a custom resource definition that has the scale subresource enabled ([#90019](https://github.com/kubernetes/kubernetes/pull/90019), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
 - Fixes a regression in 1.17 that dropped cache-control headers on API requests ([#90468](https://github.com/kubernetes/kubernetes/pull/90468), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
 - Kubeadm: increase robustness for "kubeadm join" when adding etcd members on slower setups ([#90645](https://github.com/kubernetes/kubernetes/pull/90645), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
 - Provides a fix to allow a cluster in a private Azure cloud to authenticate to ACR in the same cloud. ([#90425](https://github.com/kubernetes/kubernetes/pull/90425), [@DavidParks8](https://github.com/DavidParks8)) [SIG Cloud Provider]
 - Scheduling failures due to no nodes available are now reported as unschedulable under ```schedule_attempts_total``` metric. ([#90989](https://github.com/kubernetes/kubernetes/pull/90989), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]

### Other (Cleanup or Flake)
 - base-images: Use debian-base:v2.1.0 (includes CVE fixes)
  - base-images: Use debian-iptables:v12.1.0 (includes CVE fixes) ([#90863](https://github.com/kubernetes/kubernetes/pull/90863), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Cluster Lifecycle and Release]

## Dependencies

### Added
_Nothing has changed._

### Changed
- k8s.io/kube-openapi: bf4fb3b → 61e04a5

### Removed
- github.com/docker/libnetwork: [c8a5fca](https://github.com/docker/libnetwork/tree/c8a5fca)



# v1.18.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.2

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes.tar.gz) | `2f8e853bd59731410259d5357d9969425fbbbea378bbe6cdd0f7a9ddf5c25924838300924b03ec15d6b9030be86bea9d26bb9b63078bf2c150b0bbc0859419d7`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-src.tar.gz) | `0915b658c53b9bad1b3913470cb6728bc51fd49e8ac7778d4653c7271642d56a51ae83e58b9a1829a8df8970e73411f02c5ab277f8a9ba4befc4ba933800a9c5`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-darwin-386.tar.gz) | `0a0c94fe16819eb16ca7ef0110a2a45ad5368a5cb326ca48e1d72ef56488c5c273fcfa15e9704492dfb3188447800cf109bf434b6d646a2c01c833eccaa7ebbe`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-darwin-amd64.tar.gz) | `46a056b3bf9936498c1bbb78ca6d882c17271723676ec53409fe6fd67c7f8a9cb0ab00e286f8ab2231216fabbcfec72f943c29827068ab66d6029f585faccfcb`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-linux-386.tar.gz) | `58f137f3d13b213a153e7589d82040d5f1aee525368de974c134494c14d0f88526e4b1db9022dd728d47fe13ff1c4c97fba94e3b2ebc746ec537bebf41817d53`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-linux-amd64.tar.gz) | `ed36f49e19d8e0a98add7f10f981feda8e59d32a8cb41a3ac6abdfb2491b3b5b3b6e0b00087525aa8473ed07c0e8a171ad43f311ab041dcc40f72b36fa78af95`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-linux-arm.tar.gz) | `ae3b7a8f85d2f262b0f24d277602034cd6657aa0a0467768b87c379b821963c10e35cff8131f666038feb2a2e543725c0025ed84a7a254c6e96036b911988530`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-linux-arm64.tar.gz) | `54b10261c354e99d3eeee862461f0c3f99ff0e3b603230da7a48e182fd5890e438ff5bc8daafc74a3be9411a873d9ed611cea99c038fff53aa9e57d8d7140662`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-linux-ppc64le.tar.gz) | `b9694a0cf9e42bc9299d923de79e61ec52419a1889605cfd2eb5e6f9277191a0a1c6b7ceaf251735e576ad4aa73465c1c6a48b977243e45b6449fe7017dad18b`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-linux-s390x.tar.gz) | `144861c7cfc28b63da11de4b847d68bb4a984b5eeb54ccbccf998bd87e0e2832d38d3610056cf2e5d3a59535a0f8213bebd9369f0f6c1fd947afad41fdcc8837`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-windows-386.tar.gz) | `3fa6e6fdf88b7f9ae7dc8f95526977aea6e2fe65fdbb988c2ea40d160ba30342453ee9e0ce022c69d8635405ad92c6672d775e8e477cf3c937c1bbb7217fb279`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-client-windows-amd64.tar.gz) | `733887310c94e70fb33c6fbea9c5e7d4a74b4c2402735ed7856eb2e009bb0ed2aab3abeb1d13b46681e876df6901bd2be6547a25061a8e8df442301179b82fc9`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-server-linux-amd64.tar.gz) | `f808e85a5e6f8dfed18ee3479691be8283c13c787ad5abb1a06f1c84aa7e7894af9028c6edcc4cdfe2cccf58e9c8394a6958facc92364d388a62ebf6aa9db2e8`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-server-linux-arm.tar.gz) | `7ec6d47cda5f8f2cafaa82ac1179dc181d93562d1a2ad7687dca5dba8737498c0488275eb0ba33018c48e684550fa8612d03726251ed24fdebc893286528a401`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-server-linux-arm64.tar.gz) | `f5341be0c84cbf383662ed333bb2f9a4b83f80b6ebe77526ed2a407e3cd566f643be030c367606da4eb563523119f7bd8a9a7173b91cc5466602b5f9e1c34921`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-server-linux-ppc64le.tar.gz) | `1c861320ddd63c9731781079fb00d9b0c80befe9b98103056f3abdd214cdd4974b1d79196491fb7c953d0126343fb6d7b9a6a7ca1684b8d8c5db686e1647106b`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-server-linux-s390x.tar.gz) | `5e57f536844d606873412a5ca46e85c4a6deae5e5dc415b3fbd0b20a58750cd0360265bc237102a8050db795c1293d6d7530cb5183e0ca25bc8e93f3f5fc650c`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-node-linux-amd64.tar.gz) | `b342dbb9fce1c2667ed255e0b7457063e7f4827a74d4c946087bb471144a552e93e17e624075273fd72b1788fc9219ae46a8d8b1c247b2f26320e932143fef1b`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-node-linux-arm.tar.gz) | `d74d6b9a0c05623fb5f3b7423517c3a8c03f6fd18525554cae5704cadc3676d70d42d0537de70f7b7619e31640a8c510b4321e0eeada49300d9c92cac2a9217a`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-node-linux-arm64.tar.gz) | `d80716155df8ee997b4d81573ab713a04f64e91ec0e7c6c77af2e0031bbbe1f250cc463a7788e548217f6e7071caf458239869f2863737f3007a8cf79abff9cc`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-node-linux-ppc64le.tar.gz) | `89fe1dbbbefe36169232b46b564fc46b96cf6987bca8c1f9c61475d07a771c65f63cdf4c66168e9c515ec862f65b83d5a7668425e956a109e1aa60e0988d4a57`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-node-linux-s390x.tar.gz) | `734e11c10c4e8dea9931ce0e832dac8495808acb7940ca2be4a13cedbea53b537973320e80b9f021c37c10c9510dcf328b0f975c40afebc301ea3d3937a3c36a`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.2/kubernetes-node-windows-amd64.tar.gz) | `f121f7893c102ecd491189077ccbddd7aa0625cf2bfe855a7be00cfe615e6d397928cb5448092993626f33b216e89ac11bd0c09255847e1b6ba9a54a933eee53`

## Changelog since v1.18.1

## Changes by Kind

### Bug or Regression

- Client-go: resolves an issue with informers falling back to full list requests when timeouts are encountered, rather than re-establishing a watch. ([#89975](https://github.com/kubernetes/kubernetes/pull/89975), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Fix scheduler crash when removing node before its pods ([#89908](https://github.com/kubernetes/kubernetes/pull/89908), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Fixes conversion error for HorizontalPodAutoscaler objects with invalid annotations ([#89965](https://github.com/kubernetes/kubernetes/pull/89965), [@liggitt](https://github.com/liggitt)) [SIG Autoscaling]
- Fixes kubectl apply/prune in namespace other than default. ([#90016](https://github.com/kubernetes/kubernetes/pull/90016), [@seans3](https://github.com/seans3)) [SIG CLI and Testing]
- For GCE cluster provider, fix bug of not being able to create internal type load balancer for clusters with more than 1000 nodes in a single zone. ([#89902](https://github.com/kubernetes/kubernetes/pull/89902), [@wojtek-t](https://github.com/wojtek-t)) [SIG Cloud Provider, Network and Scalability]
- Restores priority of static control plane pods in the cluster/gce/manifests control-plane manifests ([#89970](https://github.com/kubernetes/kubernetes/pull/89970), [@liggitt](https://github.com/liggitt)) [SIG Cluster Lifecycle and Node]


# v1.18.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.1

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes.tar.gz) | `460dcc0b27fdfd9b4a574287708c0fef22224bd4c1bc777654a69a76c7dafb37e6a37b028aeaa8d79e202c2265fe4b322af6a95515cd438e44de7d55dac176b3`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-src.tar.gz) | `adc6b3ccc9792794b97d2c8c7e5d582ac92aedfa83bb9cdfb782057ce4e80985e940eeb8f3e943b90919927fe8ce65863077e526b56e7675ee1d5d66760d08b6`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-darwin-386.tar.gz) | `fe7c496778172012504839175c48c69337afc7341c8c71d2858bf9319a2bb4673abeb95b1415ae4abaa2364f24b410edaada963beeb614b361f6d412bf4c9352`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-darwin-amd64.tar.gz) | `a62a894ae001cb3f245595488a46c8c8c5c52d15eb9eefc7b458df6c93399e58eba52d1ca0dc8df1ee74ea75ecea3ab4853eea4bf830dfb89be3b4d6a5d0d83c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-linux-386.tar.gz) | `4cd898e86510f17d0a34c8721f942d81bdaafbf4d6513efde2710aad7dc44e89b6d986491c444f57ad9f183ac376031ecdb8e6dbee7dea76f7e4df116fb3998a`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-linux-amd64.tar.gz) | `37e664e40bb31765572215cf262a5c9bbc7748d158d0db58dbec2d5e593b54d71586af77296eda1cec2a2230b1d27260c51f6410b83afeeafc3c5354c308b4c4`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-linux-arm.tar.gz) | `196977d4a09046abb168ea4c6cde261a90226cd391d74877ce1d9907bc8ba670d0365311980422493125a2cde8648ece0035ff1af6d9507975428129de603c83`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-linux-arm64.tar.gz) | `675f27c170eb888f08db834f03b8123d19f0f2dd357c694c6c1cae59067c8d6b0b2db82b9cefc105dd16079ef6f7261e03fe9a73089c03dd3d53b1d68ed1cf68`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-linux-ppc64le.tar.gz) | `dd317cf29ed7cfa664a0f88651273565ca831138994cb37d8d53f5ba3993a6da529a377ded98d65c31450b31e1647fb87bc708f2aec6b2ef6a2bea9c73fb73fb`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-linux-s390x.tar.gz) | `57db3fcc952ad57d94f3b92022c1881b3852b321535501af7b2dfca9eb0acde03a34c5863aad6fe304fc4aaa922ff538d5dae7bc8c375a05956f892c265fbaf1`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-windows-386.tar.gz) | `ad52ae356e9d0156bdaa5ed4c77cd0226610fd715093e2caf7466c1bf87bb9bb2f21a48e9676f265c52de301bc416ce17cda5ac9fd7d379323c977c5f07ee9cd`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-client-windows-amd64.tar.gz) | `efe66bb5ae58e06c7787b98fc69e191502dadecf719636788f25bff7bd0e50d7d170c5e729611406c304afd159b33b0048c436e823276b0fc9d4c07904ba7974`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-server-linux-amd64.tar.gz) | `2183d1fcfca1370f75146797100801d7fbfec97789d1ca5eef4aff79bf66e01869ddaac51ab76abc073026682fe4d7ee658ffed99b5324ad285019c8dabcfea6`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-server-linux-arm.tar.gz) | `8ca77be8dd999e0a31bb9de597f383628941b7d6537cec19ce3a77c8f4fc537c649e9ce0bd43ec4ecb387d224654804f6f4682c80c26ba7e97cab6aff5b57e21`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-server-linux-arm64.tar.gz) | `b1eacba21d8740bba785f94b66aea1fb9e4529bea9740d938cd52409acc970f0a93ec5c857059d3d3f555f9eb6cd4f799c42266ad493676b3fed7f1deaf5a878`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-server-linux-ppc64le.tar.gz) | `dc8426bd333aa2fe703003356a6237df760c6753c142e6fea28cbf13656e53eb26d99b862000813ba94fe2041fc073271217879b7036546a7bd4b848aa569f6b`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-server-linux-s390x.tar.gz) | `2398638d5724627573326b6820cb268d30d47f18afc913d367f518ba8cde8a419b0dc394b925be310c95aa0f3705f625ff0f628a11da38cdf94da4f493e9bb75`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-node-linux-amd64.tar.gz) | `88a9b68c8cba77fe50751d998117ab632d1e8aa12a45f6bef71a24ee5a8fb6f559d00f129b8682f9d5838671edb6649e3c9caebdf9ce2a37f282f21316a522e0`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-node-linux-arm.tar.gz) | `3b558a1743893a994ec061a86aaf343d90e800d7ccd69c771b92d8915fc14217046a2f4817829dbdbe025331fe733ae401dbf671b4345d7a88b0c6470fbd99e8`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-node-linux-arm64.tar.gz) | `534b3db7e21f247189a484bb57958a3276bf74268d5943d712e68db50806afeeb1253acdc6a4c639c6ac08045e95ac4f9aaf95fa8192c85a19831329068bf5c6`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-node-linux-ppc64le.tar.gz) | `0bb1c7ee23ce7dbee0614e2d8fb8d79e0a36615ea4ea39ef97acf4e907ca5a9b57ee3735afc7b5bce8dd5b15f104cf37e1d92168684bb4e5fd053ed3c35802e6`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-node-linux-s390x.tar.gz) | `e4529b0804696c8bae9430411d5b51087fa6c204bef37a1c6e30d01490c7e996367f1e9aa1d9c10ca2faf9d90c02459fc36c64b1790fa88b703bb134df54c1c1`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.1/kubernetes-node-windows-amd64.tar.gz) | `7d976b1b22766cdd65b2b84602053b765e487d947966b4aaa3b169bb462d095e6a794154cd920b2c5e8a2d59695b75435ddb1237bc07404bba1273a0db01539e`

## Changelog since v1.18.0

## Changes by Kind

### Feature

- deps: Update to Golang 1.13.9
  - build: Remove kube-cross image building ([#89398](https://github.com/kubernetes/kubernetes/pull/89398), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]

### Other (Bug, Cleanup or Flake)

- Azure: fix concurreny issue in lb creation ([#89604](https://github.com/kubernetes/kubernetes/pull/89604), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Ensure Azure availability zone is always in lower cases. ([#89722](https://github.com/kubernetes/kubernetes/pull/89722), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix kubectl diff so it doesn't actually persist patches ([#89795](https://github.com/kubernetes/kubernetes/pull/89795), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Fix: get attach disk error due to missing item in max count table ([#89768](https://github.com/kubernetes/kubernetes/pull/89768), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fixed the EndpointSlice controller to run without error on a cluster with the OwnerReferencesPermissionEnforcement validating admission plugin enabled. ([#89804](https://github.com/kubernetes/kubernetes/pull/89804), [@marun](https://github.com/marun)) [SIG Auth and Network]
- Fixes kubectl to apply all validly built objects, instead of stopping on error. ([#89864](https://github.com/kubernetes/kubernetes/pull/89864), [@seans3](https://github.com/seans3)) [SIG CLI and Testing]
- In the kubelet resource metrics endpoint at /metrics/resource, change the names of the following metrics:
  - node_cpu_usage_seconds --> node_cpu_usage_seconds_total
  - container_cpu_usage_seconds --> container_cpu_usage_seconds_total
  This is a partial revert of &#35;86282, which was added in 1.18.0, and initially removed the _total suffix ([#89540](https://github.com/kubernetes/kubernetes/pull/89540), [@dashpole](https://github.com/dashpole)) [SIG Instrumentation and Node]
- Kubeadm: during join when a check is performed that a Node with the same name already exists in the cluster, make sure the NodeReady condition is properly validated ([#89602](https://github.com/kubernetes/kubernetes/pull/89602), [@kvaps](https://github.com/kvaps)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where post upgrade to 1.18.x, nodes cannot join the cluster due to missing RBAC ([#89537](https://github.com/kubernetes/kubernetes/pull/89537), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl azure authentication: fixed a regression in 1.18.0 where "spn:" prefix was unexpectedly added to the `apiserver-id` configuration in the kubeconfig file ([#89706](https://github.com/kubernetes/kubernetes/pull/89706), [@weinong](https://github.com/weinong)) [SIG API Machinery and Auth]
- Kubectl: Fixes bug by aggregating 'apply' errors instead of failing after first error ([#89607](https://github.com/kubernetes/kubernetes/pull/89607), [@seans3](https://github.com/seans3)) [SIG CLI and Testing]
- Reduce event spam during a volume operation error. ([#89796](https://github.com/kubernetes/kubernetes/pull/89796), [@msau42](https://github.com/msau42)) [SIG Storage]


# v1.18.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes.tar.gz) | `cd5b86a3947a4f2cea6d857743ab2009be127d782b6f2eb4d37d88918a5e433ad2c7ba34221c34089ba5ba13701f58b657f0711401e51c86f4007cb78744dee7`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-src.tar.gz) | `fb42cf133355ef18f67c8c4bb555aa1f284906c06e21fa41646e086d34ece774e9d547773f201799c0c703ce48d4d0e62c6ba5b2a4d081e12a339a423e111e52`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-darwin-386.tar.gz) | `26df342ef65745df12fa52931358e7f744111b6fe1e0bddb8c3c6598faf73af997c00c8f9c509efcd7cd7e82a0341a718c08fbd96044bfb58e80d997a6ebd3c2`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-darwin-amd64.tar.gz) | `803a0fed122ef6b85f7a120b5485723eaade765b7bc8306d0c0da03bd3df15d800699d15ea2270bb7797fa9ce6a81da90e730dc793ea4ed8c0149b63d26eca30`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-linux-386.tar.gz) | `110844511b70f9f3ebb92c15105e6680a05a562cd83f79ce2d2e25c2dd70f0dbd91cae34433f61364ae1ce4bd573b635f2f632d52de8f72b54acdbc95a15e3f0`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-linux-amd64.tar.gz) | `594ca3eadc7974ec4d9e4168453e36ca434812167ef8359086cd64d048df525b7bd46424e7cc9c41e65c72bda3117326ba1662d1c9d739567f10f5684fd85bee`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-linux-arm.tar.gz) | `d3627b763606557a6c9a5766c34198ec00b3a3cd72a55bc2cb47731060d31c4af93543fb53f53791062bb5ace2f15cbaa8592ac29009641e41bd656b0983a079`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-linux-arm64.tar.gz) | `ba9056eff1452cbdaef699efbf88f74f5309b3f7808d372ebf6918442d0c9fea1653c00b9db3b7626399a460eef9b1fa9e29b827b7784f34561cbc380554e2ea`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-linux-ppc64le.tar.gz) | `f80fb3769358cb20820ff1a1ce9994de5ed194aabe6c73fb8b8048bffc394d1b926de82c204f0e565d53ffe7562faa87778e97a3ccaaaf770034a992015e3a86`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-linux-s390x.tar.gz) | `a9b658108b6803d60fa3cd4e76d9e58bf75201017164fe54054b7ccadbb68c4ad7ba7800746940bc518d90475e6c0a96965a26fa50882f4f0e56df404f4ae586`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-windows-386.tar.gz) | `18adffab5d1be146906fd8531f4eae7153576aac235150ce2da05aee5ae161f6bd527e8dec34ae6131396cd4b3771e0d54ce770c065244ad3175a1afa63c89e1`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-client-windows-amd64.tar.gz) | `162396256429cef07154f817de2a6b67635c770311f414e38b1e2db25961443f05d7b8eb1f8da46dec8e31c5d1d2cd45f0c95dad1bc0e12a0a7278a62a0b9a6b`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-server-linux-amd64.tar.gz) | `a92f8d201973d5dfa44a398e95fcf6a7b4feeb1ef879ab3fee1c54370e21f59f725f27a9c09ace8c42c96ac202e297fd458e486c489e05f127a5cade53b8d7c4`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-server-linux-arm.tar.gz) | `62fbff3256bc0a83f70244b09149a8d7870d19c2c4b6dee8ca2714fc7388da340876a0f540d2ae9bbd8b81fdedaf4b692c72d2840674db632ba2431d1df1a37d`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-server-linux-arm64.tar.gz) | `842910a7013f61a60d670079716b207705750d55a9e4f1f93696d19d39e191644488170ac94d8740f8e3aa3f7f28f61a4347f69d7e93d149c69ac0efcf3688fe`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-server-linux-ppc64le.tar.gz) | `95c5b952ac1c4127a5c3b519b664972ee1fb5e8e902551ce71c04e26ad44b39da727909e025614ac1158c258dc60f504b9a354c5ab7583c2ad769717b30b3836`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-server-linux-s390x.tar.gz) | `a46522d2119a0fd58074564c1fa95dd8a929a79006b82ba3c4245611da8d2db9fd785c482e1b61a9aa361c5c9a6d73387b0e15e6a7a3d84fffb3f65db3b9deeb`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-node-linux-amd64.tar.gz) | `f714f80feecb0756410f27efb4cf4a1b5232be0444fbecec9f25cb85a7ccccdcb5be588cddee935294f460046c0726b90f7acc52b20eeb0c46a7200cf10e351a`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-node-linux-arm.tar.gz) | `806000b5f6d723e24e2f12d19d1b9b3d16c74b855f51c7063284adf1fcc57a96554a3384f8c05a952c6f6b929a05ed12b69151b1e620c958f74c9600f3db0fcb`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-node-linux-arm64.tar.gz) | `c207e9ab60587d135897b5366af79efe9d2833f33401e469b2a4e0d74ecd2cf6bb7d1e5bc18d80737acbe37555707f63dd581ccc6304091c1d98dafdd30130b7`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-node-linux-ppc64le.tar.gz) | `a542ed5ed02722af44ef12d1602f363fcd4e93cf704da2ea5d99446382485679626835a40ae2ba47a4a26dce87089516faa54479a1cfdee2229e8e35aa1c17d7`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-node-linux-s390x.tar.gz) | `651e0db73ee67869b2ae93cb0574168e4bd7918290fc5662a6b12b708fa628282e3f64be2b816690f5a2d0f4ff8078570f8187e65dee499a876580a7a63d1d19`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0/kubernetes-node-windows-amd64.tar.gz) | `d726ed904f9f7fe7e8831df621dc9094b87e767410a129aa675ee08417b662ddec314e165f29ecb777110fbfec0dc2893962b6c71950897ba72baaa7eb6371ed`

## Changelog since v1.17.0

A complete changelog for the release notes is now hosted in a customizable
format at [https://relnotes.k8s.io][1]. Check it out and please give us your
feedback!

[1]: https://relnotes.k8s.io/?releaseVersions=1.18.0

## What’s New (Major Themes)

### Kubernetes Topology Manager Moves to Beta - Align Up!

A beta feature of Kubernetes in release 1.18, the [Topology Manager feature](https://github.com/nolancon/website/blob/f4200307260ea3234540ef13ed80de325e1a7267/content/en/docs/tasks/administer-cluster/topology-manager.md) enables NUMA alignment of CPU and devices (such as SR-IOV VFs) that will allow your workload to run in an environment optimized for low-latency. Prior to the introduction of the Topology Manager, the CPU and Device Manager would make resource allocation decisions independent of each other. This could result in undesirable allocations on multi-socket systems, causing degraded performance on latency critical applications.

### Serverside Apply - Beta 2

Server-side Apply was promoted to Beta in 1.16, but is now introducing a second Beta in 1.18. This new version will track and manage changes to fields of all new Kubernetes objects, allowing you to know what changed your resources and when.

### Extending Ingress with and replacing a deprecated annotation with IngressClass

In Kubernetes 1.18, there are two significant additions to Ingress: A new `pathType` field and a new `IngressClass` resource. The `pathType` field allows specifying how paths should be matched. In addition to the default `ImplementationSpecific` type, there are new `Exact` and `Prefix` path types. 

The `IngressClass` resource is used to describe a type of Ingress within a Kubernetes cluster. Ingresses can specify the class they are associated with by using a new `ingressClassName` field on Ingresses. This new resource and field replace the deprecated `kubernetes.io/ingress.class` annotation.

### SIG CLI introduces kubectl debug

SIG CLI was debating the need for a debug utility for quite some time already. With the development of [ephemeral containers](https://kubernetes.io/docs/concepts/workloads/pods/ephemeral-containers/), it became more obvious how we can support developers with tooling built on top of `kubectl exec`. The addition of the `kubectl debug` [command](https://github.com/kubernetes/enhancements/blob/master/keps/sig-cli/20190805-kubectl-debug.md) (it is alpha but your feedback is more than welcome), allows developers to easily debug their Pods inside the cluster. We think this addition is invaluable.  This command allows one to create a temporary container which runs next to the Pod one is trying to examine, but also attaches to the console for interactive troubleshooting.

### Introducing Windows CSI support alpha for Kubernetes

With the release of Kubernetes 1.18, an alpha version of CSI Proxy for Windows is getting released. CSI proxy enables non-privileged (pre-approved) containers to perform privileged storage operations on Windows. CSI drivers can now be supported in Windows by leveraging CSI proxy.
SIG Storage made a lot of progress in the 1.18 release.
In particular, the following storage features are moving to GA in Kubernetes 1.18:
- Raw Block Support: Allow volumes to be surfaced as block devices inside containers instead of just mounted filesystems.
- Volume Cloning: Duplicate a PersistentVolumeClaim and underlying storage volume using the Kubernetes API via CSI.
- CSIDriver Kubernetes API Object: Simplifies CSI driver discovery and allows CSI Drivers to customize Kubernetes behavior.

SIG Storage is also introducing the following new storage features as alpha in Kubernetes 1.18:
- Windows CSI Support: Enabling containerized CSI node plugins in Windows via new [CSIProxy](https://github.com/kubernetes-csi/csi-proxy)
- Recursive Volume Ownership OnRootMismatch Option: Add a new “OnRootMismatch” policy that can help shorten the mount time for volumes that require ownership change and have many directories and files.

### Other notable announcements

SIG Network is moving IPv6 to Beta in Kubernetes 1.18, after incrementing significantly the test coverage with new CI jobs.

NodeLocal DNSCache is an add-on that runs a dnsCache pod as a daemonset to improve clusterDNS performance and reliability. The feature has been in Alpha since 1.13 release. The SIG Network is announcing the GA graduation of Node Local DNSCache [#1351](https://github.com/kubernetes/enhancements/pull/1351)

## Known Issues

No Known Issues Reported

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

#### kube-apiserver:
- in an `--encryption-provider-config` config file, an explicit `cacheSize: 0` parameter previously silently defaulted to caching 1000 keys. In Kubernetes 1.18, this now returns a config validation error. To disable caching, you can specify a negative cacheSize value in Kubernetes 1.18+.
- consumers of the 'certificatesigningrequests/approval' API must now have permission to 'approve' CSRs for the specific signer requested by the CSR. More information on the new signerName field and the required authorization can be found at https://kubernetes.io/docs/reference/access-authn-authz/certificate-signing-requests#authorization ([#88246](https://github.com/kubernetes/kubernetes/pull/88246), [@munnerz](https://github.com/munnerz)) [SIG API Machinery, Apps, Auth, CLI, Node and Testing]
- The following features are unconditionally enabled and the corresponding `--feature-gates` flags have been removed: `PodPriority`, `TaintNodesByCondition`, `ResourceQuotaScopeSelectors` and `ScheduleDaemonSetPods` ([#86210](https://github.com/kubernetes/kubernetes/pull/86210), [@draveness](https://github.com/draveness)) [SIG Apps and Scheduling]

#### kubelet:
- `--enable-cadvisor-endpoints` is now disabled by default. If you need access to the cAdvisor v1 Json API please enable it explicitly in the kubelet command line. Please note that this flag was deprecated in 1.15 and will be removed in 1.19. ([#87440](https://github.com/kubernetes/kubernetes/pull/87440), [@dims](https://github.com/dims)) [SIG Instrumentation, Node and Testing]
- Promote CSIMigrationOpenStack to Beta (off by default since it requires installation of the OpenStack Cinder CSI Driver. The in-tree AWS OpenStack Cinder driver "kubernetes.io/cinder" was deprecated in 1.16 and will be removed in 1.20. Users should enable CSIMigration + CSIMigrationOpenStack features and install the OpenStack Cinder CSI Driver (https://github.com/kubernetes-sigs/cloud-provider-openstack) to avoid disruption to existing Pod and PVC objects at that time. Users should start using the OpenStack Cinder CSI Driver directly for any new volumes. ([#85637](https://github.com/kubernetes/kubernetes/pull/85637), [@dims](https://github.com/dims)) [SIG Cloud Provider]

#### kubectl:
- `kubectl` and k8s.io/client-go no longer default to a server address of `http://localhost:8080`. If you own one of these legacy clusters, you are *strongly* encouraged to secure your server. If you cannot secure your server, you can set the `$KUBERNETES_MASTER` environment variable to `http://localhost:8080` to continue defaulting the server address. `kubectl` users can also set the server address using the `--server` flag, or in a kubeconfig file specified via `--kubeconfig` or `$KUBECONFIG`. ([#86173](https://github.com/kubernetes/kubernetes/pull/86173), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI and Testing]
- `kubectl run` has removed the previously deprecated generators, along with flags unrelated to creating pods. `kubectl run` now only creates pods. See specific `kubectl create` subcommands to create objects other than pods. 
([#87077](https://github.com/kubernetes/kubernetes/pull/87077), [@soltysh](https://github.com/soltysh)) [SIG Architecture, CLI and Testing]
- The deprecated command `kubectl rolling-update` has been removed ([#88057](https://github.com/kubernetes/kubernetes/pull/88057), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG Architecture, CLI and Testing]

#### client-go:
- Signatures on methods in generated clientsets, dynamic, metadata, and scale clients have been modified to accept `context.Context` as a first argument. Signatures of Create, Update, and Patch methods have been updated to accept CreateOptions, UpdateOptions and PatchOptions respectively. Signatures of Delete and DeleteCollection methods now accept DeleteOptions by value instead of by reference. Generated clientsets with the previous interface have been added in new "deprecated" packages to allow incremental migration to the new APIs. The deprecated packages will be removed in the 1.21 release. A tool is available at http://sigs.k8s.io/clientgofix to rewrite method invocations to the new signatures.

- The following deprecated metrics are removed, please convert to the corresponding metrics:
  - The following replacement metrics are available from v1.14.0:
      - `rest_client_request_latency_seconds` -> `rest_client_request_duration_seconds`
      - `scheduler_scheduling_latency_seconds` -> `scheduler_scheduling_duration_seconds `
      - `docker_operations` -> `docker_operations_total`
      - `docker_operations_latency_microseconds` -> `docker_operations_duration_seconds`
      - `docker_operations_errors` -> `docker_operations_errors_total`
      - `docker_operations_timeout` -> `docker_operations_timeout_total`
      - `network_plugin_operations_latency_microseconds` -> `network_plugin_operations_duration_seconds`
      - `kubelet_pod_worker_latency_microseconds` -> `kubelet_pod_worker_duration_seconds`
      - `kubelet_pod_start_latency_microseconds` -> `kubelet_pod_start_duration_seconds`
      - `kubelet_cgroup_manager_latency_microseconds` -> `kubelet_cgroup_manager_duration_seconds`
      - `kubelet_pod_worker_start_latency_microseconds` -> `kubelet_pod_worker_start_duration_seconds`
      - `kubelet_pleg_relist_latency_microseconds` -> `kubelet_pleg_relist_duration_seconds`
      - `kubelet_pleg_relist_interval_microseconds` -> `kubelet_pleg_relist_interval_seconds`
      - `kubelet_eviction_stats_age_microseconds` -> `kubelet_eviction_stats_age_seconds`
      - `kubelet_runtime_operations` -> `kubelet_runtime_operations_total`
      - `kubelet_runtime_operations_latency_microseconds` -> `kubelet_runtime_operations_duration_seconds`
      - `kubelet_runtime_operations_errors` -> `kubelet_runtime_operations_errors_total`
      - `kubelet_device_plugin_registration_count` -> `kubelet_device_plugin_registration_total`
      - `kubelet_device_plugin_alloc_latency_microseconds` -> `kubelet_device_plugin_alloc_duration_seconds`
      - `scheduler_e2e_scheduling_latency_microseconds` -> `scheduler_e2e_scheduling_duration_seconds`
      - `scheduler_scheduling_algorithm_latency_microseconds` -> `scheduler_scheduling_algorithm_duration_seconds`
      - `scheduler_scheduling_algorithm_predicate_evaluation` -> `scheduler_scheduling_algorithm_predicate_evaluation_seconds`
      - `scheduler_scheduling_algorithm_priority_evaluation` -> `scheduler_scheduling_algorithm_priority_evaluation_seconds`
      - `scheduler_scheduling_algorithm_preemption_evaluation` -> `scheduler_scheduling_algorithm_preemption_evaluation_seconds`
      - `scheduler_binding_latency_microseconds` -> `scheduler_binding_duration_seconds`
      - `kubeproxy_sync_proxy_rules_latency_microseconds` -> `kubeproxy_sync_proxy_rules_duration_seconds`
      - `apiserver_request_latencies` -> `apiserver_request_duration_seconds`
      - `apiserver_dropped_requests` -> `apiserver_dropped_requests_total`
      - `etcd_request_latencies_summary` -> `etcd_request_duration_seconds`
      - `apiserver_storage_transformation_latencies_microseconds ` -> `apiserver_storage_transformation_duration_seconds`
      - `apiserver_storage_data_key_generation_latencies_microseconds` -> `apiserver_storage_data_key_generation_duration_seconds`
      - `apiserver_request_count` -> `apiserver_request_total`
      - `apiserver_request_latencies_summary`
  - The following replacement metrics are available from v1.15.0:
      - `apiserver_storage_transformation_failures_total` -> `apiserver_storage_transformation_operations_total` ([#76496](https://github.com/kubernetes/kubernetes/pull/76496), [@danielqsj](https://github.com/danielqsj)) [SIG API Machinery, Cluster Lifecycle, Instrumentation, Network, Node and Scheduling]

## Changes by Kind

### Deprecation

#### kube-apiserver:
- the following deprecated APIs can no longer be served:
  - All resources under `apps/v1beta1` and `apps/v1beta2` - use `apps/v1` instead
  - `daemonsets`, `deployments`, `replicasets` resources under `extensions/v1beta1` - use `apps/v1` instead
  - `networkpolicies` resources under `extensions/v1beta1` - use `networking.k8s.io/v1` instead
  - `podsecuritypolicies` resources under `extensions/v1beta1` - use `policy/v1beta1` instead ([#85903](https://github.com/kubernetes/kubernetes/pull/85903), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Cluster Lifecycle, Instrumentation and Testing]

#### kube-controller-manager:
- Azure service annotation service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset has been deprecated. Its support would be removed in a future release. ([#88462](https://github.com/kubernetes/kubernetes/pull/88462), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]

#### kubelet:
- The StreamingProxyRedirects feature and `--redirect-container-streaming` flag are deprecated, and will be removed in a future release. The default behavior (proxy streaming requests through the kubelet) will be the only supported option. If you are setting `--redirect-container-streaming=true`, then you must migrate off this configuration. The flag will no longer be able to be enabled starting in v1.20. If you are not setting the flag, no action is necessary. ([#88290](https://github.com/kubernetes/kubernetes/pull/88290), [@tallclair](https://github.com/tallclair)) [SIG API Machinery and Node]
- resource metrics endpoint `/metrics/resource/v1alpha1` as well as all metrics under this endpoint have been deprecated. Please convert to the following metrics emitted by endpoint `/metrics/resource`:
      - scrape_error --> scrape_error
      - node_cpu_usage_seconds_total --> node_cpu_usage_seconds
      - node_memory_working_set_bytes --> node_memory_working_set_bytes
      - container_cpu_usage_seconds_total --> container_cpu_usage_seconds
      - container_memory_working_set_bytes --> container_memory_working_set_bytes
      - scrape_error --> scrape_error 
      ([#86282](https://github.com/kubernetes/kubernetes/pull/86282), [@RainbowMango](https://github.com/RainbowMango)) [SIG Node]
- In a future release, kubelet will no longer create the CSI NodePublishVolume target directory, in accordance with the CSI specification. CSI drivers may need to be updated accordingly to properly create and process the target path. ([#75535](https://github.com/kubernetes/kubernetes/issues/75535)) [SIG Storage]

#### kube-proxy:
- `--healthz-port` and `--metrics-port` flags are deprecated, please use `--healthz-bind-address` and `--metrics-bind-address` instead ([#88512](https://github.com/kubernetes/kubernetes/pull/88512), [@SataQiu](https://github.com/SataQiu)) [SIG Network]
- a new `EndpointSliceProxying` feature gate has been added to control the use of EndpointSlices in kube-proxy. The EndpointSlice feature gate that used to control this behavior no longer affects kube-proxy. This feature has been disabled by default. ([#86137](https://github.com/kubernetes/kubernetes/pull/86137), [@robscott](https://github.com/robscott)) 

#### kubeadm:
- command line option "kubelet-version" for `kubeadm upgrade node` has been deprecated and will be removed in a future release. ([#87942](https://github.com/kubernetes/kubernetes/pull/87942), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- deprecate the usage of the experimental flag '--use-api' under the 'kubeadm alpha certs renew' command. ([#88827](https://github.com/kubernetes/kubernetes/pull/88827), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- kube-dns is deprecated and will not be supported in a future version ([#86574](https://github.com/kubernetes/kubernetes/pull/86574), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- the `ClusterStatus` struct present in the kubeadm-config ConfigMap is deprecated and will be removed in a future version. It is going to be maintained by kubeadm until it gets removed. The same information can be found on `etcd` and `kube-apiserver` pod annotations, `kubeadm.kubernetes.io/etcd.advertise-client-urls` and `kubeadm.kubernetes.io/kube-apiserver.advertise-address.endpoint` respectively. ([#87656](https://github.com/kubernetes/kubernetes/pull/87656), [@ereslibre](https://github.com/ereslibre)) [SIG Cluster Lifecycle]

#### kubectl:
- the boolean and unset values for the --dry-run flag are deprecated and a value --dry-run=server|client|none will be required in a future version. ([#87580](https://github.com/kubernetes/kubernetes/pull/87580), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI]
- `kubectl apply --server-dry-run` is deprecated and replaced with --dry-run=server ([#87580](https://github.com/kubernetes/kubernetes/pull/87580), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI]

#### add-ons:
- Remove cluster-monitoring addon ([#85512](https://github.com/kubernetes/kubernetes/pull/85512), [@serathius](https://github.com/serathius)) [SIG Cluster Lifecycle, Instrumentation, Scalability and Testing]

#### kube-scheduler:
- The `scheduling_duration_seconds` summary metric is deprecated ([#86586](https://github.com/kubernetes/kubernetes/pull/86586), [@xiaoanyunfei](https://github.com/xiaoanyunfei)) [SIG Scheduling]
- The `scheduling_algorithm_predicate_evaluation_seconds` and
  `scheduling_algorithm_priority_evaluation_seconds` metrics are deprecated, replaced by `framework_extension_point_duration_seconds[extension_point="Filter"]` and `framework_extension_point_duration_seconds[extension_point="Score"]`. ([#86584](https://github.com/kubernetes/kubernetes/pull/86584), [@xiaoanyunfei](https://github.com/xiaoanyunfei)) [SIG Scheduling]
- `AlwaysCheckAllPredicates` is deprecated in scheduler Policy API. ([#86369](https://github.com/kubernetes/kubernetes/pull/86369), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]

#### Other deprecations:
- The k8s.io/node-api component is no longer updated. Instead, use the RuntimeClass types located within k8s.io/api, and the generated clients located within k8s.io/client-go ([#87503](https://github.com/kubernetes/kubernetes/pull/87503), [@liggitt](https://github.com/liggitt)) [SIG Node and Release]
- Removed the 'client' label from apiserver_request_total. ([#87669](https://github.com/kubernetes/kubernetes/pull/87669), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Instrumentation]

### API Change

#### New API types/versions:
- A new IngressClass resource has been added to enable better Ingress configuration. ([#88509](https://github.com/kubernetes/kubernetes/pull/88509), [@robscott](https://github.com/robscott)) [SIG API Machinery, Apps, CLI, Network, Node and Testing]
- The CSIDriver API has graduated to storage.k8s.io/v1, and is now available for use. ([#84814](https://github.com/kubernetes/kubernetes/pull/84814), [@huffmanca](https://github.com/huffmanca)) [SIG Storage]

#### New API fields:
- autoscaling/v2beta2 HorizontalPodAutoscaler added a `spec.behavior` field that allows scale behavior to be configured. Behaviors are specified separately for scaling up and down. In each direction a stabilization window can be specified as well as a list of policies and how to select amongst them. Policies can limit the absolute number of pods added or removed, or the percentage of pods added or removed. ([#74525](https://github.com/kubernetes/kubernetes/pull/74525), [@gliush](https://github.com/gliush)) [SIG API Machinery, Apps, Autoscaling and CLI]
- Ingress:
  - `spec.ingressClassName` replaces the deprecated `kubernetes.io/ingress.class` annotation, and allows associating an Ingress object with a particular controller.
  - path definitions added a `pathType` field to allow indicating how the specified path should be matched against incoming requests. Valid values are `Exact`, `Prefix`, and `ImplementationSpecific` ([#88587](https://github.com/kubernetes/kubernetes/pull/88587), [@cmluciano](https://github.com/cmluciano)) [SIG Apps, Cluster Lifecycle and Network]
- The alpha feature `AnyVolumeDataSource` enables PersistentVolumeClaim objects to use the spec.dataSource field to reference a custom type as a data source ([#88636](https://github.com/kubernetes/kubernetes/pull/88636), [@bswartz](https://github.com/bswartz)) [SIG Apps and Storage]
- The alpha feature `ConfigurableFSGroupPolicy` enables v1 Pods to specify a spec.securityContext.fsGroupChangePolicy policy to control how file permissions are applied to volumes mounted into the pod. ([#88488](https://github.com/kubernetes/kubernetes/pull/88488), [@gnufied](https://github.com/gnufied)) [SIG  Storage]
- The alpha feature `ServiceAppProtocol` enables setting an `appProtocol` field in ServicePort and EndpointPort definitions. ([#88503](https://github.com/kubernetes/kubernetes/pull/88503), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- The alpha feature `ImmutableEphemeralVolumes` enables an `immutable` field in both Secret and ConfigMap objects to mark their contents as immutable. ([#86377](https://github.com/kubernetes/kubernetes/pull/86377), [@wojtek-t](https://github.com/wojtek-t)) [SIG Apps, CLI and Testing]

#### Other API changes:
- The beta feature `ServerSideApply` enables tracking and managing changed fields for all new objects, which means there will be `managedFields` in `metadata` with the list of managers and their owned fields.
- The alpha feature `ServiceAccountIssuerDiscovery` enables publishing OIDC discovery information and service account token verification keys at `/.well-known/openid-configuration` and `/openid/v1/jwks` endpoints by API servers configured to issue service account tokens. ([#80724](https://github.com/kubernetes/kubernetes/pull/80724), [@cceckman](https://github.com/cceckman)) [SIG API Machinery, Auth, Cluster Lifecycle and Testing]
- CustomResourceDefinition schemas that use `x-kubernetes-list-map-keys` to specify properties that uniquely identify list items must make those properties required or have a default value, to ensure those properties are present for all list items. See https://kubernetes.io/docs/reference/using-api/api-concepts/&#35;merge-strategy for details. ([#88076](https://github.com/kubernetes/kubernetes/pull/88076), [@eloyekunle](https://github.com/eloyekunle)) [SIG API Machinery and Testing]
- CustomResourceDefinition schemas that use `x-kubernetes-list-type: map` or `x-kubernetes-list-type: set` now enable validation that the list items in the corresponding custom resources are unique. ([#84920](https://github.com/kubernetes/kubernetes/pull/84920), [@sttts](https://github.com/sttts)) [SIG API Machinery]
 
#### Configuration file changes:

#### kube-apiserver:
- The `--egress-selector-config-file` configuration file now accepts an apiserver.k8s.io/v1beta1  EgressSelectorConfiguration configuration object, and has been updated to allow specifying HTTP or GRPC connections to the network proxy ([#87179](https://github.com/kubernetes/kubernetes/pull/87179), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Cloud Provider and Cluster Lifecycle]

#### kube-scheduler:
- A kubescheduler.config.k8s.io/v1alpha2 configuration file version is now accepted, with support for multiple scheduling profiles ([#87628](https://github.com/kubernetes/kubernetes/pull/87628), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
  - HardPodAffinityWeight moved from a top level ComponentConfig parameter to a PluginConfig parameter of InterPodAffinity Plugin in `kubescheduler.config.k8s.io/v1alpha2` ([#88002](https://github.com/kubernetes/kubernetes/pull/88002), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling and Testing]
  - Kube-scheduler can run more than one scheduling profile. Given a pod, the profile is selected by using its `.spec.schedulerName`. ([#88285](https://github.com/kubernetes/kubernetes/pull/88285), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps, Scheduling and Testing]
  - Scheduler Extenders can now be configured in the v1alpha2 component config ([#88768](https://github.com/kubernetes/kubernetes/pull/88768), [@damemi](https://github.com/damemi)) [SIG Release, Scheduling and Testing]
  - The PostFilter of scheduler framework is renamed to PreScore in kubescheduler.config.k8s.io/v1alpha2. ([#87751](https://github.com/kubernetes/kubernetes/pull/87751), [@skilxn-go](https://github.com/skilxn-go)) [SIG Scheduling and Testing]
 
#### kube-proxy:
- Added kube-proxy flags `--ipvs-tcp-timeout`, `--ipvs-tcpfin-timeout`, `--ipvs-udp-timeout` to configure IPVS connection timeouts. ([#85517](https://github.com/kubernetes/kubernetes/pull/85517), [@andrewsykim](https://github.com/andrewsykim)) [SIG Cluster Lifecycle and Network]
- Added optional `--detect-local-mode` flag to kube-proxy. Valid values are "ClusterCIDR" (default matching previous behavior) and "NodeCIDR" ([#87748](https://github.com/kubernetes/kubernetes/pull/87748), [@satyasm](https://github.com/satyasm)) [SIG Cluster Lifecycle, Network and Scheduling]
- Kube-controller-manager and kube-scheduler expose profiling by default to match the kube-apiserver.  Use `--enable-profiling=false` to disable. ([#88663](https://github.com/kubernetes/kubernetes/pull/88663), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Cloud Provider and Scheduling]
- Kubelet pod resources API now provides the information about active pods only. ([#79409](https://github.com/kubernetes/kubernetes/pull/79409), [@takmatsu](https://github.com/takmatsu)) [SIG Node]
- New flag `--endpointslice-updates-batch-period` in kube-controller-manager can be used to reduce the number of endpointslice updates generated by pod changes. ([#88745](https://github.com/kubernetes/kubernetes/pull/88745), [@mborsz](https://github.com/mborsz)) [SIG API Machinery, Apps and Network]
- New flag `--show-hidden-metrics-for-version` in kube-proxy, kubelet, kube-controller-manager, and kube-scheduler can be used to show all hidden metrics that are deprecated in the previous minor release. ([#85279](https://github.com/kubernetes/kubernetes/pull/85279), [@RainbowMango](https://github.com/RainbowMango)) [SIG Cluster Lifecycle and Network]

#### Features graduated to beta:
  - StartupProbe ([#83437](https://github.com/kubernetes/kubernetes/pull/83437), [@matthyx](https://github.com/matthyx)) [SIG Node, Scalability and Testing]

#### Features graduated to GA:
  - VolumePVCDataSource ([#88686](https://github.com/kubernetes/kubernetes/pull/88686), [@j-griffith](https://github.com/j-griffith)) [SIG Storage]
  - TaintBasedEvictions ([#87487](https://github.com/kubernetes/kubernetes/pull/87487), [@skilxn-go](https://github.com/skilxn-go)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
  - BlockVolume and CSIBlockVolume ([#88673](https://github.com/kubernetes/kubernetes/pull/88673), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
  - Windows RunAsUserName ([#87790](https://github.com/kubernetes/kubernetes/pull/87790), [@marosset](https://github.com/marosset)) [SIG Apps and Windows]
- The following feature gates are removed, because the associated features were unconditionally enabled in previous releases: CustomResourceValidation, CustomResourceSubresources, CustomResourceWebhookConversion, CustomResourcePublishOpenAPI, CustomResourceDefaulting ([#87475](https://github.com/kubernetes/kubernetes/pull/87475), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]

### Feature

- API request throttling (due to a high rate of requests) is now reported in client-go logs at log level 2.  The messages are of the form:`Throttling request took 1.50705208s, request: GET:<URL>` The presence of these messages may indicate to the administrator the need to tune the cluster accordingly. ([#87740](https://github.com/kubernetes/kubernetes/pull/87740), [@jennybuckley](https://github.com/jennybuckley)) [SIG API Machinery]
- Add support for mount options to the FC volume plugin ([#87499](https://github.com/kubernetes/kubernetes/pull/87499), [@ejweber](https://github.com/ejweber)) [SIG Storage]
- Added a config-mode flag in azure auth module to enable getting AAD token without spn: prefix in audience claim. When it's not specified, the default behavior doesn't change. ([#87630](https://github.com/kubernetes/kubernetes/pull/87630), [@weinong](https://github.com/weinong)) [SIG API Machinery, Auth, CLI and Cloud Provider]
- Allow for configuration of CoreDNS replica count ([#85837](https://github.com/kubernetes/kubernetes/pull/85837), [@pickledrick](https://github.com/pickledrick)) [SIG Cluster Lifecycle]
- Allow user to specify resource using --filename flag when invoking kubectl exec ([#88460](https://github.com/kubernetes/kubernetes/pull/88460), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Apiserver added a new flag --goaway-chance which is the fraction of requests that will be closed gracefully(GOAWAY) to prevent HTTP/2 clients from getting stuck on a single apiserver. ([#88567](https://github.com/kubernetes/kubernetes/pull/88567), [@answer1991](https://github.com/answer1991)) [SIG API Machinery]
- Azure Cloud Provider now supports using Azure network resources (Virtual Network, Load Balancer, Public IP, Route Table, Network Security Group, etc.) in different AAD Tenant and Subscription than those for the Kubernetes cluster. To use the feature, please reference https://github.com/kubernetes-sigs/cloud-provider-azure/blob/master/docs/cloud-provider-config.md&#35;host-network-resources-in-different-aad-tenant-and-subscription. ([#88384](https://github.com/kubernetes/kubernetes/pull/88384), [@bowen5](https://github.com/bowen5)) [SIG Cloud Provider]
- Azure VMSS/VMSSVM clients now suppress requests on throttling ([#86740](https://github.com/kubernetes/kubernetes/pull/86740), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Azure cloud provider cache TTL is configurable, list of the azure cloud provider is as following:
  - "availabilitySetNodesCacheTTLInSeconds"
  - "vmssCacheTTLInSeconds"
  - "vmssVirtualMachinesCacheTTLInSeconds"
  - "vmCacheTTLInSeconds"
  - "loadBalancerCacheTTLInSeconds"
  - "nsgCacheTTLInSeconds"
  - "routeTableCacheTTLInSeconds"
  ([#86266](https://github.com/kubernetes/kubernetes/pull/86266), [@zqingqing1](https://github.com/zqingqing1)) [SIG Cloud Provider]
- Azure global rate limit is switched to per-client. A set of new rate limit configure options are introduced, including routeRateLimit, SubnetsRateLimit, InterfaceRateLimit, RouteTableRateLimit, LoadBalancerRateLimit, PublicIPAddressRateLimit, SecurityGroupRateLimit, VirtualMachineRateLimit, StorageAccountRateLimit, DiskRateLimit, SnapshotRateLimit, VirtualMachineScaleSetRateLimit and VirtualMachineSizeRateLimit. The original rate limit options would be default values for those new client's rate limiter. ([#86515](https://github.com/kubernetes/kubernetes/pull/86515), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Azure network and VM clients now suppress requests on throttling ([#87122](https://github.com/kubernetes/kubernetes/pull/87122), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Azure storage clients now suppress requests on throttling ([#87306](https://github.com/kubernetes/kubernetes/pull/87306), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Azure: add support for single stack IPv6 ([#88448](https://github.com/kubernetes/kubernetes/pull/88448), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- DefaultConstraints can be specified for PodTopologySpread Plugin in the scheduler’s ComponentConfig ([#88671](https://github.com/kubernetes/kubernetes/pull/88671), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- DisableAvailabilitySetNodes is added to avoid VM list for VMSS clusters. It should only be used when vmType is "vmss" and all the nodes (including control plane nodes) are VMSS virtual machines. ([#87685](https://github.com/kubernetes/kubernetes/pull/87685), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Elasticsearch supports automatically setting the advertise address ([#85944](https://github.com/kubernetes/kubernetes/pull/85944), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle and Instrumentation]
- EndpointSlices will now be enabled by default. A new `EndpointSliceProxying` feature gate determines if kube-proxy will use EndpointSlices, this is disabled by default. ([#86137](https://github.com/kubernetes/kubernetes/pull/86137), [@robscott](https://github.com/robscott)) [SIG Network]
- Kube-proxy: Added dual-stack IPv4/IPv6 support to the iptables proxier. ([#82462](https://github.com/kubernetes/kubernetes/pull/82462), [@vllry](https://github.com/vllry)) [SIG Network]
- Kubeadm now supports automatic calculations of dual-stack node cidr masks to kube-controller-manager. ([#85609](https://github.com/kubernetes/kubernetes/pull/85609), [@Arvinderpal](https://github.com/Arvinderpal)) [SIG Cluster Lifecycle]
- Kubeadm: add a upgrade health check that deploys a Job ([#81319](https://github.com/kubernetes/kubernetes/pull/81319), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: add the experimental feature gate PublicKeysECDSA that can be used to create a
  cluster with ECDSA certificates from "kubeadm init". Renewal of existing ECDSA certificates is also supported using "kubeadm alpha certs renew", but not switching between the RSA and ECDSA algorithms on the fly or during upgrades. ([#86953](https://github.com/kubernetes/kubernetes/pull/86953), [@rojkov](https://github.com/rojkov)) [SIG API Machinery, Auth and Cluster Lifecycle]
- Kubeadm: implemented structured output of 'kubeadm config images list' command in JSON, YAML, Go template and JsonPath formats ([#86810](https://github.com/kubernetes/kubernetes/pull/86810), [@bart0sh](https://github.com/bart0sh)) [SIG Cluster Lifecycle]
- Kubeadm: on kubeconfig certificate renewal, keep the embedded CA in sync with the one on disk ([#88052](https://github.com/kubernetes/kubernetes/pull/88052), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: reject a node joining the cluster if a node with the same name already exists ([#81056](https://github.com/kubernetes/kubernetes/pull/81056), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: support Windows specific kubelet flags in kubeadm-flags.env ([#88287](https://github.com/kubernetes/kubernetes/pull/88287), [@gab-satchi](https://github.com/gab-satchi)) [SIG Cluster Lifecycle and Windows]
- Kubeadm: support automatic retry after failing to pull image ([#86899](https://github.com/kubernetes/kubernetes/pull/86899), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: upgrade supports fallback to the nearest known etcd version if an unknown k8s version is passed ([#88373](https://github.com/kubernetes/kubernetes/pull/88373), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl/drain: add disable-eviction option.Force drain to use delete, even if eviction is supported. This will bypass checking PodDisruptionBudgets, and should be used with caution. ([#85571](https://github.com/kubernetes/kubernetes/pull/85571), [@michaelgugino](https://github.com/michaelgugino)) [SIG CLI]
- Kubectl/drain: add skip-wait-for-delete-timeout option. If a pod’s  `DeletionTimestamp` is older than N seconds, skip waiting for the pod.  Seconds must be greater than 0 to skip. ([#85577](https://github.com/kubernetes/kubernetes/pull/85577), [@michaelgugino](https://github.com/michaelgugino)) [SIG CLI]
- Option `preConfiguredBackendPoolLoadBalancerTypes` is added to azure cloud provider for the pre-configured load balancers, possible values: `""`, `"internal"`, `"external"`,`"all"` ([#86338](https://github.com/kubernetes/kubernetes/pull/86338), [@gossion](https://github.com/gossion)) [SIG Cloud Provider]
- PodTopologySpread plugin now excludes terminatingPods when making scheduling decisions. ([#87845](https://github.com/kubernetes/kubernetes/pull/87845), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- Provider/azure: Network security groups can now be in a separate resource group. ([#87035](https://github.com/kubernetes/kubernetes/pull/87035), [@CecileRobertMichon](https://github.com/CecileRobertMichon)) [SIG Cloud Provider]
- SafeSysctlWhitelist: add net.ipv4.ping_group_range ([#85463](https://github.com/kubernetes/kubernetes/pull/85463), [@AkihiroSuda](https://github.com/AkihiroSuda)) [SIG Auth]
- Scheduler framework permit plugins now run at the end of the scheduling cycle, after reserve plugins. Waiting on permit will remain in the beginning of the binding cycle. ([#88199](https://github.com/kubernetes/kubernetes/pull/88199), [@mateuszlitwin](https://github.com/mateuszlitwin)) [SIG Scheduling]
- Scheduler: Add DefaultBinder plugin ([#87430](https://github.com/kubernetes/kubernetes/pull/87430), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling and Testing]
- Skip default spreading scoring plugin for pods that define TopologySpreadConstraints ([#87566](https://github.com/kubernetes/kubernetes/pull/87566), [@skilxn-go](https://github.com/skilxn-go)) [SIG Scheduling]
- The kubectl --dry-run flag now accepts the values 'client', 'server', and 'none', to support client-side and server-side dry-run strategies. The boolean and unset values for the --dry-run flag are deprecated and a value will be required in a future version. ([#87580](https://github.com/kubernetes/kubernetes/pull/87580), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI]
- Support server-side dry-run in kubectl with --dry-run=server for commands including apply, patch, create, run, annotate, label, set, autoscale, drain, rollout undo, and expose. ([#87714](https://github.com/kubernetes/kubernetes/pull/87714), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG API Machinery, CLI and Testing]
- Add --dry-run=server|client to kubectl delete, taint, replace ([#88292](https://github.com/kubernetes/kubernetes/pull/88292), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- The feature PodTopologySpread (feature gate `EvenPodsSpread`) has been enabled by default in 1.18. ([#88105](https://github.com/kubernetes/kubernetes/pull/88105), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- The kubelet and the default docker runtime now support running ephemeral containers in the Linux process namespace of a target container. Other container runtimes must implement support for this feature before it will be available for that runtime. ([#84731](https://github.com/kubernetes/kubernetes/pull/84731), [@verb](https://github.com/verb)) [SIG Node]
- The underlying format of the `CPUManager` state file has changed. Upgrades should be seamless, but any third-party tools that rely on reading the previous format need to be updated. ([#84462](https://github.com/kubernetes/kubernetes/pull/84462), [@klueska](https://github.com/klueska)) [SIG Node and Testing]
- Update CNI version to v0.8.5 ([#78819](https://github.com/kubernetes/kubernetes/pull/78819), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Cluster Lifecycle, Network, Release and Testing]
- Webhooks have alpha support for network proxy ([#85870](https://github.com/kubernetes/kubernetes/pull/85870), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Auth and Testing]
- When client certificate files are provided, reload files for new connections, and close connections when a certificate changes. ([#79083](https://github.com/kubernetes/kubernetes/pull/79083), [@jackkleeman](https://github.com/jackkleeman)) [SIG API Machinery, Auth, Node and Testing]
- When deleting objects using kubectl with the --force flag, you are no longer required to also specify --grace-period=0. ([#87776](https://github.com/kubernetes/kubernetes/pull/87776), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Windows nodes on GCE can use virtual TPM-based authentication to the control plane. ([#85466](https://github.com/kubernetes/kubernetes/pull/85466), [@pjh](https://github.com/pjh)) [SIG Cluster Lifecycle]
- You can now pass "--node-ip ::" to kubelet to indicate that it should autodetect an IPv6 address to use as the node's primary address. ([#85850](https://github.com/kubernetes/kubernetes/pull/85850), [@danwinship](https://github.com/danwinship)) [SIG Cloud Provider, Network and Node]
- `kubectl` now contains a `kubectl alpha debug` command. This command allows attaching an ephemeral container to a running pod for the purposes of debugging. ([#88004](https://github.com/kubernetes/kubernetes/pull/88004), [@verb](https://github.com/verb)) [SIG CLI]
- TLS Server Name overrides can now be specified in a kubeconfig file and via --tls-server-name in kubectl ([#88769](https://github.com/kubernetes/kubernetes/pull/88769), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Auth and CLI]

#### Metrics:
- Add `rest_client_rate_limiter_duration_seconds` metric to component-base to track client side rate limiter latency in seconds. Broken down by verb and URL. ([#88134](https://github.com/kubernetes/kubernetes/pull/88134), [@jennybuckley](https://github.com/jennybuckley)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Added two client certificate metrics for exec auth:
    - `rest_client_certificate_expiration_seconds` a gauge reporting the lifetime of the current client certificate. Reports the time of expiry in seconds since January 1, 1970 UTC.
    - `rest_client_certificate_rotation_age` a histogram reporting the age of a just rotated client certificate in seconds. ([#84382](https://github.com/kubernetes/kubernetes/pull/84382), [@sambdavidson](https://github.com/sambdavidson)) [SIG API Machinery, Auth, Cluster Lifecycle and Instrumentation]
- Controller manager serve workqueue metrics ([#87967](https://github.com/kubernetes/kubernetes/pull/87967), [@zhan849](https://github.com/zhan849)) [SIG API Machinery]
- Following metrics have been turned off:
  - kubelet_pod_worker_latency_microseconds
  - kubelet_pod_start_latency_microseconds
  - kubelet_cgroup_manager_latency_microseconds
  - kubelet_pod_worker_start_latency_microseconds
  - kubelet_pleg_relist_latency_microseconds
  - kubelet_pleg_relist_interval_microseconds
  - kubelet_eviction_stats_age_microseconds
  - kubelet_runtime_operations
  - kubelet_runtime_operations_latency_microseconds
  - kubelet_runtime_operations_errors
  - kubelet_device_plugin_registration_count
  - kubelet_device_plugin_alloc_latency_microseconds
  - kubelet_docker_operations
  - kubelet_docker_operations_latency_microseconds
  - kubelet_docker_operations_errors
  - kubelet_docker_operations_timeout
  - network_plugin_operations_latency_microseconds ([#83841](https://github.com/kubernetes/kubernetes/pull/83841), [@RainbowMango](https://github.com/RainbowMango)) [SIG Network and Node]
- Kube-apiserver metrics will now include request counts, latencies, and response sizes for /healthz, /livez, and /readyz requests. ([#83598](https://github.com/kubernetes/kubernetes/pull/83598), [@jktomer](https://github.com/jktomer)) [SIG API Machinery]
- Kubelet now exports a `server_expiration_renew_failure` and `client_expiration_renew_failure` metric counter if the certificate rotations cannot be performed. ([#84614](https://github.com/kubernetes/kubernetes/pull/84614), [@rphillips](https://github.com/rphillips)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Release]
- Kubelet: the metric process_start_time_seconds be marked as with the ALPHA stability level. ([#85446](https://github.com/kubernetes/kubernetes/pull/85446), [@RainbowMango](https://github.com/RainbowMango)) [SIG API Machinery, Cluster Lifecycle, Instrumentation and Node]
- New metric `kubelet_pleg_last_seen_seconds` to aid diagnosis of PLEG not healthy issues. ([#86251](https://github.com/kubernetes/kubernetes/pull/86251), [@bboreham](https://github.com/bboreham)) [SIG Node]

### Other (Bug, Cleanup or Flake)

- Fixed a regression with clients prior to 1.15 not being able to update podIP in pod status, or podCIDR in node spec, against >= 1.16 API servers ([#88505](https://github.com/kubernetes/kubernetes/pull/88505), [@liggitt](https://github.com/liggitt)) [SIG Apps and Network]
- Fixed "kubectl describe statefulsets.apps" printing garbage for rolling update partition ([#85846](https://github.com/kubernetes/kubernetes/pull/85846), [@phil9909](https://github.com/phil9909)) [SIG CLI]
- Add a event to PV when filesystem on PV does not match actual filesystem on disk ([#86982](https://github.com/kubernetes/kubernetes/pull/86982), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Add azure disk WriteAccelerator support ([#87945](https://github.com/kubernetes/kubernetes/pull/87945), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Add delays between goroutines for vm instance update ([#88094](https://github.com/kubernetes/kubernetes/pull/88094), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Add init containers log to cluster dump info. ([#88324](https://github.com/kubernetes/kubernetes/pull/88324), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Addons: elasticsearch discovery supports IPv6 ([#85543](https://github.com/kubernetes/kubernetes/pull/85543), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle and Instrumentation]
- Adds "volume.beta.kubernetes.io/migrated-to" annotation to PV's and PVC's when they are migrated to signal external provisioners to pick up those objects for Provisioning and Deleting. ([#87098](https://github.com/kubernetes/kubernetes/pull/87098), [@davidz627](https://github.com/davidz627)) [SIG  Storage]
- All api-server log request lines in a more greppable format. ([#87203](https://github.com/kubernetes/kubernetes/pull/87203), [@lavalamp](https://github.com/lavalamp)) [SIG API Machinery]
- Azure VMSS LoadBalancerBackendAddressPools updating has been improved with sequential-sync + concurrent-async requests. ([#88699](https://github.com/kubernetes/kubernetes/pull/88699), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Azure cloud provider now obtains AAD token who audience claim will not have spn: prefix ([#87590](https://github.com/kubernetes/kubernetes/pull/87590), [@weinong](https://github.com/weinong)) [SIG Cloud Provider]
- AzureFile and CephFS use the new Mount library that prevents logging of sensitive mount options. ([#88684](https://github.com/kubernetes/kubernetes/pull/88684), [@saad-ali](https://github.com/saad-ali)) [SIG Storage]
- Bind dns-horizontal containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83364](https://github.com/kubernetes/kubernetes/pull/83364), [@wawa0210](https://github.com/wawa0210)) [SIG Cluster Lifecycle and Windows]
- Bind kube-dns containers to linux nodes to avoid Windows scheduling ([#83358](https://github.com/kubernetes/kubernetes/pull/83358), [@wawa0210](https://github.com/wawa0210)) [SIG Cluster Lifecycle and Windows]
- Bind metadata-agent containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83363](https://github.com/kubernetes/kubernetes/pull/83363), [@wawa0210](https://github.com/wawa0210)) [SIG Cluster Lifecycle, Instrumentation and Windows]
- Bind metrics-server containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83362](https://github.com/kubernetes/kubernetes/pull/83362), [@wawa0210](https://github.com/wawa0210)) [SIG Cluster Lifecycle, Instrumentation and Windows]
- Bug fixes: Make sure we include latest packages node &#35;351 (@caseydavenport) ([#84163](https://github.com/kubernetes/kubernetes/pull/84163), [@david-tigera](https://github.com/david-tigera)) [SIG Cluster Lifecycle]
- CPU limits are now respected for Windows containers. If a node is over-provisioned, no weighting is used, only limits are respected. ([#86101](https://github.com/kubernetes/kubernetes/pull/86101), [@PatrickLang](https://github.com/PatrickLang)) [SIG Node, Testing and Windows]
- Changed core_pattern on COS nodes to be an absolute path. ([#86329](https://github.com/kubernetes/kubernetes/pull/86329), [@mml](https://github.com/mml)) [SIG Cluster Lifecycle and Node]
- Client-go certificate manager rotation gained the ability to preserve optional intermediate chains accompanying issued certificates ([#88744](https://github.com/kubernetes/kubernetes/pull/88744), [@jackkleeman](https://github.com/jackkleeman)) [SIG API Machinery and Auth]
- Cloud provider config CloudProviderBackoffMode has been removed since it won't be used anymore. ([#88463](https://github.com/kubernetes/kubernetes/pull/88463), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Conformance image now depends on stretch-slim instead of debian-hyperkube-base as that image is being deprecated and removed. ([#88702](https://github.com/kubernetes/kubernetes/pull/88702), [@dims](https://github.com/dims)) [SIG Cluster Lifecycle, Release and Testing]
- Deprecate --generator flag from kubectl create commands ([#88655](https://github.com/kubernetes/kubernetes/pull/88655), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- During initialization phase (preflight), kubeadm now verifies the presence of the conntrack executable ([#85857](https://github.com/kubernetes/kubernetes/pull/85857), [@hnanni](https://github.com/hnanni)) [SIG Cluster Lifecycle]
- EndpointSlice should not contain endpoints for terminating pods ([#89056](https://github.com/kubernetes/kubernetes/pull/89056), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
- Evictions due to pods breaching their ephemeral storage limits are now recorded by the `kubelet_evictions` metric and can be alerted on. ([#87906](https://github.com/kubernetes/kubernetes/pull/87906), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Filter published OpenAPI schema by making nullable, required fields non-required in order to avoid kubectl to wrongly reject null values. ([#85722](https://github.com/kubernetes/kubernetes/pull/85722), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Fix /readyz to return error immediately after a shutdown is initiated, before the --shutdown-delay-duration has elapsed. ([#88911](https://github.com/kubernetes/kubernetes/pull/88911), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fix API Server potential memory leak issue in processing watch request. ([#85410](https://github.com/kubernetes/kubernetes/pull/85410), [@answer1991](https://github.com/answer1991)) [SIG API Machinery]
- Fix EndpointSlice controller race condition and ensure that it handles external changes to EndpointSlices. ([#85703](https://github.com/kubernetes/kubernetes/pull/85703), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Fix IPv6 addresses lost issue in pure ipv6 vsphere environment ([#86001](https://github.com/kubernetes/kubernetes/pull/86001), [@hubv](https://github.com/hubv)) [SIG Cloud Provider]
- Fix LoadBalancer rule checking so that no unexpected LoadBalancer updates are made ([#85990](https://github.com/kubernetes/kubernetes/pull/85990), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix a bug in kube-proxy that caused it to crash when using load balancers with a different IP family ([#87117](https://github.com/kubernetes/kubernetes/pull/87117), [@aojea](https://github.com/aojea)) [SIG Network]
- Fix a bug in port-forward: named port not working with service ([#85511](https://github.com/kubernetes/kubernetes/pull/85511), [@oke-py](https://github.com/oke-py)) [SIG CLI]
- Fix a bug in the dual-stack IPVS proxier where stale IPv6 endpoints were not being cleaned up ([#87695](https://github.com/kubernetes/kubernetes/pull/87695), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network]
- Fix a bug that orphan revision cannot be adopted and statefulset cannot be synced ([#86801](https://github.com/kubernetes/kubernetes/pull/86801), [@likakuli](https://github.com/likakuli)) [SIG Apps]
- Fix a bug where ExternalTrafficPolicy is not applied to service ExternalIPs. ([#88786](https://github.com/kubernetes/kubernetes/pull/88786), [@freehan](https://github.com/freehan)) [SIG Network]
- Fix a bug where kubenet fails to parse the tc output. ([#83572](https://github.com/kubernetes/kubernetes/pull/83572), [@chendotjs](https://github.com/chendotjs)) [SIG Network]
- Fix a regression in kubenet that prevent pods to obtain ip addresses ([#85993](https://github.com/kubernetes/kubernetes/pull/85993), [@chendotjs](https://github.com/chendotjs)) [SIG Network and Node]
- Fix azure file AuthorizationFailure ([#85475](https://github.com/kubernetes/kubernetes/pull/85475), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix bug where EndpointSlice controller would attempt to modify shared objects. ([#85368](https://github.com/kubernetes/kubernetes/pull/85368), [@robscott](https://github.com/robscott)) [SIG API Machinery, Apps and Network]
- Fix handling of aws-load-balancer-security-groups annotation. Security-Groups assigned with this annotation are no longer modified by kubernetes which is the expected behaviour of most users. Also no unnecessary Security-Groups are created anymore if this annotation is used. ([#83446](https://github.com/kubernetes/kubernetes/pull/83446), [@Elias481](https://github.com/Elias481)) [SIG Cloud Provider]
- Fix invalid VMSS updates due to incorrect cache ([#89002](https://github.com/kubernetes/kubernetes/pull/89002), [@ArchangelSDY](https://github.com/ArchangelSDY)) [SIG Cloud Provider]
- Fix isCurrentInstance for Windows by removing the dependency of hostname. ([#89138](https://github.com/kubernetes/kubernetes/pull/89138), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix issue #85805 about a resource not found in azure cloud provider when LoadBalancer specified in another resource group. ([#86502](https://github.com/kubernetes/kubernetes/pull/86502), [@levimm](https://github.com/levimm)) [SIG Cloud Provider]
- Fix kubectl annotate error when local=true is set ([#86952](https://github.com/kubernetes/kubernetes/pull/86952), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix kubectl create deployment image name ([#86636](https://github.com/kubernetes/kubernetes/pull/86636), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix `kubectl drain ignore` daemonsets and others. ([#87361](https://github.com/kubernetes/kubernetes/pull/87361), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix missing "apiVersion" for "involvedObject" in Events for Nodes. ([#87537](https://github.com/kubernetes/kubernetes/pull/87537), [@uthark](https://github.com/uthark)) [SIG Apps and Node]
- Fix nil pointer dereference in azure cloud provider ([#85975](https://github.com/kubernetes/kubernetes/pull/85975), [@ldx](https://github.com/ldx)) [SIG Cloud Provider]
- Fix regression in statefulset conversion which prevents applying a statefulset multiple times. ([#87706](https://github.com/kubernetes/kubernetes/pull/87706), [@liggitt](https://github.com/liggitt)) [SIG Apps and Testing]
- Fix route conflicted operations when updating multiple routes together ([#88209](https://github.com/kubernetes/kubernetes/pull/88209), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix that prevents repeated fetching of PVC/PV objects by kubelet when processing of pod volumes fails. While this prevents hammering API server in these error scenarios, it means that some errors in processing volume(s) for a pod could now take up to 2-3 minutes before retry. ([#88141](https://github.com/kubernetes/kubernetes/pull/88141), [@tedyu](https://github.com/tedyu)) [SIG Node and Storage]
- Fix the bug PIP's DNS is deleted if no DNS label service annotation isn't set. ([#87246](https://github.com/kubernetes/kubernetes/pull/87246), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fix control plane hosts rolling upgrade causing thundering herd of LISTs on etcd leading to control plane unavailability. ([#86430](https://github.com/kubernetes/kubernetes/pull/86430), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Node and Testing]
- Fix: add azure disk migration support for CSINode ([#88014](https://github.com/kubernetes/kubernetes/pull/88014), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: add non-retriable errors in azure clients ([#87941](https://github.com/kubernetes/kubernetes/pull/87941), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: add remediation in azure disk attach/detach ([#88444](https://github.com/kubernetes/kubernetes/pull/88444), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: azure data disk should use same key as os disk by default ([#86351](https://github.com/kubernetes/kubernetes/pull/86351), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: azure disk could not mounted on Standard_DC4s/DC2s instances ([#86612](https://github.com/kubernetes/kubernetes/pull/86612), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: azure file mount timeout issue ([#88610](https://github.com/kubernetes/kubernetes/pull/88610), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: check disk status before disk azure disk ([#88360](https://github.com/kubernetes/kubernetes/pull/88360), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: corrupted mount point in csi driver ([#88569](https://github.com/kubernetes/kubernetes/pull/88569), [@andyzhangx](https://github.com/andyzhangx)) [SIG Storage]
- Fix: get azure disk lun timeout issue ([#88158](https://github.com/kubernetes/kubernetes/pull/88158), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: update azure disk max count ([#88201](https://github.com/kubernetes/kubernetes/pull/88201), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fixed "requested device X but found Y" attach error on AWS. ([#85675](https://github.com/kubernetes/kubernetes/pull/85675), [@jsafrane](https://github.com/jsafrane)) [SIG Cloud Provider and Storage]
- Fixed NetworkPolicy validation that `Except` values are accepted when they are outside the CIDR range. ([#86578](https://github.com/kubernetes/kubernetes/pull/86578), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Fixed a bug in the TopologyManager. Previously, the TopologyManager would only guarantee alignment if container creation was serialized in some way. Alignment is now guaranteed under all scenarios of container creation. ([#87759](https://github.com/kubernetes/kubernetes/pull/87759), [@klueska](https://github.com/klueska)) [SIG Node]
- Fixed a bug which could prevent a provider ID from ever being set for node if an error occurred determining the provider ID when the node was added. ([#87043](https://github.com/kubernetes/kubernetes/pull/87043), [@zjs](https://github.com/zjs)) [SIG Apps and Cloud Provider]
- Fixed a data race in the kubelet image manager that can cause static pod workers to silently stop working. ([#88915](https://github.com/kubernetes/kubernetes/pull/88915), [@roycaihw](https://github.com/roycaihw)) [SIG Node]
- Fixed a panic in the kubelet cleaning up pod volumes ([#86277](https://github.com/kubernetes/kubernetes/pull/86277), [@tedyu](https://github.com/tedyu)) [SIG Storage]
- Fixed a regression where the kubelet would fail to update the ready status of pods. ([#84951](https://github.com/kubernetes/kubernetes/pull/84951), [@tedyu](https://github.com/tedyu)) [SIG Node]
- Fixed an issue that could cause the kubelet to incorrectly run concurrent pod reconciliation loops and crash. ([#89055](https://github.com/kubernetes/kubernetes/pull/89055), [@tedyu](https://github.com/tedyu)) [SIG Node]
- Fixed block CSI volume cleanup after timeouts. ([#88660](https://github.com/kubernetes/kubernetes/pull/88660), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixed cleaning of CSI raw block volumes. ([#87978](https://github.com/kubernetes/kubernetes/pull/87978), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixed AWS Cloud Provider attempting to delete LoadBalancer security group it didn’t provision, and fixed AWS Cloud Provider creating a default LoadBalancer security group even if annotation `service.beta.kubernetes.io/aws-load-balancer-security-groups` is present because the intended behavior of aws-load-balancer-security-groups is to replace all security groups assigned to the load balancer. ([#84265](https://github.com/kubernetes/kubernetes/pull/84265), [@bhagwat070919](https://github.com/bhagwat070919)) [SIG Cloud Provider]
- Fixed two scheduler metrics (pending_pods and schedule_attempts_total) not being recorded ([#87692](https://github.com/kubernetes/kubernetes/pull/87692), [@everpeace](https://github.com/everpeace)) [SIG Scheduling]
- Fixes an issue with kubelet-reported pod status on deleted/recreated pods. ([#86320](https://github.com/kubernetes/kubernetes/pull/86320), [@liggitt](https://github.com/liggitt)) [SIG Node]
- Fixes conversion error in multi-version custom resources that could cause metadata.generation to increment on no-op patches or updates of a custom resource. ([#88995](https://github.com/kubernetes/kubernetes/pull/88995), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Fixes issue where AAD token obtained by kubectl is incompatible with on-behalf-of flow and oidc. The audience claim before this fix has "spn:" prefix. After this fix, "spn:" prefix is omitted. ([#86412](https://github.com/kubernetes/kubernetes/pull/86412), [@weinong](https://github.com/weinong)) [SIG API Machinery, Auth and Cloud Provider]
- Fixes an issue where you can't attach more than 15 GCE Persistent Disks to c2, n2, m1, m2 machine types. ([#88602](https://github.com/kubernetes/kubernetes/pull/88602), [@yuga711](https://github.com/yuga711)) [SIG Storage]
- Fixes kube-proxy when EndpointSlice feature gate is enabled on Windows. ([#86016](https://github.com/kubernetes/kubernetes/pull/86016), [@robscott](https://github.com/robscott)) [SIG Auth and Network]
- Fixes kubelet crash in client certificate rotation cases ([#88079](https://github.com/kubernetes/kubernetes/pull/88079), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth and Node]
- Fixes service account token admission error in clusters that do not run the service account token controller ([#87029](https://github.com/kubernetes/kubernetes/pull/87029), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Fixes v1.17.0 regression in --service-cluster-ip-range handling with IPv4 ranges larger than 65536 IP addresses ([#86534](https://github.com/kubernetes/kubernetes/pull/86534), [@liggitt](https://github.com/liggitt)) [SIG Network]
- Fixes wrong validation result of NetworkPolicy PolicyTypes ([#85747](https://github.com/kubernetes/kubernetes/pull/85747), [@tnqn](https://github.com/tnqn)) [SIG Network]
- For subprotocol negotiation, both client and server protocol is required now. ([#86646](https://github.com/kubernetes/kubernetes/pull/86646), [@tedyu](https://github.com/tedyu)) [SIG API Machinery and Node]
- For volumes that allow attaches across multiple nodes, attach and detach operations across different nodes are now executed in parallel. ([#88678](https://github.com/kubernetes/kubernetes/pull/88678), [@verult](https://github.com/verult)) [SIG Storage]
- Garbage collector now can correctly orphan ControllerRevisions when StatefulSets are deleted with orphan propagation policy. ([#84984](https://github.com/kubernetes/kubernetes/pull/84984), [@cofyc](https://github.com/cofyc)) [SIG Apps]
- `Get-kube.sh` uses the gcloud's current local GCP service account for auth when the provider is GCE or GKE instead of the metadata server default ([#88383](https://github.com/kubernetes/kubernetes/pull/88383), [@BenTheElder](https://github.com/BenTheElder)) [SIG Cluster Lifecycle]
- Golang/x/net has been updated to bring in fixes for CVE-2020-9283 ([#88381](https://github.com/kubernetes/kubernetes/pull/88381), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- If a serving certificate’s param specifies a name that is an IP for an SNI certificate, it will have priority for replying to server connections. ([#85308](https://github.com/kubernetes/kubernetes/pull/85308), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Improved yaml parsing performance ([#85458](https://github.com/kubernetes/kubernetes/pull/85458), [@cjcullen](https://github.com/cjcullen)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Node]
- Improves performance of the node authorizer ([#87696](https://github.com/kubernetes/kubernetes/pull/87696), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- In GKE alpha clusters it will be possible to use the service annotation `cloud.google.com/network-tier: Standard` ([#88487](https://github.com/kubernetes/kubernetes/pull/88487), [@zioproto](https://github.com/zioproto)) [SIG Cloud Provider]
- Includes FSType when describing CSI persistent volumes. ([#85293](https://github.com/kubernetes/kubernetes/pull/85293), [@huffmanca](https://github.com/huffmanca)) [SIG CLI and Storage]
- Iptables/userspace proxy: improve performance by getting local addresses only once per sync loop, instead of for every external IP ([#85617](https://github.com/kubernetes/kubernetes/pull/85617), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Network]
- Kube-aggregator: always sets unavailableGauge metric to reflect the current state of a service. ([#87778](https://github.com/kubernetes/kubernetes/pull/87778), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kube-apiserver: fixed a conflict error encountered attempting to delete a pod with gracePeriodSeconds=0 and a resourceVersion precondition ([#85516](https://github.com/kubernetes/kubernetes/pull/85516), [@michaelgugino](https://github.com/michaelgugino)) [SIG API Machinery]
- Kube-proxy no longer modifies shared EndpointSlices. ([#86092](https://github.com/kubernetes/kubernetes/pull/86092), [@robscott](https://github.com/robscott)) [SIG Network]
- Kube-proxy: on dual-stack mode, if it is not able to get the IP Family of an endpoint, logs it with level InfoV(4) instead of Warning, avoiding flooding the logs for endpoints without addresses ([#88934](https://github.com/kubernetes/kubernetes/pull/88934), [@aojea](https://github.com/aojea)) [SIG Network]
- Kubeadm allows to configure single-stack clusters if dual-stack is enabled ([#87453](https://github.com/kubernetes/kubernetes/pull/87453), [@aojea](https://github.com/aojea)) [SIG API Machinery, Cluster Lifecycle and Network]
- Kubeadm now includes CoreDNS version 1.6.7 ([#86260](https://github.com/kubernetes/kubernetes/pull/86260), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cluster Lifecycle]
- Kubeadm upgrades always persist the etcd backup for stacked ([#86861](https://github.com/kubernetes/kubernetes/pull/86861), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm:  'kubeadm alpha kubelet config download' has been removed, please use 'kubeadm upgrade node phase kubelet-config' instead ([#87944](https://github.com/kubernetes/kubernetes/pull/87944), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: Forward cluster name to the controller-manager arguments ([#85817](https://github.com/kubernetes/kubernetes/pull/85817), [@ereslibre](https://github.com/ereslibre)) [SIG Cluster Lifecycle]
- Kubeadm: add support for the "ci/k8s-master" version label as a replacement for "ci-cross/*", which no longer exists. ([#86609](https://github.com/kubernetes/kubernetes/pull/86609), [@Pensu](https://github.com/Pensu)) [SIG Cluster Lifecycle]
- Kubeadm: apply further improvements to the tentative support for concurrent etcd member join. Fixes a bug where multiple members can receive the same hostname. Increase the etcd client dial timeout and retry timeout for add/remove/... operations. ([#87505](https://github.com/kubernetes/kubernetes/pull/87505), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: don't write the kubelet environment file on "upgrade apply" ([#85412](https://github.com/kubernetes/kubernetes/pull/85412), [@boluisa](https://github.com/boluisa)) [SIG Cluster Lifecycle]
- Kubeadm: fix potential panic when executing "kubeadm reset" with a corrupted kubelet.conf file ([#86216](https://github.com/kubernetes/kubernetes/pull/86216), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that 'kubeadm upgrade' hangs in single node cluster ([#88434](https://github.com/kubernetes/kubernetes/pull/88434), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: make sure images are pre-pulled even if a tag did not change but their contents changed ([#85603](https://github.com/kubernetes/kubernetes/pull/85603), [@bart0sh](https://github.com/bart0sh)) [SIG Cluster Lifecycle]
- Kubeadm: remove 'kubeadm upgrade node config' command since it was deprecated in v1.15, please use 'kubeadm upgrade node phase kubelet-config' instead ([#87975](https://github.com/kubernetes/kubernetes/pull/87975), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated CoreDNS feature-gate. It was set to "true" since v1.11 when the feature went GA. In v1.13 it was marked as deprecated and hidden from the CLI. ([#87400](https://github.com/kubernetes/kubernetes/pull/87400), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: retry `kubeadm-config` ConfigMap creation or mutation if the apiserver is not responding. This will improve resiliency when joining new control plane nodes. ([#85763](https://github.com/kubernetes/kubernetes/pull/85763), [@ereslibre](https://github.com/ereslibre)) [SIG Cluster Lifecycle]
- Kubeadm: tolerate whitespace when validating certificate authority PEM data in kubeconfig files ([#86705](https://github.com/kubernetes/kubernetes/pull/86705), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: use bind-address option to configure the kube-controller-manager and kube-scheduler http probes ([#86493](https://github.com/kubernetes/kubernetes/pull/86493), [@aojea](https://github.com/aojea)) [SIG Cluster Lifecycle]
- Kubeadm: uses the api-server AdvertiseAddress IP family to choose the etcd endpoint IP family for non external etcd clusters ([#85745](https://github.com/kubernetes/kubernetes/pull/85745), [@aojea](https://github.com/aojea)) [SIG Cluster Lifecycle]
- Kubectl cluster-info dump --output-directory=xxx now generates files with an extension depending on the output format. ([#82070](https://github.com/kubernetes/kubernetes/pull/82070), [@olivierlemasle](https://github.com/olivierlemasle)) [SIG CLI]
- `Kubectl describe <type>` and `kubectl top pod` will return a message saying `"No resources found"` or `"No resources found in <namespace> namespace"` if there are no results to display. ([#87527](https://github.com/kubernetes/kubernetes/pull/87527), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- `Kubectl drain node --dry-run` will list pods that would be evicted or deleted ([#82660](https://github.com/kubernetes/kubernetes/pull/82660), [@sallyom](https://github.com/sallyom)) [SIG CLI]
- `Kubectl set resources` will no longer return an error if passed an empty change for a resource. `kubectl set subject` will no longer return an error if passed an empty change for a resource. ([#85490](https://github.com/kubernetes/kubernetes/pull/85490), [@sallyom](https://github.com/sallyom)) [SIG CLI]
- Kubelet metrics gathered through metrics-server or prometheus should no longer timeout for Windows nodes running more than 3 pods. ([#87730](https://github.com/kubernetes/kubernetes/pull/87730), [@marosset](https://github.com/marosset)) [SIG Node, Testing and Windows]
- Kubelet metrics have been changed to buckets. For example the `exec/{podNamespace}/{podID}/{containerName}` is now just exec. ([#87913](https://github.com/kubernetes/kubernetes/pull/87913), [@cheftako](https://github.com/cheftako)) [SIG Node]
- Kubelets perform fewer unnecessary pod status update operations on the API server. ([#88591](https://github.com/kubernetes/kubernetes/pull/88591), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Scalability]
- Kubernetes will try to acquire the iptables lock every 100 msec during 5 seconds instead of every second. This is especially useful for environments using kube-proxy in iptables mode with a high churn rate of services. ([#85771](https://github.com/kubernetes/kubernetes/pull/85771), [@aojea](https://github.com/aojea)) [SIG Network]
- Limit number of instances in a single update to GCE target pool to 1000. ([#87881](https://github.com/kubernetes/kubernetes/pull/87881), [@wojtek-t](https://github.com/wojtek-t)) [SIG Cloud Provider, Network and Scalability]
- Make Azure clients only retry on specified HTTP status codes ([#88017](https://github.com/kubernetes/kubernetes/pull/88017), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Make error message and service event message more clear ([#86078](https://github.com/kubernetes/kubernetes/pull/86078), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Minimize AWS NLB health check timeout when externalTrafficPolicy set to Local ([#73363](https://github.com/kubernetes/kubernetes/pull/73363), [@kellycampbell](https://github.com/kellycampbell)) [SIG Cloud Provider]
- Pause image contains "Architecture" in non-amd64 images ([#87954](https://github.com/kubernetes/kubernetes/pull/87954), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Pause image upgraded to 3.2 in kubelet and kubeadm. ([#88173](https://github.com/kubernetes/kubernetes/pull/88173), [@BenTheElder](https://github.com/BenTheElder)) [SIG CLI, Cluster Lifecycle, Node and Testing]
- Plugin/PluginConfig and Policy APIs are mutually exclusive when running the scheduler ([#88864](https://github.com/kubernetes/kubernetes/pull/88864), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Remove `FilteredNodesStatuses` argument from `PreScore`'s interface. ([#88189](https://github.com/kubernetes/kubernetes/pull/88189), [@skilxn-go](https://github.com/skilxn-go)) [SIG Scheduling and Testing]
- Resolved a performance issue in the node authorizer index maintenance. ([#87693](https://github.com/kubernetes/kubernetes/pull/87693), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Resolved regression in admission, authentication, and authorization webhook performance in v1.17.0-rc.1 ([#85810](https://github.com/kubernetes/kubernetes/pull/85810), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Resolves performance regression in `kubectl get all` and in client-go discovery clients constructed using `NewDiscoveryClientForConfig` or `NewDiscoveryClientForConfigOrDie`. ([#86168](https://github.com/kubernetes/kubernetes/pull/86168), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Reverted a kubectl azure auth module change where oidc claim spn: prefix was omitted resulting a breaking behavior with existing Azure AD OIDC enabled api-server ([#87507](https://github.com/kubernetes/kubernetes/pull/87507), [@weinong](https://github.com/weinong)) [SIG API Machinery, Auth and Cloud Provider]
- Shared informers are now more reliable in the face of network disruption. ([#86015](https://github.com/kubernetes/kubernetes/pull/86015), [@squeed](https://github.com/squeed)) [SIG API Machinery]
- Specifying PluginConfig for the same plugin more than once fails scheduler startup.
  Specifying extenders and configuring .ignoredResources for the NodeResourcesFit plugin fails ([#88870](https://github.com/kubernetes/kubernetes/pull/88870), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Terminating a restartPolicy=Never pod no longer has a chance to report the pod succeeded when it actually failed. ([#88440](https://github.com/kubernetes/kubernetes/pull/88440), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- The CSR signing cert/key pairs will be reloaded from disk like the kube-apiserver cert/key pairs ([#86816](https://github.com/kubernetes/kubernetes/pull/86816), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Apps and Auth]
- The EventRecorder from k8s.io/client-go/tools/events will now create events in the default namespace (instead of kube-system) when the related object does not have it set. ([#88815](https://github.com/kubernetes/kubernetes/pull/88815), [@enj](https://github.com/enj)) [SIG API Machinery]
- The audit event sourceIPs list will now always end with the IP that sent the request directly to the API server. ([#87167](https://github.com/kubernetes/kubernetes/pull/87167), [@tallclair](https://github.com/tallclair)) [SIG API Machinery and Auth]
- The sample-apiserver aggregated conformance test has updated to use the Kubernetes v1.17.0 sample apiserver ([#84735](https://github.com/kubernetes/kubernetes/pull/84735), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Architecture, CLI and Testing]
- To reduce chances of throttling, VM cache is set to nil when Azure node provisioning state is deleting ([#87635](https://github.com/kubernetes/kubernetes/pull/87635), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- VMSS cache is added so that less chances of VMSS GET throttling ([#85885](https://github.com/kubernetes/kubernetes/pull/85885), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Wait for kubelet & kube-proxy to be ready on Windows node within 10s ([#85228](https://github.com/kubernetes/kubernetes/pull/85228), [@YangLu1031](https://github.com/YangLu1031)) [SIG Cluster Lifecycle]
- `kubectl apply  -f <file> --prune -n <namespace>` should prune all resources not defined in the file in the cli specified namespace. ([#85613](https://github.com/kubernetes/kubernetes/pull/85613), [@MartinKaburu](https://github.com/MartinKaburu)) [SIG CLI]
- `kubectl create clusterrolebinding` creates rbac.authorization.k8s.io/v1 object ([#85889](https://github.com/kubernetes/kubernetes/pull/85889), [@oke-py](https://github.com/oke-py)) [SIG CLI]
- `kubectl diff` now returns 1 only on diff finding changes, and >1 on kubectl errors. The "exit status code 1" message has also been muted. ([#87437](https://github.com/kubernetes/kubernetes/pull/87437), [@apelisse](https://github.com/apelisse)) [SIG CLI and Testing]

## Dependencies

- Update Calico to v3.8.4 ([#84163](https://github.com/kubernetes/kubernetes/pull/84163), [@david-tigera](https://github.com/david-tigera))[SIG Cluster Lifecycle]
- Update aws-sdk-go dependency to v1.28.2 ([#87253](https://github.com/kubernetes/kubernetes/pull/87253), [@SaranBalaji90](https://github.com/SaranBalaji90))[SIG API Machinery and Cloud Provider]
- Update CNI version to v0.8.5 ([#78819](https://github.com/kubernetes/kubernetes/pull/78819), [@justaugustus](https://github.com/justaugustus))[SIG Release, Testing, Network, Cluster Lifecycle and API Machinery]
- Update cri-tools to v1.17.0 ([#86305](https://github.com/kubernetes/kubernetes/pull/86305), [@saschagrunert](https://github.com/saschagrunert))[SIG Release and Cluster Lifecycle]
- Pause image upgraded to 3.2 in kubelet and kubeadm ([#88173](https://github.com/kubernetes/kubernetes/pull/88173), [@BenTheElder](https://github.com/BenTheElder))[SIG CLI, Node, Testing and Cluster Lifecycle]
- Update CoreDNS version to 1.6.7 in kubeadm ([#86260](https://github.com/kubernetes/kubernetes/pull/86260), [@rajansandeep](https://github.com/rajansandeep))[SIG Cluster Lifecycle]
- Update golang.org/x/crypto to fix CVE-2020-9283 ([#8838](https://github.com/kubernetes/kubernetes/pull/88381), [@BenTheElder](https://github.com/BenTheElder))[SIG CLI, Instrumentation, API Machinery, CLuster Lifecycle and Cloud Provider]
- Update Go to 1.13.8 ([#87648](https://github.com/kubernetes/kubernetes/pull/87648), [@ialidzhikov](https://github.com/ialidzhikov))[SIG Release and Testing]
- Update Cluster-Autoscaler to 1.18.0 ([#89095](https://github.com/kubernetes/kubernetes/pull/89095), [@losipiuk](https://github.com/losipiuk))[SIG Autoscaling and Cluster Lifecycle]



# v1.18.0-rc.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-rc.1

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes.tar.gz) | `c17231d5de2e0677e8af8259baa11a388625821c79b86362049f2edb366404d6f4b4587b8f13ccbceeb2f32c6a9fe98607f779c0f3e1caec438f002e3a2c8c21`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-src.tar.gz) | `e84ffad57c301f5d6e90f916b996d5abb0c987928c3ca6b1565f7b042588f839b994ca12c43fc36f0ffb63f9fabc15110eb08be253b8939f49cd951e956da618`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-darwin-386.tar.gz) | `1aea99923d492436b3eb91aaecffac94e5d0aa2b38a0930d266fda85c665bbc4569745c409aa302247df3b578ce60324e7a489eb26240e97d4e65a67428ea3d1`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | `07fa7340a959740bd52b83ff44438bbd988e235277dad1e43f125f08ac85230a24a3b755f4e4c8645743444fa2b66a3602fc445d7da6d2fc3770e8c21ba24b33`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-linux-386.tar.gz) | `48cebd26448fdd47aa36257baa4c716a98fda055bbf6a05230f2a3fe3c1b99b4e483668661415392190f3eebb9cb6e15c784626b48bb2541d93a37902f0e3974`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | `c3a5fedf263f07a07f59c01fea6c63c1e0b76ee8dc67c45b6c134255c28ed69171ccc2f91b6a45d6a8ec5570a0a7562e24c33b9d7b0d1a864f4dc04b178b3c04`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-linux-arm.tar.gz) | `a6b11a55bd38583bbaac14931a6862f8ce6493afe30947ba29e5556654a571593358278df59412bbeb6888fa127e9ae4c0047a9d46cb59394995010796df6b14`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | `9e15331ac8010154a9b64f5488969fc8ee2f21059639896cb84c5cf4f05f4c9d1d8970cb6f9831de6b34013848227c1972c12a698d07aac1ecc056e972fe6f79`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | `f828fe6252678de9d4822e482f5873309ae9139b2db87298ab3273ce45d38aa07b6b9b42b76c140705f27ba71e101d58b43e59ac7259d7c08dc647ea809e207c`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | `19da4b45f0666c063934af616f3e7ed3caa99d4ee1e46d53efadc7a8a4d38e43a36ced7249acd7ad3dcc4b4f60d8451b4f7ec7727e478ee2fadd14d353228bce`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-windows-386.tar.gz) | `775c9afb6cb3e7c4ba53e9f48a5df2cf207234a33059bd74448bc9f177dd120fb3f9c58ab45048a566326acc43bc8a67e886e10ef99f20780c8f63bb17426ebd`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | `208d2595a5b57ac97aac75b4a2a6130f0c937f781a030bde1a432daf4bc51f2fa523fca2eb84c38798489c4b536ee90aad22f7be8477985d9691d51ad8e1c4dc`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | `dcf832eae04f9f52ff473754ef5cfe697b35f4dc1a282622c94fa10943c8c35f4a8777a0c58c7de871c3c428c8973bf72d6bcd8751416d4c682125268b8fcefe`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-server-linux-arm.tar.gz) | `a04e34bea28eb1c8b492e8b1dd3c0dd87ebee71a7dbbef72be10a335e553361af7e48296e504f9844496b04e66350871114d20cfac3f3b49550d8be60f324ba3`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | `a6af086b07a8c2e498f32b43e6511bf6a5e6baf358c572c6910c8df17cd6cae94f562f459714fcead1595767cb14c7f639c5735f1411173bbd38d5604c082a77`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | `5a960ef5ba0c255f587f2ac0b028cd03136dc91e4efc5d1becab46417852e5524d18572b6f66259531ec6fea997da3c4d162ac153a9439672154375053fec6c7`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | `0f32c7d9b14bc238b9a5764d8f00edc4d3bf36bcf06b340b81061424e6070768962425194a8c2025c3a7ffb97b1de551d3ad23d1591ae34dd4e3ba25ab364c33`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | `27d8955d535d14f3f4dca501fd27e4f06fad84c6da878ea5332a5c83b6955667f6f731bfacaf5a3a23c09f14caa400f9bee927a0f269f5374de7f79cd1919b3b`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-node-linux-arm.tar.gz) | `0d56eccad63ba608335988e90b377fe8ae978b177dc836cdb803a5c99d99e8f3399a666d9477ca9cfe5964944993e85c416aec10a99323e3246141efc0b1cc9e`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | `79bb9be66f9e892d866b28e5cc838245818edb9706981fab6ccbff493181b341c1fcf6fe5d2342120a112eb93af413f5ba191cfba1ab4c4a8b0546a5ad8ec220`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | `3e9e2c6f9a2747d828069511dce8b4034c773c2d122f005f4508e22518055c1e055268d9d86773bbd26fbd2d887d783f408142c6c2f56ab2f2365236fd4d2635`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | `4f96e018c336fa13bb6df6f7217fe46a2b5c47f806f786499c429604ccba2ebe558503ab2c72f63250aa25b61dae2d166e4b80ae10f6ab37d714f87c1dcf6691`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | `ab110d76d506746af345e5897ef4f6993d5f53ac818ba69a334f3641047351aa63bfb3582841a9afca51dd0baff8b9010077d9c8ec85d2d69e4172b8d4b338b0`

## Changelog since v1.18.0-beta.2

## Changes by Kind

### API Change

- Removes ConfigMap as suggestion for IngressClass parameters ([#89093](https://github.com/kubernetes/kubernetes/pull/89093), [@robscott](https://github.com/robscott)) [SIG Network]

### Other (Bug, Cleanup or Flake)

- EndpointSlice should not contain endpoints for terminating pods ([#89056](https://github.com/kubernetes/kubernetes/pull/89056), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
- Fix a bug where ExternalTrafficPolicy is not applied to service ExternalIPs. ([#88786](https://github.com/kubernetes/kubernetes/pull/88786), [@freehan](https://github.com/freehan)) [SIG Network]
- Fix invalid VMSS updates due to incorrect cache ([#89002](https://github.com/kubernetes/kubernetes/pull/89002), [@ArchangelSDY](https://github.com/ArchangelSDY)) [SIG Cloud Provider]
- Fix isCurrentInstance for Windows by removing the dependency of hostname. ([#89138](https://github.com/kubernetes/kubernetes/pull/89138), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fixed a data race in kubelet image manager that can cause static pod workers to silently stop working. ([#88915](https://github.com/kubernetes/kubernetes/pull/88915), [@roycaihw](https://github.com/roycaihw)) [SIG Node]
- Fixed an issue that could cause the kubelet to incorrectly run concurrent pod reconciliation loops and crash. ([#89055](https://github.com/kubernetes/kubernetes/pull/89055), [@tedyu](https://github.com/tedyu)) [SIG Node]
- Kube-proxy: on dual-stack mode, if it is not able to get the IP Family of an endpoint, logs it with level InfoV(4) instead of Warning, avoiding flooding the logs for endpoints without addresses ([#88934](https://github.com/kubernetes/kubernetes/pull/88934), [@aojea](https://github.com/aojea)) [SIG Network]
- Update Cluster Autoscaler to 1.18.0; changelog: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.18.0 ([#89095](https://github.com/kubernetes/kubernetes/pull/89095), [@losipiuk](https://github.com/losipiuk)) [SIG Autoscaling and Cluster Lifecycle]


# v1.18.0-beta.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-beta.2

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes.tar.gz) | `3017430ca17f8a3523669b4a02c39cedfc6c48b07281bc0a67a9fbe9d76547b76f09529172cc01984765353a6134a43733b7315e0dff370bba2635dd2a6289af`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-src.tar.gz) | `c5fd60601380a99efff4458b1c9cf4dc02195f6f756b36e590e54dff68f7064daf32cf63980dddee13ef9dec7a60ad4eeb47a288083fdbbeeef4bc038384e9ea`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `7e49ede167b9271d4171e477fa21d267b2fb35f80869337d5b323198dc12f71b61441975bf925ad6e6cd7b61cbf6372d386417dc1e5c9b3c87ae651021c37237`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `3f5cdf0e85eee7d0773e0ae2df1c61329dea90e0da92b02dae1ffd101008dc4bade1c4951fc09f0cad306f0bcb7d16da8654334ddee43d5015913cc4ac8f3eda`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-linux-386.tar.gz) | `b67b41c11bfecb88017c33feee21735c56f24cf6f7851b63c752495fc0fb563cd417a67a81f46bca091f74dc00fca1f296e483d2e3dfe2004ea4b42e252d30b9`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `1fef2197cb80003e3a5c26f05e889af9d85fbbc23e27747944d2997ace4bfa28f3670b13c08f5e26b7e274176b4e2df89c1162aebd8b9506e63b39b311b2d405`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `84e5f4d9776490219ee94a84adccd5dfc7c0362eb330709771afcde95ec83f03d96fe7399eec218e47af0a1e6445e24d95e6f9c66c0882ef8233a09ff2022420`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `ba613b114e0cca32fa21a3d10f845aa2f215d3af54e775f917ff93919f7dd7075efe254e4047a85a1f4b817fc2bd78006c2e8873885f1208cbc02db99e2e2e25`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | `502a6938d8c4bbe04abbd19b59919d86765058ff72334848be4012cec493e0e7027c6cd950cf501367ac2026eea9f518110cb72d1c792322b396fc2f73d23217`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | `c24700e0ed2ef5c1d2dd282d638c88d90392ae90ea420837b39fd8e1cfc19525017325ccda71d8472fdaea174762208c09e1bba9bbc77c89deef6fac5e847ba2`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-windows-386.tar.gz) | `0d4c5a741b052f790c8b0923c9586ee9906225e51cf4dc8a56fc303d4d61bb5bf77fba9e65151dec7be854ff31da8fc2dcd3214563e1b4b9951e6af4aa643da4`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `841ef2e306c0c9593f04d9528ee019bf3b667761227d9afc1d6ca8bf1aa5631dc25f5fe13ff329c4bf0c816b971fd0dec808f879721e0f3bf51ce49772b38010`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `b373df2e6ef55215e712315a5508e85a39126bd81b7b93c6b6305238919a88c740077828a6f19bcd97141951048ef7a19806ef6b1c3e1772dbc45715c5fcb3af`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `b8103cb743c23076ce8dd7c2da01c8dd5a542fbac8480e82dc673139c8ee5ec4495ca33695e7a18dd36412cf1e18ed84c8de05042525ddd8e869fbdfa2766569`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `8f8f05cf64fb9c8d80cdcb4935b2d3e3edc48bdd303231ae12f93e3f4d979237490744a11e24ba7f52dbb017ca321a8e31624dcffa391b8afda3d02078767fa0`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | `b313b911c46f2ec129537407af3f165f238e48caeb4b9e530783ffa3659304a544ed02bef8ece715c279373b9fb2c781bd4475560e02c4b98a6d79837bc81938`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | `a1b6b06571141f507b12e5ef98efb88f4b6b9aba924722b2a74f11278d29a2972ab8290608360151d124608e6e24da0eb3516d484cb5fa12ff2987562f15964a`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-node-linux-amd64.tar.gz) | `20e02ca327543cddb2568ead3d5de164cbfb2914ab6416106d906bf12fcfbc4e55b13bea4d6a515e8feab038e2c929d72c4d6909dfd7881ba69fd1e8c772ab99`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-node-linux-arm.tar.gz) | `ecd817ef05d6284f9c6592b84b0a48ea31cf4487030c9fb36518474b2a33dad11b9c852774682e60e4e8b074e6bea7016584ca281dddbe2994da5eaf909025c0`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-node-linux-arm64.tar.gz) | `0020d32b7908ffd5055c8b26a8b3033e4702f89efcfffe3f6fcdb8a9921fa8eaaed4193c85597c24afd8c523662454f233521bb7055841a54c182521217ccc9d`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-node-linux-ppc64le.tar.gz) | `e065411d66d486e7793449c1b2f5a412510b913bf7f4e728c0a20e275642b7668957050dc266952cdff09acc391369ae6ac5230184db89af6823ba400745f2fc`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-node-linux-s390x.tar.gz) | `082ee90413beaaea41d6cbe9a18f7d783a95852607f3b94190e0ca12aacdd97d87e233b87117871bfb7d0a4b6302fbc7688549492a9bc50a2f43a5452504d3ce`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.2/kubernetes-node-windows-amd64.tar.gz) | `fb5aca0cc36be703f9d4033eababd581bac5de8399c50594db087a99ed4cb56e4920e960eb81d0132d696d094729254eeda2a5c0cb6e65e3abca6c8d61da579e`

## Changelog since v1.18.0-beta.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

- `kubectl` no longer defaults to `http://localhost:8080`.  If you own one of these legacy clusters, you are *strongly- encouraged to secure your server.   If you cannot secure your server, you can set `KUBERNETES_MASTER` if you were relying on that behavior and you're a client-go user. Set `--server`, `--kubeconfig` or `KUBECONFIG` to make it work in `kubectl`. ([#86173](https://github.com/kubernetes/kubernetes/pull/86173), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI and Testing]

## Changes by Kind

### Deprecation

- AlgorithmSource is removed from v1alpha2 Scheduler ComponentConfig ([#87999](https://github.com/kubernetes/kubernetes/pull/87999), [@damemi](https://github.com/damemi)) [SIG Scheduling]
- Kube-proxy: deprecate `--healthz-port` and `--metrics-port` flag, please use `--healthz-bind-address` and `--metrics-bind-address` instead ([#88512](https://github.com/kubernetes/kubernetes/pull/88512), [@SataQiu](https://github.com/SataQiu)) [SIG Network]
- Kubeadm: deprecate the usage of the experimental flag '--use-api' under the 'kubeadm alpha certs renew' command. ([#88827](https://github.com/kubernetes/kubernetes/pull/88827), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]

### API Change

- A new IngressClass resource has been added to enable better Ingress configuration. ([#88509](https://github.com/kubernetes/kubernetes/pull/88509), [@robscott](https://github.com/robscott)) [SIG API Machinery, Apps, CLI, Network, Node and Testing]
- Added GenericPVCDataSource feature gate to enable using arbitrary custom resources as the data source for a PVC. ([#88636](https://github.com/kubernetes/kubernetes/pull/88636), [@bswartz](https://github.com/bswartz)) [SIG Apps and Storage]
- Allow user to specify fsgroup permission change policy for pods ([#88488](https://github.com/kubernetes/kubernetes/pull/88488), [@gnufied](https://github.com/gnufied)) [SIG Apps and Storage]
- BlockVolume and CSIBlockVolume features are now GA. ([#88673](https://github.com/kubernetes/kubernetes/pull/88673), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Node and Storage]
- CustomResourceDefinition schemas that use `x-kubernetes-list-map-keys` to specify properties that uniquely identify list items must make those properties required or have a default value, to ensure those properties are present for all list items. See https://kubernetes.io/docs/reference/using-api/api-concepts/&#35;merge-strategy for details. ([#88076](https://github.com/kubernetes/kubernetes/pull/88076), [@eloyekunle](https://github.com/eloyekunle)) [SIG API Machinery and Testing]
- Fixes a regression with clients prior to 1.15 not being able to update podIP in pod status, or podCIDR in node spec, against >= 1.16 API servers ([#88505](https://github.com/kubernetes/kubernetes/pull/88505), [@liggitt](https://github.com/liggitt)) [SIG Apps and Network]
- Ingress: Add Exact and Prefix maching to Ingress PathTypes ([#88587](https://github.com/kubernetes/kubernetes/pull/88587), [@cmluciano](https://github.com/cmluciano)) [SIG Apps, Cluster Lifecycle and Network]
- Ingress: Add alternate backends via TypedLocalObjectReference ([#88775](https://github.com/kubernetes/kubernetes/pull/88775), [@cmluciano](https://github.com/cmluciano)) [SIG Apps and Network]
- Ingress: allow wildcard hosts in IngressRule ([#88858](https://github.com/kubernetes/kubernetes/pull/88858), [@cmluciano](https://github.com/cmluciano)) [SIG Network]
- Kube-controller-manager and kube-scheduler expose profiling by default to match the kube-apiserver.  Use `--enable-profiling=false` to disable. ([#88663](https://github.com/kubernetes/kubernetes/pull/88663), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Cloud Provider and Scheduling]
- Move TaintBasedEvictions feature gates to GA ([#87487](https://github.com/kubernetes/kubernetes/pull/87487), [@skilxn-go](https://github.com/skilxn-go)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- New flag --endpointslice-updates-batch-period in kube-controller-manager can be used to reduce number of endpointslice updates generated by pod changes. ([#88745](https://github.com/kubernetes/kubernetes/pull/88745), [@mborsz](https://github.com/mborsz)) [SIG API Machinery, Apps and Network]
- Scheduler Extenders can now be configured in the v1alpha2 component config ([#88768](https://github.com/kubernetes/kubernetes/pull/88768), [@damemi](https://github.com/damemi)) [SIG Release, Scheduling and Testing]
- The apiserver/v1alph1&#35;EgressSelectorConfiguration API is now beta. ([#88502](https://github.com/kubernetes/kubernetes/pull/88502), [@caesarxuchao](https://github.com/caesarxuchao)) [SIG API Machinery]
- The storage.k8s.io/CSIDriver has moved to GA, and is now available for use. ([#84814](https://github.com/kubernetes/kubernetes/pull/84814), [@huffmanca](https://github.com/huffmanca)) [SIG API Machinery, Apps, Auth, Node, Scheduling, Storage and Testing]
- VolumePVCDataSource moves to GA in 1.18 release ([#88686](https://github.com/kubernetes/kubernetes/pull/88686), [@j-griffith](https://github.com/j-griffith)) [SIG Apps, CLI and Cluster Lifecycle]

### Feature

- Add `rest_client_rate_limiter_duration_seconds` metric to component-base to track client side rate limiter latency in seconds. Broken down by verb and URL. ([#88134](https://github.com/kubernetes/kubernetes/pull/88134), [@jennybuckley](https://github.com/jennybuckley)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Allow user to specify resource using --filename flag when invoking kubectl exec ([#88460](https://github.com/kubernetes/kubernetes/pull/88460), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Apiserver add a new flag --goaway-chance which is the fraction of requests that will be closed gracefully(GOAWAY) to prevent HTTP/2 clients from getting stuck on a single apiserver. 
  After the connection closed(received GOAWAY), the client's other in-flight requests won't be affected, and the client will reconnect. 
  The flag min value is 0 (off), max is .02 (1/50 requests); .001 (1/1000) is a recommended starting point.
  Clusters with single apiservers, or which don't use a load balancer, should NOT enable this. ([#88567](https://github.com/kubernetes/kubernetes/pull/88567), [@answer1991](https://github.com/answer1991)) [SIG API Machinery]
- Azure: add support for single stack IPv6 ([#88448](https://github.com/kubernetes/kubernetes/pull/88448), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- DefaultConstraints can be specified for the PodTopologySpread plugin in the component config ([#88671](https://github.com/kubernetes/kubernetes/pull/88671), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Kubeadm: support Windows specific kubelet flags in kubeadm-flags.env ([#88287](https://github.com/kubernetes/kubernetes/pull/88287), [@gab-satchi](https://github.com/gab-satchi)) [SIG Cluster Lifecycle and Windows]
- Kubectl cluster-info dump changed to only display a message telling you the location where the output was written when the output is not standard output. ([#88765](https://github.com/kubernetes/kubernetes/pull/88765), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Print NotReady when pod is not ready based on its conditions. ([#88240](https://github.com/kubernetes/kubernetes/pull/88240), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- Scheduler Extender API is now located under k8s.io/kube-scheduler/extender ([#88540](https://github.com/kubernetes/kubernetes/pull/88540), [@damemi](https://github.com/damemi)) [SIG Release, Scheduling and Testing]
- Signatures on scale client methods have been modified to accept `context.Context` as a first argument. Signatures of Get, Update, and Patch methods have been updated to accept GetOptions, UpdateOptions and PatchOptions respectively. ([#88599](https://github.com/kubernetes/kubernetes/pull/88599), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG API Machinery, Apps, Autoscaling and CLI]
- Signatures on the dynamic client methods have been modified to accept `context.Context` as a first argument. Signatures of Delete and DeleteCollection methods now accept DeleteOptions by value instead of by reference. ([#88906](https://github.com/kubernetes/kubernetes/pull/88906), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, CLI, Cluster Lifecycle, Storage and Testing]
- Signatures on the metadata client methods have been modified to accept `context.Context` as a first argument. Signatures of Delete and DeleteCollection methods now accept DeleteOptions by value instead of by reference. ([#88910](https://github.com/kubernetes/kubernetes/pull/88910), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps and Testing]
- Webhooks will have alpha support for network proxy ([#85870](https://github.com/kubernetes/kubernetes/pull/85870), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Auth and Testing]
- When client certificate files are provided, reload files for new connections, and close connections when a certificate changes. ([#79083](https://github.com/kubernetes/kubernetes/pull/79083), [@jackkleeman](https://github.com/jackkleeman)) [SIG API Machinery, Auth, Node and Testing]
- When deleting objects using kubectl with the --force flag, you are no longer required to also specify --grace-period=0. ([#87776](https://github.com/kubernetes/kubernetes/pull/87776), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- `kubectl` now contains a `kubectl alpha debug` command. This command allows attaching an ephemeral container to a running pod for the purposes of debugging. ([#88004](https://github.com/kubernetes/kubernetes/pull/88004), [@verb](https://github.com/verb)) [SIG CLI]

### Documentation

- Update Japanese translation for kubectl help ([#86837](https://github.com/kubernetes/kubernetes/pull/86837), [@inductor](https://github.com/inductor)) [SIG CLI and Docs]
- `kubectl plugin` now prints a note how to install krew ([#88577](https://github.com/kubernetes/kubernetes/pull/88577), [@corneliusweig](https://github.com/corneliusweig)) [SIG CLI]

### Other (Bug, Cleanup or Flake)

- Azure VMSS LoadBalancerBackendAddressPools updating has been improved with squential-sync + concurrent-async requests. ([#88699](https://github.com/kubernetes/kubernetes/pull/88699), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- AzureFile and CephFS use new Mount library that prevents logging of sensitive mount options. ([#88684](https://github.com/kubernetes/kubernetes/pull/88684), [@saad-ali](https://github.com/saad-ali)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Build: Enable kube-cross image-building on K8s Infra ([#88562](https://github.com/kubernetes/kubernetes/pull/88562), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Client-go certificate manager rotation gained the ability to preserve optional intermediate chains accompanying issued certificates ([#88744](https://github.com/kubernetes/kubernetes/pull/88744), [@jackkleeman](https://github.com/jackkleeman)) [SIG API Machinery and Auth]
- Conformance image now depends on stretch-slim instead of debian-hyperkube-base as that image is being deprecated and removed. ([#88702](https://github.com/kubernetes/kubernetes/pull/88702), [@dims](https://github.com/dims)) [SIG Cluster Lifecycle, Release and Testing]
- Deprecate --generator flag from kubectl create commands ([#88655](https://github.com/kubernetes/kubernetes/pull/88655), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- FIX: prevent apiserver from panicking when failing to load audit webhook config file ([#88879](https://github.com/kubernetes/kubernetes/pull/88879), [@JoshVanL](https://github.com/JoshVanL)) [SIG API Machinery and Auth]
- Fix /readyz to return error immediately after a shutdown is initiated, before the --shutdown-delay-duration has elapsed. ([#88911](https://github.com/kubernetes/kubernetes/pull/88911), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fix a bug where kubenet fails to parse the tc output. ([#83572](https://github.com/kubernetes/kubernetes/pull/83572), [@chendotjs](https://github.com/chendotjs)) [SIG Network]
- Fix describe ingress annotations not sorted. ([#88394](https://github.com/kubernetes/kubernetes/pull/88394), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix handling of aws-load-balancer-security-groups annotation. Security-Groups assigned with this annotation are no longer modified by kubernetes which is the expected behaviour of most users. Also no unnecessary Security-Groups are created anymore if this annotation is used. ([#83446](https://github.com/kubernetes/kubernetes/pull/83446), [@Elias481](https://github.com/Elias481)) [SIG Cloud Provider]
- Fix kubectl create deployment image name ([#86636](https://github.com/kubernetes/kubernetes/pull/86636), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix missing "apiVersion" for "involvedObject" in Events for Nodes. ([#87537](https://github.com/kubernetes/kubernetes/pull/87537), [@uthark](https://github.com/uthark)) [SIG Apps and Node]
- Fix that prevents repeated fetching of PVC/PV objects by kubelet when processing of pod volumes fails. While this prevents hammering API server in these error scenarios, it means that some errors in processing volume(s) for a pod could now take up to 2-3 minutes before retry. ([#88141](https://github.com/kubernetes/kubernetes/pull/88141), [@tedyu](https://github.com/tedyu)) [SIG Node and Storage]
- Fix: azure file mount timeout issue ([#88610](https://github.com/kubernetes/kubernetes/pull/88610), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: corrupted mount point in csi driver ([#88569](https://github.com/kubernetes/kubernetes/pull/88569), [@andyzhangx](https://github.com/andyzhangx)) [SIG Storage]
- Fixed a bug in the TopologyManager. Previously, the TopologyManager would only guarantee alignment if container creation was serialized in some way. Alignment is now guaranteed under all scenarios of container creation. ([#87759](https://github.com/kubernetes/kubernetes/pull/87759), [@klueska](https://github.com/klueska)) [SIG Node]
- Fixed block CSI volume cleanup after timeouts. ([#88660](https://github.com/kubernetes/kubernetes/pull/88660), [@jsafrane](https://github.com/jsafrane)) [SIG Node and Storage]
- Fixes issue where you can't attach more than 15 GCE Persistent Disks to c2, n2, m1, m2 machine types. ([#88602](https://github.com/kubernetes/kubernetes/pull/88602), [@yuga711](https://github.com/yuga711)) [SIG Storage]
- For volumes that allow attaches across multiple nodes, attach and detach operations across different nodes are now executed in parallel. ([#88678](https://github.com/kubernetes/kubernetes/pull/88678), [@verult](https://github.com/verult)) [SIG Apps, Node and Storage]
- Hide kubectl.kubernetes.io/last-applied-configuration in describe command ([#88758](https://github.com/kubernetes/kubernetes/pull/88758), [@soltysh](https://github.com/soltysh)) [SIG Auth and CLI]
- In GKE alpha clusters it will be possible to use the service annotation `cloud.google.com/network-tier: Standard` ([#88487](https://github.com/kubernetes/kubernetes/pull/88487), [@zioproto](https://github.com/zioproto)) [SIG Cloud Provider]
- Kubelets perform fewer unnecessary pod status update operations on the API server. ([#88591](https://github.com/kubernetes/kubernetes/pull/88591), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Scalability]
- Plugin/PluginConfig and Policy APIs are mutually exclusive when running the scheduler ([#88864](https://github.com/kubernetes/kubernetes/pull/88864), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Specifying PluginConfig for the same plugin more than once fails scheduler startup.
  
  Specifying extenders and configuring .ignoredResources for the NodeResourcesFit plugin fails ([#88870](https://github.com/kubernetes/kubernetes/pull/88870), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Support TLS Server Name overrides in kubeconfig file and via --tls-server-name in kubectl ([#88769](https://github.com/kubernetes/kubernetes/pull/88769), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Auth and CLI]
- Terminating a restartPolicy=Never pod no longer has a chance to report the pod succeeded when it actually failed. ([#88440](https://github.com/kubernetes/kubernetes/pull/88440), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- The EventRecorder from k8s.io/client-go/tools/events will now create events in the default namespace (instead of kube-system) when the related object does not have it set. ([#88815](https://github.com/kubernetes/kubernetes/pull/88815), [@enj](https://github.com/enj)) [SIG API Machinery]
- The audit event sourceIPs list will now always end with the IP that sent the request directly to the API server. ([#87167](https://github.com/kubernetes/kubernetes/pull/87167), [@tallclair](https://github.com/tallclair)) [SIG API Machinery and Auth]
- Update to use golang 1.13.8 ([#87648](https://github.com/kubernetes/kubernetes/pull/87648), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Release and Testing]
- Validate kube-proxy flags --ipvs-tcp-timeout, --ipvs-tcpfin-timeout, --ipvs-udp-timeout ([#88657](https://github.com/kubernetes/kubernetes/pull/88657), [@chendotjs](https://github.com/chendotjs)) [SIG Network]


# v1.18.0-beta.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-beta.1

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes.tar.gz) | `7c182ca905b3a31871c01ab5fdaf46f074547536c7975e069ff230af0d402dfc0346958b1d084bd2c108582ffc407484e6a15a1cd93e9affbe34b6e99409ef1f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-src.tar.gz) | `d104b8c792b1517bd730787678c71c8ee3b259de81449192a49a1c6e37a6576d28f69b05c2019cc4a4c40ddeb4d60b80138323df3f85db8682caabf28e67c2de`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `bc337bb8f200a789be4b97ce99b9d7be78d35ebd64746307c28339dc4628f56d9903e0818c0888aaa9364357a528d1ac6fd34f74377000f292ec502fbea3837e`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `38dfa5e0b0cfff39942c913a6bcb2ad8868ec43457d35cffba08217bb6e7531720e0731f8588505f4c81193ce5ec0e5fe6870031cf1403fbbde193acf7e53540`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-linux-386.tar.gz) | `8e63ec7ce29c69241120c037372c6c779e3f16253eabd612c7cbe6aa89326f5160eb5798004d723c5cd72d458811e98dac3574842eb6a57b2798ecd2bbe5bcf9`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `c1be9f184a7c3f896a785c41cd6ece9d90d8cb9b1f6088bdfb5557d8856c55e455f6688f5f54c2114396d5ae7adc0361e34ebf8e9c498d0187bd785646ccc1d0`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `8eab02453cfd9e847632a774a0e0cf3a33c7619fb4ced7f1840e1f71444e8719b1c8e8cbfdd1f20bb909f3abe39cdcac74f14cb9c878c656d35871b7c37c7cbe`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `f7df0ec02d2e7e63278d5386e8153cfe2b691b864f17b6452cc824a5f328d688976c975b076e60f1c6b3c859e93e477134fbccc53bb49d9e846fb038b34eee48`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `36dd5b10addca678a518e6d052c9d6edf473e3f87388a2f03f714c93c5fbfe99ace16cf3b382a531be20a8fe6f4160f8d891800dd2cff5f23c9ca12c2f4a151b`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `5bdbb44b996ab4ccf3a383780270f5cfdbf174982c300723c8bddf0a48ae5e459476031c1d51b9d30ffd621d0a126c18a5de132ef1d92fca2f3e477665ea10cc`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-windows-386.tar.gz) | `5dea3d4c4e91ef889850143b361974250e99a3c526f5efee23ff9ccdcd2ceca4a2247e7c4f236bdfa77d2150157da5d676ac9c3ba26cf3a2f1e06d8827556f77`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `db298e698391368703e6aea7f4345aec5a4b8c69f9d8ff6c99fb5804a6cea16d295fb01e70fe943ade3d4ce9200a081ad40da21bd331317ec9213f69b4d6c48f`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `c6284929dd5940e750b48db72ffbc09f73c5ec31ab3db283babb8e4e07cd8cbb27642f592009caae4717981c0db82c16312849ef4cbafe76acc4264c7d5864ac`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `6fc9552cf082c54cc0833b19876117c87ba7feb5a12c7e57f71b52208daf03eaef3ca56bd22b7bce2d6e81b5a23537cf6f5497a6eaa356c0aab1d3de26c309f9`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `b794b9c399e548949b5bfb2fe71123e86c2034847b2c99aca34b6de718a35355bbecdae9dc2a81c49e3c82fb4b5862526a3f63c2862b438895e12c5ea884f22e`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `fddaed7a54f97046a91c29534645811c6346e973e22950b2607b8c119c2377e9ec2d32144f81626078cdaeca673129cc4016c1a3dbd3d43674aa777089fb56ac`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `65951a534bb55069c7419f41cbcdfe2fae31541d8a3f9eca11fc2489addf281c5ad2d13719212657da0be5b898f22b57ac39446d99072872fbacb0a7d59a4f74`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `992059efb5cae7ed0ef55820368d854bad1c6d13a70366162cd3b5111ce24c371c7c87ded2012f055e08b2ff1b4ef506e1f4e065daa3ac474fef50b5efa4fb07`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `c63ae0f8add5821ad267774314b8c8c1ffe3b785872bf278e721fd5dfdad1a5db1d4db3720bea0a36bf10d9c6dd93e247560162c0eac6e1b743246f587d3b27a`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `47adb9ddf6eaf8f475b89f59ee16fbd5df183149a11ad1574eaa645b47a6d58aec2ca70ba857ce9f1a5793d44cf7a61ebc6874793bb685edaf19410f4f76fd13`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `a3bc4a165567c7b76a3e45ab7b102d6eb3ecf373eb048173f921a4964cf9be8891d0d5b8dafbd88c3af7b0e21ef3d41c1e540c3347ddd84b929b3a3d02ceb7b2`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `109ddf37c748f69584c829db57107c3518defe005c11fcd2a1471845c15aae0a3c89aafdd734229f4069ed18856cc650c80436684e1bdc43cfee3149b0324746`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `a3a75d2696ad3136476ad7d811e8eabaff5111b90e592695e651d6111f819ebf0165b8b7f5adc05afb5f7f01d1e5fb64876cb696e492feb20a477a5800382b7a`

## Changelog since v1.18.0-beta.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

- The StreamingProxyRedirects feature and `--redirect-container-streaming` flag are deprecated, and will be removed in a future release. The default behavior (proxy streaming requests through the kubelet) will be the only supported option.
  If you are setting `--redirect-container-streaming=true`, then you must migrate off this configuration. The flag will no longer be able to be enabled starting in v1.20. If you are not setting the flag, no action is necessary. ([#88290](https://github.com/kubernetes/kubernetes/pull/88290), [@tallclair](https://github.com/tallclair)) [SIG API Machinery and Node]

- Yes.
  
  Feature Name: Support using network resources (VNet, LB, IP, etc.) in different AAD Tenant and Subscription than those for the cluster.
  
  Changes in Pull Request:
  
    1. Add properties `networkResourceTenantID` and `networkResourceSubscriptionID` in cloud provider auth config section, which indicates the location of network resources.
    2. Add function `GetMultiTenantServicePrincipalToken` to fetch multi-tenant service principal token, which will be used by Azure VM/VMSS Clients in this feature.
    3. Add function `GetNetworkResourceServicePrincipalToken` to fetch network resource service principal token, which will be used by Azure Network Resource (Load Balancer, Public IP, Route Table, Network Security Group and their sub level resources) Clients in this feature.
    4. Related unit tests.
  
  None.
  
  User Documentation: In PR https://github.com/kubernetes-sigs/cloud-provider-azure/pull/301 ([#88384](https://github.com/kubernetes/kubernetes/pull/88384), [@bowen5](https://github.com/bowen5)) [SIG Cloud Provider]

## Changes by Kind

### Deprecation

- Azure service annotation service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset has been deprecated. Its support would be removed in a future release. ([#88462](https://github.com/kubernetes/kubernetes/pull/88462), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]

### API Change

- API additions to apiserver types ([#87179](https://github.com/kubernetes/kubernetes/pull/87179), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Cloud Provider and Cluster Lifecycle]
- Add Scheduling Profiles to kubescheduler.config.k8s.io/v1alpha2 ([#88087](https://github.com/kubernetes/kubernetes/pull/88087), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling and Testing]
- Added support for multiple sizes huge pages on a container level ([#84051](https://github.com/kubernetes/kubernetes/pull/84051), [@bart0sh](https://github.com/bart0sh)) [SIG Apps, Node and Storage]
- AppProtocol is a new field on Service and Endpoints resources, enabled with the ServiceAppProtocol feature gate. ([#88503](https://github.com/kubernetes/kubernetes/pull/88503), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Fixed missing validation of uniqueness of list items in lists with `x-kubernetes-list-type: map` or x-kubernetes-list-type: set` in CustomResources. ([#84920](https://github.com/kubernetes/kubernetes/pull/84920), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Introduces optional --detect-local flag to kube-proxy. 
  Currently the only supported value is "cluster-cidr", 
  which is the default if not specified. ([#87748](https://github.com/kubernetes/kubernetes/pull/87748), [@satyasm](https://github.com/satyasm)) [SIG Cluster Lifecycle, Network and Scheduling]
- Kube-scheduler can run more than one scheduling profile. Given a pod, the profile is selected by using its `.spec.SchedulerName`. ([#88285](https://github.com/kubernetes/kubernetes/pull/88285), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps, Scheduling and Testing]
- Moving Windows RunAsUserName feature to GA ([#87790](https://github.com/kubernetes/kubernetes/pull/87790), [@marosset](https://github.com/marosset)) [SIG Apps and Windows]

### Feature

- Add --dry-run to kubectl delete, taint, replace ([#88292](https://github.com/kubernetes/kubernetes/pull/88292), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Add huge page stats to Allocated resources in "kubectl describe node" ([#80605](https://github.com/kubernetes/kubernetes/pull/80605), [@odinuge](https://github.com/odinuge)) [SIG CLI]
- Kubeadm: The ClusterStatus struct present in the kubeadm-config ConfigMap is deprecated and will be removed on a future version. It is going to be maintained by kubeadm until it gets removed. The same information can be found on `etcd` and `kube-apiserver` pod annotations, `kubeadm.kubernetes.io/etcd.advertise-client-urls` and `kubeadm.kubernetes.io/kube-apiserver.advertise-address.endpoint` respectively. ([#87656](https://github.com/kubernetes/kubernetes/pull/87656), [@ereslibre](https://github.com/ereslibre)) [SIG Cluster Lifecycle]
- Kubeadm: add the experimental feature gate PublicKeysECDSA that can be used to create a
  cluster with ECDSA certificates from "kubeadm init". Renewal of existing ECDSA certificates is
  also supported using "kubeadm alpha certs renew", but not switching between the RSA and
  ECDSA algorithms on the fly or during upgrades. ([#86953](https://github.com/kubernetes/kubernetes/pull/86953), [@rojkov](https://github.com/rojkov)) [SIG API Machinery, Auth and Cluster Lifecycle]
- Kubeadm: on kubeconfig certificate renewal, keep the embedded CA in sync with the one on disk ([#88052](https://github.com/kubernetes/kubernetes/pull/88052), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: upgrade supports fallback to the nearest known etcd version if an unknown k8s version is passed ([#88373](https://github.com/kubernetes/kubernetes/pull/88373), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- New flag `--show-hidden-metrics-for-version` in kube-scheduler can be used to show all hidden metrics that deprecated in the previous minor release. ([#84913](https://github.com/kubernetes/kubernetes/pull/84913), [@serathius](https://github.com/serathius)) [SIG Instrumentation and Scheduling]
- Scheduler framework permit plugins now run at the end of the scheduling cycle, after reserve plugins. Waiting on permit will remain in the beginning of the binding cycle. ([#88199](https://github.com/kubernetes/kubernetes/pull/88199), [@mateuszlitwin](https://github.com/mateuszlitwin)) [SIG Scheduling]
- The kubelet and the default docker runtime now support running ephemeral containers in the Linux process namespace of a target container. Other container runtimes must implement this feature before it will be available in that runtime. ([#84731](https://github.com/kubernetes/kubernetes/pull/84731), [@verb](https://github.com/verb)) [SIG Node]

### Other (Bug, Cleanup or Flake)

- Add delays between goroutines for vm instance update ([#88094](https://github.com/kubernetes/kubernetes/pull/88094), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Add init containers log to cluster dump info. ([#88324](https://github.com/kubernetes/kubernetes/pull/88324), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- CPU limits are now respected for Windows containers. If a node is over-provisioned, no weighting is used - only limits are respected. ([#86101](https://github.com/kubernetes/kubernetes/pull/86101), [@PatrickLang](https://github.com/PatrickLang)) [SIG Node, Testing and Windows]
- Cloud provider config CloudProviderBackoffMode has been removed since it won't be used anymore. ([#88463](https://github.com/kubernetes/kubernetes/pull/88463), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Evictions due to pods breaching their ephemeral storage limits are now recorded by the `kubelet_evictions` metric and can be alerted on. ([#87906](https://github.com/kubernetes/kubernetes/pull/87906), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Fix: add remediation in azure disk attach/detach ([#88444](https://github.com/kubernetes/kubernetes/pull/88444), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: check disk status before disk azure disk ([#88360](https://github.com/kubernetes/kubernetes/pull/88360), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fixed cleaning of CSI raw block volumes. ([#87978](https://github.com/kubernetes/kubernetes/pull/87978), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Get-kube.sh uses the gcloud's current local GCP service account for auth when the provider is GCE or GKE instead of the metadata server default ([#88383](https://github.com/kubernetes/kubernetes/pull/88383), [@BenTheElder](https://github.com/BenTheElder)) [SIG Cluster Lifecycle]
- Golang/x/net has been updated to bring in fixes for CVE-2020-9283 ([#88381](https://github.com/kubernetes/kubernetes/pull/88381), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Kubeadm now includes CoreDNS version 1.6.7 ([#86260](https://github.com/kubernetes/kubernetes/pull/86260), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that 'kubeadm upgrade' hangs in single node cluster ([#88434](https://github.com/kubernetes/kubernetes/pull/88434), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Optimize kubectl version help info ([#88313](https://github.com/kubernetes/kubernetes/pull/88313), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Removes the deprecated command `kubectl rolling-update` ([#88057](https://github.com/kubernetes/kubernetes/pull/88057), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG Architecture, CLI and Testing]


# v1.18.0-alpha.5

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-alpha.5

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes.tar.gz) | `6452cac2b80721e9f577cb117c29b9ac6858812b4275c2becbf74312566f7d016e8b34019bd1bf7615131b191613bf9b973e40ad9ac8f6de9007d41ef2d7fd70`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-src.tar.gz) | `e41d9d4dd6910a42990051fcdca4bf5d3999df46375abd27ffc56aae9b455ae984872302d590da6aa85bba6079334fb5fe511596b415ee79843dee1c61c137da`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-darwin-386.tar.gz) | `5c95935863492b31d4aaa6be93260088dafea27663eb91edca980ca3a8485310e60441bc9050d4d577e9c3f7ffd96db516db8d64321124cec1b712e957c9fe1c`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-darwin-amd64.tar.gz) | `868faa578b3738604d8be62fae599ccc556799f1ce54807f1fe72599f20f8a1f98ad8152fac14a08a463322530b696d375253ba3653325e74b587df6e0510da3`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-linux-386.tar.gz) | `76a89d1d30b476b47f8fb808e342f89608e5c1c1787c4c06f2d7e763f9482e2ae8b31e6ad26541972e2b9a3a7c28327e3150cdd355e8b8d8b050a801bbf08d49`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-linux-amd64.tar.gz) | `07ad96a09b44d1c707d7c68312c5d69b101a3424bf1e6e9400b2e7a3fba78df04302985d473ddd640d8f3f0257be34110dbe1304b9565dd9d7a4639b7b7b85fd`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-linux-arm.tar.gz) | `c04fed9fa370a75c1b8e18b2be0821943bb9befcc784d14762ea3278e73600332a9b324d5eeaa1801d20ad6be07a553c41dcf4fa7ab3eadd0730ab043d687c8c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-linux-arm64.tar.gz) | `4199147dea9954333df26d34248a1cb7b02ebbd6380ffcd42d9f9ed5fdabae45a59215474dab3c11436c82e60bd27cbd03b3dde288bf611cd3e78b87c783c6a9`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-linux-ppc64le.tar.gz) | `4f6d4d61d1c52d3253ca19031ebcd4bad06d19b68bbaaab5c8e8c590774faea4a5ceab1f05f2706b61780927e1467815b3479342c84d45df965aba78414727c4`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-linux-s390x.tar.gz) | `e2a454151ae5dd891230fb516a3f73f73ab97832db66fd3d12e7f1657a569f58a9fe2654d50ddd7d8ec88a5ff5094199323a4c6d7d44dcf7edb06cca11dd4de1`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-windows-386.tar.gz) | `14b262ba3b71c41f545db2a017cf1746075ada5745a858d2a62bc9df7c5dc10607220375db85e2c4cb85307b09709e58bc66a407488e0961191e3249dc7742b0`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-client-windows-amd64.tar.gz) | `26353c294755a917216664364b524982b7f5fc6aa832ce90134bb178df8a78604963c68873f121ea5f2626ff615bdbf2ffe54e00578739cde6df42ffae034732`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-server-linux-amd64.tar.gz) | `ba77e0e7c610f59647c1b2601f82752964a0f54b7ad609a89b00fcfd553d0f0249f6662becbabaa755bb769b36a2000779f08022c40fb8cc61440337481317a1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-server-linux-arm.tar.gz) | `45e87b3e844ea26958b0b489e8c9b90900a3253000850f5ff9e87ffdcafba72ab8fd17b5ba092051a58a4bc277912c047a85940ec7f093dff6f9e8bf6fed3b42`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-server-linux-arm64.tar.gz) | `155e136e3124ead69c594eead3398d6cfdbb8f823c324880e8a7bbd1b570b05d13a77a69abd0a6758cfcc7923971cc6da4d3e0c1680fd519b632803ece00d5ce`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-server-linux-ppc64le.tar.gz) | `3fa0fb8221da19ad9d03278961172b7fa29a618b30abfa55e7243bb937dede8df56658acf02e6b61e7274fbc9395e237f49c62f2a83017eca2a69f67af31c01c`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-server-linux-s390x.tar.gz) | `db3199c3d7ba0b326d71dc8b80f50b195e79e662f71386a3b2976d47d13d7b0136887cc21df6f53e70a3d733da6eac7bbbf3bab2df8a1909a3cee4b44c32dd0b`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-node-linux-amd64.tar.gz) | `addcdfbad7f12647e6babb8eadf853a374605c8f18bf63f416fa4d3bf1b903aa206679d840433206423a984bb925e7983366edcdf777cf5daef6ef88e53d6dfa`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-node-linux-arm.tar.gz) | `b2ac54e0396e153523d116a2aaa32c919d6243931e0104cd47a23f546d710e7abdaa9eae92d978ce63c92041e63a9b56f5dd8fd06c812a7018a10ecac440f768`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-node-linux-arm64.tar.gz) | `7aab36f2735cba805e4fd109831a1af0f586a88db3f07581b6dc2a2aab90076b22c96b490b4f6461a8fb690bf78948b6d514274f0d6fb0664081de2d44dc48e1`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-node-linux-ppc64le.tar.gz) | `a579936f07ebf86f69f297ac50ba4c34caf2c0b903f73190eb581c78382b05ef36d41ade5bfd25d7b1b658cfcbee3d7125702a18e7480f9b09a62733a512a18a`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-node-linux-s390x.tar.gz) | `58fa0359ddd48835192fab1136a2b9b45d1927b04411502c269cda07cb8a8106536973fb4c7fedf1d41893a524c9fe2e21078fdf27bfbeed778273d024f14449`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.5/kubernetes-node-windows-amd64.tar.gz) | `9086c03cd92b440686cea6d8c4e48045cc46a43ab92ae0e70350b3f51804b9e2aaae7178142306768bae00d9ef6dd938167972bfa90b12223540093f735a45db`

## Changelog since v1.18.0-alpha.3

### Deprecation

- Kubeadm: command line option "kubelet-version" for `kubeadm upgrade node` has been deprecated and will be removed in a future release. ([#87942](https://github.com/kubernetes/kubernetes/pull/87942), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]

### API Change

- Kubelet podresources API now provides the information about active pods only. ([#79409](https://github.com/kubernetes/kubernetes/pull/79409), [@takmatsu](https://github.com/takmatsu)) [SIG Node]
- Remove deprecated fields from .leaderElection in kubescheduler.config.k8s.io/v1alpha2 ([#87904](https://github.com/kubernetes/kubernetes/pull/87904), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Signatures on generated clientset methods have been modified to accept `context.Context` as a first argument. Signatures of generated Create, Update, and Patch methods have been updated to accept CreateOptions, UpdateOptions and PatchOptions respectively. Clientsets that with the previous interface have been added in new "deprecated" packages to allow incremental migration to the new APIs. The deprecated packages will be removed in the 1.21 release. ([#87299](https://github.com/kubernetes/kubernetes/pull/87299), [@mikedanese](https://github.com/mikedanese)) [SIG API Machinery, Apps, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling, Storage, Testing and Windows]
- The k8s.io/node-api component is no longer updated. Instead, use the RuntimeClass types located within k8s.io/api, and the generated clients located within k8s.io/client-go ([#87503](https://github.com/kubernetes/kubernetes/pull/87503), [@liggitt](https://github.com/liggitt)) [SIG Node and Release]

### Feature

- Add indexer for storage cacher ([#85445](https://github.com/kubernetes/kubernetes/pull/85445), [@shaloulcy](https://github.com/shaloulcy)) [SIG API Machinery]
- Add support for mount options to the FC volume plugin ([#87499](https://github.com/kubernetes/kubernetes/pull/87499), [@ejweber](https://github.com/ejweber)) [SIG Storage]
- Added a config-mode flag in azure auth module to enable getting AAD token without spn: prefix in audience claim. When it's not specified, the default behavior doesn't change. ([#87630](https://github.com/kubernetes/kubernetes/pull/87630), [@weinong](https://github.com/weinong)) [SIG API Machinery, Auth, CLI and Cloud Provider]
- Introduced BackoffManager interface for backoff management ([#87829](https://github.com/kubernetes/kubernetes/pull/87829), [@zhan849](https://github.com/zhan849)) [SIG API Machinery]
- PodTopologySpread plugin now excludes terminatingPods when making scheduling decisions. ([#87845](https://github.com/kubernetes/kubernetes/pull/87845), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- Promote CSIMigrationOpenStack to Beta (off by default since it requires installation of the OpenStack Cinder CSI Driver)
  The in-tree AWS OpenStack Cinder "kubernetes.io/cinder" was already deprecated a while ago and will be removed in 1.20. Users should enable CSIMigration + CSIMigrationOpenStack features and install the OpenStack Cinder CSI Driver (https://github.com/kubernetes-sigs/cloud-provider-openstack) to avoid disruption to existing Pod and PVC objects at that time.
  Users should start using the OpenStack Cinder CSI Driver directly for any new volumes. ([#85637](https://github.com/kubernetes/kubernetes/pull/85637), [@dims](https://github.com/dims)) [SIG Cloud Provider]

### Design

- The scheduler Permit extension point doesn't return a boolean value in its Allow() and Reject() functions. ([#87936](https://github.com/kubernetes/kubernetes/pull/87936), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]

### Other (Bug, Cleanup or Flake)

- Adds "volume.beta.kubernetes.io/migrated-to" annotation to PV's and PVC's when they are migrated to signal external provisioners to pick up those objects for Provisioning and Deleting. ([#87098](https://github.com/kubernetes/kubernetes/pull/87098), [@davidz627](https://github.com/davidz627)) [SIG Apps and Storage]
- Fix a bug in the dual-stack IPVS proxier where stale IPv6 endpoints were not being cleaned up ([#87695](https://github.com/kubernetes/kubernetes/pull/87695), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network]
- Fix kubectl drain ignore daemonsets and others. ([#87361](https://github.com/kubernetes/kubernetes/pull/87361), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix: add azure disk migration support for CSINode ([#88014](https://github.com/kubernetes/kubernetes/pull/88014), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: add non-retriable errors in azure clients ([#87941](https://github.com/kubernetes/kubernetes/pull/87941), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fixed NetworkPolicy validation that Except values are accepted when they are outside the CIDR range. ([#86578](https://github.com/kubernetes/kubernetes/pull/86578), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Improves performance of the node authorizer ([#87696](https://github.com/kubernetes/kubernetes/pull/87696), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Iptables/userspace proxy: improve performance by getting local addresses only once per sync loop, instead of for every external IP ([#85617](https://github.com/kubernetes/kubernetes/pull/85617), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Network]
- Kube-aggregator: always sets unavailableGauge metric to reflect the current state of a service. ([#87778](https://github.com/kubernetes/kubernetes/pull/87778), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kubeadm allows to configure single-stack clusters if dual-stack is enabled ([#87453](https://github.com/kubernetes/kubernetes/pull/87453), [@aojea](https://github.com/aojea)) [SIG API Machinery, Cluster Lifecycle and Network]
- Kubeadm:  'kubeadm alpha kubelet config download' has been removed, please use 'kubeadm upgrade node phase kubelet-config' instead ([#87944](https://github.com/kubernetes/kubernetes/pull/87944), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: remove 'kubeadm upgrade node config' command since it was deprecated in v1.15, please use 'kubeadm upgrade node phase kubelet-config' instead ([#87975](https://github.com/kubernetes/kubernetes/pull/87975), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl describe <type> and kubectl top pod will return a message saying "No resources found" or "No resources found in <namespace> namespace" if there are no results to display. ([#87527](https://github.com/kubernetes/kubernetes/pull/87527), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Kubelet metrics gathered through metrics-server or prometheus should no longer timeout for Windows nodes running more than 3 pods. ([#87730](https://github.com/kubernetes/kubernetes/pull/87730), [@marosset](https://github.com/marosset)) [SIG Node, Testing and Windows]
- Kubelet metrics have been changed to buckets.
  For example the exec/{podNamespace}/{podID}/{containerName} is now just exec. ([#87913](https://github.com/kubernetes/kubernetes/pull/87913), [@cheftako](https://github.com/cheftako)) [SIG Node]
- Limit number of instances in a single update to GCE target pool to 1000. ([#87881](https://github.com/kubernetes/kubernetes/pull/87881), [@wojtek-t](https://github.com/wojtek-t)) [SIG Cloud Provider, Network and Scalability]
- Make Azure clients only retry on specified HTTP status codes ([#88017](https://github.com/kubernetes/kubernetes/pull/88017), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Pause image contains "Architecture" in non-amd64 images ([#87954](https://github.com/kubernetes/kubernetes/pull/87954), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Pods that are considered for preemption and haven't started don't produce an error log. ([#87900](https://github.com/kubernetes/kubernetes/pull/87900), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Prevent error message from being displayed when running kubectl plugin list and your path includes an empty string ([#87633](https://github.com/kubernetes/kubernetes/pull/87633), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- `kubectl create clusterrolebinding` creates rbac.authorization.k8s.io/v1 object ([#85889](https://github.com/kubernetes/kubernetes/pull/85889), [@oke-py](https://github.com/oke-py)) [SIG CLI]

# v1.18.0-alpha.4

[Documentation](https://docs.k8s.io)

## Important note about manual tag

Due to a [tagging bug in our Release Engineering tooling](https://github.com/kubernetes/release/issues/1080) during `v1.18.0-alpha.3`, we needed to push a manual tag (`v1.18.0-alpha.4`).

**No binaries have been produced or will be provided for `v1.18.0-alpha.4`.**

The changelog for `v1.18.0-alpha.4` is included as part of the [changelog since v1.18.0-alpha.3][#changelog-since-v1180-alpha3] section.

# v1.18.0-alpha.3

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-alpha.3

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes.tar.gz) | `60bf3bfc23b428f53fd853bac18a4a905b980fcc0bacd35ccd6357a89cfc26e47de60975ea6b712e65980e6b9df82a22331152d9f08ed4dba44558ba23a422d4`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-src.tar.gz) | `8adf1016565a7c93713ab6fa4293c2d13b4f6e4e1ec4dcba60bd71e218b4dbe9ef5eb7dbb469006743f498fc7ddeb21865cd12bec041af60b1c0edce8b7aecd5`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `abb32e894e8280c772e96227b574da81cd1eac374b8d29158b7f222ed550087c65482eef4a9817dfb5f2baf0d9b85fcdfa8feced0fbc1aacced7296853b57e1f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `5e4b1a993264e256ec1656305de7c306094cae9781af8f1382df4ce4eed48ce030827fde1a5e757d4ad57233d52075c9e4e93a69efbdc1102e4ba810705ccddc`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `68da39c2ae101d2b38f6137ceda07eb0c2124794982a62ef483245dbffb0611c1441ca085fa3127e7a9977f45646788832a783544ff06954114548ea0e526e46`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `dc236ffa8ad426620e50181419e9bebe3c161e953dbfb8a019f61b11286e1eb950b40d7cc03423bdf3e6974973bcded51300f98b55570c29732fa492dcde761d`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `ab0a8bd6dc31ea160b731593cdc490b3cc03668b1141cf95310bd7060dcaf55c7ee9842e0acae81063fdacb043c3552ccdd12a94afd71d5310b3ce056fdaa06c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `159ea083c601710d0d6aea423eeb346c99ffaf2abd137d35a53e87a07f5caf12fca8790925f3196f67b768fa92a024f83b50325dbca9ccd4dde6c59acdce3509`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `16b0459adfa26575d13be49ab53ac7f0ffd05e184e4e13d2dfbfe725d46bb8ac891e1fd8aebe36ecd419781d4cc5cf3bd2aaaf5263cf283724618c4012408f40`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `d5aa1f5d89168995d2797eb839a04ce32560f405b38c1c0baaa0e313e4771ae7bb3b28e22433ad5897d36aadf95f73eb69d8d411d31c4115b6b0adf5fe041f85`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `374e16a1e52009be88c94786f80174d82dff66399bf294c9bee18a2159c42251c5debef1109a92570799148b08024960c6c50b8299a93fd66ebef94f198f34e9`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `5a94c1068c19271f810b994adad8e62fae03b3d4473c7c9e6d056995ff7757ea61dd8d140c9267dd41e48808876673ce117826d35a3c1bb5652752f11a044d57`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `a677bec81f0eba75114b92ff955bac74512b47e53959d56a685dae5edd527283d91485b1e86ad74ef389c5405863badf7eb22e2f0c9a568a4d0cb495c6a5c32f`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `2fb696f86ff13ebeb5f3cf2b254bf41303644c5ea84a292782eac6123550702655284d957676d382698c091358e5c7fe73f32803699c19be7138d6530fe413b6`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `738e95da9cfb8f1309479078098de1c38cef5e1dd5ee1129b77651a936a412b7cd0cf15e652afc7421219646a98846ab31694970432e48dea9c9cafa03aa59cf`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `7a85bfcbb2aa636df60c41879e96e788742ecd72040cb0db2a93418439c125218c58a4cfa96d01b0296c295793e94c544e87c2d98d50b49bc4cb06b41f874376`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `1f1cdb2efa3e7cac857203d8845df2fdaa5cf1f20df764efffff29371945ec58f6deeba06f8fbf70b96faf81b0c955bf4cb84e30f9516cb2cc1ed27c2d2185a6`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `4ccfced3f5ba4adfa58f4a9d1b2c5bdb3e89f9203ab0e27d11eb1c325ac323ebe63c015d2c9d070b233f5d1da76cab5349da3528511c1cd243e66edc9af381c4`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `d695a69d18449062e4c129e54ec8384c573955f8108f4b78adc2ec929719f2196b995469c728dd6656c63c44cda24315543939f85131ebc773cfe0de689df55b`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `21df1da88c89000abc22f97e482c3aaa5ce53ec9628d83dda2e04a1d86c4d53be46c03ed6f1f211df3ee5071bce39d944ff7716b5b6ada3b9c4821d368b0a898`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `ff77e3aacb6ed9d89baed92ef542c8b5cec83151b6421948583cf608bca3b779dce41fc6852961e00225d5e1502f6a634bfa61a36efa90e1aee90dedb787c2d2`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `57d75b7977ec1a0f6e7ed96a304dbb3b8664910f42ca19aab319a9ec33535ff5901dfca4abcb33bf5741cde6d152acd89a5f8178f0efe1dc24430e0c1af5b98f`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `63fdbb71773cfd73a914c498e69bb9eea3fc314366c99ffb8bd42ec5b4dae807682c83c1eb5cfb1e2feb4d11d9e49cc85ba644e954241320a835798be7653d61`

## Changelog since v1.18.0-alpha.2

### Deprecation

- Remove all the generators from kubectl run. It will now only create pods. Additionally, deprecates all the flags that are not relevant anymore. ([#87077](https://github.com/kubernetes/kubernetes/pull/87077), [@soltysh](https://github.com/soltysh)) [SIG Architecture, SIG CLI, and SIG Testing]
- kubeadm: kube-dns is deprecated and will not be supported in a future version ([#86574](https://github.com/kubernetes/kubernetes/pull/86574), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]

### API Change

- Add kubescheduler.config.k8s.io/v1alpha2 ([#87628](https://github.com/kubernetes/kubernetes/pull/87628), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- --enable-cadvisor-endpoints is now disabled by default. If you need access to the cAdvisor v1 Json API please enable it explicitly in the kubelet command line. Please note that this flag was deprecated in 1.15 and will be removed in 1.19. ([#87440](https://github.com/kubernetes/kubernetes/pull/87440), [@dims](https://github.com/dims)) [SIG Instrumentation, SIG Node, and SIG Testing]
- The following feature gates are removed, because the associated features were unconditionally enabled in previous releases: CustomResourceValidation, CustomResourceSubresources, CustomResourceWebhookConversion, CustomResourcePublishOpenAPI, CustomResourceDefaulting ([#87475](https://github.com/kubernetes/kubernetes/pull/87475), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]

### Feature

- The aggregation API will have alpha support for network proxy ([#87515](https://github.com/kubernetes/kubernetes/pull/87515), [@Sh4d1](https://github.com/Sh4d1)) [SIG API Machinery]
- API request throttling (due to a high rate of requests) is now reported in client-go logs at log level 2.  The messages are of the form
  
  Throttling request took 1.50705208s, request: GET:<URL>
  
  The presence of these messages, may indicate to the administrator the need to tune the cluster accordingly. ([#87740](https://github.com/kubernetes/kubernetes/pull/87740), [@jennybuckley](https://github.com/jennybuckley)) [SIG API Machinery]
- kubeadm: reject a node joining the cluster if a node with the same name already exists ([#81056](https://github.com/kubernetes/kubernetes/pull/81056), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- disableAvailabilitySetNodes is added to avoid VM list for VMSS clusters. It should only be used when vmType is "vmss" and all the nodes (including masters) are VMSS virtual machines. ([#87685](https://github.com/kubernetes/kubernetes/pull/87685), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- The kubectl --dry-run flag now accepts the values 'client', 'server', and 'none', to support client-side and server-side dry-run strategies. The boolean and unset values for the --dry-run flag are deprecated and a value will be required in a future version. ([#87580](https://github.com/kubernetes/kubernetes/pull/87580), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI]
- Add support for pre-allocated hugepages for more than one page size ([#82820](https://github.com/kubernetes/kubernetes/pull/82820), [@odinuge](https://github.com/odinuge)) [SIG Apps]
- Update CNI version to v0.8.5 ([#78819](https://github.com/kubernetes/kubernetes/pull/78819), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, SIG Cluster Lifecycle, SIG Network, SIG Release, and SIG Testing]
- Skip default spreading scoring plugin for pods that define TopologySpreadConstraints ([#87566](https://github.com/kubernetes/kubernetes/pull/87566), [@skilxn-go](https://github.com/skilxn-go)) [SIG Scheduling]
- Added more details to taint toleration errors ([#87250](https://github.com/kubernetes/kubernetes/pull/87250), [@starizard](https://github.com/starizard)) [SIG Apps, and SIG Scheduling]
- Scheduler: Add DefaultBinder plugin ([#87430](https://github.com/kubernetes/kubernetes/pull/87430), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling, and SIG Testing]
- Kube-apiserver metrics will now include request counts, latencies, and response sizes for /healthz, /livez, and /readyz requests. ([#83598](https://github.com/kubernetes/kubernetes/pull/83598), [@jktomer](https://github.com/jktomer)) [SIG API Machinery]

### Other (Bug, Cleanup or Flake)

- Fix the masters rolling upgrade causing thundering herd of LISTs on etcd leading to control plane unavailability. ([#86430](https://github.com/kubernetes/kubernetes/pull/86430), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, SIG Node, and SIG Testing]
- `kubectl diff` now returns 1 only on diff finding changes, and >1 on kubectl errors. The "exit status code 1" message as also been muted. ([#87437](https://github.com/kubernetes/kubernetes/pull/87437), [@apelisse](https://github.com/apelisse)) [SIG CLI, and SIG Testing]
- To reduce chances of throttling, VM cache is set to nil when Azure node provisioning state is deleting ([#87635](https://github.com/kubernetes/kubernetes/pull/87635), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix regression in statefulset conversion which prevented applying a statefulset multiple times. ([#87706](https://github.com/kubernetes/kubernetes/pull/87706), [@liggitt](https://github.com/liggitt)) [SIG Apps, and SIG Testing]
- fixed two scheduler metrics (pending_pods and schedule_attempts_total) not being recorded ([#87692](https://github.com/kubernetes/kubernetes/pull/87692), [@everpeace](https://github.com/everpeace)) [SIG Scheduling]
- Resolved a performance issue in the node authorizer index maintenance. ([#87693](https://github.com/kubernetes/kubernetes/pull/87693), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Removed the 'client' label from apiserver_request_total. ([#87669](https://github.com/kubernetes/kubernetes/pull/87669), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery, and SIG Instrumentation]
- `(*"k8s.io/client-go/rest".Request).{Do,DoRaw,Stream,Watch}` now require callers to pass a `context.Context` as an argument. The context is used for timeout and cancellation signaling and to pass supplementary information to round trippers in the wrapped transport chain. If you don't need any of this functionality, it is sufficient to pass a context created with `context.Background()` to these functions. The `(*"k8s.io/client-go/rest".Request).Context` method is removed now that all methods that execute a request accept a context directly. ([#87597](https://github.com/kubernetes/kubernetes/pull/87597), [@mikedanese](https://github.com/mikedanese)) [SIG API Machinery, SIG Apps, SIG Auth, SIG Autoscaling, SIG CLI, SIG Cloud Provider, SIG Cluster Lifecycle, SIG Instrumentation, SIG Network, SIG Node, SIG Scheduling, SIG Storage, and SIG Testing]
- For volumes that allow attaches across multiple nodes, attach and detach operations across different nodes are now executed in parallel. ([#87258](https://github.com/kubernetes/kubernetes/pull/87258), [@verult](https://github.com/verult)) [SIG Apps, SIG Node, and SIG Storage]
- kubeadm: apply further improvements to the tentative support for concurrent etcd member join. Fixes a bug where multiple members can receive the same hostname. Increase the etcd client dial timeout and retry timeout for add/remove/... operations. ([#87505](https://github.com/kubernetes/kubernetes/pull/87505), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Reverted a kubectl azure auth module change where oidc claim spn: prefix was omitted resulting a breaking behavior with existing Azure AD OIDC enabled api-server ([#87507](https://github.com/kubernetes/kubernetes/pull/87507), [@weinong](https://github.com/weinong)) [SIG API Machinery, SIG Auth, and SIG Cloud Provider]
- Update cri-tools to v1.17.0 ([#86305](https://github.com/kubernetes/kubernetes/pull/86305), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cluster Lifecycle, and SIG Release]
- kubeadm: remove the deprecated CoreDNS feature-gate. It was set to "true" since v1.11 when the feature went GA. In v1.13 it was marked as deprecated and hidden from the CLI. ([#87400](https://github.com/kubernetes/kubernetes/pull/87400), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Shared informers are now more reliable in the face of network disruption. ([#86015](https://github.com/kubernetes/kubernetes/pull/86015), [@squeed](https://github.com/squeed)) [SIG API Machinery]
- the CSR signing cert/key pairs will be reloaded from disk like the kube-apiserver cert/key pairs ([#86816](https://github.com/kubernetes/kubernetes/pull/86816), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, SIG Apps, and SIG Auth]
- "kubectl describe statefulsets.apps" prints garbage for rolling update partition ([#85846](https://github.com/kubernetes/kubernetes/pull/85846), [@phil9909](https://github.com/phil9909)) [SIG CLI]


<!-- NEW RELEASE NOTES ENTRY -->


# v1.18.0-alpha.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-alpha.2


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes.tar.gz) | `7af83386b4b35353f0aa1bdaf73599eb08b1d1ca11ecc2c606854aff754db69f3cd3dc761b6d7fc86f01052f615ca53185f33dbf9e53b2f926b0f02fc103fbd3`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-src.tar.gz) | `a14b02a0a0bde97795a836a8f5897b0ee6b43e010e13e43dd4cca80a5b962a1ef3704eedc7916fed1c38ec663a71db48c228c91e5daacba7d9370df98c7ddfb6`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `427f214d47ded44519007de2ae87160c56c2920358130e474b768299751a9affcbc1b1f0f936c39c6138837bca2a97792a6700896976e98c4beee8a1944cfde1`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `861fd81ac3bd45765575bedf5e002a2294aba48ef9e15980fc7d6783985f7d7fcde990ea0aef34690977a88df758722ec0a2e170d5dcc3eb01372e64e5439192`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `7d59b05d6247e2606a8321c72cd239713373d876dbb43b0fb7f1cb857fa6c998038b41eeed78d9eb67ce77b0b71776ceed428cce0f8d2203c5181b473e0bd86c`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `7cdefb4e32bad9d2df5bb8e7e0a6f4dab2ae6b7afef5d801ac5c342d4effdeacd799081fa2dec699ecf549200786c7623c3176252010f12494a95240dd63311d`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `6212bbf0fa1d01ced77dcca2c4b76b73956cd3c6b70e0701c1fe0df5ff37160835f6b84fa2481e0e6979516551b14d8232d1c72764a559a3652bfe2a1e7488ff`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `1f0d9990700510165ee471acb2f88222f1b80e8f6deb351ce14cf50a70a9840fb99606781e416a13231c74b2bd7576981b5348171aa33b628d2666e366cd4629`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `77e00ba12a32db81e96f8de84609de93f32c61bb3f53875a57496d213aa6d1b92c09ad5a6de240a78e1a5bf77fac587ff92874f34a10f8909ae08ca32fda45d2`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `a39ec2044bed5a4570e9c83068e0fc0ce923ccffa44380f8bbc3247426beaff79c8a84613bcb58b05f0eb3afbc34c79fe3309aa2e0b81abcfd0aa04770e62e05`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `1a0ab88f9b7e34b60ab31d5538e97202a256ad8b7b7ed5070cae5f2f12d5d4edeae615db7a34ebbe254004b6393c6b2480100b09e30e59c9139492a3019a596a`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `1966eb5dfb78c1bc33aaa6389f32512e3aa92584250a0164182f3566c81d901b59ec78ee4e25df658bc1dd221b5a9527d6ce3b6c487ca3e3c0b319a077caa735`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `f814d6a3872e4572aa4da297c29def4c1fad8eba0903946780b6bf9788c72b99d71085c5aef9e12c01133b26fa4563c1766ba724ad2a8af2670a24397951a94d`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `56aa08225e546c92c2ff88ac57d3db7dd5e63640772ea72a429f080f7069827138cbc206f6f5fe3a0c01bfca043a9eda305ecdc1dcb864649114893e46b6dc84`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `fb87128d905211ba097aa860244a376575ae2edbaca6e51402a24bc2964854b9b273e09df3d31a2bcffc91509f7eecb2118b183fb0e0eb544f33403fa235c274`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `6d21fbf39b9d3a0df9642407d6f698fabdc809aca83af197bceb58a81b25846072f407f8fb7caae2e02dc90912e3e0f5894f062f91bcb69f8c2329625d3dfeb7`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `ddcda4dc360ca97705f71bf2a18ddacd7b7ddf77535b62e699e97a1b2dd24843751313351d0112e238afe69558e8271eba4d27ab77bb67b4b9e3fbde6eec85c9`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `78915a9bde35c70c67014f0cea8754849db4f6a84491a3ad9678fd3bc0203e43af5a63cfafe104ae1d56b05ce74893a87a6dcd008d7859e1af6b3bce65425b5d`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `3218e811abcb0cb09d80742def339be3916db5e9bbc62c0dc8e6d87085f7e3d9eeed79dea081906f1de78ddd07b7e3acdbd7765fdb838d262bb35602fd1df106`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `fa22de9c4440b8fb27f4e77a5a63c5e1c8aa8aa30bb79eda843b0f40498c21b8c0ad79fff1d841bb9fef53fe20da272506de9a86f81a0b36d028dbeab2e482ce`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `bbda9b5cc66e8f13d235703b2a85e2c4f02fa16af047be4d27a3e198e11eb11706e4a0fbb6c20978c770b069cd4cd9894b661f09937df9d507411548c36576e0`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `b2ed1eda013069adce2aac00b86d75b84e006cfce9bafac0b5a2bafcb60f8f2cb346b5ea44eafa72d777871abef1ea890eb3a2a05de28968f9316fa88886a8ed`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `bd8eb23dba711f31b5148257076b1bbe9629f2a75de213b2c779bd5b29279e9bf22f8bde32f4bc814f4c0cc49e19671eb8b24f4105f0fe2c1490c4b78ec3c704`

## Changelog since v1.18.0-alpha.1

### Other notable changes

* Bump golang/mock version to v1.3.1 ([#87326](https://github.com/kubernetes/kubernetes/pull/87326), [@wawa0210](https://github.com/wawa0210))
* fix a bug that orphan revision cannot be adopted and statefulset cannot be synced ([#86801](https://github.com/kubernetes/kubernetes/pull/86801), [@likakuli](https://github.com/likakuli))
* Azure storage clients now suppress requests on throttling ([#87306](https://github.com/kubernetes/kubernetes/pull/87306), [@feiskyer](https://github.com/feiskyer))
* Introduce Alpha field `Immutable` in both Secret and ConfigMap objects to mark their contents as immutable. The implementation is hidden behind feature gate `ImmutableEphemeralVolumes` (currently in Alpha stage). ([#86377](https://github.com/kubernetes/kubernetes/pull/86377), [@wojtek-t](https://github.com/wojtek-t))
* EndpointSlices will now be enabled by default. A new `EndpointSliceProxying` feature gate determines if kube-proxy will use EndpointSlices, this is disabled by default. ([#86137](https://github.com/kubernetes/kubernetes/pull/86137), [@robscott](https://github.com/robscott))
* kubeadm upgrades always persist the etcd backup for stacked ([#86861](https://github.com/kubernetes/kubernetes/pull/86861), [@SataQiu](https://github.com/SataQiu))
* Fix the bug PIP's DNS is deleted if no DNS label service annotation isn't set. ([#87246](https://github.com/kubernetes/kubernetes/pull/87246), [@nilo19](https://github.com/nilo19))
* New flag `--show-hidden-metrics-for-version` in kube-controller-manager can be used to show all hidden metrics that deprecated in the previous minor release. ([#85281](https://github.com/kubernetes/kubernetes/pull/85281), [@RainbowMango](https://github.com/RainbowMango))
* Azure network and VM clients now suppress requests on throttling ([#87122](https://github.com/kubernetes/kubernetes/pull/87122), [@feiskyer](https://github.com/feiskyer))
* `kubectl apply  -f <file> --prune -n <namespace>` should prune all resources not defined in the file in the cli specified namespace. ([#85613](https://github.com/kubernetes/kubernetes/pull/85613), [@MartinKaburu](https://github.com/MartinKaburu))
* Fixes service account token admission error in clusters that do not run the service account token controller ([#87029](https://github.com/kubernetes/kubernetes/pull/87029), [@liggitt](https://github.com/liggitt))
* CustomResourceDefinition status fields are no longer required for client validation when submitting manifests.  ([#87213](https://github.com/kubernetes/kubernetes/pull/87213), [@hasheddan](https://github.com/hasheddan))
* All apiservers log request lines in a more greppable format. ([#87203](https://github.com/kubernetes/kubernetes/pull/87203), [@lavalamp](https://github.com/lavalamp))
* provider/azure: Network security groups can now be in a separate resource group. ([#87035](https://github.com/kubernetes/kubernetes/pull/87035), [@CecileRobertMichon](https://github.com/CecileRobertMichon))
* Cleaned up the output from `kubectl describe CSINode <name>`. ([#85283](https://github.com/kubernetes/kubernetes/pull/85283), [@huffmanca](https://github.com/huffmanca))
* Fixed the following  ([#84265](https://github.com/kubernetes/kubernetes/pull/84265), [@bhagwat070919](https://github.com/bhagwat070919))
    * -  AWS Cloud Provider attempts to delete LoadBalancer security group it didn’t provision
    * -  AWS Cloud Provider creates default LoadBalancer security group even if annotation [service.beta.kubernetes.io/aws-load-balancer-security-groups] is present
* kubelet: resource metrics endpoint `/metrics/resource/v1alpha1` as well as all metrics under this endpoint have been deprecated. ([#86282](https://github.com/kubernetes/kubernetes/pull/86282), [@RainbowMango](https://github.com/RainbowMango))
    * Please convert to the following metrics emitted by endpoint `/metrics/resource`:
    * - scrape_error --> scrape_error
    * - node_cpu_usage_seconds_total --> node_cpu_usage_seconds
    * - node_memory_working_set_bytes --> node_memory_working_set_bytes
    * - container_cpu_usage_seconds_total --> container_cpu_usage_seconds
    * - container_memory_working_set_bytes --> container_memory_working_set_bytes
    * - scrape_error --> scrape_error
* You can now pass "--node-ip ::" to kubelet to indicate that it should autodetect an IPv6 address to use as the node's primary address. ([#85850](https://github.com/kubernetes/kubernetes/pull/85850), [@danwinship](https://github.com/danwinship))
* kubeadm: support automatic retry after failing to pull image ([#86899](https://github.com/kubernetes/kubernetes/pull/86899), [@SataQiu](https://github.com/SataQiu))
* TODO ([#87044](https://github.com/kubernetes/kubernetes/pull/87044), [@jennybuckley](https://github.com/jennybuckley))
* Improved yaml parsing performance ([#85458](https://github.com/kubernetes/kubernetes/pull/85458), [@cjcullen](https://github.com/cjcullen))
* Fixed a bug which could prevent a provider ID from ever being set for node if an error occurred determining the provider ID when the node was added. ([#87043](https://github.com/kubernetes/kubernetes/pull/87043), [@zjs](https://github.com/zjs))
* fix a regression in kubenet that prevent pods to obtain ip addresses ([#85993](https://github.com/kubernetes/kubernetes/pull/85993), [@chendotjs](https://github.com/chendotjs))
* Bind kube-dns containers to linux nodes to avoid Windows scheduling ([#83358](https://github.com/kubernetes/kubernetes/pull/83358), [@wawa0210](https://github.com/wawa0210))
* The following features are unconditionally enabled and the corresponding `--feature-gates` flags have been removed: `PodPriority`, `TaintNodesByCondition`, `ResourceQuotaScopeSelectors` and `ScheduleDaemonSetPods` ([#86210](https://github.com/kubernetes/kubernetes/pull/86210), [@draveness](https://github.com/draveness))
* Bind dns-horizontal containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83364](https://github.com/kubernetes/kubernetes/pull/83364), [@wawa0210](https://github.com/wawa0210))
* fix kubectl annotate error when local=true is set ([#86952](https://github.com/kubernetes/kubernetes/pull/86952), [@zhouya0](https://github.com/zhouya0))
* Bug fixes: ([#84163](https://github.com/kubernetes/kubernetes/pull/84163), [@david-tigera](https://github.com/david-tigera))
    * Make sure we include latest packages node #351 ([@caseydavenport](https://github.com/caseydavenport))
* fix kuebctl apply set-last-applied namespaces error   ([#86474](https://github.com/kubernetes/kubernetes/pull/86474), [@zhouya0](https://github.com/zhouya0))
* Add VolumeBinder method to FrameworkHandle interface, which allows user to get the volume binder when implementing scheduler framework plugins. ([#86940](https://github.com/kubernetes/kubernetes/pull/86940), [@skilxn-go](https://github.com/skilxn-go))
* elasticsearch supports automatically setting the advertise address ([#85944](https://github.com/kubernetes/kubernetes/pull/85944), [@SataQiu](https://github.com/SataQiu))
* If a serving certificates param specifies a name that is an IP for an SNI certificate, it will have priority for replying to server connections. ([#85308](https://github.com/kubernetes/kubernetes/pull/85308), [@deads2k](https://github.com/deads2k))
* kube-proxy: Added dual-stack IPv4/IPv6 support to the iptables proxier. ([#82462](https://github.com/kubernetes/kubernetes/pull/82462), [@vllry](https://github.com/vllry))
* Azure VMSS/VMSSVM clients now suppress requests on throttling ([#86740](https://github.com/kubernetes/kubernetes/pull/86740), [@feiskyer](https://github.com/feiskyer))
* New metric kubelet_pleg_last_seen_seconds to aid diagnosis of PLEG not healthy issues. ([#86251](https://github.com/kubernetes/kubernetes/pull/86251), [@bboreham](https://github.com/bboreham))
* For subprotocol negotiation, both client and server protocol is required now. ([#86646](https://github.com/kubernetes/kubernetes/pull/86646), [@tedyu](https://github.com/tedyu))
* kubeadm: use bind-address option to configure the kube-controller-manager and kube-scheduler http probes ([#86493](https://github.com/kubernetes/kubernetes/pull/86493), [@aojea](https://github.com/aojea))
* Marked scheduler's metrics scheduling_algorithm_predicate_evaluation_seconds and ([#86584](https://github.com/kubernetes/kubernetes/pull/86584), [@xiaoanyunfei](https://github.com/xiaoanyunfei))
    * scheduling_algorithm_priority_evaluation_seconds as deprecated. Those are replaced by framework_extension_point_duration_seconds[extenstion_point="Filter"] and framework_extension_point_duration_seconds[extenstion_point="Score"] respectively.
* Marked scheduler's scheduling_duration_seconds Summary metric as deprecated ([#86586](https://github.com/kubernetes/kubernetes/pull/86586), [@xiaoanyunfei](https://github.com/xiaoanyunfei))
* Add instructions about how to bring up e2e test cluster ([#85836](https://github.com/kubernetes/kubernetes/pull/85836), [@YangLu1031](https://github.com/YangLu1031))
* If a required flag is not provided to a command, the user will only see the required flag error message, instead of the entire usage menu. ([#86693](https://github.com/kubernetes/kubernetes/pull/86693), [@sallyom](https://github.com/sallyom))
* kubeadm: tolerate whitespace when validating certificate authority PEM data in kubeconfig files ([#86705](https://github.com/kubernetes/kubernetes/pull/86705), [@neolit123](https://github.com/neolit123))
* kubeadm: add support for the "ci/k8s-master" version label as a replacement for "ci-cross/*", which no longer exists. ([#86609](https://github.com/kubernetes/kubernetes/pull/86609), [@Pensu](https://github.com/Pensu))
* Fix EndpointSlice controller race condition and ensure that it handles external changes to EndpointSlices. ([#85703](https://github.com/kubernetes/kubernetes/pull/85703), [@robscott](https://github.com/robscott))
* Fix nil pointer dereference in azure cloud provider ([#85975](https://github.com/kubernetes/kubernetes/pull/85975), [@ldx](https://github.com/ldx))
* fix: azure disk could not mounted on Standard_DC4s/DC2s instances ([#86612](https://github.com/kubernetes/kubernetes/pull/86612), [@andyzhangx](https://github.com/andyzhangx))
* Fixes v1.17.0 regression in --service-cluster-ip-range handling with IPv4 ranges larger than 65536 IP addresses ([#86534](https://github.com/kubernetes/kubernetes/pull/86534), [@liggitt](https://github.com/liggitt))
* Adds back support for AlwaysCheckAllPredicates flag. ([#86496](https://github.com/kubernetes/kubernetes/pull/86496), [@ahg-g](https://github.com/ahg-g))
* Azure global rate limit is switched to per-client. A set of new rate limit configure options are introduced, including routeRateLimit, SubnetsRateLimit, InterfaceRateLimit, RouteTableRateLimit, LoadBalancerRateLimit, PublicIPAddressRateLimit, SecurityGroupRateLimit, VirtualMachineRateLimit, StorageAccountRateLimit, DiskRateLimit, SnapshotRateLimit, VirtualMachineScaleSetRateLimit and VirtualMachineSizeRateLimit. ([#86515](https://github.com/kubernetes/kubernetes/pull/86515), [@feiskyer](https://github.com/feiskyer))
    * The original rate limit options would be default values for those new client's rate limiter.
* Fix issue [#85805](https://github.com/kubernetes/kubernetes/pull/85805) about resource not found in azure cloud provider when lb specified in other resource group. ([#86502](https://github.com/kubernetes/kubernetes/pull/86502), [@levimm](https://github.com/levimm))
* `AlwaysCheckAllPredicates` is deprecated in scheduler Policy API. ([#86369](https://github.com/kubernetes/kubernetes/pull/86369), [@Huang-Wei](https://github.com/Huang-Wei))
* Kubernetes KMS provider for data encryption now supports disabling the in-memory data encryption key (DEK) cache by setting cachesize to a negative value. ([#86294](https://github.com/kubernetes/kubernetes/pull/86294), [@enj](https://github.com/enj))
* option `preConfiguredBackendPoolLoadBalancerTypes` is added to azure cloud provider for the pre-configured load balancers, possible values: `""`, `"internal"`, "external"`, `"all"` ([#86338](https://github.com/kubernetes/kubernetes/pull/86338), [@gossion](https://github.com/gossion))
* Promote StartupProbe to beta for 1.18 release ([#83437](https://github.com/kubernetes/kubernetes/pull/83437), [@matthyx](https://github.com/matthyx))
* Fixes issue where AAD token obtained by kubectl is incompatible with on-behalf-of flow and oidc. ([#86412](https://github.com/kubernetes/kubernetes/pull/86412), [@weinong](https://github.com/weinong))
    * The audience claim before this fix has "spn:" prefix. After this fix, "spn:" prefix is omitted.
* change CounterVec to Counter about  PLEGDiscardEvent ([#86167](https://github.com/kubernetes/kubernetes/pull/86167), [@yiyang5055](https://github.com/yiyang5055))
* hollow-node do not use remote CRI anymore ([#86425](https://github.com/kubernetes/kubernetes/pull/86425), [@jkaniuk](https://github.com/jkaniuk))
* hollow-node use fake CRI ([#85879](https://github.com/kubernetes/kubernetes/pull/85879), [@gongguan](https://github.com/gongguan))



# v1.18.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.18.0-alpha.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes.tar.gz) | `0c4904efc7f4f1436119c91dc1b6c93b3bd9c7490362a394bff10099c18e1e7600c4f6e2fcbaeb2d342a36c4b20692715cf7aa8ada6dfac369f44cc9292529d7`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-src.tar.gz) | `0a50fc6816c730ca5ae4c4f26d5ad7b049607d29f6a782a4e5b4b05ac50e016486e269dafcc6a163bd15e1a192780a9a987f1bb959696993641c603ed1e841c8`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `c6d75f7f3f20bef17fc7564a619b54e6f4a673d041b7c9ec93663763a1cc8dd16aecd7a2af70e8d54825a0eecb9762cf2edfdade840604c9a32ecd9cc2d5ac3c`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `ca1f19db289933beace6daee6fc30af19b0e260634ef6e89f773464a05e24551c791be58b67da7a7e2a863e28b7cbcc7b24b6b9bf467113c26da76ac8f54fdb6`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `af2e673653eb39c3f24a54efc68e1055f9258bdf6cf8fea42faf42c05abefc2da853f42faac3b166c37e2a7533020b8993b98c0d6d80a5b66f39e91d8ae0a3fb`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `9009032c3f94ac8a78c1322a28e16644ce3b20989eb762685a1819148aed6e883ca8e1200e5ec37ec0853f115c67e09b5d697d6cf5d4c45f653788a2d3a2f84f`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `afba9595b37a3f2eead6e3418573f7ce093b55467dce4da0b8de860028576b96b837a2fd942f9c276e965da694e31fbd523eeb39aefb902d7e7a2f169344d271`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `04fc3b2fe3f271807f0bc6c61be52456f26a1af904964400be819b7914519edc72cbab9afab2bb2e2ba1a108963079367cedfb253c9364c0175d1fcc64d52f5c`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `04c7edab874b33175ff7bebfff5b3a032bc6eb088fcd7387ffcd5b3fa71395ca8c5f9427b7ddb496e92087dfdb09eaf14a46e9513071d3bd73df76c182922d38`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `499287dbbc33399a37b9f3b35e0124ff20b17b6619f25a207ee9c606ef261af61fa0c328dde18c7ce2d3dfb2eea2376623bc3425d16bc8515932a68b44f8bede`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `cf84aeddf00f126fb13c0436b116dd0464a625659e44c84bf863517db0406afb4eefd86807e7543c4f96006d275772fbf66214ae7d582db5865c84ac3545b3e6`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `69f20558ccd5cd6dbaccf29307210db4e687af21f6d71f68c69d3a39766862686ac1333ab8a5012010ca5c5e3c11676b45e498e3d4c38773da7d24bcefc46d95`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `3f29df2ce904a0f10db4c1d7a425a36f420867b595da3fa158ae430bfead90def2f2139f51425b349faa8a9303dcf20ea01657cb6ea28eb6ad64f5bb32ce2ed1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `4a21073b2273d721fbf062c254840be5c8471a010bcc0c731b101729e36e61f637cb7fcb521a22e8d24808510242f4fff8a6ca40f10e9acd849c2a47bf135f27`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `7f1cb6d721bedc90e28b16f99bea7e59f5ad6267c31ef39c14d34db6ad6aad87ee51d2acdd01b6903307c1c00b58ff6b785a03d5a491cc3f8a4df9a1d76d406c`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `8f2b552030b5274b1c2c7c166eacd5a14b0c6ca0f23042f4c52efe87e22a167ba4460dcd66615a5ecd26d9e88336be1fb555548392e70efe59070dd2c314da98`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `8d9f2c96f66edafb7c8b3aa90960d29b41471743842aede6b47b3b2e61f4306fb6fc60b9ebc18820c547ee200bfedfe254c1cde962d447c791097dd30e79abdb`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `84194cb081d1502f8ca68143569f9707d96f1a28fcf0c574ebd203321463a8b605f67bb2a365eaffb14fbeb8d55c8d3fa17431780b242fb9cba3a14426a0cd4a`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `0091e108ab94fd8683b89c597c4fdc2fbf4920b007cfcd5297072c44bc3a230dfe5ceed16473e15c3e6cf5edab866d7004b53edab95be0400cc60e009eee0d9d`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `b7e85682cc2848a35d52fd6f01c247f039ee1b5dd03345713821ea10a7fa9939b944f91087baae95eaa0665d11857c1b81c454f720add077287b091f9f19e5d3`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `cd1f0849e9c62b5d2c93ff0cebf58843e178d8a88317f45f76de0db5ae020b8027e9503a5fccc96445184e0d77ecdf6f57787176ac31dbcbd01323cd0a190cbb`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `e1e697a34424c75d75415b613b81c8af5f64384226c5152d869f12fd7db1a3e25724975b73fa3d89e56e4bf78d5fd07e68a709ba8566f53691ba6a88addc79ea`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.18.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `c725a19a4013c74e22383ad3fb4cb799b3e161c4318fdad066daf806730a89bc3be3ff0f75678d02b3cbe52b2ef0c411c0639968e200b9df470be40bb2c015cc`

## Changelog since v1.17.0

### Action Required

* action required ([#85363](https://github.com/kubernetes/kubernetes/pull/85363), [@immutableT](https://github.com/immutableT))
    * 1. Currently, if users were to explicitly specify CacheSize of 0 for KMS provider, they would end-up with a provider that caches up to 1000 keys. This PR changes this behavior.
    * Post this PR, when users supply 0 for CacheSize this will result in a validation error.
    * 2. CacheSize type was changed from int32 to *int32. This allows defaulting logic to differentiate between cases where users explicitly supplied 0 vs. not supplied any value.
    * 3. KMS Provider's endpoint (path to Unix socket) is now validated when the EncryptionConfiguration files is loaded. This used to be handled by the GRPCService.

### Other notable changes

* fix: azure data disk should use same key as os disk by default ([#86351](https://github.com/kubernetes/kubernetes/pull/86351), [@andyzhangx](https://github.com/andyzhangx))
* New flag `--show-hidden-metrics-for-version` in kube-proxy can be used to show all hidden metrics that deprecated in the previous minor release. ([#85279](https://github.com/kubernetes/kubernetes/pull/85279), [@RainbowMango](https://github.com/RainbowMango))
* Remove cluster-monitoring addon ([#85512](https://github.com/kubernetes/kubernetes/pull/85512), [@serathius](https://github.com/serathius))
* Changed core_pattern on COS nodes to be an absolute path. ([#86329](https://github.com/kubernetes/kubernetes/pull/86329), [@mml](https://github.com/mml))
* Track mount operations as uncertain if operation fails with non-final error ([#82492](https://github.com/kubernetes/kubernetes/pull/82492), [@gnufied](https://github.com/gnufied))
* add kube-proxy flags --ipvs-tcp-timeout, --ipvs-tcpfin-timeout, --ipvs-udp-timeout to configure IPVS connection timeouts. ([#85517](https://github.com/kubernetes/kubernetes/pull/85517), [@andrewsykim](https://github.com/andrewsykim))
* The sample-apiserver aggregated conformance test has updated to use the Kubernetes v1.17.0 sample apiserver ([#84735](https://github.com/kubernetes/kubernetes/pull/84735), [@liggitt](https://github.com/liggitt))
* The underlying format of the `CPUManager` state file has changed. Upgrades should be seamless, but any third-party tools that rely on reading the previous format need to be updated. ([#84462](https://github.com/kubernetes/kubernetes/pull/84462), [@klueska](https://github.com/klueska))
* kubernetes will try to acquire the iptables lock every 100 msec during 5 seconds instead of every second. This specially useful for environments using kube-proxy in iptables mode with a high churn rate of services. ([#85771](https://github.com/kubernetes/kubernetes/pull/85771), [@aojea](https://github.com/aojea))
* Fixed a panic in the kubelet cleaning up pod volumes ([#86277](https://github.com/kubernetes/kubernetes/pull/86277), [@tedyu](https://github.com/tedyu))
* azure cloud provider cache TTL is configurable, list of the azure cloud provider is as following: ([#86266](https://github.com/kubernetes/kubernetes/pull/86266), [@zqingqing1](https://github.com/zqingqing1))
    * - "availabilitySetNodesCacheTTLInSeconds"
    * - "vmssCacheTTLInSeconds"
    * - "vmssVirtualMachinesCacheTTLInSeconds"
    * - "vmCacheTTLInSeconds"
    * - "loadBalancerCacheTTLInSeconds"
    * - "nsgCacheTTLInSeconds"
    * - "routeTableCacheTTLInSeconds"
* Fixes kube-proxy when EndpointSlice feature gate is enabled on Windows. ([#86016](https://github.com/kubernetes/kubernetes/pull/86016), [@robscott](https://github.com/robscott))
* Fixes wrong validation result of NetworkPolicy PolicyTypes ([#85747](https://github.com/kubernetes/kubernetes/pull/85747), [@tnqn](https://github.com/tnqn))
* Fixes an issue with kubelet-reported pod status on deleted/recreated pods. ([#86320](https://github.com/kubernetes/kubernetes/pull/86320), [@liggitt](https://github.com/liggitt))
* kube-apiserver no longer serves the following deprecated APIs: ([#85903](https://github.com/kubernetes/kubernetes/pull/85903), [@liggitt](https://github.com/liggitt))
        * All resources under `apps/v1beta1` and `apps/v1beta2` - use `apps/v1` instead
        * `daemonsets`, `deployments`, `replicasets` resources under `extensions/v1beta1` - use `apps/v1` instead
        * `networkpolicies` resources under `extensions/v1beta1` - use `networking.k8s.io/v1` instead
        * `podsecuritypolicies` resources under `extensions/v1beta1` - use `policy/v1beta1` instead
* kubeadm: fix potential panic when executing "kubeadm reset" with a corrupted kubelet.conf file ([#86216](https://github.com/kubernetes/kubernetes/pull/86216), [@neolit123](https://github.com/neolit123))
* Fix a bug in port-forward: named port not working with service ([#85511](https://github.com/kubernetes/kubernetes/pull/85511), [@oke-py](https://github.com/oke-py))
* kube-proxy no longer modifies shared EndpointSlices. ([#86092](https://github.com/kubernetes/kubernetes/pull/86092), [@robscott](https://github.com/robscott))
* allow for configuration of CoreDNS replica count ([#85837](https://github.com/kubernetes/kubernetes/pull/85837), [@pickledrick](https://github.com/pickledrick))
* Fixed a regression where the kubelet would fail to update the ready status of pods. ([#84951](https://github.com/kubernetes/kubernetes/pull/84951), [@tedyu](https://github.com/tedyu))
* Resolves performance regression in client-go discovery clients constructed using `NewDiscoveryClientForConfig` or `NewDiscoveryClientForConfigOrDie`. ([#86168](https://github.com/kubernetes/kubernetes/pull/86168), [@liggitt](https://github.com/liggitt))
* Make error message and service event message more clear ([#86078](https://github.com/kubernetes/kubernetes/pull/86078), [@feiskyer](https://github.com/feiskyer))
* e2e-test-framework: add e2e test namespace dump if all tests succeed but the cleanup fails. ([#85542](https://github.com/kubernetes/kubernetes/pull/85542), [@schrodit](https://github.com/schrodit))
* SafeSysctlWhitelist: add net.ipv4.ping_group_range ([#85463](https://github.com/kubernetes/kubernetes/pull/85463), [@AkihiroSuda](https://github.com/AkihiroSuda))
* kubelet: the metric process_start_time_seconds be marked as with the ALPHA stability level. ([#85446](https://github.com/kubernetes/kubernetes/pull/85446), [@RainbowMango](https://github.com/RainbowMango))
* API request throttling (due to a high rate of requests) is now reported in the kubelet (and other component) logs by default.  The messages are of the form ([#80649](https://github.com/kubernetes/kubernetes/pull/80649), [@RobertKrawitz](https://github.com/RobertKrawitz))
    * Throttling request took 1.50705208s, request: GET:<URL>
    * The presence of large numbers of these messages, particularly with long delay times, may indicate to the administrator the need to tune the cluster accordingly.
* Fix API Server potential memory leak issue in processing watch request. ([#85410](https://github.com/kubernetes/kubernetes/pull/85410), [@answer1991](https://github.com/answer1991))
* Verify kubelet & kube-proxy can recover after being killed on Windows nodes ([#84886](https://github.com/kubernetes/kubernetes/pull/84886), [@YangLu1031](https://github.com/YangLu1031))
* Fixed an issue that the scheduler only returns the first failure reason. ([#86022](https://github.com/kubernetes/kubernetes/pull/86022), [@Huang-Wei](https://github.com/Huang-Wei))
* kubectl/drain: add skip-wait-for-delete-timeout option. ([#85577](https://github.com/kubernetes/kubernetes/pull/85577), [@michaelgugino](https://github.com/michaelgugino))
    * If pod DeletionTimestamp older than N seconds, skip waiting for the pod.  Seconds must be greater than 0 to skip.
* Following metrics have been turned off: ([#83841](https://github.com/kubernetes/kubernetes/pull/83841), [@RainbowMango](https://github.com/RainbowMango))
    * - kubelet_pod_worker_latency_microseconds
    * - kubelet_pod_start_latency_microseconds
    * - kubelet_cgroup_manager_latency_microseconds
    * - kubelet_pod_worker_start_latency_microseconds
    * - kubelet_pleg_relist_latency_microseconds
    * - kubelet_pleg_relist_interval_microseconds
    * - kubelet_eviction_stats_age_microseconds
    * - kubelet_runtime_operations
    * - kubelet_runtime_operations_latency_microseconds
    * - kubelet_runtime_operations_errors
    * - kubelet_device_plugin_registration_count
    * - kubelet_device_plugin_alloc_latency_microseconds
    * - kubelet_docker_operations
    * - kubelet_docker_operations_latency_microseconds
    * - kubelet_docker_operations_errors
    * - kubelet_docker_operations_timeout
    * - network_plugin_operations_latency_microseconds
* - Renamed Kubelet metric certificate_manager_server_expiration_seconds to certificate_manager_server_ttl_seconds and changed to report the second until expiration at read time rather than absolute time of expiry. ([#85874](https://github.com/kubernetes/kubernetes/pull/85874), [@sambdavidson](https://github.com/sambdavidson))
    * - Improved accuracy of Kubelet metric rest_client_exec_plugin_ttl_seconds.
* Bind metadata-agent containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83363](https://github.com/kubernetes/kubernetes/pull/83363), [@wawa0210](https://github.com/wawa0210))
* Bind metrics-server containers to linux nodes to avoid Windows scheduling on kubernetes cluster includes linux nodes and windows nodes ([#83362](https://github.com/kubernetes/kubernetes/pull/83362), [@wawa0210](https://github.com/wawa0210))
* During initialization phase (preflight), kubeadm now verifies the presence of the conntrack executable ([#85857](https://github.com/kubernetes/kubernetes/pull/85857), [@hnanni](https://github.com/hnanni))
* VMSS cache is added so that less chances of VMSS GET throttling ([#85885](https://github.com/kubernetes/kubernetes/pull/85885), [@nilo19](https://github.com/nilo19))
* Update go-winio module version from 0.4.11 to 0.4.14 ([#85739](https://github.com/kubernetes/kubernetes/pull/85739), [@wawa0210](https://github.com/wawa0210))
* Fix LoadBalancer rule checking so that no unexpected LoadBalancer updates are made ([#85990](https://github.com/kubernetes/kubernetes/pull/85990), [@feiskyer](https://github.com/feiskyer))
* kubectl drain node --dry-run will list pods that would be evicted or deleted ([#82660](https://github.com/kubernetes/kubernetes/pull/82660), [@sallyom](https://github.com/sallyom))
* Windows nodes on GCE can use TPM-based authentication to the master. ([#85466](https://github.com/kubernetes/kubernetes/pull/85466), [@pjh](https://github.com/pjh))
* kubectl/drain: add disable-eviction option. ([#85571](https://github.com/kubernetes/kubernetes/pull/85571), [@michaelgugino](https://github.com/michaelgugino))
    * Force drain to use delete, even if eviction is supported. This will bypass checking PodDisruptionBudgets, and should be used with caution.
* kubeadm now errors out whenever a not supported component config version is supplied for the kubelet and kube-proxy ([#85639](https://github.com/kubernetes/kubernetes/pull/85639), [@rosti](https://github.com/rosti))
* Fixed issue with addon-resizer using deprecated extensions APIs ([#85793](https://github.com/kubernetes/kubernetes/pull/85793), [@bskiba](https://github.com/bskiba))
* Includes FSType when describing CSI persistent volumes. ([#85293](https://github.com/kubernetes/kubernetes/pull/85293), [@huffmanca](https://github.com/huffmanca))
* kubelet now exports a "server_expiration_renew_failure" and "client_expiration_renew_failure" metric counter if the certificate rotations cannot be performed. ([#84614](https://github.com/kubernetes/kubernetes/pull/84614), [@rphillips](https://github.com/rphillips))
* kubeadm: don't write the kubelet environment file on "upgrade apply" ([#85412](https://github.com/kubernetes/kubernetes/pull/85412), [@boluisa](https://github.com/boluisa))
* fix azure file AuthorizationFailure ([#85475](https://github.com/kubernetes/kubernetes/pull/85475), [@andyzhangx](https://github.com/andyzhangx))
* Resolved regression in admission, authentication, and authorization webhook performance in v1.17.0-rc.1 ([#85810](https://github.com/kubernetes/kubernetes/pull/85810), [@liggitt](https://github.com/liggitt))
* kubeadm: uses the apiserver AdvertiseAddress IP family to choose the etcd endpoint IP family for non external etcd clusters ([#85745](https://github.com/kubernetes/kubernetes/pull/85745), [@aojea](https://github.com/aojea))
* kubeadm: Forward cluster name to the controller-manager arguments ([#85817](https://github.com/kubernetes/kubernetes/pull/85817), [@ereslibre](https://github.com/ereslibre))
* Fixed "requested device X but found Y" attach error on AWS. ([#85675](https://github.com/kubernetes/kubernetes/pull/85675), [@jsafrane](https://github.com/jsafrane))
* addons: elasticsearch discovery supports IPv6 ([#85543](https://github.com/kubernetes/kubernetes/pull/85543), [@SataQiu](https://github.com/SataQiu))
* kubeadm: retry `kubeadm-config` ConfigMap creation or mutation if the apiserver is not responding. This will improve resiliency when joining new control plane nodes. ([#85763](https://github.com/kubernetes/kubernetes/pull/85763), [@ereslibre](https://github.com/ereslibre))
* Update Cluster Autoscaler to 1.17.0; changelog: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.17.0 ([#85610](https://github.com/kubernetes/kubernetes/pull/85610), [@losipiuk](https://github.com/losipiuk))
* Filter published OpenAPI schema by making nullable, required fields non-required in order to avoid kubectl to wrongly reject null values. ([#85722](https://github.com/kubernetes/kubernetes/pull/85722), [@sttts](https://github.com/sttts))
* kubectl set resources will no longer return an error if passed an empty change for a resource.  ([#85490](https://github.com/kubernetes/kubernetes/pull/85490), [@sallyom](https://github.com/sallyom))
    * kubectl set subject will no longer return an error if passed an empty change for a resource.  
* kube-apiserver: fixed a conflict error encountered attempting to delete a pod with gracePeriodSeconds=0 and a resourceVersion precondition ([#85516](https://github.com/kubernetes/kubernetes/pull/85516), [@michaelgugino](https://github.com/michaelgugino))
* kubeadm: add a upgrade health check that deploys a Job ([#81319](https://github.com/kubernetes/kubernetes/pull/81319), [@neolit123](https://github.com/neolit123))
* kubeadm: make sure images are pre-pulled even if a tag did not change but their contents changed ([#85603](https://github.com/kubernetes/kubernetes/pull/85603), [@bart0sh](https://github.com/bart0sh))
* kube-apiserver: Fixes a bug that hidden metrics can not be enabled by the command-line option `--show-hidden-metrics-for-version`. ([#85444](https://github.com/kubernetes/kubernetes/pull/85444), [@RainbowMango](https://github.com/RainbowMango))
* kubeadm now supports automatic calculations of dual-stack node cidr masks to kube-controller-manager.  ([#85609](https://github.com/kubernetes/kubernetes/pull/85609), [@Arvinderpal](https://github.com/Arvinderpal))
* Fix bug where EndpointSlice controller would attempt to modify shared objects. ([#85368](https://github.com/kubernetes/kubernetes/pull/85368), [@robscott](https://github.com/robscott))
* Use context to check client closed instead of http.CloseNotifier in processing watch request which will reduce 1 goroutine for each request if proto is HTTP/2.x . ([#85408](https://github.com/kubernetes/kubernetes/pull/85408), [@answer1991](https://github.com/answer1991))
* kubeadm: reset raises warnings if it cannot delete folders ([#85265](https://github.com/kubernetes/kubernetes/pull/85265), [@SataQiu](https://github.com/SataQiu))
* Wait for kubelet & kube-proxy to be ready on Windows node within 10s ([#85228](https://github.com/kubernetes/kubernetes/pull/85228), [@YangLu1031](https://github.com/YangLu1031))