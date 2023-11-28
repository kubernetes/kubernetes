<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.29.0-rc.1](#v1290-rc1)
  - [Downloads for v1.29.0-rc.1](#downloads-for-v1290-rc1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.29.0-rc.0](#changelog-since-v1290-rc0)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.29.0-rc.0](#v1290-rc0)
  - [Downloads for v1.29.0-rc.0](#downloads-for-v1290-rc0)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.29.0-alpha.3](#changelog-since-v1290-alpha3)
  - [Changes by Kind](#changes-by-kind)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.29.0-alpha.3](#v1290-alpha3)
  - [Downloads for v1.29.0-alpha.3](#downloads-for-v1290-alpha3)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.29.0-alpha.2](#changelog-since-v1290-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.29.0-alpha.2](#v1290-alpha2)
  - [Downloads for v1.29.0-alpha.2](#downloads-for-v1290-alpha2)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.29.0-alpha.1](#changelog-since-v1290-alpha1)
  - [Changes by Kind](#changes-by-kind-2)
    - [Feature](#feature-2)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.29.0-alpha.1](#v1290-alpha1)
  - [Downloads for v1.29.0-alpha.1](#downloads-for-v1290-alpha1)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
    - [Container Images](#container-images-4)
  - [Changelog since v1.28.0](#changelog-since-v1280)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-2)
    - [Feature](#feature-3)
    - [Documentation](#documentation-1)
    - [Failing Test](#failing-test-2)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)

<!-- END MUNGE: GENERATED_TOC -->

# v1.29.0-rc.1


## Downloads for v1.29.0-rc.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes.tar.gz) | e44677de7af6634c31b86672dc6755d97ae145fd0497229c08b156d11dcdbc922f15c715fd878b585d21a6c7dd10fde0b43135f0b6f7e77a9f957f2280a32018
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-src.tar.gz) | 63e197478e315a64dae6282c1e4ce2b672f0a3941bea9920094b703d44a09aaed74228a7c29fbec4051a4cb832f6e791d54f5e76e006aada1216a502c6d2e744

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | 167d931f6b9540b9fbd8501e3dd9d2ea032ead96f57cf4a929020fff3c4efb0456e3bc5eee2a8436670589ef18b48018dabf57081ff13393542007e9d0c8cb72
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-darwin-arm64.tar.gz) | 6623d8b08b69beea8832a1130c50624f248464a01fec4fce720700ccebdd3ed440af664f5beb49a294557db3c4ff7a8fecfd3cf48f9dabcc48b9bda2d791c08e
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-linux-386.tar.gz) | 11f6c0d6f0954938c4217436536a67713c403b1f3c2b988d26944374fab6f70c385cb9356e6aa51b480fdcd07539de4667be2e72fa8128ba792a607bb388254f
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | ee7605531629a2e320f299ef49bfb87566b73245f16dee79a40b824d637bd97ca8bd7f81a177320c2e4bbb82acd7f5d840359dd79c3062c2136e0b5f04eeb90e
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-linux-arm.tar.gz) | baa1af4d932c3d36ff084f2fc4c7676e76a2f0e4c4c746495f932c56a583b4390376b0b631b13f6dc3b03bd874d84c20f82e71d2483940b37ea439bc7d21dadf
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | 26a20dfbeca7abb73a73f1cfb5337b4af71bb3d2810d053f9d94e4a3709d282e515a542f02502a7e86adf3b32ac52e7b434d1604fa9664729682d083a55b314d
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | eca5ca3028b64ea44138b08831e998ac85bd054785099366e295b88b8b568c455a54f4fd110b568208169b9a3aef918c7f6caf8e05f9e73f85c26f973a589e2d
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | 5bcde8b36b8dc1d3ed83337322fc260311238e9f067301838c5002bb0dc63153f82c2602d8c9a28c1ae5dd85b0eecc4d3e8c31dcf75f4653c6b9bebd3a564321
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-windows-386.tar.gz) | d183c3183bfac0878377eaa8adf00e6ecdb3f252ed47180b8f9231d757b44c30931620b04da52c3470a1a278fa5a76e99f0b7c587b696f517caec5ff16103480
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | 7a51ed89ad5f850bfc94e4175294d944eae9628c281fea2c18939417f84c438b82246d262e645bea9fd257deb60b51c05b1c1ebd321b35aafeb87d1b4f83ebe6
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-client-windows-arm64.tar.gz) | 2efc1fd75461ed5e0bceba78681804567e731a545a801b28e8291d2f3cc8e2c8c22d3e418888a666dc1e40754681acbcbfc64fc81772088b8524fde7c55e7e3a

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | 28eeda8ab821891ca445b4a884ae70028146f6d4264e364a7ca88c819ea80bd95653c7f538af6c5f660921f140026f3d61c469afb0109c2f5111245592439fd7
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | d786685de63f060910181f0de66511fe8f8f0fa66f00ee0c76957246a62e3de3208ec009908c6700ff83dc7c5e5c24f3c1c3118d06b07568fcb004190d1e5fe6
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | b9d14fdcf282b4f9e523d81f71cd82d64bfb9bf9ba4affb7d6dbcfa8191f9b0d238f7091c65c1f1d516c6bcaa13f68affdd3bfda1ec759f35d505284a87494e1
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | f84c315e3d3ab6e3124106783e43b18d407d5c4ef09910641ae51c034b550fba0581515181ae4d355eea5b75eb0688460a891141c47d13be902e06156908d26d

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | b2a974c505878c757bc643c64cf24208ed92bfe7943f6e9e50d215f027e4fc2f4a578d3975721f53ff6a06e6560dec362887a236052213784222695c916a59fb
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | 7f7b6a3bcdef051ae0126a811d02b327f6f45b13050934dd4d90219a23efd5aeabb945e040a9aa57a24a4151222b558a4e2a996bad5e5f7ef508c26e6cca85ee
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | 6245a8645a6b52c2bd40e1104d4fbba015bf583f50478b2686ecf0c3c8fdc05c6729da76dc6301ca0f2f0d63f1a5df84a0871fb4793fc3aa9e8e2b6cdae37d34
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | fe530554696b84db43372ca48e95a01c46cf3a09dd51fb569b1792a341827a428bfbb04c3bab59d6aea095687eafe144534a0d98df8852d469e8bdc69a8d2d1a
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | 0a80378b4037f8d325fdbdc065ce5fd841b0b66242d64ec5c6b8ad6af0e430fc020d4fb0d624909b0725761d2c249a8f75c91eae22af03cefe4a4338b57d3d29

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.29.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.29.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.29.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.29.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.29.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.29.0-rc.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.29.0-rc.0

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.29.0-rc.0


## Downloads for v1.29.0-rc.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes.tar.gz) | 6a5b027a35b96d1cf8495efce0f9f518499b94e63e1d11058876d1b364d0bba42ccedac4612082771eb38cd54be0d8868a808de05c7e9077b8644f15a5c6f413
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-src.tar.gz) | d92897e5e28a14f0fbd3f03e9016e9c86f30bf097c4e709e6dba74b1a9897ce016e3c3a44aed9d5f851af1f5d5bd0ea2240efe8d8d12d7893b7f9cff66caff55

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | a4e8dd4e65158024a46843701ed24082eefde5d407c6d6a191b7b7f690413ea65c5422ba578e2813cd6624ac7327174554d879dbbbb324b56fbfe99892eb8d80
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | ef14378eaa3a35a34ab5e9b06c9856ff46165bbee2a4efc1b8512de47e8f584449d94155665978eca6264e23f131e31d072f9333117c11a3e92aadeea367b8e5
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-linux-386.tar.gz) | 686a8b69525e8e1494cdc890e8023ba60f86e41ceb28cb5df7e33f152ecc3ac8c62b0b1d24fa6c8198278a9d585bfd8962d058daf7f27dfa658580598b45cafe
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | 7ebe8d866f8fd1dccd6761be0ad5096cb861e5fb20bdea0ac65a3d63230d9a7d47df16a6933fcf4069cf819ab90d12eaf87ec53873eacd88c3feab009e85e430
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-linux-arm.tar.gz) | f8a336b48c27819f979336fff3ffa7eeb5512330f3eafe7a3b85ea65a4d94b213ed20d4d35d6fe3f92cb557037a051023eb47fd6e6dcaf3e0a6fe88a5c6cd632
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | a72602ac48b13c6a97883c34170fd64095539d4f9a3900367ff628a195aa931c27d7c9582f864c669332bbc58b4883d5e41bc65d5ac83337bdf7066e538deceb
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | 002b2e685758ad6fa2a18d7706a335249f55a786a4315d3f2cab8e34d38a01302af91063742729de664c7ab06bd656b388166f82010af36e43e931e8ddd93752
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | 21da21e1f7ba24b6967b5e22abb62e1c1691cd7cc15eb5ecd9777fa51d788a7a132f31c04306d3a59e5cee96bb58b9d1838630de1bcbf168cabe8f4afb514501
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-windows-386.tar.gz) | 21494c5fe65e6a9aaf2f7f11996219155ed85a4f54d048b64df05de1adbd925af40ee51d4119801333143364902b9805cefceafac8d407f62eef1e7f07b686ee
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | eaaddeac2e0a69a618f606574044eec8b41f4c3d4f6cf0045e4456ad57d44c865d1f183b6e0929f6913e28febe67b178662dfce3e40395c6d97180985b4fb48d
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | dad6a73bf2530c0c2f58b8e77956ec444b6795c9882b0f2b960998fbd9e22720fc6fff114af3b0ad10655e9e1d627f70bc6f67fdd388d0e995aeb9bd4bf9bea2

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | c64651213144ef4696fa11da0ec93c6fd7540798bfc28df8e69ee8bdb35dbc7114ee043cd38dc86c75a3dbff5e45ed4474be22ce74b8c4b3206030a10cd20f8d
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | 0724ce02d551d72f39c7ac6b29c78dcaeae7878126b33cde7a949d0b9be0b35b3977f5494ab48f02a382bf83a70a8ad035f4962b0644a6fedc084068b525ddf9
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | 7f527bb02e046308b2720a99d8f6ac13e1daee23b44e77603b75aa5569a9b4baf29a7b19f3076219ff94f296ce8316fdaadb9cabaf1c58173a7e3719e94f3917
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | 0285e04f2834bdbb66b46193f54724e6f9264ff992b10dbaa3694abbab297f5e1f4e95ade14f7dcb41f856d9e3a292f1af16f3ed59e2b02961451973d4972f1d

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | b4c5e4a0e818eb9f88128e2a051591b4955a858e400489d04b75cdfb68eb3a7d004ced839c2916bf5ca885d7ae496fb68c0620b2c3352cb5435c435756b0a70a
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | e525decd637860b9621ec7ec8c42913c419bb81577a1a359e752e7628507b6e9b1a82889ef0ba17ca975aa8630edba12a87d38b75ea3f9e213493873036b92c8
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | 4282286b775a5bdaab753c911fa0f351476d89070a34569bedb104cb2c56a408d125d44c4895cc28fcb8cd5c12585f3cbecccfe045880b546766202113a703b1
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | af88dac8622e10e336e5d79f9d4511de3eceed384da210dda83223c7b6582133acaa7d8f6b361cf213a4ca3ea51379bd828816c991ed7b4c62cf6fd9830f0c30
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | 7b322df6a7e9e0b0b881f99d7ef76b3eda0f856345eb888efa62daf1f3638f88a630fc40626800075db90215c68a956fec7ce274381e06a173d494e4b03b4f49

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.29.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.29.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.29.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.29.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.29.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.29.0-rc.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.29.0-alpha.3

## Changes by Kind

### API Change

- Added support for projecting certificates.k8s.io/v1alpha1 ClusterTrustBundle objects into pods. ([#113374](https://github.com/kubernetes/kubernetes/pull/113374), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Apps, Auth, Node, Storage and Testing]
- Adds `optionalOldSelf` to `x-kubernetes-validations` to support ratcheting CRD schema constraints ([#121034](https://github.com/kubernetes/kubernetes/pull/121034), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- Fix API comment for the Job Ready field in status ([#121765](https://github.com/kubernetes/kubernetes/pull/121765), [@mimowo](https://github.com/mimowo)) [SIG API Machinery and Apps]
- Fix API comments for the FailIndex Job pod failure policy action. ([#121764](https://github.com/kubernetes/kubernetes/pull/121764), [@mimowo](https://github.com/mimowo)) [SIG API Machinery and Apps]

### Feature

- A customizable OrderedScoreFuncs() function is introduced. Out-of-tree plugins that use scheduler's preemption interface can implement this function for custom preemption preferences, or return nil to keep current behavior. ([#121867](https://github.com/kubernetes/kubernetes/pull/121867), [@lianghao208](https://github.com/lianghao208)) [SIG Scheduling]
- Bump distroless-iptables to 0.4.1 based on Go 1.21.3 ([#121871](https://github.com/kubernetes/kubernetes/pull/121871), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- Fix overriding default KubeletConfig fields in drop-in configs if not set ([#121193](https://github.com/kubernetes/kubernetes/pull/121193), [@sohankunkerkar](https://github.com/sohankunkerkar)) [SIG Node and Testing]
- KEP-4191- add support for split image filesystem in kubelet ([#120616](https://github.com/kubernetes/kubernetes/pull/120616), [@kannon92](https://github.com/kannon92)) [SIG Node and Testing]
- Kubeadm: support updating certificate organization during 'kubeadm certs renew' ([#121841](https://github.com/kubernetes/kubernetes/pull/121841), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Go 1.21.4 ([#121808](https://github.com/kubernetes/kubernetes/pull/121808), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Storage and Testing]

### Bug or Regression

- Fix: statle smb mount issue when smb file share is deleted and then unmount ([#121851](https://github.com/kubernetes/kubernetes/pull/121851), [@andyzhangx](https://github.com/andyzhangx)) [SIG Storage]
- KCCM: fix transient node addition + removal caused by #121090 while syncing load balancers on large clusters with a lot of churn ([#121091](https://github.com/kubernetes/kubernetes/pull/121091), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider, Network and Testing]
- Kubeadm: change the "system:masters" Group in the apiserver-kubelet-client.crt certificate Subject to be "kubeadm:cluster-admins" which is a less privileged Group. ([#121837](https://github.com/kubernetes/kubernetes/pull/121837), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Scheduler: in 1.29 pre-releases, enabling contextual logging slowed down pod scheduling. ([#121715](https://github.com/kubernetes/kubernetes/pull/121715), [@pohly](https://github.com/pohly)) [SIG Instrumentation and Scheduling]

### Other (Cleanup or Flake)

- Update runc to 1.1.10 ([#121739](https://github.com/kubernetes/kubernetes/pull/121739), [@ty-dc](https://github.com/ty-dc)) [SIG Architecture and Node]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/mrunalp/fileutils: [v0.5.0 → v0.5.1](https://github.com/mrunalp/fileutils/compare/v0.5.0...v0.5.1)
- github.com/opencontainers/runc: [v1.1.9 → v1.1.10](https://github.com/opencontainers/runc/compare/v1.1.9...v1.1.10)

### Removed
_Nothing has changed._



# v1.29.0-alpha.3


## Downloads for v1.29.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes.tar.gz) | 998a680aee880601d65c14cf43a8ace13aacb3d693ac2f32c40ddc5c0a567fd4cc5627f397bf5612ed83d6b37ef568260f2700d46592bbc74174e155bf8f0606
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-src.tar.gz) | ca46836dabd989a8dc6ee61032ab7f73747a5e2ef3bc11437e4036d95cbfbb9574f647b1672a098625729b62f1ff663726fcaf2dc3ea472e7b27d6b373d8afa9

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | b82008b54b2a90e3640e786782cc20cf3a7d6a5011974f6710d418770541b53edb7d9d4ccd9489d4d81fcf7df7db38a3766db19898c86381b6fcfd7b261bc06a
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | b389eece6ea7ba07fdff76a6acdf36e77ed81e474277b62ef40b91ccc0d00c37f6f7c1194cacb14df844b1ea4dc66895b1c73bfedba570c71b72c6ab9a697861
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 5fa044082a1d2fb9d0b428fd2ba913196b4891f4c0571a7061ef1b6fee19ff820ff2b67506edad27fe0ddc735630d7398f66160836620188a527a8f3dbeb6b09
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | 782d262f696e9b706de195870e5589fe3a0c4c11698574709668d4f60fcfe3cfb0137e86fb2d43a27a297bf88c29552216d30ac72255b4a757525f0b7e2385a1
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | cd9038cd3fa938aac9a0b462f7c6822d031f4e05e2529df378b4069f2d69d362236e5fc6e464d20cc42549f84f283f302b6cb33eb4a128ab91dcaa1cf04552e8
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | d0fdf61def1be6c3b9e5259c13e8dbd764af44ed3dcdeb83c6a7d6cfe87b2293cbd88ceaad0aac87b448fe4766635b6b9eca40bc1a302f717d1bc0e26dac60ed
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | c8b148404eecdff20939f0bec92024be58cb9629802c3768085834998e97b82e87f37443a867dfbb73e9922aada038d308ef02ec52a078aefb1f76360220c77b
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | fb7070ef9d610fae614eadf9ec7fcfe68958143010b709586f0339309336e87f06d03ff8df5108b340ea063082e7b1d393b519ee6a4b4ed302427fd66e896295
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-windows-386.tar.gz) | c0021a7668504a0a2be408cb2d1754bd20fc9afeeeb31dfb11f11787eb7047540c544888058300a83662dab845c726c7001562ee712b4c1d485ff0c3a88827f9
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 5678ab6523345ec38ba49a81c945223112228e75b82d0b459f0c9d6c37d3a0c93af4b82f05226e2d1110c1315ae5ed7ed3ad0bb085afad7f376b9715fe8fee75
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | 2d7ac1add995683b29396fbbc06b15bf81ca62338ba0ed6d4738283752560fa167d6b9ffc44aeb4ed9b1bae6bc2ed8fe1d6346436c34a2acd3a5467ac68041bb

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | a948e26c77fb7cef3c50543c0b92c1ec1085516c4f16f9cd6695a02e1d88e44d5cd1f6fcc13bda48a580a26d024d1665ca2a960b37b31f0a7f0186e941a21e7e
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 14538ca02dc149a57c1cb30206b281248f9a84b024dad777e0326a9c6dc6c74228211e1a49f0480e2b7f825b12891d176489bcafcfab0fa05b18acc33c5044f8
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | d347d6072f5a4c6c14ddb9418eadcc075824fa2dd15e49bfb79ee3fab7b2cc0efdd18021d7baed9024a4b83b4f9b800cedaeb8fa3917bfd47c4e5935146fc9fc
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 210ed1f933ba611cc3a828382ad15b02d2e35e74e99baed7077c21708249c5a561963e25f2582562773f1ea8f3eccb89b9fa25ca58da6eec9a516652efd432a5

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | 6d9b9e382a3137a2622a4631162b5c6a0c0c709fe95b76a7d5af610aec2a292b2f5a0b3378ddf8243450d774f6c1cd2ca16cbf240aed7109d819cd366b7abda9
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | 0951c701155c914a0578dab9c8d584d32e8260ca923e1efccd0739db2065bf3d37d5d1b6584bc38f7873e20d164e57603e79abfecfbbe89e5386b6d7738d521b
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | afbbedb58bd8344608e1fe047666914874419aef7f31c057a992e0dc24acae6151a7b0c53c2cfc8144ab8e0e914ee8b3a2f11adbb3791fb3b412172ade67439d
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | 10c73d669dde0841078e5cee9158fa1a551c8bfe668d07beadf316386f815979f8729b89a0bed7e9e76350e82f6fd94d204187b9c3fe6e2bc1aabb2e580fee87
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.29.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | c94d9f4979aeebfae9e66e029ea99ef6e349209f641d853032f87bb9cb646e885b995a512be3eef8e8cf2def76418e70086a03c00121e22c082b67b45562a6c5

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.29.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.29.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.29.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.29.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.29.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.29.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.29.0-alpha.2

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Kubeadm: deploy a separate "super-admin.conf" file. The User in "admin.conf" is now bound to a new RBAC Group "kubeadm:cluster-admins" that have "cluster-admin" ClusterRole access. The User in "super-admin.conf" is bound to the "system:masters" built-in super-powers / break-glass Group that can bypass RBAC. Before this change the default "admin.conf" was bound to "system:masters" Group which was undesired. Executing "kubeadm init phase kubeconfig all" or just "kubeadm init" will now generate the new "super-admin.conf" file. The cluster admin can then decide to keep the file present on a node host or move it to a safe location. "kubadm certs renew" will renew the certificate in "super-admin.conf" to one year if the file exists. If it does not exist a "MISSING" note will be printed. "kubeadm upgrade apply" for this release will migrate this particular node to the two file setup. Subsequent kubeadm releases will continue to optionally renew the certificate in "super-admin.conf" if the file exists on disk and if renew on upgrade is not disabled. "kubeadm join --control-plane" will now generate only an "admin.conf" file that has the less privileged User. ([#121305](https://github.com/kubernetes/kubernetes/pull/121305), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
  - Stop accepting component configuration for kube-proxy and kubelet during `kubeadm upgrade plan --config`. This is a legacy behavior that is not well supported for upgrades and can be used only at the plan stage to determine if the configuration for these components stored in the cluster needs manual version  migration. In the future, kubeadm will attempt alternative component config migration approaches. ([#120788](https://github.com/kubernetes/kubernetes/pull/120788), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
 
## Changes by Kind

### Deprecation

- Creation of new CronJob objects containing `TZ` or `CRON_TZ` in `.spec.schedule`, accidentally enabled in 1.22, is now disallowed. Use the `.spec.timeZone` field instead, supported in 1.25+ clusters in default configurations. See https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/#unsupported-timezone-specification for more information. ([#116252](https://github.com/kubernetes/kubernetes/pull/116252), [@soltysh](https://github.com/soltysh)) [SIG Apps]
- Remove the networking alpha API ClusterCIDR ([#121229](https://github.com/kubernetes/kubernetes/pull/121229), [@aojea](https://github.com/aojea)) [SIG Apps, CLI, Cloud Provider, Network and Testing]

### API Change

- A new sleep action for the PreStop lifecycle hook is added, allowing containers to pause for a specified duration before termination. ([#119026](https://github.com/kubernetes/kubernetes/pull/119026), [@AxeZhan](https://github.com/AxeZhan)) [SIG API Machinery, Apps, Node and Testing]
- Add ImageMaximumGCAge field to Kubelet configuration, which allows a user to set the maximum age an image is unused before it's garbage collected. ([#121275](https://github.com/kubernetes/kubernetes/pull/121275), [@haircommander](https://github.com/haircommander)) [SIG API Machinery and Node]
- Add a new ServiceCIDR type that allows to dynamically configure the cluster range used to allocate Service ClusterIPs addresses ([#116516](https://github.com/kubernetes/kubernetes/pull/116516), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Auth, CLI, Network and Testing]
- Add the DisableNodeKubeProxyVersion feature gate. If DisableNodeKubeProxyVersion is enabled, the kubeProxyVersion field is not set. ([#120954](https://github.com/kubernetes/kubernetes/pull/120954), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery, Apps and Node]
- Added Windows support for InPlace Pod Vertical Scaling feature. ([#112599](https://github.com/kubernetes/kubernetes/pull/112599), [@fabi200123](https://github.com/fabi200123)) [SIG Autoscaling, Node, Scalability, Scheduling and Windows]
- Added `UserNamespacesPodSecurityStandards` feature gate to enable user namespace support for Pod Security Standards.
  Enabling this feature will modify all Pod Security Standard rules to allow setting: `spec[.*].securityContext.[runAsNonRoot,runAsUser]`.
  This feature gate should only be enabled if all nodes in the cluster support the user namespace feature and have it enabled.
  The feature gate will not graduate or be enabled by default in future Kubernetes releases. ([#118760](https://github.com/kubernetes/kubernetes/pull/118760), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery, Auth, Node and Release]
- Added options for configuring nf_conntrack_udp_timeout, and nf_conntrack_udp_timeout_stream variables of netfilter conntrack subsystem. ([#120808](https://github.com/kubernetes/kubernetes/pull/120808), [@aroradaman](https://github.com/aroradaman)) [SIG API Machinery and Network]
- Adds CEL expressions to v1alpha1 AuthenticationConfiguration. ([#121078](https://github.com/kubernetes/kubernetes/pull/121078), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Adds support for CEL expressions to v1alpha1 AuthorizationConfiguration webhook matchConditions. ([#121223](https://github.com/kubernetes/kubernetes/pull/121223), [@ritazh](https://github.com/ritazh)) [SIG API Machinery and Auth]
- CSINodeExpandSecret feature has been promoted to GA in this release and enabled by default. The CSI drivers can make use of the `secretRef` values passed in NodeExpansion request optionally sent by the CSI Client from this release onwards. ([#121303](https://github.com/kubernetes/kubernetes/pull/121303), [@humblec](https://github.com/humblec)) [SIG API Machinery, Apps and Storage]
- Graduate Job BackoffLimitPerIndex feature to Beta ([#121356](https://github.com/kubernetes/kubernetes/pull/121356), [@mimowo](https://github.com/mimowo)) [SIG Apps]
- Kube-apiserver: adds --authorization-config flag for reading a configuration file containing an apiserver.config.k8s.io/v1alpha1 AuthorizationConfiguration object. --authorization-config flag is mutually exclusive with --authorization-modes and --authorization-webhook-* flags. The alpha StructuredAuthorizationConfiguration feature flag must be enabled for --authorization-config to be specified. ([#120154](https://github.com/kubernetes/kubernetes/pull/120154), [@palnabarun](https://github.com/palnabarun)) [SIG API Machinery, Auth and Testing]
- Kube-proxy now has a new nftables-based mode, available by running
  
      kube-proxy --feature-gates NFTablesProxyMode=true --proxy-mode nftables
  
  This is currently an alpha-level feature and while it probably will not
  eat your data, it may nibble at it a bit. (It passes e2e testing but has
  not yet seen real-world use.)
  
  At this point it should be functionally mostly identical to the iptables
  mode, except that it does not (and will not) support Service NodePorts on
  127.0.0.1. (Also note that there are currently no command-line arguments
  for the nftables-specific config; you will need to use a config file if
  you want to set the equivalent of any of the `--iptables-xxx` options.)
  
  As this code is still very new, it has not been heavily optimized yet;
  while it is expected to _eventually_ have better performance than the
  iptables backend, very little performance testing has been done so far. ([#121046](https://github.com/kubernetes/kubernetes/pull/121046), [@danwinship](https://github.com/danwinship)) [SIG API Machinery and Network]
- Kube-proxy: Added an option/flag for configuring the `nf_conntrack_tcp_be_liberal` sysctl (in the kernel's netfilter conntrack subsystem).  When enabled, kube-proxy will not install the DROP rule for invalid conntrack states, which currently breaks users of asymmetric routing. ([#120354](https://github.com/kubernetes/kubernetes/pull/120354), [@aroradaman](https://github.com/aroradaman)) [SIG API Machinery and Network]
- PersistentVolumeLastPhaseTransitionTime is now beta, enabled by default. ([#120627](https://github.com/kubernetes/kubernetes/pull/120627), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]
- Promote PodReadyToStartContainers condition to beta. ([#119659](https://github.com/kubernetes/kubernetes/pull/119659), [@kannon92](https://github.com/kannon92)) [SIG Node and Testing]
- The flowcontrol.apiserver.k8s.io/v1beta3 FlowSchema and PriorityLevelConfiguration APIs has been promoted to flowcontrol.apiserver.k8s.io/v1, with the following changes:
  - PriorityLevelConfiguration: the `.spec.limited.nominalConcurrencyShares` field defaults to `30` only if the field is omitted (v1beta3 also defaulted an explicit `0` value to `30`). Specifying an explicit `0` value is not allowed in the `v1` version in v1.29 to ensure compatibility with 1.28 API servers. In v1.30, explicit `0` values will be allowed in this field in the `v1` API.
  The flowcontrol.apiserver.k8s.io/v1beta3 APIs are deprecated and will no longer be served in v1.32. All existing objects are available via the `v1` APIs. Transition clients and manifests to use the `v1` APIs before upgrading to v1.32. ([#121089](https://github.com/kubernetes/kubernetes/pull/121089), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Testing]
- The kube-proxy command-line documentation was updated to clarify that
  `--bind-address` does not actually have anything to do with binding to an
  address, and you probably don't actually want to be using it. ([#120274](https://github.com/kubernetes/kubernetes/pull/120274), [@danwinship](https://github.com/danwinship)) [SIG Network]
- The matchLabelKeys/mismatchLabelKeys feature is introduced to the hard/soft PodAffinity/PodAntiAffinity. ([#116065](https://github.com/kubernetes/kubernetes/pull/116065), [@sanposhiho](https://github.com/sanposhiho)) [SIG API Machinery, Apps, Cloud Provider, Scheduling and Testing]
- ValidatingAdmissionPolicy Type Checking now supports CRDs and API extensions types. ([#119109](https://github.com/kubernetes/kubernetes/pull/119109), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery, Apps, Auth and Testing]
- When updating a CRD, per-expression cost limit check is skipped for x-kubernetes-validations rules of versions that are not mutated. ([#121460](https://github.com/kubernetes/kubernetes/pull/121460), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery]

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
  --> ([#119517](https://github.com/kubernetes/kubernetes/pull/119517), [@sanposhiho](https://github.com/sanposhiho)) [SIG Node, Scheduling and Testing]
- --interactive flag in kubectl delete will be visible to all users by default. ([#120416](https://github.com/kubernetes/kubernetes/pull/120416), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Add container filesystem to the ImageFsInfoResponse. ([#120914](https://github.com/kubernetes/kubernetes/pull/120914), [@kannon92](https://github.com/kannon92)) [SIG Node and Testing]
- Add job_pods_creation_total metrics for tracking Pods created by the Job controller labeled by events which triggered the Pod creation ([#121481](https://github.com/kubernetes/kubernetes/pull/121481), [@dejanzele](https://github.com/dejanzele)) [SIG Apps and Testing]
- Add multiplication functionality to Quantity. ([#117411](https://github.com/kubernetes/kubernetes/pull/117411), [@tenzen-y](https://github.com/tenzen-y)) [SIG API Machinery]
- Added a new `--init-only` command line flag to `kube-proxy`. Setting the flag makes `kube-proxy` perform its initial configuration that requires privileged mode, and then exit. The `--init-only` mode is intended to be executed in a privileged init container, so that the main container may run with a stricter `securityContext`. ([#120864](https://github.com/kubernetes/kubernetes/pull/120864), [@uablrek](https://github.com/uablrek)) [SIG Network and Scalability]
- Added new feature gate called "RuntimeClassInImageCriApi" to address kubelet changes needed for KEP 4216.
  Noteable changes:
  1. Populate new RuntimeHandler field in CRI's ImageSpec struct during image pulls from container runtimes.
  2. Pass runtimeHandler field in RemoveImage() call to container runtime in kubelet's image garbage collection ([#121456](https://github.com/kubernetes/kubernetes/pull/121456), [@kiashok](https://github.com/kiashok)) [SIG Node and Windows]
- Adds `apiextensions_apiserver_update_ratcheting_time` metric for tracking time taken during requests by feature `CRDValidationRatcheting` ([#121462](https://github.com/kubernetes/kubernetes/pull/121462), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- Bump cel-go to v0.17.7 and introduce set ext library with new options. ([#121577](https://github.com/kubernetes/kubernetes/pull/121577), [@cici37](https://github.com/cici37)) [SIG API Machinery, Auth and Cloud Provider]
- Bump distroless-iptables to 0.4.1 based on Go 1.21.3 ([#121216](https://github.com/kubernetes/kubernetes/pull/121216), [@cpanato](https://github.com/cpanato)) [SIG Testing]
- CEL can now correctly handle a CRD openAPIV3Schema that has neither Properties nor AdditionalProperties. ([#121459](https://github.com/kubernetes/kubernetes/pull/121459), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery and Testing]
- CEL cost estimator no longer treats enums as unbounded strings when determining its length. Instead, the length is set to the longest possible enum value. ([#121085](https://github.com/kubernetes/kubernetes/pull/121085), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery]
- CRDValidationRatcheting: Adds support for ratcheting `x-kubernetes-validations` in schema ([#121016](https://github.com/kubernetes/kubernetes/pull/121016), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- CRI: support image pull per runtime class ([#121121](https://github.com/kubernetes/kubernetes/pull/121121), [@kiashok](https://github.com/kiashok)) [SIG Node and Windows]
- Calculate restartable init containers resource in pod autoscaler ([#120001](https://github.com/kubernetes/kubernetes/pull/120001), [@qingwave](https://github.com/qingwave)) [SIG Apps and Autoscaling]
- Certain requestBody params in the OpenAPI v3 are correctly marked as required ([#120735](https://github.com/kubernetes/kubernetes/pull/120735), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node and Storage]
- Client-side apply will use OpenAPI V3 by default ([#120707](https://github.com/kubernetes/kubernetes/pull/120707), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery and CLI]
- Cluster/gce: add webhook to replace PersistentVolumeLabel admission controller ([#121628](https://github.com/kubernetes/kubernetes/pull/121628), [@andrewsykim](https://github.com/andrewsykim)) [SIG Cloud Provider]
- Decouple TaintManager from NodeLifeCycleController (KEP-3902) ([#119208](https://github.com/kubernetes/kubernetes/pull/119208), [@atosatto](https://github.com/atosatto)) [SIG API Machinery, Apps, Instrumentation, Node, Scheduling and Testing]
- DevicePluginCDIDevices feature has been graduated to Beta and enabled by default in the Kubelet ([#121254](https://github.com/kubernetes/kubernetes/pull/121254), [@bart0sh](https://github.com/bart0sh)) [SIG Node]
- Dra: the scheduler plugin avoids additional scheduling attempts in some cases by falling back to SSA after a conflict ([#120534](https://github.com/kubernetes/kubernetes/pull/120534), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Enable traces for KMSv2 encrypt/decrypt operations. ([#121095](https://github.com/kubernetes/kubernetes/pull/121095), [@aramase](https://github.com/aramase)) [SIG API Machinery, Architecture, Auth, Instrumentation and Testing]
- Etcd: build image for v3.5.9 ([#121567](https://github.com/kubernetes/kubernetes/pull/121567), [@mzaian](https://github.com/mzaian)) [SIG API Machinery]
- Fixes bugs in handling of server-side apply, create, and update API requests for objects containing duplicate items in keyed lists.
  - A `create` or `update` API request with duplicate items in a keyed list no longer wipes out managedFields. Examples include env var entries with the same name, or port entries with the same containerPort in a pod spec.
  - A server-side apply request that makes unrelated changes to an object which has duplicate items in a keyed list no longer fails, and leaves the existing duplicate items as-is.
  - A server-side apply request that changes an object which has duplicate items in a keyed list, and modifies the duplicated item removes the duplicates and replaces them with the single item contained in the server-side apply request. ([#121575](https://github.com/kubernetes/kubernetes/pull/121575), [@apelisse](https://github.com/apelisse)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Storage and Testing]
- Graduate the `ReadWriteOncePod` feature gate to GA ([#121077](https://github.com/kubernetes/kubernetes/pull/121077), [@chrishenzie](https://github.com/chrishenzie)) [SIG Apps, Node, Scheduling, Storage and Testing]
- Introduce the job_finished_indexes_total metric for BackoffLimitPerIndex feature ([#121292](https://github.com/kubernetes/kubernetes/pull/121292), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- KEP-4191- add support for split image filesystem in kubelet ([#120616](https://github.com/kubernetes/kubernetes/pull/120616), [@kannon92](https://github.com/kannon92)) [SIG Node and Testing]
- Kube-apiserver adds alpha support (guarded by the ServiceAccountTokenJTI feature gate) for adding a `jti` (JWT ID) claim to service account tokens it issues, adding an `authentication.kubernetes.io/credential-id` audit annotation in audit logs when the tokens are issued, and `authentication.kubernetes.io/credential-id` entry in the extra user info when the token is used to authenticate.
  - kube-apiserver adds alpha support (guarded by the ServiceAccountTokenPodNodeInfo feature gate) for including the node name (and uid, if the node exists) as additional claims in service account tokens it issues which are bound to pods, and `authentication.kubernetes.io/node-name` and `authentication.kubernetes.io/node-uid` extra user info when the token is used to authenticate.
  - kube-apiserver adds alpha support (guarded by the ServiceAccountTokenNodeBinding feature gate) for allowing TokenRequests that bind tokens directly to nodes, and (guarded by the ServiceAccountTokenNodeBindingValidation feature gate) for validating the node name and uid still exist when the token is used. ([#120780](https://github.com/kubernetes/kubernetes/pull/120780), [@munnerz](https://github.com/munnerz)) [SIG API Machinery, Apps, Auth, CLI and Testing]
- Kube-controller-manager: The `LegacyServiceAccountTokenCleanUp` feature gate is now beta and enabled by default. When enabled, legacy auto-generated service account token secrets are auto-labeled with a `kubernetes.io/legacy-token-invalid-since` label if the credentials have not been used in the time specified by `--legacy-service-account-token-clean-up-period` (defaulting to one year), **and** are referenced from the `.secrets` list of a ServiceAccount object, **and**  are not referenced from pods. This label causes the authentication layer to reject use of the credentials. After being labeled as invalid, if the time specified by `--legacy-service-account-token-clean-up-period` (defaulting to one year) passes without the credential being used, the secret is automatically deleted. Secrets labeled as invalid which have not been auto-deleted yet can be re-activated by removing the `kubernetes.io/legacy-token-invalid-since` label. ([#120682](https://github.com/kubernetes/kubernetes/pull/120682), [@yt2985](https://github.com/yt2985)) [SIG Apps, Auth and Testing]
- Kube-scheduler implements scheduling hints for the NodeAffinity plugin.
  The scheduling hints allow the scheduler to only retry scheduling a Pod
  that was previously rejected by the NodeAffinity plugin if a new Node or a Node update matches the Pod's node affinity. ([#119155](https://github.com/kubernetes/kubernetes/pull/119155), [@carlory](https://github.com/carlory)) [SIG Scheduling]
- Kubeadm: Turn on FeatureGate `MergeCLIArgumentsWithConfig` to merge the config from flag and config file, otherwise, If the flag `--ignore-preflight-errors` is set from CLI, then the value from config file will be ignored. ([#119946](https://github.com/kubernetes/kubernetes/pull/119946), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubeadm: allow deploying a kubelet that is 3 versions older than the version of kubeadm (N-3). This aligns with the recent change made by SIG Architecture that extends the support skew between the control plane and kubelets. Tolerate this new kubelet skew for the commands "init", "join" and "upgrade". Note that if the kubeadm user applies a control plane version that is older than the kubeadm version (N-1 maximum) then the skew between the kubelet and control plane would become a maximum of N-2. ([#120825](https://github.com/kubernetes/kubernetes/pull/120825), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubelet allows pods to use the `net.ipv4.tcp_fin_timeout` , “net.ipv4.tcp_keepalive_intvl” and “net.ipv4.tcp_keepalive_probes“ sysctl by default; Pod Security admission allows this sysctl in v1.29+ versions of the baseline and restricted policies. ([#121240](https://github.com/kubernetes/kubernetes/pull/121240), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Auth and Node]
- Kubelet allows pods to use the `net.ipv4.tcp_keepalive_time` sysctl by default and the minimal kernel version is 4.5; Pod Security admission allows this sysctl in v1.29+ versions of the baseline and restricted policies. ([#118846](https://github.com/kubernetes/kubernetes/pull/118846), [@cyclinder](https://github.com/cyclinder)) [SIG Auth, Network and Node]
- Kubelet emits a metric for end-to-end pod startup latency including image pull. ([#121041](https://github.com/kubernetes/kubernetes/pull/121041), [@ruiwen-zhao](https://github.com/ruiwen-zhao)) [SIG Node]
- Kubernetes is now built with Go 1.21.3 ([#121149](https://github.com/kubernetes/kubernetes/pull/121149), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Make decoding etcd's response respect the timeout context. ([#121614](https://github.com/kubernetes/kubernetes/pull/121614), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery]
- Priority and Fairness feature is stable in 1.29, the feature gate will be removed in 1.31 ([#121638](https://github.com/kubernetes/kubernetes/pull/121638), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Testing]
- Promote PodHostIPs condition to beta. ([#120257](https://github.com/kubernetes/kubernetes/pull/120257), [@wzshiming](https://github.com/wzshiming)) [SIG Network, Node and Testing]
- Promote PodHostIPs condition to beta. ([#121477](https://github.com/kubernetes/kubernetes/pull/121477), [@wzshiming](https://github.com/wzshiming)) [SIG Network and Testing]
- Promote PodReplacementPolicy to beta. ([#121491](https://github.com/kubernetes/kubernetes/pull/121491), [@dejanzele](https://github.com/dejanzele)) [SIG Apps and Testing]
- Promotes plugin subcommand resolution feature to beta ([#120663](https://github.com/kubernetes/kubernetes/pull/120663), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Sidecar termination is now serialized and each sidecar container will receive a SIGTERM after all main containers and later starting sidecar containers have terminated. ([#120620](https://github.com/kubernetes/kubernetes/pull/120620), [@tzneal](https://github.com/tzneal)) [SIG Node and Testing]
- The CRD validation rule with feature gate `CustomResourceValidationExpressions` is promoted to GA. ([#121373](https://github.com/kubernetes/kubernetes/pull/121373), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]
- The KMSv2 feature with feature gates `KMSv2` and `KMSv2KDF` are promoted to GA.  The `KMSv1` feature gate is now disabled by default. ([#121485](https://github.com/kubernetes/kubernetes/pull/121485), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth and Testing]
- The SidecarContainers feature has graduated to beta and is enabled by default. ([#121579](https://github.com/kubernetes/kubernetes/pull/121579), [@gjkim42](https://github.com/gjkim42)) [SIG Node]
- Updated the generic apiserver library to produce an error if a new API server is configured with support for a data format other than JSON, YAML, or Protobuf. ([#121325](https://github.com/kubernetes/kubernetes/pull/121325), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- ValidatingAdmissionPolicy now preserves types of composition variables, and raise type-related errors early. ([#121001](https://github.com/kubernetes/kubernetes/pull/121001), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery and Testing]

### Documentation

- When the Kubelet fails to assign CPUs to a Pod because there less available CPUs than the Pod requests, the error message changed from
  "not enough cpus available to satisfy request" to "not enough cpus available to satisfy request: <num_requested> requested, only <num_available> available". ([#121059](https://github.com/kubernetes/kubernetes/pull/121059), [@matte21](https://github.com/matte21)) [SIG Node]

### Failing Test

- K8s.io/dynamic-resource-allocation: DRA drivers updating to this release are compatible with Kubernetes 1.27 and 1.28. ([#120868](https://github.com/kubernetes/kubernetes/pull/120868), [@pohly](https://github.com/pohly)) [SIG Node]

### Bug or Regression

- Add CAP_NET_RAW to netadmin debug profile and remove privileges when debugging nodes ([#118647](https://github.com/kubernetes/kubernetes/pull/118647), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI and Testing]
- Add a check: if a user attempts to create a static pod via the kubelet without specifying a name, they will get a visible validation error. ([#119522](https://github.com/kubernetes/kubernetes/pull/119522), [@YTGhost](https://github.com/YTGhost)) [SIG Node]
- Bugfix: OpenAPI spec no longer includes default of `{}` for certain fields where it did not make sense ([#120757](https://github.com/kubernetes/kubernetes/pull/120757), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node and Storage]
- Changed kubelet logs from error to info for uncached partitions when using CRI stats provider ([#100448](https://github.com/kubernetes/kubernetes/pull/100448), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Do not assign an empty value to the resource (CPU or memory) that not defined when stores the resources allocated to the pod in checkpoint ([#117615](https://github.com/kubernetes/kubernetes/pull/117615), [@aheng-ch](https://github.com/aheng-ch)) [SIG Node]
- Etcd: Update to v3.5.10 ([#121566](https://github.com/kubernetes/kubernetes/pull/121566), [@mzaian](https://github.com/mzaian)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Fix 121094 by re-introducing the readiness predicate for externalTrafficPolicy: Local services. ([#121116](https://github.com/kubernetes/kubernetes/pull/121116), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Cloud Provider and Network]
- Fix panic in Job controller when podRecreationPolicy: Failed is used, and the number of terminating pods exceeds parallelism. ([#121147](https://github.com/kubernetes/kubernetes/pull/121147), [@kannon92](https://github.com/kannon92)) [SIG Apps]
- Fix systemLogQuery service name matching ([#120678](https://github.com/kubernetes/kubernetes/pull/120678), [@rothgar](https://github.com/rothgar)) [SIG Node]
- Fixed a 1.28.0 regression where kube-controller-manager can crash when StatefulSet with Parallel policy and PVC labels is scaled up. ([#121142](https://github.com/kubernetes/kubernetes/pull/121142), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska)) [SIG Apps]
- Fixed a bug around restarting init containers in the right order relative to normal containers with SidecarContainers feature enabled. ([#120269](https://github.com/kubernetes/kubernetes/pull/120269), [@gjkim42](https://github.com/gjkim42)) [SIG Node and Testing]
- Fixed a bug where an API group's path was not unregistered from the API server's root paths when the group was deleted. ([#121283](https://github.com/kubernetes/kubernetes/pull/121283), [@tnqn](https://github.com/tnqn)) [SIG API Machinery and Testing]
- Fixed a bug where the CPU set allocated to an init container, with containerRestartPolicy of `Always`, were erroneously reused by a regular container. ([#119447](https://github.com/kubernetes/kubernetes/pull/119447), [@gjkim42](https://github.com/gjkim42)) [SIG Node and Testing]
- Fixed a bug where the device resources allocated to an init container, with containerRestartPolicy of `Always`, were erroneously reused by a regular container. ([#120461](https://github.com/kubernetes/kubernetes/pull/120461), [@gjkim42](https://github.com/gjkim42)) [SIG Node and Testing]
- Fixed a bug where the memory resources allocated to an init container, with containerRestartPolicy of `Always`, were erroneously reused by a regular container. ([#120715](https://github.com/kubernetes/kubernetes/pull/120715), [@gjkim42](https://github.com/gjkim42)) [SIG Node]
- Fixed a regression in default configurations, which enabled PodDisruptionConditions by default, 
  that prevented the control plane's pod garbage collector from deleting pods that contained duplicated field keys (env. variables with repeated keys or container ports). ([#121103](https://github.com/kubernetes/kubernetes/pull/121103), [@mimowo](https://github.com/mimowo)) [SIG Apps, Auth, Node, Scheduling and Testing]
- Fixed a regression in the Kubelet's behavior while creating a container when the `EventedPLEG` feature gate is enabled ([#120942](https://github.com/kubernetes/kubernetes/pull/120942), [@sairameshv](https://github.com/sairameshv)) [SIG Node]
- Fixed a regression since 1.27.0 in scheduler framework when running score plugins. 
  The `skippedScorePlugins` number might be greater than `enabledScorePlugins`, 
  so when initializing a slice the cap(len(skippedScorePlugins) - len(enabledScorePlugins)) is negative, 
  which is not allowed. ([#121632](https://github.com/kubernetes/kubernetes/pull/121632), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Fixed bug that kubelet resource metric `container_start_time_seconds` had timestamp equal to container start time. ([#120518](https://github.com/kubernetes/kubernetes/pull/120518), [@saschagrunert](https://github.com/saschagrunert)) [SIG Instrumentation, Node and Testing]
- Fixed inconsistency in the calculation of number of nodes that have an image, which affect the scoring in the ImageLocality plugin ([#116938](https://github.com/kubernetes/kubernetes/pull/116938), [@olderTaoist](https://github.com/olderTaoist)) [SIG Scheduling]
- Fixed some invalid and unimportant log calls. ([#121249](https://github.com/kubernetes/kubernetes/pull/121249), [@pohly](https://github.com/pohly)) [SIG Cloud Provider, Cluster Lifecycle and Testing]
- Fixed the bug that kubelet could't output logs after log file rotated when kubectl logs POD_NAME -f is running. ([#115702](https://github.com/kubernetes/kubernetes/pull/115702), [@xyz-li](https://github.com/xyz-li)) [SIG Node]
- Fixed the issue where pod with ordinal number lower than the rolling partitioning number was being deleted it was coming up with updated image. ([#120731](https://github.com/kubernetes/kubernetes/pull/120731), [@adilGhaffarDev](https://github.com/adilGhaffarDev)) [SIG Apps and Testing]
- Fixed tracking of terminating Pods in the Job status. The field was not updated unless there were other changes to apply ([#121342](https://github.com/kubernetes/kubernetes/pull/121342), [@dejanzele](https://github.com/dejanzele)) [SIG Apps and Testing]
- Fixes an issue where StatefulSet might not restart a pod after eviction or node failure. ([#121389](https://github.com/kubernetes/kubernetes/pull/121389), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska)) [SIG Apps and Testing]
- Fixes calculating the requeue time in the cronjob controller, which results in properly handling failed/stuck jobs ([#121327](https://github.com/kubernetes/kubernetes/pull/121327), [@soltysh](https://github.com/soltysh)) [SIG Apps]
- Forbid sysctls for pod sharing the respective namespaces with the host when creating and update pod without such sysctls ([#118705](https://github.com/kubernetes/kubernetes/pull/118705), [@pacoxu](https://github.com/pacoxu)) [SIG Apps and Node]
- K8s.io/dynamic-resource-allocation/controller: ResourceClaimParameters and ResourceClassParameters validation errors were not visible on ResourceClaim, ResourceClass and Pod. ([#121065](https://github.com/kubernetes/kubernetes/pull/121065), [@byako](https://github.com/byako)) [SIG Node]
- Kube-proxy now reports its health more accurately in dual-stack clusters when there are problems with only one IP family. ([#118146](https://github.com/kubernetes/kubernetes/pull/118146), [@aroradaman](https://github.com/aroradaman)) [SIG Network and Windows]
- Metric buckets for pod_start_duration_seconds are changed to {0.5, 1, 2, 3, 4, 5, 6, 8, 10, 20, 30, 45, 60, 120, 180, 240, 300, 360, 480, 600, 900, 1200, 1800, 2700, 3600} ([#120680](https://github.com/kubernetes/kubernetes/pull/120680), [@ruiwen-zhao](https://github.com/ruiwen-zhao)) [SIG Instrumentation and Node]
- Mitigates http/2 DOS vulnerabilities for CVE-2023-44487 and CVE-2023-39325 for the API server when the client is unauthenticated. The mitigation may be disabled by setting the `UnauthenticatedHTTP2DOSMitigation` feature gate to `false` (it is enabled by default). An API server fronted by an L7 load balancer that already mitigates these http/2 attacks may choose to disable the kube-apiserver mitigation to avoid disrupting load balancer → kube-apiserver connections if http/2 requests from multiple clients share the same backend connection. An API server on a private network may opt to disable the kube-apiserver mitigation to prevent performance regressions for unauthenticated clients. Authenticated requests rely on the fix in golang.org/x/net v0.17.0 alone. https://issue.k8s.io/121197 tracks further mitigation of http/2 attacks by authenticated clients. ([#121120](https://github.com/kubernetes/kubernetes/pull/121120), [@enj](https://github.com/enj)) [SIG API Machinery]
- Registered metric `apiserver_request_body_size_bytes` to track the size distribution of requests by `resource` and `verb`. ([#120474](https://github.com/kubernetes/kubernetes/pull/120474), [@YaoC](https://github.com/YaoC)) [SIG API Machinery and Instrumentation]
- Update the CRI-O socket path, so users who configure kubelet to use a location like `/run/crio/crio.sock` don't see strange behaviour from CRI stats provider. ([#118704](https://github.com/kubernetes/kubernetes/pull/118704), [@dgl](https://github.com/dgl)) [SIG Node]
- Wait.PollUntilContextTimeout function, if immediate is true, the condition will be invoked before waiting and guarantees that the condition is invoked at least once and then wait a interval before executing again. ([#119762](https://github.com/kubernetes/kubernetes/pull/119762), [@AxeZhan](https://github.com/AxeZhan)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Allow using lower and upper case feature flag value, the name has to match still ([#121441](https://github.com/kubernetes/kubernetes/pull/121441), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- E2E storage tests: setting test tags like `[Slow]` via the DriverInfo.FeatureTag field is no longer supported. ([#121391](https://github.com/kubernetes/kubernetes/pull/121391), [@pohly](https://github.com/pohly)) [SIG Storage and Testing]
- EnqueueExtensions from plugins other than PreEnqueue, PreFilter, Filter, Reserve and Permit are ignored.
  It reduces the number of kinds of cluster events the scheduler needs to subscribe/handle. ([#121571](https://github.com/kubernetes/kubernetes/pull/121571), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- GetPodQOS(pod *core.Pod) function now returns the stored value from PodStatus.QOSClass, if set. To compute/evaluate the value of QOSClass from scratch, ComputePodQOS(pod *core.Pod) must be used. ([#119665](https://github.com/kubernetes/kubernetes/pull/119665), [@vinaykul](https://github.com/vinaykul)) [SIG API Machinery, Apps, CLI, Node, Scheduling and Testing]
- Graduate JobReadyPods to stable. The feature gate can no longer be disabled. ([#121302](https://github.com/kubernetes/kubernetes/pull/121302), [@stuton](https://github.com/stuton)) [SIG Apps and Testing]
- Kube-controller-manager's help will include controllers behind a feature gate in `--controllers` flag ([#120371](https://github.com/kubernetes/kubernetes/pull/120371), [@atiratree](https://github.com/atiratree)) [SIG API Machinery]
- Kubeadm: remove leftover ALPHA disclaimer that can be seen in the "kubeadm init phase certs" command help screen. The "certs" phase of "init" is not ALPHA. ([#121172](https://github.com/kubernetes/kubernetes/pull/121172), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Migrated the remainder of the scheduler to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#120933](https://github.com/kubernetes/kubernetes/pull/120933), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation, Scheduling and Testing]
- Previous versions of Kubernetes on Google Cloud required that workloads (e.g. Deployments, DaemonSets, etc.) which used PersistentDisk volumes were using them in read-only mode.  This validation provided very little value at relatively host implementation cost, and will no longer be validated.  If this is a problem for a specific use-case, please set the `SkipReadOnlyValidationGCE` gate to false to re-enable the validation, and file a kubernetes bug with details. ([#121083](https://github.com/kubernetes/kubernetes/pull/121083), [@thockin](https://github.com/thockin)) [SIG Apps]
- Remove GA featuregate about CSIMigrationvSphere in 1.29 ([#121291](https://github.com/kubernetes/kubernetes/pull/121291), [@bzsuni](https://github.com/bzsuni)) [SIG API Machinery, Node and Storage]
- Remove GA featuregate about ProbeTerminationGracePeriod in 1.29 ([#121257](https://github.com/kubernetes/kubernetes/pull/121257), [@bzsuni](https://github.com/bzsuni)) [SIG Node and Testing]
- Remove GA featuregate for JobTrackingWithFinalizers in 1.28 ([#119100](https://github.com/kubernetes/kubernetes/pull/119100), [@bzsuni](https://github.com/bzsuni)) [SIG Apps]
- Remove GAed feature gates OpenAPIV3 ([#121255](https://github.com/kubernetes/kubernetes/pull/121255), [@tukwila](https://github.com/tukwila)) [SIG API Machinery and Testing]
- Remove GAed feature gates SeccompDefault ([#121246](https://github.com/kubernetes/kubernetes/pull/121246), [@tukwila](https://github.com/tukwila)) [SIG Node]
- Remove GAed feature gates TopologyManager ([#121252](https://github.com/kubernetes/kubernetes/pull/121252), [@tukwila](https://github.com/tukwila)) [SIG Node]
- Removed the `CronJobTimeZone` feature gate (the feature is stable and always enabled)
  - Removed the `JobMutableNodeSchedulingDirectives` feature gate (the feature is stable and always enabled)
  - Removed the `LegacyServiceAccountTokenNoAutoGeneration` feature gate (the feature is stable and always enabled) ([#120192](https://github.com/kubernetes/kubernetes/pull/120192), [@SataQiu](https://github.com/SataQiu)) [SIG Apps, Auth and Scheduling]
- Removed the `DownwardAPIHugePages` feature gate (the feature is stable and always enabled) ([#120249](https://github.com/kubernetes/kubernetes/pull/120249), [@pacoxu](https://github.com/pacoxu)) [SIG Apps and Node]
- Removed the `GRPCContainerProbe` feature gate (the feature is stable and always enabled) ([#120248](https://github.com/kubernetes/kubernetes/pull/120248), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery, CLI and Node]
- Rename apiserver_request_body_sizes metric to apiserver_request_body_size_bytes ([#120503](https://github.com/kubernetes/kubernetes/pull/120503), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery]
- RetroactiveDefaultStorageClass feature gate that graduated to GA in 1.28 and was unconditionally enabled has been removed in v1.29. ([#120861](https://github.com/kubernetes/kubernetes/pull/120861), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]

## Dependencies

### Added
- cloud.google.com/go/dataproc/v2: v2.0.1
- github.com/danwinship/knftables: [v0.0.13](https://github.com/danwinship/knftables/tree/v0.0.13)
- github.com/google/s2a-go: [v0.1.7](https://github.com/google/s2a-go/tree/v0.1.7)
- google.golang.org/genproto/googleapis/bytestream: e85fd2c

### Changed
- cloud.google.com/go/accessapproval: v1.6.0 → v1.7.1
- cloud.google.com/go/accesscontextmanager: v1.7.0 → v1.8.1
- cloud.google.com/go/aiplatform: v1.37.0 → v1.48.0
- cloud.google.com/go/analytics: v0.19.0 → v0.21.3
- cloud.google.com/go/apigateway: v1.5.0 → v1.6.1
- cloud.google.com/go/apigeeconnect: v1.5.0 → v1.6.1
- cloud.google.com/go/apigeeregistry: v0.6.0 → v0.7.1
- cloud.google.com/go/appengine: v1.7.1 → v1.8.1
- cloud.google.com/go/area120: v0.7.1 → v0.8.1
- cloud.google.com/go/artifactregistry: v1.13.0 → v1.14.1
- cloud.google.com/go/asset: v1.13.0 → v1.14.1
- cloud.google.com/go/assuredworkloads: v1.10.0 → v1.11.1
- cloud.google.com/go/automl: v1.12.0 → v1.13.1
- cloud.google.com/go/baremetalsolution: v0.5.0 → v1.1.1
- cloud.google.com/go/batch: v0.7.0 → v1.3.1
- cloud.google.com/go/beyondcorp: v0.5.0 → v1.0.0
- cloud.google.com/go/bigquery: v1.50.0 → v1.53.0
- cloud.google.com/go/billing: v1.13.0 → v1.16.0
- cloud.google.com/go/binaryauthorization: v1.5.0 → v1.6.1
- cloud.google.com/go/certificatemanager: v1.6.0 → v1.7.1
- cloud.google.com/go/channel: v1.12.0 → v1.16.0
- cloud.google.com/go/cloudbuild: v1.9.0 → v1.13.0
- cloud.google.com/go/clouddms: v1.5.0 → v1.6.1
- cloud.google.com/go/cloudtasks: v1.10.0 → v1.12.1
- cloud.google.com/go/compute: v1.19.0 → v1.23.0
- cloud.google.com/go/contactcenterinsights: v1.6.0 → v1.10.0
- cloud.google.com/go/container: v1.15.0 → v1.24.0
- cloud.google.com/go/containeranalysis: v0.9.0 → v0.10.1
- cloud.google.com/go/datacatalog: v1.13.0 → v1.16.0
- cloud.google.com/go/dataflow: v0.8.0 → v0.9.1
- cloud.google.com/go/dataform: v0.7.0 → v0.8.1
- cloud.google.com/go/datafusion: v1.6.0 → v1.7.1
- cloud.google.com/go/datalabeling: v0.7.0 → v0.8.1
- cloud.google.com/go/dataplex: v1.6.0 → v1.9.0
- cloud.google.com/go/dataqna: v0.7.0 → v0.8.1
- cloud.google.com/go/datastore: v1.11.0 → v1.13.0
- cloud.google.com/go/datastream: v1.7.0 → v1.10.0
- cloud.google.com/go/deploy: v1.8.0 → v1.13.0
- cloud.google.com/go/dialogflow: v1.32.0 → v1.40.0
- cloud.google.com/go/dlp: v1.9.0 → v1.10.1
- cloud.google.com/go/documentai: v1.18.0 → v1.22.0
- cloud.google.com/go/domains: v0.8.0 → v0.9.1
- cloud.google.com/go/edgecontainer: v1.0.0 → v1.1.1
- cloud.google.com/go/essentialcontacts: v1.5.0 → v1.6.2
- cloud.google.com/go/eventarc: v1.11.0 → v1.13.0
- cloud.google.com/go/filestore: v1.6.0 → v1.7.1
- cloud.google.com/go/firestore: v1.9.0 → v1.11.0
- cloud.google.com/go/functions: v1.13.0 → v1.15.1
- cloud.google.com/go/gkebackup: v0.4.0 → v1.3.0
- cloud.google.com/go/gkeconnect: v0.7.0 → v0.8.1
- cloud.google.com/go/gkehub: v0.12.0 → v0.14.1
- cloud.google.com/go/gkemulticloud: v0.5.0 → v1.0.0
- cloud.google.com/go/gsuiteaddons: v1.5.0 → v1.6.1
- cloud.google.com/go/iam: v0.13.0 → v1.1.1
- cloud.google.com/go/iap: v1.7.1 → v1.8.1
- cloud.google.com/go/ids: v1.3.0 → v1.4.1
- cloud.google.com/go/iot: v1.6.0 → v1.7.1
- cloud.google.com/go/kms: v1.10.1 → v1.15.0
- cloud.google.com/go/language: v1.9.0 → v1.10.1
- cloud.google.com/go/lifesciences: v0.8.0 → v0.9.1
- cloud.google.com/go/longrunning: v0.4.1 → v0.5.1
- cloud.google.com/go/managedidentities: v1.5.0 → v1.6.1
- cloud.google.com/go/maps: v0.7.0 → v1.4.0
- cloud.google.com/go/mediatranslation: v0.7.0 → v0.8.1
- cloud.google.com/go/memcache: v1.9.0 → v1.10.1
- cloud.google.com/go/metastore: v1.10.0 → v1.12.0
- cloud.google.com/go/monitoring: v1.13.0 → v1.15.1
- cloud.google.com/go/networkconnectivity: v1.11.0 → v1.12.1
- cloud.google.com/go/networkmanagement: v1.6.0 → v1.8.0
- cloud.google.com/go/networksecurity: v0.8.0 → v0.9.1
- cloud.google.com/go/notebooks: v1.8.0 → v1.9.1
- cloud.google.com/go/optimization: v1.3.1 → v1.4.1
- cloud.google.com/go/orchestration: v1.6.0 → v1.8.1
- cloud.google.com/go/orgpolicy: v1.10.0 → v1.11.1
- cloud.google.com/go/osconfig: v1.11.0 → v1.12.1
- cloud.google.com/go/oslogin: v1.9.0 → v1.10.1
- cloud.google.com/go/phishingprotection: v0.7.0 → v0.8.1
- cloud.google.com/go/policytroubleshooter: v1.6.0 → v1.8.0
- cloud.google.com/go/privatecatalog: v0.8.0 → v0.9.1
- cloud.google.com/go/pubsub: v1.30.0 → v1.33.0
- cloud.google.com/go/pubsublite: v1.7.0 → v1.8.1
- cloud.google.com/go/recaptchaenterprise/v2: v2.7.0 → v2.7.2
- cloud.google.com/go/recommendationengine: v0.7.0 → v0.8.1
- cloud.google.com/go/recommender: v1.9.0 → v1.10.1
- cloud.google.com/go/redis: v1.11.0 → v1.13.1
- cloud.google.com/go/resourcemanager: v1.7.0 → v1.9.1
- cloud.google.com/go/resourcesettings: v1.5.0 → v1.6.1
- cloud.google.com/go/retail: v1.12.0 → v1.14.1
- cloud.google.com/go/run: v0.9.0 → v1.2.0
- cloud.google.com/go/scheduler: v1.9.0 → v1.10.1
- cloud.google.com/go/secretmanager: v1.10.0 → v1.11.1
- cloud.google.com/go/security: v1.13.0 → v1.15.1
- cloud.google.com/go/securitycenter: v1.19.0 → v1.23.0
- cloud.google.com/go/servicedirectory: v1.9.0 → v1.11.0
- cloud.google.com/go/shell: v1.6.0 → v1.7.1
- cloud.google.com/go/spanner: v1.45.0 → v1.47.0
- cloud.google.com/go/speech: v1.15.0 → v1.19.0
- cloud.google.com/go/storagetransfer: v1.8.0 → v1.10.0
- cloud.google.com/go/talent: v1.5.0 → v1.6.2
- cloud.google.com/go/texttospeech: v1.6.0 → v1.7.1
- cloud.google.com/go/tpu: v1.5.0 → v1.6.1
- cloud.google.com/go/trace: v1.9.0 → v1.10.1
- cloud.google.com/go/translate: v1.7.0 → v1.8.2
- cloud.google.com/go/video: v1.15.0 → v1.19.0
- cloud.google.com/go/videointelligence: v1.10.0 → v1.11.1
- cloud.google.com/go/vision/v2: v2.7.0 → v2.7.2
- cloud.google.com/go/vmmigration: v1.6.0 → v1.7.1
- cloud.google.com/go/vmwareengine: v0.3.0 → v1.0.0
- cloud.google.com/go/vpcaccess: v1.6.0 → v1.7.1
- cloud.google.com/go/webrisk: v1.8.0 → v1.9.1
- cloud.google.com/go/websecurityscanner: v1.5.0 → v1.6.1
- cloud.google.com/go/workflows: v1.10.0 → v1.11.1
- cloud.google.com/go: v0.110.0 → v0.110.6
- github.com/alecthomas/template: [fb15b89 → a0175ee](https://github.com/alecthomas/template/compare/fb15b89...a0175ee)
- github.com/cncf/xds/go: [06c439d → e9ce688](https://github.com/cncf/xds/go/compare/06c439d...e9ce688)
- github.com/cyphar/filepath-securejoin: [v0.2.3 → v0.2.4](https://github.com/cyphar/filepath-securejoin/compare/v0.2.3...v0.2.4)
- github.com/docker/distribution: [v2.8.1+incompatible → v2.8.2+incompatible](https://github.com/docker/distribution/compare/v2.8.1...v2.8.2)
- github.com/docker/docker: [v20.10.21+incompatible → v20.10.24+incompatible](https://github.com/docker/docker/compare/v20.10.21...v20.10.24)
- github.com/envoyproxy/go-control-plane: [v0.10.3 → v0.11.1](https://github.com/envoyproxy/go-control-plane/compare/v0.10.3...v0.11.1)
- github.com/envoyproxy/protoc-gen-validate: [v0.9.1 → v1.0.2](https://github.com/envoyproxy/protoc-gen-validate/compare/v0.9.1...v1.0.2)
- github.com/fsnotify/fsnotify: [v1.6.0 → v1.7.0](https://github.com/fsnotify/fsnotify/compare/v1.6.0...v1.7.0)
- github.com/go-logr/logr: [v1.2.4 → v1.3.0](https://github.com/go-logr/logr/compare/v1.2.4...v1.3.0)
- github.com/godbus/dbus/v5: [v5.0.6 → v5.1.0](https://github.com/godbus/dbus/v5/compare/v5.0.6...v5.1.0)
- github.com/golang/glog: [v1.0.0 → v1.1.0](https://github.com/golang/glog/compare/v1.0.0...v1.1.0)
- github.com/google/cadvisor: [v0.47.3 → v0.48.1](https://github.com/google/cadvisor/compare/v0.47.3...v0.48.1)
- github.com/google/cel-go: [v0.17.6 → v0.17.7](https://github.com/google/cel-go/compare/v0.17.6...v0.17.7)
- github.com/google/go-cmp: [v0.5.9 → v0.6.0](https://github.com/google/go-cmp/compare/v0.5.9...v0.6.0)
- github.com/googleapis/gax-go/v2: [v2.7.1 → v2.11.0](https://github.com/googleapis/gax-go/v2/compare/v2.7.1...v2.11.0)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.7.0 → v2.16.0](https://github.com/grpc-ecosystem/grpc-gateway/v2/compare/v2.7.0...v2.16.0)
- github.com/ishidawataru/sctp: [7c296d4 → 7ff4192](https://github.com/ishidawataru/sctp/compare/7c296d4...7ff4192)
- github.com/konsorten/go-windows-terminal-sequences: [v1.0.3 → v1.0.1](https://github.com/konsorten/go-windows-terminal-sequences/compare/v1.0.3...v1.0.1)
- github.com/onsi/gomega: [v1.28.0 → v1.29.0](https://github.com/onsi/gomega/compare/v1.28.0...v1.29.0)
- github.com/spf13/afero: [v1.2.2 → v1.1.2](https://github.com/spf13/afero/compare/v1.2.2...v1.1.2)
- github.com/stretchr/testify: [v1.8.2 → v1.8.4](https://github.com/stretchr/testify/compare/v1.8.2...v1.8.4)
- go.etcd.io/bbolt: v1.3.7 → v1.3.8
- go.etcd.io/etcd/api/v3: v3.5.9 → v3.5.10
- go.etcd.io/etcd/client/pkg/v3: v3.5.9 → v3.5.10
- go.etcd.io/etcd/client/v2: v2.305.9 → v2.305.10
- go.etcd.io/etcd/client/v3: v3.5.9 → v3.5.10
- go.etcd.io/etcd/pkg/v3: v3.5.9 → v3.5.10
- go.etcd.io/etcd/raft/v3: v3.5.9 → v3.5.10
- go.etcd.io/etcd/server/v3: v3.5.9 → v3.5.10
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.35.0 → v0.42.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.35.0 → v0.42.0
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.35.1 → v0.44.0
- go.opentelemetry.io/contrib/propagators/b3: v1.10.0 → v1.17.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.10.0 → v1.19.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.10.0 → v1.19.0
- go.opentelemetry.io/otel/metric: v0.31.0 → v1.19.0
- go.opentelemetry.io/otel/sdk: v1.10.0 → v1.19.0
- go.opentelemetry.io/otel/trace: v1.10.0 → v1.19.0
- go.opentelemetry.io/otel: v1.10.0 → v1.19.0
- go.opentelemetry.io/proto/otlp: v0.19.0 → v1.0.0
- golang.org/x/crypto: v0.12.0 → v0.14.0
- golang.org/x/net: v0.14.0 → v0.17.0
- golang.org/x/oauth2: v0.8.0 → v0.10.0
- golang.org/x/sys: v0.12.0 → v0.13.0
- golang.org/x/term: v0.11.0 → v0.13.0
- golang.org/x/text: v0.12.0 → v0.13.0
- google.golang.org/api: v0.114.0 → v0.126.0
- google.golang.org/genproto/googleapis/api: dd9d682 → 23370e0
- google.golang.org/genproto/googleapis/rpc: 28d5490 → b8732ec
- google.golang.org/genproto: 0005af6 → f966b18
- google.golang.org/grpc: v1.54.0 → v1.58.3
- k8s.io/klog/v2: v2.100.1 → v2.110.1
- k8s.io/kube-openapi: d090da1 → 2dd684a
- sigs.k8s.io/structured-merge-diff/v4: v4.3.0 → v4.4.1

### Removed
- cloud.google.com/go/dataproc: v1.12.0
- cloud.google.com/go/gaming: v1.9.0
- github.com/blang/semver: [v3.5.1+incompatible](https://github.com/blang/semver/tree/v3.5.1)
- github.com/jmespath/go-jmespath/internal/testify: [v1.5.1](https://github.com/jmespath/go-jmespath/internal/testify/tree/v1.5.1)
- go.opentelemetry.io/otel/exporters/otlp/internal/retry: v1.10.0



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