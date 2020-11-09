<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.19.3](#v1193)
  - [Downloads for v1.19.3](#downloads-for-v1193)
    - [Source Code](#source-code)
    - [Client binaries](#client-binaries)
    - [Server binaries](#server-binaries)
    - [Node binaries](#node-binaries)
  - [Changelog since v1.19.2](#changelog-since-v1192)
  - [Changes by Kind](#changes-by-kind)
    - [Feature](#feature)
    - [Design](#design)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.19.2](#v1192)
  - [Downloads for v1.19.2](#downloads-for-v1192)
    - [Source Code](#source-code-1)
    - [Client binaries](#client-binaries-1)
    - [Server binaries](#server-binaries-1)
    - [Node binaries](#node-binaries-1)
  - [Changelog since v1.19.1](#changelog-since-v1191)
  - [Changes by Kind](#changes-by-kind-1)
    - [API Change](#api-change)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.19.1](#v1191)
  - [Downloads for v1.19.1](#downloads-for-v1191)
    - [Source Code](#source-code-2)
    - [Client binaries](#client-binaries-2)
    - [Server binaries](#server-binaries-2)
    - [Node binaries](#node-binaries-2)
  - [Changelog since v1.19.0](#changelog-since-v1190)
  - [Changes by Kind](#changes-by-kind-2)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.19.0](#v1190)
  - [Downloads for v1.19.0](#downloads-for-v1190)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.18.0](#changelog-since-v1180)
  - [What’s New (Major Themes)](#what’s-new-major-themes)
    - [Deprecation warnings](#deprecation-warnings)
    - [Avoiding permanent beta](#avoiding-permanent-beta)
    - [Expanded CLI support for debugging workloads and nodes](#expanded-cli-support-for-debugging-workloads-and-nodes)
    - [Structured logging](#structured-logging)
    - [EndpointSlices are now enabled by default](#endpointslices-are-now-enabled-by-default)
    - [Ingress graduates to General Availability](#ingress-graduates-to-general-availability)
    - [seccomp graduates to General Availability](#seccomp-graduates-to-general-availability)
    - [Production images moved to community control](#production-images-moved-to-community-control)
    - [KubeSchedulerConfiguration graduates to Beta](#kubeschedulerconfiguration-graduates-to-beta)
    - [CSI Migration - AzureDisk and vSphere (beta)](#csi-migration---azuredisk-and-vsphere-beta)
    - [Storage capacity tracking](#storage-capacity-tracking)
    - [CSI Volume health monitoring](#csi-volume-health-monitoring)
    - [General ephemeral volumes](#general-ephemeral-volumes)
    - [Immutable Secrets and ConfigMaps (beta)](#immutable-secrets-and-configmaps-beta)
    - [CSI Proxy for Windows](#csi-proxy-for-windows)
    - [Dashboard v2](#dashboard-v2)
    - [Windows containerd support graduates to beta](#windows-containerd-support-graduates-to-beta)
    - [Increase the Kubernetes support window to one year](#increase-the-kubernetes-support-window-to-one-year)
  - [Known Issues](#known-issues)
  - [Urgent Upgrade Notes ](#urgent-upgrade-notes-)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)
- [v1.19.0-rc.4](#v1190-rc4)
  - [Downloads for v1.19.0-rc.4](#downloads-for-v1190-rc4)
    - [Source Code](#source-code-3)
    - [Client binaries](#client-binaries-4)
    - [Server binaries](#server-binaries-4)
    - [Node binaries](#node-binaries-4)
  - [Changelog since v1.19.0-rc.3](#changelog-since-v1190-rc3)
  - [Changes by Kind](#changes-by-kind-4)
    - [Deprecation](#deprecation-1)
    - [Bug or Regression](#bug-or-regression-4)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-5)
    - [Added](#added-5)
    - [Changed](#changed-5)
    - [Removed](#removed-5)
- [v1.19.0-rc.3](#v1190-rc3)
  - [Downloads for v1.19.0-rc.3](#downloads-for-v1190-rc3)
    - [Source Code](#source-code-4)
    - [Client binaries](#client-binaries-5)
    - [Server binaries](#server-binaries-5)
    - [Node binaries](#node-binaries-5)
  - [Changelog since v1.19.0-rc.2](#changelog-since-v1190-rc2)
  - [Changes by Kind](#changes-by-kind-5)
    - [API Change](#api-change-2)
    - [Bug or Regression](#bug-or-regression-5)
  - [Dependencies](#dependencies-6)
    - [Added](#added-6)
    - [Changed](#changed-6)
    - [Removed](#removed-6)
- [v1.19.0-rc.2](#v1190-rc2)
  - [Downloads for v1.19.0-rc.2](#downloads-for-v1190-rc2)
    - [Source Code](#source-code-5)
    - [Client binaries](#client-binaries-6)
    - [Server binaries](#server-binaries-6)
    - [Node binaries](#node-binaries-6)
  - [Changelog since v1.19.0-rc.1](#changelog-since-v1190-rc1)
  - [Changes by Kind](#changes-by-kind-6)
    - [API Change](#api-change-3)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-6)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-5)
  - [Dependencies](#dependencies-7)
    - [Added](#added-7)
    - [Changed](#changed-7)
    - [Removed](#removed-7)
- [v1.19.0-rc.1](#v1190-rc1)
  - [Downloads for v1.19.0-rc.1](#downloads-for-v1190-rc1)
    - [Source Code](#source-code-6)
    - [Client binaries](#client-binaries-7)
    - [Server binaries](#server-binaries-7)
    - [Node binaries](#node-binaries-7)
  - [Changelog since v1.19.0-rc.0](#changelog-since-v1190-rc0)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-7)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-4)
    - [Feature](#feature-3)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-7)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-6)
  - [Dependencies](#dependencies-8)
    - [Added](#added-8)
    - [Changed](#changed-8)
    - [Removed](#removed-8)
- [v1.19.0-beta.2](#v1190-beta2)
  - [Downloads for v1.19.0-beta.2](#downloads-for-v1190-beta2)
    - [Source Code](#source-code-7)
    - [Client binaries](#client-binaries-8)
    - [Server binaries](#server-binaries-8)
    - [Node binaries](#node-binaries-8)
  - [Changelog since v1.19.0-beta.1](#changelog-since-v1190-beta1)
  - [Changes by Kind](#changes-by-kind-8)
    - [Deprecation](#deprecation-3)
    - [API Change](#api-change-5)
    - [Feature](#feature-4)
    - [Bug or Regression](#bug-or-regression-8)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-7)
  - [Dependencies](#dependencies-9)
    - [Added](#added-9)
    - [Changed](#changed-9)
    - [Removed](#removed-9)
- [v1.19.0-beta.1](#v1190-beta1)
  - [Downloads for v1.19.0-beta.1](#downloads-for-v1190-beta1)
    - [Source Code](#source-code-8)
    - [Client binaries](#client-binaries-9)
    - [Server binaries](#server-binaries-9)
    - [Node binaries](#node-binaries-9)
  - [Changelog since v1.19.0-alpha.3](#changelog-since-v1190-alpha3)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-9)
    - [API Change](#api-change-6)
    - [Feature](#feature-5)
    - [Bug or Regression](#bug-or-regression-9)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-8)
  - [Dependencies](#dependencies-10)
    - [Added](#added-10)
    - [Changed](#changed-10)
    - [Removed](#removed-10)
- [v1.19.0-beta.0](#v1190-beta0)
  - [Downloads for v1.19.0-beta.0](#downloads-for-v1190-beta0)
    - [Source Code](#source-code-9)
    - [Client binaries](#client-binaries-10)
    - [Server binaries](#server-binaries-10)
    - [Node binaries](#node-binaries-10)
  - [Changelog since v1.19.0-alpha.3](#changelog-since-v1190-alpha3-1)
  - [Changes by Kind](#changes-by-kind-10)
    - [API Change](#api-change-7)
    - [Feature](#feature-6)
    - [Bug or Regression](#bug-or-regression-10)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-9)
  - [Dependencies](#dependencies-11)
    - [Added](#added-11)
    - [Changed](#changed-11)
    - [Removed](#removed-11)
- [v1.19.0-alpha.3](#v1190-alpha3)
  - [Downloads for v1.19.0-alpha.3](#downloads-for-v1190-alpha3)
    - [Client Binaries](#client-binaries-11)
    - [Server Binaries](#server-binaries-11)
    - [Node Binaries](#node-binaries-11)
  - [Changelog since v1.19.0-alpha.2](#changelog-since-v1190-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-3)
  - [Changes by Kind](#changes-by-kind-11)
    - [Deprecation](#deprecation-4)
    - [API Change](#api-change-8)
    - [Feature](#feature-7)
    - [Bug or Regression](#bug-or-regression-11)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-10)
- [v1.19.0-alpha.2](#v1190-alpha2)
  - [Downloads for v1.19.0-alpha.2](#downloads-for-v1190-alpha2)
    - [Client Binaries](#client-binaries-12)
    - [Server Binaries](#server-binaries-12)
    - [Node Binaries](#node-binaries-12)
  - [Changelog since v1.19.0-alpha.1](#changelog-since-v1190-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-3)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-4)
  - [Changes by Kind](#changes-by-kind-12)
    - [API Change](#api-change-9)
    - [Feature](#feature-8)
    - [Bug or Regression](#bug-or-regression-12)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-11)
- [v1.19.0-alpha.1](#v1190-alpha1)
  - [Downloads for v1.19.0-alpha.1](#downloads-for-v1190-alpha1)
    - [Client Binaries](#client-binaries-13)
    - [Server Binaries](#server-binaries-13)
    - [Node Binaries](#node-binaries-13)
  - [Changelog since v1.19.0-alpha.0](#changelog-since-v1190-alpha0)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-4)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-5)
  - [Changes by Kind](#changes-by-kind-13)
    - [Deprecation](#deprecation-5)
    - [API Change](#api-change-10)

<!-- END MUNGE: GENERATED_TOC -->

# v1.19.3


## Downloads for v1.19.3

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes.tar.gz) | a9f627b2d35b0aa543863986668de6df316e0649df7fc9b5514c52d5c2bfdbc5a7c416784067c40aeba96a25508a1916439cb76436a7c0dc107c95077eca6cd3
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-src.tar.gz) | ca92ec0c578ac80cfa58ef169403a08ea0daee14525c79776ba8e3ae349a1fc3d530286d30da7d7d7065916a3aa51f9e89366f2ba6941ca6da539f475120c4b8

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-darwin-amd64.tar.gz) | 5415f82dcdd3c0f586235187dbea696fadb66fbf1f5d77043fdf150add8a10f4ae2e3bfe5bc3996e112ee3cfeb1421c03f681d0db812bcd2caf5abb1fb9a6e8c
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-linux-386.tar.gz) | bb24b7a22fda9f372cd342bf79b4d01b1d51d3fc9961a5e0653166e558c7081f71e92f9c5a997c091ca9bff5b11f3cb12989e0393351040902aa403e6cbafc8e
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-linux-amd64.tar.gz) | d9a6b28cddb673e1ad9e5e8befb98f1ff8ab25778c2aa4c7c377ade84c07fa484aa35b43a32b802e9e9cd5945b3219a2b28a87e02717a5dcb39acadb4ea52ae3
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-linux-arm.tar.gz) | 8f3c3198ebe4af74057342dcbebbba2dd6b5797b27b0fed8143d82cbb9ffff5e83a6ce72e9e695f5a67bd8959a1cb552795548ed7be707d23e7e8bce23b722a6
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-linux-arm64.tar.gz) | d850d1a95ff5de0ee6be6717ead2a376069c7ab83720400a2cdbba77647c14706ff5ff927ea656d4fc82ac2d01a8b3d0ff0da1c391fcf2cf7d876cfd45136d18
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-linux-ppc64le.tar.gz) | d1ecef91d6ff547bf8fa1d1e2fe56d333c630d45eb94a31850717de3ed1a69c5d8241e5e2ae251bc841b4c76759c6976619175f2d2e82b57047554aaa5e17f24
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-linux-s390x.tar.gz) | aaa38f257e6e720bc509673b1ada34b1712941b5874110a9245943e613d9a10c7a2caa770d8c886965910232c116cc1a91bdf7c35d4c49027d6716ea46ae7d2b
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-windows-386.tar.gz) | a3aa1720ff3145d228066b549f2d74a7264092e4396932b396f2bb8dfd80361334bde9467f6b179c97f2dbe53fc7f2ca93b97b595e390b0b60c3a99555f8818d
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-client-windows-amd64.tar.gz) | 3123ceb0f3c317e02dfb19a9260ee4954d24574fb44098a4bbdb943b0d9e4bf513534babcb3b68d352ebf3f635193b79592617f7e0fcf1ceeef6fede8f2d71eb

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-server-linux-amd64.tar.gz) | ebe86d27275a3ed1208b6db99a65cc9cf24b60fd3184b9f0fb769bc4b1b162dfd8330333fbe4a18df765a39211595101d1bb3f8671b411cb7a58a6cb8ced58b2
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-server-linux-arm.tar.gz) | 15b106056e5e71f2d67d7e8fb2d510f275e98a11f578fe820bba212674d09d3e5c9ffabfb335ad4b9487484221fe9a95ea9eb981adb1586c8ee013f5d4e3a882
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-server-linux-arm64.tar.gz) | c4b3f6f57065f741ed99f85880448a08fd98687f07fbd129bad22ae1eab8966a685102226c65c7ecf6f9d5d9c7c7802a69c0b9587c9a2bf3043acccfdb894ad6
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-server-linux-ppc64le.tar.gz) | 022aa2956638ad28d552ff365523ce9a6b4d9530236135b66b44af5d42943df466249426b79c9597373888859c0ccf69253ef9d25d20ba760e1041bb74e37e73
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-server-linux-s390x.tar.gz) | 9bfee4f0a4c85482b60a71fb486c4e367ca558db1a84dd0607e2f1912f1b0468db0c2c05b96b5430ccb4e7c393aa3b2b4f832c4f0ad0aaaf3173b18539920307

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-node-linux-amd64.tar.gz) | 9885a94a993a26b24b9734c3b722bbf78086909103f361ca7b4ca5af536b904ebca399ccfcc112b87d45e55f2539ce4bcafda7de325b57ce17be36db2ef13d88
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-node-linux-arm.tar.gz) | 3c525c726f3aac3bf0671caa5c843caa7fe552d6a2e41557516d001ab20b32b954185ad31255e80b4b2e25de4d09426a607c1e450607634ce60cb8cad4358969
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-node-linux-arm64.tar.gz) | 5f43317127308cca08e7fd4c3aae96df55735f525e59c837376971b50106f5c87a298a1e1edb7980e0283e1b2c009d1c4fef4bf44fba6f3beef1c78491a25052
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-node-linux-ppc64le.tar.gz) | 8a69bf376105de8c85df48dfcbdbd6b24775c15e5772eeea0cebad34bc33340cd83ddecb9b80ce515766629d46bf9f22fd11824c960b17e91fb7c25d05f6610a
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-node-linux-s390x.tar.gz) | 850f894985ece111f596875c9bd86c304de13968b689d09e4fbbff149e8403e564606e88ea51c87be2c14598dab28c641f6055de66335286344d811a6eaaaa54
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.3/kubernetes-node-windows-amd64.tar.gz) | 5aa2d9ee71f6992948cca929caf5c413b2c898e96ed1d10eaf6f35ca656e4801bf7fe6fbd63f5260811c8e46b7179c68d5713160ee1bedffcf7b3ed2f1350637

## Changelog since v1.19.2

## Changes by Kind

### Feature

- Kubernetes is now built using go1.15.2
  - build: Update to k/repo-infra@v0.1.1 (supports go1.15.2)
  - build: Use go-runner:buster-v2.0.1 (built using go1.15.1)
  - bazel: Replace --features with Starlark build settings flag
  - hack/lib/util.sh: some bash cleanups
    
    - switched one spot to use kube::logging
    - make kube::util::find-binary return an error when it doesn't find
      anything so that hack scripts fail fast instead of with '' binary not
      found errors.
    - this required deleting some genfeddoc stuff. the binary no longer
      exists in k/k repo since we removed federation/, and I don't see it
      in https://github.com/kubernetes-sigs/kubefed/ either. I'm assuming
      that it's gone for good now.
  
  - bazel: output go_binary rule directly from go_binary_conditional_pure
    
    From: @mikedanese:
    Instead of aliasing. Aliases are annoying in a number of ways. This is
    specifically bugging me now because they make the action graph harder to
    analyze programmatically. By using aliases here, we would need to handle
    potentially aliased go_binary targets and dereference to the effective
    target.
  
    The comment references an issue with `pure = select(...)` which appears
    to be resolved considering this now builds.
  
  - make kube::util::find-binary not dependent on bazel-out/ structure
  
    Implement an aspect that outputs go_build_mode metadata for go binaries,
    and use that during binary selection. ([#94838](https://github.com/kubernetes/kubernetes/pull/94838), [@justaugustus](https://github.com/justaugustus)) [SIG Architecture, Release and Testing]

### Design

- Prevent logging of docker config contents if file is malformed ([#95346](https://github.com/kubernetes/kubernetes/pull/95346), [@sfowl](https://github.com/sfowl)) [SIG Auth and Node]

### Bug or Regression

- Do not fail sorting empty elements. ([#94666](https://github.com/kubernetes/kubernetes/pull/94666), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- Ensure getPrimaryInterfaceID not panic when network interfaces for Azure VMSS are null ([#94802](https://github.com/kubernetes/kubernetes/pull/94802), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fix azure file migration panic ([#94853](https://github.com/kubernetes/kubernetes/pull/94853), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix bug where loadbalancer deletion gets stuck because of missing resource group &#35;75198 ([#93962](https://github.com/kubernetes/kubernetes/pull/93962), [@phiphi282](https://github.com/phiphi282)) [SIG Cloud Provider]
- Fix detach azure disk issue when vm not exist ([#95177](https://github.com/kubernetes/kubernetes/pull/95177), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix etcd_object_counts metric reported by kube-apiserver ([#94819](https://github.com/kubernetes/kubernetes/pull/94819), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fix network_programming_latency metric reporting for Endpoints/EndpointSlice deletions, where we don't have correct timestamp ([#95363](https://github.com/kubernetes/kubernetes/pull/95363), [@wojtek-t](https://github.com/wojtek-t)) [SIG Network and Scalability]
- Fix scheduler cache snapshot when a Node is deleted before its Pods ([#95153](https://github.com/kubernetes/kubernetes/pull/95153), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Fix the `cloudprovider_azure_api_request_duration_seconds` metric buckets to correctly capture the latency metrics. Previously, the majority of the calls would fall in the "+Inf" bucket. ([#94943](https://github.com/kubernetes/kubernetes/pull/94943), [@marwanad](https://github.com/marwanad)) [SIG Cloud Provider and Instrumentation]
- Fix: azure disk resize error if source does not exist ([#93011](https://github.com/kubernetes/kubernetes/pull/93011), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: detach azure disk broken on Azure Stack ([#94885](https://github.com/kubernetes/kubernetes/pull/94885), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fixed a bug where improper storage and comparison of endpoints led to excessive API traffic from the endpoints controller ([#94937](https://github.com/kubernetes/kubernetes/pull/94937), [@damemi](https://github.com/damemi)) [SIG Apps, Network and Testing]
- Fixed a regression that sometimes prevented `kubectl portforward` to work when TCP and UDP services were configured on the same port ([#94728](https://github.com/kubernetes/kubernetes/pull/94728), [@amorenoz](https://github.com/amorenoz)) [SIG CLI]
- Fixes a bug where EndpointSlices would not be recreated after rapid Service recreation. ([#94730](https://github.com/kubernetes/kubernetes/pull/94730), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- Fixes a race condition in kubelet pod handling ([#94774](https://github.com/kubernetes/kubernetes/pull/94774), [@auxten](https://github.com/auxten)) [SIG Node]
- Gracefully delete nodes when their parent scale set went missing ([#95289](https://github.com/kubernetes/kubernetes/pull/95289), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- Kubeadm: relax the validation of kubeconfig server URLs. Allow the user to define custom kubeconfig server URLs without erroring out during validation of existing kubeconfig files (e.g. when using external CA mode). ([#94816](https://github.com/kubernetes/kubernetes/pull/94816), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: warn but do not error out on missing "ca.key" files for root CA, front-proxy CA and etcd CA, during "kubeadm join --control-plane" if the user has provided all certificates, keys and kubeconfig files which require signing with the given CA keys. ([#94988](https://github.com/kubernetes/kubernetes/pull/94988), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]

### Other (Cleanup or Flake)

- Masks ceph RBD adminSecrets in logs when logLevel >= 4 ([#95245](https://github.com/kubernetes/kubernetes/pull/95245), [@sfowl](https://github.com/sfowl)) [SIG Storage]
- Vsphere: improve logging message on node cache refresh event ([#95236](https://github.com/kubernetes/kubernetes/pull/95236), [@andrewsykim](https://github.com/andrewsykim)) [SIG Cloud Provider]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.19.2


## Downloads for v1.19.2

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes.tar.gz) | b0f01f8b98e37cfb706d7b0b53a527232f3507be52252614d6c55e41e36cff592c70d704083058bbc557536d9a4340174922f03e05d4f2170186a37be4b65d05
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-src.tar.gz) | 6defaedb3edb9302732d168b45074a1fbcfbf17828e6a928eeae04ff88b946c87cd388e3e793903ebdbaa56a76e7884999d437e78cbc918a2e3347b310a1937a

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-darwin-amd64.tar.gz) | a2ed8bdb181dadd509f769b300413675c92fd864a70050ca916b4f0fbc4499e24acce48794d346d7de1531cc96c7cd93aa47913f1dab7c9b2c2a3df95a574fd7
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-linux-386.tar.gz) | d4efc6bb7123ece383d52cecea8df3fa4ae10a205867a70b86eb6213ffff2b2b1f32e75f252cf64724d0e6547f7a584e41c59a17d52dbb84278d9fbfa3c15860
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-linux-amd64.tar.gz) | fe1aa1fa3d0c1a311d26159cb6b8acdc13d9201b647cc65b7bf2ac6e13400c07a0947fea479d1abd2da499809116dc64a1ee973ac33c81514d6d418f8bc6f5ac
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-linux-arm.tar.gz) | 66f735bcc74409bc53ccb935d53447391748b6962519c8f08a421406b6adf0bf492d1ab39616110d1d99c9a768d35aed104dc50b54c2bff09f68cb123fb8b21c
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-linux-arm64.tar.gz) | 1efe5cdd2adfa6aa8b60a258d53c3a8fb23d0a74cbb07256b13c7d423079c676e9e68dd57629b9cc82d8182837161e47be89405d45f7acecd0953c0260faa3fa
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-linux-ppc64le.tar.gz) | 49555175731d3026e373ddbb4b7c42fc6eb8e6d7b91aa383fe6e85bed69e64add5947eb41d7849ca5a4026fea07b6c492de823350877ee6a9baae23c4c0729d8
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-linux-s390x.tar.gz) | f1ee570e28b04a1c9faadd27367e7e068d5ed7fc9ad92dc9d11bc5fa8f38033b7a2c35ab7d91b3599b441737933172768cdfee3b330928a45109dddb0344f1ff
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-windows-386.tar.gz) | 128158500bab2be0b35f16dda329679ef10a559decf2b0162037fdb042463dcfe4d0d35ada070370d9a06e30af37a8716af619235983c19d08cd15123b012126
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-client-windows-amd64.tar.gz) | ab91df20937523378c3ab8d7ba89a867bc7a10b12ee8432a114b83411ef47e4f791bf49655381344315a4ff1c44ace52f084556f21be4de4ba49284139a20642

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-server-linux-amd64.tar.gz) | a62092bd3eaeb1f63e73478eb5436ba094d6822bfcc7e8eabb6ead9411da7ca5032f00a7328914461f29b5b184b12456e025e85e3a2b12f6137596d78b6febe7
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-server-linux-arm.tar.gz) | d166aa50b815abfb5489e7e1bf33e3624155ece72f6ef58b2faecb79925f61259100179410932fa30dd125a3ef9b734e247092d24b116476238e42f3d1bdf314
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-server-linux-arm64.tar.gz) | bfca09d2912a8afbe3bb4058f0da7a9f6ae5866550beed5c07e1c633586e00345eeb5c3b1114799c9846ee5924f44c570afb150d51e39f2f83170279ea2f298c
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-server-linux-ppc64le.tar.gz) | c175bba17f77506502dd0715a67c77aa57756d6580a806e08100f2483235b819ddbb2e602efdbd1fa5ebeb72ca4ca72bd38d07b5daea4389a09d90d061187358
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-server-linux-s390x.tar.gz) | 9a3f8fec9dc619250016124d96ebf2863b1523937bd88b9d8f2d5da5f6fa9728b96c4c8f01eb7366f4133c4420005c00eebe8862d71e6f640e81b53de0f84da1

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-node-linux-amd64.tar.gz) | aab323493d570aa7122a43051542a84b7e7a9f82fa5778376afa543f698959baf04b1cbe9c15a0c8eacad2cbdd2007e4d4d2d28ea790fbca4d9b2b837bc756e9
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-node-linux-arm.tar.gz) | 9652f9bf3a3a4199bdcc7a07adc122c7c9b643a1c40b1d8f2ab3eee098362d2c891a95f7bf8bbc2a71773110f3b712ddd90e05a715caf819754777c216fe2dbc
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-node-linux-arm64.tar.gz) | 0816c2a034ee59daf1c35335d7a7f17af2eda94cf887b1f0329ecf2224760f633d02a5460eb01c542f119e6aceeb1c4fd7b86db2dd30255c55d16fcb48b96dbd
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-node-linux-ppc64le.tar.gz) | bddb8e1ca7ac86e7996b100c8f95c046708e8f0a75bc1db5a8c21f38cf2de85760e723cdc0a1e0cff154ec1fe8dd0ba69b4bd576942d668f58f16f3f6880994c
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-node-linux-s390x.tar.gz) | 90372098ad3e2a0681db8923d9858810fe2ce618e9fcfcc00396706ccc75045db8ab3dcdc09b24430bd8e1f96613c8855cc69b8f76210f87ba17468a675d238a
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.2/kubernetes-node-windows-amd64.tar.gz) | 84b2c2a2b3bc24e33e9cf3afccca0fa3e0fd9edb4f89c61d1bc7024d78bbd06ba39e271a2e1106a9c31754780a220f90b505c16179c0270c81da12795541f4ba

## Changelog since v1.19.1

## Changes by Kind

### API Change

- Fix conversions for custom metrics. ([#94654](https://github.com/kubernetes/kubernetes/pull/94654), [@wojtek-t](https://github.com/wojtek-t)) [SIG Instrumentation]

### Bug or Regression

- Fixed a panic in kubectl debug when pod has multiple init containers or ephemeral containers ([#94580](https://github.com/kubernetes/kubernetes/pull/94580), [@kiyoshim55](https://github.com/kiyoshim55)) [SIG CLI]

### Other (Cleanup or Flake)

- Update CNI plugins to v0.8.7 ([#94367](https://github.com/kubernetes/kubernetes/pull/94367), [@justaugustus](https://github.com/justaugustus)) [SIG Cloud Provider, Network, Node, Release and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.19.1


## Downloads for v1.19.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes.tar.gz) | cbcac78b122cfe7301247c935efdfe0ad42ca8b1229f35d4ba2feb3286eca86f7996bfcbbd810e611f6efa37a74ece7d46eac585f2b7a9fda5e9461f8bf5dbac
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-src.tar.gz) | 00ea5562087bfc902f3081385bd8200e003038f86cd62e02da20b70583c33c8b8ec2f4d46268ea7278d447cbddb1ab6840b2174ee71a4c68e2306ed416159c35

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-darwin-amd64.tar.gz) | e2926509570d772816de6dd23dc31c23833ed777be924abfb6d14901bd8b653fa8d7345b21b91f005e748813d5cbdb26168b6065a31219dfb6cc2686b4e878ac
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-linux-386.tar.gz) | 0a2b4cb64ed7f1e6ac107ee3fdea911d90fa5d7cc907c3fefc9da732cc5a8082f18a8167d7ece68f1e15f6f400e6b4b7a6ad0986205965acc8fc403ce22613b2
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-linux-amd64.tar.gz) | 424ba5a094947e62c935ad7049efc13762f2f232ed05872e4f3945b3c39d6e6c82d1e3f60d0970fdc8f01cf4886ef5de77224df7e481774c04d86429dbfb2055
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-linux-arm.tar.gz) | 48ac4e600ff914f307fc4c194cdf925c3d9ca077d54d4e09d32941079e610f1d42333e1e31f334c4ae758e4957dc5dad2e6bcb026a3ed5befed2dcb942bf3d75
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-linux-arm64.tar.gz) | 765446ed59124fef6ff3aff0b2221935d1554b095144f9af37ea7da20a11ce328f7a60fd18f9d75ca3649c02667f0e1334cdec22f39f81fb6992c0071447a5e9
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-linux-ppc64le.tar.gz) | e220220fb48adb79258ca803b172a7dcc5ac4ced079aea73c4b172c57738840c59dda9f6ef207ef70e937a6599d748d2d0d2b02f7814281a56eb91dbfc55f59a
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-linux-s390x.tar.gz) | 9c15feb128348df0e843af439a6c5b7058a31b815da973829da8140021c0e3f1d1ac881f6e9357737a9244b58fbd7f34664903fe34414d802df67e901e57742a
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-windows-386.tar.gz) | 3404e4c903715e37ad501cf51d405e636827059a76e00960629ded4479174fdda8148166c751e240e94048a2408e6b94b964436879f7764be2f3e618556eda6e
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-client-windows-amd64.tar.gz) | a3b219ffeb432ecefd64f7734207b51e048c631cb28e989f86ce75815a5cc9927b9d3e7ee06d0f02c5f5a299c0e3b21d62c9c462c52d80810fb3a8424087c445

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-server-linux-amd64.tar.gz) | 70f0389c8c5f35eaa67736aebb857f0b26ca777a77662c21ebd2b719608c16663e888b8e1b93e90564c206f179524ff421dbb5cb1f3289a71f27eca388087451
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-server-linux-arm.tar.gz) | 214372f65ea6f9c90d54d8b9e531e1f6cc2a0a828dffab7e6cda0a747926580509930d25e44354b27a4933df49c016e4a0ae51439228cd31cc582a6996bd1615
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-server-linux-arm64.tar.gz) | 4ce4f599057e1f86a37f070f91e9e8e624b0fd73dd8a4d79bae60097734a0ed2a9855bf1f5795be76ade5815d1f97ca06b764ba9efe5af11afe420cdc659fa84
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-server-linux-ppc64le.tar.gz) | 47d788d472585857cb7dec42ccd5d1eaec06970c4257617a53a6b1dc774ee07b33c7b5d87d741486800ffbed82b6ff3505504c1ea03de68bce0382ed44829473
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-server-linux-s390x.tar.gz) | d911a7345a114b769a8ef9f26eb6d06c321220da07d8d6094f5e4c1dff6394b24ddd2859f55986ff87397c3f59355cd9c28f243c96da432ea385ec09631de223

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-node-linux-amd64.tar.gz) | 16d32f85594ca6b3f26ac6827be2ea927b79966efa55edcab1f3d5628744759a1e1f7c04e50dfe47da17a2d63f70cc5989ae47f057bc958c6f4c52461406090c
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-node-linux-arm.tar.gz) | b73af373e504fadbff9ef88c2fee4fe1d449c9343ed73d37a0c6b39bef0038c5bf15783062cc3fa50ca63969f5e8062517e9e0b5ce7a6db633eb5555a4611276
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-node-linux-arm64.tar.gz) | 842e5687d1c99f655431732974289825986425949d2cc0a1c35326c8e6c446cc5ffaeef69d9efdb9bba16c7f6aa8c8ae59e9575929747bff1238a47c8d04e959
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-node-linux-ppc64le.tar.gz) | 222e9e10f2f38fdc160ddb73f01d106788d4b9863f8aef2ad427a965187ebc90f1cbce3985d9197db2512a0918b59e7dd79e20c783cb95a06c40ce6afe0594a9
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-node-linux-s390x.tar.gz) | 7edf2e03726ff3a340fdb09d8c2fc9aaff8e48a06499cdfb9956879552d18cb12e9ea846296c3d2c7237f6941967d3fed7548caef3ce65fcd7ec06736a9a494a
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.1/kubernetes-node-windows-amd64.tar.gz) | 09997c6edc473964f1d2ebf2fa4c8287e9146088f26257fc29277c5a1ad6794ff251c33fb4777919440beee0d46ea45cd42a0f18721bc38d4a8b5c65d57cece2

## Changelog since v1.19.0

## Changes by Kind

### Bug or Regression

- Azure: fix a bug that kube-controller-manager would panic if wrong Azure VMSS name is configured ([#94306](https://github.com/kubernetes/kubernetes/pull/94306), [@knight42](https://github.com/knight42)) [SIG Cloud Provider]
- Build/lib/release: Explicitly use '--platform' in building server images
  
  When we switched to go-runner for building the apiserver,
  controller-manager, and scheduler server components, we no longer
  reference the individual architectures in the image names, specifically
  in the 'FROM' directive of the server image Dockerfiles.
  
  As a result, server images for non-amd64 images copy in the go-runner
  amd64 binary instead of the go-runner that matches that architecture.
  
  This commit explicitly sets the '--platform=linux/${arch}' to ensure
  we're pulling the correct go-runner arch from the manifest list.
  
  Before:
  `FROM ${base_image}`
  
  After:
  `FROM --platform=linux/${arch} ${base_image}` ([#94613](https://github.com/kubernetes/kubernetes/pull/94613), [@justaugustus](https://github.com/justaugustus)) [SIG Release]
- Fix a concurrent map writes error in kubelet ([#93773](https://github.com/kubernetes/kubernetes/pull/93773), [@knight42](https://github.com/knight42)) [SIG Node]
- Fix a regression where kubeadm bails out with a fatal error when an optional version command line argument is supplied to the "kubeadm upgrade plan" command ([#94421](https://github.com/kubernetes/kubernetes/pull/94421), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Fixed bug in reflector that couldn't recover from "Too large resource version" errors with API servers 1.17.0-1.18.5 ([#94316](https://github.com/kubernetes/kubernetes/pull/94316), [@janeczku](https://github.com/janeczku)) [SIG API Machinery]
- Kubeadm: make the kubeconfig files for the kube-controller-manager and kube-scheduler use the LocalAPIEndpoint instead of the ControlPlaneEndpoint. This makes kubeadm clusters more reseliant to version skew problems during immutable upgrades: https://kubernetes.io/docs/setup/release/version-skew-policy/&#35;kube-controller-manager-kube-scheduler-and-cloud-controller-manager ([#94398](https://github.com/kubernetes/kubernetes/pull/94398), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Update default etcd server version to 3.4.13 ([#94536](https://github.com/kubernetes/kubernetes/pull/94536), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Upon successful authorization check, an impersonated user is added to the system:authenticated group.
  system:anonymous when impersonated is added to the system:unauthenticated group. ([#94408](https://github.com/kubernetes/kubernetes/pull/94408), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Testing]
- Use NLB Subnet CIDRs instead of VPC CIDRs in Health Check SG Rules ([#93515](https://github.com/kubernetes/kubernetes/pull/93515), [@t0rr3sp3dr0](https://github.com/t0rr3sp3dr0)) [SIG Cloud Provider]

### Other (Cleanup or Flake)

- Fixes the flooding warning messages about setting volume ownership for configmap/secret volumes ([#92878](https://github.com/kubernetes/kubernetes/pull/92878), [@jvanz](https://github.com/jvanz)) [SIG Instrumentation, Node and Storage]
- Kubeadm: remove the CoreDNS check for known image digests when applying the addon ([#94506](https://github.com/kubernetes/kubernetes/pull/94506), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.19.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.19.0

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes.tar.gz) | `448b941e973a519a500eb24786f6deb7eebd0e1ecb034941e382790ff69dfc2838715a222cfc53bea7b75f2c6aedc7425eded4aad69bf88773393155c737f9c0`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-src.tar.gz) | `47d253e6eb1f6da730f4f3885e205e6bfde88ffe66d92915465108c9eaf8e3c5d1ef515f8bf804a726db057433ecd25008ecdef624ee68ad9c103d1c7a615aad`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-darwin-amd64.tar.gz) | `7093a34298297e46bcd1ccb77a9c83ca93b8ccb63ce2099d3d8cd8911ccc384470ac202644843406f031c505a8960d247350a740d683d8910ca70a0b58791a1b`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-linux-386.tar.gz) | `891569cff7906732a42b20b86d1bf20a9fe873f87b106e717a5c0f80728b5823c2a00c7ccea7ec368382509f095735089ddd582190bc51dcbbcef6b8ebdbd5cc`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-linux-amd64.tar.gz) | `1590d4357136a71a70172e32820c4a68430d1b94cf0ac941ea17695fbe0c5440d13e26e24a2e9ebdd360c231d4cd16ffffbbe5b577c898c78f7ebdc1d8d00fa3`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-linux-arm.tar.gz) | `bc0fb19fb6af47f591adc64b5a36d3dffcadc35fdfd77a4a222e037dbd2ee53fafb84f13c4e307910cfa36b3a46704063b42a14ceaad902755ec14c492ccd51d`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-linux-arm64.tar.gz) | `6ff47f4fdfb3b5f2bfe18fd792fe9bfc747f06bf52de062ee803cda87ac4a98868d8e1211742e32dd443a4bdb770018bbdde704dae6abfc6d80c02bdfb4e0311`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-linux-ppc64le.tar.gz) | `d8816518adc3a7fc00f996f23ff84e6782a3ebbba7ef37ba44def47b0e6506fefeeaf37d0e197cecf0deb5bd1a8f9dd1ba82af6c29a6b9d21b8e62af965b6b81`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-linux-s390x.tar.gz) | `662fd4618f2b747d2b0951454b9148399f6cd25d3ca7c40457b6e02cb20df979138cad8cccd18fc8b265d9426c90828d3f0b2a6b40d9cd1a1bdc17219e35ed33`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-windows-386.tar.gz) | `d90cb92eb33ecbfab7a0e3a2da60ab10fc59132e4bc9abe0a1461a13222b5016704a7cfe0bf9bcf5d4ec55f505ffbbf53162dfe570e8f210e3f68b0d3a6bf7e3`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-client-windows-amd64.tar.gz) | `6ec32a8a62b69363a524c4f8db765ff4bd16ea7e5b0eb04aa5a667f8653eda18c357a97513d9e12f0ba1612516acb150deffb6e3608633c62b97a15b6efa7cc0`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-server-linux-amd64.tar.gz) | `7c268bd58e67d3c5016f3fcc9f4b6d2da7558af5a2c708ff3baf767b39e847e3d35d4fd2fa0f640bedbfb09a445036cafbe2f04357a88dada405cfc2ded76972`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-server-linux-arm.tar.gz) | `fcbf8d9004f1cd244a82b685abaf81f9638c3cc1373d78e705050042cfa6a004f8eed92f4721539dcd169c55b662d10416af19cff7537a8dfef802dc41b4088b`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-server-linux-arm64.tar.gz) | `e21f54a35ff29e919e98fe81758f654ea735983d5a9d08dab9484598b116843830a86ceb5cf0a23d27b7f9aba77e5f0aa107c171a0837ba781d508ebbea76f55`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-server-linux-ppc64le.tar.gz) | `c7014c782683f8f612c7805654b632aab4c5dce895ee8f9ef24360616e24240ce59ddf3cf27c3170df5450d8fe14fbca3fb7cddfc9b74ae37943081f0fa4b6b3`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-server-linux-s390x.tar.gz) | `3ac2d6b273e5b650f63260aae164fc6781ad5760f63cca911f5db9652c4bf32e7e7b25728987befc6dfda89c5c56969681b75f12b17141527d4e1d12f3d41f3c`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-node-linux-amd64.tar.gz) | `d5e21432a4ab019f00cd1a52bbbdb00feb3db2ce96b41a58b1ee27d8847c485f5d0efe13036fd1155469d6d15f5873a5a892ecc0198f1bae1bf5b586a0129e75`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-node-linux-arm.tar.gz) | `bd57adf060813b06be2b33439d6f60d13630c0251ef96ba473274073200ea118f5622ec31ed714cc57bd9da410655e958a7700a5742ae7e4b6406ab12fbf21f3`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-node-linux-arm64.tar.gz) | `3ee70abc0a5cbf1ef5dde0d27055f4d17084585c36a2cf41e3fd925d206df0b583f50dc1c118472f198788b65b2c447aa40ad41646b88791659d2dfb69b3890b`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-node-linux-ppc64le.tar.gz) | `0f4368f229c082b2a75e7089a259e487d60b20bc8edf650dd7ca0fe23c51632397c2ef24c9c6cef078c95fce70d9229a5b4ff682c34f65a44bc4be3329c8ccde`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-node-linux-s390x.tar.gz) | `8f0b6839fc0ad51300221fa7f32134f8c687073715cc0839f7aacb21a075c66dab113369707d03e9e0e53be62ca2e1bdf04d4b26cff805ae9c7a5a4b864e3eae`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0/kubernetes-node-windows-amd64.tar.gz) | `587651158c9999e64e06186ef2e65fe14d46ffdae28c5d8ee6261193bfe4967717f997ebe13857fa1893bbf492e1cc1f816bce86a94c6df9b7a0264848391397`

## Changelog since v1.18.0

## What’s New (Major Themes)

### Deprecation warnings

SIG API Machinery implemented [warnings when using deprecated APIs](https://kubernetes.io/docs/reference/using-api/deprecation-policy/#rest-resources-aka-api-objects)
that are visible to `kubectl` users and API consumers, and metrics visible to cluster administrators.
Requests to a deprecated API are returned with a warning containing a target removal release and any replacement API.
Warnings can also be returned by [admission webhooks](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/#admissionreview-response-warning),
and specified for [deprecated versions of custom resources](https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definition-versioning/#version-deprecation).

### Avoiding permanent beta

From Kubernetes 1.20 onwards, SIG Architecture will implement a new policy to transition all REST APIs out of beta within nine months. The idea behind the new policy is to avoid features staying in beta for a long time. Once a new API enters beta, it will have nine months to either:

 - reach GA, and deprecate the beta, or
 - have a new beta version _(and deprecate the previous beta)_.

If a REST API reaches the end of that nine-month countdown, then the next Kubernetes release will deprecate that API version. More information can be found on [the Kubernetes Blog](https://kubernetes.io/blog/2020/08/21/moving-forward-from-beta/).

### Expanded CLI support for debugging workloads and nodes

SIG CLI expanded on debugging with `kubectl` to support two new debugging workflows: debugging workloads by creating a copy, and debugging nodes by creating a container in host namespaces. These can be convenient to:
 - Insert a debug container in clusters that don’t have ephemeral containers enabled 
 - Modify a crashing container for easier debugging by changing its image, for example to busybox, or its command, for example, to `sleep 1d` so you have time to `kubectl exec`.
 - Inspect configuration files on a node's host filesystem

Since these new workflows don’t require any new cluster features, they’re available for experimentation with your existing clusters via `kubectl alpha debug`. We’d love to hear your feedback on debugging with `kubectl`. Reach us by opening an issue, visiting [#sig-cli](https://kubernetes.slack.com/messages/sig-cli) or commenting on enhancement [#1441](https://features.k8s.io/1441).

### Structured logging

SIG Instrumentation standardized the structure of log messages and references to Kubernetes objects. Structured logging makes parsing, processing, storing, querying and analyzing logs easier. New methods in the klog library enforce log message structure.

### EndpointSlices are now enabled by default

EndpointSlices are an exciting new API that provides a scalable and extensible alternative to the Endpoints API. EndpointSlices track IP addresses, ports, readiness, and topology information for Pods backing a Service.

In Kubernetes 1.19 this feature will be enabled by default with kube-proxy reading from EndpointSlices instead of Endpoints. Although this will mostly be an invisible change, it should result in noticeable scalability improvements in large clusters. It will also enable significant new features in future Kubernetes releases like Topology Aware Routing.

### Ingress graduates to General Availability

SIG Network has graduated the widely used [Ingress API](https://kubernetes.io/docs/concepts/services-networking/ingress/) to general availability in Kubernetes 1.19. This change recognises years of hard work by Kubernetes contributors, and paves the way for further work on future networking APIs in Kubernetes.

### seccomp graduates to General Availability

The seccomp (secure computing mode) support for Kubernetes has graduated to General Availability (GA). This feature can be used to increase the workload security by restricting the system calls for a Pod (applies to all containers) or single containers.

Technically this means that a first class `seccompProfile` field has been added to the Pod and Container `securityContext` objects:

```yaml
securityContext:
  seccompProfile:
    type: RuntimeDefault|Localhost|Unconfined # choose one of the three
    localhostProfile: my-profiles/profile-allow.json # only necessary if type == Localhost
```

The support for `seccomp.security.alpha.kubernetes.io/pod` and `container.seccomp.security.alpha.kubernetes.io/...` annotations are now deprecated, and will be removed in Kubernetes v1.22.0. Right now, an automatic version skew handling will convert the new field into the annotations and vice versa. This means there is no action required for converting existing workloads in a cluster.

You can find more information about how to restrict container system calls with seccomp in the new [documentation page on Kubernetes.io][seccomp-docs]

[seccomp-docs]: https://kubernetes.io/docs/tutorials/clusters/seccomp/


### Production images moved to community control

As of Kuberenetes v1.19, Kubernetes container images are stored on a community-controlled storage bucket, 
located at `{asia,eu,us}.gcr.io/k8s-artifacts-prod`. The `k8s.gcr.io` vanity domain has been updated 
to this new bucket. This brings production artifacts under community control.

### KubeSchedulerConfiguration graduates to Beta

SIG Scheduling graduates `KubeSchedulerConfiguration` to Beta. The [KubeSchedulerConfiguration](https://kubernetes.io/docs/reference/scheduling/config) feature allows you to tune the algorithms and other settings of the kube-scheduler. You can easily enable or disable specific functionality (contained in plugins) in selected scheduling phases without having to rewrite the rest of the configuration. Furthermore, a single kube-scheduler instance can serve different configurations, called profiles. Pods can select the profile they want to be scheduled under via the `.spec.schedulerName` field.

### CSI Migration - AzureDisk and vSphere (beta)
 
In-tree volume plugins and all cloud provider dependencies are being moved out of the Kubernetes core. The CSI migration feature allows existing volumes using the legacy APIs to continue to function even when the code has been removed, by routing all the volume operations to the respective CSI driver. The AzureDisk and vSphere implementations of this feature have been promoted to beta.

### Storage capacity tracking

Traditionally, the Kubernetes scheduler was based on the assumption that additional persistent storage is available everywhere in the cluster and has infinite capacity. Topology constraints addressed the first point, but up to now pod scheduling was still done without considering that the remaining storage capacity may not be enough to start a new pod. [Storage capacity tracking](https://github.com/kubernetes/enhancements/tree/master/keps/sig-storage/1472-storage-capacity-tracking), a new alpha feature, addresses that by adding an API for a CSI driver to report storage capacity and uses that information in the Kubernetes scheduler when choosing a node for a pod. This feature serves as a stepping stone for supporting dynamic provisioning for local volumes and other volume types that are more capacity constrained.

### CSI Volume health monitoring
 
The alpha version of CSI health monitoring is being released with Kubernetes 1.19. This feature enables CSI Drivers to share abnormal volume conditions from the underlying storage systems with Kubernetes so that they can be reported as events on PVCs or Pods. This feature serves as a stepping stone towards programmatic detection and resolution of individual volume health issues by Kubernetes.

### General ephemeral volumes

Kubernetes provides volume plugins whose lifecycle is tied to a pod and can be used as scratch space (e.g. the builtin “empty dir” volume type) or to load some data in to a pod (e.g. the builtin ConfigMap and Secret volume types or “CSI inline volumes”). The new [generic ephemeral volumes](https://github.com/kubernetes/enhancements/tree/master/keps/sig-storage/1698-generic-ephemeral-volumes) alpha feature allows any existing storage driver that supports dynamic provisioning to be used as an ephemeral volume with the volume’s lifecycle bound to the Pod.
 - It can be used to provide scratch storage that is different from the root disk, for example persistent memory, or a separate local disk on that node.
 - All StorageClass parameters for volume provisioning are supported.
 - All features supported with PersistentVolumeClaims are supported, such as storage capacity tracking, snapshots and restore, and volume resizing.

### Immutable Secrets and ConfigMaps (beta)

Secret and ConfigMap volumes can be marked as immutable, which significantly reduces load on the API server if there are many Secret and ConfigMap volumes in the cluster.
See [ConfigMap](https://kubernetes.io/docs/concepts/configuration/configmap/) and [Secret](https://kubernetes.io/docs/concepts/configuration/secret/) for more information.

### CSI Proxy for Windows

The CSI Proxy for Windows is being promoted to beta along with the 1.19 release. This CSI Proxy enables CSI Drivers to run on Windows by allowing containers in Windows to perform privileged storage operations. At beta, the CSI Proxy for Windows supports storage drivers using direct attached disks and SMB.

### Dashboard v2

SIG UI has released v2 of the Kubernetes Dashboard add-on. You can find the most recent release in the [kubernetes/dashboard](https://github.com/kubernetes/dashboard/releases) repository. Kubernetes Dashboard now includes CRD support, new translations, and an updated version of AngularJS.

### Windows containerd support graduates to beta

Initially introduced in Kubernetes 1.18, Windows containerd support goes to Beta on this release. This includes the added support for Windows Server version 2004 (complete version compatibility can be found in the [documentation for Windows](https://kubernetes.io/docs/setup/production-environment/windows/intro-windows-in-kubernetes/#cri-containerd)).

SIG Windows is also including several addition to this release:
 - Direct Server Return (DSR) mode support, allowing large numbers of services to scale up efficiently
 - Windows containers  now honor CPU limits
 - Performance improvements for collections of metrics and summary

### Increase the Kubernetes support window to one year

As of Kubernetes 1.19, bugfix support via patch releases for a Kubernetes minor release has increased from 9 months to 1 year.

A survey conducted in early 2019 by the working group (WG) Long Term Support (LTS) showed that a significant subset of Kubernetes end-users fail to upgrade within the previous 9-month support period. 
A yearly support period provides the cushion end-users appear to desire, and is more in harmony with familiar annual planning cycles.

## Known Issues

The new storage capacity tracking alpha feature is known to be affected by a limitation of the WaitForFirstConsumer volume binding mode: [#94217](https://github.com/kubernetes/kubernetes/issues/94217)

## Urgent Upgrade Notes 

### (No, really, you MUST read this before you upgrade)

- ACTION REQUIRED: Switch core master base images (kube-controller-manager) from debian to distroless. If you need Flex Volumes support using scripts, please build your own image with required packages (like bash) ([#91329](https://github.com/kubernetes/kubernetes/pull/91329), [@dims](https://github.com/dims)) [SIG Cloud Provider, Release, Storage and Testing]
- ACTION REQUIRED: Support for basic authentication via the --basic-auth-file flag has been removed.  Users should migrate to --token-auth-file for similar functionality. ([#89069](https://github.com/kubernetes/kubernetes/pull/89069), [@enj](https://github.com/enj)) [SIG API Machinery]
 - Azure blob disk feature(`kind`: `Shared`, `Dedicated`) has been deprecated, you should use `kind`: `Managed` in `kubernetes.io/azure-disk` storage class. ([#92905](https://github.com/kubernetes/kubernetes/pull/92905), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
 - CVE-2020-8559 (Medium): Privilege escalation from compromised node to cluster. See https://github.com/kubernetes/kubernetes/issues/92914 for more details.
  The API Server will no longer proxy non-101 responses for upgrade requests. This could break proxied backends (such as an extension API server) that respond to upgrade requests with a non-101 response code. ([#92941](https://github.com/kubernetes/kubernetes/pull/92941), [@tallclair](https://github.com/tallclair)) [SIG API Machinery]
 - Kubeadm does not set the deprecated '--cgroup-driver' flag in /var/lib/kubelet/kubeadm-flags.env, it will be set in the kubelet config.yaml. If you have this flag in /var/lib/kubelet/kubeadm-flags.env or /etc/default/kubelet (/etc/sysconfig/kubelet for RPMs) please remove it and set the value using KubeletConfiguration ([#90513](https://github.com/kubernetes/kubernetes/pull/90513), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
 - Kubeadm now respects user specified etcd versions in the ClusterConfiguration and properly uses them. If users do not want to stick to the version specified in the ClusterConfiguration, they should edit the kubeadm-config config map and delete it. ([#89588](https://github.com/kubernetes/kubernetes/pull/89588), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
 - Kubeadm respects resolvConf value set by user even if systemd-resolved service is active. kubeadm no longer sets the flag in '--resolv-conf' in /var/lib/kubelet/kubeadm-flags.env. If you have this flag in /var/lib/kubelet/kubeadm-flags.env or /etc/default/kubelet (/etc/sysconfig/kubelet for RPMs) please remove it and set the value using KubeletConfiguration ([#90394](https://github.com/kubernetes/kubernetes/pull/90394), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
 - Kubeadm: Move the "kubeadm init" phase "kubelet-start" later in the init workflow, after the "kubeconfig" phase. This makes kubeadm start the kubelet only after the KubeletConfiguration component config file (/var/lib/kubelet/config.yaml) is generated and solves a problem where init systems like OpenRC cannot crashloop the kubelet service. ([#90892](https://github.com/kubernetes/kubernetes/pull/90892), [@xphoniex](https://github.com/xphoniex)) [SIG Cluster Lifecycle]
 - The 'kubeadm config upload' command is finally removed after a full GA deprecation cycle. If you still use it, please, use 'kubeadm init phase upload-config' instead ([#92610](https://github.com/kubernetes/kubernetes/pull/92610), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
 - Upgrade kubescheduler.config.k8s.io/v1alpha2 to kubescheduler.config.k8s.io/v1beta1
  
   - `.bindTimeoutSeconds` was moved as part of plugin args for `VolumeBinding`,
     which can be configured separately per [profile](&#35;profiles).
   - `.extenders` are updated to satisfy API standards. In particular:
     - `.extenders` decoding is case sensitive. All fields are affected.
     - `.extenders[*].httpTimeout` is of type `metav1.Duration`.
     - `.extenders[*].enableHttps` is renamed to `.extenders[*].enableHTTPS`.
   - `RequestedToCapacityRatio` args decoding is case sensitive. All fields are affected.
   - `DefaultPodTopologySpread` [plugin](&#35;scheduling-plugins) is renamed to `SelectorSpread`.
   - `Unreserve` extension point is removed from Profile definition. All `Reserve`
     plugins implement an `Unreserve` call.
   - `.disablePreemption` was removed. Users can disable preemption by disabling the
     "DefaultPreemption" PostFilter plugin. ([#91420](https://github.com/kubernetes/kubernetes/pull/91420), [@pancernik](https://github.com/pancernik)) [SIG Scheduling]
 
## Changes by Kind

### Deprecation

- Added support for vSphere in-tree volumes migration to vSphere CSI driver. The in-tree vSphere Volume plugin will be deprecated and removed in a future release.
  
  Users that self-deploy Kubernetes on vSphere should enable CSIMigration + CSIMigrationvSphere features and install the vSphere CSI Driver (https://github.com/kubernetes-sigs/vsphere-csi-driver) to avoid disruption to existing Pod and PVC objects at that time. Users should start using the vSphere CSI driver directly for any new volumes.
  
  The CSI Migration feature for vSphere Volume also requires minimum vSphere vCenter/ESXi Version to be 7.0u1 and minimum HW Version to be VM version 15.
  
  vSAN raw policy parameter is deprecated for the in-tree vSphere Volume plugin and will be removed in a future release. ([#90911](https://github.com/kubernetes/kubernetes/pull/90911), [@divyenpatel](https://github.com/divyenpatel)) [SIG API Machinery, Node and Storage]
- Apiextensions.k8s.io/v1beta1 is deprecated in favor of apiextensions.k8s.io/v1 ([#90673](https://github.com/kubernetes/kubernetes/pull/90673), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Apiregistration.k8s.io/v1beta1 is deprecated in favor of apiregistration.k8s.io/v1 ([#90672](https://github.com/kubernetes/kubernetes/pull/90672), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Authentication.k8s.io/v1beta1 and authorization.k8s.io/v1beta1 are deprecated in 1.19 in favor of v1 levels and will be removed in 1.22 ([#90458](https://github.com/kubernetes/kubernetes/pull/90458), [@deads2k](https://github.com/deads2k)) [SIG API Machinery and Auth]
- Autoscaling/v2beta1 is deprecated in favor of autoscaling/v2beta2 ([#90463](https://github.com/kubernetes/kubernetes/pull/90463), [@deads2k](https://github.com/deads2k)) [SIG Autoscaling]
- Coordination.k8s.io/v1beta1 is deprecated in 1.19, targeted for removal in 1.22, use v1 instead. ([#90559](https://github.com/kubernetes/kubernetes/pull/90559), [@deads2k](https://github.com/deads2k)) [SIG Scalability]
- Ensure that volume capability and staging target fields are present in nodeExpansion CSI calls

  Behaviour of NodeExpandVolume being called between NodeStage and NodePublish is deprecated for CSI volumes. CSI drivers should support calling NodeExpandVolume after NodePublish if they have node EXPAND_VOLUME capability ([#86968](https://github.com/kubernetes/kubernetes/pull/86968), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Feat: azure disk migration go beta in 1.19. Feature gates CSIMigration to Beta (on by default) and CSIMigrationAzureDisk to Beta (off by default since it requires installation of the AzureDisk CSI Driver)
  The in-tree AzureDisk plugin "kubernetes.io/azure-disk" is now deprecated and will be removed in 1.23. Users should enable CSIMigration + CSIMigrationAzureDisk features and install the AzureDisk CSI Driver (https://github.com/kubernetes-sigs/azuredisk-csi-driver) to avoid disruption to existing Pod and PVC objects at that time.
  Users should start using the AzureDisk CSI Driver directly for any new volumes. ([#90896](https://github.com/kubernetes/kubernetes/pull/90896), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Kube-apiserver: the componentstatus API is deprecated. This API provided status of etcd, kube-scheduler, and kube-controller-manager components, but only worked when those components were local to the API server, and when kube-scheduler and kube-controller-manager exposed unsecured health endpoints. Instead of this API, etcd health is included in the kube-apiserver health check and kube-scheduler/kube-controller-manager health checks can be made directly against those components' health endpoints. ([#93570](https://github.com/kubernetes/kubernetes/pull/93570), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps and Cluster Lifecycle]
- Kubeadm: `kubeadm config view` command has been deprecated and will be removed in a feature release, please use `kubectl get cm -o yaml -n kube-system kubeadm-config` to get the kubeadm config directly ([#92740](https://github.com/kubernetes/kubernetes/pull/92740), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: deprecate the "kubeadm alpha kubelet config enable-dynamic" command. To continue using the feature please defer to the guide for "Dynamic Kubelet Configuration" at k8s.io. ([#92881](https://github.com/kubernetes/kubernetes/pull/92881), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: deprecate the feature `--experimental-kustomize` in favor of `--experimental-patches`. The supported patch formats are the same as "kubectl patch". They are read as files from a directory and can be applied to kubeadm components during init/join/upgrade. Only patching of static Pods is supported for the time being. ([#92017](https://github.com/kubernetes/kubernetes/pull/92017), [@neolit123](https://github.com/neolit123))
- Kubeadm: remove the deprecated "--use-api" flag for "kubeadm alpha certs renew" ([#90143](https://github.com/kubernetes/kubernetes/pull/90143), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes no longer supports building hyperkube images ([#88676](https://github.com/kubernetes/kubernetes/pull/88676), [@dims](https://github.com/dims)) [SIG Cluster Lifecycle and Release]
- Remove --export flag from kubectl get command. ([#88649](https://github.com/kubernetes/kubernetes/pull/88649), [@oke-py](https://github.com/oke-py)) [SIG CLI and Testing]
- Scheduler's alpha feature 'ResourceLimitsPriorityFunction' is completely removed due to lack of usage ([#91883](https://github.com/kubernetes/kubernetes/pull/91883), [@SataQiu](https://github.com/SataQiu)) [SIG Scheduling and Testing]
- Storage.k8s.io/v1beta1 is deprecated in favor of storage.k8s.io/v1 ([#90671](https://github.com/kubernetes/kubernetes/pull/90671), [@deads2k](https://github.com/deads2k)) [SIG Storage]

### API Change

- A new alpha-level field, `SupportsFsGroup`, has been introduced for CSIDrivers to allow them to specify whether they support volume ownership and permission modifications. The `CSIVolumeSupportFSGroup` feature gate must be enabled to allow this field to be used. ([#92001](https://github.com/kubernetes/kubernetes/pull/92001), [@huffmanca](https://github.com/huffmanca)) [SIG API Machinery, CLI and Storage]
- Added pod version skew strategy for seccomp profile to synchronize the deprecated annotations with the new API Server fields. Please see the corresponding section [in the KEP](https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/20190717-seccomp-ga.md&#35;version-skew-strategy) for more detailed explanations. ([#91408](https://github.com/kubernetes/kubernetes/pull/91408), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps, Auth, CLI and Node]
- Adds the ability to disable Accelerator/GPU metrics collected by Kubelet ([#91930](https://github.com/kubernetes/kubernetes/pull/91930), [@RenaudWasTaken](https://github.com/RenaudWasTaken)) [SIG Node]
- Admission webhooks can now return warning messages that are surfaced to API clients, using the `.response.warnings` field in the admission review response. ([#92667](https://github.com/kubernetes/kubernetes/pull/92667), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- CertificateSigningRequest API conditions were updated:
  - a `status` field was added; this field defaults to `True`, and may only be set to `True` for `Approved`, `Denied`, and `Failed` conditions
  - a `lastTransitionTime` field was added
  - a `Failed` condition type was added to allow signers to indicate permanent failure; this condition can be added via the `certificatesigningrequests/status` subresource.
  - `Approved` and `Denied` conditions are mutually exclusive
  - `Approved`, `Denied`, and `Failed` conditions can no longer be removed from a CSR ([#90191](https://github.com/kubernetes/kubernetes/pull/90191), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Auth, CLI and Node]
- Cluster admins can now turn off /logs endpoint in kubelet by setting enableSystemLogHandler to false in their kubelet configuration file. enableSystemLogHandler can be set to true only when enableDebuggingHandlers is also set to true. ([#87273](https://github.com/kubernetes/kubernetes/pull/87273), [@SaranBalaji90](https://github.com/SaranBalaji90)) [SIG Node]
- Custom Endpoints are now mirrored to EndpointSlices by a new EndpointSliceMirroring controller. ([#91637](https://github.com/kubernetes/kubernetes/pull/91637), [@robscott](https://github.com/robscott)) [SIG API Machinery, Apps, Auth, Cloud Provider, Instrumentation, Network and Testing]
- CustomResourceDefinitions added support for marking versions as deprecated by setting `spec.versions[*].deprecated` to `true`, and for optionally overriding the default deprecation warning with a `spec.versions[*].deprecationWarning` field. ([#92329](https://github.com/kubernetes/kubernetes/pull/92329), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- EnvVarSource api doc bug fixes ([#91194](https://github.com/kubernetes/kubernetes/pull/91194), [@wawa0210](https://github.com/wawa0210)) [SIG Apps]
- Fix bug in reflector that couldn't recover from "Too large resource version" errors ([#92537](https://github.com/kubernetes/kubernetes/pull/92537), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Fixed: log timestamps now include trailing zeros to maintain a fixed width ([#91207](https://github.com/kubernetes/kubernetes/pull/91207), [@iamchuckss](https://github.com/iamchuckss)) [SIG Apps and Node]
- Generic ephemeral volumes, a new alpha feature under the `GenericEphemeralVolume` feature gate, provide a more flexible alternative to `EmptyDir` volumes: as with `EmptyDir`, volumes are created and deleted for each pod automatically by Kubernetes. But because the normal provisioning process is used (`PersistentVolumeClaim`), storage can be provided by third-party storage vendors and all of the usual volume features work. Volumes don't need to be empt; for example, restoring from snapshot is supported. ([#92784](https://github.com/kubernetes/kubernetes/pull/92784), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Instrumentation, Node, Scheduling, Storage and Testing]
- Go1.14.4 is now the minimum version required for building Kubernetes ([#92438](https://github.com/kubernetes/kubernetes/pull/92438), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Storage and Testing]
- Hide managedFields from kubectl edit command ([#91946](https://github.com/kubernetes/kubernetes/pull/91946), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- K8s.io/apimachinery - scheme.Convert() now uses only explicitly registered conversions - default reflection based conversion is no longer available. `+k8s:conversion-gen` tags can be used with the `k8s.io/code-generator` component to generate conversions. ([#90018](https://github.com/kubernetes/kubernetes/pull/90018), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Apps and Testing]
- Kube-proxy: add `--bind-address-hard-fail` flag to treat failure to bind to a port as fatal ([#89350](https://github.com/kubernetes/kubernetes/pull/89350), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle and Network]
- Kubebuilder validation tags are set on metav1.Condition for CRD generation ([#92660](https://github.com/kubernetes/kubernetes/pull/92660), [@damemi](https://github.com/damemi)) [SIG API Machinery]
- Kubelet's --runonce option is now also available in Kubelet's config file as `runOnce`. ([#89128](https://github.com/kubernetes/kubernetes/pull/89128), [@vincent178](https://github.com/vincent178)) [SIG Node]
- Kubelet: add '--logging-format' flag to support structured logging ([#91532](https://github.com/kubernetes/kubernetes/pull/91532), [@afrouzMashaykhi](https://github.com/afrouzMashaykhi)) [SIG API Machinery, Cluster Lifecycle, Instrumentation and Node]
- Kubernetes is now built with golang 1.15.0-rc.1.
  - The deprecated, legacy behavior of treating the CommonName field on X.509 serving certificates as a host name when no Subject Alternative Names are present is now disabled by default. It can be temporarily re-enabled by adding the value x509ignoreCN=0 to the GODEBUG environment variable. ([#93264](https://github.com/kubernetes/kubernetes/pull/93264), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Storage and Testing]
- Promote Immutable Secrets/ConfigMaps feature to Beta and enable the feature by default.
  This allows to set `Immutable` field in Secrets or ConfigMap object to mark their contents as immutable. ([#89594](https://github.com/kubernetes/kubernetes/pull/89594), [@wojtek-t](https://github.com/wojtek-t)) [SIG Apps and Testing]
- Remove `BindTimeoutSeconds` from schedule configuration `KubeSchedulerConfiguration` ([#91580](https://github.com/kubernetes/kubernetes/pull/91580), [@cofyc](https://github.com/cofyc)) [SIG Scheduling and Testing]
- Remove kubescheduler.config.k8s.io/v1alpha1 ([#89298](https://github.com/kubernetes/kubernetes/pull/89298), [@gavinfish](https://github.com/gavinfish)) [SIG Scheduling]
- Reserve plugins that fail to reserve will trigger the unreserve extension point ([#92391](https://github.com/kubernetes/kubernetes/pull/92391), [@adtac](https://github.com/adtac)) [SIG Scheduling and Testing]
- Resolve regression in `metadata.managedFields` handling in update/patch requests submitted by older API clients ([#91748](https://github.com/kubernetes/kubernetes/pull/91748), [@apelisse](https://github.com/apelisse))
- Scheduler: optionally check for available storage capacity before scheduling pods which have unbound volumes (alpha feature with the new `CSIStorageCapacity` feature gate, only works for CSI drivers and depends on support for the feature in a CSI driver deployment) ([#92387](https://github.com/kubernetes/kubernetes/pull/92387), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Scheduling, Storage and Testing]
- Seccomp support has graduated to GA. A new `seccompProfile` field is added to pod and container securityContext objects. Support for `seccomp.security.alpha.kubernetes.io/pod` and `container.seccomp.security.alpha.kubernetes.io/...` annotations is deprecated, and will be removed in v1.22. ([#91381](https://github.com/kubernetes/kubernetes/pull/91381), [@pjbgf](https://github.com/pjbgf)) [SIG Apps, Auth, Node, Release, Scheduling and Testing]
- ServiceAppProtocol feature gate is now beta and enabled by default, adding new AppProtocol field to Services and Endpoints. ([#90023](https://github.com/kubernetes/kubernetes/pull/90023), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- SetHostnameAsFQDN is a new field in PodSpec. When set to true, the fully 
  qualified domain name (FQDN) of a Pod is set as hostname of its containers. 
  In Linux containers, this means setting the FQDN in the hostname field of the 
  kernel (the nodename field of struct utsname).  In Windows containers, this
  means setting the this means setting the registry value of hostname for the registry key HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters to FQDN. 
  If a pod does not have FQDN, this has no effect. ([#91699](https://github.com/kubernetes/kubernetes/pull/91699), [@javidiaz](https://github.com/javidiaz)) [SIG Apps, Network, Node and Testing]
- The CertificateSigningRequest API is promoted to certificates.k8s.io/v1 with the following changes:
  - `spec.signerName` is now required, and requests for `kubernetes.io/legacy-unknown` are not allowed to be created via the `certificates.k8s.io/v1` API
  - `spec.usages` is now required, may not contain duplicate values, and must only contain known usages
  - `status.conditions` may not contain duplicate types
  - `status.conditions[*].status` is now required
  - `status.certificate` must be PEM-encoded, and contain only CERTIFICATE blocks ([#91685](https://github.com/kubernetes/kubernetes/pull/91685), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Architecture, Auth, CLI and Testing]
- The HugePageStorageMediumSize feature gate is now on by default allowing usage of multiple sizes huge page resources on a container level. ([#90592](https://github.com/kubernetes/kubernetes/pull/90592), [@bart0sh](https://github.com/bart0sh)) [SIG Node]
- The Kubelet's --node-status-max-images option is now available via the Kubelet config file field nodeStatusMaxImage ([#91275](https://github.com/kubernetes/kubernetes/pull/91275), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's --seccomp-profile-root option is now marked as deprecated. ([#91182](https://github.com/kubernetes/kubernetes/pull/91182), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's `--bootstrap-checkpoint-path` option is now removed. ([#91577](https://github.com/kubernetes/kubernetes/pull/91577), [@knabben](https://github.com/knabben)) [SIG Apps and Node]
- The Kubelet's `--cloud-provider` and `--cloud-config` options are now marked as deprecated. ([#90408](https://github.com/kubernetes/kubernetes/pull/90408), [@knabben](https://github.com/knabben)) [SIG Cloud Provider and Node]
- The Kubelet's `--enable-server` and `--provider-id` option is now available via the Kubelet config file field `enableServer` and `providerID` respectively. ([#90494](https://github.com/kubernetes/kubernetes/pull/90494), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's `--kernel-memcg-notification` option is now available via the Kubelet config file field kernelMemcgNotification ([#91863](https://github.com/kubernetes/kubernetes/pull/91863), [@knabben](https://github.com/knabben)) [SIG Cloud Provider, Node and Testing]
- The Kubelet's `--really-crash-for-testing` and  `--chaos-chance` options are now marked as deprecated. ([#90499](https://github.com/kubernetes/kubernetes/pull/90499), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's `--volume-plugin-dir` option is now available via the Kubelet config file field `VolumePluginDir`. ([#88480](https://github.com/kubernetes/kubernetes/pull/88480), [@savitharaghunathan](https://github.com/savitharaghunathan)) [SIG Node]
- The `DefaultIngressClass` feature is now GA. The `--feature-gate` parameter will be removed in 1.20. ([#91957](https://github.com/kubernetes/kubernetes/pull/91957), [@cmluciano](https://github.com/cmluciano)) [SIG API Machinery, Apps, Network and Testing]
- The alpha `DynamicAuditing` feature gate and `auditregistration.k8s.io/v1alpha1` API have been removed and are no longer supported. ([#91502](https://github.com/kubernetes/kubernetes/pull/91502), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Auth and Testing]
- The kube-controller-manager managed signers can now have distinct signing certificates and keys.  See the help about `--cluster-signing-[signer-name]-{cert,key}-file`.  `--cluster-signing-{cert,key}-file` is still the default. ([#90822](https://github.com/kubernetes/kubernetes/pull/90822), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Apps and Auth]
- The unused `series.state` field, deprecated since v1.14, is removed from the `events.k8s.io/v1beta1` and `v1` Event types. ([#90449](https://github.com/kubernetes/kubernetes/pull/90449), [@wojtek-t](https://github.com/wojtek-t)) [SIG Apps]
- Unreserve extension point for scheduler plugins is merged into Reserve extension point ([#92200](https://github.com/kubernetes/kubernetes/pull/92200), [@adtac](https://github.com/adtac)) [SIG Scheduling and Testing]
- Update Golang to v1.14.4 ([#88638](https://github.com/kubernetes/kubernetes/pull/88638), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Cloud Provider, Release and Testing]
- Updated the API documentation for Service.Spec.IPFamily to warn that its exact
  semantics will probably change before the dual-stack feature goes GA, and users
  should look at ClusterIP or Endpoints, not IPFamily, to figure out if an existing
  Service is IPv4, IPv6, or dual-stack. ([#91527](https://github.com/kubernetes/kubernetes/pull/91527), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]
- Users can configure a resource prefix to ignore a group of resources. ([#88842](https://github.com/kubernetes/kubernetes/pull/88842), [@angao](https://github.com/angao)) [SIG Node and Scheduling]
- `Ingress` and `IngressClass` resources have graduated to `networking.k8s.io/v1`. Ingress and IngressClass types in the `extensions/v1beta1` and `networking.k8s.io/v1beta1` API versions are deprecated and will no longer be served in 1.22+. Persisted objects can be accessed via the `networking.k8s.io/v1` API. Notable changes in v1 Ingress objects (v1beta1 field names are unchanged):
  - `spec.backend` -> `spec.defaultBackend`
  - `serviceName` -> `service.name`
  - `servicePort` -> `service.port.name` (for string values)
  - `servicePort` -> `service.port.number` (for numeric values)
  - `pathType` no longer has a default value in v1; "Exact", "Prefix", or "ImplementationSpecific" must be specified
  Other Ingress API updates:
  - backends can now be resource or service backends
  - `path` is no longer required to be a valid regular expression ([#89778](https://github.com/kubernetes/kubernetes/pull/89778), [@cmluciano](https://github.com/cmluciano)) [SIG API Machinery, Apps, CLI, Network and Testing]
- `NodeResourcesLeastAllocated` and `NodeResourcesMostAllocated` plugins now support customized weight on the CPU and memory. ([#90544](https://github.com/kubernetes/kubernetes/pull/90544), [@chendave](https://github.com/chendave)) [SIG Scheduling]
- `PostFilter` type is added to scheduler component config API on version v1beta1. ([#91547](https://github.com/kubernetes/kubernetes/pull/91547), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- `RequestedToCapacityRatioArgs` encoding is now strict ([#91603](https://github.com/kubernetes/kubernetes/pull/91603), [@pancernik](https://github.com/pancernik)) [SIG Scheduling]
- `v1beta1` Scheduler `Extender` encoding is case-sensitive (`v1alpha1`/`v1alpha2` was case-insensitive), its `httpTimeout` field uses duration encoding (for example, one second is specified as `"1s"`), and the `enableHttps` field in `v1alpha1`/`v1alpha2` was renamed to `enableHTTPS`. ([#91625](https://github.com/kubernetes/kubernetes/pull/91625), [@pancernik](https://github.com/pancernik)) [SIG Scheduling]

### Feature

- A defaultpreemption plugin is registered and enabled in scheduler which replaces the legacy hard-coded Pod preemption logic. ([#92049](https://github.com/kubernetes/kubernetes/pull/92049), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- A new extension point `PostFilter` is introduced to scheduler framework which runs after Filter phase to resolve scheduling filter failures. A typical implementation is running preemption logic. ([#91314](https://github.com/kubernetes/kubernetes/pull/91314), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- ACTION REQUIRED : In CoreDNS v1.7.0, [metrics names have been changed](https://github.com/coredns/coredns/blob/master/notes/coredns-1.7.0.md&#35;metric-changes) which will be backward incompatible with existing reporting formulas that use the old metrics' names. Adjust your formulas to the new names before upgrading. 
  
  Kubeadm now includes CoreDNS version v1.7.0. Some of the major changes include:
  -  Fixed a bug that could cause CoreDNS to stop updating service records.
  -  Fixed a bug in the forward plugin where only the first upstream server is always selected no matter which policy is set.
  -  Remove already deprecated options `resyncperiod` and `upstream` in the Kubernetes plugin.
  -  Includes Prometheus metrics name changes (to bring them in line with standard Prometheus metrics naming convention). They will be backward incompatible with existing reporting formulas that use the old metrics' names.
  -  The federation plugin (allows for v1 Kubernetes federation) has been removed.
  More details are available in https://coredns.io/2020/06/15/coredns-1.7.0-release/ ([#92651](https://github.com/kubernetes/kubernetes/pull/92651), [@rajansandeep](https://github.com/rajansandeep)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- API requests to deprecated versions now receive a warning header in the API response, and cause a metric indicating use of a deprecated API to be published:
  - `kubectl` outputs warnings to stderr, and accepts a `--warnings-as-errors` option to treat warnings as fatal errors
  - `k8s.io/client-go` outputs warnings to stderr by default; override this per-client by setting `config.WarningHandler`, or per-process with `rest.SetDefaultWarningHandler()`
  - `kube-apiserver` publishes `apiserver_requested_deprecated_apis` gauge metrics set to `1` for deprecated APIs which have been requested, with `group`, `version`, `resource`, `subresource`, and `removed_release` labels ([#73032](https://github.com/kubernetes/kubernetes/pull/73032), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, CLI, Instrumentation and Testing]
- Add --logging-format flag for component-base. Defaults to "text" using unchanged klog. ([#89683](https://github.com/kubernetes/kubernetes/pull/89683), [@yuzhiquan](https://github.com/yuzhiquan)) [SIG Instrumentation]
- Add --port flag to kubectl create deployment ([#91113](https://github.com/kubernetes/kubernetes/pull/91113), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Add .import-restrictions file to cmd/cloud-controller-manager. ([#90630](https://github.com/kubernetes/kubernetes/pull/90630), [@nilo19](https://github.com/nilo19)) [SIG API Machinery and Cloud Provider]
- Add Annotations to CRI-API ImageSpec objects. ([#90061](https://github.com/kubernetes/kubernetes/pull/90061), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Add attempts label to scheduler's PodSchedulingDuration metric. ([#92650](https://github.com/kubernetes/kubernetes/pull/92650), [@ahg-g](https://github.com/ahg-g)) [SIG Instrumentation and Scheduling]
- Add client-side and server-side dry-run support to kubectl scale ([#89666](https://github.com/kubernetes/kubernetes/pull/89666), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Add selectors to kubectl diff ([#90857](https://github.com/kubernetes/kubernetes/pull/90857), [@sethpollack](https://github.com/sethpollack)) [SIG CLI]
- Add support for cgroups v2 node validation ([#89901](https://github.com/kubernetes/kubernetes/pull/89901), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
- Add support for pre allocated huge pages with different sizes, on node level ([#89252](https://github.com/kubernetes/kubernetes/pull/89252), [@odinuge](https://github.com/odinuge)) [SIG Apps and Node]
- Add tags support for Azure File Driver ([#92825](https://github.com/kubernetes/kubernetes/pull/92825), [@ZeroMagic](https://github.com/ZeroMagic)) [SIG Cloud Provider and Storage]
- Add tags support for azure disk driver ([#92356](https://github.com/kubernetes/kubernetes/pull/92356), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Added --privileged flag to kubectl run ([#90569](https://github.com/kubernetes/kubernetes/pull/90569), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Added a new `GetPreferredAllocation()` call to the `v1beta1` device plugin API. ([#92665](https://github.com/kubernetes/kubernetes/pull/92665), [@klueska](https://github.com/klueska)) [SIG Node and Testing]
- Added feature support to Windows for configuring session affinity of Kubernetes services.
  required: [Windows Server vNext Insider Preview Build 19551](https://blogs.windows.com/windowsexperience/2020/01/28/announcing-windows-server-vnext-insider-preview-build-19551/) (or higher) ([#91701](https://github.com/kubernetes/kubernetes/pull/91701), [@elweb9858](https://github.com/elweb9858)) [SIG Network and Windows]
- Added kube-apiserver metrics: apiserver_current_inflight_request_measures and, when API Priority and Fairness is enable, windowed_request_stats. ([#91177](https://github.com/kubernetes/kubernetes/pull/91177), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery, Instrumentation and Testing]
- Added service.beta.kubernetes.io/aws-load-balancer-target-node-labels annotation to target nodes in AWS LoadBalancer Services ([#90943](https://github.com/kubernetes/kubernetes/pull/90943), [@foobarfran](https://github.com/foobarfran)) [SIG Cloud Provider]
- Adding a set of debugging endpoints under prefix "/debug/flowcontrol/*" for dumping internal states of flow-control system with different granularity. ([#90967](https://github.com/kubernetes/kubernetes/pull/90967), [@yue9944882](https://github.com/yue9944882)) [SIG API Machinery]
- Adds profile label to kube-scheduler metric framework_extension_point_duration_seconds ([#92268](https://github.com/kubernetes/kubernetes/pull/92268), [@alculquicondor](https://github.com/alculquicondor)) [SIG Instrumentation and Scheduling]
- Adds profile label to kube-scheduler metric schedule_attempts_total
  - Adds result and profile label to e2e_scheduling_duration_seconds. Times for unschedulable and error attempts are now recorded. ([#92202](https://github.com/kubernetes/kubernetes/pull/92202), [@alculquicondor](https://github.com/alculquicondor)) [SIG Instrumentation and Scheduling]
- Audit events for API requests to deprecated API versions now include a `"k8s.io/deprecated": "true"` audit annotation. If a target removal release is identified, the audit event includes a `"k8s.io/removal-release": "<majorVersion>.<minorVersion>"` audit annotation as well. ([#92842](https://github.com/kubernetes/kubernetes/pull/92842), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Instrumentation]
- Bump Dashboard to v2.0.1 ([#91526](https://github.com/kubernetes/kubernetes/pull/91526), [@maciaszczykm](https://github.com/maciaszczykm)) [SIG Cloud Provider]
- Cloud node-controller use InstancesV2 ([#91319](https://github.com/kubernetes/kubernetes/pull/91319), [@gongguan](https://github.com/gongguan)) [SIG Apps, Cloud Provider, Scalability and Storage]
- Deps: Update to Golang 1.13.9
  - build: Remove kube-cross image building ([#89275](https://github.com/kubernetes/kubernetes/pull/89275), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Detailed scheduler scoring result can be printed at verbose level 10. ([#89384](https://github.com/kubernetes/kubernetes/pull/89384), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- E2e.test can print the list of conformance tests that need to pass for the cluster to be conformant. ([#88924](https://github.com/kubernetes/kubernetes/pull/88924), [@dims](https://github.com/dims)) [SIG Architecture and Testing]
- Enable feature Gate DefaultPodTopologySpread to use PodTopologySpread plugin to do defaultspreading. In doing so, legacy DefaultPodTopologySpread plugin is disabled. ([#91793](https://github.com/kubernetes/kubernetes/pull/91793), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- EndpointSlice controller waits longer to retry failed sync. ([#89438](https://github.com/kubernetes/kubernetes/pull/89438), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Extend AWS azToRegion method to support Local Zones ([#90874](https://github.com/kubernetes/kubernetes/pull/90874), [@Jeffwan](https://github.com/Jeffwan)) [SIG Cloud Provider]
- Feat: add azure shared disk support ([#89511](https://github.com/kubernetes/kubernetes/pull/89511), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Feat: change azure disk api-version ([#89250](https://github.com/kubernetes/kubernetes/pull/89250), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Feat: support [Azure shared disk](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/disks-shared-enable), added a new field(`maxShares`) in azure disk storage class:
  
  kind: StorageClass
  apiVersion: storage.k8s.io/v1
  metadata:
    name: shared-disk
  provisioner: kubernetes.io/azure-disk
  parameters:
    skuname: Premium_LRS  &#35; Currently only available with premium SSDs.
    cachingMode: None  &#35; ReadOnly host caching is not available for premium SSDs with maxShares>1
    maxShares: 2 ([#89328](https://github.com/kubernetes/kubernetes/pull/89328), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Improves IPVS proxy performance by only running `EnsureDummyInterface` if the virtual server address is not binded already. ([#92609](https://github.com/kubernetes/kubernetes/pull/92609), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network]
- Kube-Proxy now supports EndpointSlices on Windows with the EndpointSliceProxying feature gate. ([#90909](https://github.com/kubernetes/kubernetes/pull/90909), [@kumarvin123](https://github.com/kumarvin123)) [SIG Network and Windows]
- Kube-Proxy now supports IPv6DualStack on Windows with the IPv6DualStack feature gate. ([#90853](https://github.com/kubernetes/kubernetes/pull/90853), [@kumarvin123](https://github.com/kumarvin123)) [SIG Network, Node and Windows]
- Kube-addon-manager has been updated to v9.1.1 to allow overriding the default list of whitelisted resources (https://github.com/kubernetes/kubernetes/pull/91018) ([#91240](https://github.com/kubernetes/kubernetes/pull/91240), [@tosi3k](https://github.com/tosi3k)) [SIG Cloud Provider, Scalability and Testing]
- Kube-apiserver backed by etcd3 exports metric showing the database file size. ([#89151](https://github.com/kubernetes/kubernetes/pull/89151), [@jingyih](https://github.com/jingyih)) [SIG API Machinery]
- Kube-apiserver, kube-scheduler and kube-controller manager now use SO_REUSEPORT socket option when listening on address defined by --bind-address and --secure-port flags, when running on Unix systems (Windows is NOT supported). This allows to run multiple instances of those processes on a single host with the same configuration, which allows to update/restart them in a graceful way, without causing downtime. ([#88893](https://github.com/kubernetes/kubernetes/pull/88893), [@invidian](https://github.com/invidian)) [SIG API Machinery, Scheduling and Testing]
- Kube-apiserver: The NodeRestriction admission plugin now restricts Node labels kubelets are permitted to set when creating a new Node to the `--node-labels` parameters accepted by kubelets in 1.16+. ([#90307](https://github.com/kubernetes/kubernetes/pull/90307), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
- Kube-controller-manager: add '--logging-format' flag to support structured logging ([#91521](https://github.com/kubernetes/kubernetes/pull/91521), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery and Instrumentation]
- Kube-controller-manager: the `--experimental-cluster-signing-duration` flag is marked as deprecated for removal in v1.22, and is replaced with `--cluster-signing-duration`. ([#91154](https://github.com/kubernetes/kubernetes/pull/91154), [@liggitt](https://github.com/liggitt)) [SIG Auth and Cloud Provider]
- Kube-proxy now consumes EndpointSlices instead of Endpoints by default on Linux. A new alpha `WindowsEndpointSliceProxying` feature gate allows the feature to be enabled on Windows. ([#92736](https://github.com/kubernetes/kubernetes/pull/92736), [@robscott](https://github.com/robscott)) [SIG Network]
- Kube-scheduler: add '--logging-format' flag to support structured logging ([#91522](https://github.com/kubernetes/kubernetes/pull/91522), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery, Cluster Lifecycle, Instrumentation and Scheduling]
- Kubeadm now distinguishes between generated and user supplied component configs, regenerating the former ones if a config upgrade is required ([#86070](https://github.com/kubernetes/kubernetes/pull/86070), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: Allow manually upgraded component configs to be supplied in a YAML file via the --config option during upgrade plan & apply. The old behavior of --config in which kubeadm configuration and component configs that overwrite everything cluster stored is preserved too. The behavior to use with --config is now determined based on whether kubeadm config API objects (API group "kubeadm.kubernetes.io") were supplied in the file or not. ([#91980](https://github.com/kubernetes/kubernetes/pull/91980), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: add startup probes for static Pods to protect slow starting containers ([#91179](https://github.com/kubernetes/kubernetes/pull/91179), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: deprecate the "--csr-only" and "--csr-dir" flags of the "kubeadm init phase certs" subcommands. Please use "kubeadm alpha certs generate-csr" instead. This new command allows you to generate new private keys and certificate signing requests for all the control-plane components, so that the certificates can be signed by an external CA. ([#92183](https://github.com/kubernetes/kubernetes/pull/92183), [@wallrj](https://github.com/wallrj)) [SIG Cluster Lifecycle]
- Kubeadm: during 'upgrade apply', if the kube-proxy ConfigMap is missing, assume that kube-proxy should not be upgraded. Same applies to a missing kube-dns/coredns ConfigMap for the DNS server addon. Note that this is a temporary workaround until 'upgrade apply' supports phases. Once phases are supported the kube-proxy/dns upgrade should be skipped manually. ([#89593](https://github.com/kubernetes/kubernetes/pull/89593), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: switch control-plane static Pods to the "system-node-critical" priority class ([#90063](https://github.com/kubernetes/kubernetes/pull/90063), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: upgrade plan now prints a table indicating the state of known component configs prior to upgrade ([#88124](https://github.com/kubernetes/kubernetes/pull/88124), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubectl supports taint no without specifying(without having to type the full resource name) ([#88723](https://github.com/kubernetes/kubernetes/pull/88723), [@wawa0210](https://github.com/wawa0210)) [SIG CLI]
- Kubelet: following metrics have been renamed:
  kubelet_running_container_count --> kubelet_running_containers
  kubelet_running_pod_count --> kubelet_running_pods ([#92407](https://github.com/kubernetes/kubernetes/pull/92407), [@RainbowMango](https://github.com/RainbowMango)) [SIG API Machinery, Cluster Lifecycle, Instrumentation and Node]
- Kubelets configured to rotate client certificates now publish a `certificate_manager_server_ttl_seconds` gauge metric indicating the remaining seconds until certificate expiration. ([#91148](https://github.com/kubernetes/kubernetes/pull/91148), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
- New scoring for PodTopologySpreading that yields better spreading ([#90475](https://github.com/kubernetes/kubernetes/pull/90475), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- No actions required. This is a small enhancement to a utility library. ([#92440](https://github.com/kubernetes/kubernetes/pull/92440), [@luigibk](https://github.com/luigibk)) [SIG Network]
- PodTolerationRestriction: Mention Whitelist Scope in Error ([#87582](https://github.com/kubernetes/kubernetes/pull/87582), [@mrueg](https://github.com/mrueg)) [SIG Scheduling]
- Provider-specific Notes: vsphere: vsphere.conf - new option to disable credentials secret management for performance concerns ([#90836](https://github.com/kubernetes/kubernetes/pull/90836), [@Danil-Grigorev](https://github.com/Danil-Grigorev)) [SIG Cloud Provider]
- Rename pod_preemption_metrics to preemption_metrics. ([#93256](https://github.com/kubernetes/kubernetes/pull/93256), [@ahg-g](https://github.com/ahg-g)) [SIG Instrumentation and Scheduling]
- Rest.Config now supports a flag to override proxy configuration that was previously only configurable through environment variables. ([#81443](https://github.com/kubernetes/kubernetes/pull/81443), [@mikedanese](https://github.com/mikedanese)) [SIG API Machinery and Node]
- Scores from PodTopologySpreading have reduced differentiation as maxSkew increases. ([#90820](https://github.com/kubernetes/kubernetes/pull/90820), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Server-side apply behavior has been regularized in the case where a field is removed from the applied configuration. Removed fields which have no other owners are deleted from the live object, or reset to their default value if they have one. Safe ownership transfers, such as the transfer of a `replicas` field from a user to an HPA without resetting to the default value are documented in [Transferring Ownership](https://kubernetes.io/docs/reference/using-api/api-concepts/&#35;transferring-ownership) ([#92661](https://github.com/kubernetes/kubernetes/pull/92661), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Testing]
- Service controller: only sync LB node pools when relevant fields in Node changes ([#90769](https://github.com/kubernetes/kubernetes/pull/90769), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
- Set CSIMigrationvSphere feature gates to beta.
  Users should enable CSIMigration + CSIMigrationvSphere features and install the vSphere CSI Driver (https://github.com/kubernetes-sigs/vsphere-csi-driver) to move workload from the in-tree vSphere plugin "kubernetes.io/vsphere-volume" to vSphere CSI Driver.
  
  Requires: vSphere vCenter/ESXi Version: 7.0u1, HW Version: VM version 15 ([#92816](https://github.com/kubernetes/kubernetes/pull/92816), [@divyenpatel](https://github.com/divyenpatel)) [SIG Cloud Provider and Storage]
- Support `kubectl create deployment` with replicas ([#91562](https://github.com/kubernetes/kubernetes/pull/91562), [@zhouya0](https://github.com/zhouya0))
- Support a smooth upgrade from client-side apply to server-side apply without conflicts, as well as support the corresponding downgrade. ([#90187](https://github.com/kubernetes/kubernetes/pull/90187), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG API Machinery and Testing]
- Support create or update VMSS asynchronously. ([#89248](https://github.com/kubernetes/kubernetes/pull/89248), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Support for running on a host that uses cgroups v2 unified mode ([#85218](https://github.com/kubernetes/kubernetes/pull/85218), [@giuseppe](https://github.com/giuseppe)) [SIG Node]
- Switch core master base images (kube-apiserver, kube-scheduler) from debian to distroless ([#90674](https://github.com/kubernetes/kubernetes/pull/90674), [@dims](https://github.com/dims)) [SIG Cloud Provider, Release and Scalability]
- Switch etcd image (with migration scripts) from debian to distroless ([#91171](https://github.com/kubernetes/kubernetes/pull/91171), [@dims](https://github.com/dims)) [SIG API Machinery and Cloud Provider]
- The RotateKubeletClientCertificate feature gate has been promoted to GA, and the kubelet --feature-gate RotateKubeletClientCertificate parameter will be removed in 1.20. ([#91780](https://github.com/kubernetes/kubernetes/pull/91780), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
- The SCTPSupport feature is now active by default. ([#88932](https://github.com/kubernetes/kubernetes/pull/88932), [@janosi](https://github.com/janosi)) [SIG Network]
- The `certificatesigningrequests/approval` subresource now supports patch API requests ([#91558](https://github.com/kubernetes/kubernetes/pull/91558), [@liggitt](https://github.com/liggitt)) [SIG Auth and Testing]
- The metric label name of `kubernetes_build_info` has been updated from `camel case` to `snake case`:
  - gitVersion --> git_version
  - gitCommit --> git_commit
  - gitTreeState --> git_tree_state
  - buildDate --> build_date
  - goVersion --> go_version
  
  This change happens in `kube-apiserver`、`kube-scheduler`、`kube-proxy` and `kube-controller-manager`. ([#91805](https://github.com/kubernetes/kubernetes/pull/91805), [@RainbowMango](https://github.com/RainbowMango)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Trace output in apiserver logs is more organized and comprehensive. Traces are nested, and for all non-long running request endpoints, the entire filter chain is instrumented (e.g. authentication check is included). ([#88936](https://github.com/kubernetes/kubernetes/pull/88936), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Scheduling]
- Try to send watch bookmarks (if requested) periodically in addition to sending them right before timeout ([#90560](https://github.com/kubernetes/kubernetes/pull/90560), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Update cri-tools to v1.18.0 ([#89720](https://github.com/kubernetes/kubernetes/pull/89720), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Cluster Lifecycle, Release and Scalability]
- Update etcd client side to v3.4.4 ([#89169](https://github.com/kubernetes/kubernetes/pull/89169), [@jingyih](https://github.com/jingyih)) [SIG API Machinery and Cloud Provider]
- Update etcd client side to v3.4.7 ([#89822](https://github.com/kubernetes/kubernetes/pull/89822), [@jingyih](https://github.com/jingyih)) [SIG API Machinery and Cloud Provider]
- Update etcd client side to v3.4.9 ([#92075](https://github.com/kubernetes/kubernetes/pull/92075), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cloud Provider and Instrumentation]
- Upgrade to azure-sdk v40.2.0 ([#89105](https://github.com/kubernetes/kubernetes/pull/89105), [@andyzhangx](https://github.com/andyzhangx)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- Warn users that `kubectl port-forward` does not support UDP now ([#91616](https://github.com/kubernetes/kubernetes/pull/91616), [@knight42](https://github.com/knight42)) [SIG CLI]
- Weight of PodTopologySpread scheduling Score is doubled. ([#91258](https://github.com/kubernetes/kubernetes/pull/91258), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- `EventRecorder()` is exposed to `FrameworkHandle` interface so that scheduler plugin developers can choose to log cluster-level events. ([#92010](https://github.com/kubernetes/kubernetes/pull/92010), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- `kubectl alpha debug` command now supports debugging pods by copy the original one. ([#90094](https://github.com/kubernetes/kubernetes/pull/90094), [@aylei](https://github.com/aylei)) [SIG CLI]
- `kubectl alpha debug` now supports debugging nodes by creating a debugging container running in the node's host namespaces. ([#92310](https://github.com/kubernetes/kubernetes/pull/92310), [@verb](https://github.com/verb)) [SIG CLI]
- `local-up-cluster.sh` installs CSI snapshotter by default now, can be disabled with `ENABLE_CSI_SNAPSHOTTER=false`. ([#91504](https://github.com/kubernetes/kubernetes/pull/91504), [@pohly](https://github.com/pohly))
- `maxThreshold` of `ImageLocality` plugin is now scaled by the number of images in the pod, which helps to distinguish the node priorities for pod with several images. ([#91138](https://github.com/kubernetes/kubernetes/pull/91138), [@chendave](https://github.com/chendave)) [SIG Scheduling]

### Documentation

- Updated the instructions for deploying the sample app. ([#82785](https://github.com/kubernetes/kubernetes/pull/82785), [@ashish-billore](https://github.com/ashish-billore)) [SIG API Machinery]

### Failing Test

- Kube-proxy iptables min-sync-period defaults to 1 sec. Previously, it was 0. ([#92836](https://github.com/kubernetes/kubernetes/pull/92836), [@aojea](https://github.com/aojea)) [SIG Network]

### Bug or Regression

- A PV set from in-tree source will have ordered requirement values in NodeAffinity when converted to CSIPersistentVolumeSource ([#88987](https://github.com/kubernetes/kubernetes/pull/88987), [@jiahuif](https://github.com/jiahuif)) [SIG Storage]
- A panic in the apiserver caused by the `informer-sync` health checker is now fixed. ([#93600](https://github.com/kubernetes/kubernetes/pull/93600), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- An issue preventing GCP cloud-controller-manager running out-of-cluster to initialize new Nodes is now fixed. ([#90057](https://github.com/kubernetes/kubernetes/pull/90057), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Apps and Cloud Provider]
- Avoid GCE API calls when initializing GCE CloudProvider for Kubelets. ([#90218](https://github.com/kubernetes/kubernetes/pull/90218), [@wojtek-t](https://github.com/wojtek-t)) [SIG Cloud Provider and Scalability]
- Avoid unnecessary GCE API calls when adding IP alises or reflecting them in Node object in GCE cloud provider. ([#90242](https://github.com/kubernetes/kubernetes/pull/90242), [@wojtek-t](https://github.com/wojtek-t)) [SIG Apps, Cloud Provider and Network]
- Avoid unnecessary scheduling churn when annotations are updated while Pods are being scheduled. ([#90373](https://github.com/kubernetes/kubernetes/pull/90373), [@fabiokung](https://github.com/fabiokung)) [SIG Scheduling]
- Azure auth module for kubectl now requests login after refresh token expires. ([#86481](https://github.com/kubernetes/kubernetes/pull/86481), [@tdihp](https://github.com/tdihp)) [SIG API Machinery and Auth]
- Azure: fix concurreny issue in lb creation ([#89604](https://github.com/kubernetes/kubernetes/pull/89604), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Azure: per VMSS VMSS VMs cache to prevent throttling on clusters having many attached VMSS ([#93107](https://github.com/kubernetes/kubernetes/pull/93107), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- Azure: set dest prefix and port for IPv6 inbound security rule ([#91831](https://github.com/kubernetes/kubernetes/pull/91831), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Base-images: Update to kube-cross:v1.13.9-5 ([#90963](https://github.com/kubernetes/kubernetes/pull/90963), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Bug fix for AWS NLB service when nodePort for existing servicePort changed manually. ([#89562](https://github.com/kubernetes/kubernetes/pull/89562), [@M00nF1sh](https://github.com/M00nF1sh)) [SIG Cloud Provider]
- CSINode initialization does not crash kubelet on startup when APIServer is not reachable or kubelet has not the right credentials yet. ([#89589](https://github.com/kubernetes/kubernetes/pull/89589), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- CVE-2020-8557 (Medium): Node-local denial of service via container /etc/hosts file. See https://github.com/kubernetes/kubernetes/issues/93032 for more details. ([#92916](https://github.com/kubernetes/kubernetes/pull/92916), [@joelsmith](https://github.com/joelsmith)) [SIG Node]
- Client-go: resolves an issue with informers falling back to full list requests when timeouts are encountered, rather than re-establishing a watch. ([#89652](https://github.com/kubernetes/kubernetes/pull/89652), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- CloudNodeLifecycleController will check node existence status before shutdown status when monitoring nodes. ([#90737](https://github.com/kubernetes/kubernetes/pull/90737), [@jiahuif](https://github.com/jiahuif)) [SIG Apps and Cloud Provider]
- Containers which specify a `startupProbe` but not a `readinessProbe` were previously considered "ready" before the `startupProbe` completed, but are now considered "not-ready". ([#92196](https://github.com/kubernetes/kubernetes/pull/92196), [@thockin](https://github.com/thockin)) [SIG Node]
- Cordoned nodes are now deregistered from AWS target groups. ([#85920](https://github.com/kubernetes/kubernetes/pull/85920), [@hoelzro](https://github.com/hoelzro)) [SIG Cloud Provider]
- Do not add nodes labeled with kubernetes.azure.com/managed=false to backend pool of load balancer. ([#93034](https://github.com/kubernetes/kubernetes/pull/93034), [@matthias50](https://github.com/matthias50)) [SIG Cloud Provider]
- Do not retry volume expansion if CSI driver returns FailedPrecondition error ([#92986](https://github.com/kubernetes/kubernetes/pull/92986), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Dockershim security: pod sandbox now always run with `no-new-privileges` and `runtime/default` seccomp profile
  dockershim seccomp: custom profiles can now have smaller seccomp profiles when set at pod level ([#90948](https://github.com/kubernetes/kubernetes/pull/90948), [@pjbgf](https://github.com/pjbgf)) [SIG Node]
- Dual-stack: fix the bug that Service clusterIP does not respect specified ipFamily ([#89612](https://github.com/kubernetes/kubernetes/pull/89612), [@SataQiu](https://github.com/SataQiu)) [SIG Network]
- EndpointSliceMirroring controller now copies labels from Endpoints to EndpointSlices. ([#93442](https://github.com/kubernetes/kubernetes/pull/93442), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Ensure Azure availability zone is always in lower cases. ([#89722](https://github.com/kubernetes/kubernetes/pull/89722), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Eviction requests for pods that have a non-zero DeletionTimestamp will always succeed ([#91342](https://github.com/kubernetes/kubernetes/pull/91342), [@michaelgugino](https://github.com/michaelgugino)) [SIG Apps]
- Explain CRDs whose resource name are the same as builtin objects ([#89505](https://github.com/kubernetes/kubernetes/pull/89505), [@knight42](https://github.com/knight42)) [SIG API Machinery, CLI and Testing]
- Extend kube-apiserver /readyz with new "informer-sync" check ensuring that internal informers are synced. ([#92644](https://github.com/kubernetes/kubernetes/pull/92644), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Testing]
- Extended DSR loadbalancer feature in winkernel kube-proxy to HNS versions 9.3-9.max, 10.2+ ([#93080](https://github.com/kubernetes/kubernetes/pull/93080), [@elweb9858](https://github.com/elweb9858)) [SIG Network]
- First pod with required affinity terms can schedule only on nodes with matching topology keys. ([#91168](https://github.com/kubernetes/kubernetes/pull/91168), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]
- Fix AWS Loadbalancer VPC CIDR calculation when CIDR in disassociated state exists. ([#92227](https://github.com/kubernetes/kubernetes/pull/92227), [@M00nF1sh](https://github.com/M00nF1sh)) [SIG Cloud Provider]
- Fix InstanceMetadataByProviderID for unmanaged nodes ([#92572](https://github.com/kubernetes/kubernetes/pull/92572), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix `VirtualMachineScaleSets.virtualMachines.GET` not allowed issues when customers have set VMSS orchestrationMode. ([#91097](https://github.com/kubernetes/kubernetes/pull/91097), [@feiskyer](https://github.com/feiskyer))
- Fix a bug that didn't allow to use IPv6 addresses with leading zeros ([#89341](https://github.com/kubernetes/kubernetes/pull/89341), [@aojea](https://github.com/aojea)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Fix a bug where ExternalTrafficPolicy is not applied to service ExternalIPs. ([#90537](https://github.com/kubernetes/kubernetes/pull/90537), [@freehan](https://github.com/freehan)) [SIG Network]
- Fix a condition when expiring nil VM entry in VMSS cache ([#92681](https://github.com/kubernetes/kubernetes/pull/92681), [@ArchangelSDY](https://github.com/ArchangelSDY)) [SIG Cloud Provider]
- Fix a racing issue where the scheduler may perform unnecessary scheduling attempts. ([#90660](https://github.com/kubernetes/kubernetes/pull/90660), [@Huang-Wei](https://github.com/Huang-Wei))
- Fix an issue with container restarts using a modified configmap or secret subpath volume mount. ([#89629](https://github.com/kubernetes/kubernetes/pull/89629), [@fatedier](https://github.com/fatedier)) [SIG Architecture, Storage and Testing]
- Fix bug in the port allocation logic that caused that the NodePort creation with statically assigned portNumber collide in multi-master HA cluster ([#89937](https://github.com/kubernetes/kubernetes/pull/89937), [@aojea](https://github.com/aojea)) [SIG Network and Testing]
- Fix bug with xfs_repair from stopping xfs mount ([#89444](https://github.com/kubernetes/kubernetes/pull/89444), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Fix clusterdump info namespaces flag not working ([#91890](https://github.com/kubernetes/kubernetes/pull/91890), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix detection of SystemOOMs in which the victim is a container. ([#88871](https://github.com/kubernetes/kubernetes/pull/88871), [@dashpole](https://github.com/dashpole)) [SIG Node]
- Fix detection of image filesystem, disk metrics for devicemapper, detection of OOM Kills on 5.0+ linux kernels. ([#92919](https://github.com/kubernetes/kubernetes/pull/92919), [@dashpole](https://github.com/dashpole)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Node]
- Fix etcd version migration script in etcd image. ([#91925](https://github.com/kubernetes/kubernetes/pull/91925), [@wenjiaswe](https://github.com/wenjiaswe)) [SIG API Machinery]
- Fix flaws in Azure File CSI translation ([#90162](https://github.com/kubernetes/kubernetes/pull/90162), [@rfranzke](https://github.com/rfranzke)) [SIG Release and Storage]
- Fix instance not found issues when an Azure Node is recreated in a short time ([#93316](https://github.com/kubernetes/kubernetes/pull/93316), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix issues when supported huge page sizes changes ([#80831](https://github.com/kubernetes/kubernetes/pull/80831), [@odinuge](https://github.com/odinuge)) [SIG Node and Testing]
- Fix kube-apiserver startup to wait for APIServices to be installed into the HTTP handler before reporting readiness. ([#89147](https://github.com/kubernetes/kubernetes/pull/89147), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Fix kubectl create --dryrun client ignores namespace ([#90502](https://github.com/kubernetes/kubernetes/pull/90502), [@zhouya0](https://github.com/zhouya0))
- Fix kubectl create secret docker-registry --from-file not usable ([#90960](https://github.com/kubernetes/kubernetes/pull/90960), [@zhouya0](https://github.com/zhouya0)) [SIG CLI and Testing]
- Fix kubectl describe CSINode nil pointer error ([#89646](https://github.com/kubernetes/kubernetes/pull/89646), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix kubectl describe node for users not having access to lease information. ([#90469](https://github.com/kubernetes/kubernetes/pull/90469), [@uthark](https://github.com/uthark)) [SIG CLI]
- Fix kubectl describe output format for empty annotations. ([#91405](https://github.com/kubernetes/kubernetes/pull/91405), [@iyashu](https://github.com/iyashu)) [SIG CLI]
- Fix kubectl diff so it doesn't actually persist patches ([#89795](https://github.com/kubernetes/kubernetes/pull/89795), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Fix kubectl run  --dry-run client  ignore namespace ([#90785](https://github.com/kubernetes/kubernetes/pull/90785), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix kubectl version should print version info without config file ([#89913](https://github.com/kubernetes/kubernetes/pull/89913), [@zhouya0](https://github.com/zhouya0)) [SIG API Machinery and CLI]
- Fix missing `-c` shorthand for `--container` flag of `kubectl alpha debug` ([#89674](https://github.com/kubernetes/kubernetes/pull/89674), [@superbrothers](https://github.com/superbrothers)) [SIG CLI]
- Fix printers ignoring object average value ([#89142](https://github.com/kubernetes/kubernetes/pull/89142), [@zhouya0](https://github.com/zhouya0)) [SIG API Machinery]
- Fix public IP not shown issues after assigning public IP to Azure VMs ([#90886](https://github.com/kubernetes/kubernetes/pull/90886), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix scheduler crash when removing node before its pods ([#89908](https://github.com/kubernetes/kubernetes/pull/89908), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Fix the VMSS name and resource group name when updating Azure VMSS for LoadBalancer backendPools ([#89337](https://github.com/kubernetes/kubernetes/pull/89337), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix throttling issues when Azure VM computer name prefix is different from VMSS name ([#92793](https://github.com/kubernetes/kubernetes/pull/92793), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix: Azure deallocating node should be regarded as shut down ([#92257](https://github.com/kubernetes/kubernetes/pull/92257), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: GetLabelsForVolume panic issue for azure disk PV ([#92166](https://github.com/kubernetes/kubernetes/pull/92166), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: add azure file migration support on annotation support ([#91093](https://github.com/kubernetes/kubernetes/pull/91093), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Node]
- Fix: azure disk dangling attach issue which would cause API throttling ([#90749](https://github.com/kubernetes/kubernetes/pull/90749), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: determine the correct ip config based on ip family ([#93043](https://github.com/kubernetes/kubernetes/pull/93043), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Fix: don't use docker config cache if it's empty ([#92330](https://github.com/kubernetes/kubernetes/pull/92330), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: fix topology issue in azure disk storage class migration ([#91196](https://github.com/kubernetes/kubernetes/pull/91196), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: get attach disk error due to missing item in max count table ([#89768](https://github.com/kubernetes/kubernetes/pull/89768), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: incorrect max azure disk max count ([#92331](https://github.com/kubernetes/kubernetes/pull/92331), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: initial delay in mounting azure disk & file ([#93052](https://github.com/kubernetes/kubernetes/pull/93052), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: support removal of nodes backed by deleted non VMSS instances on Azure ([#91184](https://github.com/kubernetes/kubernetes/pull/91184), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- Fix: use force detach for azure disk ([#91948](https://github.com/kubernetes/kubernetes/pull/91948), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fixed a 1.18 regression in wait.Forever that skips the backoff period on the first repeat ([#90476](https://github.com/kubernetes/kubernetes/pull/90476), [@zhan849](https://github.com/zhan849)) [SIG API Machinery]
- Fixed a bug that mistake use newObj as oldObj in endpoint slice update. ([#92339](https://github.com/kubernetes/kubernetes/pull/92339), [@fatkun](https://github.com/fatkun)) [SIG Apps and Network]
- Fixed a bug where executing a kubectl command with a jsonpath output expression that has a nested range would ignore expressions following the nested range. ([#88464](https://github.com/kubernetes/kubernetes/pull/88464), [@brianpursley](https://github.com/brianpursley)) [SIG API Machinery]
- Fixed a bug whereby the allocation of reusable CPUs and devices was not being honored when the TopologyManager was enabled ([#93189](https://github.com/kubernetes/kubernetes/pull/93189), [@klueska](https://github.com/klueska)) [SIG Node]
- Fixed a performance issue applying json patches to deeply nested objects ([#92069](https://github.com/kubernetes/kubernetes/pull/92069), [@tapih](https://github.com/tapih)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Fixed a regression preventing garbage collection of RBAC role and binding objects ([#90534](https://github.com/kubernetes/kubernetes/pull/90534), [@apelisse](https://github.com/apelisse)) [SIG Auth]
- Fixed a regression running kubectl commands with  --local or --dry-run flags when no kubeconfig file is present ([#90243](https://github.com/kubernetes/kubernetes/pull/90243), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI and Testing]
- Fixed ambiguous behavior when bearer token (kubectl --token=..) and an exec credential plugin was configured in the same context - the bearer token now takes precedence. ([#91745](https://github.com/kubernetes/kubernetes/pull/91745), [@anderseknert](https://github.com/anderseknert)) [SIG API Machinery, Auth and Testing]
- Fixed an issue mounting credentials for service accounts whose name contains `.` characters ([#89696](https://github.com/kubernetes/kubernetes/pull/89696), [@nabokihms](https://github.com/nabokihms)) [SIG Auth]
- Fixed an issue that a Pod's nominatedNodeName cannot be cleared upon node deletion. ([#91750](https://github.com/kubernetes/kubernetes/pull/91750), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fixed bug where a nonzero exit code was returned when initializing zsh completion even though zsh completion was successfully initialized ([#88165](https://github.com/kubernetes/kubernetes/pull/88165), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Fixed memory leak in endpointSliceTracker ([#92838](https://github.com/kubernetes/kubernetes/pull/92838), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Fixed mountOptions in iSCSI and FibreChannel volume plugins. ([#89172](https://github.com/kubernetes/kubernetes/pull/89172), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixed node data lost in kube-scheduler for clusters with imbalance on number of nodes across zones ([#93355](https://github.com/kubernetes/kubernetes/pull/93355), [@maelk](https://github.com/maelk))
- Fixed several bugs involving the IPFamily field when creating or updating services
  in clusters with the IPv6DualStack feature gate enabled.
  
  Beware that the behavior of the IPFamily field is strange and inconsistent and will
  likely be changed before the dual-stack feature goes GA. Users should treat the
  field as "write-only" for now and should not make any assumptions about a service
  based on its current IPFamily value. ([#91400](https://github.com/kubernetes/kubernetes/pull/91400), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]
- Fixed the EndpointSlice controller to run without error on a cluster with the OwnerReferencesPermissionEnforcement validating admission plugin enabled. ([#89741](https://github.com/kubernetes/kubernetes/pull/89741), [@marun](https://github.com/marun)) [SIG Auth and Network]
- Fixed the EndpointSliceController to correctly create endpoints for IPv6-only pods.
  
  Fixed the EndpointController to allow IPv6 headless services, if the IPv6DualStack
  feature gate is enabled, by specifying `ipFamily: IPv6` on the service. (This already
  worked with the EndpointSliceController.) ([#91399](https://github.com/kubernetes/kubernetes/pull/91399), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]
- Fixed using of a read-only iSCSI volume in multiple pods. ([#91738](https://github.com/kubernetes/kubernetes/pull/91738), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]
- Fixes CSI volume attachment scaling issue by using informers. ([#91307](https://github.com/kubernetes/kubernetes/pull/91307), [@yuga711](https://github.com/yuga711)) [SIG API Machinery, Apps, Node, Storage and Testing]
- Fixes a bug defining a default value for a replicas field in a custom resource definition that has the scale subresource enabled ([#89833](https://github.com/kubernetes/kubernetes/pull/89833), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Fixes a bug that non directory hostpath type can be recognized as HostPathFile and adds e2e tests for HostPathType ([#64829](https://github.com/kubernetes/kubernetes/pull/64829), [@dixudx](https://github.com/dixudx)) [SIG Apps, Storage and Testing]
- Fixes a problem with 63-second or 1-second connection delays with some VXLAN-based
  network plugins which was first widely noticed in 1.16 (though some users saw it
  earlier than that, possibly only with specific network plugins). If you were previously
  using ethtool to disable checksum offload on your primary network interface, you should
  now be able to stop doing that. ([#92035](https://github.com/kubernetes/kubernetes/pull/92035), [@danwinship](https://github.com/danwinship)) [SIG Network and Node]
- Fixes a regression in 1.17 that dropped cache-control headers on API requests ([#90468](https://github.com/kubernetes/kubernetes/pull/90468), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Fixes conversion error for HorizontalPodAutoscaler objects with invalid annotations ([#89963](https://github.com/kubernetes/kubernetes/pull/89963), [@liggitt](https://github.com/liggitt)) [SIG Autoscaling]
- Fixes kubectl to apply all validly built objects, instead of stopping on error. ([#89848](https://github.com/kubernetes/kubernetes/pull/89848), [@seans3](https://github.com/seans3)) [SIG CLI and Testing]
- Fixes regression in CPUManager that caused freeing of exclusive CPUs at incorrect times ([#90377](https://github.com/kubernetes/kubernetes/pull/90377), [@cbf123](https://github.com/cbf123)) [SIG Cloud Provider and Node]
- Fixes regression in CPUManager that had the (rare) possibility to release exclusive CPUs in app containers inherited from init containers. ([#90419](https://github.com/kubernetes/kubernetes/pull/90419), [@klueska](https://github.com/klueska)) [SIG Node]
- Fixes v1.18.0-rc.1 regression in `kubectl port-forward` when specifying a local and remote port ([#89401](https://github.com/kubernetes/kubernetes/pull/89401), [@liggitt](https://github.com/liggitt))
- Fixing race condition with EndpointSlice controller garbage collection. ([#91311](https://github.com/kubernetes/kubernetes/pull/91311), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- For GCE cluster provider, fix bug of not being able to create internal type load balancer for clusters with more than 1000 nodes in a single zone. ([#89902](https://github.com/kubernetes/kubernetes/pull/89902), [@wojtek-t](https://github.com/wojtek-t)) [SIG Cloud Provider, Network and Scalability]
- For external storage e2e test suite, update external driver, to pick snapshot provisioner from VolumeSnapshotClass, when a VolumeSnapshotClass is explicitly provided as an input. ([#90878](https://github.com/kubernetes/kubernetes/pull/90878), [@saikat-royc](https://github.com/saikat-royc)) [SIG Storage and Testing]
- Get-kube.sh: fix order to get the binaries from the right bucket ([#91635](https://github.com/kubernetes/kubernetes/pull/91635), [@cpanato](https://github.com/cpanato)) [SIG Release]
- If firstTimestamp is not set use eventTime when printing event ([#89999](https://github.com/kubernetes/kubernetes/pull/89999), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- If we set parameter cgroupPerQos=false and cgroupRoot=/docker，this function will retrun  nodeAllocatableRoot=/docker/kubepods, it is not right, the correct return should be /docker.
  cm.NodeAllocatableRoot(s.CgroupRoot, s.CgroupDriver)
  
  kubeDeps.CAdvisorInterface, err = cadvisor.New(imageFsInfoProvider, s.RootDirectory, cgroupRoots, cadvisor.UsingLegacyCadvisorStats(s.ContainerRuntime, s.RemoteRuntimeEndpoint))
  the above funtion，as we use cgroupRoots to create cadvisor interface，the wrong parameter cgroupRoots will lead eviction manager not  to collect metric from /docker, then kubelet frequently print those error：
  E0303 17:25:03.436781 63839 summary_sys_containers.go:47] Failed to get system container stats for "/docker": failed to get cgroup stats for "/docker": failed to get container info for "/docker": unknown container "/docker"
  E0303 17:25:03.436809 63839 helpers.go:680] eviction manager: failed to construct signal: "allocatableMemory.available" error: system container "pods" not found in metrics ([#88970](https://github.com/kubernetes/kubernetes/pull/88970), [@mysunshine92](https://github.com/mysunshine92)) [SIG Node]
- In a HA env, during the period a standby scheduler lost connection to API server, if a Pod is deleted and recreated, and the standby scheduler becomes master afterwards, there could be a scheduler cache corruption. This PR fixes this issue. ([#91126](https://github.com/kubernetes/kubernetes/pull/91126), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- In the kubelet resource metrics endpoint at /metrics/resource, change the names of the following metrics:
  - node_cpu_usage_seconds --> node_cpu_usage_seconds_total
  - container_cpu_usage_seconds --> container_cpu_usage_seconds_total
  This is a partial revert of &#35;86282, which was added in 1.18.0, and initially removed the _total suffix ([#89540](https://github.com/kubernetes/kubernetes/pull/89540), [@dashpole](https://github.com/dashpole)) [SIG Instrumentation and Node]
- Ipvs: only attempt setting of sysctlconnreuse on supported kernels ([#88541](https://github.com/kubernetes/kubernetes/pull/88541), [@cmluciano](https://github.com/cmluciano)) [SIG Network]
- Jsonpath support in kubectl / client-go serializes complex types (maps / slices / structs) as json instead of Go-syntax. ([#89660](https://github.com/kubernetes/kubernetes/pull/89660), [@pjferrell](https://github.com/pjferrell)) [SIG API Machinery, CLI and Cluster Lifecycle]
- Kube-aggregator certificates are dynamically loaded on change from disk ([#92791](https://github.com/kubernetes/kubernetes/pull/92791), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kube-apiserver: fixes scale subresource patch handling to avoid returning unnecessary 409 Conflict error to clients ([#90342](https://github.com/kubernetes/kubernetes/pull/90342), [@liggitt](https://github.com/liggitt)) [SIG Apps, Autoscaling and Testing]
- Kube-apiserver: jsonpath expressions with consecutive recursive descent operators are no longer evaluated for custom resource printer columns ([#93408](https://github.com/kubernetes/kubernetes/pull/93408), [@joelsmith](https://github.com/joelsmith)) [SIG API Machinery]
- Kube-apiserver: multiple comma-separated protocols in a single X-Stream-Protocol-Version header are now recognized, in addition to multiple headers, complying with RFC2616 ([#89857](https://github.com/kubernetes/kubernetes/pull/89857), [@tedyu](https://github.com/tedyu)) [SIG API Machinery]
- Kube-proxy IP family will be determined by the nodeIP used by the proxier. The order of precedence is:
  1. the configured --bind-address if the bind address is not 0.0.0.0 or ::
  2. the primary IP from the Node object, if set.
  3. if no IP is found, NodeIP defaults to 127.0.0.1 and the IP family to IPv4 ([#91725](https://github.com/kubernetes/kubernetes/pull/91725), [@aojea](https://github.com/aojea)) [SIG Network]
- Kube-proxy, in dual-stack mode, infers the service IP family from the ClusterIP instead of using the `Service.Spec.IPFamily` field ([#91357](https://github.com/kubernetes/kubernetes/pull/91357), [@aojea](https://github.com/aojea))
- Kube-up now includes CoreDNS version v1.7.0. Some of the major changes include:
  -  Fixed a bug that could cause CoreDNS to stop updating service records.
  -  Fixed a bug in the forward plugin where only the first upstream server is always selected no matter which policy is set.
  -  Remove already deprecated options `resyncperiod` and `upstream` in the Kubernetes plugin.
  -  Includes Prometheus metrics name changes (to bring them in line with standard Prometheus metrics naming convention). They will be backward incompatible with existing reporting formulas that use the old metrics' names.
  -  The federation plugin (allows for v1 Kubernetes federation) has been removed.
  More details are available in https://coredns.io/2020/06/15/coredns-1.7.0-release/ ([#92718](https://github.com/kubernetes/kubernetes/pull/92718), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cloud Provider]
- Kube-up: fixes setup of validating admission webhook credential configuration ([#91995](https://github.com/kubernetes/kubernetes/pull/91995), [@liggitt](https://github.com/liggitt)) [SIG Cloud Provider and Cluster Lifecycle]
- Kubeadm increased to 5 minutes its timeout for the TLS bootstrapping process to complete upon join ([#89735](https://github.com/kubernetes/kubernetes/pull/89735), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: Add retries for kubeadm join / UpdateStatus to make update status more resilient by adding a retry loop to this operation ([#91952](https://github.com/kubernetes/kubernetes/pull/91952), [@xlgao-zju](https://github.com/xlgao-zju)) [SIG Cluster Lifecycle]
- Kubeadm: add the deprecated flag --port=0 to kube-controller-manager and kube-scheduler manifests to disable insecure serving. Without this flag the components by default serve (e.g. /metrics) insecurely on the default node interface (controlled by --address). Users that wish to override this behavior and enable insecure serving can pass a custom --port=X via kubeadm's "extraArgs" mechanic for these components. ([#92720](https://github.com/kubernetes/kubernetes/pull/92720), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: during "join", don't re-add an etcd member if it already exists in the cluster. ([#92118](https://github.com/kubernetes/kubernetes/pull/92118), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: during "reset" do not remove the only remaining stacked etcd member from the cluster and just proceed with the cleanup of the local etcd storage. ([#91145](https://github.com/kubernetes/kubernetes/pull/91145), [@tnqn](https://github.com/tnqn)) [SIG Cluster Lifecycle]
- Kubeadm: during join when a check is performed that a Node with the same name already exists in the cluster, make sure the NodeReady condition is properly validated ([#89602](https://github.com/kubernetes/kubernetes/pull/89602), [@kvaps](https://github.com/kvaps)) [SIG Cluster Lifecycle]
- Kubeadm: ensure `image-pull-timeout` flag is respected during upgrade phase ([#90328](https://github.com/kubernetes/kubernetes/pull/90328), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where post upgrade to 1.18.x, nodes cannot join the cluster due to missing RBAC ([#89537](https://github.com/kubernetes/kubernetes/pull/89537), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix misleading warning about passing control-plane related flags on 'kubeadm join' ([#89596](https://github.com/kubernetes/kubernetes/pull/89596), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: increase robustness for "kubeadm join" when adding etcd members on slower setups ([#90645](https://github.com/kubernetes/kubernetes/pull/90645), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove duplicate DNS names and IP addresses from generated certificates ([#92753](https://github.com/kubernetes/kubernetes/pull/92753), [@QianChenglong](https://github.com/QianChenglong)) [SIG Cluster Lifecycle]
- Kubectl azure authentication: fixed a regression in 1.18.0 where "spn:" prefix was unexpectedly added to the `apiserver-id` configuration in the kubeconfig file ([#89706](https://github.com/kubernetes/kubernetes/pull/89706), [@weinong](https://github.com/weinong)) [SIG API Machinery and Auth]
- Kubectl: fix the bug that kubectl autoscale does not honor '--name' flag ([#91855](https://github.com/kubernetes/kubernetes/pull/91855), [@SataQiu](https://github.com/SataQiu)) [SIG CLI]
- Kubectl: fix the bug that kubectl scale does not honor '--timeout' flag ([#91858](https://github.com/kubernetes/kubernetes/pull/91858), [@SataQiu](https://github.com/SataQiu)) [SIG CLI]
- Kubelet: fix the bug that kubelet help information can not show the right type of flags ([#88515](https://github.com/kubernetes/kubernetes/pull/88515), [@SataQiu](https://github.com/SataQiu)) [SIG Docs and Node]
- Kuberuntime security: pod sandbox now always runs with `runtime/default` seccomp profile
  kuberuntime seccomp: custom profiles can now have smaller seccomp profiles when set at pod level ([#90949](https://github.com/kubernetes/kubernetes/pull/90949), [@pjbgf](https://github.com/pjbgf)) [SIG Node]
- Make Kubelet bootstrap certificate signal aware ([#92786](https://github.com/kubernetes/kubernetes/pull/92786), [@answer1991](https://github.com/answer1991)) [SIG API Machinery, Auth and Node]
- Node ([#89677](https://github.com/kubernetes/kubernetes/pull/89677), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- On AWS nodes with multiple network interfaces, kubelet should now more reliably report addresses from secondary interfaces. ([#91889](https://github.com/kubernetes/kubernetes/pull/91889), [@anguslees](https://github.com/anguslees)) [SIG Cloud Provider]
- Pod Conditions updates are skipped for re-scheduling attempts ([#91252](https://github.com/kubernetes/kubernetes/pull/91252), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Pods can now be considered for preemption after a previously nominated node has become unschedulable and unresolvable. ([#92604](https://github.com/kubernetes/kubernetes/pull/92604), [@soulxu](https://github.com/soulxu))
- Prevent PVC requested size overflow when expanding or creating a volume ([#90907](https://github.com/kubernetes/kubernetes/pull/90907), [@gnufied](https://github.com/gnufied)) [SIG Cloud Provider and Storage]
- Provides a fix to allow a cluster in a private Azure cloud to authenticate to ACR in the same cloud. ([#90425](https://github.com/kubernetes/kubernetes/pull/90425), [@DavidParks8](https://github.com/DavidParks8)) [SIG Cloud Provider]
- Refine aws loadbalancer worker node SG rule logic to be deterministic ([#92224](https://github.com/kubernetes/kubernetes/pull/92224), [@M00nF1sh](https://github.com/M00nF1sh)) [SIG Cloud Provider]
- Resolve regression in metadata.managedFields handling in create/update/patch requests not using server-side apply ([#91690](https://github.com/kubernetes/kubernetes/pull/91690), [@apelisse](https://github.com/apelisse)) [SIG API Machinery and Testing]
- Resolved a regression in v1.18.0-rc.1 mounting windows volumes ([#89319](https://github.com/kubernetes/kubernetes/pull/89319), [@mboersma](https://github.com/mboersma)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Resolves an issue using `kubectl certificate approve/deny` against a server serving the v1 CSR API ([#91691](https://github.com/kubernetes/kubernetes/pull/91691), [@liggitt](https://github.com/liggitt)) [SIG Auth and CLI]
- Restore the ability to `kubectl apply --prune` without --namespace flag.  Since 1.17, `kubectl apply --prune` only prunes resources in the default namespace (or from kubeconfig) or explicitly specified in command line flag.  But this is s breaking change from kubectl 1.16, which can prune resources in all namespace in config file.  This patch restores the kubectl 1.16 behaviour. ([#89551](https://github.com/kubernetes/kubernetes/pull/89551), [@tatsuhiro-t](https://github.com/tatsuhiro-t)) [SIG CLI and Testing]
- Restores priority of static control plane pods in the cluster/gce/manifests control-plane manifests ([#89970](https://github.com/kubernetes/kubernetes/pull/89970), [@liggitt](https://github.com/liggitt)) [SIG Cluster Lifecycle and Node]
- Reverted devicemanager for Windows node added in 1.19rc1. ([#93263](https://github.com/kubernetes/kubernetes/pull/93263), [@liggitt](https://github.com/liggitt)) [SIG Node and Windows]
- Scheduler v1 Policy config or algorithm-provider settings can now be passed alongside v1beta1 ComponentConfig to aid transition from Policy to CC ([#92531](https://github.com/kubernetes/kubernetes/pull/92531), [@damemi](https://github.com/damemi)) [SIG Scheduling]
- Scheduling failures due to no nodes available are now reported as unschedulable under ```schedule_attempts_total``` metric. ([#90989](https://github.com/kubernetes/kubernetes/pull/90989), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]
- Service account tokens bound to pods can now be used during the pod deletion grace period. ([#89583](https://github.com/kubernetes/kubernetes/pull/89583), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Service load balancers no longer exclude nodes marked unschedulable from the candidate nodes. The service load balancer exclusion label should be used instead.
  
  Users upgrading from 1.18 who have cordoned nodes should set the `node.kubernetes.io/exclude-from-external-load-balancers` label on the impacted nodes before upgrading if they wish those nodes to remain excluded from service load balancers. ([#90823](https://github.com/kubernetes/kubernetes/pull/90823), [@smarterclayton](https://github.com/smarterclayton)) [SIG Apps, Cloud Provider and Network]
- Support kubectl annotate --list option ([#92576](https://github.com/kubernetes/kubernetes/pull/92576), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Sync LB backend nodes for Service `Type=LoadBalancer` on Add/Delete node events. ([#81185](https://github.com/kubernetes/kubernetes/pull/81185), [@andrewsykim](https://github.com/andrewsykim))
- The following components that do not expect non-empty, non-flag arguments will now print an error message and exit if an argument is specified: cloud-controller-manager, kube-apiserver, kube-controller-manager, kube-proxy, kubeadm {alpha|config|token|version}, kubemark. Flags should be prefixed with a single dash "-" (0x45) for short form or double dash "--" for long form. Before this change, malformed flags (for example, starting with a non-ascii dash character such as 0x8211: "–") would have been silently treated as positional arguments and ignored. ([#91349](https://github.com/kubernetes/kubernetes/pull/91349), [@neolit123](https://github.com/neolit123)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Network and Scheduling]
- The terminationGracePeriodSeconds from pod spec is respected for the mirror pod. ([#92442](https://github.com/kubernetes/kubernetes/pull/92442), [@tedyu](https://github.com/tedyu)) [SIG Node and Testing]
- Update github.com/moby/ipvs to v1.0.1 to fix IPVS compatibility issue with older kernels ([#90555](https://github.com/kubernetes/kubernetes/pull/90555), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network]
- Updates to pod status via the status subresource now validate that `status.podIP` and `status.podIPs` fields are well-formed. ([#90628](https://github.com/kubernetes/kubernetes/pull/90628), [@liggitt](https://github.com/liggitt)) [SIG Apps and Node]
- Wait for all CRDs to show up in discovery endpoint before reporting readiness. ([#89145](https://github.com/kubernetes/kubernetes/pull/89145), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- When evicting, Pods in Pending state are removed without checking PDBs. ([#83906](https://github.com/kubernetes/kubernetes/pull/83906), [@michaelgugino](https://github.com/michaelgugino)) [SIG API Machinery, Apps, Node and Scheduling]
- [security] Vulnerability in golang.org/x/text/encoding/unicode ([#92219](https://github.com/kubernetes/kubernetes/pull/92219), [@voor](https://github.com/voor)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Node]

### Other (Cleanup or Flake)

- --cache-dir sets cache directory for both http and discovery, defaults to $HOME/.kube/cache ([#92910](https://github.com/kubernetes/kubernetes/pull/92910), [@soltysh](https://github.com/soltysh)) [SIG API Machinery and CLI]
- Add `pod.Namespace` to the image log ([#91945](https://github.com/kubernetes/kubernetes/pull/91945), [@zhipengzuo](https://github.com/zhipengzuo))
- Add the ability to disable kubeconfig file lock through DISABLE_KUBECONFIG_LOCK env var ([#92513](https://github.com/kubernetes/kubernetes/pull/92513), [@soltysh](https://github.com/soltysh)) [SIG API Machinery and CLI]
- Adds additional testing to ensure that udp pods conntrack are cleaned up ([#90180](https://github.com/kubernetes/kubernetes/pull/90180), [@JacobTanenbaum](https://github.com/JacobTanenbaum)) [SIG Architecture, Network and Testing]
- Adjusts the fsType for cinder values to be `ext4` if no fsType is specified. ([#90608](https://github.com/kubernetes/kubernetes/pull/90608), [@huffmanca](https://github.com/huffmanca)) [SIG Storage]
- Base-images: Use debian-base:v2.1.0 ([#90697](https://github.com/kubernetes/kubernetes/pull/90697), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery and Release]
- Base-images: Use debian-iptables:v12.1.0 ([#90782](https://github.com/kubernetes/kubernetes/pull/90782), [@justaugustus](https://github.com/justaugustus)) [SIG Release]
- Beta.kubernetes.io/arch is already deprecated since v1.14, are targeted for removal in v1.18 ([#89462](https://github.com/kubernetes/kubernetes/pull/89462), [@wawa0210](https://github.com/wawa0210)) [SIG Testing]
- Build: Update to debian-base@v2.1.2 and debian-iptables@v12.1.1 ([#93667](https://github.com/kubernetes/kubernetes/pull/93667), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Release and Testing]
- Change beta.kubernetes.io/os  to kubernetes.io/os ([#89460](https://github.com/kubernetes/kubernetes/pull/89460), [@wawa0210](https://github.com/wawa0210)) [SIG Testing and Windows]
- Change beta.kubernetes.io/os to kubernetes.io/os ([#89461](https://github.com/kubernetes/kubernetes/pull/89461), [@wawa0210](https://github.com/wawa0210)) [SIG Cloud Provider and Cluster Lifecycle]
- Changes not found message when using `kubectl get` to retrieve not namespaced resources ([#89861](https://github.com/kubernetes/kubernetes/pull/89861), [@rccrdpccl](https://github.com/rccrdpccl)) [SIG CLI]
- CoreDNS will no longer be supporting Federation data translation for kube-dns ConfigMap ([#92716](https://github.com/kubernetes/kubernetes/pull/92716), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cluster Lifecycle]
- Deprecate kubectl top flags related to heapster
  Drop support of heapster in kubectl top ([#87498](https://github.com/kubernetes/kubernetes/pull/87498), [@serathius](https://github.com/serathius)) [SIG CLI]
- Deprecate the `--target-ram-md` flags that is no longer used for anything. ([#91818](https://github.com/kubernetes/kubernetes/pull/91818), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Drop some conformance tests that rely on Kubelet API directly ([#90615](https://github.com/kubernetes/kubernetes/pull/90615), [@dims](https://github.com/dims)) [SIG Architecture, Network, Release and Testing]
- Emit `WaitingForPodScheduled` event if the unbound PVC is in delay binding mode but used by a pod ([#91455](https://github.com/kubernetes/kubernetes/pull/91455), [@cofyc](https://github.com/cofyc)) [SIG Storage]
- Fix: license issue in blob disk feature ([#92824](https://github.com/kubernetes/kubernetes/pull/92824), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Improve server-side apply conflict errors by setting dedicated kubectl subcommand field managers ([#88885](https://github.com/kubernetes/kubernetes/pull/88885), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- IsFullyQualifiedDomainName() validates each label based on IsDNS1123Label. ([#90172](https://github.com/kubernetes/kubernetes/pull/90172), [@nak3](https://github.com/nak3)) [SIG API Machinery and Network]
- It is now possible to use the service annotation `cloud.google.com/network-tier: Standard` to configure the Network Tier of the GCE Loadbalancer ([#88532](https://github.com/kubernetes/kubernetes/pull/88532), [@zioproto](https://github.com/zioproto)) [SIG Cloud Provider, Network and Testing]
- Kube-aggregator: renames aggregator_unavailable_apiservice_count metric to aggregator_unavailable_apiservice_total ([#88156](https://github.com/kubernetes/kubernetes/pull/88156), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kube-apiserver: openapi schemas published for custom resources now reference standard ListMeta schema definitions ([#92546](https://github.com/kubernetes/kubernetes/pull/92546), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-proxy exposes a new metric, `kubeproxy_sync_proxy_rules_last_queued_timestamp_seconds`, that indicates the last time a change for kube-proxy was queued to be applied. ([#90175](https://github.com/kubernetes/kubernetes/pull/90175), [@squeed](https://github.com/squeed)) [SIG Instrumentation and Network]
- Kube-scheduler: The metric name `scheduler_total_preemption_attempts` has been renamed to `scheduler_preemption_attempts_total`. ([#91448](https://github.com/kubernetes/kubernetes/pull/91448), [@RainbowMango](https://github.com/RainbowMango)) [SIG API Machinery, Cluster Lifecycle, Instrumentation and Scheduling]
- Kube-up: defaults to limiting critical pods to the kube-system namespace to match behavior prior to 1.17 ([#93121](https://github.com/kubernetes/kubernetes/pull/93121), [@liggitt](https://github.com/liggitt)) [SIG Cloud Provider and Scheduling]
- Kubeadm now forwards the IPv6DualStack feature gate using the kubelet component config, instead of the kubelet command line ([#90840](https://github.com/kubernetes/kubernetes/pull/90840), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: do not use a DaemonSet for the pre-pull of control-plane images during "kubeadm upgrade apply". Individual node upgrades now pull the required images using a preflight check. The flag "--image-pull-timeout" for "kubeadm upgrade apply" is now deprecated and will be removed in a future release following a GA deprecation policy. ([#90788](https://github.com/kubernetes/kubernetes/pull/90788), [@xlgao-zju](https://github.com/xlgao-zju)) [SIG Cluster Lifecycle]
- Kubeadm: use two separate checks on /livez and /readyz for the kube-apiserver static Pod instead of using /healthz ([#90970](https://github.com/kubernetes/kubernetes/pull/90970), [@johscheuer](https://github.com/johscheuer)) [SIG Cluster Lifecycle]
- NONE ([#91597](https://github.com/kubernetes/kubernetes/pull/91597), [@elmiko](https://github.com/elmiko)) [SIG Autoscaling and Testing]
- Openapi-controller: remove the trailing `1` character literal from the rate limiting metric `APIServiceOpenAPIAggregationControllerQueue1` and rename it to `open_api_aggregation_controller` to adhere to Prometheus best practices. ([#77979](https://github.com/kubernetes/kubernetes/pull/77979), [@s-urbaniak](https://github.com/s-urbaniak)) [SIG API Machinery]
- Reduce event spam during a volume operation error. ([#89794](https://github.com/kubernetes/kubernetes/pull/89794), [@msau42](https://github.com/msau42)) [SIG Storage]
- Refactor the local nodeipam range allocator and instrument the cidrset used to store the allocated CIDRs with the following metrics:
  "cidrset_cidrs_allocations_total",
  "cidrset_cidrs_releases_total",
  "cidrset_usage_cidrs",
  "cidrset_allocation_tries_per_request", ([#90288](https://github.com/kubernetes/kubernetes/pull/90288), [@aojea](https://github.com/aojea)) [SIG Apps, Instrumentation, Network and Scalability]
- Remove deprecated --server-dry-run flag from kubectl apply ([#91308](https://github.com/kubernetes/kubernetes/pull/91308), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Renamed DefaultPodTopologySpread plugin to SelectorSpread plugin to avoid naming conflicts with feature Gate DefaultPodTopologySpread ([#92501](https://github.com/kubernetes/kubernetes/pull/92501), [@rakeshreddybandi](https://github.com/rakeshreddybandi)) [SIG Release, Scheduling and Testing]
- Replace framework.Failf with ExpectNoError ([#91811](https://github.com/kubernetes/kubernetes/pull/91811), [@lixiaobing1](https://github.com/lixiaobing1)) [SIG Instrumentation, Storage and Testing]
- Scheduler PreScore plugins are not executed if there is one filtered node or less. ([#89370](https://github.com/kubernetes/kubernetes/pull/89370), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]
- The "HostPath should give a volume the correct mode" is no longer a conformance test ([#90861](https://github.com/kubernetes/kubernetes/pull/90861), [@dims](https://github.com/dims)) [SIG Architecture and Testing]
- The Kubelet's `--experimental-allocatable-ignore-eviction` option is now marked as deprecated. ([#91578](https://github.com/kubernetes/kubernetes/pull/91578), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's `--experimental-mounter-path` and `--experimental-check-node-capabilities-before-mount` options are now marked as deprecated. ([#91373](https://github.com/kubernetes/kubernetes/pull/91373), [@knabben](https://github.com/knabben))
- The PR adds functionality to generate events when a PV or PVC processing encounters certain failures. The events help users to know the reason for the failure so they can take necessary recovery actions. ([#89845](https://github.com/kubernetes/kubernetes/pull/89845), [@yuga711](https://github.com/yuga711)) [SIG Apps]
- The PodShareProcessNamespace feature gate has been removed, and the PodShareProcessNamespace is unconditionally enabled. ([#90099](https://github.com/kubernetes/kubernetes/pull/90099), [@tanjunchen](https://github.com/tanjunchen)) [SIG Node]
- The kube-apiserver `--kubelet-https` flag is deprecated. kube-apiserver connections to kubelets now unconditionally use `https` (kubelets have unconditionally used `https` to serve the endpoints the apiserver communicates with since before v1.0). ([#91630](https://github.com/kubernetes/kubernetes/pull/91630), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Node]
- Update CNI to v0.8.6 ([#91370](https://github.com/kubernetes/kubernetes/pull/91370), [@justaugustus](https://github.com/justaugustus)) [SIG Cloud Provider, Network, Release and Testing]
- Update Golang to v1.14.5
  - Update repo-infra to 0.0.7 (to support go1.14.5 and go1.13.13)
    - Includes:
      - bazelbuild/bazel-toolchains@3.3.2
      - bazelbuild/rules_go@v0.22.7 ([#93088](https://github.com/kubernetes/kubernetes/pull/93088), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Update Golang to v1.14.6
  - Update repo-infra to 0.0.8 (to support go1.14.6 and go1.13.14)
    - Includes:
      - bazelbuild/bazel-toolchains@3.4.0
      - bazelbuild/rules_go@v0.22.8 ([#93198](https://github.com/kubernetes/kubernetes/pull/93198), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Update corefile-migration library to 1.0.8 ([#91856](https://github.com/kubernetes/kubernetes/pull/91856), [@wawa0210](https://github.com/wawa0210)) [SIG Node]
- Update default etcd server version to 3.4.4 ([#89214](https://github.com/kubernetes/kubernetes/pull/89214), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cluster Lifecycle and Testing]
- Update default etcd server version to 3.4.7 ([#89895](https://github.com/kubernetes/kubernetes/pull/89895), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cluster Lifecycle and Testing]
- Update default etcd server version to 3.4.9 ([#92349](https://github.com/kubernetes/kubernetes/pull/92349), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Update go.etcd.io/bbolt to v1.3.5 ([#92350](https://github.com/kubernetes/kubernetes/pull/92350), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery and Cloud Provider]
- Update opencontainers/runtime-spec dependency to v1.0.2 ([#89644](https://github.com/kubernetes/kubernetes/pull/89644), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- `beta.kubernetes.io/os` and `beta.kubernetes.io/arch` node labels are deprecated. Update node selectors to use `kubernetes.io/os` and `kubernetes.io/arch`. ([#91046](https://github.com/kubernetes/kubernetes/pull/91046), [@wawa0210](https://github.com/wawa0210)) [SIG Apps and Node]
- `kubectl config view` now redacts bearer tokens by default, similar to client certificates. The `--raw` flag can still be used to output full content. ([#88985](https://github.com/kubernetes/kubernetes/pull/88985), [@puerco](https://github.com/puerco))

## Dependencies

### Added
- cloud.google.com/go/bigquery: v1.0.1
- cloud.google.com/go/datastore: v1.0.0
- cloud.google.com/go/pubsub: v1.0.1
- cloud.google.com/go/storage: v1.0.0
- dmitri.shuralyov.com/gpu/mtl: 666a987
- github.com/cespare/xxhash/v2: [v2.1.1](https://github.com/cespare/xxhash/v2/tree/v2.1.1)
- github.com/checkpoint-restore/go-criu/v4: [v4.0.2](https://github.com/checkpoint-restore/go-criu/v4/tree/v4.0.2)
- github.com/chzyer/logex: [v1.1.10](https://github.com/chzyer/logex/tree/v1.1.10)
- github.com/chzyer/readline: [2972be2](https://github.com/chzyer/readline/tree/2972be2)
- github.com/chzyer/test: [a1ea475](https://github.com/chzyer/test/tree/a1ea475)
- github.com/containerd/cgroups: [0dbf7f0](https://github.com/containerd/cgroups/tree/0dbf7f0)
- github.com/containerd/continuity: [aaeac12](https://github.com/containerd/continuity/tree/aaeac12)
- github.com/containerd/fifo: [a9fb20d](https://github.com/containerd/fifo/tree/a9fb20d)
- github.com/containerd/go-runc: [5a6d9f3](https://github.com/containerd/go-runc/tree/5a6d9f3)
- github.com/containerd/ttrpc: [v1.0.0](https://github.com/containerd/ttrpc/tree/v1.0.0)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/go-systemd/v22: [v22.1.0](https://github.com/coreos/go-systemd/v22/tree/v22.1.0)
- github.com/cpuguy83/go-md2man/v2: [v2.0.0](https://github.com/cpuguy83/go-md2man/v2/tree/v2.0.0)
- github.com/docopt/docopt-go: [ee0de3b](https://github.com/docopt/docopt-go/tree/ee0de3b)
- github.com/go-gl/glfw/v3.3/glfw: [12ad95a](https://github.com/go-gl/glfw/v3.3/glfw/tree/12ad95a)
- github.com/go-ini/ini: [v1.9.0](https://github.com/go-ini/ini/tree/v1.9.0)
- github.com/godbus/dbus/v5: [v5.0.3](https://github.com/godbus/dbus/v5/tree/v5.0.3)
- github.com/ianlancetaylor/demangle: [5e5cf60](https://github.com/ianlancetaylor/demangle/tree/5e5cf60)
- github.com/ishidawataru/sctp: [7c296d4](https://github.com/ishidawataru/sctp/tree/7c296d4)
- github.com/moby/ipvs: [v1.0.1](https://github.com/moby/ipvs/tree/v1.0.1)
- github.com/moby/sys/mountinfo: [v0.1.3](https://github.com/moby/sys/mountinfo/tree/v0.1.3)
- github.com/moby/term: [672ec06](https://github.com/moby/term/tree/672ec06)
- github.com/russross/blackfriday/v2: [v2.0.1](https://github.com/russross/blackfriday/v2/tree/v2.0.1)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/ugorji/go: [v1.1.4](https://github.com/ugorji/go/tree/v1.1.4)
- github.com/yuin/goldmark: [v1.1.27](https://github.com/yuin/goldmark/tree/v1.1.27)
- google.golang.org/protobuf: v1.24.0
- gotest.tools/v3: v3.0.2
- k8s.io/klog/v2: v2.2.0

### Changed
- cloud.google.com/go: v0.38.0 → v0.51.0
- github.com/Azure/azure-sdk-for-go: [v35.0.0+incompatible → v43.0.0+incompatible](https://github.com/Azure/azure-sdk-for-go/compare/v35.0.0...v43.0.0)
- github.com/Azure/go-autorest/autorest/adal: [v0.5.0 → v0.8.2](https://github.com/Azure/go-autorest/autorest/adal/compare/v0.5.0...v0.8.2)
- github.com/Azure/go-autorest/autorest/date: [v0.1.0 → v0.2.0](https://github.com/Azure/go-autorest/autorest/date/compare/v0.1.0...v0.2.0)
- github.com/Azure/go-autorest/autorest/mocks: [v0.2.0 → v0.3.0](https://github.com/Azure/go-autorest/autorest/mocks/compare/v0.2.0...v0.3.0)
- github.com/Azure/go-autorest/autorest: [v0.9.0 → v0.9.6](https://github.com/Azure/go-autorest/autorest/compare/v0.9.0...v0.9.6)
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [27a4ced → 7901bc8](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/27a4ced...7901bc8)
- github.com/Microsoft/go-winio: [v0.4.14 → fc70bd9](https://github.com/Microsoft/go-winio/compare/v0.4.14...fc70bd9)
- github.com/Microsoft/hcsshim: [672e52e → 5eafd15](https://github.com/Microsoft/hcsshim/compare/672e52e...5eafd15)
- github.com/alecthomas/template: [a0175ee → fb15b89](https://github.com/alecthomas/template/compare/a0175ee...fb15b89)
- github.com/alecthomas/units: [2efee85 → c3de453](https://github.com/alecthomas/units/compare/2efee85...c3de453)
- github.com/beorn7/perks: [v1.0.0 → v1.0.1](https://github.com/beorn7/perks/compare/v1.0.0...v1.0.1)
- github.com/cilium/ebpf: [95b36a5 → 1c8d4c9](https://github.com/cilium/ebpf/compare/95b36a5...1c8d4c9)
- github.com/containerd/console: [84eeaae → v1.0.0](https://github.com/containerd/console/compare/84eeaae...v1.0.0)
- github.com/containerd/containerd: [v1.0.2 → v1.3.3](https://github.com/containerd/containerd/compare/v1.0.2...v1.3.3)
- github.com/containerd/typeurl: [2a93cfd → v1.0.0](https://github.com/containerd/typeurl/compare/2a93cfd...v1.0.0)
- github.com/containernetworking/cni: [v0.7.1 → v0.8.0](https://github.com/containernetworking/cni/compare/v0.7.1...v0.8.0)
- github.com/coredns/corefile-migration: [v1.0.6 → v1.0.10](https://github.com/coredns/corefile-migration/compare/v1.0.6...v1.0.10)
- github.com/coreos/pkg: [97fdf19 → 399ea9e](https://github.com/coreos/pkg/compare/97fdf19...399ea9e)
- github.com/docker/docker: [be7ac8b → aa6a989](https://github.com/docker/docker/compare/be7ac8b...aa6a989)
- github.com/docker/go-connections: [v0.3.0 → v0.4.0](https://github.com/docker/go-connections/compare/v0.3.0...v0.4.0)
- github.com/evanphx/json-patch: [v4.2.0+incompatible → e83c0a1](https://github.com/evanphx/json-patch/compare/v4.2.0...e83c0a1)
- github.com/fsnotify/fsnotify: [v1.4.7 → v1.4.9](https://github.com/fsnotify/fsnotify/compare/v1.4.7...v1.4.9)
- github.com/go-kit/kit: [v0.8.0 → v0.9.0](https://github.com/go-kit/kit/compare/v0.8.0...v0.9.0)
- github.com/go-logfmt/logfmt: [v0.3.0 → v0.4.0](https://github.com/go-logfmt/logfmt/compare/v0.3.0...v0.4.0)
- github.com/go-logr/logr: [v0.1.0 → v0.2.0](https://github.com/go-logr/logr/compare/v0.1.0...v0.2.0)
- github.com/golang/groupcache: [02826c3 → 215e871](https://github.com/golang/groupcache/compare/02826c3...215e871)
- github.com/golang/protobuf: [v1.3.2 → v1.4.2](https://github.com/golang/protobuf/compare/v1.3.2...v1.4.2)
- github.com/google/cadvisor: [v0.35.0 → v0.37.0](https://github.com/google/cadvisor/compare/v0.35.0...v0.37.0)
- github.com/google/go-cmp: [v0.3.0 → v0.4.0](https://github.com/google/go-cmp/compare/v0.3.0...v0.4.0)
- github.com/google/pprof: [3ea8567 → d4f498a](https://github.com/google/pprof/compare/3ea8567...d4f498a)
- github.com/googleapis/gax-go/v2: [v2.0.4 → v2.0.5](https://github.com/googleapis/gax-go/v2/compare/v2.0.4...v2.0.5)
- github.com/googleapis/gnostic: [v0.1.0 → v0.4.1](https://github.com/googleapis/gnostic/compare/v0.1.0...v0.4.1)
- github.com/gorilla/mux: [v1.7.0 → v1.7.3](https://github.com/gorilla/mux/compare/v1.7.0...v1.7.3)
- github.com/json-iterator/go: [v1.1.8 → v1.1.10](https://github.com/json-iterator/go/compare/v1.1.8...v1.1.10)
- github.com/jstemmer/go-junit-report: [af01ea7 → v0.9.1](https://github.com/jstemmer/go-junit-report/compare/af01ea7...v0.9.1)
- github.com/konsorten/go-windows-terminal-sequences: [v1.0.1 → v1.0.3](https://github.com/konsorten/go-windows-terminal-sequences/compare/v1.0.1...v1.0.3)
- github.com/kr/pretty: [v0.1.0 → v0.2.0](https://github.com/kr/pretty/compare/v0.1.0...v0.2.0)
- github.com/mattn/go-isatty: [v0.0.9 → v0.0.4](https://github.com/mattn/go-isatty/compare/v0.0.9...v0.0.4)
- github.com/matttproud/golang_protobuf_extensions: [v1.0.1 → c182aff](https://github.com/matttproud/golang_protobuf_extensions/compare/v1.0.1...c182aff)
- github.com/mistifyio/go-zfs: [v2.1.1+incompatible → f784269](https://github.com/mistifyio/go-zfs/compare/v2.1.1...f784269)
- github.com/mrunalp/fileutils: [7d4729f → abd8a0e](https://github.com/mrunalp/fileutils/compare/7d4729f...abd8a0e)
- github.com/opencontainers/runc: [v1.0.0-rc10 → 819fcc6](https://github.com/opencontainers/runc/compare/v1.0.0-rc10...819fcc6)
- github.com/opencontainers/runtime-spec: [v1.0.0 → 237cc4f](https://github.com/opencontainers/runtime-spec/compare/v1.0.0...237cc4f)
- github.com/opencontainers/selinux: [5215b18 → v1.5.2](https://github.com/opencontainers/selinux/compare/5215b18...v1.5.2)
- github.com/pkg/errors: [v0.8.1 → v0.9.1](https://github.com/pkg/errors/compare/v0.8.1...v0.9.1)
- github.com/prometheus/client_golang: [v1.0.0 → v1.7.1](https://github.com/prometheus/client_golang/compare/v1.0.0...v1.7.1)
- github.com/prometheus/common: [v0.4.1 → v0.10.0](https://github.com/prometheus/common/compare/v0.4.1...v0.10.0)
- github.com/prometheus/procfs: [v0.0.2 → v0.1.3](https://github.com/prometheus/procfs/compare/v0.0.2...v0.1.3)
- github.com/rubiojr/go-vhd: [0bfd3b3 → 02e2102](https://github.com/rubiojr/go-vhd/compare/0bfd3b3...02e2102)
- github.com/sirupsen/logrus: [v1.4.2 → v1.6.0](https://github.com/sirupsen/logrus/compare/v1.4.2...v1.6.0)
- github.com/spf13/cobra: [v0.0.5 → v1.0.0](https://github.com/spf13/cobra/compare/v0.0.5...v1.0.0)
- github.com/spf13/viper: [v1.3.2 → v1.4.0](https://github.com/spf13/viper/compare/v1.3.2...v1.4.0)
- github.com/tmc/grpc-websocket-proxy: [89b8d40 → 0ad062e](https://github.com/tmc/grpc-websocket-proxy/compare/89b8d40...0ad062e)
- github.com/urfave/cli: [v1.20.0 → v1.22.2](https://github.com/urfave/cli/compare/v1.20.0...v1.22.2)
- github.com/vishvananda/netlink: [v1.0.0 → v1.1.0](https://github.com/vishvananda/netlink/compare/v1.0.0...v1.1.0)
- github.com/vishvananda/netns: [be1fbed → 52d707b](https://github.com/vishvananda/netns/compare/be1fbed...52d707b)
- go.etcd.io/bbolt: v1.3.3 → v1.3.5
- go.etcd.io/etcd: 3cf2f69 → 18dfb9c
- go.opencensus.io: v0.21.0 → v0.22.2
- go.uber.org/atomic: v1.3.2 → v1.4.0
- golang.org/x/crypto: bac4c82 → 75b2880
- golang.org/x/exp: 4b39c73 → da58074
- golang.org/x/image: 0694c2d → cff245a
- golang.org/x/lint: 959b441 → fdd1cda
- golang.org/x/mobile: d3739f8 → d2bd2a2
- golang.org/x/mod: 4bf6d31 → v0.3.0
- golang.org/x/net: 13f9640 → ab34263
- golang.org/x/oauth2: 0f29369 → 858c2ad
- golang.org/x/sys: fde4db3 → ed371f2
- golang.org/x/text: v0.3.2 → v0.3.3
- golang.org/x/time: 9d24e82 → 555d28b
- golang.org/x/tools: 65e3620 → c1934b7
- golang.org/x/xerrors: a985d34 → 9bdfabe
- google.golang.org/api: 5213b80 → v0.15.1
- google.golang.org/appengine: v1.5.0 → v1.6.5
- google.golang.org/genproto: 24fa4b2 → cb27e3a
- google.golang.org/grpc: v1.26.0 → v1.27.0
- gopkg.in/check.v1: 788fd78 → 41f04d3
- honnef.co/go/tools: v0.0.1-2019.2.2 → v0.0.1-2019.2.3
- k8s.io/gengo: 36b2048 → 8167cfd
- k8s.io/kube-openapi: bf4fb3b → 656914f
- k8s.io/system-validators: v1.0.4 → v1.1.2
- k8s.io/utils: 0a110f9 → d5654de
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.7 → v0.0.9
- sigs.k8s.io/structured-merge-diff/v3: v3.0.0 → 43c19bb

### Removed
- github.com/OpenPeeDeeP/depguard: [v1.0.1](https://github.com/OpenPeeDeeP/depguard/tree/v1.0.1)
- github.com/Rican7/retry: [v0.1.0](https://github.com/Rican7/retry/tree/v0.1.0)
- github.com/StackExchange/wmi: [5d04971](https://github.com/StackExchange/wmi/tree/5d04971)
- github.com/anmitsu/go-shlex: [648efa6](https://github.com/anmitsu/go-shlex/tree/648efa6)
- github.com/bazelbuild/bazel-gazelle: [70208cb](https://github.com/bazelbuild/bazel-gazelle/tree/70208cb)
- github.com/bazelbuild/buildtools: [69366ca](https://github.com/bazelbuild/buildtools/tree/69366ca)
- github.com/bazelbuild/rules_go: [6dae44d](https://github.com/bazelbuild/rules_go/tree/6dae44d)
- github.com/bradfitz/go-smtpd: [deb6d62](https://github.com/bradfitz/go-smtpd/tree/deb6d62)
- github.com/cespare/prettybench: [03b8cfe](https://github.com/cespare/prettybench/tree/03b8cfe)
- github.com/checkpoint-restore/go-criu: [17b0214](https://github.com/checkpoint-restore/go-criu/tree/17b0214)
- github.com/client9/misspell: [v0.3.4](https://github.com/client9/misspell/tree/v0.3.4)
- github.com/coreos/go-etcd: [v2.0.0+incompatible](https://github.com/coreos/go-etcd/tree/v2.0.0)
- github.com/cpuguy83/go-md2man: [v1.0.10](https://github.com/cpuguy83/go-md2man/tree/v1.0.10)
- github.com/docker/libnetwork: [c8a5fca](https://github.com/docker/libnetwork/tree/c8a5fca)
- github.com/gliderlabs/ssh: [v0.1.1](https://github.com/gliderlabs/ssh/tree/v0.1.1)
- github.com/go-critic/go-critic: [1df3008](https://github.com/go-critic/go-critic/tree/1df3008)
- github.com/go-lintpack/lintpack: [v0.5.2](https://github.com/go-lintpack/lintpack/tree/v0.5.2)
- github.com/go-ole/go-ole: [v1.2.1](https://github.com/go-ole/go-ole/tree/v1.2.1)
- github.com/go-toolsmith/astcast: [v1.0.0](https://github.com/go-toolsmith/astcast/tree/v1.0.0)
- github.com/go-toolsmith/astcopy: [v1.0.0](https://github.com/go-toolsmith/astcopy/tree/v1.0.0)
- github.com/go-toolsmith/astequal: [v1.0.0](https://github.com/go-toolsmith/astequal/tree/v1.0.0)
- github.com/go-toolsmith/astfmt: [v1.0.0](https://github.com/go-toolsmith/astfmt/tree/v1.0.0)
- github.com/go-toolsmith/astinfo: [9809ff7](https://github.com/go-toolsmith/astinfo/tree/9809ff7)
- github.com/go-toolsmith/astp: [v1.0.0](https://github.com/go-toolsmith/astp/tree/v1.0.0)
- github.com/go-toolsmith/pkgload: [v1.0.0](https://github.com/go-toolsmith/pkgload/tree/v1.0.0)
- github.com/go-toolsmith/strparse: [v1.0.0](https://github.com/go-toolsmith/strparse/tree/v1.0.0)
- github.com/go-toolsmith/typep: [v1.0.0](https://github.com/go-toolsmith/typep/tree/v1.0.0)
- github.com/gobwas/glob: [v0.2.3](https://github.com/gobwas/glob/tree/v0.2.3)
- github.com/godbus/dbus: [2ff6f7f](https://github.com/godbus/dbus/tree/2ff6f7f)
- github.com/golangci/check: [cfe4005](https://github.com/golangci/check/tree/cfe4005)
- github.com/golangci/dupl: [3e9179a](https://github.com/golangci/dupl/tree/3e9179a)
- github.com/golangci/errcheck: [ef45e06](https://github.com/golangci/errcheck/tree/ef45e06)
- github.com/golangci/go-misc: [927a3d8](https://github.com/golangci/go-misc/tree/927a3d8)
- github.com/golangci/go-tools: [e32c541](https://github.com/golangci/go-tools/tree/e32c541)
- github.com/golangci/goconst: [041c5f2](https://github.com/golangci/goconst/tree/041c5f2)
- github.com/golangci/gocyclo: [2becd97](https://github.com/golangci/gocyclo/tree/2becd97)
- github.com/golangci/gofmt: [0b8337e](https://github.com/golangci/gofmt/tree/0b8337e)
- github.com/golangci/golangci-lint: [v1.18.0](https://github.com/golangci/golangci-lint/tree/v1.18.0)
- github.com/golangci/gosec: [66fb7fc](https://github.com/golangci/gosec/tree/66fb7fc)
- github.com/golangci/ineffassign: [42439a7](https://github.com/golangci/ineffassign/tree/42439a7)
- github.com/golangci/lint-1: [ee948d0](https://github.com/golangci/lint-1/tree/ee948d0)
- github.com/golangci/maligned: [b1d8939](https://github.com/golangci/maligned/tree/b1d8939)
- github.com/golangci/misspell: [950f5d1](https://github.com/golangci/misspell/tree/950f5d1)
- github.com/golangci/prealloc: [215b22d](https://github.com/golangci/prealloc/tree/215b22d)
- github.com/golangci/revgrep: [d9c87f5](https://github.com/golangci/revgrep/tree/d9c87f5)
- github.com/golangci/unconvert: [28b1c44](https://github.com/golangci/unconvert/tree/28b1c44)
- github.com/google/go-github: [v17.0.0+incompatible](https://github.com/google/go-github/tree/v17.0.0)
- github.com/google/go-querystring: [v1.0.0](https://github.com/google/go-querystring/tree/v1.0.0)
- github.com/gostaticanalysis/analysisutil: [v0.0.3](https://github.com/gostaticanalysis/analysisutil/tree/v0.0.3)
- github.com/jellevandenhooff/dkim: [f50fe3d](https://github.com/jellevandenhooff/dkim/tree/f50fe3d)
- github.com/klauspost/compress: [v1.4.1](https://github.com/klauspost/compress/tree/v1.4.1)
- github.com/logrusorgru/aurora: [a7b3b31](https://github.com/logrusorgru/aurora/tree/a7b3b31)
- github.com/mattn/go-shellwords: [v1.0.5](https://github.com/mattn/go-shellwords/tree/v1.0.5)
- github.com/mattn/goveralls: [v0.0.2](https://github.com/mattn/goveralls/tree/v0.0.2)
- github.com/mesos/mesos-go: [v0.0.9](https://github.com/mesos/mesos-go/tree/v0.0.9)
- github.com/mitchellh/go-ps: [4fdf99a](https://github.com/mitchellh/go-ps/tree/4fdf99a)
- github.com/mozilla/tls-observatory: [8791a20](https://github.com/mozilla/tls-observatory/tree/8791a20)
- github.com/nbutton23/zxcvbn-go: [eafdab6](https://github.com/nbutton23/zxcvbn-go/tree/eafdab6)
- github.com/pquerna/ffjson: [af8b230](https://github.com/pquerna/ffjson/tree/af8b230)
- github.com/quasilyte/go-consistent: [c6f3937](https://github.com/quasilyte/go-consistent/tree/c6f3937)
- github.com/ryanuber/go-glob: [256dc44](https://github.com/ryanuber/go-glob/tree/256dc44)
- github.com/shirou/gopsutil: [c95755e](https://github.com/shirou/gopsutil/tree/c95755e)
- github.com/shirou/w32: [bb4de01](https://github.com/shirou/w32/tree/bb4de01)
- github.com/shurcooL/go-goon: [37c2f52](https://github.com/shurcooL/go-goon/tree/37c2f52)
- github.com/shurcooL/go: [9e1955d](https://github.com/shurcooL/go/tree/9e1955d)
- github.com/sourcegraph/go-diff: [v0.5.1](https://github.com/sourcegraph/go-diff/tree/v0.5.1)
- github.com/tarm/serial: [98f6abe](https://github.com/tarm/serial/tree/98f6abe)
- github.com/timakin/bodyclose: [87058b9](https://github.com/timakin/bodyclose/tree/87058b9)
- github.com/ugorji/go/codec: [d75b2dc](https://github.com/ugorji/go/codec/tree/d75b2dc)
- github.com/ultraware/funlen: [v0.0.2](https://github.com/ultraware/funlen/tree/v0.0.2)
- github.com/valyala/bytebufferpool: [v1.0.0](https://github.com/valyala/bytebufferpool/tree/v1.0.0)
- github.com/valyala/fasthttp: [v1.2.0](https://github.com/valyala/fasthttp/tree/v1.2.0)
- github.com/valyala/quicktemplate: [v1.1.1](https://github.com/valyala/quicktemplate/tree/v1.1.1)
- github.com/valyala/tcplisten: [ceec8f9](https://github.com/valyala/tcplisten/tree/ceec8f9)
- go4.org: 417644f
- golang.org/x/build: 2835ba2
- golang.org/x/perf: 6e6d33e
- gopkg.in/airbrake/gobrake.v2: v2.0.9
- gopkg.in/gemnasium/logrus-airbrake-hook.v2: v2.1.2
- gotest.tools/gotestsum: v0.3.5
- grpc.go4.org: 11d0a25
- k8s.io/klog: v1.0.0
- k8s.io/repo-infra: v0.0.1-alpha.1
- mvdan.cc/interfacer: c200402
- mvdan.cc/lint: adc824a
- mvdan.cc/unparam: fbb5962
- sourcegraph.com/sqs/pbtypes: d3ebe8f


## Dependencies

### Added
- cloud.google.com/go/bigquery: v1.0.1
- cloud.google.com/go/datastore: v1.0.0
- cloud.google.com/go/pubsub: v1.0.1
- cloud.google.com/go/storage: v1.0.0
- dmitri.shuralyov.com/gpu/mtl: 666a987
- github.com/cespare/xxhash/v2: [v2.1.1](https://github.com/cespare/xxhash/v2/tree/v2.1.1)
- github.com/checkpoint-restore/go-criu/v4: [v4.0.2](https://github.com/checkpoint-restore/go-criu/v4/tree/v4.0.2)
- github.com/chzyer/logex: [v1.1.10](https://github.com/chzyer/logex/tree/v1.1.10)
- github.com/chzyer/readline: [2972be2](https://github.com/chzyer/readline/tree/2972be2)
- github.com/chzyer/test: [a1ea475](https://github.com/chzyer/test/tree/a1ea475)
- github.com/containerd/cgroups: [0dbf7f0](https://github.com/containerd/cgroups/tree/0dbf7f0)
- github.com/containerd/continuity: [aaeac12](https://github.com/containerd/continuity/tree/aaeac12)
- github.com/containerd/fifo: [a9fb20d](https://github.com/containerd/fifo/tree/a9fb20d)
- github.com/containerd/go-runc: [5a6d9f3](https://github.com/containerd/go-runc/tree/5a6d9f3)
- github.com/containerd/ttrpc: [v1.0.0](https://github.com/containerd/ttrpc/tree/v1.0.0)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/go-systemd/v22: [v22.1.0](https://github.com/coreos/go-systemd/v22/tree/v22.1.0)
- github.com/cpuguy83/go-md2man/v2: [v2.0.0](https://github.com/cpuguy83/go-md2man/v2/tree/v2.0.0)
- github.com/docopt/docopt-go: [ee0de3b](https://github.com/docopt/docopt-go/tree/ee0de3b)
- github.com/go-gl/glfw/v3.3/glfw: [12ad95a](https://github.com/go-gl/glfw/v3.3/glfw/tree/12ad95a)
- github.com/go-ini/ini: [v1.9.0](https://github.com/go-ini/ini/tree/v1.9.0)
- github.com/godbus/dbus/v5: [v5.0.3](https://github.com/godbus/dbus/v5/tree/v5.0.3)
- github.com/ianlancetaylor/demangle: [5e5cf60](https://github.com/ianlancetaylor/demangle/tree/5e5cf60)
- github.com/ishidawataru/sctp: [7c296d4](https://github.com/ishidawataru/sctp/tree/7c296d4)
- github.com/moby/ipvs: [v1.0.1](https://github.com/moby/ipvs/tree/v1.0.1)
- github.com/moby/sys/mountinfo: [v0.1.3](https://github.com/moby/sys/mountinfo/tree/v0.1.3)
- github.com/moby/term: [672ec06](https://github.com/moby/term/tree/672ec06)
- github.com/russross/blackfriday/v2: [v2.0.1](https://github.com/russross/blackfriday/v2/tree/v2.0.1)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/ugorji/go: [v1.1.4](https://github.com/ugorji/go/tree/v1.1.4)
- github.com/yuin/goldmark: [v1.1.27](https://github.com/yuin/goldmark/tree/v1.1.27)
- google.golang.org/protobuf: v1.24.0
- gotest.tools/v3: v3.0.2
- k8s.io/klog/v2: v2.2.0
- sigs.k8s.io/structured-merge-diff/v4: v4.0.1

### Changed
- cloud.google.com/go: v0.38.0 → v0.51.0
- github.com/Azure/azure-sdk-for-go: [v35.0.0+incompatible → v43.0.0+incompatible](https://github.com/Azure/azure-sdk-for-go/compare/v35.0.0...v43.0.0)
- github.com/Azure/go-autorest/autorest/adal: [v0.5.0 → v0.8.2](https://github.com/Azure/go-autorest/autorest/adal/compare/v0.5.0...v0.8.2)
- github.com/Azure/go-autorest/autorest/date: [v0.1.0 → v0.2.0](https://github.com/Azure/go-autorest/autorest/date/compare/v0.1.0...v0.2.0)
- github.com/Azure/go-autorest/autorest/mocks: [v0.2.0 → v0.3.0](https://github.com/Azure/go-autorest/autorest/mocks/compare/v0.2.0...v0.3.0)
- github.com/Azure/go-autorest/autorest: [v0.9.0 → v0.9.6](https://github.com/Azure/go-autorest/autorest/compare/v0.9.0...v0.9.6)
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [27a4ced → 7901bc8](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/27a4ced...7901bc8)
- github.com/Microsoft/go-winio: [v0.4.14 → fc70bd9](https://github.com/Microsoft/go-winio/compare/v0.4.14...fc70bd9)
- github.com/Microsoft/hcsshim: [672e52e → 5eafd15](https://github.com/Microsoft/hcsshim/compare/672e52e...5eafd15)
- github.com/alecthomas/template: [a0175ee → fb15b89](https://github.com/alecthomas/template/compare/a0175ee...fb15b89)
- github.com/alecthomas/units: [2efee85 → c3de453](https://github.com/alecthomas/units/compare/2efee85...c3de453)
- github.com/beorn7/perks: [v1.0.0 → v1.0.1](https://github.com/beorn7/perks/compare/v1.0.0...v1.0.1)
- github.com/cilium/ebpf: [95b36a5 → 1c8d4c9](https://github.com/cilium/ebpf/compare/95b36a5...1c8d4c9)
- github.com/containerd/console: [84eeaae → v1.0.0](https://github.com/containerd/console/compare/84eeaae...v1.0.0)
- github.com/containerd/containerd: [v1.0.2 → v1.3.3](https://github.com/containerd/containerd/compare/v1.0.2...v1.3.3)
- github.com/containerd/typeurl: [2a93cfd → v1.0.0](https://github.com/containerd/typeurl/compare/2a93cfd...v1.0.0)
- github.com/containernetworking/cni: [v0.7.1 → v0.8.0](https://github.com/containernetworking/cni/compare/v0.7.1...v0.8.0)
- github.com/coredns/corefile-migration: [v1.0.6 → v1.0.10](https://github.com/coredns/corefile-migration/compare/v1.0.6...v1.0.10)
- github.com/coreos/pkg: [97fdf19 → 399ea9e](https://github.com/coreos/pkg/compare/97fdf19...399ea9e)
- github.com/docker/docker: [be7ac8b → aa6a989](https://github.com/docker/docker/compare/be7ac8b...aa6a989)
- github.com/docker/go-connections: [v0.3.0 → v0.4.0](https://github.com/docker/go-connections/compare/v0.3.0...v0.4.0)
- github.com/evanphx/json-patch: [v4.2.0+incompatible → v4.9.0+incompatible](https://github.com/evanphx/json-patch/compare/v4.2.0...v4.9.0)
- github.com/fsnotify/fsnotify: [v1.4.7 → v1.4.9](https://github.com/fsnotify/fsnotify/compare/v1.4.7...v1.4.9)
- github.com/go-kit/kit: [v0.8.0 → v0.9.0](https://github.com/go-kit/kit/compare/v0.8.0...v0.9.0)
- github.com/go-logfmt/logfmt: [v0.3.0 → v0.4.0](https://github.com/go-logfmt/logfmt/compare/v0.3.0...v0.4.0)
- github.com/go-logr/logr: [v0.1.0 → v0.2.0](https://github.com/go-logr/logr/compare/v0.1.0...v0.2.0)
- github.com/golang/groupcache: [02826c3 → 215e871](https://github.com/golang/groupcache/compare/02826c3...215e871)
- github.com/golang/protobuf: [v1.3.2 → v1.4.2](https://github.com/golang/protobuf/compare/v1.3.2...v1.4.2)
- github.com/google/cadvisor: [v0.35.0 → v0.37.0](https://github.com/google/cadvisor/compare/v0.35.0...v0.37.0)
- github.com/google/go-cmp: [v0.3.0 → v0.4.0](https://github.com/google/go-cmp/compare/v0.3.0...v0.4.0)
- github.com/google/pprof: [3ea8567 → d4f498a](https://github.com/google/pprof/compare/3ea8567...d4f498a)
- github.com/googleapis/gax-go/v2: [v2.0.4 → v2.0.5](https://github.com/googleapis/gax-go/v2/compare/v2.0.4...v2.0.5)
- github.com/googleapis/gnostic: [v0.1.0 → v0.4.1](https://github.com/googleapis/gnostic/compare/v0.1.0...v0.4.1)
- github.com/gorilla/mux: [v1.7.0 → v1.7.3](https://github.com/gorilla/mux/compare/v1.7.0...v1.7.3)
- github.com/json-iterator/go: [v1.1.8 → v1.1.10](https://github.com/json-iterator/go/compare/v1.1.8...v1.1.10)
- github.com/jstemmer/go-junit-report: [af01ea7 → v0.9.1](https://github.com/jstemmer/go-junit-report/compare/af01ea7...v0.9.1)
- github.com/konsorten/go-windows-terminal-sequences: [v1.0.1 → v1.0.3](https://github.com/konsorten/go-windows-terminal-sequences/compare/v1.0.1...v1.0.3)
- github.com/kr/pretty: [v0.1.0 → v0.2.0](https://github.com/kr/pretty/compare/v0.1.0...v0.2.0)
- github.com/mattn/go-isatty: [v0.0.9 → v0.0.4](https://github.com/mattn/go-isatty/compare/v0.0.9...v0.0.4)
- github.com/matttproud/golang_protobuf_extensions: [v1.0.1 → c182aff](https://github.com/matttproud/golang_protobuf_extensions/compare/v1.0.1...c182aff)
- github.com/mistifyio/go-zfs: [v2.1.1+incompatible → f784269](https://github.com/mistifyio/go-zfs/compare/v2.1.1...f784269)
- github.com/mrunalp/fileutils: [7d4729f → abd8a0e](https://github.com/mrunalp/fileutils/compare/7d4729f...abd8a0e)
- github.com/opencontainers/runc: [v1.0.0-rc10 → 819fcc6](https://github.com/opencontainers/runc/compare/v1.0.0-rc10...819fcc6)
- github.com/opencontainers/runtime-spec: [v1.0.0 → 237cc4f](https://github.com/opencontainers/runtime-spec/compare/v1.0.0...237cc4f)
- github.com/opencontainers/selinux: [5215b18 → v1.5.2](https://github.com/opencontainers/selinux/compare/5215b18...v1.5.2)
- github.com/pkg/errors: [v0.8.1 → v0.9.1](https://github.com/pkg/errors/compare/v0.8.1...v0.9.1)
- github.com/prometheus/client_golang: [v1.0.0 → v1.7.1](https://github.com/prometheus/client_golang/compare/v1.0.0...v1.7.1)
- github.com/prometheus/common: [v0.4.1 → v0.10.0](https://github.com/prometheus/common/compare/v0.4.1...v0.10.0)
- github.com/prometheus/procfs: [v0.0.2 → v0.1.3](https://github.com/prometheus/procfs/compare/v0.0.2...v0.1.3)
- github.com/rubiojr/go-vhd: [0bfd3b3 → 02e2102](https://github.com/rubiojr/go-vhd/compare/0bfd3b3...02e2102)
- github.com/sirupsen/logrus: [v1.4.2 → v1.6.0](https://github.com/sirupsen/logrus/compare/v1.4.2...v1.6.0)
- github.com/spf13/cobra: [v0.0.5 → v1.0.0](https://github.com/spf13/cobra/compare/v0.0.5...v1.0.0)
- github.com/spf13/viper: [v1.3.2 → v1.4.0](https://github.com/spf13/viper/compare/v1.3.2...v1.4.0)
- github.com/tmc/grpc-websocket-proxy: [89b8d40 → 0ad062e](https://github.com/tmc/grpc-websocket-proxy/compare/89b8d40...0ad062e)
- github.com/urfave/cli: [v1.20.0 → v1.22.2](https://github.com/urfave/cli/compare/v1.20.0...v1.22.2)
- github.com/vishvananda/netlink: [v1.0.0 → v1.1.0](https://github.com/vishvananda/netlink/compare/v1.0.0...v1.1.0)
- github.com/vishvananda/netns: [be1fbed → 52d707b](https://github.com/vishvananda/netns/compare/be1fbed...52d707b)
- go.etcd.io/bbolt: v1.3.3 → v1.3.5
- go.etcd.io/etcd: 3cf2f69 → 17cef6e
- go.opencensus.io: v0.21.0 → v0.22.2
- go.uber.org/atomic: v1.3.2 → v1.4.0
- golang.org/x/crypto: bac4c82 → 75b2880
- golang.org/x/exp: 4b39c73 → da58074
- golang.org/x/image: 0694c2d → cff245a
- golang.org/x/lint: 959b441 → fdd1cda
- golang.org/x/mobile: d3739f8 → d2bd2a2
- golang.org/x/mod: 4bf6d31 → v0.3.0
- golang.org/x/net: 13f9640 → ab34263
- golang.org/x/oauth2: 0f29369 → 858c2ad
- golang.org/x/sys: fde4db3 → ed371f2
- golang.org/x/text: v0.3.2 → v0.3.3
- golang.org/x/time: 9d24e82 → 555d28b
- golang.org/x/tools: 65e3620 → c1934b7
- golang.org/x/xerrors: a985d34 → 9bdfabe
- google.golang.org/api: 5213b80 → v0.15.1
- google.golang.org/appengine: v1.5.0 → v1.6.5
- google.golang.org/genproto: 24fa4b2 → cb27e3a
- google.golang.org/grpc: v1.26.0 → v1.27.0
- gopkg.in/check.v1: 788fd78 → 41f04d3
- honnef.co/go/tools: v0.0.1-2019.2.2 → v0.0.1-2019.2.3
- k8s.io/gengo: 36b2048 → 8167cfd
- k8s.io/kube-openapi: bf4fb3b → 6aeccd4
- k8s.io/system-validators: v1.0.4 → v1.1.2
- k8s.io/utils: a9aa75a → d5654de
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.7 → v0.0.9

### Removed
- github.com/OpenPeeDeeP/depguard: [v1.0.1](https://github.com/OpenPeeDeeP/depguard/tree/v1.0.1)
- github.com/Rican7/retry: [v0.1.0](https://github.com/Rican7/retry/tree/v0.1.0)
- github.com/StackExchange/wmi: [5d04971](https://github.com/StackExchange/wmi/tree/5d04971)
- github.com/anmitsu/go-shlex: [648efa6](https://github.com/anmitsu/go-shlex/tree/648efa6)
- github.com/bazelbuild/bazel-gazelle: [70208cb](https://github.com/bazelbuild/bazel-gazelle/tree/70208cb)
- github.com/bazelbuild/buildtools: [69366ca](https://github.com/bazelbuild/buildtools/tree/69366ca)
- github.com/bazelbuild/rules_go: [6dae44d](https://github.com/bazelbuild/rules_go/tree/6dae44d)
- github.com/bradfitz/go-smtpd: [deb6d62](https://github.com/bradfitz/go-smtpd/tree/deb6d62)
- github.com/cespare/prettybench: [03b8cfe](https://github.com/cespare/prettybench/tree/03b8cfe)
- github.com/checkpoint-restore/go-criu: [17b0214](https://github.com/checkpoint-restore/go-criu/tree/17b0214)
- github.com/client9/misspell: [v0.3.4](https://github.com/client9/misspell/tree/v0.3.4)
- github.com/coreos/go-etcd: [v2.0.0+incompatible](https://github.com/coreos/go-etcd/tree/v2.0.0)
- github.com/cpuguy83/go-md2man: [v1.0.10](https://github.com/cpuguy83/go-md2man/tree/v1.0.10)
- github.com/docker/libnetwork: [c8a5fca](https://github.com/docker/libnetwork/tree/c8a5fca)
- github.com/gliderlabs/ssh: [v0.1.1](https://github.com/gliderlabs/ssh/tree/v0.1.1)
- github.com/go-critic/go-critic: [1df3008](https://github.com/go-critic/go-critic/tree/1df3008)
- github.com/go-lintpack/lintpack: [v0.5.2](https://github.com/go-lintpack/lintpack/tree/v0.5.2)
- github.com/go-ole/go-ole: [v1.2.1](https://github.com/go-ole/go-ole/tree/v1.2.1)
- github.com/go-toolsmith/astcast: [v1.0.0](https://github.com/go-toolsmith/astcast/tree/v1.0.0)
- github.com/go-toolsmith/astcopy: [v1.0.0](https://github.com/go-toolsmith/astcopy/tree/v1.0.0)
- github.com/go-toolsmith/astequal: [v1.0.0](https://github.com/go-toolsmith/astequal/tree/v1.0.0)
- github.com/go-toolsmith/astfmt: [v1.0.0](https://github.com/go-toolsmith/astfmt/tree/v1.0.0)
- github.com/go-toolsmith/astinfo: [9809ff7](https://github.com/go-toolsmith/astinfo/tree/9809ff7)
- github.com/go-toolsmith/astp: [v1.0.0](https://github.com/go-toolsmith/astp/tree/v1.0.0)
- github.com/go-toolsmith/pkgload: [v1.0.0](https://github.com/go-toolsmith/pkgload/tree/v1.0.0)
- github.com/go-toolsmith/strparse: [v1.0.0](https://github.com/go-toolsmith/strparse/tree/v1.0.0)
- github.com/go-toolsmith/typep: [v1.0.0](https://github.com/go-toolsmith/typep/tree/v1.0.0)
- github.com/gobwas/glob: [v0.2.3](https://github.com/gobwas/glob/tree/v0.2.3)
- github.com/godbus/dbus: [2ff6f7f](https://github.com/godbus/dbus/tree/2ff6f7f)
- github.com/golangci/check: [cfe4005](https://github.com/golangci/check/tree/cfe4005)
- github.com/golangci/dupl: [3e9179a](https://github.com/golangci/dupl/tree/3e9179a)
- github.com/golangci/errcheck: [ef45e06](https://github.com/golangci/errcheck/tree/ef45e06)
- github.com/golangci/go-misc: [927a3d8](https://github.com/golangci/go-misc/tree/927a3d8)
- github.com/golangci/go-tools: [e32c541](https://github.com/golangci/go-tools/tree/e32c541)
- github.com/golangci/goconst: [041c5f2](https://github.com/golangci/goconst/tree/041c5f2)
- github.com/golangci/gocyclo: [2becd97](https://github.com/golangci/gocyclo/tree/2becd97)
- github.com/golangci/gofmt: [0b8337e](https://github.com/golangci/gofmt/tree/0b8337e)
- github.com/golangci/golangci-lint: [v1.18.0](https://github.com/golangci/golangci-lint/tree/v1.18.0)
- github.com/golangci/gosec: [66fb7fc](https://github.com/golangci/gosec/tree/66fb7fc)
- github.com/golangci/ineffassign: [42439a7](https://github.com/golangci/ineffassign/tree/42439a7)
- github.com/golangci/lint-1: [ee948d0](https://github.com/golangci/lint-1/tree/ee948d0)
- github.com/golangci/maligned: [b1d8939](https://github.com/golangci/maligned/tree/b1d8939)
- github.com/golangci/misspell: [950f5d1](https://github.com/golangci/misspell/tree/950f5d1)
- github.com/golangci/prealloc: [215b22d](https://github.com/golangci/prealloc/tree/215b22d)
- github.com/golangci/revgrep: [d9c87f5](https://github.com/golangci/revgrep/tree/d9c87f5)
- github.com/golangci/unconvert: [28b1c44](https://github.com/golangci/unconvert/tree/28b1c44)
- github.com/google/go-github: [v17.0.0+incompatible](https://github.com/google/go-github/tree/v17.0.0)
- github.com/google/go-querystring: [v1.0.0](https://github.com/google/go-querystring/tree/v1.0.0)
- github.com/gostaticanalysis/analysisutil: [v0.0.3](https://github.com/gostaticanalysis/analysisutil/tree/v0.0.3)
- github.com/jellevandenhooff/dkim: [f50fe3d](https://github.com/jellevandenhooff/dkim/tree/f50fe3d)
- github.com/klauspost/compress: [v1.4.1](https://github.com/klauspost/compress/tree/v1.4.1)
- github.com/logrusorgru/aurora: [a7b3b31](https://github.com/logrusorgru/aurora/tree/a7b3b31)
- github.com/mattn/go-shellwords: [v1.0.5](https://github.com/mattn/go-shellwords/tree/v1.0.5)
- github.com/mattn/goveralls: [v0.0.2](https://github.com/mattn/goveralls/tree/v0.0.2)
- github.com/mesos/mesos-go: [v0.0.9](https://github.com/mesos/mesos-go/tree/v0.0.9)
- github.com/mitchellh/go-ps: [4fdf99a](https://github.com/mitchellh/go-ps/tree/4fdf99a)
- github.com/mozilla/tls-observatory: [8791a20](https://github.com/mozilla/tls-observatory/tree/8791a20)
- github.com/nbutton23/zxcvbn-go: [eafdab6](https://github.com/nbutton23/zxcvbn-go/tree/eafdab6)
- github.com/pquerna/ffjson: [af8b230](https://github.com/pquerna/ffjson/tree/af8b230)
- github.com/quasilyte/go-consistent: [c6f3937](https://github.com/quasilyte/go-consistent/tree/c6f3937)
- github.com/ryanuber/go-glob: [256dc44](https://github.com/ryanuber/go-glob/tree/256dc44)
- github.com/shirou/gopsutil: [c95755e](https://github.com/shirou/gopsutil/tree/c95755e)
- github.com/shirou/w32: [bb4de01](https://github.com/shirou/w32/tree/bb4de01)
- github.com/shurcooL/go-goon: [37c2f52](https://github.com/shurcooL/go-goon/tree/37c2f52)
- github.com/shurcooL/go: [9e1955d](https://github.com/shurcooL/go/tree/9e1955d)
- github.com/sourcegraph/go-diff: [v0.5.1](https://github.com/sourcegraph/go-diff/tree/v0.5.1)
- github.com/tarm/serial: [98f6abe](https://github.com/tarm/serial/tree/98f6abe)
- github.com/timakin/bodyclose: [87058b9](https://github.com/timakin/bodyclose/tree/87058b9)
- github.com/ugorji/go/codec: [d75b2dc](https://github.com/ugorji/go/codec/tree/d75b2dc)
- github.com/ultraware/funlen: [v0.0.2](https://github.com/ultraware/funlen/tree/v0.0.2)
- github.com/valyala/bytebufferpool: [v1.0.0](https://github.com/valyala/bytebufferpool/tree/v1.0.0)
- github.com/valyala/fasthttp: [v1.2.0](https://github.com/valyala/fasthttp/tree/v1.2.0)
- github.com/valyala/quicktemplate: [v1.1.1](https://github.com/valyala/quicktemplate/tree/v1.1.1)
- github.com/valyala/tcplisten: [ceec8f9](https://github.com/valyala/tcplisten/tree/ceec8f9)
- go4.org: 417644f
- golang.org/x/build: 2835ba2
- golang.org/x/perf: 6e6d33e
- gopkg.in/airbrake/gobrake.v2: v2.0.9
- gopkg.in/gemnasium/logrus-airbrake-hook.v2: v2.1.2
- gotest.tools/gotestsum: v0.3.5
- grpc.go4.org: 11d0a25
- k8s.io/klog: v1.0.0
- k8s.io/repo-infra: v0.0.1-alpha.1
- mvdan.cc/interfacer: c200402
- mvdan.cc/lint: adc824a
- mvdan.cc/unparam: fbb5962
- sigs.k8s.io/structured-merge-diff/v3: v3.0.0
- sourcegraph.com/sqs/pbtypes: d3ebe8f



# v1.19.0-rc.4


## Downloads for v1.19.0-rc.4

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes.tar.gz) | 98bb6e2ac98a3176a9592982fec04b037d189de73cb7175d51596075bfd008c7ec0a2301b9511375626581f864ea74b5731e2ef2b4c70363f1860d11eb827de1
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-src.tar.gz) | d4686f8d07fe6f324f46880a4dc5af9afa314a6b7dca82d0edb50290b769d25d18babcc5257a96a51a046052c7747e324b025a90a36ca5e62f67642bb03c44f6

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-darwin-amd64.tar.gz) | e9184ceb37491764c1ea2ef0b1eca55f27109bb973c7ff7c78e83c5945840baf28fdead21ef861b0c5cb81f4dc39d0af86ed7b17ed6f087f211084d0033dad11
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-linux-386.tar.gz) | c9f1ec4e8d9c6245f955b2132c0fae6d851a6a49a5b7a2333c01ba9fafa3c4e8a07c6462e525179c25e308520502544ab4dc570e1b9d0090d58b6d18bcfcba47
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-linux-amd64.tar.gz) | d23858b03c3554ad12517ce5f7855ceccaa9425c2d19fbc9cf902c0e796a8182f8b0e8eeeeefff0f46e960dfee96b2a2033a04a3194ac34dfd2a32003775d060
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-linux-arm.tar.gz) | a745b3a06fe992713e4d7f921e2f36c5b39222d7b1a3e13299d15925743dd99965c2bdf05b4deda30a6f6232a40588e154fdd83f40d9d260d7ac8f70b18cad48
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-linux-arm64.tar.gz) | 719b1f30e4bbb05d638ee78cf0145003a1e783bbd0c2f0952fcb30702dd27dfd44c3bc85baaf9a776e490ed53c638327ed1c0e5a882dc93c24d7cac20e4f1dd0
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-linux-ppc64le.tar.gz) | fba0794e9dc0f231da5a4e85e37c2d8260e5205574e0421f5122a7d60a05ca6546912519a28e8d6c241904617234e1b0b5c94f890853ad5ae4e329ef8085a092
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-linux-s390x.tar.gz) | edce96e37e37fd2b60e34fe56240461094e5784985790453becdfe09011305fcbd8a50ee5bf6d82a70d208d660796d0ddf160bed0292271b6617049db800962f
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-windows-386.tar.gz) | 06c849b35d886bebedfd8d906ac37ccda10e05b06542fefe6440268c5e937f235915e53daafe35076b68e0af0d4ddeab4240da55b09ee52fa26928945f1a2ecd
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-client-windows-amd64.tar.gz) | a13e6ec70f76d6056d5668b678ba6f223e35756cded6c84dfb58e28b3741fecfa7cb5e6ae2239392d770028d1b55ca8eb520c0b24e13fc3bd38720134b472d53

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-server-linux-amd64.tar.gz) | ff7fbf211c29b94c19466337e6c142e733c8c0bac815a97906168e57d21ad1b2965e4b0033b525de8fed9a91ab943e3eb6d330f8095660e32be2791f8161a6a2
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-server-linux-arm.tar.gz) | 218a35466ebcc0bc0e8eff9bbb2e58f0ac3bec6a75f45a7c1487aa4fc3e2bddb90b74e91a2b81bbbbb1eb1de2df310adab4c07c2a2c38a9973580b4f85734a1f
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-server-linux-arm64.tar.gz) | 8a81d727e63875d18336fda8bb6f570084553fc346b7e7df2fc3e1c04a8ef766f61d96d445537e4660ce2f46b170a04218a4d8a270b3ad373caa3f815c0fec93
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-server-linux-ppc64le.tar.gz) | 9b5afa44ac2e1232cd0c54b3602a2027bc8a08b30809b3ef973f75793b35a596491e6056d7995e493a1e4f48d83389240ac2e609b9f76d2715b8e115e6648716
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-server-linux-s390x.tar.gz) | f3034b2e88b5c1d362d84f78dfd1761d0fc21ada1cd6b1a6132a709c663a1206651df96c948534b3661f6b70b651e33021aced3a7574a0e5fc88ace73fff2a6f

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-node-linux-amd64.tar.gz) | 2061a8f5bc2060b071564c92b693950eda7768a9ceb874982f0e91aa78284fb477becb55ecf099acae73c447271240caecefc19b3b29024e9b818e0639c2fc70
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-node-linux-arm.tar.gz) | c06b817b191ff9a4b05bf70fc14edcf01d4ded204e489966b1761dd68d45d054028870301e45ebba648c0be097c7c42120867c8b28fdd625c8eb5a5bc3ace71d
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-node-linux-arm64.tar.gz) | 21efb3bf23628546de97210074f48e928fec211b81215eff8b10c3f5f7e79bb5911c1393a66a8363a0183fe299bf98b316c0c2d77a13c8c5b798255c056bd806
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-node-linux-ppc64le.tar.gz) | ce31dd65b9cbfaabdc3c93e8afee0ea5606c64e6bf4452078bee73b1d328d23ebdbc871a00edd16fa4e676406a707cf9113fdaec76489681c379c35c3fd6aeb0
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-node-linux-s390x.tar.gz) | 523a8e1d6e0eff70810e846c171b7f74a4aaecb25237addf541a9f8adb3797402b6e57abf9030f62d5bb333d5f5e8960199a44fe48696a4da98f7ed7d993cde1
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.4/kubernetes-node-windows-amd64.tar.gz) | a7fbcd11ea8b6427e7846e39b2fdeae41d484320faa8f3e9b6a777d87ba62e7399ad0ec6a33d9a4675001898881e444f336eebd5c97b3903dee803834a646f3d

## Changelog since v1.19.0-rc.3

## Changes by Kind

### Deprecation

- Kube-apiserver: the componentstatus API is deprecated. This API provided status of etcd, kube-scheduler, and kube-controller-manager components, but only worked when those components were local to the API server, and when kube-scheduler and kube-controller-manager exposed unsecured health endpoints. Instead of this API, etcd health is included in the kube-apiserver health check and kube-scheduler/kube-controller-manager health checks can be made directly against those components' health endpoints. ([#93570](https://github.com/kubernetes/kubernetes/pull/93570), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps and Cluster Lifecycle]

### Bug or Regression

- A panic in the apiserver caused by the `informer-sync` health checker is now fixed. ([#93600](https://github.com/kubernetes/kubernetes/pull/93600), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- EndpointSliceMirroring controller now copies labels from Endpoints to EndpointSlices. ([#93442](https://github.com/kubernetes/kubernetes/pull/93442), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Kube-apiserver: jsonpath expressions with consecutive recursive descent operators are no longer evaluated for custom resource printer columns ([#93408](https://github.com/kubernetes/kubernetes/pull/93408), [@joelsmith](https://github.com/joelsmith)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Build: Update to debian-base@v2.1.0 and debian-iptables@v12.1.1 ([#93667](https://github.com/kubernetes/kubernetes/pull/93667), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Release and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
- k8s.io/utils: 0bdb4ca → d5654de

### Removed
_Nothing has changed._



# v1.19.0-rc.3


## Downloads for v1.19.0-rc.3

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes.tar.gz) | 31f98fb8d51c6dfa60e2cf710a35af14bc17a6b3833b3802cebc92586b01996c091943087dc818541fc13ad75f051d20c176d9506fc0c86ab582a9295fb7ed59
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-src.tar.gz) | 4886180edf6287adf9db1cdab1e8439c41296c6b5b40af9c4642bb6cfd1fb894313c6d1210c2b882f1bc40dbfd17ed20c5159ea3a8c79ad2ef7a7630016e99de

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-darwin-amd64.tar.gz) | 19b0f9fe95e135329ce2cb9dd3e95551f3552be035ce7235e055c9d775dfa747c773b0806b5c2eef1e69863368be13adcb4c5ef78ae05af65483434686e9a773
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-linux-386.tar.gz) | 219a15b54ba616938960ac38869c87be573e3cd7897e4790c31cdeb819415fcefb4f293fc49d63901b42f70e66555c72a8a774cced7ec15a93592dffef3b1336
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-linux-amd64.tar.gz) | 7c5a2163e0e163d3b1819acc7c4475d35b853318dd5a6084ea0785744a92063edf65254b0f0eae0f69f4598561c182033a9722c1b8a61894959333f1357cb1f9
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-linux-arm.tar.gz) | 5d48f78da6a54b63d8ea68e983d780c672b546b4a5d1fb2c15033377dd3098f0011516b55cc47db316dacabdbbd3660108014d12899ef1f4a6a03158ef503101
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-linux-arm64.tar.gz) | c2db09db465f8ea2bd7b82971a59a2be394b2f9a8c15ff78ab06c3a41d9f1292175de19fdc7450cc28746027d59dc3162cb47b64555e91d324d33d699d89f408
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-linux-ppc64le.tar.gz) | f28c9c672bc2c5e780f6fdcf019a5dad7172537e38e2ab7d52a1ea55babb41d296cef97b482133c7fce0634b1aed1b5322d1e0061d30c3848e4c912a7e1ca248
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-linux-s390x.tar.gz) | 22844af3c97eb9f36a038c552e9818b8670cd02acc98defe5c131c7f88621014cd51c343c1e0921b88ebbfd9850a5c277f50df78350f7565db4e356521d415d4
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-windows-386.tar.gz) | edabe78a1337f73caa81c885d722544fec443f875297291e57608d4f486c897af6c602656048a4325fcc957ce1d7acb1c1cf06cab0bd2e36acee1d6be206d3c6
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-client-windows-amd64.tar.gz) | f1a370b9ec298838e302909dd826760b50b593ee2d2247416d345ff00331973e7b6b29cef69f07d6c1477ab534d6ec9d1bbf5d3c2d1bb9b5b2933e088c8de3f1

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-server-linux-amd64.tar.gz) | 193c023306d7478c2e43c4039642649c77598c05b07dbc466611e166f0233a7ea2a7f2ff61763b2630988c151a591f44428942d8ee06ce6766641e1dcfaac588
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-server-linux-arm.tar.gz) | c1aa489779fb74855385f24120691771a05b57069064c99471b238e5d541d94d4356e7d2cd5b66c284c46bde1fc3eff2a1cebfcd9e72a78377b76e32a1dbf57a
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-server-linux-arm64.tar.gz) | 73400003571f9f0308051ca448b1f96d83e9d211876a57b572ffb787ad0c3bb5f1e20547d959f0fac51a916cf7f26f8839ddddd55d4a38e59c8270d5eb48a970
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-server-linux-ppc64le.tar.gz) | bebf75d884d024ffebfda40abaa0bfec99a6d4cd3cc0fac904a1c4c190e6eb8fc9412c7790b2f8a2b3cc8ccdf8556d9a93eec37e5c298f8abd62ee41de641a42
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-server-linux-s390x.tar.gz) | 8374dfb689abae31480814d6849aaa51e30666b7203cdcf204d49c9a0344391232f40d135671ec8316e26d1685e1cbbea4b829ff3b9f83c08c2d1ba50cd5aeb2

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-node-linux-amd64.tar.gz) | 194ee29b012463e6d90c346f76d53f94778f75cc916b0e10a5ee174983fac6e848438e0d9e36a475c5d7ba7b0f3ad5debc039ec8f95fdfb6229843f04dfacb53
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-node-linux-arm.tar.gz) | f0d075eaa84dae7ce2101dfa421021b0bfea235fe606d693e881775cd37ff0b82ca6a419dfe48becd2eddc5f882e97ba838164e6ac5991445225c31f147b4f97
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-node-linux-arm64.tar.gz) | 3dc69981f31b01a2d8c439f7047f73e5699a121501c516ed17b3e91ed358ee97e43fa955eb9e1434cbf7864e51097e76c216075d34f4b455930a44af6c64be5c
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-node-linux-ppc64le.tar.gz) | 4a77720373960a0cc20bbcfcdfe17f8d5ddaaf2e38bad607cfe05831029e8e14559e78cd0b5b80ab9c9268a04a8b6bd54ad7232c29301a1f6a6392fcd38ecedf
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-node-linux-s390x.tar.gz) | 319e684340aab739e3da46c6407851ff1c42463ba176bf190e58faa48d143975f02df3443ac287cdfcf652b5d6b6e6721d9e4f35995c4e705297a97dd777fe7e
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.3/kubernetes-node-windows-amd64.tar.gz) | 1ff22497a3f0844ffa8593a2a444a8fcb45d0123da49fd58e17cfc1477d22be7f6809d923898b6aa7a9ce519b0a6e0825f575f6cf71da5c8a0fa5f6b4d0905b6

## Changelog since v1.19.0-rc.2

## Changes by Kind

### API Change

- Adds the ability to disable Accelerator/GPU metrics collected by Kubelet ([#91930](https://github.com/kubernetes/kubernetes/pull/91930), [@RenaudWasTaken](https://github.com/RenaudWasTaken)) [SIG Node]
- Kubernetes is now built with golang 1.15.0-rc.1.
  - The deprecated, legacy behavior of treating the CommonName field on X.509 serving certificates as a host name when no Subject Alternative Names are present is now disabled by default. It can be temporarily re-enabled by adding the value x509ignoreCN=0 to the GODEBUG environment variable. ([#93264](https://github.com/kubernetes/kubernetes/pull/93264), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Storage and Testing]

### Bug or Regression

- Azure: per VMSS VMSS VMs cache to prevent throttling on clusters having many attached VMSS ([#93107](https://github.com/kubernetes/kubernetes/pull/93107), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- Extended DSR loadbalancer feature in winkernel kube-proxy to HNS versions 9.3-9.max, 10.2+ ([#93080](https://github.com/kubernetes/kubernetes/pull/93080), [@elweb9858](https://github.com/elweb9858)) [SIG Network]
- Fix instance not found issues when an Azure Node is recreated in a short time ([#93316](https://github.com/kubernetes/kubernetes/pull/93316), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]

## Dependencies

### Added
- github.com/yuin/goldmark: [v1.1.27](https://github.com/yuin/goldmark/tree/v1.1.27)

### Changed
- github.com/Microsoft/hcsshim: [v0.8.9 → 5eafd15](https://github.com/Microsoft/hcsshim/compare/v0.8.9...5eafd15)
- github.com/containerd/cgroups: [bf292b2 → 0dbf7f0](https://github.com/containerd/cgroups/compare/bf292b2...0dbf7f0)
- github.com/urfave/cli: [v1.22.1 → v1.22.2](https://github.com/urfave/cli/compare/v1.22.1...v1.22.2)
- golang.org/x/crypto: bac4c82 → 75b2880
- golang.org/x/mod: v0.1.0 → v0.3.0
- golang.org/x/net: d3edc99 → ab34263
- golang.org/x/tools: c00d67e → c1934b7

### Removed
- github.com/godbus/dbus: [ade71ed](https://github.com/godbus/dbus/tree/ade71ed)



# v1.19.0-rc.2


## Downloads for v1.19.0-rc.2

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes.tar.gz) | 7a9fa6af3772be18f8c427d8b96836bd77e271a08fffeba92d01b3fac4bd69d2be1bbc404cdd4fc259dda42b16790a7943eddb7c889b918d7631857e127a724c
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-src.tar.gz) | 60184627a181ac99cd914acb0ba61c22f31b315ef13be5504f3cb43dea1fa84abb2142c8a1ba9e98e037e0d9d2765e8d85bd12903b03a86538d7638ceb6ac5c9

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-darwin-386.tar.gz) | 03332cd70ce6a9c8e533d93d694da32b549bef486cf73c649bcb1c85fc314b0ac0f95e035de7b54c81112c1ac39029abeb8f246d359384bde2119ea5ea3ebe66
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-darwin-amd64.tar.gz) | e82c2908366cc27cbc1d72f89fdc13414b484dfdf26c39c6180bf2e5734169cc97d77a2d1ac051cdb153582a38f4805e5c5b5b8eb88022c914ffb4ef2a8202d3
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-linux-386.tar.gz) | 948be72e8162ee109c670a88c443ba0907acfd0ffb64df62afe41762717bc2fb9308cbc4eb2012a14e0203197e8576e3700ad9f105379841d46acafad2a4c6dc
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-linux-amd64.tar.gz) | 54e1980b6967dab1e70be2b4df0cd0171f04c92f12dcdf80908b087facb9d5cc1399a7d9455a4a799daa8e9d48b6ad86cb3666a131e5adfcd36b008d25138fa3
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-linux-arm.tar.gz) | 4edcd2e1a866a16b8b0f6228f93b4a61cdd43dca36dcb53a5dbd865cc5a143ef6fd3b8575925acc8af17cff21dee993df9b88db5724320e7b420ca9d0427677f
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-linux-arm64.tar.gz) | 138b215e35cfb5d05bda766763e92352171e018a090d516dbf0c280588c5e6f36228163a75a8147c7bac46e773ad0353daaf550d8fa0e91b1e05c5bc0242531c
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-linux-ppc64le.tar.gz) | 3b8e7f5f1f2e34df5dbb06c12f07f660a2a732846c56d0f4b0a939b8121361d381325565bdda3182ef8951f4c2513a2c255940f97011034063ffb506d5aedeab
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-linux-s390x.tar.gz) | b695cc0695bd324c51084e64bea113aaad3c0b5ba44b5d122db9da6e359a4108008a80944cbe96c405bd2cf57f5f31b3eaf50f33c23d980bdb9f272937c88d1c
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-windows-386.tar.gz) | 8e370a66545cdebe0ae0816afe361c7579c7c6e8ee5652e4e01c6fcc3d6d2a6557101be39be24ceb14302fb30855730894a17f6ae11586759257f12406c653e2
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-client-windows-amd64.tar.gz) | 89e0fe5aac33c991891b08e5a3891ecbda3e038f0ee6a5cdd771ea294ec84292bd5f65f1a895f23e6892ec28f001f66d0166d204bf135cb1aa467ae56ccc1260

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-server-linux-amd64.tar.gz) | 2b0a1b107bf31913d9deec57eab9d3db2ea512c995ce3b4fe247f91c36fdebcc4484a2f8ff53d40a5bc6a04c7144827b85b40ac90c46a9b0cec8a680700f1b1c
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-server-linux-arm.tar.gz) | 2f1ab3bcacd82a9b6d92e09b7cdd63f57fc44623cdfb517512b634264fed87999d78b8571c7930839381b1ed4793b68343e85956d7a8c5bae77ba8f8ade06afa
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-server-linux-arm64.tar.gz) | ea67613c8356f650891a096881546afb27f00e86a9c777617817583628d38b4725f0f65da3b0732414cbc8f97316b3029a355177342a4b1d94cf02d79542e4cd
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-server-linux-ppc64le.tar.gz) | d1b151f3f47c28ead2304d2477fa25f24d12e3fd80e9d1b3b593db99b9a1c5821db4d089f4f1dd041796ea3fd814000c225a7e75aac1e5891a4e16517bcaceee
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-server-linux-s390x.tar.gz) | 69bf215fdc3ad53834eaa9a918452feb0803dffe381b6e03b73141364a697a576e5ed0242d448616707cb386190c21564fe89f8cf3409a7c621a86d86b2c7680

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-node-linux-amd64.tar.gz) | 88ae137316bab3bb1dcb6c78a4d725face618d41714400505b97ce9d3fa37a6caa036b9e8508ade6dd679e3a8c483a32aef9e400ab08d86b6bf39bc13f34e435
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-node-linux-arm.tar.gz) | 7eaaf2a2a4ee5181cb4c1567e99b8bf82a3da342799f5d2b34dd7f133313c3e3d2ac5a778110e178161788cb226bd64836fba35fbec21c8384e7725cae9b756c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-node-linux-arm64.tar.gz) | 4f0ef95abc52da0e5d0c40434f8c324ddfb218a577114c4ead00f2ac1c01439922aee6fe347f702927a73b0166cd8b9f4c491d3a18a1a951d24c9ea7259d2655
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-node-linux-ppc64le.tar.gz) | 0424896e2fedae3a566a5aa2e4af26977a578066d49e3ad66307839c2d2dd1c53d1afcf16b2f6cebf0c74d2d60dbc118e6446d9c02aaab27e95b3a6d26889f51
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-node-linux-s390x.tar.gz) | 294788687a6e6d1ca2e4d56435b1174e4330abe64cc58b1372c3b9caaab4455586da4e3bfc62616b52ea7d678561fb77ce1f8d0023fd7d1e75e1db348c69939c
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.2/kubernetes-node-windows-amd64.tar.gz) | 61389f8c05c682102e3432a2f05f41b11d531124f61443429627f94ef6e970d44240d44d32aa467b814de0b54a17208b2d2696602ba5dd6d30f64db964900230

## Changelog since v1.19.0-rc.1

## Changes by Kind

### API Change

- A new alpha-level field, `SupportsFsGroup`, has been introduced for CSIDrivers to allow them to specify whether they support volume ownership and permission modifications. The `CSIVolumeSupportFSGroup` feature gate must be enabled to allow this field to be used. ([#92001](https://github.com/kubernetes/kubernetes/pull/92001), [@huffmanca](https://github.com/huffmanca)) [SIG API Machinery, CLI and Storage]
- The kube-controller-manager managed signers can now have distinct signing certificates and keys.  See the help about `--cluster-signing-[signer-name]-{cert,key}-file`.  `--cluster-signing-{cert,key}-file` is still the default. ([#90822](https://github.com/kubernetes/kubernetes/pull/90822), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Apps and Auth]

### Feature

- Added kube-apiserver metrics: apiserver_current_inflight_request_measures and, when API Priority and Fairness is enable, windowed_request_stats. ([#91177](https://github.com/kubernetes/kubernetes/pull/91177), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery, Instrumentation and Testing]
- Rename pod_preemption_metrics to preemption_metrics. ([#93256](https://github.com/kubernetes/kubernetes/pull/93256), [@ahg-g](https://github.com/ahg-g)) [SIG Instrumentation and Scheduling]

### Bug or Regression

- Do not add nodes labeled with kubernetes.azure.com/managed=false to backend pool of load balancer. ([#93034](https://github.com/kubernetes/kubernetes/pull/93034), [@matthias50](https://github.com/matthias50)) [SIG Cloud Provider]
- Do not retry volume expansion if CSI driver returns FailedPrecondition error ([#92986](https://github.com/kubernetes/kubernetes/pull/92986), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Fix: determine the correct ip config based on ip family ([#93043](https://github.com/kubernetes/kubernetes/pull/93043), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Fix: initial delay in mounting azure disk & file ([#93052](https://github.com/kubernetes/kubernetes/pull/93052), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fixed the EndpointSliceController to correctly create endpoints for IPv6-only pods.
  
  Fixed the EndpointController to allow IPv6 headless services, if the IPv6DualStack
  feature gate is enabled, by specifying `ipFamily: IPv6` on the service. (This already
  worked with the EndpointSliceController.) ([#91399](https://github.com/kubernetes/kubernetes/pull/91399), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]

### Other (Cleanup or Flake)

- Kube-up: defaults to limiting critical pods to the kube-system namespace to match behavior prior to 1.17 ([#93121](https://github.com/kubernetes/kubernetes/pull/93121), [@liggitt](https://github.com/liggitt)) [SIG Cloud Provider and Scheduling]
- Update Golang to v1.14.5
  - Update repo-infra to 0.0.7 (to support go1.14.5 and go1.13.13)
    - Includes:
      - bazelbuild/bazel-toolchains@3.3.2
      - bazelbuild/rules_go@v0.22.7 ([#93088](https://github.com/kubernetes/kubernetes/pull/93088), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Update Golang to v1.14.6
  - Update repo-infra to 0.0.8 (to support go1.14.6 and go1.13.14)
    - Includes:
      - bazelbuild/bazel-toolchains@3.4.0
      - bazelbuild/rules_go@v0.22.8 ([#93198](https://github.com/kubernetes/kubernetes/pull/93198), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Update default etcd server version to 3.4.9 ([#92349](https://github.com/kubernetes/kubernetes/pull/92349), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
- go.etcd.io/etcd: 54ba958 → 18dfb9c
- k8s.io/utils: 6e3d28b → 0bdb4ca

### Removed
_Nothing has changed._



# v1.19.0-rc.1


## Downloads for v1.19.0-rc.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes.tar.gz) | d4bc1d86ff77a1a8695091207b8181a246c8964ae1dd8967392aae95197c0339c7915a016c017ecab0b9d203b3205221ca766ce568d7ee52947e7f50f057af4f
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-src.tar.gz) | 79af4e01b0d5432f92b026730a0c60523069d312858c30fdcaeaf6ee159c71f3413a5075d82c0acd9b135b7a06d5ecb0c0d38b8a8d0f301a9d9bffb35d22f029

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-darwin-386.tar.gz) | 7d21bf9733810659576e67986d129208894adea3c571de662dbf80fb822e18abfc1644ea60a4e5fbe244a23b56aa973b76dafe789ead1bf7539f41bdd9bca886
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | b4622e06c09bb08a0dc0115bfcd991c50459c7b772889820648ed1c05a425605d10b71b92c58c119b77baa3bca209f9c75827d2cde69d128a5cfcada5f37be39
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-linux-386.tar.gz) | f51032ad605543f68a2a4da3bede1f3e7be0dd63b03b751fef5f133e8d64bec02bfe7433b75e3d0c4ae122d4e0cf009095800c638d2cc81f6fb81b488f5a6dab
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | 48489d22969f69a5015988e596d597c64ea18675649afe55ad119dbbe98ba9a4104d5e323704cf1f3cbdfca3feac629d3813e260a330a72da12f1a794d054f76
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-linux-arm.tar.gz) | d9f8a6f6f3d09be9c08588c2b5153a4d8cc9db496bde3da2f3af472c260566d1391cd8811f2c05d4f302db849a38432f25228d9bbb59aaaf0dfba64b33f8ee8e
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | 1c3590750a3f02e0e5845e1135cc3ab990309bdecfe64c089842a134eae57b573488531696796185ed12dde2d6f95d2e3656dd9893d04cd0adbe025513ffff30
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | 158a562d5dbbe90cd56b5d757823adda1919e9b5db8005fb6e2523358e5a20628d55ec1903c0e317a0d8ac9b9a649eea23d9ea746db22b73d6d580ae8c067077
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | 47c140567dc593caf065f295ed6006efcde010a526a96c8d3ef5f3d9a9dc6b413bc197dc822648067fe16299908ada7046c2a8a3213d4296b04b51a264ad40e9
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-windows-386.tar.gz) | e25d7d4ad3e6f6e6cfba181c5871e56de2751f88b640502745f6693ddd86ccd7eef8aebaa30955afdbbd0320a5b51d4e9e17f71baab37a470aac284178a0e21c
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | fd8463b04b5d7f115104245fa1dd53de6656b349dad4cfd55f239012d4f2c1a8e35aa3f3554138df9ddfea9d7702b51a249f6db698c0cea7c36e5bc98a017fe7

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | 96acce78bba3779bef616de28da5d75bc4dc0b52fe0bf03b169c469ade9a8cd38b19c4620d222d67bff9ceeb0c5ebf893f55c1de02356bcebe5689890d0478f7
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-server-linux-arm.tar.gz) | 1e561f3edbc66d2ab7f6f1ffe8dc1c01cec13ee3ba700458bd5d87202723cc832f3305a864a3b569463c96d60c9f60c03b77f210663cc40589e40515b3a32e75
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | ba8fc011ac0e54cb1a0e0e3ee5f1cff4d877f4fdd75e15bf25b1cf817b3cf2bc85f9809d3cc76e9145f07a837960843ca68bdf02fe970c0043fc9ff7b53da021
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | 1f506676284ab2f6bd3fc8a29a062f4fddf5346ef30be9363f640467c64011144381180c5bf74ec885d2f54524e82e21c745c5d2f1b191948bc40db2a09a2900
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | 5a7101288d51297c3346d028176b4b997afd8652d6481cec82f8863a91209fec6e8a9286a9bd7543b428e6ef82c1c68a7ce0782191c4682634015a032f749554

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | 6852edc9818cb51a7e738e44a8bca2290777320e62518c024962fddd05f7ef390fb5696537068fd75e340bae909602f0bbc2aa5ebf6c487c7b1e990250f16810
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-node-linux-arm.tar.gz) | f13edad4684d0de61e4cd7e524f891c75e0efe1050911d9bf0ee3a77cac28f57dca68fb990df6b5d9646e9b389527cbb861de10e12a84e57788f339db05936cb
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | 69480150325525459aed212b8c96cb1865598cb5ecbeb57741134142d65e8a96258ec298b86d533ce88d2c499c4ad17e66dd3f0f7b5e9c34882889e9cb384805
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | 774cfa9a4319ede166674d3e8c46900c9319d98ffba5b46684244e4bb15d94d31df8a6681e4711bc744d7e92fd23f207505eda98f43c8e2383107badbd43f289
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | 7e696988febb1e913129353134191b23c6aa5b0bea7c9c9168116596b827c091a88049ca8b8847dda25ecd4467cca4cc48cae8699635b5e78b83aab482c109f5
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | 067182292d9e17f0d4974051681bedcf5ed6017dc80485541f89ea1f211085714165941a5262a4997b7bfc2bd190f2255df4c1b39f86a3278487248111d83cd4

## Changelog since v1.19.0-rc.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Azure blob disk feature(`kind`: `Shared`, `Dedicated`) has been deprecated, you should use `kind`: `Managed` in `kubernetes.io/azure-disk` storage class. ([#92905](https://github.com/kubernetes/kubernetes/pull/92905), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
 
## Changes by Kind

### Deprecation

- Kubeadm: deprecate the "kubeadm alpha kubelet config enable-dynamic" command. To continue using the feature please defer to the guide for "Dynamic Kubelet Configuration" at k8s.io. ([#92881](https://github.com/kubernetes/kubernetes/pull/92881), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]

### API Change

- Added pod version skew strategy for seccomp profile to synchronize the deprecated annotations with the new API Server fields. Please see the corresponding section [in the KEP](https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/20190717-seccomp-ga.mdversion-skew-strategy) for more detailed explanations. ([#91408](https://github.com/kubernetes/kubernetes/pull/91408), [@saschagrunert](https://github.com/saschagrunert)) [SIG Apps, Auth, CLI and Node]
- Custom Endpoints are now mirrored to EndpointSlices by a new EndpointSliceMirroring controller. ([#91637](https://github.com/kubernetes/kubernetes/pull/91637), [@robscott](https://github.com/robscott)) [SIG API Machinery, Apps, Auth, Cloud Provider, Instrumentation, Network and Testing]
- Generic ephemeral volumes, a new alpha feature under the `GenericEphemeralVolume` feature gate, provide a more flexible alternative to `EmptyDir` volumes: as with `EmptyDir`, volumes are created and deleted for each pod automatically by Kubernetes. But because the normal provisioning process is used (`PersistentVolumeClaim`), storage can be provided by third-party storage vendors and all of the usual volume features work. Volumes don't need to be empt; for example, restoring from snapshot is supported. ([#92784](https://github.com/kubernetes/kubernetes/pull/92784), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Instrumentation, Node, Scheduling, Storage and Testing]

### Feature

- ACTION REQUIRED : In CoreDNS v1.7.0, [metrics names have been changed](https://github.com/coredns/coredns/blob/master/notes/coredns-1.7.0.md&#35;metric-changes) which will be backward incompatible with existing reporting formulas that use the old metrics' names. Adjust your formulas to the new names before upgrading. 
  
  Kubeadm now includes CoreDNS version v1.7.0. Some of the major changes include:
  -  Fixed a bug that could cause CoreDNS to stop updating service records.
  -  Fixed a bug in the forward plugin where only the first upstream server is always selected no matter which policy is set.
  -  Remove already deprecated options `resyncperiod` and `upstream` in the Kubernetes plugin.
  -  Includes Prometheus metrics name changes (to bring them in line with standard Prometheus metrics naming convention). They will be backward incompatible with existing reporting formulas that use the old metrics' names.
  -  The federation plugin (allows for v1 Kubernetes federation) has been removed.
  More details are available in https://coredns.io/2020/06/15/coredns-1.7.0-release/ ([#92651](https://github.com/kubernetes/kubernetes/pull/92651), [@rajansandeep](https://github.com/rajansandeep)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Add tags support for Azure File Driver ([#92825](https://github.com/kubernetes/kubernetes/pull/92825), [@ZeroMagic](https://github.com/ZeroMagic)) [SIG Cloud Provider and Storage]
- Audit events for API requests to deprecated API versions now include a `"k8s.io/deprecated": "true"` audit annotation. If a target removal release is identified, the audit event includes a `"k8s.io/removal-release": "<majorVersion>.<minorVersion>"` audit annotation as well. ([#92842](https://github.com/kubernetes/kubernetes/pull/92842), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Instrumentation]
- Cloud node-controller use InstancesV2 ([#91319](https://github.com/kubernetes/kubernetes/pull/91319), [@gongguan](https://github.com/gongguan)) [SIG Apps, Cloud Provider, Scalability and Storage]
- Kubeadm: deprecate the "--csr-only" and "--csr-dir" flags of the "kubeadm init phase certs" subcommands. Please use "kubeadm alpha certs generate-csr" instead. This new command allows you to generate new private keys and certificate signing requests for all the control-plane components, so that the certificates can be signed by an external CA. ([#92183](https://github.com/kubernetes/kubernetes/pull/92183), [@wallrj](https://github.com/wallrj)) [SIG Cluster Lifecycle]
- Server-side apply behavior has been regularized in the case where a field is removed from the applied configuration. Removed fields which have no other owners are deleted from the live object, or reset to their default value if they have one. Safe ownership transfers, such as the transfer of a `replicas` field from a user to an HPA without resetting to the default value are documented in [Transferring Ownership](https://kubernetes.io/docs/reference/using-api/api-concepts/&#35;transferring-ownership) ([#92661](https://github.com/kubernetes/kubernetes/pull/92661), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Testing]
- Set CSIMigrationvSphere feature gates to beta.
  Users should enable CSIMigration + CSIMigrationvSphere features and install the vSphere CSI Driver (https://github.com/kubernetes-sigs/vsphere-csi-driver) to move workload from the in-tree vSphere plugin "kubernetes.io/vsphere-volume" to vSphere CSI Driver.
  
  Requires: vSphere vCenter/ESXi Version: 7.0u1, HW Version: VM version 15 ([#92816](https://github.com/kubernetes/kubernetes/pull/92816), [@divyenpatel](https://github.com/divyenpatel)) [SIG Cloud Provider and Storage]
- Support a smooth upgrade from client-side apply to server-side apply without conflicts, as well as support the corresponding downgrade. ([#90187](https://github.com/kubernetes/kubernetes/pull/90187), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG API Machinery and Testing]
- Trace output in apiserver logs is more organized and comprehensive. Traces are nested, and for all non-long running request endpoints, the entire filter chain is instrumented (e.g. authentication check is included). ([#88936](https://github.com/kubernetes/kubernetes/pull/88936), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Scheduling]
- `kubectl alpha debug` now supports debugging nodes by creating a debugging container running in the node's host namespaces. ([#92310](https://github.com/kubernetes/kubernetes/pull/92310), [@verb](https://github.com/verb)) [SIG CLI]

### Failing Test

- Kube-proxy iptables min-sync-period defaults to 1 sec. Previously, it was 0. ([#92836](https://github.com/kubernetes/kubernetes/pull/92836), [@aojea](https://github.com/aojea)) [SIG Network]

### Bug or Regression

- Dockershim security: pod sandbox now always run with `no-new-privileges` and `runtime/default` seccomp profile
  dockershim seccomp: custom profiles can now have smaller seccomp profiles when set at pod level ([#90948](https://github.com/kubernetes/kubernetes/pull/90948), [@pjbgf](https://github.com/pjbgf)) [SIG Node]
- Eviction requests for pods that have a non-zero DeletionTimestamp will always succeed ([#91342](https://github.com/kubernetes/kubernetes/pull/91342), [@michaelgugino](https://github.com/michaelgugino)) [SIG Apps]
- Fix detection of image filesystem, disk metrics for devicemapper, detection of OOM Kills on 5.0+ linux kernels. ([#92919](https://github.com/kubernetes/kubernetes/pull/92919), [@dashpole](https://github.com/dashpole)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Node]
- Fixed memory leak in endpointSliceTracker ([#92838](https://github.com/kubernetes/kubernetes/pull/92838), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Kube-aggregator certificates are dynamically loaded on change from disk ([#92791](https://github.com/kubernetes/kubernetes/pull/92791), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kube-up now includes CoreDNS version v1.7.0. Some of the major changes include:
  -  Fixed a bug that could cause CoreDNS to stop updating service records.
  -  Fixed a bug in the forward plugin where only the first upstream server is always selected no matter which policy is set.
  -  Remove already deprecated options `resyncperiod` and `upstream` in the Kubernetes plugin.
  -  Includes Prometheus metrics name changes (to bring them in line with standard Prometheus metrics naming convention). They will be backward incompatible with existing reporting formulas that use the old metrics' names.
  -  The federation plugin (allows for v1 Kubernetes federation) has been removed.
  More details are available in https://coredns.io/2020/06/15/coredns-1.7.0-release/ ([#92718](https://github.com/kubernetes/kubernetes/pull/92718), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cloud Provider]
- The apiserver will no longer proxy non-101 responses for upgrade requests. This could break proxied backends (such as an extension API server) that respond to upgrade requests with a non-101 response code. ([#92941](https://github.com/kubernetes/kubernetes/pull/92941), [@tallclair](https://github.com/tallclair)) [SIG API Machinery]
- The terminationGracePeriodSeconds from pod spec is respected for the mirror pod. ([#92442](https://github.com/kubernetes/kubernetes/pull/92442), [@tedyu](https://github.com/tedyu)) [SIG Node and Testing]

### Other (Cleanup or Flake)

- --cache-dir sets cache directory for both http and discovery, defaults to $HOME/.kube/cache ([#92910](https://github.com/kubernetes/kubernetes/pull/92910), [@soltysh](https://github.com/soltysh)) [SIG API Machinery and CLI]
- Fix: license issue in blob disk feature ([#92824](https://github.com/kubernetes/kubernetes/pull/92824), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/cilium/ebpf: [9f1617e → 1c8d4c9](https://github.com/cilium/ebpf/compare/9f1617e...1c8d4c9)
- github.com/coredns/corefile-migration: [v1.0.8 → v1.0.10](https://github.com/coredns/corefile-migration/compare/v1.0.8...v1.0.10)
- github.com/google/cadvisor: [8450c56 → v0.37.0](https://github.com/google/cadvisor/compare/8450c56...v0.37.0)
- github.com/json-iterator/go: [v1.1.9 → v1.1.10](https://github.com/json-iterator/go/compare/v1.1.9...v1.1.10)
- github.com/opencontainers/runc: [1b94395 → 819fcc6](https://github.com/opencontainers/runc/compare/1b94395...819fcc6)
- github.com/prometheus/client_golang: [v1.6.0 → v1.7.1](https://github.com/prometheus/client_golang/compare/v1.6.0...v1.7.1)
- github.com/prometheus/common: [v0.9.1 → v0.10.0](https://github.com/prometheus/common/compare/v0.9.1...v0.10.0)
- github.com/prometheus/procfs: [v0.0.11 → v0.1.3](https://github.com/prometheus/procfs/compare/v0.0.11...v0.1.3)
- github.com/rubiojr/go-vhd: [0bfd3b3 → 02e2102](https://github.com/rubiojr/go-vhd/compare/0bfd3b3...02e2102)
- sigs.k8s.io/structured-merge-diff/v3: v3.0.0 → 43c19bb

### Removed
_Nothing has changed._



# v1.19.0-beta.2


## Downloads for v1.19.0-beta.2

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes.tar.gz) | 806c1734a57dfc1800730fcb25aeb60d50d19d248c0e2a92ede4b6c4565745b4f370d4fd925bef302a96fba89102b7560b8f067240e0f35f6ec6caa29971dea4
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-src.tar.gz) | 507372c6d7ea380ec68ea237141a2b62953a2e1d1d16288f37820b605e33778c5f43ac5a3dedf39f7907d501749916221a8fa4d50be1e5a90b3ce23d36eaa075

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-darwin-386.tar.gz) | 6d20ca8d37b01213dcb98a1e49d44d414043ce485ae7df9565dfb7914acb1ec42b7aeb0c503b8febc122a8b444c6ed13eec0ff3c88033c6db767e7af5dbbc65d
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | e9caa5463a662869cfc8b9254302641aee9b53fa2119244bd65ef2c66e8c617f7db9b194a672ff80d7bc42256e6560db9fe8a00b2214c0ef023e2d6feed58a3a
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-linux-386.tar.gz) | 48296417fcd2c2f6d01c30dcf66956401ea46455c52a2bbd76feb9b117502ceaa2fb10dae944e087e7038b9fdae5b835497213894760ca01698eb892087490d2
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | e2cc7819974316419a8973f0d77050b3262c4e8d078946ff9f6f013d052ec1dd82893313feff6e4493ae0fd3fb62310e6ce4de49ba6e80f8b9979650debf53f2
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-linux-arm.tar.gz) | 484aac48a7a736970ea0766547453b7d37b25ed29fdee771734973e3e080b33f6731eecc458647db962290b512d32546e675e4658287ced3214e87292b98a643
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | f793078dc2333825a6679126b279cb0a3415ded8c650478e73c37735c6aa9576b68b2a4165bb77ef475884d50563ea236d8db4c72b2e5552b5418ea06268daae
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | 4c204b8d3b2717470ee460230b6bdc63725402ad3d24789397934bfe077b94d68041a376864b618e01f541b5bd00d0e63d75aa531a327ab0082c01eb4b9aa5ee
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | d0f6e4ddbf122ebcb4c5a980d5f8e33a23213cb438983341870f288afd17e73ec42f0ded55a3a9622c57700e68999228508d449ca206aca85f3254f7622375db
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-windows-386.tar.gz) | a615a7821bba1f8e4115b7981347ed94a79947c78d32c692cd600e21e0de29fedfc4a39dc08ca516f2f35261cf4a6d6ce557008f034e0e1d311fa9e75478ec0c
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | 34046130c5ebb3afe17e6e3cf88229b8d3281a9ac9c28dece1fd2d49a11b7be011700b74d9b8111dee7d0943e5ebfa208185bae095c2571aa54e0f9201e2cddd

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | c922058ce9c665e329d3d4647aac5d2dd22d9a8af63a21e6af98943dfd14f2b90268c53876f42a64093b96499ee1109803868c9aead4c15fd8db4b1bbec58fd9
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-server-linux-arm.tar.gz) | 4f17489b946dc04570bfab87015f2c2401b139b9ee745ed659bc94ccd116f3f23e249f83e19aaa418aa980874fffb478b1ec7340aa25292af758c9eabd4c2022
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | 69e44a63d15962de95a484e311130d415ebfec16a9da54989afc53a835c5b67de20911d71485950d07259a0f8286a299f4d74f90c73530e905da8dc60e391597
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | 66b30ebad7a8226304150aa42a1bd660a0b3975fecbfd8dbbea3092936454d9f81c8083841cc67c6645ab771383b66c7f980dd65319803078c91436c55d5217a
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | 0e197280f99654ec9e18ea01a9fc848449213ce28521943bc5d593dd2cac65310b6a918f611ea283b3a0377347eb718e99dd59224b8fad8adb223d483fa9fecb

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-node-linux-amd64.tar.gz) | f40afee38155c5163ba92e3fa3973263ca975f3b72ac18535799fb29180413542ef86f09c87681161affeef94eb0bd38e7cf571a73ab0f51a88420f1aedeaeec
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-node-linux-arm.tar.gz) | 6088b11767b77f0ec932a9f1aee9f0c7795c3627529f259edf4d8b1be2e1a324a75c89caed65c6aa277c2fd6ee23b3ebeb05901f351cd2dde0a833bbbd6d6d07
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-node-linux-arm64.tar.gz) | e790c491d057721b94d0d2ad22dd5c75400e8602e95276471f20cd2181f52c5be38e66b445d8360e1fb671627217eb0b7735b485715844d0e9908cf3de249464
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-node-linux-ppc64le.tar.gz) | 04f696cfab66f92b4b22c23807a49c344d6a157a9ac3284a267613369b7f9f5887f67902cb8a2949caa204f89fdc65fe442a03c2c454013523f81b56476d39a0
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-node-linux-s390x.tar.gz) | c671e20f69f70ec567fb16bbed2fecac3099998a3365def1e0755e41509531fd65768f7a04015b27b17e6a5884e65cddb82ff30a8374ed011c5e2008817259db
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.2/kubernetes-node-windows-amd64.tar.gz) | 23d712fb2d455b5095d31b9c280d92442f7871786808528a1b39b9babf169dc7ae467f1ee2b2820089d69aa2342441d0290edf4f710808c78277e612f870321d

## Changelog since v1.19.0-beta.1

## Changes by Kind

### Deprecation

- Kubeadm: remove the deprecated "--use-api" flag for "kubeadm alpha certs renew" ([#90143](https://github.com/kubernetes/kubernetes/pull/90143), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Scheduler's alpha feature 'ResourceLimitsPriorityFunction' is completely removed due to lack of usage ([#91883](https://github.com/kubernetes/kubernetes/pull/91883), [@SataQiu](https://github.com/SataQiu)) [SIG Scheduling and Testing]

### API Change

- Remove `BindTimeoutSeconds` from schedule configuration `KubeSchedulerConfiguration` ([#91580](https://github.com/kubernetes/kubernetes/pull/91580), [@cofyc](https://github.com/cofyc)) [SIG Scheduling and Testing]
- Resolve regression in metadata.managedFields handling in update/patch requests submitted by older API clients ([#91748](https://github.com/kubernetes/kubernetes/pull/91748), [@apelisse](https://github.com/apelisse)) [SIG API Machinery and Testing]
- The CertificateSigningRequest API is promoted to certificates.k8s.io/v1 with the following changes:
  - `spec.signerName` is now required, and requests for `kubernetes.io/legacy-unknown` are not allowed to be created via the `certificates.k8s.io/v1` API
  - `spec.usages` is now required, may not contain duplicate values, and must only contain known usages
  - `status.conditions` may not contain duplicate types
  - `status.conditions[*].status` is now required
  - `status.certificate` must be PEM-encoded, and contain only CERTIFICATE blocks ([#91685](https://github.com/kubernetes/kubernetes/pull/91685), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Architecture, Auth, CLI and Testing]
- The Kubelet's `--cloud-provider` and `--cloud-config` options are now marked as deprecated. ([#90408](https://github.com/kubernetes/kubernetes/pull/90408), [@knabben](https://github.com/knabben)) [SIG Cloud Provider and Node]

### Feature

- A new extension point `PostFilter` is introduced to scheduler framework which runs after Filter phase to resolve scheduling filter failures. A typical implementation is running preemption logic. ([#91314](https://github.com/kubernetes/kubernetes/pull/91314), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Added --privileged flag to kubectl run ([#90569](https://github.com/kubernetes/kubernetes/pull/90569), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Enable feature Gate DefaultPodTopologySpread to use PodTopologySpread plugin to do defaultspreading. In doing so, legacy DefaultPodTopologySpread plugin is disabled. ([#91793](https://github.com/kubernetes/kubernetes/pull/91793), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Extend AWS azToRegion method to support Local Zones ([#90874](https://github.com/kubernetes/kubernetes/pull/90874), [@Jeffwan](https://github.com/Jeffwan)) [SIG Cloud Provider]
- Kube-Proxy now supports IPv6DualStack on Windows with the IPv6DualStack feature gate. ([#90853](https://github.com/kubernetes/kubernetes/pull/90853), [@kumarvin123](https://github.com/kumarvin123)) [SIG Network, Node and Windows]
- Kube-controller-manager: the `--experimental-cluster-signing-duration` flag is marked as deprecated for removal in v1.22, and is replaced with `--cluster-signing-duration`. ([#91154](https://github.com/kubernetes/kubernetes/pull/91154), [@liggitt](https://github.com/liggitt)) [SIG Auth and Cloud Provider]
- Support kubectl create deployment with replicas ([#91562](https://github.com/kubernetes/kubernetes/pull/91562), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- The RotateKubeletClientCertificate feature gate has been promoted to GA, and the kubelet --feature-gate RotateKubeletClientCertificate parameter will be removed in 1.20. ([#91780](https://github.com/kubernetes/kubernetes/pull/91780), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
- The metric label name of `kubernetes_build_info` has been updated from `camel case` to `snake case`:
  - gitVersion --> git_version
  - gitCommit --> git_commit
  - gitTreeState --> git_tree_state
  - buildDate --> build_date
  - goVersion --> go_version
  
  This change happens in `kube-apiserver`、`kube-scheduler`、`kube-proxy` and `kube-controller-manager`. ([#91805](https://github.com/kubernetes/kubernetes/pull/91805), [@RainbowMango](https://github.com/RainbowMango)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- `EventRecorder()` is exposed to `FrameworkHandle` interface so that scheduler plugin developers can choose to log cluster-level events. ([#92010](https://github.com/kubernetes/kubernetes/pull/92010), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]

### Bug or Regression

- Azure: set dest prefix and port for IPv6 inbound security rule ([#91831](https://github.com/kubernetes/kubernetes/pull/91831), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Fix etcd version migration script in etcd image. ([#91925](https://github.com/kubernetes/kubernetes/pull/91925), [@wenjiaswe](https://github.com/wenjiaswe)) [SIG API Machinery]
- Fix issues when supported huge page sizes changes ([#80831](https://github.com/kubernetes/kubernetes/pull/80831), [@odinuge](https://github.com/odinuge)) [SIG Node and Testing]
- Fix kubectl describe output format for empty annotations. ([#91405](https://github.com/kubernetes/kubernetes/pull/91405), [@iyashu](https://github.com/iyashu)) [SIG CLI]
- Fixed an issue that a Pod's nominatedNodeName cannot be cleared upon node deletion. ([#91750](https://github.com/kubernetes/kubernetes/pull/91750), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fixed several bugs involving the IPFamily field when creating or updating services
  in clusters with the IPv6DualStack feature gate enabled.
  
  Beware that the behavior of the IPFamily field is strange and inconsistent and will
  likely be changed before the dual-stack feature goes GA. Users should treat the
  field as "write-only" for now and should not make any assumptions about a service
  based on its current IPFamily value. ([#91400](https://github.com/kubernetes/kubernetes/pull/91400), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]
- Kube-apiserver: fixes scale subresource patch handling to avoid returning unnecessary 409 Conflict error to clients ([#90342](https://github.com/kubernetes/kubernetes/pull/90342), [@liggitt](https://github.com/liggitt)) [SIG Apps, Autoscaling and Testing]
- Kube-up: fixes setup of validating admission webhook credential configuration ([#91995](https://github.com/kubernetes/kubernetes/pull/91995), [@liggitt](https://github.com/liggitt)) [SIG Cloud Provider and Cluster Lifecycle]
- Kubeadm: Add retries for kubeadm join / UpdateStatus to make update status more resilient by adding a retry loop to this operation ([#91952](https://github.com/kubernetes/kubernetes/pull/91952), [@xlgao-zju](https://github.com/xlgao-zju)) [SIG Cluster Lifecycle]
- On AWS nodes with multiple network interfaces, kubelet should now more reliably report addresses from secondary interfaces. ([#91889](https://github.com/kubernetes/kubernetes/pull/91889), [@anguslees](https://github.com/anguslees)) [SIG Cloud Provider]
- Resolve regression in metadata.managedFields handling in create/update/patch requests not using server-side apply ([#91690](https://github.com/kubernetes/kubernetes/pull/91690), [@apelisse](https://github.com/apelisse)) [SIG API Machinery and Testing]

### Other (Cleanup or Flake)

- Deprecate the `--target-ram-md` flags that is no longer used for anything. ([#91818](https://github.com/kubernetes/kubernetes/pull/91818), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Replace framework.Failf with ExpectNoError ([#91811](https://github.com/kubernetes/kubernetes/pull/91811), [@lixiaobing1](https://github.com/lixiaobing1)) [SIG Instrumentation, Storage and Testing]
- The Kubelet's `--experimental-allocatable-ignore-eviction` option is now marked as deprecated. ([#91578](https://github.com/kubernetes/kubernetes/pull/91578), [@knabben](https://github.com/knabben)) [SIG Node]
- Update corefile-migration library to 1.0.8 ([#91856](https://github.com/kubernetes/kubernetes/pull/91856), [@wawa0210](https://github.com/wawa0210)) [SIG Node]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/Azure/azure-sdk-for-go: [v40.2.0+incompatible → v43.0.0+incompatible](https://github.com/Azure/azure-sdk-for-go/compare/v40.2.0...v43.0.0)
- github.com/coredns/corefile-migration: [v1.0.6 → v1.0.8](https://github.com/coredns/corefile-migration/compare/v1.0.6...v1.0.8)
- k8s.io/klog/v2: v2.0.0 → v2.1.0

### Removed
_Nothing has changed._



# v1.19.0-beta.1


## Downloads for v1.19.0-beta.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes.tar.gz) | c4ab79e987790fbda842310525abecee60861e44374c414159e60d74e85b4dd36d9d49253b8e7f08aec36a031726f9517d0a401fb748e41835ae2dc86aee069d
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-src.tar.gz) | 08d1aadb8a31b35f3bc39f44d8f97b7e98951f833bb87f485f318c6acfdb53539851fbb2d4565036e00b6f620c5b1882c6f9620759c3b36833da1d6b2b0610f2

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-darwin-386.tar.gz) | 55eb230fdb4e60ded6c456ec6e03363c6d55e145a956aa5eff0c2b38d8ecfe848b4a404169def45d392e747e4d04ee71fe3182ab1e6426110901ccfb2e1bc17f
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | ddc03644081928bd352c40077f2a075961c90a7159964be072b3e05ec170a17d6d78182d90210c18d24d61e75b45eae3d1b1486626db9e28f692dfb33196615c
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-linux-386.tar.gz) | 6e1e00a53289bd9a4d74a61fce4665786051aafe8fef8d1d42de88ba987911bfb7fd5f4a2c3771ae830819546cf9f4badd94fd90c50ca74367c1ace748e8eafd
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | 2c4db87c61bc4a528eb2af9246648fc7a015741fe52f551951fda786c252eca1dc48a4325be70e6f80f1560f773b763242334ad4fe06657af290e610f10bc231
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-linux-arm.tar.gz) | 8a2bebf67cbd8f91ba38edc36a239aa50d3e58187827763eb5778a5ca0d9d35be97e193b794bff415e8f5de071e47659033dc0420e038d78cc32e841a417a62a
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | f2d0029efc03bf17554c01c11e77b161b8956d9da4b17962ca878378169cbdee04722bbda87279f4b7431c91db0e92bfede45dcc6d971f34d3fe891339b7c47b
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | 45eb3fe40951ba152f05aa0fe41b7c17ffb91ee3cecb12ec19d2d9cdb467267c1eb5696660687852da314eb8a14a9ebf5f5da21eca252e1c2e3b18dca151ad0d
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | 2097ac5d593dd0951a34df9bdf7883b5c228da262042904ee3a2ccfd1f9c955ff6a3a59961850053e41646bce8fc70a023efe9e9fe49f14f9a6276c8da22f907
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-windows-386.tar.gz) | c38b034e8ac3a5972a01f36b184fe1a195f6a422a3c6564f1f3faff858b1220173b6ab934e7b7ec200931fd7d9456e947572620d82d02e7b05fc61a7fb67ec70
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | 0501694734381914882836e067dc177e8bccd48a4826e286017dc5f858f27cdef348edbb664dda59162f6cd3ac14a9e491e314a3ea032dec43bc77610ce8c8bc

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | 0dd2058889eabbf0b05b6fafd593997ff9911467f0fc567c142583adf0474f4d0e2f4024b4906ff9ee4264d1cbbfde66596ccb8c73b3d5bb79f67e5eb4b3258a
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-server-linux-arm.tar.gz) | 9c3a33d7c198116386178a4f8ee7d4df82e810d6f26833f19f93eff112c29f9f89e5ee790013ad1d497856ecb2662ee95a49fc6a41f0d33cc67e431d06135b88
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | 11f83132f729bec4a4d84fc1983dbd5ddd1643d000dc74c6e05f35637de21533834a572692fc1281c7b0bd29ee93e721fb00e276983e36c327a1950266b17f6d
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | 949334065d968f10207089db6175dcc4bf9432b3b48b120f689cd39c56562a0f4f60d774c95a20a5391d0467140a4c3cb6b2a2dfedccfda6c20f333a63ebcf81
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | 29e8f6a22969d8ab99bf6d272215f53d8f7a125d0c5c20981dcfe960ed440369f831c71a94bb61974b486421e4e9ed936a9421a1be6f02a40e456daab4995663

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | 3d9767e97a40b501f29bbfc652c8fd841eae1dee22a97fdc20115e670081de7fa8e84f6e1be7bbf2376b59c5eef15fb5291415ae2e24ce4c9c5e141faa38c47c
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-node-linux-arm.tar.gz) | 8ccf401e0bd0c59403af49046b49cf556ff164fca12c5233169a80e18cc4367f404fd7edd236bb862bff9fd25b687d48a8d57d5567809b89fd2727549d0dc48f
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | 3e1fa2bde05a4baec6ddd43cd1994d155a143b9c825ab5dafe766efc305cb1aad92d6026c41c05e9da114a04226361fb6b0510b98e3b05c3ed510da23db403b3
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | 01df4be687f5634afa0ab5ef06f8cee17079264aa452f00a45eccb8ace654c9acc6582f4c74e8242e6ca7715bc48bf2a7d2c4d3d1eef69106f99c8208bc245c4
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | 5523b0b53c30b478b1a9e1df991607886acdcde8605e1b44ef91c94993ca2256c74f6e38fbdd24918d7dbf7afd5cd73d24a3f7ff911e9762819776cc19935363
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | 8e7ebf000bc8dec1079a775576807c0a11764d20a59e16f89d93c948532ba5e6864efd3e08c3e8cc5bd7e7f97bb65baefbf2f01cb226897abd5e01997a4c4f75

## Changelog since v1.19.0-alpha.3

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - ACTION REQUIRED : Switch core master base images (kube-controller-manager) from debian to distroless. If you need Flex Volumes support using scripts, please build your own image with required packages (like bash) ([#91329](https://github.com/kubernetes/kubernetes/pull/91329), [@dims](https://github.com/dims)) [SIG Cloud Provider, Release, Storage and Testing]
  - Kubeadm: Move the "kubeadm init" phase "kubelet-start" later in the init workflow, after the "kubeconfig" phase. This makes kubeadm start the kubelet only after the KubeletConfiguration component config file (/var/lib/kubelet/config.yaml) is generated and solves a problem where init systems like OpenRC cannot crashloop the kubelet service. ([#90892](https://github.com/kubernetes/kubernetes/pull/90892), [@xphoniex](https://github.com/xphoniex)) [SIG Cluster Lifecycle]
 
## Changes by Kind

### API Change

- CertificateSigningRequest API conditions were updated:
  - a `status` field was added; this field defaults to `True`, and may only be set to `True` for `Approved`, `Denied`, and `Failed` conditions
  - a `lastTransitionTime` field was added
  - a `Failed` condition type was added to allow signers to indicate permanent failure; this condition can be added via the `certificatesigningrequests/status` subresource.
  - `Approved` and `Denied` conditions are mutually exclusive
  - `Approved`, `Denied`, and `Failed` conditions can no longer be removed from a CSR ([#90191](https://github.com/kubernetes/kubernetes/pull/90191), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Auth, CLI and Node]
- EnvVarSource api doc bug fixes ([#91194](https://github.com/kubernetes/kubernetes/pull/91194), [@wawa0210](https://github.com/wawa0210)) [SIG Apps]
- Fixed: log timestamps now include trailing zeros to maintain a fixed width ([#91207](https://github.com/kubernetes/kubernetes/pull/91207), [@iamchuckss](https://github.com/iamchuckss)) [SIG Apps and Node]
- The Kubelet's --node-status-max-images option is now available via the Kubelet config file field nodeStatusMaxImage ([#91275](https://github.com/kubernetes/kubernetes/pull/91275), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's --seccomp-profile-root option is now available via the Kubelet config file field seccompProfileRoot. ([#91182](https://github.com/kubernetes/kubernetes/pull/91182), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's `--enable-server` and `--provider-id` option is now available via the Kubelet config file field `enableServer` and `providerID` respectively. ([#90494](https://github.com/kubernetes/kubernetes/pull/90494), [@knabben](https://github.com/knabben)) [SIG Node]
- The Kubelet's `--really-crash-for-testing` and  `--chaos-chance` options are now marked as deprecated. ([#90499](https://github.com/kubernetes/kubernetes/pull/90499), [@knabben](https://github.com/knabben)) [SIG Node]
- The alpha `DynamicAuditing` feature gate and `auditregistration.k8s.io/v1alpha1` API have been removed and are no longer supported. ([#91502](https://github.com/kubernetes/kubernetes/pull/91502), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Auth and Testing]
- `NodeResourcesLeastAllocated` and `NodeResourcesMostAllocated` plugins now support customized weight on the CPU and memory. ([#90544](https://github.com/kubernetes/kubernetes/pull/90544), [@chendave](https://github.com/chendave)) [SIG Scheduling]
- `PostFilter` type is added to scheduler component config API on version v1beta1. ([#91547](https://github.com/kubernetes/kubernetes/pull/91547), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- `kubescheduler.config.k8s.io` is now beta ([#91420](https://github.com/kubernetes/kubernetes/pull/91420), [@pancernik](https://github.com/pancernik)) [SIG Scheduling]

### Feature

- Add --logging-format flag for component-base. Defaults to "text" using unchanged klog. ([#89683](https://github.com/kubernetes/kubernetes/pull/89683), [@yuzhiquan](https://github.com/yuzhiquan)) [SIG Instrumentation]
- Add --port flag to kubectl create deployment ([#91113](https://github.com/kubernetes/kubernetes/pull/91113), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Add .import-restrictions file to cmd/cloud-controller-manager. ([#90630](https://github.com/kubernetes/kubernetes/pull/90630), [@nilo19](https://github.com/nilo19)) [SIG API Machinery and Cloud Provider]
- Add Annotations to CRI-API ImageSpec objects. ([#90061](https://github.com/kubernetes/kubernetes/pull/90061), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Added feature support to Windows for configuring session affinity of Kubernetes services.
  required: [Windows Server vNext Insider Preview Build 19551](https://blogs.windows.com/windowsexperience/2020/01/28/announcing-windows-server-vnext-insider-preview-build-19551/) (or higher) ([#91701](https://github.com/kubernetes/kubernetes/pull/91701), [@elweb9858](https://github.com/elweb9858)) [SIG Network and Windows]
- Added service.beta.kubernetes.io/aws-load-balancer-target-node-labels annotation to target nodes in AWS LoadBalancer Services ([#90943](https://github.com/kubernetes/kubernetes/pull/90943), [@foobarfran](https://github.com/foobarfran)) [SIG Cloud Provider]
- Feat: azure disk migration go beta in 1.19 ([#90896](https://github.com/kubernetes/kubernetes/pull/90896), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Kube-addon-manager has been updated to v9.1.1 to allow overriding the default list of whitelisted resources (https://github.com/kubernetes/kubernetes/pull/91018) ([#91240](https://github.com/kubernetes/kubernetes/pull/91240), [@tosi3k](https://github.com/tosi3k)) [SIG Cloud Provider, Scalability and Testing]
- Kubeadm now distinguishes between generated and user supplied component configs, regenerating the former ones if a config upgrade is required ([#86070](https://github.com/kubernetes/kubernetes/pull/86070), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: add startup probes for static Pods to protect slow starting containers ([#91179](https://github.com/kubernetes/kubernetes/pull/91179), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelets configured to rotate client certificates now publish a `certificate_manager_server_ttl_seconds` gauge metric indicating the remaining seconds until certificate expiration. ([#91148](https://github.com/kubernetes/kubernetes/pull/91148), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
- Local-up-cluster.sh installs CSI snapshotter by default now, can be disabled with ENABLE_CSI_SNAPSHOTTER=false. ([#91504](https://github.com/kubernetes/kubernetes/pull/91504), [@pohly](https://github.com/pohly)) [SIG Storage]
- Rest.Config now supports a flag to override proxy configuration that was previously only configurable through environment variables. ([#81443](https://github.com/kubernetes/kubernetes/pull/81443), [@mikedanese](https://github.com/mikedanese)) [SIG API Machinery and Node]
- Scores from PodTopologySpreading have reduced differentiation as maxSkew increases. ([#90820](https://github.com/kubernetes/kubernetes/pull/90820), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Service controller: only sync LB node pools when relevant fields in Node changes ([#90769](https://github.com/kubernetes/kubernetes/pull/90769), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
- Switch core master base images (kube-apiserver, kube-scheduler) from debian to distroless ([#90674](https://github.com/kubernetes/kubernetes/pull/90674), [@dims](https://github.com/dims)) [SIG Cloud Provider, Release and Scalability]
- Switch etcd image (with migration scripts) from debian to distroless ([#91171](https://github.com/kubernetes/kubernetes/pull/91171), [@dims](https://github.com/dims)) [SIG API Machinery and Cloud Provider]
- The `certificatesigningrequests/approval` subresource now supports patch API requests ([#91558](https://github.com/kubernetes/kubernetes/pull/91558), [@liggitt](https://github.com/liggitt)) [SIG Auth and Testing]
- Update cri-tools to v1.18.0 ([#89720](https://github.com/kubernetes/kubernetes/pull/89720), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Cluster Lifecycle, Release and Scalability]
- Weight of PodTopologySpread scheduling Score is doubled. ([#91258](https://github.com/kubernetes/kubernetes/pull/91258), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- `maxThreshold` of `ImageLocality` plugin is now scaled by the number of images in the pod, which helps to distinguish the node priorities for pod with several images. ([#91138](https://github.com/kubernetes/kubernetes/pull/91138), [@chendave](https://github.com/chendave)) [SIG Scheduling]

### Bug or Regression

- Add support for TLS 1.3 ciphers: TLS_AES_128_GCM_SHA256, TLS_CHACHA20_POLY1305_SHA256 and TLS_AES_256_GCM_SHA384. ([#90843](https://github.com/kubernetes/kubernetes/pull/90843), [@pjbgf](https://github.com/pjbgf)) [SIG API Machinery, Auth and Cluster Lifecycle]
- Base-images: Update to kube-cross:v1.13.9-5 ([#90963](https://github.com/kubernetes/kubernetes/pull/90963), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- CloudNodeLifecycleController will check node existence status before shutdown status when monitoring nodes. ([#90737](https://github.com/kubernetes/kubernetes/pull/90737), [@jiahuif](https://github.com/jiahuif)) [SIG Apps and Cloud Provider]
- First pod with required affinity terms can schedule only on nodes with matching topology keys. ([#91168](https://github.com/kubernetes/kubernetes/pull/91168), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]
- Fix VirtualMachineScaleSets.virtualMachines.GET not allowed issues when customers have set VMSS orchestrationMode. ([#91097](https://github.com/kubernetes/kubernetes/pull/91097), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix a racing issue that scheduler may perform unnecessary scheduling attempt. ([#90660](https://github.com/kubernetes/kubernetes/pull/90660), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fix kubectl create --dryrun client ignore namespace ([#90502](https://github.com/kubernetes/kubernetes/pull/90502), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix kubectl create secret docker-registry --from-file not usable ([#90960](https://github.com/kubernetes/kubernetes/pull/90960), [@zhouya0](https://github.com/zhouya0)) [SIG CLI and Testing]
- Fix kubectl describe node for users not having access to lease information. ([#90469](https://github.com/kubernetes/kubernetes/pull/90469), [@uthark](https://github.com/uthark)) [SIG CLI]
- Fix kubectl run  --dry-run client  ignore namespace ([#90785](https://github.com/kubernetes/kubernetes/pull/90785), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix public IP not shown issues after assigning public IP to Azure VMs ([#90886](https://github.com/kubernetes/kubernetes/pull/90886), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix: add azure file migration support on annotation support ([#91093](https://github.com/kubernetes/kubernetes/pull/91093), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Node]
- Fix: azure disk dangling attach issue which would cause API throttling ([#90749](https://github.com/kubernetes/kubernetes/pull/90749), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: fix topology issue in azure disk storage class migration ([#91196](https://github.com/kubernetes/kubernetes/pull/91196), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: support removal of nodes backed by deleted non VMSS instances on Azure ([#91184](https://github.com/kubernetes/kubernetes/pull/91184), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
- Fixed a regression preventing garbage collection of RBAC role and binding objects ([#90534](https://github.com/kubernetes/kubernetes/pull/90534), [@apelisse](https://github.com/apelisse)) [SIG Auth]
- For external storage e2e test suite, update external driver, to pick snapshot provisioner from VolumeSnapshotClass, when a VolumeSnapshotClass is explicitly provided as an input. ([#90878](https://github.com/kubernetes/kubernetes/pull/90878), [@saikat-royc](https://github.com/saikat-royc)) [SIG Storage and Testing]
- Get-kube.sh: fix order to get the binaries from the right bucket ([#91635](https://github.com/kubernetes/kubernetes/pull/91635), [@cpanato](https://github.com/cpanato)) [SIG Release]
- In a HA env, during the period a standby scheduler lost connection to API server, if a Pod is deleted and recreated, and the standby scheduler becomes master afterwards, there could be a scheduler cache corruption. This PR fixes this issue. ([#91126](https://github.com/kubernetes/kubernetes/pull/91126), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- Kubeadm: during "reset" do not remove the only remaining stacked etcd member from the cluster and just proceed with the cleanup of the local etcd storage. ([#91145](https://github.com/kubernetes/kubernetes/pull/91145), [@tnqn](https://github.com/tnqn)) [SIG Cluster Lifecycle]
- Kubeadm: increase robustness for "kubeadm join" when adding etcd members on slower setups ([#90645](https://github.com/kubernetes/kubernetes/pull/90645), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Pod Conditions updates are skipped for re-scheduling attempts ([#91252](https://github.com/kubernetes/kubernetes/pull/91252), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Prevent PVC requested size overflow when expanding or creating a volume ([#90907](https://github.com/kubernetes/kubernetes/pull/90907), [@gnufied](https://github.com/gnufied)) [SIG Cloud Provider and Storage]
- Resolves an issue using `kubectl certificate approve/deny` against a server serving the v1 CSR API ([#91691](https://github.com/kubernetes/kubernetes/pull/91691), [@liggitt](https://github.com/liggitt)) [SIG Auth and CLI]
- Scheduling failures due to no nodes available are now reported as unschedulable under ```schedule_attempts_total``` metric. ([#90989](https://github.com/kubernetes/kubernetes/pull/90989), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]
- The following components that do not expect non-empty, non-flag arguments will now print an error message and exit if an argument is specified: cloud-controller-manager, kube-apiserver, kube-controller-manager, kube-proxy, kubeadm {alpha|config|token|version}, kubemark. Flags should be prefixed with a single dash "-" (0x45) for short form or double dash "--" for long form. Before this change, malformed flags (for example, starting with a non-ascii dash character such as 0x8211: "–") would have been silently treated as positional arguments and ignored. ([#91349](https://github.com/kubernetes/kubernetes/pull/91349), [@neolit123](https://github.com/neolit123)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Network and Scheduling]
- When evicting, Pods in Pending state are removed without checking PDBs. ([#83906](https://github.com/kubernetes/kubernetes/pull/83906), [@michaelgugino](https://github.com/michaelgugino)) [SIG API Machinery, Apps, Node and Scheduling]

### Other (Cleanup or Flake)

- Adds additional testing to ensure that udp pods conntrack are cleaned up ([#90180](https://github.com/kubernetes/kubernetes/pull/90180), [@JacobTanenbaum](https://github.com/JacobTanenbaum)) [SIG Architecture, Network and Testing]
- Adjusts the fsType for cinder values to be `ext4` if no fsType is specified. ([#90608](https://github.com/kubernetes/kubernetes/pull/90608), [@huffmanca](https://github.com/huffmanca)) [SIG Storage]
- Change beta.kubernetes.io/os to kubernetes.io/os ([#89461](https://github.com/kubernetes/kubernetes/pull/89461), [@wawa0210](https://github.com/wawa0210)) [SIG Cloud Provider and Cluster Lifecycle]
- Content-type and verb for request metrics are now bounded to a known set. ([#89451](https://github.com/kubernetes/kubernetes/pull/89451), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Instrumentation]
- Emit `WaitingForPodScheduled` event if the unbound PVC is in delay binding mode but used by a pod ([#91455](https://github.com/kubernetes/kubernetes/pull/91455), [@cofyc](https://github.com/cofyc)) [SIG Storage]
- Improve server-side apply conflict errors by setting dedicated kubectl subcommand field managers ([#88885](https://github.com/kubernetes/kubernetes/pull/88885), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- It is now possible to use the service annotation `cloud.google.com/network-tier: Standard` to configure the Network Tier of the GCE Loadbalancer ([#88532](https://github.com/kubernetes/kubernetes/pull/88532), [@zioproto](https://github.com/zioproto)) [SIG Cloud Provider, Network and Testing]
- Kube-scheduler: The metric name `scheduler_total_preemption_attempts` has been renamed to `scheduler_preemption_attempts_total`. ([#91448](https://github.com/kubernetes/kubernetes/pull/91448), [@RainbowMango](https://github.com/RainbowMango)) [SIG API Machinery, Cluster Lifecycle, Instrumentation and Scheduling]
- Kubeadm now forwards the IPv6DualStack feature gate using the kubelet component config, instead of the kubelet command line ([#90840](https://github.com/kubernetes/kubernetes/pull/90840), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: do not use a DaemonSet for the pre-pull of control-plane images during "kubeadm upgrade apply". Individual node upgrades now pull the required images using a preflight check. The flag "--image-pull-timeout" for "kubeadm upgrade apply" is now deprecated and will be removed in a future release following a GA deprecation policy. ([#90788](https://github.com/kubernetes/kubernetes/pull/90788), [@xlgao-zju](https://github.com/xlgao-zju)) [SIG Cluster Lifecycle]
- Kubeadm: use two separate checks on /livez and /readyz for the kube-apiserver static Pod instead of using /healthz ([#90970](https://github.com/kubernetes/kubernetes/pull/90970), [@johscheuer](https://github.com/johscheuer)) [SIG Cluster Lifecycle]
- NONE ([#91597](https://github.com/kubernetes/kubernetes/pull/91597), [@elmiko](https://github.com/elmiko)) [SIG Autoscaling and Testing]
- Remove deprecated --server-dry-run flag from kubectl apply ([#91308](https://github.com/kubernetes/kubernetes/pull/91308), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- The "HostPath should give a volume the correct mode" is no longer a conformance test ([#90861](https://github.com/kubernetes/kubernetes/pull/90861), [@dims](https://github.com/dims)) [SIG Architecture and Testing]
- The Kubelet's --experimental-mounter-path and --experimental-check-node-capabilities-before-mount options are now marked as deprecated. ([#91373](https://github.com/kubernetes/kubernetes/pull/91373), [@knabben](https://github.com/knabben)) [SIG Node]
- The kube-apiserver `--kubelet-https` flag is deprecated. kube-apiserver connections to kubelets now unconditionally use `https` (kubelets have unconditionally used `https` to serve the endpoints the apiserver communicates with since before v1.0). ([#91630](https://github.com/kubernetes/kubernetes/pull/91630), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Node]
- Update CNI to v0.8.6 ([#91370](https://github.com/kubernetes/kubernetes/pull/91370), [@justaugustus](https://github.com/justaugustus)) [SIG Cloud Provider, Network, Release and Testing]
- `beta.kubernetes.io/os` and `beta.kubernetes.io/arch` node labels are deprecated. Update node selectors to use `kubernetes.io/os` and `kubernetes.io/arch`. ([#91046](https://github.com/kubernetes/kubernetes/pull/91046), [@wawa0210](https://github.com/wawa0210)) [SIG Apps and Node]
- base-images: Use debian-base:v2.1.0 ([#90697](https://github.com/kubernetes/kubernetes/pull/90697), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery and Release]
- base-images: Use debian-iptables:v12.1.0 ([#90782](https://github.com/kubernetes/kubernetes/pull/90782), [@justaugustus](https://github.com/justaugustus)) [SIG Release]

## Dependencies

### Added
- cloud.google.com/go/bigquery: v1.0.1
- cloud.google.com/go/datastore: v1.0.0
- cloud.google.com/go/pubsub: v1.0.1
- cloud.google.com/go/storage: v1.0.0
- dmitri.shuralyov.com/gpu/mtl: 666a987
- github.com/cespare/xxhash/v2: [v2.1.1](https://github.com/cespare/xxhash/v2/tree/v2.1.1)
- github.com/chzyer/logex: [v1.1.10](https://github.com/chzyer/logex/tree/v1.1.10)
- github.com/chzyer/readline: [2972be2](https://github.com/chzyer/readline/tree/2972be2)
- github.com/chzyer/test: [a1ea475](https://github.com/chzyer/test/tree/a1ea475)
- github.com/containerd/cgroups: [bf292b2](https://github.com/containerd/cgroups/tree/bf292b2)
- github.com/containerd/continuity: [aaeac12](https://github.com/containerd/continuity/tree/aaeac12)
- github.com/containerd/fifo: [a9fb20d](https://github.com/containerd/fifo/tree/a9fb20d)
- github.com/containerd/go-runc: [5a6d9f3](https://github.com/containerd/go-runc/tree/5a6d9f3)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/cpuguy83/go-md2man/v2: [v2.0.0](https://github.com/cpuguy83/go-md2man/v2/tree/v2.0.0)
- github.com/go-gl/glfw/v3.3/glfw: [12ad95a](https://github.com/go-gl/glfw/v3.3/glfw/tree/12ad95a)
- github.com/google/renameio: [v0.1.0](https://github.com/google/renameio/tree/v0.1.0)
- github.com/ianlancetaylor/demangle: [5e5cf60](https://github.com/ianlancetaylor/demangle/tree/5e5cf60)
- github.com/rogpeppe/go-internal: [v1.3.0](https://github.com/rogpeppe/go-internal/tree/v1.3.0)
- github.com/russross/blackfriday/v2: [v2.0.1](https://github.com/russross/blackfriday/v2/tree/v2.0.1)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/ugorji/go: [v1.1.4](https://github.com/ugorji/go/tree/v1.1.4)
- golang.org/x/mod: v0.1.0
- google.golang.org/protobuf: v1.23.0
- gopkg.in/errgo.v2: v2.1.0
- k8s.io/klog/v2: v2.0.0

### Changed
- cloud.google.com/go: v0.38.0 → v0.51.0
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [27a4ced → 7901bc8](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/27a4ced...7901bc8)
- github.com/Microsoft/hcsshim: [672e52e → v0.8.9](https://github.com/Microsoft/hcsshim/compare/672e52e...v0.8.9)
- github.com/alecthomas/template: [a0175ee → fb15b89](https://github.com/alecthomas/template/compare/a0175ee...fb15b89)
- github.com/alecthomas/units: [2efee85 → c3de453](https://github.com/alecthomas/units/compare/2efee85...c3de453)
- github.com/beorn7/perks: [v1.0.0 → v1.0.1](https://github.com/beorn7/perks/compare/v1.0.0...v1.0.1)
- github.com/coreos/pkg: [97fdf19 → 399ea9e](https://github.com/coreos/pkg/compare/97fdf19...399ea9e)
- github.com/go-kit/kit: [v0.8.0 → v0.9.0](https://github.com/go-kit/kit/compare/v0.8.0...v0.9.0)
- github.com/go-logfmt/logfmt: [v0.3.0 → v0.4.0](https://github.com/go-logfmt/logfmt/compare/v0.3.0...v0.4.0)
- github.com/golang/groupcache: [02826c3 → 215e871](https://github.com/golang/groupcache/compare/02826c3...215e871)
- github.com/golang/protobuf: [v1.3.3 → v1.4.2](https://github.com/golang/protobuf/compare/v1.3.3...v1.4.2)
- github.com/google/cadvisor: [8af10c6 → 6a8d614](https://github.com/google/cadvisor/compare/8af10c6...6a8d614)
- github.com/google/pprof: [3ea8567 → d4f498a](https://github.com/google/pprof/compare/3ea8567...d4f498a)
- github.com/googleapis/gax-go/v2: [v2.0.4 → v2.0.5](https://github.com/googleapis/gax-go/v2/compare/v2.0.4...v2.0.5)
- github.com/json-iterator/go: [v1.1.8 → v1.1.9](https://github.com/json-iterator/go/compare/v1.1.8...v1.1.9)
- github.com/jstemmer/go-junit-report: [af01ea7 → v0.9.1](https://github.com/jstemmer/go-junit-report/compare/af01ea7...v0.9.1)
- github.com/prometheus/client_golang: [v1.0.0 → v1.6.0](https://github.com/prometheus/client_golang/compare/v1.0.0...v1.6.0)
- github.com/prometheus/common: [v0.4.1 → v0.9.1](https://github.com/prometheus/common/compare/v0.4.1...v0.9.1)
- github.com/prometheus/procfs: [v0.0.5 → v0.0.11](https://github.com/prometheus/procfs/compare/v0.0.5...v0.0.11)
- github.com/spf13/cobra: [v0.0.5 → v1.0.0](https://github.com/spf13/cobra/compare/v0.0.5...v1.0.0)
- github.com/spf13/viper: [v1.3.2 → v1.4.0](https://github.com/spf13/viper/compare/v1.3.2...v1.4.0)
- github.com/tmc/grpc-websocket-proxy: [89b8d40 → 0ad062e](https://github.com/tmc/grpc-websocket-proxy/compare/89b8d40...0ad062e)
- go.opencensus.io: v0.21.0 → v0.22.2
- go.uber.org/atomic: v1.3.2 → v1.4.0
- golang.org/x/exp: 4b39c73 → da58074
- golang.org/x/image: 0694c2d → cff245a
- golang.org/x/lint: 959b441 → fdd1cda
- golang.org/x/mobile: d3739f8 → d2bd2a2
- golang.org/x/oauth2: 0f29369 → 858c2ad
- google.golang.org/api: 5213b80 → v0.15.1
- google.golang.org/appengine: v1.5.0 → v1.6.5
- google.golang.org/genproto: f3c370f → ca5a221
- honnef.co/go/tools: e561f67 → v0.0.1-2019.2.3
- k8s.io/gengo: e0e292d → 8167cfd
- k8s.io/kube-openapi: e1beb1b → 656914f
- k8s.io/utils: a9aa75a → 2df71eb
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.7 → 33b9978

### Removed
- github.com/coreos/go-etcd: [v2.0.0+incompatible](https://github.com/coreos/go-etcd/tree/v2.0.0)
- github.com/ugorji/go/codec: [d75b2dc](https://github.com/ugorji/go/codec/tree/d75b2dc)
- k8s.io/klog: v1.0.0



# v1.19.0-beta.0


## Downloads for v1.19.0-beta.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes.tar.gz) | 8c7e820b8bd7a8f742b7560cafe6ae1acc4c9836ae23d1b10d987b4de6a690826be75c68b8f76ec027097e8dfd861afb1d229b3687f0b82afcfe7b4d6481242e
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-src.tar.gz) | 543e9d36fd8b2de3e19631d3295d3a7706e6e88bbd3adb2d558b27b3179a3961455f4f04f0d4a5adcff1466779e1b08023fe64dc2ab39813b37adfbbc779dec7

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-darwin-386.tar.gz) | 3ef37ef367a8d9803f023f6994d73ff217865654a69778c1ea3f58c88afbf25ff5d8d6bec9c608ac647c2654978228c4e63f30eec2a89d16d60f4a1c5f333b22
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | edb02b0b8d6a1c2167fbce4a85d84fb413566d3a76839fd366801414ca8ad2d55a5417b39b4cac6b65fddf13c1b3259791a607703773241ca22a67945ecb0014
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-linux-386.tar.gz) | dafe93489df7328ae23f4bdf0a9d2e234e18effe7e042b217fe2dd1355e527a54bab3fb664696ed606a8ebedce57da4ee12647ec1befa2755bd4c43d9d016063
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | d8e2bf8c9dd665410c2e7ceaa98bc4fc4f966753b7ade91dcef3b5eff45e0dda63bd634610c8761392a7804deb96c6b030c292280bf236b8b29f63b7f1af3737
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-linux-arm.tar.gz) | d590d3d07d0ebbb562bce480c7cbe4e60b99feba24376c216fe73d8b99a246e2cd2acb72abe1427bde3e541d94d55b7688daf9e6961e4cbc6b875ac4eeea6e62
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | f9647a99a566c9febd348c1c4a8e5c05326058eab076292a8bb5d3a2b882ee49287903f8e0e036b40af294aa3571edd23e65f3de91330ac9af0c10350b02583d
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | 662f009bc393734a89203d7956942d849bad29e28448e7baa017d1ac2ec2d26d7290da4a44bccb99ed960b2e336d9d98908c98f8a3d9fe1c54df2d134c799cad
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | 61fdf4aff78dcdb721b82a3602bf5bc94d44d51ab6607b255a9c2218bb3e4b57f6e656c2ee0dd68586fb53acbeff800d6fd03e4642dded49735a93356e7c5703
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-windows-386.tar.gz) | 20d1e803b10b3bee09a7a206473ba320cc5f1120278d8f6e0136c388b2720da7264b917cd4738488b1d0a9aa922eb581c1f540715a6c2042c4dd7b217b6a9a0a
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | b85d729ec269f6aad0b6d2f95f3648fbea84330d2fbfde2267a519bc08c42d70d7b658b0e41c3b0d5f665702a8f1bbb37652753de34708ae3a03e45175c8b92c

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | c3641bdb0a8d8eff5086d24b71c6547131092b21f976b080dc48129f91de3da560fed6edf880eab1d205017ad74be716a5b970e4bbc00d753c005e5932b3d319
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-server-linux-arm.tar.gz) | 7c29b8e33ade23a787330d28da22bf056610dae4d3e15574c56c46340afe5e0fdb00126ae3fd64fd70a26d1a87019f47e401682b88fa1167368c7edbecc72ccf
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | 27cd6042425eb94bb468431599782467ed818bcc51d75e8cb251c287a806b60a5cce50d4ae7525348c5446eaa45f849bc3fe3e6ac7248b54f3ebae8bf6553c3f
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | ede896424eb12ec07dd3756cbe808ca3915f51227e7b927795402943d81a99bb61654fd8f485a838c2faf199d4a55071af5bd8e69e85669a7f4a0b0e84a093cc
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | 4e48d4f5afd22f0ae6ade7da4877238fd2a5c10ae3dea2ae721c39ac454b0b295e1d7501e26bddee4bc0289e79e33dadca255a52a645bee98cf81acf937db0ef

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 8025bd8deb9586487fcf268bdaf99e8fd9f9433d9e7221c29363d1d66c4cbd55a2c44e6c89bc8133828c6a1aa0c42c2359b74846dfb71765c9ae8f21b8170625
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-node-linux-arm.tar.gz) | 25787d47c8cc1e9445218d3a947b443d261266033187f8b7bc6141ae353a6806503fe72e3626f058236d4cd7f284348d2cc8ccb7a0219b9ddd7c6a336dae360b
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | ff737a7310057bdfd603f2853b15f79dc2b54a3cbbbd7a8ffd4d9756720fa5a02637ffc10a381eeee58bef61024ff348a49f3044a6dfa0ba99645fda8d08e2da
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 2b1144c9ae116306a2c3214b02361083a60a349afc804909f95ea85db3660de5025de69a1860e8fc9e7e92ded335c93b74ecbbb20e1f6266078842d4adaf4161
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | 822ec64aef3d65faa668a91177aa7f5d0c78a83cc1284c5e30629eda448ee4b2874cf4cfa6f3d68ad8eb8029dd035bf9fe15f68cc5aa4b644513f054ed7910ae
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 3957cae43211df050c5a9991a48e23ac27d20aec117c580c53fc7edf47caf79ed1e2effa969b5b972968a83e9bdba0b20c46705caca0c35571713041481c1966

## Changelog since v1.19.0-alpha.3

## Changes by Kind

### API Change
 - EnvVarSource api doc bug fixes ([#91194](https://github.com/kubernetes/kubernetes/pull/91194), [@wawa0210](https://github.com/wawa0210)) [SIG Apps]
 - The Kubelet's `--really-crash-for-testing` and  `--chaos-chance` options are now marked as deprecated. ([#90499](https://github.com/kubernetes/kubernetes/pull/90499), [@knabben](https://github.com/knabben)) [SIG Node]
 - `NodeResourcesLeastAllocated` and `NodeResourcesMostAllocated` plugins now support customized weight on the CPU and memory. ([#90544](https://github.com/kubernetes/kubernetes/pull/90544), [@chendave](https://github.com/chendave)) [SIG Scheduling]

### Feature
 - Add .import-restrictions file to cmd/cloud-controller-manager. ([#90630](https://github.com/kubernetes/kubernetes/pull/90630), [@nilo19](https://github.com/nilo19)) [SIG API Machinery and Cloud Provider]
 - Add Annotations to CRI-API ImageSpec objects. ([#90061](https://github.com/kubernetes/kubernetes/pull/90061), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
 - Kubelets configured to rotate client certificates now publish a `certificate_manager_server_ttl_seconds` gauge metric indicating the remaining seconds until certificate expiration. ([#91148](https://github.com/kubernetes/kubernetes/pull/91148), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
 - Rest.Config now supports a flag to override proxy configuration that was previously only configurable through environment variables. ([#81443](https://github.com/kubernetes/kubernetes/pull/81443), [@mikedanese](https://github.com/mikedanese)) [SIG API Machinery and Node]
 - Scores from PodTopologySpreading have reduced differentiation as maxSkew increases. ([#90820](https://github.com/kubernetes/kubernetes/pull/90820), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
 - Service controller: only sync LB node pools when relevant fields in Node changes ([#90769](https://github.com/kubernetes/kubernetes/pull/90769), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
 - Switch core master base images (kube-apiserver, kube-scheduler) from debian to distroless ([#90674](https://github.com/kubernetes/kubernetes/pull/90674), [@dims](https://github.com/dims)) [SIG Cloud Provider, Release and Scalability]
 - Update cri-tools to v1.18.0 ([#89720](https://github.com/kubernetes/kubernetes/pull/89720), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Cluster Lifecycle, Release and Scalability]

### Bug or Regression
 - Add support for TLS 1.3 ciphers: TLS_AES_128_GCM_SHA256, TLS_CHACHA20_POLY1305_SHA256 and TLS_AES_256_GCM_SHA384. ([#90843](https://github.com/kubernetes/kubernetes/pull/90843), [@pjbgf](https://github.com/pjbgf)) [SIG API Machinery, Auth and Cluster Lifecycle]
 - Base-images: Update to kube-cross:v1.13.9-5 ([#90963](https://github.com/kubernetes/kubernetes/pull/90963), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
 - CloudNodeLifecycleController will check node existence status before shutdown status when monitoring nodes. ([#90737](https://github.com/kubernetes/kubernetes/pull/90737), [@jiahuif](https://github.com/jiahuif)) [SIG Apps and Cloud Provider]
 - First pod with required affinity terms can schedule only on nodes with matching topology keys. ([#91168](https://github.com/kubernetes/kubernetes/pull/91168), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]
 - Fix VirtualMachineScaleSets.virtualMachines.GET not allowed issues when customers have set VMSS orchestrationMode. ([#91097](https://github.com/kubernetes/kubernetes/pull/91097), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
 - Fix a racing issue that scheduler may perform unnecessary scheduling attempt. ([#90660](https://github.com/kubernetes/kubernetes/pull/90660), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
 - Fix kubectl run  --dry-run client  ignore namespace ([#90785](https://github.com/kubernetes/kubernetes/pull/90785), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
 - Fix public IP not shown issues after assigning public IP to Azure VMs ([#90886](https://github.com/kubernetes/kubernetes/pull/90886), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
 - Fix: azure disk dangling attach issue which would cause API throttling ([#90749](https://github.com/kubernetes/kubernetes/pull/90749), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
 - Fix: support removal of nodes backed by deleted non VMSS instances on Azure ([#91184](https://github.com/kubernetes/kubernetes/pull/91184), [@bpineau](https://github.com/bpineau)) [SIG Cloud Provider]
 - Fixed a regression preventing garbage collection of RBAC role and binding objects ([#90534](https://github.com/kubernetes/kubernetes/pull/90534), [@apelisse](https://github.com/apelisse)) [SIG Auth]
 - For external storage e2e test suite, update external driver, to pick snapshot provisioner from VolumeSnapshotClass, when a VolumeSnapshotClass is explicitly provided as an input. ([#90878](https://github.com/kubernetes/kubernetes/pull/90878), [@saikat-royc](https://github.com/saikat-royc)) [SIG Storage and Testing]
 - In a HA env, during the period a standby scheduler lost connection to API server, if a Pod is deleted and recreated, and the standby scheduler becomes master afterwards, there could be a scheduler cache corruption. This PR fixes this issue. ([#91126](https://github.com/kubernetes/kubernetes/pull/91126), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
 - Kubeadm: increase robustness for "kubeadm join" when adding etcd members on slower setups ([#90645](https://github.com/kubernetes/kubernetes/pull/90645), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
 - Prevent PVC requested size overflow when expanding or creating a volume ([#90907](https://github.com/kubernetes/kubernetes/pull/90907), [@gnufied](https://github.com/gnufied)) [SIG Cloud Provider and Storage]
 - Scheduling failures due to no nodes available are now reported as unschedulable under ```schedule_attempts_total``` metric. ([#90989](https://github.com/kubernetes/kubernetes/pull/90989), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]

### Other (Cleanup or Flake)
 - Adds additional testing to ensure that udp pods conntrack are cleaned up ([#90180](https://github.com/kubernetes/kubernetes/pull/90180), [@JacobTanenbaum](https://github.com/JacobTanenbaum)) [SIG Architecture, Network and Testing]
 - Adjusts the fsType for cinder values to be `ext4` if no fsType is specified. ([#90608](https://github.com/kubernetes/kubernetes/pull/90608), [@huffmanca](https://github.com/huffmanca)) [SIG Storage]
 - Change beta.kubernetes.io/os to kubernetes.io/os ([#89461](https://github.com/kubernetes/kubernetes/pull/89461), [@wawa0210](https://github.com/wawa0210)) [SIG Cloud Provider and Cluster Lifecycle]
 - Improve server-side apply conflict errors by setting dedicated kubectl subcommand field managers ([#88885](https://github.com/kubernetes/kubernetes/pull/88885), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
 - It is now possible to use the service annotation `cloud.google.com/network-tier: Standard` to configure the Network Tier of the GCE Loadbalancer ([#88532](https://github.com/kubernetes/kubernetes/pull/88532), [@zioproto](https://github.com/zioproto)) [SIG Cloud Provider, Network and Testing]
 - Kubeadm now forwards the IPv6DualStack feature gate using the kubelet component config, instead of the kubelet command line ([#90840](https://github.com/kubernetes/kubernetes/pull/90840), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
 - Kubeadm: do not use a DaemonSet for the pre-pull of control-plane images during "kubeadm upgrade apply". Individual node upgrades now pull the required images using a preflight check. The flag "--image-pull-timeout" for "kubeadm upgrade apply" is now deprecated and will be removed in a future release following a GA deprecation policy. ([#90788](https://github.com/kubernetes/kubernetes/pull/90788), [@xlgao-zju](https://github.com/xlgao-zju)) [SIG Cluster Lifecycle]
 - Kubeadm: use two separate checks on /livez and /readyz for the kube-apiserver static Pod instead of using /healthz ([#90970](https://github.com/kubernetes/kubernetes/pull/90970), [@johscheuer](https://github.com/johscheuer)) [SIG Cluster Lifecycle]
 - The "HostPath should give a volume the correct mode" is no longer a conformance test ([#90861](https://github.com/kubernetes/kubernetes/pull/90861), [@dims](https://github.com/dims)) [SIG Architecture and Testing]
 - `beta.kubernetes.io/os` and `beta.kubernetes.io/arch` node labels are deprecated. Update node selectors to use `kubernetes.io/os` and `kubernetes.io/arch`. ([#91046](https://github.com/kubernetes/kubernetes/pull/91046), [@wawa0210](https://github.com/wawa0210)) [SIG Apps and Node]
 - base-images: Use debian-base:v2.1.0 ([#90697](https://github.com/kubernetes/kubernetes/pull/90697), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery and Release]
 - base-images: Use debian-iptables:v12.1.0 ([#90782](https://github.com/kubernetes/kubernetes/pull/90782), [@justaugustus](https://github.com/justaugustus)) [SIG Release]

## Dependencies

### Added
- cloud.google.com/go/bigquery: v1.0.1
- cloud.google.com/go/datastore: v1.0.0
- cloud.google.com/go/pubsub: v1.0.1
- cloud.google.com/go/storage: v1.0.0
- dmitri.shuralyov.com/gpu/mtl: 666a987
- github.com/cespare/xxhash/v2: [v2.1.1](https://github.com/cespare/xxhash/v2/tree/v2.1.1)
- github.com/chzyer/logex: [v1.1.10](https://github.com/chzyer/logex/tree/v1.1.10)
- github.com/chzyer/readline: [2972be2](https://github.com/chzyer/readline/tree/2972be2)
- github.com/chzyer/test: [a1ea475](https://github.com/chzyer/test/tree/a1ea475)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/cpuguy83/go-md2man/v2: [v2.0.0](https://github.com/cpuguy83/go-md2man/v2/tree/v2.0.0)
- github.com/go-gl/glfw/v3.3/glfw: [12ad95a](https://github.com/go-gl/glfw/v3.3/glfw/tree/12ad95a)
- github.com/google/renameio: [v0.1.0](https://github.com/google/renameio/tree/v0.1.0)
- github.com/ianlancetaylor/demangle: [5e5cf60](https://github.com/ianlancetaylor/demangle/tree/5e5cf60)
- github.com/rogpeppe/go-internal: [v1.3.0](https://github.com/rogpeppe/go-internal/tree/v1.3.0)
- github.com/russross/blackfriday/v2: [v2.0.1](https://github.com/russross/blackfriday/v2/tree/v2.0.1)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/ugorji/go: [v1.1.4](https://github.com/ugorji/go/tree/v1.1.4)
- golang.org/x/mod: v0.1.0
- google.golang.org/protobuf: v1.23.0
- gopkg.in/errgo.v2: v2.1.0
- k8s.io/klog/v2: v2.0.0

### Changed
- cloud.google.com/go: v0.38.0 → v0.51.0
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [27a4ced → 7901bc8](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/27a4ced...7901bc8)
- github.com/alecthomas/template: [a0175ee → fb15b89](https://github.com/alecthomas/template/compare/a0175ee...fb15b89)
- github.com/alecthomas/units: [2efee85 → c3de453](https://github.com/alecthomas/units/compare/2efee85...c3de453)
- github.com/beorn7/perks: [v1.0.0 → v1.0.1](https://github.com/beorn7/perks/compare/v1.0.0...v1.0.1)
- github.com/coreos/pkg: [97fdf19 → 399ea9e](https://github.com/coreos/pkg/compare/97fdf19...399ea9e)
- github.com/go-kit/kit: [v0.8.0 → v0.9.0](https://github.com/go-kit/kit/compare/v0.8.0...v0.9.0)
- github.com/go-logfmt/logfmt: [v0.3.0 → v0.4.0](https://github.com/go-logfmt/logfmt/compare/v0.3.0...v0.4.0)
- github.com/golang/groupcache: [02826c3 → 215e871](https://github.com/golang/groupcache/compare/02826c3...215e871)
- github.com/golang/protobuf: [v1.3.3 → v1.4.2](https://github.com/golang/protobuf/compare/v1.3.3...v1.4.2)
- github.com/google/cadvisor: [8af10c6 → 6a8d614](https://github.com/google/cadvisor/compare/8af10c6...6a8d614)
- github.com/google/pprof: [3ea8567 → d4f498a](https://github.com/google/pprof/compare/3ea8567...d4f498a)
- github.com/googleapis/gax-go/v2: [v2.0.4 → v2.0.5](https://github.com/googleapis/gax-go/v2/compare/v2.0.4...v2.0.5)
- github.com/json-iterator/go: [v1.1.8 → v1.1.9](https://github.com/json-iterator/go/compare/v1.1.8...v1.1.9)
- github.com/jstemmer/go-junit-report: [af01ea7 → v0.9.1](https://github.com/jstemmer/go-junit-report/compare/af01ea7...v0.9.1)
- github.com/prometheus/client_golang: [v1.0.0 → v1.6.0](https://github.com/prometheus/client_golang/compare/v1.0.0...v1.6.0)
- github.com/prometheus/common: [v0.4.1 → v0.9.1](https://github.com/prometheus/common/compare/v0.4.1...v0.9.1)
- github.com/prometheus/procfs: [v0.0.5 → v0.0.11](https://github.com/prometheus/procfs/compare/v0.0.5...v0.0.11)
- github.com/spf13/cobra: [v0.0.5 → v1.0.0](https://github.com/spf13/cobra/compare/v0.0.5...v1.0.0)
- github.com/spf13/viper: [v1.3.2 → v1.4.0](https://github.com/spf13/viper/compare/v1.3.2...v1.4.0)
- github.com/tmc/grpc-websocket-proxy: [89b8d40 → 0ad062e](https://github.com/tmc/grpc-websocket-proxy/compare/89b8d40...0ad062e)
- go.opencensus.io: v0.21.0 → v0.22.2
- go.uber.org/atomic: v1.3.2 → v1.4.0
- golang.org/x/exp: 4b39c73 → da58074
- golang.org/x/image: 0694c2d → cff245a
- golang.org/x/lint: 959b441 → fdd1cda
- golang.org/x/mobile: d3739f8 → d2bd2a2
- golang.org/x/oauth2: 0f29369 → 858c2ad
- google.golang.org/api: 5213b80 → v0.15.1
- google.golang.org/appengine: v1.5.0 → v1.6.5
- google.golang.org/genproto: f3c370f → ca5a221
- honnef.co/go/tools: e561f67 → v0.0.1-2019.2.3
- k8s.io/gengo: e0e292d → 8167cfd
- k8s.io/kube-openapi: e1beb1b → 656914f
- k8s.io/utils: a9aa75a → 2df71eb
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.7 → 33b9978

### Removed
- github.com/coreos/go-etcd: [v2.0.0+incompatible](https://github.com/coreos/go-etcd/tree/v2.0.0)
- github.com/ugorji/go/codec: [d75b2dc](https://github.com/ugorji/go/codec/tree/d75b2dc)
- k8s.io/klog: v1.0.0



# v1.19.0-alpha.3

[Documentation](https://docs.k8s.io)

## Downloads for v1.19.0-alpha.3

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes.tar.gz) | `49df3a77453b759d3262be6883dd9018426666b4261313725017eed42da1bc8dd1af037ec6c11357a6360c0c32c2486490036e9e132c9026f491325ce353c84b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-src.tar.gz) | `ddbb0baaf77516dc885c41017f4a8d91d0ff33eeab14009168a1e4d975939ccc6a053a682c2af14346c67fe7b142aa2c1ba32e86a30f2433cefa423764c5332d`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `c0fb1afb5b22f6e29cf3e5121299d3a5244a33b7663e041209bcc674a0009842b35b9ebdafa5bd6b91a1e1b67fa891e768627b97ea5258390d95250f07c2defc`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `f32596863fed32bc8e3f032ef1e4f9f232898ed506624cb1b4877ce2ced2a0821d70b15599258422aa13181ab0e54f38837399ca611ab86cbf3feec03ede8b95`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `37290244cee54ff05662c2b14b69445eee674d385e6b05ca0b8c8b410ba047cf054033229c78af91670ca1370807753103c25dbb711507edc1c6beca87bd0988`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `3753eb28b9d68a47ef91fff3e91215015c28bce12828f81c0bbddbde118fd2cf4d580e474e54b1e8176fa547829e2ed08a4df36bbf83b912c831a459821bd581`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `86b1cdb59a6b4e9de4496e5aa817b1ae7687ac6a93f8b8259cdeb356020773711d360a2ea35f7a8dc1bdd6d31c95e6491abf976afaff3392eb7d2df1008e192c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `fbf324e92b93cd8048073b2a627ddc8866020bc4f086604d82bf4733d463411a534d8c8f72565976eb1b32be64aecae8858cd140ef8b7a3c96fcbbf92ca54689`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `7a6551eca17d29efb5d818e360b53ab2f0284e1091cc537e0a7ce39843d0b77579f26eb14bdeca9aa9e0aa0ef92ce1ccde34bdce84b4a5c1e090206979afb0ea`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `46352be54882cf3edb949b355e71daea839c9b1955ccfe1085590b81326665d81cabde192327d82e56d6a157e224caefdcfbec3364b9f8b18b5da0cfcb97fc0c`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `d049bf5f27e5e646ea4aa657aa0a694de57394b0dc60eadf1f7516d1ca6a6db39fc89d34bb6bba0a82f0c140113c2a91c41ad409e0ab41118a104f47eddcb9d2`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `2e585f6f97b86443a6e3a847c8dfaa29c6323f8d5bbfdb86dc7bf5465ba54f64b35ee55a6d38e9be105a67fff39057ad16db3f3b1c3b9c909578517f4da7e51e`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `8c41c6abf32ba7040c2cc654765d443e615d96891eacf6bcec24146a8aaf79b9206d13358518958e5ec04eb911ade108d4522ebd8603b88b3e3d95e7d5b24e60`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `7e54c60bf724e2e3e2cff1197512ead0f73030788877f2f92a7e0deeeabd86e75ce8120eb815bf63909f8a110e647a5fcfddd510efffbd9c339bd0f90caa6706`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `7c57fd80b18be6dd6b6e17558d12ec0c07c06ce248e99837737fdd39b7f5d752597679748dc6294563f30def986ed712a8f469f3ea1c3a4cbe5d63c44f1d41dc`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `d22b1d4d8ccf9e9df8f90d35b8d2a1e7916f8d809806743cddc00b15d8ace095c54c61d7c9affd6609a316ee14ba43bf760bfec4276aee8273203aab3e7ac3c1`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `3177c9a2d6bd116d614fa69ff9cb16b822bee4e36e38f93ece6aeb5d118ae67dbe61546c7f628258ad719e763c127ca32437ded70279ea869cfe4869e06cbdde`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `543248e35c57454bfc4b6f3cf313402d7cf81606b9821a5dd95c6758d55d5b9a42e283a7fb0d45322ad1014e3382aafaee69879111c0799dac31d5c4ad1b8041`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `c94bed3861376d3fd41cb7bc93b5a849612bc7346ed918f6b5b634449cd3acef69ff63ca0b6da29f45df68402f64f3d290d7688bc50f46dac07e889219dac30c`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `3649dbca59d08c3922830b7acd8176e8d2f622fbf6379288f3a70045763d5d72c944d241f8a2c57306f23e6e44f7cc3b912554442f77e0f90e9f876f240114a8`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `5655d1d48a1ae97352af2d703954c7a28c2d1c644319c4eb24fe19ccc5fb546c30b34cc86d8910f26c88feee88d7583bc085ebfe58916054f73dcf372a824fd9`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `55190804357a687c37d1abb489d5aef7cea209d1c03778548f0aa4dab57a0b98b710fda09ff5c46d0963f2bb674726301d544b359f673df8f57226cafa831ce3`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `d8ffbe8dc9a0b0b55db357afa6ef94e6145f9142b1bc505897cac9ee7c950ef527a189397a8e61296e66ce76b020eccb276668256927d2273d6079b9ffebef24`

## Changelog since v1.19.0-alpha.2

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

- Kubeadm does not set the deprecated '--cgroup-driver' flag in /var/lib/kubelet/kubeadm-flags.env, it will be set in the kubelet config.yaml. If you have this flag in /var/lib/kubelet/kubeadm-flags.env or /etc/default/kubelet (/etc/sysconfig/kubelet for RPMs) please remove it and set the value using KubeletConfiguration ([#90513](https://github.com/kubernetes/kubernetes/pull/90513), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]

- Kubeadm respects resolvConf value set by user even if systemd-resolved service is active. kubeadm no longer sets the flag in '--resolv-conf' in /var/lib/kubelet/kubeadm-flags.env. If you have this flag in /var/lib/kubelet/kubeadm-flags.env or /etc/default/kubelet (/etc/sysconfig/kubelet for RPMs) please remove it and set the value using KubeletConfiguration ([#90394](https://github.com/kubernetes/kubernetes/pull/90394), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]

## Changes by Kind

### Deprecation

- Apiextensions.k8s.io/v1beta1 is deprecated in favor of apiextensions.k8s.io/v1 ([#90673](https://github.com/kubernetes/kubernetes/pull/90673), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Apiregistration.k8s.io/v1beta1 is deprecated in favor of apiregistration.k8s.io/v1 ([#90672](https://github.com/kubernetes/kubernetes/pull/90672), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Authentication.k8s.io/v1beta1 and authorization.k8s.io/v1beta1 are deprecated in 1.19 in favor of v1 levels and will be removed in 1.22 ([#90458](https://github.com/kubernetes/kubernetes/pull/90458), [@deads2k](https://github.com/deads2k)) [SIG API Machinery and Auth]
- Autoscaling/v2beta1 is deprecated in favor of autoscaling/v2beta2 ([#90463](https://github.com/kubernetes/kubernetes/pull/90463), [@deads2k](https://github.com/deads2k)) [SIG Autoscaling]
- Coordination.k8s.io/v1beta1 is deprecated in 1.19, targeted for removal in 1.22, use v1 instead. ([#90559](https://github.com/kubernetes/kubernetes/pull/90559), [@deads2k](https://github.com/deads2k)) [SIG Scalability]
- Storage.k8s.io/v1beta1 is deprecated in favor of storage.k8s.io/v1 ([#90671](https://github.com/kubernetes/kubernetes/pull/90671), [@deads2k](https://github.com/deads2k)) [SIG Storage]

### API Change

- K8s.io/apimachinery - scheme.Convert() now uses only explicitly registered conversions - default reflection based conversion is no longer available. `+k8s:conversion-gen` tags can be used with the `k8s.io/code-generator` component to generate conversions. ([#90018](https://github.com/kubernetes/kubernetes/pull/90018), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Apps and Testing]
- Kubelet's --runonce option is now also available in Kubelet's config file as `runOnce`. ([#89128](https://github.com/kubernetes/kubernetes/pull/89128), [@vincent178](https://github.com/vincent178)) [SIG Node]
- Promote Immutable Secrets/ConfigMaps feature to Beta and enable the feature by default.
  This allows to set `Immutable` field in Secrets or ConfigMap object to mark their contents as immutable. ([#89594](https://github.com/kubernetes/kubernetes/pull/89594), [@wojtek-t](https://github.com/wojtek-t)) [SIG Apps and Testing]
- The unused `series.state` field, deprecated since v1.14, is removed from the `events.k8s.io/v1beta1` and `v1` Event types. ([#90449](https://github.com/kubernetes/kubernetes/pull/90449), [@wojtek-t](https://github.com/wojtek-t)) [SIG Apps]

### Feature

- Kube-apiserver: The NodeRestriction admission plugin now restricts Node labels kubelets are permitted to set when creating a new Node to the `--node-labels` parameters accepted by kubelets in 1.16+. ([#90307](https://github.com/kubernetes/kubernetes/pull/90307), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
- Kubectl supports taint no without specifying(without having to type the full resource name) ([#88723](https://github.com/kubernetes/kubernetes/pull/88723), [@wawa0210](https://github.com/wawa0210)) [SIG CLI]
- New scoring for PodTopologySpreading that yields better spreading ([#90475](https://github.com/kubernetes/kubernetes/pull/90475), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- No ([#89549](https://github.com/kubernetes/kubernetes/pull/89549), [@happinesstaker](https://github.com/happinesstaker)) [SIG API Machinery, Auth, Instrumentation and Testing]
- Try to send watch bookmarks (if requested) periodically in addition to sending them right before timeout ([#90560](https://github.com/kubernetes/kubernetes/pull/90560), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]

### Bug or Regression

- Avoid GCE API calls when initializing GCE CloudProvider for Kubelets. ([#90218](https://github.com/kubernetes/kubernetes/pull/90218), [@wojtek-t](https://github.com/wojtek-t)) [SIG Cloud Provider and Scalability]
- Avoid unnecessary scheduling churn when annotations are updated while Pods are being scheduled. ([#90373](https://github.com/kubernetes/kubernetes/pull/90373), [@fabiokung](https://github.com/fabiokung)) [SIG Scheduling]
- Fix a bug where ExternalTrafficPolicy is not applied to service ExternalIPs. ([#90537](https://github.com/kubernetes/kubernetes/pull/90537), [@freehan](https://github.com/freehan)) [SIG Network]
- Fixed a regression in wait.Forever that skips the backoff period on the first repeat ([#90476](https://github.com/kubernetes/kubernetes/pull/90476), [@zhan849](https://github.com/zhan849)) [SIG API Machinery]
- Fixes a bug that non directory hostpath type can be recognized as HostPathFile and adds e2e tests for HostPathType ([#64829](https://github.com/kubernetes/kubernetes/pull/64829), [@dixudx](https://github.com/dixudx)) [SIG Apps, Storage and Testing]
- Fixes a regression in 1.17 that dropped cache-control headers on API requests ([#90468](https://github.com/kubernetes/kubernetes/pull/90468), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Fixes regression in CPUManager that caused freeing of exclusive CPUs at incorrect times ([#90377](https://github.com/kubernetes/kubernetes/pull/90377), [@cbf123](https://github.com/cbf123)) [SIG Cloud Provider and Node]
- Fixes regression in CPUManager that had the (rare) possibility to release exclusive CPUs in app containers inherited from init containers. ([#90419](https://github.com/kubernetes/kubernetes/pull/90419), [@klueska](https://github.com/klueska)) [SIG Node]
- Jsonpath support in kubectl / client-go serializes complex types (maps / slices / structs) as json instead of Go-syntax. ([#89660](https://github.com/kubernetes/kubernetes/pull/89660), [@pjferrell](https://github.com/pjferrell)) [SIG API Machinery, CLI and Cluster Lifecycle]
- Kubeadm: ensure `image-pull-timeout` flag is respected during upgrade phase ([#90328](https://github.com/kubernetes/kubernetes/pull/90328), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: fix misleading warning for the kube-apiserver authz modes during "kubeadm init" ([#90064](https://github.com/kubernetes/kubernetes/pull/90064), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Provides a fix to allow a cluster in a private Azure cloud to authenticate to ACR in the same cloud. ([#90425](https://github.com/kubernetes/kubernetes/pull/90425), [@DavidParks8](https://github.com/DavidParks8)) [SIG Cloud Provider]
- Update github.com/moby/ipvs to v1.0.1 to fix IPVS compatibility issue with older kernels ([#90555](https://github.com/kubernetes/kubernetes/pull/90555), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network]
- Updates to pod status via the status subresource now validate that `status.podIP` and `status.podIPs` fields are well-formed. ([#90628](https://github.com/kubernetes/kubernetes/pull/90628), [@liggitt](https://github.com/liggitt)) [SIG Apps and Node]

### Other (Cleanup or Flake)

- Drop some conformance tests that rely on Kubelet API directly ([#90615](https://github.com/kubernetes/kubernetes/pull/90615), [@dims](https://github.com/dims)) [SIG Architecture, Network, Release and Testing]
- Kube-proxy exposes a new metric, `kubeproxy_sync_proxy_rules_last_queued_timestamp_seconds`, that indicates the last time a change for kube-proxy was queued to be applied. ([#90175](https://github.com/kubernetes/kubernetes/pull/90175), [@squeed](https://github.com/squeed)) [SIG Instrumentation and Network]
- Kubeadm: fix badly formatted error message for small service CIDRs ([#90411](https://github.com/kubernetes/kubernetes/pull/90411), [@johscheuer](https://github.com/johscheuer)) [SIG Cluster Lifecycle]
- None. ([#90484](https://github.com/kubernetes/kubernetes/pull/90484), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Remove the repeated calculation of nodeName and hostname during kubelet startup, these parameters are all calculated in the `RunKubelet` method ([#90284](https://github.com/kubernetes/kubernetes/pull/90284), [@wawa0210](https://github.com/wawa0210)) [SIG Node]
- UI change ([#87743](https://github.com/kubernetes/kubernetes/pull/87743), [@u2takey](https://github.com/u2takey)) [SIG Apps and Node]
- Update opencontainers/runtime-spec dependency to v1.0.2 ([#89644](https://github.com/kubernetes/kubernetes/pull/89644), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]


# v1.19.0-alpha.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.19.0-alpha.2

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes.tar.gz) | `a1106309d18a5d73882650f8a5cbd1f287436a0dc527136808e5e882f5e98d6b0d80029ff53abc0c06ac240f6b879167437f15906e5309248d536ec1675ed909`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-src.tar.gz) | `c24c0b2a99ad0d834e0f017d7436fa84c6de8f30e8768ee59b1a418eb66a9b34ed4bcc25e03c04b19ea17366564f4ee6fe55a520fa4d0837e86c0a72fc7328c1`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `51ede026b0f8338f7fd293fb096772a67f88f23411c3280dff2f9efdd3ad7be7917d5c32ba764162c1a82b14218a90f624271c3cd8f386c8e41e4a9eac28751f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `4ed4358cabbecf724d974207746303638c7f23d422ece9c322104128c245c8485e37d6ffdd9d17e13bb1d8110e870c0fe17dcc1c9e556b69a4df7d34b6ff66d5`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `a57b10f146083828f18d809dbe07938b72216fa21083e7dbb9acce7dbcc3e8c51b8287d3bf89e81c8e1af4dd139075c675cc0f6ae7866ef69a3813db09309b97`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `099247419dd34dc78131f24f1890cc5c6a739e887c88fae96419d980c529456bfd45c4e451ba5b6425320ddc764245a2eab1bd5e2b5121d9a2774bdb5df9438b`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `d12704bc6c821d3afcd206234fbd32e57cefcb5a5d15a40434b6b0ef4781d7fa77080e490678005225f24b116540ff51e436274debf66a6eb2247cd1dc833e6c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `da0d110751fa9adac69ed2166eb82b8634989a32b65981eff014c84449047abfb94fe015e2d2e22665d57ff19f673e2c9f6549c578ad1b1e2f18b39871b50b81`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `7ac2b85bba9485dd38aed21895d627d34beb9e3b238e0684a9864f4ce2cfa67d7b3b7c04babc2ede7144d05beacdbe11c28c7d53a5b0041004700b2854b68042`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `ac447eabc5002a059e614b481d25e668735a7858134f8ad49feb388bb9f9191ff03b65da57bb49811119983e8744c8fdc7d19c184d9232bd6d038fae9eeec7c6`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `7c7dac7af329e4515302e7c35d3a19035352b4211942f254a4bb94c582a89d740b214d236ba6e35b9e78945a06b7e6fe8d70da669ecc19a40b7a9e8eaa2c0a28`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `0c89b70a25551123ffdd7c5d3cc499832454745508c5f539f13b4ea0bf6eea1afd16e316560da9cf68e5178ae69d91ccfe6c02d7054588db3fac15c30ed96f4b`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `3396e6e0516a09999ec26631e305cf0fb1eb0109ca1490837550b7635eb051dd92443de8f4321971fc2b4030ea2d8da4bfe8b85887505dec96e2a136b6a46617`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `cdea122a2d8d602ec0c89c1135ecfc27c47662982afc5b94edf4a6db7d759f27d6fe8d8b727bddf798bfec214a50e8d8a6d8eb0bca2ad5b1f72eb3768afd37f1`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `6543186a3f4437fb475fbc6a5f537640ab00afb2a22678c468c3699b3f7493f8b35fb6ca14694406ffc90ff8faad17a1d9d9d45732baa976cb69f4b27281295a`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `fde8dfeb9a0b243c8bef5127a9c63bf685429e2ff7e486ac8bae373882b87a4bd1b28a12955e3cce1c04eb0e6a67aabba43567952f9deef943a75fcb157a949c`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `399d004ee4db5d367f37a1fa9ace63b5db4522bd25eeb32225019f3df9b70c715d2159f6556015ddffe8f49aa0f72a1f095f742244637105ddbed3fb09570d0d`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `fd865c2fcc71796d73c90982f90c789a44a921cf1d56aee692bd00efaa122dcc903b0448f285a06b0a903e809f8310546764b742823fb8d10690d36ec9e27cbd`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `63aeb35222241e2a9285aeee4190b4b49c49995666db5cdb142016ca87872e7fdafc9723bc5de1797a45cc7e950230ed27be93ac165b8cda23ca2a9f9233c27a`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `3532574d9babfc064ce90099b514eadfc2a4ce69091f92d9c1a554ead91444373416d1506a35ef557438606a96cf0e5168a83ddd56c92593ea4adaa15b0b56a8`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `de59d91e5b0e4549e9a97f3a0243236e97babaed08c70f1a17273abf1966e6127db7546e1f91c3d66e933ce6eeb70bc65632ab473aa2c1be2a853da026c9d725`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `0cb8cf6f8dffd63122376a2f3e8986a2db155494a45430beea7cb5d1180417072428dabebd1af566ea13a4f079d46368c8b549be4b8a6c0f62a974290fd2fdb0`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `f1faf695f9f6fded681653f958b48779a2fecf50803af49787acba192441790c38b2b611ec8e238971508c56e67bb078fb423e8f6d9bddb392c199b5ee47937c`

## Changelog since v1.19.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

- Kubeadm now respects user specified etcd versions in the ClusterConfiguration and properly uses them. If users do not want to stick to the version specified in the ClusterConfiguration, they should edit the kubeadm-config config map and delete it. ([#89588](https://github.com/kubernetes/kubernetes/pull/89588), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]

## Changes by Kind

### API Change

- Kube-proxy: add `--bind-address-hard-fail` flag to treat failure to bind to a port as fatal ([#89350](https://github.com/kubernetes/kubernetes/pull/89350), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle and Network]
- Remove kubescheduler.config.k8s.io/v1alpha1 ([#89298](https://github.com/kubernetes/kubernetes/pull/89298), [@gavinfish](https://github.com/gavinfish)) [SIG Scheduling]
- ServiceAppProtocol feature gate is now beta and enabled by default, adding new AppProtocol field to Services and Endpoints. ([#90023](https://github.com/kubernetes/kubernetes/pull/90023), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- The Kubelet's `--volume-plugin-dir` option is now available via the Kubelet config file field `VolumePluginDir`. ([#88480](https://github.com/kubernetes/kubernetes/pull/88480), [@savitharaghunathan](https://github.com/savitharaghunathan)) [SIG Node]

### Feature

- Add client-side and server-side dry-run support to kubectl scale ([#89666](https://github.com/kubernetes/kubernetes/pull/89666), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Add support for cgroups v2 node validation ([#89901](https://github.com/kubernetes/kubernetes/pull/89901), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
- Detailed scheduler scoring result can be printed at verbose level 10. ([#89384](https://github.com/kubernetes/kubernetes/pull/89384), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- E2e.test can print the list of conformance tests that need to pass for the cluster to be conformant. ([#88924](https://github.com/kubernetes/kubernetes/pull/88924), [@dims](https://github.com/dims)) [SIG Architecture and Testing]
- Feat: add azure shared disk support ([#89511](https://github.com/kubernetes/kubernetes/pull/89511), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Kube-apiserver backed by etcd3 exports metric showing the database file size. ([#89151](https://github.com/kubernetes/kubernetes/pull/89151), [@jingyih](https://github.com/jingyih)) [SIG API Machinery]
- Kube-apiserver: The NodeRestriction admission plugin now restricts Node labels kubelets are permitted to set when creating a new Node to the `--node-labels` parameters accepted by kubelets in 1.16+. ([#90307](https://github.com/kubernetes/kubernetes/pull/90307), [@liggitt](https://github.com/liggitt)) [SIG Auth and Node]
- Kubeadm: during 'upgrade apply', if the kube-proxy ConfigMap is missing, assume that kube-proxy should not be upgraded. Same applies to a missing kube-dns/coredns ConfigMap for the DNS server addon. Note that this is a temporary workaround until 'upgrade apply' supports phases. Once phases are supported the kube-proxy/dns upgrade should be skipped manually. ([#89593](https://github.com/kubernetes/kubernetes/pull/89593), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: switch control-plane static Pods to the "system-node-critical" priority class ([#90063](https://github.com/kubernetes/kubernetes/pull/90063), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Support for running on a host that uses cgroups v2 unified mode ([#85218](https://github.com/kubernetes/kubernetes/pull/85218), [@giuseppe](https://github.com/giuseppe)) [SIG Node]
- Update etcd client side to v3.4.7 ([#89822](https://github.com/kubernetes/kubernetes/pull/89822), [@jingyih](https://github.com/jingyih)) [SIG API Machinery and Cloud Provider]

### Bug or Regression

- An issue preventing GCP cloud-controller-manager running out-of-cluster to initialize new Nodes is now fixed. ([#90057](https://github.com/kubernetes/kubernetes/pull/90057), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Apps and Cloud Provider]
- Avoid unnecessary GCE API calls when adding IP alises or reflecting them in Node object in GCE cloud provider. ([#90242](https://github.com/kubernetes/kubernetes/pull/90242), [@wojtek-t](https://github.com/wojtek-t)) [SIG Apps, Cloud Provider and Network]
- Azure: fix concurreny issue in lb creation ([#89604](https://github.com/kubernetes/kubernetes/pull/89604), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Bug fix for AWS NLB service when nodePort for existing servicePort changed manually. ([#89562](https://github.com/kubernetes/kubernetes/pull/89562), [@M00nF1sh](https://github.com/M00nF1sh)) [SIG Cloud Provider]
- CSINode initialization does not crash kubelet on startup when APIServer is not reachable or kubelet has not the right credentials yet. ([#89589](https://github.com/kubernetes/kubernetes/pull/89589), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Client-go: resolves an issue with informers falling back to full list requests when timeouts are encountered, rather than re-establishing a watch. ([#89652](https://github.com/kubernetes/kubernetes/pull/89652), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Dual-stack: fix the bug that Service clusterIP does not respect specified ipFamily ([#89612](https://github.com/kubernetes/kubernetes/pull/89612), [@SataQiu](https://github.com/SataQiu)) [SIG Network]
- Ensure Azure availability zone is always in lower cases. ([#89722](https://github.com/kubernetes/kubernetes/pull/89722), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Explain CRDs whose resource name are the same as builtin objects ([#89505](https://github.com/kubernetes/kubernetes/pull/89505), [@knight42](https://github.com/knight42)) [SIG API Machinery, CLI and Testing]
- Fix flaws in Azure File CSI translation ([#90162](https://github.com/kubernetes/kubernetes/pull/90162), [@rfranzke](https://github.com/rfranzke)) [SIG Release and Storage]
- Fix kubectl describe CSINode nil pointer error ([#89646](https://github.com/kubernetes/kubernetes/pull/89646), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix kubectl diff so it doesn't actually persist patches ([#89795](https://github.com/kubernetes/kubernetes/pull/89795), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Fix kubectl version should print version info without config file ([#89913](https://github.com/kubernetes/kubernetes/pull/89913), [@zhouya0](https://github.com/zhouya0)) [SIG API Machinery and CLI]
- Fix missing `-c` shorthand for `--container` flag of `kubectl alpha debug` ([#89674](https://github.com/kubernetes/kubernetes/pull/89674), [@superbrothers](https://github.com/superbrothers)) [SIG CLI]
- Fix printers ignoring object average value ([#89142](https://github.com/kubernetes/kubernetes/pull/89142), [@zhouya0](https://github.com/zhouya0)) [SIG API Machinery]
- Fix scheduler crash when removing node before its pods ([#89908](https://github.com/kubernetes/kubernetes/pull/89908), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Fix: get attach disk error due to missing item in max count table ([#89768](https://github.com/kubernetes/kubernetes/pull/89768), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fixed a bug where executing a kubectl command with a jsonpath output expression that has a nested range would ignore expressions following the nested range. ([#88464](https://github.com/kubernetes/kubernetes/pull/88464), [@brianpursley](https://github.com/brianpursley)) [SIG API Machinery]
- Fixed a regression running kubectl commands with  --local or --dry-run flags when no kubeconfig file is present ([#90243](https://github.com/kubernetes/kubernetes/pull/90243), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI and Testing]
- Fixed an issue mounting credentials for service accounts whose name contains `.` characters ([#89696](https://github.com/kubernetes/kubernetes/pull/89696), [@nabokihms](https://github.com/nabokihms)) [SIG Auth]
- Fixed mountOptions in iSCSI and FibreChannel volume plugins. ([#89172](https://github.com/kubernetes/kubernetes/pull/89172), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixed the EndpointSlice controller to run without error on a cluster with the OwnerReferencesPermissionEnforcement validating admission plugin enabled. ([#89741](https://github.com/kubernetes/kubernetes/pull/89741), [@marun](https://github.com/marun)) [SIG Auth and Network]
- Fixes a bug defining a default value for a replicas field in a custom resource definition that has the scale subresource enabled ([#89833](https://github.com/kubernetes/kubernetes/pull/89833), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Fixes conversion error for HorizontalPodAutoscaler objects with invalid annotations ([#89963](https://github.com/kubernetes/kubernetes/pull/89963), [@liggitt](https://github.com/liggitt)) [SIG Autoscaling]
- Fixes kubectl to apply all validly built objects, instead of stopping on error. ([#89848](https://github.com/kubernetes/kubernetes/pull/89848), [@seans3](https://github.com/seans3)) [SIG CLI and Testing]
- For GCE cluster provider, fix bug of not being able to create internal type load balancer for clusters with more than 1000 nodes in a single zone. ([#89902](https://github.com/kubernetes/kubernetes/pull/89902), [@wojtek-t](https://github.com/wojtek-t)) [SIG Cloud Provider, Network and Scalability]
- If firstTimestamp is not set use eventTime when printing event ([#89999](https://github.com/kubernetes/kubernetes/pull/89999), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- If we set parameter cgroupPerQos=false and cgroupRoot=/docker，this function will retrun  nodeAllocatableRoot=/docker/kubepods, it is not right, the correct return should be /docker.
  cm.NodeAllocatableRoot(s.CgroupRoot, s.CgroupDriver)
  
  kubeDeps.CAdvisorInterface, err = cadvisor.New(imageFsInfoProvider, s.RootDirectory, cgroupRoots, cadvisor.UsingLegacyCadvisorStats(s.ContainerRuntime, s.RemoteRuntimeEndpoint))
  the above function，as we use cgroupRoots to create cadvisor interface，the wrong parameter cgroupRoots will lead eviction manager not  to collect metric from /docker, then kubelet frequently print those error：
  E0303 17:25:03.436781 63839 summary_sys_containers.go:47] Failed to get system container stats for "/docker": failed to get cgroup stats for "/docker": failed to get container info for "/docker": unknown container "/docker"
  E0303 17:25:03.436809 63839 helpers.go:680] eviction manager: failed to construct signal: "allocatableMemory.available" error: system container "pods" not found in metrics ([#88970](https://github.com/kubernetes/kubernetes/pull/88970), [@mysunshine92](https://github.com/mysunshine92)) [SIG Node]
- In the kubelet resource metrics endpoint at /metrics/resource, change the names of the following metrics:
  - node_cpu_usage_seconds --> node_cpu_usage_seconds_total
  - container_cpu_usage_seconds --> container_cpu_usage_seconds_total
  This is a partial revert of &#35;86282, which was added in 1.18.0, and initially removed the _total suffix ([#89540](https://github.com/kubernetes/kubernetes/pull/89540), [@dashpole](https://github.com/dashpole)) [SIG Instrumentation and Node]
- Kube-apiserver: multiple comma-separated protocols in a single X-Stream-Protocol-Version header are now recognized, in addition to multiple headers, complying with RFC2616 ([#89857](https://github.com/kubernetes/kubernetes/pull/89857), [@tedyu](https://github.com/tedyu)) [SIG API Machinery]
- Kubeadm increased to 5 minutes its timeout for the TLS bootstrapping process to complete upon join ([#89735](https://github.com/kubernetes/kubernetes/pull/89735), [@rosti](https://github.com/rosti)) [SIG Cluster Lifecycle]
- Kubeadm: during join when a check is performed that a Node with the same name already exists in the cluster, make sure the NodeReady condition is properly validated ([#89602](https://github.com/kubernetes/kubernetes/pull/89602), [@kvaps](https://github.com/kvaps)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where post upgrade to 1.18.x, nodes cannot join the cluster due to missing RBAC ([#89537](https://github.com/kubernetes/kubernetes/pull/89537), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix misleading warning about passing control-plane related flags on 'kubeadm join' ([#89596](https://github.com/kubernetes/kubernetes/pull/89596), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl azure authentication: fixed a regression in 1.18.0 where "spn:" prefix was unexpectedly added to the `apiserver-id` configuration in the kubeconfig file ([#89706](https://github.com/kubernetes/kubernetes/pull/89706), [@weinong](https://github.com/weinong)) [SIG API Machinery and Auth]
- Restore the ability to `kubectl apply --prune` without --namespace flag.  Since 1.17, `kubectl apply --prune` only prunes resources in the default namespace (or from kubeconfig) or explicitly specified in command line flag.  But this is s breaking change from kubectl 1.16, which can prune resources in all namespace in config file.  This patch restores the kubectl 1.16 behaviour. ([#89551](https://github.com/kubernetes/kubernetes/pull/89551), [@tatsuhiro-t](https://github.com/tatsuhiro-t)) [SIG CLI and Testing]
- Restores priority of static control plane pods in the cluster/gce/manifests control-plane manifests ([#89970](https://github.com/kubernetes/kubernetes/pull/89970), [@liggitt](https://github.com/liggitt)) [SIG Cluster Lifecycle and Node]
- Service account tokens bound to pods can now be used during the pod deletion grace period. ([#89583](https://github.com/kubernetes/kubernetes/pull/89583), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Sync LB backend nodes for Service Type=LoadBalancer on Add/Delete node events. ([#81185](https://github.com/kubernetes/kubernetes/pull/81185), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]

### Other (Cleanup or Flake)

- Change beta.kubernetes.io/os  to kubernetes.io/os ([#89460](https://github.com/kubernetes/kubernetes/pull/89460), [@wawa0210](https://github.com/wawa0210)) [SIG Testing and Windows]
- Changes not found message when using `kubectl get` to retrieve not namespaced resources ([#89861](https://github.com/kubernetes/kubernetes/pull/89861), [@rccrdpccl](https://github.com/rccrdpccl)) [SIG CLI]
- Node ([#76443](https://github.com/kubernetes/kubernetes/pull/76443), [@mgdevstack](https://github.com/mgdevstack)) [SIG Architecture, Network, Node, Testing and Windows]
- None. ([#90273](https://github.com/kubernetes/kubernetes/pull/90273), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Reduce event spam during a volume operation error. ([#89794](https://github.com/kubernetes/kubernetes/pull/89794), [@msau42](https://github.com/msau42)) [SIG Storage]
- The PR adds functionality to generate events when a PV or PVC processing encounters certain failures. The events help users to know the reason for the failure so they can take necessary recovery actions. ([#89845](https://github.com/kubernetes/kubernetes/pull/89845), [@yuga711](https://github.com/yuga711)) [SIG Apps]
- The PodShareProcessNamespace feature gate has been removed, and the PodShareProcessNamespace is unconditionally enabled. ([#90099](https://github.com/kubernetes/kubernetes/pull/90099), [@tanjunchen](https://github.com/tanjunchen)) [SIG Node]
- Update default etcd server version to 3.4.4 ([#89214](https://github.com/kubernetes/kubernetes/pull/89214), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cluster Lifecycle and Testing]
- Update default etcd server version to 3.4.7 ([#89895](https://github.com/kubernetes/kubernetes/pull/89895), [@jingyih](https://github.com/jingyih)) [SIG API Machinery, Cluster Lifecycle and Testing]


# v1.19.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.19.0-alpha.1

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes.tar.gz) | `d5930e62f98948e3ae2bc0a91b2cb93c2009202657b9e798e43fcbf92149f50d991af34a49049b2640db729efc635d643d008f4b3dd6c093cac4426ee3d5d147`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-src.tar.gz) | `5d92125ec3ca26b6b0af95c6bb3289bb7cf60a4bad4e120ccdad06ffa523c239ca8e608015b7b5a1eb789bfdfcedbe0281518793da82a7959081fb04cf53c174`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `08d307dafdd8e1aa27721f97f038210b33261d1777ea173cc9ed4b373c451801988a7109566425fce32d38df70bdf0be6b8cfff69da768fbd3c303abd6dc13a5`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `08c3b722a62577d051e300ebc3c413ead1bd3e79555598a207c704064116087323215fb402bae7584b9ffd08590f36fa8a35f13f8fea1ce92e8f144e3eae3384`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `0735978b4d4cb0601171eae3cc5603393c00f032998f51d79d3b11e4020f4decc9559905e9b02ddcb0b6c3f4caf78f779940ebc97996e3b96b98ba378fbe189d`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `ca55fc431d59c1a0bf1f1c248da7eab65215e438fcac223d4fc3a57fae0205869e1727b2475dfe9b165921417d68ac380a6e42bf7ea6732a34937ba2590931ce`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `4e1aa9e640d7cf0ccaad19377e4c3ca9a60203daa2ce0437d1d40fdea0e43759ef38797e948cdc3c676836b01e83f1bfde51effc0579bf832f6f062518f03f06`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `fca5df8c2919a9b3d99248120af627d9a1b5ddf177d9a10f04eb4e486c14d4e3ddb72e3abc4733b5078e0d27204a51e2f714424923fb92a5351137f82d87d6ea`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `6a98a4f99aa8b72ec815397c5062b90d5c023092da28fa7bca1cdadf406e2d86e2fd3a0eeab28574064959c6926007423c413d9781461e433705452087430d57`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `94724c17985ae2dbd3888e6896f300f95fec8dc2bf08e768849e98b05affc4381b322d802f41792b8e6da4708ce1ead2edcb8f4d5299be6267f6559b0d49e484`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `5a076bf3a5926939c170a501f8292a38003552848c45c1f148a97605b7ac9843fb660ef81a46abe6d139f4c5eaa342d4b834a799ee7055d5a548d189b31d7124`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `4b395894bfd9cfa0976512d1d58c0056a80bacefc798de294db6d3f363bd5581fd3ce2e4bdc1b902d46c8ce2ac87a98ced56b6b29544c86e8444fb8e9465faea`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `6720d1b826dc20e56b0314e580403cd967430ff25bdbe08e8bf453fed339557d2a4ace114c2f524e6b6814ec9341ccdea870f784ebb53a52056ca3ab22e5cc36`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `f09b295f5a95cc72494eb1c0e9706b237a8523eacda182778e9afdb469704c7eacd29614aff6d3d7aff3bc1783fb277d52ad56a1417f1bd973eeb9bdc8086695`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `24787767abd1d67a4d0234433e1693ea3e1e906364265ee03e58ba203b66583b75d4ce0c4185756fc529997eb9a842d65841962cd228df9c182a469dbd72493d`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `a117e609729263d7bd58aac156efa33941f0f9aa651892d1abf32cfa0a984aa495fccd3be8385cae083415bfa8f81942648d5978f72e950103e42184fd0d7527`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `19280a6dc20f019d23344934f8f1ec6aa17c3374b9c569d4c173535a8cd9e298b8afcabe06d232a146c9c7cb4bfe7d1d0e10aa2ab9184ace0b7987e36973aaef`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `c4b23f113ed13edb91b59a498d15de8b62ff1005243f2d6654a11468511c9d0ebaebb6dc02d2fa505f18df446c9221e77d7fc3147fa6704cde9bec5d6d80b5a3`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `8dcf5531a5809576049c455d3c5194f09ddf3b87995df1e8ca4543deff3ffd90a572539daff9aa887e22efafedfcada2e28035da8573e3733c21778e4440677a`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `4b3f4dfee2034ce7d01fef57b8766851fe141fc72da0f9edeb39aca4c7a937e2dccd2c198a83fbb92db7911d81e50a98bd0a17b909645adbeb26e420197db2cd`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `df0e87f5e42056db2bbc7ef5f08ecda95d66afc3f4d0bc57f6efcc05834118c39ab53d68595d8f2bb278829e33b9204c5cce718d8bf841ce6cccbb86d0d20730`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `3a6499b008a68da52f8ae12eb694885d9e10a8f805d98f28fc5f7beafea72a8e180df48b5ca31097b2d4779c61ff67216e516c14c2c812163e678518d95f22d6`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.19.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `c311373506cbfa0244ac92a709fbb9bddb46cbeb130733bdb689641ecee6b21a7a7f020eae4856a3f04a3845839dc5e0914cddc3478d55cd3d5af3d7804aa5ba`

## Changelog since v1.19.0-alpha.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

- The StreamingProxyRedirects feature and `--redirect-container-streaming` flag are deprecated, and will be removed in a future release. The default behavior (proxy streaming requests through the kubelet) will be the only supported option.
  If you are setting `--redirect-container-streaming=true`, then you must migrate off this configuration. The flag will no longer be able to be enabled starting in v1.20. If you are not setting the flag, no action is necessary. ([#88290](https://github.com/kubernetes/kubernetes/pull/88290), [@tallclair](https://github.com/tallclair)) [SIG API Machinery and Node]

- `kubectl` no longer defaults to `http://localhost:8080`.  If you own one of these legacy clusters, you are *strongly- encouraged to secure your server.   If you cannot secure your server, you can set `KUBERNETES_MASTER` if you were relying on that behavior and you're client-go user. Set `--server`, `--kubeconfig` or `KUBECONFIG` to make it work in `kubectl`. ([#86173](https://github.com/kubernetes/kubernetes/pull/86173), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI and Testing]

## Changes by Kind

### Deprecation

- AlgorithmSource is removed from v1alpha2 Scheduler ComponentConfig ([#87999](https://github.com/kubernetes/kubernetes/pull/87999), [@damemi](https://github.com/damemi)) [SIG Scheduling]
- Azure service annotation service.beta.kubernetes.io/azure-load-balancer-disable-tcp-reset has been deprecated. Its support would be removed in a future release. ([#88462](https://github.com/kubernetes/kubernetes/pull/88462), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Kube-proxy: deprecate `--healthz-port` and `--metrics-port` flag, please use `--healthz-bind-address` and `--metrics-bind-address` instead ([#88512](https://github.com/kubernetes/kubernetes/pull/88512), [@SataQiu](https://github.com/SataQiu)) [SIG Network]
- Kubeadm: deprecate the usage of the experimental flag '--use-api' under the 'kubeadm alpha certs renew' command. ([#88827](https://github.com/kubernetes/kubernetes/pull/88827), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes no longer supports building hyperkube images ([#88676](https://github.com/kubernetes/kubernetes/pull/88676), [@dims](https://github.com/dims)) [SIG Cluster Lifecycle and Release]

### API Change

- A new IngressClass resource has been added to enable better Ingress configuration. ([#88509](https://github.com/kubernetes/kubernetes/pull/88509), [@robscott](https://github.com/robscott)) [SIG API Machinery, Apps, CLI, Network, Node and Testing]
- API additions to apiserver types ([#87179](https://github.com/kubernetes/kubernetes/pull/87179), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Cloud Provider and Cluster Lifecycle]
- Add Scheduling Profiles to kubescheduler.config.k8s.io/v1alpha2 ([#88087](https://github.com/kubernetes/kubernetes/pull/88087), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling and Testing]
- Added GenericPVCDataSource feature gate to enable using arbitrary custom resources as the data source for a PVC. ([#88636](https://github.com/kubernetes/kubernetes/pull/88636), [@bswartz](https://github.com/bswartz)) [SIG Apps and Storage]
- Added support for multiple sizes huge pages on a container level ([#84051](https://github.com/kubernetes/kubernetes/pull/84051), [@bart0sh](https://github.com/bart0sh)) [SIG Apps, Node and Storage]
- Allow user to specify fsgroup permission change policy for pods ([#88488](https://github.com/kubernetes/kubernetes/pull/88488), [@gnufied](https://github.com/gnufied)) [SIG Apps and Storage]
- AppProtocol is a new field on Service and Endpoints resources, enabled with the ServiceAppProtocol feature gate. ([#88503](https://github.com/kubernetes/kubernetes/pull/88503), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- BlockVolume and CSIBlockVolume features are now GA. ([#88673](https://github.com/kubernetes/kubernetes/pull/88673), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Node and Storage]
- Consumers of the 'certificatesigningrequests/approval' API must now grant permission to 'approve' CSRs for the 'signerName' specified on the CSR. More information on the new signerName field can be found at https://github.com/kubernetes/enhancements/blob/master/keps/sig-auth/20190607-certificates-api.md&#35;signers ([#88246](https://github.com/kubernetes/kubernetes/pull/88246), [@munnerz](https://github.com/munnerz)) [SIG API Machinery, Apps, Auth, CLI, Node and Testing]
- CustomResourceDefinition schemas that use `x-kubernetes-list-map-keys` to specify properties that uniquely identify list items must make those properties required or have a default value, to ensure those properties are present for all list items. See https://kubernetes.io/docs/reference/using-api/api-concepts/&#35;merge-strategy for details. ([#88076](https://github.com/kubernetes/kubernetes/pull/88076), [@eloyekunle](https://github.com/eloyekunle)) [SIG API Machinery and Testing]
- Fixed missing validation of uniqueness of list items in lists with `x-kubernetes-list-type: map` or x-kubernetes-list-type: set` in CustomResources. ([#84920](https://github.com/kubernetes/kubernetes/pull/84920), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Fixes a regression with clients prior to 1.15 not being able to update podIP in pod status, or podCIDR in node spec, against >= 1.16 API servers ([#88505](https://github.com/kubernetes/kubernetes/pull/88505), [@liggitt](https://github.com/liggitt)) [SIG Apps and Network]
- Ingress: Add Exact and Prefix maching to Ingress PathTypes ([#88587](https://github.com/kubernetes/kubernetes/pull/88587), [@cmluciano](https://github.com/cmluciano)) [SIG Apps, Cluster Lifecycle and Network]
- Ingress: Add alternate backends via TypedLocalObjectReference ([#88775](https://github.com/kubernetes/kubernetes/pull/88775), [@cmluciano](https://github.com/cmluciano)) [SIG Apps and Network]
- Ingress: allow wildcard hosts in IngressRule ([#88858](https://github.com/kubernetes/kubernetes/pull/88858), [@cmluciano](https://github.com/cmluciano)) [SIG Network]
- Introduces optional --detect-local flag to kube-proxy. 
  Currently the only supported value is "cluster-cidr", 
  which is the default if not specified. ([#87748](https://github.com/kubernetes/kubernetes/pull/87748), [@satyasm](https://github.com/satyasm)) [SIG Cluster Lifecycle, Network and Scheduling]
- Kube-controller-manager and kube-scheduler expose profiling by default to match the kube-apiserver.  Use `--enable-profiling=false` to disable. ([#88663](https://github.com/kubernetes/kubernetes/pull/88663), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Cloud Provider and Scheduling]
- Kube-scheduler can run more than one scheduling profile. Given a pod, the profile is selected by using its `.spec.SchedulerName`. ([#88285](https://github.com/kubernetes/kubernetes/pull/88285), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps, Scheduling and Testing]
- Move TaintBasedEvictions feature gates to GA ([#87487](https://github.com/kubernetes/kubernetes/pull/87487), [@skilxn-go](https://github.com/skilxn-go)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Moving Windows RunAsUserName feature to GA ([#87790](https://github.com/kubernetes/kubernetes/pull/87790), [@marosset](https://github.com/marosset)) [SIG Apps and Windows]
- New flag --endpointslice-updates-batch-period in kube-controller-manager can be used to reduce number of endpointslice updates generated by pod changes. ([#88745](https://github.com/kubernetes/kubernetes/pull/88745), [@mborsz](https://github.com/mborsz)) [SIG API Machinery, Apps and Network]
- New flag `--show-hidden-metrics-for-version` in kubelet can be used to show all hidden metrics that deprecated in the previous minor release. ([#85282](https://github.com/kubernetes/kubernetes/pull/85282), [@serathius](https://github.com/serathius)) [SIG Node]
- Removes ConfigMap as suggestion for IngressClass parameters ([#89093](https://github.com/kubernetes/kubernetes/pull/89093), [@robscott](https://github.com/robscott)) [SIG Network]
- Scheduler Extenders can now be configured in the v1alpha2 component config ([#88768](https://github.com/kubernetes/kubernetes/pull/88768), [@damemi](https://github.com/damemi)) [SIG Release, Scheduling and Testing]
- The apiserver/v1alph1&#35;EgressSelectorConfiguration API is now beta. ([#88502](https://github.com/kubernetes/kubernetes/pull/88502), [@caesarxuchao](https://github.com/caesarxuchao)) [SIG API Machinery]
- The storage.k8s.io/CSIDriver has moved to GA, and is now available for use. ([#84814](https://github.com/kubernetes/kubernetes/pull/84814), [@huffmanca](https://github.com/huffmanca)) [SIG API Machinery, Apps, Auth, Node, Scheduling, Storage and Testing]
- VolumePVCDataSource moves to GA in 1.18 release ([#88686](https://github.com/kubernetes/kubernetes/pull/88686), [@j-griffith](https://github.com/j-griffith)) [SIG Apps, CLI and Cluster Lifecycle]

### Feature

- deps: Update to Golang 1.13.9
  - build: Remove kube-cross image building ([#89275](https://github.com/kubernetes/kubernetes/pull/89275), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- Add --dry-run to kubectl delete, taint, replace ([#88292](https://github.com/kubernetes/kubernetes/pull/88292), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Add `rest_client_rate_limiter_duration_seconds` metric to component-base to track client side rate limiter latency in seconds. Broken down by verb and URL. ([#88134](https://github.com/kubernetes/kubernetes/pull/88134), [@jennybuckley](https://github.com/jennybuckley)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Add huge page stats to Allocated resources in "kubectl describe node" ([#80605](https://github.com/kubernetes/kubernetes/pull/80605), [@odinuge](https://github.com/odinuge)) [SIG CLI]
- Add support for pre allocated huge pages with different sizes, on node level ([#89252](https://github.com/kubernetes/kubernetes/pull/89252), [@odinuge](https://github.com/odinuge)) [SIG Apps and Node]
- Adds support for NodeCIDR as an argument to --detect-local-mode ([#88935](https://github.com/kubernetes/kubernetes/pull/88935), [@satyasm](https://github.com/satyasm)) [SIG Network]
- Allow user to specify resource using --filename flag when invoking kubectl exec ([#88460](https://github.com/kubernetes/kubernetes/pull/88460), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Apiserver add a new flag --goaway-chance which is the fraction of requests that will be closed gracefully(GOAWAY) to prevent HTTP/2 clients from getting stuck on a single apiserver. 
  After the connection closed(received GOAWAY), the client's other in-flight requests won't be affected, and the client will reconnect. 
  The flag min value is 0 (off), max is .02 (1/50 requests); .001 (1/1000) is a recommended starting point.
  Clusters with single apiservers, or which don't use a load balancer, should NOT enable this. ([#88567](https://github.com/kubernetes/kubernetes/pull/88567), [@answer1991](https://github.com/answer1991)) [SIG API Machinery]
- Azure Cloud Provider now supports using Azure network resources (Virtual Network, Load Balancer, Public IP, Route Table, Network Security Group, etc.) in different AAD Tenant and Subscription than those for the Kubernetes cluster. To use the feature, please reference https://github.com/kubernetes-sigs/cloud-provider-azure/blob/master/docs/cloud-provider-config.md&#35;host-network-resources-in-different-aad-tenant-and-subscription. ([#88384](https://github.com/kubernetes/kubernetes/pull/88384), [@bowen5](https://github.com/bowen5)) [SIG Cloud Provider]
- Azure: add support for single stack IPv6 ([#88448](https://github.com/kubernetes/kubernetes/pull/88448), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- DefaultConstraints can be specified for the PodTopologySpread plugin in the component config ([#88671](https://github.com/kubernetes/kubernetes/pull/88671), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- EndpointSlice controller waits longer to retry failed sync. ([#89438](https://github.com/kubernetes/kubernetes/pull/89438), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Feat: change azure disk api-version ([#89250](https://github.com/kubernetes/kubernetes/pull/89250), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Feat: support [Azure shared disk](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/disks-shared-enable), added a new field(`maxShares`) in azure disk storage class:
  
  kind: StorageClass
  apiVersion: storage.k8s.io/v1
  metadata:
    name: shared-disk
  provisioner: kubernetes.io/azure-disk
  parameters:
    skuname: Premium_LRS  &#35; Currently only available with premium SSDs.
    cachingMode: None  &#35; ReadOnly host caching is not available for premium SSDs with maxShares>1
    maxShares: 2 ([#89328](https://github.com/kubernetes/kubernetes/pull/89328), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Kube-apiserver, kube-scheduler and kube-controller manager now use SO_REUSEPORT socket option when listening on address defined by --bind-address and --secure-port flags, when running on Unix systems (Windows is NOT supported). This allows to run multiple instances of those processes on a single host with the same configuration, which allows to update/restart them in a graceful way, without causing downtime. ([#88893](https://github.com/kubernetes/kubernetes/pull/88893), [@invidian](https://github.com/invidian)) [SIG API Machinery, Scheduling and Testing]
- Kubeadm: The ClusterStatus struct present in the kubeadm-config ConfigMap is deprecated and will be removed on a future version. It is going to be maintained by kubeadm until it gets removed. The same information can be found on `etcd` and `kube-apiserver` pod annotations, `kubeadm.kubernetes.io/etcd.advertise-client-urls` and `kubeadm.kubernetes.io/kube-apiserver.advertise-address.endpoint` respectively. ([#87656](https://github.com/kubernetes/kubernetes/pull/87656), [@ereslibre](https://github.com/ereslibre)) [SIG Cluster Lifecycle]
- Kubeadm: add the experimental feature gate PublicKeysECDSA that can be used to create a
  cluster with ECDSA certificates from "kubeadm init". Renewal of existing ECDSA certificates is
  also supported using "kubeadm alpha certs renew", but not switching between the RSA and
  ECDSA algorithms on the fly or during upgrades. ([#86953](https://github.com/kubernetes/kubernetes/pull/86953), [@rojkov](https://github.com/rojkov)) [SIG API Machinery, Auth and Cluster Lifecycle]
- Kubeadm: on kubeconfig certificate renewal, keep the embedded CA in sync with the one on disk ([#88052](https://github.com/kubernetes/kubernetes/pull/88052), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: support Windows specific kubelet flags in kubeadm-flags.env ([#88287](https://github.com/kubernetes/kubernetes/pull/88287), [@gab-satchi](https://github.com/gab-satchi)) [SIG Cluster Lifecycle and Windows]
- Kubeadm: upgrade supports fallback to the nearest known etcd version if an unknown k8s version is passed ([#88373](https://github.com/kubernetes/kubernetes/pull/88373), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl cluster-info dump changed to only display a message telling you the location where the output was written when the output is not standard output. ([#88765](https://github.com/kubernetes/kubernetes/pull/88765), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- New flag `--show-hidden-metrics-for-version` in kube-scheduler can be used to show all hidden metrics that deprecated in the previous minor release. ([#84913](https://github.com/kubernetes/kubernetes/pull/84913), [@serathius](https://github.com/serathius)) [SIG Instrumentation and Scheduling]
- Print NotReady when pod is not ready based on its conditions. ([#88240](https://github.com/kubernetes/kubernetes/pull/88240), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- Scheduler Extender API is now located under k8s.io/kube-scheduler/extender ([#88540](https://github.com/kubernetes/kubernetes/pull/88540), [@damemi](https://github.com/damemi)) [SIG Release, Scheduling and Testing]
- Scheduler framework permit plugins now run at the end of the scheduling cycle, after reserve plugins. Waiting on permit will remain in the beginning of the binding cycle. ([#88199](https://github.com/kubernetes/kubernetes/pull/88199), [@mateuszlitwin](https://github.com/mateuszlitwin)) [SIG Scheduling]
- Signatures on scale client methods have been modified to accept `context.Context` as a first argument. Signatures of Get, Update, and Patch methods have been updated to accept GetOptions, UpdateOptions and PatchOptions respectively. ([#88599](https://github.com/kubernetes/kubernetes/pull/88599), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG API Machinery, Apps, Autoscaling and CLI]
- Signatures on the dynamic client methods have been modified to accept `context.Context` as a first argument. Signatures of Delete and DeleteCollection methods now accept DeleteOptions by value instead of by reference. ([#88906](https://github.com/kubernetes/kubernetes/pull/88906), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, CLI, Cluster Lifecycle, Storage and Testing]
- Signatures on the metadata client methods have been modified to accept `context.Context` as a first argument. Signatures of Delete and DeleteCollection methods now accept DeleteOptions by value instead of by reference. ([#88910](https://github.com/kubernetes/kubernetes/pull/88910), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps and Testing]
- Support create or update VMSS asynchronously. ([#89248](https://github.com/kubernetes/kubernetes/pull/89248), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- The kubelet and the default docker runtime now support running ephemeral containers in the Linux process namespace of a target container. Other container runtimes must implement this feature before it will be available in that runtime. ([#84731](https://github.com/kubernetes/kubernetes/pull/84731), [@verb](https://github.com/verb)) [SIG Node]
- Update etcd client side to v3.4.4 ([#89169](https://github.com/kubernetes/kubernetes/pull/89169), [@jingyih](https://github.com/jingyih)) [SIG API Machinery and Cloud Provider]
- Upgrade to azure-sdk v40.2.0 ([#89105](https://github.com/kubernetes/kubernetes/pull/89105), [@andyzhangx](https://github.com/andyzhangx)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- Webhooks will have alpha support for network proxy ([#85870](https://github.com/kubernetes/kubernetes/pull/85870), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Auth and Testing]
- When client certificate files are provided, reload files for new connections, and close connections when a certificate changes. ([#79083](https://github.com/kubernetes/kubernetes/pull/79083), [@jackkleeman](https://github.com/jackkleeman)) [SIG API Machinery, Auth, Node and Testing]
- When deleting objects using kubectl with the --force flag, you are no longer required to also specify --grace-period=0. ([#87776](https://github.com/kubernetes/kubernetes/pull/87776), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- `kubectl` now contains a `kubectl alpha debug` command. This command allows attaching an ephemeral container to a running pod for the purposes of debugging. ([#88004](https://github.com/kubernetes/kubernetes/pull/88004), [@verb](https://github.com/verb)) [SIG CLI]

### Documentation

- Improved error message for incorrect auth field. ([#82829](https://github.com/kubernetes/kubernetes/pull/82829), [@martin-schibsted](https://github.com/martin-schibsted)) [SIG Auth]
- Update Japanese translation for kubectl help ([#86837](https://github.com/kubernetes/kubernetes/pull/86837), [@inductor](https://github.com/inductor)) [SIG CLI and Docs]
- Updated the instructions for deploying the sample app. ([#82785](https://github.com/kubernetes/kubernetes/pull/82785), [@ashish-billore](https://github.com/ashish-billore)) [SIG API Machinery]
- `kubectl plugin` now prints a note how to install krew ([#88577](https://github.com/kubernetes/kubernetes/pull/88577), [@corneliusweig](https://github.com/corneliusweig)) [SIG CLI]

### Other (Bug, Cleanup or Flake)

- A PV set from in-tree source will have ordered requirement values in NodeAffinity when converted to CSIPersistentVolumeSource ([#88987](https://github.com/kubernetes/kubernetes/pull/88987), [@jiahuif](https://github.com/jiahuif)) [SIG Storage]
- Add delays between goroutines for vm instance update ([#88094](https://github.com/kubernetes/kubernetes/pull/88094), [@aramase](https://github.com/aramase)) [SIG Cloud Provider]
- Add init containers log to cluster dump info. ([#88324](https://github.com/kubernetes/kubernetes/pull/88324), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Azure VMSS LoadBalancerBackendAddressPools updating has been improved with squential-sync + concurrent-async requests. ([#88699](https://github.com/kubernetes/kubernetes/pull/88699), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Azure auth module for kubectl now requests login after refresh token expires. ([#86481](https://github.com/kubernetes/kubernetes/pull/86481), [@tdihp](https://github.com/tdihp)) [SIG API Machinery and Auth]
- AzureFile and CephFS use new Mount library that prevents logging of sensitive mount options. ([#88684](https://github.com/kubernetes/kubernetes/pull/88684), [@saad-ali](https://github.com/saad-ali)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Beta.kubernetes.io/arch is already deprecated since v1.14, are targeted for removal in v1.18 ([#89462](https://github.com/kubernetes/kubernetes/pull/89462), [@wawa0210](https://github.com/wawa0210)) [SIG Testing]
- Build: Enable kube-cross image-building on K8s Infra ([#88562](https://github.com/kubernetes/kubernetes/pull/88562), [@justaugustus](https://github.com/justaugustus)) [SIG Release and Testing]
- CPU limits are now respected for Windows containers. If a node is over-provisioned, no weighting is used - only limits are respected. ([#86101](https://github.com/kubernetes/kubernetes/pull/86101), [@PatrickLang](https://github.com/PatrickLang)) [SIG Node, Testing and Windows]
- Client-go certificate manager rotation gained the ability to preserve optional intermediate chains accompanying issued certificates ([#88744](https://github.com/kubernetes/kubernetes/pull/88744), [@jackkleeman](https://github.com/jackkleeman)) [SIG API Machinery and Auth]
- Cloud provider config CloudProviderBackoffMode has been removed since it won't be used anymore. ([#88463](https://github.com/kubernetes/kubernetes/pull/88463), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Conformance image now depends on stretch-slim instead of debian-hyperkube-base as that image is being deprecated and removed. ([#88702](https://github.com/kubernetes/kubernetes/pull/88702), [@dims](https://github.com/dims)) [SIG Cluster Lifecycle, Release and Testing]
- Deprecate --generator flag from kubectl create commands ([#88655](https://github.com/kubernetes/kubernetes/pull/88655), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- Deprecate kubectl top flags related to heapster
  Drop support of heapster in kubectl top ([#87498](https://github.com/kubernetes/kubernetes/pull/87498), [@serathius](https://github.com/serathius)) [SIG CLI]
- EndpointSlice should not contain endpoints for terminating pods ([#89056](https://github.com/kubernetes/kubernetes/pull/89056), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
- Evictions due to pods breaching their ephemeral storage limits are now recorded by the `kubelet_evictions` metric and can be alerted on. ([#87906](https://github.com/kubernetes/kubernetes/pull/87906), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- FIX: prevent apiserver from panicking when failing to load audit webhook config file ([#88879](https://github.com/kubernetes/kubernetes/pull/88879), [@JoshVanL](https://github.com/JoshVanL)) [SIG API Machinery and Auth]
- Fix /readyz to return error immediately after a shutdown is initiated, before the --shutdown-delay-duration has elapsed. ([#88911](https://github.com/kubernetes/kubernetes/pull/88911), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Fix a bug that didn't allow to use IPv6 addresses with leading zeros ([#89341](https://github.com/kubernetes/kubernetes/pull/89341), [@aojea](https://github.com/aojea)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Fix a bug where ExternalTrafficPolicy is not applied to service ExternalIPs. ([#88786](https://github.com/kubernetes/kubernetes/pull/88786), [@freehan](https://github.com/freehan)) [SIG Network]
- Fix a bug where kubenet fails to parse the tc output. ([#83572](https://github.com/kubernetes/kubernetes/pull/83572), [@chendotjs](https://github.com/chendotjs)) [SIG Network]
- Fix bug with xfs_repair from stopping xfs mount ([#89444](https://github.com/kubernetes/kubernetes/pull/89444), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Fix describe ingress annotations not sorted. ([#88394](https://github.com/kubernetes/kubernetes/pull/88394), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix detection of SystemOOMs in which the victim is a container. ([#88871](https://github.com/kubernetes/kubernetes/pull/88871), [@dashpole](https://github.com/dashpole)) [SIG Node]
- Fix handling of aws-load-balancer-security-groups annotation. Security-Groups assigned with this annotation are no longer modified by kubernetes which is the expected behaviour of most users. Also no unnecessary Security-Groups are created anymore if this annotation is used. ([#83446](https://github.com/kubernetes/kubernetes/pull/83446), [@Elias481](https://github.com/Elias481)) [SIG Cloud Provider]
- Fix invalid VMSS updates due to incorrect cache ([#89002](https://github.com/kubernetes/kubernetes/pull/89002), [@ArchangelSDY](https://github.com/ArchangelSDY)) [SIG Cloud Provider]
- Fix isCurrentInstance for Windows by removing the dependency of hostname. ([#89138](https://github.com/kubernetes/kubernetes/pull/89138), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix kube-apiserver startup to wait for APIServices to be installed into the HTTP handler before reporting readiness. ([#89147](https://github.com/kubernetes/kubernetes/pull/89147), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Fix kubectl create deployment image name ([#86636](https://github.com/kubernetes/kubernetes/pull/86636), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Fix missing "apiVersion" for "involvedObject" in Events for Nodes. ([#87537](https://github.com/kubernetes/kubernetes/pull/87537), [@uthark](https://github.com/uthark)) [SIG Apps and Node]
- Fix that prevents repeated fetching of PVC/PV objects by kubelet when processing of pod volumes fails. While this prevents hammering API server in these error scenarios, it means that some errors in processing volume(s) for a pod could now take up to 2-3 minutes before retry. ([#88141](https://github.com/kubernetes/kubernetes/pull/88141), [@tedyu](https://github.com/tedyu)) [SIG Node and Storage]
- Fix the VMSS name and resource group name when updating Azure VMSS for LoadBalancer backendPools ([#89337](https://github.com/kubernetes/kubernetes/pull/89337), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix: add remediation in azure disk attach/detach ([#88444](https://github.com/kubernetes/kubernetes/pull/88444), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: azure file mount timeout issue ([#88610](https://github.com/kubernetes/kubernetes/pull/88610), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: check disk status before delete azure disk ([#88360](https://github.com/kubernetes/kubernetes/pull/88360), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: corrupted mount point in csi driver ([#88569](https://github.com/kubernetes/kubernetes/pull/88569), [@andyzhangx](https://github.com/andyzhangx)) [SIG Storage]
- Fixed a bug in the TopologyManager. Previously, the TopologyManager would only guarantee alignment if container creation was serialized in some way. Alignment is now guaranteed under all scenarios of container creation. ([#87759](https://github.com/kubernetes/kubernetes/pull/87759), [@klueska](https://github.com/klueska)) [SIG Node]
- Fixed a data race in kubelet image manager that can cause static pod workers to silently stop working. ([#88915](https://github.com/kubernetes/kubernetes/pull/88915), [@roycaihw](https://github.com/roycaihw)) [SIG Node]
- Fixed an issue that could cause the kubelet to incorrectly run concurrent pod reconciliation loops and crash. ([#89055](https://github.com/kubernetes/kubernetes/pull/89055), [@tedyu](https://github.com/tedyu)) [SIG Node]
- Fixed block CSI volume cleanup after timeouts. ([#88660](https://github.com/kubernetes/kubernetes/pull/88660), [@jsafrane](https://github.com/jsafrane)) [SIG Node and Storage]
- Fixed bug where a nonzero exit code was returned when initializing zsh completion even though zsh completion was successfully initialized ([#88165](https://github.com/kubernetes/kubernetes/pull/88165), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Fixed cleaning of CSI raw block volumes. ([#87978](https://github.com/kubernetes/kubernetes/pull/87978), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixes conversion error in multi-version custom resources that could cause metadata.generation to increment on no-op patches or updates of a custom resource. ([#88995](https://github.com/kubernetes/kubernetes/pull/88995), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Fixes issue where you can't attach more than 15 GCE Persistent Disks to c2, n2, m1, m2 machine types. ([#88602](https://github.com/kubernetes/kubernetes/pull/88602), [@yuga711](https://github.com/yuga711)) [SIG Storage]
- Fixes v1.18.0-rc.1 regression in `kubectl port-forward` when specifying a local and remote port ([#89401](https://github.com/kubernetes/kubernetes/pull/89401), [@liggitt](https://github.com/liggitt)) [SIG CLI]
- For volumes that allow attaches across multiple nodes, attach and detach operations across different nodes are now executed in parallel. ([#88678](https://github.com/kubernetes/kubernetes/pull/88678), [@verult](https://github.com/verult)) [SIG Apps, Node and Storage]
- Get-kube.sh uses the gcloud's current local GCP service account for auth when the provider is GCE or GKE instead of the metadata server default ([#88383](https://github.com/kubernetes/kubernetes/pull/88383), [@BenTheElder](https://github.com/BenTheElder)) [SIG Cluster Lifecycle]
- Golang/x/net has been updated to bring in fixes for CVE-2020-9283 ([#88381](https://github.com/kubernetes/kubernetes/pull/88381), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Hide kubectl.kubernetes.io/last-applied-configuration in describe command ([#88758](https://github.com/kubernetes/kubernetes/pull/88758), [@soltysh](https://github.com/soltysh)) [SIG Auth and CLI]
- In GKE alpha clusters it will be possible to use the service annotation `cloud.google.com/network-tier: Standard` ([#88487](https://github.com/kubernetes/kubernetes/pull/88487), [@zioproto](https://github.com/zioproto)) [SIG Cloud Provider]
- Ipvs: only attempt setting of sysctlconnreuse on supported kernels ([#88541](https://github.com/kubernetes/kubernetes/pull/88541), [@cmluciano](https://github.com/cmluciano)) [SIG Network]
- Kube-proxy: on dual-stack mode, if it is not able to get the IP Family of an endpoint, logs it with level InfoV(4) instead of Warning, avoiding flooding the logs for endpoints without addresses ([#88934](https://github.com/kubernetes/kubernetes/pull/88934), [@aojea](https://github.com/aojea)) [SIG Network]
- Kubeadm now includes CoreDNS version 1.6.7 ([#86260](https://github.com/kubernetes/kubernetes/pull/86260), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that 'kubeadm upgrade' hangs in single node cluster ([#88434](https://github.com/kubernetes/kubernetes/pull/88434), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet: fix the bug that kubelet help information can not show the right type of flags ([#88515](https://github.com/kubernetes/kubernetes/pull/88515), [@SataQiu](https://github.com/SataQiu)) [SIG Docs and Node]
- Kubelets perform fewer unnecessary pod status update operations on the API server. ([#88591](https://github.com/kubernetes/kubernetes/pull/88591), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Scalability]
- Optimize kubectl version help info ([#88313](https://github.com/kubernetes/kubernetes/pull/88313), [@zhouya0](https://github.com/zhouya0)) [SIG CLI]
- Plugin/PluginConfig and Policy APIs are mutually exclusive when running the scheduler ([#88864](https://github.com/kubernetes/kubernetes/pull/88864), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Removes the deprecated command `kubectl rolling-update` ([#88057](https://github.com/kubernetes/kubernetes/pull/88057), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG Architecture, CLI and Testing]
- Resolved a regression in v1.18.0-rc.1 mounting windows volumes ([#89319](https://github.com/kubernetes/kubernetes/pull/89319), [@mboersma](https://github.com/mboersma)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Scheduler PreScore plugins are not executed if there is one filtered node or less. ([#89370](https://github.com/kubernetes/kubernetes/pull/89370), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling]
- Specifying PluginConfig for the same plugin more than once fails scheduler startup.
  
  Specifying extenders and configuring .ignoredResources for the NodeResourcesFit plugin fails ([#88870](https://github.com/kubernetes/kubernetes/pull/88870), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Support TLS Server Name overrides in kubeconfig file and via --tls-server-name in kubectl ([#88769](https://github.com/kubernetes/kubernetes/pull/88769), [@deads2k](https://github.com/deads2k)) [SIG API Machinery, Auth and CLI]
- Terminating a restartPolicy=Never pod no longer has a chance to report the pod succeeded when it actually failed. ([#88440](https://github.com/kubernetes/kubernetes/pull/88440), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- The EventRecorder from k8s.io/client-go/tools/events will now create events in the default namespace (instead of kube-system) when the related object does not have it set. ([#88815](https://github.com/kubernetes/kubernetes/pull/88815), [@enj](https://github.com/enj)) [SIG API Machinery]
- The audit event sourceIPs list will now always end with the IP that sent the request directly to the API server. ([#87167](https://github.com/kubernetes/kubernetes/pull/87167), [@tallclair](https://github.com/tallclair)) [SIG API Machinery and Auth]
- Update Cluster Autoscaler to 1.18.0; changelog: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.18.0 ([#89095](https://github.com/kubernetes/kubernetes/pull/89095), [@losipiuk](https://github.com/losipiuk)) [SIG Autoscaling and Cluster Lifecycle]
- Update to use golang 1.13.8 ([#87648](https://github.com/kubernetes/kubernetes/pull/87648), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Release and Testing]
- Validate kube-proxy flags --ipvs-tcp-timeout, --ipvs-tcpfin-timeout, --ipvs-udp-timeout ([#88657](https://github.com/kubernetes/kubernetes/pull/88657), [@chendotjs](https://github.com/chendotjs)) [SIG Network]
- Wait for all CRDs to show up in discovery endpoint before reporting readiness. ([#89145](https://github.com/kubernetes/kubernetes/pull/89145), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- `kubectl config view` now redacts bearer tokens by default, similar to client certificates. The `--raw` flag can still be used to output full content. ([#88985](https://github.com/kubernetes/kubernetes/pull/88985), [@brianpursley](https://github.com/brianpursley)) [SIG API Machinery and CLI]