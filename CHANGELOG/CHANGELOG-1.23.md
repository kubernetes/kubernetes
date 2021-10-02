<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.23.0-alpha.3](#v1230-alpha3)
  - [Downloads for v1.23.0-alpha.3](#downloads-for-v1230-alpha3)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.23.0-alpha.2](#changelog-since-v1230-alpha2)
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
- [v1.23.0-alpha.2](#v1230-alpha2)
  - [Downloads for v1.23.0-alpha.2](#downloads-for-v1230-alpha2)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.23.0-alpha.1](#changelog-since-v1230-alpha1)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Documentation](#documentation)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.23.0-alpha.1](#v1230-alpha1)
  - [Downloads for v1.23.0-alpha.1](#downloads-for-v1230-alpha1)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.22.0](#changelog-since-v1220)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-2)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Documentation](#documentation-1)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)

<!-- END MUNGE: GENERATED_TOC -->

# v1.23.0-alpha.3


## Downloads for v1.23.0-alpha.3

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes.tar.gz) | 083e6ca03c9d701768b1b5666f354223a3f7dca9fc6410ce45bbf5947152620e300b46df9b6019134e7d736ba44916537eb3bea8fa57e5f7bc3cc34898b4a5dd
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-src.tar.gz) | c3fc74d52e1b7e808c03b9caa30e3e73be30eb8330ce676000b93d5324bbdba93bd005d125b999ba937b79d4751af99b37986911365416f7175d223345f95914

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | 31d8adc657afbd305df18bfec397a825536357e23b241a19aa538b6ddefefc59743f737db98756e04deea89cc6f260d40a80f02b4d1dc34af1d19e8d796dcd8a
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | b69c4d6cde1c476bafa2ca9916ce3e5bf7286be0ff6a08193bdd1a954ba89b64b1b14193d1acec17ccc141024ee3097971448017b5c9f1327e0961b1e92b2224
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 059f25ee48aa4b0d1621d6ba87af8fb7e765634d723d98a4e9739f50d3703e7dd3973f4d1ed886c0f3ad6eba165ed81d4e63ecde3b39e66fcbec7d3aa2dfed2e
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | 291dba14160803065895799adcde39bdad7a5b0372403f283d6d5e9a094fe1fc79c70e7546f93ee692b9fd297e2667cb558e4209161ecb4bf89965df5746ed4d
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | 988e12cd7466033578acc487447df376c409e4f79726a4721af1aedbe931e927b22a93d6224891b61b55c7a0ec12e42d8cfcd40e15a9a0cbbc1dbf0e59ab0341
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | b3f21dac41b38e671fa7a95892468e2c27fab51abf9c77b336550e5ec213af204e16cac11dd76262fedb0087cf5ad1950af7e36599a38d50cc270cf831cd4f0b
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | beebf01e2e4ff09bb711284bb9a5c7cc519e4ac8a826dc829394fa28bd9a3149ba73088eaf6712d39a8cab96b0a1c2859e9d5955fee892b759eaddcdeaa8b93c
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 87e5d3d8ba01f9fefb2300e9f06146a254d39d72eaa10cad8c444428b738b3763483ee9eb82f0a13d2ff5aba35fdcb4320598fd5a6a2a07ea3fd00b4ac682d3c
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 71bfc5a1df9c47735476af10225830212f68c83357ff7d443e18f9b7881524db910781a95d11ff6697cb587352059b5841f7b24fda40b5302ad252bfb6da7e51
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 078b0c698f9535f3eee41ecf162d57e2ace67243da36067b78b30cfbb7b27cfcf97af4c5db48cdd592953e26b42b31794002eb96317476849e89e2126c6df99d

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 951b790158dadf46c32e1a1e9c12f2cc8f41e1645602ebff6b4130a08a377bc6d92549186b420332d620d67191123d98a5d717ac0f5ee9643bebe88947ead8fa
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | 0e7a5b9f39b4f45c45bdb5a19dd3695d28f53e1039d76bc572421c707917944d28b1dbfc36e59214b5bc2b93a787900d8e6eb0b587aa801ea8a8faacdb814a4e
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 921e060120b8651a0f80977360faca9f207189cee10bc61f669ceba4e540ef48c0ceff1a877ee4c7d31b01b88096bce93c577f68f93b2341c8542dfd89972b60
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | 292cde446b754a87f4ef5384fadbd30017e53ed2744d45a724be467c86ccd9837bfb490db6396642a869937f2f0d080d9655e89ca3345f8365d109a9bcdd18d9
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | e0ea667f828ce3b36ca4b2a05fb286da5eb321852c50caf0957694553caf2908b27bcc37a5a82277a2606cf6ff4d9e33617ad61628845d9c21f5cf68c960ca92

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | e13cd3f75628d354bd1544a5495600fb905741431eb4af4da3d980cc0b7565e3f9c1585d9686cc4e967e54fb854f05bbedfe0c60bb7b855fa027ac8ac45b26e0
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | 6c91b42350528692ff558b667bffd41c5b967c7aa6101471274e4b16b0ac6f84afe01722881328fd4f6f8fe71c7852620fa000186c6f7e56e498fcc2c67ad793
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | 81728e1388e9cdb436d6847c868f28ab2771331e5e40cd5a7af13cb8dc80a7e4e66a215c12f8183b4884807a3962f913ef5343b889e3c4ecd0e410e8d53aaea9
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 299649f1b25cc38f3a7543ef4d3ee6d42c85e24ac41b4eb61927bc5c5f0c533a39f9ddd4d5ad1df54c625d77aeb41f6c31b1ca7fd8983262f84fefdf1cb2cfd0
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | fd6cbc93f98abff9803b43215af6e75a4f7b91ca06969220a779468f34b5ec5ec69f20b529e0cd7b10ba8769bbe2507d46f84ce1d8cd0760380ab9264dd94672
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | a5bfaf2e3ad8d3d2127c3e3e0f131c615a03563253da6bf0e1fd793f6ef71287f341ce1bd0d35eb9a81e0721a5baf03e7c72863b5ed8eb45e8fe70573904ed54

## Changelog since v1.23.0-alpha.2

## Changes by Kind

### Deprecation

- Remove 'master' as a valid EgressSelection type in the EgressSelectorConfiguration API. ([#102242](https://github.com/kubernetes/kubernetes/pull/102242), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery and Cloud Provider]
- Remove VolumeSubpath feature gate ([#105090](https://github.com/kubernetes/kubernetes/pull/105090), [@saad-ali](https://github.com/saad-ali)) [SIG Apps, Node and Storage]
- The deprecated --experimental-bootstrap-kubeconfig flag has been removed.
  This can be set via --bootstrap-kubeconfig. ([#103172](https://github.com/kubernetes/kubernetes/pull/103172), [@niulechuan](https://github.com/niulechuan)) [SIG Node]

### API Change

- Client-go impersonation config can specify a UID to pass impersonated uid information through in requests. ([#104483](https://github.com/kubernetes/kubernetes/pull/104483), [@margocrawf](https://github.com/margocrawf)) [SIG API Machinery, Auth and Testing]
- IPv6DualStack feature moved to stable.
  Controller Manager flags for the node IPAM controller have slightly changed:
  1. When configuring a dual-stack cluster, the user must specify both --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 to set the per-node IP mask sizes, instead of the previous --node-cidr-mask-size flag.
  2. The --node-cidr-mask-size flag is mutually exclusive with --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6.
  3. Single-stack clusters do not need to change, but may choose to use the more specific flags.  Users can use either the older --node-cidr-mask-size flag or one of the newer --node-cidr-mask-size-ipv4 or --node-cidr-mask-size-ipv6 flags to configure the per-node IP mask size, provided that the flag's IP family matches the cluster's IP family (--cluster-cidr). ([#104691](https://github.com/kubernetes/kubernetes/pull/104691), [@khenidak](https://github.com/khenidak)) [SIG API Machinery, Apps, Auth, Cloud Provider, Cluster Lifecycle, Network, Node and Testing]
- Kubelet: turn the KubeletConfiguration v1beta1 `ResolverConfig` field from a `string` to `*string`. ([#104624](https://github.com/kubernetes/kubernetes/pull/104624), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle and Node]

### Feature

- Add mechanism to load simple sniffer class into fluentd-elasticsearch image ([#92853](https://github.com/kubernetes/kubernetes/pull/92853), [@cosmo0920](https://github.com/cosmo0920)) [SIG Cloud Provider and Instrumentation]
- Kubeadm: do not check if the '/etc/kubernetes/manifests' folder is empty on joining worker nodes during preflight ([#104942](https://github.com/kubernetes/kubernetes/pull/104942), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- The kube-apiserver's Prometheus metrics have been extended with some that describe the costs of handling LIST requests.  They are as follows.
  - *apiserver_cache_list_total*: Counter of LIST requests served from watch cache, broken down by resource_prefix and index_name
  - *apiserver_cache_list_fetched_objects_total*: Counter of objects read from watch cache in the course of serving a LIST request, broken down by resource_prefix and index_name
  - *apiserver_cache_list_evaluated_objects_total*: Counter of objects tested in the course of serving a LIST request from watch cache, broken down by resource_prefix
  - *apiserver_cache_list_returned_objects_total*: Counter of objects returned for a LIST request from watch cache, broken down by resource_prefix
  - *apiserver_storage_list_total*: Counter of LIST requests served from etcd, broken down by resource
  - *apiserver_storage_list_fetched_objects_total*: Counter of objects read from etcd in the course of serving a LIST request, broken down by resource
  - *apiserver_storage_list_evaluated_objects_total*: Counter of objects tested in the course of serving a LIST request from etcd, broken down by resource
  - *apiserver_storage_list_returned_objects_total*: Counter of objects returned for a LIST request from etcd, broken down by resource ([#104983](https://github.com/kubernetes/kubernetes/pull/104983), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery and Instrumentation]
- Turn on CSIMigrationAzureDisk by default on 1.23 ([#104670](https://github.com/kubernetes/kubernetes/pull/104670), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]

### Bug or Regression

- Changes behaviour of kube-proxy start; does not attempt to set specific sysctl values (which does not work in recent Kernel versions anymore in non-init namespaces), when the current sysctl values are already set higher. ([#103174](https://github.com/kubernetes/kubernetes/pull/103174), [@Napsty](https://github.com/Napsty)) [SIG Network]
- Fix job controller syncs: In case of conflicts, ensure that the sync happens with the most up to date information. Improves reliability of JobTrackingWithFinalizers. ([#105214](https://github.com/kubernetes/kubernetes/pull/105214), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps]
- Fix system default topology spreading when nodes don't have zone labels. Pods correctly spread by default now. ([#105046](https://github.com/kubernetes/kubernetes/pull/105046), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Headless Services with no selector which were created without dual-stack enabled will be defaulted to RequireDualStack instead of PreferDualStack.  This is consistent with such Services which are created with dual-stack enabled. ([#104986](https://github.com/kubernetes/kubernetes/pull/104986), [@thockin](https://github.com/thockin)) [SIG Network]
- Kube-apiserver: events created via the `events.k8s.io` API group for cluster-scoped objects are now permitted in the default namespace as well for compatibility with events clients and the `v1` API ([#100125](https://github.com/kubernetes/kubernetes/pull/100125), [@h4ghhh](https://github.com/h4ghhh)) [SIG API Machinery, Apps and Testing]
- Kube-controller incorrectly enabled support for generic ephemeral inline volumes if the storage object in use protection feature was enabled. ([#104913](https://github.com/kubernetes/kubernetes/pull/104913), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- Kubeadm: switch the preflight check (called 'Swap') that verifies if swap is enabled on Linux hosts to report a warning instead of an error. This is related to the graduation of the NodeSwap feature gate in the kubelet to Beta and being enabled by default in 1.23 - allows swap support on Linux hosts. In the next release of kubeadm (1.24) the preflight check will be removed, thus we recommend that you stop using it - e.g. via --ignore-preflight-errors or the kubeadm config. ([#104854](https://github.com/kubernetes/kubernetes/pull/104854), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Makes the etcd client (used by the API server) retry certain types of errors. The full list of retriable (codes.Unavailable) errors can be found at https://github.com/etcd-io/etcd/blob/main/api/v3rpc/rpctypes/error.go#L72 ([#105069](https://github.com/kubernetes/kubernetes/pull/105069), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- When a static pod file is deleted and recreated while using a fixed UID, the pod was not properly restarted. ([#104847](https://github.com/kubernetes/kubernetes/pull/104847), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- XFS-filesystems are now force-formatted (option `-f`) in order to avoid problems being formatted due to detection of magic super-blocks. This aligns with the behaviour of formatting of ext3/4 filesystems. ([#104923](https://github.com/kubernetes/kubernetes/pull/104923), [@davidkarlsen](https://github.com/davidkarlsen)) [SIG Storage]

### Other (Cleanup or Flake)

- Enhanced error message for nodes not selected by scheduler due to pod's PersistentVolumeClaim(s) bound to PersistentVolume(s) that do not exist. ([#105196](https://github.com/kubernetes/kubernetes/pull/105196), [@yibozhuang](https://github.com/yibozhuang)) [SIG Scheduling and Storage]
- Kubeadm: remove the --port flag from the manifest for the kube-scheduler since the flag has been a NO-OP since 1.23 and insecure serving was removed for the component. ([#105034](https://github.com/kubernetes/kubernetes/pull/105034), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Migrate `cmd/proxy/{config, healthcheck, winkernel}` to structured logging ([#104944](https://github.com/kubernetes/kubernetes/pull/104944), [@jyz0309](https://github.com/jyz0309)) [SIG Network]
- Migrate cmd/proxy/app and pkg/proxy/meta_proxier to structured logging ([#104928](https://github.com/kubernetes/kubernetes/pull/104928), [@jyz0309](https://github.com/jyz0309)) [SIG Apps, Cluster Lifecycle, Network, Node and Testing]
- Migrate pkg/proxy to structured logs ([#104908](https://github.com/kubernetes/kubernetes/pull/104908), [@CIPHERTron](https://github.com/CIPHERTron)) [SIG Network]
- Migrated pkg/proxy/winuserspace to structured logging ([#105035](https://github.com/kubernetes/kubernetes/pull/105035), [@shivanshu1333](https://github.com/shivanshu1333)) [SIG Network]
- The `BoundServiceAccountTokenVolume` feature gate that is GA since v1.22 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104167](https://github.com/kubernetes/kubernetes/pull/104167), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Auth]
- The `SupportPodPidsLimit` and  `SupportNodePidsLimit` feature gates that are GA since v1.20 are unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104163](https://github.com/kubernetes/kubernetes/pull/104163), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Node]
- Update build images to Debian 11 (Bullseye)
  - debian-base:bullseye-v1.0.0
  - debian-iptables:bullseye-v1.0.0
  - go-runner:v2.3.1-go1.17.1-bullseye.0
  - kube-cross:v1.23.0-go1.17.1-bullseye.1
  - setcap:bullseye-v1.0.0
  - cluster/images/etcd: Build 3.5.0-2 image
  - test/conformance/image: Update runner image to base-debian11 ([#105158](https://github.com/kubernetes/kubernetes/pull/105158), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Architecture, Release and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/json-iterator/go: [v1.1.11 → v1.1.12](https://github.com/json-iterator/go/compare/v1.1.11...v1.1.12)
- github.com/modern-go/reflect2: [v1.0.1 → v1.0.2](https://github.com/modern-go/reflect2/compare/v1.0.1...v1.0.2)

### Removed
_Nothing has changed._



# v1.23.0-alpha.2


## Downloads for v1.23.0-alpha.2

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes.tar.gz) | 121d51f42a52b28e27a4b2f914a4f80fa3fba6328e6a4a5c96dec39c5b28c05461fcc290ef35a49058e237091532b24db3cd8c61801bcb6736aee1dd7dbcffc3
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-src.tar.gz) | 641d47241acfadb3b13bccec57795749d2c9e3e07ffa7aa4b30df3a488643631eb8e5cd581bcfb764dff4ac5ed755f72d94e80746142123b09e1675e81421a91

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | f734cb514ee56adcb2d991a6f0550df907c72f8a61cc2a13117e61b8d5826ff942a582a2e9383deb1a61d5df2243362f1327942a3b4883490eb3296647ce3737
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 24d1f851cd5782f8f39054e37beda1554dadd8a28cb3272b00d50fc095d1fc3018768c1ea72a44eda61ff0f58f71b33dd28cbdc54467d620e87c3694ecf14cc2
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 082ad4abea58de3b629fc2ed4560a836cdbeb1adefb0c4cf47044bf33c750d8fcd8a06e2c4ce365853e83a58d52e0129d510a698dd894bd1261f8184dd1cab42
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | b3b0b23479c05b57ca574cf17cdcde7e716033bc4f6a80532d1175d8e533e3202bece0dcf503731d5a60319c526ce1ce4a0bc900bf87536321208a59cf890e35
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | f5dac2976ce04310f74bba6102080554309b851fbd966ff1220d3eb23089db8eb8da519a6bd8865c94f2f24346a4d27eb40fd0a3ff06ca9c6874e1fc6f356b67
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 057b372150749b13a38e04802c7cf566765e0fbb27f1b5f7bf6d3cc3f71eb3020916ea7f8579ecc7fcc10e2db1b5c8caa31a1e8a3aac80da86e4e777f515d42f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 9a090d22aeba011c6d039bff59dbdc23ac4a112828db3cbba588d8b0ee1cd14d16e0eacefbb000e5a3ff26bcce4730824819f86a99b7a9826f35fa9964f9f27a
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 435e20055badb619289dc7c572af300bd2f86068d0b8f326e8d9abfda5347f2449e316158c412e9b946a2541208c3e8cc6e5c823946e74ac4fc2d594d410179a
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 55f192a4d095d494bb53af1b7133124b762a677eb46247b9dba71d10ea6830b37c30d603908e7a9c63f371baff508b19406e89b231ed5ece0497627f09753f68
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 944059d1f1918a793490b95be8130d06189508ba8e79e79ca8cfd2ab98bf396ac551786514b093cc6afe4b3fd15736d728cfcdce18bb32fbee41bc0a97f5c4be

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | a76a4b86ee151ba027f7cf4a2072451ae4c829182bb14e00ce1967421744bfc1e58f141b6eaf2ab27ece67054ae307f8e0768477ab9c3c4749eaad397d495182
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | 95aeb4eb473ab4920d81904bc89c6126732b9c6888f9e57493ee99d692042ca44f6844ac1dade1409565f4d9fbec59445402e1f7deac6cbf5b6df16ac814b58c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 3c56e906aafc2a1ac72300352a334662bec5d59e3e523c19b9d65bc52ad9075dc2631f259513efd0f654e220fe0e7d54dfa5028d7eaad81d5d87ca251653f75d
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | b74bacafe9bb6a7cf407747b03e78ae3873e50deec4eaa08758d5e1d5287ac23af59b3ef26f888fe4cd44ccb1455beafcd1384e700230eb445720e3acae5f2e3
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | d3f8f8d9c233b114129f615252d42782cd366978a49506393a40af3f8b5b1250ce99e9806881675e112a69270a0411fb2f00ea19b99ad7415b9e0074beb2726d

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 146e2f762c179178a57a8c7af7c26470c5d580b8ff8400615162ad1056625f87ce2b32598538d82652f88639e54afb782810529b074c36eb52cc6374414a6181
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 9357d1b387e1b049fb6cec06a7081afc2ce7e906484c9b061fb0449d147a6c4f9c9dc7a9219cdca5ed71df6c73784f360018d9e48d4fa2aa7eeabef60649d7a4
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 8394f8f9d6ee823cb9a470ea67e15d4d0c6aca7065fe826788f50955905373fc3cdddd6db43901c07736588d8d6a3d3e2916bc8d45fd6bd06307583686137a0a
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 7211cb426834484bff39f1ab3c9541203429039f8f5e522ca9e28c43da749e197128a3cae28db0467fc339305d2f23f85e8b4ed9ec116506c3d8076744a88d5e
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | a7c1a38250398171d3df5865749e9928867c4f44106ae66d44cf9f948ce4f4eed9d1f273a5d369996425b1e12482fceccde4c7652770a8c9fb3f161811323b69
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 2007b3b16597cc06b486f87f35b6c637404f07c11d88b8c8e1c2c9bbea97f762bd7d4f9a31f42f78a917c595af5cb89e6885dd88f3766836dc6e4ec79cf084f2

## Changelog since v1.23.0-alpha.1

## Changes by Kind

### Deprecation

- Controller-manager: the following flags have no effect and would be removed in v1.24:
  - `--port`
  - `--address`
  The insecure port flags `--port` may only be set to 0 now.
  Also `metricsBindAddress` and `healthzBindAddress` fields from `kubescheduler.config.k8s.io/v1beta1` are no-op and expected to be empty. Removed in `kubescheduler.config.k8s.io/v1beta2` completely.
  
  In addition, please be careful that:
  - kube-scheduler MUST start with `--authorization-kubeconfig` and `--authentication-kubeconfig` correctly set to get authentication/authorization working.
  - liveness/readiness probes to kube-scheduler MUST use HTTPS now, and the default port has been changed to 10259.
  - Applications that fetch metrics from kube-scheduler should use a dedicated service account which is allowed to access nonResourceURLs `/metrics`. ([#96345](https://github.com/kubernetes/kubernetes/pull/96345), [@ingvagabund](https://github.com/ingvagabund)) [SIG Cloud Provider, Scheduling and Testing]
- Removed deprecated metric `scheduler_volume_scheduling_duration_seconds` ([#104518](https://github.com/kubernetes/kubernetes/pull/104518), [@dntosas](https://github.com/dntosas)) [SIG Instrumentation, Scheduling and Storage]

### API Change

- A small regression in Service updates was fixed.  The circumstances are so unlikely that probably nobody would ever hit it. ([#104601](https://github.com/kubernetes/kubernetes/pull/104601), [@thockin](https://github.com/thockin)) [SIG Network]
- Introduce v1beta2 for Priority and Fairness with no changes in API spec ([#104399](https://github.com/kubernetes/kubernetes/pull/104399), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Testing]
- Kube-apiserver: Fixes handling of CRD schemas containing literal null values in enums. ([#104969](https://github.com/kubernetes/kubernetes/pull/104969), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps and Network]
- Kubelet: turn the KubeletConfiguration v1beta1 `ResolverConfig` field from a `string` to `*string`. ([#104624](https://github.com/kubernetes/kubernetes/pull/104624), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle and Node]
- Kubernetes is now built using go1.17 ([#103692](https://github.com/kubernetes/kubernetes/pull/103692), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Removed deprecated `--seccomp-profile-root`/`seccompProfileRoot` config ([#103941](https://github.com/kubernetes/kubernetes/pull/103941), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Since golang 1.17 both net.ParseIP and net.ParseCIDR rejects leading zeros in the dot-decimal notation of IPv4 addresses.
  Kubernetes will keep allowing leading zeros on IPv4 address to not break the compatibility.
  IMPORTANT: Kubernetes interprets leading zeros on IPv4 addresses as decimal, users must not rely on parser alignment to not being impacted by the associated security advisory:
  CVE-2021-29923 golang standard library "net" - Improper Input Validation of octal literals in golang 1.16.2 and below standard library "net" results in indeterminate SSRF & RFI vulnerabilities.
  Reference: https://nvd.nist.gov/vuln/detail/CVE-2021-29923 ([#104368](https://github.com/kubernetes/kubernetes/pull/104368), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage and Testing]
- StatefulSet minReadySeconds is promoted to beta ([#104045](https://github.com/kubernetes/kubernetes/pull/104045), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps and Testing]
- The `Service.spec.ipFamilyPolicy` field is now *required* in order to create or update a Service as dual-stack.  This is a breaking change from the beta behavior.  Previously the server would try to infer the value of that field from either `ipFamilies` or `clusterIPs`, but that caused ambiguity on updates.  Users who want a dual-stack Service MUST specify `ipFamilyPolicy` as either "PreferDualStack" or "RequireDualStack". ([#96684](https://github.com/kubernetes/kubernetes/pull/96684), [@thockin](https://github.com/thockin)) [SIG API Machinery, Apps, Network and Testing]
- Users of LogFormatRegistry in component-base must update their code to use the logr v1.0.0 API. The JSON log output now uses the format from go-logr/zapr (no `v` field for error messages, additional information for invalid calls) and has some fixes (correct source code location for warnings about invalid log calls). ([#104103](https://github.com/kubernetes/kubernetes/pull/104103), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- When creating an object with generateName, if a conflict occurs the server now returns an AlreadyExists error with a retry option. ([#104699](https://github.com/kubernetes/kubernetes/pull/104699), [@vincepri](https://github.com/vincepri)) [SIG API Machinery]

### Feature

- Add fish shell completion to kubectl ([#92989](https://github.com/kubernetes/kubernetes/pull/92989), [@WLun001](https://github.com/WLun001)) [SIG CLI]
- Added PowerShell completion generation by running `kubectl completion powershell` ([#103758](https://github.com/kubernetes/kubernetes/pull/103758), [@zikhan](https://github.com/zikhan)) [SIG CLI]
- Added a `Processing` condition for the workqueue API
  Changed `Shutdown` for the workqueue API to wait until the work queue finishes processing all in-flight items. ([#101928](https://github.com/kubernetes/kubernetes/pull/101928), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG API Machinery and Apps]
- Added a new flag `--append-server-path` to `kubectl proxy` that will automatically append the kube context server path to each request. ([#97350](https://github.com/kubernetes/kubernetes/pull/97350), [@FabianKramm](https://github.com/FabianKramm)) [SIG API Machinery, CLI and Testing]
- Added support for setting controller-manager log level online ([#104571](https://github.com/kubernetes/kubernetes/pull/104571), [@h4ghhh](https://github.com/h4ghhh)) [SIG API Machinery, Apps and Cloud Provider]
- Adding support for multiple --from-env-file flags ([#104232](https://github.com/kubernetes/kubernetes/pull/104232), [@lauchokyip](https://github.com/lauchokyip)) [SIG CLI]
- Cloud providers can set service account names for cloud controllers. ([#103178](https://github.com/kubernetes/kubernetes/pull/103178), [@nckturner](https://github.com/nckturner)) [SIG API Machinery and Cloud Provider]
- Health check of kube-controller-manager now includes each controller. ([#104667](https://github.com/kubernetes/kubernetes/pull/104667), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery and Cloud Provider]
- Kube-scheduler now logs node and plugin scoring  even though --v<10
  - socres of the top 3 plugins in the top 3 nodes are dumped if --v=4,5
  - socres of all plugins in the top 6 nodes are dumped if --v=6,7,8,9 ([#103515](https://github.com/kubernetes/kubernetes/pull/103515), [@muma378](https://github.com/muma378)) [SIG Scheduling]
- Kubernetes is now built with Golang 1.17.1 ([#104904](https://github.com/kubernetes/kubernetes/pull/104904), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- The pause image list now contains Windows Server 2022 ([#104438](https://github.com/kubernetes/kubernetes/pull/104438), [@nick5616](https://github.com/nick5616)) [SIG Windows]
- Updates  debian-iptables to v1.6.7 to pick up CVE fixes ([#104970](https://github.com/kubernetes/kubernetes/pull/104970), [@PushkarJ](https://github.com/PushkarJ)) [SIG API Machinery, Network, Release, Security and Testing]

### Documentation

- Conformance: the test "[sig-network] EndpointSlice should have Endpoints and EndpointSlices pointing to API Server [Conformance]" only requires that there is an EndpointSlice that references the "kubernetes.default" service, it no longer requires that its named "kubernetes". ([#104664](https://github.com/kubernetes/kubernetes/pull/104664), [@aojea](https://github.com/aojea)) [SIG Architecture, Network and Testing]

### Bug or Regression

- A pod that the Kubelet rejects was still considered as being accepted for a brief period of time after rejection, which might cause some pods to be rejected briefly that could fit on the node.  A pod that is still terminating (but has status indicating it has failed) may also still be consuming resources and so should also be considered. ([#104817](https://github.com/kubernetes/kubernetes/pull/104817), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Changed kubectl describe to compute Age of an event using the count and lastObservedTime fields available in the event series ([#104482](https://github.com/kubernetes/kubernetes/pull/104482), [@harjas27](https://github.com/harjas27)) [SIG CLI]
- Don't prematurely close reflectors in case of slow initialization in watch based manager to fix issues with inability to properly mount secrets/configmaps. ([#104604](https://github.com/kubernetes/kubernetes/pull/104604), [@wojtek-t](https://github.com/wojtek-t)) [SIG Node]
- Fix Job tracking with finalizers for more than 500 pods, ensuring all finalizers are removed before counting the Pod. ([#104666](https://github.com/kubernetes/kubernetes/pull/104666), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Instrumentation]
- Fix a regression where the Kubelet failed to exclude already completed pods from calculations about how many resources it was currently using when deciding whether to allow more pods. ([#104577](https://github.com/kubernetes/kubernetes/pull/104577), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Fix detach disk issue on deleting vmss node ([#104572](https://github.com/kubernetes/kubernetes/pull/104572), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: ensure InstanceShutdownByProviderID return false for creating Azure VMs ([#104382](https://github.com/kubernetes/kubernetes/pull/104382), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix: ignore the case when comparing azure tags in service annotation ([#104705](https://github.com/kubernetes/kubernetes/pull/104705), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fix: ignore the case when updating Azure tags ([#104593](https://github.com/kubernetes/kubernetes/pull/104593), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fixed bug where kubectl would emit duplicate warning messages for flag names that contain an underscore and recommend using a nonexistent flag in some cases ([#103852](https://github.com/kubernetes/kubernetes/pull/103852), [@brianpursley](https://github.com/brianpursley)) [SIG CLI and Cluster Lifecycle]
- Fixed client IP preservation for NodePort service with protocol SCTP in ipvs mode ([#104756](https://github.com/kubernetes/kubernetes/pull/104756), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Fixed occasional pod cgroup freeze when using cgroup v1 and systemd driver. ([#104528](https://github.com/kubernetes/kubernetes/pull/104528), [@kolyshkin](https://github.com/kolyshkin)) [SIG Node]
- Fixes a regression that could cause panics in LRU caches in controller-manager, kubelet, kube-apiserver, or client-go ([#104466](https://github.com/kubernetes/kubernetes/pull/104466), [@stbenjam](https://github.com/stbenjam)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Kube-apiserver: fixes an issue where an admission webhook can observe a v1 Pod object that does not have the `defaultMode` field set in the injected service account token volume ([#104523](https://github.com/kubernetes/kubernetes/pull/104523), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Kube-proxy health check ports used to listen to :<port> for each of the services. This is not needed and opens ports in addresses the cluster user may not have intended. The PR limits listening to all node address which are controlled by `--nodeport-addresses` flag. if no addresses are provided then we default to existing behavior by listening to :<port> for each service ([#104742](https://github.com/kubernetes/kubernetes/pull/104742), [@khenidak](https://github.com/khenidak)) [SIG Network]
- Kube-scheduler now doesn't print any usage message when unknown flag is specified ([#104503](https://github.com/kubernetes/kubernetes/pull/104503), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Metrics changes: Fix exposed buckets of `scheduler_volume_scheduling_duration_seconds_bucket` metric ([#100720](https://github.com/kubernetes/kubernetes/pull/100720), [@dntosas](https://github.com/dntosas)) [SIG Apps, Instrumentation, Scheduling and Storage]
- Scheduler resource metrics over fractional binary quantities (2.5Gi, 1.1Ki) were incorrectly reported as very small values. ([#103751](https://github.com/kubernetes/kubernetes/pull/103751), [@y-tag](https://github.com/y-tag)) [SIG API Machinery and Scheduling]

### Other (Cleanup or Flake)

- Generic ephemeral volumes: better pod events ("waiting for ephemeral volume controller to create the persistentvolumeclaim"" instead of "persistentvolumeclaim not found") ([#104605](https://github.com/kubernetes/kubernetes/pull/104605), [@pohly](https://github.com/pohly)) [SIG Scheduling and Storage]
- Kubeadm: remove the deprecated flags "--csr-only" and "--csr-dir" from "kubeadm certs renew". Please use "kubeadm certs generate-csr" instead. ([#104796](https://github.com/kubernetes/kubernetes/pull/104796), [@RA489](https://github.com/RA489)) [SIG Cluster Lifecycle]
- Migrate `pkg/scheduler` to structured logging ([#99273](https://github.com/kubernetes/kubernetes/pull/99273), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Scheduling]
- Migrated pkg/proxy/userspace to structured logging ([#104931](https://github.com/kubernetes/kubernetes/pull/104931), [@shivanshu1333](https://github.com/shivanshu1333)) [SIG Network]
- More detailed logging has been added to the EndpointSlice controller for Topology Aware Hints. ([#104741](https://github.com/kubernetes/kubernetes/pull/104741), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Support for Windows Server 2022 was added to the k8s.gcr.io/pause:3.6 image. ([#104711](https://github.com/kubernetes/kubernetes/pull/104711), [@claudiubelu](https://github.com/claudiubelu)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Node, Release and Testing]
- The maximum length of the CSINode id field has increased to 256 bytes to match the CSI spec ([#104160](https://github.com/kubernetes/kubernetes/pull/104160), [@pacoxu](https://github.com/pacoxu)) [SIG Storage]
- Update conformance image to use debian-base:buster-v1.9.0 ([#104696](https://github.com/kubernetes/kubernetes/pull/104696), [@PushkarJ](https://github.com/PushkarJ)) [SIG Architecture, Release, Security and Testing]
- `volume.kubernetes.io/storage-provisioner` annotation will be added to dynamic provisioning required PVC. `volume.beta.kubernetes.io/storage-provisioner` annotation is deprecated. ([#104590](https://github.com/kubernetes/kubernetes/pull/104590), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG Apps and Storage]

## Dependencies

### Added
- bazil.org/fuse: 371fbbd
- github.com/go-logr/zapr: [v1.1.0](https://github.com/go-logr/zapr/tree/v1.1.0)
- github.com/kr/fs: [v0.1.0](https://github.com/kr/fs/tree/v0.1.0)
- github.com/pkg/sftp: [v1.10.1](https://github.com/pkg/sftp/tree/v1.10.1)

### Changed
- github.com/Microsoft/go-winio: [v0.4.15 → v0.4.17](https://github.com/Microsoft/go-winio/compare/v0.4.15...v0.4.17)
- github.com/Microsoft/hcsshim: [5eafd15 → v0.8.22](https://github.com/Microsoft/hcsshim/compare/5eafd15...v0.8.22)
- github.com/benbjohnson/clock: [v1.0.3 → v1.1.0](https://github.com/benbjohnson/clock/compare/v1.0.3...v1.1.0)
- github.com/bketelsen/crypt: [5cbc8cc → v0.0.4](https://github.com/bketelsen/crypt/compare/5cbc8cc...v0.0.4)
- github.com/containerd/cgroups: [0dbf7f0 → v1.0.1](https://github.com/containerd/cgroups/compare/0dbf7f0...v1.0.1)
- github.com/containerd/containerd: [v1.4.4 → v1.4.9](https://github.com/containerd/containerd/compare/v1.4.4...v1.4.9)
- github.com/containerd/continuity: [aaeac12 → v0.1.0](https://github.com/containerd/continuity/compare/aaeac12...v0.1.0)
- github.com/containerd/fifo: [a9fb20d → v1.0.0](https://github.com/containerd/fifo/compare/a9fb20d...v1.0.0)
- github.com/containerd/go-runc: [5a6d9f3 → v1.0.0](https://github.com/containerd/go-runc/compare/5a6d9f3...v1.0.0)
- github.com/containerd/typeurl: [v1.0.1 → v1.0.2](https://github.com/containerd/typeurl/compare/v1.0.1...v1.0.2)
- github.com/go-logr/logr: [v0.4.0 → v1.1.0](https://github.com/go-logr/logr/compare/v0.4.0...v1.1.0)
- github.com/magiconair/properties: [v1.8.1 → v1.8.5](https://github.com/magiconair/properties/compare/v1.8.1...v1.8.5)
- github.com/mitchellh/go-homedir: [v1.1.0 → v1.0.0](https://github.com/mitchellh/go-homedir/compare/v1.1.0...v1.0.0)
- github.com/mitchellh/mapstructure: [v1.1.2 → v1.4.1](https://github.com/mitchellh/mapstructure/compare/v1.1.2...v1.4.1)
- github.com/opencontainers/runc: [v1.0.1 → v1.0.2](https://github.com/opencontainers/runc/compare/v1.0.1...v1.0.2)
- github.com/pelletier/go-toml: [v1.2.0 → v1.9.3](https://github.com/pelletier/go-toml/compare/v1.2.0...v1.9.3)
- github.com/spf13/afero: [v1.2.2 → v1.6.0](https://github.com/spf13/afero/compare/v1.2.2...v1.6.0)
- github.com/spf13/cast: [v1.3.0 → v1.3.1](https://github.com/spf13/cast/compare/v1.3.0...v1.3.1)
- github.com/spf13/cobra: [v1.1.3 → v1.2.1](https://github.com/spf13/cobra/compare/v1.1.3...v1.2.1)
- github.com/spf13/jwalterweatherman: [v1.0.0 → v1.1.0](https://github.com/spf13/jwalterweatherman/compare/v1.0.0...v1.1.0)
- github.com/spf13/viper: [v1.7.0 → v1.8.1](https://github.com/spf13/viper/compare/v1.7.0...v1.8.1)
- github.com/yuin/goldmark: [v1.3.5 → v1.4.0](https://github.com/yuin/goldmark/compare/v1.3.5...v1.4.0)
- go.uber.org/zap: v1.17.0 → v1.19.0
- golang.org/x/crypto: 5ea612d → 32db794
- golang.org/x/net: abc4532 → 60bc85c
- golang.org/x/oauth2: f6687ab → 2bc19b1
- golang.org/x/sys: 59db8d7 → 41cdb87
- golang.org/x/term: 6a3ed07 → 6886f2d
- golang.org/x/tools: v0.1.2 → d4cc65f
- gopkg.in/ini.v1: v1.51.0 → v1.62.0
- k8s.io/klog/v2: v2.9.0 → v2.20.0
- k8s.io/utils: efc7438 → bdf08cb
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.22 → v0.0.23

### Removed
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/etcd: [v3.3.13+incompatible](https://github.com/coreos/etcd/tree/v3.3.13)
- github.com/coreos/go-systemd: [95778df](https://github.com/coreos/go-systemd/tree/95778df)
- github.com/coreos/pkg: [399ea9e](https://github.com/coreos/pkg/tree/399ea9e)
- github.com/dgrijalva/jwt-go: [v3.2.0+incompatible](https://github.com/dgrijalva/jwt-go/tree/v3.2.0)
- gotest.tools: v2.2.0+incompatible



# v1.23.0-alpha.1


## Downloads for v1.23.0-alpha.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes.tar.gz) | f7c76f1e077b5d98019347b2c9b79eaa0c79d428542b9c15dab23886c276ca16314f200ca37af914c52264c0e1e5d0bde639d6adf37368d5e7b29d230df00d95
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-src.tar.gz) | f267f26eca20cd7018e68abeeed38aed5c10dbbae7c531c4e08e507196a4dd3f511eb8d41ee8b09495544337d8e1940a8ca04e94084f8dd172698a96564fb070

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | deb110839c2c3cf94ca9b29df2f0b07b3fad6937d7bb6e9d2516d01345c8e324f6ab86fe1d34f1443f04c3d1fc328b53b3d756c295f4ed22f1994071fbc8c9cb
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 1473cb9fc4847b0daff6c9e3189ce55fadc22fb6190161e744e5438066a714cb467fdebfb35f6445a27f5010df94ee602fff492a2382e0f308fda111d53af1f4
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-386.tar.gz) | ed5f5b0777ca51790d185764afc2c812f82ae27c35d897570fc86cabee90dc0a445d9d8c37c981bd3684ba9cd47dc0d75d0094578e79ef7b591d3c1b6564280f
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 39f2a888e7a43c9e4a4018301894786f6babe23d79ab7a143e06444f69bc14aec2e158d355c5b48da4356e7bd72ec9b1268f8b12815c8b709395f36ad9a68a2f
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | b6b8333d8adb4bc6a943bcd2c6cd1a0aeaf0b926d06aa03b759e3c723c81ccc91804debc64fedcd7d678eefdee9bdacc52b2891bd084a15fd5f7918a70e51a15
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 3cb8217b9a5363cebad4989253e02c8a37259b61eafc2f08681508c11c5f68448cad43282257c3d90ad510cc9a62645b7f1adeb99fedf5e13c181495e3754ee4
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | e411700fb13b25deca6347983cdafe47199f0df00086ccd7b3e7d52a7b3bee7e96a85c2568dd52c956fd4ea8b4a6991859c57c9b73a13e06440b456c65b11687
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 6c1395792a175de77436352d0893476363497b0f6a616f4415f91aed5e780d1f25b515021939a7563046237c7b651caba0d1fbf7c4c461677d1b9308b227e94c
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-windows-386.tar.gz) | f3aec7136c21d24a99145ce294a859078fcbf11bae132b8b4081555a6656c0d95ccbaca02a86dc257d557ecebc0673d0771b9cdd10593712a643e8cc0f61d681
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | b29697ba0a25f3d871ffbe5800dcb23ec9fd27c0122a284e17c21f1258f7dd9d341813aeb7826159c7999581a16db19fbb6eeeab48f5c89975df7595d19102c3

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | a5b3edca559b84cd9d22b43b23d0607951d434e185dcb313b831604d83dd306cfc017599994d3944ce77360116024eb59a302851325bb2c29c185a80db2e6eac
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | 2334dbcff3ba22a50f252998eb63991b6c816659dbaa5f749370fc1b1f78f0af7739e50ab64c14a23c4e7dfa8917568e2a3b85bdffdb2cc691ee23ae8f5c8326
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 58674443ce6e359a995dd7c4289bf730e616bcaf336837b77333a206d4e98693d9356a0a670ffbe0b274e2997a8b76a164153cf084f0ff5f91f40f00b5512684
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | f60ebdd04e2348b1ba51540cad93fa24cb133fd25db97150000bffaff8ccb41e1b6506bcde6b7d913aee7701478f975a97775430a82980105383fdb1cc13d260
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | ff008aa0ba1bf755f32c7251c6aceb12b6f9de00d2e2729302b51960e70e486bd82da62d21d70ad81c14e01910ab2afe0fd2509ebfdec050d36f88ee1f0330b2

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 352502f10fbc4579bd9556e3f73ca7513184371ea563d12a39d655d39bb14ccf0f485f4f2b54a77d984c91ff0de2acea7225f98532a1247da5b9ecc65081bc1a
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | af9de95e2b9e4c1f39cb9757d4dca020f7d276b6702302a2d92e7a93e9986528615ce54531e62b96f6e8a0b9863cddbb264f42b1f59374948ac3499af60d9532
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 45a286cb1d469b16d046af02047cf63a8407222e4a39fe696f5652e0587e0c9ffbdbab6505ce85e2726ba10db3189a7fbe70e316bc610caedc8cbb49fed28076
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 7a540a3ff0295998a1679b0ccd50cb1825faf1d0afd6ed08138ab3767c83a2743aa43b122c8da89ee00161f57c0af8d76012e890f9fe6d77b4ee8aff4e32e50f
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 3cd7656221ac2fa161abcf237878cff26c1d97cf77d9b784736c97a56841397ff859e43947d81a83f8fe4164701da41a1dad69b551c4e1fee49b3f8196878236
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 21e63913024e88a48244a598cd400fbae6ce8f8910202f1b635812fbc9281b7c6097eb10a321dd18846484a198845bba58970d83b5119a367862cf8418d4d08c

## Changelog since v1.22.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

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
  --> ([#104389](https://github.com/kubernetes/kubernetes/pull/104389), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
  - Kubeadm: remove the deprecated flag --experimental-patches for the init|join|upgrade commands. The flag --patches is no longer allowed in a mixture with the flag --config. Please use the kubeadm configuration for setting patches for a node using {Init|Join}Configuration.patches. ([#104065](https://github.com/kubernetes/kubernetes/pull/104065), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
 
## Changes by Kind

### Deprecation

- Add apiserver_longrunning_requests metric to replace the soon to be deprecated apiserver_longrunning_gauge metric. ([#103799](https://github.com/kubernetes/kubernetes/pull/103799), [@jyz0309](https://github.com/jyz0309)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Kubeadm: remove the --port flag from the manifest for the kube-controller-manager since the flag has been a NO-OP since 1.22 and insecure serving was removed for the component. ([#104157](https://github.com/kubernetes/kubernetes/pull/104157), [@knight42](https://github.com/knight42)) [SIG Cluster Lifecycle]

### API Change

- CSIDriver.Spec.StorageCapacity can now be modified. ([#101789](https://github.com/kubernetes/kubernetes/pull/101789), [@pohly](https://github.com/pohly)) [SIG Storage]
- Kube-apiserver: The `rbac.authorization.k8s.io/v1alpha1` API version is removed; use the `rbac.authorization.k8s.io/v1` API, available since v1.8. The `scheduling.k8s.io/v1alpha1` API version is removed; use the `scheduling.k8s.io/v1` API, available since v1.14. ([#104248](https://github.com/kubernetes/kubernetes/pull/104248), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, Network and Testing]
- Kube-controller-manager supports '--concurrent-ephemeralvolume-syncs' flag to set the number of ephemeral volume controller workers. ([#102981](https://github.com/kubernetes/kubernetes/pull/102981), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery and Apps]

### Feature

- Adding support for multiple --from-env-file flags ([#101646](https://github.com/kubernetes/kubernetes/pull/101646), [@lauchokyip](https://github.com/lauchokyip)) [SIG CLI]
- All folks to build kubernetes with a custom kube-cross image ([#104185](https://github.com/kubernetes/kubernetes/pull/104185), [@dims](https://github.com/dims)) [SIG Release and Testing]
- Allow node expansion of local volumes ([#102886](https://github.com/kubernetes/kubernetes/pull/102886), [@gnufied](https://github.com/gnufied)) [SIG Storage and Testing]
- Client-go event library allows customizing spam filtering function. 
  It is now possible to override `SpamKeyFunc`, which is used by event filtering to detect spam in the events. ([#103918](https://github.com/kubernetes/kubernetes/pull/103918), [@olagacek](https://github.com/olagacek)) [SIG API Machinery and Instrumentation]
- Constants/variables from k8s.io for STABLE metrics is now supported ([#103654](https://github.com/kubernetes/kubernetes/pull/103654), [@coffeepac](https://github.com/coffeepac)) [SIG Auth, Instrumentation, Node and Testing]
- Display Labels when kubectl describe ingress ([#103894](https://github.com/kubernetes/kubernetes/pull/103894), [@kabab](https://github.com/kabab)) [SIG CLI]
- Expose a `NewUnstructuredExtractor` from apply configurations `meta/v1` package that enables extracting objects into unstructured apply configurations ([#103564](https://github.com/kubernetes/kubernetes/pull/103564), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery, Cluster Lifecycle, Release and Testing]
- Introduce a feature gate DisableKubeletCloudCredentialProviders which allows disabling the in-tree kubelet credential providers.
  
  The DisableKubeletCloudCredentialProviders FeatureGate is currently in Alpha, which means is currently disabled by default. Once the FeatureGate moves to beta, in-tree credential providers will be disabled by default, and users will need to migrate to using external credential providers. ([#102507](https://github.com/kubernetes/kubernetes/pull/102507), [@ostrain](https://github.com/ostrain)) [SIG Cloud Provider]
- Introduces a new metric: admission_webhook_request_total with the following labels: name (string) - the webhook name, type (string) - the admission type, operation (string) - the requested verb, code (int) - the HTTP status code, rejected (bool) - whether the request was rejected, namespace (string) - the namespace of the requested resource. ([#103162](https://github.com/kubernetes/kubernetes/pull/103162), [@rmoriar1](https://github.com/rmoriar1)) [SIG API Machinery and Instrumentation]
- Kube-up.sh installs csi-proxy v1.0.1-gke.0 ([#104426](https://github.com/kubernetes/kubernetes/pull/104426), [@mauriciopoppe](https://github.com/mauriciopoppe)) [SIG Cloud Provider, Storage and Windows]
- Kubeadm: add support for dry running "kubeadm join". The new flag "kubeadm join --dry-run" is similar to the existing flag for "kubeadm init/upgrade" and allows you to see what changes would be applied. ([#103027](https://github.com/kubernetes/kubernetes/pull/103027), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Golang 1.16.7 ([#104199](https://github.com/kubernetes/kubernetes/pull/104199), [@cpanato](https://github.com/cpanato)) [SIG Cloud Provider, Instrumentation, Release and Testing]
- The ServiceAccountIssuerDiscovery feature gate is removed. It reached GA in Kubernetes 1.21. ([#103685](https://github.com/kubernetes/kubernetes/pull/103685), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery and Auth]
- Updated Cluster Autosaler to version 1.22.0. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.22.0 ([#104293](https://github.com/kubernetes/kubernetes/pull/104293), [@x13n](https://github.com/x13n)) [SIG Autoscaling and Cloud Provider]
- Updates the following images to pick up CVE fixes:
  - debian to v1.9.0
  - debian-iptables to v1.6.6
  - setcap to v2.0.4 ([#104142](https://github.com/kubernetes/kubernetes/pull/104142), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery, Release and Testing]

### Documentation

- Update description of --audit-log-maxbackup to describe behavior when value = 0 ([#103843](https://github.com/kubernetes/kubernetes/pull/103843), [@Arkessler](https://github.com/Arkessler)) [SIG API Machinery]

### Bug or Regression

- 1. Changes json representation for a conflicted taint to Key=Effect when a conflicted taint occurs in kubectl taint. ([#104011](https://github.com/kubernetes/kubernetes/pull/104011), [@manugupt1](https://github.com/manugupt1)) [SIG CLI]
- A new server run option 'shutdown-send-retry-after'  has been introduced. If true the HTTP Server
  will continue listening until all non longrunning request(s) in flight have been drained, during this window all 
  incoming requests will be rejected with a status code 429 and a 'Retry-After' response header. ([#101257](https://github.com/kubernetes/kubernetes/pull/101257), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Adds Kubernetes Events to the Kubelet Graceful Shutdown feature ([#101081](https://github.com/kubernetes/kubernetes/pull/101081), [@rphillips](https://github.com/rphillips)) [SIG Node]
- CA, certificate and key bundles for the generic-apiserver based servers will be reloaded immediately after the files are changed. ([#104102](https://github.com/kubernetes/kubernetes/pull/104102), [@tnqn](https://github.com/tnqn)) [SIG API Machinery and Testing]
- Fix kube-apiserver metric reporting for the deprecated watch path of /api/<version>/watch/... ([#104161](https://github.com/kubernetes/kubernetes/pull/104161), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Instrumentation]
- Fix: skip case sensitivity when checking Azure NSG rules ([#104384](https://github.com/kubernetes/kubernetes/pull/104384), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fixed an issue which didn't append OS's environment variables with the one provided in Credential Provider Config file, which may lead to failed execution of external credential provider binary. 
  See https://github.com/kubernetes/kubernetes/issues/102750 ([#103231](https://github.com/kubernetes/kubernetes/pull/103231), [@n4j](https://github.com/n4j)) [SIG Auth and Node]
- Fixed architecture within manifest for non `amd64` etcd images. ([#104116](https://github.com/kubernetes/kubernetes/pull/104116), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery]
- Fixed bug where kubectl would emit duplicate warning messages for flag names that contain an underscore and recommend using a nonexistent flag in some cases ([#103852](https://github.com/kubernetes/kubernetes/pull/103852), [@brianpursley](https://github.com/brianpursley)) [SIG CLI and Cluster Lifecycle]
- Graceful node shutdown, allow the actual inhibit delay to be greater than the expected inhibit delay ([#103137](https://github.com/kubernetes/kubernetes/pull/103137), [@wzshiming](https://github.com/wzshiming)) [SIG Node]
- Kube-apiserver: Avoids unnecessary repeated calls to admission webhooks that reject an update or delete request. ([#104182](https://github.com/kubernetes/kubernetes/pull/104182), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-proxy: delete stale conntrack UDP entries for loadbalancer ingress IP. ([#104009](https://github.com/kubernetes/kubernetes/pull/104009), [@aojea](https://github.com/aojea)) [SIG Network]
- Kubeadm: When adding an etcd peer to an existing cluster, if an error is returned indicating the peer has already been added, this is accepted and a ListMembers call is used instead to return the existing cluster. This helps diminish the exponential backoff when the first AddMember call times out, while still retaining a similar performance when the peer had already been added from a previous call. ([#104134](https://github.com/kubernetes/kubernetes/pull/104134), [@ihgann](https://github.com/ihgann)) [SIG Cluster Lifecycle]
- Pass additional flags to subpath mount to avoid flakes in certain conditions ([#104253](https://github.com/kubernetes/kubernetes/pull/104253), [@mauriciopoppe](https://github.com/mauriciopoppe)) [SIG Storage]
- Update Go used to build migrate script in etcd image to v1.16.7 ([#104301](https://github.com/kubernetes/kubernetes/pull/104301), [@serathius](https://github.com/serathius)) [SIG API Machinery and Release]

### Other (Cleanup or Flake)

- Deprecate apiserver_longrunning_gauge and apiserver_register_watchers in 1.23.0 ([#103793](https://github.com/kubernetes/kubernetes/pull/103793), [@yan-lgtm](https://github.com/yan-lgtm)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Kube-apiserver: sets an upper-bound on the lifetime of idle keep-alive connections and time to read the headers of incoming requests ([#103958](https://github.com/kubernetes/kubernetes/pull/103958), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Node]
- Kubeadm: external etcd endpoints passed in the ClusterConfiguration that have Unicode characters are no longer IDNA encoded (converted to Punycode). They are now just URL encoded as per Go's implementation of RFC-3986, have duplicate "/" removed from the URL paths, and passed like that directly to the kube-apiserver --etcd-servers flag. If you have etcd endpoints that have Unicode characters, it is advisable to encode them in advance with tooling that is fully IDNA compliant. If you don't do that, the Go standard library (used in k8s and etcd) would do it for you when making requests to the endpoints. ([#103801](https://github.com/kubernetes/kubernetes/pull/103801), [@gkarthiks](https://github.com/gkarthiks)) [SIG Cluster Lifecycle]
- Kubeadm: update references to legacy artifacts locations, the 'ci-cross' prefix has been removed from the version match as it does not exist in the new 'gs://k8s-release-dev' bucket ([#103813](https://github.com/kubernetes/kubernetes/pull/103813), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Migratecmd/kube-proxy/app logs to structured logging ([#98913](https://github.com/kubernetes/kubernetes/pull/98913), [@yxxhero](https://github.com/yxxhero)) [SIG Network]
- Surface warning when users don't set propagationPolicy for jobs while deleting ([#104080](https://github.com/kubernetes/kubernetes/pull/104080), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps]
- The AllowInsecureBackendProxy feature gate is removed. It reached GA in Kubernetes 1.21. ([#103796](https://github.com/kubernetes/kubernetes/pull/103796), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery]
- The `StartupProbe` feature gate that is GA since v1.20 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104168](https://github.com/kubernetes/kubernetes/pull/104168), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Node]
- The apiserver exposes 4 new metrics that allow to track the status of the Service CIDRs allocations:
      - current number of available IPs per Service CIDR
      - current number of used IPs per Service CIDR
      - total number of allocation per Service CIDR
      - total number of allocation errors per ServiceCIDR ([#104119](https://github.com/kubernetes/kubernetes/pull/104119), [@aojea](https://github.com/aojea)) [SIG Apps, Instrumentation and Network]
- The flag `--deployment-controller-sync-period` has no effect now, deprecate it and will be removed in v1.24. ([#103538](https://github.com/kubernetes/kubernetes/pull/103538), [@Pingan2017](https://github.com/Pingan2017)) [SIG Apps]
- Troubleshooting: informers log handlers that take more than 100 milliseconds to process an object if the DeltaFIFO queue starts to grow beyond 10 elements. ([#103917](https://github.com/kubernetes/kubernetes/pull/103917), [@aojea](https://github.com/aojea)) [SIG API Machinery]
- Update cri-tools dependency to v1.22.0 ([#104430](https://github.com/kubernetes/kubernetes/pull/104430), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- ``gcr.io/kubernetes-e2e-test-images`` will no longer be used in E2E / CI testing, ``k8s.gcr.io/e2e-test-images`` will be used instead. ([#103724](https://github.com/kubernetes/kubernetes/pull/103724), [@claudiubelu](https://github.com/claudiubelu)) [SIG API Machinery and Testing]

## Dependencies

### Added
- github.com/google/martian/v3: [v3.1.0](https://github.com/google/martian/v3/tree/v3.1.0)
- github.com/kr/fs: [v0.1.0](https://github.com/kr/fs/tree/v0.1.0)
- github.com/pkg/sftp: [v1.10.1](https://github.com/pkg/sftp/tree/v1.10.1)

### Changed
- cloud.google.com/go/bigquery: v1.4.0 → v1.8.0
- cloud.google.com/go/storage: v1.6.0 → v1.10.0
- cloud.google.com/go: v0.54.0 → v0.81.0
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [7901bc8 → ea6160c](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/7901bc8...ea6160c)
- github.com/bketelsen/crypt: [5cbc8cc → v0.0.4](https://github.com/bketelsen/crypt/compare/5cbc8cc...v0.0.4)
- github.com/golang/mock: [v1.4.4 → v1.5.0](https://github.com/golang/mock/compare/v1.4.4...v1.5.0)
- github.com/google/pprof: [1ebb73c → cbba55b](https://github.com/google/pprof/compare/1ebb73c...cbba55b)
- github.com/hashicorp/golang-lru: [v0.5.1 → v0.5.0](https://github.com/hashicorp/golang-lru/compare/v0.5.1...v0.5.0)
- github.com/ianlancetaylor/demangle: [5e5cf60 → 28f6c0f](https://github.com/ianlancetaylor/demangle/compare/5e5cf60...28f6c0f)
- github.com/magiconair/properties: [v1.8.1 → v1.8.5](https://github.com/magiconair/properties/compare/v1.8.1...v1.8.5)
- github.com/mitchellh/go-homedir: [v1.1.0 → v1.0.0](https://github.com/mitchellh/go-homedir/compare/v1.1.0...v1.0.0)
- github.com/mitchellh/mapstructure: [v1.1.2 → v1.4.1](https://github.com/mitchellh/mapstructure/compare/v1.1.2...v1.4.1)
- github.com/pelletier/go-toml: [v1.2.0 → v1.9.3](https://github.com/pelletier/go-toml/compare/v1.2.0...v1.9.3)
- github.com/prometheus/common: [v0.26.0 → v0.28.0](https://github.com/prometheus/common/compare/v0.26.0...v0.28.0)
- github.com/spf13/afero: [v1.2.2 → v1.6.0](https://github.com/spf13/afero/compare/v1.2.2...v1.6.0)
- github.com/spf13/cast: [v1.3.0 → v1.3.1](https://github.com/spf13/cast/compare/v1.3.0...v1.3.1)
- github.com/spf13/cobra: [v1.1.3 → v1.2.1](https://github.com/spf13/cobra/compare/v1.1.3...v1.2.1)
- github.com/spf13/jwalterweatherman: [v1.0.0 → v1.1.0](https://github.com/spf13/jwalterweatherman/compare/v1.0.0...v1.1.0)
- github.com/spf13/viper: [v1.7.0 → v1.8.1](https://github.com/spf13/viper/compare/v1.7.0...v1.8.1)
- go.opencensus.io: v0.22.3 → v0.23.0
- golang.org/x/net: 37e1c6a → abc4532
- golang.org/x/oauth2: bf48bf1 → f6687ab
- google.golang.org/api: v0.20.0 → v0.46.0
- google.golang.org/appengine: v1.6.5 → v1.6.7
- gopkg.in/ini.v1: v1.51.0 → v1.62.0
- honnef.co/go/tools: v0.0.1-2020.1.3 → v0.0.1-2020.1.4
- k8s.io/gengo: b6c5ce2 → 485abfe
- k8s.io/kube-openapi: 9528897 → 7fbd8d5
- k8s.io/utils: 4b05e18 → efc7438

### Removed
- cloud.google.com/go/datastore: v1.1.0
- cloud.google.com/go/pubsub: v1.2.0
- github.com/alecthomas/units: [f65c72e](https://github.com/alecthomas/units/tree/f65c72e)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/etcd: [v3.3.13+incompatible](https://github.com/coreos/etcd/tree/v3.3.13)
- github.com/coreos/go-systemd: [95778df](https://github.com/coreos/go-systemd/tree/95778df)
- github.com/coreos/pkg: [399ea9e](https://github.com/coreos/pkg/tree/399ea9e)
- github.com/dgrijalva/jwt-go: [v3.2.0+incompatible](https://github.com/dgrijalva/jwt-go/tree/v3.2.0)
- github.com/google/martian: [v2.1.0+incompatible](https://github.com/google/martian/tree/v2.1.0)
- github.com/jpillora/backoff: [v1.0.0](https://github.com/jpillora/backoff/tree/v1.0.0)