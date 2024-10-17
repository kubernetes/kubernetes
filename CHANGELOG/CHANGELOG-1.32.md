<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.32.0-alpha.2](#v1320-alpha2)
  - [Downloads for v1.32.0-alpha.2](#downloads-for-v1320-alpha2)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.32.0-alpha.1](#changelog-since-v1320-alpha1)
  - [Changes by Kind](#changes-by-kind)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Documentation](#documentation)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.32.0-alpha.1](#v1320-alpha1)
  - [Downloads for v1.32.0-alpha.1](#downloads-for-v1320-alpha1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.31.0](#changelog-since-v1310)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Documentation](#documentation-1)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)

<!-- END MUNGE: GENERATED_TOC -->

# v1.32.0-alpha.2


## Downloads for v1.32.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes.tar.gz) | 12fa6fbea15ce6c682f35d6a1942248a6e3d02112b5d4cd8ad4cb71c05234469a61e0a0a24cd7c0f31d03dbbfdba0c1f824b3c813ffade22c1df880d71961808
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-src.tar.gz) | 41a87e299da2e0793859bf2ce61356313215f23036b1c15a56040089d0a6a049a38374cc4d55c25f1167f7b111c0b23745ebd271194392f67d57784f6b310079

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | 5eaef34ed732b964eea1c695634c0a2310fc7383df59b10ee5ae620eea6df86ac089c77e5ea49e0a48ef3b4bbeeee5f98917cc1d82550f8ffd915829aa182c2d
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 2d25f8d105a2bb1cf5087e63689703a9bcaf89c98cd92bf9b95204c5544c7459ffcc62998cbb5118b26591ee56c75610b2407fa14e28af575c55d7f67e3f005f
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-linux-386.tar.gz) | a6626f989b0045d8c12cda459596766ba591dd4586a1d2ab2de25433f9195015b46b4cf1cc9db75945e0ca8e5453fd86b4f6dd49df8ec2ac0c40edcb4d7f21c9
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | d80eebb21798b8c5043c7b08b15d634c8c9e9179b44ef1cd9601fa05223c7ba696e5fe833f34778c457ae6e20b603156501122602697a159f790edb90659fa49
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | d3a90dd1e38f379a5433023f2d10620a96a8b667baf51bc893b8ebb622ea675e7f965b13e5f94d0c0346f426ba7912ae80e31e36982bb30c3efd0f9e2dbd44c3
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | ab7f0dca923cfbca492cf02c4625e946d4d9013d00ceee91c8adbb66cd0c42c305b2a0912fee65fba6f93d4ac7180729afbe65e02a98453334489fbddcfa81dd
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | f669e9d18a6d36462a13c5b1e3f71fd812554671b27070445275852788ad927d5f5a95964a6e2f035fc7cdcaeab68f130c97b256a1a3101877883f50b89d4a56
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 870a52113f5c678271db4adbfd86c42710b9299d2d6f94581288ee5bce619723f3317bb0f36fa964d972c22d0a4539caee9a7caeb342fe1595f845de1b222812
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-windows-386.tar.gz) | caed3c909f1edb95d26e8ba1fd4a4dba8a2b377c22e9646cb85d208e4eb15dedf829b1a9f4b3c2afde85177b891d0482e3213668f8db0dcb549b40d209ec7ae5
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | f020c3de77e4a6b34d3fc529932daec3bfafcf718e229fa111903a79635cae1012fc62225e5513c28fb173a0c52927ad152419fba6ff4c8afb148ea1a6ceba6f
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | a0e1c0f0dbe19ff8dcffd3713b828088b30c9f0ede4f7e65e083e3714e15da26bb361f2924a5edc7cf4f97c23cf9eab806cd11d8a616cb77df097a5ca1812e0f

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | d40f6a3dc056b68eb78788bb91e6f1d07f81b8b58ae0bda787be99c0f41c0ec87d2f652eb15aba0df5ab41f5c96144980415856155a7011d3f6195aba8030ff3
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 7a56e4537b3d61875e8d61645383b82c4609b26b0eef17a1d6967cb52d990ad64a2f0c39910b0a2188930dc28ce1cec44f6aec86eba0dc4bdfc7329553d5b3d9
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | afdd9540cee13f8196fdaf5edbaf5f2ae5c792b94dbfaab461345a62d709591f13a06a037d3dd9374775fb1a3db82bf337a873391c989ed864790089f332f3a8
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | f5d8998bc1be3a31bf510af6dd5aa43d165d4424faa5157dd9fc6640f34e75c967379f3ea51f2049675843f8f3222d42cdb8ad61da0ccc5b35b21925f7318d02

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 0414c3d74019d5f932b3effba27580bd86ae6d8a6ae9f4c2a8967f70f15167f8c2805451fb4f18aaab8b9e1c0e47eaf627e4ea5844311ba095ddcfa2383ba4ff
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 96a13271ab2cd2a3c5fe556de71f3b862b6263abe793a87ed123ac4bb928dc22ff9ad0219a0dc21669cc5fc333000091185fbc4bd8415f370870b56491f0fed4
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 2c92a70ca1285b3146b743dc812323db3eb1f52e0978ab4c42af9d4218260a4eb445928453298d264166768eb87f4b0db997e3cfd370112685e9836e890562bb
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 65a84611fe4805c7937b0406a3818be923036402339a61cf1f0ce580229186bd520c65e083af8f9c9fce5dba15c4786c146d4d5254c878bc3d989bfc9b21db49
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | f29148bf2230b726d57120cb62ebaf2f0d47b46fc4e5ad5d5a332c79a93e310bfacb471e7e95a79ba850933c47471bd934415fa1aec3cb655433fc034ed54296

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.32.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.32.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.32.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.32.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.32.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.32.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.32.0-alpha.1

## Changes by Kind

### API Change

- Fixed a bug in the NestedNumberAsFloat64 Unstructured field accessor that could cause it to return rounded float64 values instead of errors when accessing very large int64 values. ([#128099](https://github.com/kubernetes/kubernetes/pull/128099), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Introduce compressible resource setting on system reserved and kube reserved slices ([#125982](https://github.com/kubernetes/kubernetes/pull/125982), [@harche](https://github.com/harche)) [SIG Node]
- Kubelet: the `--image-credential-provider-config` file is now loaded with strict deserialization, which fails if the config file contains duplicate or unknown fields. This protects against accidentally running with config files that are malformed, mis-indented, or have typos in field names, and getting unexpected behavior. ([#128062](https://github.com/kubernetes/kubernetes/pull/128062), [@aramase](https://github.com/aramase)) [SIG Auth and Node]
- Promoted `CustomResourceFieldSelectors` to stable; the feature is enabled by default. `--feature-gates=CustomResourceFieldSelectors=true` not needed on kube-apiserver binaries and will be removed in a future release. ([#127673](https://github.com/kubernetes/kubernetes/pull/127673), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]

### Feature

- Add option to enable leader election in local-up-cluster.sh via the LEADER_ELECT cli flag. ([#127786](https://github.com/kubernetes/kubernetes/pull/127786), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Added status for extended Pod resources within the `status.containerStatuses[].resources` field. ([#124227](https://github.com/kubernetes/kubernetes/pull/124227), [@iholder101](https://github.com/iholder101)) [SIG Node and Testing]
- Allow pods to use the `net.ipv4.tcp_rmem` and `net.ipv4.tcp_wmem` sysctl by default
  when the kernel version is 4.15 or higher. With the kernel 4.15 the sysctl became namespaced.
  Pod Security admission allows these sysctl in v1.32+ versions of the baseline and restricted policies. ([#127489](https://github.com/kubernetes/kubernetes/pull/127489), [@pacoxu](https://github.com/pacoxu)) [SIG Auth, Network and Node]
- Graduates the `WatchList` feature gate to Beta for kube-apiserver and enables `WatchListClient` for KCM. ([#128053](https://github.com/kubernetes/kubernetes/pull/128053), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Kubernetes is now built with go 1.23.1 ([#127611](https://github.com/kubernetes/kubernetes/pull/127611), [@haitch](https://github.com/haitch)) [SIG Release and Testing]
- Kubernetes is now built with go 1.23.2 ([#128110](https://github.com/kubernetes/kubernetes/pull/128110), [@haitch](https://github.com/haitch)) [SIG Release and Testing]
- LoadBalancerIPMode feature is now marked as GA. ([#127348](https://github.com/kubernetes/kubernetes/pull/127348), [@RyanAoh](https://github.com/RyanAoh)) [SIG Apps, Network and Testing]
- Output for the `ScalingReplicaSet` event has changed from:
      Scaled <up|down> replica set <replica-set-name> to <new-value> from <old-value>
  to:
      Scaled <up|down> replica set <replica-set-name> from <old-value> to <new-value> ([#125118](https://github.com/kubernetes/kubernetes/pull/125118), [@jsoref](https://github.com/jsoref)) [SIG Apps and CLI]
- Promote the feature gates `StrictCostEnforcementForVAP` and `StrictCostEnforcementForWebhooks` to GA. ([#127302](https://github.com/kubernetes/kubernetes/pull/127302), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]
- Removed attachable volume limits from the capacity of the node for the following volume type when the kubelet is started, affecting the following volume types when the corresponding csi driver is installed:
  - `awsElasticBlockStore` for `ebs.csi.aws.com`
  - `azureDisk` for `disk.csi.azure.com`
  - `gcePersistentDisk` for `pd.csi.storage.googleapis.com`
  - `cinder` for `cinder.csi.openstack.org`
  - `csi`
  But it's still enforced using a limit in CSINode objects. ([#126924](https://github.com/kubernetes/kubernetes/pull/126924), [@carlory](https://github.com/carlory)) [SIG Storage]
- Revert Go version used to build Kubernetes to 1.23.0 ([#127861](https://github.com/kubernetes/kubernetes/pull/127861), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- The scheduler implements QueueingHint in VolumeBinding plugin's CSIDriver event, which enhances the throughput of scheduling. ([#125171](https://github.com/kubernetes/kubernetes/pull/125171), [@YamasouA](https://github.com/YamasouA)) [SIG Scheduling and Storage]
- Vendor: updated system-validators to v1.9.0 ([#128149](https://github.com/kubernetes/kubernetes/pull/128149), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]

### Documentation

- Kubeadm: fixed a misleading output (typo) when executing the "kubeadm init" command. ([#128118](https://github.com/kubernetes/kubernetes/pull/128118), [@amaddio](https://github.com/amaddio)) [SIG Cluster Lifecycle]

### Bug or Regression

- Fix a bug where the kubelet ephemerally fails with `failed to initialize top level QOS containers: root container [kubepods] doesn't exist`, due to the cpuset cgroup being deleted on v2 with systemd cgroup manager. ([#125923](https://github.com/kubernetes/kubernetes/pull/125923), [@haircommander](https://github.com/haircommander)) [SIG Node and Testing]
- Fix data race in kubelet/volumemanager ([#127919](https://github.com/kubernetes/kubernetes/pull/127919), [@carlory](https://github.com/carlory)) [SIG Apps, Node and Storage]
- Fixes a race condition that could result in erroneous volume unmounts for flex volume plugins on kubelet restart ([#127669](https://github.com/kubernetes/kubernetes/pull/127669), [@olyazavr](https://github.com/olyazavr)) [SIG Storage]
- Fixes a regression introduced in 1.29 where conntrack entries for UDP connections
  to deleted pods did not get cleaned up correctly, which could (among other things)
  cause DNS problems when DNS pods were restarted. ([#127780](https://github.com/kubernetes/kubernetes/pull/127780), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Node shutdown controller now makes a best effort to wait for CSI Drivers to complete the volume teardown process according to the pod priority groups. ([#125070](https://github.com/kubernetes/kubernetes/pull/125070), [@torredil](https://github.com/torredil)) [SIG Node, Storage and Testing]
- Reduce memory usage/allocations during wait for volume attachment ([#126575](https://github.com/kubernetes/kubernetes/pull/126575), [@Lucaber](https://github.com/Lucaber)) [SIG Node and Storage]
- Scheduler will start considering the resource requests of existing sidecar containers during the scoring process. ([#127878](https://github.com/kubernetes/kubernetes/pull/127878), [@AxeZhan](https://github.com/AxeZhan)) [SIG Scheduling and Testing]
- The name port of the sidecar will also be allowed to be used ([#127976](https://github.com/kubernetes/kubernetes/pull/127976), [@chengjoey](https://github.com/chengjoey)) [SIG Network]
- Unallowed label values will show up as "unexpected" in all system components metrics ([#128100](https://github.com/kubernetes/kubernetes/pull/128100), [@yongruilin](https://github.com/yongruilin)) [SIG Architecture and Instrumentation]

### Other (Cleanup or Flake)

- CRI client: use default timeout for `ImageFsInfo` RPC ([#128052](https://github.com/kubernetes/kubernetes/pull/128052), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Fix spacing in --validate flag description in kubectl. ([#128081](https://github.com/kubernetes/kubernetes/pull/128081), [@soltysh](https://github.com/soltysh)) [SIG CLI]
- Kube-apiserver ResourceQuotaConfiguration admission plugin subsection within `--admission-control-config-file` files are now validated strictly (EnableStrict). Duplicate and unknown fields in the configuration will now cause an error. ([#128038](https://github.com/kubernetes/kubernetes/pull/128038), [@seans3](https://github.com/seans3)) [SIG API Machinery]
- Kube-apiserver `--egress-selector-config-file` files are now validated strictly (EnableStrict). Duplicate and unknown fields in the configuration will now cause an error. ([#128011](https://github.com/kubernetes/kubernetes/pull/128011), [@seans3](https://github.com/seans3)) [SIG API Machinery and Testing]
- Kube-apiserver `--tracing-config-file` file is now validated strictly (EnableStrict). Duplicate and unknown fields in the configuration will now cause an error. ([#128073](https://github.com/kubernetes/kubernetes/pull/128073), [@seans3](https://github.com/seans3)) [SIG API Machinery]
- Kube-controller-manager `--leader-migration-config` files are now validated strictly (EnableStrict). Duplicate and unknown fields in the configuration will now cause an error. ([#128009](https://github.com/kubernetes/kubernetes/pull/128009), [@seans3](https://github.com/seans3)) [SIG API Machinery and Cloud Provider]
- Kubeadm: increased the verbosity of API client dry-run actions during the subcommands "init", "join", "upgrade" and "reset". Allowed dry-run on 'kubeadm join' even if there is no existing cluster by utilizing a faked, in-memory cluster-info ConfigMap. ([#126776](https://github.com/kubernetes/kubernetes/pull/126776), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl: `-o` can now be used as a shortcut for `--output` in `kubectl explain <resource> --output plaintext-openapiv2` ([#127869](https://github.com/kubernetes/kubernetes/pull/127869), [@ak20102763](https://github.com/ak20102763)) [SIG CLI]
- Removes the feature gate ComponentSLIs, which has been promoted to stable since 1.29. ([#127787](https://github.com/kubernetes/kubernetes/pull/127787), [@Jefftree](https://github.com/Jefftree)) [SIG Architecture and Instrumentation]
- The getters for the field name and typeDescription of the Reflector struct were renamed. ([#128035](https://github.com/kubernetes/kubernetes/pull/128035), [@alexanderstephan](https://github.com/alexanderstephan)) [SIG API Machinery]
- The kube-proxy command line flags `--healthz-port` and `--metrics-port`, which were previously deprecated, have now been removed. ([#127930](https://github.com/kubernetes/kubernetes/pull/127930), [@aroradaman](https://github.com/aroradaman)) [SIG Network and Windows]
- The members name and typeDescription of the Reflector struct are now exported to allow for better user extensibility. ([#127663](https://github.com/kubernetes/kubernetes/pull/127663), [@alexanderstephan](https://github.com/alexanderstephan)) [SIG API Machinery]
- Upgrades functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.4.2 and https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.5.0 ([#127965](https://github.com/kubernetes/kubernetes/pull/127965), [@koba1t](https://github.com/koba1t)) [SIG CLI]
- `kubectl apply --server-side` now supports `--subresource` congruent to `kubelctl patch` ([#127634](https://github.com/kubernetes/kubernetes/pull/127634), [@deads2k](https://github.com/deads2k)) [SIG CLI and Testing]

## Dependencies

### Added
- github.com/Microsoft/hnslib: [v0.0.7](https://github.com/Microsoft/hnslib/tree/v0.0.7)

### Changed
- github.com/armon/circbuf: [bbbad09 → 5111143](https://github.com/armon/circbuf/compare/bbbad09...5111143)
- github.com/docker/docker: [v27.1.1+incompatible → v26.1.4+incompatible](https://github.com/docker/docker/compare/v27.1.1...v26.1.4)
- github.com/exponent-io/jsonpath: [d6023ce → 1de76d7](https://github.com/exponent-io/jsonpath/compare/d6023ce...1de76d7)
- github.com/google/cel-go: [v0.20.1 → v0.21.0](https://github.com/google/cel-go/compare/v0.20.1...v0.21.0)
- github.com/gregjones/httpcache: [9cad4c3 → 901d907](https://github.com/gregjones/httpcache/compare/9cad4c3...901d907)
- github.com/jonboulle/clockwork: [v0.2.2 → v0.4.0](https://github.com/jonboulle/clockwork/compare/v0.2.2...v0.4.0)
- github.com/moby/spdystream: [v0.4.0 → v0.5.0](https://github.com/moby/spdystream/compare/v0.4.0...v0.5.0)
- github.com/moby/sys/mountinfo: [v0.7.1 → v0.7.2](https://github.com/moby/sys/compare/mountinfo/v0.7.1...mountinfo/v0.7.2)
- github.com/mohae/deepcopy: [491d360 → c48cc78](https://github.com/mohae/deepcopy/compare/491d360...c48cc78)
- github.com/opencontainers/runc: [v1.1.14 → v1.1.15](https://github.com/opencontainers/runc/compare/v1.1.14...v1.1.15)
- github.com/stoewer/go-strcase: [v1.2.0 → v1.3.0](https://github.com/stoewer/go-strcase/compare/v1.2.0...v1.3.0)
- github.com/urfave/cli: [v1.22.15 → v1.22.1](https://github.com/urfave/cli/compare/v1.22.15...v1.22.1)
- github.com/xiang90/probing: [43a291a → a49e3df](https://github.com/xiang90/probing/compare/43a291a...a49e3df)
- golang.org/x/crypto: v0.26.0 → v0.28.0
- golang.org/x/mod: v0.20.0 → v0.21.0
- golang.org/x/net: v0.28.0 → v0.30.0
- golang.org/x/oauth2: v0.21.0 → v0.23.0
- golang.org/x/sys: v0.23.0 → v0.26.0
- golang.org/x/term: v0.23.0 → v0.25.0
- golang.org/x/text: v0.17.0 → v0.19.0
- golang.org/x/time: v0.3.0 → v0.7.0
- golang.org/x/tools: v0.24.0 → v0.26.0
- k8s.io/system-validators: v1.8.0 → v1.9.0
- sigs.k8s.io/json: bc3834c → 9aa6b5e
- sigs.k8s.io/kustomize/api: v0.17.2 → v0.18.0
- sigs.k8s.io/kustomize/cmd/config: v0.14.1 → v0.15.0
- sigs.k8s.io/kustomize/kustomize/v5: v5.4.2 → v5.5.0
- sigs.k8s.io/kustomize/kyaml: v0.17.1 → v0.18.1

### Removed
- github.com/Microsoft/cosesign1go: [v1.1.0](https://github.com/Microsoft/cosesign1go/tree/v1.1.0)
- github.com/Microsoft/didx509go: [v0.0.3](https://github.com/Microsoft/didx509go/tree/v0.0.3)
- github.com/Microsoft/hcsshim: [v0.12.6](https://github.com/Microsoft/hcsshim/tree/v0.12.6)
- github.com/OneOfOne/xxhash: [v1.2.8](https://github.com/OneOfOne/xxhash/tree/v1.2.8)
- github.com/agnivade/levenshtein: [v1.1.1](https://github.com/agnivade/levenshtein/tree/v1.1.1)
- github.com/akavel/rsrc: [v0.10.2](https://github.com/akavel/rsrc/tree/v0.10.2)
- github.com/chzyer/logex: [v1.1.10](https://github.com/chzyer/logex/tree/v1.1.10)
- github.com/chzyer/test: [a1ea475](https://github.com/chzyer/test/tree/a1ea475)
- github.com/containerd/cgroups/v3: [v3.0.3](https://github.com/containerd/cgroups/tree/v3.0.3)
- github.com/containerd/containerd: [v1.7.20](https://github.com/containerd/containerd/tree/v1.7.20)
- github.com/containerd/continuity: [v0.4.2](https://github.com/containerd/continuity/tree/v0.4.2)
- github.com/containerd/fifo: [v1.1.0](https://github.com/containerd/fifo/tree/v1.1.0)
- github.com/containerd/go-runc: [v1.0.0](https://github.com/containerd/go-runc/tree/v1.0.0)
- github.com/containerd/protobuild: [v0.3.0](https://github.com/containerd/protobuild/tree/v0.3.0)
- github.com/containerd/stargz-snapshotter/estargz: [v0.14.3](https://github.com/containerd/stargz-snapshotter/tree/estargz/v0.14.3)
- github.com/decred/dcrd/dcrec/secp256k1/v4: [v4.2.0](https://github.com/decred/dcrd/tree/dcrec/secp256k1/v4/v4.2.0)
- github.com/docker/cli: [v24.0.0+incompatible](https://github.com/docker/cli/tree/v24.0.0)
- github.com/docker/distribution: [v2.8.2+incompatible](https://github.com/docker/distribution/tree/v2.8.2)
- github.com/docker/docker-credential-helpers: [v0.7.0](https://github.com/docker/docker-credential-helpers/tree/v0.7.0)
- github.com/docker/go-events: [e31b211](https://github.com/docker/go-events/tree/e31b211)
- github.com/go-ini/ini: [v1.67.0](https://github.com/go-ini/ini/tree/v1.67.0)
- github.com/gobwas/glob: [v0.2.3](https://github.com/gobwas/glob/tree/v0.2.3)
- github.com/goccy/go-json: [v0.10.2](https://github.com/goccy/go-json/tree/v0.10.2)
- github.com/google/go-containerregistry: [v0.20.1](https://github.com/google/go-containerregistry/tree/v0.20.1)
- github.com/gorilla/mux: [v1.8.1](https://github.com/gorilla/mux/tree/v1.8.1)
- github.com/josephspurrier/goversioninfo: [v1.4.0](https://github.com/josephspurrier/goversioninfo/tree/v1.4.0)
- github.com/klauspost/compress: [v1.17.0](https://github.com/klauspost/compress/tree/v1.17.0)
- github.com/lestrrat-go/backoff/v2: [v2.0.8](https://github.com/lestrrat-go/backoff/tree/v2.0.8)
- github.com/lestrrat-go/blackmagic: [v1.0.2](https://github.com/lestrrat-go/blackmagic/tree/v1.0.2)
- github.com/lestrrat-go/httpcc: [v1.0.1](https://github.com/lestrrat-go/httpcc/tree/v1.0.1)
- github.com/lestrrat-go/iter: [v1.0.2](https://github.com/lestrrat-go/iter/tree/v1.0.2)
- github.com/lestrrat-go/jwx: [v1.2.28](https://github.com/lestrrat-go/jwx/tree/v1.2.28)
- github.com/lestrrat-go/option: [v1.0.1](https://github.com/lestrrat-go/option/tree/v1.0.1)
- github.com/linuxkit/virtsock: [f8cee7d](https://github.com/linuxkit/virtsock/tree/f8cee7d)
- github.com/mattn/go-shellwords: [v1.0.12](https://github.com/mattn/go-shellwords/tree/v1.0.12)
- github.com/mitchellh/go-homedir: [v1.1.0](https://github.com/mitchellh/go-homedir/tree/v1.1.0)
- github.com/moby/sys/sequential: [v0.5.0](https://github.com/moby/sys/tree/sequential/v0.5.0)
- github.com/open-policy-agent/opa: [v0.67.1](https://github.com/open-policy-agent/opa/tree/v0.67.1)
- github.com/pelletier/go-toml: [v1.9.5](https://github.com/pelletier/go-toml/tree/v1.9.5)
- github.com/rcrowley/go-metrics: [10cdbea](https://github.com/rcrowley/go-metrics/tree/10cdbea)
- github.com/tchap/go-patricia/v2: [v2.3.1](https://github.com/tchap/go-patricia/tree/v2.3.1)
- github.com/vbatts/tar-split: [v0.11.3](https://github.com/vbatts/tar-split/tree/v0.11.3)
- github.com/veraison/go-cose: [v1.2.0](https://github.com/veraison/go-cose/tree/v1.2.0)
- github.com/xeipuuv/gojsonpointer: [02993c4](https://github.com/xeipuuv/gojsonpointer/tree/02993c4)
- github.com/xeipuuv/gojsonreference: [bd5ef7b](https://github.com/xeipuuv/gojsonreference/tree/bd5ef7b)
- github.com/yashtewari/glob-intersection: [v0.2.0](https://github.com/yashtewari/glob-intersection/tree/v0.2.0)
- go.starlark.net: a134d8f
- go.uber.org/mock: v0.4.0
- google.golang.org/grpc/cmd/protoc-gen-go-grpc: v1.5.1



# v1.32.0-alpha.1


## Downloads for v1.32.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes.tar.gz) | 86532c5440a87a6f6f0581cdddfdc68ea3f3f13a6478093518d8445c5ade8c448248de3f2102f29dc327f2055805a573cb60c36d7cce93605ed58b8b2ab23a5c
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-src.tar.gz) | 9cdce49ad47d92b14d88fbe0acdf67cce94dfd57f21d2a048ed46b370ff32f3b852ebbd1dfc646126cf30d20927d8e707500128c2ff193810ba7d7b68f612e94

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 742727920beab9ac9285ea98238be4e7a9099205ca95a52c930f2ebff2ded5617b13d5c861c4579c2316b3cb8398959ecb66c72f061724df6079d491c0f4fa5a
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 7bd4af634ccbf510d83a3468f288a3d91abf20146fd54e558324cb0dcaaa722a9e07f544699c2c73f033a5cf812cdfd9b8b36e3c612c0148792e1f8370a5d33e
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 39d34eca859b53fda63bda7df3ed45ba5e7e6cf406895d454da0291c6dd403139b4bfc46584595ddabaee890511df76d71252ebc1e1dda42f0ba941cec296cd9
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | f71a38447431dc7289caed55fd4846a4990247e4996c22b7c98aa9304959a5e25bf5aeb117d443481c411e6cc497051d8c75bde1ef3a7cb4ab8ff6f2abe43a39
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 21b75e8d69e98842704b2d1e468bbdaa62031d8570d35398095e6b7c96825af0276f668064722d6043788e7f2b8b0d093bbaed8fa93126f3e2d8720bc3fecf9b
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 498fc9962c02c60823832207f85ce919bb0c405b73feb931a7186babd644c928cee377c4ae0286f3e981328995d96586e4ae4783e38b879eb3caab8f9c9d0a5b
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 9bed5cf8bb05dc529f9ac7a637a657e1312065a2ee39c1d809f926b542547b8ddc674addae84cb523569a8a5a7f183a598b2d0566d9e58317bccd61558ca7192
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 6c5aa276aa65d969826ad49d901bc95fb7290cd00778c03f681ccdc12f3dc7cd77752e2895400250875a3c0a7548e20fe6f958bace1482f9a9b88c8581c10d95
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 5d45f1c1e0e984fa85ed99ac58dda6c475c3a2120a911425272187fde03b8017cdb14d71b2d6d9a23c946166fd2c374c42ffa32186c74546d7ea0146271cd50f
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | f0e3b6e845053c753640a46c3258eec96b04e7c95f044e8b980300ad32dadab2f0fef735213ba3de9b98dca2d7106a7f51e0f08c28a75cbe89f5a9f36f7e29a4
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 1a86995fc7284db06c23af66d82d836be36a6efcba7e2ef296c14bff56d39392a444cb399ce1f999181ec1ff7ac3edfdff84c3ccb63b0c6564550a8c0c948cef

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | dd0cfd5d57ad9c82ea52c98c80df8fe63a349bfbb16e42b30b1fe4c3b765327250397438e75e49014e6afffbaa7514daf830b8f7c781362241fb527196d8dc86
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | dbd29ab7bdfe97b8f9261cf3e727065f301bced78c866ead01d932de92e26476d3824c8f1023a8ebc63a63a3a79001dd2493c0f70118580841922b59ab1632c1
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | f37b92ed3ef9eeb3c40973068ef6131441abd6f4eabf1f1b4845f5774f116efbdf7d73f870f5268137d0ff4f406f443522f8adf63a043aaedcb67672246f0b55
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 58531d380dc3ddbff5b8e6e3cef8cc58f6c47aea0b4a3c907805836e35f571dc1e231e4dbbf635115bb70357408cf23ad68a86dd725a5abbe5025b2945cf1ddf

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 4273a6fc9fec18f408c0e559d3680270572250fc3d4c997439dfe844dca138a1a7277852882184601c4960a52525a6594b274f251bcca78df02104d296302e12
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 931eea6e9e6809a13a28519b03022bda056ac6215cd2b1bcd4186efa8204bc1b9245c3893292ad0ba823dc9cf008afd82dc4988cee2ea09eef3d5bb073945b1d
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | a35ed30cafb4aebb541d6a7a8d1995e773877cdda3e8b413a81eddc1eeb989b086765c6396df3d1d1dde86fb62ae7684401aa6dcedfcbe6940ada470549fe6e6
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | cc9b57d9fa7561d015288789cf7949dc7a68d4e6f006aa5b354941e736490b92480bd65f36090c53ddacde00f5a6a34b7a7a2b8c4912dfed3ec36e4c37759e9f
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.32.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | be118da99917ca00cff3f5ba9bb1a747c112c26522c4cc695d6cd2b2badfdf2ebcf79cb8885dbcf9986fc392510ec8a6c746cdf4ea7c984ed86a49f206ba68c2

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.32.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.32.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.32.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.32.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.32.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.32.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.31.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - ACTION REQUIRED for custom scheduler plugin developers:
  `PodEligibleToPreemptOthers` in the `preemption` interface gets `ctx` in the parameters.
  Please change your plugins' implementation accordingly. ([#126465](https://github.com/kubernetes/kubernetes/pull/126465), [@googs1025](https://github.com/googs1025)) [SIG Scheduling]
  - Changed NodeToStatusMap from map to struct and exposed methods to access the entries. Added absentNodesStatus, which inform what is the status of nodes that are absent in the map. 
  
  For developers of out-of-tree PostFilter plugins, make sure to update usage of NodeToStatusMap. Additionally, NodeToStatusMap should be eventually renamed to NodeToStatusReader. ([#126022](https://github.com/kubernetes/kubernetes/pull/126022), [@macsko](https://github.com/macsko)) [SIG Node, Scheduling and Testing]
 
## Changes by Kind

### Deprecation

- Reverted the `DisableNodeKubeProxyVersion` feature gate to default-off to give a full year from deprecation announcement in 1.29 to clearing the field by default, per the [Kubernetes deprecation policy](https://kubernetes.io/docs/reference/using-api/deprecation-policy/). ([#126720](https://github.com/kubernetes/kubernetes/pull/126720), [@liggitt](https://github.com/liggitt)) [SIG Architecture and Node]

### API Change

- Allow for Pod search domains to be a single dot "." or contain an underscore "_" ([#127167](https://github.com/kubernetes/kubernetes/pull/127167), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Apps, Network and Testing]
- Disallow `k8s.io` and `kubernetes.io` namespaced extra key in structured authentication configuration. ([#126553](https://github.com/kubernetes/kubernetes/pull/126553), [@aramase](https://github.com/aramase)) [SIG Auth]
- Fix the bug where spec.terminationGracePeriodSeconds of the pod will always be overwritten by the MaxPodGracePeriodSeconds of the soft eviction, you can enable the `AllowOverwriteTerminationGracePeriodSeconds` feature gate, which will restore the previous behavior.  If you do need to set this, please file an issue with the Kubernetes project to help contributors understand why you need it. ([#122890](https://github.com/kubernetes/kubernetes/pull/122890), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery, Architecture, Node and Testing]
- Kube-scheduler removed the following plugins: 
  - AzureDiskLimits 
  - CinderLimits
  - EBSLimits
  - GCEPDLimits
  Because the corresponding CSI driver reports how many volumes a node can handle in NodeGetInfoResponse, the kubelet stores this limit in CSINode and the scheduler then knows the driver's limit on the node.
  Remove plugins AzureDiskLimits, CinderLimits, EBSLimits and GCEPDLimits if you explicitly enabled them in the scheduler config. ([#124003](https://github.com/kubernetes/kubernetes/pull/124003), [@carlory](https://github.com/carlory)) [SIG Scheduling, Storage and Testing]
- Promoted `CustomResourceFieldSelectors` to stable; the feature is enabled by default. `--feature-gates=CustomResourceFieldSelectors=true` not needed on kube-apiserver binaries and will be removed in a future release. ([#127673](https://github.com/kubernetes/kubernetes/pull/127673), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]
- The default value for node-monitor-grace-period has been increased to 50s (earlier 40s) (Ref - https://github.com/kubernetes/kubernetes/issues/121793) ([#126287](https://github.com/kubernetes/kubernetes/pull/126287), [@devppratik](https://github.com/devppratik)) [SIG API Machinery, Apps and Node]
- The resource/v1alpha3.ResourceSliceList filed which should have been named "metadata" but was instead named "listMeta" is now properly "metadata". ([#126749](https://github.com/kubernetes/kubernetes/pull/126749), [@thockin](https://github.com/thockin)) [SIG API Machinery]
- The synthetic "Bookmark" event for the watch stream requests will now include a new annotation: `kubernetes.io/initial-events-list-blueprint`. THe annotation contains an empty, versioned list that is encoded in the requested format (such as protobuf, JSON, or CBOR), then base64-encoded and stored as a string. ([#127587](https://github.com/kubernetes/kubernetes/pull/127587), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- To enhance usability and developer experience, CRD validation rules now support direct use of (CEL) reserved keywords as field names in object validation expressions.
  Name format CEL library is supported in new expressions. ([#126977](https://github.com/kubernetes/kubernetes/pull/126977), [@aaron-prindle](https://github.com/aaron-prindle)) [SIG API Machinery, Architecture, Auth, Etcd, Instrumentation, Release, Scheduling and Testing]
- Updated incorrect description of persistentVolumeClaimRetentionPolicy ([#126545](https://github.com/kubernetes/kubernetes/pull/126545), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG API Machinery, Apps and CLI]
- X.509 client certificate authentication to kube-apiserver now produces credential IDs (derived from the certificate's signature) for use by audit logging. ([#125634](https://github.com/kubernetes/kubernetes/pull/125634), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Auth and Testing]

### Feature

- Added new functionality into the Go client code (`client-go`) library. The `List()` method for the metadata client allows enabling API streaming when fetching collections; this improves performance when listing many objects.
  To request this behaviour, your client software must enable the `WatchListClient` client-go feature gate. Additionally, streaming is only available if supported by the cluster; the API server that you connect to must also support streaming.
  If the API server does not support or allow streaming, then `client-go` falls back to fetching the collection using the **list** API verb. ([#127388](https://github.com/kubernetes/kubernetes/pull/127388), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Added preemptionPolicy field when using `kubectl get PriorityClass -owide` ([#126529](https://github.com/kubernetes/kubernetes/pull/126529), [@googs1025](https://github.com/googs1025)) [SIG CLI]
- Client-go/rest: contextual logging of request/response with accurate source code location of the caller ([#126999](https://github.com/kubernetes/kubernetes/pull/126999), [@pohly](https://github.com/pohly)) [SIG API Machinery and Instrumentation]
- Enabled kube-controller-manager '--concurrent-job-syncs' flag works on orphan Pod processors ([#126567](https://github.com/kubernetes/kubernetes/pull/126567), [@fusida](https://github.com/fusida)) [SIG Apps]
- Extend discovery GroupManager with Group lister interface ([#127524](https://github.com/kubernetes/kubernetes/pull/127524), [@mjudeikis](https://github.com/mjudeikis)) [SIG API Machinery]
- Fix kubectl doesn't print image volume when kubectl describe a pod with that volume ([#126706](https://github.com/kubernetes/kubernetes/pull/126706), [@carlory](https://github.com/carlory)) [SIG CLI]
- Graduate the AnonymousAuthConfigurableEndpoints feature gate to beta and enable by default to allow configurable endpoints for anonymous authentication. ([#127009](https://github.com/kubernetes/kubernetes/pull/127009), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Auth]
- Implement a queueing hint for PersistentVolumeClaim/Add event in CSILimit plugin. ([#124703](https://github.com/kubernetes/kubernetes/pull/124703), [@utam0k](https://github.com/utam0k)) [SIG Scheduling and Storage]
- Implement new cluster events UpdatePodSchedulingGatesEliminated and UpdatePodTolerations for scheduler plugins. ([#127083](https://github.com/kubernetes/kubernetes/pull/127083), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Improve Node QueueHint in the NodeAffinty plugin by ignoring unrelated changes that keep pods unschedulable. ([#127444](https://github.com/kubernetes/kubernetes/pull/127444), [@dom4ha](https://github.com/dom4ha)) [SIG Scheduling and Testing]
- Improve Node QueueHint in the NodeResource Fit plugin by ignoring unrelated changes that keep pods unschedulable. ([#127473](https://github.com/kubernetes/kubernetes/pull/127473), [@dom4ha](https://github.com/dom4ha)) [SIG Scheduling and Testing]
- Improve performance of the job controller when handling job delete events. ([#127378](https://github.com/kubernetes/kubernetes/pull/127378), [@hakuna-matatah](https://github.com/hakuna-matatah)) [SIG Apps]
- Improve performance of the job controller when handling job update events. ([#127228](https://github.com/kubernetes/kubernetes/pull/127228), [@hakuna-matatah](https://github.com/hakuna-matatah)) [SIG Apps]
- JWT authenticators now set the `jti` claim (if present and is a string value) as credential id for use by audit logging. ([#127010](https://github.com/kubernetes/kubernetes/pull/127010), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-apiserver: a new `--requestheader-uid-headers` flag allows configuring request header authentication to obtain the authenticating user's UID from the specified headers. The suggested value for the new option is `X-Remote-Uid`. When specified, the `kube-system/extension-apiserver-authentication` configmap will include the value in its `.data[requestheader-uid-headers]` field. ([#115834](https://github.com/kubernetes/kubernetes/pull/115834), [@stlaz](https://github.com/stlaz)) [SIG API Machinery, Auth, Cloud Provider and Testing]
- Kube-proxy uses  field-selector clusterIP!=None on Services to avoid watching for Headless Services, reduce  unnecessary network bandwidth ([#126769](https://github.com/kubernetes/kubernetes/pull/126769), [@Sakuralbj](https://github.com/Sakuralbj)) [SIG Network]
- Kubeadm: `kubeadm upgrade apply` now supports phase sub-command, user can use `kubeadm upgrade apply phase <phase-name>` to execute the specified phase, or use `kubeadm upgrade apply --skip-phases <phase-names>` to skip some phases during cluster upgrade. ([#126032](https://github.com/kubernetes/kubernetes/pull/126032), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: `kubeadm upgrade node` now supports `addon` and `post-upgrade` phases. User can use `kubeadm upgrade node phase addon` to execute the addon upgrade, or use `kubeadm upgrade node --skip-phases addon` to skip the addon upgrade. Currently, the `post-upgrade` phase is no-op, and it is mainly used to handle some release specific post-upgrade tasks. ([#127242](https://github.com/kubernetes/kubernetes/pull/127242), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: add a validation warning when the certificateValidityPeriod is more than the caCertificateValidityPeriod ([#126538](https://github.com/kubernetes/kubernetes/pull/126538), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: allow mixing the flag --config with the special flag --print-manifest of the subphases of 'kubeadm init phase addon'. ([#126740](https://github.com/kubernetes/kubernetes/pull/126740), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: if an unknown command name is passed to any parent command such as 'kubeadm init phase' return an error. If 'kubeadm init phase' or another command that has subcommands is called without subcommand name, print the available commands and also return an error. ([#127096](https://github.com/kubernetes/kubernetes/pull/127096), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: promoted feature gate `EtcdLearnerMode` to GA. Learner mode in etcd deployed by kubeadm is now locked to enabled by default. ([#126374](https://github.com/kubernetes/kubernetes/pull/126374), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubelet: add log and event for cgroup v2 with kernel older than 5.8. ([#126595](https://github.com/kubernetes/kubernetes/pull/126595), [@pacoxu](https://github.com/pacoxu)) [SIG Node]
- Kubernetes is now built with go 1.23.0 ([#127076](https://github.com/kubernetes/kubernetes/pull/127076), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Promoted `RetryGenerateName` to stable; the feature is enabled by default. `--feature-gates=RetryGenerateName=true` not needed on kube-apiserver binaries and will be removed in a future release. ([#127093](https://github.com/kubernetes/kubernetes/pull/127093), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Support inflight_events metric in the scheduler for QueueingHint (alpha feature). ([#127052](https://github.com/kubernetes/kubernetes/pull/127052), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Support specifying a custom network parameter when running e2e-node-tests with the remote option. ([#127574](https://github.com/kubernetes/kubernetes/pull/127574), [@bouaouda-achraf](https://github.com/bouaouda-achraf)) [SIG Node and Testing]
- The scheduler retries gated Pods more appropriately, giving them a backoff penalty too. ([#126029](https://github.com/kubernetes/kubernetes/pull/126029), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Transformation_operations_total metric will have additional resource label which can be used for resource specific validations for example handling of encryption config by the apiserver. ([#126512](https://github.com/kubernetes/kubernetes/pull/126512), [@kmala](https://github.com/kmala)) [SIG API Machinery, Auth, Etcd and Testing]
- Unallowed label values will show up as "unexpected" in scheduler metrics ([#126762](https://github.com/kubernetes/kubernetes/pull/126762), [@richabanker](https://github.com/richabanker)) [SIG Instrumentation and Scheduling]
- When SchedulerQueueingHint is enabled,
  the scheduler's in-tree plugins now subscribe to specific node events to decide whether to requeue Pods.
  This allows the scheduler to handle cluster events faster with less memory.
  
  Specific node events include updates to taints, tolerations or allocatable.
  In-tree plugins now ignore node updates that don't modify any of these fields. ([#127220](https://github.com/kubernetes/kubernetes/pull/127220), [@sanposhiho](https://github.com/sanposhiho)) [SIG Node, Scheduling and Storage]
- When SchedulerQueueingHints is enabled, clear events cached in the scheduling queue as soon as possible so that the scheduler consumes less memory. ([#120586](https://github.com/kubernetes/kubernetes/pull/120586), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]

### Documentation

- Clarified the kube-controller-manager documentation for --allocate-node-cidrs, --cluster-cidr, and --service-cluster-ip-range flags to accurately reflect their dependencies and usage conditions. ([#126784](https://github.com/kubernetes/kubernetes/pull/126784), [@eminwux](https://github.com/eminwux)) [SIG API Machinery, Cloud Provider and Docs]
- Documented the `--for=create` option to `kubectl wait` ([#127327](https://github.com/kubernetes/kubernetes/pull/127327), [@ryanwinter](https://github.com/ryanwinter)) [SIG CLI]

### Failing Test

- Kubelet Plugins are now re-registered properly  on Windows if the re-registration period is < 15ms. ([#114136](https://github.com/kubernetes/kubernetes/pull/114136), [@claudiubelu](https://github.com/claudiubelu)) [SIG Node, Storage, Testing and Windows]

### Bug or Regression

- API emulation versioning honors cohabitating resources ([#127239](https://github.com/kubernetes/kubernetes/pull/127239), [@xuzhenglun](https://github.com/xuzhenglun)) [SIG API Machinery]
- Apiserver repair controller is resilient to etcd errors during bootstrap and retries during 30 seconds before failing. ([#126671](https://github.com/kubernetes/kubernetes/pull/126671), [@fusida](https://github.com/fusida)) [SIG Network]
- Applyconfiguration-gen no longer generates duplicate methods and ambiguous member accesses when types end up with multiple members of the same name (through embedded structs). ([#127001](https://github.com/kubernetes/kubernetes/pull/127001), [@skitt](https://github.com/skitt)) [SIG API Machinery]
- DRA: when a DRA driver was started after creating pods which need resources from that driver, no additional attempt was made to schedule such unschedulable pods again. Only affected DRA with structured parameters. ([#126807](https://github.com/kubernetes/kubernetes/pull/126807), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- DRA: when enabling the scheduler queuing hint feature, pods got stuck as unschedulable for a while unnecessarily because recording the name of the generated ResourceClaim did not trigger scheduling. ([#127497](https://github.com/kubernetes/kubernetes/pull/127497), [@pohly](https://github.com/pohly)) [SIG Auth, Node, Scheduling and Testing]
- Discarded the output streams of destination path check in kubectl cp when copying from local to pod and added a 3 seconds timeout to this check ([#126652](https://github.com/kubernetes/kubernetes/pull/126652), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Fix CEL estimated cost of expressions that perform equality checks of IPs, CIDRs, Quantities, Formats and URLs. ([#126359](https://github.com/kubernetes/kubernetes/pull/126359), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fix a bug on the endpoints controller that does not reconcile the Endpoint object after this is truncated (it gets more than 1000 endpoints addresses) ([#127417](https://github.com/kubernetes/kubernetes/pull/127417), [@aojea](https://github.com/aojea)) [SIG Apps, Network and Testing]
- Fix a bug when the hostname label of a node does not match the node name, pods bound to a PV with nodeAffinity using the hostname may be scheduled to the wrong node or experience scheduling failures. ([#125398](https://github.com/kubernetes/kubernetes/pull/125398), [@AxeZhan](https://github.com/AxeZhan)) [SIG Scheduling and Storage]
- Fix a bug with dual stack clusters using the beta feature MultiCIDRServiceAllocator can not create dual stack Services or Services with IPs on the secondary range. User that want to use this feature in 1.30 with dual stack clusters can workaround the issue by setting the feature gate DisableAllocatorDualWrite to true ([#127598](https://github.com/kubernetes/kubernetes/pull/127598), [@aojea](https://github.com/aojea)) [SIG Network and Testing]
- Fix a potential memory leak in QueueingHint (alpha feature) ([#127016](https://github.com/kubernetes/kubernetes/pull/127016), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Fix a scheduler preemption issue where the victim pod was not deleted due to incorrect status patching. This issue occurred when the preemptor and victim pods had different QoS classes in their status, causing the preemption to fail entirely. ([#126644](https://github.com/kubernetes/kubernetes/pull/126644), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- Fix fake client to accept request without metadata.name to better emulate behavior of actual client. ([#126727](https://github.com/kubernetes/kubernetes/pull/126727), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fix race condition in kube-proxy initialization that could blackhole UDP traffic to service VIP. ([#126532](https://github.com/kubernetes/kubernetes/pull/126532), [@wedaly](https://github.com/wedaly)) [SIG Network]
- Fix the wrong hierarchical structure for the child span and the parent span (i.e. `SerializeObject` and `List`). In the past, some children's spans appeared parallel to their parents. ([#127551](https://github.com/kubernetes/kubernetes/pull/127551), [@carlory](https://github.com/carlory)) [SIG API Machinery and Instrumentation]
- Fixed a bug where init containers may fail to start due to a temporary container runtime failure. ([#126543](https://github.com/kubernetes/kubernetes/pull/126543), [@gjkim42](https://github.com/gjkim42)) [SIG Node]
- Fixed a bug which the scheduler didn't correctly tell plugins Node deletion.
  This bug could impact all scheduler plugins subscribing to Node/Delete event, making the queue keep the Pods rejected by those plugins incorrectly at Node deletion. Among the in-tree plugins, PodTopologySpread is the only victim. ([#127464](https://github.com/kubernetes/kubernetes/pull/127464), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Fixed a possible memory leak for QueueingHint (alpha feature) ([#126962](https://github.com/kubernetes/kubernetes/pull/126962), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Fixed a regression in 1.29+ default configurations, where regular init containers may fail to start due to a temporary container runtime failure. ([#127162](https://github.com/kubernetes/kubernetes/pull/127162), [@gjkim42](https://github.com/gjkim42)) [SIG Node]
- Fixed an issue where requests sent by the KMSv2 service would be rejected due to having an invalid authority header. ([#126930](https://github.com/kubernetes/kubernetes/pull/126930), [@Ruddickmg](https://github.com/Ruddickmg)) [SIG API Machinery and Auth]
- Fixed: dynamic client-go can now handle subresources with an UnstructuredList response ([#126809](https://github.com/kubernetes/kubernetes/pull/126809), [@ryantxu](https://github.com/ryantxu)) [SIG API Machinery]
- Fixes a bug in the garbage collector controller which could block indefinitely on a cache sync failure. This fix allows the garbage collector to eventually continue garbage collecting other resources if a given resource cannot be listed or watched. Any objects in the unsynced resource type with owner references with `blockOwnerDeletion: true` will not be known to the garbage collector. Use of `blockOwnerDeletion` has always been best-effort and racy on startup and object creation, with this fix, it continues to be best-effort for resources that cannot be synced by the garbage collector controller. ([#125796](https://github.com/kubernetes/kubernetes/pull/125796), [@haorenfsa](https://github.com/haorenfsa)) [SIG API Machinery, Apps and Testing]
- Fixes a bug where restartable and non-restartable init containers were not accounted for in the message and annotations of eviction event. ([#124947](https://github.com/kubernetes/kubernetes/pull/124947), [@toVersus](https://github.com/toVersus)) [SIG Node]
- Fixes the ability to set the `resolvConf` option in drop-in kubelet configuration files, validates that drop-in kubelet configuration files are in a supported version. ([#127421](https://github.com/kubernetes/kubernetes/pull/127421), [@liggitt](https://github.com/liggitt)) [SIG Node]
- Fixes the bug in NodeUnschedulable that only happens with QHint enabled, 
  which the scheduler might miss some updates for the Pods rejected by NodeUnschedulable plugin and put the Pods in the queue for a longer time than needed. ([#127427](https://github.com/kubernetes/kubernetes/pull/127427), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Fixes the bug in PodTopologySpread that only happens with QHint enabled, 
  which the scheduler might miss some updates for the Pods rejected by PodTopologySpread plugin and put the Pods in the queue for a longer time than needed. ([#127447](https://github.com/kubernetes/kubernetes/pull/127447), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- HostNetwork pods no longer depend on the PodIPs to be assigned to configure the defined hostAliases on the Pod ([#126460](https://github.com/kubernetes/kubernetes/pull/126460), [@aojea](https://github.com/aojea)) [SIG Network, Node and Testing]
- If a client makes an API streaming requests and specifies an `application/json;as=Table` content type, the API server now responds with a 406 (Not Acceptable) error.
  This change helps to ensure that unsupported formats, such as `Table` representations are correctly rejected. ([#126996](https://github.com/kubernetes/kubernetes/pull/126996), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- If an old pod spec has used image volume source, we must allow it when updating the resource even if the feature-gate ImageVolume is disabled. ([#126733](https://github.com/kubernetes/kubernetes/pull/126733), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps and Node]
- Improve PVC Protection Controller's scalability by batch-processing PVCs by namespace with lazy live pod listing. ([#125372](https://github.com/kubernetes/kubernetes/pull/125372), [@hungnguyen243](https://github.com/hungnguyen243)) [SIG Apps, Node, Storage and Testing]
- Improve PVC Protection Controller's scalability by batch-processing PVCs by namespace with lazy live pod listing. ([#126745](https://github.com/kubernetes/kubernetes/pull/126745), [@hungnguyen243](https://github.com/hungnguyen243)) [SIG Apps, Storage and Testing]
- Kube-apiserver: Fixes a 1.31 regression that stopped honoring build ID overrides with the --version flag ([#126665](https://github.com/kubernetes/kubernetes/pull/126665), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kubeadm: ensure that Pods from the upgrade preflight check `CreateJob` are properly terminated after a timeout. ([#127333](https://github.com/kubernetes/kubernetes/pull/127333), [@yuyabee](https://github.com/yuyabee)) [SIG Cluster Lifecycle]
- Kubeadm: when adding new control plane nodes with "kubeamd join", ensure that the etcd member addition is performed only if a given member URL does not already exist in the list of members. Similarly, on "kubeadm reset" only remove an etcd member if its ID exists. ([#127491](https://github.com/kubernetes/kubernetes/pull/127491), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet now attempts to get an existing node if the request to create it fails with StatusForbidden. ([#126318](https://github.com/kubernetes/kubernetes/pull/126318), [@hoskeri](https://github.com/hoskeri)) [SIG Node]
- Kubelet: use the CRI stats provider if `PodAndContainerStatsFromCRI` feature is enabled ([#126488](https://github.com/kubernetes/kubernetes/pull/126488), [@haircommander](https://github.com/haircommander)) [SIG Node]
- Removed unneeded permissions for system:controller:persistent-volume-binder and system:controller:expand-controller clusterroles ([#125995](https://github.com/kubernetes/kubernetes/pull/125995), [@carlory](https://github.com/carlory)) [SIG Auth and Storage]
- Revert "fix: handle socket file detection on Windows" ([#126976](https://github.com/kubernetes/kubernetes/pull/126976), [@jsturtevant](https://github.com/jsturtevant)) [SIG Node]
- Send an error on `ResultChan` and close the `RetryWatcher` when the client is forbidden or unauthorized from watching the resource. ([#126038](https://github.com/kubernetes/kubernetes/pull/126038), [@mprahl](https://github.com/mprahl)) [SIG API Machinery]
- Send bookmark right now after sending all items in watchCache store ([#127012](https://github.com/kubernetes/kubernetes/pull/127012), [@Chaunceyctx](https://github.com/Chaunceyctx)) [SIG API Machinery]
- Terminated Pods on a node will not be re-admitted on kubelet restart. This fixes the problem of Completed Pods awaiting for the finalizer marked as Failed after the kubelet restart. ([#126343](https://github.com/kubernetes/kubernetes/pull/126343), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node and Testing]
- The CSI volume plugin stopped watching the VolumeAttachment object if the object is not found or the volume is not attached when kubelet waits for a volume attached. In the past, it would fail due to missing permission. ([#126961](https://github.com/kubernetes/kubernetes/pull/126961), [@carlory](https://github.com/carlory)) [SIG Storage]
- The Usage and VolumeCondition are both optional in the response and if CSIVolumeHealth feature gate is enabled kubelet needs to consider returning metrics if either one is set. ([#127021](https://github.com/kubernetes/kubernetes/pull/127021), [@Madhu-1](https://github.com/Madhu-1)) [SIG Storage]
- Upgrade coreDNS to v1.11.3 ([#126449](https://github.com/kubernetes/kubernetes/pull/126449), [@BenTheElder](https://github.com/BenTheElder)) [SIG Cloud Provider and Cluster Lifecycle]
- Use allocatedResources on PVC for node expansion in kubelet ([#126600](https://github.com/kubernetes/kubernetes/pull/126600), [@gnufied](https://github.com/gnufied)) [SIG Node, Storage and Testing]
- When entering a value other than "external" to the "--cloud-provider" flag for the kubelet, kube-controller-manager, and kube-apiserver, the user will now receive a warning in the logs about the disablement of internal cloud providers, this is in contrast to the previous warnings about deprecation. ([#127711](https://github.com/kubernetes/kubernetes/pull/127711), [@elmiko](https://github.com/elmiko)) [SIG API Machinery, Cloud Provider and Node]

### Other (Cleanup or Flake)

- Added an example for kubectl delete with the --interactive flag. ([#127512](https://github.com/kubernetes/kubernetes/pull/127512), [@bergerhoffer](https://github.com/bergerhoffer)) [SIG CLI]
- Aggregated Discovery v2beta1 fixture is removed in `./api/discovery`. Please use v2 ([#127008](https://github.com/kubernetes/kubernetes/pull/127008), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- Device manager: stop using annotations to pass CDI device info to runtimes. Containerd versions older than v1.7.2 don't support passing CDI info through CRI and need to be upgraded. ([#126435](https://github.com/kubernetes/kubernetes/pull/126435), [@bart0sh](https://github.com/bart0sh)) [SIG Node]
- Feature gate "AllowServiceLBStatusOnNonLB" has been removed.  This gate has been stable and unchanged for over a year. ([#126786](https://github.com/kubernetes/kubernetes/pull/126786), [@thockin](https://github.com/thockin)) [SIG Apps]
- Fix a warning message about the gce in-tree cloud provider state ([#126773](https://github.com/kubernetes/kubernetes/pull/126773), [@carlory](https://github.com/carlory)) [SIG Cloud Provider]
- Kube-proxy initialization waits for all pre-sync events from node and serviceCIDR informers to be delivered. ([#126561](https://github.com/kubernetes/kubernetes/pull/126561), [@wedaly](https://github.com/wedaly)) [SIG Network]
- Kube-proxy will no longer depend on conntrack binary for stale UDP connections cleanup ([#126847](https://github.com/kubernetes/kubernetes/pull/126847), [@aroradaman](https://github.com/aroradaman)) [SIG Cluster Lifecycle, Network and Testing]
- Kubeadm: don't warn if `crictl` binary does not exist since kubeadm does not rely on `crictl` since v1.31. ([#126596](https://github.com/kubernetes/kubernetes/pull/126596), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cluster Lifecycle]
- Kubeadm: make sure the extra environment variables written to a kubeadm managed PodSpec are sorted alpha-numerically by the environment variable name. ([#126743](https://github.com/kubernetes/kubernetes/pull/126743), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated sub-phase of 'init kubelet-finilize' called `experimental-cert-rotation`, and use 'enable-client-cert-rotation' instead. ([#126913](https://github.com/kubernetes/kubernetes/pull/126913), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: removed `socat` and `ebtables` from kubeadm preflight checks ([#127151](https://github.com/kubernetes/kubernetes/pull/127151), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cluster Lifecycle]
- Kubeadm: removed the deprecated and NO-OP flags `--features-gates` for `kubeadm upgrde apply` and `--api-server-manfiest`, `--controller-manager-manfiest` and `--scheduler-manifest` for `kubeadm upgrade diff`. ([#127123](https://github.com/kubernetes/kubernetes/pull/127123), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: removed the deprecated flag '--experimental-output', please use the flag '--output' instead that serves the same purpose. Affected commands are - "kubeadm config images list", "kubeadm token list", "kubeadm upgade plan", "kubeadm certs check-expiration". ([#126914](https://github.com/kubernetes/kubernetes/pull/126914), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: switched the kube-scheduler static Pod to use the endpoints /livez (for startup and liveness probes) and /readyz (for the readiness probe). Previously /healthz was used for all probes, which is deprecated behavior in the scope of this component. ([#126945](https://github.com/kubernetes/kubernetes/pull/126945), [@liangyuanpeng](https://github.com/liangyuanpeng)) [SIG Cluster Lifecycle]
- Optimize code, filter podUID is empty string when call this `getPodAndContainerForDevice` method. ([#126997](https://github.com/kubernetes/kubernetes/pull/126997), [@lengrongfu](https://github.com/lengrongfu)) [SIG Node]
- Remove GAed feature gates ServerSideApply/ServerSideFieldValidation ([#127058](https://github.com/kubernetes/kubernetes/pull/127058), [@carlory](https://github.com/carlory)) [SIG API Machinery]
- Removed feature gate `ValiatingAdmissionPolicy`. ([#126645](https://github.com/kubernetes/kubernetes/pull/126645), [@cici37](https://github.com/cici37)) [SIG API Machinery, Auth and Testing]
- Removed generally available feature gate `CloudDualStackNodeIPs`. ([#126840](https://github.com/kubernetes/kubernetes/pull/126840), [@carlory](https://github.com/carlory)) [SIG API Machinery and Cloud Provider]
- Removed generally available feature gate `LegacyServiceAccountTokenCleanUp`. ([#126839](https://github.com/kubernetes/kubernetes/pull/126839), [@carlory](https://github.com/carlory)) [SIG Auth]
- Removed generally available feature gate `MinDomainsInPodTopologySpread` ([#126863](https://github.com/kubernetes/kubernetes/pull/126863), [@carlory](https://github.com/carlory)) [SIG Scheduling]
- Removed generally available feature gate `NewVolumeManagerReconstruction`. ([#126775](https://github.com/kubernetes/kubernetes/pull/126775), [@carlory](https://github.com/carlory)) [SIG Node and Storage]
- Removed generally available feature gate `NodeOutOfServiceVolumeDetach` ([#127019](https://github.com/kubernetes/kubernetes/pull/127019), [@carlory](https://github.com/carlory)) [SIG Apps and Testing]
- Removed generally available feature gate `StableLoadBalancerNodeSet`. ([#126841](https://github.com/kubernetes/kubernetes/pull/126841), [@carlory](https://github.com/carlory)) [SIG API Machinery, Cloud Provider and Network]
- Removed the `KMSv2` and `KMSv2KDF` feature gates. The associated features graduated to stable in the Kubernetes v1.29 release. ([#126698](https://github.com/kubernetes/kubernetes/pull/126698), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Short circuit if the compaction request from apiserver is disabled. ([#126627](https://github.com/kubernetes/kubernetes/pull/126627), [@fusida](https://github.com/fusida)) [SIG Etcd]
- Show a warning message to inform users that the `legacy` profile is planned to be deprecated. ([#127230](https://github.com/kubernetes/kubernetes/pull/127230), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI]
- The `flowcontrol.apiserver.k8s.io/v1beta3` API version of `FlowSchema` and `PriorityLevelConfiguration` is no longer served in v1.32. Migrate manifests and API clients to use the `flowcontrol.apiserver.k8s.io/v1` API version, available since v1.29. More information is at https://kubernetes.io/docs/reference/using-api/deprecation-guide/#flowcontrol-resources-v132 ([#127017](https://github.com/kubernetes/kubernetes/pull/127017), [@carlory](https://github.com/carlory)) [SIG API Machinery and Testing]
- The kube-proxy command line flags `--healthz-port` and `--metrics-port`, which were previously deprecated, have now been removed. ([#126889](https://github.com/kubernetes/kubernetes/pull/126889), [@aroradaman](https://github.com/aroradaman)) [SIG Network and Windows]
- The percentage display in kubectl top node is changed from % -> (%) ([#126995](https://github.com/kubernetes/kubernetes/pull/126995), [@googs1025](https://github.com/googs1025)) [SIG CLI]
- Update github.com/coredns/corefile-migration to v1.0.24 ([#126851](https://github.com/kubernetes/kubernetes/pull/126851), [@BenTheElder](https://github.com/BenTheElder)) [SIG Architecture and Cluster Lifecycle]
- Updated cni-plugins to [v1.5.1](https://github.com/containernetworking/plugins/releases/tag/v1.5.1). ([#126966](https://github.com/kubernetes/kubernetes/pull/126966), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.31.0. ([#126590](https://github.com/kubernetes/kubernetes/pull/126590), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- Upgrade etcd client to v3.5.16 ([#127279](https://github.com/kubernetes/kubernetes/pull/127279), [@serathius](https://github.com/serathius)) [SIG API Machinery, Auth, Cloud Provider and Node]

## Dependencies

### Added
- github.com/Microsoft/cosesign1go: [v1.1.0](https://github.com/Microsoft/cosesign1go/tree/v1.1.0)
- github.com/Microsoft/didx509go: [v0.0.3](https://github.com/Microsoft/didx509go/tree/v0.0.3)
- github.com/agnivade/levenshtein: [v1.1.1](https://github.com/agnivade/levenshtein/tree/v1.1.1)
- github.com/akavel/rsrc: [v0.10.2](https://github.com/akavel/rsrc/tree/v0.10.2)
- github.com/aws/aws-sdk-go-v2/config: [v1.27.24](https://github.com/aws/aws-sdk-go-v2/tree/config/v1.27.24)
- github.com/aws/aws-sdk-go-v2/credentials: [v1.17.24](https://github.com/aws/aws-sdk-go-v2/tree/credentials/v1.17.24)
- github.com/aws/aws-sdk-go-v2/feature/ec2/imds: [v1.16.9](https://github.com/aws/aws-sdk-go-v2/tree/feature/ec2/imds/v1.16.9)
- github.com/aws/aws-sdk-go-v2/internal/configsources: [v1.3.13](https://github.com/aws/aws-sdk-go-v2/tree/internal/configsources/v1.3.13)
- github.com/aws/aws-sdk-go-v2/internal/endpoints/v2: [v2.6.13](https://github.com/aws/aws-sdk-go-v2/tree/internal/endpoints/v2/v2.6.13)
- github.com/aws/aws-sdk-go-v2/internal/ini: [v1.8.0](https://github.com/aws/aws-sdk-go-v2/tree/internal/ini/v1.8.0)
- github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding: [v1.11.3](https://github.com/aws/aws-sdk-go-v2/tree/service/internal/accept-encoding/v1.11.3)
- github.com/aws/aws-sdk-go-v2/service/internal/presigned-url: [v1.11.15](https://github.com/aws/aws-sdk-go-v2/tree/service/internal/presigned-url/v1.11.15)
- github.com/aws/aws-sdk-go-v2/service/sso: [v1.22.1](https://github.com/aws/aws-sdk-go-v2/tree/service/sso/v1.22.1)
- github.com/aws/aws-sdk-go-v2/service/ssooidc: [v1.26.2](https://github.com/aws/aws-sdk-go-v2/tree/service/ssooidc/v1.26.2)
- github.com/aws/aws-sdk-go-v2/service/sts: [v1.30.1](https://github.com/aws/aws-sdk-go-v2/tree/service/sts/v1.30.1)
- github.com/aws/aws-sdk-go-v2: [v1.30.1](https://github.com/aws/aws-sdk-go-v2/tree/v1.30.1)
- github.com/aws/smithy-go: [v1.20.3](https://github.com/aws/smithy-go/tree/v1.20.3)
- github.com/containerd/cgroups/v3: [v3.0.3](https://github.com/containerd/cgroups/tree/v3.0.3)
- github.com/containerd/containerd/api: [v1.7.19](https://github.com/containerd/containerd/tree/api/v1.7.19)
- github.com/containerd/errdefs: [v0.1.0](https://github.com/containerd/errdefs/tree/v0.1.0)
- github.com/containerd/log: [v0.1.0](https://github.com/containerd/log/tree/v0.1.0)
- github.com/containerd/protobuild: [v0.3.0](https://github.com/containerd/protobuild/tree/v0.3.0)
- github.com/containerd/stargz-snapshotter/estargz: [v0.14.3](https://github.com/containerd/stargz-snapshotter/tree/estargz/v0.14.3)
- github.com/containerd/typeurl/v2: [v2.2.0](https://github.com/containerd/typeurl/tree/v2.2.0)
- github.com/decred/dcrd/dcrec/secp256k1/v4: [v4.2.0](https://github.com/decred/dcrd/tree/dcrec/secp256k1/v4/v4.2.0)
- github.com/docker/cli: [v24.0.0+incompatible](https://github.com/docker/cli/tree/v24.0.0)
- github.com/docker/docker-credential-helpers: [v0.7.0](https://github.com/docker/docker-credential-helpers/tree/v0.7.0)
- github.com/docker/go-events: [e31b211](https://github.com/docker/go-events/tree/e31b211)
- github.com/go-ini/ini: [v1.67.0](https://github.com/go-ini/ini/tree/v1.67.0)
- github.com/gobwas/glob: [v0.2.3](https://github.com/gobwas/glob/tree/v0.2.3)
- github.com/goccy/go-json: [v0.10.2](https://github.com/goccy/go-json/tree/v0.10.2)
- github.com/google/go-containerregistry: [v0.20.1](https://github.com/google/go-containerregistry/tree/v0.20.1)
- github.com/gorilla/mux: [v1.8.1](https://github.com/gorilla/mux/tree/v1.8.1)
- github.com/josephspurrier/goversioninfo: [v1.4.0](https://github.com/josephspurrier/goversioninfo/tree/v1.4.0)
- github.com/klauspost/compress: [v1.17.0](https://github.com/klauspost/compress/tree/v1.17.0)
- github.com/lestrrat-go/backoff/v2: [v2.0.8](https://github.com/lestrrat-go/backoff/tree/v2.0.8)
- github.com/lestrrat-go/blackmagic: [v1.0.2](https://github.com/lestrrat-go/blackmagic/tree/v1.0.2)
- github.com/lestrrat-go/httpcc: [v1.0.1](https://github.com/lestrrat-go/httpcc/tree/v1.0.1)
- github.com/lestrrat-go/iter: [v1.0.2](https://github.com/lestrrat-go/iter/tree/v1.0.2)
- github.com/lestrrat-go/jwx: [v1.2.28](https://github.com/lestrrat-go/jwx/tree/v1.2.28)
- github.com/lestrrat-go/option: [v1.0.1](https://github.com/lestrrat-go/option/tree/v1.0.1)
- github.com/linuxkit/virtsock: [f8cee7d](https://github.com/linuxkit/virtsock/tree/f8cee7d)
- github.com/mattn/go-shellwords: [v1.0.12](https://github.com/mattn/go-shellwords/tree/v1.0.12)
- github.com/moby/docker-image-spec: [v1.3.1](https://github.com/moby/docker-image-spec/tree/v1.3.1)
- github.com/moby/sys/sequential: [v0.5.0](https://github.com/moby/sys/tree/sequential/v0.5.0)
- github.com/open-policy-agent/opa: [v0.67.1](https://github.com/open-policy-agent/opa/tree/v0.67.1)
- github.com/rcrowley/go-metrics: [10cdbea](https://github.com/rcrowley/go-metrics/tree/10cdbea)
- github.com/tchap/go-patricia/v2: [v2.3.1](https://github.com/tchap/go-patricia/tree/v2.3.1)
- github.com/vbatts/tar-split: [v0.11.3](https://github.com/vbatts/tar-split/tree/v0.11.3)
- github.com/veraison/go-cose: [v1.2.0](https://github.com/veraison/go-cose/tree/v1.2.0)
- github.com/xeipuuv/gojsonpointer: [02993c4](https://github.com/xeipuuv/gojsonpointer/tree/02993c4)
- github.com/xeipuuv/gojsonreference: [bd5ef7b](https://github.com/xeipuuv/gojsonreference/tree/bd5ef7b)
- github.com/yashtewari/glob-intersection: [v0.2.0](https://github.com/yashtewari/glob-intersection/tree/v0.2.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp: v1.27.0
- go.uber.org/mock: v0.4.0
- google.golang.org/grpc/cmd/protoc-gen-go-grpc: v1.5.1

### Changed
- cloud.google.com/go/accessapproval: v1.7.1 → v1.7.4
- cloud.google.com/go/accesscontextmanager: v1.8.1 → v1.8.4
- cloud.google.com/go/aiplatform: v1.48.0 → v1.58.0
- cloud.google.com/go/analytics: v0.21.3 → v0.22.0
- cloud.google.com/go/apigateway: v1.6.1 → v1.6.4
- cloud.google.com/go/apigeeconnect: v1.6.1 → v1.6.4
- cloud.google.com/go/apigeeregistry: v0.7.1 → v0.8.2
- cloud.google.com/go/appengine: v1.8.1 → v1.8.4
- cloud.google.com/go/area120: v0.8.1 → v0.8.4
- cloud.google.com/go/artifactregistry: v1.14.1 → v1.14.6
- cloud.google.com/go/asset: v1.14.1 → v1.17.0
- cloud.google.com/go/assuredworkloads: v1.11.1 → v1.11.4
- cloud.google.com/go/automl: v1.13.1 → v1.13.4
- cloud.google.com/go/baremetalsolution: v1.1.1 → v1.2.3
- cloud.google.com/go/batch: v1.3.1 → v1.7.0
- cloud.google.com/go/beyondcorp: v1.0.0 → v1.0.3
- cloud.google.com/go/bigquery: v1.53.0 → v1.58.0
- cloud.google.com/go/billing: v1.16.0 → v1.18.0
- cloud.google.com/go/binaryauthorization: v1.6.1 → v1.8.0
- cloud.google.com/go/certificatemanager: v1.7.1 → v1.7.4
- cloud.google.com/go/channel: v1.16.0 → v1.17.4
- cloud.google.com/go/cloudbuild: v1.13.0 → v1.15.0
- cloud.google.com/go/clouddms: v1.6.1 → v1.7.3
- cloud.google.com/go/cloudtasks: v1.12.1 → v1.12.4
- cloud.google.com/go/compute: v1.23.0 → v1.25.1
- cloud.google.com/go/contactcenterinsights: v1.10.0 → v1.12.1
- cloud.google.com/go/container: v1.24.0 → v1.29.0
- cloud.google.com/go/containeranalysis: v0.10.1 → v0.11.3
- cloud.google.com/go/datacatalog: v1.16.0 → v1.19.2
- cloud.google.com/go/dataflow: v0.9.1 → v0.9.4
- cloud.google.com/go/dataform: v0.8.1 → v0.9.1
- cloud.google.com/go/datafusion: v1.7.1 → v1.7.4
- cloud.google.com/go/datalabeling: v0.8.1 → v0.8.4
- cloud.google.com/go/dataplex: v1.9.0 → v1.14.0
- cloud.google.com/go/dataproc/v2: v2.0.1 → v2.3.0
- cloud.google.com/go/dataqna: v0.8.1 → v0.8.4
- cloud.google.com/go/datastore: v1.13.0 → v1.15.0
- cloud.google.com/go/datastream: v1.10.0 → v1.10.3
- cloud.google.com/go/deploy: v1.13.0 → v1.17.0
- cloud.google.com/go/dialogflow: v1.40.0 → v1.48.1
- cloud.google.com/go/dlp: v1.10.1 → v1.11.1
- cloud.google.com/go/documentai: v1.22.0 → v1.23.7
- cloud.google.com/go/domains: v0.9.1 → v0.9.4
- cloud.google.com/go/edgecontainer: v1.1.1 → v1.1.4
- cloud.google.com/go/essentialcontacts: v1.6.2 → v1.6.5
- cloud.google.com/go/eventarc: v1.13.0 → v1.13.3
- cloud.google.com/go/filestore: v1.7.1 → v1.8.0
- cloud.google.com/go/firestore: v1.12.0 → v1.14.0
- cloud.google.com/go/functions: v1.15.1 → v1.15.4
- cloud.google.com/go/gkebackup: v1.3.0 → v1.3.4
- cloud.google.com/go/gkeconnect: v0.8.1 → v0.8.4
- cloud.google.com/go/gkehub: v0.14.1 → v0.14.4
- cloud.google.com/go/gkemulticloud: v1.0.0 → v1.1.0
- cloud.google.com/go/gsuiteaddons: v1.6.1 → v1.6.4
- cloud.google.com/go/iam: v1.1.1 → v1.1.5
- cloud.google.com/go/iap: v1.8.1 → v1.9.3
- cloud.google.com/go/ids: v1.4.1 → v1.4.4
- cloud.google.com/go/iot: v1.7.1 → v1.7.4
- cloud.google.com/go/kms: v1.15.0 → v1.15.5
- cloud.google.com/go/language: v1.10.1 → v1.12.2
- cloud.google.com/go/lifesciences: v0.9.1 → v0.9.4
- cloud.google.com/go/logging: v1.7.0 → v1.9.0
- cloud.google.com/go/longrunning: v0.5.1 → v0.5.4
- cloud.google.com/go/managedidentities: v1.6.1 → v1.6.4
- cloud.google.com/go/maps: v1.4.0 → v1.6.3
- cloud.google.com/go/mediatranslation: v0.8.1 → v0.8.4
- cloud.google.com/go/memcache: v1.10.1 → v1.10.4
- cloud.google.com/go/metastore: v1.12.0 → v1.13.3
- cloud.google.com/go/monitoring: v1.15.1 → v1.17.0
- cloud.google.com/go/networkconnectivity: v1.12.1 → v1.14.3
- cloud.google.com/go/networkmanagement: v1.8.0 → v1.9.3
- cloud.google.com/go/networksecurity: v0.9.1 → v0.9.4
- cloud.google.com/go/notebooks: v1.9.1 → v1.11.2
- cloud.google.com/go/optimization: v1.4.1 → v1.6.2
- cloud.google.com/go/orchestration: v1.8.1 → v1.8.4
- cloud.google.com/go/orgpolicy: v1.11.1 → v1.12.0
- cloud.google.com/go/osconfig: v1.12.1 → v1.12.4
- cloud.google.com/go/oslogin: v1.10.1 → v1.13.0
- cloud.google.com/go/phishingprotection: v0.8.1 → v0.8.4
- cloud.google.com/go/policytroubleshooter: v1.8.0 → v1.10.2
- cloud.google.com/go/privatecatalog: v0.9.1 → v0.9.4
- cloud.google.com/go/pubsub: v1.33.0 → v1.34.0
- cloud.google.com/go/recaptchaenterprise/v2: v2.7.2 → v2.9.0
- cloud.google.com/go/recommendationengine: v0.8.1 → v0.8.4
- cloud.google.com/go/recommender: v1.10.1 → v1.12.0
- cloud.google.com/go/redis: v1.13.1 → v1.14.1
- cloud.google.com/go/resourcemanager: v1.9.1 → v1.9.4
- cloud.google.com/go/resourcesettings: v1.6.1 → v1.6.4
- cloud.google.com/go/retail: v1.14.1 → v1.14.4
- cloud.google.com/go/run: v1.2.0 → v1.3.3
- cloud.google.com/go/scheduler: v1.10.1 → v1.10.5
- cloud.google.com/go/secretmanager: v1.11.1 → v1.11.4
- cloud.google.com/go/security: v1.15.1 → v1.15.4
- cloud.google.com/go/securitycenter: v1.23.0 → v1.24.3
- cloud.google.com/go/servicedirectory: v1.11.0 → v1.11.3
- cloud.google.com/go/shell: v1.7.1 → v1.7.4
- cloud.google.com/go/spanner: v1.47.0 → v1.55.0
- cloud.google.com/go/speech: v1.19.0 → v1.21.0
- cloud.google.com/go/storagetransfer: v1.10.0 → v1.10.3
- cloud.google.com/go/talent: v1.6.2 → v1.6.5
- cloud.google.com/go/texttospeech: v1.7.1 → v1.7.4
- cloud.google.com/go/tpu: v1.6.1 → v1.6.4
- cloud.google.com/go/trace: v1.10.1 → v1.10.4
- cloud.google.com/go/translate: v1.8.2 → v1.10.0
- cloud.google.com/go/video: v1.19.0 → v1.20.3
- cloud.google.com/go/videointelligence: v1.11.1 → v1.11.4
- cloud.google.com/go/vision/v2: v2.7.2 → v2.7.5
- cloud.google.com/go/vmmigration: v1.7.1 → v1.7.4
- cloud.google.com/go/vmwareengine: v1.0.0 → v1.0.3
- cloud.google.com/go/vpcaccess: v1.7.1 → v1.7.4
- cloud.google.com/go/webrisk: v1.9.1 → v1.9.4
- cloud.google.com/go/websecurityscanner: v1.6.1 → v1.6.4
- cloud.google.com/go/workflows: v1.11.1 → v1.12.3
- cloud.google.com/go: v0.110.7 → v0.112.0
- github.com/Azure/go-ansiterm: [d185dfc → 306776e](https://github.com/Azure/go-ansiterm/compare/d185dfc...306776e)
- github.com/Microsoft/go-winio: [v0.6.0 → v0.6.2](https://github.com/Microsoft/go-winio/compare/v0.6.0...v0.6.2)
- github.com/Microsoft/hcsshim: [v0.8.26 → v0.12.6](https://github.com/Microsoft/hcsshim/compare/v0.8.26...v0.12.6)
- github.com/OneOfOne/xxhash: [v1.2.2 → v1.2.8](https://github.com/OneOfOne/xxhash/compare/v1.2.2...v1.2.8)
- github.com/cilium/ebpf: [v0.9.1 → v0.11.0](https://github.com/cilium/ebpf/compare/v0.9.1...v0.11.0)
- github.com/containerd/console: [v1.0.3 → v1.0.4](https://github.com/containerd/console/compare/v1.0.3...v1.0.4)
- github.com/containerd/containerd: [v1.4.9 → v1.7.20](https://github.com/containerd/containerd/compare/v1.4.9...v1.7.20)
- github.com/containerd/continuity: [v0.1.0 → v0.4.2](https://github.com/containerd/continuity/compare/v0.1.0...v0.4.2)
- github.com/containerd/fifo: [v1.0.0 → v1.1.0](https://github.com/containerd/fifo/compare/v1.0.0...v1.1.0)
- github.com/containerd/ttrpc: [v1.2.2 → v1.2.5](https://github.com/containerd/ttrpc/compare/v1.2.2...v1.2.5)
- github.com/coredns/corefile-migration: [v1.0.21 → v1.0.24](https://github.com/coredns/corefile-migration/compare/v1.0.21...v1.0.24)
- github.com/distribution/reference: [v0.5.0 → v0.6.0](https://github.com/distribution/reference/compare/v0.5.0...v0.6.0)
- github.com/docker/docker: [v20.10.27+incompatible → v27.1.1+incompatible](https://github.com/docker/docker/compare/v20.10.27...v27.1.1)
- github.com/docker/go-connections: [v0.4.0 → v0.5.0](https://github.com/docker/go-connections/compare/v0.4.0...v0.5.0)
- github.com/frankban/quicktest: [v1.14.0 → v1.14.5](https://github.com/frankban/quicktest/compare/v1.14.0...v1.14.5)
- github.com/go-openapi/jsonpointer: [v0.19.6 → v0.21.0](https://github.com/go-openapi/jsonpointer/compare/v0.19.6...v0.21.0)
- github.com/go-openapi/swag: [v0.22.4 → v0.23.0](https://github.com/go-openapi/swag/compare/v0.22.4...v0.23.0)
- github.com/golang/mock: [v1.3.1 → v1.1.1](https://github.com/golang/mock/compare/v1.3.1...v1.1.1)
- github.com/google/cadvisor: [v0.49.0 → v0.50.0](https://github.com/google/cadvisor/compare/v0.49.0...v0.50.0)
- github.com/google/pprof: [4bfdf5a → 813a5fb](https://github.com/google/pprof/compare/4bfdf5a...813a5fb)
- github.com/opencontainers/image-spec: [v1.0.2 → v1.1.0](https://github.com/opencontainers/image-spec/compare/v1.0.2...v1.1.0)
- github.com/opencontainers/runc: [v1.1.13 → v1.1.14](https://github.com/opencontainers/runc/compare/v1.1.13...v1.1.14)
- github.com/opencontainers/runtime-spec: [494a5a6 → v1.2.0](https://github.com/opencontainers/runtime-spec/compare/494a5a6...v1.2.0)
- github.com/pelletier/go-toml: [v1.2.0 → v1.9.5](https://github.com/pelletier/go-toml/compare/v1.2.0...v1.9.5)
- github.com/urfave/cli: [v1.22.2 → v1.22.15](https://github.com/urfave/cli/compare/v1.22.2...v1.22.15)
- github.com/vishvananda/netlink: [v1.1.0 → v1.3.0](https://github.com/vishvananda/netlink/compare/v1.1.0...v1.3.0)
- go.etcd.io/bbolt: v1.3.9 → v1.3.11
- go.etcd.io/etcd/api/v3: v3.5.14 → v3.5.16
- go.etcd.io/etcd/client/pkg/v3: v3.5.14 → v3.5.16
- go.etcd.io/etcd/client/v2: v2.305.13 → v2.305.16
- go.etcd.io/etcd/client/v3: v3.5.14 → v3.5.16
- go.etcd.io/etcd/pkg/v3: v3.5.13 → v3.5.16
- go.etcd.io/etcd/raft/v3: v3.5.13 → v3.5.16
- go.etcd.io/etcd/server/v3: v3.5.13 → v3.5.16
- go.uber.org/zap: v1.26.0 → v1.27.0
- golang.org/x/crypto: v0.24.0 → v0.26.0
- golang.org/x/exp: f3d0a9c → 8a7402a
- golang.org/x/lint: 1621716 → d0100b6
- golang.org/x/mod: v0.17.0 → v0.20.0
- golang.org/x/net: v0.26.0 → v0.28.0
- golang.org/x/sync: v0.7.0 → v0.8.0
- golang.org/x/sys: v0.21.0 → v0.23.0
- golang.org/x/telemetry: f48c80b → bda5523
- golang.org/x/term: v0.21.0 → v0.23.0
- golang.org/x/text: v0.16.0 → v0.17.0
- golang.org/x/tools: e35e4cc → v0.24.0
- golang.org/x/xerrors: 04be3eb → 5ec99f8
- google.golang.org/genproto: b8732ec → ef43131
- gotest.tools/v3: v3.0.3 → v3.0.2
- honnef.co/go/tools: v0.0.1-2019.2.3 → ea95bdf
- k8s.io/gengo/v2: 51d4e06 → 2b36238
- k8s.io/kube-openapi: 70dd376 → f7e401e

### Removed
- bazil.org/fuse: 371fbbd
- cloud.google.com/go/storage: v1.0.0
- dmitri.shuralyov.com/gpu/mtl: 666a987
- github.com/BurntSushi/xgb: [27f1227](https://github.com/BurntSushi/xgb/tree/27f1227)
- github.com/alecthomas/template: [a0175ee](https://github.com/alecthomas/template/tree/a0175ee)
- github.com/armon/consul-api: [eb2c6b5](https://github.com/armon/consul-api/tree/eb2c6b5)
- github.com/armon/go-metrics: [f0300d1](https://github.com/armon/go-metrics/tree/f0300d1)
- github.com/armon/go-radix: [7fddfc3](https://github.com/armon/go-radix/tree/7fddfc3)
- github.com/aws/aws-sdk-go: [v1.35.24](https://github.com/aws/aws-sdk-go/tree/v1.35.24)
- github.com/bgentry/speakeasy: [v0.1.0](https://github.com/bgentry/speakeasy/tree/v0.1.0)
- github.com/bketelsen/crypt: [5cbc8cc](https://github.com/bketelsen/crypt/tree/5cbc8cc)
- github.com/cespare/xxhash: [v1.1.0](https://github.com/cespare/xxhash/tree/v1.1.0)
- github.com/containerd/typeurl: [v1.0.2](https://github.com/containerd/typeurl/tree/v1.0.2)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/etcd: [v3.3.13+incompatible](https://github.com/coreos/etcd/tree/v3.3.13)
- github.com/coreos/go-systemd: [95778df](https://github.com/coreos/go-systemd/tree/95778df)
- github.com/coreos/pkg: [399ea9e](https://github.com/coreos/pkg/tree/399ea9e)
- github.com/dgrijalva/jwt-go: [v3.2.0+incompatible](https://github.com/dgrijalva/jwt-go/tree/v3.2.0)
- github.com/dgryski/go-sip13: [e10d5fe](https://github.com/dgryski/go-sip13/tree/e10d5fe)
- github.com/fatih/color: [v1.7.0](https://github.com/fatih/color/tree/v1.7.0)
- github.com/go-gl/glfw: [e6da0ac](https://github.com/go-gl/glfw/tree/e6da0ac)
- github.com/gogo/googleapis: [v1.4.1](https://github.com/gogo/googleapis/tree/v1.4.1)
- github.com/google/martian: [v2.1.0+incompatible](https://github.com/google/martian/tree/v2.1.0)
- github.com/google/renameio: [v0.1.0](https://github.com/google/renameio/tree/v0.1.0)
- github.com/googleapis/gax-go/v2: [v2.0.5](https://github.com/googleapis/gax-go/tree/v2.0.5)
- github.com/gopherjs/gopherjs: [0766667](https://github.com/gopherjs/gopherjs/tree/0766667)
- github.com/hashicorp/consul/api: [v1.1.0](https://github.com/hashicorp/consul/tree/api/v1.1.0)
- github.com/hashicorp/consul/sdk: [v0.1.1](https://github.com/hashicorp/consul/tree/sdk/v0.1.1)
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
- github.com/hashicorp/golang-lru: [v0.5.1](https://github.com/hashicorp/golang-lru/tree/v0.5.1)
- github.com/hashicorp/hcl: [v1.0.0](https://github.com/hashicorp/hcl/tree/v1.0.0)
- github.com/hashicorp/logutils: [v1.0.0](https://github.com/hashicorp/logutils/tree/v1.0.0)
- github.com/hashicorp/mdns: [v1.0.0](https://github.com/hashicorp/mdns/tree/v1.0.0)
- github.com/hashicorp/memberlist: [v0.1.3](https://github.com/hashicorp/memberlist/tree/v0.1.3)
- github.com/hashicorp/serf: [v0.8.2](https://github.com/hashicorp/serf/tree/v0.8.2)
- github.com/imdario/mergo: [v0.3.6](https://github.com/imdario/mergo/tree/v0.3.6)
- github.com/jmespath/go-jmespath: [v0.4.0](https://github.com/jmespath/go-jmespath/tree/v0.4.0)
- github.com/jstemmer/go-junit-report: [af01ea7](https://github.com/jstemmer/go-junit-report/tree/af01ea7)
- github.com/jtolds/gls: [v4.20.0+incompatible](https://github.com/jtolds/gls/tree/v4.20.0)
- github.com/magiconair/properties: [v1.8.1](https://github.com/magiconair/properties/tree/v1.8.1)
- github.com/mattn/go-colorable: [v0.0.9](https://github.com/mattn/go-colorable/tree/v0.0.9)
- github.com/mattn/go-isatty: [v0.0.3](https://github.com/mattn/go-isatty/tree/v0.0.3)
- github.com/miekg/dns: [v1.0.14](https://github.com/miekg/dns/tree/v1.0.14)
- github.com/mitchellh/cli: [v1.0.0](https://github.com/mitchellh/cli/tree/v1.0.0)
- github.com/mitchellh/go-testing-interface: [v1.0.0](https://github.com/mitchellh/go-testing-interface/tree/v1.0.0)
- github.com/mitchellh/gox: [v0.4.0](https://github.com/mitchellh/gox/tree/v0.4.0)
- github.com/mitchellh/iochan: [v1.0.0](https://github.com/mitchellh/iochan/tree/v1.0.0)
- github.com/mitchellh/mapstructure: [v1.1.2](https://github.com/mitchellh/mapstructure/tree/v1.1.2)
- github.com/oklog/ulid: [v1.3.1](https://github.com/oklog/ulid/tree/v1.3.1)
- github.com/pascaldekloe/goe: [57f6aae](https://github.com/pascaldekloe/goe/tree/57f6aae)
- github.com/posener/complete: [v1.1.1](https://github.com/posener/complete/tree/v1.1.1)
- github.com/prometheus/tsdb: [v0.7.1](https://github.com/prometheus/tsdb/tree/v0.7.1)
- github.com/ryanuber/columnize: [9b3edd6](https://github.com/ryanuber/columnize/tree/9b3edd6)
- github.com/sean-/seed: [e2103e2](https://github.com/sean-/seed/tree/e2103e2)
- github.com/smartystreets/assertions: [b2de0cb](https://github.com/smartystreets/assertions/tree/b2de0cb)
- github.com/smartystreets/goconvey: [v1.6.4](https://github.com/smartystreets/goconvey/tree/v1.6.4)
- github.com/spaolacci/murmur3: [f09979e](https://github.com/spaolacci/murmur3/tree/f09979e)
- github.com/spf13/afero: [v1.1.2](https://github.com/spf13/afero/tree/v1.1.2)
- github.com/spf13/cast: [v1.3.0](https://github.com/spf13/cast/tree/v1.3.0)
- github.com/spf13/jwalterweatherman: [v1.0.0](https://github.com/spf13/jwalterweatherman/tree/v1.0.0)
- github.com/spf13/viper: [v1.7.0](https://github.com/spf13/viper/tree/v1.7.0)
- github.com/subosito/gotenv: [v1.2.0](https://github.com/subosito/gotenv/tree/v1.2.0)
- github.com/ugorji/go: [v1.1.4](https://github.com/ugorji/go/tree/v1.1.4)
- github.com/xordataexchange/crypt: [b2862e3](https://github.com/xordataexchange/crypt/tree/b2862e3)
- golang.org/x/image: cff245a
- golang.org/x/mobile: d2bd2a2
- google.golang.org/api: v0.13.0
- gopkg.in/alecthomas/kingpin.v2: v2.2.6
- gopkg.in/errgo.v2: v2.1.0
- gopkg.in/ini.v1: v1.51.0
- gopkg.in/resty.v1: v1.12.0
- rsc.io/binaryregexp: v0.2.0