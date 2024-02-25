<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.30.0-alpha.2](#v1300-alpha2)
  - [Downloads for v1.30.0-alpha.2](#downloads-for-v1300-alpha2)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.30.0-alpha.1](#changelog-since-v1300-alpha1)
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
- [v1.30.0-alpha.1](#v1300-alpha1)
  - [Downloads for v1.30.0-alpha.1](#downloads-for-v1300-alpha1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.29.0](#changelog-since-v1290)
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

<!-- END MUNGE: GENERATED_TOC -->

# v1.30.0-alpha.2


## Downloads for v1.30.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes.tar.gz) | b6946e906e2d089431132ff4d8e24cb1b61f676f4df09b21b22a472c5aa796513ce8d7c39a312c8c0447ba0bb6cb5c4157c2be7645f91d6cf949a03a01cf9458
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-src.tar.gz) | a339603f532774a24d9dcbde8ebc2188729a469cc670ba5f00a09cf8465f2e00bb364b5f6739d79dfac9d20a7347f495672d2f184cfce73407925e0314633a3b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | 2930b28b275662ac7a78e6d59539809138b173a930c360a417f429bbcf31e7c3ef0a1a544028c5f81e1972a9f07ac0b459f6c02e97d7c0ccbcaa39ed229ef60a
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 6e8131d70116dce503a6800504ac349c9e4f3d359c31821083ceab936b8bd782a5f2e3027b4222fa133b7d27def3b15312fa022eb421ce2b3cfdd89f75300b5b
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 9272c915586ab46cd9cef8b7029958e7c9771a0109f83eb0d9991bfe7c0468a5c6d55329e656be9cf13217b6a06875bdde2eec1a870328397a54500836267ab8
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | fd8d6c83b91b13b80dd2a3000ae11746e664039fcf4bd7f1704dc6e53391e0114ab9d53dee83edb29d54ddd22d6ec042735b1e6e0930626f441147e6f4b4cfe7
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 57b1df4ea4fedd6555dd297808ac23e9ffd7da4b5fd4876088863a287edef34b0d697f296c3da405649146c4c84f72e41155dcf858990ae6e810adb800452539
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 83e61c039bd2a7d113b68c97a06e55deff2633abd9e6f1afa98ef22a4308383f2fba3309e3b9ba23f27d0d6a3a99232e0b3404f3848c94f927d654e6317f300a
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | cf78c218e4c23e1ad13dc75b465d38c57c2fc284eafe342adaf3b84568965f3629e2c5543c38f2c24e93ca8f5ef72c755c401fd9b5f46e8742095734784f324a
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 64913790635f51dc012d463b4f2483453483d21c6d228f2c2ac740b8c1abcf25251baffca8331c7d34a8eb945df96efd24f4d23089cc13c992baddb678ebe2b3
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 066fe65b02c68858f09119b657d23b19d770f1432790666e80fd2644251cfc949d323857d5e2308a865442714138be40ee7269e8109314d3e9e99e7917380786
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 057b9d0eac9d6f8f96b29a237692f346bab054947d6493fa1b75d143d457c146e46713694e5987e5fc7adf2950d5a16a974f1eb6ffb204a992b6d852435910b6
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 0338179407fca68fc67e019fa89075eef497a130d7a09f974692b715a803e1d6521d8d31d55421117e6cefc5aee2902b3afc095fdcacd06438a1673ba9a23cd6

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | fb41f7e577b6e2501819cbb71761e29e38d50d0279fa41508af63ea3857c0c05ca5feb584f65d784d1fb6f765d6c7e9d479c91f904feebd297b05ef296567ce8
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 273796e1bcea82151b64974f000813f9e8e63bf8314dc2980d99610363967a8928e52d4958a03f413cb762d69b3d89918e43dac33921f2855acace09d5a74e47
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 14061e55d204a09e0c1ac7c55931ee62ca1ce9e4c843bd4c7ad42c746a5ab6812d74642bf16146d6191dc72432ebb1fc1304e9486643adfcc8419c46753b4d74
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | d1a4ef0c30d68eda1710c032ded345acfc295a33aff37b01cb185bc5643efb1a9c27ac90dfb5afa4f95741b03ff4a55a11063e06b720715f425e9178da9ed3f9

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 6c9589d2dc82cc838ef27f2370d503f2750aa8feaef592dd7353bd74a482a2904078df3a3488ccd3e6f64f180f1d27b8931b75f7cc97f4a1f9d543299f0b8db8
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 862d0c46d911ce78d191b0996e74263fc14db461cacfb8fb4fdddf4b6b982f4f72feaa1cba960c30dc0af007e718f2266a18e87cdda87fca54c511ab667773da
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 35bcf7be699b443f69b76b7133e94da69c234e3d4d021a3e41a0f09837466521d032422eaf6fd7dbc9b96eccdc97ec5c3a339bd410d1befcd1cad2de1efbd7f6
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 15a52713d9640ca4365a9ba40b3523e658a2889bd1e25b3e40d97d78bc03ce3d2e189d9696210059438393a4decc636e164d92d716d0c7eadd35ff7c22bcd3b3
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 0452a35597a22014571bac052947cc751d3ac78ac02cc6b9cee206e12717930f847cde3fe84d7f44c52b274c00513c2d7c4423b1d69ee50c25973371803e49cb

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.30.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.30.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.30.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.30.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.30.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.30.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.30.0-alpha.1

## Changes by Kind

### Deprecation

- Removed the `SecurityContextDeny` admission plugin, deprecated since v1.27. The Pod Security Admission plugin, available since v1.25, is recommended instead. See https://kubernetes.io/docs/reference/access-authn-authz/admission-controllers/#securitycontextdeny for more information. ([#122612](https://github.com/kubernetes/kubernetes/pull/122612), [@mtardy](https://github.com/mtardy)) [SIG Auth, Security and Testing]

### API Change

- Updated an audit annotation key used by the `…/serviceaccounts/<name>/token` resource handler.
  The annotation used to persist the issued credential identifier is now `authentication.kubernetes.io/issued-credential-id`. ([#123098](https://github.com/kubernetes/kubernetes/pull/123098), [@munnerz](https://github.com/munnerz)) [SIG Auth]

### Feature

- Add apiserver.latency.k8s.io/decode-response-object annotation to the audit log to record the decoding time ([#121512](https://github.com/kubernetes/kubernetes/pull/121512), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery]
- Added apiserver_encryption_config_controller_automatic_reloads_total to measure total number of reload successes and failures of encryption configuration. This metric contains the `status` label with enum value of `success` and `failure`.
    - Deprecated apiserver_encryption_config_controller_automatic_reload_success_total and apiserver_encryption_config_controller_automatic_reload_failure_total metrics. Use apiserver_encryption_config_controller_automatic_reloads_total instead. ([#123179](https://github.com/kubernetes/kubernetes/pull/123179), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Allow a zero value for the 'nominalConcurrencyShares' field of the PriorityLevelConfiguration object 
  either using the flowcontrol.apiserver.k8s.io/v1 or flowcontrol.apiserver.k8s.io/v1beta3 API ([#123001](https://github.com/kubernetes/kubernetes/pull/123001), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Graduated support for passing dual-stack `kubelet --node-ip` values when using
  a cloud provider. The feature is now GA and the `CloudDualStackNodeIPs` feature
  gate is always enabled. ([#123134](https://github.com/kubernetes/kubernetes/pull/123134), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Cloud Provider and Node]
- Kubernetes is now built with go 1.22 ([#123217](https://github.com/kubernetes/kubernetes/pull/123217), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- The scheduler retries Pods, which are failed by nodevolumelimits due to not found PVCs, only when new PVCs are added. ([#121952](https://github.com/kubernetes/kubernetes/pull/121952), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Storage]
- Update distroless-iptables to v0.5.0 debian-base to bookworm-v1.0.1 and setcap to bookworm-v1.0.1 ([#123170](https://github.com/kubernetes/kubernetes/pull/123170), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Architecture, Cloud Provider, Release, Storage and Testing]
- Users can traverse all the pods that are in the scheduler and waiting in the permit stage through method `IterateOverWaitingPods`. In other words,  all waitingPods in scheduler can be obtained from any profiles. Before this commit, each profile could only obtain waitingPods within that profile. ([#122946](https://github.com/kubernetes/kubernetes/pull/122946), [@NoicFank](https://github.com/NoicFank)) [SIG Scheduling]
- ValidatingAdmissionPolicy now supports type checking policies that make use of `variables`. ([#123083](https://github.com/kubernetes/kubernetes/pull/123083), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery]

### Bug or Regression

- Fix Pod stuck in Terminating because of GenerateUnmapVolumeFunc missing globalUnmapPath when kubelet tries to clean up all volumes that failed reconstruction. ([#123032](https://github.com/kubernetes/kubernetes/pull/123032), [@carlory](https://github.com/carlory)) [SIG Storage]
- Fix deprecated version for pod_scheduling_duration_seconds that caused the metric to be hidden by default in 1.29. ([#123038](https://github.com/kubernetes/kubernetes/pull/123038), [@alculquicondor](https://github.com/alculquicondor)) [SIG Instrumentation and Scheduling]
- Fix error when trying to expand a volume that does not require node expansion ([#123055](https://github.com/kubernetes/kubernetes/pull/123055), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Fix the following volume plugins may not create user visible files after kubelet was restarted. 
  - configmap 
  - secret 
  - projected
  - downwardapi ([#122807](https://github.com/kubernetes/kubernetes/pull/122807), [@carlory](https://github.com/carlory)) [SIG Storage]
- Fixed cleanup of Pod volume mounts when a file was used as a subpath. ([#123052](https://github.com/kubernetes/kubernetes/pull/123052), [@jsafrane](https://github.com/jsafrane)) [SIG Node]
- Fixes an issue calculating total CPU usage reported for Windows nodes ([#122999](https://github.com/kubernetes/kubernetes/pull/122999), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Fixing issue where AvailableBytes sometimes does not report correctly on WindowsNodes when PodAndContainerStatsFromCRI feature is enabled. ([#122846](https://github.com/kubernetes/kubernetes/pull/122846), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Kubeadm: do not upload kubelet patch configuration into `kube-system/kubelet-config` ConfigMap ([#123093](https://github.com/kubernetes/kubernetes/pull/123093), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where the --rootfs global flag does not work with "kubeadm upgrade node" for control plane nodes. ([#123077](https://github.com/kubernetes/kubernetes/pull/123077), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: kubelet-finalize phase of "kubeadm init" no longer requires kubelet kubeconfig to have a specific authinfo ([#123171](https://github.com/kubernetes/kubernetes/pull/123171), [@vrutkovs](https://github.com/vrutkovs)) [SIG Cluster Lifecycle]
- Show enum values in kubectl explain if they were defined ([#123023](https://github.com/kubernetes/kubernetes/pull/123023), [@ah8ad3](https://github.com/ah8ad3)) [SIG CLI]

### Other (Cleanup or Flake)

- Build etcd image v3.5.12 ([#123069](https://github.com/kubernetes/kubernetes/pull/123069), [@bzsuni](https://github.com/bzsuni)) [SIG API Machinery and Etcd]
- Fix registered wildcard clusterEvents doesn't work in scheduler requeueing. ([#123117](https://github.com/kubernetes/kubernetes/pull/123117), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Promote feature-gate LegacyServiceAccountTokenCleanUp to GA and lock to default ([#122635](https://github.com/kubernetes/kubernetes/pull/122635), [@carlory](https://github.com/carlory)) [SIG API Machinery, Auth and Testing]
- Update etcd to version 3.5.12 ([#123150](https://github.com/kubernetes/kubernetes/pull/123150), [@bzsuni](https://github.com/bzsuni)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]

## Dependencies

### Added
- github.com/fxamacker/cbor/v2: [v2.5.0](https://github.com/fxamacker/cbor/v2/tree/v2.5.0)
- github.com/x448/float16: [v0.8.4](https://github.com/x448/float16/tree/v0.8.4)

### Changed
- github.com/opencontainers/runc: [v1.1.11 → v1.1.12](https://github.com/opencontainers/runc/compare/v1.1.11...v1.1.12)
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.28.0 → v0.29.0

### Removed
_Nothing has changed._



# v1.30.0-alpha.1


## Downloads for v1.30.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes.tar.gz) | f9e74c1f8400e8c85a65cf85418a95e06a558d230539f4b2f7882b96709eeb3656277a7a1e59ccd699a085d6c94d31bd2dcc83a48669d610ca2064a0c978cbeb
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-src.tar.gz) | 413f02b4cba6db36625a14095fb155b12685991ae4ece29e9d91016714aadcfbd06ac88f7766a0943445d05145980a54208cc2ed9bc29f3976f0b61a1492ace2

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | d06d723da34e021db3dba1890970f5dc5e27209befb4da9cc5a8255bd124e1ea31c273d71c0ee864166acb2afa0cb08a492896c3e85efeccbbb02685c1a3b271
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 7132d1a1ad0f6222eae02251ecd9f6df5dfbf26c6f7f789d1e81d756049eccdd68fc3f6710606bce12b24b887443553198efc801be55e94d83767341f306650e
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 09500370309fe1d6472535ed048a5f173ef3bd3e12cbc74ba67e48767b07e7b295df78cabffa5eda140e659da602d17b961563a2ef2a20b2d38074d826a47a35
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 154dafa5fae88a8aeed82c0460fa37679da60327fdab8f966357fbcb905e6e6b5473eacb524c39adddccf245fcf3dea8d5715a497f0230d98df21c4cb3b450eb
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | d055b29111a90b2c19e9f45bd56e2ba0b779dc35562f21330cda7ed57d945a65343552019f0efe159a87e3a2973c9f0b86f8c16edebdb44b8b8f773354fec7b3
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | c498a0c7b4ce59b198105c88ef1d29a8c345f3e1b31ba083c3f79bfcca35ae32776fd38a3b6b0bad187e14f7d54eeb0e2471634caac631039a989bd6119ab244
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 50e5c8bb07fac4304b067a161c34021d0c090bb5d04aed2eff4d43cab5a8cdcffc72fe97b4231f986a5b55987ebc6f6142a7e779b82ad49a109d772c3eade979
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 91b10c0f531ba530ca9766e509d1bb717531ff70061735082664da8a2bd7b3282743f53a60d74a5cb1867206f06287aa60fdec1bb41c77b14748330c5ce1199c
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-windows-386.tar.gz) | eaa83eab240ccf54ad54e0f66eba55bd4b15c7c37ea9a015b2b69638d90a1d5e146f989912c7745e0cbb52f846aa0135dd943b2b4b600fcbc3f9c43352f678f3
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 874ad471bc887f0ae2c73d636475793716021b688baf9ae85bd9229d9ceb5ec4bab3bc9f423e2665b2a6f33697d0f5c0a838f274bb4539ea0031018687f39e85
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 5f20a1efba7eec42f1ff1811af3b7c2703d7323e5577fd131fe79c8e53da33973a7922e794f4bc64f1fa16696cdc01e4826d0878a2e46158350a9b6de4eb345b

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | fd631b9f8e500eee418a680bd5ee104508192136701642938167f8b42ee4d2577092bada924e7b56d05db534920faeca416292bf0c1636f816ac35db30d80693
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | cc20574eac935a61e9c23c056d8c325cf095e4217d7d23d278dcf0d2ca32c2651febd3eb3de51536fd48e0fd17cf6ec156bdcf53178c1959efc92e078d9aed44
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | e8aa36ba41856b7e73fe4a52e725b1b52c70701822f17af10b3ddd03566cf41ab280b69a99c39b8dca85a0b7d80c3f88f7b0b5d5cd1da551701958f8bd176a11
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | fdf61522374eeccda5c32b6c9dc5927a92f68c78af811976f798dce483856ebc1e52a6a2b08a121ba7a3b60f0f8e2d727814ff7aed7edd1e7282288a1cacb742

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | cc8d03394114c292eca5be257b667d5114d7934f58d1c14365ea0a68fdb4e699437f3ea1a28476c65a1247cf5b877e40c0dabd295792d2d0de160f2807f9a7de
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 1602ecf70f2d9e8ec077bdb4d45a18027c702be24d474c3fdaf6ad2e3a56527ee533b53a1b4bbbe501404cc3f2d7d60a88f7f083352a57944e20b4d7109109e6
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 6494efec3efb3b0cc20170948eb2eb2e1a51c4913d26c0682de4ddcb4c20629232bc83020f62c1c618986df598008047258019e31d0ec444308064fafdbc861c
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 265041c73c045f567e6d014b594910524daef10cc0ce27ad760fb0188c34aeee52588dc1fbef1d9f474d11d032946bdbd527e9c04196294991d0fbe71ae5e678
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.30.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | faa5b4598326a9bd08715f5d6d0c1ac2f47fb20c0eb5745352f76b779d99a20480a9a79c6549e352d2a092b829e1926990b5fa859392603c1c510bf571b6094f

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.30.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.30.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.30.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.30.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.30.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.30.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-amd64), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kubectl-s390x)

## Changelog since v1.29.0

## Changes by Kind

### Deprecation

- Kubectl: remove deprecated flag prune-whitelist for apply, use flag prune-allowlist instead. ([#120246](https://github.com/kubernetes/kubernetes/pull/120246), [@pacoxu](https://github.com/pacoxu)) [SIG CLI and Testing]

### API Change

- Add CEL library for IP Addresses and CIDRs. This will not be available for use until 1.31. ([#121912](https://github.com/kubernetes/kubernetes/pull/121912), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery]
- Added to MutableFeatureGate the ability to override the default setting of feature gates, to allow default-enabling a feature on a component-by-component basis instead of for all affected components simultaneously. ([#122647](https://github.com/kubernetes/kubernetes/pull/122647), [@benluddy](https://github.com/benluddy)) [SIG API Machinery and Cluster Lifecycle]
- Adds a rule on the kube_codegen tool to ignore vendor folder during the code generation. ([#122729](https://github.com/kubernetes/kubernetes/pull/122729), [@jparrill](https://github.com/jparrill)) [SIG API Machinery and Cluster Lifecycle]
- Allow users to mutate FSGroupPolicy and PodInfoOnMount in CSIDriver.Spec ([#116209](https://github.com/kubernetes/kubernetes/pull/116209), [@haoruan](https://github.com/haoruan)) [SIG API Machinery, Storage and Testing]
- Client-go events: `NewEventBroadcasterAdapterWithContext` should be used instead of `NewEventBroadcasterAdapter` if the goal is to support contextual logging. ([#122142](https://github.com/kubernetes/kubernetes/pull/122142), [@pohly](https://github.com/pohly)) [SIG API Machinery, Instrumentation and Scheduling]
- Fixes accidental enablement of the new alpha `optionalOldSelf` API field in CustomResourceDefinition validation rules, which should only be allowed to be set when the CRDValidationRatcheting feature gate is enabled. ([#122329](https://github.com/kubernetes/kubernetes/pull/122329), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Implement  `prescore` extension point for `volumeBinding` plugin. Return skip if it doesn't do anything in Score. ([#115768](https://github.com/kubernetes/kubernetes/pull/115768), [@AxeZhan](https://github.com/AxeZhan)) [SIG Scheduling, Storage and Testing]
- Resource.k8s.io/ResourceClaim (alpha API): the strategic merge patch strategy for the `status.reservedFor` array was changed such that a strategic-merge-patch can add individual entries. This breaks clients using strategic merge patch to update status which rely on the previous behavior (replacing the entire array). ([#122276](https://github.com/kubernetes/kubernetes/pull/122276), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- When scheduling a mixture of pods using ResourceClaims and others which don't, scheduling a pod with ResourceClaims impacts scheduling latency less. ([#121876](https://github.com/kubernetes/kubernetes/pull/121876), [@pohly](https://github.com/pohly)) [SIG API Machinery, Node, Scheduling and Testing]

### Feature

- Add Timezone column in the output of kubectl get cronjob command ([#122231](https://github.com/kubernetes/kubernetes/pull/122231), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Add `WatchListClient` feature gate to `client-go`. When enabled it allows the client to get a stream of individual items instead of chunking from the server. ([#122571](https://github.com/kubernetes/kubernetes/pull/122571), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Add process_start_time_seconds to /metrics/slis endpoint of all components ([#122750](https://github.com/kubernetes/kubernetes/pull/122750), [@Richabanker](https://github.com/Richabanker)) [SIG Architecture, Instrumentation and Testing]
- Adds exec-interactive-mode and exec-provide-cluster-info flags in kubectl config set-credentials command ([#122023](https://github.com/kubernetes/kubernetes/pull/122023), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Allow scheduling framework plugins that implement io.Closer to be gracefully closed. ([#122498](https://github.com/kubernetes/kubernetes/pull/122498), [@Gekko0114](https://github.com/Gekko0114)) [SIG Scheduling]
- Change --nodeport-addresses behavior to default to "primary node IP(s) only" rather than "all node IPs". ([#122724](https://github.com/kubernetes/kubernetes/pull/122724), [@nayihz](https://github.com/nayihz)) [SIG Network and Windows]
- Etcd: build image for v3.5.11 ([#122233](https://github.com/kubernetes/kubernetes/pull/122233), [@mzaian](https://github.com/mzaian)) [SIG API Machinery]
- Informers now support adding Indexers after the informer starts ([#117046](https://github.com/kubernetes/kubernetes/pull/117046), [@howardjohn](https://github.com/howardjohn)) [SIG API Machinery]
- Introduce a feature gate mechanism to client-go. Depending on the actual implementation, users can control features via environmental variables or command line options. ([#122555](https://github.com/kubernetes/kubernetes/pull/122555), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kube-scheduler implements scheduling hints for the NodeAffinity plugin.
  The scheduling hints allow the scheduler to only retry scheduling a Pod
  that was previously rejected by the NodeAffinity plugin if a new Node or a Node update matches the Pod's node affinity. ([#122309](https://github.com/kubernetes/kubernetes/pull/122309), [@carlory](https://github.com/carlory)) [SIG Scheduling]
- Kube-scheduler implements scheduling hints for the NodeResourceFit plugin.
  The scheduling hints allow the scheduler to only retry scheduling a Pod
  that was previously rejected by the NodeResourceFit plugin if a new Node or 
  a Node update matches the Pod's resource requirements or if an old pod update 
  or delete matches the  Pod's resource requirements. ([#119177](https://github.com/kubernetes/kubernetes/pull/119177), [@carlory](https://github.com/carlory)) [SIG Scheduling]
- Kube-scheduler implements scheduling hints for the NodeUnschedulable plugin.
  The scheduling hints allow the scheduler to only retry scheduling a Pod
  that was previously rejected by the NodeSchedulable plugin if a new Node or a Node update sets .spec.unschedulable to false. ([#122334](https://github.com/kubernetes/kubernetes/pull/122334), [@carlory](https://github.com/carlory)) [SIG Scheduling]
- Kube-scheduler implements scheduling hints for the PodTopologySpread plugin.
  The scheduling hints allow the scheduler to retry scheduling a Pod
  that was previously rejected by the PodTopologySpread plugin if create/delete/update a related Pod or a node which matches the toplogyKey. ([#122195](https://github.com/kubernetes/kubernetes/pull/122195), [@nayihz](https://github.com/nayihz)) [SIG Scheduling]
- Kubeadm: add better handling of errors during unmount when calling "kubeadm reset". When failing to unmount directories under "/var/run/kubelet", kubeadm will now throw an error instead of showing a warning and continuing to cleanup said directory. In such situations it is better for you to inspect the problem and resolve it manually, then you can call "kubeadm reset" again to complete the cleanup. ([#122530](https://github.com/kubernetes/kubernetes/pull/122530), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl debug: add sysadmin profile ([#119200](https://github.com/kubernetes/kubernetes/pull/119200), [@eiffel-fl](https://github.com/eiffel-fl)) [SIG CLI and Testing]
- Kubernetes is now built with Go 1.21.6 ([#122705](https://github.com/kubernetes/kubernetes/pull/122705), [@cpanato](https://github.com/cpanato)) [SIG Architecture, Release and Testing]
- Kubernetes is now built with go 1.22rc2 ([#122889](https://github.com/kubernetes/kubernetes/pull/122889), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Print more information when kubectl describe a VolumeAttributesClass ([#122640](https://github.com/kubernetes/kubernetes/pull/122640), [@carlory](https://github.com/carlory)) [SIG CLI]
- Promote KubeProxyDrainingTerminatingNodes to Beta ([#122914](https://github.com/kubernetes/kubernetes/pull/122914), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Network]
- Promote feature gate StableLoadBalancerNodeSet to GA ([#122961](https://github.com/kubernetes/kubernetes/pull/122961), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG API Machinery, Cloud Provider and Network]
- Scheduler skips NodeAffinity Score plugin when NodeAffinity Score plugin has nothing to do with a Pod.
  You might notice an increase in the metric plugin_execution_duration_seconds for extension_point=score plugin=NodeAffinity, because the plugin will only run when the plugin is relevant ([#117024](https://github.com/kubernetes/kubernetes/pull/117024), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- The option `ignorable` of scheduler extender can skip error both filter and bind. ([#122503](https://github.com/kubernetes/kubernetes/pull/122503), [@sunbinnnnn](https://github.com/sunbinnnnn)) [SIG Scheduling]
- Update kubedns and nodelocaldns to release version 1.22.28 ([#121908](https://github.com/kubernetes/kubernetes/pull/121908), [@mzaian](https://github.com/mzaian)) [SIG Cloud Provider]
- Update some interfaces' signature in scheduler:
  
  1. PluginsRunner: use NodeInfo in `RunPreScorePlugins` and `RunScorePlugins`.
  2. PreScorePlugin: use NodeInfo in `PreScore`.
  3. Extender: use NodeInfo in `Filter` and `Prioritize`. ([#121954](https://github.com/kubernetes/kubernetes/pull/121954), [@AxeZhan](https://github.com/AxeZhan)) [SIG Autoscaling, Node, Scheduling, Storage and Testing]
- When PreFilterResult filters out some Nodes, the scheduling framework assumes them as rejected via `UnschedulableAndUnresolvable`, 
  that is those nodes won't be in the candidates of preemption process.
  Also, corrected how the scheduling framework handle Unschedulable status from PreFilter. 
  Before this PR, if PreFilter return `Unschedulable`, it may result in an unexpected abortion in the preemption, 
  which shouldn't happen in the default scheduler, but may happen in schedulers with a custom plugin. ([#119779](https://github.com/kubernetes/kubernetes/pull/119779), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- `kubectl describe`: added Suspend to job, and Node-Selectors and Tolerations to pod template output ([#122618](https://github.com/kubernetes/kubernetes/pull/122618), [@ivanvc](https://github.com/ivanvc)) [SIG CLI]

### Documentation

- A deprecated flag `--pod-max-in-unschedulable-pods-duration` was initially planned to be removed in v1.26, but we have to change this plan. We found [an issue](https://github.com/kubernetes/kubernetes/issues/110175) in which Pods can be stuck in the unschedulable pod pool for 5 min, and using this flag is the only workaround for this issue. 
  This issue only could happen if you use custom plugins or if you change plugin set being used in your scheduler via the scheduler config. ([#122013](https://github.com/kubernetes/kubernetes/pull/122013), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Fix delete pod declare no controllor note. ([#120159](https://github.com/kubernetes/kubernetes/pull/120159), [@Ithrael](https://github.com/Ithrael)) [SIG CLI]

### Bug or Regression

- Add imagefs.inodesfree to default EvictionHard settings ([#121834](https://github.com/kubernetes/kubernetes/pull/121834), [@vaibhav2107](https://github.com/vaibhav2107)) [SIG Node]
- Added metric name along with the utilization information when running kubectl get hpa ([#122804](https://github.com/kubernetes/kubernetes/pull/122804), [@sreeram-venkitesh](https://github.com/sreeram-venkitesh)) [SIG CLI]
- Allow deletion of pods that use raw block volumes on node reboot ([#122211](https://github.com/kubernetes/kubernetes/pull/122211), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Changed the API server so that for admission webhooks that have a URL matching the hostname `localhost`, or a loopback IP address, the connection supports HTTP/2 where it can be negotiated. ([#122558](https://github.com/kubernetes/kubernetes/pull/122558), [@linxiulei](https://github.com/linxiulei)) [SIG API Machinery and Testing]
- Etcd: Update to v3.5.11 ([#122393](https://github.com/kubernetes/kubernetes/pull/122393), [@mzaian](https://github.com/mzaian)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Fix Windows credential provider cannot find binary. Windows credential provider binary path may have ".exe" suffix so it is better to use LookPath() to support it flexibly. ([#120291](https://github.com/kubernetes/kubernetes/pull/120291), [@lzhecheng](https://github.com/lzhecheng)) [SIG Cloud Provider]
- Fix an issue where kubectl apply could panic when imported as a library ([#122346](https://github.com/kubernetes/kubernetes/pull/122346), [@Jefftree](https://github.com/Jefftree)) [SIG CLI]
- Fix panic of Evented PLEG during kubelet start-up ([#122475](https://github.com/kubernetes/kubernetes/pull/122475), [@pacoxu](https://github.com/pacoxu)) [SIG Node]
- Fix resource deletion failure caused by quota calculation error when InPlacePodVerticalScaling is turned on ([#122701](https://github.com/kubernetes/kubernetes/pull/122701), [@carlory](https://github.com/carlory)) [SIG API Machinery, Node and Testing]
- Fix the following volume plugins may not create user visible files after kubelet was restarted. 
  - configmap 
  - secret 
  - projected
  - downwardapi ([#122807](https://github.com/kubernetes/kubernetes/pull/122807), [@carlory](https://github.com/carlory)) [SIG Storage]
- Fix: Ignore unnecessary node events and improve daemonset controller performance. ([#121669](https://github.com/kubernetes/kubernetes/pull/121669), [@xigang](https://github.com/xigang)) [SIG Apps]
- Fix: Mount point may become local without calling NodePublishVolume after node rebooting. ([#119923](https://github.com/kubernetes/kubernetes/pull/119923), [@cvvz](https://github.com/cvvz)) [SIG Node and Storage]
- Fixed a bug where kubectl drain would consider a pod as having been deleted if an error occurs while calling the API. ([#122574](https://github.com/kubernetes/kubernetes/pull/122574), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Fixed a regression since 1.24 in the scheduling framework when overriding MultiPoint plugins (e.g. default plugins).
  The incorrect loop logic might lead to a plugin being loaded multiple times, consequently preventing any Pod from being scheduled, which is unexpected. ([#122068](https://github.com/kubernetes/kubernetes/pull/122068), [@caohe](https://github.com/caohe)) [SIG Scheduling]
- Fixed migration of in-tree vSphere volumes to the CSI driver. ([#122341](https://github.com/kubernetes/kubernetes/pull/122341), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixes a race condition in the iptables mode of kube-proxy in 1.27 and later
  that could result in some updates getting lost (e.g., when a service gets a
  new endpoint, the rules for the new endpoint might not be added until
  much later). ([#122204](https://github.com/kubernetes/kubernetes/pull/122204), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Fixes bug in ValidatingAdmissionPolicy which caused policies using CRD params to not successfully sync ([#123003](https://github.com/kubernetes/kubernetes/pull/123003), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery and Testing]
- For statically provisioned PVs, if its volume source is CSI type or it has migrated annotation, when it's deleted, the PersisentVolume controller won't changes its phase to the Failed state. 
  
  With this patch, the external provisioner can remove the finalizer in next reconcile loop. Unfortunately if the provious existing pv has the Failed state, this patch won't take effort. It requires users to remove finalizer. ([#122030](https://github.com/kubernetes/kubernetes/pull/122030), [@carlory](https://github.com/carlory)) [SIG Apps and Storage]
- If a pvc has an empty storageClassName, persistentvolume controller won't try to assign a default StorageClass ([#122704](https://github.com/kubernetes/kubernetes/pull/122704), [@carlory](https://github.com/carlory)) [SIG Apps and Storage]
- Improves scheduler performance when no scoring plugins are defined. ([#122058](https://github.com/kubernetes/kubernetes/pull/122058), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska)) [SIG Scheduling]
- Improves scheduler performance when no scoring plugins are defined. ([#122435](https://github.com/kubernetes/kubernetes/pull/122435), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska)) [SIG Scheduling]
- Kube-proxy: fixed LoadBalancerSourceRanges not working for nftables mode ([#122614](https://github.com/kubernetes/kubernetes/pull/122614), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Kubeadm: fix a regression in "kubeadm init" that caused a user-specified --kubeconfig file to be ignored. ([#122735](https://github.com/kubernetes/kubernetes/pull/122735), [@avorima](https://github.com/avorima)) [SIG Cluster Lifecycle]
- Make decoding etcd's response respect the timeout context. ([#121815](https://github.com/kubernetes/kubernetes/pull/121815), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery]
- QueueingHint implementation for NodeAffinity is reverted because we found potential scenarios where events that make Pods schedulable could be missed. ([#122285](https://github.com/kubernetes/kubernetes/pull/122285), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- QueueingHint implementation for NodeUnschedulable is reverted because we found potential scenarios where events that make Pods schedulable could be missed. ([#122288](https://github.com/kubernetes/kubernetes/pull/122288), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Remove wrong warning event (FileSystemResizeFailed) during a pod creation if it uses a readonly volume and the capacity of the volume is greater or equal to its request storage. ([#122508](https://github.com/kubernetes/kubernetes/pull/122508), [@carlory](https://github.com/carlory)) [SIG Storage]
- Reverts the EventedPLEG feature (beta, but disabled by default) back to alpha for a known issue ([#122697](https://github.com/kubernetes/kubernetes/pull/122697), [@pacoxu](https://github.com/pacoxu)) [SIG Node]
- The scheduling queue didn't notice any extenders' failures, it could miss some cluster events,
  and it could end up Pods rejected by Extenders stuck in unschedulable pod pool in 5min in the worst-case scenario.
  Now, the scheduling queue notices extenders' failures and requeue Pods rejected by Extenders appropriately. ([#122022](https://github.com/kubernetes/kubernetes/pull/122022), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Use errors.Is() to handle err returned by LookPath() ([#122600](https://github.com/kubernetes/kubernetes/pull/122600), [@lzhecheng](https://github.com/lzhecheng)) [SIG Cloud Provider]
- ValidateVolumeAttributesClassUpdate also validates new vac object. ([#122449](https://github.com/kubernetes/kubernetes/pull/122449), [@carlory](https://github.com/carlory)) [SIG Storage]
- When using a claim with immediate allocation and a pod referencing that claim couldn't get scheduled, the scheduler incorrectly may have tried to deallocate that claim. ([#122415](https://github.com/kubernetes/kubernetes/pull/122415), [@pohly](https://github.com/pohly)) [SIG Node and Scheduling]

### Other (Cleanup or Flake)

- Add warning for PV on relaim policy when it is Recycle ([#122339](https://github.com/kubernetes/kubernetes/pull/122339), [@carlory](https://github.com/carlory)) [SIG Storage]
- Cleanup: remove getStorageAccountName warning messages ([#121983](https://github.com/kubernetes/kubernetes/pull/121983), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Client-go: Optimized leaders renewing leases by updating leader lock optimistically without getting the record from the apiserver first. Also added a new metric `leader_election_slowpath_total` that allow users to monitor how many leader elections are updated non-optimistically. ([#122069](https://github.com/kubernetes/kubernetes/pull/122069), [@linxiulei](https://github.com/linxiulei)) [SIG API Machinery, Architecture and Instrumentation]
- Kube-proxy nftables mode is now compatible with kernel 5.4 ([#122296](https://github.com/kubernetes/kubernetes/pull/122296), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Kubeadm: improve the overall logic, error handling and output messages when waiting for the kubelet and API server /healthz endpoints to return 'ok'. The kubelet and API server checks no longer run in parallel, but one after another (in serial). ([#121958](https://github.com/kubernetes/kubernetes/pull/121958), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: show the supported shell types of 'kubeadm completion' in the error message when an invalid shell was specified ([#122477](https://github.com/kubernetes/kubernetes/pull/122477), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: use `ttlSecondsAfterFinished` to automatically clean up the `upgrade-health-check` Job that runs during upgrade preflighting. ([#122079](https://github.com/kubernetes/kubernetes/pull/122079), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Lock GA feature-gate ConsistentHTTPGetHandlers to default ([#122578](https://github.com/kubernetes/kubernetes/pull/122578), [@carlory](https://github.com/carlory)) [SIG Node]
- Migrate client-go/metadata to contextual logging ([#122225](https://github.com/kubernetes/kubernetes/pull/122225), [@ricardoapl](https://github.com/ricardoapl)) [SIG API Machinery]
- Migrated the cmd/kube-proxy to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#122197](https://github.com/kubernetes/kubernetes/pull/122197), [@fatsheep9146](https://github.com/fatsheep9146)) [SIG Network]
- Remove GA featuregate RemoveSelfLink ([#122468](https://github.com/kubernetes/kubernetes/pull/122468), [@carlory](https://github.com/carlory)) [SIG API Machinery]
- Remove GA featuregate about ExperimentalHostUserNamespaceDefaultingGate in 1.30 ([#122088](https://github.com/kubernetes/kubernetes/pull/122088), [@bzsuni](https://github.com/bzsuni)) [SIG Node]
- Remove GA featuregate about IPTablesOwnershipCleanup in 1.30 ([#122137](https://github.com/kubernetes/kubernetes/pull/122137), [@bzsuni](https://github.com/bzsuni)) [SIG Network]
- Removed generally available feature gate `ExpandedDNSConfig`. ([#122086](https://github.com/kubernetes/kubernetes/pull/122086), [@bzsuni](https://github.com/bzsuni)) [SIG Network]
- Removed generally available feature gate `KubeletPodResourcesGetAllocatable`. ([#122138](https://github.com/kubernetes/kubernetes/pull/122138), [@ii2day](https://github.com/ii2day)) [SIG Node]
- Removed generally available feature gate `KubeletPodResources`. ([#122139](https://github.com/kubernetes/kubernetes/pull/122139), [@bzsuni](https://github.com/bzsuni)) [SIG Node]
- Removed generally available feature gate `MinimizeIPTablesRestore`. ([#122136](https://github.com/kubernetes/kubernetes/pull/122136), [@ty-dc](https://github.com/ty-dc)) [SIG Network]
- Removed generally available feature gate `ProxyTerminatingEndpoints`. ([#122134](https://github.com/kubernetes/kubernetes/pull/122134), [@ty-dc](https://github.com/ty-dc)) [SIG Network]
- Removed the deprecated `azureFile` in-tree storage plugin ([#122576](https://github.com/kubernetes/kubernetes/pull/122576), [@carlory](https://github.com/carlory)) [SIG API Machinery, Cloud Provider, Node and Storage]
- Setting `--cidr-allocator-type` to `CloudAllocator` for `kube-controller-manager` will be removed in a future release. Please switch to and explore the options available in your external cloud provider ([#123011](https://github.com/kubernetes/kubernetes/pull/123011), [@dims](https://github.com/dims)) [SIG API Machinery and Network]
- The GA feature-gate APISelfSubjectReview is removed, and the feature is unconditionally enabled. ([#122032](https://github.com/kubernetes/kubernetes/pull/122032), [@carlory](https://github.com/carlory)) [SIG Auth and Testing]
- The feature gate `LegacyServiceAccountTokenTracking` (GA since 1.28) is now removed, since the feature is unconditionally enabled. ([#122409](https://github.com/kubernetes/kubernetes/pull/122409), [@Rei1010](https://github.com/Rei1010)) [SIG Auth]
- The in-tree cloud provider for azure has now been removed. Please use the external cloud provider and CSI driver from https://github.com/kubernetes/cloud-provider-azure instead. ([#122857](https://github.com/kubernetes/kubernetes/pull/122857), [@nilo19](https://github.com/nilo19)) [SIG API Machinery, Cloud Provider, Instrumentation, Node and Testing]
- The in-tree cloud provider for vSphere has now been removed. Please use the external cloud provider and CSI driver from https://github.com/kubernetes/cloud-provider-vsphere instead. ([#122937](https://github.com/kubernetes/kubernetes/pull/122937), [@dims](https://github.com/dims)) [SIG API Machinery, Cloud Provider, Storage and Testing]
- Update kube-dns to v1.22.27 ([#121736](https://github.com/kubernetes/kubernetes/pull/121736), [@ty-dc](https://github.com/ty-dc)) [SIG Cloud Provider]
- Updated cni-plugins to v1.4.0. ([#122178](https://github.com/kubernetes/kubernetes/pull/122178), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.29.0. ([#122271](https://github.com/kubernetes/kubernetes/pull/122271), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]

## Dependencies

### Added
- sigs.k8s.io/knftables: v0.0.14

### Changed
- github.com/go-logr/logr: [v1.3.0 → v1.4.1](https://github.com/go-logr/logr/compare/v1.3.0...v1.4.1)
- github.com/go-logr/zapr: [v1.2.3 → v1.3.0](https://github.com/go-logr/zapr/compare/v1.2.3...v1.3.0)
- github.com/onsi/ginkgo/v2: [v2.13.0 → v2.15.0](https://github.com/onsi/ginkgo/v2/compare/v2.13.0...v2.15.0)
- github.com/onsi/gomega: [v1.29.0 → v1.31.0](https://github.com/onsi/gomega/compare/v1.29.0...v1.31.0)
- github.com/opencontainers/runc: [v1.1.10 → v1.1.11](https://github.com/opencontainers/runc/compare/v1.1.10...v1.1.11)
- go.uber.org/atomic: v1.10.0 → v1.7.0
- go.uber.org/goleak: v1.2.1 → v1.3.0
- go.uber.org/zap: v1.19.0 → v1.26.0
- golang.org/x/crypto: v0.14.0 → v0.16.0
- golang.org/x/mod: v0.12.0 → v0.14.0
- golang.org/x/net: v0.17.0 → v0.19.0
- golang.org/x/sync: v0.3.0 → v0.5.0
- golang.org/x/sys: v0.13.0 → v0.15.0
- golang.org/x/term: v0.13.0 → v0.15.0
- golang.org/x/text: v0.13.0 → v0.14.0
- golang.org/x/tools: v0.12.0 → v0.16.1
- k8s.io/klog/v2: v2.110.1 → v2.120.1
- k8s.io/kube-openapi: 2dd684a → 778a556

### Removed
- github.com/Azure/azure-sdk-for-go: [v68.0.0+incompatible](https://github.com/Azure/azure-sdk-for-go/tree/v68.0.0)
- github.com/Azure/go-autorest/autorest/adal: [v0.9.23](https://github.com/Azure/go-autorest/autorest/adal/tree/v0.9.23)
- github.com/Azure/go-autorest/autorest/date: [v0.3.0](https://github.com/Azure/go-autorest/autorest/date/tree/v0.3.0)
- github.com/Azure/go-autorest/autorest/mocks: [v0.4.2](https://github.com/Azure/go-autorest/autorest/mocks/tree/v0.4.2)
- github.com/Azure/go-autorest/autorest/to: [v0.4.0](https://github.com/Azure/go-autorest/autorest/to/tree/v0.4.0)
- github.com/Azure/go-autorest/autorest/validation: [v0.3.1](https://github.com/Azure/go-autorest/autorest/validation/tree/v0.3.1)
- github.com/Azure/go-autorest/autorest: [v0.11.29](https://github.com/Azure/go-autorest/autorest/tree/v0.11.29)
- github.com/Azure/go-autorest/logger: [v0.2.1](https://github.com/Azure/go-autorest/logger/tree/v0.2.1)
- github.com/Azure/go-autorest/tracing: [v0.6.0](https://github.com/Azure/go-autorest/tracing/tree/v0.6.0)
- github.com/Azure/go-autorest: [v14.2.0+incompatible](https://github.com/Azure/go-autorest/tree/v14.2.0)
- github.com/a8m/tree: [10a5fd5](https://github.com/a8m/tree/tree/10a5fd5)
- github.com/benbjohnson/clock: [v1.1.0](https://github.com/benbjohnson/clock/tree/v1.1.0)
- github.com/danwinship/knftables: [v0.0.13](https://github.com/danwinship/knftables/tree/v0.0.13)
- github.com/dnaeon/go-vcr: [v1.2.0](https://github.com/dnaeon/go-vcr/tree/v1.2.0)
- github.com/dougm/pretty: [2ee9d74](https://github.com/dougm/pretty/tree/2ee9d74)
- github.com/gofrs/uuid: [v4.4.0+incompatible](https://github.com/gofrs/uuid/tree/v4.4.0)
- github.com/rasky/go-xdr: [4930550](https://github.com/rasky/go-xdr/tree/4930550)
- github.com/rubiojr/go-vhd: [02e2102](https://github.com/rubiojr/go-vhd/tree/02e2102)
- github.com/vmware/govmomi: [v0.30.6](https://github.com/vmware/govmomi/tree/v0.30.6)
- github.com/vmware/vmw-guestinfo: [25eff15](https://github.com/vmware/vmw-guestinfo/tree/25eff15)