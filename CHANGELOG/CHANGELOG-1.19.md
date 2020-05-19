<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.19.0-beta.0](#v1190-beta0)
  - [Downloads for v1.19.0-beta.0](#downloads-for-v1190-beta0)
    - [Source Code](#source-code)
    - [Client binaries](#client-binaries)
    - [Server binaries](#server-binaries)
    - [Node binaries](#node-binaries)
  - [Changelog since v1.19.0-alpha.3](#changelog-since-v1190-alpha3)
  - [Changes by Kind](#changes-by-kind)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.19.0-alpha.3](#v1190-alpha3)
  - [Downloads for v1.19.0-alpha.3](#downloads-for-v1190-alpha3)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.19.0-alpha.2](#changelog-since-v1190-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
- [v1.19.0-alpha.2](#v1190-alpha2)
  - [Downloads for v1.19.0-alpha.2](#downloads-for-v1190-alpha2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.19.0-alpha.1](#changelog-since-v1190-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-2)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
- [v1.19.0-alpha.1](#v1190-alpha1)
  - [Downloads for v1.19.0-alpha.1](#downloads-for-v1190-alpha1)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.19.0-alpha.0](#changelog-since-v1190-alpha0)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-3)

<!-- END MUNGE: GENERATED_TOC -->

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
- Update github.com/moby/ipvs to v1.0.1 to fix IPVS compatiblity issue with older kernels ([#90555](https://github.com/kubernetes/kubernetes/pull/90555), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network]
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
  the above funtion，as we use cgroupRoots to create cadvisor interface，the wrong parameter cgroupRoots will lead eviction manager not  to collect metric from /docker, then kubelet frequently print those error：
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