<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.27.0-alpha.2](#v1270-alpha2)
  - [Downloads for v1.27.0-alpha.2](#downloads-for-v1270-alpha2)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.27.0-alpha.1](#changelog-since-v1270-alpha1)
  - [Changes by Kind](#changes-by-kind)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.27.0-alpha.1](#v1270-alpha1)
  - [Downloads for v1.27.0-alpha.1](#downloads-for-v1270-alpha1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.26.0](#changelog-since-v1260)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)

<!-- END MUNGE: GENERATED_TOC -->

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