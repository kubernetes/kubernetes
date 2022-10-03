<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.26.0-alpha.1](#v1260-alpha1)
  - [Downloads for v1.26.0-alpha.1](#downloads-for-v1260-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.25.0](#changelog-since-v1250)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Documentation](#documentation)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)

<!-- END MUNGE: GENERATED_TOC -->

# v1.26.0-alpha.1


## Downloads for v1.26.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes.tar.gz) | dfcda26750af76145c47aebe4d5e9f49569273da3545338814c99ce0657bea48e552c4ded5539ef484cd54f40800ef2c096c5bb556e4ae57f6027661a36c366b
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-src.tar.gz) | c87d326a23cb5bf1e276259d89d66eaadbc5402dfe74c47c83490ea987d3fa74dafa96ef7730bfe0300662587076a75af9eb308c18ee1ae860b786256bcfe546

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 2573e0f0cb8dcb9332056353b077305d33ed172fbd2872ba7c086da7187f19c3192ab1ca11c08747e27598f325d4ec55447b4329f1b2bd1dafd968f803714e99
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 1e47e1944ea5abc71f66588e87ea37a46026e8ba3f0ccf429c3db03e97524642dc32064854bffc46657e7144b4ed35ae83e59b35239c633851018b7613ec00a7
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 6006d4b9ac6044139b157ebe8d4744c88864630bf8970d0e4df452bc14d31ae4d27ab1048b044a1e90001efa8645e4a75f1c4870a2715a25395a27a5ab16e9e8
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | bb756cc5dd5264d2fe4e58782ea8e6b37e14188dbdd18ca0bb359a4b484a194890dbd2450bff5e2f7605379b3f421c86d22d4d5920c2074e7353bbd4cb0e1221
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 0d6c565474d0929a770d0ab08d85c89f80706f11325063d346fce4519bb9fbdf6fdda6a34691fc94a645e55833b50ef5f3f3f2d318abbcc3b12280d3e30386c8
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 0465e93a8d50370cae1840f2d098647e4d8648b0aa9e3a37ebd00749e86cf2d12e97600e4ba1f11f74f87f3dc088b929e7aca082ef27a181154e5e1aa204e4ec
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | f4885fedf19c43fcb609bf86c1c01a13843004fe87cb843b60623509c031ae44bffd1357c61b83609c3ecb6294a96047d8d9d2148a66626a1e3765d6a09a6400
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 3992a39912ec45ef06f287fab11101d5397933984ca80867f787704108865b6a312fc70e95721ddb352e87c47724fbfc8767e97985c74b65aa9e275cf26d23ec
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 65fc586f14f0cc1765eae377f3a3eba4bfafcb574d5d255627aed65128a0b56d5ea5ec913d29e7431333ef51a7d917a9703ea0f974feb98e9f2543a3612440a8
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 239fc7c6a6ef6fe3c1b20cff5d9793d1ac21225a289320ec6615c1305336ef40676096d4a9ebe60947d8b162b4e9a9df44e12c52581003ee0a3a0733c98edac0
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | c54b0a367e655842af4785a181dc366b3872fc8322495cfcb309518a854f87ed1064ed1c0a72fc663c31c8de7801fd041f0f05f7b788663c5e6941fc1313bcae

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | 736b911f58e4cf532726acba525a884827a627a95ed35be3ea9b444521183af6c4fb183afb3f82ceb715f0407fafb2e928f0e22fbc45ab62829721dd9f7c0811
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | a78bbd06e8581e4132abf7ddbb62f0d95bff61bf9c706c651f8d4d085372c9c787919a4f7d75a49a256f8f58f78e396cf7effa64f84405be03feb63c1e2d1474
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 5994ab21f9c7b85566de977a0cf4067c020aeadb3007edad4072edd7692b0fb903a57de732e694d9f5777358c12ee650aca7b58985b11243e0d1b2c565b7d03d
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | bd605e9b0e511805c7dbff4dbac8b111bc738343eacaeba1b234b4518da0f7bf33e64f26c231cdfb89532e57211b7dec10eb4d86969997dfc6a6cdbce0a6ccc9
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 39a8008ff250656bb6eeb3645ad3260e560c1565ef71e93e2908a73c88462906f776ed55623f8595f9e2a6a452ed75babd872df9e6506a8ed93828dbe4a5dc57

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 26965fb576f6adf0c894e30619b89c1d72048809e61cbfd45bf1da1a159f21d4920a894acdcf8c9cfc2d22e21a623f81c9a099150e9769419c3f91f1dd058de6
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | cfa1033c9caec034018503887e59013110725cde423b43a6f774900316888442d060cf9d9bb7d84286eea4f53b1bea409390540438bc363207dc962b083008b0
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | ac4f903a2aba9f98ca7ca221456f6e846a6cc5b71a8b58fa5e948a7c39057a35d2bf26b0e901abb9ae66e0916bb5ae71f0cea99aae33f254a9584a907047b8b2
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 8bc8aa34cec00241a3dbda0721af30f77401bc999020104d291c20b484e7ec4735395fc4c7859b58745046783a75f07590ae2b61ecd6a0eec734f44d682f2cb2
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 91373c07bf2a1b381f4c410c38b3f70d6914bacbe38090799878a45c3caf7d6acf0777225562ffc059d09f9f8fbe8fff5c3598f9c9403b0102bbf68911d84eea
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 91ca1e2cfe2d3e998af00bc443c12eaaf9a4061579c271de8dc409f1a61a746bb894743406ecb8c549900893ec30409eac0fd181eaa9bc2f283d00d108c7d606

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.26.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.26.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.26.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.26.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.26.0-alpha.1](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.25.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Deprecated beta APIs scheduled for removal in 1.26 are no longer served. See https://kubernetes.io/docs/reference/using-api/deprecation-guide/#v1-26 for more information. ([#111973](https://github.com/kubernetes/kubernetes/pull/111973), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
 
## Changes by Kind

### Deprecation

- The `gcp` and `azure` auth plugins have been removed from client-go and kubectl. See https://github.com/Azure/kubelogin and https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke for details about the cloud-specific replacements. ([#112341](https://github.com/kubernetes/kubernetes/pull/112341), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]

### API Change

- Add auth API to get self subject attributes (new selfsubjectreviews API is added). 
  The corresponding command for kubctl is provided - `kubectl auth whoami`. ([#111333](https://github.com/kubernetes/kubernetes/pull/111333), [@nabokihms](https://github.com/nabokihms)) [SIG API Machinery, Auth, CLI and Testing]
- Clarified the CFS quota as 100ms in the code comments and set the minimum cpuCFSQuotaPeriod to 1ms to match Linux kernel expectations. ([#112123](https://github.com/kubernetes/kubernetes/pull/112123), [@paskal](https://github.com/paskal)) [SIG API Machinery and Node]
- Component-base: make the validation logic about LeaderElectionConfiguration consistent between component-base and client-go ([#111758](https://github.com/kubernetes/kubernetes/pull/111758), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery and Scheduling]
- Fixes spurious `field is immutable` errors validating updates to Event API objects via the `events.k8s.io/v1` API ([#112183](https://github.com/kubernetes/kubernetes/pull/112183), [@liggitt](https://github.com/liggitt)) [SIG Apps]
- Protobuf serialization of metav1.MicroTime timestamps (used in `Lease` and `Event` API objects) has been corrected to truncate to microsecond precision, to match the documented behavior and JSON/YAML serialization. Any existing persisted data is truncated to microsecond when read from etcd. ([#111936](https://github.com/kubernetes/kubernetes/pull/111936), [@haoruan](https://github.com/haoruan)) [SIG API Machinery]
- Revert regression that prevented client-go latency metrics to be reported with a template URL to avoid label cardinality. ([#111752](https://github.com/kubernetes/kubernetes/pull/111752), [@aanm](https://github.com/aanm)) [SIG API Machinery]
- [kubelet] Change default `cpuCFSQuotaPeriod` value with enabled `cpuCFSQuotaPeriod` flag from 100ms to 100µs to match the Linux CFS and k8s defaults. `cpuCFSQuotaPeriod` of 100ms now requires `customCPUCFSQuotaPeriod` flag to be set to work. ([#111520](https://github.com/kubernetes/kubernetes/pull/111520), [@paskal](https://github.com/paskal)) [SIG API Machinery and Node]

### Feature

- A new "DisableCompression" field (default = false) has been added to kubeconfig under cluster info. When set to true, clients using the kubeconfig opt out of response compression for all requests to the apiserver. This can help improve list call latencies significantly when client-server network bandwidth is ample (>30MB/s) or if the server is CPU-constrained. ([#112309](https://github.com/kubernetes/kubernetes/pull/112309), [@shyamjvs](https://github.com/shyamjvs)) [SIG API Machinery and Auth]
- API Server tracing root span name for opentelemetry is changed from "KubernetesAPI" to "HTTP GET" ([#112545](https://github.com/kubernetes/kubernetes/pull/112545), [@dims](https://github.com/dims)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Storage and Testing]
- Add new Golang runtime-related metrics to Kubernetes components:
  - go_gc_cycles_automatic_gc_cycles_total
  - go_gc_cycles_forced_gc_cycles_total
  - go_gc_cycles_total_gc_cycles_total
  - go_gc_heap_allocs_by_size_bytes
  - go_gc_heap_allocs_bytes_total
  - go_gc_heap_allocs_objects_total
  - go_gc_heap_frees_by_size_bytes
  - go_gc_heap_frees_bytes_total
  - go_gc_heap_frees_objects_total
  - go_gc_heap_goal_bytes
  - go_gc_heap_objects_objects
  - go_gc_heap_tiny_allocs_objects_total
  - go_gc_pauses_seconds
  - go_memory_classes_heap_free_bytes
  - go_memory_classes_heap_objects_bytes
  - go_memory_classes_heap_released_bytes
  - go_memory_classes_heap_stacks_bytes
  - go_memory_classes_heap_unused_bytes
  - go_memory_classes_metadata_mcache_free_bytes
  - go_memory_classes_metadata_mcache_inuse_bytes
  - go_memory_classes_metadata_mspan_free_bytes
  - go_memory_classes_metadata_mspan_inuse_bytes
  - go_memory_classes_metadata_other_bytes
  - go_memory_classes_os_stacks_bytes
  - go_memory_classes_other_bytes
  - go_memory_classes_profiling_buckets_bytes
  - go_memory_classes_total_bytes
  - go_sched_goroutines_goroutines
  - go_sched_latencies_seconds ([#111910](https://github.com/kubernetes/kubernetes/pull/111910), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery, Architecture, Auth, Cloud Provider and Instrumentation]
- CSRDuration feature gate that graduated to GA in 1.24 and was unconditionally enabled has been removed in v1.26 ([#112386](https://github.com/kubernetes/kubernetes/pull/112386), [@Shubham82](https://github.com/Shubham82)) [SIG Auth]
- Client-go: SharedInformerFactory supports waiting for goroutines during shutdown ([#112200](https://github.com/kubernetes/kubernetes/pull/112200), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- Kube-apiserver: gzip compression switched from level 4 to level 1 to improve large list call latencies in exchange for higher network bandwidth usage (10-50% higher). This increases the headroom before very large unpaged list calls exceed request timeout limits. ([#112299](https://github.com/kubernetes/kubernetes/pull/112299), [@shyamjvs](https://github.com/shyamjvs)) [SIG API Machinery]
- Kubeadm: "show-join-command" has been added as a new separate phase at the end of "kubeadm init". You can skip printing the join information by using "kubeadm init --skip-phases=show-join-command". Executing only this phase on demand will throw an error because the phase needs dependencies such as bootstrap tokens to be pre-populated. ([#111512](https://github.com/kubernetes/kubernetes/pull/111512), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: add the flag "--cleanup-tmp-dir" for "kubeadm reset". It will cleanup the contents of "/etc/kubernetes/tmp". The flag is off by default. ([#112172](https://github.com/kubernetes/kubernetes/pull/112172), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubeadm: try to load CA cert from external CertificateAuthority file when CertificateAuthorityData is empty for existing kubeconfig ([#111783](https://github.com/kubernetes/kubernetes/pull/111783), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Go 1.19.1 ([#112287](https://github.com/kubernetes/kubernetes/pull/112287), [@palnabarun](https://github.com/palnabarun)) [SIG Release and Testing]
- Scheduler now retries updating a pod's status on ServiceUnavailable and InternalError errors, in addition to net ConnectionRefused error. ([#111809](https://github.com/kubernetes/kubernetes/pull/111809), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling]
- The `goroutines` metric is newly added in the scheduler. 
  It replaces `scheduler_goroutines` metric and it counts the number of goroutine in more places than `scheduler_goroutine` does. ([#112003](https://github.com/kubernetes/kubernetes/pull/112003), [@sanposhiho](https://github.com/sanposhiho)) [SIG Instrumentation and Scheduling]

### Documentation

- Clarified the default CFS quota period as being 100µs and not 100ms. ([#111554](https://github.com/kubernetes/kubernetes/pull/111554), [@paskal](https://github.com/paskal)) [SIG Node]

### Bug or Regression

- Adds back in unused flags on kubectl run command, which did not go through the required deprecation period before being removed. ([#112243](https://github.com/kubernetes/kubernetes/pull/112243), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Allow Label section in vsphere e2e cloudprovider configuration ([#112427](https://github.com/kubernetes/kubernetes/pull/112427), [@gnufied](https://github.com/gnufied)) [SIG Storage and Testing]
- Apiserver /healthz/etcd endpoint rate limits the number of forwarded health check requests to the etcd backends, answering with the last known state if the rate limit is exceeded. The rate limit is based on 1/2 of the timeout configured, with no burst allowed. ([#112046](https://github.com/kubernetes/kubernetes/pull/112046), [@aojea](https://github.com/aojea)) [SIG API Machinery]
- Avoid propagating hosts' `search .` into containers' `/etc/resolv.conf` ([#112157](https://github.com/kubernetes/kubernetes/pull/112157), [@dghubble](https://github.com/dghubble)) [SIG Network and Node]
- Callers using DelegatingAuthenticationOptions can use DisableAnonymous to disable Anonymous authentication. ([#112181](https://github.com/kubernetes/kubernetes/pull/112181), [@xueqzhan](https://github.com/xueqzhan)) [SIG API Machinery and Auth]
- Change error message when resource is not supported by given patch type in kubectl patch ([#112556](https://github.com/kubernetes/kubernetes/pull/112556), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Correct the calculating error in podTopologySpread plugin to avoid unexpected scheduling results. ([#112507](https://github.com/kubernetes/kubernetes/pull/112507), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Etcd: Update to v3.5.5 ([#112489](https://github.com/kubernetes/kubernetes/pull/112489), [@dims](https://github.com/dims)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle and Testing]
- Fix an ephemeral port exhaustion bug caused by improper connection management that occurred when a large number of objects were handled by kubectl while exec auth was in use. ([#112017](https://github.com/kubernetes/kubernetes/pull/112017), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]
- Fix list cost estimation in Priority and Fairness for list requests with metadata.name specified. ([#112557](https://github.com/kubernetes/kubernetes/pull/112557), [@marseel](https://github.com/marseel)) [SIG API Machinery]
- Fix race condition in GCE between containerized mounter setup in the kubelet and node startup. ([#112195](https://github.com/kubernetes/kubernetes/pull/112195), [@mattcary](https://github.com/mattcary)) [SIG Cloud Provider and Storage]
- Fix relative cpu priority for pods where containers explicitly request zero cpu by giving the lowest priority instead of falling back to the cpu limit to avoid possible cpu starvation of other pods ([#108832](https://github.com/kubernetes/kubernetes/pull/108832), [@waynepeking348](https://github.com/waynepeking348)) [SIG Node]
- Fixed bug in kubectl rollout history where only the latest revision was displayed when a specific revision was requested and an output format was specified ([#111093](https://github.com/kubernetes/kubernetes/pull/111093), [@brianpursley](https://github.com/brianpursley)) [SIG CLI and Testing]
- Fixed bug where dry run message was not printed when running kubectl label with --dry-run flag. ([#111571](https://github.com/kubernetes/kubernetes/pull/111571), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- For raw block CSI volumes on Kubernetes, kubelet was incorrectly calling CSI NodeStageVolume for every single "map" (i.e. raw block "mount") operation for a volume already attached to the node. This PR ensures it is only called once per volume per node. ([#112403](https://github.com/kubernetes/kubernetes/pull/112403), [@akankshakumari393](https://github.com/akankshakumari393)) [SIG Storage]
- Improves kubectl display of invalid request errors returned by the API server ([#112150](https://github.com/kubernetes/kubernetes/pull/112150), [@liggitt](https://github.com/liggitt)) [SIG CLI]
- Increase the maximum backoff delay of the endpointslice controller to match the expected sequence of delays when syncing Services. ([#112353](https://github.com/kubernetes/kubernetes/pull/112353), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Apps and Network]
- Kube-apiserver: redirect responses are no longer returned from backends by default. Set `--aggregator-reject-forwarding-redirect=false` to continue forwarding redirect responses. ([#112193](https://github.com/kubernetes/kubernetes/pull/112193), [@jindijamie](https://github.com/jindijamie)) [SIG API Machinery and Testing]
- Kube-apiserver: resolved a regression that treated `304 Not Modified` responses from aggregated API servers as internal errors ([#112526](https://github.com/kubernetes/kubernetes/pull/112526), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-apiserver: x-kubernetes-list-type validation is now enforced when updating status of custom resources ([#111866](https://github.com/kubernetes/kubernetes/pull/111866), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery]
- Kube-proxy no longer falls back from ipvs mode to iptables mode if you ask it to do ipvs but the system is not correctly configured. Instead, it will just exit with an error. ([#111806](https://github.com/kubernetes/kubernetes/pull/111806), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kube-scheduler: add taints filtering logic consistent with TaintToleration plugin for PodTopologySpread plugin ([#112357](https://github.com/kubernetes/kubernetes/pull/112357), [@SataQiu](https://github.com/SataQiu)) [SIG Scheduling and Testing]
- Kubeadm will cleanup the stale data on best effort basis. Stale data will be removed when each reset phase are executed, default etcd data directory will be cleanup when the `remove-etcd-member` phase are executed. ([#110972](https://github.com/kubernetes/kubernetes/pull/110972), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubeadm: allow RSA and ECDSA format keys in preflight check ([#112508](https://github.com/kubernetes/kubernetes/pull/112508), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: when a subcommand is needed but not provided for a kubeadm command, print a help screen instead of showing a short message. ([#111277](https://github.com/kubernetes/kubernetes/pull/111277), [@chymy](https://github.com/chymy)) [SIG Cluster Lifecycle]
- Log messages and metrics for the watch cache are now keyed by `<resource>.<group>` instead of `go` struct type. This means e.g. that `*v1.Pod` becomes `pods`. Additionally, resources that come from CustomResourceDefinitions are now displayed as the correct resource and group, instead of `*unstructured.Unstructured`. ([#111807](https://github.com/kubernetes/kubernetes/pull/111807), [@ncdc](https://github.com/ncdc)) [SIG API Machinery and Instrumentation]
- Move LocalStorageCapacityIsolationFSQuotaMonitoring back to Alpha. ([#112076](https://github.com/kubernetes/kubernetes/pull/112076), [@rphillips](https://github.com/rphillips)) [SIG Node and Testing]
- Pod failed in scheduling due to expected error will be updated with the reason of "SchedulerError" 
  rather than "Unschedulable" ([#111999](https://github.com/kubernetes/kubernetes/pull/111999), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling and Testing]
- Services of type LoadBalancer create fewer AWS security group rules in most cases ([#112267](https://github.com/kubernetes/kubernetes/pull/112267), [@sjenning](https://github.com/sjenning)) [SIG Cloud Provider]
- The errors in k8s.io/apimachinery/pkg/api/meta gained support for the stdlibs errors.Is matching, including when wrapped ([#111808](https://github.com/kubernetes/kubernetes/pull/111808), [@alvaroaleman](https://github.com/alvaroaleman)) [SIG API Machinery]
- The metrics etcd_request_duration_seconds and etcd_bookmark_counts now differentiate by group resource instead of object type, allowing unique entries per CustomResourceDefinition, instead of grouping them all under `*unstructured.Unstructured`. ([#112042](https://github.com/kubernetes/kubernetes/pull/112042), [@ncdc](https://github.com/ncdc)) [SIG API Machinery]
- Update the system-validators library to v1.8.0 ([#112026](https://github.com/kubernetes/kubernetes/pull/112026), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]

### Other (Cleanup or Flake)

- E2e: tests can now register callbacks with ginkgo.BeforeEach/AfterEach/DeferCleanup directly after creating a framework instance and are guaranteed that their code is called after the framework is initialized and before it gets cleaned up. ginkgo.DeferCleanup replaces f.AddAfterEach and AddCleanupAction which got removed to simplify the framework. ([#111998](https://github.com/kubernetes/kubernetes/pull/111998), [@pohly](https://github.com/pohly)) [SIG Storage and Testing]
- GlusterFS in-tree storage driver which was deprecated at kubernetes 1.25 release has been removed entirely in 1.26. ([#112015](https://github.com/kubernetes/kubernetes/pull/112015), [@humblec](https://github.com/humblec)) [SIG API Machinery, Cloud Provider, Instrumentation, Node, Scalability, Storage and Testing]
- Kube scheduler Component Config release version v1beta3 is deprecated in v1.26 and will be removed in v1.29, 
  also v1beta2 will be removed in v1.28. ([#112257](https://github.com/kubernetes/kubernetes/pull/112257), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Kube-scheduler: the DefaultPodTopologySpread, NonPreemptingPriority,  PodAffinityNamespaceSelector, PreferNominatedNode feature gates that graduated to GA in 1.24 and were unconditionally enabled have been removed in v1.26 ([#112567](https://github.com/kubernetes/kubernetes/pull/112567), [@SataQiu](https://github.com/SataQiu)) [SIG Scheduling]
- Kubeadm: remove the toleration for the "node-role.kubernetes.io/master" taint from the CoreDNS deployment of kubeadm. With the 1.25 release of kubeadm the taint "node-role.kubernetes.io/master" is no longer applied to control plane nodes and the toleration for it can be removed with the release of 1.26. You can also perform the same toleration removal from your own addon manifests. ([#112008](https://github.com/kubernetes/kubernetes/pull/112008), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: remove the usage of the --container-runtime=remote flag for the kubelet during kubeadm init/join/upgrade. The flag value "remote" has been the only possible value since dockershim was removed from the kubelet. ([#112000](https://github.com/kubernetes/kubernetes/pull/112000), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- NoneNone ([#111533](https://github.com/kubernetes/kubernetes/pull/111533), [@zhoumingcheng](https://github.com/zhoumingcheng)) [SIG CLI]
- Release-note ([#111708](https://github.com/kubernetes/kubernetes/pull/111708), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Apps, Instrumentation and Network]
- Scheduler dumper now exposes a summary to indicate the number of pending pods in each internal queue. ([#111726](https://github.com/kubernetes/kubernetes/pull/111726), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- The IndexedJob and SuspendJob feature gates that graduated to GA in 1.24 and were unconditionally enabled have been removed in v1.26 ([#112589](https://github.com/kubernetes/kubernetes/pull/112589), [@SataQiu](https://github.com/SataQiu)) [SIG Apps]
- The in-tree cloud provider for OpenStack (and the cinder volume provider) has now been removed. Please use the external cloud provider and csi driver from https://github.com/kubernetes/cloud-provider-openstack instead. ([#67782](https://github.com/kubernetes/kubernetes/pull/67782), [@dims](https://github.com/dims)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Release, Scheduling, Storage and Testing]

## Dependencies

### Added
- github.com/cenkalti/backoff/v4: [v4.1.3](https://github.com/cenkalti/backoff/v4/tree/v4.1.3)
- github.com/go-logr/stdr: [v1.2.2](https://github.com/go-logr/stdr/tree/v1.2.2)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.7.0](https://github.com/grpc-ecosystem/grpc-gateway/v2/tree/v2.7.0)
- github.com/jpillora/backoff: [v1.0.0](https://github.com/jpillora/backoff/tree/v1.0.0)
- go.opentelemetry.io/contrib/propagators/b3: v1.10.0
- go.opentelemetry.io/otel/exporters/otlp/internal/retry: v1.10.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.10.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.10.0

### Changed
- github.com/antlr/antlr4/runtime/Go/antlr: [f25a4f6 → v1.4.10](https://github.com/antlr/antlr4/runtime/Go/antlr/compare/f25a4f6...v1.4.10)
- github.com/cpuguy83/go-md2man/v2: [v2.0.1 → v2.0.2](https://github.com/cpuguy83/go-md2man/v2/compare/v2.0.1...v2.0.2)
- github.com/emicklei/go-restful/v3: [v3.8.0 → v3.9.0](https://github.com/emicklei/go-restful/v3/compare/v3.8.0...v3.9.0)
- github.com/felixge/httpsnoop: [v1.0.1 → v1.0.3](https://github.com/felixge/httpsnoop/compare/v1.0.1...v1.0.3)
- github.com/go-kit/log: [v0.1.0 → v0.2.0](https://github.com/go-kit/log/compare/v0.1.0...v0.2.0)
- github.com/go-logfmt/logfmt: [v0.5.0 → v0.5.1](https://github.com/go-logfmt/logfmt/compare/v0.5.0...v0.5.1)
- github.com/google/cel-go: [v0.12.4 → v0.12.5](https://github.com/google/cel-go/compare/v0.12.4...v0.12.5)
- github.com/google/go-cmp: [v0.5.6 → v0.5.9](https://github.com/google/go-cmp/compare/v0.5.6...v0.5.9)
- github.com/onsi/ginkgo/v2: [v2.1.4 → v2.2.0](https://github.com/onsi/ginkgo/v2/compare/v2.1.4...v2.2.0)
- github.com/onsi/gomega: [v1.19.0 → v1.20.1](https://github.com/onsi/gomega/compare/v1.19.0...v1.20.1)
- github.com/prometheus/client_golang: [v1.12.1 → v1.13.0](https://github.com/prometheus/client_golang/compare/v1.12.1...v1.13.0)
- github.com/prometheus/common: [v0.32.1 → v0.37.0](https://github.com/prometheus/common/compare/v0.32.1...v0.37.0)
- github.com/prometheus/procfs: [v0.7.3 → v0.8.0](https://github.com/prometheus/procfs/compare/v0.7.3...v0.8.0)
- github.com/spf13/cobra: [v1.4.0 → v1.5.0](https://github.com/spf13/cobra/compare/v1.4.0...v1.5.0)
- github.com/stretchr/objx: [v0.2.0 → v0.4.0](https://github.com/stretchr/objx/compare/v0.2.0...v0.4.0)
- github.com/stretchr/testify: [v1.7.0 → v1.8.0](https://github.com/stretchr/testify/compare/v1.7.0...v1.8.0)
- go.etcd.io/etcd/api/v3: v3.5.4 → v3.5.5
- go.etcd.io/etcd/client/pkg/v3: v3.5.4 → v3.5.5
- go.etcd.io/etcd/client/v2: v2.305.4 → v2.305.5
- go.etcd.io/etcd/client/v3: v3.5.4 → v3.5.5
- go.etcd.io/etcd/pkg/v3: v3.5.4 → v3.5.5
- go.etcd.io/etcd/raft/v3: v3.5.4 → v3.5.5
- go.etcd.io/etcd/server/v3: v3.5.4 → v3.5.5
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.20.0 → v0.35.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.20.0 → v0.35.0
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.20.0 → v0.35.0
- go.opentelemetry.io/otel/metric: v0.20.0 → v0.31.0
- go.opentelemetry.io/otel/sdk: v0.20.0 → v1.10.0
- go.opentelemetry.io/otel/trace: v0.20.0 → v1.10.0
- go.opentelemetry.io/otel: v0.20.0 → v1.10.0
- go.opentelemetry.io/proto/otlp: v0.7.0 → v0.19.0
- go.uber.org/goleak: v1.1.10 → v1.1.12
- golang.org/x/crypto: 3147a52 → 7b82a4e
- golang.org/x/lint: 6edffad → 1621716
- golang.org/x/oauth2: d3ed0bb → ee48083
- google.golang.org/grpc: v1.47.0 → v1.49.0
- google.golang.org/protobuf: v1.28.0 → v1.28.1
- k8s.io/gengo: c02415c → c0856e2
- k8s.io/klog/v2: v2.70.1 → v2.80.1
- k8s.io/system-validators: v1.7.0 → v1.8.0

### Removed
- github.com/auth0/go-jwt-middleware: [v1.0.1](https://github.com/auth0/go-jwt-middleware/tree/v1.0.1)
- github.com/boltdb/bolt: [v1.3.1](https://github.com/boltdb/bolt/tree/v1.3.1)
- github.com/go-ozzo/ozzo-validation: [v3.5.0+incompatible](https://github.com/go-ozzo/ozzo-validation/tree/v3.5.0)
- github.com/gophercloud/gophercloud: [v0.1.0](https://github.com/gophercloud/gophercloud/tree/v0.1.0)
- github.com/gopherjs/gopherjs: [fce0ec3](https://github.com/gopherjs/gopherjs/tree/fce0ec3)
- github.com/gorilla/mux: [v1.8.0](https://github.com/gorilla/mux/tree/v1.8.0)
- github.com/heketi/heketi: [v10.3.0+incompatible](https://github.com/heketi/heketi/tree/v10.3.0)
- github.com/heketi/tests: [f3775cb](https://github.com/heketi/tests/tree/f3775cb)
- github.com/jtolds/gls: [v4.20.0+incompatible](https://github.com/jtolds/gls/tree/v4.20.0)
- github.com/lpabon/godbc: [v0.1.1](https://github.com/lpabon/godbc/tree/v0.1.1)
- github.com/smartystreets/assertions: [v1.1.0](https://github.com/smartystreets/assertions/tree/v1.1.0)
- github.com/smartystreets/goconvey: [v1.6.4](https://github.com/smartystreets/goconvey/tree/v1.6.4)
- github.com/urfave/negroni: [v1.0.0](https://github.com/urfave/negroni/tree/v1.0.0)
- go.opentelemetry.io/contrib/propagators: v0.20.0
- go.opentelemetry.io/contrib: v0.20.0
- go.opentelemetry.io/otel/exporters/otlp: v0.20.0
- go.opentelemetry.io/otel/oteltest: v0.20.0
- go.opentelemetry.io/otel/sdk/export/metric: v0.20.0
- go.opentelemetry.io/otel/sdk/metric: v0.20.0