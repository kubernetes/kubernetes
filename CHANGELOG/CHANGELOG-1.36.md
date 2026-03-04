<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.36.0-alpha.2](#v1360-alpha2)
  - [Downloads for v1.36.0-alpha.2](#downloads-for-v1360-alpha2)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.36.0-alpha.1](#changelog-since-v1360-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Dependency](#dependency)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.36.0-alpha.1](#v1360-alpha1)
  - [Downloads for v1.36.0-alpha.1](#downloads-for-v1360-alpha1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.35.0](#changelog-since-v1350)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-1)
    - [Dependency](#dependency-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)

<!-- END MUNGE: GENERATED_TOC -->

# v1.36.0-alpha.2


## Downloads for v1.36.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes.tar.gz) | 7aa0ed3b03a1574ec0db00d47381e3c76610be89f5ceb52145add9f01533d72833b9594499e0a9af43c41df38d4c448906689cd5662532f32728e11f0f0b39d7
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-src.tar.gz) | a6858468f30d207b375dd2278d8653c69211b364136b212e84ee563bba6ca2bfe89d5ecefda3bc88671ee0c901f02d0f50eefe1e12b7cc58a842fad90811bd23

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | 2dde9c67ba2d165e46d4e09c886b8b17563f3767f6fcb5ed98883dfe5bd2ecd7f70af94b0d71678a36916f8d6a6ba1cd3b2c9b4e8b7084e94dfef075291e32c5
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 70178cdf6431041bcd6bcd55e67ba978ec5fe53a47ded53d6fa932e65af8742a47099de5c20fb44da279e957abf57f54281b4ce8f7f7e638810c89851751c7fd
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 5ca76f2388552d2a4d601a19c0120d8db65f8f151ac2223a69cf426523de7d2d9532104d5464b15a588cb3c283203069f5347edd5a3ac912d459d7c703467b81
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | f9901ed9e5a1d9b8638a2ac84e65e7986c81d2d72d086aa14cf298397d72a290e6b22b644cb74aa61ff0181df6e3ca38103863289679d4ae00195827a3333c93
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | a347d318571485aa21325ba4de7dd4afa29bffa4b8060ed04a9ef3afe7cea4f12c520e6eebc1b96e991147f0123096a1a745f905b687baa6981d1f8d16971452
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 2e493202a0023af3cff5f5e6d19307791a994eac931f18fe33fe5016861d5b17690b75aece91c02b5e022d4e00a824e85cb7fdc43ca45e238cea16d202ef7ec6
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 9feebb6cca23158f299085945e1ff6db5c1742a72d157f89d837811f84db5f9cb9b5db042f2898840a86839621b89066a1ccc925cd2a981af64bbf513d94e229
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | c8a42085199cf548119fd2af703c84cc1d2fda61086b50e09d03ea18cd0aaf28063d21eb258b64c30884022c6ab81b78bfd6d848fc4bf04b410d2972eb23266f
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-windows-386.tar.gz) | ccb9320af60720149913f39a88d4abdf5c9d53d3ddd396a61504288576e6f2f27ef21cb26a23e27e34b4c56c95eb1daa54ed389460c51660074a92a3c34eee23
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | c068c174870aca6710c688088b46bee4e852ea97be7e8c02aad10bef81d1a3e6485b50ff41bee11121cce357f74cf54d67f65df04087f590ff6259bd03eb2d11
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 9247e597dab4fb45facc3c32d4b2772e7bb5c9814a1f5ae64b5fdba5469784957df154a7b7aedc99e79e4fd81e07f7fe2b0eb492e17492fba497e94e5e501a8f

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | ef279ca406ed3e939972353287b9ad769df0915749c5ffe61f096b6f75ab7604ff03177affc8c66a2dd688fe8b3fb6c1a7e8deabb4985fa7f40517f483fc5983
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | a3414c851a8c0cf42abfb1ba2e17fb40b31b4d2cd9a4224ed8ecfc951b127a829f4ab14fa2cf68dbabfbe19aa6212f09c207d25518832b13e9f81986415a1a88
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 841d2a9dd87ab1232d79611c855c1dd2d9aca11afdd4d790692e38894ef6fd37b382c68aacaea32daf71b39f196bea2448d860ce9ed70b03766d83895779a68b
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | ec89bea22f65a612eb0e52b03c6d277c35f5f2e2cbbc4cd50510f013b4d7ca6702f47017dcb417e1cc257017b1755ae02f6c21fac0d44881be2814aed0130127

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 37fca1f367e1b78035fd95f1bc547fca89018fc42f450e3660d6f108890e6f319a1e94bf52584178558101375846baf7721a59e88ef3a22f8a493381bbc2a6e1
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 6ddbfbe217414c874d2367cdba0943d9a8ead4c98bb4e2ccaec64cdb2a0218bae46f108b4fb3ba949f78707cc47477fd0b97dc2291a83d5c08832201a185f595
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 6e1707b097daae5e2a68ffd068245034e056ee49d96a70bfce430de3b752ddc148027a9b94155d77da270df617d9450b59e02749fc06a41622e48c9505f00553
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 12674c621376b790dc2421a113ce3ca226af8d649a65d465ddf0ce29f21ecea39f74283b39ced2f8b894595b86f0c40bfcaea37e9d7b44e95d1f8cdb0f01e6b8
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 25e2cd21678e8051716172074a8cc9efeb53e8ff275a15f9d0e801616804b9cbb1a5b16378e057b905dad75ce0e94e9837934343374bef87d4bf9843c8b3ef76

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.36.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Extended PostFilterResult with a list of Pods that the PostFilter plugin preempted.
  PostFilter plugin implementations need to be updated to return the preempted Pods (if any) in the PostFilterResult from the PostFilter method; returning nil means the plugin did not preempt any Pods at all. ([#136254](https://github.com/kubernetes/kubernetes/pull/136254), [@tosi3k](https://github.com/tosi3k)) [SIG Scheduling]
  - Kubeadm: removed the integrated support for flex-volumes in kubeadm. Users were advised to migrate away from flex-volumes as recommended by SIG Storage, since 1.22. If kubeadm users wish to continue using the feature, they would need a custom image for the KCM that is not based on distroless, pass the KCM flag `--flex-volume-plugin-dir` and mount the directory `/usr/libexec/kubernetes/kubelet-plugins/volume/exec` in the KCM static pod using kubeadm's `extraVolumes` mechanism before upgrading to 1.36. Up until now, kubeadm automatically did the mounting if the user passed the flag. ([#136423](https://github.com/kubernetes/kubernetes/pull/136423), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
 
## Changes by Kind

### Dependency

- Updates the etcd client library to v3.6.8 ([#137225](https://github.com/kubernetes/kubernetes/pull/137225), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Auth, Cloud Provider, Cluster Lifecycle, Etcd, Node, Scheduling and Testing]

### Deprecation

- Disabled git-repo volume plugin by default, with no option to turn it back on. ([#136400](https://github.com/kubernetes/kubernetes/pull/136400), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Storage]

### API Change

- Add a new concurrent-node-status-updates flag that is split from the concurrent-node-syncs flag ([#136716](https://github.com/kubernetes/kubernetes/pull/136716), [@yonizxz](https://github.com/yonizxz)) [SIG Cloud Provider]
- Fixed an issue in kube-apiserver, allowing it to recover from an established connection to an incorrect server that never returns the expected response during APIService availability checks. ([#137157](https://github.com/kubernetes/kubernetes/pull/137157), [@bsalamat](https://github.com/bsalamat)) [SIG API Machinery]
- Graduated MutatingAdmissionPolicy to GA (v1) in Kubernetes 1.36. The feature is now enabled by default. ([#136039](https://github.com/kubernetes/kubernetes/pull/136039), [@lalitc375](https://github.com/lalitc375)) [SIG API Machinery, Architecture, Etcd and Testing]
- Introduced stability-based lifecycle for declarative validation (Alpha/Beta/Stable). Scheduling Workload v1alpha1 now uses explicit declarative enforcement. ([#136793](https://github.com/kubernetes/kubernetes/pull/136793), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Scheduling]
- K8s.io/api: REST API types no longer implement marker `ProtoMessage()` methods, identifying them as standard v1 proto messages. Protobuf serialization of Kubernetes API types should use [k8s.io/apimachinery/pkg/runtime/serializer/protobuf](https://pkg.go.dev/k8s.io/apimachinery/pkg/runtime/serializer/protobuf). See [KEP-5589](https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/5589-gogo-dependency) for more details. ([#137084](https://github.com/kubernetes/kubernetes/pull/137084), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, Node, Scheduling and Storage]
- Pod Certificates (beta) now includes a PKCS#10 certificate signing request for wider compatibility with existing certificate authority software. ([#136729](https://github.com/kubernetes/kubernetes/pull/136729), [@ahmedtd](https://github.com/ahmedtd)) [SIG API Machinery, Auth, Node and Testing]
- Promoted several component-base metrics (`kubernetes_build_info`, `rest_client_requests_total`, `rest_client_request_duration_seconds`, `running_managed_controllers`) from Alpha to Beta stability, providing stronger API and label stability guarantees for consumers. ([#136154](https://github.com/kubernetes/kubernetes/pull/136154), [@bhope](https://github.com/bhope)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scalability, Scheduling, Storage and Testing]
- Replace deprecated sets.String with sets.Set[string] in apiserver admission subsystem. This is a breaking change for consumers of the NewLifecycle function. ([#134044](https://github.com/kubernetes/kubernetes/pull/134044), [@mcallzbl](https://github.com/mcallzbl)) [SIG API Machinery and Auth]
- Updates API server internal API group to improve openapi schema correctness for fields being optional or required ([#134675](https://github.com/kubernetes/kubernetes/pull/134675), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery, Apps, Auth, Node and Storage]

### Feature

- Add an alpha informer_processing_latency_seconds histogram metric to measure event handler execution time in RealFIFO. ([#137101](https://github.com/kubernetes/kubernetes/pull/137101), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Add the kubelet_metrics_provider metric to help users identify where kubelet's metrics are coming from. ([#136952](https://github.com/kubernetes/kubernetes/pull/136952), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Node]
- Added a show-secret flag to the diff command to explicitly allow secret values to be displayed during the diff operation. ([#137019](https://github.com/kubernetes/kubernetes/pull/137019), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Adds alpha metrics `apiserver_peer_proxy_errors_total` and `apiserver_peer_discovery_sync_errors_total` to apiserver to track errors encountered in peer proxying and peer discovery ([#137065](https://github.com/kubernetes/kubernetes/pull/137065), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Client-go: informer stores now keep track the resourceVersion they are synced to (via add/update/delete events, or replace calls, or bookmark events), and provide a `LastStoreSyncResourceVersion` method to obtain this resource version. This method can return `""` if the store has not been synced to yet, and depends on the AtomicFIFO feature being enabled. ([#134827](https://github.com/kubernetes/kubernetes/pull/134827), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Testing]
- Introduced PodGroup scheduling cycle in kube-scheduler to schedule entire PodGroup in one cycle. ([#136618](https://github.com/kubernetes/kubernetes/pull/136618), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- K8s.io/cloud-provider: Adds missing TLS flags to webhook serving options ([#136816](https://github.com/kubernetes/kubernetes/pull/136816), [@damdo](https://github.com/damdo)) [SIG Cloud Provider]
- Kube-controller-manager: The daemonset controller now defers syncing a DaemonSet object when the controller has not yet observed daemonset or pod writes from the last time the object was synced. This prevents spurious creation of duplicate pods for nodes when the controller's cache is stale. When a sync is deferred for this reason, a `daemonset_controller_stale_sync_skips_total` metric is incremented and a message is logged by the daemonset controller. This behavior can be temporarily disabled by setting the `StaleControllerConsistencyDaemonSet` feature gate to false. ([#134937](https://github.com/kubernetes/kubernetes/pull/134937), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Kube-controller-manager: The job controller now defers syncing a Job object when the controller has not yet observed job or pod writes from the last time the object was synced. This prevents spurious creation of duplicate pods for jobs when the controller's cache is stale. When a sync is deferred for this reason, a `job_controller_stale_sync_skips_total` metric is incremented and a message is logged by the job controller. This behavior can be temporarily disabled by setting the `StaleControllerConsistencyJob` feature gate to false. ([#137210](https://github.com/kubernetes/kubernetes/pull/137210), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Apps]
- Kubeadm: the preflight check `ContainerRuntimeVersion` validates if the installed container runtime supports the `RuntimeConfig` gRPC method. For older kubelet versions than 1.37, it will return a preflight warning. ([#136898](https://github.com/kubernetes/kubernetes/pull/136898), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubelet: defer the configurations flags (and the related fallback behavior) deprecation removal timeline from 1.36 to 1.37 to align with containerd v1.7 support ([#136846](https://github.com/kubernetes/kubernetes/pull/136846), [@carlory](https://github.com/carlory)) [SIG Node and Testing]
- Kubernetes is now built using Go 1.25.7 ([#136982](https://github.com/kubernetes/kubernetes/pull/136982), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Pods that are preempted when in PreBind phase will go back to backoff queue instead of being completely deleted from apisever. ([#135502](https://github.com/kubernetes/kubernetes/pull/135502), [@Argh4k](https://github.com/Argh4k)) [SIG Scheduling and Testing]
- Prevent the replicaset controller from spuriously reconciling while its own writes have not been read. ([#137212](https://github.com/kubernetes/kubernetes/pull/137212), [@michaelasp](https://github.com/michaelasp)) [SIG Apps]
- Updated feature gate `MutablePodResourcesForSuspendedJobs` and `MutableSchedulingDirectivesForSuspendedJobs` to be enabled by default. ([#135965](https://github.com/kubernetes/kubernetes/pull/135965), [@kannon92](https://github.com/kannon92)) [SIG Apps and Testing]

### Failing Test

- (reverts #136796, so ignore the release note from there...) ([#137169](https://github.com/kubernetes/kubernetes/pull/137169), [@danwinship](https://github.com/danwinship)) [SIG Network]

### Bug or Regression

- Extend unsupported Table object detection from watchlist only to all List and Watch operations. This prevents the reflector from processing resources returned in Table format which it cannot properly handle ([#136937](https://github.com/kubernetes/kubernetes/pull/136937), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Fix PodCertificateRequest OwnerReference using incorrect apiVersion "core/v1" instead of "v1", which prevented garbage collection of PodCertificateRequests when their owning Pod was deleted. ([#137008](https://github.com/kubernetes/kubernetes/pull/137008), [@srhppr](https://github.com/srhppr)) [SIG Auth and Node]
- Fix a bug where the DRA manager did not initialize sharedID from cache when DRAConsumableCapacity is enabled. ([#136734](https://github.com/kubernetes/kubernetes/pull/136734), [@sunya-ch](https://github.com/sunya-ch)) [SIG Node and Scheduling]
- Fixed /metrics/resource container_swap_usage_bytes to report the correct container swap usage ([#137098](https://github.com/kubernetes/kubernetes/pull/137098), [@yuanwang04](https://github.com/yuanwang04)) [SIG Apps, Node and Testing]
- Fixed a bug where `kubectl plugin list` failed to detect overshadowed plugins on Windows. ([#136689](https://github.com/kubernetes/kubernetes/pull/136689), [@kfess](https://github.com/kfess)) [SIG CLI]
- Fixed a bug where the event_handling_duration_seconds/preemption_goroutines_duration_seconds/run_podsandbox_duration_seconds/store_schedule_results_duration_seconds metric was recording
  near-zero latency values instead of actual value. ([#135749](https://github.com/kubernetes/kubernetes/pull/135749), [@novahe](https://github.com/novahe)) [SIG Architecture, Instrumentation, Node and Scheduling]
- Fixed a data race in k8s.io/apiserver/pkg/cel/openapi/resolver with (probably) no real-world impact. ([#136802](https://github.com/kubernetes/kubernetes/pull/136802), [@pohly](https://github.com/pohly)) [SIG API Machinery, Node and Testing]
- Fixed kubectl logs -f to wait for containers to start instead of failing 
  immediately when pods are in ContainerCreating or PodInitializing states ([#136411](https://github.com/kubernetes/kubernetes/pull/136411), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Fixes kube-proxy log spam when all of a Service's endpoints were unready. ([#136743](https://github.com/kubernetes/kubernetes/pull/136743), [@ansilh](https://github.com/ansilh)) [SIG Network]
- Fixes kube-proxy's nftables mode to work on systems with nft 1.1.3. ([#136796](https://github.com/kubernetes/kubernetes/pull/136796), [@kairosci](https://github.com/kairosci)) [SIG API Machinery, Auth and Network]
- Kubeadm: do not add learner member to etcd client endpoints ([#137251](https://github.com/kubernetes/kubernetes/pull/137251), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- The metrics `container_cpu_load_average_10s`, `container_cpu_load_d_average_10s`, and `cpu_tasks_state` have been dropped from the reported metrics by cadvisor. This is done because the values were always 0, because a flag was not enabled in the kubelet. ([#134981](https://github.com/kubernetes/kubernetes/pull/134981), [@haircommander](https://github.com/haircommander)) [SIG Node and Testing]

### Other (Cleanup or Flake)

- Apiserver_rerouted_request_total metric will expose labels for group, version and resource. ([#137063](https://github.com/kubernetes/kubernetes/pull/137063), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Removed the generally available feature gate `HonorPVReclaimPolicy`, which was locked and enabled since 1.33. ([#135335](https://github.com/kubernetes/kubernetes/pull/135335), [@carlory](https://github.com/carlory)) [SIG Apps and Storage]
- The deprecated SeparateCacheWatchRPC feature gate is now locked to its default value (false) and can no longer be overridden. The feature gate will be removed in a future release. ([#135808](https://github.com/kubernetes/kubernetes/pull/135808), [@tico88612](https://github.com/tico88612)) [SIG API Machinery]
- Update etcd images to v3.6.8 ([#137107](https://github.com/kubernetes/kubernetes/pull/137107), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Upgrades functionality of `kubectl kustomize` as described at https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.8.1 ([#136892](https://github.com/kubernetes/kubernetes/pull/136892), [@koba1t](https://github.com/koba1t)) [SIG Architecture and CLI]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.27.4 → v2.27.7](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.27.4...v2.27.7)
- github.com/sergi/go-diff: [v1.2.0 → v1.4.0](https://github.com/sergi/go-diff/compare/v1.2.0...v1.4.0)
- go.etcd.io/etcd/api/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/client/pkg/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/client/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/pkg/v3: v3.6.7 → v3.6.8
- go.etcd.io/etcd/server/v3: v3.6.7 → v3.6.8
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.64.0 → v0.65.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.63.0 → v0.65.0
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.64.0 → v0.65.0
- go.opentelemetry.io/contrib/propagators/b3: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/exporters/stdout/stdouttrace: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/metric: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/sdk/metric: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/sdk: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel/trace: v1.39.0 → v1.40.0
- go.opentelemetry.io/otel: v1.39.0 → v1.40.0
- google.golang.org/genproto/googleapis/api: 99fd39f → 8636f87
- google.golang.org/genproto/googleapis/rpc: 99fd39f → 8636f87
- k8s.io/utils: 914a6e7 → b8788ab
- sigs.k8s.io/kustomize/api: v0.20.1 → v0.21.1
- sigs.k8s.io/kustomize/cmd/config: v0.20.1 → v0.21.1
- sigs.k8s.io/kustomize/kustomize/v5: v5.7.1 → v5.8.1
- sigs.k8s.io/kustomize/kyaml: v0.20.1 → v0.21.1
- sigs.k8s.io/structured-merge-diff/v6: v6.3.1 → v6.3.2

### Removed
- github.com/pkg/errors: [v0.9.1](https://github.com/pkg/errors/tree/v0.9.1)



# v1.36.0-alpha.1


## Downloads for v1.36.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes.tar.gz) | 79ec354722859240b1d715c0da181e5a2a0fd354984ae9d511d58e7c09ec5cf7e54421db38a6d96b409b9a08b1c1b9dc13e7df20e7ab21299837d00cf72f162c
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-src.tar.gz) | f274bff791b16bb4de10730aabbfc027220f45c44d2dc8d1a8b575cc86421ec01fb106bcb2f3cb137145e64396ca37f2ec689932395162dcae5d3b6b65fc97ec

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 6fc2c7b184ee6435c0e7179dbe8ff63549631d9a5eb28262b10596a6f26e245ab2cf16402a6466e37b81a42760f811808796d1e83dd205125c0e64e1330772bd
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | e56eb183ba431d530b6cbd83ee94c1c398f3f4969cdee247092738a5cbe2b567d705788f95adec2f13cf17ebd791165903a0f1fcac9fcbf36ed65b9a00f38ac3
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 1a65eb81a4cf1631fa6fb102f2dfcffd29732bb96479c0432c9780f2dcb4600f8c85991b0f68f038ae963348d39291c74a855a71705093ecc09218ee4ed5271c
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | a1824ed2091dac2289c99c67e504b01b0176657675496752136a630ecf57347e0a5578a21c3bd74d0baad995d1e99ba0c78b5dfc2a61316227171c25a216111b
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 9e368482df69b6990c917d6df0f3589851d72bcd7304226970eae32898baeb76e5955e93bda577d72231ca2342562ab91dfecb19dd5479f91352a4577f8f7d92
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 7894d5868aa7888a26648ea338fbe63031e2b3ce0919a337e69cba002c369a0bbe6971ed7add1fd5e2284ebc696f9bb9c0180c298fe349140482b2e93a51d72a
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | ae379c93762b86b8ddd1f1076b2e37c2866991e5a5700eb08ff5b65aaeca552764c6bb0506376b502e67f1734f5655b59ca4aa751b572667a9522065822874f1
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 316d1093cd109d91b55c5ca18a8aa2d0e04feb29bb688b3d52018818329519c8fa6b226de8505de3ced920c064b8f150dccb2829d2bb2a4bed594e4d663377cd
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 48f7ca49e081393474c644c31a7bae8810dfc7673c2f1800207960ea14e73616e1b7717d312e4787e8b5179c5a46a256cdbaa80e01cdf37cf999f370a17060e1
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | d70670136a91a5f81f7b030f258881917371102e958382d3dfef425d5d76718cc242d7ee86dbaaaa5705e019b4178e323a29cc0d4ace6d803550ea8d412189e2
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 99c08b44870989a9629573f110317215e4ec177a25298e28f497deb75867838085b987a100c8ee678ccf1345e4cef075d159c5cb9198a3a342a92733309820d5

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | 9208265b86d2d7ebf8dd8f771a586de571aa07ab54c1d7428deb8803dbf63f2e396e30b103cac4da8ddf791be8b66dfce90e428906ec2d59e485953a6b6e1b7b
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 704ce893ee7b239194aae37b608b175940efbf60072622a661d866221f6fdba9c6b6c4ca11008bf2be1cf2c70cb9f6e5aece9052efa2468c1e3b73510c35d2f0
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 99cc1cf4169b4c8a00b7e7e4f49c98703e0191dd0aa0e4f641a0b35a92e7d5df14587d5b56f571fa4d00290d806bbf916cb3d934537883f9364a04d66d0ba958
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | ea9ac2489aa2b8d9a0bd8a8a12958b754edd0c9227aaee23dadf7fc7711cab5033edcb85dc545e6bd8cb78336e073b2cfb8ed8655932b91a4d8d22c3eb3dfb90

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 1cff3dd843e8ffbe2728967c3520bb000a551573fc9049fb4d3e6734d76a9ea72a2e54a0eca18bf68b0f023106714a877fb01b2e942720b707f738df1709361d
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 55a65f51bc9d25ca2993bf2d277e1ceadc585d8af90e3f92b9b6fd682b7fa78f8fd0bb4bba127143825c70227d8815b037159feb8a319c0b4fb406e3b4dfa913
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | e67d17e2d38716e6d20a777d360454f0a0e04d8f5dc94f8888c823df55ddb96973e819229e043d0e9a7f25027f4f64583f13443d633f568b62237e1fcceae8b4
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 807376145bb55d7b534523c18fac4f61ddb8d20e0646c2152de28cd547f1ef82787f01e8b43a7ea1128a9c76c7ee14183e28629032396ff124629c1a6c6c9ab4
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.36.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 1f204db26af19933504f50292513ea3e238f57ac9bc8c4faad8a71271cd2c2920515ad0267ed87618ac4d6f510f111fe5a2feeebe084425b0f369905f577cac8

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.36.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.35.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Added support for running PreBind plugins in parallel in the scheduler framework to improve the binding latency.
  Plugins can now opt-in to parallel execution by returning `AllowParallel: true` from the PreBindPreFlight method. PreBind plugin implementations need to be updated to return the PreBindPreFlightResult from the PreBindPreFlight method; returning nil retains the existing sequential behavior. ([#135393](https://github.com/kubernetes/kubernetes/pull/135393), [@tosi3k](https://github.com/tosi3k)) [SIG Node, Scheduling, Storage and Testing]
 
## Changes by Kind

### Dependency

- Fix a bug where pod lifecycle hooks could run for their full duration when pods are terminated. ([#136598](https://github.com/kubernetes/kubernetes/pull/136598), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]

### API Change

- Add --concurrent-resourceclaim-syncs to configure kube-controller-manager resource claim reconcile concurrency ([#134701](https://github.com/kubernetes/kubernetes/pull/134701), [@anson627](https://github.com/anson627)) [SIG API Machinery, Apps, Node and Testing]
- Added negative duration validation for imageMinimumGCAge ([#135997](https://github.com/kubernetes/kubernetes/pull/135997), [@ngopalak-redhat](https://github.com/ngopalak-redhat)) [SIG API Machinery and Node]
- Clarified documentation and comments to indicate that the `cpuCFSQuotaPeriod` kubelet config field requires the `CustomCPUCFSQuotaPeriod` feature gate when using non-default values. No functional changes introduced. ([#133845](https://github.com/kubernetes/kubernetes/pull/133845), [@rbiamru](https://github.com/rbiamru)) [SIG Node and Release]
- Correct openapi schema union validation for the PodGroupPolicy struct in scheduling v1alpha1 ([#136424](https://github.com/kubernetes/kubernetes/pull/136424), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery and Scheduling]
- Fixed a potential nil pointer dereference in the scheduler's NodeResourcesFitArgs validation when using RequestedToCapacityRatio scoring strategy ([#132120](https://github.com/kubernetes/kubernetes/pull/132120), [@flpanbin](https://github.com/flpanbin)) [SIG Scheduling]
- Fixes `fake.NewClientset()` to work properly with correct schema. ([#131068](https://github.com/kubernetes/kubernetes/pull/131068), [@soltysh](https://github.com/soltysh)) [SIG API Machinery]
- Generate fake.NewClientset which replace the deprecated NewSimpleClientset, for kube-aggregator and sample-apiserver ([#136537](https://github.com/kubernetes/kubernetes/pull/136537), [@soltysh](https://github.com/soltysh)) [SIG API Machinery]
- Graduate watch_list_duration_seconds from ALPHA to BETA ([#136086](https://github.com/kubernetes/kubernetes/pull/136086), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation, Node and Testing]
- Kube-apiserver: the `--audit-policy-file` config file now supports specifying `group: "*"` in resources rules to match all API groups ([#135262](https://github.com/kubernetes/kubernetes/pull/135262), [@cmuuss](https://github.com/cmuuss)) [SIG API Machinery, Auth and Testing]
- Kube-controller-manager: alpha gauge metrics for informer queue length are now published as `informer_queued_items{name=kube-controller-manager,group=<group>,resource=<resource>,version=<version>} <count>` ([#135782](https://github.com/kubernetes/kubernetes/pull/135782), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Architecture, Instrumentation and Testing]
- Locked the feature-gate VolumeAttributesClass to default (true) and bump VolumeAttributesClass preferred storage version to `storage.k8s.io/v1` ([#134556](https://github.com/kubernetes/kubernetes/pull/134556), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Etcd, Network, Node, Scheduling, Storage and Testing]
- Promote workqueue metrics from ALPHA to BETA ([#135522](https://github.com/kubernetes/kubernetes/pull/135522), [@petern48](https://github.com/petern48)) [SIG Architecture, Instrumentation and Testing]
- Removed the generally available feature gate `CSIMigrationPortworx`, which was locked and enabled since 1.33.
  - Removed alpha feature gate `InTreePluginPortworxUnregister`
  - Removed Portworx volume plugin from in-tree plugins because all operations are redirected to CSI. ([#135322](https://github.com/kubernetes/kubernetes/pull/135322), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Auth, Node, Scalability, Scheduling, Storage and Testing]
- The ImageVolumeWithDigest is added which adds the digest of image volumes to the container's status. ([#132807](https://github.com/kubernetes/kubernetes/pull/132807), [@iholder101](https://github.com/iholder101)) [SIG API Machinery, Apps, Node and Testing]
- The `endpoints` field in discovery.k8s.io/v1 EndpointSlice is now correctly defined as optional in the OpenAPI specification, matching the server's behavior. ([#136111](https://github.com/kubernetes/kubernetes/pull/136111), [@aojea](https://github.com/aojea)) [SIG Network]
- Update API comments to reflect that stable state of Dynamic Resource Allocation ([#136441](https://github.com/kubernetes/kubernetes/pull/136441), [@kannon92](https://github.com/kannon92)) [SIG API Machinery]

### Feature

- Add architecture to the kernel version column in the `kubectl get node -owide` output. ([#132402](https://github.com/kubernetes/kubernetes/pull/132402), [@astraw99](https://github.com/astraw99)) [SIG CLI]
- Add the `appProtocol` field to the service describe output. ([#135744](https://github.com/kubernetes/kubernetes/pull/135744), [@ali-a-a](https://github.com/ali-a-a)) [SIG CLI]
- Add write and read permissions for workloads to the admin cluster role. Add write permissions for workloads to the edit cluster role. Add read permissions for workloads to the view cluster role. ([#135418](https://github.com/kubernetes/kubernetes/pull/135418), [@carlory](https://github.com/carlory)) [SIG Auth]
- Added ALPHA metric `scheduler_pod_scheduled_after_flush_total` to count pods scheduled after being flushed from unschedulablePods due to timeout ([#135126](https://github.com/kubernetes/kubernetes/pull/135126), [@mrvarmazyar](https://github.com/mrvarmazyar)) [SIG Scheduling]
- Added kubectl explain -r flag as a shorthand for --recursive ([#135283](https://github.com/kubernetes/kubernetes/pull/135283), [@laervn](https://github.com/laervn)) [SIG CLI]
- Align the meaning of victim metrics between async preemption and sync preemption. The definition has been standardized to refer to the number of Pods chosen as victims. ([#135955](https://github.com/kubernetes/kubernetes/pull/135955), [@utam0k](https://github.com/utam0k)) [SIG Scheduling]
- CRD validation now strictly enforces ranges for numeric formats (int32, int64, float, double) when specified in the schema. Existing objects with out-of-range values are preserved via validation ratcheting ([#136582](https://github.com/kubernetes/kubernetes/pull/136582), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling and Storage]
- Change the default debug profile from `legacy` to `general`. `legacy` profile is planned to be removed in v1.39. ([#135874](https://github.com/kubernetes/kubernetes/pull/135874), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI and Testing]
- Client-go informers can now enqueue new watch events while already-queued events are being processed. This avoids dropping watches during a burst of incoming events due to contention on slow processing. This behavior is controlled by the UnlockWhileProcessing client-go feature gate, which is enabled by default. ([#136264](https://github.com/kubernetes/kubernetes/pull/136264), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery and Scheduling]
- Client-go: Informer resync processing improved handling of Resync handling. This reduces contention on store locks between incoming events and handler updates, which may result in observable timing differences of handler invocations. This behavior is guarded by an AtomicFIFO feature gate. This gate is enabled by default in 1.36, but can be disabled if needed to temporarily regain the previous behavior. ([#136008](https://github.com/kubernetes/kubernetes/pull/136008), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Client-go: default informer behavior now updates store state with all the objects in a list or relist, before calling handler OnDelete/OnAdd/OnUpdate methods for individual items which were deleted/added/removed. This ensures that the store state which can be inspected by handlers actually corresponds to a set of objects that existed at a particular resource version on the server. This behavior is guarded by an AtomicFIFO feature gate. This gate is enabled by default in 1.36, but can be disabled if needed to temporarily regain the previous behavior. ([#135462](https://github.com/kubernetes/kubernetes/pull/135462), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Cloud Controller Manager now exports the counter metric `route_controller_route_sync_total`, which increments each time routes are synced with the cloud provider. This metric is in alpha stage. ([#136539](https://github.com/kubernetes/kubernetes/pull/136539), [@lukasmetzner](https://github.com/lukasmetzner)) [SIG API Machinery, Cloud Provider and Instrumentation]
- Enable WatchCacheInitializationPostStartHook by default ([#135777](https://github.com/kubernetes/kubernetes/pull/135777), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Graduated fine-grained kubelet API authorization to stable. ([#136116](https://github.com/kubernetes/kubernetes/pull/136116), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Node]
- ImageLocality plugin: consider ImageVolume images when scoring nodes for pod scheduling. ([#130231](https://github.com/kubernetes/kubernetes/pull/130231), [@Barakmor1](https://github.com/Barakmor1)) [SIG Scheduling]
- Kube-apiserver: Promoted `ExternalServiceAccountTokenSigner` feature to GA. ([#136118](https://github.com/kubernetes/kubernetes/pull/136118), [@HarshalNeelkamal](https://github.com/HarshalNeelkamal)) [SIG API Machinery and Auth]
- Kubeadm: Upgraded the `NodeLocalCRISocket` feature gating to GA and locked it to be enabled. ([#135742](https://github.com/kubernetes/kubernetes/pull/135742), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Kubeadm: added the flag --allow-deprecated-api to 'kubeadm config validate'. By default the command will print a warning for a deprecated API unless the flag is passed. Additionally, added missing support for v1beta4 UpgradeConfiguration to 'kubeadm config migrate|validate' commands. ([#135148](https://github.com/kubernetes/kubernetes/pull/135148), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: bumped the timeout of the `kubeadm upgrade` `CreateJob` preflight check to 1 minute. This allows Windows worker nodes to have more time to run the preflight check. It uses the `pause` image, so if you are experiencing slow pull times, you can either pre-pull the new pause on the work using `kubeadm config images pull --kubernetes-version TARGET` or skip the preflight check with `--ignore-preflight-errors`. ([#136273](https://github.com/kubernetes/kubernetes/pull/136273), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: removed the kubeadm specific feature gate ControlPlaneKubeletLocalMode which became GA in 1.35 and was locked to enabled. ([#135773](https://github.com/kubernetes/kubernetes/pull/135773), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: when patching a Node object do not exit early on unknown (non-allowlisted) API errors. Instead, always retry within the duration of the polling for getting and patching a Node object. ([#135776](https://github.com/kubernetes/kubernetes/pull/135776), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl get ingressclass now displays (default) marker for default IngressClass ([#134422](https://github.com/kubernetes/kubernetes/pull/134422), [@jaehanbyun](https://github.com/jaehanbyun)) [SIG CLI and Network]
- Kubernetes is now built using Go 1.25.6 ([#136465](https://github.com/kubernetes/kubernetes/pull/136465), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with Go 1.25.6 ([#136257](https://github.com/kubernetes/kubernetes/pull/136257), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Kubernetes is now built with Go 1.25.7 ([#136750](https://github.com/kubernetes/kubernetes/pull/136750), [@BenTheElder](https://github.com/BenTheElder)) [SIG Release]
- Promote Relaxed validation for Services names to beta (enabled by default)
  
  Promote `RelaxedServiceNameValidation` feature to beta (enabled by default)
  The names of new Services names are validation with `NameIsDNSLabel()`,
  relaxing the pre-existing validation. ([#136389](https://github.com/kubernetes/kubernetes/pull/136389), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network]
- Promoted the `CSIServiceAccountTokenSecrets` feature gate to GA. ([#136596](https://github.com/kubernetes/kubernetes/pull/136596), [@aramase](https://github.com/aramase)) [SIG Auth and Storage]
- Promoting kubectl kuberc commands to beta ([#136643](https://github.com/kubernetes/kubernetes/pull/136643), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- The ResourceClaim controller now correctly handles unknown (non-pod) references in the status.reservedFor field by skipping them instead of halting the sync process. ([#136450](https://github.com/kubernetes/kubernetes/pull/136450), [@MohammedSaalif](https://github.com/MohammedSaalif)) [SIG Apps and Node]
- Update to latest cAdvisor 0.55.0 in our vendor dependencies ([#135829](https://github.com/kubernetes/kubernetes/pull/135829), [@dims](https://github.com/dims)) [SIG Node]
- Using pytorch based e2e integration test instead of tensorflow in some node e2e CI tests. ([#136397](https://github.com/kubernetes/kubernetes/pull/136397), [@dims](https://github.com/dims)) [SIG Testing]
- Using pytorch based e2e integration test instead of tensorflow in some node e2e CI tests. ([#136398](https://github.com/kubernetes/kubernetes/pull/136398), [@dims](https://github.com/dims)) [SIG Node and Testing]

### Failing Test

- Fixed device plugin test failures after kubelet restart. ([#135485](https://github.com/kubernetes/kubernetes/pull/135485), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node and Testing]

### Bug or Regression

- Added extra check to prevent users to work around DRA extended resource quota set by system admin ([#135434](https://github.com/kubernetes/kubernetes/pull/135434), [@yliaog](https://github.com/yliaog)) [SIG API Machinery, Apps, Node, Scheduling and Testing]
- Aligned `kubectl label` output message to include 'modified' when labels are both added and removed ([#134849](https://github.com/kubernetes/kubernetes/pull/134849), [@tchap](https://github.com/tchap)) [SIG CLI]
- Apiserver liveness probes will now fail when the loopback client certificate expires. ([#136477](https://github.com/kubernetes/kubernetes/pull/136477), [@everettraven](https://github.com/everettraven)) [SIG API Machinery and Testing]
- Changed the behavior of default scheduler preemption plugin when preempting pods that are in "WaitOnPermit" phase. They are now moved to the scheduler backoff queue instead of being marked as unschedulable. ([#135719](https://github.com/kubernetes/kubernetes/pull/135719), [@Argh4k](https://github.com/Argh4k)) [SIG Scheduling and Testing]
- Changes some instances of error logs to info logs with verbosity level inside of controller/resourcequota and controller/garbagecollector ([#136040](https://github.com/kubernetes/kubernetes/pull/136040), [@petern48](https://github.com/petern48)) [SIG API Machinery and Apps]
- Changes the nodeGetCapabilities method of csiDriverClient returning NewUncertainProgressError while received a non final GRPC error ([#135930](https://github.com/kubernetes/kubernetes/pull/135930), [@249043822](https://github.com/249043822)) [SIG Node and Storage]
- Client-go informers: fix an unlikely deadlock during informer startup. ([#136509](https://github.com/kubernetes/kubernetes/pull/136509), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- DRA: when scheduling many pods very rapidly, sometimes the same device was allocated twice for different ResourceClaims due races between data processing in different goroutines. Depending on whether DRA drivers check for this during NodePrepareResources (they should, but maybe not all implement this properly), the second pod using the same device then failed to start until the first one is done or (worse) ran in parallel. ([#136269](https://github.com/kubernetes/kubernetes/pull/136269), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Disabled `SchedulerAsyncAPICalls` feature gate due to performance issues caused by API client throttling. ([#135903](https://github.com/kubernetes/kubernetes/pull/135903), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Ensures a couple of feature gates - ChangeContainerStatusOnKubeletRestart and StatefulSetSemanticRevisionComparison are visible from the "--help" in different components ([#135515](https://github.com/kubernetes/kubernetes/pull/135515), [@dims](https://github.com/dims)) [SIG Architecture]
- Fix a nil pointer dereference in Kubelet when handling pod updates of mirror pods with the NodeDeclaredFeatures feature gate enabled. ([#136037](https://github.com/kubernetes/kubernetes/pull/136037), [@pravk03](https://github.com/pravk03)) [SIG Node]
- Fix apiserver request latency annotation in the audit log when request took more than 500ms ([#135685](https://github.com/kubernetes/kubernetes/pull/135685), [@chaochn47](https://github.com/chaochn47)) [SIG API Machinery]
- Fix data race in kubelet container manager. ([#136206](https://github.com/kubernetes/kubernetes/pull/136206), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Fix data race in kubelet pod allocated resources. ([#136226](https://github.com/kubernetes/kubernetes/pull/136226), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Fix data race in kubelet status manager. ([#136205](https://github.com/kubernetes/kubernetes/pull/136205), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Fix issues where server side apply patches operations incorrectly treat empty arrays and maps as absent.
  Fix issue where client-go's `Extract{TypeName}()` and `Extract{TypeName}From() functions incorrectly treat empty arrays and maps as absent.
  Fix issue where client-go's `Extract{TypeName}()` and `Extract{TypeName}From() functions would incorrectly duplicate atomic elements from associative lists. ([#135391](https://github.com/kubernetes/kubernetes/pull/135391), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Network, Node, Scheduling and Storage]
- Fix log verbosity level in apiserver's unsafe delete authorization check that was incorrectly using Error level instead of Info level ([#136229](https://github.com/kubernetes/kubernetes/pull/136229), [@thc1006](https://github.com/thc1006)) [SIG API Machinery]
- Fix queue hint for the interpodaffinity plugin in case target pod labels change ([#135394](https://github.com/kubernetes/kubernetes/pull/135394), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fix static pod status is always Init:0/1 if unable to get init container status from container runtime. ([#131317](https://github.com/kubernetes/kubernetes/pull/131317), [@bitoku](https://github.com/bitoku)) [SIG Node and Testing]
- Fix the log verbosity level of some non-error logs that were incorrectly logged at error level ([#136046](https://github.com/kubernetes/kubernetes/pull/136046), [@Tanner-Gladson](https://github.com/Tanner-Gladson)) [SIG API Machinery and Apps]
- Fix the log verbosity level of some non-error logs that were incorrectly logged at error level ([#136050](https://github.com/kubernetes/kubernetes/pull/136050), [@ShaanveerS](https://github.com/ShaanveerS)) [SIG Apps and Storage]
- Fixed SELinux warning controller not to emit events for completed pods. ([#135629](https://github.com/kubernetes/kubernetes/pull/135629), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Storage and Testing]
- Fixed a bug causing clients to error out when decoding large CBOR encoded lists. ([#135340](https://github.com/kubernetes/kubernetes/pull/135340), [@ricardomaraschini](https://github.com/ricardomaraschini)) [SIG API Machinery]
- Fixed a bug in DeepEqualWithNilDifferentFromEmpty where empty slices/maps were incorrectly considered equal to non-empty ones due to using OR (||) instead of AND (&&) logic. This could cause managed fields timestamps to not update when the only change was adding or removing all elements from a list or map. ([#135636](https://github.com/kubernetes/kubernetes/pull/135636), [@mikecook](https://github.com/mikecook)) [SIG API Machinery]
- Fixed a bug in the `dra_operations_duration_seconds` metric where the `is_error` label was recording inverted values. Error operations now correctly report `is_error="true"`, and successful operations report `is_error="false"`. ([#135227](https://github.com/kubernetes/kubernetes/pull/135227), [@hime](https://github.com/hime)) [SIG Node]
- Fixed a bug that caused endpoint slice churn for headless services with no ports defined (#133474) ([#136502](https://github.com/kubernetes/kubernetes/pull/136502), [@tzneal](https://github.com/tzneal)) [SIG Network]
- Fixed a bug where `kubectl apply --dry-run=client` would only output server state instead of merged manifest values when the resource already exists. ([#135513](https://github.com/kubernetes/kubernetes/pull/135513), [@grandeit](https://github.com/grandeit)) [SIG CLI]
- Fixed a bug where the Gated pods metric was not updated when a Pod transitioned from Unschedulable to Gated during an update. ([#135368](https://github.com/kubernetes/kubernetes/pull/135368), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling]
- Fixed a bug where the `scheduler_unschedulable_pods` metric could be artificially inflated (leak) when a pod fails `PreEnqueue` plugins after being previously marked unschedulable. ([#135981](https://github.com/kubernetes/kubernetes/pull/135981), [@vshkrabkov](https://github.com/vshkrabkov)) [SIG Scheduling]
- Fixed a panic in `kubectl exec` when the terminal size queue delegate is uninitialized. ([#135918](https://github.com/kubernetes/kubernetes/pull/135918), [@MarcosDaNight](https://github.com/MarcosDaNight)) [SIG CLI]
- Fixed a panic in kubectl when processing pods with nil resource requests but populated container status resources. ([#136534](https://github.com/kubernetes/kubernetes/pull/136534), [@dmaizel](https://github.com/dmaizel)) [SIG CLI]
- Fixed a race condition in the CEL compiler that could occur when initializing composited policies concurrently. 
  
  ### Description
  Fixes a fatal crash (concurrent map read/write) in `NewCompositedCompilerFromTemplate`.
  
  The `NewCompositedCompilerFromTemplate` function previously performed a shallow copy of `CompositionEnv`, sharing the `MapType` pointer across all compilers. Under high concurrency, this caused a race condition when `FindStructFieldType` (reader) and `AddField` (writer) accessed `MapType.Fields` simultaneously, leading to an APIServer panic.
  
  This change implements a deep copy of the `Fields` map for each composition environment, ensuring thread safety.
  
  ### Issue
  Fixes #135757 ([#135759](https://github.com/kubernetes/kubernetes/pull/135759), [@Abhigyan-Shekhar](https://github.com/Abhigyan-Shekhar)) [SIG API Machinery and CLI]
- Fixed an issue in the Windows kube-proxy (winkernel) where IPv4 and IPv6 Service load balancers could be incorrectly shared, causing broken dual-stack Service behavior. The kube-proxy now tracks load balancers per IP family, enabling correct support for PreferDualStack and RequireDualStack Services on Windows nodes. ([#136241](https://github.com/kubernetes/kubernetes/pull/136241), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]
- Fixed issue where `kubectl run -i/-it` would miss container output written before the attach connection was established. ([#136010](https://github.com/kubernetes/kubernetes/pull/136010), [@olamilekan000](https://github.com/olamilekan000)) [SIG CLI]
- Fixed kubelet logging to properly respect verbosity levels. Previously, some debug/info messages using V().Error() would always be printed regardless of the configured log verbosity. ([#136028](https://github.com/kubernetes/kubernetes/pull/136028), [@thc1006](https://github.com/thc1006)) [SIG Node]
- Fixed queue hint for certain plugins on change to pods with nominated nodes ([#135392](https://github.com/kubernetes/kubernetes/pull/135392), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Fixed queue hint for inter-pod anti-affinity in case deleted pod's anti-affinity matched the pending pod, which might have caused delays in scheduling. ([#135325](https://github.com/kubernetes/kubernetes/pull/135325), [@brejman](https://github.com/brejman)) [SIG Scheduling and Testing]
- Fixed volumeattachment cleanup in kube-controller-manager when CSI's attachRequired switches from true to false ([#129664](https://github.com/kubernetes/kubernetes/pull/129664), [@hkttty2009](https://github.com/hkttty2009)) [SIG Storage and Testing]
- Fixes a 1.29 regression in the apiserver_watch_events_sizes metric to report total outgoing watch traffic again ([#135367](https://github.com/kubernetes/kubernetes/pull/135367), [@mborsz](https://github.com/mborsz)) [SIG API Machinery]
- Fixes a 1.34 regression starting pods with environment variables with a value containing `$` followed by a multi-byte character ([#136325](https://github.com/kubernetes/kubernetes/pull/136325), [@AutuSnow](https://github.com/AutuSnow)) [SIG Architecture]
- Fixes a 1.34+ regression in ipvs and winkernel kube-proxy backends; these are now reverted back to their
  pre-1.34 behavior of regularly rechecking all of their rules even when no
  Services or EndpointSlices change. ([#135631](https://github.com/kubernetes/kubernetes/pull/135631), [@danwinship](https://github.com/danwinship)) [SIG Network and Windows]
- Fixes kube-proxy log spam when all of a Service's endpoints were unready. ([#136743](https://github.com/kubernetes/kubernetes/pull/136743), [@ansilh](https://github.com/ansilh)) [SIG Network]
- Kube-apiserver: setting `--audit-log-maxsize=0` now disables audit log rotation (the default remains `100` MB). In order to avoid outages due to filling disks with ever-growing audit logs, `--audit-log-maxage` now defaults to 366 (1 year) and `--audit-log-maxbackup` now defaults to 100. If retention of all rotated logs is desired, age and count-based pruning can be disabled by explicitly specifying `--audit-log-maxage=0` and `--audit-log-maxbackup=0`. ([#136478](https://github.com/kubernetes/kubernetes/pull/136478), [@kairosci](https://github.com/kairosci)) [SIG API Machinery]
- Kube-proxy now correctly handles the case where a pod IP gets assigned to
  a newly-created pod when the pod that previously had that IP has been
  terminated but is not yet fully deleted. ([#135593](https://github.com/kubernetes/kubernetes/pull/135593), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kubeadm: fix a bug where kubeadm upgrade is failed if the content of the `kubeadm-flags.env` file is `KUBELET_KUBEADM_ARGS=""` ([#136127](https://github.com/kubernetes/kubernetes/pull/136127), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: waiting for etcd learner member to be started before promoting during 'kubeadm join' ([#136014](https://github.com/kubernetes/kubernetes/pull/136014), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: when applying the overrides provided by the user using "extraArgs", do not sort the resulted list of arguments alpha-numerically. Instead, only sort the list of default arguments and keep the list of overrides unsorted. This allows finer control for flags which have an order that matters, such as, "--service-account-issuer" for kube-apiserver. ([#135400](https://github.com/kubernetes/kubernetes/pull/135400), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl: fixes kyaml output of `kubectl get ... --output-watch-events -o kyaml` ([#136110](https://github.com/kubernetes/kubernetes/pull/136110), [@liggitt](https://github.com/liggitt)) [SIG CLI]
- Kubelet(dra): correctly handles multiple ResourceClaims even if one is already prepared ([#135919](https://github.com/kubernetes/kubernetes/pull/135919), [@rogowski-piotr](https://github.com/rogowski-piotr)) [SIG Node and Testing]
- Kubelet: fix data race in volume manager's WaitForAllPodsUnmount that could cause errors to be lost during concurrent pod unmount operations. ([#135794](https://github.com/kubernetes/kubernetes/pull/135794), [@AutuSnow](https://github.com/AutuSnow)) [SIG Node and Storage]
- Kubelet: fixed reloading of kubelet server certificate files when they are changed on disk, and kubelet is dialed by IP address instead of DNS/hostname ([#133654](https://github.com/kubernetes/kubernetes/pull/133654), [@kwohlfahrt](https://github.com/kwohlfahrt)) [SIG API Machinery, Auth, Node and Testing]
- Optimized kube-proxy conntrack cleanup logic, reducing the time complexity of deleting stale UDP entries. This significantly improves performance when there are many stale connections to clean up. ([#135511](https://github.com/kubernetes/kubernetes/pull/135511), [@aojea](https://github.com/aojea)) [SIG Network]
- ReadWriteOncePod preemption e2e test no longer causes other random e2e tests to flake randomly. ([#135623](https://github.com/kubernetes/kubernetes/pull/135623), [@jsafrane](https://github.com/jsafrane)) [SIG Storage and Testing]
- Sort runtime handlers list coming from the CRI runtime ([#135358](https://github.com/kubernetes/kubernetes/pull/135358), [@harche](https://github.com/harche)) [SIG Node]
- StatefulSets should always count `.status.availableReplicas` at the correct time without a delay. This results in faster progress of StatefulSet rollout. ([#135428](https://github.com/kubernetes/kubernetes/pull/135428), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- The kubelet plugin manager now properly handles plugin registration failures by removing failed plugins from the actual state and retrying with exponential backoff (initial delay 500ms, doubling each failure up to ~2 minutes maximum) to protect against broken plugins causing denial of service while still allowing recovery from transient failures. ([#133335](https://github.com/kubernetes/kubernetes/pull/133335), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Storage and Testing]
- The nftables mode of kube-proxy now uses less CPU when loading
  very large rulesets. ([#135800](https://github.com/kubernetes/kubernetes/pull/135800), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Updated `NodeResourcesBalancedAllocation` scoring algorithm to align with the documentation. The score will now take into consideration both balance with and without the requested pod. Previous algorithm only considered balance with the requested pod. This can change the scheduling decisions in some cases. ([#135573](https://github.com/kubernetes/kubernetes/pull/135573), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- When use kubectl command to delete multiple sts pods, the kubectl command deletes pods and exits normally. ([#135563](https://github.com/kubernetes/kubernetes/pull/135563), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG CLI, Network and Node]

### Other (Cleanup or Flake)

- Added missing tests for client-go metrics ([#136052](https://github.com/kubernetes/kubernetes/pull/136052), [@sreeram-venkitesh](https://github.com/sreeram-venkitesh)) [SIG Architecture and Instrumentation]
- Adds audit-id to 'Starting watch' log line ([#136084](https://github.com/kubernetes/kubernetes/pull/136084), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Adds explicit logging when WatchList requests complete their initial listing phase. ([#136085](https://github.com/kubernetes/kubernetes/pull/136085), [@richabanker](https://github.com/richabanker)) [SIG API Machinery]
- Client-go: Reflector no longer gets confused about the resource version it should use to restart a watch while receiving synthetic ADDED events at the beginning of a watch from resourceVersion "0" or "". ([#136583](https://github.com/kubernetes/kubernetes/pull/136583), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Client-go: fake client-go (i.e. anything using k8s.io/client-go/testing) now supports separate List+Watch calls  with checking of ResourceVersion in the Watch call. This closes a race condition where creating an object directly after an informer cache has synced (= List call completed) and before the Watch call completed would cause that object to not be sent to the informer. A visible side-effect of adding that support is that List meta data contains a ResourceVersion (starting at "1" for the empty set, incremented by one  for each add/update) and that Watch may return objects where it previously didn't.
  
  Note that this List+Watch is not to be confused with the ListWatch feature, which uses a single call. That feature is still not supported by fake client-go. ([#136143](https://github.com/kubernetes/kubernetes/pull/136143), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth and CLI]
- DRA device taint eviction: the controller might have reported "1 pod needs to be evicted in 1 namespace. 1 pod evicted since starting the controller." when only a single pod is involved, depending on timing (pod evicted, informer cache not updated yet). It would eventually arrive at the correct "1 pod evicted since starting the controller.", but now it tries harder to avoid the confusing intermediate state by delaying the status update after eviction. ([#135611](https://github.com/kubernetes/kubernetes/pull/135611), [@Karthik-K-N](https://github.com/Karthik-K-N)) [SIG Apps and Scheduling]
- DRA: Fixed Kubelet admission to correctly handle DRA-backed extended resources, allowing pods to be admitted even when these resources are not present in the node's allocatable capacity. ([#135725](https://github.com/kubernetes/kubernetes/pull/135725), [@bart0sh](https://github.com/bart0sh)) [SIG Node, Scheduling and Testing]
- Enables YAML support for statusz and flagz. ([#135309](https://github.com/kubernetes/kubernetes/pull/135309), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Instrumentation and Testing]
- Kubeadm: removed the cleanup of the "--pod-infra-container-image" kubelet flag from the "/var/lib/kubelet/kubeadm-flags.env" on upgrade. This cleanup was necessary when upgrading to 1.35. ([#135807](https://github.com/kubernetes/kubernetes/pull/135807), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: removed usage of the deprecated flags '--experimental-initial-corrupt-check' and '--experimental-watch-progress-notify-interval' if the etcd version is < 3.6.0. In this version of kubeadm, etcd < 3.6.0 is no longer supported in terms of the k8s / etcd version mapping. These deprecated flags have been replaced by '--feature-gates=InitialCorruptCheck=true' and '--watch-progress-notify-interval'. ([#135701](https://github.com/kubernetes/kubernetes/pull/135701), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Lock the `DisableNodeKubeProxyVersion` feature gate to be enabled by default. ([#136673](https://github.com/kubernetes/kubernetes/pull/136673), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG CLI and Network]
- Remove WatchFromStorageWithoutResourceVersion feature gate ([#136066](https://github.com/kubernetes/kubernetes/pull/136066), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Remove event listing behavior when describing a removed pod from file. ([#135281](https://github.com/kubernetes/kubernetes/pull/135281), [@scaliby](https://github.com/scaliby)) [SIG CLI]
- Renamed PodGroupInfo to PodGroupState, which can break custom scheduler plugins that use Handle.WorkloadManager ([#136344](https://github.com/kubernetes/kubernetes/pull/136344), [@brejman](https://github.com/brejman)) [SIG Scheduling]
- Set InOrderInformers to GA via the usage of RealFIFO, this means that DeltaFIFO will gradually be deprecated in favor of RealFIFO in internal implementations. ([#136601](https://github.com/kubernetes/kubernetes/pull/136601), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery]
- Updated cri-tools to v1.35.0. ([#135694](https://github.com/kubernetes/kubernetes/pull/135694), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- Updates the etcd client library to v3.6.6 ([#135331](https://github.com/kubernetes/kubernetes/pull/135331), [@yashsingh74](https://github.com/yashsingh74)) [SIG API Machinery, Auth, Cloud Provider, Etcd, Node and Scheduling]
- Updates the etcd client library to v3.6.7 ([#136407](https://github.com/kubernetes/kubernetes/pull/136407), [@ivanvc](https://github.com/ivanvc)) [SIG API Machinery, Auth, Cloud Provider, Node and Scheduling]

## Dependencies

### Added
- buf.build/go/protovalidate: v0.12.0
- github.com/cenkalti/backoff/v5: [v5.0.3](https://github.com/cenkalti/backoff/tree/v5.0.3)
- github.com/moby/moby/api: [v1.52.0](https://github.com/moby/moby/tree/api/v1.52.0)
- github.com/moby/moby/client: [v0.2.1](https://github.com/moby/moby/tree/client/v0.2.1)
- go.opentelemetry.io/otel/exporters/stdout/stdouttrace: v1.39.0
- gonum.org/v1/gonum: v0.16.0

### Changed
- buf.build/gen/go/bufbuild/protovalidate/protocolbuffers/go: 63bb56e → 8976f5b
- cloud.google.com/go/compute/metadata: v0.7.0 → v0.9.0
- cyphar.com/go-pathrs: v0.2.1 → v0.2.2
- github.com/GoogleCloudPlatform/opentelemetry-operations-go/detectors/gcp: [v1.26.0 → v1.30.0](https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/compare/detectors/gcp/v1.26.0...detectors/gcp/v1.30.0)
- github.com/Microsoft/hnslib: [v0.1.1 → v0.1.2](https://github.com/Microsoft/hnslib/compare/v0.1.1...v0.1.2)
- github.com/alecthomas/units: [b94a6e3 → 0f3dac3](https://github.com/alecthomas/units/compare/b94a6e3...0f3dac3)
- github.com/cncf/xds/go: [2f00578 → 0feb691](https://github.com/cncf/xds/compare/2f00578...0feb691)
- github.com/containerd/containerd/api: [v1.9.0 → v1.10.0](https://github.com/containerd/containerd/compare/api/v1.9.0...api/v1.10.0)
- github.com/coredns/corefile-migration: [v1.0.29 → v1.0.30](https://github.com/coredns/corefile-migration/compare/v1.0.29...v1.0.30)
- github.com/coreos/go-oidc: [v2.3.0+incompatible → v2.5.0+incompatible](https://github.com/coreos/go-oidc/compare/v2.3.0...v2.5.0)
- github.com/coreos/go-systemd/v22: [v22.5.0 → v22.7.0](https://github.com/coreos/go-systemd/compare/v22.5.0...v22.7.0)
- github.com/cyphar/filepath-securejoin: [v0.6.0 → v0.6.1](https://github.com/cyphar/filepath-securejoin/compare/v0.6.0...v0.6.1)
- github.com/davecgh/go-spew: [v1.1.1 → d8f796a](https://github.com/davecgh/go-spew/compare/v1.1.1...d8f796a)
- github.com/docker/go-connections: [v0.5.0 → v0.6.0](https://github.com/docker/go-connections/compare/v0.5.0...v0.6.0)
- github.com/emicklei/go-restful/v3: [v3.12.2 → v3.13.0](https://github.com/emicklei/go-restful/compare/v3.12.2...v3.13.0)
- github.com/envoyproxy/go-control-plane/envoy: [v1.32.4 → v1.35.0](https://github.com/envoyproxy/go-control-plane/compare/envoy/v1.32.4...envoy/v1.35.0)
- github.com/envoyproxy/go-control-plane: [v0.13.4 → 75eaa19](https://github.com/envoyproxy/go-control-plane/compare/v0.13.4...75eaa19)
- github.com/go-jose/go-jose/v4: [v4.0.4 → v4.1.3](https://github.com/go-jose/go-jose/compare/v4.0.4...v4.1.3)
- github.com/godbus/dbus/v5: [v5.1.0 → v5.2.2](https://github.com/godbus/dbus/compare/v5.1.0...v5.2.2)
- github.com/golang-jwt/jwt/v5: [v5.2.2 → v5.3.0](https://github.com/golang-jwt/jwt/compare/v5.2.2...v5.3.0)
- github.com/golang/glog: [v1.2.4 → v1.2.5](https://github.com/golang/glog/compare/v1.2.4...v1.2.5)
- github.com/google/cadvisor: [v0.53.0 → v0.56.0](https://github.com/google/cadvisor/compare/v0.53.0...v0.56.0)
- github.com/grpc-ecosystem/go-grpc-middleware/providers/prometheus: [v1.0.1 → v1.1.0](https://github.com/grpc-ecosystem/go-grpc-middleware/compare/providers/prometheus/v1.0.1...providers/prometheus/v1.1.0)
- github.com/grpc-ecosystem/go-grpc-middleware/v2: [v2.3.0 → v2.3.3](https://github.com/grpc-ecosystem/go-grpc-middleware/compare/v2.3.0...v2.3.3)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.26.3 → v2.27.4](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.26.3...v2.27.4)
- github.com/onsi/ginkgo/v2: [v2.27.2 → v2.27.4](https://github.com/onsi/ginkgo/compare/v2.27.2...v2.27.4)
- github.com/onsi/gomega: [v1.38.2 → v1.39.0](https://github.com/onsi/gomega/compare/v1.38.2...v1.39.0)
- github.com/opencontainers/cgroups: [v0.0.3 → v0.0.6](https://github.com/opencontainers/cgroups/compare/v0.0.3...v0.0.6)
- github.com/opencontainers/runc: [v1.3.0 → v1.4.0](https://github.com/opencontainers/runc/compare/v1.3.0...v1.4.0)
- github.com/opencontainers/runtime-spec: [v1.2.1 → v1.3.0](https://github.com/opencontainers/runtime-spec/compare/v1.2.1...v1.3.0)
- github.com/opencontainers/selinux: [v1.13.0 → v1.13.1](https://github.com/opencontainers/selinux/compare/v1.13.0...v1.13.1)
- github.com/pmezard/go-difflib: [v1.0.0 → 5d4384e](https://github.com/pmezard/go-difflib/compare/v1.0.0...5d4384e)
- github.com/prometheus/common: [v0.66.1 → v0.67.5](https://github.com/prometheus/common/compare/v0.66.1...v0.67.5)
- github.com/prometheus/procfs: [v0.16.1 → v0.19.2](https://github.com/prometheus/procfs/compare/v0.16.1...v0.19.2)
- github.com/spiffe/go-spiffe/v2: [v2.5.0 → v2.6.0](https://github.com/spiffe/go-spiffe/compare/v2.5.0...v2.6.0)
- go.etcd.io/etcd/api/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/client/pkg/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/client/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/pkg/v3: v3.6.5 → v3.6.7
- go.etcd.io/etcd/server/v3: v3.6.5 → v3.6.7
- go.opentelemetry.io/auto/sdk: v1.1.0 → v1.2.1
- go.opentelemetry.io/contrib/detectors/gcp: v1.34.0 → v1.38.0
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.44.0 → v0.64.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.60.0 → v0.63.0
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.61.0 → v0.64.0
- go.opentelemetry.io/contrib/propagators/b3: v1.19.0 → v1.39.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.34.0 → v1.39.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.34.0 → v1.39.0
- go.opentelemetry.io/otel/metric: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel/sdk/metric: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel/sdk: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel/trace: v1.36.0 → v1.39.0
- go.opentelemetry.io/otel: v1.36.0 → v1.39.0
- go.opentelemetry.io/proto/otlp: v1.5.0 → v1.9.0
- go.uber.org/zap: v1.27.0 → v1.27.1
- golang.org/x/crypto: v0.45.0 → v0.47.0
- golang.org/x/exp: 8a7402a → 944ab1f
- golang.org/x/mod: v0.29.0 → v0.32.0
- golang.org/x/net: v0.47.0 → v0.49.0
- golang.org/x/oauth2: v0.30.0 → v0.34.0
- golang.org/x/sync: v0.18.0 → v0.19.0
- golang.org/x/sys: v0.38.0 → v0.40.0
- golang.org/x/telemetry: 078029d → 8fff8a5
- golang.org/x/term: v0.37.0 → v0.39.0
- golang.org/x/text: v0.31.0 → v0.33.0
- golang.org/x/time: v0.9.0 → v0.14.0
- golang.org/x/tools: v0.38.0 → v0.40.0
- google.golang.org/genproto/googleapis/api: a0af3ef → 99fd39f
- google.golang.org/genproto/googleapis/rpc: 200df99 → 99fd39f
- google.golang.org/grpc: v1.72.2 → v1.78.0
- google.golang.org/protobuf: v1.36.8 → v1.36.11
- k8s.io/kube-openapi: 589584f → a19766b
- k8s.io/utils: bc988d5 → 914a6e7
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.31.2 → v0.34.0
- sigs.k8s.io/structured-merge-diff/v6: v6.3.0 → v6.3.1

### Removed
- github.com/armon/circbuf: [5111143](https://github.com/armon/circbuf/tree/5111143)
- github.com/bufbuild/protovalidate-go: [v0.9.1](https://github.com/bufbuild/protovalidate-go/tree/v0.9.1)
- github.com/docker/docker: [v28.2.2+incompatible](https://github.com/docker/docker/tree/v28.2.2)
- github.com/gregjones/httpcache: [901d907](https://github.com/gregjones/httpcache/tree/901d907)
- github.com/grpc-ecosystem/go-grpc-prometheus: [v1.2.0](https://github.com/grpc-ecosystem/go-grpc-prometheus/tree/v1.2.0)
- github.com/karrick/godirwalk: [v1.17.0](https://github.com/karrick/godirwalk/tree/v1.17.0)
- github.com/libopenstorage/openstorage: [v1.0.0](https://github.com/libopenstorage/openstorage/tree/v1.0.0)
- github.com/moby/sys/atomicwriter: [v0.1.0](https://github.com/moby/sys/tree/atomicwriter/v0.1.0)
- github.com/mohae/deepcopy: [c48cc78](https://github.com/mohae/deepcopy/tree/c48cc78)
- github.com/morikuni/aec: [v1.0.0](https://github.com/morikuni/aec/tree/v1.0.0)
- github.com/mrunalp/fileutils: [v0.5.1](https://github.com/mrunalp/fileutils/tree/v0.5.1)
- github.com/zeebo/errs: [v1.4.0](https://github.com/zeebo/errs/tree/v1.4.0)
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp: v1.27.0
- gotest.tools/v3: v3.0.2