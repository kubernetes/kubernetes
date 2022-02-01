<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.24.0-alpha.2](#v1240-alpha2)
  - [Downloads for v1.24.0-alpha.2](#downloads-for-v1240-alpha2)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.24.0-alpha.1](#changelog-since-v1240-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
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
- [v1.24.0-alpha.1](#v1240-alpha1)
  - [Downloads for v1.24.0-alpha.1](#downloads-for-v1240-alpha1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.23.0](#changelog-since-v1230)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)

<!-- END MUNGE: GENERATED_TOC -->

# v1.24.0-alpha.2


## Downloads for v1.24.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes.tar.gz) | bd3257bbae848869e20696e4570f29d61d78187d710c99fa01c5602e4edcf818f8129a68d80e83e51cc4b1010eea8e61691a9439c6c72607b5e1b6e32cd2a60e
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-src.tar.gz) | d8235197b71248ffa5fcbabbdab11c208f9d55f58db498e038e7464c0caf99bfddfa8d34e8af46ca3f908d865d6836786c0030afce15a3d1ff5f4d1cdfc69929

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | a17876d27eb72590893ddf440f9c0fd18137a6e5f9dc57b34a8a9057fffd6b6a5356bca92adf888e3e223b0aa58f47dc08594fbdb6d0e1934d86fdd167b7aca9
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 9f513e665ebf86d795933d55ba7b2d9e183761d6ff36e04626cb2e597ba4af9a840dbf995466a4c4d4ee89f9a9b0cbfa9217ce69bf7d6d66d65989e02e04ae73
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 511b2147da305368cc24c372f052aeaf1f2aa7bf7fdfdb4fc81a6b3163cda4bc8392b0392610799f0bd96500daeae98aa39f657ca37811fed326a21e2d43f218
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 94514f56a6fab4887ea44405ee59020115cfb53d7c1a4d2464fa3ceb804c3d141d4c1d090e7d7652d9514950ea7f52f96b1f59a560359673aa0bb7dffe307198
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 4a3a0c2fa1875caf5c0715b67a8b0e375362e02cd9be88439c32a853a73eff26b419da58772ab1b13ecaed0480a6f7d6d85681d71b096cf941eb9d45e137e157
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 831b837ce1159bdd0e4b7a238d0bbf998b24495cf335fdf960b789fbe255ebee75a7f3d6e9831782b0967bc04323abced69b1384411fbfd637b06d7483a24053
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 09cd6c441ee4b57261966e8c93d45374b577818f79ccaae3042a17bb06203ad41a1a0d046d28382782f2cd8a49a0677fbdfa600783cd61ee36a2cc8ae9ce9e7e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | fe29b548df2d6016a98b4409ff783001be70945875f036d7a799445ef60a1493fd52618c8136cbb6a089d98703148076a09286701e2918400cf1a3ed77aac953
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 3ca9c008c79575525b1b758240e24466a60f9a34c5c12895bc0d8f79ff6b5ab057f3be1d1a7bb561084092cf18d9d46d80698fc0691947fc86b63ec1a4c0decf
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | ee32c78eeae2c8db9f8fc4bac02b5c5a0b9eb29612bfac71f0c9c48f83fd03c31aa2b459a41f0a06087dafbb71cd8c109e797bd243fe72f32db05a584e03f697
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | debcc893f4c4be2ef034e056b126ef5b7c0f60a0a7d43117e2271850ded56dcc9eb103cc764337506bcd5d4bc22a87611c1501c17dfbfe89c62185588a6356ee

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | ec33945179f1ea5ca6334cf761247c975e5d22b1bf9b415dae9903aef67443c94894794f1e2ae932421c847cc7388c6c31307d2c7dd8b28aa8c2f39483f83de6
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | 77a7a799e675ee4fc5371768cbca36f624e2419611740393fd850c0f2506cd4926a0d31d8ed754c06bb1b1852cd53b073a21b7b6a03c6059efe64316a1d39f69
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 6d198e0edf891f4b161d2f0df8945113264b31bafa153ca2a22f4cf0043a2810e2f1687d41e9f7fd2351704d2c720c45ab8cd235ee452897f8322a233e65c435
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 4c1e2a4f076297f4684f8c4781b5cfca685423a3b0b7e761b74d8e35860546437901f8a896a182c2ae6fe69dfd5f2a32468a8fffb2ca08a7c80be6afb444617c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | b3df4a87f3d3d9a2e83d7136bb8ebf6c2a625b893812e1c01bf6e7424e41b8c5c0373912b80e0a309a36936181439b3b5700dc97451fcf86fb7d983d15e8d284

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 20aa621176d9f09cb4e32e4e56eaa933953871d877d4d9a55963f73290e3acce3773446c32e69624f15483c29fb5c05166d0ddc4e413cc5d9dd27f93109b86f1
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 6c7e549b50ba0a1d1ac6371bf72e3833f92187848fae3d75a52f7087336d2e85f976dbf8104ac01109177a8478d82efe12de905db7a88b1b7a11d4f05649e02c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 99f25bc10d2f9139d92e3fb12186854da8b987fd3f060b5b7a906bc27345b93e3cde23b07527f42c3ffc34288e2dc87d957aa73e91cfa4c5f2a0f43bfb00037b
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | aa68d2de162cf75f58fb97b2a65a7d1963a9a2483dae565846da44a335696733aa10e0982badebe4fc9048716cd0a85aef32ba9cf9f22d244f69f2adfe60bc12
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | d64ed933b24b7d897194bc70954a42d8217e9e4bc5f0bf797cad3fa54a16b63b5b5a1731886d4bde9b5e80306481686620b027caa0ef413925bb446f6d0a96a9
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | f3af562d8b4f3b17d039b56bec041166bcf9dfa831b3bbbc1ea67864dae00093e564ce7605855d57158f6ab3aaa7b847bebb91948563b439388c028617184429

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
k8s.gcr.io/kube-apiserver:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-controller-manager:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-proxy:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-scheduler:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/conformance:v1.24.0-alpha.2 | amd64, arm, arm64, ppc64le, s390x

## Changelog since v1.24.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Docker runtime support using dockshim in the kubelet is now completely removed in 1.24. The kubelet used to have a a module called "dockershim" which implements CRI support for Docker and it has seen maintenance issues in the Kubernetes community. From 1.24 onwards, please move to a container runtime that is a full-fledged implementation of CRI (v1alpha1 or v1 compliant) as they become available. ([#97252](https://github.com/kubernetes/kubernetes/pull/97252), [@dims](https://github.com/dims)) [SIG Cloud Provider, Instrumentation, Network, Node and Testing]
  - The calculations for Pod topology spread skew now excludes nodes that
  don't match the node affinity/selector. This may lead to unschedulable pods if you previously had pods
  matching the spreading selector on those excluded nodes (not matching the node affinity/selector),
  especially when the topologyKey is not node-level. Revisit the node affinity and/or pod selector in the
  topology spread constraints to avoid this scenario. ([#107009](https://github.com/kubernetes/kubernetes/pull/107009), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
 
## Changes by Kind

### Deprecation

- "kubeadm.k8s.io/v1beta2" has been deprecated and will be removed in a future release, possibly in 3 releases (one year). You should start using "kubeadm.k8s.io/v1beta3" for new clusters. To migrate your old configuration files on disk you can use the "kubeadm config migrate" command. ([#107013](https://github.com/kubernetes/kubernetes/pull/107013), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Deprecate Service.Spec.LoadBalancerIP. This field was under-specified and its meaning varies across implementations.  As of Kubernetes v1.24, users are encouraged to use implementation-specific annotations when available.  This field may be removed in a future API version. ([#107235](https://github.com/kubernetes/kubernetes/pull/107235), [@uablrek](https://github.com/uablrek)) [SIG Apps and Network]
- Kube-apiserver: the insecure address flags `--address`, `--insecure-bind-address`, `--port` and `--insecure-port` (inert since 1.20) are removed ([#106859](https://github.com/kubernetes/kubernetes/pull/106859), [@knight42](https://github.com/knight42)) [SIG API Machinery, Cloud Provider and Cluster Lifecycle]
- The experimental dynamic log sanitization feature has been deprecated and removed in the 1.24 release. The feature is no longer available for use. ([#107207](https://github.com/kubernetes/kubernetes/pull/107207), [@ehashman](https://github.com/ehashman)) [SIG Instrumentation, Scheduling and Security]
- The insecure address flags `--address` and `--port` in kube-controller-manager have been no effect since v1.20 and is removed in v1.24. ([#106860](https://github.com/kubernetes/kubernetes/pull/106860), [@knight42](https://github.com/knight42)) [SIG API Machinery, Node and Testing]

### API Change

- Add a new metric `webhook_fail_open_count` to monitor webhooks that fail open ([#107171](https://github.com/kubernetes/kubernetes/pull/107171), [@ltagliamonte-dd](https://github.com/ltagliamonte-dd)) [SIG API Machinery and Instrumentation]
- Fix failed flushing logs in defer function when kubelet cmd exit 1. ([#104774](https://github.com/kubernetes/kubernetes/pull/104774), [@kerthcet](https://github.com/kerthcet)) [SIG Node and Scheduling]
- Rename metrics `evictions_number` to `evictions_total` and mark it as stable. The original `evictions_number` metrics name is marked as "Deprecated" and will be removed in kubernetes 1.23 ([#106366](https://github.com/kubernetes/kubernetes/pull/106366), [@cyclinder](https://github.com/cyclinder)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage, Testing and Windows]
- The `ServiceLBNodePortControl` feature graduates to GA. The feature gate will be removed in 1.26. ([#107027](https://github.com/kubernetes/kubernetes/pull/107027), [@uablrek](https://github.com/uablrek)) [SIG Network and Testing]
- The feature DynamicKubeletConfig is removed from the kubelet. ([#106932](https://github.com/kubernetes/kubernetes/pull/106932), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Apps, Auth, Instrumentation, Node and Testing]
- Update default API priority-and-fairness config to avoid endpoint/configmaps operations from controller-manager to all match leader-election priority level. ([#106725](https://github.com/kubernetes/kubernetes/pull/106725), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]

### Feature

- A new Priority and Fairness metric 'apiserver_flowcontrol_work_estimate_seats_samples' has been 
  added that tracks the estimated seats associated with a request ([#106628](https://github.com/kubernetes/kubernetes/pull/106628), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Instrumentation]
- Add completion for `kubectl config set-context`. ([#106739](https://github.com/kubernetes/kubernetes/pull/106739), [@kebe7jun](https://github.com/kebe7jun)) [SIG CLI]
- Add metric for measuring end-to-end volume mount timing ([#107006](https://github.com/kubernetes/kubernetes/pull/107006), [@gnufied](https://github.com/gnufied)) [SIG Node and Storage]
- Add more message for no PodSandbox container ([#107116](https://github.com/kubernetes/kubernetes/pull/107116), [@yxxhero](https://github.com/yxxhero)) [SIG Node]
- Added field add_ambient_capabilities to the Capabilities message in the CRI-API. ([#104620](https://github.com/kubernetes/kubernetes/pull/104620), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Node]
- Added label selector flag to all "kubectl rollout" commands ([#99758](https://github.com/kubernetes/kubernetes/pull/99758), [@aramperes](https://github.com/aramperes)) [SIG CLI]
- Added prune flag into diff command to simulate `apply --prune` ([#105164](https://github.com/kubernetes/kubernetes/pull/105164), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Adds SetTransform to SharedInformer to allow users to transform objects before they are stored. ([#107507](https://github.com/kubernetes/kubernetes/pull/107507), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- Adds proxy-url flag into kubectl config set-cluster ([#105566](https://github.com/kubernetes/kubernetes/pull/105566), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Adds support for kubectl commands (`kubectl exec` and `kubectl port-forward`) via a SOCKS5 proxy. ([#105632](https://github.com/kubernetes/kubernetes/pull/105632), [@xens](https://github.com/xens)) [SIG API Machinery, Architecture, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Feature of `PreferNominatedNode` is graduated  to GA ([#106619](https://github.com/kubernetes/kubernetes/pull/106619), [@chendave](https://github.com/chendave)) [SIG Scheduling and Testing]
- In text format, log messages that previously used quoting to prevent multi-line output (for example, text="some \"quotation\", a\nline break") will now be printed with more readable multi-line output without the escape sequences. ([#107103](https://github.com/kubernetes/kubernetes/pull/107103), [@pohly](https://github.com/pohly)) [SIG Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Kube-apiserver: when merging lists, Server Side Apply now prefers the order of the submitted request instead of the existing persisted object ([#107565](https://github.com/kubernetes/kubernetes/pull/107565), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- Kube-scheduler remove insecure flags. You can use --bind-address and --secure-port instead. ([#106865](https://github.com/kubernetes/kubernetes/pull/106865), [@jonyhy96](https://github.com/jonyhy96)) [SIG Scheduling]
- Kubeadm: add support for dry running "kubeadm reset". The new flag "kubeadm reset --dry-run" is similar to the existing flag for "kubeadm init/join/upgrade" and allows you to see what changes would be applied. ([#107512](https://github.com/kubernetes/kubernetes/pull/107512), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: default the kubeadm configuration to the containerd socket (Unix: unix:///var/run/containerd/containerd.sock, Windows: "npipe:////./pipe/containerd-containerd") instead of the one for Docker. If the "Init|JoinConfiguration.nodeRegistration.criSocket" field is empty during cluster creation and multiple sockets are found on the host always throw an error and ask the user to specify which one to use by setting the value in the field. Make sure you update any kubeadm configuration files on disk, to not include the dockershim socket unless you are still using kubelet version < 1.24 with kubeadm >= 1.24.
  
  Remove the DockerValidor and ServiceCheck for the "docker" service from kubeadm preflight. Docker is no longer special cased during host validation and ideally this task should be done in the now external cri-dockerd project where the importance of the compatibility matters.
  
  Use crictl for all communication with CRI sockets for actions like pulling images and obtaining a list of running containers instead of using the docker CLI in the case of Docker. ([#107317](https://github.com/kubernetes/kubernetes/pull/107317), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl logs will now warn and default to the first container in a pod. This new behavior brings it in line with kubectl exec. ([#105964](https://github.com/kubernetes/kubernetes/pull/105964), [@kidlj](https://github.com/kidlj)) [SIG CLI]
- Kubelet: following dockershim related flags are also removed along with dockershim 
  --experimental-dockershim-root-directory, --docker-endpoint, --image-pull-progress-deadline, --network-plugin, 
  --cni-conf-dir,--cni-bin-dir, --cni-cache-dir, --network-plugin-mtu ([#106907](https://github.com/kubernetes/kubernetes/pull/106907), [@cyclinder](https://github.com/cyclinder)) [SIG Cloud Provider, Node and Testing]
- Kubernetes is now built with Golang 1.17.5 ([#106956](https://github.com/kubernetes/kubernetes/pull/106956), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- Kubernetes is now built with Golang 1.17.6 ([#107612](https://github.com/kubernetes/kubernetes/pull/107612), [@palnabarun](https://github.com/palnabarun)) [SIG Release and Testing]
- OpenStack Cinder CSI migration is now GA and switched on by default, Cinder CSI driver must be installed on clusters on OpenStack for Cinder volumes to work (has been since v1.21). ([#107462](https://github.com/kubernetes/kubernetes/pull/107462), [@dims](https://github.com/dims)) [SIG Scheduling and Storage]
- Remove feature gate `ImmutableEphemeralVolumes`. ([#107152](https://github.com/kubernetes/kubernetes/pull/107152), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Node and Storage]
- This adds a path `/header?key=` to `agnhost netexec` allowing one to view what the header value is of the incoming request.
  
  Ex:
  
  $ curl -H "X-Forwarded-For: something" 172.17.0.2:8080/header?key=X-Forwarded-For
  something ([#107796](https://github.com/kubernetes/kubernetes/pull/107796), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Testing]
- Update golang.org/x/net to v0.0.0-20211209124913-491a49abca63 ([#106949](https://github.com/kubernetes/kubernetes/pull/106949), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- We have added a new Priority and Fairness metric apiserver_flowcontrol_request_dispatch_no_accommodation_total' 
  to track the number of times a request dispatch attempt results in a no-accommodation status due to lack of available seats ([#106629](https://github.com/kubernetes/kubernetes/pull/106629), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Instrumentation]

### Bug or Regression

- A new label `type` has been added to `apiserver_flowcontrol_request_execution_seconds` metric - it has the following values: 
  - 'regular': indicates that it is a non long running request
  - 'watch': indicates that it is a watch request ([#105517](https://github.com/kubernetes/kubernetes/pull/105517), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Instrumentation]
- Add a test to guarantee that conformance clusters require at least 2 untainted nodes ([#106313](https://github.com/kubernetes/kubernetes/pull/106313), [@aojea](https://github.com/aojea)) [SIG Architecture and Testing]
- Allow attached volumes to be mounted quicker by skipping exp. backoff when checking for reported-in-use volumes ([#106853](https://github.com/kubernetes/kubernetes/pull/106853), [@gnufied](https://github.com/gnufied)) [SIG Apps, Node and Storage]
- An inefficient lock in EndpointSlice controller metrics cache has been reworked. Network programming latency may be significantly reduced in certain scenarios, especially in clusters with a large number of Services. ([#107091](https://github.com/kubernetes/kubernetes/pull/107091), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Scalability]
- Apiserver will now reject connection attempts to 0.0.0.0/:: when handling a proxy subresource request ([#107402](https://github.com/kubernetes/kubernetes/pull/107402), [@anguslees](https://github.com/anguslees)) [SIG Network]
- Apiserver, if configured to reconcile the kubernetes.default service endpoints, checks if the configured Service IP range matches the apiserver public address IP family, and fails to start if not. ([#106721](https://github.com/kubernetes/kubernetes/pull/106721), [@aojea](https://github.com/aojea)) [SIG API Machinery and Testing]
- Change node staging path for csi driver to use a PV agnostic path. Nodes must be drained before updating the kubelet with this change. ([#107065](https://github.com/kubernetes/kubernetes/pull/107065), [@saikat-royc](https://github.com/saikat-royc)) [SIG Storage and Testing]
- Client-go: fix that paged list calls with ResourceVersionMatch set would fail once paging kicked in. ([#107311](https://github.com/kubernetes/kubernetes/pull/107311), [@fasaxc](https://github.com/fasaxc)) [SIG API Machinery]
- Fix Azurefile volumeid collision issue in csi migration ([#107575](https://github.com/kubernetes/kubernetes/pull/107575), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix a panic when using invalid output format in kubectl create secret command ([#107221](https://github.com/kubernetes/kubernetes/pull/107221), [@rikatz](https://github.com/rikatz)) [SIG CLI]
- Fix libct/cg/fs2: fix GetStats for unsupported hugetlb error on Raspbian Bullseye ([#106912](https://github.com/kubernetes/kubernetes/pull/106912), [@Letme](https://github.com/Letme)) [SIG Node]
- Fix performance regression in JSON logging caused by syncing stdout every time error was logged. ([#107035](https://github.com/kubernetes/kubernetes/pull/107035), [@serathius](https://github.com/serathius)) [SIG Instrumentation and Scalability]
- Fix: azuredisk parameter lowercase translation issue ([#107429](https://github.com/kubernetes/kubernetes/pull/107429), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fix: delete non existing Azure disk issue ([#107406](https://github.com/kubernetes/kubernetes/pull/107406), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: remove outdated ipv4 route when the corresponding node is deleted ([#106164](https://github.com/kubernetes/kubernetes/pull/106164), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fixed a bug that a pod's .status.nominatedNodeName is not cleared properly, and thus over-occupied system resources. ([#106816](https://github.com/kubernetes/kubernetes/pull/106816), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fixed a bug that could cause a panic when a /healthz request times out. ([#107034](https://github.com/kubernetes/kubernetes/pull/107034), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Fixed a bug where vSphere client connections where not being closed during testing. Leaked vSphere client sessions were causing resource exhaustion during automated testing. ([#107337](https://github.com/kubernetes/kubernetes/pull/107337), [@derek-pryor](https://github.com/derek-pryor)) [SIG Storage and Testing]
- Fixed detaching CSI volumes from nodes when a CSI driver name has prefix "csi-". ([#107025](https://github.com/kubernetes/kubernetes/pull/107025), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixed duplicate port opening in kube-proxy when "--nodeport-addresses" is empty ([#107413](https://github.com/kubernetes/kubernetes/pull/107413), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Fixed kubectl bug where bash completions don't work if --context flag is specified with a value that contains a colon ([#107439](https://github.com/kubernetes/kubernetes/pull/107439), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Fixes a bug where unwanted fields were being returned from a create dry-run: uid and, if generateName was used, name. ([#107088](https://github.com/kubernetes/kubernetes/pull/107088), [@joejulian](https://github.com/joejulian)) [SIG API Machinery and Testing]
- Fixes a rare race condition handling requests that timeout ([#107452](https://github.com/kubernetes/kubernetes/pull/107452), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Fixes a regression in 1.23 that incorrectly pruned data from array items of a custom resource that set `x-kubernetes-preserve-unknown-fields: true` ([#107688](https://github.com/kubernetes/kubernetes/pull/107688), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Fixes a regression in 1.23 where update requests to previously persisted `Service` objects that have not been modified since 1.19 can be rejected with an incorrect `spec.clusterIPs: Required value` error ([#107847](https://github.com/kubernetes/kubernetes/pull/107847), [@thockin](https://github.com/thockin)) [SIG API Machinery, Network and Testing]
- Fixes handling of objects with invalid selectors ([#107559](https://github.com/kubernetes/kubernetes/pull/107559), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Scheduling and Storage]
- Fixes regression in CPUManager that it will release exclusive CPUs in app containers inherited from init containers when the init containers were removed. ([#104837](https://github.com/kubernetes/kubernetes/pull/104837), [@eggiter](https://github.com/eggiter)) [SIG Node]
- Fixes static pod add and removes restarts in certain cases. ([#107695](https://github.com/kubernetes/kubernetes/pull/107695), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Improve handling of unmount failures when device may be in-use by another container/process ([#107789](https://github.com/kubernetes/kubernetes/pull/107789), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Improve rounding of PodTopologySpread scores to offer better scoring when spreading a low number of pods. ([#107384](https://github.com/kubernetes/kubernetes/pull/107384), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Kubeadm: during execution of the "check expiration" command, treat the etcd CA as external if there is a missing etcd CA key file (etcd/ca.key) and perform the proper validation on certificates signed by the etcd CA. Additionally, make sure that the CA for all entries in the output table is included - for both certificates on disk and in kubeconfig files. ([#106891](https://github.com/kubernetes/kubernetes/pull/106891), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- No ([#107769](https://github.com/kubernetes/kubernetes/pull/107769), [@liurupeng](https://github.com/liurupeng)) [SIG Cloud Provider and Windows]
- NodeRestriction admission: nodes are now allowed to update PersistentVolumeClaim status fields `resizeStatus` and `allocatedResources` when the `RecoverVolumeExpansionFailure` feature is enabled ([#107686](https://github.com/kubernetes/kubernetes/pull/107686), [@gnufied](https://github.com/gnufied)) [SIG Auth and Storage]
- Only extend token lifetimes when --service-account-extend-token-expiration is true and the requested token audiences are empty or exactly match all values for --api-audiences ([#105954](https://github.com/kubernetes/kubernetes/pull/105954), [@jyotimahapatra](https://github.com/jyotimahapatra)) [SIG Auth and Testing]
- Removed validation if AppArmor profiles are loaded on the local node. This should be handled by the
  container runtime. ([#97966](https://github.com/kubernetes/kubernetes/pull/97966), [@saschagrunert](https://github.com/saschagrunert)) [SIG Auth, Node and Security]
- Restore NumPDBViolations info of nodes, when HTTPExtender ProcessPreemption. This info will be used in subsequent filtering steps - pickOneNodeForPreemption ([#105853](https://github.com/kubernetes/kubernetes/pull/105853), [@caden2016](https://github.com/caden2016)) [SIG Scheduling]
- Reverts graceful node shutdown to match 1.21 behavior of setting pods that have not yet successfully completed to "Failed" phase if the GracefulNodeShutdown feature is enabled in kubelet. The GracefulNodeShutdown feature is beta and must be explicitly configured via kubelet config to be enabled in 1.21+. This changes 1.22 and 1.23 behavior on node shutdown to match 1.21. If you do not want pods to be marked terminated on node shutdown in 1.22 and 1.23, disable the GracefulNodeShutdown feature. ([#106901](https://github.com/kubernetes/kubernetes/pull/106901), [@bobbypage](https://github.com/bobbypage)) [SIG Node and Testing]
- Some command line errors (for example, "kubectl list" -> "unknown command") were printed as log message with escaped line breaks instead of a multi-line plain text, which made the error harder to read. ([#107044](https://github.com/kubernetes/kubernetes/pull/107044), [@pohly](https://github.com/pohly)) [SIG CLI and Testing]
- Some log messages were logged with `"v":0` in JSON output although they are debug messages with a higher verbosity. ([#106978](https://github.com/kubernetes/kubernetes/pull/106978), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Scheduling and Storage]
- The Service field spec.internalTrafficPolicy is no longer defaulted for Services when the type is ExternalName. The field is also dropped on read when the Service type is ExternalName. ([#104846](https://github.com/kubernetes/kubernetes/pull/104846), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps and Network]
- The feature gate was mentioned as `csiMigrationRBD` where it should have been `CSIMigrationRBD` to be in parity with other migration plugins. This release correct the same and keep it as `CSIMigrationRBD`.
  
  users who have configured this feature gate as `csiMigrationRBD` has to reconfigure the same to `CSIMigrationRBD` from this release. ([#107554](https://github.com/kubernetes/kubernetes/pull/107554), [@humblec](https://github.com/humblec)) [SIG Storage]
- When doing `make test-integration`, you can now usefully include `-args $prog_args` in KUBE_TEST_ARGS. ([#107516](https://github.com/kubernetes/kubernetes/pull/107516), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG Testing]

### Other (Cleanup or Flake)

- --container-runtime kubelet flag is deprecated and will be removed in future releases ([#107094](https://github.com/kubernetes/kubernetes/pull/107094), [@adisky](https://github.com/adisky)) [SIG Node]
- Add details about preemption in the event for scheduling failed ([#107775](https://github.com/kubernetes/kubernetes/pull/107775), [@denkensk](https://github.com/denkensk)) [SIG Scheduling]
- Build/dependencies.yaml: remove the dependency on Docker. With the dockershim removal, core Kubernetes no longer
  has to track the latest validated version of Docker. ([#107607](https://github.com/kubernetes/kubernetes/pull/107607), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
- Correct the error message to not use the "--max-resource-write-bytes" & "--json-patch-max-copy-bytes" string. ([#106875](https://github.com/kubernetes/kubernetes/pull/106875), [@warmchang](https://github.com/warmchang)) [SIG API Machinery]
- E2e tests wait for kube-root-ca.crt to be populated in namespaces for use with projected service account tokens, reducing delays starting those test pods and errors in the logs. ([#107763](https://github.com/kubernetes/kubernetes/pull/107763), [@smarterclayton](https://github.com/smarterclayton)) [SIG Testing]
- Fix documentation typo in cloud-provider ([#106445](https://github.com/kubernetes/kubernetes/pull/106445), [@majst01](https://github.com/majst01)) [SIG Cloud Provider]
- Fix spelling of implemented in pkg/proxy/apis/config/types.go line 206 ([#106453](https://github.com/kubernetes/kubernetes/pull/106453), [@davidleitw](https://github.com/davidleitw)) [SIG Network]
- Kubeadm: all warning messages are printed to stderr instead of stdout. ([#107467](https://github.com/kubernetes/kubernetes/pull/107467), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: handle the removal of dockershim related flags for new kubeadm clusters. If kubelet <1.24 is on the host, kubeadm >=1.24 can continue using the built-in dockershim in the kubelet if the user passes the "{Init|Join}Configuration.nodeRegistration.criSocket" value in the kubeadm configuration to be equal to "unix:///var/run/dockershim.sock" on Unix or "npipe:////./pipe/dockershim" on Windows. If kubelet version >=1.24 is on the host, kubeadm >=1.24 will treat all container runtimes as "remote" using the kubelet flags "--container-runtime=remote --container-runtime-endpoint=scheme://some/path". The special management for kubelet <1.24 will be removed in kubeadm 1.25. ([#106973](https://github.com/kubernetes/kubernetes/pull/106973), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: make sure that "kubeadm init/join" always use a URL scheme (unix:// on Linux and npipe:// on Windows) when passing a value to the "--container-runtime-endpoint" kubelet flag. This flag's value is taken from the kubeadm configuration "criSocket" field or the "--cri-socket" CLI flag. Automatically add a missing URL scheme to the user configuration in memory, but warn them that they should also update their configuration on disk manually. During "kubeadm upgrade apply/node" mutate the "/var/lib/kubelet/kubeadm-flags.env" file on disk and the "kubeadm.alpha.kubernetes.io/cri-socket" annotation Node object if needed. These automatic actions are temporary and will be removed in a future release. In the future the kubelet may not support CRI endpoints without an URL scheme. ([#107295](https://github.com/kubernetes/kubernetes/pull/107295), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the IPv6DualStack feature gate. The feature has been GA and locked to enabled since 1.23. ([#106648](https://github.com/kubernetes/kubernetes/pull/106648), [@calvin0327](https://github.com/calvin0327)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: remove the deprecated output/v1alpha1 API used for machine readable output by some kubeadm commands. In 1.23 kubeadm started using the newer version output/v1alpha2 for the same purpose. ([#107468](https://github.com/kubernetes/kubernetes/pull/107468), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the restriction that the ca.crt can only contain one certificate. If there is more than one certificate in the ca.crt file, kubeadm will pick the first one by default. ([#107327](https://github.com/kubernetes/kubernetes/pull/107327), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl: restores `--dry-run`, `--dry-run=true`, and `--dry-run=false` for compatibility with pre-1.23 invocations. ([#107003](https://github.com/kubernetes/kubernetes/pull/107003), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG CLI and Testing]
- Kubernetes e2e framework will use the url "invalid.registry.k8s.io/invalid" instead "invalid.com/invalid" for test that use an invalid registry. ([#107455](https://github.com/kubernetes/kubernetes/pull/107455), [@aojea](https://github.com/aojea)) [SIG Testing]
- Mark kubelet `--container-runtime-endpoint` and `--image-service-endpoint` CLI flags as stable ([#106954](https://github.com/kubernetes/kubernetes/pull/106954), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Migrate volume/csi/csi-client.go logs to structured logging ([#99441](https://github.com/kubernetes/kubernetes/pull/99441), [@CKchen0726](https://github.com/CKchen0726)) [SIG Storage]
- Please check your kubelet command line for enabling features and drop "RuntimeClass" if present. Note that this feature has been on by default since 1.14 and was GA'ed in 1.20. ([#106882](https://github.com/kubernetes/kubernetes/pull/106882), [@cyclinder](https://github.com/cyclinder)) [SIG Node]
- The fluentd-elasticsearch addon is no longer included in the cluster directory. It is available from https://github.com/kubernetes-sigs/instrumentation-addons/tree/master/fluentd-elasticsearch ([#107553](https://github.com/kubernetes/kubernetes/pull/107553), [@liggitt](https://github.com/liggitt)) [SIG Cloud Provider and Instrumentation]
- This PR deprecates types in `k8s.io/apimachinery/util/clock`. Please use `k8s.io/utils/clock` instead. ([#106850](https://github.com/kubernetes/kubernetes/pull/106850), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Updated cri-tools to [v1.23.0](https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.23.0) ([#107604](https://github.com/kubernetes/kubernetes/pull/107604), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Release]

## Dependencies

### Added
- github.com/armon/go-socks5: [e753329](https://github.com/armon/go-socks5/tree/e753329)

### Changed
- github.com/cespare/xxhash/v2: [v2.1.1 → v2.1.2](https://github.com/cespare/xxhash/v2/compare/v2.1.1...v2.1.2)
- github.com/moby/term: [9d4ed18 → 3f7ff69](https://github.com/moby/term/compare/9d4ed18...3f7ff69)
- github.com/opencontainers/runc: [v1.0.2 → v1.0.3](https://github.com/opencontainers/runc/compare/v1.0.2...v1.0.3)
- github.com/prometheus/client_golang: [v1.11.0 → v1.12.0](https://github.com/prometheus/client_golang/compare/v1.11.0...v1.12.0)
- github.com/prometheus/common: [v0.28.0 → v0.32.1](https://github.com/prometheus/common/compare/v0.28.0...v0.32.1)
- github.com/prometheus/procfs: [v0.6.0 → v0.7.3](https://github.com/prometheus/procfs/compare/v0.6.0...v0.7.3)
- github.com/yuin/goldmark: [v1.4.0 → v1.4.1](https://github.com/yuin/goldmark/compare/v1.4.0...v1.4.1)
- golang.org/x/mod: v0.4.2 → v0.5.1
- golang.org/x/net: e898025 → 491a49a
- golang.org/x/sys: f4d4317 → da31bd3
- golang.org/x/tools: d4cc65f → v0.1.8
- k8s.io/gengo: 485abfe → c02415c
- k8s.io/klog/v2: v2.30.0 → v2.40.1
- k8s.io/utils: cb0fa31 → 7d6a63d
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.25 → v0.0.27
- sigs.k8s.io/json: c049b76 → 9f7c6b3
- sigs.k8s.io/structured-merge-diff/v4: v4.1.2 → v4.2.1

### Removed
_Nothing has changed._



# v1.24.0-alpha.1


## Downloads for v1.24.0-alpha.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes.tar.gz) | 966bcdcaadb18787bab26852602a56dc973d785d7d9620c9ca870eba7133d93b2aaebf369ce52ae9b49160a4cd0101f7356a080b34c4a9a3a6ed2ff82ffd6400
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-src.tar.gz) | 107fe6bfba5ff79b28ab28a3652b6a3d03fe5a667217e3e6e8aabe391b95ddc8109e62b72239d0f66b31c99a8c0d7efb4a74ea49337c0986a53e4628cd4c45e2

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 70cc548677446b9e523c00b76b928ab7af0685bae57b4e52eb9916fd929d540a05596505cd1e198bdf41f85cebc38ddbde95d5214bfba0de1d24593ea1a047a7
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | fdfa4ee47ea5fa40782cf1c719a1ae2bb33a491209e53761f3368fa409f81d0dfeceafa10fa4659032a1fc1a5ff2c1959cba575c8a6bbfa151abadec01c180ab
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-386.tar.gz) | e7dbad9054cd7b2e7b212cb6403d8727470564b967e95f53e8ff1648f6fe7f63cee22fb1622fb4b278ad911f67c3488f8446e145f44e7e0befe85bba9c94ea11
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 28e9c8e79dc87dc701c87195589a5a38da7563f0c05ad1c0d40a1f545ef51ec4f6973b02e970bf74167a7534c5b788c5b01a94df570032306d561c2b3f7bbde4
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 57f3ad5670e3a52a6f988a6c0177f530ec9cf1841829b5ee439dad57243898ddd53b89988873b60bd6128cff430b4ff24244f48edbcec4ecb1885f7d5cd09bb8
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 99272cdc6adddf2f15b18070e8a174591192c27d419d80ce6f03f584e283c7626dea8b494c1f3b6b3607e94c6ccfeba678713e6041a23a5833880938bd356906
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | dc638a62b53f15038554aea9a652b3688c7f9843f61d183d7984f20195d1d4183baa923ce0c17ccd0fbae98192be97ccc8f2bd32fa1b774d32160196f6c2debc
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | e330d88076c4bd788a17da59281a23fe31076c8c5409df688091dd8688f4f94028db06f3f6dd777ab019184e4287487db76599eeb6647ee8fb545fd1e54b0dd9
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 0b594d163496eadc8f1643e4d383b0fc96f820c47ec649b0d843cba7b43eb0df050c4fb7b6a23e3f5b2696629d2ba9725d0b59a9e3256e6fdda470eb9a726424
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 6c6656618e461a0c398cfc4fd69b5b2aa959c8ef6a25ec23e62e5504e5bd5c72572d6a5dbe795a469a85a330fb5ca3d86aece447c0fbf8067f8ef7d8592359c2
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 8394dec41b013f3869b32ee17ad82e55201f77573a84037c21511f732c851f6297dfd7c145fc9b65e1d0aa8cecca6dd04027bef36942af9fa140260e48851aad

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | b781f1aa2ebdb89c0c2b35cba35c5c000cf8e6f87c71cc5cd9ac5938081d6914fb325a4a902e060a16ba31ada136f8d0d8dbbf2a27eb1c426428cda3e8166580
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | 0b92d1a3020c8128ea6dc337ce2fffb5dc8bf2500a02467434e90ad3025a699fea4eaca837bc9eea291d87b8adbc2b2814d9ab078ed49ecbabb47c42d9b910cf
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | e0804c2fa12d6c356a2dd32c26df3ae2b389ac21f5ea426abe1d3f99e0460d4096ad0a42bdf96fd1d4392874afa5fe16f5796a075f99c3690340fce5533377b3
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 2a520b5ea04d00c3c6f54f4ddb75b6e6ffa3c472d4951e51674b103187c8f129e20a5b1c22b0b3ce64281ae9fbf192069ad849af5ce4d2f1cdc394269c983b55
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 4993090e12df0cb1a3a9abea52e1f6bc5efefe7202d81ec36646b02799200c7128721bffb940d88d763effbcb094d159a18aabad476b39b1fcae461dfec1967e

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 4e8e4cea3ee4f2dfe12ad5d2361ac43dd1d961aa1bf0e5f9cedbe18ef37eae76bed6f9643ef4d771a5eef70ffb65e49e9dd917591dbd0ec0de243df85c20e86a
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | 10823d27fe41b45ea61a75974657d1a178af0ff2535dc8fb4aaf18ae69ddac73375be124294428e723f775fcd7a01b394d65aa623875393f5dcf8c60e51b2709
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 5e3ff263b9b236c78af8fec2372e42ffaa5518a95b086ebe7cd133d0553581ccdba52048297614913ef5f9580a2c2a978ac99152c4cf8871bbab9986c61efb96
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 2bcddf4c9ad002cc8314ff528ae8d91cc3e83123ab8666ae92ea15e024469a01ff0ae18558f521489ec1e0e07f268f5e2324943243bb8fdf3f927205843f057d
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | cf1f4dddbc37d77b41e5bc2cea7c4086d1a45dc018a9b8a2cd91764c70c4818c4deb2d5be451720c47ee373bec2e84f9aba64b99b3363cf98534acf340cf03e3
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 5f8ded94c3833e3748eab6e45192aa49ec50adf6eb7eca57f9342d96c8592d7a33860c794a522276212b988d751bdbc07ff345eb024fc2a29488cca25ea6ddef

## Changelog since v1.23.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Docker runtime support using dockshim in the kubelet is now completely removed in 1.24. The kubelet used to have a a module called "dockershim" which implements CRI support for Docker and it has seen maintenance issues in the Kubernetes community. From 1.24 onwards, please move to a container runtime that is a full-fledged implementation of CRI (v1alpha1 or v1 compliant) as they become available. ([#97252](https://github.com/kubernetes/kubernetes/pull/97252), [@dims](https://github.com/dims)) [SIG Cloud Provider, Instrumentation, Network, Node and Testing]
 
## Changes by Kind

### Feature

- Kubernetes is now built with Golang 1.17.4 ([#106833](https://github.com/kubernetes/kubernetes/pull/106833), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- The `NamespaceDefaultLabelName` feature gate, GA since v1.22, is now removed. ([#106838](https://github.com/kubernetes/kubernetes/pull/106838), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps and Node]

### Bug or Regression

- Address a bug in rbd migration translation plugin ([#106878](https://github.com/kubernetes/kubernetes/pull/106878), [@humblec](https://github.com/humblec)) [SIG Storage]
- Fix bug in error messaging for basic-auth and ssh secret validations. ([#106179](https://github.com/kubernetes/kubernetes/pull/106179), [@vivek-koppuru](https://github.com/vivek-koppuru)) [SIG Apps and Auth]
- Kubeadm: allow the "certs check-expiration" command to not require the existence of the cluster CA key (ca.key file) when checking the expiration of managed certificates in kubeconfig files. ([#106854](https://github.com/kubernetes/kubernetes/pull/106854), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Publishing kube-proxy metrics for Windows kernel-mode ([#106581](https://github.com/kubernetes/kubernetes/pull/106581), [@knabben](https://github.com/knabben)) [SIG Instrumentation, Network and Windows]
- The deprecated flag `--really-crash-for-testing` is removed. ([#101719](https://github.com/kubernetes/kubernetes/pull/101719), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Network, Node and Testing]
- [Metrics Server] Bump image to v0.5.2 ([#106492](https://github.com/kubernetes/kubernetes/pull/106492), [@serathius](https://github.com/serathius)) [SIG Cloud Provider and Instrumentation]

### Other (Cleanup or Flake)

- Added an example for the kubectl plugin list command. ([#106600](https://github.com/kubernetes/kubernetes/pull/106600), [@bergerhoffer](https://github.com/bergerhoffer)) [SIG CLI]
- Kubelet config validation error messages are updated ([#105360](https://github.com/kubernetes/kubernetes/pull/105360), [@shuheiktgw](https://github.com/shuheiktgw)) [SIG Node]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
- github.com/containernetworking/cni: [v0.8.1](https://github.com/containernetworking/cni/tree/v0.8.1)