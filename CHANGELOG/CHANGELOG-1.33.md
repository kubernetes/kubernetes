<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.33.0-alpha.1](#v1330-alpha1)
  - [Downloads for v1.33.0-alpha.1](#downloads-for-v1330-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.32.0](#changelog-since-v1320)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
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

<!-- END MUNGE: GENERATED_TOC -->

# v1.33.0-alpha.1


## Downloads for v1.33.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes.tar.gz) | 809c3565365eccf43761888113fe63c37a700edb6c662f4a29b93768d8d49d6c8ef052a6ffc41f61e9eecb22e006dc03c4399ad05886dc6a7635b2e573d0097d
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-src.tar.gz) | 204a8f6723e8c0b0350994174b43f3a9272dacbd4f2992919b8ec95748df6af53dea385210b89417f1eeaa733732fee6c80559f0779f02f7cb73ccde6384bc9b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 7762f1e33b94102a7fb943dfda3067e69ac534aeca040e95462781bd5973ee2436fe60c4ca2eeaea79f210a07c91167629d620bafc5b108839c02a4865ee0b64
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | ece5bda2f89981659957cc7bc40cd7db20283778c8f1755b9a21499057ec808708eeb7db3f195c0231ba43a0fd9165fb4bf6367183a486d82145414db2327790
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 559689427abb113695ea3a1a1b3cbd388c0887dc8f775878337c1d413c1eb0fccfad161c9af23d7a40a0536b438bd800078fae182fcfde2905568ef4079b1062
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | ba65065523407b5596a9efc53f7dd2e5e37b39c3968bbdb13a50944a80635dfc5903395741b5cb0f5f24482384788271fa1354b56f7f6b0b2f7482237aea8cc8
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 585edd8319aec86378c16da7515f42fdcae5c618fba5dfba4af1455d5db8f5433fe16b95ff7193a2e648a847261ea51d3b412133459d33b48159ddf695a76f26
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 5d228232661dd237df57181920ee73008e1b28eda0366a85d125f569b15a21ebae8f9e2536b244908f9f82184e097b4ac9722863eed352cd0c957b7444bcc5fa
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 59e93927f46aff4f304ccad25a0d6220fa643c42c81b65015bd450d7615a809a8b4912efba0e66fe37f33def4b9fe77785ce43688582003c849377bde3277006
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 7c3bd8c464b0a46a216deb1144e3b042cc218464de6e418345a644024de09a04ec78e13a7c5a3f17d90ad9fda254482dd17d05ae67cd267ee2e0504da8258cf2
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 0ea8503268858c551f9b9e51eb360cc160c76cb19c72c434df79ed421766bcb9addd33e6092525ab8e3556f217ae55dfc13f4506afd27585b5031118a6005403
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | f811e3c8e5b4fa31f9ae3493d757b4511de6cf0fc37a161da3c25f1503cf11149af6b79b9abf11314abf2e4cf410f1e41b10414981c141f702bec297a2beeae7
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | a8dfbb963a5d719dc8890ef14340ce35880e006955a229ff9204bb35da2a29df41b6797dc02269f2cc8de361014f8dd6b2535a9414359b48d820ff2cf536c4e1

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | daf5f5f38ab4357a724d688bfc33f3344f340fc4896d6d0c3da777beb76abe133707bbb6bd47cb954cd46bd62d5f4a7311fcaa5cd99f3389472d846c15d2e604
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 28d03d130e28eb7e812db35ca387eb515dfe8c21bbb2e7690285343d381ecd87828c0362ad19b3d13ec8d1d37763924cf9fdb1d814eb75d6e695322c27db06b4
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | b479688f8aaa93d48d5809d21f21837b67144a5c115370f5154b9a13005f47e579f9f54b8f6d371e97165bd4f1a3d8eda85d2a37c83ac1615ca4dad7155d9a6e
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | ed02308911595375b313b7df2fc6ad94b7dbcfc6f57fb0b9ced5512c4eca8f086852ea24bbfa7f3c146dc9cb98a1e5964dfc911dd46e41f815eeb884b82efdab

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 846d0079fe2c53bdec279d6cc185f968cfed908762ce63c053830fdaeda78da4856f19253f98b908406694179da82dd2c387a4a08ad01d2522dc67832c7e2ac5
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | c6b35f71acf7e9009ba1c6d274f1d2655039a0de59c0dd3f544bf240a8e74c43fa7bf830377f7d87dc14ce271e2f312a85930804ddd236a6877d13410131028e
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | c67735374d4f9062c495040c1bb28fc7f15362908d116542e663c58c900fc5e7939468118603d2233c8a951175484d839039f9d2ee1e0473e227fa994a391480
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 2161369d2590959d8d28f81fa1d642028c816a4ce761d7af3d3edae369cda2a58fe8fa466d16e071d34148331ae572512421296ec53a1f5a1312a00376d67a01
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.33.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | f8051a237f06566e6bfd51881e1ae50a359b76dd5c8865ba6f3bf936e8be327a9a71d22192e252d49a2fb243be601fd2ceb17ea989b21e57c35f833e7b977341

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.33.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.33.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.33.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.33.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.33.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.33.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.32.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Action required for custom plugin developers.
  The `UpdatePodTolerations` action type is renamed to `UpdatePodToleration`, you have to follow the renaming if you're using it. ([#129023](https://github.com/kubernetes/kubernetes/pull/129023), [@zhifei92](https://github.com/zhifei92)) [SIG Scheduling and Testing]
 
## Changes by Kind

### API Change

- A new status field `.status.terminatingReplicas` is added to Deployments and ReplicaSets to allow tracking of terminating pods when the DeploymentPodReplacementPolicy feature-gate is enabled. ([#128546](https://github.com/kubernetes/kubernetes/pull/128546), [@atiratree](https://github.com/atiratree)) [SIG API Machinery, Apps and Testing]
- DRA API: the maximum number of pods which can use the same ResourceClaim is now 256 instead of 32. Beware that downgrading a cluster where this relaxed limit is in use to Kubernetes 1.32.0 is not supported because 1.32.0 would refuse to update ResourceClaims with more than 32 entries in the status.reservedFor field. ([#129543](https://github.com/kubernetes/kubernetes/pull/129543), [@pohly](https://github.com/pohly)) [SIG API Machinery, Node and Testing]
- DRA: CEL expressions using attribute strings exceeded the cost limit because their cost estimation was incomplete. ([#129661](https://github.com/kubernetes/kubernetes/pull/129661), [@pohly](https://github.com/pohly)) [SIG Node]
- DRA: when asking for "All" devices on a node, Kubernetes <= 1.32 proceeded to schedule pods onto nodes with no devices by not allocating any devices for those pods. Kubernetes 1.33 changes that to only picking nodes which have at least one device. Users who want the "proceed with scheduling also without devices" semantic can use the upcoming prioritized list feature with one sub-request for "all" devices and a second alternative with "count: 0". ([#129560](https://github.com/kubernetes/kubernetes/pull/129560), [@bart0sh](https://github.com/bart0sh)) [SIG API Machinery and Node]
- Graduate MultiCIDRServiceAllocator to stable and DisableAllocatorDualWrite to beta (disabled by default).
  Action required for Kubernetes distributions that manage the cluster Service CIDR.
  This feature allows users to define the cluster Service CIDR via a new API object: ServiceCIDR.
  Distributions or administrators of Kubernetes may want to control that new Service CIDRs added to the cluster does not overlap with other networks on the cluster, that only belong to a specific range of IPs or just simple retain the existing behavior of only having one ServiceCIDR per cluster.  An example of a Validation Admission Policy to achieve this is:
  
  ---
  apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingAdmissionPolicy
  metadata:
    name: "servicecidrs.default"
  spec:
    failurePolicy: Fail
    matchConstraints:
      resourceRules:
      - apiGroups:   ["networking.k8s.io"]
        apiVersions: ["v1","v1beta1"]
        operations:  ["CREATE", "UPDATE"]
        resources:   ["servicecidrs"]
    matchConditions:
    - name: 'exclude-default-servicecidr'
      expression: "object.metadata.name != 'kubernetes'"
    variables:
    - name: allowed
      expression: "['10.96.0.0/16','2001:db8::/64']"
    validations:
    - expression: "object.spec.cidrs.all(i , variables.allowed.exists(j , cidr(j).containsCIDR(i)))"
  ---
  apiVersion: admissionregistration.k8s.io/v1
  kind: ValidatingAdmissionPolicyBinding
  metadata:
    name: "servicecidrs-binding"
  spec:
    policyName: "servicecidrs.default"
    validationActions: [Deny,Audit]
  --- ([#128971](https://github.com/kubernetes/kubernetes/pull/128971), [@aojea](https://github.com/aojea)) [SIG Apps, Architecture, Auth, CLI, Etcd, Network, Release and Testing]
- Kubenetes starts validating NodeSelectorRequirement's values when creating pods. ([#128212](https://github.com/kubernetes/kubernetes/pull/128212), [@AxeZhan](https://github.com/AxeZhan)) [SIG Apps and Scheduling]
- Kubernetes components that accept x509 client certificate authentication now read the user UID from a certificate subject name RDN with object id 1.3.6.1.4.1.57683.2. An RDN with this object id must contain a string value, and appear no more than once in the certificate subject. Reading the user UID from this RDN can be disabled by setting the beta feature gate `AllowParsingUserUIDFromCertAuth` to false (until the feature gate graduates to GA). ([#127897](https://github.com/kubernetes/kubernetes/pull/127897), [@modulitos](https://github.com/modulitos)) [SIG API Machinery, Auth and Testing]
- Removed general available feature-gate `PDBUnhealthyPodEvictionPolicy`. ([#129500](https://github.com/kubernetes/kubernetes/pull/129500), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps and Auth]
- `kubectl apply` now coerces `null` values for labels and annotations in manifests to empty string values, consistent with typed JSON metadata decoding, rather than dropping all labels and annotations ([#129257](https://github.com/kubernetes/kubernetes/pull/129257), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]

### Feature

- Add unit test helpers to validate CEL and patterns in CustomResourceDefinitions. ([#129028](https://github.com/kubernetes/kubernetes/pull/129028), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Added a `/flagz` endpoint for kube-proxy ([#128985](https://github.com/kubernetes/kubernetes/pull/128985), [@yongruilin](https://github.com/yongruilin)) [SIG Instrumentation and Network]
- Added a `/status` endpoint for kube-proxy ([#128989](https://github.com/kubernetes/kubernetes/pull/128989), [@Henrywu573](https://github.com/Henrywu573)) [SIG Instrumentation and Network]
- Added e2e tests for volume group snapshots. ([#128972](https://github.com/kubernetes/kubernetes/pull/128972), [@manishym](https://github.com/manishym)) [SIG Cloud Provider, Storage and Testing]
- Adds a /flagz endpoint for kube-scheduler endpoint ([#128818](https://github.com/kubernetes/kubernetes/pull/128818), [@yongruilin](https://github.com/yongruilin)) [SIG Architecture, Instrumentation, Scheduling and Testing]
- Adds a /statusz endpoint for kubelet endpoint ([#128811](https://github.com/kubernetes/kubernetes/pull/128811), [@zhifei92](https://github.com/zhifei92)) [SIG Architecture, Instrumentation and Node]
- Bugfix: Ensure container-level swap metrics are collected ([#129486](https://github.com/kubernetes/kubernetes/pull/129486), [@iholder101](https://github.com/iholder101)) [SIG Node and Testing]
- Calculated pod resources are now cached when adding pods to NodeInfo in the scheduler framework, improving performance when processing unschedulable pods. ([#129635](https://github.com/kubernetes/kubernetes/pull/129635), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Cel-go has been bumped to v0.23.2. ([#129844](https://github.com/kubernetes/kubernetes/pull/129844), [@cici37](https://github.com/cici37)) [SIG API Machinery, Auth, Cloud Provider and Node]
- Client-go/rest: fully supports contextual logging. BackoffManagerWithContext should be used instead of BackoffManager to ensure that the caller can interrupt the sleep. ([#127709](https://github.com/kubernetes/kubernetes/pull/127709), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, Cloud Provider, Instrumentation, Network and Node]
- Graduated the `KubeletFineGrainedAuthz` feature gate to beta; the gate is now enabled by default. ([#129656](https://github.com/kubernetes/kubernetes/pull/129656), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Auth, CLI, Node, Storage and Testing]
- Improved scheduling performance of pods with required topology spreading. ([#129119](https://github.com/kubernetes/kubernetes/pull/129119), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Kube-apiserver: Promoted the  `ServiceAccountTokenNodeBinding` feature gate general availability. It is now locked to enabled. ([#129591](https://github.com/kubernetes/kubernetes/pull/129591), [@liggitt](https://github.com/liggitt)) [SIG Auth and Testing]
- Kube-proxy extends the schema of its healthz/ and livez/ endpoints to incorporate information about the corresponding IP family ([#129271](https://github.com/kubernetes/kubernetes/pull/129271), [@aroradaman](https://github.com/aroradaman)) [SIG Network and Windows]
- Kubeadm: graduated the WaitForAllControlPlaneComponents feature gate to Beta. When checking the health status of a control plane component, make sure that the address and port defined as arguments in the respective component's static Pod manifest are used. ([#129620](https://github.com/kubernetes/kubernetes/pull/129620), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: if the `NodeLocalCRISocket` feature gate is enabled, remove the `kubeadm.alpha.kubernetes.io/cri-socket` annotation from a given node on `kubeadm upgrade`. ([#129279](https://github.com/kubernetes/kubernetes/pull/129279), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: if the `NodeLocalCRISocket` feature gate is enabled, remove the flag `--container-runtime-endpoint` from the `/var/lib/kubelet/kubeadm-flags.env` file on `kubeadm upgrade`. ([#129278](https://github.com/kubernetes/kubernetes/pull/129278), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Kubeadm: promoted the feature gate `ControlPlaneKubeletLocalMode` to Beta. Kubeadm will per default use the local kube-apiserver endpoint for the kubelet when creating a cluster with "kubeadm init" or when joining control plane nodes with "kubeadm join". Enabling the feature gate also affects the `kubeadm init phase kubeconfig kubelet` phase, where the flag `--control-plane-endpoint` no longer affects the generated kubeconfig `Server` field, but the flag `--apiserver-advertise-address` can now be used for the same purpose. ([#129956](https://github.com/kubernetes/kubernetes/pull/129956), [@chrischdi](https://github.com/chrischdi)) [SIG Cluster Lifecycle]
- Kubeadm: removed preflight check for nsenter on Linux nodes
  kubeadm: added preflight check for `losetup` on Linux nodes. It's required by kubelet for keeping a block device opened. ([#129450](https://github.com/kubernetes/kubernetes/pull/129450), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: removed the feature gate EtcdLearnerMode which graduated to GA in 1.32. ([#129589](https://github.com/kubernetes/kubernetes/pull/129589), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes is now built with go 1.23.4 ([#129422](https://github.com/kubernetes/kubernetes/pull/129422), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with go 1.23.5 ([#129962](https://github.com/kubernetes/kubernetes/pull/129962), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Promoted the feature gate `CSIMigrationPortworx` to GA. If your applications are using Portworx volumes, please make sure that the corresponding Portworx CSI driver is installed on your cluster **before** upgrading to 1.31 or later because all operations for the in-tree `portworxVolume` type are redirected to the pxd.portworx.com CSI driver when the feature gate is enabled. ([#129297](https://github.com/kubernetes/kubernetes/pull/129297), [@gohilankit](https://github.com/gohilankit)) [SIG Storage]
- The `SidecarContainers` feature has graduated to GA. 'SidecarContainers' feature gate was locked to default value and will be removed in v1.36. If you were setting this feature gate explicitly, please remove it now. ([#129731](https://github.com/kubernetes/kubernetes/pull/129731), [@gjkim42](https://github.com/gjkim42)) [SIG Apps, Node, Scheduling and Testing]
- Upgrade autoscalingv1 to autoscalingv2 in kubectl autoscale cmd, The cmd will attempt to use the autoscaling/v2 API first. If the autoscaling/v2 API is not available or an error occurs, it will fall back to the autoscaling/v1 API. ([#128950](https://github.com/kubernetes/kubernetes/pull/128950), [@googs1025](https://github.com/googs1025)) [SIG Autoscaling and CLI]
- Validate ContainerLogMaxFiles in kubelet config validation ([#129072](https://github.com/kubernetes/kubernetes/pull/129072), [@kannon92](https://github.com/kannon92)) [SIG Node]

### Documentation

- Give example of set-based requirement for -l/--selector flag ([#129106](https://github.com/kubernetes/kubernetes/pull/129106), [@rotsix](https://github.com/rotsix)) [SIG CLI]
- Kubeadm: improved the `kubeadm reset` message for manual cleanups and referenced https://k8s.io/docs/reference/setup-tools/kubeadm/kubeadm-reset/. ([#129644](https://github.com/kubernetes/kubernetes/pull/129644), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]

### Bug or Regression

- --feature-gate=InOrderInformers (default on), causes informers to process watch streams in order as opposed to grouping updates for the same item close together.  Binaries embedding client-go, but not wiring the featuregates can disable by setting the `KUBE_FEATURE_InOrderInformers=false`. ([#129568](https://github.com/kubernetes/kubernetes/pull/129568), [@deads2k](https://github.com/deads2k)) [SIG API Machinery]
- Adding a validation for revisionHistoryLimit field in statefulset.spec to prevent it being set to negative value. ([#129017](https://github.com/kubernetes/kubernetes/pull/129017), [@ardaguclu](https://github.com/ardaguclu)) [SIG Apps]
- DRA: the explanation for why a pod which wasn't using ResourceClaims was unscheduleable included a useless "no new claims to deallocate" when it was unscheduleable for some other reasons. ([#129823](https://github.com/kubernetes/kubernetes/pull/129823), [@googs1025](https://github.com/googs1025)) [SIG Node and Scheduling]
- Enables ratcheting validation on status subresources for CustomResourceDefinitions ([#129506](https://github.com/kubernetes/kubernetes/pull/129506), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery]
- Fix the issue where the named ports exposed by restartable init containers (a.k.a. sidecar containers) cannot be accessed using a Service. ([#128850](https://github.com/kubernetes/kubernetes/pull/128850), [@toVersus](https://github.com/toVersus)) [SIG Network and Testing]
- Fixed `kubectl wait --for=create` behavior with label selectors, to properly wait for resources with matching labels to appear. ([#128662](https://github.com/kubernetes/kubernetes/pull/128662), [@omerap12](https://github.com/omerap12)) [SIG CLI and Testing]
- Fixed a bug where adding an ephemeral container to a pod which references a new secret or config map doesn't give the pod access to that new secret or config map. (#114984, @cslink) ([#129670](https://github.com/kubernetes/kubernetes/pull/129670), [@cslink](https://github.com/cslink)) [SIG Auth]
- Fixed a data race that could occur when a single Go type was serialized to CBOR concurrently for the first time within a program. ([#129170](https://github.com/kubernetes/kubernetes/pull/129170), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Fixed a storage bug around multipath. iSCSI and Fibre Channel devices attached to nodes via multipath now resolve correctly if partitioned. ([#128086](https://github.com/kubernetes/kubernetes/pull/128086), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]
- Fixed in-tree to CSI migration for Portworx volumes, in clusters where Portworx security feature is enabled (it's a Portworx feature, not Kubernetes feature). It required secret data from the secret mentioned in-tree SC, to be passed in CSI requests which was not happening before this fix. ([#129630](https://github.com/kubernetes/kubernetes/pull/129630), [@gohilankit](https://github.com/gohilankit)) [SIG Storage]
- Fixed: kube-proxy EndpointSliceCache memory is leaked ([#128929](https://github.com/kubernetes/kubernetes/pull/128929), [@orange30](https://github.com/orange30)) [SIG Network]
- Fixes CVE-2024-51744 ([#128621](https://github.com/kubernetes/kubernetes/pull/128621), [@kmala](https://github.com/kmala)) [SIG Auth, Cloud Provider and Node]
- Fixes a panic in kube-controller-manager handling StatefulSet objects when revisionHistoryLimit is negative ([#129301](https://github.com/kubernetes/kubernetes/pull/129301), [@ardaguclu](https://github.com/ardaguclu)) [SIG Apps]
- HPA's with ContainerResource metrics will no longer error when container metrics are missing, instead they will use the same logic Resource metrics are using to make calculations ([#127193](https://github.com/kubernetes/kubernetes/pull/127193), [@DP19](https://github.com/DP19)) [SIG Apps and Autoscaling]
- Implemented logging and event recording for probe results with an `Unknown` status in the kubelet's prober module. This helps in better diagnosing and monitoring cases where container probes return an `Unknown` result, improving the observability and reliability of health checks. ([#125901](https://github.com/kubernetes/kubernetes/pull/125901), [@jralmaraz](https://github.com/jralmaraz)) [SIG Node]
- Improved reboot event reporting. The kubelet will only emit one reboot Event when a server-level reboot is detected, even if the kubelet cannot write its status to the associated Node (which triggers a retry). ([#129151](https://github.com/kubernetes/kubernetes/pull/129151), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Kube-apiserver: --service-account-max-token-expiration can now be used in combination with an external token signer --service-account-signing-endpoint, as long as the --service-account-max-token-expiration is not longer than the external token signer's max expiration. ([#129816](https://github.com/kubernetes/kubernetes/pull/129816), [@sambdavidson](https://github.com/sambdavidson)) [SIG API Machinery and Auth]
- Kubeadm: avoid loading the file passed to `--kubeconfig` during `kubeadm init` phases more than once. ([#129006](https://github.com/kubernetes/kubernetes/pull/129006), [@kokes](https://github.com/kokes)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where the 'node.skipPhases' in UpgradeConfiguration is not respected by 'kubeadm upgrade node' command ([#129452](https://github.com/kubernetes/kubernetes/pull/129452), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: fixed a bug where an image is not pulled if there is an error with the sandbox image from CRI. ([#129594](https://github.com/kubernetes/kubernetes/pull/129594), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fixed the bug where the v1beta4 Timeouts.EtcdAPICall field was not respected in etcd client operations, and the default timeout of 2 minutes was always used. ([#129859](https://github.com/kubernetes/kubernetes/pull/129859), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: if an addon is disabled in the ClusterConfiguration, skip it during upgrade. ([#129418](https://github.com/kubernetes/kubernetes/pull/129418), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: run kernel version and OS version preflight checks on `kubeadm upgrade`. ([#129401](https://github.com/kubernetes/kubernetes/pull/129401), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Provides an additional function argument to directly specify the version for the tools that the consumers wishes to use ([#129658](https://github.com/kubernetes/kubernetes/pull/129658), [@unmarshall](https://github.com/unmarshall)) [SIG API Machinery]
- Remove the limitation on exposing port 10250 externally in service. ([#129174](https://github.com/kubernetes/kubernetes/pull/129174), [@RyanAoh](https://github.com/RyanAoh)) [SIG Apps and Network]
- This PR changes the signature of the `PublishResources` to accept a `resourceslice.DriverResources` parameter instead of a `Resources` parameter. ([#129142](https://github.com/kubernetes/kubernetes/pull/129142), [@googs1025](https://github.com/googs1025)) [SIG Node and Testing]
- [kubectl] Improved the describe output for projected volume sources to clearly indicate whether Secret and ConfigMap entries are optional. ([#129457](https://github.com/kubernetes/kubernetes/pull/129457), [@gshaibi](https://github.com/gshaibi)) [SIG CLI]

### Other (Cleanup or Flake)

- Implemented scheduler_cache_size metric.
  Also, scheduler_scheduler_cache_size metric is deprecated in favor of scheduler_cache_size, 
  and will be removed at v1.34. ([#128810](https://github.com/kubernetes/kubernetes/pull/128810), [@googs1025](https://github.com/googs1025)) [SIG Scheduling]
- Kube-apiserver: inactive serving code is removed for authentication.k8s.io/v1alpha1 APIs ([#129186](https://github.com/kubernetes/kubernetes/pull/129186), [@liggitt](https://github.com/liggitt)) [SIG Auth and Testing]
- Kube-proxy extends the schema of its metrics/ endpoints to incorporate information about the corresponding IP family ([#129173](https://github.com/kubernetes/kubernetes/pull/129173), [@aroradaman](https://github.com/aroradaman)) [SIG Network and Windows]
- Kube-proxy nftables logs the failed transactions and the full table when using log level 4 or higher. Logging is rate limited to one entry every 24 hours to avoid performance issues. ([#128886](https://github.com/kubernetes/kubernetes/pull/128886), [@npinaeva](https://github.com/npinaeva)) [SIG Network]
- Kubeadm: removed preflight check for `ip`, `iptables`, `ethtool` and `tc` on Linux nodes. kubelet and kube-proxy will continue to report `iptables` errors if its usage is required. The tools `ip`, `ethtool` and `tc` had legacy usage in the kubelet but are no longer required. ([#129131](https://github.com/kubernetes/kubernetes/pull/129131), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: removed preflight check for `touch` on Linux nodes. ([#129317](https://github.com/kubernetes/kubernetes/pull/129317), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- NOE ([#128856](https://github.com/kubernetes/kubernetes/pull/128856), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Apps and Network]
- Removed generally available feature gate `KubeProxyDrainingTerminatingNodes`. ([#129692](https://github.com/kubernetes/kubernetes/pull/129692), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Network]
- Removed support for v1alpha1 version of ValidatingAdmissionPolicy and ValidatingAdmissionPolicyBinding API kinds. ([#129207](https://github.com/kubernetes/kubernetes/pull/129207), [@Jefftree](https://github.com/Jefftree)) [SIG Etcd and Testing]
- The deprecated pod_scheduling_duration_seconds metric is removed.
  You can migrate to pod_scheduling_sli_duration_seconds. ([#128906](https://github.com/kubernetes/kubernetes/pull/128906), [@sanposhiho](https://github.com/sanposhiho)) [SIG Instrumentation and Scheduling]
- This renames some coredns metrics, see https://github.com/coredns/coredns/blob/v1.11.0/plugin/forward/README.md#metrics. ([#129175](https://github.com/kubernetes/kubernetes/pull/129175), [@DamianSawicki](https://github.com/DamianSawicki)) [SIG Cloud Provider]
- This renames some coredns metrics, see https://github.com/coredns/coredns/blob/v1.11.0/plugin/forward/README.md#metrics. ([#129232](https://github.com/kubernetes/kubernetes/pull/129232), [@DamianSawicki](https://github.com/DamianSawicki)) [SIG Cloud Provider]
- Updated CNI plugins to v1.6.2. ([#129776](https://github.com/kubernetes/kubernetes/pull/129776), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.32.0. ([#129116](https://github.com/kubernetes/kubernetes/pull/129116), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Upgrade CoreDNS to v1.12.0 ([#128926](https://github.com/kubernetes/kubernetes/pull/128926), [@bzsuni](https://github.com/bzsuni)) [SIG Cloud Provider and Cluster Lifecycle]

## Dependencies

### Added
- gopkg.in/go-jose/go-jose.v2: v2.6.3

### Changed
- cel.dev/expr: v0.18.0 → v0.19.1
- github.com/coredns/corefile-migration: [v1.0.24 → v1.0.25](https://github.com/coredns/corefile-migration/compare/v1.0.24...v1.0.25)
- github.com/coreos/go-oidc: [v2.2.1+incompatible → v2.3.0+incompatible](https://github.com/coreos/go-oidc/compare/v2.2.1...v2.3.0)
- github.com/cyphar/filepath-securejoin: [v0.3.4 → v0.3.5](https://github.com/cyphar/filepath-securejoin/compare/v0.3.4...v0.3.5)
- github.com/davecgh/go-spew: [d8f796a → v1.1.1](https://github.com/davecgh/go-spew/compare/d8f796a...v1.1.1)
- github.com/golang-jwt/jwt/v4: [v4.5.0 → v4.5.1](https://github.com/golang-jwt/jwt/compare/v4.5.0...v4.5.1)
- github.com/google/btree: [v1.0.1 → v1.1.3](https://github.com/google/btree/compare/v1.0.1...v1.1.3)
- github.com/google/cel-go: [v0.22.0 → v0.23.2](https://github.com/google/cel-go/compare/v0.22.0...v0.23.2)
- github.com/google/gnostic-models: [v0.6.8 → v0.6.9](https://github.com/google/gnostic-models/compare/v0.6.8...v0.6.9)
- github.com/pmezard/go-difflib: [5d4384e → v1.0.0](https://github.com/pmezard/go-difflib/compare/5d4384e...v1.0.0)
- golang.org/x/crypto: v0.28.0 → v0.31.0
- golang.org/x/net: v0.30.0 → v0.33.0
- golang.org/x/sync: v0.8.0 → v0.10.0
- golang.org/x/sys: v0.26.0 → v0.28.0
- golang.org/x/term: v0.25.0 → v0.27.0
- golang.org/x/text: v0.19.0 → v0.21.0
- k8s.io/kube-openapi: 32ad38e → 2c72e55
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.31.0 → v0.31.1
- sigs.k8s.io/kustomize/api: v0.18.0 → v0.19.0
- sigs.k8s.io/kustomize/cmd/config: v0.15.0 → v0.19.0
- sigs.k8s.io/kustomize/kustomize/v5: v5.5.0 → v5.6.0
- sigs.k8s.io/kustomize/kyaml: v0.18.1 → v0.19.0

### Removed
- github.com/asaskevich/govalidator: [f61b66f](https://github.com/asaskevich/govalidator/tree/f61b66f)
- gopkg.in/square/go-jose.v2: v2.6.0