<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.35.0-alpha.1](#v1350-alpha1)
  - [Downloads for v1.35.0-alpha.1](#downloads-for-v1350-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.34.0](#changelog-since-v1340)
  - [Changes by Kind](#changes-by-kind)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)

<!-- END MUNGE: GENERATED_TOC -->

# v1.35.0-alpha.1


## Downloads for v1.35.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes.tar.gz) | 1d6fb6a4c7f82fe04e56757b733c3fc4aac652f8c2113e79ddce83b6cbe0179404147b35ddbc18e1b60eb802acb3f6d884599fd573f3d16f0558ef7ddfb8aae2
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-src.tar.gz) | 364788bac4d405ac6180fe3cb7e3d847e7960fcb0532146b105270aeac2624ade2ff87370c5aa8f768eda07fd28e5e75f73afbdf9cc1b786827a0e123bdea561

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 1ba40849b104851d922bce32dc9306004e9b95cfadeff9ecfb65f779892009f9a70878b8efe96159088b1ad8c700bf19e58d68416dfdff7853660e6074dd3752
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 0c939895ad2d53f57e9137774eed99cbfbfa5f15d4276f5f55c4ea40b922a5f37ab375a065fdd330f5a1ddf452896f2a13621075b049f382ab42a65ea1085dac
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-386.tar.gz) | e98a9b2d5f1c8bec552be6353b623a8f12078befd968a662a933907d0ff72b0164fa8b38e4cbd4aae6191e28aedc19d996ec99298053d7bafc458f91580a7cfa
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 34ab1e9edf70c84fe58a223e91b0bf679e5d1273a2b6503a18a61a4bea79231948efe098f84e39f83dbfa2c5271aad8e9819aac104d6c3360c97e5c348b15be7
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 554f8240597e7eb8470c8e2b4bca33c06a0a91746831ef93f76b344f5c3d6226d4ef26cb59f127d1436ac091b7b79395320fe8a9b2acc512afd601989c138d4a
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 62c750654e898622aa87b2d81d4b0cbaf36614899f37181c2e3a6aa645d2270c4dedb7d6e7b974059a716ad144a5615407e6a9a4c03f761a512d37fdda796e50
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | bbe22979c4e300675dfa955ce9855b7b33b29119a9f78e58bc1b088dba2b8dedbf0b068092d42cd98dbc10cda1da317eae6414b91587593678ec76780ff575df
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | b8360d0bf930149d360da2a95549b35cb7e14932ae8507d99e34d93729ae645bab203dfba325c74db13204e09e5ee032f887cdd67badfbf3bc8a08d71ccf9c3d
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-windows-386.tar.gz) | cc935a74f30dcd1eaaeadc8f2353a9742ebc4a36b133342c6402b065750f4028a1a392bd5f7ea51533c2d799ff2bdb3d0f21493e7fabacb27289e58f011bb229
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 119742103985be0cd4296b85aa2c713cdc510b9a9412706fdf88ca1c703f69338146efc5cf37168ab56e74576ad561ce37c3f500d29b63002139d11544b1b7cf
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | 67f5fe6b14aa4c49acb6f706ec4a9e43b87f1e19555579895452183e0b2d2be2202f8d48622208ac5ef6e0fb9050d99bb7e1ed9e4e31e8fcac7c0b5e44787c39

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | a1c935db625766b02113087068fa1087a6b74e9f57ad72cc1d5d85e830c0569b9257746013053ee8dc89404940458c3bba00064666978ddff4df9f3cae0ae066
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 5f87f8f46719af413fa864b7d91d5b95fa86adf27df63cf904470b15e844ebda4c802d7ec7cb4006b4e5a3780903d0436eb57a7e2f2b79b74cf5e7e8f65496b1
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 9f2e9476ae3b95919c991dd0438b31eeded7c7f686948ef2d6227311dabc952e74f61d3638f224eb18e63a4fe4955b55a9e358032477b989800541617a8b5f6a
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 043311de9dad81d3774decc3aeeac48833d2d234a15ce4a2062fd9af778879afde5e2eb9fdec5f3641725f630e7c3bd845a348ad713b54275e77293941d4c8d0

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 865b4ee818cd53bc91001f658243e6b6fd9464f17ef8dc0cc739586689f39998e5c47630df439ad43553d6830ae3a7375cc780c3a5e49f1422d35d77194efc35
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | fc7df94bc328817d20c59e1ab1371634bf3849141ad982c9d403b136d009c3ed9ee3f7a20659a0f93af7179e13e2c0835b80fefc677b8840f7bcbde0dabb4483
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 8ec718436680766d9026b56ece5bff7a6bac9f63da0edf33843a7a6c255e1a1d22aecf260c1d5fc394c1e2f581931c5ae0acad80a0141fc8d4d7730bf04566b5
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | fff0562e3a89b4f9444ce83bb8cfc860bcd5178a2c1ba0f1404d87556f483d48241aa3f927f2534d811b1c554958c167f1e5199fa916ebfd9a754cef2f761139
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.35.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 7774299a2b581a4ab2514d8dc95c1d0dff0652aa0518efbffa9ac39f5c510cb83d8a41a70e9b4e78f0b179e5a806402310dac161ff3a2398b97755938c225586

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.35.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.34.0

## Changes by Kind

### API Change

- Added WithOrigin within apis/core/validation with adjusted tests ([#132825](https://github.com/kubernetes/kubernetes/pull/132825), [@PatrickLaabs](https://github.com/PatrickLaabs)) [SIG Apps]
- Component-base: validate that log-flush-frequency is positive and return an error instead of panic-ing ([#133540](https://github.com/kubernetes/kubernetes/pull/133540), [@BenTheElder](https://github.com/BenTheElder)) [SIG Architecture, Instrumentation, Network and Node]
- Feature gate dependencies are now explicit, and validated at startup. A feature can no longer be enabled if it depends on a disabled feature. In particular, this means that `AllAlpha=true` will no longer work without enabling disabled-by-default beta features that are depended on (either with `AllBeta=true` or explicitly enumerating the disabled dependencies). ([#133697](https://github.com/kubernetes/kubernetes/pull/133697), [@tallclair](https://github.com/tallclair)) [SIG API Machinery, Architecture, Cluster Lifecycle and Node]
- In version 1.34, the PodObservedGenerationTracking feature has been upgraded to beta, and the description of the alpha version in the openapi has been removed. ([#133883](https://github.com/kubernetes/kubernetes/pull/133883), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Apps]
- Introduce a new declarative validation tag +k8s:customUnique to control listmap uniqueness ([#134279](https://github.com/kubernetes/kubernetes/pull/134279), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery and Auth]
- Kube-apiserver: Fixed a 1.34 regression in CustomResourceDefinition handling that incorrectly warned about unrecognized formats on number and integer properties ([#133896](https://github.com/kubernetes/kubernetes/pull/133896), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Contributor Experience, Network, Node and Scheduling]
- OpenAPI model packages of API types are generated into `zz_generated.model_name.go` files and are accessible using the `OpenAPIModelName()` function.  This allows API authors to declare the desired OpenAPI model packages instead of using the go package path of API types. ([#131755](https://github.com/kubernetes/kubernetes/pull/131755), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Support for `kubectl get -o kyaml` is now on by default.  To disable it, set `KUBECTL_KYAML=false`. ([#133327](https://github.com/kubernetes/kubernetes/pull/133327), [@thockin](https://github.com/thockin)) [SIG CLI]
- The storage version for MutatingAdmissionPolicy is updated to v1beta1. ([#133715](https://github.com/kubernetes/kubernetes/pull/133715), [@cici37](https://github.com/cici37)) [SIG API Machinery, Etcd and Testing]

### Feature

- Add paths section to kubelet statusz endpoint ([#133239](https://github.com/kubernetes/kubernetes/pull/133239), [@Peac36](https://github.com/Peac36)) [SIG Node]
- Add paths section to scheduler statusz endpoint ([#132606](https://github.com/kubernetes/kubernetes/pull/132606), [@Peac36](https://github.com/Peac36)) [SIG API Machinery, Architecture, Instrumentation, Network, Node, Scheduling and Testing]
- Added kubectl config set-context -n flag as a shorthand for --namespace ([#134384](https://github.com/kubernetes/kubernetes/pull/134384), [@tchap](https://github.com/tchap)) [SIG CLI and Testing]
- Added remote runtime and image `Close()` method to be able to close the connection. ([#133211](https://github.com/kubernetes/kubernetes/pull/133211), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Adds metric for Maxunavailable feature ([#130951](https://github.com/kubernetes/kubernetes/pull/130951), [@Edwinhr716](https://github.com/Edwinhr716)) [SIG Apps and Instrumentation]
- Applyconfiguration-gen now generates extract functions for all subresources ([#132665](https://github.com/kubernetes/kubernetes/pull/132665), [@mrIncompetent](https://github.com/mrIncompetent)) [SIG API Machinery]
- Applyconfiguration-gen now preserves struct and field comments from source types in generated code ([#132663](https://github.com/kubernetes/kubernetes/pull/132663), [@mrIncompetent](https://github.com/mrIncompetent)) [SIG API Machinery]
- DRA: the resource.k8s.io API now uses the v1 API version (introduced in 1.34) as default storage version. Downgrading to 1.33 is not supported. ([#133876](https://github.com/kubernetes/kubernetes/pull/133876), [@kei01234kei](https://github.com/kei01234kei)) [SIG API Machinery, Etcd and Testing]
- Events:
    Type     Reason   Age                 From               Message
    ----     ------   ----                ----               -------
    Warning  Failed   7m11s (x2 over 7m33s) kubelet          spec.containers{nginx}: Failed to pull image "nginx": failed to pull and unpack image... ([#133627](https://github.com/kubernetes/kubernetes/pull/133627), [@itzPranshul](https://github.com/itzPranshul)) [SIG CLI]
- Introduces e2e tests that check component invariant metrics across the entire suite run. ([#133394](https://github.com/kubernetes/kubernetes/pull/133394), [@BenTheElder](https://github.com/BenTheElder)) [SIG Testing]
- K8s.io/apimachinery: Introduce a helper function to compare resourceVersion strings from two objects of the same resource ([#134330](https://github.com/kubernetes/kubernetes/pull/134330), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Apps, Auth, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Kubeadm: graduate the kubeadm specific feature gate ControlPlaneKubeletLocalMode to GA and lock it to enabled by default. To opt-out manually from this desired default behavior you must patch the "server" field in the  /etc/kubernetes/kubelet.conf file. The subphase of "kubeadm join phase control-plane-join" called "etcd" is now deprecated, hidden and replaced by the subphase with identical functionality "etcd-join". "etcd" will be removed in a follow-up release. The subphase "kubelet-wait-bootstrap" of "kubeadm join" is no longer experimental and will always run. ([#134106](https://github.com/kubernetes/kubernetes/pull/134106), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubernetes is now built using Go 1.25.1 ([#134095](https://github.com/kubernetes/kubernetes/pull/134095), [@dims](https://github.com/dims)) [SIG Release and Testing]
- Kubernetes now uses Go Language Version 1.25, including https://go.dev/blog/container-aware-gomaxprocs ([#134120](https://github.com/kubernetes/kubernetes/pull/134120), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling and Storage]
- Lock down the `AllowOverwriteTerminationGracePeriodSeconds` feature gate. ([#133792](https://github.com/kubernetes/kubernetes/pull/133792), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Metrics: exclude dryRun requests from apiserver_request_sli_duration_seconds ([#131092](https://github.com/kubernetes/kubernetes/pull/131092), [@aldudko](https://github.com/aldudko)) [SIG API Machinery and Instrumentation]
- The validation in the resouce.k8s.io has been migrated to declarative validation.
  If the `DeclarativeValidation` feature gate is enabled, mismatches with existing validation are reported via metrics.
  If the `DeclarativeValidationTakeover` feature gate is enabled, declarative validation is the primary source of errors for migrated fields. ([#134072](https://github.com/kubernetes/kubernetes/pull/134072), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps and Auth]

### Bug or Regression

- Added the correct error when eviction is blocked due to the failSafe mechanism of the DisruptionController. ([#133097](https://github.com/kubernetes/kubernetes/pull/133097), [@kei01234kei](https://github.com/kei01234kei)) [SIG Apps and Node]
- Bugfix: the default serviceCIDR controller was not logging events because the event broadcaster was shutdown during its initialization. ([#133338](https://github.com/kubernetes/kubernetes/pull/133338), [@aojea](https://github.com/aojea)) [SIG Network]
- Deprecated metrics will be hidden as per the metrics deprecation policy https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecating-a-metric ([#133436](https://github.com/kubernetes/kubernetes/pull/133436), [@richabanker](https://github.com/richabanker)) [SIG Architecture, Instrumentation and Network]
- Fix incorrect behavior of preemptor pod when preemption of the victim takes long to complete. The preemptor pod should not be circling in scheduling cycles until preemption is finished. ([#134294](https://github.com/kubernetes/kubernetes/pull/134294), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Scheduling and Testing]
- Fix missing kubelet_volume_stats_* metrics ([#133890](https://github.com/kubernetes/kubernetes/pull/133890), [@huww98](https://github.com/huww98)) [SIG Instrumentation and Node]
- Fix occasional schedule delay when the static PV is created ([#133929](https://github.com/kubernetes/kubernetes/pull/133929), [@huww98](https://github.com/huww98)) [SIG Scheduling and Storage]
- Fix resource claims deallocation for extended resource when pod is completed ([#134312](https://github.com/kubernetes/kubernetes/pull/134312), [@alaypatel07](https://github.com/alaypatel07)) [SIG Apps, Node and Testing]
- Fixed SELinux warning controller not emitting events on some SELinux label conflicts. ([#133425](https://github.com/kubernetes/kubernetes/pull/133425), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Storage and Testing]
- Fixed a bug in kube-proxy nftables mode (GA as of 1.33) that fails to determine if traffic originates from a local source on the node. The issue was caused by using the wrong meta `iif` instead of `iifname` for name based matches. ([#134024](https://github.com/kubernetes/kubernetes/pull/134024), [@jack4it](https://github.com/jack4it)) [SIG Network]
- Fixed a bug in kube-scheduler where pending pod preemption caused preemptor pods to be retried more frequently. ([#134245](https://github.com/kubernetes/kubernetes/pull/134245), [@macsko](https://github.com/macsko)) [SIG Scheduling and Testing]
- Fixed a bug that caused apiservers to send an inappropriate Content-Type request header to authorization, token authentication, imagepolicy admission, and audit webhooks when the alpha client-go feature gate "ClientsPreferCBOR" is enabled. ([#132960](https://github.com/kubernetes/kubernetes/pull/132960), [@benluddy](https://github.com/benluddy)) [SIG API Machinery and Node]
- Fixed a bug that caused duplicate validation when updating PersistentVolumeClaims, VolumeAttachments and VolumeAttributesClasses. ([#132549](https://github.com/kubernetes/kubernetes/pull/132549), [@gavinkflam](https://github.com/gavinkflam)) [SIG Storage]
- Fixed a bug that caused duplicate validation when updating role and role binding resources. ([#132550](https://github.com/kubernetes/kubernetes/pull/132550), [@gavinkflam](https://github.com/gavinkflam)) [SIG Auth]
- Fixed a bug where high latency kube-apiserver caused scheduling throughput degradation. ([#134154](https://github.com/kubernetes/kubernetes/pull/134154), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- Fixed broken shell completion for api resources. ([#133771](https://github.com/kubernetes/kubernetes/pull/133771), [@marckhouzam](https://github.com/marckhouzam)) [SIG CLI]
- Fixed validation error when ConfigFlags has CertFile and (or) KeyFile and original config also contains CertFileData and (or) KeyFileData. ([#133917](https://github.com/kubernetes/kubernetes/pull/133917), [@n2h9](https://github.com/n2h9)) [SIG API Machinery and CLI]
- Fixes a possible data race during metrics registration ([#134390](https://github.com/kubernetes/kubernetes/pull/134390), [@liggitt](https://github.com/liggitt)) [SIG Architecture and Instrumentation]
- Implicit extended resource name derived from device class (deviceclass.resource.kubernetes.io/<device-class-name>) can be used to request DRA devices matching the device class. ([#133363](https://github.com/kubernetes/kubernetes/pull/133363), [@yliaog](https://github.com/yliaog)) [SIG Node, Scheduling and Testing]
- Kube-apiserver: Fixes a 1.34 regression with spurious "Error getting keys" log messages ([#133817](https://github.com/kubernetes/kubernetes/pull/133817), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- Kube-apiserver: Fixes a possible 1.34 performance regression calculating object size statistics for resources not served from the watch cache, typically only Events ([#133873](https://github.com/kubernetes/kubernetes/pull/133873), [@serathius](https://github.com/serathius)) [SIG API Machinery and Etcd]
- Kube-apiserver: improve the validation error message shown for custom resources with CEL validation rules to include the value that failed validation ([#132798](https://github.com/kubernetes/kubernetes/pull/132798), [@cbandy](https://github.com/cbandy)) [SIG API Machinery]
- Kube-controller-manager: Fixes a possible data race in the garbage collection controller ([#134379](https://github.com/kubernetes/kubernetes/pull/134379), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Apps]
- Kubeadm: ensured waiting for apiserver uses a local client that doesn't reach to the control plane endpoint and instead reaches directly to the local API server endpoint. ([#134265](https://github.com/kubernetes/kubernetes/pull/134265), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix KUBEADM_UPGRADE_DRYRUN_DIR not honored in upgrade phase when writing kubelet config files ([#134007](https://github.com/kubernetes/kubernetes/pull/134007), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: fixed a bug where the node registration information for a given node was not fetched correctly during "kubeadm upgrade node" and the node name can end up being incorrect in cases where the node name is not the same as the host name. ([#134319](https://github.com/kubernetes/kubernetes/pull/134319), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fixed bug where v1beta3's ClusterConfiguration.APIServer.TimeoutForControlPlane is not respected in newer versions of kubeadm where v1beta4 is the default. ([#133513](https://github.com/kubernetes/kubernetes/pull/133513), [@tom1299](https://github.com/tom1299)) [SIG Cluster Lifecycle]
- Kubelet: the connection to a DRA driver became unusable because of an internal deadlock when a connection was idle for 30 minutes. ([#133926](https://github.com/kubernetes/kubernetes/pull/133926), [@pohly](https://github.com/pohly)) [SIG Node]
- Pod can have multiple volumes reference the same PVC ([#122140](https://github.com/kubernetes/kubernetes/pull/122140), [@huww98](https://github.com/huww98)) [SIG Node, Storage and Testing]
- Previously, `kubectl scale` returned the error message `error: no objects passed to scale <GroupResource> "<ResourceName>" not found` when the specified resource did not exist. 
  For consistency with other commands(e.g. `kubectl get`), it has been changed to just return `Error from server (NotFound): <GroupResource> "<ResourceName>" not found`. ([#134017](https://github.com/kubernetes/kubernetes/pull/134017), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI]
- Promote VAC API test to conformance ([#133615](https://github.com/kubernetes/kubernetes/pull/133615), [@carlory](https://github.com/carlory)) [SIG Architecture, Storage and Testing]
- Remove incorrectly printed warning for SessionAffinity whenever a headless service is creater or updated ([#134054](https://github.com/kubernetes/kubernetes/pull/134054), [@Peac36](https://github.com/Peac36)) [SIG Network]
- The SchedulerAsyncAPICalls feature gate has been disabled to mitigate a bug where its interaction with asynchronous preemption in could degrade kube-scheduler performance, particularly under high kube-apiserver load. ([#134400](https://github.com/kubernetes/kubernetes/pull/134400), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- When image garbage collection is unable to free enough disk space, the FreeDiskSpaceFailed warning event is now more actionable. Example: `Insufficient free disk space on the node's image filesystem (95.0% of 10.0 GiB used). Failed to free sufficient space by deleting unused images. Consider resizing the disk or deleting unused files.` ([#132578](https://github.com/kubernetes/kubernetes/pull/132578), [@drigz](https://github.com/drigz)) [SIG Node]

### Other (Cleanup or Flake)

- Bump addon manager to use kubectl v1.32.2 ([#130548](https://github.com/kubernetes/kubernetes/pull/130548), [@Jefftree](https://github.com/Jefftree)) [SIG Cloud Provider, Scalability and Testing]
- Dropping the experimental prefix from kubectl wait command's short description, since kubectl wait command has been stable for a long time. ([#133907](https://github.com/kubernetes/kubernetes/pull/133907), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Fix formatting of assorted go API deprecations for godoc / pkgsite and enable a linter to help catch mis-formatted deprecations ([#133571](https://github.com/kubernetes/kubernetes/pull/133571), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery, Architecture, CLI, Instrumentation and Testing]
- Improved HPA performance when using container-specific resource metrics by optimizing container lookup logic to exit early once the target container is found, reducing unnecessary iterations through all containers in a pod. ([#133415](https://github.com/kubernetes/kubernetes/pull/133415), [@AadiDev005](https://github.com/AadiDev005)) [SIG Apps and Autoscaling]
- Kube-apiserver: Fixes an issue where passing invalid DeleteOptions incorrectly returned status 500 rather than 400. ([#133358](https://github.com/kubernetes/kubernetes/pull/133358), [@ostrain](https://github.com/ostrain)) [SIG API Machinery]
- Kubeadm: removed the `RootlessControlPlane` feature gate. User Namespaces will serve as its replacement. ([#134178](https://github.com/kubernetes/kubernetes/pull/134178), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Remove container name from messages for container created and started events. ([#134043](https://github.com/kubernetes/kubernetes/pull/134043), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Removed deprecated gogo protocol definitions from `k8s.io/kubelet/pkg/apis/dra` in favor of `google.golang.org/protobuf`. ([#133026](https://github.com/kubernetes/kubernetes/pull/133026), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery and Node]
- Removed general available feature-gate SizeMemoryBackedVolumes ([#133720](https://github.com/kubernetes/kubernetes/pull/133720), [@carlory](https://github.com/carlory)) [SIG Node, Storage and Testing]
- Removed the `ComponentSLIs` feature gate, which had been promoted to stable as part of the Kubernetes 1.32 release. ([#133742](https://github.com/kubernetes/kubernetes/pull/133742), [@carlory](https://github.com/carlory)) [SIG Architecture and Instrumentation]
- Removing Experimental prefix from the description of kubectl wait to emphasize that it is stable. ([#133731](https://github.com/kubernetes/kubernetes/pull/133731), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Removing the KUBECTL_OPENAPIV3_PATCH environment variable entirely, since aggregated discovery has been stable from 1.30. ([#134130](https://github.com/kubernetes/kubernetes/pull/134130), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Specifies the deprecated version of apiserver_storage_objects metric in metrics docs ([#134028](https://github.com/kubernetes/kubernetes/pull/134028), [@richabanker](https://github.com/richabanker)) [SIG API Machinery, Etcd and Instrumentation]
- Tests: switch to https://go.dev/doc/go1.25#container-aware-gomaxprocs from go.uber.org/automaxprocs ([#133492](https://github.com/kubernetes/kubernetes/pull/133492), [@BenTheElder](https://github.com/BenTheElder)) [SIG Testing]
- The `/statusz` page for `kube-proxy` now includes a list of exposed endpoints, making it easier to debug and introspect. ([#133190](https://github.com/kubernetes/kubernetes/pull/133190), [@aman4433](https://github.com/aman4433)) [SIG Network and Node]
- Types in k/k/pkg/scheduler/framework:
  Handle,
  Plugin,
  PreEnqueuePlugin, QueueSortPlugin, EnqueueExtensions, PreFilterExtensions, PreFilterPlugin, FilterPlugin, PostFilterPlugin, PreScorePlugin, ScorePlugin, ReservePlugin, PreBindPlugin, PostBindPlugin, PermitPlugin, BindPlugin,
  PodActivator, PodNominator, PluginsRunner,
  LessFunc, ScoreExtensions, NodeToStatusReader, NodeScoreList, NodeScore, NodePluginScores, PluginScore, NominatingMode, NominatingInfo, WaitingPod, PreFilterResult, PostFilterResult,
  Extender,
  NodeInfoLister, StorageInfoLister, SharedLister, ResourceSliceLister, DeviceClassLister, ResourceClaimTracker, SharedDRAManager
  
  are moved to package k8s.io/kube-scheduler/framework . Users should update import paths. The interfaces don't change.
  
  Type Parallelizer in k/k/pkg/scheduler/framework/parallelism is split into interface Parallelizer (in k8s.io/kube-scheduler/framework) and struct Parallelizer (location unchanged in k/k). Plugin developers should update the import path to staging repo. ([#133172](https://github.com/kubernetes/kubernetes/pull/133172), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Node, Release, Scheduling, Storage and Testing]
- Updated CNI plugins to v1.8.0. ([#133837](https://github.com/kubernetes/kubernetes/pull/133837), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.34.0. ([#133636](https://github.com/kubernetes/kubernetes/pull/133636), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Updated etcd to v3.6.5. ([#134251](https://github.com/kubernetes/kubernetes/pull/134251), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Upgrade CoreDNS to v1.12.3 ([#132288](https://github.com/kubernetes/kubernetes/pull/132288), [@thevilledev](https://github.com/thevilledev)) [SIG Cloud Provider and Cluster Lifecycle]
- `kubectl auth reconcile` now re-attempts reconciliation if it encounters a conflict error ([#133323](https://github.com/kubernetes/kubernetes/pull/133323), [@liggitt](https://github.com/liggitt)) [SIG Auth and CLI]
- `kubectl get` and `kubectl describe` human-readable output no longer includes counts for referenced tokens and secrets ([#117160](https://github.com/kubernetes/kubernetes/pull/117160), [@liggitt](https://github.com/liggitt)) [SIG CLI and Testing]

## Dependencies

### Added
- github.com/moby/sys/atomicwriter: [v0.1.0](https://github.com/moby/sys/tree/atomicwriter/v0.1.0)
- golang.org/x/tools/go/expect: v0.1.1-deprecated
- golang.org/x/tools/go/packages/packagestest: v0.1.1-deprecated

### Changed
- cloud.google.com/go/compute/metadata: v0.6.0 → v0.7.0
- github.com/aws/aws-sdk-go-v2/config: [v1.27.24 → v1.29.14](https://github.com/aws/aws-sdk-go-v2/compare/config/v1.27.24...config/v1.29.14)
- github.com/aws/aws-sdk-go-v2/credentials: [v1.17.24 → v1.17.67](https://github.com/aws/aws-sdk-go-v2/compare/credentials/v1.17.24...credentials/v1.17.67)
- github.com/aws/aws-sdk-go-v2/feature/ec2/imds: [v1.16.9 → v1.16.30](https://github.com/aws/aws-sdk-go-v2/compare/feature/ec2/imds/v1.16.9...feature/ec2/imds/v1.16.30)
- github.com/aws/aws-sdk-go-v2/internal/configsources: [v1.3.13 → v1.3.34](https://github.com/aws/aws-sdk-go-v2/compare/internal/configsources/v1.3.13...internal/configsources/v1.3.34)
- github.com/aws/aws-sdk-go-v2/internal/endpoints/v2: [v2.6.13 → v2.6.34](https://github.com/aws/aws-sdk-go-v2/compare/internal/endpoints/v2/v2.6.13...internal/endpoints/v2/v2.6.34)
- github.com/aws/aws-sdk-go-v2/internal/ini: [v1.8.0 → v1.8.3](https://github.com/aws/aws-sdk-go-v2/compare/internal/ini/v1.8.0...internal/ini/v1.8.3)
- github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding: [v1.11.3 → v1.12.3](https://github.com/aws/aws-sdk-go-v2/compare/service/internal/accept-encoding/v1.11.3...service/internal/accept-encoding/v1.12.3)
- github.com/aws/aws-sdk-go-v2/service/internal/presigned-url: [v1.11.15 → v1.12.15](https://github.com/aws/aws-sdk-go-v2/compare/service/internal/presigned-url/v1.11.15...service/internal/presigned-url/v1.12.15)
- github.com/aws/aws-sdk-go-v2/service/sso: [v1.22.1 → v1.25.3](https://github.com/aws/aws-sdk-go-v2/compare/service/sso/v1.22.1...service/sso/v1.25.3)
- github.com/aws/aws-sdk-go-v2/service/ssooidc: [v1.26.2 → v1.30.1](https://github.com/aws/aws-sdk-go-v2/compare/service/ssooidc/v1.26.2...service/ssooidc/v1.30.1)
- github.com/aws/aws-sdk-go-v2/service/sts: [v1.30.1 → v1.33.19](https://github.com/aws/aws-sdk-go-v2/compare/service/sts/v1.30.1...service/sts/v1.33.19)
- github.com/aws/aws-sdk-go-v2: [v1.30.1 → v1.36.3](https://github.com/aws/aws-sdk-go-v2/compare/v1.30.1...v1.36.3)
- github.com/aws/smithy-go: [v1.20.3 → v1.22.3](https://github.com/aws/smithy-go/compare/v1.20.3...v1.22.3)
- github.com/containerd/containerd/api: [v1.8.0 → v1.9.0](https://github.com/containerd/containerd/compare/api/v1.8.0...api/v1.9.0)
- github.com/containerd/ttrpc: [v1.2.6 → v1.2.7](https://github.com/containerd/ttrpc/compare/v1.2.6...v1.2.7)
- github.com/containerd/typeurl/v2: [v2.2.2 → v2.2.3](https://github.com/containerd/typeurl/compare/v2.2.2...v2.2.3)
- github.com/coredns/corefile-migration: [v1.0.26 → v1.0.27](https://github.com/coredns/corefile-migration/compare/v1.0.26...v1.0.27)
- github.com/docker/docker: [v26.1.4+incompatible → v28.2.2+incompatible](https://github.com/docker/docker/compare/v26.1.4...v28.2.2)
- github.com/go-logr/logr: [v1.4.2 → v1.4.3](https://github.com/go-logr/logr/compare/v1.4.2...v1.4.3)
- github.com/google/cadvisor: [v0.52.1 → v0.53.0](https://github.com/google/cadvisor/compare/v0.52.1...v0.53.0)
- github.com/opencontainers/cgroups: [v0.0.1 → v0.0.3](https://github.com/opencontainers/cgroups/compare/v0.0.1...v0.0.3)
- github.com/opencontainers/runc: [v1.2.5 → v1.3.0](https://github.com/opencontainers/runc/compare/v1.2.5...v1.3.0)
- github.com/opencontainers/runtime-spec: [v1.2.0 → v1.2.1](https://github.com/opencontainers/runtime-spec/compare/v1.2.0...v1.2.1)
- github.com/prometheus/client_golang: [v1.22.0 → v1.23.2](https://github.com/prometheus/client_golang/compare/v1.22.0...v1.23.2)
- github.com/prometheus/client_model: [v0.6.1 → v0.6.2](https://github.com/prometheus/client_model/compare/v0.6.1...v0.6.2)
- github.com/prometheus/common: [v0.62.0 → v0.66.1](https://github.com/prometheus/common/compare/v0.62.0...v0.66.1)
- github.com/prometheus/procfs: [v0.15.1 → v0.16.1](https://github.com/prometheus/procfs/compare/v0.15.1...v0.16.1)
- github.com/spf13/cobra: [v1.9.1 → v1.10.0](https://github.com/spf13/cobra/compare/v1.9.1...v1.10.0)
- github.com/spf13/pflag: [v1.0.6 → v1.0.9](https://github.com/spf13/pflag/compare/v1.0.6...v1.0.9)
- github.com/stretchr/testify: [v1.10.0 → v1.11.1](https://github.com/stretchr/testify/compare/v1.10.0...v1.11.1)
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.58.0 → v0.61.0
- go.opentelemetry.io/otel/metric: v1.35.0 → v1.36.0
- go.opentelemetry.io/otel/sdk/metric: v1.34.0 → v1.36.0
- go.opentelemetry.io/otel/sdk: v1.34.0 → v1.36.0
- go.opentelemetry.io/otel/trace: v1.35.0 → v1.36.0
- go.opentelemetry.io/otel: v1.35.0 → v1.36.0
- golang.org/x/crypto: v0.36.0 → v0.41.0
- golang.org/x/mod: v0.21.0 → v0.27.0
- golang.org/x/net: v0.38.0 → v0.43.0
- golang.org/x/oauth2: v0.27.0 → v0.30.0
- golang.org/x/sync: v0.12.0 → v0.16.0
- golang.org/x/sys: v0.31.0 → v0.35.0
- golang.org/x/telemetry: bda5523 → 1a19826
- golang.org/x/term: v0.30.0 → v0.34.0
- golang.org/x/text: v0.23.0 → v0.28.0
- golang.org/x/tools: v0.26.0 → v0.36.0
- google.golang.org/genproto/googleapis/rpc: a0af3ef → 200df99
- google.golang.org/grpc: v1.72.1 → v1.72.2
- google.golang.org/protobuf: v1.36.5 → v1.36.8
- gopkg.in/evanphx/json-patch.v4: v4.12.0 → v4.13.0
- k8s.io/gengo/v2: 85fd79d → ec3ebc5
- k8s.io/kube-openapi: f3f2b99 → 589584f
- k8s.io/system-validators: v1.10.1 → v1.11.1
- sigs.k8s.io/json: cfa47c3 → 2d32026

### Removed
- gopkg.in/yaml.v2: v2.4.0