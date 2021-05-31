<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.22.0-alpha.2](#v1220-alpha2)
  - [Downloads for v1.22.0-alpha.2](#downloads-for-v1220-alpha2)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.22.0-alpha.1](#changelog-since-v1220-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
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
- [v1.22.0-alpha.1](#v1220-alpha1)
  - [Downloads for v1.22.0-alpha.1](#downloads-for-v1220-alpha1)
    - [Source Code](#source-code-1)
    - [Client binaries](#client-binaries-1)
    - [Server binaries](#server-binaries-1)
    - [Node binaries](#node-binaries-1)
  - [Changelog since v1.21.0](#changelog-since-v1210)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation-1)
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

# v1.22.0-alpha.2


## Downloads for v1.22.0-alpha.2

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes.tar.gz) | 39d5177271e744058585c4b924ff91e4df654db81257a4710b77a055ac6033c8d6414772a4c42e3ec7f568ac5c9691c53225a13a68610aa0b07c3bcaf252fe4c
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-src.tar.gz) | d9832ab5ba568f89ffb7e9bfef3dd0baee69c5a29bc34e2f8f83fef08f13575e4982409dba422b912245655b326565f9e71e523bcbd391b97fd385ae7e4debaa

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | d007c403a9586a0047db4abc8766845aa501798524a259902a3a3e5d43928a819b9857ef4b49632384139e12d3b0e0c0cbf2966a5067e9e29496d4bf14a2ea24
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 75ce76788e5bebcd6c06a8cc804c39edccaf42941dfd35cea331eb86393918fb6addef2bf507b78d9dac6eb3627568c281404a5fd899fb396052ff9658dc3f70
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 9d590915534c1fe3d69c1e0df7b16c6668e52be32a3649214a0a4940a8ed1565efe2c300a1c7aa02c9605be8e829fd9a75229d2b0a9a0765f3ce16b6ad68f4b1
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 5e44d189b32a61b3f060a5ec13207cea526c7fedbc42967915e6b50f106ac862c13560bb15066bed3134407621ae506c18297c7b3ea2f561fb20a97ac02215cc
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | d8d446133b14f11da9f33a20c6700d23d2616b4d6cf750e8074526b8442b4e0e437b20444fc583f4097c8b064966a4a1e52fb2e01096e2c94ec4e05ef2d4b48e
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 20db0fee191a027885b9a12615732b40e88c148f04343f56e67dfa5a12e08a51238c6e93aed05685afd6b203dc3f1961c6db4096ba867caf299d5d0a190a91d6
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | b642a040ac656c609be3191af20f5b3142d20b1d39846e3052402e99bb5fca9211e4225cb775a9ec19b9cf7e47754ece813a7d367d9c911be18a1ea5584cf178
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 161756afb0b040dd5134d91f4982dd0233f3e4fec31375dd6b2f515c12f6fc0c7237a0c8283bf2a83e147df69403b35c3d9bbe7b872779dd5b2e43ef5c8693f8
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 3956a25f75a29f23a559a335a0629299a083143db1ccad6db2ff76c27ead72ad25a5db81b558225a530749ffc58749342079f80c5af4f0134553b6de05f60a5b
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 29ba410e9d600b92ec02f284a545045d4e3b1e6c247fc5db64c2a8536108456389986efbdb762faba6509b1b50e9bbc3638d2dca19577de79b0de34ad749e410

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | 0fad8691f0c72e0b4b09e5da9b806353f1a5c48c3b38c90674d44e673daa77ed85e727434ba9cbe2717ca65005059af17fb7b7db4d452aa67fef7cf2395da738
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | e6d8e4dbcc5e5790114834b8aa5a9bfc5b1c18c4b16cb043f3fd409c22c8b2ccfdb165357e584f650321c5c07ca5aae405f70da65efa32f5dffbeb25ebc22c42
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 4f0c46ad6a504ea0b7607175603e61530496d29759f27e6e9dac8b7bb923f8920ed6dd2afb5d709f2f96850145252d4dd702bb77254791639cfb33648f3b1f04
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 58cb325bd7470972df7d286a7c160f732f261ce4858882f99cc5ab91ba43f86d1cf1294651f61ba1416c17ff91abfe178dfbb1c264716029d58b94f595dc734d
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 993806a9f7404365ab6e8a4017a5c5dece028c9f8c376498c196dda9bb885ecbceaad5498f43bea8d1309707216ae4173dc8aa69151ad304e5f1993be1f7f6dc

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 77c1fd98ac2e8a665bffae60bdc66f1b5fc29482d29f58b4d5705b43478fc536885e6634ffd2e8a18ff0ea589a15a2df67ba86ede2025a697019030bd7893bbc
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 839ec50c2438279fae2b52efa985556a7c4ba090c8296d56ae8623b3b7123cb6c4b0a656083cc43463e57fdb3d8bae2609196879061aa806aac3a65562c02e40
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 3568f19f2cde2da63e5897a8f206a475f86c41f273dac4eb1e31416945d112c6d00ce74e4159732d3805cbe093c94ec53c573227f41ab873c6698023b473b2f5
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | f9de072d40b9a354785ae1dbc182ddcb431e0c4e00fe8f4c56e2b5ff2062845e0c740e7efd4b9697bca9848b0808ad01f20817f84d5c5d5c9c78e52be7962243
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | fc4b31c68367e938991a5ca4d9df9c38950939ada6f6c0dd6a827d43d5f003b20fdea25d34a213853d53d0a933e4715425f37668de7e110ee0722cb866fa94bd
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 8dbee9ebf915c645ce199d7f190323ffd71b810f2cec2e1dee8d35948994aa7d08ebc9a82ab083f1eb83476ae104d4a63b4bc258ecbcc9ab3f158d56f179d7a8

## Changelog since v1.22.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Intree volume plugin scaleio support is been completely removed from Kubernetes. ([#101685](https://github.com/kubernetes/kubernetes/pull/101685), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG API Machinery, Node and Storage]
  - Newly provisioned PVs by Azure disk will no longer have the beta FailureDomain label. Azure disk volume plugin will start to have GA topology label instead. ([#101534](https://github.com/kubernetes/kubernetes/pull/101534), [@kassarl](https://github.com/kassarl)) [SIG Cloud Provider and Storage]
  - Scheduler's CycleState now embeds internal read/write locking inside its Read() and Write() functions. Meanwhile, Lock() and Unlock() function are removed.
  
  scheduler plugin developers are now required to remove CycleState#Lock() and CycleState#Unlock(). Just simply use Read() and Write() as they're natively thread-safe now. ([#101542](https://github.com/kubernetes/kubernetes/pull/101542), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Storage]
 
## Changes by Kind

### Deprecation

- Controller-manager: the following flags have no effect and would be removed in v1.24:
  - `--port`
  - `--address`
  The insecure port flags `--port` may only be set to 0 now.
  
  In addtion, please be careful that:
  - controller-manager MUST start with `--authorization-kubeconfig` and `--authentication-kubeconfig` correctly set to get authentication/authorization working.
  - liveness/readiness probes to controller-manager MUST use HTTPS now, and the default port has been changed to 10257.
  - Applications that fetch metrics from controller-manager should use a dedicated service account which is allowed to access nonResourceURLs `/metrics`. ([#96216](https://github.com/kubernetes/kubernetes/pull/96216), [@knight42](https://github.com/knight42)) [SIG API Machinery, Cloud Provider, Instrumentation and Testing]
- Ingress v1beta1 has been deprecated ([#102030](https://github.com/kubernetes/kubernetes/pull/102030), [@aojea](https://github.com/aojea)) [SIG CLI, Network and Testing]
- Kubead: remove the deprecated "--csr-only" and "--csr-dir" flags from "kubeadm init phase certs". Deprecate the same flags under "kubeadm certs renew". In both cases the command "kubeadm certs generate-csr" should be used instead. ([#102108](https://github.com/kubernetes/kubernetes/pull/102108), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the ClusterStatus API from v1beta3 and its management in the kube-system/kubeadm-config ConfigMap. This method of keeping track of what API endpoints exists in the cluster was replaced (in a prior release) by a method to annotate the etcd Pods that kubeadm creates in "stacked etcd" clusters. The following CLI sub-phases are deprecated and are now a NO-OP: for " kubeadm join": "control-plane-join/update-status", for "kubeadm reset": "update-cluster-status". Unless you are using these phases explicitly, you should not be affected. ([#101915](https://github.com/kubernetes/kubernetes/pull/101915), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated command "kubeadm alpha kubeconfig". Please use "kubeadm kubeconfig" instead. ([#101938](https://github.com/kubernetes/kubernetes/pull/101938), [@knight42](https://github.com/knight42)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated command 'kubeadm config view'. A replacement for this command is 'kubectl get cm -n kube-system kubeadm-config -o=jsonpath="{.data.ClusterConfiguration}"' ([#102071](https://github.com/kubernetes/kubernetes/pull/102071), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated flag '--image-pull-timeout' for 'kubeadm upgrade apply' command ([#102093](https://github.com/kubernetes/kubernetes/pull/102093), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated flag --insecure-port from the kube-apiserver manifest that kubeadm manages. The flag had no effect since 1.20, since the insecure serving of the component was disabled in the same version. ([#102121](https://github.com/kubernetes/kubernetes/pull/102121), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated hyperkube image support in v1beta3. This implies removal of ClusterConfiguration.UseHyperKubeImage. ([#101537](https://github.com/kubernetes/kubernetes/pull/101537), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the field ClusterConfiguration.DNS.Type in v1beta3 since CoreDNS is the only supported DNS type. ([#101547](https://github.com/kubernetes/kubernetes/pull/101547), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- `storageos`,`quobyte` and `flocker` storage volume plugins are deprecated and will be removed in a later release. ([#101773](https://github.com/kubernetes/kubernetes/pull/101773), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG Storage]

### API Change

- Add alpha support for HostProcess containers on Windows ([#99576](https://github.com/kubernetes/kubernetes/pull/99576), [@marosset](https://github.com/marosset)) [SIG API Machinery, Apps, Node, Testing and Windows]
- Add three metrics to job controller to monitor if Job works in a healthy condition.
  IndexedJob promoted to Beta ([#101292](https://github.com/kubernetes/kubernetes/pull/101292), [@AliceZhang2016](https://github.com/AliceZhang2016)) [SIG Apps, Instrumentation and Testing]
- Corrected the documentation for escaping dollar signs in a container's env, command and args property. ([#101916](https://github.com/kubernetes/kubernetes/pull/101916), [@MartinKanters](https://github.com/MartinKanters)) [SIG Apps]
- Omit comparison with boolean constant ([#101523](https://github.com/kubernetes/kubernetes/pull/101523), [@GreenApple10](https://github.com/GreenApple10)) [SIG CLI and Cloud Provider]
- Pod Affinity NamespaceSelector and the associated CrossNamespaceAffinity quota scope graduated to beta ([#101496](https://github.com/kubernetes/kubernetes/pull/101496), [@ahg-g](https://github.com/ahg-g)) [SIG API Machinery, Apps and Testing]
- V1.Node .status.images[].names is now optional ([#102159](https://github.com/kubernetes/kubernetes/pull/102159), [@roycaihw](https://github.com/roycaihw)) [SIG Apps and Node]

### Feature

- Added BinaryData description to kubectl describe ([#100568](https://github.com/kubernetes/kubernetes/pull/100568), [@lauchokyip](https://github.com/lauchokyip)) [SIG CLI]
- Feat: change parittion style to GPT on Windows ([#101412](https://github.com/kubernetes/kubernetes/pull/101412), [@andyzhangx](https://github.com/andyzhangx)) [SIG Storage and Windows]
- Improve logging of APIService availability changes in kube-apiserver. ([#101420](https://github.com/kubernetes/kubernetes/pull/101420), [@sttts](https://github.com/sttts)) [SIG API Machinery]
- Kubeadm: add the RootlessControlPlane kubeadm specific feature gate (Alpha in 1.22, disabled by default).
  It can be used to enable an experimental feature that makes the control plane component static Pod containers 
  for kube-apiserver, kube-controller-manager, kube-scheduler and etcd to run as a non-root users. ([#102158](https://github.com/kubernetes/kubernetes/pull/102158), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Cluster Lifecycle]
- Kubeadm: set the seccompProfile to runtime/default in the PodSecurityContext of the  control-plane components that run as static Pods. ([#100234](https://github.com/kubernetes/kubernetes/pull/100234), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Golang 1.16.4 ([#101809](https://github.com/kubernetes/kubernetes/pull/101809), [@justaugustus](https://github.com/justaugustus)) [SIG Cloud Provider, Instrumentation, Release and Testing]
- Metrics server nanny has now poll period set to 30s (previously 5 minutes) to allow faster scaling of metrics server. ([#101869](https://github.com/kubernetes/kubernetes/pull/101869), [@olagacek](https://github.com/olagacek)) [SIG Cloud Provider and Instrumentation]
- New metrics: `apiserver_kube_aggregator_x509_missing_san_total` and `apiserver_webhooks_x509_missing_san_total`. This metric measures a number of connections to webhooks/aggregated API servers that use certificates without Subject Alternative Names. It being non-zero is a warning sign that these connections will stop functioning in the future since Golang is going to deprecate x509 certificate subject Common Names for server hostname verification. ([#95396](https://github.com/kubernetes/kubernetes/pull/95396), [@stlaz](https://github.com/stlaz)) [SIG API Machinery, Auth and Instrumentation]
- Node Problem Detector is now available for GCE Windows nodes. ([#101539](https://github.com/kubernetes/kubernetes/pull/101539), [@jeremyje](https://github.com/jeremyje)) [SIG Cloud Provider, Node and Windows]
- Secret values are now masked by default in kubectl diff output. ([#96084](https://github.com/kubernetes/kubernetes/pull/96084), [@loozhengyuan](https://github.com/loozhengyuan)) [SIG CLI]
- The `WarningHeader` feature is now GA and is unconditionally enabled. The `apiserver_requested_deprecated_apis` metric has graduated to stable status. The `WarningHeader` feature-gate is no longer operative and will be removed in v1.24. ([#100754](https://github.com/kubernetes/kubernetes/pull/100754), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Instrumentation and Testing]
- Warnings for use of deprecated and known-bad values in pod specs are now sent ([#101688](https://github.com/kubernetes/kubernetes/pull/101688), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Auth]
- You can use this Builder function to create events Field Selector ([#101817](https://github.com/kubernetes/kubernetes/pull/101817), [@cndoit18](https://github.com/cndoit18)) [SIG API Machinery and Scalability]

### Failing Test

- Fixes the `should receive events on concurrent watches in same order` conformance test to work properly on clusters that auto-create additional configmaps in namespaces ([#101950](https://github.com/kubernetes/kubernetes/pull/101950), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Resolves an issue with the "ServiceAccountIssuerDiscovery should support OIDC discovery" conformance test failing on clusters which are configured with issuers outside the cluster ([#101589](https://github.com/kubernetes/kubernetes/pull/101589), [@mtaufen](https://github.com/mtaufen)) [SIG Auth and Testing]

### Bug or Regression

- Added jitter factor to lease controller that better smears load on kube-apiserver over time. ([#101652](https://github.com/kubernetes/kubernetes/pull/101652), [@marseel](https://github.com/marseel)) [SIG API Machinery and Scalability]
- Avoid caching the Azure VMSS instances whose network profile is nil ([#100948](https://github.com/kubernetes/kubernetes/pull/100948), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Azure: avoid setting cached Sku when updating VMSS and VMSS instances ([#102005](https://github.com/kubernetes/kubernetes/pull/102005), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix Azure node public IP fetching issues from instance metadata service when the node is part of standard load balancer backend pool. ([#100690](https://github.com/kubernetes/kubernetes/pull/100690), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix EndpointSlice describe panic when an Endpoint doesn't have zone ([#101025](https://github.com/kubernetes/kubernetes/pull/101025), [@tnqn](https://github.com/tnqn)) [SIG CLI]
- Fix kubectl set env or resources not working for initcontainers ([#101669](https://github.com/kubernetes/kubernetes/pull/101669), [@carlory](https://github.com/carlory)) [SIG CLI]
- Fix resource enforcement when using systemd cgroup driver ([#102147](https://github.com/kubernetes/kubernetes/pull/102147), [@kolyshkin](https://github.com/kolyshkin)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Storage and Testing]
- Fix: avoid nil-pointer panic when checking the frontend IP configuration ([#101739](https://github.com/kubernetes/kubernetes/pull/101739), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fix: delete non existing disk issue ([#102083](https://github.com/kubernetes/kubernetes/pull/102083), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: not tagging static public IP ([#101752](https://github.com/kubernetes/kubernetes/pull/101752), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fixed a bug that `kubectl create configmap` always returns zero exit code when failed. ([#101780](https://github.com/kubernetes/kubernetes/pull/101780), [@nak3](https://github.com/nak3)) [SIG CLI]
- Fixed false-positive uncertain volume attachments, which led to unexpected detachment of CSI migrated volumes ([#101737](https://github.com/kubernetes/kubernetes/pull/101737), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG Apps and Storage]
- Fixed mounting of NFS volumes when IPv6 address is used as a server. ([#101067](https://github.com/kubernetes/kubernetes/pull/101067), [@Elbehery](https://github.com/Elbehery)) [SIG Storage]
- GCE Windows will no longer install Docker on containerd nodes. ([#101747](https://github.com/kubernetes/kubernetes/pull/101747), [@jeremyje](https://github.com/jeremyje)) [SIG Cloud Provider and Windows]
- Kube-proxy log now shows the "Skipping topology aware endpoint filtering since no hints were provided for zone" warning under the right conditions ([#101857](https://github.com/kubernetes/kubernetes/pull/101857), [@dervoeti](https://github.com/dervoeti)) [SIG Network]
- Kubeadm upgrade etcd to 3.4.13-3 ([#100612](https://github.com/kubernetes/kubernetes/pull/100612), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery, Cloud Provider and Cluster Lifecycle]
- Kubeadm: fix the bug that kubeadm only uses the first hash in caCertHashes to verify the root CA ([#101977](https://github.com/kubernetes/kubernetes/pull/101977), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl create service now respects namespace flag ([#101005](https://github.com/kubernetes/kubernetes/pull/101005), [@zxh326](https://github.com/zxh326)) [SIG CLI]
- Kubectl wait --for=delete ignores not found error correctly now. ([#96702](https://github.com/kubernetes/kubernetes/pull/96702), [@lingsamuel](https://github.com/lingsamuel)) [SIG CLI and Testing]
- Parsing of cpuset information now properly detects more invalid input such as "1--3" or "10-6" ([#100565](https://github.com/kubernetes/kubernetes/pull/100565), [@lack](https://github.com/lack)) [SIG Node]
- Register/Deregister Targets in chunks for AWS TargetGroup ([#101592](https://github.com/kubernetes/kubernetes/pull/101592), [@M00nF1sh](https://github.com/M00nF1sh)) [SIG Cloud Provider]
- Respect annotation size limit for server-side apply updates to the client-side apply annotation. Also, fix opt-out of this behavior by setting the client-side apply annotation to the empty string. ([#102105](https://github.com/kubernetes/kubernetes/pull/102105), [@julianvmodesto](https://github.com/julianvmodesto)) [SIG API Machinery]
- The conformance tests:
  - Services should serve multiport endpoints from pods
  - Services should serve a basic endpoint from pods
  were only validating the API objects, not performing any validation on the actual Services implementation.
  Those tests now validate that the Services under test are able to forward traffic to the endpoints. ([#101709](https://github.com/kubernetes/kubernetes/pull/101709), [@aojea](https://github.com/aojea)) [SIG Network and Testing]
- When `DisableAcceleratorUsageMetrics` is set, do not collect accelerator metrics using cAdvisor. ([#101712](https://github.com/kubernetes/kubernetes/pull/101712), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Instrumentation and Node]

### Other (Cleanup or Flake)

- Fake clients now implement a `FakeClient` interface ([#100940](https://github.com/kubernetes/kubernetes/pull/100940), [@markusthoemmes](https://github.com/markusthoemmes)) [SIG API Machinery and Instrumentation]
- Kubeadm: the `CriticalAddonsOnly` toleration has been removed from `kube-proxy` DaemonSet ([#101966](https://github.com/kubernetes/kubernetes/pull/101966), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Metrics Server updated to use 0.4.4 image that doesn't depend on deprecated authorization.k8s.io/v1beta1 subjectaccessreviews API version. ([#101477](https://github.com/kubernetes/kubernetes/pull/101477), [@x13n](https://github.com/x13n)) [SIG Cloud Provider and Instrumentation]
- Migrate proxy/ipvs/proxier.go logs to structured logging ([#97796](https://github.com/kubernetes/kubernetes/pull/97796), [@JornShen](https://github.com/JornShen)) [SIG Network]
- Remove duplicate packet import ([#101187](https://github.com/kubernetes/kubernetes/pull/101187), [@GreenApple10](https://github.com/GreenApple10)) [SIG API Machinery]
- The `VolumeSnapshotDataSource` feature gate that is GA since v1.20 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#101531](https://github.com/kubernetes/kubernetes/pull/101531), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Storage]
- The deprecated CRIContainerLogRotation feature-gate has been removed, since the CRIContainerLogRotation feature graduated to GA in 1.21 and was unconditionally enabled. ([#101578](https://github.com/kubernetes/kubernetes/pull/101578), [@carlory](https://github.com/carlory)) [SIG Node]
- The deprecated RootCAConfigMap feature-gate has been removed, since the RootCAConfigMap feature graduated to GA in 1.21 and was unconditionally enabled. ([#101579](https://github.com/kubernetes/kubernetes/pull/101579), [@carlory](https://github.com/carlory)) [SIG Auth]

## Dependencies

### Added
- github.com/nxadm/tail: [v1.4.4](https://github.com/nxadm/tail/tree/v1.4.4)
- rsc.io/quote/v3: v3.1.0
- rsc.io/sampler: v1.3.0

### Changed
- github.com/containernetworking/cni: [v0.8.0 → v0.8.1](https://github.com/containernetworking/cni/compare/v0.8.0...v0.8.1)
- github.com/golang/mock: [v1.4.4 → v1.4.3](https://github.com/golang/mock/compare/v1.4.4...v1.4.3)
- github.com/onsi/ginkgo: [v1.11.0 → v1.14.0](https://github.com/onsi/ginkgo/compare/v1.11.0...v1.14.0)
- github.com/onsi/gomega: [v1.7.0 → v1.10.1](https://github.com/onsi/gomega/compare/v1.7.0...v1.10.1)
- github.com/stretchr/testify: [v1.6.1 → v1.7.0](https://github.com/stretchr/testify/compare/v1.6.1...v1.7.0)

### Removed
- github.com/hpcloud/tail: [v1.0.0](https://github.com/hpcloud/tail/tree/v1.0.0)
- github.com/thecodeteam/goscaleio: [v0.1.0](https://github.com/thecodeteam/goscaleio/tree/v0.1.0)
- gopkg.in/fsnotify.v1: v1.4.7



# v1.22.0-alpha.1


## Downloads for v1.22.0-alpha.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes.tar.gz) | de3fb80c8fdcabe60f37e3dcb1c61e8733c95fc0d45840f6861eafde09a149c3880f3e0b434d33167ffa66bdfeb887696ac7bfd2b44b85c29f99ba12965305ed
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-src.tar.gz) | 753b9022b3c487d4bc9f8b302de14b7b4ef52b7664ff6d6b8bca65b6896cbc5932038de551a02c412afdd3ac2d56a8141e0dcb1dac7d24102217bd4f2beff936

### Client binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 8ba8627419704285abad0d98d28555d4bf4ce624c6958d0cca5ca8f53f1c40bb514631980ef39d52e2a604aff93bc078b30256d307d8af9839df91f8493d9aa5
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | a039181d9dbff3203e75f357c65eaaf1667ab0834167b9ac12ff76999e276b9cc077e843b6043388183bd7c350c42ea28ab2d7b074c4f1987e43298e918595e1
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-linux-386.tar.gz) | 3474282cfe7f8f2966fca742453c632294ba224126748b162d42bd68a715681f2845c740252400d0b7d21dd3a11440530a5b84e454225655c16e056ca413e9de
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 7bd1e8b21af6b72757cdef9a4d76ea0eda3dbd558f2f5a7bee8f24f2c9b05d1cf52cfebd2f5ea991811917c3c18f1ac3dbde7e5094d5cd8a73478077a797b801
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 0505f0c8e3733584ad1fc22ad729aea9f2452c8452ab1ed5e735e53ff48a92c248ba7310e5e9fa76630fa06a600c4ce8ee1b2b2845f07dba795fddbff5b7e941
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | f5cbb08845bc6519538325250a7826e65ede254e5cf700a3f9b9128fec205f8d90827639bc64146b7c44008acd6a708bba59a3fbcefec1ca8e0050f6e3330290
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | a3d90dc2ca5970ef4029ad9e9ff678816048c4dc58e7ad0f17a9a873855d71fdb3d23f4f7c88465f2261ed72747e85b78c80006e221e456bab0f07dc91022f1c
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | cfee985e127f9471da4cb538362e3150c4edf12e8c72c5415024244007c9bf46c8f4a7f19e9fa8afb3126e379efce837114f8d1cee0f78d1602fe5e807e24b06
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 47811776c0d1569afb3c8a689bb8989b57e8d3da4291606da6fc8b481e79b8632ac333f5c011e2bfd4fe4677827b27f64bd15253c2d83fdb5c0ce40671322e82
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | d009d8178f94bcd69a1ae5a6ff39438b9811204f4c4f3b11b6219bcbd7d80f86ed2d6486feb88128fa42383550e37af6b3a603f0cecae1fdb86b69725d0b331a

### Server binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | 9bec26661b3ca7a688da8cc6fbb6ba4bf5e9993599401dbc9f7d95a2805d8a5c319052c30f33236094ba0a3b984a2246173d5334457ce7453ce74c84f5012c01
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | 89737d178779c9c636c246995aca9447a8e22150c63ae57cc3f1360b905c654d0f1c47dd35f958262e26a5fe61212fad308778d2acc9dbd8baff563f4c9a3e48
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 9ddb37baa8d2589eb2f3611cea8df71be26f9f2e4d935d552a530e9c5815f20d20aec6069a476b77fb2b99b2701289def2565b27c772713fee4b0fde8b804b95
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 8db94c576b6845b52ec16fb009a158ef2d733733c8fca48b2fadaef085b371d24b5e5f68758df24ec72189ea7963a9c72cff82b6d6163d1e89ef73de7fd830bd
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 99e086b5b2e39fcc6610232493cf56548913fb5bde9323cf301834b707518e20a6ce5c6d4713f9cd304cc4b9190de077e6d935e359396fabba1c436e658cc8bc

### Node binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 45bed8e46bd18ff86346fe4c3a971411d973b69e5cfd0db58162972bdc37fdf3387642284e43b9436e3862d8f2ee51ad8b147ee13a260b8fc9f42cbca78a1209
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | 3bf9e33cf90cd87679027b63989f3110e486b101189a8f0f05d0d8bdb5d22479ab4f84697413219d54e3c503ad54c533ee985144a57b45f093899e926e5b37fd
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | ae1c5f1a0b40585a42e62075f173cfa9c6bcf81ad16fb9f04bf16e5df9bb02f5526cbdd93fbf1a811cba2001598fd04a53fad731bf4b917d498f60c93124a526
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 3dc8197d953dfd873ecd5e7a2b04d5b8b82d972b774497873f935b2e3ba033f05317866b3b795df56bb06f80e34545f100a89af9083d4ad6e9334295bb5262db
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | ec8f013c3e1a6bb151c968461b3f6b03b2a08283f4d253ec52e83acda2c03ac73fbae1de771baf69dfa26eb3a92f894fd2486ca8323f3d4750640b5b38bd99c4
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.22.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | acc8e3352a8d8ed8640d0787f2fb0d51ab0dac6f84687ab00a05c4a5470f1eb4821c878004e16a829cfd134d38e6f63b4b7f165637085d82a0a638f37e3c081e

## Changelog since v1.21.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Audit log files are now created with a mode of 0600. Existing file permissions will not be changed. If you need the audit file to be readable by a non-root user, you can pre-create the file with the desired permissions. ([#95387](https://github.com/kubernetes/kubernetes/pull/95387), [@JAORMX](https://github.com/JAORMX)) [SIG API Machinery and Auth]
 
## Changes by Kind

### Deprecation

- Kubeadm: remove the deprecated kubeadm API v1beta1. Introduce a new kubeadm API v1beta3. See https://pkg.go.dev/k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3 for a list of changes since v1beta2. Note that v1beta2 is not yet deprecated, but will be in a future release. ([#101129](https://github.com/kubernetes/kubernetes/pull/101129), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- PodUnknown phase is now deprecated. ([#95286](https://github.com/kubernetes/kubernetes/pull/95286), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Apps, CLI, Network, Node, Storage and Testing]
- Removal of the CSI nodepublish path by the kubelet is deprecated. This must be done by the CSI plugin according to the CSI spec. ([#101441](https://github.com/kubernetes/kubernetes/pull/101441), [@dobsonj](https://github.com/dobsonj)) [SIG Storage]

### API Change

- "Auto" is now a valid value for the `service.kubernetes.io/topology-aware-hints` annotation. ([#100728](https://github.com/kubernetes/kubernetes/pull/100728), [@robscott](https://github.com/robscott)) [SIG Apps, Instrumentation and Network]
- Kube-apiserver: `--service-account-issuer` can be specified multiple times now, to enable non-disruptive change of issuer. ([#101155](https://github.com/kubernetes/kubernetes/pull/101155), [@zshihang](https://github.com/zshihang)) [SIG API Machinery, Auth, Node and Testing]
- New "node-high" priority-level has been added to Suggested API Priority and Fairness configuration. ([#101151](https://github.com/kubernetes/kubernetes/pull/101151), [@mborsz](https://github.com/mborsz)) [SIG API Machinery]
- PodDeletionCost promoted to Beta ([#101080](https://github.com/kubernetes/kubernetes/pull/101080), [@ahg-g](https://github.com/ahg-g)) [SIG Apps]
- SSA treats certain structs as atomic ([#100684](https://github.com/kubernetes/kubernetes/pull/100684), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Auth, Node and Storage]
- Server Side Apply now treats all <Some>Selector fields as atomic (meaning the entire selector is managed by a single writer and updated together), since they contain interrelated and inseparable fields that do not merge in intuitive ways. ([#97989](https://github.com/kubernetes/kubernetes/pull/97989), [@Danil-Grigorev](https://github.com/Danil-Grigorev)) [SIG API Machinery]
- The `pods/ephemeralcontainers` API now returns and expects a `Pod` object instead of `EphemeralContainers`. This is incompatible with the previous alpha-level API. ([#101034](https://github.com/kubernetes/kubernetes/pull/101034), [@verb](https://github.com/verb)) [SIG Apps, Auth, CLI and Testing]
- The pod/eviction subresource now accepts policy/v1 Eviction requests in addition to policy/v1beta1 Eviction requests ([#100724](https://github.com/kubernetes/kubernetes/pull/100724), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Storage and Testing]
- Track ownership of scale subresource for all scalable resources i.e. Deployment, ReplicaSet, StatefulSet, ReplicationController, and Custom Resources. ([#98377](https://github.com/kubernetes/kubernetes/pull/98377), [@nodo](https://github.com/nodo)) [SIG API Machinery and Testing]
- We have added a new Priority & Fairness rule that exempts all probes (/readyz, /healthz, /livez) to prevent 
  restarting of "healthy" kube-apiserver instance(s) by kubelet. ([#100678](https://github.com/kubernetes/kubernetes/pull/100678), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]

### Feature

- Base image updates to mitigate kube-proxy and etcd container image CVEs
  - debian-base to buster-v1.6.0
  - debian-iptables to buster-v1.6.0 ([#100976](https://github.com/kubernetes/kubernetes/pull/100976), [@jindijamie](https://github.com/jindijamie)) [SIG Release and Testing]
- EmptyDir memory backed volumes are sized as the the minimum of pod allocatable memory on a host and an optional explicit user provided value. ([#101048](https://github.com/kubernetes/kubernetes/pull/101048), [@dims](https://github.com/dims)) [SIG Node]
- Fluentd: isolate logging resources in separate namespace ([#68004](https://github.com/kubernetes/kubernetes/pull/68004), [@saravanan30erd](https://github.com/saravanan30erd)) [SIG Cloud Provider and Instrumentation]
- It add two flags, `--max-pods` and `--extended-resources` ([#100267](https://github.com/kubernetes/kubernetes/pull/100267), [@Jeffwan](https://github.com/Jeffwan)) [SIG Node and Scalability]
- Kube config is now exposed in the scheduler framework handle. Out-of-tree plugins can leverage that to build CRD informers easily. ([#100644](https://github.com/kubernetes/kubernetes/pull/100644), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Apps, Scheduling and Testing]
- Kubeadm: add --validity-period flag for 'kubeadm kubeconfig user' command ([#100907](https://github.com/kubernetes/kubernetes/pull/100907), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubemark's hollow-node will now print flags before starting ([#101181](https://github.com/kubernetes/kubernetes/pull/101181), [@mm4tt](https://github.com/mm4tt)) [SIG Scalability]
- Kubernetes is now built with Golang 1.16.3 ([#101206](https://github.com/kubernetes/kubernetes/pull/101206), [@justaugustus](https://github.com/justaugustus)) [SIG Cloud Provider, Instrumentation, Release and Testing]
- Promote NamespaceDefaultLabelName to GA.  All Namespace API objects have a `kubernetes.io/metadata.name` label matching their metadata.name field to allow selecting any namespace by its name using a label selector. ([#101342](https://github.com/kubernetes/kubernetes/pull/101342), [@rosenhouse](https://github.com/rosenhouse)) [SIG API Machinery and Apps]
- Run etcd as non-root on GCE provider' ([#100635](https://github.com/kubernetes/kubernetes/pull/100635), [@cindy52](https://github.com/cindy52)) [SIG Cloud Provider]
- SSA is GA ([#100139](https://github.com/kubernetes/kubernetes/pull/100139), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- System-cluster-critical pods should not get a low OOM Score. 
  
  As of now both system-node-critical and system-cluster-critical pods have -997 OOM score, making them one of the last processes to be OOMKilled. By definition system-cluster-critical pods can be scheduled elsewhere if there is a resource crunch on the node where as system-node-critical pods cannot be rescheduled. This was the reason for system-node-critical to have higher priority value than system-cluster-critical.  This change allows only system-node-critical priority class to have low OOMScore.
  
  action required
  If the user wants to have the pod to be OOMKilled last and the pod has system-cluster-critical priority class, it has to be changed to system-node-critical priority class to preserve the existing behavior ([#99729](https://github.com/kubernetes/kubernetes/pull/99729), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Node]
- The job controller removes running pods when the number of completions was achieved. ([#99963](https://github.com/kubernetes/kubernetes/pull/99963), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps]
- `kubectl describe` will by default fetch large lists of resources in chunks of up to 500 items rather than requesting all resources up front from the server. A new flag `--chunk-size=SIZE` may be used to alter the number of items or disable this feature when `0` is passed.  This is a beta feature. ([#101171](https://github.com/kubernetes/kubernetes/pull/101171), [@KnVerey](https://github.com/KnVerey)) [SIG CLI and Testing]
- `kubectl drain` will by default fetch large lists of resources in chunks of up to 500 items rather than requesting all resources up front from the server. A new flag `--chunk-size=SIZE` may be used to alter the number of items or disable this feature when `0` is passed.  This is a beta feature. ([#100148](https://github.com/kubernetes/kubernetes/pull/100148), [@KnVerey](https://github.com/KnVerey)) [SIG CLI and Testing]

### Failing Test

- Fixed generic ephemeal volumes with OwnerReferencesPermissionEnforcement admission plugin enabled. ([#101186](https://github.com/kubernetes/kubernetes/pull/101186), [@jsafrane](https://github.com/jsafrane)) [SIG Auth and Storage]
- Fixes kubectl drain --dry-run=server ([#100206](https://github.com/kubernetes/kubernetes/pull/100206), [@KnVerey](https://github.com/KnVerey)) [SIG CLI and Testing]

### Bug or Regression

- Added privileges for EndpointSlice to the default view & edit RBAC roles ([#101203](https://github.com/kubernetes/kubernetes/pull/101203), [@mtougeron](https://github.com/mtougeron)) [SIG Auth and Security]
- Chain the field manager creation calls in newDefaultFieldManager ([#101076](https://github.com/kubernetes/kubernetes/pull/101076), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery]
- EndpointSlice IP validation now matches Endpoints IP validation. ([#101084](https://github.com/kubernetes/kubernetes/pull/101084), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Ensure service deleted when the Azure resource group has been deleted ([#100944](https://github.com/kubernetes/kubernetes/pull/100944), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Expose `rest_client_rate_limiter_duration_seconds` metric to component-base to track client side rate limiter latency in seconds. Broken down by verb and URL. ([#100311](https://github.com/kubernetes/kubernetes/pull/100311), [@IonutBajescu](https://github.com/IonutBajescu)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Fire an event when failing to open NodePort ([#100599](https://github.com/kubernetes/kubernetes/pull/100599), [@masap](https://github.com/masap)) [SIG Network]
- Fix a bug in kube-proxy latency metrics to calculate only the latency value for the endpoints that are created after it starts running. This is needed because all the endpoints objects are processed on restarts, independently when they were generated. ([#100861](https://github.com/kubernetes/kubernetes/pull/100861), [@aojea](https://github.com/aojea)) [SIG Instrumentation and Network]
- Fix availability set cache in vmss cache ([#100110](https://github.com/kubernetes/kubernetes/pull/100110), [@CecileRobertMichon](https://github.com/CecileRobertMichon)) [SIG Cloud Provider]
- Fix display of Job completion mode in kubectl describe ([#101160](https://github.com/kubernetes/kubernetes/pull/101160), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and CLI]
- Fix panic with kubectl create ingress annotation flag and empty value ([#101377](https://github.com/kubernetes/kubernetes/pull/101377), [@rikatz](https://github.com/rikatz)) [SIG CLI]
- Fix raw block mode CSI NodePublishVolume stage miss pod info ([#99069](https://github.com/kubernetes/kubernetes/pull/99069), [@phantooom](https://github.com/phantooom)) [SIG Storage]
- Fix rounding of volume storage requests ([#100100](https://github.com/kubernetes/kubernetes/pull/100100), [@maxlaverse](https://github.com/maxlaverse)) [SIG Cloud Provider and Storage]
- Fix: azure file inline volume namespace issue in csi migration translation ([#101235](https://github.com/kubernetes/kubernetes/pull/101235), [@andyzhangx](https://github.com/andyzhangx)) [SIG Apps, Cloud Provider, Node and Storage]
- Fix: not delete existing pip when service is deleted ([#100694](https://github.com/kubernetes/kubernetes/pull/100694), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fix: set "host is down" as corrupted mount ([#101398](https://github.com/kubernetes/kubernetes/pull/101398), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider and Storage]
- Fixed a bug where startupProbe stopped working after a container's first restart ([#101093](https://github.com/kubernetes/kubernetes/pull/101093), [@wzshiming](https://github.com/wzshiming)) [SIG Node]
- Fixed port-forward memory leak for long-running and heavily used connections. ([#99839](https://github.com/kubernetes/kubernetes/pull/99839), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery and Node]
- Fixed using volume partitions on AWS Nitro systems. ([#100500](https://github.com/kubernetes/kubernetes/pull/100500), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Generated OpenAPI now correctly specifies 201 as a possible response code for PATCH operations ([#100141](https://github.com/kubernetes/kubernetes/pull/100141), [@brendandburns](https://github.com/brendandburns)) [SIG API Machinery]
- KCM sets the upper-bound timeout limit for outgoing requests to 70s. Previously no timeout was set. Requests without explicit timeout might potentially hang forever and lead to starvation of the application. ([#99358](https://github.com/kubernetes/kubernetes/pull/99358), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Kubeadm: enable '--experimental-patches' flag for 'kubeadm join phase control-plane-join all' command ([#101110](https://github.com/kubernetes/kubernetes/pull/101110), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet: improve the performance when waiting for a synchronization of the node list with the kube-apiserver ([#99336](https://github.com/kubernetes/kubernetes/pull/99336), [@neolit123](https://github.com/neolit123)) [SIG Node]
- Logging for GCE Windows clusters will be more accurate and complete when using Fluent-bit. ([#101271](https://github.com/kubernetes/kubernetes/pull/101271), [@jeremyje](https://github.com/jeremyje)) [SIG Cloud Provider and Windows]
- No support endpointslice in linux userpace mode ([#100913](https://github.com/kubernetes/kubernetes/pull/100913), [@JornShen](https://github.com/JornShen)) [SIG Network]
- Prevent Kubelet stuck in DiskPressure when imagefs minReclaim is set ([#99095](https://github.com/kubernetes/kubernetes/pull/99095), [@maxlaverse](https://github.com/maxlaverse)) [SIG Node]
- Reduce vSphere volume name to 63 characters ([#100404](https://github.com/kubernetes/kubernetes/pull/100404), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Reduces delay initializing on non-AWS platforms docker runtime. ([#93260](https://github.com/kubernetes/kubernetes/pull/93260), [@nckturner](https://github.com/nckturner)) [SIG Cloud Provider]
- Removed `/sbin/apparmor_parser` requirement for the AppArmor host validation.
  This allows using AppArmor on distributions which ship the binary in a different path. ([#97968](https://github.com/kubernetes/kubernetes/pull/97968), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node and Testing]
- Renames the timeout field for the DelegatingAuthenticationOptions to TokenRequestTimeout and set the timeout only for the token review client. Previously the timeout was also applied to watches making them reconnecting every 10 seconds. ([#100959](https://github.com/kubernetes/kubernetes/pull/100959), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery, Auth and Cloud Provider]
- Reorganized iptables rules to reduce rules in KUBE-SERVICES and KUBE-NODEPORTS chains and improve performance ([#96959](https://github.com/kubernetes/kubernetes/pull/96959), [@tssurya](https://github.com/tssurya)) [SIG Network]
- Respect ExecProbeTimeout=false for dockershim ([#100200](https://github.com/kubernetes/kubernetes/pull/100200), [@jackfrancis](https://github.com/jackfrancis)) [SIG Node and Testing]
- Restore kind-specific output for `kubectl describe podsecuritypolicy` ([#101436](https://github.com/kubernetes/kubernetes/pull/101436), [@KnVerey](https://github.com/KnVerey)) [SIG CLI]
- The kubelet now reports distinguishes log messages about certificate rotation for its client cert and server cert separately to make debugging problems with one or the other easier. ([#101252](https://github.com/kubernetes/kubernetes/pull/101252), [@smarterclayton](https://github.com/smarterclayton)) [SIG API Machinery and Auth]
- Updates dependency sigs.k8s.io/structured-merge-diff to v4.1.1 ([#100784](https://github.com/kubernetes/kubernetes/pull/100784), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Upgrades functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv4.1.2 ([#101120](https://github.com/kubernetes/kubernetes/pull/101120), [@monopole](https://github.com/monopole)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Use default timeout of 10s for Azure ACR credential provider. ([#100686](https://github.com/kubernetes/kubernetes/pull/100686), [@hasheddan](https://github.com/hasheddan)) [SIG Cloud Provider]
- [kubeadm] Support for custom imagetags for etcd images which contain build metadata, when imagetags are in the form of version_metadata. For instance, if the etcd version is v3.4.13+patch.0, the supported imagetag would be v3.4.13_patch.0 ([#100350](https://github.com/kubernetes/kubernetes/pull/100350), [@jr0d](https://github.com/jr0d)) [SIG Cluster Lifecycle]

### Other (Cleanup or Flake)

- After the deprecation period,now the Kubelet's  `--chaos-chance` flag  are removed. ([#101057](https://github.com/kubernetes/kubernetes/pull/101057), [@wangyysde](https://github.com/wangyysde)) [SIG Node]
- DynamicFakeClient now exposes its tracker via a `Tracker()` function ([#100085](https://github.com/kubernetes/kubernetes/pull/100085), [@markusthoemmes](https://github.com/markusthoemmes)) [SIG API Machinery]
- Exposes WithCustomRoundTripper method for specifying a middleware function for custom HTTP behaviour for the delegated auth clients. ([#99775](https://github.com/kubernetes/kubernetes/pull/99775), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Migrate some log messages to structured logging in pkg/volume/volume_linux.go. ([#99566](https://github.com/kubernetes/kubernetes/pull/99566), [@huchengze](https://github.com/huchengze)) [SIG Instrumentation and Storage]
- Official binaries now include the golang generated build ID (`buildid`) instead of an empty string. ([#101411](https://github.com/kubernetes/kubernetes/pull/101411), [@saschagrunert](https://github.com/saschagrunert)) [SIG Release]
- Remove deprecated --generator flag from kubectl autoscale ([#99900](https://github.com/kubernetes/kubernetes/pull/99900), [@MadhavJivrajani](https://github.com/MadhavJivrajani)) [SIG CLI]
- Remove the deprecated flag --generator from kubectl create deployment command ([#99915](https://github.com/kubernetes/kubernetes/pull/99915), [@BLasan](https://github.com/BLasan)) [SIG CLI]
- Update Azure Go SDK version to v53.1.0 ([#101357](https://github.com/kubernetes/kubernetes/pull/101357), [@feiskyer](https://github.com/feiskyer)) [SIG API Machinery, CLI, Cloud Provider, Cluster Lifecycle and Instrumentation]
- Update cri-tools dependency to v1.21.0 ([#100956](https://github.com/kubernetes/kubernetes/pull/100956), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]

## Dependencies

### Added
- github.com/gofrs/uuid: [v4.0.0+incompatible](https://github.com/gofrs/uuid/tree/v4.0.0)
- github.com/stoewer/go-strcase: [v1.2.0](https://github.com/stoewer/go-strcase/tree/v1.2.0)
- go.uber.org/tools: 2cfd321

### Changed
- github.com/Azure/azure-sdk-for-go: [v43.0.0+incompatible → v53.1.0+incompatible](https://github.com/Azure/azure-sdk-for-go/compare/v43.0.0...v53.1.0)
- github.com/Azure/go-autorest/autorest/adal: [v0.9.5 → v0.9.10](https://github.com/Azure/go-autorest/autorest/adal/compare/v0.9.5...v0.9.10)
- github.com/Azure/go-autorest/autorest: [v0.11.12 → v0.11.17](https://github.com/Azure/go-autorest/autorest/compare/v0.11.12...v0.11.17)
- github.com/googleapis/gnostic: [v0.4.1 → v0.5.1](https://github.com/googleapis/gnostic/compare/v0.4.1...v0.5.1)
- go.uber.org/atomic: v1.4.0 → v1.6.0
- go.uber.org/multierr: v1.1.0 → v1.5.0
- go.uber.org/zap: v1.10.0 → v1.16.0
- gopkg.in/yaml.v3: 9f266ea → eeeca48
- k8s.io/kube-openapi: 591a79e → 9528897
- sigs.k8s.io/kustomize/api: v0.8.5 → v0.8.8
- sigs.k8s.io/kustomize/cmd/config: v0.9.7 → v0.9.10
- sigs.k8s.io/kustomize/kustomize/v4: v4.0.5 → v4.1.2
- sigs.k8s.io/kustomize/kyaml: v0.10.15 → v0.10.17
- sigs.k8s.io/structured-merge-diff/v4: v4.1.0 → v4.1.1

### Removed
- github.com/satori/go.uuid: [v1.2.0](https://github.com/satori/go.uuid/tree/v1.2.0)