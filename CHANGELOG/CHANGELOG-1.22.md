<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.22.0-alpha.1](#v1220-alpha1)
  - [Downloads for v1.22.0-alpha.1](#downloads-for-v1220-alpha1)
    - [Source Code](#source-code)
    - [Client binaries](#client-binaries)
    - [Server binaries](#server-binaries)
    - [Node binaries](#node-binaries)
  - [Changelog since v1.21.0](#changelog-since-v1210)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)

<!-- END MUNGE: GENERATED_TOC -->

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