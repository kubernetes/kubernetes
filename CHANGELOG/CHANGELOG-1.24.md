<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.24.0-alpha.4](#v1240-alpha4)
  - [Downloads for v1.24.0-alpha.4](#downloads-for-v1240-alpha4)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.24.0-alpha.3](#changelog-since-v1240-alpha3)
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
- [v1.24.0-alpha.3](#v1240-alpha3)
  - [Downloads for v1.24.0-alpha.3](#downloads-for-v1240-alpha3)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.24.0-alpha.2](#changelog-since-v1240-alpha2)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-1)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.24.0-alpha.2](#v1240-alpha2)
  - [Downloads for v1.24.0-alpha.2](#downloads-for-v1240-alpha2)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.24.0-alpha.1](#changelog-since-v1240-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-2)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.24.0-alpha.1](#v1240-alpha1)
  - [Downloads for v1.24.0-alpha.1](#downloads-for-v1240-alpha1)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.23.0](#changelog-since-v1230)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-3)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-3)
  - [Changes by Kind](#changes-by-kind-3)
    - [Feature](#feature-3)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)

<!-- END MUNGE: GENERATED_TOC -->

# v1.24.0-alpha.4


## Downloads for v1.24.0-alpha.4



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes.tar.gz) | 951531e83aed1aaaf6df424e195a913aa7c6faf9aae4f4b55970b37bc223727201088011f5ed35b988aca36e30b8cea75f6a666721b2c52d672c6c8406d3d9c4
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-src.tar.gz) | c715efaa416a7fe208188bf01f40c34e559cf1c2ed6b153eb843563398ec05b1b4574219bb9a4a548e5f726c30d0739753c7a8086837c0ffeee8f2053a6c463b

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-darwin-amd64.tar.gz) | dd7d18c1babbc2fcbe481f8e41335cadcd9274b27f05c2d3ded19e820e9b8cc55be72eb2cc404afeea8107c503731cd62b0684533582bfb05b3b58b74d8c091e
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-darwin-arm64.tar.gz) | 0ce731610736b3ed26ac0aa9f193cd76adee1e7d34e4dcfb233dbcb83fcd8620e13371ec9583629bb8a404506c779e8f51415756049849c11c230cec475cfbe4
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-386.tar.gz) | bc5e071407dd994bfa45788e4aa395d23e0a3c0431703804db910dc76eea9cff2ff3d02046e4ead8b04c7bec0d148cdd1332f9951a4b546a32b6f3ca2c8e839d
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-amd64.tar.gz) | 2e52b5d5b7852f1d61a7d03bbcf2d20967846f3295501b32014ed99db0694868a5f67e8ea835d58bd6835d1dcba9bdfba4418f10669a71a8859f7768037fe4c9
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-arm.tar.gz) | 6210c9e5a0327b483fa243b88be0f9afeec36c435c0e001bc25360204ea32ebddf98d4dfdf42b93cad683665ad7976706214abfc84f479c0e47f26d971a9752b
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-arm64.tar.gz) | acba30ad585a11e1a875660556118fb449a2e2e92c19d647c030323bbb3f265face715bda90e67458fbd3272fe2c23abad5dfd712874a5c1a232017ca8747984
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-ppc64le.tar.gz) | 9336988dd0933424f70772b21d17a8c798965abbb77722dec58b3a92da8e6ab2f2dcf8702def7f0d4498b9f76a90dcb6af316650569328cc2f988c015ba9c9b7
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-linux-s390x.tar.gz) | d073aa8ad2ee476b12ae1fabcc762d514a01dab18e3e0afb27396e4d2d77bc9091858146c30c8906e9e577411386c009fa591b2f667765dabf5df04c57330f4b
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-windows-386.tar.gz) | 4096d1de90320c4ad68abf7458eb3d57e6e4e8603a430e0109d4b3e1086a8784f284adc567bd99b063e928e7090d60d17491e528416a7e99c39a00aae7679e9e
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-windows-amd64.tar.gz) | e86cc5e6817f5defbc344958ca88c5d0272d65449bf9bedf95a4588ba188d1c1b870c71513fdbf8159d8bce9b7d3026c37f18c42f60649225d542ab6c545f842
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-client-windows-arm64.tar.gz) | bcac6d5daeb604857feff355a6e999ae7f64d748a4c9ecb4393282b6a6ad488bbb43f770de86e173bd2299b18cc98bb9f1690664dc8857753a2f625cc82b15a2

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-amd64.tar.gz) | d511b8ab8d3ecb70f35f472a25de8d3b301d384e347f5aedcaeb286e82f264c8715bd87ac9dbf8474c431c6d10290d49b6b2e92cba9bbfe8b6f0af8f11e434e3
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-arm.tar.gz) | 2b2f4f596675ac3871f0f8ba12f619e550395c3ac40cf3c26b8b5aab4f0a9c0e5b30398bcb9aaf5c5e717ba0eb53e317f01d9f6ff4c37c8a8f2bd644a864b43c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-arm64.tar.gz) | 4da262bb6112ae5a8bf0b65659a8a15fda17fdeab4935f137e7680bef03e6262b7db57cea2db6c1a143ef43ca3346d845bba3e1fa97ca0dba7e3769d333310b4
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-ppc64le.tar.gz) | e64aa2f04c46deef946a88a87449dd46f6cbc5ee7cf4662ba6f455877c4bdc3527edeb8eb861e56fe458d8e49d933db5c4a485b587252ee6f55b7751f01c62cd
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-server-linux-s390x.tar.gz) | 75a4ad3e091709ff2e8ac04f7a7af2999527322a9e857a41985413080a01c5989208648765617ee8bfbc00b462831f3dea5fc512228bda4fcebb45055417c2f7

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-amd64.tar.gz) | 1593bd6ced1aa42e1834a6e049d257a298c7964c706ac07f57d087271e5a4d13866fadc36661affa078bc64f91a7058666fe85879d1d339015e3637669c4c8c6
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-arm.tar.gz) | 883a6b24e134d825330baf1805d563f94300fa060b3c978da9138bba174059b604e28c115437673b0411fbb38e3a7949b99a7089da716ef9a482386dc0b45ca4
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-arm64.tar.gz) | d317105d7ef00696cc27660a871f181a319fcf2d4c1f19a88091f8c7d2bbfbd0a9a4ac8f0fa1e28e432df6ea1215848e23e002eb7f7f51b1f01b67d4acaaaa5a
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-ppc64le.tar.gz) | 01bd08aabd58aee0db6b1e376e76d741d2ebd592995407a47cb315f73c2a5b311a540ae620eb4471eb2aea74c13162107f299efe98c15e5ec6948d9d3e1cd378
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-linux-s390x.tar.gz) | 81cd8e22010cab4cabefb4e2a879a61dd1d5d057ec15bb3b98d5aa5e79820abf0a650cb7c667d7fb38afd9c851a5043df8a148e303a77da2ed14a69a7747c18e
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.4/kubernetes-node-windows-amd64.tar.gz) | 033527ea64ddc4d4f166ebee9e171958fd8b25f1f515c736fe820ae1e7c763b1a79e402e2f8446a216582dda23564dd810bed064d35e8652d27bc51dca224e10

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.24.0-alpha.4](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.24.0-alpha.3

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - The `LegacyServiceAccountTokenNoAutoGeneration` feature gate is beta, and enabled by default. When enabled, Secret API objects containing service account tokens are no longer auto-generated for every ServiceAccount. Use the [TokenRequest](https://kubernetes.io/docs/reference/kubernetes-api/authentication-resources/token-request-v1/) API to acquire service account tokens, or if a non-expiring token is required, create a Secret API object for the token controller to populate with a service account token by following this [guide](https://kubernetes.io/docs/concepts/configuration/secret/#service-account-token-secrets). ([#108309](https://github.com/kubernetes/kubernetes/pull/108309), [@zshihang](https://github.com/zshihang)) [SIG API Machinery, Apps, Auth and Testing]
 
## Changes by Kind

### Deprecation

- --pod-infra-container-image kubelet flag is deprecated and will be removed in future releases ([#108045](https://github.com/kubernetes/kubernetes/pull/108045), [@hakman](https://github.com/hakman)) [SIG Node]
- Client.authentication.k8s.io/v1alpha1 ExecCredential has been removed. If you are using a client-go credential plugin that relies on the v1alpha1 API please contact the distributor of your plugin for instructions on how to migrate to the v1 API. ([#108616](https://github.com/kubernetes/kubernetes/pull/108616), [@margocrawf](https://github.com/margocrawf)) [SIG API Machinery and Auth]
- Remove deprecated feature gates ValidateProxyRedirects and StreamingProxyRedirects ([#106830](https://github.com/kubernetes/kubernetes/pull/106830), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery]
- The node.k8s.io/v1alpha1 RuntimeClass API is no longer served. Use the node.k8s.io/v1 API version, available since v1.20 ([#103061](https://github.com/kubernetes/kubernetes/pull/103061), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, CLI, Node and Testing]

### API Change

- Add 2 new options for kube-proxy running in winkernel mode. 
  `--forward-healthcheck-vip`, if specified as true, health check traffic whose destination is service VIP will be forwarded to kube-proxy's healthcheck service. `--root-hnsendpoint-name` specifies the name of the hns endpoint for the root network namespace.
  This option enables the pass-through load balancers like Google's GCLB to correctly health check the backend services. Without this change, the health check packets is dropped, and Windows node will be considered to be unhealthy by those load balancers. ([#99287](https://github.com/kubernetes/kubernetes/pull/99287), [@anfernee](https://github.com/anfernee)) [SIG API Machinery, Cloud Provider, Network, Testing and Windows]
- Added CEL runtime cost calculation into CustomerResource validation. CustomerResource validation will fail if runtime cost exceeds the budget. ([#108482](https://github.com/kubernetes/kubernetes/pull/108482), [@cici37](https://github.com/cici37)) [SIG API Machinery]
- CRD writes will generate validation errors if a CEL validation rule references the identifier "oldSelf" on a part of the schema that does not support it. ([#108013](https://github.com/kubernetes/kubernetes/pull/108013), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Feature of `DefaultPodTopologySpread` is graduated to GA ([#108278](https://github.com/kubernetes/kubernetes/pull/108278), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Feature of `PodOverhead` is graduated to GA ([#108441](https://github.com/kubernetes/kubernetes/pull/108441), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery, Apps, Node and Scheduling]
- Fixes a regression in v1beta1 PodDisruptionBudget handling of "strategic merge patch"-type API requests for the `selector` field. Prior to 1.21, these requests would merge `matchLabels` content and replace `matchExpressions` content. In 1.21, patch requests touching the `selector` field started replacing the entire selector. This is consistent with server-side apply and the v1 PodDisruptionBudget behavior, but should not have been changed for v1beta1. ([#108138](https://github.com/kubernetes/kubernetes/pull/108138), [@liggitt](https://github.com/liggitt)) [SIG Apps, Auth and Testing]
- Kube-apiserver: --audit-log-version and --audit-webhook-version now only support the default value of audit.k8s.io/v1. The v1alpha1 and v1beta1 audit log versions, deprecated since 1.13, have been removed. ([#108092](https://github.com/kubernetes/kubernetes/pull/108092), [@carlory](https://github.com/carlory)) [SIG API Machinery, Auth and Testing]
- Pod-affinity namespace selector and cross-namespace quota graduated to GA. The feature gate PodAffinityNamespaceSelector is locked and will be removed in 1.26. ([#108136](https://github.com/kubernetes/kubernetes/pull/108136), [@ahg-g](https://github.com/ahg-g)) [SIG API Machinery, Apps, Scheduling and Testing]
- Suspend job to GA. The feature gate SuspendJob is locked and will be removed in 1.26. ([#108129](https://github.com/kubernetes/kubernetes/pull/108129), [@ahg-g](https://github.com/ahg-g)) [SIG Apps and Testing]
- The CertificateSigningRequest `spec.expirationSeconds` API field has graduated to GA. The `CSRDuration` feature gate for the field is now unconditionally enabled and will be removed in 1.26. ([#108782](https://github.com/kubernetes/kubernetes/pull/108782), [@cfryanr](https://github.com/cfryanr)) [SIG API Machinery, Apps, Auth, Instrumentation and Testing]
- TopologySpreadConstraints includes minDomains field to limit the minimum number of topology domains. ([#107674](https://github.com/kubernetes/kubernetes/pull/107674), [@sanposhiho](https://github.com/sanposhiho)) [SIG API Machinery, Apps and Scheduling]

### Feature

- Add a deprecated cmd flag for the time interval between flushing pods from unschedualbeQ to activeQ or backoffQ. ([#108017](https://github.com/kubernetes/kubernetes/pull/108017), [@denkensk](https://github.com/denkensk)) [SIG Scheduling]
- Add one metrics(`kubelet_volume_stats_health_abnormal`) of volume health state to kubelet ([#105585](https://github.com/kubernetes/kubernetes/pull/105585), [@fengzixu](https://github.com/fengzixu)) [SIG Instrumentation, Node, Storage and Testing]
- Add the metric `container_oom_events_total` to kubelet's cAdvisor metric endpoint. ([#108004](https://github.com/kubernetes/kubernetes/pull/108004), [@jonkerj](https://github.com/jonkerj)) [SIG Node]
- Added support for btrfs resizing ([#108561](https://github.com/kubernetes/kubernetes/pull/108561), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]
- CRD x-kubernetes-validations rules now support the CEL functions: isSorted, sum, min, max, indexOf, lastIndexOf, find and findAll. ([#108312](https://github.com/kubernetes/kubernetes/pull/108312), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Client-go metrics: change bucket distribution for rest_client_request_duration_seconds and rest_client_rate_limiter_duration_seconds from [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512] to [0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0}] ([#106911](https://github.com/kubernetes/kubernetes/pull/106911), [@aojea](https://github.com/aojea)) [SIG API Machinery, Architecture and Instrumentation]
- Client-go: add new histogram metric to record the size of the requests and responses. ([#108296](https://github.com/kubernetes/kubernetes/pull/108296), [@aojea](https://github.com/aojea)) [SIG API Machinery, Architecture and Instrumentation]
- Cluster/gce/gci/configure.sh now supports downloading crictl on ARM64 nodes ([#108034](https://github.com/kubernetes/kubernetes/pull/108034), [@tstapler](https://github.com/tstapler)) [SIG Cloud Provider]
- Env var for additional cli flags used in the csi-proxy binary when a Windows nodepool is created with kube-up.sh ([#107806](https://github.com/kubernetes/kubernetes/pull/107806), [@mauriciopoppe](https://github.com/mauriciopoppe)) [SIG Cloud Provider and Windows]
- Increase default value of discovery cache TTL for kubectl to 6 hours. ([#107141](https://github.com/kubernetes/kubernetes/pull/107141), [@mk46](https://github.com/mk46)) [SIG CLI]
- Introduce policy to allow the HPA to consume the external.metrics.k8s.io API group. ([#104244](https://github.com/kubernetes/kubernetes/pull/104244), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Auth, Autoscaling and Instrumentation]
- Kubeadm: apply "second stage" of the plan to migrate kubeadm away from the usage of the word "master" in labels and taints. For new clusters, the label "node-role.kubernetes.io/master" will no longer be added to control plane nodes, only the label "node-role.kubernetes.io/control-plane" will be added. For clusters that are being upgraded to 1.24 with "kubeadm upgrade apply", the command will remove the label "node-role.kubernetes.io/master" from existing control plane nodes. For new clusters, both the old taint "node-role.kubernetes.io/master:NoSchedule" and new taint "node-role.kubernetes.io/control-plane:NoSchedule" will be added to control plane nodes. In release 1.20 ("first stage"), a release note instructed to preemptively tolerate the new taint. For clusters that are being upgraded to 1.24 with "kubeadm upgrade apply", the command will add the new taint "node-role.kubernetes.io/control-plane:NoSchedule" to existing control plane nodes. Please adapt your infrastructure to these changes. In 1.25 the old taint "node-role.kubernetes.io/master:NoSchedule" will be removed. ([#107533](https://github.com/kubernetes/kubernetes/pull/107533), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: better surface errors during "kubeadm upgrade" when waiting for the kubelet to restart static pods on control plane nodes ([#108315](https://github.com/kubernetes/kubernetes/pull/108315), [@Monokaix](https://github.com/Monokaix)) [SIG Cluster Lifecycle]
- Kubeadm: improve the strict parsing of user YAML/JSON configuration files. Next to printing warnings for unknown and duplicate fields (current state), also print warnings for fields with incorrect case sensitivity - e.g. "controlPlaneEndpoint" (valid), "ControlPlaneEndpoint" (invalid). Instead of only printing warnings during "init" and "join" also print warnings when downloading the ClusterConfiguration, KubeletConfiguration or KubeProxyConfiguration objects from the cluster. This can be useful if the user has patched these objects in their respective ConfigMaps with mistakes. ([#107725](https://github.com/kubernetes/kubernetes/pull/107725), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubelet: add kubelet_volume_metric_collection_duration_seconds metrics for volume disk usage calculation duration ([#107201](https://github.com/kubernetes/kubernetes/pull/107201), [@pacoxu](https://github.com/pacoxu)) [SIG Instrumentation, Node and Storage]
- Kubernetes in now built with go1.18rc1 ([#107105](https://github.com/kubernetes/kubernetes/pull/107105), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Release, Storage and Testing]
- No ([#108432](https://github.com/kubernetes/kubernetes/pull/108432), [@iXinqi](https://github.com/iXinqi)) [SIG Testing and Windows]
- PreFilter extension in the scheduler framework now returns not only status but also PreFilterResult ([#108648](https://github.com/kubernetes/kubernetes/pull/108648), [@ahg-g](https://github.com/ahg-g)) [SIG Scheduling, Storage and Testing]
- Remove the deprecated flag `--experimental-check-node-capabilities-before-mount`. With CSI now GA, there is a better alternative. Remove any use of  `--experimental-check-node-capabilities-before-mount` from your kubelet scripts or manifests. ([#104732](https://github.com/kubernetes/kubernetes/pull/104732), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps, Cloud Provider, Node and Storage]
- Set PodMaxUnschedulableQDuration as 5 min. ([#108761](https://github.com/kubernetes/kubernetes/pull/108761), [@denkensk](https://github.com/denkensk)) [SIG Scheduling]
- Support in-tree PV deletion protection finalizer. ([#108400](https://github.com/kubernetes/kubernetes/pull/108400), [@deepakkinni](https://github.com/deepakkinni)) [SIG Apps and Storage]
- The .spec.loadBalancerClass field for Services is now generally available. ([#107979](https://github.com/kubernetes/kubernetes/pull/107979), [@XudongLiuHarold](https://github.com/XudongLiuHarold)) [SIG Cloud Provider, Network and Testing]
- Turn on CSIMigrationAzureFile by default on 1.24 ([#105070](https://github.com/kubernetes/kubernetes/pull/105070), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- When invoked with `-list-images`, the e2e.test binary now also lists the images that might be needed for storage tests. ([#108458](https://github.com/kubernetes/kubernetes/pull/108458), [@pohly](https://github.com/pohly)) [SIG Testing]

### Bug or Regression

- <kubectl version> fails for unexpected extra arguments ([#107967](https://github.com/kubernetes/kubernetes/pull/107967), [@jlsong01](https://github.com/jlsong01)) [SIG CLI]
- Bug: client-go clientset was not defaulting the user agent, using the default golang agent for all the requests. ([#108772](https://github.com/kubernetes/kubernetes/pull/108772), [@aojea](https://github.com/aojea)) [SIG API Machinery and Instrumentation]
- Bump sigs.k8s.io/apiserver-network-proxy/konnectivity-client@v0.0.30 to fix a goroutine leak in kube-apiserver when using egress selctor with the gRPC mode. ([#108437](https://github.com/kubernetes/kubernetes/pull/108437), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, Auth and Cloud Provider]
- Existing InTree AzureFile PVs which don't have a secret namespace defined will now work properly after enabling CSI migration - the namespace will be obtained from ClaimRef. ([#108000](https://github.com/kubernetes/kubernetes/pull/108000), [@RomanBednar](https://github.com/RomanBednar)) [SIG Cloud Provider and Storage]
- Fix a bug in attachdetach controller that didn't properly handle kube-apiserver errors leading to stuck attachments/detachments. ([#108167](https://github.com/kubernetes/kubernetes/pull/108167), [@jfremy](https://github.com/jfremy)) [SIG Apps]
- Fix a bug that out-of-tree plugin is misplaced when using scheduler v1beta3 config ([#108613](https://github.com/kubernetes/kubernetes/pull/108613), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Scheduling and Testing]
- Fix container creation errors for pods with cpu requests bigger than 256 cpus ([#106570](https://github.com/kubernetes/kubernetes/pull/106570), [@odinuge](https://github.com/odinuge)) [SIG Node]
- Fix to allow fsGroup to be applied for CSI Inline Volumes ([#108662](https://github.com/kubernetes/kubernetes/pull/108662), [@dobsonj](https://github.com/dobsonj)) [SIG Storage]
- Fix: do not return early in the node informer when there is no change of the topology label. ([#108149](https://github.com/kubernetes/kubernetes/pull/108149), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fixed a bug that caused credentials in an exec plugin to override the static certificates set in a kubeconfig. ([#107410](https://github.com/kubernetes/kubernetes/pull/107410), [@margocrawf](https://github.com/margocrawf)) [SIG API Machinery, Auth and Testing]
- Fixed a regression that could incorrectly reject pods with OutOfCpu errors if they were rapidly scheduled after other pods were reported as complete in the API. The Kubelet now waits to report the phase of a pod as terminal in the API until all running containers are guaranteed to have stopped and no new containers can be started.  Short-lived pods may take slightly longer (~1s) to report Succeeded or Failed after this change. ([#108366](https://github.com/kubernetes/kubernetes/pull/108366), [@smarterclayton](https://github.com/smarterclayton)) [SIG Apps, Node and Testing]
- Fixes a bug where a partial EndpointSlice update could cause node name information to be dropped from endpoints that were not updated. ([#108198](https://github.com/kubernetes/kubernetes/pull/108198), [@liggitt](https://github.com/liggitt)) [SIG Network]
- Fixes bug in CronJob Controller V2 where it would lose track of jobs upon job template labels change. ([#107997](https://github.com/kubernetes/kubernetes/pull/107997), [@d-honeybadger](https://github.com/d-honeybadger)) [SIG Apps]
- Improved logging when volume times out waiting for attach/detach. ([#108628](https://github.com/kubernetes/kubernetes/pull/108628), [@RomanBednar](https://github.com/RomanBednar)) [SIG Storage]
- Increase Azure ACR credential provider timeout ([#108209](https://github.com/kubernetes/kubernetes/pull/108209), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Kube-apiserver: ensures the namespace of objects sent to admission webhooks matches the request namespace. Previously, objects without a namespace set would have the request namespace populated after mutating admission, and objects with a namespace that did not match the request namespace would be rejected after admission. ([#94637](https://github.com/kubernetes/kubernetes/pull/94637), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Kube-apiserver: removed apf_fd from server logs which could contain data identifying the requesting user ([#108631](https://github.com/kubernetes/kubernetes/pull/108631), [@jupblb](https://github.com/jupblb)) [SIG API Machinery]
- Kube-proxy in iptables mode now only logs the full iptables input at -v=9 rather than -v=5. ([#108224](https://github.com/kubernetes/kubernetes/pull/108224), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Kube-proxy will no longer hold service node ports open on the node. Users are still advised not to run any listener on node ports range used by kube-proxy. ([#108496](https://github.com/kubernetes/kubernetes/pull/108496), [@khenidak](https://github.com/khenidak)) [SIG Network]
- Kubeadm: fix a bug when using "kubeadm init --dry-run" with certificate authority files (ca.key / ca.crt) present in /etc/kubernetes/pki) ([#108410](https://github.com/kubernetes/kubernetes/pull/108410), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where Windows nodes fail to join an IPv6 cluster due to preflight errors ([#108769](https://github.com/kubernetes/kubernetes/pull/108769), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet don't forcefully close active connections on heartbeat failures, using the http2 health check mechanism to detect broken connections. Users can force the previous behavior of the kubelet by setting the environment variable DISABLE_HTTP2. ([#108107](https://github.com/kubernetes/kubernetes/pull/108107), [@aojea](https://github.com/aojea)) [SIG API Machinery and Node]
- Prevent unnecessary Endpoints and EndpointSlice updates caused by Pod ResourceVersion change ([#108078](https://github.com/kubernetes/kubernetes/pull/108078), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Print <default> as the value in case kubectl describe ingress shows default-backend:80 when no default backend is present ([#108506](https://github.com/kubernetes/kubernetes/pull/108506), [@jlsong01](https://github.com/jlsong01)) [SIG CLI]
- Replace the url label of rest_client_request_duration_seconds and rest_client_rate_limiter_duration_seconds metrics with a host label to prevent cardinality explosions and keep only the useful information. This is a breaking change required for security reasons. ([#106539](https://github.com/kubernetes/kubernetes/pull/106539), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG Instrumentation]
- The `TopologyAwareHints` feature gate is now enabled by default. This will allow users to opt in to Topology Aware Hints by setting the `service.kubernetes.io/topology-aware-hints` on a Service. This will not affect any Services without that annotation set. ([#108747](https://github.com/kubernetes/kubernetes/pull/108747), [@robscott](https://github.com/robscott)) [SIG Network]
- This code change fixes the bug that UDP services would trigger unnecessary LoadBalancer updates. The root cause is that a field not working for non-TCP protocols is considered.
  ref: https://github.com/kubernetes-sigs/cloud-provider-azure/pull/1090 ([#107981](https://github.com/kubernetes/kubernetes/pull/107981), [@lzhecheng](https://github.com/lzhecheng)) [SIG Cloud Provider]
- Topology translation of in-tree vSphere volume to vSphere CSI. ([#108611](https://github.com/kubernetes/kubernetes/pull/108611), [@divyenpatel](https://github.com/divyenpatel)) [SIG Storage]

### Other (Cleanup or Flake)

- API server's deprecated `--deserialization-cache-size` flag is now removed. ([#108448](https://github.com/kubernetes/kubernetes/pull/108448), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- API server's deprecated `--experimental-encryption-provider-config` flag is now removed. Adapt your machinery to use the `--encryption-provider-config` flag that is available since v1.13. ([#108423](https://github.com/kubernetes/kubernetes/pull/108423), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- API server's deprecated `--target-ram-mb` flag is now removed. ([#108457](https://github.com/kubernetes/kubernetes/pull/108457), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery, Cloud Provider, Scalability and Testing]
- Endpoints and EndpointSlice controllers no longer populate [resourceVersion of targetRef](https://kubernetes.io/docs/reference/kubernetes-api/common-definitions/object-reference/#ObjectReference) in Endpoints and EndpointSlices ([#108450](https://github.com/kubernetes/kubernetes/pull/108450), [@tnqn](https://github.com/tnqn)) [SIG Apps and Network]
- Improve error message when applying CRDs before the CRD exists in a cluster ([#107363](https://github.com/kubernetes/kubernetes/pull/107363), [@eddiezane](https://github.com/eddiezane)) [SIG CLI]
- Improved algorithm for selecting "best" non-preferred hint in the TopologyManager ([#108154](https://github.com/kubernetes/kubernetes/pull/108154), [@klueska](https://github.com/klueska)) [SIG Node]
- Kube-proxy doesn't set the sysctl net.ipv4.conf.all.route_localnet=1 if no IPv4 loopback address is selected by the nodePortAddresses configuration parameter. ([#107684](https://github.com/kubernetes/kubernetes/pull/107684), [@aojea](https://github.com/aojea)) [SIG Network]
- Remove support for node-expansion between node-stage and node-publish ([#108614](https://github.com/kubernetes/kubernetes/pull/108614), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- The `WarningHeaders` feature gate that is GA since v1.22 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#108394](https://github.com/kubernetes/kubernetes/pull/108394), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG API Machinery]
- The e2e.test binary supports a new `--kubelet-root` parameter to override the default `/var/lib/kubelet` path. CSI storage tests use this. ([#108253](https://github.com/kubernetes/kubernetes/pull/108253), [@pohly](https://github.com/pohly)) [SIG Node, Storage and Testing]
- The scheduler framework option `runAllFilters` is removed. ([#108829](https://github.com/kubernetes/kubernetes/pull/108829), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Windows Pause no longer has support for SAC releases 1903, 1909, 2004. Windows image support is now Ltcs 2019 (1809), 20H2, LTSC 2022 ([#107056](https://github.com/kubernetes/kubernetes/pull/107056), [@jsturtevant](https://github.com/jsturtevant)) [SIG Windows]
- `kube-addon-manager` image version is bumped to 9.1.6 ([#108341](https://github.com/kubernetes/kubernetes/pull/108341), [@zshihang](https://github.com/zshihang)) [SIG Cloud Provider, Scalability and Testing]

## Dependencies

### Added
- github.com/google/gnostic: [v0.5.7-v3refs](https://github.com/google/gnostic/tree/v0.5.7-v3refs)

### Changed
- github.com/cpuguy83/go-md2man/v2: [v2.0.0 → v2.0.1](https://github.com/cpuguy83/go-md2man/v2/compare/v2.0.0...v2.0.1)
- github.com/google/cel-go: [v0.9.0 → v0.10.1](https://github.com/google/cel-go/compare/v0.9.0...v0.10.1)
- github.com/prometheus/client_golang: [v1.12.0 → v1.12.1](https://github.com/prometheus/client_golang/compare/v1.12.0...v1.12.1)
- github.com/russross/blackfriday/v2: [v2.0.1 → v2.1.0](https://github.com/russross/blackfriday/v2/compare/v2.0.1...v2.1.0)
- github.com/spf13/cobra: [v1.2.1 → v1.4.0](https://github.com/spf13/cobra/compare/v1.2.1...v1.4.0)
- golang.org/x/crypto: 32db794 → 8634188
- golang.org/x/mod: v0.5.1 → 9b9b3d8
- golang.org/x/net: 491a49a → cd36cc0
- golang.org/x/oauth2: 2bc19b1 → d3ed0bb
- golang.org/x/sys: da31bd3 → 3681064
- golang.org/x/term: 6886f2d → 03fcf44
- golang.org/x/time: 1f47c86 → 90d013b
- golang.org/x/tools: v0.1.8 → 897bd77
- google.golang.org/genproto: fe13028 → 42d7afd
- k8s.io/kube-openapi: e816edb → ddc6692
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.27 → v0.0.30

### Removed
- cloud.google.com/go/firestore: v1.1.0
- github.com/armon/go-metrics: [f0300d1](https://github.com/armon/go-metrics/tree/f0300d1)
- github.com/armon/go-radix: [7fddfc3](https://github.com/armon/go-radix/tree/7fddfc3)
- github.com/bgentry/speakeasy: [v0.1.0](https://github.com/bgentry/speakeasy/tree/v0.1.0)
- github.com/bketelsen/crypt: [v0.0.4](https://github.com/bketelsen/crypt/tree/v0.0.4)
- github.com/fatih/color: [v1.7.0](https://github.com/fatih/color/tree/v1.7.0)
- github.com/googleapis/gnostic: [v0.5.5](https://github.com/googleapis/gnostic/tree/v0.5.5)
- github.com/hashicorp/consul/api: [v1.1.0](https://github.com/hashicorp/consul/api/tree/v1.1.0)
- github.com/hashicorp/consul/sdk: [v0.1.1](https://github.com/hashicorp/consul/sdk/tree/v0.1.1)
- github.com/hashicorp/errwrap: [v1.0.0](https://github.com/hashicorp/errwrap/tree/v1.0.0)
- github.com/hashicorp/go-cleanhttp: [v0.5.1](https://github.com/hashicorp/go-cleanhttp/tree/v0.5.1)
- github.com/hashicorp/go-immutable-radix: [v1.0.0](https://github.com/hashicorp/go-immutable-radix/tree/v1.0.0)
- github.com/hashicorp/go-msgpack: [v0.5.3](https://github.com/hashicorp/go-msgpack/tree/v0.5.3)
- github.com/hashicorp/go-multierror: [v1.0.0](https://github.com/hashicorp/go-multierror/tree/v1.0.0)
- github.com/hashicorp/go-rootcerts: [v1.0.0](https://github.com/hashicorp/go-rootcerts/tree/v1.0.0)
- github.com/hashicorp/go-sockaddr: [v1.0.0](https://github.com/hashicorp/go-sockaddr/tree/v1.0.0)
- github.com/hashicorp/go-syslog: [v1.0.0](https://github.com/hashicorp/go-syslog/tree/v1.0.0)
- github.com/hashicorp/go-uuid: [v1.0.1](https://github.com/hashicorp/go-uuid/tree/v1.0.1)
- github.com/hashicorp/go.net: [v0.0.1](https://github.com/hashicorp/go.net/tree/v0.0.1)
- github.com/hashicorp/golang-lru: [v0.5.0](https://github.com/hashicorp/golang-lru/tree/v0.5.0)
- github.com/hashicorp/hcl: [v1.0.0](https://github.com/hashicorp/hcl/tree/v1.0.0)
- github.com/hashicorp/logutils: [v1.0.0](https://github.com/hashicorp/logutils/tree/v1.0.0)
- github.com/hashicorp/mdns: [v1.0.0](https://github.com/hashicorp/mdns/tree/v1.0.0)
- github.com/hashicorp/memberlist: [v0.1.3](https://github.com/hashicorp/memberlist/tree/v0.1.3)
- github.com/hashicorp/serf: [v0.8.2](https://github.com/hashicorp/serf/tree/v0.8.2)
- github.com/magiconair/properties: [v1.8.5](https://github.com/magiconair/properties/tree/v1.8.5)
- github.com/mattn/go-colorable: [v0.0.9](https://github.com/mattn/go-colorable/tree/v0.0.9)
- github.com/mattn/go-isatty: [v0.0.3](https://github.com/mattn/go-isatty/tree/v0.0.3)
- github.com/miekg/dns: [v1.0.14](https://github.com/miekg/dns/tree/v1.0.14)
- github.com/mitchellh/cli: [v1.0.0](https://github.com/mitchellh/cli/tree/v1.0.0)
- github.com/mitchellh/go-homedir: [v1.0.0](https://github.com/mitchellh/go-homedir/tree/v1.0.0)
- github.com/mitchellh/go-testing-interface: [v1.0.0](https://github.com/mitchellh/go-testing-interface/tree/v1.0.0)
- github.com/mitchellh/gox: [v0.4.0](https://github.com/mitchellh/gox/tree/v0.4.0)
- github.com/mitchellh/iochan: [v1.0.0](https://github.com/mitchellh/iochan/tree/v1.0.0)
- github.com/pascaldekloe/goe: [57f6aae](https://github.com/pascaldekloe/goe/tree/57f6aae)
- github.com/pelletier/go-toml: [v1.9.3](https://github.com/pelletier/go-toml/tree/v1.9.3)
- github.com/posener/complete: [v1.1.1](https://github.com/posener/complete/tree/v1.1.1)
- github.com/ryanuber/columnize: [9b3edd6](https://github.com/ryanuber/columnize/tree/9b3edd6)
- github.com/sean-/seed: [e2103e2](https://github.com/sean-/seed/tree/e2103e2)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/spf13/cast: [v1.3.1](https://github.com/spf13/cast/tree/v1.3.1)
- github.com/spf13/jwalterweatherman: [v1.1.0](https://github.com/spf13/jwalterweatherman/tree/v1.1.0)
- github.com/spf13/viper: [v1.8.1](https://github.com/spf13/viper/tree/v1.8.1)
- github.com/subosito/gotenv: [v1.2.0](https://github.com/subosito/gotenv/tree/v1.2.0)
- gopkg.in/ini.v1: v1.62.0



# v1.24.0-alpha.3


## Downloads for v1.24.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes.tar.gz) | 9de65cc28a4a641c4bbfd386706cf61a8c36d8928623ab6cac28ac378d77aed458e14e8782703d58182a2aab31a71ea5806bb7013f7ddb8657fdc350790f03cd
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-src.tar.gz) | 7f76982cbc1eb84883b86ac2b3b017b02dd970cd3a57883306ec0cbf7dfac9009ee368ca93cec43261dd186d1b75bcde4fe61a6eb4b804361b23b2c8ef2755df

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | ff98f55de0e1db549135811e801afe85e7e4b2216d4dc543adc9e6039f997ad9f2ecbd06daf3deb14d86afb736a6a114ddde760925b647be1f6ea6fb1d9b5153
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | 8582dcb6cbf56eca07cbfe98f06adf0f6e0c2827783d6a8c3fbc7e7e5b69f4b5781643beae37fe0868bde8915aa45c991db550e6fec619a7bec94eb38ed4c3b5
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 15a56b02915397970c90615ad921e9e3125e39b8ee8a857c22c855957fee8b2de540eb51b2be4f3f8138bee1bf5ed7dfe0b8f1171df9ff25b6789c5ba420dc3c
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | a16a6762713dd8aaf5a8f5cab4f9be949ac04ce7bd1e182858c47a1769f43d5c99faebee228e8543c237444a3ffb55bf1b6c5272a370f545c137b699c12cdce6
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | ff9432a0b3538e5170276ced1f4878570a2a2ddb5c59c25d735eed09dde492bc2a2bbde5cccbcd835cbae94f4912f6b5c1b82168e81b4e106204ebe342c4e866
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | bbc22c098152f1a0e2532e3d7e700116a31fe59212ca8d47f9f95d3f3c997c1d8e8a1b75619bc73feb9f2c4953733d7e5b67ee34e8b5781b3b8d9865075f4b7f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | a6046b9726268f0557ab8c72fe5274ea5554afdf7b6b2aa77d41a207ecb5460e42d5ba68de3e359318956af5c390e46b3471418b413d6ee54169262245f8fe46
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | c5701618eba6bec591445f2bd192301178fa6b0b8c5359d81faea92a393611a5fe697c2dea241769b7c55f2a2dd5fbf1361ee5f0edefa6582ec88ce0f248b1b4
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 87f8d99bacbf846b3cf0e6ccc72e99c38363ef606103ea924330a49f1a9d46c3043c8eda397246246d0fac0fe1204944628bf5add48b1a51a618ba4523df4673
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 74ba2c6770113dc91f18072a1df15f368ea23312e418a016e25f93545a5e234d1e133210f0e136642d2a4e23574a015625ca434234b085d1f1a16821ba2dc266
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | f72d3d0948e0669b23227cc952d20e40054088df52c12c0f84ff9b2c6714e558f7d2349fc8b78758d57bd251e1f2ea50296a8a0f22fbea762b048c547eb78e26

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | f68d4691dfa7916c8e566c1ca2dfb11b0c675223a95cef8b90586ead93217713122ae402f1276246a4c1daa45f3545ee9fd600a7f97fdc056a23b5e66acb9921
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | babf77bfb1061120dd525b25bf89b7bb0b81892dddfd70a1693ddbe642e3066104dd54938e1bfff86069b66cf5af6e0698236cc25c62daa1187548255c512bb4
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 0b8846b01d20fada109fef957b5670ee2da9cb334090475ecf9de508beae912a330cb446d77a8308782ab77ed37784838dd8b287c55d5a54a64588992fa9c385
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | a6d3c3b261aa0abcfb5ecb8c94cec75ca01da839100c6ec0456a5b0e4e118b4c603954b5094e434da25489528a4cf346675e04d4c213202053f6b5ab431bad67
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 10cd559a40cac8e35d89976703a839a6095bf7a52e2cf1c090007e43bf3b487a6eba19e86d066f4a76922acd3217ec4b62dfbb2f1ccb3b422bc84ffdf844b161

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | 32702b81a4f0a19c9ef28978880b150284d12c766c25d89b83808bff44ec99ebf629585ec9a5ee4cec72ecb27e8282802578e39c586b4bb53e5caf02f80012a0
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | 2fca5344d8debe5a37676793992db19a1ef296d3d0d14682f7a8f023a5ee5c5a1691bbd585483b277ea81ed2487d2260da9cd869b55116d6c2606a630c2a29d7
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | 424ea735c4bde3e3e70ae5e6517ec4c9ca46628b8f8e74eba5775b85faf31c9e3d9c6fb876f2965b86e596f1268741f75851d95ec91a3c9eff4ef0b772a3e3b6
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 517aff6677ac101bb09b2f2aba7d96d8901ae4b0ed6f7ba33bd8ab33447babd7eaf2a9482d0ebdc1b3f235c7fa9e6b4437fd19ccc6518889265e78fcdc8a6d27
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | dcc61fcc5ac44a6e25f53f2538e41bd23e8c34c2b481c3bb8be6caf42ab74b0363f9ddce16c41c5083a70ff4fe1c569d26dc6fcd71b06ad6203ddc3fd7c1be0f
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.24.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | 95afe157524f285127fe2e808ec785fb0aa3205dd0b1f34233d35c0eabd8efbcba6b9cd98d69b66464cdb1b178d28c33fd091d95fdcbdfe5a8d6af2df7d44363

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
k8s.gcr.io/conformance:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-apiserver:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-controller-manager:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-proxy:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x
k8s.gcr.io/kube-scheduler:v1.24.0-alpha.3 | amd64, arm, arm64, ppc64le, s390x

## Changelog since v1.24.0-alpha.2

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Fixed bug with leads to Node goes "Not-ready" state when credentials for vCenter stored in a secret and Zones feature is in use.
  Zone labels setup moved to KCM component, kubelet skips this step during startup in such case. If credentials stored in cloud-provider config file as plaintext current behaviour does not change and no action required.
  
  For proper functioning "kube-system:vsphere-legacy-cloud-provider" should be allowed to update node object if vCenter credentials stored in secret and Zone feature used. ([#101028](https://github.com/kubernetes/kubernetes/pull/101028), [@lobziik](https://github.com/lobziik)) [SIG Cloud Provider]
 
## Changes by Kind

### Deprecation

- Cluster addon for dashboard was removed. To install dashboard, see [here](https://github.com/kubernetes/dashboard/blob/master/docs/user/README.md). ([#107481](https://github.com/kubernetes/kubernetes/pull/107481), [@shu-mutou](https://github.com/shu-mutou)) [SIG Cloud Provider and Testing]
- Kube-apiserver: the --master-count flag and --endpoint-reconciler-type=master-count reconciler are deprecated in favor of the lease reconciler ([#108062](https://github.com/kubernetes/kubernetes/pull/108062), [@aojea](https://github.com/aojea)) [SIG API Machinery]
- Kubeadm: graduate the UnversionedKubeletConfigMap feature gate to Beta and enable the feature by default. This implies that 1) for new clusters kubeadm will start using the "kube-system/kubelet-config" naming scheme for the kubelet ConfigMap and RBAC rules, instead of the legacy "kubelet-config-x.yy" naming. 2) during upgrade, kubeadm will only write the new scheme ConfigMap and RBAC objects. To disable the feature you can pass "UnversionedKubeletConfigMap: false" in the kubeadm config for new clusters. For upgrade on existing clusters you can also override the behavior by patching the ClusterConfiguration object in "kube-system/kubeadm-config". More details in the associated KEP. ([#108027](https://github.com/kubernetes/kubernetes/pull/108027), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- Remove `tolerate-unready-endpoints` annotation in Service deprecated from 1.11, use `Service.spec.publishNotReadyAddresses` instead. ([#108020](https://github.com/kubernetes/kubernetes/pull/108020), [@tossmilestone](https://github.com/tossmilestone)) [SIG Apps and Network]
- The in-tree azure plugin has been deprecated.  The https://github.com/Azure/kubelogin serves as an out-of-tree replacement via the kubectl/client-go credential plugin mechanism.  Users will now see a warning in the logs regarding this deprecation. ([#107904](https://github.com/kubernetes/kubernetes/pull/107904), [@sabbey37](https://github.com/sabbey37)) [SIG Auth]

### API Change

- CRD deep copies should no longer contain shallow copies of JSONSchemaProps.XValidations. ([#107956](https://github.com/kubernetes/kubernetes/pull/107956), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Feature of `NonPreemptingPriority` is graduated  to GA ([#107432](https://github.com/kubernetes/kubernetes/pull/107432), [@denkensk](https://github.com/denkensk)) [SIG Apps, Scheduling and Testing]
- Fix OpenAPI serialization of the x-kubernetes-validations field ([#107970](https://github.com/kubernetes/kubernetes/pull/107970), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-apiserver: the `metadata.selfLink` field can no longer be populated by kube-apiserver; it was deprecated in 1.16 and has not been populated by default in 1.20+. ([#107527](https://github.com/kubernetes/kubernetes/pull/107527), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Apps, Auth, Autoscaling, CLI, Cloud Provider, Network, Scheduling, Storage and Testing]

### Feature

- Kubernetes is now built with Golang 1.17.7 ([#108091](https://github.com/kubernetes/kubernetes/pull/108091), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Remove feature gate `SetHostnameAsFQDN`. ([#108038](https://github.com/kubernetes/kubernetes/pull/108038), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Node]
- The output of `kubectl describe ingress` now includes an IngressClass name if available ([#107921](https://github.com/kubernetes/kubernetes/pull/107921), [@mpuckett159](https://github.com/mpuckett159)) [SIG CLI]
- The scheduler prints info logs when the extender returned an error. (--v>5) ([#107974](https://github.com/kubernetes/kubernetes/pull/107974), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- `kubectl config delete-user` now supports completion ([#107142](https://github.com/kubernetes/kubernetes/pull/107142), [@dimbleby](https://github.com/dimbleby)) [SIG CLI]
- `kubectl create token` can now be used to request a service account token, and permission to request service account tokens is added to the `edit` and `admin` RBAC roles ([#107880](https://github.com/kubernetes/kubernetes/pull/107880), [@liggitt](https://github.com/liggitt)) [SIG Auth, CLI and Testing]

### Bug or Regression

- A static pod that is rapidly updated was failing to start until the Kubelet was restarted. ([#107900](https://github.com/kubernetes/kubernetes/pull/107900), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- CRI-API: IPs returned by PodSandboxNetworkStatus are ignored by the kubelet for host-network pods. ([#106715](https://github.com/kubernetes/kubernetes/pull/106715), [@aojea](https://github.com/aojea)) [SIG Node]
- Fixes bug in TopologyManager for ensuring aligned allocations on machines with more than 2 NUMA nodes ([#108052](https://github.com/kubernetes/kubernetes/pull/108052), [@klueska](https://github.com/klueska)) [SIG Node]
- Kubeadm: fix a bug related to a warning printed if the KubeletConfiguration "resolvConf" field value does not match "/run/systemd/resolve/resolv.conf" ([#107785](https://github.com/kubernetes/kubernetes/pull/107785), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubeadm: fix the bug that 'kubeadm certs generate-csr' command does not remove duplicated SANs ([#107982](https://github.com/kubernetes/kubernetes/pull/107982), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- RunCordonOrUncordon error if drainer has nil Ctx or Client ([#105297](https://github.com/kubernetes/kubernetes/pull/105297), [@jackfrancis](https://github.com/jackfrancis)) [SIG CLI]

### Other (Cleanup or Flake)

- Added an e2e test to verify that the cluster is not vulnerable to CVE-2021-29923 when using Services with IPs with leading zeros, note that this test is a necessary but not sufficient condition, all the components in the clusters that consume IPs addresses from the APIs MUST interpret them as decimal or discard them. ([#107552](https://github.com/kubernetes/kubernetes/pull/107552), [@aojea](https://github.com/aojea)) [SIG Network and Testing]
- Kubectl stack traces will now only print at -v=99 and not -v=6 ([#108053](https://github.com/kubernetes/kubernetes/pull/108053), [@eddiezane](https://github.com/eddiezane)) [SIG CLI]
- Remove kubelet `--non-masquerade-cidr` deprecated CLI flag ([#107096](https://github.com/kubernetes/kubernetes/pull/107096), [@hakman](https://github.com/hakman)) [SIG Cloud Provider and Node]
- [k8s.io/utils/clock]: IntervalClock is now deprecated in favour of SimpleIntervalClock ([#108059](https://github.com/kubernetes/kubernetes/pull/108059), [@RaghavRoy145](https://github.com/RaghavRoy145)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]

## Dependencies

### Added
_Nothing has changed._

### Changed
- k8s.io/utils: 7d6a63d → 3a6ce19

### Removed
_Nothing has changed._



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

 - Docker runtime support using dockshim in the kubelet is now completely removed in 1.24. The kubelet used to have a module called "dockershim" which implements CRI support for Docker, and it has seen maintenance issues in the Kubernetes community. From 1.24 onwards, please move to a container runtime that is a full-fledged implementation of CRI (v1alpha1 or v1 compliant) as they become available. ([#97252](https://github.com/kubernetes/kubernetes/pull/97252), [@dims](https://github.com/dims)) [SIG Cloud Provider, Instrumentation, Network, Node and Testing]
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
  - 'regular': indicates that it is a non-long-running request
  - 'watch': indicates that it is a watch request ([#105517](https://github.com/kubernetes/kubernetes/pull/105517), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Instrumentation]
- Add a test to guarantee that conformance clusters require at least 2 untainted nodes ([#106313](https://github.com/kubernetes/kubernetes/pull/106313), [@aojea](https://github.com/aojea)) [SIG Architecture and Testing]
- Allow attached volumes to be mounted quicker by skipping exp. backoff when checking for reported-in-use volumes ([#106853](https://github.com/kubernetes/kubernetes/pull/106853), [@gnufied](https://github.com/gnufied)) [SIG Apps, Node and Storage]
- An inefficient lock in EndpointSlice controller metrics cache has been reworked. Network programming latency may be significantly reduced in certain scenarios, especially in clusters with a large number of Services. ([#107091](https://github.com/kubernetes/kubernetes/pull/107091), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Scalability]
- Apiserver will now reject connection attempts to 0.0.0.0/:: when handling a proxy subresource request ([#107402](https://github.com/kubernetes/kubernetes/pull/107402), [@anguslees](https://github.com/anguslees)) [SIG Network]
- Apiserver, if configured to reconcile the kubernetes. Default service endpoints, checks if the configured Service IP range matches the apiserver public address IP family, and fails to start if not. ([#106721](https://github.com/kubernetes/kubernetes/pull/106721), [@aojea](https://github.com/aojea)) [SIG API Machinery and Testing]
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
- Some log messages were logged with `"v":0` in JSON output, although they are debug messages with a higher verbosity. ([#106978](https://github.com/kubernetes/kubernetes/pull/106978), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Scheduling and Storage]
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
- Kubeadm: make sure that "kubeadm init/join" always use a URL scheme (unix:// on Linux and npipe:// on Windows) when passing a value to the "--container-runtime-endpoint" kubelet flag. This flag's value is taken from the kubeadm configuration "criSocket" field or the "--cri-socket" CLI flag. Automatically add a missing URL scheme to the user configuration in memory, but warn them that they should also update their configuration on disk manually. During "kubeadm upgrade apply/node" mutate the "/var/lib/kubelet/kubeadm-flags.env" file on disk and the "kubeadm.alpha.kubernetes.io/cri-socket" annotation Node object if needed. These automatic actions are temporary and will be removed in a future release. In the future the kubelet may not support CRI endpoints without a URL scheme. ([#107295](https://github.com/kubernetes/kubernetes/pull/107295), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the IPv6DualStack feature gate. The feature has been GA and locked to enabled since 1.23. ([#106648](https://github.com/kubernetes/kubernetes/pull/106648), [@calvin0327](https://github.com/calvin0327)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: remove the deprecated output/v1alpha1 API used for machine-readable output by some kubeadm commands. In 1.23 kubeadm started using the newer version output/v1alpha2 for the same purpose. ([#107468](https://github.com/kubernetes/kubernetes/pull/107468), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
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

 - Docker runtime support using dockshim in the kubelet is now completely removed in 1.24. The kubelet used to have a module called "dockershim" which implements CRI support for Docker, and it has seen maintenance issues in the Kubernetes community. From 1.24 onwards, please move to a container runtime that is a full-fledged implementation of CRI (v1alpha1 or v1 compliant) as they become available. ([#97252](https://github.com/kubernetes/kubernetes/pull/97252), [@dims](https://github.com/dims)) [SIG Cloud Provider, Instrumentation, Network, Node and Testing]
 
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