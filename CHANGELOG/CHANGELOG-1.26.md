<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.26.0-beta.0](#v1260-beta0)
  - [Downloads for v1.26.0-beta.0](#downloads-for-v1260-beta0)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.26.0-alpha.3](#changelog-since-v1260-alpha3)
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
- [v1.26.0-alpha.3](#v1260-alpha3)
  - [Downloads for v1.26.0-alpha.3](#downloads-for-v1260-alpha3)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.26.0-alpha.2](#changelog-since-v1260-alpha2)
  - [Changes by Kind](#changes-by-kind-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.26.0-alpha.2](#v1260-alpha2)
  - [Downloads for v1.26.0-alpha.2](#downloads-for-v1260-alpha2)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.26.0-alpha.1](#changelog-since-v1260-alpha1)
  - [Changes by Kind](#changes-by-kind-2)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.26.0-alpha.1](#v1260-alpha1)
  - [Downloads for v1.26.0-alpha.1](#downloads-for-v1260-alpha1)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.25.0](#changelog-since-v1250)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-3)
    - [Feature](#feature-3)
    - [Documentation](#documentation)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)

<!-- END MUNGE: GENERATED_TOC -->

# v1.26.0-beta.0


## Downloads for v1.26.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes.tar.gz) | 9aa7ea4dac63ca19b62dbb5ff3769f96d52f17d14050bdb4832936b6732879b93544ffae4411783e57b5171e12bc7bba8dbd275fdbc0755712a0b80069d06097
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-src.tar.gz) | 350ee84981bdc47f1ccee421efe2102d1323195b605c79884a0a3628c49d20533bbf3f49d54a3ce94b2a5627290103a4edd14cfdd1bd732c859f88ad06ad178a

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | 8333a7b382ce29c79f9d2958c90e5e34c3af205a64d7f99bf94817df92879b136ba1f40a675555368aee68a9278a03142f20b8cb1797d1eaa3ba2344e2109904
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | 5f263002532b818c9dca80119f7fe78474f7fee66d13409e8fad588b1aa7edda7a333a1f0982b86582b0a202f57253a6ae7a64ecba9569e9b08f478f1cc2c2e3
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-linux-386.tar.gz) | 344a33e30a29043533810d48f42d34d25a919925f85610b232c8c2f9da04c6faa2e43bc45dc7cb2d04c4c7bc24e6d77621abdc667a4a0707082212505babe5d5
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | 267dd3143813d7462dc821ec2ebf22c266280420fdecbbaf73e4f03a803ef4be5e0e98bbac036e0ba96e4c56ba937cf1064ed91208dff0b91797b3243810d097
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-linux-arm.tar.gz) | c1779aa4bed88510640de2f2c964c981f188a6a15d2e468e503982ad63f20a0f282752d9e3d9e811895ccf7e8847fe9c7bbecb76d64d087e83a9677c8d6a6ad0
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | 82170b76010c8f54c8a40684a1226433626afabd6c585cd41035e17aa8923d1c3991cbae0d77ca79153a972d8840b92d1958e253e3a9ae5eda2b9e8d9c09d01e
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | d0e59ec798ef03c01990e184847b1bfd38805d9e95901699c5bbbdf31d2e942dab63a8fb68656dc9affd0fa483efb360751d5d0b445f9d6c1e9713c1f10d1f7f
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | d2ebeadcdd809f9f1ee4bd1884efd5093279cc3511c791007061ee980becdf7e1e3980f61f644b7425d1ec10c386d8c74e9296031472376bf8ea481c047920e9
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-windows-386.tar.gz) | 04d7a1387112428283081fc74bcaf83d7a7dbe59f58bad45794603e8dfa4cee723aa1b4fd1616c96dc9ff2e49a246345d4135756d9779a86916d41e4cbeae46c
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | db7680b960de8f2f0da782ae2e6b2e396c5b4606e7c894af5bf4e5627fb83d635b3b7f893af80252515bd0fef2accf6b598ba5042ffd5e9c8d71cff68fc4ab25
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | 7ff409c0c1f2ca26f42dd6199b559df390238f17c1bf868a10e8d1433bfe7305bed57f20b187a806076f2788a64ad224998ac5dfaeb20a7cef01dfc6bf025de0

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | ef82141b01f845ad0576207cf528a9d1f8be681e1fd4744d4e01f3692491a0a640de92f79ef4294d924deee29926de4b0eefc6757addc6a27557c79ca94e3c46
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-server-linux-arm.tar.gz) | 14f2be17866492accd69225b55ddca636aa46cd825a9092bb2bf05cd2adc04c59e0b8271adab4b345b8368337863a3884d608ca7e8de48d3598d1b144e4142dc
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | c7df332e9bb9c20abdd3a0e2a57509e3ef7b5ea0eadee6cffc09c6cbffb0e01fb845a1135a7d4ea3e784227022839bae8936a3d95846b5fde23bf7e096413c1c
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | a5e5b2d60a4fde3db2214a0509c677e94c205fab7350b57dca79558a999f28752fe096ee863d4c9c410079fab3a08665aec84cb1a1732d53e9ac09cffe65b389
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | 4d84170a3a5bcea73db3c922154724e4021dd3fd20833698428002975eec1a958f528f54d747870ace58859741eeebc7caec1074ae84ba08b35d5a1efa1ab0d1

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 2a194c2e2da4949df32806a7592716406ab3148287e6c97285155e0c7390b8cbcdbd426fb4ff40885f1db7b31355e7bcf9f590edeb77318ba5e39e92e40569f1
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-node-linux-arm.tar.gz) | 960ab6a725cd5b9ac59449cda00605a4f4d876541b362852cd2c915b1cf449713139c540f1a7d8e48920a67515fc3389007313cfa348e886ec7e4cb7c783e90e
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | ee3c62ab7174e737c372325a5bc086b61d42b957211e6fb1061aafee8f24284ceca22c0d7c2e92020327a8cf4bd1fe9a8cd685174c0c5ae03bb7ed293c1d6dc6
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 28dc29007d319172c82b6ae675a218ce4dc484ddb81371ddccd5e5aeced90aa4033a08eb6ac3d562627b7762988d7de2f72fbfddada454009b9f3c0137d23864
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | ebdb47d96ae97ec6abfa9ec0863b1ead84615c49be950e79dff27a8a6a2454044854976545017947528ab104007f9010a27d68b96916219934e541bbafd23851
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | f2197c28414f98a77cc501a47a960be22c45a19f1023c0a4a426442aa719a7c6b66a660ad9721d817216e59ac2ce8a2d0e7db89c1e36b4938833329af40b85af

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.26.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.26.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.26.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.26.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.26.0-beta.0](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.26.0-alpha.3

## Changes by Kind

### Deprecation

- CLI flag `pod-eviction-timeout` is deprecated and will be removed together with `enable-taint-manager` in v1.27. ([#113710](https://github.com/kubernetes/kubernetes/pull/113710), [@kerthcet](https://github.com/kerthcet)) [SIG API Machinery and Apps]

### API Change

- A new `preEnqueue` extension point is added to scheduler's component config v1beta2/v1beta3/v1. ([#113275](https://github.com/kubernetes/kubernetes/pull/113275), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG API Machinery, Apps, Instrumentation, Scheduling and Testing]
- Add DataSourceRef2 alpha field to PersistentVolumeClaimSpec API. ([#113186](https://github.com/kubernetes/kubernetes/pull/113186), [@ttakahashi21](https://github.com/ttakahashi21)) [SIG API Machinery, Apps, Storage and Testing]
- Add a kube-proxy flag (--iptables-localhost-nodeports, default true) to allow disabling NodePort services on loopback addresses. Note: this only applies to iptables mode and ipv4. ([#108250](https://github.com/kubernetes/kubernetes/pull/108250), [@cyclinder](https://github.com/cyclinder)) [SIG API Machinery, Cloud Provider, Network, Node, Scalability, Storage and Testing]
- Added a --topology-manager-policy-options flag to the kubelet to support fine tuning the topology manager policies. The first policy option, `prefer-closest-numa-nodes`, allows these policies to favor sets of NUMA nodes with shorter distance between nodes when making admission decisions. ([#112914](https://github.com/kubernetes/kubernetes/pull/112914), [@PiotrProkop](https://github.com/PiotrProkop)) [SIG API Machinery and Node]
- Added a feature that allows a StatefulSet to start numbering replicas from an arbitrary non-negative ordinal, using the `.spec.ordinals.start` field. ([#112744](https://github.com/kubernetes/kubernetes/pull/112744), [@pwschuurman](https://github.com/pwschuurman)) [SIG API Machinery and Apps]
- Deprecate the apiserver_request_slo_duration_seconds metric for v1.27 in favor of apiserver_request_sli_duration_seconds for naming consistency purposes with other SLI-specific metrics and to avoid any confusion between SLOs and SLIs. ([#112679](https://github.com/kubernetes/kubernetes/pull/112679), [@dgrisonnet](https://github.com/dgrisonnet)) [SIG API Machinery and Instrumentation]
- Enable the "Retriable and non-retriable pod failures for jobs" feature into beta ([#113360](https://github.com/kubernetes/kubernetes/pull/113360), [@mimowo](https://github.com/mimowo)) [SIG Apps, Auth, Node, Scheduling and Testing]
- Graduate JobTrackingWithFinalizers to stable.
  Jobs created before the feature was enabled are still tracked without finalizers.
  Users can choose to migrate jobs to tracking with finalizers by adding the annotation batch.kubernetes.io/job-tracking.
  If the annotation was already present and the user attempts to remove it, the control plane adds the annotation back. ([#113510](https://github.com/kubernetes/kubernetes/pull/113510), [@alculquicondor](https://github.com/alculquicondor)) [SIG API Machinery, Apps and Testing]
- Graduate ServiceInternalTrafficPolicy feature to GA ([#113496](https://github.com/kubernetes/kubernetes/pull/113496), [@avoltz](https://github.com/avoltz)) [SIG Apps and Network]
- If you enabled automatic reload of encryption configuration with API server flag --encryption-provider-config-automatic-reload, ensure all the KMS provider names (v1 and v2) in the encryption configuration are unique. ([#113697](https://github.com/kubernetes/kubernetes/pull/113697), [@aramase](https://github.com/aramase)) [SIG API Machinery and Auth]
- Introduce v1alpha1 API for validating admission policies, enabling extensible admission control via CEL expressions (KEP  3488: CEL for Admission Control). To use, enable the `ValidatingAdmissionPolicy` feature gate and the `admissionregistration.k8s.io/v1alpha1` API via `--runtime-config`. ([#113314](https://github.com/kubernetes/kubernetes/pull/113314), [@cici37](https://github.com/cici37)) [SIG API Machinery, Auth, Cloud Provider and Testing]
- Kubelet adds the following pod failure conditions:
  - DisruptionTarget (graceful node shutdown, node pressure eviction) ([#112360](https://github.com/kubernetes/kubernetes/pull/112360), [@mimowo](https://github.com/mimowo)) [SIG Apps, Node and Testing]
- Metav1.LabelSelectors specified in API objects are now validated to ensure they do not contain invalid label values that will error at time of use. Existing invalid objects can be updated, but new objects are required to contain valid label selectors. ([#113699](https://github.com/kubernetes/kubernetes/pull/113699), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Auth, Network and Storage]
- Moving MixedProtocolLBService from beta to GA ([#112895](https://github.com/kubernetes/kubernetes/pull/112895), [@janosi](https://github.com/janosi)) [SIG Apps, Network and Testing]
- New Pod API field `.spec.schedulingGates` is introduced to enable users to control when to mark a Pod as scheduling ready. ([#113274](https://github.com/kubernetes/kubernetes/pull/113274), [@Huang-Wei](https://github.com/Huang-Wei)) [SIG Apps, Scheduling and Testing]
- NodeInclusionPolicy in podTopologySpread plugin is enabled by default. ([#113500](https://github.com/kubernetes/kubernetes/pull/113500), [@kerthcet](https://github.com/kerthcet)) [SIG API Machinery, Apps, Scheduling and Testing]
- Priority and Fairness has introduced a new feature called _borrowing_ that allows an API priority level
  to borrow a number of seats from other priority level(s). As a cluster operator, you can enable borrowing
  for a certain priority level configuration object via the two newly introduced fields `lendablePercent`, and
  `borrowingLimitPercent` located under the `.spec.limited` field of the designated priority level.
  This PR adds the following metrics.
  - `apiserver_flowcontrol_nominal_limit_seats`: Nominal number of execution seats configured for each priority level
  - `apiserver_flowcontrol_lower_limit_seats`: Configured lower bound on number of execution seats available to each priority level
  - `apiserver_flowcontrol_upper_limit_seats`: Configured upper bound on number of execution seats available to each priority level
  - `apiserver_flowcontrol_demand_seats`: Observations, at the end of every nanosecond, of (the number of seats each priority level could use) / (nominal number of seats for that level)
  - `apiserver_flowcontrol_demand_seats_high_watermark`: High watermark, over last adjustment period, of demand_seats
  - `apiserver_flowcontrol_demand_seats_average`: Time-weighted average, over last adjustment period, of demand_seats
  - `apiserver_flowcontrol_demand_seats_stdev`: Time-weighted standard deviation, over last adjustment period, of demand_seats
  - `apiserver_flowcontrol_demand_seats_smoothed`: Smoothed seat demands
  - `apiserver_flowcontrol_target_seats`: Seat allocation targets
  - `apiserver_flowcontrol_seat_fair_frac`: Fair fraction of server's concurrency to allocate to each priority level that can use it
  - `apiserver_flowcontrol_current_limit_seats`: current derived number of execution seats available to each priority level
  
  The possibility of borrowing means that the old metric apiserver_flowcontrol_request_concurrency_limit can no longer mean both the configured concurrency limit and the enforced concurrency limit.  Henceforth it means the configured concurrency limit. ([#113485](https://github.com/kubernetes/kubernetes/pull/113485), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery and Testing]
- The EndpointSliceTerminatingCondition feature gate has graduated to GA. The gate is now locked and will be removed in v1.28. ([#113351](https://github.com/kubernetes/kubernetes/pull/113351), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, Apps, Network and Testing]
- Yes, aggregated discovery will be alpha and can be toggled with the AggregatedDiscoveryEndpoint feature flag ([#113171](https://github.com/kubernetes/kubernetes/pull/113171), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Network, Node, Release, Scalability, Scheduling, Storage and Testing]

### Feature

- API Server tracing now includes the latency of authorization, priorityandfairness, impersonation, audit, and authentication filters. ([#113217](https://github.com/kubernetes/kubernetes/pull/113217), [@dashpole](https://github.com/dashpole)) [SIG API Machinery and Instrumentation]
- Add a method `StreamWithContext` to remotecommand.Executor to support cancelable SPDY executor stream. ([#103177](https://github.com/kubernetes/kubernetes/pull/103177), [@arkbriar](https://github.com/arkbriar)) [SIG API Machinery, CLI, Node and Testing]
- Add alpha support for returning container and pod metrics from CRI, instead of cAdvsior ([#113609](https://github.com/kubernetes/kubernetes/pull/113609), [@haircommander](https://github.com/haircommander)) [SIG Architecture, Instrumentation and Node]
- Add support for Evented PLEG feature gate ([#111384](https://github.com/kubernetes/kubernetes/pull/111384), [@harche](https://github.com/harche)) [SIG Node and Testing]
- Add the metric pod_start_sli_duration_seconds to kubelet ([#111930](https://github.com/kubernetes/kubernetes/pull/111930), [@azylinski](https://github.com/azylinski)) [SIG Instrumentation, Node and Testing]
- Added reconstruction of SELinux mount context after kubelet restart. Feature SELinuxMountReadWriteOncePod is now fully implemented and kubelet does not lose its cache of SELinux contexts after kubelet process restart. ([#113596](https://github.com/kubernetes/kubernetes/pull/113596), [@jsafrane](https://github.com/jsafrane)) [SIG Apps, Node, Storage and Testing]
- Added selector validation to HorizontalPodAutoscaler: when multiple HPAs select the same set of Pods, scaling now will be disabled for those HPAs with the reason `AmbiguousSelector`. This change also covers a case when multiple HPAs point to the same deployment. ([#112011](https://github.com/kubernetes/kubernetes/pull/112011), [@pbeschetnov](https://github.com/pbeschetnov)) [SIG Apps and Autoscaling]
- Added: publishing events when enabling/disabling topologyAwareHints. ([#113544](https://github.com/kubernetes/kubernetes/pull/113544), [@LiorLieberman](https://github.com/LiorLieberman)) [SIG Apps and Network]
- Adding alpha support for WindowsHostNetworking feature ([#112961](https://github.com/kubernetes/kubernetes/pull/112961), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- Adds alpha --output plaintext protected by environment variable `KUBECTL_EXPLAIN_OPENAPIV3` ([#113146](https://github.com/kubernetes/kubernetes/pull/113146), [@alexzielenski](https://github.com/alexzielenski)) [SIG CLI]
- Adds metrics `force_delete_pods_total` and `force_delete_pod_errors_total` in the Pod GC Controller. ([#113519](https://github.com/kubernetes/kubernetes/pull/113519), [@xing-yang](https://github.com/xing-yang)) [SIG Apps]
- CSIMigrationvSphere upgraded to GA and locked to true. Do not upgrade to K8s 1.26 if you need Windows support until vSphere CSI Driver adds support for it in a version post v2.7.x. ([#113336](https://github.com/kubernetes/kubernetes/pull/113336), [@divyenpatel](https://github.com/divyenpatel)) [SIG Storage]
- DelegateFSGroupToCSIDriver feature is GA. ([#113225](https://github.com/kubernetes/kubernetes/pull/113225), [@bertinatto](https://github.com/bertinatto)) [SIG Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- Graduate Kubelet CPU Manager to GA. ([#113018](https://github.com/kubernetes/kubernetes/pull/113018), [@fromanirh](https://github.com/fromanirh)) [SIG Node and Testing]
- Graduate Kubelet Device Manager to GA. ([#112980](https://github.com/kubernetes/kubernetes/pull/112980), [@swatisehgal](https://github.com/swatisehgal)) [SIG Cloud Provider and Node]
- If `ComponentSLIs` feature gate is enabled, then `/metrics/slis` becomes available on cloud-controller-manager allowing you to scrape health check metrics. ([#113340](https://github.com/kubernetes/kubernetes/pull/113340), [@Richabanker](https://github.com/Richabanker)) [SIG Cloud Provider]
- Kubectl config view now automatically redacts any secret fields marked with a datapolicy tag ([#109189](https://github.com/kubernetes/kubernetes/pull/109189), [@mpuckett159](https://github.com/mpuckett159)) [SIG API Machinery, Auth, CLI and Testing]
- Kubectl shell completions for the bash shell now include descriptions. ([#113636](https://github.com/kubernetes/kubernetes/pull/113636), [@marckhouzam](https://github.com/marckhouzam)) [SIG CLI]
- Kubernetes is now built with Go 1.19.3 ([#113550](https://github.com/kubernetes/kubernetes/pull/113550), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Make Azure File CSI migration as GA in 1.26 ([#113160](https://github.com/kubernetes/kubernetes/pull/113160), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- NodeOutOfServiceVolumeDetach is now beta. ([#113511](https://github.com/kubernetes/kubernetes/pull/113511), [@xing-yang](https://github.com/xing-yang)) [SIG Node and Storage]
- Pod Security admission: the pod-security `warn` level will now default to the `enforce` level. ([#113491](https://github.com/kubernetes/kubernetes/pull/113491), [@tallclair](https://github.com/tallclair)) [SIG Auth and Security]
- Promote kubectl alpha events to kubectl events ([#113819](https://github.com/kubernetes/kubernetes/pull/113819), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Promote the `APIServerIdentity` feature to Beta. By default, each kube-apiserver will now create a Lease in the `kube-system` namespace. These lease objects can be used to identify the number of active API servers in the cluster, and may also be used for future features such as the Storage Version API. ([#113629](https://github.com/kubernetes/kubernetes/pull/113629), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery and Testing]
- Promoting WindowsHostProcessContainers to stable ([#113476](https://github.com/kubernetes/kubernetes/pull/113476), [@marosset](https://github.com/marosset)) [SIG Apps, Node, Testing and Windows]
- RetroactiveDefaultStorageClass feature is now beta. ([#113329](https://github.com/kubernetes/kubernetes/pull/113329), [@RomanBednar](https://github.com/RomanBednar)) [SIG Apps, Storage and Testing]
- The LegacyServiceAccountTokenNoAutoGeneration feature gate has been promoted to GA ([#112838](https://github.com/kubernetes/kubernetes/pull/112838), [@zshihang](https://github.com/zshihang)) [SIG API Machinery, Apps, Auth and Testing]
- The ProxyTerminatingEndpoints feature is now Beta and enabled by default. When enabled, kube-proxy will attempt to route traffic to terminating pods when the traffic policy is Local and there are only terminating pods remaining on a node. ([#113363](https://github.com/kubernetes/kubernetes/pull/113363), [@andrewsykim](https://github.com/andrewsykim)) [SIG Network]
- The iptables kube-proxy backend should process service/endpoint changes
  more efficiently in very large clusters. ([#110268](https://github.com/kubernetes/kubernetes/pull/110268), [@danwinship](https://github.com/danwinship)) [SIG Instrumentation and Network]
- Update the Lease identity naming format for the APIServerIdentity feature to use a persistent name ([#113307](https://github.com/kubernetes/kubernetes/pull/113307), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery, Node and Testing]
- Updated cAdvisor to v0.46.0 ([#113769](https://github.com/kubernetes/kubernetes/pull/113769), [@bobbypage](https://github.com/bobbypage)) [SIG Architecture, CLI, Cloud Provider, Node and Storage]

### Bug or Regression

- Apiserver: use the correct error when logging errors updating managedFields ([#113711](https://github.com/kubernetes/kubernetes/pull/113711), [@andrewsykim](https://github.com/andrewsykim)) [SIG API Machinery]
- Bump runc to v1.1.4 ([#113719](https://github.com/kubernetes/kubernetes/pull/113719), [@pacoxu](https://github.com/pacoxu)) [SIG Node]
- Do not raise an error when setting an annotation with the same value, just ignore it. ([#109505](https://github.com/kubernetes/kubernetes/pull/109505), [@zigarn](https://github.com/zigarn)) [SIG CLI]
- Fix cost estimation of token creation request for service account in Priority and Fairness. ([#113206](https://github.com/kubernetes/kubernetes/pull/113206), [@marseel](https://github.com/marseel)) [SIG API Machinery]
- Fix that disruption controller changes the status of a stale disruption condition after 2 min when the PodDisruptionConditions feature gate is enabled ([#113580](https://github.com/kubernetes/kubernetes/pull/113580), [@mimowo](https://github.com/mimowo)) [SIG Auth]
- Fix the PodAndContainerStatsFromCRI feature, instead of supplementing with stats from cAdvisor. ([#113291](https://github.com/kubernetes/kubernetes/pull/113291), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation and Node]
- Fixed DaemonSet to update the status even if it fails to create a pod. ([#112127](https://github.com/kubernetes/kubernetes/pull/112127), [@gjkim42](https://github.com/gjkim42)) [SIG Apps and Testing]
- For `kubectl`, `--server-side` now migrates ownership of all fields used by client-side-apply to the specified `--fieldmanager`. This prevents fields previously specified using kubectl from being able to live outside of server-side-apply's management and become undeleteable. ([#112905](https://github.com/kubernetes/kubernetes/pull/112905), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery, CLI and Testing]
- Kubectl apply: warning that kubectl will ignore no-namespaced resource `pv & namespace` in a future release if the namespace is specified and allowlist is not specified ([#110907](https://github.com/kubernetes/kubernetes/pull/110907), [@pacoxu](https://github.com/pacoxu)) [SIG CLI]
- Kubelet: Fixes a startup crash in devicemanager ([#113021](https://github.com/kubernetes/kubernetes/pull/113021), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Kubelet: fix nil pointer in reflector start for standalone mode ([#113501](https://github.com/kubernetes/kubernetes/pull/113501), [@pacoxu](https://github.com/pacoxu)) [SIG Node]
- NOTE ([#113749](https://github.com/kubernetes/kubernetes/pull/113749), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Pod logs using --timestamps are not broken up with timestamps anymore. ([#113481](https://github.com/kubernetes/kubernetes/pull/113481), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Resolves an issue that causes winkernel proxier to treat stale VIPs as valid ([#113521](https://github.com/kubernetes/kubernetes/pull/113521), [@daschott](https://github.com/daschott)) [SIG Network and Windows]
- The resourceVersion returned in objects from delete responses is now consistent with the resourceVersion contained in the delete watch event ([#113369](https://github.com/kubernetes/kubernetes/pull/113369), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]

### Other (Cleanup or Flake)

- A new API server flag --encryption-provider-config-automatic-reload has been added to control when the encryption config should be automatically reloaded without needing to restart the server.  All KMS plugins are merged into a single healthz check at /healthz/kms-providers when reload is enabled, or when only KMS v2 plugins are used. ([#113529](https://github.com/kubernetes/kubernetes/pull/113529), [@enj](https://github.com/enj)) [SIG API Machinery, Auth and Testing]
- Added a `--prune-allowlist` flag that can be used with `kubectl apply --prune`. This flag replaces and functions the same as the `--prune-whitelist` flag, which has been deprecated. ([#113116](https://github.com/kubernetes/kubernetes/pull/113116), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Deprecated the following kubectl run flags, which are ignored if set: --cascade, --filename, --force, --grace-period, --kustomize, --recursive, --timeout, --wait ([#112261](https://github.com/kubernetes/kubernetes/pull/112261), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Dropped support for the Container Runtime Interface (CRI) version `v1alpha2`, which means that container runtimes just have to implement `v1`. ([#110618](https://github.com/kubernetes/kubernetes/pull/110618), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node and Security]
- Promote  job-related metrics to stable to follow IndexedJobs GA, the following metrics had their name updated to match metrics API guidelines:
  - job_sync_total -> job_syncs_total
  - job_finished_total -> jobs_finished_total ([#113010](https://github.com/kubernetes/kubernetes/pull/113010), [@soltysh](https://github.com/soltysh)) [SIG Apps and Instrumentation]
- Promote cronjob_job_creation_skew metric to stable to follow the cronjob v2 controller, the following metrics had their name updated to match metrics API guidelines:
  - cronjob_job_creation_skew_duration_seconds -> job_creation_skew_duration_seconds ([#113008](https://github.com/kubernetes/kubernetes/pull/113008), [@soltysh](https://github.com/soltysh)) [SIG Apps and Instrumentation]
- Rename the feature gate for CEL in Admission Control to `ValidatingAdmissionPolicy`. ([#113735](https://github.com/kubernetes/kubernetes/pull/113735), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]
- `kubelet_kubelet_credential_provider_plugin_duration` is renamed `kubelet_credential_provider_plugin_duration` and `kubelet_kubelet_credential_provider_plugin_errors` is renamed `kubelet_credential_provider_plugin_errors`. ([#113754](https://github.com/kubernetes/kubernetes/pull/113754), [@logicalhan](https://github.com/logicalhan)) [SIG Instrumentation and Node]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/container-storage-interface/spec: [v1.6.0 → v1.7.0](https://github.com/container-storage-interface/spec/compare/v1.6.0...v1.7.0)
- github.com/containerd/ttrpc: [v1.0.2 → v1.1.0](https://github.com/containerd/ttrpc/compare/v1.0.2...v1.1.0)
- github.com/docker/docker: [v20.10.17+incompatible → v20.10.18+incompatible](https://github.com/docker/docker/compare/v20.10.17...v20.10.18)
- github.com/docker/go-units: [v0.4.0 → v0.5.0](https://github.com/docker/go-units/compare/v0.4.0...v0.5.0)
- github.com/google/cadvisor: [v0.45.0 → v0.46.0](https://github.com/google/cadvisor/compare/v0.45.0...v0.46.0)
- github.com/karrick/godirwalk: [v1.16.1 → v1.17.0](https://github.com/karrick/godirwalk/compare/v1.16.1...v1.17.0)
- github.com/moby/sys/mountinfo: [v0.6.0 → v0.6.2](https://github.com/moby/sys/mountinfo/compare/v0.6.0...v0.6.2)
- github.com/moby/term: [3f7ff69 → 39b0c02](https://github.com/moby/term/compare/3f7ff69...39b0c02)
- github.com/opencontainers/runc: [v1.1.3 → v1.1.4](https://github.com/opencontainers/runc/compare/v1.1.3...v1.1.4)
- github.com/prometheus/client_golang: [v1.13.0 → v1.14.0](https://github.com/prometheus/client_golang/compare/v1.13.0...v1.14.0)
- github.com/prometheus/client_model: [v0.2.0 → v0.3.0](https://github.com/prometheus/client_model/compare/v0.2.0...v0.3.0)
- k8s.io/utils: 665eaae → 1a15be2

### Removed
_Nothing has changed._



# v1.26.0-alpha.3


## Downloads for v1.26.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes.tar.gz) | e38caad0331adb21176e326bbcf1b4f55385b00011fd476c12a7872a2cfa1d25d2d961f2986cc336549be71bc8f3513578317a852d39c0ad6c62bd3bd4ce17d1
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-src.tar.gz) | f5931854054f3d739636fccc78c29a9cb579e3885143dae7d89c72d9e41857acf91ee0166ef92766e414355258409ca209ce7c2cf512322cc0f2b17bb9670098

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | fbfa2b4af54b8765465a8892ad7ceb56f019132805d7c1f0d63e9e569a33a722862234c1184f395c865bf1e8393951b8ea50bf972766d8283281bd2fc0ada86a
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | 5b9af35b9284a7f4813031f913e3aa014d3a1b1fa7823564409c8c6ce7a8a2272b2fb306bcbb777736d241b7997fcf4fed97388614ccdb25463f8f41f3398204
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 0fe3a80e9dfa85798f1da7b24c94c83328d4a00e4032b6dbe03c674fe39bd35a8a53c7b005fc9d81f0992920d4b66d25dc3a50c87df9884cdc4df9a61cca82f8
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | 36ef2a886f871bbf674aa740de4c2f61be4f80b8dd68c38b784af4a92a03af659ffab8de6a06cb400fbeea6bcb6f2eef9e809c5959a9050de5e0dd63521fcefa
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | ae8a4f8e568843ea6f4321b0b7e2779ef1177e86ea979355509cf5788a05d1b08eaae54d400296b41ef5463d7e576df9d5943104ac830cf17f7e9e879460dc43
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | c8f10f35b38509b9c02af2946dfa96e86a8dca69c587754cdfe0cda0cfee8fbcf118fd8fa92543a04571d370c3a9942f40d19e0740efa301a2f9e542be4c0051
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | c416055c92483ec93d0ecfd4c1738ed1c9aafd9d68f806dc0e9b3fbe994c60fbb41baaa9a64fd1a5725ae16d6e22e93c40f8be5af4e122d1eb1d095d2f7a6c26
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 88155930c24f784d6f73864af07518d0cdde486e32857d1d057d0ed0bfce7c8c7501f1fc29275747d7b50d812e910b1fb2fa14f7ef91aeedeaf530cdbe9094f7
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 3c89c5044ca402bb022af618404839eb0cadb3e89552c4002386503911b1e4622a16a4452c26310ee2ead49058faa17711136435f3303271be903dabce93ae0d
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 8d053b7eb02e850d2d70492d0cbd76e6b68ff43b3373539563c67ab2dc66895a387f4e7dfb31dffde85a32086beb8db3a12a55dd5e9f4261bd768c853be3e6fa
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | 0034b8b34b66cf72f9a77ffe0ab441517f12ba7bafbcf8abea7b6650f8936da0a4e90f56a315dc2d16392684167850401de6c1dbf386bce2e28aeab9e3b4963d

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 1b7c60428a326c337c59080923766d46c7b4a9c0ffc7125d6000f410a1cbd488403a00d36ea22a7fbddf70aa78c53e06c4d0cb434f254b7e883481393a99dec7
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | 4eec57f333e4d5cc99929465e6bf01f057351925d5b38c92751c1b30a2f1f61d81db14d95100c5ea01950eee4ef172772e91dc38000ec50bfa1e01f8a62e0d54
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | a6a11db5f669e0705083269df9813c8da9e7d61f76c78ac0cb55ea9cfffad865e0de7b129727192cec03ecae89d7e51f1097b010290106a5ac8a563294fb69b2
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | 3b66310e6adad44b38a02eb315281e15e3f6f4fd6490ea279a3f962ca27b6537f4210a2db8e0648816614e2a9d73a622fdf0bac2ea0dc2c462543ce44acc3a45
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 4dd94f22f4a309aa3fc0f92d1708a31660b291270107baf8ed80bbb49677136cce17345edf21e1a132584702218192e85876d36211884880629a8a38b98e6680

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | 1c8b21a36d18a9a6c3ebf92cd61d9b36ef06b5e415f4bbf86518d3a740cb132a767c1b806e0b65e3a52eb8a2fc6452d0e897aaeea4505ad5206b0b304f784e0e
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | 23797df2c5eb2fe473d06edb3fe032f2c55bb68eaa2074d1ce3f0455545076be5859288bc835d194da2291dc488da375ecbf21511f9876ca80da978f5ff3491c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | 5f2ce6867228dc4cdb4a418421ac28d7f3a4e4e2f5ba7470cd141dc3b45978537d19ba81f0bd3751ba0ba31a3f67b41d7fac7932623cfd719b526fe3a879e42c
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 9acdaf23e8e8a8bc5afac8fbccd1325f45a0bcf7ac1e02beaf2e54e54dfc95e5d90b5efeb9dfbfa9ebfd83331f7daeb928f2ad579ba02d234e0294a9df1e1d6e
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | d971dc06b94a02b23d384c366946a0304f53513717180351c70eef989841a33940fa56698d4f30161cd15ac47f2577569ff2cf63775d5c191ba798695d48f252
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | a2db5de15c72c6bc9bcd29dca31f6abf5c6cd3dd60c448bf4c4d93ad13f2864cd0904feaa5f0d3954c0199dcd64191dc97128863275e69705a3208c91f7f8b73

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.26.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.26.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.26.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.26.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.26.0-alpha.3](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.26.0-alpha.2

## Changes by Kind

### API Change

- **Additional documentation e.g., KEPs (Kubernetes Enhancement Proposals), usage docs, etc.**:
  
  <!--
  This section can be blank if this pull request does not require a release note.
  
  When adding links which point to resources within git repositories, like
  KEPs or supporting documentation, please reference a specific commit and avoid
  linking directly to the master branch. This ensures that links reference a
  specific point in time, rather than a document that may change over time.
  
  See here for guidance on getting permanent links to files: https://help.github.com/en/articles/getting-permanent-links-to-files
  
  Please use the following format for linking documentation:
  - [KEP]: <link>
  - [Usage]: <link>
  - [Other doc]: <link>
  --> ([#86139](https://github.com/kubernetes/kubernetes/pull/86139), [@jasimmons](https://github.com/jasimmons)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Contributor Experience, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Add percentageOfNodesToScore as a scheduler profile level parameter to API version v1. If a profile percentageOfNodesToScore is set, it will override global percentageOfNodesToScore. ([#112521](https://github.com/kubernetes/kubernetes/pull/112521), [@yuanchen8911](https://github.com/yuanchen8911)) [SIG API Machinery, Scheduling and Testing]
- Kube-controller-manager supports '--concurrent-horizontal-pod-autoscaler-syncs' flag to set the number of horizontal pod autoscaler controller workers. ([#108501](https://github.com/kubernetes/kubernetes/pull/108501), [@zroubalik](https://github.com/zroubalik)) [SIG API Machinery, Apps and Autoscaling]
- Kube-proxy: The "userspace" proxy mode (deprecated for over a year) is no longer supported on either Linux or Windows.  Users should use "iptables" or "ipvs" on Linux, or "kernelspace" on Windows. ([#112133](https://github.com/kubernetes/kubernetes/pull/112133), [@knabben](https://github.com/knabben)) [SIG API Machinery, Network, Scalability, Testing and Windows]
- Kubectl wait command with jsonpath flag will wait for target path appear until timeout. ([#109525](https://github.com/kubernetes/kubernetes/pull/109525), [@jonyhy96](https://github.com/jonyhy96)) [SIG CLI and Testing]
- Kubelet external Credential Provider feature is moved to GA. Credential Provider Plugin and Credential Provider Config APIs updated from v1beta1 to v1 with no API changes. ([#111616](https://github.com/kubernetes/kubernetes/pull/111616), [@ndixita](https://github.com/ndixita)) [SIG API Machinery, Node, Scheduling and Testing]
- The `DynamicKubeletConfig` feature gate has been removed from the API server. Dynamic kubelet reconfiguration now cannot be used even when older nodes are still attempting to rely on it. This is aligned with the Kubernetes version skew policy. ([#112643](https://github.com/kubernetes/kubernetes/pull/112643), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Apps, Auth, Node and Testing]

### Feature

- API Server Tracing now includes a variety of new spans and span events. ([#113172](https://github.com/kubernetes/kubernetes/pull/113172), [@dashpole](https://github.com/dashpole)) [SIG API Machinery, Architecture, Auth, Instrumentation, Network, Node and Scheduling]
- Add kubelet metrics to track the cpumanager cpu allocation and pinning ([#112855](https://github.com/kubernetes/kubernetes/pull/112855), [@fromanirh](https://github.com/fromanirh)) [SIG Instrumentation, Node and Testing]
- Added categories column to the `kubectl api-resources` command's wide output (`-o wide`).
  Added `--categories` flag to the `kubectl api-resources` command, which can be used to filter the output to show only resources belonging to one or more categories. ([#111096](https://github.com/kubernetes/kubernetes/pull/111096), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Admission control plugin "DefaultStorageClass": If more than one StorageClass is designated as default (via the "storageclass.kubernetes.io/is-default-class" annotation), choose the newest one instead of throwing an error. ([#110559](https://github.com/kubernetes/kubernetes/pull/110559), [@danishprakash](https://github.com/danishprakash)) [SIG Apps and Storage]
- Change  `preemption_victims` metric bucket from `LinearBuckets` to `ExponentialBuckets`. ([#112939](https://github.com/kubernetes/kubernetes/pull/112939), [@lengrongfu](https://github.com/lengrongfu)) [SIG Instrumentation and Scheduling]
- Extend the job `job_finished_total metric by new `reason` label and introduce a new job metric to count pod failures
  handled by pod failure policy with respect to the action applied. ([#113324](https://github.com/kubernetes/kubernetes/pull/113324), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- Graduate ServiceIPStaticSubrange feature to GA ([#112163](https://github.com/kubernetes/kubernetes/pull/112163), [@aojea](https://github.com/aojea)) [SIG Network]
- If `ComponentSLIs` feature gate is enabled, then `/metrics/slis` becomes available on kube-controller-manager, allowing you to scrape health check metrics. ([#112978](https://github.com/kubernetes/kubernetes/pull/112978), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery and Cloud Provider]
- If `ComponentSLIs` feature gate is enabled, then `/metrics/slis` becomes available on kube-proxy allowing you to scrape health check metrics. ([#113057](https://github.com/kubernetes/kubernetes/pull/113057), [@Richabanker](https://github.com/Richabanker)) [SIG Network]
- If `ComponentSLIs` feature gate is enabled, then `/metrics/slis` becomes available on kube-scheduler, allowing you to scrape health check metrics. ([#113026](https://github.com/kubernetes/kubernetes/pull/113026), [@Richabanker](https://github.com/Richabanker)) [SIG Scheduling]
- If `ComponentSLIs` feature gate is enabled, then `/metrics/slis` becomes available on kubelet, allowing you to scrape health check metrics. ([#113030](https://github.com/kubernetes/kubernetes/pull/113030), [@Richabanker](https://github.com/Richabanker)) [SIG Node]
- Kubeadm: command `kubeadm join phase control-plane-prepare certs` is now supporting to run with `dry-run` mode on it's own ([#113005](https://github.com/kubernetes/kubernetes/pull/113005), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Logs of requests that were timed out by a timeout handler will no longer contain a "statusStack" and "logging error output" fields ([#112374](https://github.com/kubernetes/kubernetes/pull/112374), [@Argh4k](https://github.com/Argh4k)) [SIG API Machinery]
- Metrics for RetroactiveDefaultStorageClass feature are now available. To see an attempt count for updating PVC retroactively with a default StorageClass see `retroactive_storageclass_total` metric and for total numer of errors see `retroactive_storageclass_errors_total`. ([#113323](https://github.com/kubernetes/kubernetes/pull/113323), [@RomanBednar](https://github.com/RomanBednar)) [SIG Apps]
- New metric job_controller_terminated_pods_tracking_finalizer can be used to monitor whether the job controller is removing Pod finalizers from terminated Pods after accounting them in Job status. ([#113176](https://github.com/kubernetes/kubernetes/pull/113176), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps, Instrumentation and Testing]
- Shell completion will now show plugin names when appropriate.  Furthermore, shell completion will work for plugins that provide such support. ([#105867](https://github.com/kubernetes/kubernetes/pull/105867), [@marckhouzam](https://github.com/marckhouzam)) [SIG CLI]
- The ExpandedDNSConfig feature has graduated to beta and is enabled by default. Note that this feature requires container runtime support. ([#112824](https://github.com/kubernetes/kubernetes/pull/112824), [@gjkim42](https://github.com/gjkim42)) [SIG Network and Testing]
- When the alpha `LegacyServiceAccountTokenTracking` feature gate is enabled, secret-based service account tokens will have a `kubernetes.io/legacy-token-last-used` applied to them containing the date they were last used. ([#108858](https://github.com/kubernetes/kubernetes/pull/108858), [@zshihang](https://github.com/zshihang)) [SIG API Machinery, Auth and Testing]

### Bug or Regression

- Bump golang.org/x/net to v0.1.1-0.20221027164007-c63010009c80 ([#112693](https://github.com/kubernetes/kubernetes/pull/112693), [@aimuz](https://github.com/aimuz)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Security and Storage]
- Fix the occasional double-counting of the job_finished_total metric ([#112948](https://github.com/kubernetes/kubernetes/pull/112948), [@mimowo](https://github.com/mimowo)) [SIG Apps, Architecture, Instrumentation and Testing]
- Fixed a bug where a change in the `appProtocol` for a Service did not trigger a load balancer update. ([#112785](https://github.com/kubernetes/kubernetes/pull/112785), [@MartinForReal](https://github.com/MartinForReal)) [SIG Cloud Provider and Network]
- Fixed a bug where the kubelet chooses the wrong container by its name when running `kubectl exec`. ([#113041](https://github.com/kubernetes/kubernetes/pull/113041), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Fixed an issue where the APIServer would panic on startup if an egress selector without a controlplane configuration is specified when using APIServerTracing ([#112979](https://github.com/kubernetes/kubernetes/pull/112979), [@dashpole](https://github.com/dashpole)) [SIG API Machinery, Instrumentation and Testing]
- Fixed: #22422 Admission controllers can cause unnecessary significant load on apiserver ([#112696](https://github.com/kubernetes/kubernetes/pull/112696), [@aimuz](https://github.com/aimuz)) [SIG Scalability]
- Kube-apiserver: DELETECOLLECTION API requests are now recorded in metrics with the correct verb. ([#113133](https://github.com/kubernetes/kubernetes/pull/113133), [@sxllwx](https://github.com/sxllwx)) [SIG API Machinery]
- Kube-apiserver: custom resources can now be specified in the --encryption-provider-config file and can be encrypted in etcd ([#113015](https://github.com/kubernetes/kubernetes/pull/113015), [@ritazh](https://github.com/ritazh)) [SIG API Machinery, Auth and Testing]
- Kube-proxy, will restart in case it detects that the Node assigned pod.Spec.PodCIDRs have changed ([#111344](https://github.com/kubernetes/kubernetes/pull/111344), [@aojea](https://github.com/aojea)) [SIG Network]
- Kubectl now escapes terminal special characters in output. This fixes CVE-2021-25743. ([#112553](https://github.com/kubernetes/kubernetes/pull/112553), [@dgl](https://github.com/dgl)) [SIG CLI and Security]
- Kubectl: fixed a bug where `kubectl convert` did not pick the right API version ([#112700](https://github.com/kubernetes/kubernetes/pull/112700), [@SataQiu](https://github.com/SataQiu)) [SIG CLI]
- Kubelet: Fixes a startup crash in devicemanager ([#113021](https://github.com/kubernetes/kubernetes/pull/113021), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Nested mountpoints are now group correctly on all cases. ([#112571](https://github.com/kubernetes/kubernetes/pull/112571), [@claudiubelu](https://github.com/claudiubelu)) [SIG Storage and Windows]
- The metrics(time duration) of a failed or unschedulable scheduling attempt will be longer for
  it will include the duration time of the unreserve operation now. ([#113113](https://github.com/kubernetes/kubernetes/pull/113113), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- Updates golang.org/x/text to v0.3.8 to fix CVE-2022-32149 ([#112989](https://github.com/kubernetes/kubernetes/pull/112989), [@ameukam](https://github.com/ameukam)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node and Storage]
- Use SSA to add pod disruption conditions by scheduler and controller-manager ([#113304](https://github.com/kubernetes/kubernetes/pull/113304), [@mimowo](https://github.com/mimowo)) [SIG API Machinery, Apps and Scheduling]

### Other (Cleanup or Flake)

- Kubeadm: remove the UnversionedKubeletConfigMap feature gate. The feature has been GA and locked to enabled since 1.25. ([#113448](https://github.com/kubernetes/kubernetes/pull/113448), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Removing Windows Server, Version 20H2 flavors from various container images ([#112924](https://github.com/kubernetes/kubernetes/pull/112924), [@marosset](https://github.com/marosset)) [SIG Testing and Windows]
- Service session affinity timeout tests are no longer required for Kubernetes network plugin conformance due to variations in existing implementations. New conformance tests will be developed to better express conformance in future releases. ([#112806](https://github.com/kubernetes/kubernetes/pull/112806), [@dcbw](https://github.com/dcbw)) [SIG Architecture, Network and Testing]
- The e2e.test binary no longer emits JSON structs to document progress. ([#113212](https://github.com/kubernetes/kubernetes/pull/113212), [@pohly](https://github.com/pohly)) [SIG Testing]
- The metric `etcd_db_total_size_in_bytes` is renamed to `apiserver_storage_db_total_size_in_bytes`. ([#113310](https://github.com/kubernetes/kubernetes/pull/113310), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery]

## Dependencies

### Added
- cloud.google.com/go/datastore: v1.1.0
- cloud.google.com/go/firestore: v1.1.0
- cloud.google.com/go/pubsub: v1.3.1
- dmitri.shuralyov.com/gpu/mtl: 666a987
- github.com/BurntSushi/xgb: [27f1227](https://github.com/BurntSushi/xgb/tree/27f1227)
- github.com/OneOfOne/xxhash: [v1.2.2](https://github.com/OneOfOne/xxhash/tree/v1.2.2)
- github.com/alecthomas/template: [fb15b89](https://github.com/alecthomas/template/tree/fb15b89)
- github.com/alecthomas/units: [f65c72e](https://github.com/alecthomas/units/tree/f65c72e)
- github.com/armon/consul-api: [eb2c6b5](https://github.com/armon/consul-api/tree/eb2c6b5)
- github.com/armon/go-metrics: [f0300d1](https://github.com/armon/go-metrics/tree/f0300d1)
- github.com/armon/go-radix: [7fddfc3](https://github.com/armon/go-radix/tree/7fddfc3)
- github.com/bgentry/speakeasy: [v0.1.0](https://github.com/bgentry/speakeasy/tree/v0.1.0)
- github.com/bketelsen/crypt: [5cbc8cc](https://github.com/bketelsen/crypt/tree/5cbc8cc)
- github.com/cespare/xxhash: [v1.1.0](https://github.com/cespare/xxhash/tree/v1.1.0)
- github.com/client9/misspell: [v0.3.4](https://github.com/client9/misspell/tree/v0.3.4)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/etcd: [v3.3.13+incompatible](https://github.com/coreos/etcd/tree/v3.3.13)
- github.com/coreos/go-systemd: [95778df](https://github.com/coreos/go-systemd/tree/95778df)
- github.com/coreos/pkg: [399ea9e](https://github.com/coreos/pkg/tree/399ea9e)
- github.com/dgrijalva/jwt-go: [v3.2.0+incompatible](https://github.com/dgrijalva/jwt-go/tree/v3.2.0)
- github.com/dgryski/go-sip13: [e10d5fe](https://github.com/dgryski/go-sip13/tree/e10d5fe)
- github.com/fatih/color: [v1.7.0](https://github.com/fatih/color/tree/v1.7.0)
- github.com/go-gl/glfw/v3.3/glfw: [6f7a984](https://github.com/go-gl/glfw/v3.3/glfw/tree/6f7a984)
- github.com/go-gl/glfw: [e6da0ac](https://github.com/go-gl/glfw/tree/e6da0ac)
- github.com/google/martian: [v2.1.0+incompatible](https://github.com/google/martian/tree/v2.1.0)
- github.com/gopherjs/gopherjs: [0766667](https://github.com/gopherjs/gopherjs/tree/0766667)
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
- github.com/hashicorp/golang-lru: [v0.5.1](https://github.com/hashicorp/golang-lru/tree/v0.5.1)
- github.com/hashicorp/hcl: [v1.0.0](https://github.com/hashicorp/hcl/tree/v1.0.0)
- github.com/hashicorp/logutils: [v1.0.0](https://github.com/hashicorp/logutils/tree/v1.0.0)
- github.com/hashicorp/mdns: [v1.0.0](https://github.com/hashicorp/mdns/tree/v1.0.0)
- github.com/hashicorp/memberlist: [v0.1.3](https://github.com/hashicorp/memberlist/tree/v0.1.3)
- github.com/hashicorp/serf: [v0.8.2](https://github.com/hashicorp/serf/tree/v0.8.2)
- github.com/jstemmer/go-junit-report: [v0.9.1](https://github.com/jstemmer/go-junit-report/tree/v0.9.1)
- github.com/jtolds/gls: [v4.20.0+incompatible](https://github.com/jtolds/gls/tree/v4.20.0)
- github.com/kr/logfmt: [b84e30a](https://github.com/kr/logfmt/tree/b84e30a)
- github.com/kr/pty: [v1.1.1](https://github.com/kr/pty/tree/v1.1.1)
- github.com/magiconair/properties: [v1.8.1](https://github.com/magiconair/properties/tree/v1.8.1)
- github.com/mattn/go-colorable: [v0.0.9](https://github.com/mattn/go-colorable/tree/v0.0.9)
- github.com/mattn/go-isatty: [v0.0.3](https://github.com/mattn/go-isatty/tree/v0.0.3)
- github.com/miekg/dns: [v1.0.14](https://github.com/miekg/dns/tree/v1.0.14)
- github.com/mitchellh/cli: [v1.0.0](https://github.com/mitchellh/cli/tree/v1.0.0)
- github.com/mitchellh/go-homedir: [v1.1.0](https://github.com/mitchellh/go-homedir/tree/v1.1.0)
- github.com/mitchellh/go-testing-interface: [v1.0.0](https://github.com/mitchellh/go-testing-interface/tree/v1.0.0)
- github.com/mitchellh/gox: [v0.4.0](https://github.com/mitchellh/gox/tree/v0.4.0)
- github.com/mitchellh/iochan: [v1.0.0](https://github.com/mitchellh/iochan/tree/v1.0.0)
- github.com/oklog/ulid: [v1.3.1](https://github.com/oklog/ulid/tree/v1.3.1)
- github.com/pascaldekloe/goe: [57f6aae](https://github.com/pascaldekloe/goe/tree/57f6aae)
- github.com/pelletier/go-toml: [v1.2.0](https://github.com/pelletier/go-toml/tree/v1.2.0)
- github.com/posener/complete: [v1.1.1](https://github.com/posener/complete/tree/v1.1.1)
- github.com/prometheus/tsdb: [v0.7.1](https://github.com/prometheus/tsdb/tree/v0.7.1)
- github.com/ryanuber/columnize: [9b3edd6](https://github.com/ryanuber/columnize/tree/9b3edd6)
- github.com/sean-/seed: [e2103e2](https://github.com/sean-/seed/tree/e2103e2)
- github.com/shurcooL/sanitized_anchor_name: [v1.0.0](https://github.com/shurcooL/sanitized_anchor_name/tree/v1.0.0)
- github.com/smartystreets/assertions: [b2de0cb](https://github.com/smartystreets/assertions/tree/b2de0cb)
- github.com/smartystreets/goconvey: [v1.6.4](https://github.com/smartystreets/goconvey/tree/v1.6.4)
- github.com/spaolacci/murmur3: [f09979e](https://github.com/spaolacci/murmur3/tree/f09979e)
- github.com/spf13/afero: [v1.2.2](https://github.com/spf13/afero/tree/v1.2.2)
- github.com/spf13/cast: [v1.3.0](https://github.com/spf13/cast/tree/v1.3.0)
- github.com/spf13/jwalterweatherman: [v1.0.0](https://github.com/spf13/jwalterweatherman/tree/v1.0.0)
- github.com/spf13/viper: [v1.7.0](https://github.com/spf13/viper/tree/v1.7.0)
- github.com/subosito/gotenv: [v1.2.0](https://github.com/subosito/gotenv/tree/v1.2.0)
- github.com/ugorji/go: [v1.1.4](https://github.com/ugorji/go/tree/v1.1.4)
- github.com/xordataexchange/crypt: [b2862e3](https://github.com/xordataexchange/crypt/tree/b2862e3)
- golang.org/x/exp: 6cc2880
- golang.org/x/image: cff245a
- golang.org/x/mobile: d2bd2a2
- gopkg.in/ini.v1: v1.51.0
- gopkg.in/resty.v1: v1.12.0
- rsc.io/binaryregexp: v0.2.0
- rsc.io/quote/v3: v3.1.0
- rsc.io/sampler: v1.3.0

### Changed
- github.com/aws/aws-sdk-go: [v1.38.49 → v1.44.116](https://github.com/aws/aws-sdk-go/compare/v1.38.49...v1.44.116)
- github.com/dnaeon/go-vcr: [v1.0.1 → v1.2.0](https://github.com/dnaeon/go-vcr/compare/v1.0.1...v1.2.0)
- github.com/fsnotify/fsnotify: [v1.5.4 → v1.6.0](https://github.com/fsnotify/fsnotify/compare/v1.5.4...v1.6.0)
- github.com/google/pprof: [94a9f03 → 4bb14d4](https://github.com/google/pprof/compare/94a9f03...4bb14d4)
- github.com/inconshreveable/mousetrap: [v1.0.0 → v1.0.1](https://github.com/inconshreveable/mousetrap/compare/v1.0.0...v1.0.1)
- github.com/konsorten/go-windows-terminal-sequences: [v1.0.2 → v1.0.3](https://github.com/konsorten/go-windows-terminal-sequences/compare/v1.0.2...v1.0.3)
- github.com/onsi/ginkgo/v2: [v2.2.0 → v2.4.0](https://github.com/onsi/ginkgo/v2/compare/v2.2.0...v2.4.0)
- github.com/onsi/gomega: [v1.20.1 → v1.23.0](https://github.com/onsi/gomega/compare/v1.20.1...v1.23.0)
- github.com/spf13/cobra: [v1.5.0 → v1.6.0](https://github.com/spf13/cobra/compare/v1.5.0...v1.6.0)
- golang.org/x/crypto: 7b82a4e → v0.1.0
- golang.org/x/lint: 1621716 → 6edffad
- golang.org/x/mod: 86c51ed → v0.6.0
- golang.org/x/net: a158d28 → c630100
- golang.org/x/sys: 8c9f86f → v0.1.0
- golang.org/x/term: 03fcf44 → v0.1.0
- golang.org/x/text: v0.3.7 → v0.4.0
- golang.org/x/tools: v0.1.12 → v0.2.0
- k8s.io/kube-openapi: 67bda5d → 172d655

### Removed
- github.com/getkin/kin-openapi: [v0.76.0](https://github.com/getkin/kin-openapi/tree/v0.76.0)



# v1.26.0-alpha.2


## Downloads for v1.26.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes.tar.gz) | ed5a0f1b0d45e3b6ea0f3d05f5aa4e924e8205a45b0238080e02a6ad60004f106f3f5442a302a44ae5b848ba7426e63e22a42806ddc1977214d964874165b6ca
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-src.tar.gz) | 7db2dcb528ead7a879aa12a525e488d1175ab55f5a710833bf80e3380a8db53fa5d35020b02d39a653465b7686fd4f6520a41218c7c55358d3de6ef54fa0b61a

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | c024438cb9ab879e6489413a969e891e2ba3216940e4ff6c8d5a79a6956312d9e6143a4b469b0decfd94358b60844c845f15426574673ea54460dc8f5ee7053f
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | ba9c72d75ae09c0fac0c29ff9034e3a94882f3bbcb1de1f8bcf7c65453048a4588980694b4d33af4ccd0458dfb9549fd8fd07a594dcc995743a8d154066ec09e
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 550a6348ee1d2ca9f2395855e740e4ef7efbba5b223088dae718fd6f9d5ae1cb6194a1e9cf123fee78414074eddda9ce534151a7e7ed4eaffa2d3f74830da623
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | 9ace57236b3581ebbb517cc1d730ffa0120fc5049c430dbf5b566786414e3a846a0db7edc9c6b7956fcc39267c3ef0dae3868f13fefafe971cd8fd6a8af946fa
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 6473fed98b9e55b50b600ea522d0a8b701ae483c1ae237014ccf2afbd7a6d85c3b2271d541eb1a3e897264ad455eedb6794c7cf48c07e62fa1778609794c293f
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 0ea3085811cd374473afb2cf7448ea018bbb3140c7f97e0d9b26f6fcf31c38734d0a08988b2d20427f476b69406f9835c0afa5196c95328941b56e51166405a9
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 6ca4a7bca557732e1f5b9cba044457e76c8d660b5ef6f29bb029f072edcd8f359600c4334ecbb2aa6a84c6c206188f51d1c3736a6ae36f1c860627ce0414bc5e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 39004468389c84931ca2a9c76e69949eb1d1f0a752293bb430568f5779852e1e889ecd920629b8358a651f787a770f6dff4839d4c4e02ae3b561e4ae8a68d7fa
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-windows-386.tar.gz) | ee407cd43b50ab75bf0b5fe2b270b357f5797a9e0af31b198262e72c0dd5ca978ccfe0848ea05841ef006334e645b32a52992d30219ccf8df4c8123083a10ddc
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | af44e5848f5290db7138cfeafdf42c72a34b9eca3bdbff7058c02f88fd0a242cd5ecf19c392b6ea4145bc2df8c194967d22164a739c8d17160b6d6d7a6a5f738
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | 2a2b664e7f98f02a81f6845e05e2ed1159381f3fb68fffd8edcbd2c36e942962260da980887b44e1cf150b25abb5e69b9519be7c9cb9e5b3a5e7ef8af7523654

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | 7700f4f8c2bd7944698c455b4bb3e40d7e877d194fb20baea4956fb76a97b21dc10731c7280c4b5ec8824fde4dad19e6845f8abb1c5f651d736bdb045bac0a90
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | ae796dde4c3316a6b8c429107fd782a6a3c038d7d99770c8060f2aa350f44f9805d1d2e567885106840f8ac684258b3247271ca6e807cd72a65299b1c697677a
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 5e32de6a1a768b8e9b56bdf6af8462bb0a70c5a8dd3c1dc4c1f8e19023b59dc33375b3166db66690468631202e781f2b9b995a76e98f6a14fb481e330c9fb5f2
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | feb17b943812dc4f3aca7d7c0853d9e86509a5896952146abc210f07b832b8de0eaebb833e62f5ca9cc2673e77c28f76af7dd9d822b1504ec4753347aa912c91
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | ce497964a7aad6b9bc0a95f1214c706e83c01c5d828818cef685c0df4eb3187018bcbb47b57a62cbc7da7c4d8c32f9c423eb7d07dd68e5f55afa09345ebaf9ce

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 545617a9b0c3e410a4c98547e1c9a208f41c0217b542fcc12103a0e077d56888d20e36d1b4a86ecd1bea8765b1f71ed08b1210e6fca688101b087a6bc42a8203
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 13fc8a993cc6ad5c5d260f559dbe613b8aa853f216caf56357610b77500c8a4946586e0c283021dc46e3b7f1272779a913d9ef050a7c7f8273c04abcae009c5c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 66a7911433f57cdf1d28115f0067dad2ed7987d4ea611110b52d1af055df5f589542601481586de8c7f5f807e9ffe5326048cd9f21c131cea7a7960e34203d32
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 06967565fed29fcc299c88392666256edb6f4e5783bda8d7f798a501d53a56001bc3ec05d79b09d2e0dae2281374a853cf883bccbb37c34a7a48ec650c3626bb
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | cf69a140ac23ad5af38b49b3125edc2f8102420ac2183a8a4bd3be855cbafdae48e25c1103d24fe8e0d7e01eb9b6d98b47383440ec7f5dde7b9e38580c1af1df
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.26.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 3454248b6393437372a44b2ce2dbe71eaae61d2b2ecec056bce49fec4670a9ce45255a4139afef32e73a84f7c9383e77b6d68883474bf169ed914d7282803547

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[k8s.gcr.io/conformance:v1.26.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/conformance-s390x)
[k8s.gcr.io/kube-apiserver:v1.26.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-apiserver-s390x)
[k8s.gcr.io/kube-controller-manager:v1.26.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-controller-manager-s390x)
[k8s.gcr.io/kube-proxy:v1.26.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-proxy-s390x)
[k8s.gcr.io/kube-scheduler:v1.26.0-alpha.2](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler) | [amd64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-amd64), [arm](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm), [arm64](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/gcr/images/k8s-artifacts-prod/us/kube-scheduler-s390x)

## Changelog since v1.26.0-alpha.1

## Changes by Kind

### Deprecation

- Kube-apiserver: the unused '--master-service-namespace' flag is deprecated and will be removed in v1.27. ([#112797](https://github.com/kubernetes/kubernetes/pull/112797), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery]

### API Change

- Add `kubernetes_feature_enabled` metric series to track whether each active feature gate is enabled. ([#112690](https://github.com/kubernetes/kubernetes/pull/112690), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery, Architecture, Cluster Lifecycle, Instrumentation, Network, Node and Scheduling]
- Introduce v1beta3 for Priority and Fairness with the following changes to the API spec:
  - rename 'assuredConcurrencyShares' (located under spec.limited') to 'nominalConcurrencyShares'
  - apply strategic merge patch annotations to 'Conditions' of flowschemas and prioritylevelconfigurations ([#112306](https://github.com/kubernetes/kubernetes/pull/112306), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Testing]
- Legacy klog flags are no longer available. Only `-v` and `-vmodule` are still supported. ([#112120](https://github.com/kubernetes/kubernetes/pull/112120), [@pohly](https://github.com/pohly)) [SIG Architecture, CLI, Instrumentation, Node and Testing]
- The feature gates ServiceLoadBalancerClass and ServiceLBNodePortControl have been removed. These feature gates were enabled (and locked) since v1.24. ([#112577](https://github.com/kubernetes/kubernetes/pull/112577), [@andrewsykim](https://github.com/andrewsykim)) [SIG Apps]

### Feature

- A new --disable-compression flag has been added to kubectl (default = false). When true, it opts out of response compression for all requests to the apiserver. This can help improve list call latencies significantly when client-server network bandwidth is ample (>30MB/s) or if the server is CPU-constrained. ([#112580](https://github.com/kubernetes/kubernetes/pull/112580), [@shyamjvs](https://github.com/shyamjvs)) [SIG CLI and Testing]
- A new `pod_status_sync_duration_seconds` histogram is reported at alpha metrics stability that estimates how long the Kubelet takes to write a pod status change once it is detected. ([#107896](https://github.com/kubernetes/kubernetes/pull/107896), [@smarterclayton](https://github.com/smarterclayton)) [SIG Apps, Architecture, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Added a new feature gate `CELValidatingAdmission` to enable expression validation for Admission Control. ([#112792](https://github.com/kubernetes/kubernetes/pull/112792), [@cici37](https://github.com/cici37)) [SIG API Machinery]
- Added validation for the --container-runtime-endpoint flag of kubelet to be non-empty. ([#112542](https://github.com/kubernetes/kubernetes/pull/112542), [@astraw99](https://github.com/astraw99)) [SIG Node]
- Expose health check SLI metrics on "metrics/slis" for apiserver ([#112741](https://github.com/kubernetes/kubernetes/pull/112741), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery, Architecture, Auth and Instrumentation]
- Kubeadm: sub-phases are now able to support the dry-run mode, e.g. kubeadm reset phase cleanup-node --dry-run ([#112945](https://github.com/kubernetes/kubernetes/pull/112945), [@chendave](https://github.com/chendave)) [SIG Cluster Lifecycle]
- Kubeadm: support image repository format validation ([#112732](https://github.com/kubernetes/kubernetes/pull/112732), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Go 1.19.2 ([#112900](https://github.com/kubernetes/kubernetes/pull/112900), [@xmudrii](https://github.com/xmudrii)) [SIG Release and Testing]
- Switch kubectl to use `github.com/russross/blackfriday/v2` ([#112731](https://github.com/kubernetes/kubernetes/pull/112731), [@pacoxu](https://github.com/pacoxu)) [SIG CLI]
- `registered_metric_total` now reports the number of metrics broken down by stability level and deprecated version ([#112907](https://github.com/kubernetes/kubernetes/pull/112907), [@logicalhan](https://github.com/logicalhan)) [SIG Architecture and Instrumentation]

### Bug or Regression

- Consider only plugin directory and not entire kubelet root when cleaning up mounts ([#112607](https://github.com/kubernetes/kubernetes/pull/112607), [@mattcary](https://github.com/mattcary)) [SIG Storage]
- Fix that pods running on nodes tainted with NoExecute continue to run when the PodDisruptionConditions feature gate is enabled ([#112518](https://github.com/kubernetes/kubernetes/pull/112518), [@mimowo](https://github.com/mimowo)) [SIG Apps and Auth]
- Fixes an issue in winkernel proxier that causes proxy rules to leak anytime service backends are modified. ([#112837](https://github.com/kubernetes/kubernetes/pull/112837), [@daschott](https://github.com/daschott)) [SIG Network and Windows]
- Kube-apiserver: redirects from backend API servers are no longer followed when checking availability with requests to `/apis/$group/$version` ([#112772](https://github.com/kubernetes/kubernetes/pull/112772), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Testing]
- Kubeadm: fix a bug when performing validation on ClusterConfiguration networking fields ([#112751](https://github.com/kubernetes/kubernetes/pull/112751), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet now cleans up the Node's cloud node IP annotation correctly if you
  stop using `--node-ip`. (In particular, this fixes the problem where people who
  were unnecessarily using `--node-ip` with an external cloud provider in 1.23,
  and then running into problems with 1.24, could not fix the problem by just
  removing the unnecessary `--node-ip` from the kubelet arguments, because
  that wouldn't remove the annotation that caused the problems.) ([#112184](https://github.com/kubernetes/kubernetes/pull/112184), [@danwinship](https://github.com/danwinship)) [SIG Network and Node]
- Kubelet: Fix log spam from kubelet_getters.go "Path does not exist" ([#112650](https://github.com/kubernetes/kubernetes/pull/112650), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Kubelet: when there are multi option lines in /etc/resolv.conf, merge all options into one line in a pod with the `Default` DNS policy. ([#112414](https://github.com/kubernetes/kubernetes/pull/112414), [@pacoxu](https://github.com/pacoxu)) [SIG Network and Node]
- The pod admission error message was improved for usability. ([#112644](https://github.com/kubernetes/kubernetes/pull/112644), [@vitorfhc](https://github.com/vitorfhc)) [SIG Node]

### Other (Cleanup or Flake)

- Adds a kubernetes_feature_enabled metric which will tell you if a feature is enabled. ([#112652](https://github.com/kubernetes/kubernetes/pull/112652), [@logicalhan](https://github.com/logicalhan)) [SIG Architecture and Instrumentation]
- Introduce `ComponentSLIs` alpha feature-gate for component SLIs metrics endpoint. ([#112884](https://github.com/kubernetes/kubernetes/pull/112884), [@logicalhan](https://github.com/logicalhan)) [SIG API Machinery]
- Lock ServerSideApply feature gate to true with the feature already being GA. ([#112748](https://github.com/kubernetes/kubernetes/pull/112748), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery, Apps, Instrumentation and Testing]
- PodOverhead feature gate was removed as the feature is in GA since 1.24 ([#112579](https://github.com/kubernetes/kubernetes/pull/112579), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG Node and Scheduling]
- Reworded log message upon image garbage collection failure to be more clear. ([#112631](https://github.com/kubernetes/kubernetes/pull/112631), [@tzneal](https://github.com/tzneal)) [SIG Node]
- The IndexedJob and SuspendJob feature gates that graduated to GA in 1.24 and were unconditionally enabled have been removed in v1.26 ([#112589](https://github.com/kubernetes/kubernetes/pull/112589), [@SataQiu](https://github.com/SataQiu)) [SIG Apps]
- The test/e2e/framework was refactored so that the core framework is smaller. Optional functionality like resource monitoring, log size monitoring, metrics gathering and debug information dumping must be imported by specific e2e test suites. Init packages are provided which can be imported to re-enable the functionality that traditionally was in the core framework. If you have code that no longer compiles because of this PR, you can use the script [from a commit message](https://github.com/kubernetes/kubernetes/pull/112043/commits/dfdf88d4faafa6fd39988832ea0ef6d668f490e9) to update that code. ([#112043](https://github.com/kubernetes/kubernetes/pull/112043), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Scheduling, Storage, Testing and Windows]
- Updated cri-tools to [v1.25.0(https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.25.0) ([#112058](https://github.com/kubernetes/kubernetes/pull/112058), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Release]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/fsnotify/fsnotify: [v1.4.9 → v1.5.4](https://github.com/fsnotify/fsnotify/compare/v1.4.9...v1.5.4)
- github.com/go-openapi/jsonreference: [v0.19.5 → v0.20.0](https://github.com/go-openapi/jsonreference/compare/v0.19.5...v0.20.0)
- github.com/matttproud/golang_protobuf_extensions: [v1.0.1 → v1.0.2](https://github.com/matttproud/golang_protobuf_extensions/compare/v1.0.1...v1.0.2)
- go.uber.org/goleak: v1.1.12 → v1.2.0
- k8s.io/utils: ee6ede2 → 665eaae
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.32 → v0.0.33
- sigs.k8s.io/yaml: v1.2.0 → v1.3.0

### Removed
- dmitri.shuralyov.com/gpu/mtl: 28db891
- github.com/BurntSushi/xgb: [27f1227](https://github.com/BurntSushi/xgb/tree/27f1227)
- github.com/ajstarks/svgo: [644b8db](https://github.com/ajstarks/svgo/tree/644b8db)
- github.com/fogleman/gg: [0403632](https://github.com/fogleman/gg/tree/0403632)
- github.com/go-gl/glfw/v3.3/glfw: [6f7a984](https://github.com/go-gl/glfw/v3.3/glfw/tree/6f7a984)
- github.com/golang/freetype: [e2365df](https://github.com/golang/freetype/tree/e2365df)
- github.com/jung-kurt/gofpdf: [24315ac](https://github.com/jung-kurt/gofpdf/tree/24315ac)
- github.com/kr/fs: [v0.1.0](https://github.com/kr/fs/tree/v0.1.0)
- github.com/mvdan/xurls: [v1.1.0](https://github.com/mvdan/xurls/tree/v1.1.0)
- github.com/pkg/sftp: [v1.10.1](https://github.com/pkg/sftp/tree/v1.10.1)
- github.com/remyoudompheng/bigfft: [52369c6](https://github.com/remyoudompheng/bigfft/tree/52369c6)
- github.com/russross/blackfriday: [v1.5.2](https://github.com/russross/blackfriday/tree/v1.5.2)
- github.com/spf13/afero: [v1.6.0](https://github.com/spf13/afero/tree/v1.6.0)
- golang.org/x/exp: 85be41e
- golang.org/x/image: cff245a
- golang.org/x/mobile: e6ae53a
- gonum.org/v1/gonum: v0.6.2
- gonum.org/v1/netlib: 7672324
- gonum.org/v1/plot: e2840ee
- modernc.org/cc: v1.0.0
- modernc.org/golex: v1.0.0
- modernc.org/mathutil: v1.0.0
- modernc.org/strutil: v1.0.0
- modernc.org/xc: v1.0.0
- rsc.io/pdf: v0.1.1



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
