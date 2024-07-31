<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.31.0-rc.0](#v1310-rc0)
  - [Downloads for v1.31.0-rc.0](#downloads-for-v1310-rc0)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.31.0-beta.0](#changelog-since-v1310-beta0)
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
- [v1.31.0-beta.0](#v1310-beta0)
  - [Downloads for v1.31.0-beta.0](#downloads-for-v1310-beta0)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
    - [Container Images](#container-images-1)
  - [Changelog since v1.31.0-alpha.3](#changelog-since-v1310-alpha3)
  - [Changes by Kind](#changes-by-kind-1)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-1)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.31.0-alpha.3](#v1310-alpha3)
  - [Downloads for v1.31.0-alpha.3](#downloads-for-v1310-alpha3)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
    - [Container Images](#container-images-2)
  - [Changelog since v1.31.0-alpha.2](#changelog-since-v1310-alpha2)
  - [Changes by Kind](#changes-by-kind-2)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Bug or Regression](#bug-or-regression-2)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.31.0-alpha.2](#v1310-alpha2)
  - [Downloads for v1.31.0-alpha.2](#downloads-for-v1310-alpha2)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
    - [Container Images](#container-images-3)
  - [Changelog since v1.31.0-alpha.1](#changelog-since-v1310-alpha1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-3)
    - [API Change](#api-change-3)
    - [Feature](#feature-3)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.31.0-alpha.1](#v1310-alpha1)
  - [Downloads for v1.31.0-alpha.1](#downloads-for-v1310-alpha1)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
    - [Container Images](#container-images-4)
  - [Changelog since v1.30.0](#changelog-since-v1300)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-4)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-4)
    - [Feature](#feature-4)
    - [Failing Test](#failing-test-2)
    - [Bug or Regression](#bug-or-regression-4)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)

<!-- END MUNGE: GENERATED_TOC -->

# v1.31.0-rc.0


## Downloads for v1.31.0-rc.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes.tar.gz) | 21cc56e80b1bdc02005351f82cf9ac140b6785ddbb50f2bc14109f8a8dd5b1de0004c5bae660f361333f949b46f3a8e012b517a2e8d21429d2bc4952eb1aae96
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-src.tar.gz) | b0817c03e5c060b94bfaa12c7ddcd9ed9146b468a21af71b70b1ec83ff9f20d584d3ee2c402a8324e045bf6b357b9f9846b54ab29c8a3ecade26880a8a2de193

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | 491f352be31bb3cfdbc2127c771aecd4f5959003af562fe9f413ff57535a50e27ff5240067d2bf7117ce61edcea601b2f80b4d1443533e955e874c4a188a432f
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | 1415ebf19094ea907665d30bd5af8d3885c203c6c9c31229804762f52149ef793cb7872499cb37baced9f922e6e10167ca9bf13d5729e6adde890d1bc5039736
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-linux-386.tar.gz) | ced0745e2c5c958370eb4e1f2d1dd33efae13df348f189c75c64e18499d0781df6fde8c730e68703758802c33c2f4db118a69584a2666614f1bf0e1b7634ed73
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | d80c333b4a85c8d4975445ec6fa86ca4c1c8625dc11d807dd4b7460106931b891c05739ee31b6ccdf0648aefa12de00bffb6dc511b8f5eeef747c20d73613e82
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-linux-arm.tar.gz) | a40f91682b349a488687cf80795b40db923e7e6ca35265d531e73cb17a263d20f3418b7b6214a4d2e4816f7381e35d8938ea8d55e5fb8d52e6873eb3820a56f7
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | 746e31291d679e93d68e618dd4d371a9b9ba3492a4df545ea08eb70a05d32dbe8451f4c6ce8c35a1484fc1edeb4d19c0119c1dc0ed50326edae2247291be8a55
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | 9347f378624df1f709b6390e22792b9cc743dc5e29ce9b0ef0487f58af5592b55c1c8ad92af22969feff23379712a8f3d50511fa1baccdc5826916d07ef81ffb
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | dc7b1f3c0f1f128aa503debeaaf93d692bc85a57bfc3d1cb771b786c0ea8fb3d5c56e7bed77258ce70d2763b5bc23e7564a05a031776890abf69c36de5cd2430
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-windows-386.tar.gz) | b5262ed3cb3d3d645c9fc4b5040d4cd77ce2337c2a466b8ea9a76988ec35867b9059a123740df87051055b0e89ec1d91e89851f0659fd2692d840cede007b0c7
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | 8560cdf5501d4b12ed766041c6170479b6f33c12c69fe1ade2687b65c5f02737570125286eca32fe327ff068e34b1b45d4fef7acde9e080515e62d5dad648723
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | b821fb80d384be4f37e4d3303b364ab29243e078a6665b970723f6b1be92ba60ce8316e94a453a56b1c0229ce1ecb3f14d16ba56c2641883523645edc27b42f8

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | 782c376c100cd482adefd1cc030d4de56249c987eba951797f0a6afe70703085b67fc8e0d07c5cf895d200e35039f2c988c4b65430dcb291979e06f4310d22dc
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | 15a9805ce071e6e86987e027f8b27e94c0bbaea423bb5f690c0801403a043ca36fe62ba6e27595c5874d0fef1ebb61029e4c0279f92d8f9959f7e1243d76e726
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | 2eaf285b8aff497dbff4196dc6c316d9283ebed1cc01ddae8392ee2272cfd03a1c92f25d50797eb446111e3027032ac4ee90c15ac352d48990815064114392c5
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | a20a8e3b5bc8ea80634fa3b0df3d63b0da57254ef43eb4ac5459cd8f7d673931d7ec6664bd9359277325a1b9541e69606c611ccfa269582fb535d46810b0f540

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | 58a6fc3ab4440a9b6c9968fb789ec3cdbd450ed58676aeaa6c336ce2d3dd6c44fc9080d84f6e70de10552066efe3a89f318e6944ee3aa1a67f8673688b96274c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | cf88294e9a6ab61ada2c7af81f9db2322312f39f4d1ab26f497a915321797a345667968d863024c997ef925de9a31ef0d3bc7be9d032283441bdc1c7c3b12d6c
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | e2480f1d518bcd6ebe0a3daf19148f8135bfc9d14a39b7e28e6d4104e026b7778cd3aa2fd2be103d081474437353b976d9dcbda67174dbfbd11200595e39b88e
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | 30e3a0479974413cadb7929941cb8ad14ae8b0ba280d35da16e5c115428629e60b00f5c9f515ef1de0a51323f50e61617b6cdecd5ef9c352aab18add02b89cbf
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | f163c968132b9d4301b48d09ae1751bc2b76ba56db9eb3de766674059271458a2fd04f78112f655d9fc1a64999d1dc001c3d450cbf83ef4324365cbde2746ed2

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.31.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.31.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.31.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.31.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.31.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.31.0-rc.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.31.0-beta.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Reduce state change noise when volume expansion fails. Also mark certain failures as infeasible.
  
  If you are using the RecoverVolumeExpansionFailure alpha feature, after upgrading to this release, existing PVCs with status.allocatedResourceStatus set to "ControllerResizeFailed" or "NodeResizeFailed" should have their status.allocatedResourceStatus cleared. ([#126108](https://github.com/kubernetes/kubernetes/pull/126108), [@gnufied](https://github.com/gnufied)) [SIG Apps, Auth, Node, Storage and Testing]
 
## Changes by Kind

### Deprecation

- Added a warning when creating or updating a PV with the deprecated annotation `volume.beta.kubernetes.io/mount-options` ([#124819](https://github.com/kubernetes/kubernetes/pull/124819), [@carlory](https://github.com/carlory)) [SIG Storage]

### API Change

- Add Coordinated Leader Election as alpha under the CoordinatedLeaderElection feature gate. With the feature enabled, the control plane can use LeaseCandidate objects (coordination.k8s.io/v1alpha1 API group) to participate in a leader election and let the kube-apiserver select the best instance according to some strategy. ([#124012](https://github.com/kubernetes/kubernetes/pull/124012), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery, Apps, Auth, Cloud Provider, Etcd, Node, Release, Scheduling and Testing]
- Add an AllocatedResourcesStatus to each container status to indicate the health status of devices exposed by the device plugin. ([#126243](https://github.com/kubernetes/kubernetes/pull/126243), [@SergeyKanzhelev](https://github.com/SergeyKanzhelev)) [SIG API Machinery, Apps, Node and Testing]
- Added Node.Status.Features.SupplementalGroupsPolicy field which is set to true when the feature is implemented in the CRI implementation (KEP-3619) ([#125470](https://github.com/kubernetes/kubernetes/pull/125470), [@everpeace](https://github.com/everpeace)) [SIG API Machinery, Apps, Node and Testing]
- CustomResourceDefinition objects created with non-empty `caBundle` fields which are invalid or do not contain any certificates will not appear in discovery or serve endpoints until a valid `caBundle` is provided. Updates to CustomResourceDefinition are no longer allowed to transition a valid `caBundle` field to an invalid `caBundle` field. ([#124061](https://github.com/kubernetes/kubernetes/pull/124061), [@Jefftree](https://github.com/Jefftree)) [SIG API Machinery]
- DRA: The DRA driver's daemonset must be deployed with a service account that enables writing ResourceSlice and reading ResourceClaim objects. ([#125163](https://github.com/kubernetes/kubernetes/pull/125163), [@pohly](https://github.com/pohly)) [SIG Auth, Node and Testing]
- DRA: new API and several new features ([#125488](https://github.com/kubernetes/kubernetes/pull/125488), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Cluster Lifecycle, Etcd, Node, Release, Scheduling, Storage and Testing]
- DRA: the number of ResourceClaim objects can be limited per namespace and by the number of devices requested through a specific class via the v1.ResourceQuota mechanism. ([#120611](https://github.com/kubernetes/kubernetes/pull/120611), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, CLI, Etcd, Node, Release, Scheduling and Testing]
- Fix the documentation for the default value of the procMount entry in the pod securityContext.
  The documentation was previously using the name of the internal variable 'DefaultProcMount' rather than the actual value 'Default'. ([#125782](https://github.com/kubernetes/kubernetes/pull/125782), [@aborrero](https://github.com/aborrero)) [SIG Apps and Node]
- Fixed a bug in the API server where empty collections of ValidatingAdmissionPolicies did not have an `items` field. ([#124568](https://github.com/kubernetes/kubernetes/pull/124568), [@xyz-li](https://github.com/xyz-li)) [SIG API Machinery]
- Graduate the Job SuccessPolicy to Beta.
  
  The new reason label, "SuccessPolicy" and "CompletionsReached" are added to the "jobs_finished_total" metric.
  Additionally, If we enable the "JobSuccessPolicy" feature gate, the Job gets "CompletionsReached" reason for the "SuccessCriteriaMet" and "Complete" condition type
  when the number of succeeded Job Pods (".status.succeeded") reached the desired completions (".spec.completions"). ([#126067](https://github.com/kubernetes/kubernetes/pull/126067), [@tenzen-y](https://github.com/tenzen-y)) [SIG API Machinery, Apps and Testing]
- Introduce a new boolean kubelet flag --fail-cgroupv1 ([#126031](https://github.com/kubernetes/kubernetes/pull/126031), [@harche](https://github.com/harche)) [SIG API Machinery and Node]
- Kube-apiserver: adds an alpha AuthorizeWithSelectors feature that includes field and label selector information from requests in webhook authorization calls; adds an alpha AuthorizeNodeWithSelectors feature that makes the node authorizer limit requests from node API clients to get / list / watch its own Node API object, and to get / list / watch its own Pod API objects. Clients using kubelet credentials to read other nodes or unrelated pods must change their authentication credentials (recommended), adjust their usage, or grant broader read access independent of the node authorizer. ([#125571](https://github.com/kubernetes/kubernetes/pull/125571), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, Node, Scheduling and Testing]
- Kube-proxy Windows service control manager integration(--windows-service) is now configurable in v1alpha1 component configuration via `WindowsRunAsService` field ([#126072](https://github.com/kubernetes/kubernetes/pull/126072), [@aroradaman](https://github.com/aroradaman)) [SIG Network and Scalability]
- Promote LocalStorageCapacityIsolation to beta and enable if user namespace is enabled for the pod ([#126014](https://github.com/kubernetes/kubernetes/pull/126014), [@PannagaRao](https://github.com/PannagaRao)) [SIG Apps, Autoscaling, Node, Storage and Testing]
- Promote StatefulSetStartOrdinal to stable. This means `--feature-gates=StatefulSetStartOrdinal=true` are not needed on kube-apiserver and kube-controller-manager binaries and they'll be removed soon following policy at https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecation ([#125374](https://github.com/kubernetes/kubernetes/pull/125374), [@pwschuurman](https://github.com/pwschuurman)) [SIG API Machinery, Apps and Testing]
- Promoted feature-gate `VolumeAttributesClass` to beta (disabled by default). Users need to enable the feature gate and the storage v1beta1 group to use this new feature.
  - Promoted API `VolumeAttributesClass` and `VolumeAttributesClassList` to `storage.k8s.io/v1beta1`. ([#126145](https://github.com/kubernetes/kubernetes/pull/126145), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, CLI, Etcd, Storage and Testing]
- Removed feature gate `CustomResourceValidationExpressions`. ([#126136](https://github.com/kubernetes/kubernetes/pull/126136), [@cici37](https://github.com/cici37)) [SIG API Machinery, Cloud Provider and Testing]
- Revert "Move ConsistentListFromCache feature flag to Beta and enable it by default" ([#126139](https://github.com/kubernetes/kubernetes/pull/126139), [@enj](https://github.com/enj)) [SIG API Machinery]
- Revised the Pod API with alpha support for volumes derived from OCI artefacts.
  This feature is behind the `ImageVolume` feature gate. ([#125660](https://github.com/kubernetes/kubernetes/pull/125660), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery, Apps and Node]
- The Ingress.spec.defaultBackend is now considered an atomic struct for the purposes of server-side-apply.  This means that any field-owner who sets values in that struct (they are mutually exclusive) owns the whole struct.  For almost all users this change has no impact.  For controllers which want to change port from number to name (or vice-versa), this makes it easier. ([#126207](https://github.com/kubernetes/kubernetes/pull/126207), [@thockin](https://github.com/thockin)) [SIG API Machinery]
- To enhance usability and developer experience, CRD validation rules now support direct use of (CEL) reserved keywords as field names in object validation expressions for existing expressions in storage, will fully support runtime in next release for compatibility concern. ([#126188](https://github.com/kubernetes/kubernetes/pull/126188), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]

### Feature

- ACTION REQUIRED for custom scheduler plugin developers:
  `EventsToRegister` in the `EnqueueExtensions` interface gets `ctx` in the parameters and `error` in the return values.
  Please change your plugins' implementation accordingly. ([#126113](https://github.com/kubernetes/kubernetes/pull/126113), [@googs1025](https://github.com/googs1025)) [SIG Node, Scheduling, Storage and Testing]
- Added `storage_class` and `volume_attributes_class` labels to `pv_collector_bound_pvc_count` and `pv_collector_unbound_pvc_count` metrics. ([#126166](https://github.com/kubernetes/kubernetes/pull/126166), [@AndrewSirenko](https://github.com/AndrewSirenko)) [SIG Apps, Instrumentation, Storage and Testing]
- Changed Linux swap handling to restrict access to swap for containers in high priority Pods.
  New Pods that have a node- or cluster-critical priority are prohibited from accessing swap on Linux,
  even if your cluster and node configuration could otherwise allow this. ([#125277](https://github.com/kubernetes/kubernetes/pull/125277), [@iholder101](https://github.com/iholder101)) [SIG Node and Testing]
- Fixed a missing behavior where Windows nodes did not implement memory-pressure eviction. ([#122922](https://github.com/kubernetes/kubernetes/pull/122922), [@marosset](https://github.com/marosset)) [SIG Node, Testing and Windows]
- Graduate Kubernetes' support for AppArmor to GA. ([#125257](https://github.com/kubernetes/kubernetes/pull/125257), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Apps, Node and Testing]
- If the feature-gate VolumeAttributesClass is enabled, when finding a suitable persistent volume for a claim, the kube-controller-manager will be aware of the `volumeAttributesClassName` field of PVC and PV objects. The `volumeAttributesClassName` field is a reference to a VolumeAttributesClass object, which contains a set of key-value pairs that present mutable attributes of the volume. It's forbidden to change the `volumeAttributesClassName` field of a PVC object until the PVC is bound to a PV object. During the binding process, if a PVC has a `volumeAttributesClassName` field set, the controller will only consider volumes that have the same `volumeAttributesClassName` as the PVC. If the `volumeAttributesClassName` field is not set or set to an empty string, only volumes with empty `volumeAttributesClassName` will be considered. ([#121902](https://github.com/kubernetes/kubernetes/pull/121902), [@carlory](https://github.com/carlory)) [SIG Apps, Scheduling, Storage and Testing]
- Implement `event_handling_duration_seconds` metric, which is the time the scheduler takes to handle each kind of events. ([#125929](https://github.com/kubernetes/kubernetes/pull/125929), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Implement `queueing_hint_execution_duration_seconds` metric, which is the time the QueueingHint function takes. ([#126227](https://github.com/kubernetes/kubernetes/pull/126227), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Implement new cluster events UpdatePodScaleDown and UpdatePodLabel for scheduler plugins. ([#122628](https://github.com/kubernetes/kubernetes/pull/122628), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Kube-apiserver: when the alpha `UserNamespacesPodSecurityStandards` feature gate is enabled, Pod Security Admission enforcement of the baseline policy now allows `procMount=Unmasked` for user namespace pods that set `hostUsers=false`. ([#126163](https://github.com/kubernetes/kubernetes/pull/126163), [@haircommander](https://github.com/haircommander)) [SIG Auth]
- Kube-scheduler implements scheduling hints for the VolumeBinding plugin.
  The scheduling hints allow the scheduler to retry scheduling a Pod that was previously rejected by the VolumeBinding plugin only if a new resource referenced by the plugin was created or an existing resource referenced by the plugin was updated. ([#124958](https://github.com/kubernetes/kubernetes/pull/124958), [@bells17](https://github.com/bells17)) [SIG Scheduling and Storage]
- Kube-scheduler implements scheduling hints for the VolumeBinding plugin.
  The scheduling hints allow the scheduler to retry scheduling a Pod that was previously rejected by the VolumeBinding plugin only if a new resource referenced by the plugin was created or an existing resource referenced by the plugin was updated. ([#124959](https://github.com/kubernetes/kubernetes/pull/124959), [@bells17](https://github.com/bells17)) [SIG Scheduling and Storage]
- Kube-scheduler implements scheduling hints for the VolumeBinding plugin.
  The scheduling hints allow the scheduler to retry scheduling a Pod that was previously rejected by the VolumeBinding plugin only if a new resource referenced by the plugin was created or an existing resource referenced by the plugin was updated. ([#124961](https://github.com/kubernetes/kubernetes/pull/124961), [@bells17](https://github.com/bells17)) [SIG Scheduling and Storage]
- Kubelet now requests serving certificates only once it has at least one IP address in the `.status.addresses` of its associated Node object. This avoids requesting DNS-only serving certificates before externally set addresses are in place. Until 1.33, the previous behavior can be opted back into by setting the deprecated AllowDNSOnlyNodeCSR feature gate to true in the kubelet. ([#125813](https://github.com/kubernetes/kubernetes/pull/125813), [@aojea](https://github.com/aojea)) [SIG Auth, Cloud Provider and Node]
- Kubelet/stats: set INFO log level for stats not found in cadvisor memory cache error ([#125656](https://github.com/kubernetes/kubernetes/pull/125656), [@gyuho](https://github.com/gyuho)) [SIG Node]
- Kubernetes is now built with go 1.23rc2 ([#126047](https://github.com/kubernetes/kubernetes/pull/126047), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Promote KEP-4191 "Split Image Filesystem" to Beta. ([#126205](https://github.com/kubernetes/kubernetes/pull/126205), [@kwilczynski](https://github.com/kwilczynski)) [SIG Node]
- Promote ProcMountType feature to Beta ([#125259](https://github.com/kubernetes/kubernetes/pull/125259), [@sohankunkerkar](https://github.com/sohankunkerkar)) [SIG Node]
- Promoted the metrics for both VAP and CRD validation rules to beta. ([#126237](https://github.com/kubernetes/kubernetes/pull/126237), [@cici37](https://github.com/cici37)) [SIG API Machinery and Instrumentation]
- Report an event to pod if kubelet does attach operation failed when kubelet is running with `--enable-controller-attach-detach=false` ([#124884](https://github.com/kubernetes/kubernetes/pull/124884), [@carlory](https://github.com/carlory)) [SIG Storage]
- Starting in 1.31, `container_engine_t` is in the list of allowed SELinux types in the baseline Pod Security Standards profile ([#126165](https://github.com/kubernetes/kubernetes/pull/126165), [@haircommander](https://github.com/haircommander)) [SIG Auth]
- The kube-proxy command line flag `--proxy-port-range`, which was previously deprecated and non-functional, has now been removed. ([#126293](https://github.com/kubernetes/kubernetes/pull/126293), [@aroradaman](https://github.com/aroradaman)) [SIG Network]

### Failing Test

- Fix bug in KEP-4191 if feature gate is turned on but container runtime is not configured. ([#126335](https://github.com/kubernetes/kubernetes/pull/126335), [@kannon92](https://github.com/kannon92)) [SIG Node]

### Bug or Regression

- Allow calling Stop multiple times on RetryWatcher without panicking ([#126125](https://github.com/kubernetes/kubernetes/pull/126125), [@mprahl](https://github.com/mprahl)) [SIG API Machinery]
- Fix a bug where the Kubelet didn't calculate the process usage of pods correctly, leading to pods never getting evicted for PID use. ([#124101](https://github.com/kubernetes/kubernetes/pull/124101), [@haircommander](https://github.com/haircommander)) [SIG Node and Testing]
- Fix fake clientset ApplyScale subresource from 'status' to 'scale' ([#126073](https://github.com/kubernetes/kubernetes/pull/126073), [@a7i](https://github.com/a7i)) [SIG API Machinery]
- Fix node report notReady with reason 'container runtime status check may not have completed yet' after Kubelet restart ([#124430](https://github.com/kubernetes/kubernetes/pull/124430), [@AllenXu93](https://github.com/AllenXu93)) [SIG Node]
- Fixed a bug in storage-version-migrator-controller that would cause migration attempts to fail if resources were deleted when the migration was in progress. ([#126107](https://github.com/kubernetes/kubernetes/pull/126107), [@enj](https://github.com/enj)) [SIG API Machinery, Apps, Auth and Testing]
- Fixed a bug that init containers with `Always` restartPolicy may not terminate gracefully if the pod hasn't initialized yet. ([#125935](https://github.com/kubernetes/kubernetes/pull/125935), [@gjkim42](https://github.com/gjkim42)) [SIG Node and Testing]
- Kube-apiserver: fixes a potential crash serving CustomResourceDefinitions that combine an invalid schema and CEL validation rules. ([#126167](https://github.com/kubernetes/kubernetes/pull/126167), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]
- Kubeadm: fixed a bug on 'kubeadm join' where using patches with a kubeletconfiguration target was not respected when performing the local kubelet healthz check. ([#126224](https://github.com/kubernetes/kubernetes/pull/126224), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Mount-utils: treat syscall.ENODEV as corrupted mount ([#126174](https://github.com/kubernetes/kubernetes/pull/126174), [@dobsonj](https://github.com/dobsonj)) [SIG Storage]
- Revert Graduates the `WatchList` feature gate to Beta for kube-apiserver and enables `WatchListClient` for KCM. ([#126191](https://github.com/kubernetes/kubernetes/pull/126191), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Set ProcMountType feature to disabled by default, to follow the lead of UserNamespacesSupport (which it relies on). ([#126291](https://github.com/kubernetes/kubernetes/pull/126291), [@haircommander](https://github.com/haircommander)) [SIG Node]

### Other (Cleanup or Flake)

- Clean deprecated context.StopCh in favor of ctx ([#125661](https://github.com/kubernetes/kubernetes/pull/125661), [@mjudeikis](https://github.com/mjudeikis)) [SIG API Machinery]
- Finish initial generic controlplane refactor of kube-apiserver, providing a sample binariy building a kube-like controlplane without contrainer orchestration resources. ([#124530](https://github.com/kubernetes/kubernetes/pull/124530), [@sttts](https://github.com/sttts)) [SIG API Machinery, Apps, Cloud Provider, Network, Node and Testing]
- Kubernetes is now built with go 1.22.5 ([#126330](https://github.com/kubernetes/kubernetes/pull/126330), [@ArkaSaha30](https://github.com/ArkaSaha30)) [SIG Release and Testing]
- Removed the following feature gates:
  - `InTreePluginAWSUnregister`
  - `InTreePluginAzureDiskUnregister`
  - `InTreePluginAzureFileUnregister`
  - `InTreePluginGCEUnregister`
  - `InTreePluginOpenStackUnregister`
  - `InTreePluginvSphereUnregister` ([#124815](https://github.com/kubernetes/kubernetes/pull/124815), [@carlory](https://github.com/carlory)) [SIG Storage]
- Set LocalStorageCapacityIsolationFSQuotaMonitoring to false by default, to match UserNamespacesSupport (which the feature relies on) ([#126355](https://github.com/kubernetes/kubernetes/pull/126355), [@haircommander](https://github.com/haircommander)) [SIG Node]
- The Node Admission plugin now rejects CSR requests created by a node identity for the signers `kubernetes.io/kubelet-serving` or `kubernetes.io/kube-apiserver-client-kubelet` with a CN starting with `system:node:`, but where the CN is not `system:node:${node-name}`. The feature gate `AllowInsecureKubeletCertificateSigningRequests` defaults to `false`, but can be enabled to revert to the previous behavior. This feature gate will be removed in Kubernetes v1.33 ([#126441](https://github.com/kubernetes/kubernetes/pull/126441), [@micahhausler](https://github.com/micahhausler)) [SIG Auth]
- The ValidatingAdmissionPolicy metrics have been redone to count and time all validations, including failures and admissions. ([#126124](https://github.com/kubernetes/kubernetes/pull/126124), [@cici37](https://github.com/cici37)) [SIG API Machinery and Instrumentation]

## Dependencies

### Added
_Nothing has changed._

### Changed
- sigs.k8s.io/knftables: v0.0.16 → v0.0.17

### Removed
_Nothing has changed._



# v1.31.0-beta.0


## Downloads for v1.31.0-beta.0



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes.tar.gz) | feed42d09f9b053547d6e74a57bdad9ad629397247ca1b319f35223221b44f1986f8e8137e5ea6e3cd3697c92f30f1a0ff267ad5c63ba7461cb2ccad1a4893af
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-src.tar.gz) | 62ad62af35b3309e58d14edf264e3c1aed6cbd4ccb0f30d577856605be0d712b31c16bab1374874e814d177583fd66eb631f7f260da2c4944ee9a9d856751031

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | b04340d72abefe8eab81a24390f3d0446dfddc445b17202c8a5ff6ef408db8a7417c1bf3c8979cb1febfb72fc76c438ebec665d9297b06a7f3e4127976f9d897
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | 0770657abdf8d7ea3d42d3fb3b13f60095b767cf404d3baa375a6e78522948fa3c6f7df6fd24de6a429e4efe2c888349c9fd79057d095e33419b7056368b3691
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-linux-386.tar.gz) | 2763b17ec9bca7fe9fccb70f222647c7eb18d980897c723a93fa6f50c7e52500e231340eda42a9c3883680277e3adaa305776bba424666c6c90b68274e1d1bbc
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | c9cf45d9250c4832470a3a81373d2ac3d0e9a38ef40751c268228251358fe94f097efdf43ad63f88f26c61d45ac79f3c297d66f0b0b7d8885fed6276d8ec83a9
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-linux-arm.tar.gz) | 6f7879284fd956913c9c2e0c43b25fd6995524260069a3d4d3d35bdce776c8539301cbab50930dfa090a5179438d94a36939aceb5127cc6bf8b360e9d49f6186
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | 890b6eed70793d0fa5cfc8540de365e787608d8781bc39055ace1a4a7ae61886dd9297fae92c0fe76c4b7ed9b3fc1f1794d88c0134c4a336fe7217daf68b470d
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | 6a81375f99f4176a26ac05144bd82f1f2fd247b88041fd3f80ab2212c6623f0843e979edf753a65b43b508d9cefca8d567ac299125cb303b281ea0f87bcd1599
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | 241d1ced25ff6b99bd32ebf25bc6b53cbcf0582ee41476d44b13fff9f9b9264a13109ec56e64ed9c2588a7a7e25c4673fa2cc7299fe5d4597bef45784351c247
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-windows-386.tar.gz) | 3e9186866d1b4df935d7892a750df9e510c1d5b44682b270c29c58d547bf3cc3c2758500a015f1d56d00bbacd50bf01a022088c8b1d8e57ec5236cb613cab4f0
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | 7e1e1af36e28db6c8079fade9616004fd57437f8c6c2f7886bdae2e9485d41cf570ab7cdc6db5fcd033f669580e2441cd3a094548f795a20afde7f61243ef678
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | 70aee5f8b2b6d7882a8e69dfedbe21bc9300cf6ea008433a5fb61585bf78e54a714b0b4506e1372a85369d74bb9cffd807ca02b59f63cc5c9f64272a7858abb9

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | daf615524788e6c69c301de9d9ae7a0b21282168e1385a79faf0495df5b17ade093b89bbb704b95e5af5982863c6e9717bbee1b7aeeef4577bfa55d4f222737c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | dc8822d3423b68f8b34f14942ea9767b9d88f18a8f28eb7e65aab76454f717ba8c8a7ee9760c350282a95d57a5dd915416b14596adfb4b3711f24cd24d2bfe24
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | 8e1e363ff8f4e22e6f011fbd50955185e8dda432717dd46572d14327fd81c9785c1c9a22ae33d8d15e821fa29d428335b04058010b05dd472c8415fa4b0e8d94
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | 27ddd1b7c2ff823832a837ea5dffbadd2c58b678c8d65d296e099799234b8ebb16cba3e24e2214d0b3bf6c39162cc9c24275186ad3624a166a3b81f4a1782be7

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | c1ab508ec22f2f2b37c5643814de7f489b5d900d9732aa69393f52c7a18cd7c3c6f24ec4e7a6e82f1c278c8c213e34b28de0d6531ce22317bbcf539bdf490728
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | 4d45b093c44ab033f70391d50553e67fc50942cd81fa0f502c9cdebea34be92f217cd44da1daa942b966e67e7109683bb7c0dff94f884528fbb6dab1de2d98d9
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 9d8fdd8c757100ba28eea9a2fda5e2883913d73cfdb3d0092a38a124fb1e23c49d601b665b79f23f8557562a5c6b3e8c4a461bfaedc96c21b27fe301880b3188
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | bdaa11bba13e6d2f97de2d79b4493e0713dfc89b5ca2dffcedd75dd4369e076107f8a01280e1b2ed5a0f771991f23f2a98e85d54c060eab97b480208a70f5b0d
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 775a4ec0a9216d4f9a84c4aa26e009c553b4b664af676dc2f0d7e16ed4c9e79cd050aead281a1f947e6475f4adb67c4d5f0e643703b44f4bcf69deb9216bd5f0

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.31.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.31.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.31.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.31.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.31.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.31.0-beta.0](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.31.0-alpha.3

## Changes by Kind

### API Change

- Add UserNamespaces field to NodeRuntimeHandlerFeatures ([#126034](https://github.com/kubernetes/kubernetes/pull/126034), [@sohankunkerkar](https://github.com/sohankunkerkar)) [SIG API Machinery, Apps and Node]
- Fixes a 1.30.0 regression in openapi descriptions of PodIP.IP  and HostIP.IP fields to mark the fields used as keys in those lists as required. ([#126057](https://github.com/kubernetes/kubernetes/pull/126057), [@thockin](https://github.com/thockin)) [SIG API Machinery]
- Graduate JobPodFailurePolicy to GA and lock ([#125442](https://github.com/kubernetes/kubernetes/pull/125442), [@mimowo](https://github.com/mimowo)) [SIG API Machinery, Apps, Scheduling and Testing]
- Graduate PodDisruptionConditions to GA and lock ([#125461](https://github.com/kubernetes/kubernetes/pull/125461), [@mimowo](https://github.com/mimowo)) [SIG Apps, Node, Scheduling and Testing]
- PersistentVolumeLastPhaseTransitionTime feature is stable and enabled by default. ([#124969](https://github.com/kubernetes/kubernetes/pull/124969), [@RomanBednar](https://github.com/RomanBednar)) [SIG API Machinery, Apps, Storage and Testing]
- The (alpha) nftables mode of kube-proxy now requires version 1.0.1 or later
  of the nft command-line, and kernel 5.13 or later. (For testing/development
  purposes, you can use older kernels, as far back as 5.4, if you set the
  `nftables.skipKernelVersionCheck` option in the kube-proxy config, but this is not
  recommended in production since it may cause problems with other nftables
  users on the system.) ([#124152](https://github.com/kubernetes/kubernetes/pull/124152), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Use omitempty for optional Job Pod Failure Policy fields ([#126046](https://github.com/kubernetes/kubernetes/pull/126046), [@mimowo](https://github.com/mimowo)) [SIG Apps]
- User can choose a different static policy option `SpreadPhysicalCPUsPreferredOption` to spread cpus across physical cpus for some specific applications ([#123733](https://github.com/kubernetes/kubernetes/pull/123733), [@Jeffwan](https://github.com/Jeffwan)) [SIG Node]

### Feature

- --custom flag in kubectl debug will be enabled by default and yaml support is added ([#125333](https://github.com/kubernetes/kubernetes/pull/125333), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- Add --for=create option to kubectl wait ([#125868](https://github.com/kubernetes/kubernetes/pull/125868), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Add a TopologyManager policy option: max-allowable-numa-nodes to configures maxAllowableNUMANodes for kubelet. ([#124148](https://github.com/kubernetes/kubernetes/pull/124148), [@cyclinder](https://github.com/cyclinder)) [SIG Node and Testing]
- Add a warning log, an event for cgroup v1 usage and a metric for cgroup version. ([#125328](https://github.com/kubernetes/kubernetes/pull/125328), [@harche](https://github.com/harche)) [SIG Node]
- Added OCI VolumeSource Container Runtime Interface API fields and types. ([#125659](https://github.com/kubernetes/kubernetes/pull/125659), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Added namespace autocompletion for kubectl config set-context command ([#124994](https://github.com/kubernetes/kubernetes/pull/124994), [@TessaIO](https://github.com/TessaIO)) [SIG CLI]
- Bump the KubeletCgroupDriverFromCRI feature gate to beta and true by default. The kubelet will continue to use its KubeletConfiguration field as a fallback if the CRI implementation doesn't support this feature. ([#125828](https://github.com/kubernetes/kubernetes/pull/125828), [@haircommander](https://github.com/haircommander)) [SIG Node]
- Delay setting terminal Job conditions until all pods are terminal.
  
  Additionally, the FailureTarget condition is also added to the Job object in the first Job
  status update as soon as the failure conditions are met (backoffLimit is exceeded, maxFailedIndexes, 
  or activeDeadlineSeconds is exceeded).
  
  Similarly, the SuccessCriteriaMet condition is added in the first update as soon as the expected number
  of pod completions is reached.
  
  Also, introduce the following validation rules for Job status when JobManagedBy is enabled:
  1. the count of ready pods is less or equal than active
  2. when transitioning to terminal phase for Job, the number of terminating pods is 0
  3. terminal Job conditions (Failed and Complete) should be preceded by adding the corresponding interim conditions: FailureTarget and SuccessCriteriaMet ([#125510](https://github.com/kubernetes/kubernetes/pull/125510), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- ElasticIndexedJob is graduated to GA ([#125751](https://github.com/kubernetes/kubernetes/pull/125751), [@ahg-g](https://github.com/ahg-g)) [SIG Apps and Testing]
- Introduces new functionality to the dynamic client's `List` method, allowing users to enable API streaming. To activate this feature, users can set the `client-go.WatchListClient` feature gate.
  
  It is important to note that the server must support streaming for this feature to function properly. If streaming is not supported by the server, the client will revert to using the normal `LIST` method to obtain data. ([#125305](https://github.com/kubernetes/kubernetes/pull/125305), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Kube-scheduler implements scheduling hints for the VolumeRestriction plugin.
  Scheduling hints allow the scheduler to retry scheduling Pods that were previously rejected by the VolumeRestriction plugin if a new pvc added, and the pvc belongs to pod. ([#125280](https://github.com/kubernetes/kubernetes/pull/125280), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Scheduling and Storage]
- Kube-scheduler implements scheduling hints for the VolumeZone plugin.
  The scheduling hints allow the scheduler to only retry scheduling a Pod
  that was previously rejected by the VolemeZone plugin if  addition/update of node, 
  addition/update of PV, addition/update of PVC, or addition of SC matches pod's topology settings. ([#124996](https://github.com/kubernetes/kubernetes/pull/124996), [@Gekko0114](https://github.com/Gekko0114)) [SIG Scheduling and Storage]
- Kube-scheduler implements scheduling hints for the VolumeZone plugin.
  The scheduling hints allow the scheduler to only retry scheduling a Pod
  that was previously rejected by the VolemeZone plugin if  addition/update of node, 
  addition/update of PV, addition/update of PVC, or addition of SC matches pod's topology settings. ([#125000](https://github.com/kubernetes/kubernetes/pull/125000), [@Gekko0114](https://github.com/Gekko0114)) [SIG Scheduling and Storage]
- Kube-scheduler implements scheduling hints for the VolumeZone plugin.
  The scheduling hints allow the scheduler to only retry scheduling a Pod
  that was previously rejected by the VolemeZone plugin if  addition/update of node, 
  addition/update of PV, addition/update of PVC, or addition of SC matches pod's topology settings. ([#125001](https://github.com/kubernetes/kubernetes/pull/125001), [@Gekko0114](https://github.com/Gekko0114)) [SIG Scheduling and Storage]
- Kubelet: warn instead of error for the unsupported options on Windows "CgroupsPerQOS" and "EnforceNodeAllocatable". ([#123137](https://github.com/kubernetes/kubernetes/pull/123137), [@neolit123](https://github.com/neolit123)) [SIG Node and Windows]
- Kubernetes is now built with go 1.22.5 ([#125894](https://github.com/kubernetes/kubernetes/pull/125894), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- The Service trafficDistribution field has graduated to beta and is now available for configuration by default, without the need to enable any feature flag. Services that do not have the field configured will continue to operate with their existing behavior. Refer to the documentation https://kubernetes.io/docs/concepts/services-networking/service/#traffic-distribution for more details. ([#125838](https://github.com/kubernetes/kubernetes/pull/125838), [@gauravkghildiyal](https://github.com/gauravkghildiyal)) [SIG Network and Testing]
- The scheduler implements QueueingHint in VolumeBinding plugin's CSINode event, which enhances the throughput of scheduling. ([#125097](https://github.com/kubernetes/kubernetes/pull/125097), [@YamasouA](https://github.com/YamasouA)) [SIG Scheduling and Storage]
- Windows Kubeproxy will use the update load balancer API for load balancer updates, instead of the previous delete and create APIs.
  - Deletion of remote endpoints will be triggered only for terminated endpoints (those present in the old endpoints map but not in the new endpoints map), whereas previously it was also done for terminating endpoints. ([#124092](https://github.com/kubernetes/kubernetes/pull/124092), [@princepereira](https://github.com/princepereira)) [SIG Network and Windows]

### Bug or Regression

- Add `/sys/devices/virtual/powercap` to default masked paths. It avoids the potential security risk that the ability to read these files may offer a power-based sidechannel attack against any workloads running on the same kernel. ([#125970](https://github.com/kubernetes/kubernetes/pull/125970), [@carlory](https://github.com/carlory)) [SIG Node]
- Fix a bug that when PodTopologySpread rejects Pods, they may be stuck in Pending state for 5 min in a worst case scenario.
  The same problem could happen with custom plugins which have Pod/Add or Pod/Update in EventsToRegister,
  which is also solved with this PR, but only when the feature flag SchedulerQueueingHints is enabled. ([#122627](https://github.com/kubernetes/kubernetes/pull/122627), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Fix endpoints status out-of-sync when the pod state changes rapidly ([#125675](https://github.com/kubernetes/kubernetes/pull/125675), [@tnqn](https://github.com/tnqn)) [SIG Apps, Network and Testing]
- Fix the bug where PodIP field is temporarily removed for a terminal pod ([#125404](https://github.com/kubernetes/kubernetes/pull/125404), [@mimowo](https://github.com/mimowo)) [SIG Node and Testing]
- For statically provisioned PVs, if its volume source is CSI type or it has migrated annotation, when it's deleted, the PersisentVolume controller won't changes its phase to the Failed state. 
  
  With this patch, the external provisioner can remove the finalizer in next reconcile loop. Unfortunately if the provious existing pv has the Failed state, this patch won't take effort. It requires users to remove finalizer. ([#125767](https://github.com/kubernetes/kubernetes/pull/125767), [@carlory](https://github.com/carlory)) [SIG Apps and Storage]
- LastSuccessfullTime in cronjobs will now be set reliably ([#122025](https://github.com/kubernetes/kubernetes/pull/122025), [@lukashankeln](https://github.com/lukashankeln)) [SIG Apps]
- Stop using wmic on Windows to get uuid in the kubelet ([#126012](https://github.com/kubernetes/kubernetes/pull/126012), [@marosset](https://github.com/marosset)) [SIG Node and Windows]
- The scheduler retries scheduling Pods rejected by PreFilterResult (PreFilter plugins) more appropriately; it now takes events registered in those rejector PreFilter plugins into consideration. ([#122251](https://github.com/kubernetes/kubernetes/pull/122251), [@olderTaoist](https://github.com/olderTaoist)) [SIG Scheduling and Testing]

### Other (Cleanup or Flake)

- API Priority and Fairness feature was promoted to GA in 1.29, the corresponding 
  feature gate 'APIPriorityAndFairness' has been removed in 1.31. ([#125846](https://github.com/kubernetes/kubernetes/pull/125846), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Drop support for the deprecated and unsupported kubectl run flags:
  - filename
  - force
  - grace-period
  - kustomize
  - recursive
  - timeout
  - wait
  
  Drop support for the deprecated --delete-local-data from kubectl drain, users should use --delete-emptydir-data, instead. ([#125842](https://github.com/kubernetes/kubernetes/pull/125842), [@soltysh](https://github.com/soltysh)) [SIG CLI]

## Dependencies

### Added
- cel.dev/expr: v0.15.0

### Changed
- github.com/cenkalti/backoff/v4: [v4.2.1 → v4.3.0](https://github.com/cenkalti/backoff/compare/v4.2.1...v4.3.0)
- github.com/cespare/xxhash/v2: [v2.2.0 → v2.3.0](https://github.com/cespare/xxhash/compare/v2.2.0...v2.3.0)
- github.com/cncf/udpa/go: [c52dc94 → 269d4d4](https://github.com/cncf/udpa/compare/c52dc94...269d4d4)
- github.com/cncf/xds/go: [e9ce688 → 555b57e](https://github.com/cncf/xds/compare/e9ce688...555b57e)
- github.com/envoyproxy/go-control-plane: [v0.11.1 → v0.12.0](https://github.com/envoyproxy/go-control-plane/compare/v0.11.1...v0.12.0)
- github.com/envoyproxy/protoc-gen-validate: [v1.0.2 → v1.0.4](https://github.com/envoyproxy/protoc-gen-validate/compare/v1.0.2...v1.0.4)
- github.com/felixge/httpsnoop: [v1.0.3 → v1.0.4](https://github.com/felixge/httpsnoop/compare/v1.0.3...v1.0.4)
- github.com/go-logr/logr: [v1.4.1 → v1.4.2](https://github.com/go-logr/logr/compare/v1.4.1...v1.4.2)
- github.com/golang/glog: [v1.1.2 → v1.2.1](https://github.com/golang/glog/compare/v1.1.2...v1.2.1)
- github.com/google/uuid: [v1.3.1 → v1.6.0](https://github.com/google/uuid/compare/v1.3.1...v1.6.0)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.16.0 → v2.20.0](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.16.0...v2.20.0)
- github.com/rogpeppe/go-internal: [v1.11.0 → v1.12.0](https://github.com/rogpeppe/go-internal/compare/v1.11.0...v1.12.0)
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.46.0 → v0.53.0
- go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp: v0.44.0 → v0.53.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.20.0 → v1.27.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.20.0 → v1.28.0
- go.opentelemetry.io/otel/metric: v1.20.0 → v1.28.0
- go.opentelemetry.io/otel/sdk: v1.20.0 → v1.28.0
- go.opentelemetry.io/otel/trace: v1.20.0 → v1.28.0
- go.opentelemetry.io/otel: v1.20.0 → v1.28.0
- go.opentelemetry.io/proto/otlp: v1.0.0 → v1.3.1
- google.golang.org/genproto/googleapis/api: b8732ec → 5315273
- google.golang.org/genproto/googleapis/rpc: b8732ec → f6361c8
- google.golang.org/grpc: v1.59.0 → v1.65.0
- k8s.io/utils: 3b25d92 → 18e509b

### Removed
_Nothing has changed._



# v1.31.0-alpha.3


## Downloads for v1.31.0-alpha.3



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes.tar.gz) | 81ee64c3ae9f3e528b79c933d9da60d3aae1a2c1a1a7378f273cfecadc6d86b8909f24cf5978aced864867de405c17f327e86d0c71c6e3f3e7add94eece310cd
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-src.tar.gz) | 269f41b1394f6531ac94fd83b18150380b7c6f0c79d46195a1191fbfd90a751582865f672d3b408e9d2b2cbc52d4d65e75d3faad1ec7144b8ddaa8a9b5f97fe6

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | 438a451916f27af8833fc7bc37e262736be418e7c2063eb70784a3b375962b2b7cbc6640cc7813a8d1a8b484bb4b9355b1862ca19461ca9fcebf4f1859e3e673
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | e2caa1e7248e8ff226afe6e2e3462617457ea00bee4b2f3ba83f859015e8315bcd5df4b4788f07e953f5e17d37e518b67da0d9288bb1ad351dd1dec9294431ea
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 53b02b393dfcc111f1c3becc437593278704cb8e44855cf0d68f11996044be6b7da5d348c493d55829f340424c38559155c759dc75ec72893958e99a84a976a0
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | 078233f079f0fcb73e5464995de716af999a3773c112834d2d3f711aff390bf0f8d1d2006c3bf7b53503ecd4757f51b02b33eabebd0b306469cf0d4353639325
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | 4214afa49d9fd177eb1cc9706b988962a6ff7ea1ac5e1c441f728b4076d6c02377c22d45ed50f497e2fe69ff767625c07150a5e41bbcab873811ec747293412f
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | 43482d72b2db2f76630981d8b21ab8a78cb23fb4ca4de1e5d8199634fc4935def3fa1493fa92c45ed387e204877b0305e066aa5bdf188f2018322128b9c836d7
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | bba166df43a4c4b157ff469f1febc5c6e999655ebb63e05100db3b3f4b772234c5291458a12f013aa5f7af030957af1680434b89d4a78fb702254fe9e29b711d
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 81e1464802062b0ceec6c52f3d43eae831109a07c37d6941d4f20767e9bba908e10f5f3e3bb8ab5efdcffbb45c62bad1f18430141c577e99a4c69540dd4e06f7
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-windows-386.tar.gz) | e4f334c8bc0f3192f8aaf07c1b6ef94ce38ac150308fd0bfb27c1433dcba1543f0d56028a6ed4197c2ac8f9e2c650654549eb740ecabc2f2e671ebe6691d06f0
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 85df16f89541ac6a3e7b1b84b690c5255a09b22a7a8a0c4a0c1386edaeaf41a155b10f54d6fd5c096d57729979821180484ad869c2c7c31e650fcd6c1028d97a
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-client-windows-arm64.tar.gz) | 4656c8ed9c72425f077d059dc0cc9e3360893799fc79e98f25866411f7d25775f3cd1d9bbb0d10e690d319cb4dfa0839547704fae3dba77853ce36590d48f006

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 4b1c40bad5b6b4669b139a8956d51435d59111df19cc81c652eb2fcd1e1e9c850dec20b12e2f00f358bb5acc5ced2a6e7dc5e14cf8f063cca226cec55e2d3c19
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 23f6d045bbb914204dae109767879c5b58d389d8ebba6969b13e794d98a62c9b49fa7955f5ed6520063434779b3f316df9ee181943cf5a67146426c1b81b19bf
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | 16830cf5852f485f0a68cfa68c8fe505019d676e6b7e80783430cff29b9a8c9cf35aea6f2fb9de608b8a177964d7b49a9335eba8a6e11ec18725b3decea1dce8
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | 8ba76e6c863cbb98e3179efcb23144ec367389c0735fe867df21fd3104945c869932684066b6009a906e3bf480ac7051a6b23c366adfd50591be93be9c6b2cf0

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | 213c7692bbd980a4df2f5cff17d5688a0c635893ebdc27a11da4b40e97bb011caf0a4b7305600ff50d9e6e5d6b4daa31ccec2c90d171a72f97ecee0532316023
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | f6a627b53d2f8ab7848eda49d69c68eb4a319e0a5721c34afb69858f2e25f9712cbf310626b4d58b0d9eed6464ee77b8eaad21e03cac4418b3659eebe4d35b11
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | a25180775ae133d3c9278758d444e4934ec1b87c3b116fde03ff9e4249e3fca3c5135195a671614bb294e38f8e708ba5b77ba30fd763b634f47145c915d4dc8a
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | aea8682dcb0cf37c5c51e817691a44d8e058cda3977a79cad973638a5a77a3d554f90c7aa1c80b441b442d223c0e995ecc187e8c977ee6bb4cfd0768bc46ca21
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | 2492217219ebf17574fba60aa612ab4adba0403f360a267657dd24092112ef7795302f255eb264ca36b0924c4bd527ade82d93ae65261f2856f512d9aa6a6104

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.31.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.31.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.31.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.31.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.31.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.31.0-alpha.3](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.31.0-alpha.2

## Changes by Kind

### API Change

- DRA: in the `pod.spec.recourceClaims` array, the `source` indirection is no longer necessary. Instead of e.g. `source: resourceClaimTemplateName: my-template`, one can write `resourceClaimTemplateName: my-template`. ([#125116](https://github.com/kubernetes/kubernetes/pull/125116), [@pohly](https://github.com/pohly)) [SIG API Machinery, Apps, Auth, Node, Scheduling and Testing]
- Fix code-generator client-gen to work with `api/v1`-like package structure. ([#125162](https://github.com/kubernetes/kubernetes/pull/125162), [@sttts](https://github.com/sttts)) [SIG API Machinery and Apps]
- KEP-1880: Users of the new feature to add multiple service CIDR will use by default a dual-write strategy on the new ClusterIP allocators to avoid the problem of possible duplicate IPs allocated to Services when running skewed kube-apiservers using different allocators. They can opt-out of this behavior by enabled the feature gate DisableAllocatorDualWrite ([#122047](https://github.com/kubernetes/kubernetes/pull/122047), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Instrumentation and Testing]
- Kube-apiserver: ControllerRevision objects are now verified to contain valid JSON data in the `data` field. ([#125549](https://github.com/kubernetes/kubernetes/pull/125549), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Apps]
- Update the feature MultiCIDRServiceAllocator to beta (disabled by default). Users need to enable the feature gate and the networking v1beta1 group to be able to use this new feature, that allows to dynamically reconfigure Service CIDR ranges. ([#125021](https://github.com/kubernetes/kubernetes/pull/125021), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, CLI, Etcd, Instrumentation, Network and Testing]
- When the featuregate AnonymousAuthConfigurableEndpoints is enabled users can update the AuthenticationConfig file with endpoints for with anonymous requests are alllowed. ([#124917](https://github.com/kubernetes/kubernetes/pull/124917), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG API Machinery, Auth, Cloud Provider, Node and Testing]

### Feature

- Add Extra.DisableAvailableConditionController for Generic Control Plane setup in kube-aggregator ([#125650](https://github.com/kubernetes/kubernetes/pull/125650), [@mjudeikis](https://github.com/mjudeikis)) [SIG API Machinery]
- Add field management support to the fake client-go typed client.
  Use `fake.NewClientset()` instead of `fake.NewSimpleClientset()` to create a clientset with managed field support. ([#125560](https://github.com/kubernetes/kubernetes/pull/125560), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Auth, Instrumentation and Testing]
- Continue streaming kubelet logs when the CRI server of the runtime is unavailable. ([#124025](https://github.com/kubernetes/kubernetes/pull/124025), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Graduates the `WatchList` feature gate to Beta for kube-apiserver and enables `WatchListClient` for KCM. ([#125591](https://github.com/kubernetes/kubernetes/pull/125591), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Improve memory usage of kube-apiserver by dropping the `.metadata.managedFields` field that self-requested informers of kube-apiserver doesn't need. ([#124667](https://github.com/kubernetes/kubernetes/pull/124667), [@linxiulei](https://github.com/linxiulei)) [SIG API Machinery]
- In the client-side apply on create, defining the null value as "delete the key associated with this value". ([#125646](https://github.com/kubernetes/kubernetes/pull/125646), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery, CLI and Testing]
- KEP-3857: promote RecursiveReadOnlyMounts feature to beta ([#125475](https://github.com/kubernetes/kubernetes/pull/125475), [@AkihiroSuda](https://github.com/AkihiroSuda)) [SIG Node]
- Kube-scheduler implements scheduling hints for the VolumeRestriction plugin.
  Scheduling hints allow the scheduler to retry scheduling Pods that were previously rejected by the VolumeRestriction plugin if the Pod is deleted and the deleted Pod conflicts with the existing volumes of the current Pod. ([#125279](https://github.com/kubernetes/kubernetes/pull/125279), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Scheduling and Storage]
- Kubeadm: added the ControlPlaneKubeletLocalMode feature gate. It can be used to tell kubeadm to use the local kube-apiserver endpoint for the kubelet when creating a cluster with "kubeadm init" or when joining control plane nodes with "kubeadm join".  The "kubeadm join" workflow now includes two new experimental phases called "control-plane-join-etcd" and "kubelet-wait-bootstrap" which will be used when the feature gate is enabled. This phases will be marked as non-experimental when ControlPlaneKubeletLocalMode becomes GA. During "kubeadm upgrade" commands, if the feature gate is enabled, modify the "/etc/kubernetes/kubelet.conf " to use the local kube-apiserver endpoint. This upgrade mechanism will be removed once the feature gate goes GA and is hardcoded to true. ([#125582](https://github.com/kubernetes/kubernetes/pull/125582), [@chrischdi](https://github.com/chrischdi)) [SIG Cluster Lifecycle]
- Move ConsistentListFromCache feature flag to Beta and enable it by default ([#123513](https://github.com/kubernetes/kubernetes/pull/123513), [@serathius](https://github.com/serathius)) [SIG API Machinery and Testing]
- Promote HonorPVReclaimPolicy to beta and enable the feature-gate by default ([#124842](https://github.com/kubernetes/kubernetes/pull/124842), [@carlory](https://github.com/carlory)) [SIG Apps, Storage and Testing]
- Promoted the feature gate `KubeProxyDrainingTerminatingNodes` to stable ([#125082](https://github.com/kubernetes/kubernetes/pull/125082), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG Network]
- The PodDisruptionBudget `spec.unhealthyPodEvictionPolicy` field has graduated to GA. This field may be set to `AlwaysAllow` to always allow unhealthy pods covered by the PodDisruptionBudget to be evicted. ([#123428](https://github.com/kubernetes/kubernetes/pull/123428), [@atiratree](https://github.com/atiratree)) [SIG Apps, Auth, Node and Testing]
- The feature-gate CSIMigrationPortworx was promoted to beta in Kubernetes 1.25, but turn it off by default. In 1.31, it was turned on by default. Before upgrading to 1.31, please make sure that the corresponding portworx csi driver is installed if you are using Portworx. ([#125016](https://github.com/kubernetes/kubernetes/pull/125016), [@carlory](https://github.com/carlory)) [SIG Storage]

### Bug or Regression

- DRA: using structured parameters with a claim that gets reused between pods may have led to a claim with an invalid state (allocated without a finalizer) which then caused scheduling of pods using the claim to stop. ([#124931](https://github.com/kubernetes/kubernetes/pull/124931), [@pohly](https://github.com/pohly)) [SIG Node and Scheduling]
- Fix a bug that Pods could stuck in the unschedulable pod pool 
  if they're rejected by PreEnqueue plugins that could change its result by a change in resources apart from Pods.
  
  DRA plugin is the only plugin that meets the criteria of the bug in in-tree, 
  and hence if you have `DynamicResourceAllocation` feature flag enabled, 
  your DRA Pods could be affected by this bug. ([#125527](https://github.com/kubernetes/kubernetes/pull/125527), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Fix bug where Server Side Apply causes spurious resourceVersion bumps on no-op patches to custom resources. ([#125263](https://github.com/kubernetes/kubernetes/pull/125263), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]
- Fix bug where Server Side Apply causing spurious resourceVersion bumps on no-op patches containing empty maps. ([#125317](https://github.com/kubernetes/kubernetes/pull/125317), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Testing]
- Fix null lastTransitionTime in Pod condition when setting scheduling gate. ([#122636](https://github.com/kubernetes/kubernetes/pull/122636), [@lianghao208](https://github.com/lianghao208)) [SIG Node and Scheduling]
- Fix recursive LIST from watch cache returning object matching key ([#125584](https://github.com/kubernetes/kubernetes/pull/125584), [@serathius](https://github.com/serathius)) [SIG API Machinery and Testing]
- Fix: during the kube-controller-manager restart, when the corresponding Endpoints resource was manually deleted and recreated, causing the endpointslice to fail to be created normally. ([#125359](https://github.com/kubernetes/kubernetes/pull/125359), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Apps and Network]
- Kube-apiserver: fixes a 1.27+ regression watching a single namespace via the deprecated /api/v1/watch/namespaces/$name endpoint where watch events were not delivered after the watch was established ([#125145](https://github.com/kubernetes/kubernetes/pull/125145), [@xyz-li](https://github.com/xyz-li)) [SIG API Machinery, Node and Testing]
- Kube-apiserver: timeouts configured for authorization webhooks in the --authorization-config file are now honored, and webhook timeouts are accurately reflected in webhook metrics with result=timeout ([#125552](https://github.com/kubernetes/kubernetes/pull/125552), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth and Testing]
- Kubeadm: Added `--yes` flag to the list of allowed flags so that it can be mixed with `kubeadm upgrade apply --config` ([#125566](https://github.com/kubernetes/kubernetes/pull/125566), [@xmudrii](https://github.com/xmudrii)) [SIG Cluster Lifecycle]
- Kubeadm: during the validation of existing kubeconfig files on disk, handle cases where the "ca.crt" is a bundle and has intermediate certificates. Find a common trust anchor between the "ca.crt" bundle and the CA in the existing kubeconfig on disk instead of treating "ca.crt" as a file containing a single CA. ([#123102](https://github.com/kubernetes/kubernetes/pull/123102), [@astundzia](https://github.com/astundzia)) [SIG Cluster Lifecycle]
- Kubeadm: fix a bug where the path of the manifest can not be specified when `kubeadm upgrade diff` specified a config file, and the `--api-server-manifest`, `--controller-manager-manifest` and `--scheduler-manifest` flags of `kubeadm upgrade diff` are marked as deprecated and will be removed in a future release. ([#125779](https://github.com/kubernetes/kubernetes/pull/125779), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: the `--feature-gates` flag is deprecated and no-op for `kubeadm upgrade apply/plan`, and it will be removed in a future release. The upgrade workflow is not designed to reconfigure the cluster. Please edit the 'featureGates' field of ClusterConfiguration which is defined in the kube-system/kubeadm-config ConfigMap instead. ([#125797](https://github.com/kubernetes/kubernetes/pull/125797), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubelet now hard rejects pods with AppArmor if the node does not have AppArmor enabled. ([#125776](https://github.com/kubernetes/kubernetes/pull/125776), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG Node]
- Now the .status.ready field is tracked faster when active Pods are deleted, specifically when Job is failed, gets suspended or has too many active pods ([#125546](https://github.com/kubernetes/kubernetes/pull/125546), [@dejanzele](https://github.com/dejanzele)) [SIG Apps]
- When schedulingQueueHint is enabled, the scheduling queue doesn't update Pods being scheduled immediately. ([#125578](https://github.com/kubernetes/kubernetes/pull/125578), [@nayihz](https://github.com/nayihz)) [SIG Scheduling]

### Other (Cleanup or Flake)

- DRA: fix some small, unlikely race condition during pod scheduling ([#124595](https://github.com/kubernetes/kubernetes/pull/124595), [@pohly](https://github.com/pohly)) [SIG Node, Scheduling and Testing]
- Kube-apiserver: the `--enable-logs-handler` flag and log-serving functionality which was already deprecated is now switched off by default and scheduled to be removed in v1.33. ([#125787](https://github.com/kubernetes/kubernetes/pull/125787), [@dims](https://github.com/dims)) [SIG API Machinery, Network and Testing]
- Kubeadm: improve the warning/error messages of `validateSupportedVersion` to include the checked resource kind name. ([#125758](https://github.com/kubernetes/kubernetes/pull/125758), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Removing deprecated kubectl exec [POD] [COMMAND] ([#125437](https://github.com/kubernetes/kubernetes/pull/125437), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI and Testing]
- This change improves documentation clarity, making it more understandable for new users and contributors. ([#125536](https://github.com/kubernetes/kubernetes/pull/125536), [@this-is-yaash](https://github.com/this-is-yaash)) [SIG Release]
- `kubectl describe service` now shows internal traffic policy and ip mode of load balancer IP ([#125117](https://github.com/kubernetes/kubernetes/pull/125117), [@tnqn](https://github.com/tnqn)) [SIG CLI and Network]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/Microsoft/hcsshim: [v0.8.25 → v0.8.26](https://github.com/Microsoft/hcsshim/compare/v0.8.25...v0.8.26)
- github.com/cpuguy83/go-md2man/v2: [v2.0.3 → v2.0.4](https://github.com/cpuguy83/go-md2man/compare/v2.0.3...v2.0.4)
- github.com/fxamacker/cbor/v2: [v2.7.0-beta → v2.7.0](https://github.com/fxamacker/cbor/compare/v2.7.0-beta...v2.7.0)
- github.com/moby/spdystream: [v0.2.0 → v0.4.0](https://github.com/moby/spdystream/compare/v0.2.0...v0.4.0)
- github.com/moby/sys/mountinfo: [v0.6.2 → v0.7.1](https://github.com/moby/sys/compare/mountinfo/v0.6.2...mountinfo/v0.7.1)
- github.com/moby/term: [1aeaba8 → v0.5.0](https://github.com/moby/term/compare/1aeaba8...v0.5.0)
- github.com/opencontainers/runc: [v1.1.12 → v1.1.13](https://github.com/opencontainers/runc/compare/v1.1.12...v1.1.13)
- github.com/prometheus/client_golang: [v1.19.0 → v1.19.1](https://github.com/prometheus/client_golang/compare/v1.19.0...v1.19.1)
- github.com/prometheus/client_model: [v0.6.0 → v0.6.1](https://github.com/prometheus/client_model/compare/v0.6.0...v0.6.1)
- github.com/prometheus/common: [v0.48.0 → v0.55.0](https://github.com/prometheus/common/compare/v0.48.0...v0.55.0)
- github.com/prometheus/procfs: [v0.12.0 → v0.15.1](https://github.com/prometheus/procfs/compare/v0.12.0...v0.15.1)
- github.com/spf13/cobra: [v1.8.0 → v1.8.1](https://github.com/spf13/cobra/compare/v1.8.0...v1.8.1)
- github.com/stretchr/objx: [v0.5.0 → v0.5.2](https://github.com/stretchr/objx/compare/v0.5.0...v0.5.2)
- github.com/stretchr/testify: [v1.8.4 → v1.9.0](https://github.com/stretchr/testify/compare/v1.8.4...v1.9.0)
- go.etcd.io/etcd/api/v3: v3.5.13 → v3.5.14
- go.etcd.io/etcd/client/pkg/v3: v3.5.13 → v3.5.14
- go.etcd.io/etcd/client/v3: v3.5.13 → v3.5.14
- golang.org/x/crypto: v0.23.0 → v0.24.0
- golang.org/x/net: v0.25.0 → v0.26.0
- golang.org/x/oauth2: v0.20.0 → v0.21.0
- golang.org/x/sys: v0.20.0 → v0.21.0
- golang.org/x/term: v0.20.0 → v0.21.0
- golang.org/x/text: v0.15.0 → v0.16.0
- golang.org/x/tools: v0.21.0 → e35e4cc
- google.golang.org/protobuf: v1.33.0 → v1.34.2
- k8s.io/klog/v2: v2.120.1 → v2.130.1

### Removed
- go.uber.org/mock: v0.4.0



# v1.31.0-alpha.2


## Downloads for v1.31.0-alpha.2



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes.tar.gz) | 16c79d46cc58352ebccbe1be1139dfc8cfd6ac522fa6e08ea54dbf1d5f9544a508431f43f82670f1a554ac9a7059307a74e20c1927f41c013cacad80951bf47d
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-src.tar.gz) | ad7637808305ea59cd61e27788fb81f51b0e5c41355c189f689a7a8e58e0d1b6fb0cd278a29fa0c74b6307b1af3a37667650e72bb6d1796b2dc1c7c13f3f4539

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | 358ee8f7e6a3afa76bdc96a2c11463b42421ee5d41ec6f3eeaaf86ccd34eb433b0c0b20bf0097085758aa95b63dce18357d34f885662724c1d965ff4f2bd21a2
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 2ce564a16b49f4da3e2fa322c3c1ee4fcc02b9a12f8261232d835094222455c9b2c55dd5fce7980aa5bf87e40752875b2124e31e93db9558ca25a4a466beec15
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-linux-386.tar.gz) | fcd1e9ed89366af00091d3626d4e3513d3ea329b25e0a4b701f981d384bc71f2a348ccd99e6c329e7076cd75dab9dc13ab31b4818b24596996711bc034c58400
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | e7b705bb04de6eca9a633a4ff3c2486e486cbd61c77a7c75c6d94f1b5612ed1e6f852c060c0194d5c2bfd84d905cdb8ea3b19ddbedb46e458b23db82c547d3a7
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | 93c5385082ecf84757ff6830678f5469a5b2463687d8a256f920c0fd25ed4c08bd06ec2beaf507d0bbe10d9489632349ee552d8d3f8f861c9977ff307bb89f23
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | e9427fe6e034e9cec5b522b01797e3144f08ba60a01cd0c86eba7cb27811e470c0e3eb007e6432f4d9005a2cc57253956b66cd2eb68cd4ae73659193733910df
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 91a3b101a9c5f513291bf80452d3023c0000078c16720a2874dd554c23a87d15632f9e1bf419614e0e3a9d8b2f3f177eee3ef08d405aca3c7e3311dec3dfebba
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 1cda8de38ccdc8fbf2a0c74bd5d35b4638f6c40c5aa157e2ade542225462a662e4415f3d3abb31c1d1783c7267f16530b3b392e72b75b9d5f797f7137eecba66
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-windows-386.tar.gz) | c7fe8d85f00150778cc3f3bde20a627cd160f495a7dcd2cf67beb1604c29b2f06c9e521d7c3249a89595d8cda4c2f6ac652fa27ec0dd761e1f2539edcbb5d0ef
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 62315ae783685e5cb74e149da7d752973d115e95d5c0e58c1c06f8ceec925f4310fb9c220be42bad6fd7dc4ff0540343a4fff12767a5eb305a29ff079f3b940a
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-client-windows-arm64.tar.gz) | eddd92de174554f026d77f333eac5266621cffe0d07ad5c32cf26d46f831742fa3b6f049494eb1a4143e90fdded80a795e5ddce37be5f15cd656bdc102f3fcb2

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | c561ebdfb17826faefcc44d4b9528890a9141a31e6d1a6935cce88a4265ba10eddbd0726bd32cffcdd09374247a1d5faf911ca717afc9669716c6a2e61741e65
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 3faed373e59cef714034110cdbdd33d861b72e939058a193f230908fea4961550216490e5eca43ffaa838cd9c2884267c685a0f4e2fc686fd0902bbb2d97a01c
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | 336d38807433e32cdcb09a0a2ee8cbb7eb2d13c9d991c5fc228298c0bec13d45b4b001db96199498a2f0f106d27c716963b6c49b9f40e07f8800801e3cea5ec9
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | 2ebc780b3323db8209f017262e3a01c040d3ee986bdd0732085fbe945cb0e135a1c8bd4adf31ded6576e19e2b5370efded9f149ef724ad5e2bbddf981e8c1bda

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 2e6d0a1d6698be1ffeadf54da70cb4b988ead6ed9e232372d008f2ec49cb1dd9e30efa5a2cc7f1768d1b9c6facde002b39931433e3a239df46f6db0c067dbbac
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 7fa93c164097f60bb6dcbaccf87024a5c6fb300915a46bf1824c57472d198c6e52c39fa27d0e3cd55acb55833579dd6ddb4024e1500f7998140ef10dbec47b22
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | b6ae26348b3703680c15d508b65253d0e58d93d3b435668a40a1d5dd65b5ed6ab2b0190ca6ea77d2091f7223dad225e3f671ae72bda4ed5be0d29b753ad498b6
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | 5247cfa0499518bc5f00dda154c1dd36ef5b62e1a2861deb3a36e3a5651eefd05f7a2004eba6500912cafd81ce485f172014c8680178ab8d3ba981616c467dea
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 5c29c702fbb78b53961e2afe3f51604199abcd664a270fbf2ff3b0273b983a02fbbbae4253a652ea4cd7cbef0543fe3b012c00f88e8071d9213f7cb6c4e86bda

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.31.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.31.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.31.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.31.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.31.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.31.0-alpha.2](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.31.0-alpha.1

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - The scheduler starts to use QueueingHint registered for Pod/Updated event to determine whether unschedulable Pods update make them schedulable, when the feature gate `SchedulerQueueingHints` is enabled.
  Previously, when unschedulable Pods are updated, the scheduler always put Pods back to activeQ/backoffQ. But, actually not all updates to Pods make Pods schedulable, especially considering many scheduling constraints nowadays are immutable.
  Now, when unschedulable Pods are updated, the scheduling queue checks with QueueingHint(s) whether the update may make the pods schedulable, and requeues them to activeQ/backoffQ **only when** at least one QueueingHint(s) return Queue. 
  
  Action required for custom scheduler plugin developers:
  Plugins **have to** implement a QueueingHint for Pod/Update event if the rejection from them could be resolved by updating unscheduled Pods themselves.
  Example: suppose you develop a custom plugin that denies Pods that have a `schedulable=false` label. 
  Given Pods with a `schedulable=false` label will be schedulable if the `schedulable=false` label is removed, this plugin would implement QueueingHint for Pod/Update event that returns Queue when such label changes are made in unscheduled Pods. ([#122234](https://github.com/kubernetes/kubernetes/pull/122234), [@AxeZhan](https://github.com/AxeZhan)) [SIG Scheduling and Testing]
 
## Changes by Kind

### API Change

- Fixed incorrect "v1 Binding is deprecated in v1.6+" warning in kube-scheduler log. ([#125540](https://github.com/kubernetes/kubernetes/pull/125540), [@pohly](https://github.com/pohly)) [SIG API Machinery]

### Feature

- Feature gates for PortForward (kubectl port-forward) over WebSockets are now enabled by default (Beta).
  - Server-side feature gate: PortForwardWebsocket
  - Client-side (kubectl) feature gate: PORT_FORWARD_WEBSOCKETS environment variable
  - To turn off PortForward over WebSockets for kubectl, the environment variable feature gate must be explicitly set - PORT_FORWARD_WEBSOCKETS=false ([#125528](https://github.com/kubernetes/kubernetes/pull/125528), [@seans3](https://github.com/seans3)) [SIG API Machinery and CLI]
- Introduces new functionality to the client-go's `List` method, allowing users to enable API streaming. To activate this feature, users can set the `client-go.WatchListClient` feature gate.
  
  It is important to note that the server must support streaming for this feature to function properly. If streaming is not supported by the server, client-go will revert to using the normal `LIST` method to obtain data. ([#124509](https://github.com/kubernetes/kubernetes/pull/124509), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery, Auth, Instrumentation and Testing]
- Kubeadm: enabled the v1beta4 API. For a complete changelog since v1beta3 please see https://kubernetes.io/docs/reference/config-api/kubeadm-config.v1beta4/. 
  
  The API does include a few breaking changes:
  - The "extraArgs" component construct is now a list of "name"/"value" pairs instead of a string/string map. This has been done to support duplicate args where needed.
  - The "JoinConfiguration.discovery.timeout" field has been replaced by "JoinConfiguration.timeouts.discovery".
  - The "ClusterConfiguration.timeoutForControlPlane" field has been replaced by "{Init|Join}Configuration.timeouts.controlPlaneComponentHealthCheck".
  Please use the command "kubeadm config migrate" to migrate your existing v1beta3 configuration to v1beta4.
  
  v1beta3 is now marked as deprecated but will continue to be supported until version 1.34 or later.
  The storage configuration in the kube-system/kubeadm-config ConfigMap is now a v1beta4 ClusterConfiguration. ([#125029](https://github.com/kubernetes/kubernetes/pull/125029), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- LogarithmicScaleDown is now GA ([#125459](https://github.com/kubernetes/kubernetes/pull/125459), [@MinpengJin](https://github.com/MinpengJin)) [SIG Apps and Scheduling]

### Failing Test

- Fixed issue where following Windows container logs would prevent container log rotation. ([#124444](https://github.com/kubernetes/kubernetes/pull/124444), [@claudiubelu](https://github.com/claudiubelu)) [SIG Node, Testing and Windows]
- Pkg k8s.io/apiserver/pkg/storage/cacher, method (*Cacher) Wait(context.Context) error ([#125450](https://github.com/kubernetes/kubernetes/pull/125450), [@mauri870](https://github.com/mauri870)) [SIG API Machinery]

### Bug or Regression

- DRA: enhance validation for the ResourceClaimParametersReference and ResourceClassParametersReference with the following rules:
  
  1. `apiGroup`: If set, it must be a valid DNS subdomain (e.g. 'example.com').
  2. `kind` and `name`: It must be valid path segment name. It may not be '.' or '..' and it may not contain '/' and '%' characters. ([#125218](https://github.com/kubernetes/kubernetes/pull/125218), [@carlory](https://github.com/carlory)) [SIG Node]
- Kubeadm: fixed a regression where the JoinConfiguration.discovery.timeout was no longer respected and the value was always hardcoded to "5m" (5 minutes). ([#125480](https://github.com/kubernetes/kubernetes/pull/125480), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]

### Other (Cleanup or Flake)

- Removed generally available feature gate `ReadWriteOncePod`. ([#124329](https://github.com/kubernetes/kubernetes/pull/124329), [@chrishenzie](https://github.com/chrishenzie)) [SIG Storage]

## Dependencies

### Added
_Nothing has changed._

### Changed
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.29.0 → v0.30.3

### Removed
_Nothing has changed._



# v1.31.0-alpha.1


## Downloads for v1.31.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes.tar.gz) | c3d3b7c0f58866a09006b47ba0e7677c95451c0c5b727963ec2bb318fcf0fd94a75f14e51485dacbcf34fab2879325216d9723162e2039d09344ab75b8313fad
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-src.tar.gz) | 16e46516d52f89b9bf623e90bab4d17708b540d67c153c0f81c42a4f6bb335f549b5c451c71701aeeb279ee3f60f1379df98bfab4d24db33a2ff7ef23b70c943

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 219fc2cfcd6da50693eca80209e6d6c7b1331c79c059126766ebdbb5dac56e8efb277bc39d0c32a4d1f4bf51445994c91ce27f291bccdda7859b4be666b2452f
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 054897580442e027c4d0c5c67769e0f98f464470147abb981b200358bcf13b134eac166845350f2e2c8460df3577982f18eafad3be698cfee6e5a4a2e088f0d3
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-linux-386.tar.gz) | a783ba568bbe28e0ddddcbd2c16771f2354786bcc5de4333e9d0a73a1027a8a45c2cc58c69b740db83fec12647e93df2536790df5e191d96dea914986b717ee6
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | f0f39dc1f8cf5dd6029afccae904cd082ed3a4da9283a4506311b0f820e50bdbe9370aaa784f382ec5cbfaa7b115ce34578801080443380f8e606fad225467f0
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 744b69d0b0a40d8fbcb8bd582ee36da3682e189c33a780de01b320cf07eac0b215e6051f6f57ea34b9417423d0d4a42df85d72753226d53b5fe59411b096335d
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | ebec17b4e81bfbd1789e2889435929c38976c5f054d093b964a12cf82c173a1d31c976db51c8a64bf822c17ef4ae41cef1a252bb53143094effe730601e63fe5
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | 0b5602ec8c9c0afafe4f7c05590bdf8176ec158abb8b52e0bea026eb937945afc52aadeb4d1547fff0883b29e1aec0b92fbbae2e950a0cffa870db21674cef9e
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 21b37221c9259e0c7a3fee00f4de20fbebe435755313ed0887d44989e365a67eff0450eda836e93fccf11395c89c9702a17dc494d51633f48c7bb9afe94253c4
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 9e261d3ce6d640e8d43f7761777ea7d62cc0b37e709a96a1e5b691bd7fc6023804dc599edadac351dc9f9107c43bd5d6b962050a3363e5d1037036e4ab51a2ed
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 53606a24ff85e011fd142a2e3b6c8cda058c77afdab6698eb488ab456bf41d299ca442c50482e00535ea6453472d987de6fd75f952febc5a33e46bb5cdf9c0ee
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | f29dd44241d3363eecdcf7063cec5e6516698434c5494e922ee561b3553fbd214075cb0f4832dfadad7a894a3b9df9ee94bb4adb542feda2314d05b1b7b71f78

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | 55b2c9cacb14c2a7b657079e1b2620c0580e3a01d91b0bd3f1e8b1a70e4bb59c4c361eb8aad425734fd579d6e944aedc7695082cb640a3af902dff623a565714
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 24422b969931754c7a72876d1d3ad773bdbdb42bb53ca8d2020b7987a03d20136ad5693c1aa81515b94e3ab71ed486c4b09a9d99b3ef4a7a78d8cd742f7cf9fd
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | 76b6cc096ed38e0d08c1af6ee0715e0a29674eb990ee9675abb3bb9345c70469ca25b62b7babc9afdd6628d1970817d36b66a7b5890572cb0bc9519735c78599
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 4b5a1660e1acfe3e2cb03097608c9c3c7ceedd80c9b71c22ac7572db49598d6e9bff903c8415236947ea1ba14f9595a6bbc178f15959191b92805ce5b01063c3

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 98b402d2cb135af8b2d328ae452fae832e4bfe9e5ab471f277fe134411a46c5493d62def5f5af1259c977bd94b90ce8c8d5e9ba8ee1c7b7fe991316510d09e71
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 052a7ccb8ed370451d883b64cd219b803141eaef4a8498ee45c61d09eff1364b7c4d5785bc8247c9a806dee5887d53abe44e645ada2d45349a0163c3e229decd
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 32a2cc80b367fb6a447d1b674eed220b13e03662f453c155b1752ccef72ccd55503ca73267cf782472e58771a57efc68eee4cb47520e09e6987a7183329d20fa
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | d358de45ae5566b534c9751e7acf0e577e73646d556b444020ee75a731e488ca467df1bfbc5c6a9b3e967f0ea9586bf82657cb22d569a2df69b317671dc6bcae
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.31.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 95c8962439485920c0d50d85ffa037cc4dacaa61392894394759d4d9efb2525d6e1b4e6177c72eed5f55511b6f9c279795601744a1a2da2ee3cb3b518ac31c8a

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.31.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.31.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.31.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.31.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.31.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.31.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.30.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Kubelet flag `--keep-terminated-pod-volumes` was removed.  This flag was deprecated in 2017. ([#122082](https://github.com/kubernetes/kubernetes/pull/122082), [@carlory](https://github.com/carlory)) [SIG Apps, Node, Storage and Testing]
 
## Changes by Kind

### Deprecation

- CephFS volume plugin ( `kubernetes.io/cephfs`) was removed in this release and the `cephfs` volume type became non-functional. Alternative is to use CephFS CSI driver (https://github.com/ceph/ceph-csi/) in your Kubernetes Cluster. A re-deployment of your application is required to use the new driver if you were using `kubernetes.io/cephfs` volume plugin before upgrading cluster version to 1.31+. ([#124544](https://github.com/kubernetes/kubernetes/pull/124544), [@carlory](https://github.com/carlory)) [SIG Node, Scalability, Storage and Testing]
- CephRBD volume plugin ( `kubernetes.io/rbd`) was removed in this release. And its csi migration support was also removed, so the `rbd` volume type became non-functional. Alternative is to use RBD CSI driver (https://github.com/ceph/ceph-csi/) in your Kubernetes Cluster. A re-deployment of your application is required to use the new driver if you were using `kubernetes.io/rbd` volume plugin before upgrading cluster version to 1.31+. ([#124546](https://github.com/kubernetes/kubernetes/pull/124546), [@carlory](https://github.com/carlory)) [SIG Node, Scalability, Scheduling, Storage and Testing]
- Kube-scheduler deprecated all non-csi volumelimit plugins and removed those from defaults plugins. 
  - AzureDiskLimits
  - CinderLimits
  - EBSLimits
  - GCEPDLimits
  
  The NodeVolumeLimits plugin can handle the same functionality as the above plugins since the above volume types are migrated to CSI.
  Please remove those plugins and replace them with the NodeVolumeLimits plugin if you explicitly use those plugins in the scheduler config.
  Those plugins will be removed in the release 1.32. ([#124500](https://github.com/kubernetes/kubernetes/pull/124500), [@carlory](https://github.com/carlory)) [SIG Scheduling and Storage]
- Kubeadm: deprecated the kubeadm `RootlessControlPlane` feature gate (previously alpha), given that the core K8s `UserNamespacesSupport` feature gate graduated to Beta in 1.30.
  Once core Kubernetes support for user namespaces is generally available and kubeadm has started to support running the control plane in userns pods, the kubeadm `RootlessControlPlane` feature gate will be removed entirely.
  Until kubeadm supports the userns functionality out of the box, users can continue using the deprecated  `RootlessControlPlane` feature gate, or  opt-in `UserNamespacesSupport` by using kubeadm patches on the static pod manifests. ([#124997](https://github.com/kubernetes/kubernetes/pull/124997), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: mark the sub-phase of 'init kubelet-finilize' called 'experimental-cert-rotation' as deprecated and print a warning if it is used directly; it will be removed in a future release. Add a replacement sub-phase 'enable-client-cert-rotation'. ([#124419](https://github.com/kubernetes/kubernetes/pull/124419), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Remove k8s.io/legacy-cloud-providers from staging ([#124767](https://github.com/kubernetes/kubernetes/pull/124767), [@carlory](https://github.com/carlory)) [SIG API Machinery, Cloud Provider and Release]
- Removed legacy cloud provider integration code (undoing a previous reverted commit) ([#124886](https://github.com/kubernetes/kubernetes/pull/124886), [@carlory](https://github.com/carlory)) [SIG Cloud Provider and Release]

### API Change

- Added the feature gates `StrictCostEnforcementForVAP` and `StrictCostEnforcementForWebhooks` to enforce the strct cost calculation for CEL extended libraries. It is strongly recommended to turn on the feature gates as early as possible. ([#124675](https://github.com/kubernetes/kubernetes/pull/124675), [@cici37](https://github.com/cici37)) [SIG API Machinery, Auth, Node and Testing]
- Component-base/logs: when compiled with Go >= 1.21, component-base will automatically configure the slog default logger together with initializing klog. ([#120696](https://github.com/kubernetes/kubernetes/pull/120696), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Storage and Testing]
- DRA: client-side validation of a ResourceHandle would have accepted a missing DriverName, whereas server-side validation then would have raised an error. ([#124075](https://github.com/kubernetes/kubernetes/pull/124075), [@pohly](https://github.com/pohly)) [SIG Apps]
- Fix Deep Copy issue in getting controller reference ([#124116](https://github.com/kubernetes/kubernetes/pull/124116), [@HiranmoyChowdhury](https://github.com/HiranmoyChowdhury)) [SIG API Machinery and Release]
- Fix the comment for the Job's managedBy field ([#124793](https://github.com/kubernetes/kubernetes/pull/124793), [@mimowo](https://github.com/mimowo)) [SIG API Machinery and Apps]
- Fixes a 1.30.0 regression in openapi descriptions of imagePullSecrets and hostAliases fields to mark the fields used as keys in those lists as either defaulted or required. ([#124553](https://github.com/kubernetes/kubernetes/pull/124553), [@pmalek](https://github.com/pmalek)) [SIG API Machinery]
- Graduate MatchLabelKeys/MismatchLabelKeys feature in PodAffinity/PodAntiAffinity to Beta ([#123638](https://github.com/kubernetes/kubernetes/pull/123638), [@sanposhiho](https://github.com/sanposhiho)) [SIG API Machinery, Apps, Scheduling and Testing]
- Graduated the `DisableNodeKubeProxyVersion` feature gate to beta. By default, the kubelet no longer attempts to set the `.status.kubeProxyVersion` field for its associated Node. ([#123845](https://github.com/kubernetes/kubernetes/pull/123845), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG API Machinery, Cloud Provider, Network, Node and Testing]
- Improved scheduling performance when many nodes, and prefilter returns 1-2 nodes (e.g. daemonset)
  
  For developers of out-of-tree PostFilter plugins, note that the semantics of NodeToStatusMap are changing: A node with an absent value in the NodeToStatusMap should be interpreted as having an UnschedulableAndUnresolvable status ([#125197](https://github.com/kubernetes/kubernetes/pull/125197), [@gabesaba](https://github.com/gabesaba)) [SIG Scheduling]
- K8s.io/apimachinery/pkg/util/runtime: new calls support handling panics and errors in the context where they occur. `PanicHandlers` and `ErrorHandlers` now must accept a context parameter for that. Log output is structured instead of unstructured. ([#121970](https://github.com/kubernetes/kubernetes/pull/121970), [@pohly](https://github.com/pohly)) [SIG API Machinery and Instrumentation]
- Kube-apiserver: the `--encryption-provider-config` file is now loaded with strict deserialization, which fails if the config file contains duplicate or unknown fields. This protects against accidentally running with config files that are malformed, mis-indented, or have typos in field names, and getting unexpected behavior. When `--encryption-provider-config-automatic-reload` is used, new encryption config files that contain typos after the kube-apiserver is running are treated as invalid and the last valid config is used. ([#124912](https://github.com/kubernetes/kubernetes/pull/124912), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]
- Kube-controller-manager removes deprecated command flags: --volume-host-cidr-denylist and --volume-host-allow-local-loopback ([#124017](https://github.com/kubernetes/kubernetes/pull/124017), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Cloud Provider and Storage]
- Kube-controller-manager: the `horizontal-pod-autoscaler-upscale-delay` and `horizontal-pod-autoscaler-downscale-delay` flags have been removed (deprecated and non-functional since v1.12) ([#124948](https://github.com/kubernetes/kubernetes/pull/124948), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery, Apps and Autoscaling]
- Support fine-grained supplemental groups policy (KEP-3619), which enables fine-grained control for supplementary groups in the first container processes. You can choose whether to include groups defined in the container image(/etc/groups) for the container's primary uid or not. ([#117842](https://github.com/kubernetes/kubernetes/pull/117842), [@everpeace](https://github.com/everpeace)) [SIG API Machinery, Apps and Node]
- The kube-proxy nodeportAddresses / --nodeport-addresses option now
  accepts the value "primary", meaning to only listen for NodePort connections
  on the node's primary IPv4 and/or IPv6 address (according to the Node object).
  This is strongly recommended, if you were not previously using
  --nodeport-addresses, to avoid surprising behavior.
  
  (This behavior is enabled by default with the nftables backend; you would
  need to explicitly request `--nodeport-addresses 0.0.0.0/0,::/0` there to get
  the traditional "listen on all interfaces" behavior.) ([#123105](https://github.com/kubernetes/kubernetes/pull/123105), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Network and Windows]

### Feature

- Add `--keep-*` flags to `kubectl debug`, which enables to control the removal of probes, labels, annotations and initContainers from copy pod. ([#123149](https://github.com/kubernetes/kubernetes/pull/123149), [@mochizuki875](https://github.com/mochizuki875)) [SIG CLI and Testing]
- Add apiserver.latency.k8s.io/apf-queue-wait annotation to the audit log to record the time spent waiting in apf queue ([#123919](https://github.com/kubernetes/kubernetes/pull/123919), [@hakuna-matatah](https://github.com/hakuna-matatah)) [SIG API Machinery]
- Add the` WatchList` method to the `rest client` in `client-go`. When used, it establishes a stream to obtain a consistent snapshot of data from the server. This method is meant to be used by the generated client. ([#122657](https://github.com/kubernetes/kubernetes/pull/122657), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Added `cri-client` staging repository. ([#123797](https://github.com/kubernetes/kubernetes/pull/123797), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery, Node, Release and Testing]
- Added flag to `kubectl logs` called `--all-pods` to get all pods from a object that uses a pod selector. ([#124732](https://github.com/kubernetes/kubernetes/pull/124732), [@cmwylie19](https://github.com/cmwylie19)) [SIG CLI and Testing]
- Added ports autocompletion for kubectl port-foward command ([#124683](https://github.com/kubernetes/kubernetes/pull/124683), [@TessaIO](https://github.com/TessaIO)) [SIG CLI]
- Added support for building Windows kube-proxy container image.
  A container image for kube-proxy on Windows can now be built with the command
  `make release-images KUBE_BUILD_WINDOWS=y`.
  The Windows kube-proxy image can be used with Windows Host Process Containers. ([#109939](https://github.com/kubernetes/kubernetes/pull/109939), [@claudiubelu](https://github.com/claudiubelu)) [SIG Windows]
- Adds completion for `kubectl set image`. ([#124592](https://github.com/kubernetes/kubernetes/pull/124592), [@ah8ad3](https://github.com/ah8ad3)) [SIG CLI]
- Allow creating ServiceAccount tokens bound to Node objects.
  This allows users to bind a service account token's validity to a named Node object, similar to Pod bound tokens.
  Use with `kubectl create token <serviceaccount-name> --bound-object-kind=Node --bound-object-node=<node-name>`. ([#125238](https://github.com/kubernetes/kubernetes/pull/125238), [@munnerz](https://github.com/munnerz)) [SIG Auth and CLI]
- CEL default compatibility environment version to updated to 1.30 so that the extended libraries added before 1.30 is available to use. ([#124779](https://github.com/kubernetes/kubernetes/pull/124779), [@cici37](https://github.com/cici37)) [SIG API Machinery]
- CEL expressions and `additionalProperties` are now allowed to be used under nested quantifiers in CRD schemas ([#124381](https://github.com/kubernetes/kubernetes/pull/124381), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- CEL: add name formats library ([#123572](https://github.com/kubernetes/kubernetes/pull/123572), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- Checking etcd version to warn about deprecated etcd versions if `ConsistentListFromCache` is enabled. ([#124612](https://github.com/kubernetes/kubernetes/pull/124612), [@ah8ad3](https://github.com/ah8ad3)) [SIG API Machinery]
- Client-go/reflector: warns when the bookmark event for initial events hasn't been received ([#124614](https://github.com/kubernetes/kubernetes/pull/124614), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Custom resource field selectors are now in beta and enabled by default. Check out https://github.com/kubernetes/enhancements/issues/4358 for more details. ([#124681](https://github.com/kubernetes/kubernetes/pull/124681), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Auth and Testing]
- Dependencies: start using registry.k8s.io/pause:3.10 ([#125112](https://github.com/kubernetes/kubernetes/pull/125112), [@neolit123](https://github.com/neolit123)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Node, Release, Testing and Windows]
- Graduated support for CDI device IDs to general availability. The `DevicePluginCDIDevices` feature gate is now enabled unconditionally. ([#123315](https://github.com/kubernetes/kubernetes/pull/123315), [@bart0sh](https://github.com/bart0sh)) [SIG Node]
- Kube-apiserver: http/2 serving can be disabled with a `--disable-http2-serving` flag ([#122176](https://github.com/kubernetes/kubernetes/pull/122176), [@slashpai](https://github.com/slashpai)) [SIG API Machinery]
- Kube-proxy's nftables mode (--proxy-mode=nftables) is now beta and available by default.
  
  FIXME ADD MORE HERE BEFORE THE RELEASE, DOCS LINKS AND STUFF ([#124383](https://github.com/kubernetes/kubernetes/pull/124383), [@danwinship](https://github.com/danwinship)) [SIG Cloud Provider and Network]
- Kube-scheduler implements scheduling hints for the CSILimit plugin.
  The scheduling hints allow the scheduler to retry scheduling a Pod that was previously rejected by the CSILimit plugin if a deleted pod has a PVC from the same driver. ([#121508](https://github.com/kubernetes/kubernetes/pull/121508), [@utam0k](https://github.com/utam0k)) [SIG Scheduling and Storage]
- Kube-scheduler implements scheduling hints for the InterPodAffinity plugin.
  The scheduling hints allow the scheduler to retry scheduling a Pod
  that was previously rejected by the InterPodAffinity plugin if create/delete/update a related Pod or a node which matches the pod affinity. ([#122471](https://github.com/kubernetes/kubernetes/pull/122471), [@nayihz](https://github.com/nayihz)) [SIG Scheduling and Testing]
- Kubeadm: during "upgrade" , if the "etcd.yaml" static pod does not need upgrade, still consider rotating the etcd certificates and restarting the etcd static pod if the "kube-apiserver.yaml" manifest is to be upgraded and if certificate renewal is not disabled. ([#124688](https://github.com/kubernetes/kubernetes/pull/124688), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: enhance the "patches" functionality to be able to patch coredns deployment. The new patch target is called "corednsdeployment" (e.g. patch file "corednsdeployment+json.json"). This makes it possible to apply custom patches to coredns deployment during "init" and "upgrade". ([#124820](https://github.com/kubernetes/kubernetes/pull/124820), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: mark the flag "--experimental-output' as deprecated (it will be removed in a future release) and add a new flag '--output" that serves the same purpose. Affected commands are - "kubeadm config images list", "kubeadm token list", "kubeadm upgade plan", "kubeadm certs check-expiration". ([#124393](https://github.com/kubernetes/kubernetes/pull/124393), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: switch to using the new etcd endpoints introduced in 3.5.11 - /livez (for liveness probe) and /readyz (for readyness and startup probe). With this change it is no longer possible to deploy a custom etcd version older than 3.5.11 with kubeadm 1.31. If so, please upgrade. ([#124465](https://github.com/kubernetes/kubernetes/pull/124465), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: switched kubeadm to start using the CRI client library instead of shelling out of the `crictl` binary
  for actions against a CRI endpoint. The kubeadm deb/rpm packages will continue to install the `cri-tools`
  package for one more release, but in you must adapt your scripts to install `crictl` manually from
  https://github.com/kubernetes-sigs/cri-tools/releases or a different location.
  
  The `kubeadm` package will stop depending on the `cri-tools` package in Kubernetes 1.32, which means that
  installing `kubeadm` will no longer automatically ensure installation of `crictl`. ([#124685](https://github.com/kubernetes/kubernetes/pull/124685), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cluster Lifecycle]
- Kubeadm: use output/v1alpha3 to print structural output for the commands "kubeadm config images list" and "kubeadm token list". ([#124464](https://github.com/kubernetes/kubernetes/pull/124464), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubelet server can now dynamically load certificate files ([#124574](https://github.com/kubernetes/kubernetes/pull/124574), [@zhangweikop](https://github.com/zhangweikop)) [SIG Auth and Node]
- Kubelet will not restart the container when fields other than image in the pod spec change. ([#124220](https://github.com/kubernetes/kubernetes/pull/124220), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Node]
- Kubemark: adds two flags, `--kube-api-qps` and `--kube-api-burst` ([#124147](https://github.com/kubernetes/kubernetes/pull/124147), [@devincd](https://github.com/devincd)) [SIG Scalability]
- Kubernetes is now built with go 1.22.3 ([#124828](https://github.com/kubernetes/kubernetes/pull/124828), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built with go 1.22.4 ([#125363](https://github.com/kubernetes/kubernetes/pull/125363), [@cpanato](https://github.com/cpanato)) [SIG Architecture, Cloud Provider, Release, Storage and Testing]
- Pause: add a -v flag to the Windows variant of the pause binary, which prints the version of pause and exits. The Linux pause already has the flag. ([#125067](https://github.com/kubernetes/kubernetes/pull/125067), [@neolit123](https://github.com/neolit123)) [SIG Windows]
- Promoted `generateName` retries to beta, and made the `NameGenerationRetries` feature gate
  enabled by default.
  You can read https://kep.k8s.io/4420 for more details. ([#124673](https://github.com/kubernetes/kubernetes/pull/124673), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Scheduler changes its logic of calculating `evaluatedNodes` from "contains the number of nodes that filtered out by PreFilterResult and Filter plugins" to "the number of nodes filtered out by Filter plugins only". ([#124735](https://github.com/kubernetes/kubernetes/pull/124735), [@AxeZhan](https://github.com/AxeZhan)) [SIG Scheduling]
- Services implement a field selector for the ClusterIP and Type fields.
  Kubelet uses the fieldselector on Services to avoid watching for Headless Services and reduce the memory consumption. ([#123905](https://github.com/kubernetes/kubernetes/pull/123905), [@aojea](https://github.com/aojea)) [SIG Apps, Node and Testing]
- The iptables mode of kube-proxy now tracks accepted packets that are destined for node-ports on localhost by introducing `kubeproxy_iptables_localhost_nodeports_accepted_packets_total` metric.
  This will help users to identify if they rely on iptables.localhostNodePorts feature and ulitmately help them to migrate from iptables to nftables. ([#125015](https://github.com/kubernetes/kubernetes/pull/125015), [@aroradaman](https://github.com/aroradaman)) [SIG Instrumentation, Network and Testing]
- The iptables mode of kube-proxy now tracks packets that are wrongfully marked invalid by conntrack and subsequently dropped by introducing `kubeproxy_iptables_ct_state_invalid_dropped_packets_total` metric ([#122812](https://github.com/kubernetes/kubernetes/pull/122812), [@aroradaman](https://github.com/aroradaman)) [SIG Instrumentation, Network and Testing]
- The name of CEL optional type has been changed from `optional` to `optional_type`. ([#124328](https://github.com/kubernetes/kubernetes/pull/124328), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Network and Node]
- The scheduler implements QueueingHint in TaintToleration plugin, which enhances the throughput of scheduling. ([#124287](https://github.com/kubernetes/kubernetes/pull/124287), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- The sidecar finish time will be accounted when calculating the job's finish time. ([#124942](https://github.com/kubernetes/kubernetes/pull/124942), [@AxeZhan](https://github.com/AxeZhan)) [SIG Apps]
- This PR adds tracing support to the kubelet's read-only endpoint, which currently does not have tracing. It makes use the WithPublicEndpoint option to prevent callers from influencing sampling decisions. ([#121770](https://github.com/kubernetes/kubernetes/pull/121770), [@frzifus](https://github.com/frzifus)) [SIG Node]
- Users can traverse all the pods that are in the scheduler and waiting in the permit stage through method `IterateOverWaitingPods`. In other words,  all waitingPods in scheduler can be obtained from any profiles. Before this commit, each profile could only obtain waitingPods within that profile. ([#124926](https://github.com/kubernetes/kubernetes/pull/124926), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]

### Failing Test

- Pkg k8s.io/apiserver/pkg/storage/cacher, method (*Cacher) Wait(context.Context) error ([#125450](https://github.com/kubernetes/kubernetes/pull/125450), [@mauri870](https://github.com/mauri870)) [SIG API Machinery]
- Revert "remove legacycloudproviders from staging" ([#124864](https://github.com/kubernetes/kubernetes/pull/124864), [@carlory](https://github.com/carlory)) [SIG Release]

### Bug or Regression

- .status.terminating field now gets correctly tracked for deleted active Pods when a Job fails. ([#125175](https://github.com/kubernetes/kubernetes/pull/125175), [@dejanzele](https://github.com/dejanzele)) [SIG Apps and Testing]
- Added an extra line between two different key value pairs under data when running kubectl describe configmap ([#123597](https://github.com/kubernetes/kubernetes/pull/123597), [@siddhantvirus](https://github.com/siddhantvirus)) [SIG CLI]
- Allow parameter to be set along with proto file path ([#124281](https://github.com/kubernetes/kubernetes/pull/124281), [@fulviodenza](https://github.com/fulviodenza)) [SIG API Machinery]
- Cel: converting a quantity value into a quantity value failed. ([#123669](https://github.com/kubernetes/kubernetes/pull/123669), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- Client-go/tools/record.Broadcaster: fixed automatic shutdown on WithContext cancellation ([#124635](https://github.com/kubernetes/kubernetes/pull/124635), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- Do not remove the "batch.kubernetes.io/job-tracking" finalizer from a Pod, in a corner
  case scenario, when the Pod is controlled by an API object which is not a batch Job
  (e.g. when the Pod is controlled by a custom CRD). ([#124798](https://github.com/kubernetes/kubernetes/pull/124798), [@mimowo](https://github.com/mimowo)) [SIG Apps and Testing]
- Drop additional rule requirement (cronjobs/finalizers) in the roles who use kubectl create cronjobs to be backwards compatible ([#124883](https://github.com/kubernetes/kubernetes/pull/124883), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Emition of RecreatingFailedPod and RecreatingTerminatedPod events has been removed from stateful set lifecycle. ([#123809](https://github.com/kubernetes/kubernetes/pull/123809), [@atiratree](https://github.com/atiratree)) [SIG Apps and Testing]
- Endpointslices mirrored from Endpoints by the EndpointSliceMirroring controller were not reconciled if modified ([#124131](https://github.com/kubernetes/kubernetes/pull/124131), [@zyjhtangtang](https://github.com/zyjhtangtang)) [SIG Apps and Network]
- Ensure daemonset controller to count old unhealthy pods towards max unavailable budget ([#123233](https://github.com/kubernetes/kubernetes/pull/123233), [@marshallbrekka](https://github.com/marshallbrekka)) [SIG Apps]
- Fix "-kube-test-repo-list" e2e flag may not take effect ([#123587](https://github.com/kubernetes/kubernetes/pull/123587), [@huww98](https://github.com/huww98)) [SIG API Machinery, Apps, Autoscaling, CLI, Network, Node, Scheduling, Storage, Testing and Windows]
- Fix a race condition in kube-controller-manager and scheduler caused by a bug in transforming informer happening when objects were accessed during Resync operation by making the transforming function idempotent. ([#124352](https://github.com/kubernetes/kubernetes/pull/124352), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Scheduling]
- Fix a race condition in transforming informer happening when objects were accessed during Resync operation ([#124344](https://github.com/kubernetes/kubernetes/pull/124344), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery]
- Fix kubelet on Windows fails if a pod has SecurityContext with RunAsUser ([#125040](https://github.com/kubernetes/kubernetes/pull/125040), [@carlory](https://github.com/carlory)) [SIG Storage, Testing and Windows]
- Fix throughput when scheduling daemonset pods to reach 300 pods/s, if the configured qps allows it. ([#124714](https://github.com/kubernetes/kubernetes/pull/124714), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Fix: the resourceclaim controller forgot to wait for podSchedulingSynced and templatesSynced ([#124589](https://github.com/kubernetes/kubernetes/pull/124589), [@carlory](https://github.com/carlory)) [SIG Apps and Node]
- Fixed EDITOR/KUBE_EDITOR with double-quoted paths with spaces when on Windows cmd.exe. ([#112104](https://github.com/kubernetes/kubernetes/pull/112104), [@oldium](https://github.com/oldium)) [SIG CLI and Windows]
- Fixed a bug in the JSON frame reader that could cause it to retain a reference to the underlying array of the byte slice passed to Read. ([#123620](https://github.com/kubernetes/kubernetes/pull/123620), [@benluddy](https://github.com/benluddy)) [SIG API Machinery]
- Fixed a bug in the scheduler where it would crash when prefilter returns a non-existent node. ([#124933](https://github.com/kubernetes/kubernetes/pull/124933), [@AxeZhan](https://github.com/AxeZhan)) [SIG Scheduling and Testing]
- Fixed a bug where `kubectl describe` incorrectly displayed NetworkPolicy port ranges
  (showing only the starting port). ([#123316](https://github.com/kubernetes/kubernetes/pull/123316), [@jcaamano](https://github.com/jcaamano)) [SIG CLI]
- Fixed a regression where `kubelet --hostname-override` no longer worked
  correctly with an external cloud provider. ([#124516](https://github.com/kubernetes/kubernetes/pull/124516), [@danwinship](https://github.com/danwinship)) [SIG Node]
- Fixed an issue that prevents the linking of trace spans for requests that are proxied through kube-aggregator. ([#124189](https://github.com/kubernetes/kubernetes/pull/124189), [@toddtreece](https://github.com/toddtreece)) [SIG API Machinery]
- Fixed bug where kubectl get with --sort-by flag does not sort strings alphanumerically. ([#124514](https://github.com/kubernetes/kubernetes/pull/124514), [@brianpursley](https://github.com/brianpursley)) [SIG CLI]
- Fixed the format of the error indicating that a user does not have permission on the object referenced by paramRef in ValidatingAdmissionPolicyBinding. ([#124653](https://github.com/kubernetes/kubernetes/pull/124653), [@m1kola](https://github.com/m1kola)) [SIG API Machinery]
- Fixes a bug where hard evictions due to resource pressure would let the pod have the full termination grace period, instead of shutting down instantly. This bug also affected force deleted pods. Both cases now get a termination grace period of 1 second. ([#124063](https://github.com/kubernetes/kubernetes/pull/124063), [@olyazavr](https://github.com/olyazavr)) [SIG Node]
- Fixes a missing `status.` prefix on custom resource validation error messages. ([#123822](https://github.com/kubernetes/kubernetes/pull/123822), [@JoelSpeed](https://github.com/JoelSpeed)) [SIG API Machinery]
- Improved scheduling latency when many gated pods ([#124618](https://github.com/kubernetes/kubernetes/pull/124618), [@gabesaba](https://github.com/gabesaba)) [SIG Scheduling and Testing]
- Job: Fix a bug that the SuccessCriteriaMet could be added to the Job with successPolicy regardless of the featureGate enabling ([#125429](https://github.com/kubernetes/kubernetes/pull/125429), [@tenzen-y](https://github.com/tenzen-y)) [SIG Apps]
- Kube-apiserver: fixes a 1.28 regression printing pods with invalid initContainer status ([#124906](https://github.com/kubernetes/kubernetes/pull/124906), [@liggitt](https://github.com/liggitt)) [SIG Node]
- Kubeadm: allow 'kubeadm init phase certs sa' to accept the '--config' flag. ([#125396](https://github.com/kubernetes/kubernetes/pull/125396), [@Kavinraja-G](https://github.com/Kavinraja-G)) [SIG Cluster Lifecycle]
- Kubeadm: don't mount /etc/pki in kube-apisever and kube-controller-manager pods as an additional Linux system CA location. Mount /etc/pki/ca-trust and /etc/pki/tls/certs instead. /etc/ca-certificate, /usr/share/ca-certificates, /usr/local/share/ca-certificates and /etc/ssl/certs continue to be mounted. ([#124361](https://github.com/kubernetes/kubernetes/pull/124361), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: during kubelet health checks, respect the healthz address:port configured in the KubeletConfiguration instead of hardcoding localhost:10248. ([#125265](https://github.com/kubernetes/kubernetes/pull/125265), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: during the preflight check "CreateJob" of "kubeadm upgrade", check if there are no nodes where a Pod can schedule. If there are none, show a warning and skip this preflight check. This can happen in single node clusters where the only node was drained. ([#124503](https://github.com/kubernetes/kubernetes/pull/124503), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fix a regression where the KubeletConfiguration is not properly downloaded during "kubeadm upgrade" commands from the kube-system/kubelet-config ConfigMap, resulting in the local '/var/lib/kubelet/config.yaml' file being written as a defaulted config. ([#124480](https://github.com/kubernetes/kubernetes/pull/124480), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: fixed a bug where the PublicKeysECDSA feature gate was not respected when generating kubeconfig files. ([#125388](https://github.com/kubernetes/kubernetes/pull/125388), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: improve the "IsPriviledgedUser" preflight check to not fail on certain Windows setups. ([#124665](https://github.com/kubernetes/kubernetes/pull/124665), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: stop storing the ResolverConfig in the global KubeletConfiguration and instead set it dynamically for each node ([#124038](https://github.com/kubernetes/kubernetes/pull/124038), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubectl support both:
  - kubectl create secret docker-registry <NAME> --from-file=<path/to/.docker/config.json>
  - kubectl create secret docker-registry <NAME> --from-file=.dockerconfigjson=<path/to/.docker/config.json> ([#119589](https://github.com/kubernetes/kubernetes/pull/119589), [@carlory](https://github.com/carlory)) [SIG CLI]
- Kubectl: Show the Pod phase in the STATUS column as 'Failed' or 'Succeeded' when the Pod is terminated ([#122038](https://github.com/kubernetes/kubernetes/pull/122038), [@lowang-bh](https://github.com/lowang-bh)) [SIG CLI]
- Kubelet no longer crashes when a DRA driver returns a nil as part of the Node(Un)PrepareResources response instead of an empty struct (did not affect drivers written in Go, first showed up with a driver written in Rust). ([#124091](https://github.com/kubernetes/kubernetes/pull/124091), [@bitoku](https://github.com/bitoku)) [SIG Node]
- Make kubectl find `kubectl-create-subcommand` plugins also when positional arguments exists, e.g. `kubectl create subcommand arg`. ([#124123](https://github.com/kubernetes/kubernetes/pull/124123), [@sttts](https://github.com/sttts)) [SIG CLI]
- Removed admission plugin PersistentVolumeLabel. Please use https://github.com/kubernetes-sigs/cloud-pv-admission-labeler instead if you need a similar functionality. ([#124505](https://github.com/kubernetes/kubernetes/pull/124505), [@jsafrane](https://github.com/jsafrane)) [SIG API Machinery, Auth and Storage]
- StatefulSet autodelete will respect controlling owners on PVC claims as described in https://github.com/kubernetes/enhancements/pull/4375 ([#122499](https://github.com/kubernetes/kubernetes/pull/122499), [@mattcary](https://github.com/mattcary)) [SIG Apps and Testing]
- The "fake" clients generated by `client-gen` now have the same semantics on
  error as the real clients; in particular, a failed Get(), Create(), etc, no longer
  returns `nil`. (It now returns a pointer to a zero-valued object, like the real
  clients do.) This will break some downstream unit tests that were testing
  `result == nil` rather than `err != nil`, and in some cases may expose bugs
  in the underlying code that were hidden by the incorrect unit tests. ([#122892](https://github.com/kubernetes/kubernetes/pull/122892), [@danwinship](https://github.com/danwinship)) [SIG API Machinery, Auth, Cloud Provider, Instrumentation and Storage]
- The Service LoadBalancer controller was not correctly considering the service.Status new IPMode field and excluding the Ports when comparing if the status has changed, causing that changes in these fields may not update the service.Status correctly ([#125225](https://github.com/kubernetes/kubernetes/pull/125225), [@aojea](https://github.com/aojea)) [SIG Apps, Cloud Provider and Network]
- The nftables kube-proxy mode now has its own metrics rather than reporting
  metrics with "iptables" in their names. ([#124557](https://github.com/kubernetes/kubernetes/pull/124557), [@danwinship](https://github.com/danwinship)) [SIG Network and Windows]
- Updated description of default values for --healthz-bind-address and --metrics-bind-address parameters ([#123545](https://github.com/kubernetes/kubernetes/pull/123545), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Network]

### Other (Cleanup or Flake)

- ACTION-REQUIRED: DRA drivers using the v1alpha2 kubelet gRPC API are no longer supported and need to be updated. ([#124316](https://github.com/kubernetes/kubernetes/pull/124316), [@pohly](https://github.com/pohly)) [SIG Node and Testing]
- Build etcd image v3.5.13 ([#124026](https://github.com/kubernetes/kubernetes/pull/124026), [@liangyuanpeng](https://github.com/liangyuanpeng)) [SIG API Machinery and Etcd]
- Build etcd image v3.5.14 ([#125235](https://github.com/kubernetes/kubernetes/pull/125235), [@humblec](https://github.com/humblec)) [SIG API Machinery]
- CSI spec support has been lifted to v1.9.0 in this release ([#125150](https://github.com/kubernetes/kubernetes/pull/125150), [@humblec](https://github.com/humblec)) [SIG Storage and Testing]
- E2e.test and e2e_node.test: tests which depend on alpha or beta feature gates now have `Feature:Alpha` or `Feature:Beta` as Ginkgo labels. The inline text is `[Alpha]` or `[Beta]`, as before. ([#124350](https://github.com/kubernetes/kubernetes/pull/124350), [@pohly](https://github.com/pohly)) [SIG Testing]
- Etcd: Update to v3.5.13 ([#124027](https://github.com/kubernetes/kubernetes/pull/124027), [@liangyuanpeng](https://github.com/liangyuanpeng)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- Expose apiserver_watch_cache_resource_version metric to simplify debugging problems with watchcache. ([#125377](https://github.com/kubernetes/kubernetes/pull/125377), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Instrumentation]
- Fixed a typo in the help text for the pod_scheduling_sli_duration_seconds metric in kube-scheduler ([#124221](https://github.com/kubernetes/kubernetes/pull/124221), [@arturhoo](https://github.com/arturhoo)) [SIG Instrumentation, Scheduling and Testing]
- Job-controller: the `JobReadyPods` feature flag has been removed (deprecated since v1.31) ([#125168](https://github.com/kubernetes/kubernetes/pull/125168), [@kaisoz](https://github.com/kaisoz)) [SIG Apps]
- Kubeadm: improve the warning message about the NodeSwap check which kubeadm performs on preflight. ([#125157](https://github.com/kubernetes/kubernetes/pull/125157), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: only enable the klog flags that are still supported for kubeadm, rather than hiding the unwanted flags. This means that the previously unrecommended hidden flags about klog (including `--alsologtostderr`, `--log-backtrace-at`, `--log-dir`, `--logtostderr`, `--log-file`, `--log-file-max-size`, `--one-output`, `--skip-log-headers`, `--stderrthreshold` and `--vmodule`) are no longer allowed to be used. ([#125179](https://github.com/kubernetes/kubernetes/pull/125179), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: remove the EXPERIMENTAL tag from the phase "kubeadm join control-plane-prepare download-certs". ([#124374](https://github.com/kubernetes/kubernetes/pull/124374), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: remove the deprecated and NO-OP "kubeadm join control-plane-join update-status"  phase. ([#124373](https://github.com/kubernetes/kubernetes/pull/124373), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: removed the deprecated output.kubeadm.k8s.io/v1alpha2 API for structured output. Please use v1alpha3 instead. ([#124496](https://github.com/kubernetes/kubernetes/pull/124496), [@carlory](https://github.com/carlory)) [SIG Cluster Lifecycle]
- Kubeadm: the deprecated `UpgradeAddonsBeforeControlPlane` featuregate has been removed, upgrade of the CoreDNS and kube-proxy addons will not be triggered until all the control plane instances have been upgraded. ([#124715](https://github.com/kubernetes/kubernetes/pull/124715), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Kubeadm: the global --rootfs flag is now considered non-experimental. ([#124375](https://github.com/kubernetes/kubernetes/pull/124375), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl describe service and ingress will now use endpointslices instead of endpoints ([#124598](https://github.com/kubernetes/kubernetes/pull/124598), [@aroradaman](https://github.com/aroradaman)) [SIG CLI and Network]
- Kubelet flags `--iptables-masquerade-bit` and `--iptables-drop-bit` were deprecated in v1.28 and have now been removed entirely. ([#122363](https://github.com/kubernetes/kubernetes/pull/122363), [@carlory](https://github.com/carlory)) [SIG Network and Node]
- Migrated the pkg/proxy to use [contextual logging](https://k8s.io/docs/concepts/cluster-administration/system-logs/#contextual-logging). ([#122979](https://github.com/kubernetes/kubernetes/pull/122979), [@fatsheep9146](https://github.com/fatsheep9146)) [SIG Network and Scalability]
- Moved remote CRI implementation from kubelet to `k8s.io/cri-client` repository. ([#124634](https://github.com/kubernetes/kubernetes/pull/124634), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node, Release and Testing]
- Remove GA ServiceNodePortStaticSubrange feature gate ([#124738](https://github.com/kubernetes/kubernetes/pull/124738), [@xuzhenglun](https://github.com/xuzhenglun)) [SIG Network]
- Removed generally available feature gate `CSINodeExpandSecret`. ([#124462](https://github.com/kubernetes/kubernetes/pull/124462), [@carlory](https://github.com/carlory)) [SIG Storage]
- Removed generally available feature gate `ConsistentHTTPGetHandlers`. ([#124463](https://github.com/kubernetes/kubernetes/pull/124463), [@carlory](https://github.com/carlory)) [SIG Node]
- Removes `ENABLE_CLIENT_GO_WATCH_LIST_ALPHA` environmental variable from the reflector.
  To activate the feature set `KUBE_FEATURE_WatchListClient` environmental variable or a corresponding command line option (this works only binaries that explicitly expose it). ([#122791](https://github.com/kubernetes/kubernetes/pull/122791), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Removing the last remaining in-tree gcp cloud provider and credential provider. Please use the external cloud provider and credential provider from https://github.com/kubernetes/cloud-provider-gcp instead. ([#124519](https://github.com/kubernetes/kubernetes/pull/124519), [@dims](https://github.com/dims)) [SIG API Machinery, Apps, Auth, Autoscaling, Cloud Provider, Instrumentation, Network, Node, Scheduling, Storage and Testing]
- Scheduler framework: PreBind implementations are now allowed to return Pending and Unschedulable status codes. ([#125360](https://github.com/kubernetes/kubernetes/pull/125360), [@pohly](https://github.com/pohly)) [SIG Scheduling]
- The feature gate "DefaultHostNetworkHostPortsInPodTemplates" has been removed.  This behavior was deprecated in v1.28, and has had no reports of trouble since. ([#124417](https://github.com/kubernetes/kubernetes/pull/124417), [@thockin](https://github.com/thockin)) [SIG Apps]
- The feature gate "SkipReadOnlyValidationGCE" has been removed.  This gate has been active for 2 releases with no reports of issues (and was such a niche thing, we didn't expect any). ([#124210](https://github.com/kubernetes/kubernetes/pull/124210), [@thockin](https://github.com/thockin)) [SIG Apps]
- The kube-scheduler exposes /livez and /readz for health checks that are in compliance with https://kubernetes.io/docs/reference/using-api/health-checks/#api-endpoints-for-health ([#118148](https://github.com/kubernetes/kubernetes/pull/118148), [@linxiulei](https://github.com/linxiulei)) [SIG API Machinery, Scheduling and Testing]
- The kubelet is no longer able to recover from device manager state file older than 1.20. If the proper recommended upgrade flow is followed, there should be no issue. ([#123398](https://github.com/kubernetes/kubernetes/pull/123398), [@ffromani](https://github.com/ffromani)) [SIG Node and Testing]
- Update CNI Plugins to v1.5.0 ([#125113](https://github.com/kubernetes/kubernetes/pull/125113), [@bzsuni](https://github.com/bzsuni)) [SIG Cloud Provider, Network, Node and Testing]
- Updated cni-plugins to v1.4.1. ([#123894](https://github.com/kubernetes/kubernetes/pull/123894), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.30.0. ([#124364](https://github.com/kubernetes/kubernetes/pull/124364), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider, Node and Release]

## Dependencies

### Added
- github.com/antlr4-go/antlr/v4: [v4.13.0](https://github.com/antlr4-go/antlr/tree/v4.13.0)
- github.com/go-task/slim-sprig/v3: [v3.0.0](https://github.com/go-task/slim-sprig/tree/v3.0.0)
- go.uber.org/mock: v0.4.0
- gopkg.in/evanphx/json-patch.v4: v4.12.0

### Changed
- cloud.google.com/go/compute/metadata: v0.2.3 → v0.3.0
- cloud.google.com/go/firestore: v1.11.0 → v1.12.0
- cloud.google.com/go/storage: v1.10.0 → v1.0.0
- cloud.google.com/go: v0.110.6 → v0.110.7
- github.com/alecthomas/kingpin/v2: [v2.3.2 → v2.4.0](https://github.com/alecthomas/kingpin/compare/v2.3.2...v2.4.0)
- github.com/chzyer/readline: [2972be2 → v1.5.1](https://github.com/chzyer/readline/compare/2972be2...v1.5.1)
- github.com/container-storage-interface/spec: [v1.8.0 → v1.9.0](https://github.com/container-storage-interface/spec/compare/v1.8.0...v1.9.0)
- github.com/cpuguy83/go-md2man/v2: [v2.0.2 → v2.0.3](https://github.com/cpuguy83/go-md2man/compare/v2.0.2...v2.0.3)
- github.com/davecgh/go-spew: [v1.1.1 → d8f796a](https://github.com/davecgh/go-spew/compare/v1.1.1...d8f796a)
- github.com/fxamacker/cbor/v2: [v2.6.0 → v2.7.0-beta](https://github.com/fxamacker/cbor/compare/v2.6.0...v2.7.0-beta)
- github.com/go-openapi/swag: [v0.22.3 → v0.22.4](https://github.com/go-openapi/swag/compare/v0.22.3...v0.22.4)
- github.com/golang/glog: [v1.1.0 → v1.1.2](https://github.com/golang/glog/compare/v1.1.0...v1.1.2)
- github.com/golang/mock: [v1.6.0 → v1.3.1](https://github.com/golang/mock/compare/v1.6.0...v1.3.1)
- github.com/google/cel-go: [v0.17.8 → v0.20.1](https://github.com/google/cel-go/compare/v0.17.8...v0.20.1)
- github.com/google/pprof: [4bb14d4 → 4bfdf5a](https://github.com/google/pprof/compare/4bb14d4...4bfdf5a)
- github.com/google/uuid: [v1.3.0 → v1.3.1](https://github.com/google/uuid/compare/v1.3.0...v1.3.1)
- github.com/googleapis/gax-go/v2: [v2.11.0 → v2.0.5](https://github.com/googleapis/gax-go/compare/v2.11.0...v2.0.5)
- github.com/ianlancetaylor/demangle: [28f6c0f → bd984b5](https://github.com/ianlancetaylor/demangle/compare/28f6c0f...bd984b5)
- github.com/jstemmer/go-junit-report: [v0.9.1 → af01ea7](https://github.com/jstemmer/go-junit-report/compare/v0.9.1...af01ea7)
- github.com/matttproud/golang_protobuf_extensions: [v1.0.4 → v1.0.2](https://github.com/matttproud/golang_protobuf_extensions/compare/v1.0.4...v1.0.2)
- github.com/onsi/ginkgo/v2: [v2.15.0 → v2.19.0](https://github.com/onsi/ginkgo/compare/v2.15.0...v2.19.0)
- github.com/onsi/gomega: [v1.31.0 → v1.33.1](https://github.com/onsi/gomega/compare/v1.31.0...v1.33.1)
- github.com/pmezard/go-difflib: [v1.0.0 → 5d4384e](https://github.com/pmezard/go-difflib/compare/v1.0.0...5d4384e)
- github.com/prometheus/client_golang: [v1.16.0 → v1.19.0](https://github.com/prometheus/client_golang/compare/v1.16.0...v1.19.0)
- github.com/prometheus/client_model: [v0.4.0 → v0.6.0](https://github.com/prometheus/client_model/compare/v0.4.0...v0.6.0)
- github.com/prometheus/common: [v0.44.0 → v0.48.0](https://github.com/prometheus/common/compare/v0.44.0...v0.48.0)
- github.com/prometheus/procfs: [v0.10.1 → v0.12.0](https://github.com/prometheus/procfs/compare/v0.10.1...v0.12.0)
- github.com/rogpeppe/go-internal: [v1.10.0 → v1.11.0](https://github.com/rogpeppe/go-internal/compare/v1.10.0...v1.11.0)
- github.com/sergi/go-diff: [v1.1.0 → v1.2.0](https://github.com/sergi/go-diff/compare/v1.1.0...v1.2.0)
- github.com/sirupsen/logrus: [v1.9.0 → v1.9.3](https://github.com/sirupsen/logrus/compare/v1.9.0...v1.9.3)
- github.com/spf13/cobra: [v1.7.0 → v1.8.0](https://github.com/spf13/cobra/compare/v1.7.0...v1.8.0)
- go.etcd.io/bbolt: v1.3.8 → v1.3.9
- go.etcd.io/etcd/api/v3: v3.5.10 → v3.5.13
- go.etcd.io/etcd/client/pkg/v3: v3.5.10 → v3.5.13
- go.etcd.io/etcd/client/v2: v2.305.10 → v2.305.13
- go.etcd.io/etcd/client/v3: v3.5.10 → v3.5.13
- go.etcd.io/etcd/pkg/v3: v3.5.10 → v3.5.13
- go.etcd.io/etcd/raft/v3: v3.5.10 → v3.5.13
- go.etcd.io/etcd/server/v3: v3.5.10 → v3.5.13
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.42.0 → v0.46.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.19.0 → v1.20.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.19.0 → v1.20.0
- go.opentelemetry.io/otel/metric: v1.19.0 → v1.20.0
- go.opentelemetry.io/otel/sdk: v1.19.0 → v1.20.0
- go.opentelemetry.io/otel/trace: v1.19.0 → v1.20.0
- go.opentelemetry.io/otel: v1.19.0 → v1.20.0
- golang.org/x/crypto: v0.21.0 → v0.23.0
- golang.org/x/exp: a9213ee → f3d0a9c
- golang.org/x/lint: 6edffad → 1621716
- golang.org/x/mod: v0.15.0 → v0.17.0
- golang.org/x/net: v0.23.0 → v0.25.0
- golang.org/x/oauth2: v0.10.0 → v0.20.0
- golang.org/x/sync: v0.6.0 → v0.7.0
- golang.org/x/sys: v0.18.0 → v0.20.0
- golang.org/x/telemetry: b75ee88 → f48c80b
- golang.org/x/term: v0.18.0 → v0.20.0
- golang.org/x/text: v0.14.0 → v0.15.0
- golang.org/x/tools: v0.18.0 → v0.21.0
- google.golang.org/api: v0.126.0 → v0.13.0
- google.golang.org/genproto/googleapis/api: 23370e0 → b8732ec
- google.golang.org/genproto: f966b18 → b8732ec
- google.golang.org/grpc: v1.58.3 → v1.59.0
- honnef.co/go/tools: v0.0.1-2020.1.4 → v0.0.1-2019.2.3
- sigs.k8s.io/knftables: v0.0.14 → v0.0.16
- sigs.k8s.io/kustomize/api: 6ce0bf3 → v0.17.2
- sigs.k8s.io/kustomize/cmd/config: v0.11.2 → v0.14.1
- sigs.k8s.io/kustomize/kustomize/v5: 6ce0bf3 → v5.4.2
- sigs.k8s.io/kustomize/kyaml: 6ce0bf3 → v0.17.1
- sigs.k8s.io/yaml: v1.3.0 → v1.4.0

### Removed
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [f118173](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/tree/f118173)
- github.com/antlr/antlr4/runtime/Go/antlr/v4: [8188dc5](https://github.com/antlr/antlr4/tree/runtime/Go/antlr/v4/8188dc5)
- github.com/evanphx/json-patch: [v4.12.0+incompatible](https://github.com/evanphx/json-patch/tree/v4.12.0)
- github.com/fvbommel/sortorder: [v1.1.0](https://github.com/fvbommel/sortorder/tree/v1.1.0)
- github.com/go-gl/glfw/v3.3/glfw: [6f7a984](https://github.com/go-gl/glfw/tree/v3.3/glfw/6f7a984)
- github.com/go-task/slim-sprig: [52ccab3](https://github.com/go-task/slim-sprig/tree/52ccab3)
- github.com/golang/snappy: [v0.0.3](https://github.com/golang/snappy/tree/v0.0.3)
- github.com/google/martian/v3: [v3.2.1](https://github.com/google/martian/tree/v3.2.1)
- github.com/google/s2a-go: [v0.1.7](https://github.com/google/s2a-go/tree/v0.1.7)
- github.com/googleapis/enterprise-certificate-proxy: [v0.2.3](https://github.com/googleapis/enterprise-certificate-proxy/tree/v0.2.3)
- google.golang.org/genproto/googleapis/bytestream: e85fd2c
- google.golang.org/grpc/cmd/protoc-gen-go-grpc: v1.1.0
- gopkg.in/gcfg.v1: v1.2.3
- gopkg.in/warnings.v0: v0.1.2
- rsc.io/quote/v3: v3.1.0
- rsc.io/sampler: v1.3.0