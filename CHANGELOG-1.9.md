<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.9.5](#v195)
  - [Downloads for v1.9.5](#downloads-for-v195)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.9.4](#changelog-since-v194)
    - [Other notable changes](#other-notable-changes)
- [v1.9.4](#v194)
  - [Downloads for v1.9.4](#downloads-for-v194)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.9.3](#changelog-since-v193)
    - [Other notable changes](#other-notable-changes-1)
- [v1.9.3](#v193)
  - [Downloads for v1.9.3](#downloads-for-v193)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.9.2](#changelog-since-v192)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-2)
- [v1.9.2](#v192)
  - [Downloads for v1.9.2](#downloads-for-v192)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.9.1](#changelog-since-v191)
    - [Other notable changes](#other-notable-changes-3)
- [v1.9.1](#v191)
  - [Downloads for v1.9.1](#downloads-for-v191)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.9.0](#changelog-since-v190)
    - [Other notable changes](#other-notable-changes-4)
- [v1.9.0](#v190)
  - [Downloads for v1.9.0](#downloads-for-v190)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [1.9 Release Notes](#19-release-notes)
  - [WARNING: etcd backup strongly recommended](#warning-etcd-backup-strongly-recommended)
  - [Introduction to 1.9.0](#introduction-to-190)
  - [Major themes](#major-themes)
    - [API Machinery](#api-machinery)
    - [Apps](#apps)
    - [Auth](#auth)
    - [AWS](#aws)
    - [Azure](#azure)
    - [Cluster Lifecycle](#cluster-lifecycle)
    - [Instrumentation](#instrumentation)
    - [Network](#network)
    - [Node](#node)
    - [OpenStack](#openstack)
    - [Storage](#storage)
    - [Windows](#windows)
  - [Before Upgrading](#before-upgrading)
    - [**API Machinery**](#api-machinery-1)
    - [**Auth**](#auth-1)
    - [**CLI**](#cli)
    - [**Cluster Lifecycle**](#cluster-lifecycle-1)
    - [**Multicluster**](#multicluster)
    - [**Node**](#node-1)
    - [**Network**](#network-1)
    - [**Scheduling**](#scheduling)
    - [**Storage**](#storage-1)
    - [**OpenStack**](#openstack-1)
  - [Known Issues](#known-issues)
  - [Deprecations](#deprecations)
    - [**API Machinery**](#api-machinery-2)
    - [**Auth**](#auth-2)
    - [**Cluster Lifecycle**](#cluster-lifecycle-2)
    - [**Network**](#network-2)
    - [**Storage**](#storage-2)
    - [**Scheduling**](#scheduling-1)
    - [**Node**](#node-2)
  - [Notable Changes](#notable-changes)
    - [**Workloads API (apps/v1)**](#workloads-api-appsv1)
    - [**API Machinery**](#api-machinery-3)
      - [**Admission Control**](#admission-control)
      - [**API & API server**](#api-&-api-server)
      - [**Audit**](#audit)
      - [**Custom Resources**](#custom-resources)
      - [**Other**](#other)
    - [**Apps**](#apps-1)
    - [**Auth**](#auth-3)
      - [**Audit**](#audit-1)
      - [**RBAC**](#rbac)
      - [**Other**](#other-1)
      - [**GCE**](#gce)
    - [**Autoscaling**](#autoscaling)
    - [**AWS**](#aws-1)
    - [**Azure**](#azure-1)
    - [**CLI**](#cli-1)
      - [**Kubectl**](#kubectl)
    - [**Cluster Lifecycle**](#cluster-lifecycle-3)
      - [**API Server**](#api-server)
      - [**Cloud Provider Integration**](#cloud-provider-integration)
      - [**Kubeadm**](#kubeadm)
      - [**Juju**](#juju)
      - [**Other**](#other-2)
      - [**GCP**](#gcp)
    - [**Instrumentation**](#instrumentation-1)
      - [**Audit**](#audit-2)
      - [**Other**](#other-3)
    - [**Multicluster**](#multicluster-1)
      - [**Federation**](#federation)
    - [**Network**](#network-3)
      - [**IPv6**](#ipv6)
      - [**IPVS**](#ipvs)
      - [**Kube-Proxy**](#kube-proxy)
      - [**CoreDNS**](#coredns)
      - [**Other**](#other-4)
    - [**Node**](#node-3)
      - [**Pod API**](#pod-api)
      - [**Hardware Accelerators**](#hardware-accelerators)
      - [**Container Runtime**](#container-runtime)
      - [**Kubelet**](#kubelet)
        - [**Other**](#other-5)
    - [**OpenStack**](#openstack-2)
    - [**Scheduling**](#scheduling-2)
      - [**Hardware Accelerators**](#hardware-accelerators-1)
      - [**Other**](#other-6)
    - [**Storage**](#storage-3)
  - [External Dependencies](#external-dependencies)
- [v1.9.0-beta.2](#v190-beta2)
  - [Downloads for v1.9.0-beta.2](#downloads-for-v190-beta2)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.9.0-beta.1](#changelog-since-v190-beta1)
    - [Other notable changes](#other-notable-changes-5)
- [v1.9.0-beta.1](#v190-beta1)
  - [Downloads for v1.9.0-beta.1](#downloads-for-v190-beta1)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
  - [Changelog since v1.9.0-alpha.3](#changelog-since-v190-alpha3)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-6)
- [v1.9.0-alpha.3](#v190-alpha3)
  - [Downloads for v1.9.0-alpha.3](#downloads-for-v190-alpha3)
    - [Client Binaries](#client-binaries-8)
    - [Server Binaries](#server-binaries-8)
    - [Node Binaries](#node-binaries-8)
  - [Changelog since v1.9.0-alpha.2](#changelog-since-v190-alpha2)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-7)
- [v1.9.0-alpha.2](#v190-alpha2)
  - [Downloads for v1.9.0-alpha.2](#downloads-for-v190-alpha2)
    - [Client Binaries](#client-binaries-9)
    - [Server Binaries](#server-binaries-9)
    - [Node Binaries](#node-binaries-9)
  - [Changelog since v1.8.0](#changelog-since-v180)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-8)
- [v1.9.0-alpha.1](#v190-alpha1)
  - [Downloads for v1.9.0-alpha.1](#downloads-for-v190-alpha1)
    - [Client Binaries](#client-binaries-10)
    - [Server Binaries](#server-binaries-10)
    - [Node Binaries](#node-binaries-10)
  - [Changelog since v1.8.0-alpha.3](#changelog-since-v180-alpha3)
    - [Action Required](#action-required-4)
    - [Other notable changes](#other-notable-changes-9)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.9.5

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.5


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes.tar.gz) | `72947d7ac9a6f5bfe9f98b3362ce176cfc4d7c35caa1cf974ca2fd6dbc8ad608`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-src.tar.gz) | `a02261fb0e1d70feb95af36d404ad247ee46103b84748a5fec8b906648f68e0f`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-darwin-386.tar.gz) | `5ee29f16801e5776e985dd2a02bebe0ba25079ba0a5d331115bc9cd2a5b1225c`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-darwin-amd64.tar.gz) | `5309c71bae2f1a8133d5c41522b827c0905fdaf2122690388fa5e15f4898a1c9`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-linux-386.tar.gz) | `8e75b94eb78187033583b9bd9e53fbe4f74b242b9a8f03ebc953db6806604ae4`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-linux-amd64.tar.gz) | `3248cff16e96610166f8b1c7c2d5b2403fb4fd15bff5af6e935388105ca62bc4`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-linux-arm.tar.gz) | `d7836eb1d723db98a3a1d9b4e5fe16f9ee5b3304b680bd78d158ebcea59fdcb3`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-linux-arm64.tar.gz) | `5ce9642d2413c87ffe42a90ba4878b0793f0fd5e143e00640c31a92ae1d80ded`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-linux-ppc64le.tar.gz) | `6fd223aa94d18abca50a1f7f9607a23fba487770eba4de0f696e9a6bebd565da`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-linux-s390x.tar.gz) | `eadc49e8c33461b1290fcf07b49db639478837f3bb7330d50fe2516fcf919e1a`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-windows-386.tar.gz) | `cd2a65af308864e1a6128c40e87ddc56ece6d20de7ecc7009f3356a534a0fcfb`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-client-windows-amd64.tar.gz) | `e1f14e4f4028f545a9ab1bb02ff47b0a80afcb4f3a6f15e7b141fe580440e15f`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-server-linux-amd64.tar.gz) | `189e299a990f3dd2be9d2ac6c8315ea43a0ce1ccfc5d9d6b8c3325537a90395f`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-server-linux-arm.tar.gz) | `408187d6a9cd8fa76581cdd64e438804993d39aea4061fd879cb9f30ddebdbda`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-server-linux-arm64.tar.gz) | `5b68c3160c5c5455049684ffd3fcbe2c8037a0e8c992d657ff6f42de9db5d602`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-server-linux-ppc64le.tar.gz) | `224d6566030d7b5e3215a257a625bda51dfa9c02af27bdcb1fbec9c0a57d7749`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-server-linux-s390x.tar.gz) | `46c07056949cea5bc8ce60fb6c2b8cefee1fe7dc7adc705aeb8ef8ad0d703738`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-node-linux-amd64.tar.gz) | `46e5d05356e9ea99877cb1b0ef033cdc5a5e87df5a5c45c5cbc1f54adb278c1d`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-node-linux-arm.tar.gz) | `022044653f19fa662467aea85ce8586d59e5888b8863cf5d95ca3c70424906c9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-node-linux-arm64.tar.gz) | `89ddd363d6ee207a9e78211dac9a74771d4eaf80668377d880632e8b31fd61d4`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-node-linux-ppc64le.tar.gz) | `77543ec335bab7c69eb0b5a651de7252ecf033120867cd45029f65e07e901027`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-node-linux-s390x.tar.gz) | `02266ccf4c16cf46fe8c218d12ad8c2c4513457f1b64aeb9d1e6ce59fb92c8b9`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.5/kubernetes-node-windows-amd64.tar.gz) | `adc21f94570496c6299d4bf002f1b749d4695f094027aeaf24deb82f9818214d`

## Changelog since v1.9.4

### Other notable changes

* gce: fixes race condition in ServiceController, where nodes weren't updated in the node sync loop, by updating TargetPools in the ensureExternalLoadBalancer call. ([#58368](https://github.com/kubernetes/kubernetes/pull/58368), [@MrHohn](https://github.com/MrHohn))
* fix race condition issue when detaching azure disk ([#60183](https://github.com/kubernetes/kubernetes/pull/60183), [@andyzhangx](https://github.com/andyzhangx))
* Get parent dir via canonical absolute path when trying to judge mount-point ([#58433](https://github.com/kubernetes/kubernetes/pull/58433), [@yue9944882](https://github.com/yue9944882))
* Set node external IP for azure node when disabling UseInstanceMetadata ([#60959](https://github.com/kubernetes/kubernetes/pull/60959), [@feiskyer](https://github.com/feiskyer))
* Fixes potential deadlock when deleting CustomResourceDefinition for custom resources with finalizers ([#60542](https://github.com/kubernetes/kubernetes/pull/60542), [@liggitt](https://github.com/liggitt))
* Unauthorized requests will not match audit policy rules where users or groups are set. ([#59398](https://github.com/kubernetes/kubernetes/pull/59398), [@CaoShuFeng](https://github.com/CaoShuFeng))
* [fluentd-gcp addon] Fixed bug with reporting metrics in event-exporter ([#60126](https://github.com/kubernetes/kubernetes/pull/60126), [@serathius](https://github.com/serathius))
* Restores the ability of older clients to delete and scale jobs with initContainers ([#59880](https://github.com/kubernetes/kubernetes/pull/59880), [@liggitt](https://github.com/liggitt))
* fixed foreground deletion of podtemplates ([#60683](https://github.com/kubernetes/kubernetes/pull/60683), [@nilebox](https://github.com/nilebox))
* Bug fix: Clusters with GCE feature 'DiskAlphaAPI' enabled were unable to dynamically provision GCE PD volumes. ([#59447](https://github.com/kubernetes/kubernetes/pull/59447), [@verult](https://github.com/verult))
* Fix a regression that prevented using `subPath` volume mounts with secret, configMap, projected, and downwardAPI volumes ([#61080](https://github.com/kubernetes/kubernetes/pull/61080), [@liggitt](https://github.com/liggitt))



# v1.9.4

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.4


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes.tar.gz) | `45b6aa8adbf3cf9fe37ddf063400a984766363b31f4da6204c00d02815616ce4`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-src.tar.gz) | `645819c4e479d80d4f489bb022e3332fcede8fcb8e4265245621547d0b5ac8a7`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-darwin-386.tar.gz) | `8a6401958fa52b0a56011a7650ca2ab06d95b6927826cbc4834287e313f7216b`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-darwin-amd64.tar.gz) | `7217f3d72ee5a23a06703c262dc051728775615236b87fa53c45969152a92c8d`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-linux-386.tar.gz) | `e5bd0c3fe36accd55e11854e4893fdced22261ef70d6ad22f67f8a70efbfee57`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-linux-amd64.tar.gz) | `db6ec27f4541ef91f73a3dd173851fea06c17a1eac52b86641d6d639f3dfc2ea`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-linux-arm.tar.gz) | `15e385d032165cade02a4969618c13fb8b3e1074c06318581b646bd93cfc7153`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-linux-arm64.tar.gz) | `a49b589cb08711714f70cbdf5fc2734a981746b0e29858c60bc83b4ca226a132`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-linux-ppc64le.tar.gz) | `3aae4268fdc5a81f0593abf5161c172b0c88c57f61ff6515a7cde1c7b35afc21`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-linux-s390x.tar.gz) | `fb28c3940f3ab905caafccf277e2fe8fcdda7627dd8b784f25ca400d0e2383a2`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-windows-386.tar.gz) | `243e4dc67fe9a55824c4648ba77589bcb7e34850095056433e5f15238cd3da32`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-client-windows-amd64.tar.gz) | `b4ce826e60550c4a6852f28537565e115392a979f8c91095fdc2b61c13431e9a`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-server-linux-amd64.tar.gz) | `ef7ffabb220df9616d9a483a815c3182d44c868a5bbb9ad1b1297270d3f59fec`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-server-linux-arm.tar.gz) | `bff92c29be844663c087f17c85d527d9cf458ddcdeee0926f74da48d0d980c31`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-server-linux-arm64.tar.gz) | `568c8966ebe05164452e6ae152352dc6d05a56bfb9b5c1860f5a68c15b55c0bd`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-server-linux-ppc64le.tar.gz) | `39a1918ae53022fb38c98a02b945ae6cd5607a3a71011b19395fbc759739a453`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-server-linux-s390x.tar.gz) | `2ea1af6654d5b3178fd4738c7fe5d019e0b4ea5fab746ebe2daa559d661844c7`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-node-linux-amd64.tar.gz) | `a8224a7bf2b1b9aeab80946cfcf0f68549b3972f41850d857487c912318780c8`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-node-linux-arm.tar.gz) | `503ae9d552e94f0fd4fe44787f38c3fc67c47f84f8d3567cc043375962d295a3`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-node-linux-arm64.tar.gz) | `e6578a7929858ab4b411a0ff6ea2c2d0c78dfc94ea4e23210d1db92fee3a930f`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-node-linux-ppc64le.tar.gz) | `c1269cb6a78fc5bb57b47ed24458e4bd17cad6c849b5951c919d39ff72bcb0da`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-node-linux-s390x.tar.gz) | `06f2027b133e3617806b327c7a7248c75025146efc31ba2cabe9480ba3dc7fed`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.4/kubernetes-node-windows-amd64.tar.gz) | `2304b297bd9aec7e4285c697db75a523ff9dbb2c00e3baccf90375d4fdab5b91`

## Changelog since v1.9.3

### Other notable changes

* Fixes CVE-2017-1002101 - See https://issue.k8s.io/60813 for details ([#61045](https://github.com/kubernetes/kubernetes/pull/61045), [@liggitt](https://github.com/liggitt))
* Fixes a case when Deployment with recreate strategy could get stuck on old failed Pod. ([#60493](https://github.com/kubernetes/kubernetes/pull/60493), [@tnozicka](https://github.com/tnozicka))
* Build using go1.9.3. ([#59012](https://github.com/kubernetes/kubernetes/pull/59012), [@ixdy](https://github.com/ixdy))
* fix device name change issue for azure disk ([#60346](https://github.com/kubernetes/kubernetes/pull/60346), [@andyzhangx](https://github.com/andyzhangx))
* Changes secret, configMap, downwardAPI and projected volumes to mount read-only, instead of allowing applications to write data and then reverting it automatically. Until version 1.11, setting the feature gate ReadOnlyAPIDataVolumes=false will preserve the old behavior. ([#58720](https://github.com/kubernetes/kubernetes/pull/58720), [@joelsmith](https://github.com/joelsmith))
* Add automatic etcd 3.2->3.1 and 3.1->3.0 minor version rollback support to gcr.io/google_container/etcd images. For HA clusters, all members must be stopped before performing a rollback. ([#59298](https://github.com/kubernetes/kubernetes/pull/59298), [@jpbetz](https://github.com/jpbetz))
* Fix the bug where kubelet in the standalone mode would wait for the update from the apiserver source. ([#59276](https://github.com/kubernetes/kubernetes/pull/59276), [@roboll](https://github.com/roboll))
* fix the create azure file pvc failure if there is no storage account in current resource group ([#56557](https://github.com/kubernetes/kubernetes/pull/56557), [@andyzhangx](https://github.com/andyzhangx))
* Increase allowed lag for ssh key sync loop in tunneler to allow for one failure ([#60068](https://github.com/kubernetes/kubernetes/pull/60068), [@wojtek-t](https://github.com/wojtek-t))
* Fixing a bug in OpenStack cloud provider, where dual stack deployments (IPv4 and IPv6) did not work well when using kubenet as the network plugin. ([#59749](https://github.com/kubernetes/kubernetes/pull/59749), [@zioproto](https://github.com/zioproto))
* Bugfix: vSphere Cloud Provider (VCP) does not need any special service account anymore. ([#59440](https://github.com/kubernetes/kubernetes/pull/59440), [@rohitjogvmw](https://github.com/rohitjogvmw))
* vSphere Cloud Provider supports VMs provisioned on vSphere v1.6.5 ([#59519](https://github.com/kubernetes/kubernetes/pull/59519), [@abrarshivani](https://github.com/abrarshivani))
* Allow node IPAM controller to configure IPAMFromCluster mode to use IP aliases instead of routes in GCP. ([#59688](https://github.com/kubernetes/kubernetes/pull/59688), [@jingax10](https://github.com/jingax10))
* Fixed an issue where Portworx volume driver wasn't passing namespace and annotations to the Portworx Create API. ([#59607](https://github.com/kubernetes/kubernetes/pull/59607), [@harsh-px](https://github.com/harsh-px))
* Use a more reliable way to get total physical memory on windows nodes ([#57124](https://github.com/kubernetes/kubernetes/pull/57124), [@JiangtianLi](https://github.com/JiangtianLi))
* Fix race causing apiserver crashes during etcd healthchecking ([#60069](https://github.com/kubernetes/kubernetes/pull/60069), [@wojtek-t](https://github.com/wojtek-t))
* return error if New-SmbGlobalMapping failed when mounting azure file on Windows ([#59540](https://github.com/kubernetes/kubernetes/pull/59540), [@andyzhangx](https://github.com/andyzhangx))
* Ensure Azure public IP removed after service deleted ([#59340](https://github.com/kubernetes/kubernetes/pull/59340), [@feiskyer](https://github.com/feiskyer))
* Map correct vmset name for Azure internal load balancers ([#59747](https://github.com/kubernetes/kubernetes/pull/59747), [@feiskyer](https://github.com/feiskyer))
* Node's providerID is following Azure resource ID format now when useInstanceMetadata is enabled ([#59539](https://github.com/kubernetes/kubernetes/pull/59539), [@feiskyer](https://github.com/feiskyer))



# v1.9.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes.tar.gz) | `b495325eacd1354514b20ef1f0b99c6a41277842fc93b6cf5c9cb6e8657c266f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-src.tar.gz) | `f99a016dc616be37e7fe161ff435335a2442ebcede622486e7a9cf0bacedb625`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-darwin-386.tar.gz) | `084dd17c182acbc1ee248ea9f9fc720224be6245f13d9904cd7ca44205eb38ed`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-darwin-amd64.tar.gz) | `c6ae13f8da18322ca3651b289c8e48475839e6f4c741ae12342cd69bde467773`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-linux-386.tar.gz) | `231d9255c11d38b88c6b7febe43d1ea9564c6b36b34cb905450c7beb7c46e051`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-linux-amd64.tar.gz) | `2f509c05f0c4e1c1ac9e98879a1924f24546905349457904344d79dc639217be`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-linux-arm64.tar.gz) | `d8fe5dc1bc80d5dfb60e811c0bfcd392b2761f76400fc4c48b17d4d4cd0aabf1`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-linux-arm.tar.gz) | `7c084e01a97379256746ada2b980e36e727acc23aaa614d98e4e0a144faad37e`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-linux-ppc64le.tar.gz) | `669629d372ebe169140238f106c6d97b53a1895f4ac8393147fbddddf83eeb47`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-linux-s390x.tar.gz) | `1627933c04ba9a155ac63c0a9a90ada32badd618c2e2919d3044cd5039963cc4`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-windows-386.tar.gz) | `68de0d599a5e09195479602390343a017296b3aa774b4a783455581e1065cc8d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-client-windows-amd64.tar.gz) | `e8872561f33258a8509e90aa955c5b57d6b5d9a657864bf5002e21285a8f4792`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-server-linux-amd64.tar.gz) | `09ab78a1b091ce8affb39d5218ba780eb36bc8026d557ed6d5efcd5a51b7927a`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-server-linux-arm64.tar.gz) | `f3e38a8ffae0b5f2ac0c776a0b4586630e8b258f2802237ebda4d612f6d41d9e`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-server-linux-arm.tar.gz) | `eeba15fc5db374e6e1b66b846988422e751752d930e4c2c89f5a92de5f466010`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-server-linux-ppc64le.tar.gz) | `ce05d9cf268b213e9a57dcbb5f9d570c62e72a15f8af9e692f4a26a8f40d8df1`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-server-linux-s390x.tar.gz) | `1ca63330add758e7638357eba79753d1af610ea5de8b082aa740ef4852abd51a`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-node-linux-amd64.tar.gz) | `c40f983c11f93752a40180cb719ddd473cbf07f43a3af5d2b575411c85b76f88`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-node-linux-arm64.tar.gz) | `7a0c5c313d14d88bd11010d416c0614e7dc2362e78e1ffb65ee098bfe944b881`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-node-linux-arm.tar.gz) | `7a3e288cb04e3fe5f2537645bd74a68d7b471c15c6eb51eb9d5e1ac6edfc7e9f`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-node-linux-ppc64le.tar.gz) | `401f763112d20cf2c613d065beecd47387bb11d82a49fd2222a2ac38a4e06c20`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-node-linux-s390x.tar.gz) | `9a6d921e1cef37dcbaac61be13a70410cd03bc26335b7730cce6d9d3c8506b22`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.3/kubernetes-node-windows-amd64.tar.gz) | `0db90e50c23ef16e9bfa5a990647bd4299a809166a2a37764e880b1910feee49`

## Changelog since v1.9.2

### Action Required

* Bug fix: webhooks now do not skip cluster-scoped resources ([#58185](https://github.com/kubernetes/kubernetes/pull/58185), [@caesarxuchao](https://github.com/caesarxuchao))
    * Action required: Before upgrading your Kubernetes clusters, double check if you had configured webhooks for cluster-scoped objects (e.g., nodes, persistentVolume), these webhooks will start to take effect. Delete/modify the configs if that's not desirable.

### Other notable changes

* CustomResourceDefinitions: OpenAPI v3 validation schemas containing `$ref`references are no longer permitted (valid references could not be constructed previously because property ids were not permitted either). Before upgrading, ensure CRD definitions do not include those `$ref` fields. ([#58438](https://github.com/kubernetes/kubernetes/pull/58438), [@carlory](https://github.com/carlory))
* Ensure IP is set for Azure internal load balancer. ([#59083](https://github.com/kubernetes/kubernetes/pull/59083), [@feiskyer](https://github.com/feiskyer))
* Configurable etcd quota backend bytes in GCE ([#59259](https://github.com/kubernetes/kubernetes/pull/59259), [@wojtek-t](https://github.com/wojtek-t))
* Updates Calico version to v2.6.7 (Fixed a bug where Felix would crash when parsing a NetworkPolicy with a named port. See https://github.com/projectcalico/calico/releases/tag/v2.6.7) ([#59130](https://github.com/kubernetes/kubernetes/pull/59130), [@caseydavenport](https://github.com/caseydavenport))
* Cluster Autoscaler 1.1.1 (details: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.1.1) ([#59272](https://github.com/kubernetes/kubernetes/pull/59272), [@mwielgus](https://github.com/mwielgus))
* cloudprovider/openstack: fix bug the tries to use octavia client to query flip ([#59075](https://github.com/kubernetes/kubernetes/pull/59075), [@jrperritt](https://github.com/jrperritt))
* Fixed a bug which caused the apiserver reboot failure in the presence of malfunctioning webhooks. ([#59073](https://github.com/kubernetes/kubernetes/pull/59073), [@caesarxuchao](https://github.com/caesarxuchao))
* Configurable etcd compaction frequency in GCE ([#59106](https://github.com/kubernetes/kubernetes/pull/59106), [@wojtek-t](https://github.com/wojtek-t))
* Prevent kubelet from getting wedged if initialization of modules returns an error. ([#59020](https://github.com/kubernetes/kubernetes/pull/59020), [@brendandburns](https://github.com/brendandburns))
* [GCE] Apiserver uses `InternalIP` as the most preferred kubelet address type by default. ([#59019](https://github.com/kubernetes/kubernetes/pull/59019), [@MrHohn](https://github.com/MrHohn))
* Expose Metrics Server metrics via /metric endpoint. ([#57456](https://github.com/kubernetes/kubernetes/pull/57456), [@kawych](https://github.com/kawych))
* Get windows kernel version directly from registry ([#58498](https://github.com/kubernetes/kubernetes/pull/58498), [@feiskyer](https://github.com/feiskyer))
* Fixes a bug where kubelet crashes trying to free memory under memory pressure ([#58574](https://github.com/kubernetes/kubernetes/pull/58574), [@yastij](https://github.com/yastij))
* Updated priority of mirror pod according to PriorityClassName. ([#58485](https://github.com/kubernetes/kubernetes/pull/58485), [@k82cn](https://github.com/k82cn))
* Access to externally managed IP addresses via the kube-apiserver service proxy subresource is no longer allowed by default. This can be re-enabled via the `ServiceProxyAllowExternalIPs` feature gate, but will be disallowed completely in 1.11 ([#57265](https://github.com/kubernetes/kubernetes/pull/57265), [@brendandburns](https://github.com/brendandburns))
* Detach and clear bad disk URI ([#58345](https://github.com/kubernetes/kubernetes/pull/58345), [@rootfs](https://github.com/rootfs))
* Add apiserver metric for number of requests dropped because of inflight limit. ([#58340](https://github.com/kubernetes/kubernetes/pull/58340), [@gmarek](https://github.com/gmarek))
* Add apiserver metric for current inflight-request usage. ([#58342](https://github.com/kubernetes/kubernetes/pull/58342), [@gmarek](https://github.com/gmarek))
* kube-apiserver is changed to use SSH tunnels for webhook iff the webhook is not directly routable from apiserver's network environment. ([#58644](https://github.com/kubernetes/kubernetes/pull/58644), [@yguo0905](https://github.com/yguo0905))
* Update Calico version to v2.6.6 ([#58482](https://github.com/kubernetes/kubernetes/pull/58482), [@tmjd](https://github.com/tmjd))
* Fix garbage collection and resource quota when the controller-manager uses --leader-elect=false ([#57340](https://github.com/kubernetes/kubernetes/pull/57340), [@jmcmeek](https://github.com/jmcmeek))
* kube-apiserver: fixes loading of `--admission-control-config-file` containing AdmissionConfiguration apiserver.k8s.io/v1alpha1 config object ([#58441](https://github.com/kubernetes/kubernetes/pull/58441), [@liggitt](https://github.com/liggitt))
* Fix a bug affecting nested data volumes such as secret, configmap, etc. ([#57422](https://github.com/kubernetes/kubernetes/pull/57422), [@joelsmith](https://github.com/joelsmith))
* Reduce Metrics Server memory requirement ([#58391](https://github.com/kubernetes/kubernetes/pull/58391), [@kawych](https://github.com/kawych))
* GCP: allow a master to not include a metadata concealment firewall rule (if it's not running the metadata proxy). ([#58104](https://github.com/kubernetes/kubernetes/pull/58104), [@ihmccreery](https://github.com/ihmccreery))
* Bump GCE metadata proxy to v0.1.9 to pick up security fixes. ([#58221](https://github.com/kubernetes/kubernetes/pull/58221), [@ihmccreery](https://github.com/ihmccreery))
* Fixes an issue where the resourceVersion of an object in a DELETE watch event was not the resourceVersion of the delete itself, but of the last update to the object. This could disrupt the ability of clients clients to re-establish watches properly. ([#58547](https://github.com/kubernetes/kubernetes/pull/58547), [@liggitt](https://github.com/liggitt))
* Fixed encryption key and encryption provider rotation ([#58375](https://github.com/kubernetes/kubernetes/pull/58375), [@liggitt](https://github.com/liggitt))
* Correctly handle transient connection reset errors on GET requests from client library. ([#58520](https://github.com/kubernetes/kubernetes/pull/58520), [@porridge](https://github.com/porridge))
* Avoid controller-manager to crash when enabling IP alias for K8s cluster. ([#58557](https://github.com/kubernetes/kubernetes/pull/58557), [@jingax10](https://github.com/jingax10))



# v1.9.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes.tar.gz) | `7a922d49b1194cb1b59b22cecb4eb1197f7c37250d4326410dc71aa5dc5ec8a2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-src.tar.gz) | `9f128809cdd442d71a13f7c61c7a0e03e832cf0c068a86184c1bcc9acdb78872`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-darwin-386.tar.gz) | `37d2dd1b1762f1040699584736bbc1a2392e94779a19061d477786bcce3d3f01`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-darwin-amd64.tar.gz) | `42adc9762b30bfd3648323f9a8f350efeedec08a901997073f6d4244f7a16f78`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-linux-386.tar.gz) | `5dde6c6388353376aaa0bd731b0366d9d2d11baee3746662b008e09d9618d55f`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-linux-amd64.tar.gz) | `c45cf9e9d27b9d1bfc6d26f86856271fec6f8e7007f014597d27668f72f8c349`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-linux-arm64.tar.gz) | `05c3810b00adcdbf7bc67671847f11e287da72f308cc704e5679e83564236fee`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-linux-arm.tar.gz) | `a9421d4627eb9eaa1e46cfd4276943e25b5b80e52db6945f173a2a45782ce42d`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-linux-ppc64le.tar.gz) | `adc345ab050e09a3069a47e862c0ce88630a586905b33f6e5fd339005ceffbbf`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-linux-s390x.tar.gz) | `fdff4b462e67569a4a1110b696d8af2c563e0a19e50a58a7b1a4346942b07993`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-windows-386.tar.gz) | `1a82e8e4213153993a6e86e74120f62f95645952b223ed8586316358dd22a225`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-client-windows-amd64.tar.gz) | `a8648d4d3e0f85597bd57de87459a040ceab4c073d647027a70b0fba8862eab3`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-server-linux-amd64.tar.gz) | `2218fe0b939273b57ce00c7d5f3f7d2c34ebde5ae500ba2646eea6ba26c7c63d`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-server-linux-arm64.tar.gz) | `3b4bc6cf91c3eaf37ef2b361dd77e838f0a8ca2b8cbb4dd42793c1fea5186b69`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-server-linux-arm.tar.gz) | `73e77da0ddc951f791b5f7b73420ba0dbb141b3637cc48b4e916a41249e40ce3`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-server-linux-ppc64le.tar.gz) | `860ba4ac773e4aff69dde781cac7ac1fb1824f2158155dfa49c50dd3acf0ab82`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-server-linux-s390x.tar.gz) | `19e0fd7863e217b4cb67f91b56ceb5939ae677f523681bdf8ccac174f36f576d`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-node-linux-amd64.tar.gz) | `f86b7038dc89d79b277c5fba499f391c25f5aba8f5caa3119c05065f9917b6f9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-node-linux-arm64.tar.gz) | `87f40c37a3e359a9350a3bcbe0e27ad6e7dfa0d8ee5f6d2ecf061813423ffa73`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-node-linux-arm.tar.gz) | `b73d879a03e7eba5543af0b56085ebb4919d401f6a06d4803517ddf606e8240e`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-node-linux-ppc64le.tar.gz) | `26331e5d84d98fc3a94d2d55fd411159b2a79b6083758cea1dac36a0a4a44336`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-node-linux-s390x.tar.gz) | `cbf52f3942965bb659d1f0f624e09ff01b2ee9f6e6217b3876c41600e1d4c711`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.2/kubernetes-node-windows-amd64.tar.gz) | `70d59046a7c949d4fd4850ee57b1cd44dddfb041c548a21354ee30d7bfb1003d`

## Changelog since v1.9.1

### Other notable changes

* Fixes authentication problem faced during various vSphere operations. ([#57978](https://github.com/kubernetes/kubernetes/pull/57978), [@prashima](https://github.com/prashima))
* The getSubnetIDForLB() should return subnet id rather than net id. ([#58208](https://github.com/kubernetes/kubernetes/pull/58208), [@FengyunPan](https://github.com/FengyunPan))
* Add cache for VM get operation in azure cloud provider ([#57432](https://github.com/kubernetes/kubernetes/pull/57432), [@karataliu](https://github.com/karataliu))
* Update kube-dns to Version 1.14.8 that includes only small changes to how Prometheus metrics are collected. ([#57918](https://github.com/kubernetes/kubernetes/pull/57918), [@rramkumar1](https://github.com/rramkumar1))
* Fixes a possible deadlock preventing quota from being recalculated ([#58107](https://github.com/kubernetes/kubernetes/pull/58107), [@ironcladlou](https://github.com/ironcladlou))
* Fixes a bug in Heapster deployment for google sink. ([#57902](https://github.com/kubernetes/kubernetes/pull/57902), [@kawych](https://github.com/kawych))
* GCE: Allows existing internal load balancers to continue using an outdated subnetwork  ([#57861](https://github.com/kubernetes/kubernetes/pull/57861), [@nicksardo](https://github.com/nicksardo))
* Update etcd version to 3.1.11 ([#57811](https://github.com/kubernetes/kubernetes/pull/57811), [@xiangpengzhao](https://github.com/xiangpengzhao))
* fix device name change issue for azure disk: add remount logic ([#57953](https://github.com/kubernetes/kubernetes/pull/57953), [@andyzhangx](https://github.com/andyzhangx))
* calico-node addon tolerates all NoExecute and NoSchedule taints by default. ([#57122](https://github.com/kubernetes/kubernetes/pull/57122), [@caseydavenport](https://github.com/caseydavenport))
* Allow kubernetes components to react to SIGTERM signal and shutdown gracefully. ([#57756](https://github.com/kubernetes/kubernetes/pull/57756), [@mborsz](https://github.com/mborsz))
* Fixes controller manager crash in certain vSphere cloud provider environment. ([#57286](https://github.com/kubernetes/kubernetes/pull/57286), [@rohitjogvmw](https://github.com/rohitjogvmw))
* fix azure disk not available issue when device name changed ([#57549](https://github.com/kubernetes/kubernetes/pull/57549), [@andyzhangx](https://github.com/andyzhangx))
* GCE: support passing kube-scheduler policy config via SCHEDULER_POLICY_CONFIG ([#57425](https://github.com/kubernetes/kubernetes/pull/57425), [@yguo0905](https://github.com/yguo0905))



# v1.9.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes.tar.gz) | `0eece0e6c1f68535ea71b58b87e239019bb57fdd61118f3d7defa6bbf4fad5ee`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-src.tar.gz) | `625ebb79412bd12feccf12e8b6a15d9c71ea681b571f34deaa59fe6c9ba55935`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-darwin-386.tar.gz) | `909556ed9b8445703d0124f2d8c1901b00afaba63a9123a4296be8663c3a2b2d`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-darwin-amd64.tar.gz) | `71e191d99d3ac1426e23e087b8d0875e793e5615d3aa7ac1e175b250f9707c48`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-linux-386.tar.gz) | `1c4e60c0c056a3300c7fcc9faccd1b1ea2b337e1360c20c5b1c25fdc47923cf0`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-linux-amd64.tar.gz) | `fe8fe40148df404b33069931ea30937699758ed4611ef6baddb4c21b7b19db5e`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-linux-arm64.tar.gz) | `921f5711b97f0b4de69784d9c79f95e80f75a550f28fc1f26597aa0ef6faa471`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-linux-arm.tar.gz) | `77b010cadef98dc832a2f560afe15e57a675ed9fbc59ffad5e19878510997874`
[kubernetes-client-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-linux-ppc64le.tar.gz) | `02aa71ddcbe8b711814af7287aac79de5d99c1c143c0d3af5e14b1ff195b8bdc`
[kubernetes-client-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-linux-s390x.tar.gz) | `7e315024267306a620045d003785ecc8d7f2e763a6108ae806d5d384aa7552cc`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-windows-386.tar.gz) | `99b2a81b7876498e119db4cb34c434b3790bc41cd882384037c1c1b18cba9f99`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-client-windows-amd64.tar.gz) | `d89d303cbbf9e57e5a540277158e4d83ad18ca7402b5b54665f1378bb4528599`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-server-linux-amd64.tar.gz) | `5acf2527461419ba883ac352f7c36c3fa0b86a618dbede187054ad90fa233b0e`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-server-linux-arm64.tar.gz) | `e1f61b4dc6e0c9986e95ec25f876f9a89966215ee8cc7f4a3539ec391b217587`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-server-linux-arm.tar.gz) | `441c45e16e63e9bdf99887a896a99b3a376af778cb778cc1d0e6afc505237200`
[kubernetes-server-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-server-linux-ppc64le.tar.gz) | `c0175f02180d9c88028ee5ad4e3ea04af8a6741a97f4900b02615f7f83c4d1c5`
[kubernetes-server-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-server-linux-s390x.tar.gz) | `2178150d31197ad7f59d44ffea37d682c2675b3a4ea2fc3fa1eaa0e768b993f7`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-node-linux-amd64.tar.gz) | `b8ff0ae693ecca4d55669c66786d6c585f8c77b41a270d65f8175eba8729663a`
[kubernetes-node-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-node-linux-arm64.tar.gz) | `f0f63baaace463dc663c98cbc9a41e52233d1ef33410571ce3f3e78bd485787e`
[kubernetes-node-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-node-linux-arm.tar.gz) | `554bdd11deaf390de85830c7c888dfd4d75d9de8ac147799df12993f27bde905`
[kubernetes-node-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-node-linux-ppc64le.tar.gz) | `913af8ca8b258930e76fd3368acc83608e36e7e270638fa01a6e3be4f682d8bd`
[kubernetes-node-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-node-linux-s390x.tar.gz) | `8192c1c80563230d727fab71514105571afa52cde8520b3d90af58e6daf0e19c`
[kubernetes-node-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.9.1/kubernetes-node-windows-amd64.tar.gz) | `4408e6d741c6008044584c0d7235e608c596e836d51346ee773589d9b4589fdc`

## Changelog since v1.9.0

### Other notable changes

* Compare correct file names for volume detach operation ([#57053](https://github.com/kubernetes/kubernetes/pull/57053), [@prashima](https://github.com/prashima))
* Fixed a garbage collection race condition where objects with ownerRefs pointing to cluster-scoped objects could be deleted incorrectly. ([#57211](https://github.com/kubernetes/kubernetes/pull/57211), [@liggitt](https://github.com/liggitt))
* Free up CPU and memory requested but unused by Metrics Server Pod Nanny. ([#57252](https://github.com/kubernetes/kubernetes/pull/57252), [@kawych](https://github.com/kawych))
* Configurable liveness probe initial delays for etcd and kube-apiserver in GCE ([#57749](https://github.com/kubernetes/kubernetes/pull/57749), [@wojtek-t](https://github.com/wojtek-t))
* Fixed garbage collection hang ([#57503](https://github.com/kubernetes/kubernetes/pull/57503), [@liggitt](https://github.com/liggitt))
* GCE: Fixes ILB creation on automatic networks with manually created subnetworks. ([#57351](https://github.com/kubernetes/kubernetes/pull/57351), [@nicksardo](https://github.com/nicksardo))
* Check for known manifests during preflight instead of only checking for non-empty manifests directory. ([#57287](https://github.com/kubernetes/kubernetes/pull/57287), [@mattkelly](https://github.com/mattkelly))
* enable flexvolume on Windows node ([#56921](https://github.com/kubernetes/kubernetes/pull/56921), [@andyzhangx](https://github.com/andyzhangx))
* change default azure file/dir mode to 0755 ([#56551](https://github.com/kubernetes/kubernetes/pull/56551), [@andyzhangx](https://github.com/andyzhangx))
* fix incorrect error info when creating an azure file PVC failed ([#56550](https://github.com/kubernetes/kubernetes/pull/56550), [@andyzhangx](https://github.com/andyzhangx))
* Retry 'connection refused' errors when setting up clusters on GCE. ([#57394](https://github.com/kubernetes/kubernetes/pull/57394), [@mborsz](https://github.com/mborsz))
* Fixes issue creating docker secrets with kubectl 1.9 for accessing docker private registries. ([#57463](https://github.com/kubernetes/kubernetes/pull/57463), [@dims](https://github.com/dims))
* Fixes a bug where if an error was returned that was not an `autorest.DetailedError` we would return `"not found", nil` which caused nodes to go to `NotReady` state. ([#57484](https://github.com/kubernetes/kubernetes/pull/57484), [@brendandburns](https://github.com/brendandburns))
* Fix Heapster configuration and Metrics Server configuration to enable overriding default resource requirements. ([#56965](https://github.com/kubernetes/kubernetes/pull/56965), [@kawych](https://github.com/kawych))



# v1.9.0

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.0


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes.tar.gz) | `d8a52a97382a418b69d46a8b3946bd95c404e03a2d50489d16b36517c9dbc7f4`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-src.tar.gz) | `95d35ad7d274e5ed207674983c3e8ec28d8190c17e635ee922e2af8349fb031b`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-darwin-386.tar.gz) | `2646aa4badf9281b42b921c1e9e2ed235e1305d331423f252a3380396e0c383f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-darwin-amd64.tar.gz) | `e76e69cf58399c10908afce8bb8d1f12cb8811de7b24e657e5f9fc80e7b9b6fb`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-linux-386.tar.gz) | `bcd5ca428eb78fdaadbcf9ff78d9cbcbf70585a2d2582342a4460e55f3bbad13`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-linux-amd64.tar.gz) | `ba96c8e71dba68b1b3abcad769392fb4df53e402cb65ef25cd176346ee2c39e8`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-linux-arm64.tar.gz) | `80ceae744fbbfc7759c3d95999075f98e5d86d80e53ea83d16fa8e849da4073d`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-linux-arm.tar.gz) | `86b271e2518230f3502708cbe8f188a3a68b913c812247b8cc6fbb4c9f35f6c8`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-linux-ppc64le.tar.gz) | `8b7506ab64ceb2ff470120432d7a6a93adf14e14e612b3c53b3c238d334b55e2`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-linux-s390x.tar.gz) | `c066aa75a99c141410f9b9a78d230aff4a14dee472fe2b17729e902739798831`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-windows-386.tar.gz) | `a315535d6a64842a7c2efbf2bb876c0b73db7efd4c848812af07956c2446f526`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-client-windows-amd64.tar.gz) | `5d2ba1f008253da1a784c8bb5266d026fb6fdac5d22133b51e86d348dbaff49b`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-server-linux-amd64.tar.gz) | `a8d7be19e3b662681dc50dc0085ca12045979530a27d0200cf986ada3eff4d32`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-server-linux-arm64.tar.gz) | `8ef6ad23c60a50b4255ff41db044b2f5922e2a4b0332303065d9e66688a0b026`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-server-linux-arm.tar.gz) | `7cb99cf65553c9637ee6f55821ea3f778873a9912917ebbd6203e06d5effb055`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-server-linux-ppc64le.tar.gz) | `529b0f45a0fc688aa624aa2b850f28807ce2be3ac1660189f20cd3ae864ac064`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-server-linux-s390x.tar.gz) | `692f0c198da712f15ff93a4634c67f9105e3ec603240b50b51a84480ed63e987`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-node-linux-amd64.tar.gz) | `7ff3f526d1c4ec23516a65ecec3b947fd8f52d8c0605473b1a87159399dfeab1`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-node-linux-arm64.tar.gz) | `fada290471467c341734a3cfff63cd0f867aad95623b67096029d76c459bde06`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-node-linux-arm.tar.gz) | `ded3640bef5f9701f7f622de4ed162cd2e5a968e80a6a56b843ba84a0b146fac`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-node-linux-ppc64le.tar.gz) | `a83ebe3b360d33c2190bffd5bf0e2c68268ca2c85e3b5295c1a71ddb517a4f90`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-node-linux-s390x.tar.gz) | `1210efdf35ec5e0b2e96ff7e456e340684ff12dbea36aa255ac592ca7195e168`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0/kubernetes-node-windows-amd64.tar.gz) | `9961ad142abc7e769bbe962aeb30a014065fae83291a2d65bc2da91f04fbf185`

## 1.9 Release Notes

## WARNING: etcd backup strongly recommended

Before updating to 1.9, you are strongly recommended to back up your etcd data. Consult the installation procedure you are using (kargo, kops, kube-up, kube-aws, kubeadm etc) for specific advice.

Some upgrade methods might upgrade etcd from 3.0 to 3.1 automatically when you upgrade from Kubernetes 1.8, unless you specify otherwise. Because [etcd does not support downgrading](https://coreos.com/etcd/docs/latest/upgrades/upgrade_3_1.html), you'll need to either remain on etcd 3.1 or restore from a backup if you want to downgrade back to Kubernetes 1.8.

## Introduction to 1.9.0

Kubernetes version 1.9 includes new features and enhancements, as well as fixes to identified issues. The release notes contain a brief overview of the important changes introduced in this release. The content is organized by Special Interest Group ([SIG](https://github.com/kubernetes/community/blob/master/sig-list.md)).

For initial installations, see the [Setup topics](https://kubernetes.io/docs/setup/pick-right-solution/) in the Kubernetes documentation.

To upgrade to this release from a previous version, first take any actions required [Before Upgrading](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.9.md#before-upgrading).

For more information about this release and for the latest documentation, see the [Kubernetes documentation](https://kubernetes.io/docs/home/).

## Major themes

Kubernetes is developed by community members whose work is organized into
[Special Interest Groups](https://github.com/kubernetes/community/blob/master/sig-list.md), which provide the themes that guide their work. For the 1.9 release, these themes included:

### API Machinery

Extensibility. SIG API Machinery added a new class of admission control webhooks (mutating), and brought the admission control webhooks to beta.

### Apps

The core workloads API, which is composed of the DaemonSet, Deployment, ReplicaSet, and StatefulSet kinds, has been promoted to GA stability in the apps/v1 group version. As such, the apps/v1beta2 group version is deprecated, and all new code should use the kinds in the apps/v1 group version. 

### Auth

SIG Auth focused on extension-related authorization improvements. Permissions can now be added to the built-in RBAC admin/edit/view roles using [cluster role aggregation](https://kubernetes.io/docs/admin/authorization/rbac/#aggregated-clusterroles). [Webhook authorizers](https://kubernetes.io/docs/admin/authorization/webhook/) can now deny requests and short-circuit checking subsequent authorizers. Performance and usability of the beta [PodSecurityPolicy](https://kubernetes.io/docs/concepts/policy/pod-security-policy/) feature was also improved.

### AWS

In v1.9 SIG AWS has improved stability of EBS support across the board. If a Volume is “stuck” in the attaching state to a node for too long a unschedulable taint will be applied to the node, so a Kubernetes admin can [take manual steps to correct the error](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-attaching-volume.html). Users are encouraged to ensure they are monitoring for the taint, and should consider automatically terminating instances in this state.

In addition, support for NVMe disks has been added to Kubernetes, and a service of type LoadBalancer can now be backed with an NLB instead of an ELB (alpha).

### Azure

SIG Azure worked on improvements in the cloud provider, including significant work on the Azure Load Balancer implementation.

### Cluster Lifecycle

SIG Cluster Lifecycle has been focusing on improving kubeadm in order to bring it to GA in a future release, as well as developing the [Cluster API](https://github.com/kubernetes/kube-deploy/tree/master/cluster-api). For kubeadm, most new features, such as support for CoreDNS, IPv6 and Dynamic Kubelet Configuration, have gone in as alpha features. We expect to graduate these features to beta and beyond in the next release. The initial Cluster API spec and GCE sample implementation were developed from scratch during this cycle, and we look forward to stabilizing them into something production-grade during 2018.

### Instrumentation

In v1.9 we focused on improving stability of the components owned by the SIG, including Heapster, Custom Metrics API adapters for Prometheus, and Stackdriver.

### Network

In v1.9 SIG Network has implemented alpha support for IPv6, and alpha support for CoreDNS as a drop-in replacement for kube-dns. Additionally, SIG Network has begun the deprecation process for the extensions/v1beta1 NetworkPolicy API in favor of the networking.k8s.io/v1 equivalent.

### Node

SIG Node iterated on the ability to support more workloads with better performance and improved reliability.  Alpha features were improved around hardware accelerator support, device plugins enablement, and cpu pinning policies to enable us to graduate these features to beta in a future release.  In addition, a number of reliability and performance enhancements were made across the node to help operators in production. 

### OpenStack

In this cycle, SIG OpenStack focused on configuration simplification through smarter defaults and the use of auto-detection wherever feasible (Block Storage API versions, Security Groups) as well as updating API support, including:

*   Block Storage (Cinder) V3 is now supported.
*   Load Balancer (Octavia) V2 is now supported, in addition to Neutron LBaaS V2.
*   Neutron LBaas V1 support has been removed.

This work enables Kubernetes to take full advantage of the relevant services as exposed by OpenStack clouds. Refer to the [Cloud Providers](https://kubernetes.io/docs/concepts/cluster-administration/cloud-providers/#openstack) documentation for more information.

### Storage

[SIG Storage](https://github.com/kubernetes/community/tree/master/sig-storage) is responsible for storage and volume plugin components.

For the 1.9 release, SIG Storage made Kubernetes more pluggable and modular by introducing an alpha implementation of the Container Storage Interface (CSI). CSI will make installing new volume plugins as easy as deploying a pod, and enable third-party storage providers to develop their plugins without the need to add code to the core Kubernetes codebase.

The SIG also focused on adding functionality to the Kubernetes volume subsystem, such as alpha support for exposing volumes as block devices inside containers, extending the alpha volume-resizing support to more volume plugins, and topology-aware volume scheduling.

### Windows

We are advancing support for Windows Server and Windows Server Containers to beta along with continued feature and functional advancements on both the Kubernetes and Windows platforms. This opens the door for many Windows-specific applications and workloads to run on Kubernetes, significantly expanding the implementation scenarios and the enterprise reach of Kubernetes.

## Before Upgrading 

Consider the following changes, limitations, and guidelines before you upgrade:

### **API Machinery**

*   The admission API, which is used when the API server calls admission control webhooks, is moved from `admission.v1alpha1` to `admission.v1beta1`. You must **delete any existing webhooks before you upgrade** your cluster, and update them to use the latest API. This change is not backward compatible.
*   The admission webhook configurations API, part of the admissionregistration API, is now at v1beta1. Delete any existing webhook configurations before you upgrade, and update your configuration files to use the latest API. For this and the previous change, see also [the documentation](https://kubernetes.io/docs/admin/extensible-admission-controllers/#external-admission-webhooks).
*   A new `ValidatingAdmissionWebhook` is added (replacing `GenericAdmissionWebhook`) and is available in the generic API server. You must update your API server configuration file to pass the webhook to the `--admission-control` flag. ([#55988](https://github.com/kubernetes/kubernetes/pull/55988),[ @caesarxuchao](https://github.com/caesarxuchao))  ([#54513](https://github.com/kubernetes/kubernetes/pull/54513),[ @deads2k](https://github.com/deads2k))
*   The deprecated options `--portal-net` and `--service-node-ports` for the API server are removed. ([#52547](https://github.com/kubernetes/kubernetes/pull/52547),[ @xiangpengzhao](https://github.com/xiangpengzhao))

### **Auth**

*   PodSecurityPolicy: A compatibility issue with the allowPrivilegeEscalation field that caused policies to start denying pods they previously allowed was fixed. If you defined PodSecurityPolicy objects using a 1.8.0 client or server and set allowPrivilegeEscalation to false, these objects must be reapplied after you upgrade. ([#53443](https://github.com/kubernetes/kubernetes/pull/53443),[ @liggitt](https://github.com/liggitt))
*   KMS: Alpha integration with GCP KMS was removed in favor of a future out-of-process extension point. Discontinue use of the GCP KMS integration and ensure [data has been decrypted](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/#decrypting-all-data) (or reencrypted with a different provider) before upgrading ([#54759](https://github.com/kubernetes/kubernetes/pull/54759),[ @sakshamsharma](https://github.com/sakshamsharma))

### **CLI**

*   Swagger 1.2 validation is removed for kubectl. The options `--use-openapi` and `--schema-cache-dir` are also removed because they are no longer needed. ([#53232](https://github.com/kubernetes/kubernetes/pull/53232),[ @apelisse](https://github.com/apelisse))

### **Cluster Lifecycle**

*   You must either specify the `--discovery-token-ca-cert-hash` flag to `kubeadm join`, or opt out of the CA pinning feature using `--discovery-token-unsafe-skip-ca-verification`.
*   The default `auto-detect` behavior of the kubelet's `--cloud-provider` flag is removed.
    *   You can manually set `--cloud-provider=auto-detect`, but be aware that this behavior will be removed completely in a future version.
    *   Best practice for version 1.9 and future versions is to explicitly set a cloud-provider. See [the documentation](https://kubernetes.io/docs/getting-started-guides/scratch/#cloud-providers)
*   The kubeadm `--skip-preflight-checks` flag is now deprecated and will be removed in a future release.
*   If you are using the cloud provider API to determine the external host address of the apiserver, set `--external-hostname` explicitly instead. The cloud provider detection has been deprecated and will be removed in the future ([#54516](https://github.com/kubernetes/kubernetes/pull/54516),[ @dims](https://github.com/dims))

### **Multicluster**

*   Development of Kubernetes Federation has moved to [github.com/kubernetes/federation](github.com/kubernetes/federation). This move out of tree also means that Federation will begin releasing separately from Kubernetes. Impact: 
    *   Federation-specific behavior will no longer be included in kubectl
    *   kubefed will no longer be released as part of Kubernetes
    *   The Federation servers will no longer be included in the hyperkube binary and image. ([#53816](https://github.com/kubernetes/kubernetes/pull/53816),[ @marun](https://github.com/marun))

### **Node**

*   The kubelet `--network-plugin-dir` flag is removed. This flag was deprecated in version 1.7, and is replaced with `--cni-bin-dir`. ([#53564](https://github.com/kubernetes/kubernetes/pull/53564),[ @supereagle](https://github.com/supereagle))
*   kubelet's `--cloud-provider` flag no longer defaults to "auto-detect". If you want cloud-provider support in kubelet, you must set a specific cloud-provider explicitly. ([#53573](https://github.com/kubernetes/kubernetes/pull/53573),[ @dims](https://github.com/dims))

### **Network**

*   NetworkPolicy objects are now stored in etcd in v1 format. After you upgrade to version 1.9, make sure that all NetworkPolicy objects are migrated to v1. ([#51955](https://github.com/kubernetes/kubernetes/pull/51955), [@danwinship](https://github.com/danwinship))
*   The API group/version for the kube-proxy configuration has changed from `componentconfig/v1alpha1` to `kubeproxy.config.k8s.io/v1alpha1`. If you are using a config file for kube-proxy instead of the command line flags, you must change its apiVersion to `kubeproxy.config.k8s.io/v1alpha1`. ([#53645](https://github.com/kubernetes/kubernetes/pull/53645), [@xiangpengzhao](https://github.com/xiangpengzhao))
*   The "ServiceNodeExclusion" feature gate must now be enabled for the `alpha.service-controller.kubernetes.io/exclude-balancer` annotation on nodes to be honored. ([#54644](https://github.com/kubernetes/kubernetes/pull/54644),[ @brendandburns](https://github.com/brendandburns))

### **Scheduling**

*   Taint key `unreachable` is now in GA.
*   Taint key `notReady` is changed to `not-ready`, and is also now in GA.
*   These changes are automatically updated for taints. Tolerations for these taints must be updated manually. Specifically, you must:
    *   Change `node.alpha.kubernetes.io/notReady` to `node.kubernetes.io/not-ready`
    *   Change `node.alpha.kubernetes.io/unreachable` to  `node.kubernetes.io/unreachable`
*   The `node.kubernetes.io/memory-pressure` taint now respects the configured whitelist.  To use it, you must add it to the whitelist.([#55251](https://github.com/kubernetes/kubernetes/pull/55251),[ @deads2k](https://github.com/deads2k))
*   Refactor kube-scheduler configuration ([#52428](https://github.com/kubernetes/kubernetes/pull/52428))
    *   The kube-scheduler command now supports a --config flag which is the location of a file containing a serialized scheduler configuration. Most other kube-scheduler flags are now deprecated. ([#52562](https://github.com/kubernetes/kubernetes/pull/52562),[ @ironcladlou](https://github.com/ironcladlou))
*   Opaque integer resources (OIR), which were (deprecated in v1.8.), have been removed. ([#55103](https://github.com/kubernetes/kubernetes/pull/55103),[ @ConnorDoyle](https://github.com/ConnorDoyle))

### **Storage**

*   [alpha] The LocalPersistentVolumes alpha feature now also requires the VolumeScheduling alpha feature.  This is a breaking change, and the following changes are required: 
    *   The VolumeScheduling feature gate must also be enabled on kube-scheduler and kube-controller-manager components.
    *   The NoVolumeNodeConflict predicate has been removed.  For non-default schedulers, update your scheduler policy.
    *   The CheckVolumeBinding predicate must be enabled in non-default schedulers. ([#55039](https://github.com/kubernetes/kubernetes/pull/55039),[ @msau42](https://github.com/msau42))

### **OpenStack**

*   Remove the LbaasV1 of OpenStack cloud provider, currently only support LbaasV2. ([#52717](https://github.com/kubernetes/kubernetes/pull/52717),[ @FengyunPan](https://github.com/FengyunPan))

## Known Issues

This section contains a list of known issues reported in Kubernetes 1.9 release. The content is populated from the [v1.9.x known issues and FAQ accumulator](https://github.com/kubernetes/kubernetes/issues/57159](https://github.com/kubernetes/kubernetes/issues/57159).

*   If you are adding Windows Server Virtual Machines as nodes to your Kubernetes environment, there is a compatibility issue with certain virtualization products. Specifically the Windows version of the kubelet.exe calls `GetPhysicallyInstalledSystemMemory` to get the physical memory installed on Windows machines and reports it as part of node metrics to heapster. This API call fails for VMware and VirtualBox virtualization environments. This issue is not present in bare metal Windows deployments, in Hyper-V, or on some of the popular public cloud providers.

*   If you run `kubectl get po` while the API server in unreachable, a misleading error is returned: `the server doesn't have a resource type "po"`. To work around this issue, specify the full resource name in the command instead of the abbreviation: `kubectl get pods`. This issue will be fixed in a future release.

  For more information, see [#57198](https://github.com/kubernetes/kubernetes/issues/57198).

*   Mutating and validating webhook configurations are continuously polled by the API server (once per second). This issue will be fixed in a future release.

  For more information, see [#56357](https://github.com/kubernetes/kubernetes/issues/56357).

*   Audit logging is slow because writes to the log are performed synchronously with requests to the log. This issue will be fixed in a future release.

  For more information, see [#53006](https://github.com/kubernetes/kubernetes/issues/53006).

*   Custom Resource Definitions (CRDs) are not properly deleted under certain conditions. This issue will be fixed in a future release.

  For more information, see [#56348](https://github.com/kubernetes/kubernetes/issues/56348).

*   API server times out after performing a rolling update of the etcd cluster. This issue will be fixed in a future release.

  For more information, see [#47131](https://github.com/kubernetes/kubernetes/issues/47131)

*   If a namespaced resource is owned by a cluster scoped resource, and the namespaced dependent is processed before the cluster scoped owner has ever been observed by the garbage collector, the dependent will be erroneously deleted.

  For more information, see [#54940](https://github.com/kubernetes/kubernetes/issues/54940)

## Deprecations

This section provides an overview of deprecated API versions, options, flags, and arguments. Deprecated means that we intend to remove the capability from a future release. After removal, the capability will no longer work. The sections are organized by SIGs.

### **API Machinery**

*   The kube-apiserver `--etcd-quorum-read` flag is deprecated and the ability to switch off quorum read will be removed in a future release. ([#53795](https://github.com/kubernetes/kubernetes/pull/53795),[ @xiangpengzhao](https://github.com/xiangpengzhao))
*   The `/ui` redirect in kube-apiserver is deprecated and will be removed in Kubernetes 1.10. ([#53046](https://github.com/kubernetes/kubernetes/pull/53046), [@maciaszczykm](https://github.com/maciaszczykm))
*   `etcd2` as a backend is deprecated and support will be removed in Kubernetes 1.13 or 1.14.

### **Auth**

*   Default controller-manager options for `--cluster-signing-cert-file` and `--cluster-signing-key-file` are deprecated and will be removed in a future release. ([#54495](https://github.com/kubernetes/kubernetes/pull/54495),[ @mikedanese](https://github.com/mikedanese))
*   RBAC objects are now stored in etcd in v1 format. After upgrading to 1.9, ensure all RBAC objects (Roles, RoleBindings, ClusterRoles, ClusterRoleBindings) are at v1. v1alpha1 support is deprecated and will be removed in a future release. ([#52950](https://github.com/kubernetes/kubernetes/pull/52950),[ @liggitt](https://github.com/liggitt))

### **Cluster Lifecycle**

*   kube-apiserver: `--ssh-user` and `--ssh-keyfile` are now deprecated and will be removed in a future release. Users of SSH tunnel functionality in Google Container Engine for the Master -> Cluster communication should plan alternate methods for bridging master and node networks. ([#54433](https://github.com/kubernetes/kubernetes/pull/54433),[ @dims](https://github.com/dims))
*   The kubeadm `--skip-preflight-checks` flag is now deprecated and will be removed in a future release.
*   If you are using the cloud provider API to determine the external host address of the apiserver, set `--external-hostname` explicitly instead. The cloud provider detection has been deprecated and will be removed in the future ([#54516](https://github.com/kubernetes/kubernetes/pull/54516),[ @dims](https://github.com/dims))

### **Network**

*   The NetworkPolicy extensions/v1beta1 API is now deprecated and will be removed in a future release. This functionality has been migrated to a dedicated v1 API - networking.k8s.io/v1. v1beta1 Network Policies can be upgraded to the v1 API with the [cluster/update-storage-objects.sh script](https://github.com/danwinship/kubernetes/blob/master/cluster/update-storage-objects.sh). Documentation can be found [here](https://kubernetes.io/docs/concepts/services-networking/network-policies/). ([#56425](https://github.com/kubernetes/kubernetes/pull/56425), [@cmluciano](https://github.com/cmluciano))

### **Storage**

*   The `volume.beta.kubernetes.io/storage-class` annotation is deprecated. It will be removed in a future release. For the StorageClass API object, use v1, and in place of the annotation use `v1.PersistentVolumeClaim.Spec.StorageClassName` and `v1.PersistentVolume.Spec.StorageClassName` instead. ([#53580](https://github.com/kubernetes/kubernetes/pull/53580),[ @xiangpengzhao](https://github.com/xiangpengzhao))

### **Scheduling**

*   The kube-scheduler command now supports a `--config` flag, which is the location of a file containing a serialized scheduler configuration. Most other kube-scheduler flags are now deprecated. ([#52562](https://github.com/kubernetes/kubernetes/pull/52562),[ @ironcladlou](https://github.com/ironcladlou))

### **Node**

*   The kubelet's `--enable-custom-metrics` flag is now deprecated. ([#54154](https://github.com/kubernetes/kubernetes/pull/54154),[ @mtaufen](https://github.com/mtaufen))

## Notable Changes

### **Workloads API (apps/v1)**

As announced with the release of version 1.8, the Kubernetes Workloads API is at v1 in version 1.9. This API consists of the DaemonSet, Deployment, ReplicaSet and StatefulSet kinds.

### **API Machinery**

#### **Admission Control**

*   Admission webhooks are now in beta, and include the following:
    *   Mutation support for admission webhooks. ([#54892](https://github.com/kubernetes/kubernetes/pull/54892),[ @caesarxuchao](https://github.com/caesarxuchao))
    *   Webhook admission now takes a config file that describes how to authenticate to webhook servers ([#54414](https://github.com/kubernetes/kubernetes/pull/54414),[ @deads2k](https://github.com/deads2k))
    *   The dynamic admission webhook now supports a URL in addition to a service reference, to accommodate out-of-cluster webhooks. ([#54889](https://github.com/kubernetes/kubernetes/pull/54889),[ @lavalamp](https://github.com/lavalamp))
    *   Added `namespaceSelector` to `externalAdmissionWebhook` configuration to allow applying webhooks only to objects in the namespaces that have matching labels. ([#54727](https://github.com/kubernetes/kubernetes/pull/54727),[ @caesarxuchao](https://github.com/caesarxuchao))
*   Metrics are added for monitoring admission plugins, including the new dynamic (webhook-based) ones. ([#55183](https://github.com/kubernetes/kubernetes/pull/55183),[ @jpbetz](https://github.com/jpbetz))
*   The PodSecurityPolicy annotation kubernetes.io/psp on pods is set only once on create. ([#55486](https://github.com/kubernetes/kubernetes/pull/55486),[ @sttts](https://github.com/sttts))

#### **API & API server**

*   Fixed a bug related to discovery information for scale subresources in the apps API group ([#54683](https://github.com/kubernetes/kubernetes/pull/54683),[ @liggitt](https://github.com/liggitt))
*   Fixed a bug that prevented client-go metrics from being registered in Prometheus. This bug affected multiple components. ([#53434](https://github.com/kubernetes/kubernetes/pull/53434),[ @crassirostris](https://github.com/crassirostris))

#### **Audit**

*   Fixed a bug so that `kube-apiserver` now waits for open connections to finish before exiting. This fix provides graceful shutdown and ensures that the audit backend no longer drops events on shutdown. ([#53695](https://github.com/kubernetes/kubernetes/pull/53695),[ @hzxuzhonghu](https://github.com/hzxuzhonghu))
*   Webhooks now always retry sending if a connection reset error is returned. ([#53947](https://github.com/kubernetes/kubernetes/pull/53947),[ @crassirostris](https://github.com/crassirostris))

#### **Custom Resources**

*   Validation of resources defined by a Custom Resource Definition (CRD) is now in beta ([#54647](https://github.com/kubernetes/kubernetes/pull/54647),[ @colemickens](https://github.com/colemickens))
*   An example CRD controller has been added, at [github.com/kubernetes/sample-controller](github.com/kubernetes/sample-controller). ([#52753](https://github.com/kubernetes/kubernetes/pull/52753),[ @munnerz](https://github.com/munnerz))
*   Custom resources served by CustomResourceDefinition objects now support field selectors for `metadata.name` and `metadata.namespace`. Also fixed an issue with watching a single object; earlier versions could watch only a collection, and so a watch on an instance would fail.  ([#53345](https://github.com/kubernetes/kubernetes/pull/53345),[ @ncdc](https://github.com/ncdc))

#### **Other**

*   `kube-apiserver` now runs with the default value for `service-cluster-ip-range` ([#52870](https://github.com/kubernetes/kubernetes/pull/52870),[ @jennybuckley](https://github.com/jennybuckley))
*   Add `--etcd-compaction-interval` to apiserver for controlling request of compaction to etcd3 from apiserver. ([#51765](https://github.com/kubernetes/kubernetes/pull/51765),[ @mitake](https://github.com/mitake))
*   The httpstream/spdy calls now support CIDR notation for NO_PROXY ([#54413](https://github.com/kubernetes/kubernetes/pull/54413),[ @kad](https://github.com/kad))
*   Code generation for CRD and User API server types is improved with the addition of two new scripts to k8s.io/code-generator: `generate-groups.sh` and `generate-internal-groups.sh`. ([#52186](https://github.com/kubernetes/kubernetes/pull/52186),[ @sttts](https://github.com/sttts))
*   [beta] Flag `--chunk-size={SIZE}` is added to `kubectl get` to customize the number of results returned in large lists of resources. This reduces the perceived latency of managing large clusters because the server returns the first set of results to the client much more quickly. Pass 0 to disable this feature.([#53768](https://github.com/kubernetes/kubernetes/pull/53768),[ @smarterclayton](https://github.com/smarterclayton))
*   [beta] API chunking via the limit and continue request parameters is promoted to beta in this release. Client libraries using the Informer or ListWatch types will automatically opt in to chunking. ([#52949](https://github.com/kubernetes/kubernetes/pull/52949),[ @smarterclayton](https://github.com/smarterclayton))
*   The `--etcd-quorum-read` flag now defaults to true to ensure correct operation with HA etcd clusters. This flag is deprecated and the flag will be removed in future versions, as well as the ability to turn off this functionality. ([#53717](https://github.com/kubernetes/kubernetes/pull/53717),[ @liggitt](https://github.com/liggitt))
*   Add events.k8s.io api group with v1beta1 API containing redesigned event type. ([#49112](https://github.com/kubernetes/kubernetes/pull/49112),[ @gmarek](https://github.com/gmarek))
*   Fixed a bug where API discovery failures were crashing the kube controller manager via the garbage collector. ([#55259](https://github.com/kubernetes/kubernetes/pull/55259),[ @ironcladlou](https://github.com/ironcladlou))
*   `conversion-gen` is now usable in a context without a vendored k8s.io/kubernetes. The Kubernetes core API is removed from `default extra-peer-dirs`. ([#54394](https://github.com/kubernetes/kubernetes/pull/54394),[ @sttts](https://github.com/sttts))
*   Fixed a bug where the `client-gen` tag for code-generator required a newline between a comment block and a statement.  tag shortcomings when newline is omitted ([#53893](https://github.com/kubernetes/kubernetes/pull/53893)) ([#55233](https://github.com/kubernetes/kubernetes/pull/55233),[ @sttts](https://github.com/sttts))
*   The Apiserver proxy now rewrites the URL when a service returns an absolute path with the request's host. ([#52556](https://github.com/kubernetes/kubernetes/pull/52556),[ @roycaihw](https://github.com/roycaihw))
*   The gRPC library is updated to pick up data race fix ([#53124](https://github.com/kubernetes/kubernetes/pull/53124)) ([#53128](https://github.com/kubernetes/kubernetes/pull/53128),[ @dixudx](https://github.com/dixudx))
*   Fixed server name verification of aggregated API servers and webhook admission endpoints ([#56415](https://github.com/kubernetes/kubernetes/pull/56415),[ @liggitt](https://github.com/liggitt))

### **Apps**

*   The `kubernetes.io/created-by` annotation is no longer added to controller-created objects. Use the `metadata.ownerReferences` item with controller set to `true` to determine which controller, if any, owns an object. ([#54445](https://github.com/kubernetes/kubernetes/pull/54445),[ @crimsonfaith91](https://github.com/crimsonfaith91))
*   StatefulSet controller now creates a label for each Pod in a StatefulSet. The label is `statefulset.kubernetes.io/pod-name`, where `pod-name` = the name of the Pod. This allows users to create a Service per Pod to expose a connection to individual Pods. ([#55329](https://github.com/kubernetes/kubernetes/pull/55329),[ @kow3ns](https://github.com/kow3ns))
*   DaemonSet status includes a new field named `conditions`, making it consistent with other workloads controllers. ([#55272](https://github.com/kubernetes/kubernetes/pull/55272),[ @janetkuo](https://github.com/janetkuo))
*   StatefulSet status now supports conditions, making it consistent with other core controllers in v1 ([#55268](https://github.com/kubernetes/kubernetes/pull/55268),[ @foxish](https://github.com/foxish))
*   The default garbage collection policy for Deployment, DaemonSet, StatefulSet, and ReplicaSet has changed from OrphanDependents to DeleteDependents when the deletion is requested through an `apps/v1` endpoint.  ([#55148](https://github.com/kubernetes/kubernetes/pull/55148),[ @dixudx](https://github.com/dixudx))
    *   Clients using older endpoints will be unaffected. This change is only at the REST API level and is independent of the default behavior of particular clients (e.g. this does not affect the default for the kubectl `--cascade` flag).
    *   If you upgrade your client-go libs and use the `AppsV1()` interface, please note that the default garbage collection behavior is changed.

### **Auth**

#### **Audit**

*   RequestReceivedTimestamp and StageTimestamp are added to audit events ([#52981](https://github.com/kubernetes/kubernetes/pull/52981),[ @CaoShuFeng](https://github.com/CaoShuFeng))
*   Advanced audit policy now supports a policy wide omitStage ([#54634](https://github.com/kubernetes/kubernetes/pull/54634),[ @CaoShuFeng](https://github.com/CaoShuFeng))

#### **RBAC**

*   New permissions have been added to default RBAC roles ([#52654](https://github.com/kubernetes/kubernetes/pull/52654),[ @liggitt](https://github.com/liggitt)):
    *   The default admin and edit roles now include read/write permissions
    *   The view role includes read permissions on poddisruptionbudget.policy resources. 
*   RBAC rules can now match the same subresource on any resource using the form `*/(subresource)`. For example, `*/scale` matches requests to `replicationcontroller/scale`. ([#53722](https://github.com/kubernetes/kubernetes/pull/53722),[ @deads2k](https://github.com/deads2k))
*   The RBAC bootstrapping policy now allows authenticated users to create selfsubjectrulesreviews. ([#56095](https://github.com/kubernetes/kubernetes/pull/56095),[ @ericchiang](https://github.com/ericchiang))
*   RBAC ClusterRoles can now select other roles to aggregate. ([#54005](https://github.com/kubernetes/kubernetes/pull/54005),[ @deads2k](https://github.com/deads2k))
*   Fixed an issue with RBAC reconciliation that caused duplicated subjects in some bootstrapped RoleBinding objects on each restart of the API server. ([#53239](https://github.com/kubernetes/kubernetes/pull/53239),[ @enj](https://github.com/enj))

#### **Other**

*   Pod Security Policy can now manage access to specific FlexVolume drivers ([#53179](https://github.com/kubernetes/kubernetes/pull/53179),[ @wanghaoran1988](https://github.com/wanghaoran1988))
*   Audit policy files without apiVersion and kind are treated as invalid. ([#54267](https://github.com/kubernetes/kubernetes/pull/54267),[ @ericchiang](https://github.com/ericchiang))
*   Fixed a bug that where forbidden errors were encountered when accessing ReplicaSet and DaemonSets objects via the apps API group. ([#54309](https://github.com/kubernetes/kubernetes/pull/54309),[ @liggitt](https://github.com/liggitt))
*   Improved PodSecurityPolicy admission latency. ([#55643](https://github.com/kubernetes/kubernetes/pull/55643),[ @tallclair](https://github.com/tallclair))
*   kube-apiserver: `--oidc-username-prefix` and `--oidc-group-prefix` flags are now correctly enabled. ([#56175](https://github.com/kubernetes/kubernetes/pull/56175),[ @ericchiang](https://github.com/ericchiang))
*   If multiple PodSecurityPolicy objects allow a submitted pod, priority is given to policies that do not require default values for any fields in the pod spec. If default values are required, the first policy ordered by name that allows the pod is used. ([#52849](https://github.com/kubernetes/kubernetes/pull/52849),[ @liggitt](https://github.com/liggitt))
*   A new controller automatically cleans up Certificate Signing Requests that are Approved and Issued, or Denied. ([#51840](https://github.com/kubernetes/kubernetes/pull/51840),[ @jcbsmpsn](https://github.com/jcbsmpsn))
*   PodSecurityPolicies have been added for all in-tree cluster addons ([#55509](https://github.com/kubernetes/kubernetes/pull/55509),[ @tallclair](https://github.com/tallclair))

#### **GCE**

*   Added support for PodSecurityPolicy on GCE: `ENABLE_POD_SECURITY_POLICY=true` enables the admission controller, and installs policies for default addons. ([#52367](https://github.com/kubernetes/kubernetes/pull/52367),[ @tallclair](https://github.com/tallclair))

### **Autoscaling**

*   HorizontalPodAutoscaler objects now properly functions on scalable resources in any API group. Fixed by adding a polymorphic scale client. ([#53743](https://github.com/kubernetes/kubernetes/pull/53743),[ @DirectXMan12](https://github.com/DirectXMan12))
*   Fixed a set of minor issues with Cluster Autoscaler 1.0.1 ([#54298](https://github.com/kubernetes/kubernetes/pull/54298),[ @mwielgus](https://github.com/mwielgus))
*   HPA tolerance is now configurable by setting the `horizontal-pod-autoscaler-tolerance` flag. ([#52275](https://github.com/kubernetes/kubernetes/pull/52275),[ @mattjmcnaughton](https://github.com/mattjmcnaughton))
*   Fixed a bug that allowed the horizontal pod autoscaler to allocate more `desiredReplica` objects than `maxReplica` objects in certain instances. ([#53690](https://github.com/kubernetes/kubernetes/pull/53690),[ @mattjmcnaughton](https://github.com/mattjmcnaughton))

### **AWS**

*   Nodes can now use instance types (such as C5) that use NVMe. ([#56607](https://github.com/kubernetes/kubernetes/pull/56607), [@justinsb](https://github.com/justinsb))
*   Nodes are now unreachable if volumes are stuck in the attaching state. Implemented by applying a taint to the node. ([#55558](https://github.com/kubernetes/kubernetes/pull/55558),[ @gnufied](https://github.com/gnufied))
*   Volumes are now checked for available state before attempting to attach or delete a volume in EBS. ([#55008](https://github.com/kubernetes/kubernetes/pull/55008),[ @gnufied](https://github.com/gnufied))
*   Fixed a bug where error log messages were breaking into two lines. ([#49826](https://github.com/kubernetes/kubernetes/pull/49826),[ @dixudx](https://github.com/dixudx))
*   Fixed a bug so that volumes are now detached from stopped nodes. ([#55893](https://github.com/kubernetes/kubernetes/pull/55893),[ @gnufied](https://github.com/gnufied))
*   You can now override the health check parameters for AWS ELBs by specifying annotations on the corresponding service. The new annotations are: `healthy-threshold`, `unhealthy-threshold`, `timeout`, `interval`. The prefix for all annotations is  `service.beta.kubernetes.io/aws-load-balancer-healthcheck-`. ([#56024](https://github.com/kubernetes/kubernetes/pull/56024),[ @dimpavloff](https://github.com/dimpavloff))
*   Fixed a bug so that AWS ECR credentials are now supported in the China region. ([#50108](https://github.com/kubernetes/kubernetes/pull/50108),[ @zzq889](https://github.com/zzq889))
*   Added Amazon NLB support ([#53400](https://github.com/kubernetes/kubernetes/pull/53400),[ @micahhausler](https://github.com/micahhausler))
*   Additional annotations are now properly set or updated  for AWS load balancers ([#55731](https://github.com/kubernetes/kubernetes/pull/55731),[ @georgebuckerfield](https://github.com/georgebuckerfield))
*   AWS SDK is updated to version 1.12.7 ([#53561](https://github.com/kubernetes/kubernetes/pull/53561),[ @justinsb](https://github.com/justinsb))

### **Azure**

*   Fixed several issues with properly provisioning Azure disk storage ([#55927](https://github.com/kubernetes/kubernetes/pull/55927),[ @andyzhangx](https://github.com/andyzhangx))
*   A new service annotation `service.beta.kubernetes.io/azure-dns-label-name` now sets the Azure DNS label for a public IP address. ([#47849](https://github.com/kubernetes/kubernetes/pull/47849),[ @tomerf](https://github.com/tomerf))
*   Support for GetMountRefs function added; warning messages no longer displayed. ([#54670](https://github.com/kubernetes/kubernetes/pull/54670), [#52401](https://github.com/kubernetes/kubernetes/pull/52401),[ @andyzhangx](https://github.com/andyzhangx))
*   Fixed an issue where an Azure PersistentVolume object would crash because the value of `volumeSource.ReadOnly` was set to nil. ([#54607](https://github.com/kubernetes/kubernetes/pull/54607),[ @andyzhangx](https://github.com/andyzhangx))
*   Fixed an issue with Azure disk mount failures on CoreOS and some other distros ([#54334](https://github.com/kubernetes/kubernetes/pull/54334),[ @andyzhangx](https://github.com/andyzhangx))
*   GRS, RAGRS storage account types are now supported for Azure disks. ([#55931](https://github.com/kubernetes/kubernetes/pull/55931),[ @andyzhangx](https://github.com/andyzhangx))
*   Azure NSG rules are now restricted so that external access is allowed only to the load balancer IP. ([#54177](https://github.com/kubernetes/kubernetes/pull/54177),[ @itowlson](https://github.com/itowlson))
*   Azure NSG rules can be consolidated to reduce the likelihood of hitting Azure resource limits (available only in regions where the Augmented Security Groups preview is available).  ([#55740](https://github.com/kubernetes/kubernetes/pull/55740), [@itowlson](https://github.com/itowlson))
*   The Azure SDK is upgraded to v11.1.1. ([#54971](https://github.com/kubernetes/kubernetes/pull/54971),[ @itowlson](https://github.com/itowlson))
*   You can now create Windows mount paths  ([#51240](https://github.com/kubernetes/kubernetes/pull/51240),[ @andyzhangx](https://github.com/andyzhangx))
*   Fixed a controller manager crash issue on a manually created k8s cluster. ([#53694](https://github.com/kubernetes/kubernetes/pull/53694),[ @andyzhangx](https://github.com/andyzhangx))
*   Azure-based clusters now support unlimited mount points. ([#54668](https://github.com/kubernetes/kubernetes/pull/54668)) ([#53629](https://github.com/kubernetes/kubernetes/pull/53629),[ @andyzhangx](https://github.com/andyzhangx))
*   Load balancer reconciliation now considers NSG rules based not only on Name, but also on Protocol, SourcePortRange, DestinationPortRange, SourceAddressPrefix, DestinationAddressPrefix, Access, and Direction. This change makes it possible to update NSG rules under more conditions.  ([#55752](https://github.com/kubernetes/kubernetes/pull/55752),[ @kevinkim9264](https://github.com/kevinkim9264))
*   Custom mountOptions for the azurefile StorageClass object are now respected. Specifically, `dir_mode` and `file_mode` can now be customized. ([#54674](https://github.com/kubernetes/kubernetes/pull/54674),[ @andyzhangx](https://github.com/andyzhangx))
*   Azure Load Balancer Auto Mode: Services can be annotated to allow auto selection of available load balancers and to provide specific availability sets that host the load balancers (for example, `service.beta.kubernetes.io/azure-load-balancer-mode=auto|as1,as2...`)

### **CLI**

#### **Kubectl**

*   `kubectl cp` can now copy a remote file into a local directory. ([#46762](https://github.com/kubernetes/kubernetes/pull/46762),[ @bruceauyeung](https://github.com/bruceauyeung))
*   `kubectl cp` now honors destination names for directories. A complete directory is now copied; in previous versions only the file contents were copied. ([#51215](https://github.com/kubernetes/kubernetes/pull/51215),[ @juanvallejo](https://github.com/juanvallejo))
*   You can now use `kubectl get` with a fieldSelector. ([#50140](https://github.com/kubernetes/kubernetes/pull/50140),[ @dixudx](https://github.com/dixudx))
*   Secret data containing Docker registry auth objects is now generated using the config.json format ([#53916](https://github.com/kubernetes/kubernetes/pull/53916),[ @juanvallejo](https://github.com/juanvallejo))
*   `kubectl apply` now calculates the diff between the current and new configurations based on the OpenAPI spec. If the OpenAPI spec is not available, it falls back to baked-in types. ([#51321](https://github.com/kubernetes/kubernetes/pull/51321),[ @mengqiy](https://github.com/mengqiy))
*   `kubectl explain` now explains `apiservices` and `customresourcedefinition`. (Updated to use OpenAPI instead of Swagger 1.2.) ([#53228](https://github.com/kubernetes/kubernetes/pull/53228),[ @apelisse](https://github.com/apelisse))
*   `kubectl get` now uses OpenAPI schema extensions by default to select columns for custom types. ([#53483](https://github.com/kubernetes/kubernetes/pull/53483),[ @apelisse](https://github.com/apelisse))
*   kubectl `top node` now sorts by name and `top pod` sorts by namespace. Fixed a bug where results were inconsistently sorted. ([#53560](https://github.com/kubernetes/kubernetes/pull/53560),[ @dixudx](https://github.com/dixudx))
*   Added --dry-run option to kubectl drain. ([#52440](https://github.com/kubernetes/kubernetes/pull/52440),[ @juanvallejo](https://github.com/juanvallejo))
*   Kubectl now outputs <none> for columns specified by -o custom-columns but not found in object, rather than "xxx is not found" ([#51750](https://github.com/kubernetes/kubernetes/pull/51750),[ @jianhuiz](https://github.com/jianhuiz))
*   `kubectl create pdb` no longer sets the min-available field by default. ([#53047](https://github.com/kubernetes/kubernetes/pull/53047),[ @yuexiao-wang](https://github.com/yuexiao-wang))
*   The canonical pronunciation of kubectl is "cube control".
*   Added --raw to kubectl create to POST using the normal transport. ([#54245](https://github.com/kubernetes/kubernetes/pull/54245),[ @deads2k](https://github.com/deads2k))
*   Added kubectl `create priorityclass` subcommand ([#54858](https://github.com/kubernetes/kubernetes/pull/54858),[ @wackxu](https://github.com/wackxu))
*   Fixed an issue where `kubectl set` commands occasionally encountered conversion errors for ReplicaSet and DaemonSet objects ([#53158](https://github.com/kubernetes/kubernetes/pull/53158),[ @liggitt](https://github.com/liggitt))

### **Cluster Lifecycle**

#### **API Server**

*   [alpha] Added an `--endpoint-reconciler-type` command-line argument to select the endpoint reconciler to use. The default is to use the 'master-count' reconciler which is the default for 1.9 and in use prior to 1.9. The 'lease' reconciler stores endpoints within the storage api for better cleanup of deleted (or removed) API servers. The 'none' reconciler is a no-op reconciler, which can be used in self-hosted environments.   ([#51698](https://github.com/kubernetes/kubernetes/pull/51698), [@rphillips](https://github.com/rphillips))

#### **Cloud Provider Integration**

*   Added `cloud-controller-manager` to `hyperkube`. This is useful as a number of deployment tools run all of the kubernetes components from the `hyperkube `image/binary. It also makes testing easier as a single binary/image can be built and pushed quickly.  ([#54197](https://github.com/kubernetes/kubernetes/pull/54197),[ @colemickens](https://github.com/colemickens))
*   Added the concurrent service sync flag to the Cloud Controller Manager to allow changing the number of workers. (`--concurrent-service-syncs`) ([#55561](https://github.com/kubernetes/kubernetes/pull/55561),[ @jhorwit2](https://github.com/jhorwit2))
*   kubelet's --cloud-provider flag no longer defaults to "auto-detect". If you want cloud-provider support in kubelet, you must set a specific cloud-provider explicitly. ([#53573](https://github.com/kubernetes/kubernetes/pull/53573),[ @dims](https://github.com/dims))

#### **Kubeadm**

*   kubeadm health checks can now be skipped with `--ignore-preflight-errors`; the `--skip-preflight-checks` flag is now deprecated and will be removed in a future release. ([#56130](https://github.com/kubernetes/kubernetes/pull/56130),[ @anguslees](https://github.com/anguslees)) ([#56072](https://github.com/kubernetes/kubernetes/pull/56072),[ @kad](https://github.com/kad))
*   You now have the option to use CoreDNS instead of KubeDNS. To install CoreDNS instead of kube-dns, set CLUSTER_DNS_CORE_DNS to 'true'. This support is experimental. ([#52501](https://github.com/kubernetes/kubernetes/pull/52501),[ @rajansandeep](https://github.com/rajansandeep)) ([#55728](https://github.com/kubernetes/kubernetes/pull/55728),[ @rajansandeep](https://github.com/rajansandeep))
*   Added --print-join-command flag for kubeadm token create. ([#56185](https://github.com/kubernetes/kubernetes/pull/56185),[ @mattmoyer](https://github.com/mattmoyer))
*   Added a new --etcd-upgrade keyword to kubeadm upgrade apply. When this keyword is specified, etcd's static pod gets upgraded to the etcd version officially recommended for a target kubernetes release. ([#55010](https://github.com/kubernetes/kubernetes/pull/55010),[ @sbezverk](https://github.com/sbezverk))
*   Kubeadm now supports Kubelet Dynamic Configuration on an alpha level. ([#55803](https://github.com/kubernetes/kubernetes/pull/55803),[ @xiangpengzhao](https://github.com/xiangpengzhao))
*   Added support for adding a Windows node ([#53553](https://github.com/kubernetes/kubernetes/pull/53553),[ @bsteciuk](https://github.com/bsteciuk))

#### **Juju**

*   Added support for SAN entries in the master node certificate. ([#54234](https://github.com/kubernetes/kubernetes/pull/54234),[ @hyperbolic2346](https://github.com/hyperbolic2346))
*   Add extra-args configs for scheduler and controller-manager to kubernetes-master charm ([#55185](https://github.com/kubernetes/kubernetes/pull/55185),[ @Cynerva](https://github.com/Cynerva))
*   Add support for RBAC ([#53820](https://github.com/kubernetes/kubernetes/pull/53820),[ @ktsakalozos](https://github.com/ktsakalozos))
*   Fixed iptables FORWARD policy for Docker 1.13 in kubernetes-worker charm ([#54796](https://github.com/kubernetes/kubernetes/pull/54796),[ @Cynerva](https://github.com/Cynerva))
*   Upgrading the kubernetes-master units now results in staged upgrades just like the kubernetes-worker nodes. Use the upgrade action in order to continue the upgrade process on each unit such as juju run-action kubernetes-master/0 upgrade ([#55990](https://github.com/kubernetes/kubernetes/pull/55990),[ @hyperbolic2346](https://github.com/hyperbolic2346))
*   Added extra_sans config option to kubeapi-load-balancer charm. This allows the user to specify extra SAN entries on the certificate generated for the load balancer. ([#54947](https://github.com/kubernetes/kubernetes/pull/54947),[ @hyperbolic2346](https://github.com/hyperbolic2346))
*   Added extra-args configs to kubernetes-worker charm ([#55334](https://github.com/kubernetes/kubernetes/pull/55334),[ @Cynerva](https://github.com/Cynerva))

#### **Other**

*   Base images have been bumped to Debian Stretch (9) ([#52744](https://github.com/kubernetes/kubernetes/pull/52744),[ @rphillips](https://github.com/rphillips))
*   Upgraded to go1.9. ([#51375](https://github.com/kubernetes/kubernetes/pull/51375),[ @cblecker](https://github.com/cblecker))
*   Add-on manager now supports HA masters. ([#55466](https://github.com/kubernetes/kubernetes/pull/55466),[ #55782](https://github.com/x13n),[ @x13n](https://github.com/x13n))
*   Hyperkube can now run from a non-standard path. ([#54570](https://github.com/kubernetes/kubernetes/pull/54570))

#### **GCP**

*   The service account made available on your nodes is now configurable. ([#52868](https://github.com/kubernetes/kubernetes/pull/52868),[ @ihmccreery](https://github.com/ihmccreery))
*   GCE nodes with NVIDIA GPUs attached now expose nvidia.com/gpu as a resource instead of alpha.kubernetes.io/nvidia-gpu. ([#54826](https://github.com/kubernetes/kubernetes/pull/54826),[ @mindprince](https://github.com/mindprince))
*   Docker's live-restore on COS/ubuntu can now be disabled ([#55260](https://github.com/kubernetes/kubernetes/pull/55260),[ @yujuhong](https://github.com/yujuhong))
*   Metadata concealment is now controlled by the ENABLE_METADATA_CONCEALMENT env var. See cluster/gce/config-default.sh for more info. ([#54150](https://github.com/kubernetes/kubernetes/pull/54150),[ @ihmccreery](https://github.com/ihmccreery))
*   Masquerading rules are now added by default to GCE/GKE ([#55178](https://github.com/kubernetes/kubernetes/pull/55178),[ @dnardo](https://github.com/dnardo))
*   Fixed master startup issues with concurrent iptables invocations. ([#55945](https://github.com/kubernetes/kubernetes/pull/55945),[ @x13n](https://github.com/x13n))
*   Fixed issue deleting internal load balancers when the firewall resource may not exist. ([#53450](https://github.com/kubernetes/kubernetes/pull/53450),[ @nicksardo](https://github.com/nicksardo))

### **Instrumentation**

#### **Audit**

*   Adjust batching audit webhook default parameters: increase queue size, batch size, and initial backoff. Add throttling to the batching audit webhook. Default rate limit is 10 QPS. ([#53417](https://github.com/kubernetes/kubernetes/pull/53417),[ @crassirostris](https://github.com/crassirostris))
    *   These parameters are also now configurable. ([#56638](https://github.com/kubernetes/kubernetes/pull/56638), [@crassirostris](https://github.com/crassirostris))

#### **Other**

*   Fix a typo in prometheus-to-sd configuration, that drops some stackdriver metrics. ([#56473](https://github.com/kubernetes/kubernetes/pull/56473),[ @loburm](https://github.com/loburm))
*   [fluentd-elasticsearch addon] Elasticsearch and Kibana are updated to version 5.6.4 ([#55400](https://github.com/kubernetes/kubernetes/pull/55400),[ @mrahbar](https://github.com/mrahbar))
*   fluentd now supports CRI log format. ([#54777](https://github.com/kubernetes/kubernetes/pull/54777),[ @Random-Liu](https://github.com/Random-Liu))
*   Bring all prom-to-sd container to the same image version ([#54583](https://github.com/kubernetes/kubernetes/pull/54583))
    *   Reduce log noise produced by prometheus-to-sd, by bumping it to version 0.2.2. ([#54635](https://github.com/kubernetes/kubernetes/pull/54635),[ @loburm](https://github.com/loburm))
*   [fluentd-elasticsearch addon] Elasticsearch service name can be overridden via env variable ELASTICSEARCH_SERVICE_NAME ([#54215](https://github.com/kubernetes/kubernetes/pull/54215),[ @mrahbar](https://github.com/mrahbar))

### **Multicluster**

#### **Federation**

*   Kubefed init now supports --imagePullSecrets and --imagePullPolicy, making it possible to use private registries. ([#50740](https://github.com/kubernetes/kubernetes/pull/50740),[ @dixudx](https://github.com/dixudx))
*   Updated cluster printer to enable --show-labels ([#53771](https://github.com/kubernetes/kubernetes/pull/53771),[ @dixudx](https://github.com/dixudx))
*   Kubefed init now supports --nodeSelector, enabling you to determine on what node the controller will be installed. ([#50749](https://github.com/kubernetes/kubernetes/pull/50749),[ @dixudx](https://github.com/dixudx))

### **Network**

#### **IPv6**

*   [alpha] IPv6 support has been added. Notable IPv6 support details include:
    *   Support for IPv6-only Kubernetes cluster deployments. **<span style="text-decoration:underline;">Note:</span>** This feature does not provide dual-stack support.
    *   Support for IPv6 Kubernetes control and data planes.
    *   Support for Kubernetes IPv6 cluster deployments using kubeadm.
    *   Support for the iptables kube-proxy backend using ip6tables.
    *   Relies on CNI 0.6.0 binaries for IPv6 pod networking.
    *   Adds IPv6 support for kube-dns using SRV records.
    *   Caveats
        *   Only the CNI bridge and local-ipam plugins have been tested for the alpha release, although other CNI plugins do support IPv6.
        *   HostPorts are not supported.
*   An IPv6 network mask for pod or cluster cidr network must be /66 or longer. For example: 2001:db1::/66, 2001:dead:beef::/76, 2001:cafe::/118 are supported. 2001:db1::/64 is not supported
*   For details, see [the complete list of merged pull requests for IPv6 support](https://github.com/kubernetes/kubernetes/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Amerged+label%3Aarea%2Fipv6).

#### **IPVS**

*   You can now use the --cleanup-ipvs flag to tell kube-proxy whether to flush all existing ipvs rules in on startup ([#56036](https://github.com/kubernetes/kubernetes/pull/56036),[ @m1093782566](https://github.com/m1093782566))
*   Graduate kube-proxy IPVS mode to beta. ([#56623](https://github.com/kubernetes/kubernetes/pull/56623), [@m1093782566](https://github.com/m1093782566))

#### **Kube-Proxy**

*   Added iptables rules to allow Pod traffic even when default iptables policy is to reject. ([#52569](https://github.com/kubernetes/kubernetes/pull/52569),[ @tmjd](https://github.com/tmjd))
*   You can once again use 0 values for conntrack min, max, max per core, tcp close wait timeout, and tcp established timeout; this functionality was broken in 1.8. ([#55261](https://github.com/kubernetes/kubernetes/pull/55261),[ @ncdc](https://github.com/ncdc))

#### **CoreDNS**

*   You now have the option to use CoreDNS instead of KubeDNS. To install CoreDNS instead of kube-dns, set CLUSTER_DNS_CORE_DNS to 'true'. This support is experimental. ([#52501](https://github.com/kubernetes/kubernetes/pull/52501),[ @rajansandeep](https://github.com/rajansandeep)) ([#55728](https://github.com/kubernetes/kubernetes/pull/55728),[ @rajansandeep](https://github.com/rajansandeep))

#### **Other**

*   Pod addresses will now be removed from the list of endpoints when the pod is in graceful termination. ([#54828](https://github.com/kubernetes/kubernetes/pull/54828),[ @freehan](https://github.com/freehan))
*   You can now use a new supported service annotation for AWS clusters, `service.beta.kubernetes.io/aws-load-balancer-ssl-negotiation-policy`, which lets you specify which [predefined AWS SSL policy](http://docs.aws.amazon.com/elasticloadbalancing/latest/classic/elb-security-policy-table.html) you would like to use. ([#54507](https://github.com/kubernetes/kubernetes/pull/54507),[ @micahhausler](https://github.com/micahhausler))
*   Termination grace period for the calico/node add-on DaemonSet has been eliminated, reducing downtime during a rolling upgrade or deletion. ([#55015](https://github.com/kubernetes/kubernetes/pull/55015),[ @fasaxc](https://github.com/fasaxc))
*   Fixed bad conversion in host port chain name generating func which led to some unreachable host ports. ([#55153](https://github.com/kubernetes/kubernetes/pull/55153),[ @chenchun](https://github.com/chenchun))
*   Fixed IPVS availability check ([#51874](https://github.com/kubernetes/kubernetes/pull/51874),[ @vfreex](https://github.com/vfreex))
*   The output for kubectl describe networkpolicy * has been enhanced to be more useful. ([#46951](https://github.com/kubernetes/kubernetes/pull/46951),[ @aanm](https://github.com/aanm))
*   Kernel modules are now loaded automatically inside a kube-proxy pod ([#52003](https://github.com/kubernetes/kubernetes/pull/52003),[ @vfreex](https://github.com/vfreex))
*   Improve resilience by annotating kube-dns addon with podAntiAffinity to prefer scheduling on different nodes. ([#52193](https://github.com/kubernetes/kubernetes/pull/52193),[ @StevenACoffman](https://github.com/StevenACoffman))
*   [alpha] Added DNSConfig field to PodSpec. "None" mode for DNSPolicy is now supported. ([#55848](https://github.com/kubernetes/kubernetes/pull/55848),[ @MrHohn](https://github.com/MrHohn))
*   You can now add "options" to the host's /etc/resolv.conf (or --resolv-conf), and they will be copied into pod's resolv.conf when dnsPolicy is Default. Being able to customize options is important because it is common to leverage options to fine-tune the behavior of DNS client. ([#54773](https://github.com/kubernetes/kubernetes/pull/54773),[ @phsiao](https://github.com/phsiao))
*   Fixed a bug so that the service controller no longer retries if doNotRetry service update fails. ([#54184](https://github.com/kubernetes/kubernetes/pull/54184),[ @MrHohn](https://github.com/MrHohn))
*   Added --no-negcache flag to kube-dns to prevent caching of NXDOMAIN responses. ([#53604](https://github.com/kubernetes/kubernetes/pull/53604),[ @cblecker](https://github.com/cblecker))

### **Node**

#### **Pod API**

*   A single value in metadata.annotations/metadata.labels can now be passed into the containers via the Downward API. ([#55902](https://github.com/kubernetes/kubernetes/pull/55902),[ @yguo0905](https://github.com/yguo0905))
*   Pods will no longer briefly transition to a "Pending" state during the deletion process. ([#54593](https://github.com/kubernetes/kubernetes/pull/54593),[ @dashpole](https://github.com/dashpole))
*   Added pod-level local ephemeral storage metric to the Summary API. Pod-level ephemeral storage reports the total filesystem usage for the containers and emptyDir volumes in the measured Pod. ([#55447](https://github.com/kubernetes/kubernetes/pull/55447),[ @jingxu97](https://github.com/jingxu97))

#### **Hardware Accelerators**

*   Kubelet now exposes metrics for NVIDIA GPUs attached to the containers. ([#55188](https://github.com/kubernetes/kubernetes/pull/55188),[ @mindprince](https://github.com/mindprince))
*   The device plugin Alpha API no longer supports returning artifacts per device as part of AllocateResponse. ([#53031](https://github.com/kubernetes/kubernetes/pull/53031),[ @vishh](https://github.com/vishh))
*   Fix to ignore extended resources that are not registered with kubelet during container resource allocation. ([#53547](https://github.com/kubernetes/kubernetes/pull/53547),[ @jiayingz](https://github.com/jiayingz))


#### **Container Runtime**
*   [alpha] [cri-tools](https://github.com/kubernetes-incubator/cri-tools): CLI and validation tools for CRI is now v1.0.0-alpha.0. This release mainly focuses on UX improvements. [[@feiskyer](https://github.com/feiskyer)]
    *   Make crictl command more user friendly and add more subcommands.
    *   Integrate with CRI verbose option to provide extra debug information.
    *   Update CRI to kubernetes v1.9.
    *   Bug fixes in validation test suites.
*   [beta] [cri-containerd](https://github.com/kubernetes-incubator/cri-containerd): CRI implementation for containerd is now v1.0.0-beta.0, [[@Random-Liu](https://github.com/Random-Liu)]
    *   This release supports Kubernetes 1.9+ and containerd v1.0.0+.
    *   Pass all Kubernetes 1.9 e2e test, node e2e test and CRI validation tests.
    *   [Kube-up.sh integration](https://github.com/kubernetes-incubator/cri-containerd/blob/master/docs/kube-up.md).
    *   [Full crictl integration including CRI verbose option.](https://github.com/kubernetes-incubator/cri-containerd/blob/master/docs/crictl.md)
    *   Integration with cadvisor to provide better summary api support.
*   [stable] [cri-o](https://github.com/kubernetes-incubator/cri-o): CRI implementation for OCI-based runtimes is now v1.9. [[@mrunalp](https://github.com/mrunalp)]
    *   Pass all the Kubernetes 1.9 end-to-end test suites and now gating PRs as well
    *   Pass all the CRI validation tests
    *   Release has been focused on bug fixes, stability and performance with runc and Clear Containers
    *   Minikube integration
*   [stable] [frakti](https://github.com/kubernetes/frakti): CRI implementation for hypervisor-based runtimes is now v1.9. [[@resouer](https://github.com/resouer)]
    *   Added ARM64 release. Upgraded to CNI 0.6.0, added block device as Pod volume mode. Fixed CNI plugin compatibility.
    *   Passed all CRI validation conformance tests and node end-to-end conformance tests.
*   [alpha] [rktlet](https://github.com/kubernetes-incubator/rktlet): CRI implementation for the rkt runtime is now v0.1.0. [[@iaguis](https://github.com/iaguis)]
    *   This is the first release of rktlet and it implements support for the CRI including fetching images, running pods, CNI networking, logging and exec.
This release passes 129/145 Kubernetes e2e conformance tests.
*   Container Runtime Interface API change. [[@yujuhong](https://github.com/yujuhong)]
    *   A new field is added to CRI container log format to support splitting a long log line into multiple lines. ([#55922](https://github.com/kubernetes/kubernetes/pull/55922), [@Random-Liu](https://github.com/Random-Liu))
    *   CRI now supports debugging via a verbose option for status functions. ([#53965](https://github.com/kubernetes/kubernetes/pull/53965), [@Random-Liu](https://github.com/Random-Liu))
    *   Kubelet can now provide full summary api support for the CRI container runtime, with the exception of container log stats. ([#55810](https://github.com/kubernetes/kubernetes/pull/55810), [@abhi](https://github.com/abhi))
    *   CRI now uses the correct localhost seccomp path when provided with input in the format of localhost//profileRoot/profileName. ([#55450](https://github.com/kubernetes/kubernetes/pull/55450), [@feiskyer](https://github.com/feiskyer))


#### **Kubelet**

*   The EvictionHard, EvictionSoft, EvictionSoftGracePeriod, EvictionMinimumReclaim, SystemReserved, and KubeReserved fields in the KubeletConfiguration object (`kubeletconfig/v1alpha1`) are now of type map[string]string, which facilitates writing JSON and YAML files. ([#54823](https://github.com/kubernetes/kubernetes/pull/54823),[ @mtaufen](https://github.com/mtaufen))
*   Relative paths in the Kubelet's local config files (`--init-config-dir`) will now be resolved relative to the location of the containing files. ([#55648](https://github.com/kubernetes/kubernetes/pull/55648),[ @mtaufen](https://github.com/mtaufen))
*   It is now possible to set multiple manifest URL headers with the kubelet's `--manifest-url-header` flag. Multiple headers for the same key will be added in the order provided. The ManifestURLHeader field in KubeletConfiguration object (kubeletconfig/v1alpha1) is now a map[string][]string, which facilitates writing JSON and YAML files. ([#54643](https://github.com/kubernetes/kubernetes/pull/54643),[ @mtaufen](https://github.com/mtaufen))
*   The Kubelet's feature gates are now specified as a map when provided via a JSON or YAML KubeletConfiguration, rather than as a string of key-value pairs, making them less awkward for users. ([#53025](https://github.com/kubernetes/kubernetes/pull/53025),[ @mtaufen](https://github.com/mtaufen))

##### **Other**

*   Fixed a performance issue ([#51899](https://github.com/kubernetes/kubernetes/pull/51899)) identified in large-scale clusters when deleting thousands of pods simultaneously across hundreds of nodes, by actively removing containers of deleted pods, rather than waiting for periodic garbage collection and batching resulting pod API deletion requests. ([#53233](https://github.com/kubernetes/kubernetes/pull/53233),[ @dashpole](https://github.com/dashpole))
*   Problems deleting local static pods have been resolved. ([#48339](https://github.com/kubernetes/kubernetes/pull/48339),[ @dixudx](https://github.com/dixudx))
*   CRI now only calls UpdateContainerResources when cpuset is set. ([#53122](https://github.com/kubernetes/kubernetes/pull/53122),[ @resouer](https://github.com/resouer))
*   Containerd monitoring is now supported. ([#56109](https://github.com/kubernetes/kubernetes/pull/56109),[ @dashpole](https://github.com/dashpole))
*   deviceplugin has been extended to more gracefully handle the full device plugin lifecycle, including: ([#55088](https://github.com/kubernetes/kubernetes/pull/55088),[ @jiayingz](https://github.com/jiayingz))
    *   Kubelet now uses an explicit cm.GetDevicePluginResourceCapacity() function that makes it possible to more accurately determine what resources are inactive and return a more accurate view of available resources.
    *   Extends the device plugin checkpoint data to record registered resources so that we can finish resource removing devices even upon kubelet restarts.
    *   Passes sourcesReady from kubelet to the device plugin to avoid removing inactive pods during the grace period of kubelet restart.
    *   Extends the gpu_device_plugin e2e_node test to verify that scheduled pods can continue to run even after a device plugin deletion and kubelet restart.
*   The NodeController no longer supports kubelet 1.2. ([#48996](https://github.com/kubernetes/kubernetes/pull/48996),[ @k82cn](https://github.com/k82cn))
*   Kubelet now provides more specific events via FailedSync when unable to sync a pod. ([#53857](https://github.com/kubernetes/kubernetes/pull/53857),[ @derekwaynecarr](https://github.com/derekwaynecarr))
*   You can now disable AppArmor by setting the AppArmor profile to unconfined. ([#52395](https://github.com/kubernetes/kubernetes/pull/52395),[ @dixudx](https://github.com/dixudx))
*   ImageGCManage now consumes ImageFS stats from StatsProvider rather than cadvisor. ([#53094](https://github.com/kubernetes/kubernetes/pull/53094),[ @yguo0905](https://github.com/yguo0905))
*   Hyperkube now supports the support --experimental-dockershim kubelet flag. ([#54508](https://github.com/kubernetes/kubernetes/pull/54508),[ @ivan4th](https://github.com/ivan4th))
*   Kubelet no longer removes default labels from Node API objects on startup ([#54073](https://github.com/kubernetes/kubernetes/pull/54073),[ @liggitt](https://github.com/liggitt))
*   The overlay2 container disk metrics for Docker and CRI-O now work properly. ([#54827](https://github.com/kubernetes/kubernetes/pull/54827),[ @dashpole](https://github.com/dashpole))
*   Removed docker dependency during kubelet start up. ([#54405](https://github.com/kubernetes/kubernetes/pull/54405),[ @resouer](https://github.com/resouer))
*   Added Windows support to the system verification check. ([#53730](https://github.com/kubernetes/kubernetes/pull/53730),[ @bsteciuk](https://github.com/bsteciuk))
*   Kubelet no longer removes unregistered extended resource capacities from node status; cluster admins will have to manually remove extended resources exposed via device plugins when they the remove plugins themselves. ([#53353](https://github.com/kubernetes/kubernetes/pull/53353),[ @jiayingz](https://github.com/jiayingz))
*   The stats summary network value now takes into account multiple network interfaces, and not just eth0. ([#52144](https://github.com/kubernetes/kubernetes/pull/52144),[ @andyxning](https://github.com/andyxning))
*   Base images have been bumped to Debian Stretch (9). ([#52744](https://github.com/kubernetes/kubernetes/pull/52744),[ @rphillips](https://github.com/rphillips))

### **OpenStack**

*   OpenStack Cinder support has been improved:
    *   Cinder version detection now works properly. ([#53115](https://github.com/kubernetes/kubernetes/pull/53115),[ @FengyunPan](https://github.com/FengyunPan))
    *   The OpenStack cloud provider now supports Cinder v3 API. ([#52910](https://github.com/kubernetes/kubernetes/pull/52910),[ @FengyunPan](https://github.com/FengyunPan))
*   Load balancing is now more flexible:
    *   The OpenStack LBaaS v2 Provider is now [configurable](https://kubernetes.io/docs/concepts/cluster-administration/cloud-providers/#openstack). ([#54176](https://github.com/kubernetes/kubernetes/pull/54176),[ @gonzolino](https://github.com/gonzolino))
    *   OpenStack Octavia v2 is now supported as a load balancer provider in addition to the existing support for the Neutron LBaaS V2 implementation. Neutron LBaaS V1 support has been removed. ([#55393](https://github.com/kubernetes/kubernetes/pull/55393),[ @jamiehannaford](https://github.com/jamiehannaford))
*   OpenStack security group support has been beefed up  ([#50836](https://github.com/kubernetes/kubernetes/pull/50836),[ @FengyunPan](https://github.com/FengyunPan)): 
    *   Kubernetes will now automatically determine the security group for the node
    *   Nodes can now belong to multiple security groups

### **Scheduling**

#### **Hardware Accelerators**

*   Add ExtendedResourceToleration admission controller. This facilitates creation of dedicated nodes with extended resources. If operators want to create dedicated nodes with extended resources (such as GPUs, FPGAs, and so on), they are expected to taint the node with extended resource name as the key. This admission controller, if enabled, automatically adds tolerations for such taints to pods requesting extended resources, so users don't have to manually add these tolerations. ([#55839](https://github.com/kubernetes/kubernetes/pull/55839),[ @mindprince](https://github.com/mindprince))

#### **Other**

*   Scheduler cache ignores updates to an assumed pod if updates are limited to pod annotations.  ([#54008](https://github.com/kubernetes/kubernetes/pull/54008),[ @yguo0905](https://github.com/yguo0905))
*   Issues with namespace deletion have been resolved. ([#53720](https://github.com/kubernetes/kubernetes/pull/53720),[ @shyamjvs](https://github.com/shyamjvs)) ([#53793](https://github.com/kubernetes/kubernetes/pull/53793),[ @wojtek-t](https://github.com/wojtek-t))
*   Pod preemption has been improved.  
    *   Now takes PodDisruptionBudget into account. ([#56178](https://github.com/kubernetes/kubernetes/pull/56178),[ @bsalamat](https://github.com/bsalamat))
    *   Nominated pods are taken into account during scheduling to avoid starvation of higher priority pods. ([#55933](https://github.com/kubernetes/kubernetes/pull/55933),[ @bsalamat](https://github.com/bsalamat))
*   Fixed 'Schedulercache is corrupted' error in kube-scheduler ([#55262](https://github.com/kubernetes/kubernetes/pull/55262),[ @liggitt](https://github.com/liggitt))
*   The kube-scheduler command now supports a --config flag which is the location of a file containing a serialized scheduler configuration. Most other kube-scheduler flags are now deprecated. ([#52562](https://github.com/kubernetes/kubernetes/pull/52562),[ @ironcladlou](https://github.com/ironcladlou))
*   A new scheduling queue helps schedule the highest priority pending pod first. ([#55109](https://github.com/kubernetes/kubernetes/pull/55109),[ @bsalamat](https://github.com/bsalamat))
*   A Pod can now listen to the same port on multiple IP addresses.  ([#52421](https://github.com/kubernetes/kubernetes/pull/52421),[ @WIZARD-CXY](https://github.com/WIZARD-CXY))
*   Object count quotas supported on all standard resources using count/<resource>.<group> syntax ([#54320](https://github.com/kubernetes/kubernetes/pull/54320),[ @derekwaynecarr](https://github.com/derekwaynecarr))
*   Apply algorithm in scheduler by feature gates. ([#52723](https://github.com/kubernetes/kubernetes/pull/52723),[ @k82cn](https://github.com/k82cn))
*   A new priority function ResourceLimitsPriorityMap (disabled by default and behind alpha feature gate and not part of the scheduler's default priority functions list) that assigns a lowest possible score of 1 to a node that satisfies one or both of input pod's cpu and memory limits, mainly to break ties between nodes with same scores. ([#55906](https://github.com/kubernetes/kubernetes/pull/55906),[ @aveshagarwal](https://github.com/aveshagarwal))
*   Kubelet evictions now take pod priority into account ([#53542](https://github.com/kubernetes/kubernetes/pull/53542),[ @dashpole](https://github.com/dashpole))
*   PodTolerationRestriction admisson plugin: if namespace level tolerations are empty, now they override cluster level tolerations. ([#54812](https://github.com/kubernetes/kubernetes/pull/54812),[ @aveshagarwal](https://github.com/aveshagarwal))

### **Storage**

*   [stable] `PersistentVolume` and `PersistentVolumeClaim` objects must now have a capacity greater than zero.
*   [stable] Mutation of `PersistentVolumeSource` after creation is no longer allowed
*   [alpha] Deletion of `PersistentVolumeClaim` objects that are in use by a pod no longer permitted (if alpha feature is enabled).
*   [alpha] Container Storage Interface
    *   New CSIVolumeSource enables Kubernetes to use external CSI drivers to provision, attach, and mount volumes.
*   [alpha] Raw block volumes 
    *   Support for surfacing volumes as raw block devices added to Kubernetes storage system.
    *   Only Fibre Channel volume plugin supports exposes this functionality, in this release.
*   [alpha] Volume resizing
    *   Added file system resizing for the following volume plugins: GCE PD, Ceph RBD, AWS EBS, OpenStack Cinder
*   [alpha] Topology Aware Volume Scheduling 
    *   Improved volume scheduling for Local PersistentVolumes, by allowing the scheduler to make PersistentVolume binding decisions while respecting the Pod's scheduling requirements.
    *   Dynamic provisioning is not supported with this feature yet.
*   [alpha] Containerized mount utilities
    *   Allow mount utilities, used to mount volumes, to run inside a container instead of on the host.
*   Bug Fixes
    *   ScaleIO volume plugin is no longer dependent on the drv_cfg binary, so a Kubernetes cluster can easily run a containerized kubelet. ([#54956](https://github.com/kubernetes/kubernetes/pull/54956),[ @vladimirvivien](https://github.com/vladimirvivien))
    *   AWS EBS Volumes are detached from stopped AWS nodes. ([#55893](https://github.com/kubernetes/kubernetes/pull/55893),[ @gnufied](https://github.com/gnufied))
    *   AWS EBS volumes are detached if attached to a different node than expected. ([#55491](https://github.com/kubernetes/kubernetes/pull/55491),[ @gnufied](https://github.com/gnufied))
    *   PV Recycle now works in environments that use architectures other than x86. ([#53958](https://github.com/kubernetes/kubernetes/pull/53958),[ @dixudx](https://github.com/dixudx))
    *   Pod Security Policy can now manage access to specific FlexVolume drivers.([#53179](https://github.com/kubernetes/kubernetes/pull/53179),[ @wanghaoran1988](https://github.com/wanghaoran1988))
    *   To prevent unauthorized access to CHAP Secrets, you can now set the secretNamespace storage class parameters for the following volume types:
        *   ScaleIO; StoragePool and ProtectionDomain attributes no longer default to the value default.  ([#54013](https://github.com/kubernetes/kubernetes/pull/54013),[ @vladimirvivien](https://github.com/vladimirvivien))
        *   RBD Persistent Volume Sources ([#54302](https://github.com/kubernetes/kubernetes/pull/54302),[ @sbezverk](https://github.com/sbezverk))
        *   iSCSI Persistent Volume Sources ([#51530](https://github.com/kubernetes/kubernetes/pull/51530),[ @rootfs](https://github.com/rootfs))
    *   In GCE multizonal clusters, `PersistentVolume` objects will no longer be dynamically provisioned in zones without nodes. ([#52322](https://github.com/kubernetes/kubernetes/pull/52322),[ @davidz627](https://github.com/davidz627))
    *   Multi Attach PVC errors and events are now more useful and less noisy. ([#53401](https://github.com/kubernetes/kubernetes/pull/53401),[ @gnufied](https://github.com/gnufied))
    *   The compute-rw scope has been removed from GCE nodes ([#53266](https://github.com/kubernetes/kubernetes/pull/53266),[ @mikedanese](https://github.com/mikedanese))
    *   Updated vSphere cloud provider to support k8s cluster spread across multiple vCenters ([#55845](https://github.com/kubernetes/kubernetes/pull/55845),[ @rohitjogvmw](https://github.com/rohitjogvmw))
    *   vSphere: Fix disk is not getting detached when PV is provisioned on clustered datastore. ([#54438](https://github.com/kubernetes/kubernetes/pull/54438),[ @pshahzeb](https://github.com/pshahzeb))
    *   If a non-absolute mountPath is passed to the kubelet, it must now be prefixed with the appropriate root path. ([#55665](https://github.com/kubernetes/kubernetes/pull/55665),[ @brendandburns](https://github.com/brendandburns))

## External Dependencies

*   The supported etcd server version is **3.1.10**, as compared to 3.0.17 in v1.8 ([#49393](https://github.com/kubernetes/kubernetes/pull/49393),[ @hongchaodeng](https://github.com/hongchaodeng))
*   The validated docker versions are the same as for v1.8: **1.11.2 to 1.13.1 and 17.03.x**
*   The Go version was upgraded from go1.8.3 to **go1.9.2** ([#51375](https://github.com/kubernetes/kubernetes/pull/51375),[ @cblecker](https://github.com/cblecker))
    *   The minimum supported go version bumps to 1.9.1. ([#55301](https://github.com/kubernetes/kubernetes/pull/55301),[ @xiangpengzhao](https://github.com/xiangpengzhao))
    *   Kubernetes has been upgraded to go1.9.2 ([#55420](https://github.com/kubernetes/kubernetes/pull/55420),[ @cblecker](https://github.com/cblecker))
*   CNI was upgraded to **v0.6.0** ([#51250](https://github.com/kubernetes/kubernetes/pull/51250),[ @dixudx](https://github.com/dixudx))
*   The dashboard add-on has been updated to [v1.8.0](https://github.com/kubernetes/dashboard/releases/tag/v1.8.0). ([#53046](https://github.com/kubernetes/kubernetes/pull/53046), [@maciaszczykm](https://github.com/maciaszczykm))
*   Heapster has been updated to [v1.5.0](https://github.com/kubernetes/heapster/releases/tag/v1.5.0). ([#57046](https://github.com/kubernetes/kubernetes/pull/57046), [@piosz](https://github.com/piosz))
*   Cluster Autoscaler has been updated to [v1.1.0](https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.1.0). ([#56969](https://github.com/kubernetes/kubernetes/pull/56969), [@mwielgus](https://github.com/mwielgus))
*   Update kube-dns 1.14.7 ([#54443](https://github.com/kubernetes/kubernetes/pull/54443),[ @bowei](https://github.com/bowei))
*   Update influxdb to v1.3.3 and grafana to v4.4.3 ([#53319](https://github.com/kubernetes/kubernetes/pull/53319),[ @kairen](https://github.com/kairen))
- [v1.9.0-beta.2](#v190-beta2)
- [v1.9.0-beta.1](#v190-beta1)
- [v1.9.0-alpha.3](#v190-alpha3)
- [v1.9.0-alpha.2](#v190-alpha2)
- [v1.9.0-alpha.1](#v190-alpha1)



# v1.9.0-beta.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.0-beta.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes.tar.gz) | `e5c88addf6aca01635f283021a72e05be99daf3e87fd3cda92477d0ed63c2d11`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-src.tar.gz) | `2419a0ef3681460b64eefc083d07377786b308f6cc62d0618a5c74dfb4729b03`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `68d971576c3e9a16fb736f06c07ce53b8371fc67c2f37fb60e9f3a366cd37a80`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `36251b7b6043adb79706ac115181aa7ecf365ced9198a4c192f1fbc2817d030c`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-linux-386.tar.gz) | `585a3dd6a3440988bce3f83ea14fb9a0a18011bc62e28959301861faa06d6da9`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `169769d6030d8c1d9d9bc01408b62ea3275d4632a7de85392fc95a48feeba522`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `7841c2af49be9ae04cda305165b172021c0e72d809c2271d05061330c220256b`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `9ab32843cec68b036de83f54a68c2273a913be5180dc20b5cf1e084b314a9a2d`
[kubernetes-client-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | `5a2bb39b78ef381382f9b8aac17d5dbcbef08a80ad3518ff2cf6c65bd7a6d07d`
[kubernetes-client-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | `ddf4b3780f5879b9fb9115353cc26234cfc3a6db63a3cd39122340189a4bf0ca`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-windows-386.tar.gz) | `5960a0a50c92a788e90eca9d85a1d12ff1d41264816b55b3a1a28ffd3f6acf93`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `d85778ace9bf25f5d3626aef3a9419a2c4aaa3847d5e0c2bf34d4dd8ae6b5205`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `43e16b3d79c2805d712fd61ed6fd110d9db09a60d39584ef78c24821eb32b77a`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `8580e454e6c467a30687ff5c85248919b3c0d2d0114e28cb3bf64d2e8998ff00`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `d2e767be85ebf7c6c537c8e796e8fe0ce8a3f2ca526984490646acd30bf5e6fc`
[kubernetes-server-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | `81dd9072e805c181b4db2dfd00fe2bdb43c00da9e07b50285bce703bfd0d75ba`
[kubernetes-server-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | `f432c816c755d05e62cb5d5e8ac08dcb60d0df6d5121e1adaf42a32de65d6174`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-node-linux-amd64.tar.gz) | `2bf2268735ca4ecbdca1a692b25329d6d9d4805963cbe0cfcbb92fc725c42481`
[kubernetes-node-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-node-linux-arm64.tar.gz) | `3bb4a695fd2e4fca1c77283c1ad6c2914d12b33d9c5f64ac9c630a42d5e30ab2`
[kubernetes-node-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-node-linux-arm.tar.gz) | `331c1efadf99dcb634c8da301349e3be63d27a5c5f06cc124b59fcc8b8a91cb0`
[kubernetes-node-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-node-linux-ppc64le.tar.gz) | `ab036fdb64ed4702d7dbbadddf77af90de35f73aa13854bb5accf82acc95c7e6`
[kubernetes-node-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-node-linux-s390x.tar.gz) | `8257af566f98325549de320d2167c1f56fd137b6225c70f6c1e34507ba124a1f`
[kubernetes-node-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-beta.2/kubernetes-node-windows-amd64.tar.gz) | `4146fcb5bb6bf3e04641b27e4aa8501649178716fa16bd9bcb7f1fe3449db7f2`

## Changelog since v1.9.0-beta.1

### Other notable changes

* Add pvc as part of equivalence hash ([#56577](https://github.com/kubernetes/kubernetes/pull/56577), [@resouer](https://github.com/resouer))
* Fix port number and default Stackdriver Metadata Agent in daemon set configuration. ([#56576](https://github.com/kubernetes/kubernetes/pull/56576), [@kawych](https://github.com/kawych))
* Declare ipvs proxier beta ([#56623](https://github.com/kubernetes/kubernetes/pull/56623), [@m1093782566](https://github.com/m1093782566))
* Enable admissionregistration.k8s.io/v1beta1 by default in kube-apiserver. ([#56687](https://github.com/kubernetes/kubernetes/pull/56687), [@sttts](https://github.com/sttts))
* Support autoprobing floating-network-id for openstack cloud provider ([#52013](https://github.com/kubernetes/kubernetes/pull/52013), [@FengyunPan](https://github.com/FengyunPan))
* Audit webhook batching parameters are now configurable via command-line flags in the apiserver. ([#56638](https://github.com/kubernetes/kubernetes/pull/56638), [@crassirostris](https://github.com/crassirostris))
* Update kubectl to the stable version ([#54345](https://github.com/kubernetes/kubernetes/pull/54345), [@zouyee](https://github.com/zouyee))
* [scheduler] Fix issue new pod with affinity stuck at `creating` because node had been deleted but its pod still exists. ([#53647](https://github.com/kubernetes/kubernetes/pull/53647), [@wenlxie](https://github.com/wenlxie))
* Updated Dashboard add-on to version 1.8.0: The Dashboard add-on now deploys with https enabled. The Dashboard can be accessed via kubectl proxy at http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/. The /ui redirect is deprecated and will be removed in 1.10. ([#53046](https://github.com/kubernetes/kubernetes/pull/53046), [@maciaszczykm](https://github.com/maciaszczykm))
* AWS: Detect EBS volumes mounted via NVME and mount them ([#56607](https://github.com/kubernetes/kubernetes/pull/56607), [@justinsb](https://github.com/justinsb))
* fix CreateVolume func: use search mode instead ([#54687](https://github.com/kubernetes/kubernetes/pull/54687), [@andyzhangx](https://github.com/andyzhangx))
* kubelet: fix bug where `runAsUser: MustRunAsNonRoot` strategy didn't reject a pod with a non-numeric `USER`. ([#56503](https://github.com/kubernetes/kubernetes/pull/56503), [@php-coder](https://github.com/php-coder))
* kube-proxy addon tolerates all NoExecute and NoSchedule taints by default. ([#56589](https://github.com/kubernetes/kubernetes/pull/56589), [@mindprince](https://github.com/mindprince))
* Do not do file system resize on read-only mounts ([#56587](https://github.com/kubernetes/kubernetes/pull/56587), [@gnufied](https://github.com/gnufied))
* Mark v1beta1 NetworkPolicy types as deprecated ([#56425](https://github.com/kubernetes/kubernetes/pull/56425), [@cmluciano](https://github.com/cmluciano))
* Fix problem with /bin/bash ending up linked to dash  ([#55018](https://github.com/kubernetes/kubernetes/pull/55018), [@dims](https://github.com/dims))
* Modifying etcd recovery steps for the case of failed upgrade ([#56500](https://github.com/kubernetes/kubernetes/pull/56500), [@sbezverk](https://github.com/sbezverk))



# v1.9.0-beta.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.9/examples)

## Downloads for v1.9.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes.tar.gz) | `ffdcf0f7cd972340bc666395d759fc18573a32775d38ed3f4fd99d4369e856e4`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-src.tar.gz) | `09bee9a955987d53c7a65d2f1a3129854ca3a34f9fb38218f0c58f5bd603494a`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `9d54db976ca7a12e9208e5595b552b094e0cc532b49ba6e919d776e52e56f4a8`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `0a22af2c6c84ff8b3022c0ecebf4ba3021048fceddf7375c87c13a83488ffe2c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-linux-386.tar.gz) | `84bb638c8e61d7a7b415d49d76d166f3924052338c454d1ae57ae36eb37445c6`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `08b56240288d17f147485e79c5f6594391c5b46e26450d64e7510f65db1f9a79`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `7206573b131a8915d3bc14aa660fb44890ed79fdbd498bc8f9951c221aa12ea5`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `7ad21796b0e0a9d247beb41d6b3a3d0aaa822b85adae4c90533ba0ef94c05b2e`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `2076328ca0958a96c8f551b91a393aa2d6fc24bef92991a1a4d9fc8df52519a7`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `17ac0aba9a4e2003cb3d06bd631032b760d1a2d521c60a25dc26687aadb5ba14`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-windows-386.tar.gz) | `3a2bebd4adb6e1bf2b30a8cedb7ec212fc43c4b02e26a0a60c3429e478a86073`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `fcc852e97f0e64d1025344aefd042ceff05227bfded80142bfa99927de1a5f0e`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `7ed2a789b86f258f1739cb165276150512a171a715da9372aeff000e946548fd`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `e4e04a33698ac665a3e61fd8d60d4010fec6b0e3b0627dee9a965c2c2a510e3a`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `befce41457fc15c8fadf37ee5bf80b83405279c60665cfb9ecfc9f61fcd549c7`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `e59e4fb84d6b890e9c6cb216ebb20546212e6c14feb077d9d0761c88e2685f4c`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `0aa47d01907ea78b9a1a8001536d5091fca93409b81bac6eb3e90a4dff6c3faa`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `107bfaf72b8b6d3b5c163e61ed169c89288958750636c16bc3d781cf94bf5f4c`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `6bc58e913a2467548664ece743617a1e595f6223100a1bad27e9a90bdf2e2927`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `d4ff8f37d7c95f7ca3aca30fa3c191f2cc5e48f0159ac6a5395ec09092574baa`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `a88d65343ccb515c4eaab11352e69afee4a19c7fa345b08aaffa854b225cf305`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `16d6a67d18273460cab4c293a5b130d4827f41ee4bf5b79b07c60ef517f580cd`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `f086659462b6dcdd78abdf13bed339dd67c1111931bae962044aa4ae2396921d`

## Changelog since v1.9.0-alpha.3

### Action Required

* Adds alpha support for volume scheduling, which allows the scheduler to make PersistentVolume binding decisions while respecting the Pod's scheduling requirements.  Dynamic provisioning is not supported with this feature yet. ([#55039](https://github.com/kubernetes/kubernetes/pull/55039), [@msau42](https://github.com/msau42))
    * Action required for existing users of the LocalPersistentVolumes alpha feature:
        * The VolumeScheduling feature gate also has to be enabled on kube-scheduler and kube-controller-manager.
        * The NoVolumeNodeConflict predicate has been removed.  For non-default schedulers, update your scheduler policy.
        * The CheckVolumeBinding predicate has to be enabled in non-default schedulers.
* Action required: ([#56004](https://github.com/kubernetes/kubernetes/pull/56004), [@caesarxuchao](https://github.com/caesarxuchao))
    * The `admission/v1alpha1` API has graduated to `v1beta1`. Please delete your existing webhooks before upgrading the cluster, and update your admission webhooks to use the latest API, because the API has backwards incompatible changes.
    * The webhook registration related part of the `admissionregistration` API has graduated to `v1beta1`. Please delete your existing configurations before upgrading the cluster, and update your configuration file to use the latest API.
* [action required] kubeadm join: Error out if CA pinning isn't used or opted out of ([#55468](https://github.com/kubernetes/kubernetes/pull/55468), [@yuexiao-wang](https://github.com/yuexiao-wang))
        * kubeadm now requires the user to specify either the `--discovery-token-ca-cert-hash` flag or the `--discovery-token-unsafe-skip-ca-verification` flag.

### Other notable changes

* A new priority function `ResourceLimitsPriorityMap` (disabled by default and behind alpha feature gate and not part of the scheduler's default priority functions list) that assigns a lowest possible score of 1 to a node that satisfies one or both of input pod's cpu and memory limits, mainly to break ties between nodes with same scores. ([#55906](https://github.com/kubernetes/kubernetes/pull/55906), [@aveshagarwal](https://github.com/aveshagarwal))
* AWS: Fix detaching volume from stopped nodes. ([#55893](https://github.com/kubernetes/kubernetes/pull/55893), [@gnufied](https://github.com/gnufied))
* Fix stats summary network value when multiple network interfaces are available. ([#52144](https://github.com/kubernetes/kubernetes/pull/52144), [@andyxning](https://github.com/andyxning))
* Fix a typo in prometheus-to-sd configuration, that drops some stackdriver metrics. ([#56473](https://github.com/kubernetes/kubernetes/pull/56473), [@loburm](https://github.com/loburm))
* Fixes server name verification of aggregated API servers and webhook admission endpoints ([#56415](https://github.com/kubernetes/kubernetes/pull/56415), [@liggitt](https://github.com/liggitt))
* OpenStack cloud provider supports Cinder v3 API. ([#52910](https://github.com/kubernetes/kubernetes/pull/52910), [@FengyunPan](https://github.com/FengyunPan))
* kube-up: Add optional addon CoreDNS.  ([#55728](https://github.com/kubernetes/kubernetes/pull/55728), [@rajansandeep](https://github.com/rajansandeep))
    * Install CoreDNS instead of kube-dns by setting CLUSTER_DNS_CORE_DNS value to 'true'.
* kubeadm health checks can also be skipped with `--ignore-checks-errors` ([#56130](https://github.com/kubernetes/kubernetes/pull/56130), [@anguslees](https://github.com/anguslees))
* Adds kubeadm support for using ComponentConfig for the kube-proxy ([#55972](https://github.com/kubernetes/kubernetes/pull/55972), [@rpothier](https://github.com/rpothier))
* Pod Security Policy can now manage access to specific FlexVolume drivers ([#53179](https://github.com/kubernetes/kubernetes/pull/53179), [@wanghaoran1988](https://github.com/wanghaoran1988))
* PVC Finalizing Controller is introduced in order to prevent deletion of a PVC that is being used by a pod. ([#55824](https://github.com/kubernetes/kubernetes/pull/55824), [@pospispa](https://github.com/pospispa))
* Kubelet can provide full summary api support except container log stats for CRI container runtime now. ([#55810](https://github.com/kubernetes/kubernetes/pull/55810), [@abhi](https://github.com/abhi))
* Add support for resizing EBS disks ([#56118](https://github.com/kubernetes/kubernetes/pull/56118), [@gnufied](https://github.com/gnufied))
* Add PodDisruptionBudget support during pod preemption ([#56178](https://github.com/kubernetes/kubernetes/pull/56178), [@bsalamat](https://github.com/bsalamat))
* Fix CRI localhost seccomp path in format localhost//profileRoot/profileName. ([#55450](https://github.com/kubernetes/kubernetes/pull/55450), [@feiskyer](https://github.com/feiskyer))
* kubeadm: Add CoreDNS support for kubeadm "upgrade" and "alpha phases addons". ([#55952](https://github.com/kubernetes/kubernetes/pull/55952), [@rajansandeep](https://github.com/rajansandeep))
* The default garbage collection policy for Deployment, DaemonSet, StatefulSet, and ReplicaSet has changed from OrphanDependents to DeleteDependents when the deletion is requested through an `apps/v1` endpoint. Clients using older endpoints will be unaffected. This change is only at the REST API level and is independent of the default behavior of particular clients (e.g. this does not affect the default for the kubectl `--cascade` flag). ([#55148](https://github.com/kubernetes/kubernetes/pull/55148), [@dixudx](https://github.com/dixudx))
    * If you upgrade your client-go libs and use the `AppsV1()` interface, please note that the default garbage collection behavior is changed.
* Add resize support for ceph RBD ([#52767](https://github.com/kubernetes/kubernetes/pull/52767), [@NickrenREN](https://github.com/NickrenREN))
* Expose single annotation/label via downward API ([#55902](https://github.com/kubernetes/kubernetes/pull/55902), [@yguo0905](https://github.com/yguo0905))
* kubeadm: added `--print-join-command` flag for `kubeadm token create`. ([#56185](https://github.com/kubernetes/kubernetes/pull/56185), [@mattmoyer](https://github.com/mattmoyer))
* Implement kubelet side file system resizing. Also implement GCE PD resizing ([#55815](https://github.com/kubernetes/kubernetes/pull/55815), [@gnufied](https://github.com/gnufied))
* Improved PodSecurityPolicy admission latency, but validation errors are no longer limited to only errors from authorized policies. ([#55643](https://github.com/kubernetes/kubernetes/pull/55643), [@tallclair](https://github.com/tallclair))
* Add containerd monitoring support ([#56109](https://github.com/kubernetes/kubernetes/pull/56109), [@dashpole](https://github.com/dashpole))
* Add pod-level CPU and memory stats from pod cgroup information ([#55969](https://github.com/kubernetes/kubernetes/pull/55969), [@jingxu97](https://github.com/jingxu97))
* kubectl apply use openapi to calculate diff be default. It will fall back to use baked-in types when openapi is not available. ([#51321](https://github.com/kubernetes/kubernetes/pull/51321), [@mengqiy](https://github.com/mengqiy))
* It is now possible to override the healthcheck parameters for AWS ELBs via annotations on the corresponding service. The new annotations are `healthy-threshold`, `unhealthy-threshold`, `timeout`, `interval` (all prefixed with `service.beta.kubernetes.io/aws-load-balancer-healthcheck-`) ([#56024](https://github.com/kubernetes/kubernetes/pull/56024), [@dimpavloff](https://github.com/dimpavloff))
* Adding etcd version display to kubeadm upgrade plan subcommand ([#56156](https://github.com/kubernetes/kubernetes/pull/56156), [@sbezverk](https://github.com/sbezverk))
* [fluentd-gcp addon] Fixes fluentd deployment on GCP when custom resources are set. ([#55950](https://github.com/kubernetes/kubernetes/pull/55950), [@crassirostris](https://github.com/crassirostris))
* [fluentd-elasticsearch addon] Elasticsearch and Kibana are updated to version 5.6.4 ([#55400](https://github.com/kubernetes/kubernetes/pull/55400), [@mrahbar](https://github.com/mrahbar))
* install ipset in debian-iptables docker image ([#56115](https://github.com/kubernetes/kubernetes/pull/56115), [@m1093782566](https://github.com/m1093782566))
* Add cleanup-ipvs flag for kube-proxy  ([#56036](https://github.com/kubernetes/kubernetes/pull/56036), [@m1093782566](https://github.com/m1093782566))
* Remove opaque integer resources (OIR) support (deprecated in v1.8.) ([#55103](https://github.com/kubernetes/kubernetes/pull/55103), [@ConnorDoyle](https://github.com/ConnorDoyle))
* Implement volume resize for cinder ([#51498](https://github.com/kubernetes/kubernetes/pull/51498), [@NickrenREN](https://github.com/NickrenREN))
* Block volumes Support: FC plugin update ([#51493](https://github.com/kubernetes/kubernetes/pull/51493), [@mtanino](https://github.com/mtanino))
* kube-apiserver: fixed --oidc-username-prefix and --oidc-group-prefix flags which previously weren't correctly enabled ([#56175](https://github.com/kubernetes/kubernetes/pull/56175), [@ericchiang](https://github.com/ericchiang))
* New kubeadm flag `--ignore-preflight-errors` that enables to decrease severity of each individual error to warning. ([#56072](https://github.com/kubernetes/kubernetes/pull/56072), [@kad](https://github.com/kad))
    * Old flag `--skip-preflight-checks` is marked as deprecated and acts as `--ignore-preflight-errors=all`
* Block volumes Support: CRI, volumemanager and operationexecutor changes ([#51494](https://github.com/kubernetes/kubernetes/pull/51494), [@mtanino](https://github.com/mtanino))
* StatefulSet controller will create a label for each Pod in a StatefulSet. The label is named statefulset.kubernetes.io/pod-name and it is equal to the name of the Pod. This allows users to create a Service per Pod to expose a connection to individual Pods. ([#55329](https://github.com/kubernetes/kubernetes/pull/55329), [@kow3ns](https://github.com/kow3ns))
* Initial basic bootstrap-checkpoint support ([#50984](https://github.com/kubernetes/kubernetes/pull/50984), [@timothysc](https://github.com/timothysc))
* Add DNSConfig field to PodSpec and support "None" mode for DNSPolicy (Alpha). ([#55848](https://github.com/kubernetes/kubernetes/pull/55848), [@MrHohn](https://github.com/MrHohn))
* Add pod-level local ephemeral storage metric in Summary API. Pod-level ephemeral storage reports the total filesystem usage for the containers and emptyDir volumes in the measured Pod. ([#55447](https://github.com/kubernetes/kubernetes/pull/55447), [@jingxu97](https://github.com/jingxu97))
* Kubernetes update Azure nsg rules based on not just difference in Name, but also in Protocol, SourcePortRange, DestinationPortRange, SourceAddressPrefix, DestinationAddressPrefix, Access, and Direction. ([#55752](https://github.com/kubernetes/kubernetes/pull/55752), [@kevinkim9264](https://github.com/kevinkim9264))
* Add support to take nominated pods into account during scheduling to avoid starvation of higher priority pods. ([#55933](https://github.com/kubernetes/kubernetes/pull/55933), [@bsalamat](https://github.com/bsalamat))
* Add Amazon NLB support - Fixes [#52173](https://github.com/kubernetes/kubernetes/pull/52173) ([#53400](https://github.com/kubernetes/kubernetes/pull/53400), [@micahhausler](https://github.com/micahhausler))
* Extends deviceplugin to gracefully handle full device plugin lifecycle. ([#55088](https://github.com/kubernetes/kubernetes/pull/55088), [@jiayingz](https://github.com/jiayingz))
* A new field is added to CRI container log format to support splitting a long log line into multiple lines. ([#55922](https://github.com/kubernetes/kubernetes/pull/55922), [@Random-Liu](https://github.com/Random-Liu))
* [advanced audit]add a policy wide omitStage ([#54634](https://github.com/kubernetes/kubernetes/pull/54634), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Fix a bug in GCE multizonal clusters where PersistentVolumes were sometimes created in zones without nodes. ([#52322](https://github.com/kubernetes/kubernetes/pull/52322), [@davidz627](https://github.com/davidz627))
* With this change ([#55845](https://github.com/kubernetes/kubernetes/pull/55845), [@rohitjogvmw](https://github.com/rohitjogvmw))
    *  - User should be able to create k8s cluster which spans across multiple ESXi clusters, datacenters or even vCenters.
    *  - vSphere cloud provider (VCP) uses OS hostname and not vSphere Inventory VM Name.
    *    That means, now  VCP can handle cases where user changes VM inventory name.
    * - VCP can handle cases where VM migrates to other ESXi cluster or datacenter or vCenter.
    * The only requirement is the shared storage. VCP needs shared storage on all Node VMs.
* The RBAC bootstrapping policy now allows authenticated users to create selfsubjectrulesreviews. ([#56095](https://github.com/kubernetes/kubernetes/pull/56095), [@ericchiang](https://github.com/ericchiang))
* Defaulting of controller-manager options for --cluster-signing-cert-file and --cluster-signing-key-file is deprecated and will be removed in a later release. ([#54495](https://github.com/kubernetes/kubernetes/pull/54495), [@mikedanese](https://github.com/mikedanese))
* Add ExtendedResourceToleration admission controller. This facilitates creation of dedicated nodes with extended resources. If operators want to create dedicated nodes with extended resources (like GPUs, FPGAs etc.), they are expected to taint the node with extended resource name as the key. This admission controller, if enabled, automatically adds tolerations for such taints to pods requesting extended resources, so users don't have to manually add these tolerations.  ([#55839](https://github.com/kubernetes/kubernetes/pull/55839), [@mindprince](https://github.com/mindprince))
* Move unreachable taint key out of alpha.  ([#54208](https://github.com/kubernetes/kubernetes/pull/54208), [@resouer](https://github.com/resouer))
    * Please note the existing pods with the alpha toleration should be updated by user himself to tolerate the GA taint.
* add GRS, RAGRS storage account type support for azure disk ([#55931](https://github.com/kubernetes/kubernetes/pull/55931), [@andyzhangx](https://github.com/andyzhangx))
* Upgrading the kubernetes-master units now results in staged upgrades just like the kubernetes-worker nodes. Use the upgrade action in order to continue the upgrade process on each unit such as `juju run-action kubernetes-master/0 upgrade` ([#55990](https://github.com/kubernetes/kubernetes/pull/55990), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Using ipset doing SNAT and packet filtering in IPVS kube-proxy ([#54219](https://github.com/kubernetes/kubernetes/pull/54219), [@m1093782566](https://github.com/m1093782566))
* Add a new scheduling queue that helps schedule the highest priority pending pod first. ([#55109](https://github.com/kubernetes/kubernetes/pull/55109), [@bsalamat](https://github.com/bsalamat))
* Adds to **kubeadm upgrade apply**, a new **--etcd-upgrade** keyword. When this keyword is specified, etcd's static pod gets upgraded to the etcd version officially recommended for a target kubernetes release. ([#55010](https://github.com/kubernetes/kubernetes/pull/55010), [@sbezverk](https://github.com/sbezverk))
* Adding vishh as an reviewer/approver for hack directory ([#54007](https://github.com/kubernetes/kubernetes/pull/54007), [@vishh](https://github.com/vishh))
* The `GenericAdmissionWebhook` is renamed as `ValidatingAdmissionWebhook`. Please update you apiserver configuration file to use the new name to pass to the apiserver's `--admission-control` flag. ([#55988](https://github.com/kubernetes/kubernetes/pull/55988), [@caesarxuchao](https://github.com/caesarxuchao))
* iSCSI Persistent Volume Sources can now reference CHAP Secrets in namespaces other than the namespace of the bound Persistent Volume Claim ([#51530](https://github.com/kubernetes/kubernetes/pull/51530), [@rootfs](https://github.com/rootfs))
* Bugfix: master startup script on GCP no longer fails randomly due to concurrent iptables invocations. ([#55945](https://github.com/kubernetes/kubernetes/pull/55945), [@x13n](https://github.com/x13n))
* fix azure disk storage account init issue ([#55927](https://github.com/kubernetes/kubernetes/pull/55927), [@andyzhangx](https://github.com/andyzhangx))
* Allow code-generator tags in the 2nd closest comment block and directly above a statement. ([#55233](https://github.com/kubernetes/kubernetes/pull/55233), [@sttts](https://github.com/sttts))
* Ensure additional resource tags are set/updated AWS load balancers ([#55731](https://github.com/kubernetes/kubernetes/pull/55731), [@georgebuckerfield](https://github.com/georgebuckerfield))
* `kubectl get` will now use OpenAPI schema extensions by default to select columns for custom types. ([#53483](https://github.com/kubernetes/kubernetes/pull/53483), [@apelisse](https://github.com/apelisse))
* AWS: Apply taint to a node if volumes being attached to it are stuck in attaching state ([#55558](https://github.com/kubernetes/kubernetes/pull/55558), [@gnufied](https://github.com/gnufied))
* Kubeadm now supports for Kubelet Dynamic Configuration. ([#55803](https://github.com/kubernetes/kubernetes/pull/55803), [@xiangpengzhao](https://github.com/xiangpengzhao))
* Added mutation supports to admission webhooks. ([#54892](https://github.com/kubernetes/kubernetes/pull/54892), [@caesarxuchao](https://github.com/caesarxuchao))
* Upgrade to go1.9.2 ([#55420](https://github.com/kubernetes/kubernetes/pull/55420), [@cblecker](https://github.com/cblecker))
* If a non-absolute mountPath is passed to the kubelet, prefix it with the appropriate root path. ([#55665](https://github.com/kubernetes/kubernetes/pull/55665), [@brendandburns](https://github.com/brendandburns))
* action-required: please update your admission webhook to use the latest [Admission API](https://github.com/kubernetes/api/tree/master/admission). ([#55829](https://github.com/kubernetes/kubernetes/pull/55829), [@cheftako](https://github.com/cheftako))
    * `admission/v1alpha1#AdmissionReview` now contains `AdmissionRequest` and `AdmissionResponse`. `AdmissionResponse` includes a `Patch` field to allow mutating webhooks to send json patch to the apiserver.
* support mount options in azure file ([#54674](https://github.com/kubernetes/kubernetes/pull/54674), [@andyzhangx](https://github.com/andyzhangx))
* Support AWS ECR credentials in China ([#50108](https://github.com/kubernetes/kubernetes/pull/50108), [@zzq889](https://github.com/zzq889))
* The EvictionHard, EvictionSoft, EvictionSoftGracePeriod, EvictionMinimumReclaim, SystemReserved, and KubeReserved fields in the KubeletConfiguration object (kubeletconfig/v1alpha1) are now of type map[string]string, which facilitates writing JSON and YAML files. ([#54823](https://github.com/kubernetes/kubernetes/pull/54823), [@mtaufen](https://github.com/mtaufen))
* Added service annotation for AWS ELB SSL policy ([#54507](https://github.com/kubernetes/kubernetes/pull/54507), [@micahhausler](https://github.com/micahhausler))
* Implement correction mechanism for dangling volumes attached for deleted pods ([#55491](https://github.com/kubernetes/kubernetes/pull/55491), [@gnufied](https://github.com/gnufied))
* Promote validation for custom resources defined through CRD to beta ([#54647](https://github.com/kubernetes/kubernetes/pull/54647), [@colemickens](https://github.com/colemickens))
* Octavia v2 now supported as a LB provider ([#55393](https://github.com/kubernetes/kubernetes/pull/55393), [@jamiehannaford](https://github.com/jamiehannaford))
* Kubelet now exposes metrics for NVIDIA GPUs attached to the containers. ([#55188](https://github.com/kubernetes/kubernetes/pull/55188), [@mindprince](https://github.com/mindprince))
* Addon manager supports HA masters. ([#55782](https://github.com/kubernetes/kubernetes/pull/55782), [@x13n](https://github.com/x13n))
* Fix kubeadm reset crictl command ([#55717](https://github.com/kubernetes/kubernetes/pull/55717), [@runcom](https://github.com/runcom))
* Fix code-generators to produce correct code when GroupName, PackageName and/or GoName differ. ([#55614](https://github.com/kubernetes/kubernetes/pull/55614), [@sttts](https://github.com/sttts))
* Fixes bad conversion in host port chain name generating func which leads to some unreachable host ports. ([#55153](https://github.com/kubernetes/kubernetes/pull/55153), [@chenchun](https://github.com/chenchun))
* Relative paths in the Kubelet's local config files (--init-config-dir) will be resolved relative to the location of the containing files. ([#55648](https://github.com/kubernetes/kubernetes/pull/55648), [@mtaufen](https://github.com/mtaufen))
* kubeadm: Fix a bug on some OSes where the kubelet tried to mount a volume path that is non-existent and on a read-only filesystem  ([#55320](https://github.com/kubernetes/kubernetes/pull/55320), [@andrewrynhard](https://github.com/andrewrynhard))
* add hostIP and protocol to the original hostport predicates procedure in scheduler. ([#52421](https://github.com/kubernetes/kubernetes/pull/52421), [@WIZARD-CXY](https://github.com/WIZARD-CXY))



# v1.9.0-alpha.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.9.0-alpha.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes.tar.gz) | `dce2a70ca51fb4f8979645330f36c346b9c02be0501708380ae50956485a53a4`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-src.tar.gz) | `4a8c8eaf32c83968e18f75888ae0d432210262090893cad0a105eebab82b0302`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `354d6c8d65e4248c3393a3789e9394b8c31c63da4c42f3da60c7b8bc4713ad51`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `98c53e4108276535218f4c89c58974121cc28308cecf5bca676f68fa083a62c5`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `c0dc219073dcae6fb654f33ca6d83faf5f37a2dcba3cc86b32ea5f9e18054faa`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `df68fc512d173d1914f7863303cc0a4335439eb76000fa5a6134d5c454f4ef31`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `edbf086c5446a7b48bbf5ac0e65dacf472e7e2eb7ac434ffb4835b0c643363a4`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `138b02e0e96e9e30772e814d2650b40594e9f190442c9b31af5dcf4bd3c29fb2`
[kubernetes-client-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `8edb568048f64052e9ab3e2f0d9d9fee3a5c90667d00669d815c07cc1986eb03`
[kubernetes-client-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `9f0f0464041e85221cb65ab5908f7295d7237acdb6a39abff062e40be0a53b4c`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `a9d4b6014c2856b0602b7124dad41f2f932cccea7f48ba57583352f0fbf2710f`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `16827a05b0538ab8ef6e47b173dc5ad1c4398070324b0d2fc0510ad1efe66567`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `e2aad29fff3cc3a98c642d8bc021a6caa42b4143696ca9d42a1ae3f7e803e777`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `a7e2370d29086dadcb59fc4c3f6e88610ef72ff168577cc1854b4e9c221cad8a`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `b8da04e06946b221b2ac4f6ebc8e0900cf8e750f0ca5d2e213984644048d1903`
[kubernetes-server-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `539db8044dcacc154fff92029d7c18ac9a68de426477cabcd52e01053e8de6e6`
[kubernetes-server-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `d793be99d39f1f7b55d381f656b059e4cd78418a6c6bcc77c2c026db82e98769`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `22dae55dd97026eae31562fde6d8459f1594b050313ef294e009144aa8c27a8e`
[kubernetes-node-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `8d9bd9307cd5463b2e13717c862e171e20a1ba29a91d86fa3918a460006c823b`
[kubernetes-node-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `c696f882b4a95b13c8cf3c2e05695decb81407359911fba169a308165b06be55`
[kubernetes-node-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `611a0e04b1014263e66be91ef108a4a56291cae1438da562b157d04dfe84fd1a`
[kubernetes-node-linux-s390x.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `61b619e3af7fcb836072c4b855978d7d76d6256aa99b9378488f063494518a0e`
[kubernetes-node-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release-dashpole/release/v1.9.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `c274258e4379f0b50b2023f659bc982a82783c3de3ae07ef2759159300175a8a`

## Changelog since v1.9.0-alpha.2

### Action Required

* action required: The `storage.k8s.io/v1beta1` API  and `volume.beta.kubernetes.io/storage-class` annotation are deprecated. They will be removed in a future release. Please use v1 API and field `v1.PersistentVolumeClaim.Spec.StorageClassName`/`v1.PersistentVolume.Spec.StorageClassName` instead. ([#53580](https://github.com/kubernetes/kubernetes/pull/53580), [@xiangpengzhao](https://github.com/xiangpengzhao))
* action required: Deprecated flags `--portal-net` and `service-node-ports` of kube-apiserver are removed. ([#52547](https://github.com/kubernetes/kubernetes/pull/52547), [@xiangpengzhao](https://github.com/xiangpengzhao))
* The `node.kubernetes.io/memory-pressure` taint now respects the configured whitelist.  If you need to use it, you'll have to add it to the whitelist. ([#55251](https://github.com/kubernetes/kubernetes/pull/55251), [@deads2k](https://github.com/deads2k))

### Other notable changes

* hyperkube: add cloud-controller-manager ([#54197](https://github.com/kubernetes/kubernetes/pull/54197), [@colemickens](https://github.com/colemickens))
* Metrics have been added for monitoring admission plugins, including the new dynamic (webhook-based) ones. ([#55183](https://github.com/kubernetes/kubernetes/pull/55183), [@jpbetz](https://github.com/jpbetz))
* Addon manager supports HA masters. ([#55466](https://github.com/kubernetes/kubernetes/pull/55466), [@x13n](https://github.com/x13n))
* - Add PodSecurityPolicies for cluster addons ([#55509](https://github.com/kubernetes/kubernetes/pull/55509), [@tallclair](https://github.com/tallclair))
    * - Remove SSL cert HostPath volumes from heapster addons
* Add iptables rules to allow Pod traffic even when default iptables policy is to reject. ([#52569](https://github.com/kubernetes/kubernetes/pull/52569), [@tmjd](https://github.com/tmjd))
* Validate positive capacity for PVs and PVCs. ([#55532](https://github.com/kubernetes/kubernetes/pull/55532), [@ianchakeres](https://github.com/ianchakeres))
* Kubelet supports running mount utilities and final mount in a container instead running them on the host. ([#53440](https://github.com/kubernetes/kubernetes/pull/53440), [@jsafrane](https://github.com/jsafrane))
* The PodSecurityPolicy annotation `kubernetes.io/psp` on pods is only set once on create. ([#55486](https://github.com/kubernetes/kubernetes/pull/55486), [@sttts](https://github.com/sttts))
* The apiserver sends external versioned object to the admission webhooks now. Please update the webhooks to expect admissionReview.spec.object.raw to be serialized external versions of objects.  ([#55127](https://github.com/kubernetes/kubernetes/pull/55127), [@caesarxuchao](https://github.com/caesarxuchao))
* RBAC ClusterRoles can now select other roles to aggregate ([#54005](https://github.com/kubernetes/kubernetes/pull/54005), [@deads2k](https://github.com/deads2k))
* GCE nodes with NVIDIA GPUs attached now expose `nvidia.com/gpu` as a resource instead of `alpha.kubernetes.io/nvidia-gpu`. ([#54826](https://github.com/kubernetes/kubernetes/pull/54826), [@mindprince](https://github.com/mindprince))
* Remove docker dependency during kubelet start up  ([#54405](https://github.com/kubernetes/kubernetes/pull/54405), [@resouer](https://github.com/resouer))
* Fix session affinity issue with external load balancer traffic when ExternalTrafficPolicy=Local. ([#55519](https://github.com/kubernetes/kubernetes/pull/55519), [@MrHohn](https://github.com/MrHohn))
* Add the concurrent service sync flag to the Cloud Controller Manager to allow changing the number of workers. (`--concurrent-service-syncs`) ([#55561](https://github.com/kubernetes/kubernetes/pull/55561), [@jhorwit2](https://github.com/jhorwit2))
* move IsMissingVersion comments ([#55523](https://github.com/kubernetes/kubernetes/pull/55523), [@chenpengdev](https://github.com/chenpengdev))
* The dynamic admission webhook now supports a URL in addition to a service reference, to accommodate out-of-cluster webhooks. ([#54889](https://github.com/kubernetes/kubernetes/pull/54889), [@lavalamp](https://github.com/lavalamp))
* Correct wording of kubeadm upgrade response for missing ConfigMap. ([#53337](https://github.com/kubernetes/kubernetes/pull/53337), [@jmhardison](https://github.com/jmhardison))
* add create priorityclass sub command ([#54858](https://github.com/kubernetes/kubernetes/pull/54858), [@wackxu](https://github.com/wackxu))
* Added namespaceSelector to externalAdmissionWebhook configuration to allow applying webhooks only to objects in the namespaces that have matching labels. ([#54727](https://github.com/kubernetes/kubernetes/pull/54727), [@caesarxuchao](https://github.com/caesarxuchao))
* Base images bumped to Debian Stretch (9) ([#52744](https://github.com/kubernetes/kubernetes/pull/52744), [@rphillips](https://github.com/rphillips))
* [fluentd-elasticsearch addon] Elasticsearch service name can be overridden via env variable ELASTICSEARCH_SERVICE_NAME ([#54215](https://github.com/kubernetes/kubernetes/pull/54215), [@mrahbar](https://github.com/mrahbar))
* Increase waiting time (120s) for docker startup in health-monitor.sh ([#54099](https://github.com/kubernetes/kubernetes/pull/54099), [@dchen1107](https://github.com/dchen1107))
* not calculate new priority when user update other spec of a pod ([#55221](https://github.com/kubernetes/kubernetes/pull/55221), [@CaoShuFeng](https://github.com/CaoShuFeng))
* kubectl create pdb will no longer set the min-available field by default.  ([#53047](https://github.com/kubernetes/kubernetes/pull/53047), [@yuexiao-wang](https://github.com/yuexiao-wang))
* StatefulSet status now has support for conditions, making it consistent with other core controllers in v1  ([#55268](https://github.com/kubernetes/kubernetes/pull/55268), [@foxish](https://github.com/foxish))
* kubeadm: use the CRI for preflights checks ([#55055](https://github.com/kubernetes/kubernetes/pull/55055), [@runcom](https://github.com/runcom))
* kubeadm now produces error during preflight checks if swap is enabled. Users, who can setup kubelet to run in unsupported environment with enabled swap, will be able to skip that preflight check. ([#55399](https://github.com/kubernetes/kubernetes/pull/55399), [@kad](https://github.com/kad))
* - kubeadm will produce error if kubelet too new for control plane ([#54868](https://github.com/kubernetes/kubernetes/pull/54868), [@kad](https://github.com/kad))
* validate if default and defaultRequest match when creating LimitRange for GPU and hugepages. ([#54919](https://github.com/kubernetes/kubernetes/pull/54919), [@tianshapjq](https://github.com/tianshapjq))
* Add extra-args configs to kubernetes-worker charm ([#55334](https://github.com/kubernetes/kubernetes/pull/55334), [@Cynerva](https://github.com/Cynerva))
* Restored kube-proxy's support for 0 values for conntrack min, max, max per core, tcp close wait timeout, and tcp established timeout. ([#55261](https://github.com/kubernetes/kubernetes/pull/55261), [@ncdc](https://github.com/ncdc))
* Audit policy files without apiVersion and kind are treated as invalid. ([#54267](https://github.com/kubernetes/kubernetes/pull/54267), [@ericchiang](https://github.com/ericchiang))
* ReplicationController now shares its underlying controller implementation with ReplicaSet to reduce the maintenance burden going forward. However, they are still separate resources and there should be no externally visible effects from this change. ([#49429](https://github.com/kubernetes/kubernetes/pull/49429), [@enisoc](https://github.com/enisoc))
* Add limitrange/resourcequota/downward_api  e2e tests for local ephemeral storage ([#52523](https://github.com/kubernetes/kubernetes/pull/52523), [@NickrenREN](https://github.com/NickrenREN))
* Support copying "options" in resolv.conf into pod sandbox when dnsPolicy is Default ([#54773](https://github.com/kubernetes/kubernetes/pull/54773), [@phsiao](https://github.com/phsiao))
* Fix support for configmap resource lock type in CCM ([#55125](https://github.com/kubernetes/kubernetes/pull/55125), [@jhorwit2](https://github.com/jhorwit2))
* The minimum supported go version bumps to 1.9.1. ([#55301](https://github.com/kubernetes/kubernetes/pull/55301), [@xiangpengzhao](https://github.com/xiangpengzhao))
* GCE: provide an option to disable docker's live-restore on COS/ubuntu ([#55260](https://github.com/kubernetes/kubernetes/pull/55260), [@yujuhong](https://github.com/yujuhong))
* Azure NSG rules for services exposed via external load balancer  ([#54177](https://github.com/kubernetes/kubernetes/pull/54177), [@itowlson](https://github.com/itowlson))
    * now limit the destination IP address to the relevant front end load 
    * balancer IP.
* DaemonSet status now has a new field named "conditions", making it consistent with other workloads controllers. ([#55272](https://github.com/kubernetes/kubernetes/pull/55272), [@janetkuo](https://github.com/janetkuo))
* kubeadm: Add an experimental mode to deploy CoreDNS instead of KubeDNS ([#52501](https://github.com/kubernetes/kubernetes/pull/52501), [@rajansandeep](https://github.com/rajansandeep))
* Allow HPA to read custom metrics. ([#54854](https://github.com/kubernetes/kubernetes/pull/54854), [@kawych](https://github.com/kawych))
* Fixed 'Schedulercache is corrupted' error in kube-scheduler ([#55262](https://github.com/kubernetes/kubernetes/pull/55262), [@liggitt](https://github.com/liggitt))
* API discovery failures no longer crash the kube controller manager via the garbage collector. ([#55259](https://github.com/kubernetes/kubernetes/pull/55259), [@ironcladlou](https://github.com/ironcladlou))
* The kube-scheduler command now supports a `--config` flag which is the location of a file containing a serialized scheduler configuration. Most other kube-scheduler flags are now deprecated. ([#52562](https://github.com/kubernetes/kubernetes/pull/52562), [@ironcladlou](https://github.com/ironcladlou))
* add field selector for kubectl get ([#50140](https://github.com/kubernetes/kubernetes/pull/50140), [@dixudx](https://github.com/dixudx))
* Removes Priority Admission Controller from kubeadm since it's alpha.  ([#55237](https://github.com/kubernetes/kubernetes/pull/55237), [@andrewsykim](https://github.com/andrewsykim))
* Add support for the webhook authorizer to make a Deny decision that short-circuits the union authorizer and immediately returns Deny.  ([#53273](https://github.com/kubernetes/kubernetes/pull/53273), [@mikedanese](https://github.com/mikedanese))
* kubeadm init: fix a bug that prevented the --token-ttl flag and tokenTTL configuration value from working as expected for infinite (0) values. ([#54640](https://github.com/kubernetes/kubernetes/pull/54640), [@mattmoyer](https://github.com/mattmoyer))
* Add CRI log parsing library at pkg/kubelet/apis/cri/logs ([#55140](https://github.com/kubernetes/kubernetes/pull/55140), [@feiskyer](https://github.com/feiskyer))
* Add extra-args configs for scheduler and controller-manager to kubernetes-master charm ([#55185](https://github.com/kubernetes/kubernetes/pull/55185), [@Cynerva](https://github.com/Cynerva))
* Add masquerading rules by default to GCE/GKE ([#55178](https://github.com/kubernetes/kubernetes/pull/55178), [@dnardo](https://github.com/dnardo))
* Upgraded Azure SDK to v11.1.1. ([#54971](https://github.com/kubernetes/kubernetes/pull/54971), [@itowlson](https://github.com/itowlson))
* Disable the termination grace period for the calico/node add-on DaemonSet to reduce downtime during a rolling upgrade or deletion. ([#55015](https://github.com/kubernetes/kubernetes/pull/55015), [@fasaxc](https://github.com/fasaxc))
* Google KMS integration was removed from in-tree in favor of a out-of-process extension point that will be used for all KMS providers. ([#54759](https://github.com/kubernetes/kubernetes/pull/54759), [@sakshamsharma](https://github.com/sakshamsharma))
* kubeadm: reset: use crictl to reset containers ([#54721](https://github.com/kubernetes/kubernetes/pull/54721), [@runcom](https://github.com/runcom))
* Check for available volume before attach/delete operation in EBS ([#55008](https://github.com/kubernetes/kubernetes/pull/55008), [@gnufied](https://github.com/gnufied))
* DaemonSet, Deployment, ReplicaSet, and StatefulSet have been promoted to GA and are available in the apps/v1 group version. ([#53679](https://github.com/kubernetes/kubernetes/pull/53679), [@kow3ns](https://github.com/kow3ns))
* In conversion-gen removed Kubernetes core API from default extra-peer-dirs. ([#54394](https://github.com/kubernetes/kubernetes/pull/54394), [@sttts](https://github.com/sttts))
* Fix IPVS availability check ([#51874](https://github.com/kubernetes/kubernetes/pull/51874), [@vfreex](https://github.com/vfreex))
* ScaleIO driver completely removes dependency on drv_cfg binary so a Kubernetes cluster can easily run a containerized kubelet. ([#54956](https://github.com/kubernetes/kubernetes/pull/54956), [@vladimirvivien](https://github.com/vladimirvivien))
* Avoid unnecessary spam in kube-controller-manager log if --cluster-cidr is not specified and --allocate-node-cidrs is false. ([#54934](https://github.com/kubernetes/kubernetes/pull/54934), [@akosiaris](https://github.com/akosiaris))
* It is now possible to set multiple manifest url headers via the Kubelet's --manifest-url-header flag. Multiple headers for the same key will be added in the order provided. The ManifestURLHeader field in KubeletConfiguration object (kubeletconfig/v1alpha1) is now a map[string][]string, which facilitates writing JSON and YAML files. ([#54643](https://github.com/kubernetes/kubernetes/pull/54643), [@mtaufen](https://github.com/mtaufen))
* Add support for PodSecurityPolicy on GCE: `ENABLE_POD_SECURITY_POLICY=true` enables the admission controller, and installs policies for default addons. ([#52367](https://github.com/kubernetes/kubernetes/pull/52367), [@tallclair](https://github.com/tallclair))
* In PodTolerationRestriction admisson plugin, if namespace level tolerations are empty, now they override cluster level tolerations.  ([#54812](https://github.com/kubernetes/kubernetes/pull/54812), [@aveshagarwal](https://github.com/aveshagarwal))
* - Fix handling of IPv6 URLs in NO_PROXY. ([#53898](https://github.com/kubernetes/kubernetes/pull/53898), [@kad](https://github.com/kad))
* Added extra_sans config option to kubeapi-load-balancer charm. This allows the user to specify extra SAN entries on the certificate generated for the load balancer. ([#54947](https://github.com/kubernetes/kubernetes/pull/54947), [@hyperbolic2346](https://github.com/hyperbolic2346))
* set leveled logging (v=4) for 'updating container' message ([#54865](https://github.com/kubernetes/kubernetes/pull/54865), [@phsiao](https://github.com/phsiao))
* Fix a bug where pod address is not removed from endpoints object while pod is in graceful termination. ([#54828](https://github.com/kubernetes/kubernetes/pull/54828), [@freehan](https://github.com/freehan))
* kubeadm: Add support for adding a Windows node ([#53553](https://github.com/kubernetes/kubernetes/pull/53553), [@bsteciuk](https://github.com/bsteciuk))



# v1.9.0-alpha.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.9.0-alpha.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes.tar.gz) | `9d548271e8475171114b3b68323ab3c0e024cf54e25debe4702ffafe3f1d0952`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-src.tar.gz) | `99901fa7f996ddf75ecab7fcd1d33a3faca38e9d1398daa2ae30c9b3ac6a71ce`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `5a5e1ce20db98d7f7f0c88957440ab6c7d4b4a4dfefcb31dcd1d6546e9db01d6`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `094481f8f650321f39ba79cd6348de5052db2bb3820f55a74cf5ce33d5c98701`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `9a7d8e682a35772ba24bd3fa7a06fb153067b9387daa4db285e15dda75de757d`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `3bb742ffed1a6a51cac01c16614873fea2864c2a4432057a15db90a9d7e40aed`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `928936f06161e8a6f40196381d3e0dc215ca7e7dbc5f7fe6ebccd8d8268b8177`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `0a0fa24107f490db0ad57f33638b1aa9ba2baccb5f250caa75405d6612a3e10a`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `a92f790d1a480318ea206d84d24d2c1d7e43c3683e60f22e7735b63ee73ccbb4`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `1bfb7f056ad91fcbc50657fb9760310a0920c15a5770eaa74cf1a17b1725a232`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `d1b0abbc9cd0376fa0d56096e42094db8a40485082b301723d05c8e78d8f4717`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `69799ea8741caadac8949a120a455e08aba4d2babba6b63fba2ee9aaeb10c84b`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `f3d9f67e94176aa65cffcc6557a7a251ec2384a3f89a81d3daedd8f8dd4c51a7`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `3747b7e26d8bfba59c53b3f20d547e7e90cbb9356e513183ac27f901d7317630`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `397b7a49adf90735ceea54720dbf012c8566b34dadde911599bceefb507bc29a`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `56f76ebb0788c4e23fc3ede36b52eb34b50b456bb5ff0cf7d78c383c04837565`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `83d961657a50513db82bf421854c567206ccd34240eb8a017167cb98bdb6d38f`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `1bb0f5ac920e27b4e51260a80fbfaa013ed7d446d58cd1f9d5f73af4d9517edf`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `47635b9097fc6e3d9b1f1f2c3bd1558d144b1a26d1bf03cfc2e97a3c6db4c439`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `212117f1d027c79d50e7c7388951da40b440943748691ba82a3f9f6af75b3ed0`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `f2b1d086d07bf2f807dbf02e1f0cd7f6528e57c55be9dadfcecde73e73980068`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `ba6803a5c065b06cf43d1db674319008f15d4bc45900299d0b90105002af245e`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `6d928e3bdba87db3b9198e02f84696f7345b4b78d07ff4ea048d47691c67b212`

## Changelog since v1.8.0

### Action Required

* PodSecurityPolicy: Fixes a compatibility issue that caused policies that previously allowed privileged pods to start forbidding them, due to an incorrect default value for `allowPrivilegeEscalation`. PodSecurityPolicy objects defined using a 1.8.0 client or server that intended to set `allowPrivilegeEscalation` to `false` must be reapplied after upgrading to 1.8.1. ([#53443](https://github.com/kubernetes/kubernetes/pull/53443), [@liggitt](https://github.com/liggitt))
* RBAC objects are now stored in etcd in v1 format. After completing an upgrade to 1.9, RBAC objects (Roles, RoleBindings, ClusterRoles, ClusterRoleBindings) should be migrated to ensure all persisted objects are written in `v1` format, prior to `v1alpha1` support being removed in a future release. ([#52950](https://github.com/kubernetes/kubernetes/pull/52950), [@liggitt](https://github.com/liggitt))

### Other notable changes

* Log error of failed healthz check ([#53048](https://github.com/kubernetes/kubernetes/pull/53048), [@mrIncompetent](https://github.com/mrIncompetent))
* fix azure file mount limit issue on windows due to using drive letter ([#53629](https://github.com/kubernetes/kubernetes/pull/53629), [@andyzhangx](https://github.com/andyzhangx))
* Update AWS SDK to 1.12.7 ([#53561](https://github.com/kubernetes/kubernetes/pull/53561), [@justinsb](https://github.com/justinsb))
* The `kubernetes.io/created-by` annotation is no longer added to controller-created objects. Use the  `metadata.ownerReferences` item that has `controller` set to `true` to determine which controller, if any, owns an object. ([#54445](https://github.com/kubernetes/kubernetes/pull/54445), [@crimsonfaith91](https://github.com/crimsonfaith91))
* Fix overlay2 container disk metrics for Docker and CRI-O ([#54827](https://github.com/kubernetes/kubernetes/pull/54827), [@dashpole](https://github.com/dashpole))
* Fix iptables FORWARD policy for Docker 1.13 in kubernetes-worker charm ([#54796](https://github.com/kubernetes/kubernetes/pull/54796), [@Cynerva](https://github.com/Cynerva))
* Fix `kubeadm upgrade plan` for offline operation: ignore errors when trying to fetch latest versions from dl.k8s.io ([#54016](https://github.com/kubernetes/kubernetes/pull/54016), [@praseodym](https://github.com/praseodym))
* fluentd now supports CRI log format. ([#54777](https://github.com/kubernetes/kubernetes/pull/54777), [@Random-Liu](https://github.com/Random-Liu))
* Validate that PersistentVolumeSource is not changed during PV Update ([#54761](https://github.com/kubernetes/kubernetes/pull/54761), [@ianchakeres](https://github.com/ianchakeres))
* If you are using the cloud provider API to determine the external host address of the apiserver, set --external-hostname explicitly instead. The cloud provider detection has been deprecated and will be removed in the future ([#54516](https://github.com/kubernetes/kubernetes/pull/54516), [@dims](https://github.com/dims))
* Fixes discovery information for scale subresources in the apps API group ([#54683](https://github.com/kubernetes/kubernetes/pull/54683), [@liggitt](https://github.com/liggitt))
* Optimize Repeated registration of AlgorithmProvider when ApplyFeatureGates ([#54047](https://github.com/kubernetes/kubernetes/pull/54047), [@kuramal](https://github.com/kuramal))
* `kubectl get` will by default fetch large lists of resources in chunks of up to 500 items rather than requesting all resources up front from the server. This reduces the perceived latency of managing large clusters since the server returns the first set of results to the client much more quickly.  A new flag `--chunk-size=SIZE` may be used to alter the number of items or disable this feature when `0` is passed.  This is a beta feature. ([#53768](https://github.com/kubernetes/kubernetes/pull/53768), [@smarterclayton](https://github.com/smarterclayton))
* Add a new feature gate for enabling an alpha annotation which, if present, excludes the annotated node from being added to a service load balancers. ([#54644](https://github.com/kubernetes/kubernetes/pull/54644), [@brendandburns](https://github.com/brendandburns))
* Implement graceful shutdown of the kube-apiserver by waiting for open connections to finish before exiting. Moreover, the audit backend will stop dropping events on shutdown. ([#53695](https://github.com/kubernetes/kubernetes/pull/53695), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* fix warning messages due to GetMountRefs func not implemented in windows ([#52401](https://github.com/kubernetes/kubernetes/pull/52401), [@andyzhangx](https://github.com/andyzhangx))
* Object count quotas supported on all standard resources using `count/<resource>.<group>` syntax ([#54320](https://github.com/kubernetes/kubernetes/pull/54320), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Add openssh-client back into the hyperkube image. This allows the gitRepo volume plugin to work properly. ([#54250](https://github.com/kubernetes/kubernetes/pull/54250), [@ixdy](https://github.com/ixdy))
* Bump version of all prometheus-to-sd images to v0.2.2. ([#54635](https://github.com/kubernetes/kubernetes/pull/54635), [@loburm](https://github.com/loburm))
* [fluentd-gcp addon] Fluentd now runs in its own network, not in the host one. ([#54395](https://github.com/kubernetes/kubernetes/pull/54395), [@crassirostris](https://github.com/crassirostris))
* fix azure storage account num exhausting issue ([#54459](https://github.com/kubernetes/kubernetes/pull/54459), [@andyzhangx](https://github.com/andyzhangx))
* Add Windows support to the system verification check ([#53730](https://github.com/kubernetes/kubernetes/pull/53730), [@bsteciuk](https://github.com/bsteciuk))
* allow windows mount path ([#51240](https://github.com/kubernetes/kubernetes/pull/51240), [@andyzhangx](https://github.com/andyzhangx))
* Development of Kubernetes Federation has moved to github.com/kubernetes/federation.  This move out of tree also means that Federation will begin releasing separately from Kubernetes.  The impact of this is Federation-specific behavior will no longer be included in kubectl, kubefed will no longer be released as part of Kubernetes, and the Federation servers will no longer be included in the hyperkube binary and image. ([#53816](https://github.com/kubernetes/kubernetes/pull/53816), [@marun](https://github.com/marun))
* Metadata concealment on GCE is now controlled by the `ENABLE_METADATA_CONCEALMENT` env var.  See cluster/gce/config-default.sh for more info. ([#54150](https://github.com/kubernetes/kubernetes/pull/54150), [@ihmccreery](https://github.com/ihmccreery))
* Fixed a bug which is causes kube-apiserver to not run without specifying service-cluster-ip-range ([#52870](https://github.com/kubernetes/kubernetes/pull/52870), [@jennybuckley](https://github.com/jennybuckley))
* the generic admission webhook is now available in the generic apiserver ([#54513](https://github.com/kubernetes/kubernetes/pull/54513), [@deads2k](https://github.com/deads2k))
* ScaleIO persistent volumes now support referencing a secret in a namespace other than the bound persistent volume claim's namespace; this is controlled during provisioning with the `secretNamespace` storage class parameter; StoragePool and ProtectionDomain attributes no longer defaults to the value `default` ([#54013](https://github.com/kubernetes/kubernetes/pull/54013), [@vladimirvivien](https://github.com/vladimirvivien))
* Feature gates now check minimum versions ([#54539](https://github.com/kubernetes/kubernetes/pull/54539), [@jamiehannaford](https://github.com/jamiehannaford))
* fix azure pv crash due to volumeSource.ReadOnly value nil ([#54607](https://github.com/kubernetes/kubernetes/pull/54607), [@andyzhangx](https://github.com/andyzhangx))
* Fix an issue where pods were briefly transitioned to a "Pending" state during the deletion process. ([#54593](https://github.com/kubernetes/kubernetes/pull/54593), [@dashpole](https://github.com/dashpole))
* move getMaxVols function to predicates.go and add some NewVolumeCountPredicate funcs ([#51783](https://github.com/kubernetes/kubernetes/pull/51783), [@jiulongzaitian](https://github.com/jiulongzaitian))
* Remove the LbaasV1 of OpenStack cloud provider, currently only support LbaasV2. ([#52717](https://github.com/kubernetes/kubernetes/pull/52717), [@FengyunPan](https://github.com/FengyunPan))
* generic webhook admission now takes a config file which describes how to authenticate to webhook servers ([#54414](https://github.com/kubernetes/kubernetes/pull/54414), [@deads2k](https://github.com/deads2k))
* The NodeController will not support kubelet 1.2. ([#48996](https://github.com/kubernetes/kubernetes/pull/48996), [@k82cn](https://github.com/k82cn))
* - fluentd-gcp runs with a dedicated fluentd-gcp service account ([#54175](https://github.com/kubernetes/kubernetes/pull/54175), [@tallclair](https://github.com/tallclair))
    * - Stop mounting the host certificates into fluentd's prometheus-to-sd container
* fix azure disk mount failure on coreos and some other distros ([#54334](https://github.com/kubernetes/kubernetes/pull/54334), [@andyzhangx](https://github.com/andyzhangx))
* Allow GCE users to configure the service account made available on their nodes ([#52868](https://github.com/kubernetes/kubernetes/pull/52868), [@ihmccreery](https://github.com/ihmccreery))
* Load kernel modules automatically inside a kube-proxy pod ([#52003](https://github.com/kubernetes/kubernetes/pull/52003), [@vfreex](https://github.com/vfreex))
* kube-apiserver: `--ssh-user` and `--ssh-keyfile` are now deprecated and will be removed in a future release. Users of SSH tunnel functionality used in Google Container Engine for the Master -> Cluster communication should plan to transition to alternate methods for bridging master and node networks. ([#54433](https://github.com/kubernetes/kubernetes/pull/54433), [@dims](https://github.com/dims))
* Fix hyperkube kubelet --experimental-dockershim ([#54508](https://github.com/kubernetes/kubernetes/pull/54508), [@ivan4th](https://github.com/ivan4th))
* Fix clustered datastore name to be absolute. ([#54438](https://github.com/kubernetes/kubernetes/pull/54438), [@pshahzeb](https://github.com/pshahzeb))
* Fix for service controller so that it won't retry on doNotRetry service update failure. ([#54184](https://github.com/kubernetes/kubernetes/pull/54184), [@MrHohn](https://github.com/MrHohn))
* Add support for RBAC support to Kubernetes via Juju ([#53820](https://github.com/kubernetes/kubernetes/pull/53820), [@ktsakalozos](https://github.com/ktsakalozos))
* RBD Persistent Volume Sources can now reference User's Secret in namespaces other than the namespace of the bound Persistent Volume Claim ([#54302](https://github.com/kubernetes/kubernetes/pull/54302), [@sbezverk](https://github.com/sbezverk))
* Apiserver proxy rewrites URL when service returns absolute path with request's host. ([#52556](https://github.com/kubernetes/kubernetes/pull/52556), [@roycaihw](https://github.com/roycaihw))
* Logging cleanups ([#54443](https://github.com/kubernetes/kubernetes/pull/54443), [@bowei](https://github.com/bowei))
        * Updates kube-dns to use client-go 3
        * Updates containers to use alpine as the base image on all platforms
        * Adds support for IPv6
* add `--raw` to `kubectl create` to POST using the normal transport ([#54245](https://github.com/kubernetes/kubernetes/pull/54245), [@deads2k](https://github.com/deads2k))
* Remove the --network-plugin-dir flag. ([#53564](https://github.com/kubernetes/kubernetes/pull/53564), [@supereagle](https://github.com/supereagle))
* Introduces a polymorphic scale client, allowing HorizontalPodAutoscalers to properly function on scalable resources in any API group. ([#53743](https://github.com/kubernetes/kubernetes/pull/53743), [@DirectXMan12](https://github.com/DirectXMan12))
* Add PodDisruptionBudget to scheduler cache. ([#53914](https://github.com/kubernetes/kubernetes/pull/53914), [@bsalamat](https://github.com/bsalamat))
* - API machinery's httpstream/spdy calls now support CIDR notation for NO_PROXY ([#54413](https://github.com/kubernetes/kubernetes/pull/54413), [@kad](https://github.com/kad))
* Added option lb-provider to OpenStack cloud provider config ([#54176](https://github.com/kubernetes/kubernetes/pull/54176), [@gonzolino](https://github.com/gonzolino))
* Allow for configuring etcd hostname in the manifest ([#54403](https://github.com/kubernetes/kubernetes/pull/54403), [@wojtek-t](https://github.com/wojtek-t))
* - kubeadm  will warn users if access to IP ranges for Pods or Services will be done via HTTP proxy. ([#52792](https://github.com/kubernetes/kubernetes/pull/52792), [@kad](https://github.com/kad))
* Resolves forbidden error when accessing replicasets and daemonsets via the apps API group ([#54309](https://github.com/kubernetes/kubernetes/pull/54309), [@liggitt](https://github.com/liggitt))
* Cluster Autoscaler 1.0.1 ([#54298](https://github.com/kubernetes/kubernetes/pull/54298), [@mwielgus](https://github.com/mwielgus))
* secret data containing Docker registry auth objects is now generated using the config.json format ([#53916](https://github.com/kubernetes/kubernetes/pull/53916), [@juanvallejo](https://github.com/juanvallejo))
* Added support for SAN entries in the master node certificate via juju kubernetes-master config. ([#54234](https://github.com/kubernetes/kubernetes/pull/54234), [@hyperbolic2346](https://github.com/hyperbolic2346))
* support imagePullSecrets and imagePullPolicy in kubefed init ([#50740](https://github.com/kubernetes/kubernetes/pull/50740), [@dixudx](https://github.com/dixudx))
* update gRPC to v1.6.0 to pick up data race fix grpc/grpc-go#1316  ([#53128](https://github.com/kubernetes/kubernetes/pull/53128), [@dixudx](https://github.com/dixudx))
* admission webhook registrations without a specific failure policy default to failing closed. ([#54162](https://github.com/kubernetes/kubernetes/pull/54162), [@deads2k](https://github.com/deads2k))
* Device plugin Alpha API no longer supports returning artifacts per device as part of AllocateResponse. ([#53031](https://github.com/kubernetes/kubernetes/pull/53031), [@vishh](https://github.com/vishh))
* admission webhook registration now allows URL paths ([#54145](https://github.com/kubernetes/kubernetes/pull/54145), [@deads2k](https://github.com/deads2k))
* The Kubelet's --enable-custom-metrics flag is now marked deprecated. ([#54154](https://github.com/kubernetes/kubernetes/pull/54154), [@mtaufen](https://github.com/mtaufen))
* Use multi-arch busybox image for e2e ([#54034](https://github.com/kubernetes/kubernetes/pull/54034), [@dixudx](https://github.com/dixudx))
* sample-controller: add example CRD controller ([#52753](https://github.com/kubernetes/kubernetes/pull/52753), [@munnerz](https://github.com/munnerz))
* RBAC PolicyRules now allow resource=`*/<subresource>` to cover `any-resource/<subresource>`.   For example, `*/scale` covers `replicationcontroller/scale`. ([#53722](https://github.com/kubernetes/kubernetes/pull/53722), [@deads2k](https://github.com/deads2k))
* Upgrade to go1.9 ([#51375](https://github.com/kubernetes/kubernetes/pull/51375), [@cblecker](https://github.com/cblecker))
* Webhook always retries connection reset error. ([#53947](https://github.com/kubernetes/kubernetes/pull/53947), [@crassirostris](https://github.com/crassirostris))
* fix PV Recycle failed on non-amd64 platform ([#53958](https://github.com/kubernetes/kubernetes/pull/53958), [@dixudx](https://github.com/dixudx))
* Verbose option is added to each status function in CRI. Container runtime could return extra information in status response for debugging. ([#53965](https://github.com/kubernetes/kubernetes/pull/53965), [@Random-Liu](https://github.com/Random-Liu))
* Fixed log fallback termination messages when using docker with journald log driver ([#52503](https://github.com/kubernetes/kubernetes/pull/52503), [@joelsmith](https://github.com/joelsmith))
* falls back to parse Docker runtime version as generic if not semver ([#54040](https://github.com/kubernetes/kubernetes/pull/54040), [@dixudx](https://github.com/dixudx))
* kubelet: prevent removal of default labels from Node API objects on startup ([#54073](https://github.com/kubernetes/kubernetes/pull/54073), [@liggitt](https://github.com/liggitt))
* Change scheduler to skip pod with updates only on pod annotations ([#54008](https://github.com/kubernetes/kubernetes/pull/54008), [@yguo0905](https://github.com/yguo0905))
* PodSecurityPolicy: when multiple policies allow a submitted pod, priority is given to ones which do not require any fields in the pod spec to be defaulted. If the pod must be defaulted, the first policy (ordered by name) that allows the pod is used. ([#52849](https://github.com/kubernetes/kubernetes/pull/52849), [@liggitt](https://github.com/liggitt))
* Control HPA tolerance through the `horizontal-pod-autoscaler-tolerance` flag. ([#52275](https://github.com/kubernetes/kubernetes/pull/52275), [@mattjmcnaughton](https://github.com/mattjmcnaughton))
* bump CNI to v0.6.0 ([#51250](https://github.com/kubernetes/kubernetes/pull/51250), [@dixudx](https://github.com/dixudx))
* Improve resilience by annotating kube-dns addon with podAntiAffinity to prefer scheduling on different nodes. ([#52193](https://github.com/kubernetes/kubernetes/pull/52193), [@StevenACoffman](https://github.com/StevenACoffman))
* Azure cloudprovider: Fix controller manager crash issue on a manually created k8s cluster. ([#53694](https://github.com/kubernetes/kubernetes/pull/53694), [@andyzhangx](https://github.com/andyzhangx))
* Enable Priority admission control in kubeadm.  ([#53175](https://github.com/kubernetes/kubernetes/pull/53175), [@andrewsykim](https://github.com/andrewsykim))
* Add --no-negcache flag to kube-dns to prevent caching of NXDOMAIN responses. ([#53604](https://github.com/kubernetes/kubernetes/pull/53604), [@cblecker](https://github.com/cblecker))
* kubelet provides more specific events when unable to sync pod ([#53857](https://github.com/kubernetes/kubernetes/pull/53857), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Kubelet evictions take pod priority into account ([#53542](https://github.com/kubernetes/kubernetes/pull/53542), [@dashpole](https://github.com/dashpole))
* Adds a new controller which automatically cleans up Certificate Signing Requests that are ([#51840](https://github.com/kubernetes/kubernetes/pull/51840), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * Approved and Issued, or Denied.
* Optimize random string generator to avoid multiple locks & use bit-masking ([#53720](https://github.com/kubernetes/kubernetes/pull/53720), [@shyamjvs](https://github.com/shyamjvs))
* update cluster printer to enable --show-labels ([#53771](https://github.com/kubernetes/kubernetes/pull/53771), [@dixudx](https://github.com/dixudx))
* add RequestReceivedTimestamp and StageTimestamp to audit event ([#52981](https://github.com/kubernetes/kubernetes/pull/52981), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Deprecation: The flag `etcd-quorum-read ` of kube-apiserver is deprecated and the ability to switch off quorum read will be removed in a future release. ([#53795](https://github.com/kubernetes/kubernetes/pull/53795), [@xiangpengzhao](https://github.com/xiangpengzhao))
* Use separate client for leader election in scheduler to avoid starving leader election by regular scheduler operations. ([#53793](https://github.com/kubernetes/kubernetes/pull/53793), [@wojtek-t](https://github.com/wojtek-t))
* Support autoprobing node-security-group for openstack cloud provider, Support multiple Security Groups for cluster's nodes. ([#50836](https://github.com/kubernetes/kubernetes/pull/50836), [@FengyunPan](https://github.com/FengyunPan))
* fix a bug where disk pressure could trigger prematurely when using overlay2 ([#53684](https://github.com/kubernetes/kubernetes/pull/53684), [@dashpole](https://github.com/dashpole))
* "kubectl cp" updated to honor destination names  ([#51215](https://github.com/kubernetes/kubernetes/pull/51215), [@juanvallejo](https://github.com/juanvallejo))
* kubeadm: Strip bootstrap tokens from the `kubeadm-config` ConfigMap ([#53559](https://github.com/kubernetes/kubernetes/pull/53559), [@fabriziopandini](https://github.com/fabriziopandini))
* Skip podpreset test if the alpha feature settings/v1alpha1 is disabled ([#53080](https://github.com/kubernetes/kubernetes/pull/53080), [@jennybuckley](https://github.com/jennybuckley))
* Log when node is successfully initialized by Cloud Controller Manager ([#53517](https://github.com/kubernetes/kubernetes/pull/53517), [@andrewsykim](https://github.com/andrewsykim))
* apiserver: --etcd-quorum-read now defaults to true, to ensure correct operation with HA etcd clusters ([#53717](https://github.com/kubernetes/kubernetes/pull/53717), [@liggitt](https://github.com/liggitt))
* The Kubelet's feature gates are now specified as a map when provided via a JSON or YAML KubeletConfiguration, rather than as a string of key-value pairs. ([#53025](https://github.com/kubernetes/kubernetes/pull/53025), [@mtaufen](https://github.com/mtaufen))
* Address a bug which allowed the horizontal pod autoscaler to allocate `desiredReplicas` > `maxReplicas` in certain instances. ([#53690](https://github.com/kubernetes/kubernetes/pull/53690), [@mattjmcnaughton](https://github.com/mattjmcnaughton))
* Horizontal pod autoscaler uses REST clients through the kube-aggregator instead of the legacy client through the API server proxy. ([#53205](https://github.com/kubernetes/kubernetes/pull/53205), [@kawych](https://github.com/kawych))
* Fix to prevent downward api change break on older versions ([#53673](https://github.com/kubernetes/kubernetes/pull/53673), [@timothysc](https://github.com/timothysc))
* API chunking via the `limit` and `continue` request parameters is promoted to beta in this release.  Client libraries using the Informer or ListWatch types will automatically opt in to chunking. ([#52949](https://github.com/kubernetes/kubernetes/pull/52949), [@smarterclayton](https://github.com/smarterclayton))
* GCE: Bump GLBC version to [0.9.7](https://github.com/kubernetes/ingress/releases/tag/0.9.7). ([#53625](https://github.com/kubernetes/kubernetes/pull/53625), [@nikhiljindal](https://github.com/nikhiljindal))
* kubelet's `--cloud-provider` flag no longer defaults to "auto-detect".  If you want cloud-provider support in kubelet, you must set a specific cloud-provider explicitly. ([#53573](https://github.com/kubernetes/kubernetes/pull/53573), [@dims](https://github.com/dims))
* Ignore extended resources that are not registered with kubelet during container resource allocation. ([#53547](https://github.com/kubernetes/kubernetes/pull/53547), [@jiayingz](https://github.com/jiayingz))
* kubectl top pod and node should sort by namespace / name so that results don't jump around. ([#53560](https://github.com/kubernetes/kubernetes/pull/53560), [@dixudx](https://github.com/dixudx))
* Added --dry-run option to `kubectl drain` ([#52440](https://github.com/kubernetes/kubernetes/pull/52440), [@juanvallejo](https://github.com/juanvallejo))
* Fix a bug that prevents client-go metrics from being registered in prometheus in multiple components. ([#53434](https://github.com/kubernetes/kubernetes/pull/53434), [@crassirostris](https://github.com/crassirostris))
* Adjust batching audit webhook default parameters: increase queue size, batch size, and initial backoff. Add throttling to the batching audit webhook. Default rate limit is 10 QPS. ([#53417](https://github.com/kubernetes/kubernetes/pull/53417), [@crassirostris](https://github.com/crassirostris))
* Added integration test for TaintNodeByCondition. ([#53184](https://github.com/kubernetes/kubernetes/pull/53184), [@k82cn](https://github.com/k82cn))
* Add API version apps/v1, and bump DaemonSet to apps/v1 ([#53278](https://github.com/kubernetes/kubernetes/pull/53278), [@janetkuo](https://github.com/janetkuo))
* Change `kubeadm create token` to default to the group that almost everyone will want to use.  The group is system:bootstrappers:kubeadm:default-node-token and is the group that kubeadm sets up, via an RBAC binding, for auto-approval (system:certificates.k8s.io:certificatesigningrequests:nodeclient). ([#53512](https://github.com/kubernetes/kubernetes/pull/53512), [@jbeda](https://github.com/jbeda))
* Using OpenStack service catalog to do version detection ([#53115](https://github.com/kubernetes/kubernetes/pull/53115), [@FengyunPan](https://github.com/FengyunPan))
* Fix metrics API group name in audit configuration ([#53493](https://github.com/kubernetes/kubernetes/pull/53493), [@piosz](https://github.com/piosz))
* GCE: Fixes ILB sync on legacy networks and auto networks with unique subnet names ([#53410](https://github.com/kubernetes/kubernetes/pull/53410), [@nicksardo](https://github.com/nicksardo))
* outputs `<none>` for columns specified by `-o custom-columns` but not found in object ([#51750](https://github.com/kubernetes/kubernetes/pull/51750), [@jianhuiz](https://github.com/jianhuiz))
* Metrics were added to network plugin to report latency of CNI operations ([#53446](https://github.com/kubernetes/kubernetes/pull/53446), [@sjenning](https://github.com/sjenning))
* GCE: Fix issue deleting internal load balancers when the firewall resource may not exist. ([#53450](https://github.com/kubernetes/kubernetes/pull/53450), [@nicksardo](https://github.com/nicksardo))
* Custom resources served through CustomResourceDefinition now support field selectors for `metadata.name` and `metadata.namespace`. ([#53345](https://github.com/kubernetes/kubernetes/pull/53345), [@ncdc](https://github.com/ncdc))
* Add generate-groups.sh and generate-internal-groups.sh to k8s.io/code-generator to easily run generators against CRD or User API Server types. ([#52186](https://github.com/kubernetes/kubernetes/pull/52186), [@sttts](https://github.com/sttts))
* kubelet `--cert-dir` now defaults to `/var/lib/kubelet/pki`, in order to ensure bootstrapped and rotated certificates persist beyond a reboot. resolves an issue in kubeadm with false-positive `/var/lib/kubelet is not empty` message during pre-flight checks ([#53317](https://github.com/kubernetes/kubernetes/pull/53317), [@liggitt](https://github.com/liggitt))
* Fix multi-attach error spam in logs and events ([#53401](https://github.com/kubernetes/kubernetes/pull/53401), [@gnufied](https://github.com/gnufied))
* Use `not-ready` to replace `notReady` in node condition taint keys. ([#51266](https://github.com/kubernetes/kubernetes/pull/51266), [@resouer](https://github.com/resouer))
* Support completion for --clusterrole of kubectl create clusterrolebinding ([#48267](https://github.com/kubernetes/kubernetes/pull/48267), [@superbrothers](https://github.com/superbrothers))
* Don't remove extended resource capacities that are not registered with kubelet from node status. ([#53353](https://github.com/kubernetes/kubernetes/pull/53353), [@jiayingz](https://github.com/jiayingz))
* Kubectl: Remove swagger 1.2 validation. Also removes options `--use-openapi` and `--schema-cache-dir` as these are no longer needed. ([#53232](https://github.com/kubernetes/kubernetes/pull/53232), [@apelisse](https://github.com/apelisse))
* `kubectl explain` now uses openapi rather than swagger 1.2. ([#53228](https://github.com/kubernetes/kubernetes/pull/53228), [@apelisse](https://github.com/apelisse))
* Fixes a performance issue ([#51899](https://github.com/kubernetes/kubernetes/pull/51899)) identified in large-scale clusters when deleting thousands of pods simultaneously across hundreds of nodes, by actively removing containers of deleted pods, rather than waiting for periodic garbage collection and batching resulting pod API deletion requests. ([#53233](https://github.com/kubernetes/kubernetes/pull/53233), [@dashpole](https://github.com/dashpole))
* Improve explanation of ReplicaSet ([#53403](https://github.com/kubernetes/kubernetes/pull/53403), [@rcorre](https://github.com/rcorre))
* avoid newline "
" in the error to break log msg to 2 lines ([#49826](https://github.com/kubernetes/kubernetes/pull/49826), [@dixudx](https://github.com/dixudx))
* don't recreate a mirror pod for static pod when node gets deleted ([#48339](https://github.com/kubernetes/kubernetes/pull/48339), [@dixudx](https://github.com/dixudx))
* Fix permissions for Metrics Server. ([#53330](https://github.com/kubernetes/kubernetes/pull/53330), [@kawych](https://github.com/kawych))
* default fail-swap-on to false for kubelet on kubernetes-worker charm ([#53386](https://github.com/kubernetes/kubernetes/pull/53386), [@wwwtyro](https://github.com/wwwtyro))
* Add --etcd-compaction-interval to apiserver for controlling request of compaction to etcd3 from apiserver. ([#51765](https://github.com/kubernetes/kubernetes/pull/51765), [@mitake](https://github.com/mitake))
* Apply algorithm in scheduler by feature gates. ([#52723](https://github.com/kubernetes/kubernetes/pull/52723), [@k82cn](https://github.com/k82cn))
* etcd: update version to 3.1.10 ([#49393](https://github.com/kubernetes/kubernetes/pull/49393), [@hongchaodeng](https://github.com/hongchaodeng))
* support nodeSelector in kubefed init ([#50749](https://github.com/kubernetes/kubernetes/pull/50749), [@dixudx](https://github.com/dixudx))
* Upgrade fluentd-elasticsearch addon to Elasticsearch/Kibana 5.6.2 ([#53307](https://github.com/kubernetes/kubernetes/pull/53307), [@aknuds1](https://github.com/aknuds1))
* enable to specific unconfined AppArmor profile ([#52395](https://github.com/kubernetes/kubernetes/pull/52395), [@dixudx](https://github.com/dixudx))
* Update Influxdb image to latest version. ([#53319](https://github.com/kubernetes/kubernetes/pull/53319), [@kairen](https://github.com/kairen))
    * Update Grafana image to latest version.
    * Change influxdb-grafana-controller resource to Deployment.
* Only do UpdateContainerResources when cpuset is set  ([#53122](https://github.com/kubernetes/kubernetes/pull/53122), [@resouer](https://github.com/resouer))
* Fixes an issue with RBAC reconciliation that could cause duplicated subjects in some bootstrapped rolebindings on each restart of the API server. ([#53239](https://github.com/kubernetes/kubernetes/pull/53239), [@enj](https://github.com/enj))
* gce: remove compute-rw, see what breaks ([#53266](https://github.com/kubernetes/kubernetes/pull/53266), [@mikedanese](https://github.com/mikedanese))
* Fix the bug that query Kubelet's stats summary with CRI stats enabled results in error. ([#53107](https://github.com/kubernetes/kubernetes/pull/53107), [@Random-Liu](https://github.com/Random-Liu))
* kubeadm allows the kubelets in the cluster to automatically renew their client certificates ([#53252](https://github.com/kubernetes/kubernetes/pull/53252), [@kad](https://github.com/kad))
* Fixes an issue with `kubectl set` commands encountering conversion errors for ReplicaSet and DaemonSet objects ([#53158](https://github.com/kubernetes/kubernetes/pull/53158), [@liggitt](https://github.com/liggitt))
* RBAC: The default `admin` and `edit` roles now include read/write permissions and the `view` role includes read permissions on `poddisruptionbudget.policy` resources. ([#52654](https://github.com/kubernetes/kubernetes/pull/52654), [@liggitt](https://github.com/liggitt))
* Change ImageGCManage to consume ImageFS stats from StatsProvider ([#53094](https://github.com/kubernetes/kubernetes/pull/53094), [@yguo0905](https://github.com/yguo0905))
* BugFix: Exited containers are not Garbage Collected by the kubelet while the pod is running ([#53167](https://github.com/kubernetes/kubernetes/pull/53167), [@dashpole](https://github.com/dashpole))
* - Improved generation of deb and rpm packages in bazel build ([#53163](https://github.com/kubernetes/kubernetes/pull/53163), [@kad](https://github.com/kad))
* Add a label which prevents a node from being added to a cloud load balancer ([#53146](https://github.com/kubernetes/kubernetes/pull/53146), [@brendandburns](https://github.com/brendandburns))
* Fixes an issue pulling pod specs referencing unqualified images from docker.io on centos/fedora/rhel ([#53161](https://github.com/kubernetes/kubernetes/pull/53161), [@dims](https://github.com/dims))
* Update kube-dns to 1.14.5 ([#53153](https://github.com/kubernetes/kubernetes/pull/53153), [@bowei](https://github.com/bowei))
* - kubeadm init can now deploy exact build from CI area by specifying ID with "ci/" prefix. Example: "ci/v1.9.0-alpha.1.123+01234567889" ([#53043](https://github.com/kubernetes/kubernetes/pull/53043), [@kad](https://github.com/kad))
    * - kubeadm upgrade apply supports all standard ways of specifying version via labels. Examples: stable-1.8, latest-1.8, ci/latest-1.9 and similar.
* - kubeadm 1.9 will detect and fail init or join pre-flight checks if kubelet is lower than 1.8.0-alpha ([#52913](https://github.com/kubernetes/kubernetes/pull/52913), [@kad](https://github.com/kad))
* s390x ingress controller support ([#52663](https://github.com/kubernetes/kubernetes/pull/52663), [@wwwtyro](https://github.com/wwwtyro))
* NONE ([#50532](https://github.com/kubernetes/kubernetes/pull/50532), [@steveperry-53](https://github.com/steveperry-53))
* CRI: Add stdout/stderr fields to Exec and Attach requests. ([#52686](https://github.com/kubernetes/kubernetes/pull/52686), [@yujuhong](https://github.com/yujuhong))
* NONE ([#53001](https://github.com/kubernetes/kubernetes/pull/53001), [@ericchiang](https://github.com/ericchiang))
* Cluster Autoscaler 1.0.0 ([#53005](https://github.com/kubernetes/kubernetes/pull/53005), [@mwielgus](https://github.com/mwielgus))
* Remove the --docker-exec-handler flag. Only native exec handler is supported. ([#52287](https://github.com/kubernetes/kubernetes/pull/52287), [@yujuhong](https://github.com/yujuhong))
* The Rackspace cloud provider has been removed after a long deprecation period. It was deprecated because it duplicates a lot of the OpenStack logic and can no longer be maintained. Please use the OpenStack cloud provider instead. ([#52855](https://github.com/kubernetes/kubernetes/pull/52855), [@NickrenREN](https://github.com/NickrenREN))
* Fixes an initializer bug where update requests which had an empty pending initializers list were erroneously rejected. ([#52558](https://github.com/kubernetes/kubernetes/pull/52558), [@jennybuckley](https://github.com/jennybuckley))
* BulkVerifyVolumes() implementation for vSphere ([#52131](https://github.com/kubernetes/kubernetes/pull/52131), [@BaluDontu](https://github.com/BaluDontu))
* added --list option to the `kubectl label` command ([#51971](https://github.com/kubernetes/kubernetes/pull/51971), [@juanvallejo](https://github.com/juanvallejo))
* Removing `--prom-push-gateway` flag from e2e tests ([#52485](https://github.com/kubernetes/kubernetes/pull/52485), [@nielsole](https://github.com/nielsole))
* If a container does not create a file at the `terminationMessagePath`, no message should be output about being unable to find the file. ([#52567](https://github.com/kubernetes/kubernetes/pull/52567), [@smarterclayton](https://github.com/smarterclayton))
* Support German cloud for azure disk mount feature ([#50673](https://github.com/kubernetes/kubernetes/pull/50673), [@clement-buchart](https://github.com/clement-buchart))
* Add s390x to juju kubernetes ([#52537](https://github.com/kubernetes/kubernetes/pull/52537), [@ktsakalozos](https://github.com/ktsakalozos))
* Fix kubernetes charms not restarting services properly after host reboot on LXD ([#52445](https://github.com/kubernetes/kubernetes/pull/52445), [@Cynerva](https://github.com/Cynerva))
* Add monitoring of Windows Server containers metrics in the kubelet via the stats/summary endpoint. ([#50396](https://github.com/kubernetes/kubernetes/pull/50396), [@bobbypage](https://github.com/bobbypage))
* Restores redirect behavior for proxy subresources ([#52933](https://github.com/kubernetes/kubernetes/pull/52933), [@liggitt](https://github.com/liggitt))
* A new service annotation has been added for services of type LoadBalancer on Azure,  ([#51757](https://github.com/kubernetes/kubernetes/pull/51757), [@itowlson](https://github.com/itowlson))
    * to specify the subnet on which the service's front end IP should be provisioned. The 
    * annotation is service.beta.kubernetes.io/azure-load-balancer-internal-subnet and its 
    * value is the subnet name (not the subnet ARM ID).  If omitted, the default is the 
    * master subnet.  It is ignored if the service is not on Azure, if the type is not 
    * LoadBalancer, or if the load balancer is not internal.
* Adds a command-line argument to kube-apiserver called ([#51698](https://github.com/kubernetes/kubernetes/pull/51698), [@rphillips](https://github.com/rphillips))
    * --alpha-endpoint-reconciler-type=(master-count, lease, none) (default
    * "master-count"). The original reconciler is 'master-count'. The 'lease'
    * reconciler uses the storageapi and a TTL to keep alive an endpoint within the
    * `kube-apiserver-endpoint` storage namespace. The 'none' reconciler is a noop
    * reconciler that does not do anything. This is useful for self-hosted
    * environments.
* Improved Italian translation for kubectl ([#51463](https://github.com/kubernetes/kubernetes/pull/51463), [@lucab85](https://github.com/lucab85))
* Add a metric to the kubelet to monitor remaining lifetime of the certificate that ([#51031](https://github.com/kubernetes/kubernetes/pull/51031), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * authenticates the kubelet to the API server.
* change AddEventHandlerWithResyncPeriod to AddEventHandler in factory.go ([#51582](https://github.com/kubernetes/kubernetes/pull/51582), [@jiulongzaitian](https://github.com/jiulongzaitian))
* Validate that cronjob names are 52 characters or less ([#52733](https://github.com/kubernetes/kubernetes/pull/52733), [@julia-stripe](https://github.com/julia-stripe))
* add readme file of ipvs ([#51937](https://github.com/kubernetes/kubernetes/pull/51937), [@Lion-Wei](https://github.com/Lion-Wei))



# v1.9.0-alpha.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.9.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes.tar.gz) | `e2dc3eebf79368c783b64f5b6642a287cc2fd777547d99f240a35cce1f620ffc`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-src.tar.gz) | `ca8659187047f2d38a7c0ee313189c19ec35646c6ebaa8f59f2f098eca33dca0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `51e0df7e6611ff4a9b3759b05e65c80555317bff03282ef39a9b53b27cdeff42`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `c6c57cc92cc456a644c0965a6aa2bd260125807b450d69376e0edb6c98aaf4d7`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `399c8cb448d76accb71edcb00bee474f172d416c8c4f5253994e4e2d71e0dece`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `fde75d7267592b34609299a93ee7e54b26a948e6f9a1f64ced666c0aae4455aa`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `b38810cf87735efb0af027b7c77e4e8c8f5821f235cf33ae9eee346e6d1a0b84`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `a36427c2f2b81d42702a12392070f7dd3635b651bb04ae925d0bdf3ec50f83aa`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `9dee0f636eef09bfec557a50e4f8f4b69e0588bbd0b77f6da50cc155e1679880`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `4a6246d5de5c3957ed41b8943fa03e74fb646595346f7c72beaf7b030fe6872e`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `1ee384f4bb02e614c86bf84cdfdc42faffa659aaba4a1c759ec26f03eb438149`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `e70d8935abefea0307780e899238bb10ec27c8f0d77702cf25de230b6abf7fb4`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `7fff06370c4f37e1fe789cc160fce0c93535991f63d7fe7d001378f17027d9d8`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `65cd60512ea0bf508aa65f8d22a6f3094db394f00b3cd6bd63fe02b795514ab2`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `0ecb341a047f1a9dface197f11f05f15853570cfb474c82538c7d61b40bd53ae`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `cea9eed4c24e7f29994ecc12674bff69d108692d3c9be3e8bd939b3c4f281892`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `4d50799e5989de6d9ec316d2051497a3617b635e89fa44e01e64fed544d96e07`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `e956b9c1e5b47f800953ad0f82fae23774a2f43079dc02d98a90d5bfdca0bad6`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `ede6a85db555dd84e8d7180bdd58712933c38567ab6c97a80d0845be2974d968`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `4ac6a1784fa1e20be8a4e7fa0ff8b4defc725e6c058ff97068bf7bfa6a11c77d`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `0d9c8c7e0892d7b678f3b4b7736087da91cb40c5f169e4302e9f4637c516207a`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `2fdde192a84410c784e5d1e813985e9a19ce62e3d9bb2215481cbce9286329da`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.9.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `543110cc69b57471f3824d96cbd16b003ac2cddaa19ca4bdefced0af61fd24f2`

## Changelog since v1.8.0-alpha.3

### Action Required

* New GCE or GKE clusters created with `cluster/kube-up.sh` will not enable the legacy ABAC authorizer by default. If you would like to enable the legacy ABAC authorizer, export ENABLE_LEGACY_ABAC=true before running `cluster/kube-up.sh`. ([#51367](https://github.com/kubernetes/kubernetes/pull/51367), [@cjcullen](https://github.com/cjcullen))
* The OwnerReferencesPermissionEnforcement admission plugin now requires `update` permission on the `finalizers` subresource of the referenced owner in order to set `blockOwnerDeletion` on an owner reference. ([#49133](https://github.com/kubernetes/kubernetes/pull/49133), [@deads2k](https://github.com/deads2k))
* The deprecated alpha and beta initContainer annotations are no longer supported. Init containers must be specified using the initContainers field in the pod spec. ([#51816](https://github.com/kubernetes/kubernetes/pull/51816), [@liggitt](https://github.com/liggitt))
* Action required: validation rule on metadata.initializers.pending[x].name is tightened. The initializer name needs to contain at least three segments separated by dots. If you create objects with pending initializers, (i.e., not relying on apiserver adding pending initializers according to initializerconfiguration), you need to update the initializer name in existing objects and in configuration files to comply to the new validation rule. ([#51283](https://github.com/kubernetes/kubernetes/pull/51283), [@caesarxuchao](https://github.com/caesarxuchao))
* Audit policy supports matching subresources and resource names, but the top level resource no longer matches the subresouce. For example "pods" no longer matches requests to the logs subresource of pods. Use "pods/logs" to match subresources. ([#48836](https://github.com/kubernetes/kubernetes/pull/48836), [@ericchiang](https://github.com/ericchiang))
* Protobuf serialization does not distinguish between `[]` and `null`. ([#45294](https://github.com/kubernetes/kubernetes/pull/45294), [@liggitt](https://github.com/liggitt))
    * API fields previously capable of storing and returning either `[]` and `null` via JSON API requests (for example, the Endpoints `subsets` field) can now store only `null` when created using the protobuf content-type or stored in etcd using protobuf serialization (the default in 1.6+). JSON API clients should tolerate `null` values for such fields, and treat `null` and `[]` as equivalent in meaning unless specifically documented otherwise for a particular field.

### Other notable changes

* PersistentVolumeLabel admission controller is now deprecated. ([#52618](https://github.com/kubernetes/kubernetes/pull/52618), [@dims](https://github.com/dims))
* Mark the LBaaS v1 of OpenStack cloud provider deprecated. ([#52821](https://github.com/kubernetes/kubernetes/pull/52821), [@FengyunPan](https://github.com/FengyunPan))
* NONE ([#52819](https://github.com/kubernetes/kubernetes/pull/52819), [@verult](https://github.com/verult))
* Mark image as deliberately optional in v1 Container struct.  Many objects in the Kubernetes API inherit the container struct and only Pods require the field to be set. ([#48406](https://github.com/kubernetes/kubernetes/pull/48406), [@gyliu513](https://github.com/gyliu513))
* [fluentd-gcp addon] Update Stackdriver plugin to version 0.6.7 ([#52565](https://github.com/kubernetes/kubernetes/pull/52565), [@crassirostris](https://github.com/crassirostris))
* Remove duplicate proto errors in kubelet. ([#52132](https://github.com/kubernetes/kubernetes/pull/52132), [@adityadani](https://github.com/adityadani))
* [fluentd-gcp addon] Remove audit logs from the fluentd configuration ([#52777](https://github.com/kubernetes/kubernetes/pull/52777), [@crassirostris](https://github.com/crassirostris))
* Set defaults for successfulJobsHistoryLimit (3) and failedJobsHistoryLimit (1) in batch/v1beta1.CronJobs ([#52533](https://github.com/kubernetes/kubernetes/pull/52533), [@soltysh](https://github.com/soltysh))
* Fix: update system spec to support Docker 17.03 ([#52666](https://github.com/kubernetes/kubernetes/pull/52666), [@yguo0905](https://github.com/yguo0905))
* Fix panic in ControllerManager on GCE when it has a problem with creating external loadbalancer healthcheck ([#52646](https://github.com/kubernetes/kubernetes/pull/52646), [@gmarek](https://github.com/gmarek))
* PSP: add support for using `*` as a value in `allowedCapabilities` to allow to request any capabilities ([#51337](https://github.com/kubernetes/kubernetes/pull/51337), [@php-coder](https://github.com/php-coder))
* [fluentd-gcp addon] By default ingest apiserver audit logs written to file in JSON format. ([#52541](https://github.com/kubernetes/kubernetes/pull/52541), [@crassirostris](https://github.com/crassirostris))
* The autoscaling/v2beta1 API group is now enabled by default. ([#52549](https://github.com/kubernetes/kubernetes/pull/52549), [@DirectXMan12](https://github.com/DirectXMan12))
* Add CLUSTER_SIGNING_DURATION environment variable to cluster ([#52497](https://github.com/kubernetes/kubernetes/pull/52497), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * configuration scripts to allow configuration of signing duration of
    * certificates issued via the Certificate Signing Request API.
* Introduce policy to allow the HPA to consume the metrics.k8s.io and custom.metrics.k8s.io API groups. ([#52572](https://github.com/kubernetes/kubernetes/pull/52572), [@DirectXMan12](https://github.com/DirectXMan12))
* kubelet to master communication when doing node status updates now has a timeout to prevent indefinite hangs ([#52176](https://github.com/kubernetes/kubernetes/pull/52176), [@liggitt](https://github.com/liggitt))
* Introduced Metrics Server in version v0.2.0. For more details see https://github.com/kubernetes-incubator/metrics-server/releases/tag/v0.2.0. ([#52548](https://github.com/kubernetes/kubernetes/pull/52548), [@piosz](https://github.com/piosz))
* Adds ROTATE_CERTIFICATES environment variable to kube-up.sh script for GCE ([#52115](https://github.com/kubernetes/kubernetes/pull/52115), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * clusters. When that var is set to true, the command line flag enabling kubelet
    * client certificate rotation will be added to the kubelet command line.
* Make sure that resources being updated are handled correctly by Quota system ([#52452](https://github.com/kubernetes/kubernetes/pull/52452), [@gnufied](https://github.com/gnufied))
* WATCHLIST calls are now reported as WATCH verbs in prometheus for the apiserver_request_* series.  A new "scope" label is added to all apiserver_request_* values that is either 'cluster', 'resource', or 'namespace' depending on which level the query is performed at. ([#52237](https://github.com/kubernetes/kubernetes/pull/52237), [@smarterclayton](https://github.com/smarterclayton))
* Fixed the webhook admission plugin so that it works even if the apiserver and the nodes are in two networks (e.g., in GKE). ([#50476](https://github.com/kubernetes/kubernetes/pull/50476), [@caesarxuchao](https://github.com/caesarxuchao))
    * Fixed the webhook admission plugin so that webhook author could use the DNS name of the service as the CommonName when generating the server cert for the webhook.
    * Action required:
    * Anyone who generated server cert for admission webhooks need to regenerate the cert. Previously, when generating server cert for the admission webhook, the CN value doesn't matter. Now you must set it to the DNS name of the webhook service, i.e., `<service.Name>.<service.Namespace>.svc`.
* Ignore pods marked for deletion that exceed their grace period in ResourceQuota ([#46542](https://github.com/kubernetes/kubernetes/pull/46542), [@derekwaynecarr](https://github.com/derekwaynecarr))
* custom resources that use unconventional pluralization now work properly with kubectl and garbage collection ([#50012](https://github.com/kubernetes/kubernetes/pull/50012), [@deads2k](https://github.com/deads2k))
* [fluentd-gcp addon] Fluentd will trim lines exceeding 100KB instead of dropping them. ([#52289](https://github.com/kubernetes/kubernetes/pull/52289), [@crassirostris](https://github.com/crassirostris))
* dockershim: check the error when syncing the checkpoint. ([#52125](https://github.com/kubernetes/kubernetes/pull/52125), [@yujuhong](https://github.com/yujuhong))
* By default, clusters on GCE no longer sends RequestReceived audit event, if advanced audit is configured. ([#52343](https://github.com/kubernetes/kubernetes/pull/52343), [@crassirostris](https://github.com/crassirostris))
* [BugFix] Soft Eviction timer works correctly ([#52046](https://github.com/kubernetes/kubernetes/pull/52046), [@dashpole](https://github.com/dashpole))
* Azuredisk mount on windows node ([#51252](https://github.com/kubernetes/kubernetes/pull/51252), [@andyzhangx](https://github.com/andyzhangx))
* [fluentd-gcp addon] Bug with event-exporter leaking memory on metrics in clusters with CA is fixed. ([#52263](https://github.com/kubernetes/kubernetes/pull/52263), [@crassirostris](https://github.com/crassirostris))
* kubeadm: Enable kubelet client certificate rotation ([#52196](https://github.com/kubernetes/kubernetes/pull/52196), [@luxas](https://github.com/luxas))
* Scheduler predicate developer should respect equivalence class cache ([#52146](https://github.com/kubernetes/kubernetes/pull/52146), [@resouer](https://github.com/resouer))
* The `kube-cloud-controller-manager` flag `--service-account-private-key-file` was non-functional and is now deprecated. ([#50289](https://github.com/kubernetes/kubernetes/pull/50289), [@liggitt](https://github.com/liggitt))
    * The `kube-cloud-controller-manager` flag `--use-service-account-credentials` is now honored consistently, regardless of whether `--service-account-private-key-file` was specified.
* Fix credentials providers for docker sandbox image. ([#51870](https://github.com/kubernetes/kubernetes/pull/51870), [@feiskyer](https://github.com/feiskyer))
* NONE ([#52120](https://github.com/kubernetes/kubernetes/pull/52120), [@abgworrall](https://github.com/abgworrall))
* Fixed an issue looking up cronjobs when they existed in more than one API version ([#52227](https://github.com/kubernetes/kubernetes/pull/52227), [@liggitt](https://github.com/liggitt))
* Add priority-based preemption to the scheduler. ([#50949](https://github.com/kubernetes/kubernetes/pull/50949), [@bsalamat](https://github.com/bsalamat))
* Add CLUSTER_SIGNING_DURATION environment variable to cluster configuration scripts ([#51844](https://github.com/kubernetes/kubernetes/pull/51844), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * to allow configuration of signing duration of certificates issued via the Certificate
    * Signing Request API.
* Adding German translation for kubectl ([#51867](https://github.com/kubernetes/kubernetes/pull/51867), [@Steffen911](https://github.com/Steffen911))
* The ScaleIO volume plugin can now read the SDC GUID value as node label scaleio.sdcGuid; if binary drv_cfg is not installed, the plugin will still work properly; if node label not found, it defaults to drv_cfg if installed. ([#50780](https://github.com/kubernetes/kubernetes/pull/50780), [@vladimirvivien](https://github.com/vladimirvivien))
* A policy with 0 rules should return an error ([#51782](https://github.com/kubernetes/kubernetes/pull/51782), [@charrywanganthony](https://github.com/charrywanganthony))
* Log a warning when --audit-policy-file not passed to apiserver ([#52071](https://github.com/kubernetes/kubernetes/pull/52071), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Fixes an issue with upgrade requests made via pod/service/node proxy subresources sending a non-absolute HTTP request-uri to backends ([#52065](https://github.com/kubernetes/kubernetes/pull/52065), [@liggitt](https://github.com/liggitt))
* kubeadm: add `kubeadm phase addons` command ([#51171](https://github.com/kubernetes/kubernetes/pull/51171), [@andrewrynhard](https://github.com/andrewrynhard))
* Fix for Nodes in vSphere lacking an InternalIP. ([#48760](https://github.com/kubernetes/kubernetes/pull/48760)) ([#49202](https://github.com/kubernetes/kubernetes/pull/49202), [@cbonte](https://github.com/cbonte))
* v2 of the autoscaling API group, including improvements to the HorizontalPodAutoscaler, has moved from alpha1 to beta1. ([#50708](https://github.com/kubernetes/kubernetes/pull/50708), [@DirectXMan12](https://github.com/DirectXMan12))
* Fixed a bug where some alpha features were enabled by default. ([#51839](https://github.com/kubernetes/kubernetes/pull/51839), [@jennybuckley](https://github.com/jennybuckley))
* Implement StatsProvider interface using CRI stats ([#51557](https://github.com/kubernetes/kubernetes/pull/51557), [@yguo0905](https://github.com/yguo0905))
* set AdvancedAuditing feature gate to true by default ([#51943](https://github.com/kubernetes/kubernetes/pull/51943), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Migrate the metrics/v1alpha1 API to metrics/v1beta1.  The HorizontalPodAutoscaler ([#51653](https://github.com/kubernetes/kubernetes/pull/51653), [@DirectXMan12](https://github.com/DirectXMan12))
    * controller REST client now uses that version.  For v1beta1, the API is now known as
    * resource-metrics.metrics.k8s.io.
* In GCE with COS, increase TasksMax for Docker service to raise cap on number of threads/processes used by containers. ([#51986](https://github.com/kubernetes/kubernetes/pull/51986), [@yujuhong](https://github.com/yujuhong))
* Fixes an issue with APIService auto-registration affecting rolling HA apiserver restarts that add or remove API groups being served. ([#51921](https://github.com/kubernetes/kubernetes/pull/51921), [@liggitt](https://github.com/liggitt))
* Sharing a PID namespace between containers in a pod is disabled by default in 1.8. To enable for a node, use the --docker-disable-shared-pid=false kubelet flag. Note that PID namespace sharing requires docker >= 1.13.1. ([#51634](https://github.com/kubernetes/kubernetes/pull/51634), [@verb](https://github.com/verb))
* Build test targets for all server platforms ([#51873](https://github.com/kubernetes/kubernetes/pull/51873), [@luxas](https://github.com/luxas))
* Add EgressRule to NetworkPolicy ([#51351](https://github.com/kubernetes/kubernetes/pull/51351), [@cmluciano](https://github.com/cmluciano))
* Allow DNS resolution of service name for COS using containerized mounter.  It fixed the issue with DNS resolution of NFS and Gluster services. ([#51645](https://github.com/kubernetes/kubernetes/pull/51645), [@jingxu97](https://github.com/jingxu97))
* Fix journalctl leak on kubelet restart ([#51751](https://github.com/kubernetes/kubernetes/pull/51751), [@dashpole](https://github.com/dashpole))
    * Fix container memory rss
    * Add hugepages monitoring support
    * Fix incorrect CPU usage metrics with 4.7 kernel
    * Add tmpfs monitoring support
* Support for Huge pages in empty_dir volume plugin ([#50072](https://github.com/kubernetes/kubernetes/pull/50072), [@squall0gd](https://github.com/squall0gd))
    * [Huge pages](https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt) can now be used with empty dir volume plugin.
* Alpha support for pre-allocated hugepages ([#50859](https://github.com/kubernetes/kubernetes/pull/50859), [@derekwaynecarr](https://github.com/derekwaynecarr))
* add support for client-side spam filtering of events ([#47367](https://github.com/kubernetes/kubernetes/pull/47367), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Provide a way to omit Event stages in audit policy ([#49280](https://github.com/kubernetes/kubernetes/pull/49280), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Introduced Metrics Server ([#51792](https://github.com/kubernetes/kubernetes/pull/51792), [@piosz](https://github.com/piosz))
* Implement Controller for growing persistent volumes ([#49727](https://github.com/kubernetes/kubernetes/pull/49727), [@gnufied](https://github.com/gnufied))
* Kubernetes 1.8 supports docker version 1.11.x, 1.12.x and 1.13.x. And also supports overlay2. ([#51845](https://github.com/kubernetes/kubernetes/pull/51845), [@Random-Liu](https://github.com/Random-Liu))
* The Deployment, DaemonSet, and ReplicaSet kinds in the extensions/v1beta1 group version are now deprecated, as are the Deployment, StatefulSet, and ControllerRevision kinds in apps/v1beta1. As they will not be removed until after a GA version becomes available, you may continue to use these kinds in existing code. However, all new code should be developed against the apps/v1beta2 group version. ([#51828](https://github.com/kubernetes/kubernetes/pull/51828), [@kow3ns](https://github.com/kow3ns))
* kubeadm: Detect kubelet readiness and error out if the kubelet is unhealthy ([#51369](https://github.com/kubernetes/kubernetes/pull/51369), [@luxas](https://github.com/luxas))
* Fix providerID update validation ([#51761](https://github.com/kubernetes/kubernetes/pull/51761), [@karataliu](https://github.com/karataliu))
* Calico has been updated to v2.5, RBAC added, and is now automatically scaled when GCE clusters are resized. ([#51237](https://github.com/kubernetes/kubernetes/pull/51237), [@gunjan5](https://github.com/gunjan5))
* Add backoff policy and failed pod limit for a job ([#51153](https://github.com/kubernetes/kubernetes/pull/51153), [@clamoriniere1A](https://github.com/clamoriniere1A))
* Adds a new alpha EventRateLimit admission control that is used to limit the number of event queries that are accepted by the API Server. ([#50925](https://github.com/kubernetes/kubernetes/pull/50925), [@staebler](https://github.com/staebler))
* The OpenID Connect authenticator can now use a custom prefix, or omit the default prefix, for username and groups claims through the --oidc-username-prefix and --oidc-groups-prefix flags. For example, the authenticator can map a user with the username "jane" to "google:jane" by supplying the "google:" username prefix. ([#50875](https://github.com/kubernetes/kubernetes/pull/50875), [@ericchiang](https://github.com/ericchiang))
* Implemented `kubeadm upgrade plan` for checking whether you can upgrade your cluster to a newer version ([#48899](https://github.com/kubernetes/kubernetes/pull/48899), [@luxas](https://github.com/luxas))
    * Implemented `kubeadm upgrade apply` for upgrading your cluster from one version to an other
* Switch to audit.k8s.io/v1beta1 in audit. ([#51719](https://github.com/kubernetes/kubernetes/pull/51719), [@soltysh](https://github.com/soltysh))
* update QEMU version to v2.9.1 ([#50597](https://github.com/kubernetes/kubernetes/pull/50597), [@dixudx](https://github.com/dixudx))
* controllers backoff better in face of quota denial ([#49142](https://github.com/kubernetes/kubernetes/pull/49142), [@joelsmith](https://github.com/joelsmith))
* The event table output under `kubectl describe` has been simplified to show only the most essential info. ([#51748](https://github.com/kubernetes/kubernetes/pull/51748), [@smarterclayton](https://github.com/smarterclayton))
* Use arm32v7|arm64v8 images instead of the deprecated armhf|aarch64 image organizations ([#50602](https://github.com/kubernetes/kubernetes/pull/50602), [@dixudx](https://github.com/dixudx))
* audit newest impersonated user info in the ResponseStarted, ResponseComplete audit stage ([#48184](https://github.com/kubernetes/kubernetes/pull/48184), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Fixed bug in AWS provider to handle multiple IPs when using more than 1 network interface per ec2 instance. ([#50112](https://github.com/kubernetes/kubernetes/pull/50112), [@jlz27](https://github.com/jlz27))
* Add flag "--include-uninitialized" to kubectl annotate, apply, edit-last-applied, delete, describe, edit, get, label, set. "--include-uninitialized=true" makes kubectl commands apply to uninitialized objects, which by default are ignored if the names of the objects are not provided. "--all" also makes kubectl commands apply to uninitialized objects. Please see the [initializer](https://kubernetes.io/docs/admin/extensible-admission-controllers/) doc for more details. ([#50497](https://github.com/kubernetes/kubernetes/pull/50497), [@dixudx](https://github.com/dixudx))
* GCE: Service object now supports "Network Tiers" as an Alpha feature via annotations. ([#51301](https://github.com/kubernetes/kubernetes/pull/51301), [@yujuhong](https://github.com/yujuhong))
* When using kube-up.sh on GCE, user could set env `ENABLE_POD_PRIORITY=true` to enable pod priority feature gate. ([#51069](https://github.com/kubernetes/kubernetes/pull/51069), [@MrHohn](https://github.com/MrHohn))
* The names generated for ControllerRevision and ReplicaSet are consistent with the GenerateName functionality of the API Server and will not contain "bad words". ([#51538](https://github.com/kubernetes/kubernetes/pull/51538), [@kow3ns](https://github.com/kow3ns))
* PersistentVolumeClaim metrics like "volume_stats_inodes" and "volume_stats_capacity_bytes" are now reported via kubelet prometheus ([#51553](https://github.com/kubernetes/kubernetes/pull/51553), [@wongma7](https://github.com/wongma7))
* When using IP aliases, use a secondary range rather than subnetwork to reserve cluster IPs. ([#51690](https://github.com/kubernetes/kubernetes/pull/51690), [@bowei](https://github.com/bowei))
* IPAM controller unifies handling of node pod CIDR range allocation. ([#51374](https://github.com/kubernetes/kubernetes/pull/51374), [@bowei](https://github.com/bowei))
    * It is intended to supersede the logic that is currently in range_allocator 
    * and cloud_cidr_allocator. (ALPHA FEATURE)
    * Note: for this change, the other allocators still exist and are the default.
    * It supports two modes:
        * CIDR range allocations done within the cluster that are then propagated out to the cloud provider.
        * Cloud provider managed IPAM that is then reflected into the cluster.
* The Kubernetes API server now supports the ability to break large LIST calls into multiple smaller chunks.  A client can specify a limit to the number of results to return, and if more results exist a token will be returned that allows the client to continue the previous list call repeatedly until all results are retrieved.  The resulting list is identical to a list call that does not perform chunking thanks to capabilities provided by etcd3.  This allows the server to use less memory and CPU responding with very large lists.  This feature is gated as APIListChunking and is not enabled by default.  The 1.9 release will begin using this by default from all informers. ([#48921](https://github.com/kubernetes/kubernetes/pull/48921), [@smarterclayton](https://github.com/smarterclayton))
* add reconcile command to kubectl auth ([#51636](https://github.com/kubernetes/kubernetes/pull/51636), [@deads2k](https://github.com/deads2k))
* Advanced audit allows logging failed login attempts ([#51119](https://github.com/kubernetes/kubernetes/pull/51119), [@soltysh](https://github.com/soltysh))
* kubeadm: Add support for using an external CA whose key is never stored in the cluster ([#50832](https://github.com/kubernetes/kubernetes/pull/50832), [@nckturner](https://github.com/nckturner))
* the custom metrics API (custom.metrics.k8s.io) has moved from v1alpha1 to v1beta1 ([#50920](https://github.com/kubernetes/kubernetes/pull/50920), [@DirectXMan12](https://github.com/DirectXMan12))
* Add backoff policy and failed pod limit for a job ([#48075](https://github.com/kubernetes/kubernetes/pull/48075), [@clamoriniere1A](https://github.com/clamoriniere1A))
* Performs validation (when applying for example) against OpenAPI schema rather than Swagger 1.0. ([#51364](https://github.com/kubernetes/kubernetes/pull/51364), [@apelisse](https://github.com/apelisse))
* Make all e2e tests lookup image to use from a centralized place. In that centralized place, add support for multiple platforms. ([#49457](https://github.com/kubernetes/kubernetes/pull/49457), [@mkumatag](https://github.com/mkumatag))
* kubelet has alpha support for mount propagation. It is disabled by default and it is there for testing only. This feature may be redesigned or even removed in a future release. ([#46444](https://github.com/kubernetes/kubernetes/pull/46444), [@jsafrane](https://github.com/jsafrane))
* Add selfsubjectrulesreview API for allowing users to query which permissions they have in a given namespace. ([#48051](https://github.com/kubernetes/kubernetes/pull/48051), [@xilabao](https://github.com/xilabao))
* Kubelet re-binds /var/lib/kubelet directory with rshared mount propagation during startup if it is not shared yet. ([#45724](https://github.com/kubernetes/kubernetes/pull/45724), [@jsafrane](https://github.com/jsafrane))
* Deviceplugin jiayingz ([#51209](https://github.com/kubernetes/kubernetes/pull/51209), [@jiayingz](https://github.com/jiayingz))
* Make logdump support kubemark and support gke with 'use_custom_instance_list' ([#51834](https://github.com/kubernetes/kubernetes/pull/51834), [@shyamjvs](https://github.com/shyamjvs))
* add apps/v1beta2 conversion test ([#49645](https://github.com/kubernetes/kubernetes/pull/49645), [@dixudx](https://github.com/dixudx))
* Fixed an issue ([#47800](https://github.com/kubernetes/kubernetes/pull/47800)) where `kubectl logs -f` failed with `unexpected stream type ""`. ([#50381](https://github.com/kubernetes/kubernetes/pull/50381), [@sczizzo](https://github.com/sczizzo))
* GCE: Internal load balancer IPs are now reserved during service sync to prevent losing the address to another service. ([#51055](https://github.com/kubernetes/kubernetes/pull/51055), [@nicksardo](https://github.com/nicksardo))
* Switch JSON marshal/unmarshal to json-iterator library.  Performance should be close to previous with no generated code. ([#48287](https://github.com/kubernetes/kubernetes/pull/48287), [@thockin](https://github.com/thockin))
* Adds optional group and version information to the discovery interface, so that if an endpoint uses non-default values, the proper value of "kind" can be determined. Scale is a common example. ([#49971](https://github.com/kubernetes/kubernetes/pull/49971), [@deads2k](https://github.com/deads2k))
* Fix security holes in GCE metadata proxy. ([#51302](https://github.com/kubernetes/kubernetes/pull/51302), [@ihmccreery](https://github.com/ihmccreery))
* [#43077](https://github.com/kubernetes/kubernetes/pull/43077) introduced a condition where DaemonSet controller did not respect the TerminationGracePeriodSeconds of the Pods it created. This is now corrected. ([#51279](https://github.com/kubernetes/kubernetes/pull/51279), [@kow3ns](https://github.com/kow3ns))
* Tainted nodes by conditions as following: ([#49257](https://github.com/kubernetes/kubernetes/pull/49257), [@k82cn](https://github.com/k82cn))
          * 'node.kubernetes.io/network-unavailable=:NoSchedule' if NetworkUnavailable is true
          * 'node.kubernetes.io/disk-pressure=:NoSchedule' if DiskPressure is true
          * 'node.kubernetes.io/memory-pressure=:NoSchedule' if MemoryPressure is true
          * 'node.kubernetes.io/out-of-disk=:NoSchedule' if OutOfDisk is true
* rbd: default image format to v2 instead of deprecated v1 ([#51574](https://github.com/kubernetes/kubernetes/pull/51574), [@dillaman](https://github.com/dillaman))
* Surface reasonable error when client detects connection closed. ([#51381](https://github.com/kubernetes/kubernetes/pull/51381), [@mengqiy](https://github.com/mengqiy))
* Allow PSP's to specify a whitelist of allowed paths for host volume ([#50212](https://github.com/kubernetes/kubernetes/pull/50212), [@jhorwit2](https://github.com/jhorwit2))
* For Deployment, ReplicaSet, and DaemonSet, selectors are now immutable when updating via the new `apps/v1beta2` API. For backward compatibility, selectors can still be changed when updating via `apps/v1beta1` or `extensions/v1beta1`. ([#50719](https://github.com/kubernetes/kubernetes/pull/50719), [@crimsonfaith91](https://github.com/crimsonfaith91))
* Allows kubectl to use http caching mechanism for the OpenAPI schema. The cache directory can be configured through `--cache-dir` command line flag to kubectl. If set to empty string, caching will be disabled. ([#50404](https://github.com/kubernetes/kubernetes/pull/50404), [@apelisse](https://github.com/apelisse))
* Pod log attempts are now reported in apiserver prometheus metrics with verb `CONNECT` since they can run for very long periods of time. ([#50123](https://github.com/kubernetes/kubernetes/pull/50123), [@WIZARD-CXY](https://github.com/WIZARD-CXY))
* The `emptyDir.sizeLimit` field is now correctly omitted from API requests and responses when unset. ([#50163](https://github.com/kubernetes/kubernetes/pull/50163), [@jingxu97](https://github.com/jingxu97))
* Promote CronJobs to batch/v1beta1. ([#51465](https://github.com/kubernetes/kubernetes/pull/51465), [@soltysh](https://github.com/soltysh))
* Add local ephemeral storage support to LimitRange ([#50757](https://github.com/kubernetes/kubernetes/pull/50757), [@NickrenREN](https://github.com/NickrenREN))
* Add mount options field to StorageClass. The options listed there are automatically added to PVs provisioned using the class. ([#51228](https://github.com/kubernetes/kubernetes/pull/51228), [@wongma7](https://github.com/wongma7))
* Add 'kubectl set env' command for setting environment variables inside containers for resources embedding pod templates ([#50998](https://github.com/kubernetes/kubernetes/pull/50998), [@zjj2wry](https://github.com/zjj2wry))
* Implement IPVS-based in-cluster service load balancing ([#46580](https://github.com/kubernetes/kubernetes/pull/46580), [@dujun1990](https://github.com/dujun1990))
* Release the kubelet client certificate rotation as beta. ([#51045](https://github.com/kubernetes/kubernetes/pull/51045), [@jcbsmpsn](https://github.com/jcbsmpsn))
* Adds --append-hash flag to kubectl create configmap/secret, which will append a short hash of the configmap/secret contents to the name during creation. ([#49961](https://github.com/kubernetes/kubernetes/pull/49961), [@mtaufen](https://github.com/mtaufen))
* Add validation for CustomResources via JSON Schema. ([#47263](https://github.com/kubernetes/kubernetes/pull/47263), [@nikhita](https://github.com/nikhita))
* enqueue a sync task to wake up jobcontroller to check job ActiveDeadlineSeconds in time ([#48454](https://github.com/kubernetes/kubernetes/pull/48454), [@weiwei04](https://github.com/weiwei04))
* Remove previous local ephemeral storage resource names: "ResourceStorageOverlay" and "ResourceStorageScratch" ([#51425](https://github.com/kubernetes/kubernetes/pull/51425), [@NickrenREN](https://github.com/NickrenREN))
* Add `retainKeys` to patchStrategy for v1 Volumes and extensions/v1beta1 DeploymentStrategy. ([#50296](https://github.com/kubernetes/kubernetes/pull/50296), [@mengqiy](https://github.com/mengqiy))
* Add mount options field to PersistentVolume spec ([#50919](https://github.com/kubernetes/kubernetes/pull/50919), [@wongma7](https://github.com/wongma7))
* Use of the alpha initializers feature now requires enabling the `Initializers` feature gate. This feature gate is auto-enabled if the `Initialzers` admission plugin is enabled. ([#51436](https://github.com/kubernetes/kubernetes/pull/51436), [@liggitt](https://github.com/liggitt))
* Fix inconsistent Prometheus cAdvisor metrics ([#51473](https://github.com/kubernetes/kubernetes/pull/51473), [@bboreham](https://github.com/bboreham))
* Add local ephemeral storage to downward API  ([#50435](https://github.com/kubernetes/kubernetes/pull/50435), [@NickrenREN](https://github.com/NickrenREN))
* kubectl zsh autocompletion will work with compinit ([#50561](https://github.com/kubernetes/kubernetes/pull/50561), [@cblecker](https://github.com/cblecker))
* [Experiment Only] When using kube-up.sh on GCE, user could set env `KUBE_PROXY_DAEMONSET=true` to run kube-proxy as a DaemonSet. kube-proxy is run as static pods by default. ([#50705](https://github.com/kubernetes/kubernetes/pull/50705), [@MrHohn](https://github.com/MrHohn))
* Add --request-timeout to kube-apiserver to make global request timeout configurable. ([#51415](https://github.com/kubernetes/kubernetes/pull/51415), [@jpbetz](https://github.com/jpbetz))
* Deprecate auto detecting cloud providers in kubelet. Auto detecting cloud providers go against the initiative for out-of-tree cloud providers as we'll now depend on cAdvisor integrations with cloud providers instead of the core repo. In the near future, `--cloud-provider` for kubelet will either be an empty string or `external`.  ([#51312](https://github.com/kubernetes/kubernetes/pull/51312), [@andrewsykim](https://github.com/andrewsykim))
* Add local ephemeral storage support to Quota ([#49610](https://github.com/kubernetes/kubernetes/pull/49610), [@NickrenREN](https://github.com/NickrenREN))
* Kubelet updates default labels if those are deprecated ([#47044](https://github.com/kubernetes/kubernetes/pull/47044), [@mrIncompetent](https://github.com/mrIncompetent))
* Add error count and time-taken metrics for storage operations such as mount and attach, per-volume-plugin. ([#50036](https://github.com/kubernetes/kubernetes/pull/50036), [@wongma7](https://github.com/wongma7))
* A new predicates, named 'CheckNodeCondition', was added to replace node condition filter. 'NetworkUnavailable', 'OutOfDisk' and 'NotReady' maybe reported as a reason when failed to schedule pods. ([#51117](https://github.com/kubernetes/kubernetes/pull/51117), [@k82cn](https://github.com/k82cn))
* Add support for configurable groups for bootstrap token authentication. ([#50933](https://github.com/kubernetes/kubernetes/pull/50933), [@mattmoyer](https://github.com/mattmoyer))
* Fix forbidden message format ([#49006](https://github.com/kubernetes/kubernetes/pull/49006), [@CaoShuFeng](https://github.com/CaoShuFeng))
* make volumesInUse sorted in node status updates ([#49849](https://github.com/kubernetes/kubernetes/pull/49849), [@dixudx](https://github.com/dixudx))
* Adds `InstanceExists` and `InstanceExistsByProviderID` to cloud provider interface for the cloud controller manager ([#51087](https://github.com/kubernetes/kubernetes/pull/51087), [@prydie](https://github.com/prydie))
* Dynamic Flexvolume plugin discovery. Flexvolume plugins can now be discovered on the fly rather than only at system initialization time. ([#50031](https://github.com/kubernetes/kubernetes/pull/50031), [@verult](https://github.com/verult))
* add fieldSelector spec.schedulerName ([#50582](https://github.com/kubernetes/kubernetes/pull/50582), [@dixudx](https://github.com/dixudx))
* Change eviction manager to manage one single local ephemeral storage resource ([#50889](https://github.com/kubernetes/kubernetes/pull/50889), [@NickrenREN](https://github.com/NickrenREN))
* Cloud Controller Manager now sets Node.Spec.ProviderID ([#50730](https://github.com/kubernetes/kubernetes/pull/50730), [@andrewsykim](https://github.com/andrewsykim))
* Paramaterize session affinity timeout seconds in service API for Client IP based session affinity. ([#49850](https://github.com/kubernetes/kubernetes/pull/49850), [@m1093782566](https://github.com/m1093782566))
* Changing scheduling part of the alpha feature 'LocalStorageCapacityIsolation' to manage one single local ephemeral storage resource ([#50819](https://github.com/kubernetes/kubernetes/pull/50819), [@NickrenREN](https://github.com/NickrenREN))
* set --audit-log-format default to json ([#50971](https://github.com/kubernetes/kubernetes/pull/50971), [@CaoShuFeng](https://github.com/CaoShuFeng))
* kubeadm: Implement a `--dry-run` mode and flag for `kubeadm` ([#51122](https://github.com/kubernetes/kubernetes/pull/51122), [@luxas](https://github.com/luxas))
* kubectl rollout `history`, `status`, and `undo` subcommands now support StatefulSets. ([#49674](https://github.com/kubernetes/kubernetes/pull/49674), [@crimsonfaith91](https://github.com/crimsonfaith91))
* Add IPBlock to Network Policy ([#50033](https://github.com/kubernetes/kubernetes/pull/50033), [@cmluciano](https://github.com/cmluciano))
* Adding Italian translation for kubectl ([#50155](https://github.com/kubernetes/kubernetes/pull/50155), [@lucab85](https://github.com/lucab85))
* Simplify capabilities handling in FlexVolume. ([#51169](https://github.com/kubernetes/kubernetes/pull/51169), [@MikaelCluseau](https://github.com/MikaelCluseau))
* NONE ([#50669](https://github.com/kubernetes/kubernetes/pull/50669), [@jiulongzaitian](https://github.com/jiulongzaitian))
* cloudprovider.Zones should support external cloud providers ([#50858](https://github.com/kubernetes/kubernetes/pull/50858), [@andrewsykim](https://github.com/andrewsykim))
* Finalizers are now honored on custom resources, and on other resources even when garbage collection is disabled via the apiserver flag `--enable-garbage-collector=false` ([#51148](https://github.com/kubernetes/kubernetes/pull/51148), [@ironcladlou](https://github.com/ironcladlou))
* Renamed the API server flag `--experimental-bootstrap-token-auth` to `--enable-bootstrap-token-auth`. The old value is accepted with a warning in 1.8 and will be removed in 1.9. ([#51198](https://github.com/kubernetes/kubernetes/pull/51198), [@mattmoyer](https://github.com/mattmoyer))
* Azure file persistent volumes can use a new `secretNamespace` field to reference a secret in a different namespace than the one containing their bound persistent volume claim. The azure file persistent volume provisioner honors a corresponding `secretNamespace` storage class parameter to determine where to place secrets containing the storage account key. ([#47660](https://github.com/kubernetes/kubernetes/pull/47660), [@rootfs](https://github.com/rootfs))
* Bumped gRPC version to 1.3.0 ([#51154](https://github.com/kubernetes/kubernetes/pull/51154), [@RenaudWasTaken](https://github.com/RenaudWasTaken))
* Delete load balancers if the UIDs for services don't match. ([#50539](https://github.com/kubernetes/kubernetes/pull/50539), [@brendandburns](https://github.com/brendandburns))
* Show events when describing service accounts ([#51035](https://github.com/kubernetes/kubernetes/pull/51035), [@mrogers950](https://github.com/mrogers950))
* implement proposal 34058: hostPath volume type ([#46597](https://github.com/kubernetes/kubernetes/pull/46597), [@dixudx](https://github.com/dixudx))
* HostAlias is now supported for both non-HostNetwork Pods and HostNetwork Pods. ([#50646](https://github.com/kubernetes/kubernetes/pull/50646), [@rickypai](https://github.com/rickypai))
* CRDs support metadata.generation and implement spec/status split ([#50764](https://github.com/kubernetes/kubernetes/pull/50764), [@nikhita](https://github.com/nikhita))
* Allow attach of volumes to multiple nodes for vSphere ([#51066](https://github.com/kubernetes/kubernetes/pull/51066), [@BaluDontu](https://github.com/BaluDontu))


Please see the [Releases Page](https://github.com/kubernetes/kubernetes/releases) for older releases.

Release notes of older releases can be found in:
- [CHANGELOG-1.2.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.2.md)
- [CHANGELOG-1.3.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.3.md)
- [CHANGELOG-1.4.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.4.md)
- [CHANGELOG-1.5.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.5.md)
- [CHANGELOG-1.6.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.6.md)
- [CHANGELOG-1.7.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.7.md)
- [CHANGELOG-1.8.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.8.md)

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CHANGELOG.md?pixel)]()
