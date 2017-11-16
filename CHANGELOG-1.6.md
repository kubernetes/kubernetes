<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.6.12](#v1612)
  - [Downloads for v1.6.12](#downloads-for-v1612)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.6.11](#changelog-since-v1611)
    - [Other notable changes](#other-notable-changes)
- [v1.6.11](#v1611)
  - [Downloads for v1.6.11](#downloads-for-v1611)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.6.10](#changelog-since-v1610)
    - [Other notable changes](#other-notable-changes-1)
- [v1.6.10](#v1610)
  - [Downloads for v1.6.10](#downloads-for-v1610)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.6.9](#changelog-since-v169)
    - [Other notable changes](#other-notable-changes-2)
- [v1.6.9](#v169)
  - [Downloads for v1.6.9](#downloads-for-v169)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.6.8](#changelog-since-v168)
    - [Other notable changes](#other-notable-changes-3)
- [v1.6.8](#v168)
  - [Downloads for v1.6.8](#downloads-for-v168)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.6.7](#changelog-since-v167)
    - [Other notable changes](#other-notable-changes-4)
- [v1.6.7](#v167)
  - [Downloads for v1.6.7](#downloads-for-v167)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.6.6](#changelog-since-v166)
    - [Other notable changes](#other-notable-changes-5)
- [v1.6.6](#v166)
  - [Downloads for v1.6.6](#downloads-for-v166)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.6.5](#changelog-since-v165)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-6)
- [v1.6.5](#v165)
  - [Known Issues for v1.6.5](#known-issues-for-v165)
  - [Downloads for v1.6.5](#downloads-for-v165)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
  - [Changelog since v1.6.4](#changelog-since-v164)
    - [Other notable changes](#other-notable-changes-7)
- [v1.6.4](#v164)
  - [Known Issues for v1.6.4](#known-issues-for-v164)
  - [Downloads for v1.6.4](#downloads-for-v164)
    - [Client Binaries](#client-binaries-8)
    - [Server Binaries](#server-binaries-8)
    - [Node Binaries](#node-binaries-8)
  - [Changelog since v1.6.3](#changelog-since-v163)
    - [Other notable changes](#other-notable-changes-8)
- [v1.6.3](#v163)
  - [Known Issues for v1.6.3](#known-issues-for-v163)
  - [Downloads for v1.6.3](#downloads-for-v163)
    - [Client Binaries](#client-binaries-9)
    - [Server Binaries](#server-binaries-9)
    - [Node Binaries](#node-binaries-9)
  - [Changelog since v1.6.2](#changelog-since-v162)
    - [Other notable changes](#other-notable-changes-9)
- [v1.6.2](#v162)
  - [Downloads for v1.6.2](#downloads-for-v162)
    - [Client Binaries](#client-binaries-10)
    - [Server Binaries](#server-binaries-10)
  - [Changelog since v1.6.1](#changelog-since-v161)
    - [Other notable changes](#other-notable-changes-10)
- [v1.6.1](#v161)
  - [Downloads for v1.6.1](#downloads-for-v161)
    - [Client Binaries](#client-binaries-11)
    - [Server Binaries](#server-binaries-11)
  - [Changelog since v1.6.0](#changelog-since-v160)
    - [Other notable changes](#other-notable-changes-11)
- [v1.6.0](#v160)
  - [Downloads for v1.6.0](#downloads-for-v160)
    - [Client Binaries](#client-binaries-12)
    - [Server Binaries](#server-binaries-12)
  - [WARNING: etcd backup strongly recommended](#warning-etcd-backup-strongly-recommended)
  - [Major updates and release themes](#major-updates-and-release-themes)
  - [Action Required](#action-required-1)
    - [Certificates API](#certificates-api)
    - [Cluster Autoscaler](#cluster-autoscaler)
    - [Deployment](#deployment)
    - [Federation](#federation)
    - [Internal Storage Layer](#internal-storage-layer)
    - [Node Components](#node-components)
    - [kubectl](#kubectl)
    - [RBAC](#rbac)
    - [Scheduling](#scheduling)
    - [Service](#service)
    - [StatefulSet](#statefulset)
    - [Volumes](#volumes)
  - [Notable Features](#notable-features)
    - [Autoscaling](#autoscaling)
    - [DaemonSet](#daemonset)
    - [Deployment](#deployment-1)
    - [Federation](#federation-1)
    - [Internal Storage Layer](#internal-storage-layer-1)
    - [kubeadm](#kubeadm)
    - [Node Components](#node-components-1)
    - [RBAC](#rbac-1)
    - [Scheduling](#scheduling-1)
    - [Service Catalog](#service-catalog)
    - [Volumes](#volumes-1)
  - [Deprecations](#deprecations)
    - [Cluster Provisioning Scripts](#cluster-provisioning-scripts)
    - [kubeadm](#kubeadm-1)
    - [Other Deprecations](#other-deprecations)
  - [Changes to API Resources](#changes-to-api-resources)
    - [ABAC](#abac)
    - [Admission Control](#admission-control)
    - [Authentication](#authentication)
    - [Authorization](#authorization)
    - [Autoscaling](#autoscaling-1)
    - [Certificates](#certificates)
    - [ConfigMap](#configmap)
    - [CronJob](#cronjob)
    - [DaemonSet](#daemonset-1)
    - [Deployment](#deployment-2)
    - [Node](#node)
    - [Pod](#pod)
    - [Pod Security Policy](#pod-security-policy)
    - [RBAC](#rbac-2)
    - [ReplicaSet](#replicaset)
    - [Secrets](#secrets)
    - [Service](#service-1)
    - [StatefulSet](#statefulset-1)
    - [Taints and Tolerations](#taints-and-tolerations)
    - [Volumes](#volumes-2)
  - [Changes to Major Components](#changes-to-major-components)
    - [API Server](#api-server)
    - [API Server Aggregator](#api-server-aggregator)
      - [Generic API Server](#generic-api-server)
    - [Client](#client)
      - [client-go](#client-go)
    - [Cloud Provider](#cloud-provider)
      - [AWS](#aws)
      - [Azure](#azure)
      - [GCE](#gce)
      - [GKE](#gke)
      - [vSphere](#vsphere)
    - [Federation](#federation-2)
      - [kubefed](#kubefed)
      - [Other Notable Changes](#other-notable-changes-12)
    - [Garbage Collector](#garbage-collector)
    - [kubeadm](#kubeadm-2)
    - [kubectl](#kubectl-1)
      - [New Commands](#new-commands)
      - [Create subcommands](#create-subcommands)
      - [Updates to existing commands](#updates-to-existing-commands)
      - [Updates to apply](#updates-to-apply)
      - [Updates to edit](#updates-to-edit)
      - [Bug fixes](#bug-fixes)
      - [Other Notable Changes](#other-notable-changes-13)
    - [Node Components](#node-components-2)
      - [Bug fixes](#bug-fixes-1)
    - [kube-controller-manager](#kube-controller-manager)
    - [kube-dns](#kube-dns)
    - [kube-proxy](#kube-proxy)
    - [Scheduler](#scheduler)
    - [Volume Plugins](#volume-plugins)
      - [Azure Disk](#azure-disk)
      - [GlusterFS](#glusterfs)
      - [Photon](#photon)
      - [rbd](#rbd)
      - [vSphere](#vsphere-1)
      - [Other Notable Changes](#other-notable-changes-14)
  - [Changes to Cluster Provisioning Scripts](#changes-to-cluster-provisioning-scripts)
    - [AWS](#aws-1)
    - [Juju](#juju)
    - [libvirt CoreOS](#libvirt-coreos)
    - [GCE](#gce-1)
    - [OpenStack](#openstack)
    - [Container Images](#container-images)
    - [Other Notable Changes](#other-notable-changes-15)
  - [Changes to Addons](#changes-to-addons)
    - [Dashboard](#dashboard)
    - [DNS](#dns)
    - [DNS Autoscaler](#dns-autoscaler)
    - [Cluster Autoscaler](#cluster-autoscaler-1)
    - [Cluster Load Balancing](#cluster-load-balancing)
    - [etcd Empty Dir Cleanup](#etcd-empty-dir-cleanup)
    - [Fluentd](#fluentd)
    - [Heapster](#heapster)
    - [Registry](#registry)
  - [External Dependency Version Information](#external-dependency-version-information)
  - [Changelog since v1.6.0-rc.1](#changelog-since-v160-rc1)
    - [Previous Releases Included in v1.6.0](#previous-releases-included-in-v160)
- [v1.6.0-rc.1](#v160-rc1)
  - [Downloads for v1.6.0-rc.1](#downloads-for-v160-rc1)
    - [Client Binaries](#client-binaries-13)
    - [Server Binaries](#server-binaries-13)
  - [Changelog since v1.6.0-beta.4](#changelog-since-v160-beta4)
    - [Other notable changes](#other-notable-changes-16)
- [v1.6.0-beta.4](#v160-beta4)
  - [Downloads for v1.6.0-beta.4](#downloads-for-v160-beta4)
    - [Client Binaries](#client-binaries-14)
    - [Server Binaries](#server-binaries-14)
  - [Changelog since v1.6.0-beta.3](#changelog-since-v160-beta3)
    - [Other notable changes](#other-notable-changes-17)
- [v1.6.0-beta.3](#v160-beta3)
  - [Downloads for v1.6.0-beta.3](#downloads-for-v160-beta3)
    - [Client Binaries](#client-binaries-15)
    - [Server Binaries](#server-binaries-15)
  - [Changelog since v1.6.0-beta.2](#changelog-since-v160-beta2)
    - [Other notable changes](#other-notable-changes-18)
- [v1.6.0-beta.2](#v160-beta2)
  - [Downloads for v1.6.0-beta.2](#downloads-for-v160-beta2)
    - [Client Binaries](#client-binaries-16)
    - [Server Binaries](#server-binaries-16)
  - [Changelog since v1.6.0-beta.1](#changelog-since-v160-beta1)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-19)
- [v1.6.0-beta.1](#v160-beta1)
  - [Downloads for v1.6.0-beta.1](#downloads-for-v160-beta1)
    - [Client Binaries](#client-binaries-17)
    - [Server Binaries](#server-binaries-17)
  - [Changelog since v1.6.0-alpha.3](#changelog-since-v160-alpha3)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-20)
- [v1.6.0-alpha.3](#v160-alpha3)
  - [Downloads for v1.6.0-alpha.3](#downloads-for-v160-alpha3)
    - [Client Binaries](#client-binaries-18)
    - [Server Binaries](#server-binaries-18)
  - [Changelog since v1.6.0-alpha.2](#changelog-since-v160-alpha2)
    - [Other notable changes](#other-notable-changes-21)
- [v1.6.0-alpha.2](#v160-alpha2)
  - [Downloads for v1.6.0-alpha.2](#downloads-for-v160-alpha2)
    - [Client Binaries](#client-binaries-19)
    - [Server Binaries](#server-binaries-19)
  - [Changelog since v1.6.0-alpha.1](#changelog-since-v160-alpha1)
    - [Other notable changes](#other-notable-changes-22)
- [v1.6.0-alpha.1](#v160-alpha1)
  - [Downloads for v1.6.0-alpha.1](#downloads-for-v160-alpha1)
    - [Client Binaries](#client-binaries-20)
    - [Server Binaries](#server-binaries-20)
  - [Changelog since v1.5.0](#changelog-since-v150)
    - [Action Required](#action-required-4)
    - [Other notable changes](#other-notable-changes-23)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.6.12

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.12


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes.tar.gz) | `d57a05942f581959bc31778def4df4b526df80adaa75942214e9439fdecbe74f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-src.tar.gz) | `c1781161d1b0fa0c8dbd3a644ee42e65968a0a4bb5bec47f2c70796edbc1d396`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-darwin-386.tar.gz) | `85956c72c8fc1a1c4768e95fb403ddb1740db1f2d6dec8263d74a67cfce429b9`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-darwin-amd64.tar.gz) | `bcea37dc2852ee83bdcac505600e724ae391e3a5879232d47ddd763cb2253c14`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-linux-386.tar.gz) | `8e0f3d0a6807b050070717efacd25db9f96ad3f0ef59d9095aae71ab20734212`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-linux-amd64.tar.gz) | `5f324e91dce59e05acdc0d6c0a8a6a56d5a5d1a3144e58b6de5c22a9b9020f8b`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-linux-arm64.tar.gz) | `442fadf05af70826231e03b0146c0f9283ec474d6a1ba390eac7204888db5dc7`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-linux-arm.tar.gz) | `2e8073e626ba6a6f6148ccca3d5bbfd20f051f9992de676446d2ae259555a75a`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-linux-ppc64le.tar.gz) | `f6af1ea1762e9fe889e10da670f61ce0bd2c1bc8fd5c1bd9fc3d1eacabe820a1`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-linux-s390x.tar.gz) | `77aee845c290b5637d2560586e9065220aa433d6e15be03c6bbb6dfc75292564`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-windows-386.tar.gz) | `6a47f85d4e01595bb7e58469a4cab09053e5e11819440c630ae47008aa0b000d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-client-windows-amd64.tar.gz) | `6b4f1ae4bed0523a0c5b2b1d6dc9cb1ac54399a2a78b548acd876b8029a2dd6c`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-server-linux-amd64.tar.gz) | `0970f6da85acd1bb61e92a77c7dbfa3f059da409db68c3d940781708795a0204`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-server-linux-arm64.tar.gz) | `d598d846882744e724e65156807a3b2fa5bb92e1c389ff2e08984455346a7305`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-server-linux-arm.tar.gz) | `6c02ade45ba8495917f0a7fe0542369ba482bc98f2dc008f7f69a21fd2e653d2`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-server-linux-ppc64le.tar.gz) | `5479030e4ff21afcd638f4dd22bd0f6fb221dc3b27731954c0a5e53877ed1131`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-server-linux-s390x.tar.gz) | `6407d6c580acc2fafbe2d87b309e60c6d16624403308e41a8a973d772bf26e3b`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-node-linux-amd64.tar.gz) | `9cb5a28417cbc34165eaa2eb412488738b0e190e9128bb68b16d68d2a942b4c9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-node-linux-arm64.tar.gz) | `e4c1d1994b7595f2ac9c7ffebaaefc0737c1fbbcc4d2a9a9097d52d060236ad8`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-node-linux-arm.tar.gz) | `77879441f3b5e753959b32103ef6a49fbc9ccea16cac1b1fb83446b5d573901d`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-node-linux-ppc64le.tar.gz) | `c6866605adf86d5bc91acb670c5d8bba00c989a8758014a17322c041289e472d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-node-linux-s390x.tar.gz) | `8a280b31c7118728660ac4b7010ed0af9bc3b058e98964c04b47f58c5fd84671`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.12/kubernetes-node-windows-amd64.tar.gz) | `bc3d39ed8f23e59692bcffc23eaae7f575bef01e3c7c1510528662f984283a9c`

## Changelog since v1.6.11

### Other notable changes

* Azure cloudprovider: Fix controller manager crash issue on a manually created k8s cluster. ([#53694](https://github.com/kubernetes/kubernetes/pull/53694), [@andyzhangx](https://github.com/andyzhangx))
* Fix kubelet reset liveness probe failure count across pod restart boundaries. ([#46371](https://github.com/kubernetes/kubernetes/pull/46371), [@sjenning](https://github.com/sjenning))



# v1.6.11

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.11


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes.tar.gz) | `0dacad1c3da0397b6234e474979c4095844733315a853feba5690dbdf8db15dc`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-src.tar.gz) | `818fdfc41d7b6d90b9dc37ca12c2fbe1b6fb20f998ee04fddce4a9bb8610351e`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-darwin-386.tar.gz) | `c9a24250d68ddde25a09955c5f944812a9aeb5e0f3fd61313dbd166348aa5954`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-darwin-amd64.tar.gz) | `f51e83ff2e026856cae7e365b17c20d94fe59d4a2749daa7bc4dbfb184f14a36`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-linux-386.tar.gz) | `16afc423b6f68cc5b24c322ee383e3f7c0fc5c3c98dd4cc90f93cfbd820964a4`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-linux-amd64.tar.gz) | `fca4eaae3bd6b9482ec130146b5ee24159effd66ea70d3c4ce174a45c770fcdd`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-linux-arm64.tar.gz) | `6d7d777357c1920b2ef4060f7f55de7c92655c99aa7caf71fbb6311ddbba4578`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-linux-arm.tar.gz) | `15bbfadbd4ce4b46d1473cb662396f1ac0372c9134ebd597de91565b59ddb200`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-linux-ppc64le.tar.gz) | `961a942875daf30aad3fdebd3796eb6311f46eb31fe8558ffde086c5424a1c2d`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-linux-s390x.tar.gz) | `3874548181ac06feb280f1cf6f7ae851599f68d0abc96d3af17264889ff9d992`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-windows-386.tar.gz) | `7c305dd4d00e877843efa187948c93907d440cf3fcccd31cc18e243c319eec7d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-client-windows-amd64.tar.gz) | `ee27b50a82d845d4e2ddecb401f36e1e47dd0fb8f67c60465e99e8947b740149`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-server-linux-amd64.tar.gz) | `daea028d6777597aaee33ea7c9e3f1210b46ce895faac9ca85c7b1553923ce82`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-server-linux-arm64.tar.gz) | `1f098c7bc06aeb7d532d270538f3aa3a029e3f6460b26e9449b361ed7de93704`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-server-linux-arm.tar.gz) | `c5d6ae53fa95eb0e3b02e046e99144b8604dba7a16f373a2a02ae2fa88818ee2`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-server-linux-ppc64le.tar.gz) | `06bba3736754cc7650b45c6a832b14d0539e63c5cec59f8ecd763803ea4397b6`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-server-linux-s390x.tar.gz) | `632fb6bb0a1144d91b1f559967731223a2bf53423539317e015dcf73aef6cb53`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-node-linux-amd64.tar.gz) | `c8a38711db9625dd4b1e55923961c22276a0a07c976d371dd91b638b6d0a6757`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-node-linux-arm64.tar.gz) | `9bd3c3cf6e98e882b397708f3fab0fd5f4476e97bd3a897598a7ded822bd5314`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-node-linux-arm.tar.gz) | `563d22c94513d287e4f01dbc40b2f300dbdf9c9dbaf8394bf18c2604796dce5b`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-node-linux-ppc64le.tar.gz) | `4d249236a64414ad5b201c994ae867458a49a4dea53c4c7eb5ba1d0af07433c2`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-node-linux-s390x.tar.gz) | `35c2132ef07dedc4d64d72fc194aa0824d427a3780733508493d9d87538cedd1`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.11/kubernetes-node-windows-amd64.tar.gz) | `b4279e7e38d1777354b557e17419ec3ab8399addb0e535669d485fb9416fb76b`

## Changelog since v1.6.10

### Other notable changes

* Update kube-dns to 1.14.5 ([#53112](https://github.com/kubernetes/kubernetes/pull/53112), [@bowei](https://github.com/bowei))
* Fix panic in ControllerManager on GCE when it has a problem with creating external loadbalancer healthcheck ([#52646](https://github.com/kubernetes/kubernetes/pull/52646), [@gmarek](https://github.com/gmarek))
* When performing a GET then PUT, the kube-apiserver must write the canonical representation of the object to etcd if the current value does not match. That allows external agents to migrate content in etcd from one API version to another, across different storage types, or across varying encryption levels. This fixes a bug introduced in 1.5 where we unintentionally stopped writing the newest data. ([#48394](https://github.com/kubernetes/kubernetes/pull/48394), [@smarterclayton](https://github.com/smarterclayton))
* StatefulSet will now fill the `hostname` and `subdomain` fields if they're empty on existing Pods it owns. This allows it to self-correct the issue where StatefulSet Pod DNS entries disappear after upgrading to v1.7.x ([#48327](https://github.com/kubernetes/kubernetes/pull/48327)). ([#51199](https://github.com/kubernetes/kubernetes/pull/51199), [@kow3ns](https://github.com/kow3ns))
* Make logdump support kubemark and support gke with 'use_custom_instance_list' ([#51834](https://github.com/kubernetes/kubernetes/pull/51834), [@shyamjvs](https://github.com/shyamjvs))
* Fix credentials providers for docker sandbox image. ([#51870](https://github.com/kubernetes/kubernetes/pull/51870), [@feiskyer](https://github.com/feiskyer))



# v1.6.10

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.10


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes.tar.gz) | `8877359b78950b12a48ea68483f4e4ba2d2521f7e8620efca6f84275cb023428`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-src.tar.gz) | `560d1441b72c670c3d21f838f8d0a94bc75628b1bdd322b18e91df2578a0f84b`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-darwin-386.tar.gz) | `087799af2856decf438dbd896260cfdec3c02c6a42e2e2f90b608f21d61a8fb6`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-darwin-amd64.tar.gz) | `8d9dbbee46a26fcf7f50af145b888881a428d62a3ee929b75e0a6833553de4ab`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-linux-386.tar.gz) | `d69bd343613bd3f57799d05de5f56ec159ddb6f38cbec1d914b5c2e7a2945f6e`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-linux-amd64.tar.gz) | `b0e2420e66257e67c9f53c996feebff20bebbbe5c9bc12b85b973a165ee436ec`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-linux-arm64.tar.gz) | `4d9b6064fd409789c7bc07b7a3746798f94a19bf811021f727fcf8afdbd432aa`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-linux-arm.tar.gz) | `7a02002aa3c4f6c5c2ff662fe023da3cd32a687967bb42a76e2014fe12a349f4`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-linux-ppc64le.tar.gz) | `f0b26ad0bdf578a8c98e870a276ad7b8d77ef13f423476b21b78f18077b83893`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-linux-s390x.tar.gz) | `da4125aff73b1d397b2917d4397690686d44f426471fd12eed080947c0de03e5`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-windows-386.tar.gz) | `0bd8aeb66a1d5235da85a71150e10838c0d8d67ecb8b368262d52ac86ff10dbd`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-client-windows-amd64.tar.gz) | `3a89271b4554e56c37038f797ad31f357107257d80fed9ab5ca80832e33cf00e`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-server-linux-amd64.tar.gz) | `ae897c9db3a0b89be6ff66ca8e35b41165be620614f60aab424d76deffa45bcc`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-server-linux-arm64.tar.gz) | `f6ce8a89f2ce9b380789828ba2723ac834d2dd40dd20403f22040ee08a390b07`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-server-linux-arm.tar.gz) | `085a3166785ab4fe17cc153fa6306df55af6fa90d5a3a4670923cf4515323f70`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-server-linux-ppc64le.tar.gz) | `c55f6741370471a2caac8b844865d908c8b327f2aea6685e193d54f4b14a5a63`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-server-linux-s390x.tar.gz) | `e36bd6f3bd7493f4ba12ceeebc9a6102778d20d203054d74cd69e929b7abcc84`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-node-linux-amd64.tar.gz) | `e083f7bc8028a2eb03bbcc6be93a95377e74906ae49970af034d36b3409f72de`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-node-linux-arm64.tar.gz) | `95895aab99979aca8cb713f53b2be0f11b16c3a76e97c206a70969a3cc3e003d`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-node-linux-arm.tar.gz) | `a1e2dee888f4ef9affd1c2b747602f4d53971911b93ea69174d7301a5f7e1ccc`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-node-linux-ppc64le.tar.gz) | `1cbaa1f6116c862acaef7a3e1ad6c27bcf5f87eb4ce01f6f9164c58caa2e0009`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-node-linux-s390x.tar.gz) | `178a53d8193464a6062b3ebdea6c8dbb3dcb9f7c0ab1f40e386e555939c0be51`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.10/kubernetes-node-windows-amd64.tar.gz) | `1c044217184bcbe7e11c9fe0c511bbc6353935fc850b2c6b0f6ca0f2cbe31a8e`

## Changelog since v1.6.9

### Other notable changes

* Add --request-timeout to kube-apiserver to make global request timeout configurable. ([#51415](https://github.com/kubernetes/kubernetes/pull/51415), [@jpbetz](https://github.com/jpbetz))
* Fix for Nodes in vSphere lacking an InternalIP. ([#48760](https://github.com/kubernetes/kubernetes/pull/48760)) ([#49202](https://github.com/kubernetes/kubernetes/pull/49202), [@cbonte](https://github.com/cbonte))
* GCE: Bump GLBC version to [0.9.6](https://github.com/kubernetes/ingress/releases/tag/0.9.6). ([#50096](https://github.com/kubernetes/kubernetes/pull/50096), [@nicksardo](https://github.com/nicksardo))
* In GCE with COS, increase TasksMax for Docker service to raise cap on number of threads/processes used by containers. ([#51986](https://github.com/kubernetes/kubernetes/pull/51986), [@yujuhong](https://github.com/yujuhong))
* Fixed an issue ([#47800](https://github.com/kubernetes/kubernetes/pull/47800)) where `kubectl logs -f` failed with `unexpected stream type ""`. ([#51872](https://github.com/kubernetes/kubernetes/pull/51872), [@feiskyer](https://github.com/feiskyer))
* Fix for Pod stuck in ContainerCreating with error "Volume is not yet attached according to node". ([#50806](https://github.com/kubernetes/kubernetes/pull/50806), [@verult](https://github.com/verult))
* Fix initial exec terminal dimensions. ([#51126](https://github.com/kubernetes/kubernetes/pull/51126), [@chen-anders](https://github.com/chen-anders))
* vSphere: Fix attach volume failing on the first try. ([#51218](https://github.com/kubernetes/kubernetes/pull/51218), [@BaluDontu](https://github.com/BaluDontu))



# v1.6.9

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.9


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes.tar.gz) | `08be94c252e7fbdd7c14811ec021818e687c1259e557b70db10aac64c0e8e4b2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-src.tar.gz) | `519501e26afc341b236c5b46602f010a33fc190e3d1bfb7802969b2e979faaeb`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-darwin-386.tar.gz) | `864f2307dd22c055063d1a55354596754a94d03f023e7278c24d5978bba00b3e`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-darwin-amd64.tar.gz) | `0a107e0a1d7e6865ddd9241f1e8357405f476889a6f1a16989ba01f6cffd3be7`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-linux-386.tar.gz) | `b20599e266248e7e176383e0318acd855c1aad8014396cc4018adde11a33d0c8`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-linux-amd64.tar.gz) | `0690a8c9858f91cc000b3acd602799bf2320756b7471e463df2e3a36fbdde886`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-linux-arm64.tar.gz) | `354897ffc6382b8eb27f434d8e7aa3cbfae4b819da3160a43db8ccb8cae1275b`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-linux-arm.tar.gz) | `6897408bf8d65d1281555c21ae978a4ccd69482a7ad2549bcec381416e312d7a`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-linux-ppc64le.tar.gz) | `2afae0c211eb415829446f90a0bf9d48b9f8311ac4566fa74a08415ed9a31e75`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-linux-s390x.tar.gz) | `abde354528cc9c8ced49bb767ffcd8bfae47a0b4b5501502f560cf663a0c4a05`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-windows-386.tar.gz) | `83083c0d78e9468c7be395282a4697d2c703d3310593e7b70cd09fa9e7791d80`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-client-windows-amd64.tar.gz) | `3471db3463d60d22d82edb34fbe3ca301cc583ebddffc2664569255302e7d304`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-server-linux-amd64.tar.gz) | `598e49c8a22e4e8db1e1c0ed9d8955c991425cd4e06c072ac36fd5ed693b1c61`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-server-linux-arm64.tar.gz) | `5ce75f57636d537b4bf3ca00c4a1322e9c1aaf273bd945304333b558af3c081b`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-server-linux-arm.tar.gz) | `afea9780049c5e6548f64973bd8679aae60672ab05027f8c36784ccf2a83a1b2`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-server-linux-ppc64le.tar.gz) | `cd131b3e39e4160cd9920fe2635b4f6da4679cce12cb2483cfe28197e366bceb`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-server-linux-s390x.tar.gz) | `93ee43f33cbe061ac088acf62099be1abd0d9c0b4a8a79be4069904c3780c76d`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-node-linux-amd64.tar.gz) | `f8f13233b168c4833af685817f9591c73658d1377ceb9d550cbea929c6e27c2e`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-node-linux-arm64.tar.gz) | `1ed434f9e6469c8cc7a3bb15404e918cf242ef92ef075e7cf479b7e951269b5c`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-node-linux-arm.tar.gz) | `3fd8e089184f83bd9ed2cf5f193253e1f7b9b853876a08a2babf91647d6d0ac8`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-node-linux-ppc64le.tar.gz) | `9673547a32f83498bb28f02212d419b28cc50a0a4d7b866396994b5ea9313e79`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-node-linux-s390x.tar.gz) | `80044cdeb4260e807660c166ed15bb2a9db03d59d8c186b1d4f9a53841cea327`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.9/kubernetes-node-windows-amd64.tar.gz) | `b1dd678ee2974dc83ea7cfe8516557c9360ed55e40cad1b68803b71786f8d16f`

## Changelog since v1.6.8

### Other notable changes

* StatefulSet: Set hostname/subdomain fields on new Pods in addition to the deprecated annotations, to allow mitigation of Pod DNS issues upon upgrading to Kubernetes v1.7.x. ([#50942](https://github.com/kubernetes/kubernetes/pull/50942), [@enisoc](https://github.com/enisoc))
* Azure: Allow VNet to be in a separate Resource Group. ([#49725](https://github.com/kubernetes/kubernetes/pull/49725), [@sylr](https://github.com/sylr))
* In GCE, add measures to prevent corruption of known_tokens.csv. ([#49897](https://github.com/kubernetes/kubernetes/pull/49897), [@mikedanese](https://github.com/mikedanese))
* Fixed a bug in the API server watch cache, which could cause a missing watch event immediately after cache initialization. ([#49992](https://github.com/kubernetes/kubernetes/pull/49992), [@liggitt](https://github.com/liggitt))



# v1.6.8

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.8


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes.tar.gz) | `c87f7826f0b7cf91baddd97ebafb33e99d91dcf6b9019a50bee0689527541ef7`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-src.tar.gz) | `591c43f9624dac351745da35444302cd694ad4953275b8f09016b4654d37b793`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-darwin-386.tar.gz) | `3f6cda6ca2cf3e8f038649f1021ca23c35f4da12d66cefaa4339c9613ca9bbd6`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-darwin-amd64.tar.gz) | `147bf5124e44a1557b95e7daa76717992b7890e79910c446dc682103f62325eb`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-linux-386.tar.gz) | `cd7238c19f9d4a4ce0b14c2d954f6ead2235caa2d74b319524a0d2ffeea0ca37`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-linux-amd64.tar.gz) | `34042be9607ca75702384552b31514f594af22d3c1d88549b0cd4ce36ee8fd6b`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-linux-arm64.tar.gz) | `3a7d4be76dda07fac50a257275369b3f4c48848e2963b55b46fa9df44477bfc8`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-linux-arm.tar.gz) | `0a060b8745b3c0e8173827af3d91a4748eb191a9c15538625eee108f6024fcfd`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-linux-ppc64le.tar.gz) | `bbc7be082d20082179de5efb85c0da9d0f3811c2119d3928bf89edc8f59e8cd0`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-linux-s390x.tar.gz) | `5e93d7ed4797f6b8742925d13f791e862bdb410bdd2b33737882132aabcc0bfd`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-windows-386.tar.gz) | `22a0a80fa5ed5f0745371cc9fd68eeeb0671242cf7c476fb4e635ccd9ef8c2b1`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-client-windows-amd64.tar.gz) | `ce42d7e826aa07bd98a424332926b04e75effbe926b098565781de3c3b6d244c`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-server-linux-amd64.tar.gz) | `9bf31375917ffdf9a9437ed562e96a1e2b43e23dcb4a42204032bb289ff12b6d`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-server-linux-arm64.tar.gz) | `51d84e7b1ace983b13639f1fe4bf1b11212d178e6a75b769de9bdac97d1fa7ae`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-server-linux-arm.tar.gz) | `b704de70774c6c0feb13a7b47d8d757e9a0438406b7fd1d33d0c5cb991d179b0`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-server-linux-ppc64le.tar.gz) | `f36f086481656fcb659a456ca832d62274e40defc1a3ed1dcc1e5ea7a696729b`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-server-linux-s390x.tar.gz) | `348f8a733556fcceaaa27d316c3e2ea01039c860988a434d7c9a850bc2412546`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-node-linux-amd64.tar.gz) | `e38255961c73e021bcca08890918f23cce39831536bf74496aa369049a1eb165`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-node-linux-arm64.tar.gz) | `be06c10320f3f996a48845eef9572353f9a0bd56330338c4cad6aca1fcc4fac4`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-node-linux-arm.tar.gz) | `06c6ecd885fbb4889791e78f50cdcb9920ee8f1e866d4fa921bc2096dbfbbd4b`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-node-linux-ppc64le.tar.gz) | `74e88435549cc46f3fc082300bf373c7d824921bd01eabf789a1b09e1a17a04a`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-node-linux-s390x.tar.gz) | `7ebe22e74653650ac0cedbfc482f5ff08713c40747018dac7506b36bb78ee8fc`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.8/kubernetes-node-windows-amd64.tar.gz) | `66b66655976647f50db3eda61849cbb26bcb06ad20a866328f24aef862758bb4`

## Changelog since v1.6.7

### Other notable changes

* Revert deprecation of vCenter port in vSphere Cloud Provider. ([#49689](https://github.com/kubernetes/kubernetes/pull/49689), [@divyenpatel](https://github.com/divyenpatel))
* kubeadm: Add preflight check for localhost resolution. ([#48875](https://github.com/kubernetes/kubernetes/pull/48875), [@craigtracey](https://github.com/craigtracey))
* Fix panic when using `kubeadm init` with vsphere cloud-provider. ([#44661](https://github.com/kubernetes/kubernetes/pull/44661), [@xiangpengzhao](https://github.com/xiangpengzhao))
* kubectl: Fix bug that showed terminated/evicted pods even without `--show-all`. ([#48786](https://github.com/kubernetes/kubernetes/pull/48786), [@janetkuo](https://github.com/janetkuo))
* Never prevent deletion of resources as part of namespace lifecycle ([#48733](https://github.com/kubernetes/kubernetes/pull/48733), [@liggitt](https://github.com/liggitt))
* AWS cloudprovider plugin: Fix for large clusters (200+ nodes). Also fix bug with volumes not getting detached from a node after reboot. ([#48312](https://github.com/kubernetes/kubernetes/pull/48312), [@gnufied](https://github.com/gnufied))
* Fix Pods using Portworx volumes getting stuck in ContainerCreating phase. ([#48898](https://github.com/kubernetes/kubernetes/pull/48898), [@harsh-px](https://github.com/harsh-px))
* RBAC role and role-binding reconciliation now ensures namespaces exist when reconciling on startup. ([#48480](https://github.com/kubernetes/kubernetes/pull/48480), [@liggitt](https://github.com/liggitt))



# v1.6.7

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.7


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes.tar.gz) | `6522086d9666543ed4e88a791626953acd1ea843eb024f16f4a4a2390dcbb2b2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-src.tar.gz) | `b2a73f140966ba0080ce16e3b9a67d5fd9849b36942f3490e9f8daa0fe4511c4`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-darwin-386.tar.gz) | `ffa06a16a3091b2697ef14f8e28bb08000455bd9b719cf0f510f011b864cd1e0`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-darwin-amd64.tar.gz) | `32de3e38f7a60c9171a63f43a2c7f0b2d8f8ba55d51468d8dbf7847dbd943b45`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-linux-386.tar.gz) | `d9c27321007607cc5afb2ff5b3cac210471d55dd1c3a478c6703ab72d187211e`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-linux-amd64.tar.gz) | `54947ef84181e89f9dbacedd54717cbed5cc7f9c36cb37bc8afc9097648e2c91`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-linux-arm64.tar.gz) | `e96d300eb6526705b1c1bedaaf3f4746f3e5d6b49ccc7e60650eb9ee022fba0e`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-linux-arm.tar.gz) | `e4605dca3948264fba603dc8f95b202528eb8ad4ca99c7f3a61f77031e7ba756`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-linux-ppc64le.tar.gz) | `8b77793aea5abf1c17b73f7e11476b9d387f3dc89e5d8405ffadd1a395258483`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-linux-s390x.tar.gz) | `ff3ddec930a0ffdc83fe324d544d4657d57a64a3973fb9df4ddaa7a98228d7fb`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-windows-386.tar.gz) | `ce09e4b071bb06039ad9bdf6a1059d59cf129dce942600fcdc9d320ff0c07a7a`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-client-windows-amd64.tar.gz) | `e985644f582945274e82764742f02bd175f05128c1945e987d06973dd5f5a56d`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-server-linux-amd64.tar.gz) | `1287bb85f1057eae53f8bb4e4475c990783e43d2f57ea1c551fdf2da7ca5345d`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-server-linux-arm64.tar.gz) | `51623850475669be59f6428922ba316d4dd60d977f892adfaf0ca0845c38506c`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-server-linux-arm.tar.gz) | `a5331022d29f085e6b7fc4ae064af64024eba6a02ae54e78c2e84b40d0aec598`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-server-linux-ppc64le.tar.gz) | `93d52e84d0fea5bdf3ede6784b8da6c501e0430c74430da3a125bd45c557e10a`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-server-linux-s390x.tar.gz) | `baccbb6fc497f433c2bd93146c31fbca1da427e0d6ac8483df26dd42ccb79c6e`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-node-linux-amd64.tar.gz) | `0cfdd51de879869e7ef40a17dfa1a303a596833fb567c3b7e4f82ba0cf863839`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-node-linux-arm64.tar.gz) | `d07ef669d94ea20a4a9e3a38868ac389dab4d3f2bdf8b27280724fe63f4de3c3`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-node-linux-arm.tar.gz) | `1cc9b6a8aee4e59967421cbded21c0a20f02c39288781f504e55ad6ca71d1037`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-node-linux-ppc64le.tar.gz) | `3f412096d8b249d671f924c3ee4aecf3656186fde4509ce9f560f67a9a166b6d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-node-linux-s390x.tar.gz) | `2cca7629c1236b3435e6e31498c1f8216d7cca4236d8ad0ae10c83a422519a34`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.7/kubernetes-node-windows-amd64.tar.gz) | `4f859fba52c044a9ce703528760967e1efa47a359603b5466c0dc0748eb25e36`

## Changelog since v1.6.6

### Other notable changes

* kubeadm: Expose only the cluster-info ConfigMap in the kube-public ns ([#48050](https://github.com/kubernetes/kubernetes/pull/48050), [@luxas](https://github.com/luxas))
* Fix kubelet request timeout when stopping a container. ([#46267](https://github.com/kubernetes/kubernetes/pull/46267), [@Random-Liu](https://github.com/Random-Liu))
* Add generic NoSchedule toleration to fluentd in gcp config. ([#48182](https://github.com/kubernetes/kubernetes/pull/48182), [@gmarek](https://github.com/gmarek))
* Update cluster-proportional-autoscaler, fluentd-gcp, and kube-addon-manager, and kube-dns addons with refreshed base images containing fixes for CVE-2016-9841, CVE-2016-9843, CVE-2017-2616, and CVE-2017-6512. ([#47454](https://github.com/kubernetes/kubernetes/pull/47454), [@ixdy](https://github.com/ixdy))
* Fix fluentd-gcp configuration to facilitate JSON parsing ([#48139](https://github.com/kubernetes/kubernetes/pull/48139), [@crassirostris](https://github.com/crassirostris))
* Bump runc to v1.0.0-rc2-49-gd223e2a - fixes `failed to initialise top level QOS containers` kubelet error. ([#48117](https://github.com/kubernetes/kubernetes/pull/48117), [@sjenning](https://github.com/sjenning))
* `kubefed init` correctly checks for RBAC API enablement. ([#48077](https://github.com/kubernetes/kubernetes/pull/48077), [@liggitt](https://github.com/liggitt))
* `kubectl api-versions` now always fetches information about enabled API groups and versions instead of using the local cache. ([#48016](https://github.com/kubernetes/kubernetes/pull/48016), [@liggitt](https://github.com/liggitt))
* Fix kubelet event recording for selected events. ([#46246](https://github.com/kubernetes/kubernetes/pull/46246), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Fix `Invalid value: "foregroundDeletion"` error when attempting to delete a resource. ([#46500](https://github.com/kubernetes/kubernetes/pull/46500), [@tnozicka](https://github.com/tnozicka))



# v1.6.6

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.6


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes.tar.gz) | `1574d868d43f5d88cfd1af255226e8cd6da72cd65fb9e1285557279c34f8a645`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-src.tar.gz) | `305f372320f78855e298b6caea062c8d1f7db117c7b44943ff5ddd0169424033`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-darwin-386.tar.gz) | `d93ca6f95cd80856b04d9c76a98ca986d6ba183d9fa100619fcda9f157bfd7f6`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-darwin-amd64.tar.gz) | `facda65133f2893296f64c1067807dd7b354e2a4440afdd1ee62c05c07bcb91a`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-linux-386.tar.gz) | `5d1bd3ecc96f9e1cb9f20cef88c5aa2ec9c09370e8892557fc8a7cfe3cba595b`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-linux-amd64.tar.gz) | `94b2c9cd29981a8e150c187193bab0d8c0b6e906260f837367feff99860a6376`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-linux-arm64.tar.gz) | `a7554e496403b50c12f5dbfaa00f2b887773905894ae5c330e2accd7b7e137c9`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-linux-arm.tar.gz) | `5a3414f4b47c84b173c879379d90b447af0540730bb86f10baf6d6933b09d41d`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-linux-ppc64le.tar.gz) | `904bab541dd8f1236d5e47f97cd2509c649f629fdc3af88a3968ca3c5575886d`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-linux-s390x.tar.gz) | `d4a85694f796e4e1c14e6bddc295d9f776001fd8ac92ed32565606b964a843b0`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-windows-386.tar.gz) | `47258f6fc7fbd17ac1ddb38387adc5a2ddc2e39c5792cf3d354f9b19d373a6b2`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-client-windows-amd64.tar.gz) | `e8c2957688acf19463e28d39cc1b4b1e9a12b3f10101dff179d1d63754b34236`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-server-linux-amd64.tar.gz) | `7bbf43f81f5dbc3729c1956705d95c218b848591d03789d48f10e58fa865a0ba`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-server-linux-arm64.tar.gz) | `f725f491a8998bdf164470029441135336ec0414360d6b57a5d8daf73d09334f`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-server-linux-arm.tar.gz) | `9026ca6fdbffef1d02409a86649e4dd0a7667ff6c27df318a3d851c271fb38d0`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-server-linux-ppc64le.tar.gz) | `cd3b1b693b4e8225f78751bff533024785b0e20c2c51228956692db2a21d9f60`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-server-linux-s390x.tar.gz) | `9cd42506f4891be4691b7f4a8be7b894109dca54d0e9130651bc869101d7ed1f`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-node-linux-amd64.tar.gz) | `962f327f9a7b038c3b125a6cd06b690012fa38c827de0dae75955bb2be717126`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-node-linux-arm64.tar.gz) | `e09af9f1130f8e1d4f6b45ec79eedb94e98354edb813b635c0dc097437834a1b`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-node-linux-arm.tar.gz) | `7647ec0a51308ca73e4c4eb4cbe09f2f9609c809e530d129a718d960a25d339d`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-node-linux-ppc64le.tar.gz) | `439b2135b179b699ca951a2d619629583d92dbdfb60b3920a1fa872b4cd65b6d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-node-linux-s390x.tar.gz) | `eed95c80bddad4a67d81c036b0d047dfb7bef8fb13a4e1b4817c1f2a595b993e`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.6/kubernetes-node-windows-amd64.tar.gz) | `611a92f9ab4f6dd48349843eec844b6abf861d76a151cce91b216a56bb6c821f`

## Changelog since v1.6.5

### Action Required

* Azure: Change container permissions to private for provisioned volumes. If you have existing Azure volumes that were created by Kubernetes v1.6.0-v1.6.5, you should change the permissions on them manually. ([#47605](https://github.com/kubernetes/kubernetes/pull/47605), [@brendandburns](https://github.com/brendandburns))

### Other notable changes

* Bump GLBC version to 0.9.5 - fixes [loss of manually modified GCLB health check settings](https://github.com/kubernetes/kubernetes/issues/47559) upon upgrade from pre-1.6.4 to either 1.6.4 or 1.6.5. ([#47567](https://github.com/kubernetes/kubernetes/pull/47567), [@nicksardo](https://github.com/nicksardo))



# v1.6.5

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Known Issues for v1.6.5

* If you use the [GLBC Ingress Controller](https://github.com/kubernetes/ingress/tree/master/controllers/gce),
  upgrading an existing pre-v1.6.4 cluster to Kubernetes v1.6.5 will cause an unintentional
  [overwrite of manual edits to GCP Health Checks](https://github.com/kubernetes/ingress/issues/842)
  managed by the GLBC Ingress Controller. This can cause the health checks to start failing,
  requiring you to reapply the manual edits.
  * This issue does not affect clusters that were already running Kubernetes v1.6.4 or higher.
  * This issue does not affect Health Checks that were left in the configuration
    [originally set by the GLBC Ingress Controller](https://github.com/kubernetes/ingress/tree/master/controllers/gce#health-checks).

## Downloads for v1.6.5


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes.tar.gz) | `e497ddd9d0fb03a71babd6c7f879951ba8e2d791c9884613d60794f7df9a5a51`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-src.tar.gz) | `3971e41435f6e22914b603ea56bec72f78673fca9f5a5436a4beda7957dc24e1`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-darwin-386.tar.gz) | `af2290d1c3c923a6f873736739e0fc381328d19eb726419a17f2ae51e3ac2b45`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-darwin-amd64.tar.gz) | `8a138c2487871807bc8461a5bb0867d75cf9da228ecb6acdc5d22c08b2ed107d`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-linux-386.tar.gz) | `3f6867dd51e7838f58cf7e95174ef914d8d280ff275ac56cc8b285566ce30996`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-linux-amd64.tar.gz) | `8f38e71b1c68d69299af86ab4432297ae0d72cdee1d1e1c9975d77e448096c9c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-linux-arm64.tar.gz) | `1d01ae4423eb9794d09202ff4f24207c9d62b32a1b8f4906801a66fcd70b8fa5`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-linux-arm.tar.gz) | `438a8b4388960fe48e87aa7e97954f1cf9f9cc5c957eee108c3cc8040786fdce`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-linux-ppc64le.tar.gz) | `3b64d6ce09ffb65d3fe8c4101309c3b39fdab3643d89f91e89445c8e3279884e`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-linux-s390x.tar.gz) | `f5fb3ddc6a6203ae68b0213624bbfa12b114a70a38e85151272273cbd8fd4fbd`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-windows-386.tar.gz) | `6ddc9b300746eefdc5d71aae193dea171115dab9db00010d416728d3f2035f19`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-client-windows-amd64.tar.gz) | `95854266bf64b84b59d1e6a3ca0501cf3c85fbd0258cb540893d5771aca74ace`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-server-linux-amd64.tar.gz) | `68809d34d3429685eaafedf398f690bba1bcc1f793cd2d8193559b90492c68b1`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-server-linux-arm64.tar.gz) | `96bf4da5cbaa8f7b0f8909276005e0766ca4fa30431396ba30b9e77be1abb7f0`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-server-linux-arm.tar.gz) | `2f9c5275a6a2b5b10416a3e76f64e996851f486eb6b2dbe0f106b81bb63e63a9`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-server-linux-ppc64le.tar.gz) | `f47bc83530dc7448879e7d11c2ba1613adc318fd6c1cbba76e5d7181d850667c`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-server-linux-s390x.tar.gz) | `ebbd82df12da7470299e9877185797def86b859a44e72486e7be995c01eae56c`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-node-linux-amd64.tar.gz) | `f842694701f8966cdfceca5f8f9d2b49bc84baf7446b67f7bf23b7581c245cc3`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-node-linux-arm64.tar.gz) | `4efc4c8d9df77ad66763ce5cc39650ea1ebd43dd4f789622c93b0aa872f7a186`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-node-linux-arm.tar.gz) | `6f4e2b764aec5c458e8bf28159190c0c725e20ab94cf9ced3279dc75caa3fe21`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-node-linux-ppc64le.tar.gz) | `ba0e8ea11050273dbdf7b6d179251a95021b85000523e539c4940312586fd716`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-node-linux-s390x.tar.gz) | `b6555e9a94a38e8bda502ec3eb951d4d854ebe8c44ba31320882a497703ab2bf`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.5/kubernetes-node-windows-amd64.tar.gz) | `0eaf3789c93996a45c74b30cc8a4d7169a639b7cf3fcf589ec387a0cfd0782d8`

## Changelog since v1.6.4

### Other notable changes

* Fix iSCSI iSER mounting. ([#47281](https://github.com/kubernetes/kubernetes/pull/47281), [@mtanino](https://github.com/mtanino))
* Added exponential backoff to Azure cloudprovider ([#46660](https://github.com/kubernetes/kubernetes/pull/46660), [@jackfrancis](https://github.com/jackfrancis))
* Update kube-dns to 1.14.2. ([#45684](https://github.com/kubernetes/kubernetes/pull/45684), [@bowei](https://github.com/bowei))
* Fix log spam due to unnecessary status update when node is deleted. ([#45923](https://github.com/kubernetes/kubernetes/pull/45923), [@verult](https://github.com/verult))
* Poll all active pods periodically for volumes to fix out of order pod & node addition events. Fixes volumes not getting detached after controller restart. ([#42033](https://github.com/kubernetes/kubernetes/pull/42033), [@NickrenREN](https://github.com/NickrenREN))
* iscsi storage plugin: Fix dangling session when using multiple target portal addresses. ([#46239](https://github.com/kubernetes/kubernetes/pull/46239), [@mtanino](https://github.com/mtanino))
* The namespace API object no longer supports the deletecollection operation, which was previously allowed by mistake and did not respect expected namespace lifecycle semantics. ([#47098](https://github.com/kubernetes/kubernetes/pull/47098), [@liggitt](https://github.com/liggitt))
* Azure: Fix support for multiple `loadBalancerSourceRanges`, and add support for UDP ports and the Service spec's `sessionAffinity`. ([#45523](https://github.com/kubernetes/kubernetes/pull/45523), [@colemickens](https://github.com/colemickens))
* Fix the bug where container cannot run as root when SecurityContext.RunAsNonRoot is false. ([#47009](https://github.com/kubernetes/kubernetes/pull/47009), [@yujuhong](https://github.com/yujuhong))
* Remove broken getvolumename and pass PV or volume name to attach call ([#46249](https://github.com/kubernetes/kubernetes/pull/46249), [@chakri-nelluri](https://github.com/chakri-nelluri))
* Portworx volume driver no longer has to run on the master. ([#45518](https://github.com/kubernetes/kubernetes/pull/45518), [@harsh-px](https://github.com/harsh-px))
* Upgrade golang version to 1.7.6 ([#46405](https://github.com/kubernetes/kubernetes/pull/46405), [@cblecker](https://github.com/cblecker))
* kube-proxy handling of services with no endpoints now applies to both INPUT and OUTPUT chains, preventing socket leak. ([#43972](https://github.com/kubernetes/kubernetes/pull/43972), [@thockin](https://github.com/thockin))
* Fix kube-apiserver crash when patching TPR data ([#44612](https://github.com/kubernetes/kubernetes/pull/44612), [@nikhita](https://github.com/nikhita))
* vSphere cloud provider: Report same Node IP as both internal and external. ([#45201](https://github.com/kubernetes/kubernetes/pull/45201), [@abrarshivani](https://github.com/abrarshivani))
* Kubelet: Fix image garbage collector attempting to remove in-use images. ([#46121](https://github.com/kubernetes/kubernetes/pull/46121), [@Random-Liu](https://github.com/Random-Liu))
* Add metrics exporter to the fluentd-gcp deployment ([#45096](https://github.com/kubernetes/kubernetes/pull/45096), [@crassirostris](https://github.com/crassirostris))
* Fix the bug where StartedAt time is not reported for exited containers. ([#45977](https://github.com/kubernetes/kubernetes/pull/45977), [@yujuhong](https://github.com/yujuhong))
* Enable basic auth username rotation for GCI ([#44590](https://github.com/kubernetes/kubernetes/pull/44590), [@ihmccreery](https://github.com/ihmccreery))
* vSphere cloud provider: Filter out IPV6 node addresses. ([#45181](https://github.com/kubernetes/kubernetes/pull/45181), [@BaluDontu](https://github.com/BaluDontu))
* vSphere cloud provider: Fix volume detach on node failure. ([#45569](https://github.com/kubernetes/kubernetes/pull/45569), [@divyenpatel](https://github.com/divyenpatel))
* Job controller: Retry with rate-limiting if a Pod create/delete operation fails. ([#45870](https://github.com/kubernetes/kubernetes/pull/45870), [@tacy](https://github.com/tacy))
* Ensure that autoscaling/v1 is the preferred version for API discovery when autoscaling/v2alpha1 is enabled. ([#45741](https://github.com/kubernetes/kubernetes/pull/45741), [@DirectXMan12](https://github.com/DirectXMan12))
* Add support for Azure internal load balancer. ([#45690](https://github.com/kubernetes/kubernetes/pull/45690), [@jdumars](https://github.com/jdumars))
* Fix `kubectl delete --cascade=false` for resources that don't have legacy overrides to orphan by default. ([#45713](https://github.com/kubernetes/kubernetes/pull/45713), [@kargakis](https://github.com/kargakis))
* Fix erroneous FailedSync and FailedMount events being periodically and indefinitely posted on Pods after kubelet is restarted ([#44781](https://github.com/kubernetes/kubernetes/pull/44781), [@wongma7](https://github.com/wongma7))
* fluentd will tolerate all NoExecute Taints when run in gcp configuration. ([#45715](https://github.com/kubernetes/kubernetes/pull/45715), [@gmarek](https://github.com/gmarek))
* Fix [Deployments with Recreate strategy not waiting for Pod termination](https://github.com/kubernetes/kubernetes/issues/27362). ([#45639](https://github.com/kubernetes/kubernetes/pull/45639), [@ChipmunkV](https://github.com/ChipmunkV))
* vSphere cloud provider: Remove the dependency of login information on worker nodes for vsphere cloud provider. ([#43545](https://github.com/kubernetes/kubernetes/pull/43545), [@luomiao](https://github.com/luomiao))
* vSphere cloud provider: Fix fetching of VM UUID on Ubuntu 16.04 and Fedora. ([#45311](https://github.com/kubernetes/kubernetes/pull/45311), [@divyenpatel](https://github.com/divyenpatel))



# v1.6.4

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6.3/examples)

## Known Issues for v1.6.4

* If you use the [GLBC Ingress Controller](https://github.com/kubernetes/ingress/tree/master/controllers/gce),
  upgrading an existing cluster to Kubernetes v1.6.4 will cause an unintentional
  [overwrite of manual edits to GCP Health Checks](https://github.com/kubernetes/ingress/issues/842)
  managed by the GLBC Ingress Controller. This can cause the health checks to start failing,
  requiring you to reapply the manual edits.
  * This issue does not affect clusters that start out with Kubernetes v1.6.4 or higher.
  * This issue does not affect Health Checks that were left in the configuration
    [originally set by the GLBC Ingress Controller](https://github.com/kubernetes/ingress/tree/master/controllers/gce#health-checks).

## Downloads for v1.6.4


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes.tar.gz) | `fef6a97be8195fee1108b40acbd7bea61ef5244b8be23e799d2d01ee365907dd`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-src.tar.gz) | `2915465e9b389c5af0fa660f6e7cbc36a416d1286094201e2a2da5b084a37cb3`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-darwin-386.tar.gz) | `e2db37a1cf3dff342e9ba25506c96edba0cbc9b65984dfe985a7ab45df64f93e`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-darwin-amd64.tar.gz) | `0d49df4b06f76b5a6e168f72ac0088194d4267cc888880f7d0f23e558cd0ee32`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-linux-386.tar.gz) | `5e218cc7f01d6fa71d0a10a30eea2724ee111db3bbae5a03f0c560cd0d24a5df`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-linux-amd64.tar.gz) | `4c8dbd03a66d871f03592e84ed98d205d2d0e0c0f6d48c7b60f3e9840ad04df8`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-linux-arm64.tar.gz) | `9bf29b0f8bdde972d62556bdd14622199f979f7dcf56d3948d429b1d73feca08`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-linux-arm.tar.gz) | `bbca1efe8fb95c6df7b330054776ce619652ba9d4c5428eabef9c435c61d1d9a`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-linux-ppc64le.tar.gz) | `7aa02e261f36a816dc1c7c898e16d732d9199d827ac4828f8e120b4a9ce5aa05`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-linux-s390x.tar.gz) | `531d00c43a49bb72365f2d6c1f31ad99ff09893e41f6b28d21980dbdd3ab0de4`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-windows-386.tar.gz) | `256fa2ffa77a1107e87a5a232bf8aa245afbcb5d383eda18f19f3fedbbad9a69`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-client-windows-amd64.tar.gz) | `c8a97087b81defdc513a9fe4aaf317d10ad6fc3368170465dd4ea64c23655939`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-server-linux-amd64.tar.gz) | `76a1d6dbce630b50fd3a5f566fcea6ef1a04996cf4f4c568338a3db0d3b6a3d5`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-server-linux-arm64.tar.gz) | `8b731307138a71ae90e025cb85ec7b4ac9179ebd8923f7abd1c839a2966ff2b0`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-server-linux-arm.tar.gz) | `0d3039f22cdc36526257f211c55099552b8d55cda25a05405f2701c869bb4be2`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-server-linux-ppc64le.tar.gz) | `6de3a85392ff65c019fd90173f1219a41f56559aebd07b18ed497e46645fcffc`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-server-linux-s390x.tar.gz) | `622a137c06a9fda23ec5941dd41607564804eeede0e6d3976cda6cc136e010c6`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-node-linux-amd64.tar.gz) | `df40c178ffbd92376e98dd258113e35c0a46a8313f188d34d391a834baeb1da8`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-node-linux-arm64.tar.gz) | `a27b15a0edcfd78470db54181ea2c2c942b5d4489b6f7a4ba78bb1fac34f8fa8`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-node-linux-arm.tar.gz) | `2b4dceee70ba7b508a0acc3cc5ce072d92f9c32c1a6961911b93a5da02ace9f7`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-node-linux-ppc64le.tar.gz) | `c5e01f9f7de6ae2d73004bbcd288f5c942414b6639099c1bf52a98e592147745`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-node-linux-s390x.tar.gz) | `eded4d2b94c9c22ae6c865e32a7965c1228d564aebf90c533607c716ed89b676`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.4/kubernetes-node-windows-amd64.tar.gz) | `da561264f5665fe1ae9a41999731403b147b91c3a5c47439eb828ed688b0738f`

## Changelog since v1.6.3

### Other notable changes

* Fix kubelet panic during disk eviction introduced in 1.6.3. ([#46049](https://github.com/kubernetes/kubernetes/pull/46049), [@enisoc](https://github.com/enisoc))
* Fix pods failing to start if they specify a file as a volume subPath to mount. ([#46046](https://github.com/kubernetes/kubernetes/pull/46046), [@enisoc](https://github.com/enisoc))



# v1.6.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Known Issues for v1.6.3

* This release introduced a regression when using [subPath](https://kubernetes.io/docs/concepts/storage/volumes/#using-subpath).
  If the `subPath` is a file rather than a directory, Pods may fail to start ([#45613](https://github.com/kubernetes/kubernetes/issues/45613)).

  **Do not upgrade to v1.6.3** if your cluster may run Pods with such subPaths.

## Downloads for v1.6.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes.tar.gz) | `0af5914fcea36b3c65c8185f31e94b2839eaed52dfdd666d31dfa14534a7d69c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-src.tar.gz) | `0d3cbc716b0d08bf3be779ddd096df965609b5bcb55c8b9ea084c6bb2d6ef1fd`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-darwin-386.tar.gz) | `2f2f58e8853eef7df293e579e8c94e1b6e75318b08bd1bf5747685ad8d16ebe2`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-darwin-amd64.tar.gz) | `122c20e2e92c1ed4a592c8a3851867452acff181ffe5251e8fee0ec8284704ac`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-linux-386.tar.gz) | `47c970bbe75a41634b9e5d0ae109a01f4823fdb83cf1c6c40a1ad4034b6d2764`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-linux-amd64.tar.gz) | `ae141e0cd011889c4468b5b8b841d8cd62c1802c4ccba4dfd8c42beaccaf7e75`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-linux-arm64.tar.gz) | `07a83a7f7df555e859f4f8e143274f9ed1f475d597f01d1c79e95cdfda289c94`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-linux-arm.tar.gz) | `4a0b89b4985e651a1c29590ae2ea16ea00203d7cbad7ffc8a541b8a13569e1be`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-linux-ppc64le.tar.gz) | `1c0116942a61580da717845c9b7fc69aa58438aaa176888cd3e57267c9c717c0`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-linux-s390x.tar.gz) | `94307d778e0819dc5a64e12d794e95a028207d06603204d82610f369e040ce67`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-windows-386.tar.gz) | `322d2db5dadd4b388c785d1caf651bcc76c566afb6d19e84bdf6abcc40fa19d4`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-client-windows-amd64.tar.gz) | `9ef35675f7cd6acb81fa69ded37174e9a55cc0f58a2f8159bfc5512230495763`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-server-linux-amd64.tar.gz) | `22eadeff9c3a45bf4d490ffca50bd257b6c282a3d16b5b8b40b8c31070a94de1`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-server-linux-arm64.tar.gz) | `2f9d976dd6d436a8258a5eb0c4a270329746f4331db607acc6b893f81f25e1c9`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-server-linux-arm.tar.gz) | `11f6a859438805250b84b415427df5f07d44a2a2ffb37591b6cdc6c8dc317382`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-server-linux-ppc64le.tar.gz) | `670fc921b50cca1c4fc20fbe58be640eeae766d38f6b2053b68c1a1293e88ba0`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-server-linux-s390x.tar.gz) | `c5f2358bf81ea34fc01dbe5b639f03a10b5799ad75f8465863bb5c2b797b4f0b`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-node-linux-amd64.tar.gz) | `428332868f42f77e02f337779a18a6332894b27f2432c5b804a8ff753895b883`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-node-linux-arm64.tar.gz) | `a8bdefd9c0ba9a247536a5a1bb7481b7a937cf39951256be774e45b8e40250cc`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-node-linux-arm.tar.gz) | `6b5aa71b27c0524b714489de80b2100e66bef99112f452aeaedcde1f890d2916`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-node-linux-ppc64le.tar.gz) | `34afa6e39ff8eb8a6f8f29874b6a3e5426fa6b64cc1b0f4466b17ae0f91f71ad`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-node-linux-s390x.tar.gz) | `170953b40e70242249c31e32456de73dacbed54e537efa4275d273441df98f7d`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.3/kubernetes-node-windows-amd64.tar.gz) | `410f175a47335b93f212cff5f3921a062ca6e945fa336d3cf0971f9bebba73e5`

## Changelog since v1.6.2

### Other notable changes

* Bump GLBC version to 0.9.3 ([#45055](https://github.com/kubernetes/kubernetes/pull/45055), [@nicksardo](https://github.com/nicksardo))
* kubeadm: Fix invalid assign statement so it is possible to register the master kubelet with other initial Taints ([#45376](https://github.com/kubernetes/kubernetes/pull/45376), [@luxas](https://github.com/luxas))
* Fix a bug that caused invalid Tolerations to be applied to Pods created by DaemonSets. ([#45349](https://github.com/kubernetes/kubernetes/pull/45349), [@gmarek](https://github.com/gmarek))
* Bump cluster autoscaler to v0.5.4, which fixes scale down issues with pods ignoring SIGTERM. ([#45483](https://github.com/kubernetes/kubernetes/pull/45483), [@mwielgus](https://github.com/mwielgus))
* Fixes and minor cleanups to pod (anti)affinity predicate ([#45098](https://github.com/kubernetes/kubernetes/pull/45098), [@wojtek-t](https://github.com/wojtek-t))
* StatefulSet: Fix to fully preserve restart and termination order when StatefulSet is rapidly scaled up and down. ([#45113](https://github.com/kubernetes/kubernetes/pull/45113), [@kow3ns](https://github.com/kow3ns))
* Fix some false negatives in detection of meaningful conflicts during strategic merge patch with maps and lists. ([#43469](https://github.com/kubernetes/kubernetes/pull/43469), [@enisoc](https://github.com/enisoc))
* cluster-autoscaler: Fix duplicate writing of logs. ([#45017](https://github.com/kubernetes/kubernetes/pull/45017), [@MaciekPytel](https://github.com/MaciekPytel))
* Fixes a bug where pods were evicted even after images are successfully deleted. ([#44986](https://github.com/kubernetes/kubernetes/pull/44986), [@dashpole](https://github.com/dashpole))
* CRI: respect container's stopping timeout. ([#44998](https://github.com/kubernetes/kubernetes/pull/44998), [@feiskyer](https://github.com/feiskyer))
* Fix problems with scaling up the cluster when unschedulable pods have some persistent volume claims. ([#44860](https://github.com/kubernetes/kubernetes/pull/44860), [@mwielgus](https://github.com/mwielgus))
* Exclude nodes labeled as master from LoadBalancer / NodePort; restores documented behaviour. ([#44745](https://github.com/kubernetes/kubernetes/pull/44745), [@justinsb](https://github.com/justinsb))
* Fix for [scaling down remaining good replicas when a failed Deployment is paused](https://github.com/kubernetes/kubernetes/issues/44436). ([#44616](https://github.com/kubernetes/kubernetes/pull/44616), [@kargakis](https://github.com/kargakis))
* kubectl commands run inside a pod using a kubeconfig file now use the namespace specified in the kubeconfig file, instead of using the pod namespace. If no kubeconfig file is used, or the kubeconfig does not specify a namespace, the pod namespace is still used as a fallback. ([#44570](https://github.com/kubernetes/kubernetes/pull/44570), [@liggitt](https://github.com/liggitt))
* Restored the ability of kubectl running inside a pod to consume resource files specifying a different namespace than the one the pod is running in. ([#44862](https://github.com/kubernetes/kubernetes/pull/44862), [@liggitt](https://github.com/liggitt))
* Fix false positive "meaningful conflict" detection for strategic merge patch with integer values. ([#44788](https://github.com/kubernetes/kubernetes/pull/44788), [@enisoc](https://github.com/enisoc))
* Fix insufficient permissions to read/write volumes when mounting them with a subPath. ([#43775](https://github.com/kubernetes/kubernetes/pull/43775), [@wongma7](https://github.com/wongma7))
* vSphere cloud provider: Allow specifying fstype in storage class. ([#41929](https://github.com/kubernetes/kubernetes/pull/41929), [@abrarshivani](https://github.com/abrarshivani))
* vSphere cloud provider: Allow specifying VSAN Storage Capabilities during dynamic volume provisioning. ([#42974](https://github.com/kubernetes/kubernetes/pull/42974), [@BaluDontu](https://github.com/BaluDontu))



# v1.6.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes.tar.gz) | `240f66a98bf75246b53ef7c1fa3a2be36a92cbc173bc8626e7bc4427bef9ce6a`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-src.tar.gz) | `dbf19a8f2e50b3e691eeba0c418fe057f1ea8527b8c0194ba9749c12c801b24b`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-darwin-386.tar.gz) | `3b1437cbc9d10e5466c83304c54ab06f5a880e0b047e2b0ea775530ee893b6b6`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-darwin-amd64.tar.gz) | `e3dad0848b3d6c1737199d0704c692e74bf979e6a81fabea79c5b2499ca3fa73`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-linux-386.tar.gz) | `962f576e67f13f8f8afc958f89f0371c7496b2540372ef7f8e1bd0e43a67e829`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-linux-amd64.tar.gz) | `f8ef17b8b4bb8f6974fa2b3faa992af3c39ad318c30bdfe1efab957361d8bdfe`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-linux-arm64.tar.gz) | `9582e6783e7effa10b0af2f662d1bc4471bbf8205d9df07a543edb099ba7f27c`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-linux-arm.tar.gz) | `165b642ba6900f7d779bc6a27f89ccdb09eefcbb310a8bc5f6d101db27e2e9cc`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-linux-ppc64le.tar.gz) | `514a308ccfb978111a74b5bf843cf6862464743f0f4e2aaffada33add4c2bb0f`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-linux-s390x.tar.gz) | `e030593109a369bc3288c9f47703843248dbe4a9ade496f936d8cc355a874ba6`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-windows-386.tar.gz) | `a2b0053de6b62d09123d8fcc1927a8cf9ab2a5a312794a858e7b423f4ffdbe3e`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-client-windows-amd64.tar.gz) | `eafdaa29a70d1820be0dc181074c5788127996402bbd5af53b0b765ed244e476`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-server-linux-amd64.tar.gz) | `016bc4db69a8f90495e82fbe6e5ec9a12e56ecab58a8eb2e5471bf9cab827ad2`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-server-linux-arm64.tar.gz) | `1985596d769656d68ec692030dd31bbec8081daf52ddaef6a2a7af7d6b1f7876`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-server-linux-arm.tar.gz) | `e0d4c53c55de5c100973379005aabe1441004ed4213b96a6e492c88d5a9b7f49`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-server-linux-ppc64le.tar.gz) | `652a8230c4511bc30d8f3a6ae11ebeee8c6d350648d879f8f2e1aa34777323d0`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.2/kubernetes-server-linux-s390x.tar.gz) | `1eab2d36beecf4f74e3b7b74734a75faf43ed6428d312aebe2e940df4cceb5ed`

## Changelog since v1.6.1

### Other notable changes

* Fix for [Windows kubectl sending full path to binary in User Agent](https://github.com/kubernetes/kubernetes/issues/44419). ([#44622](https://github.com/kubernetes/kubernetes/pull/44622), [@monopole](https://github.com/monopole))
* kube-apiserver now drops unneeded path information if an older version of Windows kubectl sends it. ([#44421](https://github.com/kubernetes/kubernetes/pull/44421), [@mml](https://github.com/mml))
* CRI: Fix kubelet failing to start when using rkt. ([#44569](https://github.com/kubernetes/kubernetes/pull/44569), [@yujuhong](https://github.com/yujuhong))
* CRI: `kubectl logs -f` now stops following when container stops, as it did pre-CRI. ([#44406](https://github.com/kubernetes/kubernetes/pull/44406), [@Random-Liu](https://github.com/Random-Liu))
* Azure cloudprovider: Reduce the polling delay for all azure clients to 5 seconds. This should speed up some operations at the cost of some additional quota. ([#43699](https://github.com/kubernetes/kubernetes/pull/43699), [@colemickens](https://github.com/colemickens))
* kubeadm: clean up exited containers and network checkpoints ([#43836](https://github.com/kubernetes/kubernetes/pull/43836), [@yujuhong](https://github.com/yujuhong))
* Fix corner-case with OnlyLocal Service healthchecks. ([#44313](https://github.com/kubernetes/kubernetes/pull/44313), [@thockin](https://github.com/thockin))
* Fix for [disabling federation controllers through override args](https://github.com/kubernetes/kubernetes/issues/42761). ([#44341](https://github.com/kubernetes/kubernetes/pull/44341), [@irfanurrehman](https://github.com/irfanurrehman))
* Fix for [federation failing to propagate cascading deletion](https://github.com/kubernetes/kubernetes/issues/44304). ([#44108](https://github.com/kubernetes/kubernetes/pull/44108), [@csbell](https://github.com/csbell))
* Fix for [failure to delete federation controllers with finalizers](https://github.com/kubernetes/kubernetes/issues/43828). ([#44084](https://github.com/kubernetes/kubernetes/pull/44084), [@nikhiljindal](https://github.com/nikhiljindal))
* Fix [transition between NotReady and Unreachable taints](https://github.com/kubernetes/kubernetes/issues/43444). ([#44042](https://github.com/kubernetes/kubernetes/pull/44042), [@gmarek](https://github.com/gmarek))
* AWS cloud provider: fix support running the master with a different AWS account or even on a different cloud provider than the nodes. ([#44235](https://github.com/kubernetes/kubernetes/pull/44235), [@mrIncompetent](https://github.com/mrIncompetent))
* kubeadm: Make `kubeadm reset` tolerant of a disabled docker service. ([#43951](https://github.com/kubernetes/kubernetes/pull/43951), [@luxas](https://github.com/luxas))
* Fix [broken service accounts when using dedicated service account key](https://github.com/kubernetes/kubernetes/issues/44285). ([#44169](https://github.com/kubernetes/kubernetes/pull/44169), [@mikedanese](https://github.com/mikedanese))
* Fix for kube-proxy healthcheck handling an update that simultaneously removes one port and adds another. ([#44246](https://github.com/kubernetes/kubernetes/pull/44246), [@thockin](https://github.com/thockin))
* Fix container hostPid settings when CRI is enabled. ([#44116](https://github.com/kubernetes/kubernetes/pull/44116), [@feiskyer](https://github.com/feiskyer))
* RBAC role and rolebinding auto-reconciliation is now performed only when the RBAC authorization mode is enabled. ([#43813](https://github.com/kubernetes/kubernetes/pull/43813), [@liggitt](https://github.com/liggitt))
* Fix incorrect conflict errors applying strategic merge patches to resources. ([#43871](https://github.com/kubernetes/kubernetes/pull/43871), [@liggitt](https://github.com/liggitt))
* Fixed an issue mounting the wrong secret into pods as a service account token. ([#44102](https://github.com/kubernetes/kubernetes/pull/44102), [@ncdc](https://github.com/ncdc))
* AWS cloud provider: allow to set KubernetesClusterID or KubernetesClusterTag in combination with VPC. ([#42512](https://github.com/kubernetes/kubernetes/pull/42512), [@scheeles](https://github.com/scheeles))
* kubeadm: Remove hard-coded default version when fetching stable version from the web fails. Fixes [kubeadm 1.6.1 deploying 1.6.0 control plane](https://github.com/kubernetes/kubeadm/issues/224). ([#43999](https://github.com/kubernetes/kubernetes/pull/43999), [@mikedanese](https://github.com/mikedanese))
* Fixed recognition of default storage class in DefaultStorageClass admission plugin. ([#43945](https://github.com/kubernetes/kubernetes/pull/43945), [@mikkeloscar](https://github.com/mikkeloscar))
* Fix [Kubelet panic when Docker is not running](https://github.com/kubernetes/kubernetes/issues/44027). ([#44047](https://github.com/kubernetes/kubernetes/pull/44047), [@yujuhong](https://github.com/yujuhong))
* Fix [vSphere volumes in SELinux environment](https://github.com/kubernetes/kubernetes/issues/42972). ([#42973](https://github.com/kubernetes/kubernetes/pull/42973), [@gnufied](https://github.com/gnufied))
* Fix bug with error "Volume has no class annotation" when deleting a PersistentVolume. ([#43982](https://github.com/kubernetes/kubernetes/pull/43982), [@jsafrane](https://github.com/jsafrane))



# v1.6.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes.tar.gz) | `f1634e22ee3fe8af5c355c3f53d1b93f946490addfab029e19acf5317c51cd38`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-src.tar.gz) | `b818f29661dd34db77d1e46c6ba98df6d35906dbc36ac1fdfe45f770b0f695c1`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-darwin-386.tar.gz) | `6eb7a0749de4c66d66630ac831f9a0aa73af9be856776c428d6adb3e07479d4a`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-darwin-amd64.tar.gz) | `05715224efdda0da3241960ec6cc71c2b008d3a53d217e5f90b159b5274db240`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-linux-386.tar.gz) | `7608a4754e48297dac8be9e863c429676f35afb97a9a10897e15038df6499a2e`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-linux-amd64.tar.gz) | `21e85cd3388b131fd1b63b06ea7ace8eef9555b7c558900b0cf1f9a3f2733e9a`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-linux-arm64.tar.gz) | `b798e4b440df52e35809310147f8678a1d9b822dce85212fcf382d19ec2bd8c3`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-linux-arm.tar.gz) | `3227e745db3986a6be9c16c8ffb4f40ce604a400c680e2e6ff92368e72a597c3`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-linux-ppc64le.tar.gz) | `ab7c9b2516d3cda8b4c236e00a179448e0787670cfd20c66dfb1b0c6c73ef0db`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-linux-s390x.tar.gz) | `9fbcb5f1b092573e5db5188689d7709a03b2bfdae635f61b5dbf72ae9dde2aaf`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-windows-386.tar.gz) | `306566c6903111769f01b0d05ba66aed63c133501adf49ef8daa38cc6a78097d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-client-windows-amd64.tar.gz) | `5ca89e1672fd29a13a7cb997480216643e98afeba4d19ab081877281d0db8060`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-server-linux-amd64.tar.gz) | `3e5c7103f44f20a95db29243a43f04aca731c8a4d411c80592ea49f7550d875c`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-server-linux-arm64.tar.gz) | `3fad77963f993396786e1777aecb770572b8b06cc3fe985c688916a70ffee2fd`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-server-linux-arm.tar.gz) | `4b981751da6a0bf471e52e55b548d62c038f7e6d8ed628b8177389f24cfd0434`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-server-linux-ppc64le.tar.gz) | `7b4bdf49cc2510af81205f2a65653a577fc79623c76c7ed3c29c2fbe1d835773`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.1/kubernetes-server-linux-s390x.tar.gz) | `3c55f1322ca39b7acb4914dd174978b015c1124e1ddd5431ec14c97b1b45f69f`

## Changelog since v1.6.0

### Other notable changes

* Fix a deadlock in kubeadm master initialization. ([#43835](https://github.com/kubernetes/kubernetes/pull/43835), [@mikedanese](https://github.com/mikedanese))
* Use Cluster Autoscaler 0.5.1, which fixes an issue in Cluster Autoscaler 0.5 where the cluster may be scaled up unnecessarily. Also the status of Cluster Autoscaler is now exposed in kube-system/cluster-autoscaler-status config map. ([#43745](https://github.com/kubernetes/kubernetes/pull/43745), [@mwielgus](https://github.com/mwielgus))



# v1.6.0

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.0


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes.tar.gz) | `e89318b88ea340e68c427d0aad701e544ce2291195dc1d5901222e7bae48f03b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-src.tar.gz) | `0b03d27e1c7af3be5d94ecd5f679e5f55588d801376cf1ae170d9ec0a229b1e2`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-darwin-386.tar.gz) | `274277a67a85e9081d9fee5e763ed7c3dd096acf641c31a9ccc916a3981fead2`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-darwin-amd64.tar.gz) | `af8d1aa034721b31fd14f7b93f5ef16f8dbac4fdcd9e312c3c61e6cf03295057`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-linux-386.tar.gz) | `4c6a3c12f131c3c88612f888257d00a3cc7fef67947757b897b0d8be9fab547c`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-linux-amd64.tar.gz) | `4a5daf0459dffc24bf0ccbb2fc881f688008e91ae41fde961b81d09b0801004c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-linux-arm64.tar.gz) | `91d5e31407140a55cf00c0dc6e20aa8433cc918615dedd842637585e81694ebd`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-linux-arm.tar.gz) | `985fecd7fb94b42c25b8a56efde1831b2616a6792d3d5a02549248e01ce524ed`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-linux-ppc64le.tar.gz) | `303279f935289cadfb97a6a5d3f58b0da67d1b88b5ed049e6a98fc203b7b9d52`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-linux-s390x.tar.gz) | `2bd216c6b7d4f09de02c3b5d20021124f7d04f483ab600b291c515386003ca74`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-windows-386.tar.gz) | `11d5459b0d413a25045c55ce3548d30616ddc2d62451280d3299baa9f3e3e014`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-client-windows-amd64.tar.gz) | `84eba6f2b2b82a95397688b1e2ca4deb8d7daf1f8a40919fa0312741ca253799`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-server-linux-amd64.tar.gz) | `3625b63d573aa4d28eaa30b291017f775f2ddc0523f40d25023ad1da9c30390e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-server-linux-arm64.tar.gz) | `906496c985d4d836466b73e1c9e618ea8ce07f396aba3a96edcdc6045e0ab4e3`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-server-linux-arm.tar.gz) | `3b63f481156f57729bc8100566d8b3d7856791e5b36bb70042e17d21f11f8d5d`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-server-linux-ppc64le.tar.gz) | `382666b3892fd4d89be5e4bb052dd0ef0d1c1d213c1ae7a92435083a105064fd`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0/kubernetes-server-linux-s390x.tar.gz) | `e15de8896bd84a9b74756adc1a2e20c6912a65f6ff0a3f3dfabc8b463e31d19b`

## WARNING: etcd backup strongly recommended

Before updating to 1.6, you are strongly recommended to back up your etcd data.
Please consult the installation procedure you are using (kargo, kops, kube-up,
kube-aws, kubeadm etc) for specific advice.

1.6 encourages etcd3, and switching from etcd2 to etcd3 involves a full
migration of data between different storage engines.  You must stop the API
from writing to etcd during an etcd2 -> etcd3 migration.  HA installations cannot
be migrated at the current time using the official Kubernetes procedure.

1.6 will also default to protobuf encoding if using etcd3.  **This change is
irreversible.**  To rollback, you must restore from a backup made before the
protobuf/etcd3 switch, and any changes since the backup will be lost.  As 1.5
does not support protobuf encoding, if you roll back to 1.5 after upgrading to
protobuf you will be forced to restore from backup, and you will lose any changes
since you converted to protobuf.  After conversion to protobuf, you should
validate the correct operation of your cluster thoroughly before returning it
to normal operation.

Backups are always a good precaution, particularly around upgrades, but this
upgrade has multiple known issues where the only remedy is to restore from
backup.

Also, please note:
- using `application/vnd.kubernetes.protobuf` as the media storage type for 1.6 is default but not required
- the ability to rollback to etcd2 can be preserved by setting the storage media type flag on `kube-apiserver`

  ```
  --storage-media-type application/json
  ```

  to continue to use `application/json` as the storage media type which can be changed to
  `application/vnd.kubernetes.protobuf` at a later time.

## Major updates and release themes

* Kubernetes now supports up to 5,000 nodes via etcd v3, which is enabled by default.
* [Role-based access control (RBAC)](https://kubernetes.io//docs/admin/authorization/rbac) has graduated to beta, and defines secure default roles for control plane, node, and controller components.
* The [`kubeadm` cluster bootstrap tool](https://kubernetes.io/docs/getting-started-guides/kubeadm/) has graduated to beta. Some highlights:
  * **WARNING:** A [known issue](https://github.com/kubernetes/kubernetes/issues/43815)
    in v1.6.0 causes `kubeadm init` to hang. Please use v1.6.1, which fixes the issue.
  * All communication is now over TLS
  * Authorization plugins can be installed by kubeadm, including the new default of RBAC
  * The bootstrap token system now allows token management and expiration
* The [`kubefed` federation bootstrap tool](https://kubernetes.io/docs/tutorials/federation/set-up-cluster-federation-kubefed/) has also graduated to beta.
* Interaction with container runtimes is now through the CRI interface, enabling easier integration of runtimes with the kubelet. Docker remains the default runtime via Docker-CRI (which moves to beta).
  * **WARNING:** A [known issue](https://github.com/kubernetes/kubernetes/issues/44041)
    in v1.6.0 causes `Pod.Spec.HostPid` (using the host PID namespace for the pod) to always
    be false. Please wait for v1.6.2, which will include a fix for this issue.
* Various scheduling features have graduated to beta:
  * You can now use [multiple schedulers](https://kubernetes.io/docs/tutorials/clusters/multiple-schedulers/)
  * [Nodes](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#node-affinity-beta-feature) and [pods](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#inter-pod-affinity-and-anti-affinity-beta-feature) now support affinity and anti-affinity
  * Advanced scheduling can be performed with [taints and tolerations](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/#taints-and-tolerations-beta-feature)
* You can now specify (per pod) how long a pod should stay bound to a node, when there is a node problem.
* Various storage features have graduated to GA:
  * StorageClass pre-installed and set as default on Azure, AWS, GCE, OpenStack, and vSphere
  * Configurable [Dynamic Provisioning](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#dynamic) and [StorageClass](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#storageclasses)
* DaemonSets [can now be updated by a rolling update](https://kubernetes.io/docs/tasks/manage-daemon/update-daemon-set).

## Action Required

### Certificates API
* Users of the alpha certificates API should delete v1alpha1 CSRs from the API before upgrading and recreate them as v1beta1 CSR after upgrading. ([#39772](https://github.com/kubernetes/kubernetes/pull/39772), [@mikedanese](https://github.com/mikedanese))

### Cluster Autoscaler
* If you are using (or planning to use) Cluster Autoscaler please wait for Kubernetes 1.6.1. In 1.6.0 Cluster Autoscaler
may occasionally increase the size of the cluster a bit more than it is actually needed, when there are
unschedulable pods, scale up is required and cloud provider is slow to set up networking for new nodes.
Anyway, the cluster should get back to the proper size after 10 min.

### Deployment
* Deployment now fully respects ControllerRef to avoid fighting over Pods and ReplicaSets. At the time of upgrade, **you must not have Deployments with selectors that overlap**, or else [ownership of ReplicaSets may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md#upgrading). ([#42175](https://github.com/kubernetes/kubernetes/pull/42175), [@enisoc](https://github.com/enisoc))

### Federation
* The --dns-provider argument of 'kubefed init' is now mandatory and does not default to `google-clouddns`. To initialize a Federation control plane with Google Cloud DNS, use the following invocation: 'kubefed init --dns-provider=google-clouddns' ([#42092](https://github.com/kubernetes/kubernetes/pull/42092), [@marun](https://github.com/marun))
* Cluster federation servers have changed the location in etcd where federated services are stored, so existing federated services must be deleted and recreated. Before upgrading, export all federated services from the federation server and delete the services. After upgrading the cluster, recreate the federated services from the exported data. ([#37770](https://github.com/kubernetes/kubernetes/pull/37770), [@enj](https://github.com/enj))

### Internal Storage Layer
* upgrade to etcd3 prior to upgrading to 1.6 **OR** explicitly specify `--storage-backend=etcd2 --storage-media-type=application/json` when starting the apiserver

### Node Components
* **Kubelet with the Docker-CRI implementation**
  * The Docker-CRI implementation is enabled by default.
  * It is not compatible with containers created by older Kubelets. It is
    recommended to drain your node before upgrade. If you choose to perform
    an in-place upgrade, the Kubelet will automatically restart all
    Kubernetes-managed containers on the node.
  * It is not compatible with CNI plugins that do not conform to the
    [error handling behavior in the spec](https://github.com/containernetworking/cni/blob/master/SPEC.md#network-configuration-list-error-handling).
    The plugins are being updated to resolve this issue ([#43488](https://github.com/kubernetes/kubernetes/issues/43488)).
    You can disable CRI explicitly (`--enable-cri=false`) as a
    temporary workaround.
      * The standard `bridge` plugin have been validated to interoperate with
         the new CRI + CNI code path.
      * If you are using plugins other than `bridge`, make sure you have
        updated custom plugins to the latest version that is compatible.
* **CNI plugins now affect node readiness**
  * Kubelet will now block node readiness until a CNI configuration file is
    present in `/etc/cni/net.d` or a given custom CNI configuration path.
    [This change](https://github.com/kubernetes/kubernetes/pull/43474) ensures
    kubelet does not start pods that require networking before
    [networking is ready](https://github.com/kubernetes/kubernetes/issues/43014).
    This change may require changes to clients that gate pod creation and/or
    scheduling on the node condition `type` `Ready` `status` being `True` for pods
    that need to run prior to the network being ready.
* **Enhance Kubelet QoS**:
  * Pods are placed under a new cgroup hierarchy by default. This feature requires
    draining and restarting the node as part of upgrades. To opt-out set
    `--cgroups-per-qos=false`.
  * If `kube-reserved` and/or `system-reserved` are specified, node allocatable
    will be enforced on all pods by default. To opt-out set
    `--enforce-node-allocatable=””`
  * Hard Eviction Thresholds will be subtracted from Capacity while calculating
    Node Allocatable. This will result in a reduction of schedulable capacity in
    clusters post upgrade where kubelet hard eviction has been turned on for
    memory. To opt-out set `--experimental-allocatable-ignore-eviction=true`.
  * More details on these feature here:
    https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/
* Drop the support for docker 1.9.x. Docker versions 1.10.3, 1.11.2, 1.12.6
  have been validated.
* The following deprecated kubelet flags are removed: `--config`, `--auth-path`,
  `--resource-container`, `--system-container`, `--reconcile-cidr`
* Remove the temporary fix for pre-1.0 mirror pods. Upgrade directly from
  pre-1.0 to 1.6 kubelet is not supported.
* Fluentd was migrated to Daemon Set, which targets nodes with
  beta.kubernetes.io/fluentd-ds-ready=true label. If you use fluentd in your
  cluster please make sure that the nodes with version 1.6+ contains this
  label.

### kubectl
* Running `kubectl taint` (alpha in 1.5) against a 1.6 server requires upgrading kubectl to version 1.6
* Running `kubectl taint` (alpha in 1.5) against a 1.5 server requires a kubectl version of 1.5
* Running `kubectl create secret` no longer accepts passing multiple values to a single --from-literal flag using comma separation
  * Update command invocations to pass separate --from-literal flags for each value

### RBAC
* Default ClusterRole and ClusterRoleBinding objects are automatically updated at server start to add missing permissions and subjects (extra permissions and subjects are left in place). To prevent autoupdating a particular role or rolebinding, annotate it with `rbac.authorization.kubernetes.io/autoupdate=false`. ([#41155](https://github.com/kubernetes/kubernetes/pull/41155), [@liggitt](https://github.com/liggitt))
* `v1beta1` RoleBinding/ClusterRoleBinding subjects changed `apiVersion` to `apiGroup` to fully-qualify a subject. ServiceAccount subjects default to an apiGroup of `""`, User and Group subjects default to an apiGroup of `"rbac.authorization.k8s.io"`. ([#41184](https://github.com/kubernetes/kubernetes/pull/41184), [@liggitt](https://github.com/liggitt))
* To create or update an RBAC RoleBinding or ClusterRoleBinding object, a user must: ([#39383](https://github.com/kubernetes/kubernetes/pull/39383), [@liggitt](https://github.com/liggitt))
    * 1. Be authorized to make the create or update API request
    * 2. Be allowed to bind the referenced role, either by already having all of the permissions contained in the referenced role, or by having the "bind" permission on the referenced role.
* The `--authorization-rbac-super-user` flag (alpha in 1.5) is deprecated; the `system:masters` group has privileged access ([#38121](https://github.com/kubernetes/kubernetes/pull/38121), [@deads2k](https://github.com/deads2k))
* special handling of the user `*` in RoleBinding and ClusterRoleBinding objects is removed in v1beta1. To match all users, explicitly bind to the group `system:authenticated` and/or `system:unauthenticated`. Existing v1alpha1 bindings to the user `*` are automatically converted to the group `system:authenticated`. ([#38981](https://github.com/kubernetes/kubernetes/pull/38981), [@liggitt](https://github.com/liggitt))

### Scheduling
* **Multiple schedulers**
  * Modify your PodSpecs that currently use the `scheduler.alpha.kubernetes.io/name` annotation on Pod, to instead use the `schedulerName` field in the PodSpec.
  * Modify any custom scheduler(s) you have written so that they read the `schedulerName` field on Pod instead of the `scheduler.alpha.kubernetes.io/name` annotation.
  * Note that you can only start using the `schedulerName` field **after** you upgrade to 1.6; it is not recognized in 1.5.

* **Node affinity/anti-affinity and pod affinity/anti-affinity**
  * You can continue to use the alpha version of this feature (with one caveat -- see below), in which you specify affinity requests using Pod annotations, in 1.6 by including `AffinityInAnnotations=true` in `--feature-gates`, such as `--feature-gates=FooBar=true,AffinityInAnnotations=true`. Otherwise, you must modify your PodSpecs that currently use the `scheduler.alpha.kubernetes.io/affinity` annotation on Pod, to instead use the `affinity` field in the PodSpec. Support for the annotation will be removed in a future release, so we encourage you to switch to using the field as soon as possible.
  * Caveat: The alpha version no longer supports, and the beta version does not support, the "empty `podAffinityTerm.namespaces` list means all namespaces" behavior. In both alpha and beta it now means "same namespace as the pod specifying this affinity rule."
  * Note that you can only start using the `affinity` field **after** you upgrade to 1.6; it is not recognized in 1.5.
  * The `--failure-domains` scheduler command line-argument is not supported in the beta version of the feature.

* **Taints**
  * You will need to use `kubectl taint` to re-create all of your taints after kubectl and the master are upgraded to 1.6. Between the time the master is upgraded to 1.6 and when you do this, your existing taints will have no effect.
  * You can find out what taints you have in place on a node while you are still running Kubernetes 1.5 by doing `kubectl describe node <node name>`; the `Taints` section will show the taints you have in place. To see the taints that were created under 1.5 when you are running 1.6, do `kubectl get node <node name> -o yaml` and look for the "Annotation" section with the annotation key `scheduler.alpha.kubernetes.io/taints`
  * You can remove the "old" taints (stored internally as annotations) at any time after the upgrade by doing `kubectl annotate nodes <node name> scheduler.alpha.kubernetes.io/taints-` (note the minus at the end, which means "delete the taint with this key")
  * Note that because any taints you might have created with Kubernetes 1.5 can only affect the scheduling of new pods (the `NoExecute` taint effect is introduced in 1.6), neither the master upgrade nor your running `kubectl taint` to re-create the taints will affect pods that are already running.
  * Rescheduler relies on taints, which were changed in a backward incompatible way. Rescheduler 0.3 shipped together with Kubernetes 1.6 understands the new taints and will clean up the old annotations, but Rescheduler 0.2 shipped together with Kubernetes 1.5 doesn't understand the new taints. In order to avoid eviction loop during 1.5->1.6 upgrade you need to either upgrade the master node atomically (for example by using `upgrade.sh` script for GCE) or ensure that the rescheduler is upgraded first, then the scheduler and then all the other master components.

* **Tolerations**
  * After your master is upgraded to 1.6, you will need to update your PodSpecs to set the `tolerations` field of the PodSpec and remove the `scheduler.alpha.kubernetes.io/tolerations` annotation on the Pod. (It's not strictly necessary to remove the annotation, as it will have no effect once you upgrade to 1.6.) Between the time the master is upgraded to 1.6 and when you do this, tolerations attached to Pods that are created will have no effect. Pods that are already running will not be affected by the upgrade.
  * You can find the PodSpec tolerations that were created as annotations (if any) in a controller definition by doing `kubectl get <controller kind> <controller name> -o yaml` and looking for the "Annotation" section with the annotation key `scheduler.alpha.kubernetes.io/tolerations` (This will work whether you are running Kubernetes 1.5 or 1.6).
  * To update a controller's PodSpec to use the field instead of the annotation, use one of the kubectl commands that do update ("kubectl replace" or "kubectl apply" or "kubectl patch") or just delete the controller entirely and re-create it with a new pod template. Note that you will be able to do these things only after the upgrade.

### Service
* The 'endpoints.beta.kubernetes.io/hostnames-map' annotation is no longer supported.  Users can use the 'Endpoints.subsets[].addresses[].hostname' field instead. ([#39284](https://github.com/kubernetes/kubernetes/pull/39284), [@bowei](https://github.com/bowei))

### StatefulSet
* StatefulSet now respects ControllerRef to avoid fighting over Pods. At the time of upgrade, **you must not have StatefulSets with selectors that overlap** with any other controllers (such as ReplicaSets), or else [ownership of Pods may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md#upgrading). ([#42080](https://github.com/kubernetes/kubernetes/pull/42080), [@enisoc](https://github.com/enisoc))

### Volumes
* StorageClass pre-installed and set as default on Azure, AWS, GCE, OpenStack, and vSphere.
  * This is something to pay close attention to if you’ve been using Kubernetes for a while, because it changes the default behavior of PersistentVolumeClaim objects on these clouds.
  * Marking a StorageClass as default makes it so that even a PersistentVolumeClaim without a StorageClass specified will trigger dynamic provisioning (instead of binding to an existing pool of PVs).
  * If you depend on the old behavior of volumes binding to existing pool of PersistentVolume objects then modify the StorageClass object and set `storageclass.beta.kubernetes.io/is-default-class` to `false`.
* Flex volume plugin is updated to support attach/detach interfaces. It broke backward compatibility. Please update your drivers and implement the new callouts.  ([#41804](https://github.com/kubernetes/kubernetes/pull/41804), [@chakri-nelluri](https://github.com/chakri-nelluri))

## Notable Features
Features for this release were tracked via the use of the [kubernetes/features](https://github.com/kubernetes/features) issues repo.  Each Feature issue is owned by a Special Interest Group from the [kubernetes/community](https://github.com/kubernetes/community/blob/master/sig-list.md).

### Autoscaling
* **[alpha]** The Horizontal Pod Autoscaler now supports drawing metrics through the API server aggregator.
* **[alpha]** The Horizontal Pod Autoscaler now supports scaling on multiple, custom metrics.
* Cluster Autoscaler publishes its status to kube-system/cluster-autoscaler-status ConfigMap.
* Cluster Autoscaler can continue operations while some nodes are broken or unready.
* Cluster Autoscaler respects Pod Disruption Budgets when removing a node.

### DaemonSet
* **[beta]** Introduce the rolling update feature for DaemonSet. See [Performing a Rolling Update on a DaemonSet](https://kubernetes.io/docs/tasks/manage-daemon/update-daemon-set/).

### Deployment
* **[beta]** Deployments that cannot make progress in rolling out the newest version will now indicate via the API they are blocked ([docs](https://kubernetes.io/docs/user-guide/deployments/#deployment-status))

### Federation
* **[beta]** `kubefed` has graduated to beta: supports hosting federation on on-prem clusters, automatically configures `kube-dns` in joining clusters and allows passing arguments to federation components.

### Internal Storage Layer
* **[stable]** The internal storage layer for kubernetes cluster state has been updated to use etcd v3 by default. Existing clusters will have to plan for a data migration window. ([docs](https://kubernetes.io/docs/tasks/administer-cluster/upgrade-1-6/))([kubernetes/features#44](https://github.com/kubernetes/features/issues/44))

### kubeadm
* **[beta]** Introduces an API for clients to request TLS certificates from the API server. See the [tutorial](https://kubernetes.io/docs/tasks/tls/managing-tls-in-a-cluster).
* **[beta]** kubeadm is enhanced and improved with a baseline feature set and command line flags that are now marked as beta. Other parts of kubeadm, including subcommands under `kubeadm alpha`, are [still in alpha](https://kubernetes.io/docs/admin/kubeadm/).  Using it is considered safe, although note that upgrades and HA are not yet supported. Please [try it out](https://kubernetes.io/docs/getting-started-guides/kubeadm/) and [give us feedback](https://kubernetes.io/docs/getting-started-guides/kubeadm/#feedback)!
* **[alpha]** New Bootstrap Token authentication and management method. Works well with kubeadm. kubeadm now supports managing tokens, including time based expiration, after the cluster is launched.  See [kubeadm reference docs](https://kubernetes.io/docs/admin/kubeadm/#manage-tokens) for details.
* **[alpha]** Adds a new cloud-controller-manager binary that may be used for testing the new out-of-core cloudprovider flow.

### Node Components
* **[stable]** Init containers have graduated to GA and now appear as a field.
  The beta annotation value will still be respected and overrides the field
  value.
* Kubelet Container Runtime Interface (CRI) support
  - **[beta]** The Docker-CRI implementation is enabled by default in kubelet.
    You can disable it by --enable-cri=false. See
    [notes on the new implementation](https://github.com/kubernetes/community/blob/master/contributors/devel/container-runtime-interface.md#kubernetes-v16-release-docker-cri-integration-beta)
    for more details.
  - **[alpha]** Alpha support for other runtimes:
    [cri-o](https://github.com/kubernetes-incubator/cri-o/releases/tag/v0.1), [frakti](https://github.com/kubernetes/frakti/releases/tag/v0.1), [rkt](https://github.com/coreos/rkt/issues?q=is%3Aopen+is%3Aissue+label%3Aarea%2Fcri).
* **[beta]** Node Problem Detector is beta (v0.3.0) now. New
  features added into node-problem-detector:v0.3.0:
  - Add support for journald.
  - The ability to monitor any system daemon logs. Previously only kernel
    logs were supported.
  - The ability to be deployed and run as a native daemon.
* **[alpha]** Critical Pods: When feature gate "ExperimentalCriticalPodAnnotation" is
  set to true, the system will guarantee scheduling and admission of pods with
  the following annotation - `scheduler.alpha.kubernetes.io/critical-pod`
  - This feature should be used in conjunction with the
    [rescheduler](https://kubernetes.io/docs/admin/rescheduler/) to guarantee
    resource availability for critical system pods.
* **[alpha]** `--experimental-nvidia-gpus` flag is replaced by `Accelerators` alpha
  feature gate along with addition of support for multiple Nvidia GPUs.
  - To use GPUs, pass `Accelerators=true` as part of `--feature-gates` flag.
  - More information [here](https://vishh.github.io/docs/user-guide/gpus/).
* A pod’s Quality of Service Class is now available in its Status.
* Upgrade cAdvisor library to v0.25.0. Notable changes include,
  - Container filesystem usage tracking disabled for device mapper due to excessive
    IOPS.
  - Ignore `.mount` cgroups, fixing disappearing stats.
* A new field `terminationMessagePolicy` has been added to containers that allows
  a user to request FallbackToLogsOnError, which will read from the container's
  logs to populate the termination message if the user does not write to the
  termination message log file. The termination message file is now properly
  readable for end users and has a maximum size (4k bytes) to prevent abuse.
  Each pod may have up to 12k bytes of termination messages before the contents
  of each will be truncated.
* Do not delete pod objects until all its compute resource footprint has been
  reclaimed.
  - This feature prevents the node and scheduler from oversubscribing resources
    that are still being used by a pod in the process of being cleaned up.
  - This feature can result in increase of Pod Deletion latency especially if the
    pod has a large filesystem footprint.

### RBAC
* **[beta]** RBAC API is promoted to v1beta1 (rbac.authorization.k8s.io/v1beta1), and defines default roles for control plane, node, and controller components.
* **[beta]** The Docker-CRI implementation is Beta and is enabled by default in kubelet.  You can disable it by `--enable-cri=false`. See [notes on the new implementation]( https://github.com/kubernetes/community/blob/master/contributors/devel/container-runtime-interface.md#kubernetes-v16-release-docker-cri-integration-beta) for more details.

### Scheduling
- **[beta]** The [multiple schedulers](https://kubernetes.io/docs/tutorials/clusters/multiple-schedulers/). This feature allows you to run multiple schedulers in parallel, each responsible for different sets of pods. When using multiple schedulers, the scheduler name is now specified in a new-in-1.6 `schedulerName` field of the PodSpec rather than using the `scheduler.alpha.kubernetes.io/name` annotation on the Pod. When you upgrade to 1.6, the Kubernetes default scheduler will start using the `schedulerName` field of the PodSpec and will ignore the annotation.
- **[beta]** [Node affinity/anti-affinity](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/) and **[beta]** [pod affinity/anti-affinity](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/). Node affinity/anti-affinity allow you to specify rules for restricting which node(s) a pod can schedule onto, based on the labels on the node. Pod affinity/anti-affinity allow you to specify rules for spreading and packing pods relative to one another, across arbitrary topologies (node, zone, etc.) These affinity rules are now be specified in a new-in-1.6 `affinity` field of the PodSpec. Kubernetes 1.6 continues to support the alpha `scheduler.alpha.kubernetes.io/affinity` annotation on the Pod if you explicitly enable the alpha feature "AffinityInAnnotations", but it will be removed in a future release. When you upgrade to 1.6, if you have not enabled the alpha feature, then the scheduler will use the `affinity` field in PodSpec and will ignore the annotation. If you have enabled the alpha feature, then the scheduler will use the `affinity` field in PodSpec if it is present, and otherwise will use the annotation.
- **[beta]** [Taints and tolerations](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/). This feature allows you to specify rules for "repelling" pods from nodes by default, which enables use cases like dedicated nodes and reserving nodes with special features for pods that need those features. We've also added a `NoExecute` taint type that evicts already-running pods, and an associated `tolerationSeconds` field to tolerations to delay the eviction for a specified amount of time. As before, taints are created using `kubectl taint` (but internally they are now represented as a field `taints` in the NodeSpec rather than using the `scheduler.alpha.kubernetes.io/taints` annotation on Node). Tolerations are now specified in a new-in-1.6 `tolerations` field of the PodSpec rather than using the `scheduler.alpha.kubernetes.io/tolerations` annotation on the Pod. When you upgrade to 1.6, the scheduler will start using the fields and will ignore the annotations.
- **[alpha]** Represent node problems "not ready" and "unreachable" using `NoExecute` taints. In combination with `tolerationSeconds` described below, this allows per-pod specification of how long to remain bound to a node that becomes unreachable or not ready, rather than using the default of 5 minutes. You can enable this alpha feature by including `TaintBasedEvictions=true` in `--feature-gates`, such as `--feature-gates=FooBar=true,TaintBasedEvictions=true`. Documentation is [here](https://kubernetes.io/docs/concepts/configuration/assign-pod-node/).

### Service Catalog
- **[alpha]** Adds a new API resource `PodPreset` and admission controller to enable defining cross-cutting injection of Volumes and Environment into Pods.

### Volumes
* **[stable]** StorageClass API is promoted to v1 (storage.k8s.io/v1).
* **[stable]** Default storage classes are deployed during installation on Azure, AWS, GCE, OpenStack and vSphere.
* **[stable]** Added ability to populate environment variables from a configmap or secret.
* **[stable]** Support for user-written/run dynamic PV provisioners. The Kubernetes Incubator contains [a golang library and examples](https://github.com/kubernetes-incubator/external-storage).
* **[stable]** Volume plugin for ScaleIO enabling pods to seamlessly access and use data stored on Dell EMC ScaleIO volumes.
* **[stable]** Volume plugin for Portworx added capability to use [Portworx](http://www.portworx.com) as a storage provider for Kubernetes clusters. Portworx pools server capacity and turns servers or cloud instances into converged, highly available compute and storage nodes.
* **[stable]** Add support to use NFSv3, NFSv4, and GlusterFS on GCE/GKE GCI image based clusters.
* **[beta]** Added support for mount options in persistent volumes.
* **[alpha]** All in one volume proposal - a new volume driver capable of projecting secrets, configmaps, and downward API items into the same directory.

## Deprecations
* Remove extensions/v1beta1 Jobs resource, and job/v1beta1 generator. ([#38614](https://github.com/kubernetes/kubernetes/pull/38614), [@soltysh](https://github.com/soltysh))
* `federation/deploy/deploy.sh` was an interim solution introduced in Kubernetes v1.4 to simplify the federation control plane deployment experience. Now that we have `kubefed`, we are deprecating `deploy.sh` scripts. ([#38902](https://github.com/kubernetes/kubernetes/pull/38902), [@madhusudancs](https://github.com/madhusudancs))


### Cluster Provisioning Scripts
* The bash AWS deployment via kube-up.sh has been deprecated. See http://kubernetes.io/docs/getting-started-guides/aws/ for alternatives. ([#38772](https://github.com/kubernetes/kubernetes/pull/38772), [@zmerlynn](https://github.com/zmerlynn))
* Remove Azure kube-up as the Azure community has focused efforts elsewhere. ([#41672](https://github.com/kubernetes/kubernetes/pull/41672), [@mikedanese](https://github.com/mikedanese))
* Remove the deprecated vsphere kube-up. ([#39140](https://github.com/kubernetes/kubernetes/pull/39140), [@kerneltime](https://github.com/kerneltime))

### kubeadm
* Quite a few flags been renamed or removed.  Those options that are removed as flags can still be accessed via the config file.  Most notably this includes external etcd settings and the option for setting the cloud provider on the API server.  The [kubeadm reference documentation](https://kubernetes.io/docs/admin/kubeadm/) is up to date with the new flags.

### Other Deprecations
* Remove cmd/kube-discovery from the tree since it's not necessary anymore ([#42070](https://github.com/kubernetes/kubernetes/pull/42070), [@luxas](https://github.com/luxas))

## Changes to API Resources
### ABAC
* ABAC policies using `"user":"*"` or `"group":"*"` to match all users or groups will only match authenticated requests. To match unauthenticated requests, ABAC policies must explicitly specify `"group":"system:unauthenticated"` ([#38968](https://github.com/kubernetes/kubernetes/pull/38968), [@liggitt](https://github.com/liggitt))

### Admission Control
* Adds a new API resource `PodPreset` and admission controller to enable defining cross-cutting injection of Volumes and Environment into Pods. ([#41931](https://github.com/kubernetes/kubernetes/pull/41931), [@jessfraz](https://github.com/jessfraz))
* Enable DefaultTolerationSeconds admission controller by default ([#41815](https://github.com/kubernetes/kubernetes/pull/41815), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* ResourceQuota ability to support default limited resources ([#36765](https://github.com/kubernetes/kubernetes/pull/36765), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Add defaultTolerationSeconds admission controller ([#41414](https://github.com/kubernetes/kubernetes/pull/41414), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* An `automountServiceAccountToken` field was added to ServiceAccount and PodSpec objects. If set to `false` on a pod spec, no service account token is automounted in the pod. If set to `false` on a service account, no service account token is automounted for that service account unless explicitly overridden in the pod spec. ([#37953](https://github.com/kubernetes/kubernetes/pull/37953), [@liggitt](https://github.com/liggitt))
* Admission control support for versioned configuration files ([#39109](https://github.com/kubernetes/kubernetes/pull/39109), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Ability to quota storage by storage class ([#34554](https://github.com/kubernetes/kubernetes/pull/34554), [@derekwaynecarr](https://github.com/derekwaynecarr))

### Authentication
* The authentication.k8s.io API group was promoted to v1([#41058](https://github.com/kubernetes/kubernetes/pull/41058), [@liggitt](https://github.com/liggitt))

### Authorization
* The authorization.k8s.io API group was promoted to v1 ([#40709](https://github.com/kubernetes/kubernetes/pull/40709), [@liggitt](https://github.com/liggitt))
* The SubjectAccessReview API now passes subresource and resource name information to the authorizer to answer authorization queries. ([#40935](https://github.com/kubernetes/kubernetes/pull/40935), [@liggitt](https://github.com/liggitt))

### Autoscaling
* Introduces an new alpha version of the Horizontal Pod Autoscaler including expanded support for specifying metrics. ([#36033](https://github.com/kubernetes/kubernetes/pull/36033), [@DirectXMan12](https://github.com/DirectXMan12)
* HorizontalPodAutoscaler is no longer supported in extensions/v1beta1 version. Use autoscaling/v1 instead. ([#35782](https://github.com/kubernetes/kubernetes/pull/35782), [@piosz](https://github.com/piosz))
* Fixes an HPA-related panic due to division-by-zero. ([#39694](https://github.com/kubernetes/kubernetes/pull/39694), [@DirectXMan12](https://github.com/DirectXMan12))

### Certificates
* The CertificateSigningRequest API added the `extra` field to persist all information about the requesting user. This mirrors the fields in the SubjectAccessReview API used to check authorization. ([#41755](https://github.com/kubernetes/kubernetes/pull/41755), [@liggitt](https://github.com/liggitt))
* Native support for token based bootstrap flow.  This includes signing a well known ConfigMap in the `kube-public` namespace and cleaning out expired tokens. ([#36101](https://github.com/kubernetes/kubernetes/pull/36101), [@jbeda](https://github.com/jbeda))

### ConfigMap
* Volumes and environment variables populated from ConfigMap and Secret objects can now tolerate the named source object or specific keys being missing, by adding `optional: true` to the volume or environment variable source specifications. ([#39981](https://github.com/kubernetes/kubernetes/pull/39981), [@fraenkel](https://github.com/fraenkel))
* Allow pods to define multiple environment variables from a whole ConfigMap ([#36245](https://github.com/kubernetes/kubernetes/pull/36245), [@fraenkel](https://github.com/fraenkel))

### CronJob
* Add configurable limits to CronJob resource to specify how many successful and failed jobs are preserved. ([#40932](https://github.com/kubernetes/kubernetes/pull/40932), [@peay](https://github.com/peay))

### DaemonSet
* DaemonSet now respects ControllerRef to avoid fighting over Pods. ([#42173](https://github.com/kubernetes/kubernetes/pull/42173), [@enisoc](https://github.com/enisoc))
* Make DaemonSet respect critical pods annotation when scheduling. ([#42028](https://github.com/kubernetes/kubernetes/pull/42028), [@janetkuo](https://github.com/janetkuo))
* Implement the update feature for DaemonSet. ([#41116](https://github.com/kubernetes/kubernetes/pull/41116), [@lukaszo](https://github.com/lukaszo))
* Make DaemonSets survive taint-based evictions when nodes turn unreachable/notReady. ([#41896](https://github.com/kubernetes/kubernetes/pull/41896), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Make DaemonSet controller respect node taints and pod tolerations. ([#41172](https://github.com/kubernetes/kubernetes/pull/41172), [@janetkuo](https://github.com/janetkuo))
* DaemonSet controller actively kills failed pods (to recreate them) ([#40330](https://github.com/kubernetes/kubernetes/pull/40330), [@janetkuo](https://github.com/janetkuo))
* DaemonSet ObservedGeneration ([#39157](https://github.com/kubernetes/kubernetes/pull/39157), [@lukaszo](https://github.com/lukaszo))

### Deployment
* Add ready replicas in Deployments ([#37959](https://github.com/kubernetes/kubernetes/pull/37959), [@kargakis](https://github.com/kargakis))
* Deployments that cannot make progress in rolling out the newest version will now indicate via the API they are blocked
* Introduce apps/v1beta1.Deployments resource with modified defaults compared to extensions/v1beta1.Deployments. ([#39683](https://github.com/kubernetes/kubernetes/pull/39683), [@soltysh](https://github.com/soltysh))
* Introduce new generator for apps/v1beta1 deployments ([#42362](https://github.com/kubernetes/kubernetes/pull/42362), [@soltysh](https://github.com/soltysh))

### Node
* Set all node conditions to Unknown when node is unreachable ([#36592](https://github.com/kubernetes/kubernetes/pull/36592), [@andrewsykim](https://github.com/andrewsykim))

### Pod
* Init containers have graduated to GA and now appear as a field.  The beta annotation value will still be respected and overrides the field value. ([#38382](https://github.com/kubernetes/kubernetes/pull/38382), [@hodovska](https://github.com/hodovska))
* A new field `terminationMessagePolicy` has been added to containers that allows a user to request `FallbackToLogsOnError`, which will read from the container's logs to populate the termination message if the user does not write to the termination message log file.  The termination message file is now properly readable for end users and has a maximum size (4k bytes) to prevent abuse.  Each pod may have up to 12k bytes of termination messages before the contents of each will be truncated. ([#39341](https://github.com/kubernetes/kubernetes/pull/39341), [@smarterclayton](https://github.com/smarterclayton))
* Fix issue with PodDisruptionBudgets in which `minAvailable` specified as a percentage did not work with StatefulSet Pods. ([#39454](https://github.com/kubernetes/kubernetes/pull/39454), [@foxish](https://github.com/foxish))
* Node affinity has moved from annotations to api fields in the pod spec.  Node affinity that is defined in the annotations will be ignored. ([#37299](https://github.com/kubernetes/kubernetes/pull/37299), [@rrati](https://github.com/rrati))
* Moved pod affinity and anti-affinity from annotations to api fields [#25319](https://github.com/kubernetes/kubernetes/pull/25319) ([#39478](https://github.com/kubernetes/kubernetes/pull/39478), [@rrati](https://github.com/rrati))
* Add QoS pod status field ([#37968](https://github.com/kubernetes/kubernetes/pull/37968), [@sjenning](https://github.com/sjenning))
### Pod Security Policy
* PodSecurityPolicy resource is now enabled by default in the extensions API group. ([#39743](https://github.com/kubernetes/kubernetes/pull/39743), [@pweil-](https://github.com/pweil-))

### RBAC
* The `attributeRestrictions` field has been removed from the PolicyRule type in the rbac.authorization.k8s.io/v1alpha1 API. The field was not used by the RBAC authorizer. ([#39625](https://github.com/kubernetes/kubernetes/pull/39625), [@deads2k](https://github.com/deads2k))
* A user can now be authorized to bind a particular role by having permission to perform the `bind` verb on the referenced role ([#39383](https://github.com/kubernetes/kubernetes/pull/39383), [@liggitt](https://github.com/liggitt))

### ReplicaSet
* ReplicaSet has onwer ref of the Deployment that created it ([#35676](https://github.com/kubernetes/kubernetes/pull/35676), [@krmayankk](https://github.com/krmayankk))

### Secrets
* Populate environment variables from a secrets. ([#39446](https://github.com/kubernetes/kubernetes/pull/39446), [@fraenkel](https://github.com/fraenkel))
* Added a new secret type "bootstrap.kubernetes.io/token" for dynamically creating TLS bootstrapping bearer tokens. ([#41281](https://github.com/kubernetes/kubernetes/pull/41281), [@ericchiang](https://github.com/ericchiang))

### Service
* Endpoints, that tolerate unready Pods, are now listing Pods in state Terminating as well ([#37093](https://github.com/kubernetes/kubernetes/pull/37093), [@simonswine](https://github.com/simonswine))
* Fix Service Update on LoadBalancerSourceRanges Field ([#37720](https://github.com/kubernetes/kubernetes/pull/37720), [@freehan](https://github.com/freehan))
* Bug fix. Incoming UDP packets not reach newly deployed services ([#32561](https://github.com/kubernetes/kubernetes/pull/32561), [@zreigz](https://github.com/zreigz))
* Services of type loadbalancer consume both loadbalancer and nodeport quota. ([#39364](https://github.com/kubernetes/kubernetes/pull/39364), [@zhouhaibing089](https://github.com/zhouhaibing089))

### StatefulSet

* Fix zone placement heuristics so that multiple mounts in a StatefulSet pod are created in the same zone ([#40910](https://github.com/kubernetes/kubernetes/pull/40910), [@justinsb](https://github.com/justinsb))
* Fixes issue [#38418](https://github.com/kubernetes/kubernetes/pull/38418) which, under circumstance, could cause StatefulSet to deadlock. ([#40838](https://github.com/kubernetes/kubernetes/pull/40838), [@kow3ns](https://github.com/kow3ns))
  * Mediates issue [#36859](https://github.com/kubernetes/kubernetes/pull/36859). StatefulSet only acts on Pods whose identity matches the StatefulSet, providing a partial mediation for overlapping controllers.

### Taints and Tolerations
* Add a manager to NodeController that is responsible for removing Pods from Nodes tainted with NoExecute Taints. This feature is beta (as the rest of taints) and enabled by default. It's gated by controller-manager enable-taint-manager flag. ([#40355](https://github.com/kubernetes/kubernetes/pull/40355), [@gmarek](https://github.com/gmarek))
* Add an alpha feature that makes NodeController set Taints instead of deleting Pods from not Ready Nodes. ([#41133](https://github.com/kubernetes/kubernetes/pull/41133), [@gmarek](https://github.com/gmarek))
* Make tolerations respect wildcard key ([#39914](https://github.com/kubernetes/kubernetes/pull/39914), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Forgiveness alpha version api definition ([#39469](https://github.com/kubernetes/kubernetes/pull/39469), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Rescheduler uses taints in v1beta1 and will remove old ones (in version v1alpha1) right after its start. ([#43106](https://github.com/kubernetes/kubernetes/pull/43106), [@piosz](https://github.com/piosz))

### Volumes
* StorageClassName attribute has been added to PersistentVolume and PersistentVolumeClaim objects and should be used instead of annotation `volume.beta.kubernetes.io/storage-class`. The beta annotation is still working in this release, however it will be removed in a future release. ([#42128](https://github.com/kubernetes/kubernetes/pull/42128), [@jsafrane](https://github.com/jsafrane))
* Parameter keys in a StorageClass `parameters` map may not use the `kubernetes.io` or `k8s.io` namespaces. ([#41837](https://github.com/kubernetes/kubernetes/pull/41837), [@liggitt](https://github.com/liggitt))
* Add storage.k8s.io/v1 API ([#40088](https://github.com/kubernetes/kubernetes/pull/40088), [@jsafrane](https://github.com/jsafrane))
* Alpha version of dynamic volume provisioning is removed in this release. Annotation ([#40000](https://github.com/kubernetes/kubernetes/pull/40000), [@jsafrane](https://github.com/jsafrane))
    * "volume.alpha.kubernetes.io/storage-class" does not have any special meaning. A default storage class
    * and  DefaultStorageClass admission plugin can be used to preserve similar behavior of Kubernetes cluster,
    * see https://kubernetes.io/docs/user-guide/persistent-volumes/#class-1 for details.
* Reduce verbosity of volume reconciler when attaching volumes ([#36900](https://github.com/kubernetes/kubernetes/pull/36900), [@codablock](https://github.com/codablock))
* We change the default attach_detach_controller sync period to 1 minute to reduce the query frequency through cloud provider to check whether volumes are attached or not. ([#41363](https://github.com/kubernetes/kubernetes/pull/41363), [@jingxu97](https://github.com/jingxu97))

## Changes to Major Components
### API Server
* **`--anonymous-auth` is enabled by default, unless the API server is started with the `AlwaysAllow` authorizer. ([#38706](https://github.com/kubernetes/kubernetes/pull/38706), [@deads2k](https://github.com/deads2k))**
* **When using OIDC authentication and specifying --oidc-username-claim=email, an `"email_verified":true` claim must be returned from the identity provider. ([#36087](https://github.com/kubernetes/kubernetes/pull/36087), [@ericchiang](https://github.com/ericchiang))**
  * `--basic-auth-file` supports optionally specifying groups in the fourth column of the file ([#39651](https://github.com/kubernetes/kubernetes/pull/39651), [@liggitt](https://github.com/liggitt))
* API server now has two separate limits for read-only and mutating inflight requests. ([#36064](https://github.com/kubernetes/kubernetes/pull/36064), [@gmarek](https://github.com/gmarek))
* Restored normalization of custom `--etcd-prefix` when `--storage-backend` is set to etcd3 ([#42506](https://github.com/kubernetes/kubernetes/pull/42506), [@liggitt](https://github.com/liggitt))
* Updating apiserver to return http status code 202 for a delete request when the resource is not immediately deleted because of user requesting cascading deletion using DeleteOptions.OrphanDependents=false. ([#41165](https://github.com/kubernetes/kubernetes/pull/41165), [@nikhiljindal](https://github.com/nikhiljindal))
* Use full package path for definition name in OpenAPI spec ([#40124](https://github.com/kubernetes/kubernetes/pull/40124), [@mbohlool](https://github.com/mbohlool))
* Follow redirects for streaming requests (exec/attach/port-forward) in the apiserver by default (alpha -> beta). ([#40039](https://github.com/kubernetes/kubernetes/pull/40039), [@timstclair](https://github.com/timstclair))
* Add 'X-Content-Type-Options: nosniff" to some error messages ([#37190](https://github.com/kubernetes/kubernetes/pull/37190), [@brendandburns](https://github.com/brendandburns))
* Fixes bug in resolving client-requested API versions ([#38533](https://github.com/kubernetes/kubernetes/pull/38533), [@DirectXMan12](https://github.com/DirectXMan12))
* Replace glog.Fatals with fmt.Errorfs ([#38175](https://github.com/kubernetes/kubernetes/pull/38175), [@sttts](https://github.com/sttts))
* Pipe get options to storage ([#37693](https://github.com/kubernetes/kubernetes/pull/37693), [@wojtek-t](https://github.com/wojtek-t))
* The --long-running-request-regexp flag to kube-apiserver is deprecated and will be removed in a future release. Long-running requests are now detected based on specific verbs (watch, proxy) or subresources (proxy, portforward, log, exec, attach). ([#38119](https://github.com/kubernetes/kubernetes/pull/38119), [@liggitt](https://github.com/liggitt))
* if kube-apiserver is started with `--storage-backend=etcd2`, the media type `application/json` is used. ([#43122](https://github.com/kubernetes/kubernetes/pull/43122), [@liggitt](https://github.com/liggitt))
* API fields that previously serialized null arrays as `null` and empty arrays as `[]` no longer distinguish between those values and always output `[]` when serializing to JSON. ([#43422](https://github.com/kubernetes/kubernetes/pull/43422), [@liggitt](https://github.com/liggitt))
* Generate OpenAPI definition for inlined types ([#39466](https://github.com/kubernetes/kubernetes/pull/39466), [@mbohlool](https://github.com/mbohlool))

### API Server Aggregator
* Rename kubernetes-discovery to kube-aggregator ([#39619](https://github.com/kubernetes/kubernetes/pull/39619), [@deads2k](https://github.com/deads2k))
* Fix connection upgrades through kuberentes-discovery ([#38724](https://github.com/kubernetes/kubernetes/pull/38724), [@deads2k](https://github.com/deads2k))

#### Generic API Server
* Move pkg/api/rest into genericapiserver ([#39948](https://github.com/kubernetes/kubernetes/pull/39948), [@sttts](https://github.com/sttts))
* Move non-generic apiserver code out of the generic packages ([#38191](https://github.com/kubernetes/kubernetes/pull/38191), [@sttts](https://github.com/sttts))
* Fixes API compatibility issue with empty lists incorrectly returning a null `items` field instead of an empty array. ([#39834](https://github.com/kubernetes/kubernetes/pull/39834), [@liggitt](https://github.com/liggitt))
* Re-add /healthz/ping handler in genericapiserver ([#38603](https://github.com/kubernetes/kubernetes/pull/38603), [@sttts](https://github.com/sttts))
* Remove genericapiserver.Options.MasterServiceNamespace ([#38186](https://github.com/kubernetes/kubernetes/pull/38186), [@sttts](https://github.com/sttts))
* genericapiserver: cut off more dependencies – episode 3 ([#40426](https://github.com/kubernetes/kubernetes/pull/40426), [@sttts](https://github.com/sttts))
* genericapiserver: more dependency cutoffs ([#40216](https://github.com/kubernetes/kubernetes/pull/40216), [@sttts](https://github.com/sttts))
* genericapiserver: cut off kube pkg/version dependency ([#39943](https://github.com/kubernetes/kubernetes/pull/39943), [@sttts](https://github.com/sttts))
* genericapiserver: cut off pkg/serviceaccount dependency ([#39945](https://github.com/kubernetes/kubernetes/pull/39945), [@sttts](https://github.com/sttts))
* genericapiserver: cut off pkg/apis/extensions and pkg/storage dependencies ([#39946](https://github.com/kubernetes/kubernetes/pull/39946), [@sttts](https://github.com/sttts))
* genericapiserver: cut off certificates api dependency ([#39947](https://github.com/kubernetes/kubernetes/pull/39947), [@sttts](https://github.com/sttts))
* genericapiserver: extract CA cert from server cert and SNI cert chains ([#39022](https://github.com/kubernetes/kubernetes/pull/39022), [@sttts](https://github.com/sttts))
* genericapiserver: turn APIContainer.SecretRoutes into a real ServeMux ([#38826](https://github.com/kubernetes/kubernetes/pull/38826), [@sttts](https://github.com/sttts))
* genericapiserver: unify swagger and openapi in config ([#38690](https://github.com/kubernetes/kubernetes/pull/38690), [@sttts](https://github.com/sttts))

### Client
* Use Prometheus instrumentation conventions ([#36704](https://github.com/kubernetes/kubernetes/pull/36704), [@fabxc](https://github.com/fabxc))
* Clients now use the `?watch=true` parameter to make watch API calls, instead of the `/watch/` path prefix ([#41722](https://github.com/kubernetes/kubernetes/pull/41722), [@liggitt](https://github.com/liggitt))
* Move private key parsing from serviceaccount/jwt.go to client-go/util/cert ([#40907](https://github.com/kubernetes/kubernetes/pull/40907), [@cblecker](https://github.com/cblecker))
* Caching added to the OIDC client auth plugin to fix races and reduce the time kubectl commands using this plugin take by several seconds. ([#38167](https://github.com/kubernetes/kubernetes/pull/38167), [@ericchiang](https://github.com/ericchiang))
* Fix resync goroutine leak in ListAndWatch ([#35672](https://github.com/kubernetes/kubernetes/pull/35672), [@tatsuhiro-t](https://github.com/tatsuhiro-t))

#### client-go
* The main repository does not keep multiple releases of clientsets anymore. Please find previous releases at https://github.com/kubernetes/client-go ([#38154](https://github.com/kubernetes/kubernetes/pull/38154), [@caesarxuchao](https://github.com/caesarxuchao))
* Support whitespace in command path for gcp auth plugin ([#41653](https://github.com/kubernetes/kubernetes/pull/41653), [@jlowdermilk](https://github.com/jlowdermilk))
* client-go no longer imports GCP OAuth2 and OpenID Connect packages by default. ([#41532](https://github.com/kubernetes/kubernetes/pull/41532), [@ericchiang](https://github.com/ericchiang))
* Added bool type support for jsonpath. ([#39063](https://github.com/kubernetes/kubernetes/pull/39063), [@xingzhou](https://github.com/xingzhou))
* Preventing nil pointer reference in client_config ([#40508](https://github.com/kubernetes/kubernetes/pull/40508), [@vjsamuel](https://github.com/vjsamuel))
* Prevent hotloops on error conditions, which could fill up the disk faster than log rotation can free space. ([#40497](https://github.com/kubernetes/kubernetes/pull/40497), [@lavalamp](https://github.com/lavalamp))

### Cloud Provider

#### AWS
* Allow to running the master with a different AWS account or even on a different cloud provider than the nodes. ([#39996](https://github.com/kubernetes/kubernetes/pull/39996), [@scheeles](https://github.com/scheeles))
* Support shared tag `kubernetes.io/cluster/<clusterid>` ([#41695](https://github.com/kubernetes/kubernetes/pull/41695), [@justinsb](https://github.com/justinsb))
* Do not consider master instance zones for dynamic volume creation ([#41702](https://github.com/kubernetes/kubernetes/pull/41702), [@justinsb](https://github.com/justinsb))
* Fix AWS device allocator to only use valid device names ([#41455](https://github.com/kubernetes/kubernetes/pull/41455), [@gnufied](https://github.com/gnufied))
* Trust region if found from AWS metadata ([#38880](https://github.com/kubernetes/kubernetes/pull/38880), [@justinsb](https://github.com/justinsb))
* Remove duplicate calls to DescribeInstance during volume operations ([#39842](https://github.com/kubernetes/kubernetes/pull/39842), [@gnufied](https://github.com/gnufied))
* Recognize eu-west-2 region ([#38746](https://github.com/kubernetes/kubernetes/pull/38746), [@justinsb](https://github.com/justinsb))
* Recognize ca-central-1 region ([#38410](https://github.com/kubernetes/kubernetes/pull/38410), [@justinsb](https://github.com/justinsb))
* Add sequential allocator for device names. ([#38818](https://github.com/kubernetes/kubernetes/pull/38818), [@jsafrane](https://github.com/jsafrane))

#### Azure
* Fix failing load balancers in Azure ([#40405](https://github.com/kubernetes/kubernetes/pull/40405), [@codablock](https://github.com/codablock))
* Reduce time needed to attach Azure disks ([#40066](https://github.com/kubernetes/kubernetes/pull/40066), [@codablock](https://github.com/codablock))
* Remove Azure Subnet RouteTable check ([#38334](https://github.com/kubernetes/kubernetes/pull/38334), [@mogthesprog](https://github.com/mogthesprog))
* Add support for Azure Container Registry, update Azure dependencies ([#37783](https://github.com/kubernetes/kubernetes/pull/37783), [@brendandburns](https://github.com/brendandburns))
* Allow backendpools in Azure Load Balancers which are not owned by cloud provider ([#36882](https://github.com/kubernetes/kubernetes/pull/36882), [@codablock](https://github.com/codablock))

#### GCE
* On GCI by default logrotate is disabled for application containers in favor of rotation mechanism provided by docker logging driver. ([#40634](https://github.com/kubernetes/kubernetes/pull/40634), [@crassirostris](https://github.com/crassirostris))

#### GKE
* New GKE certificates controller. ([#41160](https://github.com/kubernetes/kubernetes/pull/41160), [@pipejakob](https://github.com/pipejakob))

#### vSphere
* Reverts to looking up the current VM in vSphere using the machine's UUID, either obtained via sysfs or via the `vm-uuid` parameter in the cloud configuration file. ([#40892](https://github.com/kubernetes/kubernetes/pull/40892), [@robdaemon](https://github.com/robdaemon))
* Fix for detach volume when node is not present/ powered off ([#40118](https://github.com/kubernetes/kubernetes/pull/40118), [@BaluDontu](https://github.com/BaluDontu))
* Adding vmdk file extension for vmDiskPath in vsphere DeleteVolume ([#40538](https://github.com/kubernetes/kubernetes/pull/40538), [@divyenpatel](https://github.com/divyenpatel))
* Changed default scsi controller type in vSphere Cloud Provider ([#38426](https://github.com/kubernetes/kubernetes/pull/38426), [@abrarshivani](https://github.com/abrarshivani))
* Fixes NotAuthenticated errors that appear in the kubelet and kube-controller-manager due to never logging in to vSphere ([#36169](https://github.com/kubernetes/kubernetes/pull/36169), [@robdaemon](https://github.com/robdaemon))
* Fix panic in vSphere cloud provider ([#38423](https://github.com/kubernetes/kubernetes/pull/38423), [@BaluDontu](https://github.com/BaluDontu))
* Fix space issue in volumePath with vSphere Cloud Provider ([#38338](https://github.com/kubernetes/kubernetes/pull/38338), [@BaluDontu](https://github.com/BaluDontu))

### Federation

#### kubefed
* Flag cleanup ([#41335](https://github.com/kubernetes/kubernetes/pull/41335), [@irfanurrehman](https://github.com/irfanurrehman))
* Create configmap for the cluster kube-dns when cluster joins and remove when it unjoins ([#39338](https://github.com/kubernetes/kubernetes/pull/39338), [@irfanurrehman](https://github.com/irfanurrehman))
* Support configuring dns-provider ([#40528](https://github.com/kubernetes/kubernetes/pull/40528), [@shashidharatd](https://github.com/shashidharatd))
* Bug fix relating kubeconfig path in kubefed init ([#41410](https://github.com/kubernetes/kubernetes/pull/41410), [@irfanurrehman](https://github.com/irfanurrehman))
* Add override flags options to kubefed init ([#40917](https://github.com/kubernetes/kubernetes/pull/40917), [@irfanurrehman](https://github.com/irfanurrehman))
* Add option to expose federation apiserver on nodeport service ([#40516](https://github.com/kubernetes/kubernetes/pull/40516), [@shashidharatd](https://github.com/shashidharatd))
* Add option to disable persistence storage for etcd ([#40862](https://github.com/kubernetes/kubernetes/pull/40862), [@shashidharatd](https://github.com/shashidharatd))
* kubefed init creates a service account for federation controller manager in the federation-system namespace and binds that service account to the federation-system:federation-controller-manager role that has read and list access on secrets in the federation-system namespace.  ([#40392](https://github.com/kubernetes/kubernetes/pull/40392), [@madhusudancs](https://github.com/madhusudancs))
* Implement dry run support in kubefed init ([#36447](https://github.com/kubernetes/kubernetes/pull/36447), [@irfanurrehman](https://github.com/irfanurrehman))
* Make federation etcd PVC size configurable ([#36310](https://github.com/kubernetes/kubernetes/pull/36310), [@irfanurrehman](https://github.com/irfanurrehman))

#### Other Notable Changes
* Federated Ingress over GCE no longer requires separate firewall rules to be created for each cluster to circumvent flapping firewall health checks. ([#41942](https://github.com/kubernetes/kubernetes/pull/41942), [@csbell](https://github.com/csbell))
* Add support for finalizers in federated configmaps (deletes configmaps from underlying clusters). ([#40464](https://github.com/kubernetes/kubernetes/pull/40464), [@csbell](https://github.com/csbell))
* Add support for DeleteOptions.OrphanDependents for federated services. Setting it to false while deleting a federated service also deletes the corresponding services from all registered clusters. ([#36390](https://github.com/kubernetes/kubernetes/pull/36390), [@nikhiljindal](https://github.com/nikhiljindal))
* Add `--controllers` flag to federation controller manager for enable/disable federation ingress controller ([#36643](https://github.com/kubernetes/kubernetes/pull/36643), [@kzwang](https://github.com/kzwang))
* Stop deleting services from underlying clusters when federated service is deleted. ([#37353](https://github.com/kubernetes/kubernetes/pull/37353), [@nikhiljindal](https://github.com/nikhiljindal))
* Expose autoscaling apis through federation api server ([#38976](https://github.com/kubernetes/kubernetes/pull/38976), [@irfanurrehman](https://github.com/irfanurrehman))
* Federation: Add `batch/jobs` API objects to federation-apiserver ([#35943](https://github.com/kubernetes/kubernetes/pull/35943), [@jianhuiz](https://github.com/jianhuiz))
* Add logging of route53 calls ([#39964](https://github.com/kubernetes/kubernetes/pull/39964), [@justinsb](https://github.com/justinsb))

### Garbage Collector
* Added foreground garbage collection: the owner object will not be deleted until all its dependents are deleted by the garbage collector. Please checkout the [user doc](https://kubernetes.io/docs/concepts/abstractions/controllers/garbage-collection/) for details. ([#38676](https://github.com/kubernetes/kubernetes/pull/38676), [@caesarxuchao](https://github.com/caesarxuchao))
  * deleteOptions.orphanDependents is going to be deprecated in 1.7. Please use deleteOptions.propagationPolicy instead.

### kubeadm
* A new label and taint is used for marking the master. The label is `node-role.kubernetes.io/master=""` and the taint has the effect `NoSchedule`. Tolerate the `node-role.kubernetes.io/master="":NoSchedule` taint to schedule a workload on the master (a networking DaemonSet for example).
* The kubelet API is now secured, only cluster admins are allowed to access it.
* Insecure access to the API Server over `localhost:8080` is now disabled.
* The control plane components now talk securely to each other. The API Server talks securely to the kubelets in the cluster.
* kubeadm creates RBAC-enabled clusters. This means that some add-ons which worked previously won't work without having their YAML configuration updated. Please see each vendor's own documentation to check that the add-ons you are using will work with Kubernetes 1.6.
* The kube-discovery Deployment isn't used anymore when creating a kubeadm cluster and shouldn't be used anywhere else either due to its insecure nature.
* The Certificates API has graduated from alpha to beta. This is a backwards-incompatible change since the alpha support is dropped, and therefore kubeadm v1.5 and v1.6 don't work together (for example kubeadm v1.5 can't join a kubeadm v1.6 cluster).
* Supporting only etcd3, with 3.0.14 as the minimum version.
* `kubeadm reset` will no longer drain nodes automatically.  This is because the credentials on nodes do not have permission to perform this operation.  We have documented an [alternate procedure](https://kubernetes.io/docs/getting-started-guides/kubeadm/#tear-down), driven from the API server/master.
* Hook up kubeadm against the BootstrapSigner ([#41417](https://github.com/kubernetes/kubernetes/pull/41417), [@luxas](https://github.com/luxas))
* Rename some flags for beta UI and fixup some logic ([#42064](https://github.com/kubernetes/kubernetes/pull/42064), [@luxas](https://github.com/luxas))
* Insecure access to the API Server at localhost:8080 will be turned off in v1.6 when using kubeadm ([#42066](https://github.com/kubernetes/kubernetes/pull/42066), [@luxas](https://github.com/luxas))
* Flag --use-kubernetes-version for kubeadm init renamed to --kubernetes-version ([#41820](https://github.com/kubernetes/kubernetes/pull/41820), [@kad](https://github.com/kad))
* Remove the --cloud-provider flag for beta init UX ([#41710](https://github.com/kubernetes/kubernetes/pull/41710), [@luxas](https://github.com/luxas))
* Fixed an SELinux issue in kubeadm on Docker 1.12+ by moving etcd SELinux options from container to pod. ([#40682](https://github.com/kubernetes/kubernetes/pull/40682), [@dgoodwin](https://github.com/dgoodwin))
* Add authorization mode to kubeadm ([#39846](https://github.com/kubernetes/kubernetes/pull/39846), [@andrewrynhard](https://github.com/andrewrynhard))
* Refactor the certificate and kubeconfig code in the kubeadm binary into two phases ([#39280](https://github.com/kubernetes/kubernetes/pull/39280), [@luxas](https://github.com/luxas))
* Added kubeadm commands to manage bootstrap tokens and the duration they are valid for. ([#35805](https://github.com/kubernetes/kubernetes/pull/35805), [@dgoodwin](https://github.com/dgoodwin))

### kubectl

#### New Commands
- `apply set-last-applied` *updates the applied-applied-configuration annotation* ([#41694](https://github.com/kubernetes/kubernetes/pull/41694), [@shiywang](https://github.com/shiywang))
- `kubectl apply view-last-applied` *viewing the last configuration file applied* ([#41146](https://github.com/kubernetes/kubernetes/pull/41146), [@shiywang](https://github.com/shiywang))

#### Create subcommands
  - `create poddisruptionbudget` ([#36646](https://github.com/kubernetes/kubernetes/pull/36646), [@kargakis](https://github.com/kargakis))
  - `create clusterrole` ([#41538](https://github.com/kubernetes/kubernetes/pull/41538), [@xingzhou](https://github.com/xingzhou))
  - `create role` ([#39852](https://github.com/kubernetes/kubernetes/pull/39852), [@xingzhou](https://github.com/xingzhou))
  - `create clusterrolebinding` ([#37098](https://github.com/kubernetes/kubernetes/pull/37098), [@deads2k](https://github.com/deads2k))
  - `create service externalname` ([#34789](https://github.com/kubernetes/kubernetes/pull/34789), [@adohe](https://github.com/adohe))
- `set selector` - update selector labels ([#38966](https://github.com/kubernetes/kubernetes/pull/38966), [@kargakis](https://github.com/kargakis))
- `can-i` to see if you can perform an action ([#41077](https://github.com/kubernetes/kubernetes/pull/41077), [@deads2k](https://github.com/deads2k))

#### Updates to existing commands
* Printing and output
  * Import a numeric ordering sorting library and use it in the sorting printer. ([#40746](https://github.com/kubernetes/kubernetes/pull/40746), [@matthyx](https://github.com/matthyx))
  * DaemonSet get and describe show status fields.  ([#42843](https://github.com/kubernetes/kubernetes/pull/42843), [@janetkuo](https://github.com/janetkuo))
  * Pods describe shows tolerationSeconds ([#42162](https://github.com/kubernetes/kubernetes/pull/42162), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
  * Node describe contains closing paren ([#39217](https://github.com/kubernetes/kubernetes/pull/39217), [@luksa](https://github.com/luksa))
  * Display pod node selectors with kubectl describe. ([#36396](https://github.com/kubernetes/kubernetes/pull/36396), [@aveshagarwal](https://github.com/aveshagarwal))
  * Add Version to the resource printer for 'get nodes' ([#37943](https://github.com/kubernetes/kubernetes/pull/37943), [@ailusazh](https://github.com/ailusazh))
  * Added support for printing in all supported `--output` formats to `kubectl create ...` and `kubectl apply ...` ([#38112](https://github.com/kubernetes/kubernetes/pull/38112), [@juanvallejo](https://github.com/juanvallejo))
  * Add three more columns to `kubectl get deploy -o wide` output. ([#39240](https://github.com/kubernetes/kubernetes/pull/39240), [@xingzhou](https://github.com/xingzhou))
  * Fix kubectl get -f <file> -o <nondefault printer> so it prints all items in the file ([#39038](https://github.com/kubernetes/kubernetes/pull/39038), [@ncdc](https://github.com/ncdc))
  * kubectl describe no longer prints the last-applied-configuration annotation for secrets. ([#34664](https://github.com/kubernetes/kubernetes/pull/34664), [@ymqytw](https://github.com/ymqytw))
  * Completed pods should not be hidden when requested by name via `kubectl get`. ([#42216](https://github.com/kubernetes/kubernetes/pull/42216), [@smarterclayton](https://github.com/smarterclayton))
  * Improve formatting of EventSource in kubectl get and kubectl describe ([#40073](https://github.com/kubernetes/kubernetes/pull/40073), [@matthyx](https://github.com/matthyx))
* Attach now supports multiple types ([#40365](https://github.com/kubernetes/kubernetes/pull/40365), [@shiywang](https://github.com/shiywang))
* Create now accepts the label selector flag for filtering objects to create ([#40057](https://github.com/kubernetes/kubernetes/pull/40057), [@MrHohn](https://github.com/MrHohn))
* Top now accepts short forms for "node" and "pod" ("no", "po") ([#39218](https://github.com/kubernetes/kubernetes/pull/39218), [@luksa](https://github.com/luksa))
* Begin paths for internationalization in kubectl ([#36802](https://github.com/kubernetes/kubernetes/pull/36802), [@brendandburns](https://github.com/brendandburns))
  * Add initial french translations for kubectl ([#40645](https://github.com/kubernetes/kubernetes/pull/40645), [@brendandburns](https://github.com/brendandburns))

#### Updates to apply
* New command `apply set-last-applied` *updates the applied-applied-configuration annotation* ([#41694](https://github.com/kubernetes/kubernetes/pull/41694), [@shiywang](https://github.com/shiywang))
* New command `apply view-last-applied` *command for viewing the last configuration file applied* ([#41146](https://github.com/kubernetes/kubernetes/pull/41146), [@shiywang](https://github.com/shiywang))
* `apply` now supports explicitly clearing values by setting them to null ([#40630](https://github.com/kubernetes/kubernetes/pull/40630), [@liggitt](https://github.com/liggitt))
* Warn user when using `apply` on a resource lacking the `LastAppliedConfig` annotation ([#36672](https://github.com/kubernetes/kubernetes/pull/36672), [@ymqytw](https://github.com/ymqytw))
* PATCH (i.e. apply and edit) now supports merging lists of primitives ([#38665](https://github.com/kubernetes/kubernetes/pull/38665), [@ymqytw](https://github.com/ymqytw))
* `apply` falls back to generic 3-way JSON merge patch for Third Party Resource or unregistered types ([#40666](https://github.com/kubernetes/kubernetes/pull/40666), [@ymqytw](https://github.com/ymqytw))

#### Updates to edit
* `edit` now supports Third party resources and extension API servers. ([#41304](https://github.com/kubernetes/kubernetes/pull/41304), [@liggitt](https://github.com/liggitt))
  * Now to edit a particular API version, provide the fully-qualify the resource, version, and group used to fetch the object (for example, `job.v1.batch/myjob`)
  * Client-side conversion is no longer done, and the `--output-version` option is deprecated for `kubectl edit`.
* `edit` works as intended with apply and will not change the annotation
  * No longer updates the last-applied-configuration annotation when --save-config is unspecified or false. ([#41924](https://github.com/kubernetes/kubernetes/pull/41924), [@ymqytw](https://github.com/ymqytw))
  * Fixes issue that caused apply to revert changes made by edit

#### Bug fixes
* Fixed --save-config in create subcommand to save the annotation ([#40289](https://github.com/kubernetes/kubernetes/pull/40289), [@xilabao](https://github.com/xilabao))
* Fixed an issue where 'kubectl get --sort-by=' would return an error if the specified fields in sort were not specified in one or more of the returned objects. ([#40541](https://github.com/kubernetes/kubernetes/pull/40541), [@fabianofranz](https://github.com/fabianofranz))
  * Previously this would cause the command to fail regardless of whether or not the field was present in the object model
  * Now the command will succeed even if the sort-by field is missing from one or more of the objects
* Fixed issue with kubectl proxy so it will now proxy an empty path - e.g. http://localhost:8001 ([#39226](https://github.com/kubernetes/kubernetes/pull/39226), [@luksa](https://github.com/luksa))
* Fixed an issue where commas were not accepted in --from-literal flags for the creation of secrets. ([#35191](https://github.com/kubernetes/kubernetes/pull/35191), [@SamiHiltunen](https://github.com/SamiHiltunen))
  * Passing multiple values separated by a comma in a single --from-literal flag is no longer supported. Please use multiple --from-literal flags to provide multiple values.
* Fixed a bug where the --server, --token, and --certificate-authority flags were not overriding the related in-cluster configs when provided in a `kubectl` call inside a cluster. ([#39006](https://github.com/kubernetes/kubernetes/pull/39006), [@fabianofranz](https://github.com/fabianofranz))

#### Other Notable Changes
* The api server will publish the extensions/Deployments API as preferred over the apps/Deployment (introduced in 1.6). ([#43553](https://github.com/kubernetes/kubernetes/pull/43553), [@liggitt](https://github.com/liggitt))
  * This will ensure certain commands in 1.5 versions of kubectl continue function when run against a 1.6 server. (e.g. `kubectl edit deployment`)
* Taint
  * The `taint` command will not function in a skewed 1.5 / 1.6 environment - client and server versions must match (See Action required section above)
  * Change taints/tolerations to api fields ([#38957](https://github.com/kubernetes/kubernetes/pull/38957), [@aveshagarwal](https://github.com/aveshagarwal))
  * Make kubectl taint command respect effect NoExecute ([#42120](https://github.com/kubernetes/kubernetes/pull/42120), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Allow drain --force to remove pods whose managing resource is deleted. ([#41864](https://github.com/kubernetes/kubernetes/pull/41864), [@marun](https://github.com/marun))
* `--output-version` is ignored for all commands except `kubectl convert`.  This is consistent with the generic nature of `kubectl` CRUD commands and the previous removal of `--api-version`.  Specific versions can be specified in the resource field: `resource.version.group`, `jobs.v1.batch`. ([#41576](https://github.com/kubernetes/kubernetes/pull/41576), [@deads2k](https://github.com/deads2k))
* Allow missing keys in templates by default ([#39486](https://github.com/kubernetes/kubernetes/pull/39486), [@ncdc](https://github.com/ncdc))
* Add error message when trying to use clusterrole with namespace in kubectl ([#36424](https://github.com/kubernetes/kubernetes/pull/36424), [@xilabao](https://github.com/xilabao))
* When deleting an object with `--grace-period=0`, the client will begin a graceful deletion and wait until the resource is fully deleted.  To force deletion, use the `--force` flag. ([#37263](https://github.com/kubernetes/kubernetes/pull/37263), [@smarterclayton](https://github.com/smarterclayton))

### Node Components
* Kubelet config should ignore file start with dots.
  ([#39196](https://github.com/kubernetes/kubernetes/pull/39196), [@resouer](https://github.com/resouer))
* Bump GCI to gci-stable-56-9000-84-2.
  ([#41819](https://github.com/kubernetes/kubernetes/pull/41819),
  [@dchen1107](https://github.com/dchen1107))
* Bump GCE ContainerVM to container-vm-v20170214 to address CVE-2016-9962.
  ([#41449](https://github.com/kubernetes/kubernetes/pull/41449),
  [@zmerlynn](https://github.com/zmerlynn))
* Kubelet: Remove the PLEG health check from /healthz, Kubelet will now report
* NodeNotReady on failed PLEG health check.
  ([#41569](https://github.com/kubernetes/kubernetes/pull/41569),
  [@yujuhong](https://github.com/yujuhong))
* CRI: upgrade protobuf to v3. Protobuf v2 and v3 are not compatible.
  ([#39158](https://github.com/kubernetes/kubernetes/pull/39158), [@feiskyer](https://github.com/feiskyer))
* kubelet exports metrics for cgroup management ([#41988](https://github.com/kubernetes/kubernetes/pull/41988), [@sjenning](https://github.com/sjenning))
* Experimental support to reserve a pod's memory request from being utilized by
  pods in lower QoS tiers.
  ([#41149](https://github.com/kubernetes/kubernetes/pull/41149),
  [@sjenning](https://github.com/sjenning))
* Port forwarding can forward over websockets or SPDY.
  ([#33684](https://github.com/kubernetes/kubernetes/pull/33684),
  [@fraenkel](https://github.com/fraenkel))
* Flag gate faster evictions based on node memory pressure using kernel memcg
  notifications - `--experimental-kernel-memcg-notification`.
  ([#38258](https://github.com/kubernetes/kubernetes/pull/38258),
  [@derekwaynecarr](https://github.com/derekwaynecarr))
* Nodes can now report two additional address types in their status: InternalDNS
  and ExternalDNS. The apiserver can use --kubelet-preferred-address-types to
  give priority to the type of address it uses to reach nodes.
  ([#34259](https://github.com/kubernetes/kubernetes/pull/34259), [@liggitt](https://github.com/liggitt))

#### Bug fixes
* Add image cache to fix the issue where kubelet hands when reporting the node
  status. ([#38375](https://github.com/kubernetes/kubernetes/pull/38375),
  [@Random-Liu](https://github.com/Random-Liu))
* Fix logic error in graceful deletion that caused pods not being able to
  be deleted. ([#37721](https://github.com/kubernetes/kubernetes/pull/37721),
  [@derekwaynecarr](https://github.com/derekwaynecarr))
* Fix ConfigMap for Windows Containers.
  ([#39373](https://github.com/kubernetes/kubernetes/pull/39373),
  [@jbhurat](https://github.com/jbhurat))
* Fix the “pod rejected by kubelet may stay at pending forever” bug.
  (https://github.com/kubernetes/kubernetes/pull/37661),
  [@yujuhong](https://github.com/yujuhong))

### kube-controller-manager
* add --controllers to controller manager ([#39740](https://github.com/kubernetes/kubernetes/pull/39740), [@deads2k](https://github.com/deads2k))

### kube-dns
* Adds support for configurable DNS stub domains and upstream nameservers.
  The following configuration options have been added to the `kube-system:kube-dns` ConfigMap:
  ```
  "stubDomains": {
    "acme.local": ["1.2.3.4"]
  },
  ```
  is a map of domain to list of nameservers for the domain. This is used
  to inject private DNS domains into the kube-dns namespace. In the above
  example, any DNS requests for *.acme.local will be served by the
  nameserver 1.2.3.4.
  ```
  "upstreamNameservers": ["8.8.8.8", "8.8.4.4"]
  ```
  is a list of upstreamNameservers to use, overriding the configuration
  specified in /etc/resolv.conf.

* An empty `kube-system:kube-dns` ConfigMap will be created for the cluster if one did not already exist.

### kube-proxy
* **- Add tcp/udp userspace proxy support for Windows. ([#41487](https://github.com/kubernetes/kubernetes/pull/41487), [@anhowe](https://github.com/anhowe))**
* Add DNS suffix search list support in Windows kube-proxy. ([#41618](https://github.com/kubernetes/kubernetes/pull/41618), [@JiangtianLi](https://github.com/JiangtianLi))
* Add a KUBERNETES_NODE_* section to build kubelet/kube-proxy for windows ([#38919](https://github.com/kubernetes/kubernetes/pull/38919), [@brendandburns](https://github.com/brendandburns))
* Remove outdated net.experimental.kubernetes.io/proxy-mode and net.beta.kubernetes.io/proxy-mode annotations from kube-proxy. ([#40585](https://github.com/kubernetes/kubernetes/pull/40585), [@cblecker](https://github.com/cblecker))
* proxy/iptables: don't sync proxy rules if services map didn't change ([#38996](https://github.com/kubernetes/kubernetes/pull/38996), [@dcbw](https://github.com/dcbw))
* Update kube-proxy image to be based off of Debian 8.6 base image. ([#39695](https://github.com/kubernetes/kubernetes/pull/39695), [@ixdy](https://github.com/ixdy))
* Update amd64 kube-proxy base image to debian-iptables-amd64:v5 ([#39725](https://github.com/kubernetes/kubernetes/pull/39725), [@ixdy](https://github.com/ixdy))
* Clean up the kube-proxy container image by removing unnecessary packages and files. ([#42090](https://github.com/kubernetes/kubernetes/pull/42090), [@timstclair](https://github.com/timstclair))
* Better compat with very old iptables (e.g. CentOS 6) ([#37594](https://github.com/kubernetes/kubernetes/pull/37594), [@thockin](https://github.com/thockin))

### Scheduler
* Add the support to the scheduler for spreading pods of StatefulSets. ([#41708](https://github.com/kubernetes/kubernetes/pull/41708), [@bsalamat](https://github.com/bsalamat))
* Added support to minimize sending verbose node information to scheduler extender by sending only node names and expecting extenders to cache the rest of the node information ([#41119](https://github.com/kubernetes/kubernetes/pull/41119), [@sarat-k](https://github.com/sarat-k))
* Support KUBE_MAX_PD_VOLS on Azure ([#41398](https://github.com/kubernetes/kubernetes/pull/41398), [@codablock](https://github.com/codablock))
* Mark multi-scheduler graduation to beta and then v1. ([#38871](https://github.com/kubernetes/kubernetes/pull/38871), [@k82cn](https://github.com/k82cn))
* Scheduler treats StatefulSet pods as belonging to a single equivalence class. ([#39718](https://github.com/kubernetes/kubernetes/pull/39718), [@foxish](https://github.com/foxish))
* Update FitError as a message component into the PodConditionUpdater. ([#39491](https://github.com/kubernetes/kubernetes/pull/39491), [@jayunit100](https://github.com/jayunit100))
* Fix comment and optimize code ([#38084](https://github.com/kubernetes/kubernetes/pull/38084), [@tanshanshan](https://github.com/tanshanshan))
* Add flag to enable contention profiling in scheduler. ([#37357](https://github.com/kubernetes/kubernetes/pull/37357), [@gmarek](https://github.com/gmarek))
* Try self-repair scheduler cache or panic ([#37379](https://github.com/kubernetes/kubernetes/pull/37379), [@wojtek-t](https://github.com/wojtek-t))

### Volume Plugins

#### Azure Disk
* restrict name length for Azure specifications ([#40030](https://github.com/kubernetes/kubernetes/pull/40030), [@colemickens](https://github.com/colemickens))

#### GlusterFS
* The glusterfs dynamic volume provisioner will now choose a unique GID for new persistent volumes from a range that can be configured in the storage class with the "gidMin" and "gidMax" parameters. The default range is 2000 - 2147483647 (max int32). ([#37886](https://github.com/kubernetes/kubernetes/pull/37886), [@obnoxxx](https://github.com/obnoxxx))

#### Photon
* Fix photon controller plugin to construct with correct PdID ([#37167](https://github.com/kubernetes/kubernetes/pull/37167), [@luomiao](https://github.com/luomiao))

#### rbd
* force unlock rbd image if the image is not used ([#41597](https://github.com/kubernetes/kubernetes/pull/41597), [@rootfs](https://github.com/rootfs))

#### vSphere
* Fix fsGroup to vSphere ([#38655](https://github.com/kubernetes/kubernetes/pull/38655), [@abrarshivani](https://github.com/abrarshivani))
* Fix issue when attempting to unmount a wrong vSphere volume ([#37413](https://github.com/kubernetes/kubernetes/pull/37413), [@BaluDontu](https://github.com/BaluDontu))

#### Other Notable Changes
* Implement bulk polling of volumes ([#41306](https://github.com/kubernetes/kubernetes/pull/41306), [@gnufied](https://github.com/gnufied))
* Check if pathExists before performing Unmount ([#39311](https://github.com/kubernetes/kubernetes/pull/39311), [@rkouj](https://github.com/rkouj))
* Unmount operation should not fail if volume is already unmounted ([#38547](https://github.com/kubernetes/kubernetes/pull/38547), [@rkouj](https://github.com/rkouj))
* Provide kubernetes-controller-manager flags to control volume attach/detach reconciler sync.  The duration of the syncs can be controlled, and the syncs can be shut off as well. ([#39551](https://github.com/kubernetes/kubernetes/pull/39551), [@chrislovecnm](https://github.com/chrislovecnm))
* Fix unmountDevice issue caused by shared mount in GCI ([#38411](https://github.com/kubernetes/kubernetes/pull/38411), [@jingxu97](https://github.com/jingxu97))
* Fix permissions when using fsGroup ([#37009](https://github.com/kubernetes/kubernetes/pull/37009), [@sjenning](https://github.com/sjenning))
* Fixed issues ([#39202](https://github.com/kubernetes/kubernetes/pull/39202)), ([#41041](https://github.com/kubernetes/kubernetes/pull/41041)) and ([#40941](https://github.com/kubernetes/kubernetes/pull/40941)) that caused the iSCSI connections to
  be prematurely closed when deleting a pod with an iSCSI persistent volume attached and that prevented the use of newly created LUNs on targets with preestablished connections. ([#41196](https://github.com/kubernetes/kubernetes/pull/41196)), [@CristianPop](https://github.com/CristianPop))

## Changes to Cluster Provisioning Scripts

### AWS
* Deployment of AWS Kubernetes clusters using the in-tree bash deployment (i.e. cluster/kube-up.sh or get-kube.sh) is obsolete. v1.5.x will be the last release to support cluster/kube-up.sh with AWS. For a list of viable alternatives, see: http://kubernetes.io/docs/getting-started-guides/aws/  ([#42196](https://github.com/kubernetes/kubernetes/pull/42196), [@zmerlynn](https://github.com/zmerlynn))
* Fix an issue where AWS tear-down leaks an DHCP Option Set. ([#38645](https://github.com/kubernetes/kubernetes/pull/38645), [@zmerlynn](https://github.com/zmerlynn))

### Juju
* The kubernetes-master, kubernetes-worker and kubeapi-load-balancer charms have gained an nrpe-external-master relation, allowing the integration of their monitoring in an external Nagios server. ([#41923](https://github.com/kubernetes/kubernetes/pull/41923), [@Cynerva](https://github.com/Cynerva))
* Disable anonymous auth on kubelet ([#41919](https://github.com/kubernetes/kubernetes/pull/41919), [@Cynerva](https://github.com/Cynerva))
* Fix shebangs in charm actions to use python3 ([#42058](https://github.com/kubernetes/kubernetes/pull/42058), [@Cynerva](https://github.com/Cynerva))
* K8s master charm now properly keeps distributed master files in sync for an HA control plane. ([#41351](https://github.com/kubernetes/kubernetes/pull/41351), [@chuckbutler](https://github.com/chuckbutler))
* Improve status messages ([#40691](https://github.com/kubernetes/kubernetes/pull/40691), [@Cynerva](https://github.com/Cynerva))
* Splits Juju Charm layers into master/worker roles ([#40324](https://github.com/kubernetes/kubernetes/pull/40324), [@chuckbutler](https://github.com/chuckbutler))
    * Adds support for 1.5.x series of Kubernetes
    * Introduces a tactic for keeping templates in sync with upstream eliminating template drift
    * Adds CNI support to the Juju Charms
    * Adds durable storage support to the Juju Charms
    * Introduces an e2e Charm layer for repeatable testing efforts and validation of clusters

### libvirt CoreOS
* To add local registry to libvirt_coreos ([#36751](https://github.com/kubernetes/kubernetes/pull/36751), [@sdminonne](https://github.com/sdminonne))

### GCE
* **the `gce` provider enables both RBAC authorization and the permissive legacy ABAC policy that makes all service accounts superusers. To opt out of the permissive ABAC policy, export the environment variable `ENABLE_LEGACY_ABAC=false` before running `cluster/kube-up.sh`. ([#43544](https://github.com/kubernetes/kubernetes/pull/43544), [@liggitt](https://github.com/liggitt))**
* **the `gce` provider ensures the bootstrap admin token user is included in the super-user group ([#39537](https://github.com/kubernetes/kubernetes/pull/39537), [@liggitt](https://github.com/liggitt))**
* Remove support for debian masters in GCE kube-up. ([#41666](https://github.com/kubernetes/kubernetes/pull/41666), [@mikedanese](https://github.com/mikedanese))
* Remove support for trusty in GCE kube-up. ([#41670](https://github.com/kubernetes/kubernetes/pull/41670), [@mikedanese](https://github.com/mikedanese))
* Don't fail if the grep fails to match any resources ([#41933](https://github.com/kubernetes/kubernetes/pull/41933), [@ixdy](https://github.com/ixdy))
* Fix the output of health-mointor.sh ([#41525](https://github.com/kubernetes/kubernetes/pull/41525), [@yujuhong](https://github.com/yujuhong))
* Added configurable etcd initial-cluster-state to kube-up script. ([#41332](https://github.com/kubernetes/kubernetes/pull/41332), [@jszczepkowski](https://github.com/jszczepkowski))
* The kube-apiserver [basic audit log](https://kubernetes.io/docs/admin/audit/) can be enabled in GCE by exporting the environment variable `ENABLE_APISERVER_BASIC_AUDIT=true` before running `cluster/kube-up.sh`. This will log to `/var/log/kube-apiserver-audit.log` and use the same `logrotate` settings as `/var/log/kube-apiserver.log`. ([#41211](https://github.com/kubernetes/kubernetes/pull/41211), [@enisoc](https://github.com/enisoc))
* On kube-up.sh clusters on GCE, kube-scheduler now contacts the API on the secured port. ([#41285](https://github.com/kubernetes/kubernetes/pull/41285), [@liggitt](https://github.com/liggitt))
* Use existing ABAC policy file when upgrading GCE cluster ([#40172](https://github.com/kubernetes/kubernetes/pull/40172), [@liggitt](https://github.com/liggitt))
* Ensure the GCI metadata files do not have newline at the end ([#38727](https://github.com/kubernetes/kubernetes/pull/38727), [@Amey-D](https://github.com/Amey-D))
* Fixed detection of master during creation of multizone nodes cluster by kube-up. ([#38617](https://github.com/kubernetes/kubernetes/pull/38617), [@jszczepkowski](https://github.com/jszczepkowski))
* Fixed validation of multizone cluster for GCE ([#38695](https://github.com/kubernetes/kubernetes/pull/38695), [@jszczepkowski](https://github.com/jszczepkowski))
* Fix GCI mounter issue ([#38124](https://github.com/kubernetes/kubernetes/pull/38124), [@jingxu97](https://github.com/jingxu97))
* Exit with error if <version number or publication> is not the final parameter. ([#37723](https://github.com/kubernetes/kubernetes/pull/37723), [@mtaufen](https://github.com/mtaufen))
* GCI: Remove /var/lib/docker/network ([#37593](https://github.com/kubernetes/kubernetes/pull/37593), [@yujuhong](https://github.com/yujuhong))
* Fix the equality checks for numeric values in cluster/gce/util.sh. ([#37638](https://github.com/kubernetes/kubernetes/pull/37638), [@roberthbailey](https://github.com/roberthbailey))
* Modify GCI mounter to enable NFSv3 ([#37582](https://github.com/kubernetes/kubernetes/pull/37582), [@jingxu97](https://github.com/jingxu97))
* Use gsed on the Mac ([#37562](https://github.com/kubernetes/kubernetes/pull/37562), [@roberthbailey](https://github.com/roberthbailey))
* Bump GCI
* to gci-beta-56-9000-80-0 ([#41027](https://github.com/kubernetes/kubernetes/pull/41027), [@dchen1107](https://github.com/dchen1107))
* to gci-stable-56-9000-84-2 ([#41819](https://github.com/kubernetes/kubernetes/pull/41819), [@dchen1107](https://github.com/dchen1107))
* Bump GCE ContainerVM
  * to container-vm-v20161208 ([release notes](https://cloud.google.com/compute/docs/containers/container_vms#changelog)) ([#38432](https://github.com/kubernetes/kubernetes/pull/38432), [@timstclair](https://github.com/timstclair))
  * to container-vm-v20170201 to address CVE-2016-9962. ([#40828](https://github.com/kubernetes/kubernetes/pull/40828), [@zmerlynn](https://github.com/zmerlynn))
  * to container-vm-v20170117 to pick up CVE fixes in base image. ([#40094](https://github.com/kubernetes/kubernetes/pull/40094), [@zmerlynn](https://github.com/zmerlynn))
  * to container-vm-v20170214 to address CVE-2016-9962. ([#41449](https://github.com/kubernetes/kubernetes/pull/41449), [@zmerlynn](https://github.com/zmerlynn))

### OpenStack
* Do not daemonize `salt-minion` for the openstack-heat provider. ([#40722](https://github.com/kubernetes/kubernetes/pull/40722), [@micmro](https://github.com/micmro))
* OpenStack-Heat will now look for an image named "CentOS-7-x86_64-GenericCloud-1604". To restore the previous behavior set OPENSTACK_IMAGE_NAME="CentOS7" ([#40368](https://github.com/kubernetes/kubernetes/pull/40368), [@sc68cal](https://github.com/sc68cal))
* Fixes a bug in the OpenStack-Heat kubernetes provider, in the handling of differences between the Identity v2 and Identity v3 APIs ([#40105](https://github.com/kubernetes/kubernetes/pull/40105), [@sc68cal](https://github.com/sc68cal))

### Container Images
* Update gcr.io/google-containers/rescheduler to v0.2.2, which uses busybox as a base image instead of ubuntu. ([#41911](https://github.com/kubernetes/kubernetes/pull/41911), [@ixdy](https://github.com/ixdy))
* Remove unnecessary metrics (http/process/go) from being exposed by etcd-version-monitor ([#41807](https://github.com/kubernetes/kubernetes/pull/41807), [@shyamjvs](https://github.com/shyamjvs))
* Align the hyperkube image to support running binaries at /usr/local/bin/ like the other server images ([#41017](https://github.com/kubernetes/kubernetes/pull/41017), [@luxas](https://github.com/luxas))
* Bump up GLBC version from 0.9.0-beta to 0.9.1 ([#41037](https://github.com/kubernetes/kubernetes/pull/41037), [@bprashanth](https://github.com/bprashanth))

### Other Notable Changes
* The default client certificate generated by kube-up now contains the superuser `system:masters` group ([#39966](https://github.com/kubernetes/kubernetes/pull/39966), [@liggitt](https://github.com/liggitt))
* Added support for creating HA clusters for centos using kube-up.sh. ([#39462](https://github.com/kubernetes/kubernetes/pull/39462), [@Shawyeok](https://github.com/Shawyeok))
* Enable lazy inode table and journal initialization for ext3 and ext4 ([#38865](https://github.com/kubernetes/kubernetes/pull/38865), [@codablock](https://github.com/codablock))
* Since `kubernetes.tar.gz` no longer includes client or server binaries, `cluster/kube-{up,down,push}.sh` now automatically download released binaries if they are missing. ([#38730](https://github.com/kubernetes/kubernetes/pull/38730), [@ixdy](https://github.com/ixdy))
* Fix broken cluster/centos and enhance the style ([#34002](https://github.com/kubernetes/kubernetes/pull/34002), [@xiaoping378](https://github.com/xiaoping378))
* Set kernel.softlockup_panic =1 based on the flag. ([#38001](https://github.com/kubernetes/kubernetes/pull/38001), [@dchen1107](https://github.com/dchen1107))
* Configure local-up-cluster.sh to handle auth proxies ([#36838](https://github.com/kubernetes/kubernetes/pull/36838), [@deads2k](https://github.com/deads2k))
* `kube-up.sh`/`kube-down.sh` no longer force update gcloud for provider=gce|gke. ([#36292](https://github.com/kubernetes/kubernetes/pull/36292), [@jlowdermilk](https://github.com/jlowdermilk))
* Collect logs for dead kubelets too ([#37671](https://github.com/kubernetes/kubernetes/pull/37671), [@mtaufen](https://github.com/mtaufen))

## Changes to Addons

### Dashboard
* Update dashboard version to v1.6.0 ([#43210](https://github.com/kubernetes/kubernetes/pull/43210), [@floreks](https://github.com/floreks))

### DNS
* Updates the dnsmasq cache/mux layer to be managed by dnsmasq-nanny. ([#41826](https://github.com/kubernetes/kubernetes/pull/41826), [@bowei](https://github.com/bowei))
  dnsmasq-nanny manages dnsmasq based on values from the
  kube-system:kube-dns configmap:
  ```
  "stubDomains": {
  "acme.local": ["1.2.3.4"]
   },
  ```

  is a map of domain to list of nameservers for the domain. This is used
  to inject private DNS domains into the kube-dns namespace. In the above
  example, any DNS requests for `*.acme.local` will be served by the
  ```
  nameserver 1.2.3.4.
  ```

  ```
  upstreamNameservers": ["8.8.8.8", "8.8.4.4"]
  ```
  is a list of upstreamNameservers to use, overriding the configuration
  specified in `/etc/resolv.conf`.
* `kube-dns` now runs using a separate `system:serviceaccount:kube-system:kube-dns` service account which is automatically bound to the correct RBAC permissions. ([#38816](https://github.com/kubernetes/kubernetes/pull/38816), [@deads2k](https://github.com/deads2k))
* Use kube-dns:1.11.0 ([#39925](https://github.com/kubernetes/kubernetes/pull/39925), [@sadlil](https://github.com/sadlil))

### DNS Autoscaler
* Patch CVE-2016-8859 in gcr.io/google-containers/cluster-proportional-autoscaler-amd64 ([#42933](https://github.com/kubernetes/kubernetes/pull/42933), [@timstclair](https://github.com/timstclair))

### Cluster Autoscaler
* Allow the Horizontal Pod Autoscaler controller to talk to the metrics API and custom metrics API as standard APIs. ([#41824](https://github.com/kubernetes/kubernetes/pull/41824), [@DirectXMan12](https://github.com/DirectXMan12))

### Cluster Load Balancing
* Update defaultbackend image to 1.3 ([#42212](https://github.com/kubernetes/kubernetes/pull/42212), [@timstclair](https://github.com/timstclair))

### etcd Empty Dir Cleanup
* Base etcd-empty-dir-cleanup on busybox, run as nobody, and update to etcdctl 3.0.14 ([#41674](https://github.com/kubernetes/kubernetes/pull/41674), [@ixdy](https://github.com/ixdy))

### Fluentd
* Migrated fluentd addon to daemon set ([#32088](https://github.com/kubernetes/kubernetes/pull/32088), [@piosz](https://github.com/piosz))
* Fluentd was migrated to Daemon Set, which targets nodes with beta.kubernetes.io/fluentd-ds-ready=true label. If you use fluentd in your cluster please make sure that the nodes with version 1.6+ contains this label. ([#42931](https://github.com/kubernetes/kubernetes/pull/42931), [@piosz](https://github.com/piosz))
* Fluentd-gcp containers spawned by DaemonSet are now configured using ConfigMap ([#42126](https://github.com/kubernetes/kubernetes/pull/42126), [@crassirostris](https://github.com/crassirostris))
* Cleanup fluentd-gcp image: rebase on debian-base, switch to upstream packages, remove fluent-ui & rails ([#41998](https://github.com/kubernetes/kubernetes/pull/41998), [@timstclair](https://github.com/timstclair))
* On GCE, the apiserver audit log (`/var/log/kube-apiserver-audit.log`) will be sent through fluentd if enabled. It will go to the same place as `kube-apiserver.log`, but tagged as its own stream. ([#41360](https://github.com/kubernetes/kubernetes/pull/41360), [@enisoc](https://github.com/enisoc))
* If `experimentalCriticalPodAnnotation` feature gate is set to true, fluentd pods will not be evicted by the kubelet. ([#41035](https://github.com/kubernetes/kubernetes/pull/41035), [@vishh](https://github.com/vishh))
* fluentd config for GKE clusters updated: detect exceptions in container log streams and forward them as one log entry. ([#39656](https://github.com/kubernetes/kubernetes/pull/39656), [@thomasschickinger](https://github.com/thomasschickinger))
* Make fluentd pods critical ([#39146](https://github.com/kubernetes/kubernetes/pull/39146), [@crassirostris](https://github.com/crassirostris))
* Fluentd/Elastisearch add-on: correctly parse and index kubernetes labels ([#36857](https://github.com/kubernetes/kubernetes/pull/36857), [@Shrugs](https://github.com/Shrugs))

### Heapster
* Bumped Heapster to v1.3.0. ([#43298](https://github.com/kubernetes/kubernetes/pull/43298), [@piosz](https://github.com/piosz))
  * More details about the release https://github.com/kubernetes/heapster/releases/tag/v1.3.0

### Registry
* Use daemonset in docker registry add on ([#35582](https://github.com/kubernetes/kubernetes/pull/35582), [@surajssd](https://github.com/surajssd))
* contribute deis/registry-proxy as a replacement for kube-registry-proxy ([#35797](https://github.com/kubernetes/kubernetes/pull/35797), [@bacongobbler](https://github.com/bacongobbler))

## External Dependency Version Information

Continuous integration builds have used the following versions of external dependencies, however, this is not a strong recommendation and users should consult an appropriate installation or upgrade guide before deciding what versions of etcd, docker or rkt to use.

* Docker versions 1.10.3, 1.11.2, 1.12.6 have been validated
  * Docker version 1.12.6 known issues
    * overlay2 driver not fully supported
    * live-restore not fully supported
    * no shared pid namespace support
  * Docker version 1.11.2 known issues
    * Kernel crash with Aufs storage driver on Debian Jessie ([#27885](https://github.com/kubernetes/kubernetes/issues/27885))
      which can be identified by the [node problem detector](http://kubernetes.io/docs/admin/node-problem/)
    * Leaked File descriptors ([#275](https://github.com/docker/containerd/issues/275))
    * Additional memory overhead per container ([#21737](https://github.com/docker/docker/issues/21737))
  * Docker 1.10.3 contains [backports provided by RedHat](https://github.com/docker/docker/compare/v1.10.3...runcom:docker-1.10.3-stable) for known issues
  * Support for Docker version 1.9.x has been removed
* rkt version 1.23.0+
  * known issues with the rkt runtime are [listed in the Getting Started Guide](http://kubernetes.io/docs/getting-started-guides/rkt/notes/)
* etcd version 3.0.17
## Changelog since v1.6.0-rc.1


### Previous Releases Included in v1.6.0
- [v1.6.0-rc.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-rc1)
- [v1.6.0-beta.4](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-beta4)
- [v1.6.0-beta.3](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-beta3)
- [v1.6.0-beta.2](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-beta2)
- [v1.6.0-beta.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-beta1)
- [v1.6.0-alpha.3](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-alpha3)
- [v1.6.0-alpha.2](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-alpha2)
- [v1.6.0-alpha.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v160-alpha1)
**No notable changes for this release**



# v1.6.0-rc.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.0-rc.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes.tar.gz) | `b92be4b71184888ba4a2781d4068ea369442b6030dfb38e23029192a554651e5`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-src.tar.gz) | `e45e993edfdba176a6750c6d3c2207d54d60b5b1fc80a0fe47274d4d9b233d66`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-darwin-386.tar.gz) | `1d56088fb85fba02362f7f87a5d5f899c05caa605f4d11b8616749cb0d970384`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | `f3df7b558c2ecf6ed8344668515f436b7211a2f840d982f81c55586e1ec84a7b`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-linux-386.tar.gz) | `c5ee1787d69d508d8448675428936d70e21f17b21ff44e22db4462483adcebe2`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | `0960505da11330c8cc66b7df4e4413680afd2a62afc2341bad6bbd88c73e3a56`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | `dc113881b9cd09ef8cecbdf8f4ff41eddeba7df3ad7af70461e513eb79757e54`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-linux-arm.tar.gz) | `5f6bd182852ffe3776b520fbf2db3546c8246133df166dcf6c81ece4b0974227`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | `ea91e79a779eac8c00a5eb80be1fd5b227b9f5ae767e30a12354bfa691f198d5`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | `874d7410078d39b80fe07a44018f6e95655cb9e05c99ec66dea012d06633fbbb`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-windows-386.tar.gz) | `b02d6d8e436322294a65def8c9c576f232df9387093c4ca61e57dd3bdf184b87`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | `68dbf06824b3785027a85d448f9f2b9928a4092912b382e67c1579e30bb58bbd`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | `aa9e5d1cb60c1d33d048a75003fdce9ffa0985ede3748b38b4357d961943d603`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | `0343a1dead1efb8b829ab485028d2ec58ffc4aa7845b3415da5a5bb6fd8bcbfd`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-server-linux-arm.tar.gz) | `77a724b28e071e92113759440fdca7e12ea00c5b41a5334ce7581a0139d8f264`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | `ae21bc7cced29a3c20c76dbf57262a8ea276fe120e411d950ff5267fe4b6cd50`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | `424f5cd9f4aee3e53a8a760ccdc16bd7e2683913e57d56519b536bf4a98f56e5`

## Changelog since v1.6.0-beta.4

### Other notable changes

* `kube-up.sh` using the `gce` provider enables both RBAC authorization and the permissive legacy ABAC policy that makes all service accounts superusers. To opt out of the permissive ABAC policy, export the environment variable `ENABLE_LEGACY_ABAC=false` before running `cluster/kube-up.sh`. ([#43544](https://github.com/kubernetes/kubernetes/pull/43544), [@liggitt](https://github.com/liggitt))
* Bump CNI consumers to v0.5.1 ([#43546](https://github.com/kubernetes/kubernetes/pull/43546), [@calebamiles](https://github.com/calebamiles))
* The API server discovery document now prioritizes the `extensions` API group over the `apps` API group. This ensures certain commands in 1.5 versions of kubectl (such as `kubectl edit deployment`) continue to function against a 1.6 API. ([#43553](https://github.com/kubernetes/kubernetes/pull/43553), [@liggitt](https://github.com/liggitt))
* Fix adding disks to more than one scsi adapter. ([#42422](https://github.com/kubernetes/kubernetes/pull/42422), [@kerneltime](https://github.com/kerneltime))
* PodSecurityPolicy authorization is correctly enforced by the PodSecurityPolicy admission plugin. ([#43489](https://github.com/kubernetes/kubernetes/pull/43489), [@liggitt](https://github.com/liggitt))
* API fields that previously serialized null arrays as `null` and empty arrays as `[]` no longer distinguish between those values and always output `[]` when serializing to JSON. ([#43422](https://github.com/kubernetes/kubernetes/pull/43422), [@liggitt](https://github.com/liggitt))
* Apply taint tolerations for NoExecute for all static pods. ([#43116](https://github.com/kubernetes/kubernetes/pull/43116), [@dchen1107](https://github.com/dchen1107))
* Bumped Heapster to v1.3.0. ([#43298](https://github.com/kubernetes/kubernetes/pull/43298), [@piosz](https://github.com/piosz))
    * More details about the release https://github.com/kubernetes/heapster/releases/tag/v1.3.0



# v1.6.0-beta.4

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.0-beta.4


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes.tar.gz) | `8f308a87bcc367656c777f74270a82ad6986517c28a08c7158c77b1d7954e243`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-src.tar.gz) | `3ba73cf27a05f78026d1cfb3a0e47c6e5e33932aefc630a0a5aa3619561bc4dc`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-darwin-386.tar.gz) | `7007e8024257fd2436c9f68ddb25383e889d58379e30a60c9bb6bffb1a6809df`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-darwin-amd64.tar.gz) | `f1bc3f0c8e4c8c9e0aa2f66fffea163a5bf139d528160eb4266cd5322cf112e1`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-linux-386.tar.gz) | `7ceb47d4b282b31d300aa7a81bf00eef744fb58df58d613e1ea01930287c85d9`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-linux-amd64.tar.gz) | `d25c73f0ebb3338fc3e674d4a667d3023d073b8bc4942eb98f1a3fc9001675ef`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-linux-arm64.tar.gz) | `3d5a1188f638cddad7cd5eca0668d25f46a6a6f80641a8e9e6f3777a23af0f7c`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-linux-arm.tar.gz) | `e9d40ad06385266cd993adf436270972412dd5407d436e682bddf30706fddbda`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-linux-ppc64le.tar.gz) | `17f9a60eb6175e28aa0a9ba534cc0ceda24ff8ab81eaf06d04c362146f707e81`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-linux-s390x.tar.gz) | `30017ac4603bda496a125162cd956e9e874a4d04eff170972c72c8095a9f9121`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-windows-386.tar.gz) | `6165b8d0781894b36b2f2cd72d79063ce95621012cd1ca38bd7d936feeea8416`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-client-windows-amd64.tar.gz) | `ac45e5ddf44dd0a680abc5641ae1cb59ad9c7ab8d4132e3b110ebca7ed2119ac`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-server-linux-amd64.tar.gz) | `a37f0b431aea2cc7e259ddf118fd42315589236106b279de5803e2d50af08531`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-server-linux-arm64.tar.gz) | `e6a5c8a9e59a12df5294766a4e31e08603a041dd75bcc23f19fb7d20d8a30b9a`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-server-linux-arm.tar.gz) | `ebe1ccf95a80a829c294fe8bb216a10a096bc7f311fb0f74b7a121772c4d238b`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-server-linux-ppc64le.tar.gz) | `8a09baa5c2ddfbc579a1601f76b7079dab695c1423d22a04acd039256e26355c`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.4/kubernetes-server-linux-s390x.tar.gz) | `621feb08ac3bee0b9f5b31c648b3011f91883c44954db04268c0da4ef59f16f1`

## Changelog since v1.6.0-beta.3

### Other notable changes

* Update dashboard version to v1.6.0 ([#43210](https://github.com/kubernetes/kubernetes/pull/43210), [@floreks](https://github.com/floreks))
* Update photon controller go SDK in vendor code. ([#43108](https://github.com/kubernetes/kubernetes/pull/43108), [@luomiao](https://github.com/luomiao))
* Fluentd was migrated to Daemon Set, which targets nodes with beta.kubernetes.io/fluentd-ds-ready=true label. If you use fluentd in your cluster please make sure that the nodes with version 1.6+ contains this label. ([#42931](https://github.com/kubernetes/kubernetes/pull/42931), [@piosz](https://github.com/piosz))
* if kube-apiserver is started with `--storage-backend=etcd2`, the media type `application/json` is used. ([#43122](https://github.com/kubernetes/kubernetes/pull/43122), [@liggitt](https://github.com/liggitt))
* Add -p to mkdirs in gci-mounter function of gce configure.sh script ([#43134](https://github.com/kubernetes/kubernetes/pull/43134), [@shyamjvs](https://github.com/shyamjvs))
* Rescheduler uses taints in v1beta1 and will remove old ones (in version v1alpha1) right after its start. ([#43106](https://github.com/kubernetes/kubernetes/pull/43106), [@piosz](https://github.com/piosz))
* kubeadm: `kubeadm reset` won't drain and remove the current node anymore ([#42713](https://github.com/kubernetes/kubernetes/pull/42713), [@luxas](https://github.com/luxas))
* hack/godep-restore.sh: use godep v79 which works ([#42965](https://github.com/kubernetes/kubernetes/pull/42965), [@sttts](https://github.com/sttts))
* Patch CVE-2016-8859 in gcr.io/google-containers/cluster-proportional-autoscaler-amd64 ([#42933](https://github.com/kubernetes/kubernetes/pull/42933), [@timstclair](https://github.com/timstclair))
* Disable devicemapper thin_ls due to excessive iops ([#42899](https://github.com/kubernetes/kubernetes/pull/42899), [@dashpole](https://github.com/dashpole))



# v1.6.0-beta.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.0-beta.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes.tar.gz) | `3903f0a49945abe26f5775c20deb25ba65a8607a2944da8802255bd50d20aca7`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-src.tar.gz) | `62f5f9459c14163e319f878494d10296052d75345da7906e8b38a2d6d0d2a25c`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-darwin-386.tar.gz) | `1294256f09d3a78a05cf2c85466495b08f87830911e68fd0206964f0466682e3`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-darwin-amd64.tar.gz) | `ff6d8561163d9039c807f4cf05144dd3837104b111fac1ae4b667e2b8548d135`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-linux-386.tar.gz) | `d32a07b4a24a88cfee589cff91336e411a89ed470557b8f74f34bb6636adc800`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-linux-amd64.tar.gz) | `e3663e134cd42bbf71f4f6f0395e6c3ea2080d8621bdab9cc668c77f5478196a`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-linux-arm64.tar.gz) | `39465c409396a4cc0ae672f0f0c0db971e482de52e9dff835eb43a8f7e3412e9`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-linux-arm.tar.gz) | `8897b38e59cee396213f50453bdcb88808cd56d63be762362d79454ce526b1ea`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-linux-ppc64le.tar.gz) | `a32b85c5e495dd3645845f2e8ff0eb366fb4ae4795e2bdafceae97cfe71e34b5`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-linux-s390x.tar.gz) | `0af4f0d7778cb67c1acc3b2f3870283e3008c6e1ea8d532c6b90b5a7f1475db8`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-windows-386.tar.gz) | `b298561b924c8c88b062759cc69b733187310a7e1af74b1b3987ed256f422b05`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-client-windows-amd64.tar.gz) | `42ec39178885bb06cba4002844e80948e0c9c3618bfb0a009618a3fab1797a69`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-server-linux-amd64.tar.gz) | `333ea0cf5c25f65dbb5d353cac002af3fa5e6f8431e81eaba2534005164c9ce9`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-server-linux-arm64.tar.gz) | `2979a04409863f6e4dbc745eebfd57ee90e0b38ed4449dcb15cfd87d8f80dadc`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-server-linux-arm.tar.gz) | `2ed1e98b2566b4f552951d9496537b18b28ae53eb9e36c6fd17202e9e498eae5`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-server-linux-ppc64le.tar.gz) | `f4989351a6a98746c1d269d72d2fa87dba8ce782bdfc088d9f7f8d10029aa3fe`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.3/kubernetes-server-linux-s390x.tar.gz) | `31fb764136e97e851d1640464154f3ce4fc696e3286314f538da7b19eed3e2fe`

## Changelog since v1.6.0-beta.2

### Other notable changes

* Introduce new generator for apps/v1beta1 deployments ([#42362](https://github.com/kubernetes/kubernetes/pull/42362), [@soltysh](https://github.com/soltysh))
* Use Prometheus instrumentation conventions ([#36704](https://github.com/kubernetes/kubernetes/pull/36704), [@fabxc](https://github.com/fabxc))
* Add new DaemonSet status fields to kubectl printer and describer.  ([#42843](https://github.com/kubernetes/kubernetes/pull/42843), [@janetkuo](https://github.com/janetkuo))
* Dropped the support for docker 1.9.x and the belows.  ([#42694](https://github.com/kubernetes/kubernetes/pull/42694), [@dchen1107](https://github.com/dchen1107))



# v1.6.0-beta.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.0-beta.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes.tar.gz) | `8199c781f8c98ed7048e939a590ea10a6c39f6a74bd35ed526b459fa18e20f50`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-src.tar.gz) | `f8de26ab6493d4547f9068f5ef396650c169702863535ba63feaf815464a6702`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `0b2350c5ffec582f86d2bd95fa9ecd1b4213fbcd3af79f2a7f67d071c3a0373f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `a1c81457a34258f2622f841b9971ba490c66f6a0f5725c089430d0f0fb09dc8c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-linux-386.tar.gz) | `40d45492f6741980afde0c83bb752382b699b3d62ac36203faca16fcd9fadd21`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `a06baf31249b06375fde1f608ffea041bdbad0f4814ba8ea69839a7778fa4646`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `d9e18ceb7efacee5cc2a579e204919bb4c272c586bc15750963946e7fe5dc741`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `7abc1a2e5c0e40b46b9839b9c9ca065fceec486413ee3c0687e832dc668560ca`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | `84d82f1c2a2b07bea7c827c20cc208f0741b72cf732319673edbf73b42f1b687`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | `605852ee99117abb5bf62f4239c7e2c7e3976f1f497e24ffed50ba4817c301dc`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-windows-386.tar.gz) | `2c442dbfaa393f67f2fe2a1fd2c10267092e99385ca40f7bed732d48bb36ae62`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `6b64521cde0b239d9e23e0896919653dfe30ad064d363a9931305feefe04b359`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `0a49f719cd295be9a4947b7b8b0fe68c29d8168d7e03a9e37173de340490b578`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `fb1464794b9e6375cc7f5b8b72125d81921416b6825fe2c37073aef006e846d1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `27acc02302c6d45ef6314a2bceca6012e35c88d0678b219528d21d8ee4c6b424`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | `8eef21ab0700ba2802ef70d2c0b84ee0b27ae0833d2259d425429984f972690e`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | `c6e383f897ceb8143a7b1f023e155c9c39e9e7c220e989cc6c1bfcffdb886dd5`

## Changelog since v1.6.0-beta.1

### Action Required

* Deployment now fully respects ControllerRef to avoid fighting over Pods and ReplicaSets. At the time of upgrade, **you must not have Deployments with selectors that overlap**, or else [ownership of ReplicaSets may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md#upgrading). ([#42175](https://github.com/kubernetes/kubernetes/pull/42175), [@enisoc](https://github.com/enisoc))
* StatefulSet now respects ControllerRef to avoid fighting over Pods. At the time of upgrade, **you must not have StatefulSets with selectors that overlap** with any other controllers (such as ReplicaSets), or else [ownership of Pods may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md#upgrading). ([#42080](https://github.com/kubernetes/kubernetes/pull/42080), [@enisoc](https://github.com/enisoc))

### Other notable changes

* DaemonSet now respects ControllerRef to avoid fighting over Pods. ([#42173](https://github.com/kubernetes/kubernetes/pull/42173), [@enisoc](https://github.com/enisoc))
* restored normalization of custom `--etcd-prefix` when `--storage-backend` is set to etcd3 ([#42506](https://github.com/kubernetes/kubernetes/pull/42506), [@liggitt](https://github.com/liggitt))
* kubelet created cgroups follow lowercase naming conventions ([#42497](https://github.com/kubernetes/kubernetes/pull/42497), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Support whitespace in command path for gcp auth plugin ([#41653](https://github.com/kubernetes/kubernetes/pull/41653), [@jlowdermilk](https://github.com/jlowdermilk))
* Updates the dnsmasq cache/mux layer to be managed by dnsmasq-nanny. ([#41826](https://github.com/kubernetes/kubernetes/pull/41826), [@bowei](https://github.com/bowei))
    * dnsmasq-nanny manages dnsmasq based on values from the
    * kube-system:kube-dns configmap:
    * "stubDomains": {
    * 	"acme.local": ["1.2.3.4"]
    * },
    * is a map of domain to list of nameservers for the domain. This is used
    * to inject private DNS domains into the kube-dns namespace. In the above
    * example, any DNS requests for *.acme.local will be served by the
    * nameserver 1.2.3.4.
    * "upstreamNameservers": ["8.8.8.8", "8.8.4.4"]
    * is a list of upstreamNameservers to use, overriding the configuration
    * specified in /etc/resolv.conf.
* kubelet exports metrics for cgroup management ([#41988](https://github.com/kubernetes/kubernetes/pull/41988), [@sjenning](https://github.com/sjenning))
* kubectl: respect deployment strategy parameters for rollout status ([#41809](https://github.com/kubernetes/kubernetes/pull/41809), [@kargakis](https://github.com/kargakis))
* Remove cmd/kube-discovery from the tree since it's not necessary anymore ([#42070](https://github.com/kubernetes/kubernetes/pull/42070), [@luxas](https://github.com/luxas))
* kubeadm: Hook up kubeadm against the BootstrapSigner ([#41417](https://github.com/kubernetes/kubernetes/pull/41417), [@luxas](https://github.com/luxas))
* Federated Ingress over GCE no longer requires separate firewall rules to be created for each cluster to circumvent flapping firewall health checks. ([#41942](https://github.com/kubernetes/kubernetes/pull/41942), [@csbell](https://github.com/csbell))
* ScaleIO Kubernetes Volume Plugin added enabling pods to seamlessly access and use data stored on ScaleIO volumes. ([#38924](https://github.com/kubernetes/kubernetes/pull/38924), [@vladimirvivien](https://github.com/vladimirvivien))
* Pods are launched in a separate cgroup hierarchy than system services. ([#42350](https://github.com/kubernetes/kubernetes/pull/42350), [@vishh](https://github.com/vishh))
* Experimental support to reserve a pod's memory request from being utilized by pods in lower QoS tiers. ([#41149](https://github.com/kubernetes/kubernetes/pull/41149), [@sjenning](https://github.com/sjenning))
* Juju: Disable anonymous auth on kubelet ([#41919](https://github.com/kubernetes/kubernetes/pull/41919), [@Cynerva](https://github.com/Cynerva))
* Remove support for debian masters in GCE kube-up. ([#41666](https://github.com/kubernetes/kubernetes/pull/41666), [@mikedanese](https://github.com/mikedanese))
* Implement bulk polling of volumes ([#41306](https://github.com/kubernetes/kubernetes/pull/41306), [@gnufied](https://github.com/gnufied))
* stop kubectl edit from updating the last-applied-configuration annotation when --save-config is unspecified or false. ([#41924](https://github.com/kubernetes/kubernetes/pull/41924), [@ymqytw](https://github.com/ymqytw))



# v1.6.0-beta.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.6/examples)

## Downloads for v1.6.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes.tar.gz) | `ca17c4f1ebdd4bbbd0e570bf3a29d52be1e962742155bc5e20765434f3141f2d`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-src.tar.gz) | `4aefc25b42594f0aab48e43608c8ef6eca8c115022fcc76a9a0d34430e33be0f`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `7629da89467e758e6e70c513d5332e6231941de60e99b6621376bc72f9ede314`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `3a6d6f78ca307486189c7a92e874508233d6b9b5697a0c42cb2803f4b17ccfb2`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-linux-386.tar.gz) | `544b944fdcbebb0dbf0e1acedf7e1deb40fd795c46b8f5afe5d622d2091f0ac9`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `d13f3bede2beb1d7fbca7f01a2c0775938d9127073b0fa1cecba4fd152947eae`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `8820b18ae1c3bdcb8c93b5641e9322aa8dba25ec42362aa86ecbe6ae690a9809`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `d928f7e772a74cf715cf382d66ba757394afcf02a03727edfe43305f279fdb87`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `56ccc3a9def527278cd41ba1ce5b0528238ef7b7b5886d6ebc944b11e2f5228c`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `4923b9617b5306b321e47450bbfe701242b46b2d27800d82a7289fbabe7a107d`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-windows-386.tar.gz) | `598703591fa2be13cc47930088117fc12731431b679f8ca2a5717430bb45fb93`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `9e839346effcbe9c469519002967069c8d109282aaceb72e02f25cf606a691b2`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `726b9e4ead829ebd293fe6674ab334f72aa163b1544963febb9bc35d1fb26e6f`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `975d02629619d441f60442ca07c42721e103e9e5bbcc2eea302b7c936303d26b`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `98223dd80f34eed3cdb30fb57df1da96630db9c0f04aae6a685e22a29c16398d`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `a73715b7db73d6d0ad0b78b01829fe9f22566b557eebe2c1a960a81693b0c8b5`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `d3cb54e9193c773ea9998106c75bb3f0af705477fb844bcc8f82c845c44bb00d`

## Changelog since v1.6.0-alpha.3

### Action Required

* The --dns-provider argument of 'kubefed init' is now mandatory and does not default to `google-clouddns`. To initialize a Federation control plane with Google Cloud DNS, use the following invocation: 'kubefed init --dns-provider=google-clouddns' ([#42092](https://github.com/kubernetes/kubernetes/pull/42092), [@marun](https://github.com/marun))
* Change taints/tolerations to api fields ([#38957](https://github.com/kubernetes/kubernetes/pull/38957), [@aveshagarwal](https://github.com/aveshagarwal))

### Other notable changes

* kubeadm: Rename some flags for beta UI and fixup some logic ([#42064](https://github.com/kubernetes/kubernetes/pull/42064), [@luxas](https://github.com/luxas))
* StorageClassName attribute has been added to PersistentVolume and PersistentVolumeClaim objects and should be used instead of annotation `volume.beta.kubernetes.io/storage-class`. The beta annotation is still working in this release, however it will be removed in a future release. ([#42128](https://github.com/kubernetes/kubernetes/pull/42128), [@jsafrane](https://github.com/jsafrane))
* Remove Azure kube-up as the Azure community has focused efforts elsewhere. ([#41672](https://github.com/kubernetes/kubernetes/pull/41672), [@mikedanese](https://github.com/mikedanese))
* Fluentd-gcp containers spawned by DaemonSet are now configured using ConfigMap ([#42126](https://github.com/kubernetes/kubernetes/pull/42126), [@Crassirostris](https://github.com/Crassirostris))
* Modified kubemark startup scripts to restore master on reboot ([#41980](https://github.com/kubernetes/kubernetes/pull/41980), [@shyamjvs](https://github.com/shyamjvs))
* Added new Api `PodPreset` to enable defining cross-cutting injection of Volumes and Environment into Pods. ([#41931](https://github.com/kubernetes/kubernetes/pull/41931), [@jessfraz](https://github.com/jessfraz))
* AWS cloud provider: allow to run the master with a different AWS account or even on a different cloud provider than the nodes. ([#39996](https://github.com/kubernetes/kubernetes/pull/39996), [@scheeles](https://github.com/scheeles))
* Update defaultbackend image to 1.3 ([#42212](https://github.com/kubernetes/kubernetes/pull/42212), [@timstclair](https://github.com/timstclair))
* Allow the Horizontal Pod Autoscaler controller to talk to the metrics API and custom metrics API as standard APIs. ([#41824](https://github.com/kubernetes/kubernetes/pull/41824), [@DirectXMan12](https://github.com/DirectXMan12))
* Implement support for mount options in PVs ([#41906](https://github.com/kubernetes/kubernetes/pull/41906), [@gnufied](https://github.com/gnufied))
* Introduce apps/v1beta1.Deployments resource with modified defaults compared to extensions/v1beta1.Deployments. ([#39683](https://github.com/kubernetes/kubernetes/pull/39683), [@soltysh](https://github.com/soltysh))
* Add DNS suffix search list support in Windows kube-proxy. ([#41618](https://github.com/kubernetes/kubernetes/pull/41618), [@JiangtianLi](https://github.com/JiangtianLi))
* `--experimental-nvidia-gpus` flag is **replaced** by `Accelerators` alpha feature gate along with  support for multiple Nvidia GPUs.  ([#42116](https://github.com/kubernetes/kubernetes/pull/42116), [@vishh](https://github.com/vishh))
    * To use GPUs, pass `Accelerators=true` as part of `--feature-gates` flag.
    * Works only with Docker runtime.
* Clean up the kube-proxy container image by removing unnecessary packages and files. ([#42090](https://github.com/kubernetes/kubernetes/pull/42090), [@timstclair](https://github.com/timstclair))
* AWS: Support shared tag `kubernetes.io/cluster/<clusterid>` ([#41695](https://github.com/kubernetes/kubernetes/pull/41695), [@justinsb](https://github.com/justinsb))
* Insecure access to the API Server at localhost:8080 will be turned off in v1.6 when using kubeadm ([#42066](https://github.com/kubernetes/kubernetes/pull/42066), [@luxas](https://github.com/luxas))
* AWS: Do not consider master instance zones for dynamic volume creation ([#41702](https://github.com/kubernetes/kubernetes/pull/41702), [@justinsb](https://github.com/justinsb))
* Added foreground garbage collection: the owner object will not be deleted until all its dependents are deleted by the garbage collector. Please checkout the [user doc](https://kubernetes.io/docs/concepts/abstractions/controllers/garbage-collection/) for details. ([#38676](https://github.com/kubernetes/kubernetes/pull/38676), [@caesarxuchao](https://github.com/caesarxuchao))
    * deleteOptions.orphanDependents is going to be deprecated in 1.7. Please use deleteOptions.propagationPolicy instead.
* force unlock rbd image if the image is not used ([#41597](https://github.com/kubernetes/kubernetes/pull/41597), [@rootfs](https://github.com/rootfs))
* The kubernetes-master, kubernetes-worker and kubeapi-load-balancer charms have gained an nrpe-external-master relation, allowing the integration of their monitoring in an external Nagios server. ([#41923](https://github.com/kubernetes/kubernetes/pull/41923), [@Cynerva](https://github.com/Cynerva))
* make kubectl describe pod show tolerationSeconds ([#42162](https://github.com/kubernetes/kubernetes/pull/42162), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Completed pods should not be hidden when requested by name via `kubectl get`. ([#42216](https://github.com/kubernetes/kubernetes/pull/42216), [@smarterclayton](https://github.com/smarterclayton))
* [Federation][Kubefed] Flag cleanup ([#41335](https://github.com/kubernetes/kubernetes/pull/41335), [@irfanurrehman](https://github.com/irfanurrehman))
* Add the support to the scheduler for spreading pods of StatefulSets. ([#41708](https://github.com/kubernetes/kubernetes/pull/41708), [@bsalamat](https://github.com/bsalamat))
* Portworx Volume Plugin added enabling [Portworx](http://www.portworx.com) to be used as a storage provider for Kubernetes clusters. Portworx pools your servers capacity and turns your servers or cloud instances into converged, highly available compute and storage nodes. ([#39535](https://github.com/kubernetes/kubernetes/pull/39535), [@adityadani](https://github.com/adityadani))
* Remove support for trusty in GCE kube-up. ([#41670](https://github.com/kubernetes/kubernetes/pull/41670), [@mikedanese](https://github.com/mikedanese))
* Import a natural sorting library and use it in the sorting printer. ([#40746](https://github.com/kubernetes/kubernetes/pull/40746), [@matthyx](https://github.com/matthyx))
* Parameter keys in a StorageClass `parameters` map may not use the `kubernetes.io` or `k8s.io` namespaces. ([#41837](https://github.com/kubernetes/kubernetes/pull/41837), [@liggitt](https://github.com/liggitt))
* Make DaemonSet respect critical pods annotation when scheduling.  ([#42028](https://github.com/kubernetes/kubernetes/pull/42028), [@janetkuo](https://github.com/janetkuo))
* New Kubelet flag `--enforce-node-allocatable` with a default value of `pods` is added which will make kubelet create a top level cgroup for all pods to enforce Node Allocatable. Optionally, `system-reserved` & `kube-reserved` values can also be specified separated by comma to enforce node allocatable on cgroups specified via `--system-reserved-cgroup` & `--kube-reserved-cgroup` respectively. Note the default value of the latter flags are "". ([#41234](https://github.com/kubernetes/kubernetes/pull/41234), [@vishh](https://github.com/vishh))
    * This feature requires a **Node Drain** prior to upgrade failing which pods will be restarted if possible or terminated if they have a `RestartNever` policy.
* Deployment of AWS Kubernetes clusters using the in-tree bash deployment (i.e. cluster/kube-up.sh or get-kube.sh) is obsolete. v1.5.x will be the last release to support cluster/kube-up.sh with AWS. For a list of viable alternatives, see: http://kubernetes.io/docs/getting-started-guides/aws/  ([#42196](https://github.com/kubernetes/kubernetes/pull/42196), [@zmerlynn](https://github.com/zmerlynn))
* kubectl logs allows getting logs directly from deployment, job and statefulset ([#40927](https://github.com/kubernetes/kubernetes/pull/40927), [@soltysh](https://github.com/soltysh))
* make kubectl taint command respect effect NoExecute ([#42120](https://github.com/kubernetes/kubernetes/pull/42120), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Flex volume plugin is updated to support attach/detach interfaces. It broke backward compatibility. Please update your drivers and implement the new callouts.  ([#41804](https://github.com/kubernetes/kubernetes/pull/41804), [@chakri-nelluri](https://github.com/chakri-nelluri))
* Implement the update feature for DaemonSet. ([#41116](https://github.com/kubernetes/kubernetes/pull/41116), [@lukaszo](https://github.com/lukaszo))
* [Federation] Create configmap for the cluster kube-dns when cluster joins and remove when it unjoins ([#39338](https://github.com/kubernetes/kubernetes/pull/39338), [@irfanurrehman](https://github.com/irfanurrehman))
* New GKE certificates controller. ([#41160](https://github.com/kubernetes/kubernetes/pull/41160), [@pipejakob](https://github.com/pipejakob))
* Juju: Fix shebangs in charm actions to use python3 ([#42058](https://github.com/kubernetes/kubernetes/pull/42058), [@Cynerva](https://github.com/Cynerva))
* Support kubectl apply set-last-applied command to update the applied-applied-configuration annotation ([#41694](https://github.com/kubernetes/kubernetes/pull/41694), [@shiywang](https://github.com/shiywang))
* On GCI by default logrotate is disabled for application containers in favor of rotation mechanism provided by docker logging driver. ([#40634](https://github.com/kubernetes/kubernetes/pull/40634), [@Crassirostris](https://github.com/Crassirostris))
* Cleanup fluentd-gcp image: rebase on debian-base, switch to upstream packages, remove fluent-ui & rails ([#41998](https://github.com/kubernetes/kubernetes/pull/41998), [@timstclair](https://github.com/timstclair))
* Updating apiserver to return http status code 202 for a delete request when the resource is not immediately deleted because of user requesting cascading deletion using DeleteOptions.OrphanDependents=false. ([#41165](https://github.com/kubernetes/kubernetes/pull/41165), [@nikhiljindal](https://github.com/nikhiljindal))
* [Federation][kubefed] Support configuring dns-provider ([#40528](https://github.com/kubernetes/kubernetes/pull/40528), [@shashidharatd](https://github.com/shashidharatd))
* Added support to minimize sending verbose node information to scheduler extender by sending only node names and expecting extenders to cache the rest of the node information ([#41119](https://github.com/kubernetes/kubernetes/pull/41119), [@sarat-k](https://github.com/sarat-k))
* Guaranteed admission for Critical Pods ([#40952](https://github.com/kubernetes/kubernetes/pull/40952), [@dashpole](https://github.com/dashpole))
* Switch to the `node-role.kubernetes.io/master` label for marking and tainting the master node in kubeadm ([#41835](https://github.com/kubernetes/kubernetes/pull/41835), [@luxas](https://github.com/luxas))
* Allow drain --force to remove pods whose managing resource is deleted. ([#41864](https://github.com/kubernetes/kubernetes/pull/41864), [@marun](https://github.com/marun))
* add kubectl can-i to see if you can perform an action ([#41077](https://github.com/kubernetes/kubernetes/pull/41077), [@deads2k](https://github.com/deads2k))
* enable DefaultTolerationSeconds admission controller by default ([#41815](https://github.com/kubernetes/kubernetes/pull/41815), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Make DaemonSets survive taint-based evictions when nodes turn unreachable/notReady. ([#41896](https://github.com/kubernetes/kubernetes/pull/41896), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Add configurable limits to CronJob resource to specify how many successful and failed jobs are preserved. ([#40932](https://github.com/kubernetes/kubernetes/pull/40932), [@peay](https://github.com/peay))
* Deprecate outofdisk-transition-frequency and low-diskspace-threshold-mb flags ([#41941](https://github.com/kubernetes/kubernetes/pull/41941), [@dashpole](https://github.com/dashpole))
* Add OWNERS for sample-apiserver in staging ([#42094](https://github.com/kubernetes/kubernetes/pull/42094), [@sttts](https://github.com/sttts))
* Update gcr.io/google-containers/rescheduler to v0.2.2, which uses busybox as a base image instead of ubuntu. ([#41911](https://github.com/kubernetes/kubernetes/pull/41911), [@ixdy](https://github.com/ixdy))
* Add storage.k8s.io/v1 API ([#40088](https://github.com/kubernetes/kubernetes/pull/40088), [@jsafrane](https://github.com/jsafrane))
* Juju - K8s master charm now properly keeps distributed master files in sync for an HA control plane. ([#41351](https://github.com/kubernetes/kubernetes/pull/41351), [@chuckbutler](https://github.com/chuckbutler))
* Fix zsh completion: unknown file attribute error ([#38104](https://github.com/kubernetes/kubernetes/pull/38104), [@elipapa](https://github.com/elipapa))
* kubelet config should ignore file start with dots ([#39196](https://github.com/kubernetes/kubernetes/pull/39196), [@resouer](https://github.com/resouer))
* Add an alpha feature that makes NodeController set Taints instead of deleting Pods from not Ready Nodes. ([#41133](https://github.com/kubernetes/kubernetes/pull/41133), [@gmarek](https://github.com/gmarek))
* Base etcd-empty-dir-cleanup on busybox, run as nobody, and update to etcdctl 3.0.14 ([#41674](https://github.com/kubernetes/kubernetes/pull/41674), [@ixdy](https://github.com/ixdy))
* Fix zone placement heuristics so that multiple mounts in a StatefulSet pod are created in the same zone ([#40910](https://github.com/kubernetes/kubernetes/pull/40910), [@justinsb](https://github.com/justinsb))
* Flag --use-kubernetes-version for kubeadm init renamed to --kubernetes-version ([#41820](https://github.com/kubernetes/kubernetes/pull/41820), [@kad](https://github.com/kad))
* `kube-dns` now runs using a separate `system:serviceaccount:kube-system:kube-dns` service account which is automatically bound to the correct RBAC permissions. ([#38816](https://github.com/kubernetes/kubernetes/pull/38816), [@deads2k](https://github.com/deads2k))
* [Kubemark] Fixed hollow-npd container command to log to file ([#41858](https://github.com/kubernetes/kubernetes/pull/41858), [@shyamjvs](https://github.com/shyamjvs))
* kubeadm: Remove the --cloud-provider flag for beta init UX ([#41710](https://github.com/kubernetes/kubernetes/pull/41710), [@luxas](https://github.com/luxas))
* The CertificateSigningRequest API added the `extra` field to persist all information about the requesting user. This mirrors the fields in the SubjectAccessReview API used to check authorization. ([#41755](https://github.com/kubernetes/kubernetes/pull/41755), [@liggitt](https://github.com/liggitt))
* Upgrade golang versions to 1.7.5 ([#41771](https://github.com/kubernetes/kubernetes/pull/41771), [@cblecker](https://github.com/cblecker))
* Added a new secret type "bootstrap.kubernetes.io/token" for dynamically creating TLS bootstrapping bearer tokens. ([#41281](https://github.com/kubernetes/kubernetes/pull/41281), [@ericchiang](https://github.com/ericchiang))
* Remove unnecessary metrics (http/process/go) from being exposed by etcd-version-monitor ([#41807](https://github.com/kubernetes/kubernetes/pull/41807), [@shyamjvs](https://github.com/shyamjvs))
* Added `kubectl create clusterrole` command. ([#41538](https://github.com/kubernetes/kubernetes/pull/41538), [@xingzhou](https://github.com/xingzhou))
* Support new kubectl apply view-last-applied command for viewing the last configuration file applied ([#41146](https://github.com/kubernetes/kubernetes/pull/41146), [@shiywang](https://github.com/shiywang))
* Bump GCI to gci-stable-56-9000-84-2 ([#41819](https://github.com/kubernetes/kubernetes/pull/41819), [@dchen1107](https://github.com/dchen1107))
* list-resources: don't fail if the grep fails to match any resources ([#41933](https://github.com/kubernetes/kubernetes/pull/41933), [@ixdy](https://github.com/ixdy))
* client-go no longer imports GCP OAuth2 and OpenID Connect packages by default. ([#41532](https://github.com/kubernetes/kubernetes/pull/41532), [@ericchiang](https://github.com/ericchiang))
* Each pod has its own associated cgroup by default. ([#41349](https://github.com/kubernetes/kubernetes/pull/41349), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Whitelist kubemark in node_ssh_supported_providers for log dump ([#41800](https://github.com/kubernetes/kubernetes/pull/41800), [@shyamjvs](https://github.com/shyamjvs))
* Support KUBE_MAX_PD_VOLS on Azure ([#41398](https://github.com/kubernetes/kubernetes/pull/41398), [@codablock](https://github.com/codablock))
* Projected volume plugin ([#37237](https://github.com/kubernetes/kubernetes/pull/37237), [@jpeeler](https://github.com/jpeeler))
* `--output-version` is ignored for all commands except `kubectl convert`.  This is consistent with the generic nature of `kubectl` CRUD commands and the previous removal of `--api-version`.  Specific versions can be specified in the resource field: `resource.version.group`, `jobs.v1.batch`. ([#41576](https://github.com/kubernetes/kubernetes/pull/41576), [@deads2k](https://github.com/deads2k))
* Added bool type support for jsonpath. ([#39063](https://github.com/kubernetes/kubernetes/pull/39063), [@xingzhou](https://github.com/xingzhou))
* Nodes can now report two additional address types in their status: InternalDNS and ExternalDNS. The apiserver can use `--kubelet-preferred-address-types` to give priority to the type of address it uses to reach nodes. ([#34259](https://github.com/kubernetes/kubernetes/pull/34259), [@liggitt](https://github.com/liggitt))
* Clients now use the `?watch=true` parameter to make watch API calls, instead of the `/watch/` path prefix ([#41722](https://github.com/kubernetes/kubernetes/pull/41722), [@liggitt](https://github.com/liggitt))
* ResourceQuota ability to support default limited resources ([#36765](https://github.com/kubernetes/kubernetes/pull/36765), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Fix kubemark default e2e test suite's name ([#41751](https://github.com/kubernetes/kubernetes/pull/41751), [@shyamjvs](https://github.com/shyamjvs))
* federation aws: add logging of route53 calls ([#39964](https://github.com/kubernetes/kubernetes/pull/39964), [@justinsb](https://github.com/justinsb))
* Fix ConfigMap for Windows Containers. ([#39373](https://github.com/kubernetes/kubernetes/pull/39373), [@jbhurat](https://github.com/jbhurat))
* add defaultTolerationSeconds admission controller ([#41414](https://github.com/kubernetes/kubernetes/pull/41414), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Node Problem Detector is beta now. New features added: journald support, standalone mode and arbitrary system log monitoring. ([#40206](https://github.com/kubernetes/kubernetes/pull/40206), [@Random-Liu](https://github.com/Random-Liu))
* Fix the output of health-mointor.sh ([#41525](https://github.com/kubernetes/kubernetes/pull/41525), [@yujuhong](https://github.com/yujuhong))
* kubectl describe no longer prints the last-applied-configuration annotation for secrets. ([#34664](https://github.com/kubernetes/kubernetes/pull/34664), [@ymqytw](https://github.com/ymqytw))
* Report node not ready on failed PLEG health check ([#41569](https://github.com/kubernetes/kubernetes/pull/41569), [@yujuhong](https://github.com/yujuhong))
* Delay Deletion of a Pod until volumes are cleaned up ([#41456](https://github.com/kubernetes/kubernetes/pull/41456), [@dashpole](https://github.com/dashpole))
* Alpha version of dynamic volume provisioning is removed in this release. Annotation ([#40000](https://github.com/kubernetes/kubernetes/pull/40000), [@jsafrane](https://github.com/jsafrane))
    * "volume.alpha.kubernetes.io/storage-class" does not have any special meaning. A default storage class
    * and  DefaultStorageClass admission plugin can be used to preserve similar behavior of Kubernetes cluster,
    * see https://kubernetes.io/docs/user-guide/persistent-volumes/#class-1 for details.
* An `automountServiceAccountToken *bool` field was added to ServiceAccount and PodSpec objects. If set to `false` on a pod spec, no service account token is automounted in the pod. If set to `false` on a service account, no service account token is automounted for that service account unless explicitly overridden in the pod spec. ([#37953](https://github.com/kubernetes/kubernetes/pull/37953), [@liggitt](https://github.com/liggitt))
* Bump addon-manager version to v6.4-alpha.1 in kubemark ([#41506](https://github.com/kubernetes/kubernetes/pull/41506), [@shyamjvs](https://github.com/shyamjvs))
* Do not daemonize `salt-minion` for the openstack-heat provider. ([#40722](https://github.com/kubernetes/kubernetes/pull/40722), [@micmro](https://github.com/micmro))
* Move private key parsing from serviceaccount/jwt.go to client-go/util/cert ([#40907](https://github.com/kubernetes/kubernetes/pull/40907), [@cblecker](https://github.com/cblecker))
* Added configurable etcd initial-cluster-state to kube-up script. ([#41332](https://github.com/kubernetes/kubernetes/pull/41332), [@jszczepkowski](https://github.com/jszczepkowski))



# v1.6.0-alpha.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.6.0-alpha.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes.tar.gz) | `41b5e9edd973cbb8e68eb5d8d758c4f0afa11dfbd65df49e1c361206706a974c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-src.tar.gz) | `ec13e22322c85752918c23b0b498ba02087a1227b8fdc169f19acdf128f907c4`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `5a631b7604a69ef13c27b43e6df10f8bf14ff9170440fb07d0c46bc88a5a1eac`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `cfba71e38a924b783fcdbc0b1a342671d52af3588a8211e35048e9c071ed03b2`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `ceeee264b12959cb2b314efa9df4c165ea1598b8824ec652eb3994096f4ec07f`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `1bd3a4b64ab1535780f18b3e7a56dd1301a8ea8d66869ee704f66985c1fca9b4`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `d1615b3223c6e83422ed8409fc8d0a7a6069982d3413a482e12966b953520fe0`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `19133867e2d104db3e01212dbc4a702a315310a10e86076b6b80a16b94cf7954`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `0f89e17eb881c7db39195bc94874e3ec54866d2f57eef1540b5d843bedbe4326`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `3ed06cb89ffec011e4248c14d9e1c88c815b7363d1fdba217ed17e900f29960b`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `87927cbe26cefa296e2752075d018a58826bc7fa141c4cbe56116a254a3470cc`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `e97e7dafbf670140d3c4879a6738b970ac77d917861df3eea0c502238dd297b0`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `3aa82b838be450ce8dedbebfda45c453864c15aae6363ae5f1c0b0d285ffad2a`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `0bdeac3524ab7ef366f3bb75e2fbff3db156dcba2b862e8b2de393e4ec4377c9`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `1f37886aba4027ec682afe5f02a4d66a6645af2476f2954933c1b437ec66dafa`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `eb81d3cdd703790d5c96e24917183dc123aeabbe9a291c2dd86c68d21d9fd213`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `a50a57c689583f97fd4ff7af766bf7ae79c9fd97e46720bc41f385f2c51e1f99`

## Changelog since v1.6.0-alpha.2

### Other notable changes

* Fix AWS device allocator to only use valid device names ([#41455](https://github.com/kubernetes/kubernetes/pull/41455), [@gnufied](https://github.com/gnufied))
* [Federation][Kubefed] Bug fix relating kubeconfig path in kubefed init ([#41410](https://github.com/kubernetes/kubernetes/pull/41410), [@irfanurrehman](https://github.com/irfanurrehman))
* The apiserver audit log (`/var/log/kube-apiserver-audit.log`) will be sent through fluentd if enabled. ([#41360](https://github.com/kubernetes/kubernetes/pull/41360), [@enisoc](https://github.com/enisoc))
* Bump GCE ContainerVM to container-vm-v20170214 to address CVE-2016-9962. ([#41449](https://github.com/kubernetes/kubernetes/pull/41449), [@zmerlynn](https://github.com/zmerlynn))
* Fixed issues [#39202](https://github.com/kubernetes/kubernetes/pull/39202), [#41041](https://github.com/kubernetes/kubernetes/pull/41041) and [#40941](https://github.com/kubernetes/kubernetes/pull/40941) that caused the iSCSI connections to be prematurely closed when deleting a pod with an iSCSI persistent volume attached and that prevented the use of newly created LUNs on targets with preestablished connections. ([#41196](https://github.com/kubernetes/kubernetes/pull/41196), [@CristianPop](https://github.com/CristianPop))
* The kube-apiserver [basic audit log](https://kubernetes.io/docs/admin/audit/) can be enabled in GCE by exporting the environment variable `ENABLE_APISERVER_BASIC_AUDIT=true` before running `cluster/kube-up.sh`. This will log to `/var/log/kube-apiserver-audit.log` and use the same `logrotate` settings as `/var/log/kube-apiserver.log`. ([#41211](https://github.com/kubernetes/kubernetes/pull/41211), [@enisoc](https://github.com/enisoc))
* On kube-up.sh clusters on GCE, kube-scheduler now contacts the API on the secured port. ([#41285](https://github.com/kubernetes/kubernetes/pull/41285), [@liggitt](https://github.com/liggitt))
* Default RBAC ClusterRole and ClusterRoleBinding objects are automatically updated at server start to add missing permissions and subjects (extra permissions and subjects are left in place). To prevent autoupdating a particular role or rolebinding, annotate it with `rbac.authorization.kubernetes.io/autoupdate=false`. ([#41155](https://github.com/kubernetes/kubernetes/pull/41155), [@liggitt](https://github.com/liggitt))
* Make EnableCRI default to true ([#41378](https://github.com/kubernetes/kubernetes/pull/41378), [@yujuhong](https://github.com/yujuhong))
* `kubectl edit` now edits objects exactly as they were retrieved from the API. This allows using `kubectl edit` with third-party resources and extension API servers. Because client-side conversion is no longer done, the `--output-version` option is deprecated for `kubectl edit`. To edit using a particular API version, fully-qualify the resource, version, and group used to fetch the object (for example, `job.v1.batch/myjob`) ([#41304](https://github.com/kubernetes/kubernetes/pull/41304), [@liggitt](https://github.com/liggitt))
* We change the default attach_detach_controller sync period to 1 minute to reduce the query frequency through cloud provider to check whether volumes are attached or not.  ([#41363](https://github.com/kubernetes/kubernetes/pull/41363), [@jingxu97](https://github.com/jingxu97))
* RBAC `v1beta1` RoleBinding/ClusterRoleBinding subjects changed `apiVersion` to `apiGroup` to fully-qualify a subject. ServiceAccount subjects default to an apiGroup of `""`, User and Group subjects default to an apiGroup of `"rbac.authorization.k8s.io"`. ([#41184](https://github.com/kubernetes/kubernetes/pull/41184), [@liggitt](https://github.com/liggitt))
* Add support for finalizers in federated configmaps (deletes configmaps from underlying clusters). ([#40464](https://github.com/kubernetes/kubernetes/pull/40464), [@csbell](https://github.com/csbell))
* Make DaemonSet controller respect node taints and pod tolerations.  ([#41172](https://github.com/kubernetes/kubernetes/pull/41172), [@janetkuo](https://github.com/janetkuo))
* Added kubectl create role command ([#39852](https://github.com/kubernetes/kubernetes/pull/39852), [@xingzhou](https://github.com/xingzhou))
* If `experimentalCriticalPodAnnotation` feature gate is set to true, fluentd pods will not be evicted by the kubelet. ([#41035](https://github.com/kubernetes/kubernetes/pull/41035), [@vishh](https://github.com/vishh))



# v1.6.0-alpha.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.6.0-alpha.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes.tar.gz) | `d1a5c7bc435c0f58dca9eab54e8155b6e4797d4b5d6b0cb8feab968dd3132165`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-src.tar.gz) | `8d09b973f3debfe3d10b0ad392e56141446bc0d04aac60df6d761189997d97ed`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `b1a7d002768fff8282f8873fde6a0ed215d20fd72197d8e583c024b7cbbe3eb8`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `b0f872bbefdc1ecc9585dfdeb9c7e094f7a16ebbe4db11c64c70d1ef7f93e361`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `53be6adde2d13058d03d0f283ca166bb495cc49cd2e36339696dc46f85f78c8f`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `6436463d51ed54b50023cd725b054fd2b039e391095d8a618e91979fc55d4ee0`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `d6638c8950a9e03ed64c3f3e1ad82166642c02aeb8f7fb2079db1f137170a145`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `05d0e466a27fc9a5b6535cbd7683e08739cd37aa2c6213c000fdaa305e4eb888`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `38971be682cbf1f194eeb9ad683272a58042f4f91992db6fc34720de28d88dd6`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `774a002482d6b62338550b20d90b914a08481965ed1b78cd348535d90f88f344`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `690ab995ac27c90c811578677fb8688d43198499bfc451a2f908ad7a76474ee8`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `757fe9e2083e2da706b52c96da34548aa72bbbbff50bb261c1198c80b36189c3`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `5aa3161a1030ddb02985171d4343a99d14ad6e09e130549b20fc96797588ff1e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `babce8c402c8e88310ba82315f758e981d4cfe20dcbf3047e55b279d5d4f3a33`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `4b1d4c4ba95d86481d32daf09e769f7f9e4e0d71e11d39a5a4e205804b023172`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `070423e62a0dff7ef17e29a63ac2eda5a46b6e1badf67214c07970b83610bf6c`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `3a826fef775819a4f03e7db91684db6edfdaaad3e3eb366c4321bdc5ec0b0f25`

## Changelog since v1.6.0-alpha.1

### Other notable changes

* Align the hyperkube image to support running binaries at /usr/local/bin/ like the other server images ([#41017](https://github.com/kubernetes/kubernetes/pull/41017), [@luxas](https://github.com/luxas))
* Native support for token based bootstrap flow.  This includes signing a well known ConfigMap in the `kube-public` namespace and cleaning out expired tokens. ([#36101](https://github.com/kubernetes/kubernetes/pull/36101), [@jbeda](https://github.com/jbeda))
* Reverts to looking up the current VM in vSphere using the machine's UUID, either obtained via sysfs or via the `vm-uuid` parameter in the cloud configuration file. ([#40892](https://github.com/kubernetes/kubernetes/pull/40892), [@robdaemon](https://github.com/robdaemon))
* This PR adds a manager to NodeController that is responsible for removing Pods from Nodes tainted with NoExecute Taints. This feature is beta (as the rest of taints) and enabled by default. It's gated by controller-manager enable-taint-manager flag. ([#40355](https://github.com/kubernetes/kubernetes/pull/40355), [@gmarek](https://github.com/gmarek))
* The authentication.k8s.io API group was promoted to v1 ([#41058](https://github.com/kubernetes/kubernetes/pull/41058), [@liggitt](https://github.com/liggitt))
* Fixes issue [#38418](https://github.com/kubernetes/kubernetes/pull/38418) which, under circumstance, could cause StatefulSet to deadlock.  ([#40838](https://github.com/kubernetes/kubernetes/pull/40838), [@kow3ns](https://github.com/kow3ns))
    * Mediates issue [#36859](https://github.com/kubernetes/kubernetes/pull/36859). StatefulSet only acts on Pods whose identity matches the StatefulSet, providing a partial mediation for overlapping controllers.
* Introduces an new alpha version of the Horizontal Pod Autoscaler including expanded support for specifying metrics. ([#36033](https://github.com/kubernetes/kubernetes/pull/36033), [@DirectXMan12](https://github.com/DirectXMan12))
* Set all node conditions to Unknown when node is unreachable ([#36592](https://github.com/kubernetes/kubernetes/pull/36592), [@andrewsykim](https://github.com/andrewsykim))
* [Federation] Add override flags options to kubefed init ([#40917](https://github.com/kubernetes/kubernetes/pull/40917), [@irfanurrehman](https://github.com/irfanurrehman))
* Fix for detach volume when node is not present/ powered off ([#40118](https://github.com/kubernetes/kubernetes/pull/40118), [@BaluDontu](https://github.com/BaluDontu))
* Bump up GLBC version from 0.9.0-beta to 0.9.1 ([#41037](https://github.com/kubernetes/kubernetes/pull/41037), [@bprashanth](https://github.com/bprashanth))
* The deprecated flags --config, --auth-path, --resource-container, and --system-container were removed. ([#40048](https://github.com/kubernetes/kubernetes/pull/40048), [@mtaufen](https://github.com/mtaufen))
* Add `kubectl attach` support for multiple types ([#40365](https://github.com/kubernetes/kubernetes/pull/40365), [@shiywang](https://github.com/shiywang))
* [Kubelet] Delay deletion of pod from the API server until volumes are deleted ([#41095](https://github.com/kubernetes/kubernetes/pull/41095), [@dashpole](https://github.com/dashpole))
* remove the create-external-load-balancer flag in cmd/expose.go ([#38183](https://github.com/kubernetes/kubernetes/pull/38183), [@tianshapjq](https://github.com/tianshapjq))
* The authorization.k8s.io API group was promoted to v1 ([#40709](https://github.com/kubernetes/kubernetes/pull/40709), [@liggitt](https://github.com/liggitt))
* Bump GCI to gci-beta-56-9000-80-0 ([#41027](https://github.com/kubernetes/kubernetes/pull/41027), [@dchen1107](https://github.com/dchen1107))
* [Federation][kubefed] Add option to expose federation apiserver on nodeport service ([#40516](https://github.com/kubernetes/kubernetes/pull/40516), [@shashidharatd](https://github.com/shashidharatd))
* Rename --experiemental-cgroups-per-qos to --cgroups-per-qos ([#39972](https://github.com/kubernetes/kubernetes/pull/39972), [@derekwaynecarr](https://github.com/derekwaynecarr))
* PV E2E: provide each spec with a fresh nfs host ([#40879](https://github.com/kubernetes/kubernetes/pull/40879), [@copejon](https://github.com/copejon))
* Remove the temporary fix for pre-1.0 mirror pods ([#40877](https://github.com/kubernetes/kubernetes/pull/40877), [@yujuhong](https://github.com/yujuhong))
* fix --save-config in create subcommand ([#40289](https://github.com/kubernetes/kubernetes/pull/40289), [@xilabao](https://github.com/xilabao))
* We should mention the caveats of in-place upgrade in release note. ([#40727](https://github.com/kubernetes/kubernetes/pull/40727), [@Random-Liu](https://github.com/Random-Liu))
* The SubjectAccessReview API passes subresource and resource name information to the authorizer to answer authorization queries. ([#40935](https://github.com/kubernetes/kubernetes/pull/40935), [@liggitt](https://github.com/liggitt))
* When feature gate "ExperimentalCriticalPodAnnotation" is set, Kubelet will avoid evicting pods in "kube-system" namespace that contains a special annotation - `scheduler.alpha.kubernetes.io/critical-pod` ([#40655](https://github.com/kubernetes/kubernetes/pull/40655), [@vishh](https://github.com/vishh))
    * This feature should be used in conjunction with the rescheduler to guarantee availability for critical system pods - https://kubernetes.io/docs/admin/rescheduler/
* make tolerations respect wildcard key ([#39914](https://github.com/kubernetes/kubernetes/pull/39914), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* [Federation][kubefed] Add option to disable persistence storage for etcd ([#40862](https://github.com/kubernetes/kubernetes/pull/40862), [@shashidharatd](https://github.com/shashidharatd))
* Init containers have graduated to GA and now appear as a field.  The beta annotation value will still be respected and overrides the field value. ([#38382](https://github.com/kubernetes/kubernetes/pull/38382), [@hodovska](https://github.com/hodovska))
* apply falls back to generic 3-way JSON merge patch if no go struct is registered for the target GVK ([#40666](https://github.com/kubernetes/kubernetes/pull/40666), [@ymqytw](https://github.com/ymqytw))
* HorizontalPodAutoscaler is no longer supported in extensions/v1beta1 version. Use autoscaling/v1 instead. ([#35782](https://github.com/kubernetes/kubernetes/pull/35782), [@piosz](https://github.com/piosz))
* Port forwarding can forward over websockets or SPDY. ([#33684](https://github.com/kubernetes/kubernetes/pull/33684), [@fraenkel](https://github.com/fraenkel))
* Improve kubectl describe node output by adding closing paren ([#39217](https://github.com/kubernetes/kubernetes/pull/39217), [@luksa](https://github.com/luksa))
* Bump GCE ContainerVM to container-vm-v20170201 to address CVE-2016-9962. ([#40828](https://github.com/kubernetes/kubernetes/pull/40828), [@zmerlynn](https://github.com/zmerlynn))
* Use full package path for definition name in OpenAPI spec ([#40124](https://github.com/kubernetes/kubernetes/pull/40124), [@mbohlool](https://github.com/mbohlool))
* kubectl apply now supports explicitly clearing values not present in the config by setting them to null ([#40630](https://github.com/kubernetes/kubernetes/pull/40630), [@liggitt](https://github.com/liggitt))
* Fixed an issue where 'kubectl get --sort-by=' would return an error when the specified field were not present in at least one of the returned objects, even that being a valid field in the object model. ([#40541](https://github.com/kubernetes/kubernetes/pull/40541), [@fabianofranz](https://github.com/fabianofranz))
* Add initial french translations for kubectl ([#40645](https://github.com/kubernetes/kubernetes/pull/40645), [@brendandburns](https://github.com/brendandburns))
* OpenStack-Heat will now look for an image named "CentOS-7-x86_64-GenericCloud-1604". To restore the previous behavior set OPENSTACK_IMAGE_NAME="CentOS7" ([#40368](https://github.com/kubernetes/kubernetes/pull/40368), [@sc68cal](https://github.com/sc68cal))
* Preventing nil pointer reference in client_config ([#40508](https://github.com/kubernetes/kubernetes/pull/40508), [@vjsamuel](https://github.com/vjsamuel))
* The bash AWS deployment via kube-up.sh has been deprecated. See http://kubernetes.io/docs/getting-started-guides/aws/ for alternatives. ([#38772](https://github.com/kubernetes/kubernetes/pull/38772), [@zmerlynn](https://github.com/zmerlynn))
* Fix failing load balancers in Azure ([#40405](https://github.com/kubernetes/kubernetes/pull/40405), [@codablock](https://github.com/codablock))
* kubefed init creates a service account for federation controller manager in the federation-system namespace and binds that service account to the federation-system:federation-controller-manager role that has read and list access on secrets in the federation-system namespace.  ([#40392](https://github.com/kubernetes/kubernetes/pull/40392), [@madhusudancs](https://github.com/madhusudancs))
* Fixed an SELinux issue in kubeadm on Docker 1.12+ by moving etcd SELinux options from container to pod. ([#40682](https://github.com/kubernetes/kubernetes/pull/40682), [@dgoodwin](https://github.com/dgoodwin))
* Juju kubernetes-master charm: improve status messages ([#40691](https://github.com/kubernetes/kubernetes/pull/40691), [@Cynerva](https://github.com/Cynerva))



# v1.6.0-alpha.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.6.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes.tar.gz) | `abda73bc2a27ae16c66a5aea9e96cd59486ed8cf994afc55da35a3cea2edc1db`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-src.tar.gz) | `b429579ba83f9a3fa80e72ceb65b046659b17f16a0b7f70105e7096a441f32b9`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `d5b8adee6169515324f4f420e65e9d4e14cca3402f58ec99660a1947fd79d3ea`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `6232a7e46b2cbd1e30a1f44c9b424448c5def11f5132c742bf62bac8a4f26fa2`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `2974ed14b76885947c95b3c86fb8585aa08ccefe8cc11cd202dcc3cb8fcc0d1a`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `3bcb40f4aa3a295ec23fe42a5e17b081ef8de174b7dfa03cb89c27a00ac16f5a`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `f9a55bcb6af2a415d24d69ae919c56968f6b02369675fd7def63acbde6534430`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `b0bd7070eab2f19b9bc1840fb4da5307c7ce274f2e28f76f94c714aa08f087bf`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `39f38972f93f64542ae325d9c937c9d527968bc0830fdd38dd38dc246b6f0c56`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `032e21fe0000333f36e29ddf24e207cfd6c92cb9a6d69cc0123c31aacd12338c`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `84b7d58012760111067ec0907070cc2f6d4c95893e771533a9b335cc8d8c72b7`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `e9ce76f48a4cf58b261aec38f076ed3b61c444c55c123f30099c1d1763d6191f`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `fafa4b93a522b9e73b6c31e42d32dc5eb3fccb5ca1548425da03726b48a19175`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `f00f4874785d1c9cba4fd262e8086aa693027649da44b0eec69e96951e013913`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `e843a6357e0a2e441277e6b32d8d64a6b6fe367c3d223edec781c21d8b379ac6`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `12b73f7cc4928543eee09af91d632bcf5c4ed56c44d48d62d0087c7fcbaa3e02`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.6.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `aad15f260d166c2b295921468837efe4a51cb7a04bcaccb3faecc9b5979861e5`

## Changelog since v1.5.0

### Action Required

* Promote certificates.k8s.io to beta and enable it by default. Users using the alpha certificates API should delete v1alpha1 CSRs from the API before upgrading and recreate them as v1beta1 CSR after upgrading. ([#39772](https://github.com/kubernetes/kubernetes/pull/39772), [@mikedanese](https://github.com/mikedanese))
* Switch default etcd version to 3.0.14. ([#36229](https://github.com/kubernetes/kubernetes/pull/36229), [@wojtek-t](https://github.com/wojtek-t))
    * Switch default storage backend flag in apiserver to `etcd3` mode.
* RBAC's special handling of the user `*` in RoleBinding and ClusterRoleBinding objects is deprecated and will be removed in v1beta1. To match all users, explicitly bind to the group `system:authenticated` and/or `system:unauthenticated`. Existing v1alpha1 bindings to the user `*` will be automatically converted to the group `system:authenticated`. ([#38981](https://github.com/kubernetes/kubernetes/pull/38981), [@liggitt](https://github.com/liggitt))
* The 'endpoints.beta.kubernetes.io/hostnames-map' annotation is no longer supported.  Users can use the 'Endpoints.subsets[].addresses[].hostname' field instead. ([#39284](https://github.com/kubernetes/kubernetes/pull/39284), [@bowei](https://github.com/bowei))
* `federation/deploy/deploy.sh` was an interim solution introduced in Kubernetes v1.4 to simplify the federation control plane deployment experience. Now that we have `kubefed`, we are deprecating `deploy.sh` scripts. ([#38902](https://github.com/kubernetes/kubernetes/pull/38902), [@madhusudancs](https://github.com/madhusudancs))
* Cluster federation servers have changed the location in etcd where federated services are stored, so existing federated services must be deleted and recreated. Before upgrading, export all federated services from the federation server and delete the services. After upgrading the cluster, recreate the federated services from the exported data. ([#37770](https://github.com/kubernetes/kubernetes/pull/37770), [@enj](https://github.com/enj))
* etcd2: watching from 0 returns all initial states as ADDED events ([#38079](https://github.com/kubernetes/kubernetes/pull/38079), [@hongchaodeng](https://github.com/hongchaodeng))

### Other notable changes

* kube-up.sh on GCE now includes the bootstrap admin in the super-user group, and ensures the auth token file is correct on upgrades ([#39537](https://github.com/kubernetes/kubernetes/pull/39537), [@liggitt](https://github.com/liggitt))
* genericapiserver: cut off more dependencies – episode 3 ([#40426](https://github.com/kubernetes/kubernetes/pull/40426), [@sttts](https://github.com/sttts))
* Adding vmdk file extension for vmDiskPath in vsphere DeleteVolume ([#40538](https://github.com/kubernetes/kubernetes/pull/40538), [@divyenpatel](https://github.com/divyenpatel))
* Remove outdated net.experimental.kubernetes.io/proxy-mode and net.beta.kubernetes.io/proxy-mode annotations from kube-proxy. ([#40585](https://github.com/kubernetes/kubernetes/pull/40585), [@cblecker](https://github.com/cblecker))
* Improve the ARM builds and make hyperkube on ARM working again by upgrading the Go version for ARM to go1.8beta2 ([#38926](https://github.com/kubernetes/kubernetes/pull/38926), [@luxas](https://github.com/luxas))
* Prevent hotloops on error conditions, which could fill up the disk faster than log rotation can free space. ([#40497](https://github.com/kubernetes/kubernetes/pull/40497), [@lavalamp](https://github.com/lavalamp))
* DaemonSet controller actively kills failed pods (to recreate them) ([#40330](https://github.com/kubernetes/kubernetes/pull/40330), [@janetkuo](https://github.com/janetkuo))
* forgiveness alpha version api definition ([#39469](https://github.com/kubernetes/kubernetes/pull/39469), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Bump up glbc version to 0.9.0-beta.1 ([#40565](https://github.com/kubernetes/kubernetes/pull/40565), [@bprashanth](https://github.com/bprashanth))
* Improve formatting of EventSource in kubectl get and kubectl describe ([#40073](https://github.com/kubernetes/kubernetes/pull/40073), [@matthyx](https://github.com/matthyx))
* CRI: use more gogoprotobuf plugins ([#40397](https://github.com/kubernetes/kubernetes/pull/40397), [@yujuhong](https://github.com/yujuhong))
* Adds `shortNames` to the `APIResource` from discovery which is a list of recommended shortNames for clients like `kubectl`. ([#40312](https://github.com/kubernetes/kubernetes/pull/40312), [@p0lyn0mial](https://github.com/p0lyn0mial))
* Use existing ABAC policy file when upgrading GCE cluster ([#40172](https://github.com/kubernetes/kubernetes/pull/40172), [@liggitt](https://github.com/liggitt))
* Added support for creating HA clusters for centos using kube-up.sh. ([#39462](https://github.com/kubernetes/kubernetes/pull/39462), [@Shawyeok](https://github.com/Shawyeok))
* azure: fix Azure Container Registry integration ([#40142](https://github.com/kubernetes/kubernetes/pull/40142), [@colemickens](https://github.com/colemickens))
* - Splits Juju Charm layers into master/worker roles ([#40324](https://github.com/kubernetes/kubernetes/pull/40324), [@chuckbutler](https://github.com/chuckbutler))
    * - Adds support for 1.5.x series of Kubernetes
    * - Introduces a tactic for keeping templates in sync with upstream eliminating template drift
    * - Adds CNI support to the Juju Charms
    * - Adds durable storage support to the Juju Charms
    * - Introduces an e2e Charm layer for repeatable testing efforts and validation of clusters
* genericapiserver: more dependency cutoffs ([#40216](https://github.com/kubernetes/kubernetes/pull/40216), [@sttts](https://github.com/sttts))
* AWS: trust region if found from AWS metadata ([#38880](https://github.com/kubernetes/kubernetes/pull/38880), [@justinsb](https://github.com/justinsb))
* Volumes and environment variables populated from ConfigMap and Secret objects can now tolerate the named source object or specific keys being missing, by adding `optional: true` to the volume or environment variable source specifications. ([#39981](https://github.com/kubernetes/kubernetes/pull/39981), [@fraenkel](https://github.com/fraenkel))
* kubectl create now accepts the label selector flag for filtering objects to create ([#40057](https://github.com/kubernetes/kubernetes/pull/40057), [@MrHohn](https://github.com/MrHohn))
* A new field `terminationMessagePolicy` has been added to containers that allows a user to request `FallbackToLogsOnError`, which will read from the container's logs to populate the termination message if the user does not write to the termination message log file.  The termination message file is now properly readable for end users and has a maximum size (4k bytes) to prevent abuse.  Each pod may have up to 12k bytes of termination messages before the contents of each will be truncated. ([#39341](https://github.com/kubernetes/kubernetes/pull/39341), [@smarterclayton](https://github.com/smarterclayton))
* Add a special purpose tool for editing individual fields in a ConfigMap with kubectl ([#38445](https://github.com/kubernetes/kubernetes/pull/38445), [@brendandburns](https://github.com/brendandburns))
* [Federation] Expose autoscaling apis through federation api server ([#38976](https://github.com/kubernetes/kubernetes/pull/38976), [@irfanurrehman](https://github.com/irfanurrehman))
* Powershell script to start kubelet and kube-proxy ([#36250](https://github.com/kubernetes/kubernetes/pull/36250), [@jbhurat](https://github.com/jbhurat))
* Reduce time needed to attach Azure disks ([#40066](https://github.com/kubernetes/kubernetes/pull/40066), [@codablock](https://github.com/codablock))
* fixing Cassandra shutdown example to avoid data corruption ([#39199](https://github.com/kubernetes/kubernetes/pull/39199), [@deimosfr](https://github.com/deimosfr))
* kubeadm: add optional self-hosted deployment for apiserver, controller-manager and scheduler. ([#40075](https://github.com/kubernetes/kubernetes/pull/40075), [@pires](https://github.com/pires))
* kubelet tears down pod volumes on pod termination rather than pod deletion ([#37228](https://github.com/kubernetes/kubernetes/pull/37228), [@sjenning](https://github.com/sjenning))
* The default client certificate generated by kube-up now contains the superuser `system:masters` group ([#39966](https://github.com/kubernetes/kubernetes/pull/39966), [@liggitt](https://github.com/liggitt))
* CRI: upgrade protobuf to v3 ([#39158](https://github.com/kubernetes/kubernetes/pull/39158), [@feiskyer](https://github.com/feiskyer))
* Add SIGCHLD handler to pause container ([#36853](https://github.com/kubernetes/kubernetes/pull/36853), [@verb](https://github.com/verb))
* Populate environment variables from a secrets. ([#39446](https://github.com/kubernetes/kubernetes/pull/39446), [@fraenkel](https://github.com/fraenkel))
* fluentd config for GKE clusters updated: detect exceptions in container log streams and forward them as one log entry. ([#39656](https://github.com/kubernetes/kubernetes/pull/39656), [@thomasschickinger](https://github.com/thomasschickinger))
* Made multi-scheduler graduated to Beta and then v1. ([#38871](https://github.com/kubernetes/kubernetes/pull/38871), [@k82cn](https://github.com/k82cn))
* Fixed forming resolver search line for pods: exclude duplicates, obey libc limitations, logging and eventing appropriately. ([#29666](https://github.com/kubernetes/kubernetes/pull/29666), [@vefimova](https://github.com/vefimova))
* Add authorization mode to kubeadm ([#39846](https://github.com/kubernetes/kubernetes/pull/39846), [@andrewrynhard](https://github.com/andrewrynhard))
* Update dependencies: aws-sdk-go to 1.6.10; also cadvisor ([#40095](https://github.com/kubernetes/kubernetes/pull/40095), [@dashpole](https://github.com/dashpole))
* Fixes a bug in the OpenStack-Heat kubernetes provider, in the handling of differences between the Identity v2 and Identity v3 APIs ([#40105](https://github.com/kubernetes/kubernetes/pull/40105), [@sc68cal](https://github.com/sc68cal))
* Update GCE ContainerVM deployment to container-vm-v20170117 to pick up CVE fixes in base image. ([#40094](https://github.com/kubernetes/kubernetes/pull/40094), [@zmerlynn](https://github.com/zmerlynn))
* AWS: Remove duplicate calls to DescribeInstance during volume operations ([#39842](https://github.com/kubernetes/kubernetes/pull/39842), [@gnufied](https://github.com/gnufied))
* The `attributeRestrictions` field has been removed from the PolicyRule type in the rbac.authorization.k8s.io/v1alpha1 API. The field was not used by the RBAC authorizer. ([#39625](https://github.com/kubernetes/kubernetes/pull/39625), [@deads2k](https://github.com/deads2k))
* Enable lazy inode table and journal initialization for ext3 and ext4 ([#38865](https://github.com/kubernetes/kubernetes/pull/38865), [@codablock](https://github.com/codablock))
* azure disk: restrict name length for Azure specifications ([#40030](https://github.com/kubernetes/kubernetes/pull/40030), [@colemickens](https://github.com/colemickens))
* Follow redirects for streaming requests (exec/attach/port-forward) in the apiserver by default (alpha -> beta). ([#40039](https://github.com/kubernetes/kubernetes/pull/40039), [@timstclair](https://github.com/timstclair))
* Use kube-dns:1.11.0 ([#39925](https://github.com/kubernetes/kubernetes/pull/39925), [@sadlil](https://github.com/sadlil))
* Anonymous authentication is now automatically disabled if the API server is started with the AlwaysAllow authorizer. ([#38706](https://github.com/kubernetes/kubernetes/pull/38706), [@deads2k](https://github.com/deads2k))
* genericapiserver: cut off kube pkg/version dependency ([#39943](https://github.com/kubernetes/kubernetes/pull/39943), [@sttts](https://github.com/sttts))
* genericapiserver: cut off pkg/serviceaccount dependency ([#39945](https://github.com/kubernetes/kubernetes/pull/39945), [@sttts](https://github.com/sttts))
* Move pkg/api/rest into genericapiserver ([#39948](https://github.com/kubernetes/kubernetes/pull/39948), [@sttts](https://github.com/sttts))
* genericapiserver: cut off pkg/apis/extensions and pkg/storage dependencies ([#39946](https://github.com/kubernetes/kubernetes/pull/39946), [@sttts](https://github.com/sttts))
* genericapiserver: cut off certificates api dependency ([#39947](https://github.com/kubernetes/kubernetes/pull/39947), [@sttts](https://github.com/sttts))
* Admission control support for versioned configuration files ([#39109](https://github.com/kubernetes/kubernetes/pull/39109), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Fix issue around merging lists of primitives when using PATCH or kubectl apply. ([#38665](https://github.com/kubernetes/kubernetes/pull/38665), [@ymqytw](https://github.com/ymqytw))
* Fixes API compatibility issue with empty lists incorrectly returning a null `items` field instead of an empty array. ([#39834](https://github.com/kubernetes/kubernetes/pull/39834), [@liggitt](https://github.com/liggitt))
* [scheduling] Moved pod affinity and anti-affinity from annotations to api fields [#25319](https://github.com/kubernetes/kubernetes/pull/25319) ([#39478](https://github.com/kubernetes/kubernetes/pull/39478), [@rrati](https://github.com/rrati))
* PodSecurityPolicy resource is now enabled by default in the extensions API group. ([#39743](https://github.com/kubernetes/kubernetes/pull/39743), [@pweil-](https://github.com/pweil-))
* add --controllers to controller manager ([#39740](https://github.com/kubernetes/kubernetes/pull/39740), [@deads2k](https://github.com/deads2k))
* proxy/iptables: don't sync proxy rules if services map didn't change ([#38996](https://github.com/kubernetes/kubernetes/pull/38996), [@dcbw](https://github.com/dcbw))
* Update amd64 kube-proxy base image to debian-iptables-amd64:v5 ([#39725](https://github.com/kubernetes/kubernetes/pull/39725), [@ixdy](https://github.com/ixdy))
* Update dashboard version to v1.5.1 ([#39662](https://github.com/kubernetes/kubernetes/pull/39662), [@rf232](https://github.com/rf232))
* Fix kubectl get -f <file> -o <nondefault printer> so it prints all items in the file ([#39038](https://github.com/kubernetes/kubernetes/pull/39038), [@ncdc](https://github.com/ncdc))
* Scheduler treats StatefulSet pods as belonging to a single equivalence class. ([#39718](https://github.com/kubernetes/kubernetes/pull/39718), [@foxish](https://github.com/foxish))
* --basic-auth-file supports optionally specifying groups in the fourth column of the file ([#39651](https://github.com/kubernetes/kubernetes/pull/39651), [@liggitt](https://github.com/liggitt))
* To create or update an RBAC RoleBinding or ClusterRoleBinding object, a user must: ([#39383](https://github.com/kubernetes/kubernetes/pull/39383), [@liggitt](https://github.com/liggitt))
    * Be authorized to make the create or update API request
    * Be allowed to bind the referenced role, either by already having all of the permissions contained in the referenced role, or by having the `bind` permission on the referenced role.
* Fixes an HPA-related panic due to division-by-zero. ([#39694](https://github.com/kubernetes/kubernetes/pull/39694), [@DirectXMan12](https://github.com/DirectXMan12))
* federation: Adding support for DeleteOptions.OrphanDependents for federated services. Setting it to false while deleting a federated service also deletes the corresponding services from all registered clusters. ([#36390](https://github.com/kubernetes/kubernetes/pull/36390), [@nikhiljindal](https://github.com/nikhiljindal))
* Update kube-proxy image to be based off of Debian 8.6 base image. ([#39695](https://github.com/kubernetes/kubernetes/pull/39695), [@ixdy](https://github.com/ixdy))
* Update FitError as a message component into the PodConditionUpdater. ([#39491](https://github.com/kubernetes/kubernetes/pull/39491), [@jayunit100](https://github.com/jayunit100))
* rename kubernetes-discovery to kube-aggregator ([#39619](https://github.com/kubernetes/kubernetes/pull/39619), [@deads2k](https://github.com/deads2k))
* Allow missing keys in templates by default ([#39486](https://github.com/kubernetes/kubernetes/pull/39486), [@ncdc](https://github.com/ncdc))
* Caching added to the OIDC client auth plugin to fix races and reduce the time kubectl commands using this plugin take by several seconds. ([#38167](https://github.com/kubernetes/kubernetes/pull/38167), [@ericchiang](https://github.com/ericchiang))
* AWS: recognize eu-west-2 region ([#38746](https://github.com/kubernetes/kubernetes/pull/38746), [@justinsb](https://github.com/justinsb))
* Provide kubernetes-controller-manager flags to control volume attach/detach reconciler sync.  The duration of the syncs can be controlled, and the syncs can be shut off as well.  ([#39551](https://github.com/kubernetes/kubernetes/pull/39551), [@chrislovecnm](https://github.com/chrislovecnm))
* Generate OpenAPI definition for inlined types ([#39466](https://github.com/kubernetes/kubernetes/pull/39466), [@mbohlool](https://github.com/mbohlool))
* ShortcutExpander has been extended in a way that it will examine a ha… ([#38835](https://github.com/kubernetes/kubernetes/pull/38835), [@p0lyn0mial](https://github.com/p0lyn0mial))
* fixes nil dereference when doing a volume type check on persistent volumes ([#39493](https://github.com/kubernetes/kubernetes/pull/39493), [@sjenning](https://github.com/sjenning))
* Fix issue with PodDisruptionBudgets in which `minAvailable` specified as a percentage did not work with StatefulSet Pods. ([#39454](https://github.com/kubernetes/kubernetes/pull/39454), [@foxish](https://github.com/foxish))
* fix issue with kubectl proxy so that it will proxy an empty path - e.g. http://localhost:8001 ([#39226](https://github.com/kubernetes/kubernetes/pull/39226), [@luksa](https://github.com/luksa))
* Check if pathExists before performing Unmount ([#39311](https://github.com/kubernetes/kubernetes/pull/39311), [@rkouj](https://github.com/rkouj))
* Adding kubectl tests for federation ([#38844](https://github.com/kubernetes/kubernetes/pull/38844), [@nikhiljindal](https://github.com/nikhiljindal))
* Fix comment and optimize code ([#38084](https://github.com/kubernetes/kubernetes/pull/38084), [@tanshanshan](https://github.com/tanshanshan))
* When using OIDC authentication and specifying --oidc-username-claim=email, an `"email_verified":true` claim must be returned from the identity provider. ([#36087](https://github.com/kubernetes/kubernetes/pull/36087), [@ericchiang](https://github.com/ericchiang))
* Allow pods to define multiple environment variables from a whole ConfigMap ([#36245](https://github.com/kubernetes/kubernetes/pull/36245), [@fraenkel](https://github.com/fraenkel))
* Refactor the certificate and kubeconfig code in the kubeadm binary into two phases ([#39280](https://github.com/kubernetes/kubernetes/pull/39280), [@luxas](https://github.com/luxas))
* Added support for printing in all supported `--output` formats to `kubectl create ...` and `kubectl apply ...` ([#38112](https://github.com/kubernetes/kubernetes/pull/38112), [@juanvallejo](https://github.com/juanvallejo))
* genericapiserver: extract CA cert from server cert and SNI cert chains ([#39022](https://github.com/kubernetes/kubernetes/pull/39022), [@sttts](https://github.com/sttts))
* Endpoints, that tolerate unready Pods, are now listing Pods in state Terminating as well ([#37093](https://github.com/kubernetes/kubernetes/pull/37093), [@simonswine](https://github.com/simonswine))
* DaemonSet ObservedGeneration ([#39157](https://github.com/kubernetes/kubernetes/pull/39157), [@lukaszo](https://github.com/lukaszo))
* The `--reconcile-cidr` kubelet flag was removed since it had been deprecated since v1.5 ([#39322](https://github.com/kubernetes/kubernetes/pull/39322), [@luxas](https://github.com/luxas))
* Add ready replicas in Deployments ([#37959](https://github.com/kubernetes/kubernetes/pull/37959), [@kargakis](https://github.com/kargakis))
* Remove all MAINTAINER statements in Dockerfiles in the codebase as they are deprecated by docker ([#38927](https://github.com/kubernetes/kubernetes/pull/38927), [@luxas](https://github.com/luxas))
* Remove the deprecated vsphere kube-up. ([#39140](https://github.com/kubernetes/kubernetes/pull/39140), [@kerneltime](https://github.com/kerneltime))
* Kubectl top now also accepts short forms for "node" and "pod" ("no", "po") ([#39218](https://github.com/kubernetes/kubernetes/pull/39218), [@luksa](https://github.com/luksa))
* Remove 'exec' network plugin - use CNI instead ([#39254](https://github.com/kubernetes/kubernetes/pull/39254), [@freehan](https://github.com/freehan))
* Add three more columns to `kubectl get deploy -o wide` output. ([#39240](https://github.com/kubernetes/kubernetes/pull/39240), [@xingzhou](https://github.com/xingzhou))
* Add path exist check in getPodVolumePathListFromDisk ([#38909](https://github.com/kubernetes/kubernetes/pull/38909), [@jingxu97](https://github.com/jingxu97))
* Begin paths for internationalization in kubectl ([#36802](https://github.com/kubernetes/kubernetes/pull/36802), [@brendandburns](https://github.com/brendandburns))
* Fixes an issue where commas were not accepted in --from-literal flags when creating secrets. Passing multiple values separated by a comma in a single --from-literal flag is no longer supported. Please use multiple --from-literal flags to provide multiple values. ([#35191](https://github.com/kubernetes/kubernetes/pull/35191), [@SamiHiltunen](https://github.com/SamiHiltunen))
* Support loading UTF16 files if a byte-order-mark is present ([#39008](https://github.com/kubernetes/kubernetes/pull/39008), [@brendandburns](https://github.com/brendandburns))
* Fix fsGroup to vSphere ([#38655](https://github.com/kubernetes/kubernetes/pull/38655), [@abrarshivani](https://github.com/abrarshivani))
* ReplicaSet has onwer ref of the Deployment that created it ([#35676](https://github.com/kubernetes/kubernetes/pull/35676), [@krmayankk](https://github.com/krmayankk))
* Don't evict static pods ([#39059](https://github.com/kubernetes/kubernetes/pull/39059), [@bprashanth](https://github.com/bprashanth))
* Fixed a bug where the --server, --token, and --certificate-authority flags were not overriding the related in-cluster configs when provided in a `kubectl` call inside a cluster. ([#39006](https://github.com/kubernetes/kubernetes/pull/39006), [@fabianofranz](https://github.com/fabianofranz))
* Make fluentd pods critical ([#39146](https://github.com/kubernetes/kubernetes/pull/39146), [@Crassirostris](https://github.com/Crassirostris))
* assign -998 as the oom_score_adj for critical pods (e.g. kube-proxy) ([#39114](https://github.com/kubernetes/kubernetes/pull/39114), [@dchen1107](https://github.com/dchen1107))
* delete continue in monitorNodeStatus ([#38798](https://github.com/kubernetes/kubernetes/pull/38798), [@NickrenREN](https://github.com/NickrenREN))
* add create rolebinding ([#38991](https://github.com/kubernetes/kubernetes/pull/38991), [@deads2k](https://github.com/deads2k))
* Add new command "kubectl set selector" ([#38966](https://github.com/kubernetes/kubernetes/pull/38966), [@kargakis](https://github.com/kargakis))
* Federation: Add `batch/jobs` API objects to federation-apiserver ([#35943](https://github.com/kubernetes/kubernetes/pull/35943), [@jianhuiz](https://github.com/jianhuiz))
* ABAC policies using `"user":"*"` or `"group":"*"` to match all users or groups will only match authenticated requests. To match unauthenticated requests, ABAC policies must explicitly specify `"group":"system:unauthenticated"` ([#38968](https://github.com/kubernetes/kubernetes/pull/38968), [@liggitt](https://github.com/liggitt))
* To add local registry to libvirt_coreos ([#36751](https://github.com/kubernetes/kubernetes/pull/36751), [@sdminonne](https://github.com/sdminonne))
* Add a KUBERNETES_NODE_* section to build kubelet/kube-proxy for windows ([#38919](https://github.com/kubernetes/kubernetes/pull/38919), [@brendandburns](https://github.com/brendandburns))
* Added kubeadm commands to manage bootstrap tokens and the duration they are valid for. ([#35805](https://github.com/kubernetes/kubernetes/pull/35805), [@dgoodwin](https://github.com/dgoodwin))
* AWS: Recognize ca-central-1 region ([#38410](https://github.com/kubernetes/kubernetes/pull/38410), [@justinsb](https://github.com/justinsb))
* Unmount operation should not fail if volume is already unmounted ([#38547](https://github.com/kubernetes/kubernetes/pull/38547), [@rkouj](https://github.com/rkouj))
* Changed default scsi controller type in vSphere Cloud Provider ([#38426](https://github.com/kubernetes/kubernetes/pull/38426), [@abrarshivani](https://github.com/abrarshivani))
* Move non-generic apiserver code out of the generic packages ([#38191](https://github.com/kubernetes/kubernetes/pull/38191), [@sttts](https://github.com/sttts))
* Remove extensions/v1beta1 Jobs resource, and job/v1beta1 generator. ([#38614](https://github.com/kubernetes/kubernetes/pull/38614), [@soltysh](https://github.com/soltysh))
* Admit critical pods in the kubelet ([#38836](https://github.com/kubernetes/kubernetes/pull/38836), [@bprashanth](https://github.com/bprashanth))
* Use daemonset in docker registry add on ([#35582](https://github.com/kubernetes/kubernetes/pull/35582), [@surajssd](https://github.com/surajssd))
* Node affinity has moved from annotations to api fields in the pod spec.  Node affinity that is defined in the annotations will be ignored. ([#37299](https://github.com/kubernetes/kubernetes/pull/37299), [@rrati](https://github.com/rrati))
* Since `kubernetes.tar.gz` no longer includes client or server binaries, `cluster/kube-{up,down,push}.sh` now automatically download released binaries if they are missing. ([#38730](https://github.com/kubernetes/kubernetes/pull/38730), [@ixdy](https://github.com/ixdy))
* genericapiserver: turn APIContainer.SecretRoutes into a real ServeMux ([#38826](https://github.com/kubernetes/kubernetes/pull/38826), [@sttts](https://github.com/sttts))
* Migrated fluentd addon to daemon set ([#32088](https://github.com/kubernetes/kubernetes/pull/32088), [@piosz](https://github.com/piosz))
* AWS: Add sequential allocator for device names. ([#38818](https://github.com/kubernetes/kubernetes/pull/38818), [@jsafrane](https://github.com/jsafrane))
* Add 'X-Content-Type-Options: nosniff" to some error messages ([#37190](https://github.com/kubernetes/kubernetes/pull/37190), [@brendandburns](https://github.com/brendandburns))
* Display pod node selectors with kubectl describe. ([#36396](https://github.com/kubernetes/kubernetes/pull/36396), [@aveshagarwal](https://github.com/aveshagarwal))
* The main repository does not keep multiple releases of clientsets anymore. Please find previous releases at https://github.com/kubernetes/client-go ([#38154](https://github.com/kubernetes/kubernetes/pull/38154), [@caesarxuchao](https://github.com/caesarxuchao))
* Remove a release-note on multiple OpenAPI specs ([#38732](https://github.com/kubernetes/kubernetes/pull/38732), [@mbohlool](https://github.com/mbohlool))
* genericapiserver: unify swagger and openapi in config ([#38690](https://github.com/kubernetes/kubernetes/pull/38690), [@sttts](https://github.com/sttts))
* fix connection upgrades through kuberentes-discovery ([#38724](https://github.com/kubernetes/kubernetes/pull/38724), [@deads2k](https://github.com/deads2k))
* Fixes bug in resolving client-requested API versions ([#38533](https://github.com/kubernetes/kubernetes/pull/38533), [@DirectXMan12](https://github.com/DirectXMan12))
* apiserver(s): Replace glog.Fatals with fmt.Errorfs ([#38175](https://github.com/kubernetes/kubernetes/pull/38175), [@sttts](https://github.com/sttts))
* Remove Azure Subnet RouteTable check ([#38334](https://github.com/kubernetes/kubernetes/pull/38334), [@mogthesprog](https://github.com/mogthesprog))
* Ensure the GCI metadata files do not have newline at the end ([#38727](https://github.com/kubernetes/kubernetes/pull/38727), [@Amey-D](https://github.com/Amey-D))
* Significantly speed-up make ([#38700](https://github.com/kubernetes/kubernetes/pull/38700), [@sttts](https://github.com/sttts))
* add QoS pod status field ([#37968](https://github.com/kubernetes/kubernetes/pull/37968), [@sjenning](https://github.com/sjenning))
* Fixed validation of multizone cluster for GCE ([#38695](https://github.com/kubernetes/kubernetes/pull/38695), [@jszczepkowski](https://github.com/jszczepkowski))
* Fixed detection of master during creation of multizone nodes cluster by kube-up. ([#38617](https://github.com/kubernetes/kubernetes/pull/38617), [@jszczepkowski](https://github.com/jszczepkowski))
* Update CHANGELOG.md to warn about anon auth flag ([#38675](https://github.com/kubernetes/kubernetes/pull/38675), [@erictune](https://github.com/erictune))
* Fixes NotAuthenticated errors that appear in the kubelet and kube-controller-manager due to never logging in to vSphere ([#36169](https://github.com/kubernetes/kubernetes/pull/36169), [@robdaemon](https://github.com/robdaemon))
* Fix an issue where AWS tear-down leaks an DHCP Option Set. ([#38645](https://github.com/kubernetes/kubernetes/pull/38645), [@zmerlynn](https://github.com/zmerlynn))
* Issue a warning when using `kubectl apply` on a resource lacking the `LastAppliedConfig` annotation ([#36672](https://github.com/kubernetes/kubernetes/pull/36672), [@ymqytw](https://github.com/ymqytw))
* Re-add /healthz/ping handler in genericapiserver ([#38603](https://github.com/kubernetes/kubernetes/pull/38603), [@sttts](https://github.com/sttts))
* Fail kubelet if runtime is unresponsive for 30 seconds ([#38527](https://github.com/kubernetes/kubernetes/pull/38527), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Add support for Azure Container Registry, update Azure dependencies ([#37783](https://github.com/kubernetes/kubernetes/pull/37783), [@brendandburns](https://github.com/brendandburns))
* Fix panic in vSphere cloud provider ([#38423](https://github.com/kubernetes/kubernetes/pull/38423), [@BaluDontu](https://github.com/BaluDontu))
* fix broken cluster/centos and enhance the style ([#34002](https://github.com/kubernetes/kubernetes/pull/34002), [@xiaoping378](https://github.com/xiaoping378))
* Ability to quota storage by storage class ([#34554](https://github.com/kubernetes/kubernetes/pull/34554), [@derekwaynecarr](https://github.com/derekwaynecarr))
* [Part 2] Adding s390x cross-compilation support for gcr.io images in this repo ([#36050](https://github.com/kubernetes/kubernetes/pull/36050), [@gajju26](https://github.com/gajju26))
* kubectl run --rm no longer prints "pod xxx deleted" ([#38429](https://github.com/kubernetes/kubernetes/pull/38429), [@duglin](https://github.com/duglin))
* Bump GCE debian image to container-vm-v20161208 ([release notes](https://cloud.google.com/compute/docs/containers/container_vms#changelog)) ([#38432](https://github.com/kubernetes/kubernetes/pull/38432), [@timstclair](https://github.com/timstclair))
* Kubelet: Add image cache. ([#38375](https://github.com/kubernetes/kubernetes/pull/38375), [@Random-Liu](https://github.com/Random-Liu))
* Allow a selector when retrieving logs ([#32752](https://github.com/kubernetes/kubernetes/pull/32752), [@fraenkel](https://github.com/fraenkel))
* Fix unmountDevice issue caused by shared mount in GCI ([#38411](https://github.com/kubernetes/kubernetes/pull/38411), [@jingxu97](https://github.com/jingxu97))
* [Federation] Implement dry run support in kubefed init ([#36447](https://github.com/kubernetes/kubernetes/pull/36447), [@irfanurrehman](https://github.com/irfanurrehman))
* Fix space issue in volumePath with vSphere Cloud Provider ([#38338](https://github.com/kubernetes/kubernetes/pull/38338), [@BaluDontu](https://github.com/BaluDontu))
* [Federation] Make federation etcd PVC size configurable ([#36310](https://github.com/kubernetes/kubernetes/pull/36310), [@irfanurrehman](https://github.com/irfanurrehman))
* Allow no ports when exposing headless service ([#32811](https://github.com/kubernetes/kubernetes/pull/32811), [@fraenkel](https://github.com/fraenkel))
* Wait for the port to be ready before starting ([#38260](https://github.com/kubernetes/kubernetes/pull/38260), [@fraenkel](https://github.com/fraenkel))
* Add Version to the resource printer for 'get nodes' ([#37943](https://github.com/kubernetes/kubernetes/pull/37943), [@ailusazh](https://github.com/ailusazh))
* contribute deis/registry-proxy as a replacement for kube-registry-proxy ([#35797](https://github.com/kubernetes/kubernetes/pull/35797), [@bacongobbler](https://github.com/bacongobbler))
* [Part 1] Add support for cross-compiling s390x binaries ([#37092](https://github.com/kubernetes/kubernetes/pull/37092), [@gajju26](https://github.com/gajju26))
* kernel memcg notification enabled via experimental flag ([#38258](https://github.com/kubernetes/kubernetes/pull/38258), [@derekwaynecarr](https://github.com/derekwaynecarr))
* fix permissions when using fsGroup ([#37009](https://github.com/kubernetes/kubernetes/pull/37009), [@sjenning](https://github.com/sjenning))
* Pipe get options to storage ([#37693](https://github.com/kubernetes/kubernetes/pull/37693), [@wojtek-t](https://github.com/wojtek-t))
* add a configuration for kubelet to register as a node with taints ([#31647](https://github.com/kubernetes/kubernetes/pull/31647), [@mikedanese](https://github.com/mikedanese))
* Remove genericapiserver.Options.MasterServiceNamespace ([#38186](https://github.com/kubernetes/kubernetes/pull/38186), [@sttts](https://github.com/sttts))
* The --long-running-request-regexp flag to kube-apiserver is deprecated and will be removed in a future release. Long-running requests are now detected based on specific verbs (watch, proxy) or subresources (proxy, portforward, log, exec, attach). ([#38119](https://github.com/kubernetes/kubernetes/pull/38119), [@liggitt](https://github.com/liggitt))
* Better compat with very old iptables (e.g. CentOS 6) ([#37594](https://github.com/kubernetes/kubernetes/pull/37594), [@thockin](https://github.com/thockin))
* Fix GCI mounter issue ([#38124](https://github.com/kubernetes/kubernetes/pull/38124), [@jingxu97](https://github.com/jingxu97))
* Kubelet will no longer set hairpin mode on every interface on the machine when an error occurs in setting up hairpin for a specific interface. ([#36990](https://github.com/kubernetes/kubernetes/pull/36990), [@bboreham](https://github.com/bboreham))
* fix mesos unit tests ([#38196](https://github.com/kubernetes/kubernetes/pull/38196), [@deads2k](https://github.com/deads2k))
* Add `--controllers` flag to federation controller manager for enable/disable federation ingress controller ([#36643](https://github.com/kubernetes/kubernetes/pull/36643), [@kzwang](https://github.com/kzwang))
* Allow backendpools in Azure Load Balancers which are not owned by cloud provider ([#36882](https://github.com/kubernetes/kubernetes/pull/36882), [@codablock](https://github.com/codablock))
* remove rbac super user ([#38121](https://github.com/kubernetes/kubernetes/pull/38121), [@deads2k](https://github.com/deads2k))
* API server have two separate limits for read-only and mutating inflight requests. ([#36064](https://github.com/kubernetes/kubernetes/pull/36064), [@gmarek](https://github.com/gmarek))
* check the value of min and max in kubectl ([#37789](https://github.com/kubernetes/kubernetes/pull/37789), [@yarntime](https://github.com/yarntime))
* The glusterfs dynamic volume provisioner will now choose a unique GID for new persistent volumes from a range that can be configured in the storage class with the "gidMin" and "gidMax" parameters. The default range is 2000 - 2147483647 (max int32). ([#37886](https://github.com/kubernetes/kubernetes/pull/37886), [@obnoxxx](https://github.com/obnoxxx))
* Add kubectl create poddisruptionbudget command ([#36646](https://github.com/kubernetes/kubernetes/pull/36646), [@kargakis](https://github.com/kargakis))
* Set kernel.softlockup_panic =1 based on the flag. ([#38001](https://github.com/kubernetes/kubernetes/pull/38001), [@dchen1107](https://github.com/dchen1107))
* portfordwardtester: avoid data loss during send+close+exit ([#37103](https://github.com/kubernetes/kubernetes/pull/37103), [@sttts](https://github.com/sttts))
* Enable containerized mounter only for nfs and glusterfs types ([#37990](https://github.com/kubernetes/kubernetes/pull/37990), [@jingxu97](https://github.com/jingxu97))
* Add flag to enable contention profiling in scheduler. ([#37357](https://github.com/kubernetes/kubernetes/pull/37357), [@gmarek](https://github.com/gmarek))
* Add kubernetes-anywhere as a new e2e deployment option ([#37019](https://github.com/kubernetes/kubernetes/pull/37019), [@pipejakob](https://github.com/pipejakob))
* add create clusterrolebinding command ([#37098](https://github.com/kubernetes/kubernetes/pull/37098), [@deads2k](https://github.com/deads2k))
* kubectl create service externalname ([#34789](https://github.com/kubernetes/kubernetes/pull/34789), [@AdoHe](https://github.com/AdoHe))
* Fix logic error in graceful deletion ([#37721](https://github.com/kubernetes/kubernetes/pull/37721), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Exit with error if <version number or publication> is not the final parameter. ([#37723](https://github.com/kubernetes/kubernetes/pull/37723), [@mtaufen](https://github.com/mtaufen))
* Fix Service Update on LoadBalancerSourceRanges Field ([#37720](https://github.com/kubernetes/kubernetes/pull/37720), [@freehan](https://github.com/freehan))
* Bug fix. Incoming UDP packets not reach newly deployed services ([#32561](https://github.com/kubernetes/kubernetes/pull/32561), [@zreigz](https://github.com/zreigz))
* GCI: Remove /var/lib/docker/network ([#37593](https://github.com/kubernetes/kubernetes/pull/37593), [@yujuhong](https://github.com/yujuhong))
* configure local-up-cluster.sh to handle auth proxies ([#36838](https://github.com/kubernetes/kubernetes/pull/36838), [@deads2k](https://github.com/deads2k))
* Add `clusterid`, an optional parameter to storageclass. ([#36437](https://github.com/kubernetes/kubernetes/pull/36437), [@humblec](https://github.com/humblec))
* local-up-cluster: avoid sudo for control plane ([#37443](https://github.com/kubernetes/kubernetes/pull/37443), [@sttts](https://github.com/sttts))
* Add error message when trying to use clusterrole with namespace in kubectl ([#36424](https://github.com/kubernetes/kubernetes/pull/36424), [@xilabao](https://github.com/xilabao))
* `kube-up.sh`/`kube-down.sh` no longer force update gcloud for provider=gce|gke. ([#36292](https://github.com/kubernetes/kubernetes/pull/36292), [@jlowdermilk](https://github.com/jlowdermilk))
* Fix issue when attempting to unmount a wrong vSphere volume ([#37413](https://github.com/kubernetes/kubernetes/pull/37413), [@BaluDontu](https://github.com/BaluDontu))
* Fix the equality checks for numeric values in cluster/gce/util.sh. ([#37638](https://github.com/kubernetes/kubernetes/pull/37638), [@roberthbailey](https://github.com/roberthbailey))
* kubelet: don't reject pods without adding them to the pod manager ([#37661](https://github.com/kubernetes/kubernetes/pull/37661), [@yujuhong](https://github.com/yujuhong))
* Modify GCI mounter to enable NFSv3 ([#37582](https://github.com/kubernetes/kubernetes/pull/37582), [@jingxu97](https://github.com/jingxu97))
* Fix photon controller plugin to construct with correct PdID ([#37167](https://github.com/kubernetes/kubernetes/pull/37167), [@luomiao](https://github.com/luomiao))
* Collect logs for dead kubelets too ([#37671](https://github.com/kubernetes/kubernetes/pull/37671), [@mtaufen](https://github.com/mtaufen))
* Set Dashboard UI version to v1.5.0 ([#37684](https://github.com/kubernetes/kubernetes/pull/37684), [@rf232](https://github.com/rf232))
* When deleting an object with `--grace-period=0`, the client will begin a graceful deletion and wait until the resource is fully deleted.  To force deletion, use the `--force` flag. ([#37263](https://github.com/kubernetes/kubernetes/pull/37263), [@smarterclayton](https://github.com/smarterclayton))
* federation service controller: stop deleting services from underlying clusters when federated service is deleted. ([#37353](https://github.com/kubernetes/kubernetes/pull/37353), [@nikhiljindal](https://github.com/nikhiljindal))
* Fix nil pointer dereference in test framework ([#37583](https://github.com/kubernetes/kubernetes/pull/37583), [@mtaufen](https://github.com/mtaufen))
* Kubernetes now automatically installs a StorageClass object when deployed on ([#31617](https://github.com/kubernetes/kubernetes/pull/31617), [@jsafrane](https://github.com/jsafrane))
    * AWS, Google Compute Engine, Google Container Engine, and OpenStack using
    * the default kube-up.sh scripts. This StorageClass is marked as default so that
    * a PersistentVolumeClaim without a StorageClass annotation now results in
    * automatic provisioning of storage (GCE PersistentDisk on Google Cloud, AWS
    * EBS on AWS, and Cinder on OpenStack). In previous versions of Kubernetes
    * a PersistentVolumeClaim without a StorageClass annotation on these cloud
    * platforms would be satisfied by manually-created PersistentVolume objects.
    * Administrators can choose to disable this behavior by deleting the automatically
    * installed StorageClass API object. Alternatively, administrators may choose to
    * keep the automatically installed StorageClass and only disable the defaulting
    * behavior by removing the "is-default-class" annotation from the StorageClass
    * API object.
* Fix TestServiceAlloc flakes ([#37487](https://github.com/kubernetes/kubernetes/pull/37487), [@wojtek-t](https://github.com/wojtek-t))
* Use gsed on the Mac ([#37562](https://github.com/kubernetes/kubernetes/pull/37562), [@roberthbailey](https://github.com/roberthbailey))
* Try self-repair scheduler cache or panic ([#37379](https://github.com/kubernetes/kubernetes/pull/37379), [@wojtek-t](https://github.com/wojtek-t))
* Fluentd/Elastisearch add-on: correctly parse and index kubernetes labels ([#36857](https://github.com/kubernetes/kubernetes/pull/36857), [@Shrugs](https://github.com/Shrugs))
* Mention overflows when mistakenly call function FromInt ([#36487](https://github.com/kubernetes/kubernetes/pull/36487), [@xialonglee](https://github.com/xialonglee))
* Update doc for kubectl apply ([#37397](https://github.com/kubernetes/kubernetes/pull/37397), [@ymqytw](https://github.com/ymqytw))
* Removes shorthand flag -w from kubectl apply ([#37345](https://github.com/kubernetes/kubernetes/pull/37345), [@MrHohn](https://github.com/MrHohn))
* Curating Owners: pkg/api ([#36525](https://github.com/kubernetes/kubernetes/pull/36525), [@apelisse](https://github.com/apelisse))
* Reduce verbosity of volume reconciler when attaching volumes ([#36900](https://github.com/kubernetes/kubernetes/pull/36900), [@codablock](https://github.com/codablock))
* Federated Ingress Proposal ([#29793](https://github.com/kubernetes/kubernetes/pull/29793), [@quinton-hoole](https://github.com/quinton-hoole))

Please see the [Releases Page](https://github.com/kubernetes/kubernetes/releases) for older releases.

Release notes of older releases can be found in:
- [CHANGELOG-1.2.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.2.md)
- [CHANGELOG-1.3.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.3.md)
- [CHANGELOG-1.4.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.4.md)
- [CHANGELOG-1.5.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.5.md)

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CHANGELOG.md?pixel)]()
