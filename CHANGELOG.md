<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.7.7](#v177)
  - [Downloads for v1.7.7](#downloads-for-v177)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.7.6](#changelog-since-v176)
    - [Other notable changes](#other-notable-changes)
- [v1.8.0](#v180)
  - [Downloads for v1.8.0](#downloads-for-v180)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Introduction to v1.8.0](#introduction-to-v180)
  - [Major Themes](#major-themes)
    - [SIG API Machinery](#sig-api-machinery)
    - [SIG Apps](#sig-apps)
    - [SIG Auth](#sig-auth)
    - [SIG Autoscaling](#sig-autoscaling)
    - [SIG Cluster Lifecycle](#sig-cluster-lifecycle)
    - [SIG Instrumentation](#sig-instrumentation)
    - [SIG Multi-cluster (formerly known as SIG Federation)](#sig-multi-cluster-formerly-known-as-sig-federation)
    - [SIG Node](#sig-node)
    - [SIG Network](#sig-network)
    - [SIG Scalability](#sig-scalability)
    - [SIG Scheduling](#sig-scheduling)
    - [SIG Storage](#sig-storage)
  - [Before Upgrading](#before-upgrading)
  - [Known Issues](#known-issues)
  - [Deprecations](#deprecations)
    - [Apps](#apps)
    - [Auth](#auth)
    - [Autoscaling](#autoscaling)
    - [Cluster Lifecycle](#cluster-lifecycle)
    - [OpenStack](#openstack)
    - [Scheduling](#scheduling)
  - [Notable Features](#notable-features)
    - [Workloads API (apps/v1beta2)](#workloads-api-appsv1beta2)
      - [API Object Additions and Migrations](#api-object-additions-and-migrations)
      - [Behavioral Changes](#behavioral-changes)
      - [Defaults](#defaults)
    - [Workloads API (batch)](#workloads-api-batch)
      - [CLI Changes](#cli-changes)
      - [Scheduling](#scheduling-1)
      - [Storage](#storage)
    - [Cluster Federation](#cluster-federation)
      - [[alpha] Federated Jobs](#[alpha]-federated-jobs)
      - [[alpha] Federated Horizontal Pod Autoscaling (HPA)](#[alpha]-federated-horizontal-pod-autoscaling-hpa)
    - [Node Components](#node-components)
      - [Autoscaling and Metrics](#autoscaling-and-metrics)
        - [Cluster Autoscaler](#cluster-autoscaler)
      - [Container Runtime Interface (CRI)](#container-runtime-interface-cri)
      - [kubelet](#kubelet)
    - [Auth](#auth-1)
    - [Cluster Lifecycle](#cluster-lifecycle-1)
      - [kubeadm](#kubeadm)
      - [kops](#kops)
      - [Cluster Discovery/Bootstrap](#cluster-discoverybootstrap)
      - [Multi-platform](#multi-platform)
      - [Cloud Providers](#cloud-providers)
    - [Network](#network)
      - [network-policy](#network-policy)
      - [kube-proxy ipvs mode](#kube-proxy-ipvs-mode)
    - [API Machinery](#api-machinery)
      - [kube-apiserver](#kube-apiserver)
      - [Dynamic Admission Control](#dynamic-admission-control)
      - [Custom Resource Definitions (CRDs)](#custom-resource-definitions-crds)
      - [Garbage Collector](#garbage-collector)
      - [Monitoring/Prometheus](#monitoringprometheus)
      - [Go Client](#go-client)
  - [External Dependencies](#external-dependencies)
- [v1.8.0-rc.1](#v180-rc1)
  - [Downloads for v1.8.0-rc.1](#downloads-for-v180-rc1)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.8.0-beta.1](#changelog-since-v180-beta1)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-1)
- [v1.9.0-alpha.1](#v190-alpha1)
  - [Downloads for v1.9.0-alpha.1](#downloads-for-v190-alpha1)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.8.0-alpha.3](#changelog-since-v180-alpha3)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-2)
- [v1.7.6](#v176)
  - [Downloads for v1.7.6](#downloads-for-v176)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.7.5](#changelog-since-v175)
    - [Other notable changes](#other-notable-changes-3)
- [v1.8.0-beta.1](#v180-beta1)
  - [Downloads for v1.8.0-beta.1](#downloads-for-v180-beta1)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.8.0-alpha.3](#changelog-since-v180-alpha3-1)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-4)
- [v1.7.5](#v175)
  - [Downloads for v1.7.5](#downloads-for-v175)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.7.4](#changelog-since-v174)
    - [Other notable changes](#other-notable-changes-5)
- [v1.8.0-alpha.3](#v180-alpha3)
  - [Downloads for v1.8.0-alpha.3](#downloads-for-v180-alpha3)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
  - [Changelog since v1.8.0-alpha.2](#changelog-since-v180-alpha2)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-6)
- [v1.7.4](#v174)
  - [Downloads for v1.7.4](#downloads-for-v174)
    - [Client Binaries](#client-binaries-8)
    - [Server Binaries](#server-binaries-8)
    - [Node Binaries](#node-binaries-8)
  - [Changelog since v1.7.3](#changelog-since-v173)
    - [Other notable changes](#other-notable-changes-7)
- [v1.7.3](#v173)
  - [Downloads for v1.7.3](#downloads-for-v173)
    - [Client Binaries](#client-binaries-9)
    - [Server Binaries](#server-binaries-9)
    - [Node Binaries](#node-binaries-9)
  - [Changelog since v1.7.2](#changelog-since-v172)
    - [Other notable changes](#other-notable-changes-8)
- [v1.7.2](#v172)
  - [Downloads for v1.7.2](#downloads-for-v172)
    - [Client Binaries](#client-binaries-10)
    - [Server Binaries](#server-binaries-10)
    - [Node Binaries](#node-binaries-10)
  - [Changelog since v1.7.1](#changelog-since-v171)
    - [Other notable changes](#other-notable-changes-9)
- [v1.7.1](#v171)
  - [Downloads for v1.7.1](#downloads-for-v171)
    - [Client Binaries](#client-binaries-11)
    - [Server Binaries](#server-binaries-11)
    - [Node Binaries](#node-binaries-11)
  - [Changelog since v1.7.0](#changelog-since-v170)
    - [Other notable changes](#other-notable-changes-10)
- [v1.8.0-alpha.2](#v180-alpha2)
  - [Downloads for v1.8.0-alpha.2](#downloads-for-v180-alpha2)
    - [Client Binaries](#client-binaries-12)
    - [Server Binaries](#server-binaries-12)
    - [Node Binaries](#node-binaries-12)
  - [Changelog since v1.7.0](#changelog-since-v170-1)
    - [Action Required](#action-required-4)
    - [Other notable changes](#other-notable-changes-11)
- [v1.7.0](#v170)
  - [Downloads for v1.7.0](#downloads-for-v170)
    - [Client Binaries](#client-binaries-13)
    - [Server Binaries](#server-binaries-13)
    - [Node Binaries](#node-binaries-13)
  - [**Major Themes**](#major-themes-1)
  - [**Action Required Before Upgrading**](#action-required-before-upgrading)
    - [Network](#network-1)
    - [Storage](#storage-1)
    - [API Machinery](#api-machinery-1)
    - [Controller Manager](#controller-manager)
    - [kubectl (CLI)](#kubectl-cli)
    - [kubeadm](#kubeadm-1)
    - [Cloud Providers](#cloud-providers-1)
  - [**Known Issues**](#known-issues-1)
  - [**Deprecations**](#deprecations-1)
    - [Cluster provisioning scripts](#cluster-provisioning-scripts)
    - [Client libraries](#client-libraries)
    - [DaemonSet](#daemonset)
    - [kube-proxy](#kube-proxy)
    - [Namespace](#namespace)
    - [Scheduling](#scheduling-2)
  - [**Notable Features**](#notable-features-1)
  - [Kubefed](#kubefed)
    - [**Kubernetes API**](#kubernetes-api)
      - [User Provided Extensions](#user-provided-extensions)
    - [**Application Deployment**](#application-deployment)
      - [StatefulSet](#statefulset)
      - [DaemonSet](#daemonset-1)
      - [Deployments](#deployments)
      - [PodDisruptionBudget](#poddisruptionbudget)
    - [**Security**](#security)
      - [Admission Control](#admission-control)
      - [TLS Bootstrapping](#tls-bootstrapping)
      - [Audit Logging](#audit-logging)
      - [Encryption at Rest](#encryption-at-rest)
      - [Node Authorization](#node-authorization)
    - [**Application Autoscaling**](#application-autoscaling)
      - [Horizontal Pod Autoscaler](#horizontal-pod-autoscaler)
    - [**Cluster Lifecycle**](#cluster-lifecycle-2)
      - [kubeadm](#kubeadm-2)
      - [Cloud Provider Support](#cloud-provider-support)
    - [**Cluster Federation**](#cluster-federation-1)
      - [Placement Policy](#placement-policy)
      - [Cluster Selection](#cluster-selection)
    - [**Instrumentation**](#instrumentation)
      - [Core Metrics API](#core-metrics-api)
    - [**Internationalization**](#internationalization)
    - [**kubectl (CLI)**](#kubectl-cli-1)
    - [**Networking**](#networking)
      - [Network Policy](#network-policy-1)
      - [Load Balancing](#load-balancing)
    - [**Node Components**](#node-components-1)
      - [Container Runtime Interface](#container-runtime-interface)
    - [**Scheduling**](#scheduling-3)
      - [Scheduler Extender](#scheduler-extender)
    - [**Storage**](#storage-2)
      - [Local Storage](#local-storage)
      - [Volume Plugins](#volume-plugins)
      - [Metrics](#metrics)
    - [**Other notable changes**](#other-notable-changes-12)
      - [Admission plugin](#admission-plugin)
      - [API Machinery](#api-machinery-2)
      - [Application autoscaling](#application-autoscaling-1)
      - [Application Deployment](#application-deployment-1)
      - [Cluster Autoscaling](#cluster-autoscaling)
      - [Cloud Provider Enhancement](#cloud-provider-enhancement)
      - [Cluster Provisioning](#cluster-provisioning)
      - [Cluster federation](#cluster-federation-2)
      - [Credential provider](#credential-provider)
      - [Information for Kubernetes clients (openapi, swagger, client-go)](#information-for-kubernetes-clients-openapi-swagger-client-go)
      - [Instrumentation](#instrumentation-1)
      - [Internal storage layer](#internal-storage-layer)
      - [Kubernetes Dashboard](#kubernetes-dashboard)
      - [kube-dns](#kube-dns)
      - [kube-proxy](#kube-proxy-1)
      - [kube-scheduler](#kube-scheduler)
      - [Storage](#storage-3)
      - [Networking](#networking-1)
      - [Node controller](#node-controller)
      - [Node Components](#node-components-2)
      - [Scheduling](#scheduling-4)
      - [Security](#security-1)
      - [Scalability](#scalability)
  - [**External Dependency Version Information**](#external-dependency-version-information)
    - [Previous Releases Included in v1.7.0](#previous-releases-included-in-v170)
- [v1.7.0-rc.1](#v170-rc1)
  - [Downloads for v1.7.0-rc.1](#downloads-for-v170-rc1)
    - [Client Binaries](#client-binaries-14)
    - [Server Binaries](#server-binaries-14)
    - [Node Binaries](#node-binaries-14)
  - [Changelog since v1.7.0-beta.2](#changelog-since-v170-beta2)
    - [Action Required](#action-required-5)
    - [Other notable changes](#other-notable-changes-13)
- [v1.8.0-alpha.1](#v180-alpha1)
  - [Downloads for v1.8.0-alpha.1](#downloads-for-v180-alpha1)
    - [Client Binaries](#client-binaries-15)
    - [Server Binaries](#server-binaries-15)
    - [Node Binaries](#node-binaries-15)
  - [Changelog since v1.7.0-alpha.4](#changelog-since-v170-alpha4)
    - [Action Required](#action-required-6)
    - [Other notable changes](#other-notable-changes-14)
- [v1.7.0-beta.2](#v170-beta2)
  - [Downloads for v1.7.0-beta.2](#downloads-for-v170-beta2)
    - [Client Binaries](#client-binaries-16)
    - [Server Binaries](#server-binaries-16)
    - [Node Binaries](#node-binaries-16)
  - [Changelog since v1.7.0-beta.1](#changelog-since-v170-beta1)
    - [Action Required](#action-required-7)
    - [Other notable changes](#other-notable-changes-15)
- [v1.7.0-beta.1](#v170-beta1)
  - [Downloads for v1.7.0-beta.1](#downloads-for-v170-beta1)
    - [Client Binaries](#client-binaries-17)
    - [Server Binaries](#server-binaries-17)
    - [Node Binaries](#node-binaries-17)
  - [Changelog since v1.7.0-alpha.4](#changelog-since-v170-alpha4-1)
    - [Action Required](#action-required-8)
    - [Other notable changes](#other-notable-changes-16)
- [v1.7.0-alpha.4](#v170-alpha4)
  - [Downloads for v1.7.0-alpha.4](#downloads-for-v170-alpha4)
    - [Client Binaries](#client-binaries-18)
    - [Server Binaries](#server-binaries-18)
    - [Node Binaries](#node-binaries-18)
  - [Changelog since v1.7.0-alpha.3](#changelog-since-v170-alpha3)
    - [Action Required](#action-required-9)
    - [Other notable changes](#other-notable-changes-17)
- [v1.7.0-alpha.3](#v170-alpha3)
  - [Downloads for v1.7.0-alpha.3](#downloads-for-v170-alpha3)
    - [Client Binaries](#client-binaries-19)
    - [Server Binaries](#server-binaries-19)
    - [Node Binaries](#node-binaries-19)
  - [Changelog since v1.7.0-alpha.2](#changelog-since-v170-alpha2)
    - [Action Required](#action-required-10)
    - [Other notable changes](#other-notable-changes-18)
- [v1.7.0-alpha.2](#v170-alpha2)
  - [Downloads for v1.7.0-alpha.2](#downloads-for-v170-alpha2)
    - [Client Binaries](#client-binaries-20)
    - [Server Binaries](#server-binaries-20)
  - [Changelog since v1.7.0-alpha.1](#changelog-since-v170-alpha1)
    - [Action Required](#action-required-11)
    - [Other notable changes](#other-notable-changes-19)
- [v1.7.0-alpha.1](#v170-alpha1)
  - [Downloads for v1.7.0-alpha.1](#downloads-for-v170-alpha1)
    - [Client Binaries](#client-binaries-21)
    - [Server Binaries](#server-binaries-21)
  - [Changelog since v1.6.0](#changelog-since-v160)
    - [Other notable changes](#other-notable-changes-20)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.7.7

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.7


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes.tar.gz) | `1fbf1672931464c7b66b93298a6623c97727d9359e5409e7e139a7fbec486591`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-src.tar.gz) | `572eda617bdfc4456a5d370a4616bea5684ff8e999faf4677f4665f181961d86`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-darwin-386.tar.gz) | `661700a452f9ca1c91530e9d0ac1ef7552ae75cfaa86eaa99021b0f30300acd5`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-darwin-amd64.tar.gz) | `eceadcbb092f8bde9d09a1a170aa1ae2af5c07f399995750915a53f0ebbb9f45`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-linux-386.tar.gz) | `2856189ab86b440439bf1a3eab984fa24a1e2280c0741422940c5f06fe66e49e`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-linux-amd64.tar.gz) | `c314a175fe64c7874d0381037d4ffa8bbfdb729af52f8081a9530771203b3852`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-linux-arm64.tar.gz) | `5ad395eff828384feec88f624624fe4da822def6e85a540136bbc1968c5f4b6c`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-linux-arm.tar.gz) | `4636e3f5d1084a31c5abbffe775d241f75bb62d42624b87a7bb85e01f4bdd558`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-linux-ppc64le.tar.gz) | `08943b8745d463c82c29edaf8adfbb22d5409a57a1d88cbe3d08f584bcd36582`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-linux-s390x.tar.gz) | `71efa60865c2bbc7024e60f4437404d68417e7855586896ce15856f94972d4e4`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-windows-386.tar.gz) | `1af19fcb54371732839cf658cb62d6092aef335b234c735a13119b88b667893d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-client-windows-amd64.tar.gz) | `887db68565adac992e0cb2989058b958b60ce4e93704c4286a013277d5e545c5`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-server-linux-amd64.tar.gz) | `674d73c536e0fccd0c8a773d53c94c27257a63b3a91e36be7b045d6d4a43bd8a`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-server-linux-arm64.tar.gz) | `945b7e8d632e9aa5aff7f27d83049a5434472c5fc7ae60010478af42f2c7d85c`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-server-linux-arm.tar.gz) | `ae535d3875242fadac615655aa86fbefcf86ec244705a9ededbe34e46419ad22`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-server-linux-ppc64le.tar.gz) | `1bd286bc6aaea225191953a576fd3be6721624f5baa441257036a7efd382f293`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-server-linux-s390x.tar.gz) | `742e11b8eb127ed5fc1f2520f8c4428c4fdace065412f048c6fe6656d7f165be`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-node-linux-amd64.tar.gz) | `37f0e39673dcaebec761929b13d7a4951cddf9f772adf68d4e43b0783d0a0897`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-node-linux-arm64.tar.gz) | `35a1b338484aa6c031a6a3b671e626605d3c89cd9da81ab009b12e69ef9440a2`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-node-linux-arm.tar.gz) | `99bedd7379faafde9917090f7c98148b2e9a8b00705738a8ce3e6863644a030e`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-node-linux-ppc64le.tar.gz) | `5e9235f4ea823dc6c074ac2d1fcdd23786efc5c9908bf053c7d92540cbf8f4bb`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-node-linux-s390x.tar.gz) | `43ed881a44d125e0bf9b00725cfa48f77e7e61661c43e651914879fe2e3305d0`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.7/kubernetes-node-windows-amd64.tar.gz) | `938fa3b2cf0ccc6ab1deb6259e49e88e982cc46dcf801365ba48dda616841ca6`

## Changelog since v1.7.6

### Other notable changes

* Update kube-dns to 1.14.5 ([#53114](https://github.com/kubernetes/kubernetes/pull/53114), [@bowei](https://github.com/bowei))
* StatefulSet will now fill the `hostname` and `subdomain` fields if they're empty on existing Pods it owns. This allows it to self-correct the issue where StatefulSet Pod DNS entries disappear after upgrading to v1.7.x ([#48327](https://github.com/kubernetes/kubernetes/pull/48327)). ([#51199](https://github.com/kubernetes/kubernetes/pull/51199), [@kow3ns](https://github.com/kow3ns))
* Third Party Resource tests in the e2e suite were incorrectly marked as part of the conformance bucket.  They are alpha and are not required for conformance. ([#52823](https://github.com/kubernetes/kubernetes/pull/52823), [@smarterclayton](https://github.com/smarterclayton))
* fixes upgrade test to work with tightened validation of initializer names in 1.8 ([#52592](https://github.com/kubernetes/kubernetes/pull/52592), [@liggitt](https://github.com/liggitt))
* Fix inconsistent Prometheus cAdvisor metrics ([#51473](https://github.com/kubernetes/kubernetes/pull/51473), [@bboreham](https://github.com/bboreham))
* Fixed an issue reporting lack of progress for a deployment prematurely ([#52178](https://github.com/kubernetes/kubernetes/pull/52178), [@kargakis](https://github.com/kargakis))
* [fluentd-gcp addon] Bug with event-exporter leaking memory on metrics in clusters with CA is fixed. ([#52263](https://github.com/kubernetes/kubernetes/pull/52263), [@crassirostris](https://github.com/crassirostris))



# v1.8.0

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.8/examples)

## Downloads for v1.8.0


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes.tar.gz) | `802a2bc9e9da6d146c71cc446a5faf9304de47996e86134270c725e6440cbb7d`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-src.tar.gz) | `0ea97d20a2d47d9c5f8e791f63bee7e27f836e1a19cf0f15f39e726ae69906a0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-darwin-386.tar.gz) | `22d82ec72e336700562f537b2e0250eb2700391f9b85b12dfc9a4c61428f1db1`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-darwin-amd64.tar.gz) | `de86af6d5b6da9680e93c3d65d889a8ccb59a3a701f3e4ca7a810ffd85ed5eda`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-linux-386.tar.gz) | `4fef05d4b392c2df9f8ffb33e66ee5415671258100c0180f2e1c0befc37a2ac3`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-linux-amd64.tar.gz) | `bef36b2cdcf66a14aa7fc2354c692fb649dc0521d2a76cd3ebde6ca4cb6bad09`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-linux-arm64.tar.gz) | `4cfd3057db15d1e9e5cabccec3771e1efa37f9d7acb47e90d42fece86daff608`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-linux-arm.tar.gz) | `29d9a5faf6a8a1a911fe675e10b8df665f6b82e8f3ee75ca901062f7a3af43ec`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-linux-ppc64le.tar.gz) | `f467c37c75ba5b7125bc2f831efe3776e2a85eeb7245c11aa8accc26cf14585b`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-linux-s390x.tar.gz) | `e0de490b6ce67abf1b158a57c103098ed974a594c677019032fce3d1c9825138`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-windows-386.tar.gz) | `02ea1cd79b591dbc313fab3d22a985219d46f939d79ecc3106fb21d0cb1422cb`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-client-windows-amd64.tar.gz) | `8ca1f609d1cf5ec6afb330cfb87d33d20af152324bed60fe4d91995328a257ff`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-server-linux-amd64.tar.gz) | `23422a7f11c3eab59d686a52abae1bce2f9e2a0916f98ed05c10591ba9c3cbad`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-server-linux-arm64.tar.gz) | `17a1e99010ae3a38f1aec7b3a09661521aba6c93a2e6dd54a3e0534e7d2fafe4`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-server-linux-arm.tar.gz) | `3aba33a9d06068dbf40418205fb8cb62e2987106093d0c65d99cbdf130e163ee`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-server-linux-ppc64le.tar.gz) | `84192c0d520559dfc257f3823f7bf196928b993619a92a27d36f19d2ef209706`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-server-linux-s390x.tar.gz) | `246da14c49c21f50c5bc0d6fc78c023f71ccb07a83e224fd3e40d62c4d1a09d0`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-node-linux-amd64.tar.gz) | `59589cdd56f14b8e879c1854f98072e5ae7ab36835520179fca4887fa9b705e5`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-node-linux-arm64.tar.gz) | `99d17807a819dd3d2764c2105f8bc90166126451dc38869b652e9c59be85dc39`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-node-linux-arm.tar.gz) | `53b1fa21ba4172bfdad677e60360be3e3f28b26656e83d4b63b038d8a31f3cf0`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-node-linux-ppc64le.tar.gz) | `9d10e2d1417fa6c18c152a5ac0202191bf27aab49e473926707c7de479112f46`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-node-linux-s390x.tar.gz) | `cb4e8e9b00484e3f96307e56c61107329fbfcf6eba8362a971053439f64b0304`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0/kubernetes-node-windows-amd64.tar.gz) | `6ca4af62b53947d854562f5a51f4a02daa4738b68015608a599ae416887ffce8`

## Introduction to v1.8.0

Kubernetes version 1.8 includes new features and enhancements, as well as fixes to identified issues. The release notes contain a brief overview of the important changes introduced in this release. The content is organized by Special Interest Groups ([SIGs][]).

For initial installations, see the [Setup topics][] in the Kubernetes
documentation.

To upgrade to this release from a previous version, take any actions required
[Before Upgrading](#before-upgrading).

For more information about the release and for the latest documentation,
see the [Kubernetes documentation](https://kubernetes.io/docs/home/).

[Setup topics]: https://kubernetes.io/docs/setup/pick-right-solution/
[SIGs]: https://github.com/kubernetes/community/blob/master/sig-list.md


## Major Themes

Kubernetes is developed by community members whose work is organized into
[Special Interest Groups][]. For the 1.8 release, each SIG provides the
themes that guided their work.

[Special Interest Groups]: https://github.com/kubernetes/community/blob/master/sig-list.md

### SIG API Machinery

[SIG API Machinery][] is responsible for all aspects of the API server: API registration and discovery, generic API CRUD semantics, admission control, encoding/decoding, conversion, defaulting, persistence layer (etcd), OpenAPI, third-party resources, garbage collection, and client libraries.

For the 1.8 release, SIG API Machinery focused on stability and on ecosystem enablement. Features include the ability to break large LIST calls into smaller chunks, improved support for API server customization with either custom API servers or Custom Resource Definitions, and client side event spam filtering.

[Sig API Machinery]: https://github.com/kubernetes/community/tree/master/sig-api-machinery

### SIG Apps

[SIG Apps][] focuses on the Kubernetes APIs and the external tools that are required to deploy and operate Kubernetes workloads.

For the 1.8 release, SIG Apps moved the Kubernetes workloads API to the new apps/v1beta2 group and version. The DaemonSet, Deployment, ReplicaSet, and StatefulSet objects are affected by this change. The new apps/v1beta2 group and version provide a stable and consistent API surface for building applications in Kubernetes. For details about deprecations and behavioral changes, see [Notable Features](#notable-features). SIG Apps intends to promote this version to GA in a future release.

[SIG Apps]: https://github.com/kubernetes/community/tree/master/sig-apps

### SIG Auth

[SIG Auth][] is responsible for Kubernetes authentication and authorization, and for
cluster security policies.

For the 1.8 release, SIG Auth focused on stablizing existing features that were introduced
in previous releases. RBAC was moved from beta to v1, and advanced auditing was moved from alpha
to beta. Encryption of resources stored on disk (resources at rest) remained in alpha, and the SIG began exploring integrations with external key management systems.

[SIG Auth]: https://github.com/kubernetes/community/tree/master/sig-auth

### SIG Autoscaling

[SIG Autoscaling][] is responsible for autoscaling-related components,
such as the Horizontal Pod Autoscaler and Cluster Autoscaler.

For the 1.8 release, SIG Autoscaling continued to focus on stabilizing
features introduced in previous releases: the new version of the
Horizontal Pod Autoscaler API, which supports custom metrics, and
the Cluster Autoscaler, which provides improved performance and error reporting.

[SIG Autoscaling]: https://github.com/kubernetes/community/tree/master/sig-autoscaling

### SIG Cluster Lifecycle

[SIG Cluster Lifecycle][] is responsible for the user experience of deploying,
upgrading, and deleting clusters.

For the 1.8 release, SIG Cluster Lifecycle continued to focus on expanding the
capabilities of kubeadm, which is both a user-facing tool to manage clusters
and a building block for higher-level provisioning systems. Starting
with the 1.8 release, kubeadm supports a new upgrade command and includes alpha
support for self hosting the cluster control plane.

[SIG Cluster Lifecycle]: https://github.com/kubernetes/community/tree/master/sig-cluster-lifecycle

### SIG Instrumentation

[SIG Instrumentation][] is responsible for metrics production and
collection.

For the 1.8 release, SIG Instrumentation focused on stabilizing the APIs
and components that are required to support the new version of the Horizontal Pod
Autoscaler API: the resource metrics API, custom metrics API, and
metrics-server, which is the new replacement for Heapster in the default monitoring
pipeline.

[SIG Instrumentation]: https://github.com/kubernetes/community/tree/master/sig-instrumentation

### SIG Multi-cluster (formerly known as SIG Federation)

[SIG Multi-cluster][] is responsible for infrastructure that supports
the efficient and reliable management of multiple Kubernetes clusters,
and applications that run in and across multiple clusters.

For the 1.8 release, SIG Multicluster focussed on expanding the set of
Kubernetes primitives that our Cluster Federation control plane
supports, expanding the number of approaches taken to multi-cluster
management (beyond our initial Federation approach), and preparing
to release Federation for general availability ('GA').

[SIG Multi-cluster]: https://github.com/kubernetes/community/tree/master/sig-federation

### SIG Node

[SIG Node][] is responsible for the components that support the controlled
interactions between pods and host resources, and manage the lifecycle
of pods scheduled on a node.

For the 1.8 release, SIG Node continued to focus
on a broad set of workload types, including hardware and performance
sensitive workloads such as data analytics and deep learning. The SIG also
delivered incremental improvements to node reliability.

[SIG Node]: https://github.com/kubernetes/community/tree/master/sig-node

### SIG Network

[SIG Network][] is responsible for networking components, APIs, and plugins in Kubernetes.

For the 1.8 release, SIG Network enhanced the NetworkPolicy API to support pod egress traffic policies.
The SIG also provided match criteria that allow policy rules to match a source or destination CIDR. Both features are in beta. SIG Network also improved the kube-proxy to include an alpha IPVS mode in addition to the current iptables and userspace modes.

[SIG Network]: https://github.com/kubernetes/community/tree/master/sig-network

### SIG Scalability

[SIG Scalability][] is responsible for scalability testing, measuring and
improving system performance, and answering questions related to scalability.

For the 1.8 release, SIG Scalability focused on automating large cluster
scalability testing in a continuous integration (CI) environment. The SIG
defined a concrete process for scalability testing, created
documentation for the current scalability thresholds, and defined a new set of
Service Level Indicators (SLIs) and Service Level Objectives (SLOs) for the system.
Here's the release [scalability validation report].

[SIG Scalability]: https://github.com/kubernetes/community/tree/master/sig-scalability
[scalability validation report]: https://github.com/kubernetes/features/tree/master/release-1.8/scalability_validation_report.md

### SIG Scheduling

[SIG Scheduling][] is responsible for generic scheduler and scheduling components.

For the 1.8 release, SIG Scheduling extended the concept of cluster sharing by introducing
pod priority and pod preemption. These features allow mixing various types of workloads in a single cluster, and help reach
higher levels of resource utilization and availability.
These features are in alpha. SIG Scheduling also improved the internal APIs for scheduling and made them easier for other components and external schedulers to use.

[SIG Scheduling]: https://github.com/kubernetes/community/tree/master/sig-scheduling

### SIG Storage

[SIG Storage][] is responsible for storage and volume plugin components.

For the 1.8 release, SIG Storage extended the Kubernetes storage API. In addition to providing simple
volume availability, the API now enables volume resizing and snapshotting. These features are in alpha.
The SIG also focused on providing more control over storage: the ability to set requests and
limits on ephemeral storage, the ability to specify mount options, more metrics, and improvements to Flex driver deployments.

[SIG Storage]: https://github.com/kubernetes/community/tree/master/sig-storage

## Before Upgrading

Consider the following changes, limitations, and guidelines before you upgrade:

* The kubelet now fails if swap is enabled on a node. To override the default and run with /proc/swaps on, set `--fail-swap-on=false`. The experimental flag `--experimental-fail-swap-on` is deprecated in this release, and will be removed in a future release.

* The `autoscaling/v2alpha1` API is now at `autoscaling/v2beta1`. However, the form of the API remains unchanged. Migrate the `HorizontalPodAutoscaler` resources to `autoscaling/v2beta1` to persist the `HorizontalPodAutoscaler` changes introduced in `autoscaling/v2alpha1`. The Horizontal Pod Autoscaler changes include support for status conditions, and autoscaling on memory and custom metrics.

* The metrics APIs, `custom-metrics.metrics.k8s.io` and `metrics`, were moved from `v1alpha1` to `v1beta1`, and renamed to `custom.metrics.k8s.io` and `metrics.k8s.io`, respectively. If you have deployed a custom metrics adapter, ensure that it supports the new API version. If you have deployed Heapster in aggregated API server mode, upgrade Heapster to support the latest API version.

* Advanced auditing is the default auditing mechanism at `v1beta1`. The new version introduces the following changes:

  * The `--audit-policy-file` option is required if the `AdvancedAudit` feature is not explicitly turned off (`--feature-gates=AdvancedAudit=false`) on the API server.
  * The audit log file defaults to JSON encoding when using the advanced auditing feature gate.
  * The `--audit-policy-file` option requires `kind` and `apiVersion` fields specifying what format version the `Policy` is using.
  * The webhook and log file now output the `v1beta1` event format.

    For more details, see [Advanced audit](https://kubernetes.io/docs/tasks/debug-application-cluster/audit/#advanced-audit).

* The deprecated `ThirdPartyResource` (TPR) API was removed.
  To avoid losing your TPR data, [migrate to CustomResourceDefinition](https://kubernetes.io/docs/tasks/access-kubernetes-api/migrate-third-party-resource/).

* The following deprecated flags were removed from `kube-controller-manager`:

  * `replication-controller-lookup-cache-size`
  * `replicaset-lookup-cache-size`
  * `daemonset-lookup-cache-size`

  Don't use these flags. Using deprecated flags causes the server to print a warning. Using a removed flag causes the server to abort the startup.

* StatefulSet: The deprecated `pod.alpha.kubernetes.io/initialized` annotation for interrupting the StatefulSet Pod management is now ignored. StatefulSets with this annotation set to `true` or with no value will behave just as they did in previous versions. Dormant StatefulSets with the annotation set to `false` will become active after upgrading.

* The CronJob object is now enabled by default at `v1beta1`. CronJob `v2alpha1` is still available, but it must be explicitly enabled. We recommend that you move any current CronJob objects to `batch/v1beta1.CronJob`. Be aware that if you specify the deprecated version, you may encounter Resource Not Found errors. These errors occur because the new controllers look for the new version during a rolling update.

* The `batch/v2alpha1.ScheduledJob` was removed. Migrate to `batch/v1beta.CronJob` to continue managing time based jobs.

* The `rbac/v1alpha1`, `settings/v1alpha1`, and `scheduling/v1alpha1` APIs are disabled by default.

* The `system:node` role is no longer automatically granted to the `system:nodes` group in new clusters. The role gives broad read access to resources, including secrets and configmaps. Use the `Node` authorization mode to authorize the nodes in new clusters. To continue providing the `system:node` role to the members of the `system:nodes` group, create an installation-specific `ClusterRoleBinding` in the installation. ([[#49638](https://github.com/kubernetes/kubernetes/pull/49638)](https://github.com/kubernetes/kubernetes/pull/49638))

## Known Issues

This section contains a list of known issues reported in Kubernetes 1.8 release. The content is populated via [v1.8.x known issues and FAQ accumulator](https://github.com/kubernetes/kubernetes/issues/53004).

* A performance issue was identified in large-scale clusters when deleting thousands of pods simultaneously across hundreds of nodes. Kubelets in this scenario can encounter temporarily increased latency of `delete pod` API calls -- above the target service level objective of 1 second. If you run clusters with this usage pattern and if pod deletion latency could be an issue for you, you might want to wait until the issue is resolved before you upgrade. 

For more information and for updates on resolution of this issue, see [[#51899](https://github.com/kubernetes/kubernetes/pull/51899)](https://issue.k8s.io/51899).

* Audit logs might impact the API server performance and the latency of large request and response calls. The issue is observed under the following conditions: if `AdvancedAuditing` feature gate is enabled, which is the default case, if audit logging uses the log backend in JSON format, or if the audit policy records large API calls for requests or responses.

For more information, see [[#51899](https://github.com/kubernetes/kubernetes/pull/51899)](https://github.com/kubernetes/kubernetes/issues/51899).

* Minikube version 0.22.2 or lower does not work with kubectl version 1.8 or higher. This issue is caused by the presence of an unregistered type in the minikube API server. New versions of kubectl force validate the OpenAPI schema, which is not registered with all known types in the minikube API server.

For more information, see [#1996](https://github.com/kubernetes/minikube/issues/1996).

* The `ENABLE_APISERVER_BASIC_AUDIT` configuration parameter for GCE deployments is broken, but deprecated.

For more information, see [[#53154](https://github.com/kubernetes/kubernetes/pull/53154)](https://github.com/kubernetes/kubernetes/issues/53154).

* `kubectl set` commands placed on ReplicaSet and DaemonSet occasionally return version errors. All set commands, including set image, set env, set resources, and set serviceaccounts, are affected.

For more information, see [[#53040](https://github.com/kubernetes/kubernetes/pull/53040)](https://github.com/kubernetes/kubernetes/issues/53040).

* Object quotas are not consistently charged or updated. Specifically, the object count quota does not reliably account for uninitialized objects. Some quotas are charged only when an object is initialized. Others are charged when an object is created, whether it is initialized or not. We plan to fix this issue in a future release.

For more information, see [[#53109](https://github.com/kubernetes/kubernetes/pull/53109)](https://github.com/kubernetes/kubernetes/issues/53109).

## Deprecations

This section provides an overview of deprecated API versions, options, flags, and arguments. Deprecated means that we intend to remove the capability from a future release. After removal, the capability will no longer work. The sections are organized by SIGs.

### Apps

- The `.spec.rollbackTo` field of the Deployment kind is deprecated in `extensions/v1beta1`.

- The `kubernetes.io/created-by` annotation is deprecated and will be removed in version 1.9.
  Use [ControllerRef](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md) instead to determine which controller, if any, owns an object.

 - The `batch/v2alpha1.CronJob` is deprecated in favor of `batch/v1beta1`.

 - The `batch/v2alpha1.ScheduledJob` was removed. Use `batch/v1beta1.CronJob` instead.

### Auth

 - The RBAC v1alpha1 API group is deprecated in favor of RBAC v1.

 - The API server flag `--experimental-bootstrap-token-auth` is deprecated in favor of `--enable-bootstrap-token-auth`. The `--experimental-bootstrap-token-auth` flag will be removed in version 1.9.

### Autoscaling

 - Consuming metrics directly from Heapster is deprecated in favor of
   consuming metrics via an aggregated version of the resource metrics API.

   - In version 1.8, enable this behavior by setting the
     `--horizontal-pod-autoscaler-use-rest-clients` flag to `true`.

   - In version 1.9, this behavior will be enabled by default, and must be explicitly
     disabled by setting the `--horizontal-pod-autoscaler-use-rest-clients` flag to `false`.

### Cluster Lifecycle

- The `auto-detect` behavior of the kubelet's `--cloud-provider` flag is deprecated.

  - In version 1.8, the default value for the kubelet's `--cloud-provider` flag is `auto-detect`. Be aware that it works only on GCE, AWS and Azure.

  - In version 1.9, the default will be `""`, which means no built-in cloud provider extension will be enabled by default.

  - Enable an out-of-tree cloud provider with `--cloud-provider=external` in either version.

    For more information on deprecating auto-detecting cloud providers in kubelet, see [PR [#51312](https://github.com/kubernetes/kubernetes/pull/51312)](https://github.com/kubernetes/kubernetes/pull/51312) and [announcement](https://groups.google.com/forum/#!topic/kubernetes-dev/UAxwa2inbTA).

- The `PersistentVolumeLabel` admission controller in the API server is deprecated.

  - The replacement is running a cloud-specific controller-manager (often referred to as `cloud-controller-manager`) with the `PersistentVolumeLabel` controller enabled. This new controller loop operates as the `PersistentVolumeLabel` admission controller did in previous versions.

  - Do not use the `PersistentVolumeLabel` admission controller in the configuration files and scripts unless you are dependent on the in-tree GCE and AWS cloud providers.

  - The `PersistentVolumeLabel` admission controller will be removed in a future release, when the out-of-tree versions of the GCE and AWS cloud providers move to GA. The cloud providers are marked alpha in version 1.9.

### OpenStack

- The `openstack-heat` provider for `kube-up` is deprecated and will be removed
  in a future release. Refer to [Issue [#49213](https://github.com/kubernetes/kubernetes/pull/49213)](https://github.com/kubernetes/kubernetes/issues/49213)
  for background information.

### Scheduling

  - Opaque Integer Resources (OIRs) are deprecated and will be removed in
    version 1.9. Extended Resources (ERs) are a drop-in replacement for OIRs. You can use
    any domain name prefix outside of the `kubernetes.io/` domain instead of the
    `pod.alpha.kubernetes.io/opaque-int-resource-` prefix.

## Notable Features

### Workloads API (apps/v1beta2)

Kubernetes 1.8 adds the apps/v1beta2 group and version, which now consists of the
DaemonSet, Deployment, ReplicaSet and StatefulSet kinds. This group and version are part
of the Kubernetes Workloads API. We plan to move them to v1 in an upcoming release, so you might want to plan your migration accordingly.

For more information, see [the issue that describes this work in detail](https://github.com/kubernetes/features/issues/353)

#### API Object Additions and Migrations

- The DaemonSet, Deployment, ReplicaSet, and StatefulSet kinds
  are now in the apps/v1beta2 group and version.

- The apps/v1beta2 group version adds a Scale subresource for the StatefulSet
kind.

- All kinds in the apps/v1beta2 group version add a corresponding conditions
  kind.

#### Behavioral Changes

 - For all kinds in the API group version, a spec.selector default value is no longer
 available, because it's incompatible with `kubectl
 apply` and strategic merge patch. You must explicitly set the spec.selector value
 in your manifest. An object with a spec.selector value that does not match the labels in
 its spec.template is invalid.

 - Selector mutation is disabled for all kinds in the
 app/v1beta2 group version, because the controllers in the workloads API do not handle
 selector mutation in
 a consistent way. This restriction may be lifted in the future, but
 it is likely that that selectors will remain immutable after the move to v1.
 You can continue to use code that depends on mutable selectors by calling
 the apps/v1beta1 API in this release, but you should start planning for code
 that does not depend on mutable selectors.

 - Extended Resources are fully-qualified resource names outside the
 `kubernetes.io` domain. Extended Resource quantities must be integers.
 You can specify any resource name of the form `[aaa.]my-domain.bbb/ccc`
 in place of [Opaque Integer Resources](https://v1-6.docs.kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/#opaque-integer-resources-alpha-feature).
 Extended resources cannot be overcommitted, so make sure that request and limit are equal
 if both are present in a container spec.

 - The default Bootstrap Token created with `kubeadm init` v1.8 expires
 and is deleted after 24 hours by default to limit the exposure of the
 valuable credential. You can create a new Bootstrap Token with
 `kubeadm token create` or make the default token permanently valid by specifying
 `--token-ttl 0` to `kubeadm init`. The default token can later be deleted with
 `kubeadm token delete`.

 - `kubeadm join` now delegates TLS Bootstrapping to the kubelet itself, instead
 of reimplementing the process. `kubeadm join` writes the bootstrap kubeconfig
 file to `/etc/kubernetes/bootstrap-kubelet.conf`.

#### Defaults

 - The default spec.updateStrategy for the StatefulSet and DaemonSet kinds is
 RollingUpdate for the apps/v1beta2 group version. You can explicitly set
 the OnDelete strategy, and no strategy auto-conversion is applied to
 replace default values.

 - As mentioned in [Behavioral Changes](#behavioral-changes), selector
 defaults are disabled.

 - The default spec.revisionHistoryLimit for all applicable kinds in the
 apps/v1beta2 group version is 10.

 - In a CronJob object, the default spec.successfulJobsHistoryLimit is 3, and
 the default spec.failedJobsHistoryLimit is 1.

### Workloads API (batch)

- CronJob is now at `batch/v1beta1` ([[#41039](https://github.com/kubernetes/kubernetes/pull/41039)](https://github.com/kubernetes/kubernetes/issues/41039), [[@soltysh](https://github.com/soltysh)](https://github.com/soltysh)).

- `batch/v2alpha.CronJob` is deprecated in favor of `batch/v1beta` and will be removed in a future release.

- Job can now set a failure policy using `.spec.backoffLimit`. The default value for this new field is 6. ([[#30243](https://github.com/kubernetes/kubernetes/pull/30243)](https://github.com/kubernetes/kubernetes/issues/30243), [[@clamoriniere1A](https://github.com/clamoriniere1A)](https://github.com/clamoriniere1A)).

- `batch/v2alpha1.ScheduledJob` is removed.

- The Job controller now creates pods in batches instead of all at once. ([[#49142](https://github.com/kubernetes/kubernetes/pull/49142)](https://github.com/kubernetes/kubernetes/pull/49142), [[@joelsmith](https://github.com/joelsmith)](https://github.com/joelsmith)).

- Short `.spec.ActiveDeadlineSeconds` is properly applied to a Job. ([[#48545](https://github.com/kubernetes/kubernetes/pull/48545)]
(https://github.com/kubernetes/kubernetes/pull/48454), [[@weiwei4](https://github.com/weiwei4)](https://github.com/weiwei04)).


#### CLI Changes

- [alpha] `kubectl` plugins: `kubectl` now allows binary extensibility. You can extend the default set of `kubectl` commands by writing plugins
  that provide new subcommands. Refer to the documentation for more information.

- `kubectl rollout` and `rollback` now support StatefulSet.

- `kubectl scale` now uses the Scale subresource for kinds in the apps/v1beta2 group.

- `kubectl create configmap` and `kubectl create secret` subcommands now support
  the `--append-hash` flag, which enables unique but deterministic naming for
  objects generated from files, for example with `--from-file`.

- `kubectl run` can set a service account name in the generated pod
  spec with the `--serviceaccount` flag.

- `kubectl proxy` now correctly handles the `exec`, `attach`, and
  `portforward` commands.  You must pass `--disable-filter` to the command to allow these commands.

- Added `cronjobs.batch` to "all", so that `kubectl get all` returns them.

- Added flag `--include-uninitialized` to `kubectl annotate`, `apply`, `edit-last-applied`,
  `delete`, `describe`, `edit`, `get`, `label,` and `set`. `--include-uninitialized=true` makes
  kubectl commands apply to uninitialized objects, which by default are ignored
  if the names of the objects are not provided. `--all` also makes kubectl
  commands apply to uninitialized objects. See the
  [initializer documentation](https://kubernetes.io/docs/admin/extensible-admission-controllers/) for more details.

- Added RBAC reconcile commands with `kubectl auth reconcile -f FILE`. When
  passed a file which contains RBAC roles, rolebindings, clusterroles, or
  clusterrolebindings, this command computes covers and adds the missing rules.
  The logic required to properly apply RBAC permissions is more complicated
  than a JSON merge because you have to compute logical covers operations between
  rule sets. This means that we cannot use `kubectl apply` to update RBAC roles
  without risking breaking old clients, such as controllers.

- `kubectl delete` no longer scales down workload API objects before deletion.
  Users who depend on ordered termination for the Pods of their StatefulSets
  must use `kubectl scale` to scale down the StatefulSet before deletion.

- `kubectl run --env` no longer supports CSV parsing. To provide multiple environment
  variables, use the `--env` flag multiple times instead. Example: `--env ONE=1 --env TWO=2` instead of `--env ONE=1,TWO=2`.

- Removed deprecated command `kubectl stop`.

- Kubectl can now use http caching for the OpenAPI schema. The cache
  directory can be configured by passing the `--cache-dir` command line flag to kubectl.
  If set to an empty string, caching is disabled.

- Kubectl now performs validation against OpenAPI schema instead of Swagger 1.2. If
  OpenAPI is not available on the server, it falls back to the old Swagger 1.2.

- Added Italian translation for kubectl.

- Added German translation for kubectl.

#### Scheduling

* [alpha] This version now supports pod priority and creation of PriorityClasses ([user doc](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/))([design doc](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/pod-priority-api.md))

* [alpha] This version now supports priority-based preemption of pods ([user doc](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/))([design doc](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/pod-preemption.md))

* [alpha] Users can now add taints to nodes by condition ([design doc](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/taint-node-by-condition.md))

#### Storage

* [stable] Mount options

  * The ability to specify mount options for volumes is moved from beta to stable.

  * A new `MountOptions` field in the `PersistentVolume` spec is available to specify mount options. This field replaces an annotation.

  * A new `MountOptions` field in the `StorageClass` spec allows configuration of mount options for dynamically provisioned volumes.

* [stable] Support Attach and Detach operations for ReadWriteOnce (RWO) volumes that use iSCSI and Fibre Channel plugins.

* [stable] Expose storage usage metrics

  * The available capacity of a given Persistent Volume (PV) is available by calling the Kubernetes metrics API.

* [stable] Volume plugin metrics

  * Success and latency metrics for all Kubernetes calls are available by calling the Kubernetes metrics API. You can request volume operations, including mount, unmount, attach, detach, provision, and delete.

* [stable] The PV spec for Azure File, CephFS, iSCSI, and Glusterfs is modified to reference namespaced resources.

* [stable] You can now customize the iSCSI initiator name per volume in the iSCSI volume plugin.

* [stable] You can now specify the World Wide Identifier (WWID) for the volume identifier in the Fibre Channel volume plugin.

* [beta] Reclaim policy in StorageClass

  * You can now configure the reclaim policy in StorageClass, instead of defaulting to `delete` for dynamically provisioned volumes.

* [alpha] Volume resizing

  * You can now increase the size of a volume by calling the Kubernetes API.

  * For alpha, this feature increases only the size of the underlying volume. It does not support resizing the file system.

  * For alpha, volume resizing supports only Gluster volumes.

* [alpha] Provide capacity isolation and resource management for local ephemeral storage

  * You can now set container requests, container limits, and node allocatable reservations for the new `ephemeral-storage` resource.

  * The `ephemeral-storage` resource includes all the disk space a container might consume with container overlay or scratch.

* [alpha] Mount namespace propagation

  * The `VolumeMount.Propagation` field for `VolumeMount` in pod containers is now available.

  * You can now set `VolumeMount.Propagation` to `Bidirectional` to enable a particular mount for a container to propagate itself to the host or other containers.

* [alpha] Improve Flex volume deployment

  * Flex volume driver deployment is simplified in the following ways:

    * New driver files can now be automatically discovered and initialized without requiring a kubelet or controller-manager restart.

    * A sample DaemonSet to deploy Flexvolume drivers is now available.

* [prototype] Volume snapshots

  * You can now create a volume snapshot by calling the Kubernetes API.

  * Note that the prototype does not support quiescing before snapshot, so snapshots might be inconsistent.

  * In the prototype phase, this feature is external to the core Kubernetes. It's available at https://github.com/kubernetes-incubator/external-storage/tree/master/snapshot.

### Cluster Federation

#### [alpha] Federated Jobs

Federated Jobs that are automatically deployed to multiple clusters
are now supported. Cluster selection and weighting determine how Job
parallelism and completions are spread across clusters. Federated Job
status reflects the aggregate status across all underlying cluster
jobs.

#### [alpha] Federated Horizontal Pod Autoscaling (HPA)

Federated HPAs are similar to the traditional Kubernetes HPAs, except
that they span multiple clusters. Creating a Federated HPA targeting
multiple clusters ensures that cluster-level autoscalers are
consistently deployed across those clusters, and dynamically managed
to ensure that autoscaling can occur optimially in all clusters,
within a set of global constraints on the the total number of replicas
permitted across all clusters.  If replicas are not
required in some clusters due to low system load or insufficient quota
or capacity in those clusters, additional replicas are made available
to the autoscalers in other clusters if required.

### Node Components

#### Autoscaling and Metrics

* Support for custom metrics in the Horizontal Pod Autoscaler is now at v1beta1. The associated metrics APIs (custom metrics and resource/master metrics) were also moved to v1beta1. For more information, see [Before Upgrading](#before-upgrading).

* `metrics-server` is now the recommended way to provide the resource
  metrics API. Deploy `metrics-server` as an add-on in the same way that you deploy Heapster.

##### Cluster Autoscaler

* Cluster autoscaler is now GA
* Cluster support size is increased to 1000 nodes
* Respect graceful pod termination of up to 10 minutes
* Handle zone stock-outs and failures
* Improve monitoring and error reporting

#### Container Runtime Interface (CRI)

* [alpha] Add a CRI validation test suite and CRI command-line tools. ([#292](https://github.com/kubernetes/features/issues/292), [[@feiskyer](https://github.com/feiskyer)](https://github.com/feiskyer))

* [stable] [cri-o](https://github.com/kubernetes-incubator/cri-o): CRI implementation for OCI-based runtimes [[@mrunalp](https://github.com/mrunalp)]

  * Passed all the Kubernetes 1.7 end-to-end conformance test suites.
  * Verification against Kubernetes 1.8 is planned soon after the release.

* [stable] [frakti](https://github.com/kubernetes/frakti): CRI implementation for hypervisor-based runtimes is now v1.1. [[@feiskyer](https://github.com/feiskyer)]

  * Enhance CNI plugin compatibility, supports flannel, calico, weave and so on.
  * Pass all CRI validation conformance tests and node end-to-end conformance tests.
  * Add experimental Unikernel support.

* [alpha] [cri-containerd](https://github.com/kubernetes-incubator/cri-containerd): CRI implementation for containerd is now v1.0.0-alpha.0, [[@Random-Liu](https://github.com/Random-Liu)]

  * Feature complete. Support the full CRI API defined in v1.8.
  * Pass all the CRI validation tests and regular node end-to-end tests.
  * An ansible playbook is provided to configure a Kubernetes cri-containerd cluster with kubeadm.

* Add support in Kubelet to consume container metrics via CRI. [[@yguo0905](https://github.com/yguo0905)]
  * There are known bugs that result in errors when querying Kubelet's stats summary API. We expect to fix them in v1.8.1.

#### kubelet

* [alpha] Kubelet now supports alternative container-level CPU affinity policies by using the new CPU manager. ([#375](https://github.com/kubernetes/features/issues/375), [[@sjenning](https://github.com/sjenning)](https://github.com/sjenning), [[@ConnorDoyle](https://github.com/ConnorDoyle)](https://github.com/ConnorDoyle))

* [alpha] Applications may now request pre-allocated hugepages by using the new `hugepages` resource in the container resource requests. ([#275](https://github.com/kubernetes/features/issues/275), [[@derekwaynecarr](https://github.com/derekwaynecarr)](https://github.com/derekwaynecarr))

* [alpha] Add support for dynamic Kubelet configuration. ([#281](https://github.com/kubernetes/features/issues/281), [[@mtaufen](https://github.com/mtaufen)](https://github.com/mtaufen))

* [alpha] Add the Hardware Device Plugins API. ([#368](https://github.com/kubernetes/features/issues/368), [[@jiayingz](https://github.com/jiayingz)], [[@RenaudWasTaken](https://github.com/RenaudWasTaken)])

* [stable] Upgrade cAdvisor to v0.27.1 with the enhancement for node monitoring. [[@dashpole](https://github.com/dashpole)]

  * Fix journalctl leak
  * Fix container memory rss
  * Fix incorrect CPU usage with 4.7 kernel
  * OOM parser uses kmsg
  * Add hugepages support
  * Add CRI-O support

* Sharing a PID namespace between containers in a pod is disabled by default in version 1.8. To enable for a node, use the `--docker-disable-shared-pid=false` kubelet flag. Be aware that PID namespace sharing requires Docker version greater than or equal to 1.13.1.

* Fix issues related to the eviction manager.

* Fix inconsistent Prometheus cAdvisor metrics.

* Fix issues related to the local storage allocatable feature.

### Auth

* [GA] The RBAC API group has been promoted from v1beta1 to v1. No API changes were introduced.

* [beta] Advanced auditing has been promoted from alpha to beta. The webhook and logging policy formats have changed since alpha, and may require modification.

* [beta] Kubelet certificate rotation through the certificates API has been promoted from alpha to beta. RBAC cluster roles for the certificates controller have been added for common uses of the certificates API, such as the kubelet's.

* [beta] SelfSubjectRulesReview, an API that lets a user see what actions they can perform with a namespace, has been added to the authorization.k8s.io API group. This bulk query is intended to enable UIs to show/hide actions based on the end user, and for users to quickly reason about their own permissions.

* [alpha] Building on the 1.7 work to allow encryption of resources such as secrets, a mechanism to store resource encryption keys in external Key Management Systems (KMS) was introduced. This complements the original file-based storage and allows integration with multiple KMS. A Google Cloud KMS plugin was added and will be usable once the Google side of the integration is complete.

* Websocket requests may now authenticate to the API server by passing a bearer token in a websocket subprotocol of the form `base64url.bearer.authorization.k8s.io.<base64url-encoded-bearer-token>`. ([[#47740](https://github.com/kubernetes/kubernetes/pull/47740)](https://github.com/kubernetes/kubernetes/pull/47740) [[@liggitt](https://github.com/liggitt)](https://github.com/liggitt))

* Advanced audit now correctly reports impersonated user info. ([[#48184](https://github.com/kubernetes/kubernetes/pull/48184)], [[@CaoShuFeng](https://github.com/CaoShuFeng)](https://github.com/CaoShuFeng))

* Advanced audit policy now supports matching subresources and resource names, but the top level resource no longer matches the subresouce. For example "pods" no longer matches requests to the logs subresource of pods. Use "pods/logs" to match subresources. ([[#48836](https://github.com/kubernetes/kubernetes/pull/48836)](https://github.com/kubernetes/kubernetes/pull/48836), [[@ericchiang](https://github.com/ericchiang)](https://github.com/ericchiang))

* Previously a deleted service account or bootstrapping token secret would be considered valid until it was reaped. It is now invalid as soon as the `deletionTimestamp` is set. ([[#48343](https://github.com/kubernetes/kubernetes/pull/48343)](https://github.com/kubernetes/kubernetes/pull/48343), [[@deads2k](https://github.com/deads2k)](https://github.com/deads2k); [[#49057](https://github.com/kubernetes/kubernetes/pull/49057)](https://github.com/kubernetes/kubernetes/pull/49057), [[@ericchiang](https://github.com/ericchiang)](https://github.com/ericchiang))

* The `--insecure-allow-any-token` flag has been removed from the API server. Users of the flag should use impersonation headers instead for debugging. ([[#49045](https://github.com/kubernetes/kubernetes/pull/49045)](https://github.com/kubernetes/kubernetes/pull/49045), [[@ericchiang](https://github.com/ericchiang)](https://github.com/ericchiang))

* The NodeRestriction admission plugin now allows a node to evict pods bound to itself. ([[#48707](https://github.com/kubernetes/kubernetes/pull/48707)](https://github.com/kubernetes/kubernetes/pull/48707), [[@danielfm](https://github.com/danielfm)](https://github.com/danielfm))

* The OwnerReferencesPermissionEnforcement admission plugin now requires `update` permission on the `finalizers` subresource of the referenced owner in order to set `blockOwnerDeletion` on an owner reference. ([[#49133](https://github.com/kubernetes/kubernetes/pull/49133)](https://github.com/kubernetes/kubernetes/pull/49133), [[@deads2k](https://github.com/deads2k)](https://github.com/deads2k))

* The SubjectAccessReview API in the `authorization.k8s.io` API group now allows providing the user uid. ([[#49677](https://github.com/kubernetes/kubernetes/pull/49677)](https://github.com/kubernetes/kubernetes/pull/49677), [[@dims](https://github.com/dims)](https://github.com/dims))

* After a kubelet rotates its client cert, it now closes its connections to the API server to force a handshake using the new cert. Previously, the kubelet could keep its existing connection open, even if the cert used for that connection was expired and rejected by the API server. ([[#49899](https://github.com/kubernetes/kubernetes/pull/49899)](https://github.com/kubernetes/kubernetes/pull/49899), [[@ericchiang](https://github.com/ericchiang)](https://github.com/ericchiang))

* PodSecurityPolicies can now specify a whitelist of allowed paths for host volumes. ([[#50212](https://github.com/kubernetes/kubernetes/pull/50212)](https://github.com/kubernetes/kubernetes/pull/50212), [[@jhorwit2](https://github.com/jhorwit2)](https://github.com/jhorwit2))

* API server authentication now caches successful bearer token authentication results for a few seconds. ([[#50258](https://github.com/kubernetes/kubernetes/pull/50258)](https://github.com/kubernetes/kubernetes/pull/50258), [[@liggitt](https://github.com/liggitt)](https://github.com/liggitt))

* The OpenID Connect authenticator can now use a custom prefix, or omit the default prefix, for username and groups claims through the --oidc-username-prefix and --oidc-groups-prefix flags. For example, the authenticator can map a user with the username "jane" to "google:jane" by supplying the "google:" username prefix. ([[#50875](https://github.com/kubernetes/kubernetes/pull/50875)](https://github.com/kubernetes/kubernetes/pull/50875), [[@ericchiang](https://github.com/ericchiang)](https://github.com/ericchiang))

* The bootstrap token authenticator can now configure tokens with a set of extra groups in addition to `system:bootstrappers`. ([[#50933](https://github.com/kubernetes/kubernetes/pull/50933)](https://github.com/kubernetes/kubernetes/pull/50933), [[@mattmoyer](https://github.com/mattmoyer)](https://github.com/mattmoyer))

* Advanced audit allows logging failed login attempts.
 ([[#51119](https://github.com/kubernetes/kubernetes/pull/51119)](https://github.com/kubernetes/kubernetes/pull/51119), [[@soltysh](https://github.com/soltysh)](https://github.com/soltysh))

* A `kubectl auth reconcile` subcommand has been added for applying RBAC resources. When passed a file which contains RBAC roles, rolebindings, clusterroles, or clusterrolebindings, it will compute covers and add the missing rules. ([[#51636](https://github.com/kubernetes/kubernetes/pull/51636)](https://github.com/kubernetes/kubernetes/pull/51636), [[@deads2k](https://github.com/deads2k)](https://github.com/deads2k))

### Cluster Lifecycle

#### kubeadm

* [beta] A new `upgrade` subcommand allows you to automatically upgrade a self-hosted cluster created with kubeadm. ([#296](https://github.com/kubernetes/features/issues/296), [[@luxas](https://github.com/luxas)](https://github.com/luxas))

* [alpha] An experimental self-hosted cluster can now easily be created with `kubeadm init`. Enable the feature by setting the SelfHosting feature gate to true: `--feature-gates=SelfHosting=true` ([#296](https://github.com/kubernetes/features/issues/296), [[@luxas](https://github.com/luxas)](https://github.com/luxas))
   * **NOTE:** Self-hosting will be the default way to host the control plane in the next release, v1.9

* [alpha] A new `phase` subcommand supports performing only subtasks of the full `kubeadm init` flow. Combined with fine-grained configuration, kubeadm is now more easily consumable by higher-level provisioning tools like kops or GKE. ([#356](https://github.com/kubernetes/features/issues/356), [[@luxas](https://github.com/luxas)](https://github.com/luxas))
   * **NOTE:** This command is currently staged under `kubeadm alpha phase` and will be graduated to top level in a future release.

#### kops

* [alpha] Added support for targeting bare metal (or non-cloudprovider) machines. ([#360](https://github.com/kubernetes/features/issues/360), [[@justinsb](https://github.com/justinsb)](https://github.com/justinsb)).

* [alpha] kops now supports [running as a server](https://github.com/kubernetes/kops/blob/master/docs/api-server/README.md). ([#359](https://github.com/kubernetes/features/issues/359), [[@justinsb](https://github.com/justinsb)](https://github.com/justinsb)).

* [beta] GCE support is promoted from alpha to beta. ([#358](https://github.com/kubernetes/features/issues/358), [[@justinsb](https://github.com/justinsb)](https://github.com/justinsb)).

#### Cluster Discovery/Bootstrap

* [beta] The authentication and verification mechanism called Bootstrap Tokens is improved. Use Bootstrap Tokens to easily add new node identities to a cluster. ([#130](https://github.com/kubernetes/features/issues/130), [[@luxas](https://github.com/luxas)](https://github.com/luxas), [[@jbeda](https://github.com/jbeda)](https://github.com/jbeda)).

#### Multi-platform

* [alpha] The Conformance e2e test suite now passes on the arm, arm64, and ppc64le platforms. ([#288](https://github.com/kubernetes/features/issues/288), [[@luxas](https://github.com/luxas)](https://github.com/luxas), [[@mkumatag](https://github.com/mkumatag)](https://github.com/mkumatag), [[@ixdy](https://github.com/ixdy)](https://github.com/ixdy))

#### Cloud Providers

* [alpha] Support is improved for the pluggable, out-of-tree and out-of-core cloud providers. ([#88](https://github.com/kubernetes/features/issues/88), [[@wlan0](https://github.com/wlan0)](https://github.com/wlan0))

### Network

#### network-policy

* [beta] Apply NetworkPolicy based on CIDR ([[#50033](https://github.com/kubernetes/kubernetes/pull/50033)](https://github.com/kubernetes/kubernetes/pull/50033), [[@cmluciano](https://github.com/cmluciano)](https://github.com/cmluciano))

* [beta] Support EgressRules in NetworkPolicy ([[#51351](https://github.com/kubernetes/kubernetes/pull/51351)](https://github.com/kubernetes/kubernetes/pull/51351), [[@cmluciano](https://github.com/cmluciano)](https://github.com/cmluciano))

#### kube-proxy ipvs mode

[alpha] Support ipvs mode for kube-proxy([[#46580](https://github.com/kubernetes/kubernetes/pull/46580)](https://github.com/kubernetes/kubernetes/pull/46580), [[@haibinxie](https://github.com/haibinxie)](https://github.com/haibinxie))

### API Machinery

#### kube-apiserver
* Fixed an issue with `APIService` auto-registration. This issue affected rolling restarts of HA API servers that added or removed API groups being served.([[#51921](https://github.com/kubernetes/kubernetes/pull/51921)](https://github.com/kubernetes/kubernetes/pull/51921))

* [Alpha] The Kubernetes API server now supports the ability to break large LIST calls into multiple smaller chunks. A client can specify a limit to the number of results to return. If more results exist, a token is returned that allows the client to continue the previous list call repeatedly until all results are retrieved.  The resulting list is identical to a list call that does not perform chunking, thanks to capabilities provided by etcd3.  This allows the server to use less memory and CPU when very large lists are returned. This feature is gated as APIListChunking and is not enabled by default. The 1.9 release will begin using this by default.([[#48921](https://github.com/kubernetes/kubernetes/pull/48921)](https://github.com/kubernetes/kubernetes/pull/48921))

* Pods that are marked for deletion and have exceeded their grace period, but are not yet deleted, no longer count toward the resource quota.([[#46542](https://github.com/kubernetes/kubernetes/pull/46542)](https://github.com/kubernetes/kubernetes/pull/46542))


#### Dynamic Admission Control

* Pod spec is mutable when the pod is uninitialized. The API server requires the pod spec to be valid even if it's uninitialized. Updating the status field of uninitialized pods is invalid.([[#51733](https://github.com/kubernetes/kubernetes/pull/51733)](https://github.com/kubernetes/kubernetes/pull/51733))

* Use of the alpha initializers feature now requires enabling the `Initializers` feature gate. This feature gate is automatically enabled if the `Initializers` admission plugin is enabled.([[#51436](https://github.com/kubernetes/kubernetes/pull/51436)](https://github.com/kubernetes/kubernetes/pull/51436))

* [Action required] The validation rule for metadata.initializers.pending[x].name is tightened. The initializer name must contain at least three segments, separated by dots. You can create objects with pending initializers and not rely on the API server to add pending initializers according to `initializerConfiguration`. If you do so, update the initializer name in the existing objects and the configuration files to comply with the new validation rule.([[#51283](https://github.com/kubernetes/kubernetes/pull/51283)](https://github.com/kubernetes/kubernetes/pull/51283))

* The webhook admission plugin now works even if the API server and the nodes are in two separate networks,for example, in GKE.
The webhook admission plugin now lets the webhook author use the DNS name of the service as the CommonName when generating the server cert for the webhook.
Action required:
Regenerate the server cert for the admission webhooks. Previously, the CN value could be ignored while generating the server cert for the admission webhook. Now you must set it to the DNS name of the webhook service: `<service.Name>.<service.Namespace>.svc`.([[#50476](https://github.com/kubernetes/kubernetes/pull/50476)](https://github.com/kubernetes/kubernetes/pull/50476))


#### Custom Resource Definitions (CRDs)
* [alpha] The CustomResourceDefinition API can now optionally
  [validate custom objects](https://kubernetes.io/docs/tasks/access-kubernetes-api/extend-api-custom-resource-definitions/#validation)
  based on a JSON schema provided in the CRD spec.
  Enable this alpha feature with the `CustomResourceValidation` feature gate in `kube-apiserver`.

#### Garbage Collector
* The garbage collector now supports custom APIs added via Custom Resource Definitions
  or aggregated API servers. The garbage collector controller refreshes periodically.
  Therefore, expect a latency of about 30 seconds between when an API is added and when
  the garbage collector starts to manage it.


#### Monitoring/Prometheus
* [action required] The WATCHLIST calls are now reported as WATCH verbs in prometheus for the apiserver_request_* series.  A new "scope" label is added to all apiserver_request_* values that is either 'cluster', 'resource', or 'namespace' depending on which level the query is performed at.([[#52237](https://github.com/kubernetes/kubernetes/pull/52237)](https://github.com/kubernetes/kubernetes/pull/52237))


#### Go Client
* Add support for client-side spam filtering of events([[#47367](https://github.com/kubernetes/kubernetes/pull/47367)](https://github.com/kubernetes/kubernetes/pull/47367))


## External Dependencies

Continuous integration builds use Docker versions 1.11.2, 1.12.6, 1.13.1,
and 17.03.2. These versions were validated on Kubernetes 1.8. However,
consult an appropriate installation or upgrade guide before deciding what
versions of Docker to use.

- Docker 1.13.1 and 17.03.2

    - Shared PID namespace, live-restore, and overlay2 were validated.

    - **Known issues**

        - The default iptables FORWARD policy was changed from ACCEPT to
          DROP, which causes outbound container traffic to stop working by
          default. See
          [[#40182](https://github.com/kubernetes/kubernetes/pull/40182)](https://github.com/kubernetes/kubernetes/issues/40182) for
          the workaround.

        - The support for the v1 registries was removed.

- Docker 1.12.6

    - Overlay2 and live-restore are not validated.

    - **Known issues**

        - Shared PID namespace does not work properly.
          ([#207](https://github.com/kubernetes/community/pull/207#issuecomment-281870043))

        - Docker reports incorrect exit codes for containers.
          ([[#41516](https://github.com/kubernetes/kubernetes/pull/41516)](https://github.com/kubernetes/kubernetes/issues/41516))

- Docker 1.11.2

    - **Known issues**

        - Kernel crash with Aufs storage driver on Debian Jessie
          ([[#27885](https://github.com/kubernetes/kubernetes/pull/27885)](https://github.com/kubernetes/kubernetes/issues/27885)).
          The issue can be identified by using the node problem detector.

        - File descriptor leak on init/control.
          ([#275](https://github.com/containerd/containerd/issues/275))

        - Additional memory overhead per container.
          ([[#21737](https://github.com/kubernetes/kubernetes/pull/21737)](https://github.com/kubernetes/kubernetes/pull/21737))

        - Processes may be leaked when Docker is repeatedly terminated in a short
          time frame.
          ([[#41450](https://github.com/kubernetes/kubernetes/pull/41450)](https://github.com/kubernetes/kubernetes/issues/41450))
- [v1.8.0-rc.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v180-rc1)
- [v1.8.0-beta.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v180-beta1)
- [v1.8.0-alpha.3](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v180-alpha3)
- [v1.8.0-alpha.2](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v180-alpha2)
- [v1.8.0-alpha.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v180-alpha1)



# v1.8.0-rc.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.8/examples)

## Downloads for v1.8.0-rc.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes.tar.gz) | `122d3ca2addb168c68e65a515bc42c21ad6c4bc71cb71591c699ba035765994b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-src.tar.gz) | `8903266d4379f03059fcd9398d3bcd526b979b3c4b49e75aa13cce38de6f4e91`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-darwin-386.tar.gz) | `c4cae289498888db775abd833de1544adc632913e46879c6f770c931f29e5d3f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | `dd7081fc40e71be45cbae1bcd0f0932e5578d662a8054faea96c65efd1c1a134`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-linux-386.tar.gz) | `728265635b18db308a69ed249dc4a59658aa6db7d23bb3953923f1e54c2d620f`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | `2053862e5461f213c03381a9a05d70c0a28fdaf959600244257f6636ba92d058`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | `956cc6b5afb816734ff439e09323e56210293089858cb6deea92b8d3318463ac`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-linux-arm.tar.gz) | `eca7ff447699849db3ae2d76ac9dad07be86c2ebe652ef774639bf006499ddbc`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | `0d593c8034e54a683a629c034677b1c402e3e9516afcf174602e953818c8f0d1`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | `9a7c1771d710d78f4bf45ff41f1a312393a8ee8b0506ee09aeca449c3552a147`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-windows-386.tar.gz) | `345f50766c627e45f35705b72fb2f56e78cc824af123cf14f5a84954ac1d6c93`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | `ea98c0872fa6df3eb5af6f1a005c014a76a0b4b0af9a09fdf90d3bc6a7ee5725`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | `58799a02896f7d3c160088832186c8e9499c433e65f0f1eeb37763475e176473`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | `cd28ad39762cdf144e5191d4745e2629f37851265dfe44930295a0b13823a998`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-server-linux-arm.tar.gz) | `62c75adbe94d17163cbc3e8cbc4091fdb6f5e9dc90a8471aac3f9ee38e609d82`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | `cee4a1a33dec4facebfa6a1f7780aba6fee91f645fbc0a388e64e93c0d660b17`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | `b263139b615372dd95ce404b8e94a51594e7d22b25297934cb189d3c9078b4fb`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | `8d6688a224ebc5814e142d296072b7d2352fe86bd338539bd89ac9883b338c3d`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | `1db47ce33af16d8974028017738a6d211a3c7c0b6f0046f70ccc52875ef6cbdf`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-node-linux-arm.tar.gz) | `ec5eaddbb2a338ab16d801685f98746c465aad45b3d1dcf4dcfc361bd18eb124`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | `0ebf091215a2fcf1232a7f0eedf21f3786bbddae26619cf77098ab88670e1935`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | `7443c6753cc6e4d9591064ad87c619b27713f6bb45aef5dcabb1046c2d19f0f3`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | `b4bbb5521930fdc71dac52608c145e1e22cc2ab5e823492cbf1b7a46c317654a`

## Changelog since v1.8.0-beta.1

### Action Required

* New GCE or GKE clusters created with `cluster/kube-up.sh` will not enable the legacy ABAC authorizer by default. If you would like to enable the legacy ABAC authorizer, export ENABLE_LEGACY_ABAC=true before running `cluster/kube-up.sh`. ([#51367](https://github.com/kubernetes/kubernetes/pull/51367), [@cjcullen](https://github.com/cjcullen))

### Other notable changes

* kubeadm: Use the release-1.8 branch by default ([#52085](https://github.com/kubernetes/kubernetes/pull/52085), [@luxas](https://github.com/luxas))
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
* Add `retainKeys` to patchStrategy for v1 Volumes and extentions/v1beta1 DeploymentStrategy. ([#50296](https://github.com/kubernetes/kubernetes/pull/50296), [@mengqiy](https://github.com/mengqiy))
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



# v1.7.6

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.6


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes.tar.gz) | `6d2462aed79097845129e05375fdf16b724c32d47579d30a9b563a8d360d3ae3`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-src.tar.gz) | `ee66724a04900f4b90bc6eccbd6487095d888a90cf7cfdc0f5b5e9425ae95e47`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-darwin-386.tar.gz) | `fc5ee8d608cc551693839ac79c1330b7a688930a8f16b0d313128844d598e4d3`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-darwin-amd64.tar.gz) | `0e9dad45f6dd4ef06d9aef7151ba02612300ddebf7fb4b7e64174408590e340e`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-linux-386.tar.gz) | `74fc57544bd2b109fb620f0f8f1e821a66e83082700a49cfc38e5b2c1d7221a6`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-linux-amd64.tar.gz) | `0d46a9c297d193bc193487aa1734141be764a0078759748ec800f92bd183de5f`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-linux-arm64.tar.gz) | `ef9dbbd93e4ad02e02297466b631e779f5fd96f2a449a5f628b239068e615a22`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-linux-arm.tar.gz) | `25637797aed9d4904e8209d5085ade93df12a9fbcf6c09499e3a20cba6876122`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-linux-ppc64le.tar.gz) | `9a9cc9e747fd56330c87b68508c9cb6cedbe988a7682e70f6410a0d1c6bc9256`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-linux-s390x.tar.gz) | `8cdaaf06618b5e936ad90bdae608ea0e9f352b91197002031b3802fbdeda6aa3`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-windows-386.tar.gz) | `e1e74224d151d0317eba54ac02bdac21e86416af475b27a068e9f72749b10481`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-client-windows-amd64.tar.gz) | `37d9a7c0fbf3ff1e47d51a986f939c4f257bf265916c5f1b2e809b8161f48953`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-server-linux-amd64.tar.gz) | `302c3c48f9c2def14fd4503f5caf3c66e8abefd478e735ec7a270b3ba313f93c`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-server-linux-arm64.tar.gz) | `04a28285cc98e57dee3d41987adb4e08e049b9c0d493ed3ae1b7017c2d4aaa66`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-server-linux-arm.tar.gz) | `caf808442d09784dea5b18d89a39cbfe318257bd5efa03ab81b4393a5aa3e370`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-server-linux-ppc64le.tar.gz) | `b156c17df4a4c2badd1c7e580652ffe6d816c1134ebb22e1ca1fa7ef1b8326df`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-server-linux-s390x.tar.gz) | `1a4fedd1ec94429b5ea8ef894b04940e248f872fab272f28fddff5951e4ee571`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-node-linux-amd64.tar.gz) | `8d798ef84c933c9aa4ba144277ebe571879b2237239827565327be2c97726bbc`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-node-linux-arm64.tar.gz) | `ca0976faf03812a415da6a0dc244a65222a3f8d81b3da929530988a36ce0dc1a`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-node-linux-arm.tar.gz) | `92fd22d0bb51d32e24490a0ec12c48e28b5c5a19826c10f5e9061d06620ca12f`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-node-linux-ppc64le.tar.gz) | `1b39b2a89a5522a9f1d23b90a51070a13bede72a66c3b6b217289fa4fadbc0d6`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-node-linux-s390x.tar.gz) | `fda8c1ed4ebd406a6c19d0a982ba6705f0533e6c1db96e2bd121392deb4018ed`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.6/kubernetes-node-windows-amd64.tar.gz) | `325caebf0f5d9dc79259f9609014e80385753d3ad1fff7fb276b19d2f272ef3b`

## Changelog since v1.7.5

### Other notable changes

* [fluentd-gcp addon] Fluentd will trim lines exceeding 100KB instead of dropping them. ([#52289](https://github.com/kubernetes/kubernetes/pull/52289), [@crassirostris](https://github.com/crassirostris))
* Cluster Autoscaler 0.6.2 ([#52359](https://github.com/kubernetes/kubernetes/pull/52359), [@mwielgus](https://github.com/mwielgus))
* Add --request-timeout to kube-apiserver to make global request timeout configurable. ([#51415](https://github.com/kubernetes/kubernetes/pull/51415), [@jpbetz](https://github.com/jpbetz))
* Fix credentials providers for docker sandbox image. ([#51870](https://github.com/kubernetes/kubernetes/pull/51870), [@feiskyer](https://github.com/feiskyer))
* Fix security holes in GCE metadata proxy. ([#51302](https://github.com/kubernetes/kubernetes/pull/51302), [@ihmccreery](https://github.com/ihmccreery))
* Fixed an issue looking up cronjobs when they existed in more than one API version ([#52227](https://github.com/kubernetes/kubernetes/pull/52227), [@liggitt](https://github.com/liggitt))
* Fixes an issue with upgrade requests made via pod/service/node proxy subresources sending a non-absolute HTTP request-uri to backends ([#52065](https://github.com/kubernetes/kubernetes/pull/52065), [@liggitt](https://github.com/liggitt))
* Fix a kube-controller-manager crash which can result when `--concurrent-resource-quota-syncs` is >1 and pods exist in the system containing certain alpha/beta annotation keys. ([#52092](https://github.com/kubernetes/kubernetes/pull/52092), [@ironcladlou](https://github.com/ironcladlou))
* Make logdump support kubemark and support gke with 'use_custom_instance_list' ([#51834](https://github.com/kubernetes/kubernetes/pull/51834), [@shyamjvs](https://github.com/shyamjvs))
* Fixes an issue with APIService auto-registration affecting rolling HA apiserver restarts that add or remove API groups being served. ([#51921](https://github.com/kubernetes/kubernetes/pull/51921), [@liggitt](https://github.com/liggitt))
* In GCE with COS, increase TasksMax for Docker service to raise cap on number of threads/processes used by containers. ([#51986](https://github.com/kubernetes/kubernetes/pull/51986), [@yujuhong](https://github.com/yujuhong))
* Fix providerID update validation ([#51761](https://github.com/kubernetes/kubernetes/pull/51761), [@karataliu](https://github.com/karataliu))
* Automated cherry pick of [#50381](https://github.com/kubernetes/kubernetes/pull/50381) to release-1.7 ([#51871](https://github.com/kubernetes/kubernetes/pull/51871), [@feiskyer](https://github.com/feiskyer))
* The `emptyDir.sizeLimit` field is now correctly omitted from API requests and responses when unset. ([#50163](https://github.com/kubernetes/kubernetes/pull/50163), [@jingxu97](https://github.com/jingxu97))
* Calico has been updated to v2.5, RBAC added, and is now automatically scaled when GCE clusters are resized. ([#51237](https://github.com/kubernetes/kubernetes/pull/51237), [@gunjan5](https://github.com/gunjan5))



# v1.8.0-beta.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.8/examples)

## Downloads for v1.8.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes.tar.gz) | `261e5ad47a718bcbb65c163f8e1130097e2d077541d6a68f3270de4e7256d796`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-src.tar.gz) | `e414e75cd1c72ca1fd202f6f0042ba1884b87bc6809bc2493ea2654c3d965656`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `b7745121e8d7074170f1ce8ded0fbc78b84abe8f8371933e97b76ba5551f26d8`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `4cc45a3a5dbd2ca666ea6dc2a973a17929c1427f5c3296224eade50d8df10b9e`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-linux-386.tar.gz) | `a1dce30675b33e2c18a1343ee15556c9c65e85ee3c2b88f3cac414d95514a902`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `7fa5bbdc4af80a7ce00c5939896e8e93e962a66d195a95878f1e1fe4a06a5272`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `7d54528f892d3247e22093861c48407e7dc001304bb168cf8c882227d96fd6b2`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `17c074ae407b012b4bb2c88975c182df0317fefea98700fdadee12c70d114498`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `074801a87eedd2e93bdeb894822a70aa371983aafce86f66ed473a1a3bf4628b`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `2eb743f160b970a183b3ec81fc50108df2352b8a0c31951babb26e2c28fc8360`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-windows-386.tar.gz) | `21e5686253052773d7e4baa08fd4ce56c861ad01d49d87df0eb80f56801e7cc4`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `07d2446c917cf749b38fa2bcaa2bd64af743df2ba19ad4b480c07be166f9ab16`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `811eb1645f8691e5cf7f75ae8ab26e90cf0b36a69254f73c0ed4ba91f4c0db99`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `e05c53ce80354d2776aa6832e074730aa35521f64ebf03a6c5a7753e7f2df8a3`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `57bc90e040faefa6af23b8637e8221a06282041ec9a16c2a630cc655d3c170df`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `4feb30aef4f79954907fdec34d4b7d2985917abd8e35b34a9440a468889cb240`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `85c0aaff6e832f711fb572582f10d9fe172c4d0680ac7589d1ec6e54742c436c`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `5809dce1c13d05c7c85bddc4d31804b30c55fe70209c9d89b137598c25db863e`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `d70c9d99f4b155b755728029036f68d79cff1648cfd3de257e3f2ce29bc07a31`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `efa29832aea28817466e25b55375574f314848c806d76fa0e4874f835399e9f0`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `991507d4cd2014e776d63ae7a14b3bbbbf49597211d4fa1751701f21fbf44417`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `4e1bd8e4465b2761632093a1235b788cc649af74d42dec297a97de8a0f764e46`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `4f80d4c269c6f05fb30c8c682f1cdbe46f3f0e86ac7ca4b84a1ab0a835bfb24a`

## Changelog since v1.8.0-alpha.3

### Action Required

* The OwnerReferencesPermissionEnforcement admission plugin now requires `update` permission on the `finalizers` subresource of the referenced owner in order to set `blockOwnerDeletion` on an owner reference. ([#49133](https://github.com/kubernetes/kubernetes/pull/49133), [@deads2k](https://github.com/deads2k))
* The deprecated alpha and beta initContainer annotations are no longer supported. Init containers must be specified using the initContainers field in the pod spec. ([#51816](https://github.com/kubernetes/kubernetes/pull/51816), [@liggitt](https://github.com/liggitt))
* Action required: validation rule on metadata.initializers.pending[x].name is tightened. The initializer name needs to contain at least three segments separated by dots. If you create objects with pending initializers, (i.e., not relying on apiserver adding pending initializers according to initializerconfiguration), you need to update the initializer name in existing objects and in configuration files to comply to the new validation rule. ([#51283](https://github.com/kubernetes/kubernetes/pull/51283), [@caesarxuchao](https://github.com/caesarxuchao))
* Audit policy supports matching subresources and resource names, but the top level resource no longer matches the subresouce. For example "pods" no longer matches requests to the logs subresource of pods. Use "pods/logs" to match subresources. ([#48836](https://github.com/kubernetes/kubernetes/pull/48836), [@ericchiang](https://github.com/ericchiang))
* Protobuf serialization does not distinguish between `[]` and `null`. ([#45294](https://github.com/kubernetes/kubernetes/pull/45294), [@liggitt](https://github.com/liggitt))
    * API fields previously capable of storing and returning either `[]` and `null` via JSON API requests (for example, the Endpoints `subsets` field) can now store only `null` when created using the protobuf content-type or stored in etcd using protobuf serialization (the default in 1.6+). JSON API clients should tolerate `null` values for such fields, and treat `null` and `[]` as equivalent in meaning unless specifically documented otherwise for a particular field.

### Other notable changes

* Fixes an issue with upgrade requests made via pod/service/node proxy subresources sending a non-absolute HTTP request-uri to backends ([#52065](https://github.com/kubernetes/kubernetes/pull/52065), [@liggitt](https://github.com/liggitt))
* kubeadm: add `kubeadm phase addons` command ([#51171](https://github.com/kubernetes/kubernetes/pull/51171), [@andrewrynhard](https://github.com/andrewrynhard))
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
* Alpha list paging implementation ([#48921](https://github.com/kubernetes/kubernetes/pull/48921), [@smarterclayton](https://github.com/smarterclayton))
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
* [#43077](https://github.com/kubernetes/kubernetes/pull/43077) introduced a condition where DaemonSet controller did not respect the TerminationGracePeriodSeconds of the Pods it created. This is now corrected. ([#51279](https://github.com/kubernetes/kubernetes/pull/51279), [@kow3ns](https://github.com/kow3ns))
* Add a persistent volume label controller to the cloud-controller-manager ([#44680](https://github.com/kubernetes/kubernetes/pull/44680), [@rrati](https://github.com/rrati))
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
* Implement IPVS-based in-cluster service load balancing ([#46580](https://github.com/kubernetes/kubernetes/pull/46580), [@dujun1990](https://github.com/dujun1990))
* Release the kubelet client certificate rotation as beta. ([#51045](https://github.com/kubernetes/kubernetes/pull/51045), [@jcbsmpsn](https://github.com/jcbsmpsn))
* Adds --append-hash flag to kubectl create configmap/secret, which will append a short hash of the configmap/secret contents to the name during creation. ([#49961](https://github.com/kubernetes/kubernetes/pull/49961), [@mtaufen](https://github.com/mtaufen))
* Add validation for CustomResources via JSON Schema. ([#47263](https://github.com/kubernetes/kubernetes/pull/47263), [@nikhita](https://github.com/nikhita))
* enqueue a sync task to wake up jobcontroller to check job ActiveDeadlineSeconds in time ([#48454](https://github.com/kubernetes/kubernetes/pull/48454), [@weiwei04](https://github.com/weiwei04))
* Remove previous local ephemeral storage resource names: "ResourceStorageOverlay" and "ResourceStorageScratch" ([#51425](https://github.com/kubernetes/kubernetes/pull/51425), [@NickrenREN](https://github.com/NickrenREN))
* Add `retainKeys` to patchStrategy for v1 Volumes and extentions/v1beta1 DeploymentStrategy. ([#50296](https://github.com/kubernetes/kubernetes/pull/50296), [@mengqiy](https://github.com/mengqiy))
* Add mount options field to PersistentVolume spec ([#50919](https://github.com/kubernetes/kubernetes/pull/50919), [@wongma7](https://github.com/wongma7))
* Use of the alpha initializers feature now requires enabling the `Initializers` feature gate. This feature gate is auto-enabled if the `Initialzers` admission plugin is enabled. ([#51436](https://github.com/kubernetes/kubernetes/pull/51436), [@liggitt](https://github.com/liggitt))
* Fix inconsistent Prometheus cAdvisor metrics ([#51473](https://github.com/kubernetes/kubernetes/pull/51473), [@bboreham](https://github.com/bboreham))
* Add local ephemeral storage to downward API  ([#50435](https://github.com/kubernetes/kubernetes/pull/50435), [@NickrenREN](https://github.com/NickrenREN))
* kubectl zsh autocompletion will work with compinit ([#50561](https://github.com/kubernetes/kubernetes/pull/50561), [@cblecker](https://github.com/cblecker))
* When using kube-up.sh on GCE, user could set env `KUBE_PROXY_DAEMONSET=true` to run kube-proxy as a DaemonSet. kube-proxy is run as static pods by default. ([#50705](https://github.com/kubernetes/kubernetes/pull/50705), [@MrHohn](https://github.com/MrHohn))
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



# v1.7.5

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.5


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes.tar.gz) | `bc96c1ec02da6a82f90bc04064d2c4d6463a4d9dd37e5882a23f8c74bdf1b20b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-src.tar.gz) | `e06ebc6b73b6b38aeb55891b9e5c0bbd26e755e05674d70866cdc41f749f62a5`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-darwin-386.tar.gz) | `2c1c40c161e5ccae6df0dc5846a9a8bd55ebcd5b55012e09c01ec00bc81f4a81`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-darwin-amd64.tar.gz) | `6e749df53f9b4f5e2c1a94c360e06e9d4c4c0bf34c0dd2a02476d476e8da3f68`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-linux-386.tar.gz) | `d0edb7229ec27c4354589a1045766d8e12605be5c2ab82cef3e30d324ba66095`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-linux-amd64.tar.gz) | `e246dc357be1ccaad1c5f79d4696abdc31a90bd8eae642e5bacd1e7d820517ad`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-linux-arm64.tar.gz) | `bf94c70e00cb3c451a3b024e64fd5933098850fe3414e8b72d42244cbd478a2e`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-linux-arm.tar.gz) | `17d4af2b552377ee580230c0f0ea0de8469e682c01cd0ebde8f50c52cd02bed3`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-linux-ppc64le.tar.gz) | `bfa32c4b1d70474dd5fccd588bd4e836c6d330b1d6d04de3ceeb3acc4f65a21b`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-linux-s390x.tar.gz) | `c2a3822d358b24c909b8965a25ac759f510bab3f60b6117cf522dccabc724cb0`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-windows-386.tar.gz) | `b70b3de5a33eb7762aa371b1b7e426a0cafc1d468bb33dff2db20997d244bd37`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-client-windows-amd64.tar.gz) | `7f995b5a4f9338b9aa62508ac71ccd615f0ef577841d603f9e9ea6683be688b0`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-server-linux-amd64.tar.gz) | `7482c12dae75fb195f2f3afa92f62c354cafb97bee5703c4fdaa617d27c7cf68`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-server-linux-arm64.tar.gz) | `0be475479062f113fcc41d91215c21409c6e4c000e96ffc0246e4597b6737a29`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-server-linux-arm.tar.gz) | `07527fbe49a2f12eae25ccd49e8a95deae7f5a8c8bae2014e5dc2561e4a04fdb`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-server-linux-ppc64le.tar.gz) | `fed7ee43ba5db918d277e26da9ca556254fa365445d51cb33a3e304d1e3841e9`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-server-linux-s390x.tar.gz) | `47b548cc2c6e224c49fe286da3db61c0cf1905239df2869b88b9b8607edbbd73`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-node-linux-amd64.tar.gz) | `f5dd62f21d2cc516768b55d191bc20fc20901b9fa2e1165eef2adcca4821e23d`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-node-linux-arm64.tar.gz) | `8ee0d5f417651f2ce9ab5e504bbd47fbfe0f15d6e3923a1356b2def4f1012b66`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-node-linux-arm.tar.gz) | `40882a5c505fee370eb69e890b8974d3bb9c896307795d81bf7dff52797e4eeb`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-node-linux-ppc64le.tar.gz) | `597bd33af9f03874fabc0778de3df057f13364630d590cc4443e4c858ffbe7f3`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-node-linux-s390x.tar.gz) | `dd57a82a5d71d03a97cebf901bf9cc5273b935218f4fc1c3f1471b93842a4414`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.5/kubernetes-node-windows-amd64.tar.gz) | `d95511742d26c375b5a705b85b498b200c8e081fec365c4b60df18def49d151c`

## Changelog since v1.7.4

### Other notable changes

* Bumped Heapster version to 1.4.2 - more details https://github.com/kubernetes/heapster/releases/tag/v1.4.2. ([#51620](https://github.com/kubernetes/kubernetes/pull/51620), [@piosz](https://github.com/piosz))
* Fix for Pod stuck in ContainerCreating with error "Volume is not yet attached according to node". ([#50806](https://github.com/kubernetes/kubernetes/pull/50806), [@verult](https://github.com/verult))
* Fixed controller manager crash by making it tolerant to discovery errors.([#49767](https://github.com/kubernetes/kubernetes/pull/49767), [@deads2k](https://github.com/deads2k))
* Finalizers are now honored on custom resources, and on other resources even when garbage collection is disabled via the apiserver flag `--enable-garbage-collector=false` ([#51469](https://github.com/kubernetes/kubernetes/pull/51469), [@ironcladlou](https://github.com/ironcladlou))
* Allow attach of volumes to multiple nodes for vSphere ([#51066](https://github.com/kubernetes/kubernetes/pull/51066), [@BaluDontu](https://github.com/BaluDontu))
* vSphere: Fix attach volume failing on the first try. ([#51217](https://github.com/kubernetes/kubernetes/pull/51217), [@BaluDontu](https://github.com/BaluDontu))
* azure: support retrieving access tokens via managed identity extension ([#48854](https://github.com/kubernetes/kubernetes/pull/48854), [@colemickens](https://github.com/colemickens))
* Fixed a bug in strategic merge patch that caused kubectl apply to error out under some conditions ([#50862](https://github.com/kubernetes/kubernetes/pull/50862), [@guoshimin](https://github.com/guoshimin))
* It is now posible to use flexVolumes to bind mount directories and files. ([#50596](https://github.com/kubernetes/kubernetes/pull/50596), [@adelton](https://github.com/adelton))
* StatefulSet: Fix "forbidden pod updates" error on Pods created prior to upgrading to 1.7. ([#48327](https://github.com/kubernetes/kubernetes/pull/48327)) ([#51149](https://github.com/kubernetes/kubernetes/pull/51149), [@kow3ns](https://github.com/kow3ns))
* Fixed regression in initial kubectl exec terminal dimensions ([#51127](https://github.com/kubernetes/kubernetes/pull/51127), [@chen-anders](https://github.com/chen-anders))
* Enforcement of fsGroup; enable ScaleIO multiple-instance volume mapping; default PVC capacity; alignment of PVC, PV, and volume names for dynamic provisioning ([#48999](https://github.com/kubernetes/kubernetes/pull/48999), [@vladimirvivien](https://github.com/vladimirvivien))



# v1.8.0-alpha.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.8.0-alpha.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes.tar.gz) | `c99042c4826352b724dc02c8d92c01c49e1ad1663d2c55e0bce931fe4d76c1e3`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-src.tar.gz) | `3ee0cd3594bd5b326f042044d87e120fe335bd8e722635220dd5741485ab3493`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `c716e167383d118373d7b10425bb8db6033675e4520591017c688575f28a596d`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `dfe87cad00600049c841c8fd96c49088d4f7cdd34e5a903ef8048f75718f2d21`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `97242dffee822cbf4e3e373acf05e9dc2f40176b18f4532a60264ecf92738356`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `42e25e810333b00434217bae0aece145f82d0c7043faea83ff62bed079bae651`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `7f9683c90dc894ee8cd7ad30ec58d0d49068d35478a71b315d2b7805ec28e14a`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `76347a154128e97cdd81674045b28035d89d509b35dda051f2cbc58c9b67fed4`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `c991cbbf0afa6eccd005b6e5ea28b0b20ecbc79ab7d64e32c24e03fcf05b48ff`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `94c2c29e8fd20d2a5c4f96098bd5c7d879a78e872f59c3c58ca1c775a57ddefb`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `bc98fd5dc01c6e6117c2c78d65884190bf99fd1fec0904e2af05e6dbf503ccc8`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `e32b56dbc69045b5b2821a2e3eb3c3b4a18cf4c11afd44e0c7c9c0e67bb38d02`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `5446addff583b0dc977b91375f3c399242f7996e1f66f52b9e14c015add3bf13`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `91e3cffed119b5105f6a6f74f583113384a26c746b459029c12babf45f680119`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `d4cb93787651193ef4fdd1d10a4822101586b2994d6b0e04d064687df8729910`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `916e7f63a4e0c67d9f106fdda6eb24efcc94356b05cd9eb288e45fac9ff79fe8`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `15b999b08f5fe0d8252f8a1c7e936b9e06f2b01132010b3cece547ab00b45282`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `9120f6a06053ed91566d378a26ae455f521ab46911f257d64f629d93d143b369`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `30af817f5de0ecb8a95ec898fba5b97e6b4f224927e1cf7efaf2d5479b23c116`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `8b0913e461d8ac821c2104a1f0b4efe3151f0d8e8598e0945e60b4ba7ac2d1a0`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `a78a3a837c0fbf6e092b312472c89ef0f3872c268b0a5e1e344e725a88c0717d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `a0a38c5830fc1b7996c5befc24502991fc8a095f82cf81ddd0a301163143a2c5`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `8af4253fe2c582843de329d12d84dbdc5f9f823f68ee08a42809864efc7c368d`

## Changelog since v1.8.0-alpha.2

### Action Required

* Remove deprecated kubectl command aliases `apiversions, clusterinfo, resize, rollingupdate, run-container, update` ([#49935](https://github.com/kubernetes/kubernetes/pull/49935), [@xiangpengzhao](https://github.com/xiangpengzhao))
* The following deprecated flags have been removed from `kube-controller-manager`: `replication-controller-lookup-cache-size`, `replicaset-lookup-cache-size`, and `daemonset-lookup-cache-size`. Make sure you no longer attempt to set them. ([#50678](https://github.com/kubernetes/kubernetes/pull/50678), [@xiangpengzhao](https://github.com/xiangpengzhao))
* Beta annotations `service.beta.kubernetes.io/external-traffic` and `service.beta.kubernetes.io/healthcheck-nodeport` have been removed. Please use fields `service.spec.externalTrafficPolicy` and `service.spec.healthCheckNodePort` instead. ([#50224](https://github.com/kubernetes/kubernetes/pull/50224), [@xiangpengzhao](https://github.com/xiangpengzhao))
* A cluster using the AWS cloud provider will need to label existing nodes and resources with a ClusterID or the kube-controller-manager will not start.  To run without a ClusterID pass --allow-untagged-cloud=true to the kube-controller-manager on startup. ([#49215](https://github.com/kubernetes/kubernetes/pull/49215), [@rrati](https://github.com/rrati))
* RBAC: the `system:node` role is no longer automatically granted to the `system:nodes` group in new clusters. It is recommended that nodes be authorized using the `Node` authorization mode instead. Installations that wish to continue giving all members of the `system:nodes` group the `system:node` role (which grants broad read access, including all secrets and configmaps) must create an installation-specific `ClusterRoleBinding`. ([#49638](https://github.com/kubernetes/kubernetes/pull/49638), [@liggitt](https://github.com/liggitt))
* StatefulSet: The deprecated `pod.alpha.kubernetes.io/initialized` annotation for interrupting StatefulSet Pod management is now ignored. If you were setting it to `true` or leaving it unset, no action is required. However, if you were setting it to `false`, be aware that previously-dormant StatefulSets may become active after upgrading. ([#49251](https://github.com/kubernetes/kubernetes/pull/49251), [@enisoc](https://github.com/enisoc))
* add some more deprecation warnings to cluster ([#49148](https://github.com/kubernetes/kubernetes/pull/49148), [@mikedanese](https://github.com/mikedanese))
* The --insecure-allow-any-token flag has been removed from kube-apiserver. Users of the flag should use impersonation headers instead for debugging. ([#49045](https://github.com/kubernetes/kubernetes/pull/49045), [@ericchiang](https://github.com/ericchiang))
* Restored cAdvisor prometheus metrics to the main port -- a regression that existed in v1.7.0-v1.7.2 ([#49079](https://github.com/kubernetes/kubernetes/pull/49079), [@smarterclayton](https://github.com/smarterclayton))
    * cAdvisor metrics can now be scraped from `/metrics/cadvisor` on the kubelet ports.
    * Note that you have to update your scraping jobs to get kubelet-only metrics from `/metrics` and `container_*` metrics from `/metrics/cadvisor`
* Change the default kubeadm bootstrap token TTL from infinite to 24 hours. This is a breaking change. If you require the old behavior, use `kubeadm init --token-ttl 0` / `kubeadm token create --ttl 0`. ([#48783](https://github.com/kubernetes/kubernetes/pull/48783), [@mattmoyer](https://github.com/mattmoyer))

### Other notable changes

* Remove duplicate command example from `kubectl port-forward --help` ([#50229](https://github.com/kubernetes/kubernetes/pull/50229), [@tcharding](https://github.com/tcharding))
* Adds a new `kubeadm config` command that lets users tell `kubeadm upgrade` what kubeadm configuration to use and lets users view the current state. ([#50980](https://github.com/kubernetes/kubernetes/pull/50980), [@luxas](https://github.com/luxas))
* Kubectl uses openapi for validation. If OpenAPI is not available on the server, it defaults back to the old Swagger. ([#50546](https://github.com/kubernetes/kubernetes/pull/50546), [@apelisse](https://github.com/apelisse))
* kubectl show node role if defined ([#50438](https://github.com/kubernetes/kubernetes/pull/50438), [@dixudx](https://github.com/dixudx))
* iSCSI volume plugin: iSCSI initiatorname support ([#48789](https://github.com/kubernetes/kubernetes/pull/48789), [@mtanino](https://github.com/mtanino))
* On AttachDetachController node status update, do not retry when node doesn't exist but keep the node entry in cache. ([#50806](https://github.com/kubernetes/kubernetes/pull/50806), [@verult](https://github.com/verult))
* Prevent unneeded endpoint updates ([#50934](https://github.com/kubernetes/kubernetes/pull/50934), [@joelsmith](https://github.com/joelsmith))
* Affinity in annotations alpha feature is no longer supported in 1.8. Anyone upgrading from 1.7 with AffinityInAnnotation feature enabled must ensure pods (specifically with pod anti-affinity PreferredDuringSchedulingIgnoredDuringExecution) with empty TopologyKey fields must be removed before upgrading to 1.8. ([#49976](https://github.com/kubernetes/kubernetes/pull/49976), [@aveshagarwal](https://github.com/aveshagarwal))
* - kubeadm now supports "ci/latest-1.8" or "ci-cross/latest-1.8" and similar labels. ([#49119](https://github.com/kubernetes/kubernetes/pull/49119), [@kad](https://github.com/kad))
* kubeadm: Adds dry-run support for kubeadm using the `--dry-run` option ([#50631](https://github.com/kubernetes/kubernetes/pull/50631), [@luxas](https://github.com/luxas))
* Change GCE installs (kube-up.sh) to use GCI/COS for node OS, by default. ([#46512](https://github.com/kubernetes/kubernetes/pull/46512), [@thockin](https://github.com/thockin))
* Use CollisionCount for collision avoidance when creating ControllerRevisions in StatefulSet controller ([#50490](https://github.com/kubernetes/kubernetes/pull/50490), [@liyinan926](https://github.com/liyinan926))
* AWS: Arbitrarily choose first (lexicographically) subnet in AZ ([#50255](https://github.com/kubernetes/kubernetes/pull/50255), [@mattlandis](https://github.com/mattlandis))
* Change CollisionCount from int64 to int32 across controllers ([#50575](https://github.com/kubernetes/kubernetes/pull/50575), [@dixudx](https://github.com/dixudx))
* fix GPU resource validation that incorrectly allows zero limits ([#50218](https://github.com/kubernetes/kubernetes/pull/50218), [@dixudx](https://github.com/dixudx))
* The `kubernetes.io/created-by` annotation is now deprecated and will be removed in v1.9. Use [ControllerRef](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md) instead to determine which controller, if any, owns an object. ([#50536](https://github.com/kubernetes/kubernetes/pull/50536), [@crimsonfaith91](https://github.com/crimsonfaith91))
* Disable Docker's health check until we officially support it ([#50796](https://github.com/kubernetes/kubernetes/pull/50796), [@yguo0905](https://github.com/yguo0905))
* Add ControllerRevision to apps/v1beta2 ([#50698](https://github.com/kubernetes/kubernetes/pull/50698), [@liyinan926](https://github.com/liyinan926))
* StorageClass has a new field to configure reclaim policy of dynamically provisioned PVs. ([#47987](https://github.com/kubernetes/kubernetes/pull/47987), [@wongma7](https://github.com/wongma7))
* Rerun init containers when the pod needs to be restarted ([#47599](https://github.com/kubernetes/kubernetes/pull/47599), [@yujuhong](https://github.com/yujuhong))
* Resources outside the `*kubernetes.io` namespace are integers and cannot be over-committed. ([#48922](https://github.com/kubernetes/kubernetes/pull/48922), [@ConnorDoyle](https://github.com/ConnorDoyle))
* apps/v1beta2 is enabled by default. DaemonSet, Deployment, ReplicaSet, and StatefulSet have been moved to this group version. ([#50643](https://github.com/kubernetes/kubernetes/pull/50643), [@kow3ns](https://github.com/kow3ns))
* TLS cert storage for self-hosted clusters is now configurable. You can store them as secrets (alpha) or as usual host mounts. ([#50762](https://github.com/kubernetes/kubernetes/pull/50762), [@jamiehannaford](https://github.com/jamiehannaford))
* Remove deprecated command 'kubectl stop' ([#46927](https://github.com/kubernetes/kubernetes/pull/46927), [@shiywang](https://github.com/shiywang))
* Add new Prometheus metric that monitors the remaining lifetime of certificates used to authenticate requests to the API server. ([#50387](https://github.com/kubernetes/kubernetes/pull/50387), [@jcbsmpsn](https://github.com/jcbsmpsn))
* Upgrade advanced audit to version v1beta1 ([#49115](https://github.com/kubernetes/kubernetes/pull/49115), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Cluster Autoscaler - fixes issues with taints and updates kube-proxy cpu request. ([#50514](https://github.com/kubernetes/kubernetes/pull/50514), [@mwielgus](https://github.com/mwielgus))
* fluentd-elasticsearch addon: change the fluentd base image to fix crashes on systems with non-standard systemd installation ([#50679](https://github.com/kubernetes/kubernetes/pull/50679), [@aknuds1](https://github.com/aknuds1))
* advanced audit: shutdown batching audit webhook gracefully ([#50577](https://github.com/kubernetes/kubernetes/pull/50577), [@crassirostris](https://github.com/crassirostris))
* Add Priority admission controller for monitoring and resolving PriorityClasses. ([#49322](https://github.com/kubernetes/kubernetes/pull/49322), [@bsalamat](https://github.com/bsalamat))
* apiservers: add synchronous shutdown mechanism on SIGTERM+INT ([#50439](https://github.com/kubernetes/kubernetes/pull/50439), [@sttts](https://github.com/sttts))
* Fix kubernetes-worker charm hook failure when applying labels ([#50633](https://github.com/kubernetes/kubernetes/pull/50633), [@Cynerva](https://github.com/Cynerva))
* kubeadm: Implementing the controlplane phase ([#50302](https://github.com/kubernetes/kubernetes/pull/50302), [@fabriziopandini](https://github.com/fabriziopandini))
* Refactor addons into multiple packages ([#50214](https://github.com/kubernetes/kubernetes/pull/50214), [@andrewrynhard](https://github.com/andrewrynhard))
* Kubelet now manages `/etc/hosts` file for both hostNetwork Pods and non-hostNetwork Pods. ([#49140](https://github.com/kubernetes/kubernetes/pull/49140), [@rickypai](https://github.com/rickypai))
* After 1.8, admission controller will add 'MemoryPressure' toleration to Guaranteed and Burstable pods. ([#50180](https://github.com/kubernetes/kubernetes/pull/50180), [@k82cn](https://github.com/k82cn))
* A new predicates, named 'CheckNodeCondition', was added to replace node condition filter. 'NetworkUnavailable', 'OutOfDisk' and 'NotReady' maybe reported as a reason when failed to schedule pods. ([#50362](https://github.com/kubernetes/kubernetes/pull/50362), [@k82cn](https://github.com/k82cn))
* fix apps DeploymentSpec conversion issue ([#49719](https://github.com/kubernetes/kubernetes/pull/49719), [@dixudx](https://github.com/dixudx))
* fluentd-gcp addon: Fix a bug in the event-exporter, when repeated events were not sent to Stackdriver. ([#50511](https://github.com/kubernetes/kubernetes/pull/50511), [@crassirostris](https://github.com/crassirostris))
* not allowing "kubectl edit <resource>" when you got an empty list ([#50205](https://github.com/kubernetes/kubernetes/pull/50205), [@dixudx](https://github.com/dixudx))
* fixes kubefed's ability to create RBAC roles in version-skewed clusters ([#50537](https://github.com/kubernetes/kubernetes/pull/50537), [@liggitt](https://github.com/liggitt))
* API server authentication now caches successful bearer token authentication results for a few seconds. ([#50258](https://github.com/kubernetes/kubernetes/pull/50258), [@liggitt](https://github.com/liggitt))
* Added field CollisionCount to StatefulSetStatus in both apps/v1beta1 and apps/v1beta2 ([#49983](https://github.com/kubernetes/kubernetes/pull/49983), [@liyinan926](https://github.com/liyinan926))
* FC volume plugin: Support WWID for volume identifier ([#48741](https://github.com/kubernetes/kubernetes/pull/48741), [@mtanino](https://github.com/mtanino))
* kubeadm: added enhanced TLS validation for token-based discovery in `kubeadm join` using a new `--discovery-token-ca-cert-hash` flag. ([#49520](https://github.com/kubernetes/kubernetes/pull/49520), [@mattmoyer](https://github.com/mattmoyer))
* federation: Support for leader-election among federation controller-manager instances introduced. ([#46090](https://github.com/kubernetes/kubernetes/pull/46090), [@shashidharatd](https://github.com/shashidharatd))
* New get-kube.sh option: KUBERNETES_SKIP_RELEASE_VALIDATION ([#50391](https://github.com/kubernetes/kubernetes/pull/50391), [@pipejakob](https://github.com/pipejakob))
* Azure: Allow VNet to be in a separate Resource Group. ([#49725](https://github.com/kubernetes/kubernetes/pull/49725), [@sylr](https://github.com/sylr))
* fix bug when azure cloud provider configuration file is not specified ([#49283](https://github.com/kubernetes/kubernetes/pull/49283), [@dixudx](https://github.com/dixudx))
* The `rbac.authorization.k8s.io/v1beta1` API has been promoted to `rbac.authorization.k8s.io/v1` with no changes. ([#49642](https://github.com/kubernetes/kubernetes/pull/49642), [@liggitt](https://github.com/liggitt))
    * The `rbac.authorization.k8s.io/v1alpha1` version is deprecated and will be removed in a future release.
* Fix an issue where if a CSR is not approved initially by the SAR approver is not retried. ([#49788](https://github.com/kubernetes/kubernetes/pull/49788), [@mikedanese](https://github.com/mikedanese))
* The v1.Service.PublishNotReadyAddresses field is added to notify DNS addons to publish the notReadyAddresses of Enpdoints. The "service.alpha.kubernetes.io/tolerate-unready-endpoints" annotation has been deprecated and will be removed when clients have sufficient time to consume the field. ([#49061](https://github.com/kubernetes/kubernetes/pull/49061), [@kow3ns](https://github.com/kow3ns))
* vSphere cloud provider: vSphere cloud provider code refactoring ([#49164](https://github.com/kubernetes/kubernetes/pull/49164), [@BaluDontu](https://github.com/BaluDontu))
* `cluster/gke` has been removed. GKE end-to-end testing should be done using `kubetest --deployment=gke` ([#50338](https://github.com/kubernetes/kubernetes/pull/50338), [@zmerlynn](https://github.com/zmerlynn))
* kubeadm: Upload configuration used at 'kubeadm init' time to ConfigMap for easier upgrades ([#50320](https://github.com/kubernetes/kubernetes/pull/50320), [@luxas](https://github.com/luxas))
* Adds (alpha feature) the ability to dynamically configure Kubelets by enabling the DynamicKubeletConfig feature gate, posting a ConfigMap to the API server, and setting the spec.configSource field on Node objects. See the proposal at https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node/dynamic-kubelet-configuration.md for details. ([#46254](https://github.com/kubernetes/kubernetes/pull/46254), [@mtaufen](https://github.com/mtaufen))
* Remove deprecated ScheduledJobs endpoints, use CronJobs instead. ([#49930](https://github.com/kubernetes/kubernetes/pull/49930), [@soltysh](https://github.com/soltysh))
* [Federation] Make the hpa scale time window configurable ([#49583](https://github.com/kubernetes/kubernetes/pull/49583), [@irfanurrehman](https://github.com/irfanurrehman))
* fuse daemons for GlusterFS and CephFS are now run in their own systemd scope when Kubernetes runs on a system with systemd. ([#49640](https://github.com/kubernetes/kubernetes/pull/49640), [@jsafrane](https://github.com/jsafrane))
* `kubectl proxy` will now correctly handle the `exec`, `attach`, and `portforward` commands.  You must pass `--disable-filter` to the command in order to allow these endpoints. ([#49534](https://github.com/kubernetes/kubernetes/pull/49534), [@smarterclayton](https://github.com/smarterclayton))
* Copy annotations from a StatefulSet's metadata to the ControllerRevisions it owns ([#50263](https://github.com/kubernetes/kubernetes/pull/50263), [@liyinan926](https://github.com/liyinan926))
* Make rolling update the default update strategy for v1beta2.DaemonSet and v1beta2.StatefulSet ([#50175](https://github.com/kubernetes/kubernetes/pull/50175), [@foxish](https://github.com/foxish))
* Deprecate Deployment .spec.rollbackTo field  ([#49340](https://github.com/kubernetes/kubernetes/pull/49340), [@janetkuo](https://github.com/janetkuo))
* Collect metrics from Heapster in Stackdriver mode. ([#50290](https://github.com/kubernetes/kubernetes/pull/50290), [@piosz](https://github.com/piosz))
* N/A ([#50179](https://github.com/kubernetes/kubernetes/pull/50179), [@k82cn](https://github.com/k82cn))
* [Federation] HPA controller ([#45993](https://github.com/kubernetes/kubernetes/pull/45993), [@irfanurrehman](https://github.com/irfanurrehman))
* Relax restrictions on environment variable names. ([#48986](https://github.com/kubernetes/kubernetes/pull/48986), [@timoreimann](https://github.com/timoreimann))
* The node condition 'NodeInodePressure' was removed, as kubelet did not report it. ([#50124](https://github.com/kubernetes/kubernetes/pull/50124), [@k82cn](https://github.com/k82cn))
* Fix premature return ([#49834](https://github.com/kubernetes/kubernetes/pull/49834), [@guoshimin](https://github.com/guoshimin))
* StatefulSet uses scale subresource when scaling in accord with ReplicationController, ReplicaSet, and Deployment implementations. ([#49168](https://github.com/kubernetes/kubernetes/pull/49168), [@crimsonfaith91](https://github.com/crimsonfaith91))
* Feature gates now determine whether a cluster is self-hosted. For more information, see the FeatureGates configuration flag. ([#50241](https://github.com/kubernetes/kubernetes/pull/50241), [@jamiehannaford](https://github.com/jamiehannaford))
* Updates Cinder AttachDisk operation to be more reliable by delegating Detaches to volume manager. ([#50042](https://github.com/kubernetes/kubernetes/pull/50042), [@jingxu97](https://github.com/jingxu97))
* add fieldSelector podIP ([#50091](https://github.com/kubernetes/kubernetes/pull/50091), [@dixudx](https://github.com/dixudx))
* Return Audit-Id http response header for trouble shooting ([#49377](https://github.com/kubernetes/kubernetes/pull/49377), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Status objects for 404 API errors will have the correct APIVersion ([#49868](https://github.com/kubernetes/kubernetes/pull/49868), [@shiywang](https://github.com/shiywang))
* Fix incorrect retry logic in scheduler ([#50106](https://github.com/kubernetes/kubernetes/pull/50106), [@julia-stripe](https://github.com/julia-stripe))
* Enforce explicit references to API group client interfaces in clientsets to avoid ambiguity. ([#49370](https://github.com/kubernetes/kubernetes/pull/49370), [@sttts](https://github.com/sttts))
* update dashboard image version ([#49855](https://github.com/kubernetes/kubernetes/pull/49855), [@zouyee](https://github.com/zouyee))
* kubeadm: Implementing the kubeconfig phase fully ([#49419](https://github.com/kubernetes/kubernetes/pull/49419), [@fabriziopandini](https://github.com/fabriziopandini))
* fixes a bug around using the Global config ElbSecurityGroup where Kuberentes would modify the passed in Security Group. ([#49805](https://github.com/kubernetes/kubernetes/pull/49805), [@nbutton23](https://github.com/nbutton23))
* Fluentd DaemonSet in the fluentd-elasticsearch addon is configured via ConfigMap and includes journald plugin ([#50082](https://github.com/kubernetes/kubernetes/pull/50082), [@crassirostris](https://github.com/crassirostris))
    * Elasticsearch StatefulSet in the fluentd-elasticsearch addon uses local storage instead of PVC by default
* Add possibility to use multiple floatingip pools in openstack loadbalancer ([#49697](https://github.com/kubernetes/kubernetes/pull/49697), [@zetaab](https://github.com/zetaab))
* The 504 timeout error was returning a JSON error body that indicated it was a 500.  The body contents now correctly report a 500 error. ([#49678](https://github.com/kubernetes/kubernetes/pull/49678), [@smarterclayton](https://github.com/smarterclayton))
* add examples for kubectl run --labels ([#49862](https://github.com/kubernetes/kubernetes/pull/49862), [@dixudx](https://github.com/dixudx))
* Kubelet will by default fail with swap enabled from now on. The experimental flag "--experimental-fail-swap-on" has been deprecated, please set the new "--fail-swap-on" flag to false if you wish to run with /proc/swaps on. ([#47181](https://github.com/kubernetes/kubernetes/pull/47181), [@dims](https://github.com/dims))
* Fix bug in scheduler that caused initially unschedulable pods to stuck in Pending state forever. ([#50028](https://github.com/kubernetes/kubernetes/pull/50028), [@julia-stripe](https://github.com/julia-stripe))
* GCE: Bump GLBC version to 0.9.6 ([#50096](https://github.com/kubernetes/kubernetes/pull/50096), [@nicksardo](https://github.com/nicksardo))
* Remove 0,1,3 from rand.String, to avoid 'bad words' ([#50070](https://github.com/kubernetes/kubernetes/pull/50070), [@dixudx](https://github.com/dixudx))
* Fix data race during addition of new CRD ([#50098](https://github.com/kubernetes/kubernetes/pull/50098), [@nikhita](https://github.com/nikhita))
* Do not try to run preStopHook when the gracePeriod is 0 ([#49449](https://github.com/kubernetes/kubernetes/pull/49449), [@dhilipkumars](https://github.com/dhilipkumars))
* The SubjectAccessReview API in the authorization.k8s.io API group now allows providing the user uid. ([#49677](https://github.com/kubernetes/kubernetes/pull/49677), [@dims](https://github.com/dims))
* Increase default value of apps/v1beta2 DeploymentSpec.RevisionHistoryLimit to 10 ([#49924](https://github.com/kubernetes/kubernetes/pull/49924), [@dixudx](https://github.com/dixudx))
* Upgrade Elasticsearch/Kibana to 5.5.1 in fluentd-elasticsearch addon ([#48722](https://github.com/kubernetes/kubernetes/pull/48722), [@aknuds1](https://github.com/aknuds1))
        * Switch to basing our image of Elasticsearch in fluentd-elasticsearch addon off the official one
        * Switch to the official image of Kibana in fluentd-elasticsearch addon
        * Use StatefulSet for Elasticsearch instead of ReplicationController, with persistent volume claims
        * Require authenticating towards Elasticsearch, as Elasticsearch 5.5 by default requires basic authentication
* Rebase hyperkube image on debian-hyperkube-base, based on debian-base. ([#48365](https://github.com/kubernetes/kubernetes/pull/48365), [@ixdy](https://github.com/ixdy))
* change apps/v1beta2 StatefulSet observedGeneration (optional field) from a pointer to an int for consistency ([#49607](https://github.com/kubernetes/kubernetes/pull/49607), [@dixudx](https://github.com/dixudx))
* After a kubelet rotates its client cert, it now closes its connections to the API server to force a handshake using the new cert. Previously, the kubelet could keep its existing connection open, even if the cert used for that connection was expired and rejected by the API server. ([#49899](https://github.com/kubernetes/kubernetes/pull/49899), [@ericchiang](https://github.com/ericchiang))
* Improve our Instance Metadata coverage in Azure. ([#49237](https://github.com/kubernetes/kubernetes/pull/49237), [@brendandburns](https://github.com/brendandburns))
* Add etcd connectivity endpoint to healthz ([#49412](https://github.com/kubernetes/kubernetes/pull/49412), [@bjhaid](https://github.com/bjhaid))
* kube-proxy will emit "FailedToStartNodeHealthcheck" event when fails to start healthz server. ([#49267](https://github.com/kubernetes/kubernetes/pull/49267), [@MrHohn](https://github.com/MrHohn))
* Fixed a bug in the API server watch cache, which could cause a missing watch event immediately after cache initialization. ([#49992](https://github.com/kubernetes/kubernetes/pull/49992), [@liggitt](https://github.com/liggitt))
* Enforcement of fsGroup; enable ScaleIO multiple-instance volume mapping; default PVC capacity; alignment of PVC, PV, and volume names for dynamic provisioning ([#48999](https://github.com/kubernetes/kubernetes/pull/48999), [@vladimirvivien](https://github.com/vladimirvivien))
* In GCE, add measures to prevent corruption of known_tokens.csv. ([#49897](https://github.com/kubernetes/kubernetes/pull/49897), [@mikedanese](https://github.com/mikedanese))
* kubeadm: Fix join preflight check false negative ([#49825](https://github.com/kubernetes/kubernetes/pull/49825), [@erhudy](https://github.com/erhudy))
* route_controller will emit "FailedToCreateRoute" event when fails to create route. ([#49821](https://github.com/kubernetes/kubernetes/pull/49821), [@MrHohn](https://github.com/MrHohn))
* Fix incorrect parsing of io_priority in Portworx volume StorageClass and add support for new paramters. ([#49526](https://github.com/kubernetes/kubernetes/pull/49526), [@harsh-px](https://github.com/harsh-px))
* The API Server now automatically creates RBAC ClusterRoles for CSR approving.  ([#49284](https://github.com/kubernetes/kubernetes/pull/49284), [@luxas](https://github.com/luxas))
    * Each deployment method should bind users/groups to the ClusterRoles if they are using this feature.
* Adds AllowPrivilegeEscalation to control whether a process can gain more privileges than its parent process ([#47019](https://github.com/kubernetes/kubernetes/pull/47019), [@jessfraz](https://github.com/jessfraz))
* `hack/local-up-cluster.sh` now enables the Node authorizer by default. Authorization modes can be overridden with the `AUTHORIZATION_MODE` environment variable, and the `ENABLE_RBAC` environment variable is no longer used. ([#49812](https://github.com/kubernetes/kubernetes/pull/49812), [@liggitt](https://github.com/liggitt))
* rename stop.go file to delete.go to avoid confusion ([#49533](https://github.com/kubernetes/kubernetes/pull/49533), [@dixudx](https://github.com/dixudx))
* Adding option to set the federation api server port if nodeport is set ([#46283](https://github.com/kubernetes/kubernetes/pull/46283), [@ktsakalozos](https://github.com/ktsakalozos))
* The garbage collector now supports custom APIs added via CustomResourceDefinition or aggregated apiservers. Note that the garbage collector controller refreshes periodically, so there is a latency between when the API is added and when the garbage collector starts to manage it. ([#47665](https://github.com/kubernetes/kubernetes/pull/47665), [@ironcladlou](https://github.com/ironcladlou))
* set the juju master charm state to blocked if the services appear to be failing ([#49717](https://github.com/kubernetes/kubernetes/pull/49717), [@wwwtyro](https://github.com/wwwtyro))
* keep-terminated-pod-volumes flag on kubelet is deprecated. ([#47539](https://github.com/kubernetes/kubernetes/pull/47539), [@gnufied](https://github.com/gnufied))
* kubectl describe podsecuritypolicy describes all fields. ([#45813](https://github.com/kubernetes/kubernetes/pull/45813), [@xilabao](https://github.com/xilabao))
* Added flag support to kubectl plugins ([#47267](https://github.com/kubernetes/kubernetes/pull/47267), [@fabianofranz](https://github.com/fabianofranz))
* Adding metrics support to local volume ([#49598](https://github.com/kubernetes/kubernetes/pull/49598), [@sbezverk](https://github.com/sbezverk))
* Bug fix: Parsing of `--requestheader-group-headers` in requests should be case-insensitive. ([#49219](https://github.com/kubernetes/kubernetes/pull/49219), [@jmillikin-stripe](https://github.com/jmillikin-stripe))
* Fix instance metadata service URL. ([#49081](https://github.com/kubernetes/kubernetes/pull/49081), [@brendandburns](https://github.com/brendandburns))
* Add a new API object apps/v1beta2.ReplicaSet ([#49238](https://github.com/kubernetes/kubernetes/pull/49238), [@janetkuo](https://github.com/janetkuo))
* fix pdb validation bug on PodDisruptionBudgetSpec ([#48706](https://github.com/kubernetes/kubernetes/pull/48706), [@dixudx](https://github.com/dixudx))
* Revert deprecation of vCenter port in vSphere Cloud Provider. ([#49689](https://github.com/kubernetes/kubernetes/pull/49689), [@divyenpatel](https://github.com/divyenpatel))
* Rev version of Calico's Typha daemon used in add-on to v0.2.3 to pull in bug-fixes. ([#48469](https://github.com/kubernetes/kubernetes/pull/48469), [@fasaxc](https://github.com/fasaxc))
* set default adminid for rbd deleter if unset  ([#49271](https://github.com/kubernetes/kubernetes/pull/49271), [@dixudx](https://github.com/dixudx))
* Adding type apps/v1beta2.DaemonSet ([#49071](https://github.com/kubernetes/kubernetes/pull/49071), [@foxish](https://github.com/foxish))
* Fix nil value issue when creating json patch for merge ([#49259](https://github.com/kubernetes/kubernetes/pull/49259), [@dixudx](https://github.com/dixudx))
* Adds metrics for checking reflector health. ([#48224](https://github.com/kubernetes/kubernetes/pull/48224), [@deads2k](https://github.com/deads2k))
* remove deads2k from volume reviewer ([#49566](https://github.com/kubernetes/kubernetes/pull/49566), [@deads2k](https://github.com/deads2k))
* Unify genclient tags and add more fine control on verbs generated ([#49192](https://github.com/kubernetes/kubernetes/pull/49192), [@mfojtik](https://github.com/mfojtik))
* kubeadm: Fixes a small bug where `--config` and `--skip-*` flags couldn't be passed at the same time in validation. ([#49498](https://github.com/kubernetes/kubernetes/pull/49498), [@luxas](https://github.com/luxas))
* Remove depreciated flags: --low-diskspace-threshold-mb and --outofdisk-transition-frequency, which are replaced by --eviction-hard ([#48846](https://github.com/kubernetes/kubernetes/pull/48846), [@dashpole](https://github.com/dashpole))
* Fixed OpenAPI Description and Nickname of API objects with subresources ([#49357](https://github.com/kubernetes/kubernetes/pull/49357), [@mbohlool](https://github.com/mbohlool))
* set RBD default values as constant vars ([#49274](https://github.com/kubernetes/kubernetes/pull/49274), [@dixudx](https://github.com/dixudx))
* Fix a bug with binding mount directories and files using flexVolumes ([#49118](https://github.com/kubernetes/kubernetes/pull/49118), [@adelton](https://github.com/adelton))
* PodPreset is not injected if conflict occurs while applying PodPresets to a Pod. ([#47864](https://github.com/kubernetes/kubernetes/pull/47864), [@droot](https://github.com/droot))
* `kubectl drain` no longer spins trying to delete pods that do not exist ([#49444](https://github.com/kubernetes/kubernetes/pull/49444), [@eparis](https://github.com/eparis))
* Support specifying of FSType in StorageClass ([#45345](https://github.com/kubernetes/kubernetes/pull/45345), [@codablock](https://github.com/codablock))
* The NodeRestriction admission plugin now allows a node to evict pods bound to itself ([#48707](https://github.com/kubernetes/kubernetes/pull/48707), [@danielfm](https://github.com/danielfm))
* more robust stat handling from ceph df output in the kubernetes-master charm create-rbd-pv action ([#49394](https://github.com/kubernetes/kubernetes/pull/49394), [@wwwtyro](https://github.com/wwwtyro))
* added cronjobs.batch to all, so kubectl get all returns them. ([#49326](https://github.com/kubernetes/kubernetes/pull/49326), [@deads2k](https://github.com/deads2k))
* Update status to show failing services. ([#49296](https://github.com/kubernetes/kubernetes/pull/49296), [@ktsakalozos](https://github.com/ktsakalozos))
* Fixes [#49418](https://github.com/kubernetes/kubernetes/pull/49418) where kube-controller-manager can panic on volume.CanSupport methods and enter a crash loop. ([#49420](https://github.com/kubernetes/kubernetes/pull/49420), [@gnufied](https://github.com/gnufied))
* Add a new API version apps/v1beta2 ([#48746](https://github.com/kubernetes/kubernetes/pull/48746), [@janetkuo](https://github.com/janetkuo))
* Websocket requests to aggregated APIs now perform TLS verification using the service DNS name instead of the backend server's IP address, consistent with non-websocket requests. ([#49353](https://github.com/kubernetes/kubernetes/pull/49353), [@liggitt](https://github.com/liggitt))
* kubeadm: Don't set a specific `spc_t` SELinux label on the etcd Static Pod as that is more privs than etcd needs and due to that `spc_t` isn't compatible with some OSes. ([#49328](https://github.com/kubernetes/kubernetes/pull/49328), [@euank](https://github.com/euank))
* GCE Cloud Provider: New created LoadBalancer type Service will have health checks for nodes by default if all nodes have version >= v1.7.2. ([#49330](https://github.com/kubernetes/kubernetes/pull/49330), [@MrHohn](https://github.com/MrHohn))
* hack/local-up-cluster.sh now enables RBAC authorization by default ([#49323](https://github.com/kubernetes/kubernetes/pull/49323), [@mtanino](https://github.com/mtanino))
* Use port 20256 for node-problem-detector in standalone mode. ([#49316](https://github.com/kubernetes/kubernetes/pull/49316), [@ajitak](https://github.com/ajitak))
* Fixed unmounting of vSphere volumes when kubelet runs in a container. ([#49111](https://github.com/kubernetes/kubernetes/pull/49111), [@jsafrane](https://github.com/jsafrane))
* use informers for quota evaluation of core resources where possible ([#49230](https://github.com/kubernetes/kubernetes/pull/49230), [@deads2k](https://github.com/deads2k))
* additional backoff in azure cloudprovider ([#48967](https://github.com/kubernetes/kubernetes/pull/48967), [@jackfrancis](https://github.com/jackfrancis))
* allow impersonate serviceaccount in cli ([#48253](https://github.com/kubernetes/kubernetes/pull/48253), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Add PriorityClass API object under new "scheduling" API group ([#48377](https://github.com/kubernetes/kubernetes/pull/48377), [@bsalamat](https://github.com/bsalamat))
* Added golint check for pkg/kubelet. ([#47316](https://github.com/kubernetes/kubernetes/pull/47316), [@k82cn](https://github.com/k82cn))
* azure: acr: support MSI with preview ACR with AAD auth ([#48981](https://github.com/kubernetes/kubernetes/pull/48981), [@colemickens](https://github.com/colemickens))
* Set default CIDR to /16 for Juju deployments ([#49182](https://github.com/kubernetes/kubernetes/pull/49182), [@ktsakalozos](https://github.com/ktsakalozos))
* Fix pod preset to ignore input pod namespace in favor of request namespace ([#49120](https://github.com/kubernetes/kubernetes/pull/49120), [@jpeeler](https://github.com/jpeeler))
* Previously a deleted bootstrapping token secret would be considered valid until it was reaped.  Now it is invalid as soon as the deletionTimestamp is set. ([#49057](https://github.com/kubernetes/kubernetes/pull/49057), [@ericchiang](https://github.com/ericchiang))
* Set default snap channel on charms to 1.7 stable ([#48874](https://github.com/kubernetes/kubernetes/pull/48874), [@ktsakalozos](https://github.com/ktsakalozos))
* prevent unsetting of nonexistent previous port in kubeapi-load-balancer charm ([#49033](https://github.com/kubernetes/kubernetes/pull/49033), [@wwwtyro](https://github.com/wwwtyro))
* kubeadm: Make kube-proxy tolerate the external cloud provider taint so that an external cloud provider can be easily used on top of kubeadm ([#49017](https://github.com/kubernetes/kubernetes/pull/49017), [@luxas](https://github.com/luxas))
* Fix Pods using Portworx volumes getting stuck in ContainerCreating phase. ([#48898](https://github.com/kubernetes/kubernetes/pull/48898), [@harsh-px](https://github.com/harsh-px))
* hpa: Prevent scaling below MinReplicas if desiredReplicas is zero ([#48997](https://github.com/kubernetes/kubernetes/pull/48997), [@johanneswuerbach](https://github.com/johanneswuerbach))
* Kubelet CRI: move seccomp from annotations to security context. ([#46332](https://github.com/kubernetes/kubernetes/pull/46332), [@feiskyer](https://github.com/feiskyer))
* Never prevent deletion of resources as part of namespace lifecycle ([#48733](https://github.com/kubernetes/kubernetes/pull/48733), [@liggitt](https://github.com/liggitt))
* The generic RESTClient type (`k8s.io/client-go/rest`) no longer exposes `LabelSelectorParam` or `FieldSelectorParam` methods - use `VersionedParams` with `metav1.ListOptions` instead.  The `UintParam` method has been removed.  The `timeout` parameter will no longer cause an error when using `Param()`. ([#48991](https://github.com/kubernetes/kubernetes/pull/48991), [@smarterclayton](https://github.com/smarterclayton))
* Support completion for kubectl config delete-cluster ([#48381](https://github.com/kubernetes/kubernetes/pull/48381), [@superbrothers](https://github.com/superbrothers))
* Could get the patch from kubectl edit command ([#46091](https://github.com/kubernetes/kubernetes/pull/46091), [@xilabao](https://github.com/xilabao))
* Added scheduler integration test owners. ([#46930](https://github.com/kubernetes/kubernetes/pull/46930), [@k82cn](https://github.com/k82cn))
* `kubectl run` learned how to set a service account name in the generated pod spec with the `--serviceaccount` flag. ([#46318](https://github.com/kubernetes/kubernetes/pull/46318), [@liggitt](https://github.com/liggitt))
* Fix share name generation in azure file provisioner. ([#48326](https://github.com/kubernetes/kubernetes/pull/48326), [@karataliu](https://github.com/karataliu))
* Fixed a bug where a jsonpath filter would return an error if one of the items being evaluated did not contain all of the nested elements in the filter query. ([#47846](https://github.com/kubernetes/kubernetes/pull/47846), [@ncdc](https://github.com/ncdc))
* Uses the port config option in the kubeapi-load-balancer charm. ([#48958](https://github.com/kubernetes/kubernetes/pull/48958), [@wwwtyro](https://github.com/wwwtyro))
* azure: support retrieving access tokens via managed identity extension ([#48854](https://github.com/kubernetes/kubernetes/pull/48854), [@colemickens](https://github.com/colemickens))
* Add a runtime warning about the kubeadm default token TTL changes. ([#48838](https://github.com/kubernetes/kubernetes/pull/48838), [@mattmoyer](https://github.com/mattmoyer))
* Azure PD (Managed/Blob) ([#46360](https://github.com/kubernetes/kubernetes/pull/46360), [@khenidak](https://github.com/khenidak))
* Redirect all examples README to the the kubernetes/examples repo ([#46362](https://github.com/kubernetes/kubernetes/pull/46362), [@sebgoa](https://github.com/sebgoa))
* Fix a regression that broke the `--config` flag for `kubeadm init`. ([#48915](https://github.com/kubernetes/kubernetes/pull/48915), [@mattmoyer](https://github.com/mattmoyer))
* Fluentd-gcp DaemonSet exposes different set of metrics. ([#48812](https://github.com/kubernetes/kubernetes/pull/48812), [@crassirostris](https://github.com/crassirostris))
* MountPath should be absolute ([#48815](https://github.com/kubernetes/kubernetes/pull/48815), [@dixudx](https://github.com/dixudx))
* Updated comments of func in testapi. ([#48407](https://github.com/kubernetes/kubernetes/pull/48407), [@k82cn](https://github.com/k82cn))
* Fix service controller crash loop when Service with GCP LoadBalancer uses static IP ([#48848](https://github.com/kubernetes/kubernetes/pull/48848), [@nicksardo](https://github.com/nicksardo)) ([#48849](https://github.com/kubernetes/kubernetes/pull/48849), [@nicksardo](https://github.com/nicksardo))
* Fix pods failing to start when subPath is a dangling symlink from kubelet point of view, which can happen if it is running inside a container ([#48555](https://github.com/kubernetes/kubernetes/pull/48555), [@redbaron](https://github.com/redbaron))
* Add initial support for the Azure instance metadata service. ([#48243](https://github.com/kubernetes/kubernetes/pull/48243), [@brendandburns](https://github.com/brendandburns))
* Added new flag to `kubeadm init`: --node-name, that lets you specify the name of the Node object that will be created ([#48594](https://github.com/kubernetes/kubernetes/pull/48594), [@GheRivero](https://github.com/GheRivero))
* Added pod evictors for new zone. ([#47952](https://github.com/kubernetes/kubernetes/pull/47952), [@k82cn](https://github.com/k82cn))
* kube-up and kubemark will default to using cos (GCI) images for nodes. ([#48279](https://github.com/kubernetes/kubernetes/pull/48279), [@abgworrall](https://github.com/abgworrall))
    * The previous default was container-vm (CVM, "debian"), which is deprecated.
    * If you need to explicitly use container-vm for some reason, you should set
    * KUBE_NODE_OS_DISTRIBUTION=debian
* kubectl: Fix bug that showed terminated/evicted pods even without `--show-all`. ([#48786](https://github.com/kubernetes/kubernetes/pull/48786), [@janetkuo](https://github.com/janetkuo))
* Fixed GlusterFS volumes taking too long to time out ([#48709](https://github.com/kubernetes/kubernetes/pull/48709), [@jsafrane](https://github.com/jsafrane))



# v1.7.4

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.4


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes.tar.gz) | `dfc4521a81cdcb6a644757247f7b5311ed371d767053e0b28ac1c6a58a890bd2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-src.tar.gz) | `d9e0e091b202c2ca155d31ed88b616a4cb759bc14d84b637271b55d6b0774bd1`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-darwin-386.tar.gz) | `e87bb880f89766c0642eadfca387d91b82845da4c26eb4b213665b82d9060641`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-darwin-amd64.tar.gz) | `a913d8f2578449e926c822a5e96b3c7185fd0c97589d45f4f9224940f3f2e4c9`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-linux-386.tar.gz) | `03ed586c6c2c1e5fbdf3e75627b2d981b5e54fe1f4090a23759e34f1cfe6e7d0`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-linux-amd64.tar.gz) | `19eef604019d4562e9b1107ad8d1d3886512ba240a9eb82f8d6b4332b2cd5e7d`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-linux-arm64.tar.gz) | `9c60f289d55674b3af26bc219b4478aa2d46f6cbf7743493c14ad49099a17794`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-linux-arm.tar.gz) | `6fb2260f8a5ac18b5f16cfcf34579c675ee2222b54508d0abd36624acb24f314`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-linux-ppc64le.tar.gz) | `e5fe4b73cbd4e5662e77b1ca72e959f692fde39459bd1e9711814d877dabf137`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-linux-s390x.tar.gz) | `2ed3545580731b838f732cc0b8f805e0aa03478bf2913fd3ae3230042edea2c3`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-windows-386.tar.gz) | `5b1c79aea5e5174e0d135a15dd3a33cdbdb2c465f08af1878c5fc38aaf28ba7b`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-client-windows-amd64.tar.gz) | `07ca92b2f7659ecc8f5c93a707767fe6de099c20d5a81451f652968a326ec063`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-server-linux-amd64.tar.gz) | `09c420fdb9b912c172b19638d67b27bc7994e2608185051f412804fa55790076`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-server-linux-arm64.tar.gz) | `49d0a383fced290223b3727011904283e16183f0356f7d952f587eef9dbef4a8`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-server-linux-arm.tar.gz) | `74442000ff61b10b12f783594cb15b6a1db3dd0d879fe8c0863e8b5ec7de7de4`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-server-linux-ppc64le.tar.gz) | `809cf588ca15ab57ca4570aa7939fb08b7dc7e038a0475098f9f4ba5ced9e4c7`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-server-linux-s390x.tar.gz) | `33961f57ece65872976065614055b41a0bb3237152bb86ae40b9fa6a0089ab2f`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-node-linux-amd64.tar.gz) | `59e0643c46f9ad5b401b9bb8aa067d1263f0b22f06f16008b5c7518ee905324e`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-node-linux-arm64.tar.gz) | `216523d47ec6b451308708eda53ef5fe05f59c3c1c912955094be798dfe8f7bb`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-node-linux-arm.tar.gz) | `13ccad18701f67930991128c39efecea3ba873e21cecc81d79a5563c11f16ad2`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-node-linux-ppc64le.tar.gz) | `a6b644f842e84b3dc6059fae19dffe4da1d3dbc8e6464f264664169634f89a02`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-node-linux-s390x.tar.gz) | `b753f1bf1b26a62bc26def4b6b49dacdd16389d2d57ca2c384f449727daacc1d`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.4/kubernetes-node-windows-amd64.tar.gz) | `1fabda88ff9cbfcae406707c8584efc75600b2484317a0f22d56a0c44ca32184`

## Changelog since v1.7.3

### Other notable changes

* Azure: Allow VNet to be in a separate Resource Group. ([#49725](https://github.com/kubernetes/kubernetes/pull/49725), [@sylr](https://github.com/sylr))
* Fix an issue where if a CSR is not approved initially by the SAR approver is not retried. ([#49788](https://github.com/kubernetes/kubernetes/pull/49788), [@mikedanese](https://github.com/mikedanese))
* Cluster Autoscaler - fixes issues with taints and updates kube-proxy cpu request. ([#50514](https://github.com/kubernetes/kubernetes/pull/50514), [@mwielgus](https://github.com/mwielgus))
* Bumped Heapster version to 1.4.1: ([#50642](https://github.com/kubernetes/kubernetes/pull/50642), [@piosz](https://github.com/piosz))
    * - handle gracefully problem when kubelet reports duplicated stats for the same container (see [#47853](https://github.com/kubernetes/kubernetes/pull/47853)) on Heapster side
    * - fixed bugs and improved performance in Stackdriver Sink
* fluentd-gcp addon: Fix a bug in the event-exporter, when repeated events were not sent to Stackdriver. ([#50511](https://github.com/kubernetes/kubernetes/pull/50511), [@crassirostris](https://github.com/crassirostris))
* Collect metrics from Heapster in Stackdriver mode. ([#50517](https://github.com/kubernetes/kubernetes/pull/50517), [@piosz](https://github.com/piosz))
* fixes a bug around using the Global config ElbSecurityGroup where Kuberentes would modify the passed in Security Group. ([#49805](https://github.com/kubernetes/kubernetes/pull/49805), [@nbutton23](https://github.com/nbutton23))
* Updates Cinder AttachDisk operation to be more reliable by delegating Detaches to volume manager. ([#50042](https://github.com/kubernetes/kubernetes/pull/50042), [@jingxu97](https://github.com/jingxu97))
* fixes kubefed's ability to create RBAC roles in version-skewed clusters ([#50537](https://github.com/kubernetes/kubernetes/pull/50537), [@liggitt](https://github.com/liggitt))
* Fix data race during addition of new CRD ([#50098](https://github.com/kubernetes/kubernetes/pull/50098), [@nikhita](https://github.com/nikhita))
* Fix bug in scheduler that caused initially unschedulable pods to stuck in Pending state forever. ([#50028](https://github.com/kubernetes/kubernetes/pull/50028), [@julia-stripe](https://github.com/julia-stripe))
* Fix incorrect retry logic in scheduler ([#50106](https://github.com/kubernetes/kubernetes/pull/50106), [@julia-stripe](https://github.com/julia-stripe))
* GCE: Bump GLBC version to 0.9.6 ([#50096](https://github.com/kubernetes/kubernetes/pull/50096), [@nicksardo](https://github.com/nicksardo))
* The NodeRestriction admission plugin now allows a node to evict pods bound to itself ([#48707](https://github.com/kubernetes/kubernetes/pull/48707), [@danielfm](https://github.com/danielfm))
* Fixed a bug in the API server watch cache, which could cause a missing watch event immediately after cache initialization. ([#49992](https://github.com/kubernetes/kubernetes/pull/49992), [@liggitt](https://github.com/liggitt))



# v1.7.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes.tar.gz) | `8afa3919b6bff47ada1c298837881ef7eed9516694d54517ac2a59b0bbe7308c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-src.tar.gz) | `54f77cb2d392de742580fc5fb9ca5acf29adfb4620f4dcb09050d7dfbbd260d7`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-darwin-386.tar.gz) | `9a62ebc7b25847ce3201e01df6a845139e1de6ea4e9cc02ef4c713d33c5a9916`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-darwin-amd64.tar.gz) | `b786b39e89908ed567a17dac6e554cf5580f0ad817334ad2bd447a8f8b5bde95`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-linux-386.tar.gz) | `aed5d3ccaf9fafb52775234d27168674f9b536ce72cb56e51376761f2f77c653`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-linux-amd64.tar.gz) | `8d66c7912914ac9add514e660fdc8c963b748a7c588c43a14533157a9f0e1c92`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-linux-arm64.tar.gz) | `7b65dd3d72712e419679685dfe6324274b080415eb556a2dca95bcb61cbf8882`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-linux-arm.tar.gz) | `42843f265bcf56a801942cee378f235b94eea1b8ac431315a9db0fb7d78736ad`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-linux-ppc64le.tar.gz) | `c2976c26f9f4842f59cf0d5e8a79913f688b57843b825bfdd300ca4d8b4e7f1f`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-linux-s390x.tar.gz) | `7f019b5a32e927422136be0672e0dd97bcf496e7c25935a3e3d68474c2bd543d`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-windows-386.tar.gz) | `2d4d26928f31342081337bc9b8508067b3a29c9f673a6f67186e04c447d274c1`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-client-windows-amd64.tar.gz) | `90423aaa71fdd813ac58ceb25e670bd8b53a417e6ac34e67ad2cacc7f5a4c579`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-server-linux-amd64.tar.gz) | `f4ae8d6655eedc1bed14c6d7da74156cb1f43a01a554f6399a177e3acb385bf1`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-server-linux-arm64.tar.gz) | `4a2ab8183f944f7e952b929008a4f39297897b7d411b233e7f952a8a755eb65c`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-server-linux-arm.tar.gz) | `fde4d9f8a2e360d8cabfa7d56ed1b2ec25a09ce1ab8db3d2e5e673f098586488`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-server-linux-ppc64le.tar.gz) | `7d012b8393c06bd2418b1173fb306879e6fd11437f874b92bffcdba5ef4fb14a`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-server-linux-s390x.tar.gz) | `364b2c768bca178844de0752b5c0e4d3ee37cfc98ca4b8deac71e71aded84d5a`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-node-linux-amd64.tar.gz) | `29b7a0649f0fed7f4e892d4c5ecbe7dfc57d3631e29c90dfafd305b19e324e57`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-node-linux-arm64.tar.gz) | `6c8f2d8651bddd625e336a16546b923cd18a8a8f01df6d236db46b914b9edbe0`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-node-linux-arm.tar.gz) | `1ad3c378ad56f7233b4e75cdb3fb1ba52cde1f7695a536b2ccbefc614f56208f`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-node-linux-ppc64le.tar.gz) | `32860144cf02a62b29bd2a8fcaa155ccf3f004352e363d398ff1eccf90ebaae7`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-node-linux-s390x.tar.gz) | `eb34c895267d91324841abc0cc17788def37bfee297f3067cbee6f088f6c6b39`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.3/kubernetes-node-windows-amd64.tar.gz) | `de2efc1cf0979bade8db64c342bbcec021d5dd271b2e5232c9d282104afb4368`

## Changelog since v1.7.2

### Other notable changes

* fix pdb validation bug on PodDisruptionBudgetSpec ([#48706](https://github.com/kubernetes/kubernetes/pull/48706), [@dixudx](https://github.com/dixudx))
* kubeadm: Fix join preflight check false negative ([#49825](https://github.com/kubernetes/kubernetes/pull/49825), [@erhudy](https://github.com/erhudy))
* Revert deprecation of vCenter port in vSphere Cloud Provider. ([#49689](https://github.com/kubernetes/kubernetes/pull/49689), [@divyenpatel](https://github.com/divyenpatel))
* Fluentd-gcp DaemonSet exposes different set of metrics. ([#48812](https://github.com/kubernetes/kubernetes/pull/48812), [@crassirostris](https://github.com/crassirostris))
* Fixed OpenAPI Description and Nickname of API objects with subresources ([#49357](https://github.com/kubernetes/kubernetes/pull/49357), [@mbohlool](https://github.com/mbohlool))
* Websocket requests to aggregated APIs now perform TLS verification using the service DNS name instead of the backend server's IP address, consistent with non-websocket requests. ([#49353](https://github.com/kubernetes/kubernetes/pull/49353), [@liggitt](https://github.com/liggitt))
* kubeadm: Fixes a small bug where `--config` and `--skip-*` flags couldn't be passed at the same time in validation. ([#49498](https://github.com/kubernetes/kubernetes/pull/49498), [@luxas](https://github.com/luxas))
* kubeadm: Don't set a specific `spc_t` SELinux label on the etcd Static Pod as that is more privs than etcd needs and due to that `spc_t` isn't compatible with some OSes. ([#49328](https://github.com/kubernetes/kubernetes/pull/49328), [@euank](https://github.com/euank))
* Websocket requests to aggregated APIs now perform TLS verification using the service DNS name instead of the backend server's IP address, consistent with non-websocket requests. ([#49353](https://github.com/kubernetes/kubernetes/pull/49353), [@liggitt](https://github.com/liggitt))
* `kubectl drain` no longer spins trying to delete pods that do not exist ([#49444](https://github.com/kubernetes/kubernetes/pull/49444), [@eparis](https://github.com/eparis))
* Fixes [#49418](https://github.com/kubernetes/kubernetes/pull/49418) where kube-controller-manager can panic on volume.CanSupport methods and enter a crash loop. ([#49420](https://github.com/kubernetes/kubernetes/pull/49420), [@gnufied](https://github.com/gnufied))
* Fix Cinder to support http status 300 in pagination ([#47602](https://github.com/kubernetes/kubernetes/pull/47602), [@rootfs](https://github.com/rootfs))
* Automated cherry pick of [#49079](https://github.com/kubernetes/kubernetes/pull/49079) upstream release 1.7 ([#49254](https://github.com/kubernetes/kubernetes/pull/49254), [@feiskyer](https://github.com/feiskyer))
* Fixed GlusterFS volumes taking too long to time out ([#48709](https://github.com/kubernetes/kubernetes/pull/48709), [@jsafrane](https://github.com/jsafrane))
* The IP address and port for kube-proxy metrics server is now configurable via flag `--metrics-bind-address` ([#48625](https://github.com/kubernetes/kubernetes/pull/48625), [@mrhohn](https://github.com/mrhohn))
  * Special notice for kube-proxy in 1.7+ (including 1.7.0):
    * Healthz server (/healthz) will be served on 0.0.0.0:10256 by default.
    * Metrics server (/metrics and /proxyMode) will be served on 127.0.0.1:10249 by default.
    * Metrics server will continue serving /healthz.


# v1.7.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes.tar.gz) | `35281f3552ec4bdf0c219bb7d25b22033648a81e3726594d25500418653eb2f0`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-src.tar.gz) | `450ab45c9d69b12ca9d658247ace8fc67fa02a658fbb474f2a7deae85ebff223`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-darwin-386.tar.gz) | `9fc3629c9eee02008cda0a1045d8a80d6c4ede057e989bdb9c187630c8977438`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-darwin-amd64.tar.gz) | `c163afbf8effd3f1ae041fbcf147f49c478656665158503ddabfb8f64f764bdc`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-linux-386.tar.gz) | `8ec8a0f40a8c7726b2610a30dd4bfa2aef736147a9771234651c1e005e832519`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-linux-amd64.tar.gz) | `9c2363710d61a12a28df2d8a4688543b785156369973d33144ab1f2c1d5c7b53`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-linux-arm64.tar.gz) | `320e89b12fd59863ad64bb49f0a208aba98064f5ead0fe43945f7c5b3fc260e9`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-linux-arm.tar.gz) | `08566e8f7d200d4d23c59947a66b2737122bffd897e8079f056b76d39156167c`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-linux-ppc64le.tar.gz) | `681842ae5f8364be1a0dcdb0703958e450ec9c46eb7bf875a86bc3d6b21a9bb0`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-linux-s390x.tar.gz) | `a779720a07fa22bdaf0e28d93e6a946f479ce408ec25644a3b45aeb03cd04cc8`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-windows-386.tar.gz) | `3fe1e082176e09aba62b6414f5fb4ea8d43880ab04766535ae68e6500c868764`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-client-windows-amd64.tar.gz) | `1ddbdc59bd97b044b63a46da175a5e5298b8947cc49511e3b378d0298736c66d`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-server-linux-amd64.tar.gz) | `b281a1b0ff2f0f38e88642d492e184aa087a985baf54bcaae588948e675d96a3`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-server-linux-arm64.tar.gz) | `2b87266d43f7e38e8d7328b923ee75adba0fc64a2299851a8e915b9321f66e3d`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-server-linux-arm.tar.gz) | `3f00de82ba4d623fbec8f05fc9b249435671a2f6f976654ea5f1f839dca1f804`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-server-linux-ppc64le.tar.gz) | `4b70ff24a6bf9c3d9f58c51fe60a279ac3ce8d996708a4bf58295fa740168b27`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-server-linux-s390x.tar.gz) | `83da55f793bbd040f7282cb155ce219bf1039195f53762098633c44a6971b759`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-node-linux-amd64.tar.gz) | `ecee3f66f62ff87a1718ee7279b720f411fba1b4439255664364e3c5968207b5`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-node-linux-arm64.tar.gz) | `d03252370caa631afd5710e5d40ff35b1e0764bc19a911f3e3f6c9c300b2e354`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-node-linux-arm.tar.gz) | `e1885e36ca699c7ed75a2212d7e8be4482c544ea80e0a229b32703e3efd16ddc`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-node-linux-ppc64le.tar.gz) | `6a3fdc63c1fbcd66440dba4f8252a26959cb42ac92298d12c447c7f3d8d7cc29`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-node-linux-s390x.tar.gz) | `8b2eabb3cee1b990c75835a80ce3429d2a2a7bae7e90916f64efda131da70eaa`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.2/kubernetes-node-windows-amd64.tar.gz) | `8f563627db05d6f12a2034bb01961b012dcadcec17d3bc399d05b6837340d3b3`

## Changelog since v1.7.1

### Other notable changes

* Use port 20256 for node-problem-detector in standalone mode. ([#49316](https://github.com/kubernetes/kubernetes/pull/49316), [@ajitak](https://github.com/ajitak))
* GCE Cloud Provider: New created LoadBalancer type Service will have health checks for nodes by default if all nodes have version >= v1.7.2. ([#49330](https://github.com/kubernetes/kubernetes/pull/49330), [@MrHohn](https://github.com/MrHohn))
* Azure PD (Managed/Blob) ([#46360](https://github.com/kubernetes/kubernetes/pull/46360), [@khenidak](https://github.com/khenidak))
* Fix Pods using Portworx volumes getting stuck in ContainerCreating phase. ([#48898](https://github.com/kubernetes/kubernetes/pull/48898), [@harsh-px](https://github.com/harsh-px))
* kubeadm: Make kube-proxy tolerate the external cloud provider taint so that an external cloud provider can be easily used on top of kubeadm ([#49017](https://github.com/kubernetes/kubernetes/pull/49017), [@luxas](https://github.com/luxas))
* Fix pods failing to start when subPath is a dangling symlink from kubelet point of view, which can happen if it is running inside a container ([#48555](https://github.com/kubernetes/kubernetes/pull/48555), [@redbaron](https://github.com/redbaron))
* Never prevent deletion of resources as part of namespace lifecycle ([#48733](https://github.com/kubernetes/kubernetes/pull/48733), [@liggitt](https://github.com/liggitt))
* kubectl: Fix bug that showed terminated/evicted pods even without `--show-all`. ([#48786](https://github.com/kubernetes/kubernetes/pull/48786), [@janetkuo](https://github.com/janetkuo))
* Add a runtime warning about the kubeadm default token TTL changes. ([#48838](https://github.com/kubernetes/kubernetes/pull/48838), [@mattmoyer](https://github.com/mattmoyer))
* Local storage teardown fix ([#48402](https://github.com/kubernetes/kubernetes/pull/48402), [@ianchakeres](https://github.com/ianchakeres))
* Fix udp service blackhole problem when number of backends changes from 0 to non-0 ([#48524](https://github.com/kubernetes/kubernetes/pull/48524), [@freehan](https://github.com/freehan))
* hpa: Prevent scaling below MinReplicas if desiredReplicas is zero ([#48997](https://github.com/kubernetes/kubernetes/pull/48997), [@johanneswuerbach](https://github.com/johanneswuerbach))
* kubeadm: Fix a bug where `kubeadm join` would wait 5 seconds without doing anything. Now `kubeadm join` executes the tasks immediately. ([#48737](https://github.com/kubernetes/kubernetes/pull/48737), [@mattmoyer](https://github.com/mattmoyer))
* Fix a regression that broke the `--config` flag for `kubeadm init`. ([#48915](https://github.com/kubernetes/kubernetes/pull/48915), [@mattmoyer](https://github.com/mattmoyer))
* Fix service controller crash loop when Service with GCP LoadBalancer uses static IP ([#48848](https://github.com/kubernetes/kubernetes/pull/48848), [@nicksardo](https://github.com/nicksardo)) ([#48849](https://github.com/kubernetes/kubernetes/pull/48849), [@nicksardo](https://github.com/nicksardo))



# v1.7.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes.tar.gz) | `76bddfd19a50f92136456af5bbc3a9d4239260c0c40dccfe704156286a93127c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-src.tar.gz) | `159100f6506c4d59d640a3b0fc7691c4a5023b346d7c3911c5cbbedce2ad8184`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-darwin-386.tar.gz) | `340ceb858bff489fa7ae15c6b526c4316d9c7b6ca354f68ff187c8b5eff08f45`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-darwin-amd64.tar.gz) | `1f1db50d57750115abd6e6e060c914292af7a6e2933a48ccf28ebbe8942c7826`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-linux-386.tar.gz) | `5eac1c92aee40cd2ef14248639d39d7cee910f077dd006a868c510116852fbba`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-linux-amd64.tar.gz) | `6b807520a69b8432baaa89304e8d1ff286d07af20e2a3712b8b2e38d61dbb445`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-linux-arm64.tar.gz) | `a91e0ea4381f659f60380b5b9d6f8114e13337f90a32bcb4a72b8168caef2e00`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-linux-arm.tar.gz) | `6e0e2e557d4e3df18e967e6025a36205aae5b8979dcbb33df6d6e44d9224809a`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-linux-ppc64le.tar.gz) | `22264e96ceaa2d853120be7dcbdc70a9938915cd10eaf5a2c75f4fb2dd12a2eb`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-linux-s390x.tar.gz) | `9b5ac9a66df99a2a8abdc908ef3cd933010facf4c08e96597e041fc359a62aa9`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-windows-386.tar.gz) | `bd3f99ead21f6c6c34dba7ef5c2d2308ef6770bcb255f286d9d5edbf33f5ccff`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-client-windows-amd64.tar.gz) | `e2578ca743bf03b367c473c32657cbed4cf27a12545841058f8bb873fb70e872`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-server-linux-amd64.tar.gz) | `467201c89d473bdec82a67c9b24453a2037eef1a1ed552f0dc55310355d21ea3`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-server-linux-arm64.tar.gz) | `1c1c5cad62423655b1e79bc831de5765cbe683aeef4efe9a823d2597334e19c1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-server-linux-arm.tar.gz) | `17eee900df8ac9bbdd047b2f7d7cb2684820f71cb700dcb305e986acbddf66eb`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-server-linux-ppc64le.tar.gz) | `b1ae5f6d728cfe61b38acbc081e66ddf77ecc38ebdfdb42bfdd53e51fcd3aa2b`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-server-linux-s390x.tar.gz) | `20a273b20b10233fc2632d8a65e0b123fc87166e1f50171e7ede76c59f3118cd`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-node-linux-amd64.tar.gz) | `da0e6d5d6532ef7dba6e5db59e5bc142a52a0314bbb2c70e1fa8e73fe07d0e31`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-node-linux-arm64.tar.gz) | `939b6f779257671a141ecb243bc01e9a5dfb1cd05808820044d915049c3f591a`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-node-linux-arm.tar.gz) | `512fddbbb7353d6dd02e51e79e05101ab857c09e4a4970404258c783ab094c95`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-node-linux-ppc64le.tar.gz) | `795150d92ef93aa53be2db245b9f88cc40fe0fd27045835a23c8eee830c419ba`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-node-linux-s390x.tar.gz) | `58c9b1ef8f8b30fd7061ac87e60b7be9eb79b5bd50c2eef1564838768e7b1d02`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.1/kubernetes-node-windows-amd64.tar.gz) | `eae772609aa50d6a1f4f7cf6df5df2f56cbd438b9034f9be622bc0cfe1d13072`

## Changelog since v1.7.0

### Other notable changes

* Added new flag to `kubeadm init`: --node-name, that lets you specify the name of the Node object that will be created ([#48594](https://github.com/kubernetes/kubernetes/pull/48594), [@GheRivero](https://github.com/GheRivero))
* Added new flag to `kubeadm join`: --node-name, that lets you specify the name of the Node object that's gonna be created ([#48538](https://github.com/kubernetes/kubernetes/pull/48538), [@GheRivero](https://github.com/GheRivero))
* Fixes issue where you could not mount NFS or glusterFS volumes using hostnames on GCI/GKE with COS images. ([#42376](https://github.com/kubernetes/kubernetes/pull/42376), [@jingxu97](https://github.com/jingxu97))
* Reduce amount of noise in Stackdriver Logging, generated by the event-exporter component in the fluentd-gcp addon. ([#48712](https://github.com/kubernetes/kubernetes/pull/48712), [@crassirostris](https://github.com/crassirostris))
* Add generic NoSchedule toleration to fluentd in gcp config. ([#48182](https://github.com/kubernetes/kubernetes/pull/48182), [@gmarek](https://github.com/gmarek))
* RBAC role and role-binding reconciliation now ensures namespaces exist when reconciling on startup. ([#48480](https://github.com/kubernetes/kubernetes/pull/48480), [@liggitt](https://github.com/liggitt))
* Support NoSchedule taints correctly in DaemonSet controller. ([#48189](https://github.com/kubernetes/kubernetes/pull/48189), [@mikedanese](https://github.com/mikedanese))
* kubeadm: Expose only the cluster-info ConfigMap in the kube-public ns ([#48050](https://github.com/kubernetes/kubernetes/pull/48050), [@luxas](https://github.com/luxas))



# v1.8.0-alpha.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.8.0-alpha.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes.tar.gz) | `26d8079fa6b2d82682db809827d260bbab8e6d0f45e457260b8c5ce640432426`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-src.tar.gz) | `141e5c1bf66b69f3c22870b2ab6159abc3b38c12cc20f41c8193044e16df3205`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `6ca63da27ca0c1efa04d079d90eba7e6f01a6e9581317892538be6a97ee64d95`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `0bfbd97f7fb7ce5e1228134d8ca40168553d179bfa44cbd5e925a6543fb3bbf5`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `29d395cc61c91c602e32412e51d4eae333942e6b9da235270768d11c040733c3`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `b1172bbb1d80ba29612d4de08dc4942b40b0f7d580dbb8ed4423c221f78920fe`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `994621c4a9d0644e3e8a4f12f563588036412bb72f0104b888f7a2605d3a8015`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `1e0dd9e4e9730a8cd54d8eb7036d5d7307bd930a91d0fcb105601b2d03fda15d`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `bdcf58f419b42d83ce8adb350388c962b8934782294f9715b617cdbdf201cc36`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `5c58217cffb34043fae951222bfd429165c68439f590c8fb8e33e54fe1cab0de`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `f78ec125f734433c9fc75a9d35dc7bdfa6d145f1cc071ff2e3a5435beef3368f`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `78dca9aadc140e2868b0a3d1a77b5058e22f24710f9c7956d755b473b575bb9d`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `802bb71cf19147857a50e842a00d50641f78fec5c5791a524639f7af70f9e1d4`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `b8f15c32320188981d5e75c474d4e826e45f59083eb66304151da112fb3052b1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `8f800befc32d8402a581c47254db921d54caa31c50513c257b251435756918f1`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `a406bd0aaa92633dbb43216312971164b0230ea01c77679d12b9ffc873956d0d`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `8e038b4ccdfc89a08204927c8097a51bd9e786a97c2f9d73fca763ebee6c2373`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `1a9725cfb55991680fc75cb862d8a74d76f453be9e9f8ad043d62d5911ab50b9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `44fbdd86048bea2cb3d2d6ec1b6cb2c4ae19cb32f6df28e15392cd7f028a4350`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `76d9d36aa182fb93aab7a01f22f7a008ad2906a6224b4c009074100676403337`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `07327ce6fe78bbae3d34b185b54ea0204bf875df488f0293ee1271599189160d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `e84a8c638834c435f82560b86f1a14ec861a8fc967a7cd7055ab86526ce744d0`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `f0f69dc70751e3be2d564aa272f7fe67e86e91c7de3034776b98faddef51a73d`

## Changelog since v1.7.0

### Action Required

* The deprecated ThirdPartyResource (TPR) API has been removed. To avoid losing your TPR data, you must [migrate to CustomResourceDefinition](https://kubernetes.io/docs/tasks/access-kubernetes-api/migrate-third-party-resource/) before upgrading. ([#48353](https://github.com/kubernetes/kubernetes/pull/48353), [@deads2k](https://github.com/deads2k))

### Other notable changes

* Removed scheduler dependencies to testapi. ([#48405](https://github.com/kubernetes/kubernetes/pull/48405), [@k82cn](https://github.com/k82cn))
* kubeadm: Fix a bug where `kubeadm join` would wait 5 seconds without doing anything. Now `kubeadm join` executes the tasks immediately. ([#48737](https://github.com/kubernetes/kubernetes/pull/48737), [@mattmoyer](https://github.com/mattmoyer))
* Reduce amount of noise in Stackdriver Logging, generated by the event-exporter component in the fluentd-gcp addon. ([#48712](https://github.com/kubernetes/kubernetes/pull/48712), [@crassirostris](https://github.com/crassirostris))
* To allow the userspace proxy to work correctly on multi-interface hosts when using the non-default-route interface, you may now set the `bindAddress` configuration option to an IP address assigned to a network interface.  The proxy will use that IP address for any required NAT operations instead of the IP address of the interface which has the default route. ([#48613](https://github.com/kubernetes/kubernetes/pull/48613), [@dcbw](https://github.com/dcbw))
* Move Mesos Cloud Provider out of Kubernetes Repo ([#47232](https://github.com/kubernetes/kubernetes/pull/47232), [@gyliu513](https://github.com/gyliu513))
* - kubeadm now can accept versions like "1.6.4" where previously it strictly required "v1.6.4" ([#48507](https://github.com/kubernetes/kubernetes/pull/48507), [@kad](https://github.com/kad))
* kubeadm: Implementing the certificates phase fully ([#48196](https://github.com/kubernetes/kubernetes/pull/48196), [@fabriziopandini](https://github.com/fabriziopandini))
* Added case on 'terminated-but-not-yet-deleted' for Admit. ([#48322](https://github.com/kubernetes/kubernetes/pull/48322), [@k82cn](https://github.com/k82cn))
* `kubectl run --env` no longer supports CSV parsing. To provide multiple env vars, use the `--env` flag multiple times instead of having env vars separated by commas. E.g. `--env ONE=1 --env TWO=2` instead of `--env ONE=1,TWO=2`. ([#47460](https://github.com/kubernetes/kubernetes/pull/47460), [@mengqiy](https://github.com/mengqiy))
* Local storage teardown fix ([#48402](https://github.com/kubernetes/kubernetes/pull/48402), [@ianchakeres](https://github.com/ianchakeres))
* support json output for log backend of advanced audit ([#48605](https://github.com/kubernetes/kubernetes/pull/48605), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Requests with the query parameter `?watch=` are treated by the API server as a request to watch, but authorization and metrics were not correctly identifying those as watch requests, instead grouping them as list calls. ([#48583](https://github.com/kubernetes/kubernetes/pull/48583), [@smarterclayton](https://github.com/smarterclayton))
* As part of the NetworkPolicy "v1" changes, it is also now ([#47123](https://github.com/kubernetes/kubernetes/pull/47123), [@danwinship](https://github.com/danwinship))
    * possible to update the spec field of an existing
    * NetworkPolicy. (Previously you had to delete and recreate a
    * NetworkPolicy if you wanted to change it.)
* Fix udp service blackhole problem when number of backends changes from 0 to non-0 ([#48524](https://github.com/kubernetes/kubernetes/pull/48524), [@freehan](https://github.com/freehan))
* kubeadm: Make self-hosting work by using DaemonSets and split it out to a phase that can be invoked via the CLI ([#47435](https://github.com/kubernetes/kubernetes/pull/47435), [@luxas](https://github.com/luxas))
* Added new flag to `kubeadm join`: --node-name, that lets you specify the name of the Node object that's gonna be created ([#48538](https://github.com/kubernetes/kubernetes/pull/48538), [@GheRivero](https://github.com/GheRivero))
* Fix Audit-ID header key ([#48492](https://github.com/kubernetes/kubernetes/pull/48492), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Checked container spec when killing container. ([#48194](https://github.com/kubernetes/kubernetes/pull/48194), [@k82cn](https://github.com/k82cn))
* Fix kubectl describe for pods with controllerRef  ([#45467](https://github.com/kubernetes/kubernetes/pull/45467), [@ddysher](https://github.com/ddysher))
* Skip errors when unregistering juju kubernetes-workers ([#48144](https://github.com/kubernetes/kubernetes/pull/48144), [@ktsakalozos](https://github.com/ktsakalozos))
* Configures the Juju Charm code to run kube-proxy with conntrack-max-per-core set to 0 when in an lxc as a workaround for issues when mounting /sys/module/nf_conntrack/parameters/hashsize ([#48450](https://github.com/kubernetes/kubernetes/pull/48450), [@wwwtyro](https://github.com/wwwtyro))
* Group and order imported packages. ([#48399](https://github.com/kubernetes/kubernetes/pull/48399), [@k82cn](https://github.com/k82cn))
* RBAC role and role-binding reconciliation now ensures namespaces exist when reconciling on startup. ([#48480](https://github.com/kubernetes/kubernetes/pull/48480), [@liggitt](https://github.com/liggitt))
* Fix charms leaving services running after remove-unit ([#48446](https://github.com/kubernetes/kubernetes/pull/48446), [@Cynerva](https://github.com/Cynerva))
* Added helper funcs to schedulercache.Resource. ([#46926](https://github.com/kubernetes/kubernetes/pull/46926), [@k82cn](https://github.com/k82cn))
* When performing a GET then PUT, the kube-apiserver must write the canonical representation of the object to etcd if the current value does not match. That allows external agents to migrate content in etcd from one API version to another, across different storage types, or across varying encryption levels. This fixes a bug introduced in 1.5 where we unintentionally stopped writing the newest data. ([#48394](https://github.com/kubernetes/kubernetes/pull/48394), [@smarterclayton](https://github.com/smarterclayton))
* Fixed kubernetes charms not restarting services after snap upgrades ([#48440](https://github.com/kubernetes/kubernetes/pull/48440), [@Cynerva](https://github.com/Cynerva))
* Fix: namespace-create have kubectl in path ([#48439](https://github.com/kubernetes/kubernetes/pull/48439), [@ktsakalozos](https://github.com/ktsakalozos))
* add validate for advanced audit policy ([#47784](https://github.com/kubernetes/kubernetes/pull/47784), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Support NoSchedule taints correctly in DaemonSet controller. ([#48189](https://github.com/kubernetes/kubernetes/pull/48189), [@mikedanese](https://github.com/mikedanese))
* Adds configuration option for Swift object store container name to OpenStack Heat provider. ([#48281](https://github.com/kubernetes/kubernetes/pull/48281), [@hogepodge](https://github.com/hogepodge))
* Allow the system:heapster ClusterRole read access to deployments ([#48357](https://github.com/kubernetes/kubernetes/pull/48357), [@faraazkhan](https://github.com/faraazkhan))
* Ensure get_password is accessing a file that exists. ([#48351](https://github.com/kubernetes/kubernetes/pull/48351), [@ktsakalozos](https://github.com/ktsakalozos))
* GZip openapi schema if accepted by client ([#48151](https://github.com/kubernetes/kubernetes/pull/48151), [@apelisse](https://github.com/apelisse))
* Fixes issue where you could not mount NFS or glusterFS volumes using hostnames on GCI/GKE with COS images. ([#42376](https://github.com/kubernetes/kubernetes/pull/42376), [@jingxu97](https://github.com/jingxu97))
* Previously a deleted service account token secret would be considered valid until it was reaped.  Now it is invalid as soon as the deletionTimestamp is set. ([#48343](https://github.com/kubernetes/kubernetes/pull/48343), [@deads2k](https://github.com/deads2k))
* Securing the cluster created by Juju ([#47835](https://github.com/kubernetes/kubernetes/pull/47835), [@ktsakalozos](https://github.com/ktsakalozos))
* addon-resizer flapping behavior was removed. ([#46850](https://github.com/kubernetes/kubernetes/pull/46850), [@x13n](https://github.com/x13n))
* Change default `httpGet` probe `User-Agent` to `kube-probe/<version major.minor>` if none specified, overriding the default Go `User-Agent`. ([#47729](https://github.com/kubernetes/kubernetes/pull/47729), [@paultyng](https://github.com/paultyng))
* Registries backed by the generic Store's `Update` implementation support delete-on-update, which allows resources to be automatically deleted during an update provided: ([#48065](https://github.com/kubernetes/kubernetes/pull/48065), [@ironcladlou](https://github.com/ironcladlou))
        * Garbage collection is enabled for the Store
        * The resource being updated has no finalizers
        * The resource being updated has a non-nil DeletionGracePeriodSeconds equal to 0
    * With this fix, Custom Resource instances now also support delete-on-update behavior under the same circumstances.
* Fixes an edge case where "kubectl apply view-last-applied" would emit garbage if the data contained Go format codes. ([#45611](https://github.com/kubernetes/kubernetes/pull/45611), [@atombender](https://github.com/atombender))
* Bumped Heapster to v1.4.0. ([#48205](https://github.com/kubernetes/kubernetes/pull/48205), [@piosz](https://github.com/piosz))
    * More details about the release https://github.com/kubernetes/heapster/releases/tag/v1.4.0
* In GCE and in a "private master" setup, do not set the network-plugin provider to CNI by default if a network policy provider is given. ([#48004](https://github.com/kubernetes/kubernetes/pull/48004), [@dnardo](https://github.com/dnardo))
* Add generic NoSchedule toleration to fluentd in gcp config. ([#48182](https://github.com/kubernetes/kubernetes/pull/48182), [@gmarek](https://github.com/gmarek))
* kubeadm: Expose only the cluster-info ConfigMap in the kube-public ns ([#48050](https://github.com/kubernetes/kubernetes/pull/48050), [@luxas](https://github.com/luxas))
* Fixes kubelet race condition in container manager. ([#48123](https://github.com/kubernetes/kubernetes/pull/48123), [@msau42](https://github.com/msau42))
* Bump GCE ContainerVM to container-vm-v20170627 ([#48159](https://github.com/kubernetes/kubernetes/pull/48159), [@zmerlynn](https://github.com/zmerlynn))
* Add PriorityClassName and Priority fields to PodSpec. ([#45610](https://github.com/kubernetes/kubernetes/pull/45610), [@bsalamat](https://github.com/bsalamat))
* Add a failsafe for etcd not returning a connection string ([#48054](https://github.com/kubernetes/kubernetes/pull/48054), [@ktsakalozos](https://github.com/ktsakalozos))
* Fix fluentd-gcp configuration to facilitate JSON parsing ([#48139](https://github.com/kubernetes/kubernetes/pull/48139), [@crassirostris](https://github.com/crassirostris))
* Setting env var ENABLE_BIG_CLUSTER_SUBNETS=true will allow kube-up.sh to start clusters bigger that 4095 Nodes on GCE. ([#47513](https://github.com/kubernetes/kubernetes/pull/47513), [@gmarek](https://github.com/gmarek))
* When determining the default external host of the kube apiserver, any configured cloud provider is now consulted ([#47038](https://github.com/kubernetes/kubernetes/pull/47038), [@yastij](https://github.com/yastij))
* Updated comments for functions. ([#47242](https://github.com/kubernetes/kubernetes/pull/47242), [@k82cn](https://github.com/k82cn))
* Fix setting juju worker labels during deployment ([#47178](https://github.com/kubernetes/kubernetes/pull/47178), [@ktsakalozos](https://github.com/ktsakalozos))
* `kubefed init` correctly checks for RBAC API enablement. ([#48077](https://github.com/kubernetes/kubernetes/pull/48077), [@liggitt](https://github.com/liggitt))
* The garbage collector now cascades deletion properly when deleting an object with propagationPolicy="background". This resolves issue [[#44046](https://github.com/kubernetes/kubernetes/pull/44046)](https://github.com/kubernetes/kubernetes/issues/44046), so that when a deployment is deleted with propagationPolicy="background", the garbage collector ensures dependent pods are deleted as well. ([#44058](https://github.com/kubernetes/kubernetes/pull/44058), [@caesarxuchao](https://github.com/caesarxuchao))
* Fix restart action on juju kubernetes-master ([#47170](https://github.com/kubernetes/kubernetes/pull/47170), [@ktsakalozos](https://github.com/ktsakalozos))
* e2e: bump kubelet's resurce usage limit ([#47971](https://github.com/kubernetes/kubernetes/pull/47971), [@yujuhong](https://github.com/yujuhong))
* Cluster Autoscaler 0.6 ([#48074](https://github.com/kubernetes/kubernetes/pull/48074), [@mwielgus](https://github.com/mwielgus))
* Checked whether balanced Pods were created. ([#47488](https://github.com/kubernetes/kubernetes/pull/47488), [@k82cn](https://github.com/k82cn))
* Update protobuf time serialization for a one second granularity ([#47975](https://github.com/kubernetes/kubernetes/pull/47975), [@deads2k](https://github.com/deads2k))
* Bumped Heapster to v1.4.0-beta.0 ([#47961](https://github.com/kubernetes/kubernetes/pull/47961), [@piosz](https://github.com/piosz))
* `kubectl api-versions` now always fetches information about enabled API groups and versions instead of using the local cache. ([#48016](https://github.com/kubernetes/kubernetes/pull/48016), [@liggitt](https://github.com/liggitt))
* Removes alpha feature gate for affinity annotations.   ([#47869](https://github.com/kubernetes/kubernetes/pull/47869), [@timothysc](https://github.com/timothysc))
* Websocket requests may now authenticate to the API server by passing a bearer token in a websocket subprotocol of the form `base64url.bearer.authorization.k8s.io.<base64url-encoded-bearer-token>` ([#47740](https://github.com/kubernetes/kubernetes/pull/47740), [@liggitt](https://github.com/liggitt))
* Update cadvisor to v0.26.1 ([#47940](https://github.com/kubernetes/kubernetes/pull/47940), [@Random-Liu](https://github.com/Random-Liu))
* Bump up npd version to v0.4.1 ([#47892](https://github.com/kubernetes/kubernetes/pull/47892), [@ajitak](https://github.com/ajitak))
* Allow StorageClass Ceph RBD to specify image format and image features. ([#45805](https://github.com/kubernetes/kubernetes/pull/45805), [@weiwei04](https://github.com/weiwei04))
* Removed mesos related labels. ([#46824](https://github.com/kubernetes/kubernetes/pull/46824), [@k82cn](https://github.com/k82cn))
* Add RBAC support to fluentd-elasticsearch cluster addon ([#46203](https://github.com/kubernetes/kubernetes/pull/46203), [@simt2](https://github.com/simt2))
* Avoid redundant copying of tars during kube-up for gce if the same file already exists ([#46792](https://github.com/kubernetes/kubernetes/pull/46792), [@ianchakeres](https://github.com/ianchakeres))
* container runtime version has been added to the output of `kubectl get nodes -o=wide` as `CONTAINER-RUNTIME` ([#46646](https://github.com/kubernetes/kubernetes/pull/46646), [@rickypai](https://github.com/rickypai))
* cAdvisor binds only to the interface that kubelet is running on instead of all interfaces. ([#47195](https://github.com/kubernetes/kubernetes/pull/47195), [@dims](https://github.com/dims))
* The schema of the API that are served by the kube-apiserver, together with a small amount of generated code, are moved to k8s.io/api (https://github.com/kubernetes/api). ([#44784](https://github.com/kubernetes/kubernetes/pull/44784), [@caesarxuchao](https://github.com/caesarxuchao))



# v1.7.0

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.0


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes.tar.gz) | `947f1dd9a9b6b427faac84067a30c86e83e6391eb42f09ddcc50a8694765c31a`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-src.tar.gz) | `d3d8b0bfc31164dd703b38d8484cfed7981cacd1e496731880afa87f8bf39aac`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-darwin-386.tar.gz) | `da298e24318e57ac8a558c390117bd7e9e596b3bdf1c5960979898fefe6c5c88`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-darwin-amd64.tar.gz) | `c22f72e1592731155db5b05d0d660f1d7314288cb020f7980e2a109d9e7ba0e5`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-linux-386.tar.gz) | `fc8e90e96360c3a2c8ec56903ab5acde1dffa4d641e1ee27b804ee6d8e824cf6`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-linux-amd64.tar.gz) | `8b3ed03f8a4b3a1ec124abde01632ee6dcec9daf9376f0288fd7500b5173981c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-linux-arm64.tar.gz) | `8930c74dab9ada31e6994f0dc3fb22d41a602a2880b6b17112718ce73eac0574`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-linux-arm.tar.gz) | `20a6f4645cab3c0aef72f849ae90b2691605fd3f670ce36cc8aa11aef31c6edb`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-linux-ppc64le.tar.gz) | `509e214d55e8df1906894cbdc166e791761a3b82a52bcea0de65ceca3143c8b5`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-linux-s390x.tar.gz) | `fd39f47b691fc608f2ea3fed35408dd4c0b1d198605ec17363b0987b123a4702`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-windows-386.tar.gz) | `d9b72cfeefee0cd2db5f6a388bdb9da1e33514498f4d88be1b04282db5bfbd3d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-client-windows-amd64.tar.gz) | `c536952bd29a7ae12c8fa148d592cc3c353dea4d0079e8497edaf8a759a16006`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-server-linux-amd64.tar.gz) | `175fc9360d4f26b5f60b467798d851061f01d0ca555c254ef44a8a9822cf7560`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-server-linux-arm64.tar.gz) | `f1e039e0e2923d1ea02fd76453aa51715ca83c5c26ca1a761ace2c717b79154f`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-server-linux-arm.tar.gz) | `48dc95e5230d7a44b64b379f9cf2e1ec72b7c4c7c62f4f3e92a73076ad6376db`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-server-linux-ppc64le.tar.gz) | `dc079cd18333c201cfd0f5b0e93e602d020a9e665d8c13968170a2cd89eebeb4`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-server-linux-s390x.tar.gz) | `fe6674e7d69aeffd522e543e957897e2cb943e82d5ccd368ccb9009e1128273f`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-node-linux-amd64.tar.gz) | `6c6cece62bad5bfeaf4a4b14e93c9ba99c96dc82b7855a2214cdf37a65251de8`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-node-linux-arm64.tar.gz) | `dd75dc044fb1f337b60cb4b27c9bbdca4742d8bc0a1d03d13553a1b8fc593e98`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-node-linux-arm.tar.gz) | `c5d832c93c24d77414a880d8b7c4fac9a7443305e8e5c704f637ff023ff56f94`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-node-linux-ppc64le.tar.gz) | `649813a257353c5b85605869e33aeeb0c070e64e6fee18bc9c6e70472aa05677`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-node-linux-s390x.tar.gz) | `5ca0a7e9e90b2de7aff7bbdc84f662140ce847ea46cdb78802ce75459e0cc043`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0/kubernetes-node-windows-amd64.tar.gz) | `4b84b0025aff1d4406f3e5cd5fa86940f594e3ec6e1d12d3ce1eea5f5b3fc55d`

## **Major Themes**

Kubernetes 1.7 is a milestone release that adds security, stateful application, and extensibility features motivated by widespread production use of Kubernetes.

Security enhancements in this release include encrypted secrets (alpha), network policy for pod-to-pod communication, the node authorizer to limit Kubelet access to API resources, and Kubelet client / server TLS certificate rotation (alpha).  

Major features for stateful applications include automated updates to StatefulSets, enhanced updates for DaemonSets, a burst mode for faster StatefulSets scaling, and (alpha) support for local storage.

Extensibility features include API aggregation (beta), CustomResourceDefinitions (beta) in favor of ThirdPartyResources, support for extensible admission controllers (alpha), pluggable cloud providers (alpha), and container runtime interface (CRI) enhancements.

## **Action Required Before Upgrading**

### Network

* NetworkPolicy has been promoted from extensions/v1beta1 to the new networking.k8s.io/v1 API group. The structure remains unchanged from the v1beta1 API. The net.beta.kubernetes.io/network-policy annotation on Namespaces (used to opt in to isolation) has been removed. Instead, isolation is now determined on a per-pod basis. A NetworkPolicy may target a pod for isolation by including the pod in its spec.podSelector. Targeted Pods accept the traffic specified in the respective NetworkPolicy (and nothing else). Pods not targeted by any NetworkPolicy accept all traffic by default. ([#39164](https://github.com/kubernetes/kubernetes/pull/39164), [@danwinship](https://github.com/danwinship))

	**Action Required:** When upgrading to Kubernetes 1.7 (and a [network plugin](https://kubernetes.io/docs/tasks/administer-cluster/declare-network-policy/) that supports the new NetworkPolicy v1 semantics), you should consider the following.

	The v1beta1 API used an annotation on Namespaces to activate the DefaultDeny policy for an entire Namespace. To activate default deny in the v1 API, you can create a NetworkPolicy that matches all Pods but does not allow any traffic:

    ```yaml
    kind: NetworkPolicy
    apiVersion: networking.k8s.io/v1
    metadata:
      name: default-deny
    spec:
      podSelector:
    ```

	This will ensure that Pods that aren't matched by any other NetworkPolicy will continue to be fully-isolated, as they were in v1beta1.

	In Namespaces that previously did not have the "DefaultDeny" annotation, you should delete any existing NetworkPolicy objects. These had no effect in the v1beta1 API, but with v1 semantics they might cause some traffic to be unintentionally blocked.


### Storage

* Alpha volume provisioning is removed and default storage class should be used instead. ([#44090](https://github.com/kubernetes/kubernetes/pull/44090), [@NickrenREN](https://github.com/NickrenREN))

* Portworx volume driver no longer has to run on the master. ([#45518](https://github.com/kubernetes/kubernetes/pull/45518), [@harsh-px](https://github.com/harsh-px))

* Default behavior in Cinder storageclass is changed. If availability is not specified, the zone is chosen by algorithm. It makes possible to spread stateful pods across many zones. ([#44798](https://github.com/kubernetes/kubernetes/pull/44798), [@zetaab](https://github.com/zetaab))

* PodSpecs containing parent directory references such as `..` (for example, `../bar`) in hostPath volume path or in volumeMount subpaths must be changed to the simple absolute path. Backsteps `..` are no longer allowed.([#47290](https://github.com/kubernetes/kubernetes/pull/47290), [@jhorwit2](https://github.com/jhorwit2)).


### API Machinery

* The Namespace API object no longer supports the deletecollection operation. ([#46407](https://github.com/kubernetes/kubernetes/pull/46407), [@liggitt](https://github.com/liggitt))

* The following alpha API groups were unintentionally enabled by default in previous releases, and will no longer be enabled by default in v1.8: ([#47690](https://github.com/kubernetes/kubernetes/pull/47690), [@caesarxuchao](https://github.com/caesarxuchao))

    * rbac.authorization.k8s.io/v1alpha1

    * settings.k8s.io/v1alpha1

    * If you wish to continue using them in v1.8, please enable them explicitly using the `--runtime-config` flag on the apiserver (for example, `--runtime-config="rbac.authorization.k8s.io/v1alpha1,settings.k8s.io/v1alpha1"`)

* `cluster/update-storage-objects.sh` now supports updating StorageClasses in etcd to storage.k8s.io/v1. You must do this prior to upgrading to 1.8. ([#46116](https://github.com/kubernetes/kubernetes/pull/46116), [@ncdc](https://github.com/ncdc))


### Controller Manager

* kube-controller-manager has dropped support for the `--insecure-experimental-approve-all-kubelet-csrs-for-group` flag. It is accepted in 1.7, but ignored. Instead, the csrapproving controller uses authorization checks to determine whether to approve certificate signing requests: ([#45619](https://github.com/kubernetes/kubernetes/pull/45619), [@mikedanese](https://github.com/mikedanese))

    * Before upgrading, users must ensure their controller manager will enable the csrapproving controller, create an RBAC ClusterRole and ClusterRoleBinding to approve CSRs for the same group, then upgrade. Example roles to enable the equivalent behavior can be found in the [TLS bootstrapping](https://kubernetes.io/docs/admin/kubelet-tls-bootstrapping/) documentation.


### kubectl (CLI)
* `kubectl create role` and  `kubectl create clusterrole`  invocations must be updated to specify multiple resource names as repeated  `--resource-name` arguments instead of comma-separated arguments to a single `--resource-name` argument. E.g. `--resource-name=x,y` must become `--resource-name x --resource-name y` ([#44950](https://github.com/kubernetes/kubernetes/pull/44950), [@xilabao](https://github.com/xilabao))

* `kubectl create rolebinding` and `kubectl create clusterrolebinding` invocations must be updated to  specify multiple subjects as repeated  `--user`, `--group`, or `--serviceaccount` arguments instead of comma-separated arguments to a single `--user`, `--group`, or `--serviceaccount`.  E.g. `--user=x,y` must become `--user x --user y`  ([#43903](https://github.com/kubernetes/kubernetes/pull/43903), [@xilabao](https://github.com/xilabao))


### kubeadm

* kubeadm: Modifications to cluster-internal resources installed by kubeadm will be overwritten when upgrading from v1.6 to v1.7. ([#47081](https://github.com/kubernetes/kubernetes/pull/47081), [@luxas](https://github.com/luxas))

* kubeadm deb/rpm packages: cAdvisor doesn't listen on `0.0.0.0:4194` without authentication/authorization because of the possible information leakage. The cAdvisor API can still be accessed via `https://{node-ip}:10250/stats/`, though. ([kubernetes/release#356](https://github.com/kubernetes/release/pull/356), [@luxas](https://github.com/luxas))


### Cloud Providers

* Azure: Container permissions for provisioned volumes have changed to private. If you have existing Azure volumes that were created by Kubernetes v1.6.0-v1.6.5, you should change the permissions on them manually. ([#47605](https://github.com/kubernetes/kubernetes/pull/47605), [@brendandburns](https://github.com/brendandburns))

* GKE/GCE: New and upgraded 1.7 GCE/GKE clusters no longer have an RBAC ClusterRoleBinding that grants the cluster-admin ClusterRole to the default service account in the kube-system Namespace. ([#46750](https://github.com/kubernetes/kubernetes/pull/46750), [@cjcullen](https://github.com/cjcullen)). If this permission is still desired, run the following command to explicitly grant it, either before or after upgrading to 1.7:
    ```
    kubectl create clusterrolebinding kube-system-default --serviceaccount=kube-system:default --clusterrole=cluster-admin
    ```

## **Known Issues**

Populated via [v1.7.x known issues / FAQ accumulator](https://github.com/kubernetes/kubernetes/issues/46733)

* The kube-apiserver discovery APIs (for example, `/apis`) return information about the API groups being served, and can change dynamically.
During server startup, prior to the server reporting healthy (via `/healthz`), not all API groups may be reported.
Wait for the server to report healthy (via `/healthz`) before depending on the information provided by the discovery APIs.
Additionally, since the information returned from the discovery APIs may change dynamically, a cache of the results should not be considered authoritative.
ETag support is planned in a future version to facilitate client caching.
([#47977](https://github.com/kubernetes/kubernetes/pull/47977), [#44957](https://github.com/kubernetes/kubernetes/pull/44957))

* The DaemonSet controller will evict running Pods that do not tolerate the NoSchedule taint if the taint is added to a Node.  There is an open PR ([#48189](https://github.com/kubernetes/kubernetes/pull/48189)) to resolve this issue, but as this issue also exists in 1.6, and as we do not wish to risk release stability by merging it directly prior to a release without sufficient testing, we have decided to defer merging the PR until the next point release for each minor version ([#48190](https://github.com/kubernetes/kubernetes/pull/48190)).

* Protobuf serialization does not distinguish between `[]` and `null`.
API fields previously capable of storing and returning either `[]` and `null` via JSON API requests (for example, the Endpoints `subsets` field)
can now store only `null` when created using the protobuf content-type or stored in etcd using protobuf serialization (the default in 1.6).
JSON API clients should tolerate `null` values for such fields, and treat `null` and `[]` as equivalent in meaning unless specifically documented otherwise for a particular field. ([#44593](https://github.com/kubernetes/kubernetes/pull/44593))

* Local volume source paths that are directories and not mount points fail to unmount.  A fix is in process ([#48331](https://github.com/kubernetes/kubernetes/issues/48331)).

* Services of type LoadBalancer (on GCE/GKE) that have static IP addresses will cause the Service Controller to panic and thereby causing the kube-controller-manager to crash loop.
([#48848](https://github.com/kubernetes/kubernetes/issues/48848))

## **Deprecations**

### Cluster provisioning scripts
* cluster/ubuntu: Removed due to [deprecation](https://github.com/kubernetes/kubernetes/tree/master/cluster#cluster-configuration) and lack of maintenance. ([#44344](https://github.com/kubernetes/kubernetes/pull/44344), [@mikedanese](https://github.com/mikedanese))

* cluster/aws: Removed due to [deprecation](https://github.com/kubernetes/kubernetes/pull/38772) and lack of maintenance. ([#42196](https://github.com/kubernetes/kubernetes/pull/42196), [@zmerlynn](https://github.com/zmerlynn))


### Client libraries
* Swagger 1.2 spec (`/swaggerapi/*`) is deprecated. Please use OpenAPI instead.

### DaemonSet
* DaemonSet’s spec.templateGeneration has been deprecated.  ([#45924](https://github.com/kubernetes/kubernetes/pull/45924), [@janetkuo](https://github.com/janetkuo))

### kube-proxy
* In 1.7, the kube-proxy component has been converted to use a configuration file. The old flags still work in 1.7, but they are being deprecated and will be removed in a future release. Cluster administrators are advised to switch to using the configuration file, but no action is strictly necessary in 1.7. ([#34727](https://github.com/kubernetes/kubernetes/pull/34727), [@ncdc](https://github.com/ncdc))

### Namespace
* The Namespace API object no longer supports the deletecollection operation. ([#46407](https://github.com/kubernetes/kubernetes/pull/46407), [@liggitt](https://github.com/liggitt))


### Scheduling
* If you are using `AffinityInAnnotations=true` in `--feature-gates`, then the 1.7 release is your last opportunity to convert from specifying affinity/anti-affinity using the scheduler.alpha.kubernetes.io/affinity annotation on Pods, to using the Affinity field of PodSpec. Support for the alpha version of node and pod affinity (which uses the scheduler.alpha.kubernetes.io/affinity annotations on Pods) is going away **in Kubernetes 1.8** (not this release, but the next release). If you have not enabled AffinityInAnnotations=true in `--feature-gates`, then this change does not affect you.

## **Notable Features**

Features for this release were tracked via the use of the [kubernetes/features](https://github.com/kubernetes/features) issues repo. Each Feature issue is owned by a Special Interest Group from [kubernetes/community](https://github.com/kubernetes/community)

## Kubefed

* Deprecate the `--secret-name` flag from `kubefed join`, instead generating the secret name arbitrarily. ([#42513](https://github.com/kubernetes/kubernetes/pull/42513), [@perotinus](https://github.com/perotinus))


### **Kubernetes API**
#### User Provided Extensions
* [beta] ThirdPartyResource is deprecated. Please migrate to the successor, CustomResourceDefinition. For more information, see [Custom Resources](https://kubernetes.io/docs/concepts/api-extension/custom-resources/) and [Migrate a ThirdPartyResource to CustomResourceDefinition](https://kubernetes.io/docs/tasks/access-kubernetes-api/migrate-third-party-resource/).

* [beta] User-provided apiservers can be aggregated (served along with) the rest of the Kubernetes API. See [Extending the Kubernetes API with the aggregation layer](https://kubernetes.io/docs/concepts/api-extension/apiserver-aggregation/), [Configure the aggregation layer](https://kubernetes.io/docs/tasks/access-kubernetes-api/configure-aggregation-layer/), and [Setup an extension API server](https://kubernetes.io/docs/tasks/access-kubernetes-api/setup-extension-api-server/).

* [alpha] Adding admissionregistration API group which enables dynamic registration of initializers and external admission webhooks. ([#46294](https://github.com/kubernetes/kubernetes/pull/46294), [@caesarxuchao](https://github.com/caesarxuchao))


### **Application Deployment**
#### StatefulSet
* [beta] StatefulSet supports [RollingUpdate](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/#rolling-updates) and [OnDelete](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/#on-delete) update strategies.

* [alpha] StatefulSet authors should be able to relax the [ordering](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/#orderedready-pod-management) and [parallelism](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/#parallel-pod-management) policies for software that can safely support rapid, out-of-order changes.

#### DaemonSet
* [beta] DaemonSet supports history and rollback. See [Performing a Rollback on a DaemonSet](https://kubernetes.io/docs/tasks/manage-daemon/rollback-daemon-set/).

#### Deployments
* [beta] Deployments uses a hashing collision avoidance mechanism that ensures new rollouts will not block on hashing collisions anymore. ([kubernetes/features#287](https://github.com/kubernetes/features/issues/287))

#### PodDisruptionBudget
* [beta] PodDisruptionBudget has a new field MaxUnavailable, which allows users to specify the maximum number of disruptions that can be tolerated during eviction. For more information, see [Pod Disruptions](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/) and [Specifying a Disruption Budget for your Application](https://kubernetes.io/docs/tasks/run-application/configure-pdb/).
* PodDisruptionBudget now uses [ControllerRef](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/controller-ref.md) to make the right decisions about Pod eviction even if the built in application controllers have overlapping selectors.

### **Security**
#### Admission Control
* [alpha] Add [extensible external admission control](https://kubernetes.io/docs/admin/extensible-admission-controllers/).

#### TLS Bootstrapping
* [alpha] Rotation of the server TLS certificate on the kubelet. See [TLS bootstrapping - approval controller](https://kubernetes.io/docs/admin/kubelet-tls-bootstrapping/#approval-controller).

* [alpha] Rotation of the client TLS certificate on the kubelet. See [TLS bootstrapping - kubelet configuration](https://kubernetes.io/docs/admin/kubelet-tls-bootstrapping/#kubelet-configuration).

* [beta] [Kubelet TLS Bootstrap](https://kubernetes.io/docs/admin/kubelet-tls-bootstrapping/#kubelet-configuration)

#### Audit Logging
* [alpha] Advanced Auditing enhances the Kubernetes API [audit logging](https://kubernetes.io/docs/tasks/debug-application-cluster/audit/#audit-logs) capabilities through a customizable policy, pluggable audit backends, and richer audit data.

#### Encryption at Rest
* [alpha] Encrypt secrets stored in etcd. For more information, see [Securing a Cluster](https://kubernetes.io/docs/tasks/administer-cluster/securing-a-cluster/) and [Encrypting data at rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/).

#### Node Authorization
* [beta] A new Node authorization mode and NodeRestriction admission plugin, when used in combination, limit nodes' access to specific APIs, so that they may only modify their own Node API object, only modify Pod objects bound to themselves, and only retrieve secrets and configmaps referenced by pods bound to themselves. See [Using Node Authorization](https://kubernetes.io/docs/admin/authorization/node/) for more information.


### **Application Autoscaling**
#### Horizontal Pod Autoscaler
* [alpha] [HPA Status Conditions](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/#appendix-horizontal-pod-autoscaler-status-conditions).


### **Cluster Lifecycle**
#### kubeadm
* [alpha] Manual [upgrades for kubeadm from v1.6 to v1.7](https://kubernetes.io/docs/tasks/administer-cluster/kubeadm-upgrade-1-7/). Automated upgrades ([kubernetes/features#296](https://github.com/kubernetes/features/issues/296)) are targeted for v1.8.

#### Cloud Provider Support
* [alpha] Improved support for out-of-tree and out-of-process cloud providers, a.k.a pluggable cloud providers. See [Build and Run cloud-controller-manager](https://kubernetes.io/docs/tasks/administer-cluster/running-cloud-controller) documentation.


### **Cluster Federation**
#### Placement Policy
* [alpha] The federation-apiserver now supports a SchedulingPolicy admission controller that enables policy-based control over placement of federated resources. For more information, see [Set up placement policies in Federation](https://kubernetes.io/docs/tasks/federation/set-up-placement-policies-federation/).

#### Cluster Selection
* [alpha] Federation [ClusterSelector annotation](https://kubernetes.io/docs/tasks/administer-federation/cluster/#clusterselector-annotation) to direct objects to federated clusters with matching labels.


### **Instrumentation**
#### Core Metrics API
* [alpha] Introduces a lightweight monitoring component for serving the core resource metrics API used by the Horizontal Pod Autoscaler and other components ([kubernetes/features#271](https://github.com/kubernetes/features/issues/271))


### **Internationalization**

* Add Traditional Chinese translation for kubectl ([#46559](https://github.com/kubernetes/kubernetes/pull/46559), [@warmchang](https://github.com/warmchang))

* Add Japanese translation for kubectl ([#46756](https://github.com/kubernetes/kubernetes/pull/46756), [@girikuncoro](https://github.com/girikuncoro))

* Add Simplified Chinese translation for kubectl ([#45573](https://github.com/kubernetes/kubernetes/pull/45573), [@shiywang](https://github.com/shiywang))

### **kubectl (CLI)**
* Features

  * `kubectl logs` supports specifying a container name when using label selectors ([#44282](https://github.com/kubernetes/kubernetes/pull/44282), [@derekwaynecarr](https://github.com/derekwaynecarr))

  * `kubectl rollout` supports undo and history for DaemonSet ([#46144](https://github.com/kubernetes/kubernetes/pull/46144), [@janetkuo](https://github.com/janetkuo))

  * `kubectl rollout` supports status and history for StatefulSet  ([#46669](https://github.com/kubernetes/kubernetes/pull/46669), [@kow3ns](https://github.com/kow3ns)).

  * Implement `kubectl get controllerrevisions` ([#46655](https://github.com/kubernetes/kubernetes/pull/46655), [@janetkuo](https://github.com/janetkuo))

  * `kubectl create clusterrole` supports `--non-resource-url` ([#45809](https://github.com/kubernetes/kubernetes/pull/45809), [@CaoShuFeng](https://github.com/CaoShuFeng))

  * `kubectl logs` and `kubectl attach` support specifying a wait timeout with `--pod-running-timeout` ([#41813](https://github.com/kubernetes/kubernetes/pull/41813), [@shiywang](https://github.com/shiywang))

  * New commands

    * Add `kubectl config rename-context` ([#46114](https://github.com/kubernetes/kubernetes/pull/46114), [@arthur0](https://github.com/arthur0))

    * Add `kubectl apply edit-last-applied` subcommand ([#42256](https://github.com/kubernetes/kubernetes/pull/42256), [@shiywang](https://github.com/shiywang))

  * Strategic Merge Patch

    * Reference docs now display the patch type and patch merge key used by `kubectl apply` to merge and identify unique elements in arrays.

      * `kubectl edit` and `kubectl apply` will keep the ordering of elements in merged lists ([#45980](https://github.com/kubernetes/kubernetes/pull/45980), [@mengqiy](https://github.com/mengqiy))

      * New patch directive (retainKeys) to specifying clearing fields missing from the request ([#44597](https://github.com/kubernetes/kubernetes/pull/44597), [@mengqiy](https://github.com/mengqiy))

      * Open API now includes strategic merge patch tags (previously only in go struct tags) ([#44121](https://github.com/kubernetes/kubernetes/pull/44121), [@mbohlool](https://github.com/mbohlool))

  * Plugins

      * Introduces the ability to extend kubectl by adding third-party plugins. Developer preview, please refer to the documentation for instructions about how to use it. ([#37499](https://github.com/kubernetes/kubernetes/pull/37499), [@fabianofranz](https://github.com/fabianofranz))

      * Added support for a hierarchy of kubectl plugins (a tree of plugins as children of other plugins). ([#45981](https://github.com/kubernetes/kubernetes/pull/45981), [@fabianofranz](https://github.com/fabianofranz))

      * Added exported env vars to kubectl plugins so that plugin developers have access to global flags, namespace, the plugin descriptor and the full path to the caller binary.

  * Enhancement

    * `kubectl auth can-i` now supports non-resource URLs ([#46432](https://github.com/kubernetes/kubernetes/pull/46432), [@CaoShuFeng](https://github.com/CaoShuFeng))

    * `kubectl set selector` and `kubectl set subject` no longer print "running in local/dry-run mode..." at the top.  The output can now be piped and interpretted as yaml or json ([#46507](https://github.com/kubernetes/kubernetes/pull/46507), [@bboreham](https://github.com/bboreham))

    * When using an in-cluster client with an empty configuration, the `--namespace` flag is now honored ([#46299](https://github.com/kubernetes/kubernetes/pull/46299), [@ncdc](https://github.com/ncdc))

    * The help message for missingResourceError is now generic ([#45582](https://github.com/kubernetes/kubernetes/pull/45582), [@CaoShuFeng](https://github.com/CaoShuFeng))

    * `kubectl taint node` now supports label selectors ([#44740](https://github.com/kubernetes/kubernetes/pull/44740), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))

    * `kubectl proxy --www` now logs a warning when the dir is invalid  ([#44952](https://github.com/kubernetes/kubernetes/pull/44952), [@CaoShuFeng](https://github.com/CaoShuFeng))

    * `kubectl taint` output has been enhanced with the operation ([#43171](https://github.com/kubernetes/kubernetes/pull/43171), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))

    * kubectl `--user` and `--cluster` now support completion ([#44251](https://github.com/kubernetes/kubernetes/pull/44251), [@superbrothers](https://github.com/superbrothers))

    * `kubectl config use-context` now supports completion ([#42336](https://github.com/kubernetes/kubernetes/pull/42336), [@superbrothers](https://github.com/superbrothers))

    * `kubectl version` now supports `--output` ([#39858](https://github.com/kubernetes/kubernetes/pull/39858), [@alejandroEsc](https://github.com/alejandroEsc))

		* `kubectl create configmap` has a new option `--from-env-file` that populates a configmap from file which follows a key=val format for each line. ([#38882](https://github.com/kubernetes/kubernetes/pull/38882), [@fraenkel](https://github.com/fraenkel))

		* `kubectl create secret` has a new option `--from-env-file` that populates a configmap from file which follows a key=val format for each line.

  * Printing/describe

    * Print conditions of RC/RS in `kubectl describe` command. ([#44710](https://github.com/kubernetes/kubernetes/pull/44710), [@xiangpengzhao](https://github.com/xiangpengzhao))

    * Improved output on `kubectl get` and `kubectl describe` for generic objects. ([#44222](https://github.com/kubernetes/kubernetes/pull/44222), [@fabianofranz](https://github.com/fabianofranz))

    * In `kubectl describe`, find controllers with ControllerRef, instead of showing the original creator. ([#42849](https://github.com/kubernetes/kubernetes/pull/42849), [@janetkuo](https://github.com/janetkuo))

		* `kubectl version` has new flag --output (=json or yaml) allowing result of the command to be parsed in either json format or yaml. ([#39858](https://github.com/kubernetes/kubernetes/pull/39858), [@alejandroEsc](https://github.com/alejandroEsc))


  * Bug fixes

    * Fix some false negatives in detection of meaningful conflicts during strategic merge patch with maps and lists. ([#43469](https://github.com/kubernetes/kubernetes/pull/43469), [@enisoc](https://github.com/enisoc))

		* Fix false positive "meaningful conflict" detection for strategic merge patch with integer values. ([#44788](https://github.com/kubernetes/kubernetes/pull/44788), [@enisoc](https://github.com/enisoc))

		* Restored the ability of kubectl running inside a pod to consume resource files specifying a different namespace than the one the pod is running in. ([#44862](https://github.com/kubernetes/kubernetes/pull/44862), [@liggitt](https://github.com/liggitt))

    * Kubectl commands run inside a pod using a kubeconfig file now use the namespace specified in the kubeconfig file, instead of using the pod namespace. If no kubeconfig file is used, or the kubeconfig does not specify a namespace, the pod namespace is still used as a fallback. ([#44570](https://github.com/kubernetes/kubernetes/pull/44570), [@liggitt](https://github.com/liggitt))

    * Fixed `kubectl cluster-info` dump to support multi-container pod. ([#44088](https://github.com/kubernetes/kubernetes/pull/44088), [@xingzhou](https://github.com/xingzhou))

    * Kubectl will print a warning when deleting the current context ([#42538](https://github.com/kubernetes/kubernetes/pull/42538), [@adohe](https://github.com/adohe))

    * Fix VolumeClaims/capacity in `kubectl describe statefulsets` output. ([#47573](https://github.com/kubernetes/kubernetes/pull/47573), [@k82cn](https://github.com/k82cn))

		* Fixed the output of kubectl taint node command with minor improvements. ([#43171](https://github.com/kubernetes/kubernetes/pull/43171), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))


### **Networking**
#### Network Policy
* [stable] [NetworkPolicy](https://kubernetes.io/docs/concepts/services-networking/network-policies/) promoted to GA.
  * Additionally adds short name "netpol" for networkpolicies ([#42241](https://github.com/kubernetes/kubernetes/pull/42241), [@xiangpengzhao](https://github.com/xiangpengzhao))


#### Load Balancing
* [stable] Source IP Preservation - change Cloud load-balancer strategy to health-checks and respond to health check only on nodes that host pods for the service. See [Create an External Load Balancer - Preserving the client source IP](https://kubernetes.io/docs/tasks/access-application-cluster/create-external-load-balancer/#preserving-the-client-source-ip).
  Two annotations have been promoted to API fields:

  * Service.Spec.ExternalTrafficPolicy was 'service.beta.kubernetes.io/external-traffic' annotation.

  * Service.Spec.HealthCheckNodePort was 'service.beta.kubernetes.io/healthcheck-nodeport' annotation.

### **Node Components**
#### Container Runtime Interface
* [alpha] CRI validation testing, which provides a test framework and a suite of tests to validate that the CRI server implementation meets all the requirements. This allows the CRI runtime developers to verify that their runtime conforms to CRI, without needing to set up Kubernetes components or run Kubernetes end-to-end tests. ([docs](https://github.com/kubernetes/community/blob/master/contributors/devel/cri-validation.md) and [release notes](https://github.com/kubernetes-incubator/cri-tools/releases/tag/v0.1)) ([kubernetes/features#292](https://github.com/kubernetes/features/issues/292))

* [alpha] Adds support of container metrics in CRI ([docs PR](https://github.com/kubernetes/community/pull/742)) ([kubernetes/features#290](https://github.com/kubernetes/features/issues/290))

* [alpha] Integration with [containerd] (https://github.com/containerd/containerd) , which supports basic pod lifecycle and image management. ([docs](https://github.com/kubernetes-incubator/cri-containerd/blob/master/README.md) and [release notes](https://github.com/kubernetes-incubator/cri-containerd/releases/tag/v0.1.0)) ([kubernetes/features#286](https://github.com/kubernetes/features/issues/286))

* [GA] The Docker-CRI implementation is GA. The legacy, non-CRI Docker integration has been completely removed.

* [beta] [CRI-O](https://github.com/kubernetes-incubator/cri-o) v1.0.0-alpha.0. It has passed all e2e tests. ([release notes](https://github.com/kubernetes-incubator/cri-o/releases/tag/v1.0.0-alpha.0))

* [beta] [Frakti](https://github.com/kubernetes/frakti) v1.0. It has passed all node conformance tests. ([release notes](https://github.com/kubernetes/frakti/releases/tag/v1.0))



### **Scheduling**
#### Scheduler Extender
* [alpha] Support for delegating pod binding to a scheduler extender ([kubernetes/features#270](https://github.com/kubernetes/features/issues/270))

### **Storage**
#### Local Storage
* [alpha] This feature adds capacity isolation support for local storage at node, container, and volume levels. See updated [Reserve Compute Resources for System Daemons](https://kubernetes.io/docs/tasks/administer-cluster/reserve-compute-resources/) documentation.

* [alpha] Make locally attached (non-network attached) storage available as a persistent volume source. For more information, see [Storage Volumes - local](https://kubernetes.io/docs/concepts/storage/volumes/#local).

#### Volume Plugins
* [stable] Volume plugin for StorageOS provides highly-available cluster-wide persistent volumes from local or attached node storage. See [Persistent Volumes - StorageOS](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#storageos) and [Storage Volumes - StorageOS](https://kubernetes.io/docs/concepts/storage/volumes/#storageos).

#### Metrics
* [stable] Add support for cloudprovider metrics for storage API calls. See [Controller manager metrics](https://kubernetes.io/docs/concepts/cluster-administration/controller-metrics/) for more information.

### **Other notable changes**

#### Admission plugin
* OwnerReferencesPermissionEnforcement admission plugin ignores pods/status. ([#45747](https://github.com/kubernetes/kubernetes/pull/45747), [@derekwaynecarr](https://github.com/derekwaynecarr))


* Ignored mirror pods in PodPreset admission plugin. ([#45958](https://github.com/kubernetes/kubernetes/pull/45958), [@k82cn](https://github.com/k82cn))

#### API Machinery
* The protobuf serialization of API objects has been updated to store maps in a predictable order to ensure that the representation of that object does not change when saved into etcd. This prevents the same object from being seen as being modified, even when no values have changed. ([#47701](https://github.com/kubernetes/kubernetes/pull/47701), [@smarterclayton](https://github.com/smarterclayton))

* API resource discovery now includes the singularName used to refer to the resource. ([#43312](https://github.com/kubernetes/kubernetes/pull/43312), [@deads2k](https://github.com/deads2k))

* Enhance the garbage collection admission plugin so that a user who doesn't have delete permission of the owning object cannot modify the blockOwnerDeletion field of existing ownerReferences, or add new ownerReferences with blockOwnerDeletion=true ([#43876](https://github.com/kubernetes/kubernetes/pull/43876), [@caesarxuchao](https://github.com/caesarxuchao))

* Exec and portforward actions over SPDY now properly handle redirects sent by the Kubelet ([#44451](https://github.com/kubernetes/kubernetes/pull/44451), [@ncdc](https://github.com/ncdc))

* The proxy subresource APIs for nodes, services, and pods now support the HTTP PATCH method. ([#44929](https://github.com/kubernetes/kubernetes/pull/44929), [@liggitt](https://github.com/liggitt))

* The Categories []string field on discovered API resources represents the list of group aliases (e.g. "all") that each resource belongs to. ([#43338](https://github.com/kubernetes/kubernetes/pull/43338), [@fabianofranz](https://github.com/fabianofranz))

* [alpha] The Kubernetes API supports retrieving tabular output for API resources via a new mime-type application/json;as=Table;v=v1alpha1;g=meta.k8s.io. The returned object (if the server supports it) will be of type meta.k8s.io/v1alpha1 with Table, and contain column and row information related to the resource. Each row will contain information about the resource - by default it will be the object metadata, but callers can add the ?includeObject=Object query parameter and receive the full object. In the future kubectl will use this to retrieve the results of `kubectl get`. ([#40848](https://github.com/kubernetes/kubernetes/pull/40848), [@smarterclayton](https://github.com/smarterclayton))

* The behavior of some watch calls to the server when filtering on fields was incorrect. If watching objects with a filter, when an update was made that no longer matched the filter a DELETE event was correctly sent. However, the object that was returned by that delete was not the (correct) version before the update, but instead, the newer version. That meant the new object was not matched by the filter. This was a regression from behavior between cached watches on the server side and uncached watches, and thus broke downstream API clients. ([#46223](https://github.com/kubernetes/kubernetes/pull/46223), [@smarterclayton](https://github.com/smarterclayton))

* OpenAPI spec is now available in protobuf binary and gzip format (with ETag support) ([#45836](https://github.com/kubernetes/kubernetes/pull/45836), [@mbohlool](https://github.com/mbohlool))

* Updating apiserver to return UID of the deleted resource. Clients can use this UID to verify that the resource was deleted or waiting for finalizers. ([#45600](https://github.com/kubernetes/kubernetes/pull/45600), [@nikhiljindal](https://github.com/nikhiljindal))

* Fix incorrect conflict errors applying strategic merge patches to resources. ([#43871](https://github.com/kubernetes/kubernetes/pull/43871), [@liggitt](https://github.com/liggitt))

* Fix init container status reporting when active deadline is exceeded. ([#46305](https://github.com/kubernetes/kubernetes/pull/46305), [@sjenning](https://github.com/sjenning))

* Moved qos to api.helpers. ([#44906](https://github.com/kubernetes/kubernetes/pull/44906), [@k82cn](https://github.com/k82cn))

* Fix issue with the resource quota controller causing add quota to be resynced at the wrong ([#45685](https://github.com/kubernetes/kubernetes/pull/45685), [@derekwaynecarr](https://github.com/derekwaynecarr))

* Added Group/Version/Kind and Action extension to OpenAPI Operations ([#44787](https://github.com/kubernetes/kubernetes/pull/44787), [@mbohlool](https://github.com/mbohlool))

* Make clear that meta.KindToResource is only a guess ([#45272](https://github.com/kubernetes/kubernetes/pull/45272), [@sttts](https://github.com/sttts))

* Add APIService conditions ([#43301](https://github.com/kubernetes/kubernetes/pull/43301), [@deads2k](https://github.com/deads2k))

* Create and push a docker image for the cloud-controller-manager ([#45154](https://github.com/kubernetes/kubernetes/pull/45154), [@luxas](https://github.com/luxas))

* Deprecated Binding objects in 1.7. ([#47041](https://github.com/kubernetes/kubernetes/pull/47041), [@k82cn](https://github.com/k82cn))

* Adds the Categories []string field to API resources, which represents the list of group aliases (e.g. "all") that every resource belongs to. ([#43338](https://github.com/kubernetes/kubernetes/pull/43338), [@fabianofranz](https://github.com/fabianofranz))

* `--service-account-lookup` now defaults to true, requiring the Secret API object containing the token to exist in order for a service account token to be valid. This enables service account tokens to be revoked by deleting the Secret object containing the token. ([#44071](https://github.com/kubernetes/kubernetes/pull/44071), [@liggitt](https://github.com/liggitt))

* API Registration is now in beta. ([#45247](https://github.com/kubernetes/kubernetes/pull/45247), [@mbohlool](https://github.com/mbohlool))

* The Kubernetes API server now exits if it encounters a networking failure (e.g. the networking interface hosting its address goes away) to allow a process manager (systemd/kubelet/etc) to react to the problem. Previously the server would log the failure and try again to bind to its configured address:port. ([#42272](https://github.com/kubernetes/kubernetes/pull/42272), [@marun](https://github.com/marun))

* The Prometheus metrics for the kube-apiserver for tracking incoming API requests and latencies now return the subresource label for correctly attributing the type of API call. ([#46354](https://github.com/kubernetes/kubernetes/pull/46354), [@smarterclayton](https://github.com/smarterclayton))

* kube-apiserver now drops unneeded path information if an older version of Windows kubectl sends it. ([#44421](https://github.com/kubernetes/kubernetes/pull/44421), [@mml](https://github.com/mml))


#### Application autoscaling
* Make "upscale forbidden window" and "downscale forbidden window"  duration configurable in arguments of kube-controller-manager. ([#42101](https://github.com/kubernetes/kubernetes/pull/42101), [@Dmitry1987](https://github.com/Dmitry1987))

#### Application Deployment
* StatefulSetStatus now tracks replicas, readyReplicas, currentReplicas, and updatedReplicas. The semantics of replicas is now consistent with DaemonSet and ReplicaSet, and readyReplicas has the semantics that replicas did prior to 1.7 ([#46669](https://github.com/kubernetes/kubernetes/pull/46669), [@kow3ns](https://github.com/kow3ns)).

* ControllerRevision type has been added for StatefulSet and DaemonSet history. Clients should not depend on the stability of this type as it may change, as necessary, in future releases to support StatefulSet and DaemonSet update and rollback. We enable this type as we do with beta features, because StatefulSet update and DaemonSet update are enabled. ([#45867](https://github.com/kubernetes/kubernetes/pull/45867), [@kow3ns](https://github.com/kow3ns))

* PodDisruptionBudget now uses ControllerRef to decide which controller owns a given Pod, so it doesn't get confused by controllers with overlapping selectors. ([#45003](https://github.com/kubernetes/kubernetes/pull/45003), [@krmayankk](https://github.com/krmayankk))

* Deployments are updated to use (1) a more stable hashing algorithm (fnv) than the previous one (adler) and (2) a hashing collision avoidance mechanism that will ensure new rollouts will not block on hashing collisions anymore. ([#44774](https://github.com/kubernetes/kubernetes/pull/44774), [@kargakis](https://github.com/kargakis))([kubernetes/features#287](https://github.com/kubernetes/features/issues/287))

* Deployments and DaemonSets rollouts are considered complete when all of the desired replicas are updated and available. This change affects `kubectl rollout status` and Deployment condition. ([#44672](https://github.com/kubernetes/kubernetes/pull/44672), [@kargakis](https://github.com/kargakis))

* Job controller now respects ControllerRef to avoid fighting over Pods. ([#42176](https://github.com/kubernetes/kubernetes/pull/42176), [@enisoc](https://github.com/enisoc))

* CronJob controller now respects ControllerRef to avoid fighting with other controllers. ([#42177](https://github.com/kubernetes/kubernetes/pull/42177), [@enisoc](https://github.com/enisoc))

#### Cluster Autoscaling
* Cluster Autoscaler 0.6. More information available [here](https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/README.md).

* cluster-autoscaler: Fix duplicate writing of logs. ([#45017](https://github.com/kubernetes/kubernetes/pull/45017), [@MaciekPytel](https://github.com/MaciekPytel))


#### Cloud Provider Enhancement

* AWS:

  * New 'service.beta.kubernetes.io/aws-load-balancer-extra-security-groups' Service annotation to specify extra Security Groups to be added to ELB created by AWS cloudprovider ([#45268](https://github.com/kubernetes/kubernetes/pull/45268), [@redbaron](https://github.com/redbaron))

  * Clean up blackhole routes when using kubenet ([#47572](https://github.com/kubernetes/kubernetes/pull/47572), [@justinsb](https://github.com/justinsb))

  * Maintain a cache of all instances, to fix problem with > 200 nodes with ELBs ([#47410](https://github.com/kubernetes/kubernetes/pull/47410), [@justinsb](https://github.com/justinsb))

  * Avoid spurious ELB listener recreation - ignore case when matching protocol ([#47391](https://github.com/kubernetes/kubernetes/pull/47391), [@justinsb](https://github.com/justinsb))

  * Allow configuration of a single security group for ELBs ([#45500](https://github.com/kubernetes/kubernetes/pull/45500), [@nbutton23](https://github.com/nbutton23))

  * Remove check that forces loadBalancerSourceRanges to be 0.0.0.0/0. ([#38636](https://github.com/kubernetes/kubernetes/pull/38636), [@dhawal55](https://github.com/dhawal55))

	* Allow setting KubernetesClusterID or KubernetesClusterTag in combination with VPC. ([#42512](https://github.com/kubernetes/kubernetes/pull/42512), [@scheeles](https://github.com/scheeles))

	* Start recording cloud provider metrics for AWS ([#43477](https://github.com/kubernetes/kubernetes/pull/43477), [@gnufied](https://github.com/gnufied))

	* AWS: Batch DescribeInstance calls with nodeNames to 150 limit, to stay within AWS filter limits. ([#47516](https://github.com/kubernetes/kubernetes/pull/47516), [@gnufied](https://github.com/gnufied))

	* AWS: Process disk attachments even with duplicate NodeNames ([#47406](https://github.com/kubernetes/kubernetes/pull/47406), [@justinsb](https://github.com/justinsb))

  * Allow configuration of a single security group for ELBs ([#45500](https://github.com/kubernetes/kubernetes/pull/45500), [@nbutton23](https://github.com/nbutton23))

  * Fix support running the master with a different AWS account or even on a different cloud provider than the nodes. ([#44235](https://github.com/kubernetes/kubernetes/pull/44235), [@mrIncompetent](https://github.com/mrIncompetent))

  * Support node port health check ([#43585](https://github.com/kubernetes/kubernetes/pull/43585), [@foolusion](https://github.com/foolusion))

  * Support for ELB tagging by users ([#45932](https://github.com/kubernetes/kubernetes/pull/45932), [@lpabon](https://github.com/lpabon))

* Azure:

  * Add support for UDP ports ([#45523](https://github.com/kubernetes/kubernetes/pull/45523), [@colemickens](https://github.com/colemickens))

  * Fix support for multiple loadBalancerSourceRanges ([#45523](https://github.com/kubernetes/kubernetes/pull/45523), [@colemickens](https://github.com/colemickens))

  * Support the Service spec's sessionAffinity ([#45523](https://github.com/kubernetes/kubernetes/pull/45523), [@colemickens](https://github.com/colemickens))

	* Added exponential backoff to Azure cloudprovider ([#46660](https://github.com/kubernetes/kubernetes/pull/46660), [@jackfrancis](https://github.com/jackfrancis))

  * Add support for bring-your-own ip address for Services on Azure ([#42034](https://github.com/kubernetes/kubernetes/pull/42034), [@brendandburns](https://github.com/brendandburns))

  * Add support for Azure internal load balancer ([#43510](https://github.com/kubernetes/kubernetes/pull/43510), [@karataliu](https://github.com/karataliu))

	* Client poll duration is now 5 seconds ([#43699](https://github.com/kubernetes/kubernetes/pull/43699), [@colemickens](https://github.com/colemickens))

	* Azure plugin for client auth ([#43987](https://github.com/kubernetes/kubernetes/pull/43987), [@cosmincojocar](https://github.com/cosmincojocar))


* GCP:

  * Bump GLBC version to 0.9.5 - fixes [loss of manually modified GCLB health check settings](https://github.com/kubernetes/kubernetes/issues/47559) upon upgrade from pre-1.6.4 to either 1.6.4 or 1.6.5. ([#47567](https://github.com/kubernetes/kubernetes/pull/47567), [@nicksardo](https://github.com/nicksardo))

  * [beta] Support creation of GCP Internal Load Balancers from Service objects ([#46663](https://github.com/kubernetes/kubernetes/pull/46663), [@nicksardo](https://github.com/nicksardo))

  * GCE installs will now avoid IP masquerade for all RFC-1918 IP blocks, rather than just 10.0.0.0/8. This means that clusters can be created in 192.168.0.0./16 and 172.16.0.0/12 while preserving the container IPs (which would be lost before). ([#46473](https://github.com/kubernetes/kubernetes/pull/46473), [@thockin](https://github.com/thockin))

  * The Calico version included in kube-up for GCE has been updated to v2.2. ([#38169](https://github.com/kubernetes/kubernetes/pull/38169), [@caseydavenport](https://github.com/caseydavenport))

	* ip-masq-agent is now on by default for GCE ([#47794](https://github.com/kubernetes/kubernetes/pull/47794), [@dnardo](https://github.com/dnardo))

  * Add ip-masq-agent addon to the addons folder which is used in GCE if `--non-masquerade-cidr` is set to 0/0 ([#46038](https://github.com/kubernetes/kubernetes/pull/46038), [@dnardo](https://github.com/dnardo))

  * Enable kubelet csr bootstrap in GCE/GKE ([#40760](https://github.com/kubernetes/kubernetes/pull/40760), [@mikedanese](https://github.com/mikedanese))

  * Adds support for allocation of pod IPs via IP aliases. ([#42147](https://github.com/kubernetes/kubernetes/pull/42147), [@bowei](https://github.com/bowei))

	* gce kube-up: The Node authorization mode and NodeRestriction admission controller are now enabled ([#46796](https://github.com/kubernetes/kubernetes/pull/46796), [@mikedanese](https://github.com/mikedanese))

	* Tokens retrieved from Google Cloud with application default credentials will not be cached if the client fails authorization ([#46694](https://github.com/kubernetes/kubernetes/pull/46694), [@matt-tyler](https://github.com/matt-tyler))

	* Add metrics to all major gce operations {latency, errors} ([#44510](https://github.com/kubernetes/kubernetes/pull/44510), [@bowei](https://github.com/bowei))

	    * The new metrics are:

	    * cloudprovider_gce_api_request_duration_seconds{request, region, zone}

	    * cloudprovider_gce_api_request_errors{request, region, zone}

	    * request is the specific function that is used.

	    * region is the target region (Will be "<n/a>" if not applicable)

	    * zone is the target zone (Will be "<n/a>" if not applicable)

	    * Note: this fixes some issues with the previous implementation of metrics for disks:

	      * Time duration tracked was of the initial API call, not the entire operation.

	      * Metrics label tuple would have resulted in many independent histograms stored, one for each disk. (Did not aggregate well).

  * Fluentd now tolerates all NoExecute Taints when run in gcp configuration. ([#45715](https://github.com/kubernetes/kubernetes/pull/45715), [@gmarek](https://github.com/gmarek))

	* Taints support in gce/salt startup scripts. ([#47632](https://github.com/kubernetes/kubernetes/pull/47632), [@mwielgus](https://github.com/mwielgus))

	* GCE installs will now avoid IP masquerade for all RFC-1918 IP blocks, rather than just 10.0.0.0/8. This means that clusters can ([#46473](https://github.com/kubernetes/kubernetes/pull/46473), [@thockin](https://github.com/thockin)) be created in 192.168.0.0./16 and 172.16.0.0/12 while preserving the container IPs (which would be lost before).

	* Support running Ubuntu image on GCE node ([#44744](https://github.com/kubernetes/kubernetes/pull/44744), [@yguo0905](https://github.com/yguo0905))

  * The gce metadata server can now be hidden behind a proxy, hiding the kubelet's token. ([#45565](https://github.com/kubernetes/kubernetes/pull/45565), [@Q-Lee](https://github.com/Q-Lee))

* OpenStack:

    * Fix issue during LB creation where ports were incorrectly assigned to a floating IP ([#44387](https://github.com/kubernetes/kubernetes/pull/44387), [@jamiehannaford](https://github.com/jamiehannaford))

    * Openstack cinder v1/v2/auto API support ([#40423](https://github.com/kubernetes/kubernetes/pull/40423), [@mkutsevol](https://github.com/mkutsevol))

    * OpenStack clusters can now specify whether worker nodes are assigned a floating IP ([#42638](https://github.com/kubernetes/kubernetes/pull/42638), [@jamiehannaford](https://github.com/jamiehannaford))


* vSphere:

  * Fix volume detach on node failure. ([#45569](https://github.com/kubernetes/kubernetes/pull/45569), [@divyenpatel](https://github.com/divyenpatel))

  * Report same Node IP as both internal and external. ([#45201](https://github.com/kubernetes/kubernetes/pull/45201), [@abrarshivani](https://github.com/abrarshivani))

  * Filter out IPV6 node addresses. ([#45181](https://github.com/kubernetes/kubernetes/pull/45181), [@BaluDontu](https://github.com/BaluDontu))

  * Fix fetching of VM UUID on Ubuntu 16.04 and Fedora. ([#45311](https://github.com/kubernetes/kubernetes/pull/45311), [@divyenpatel](https://github.com/divyenpatel))


#### Cluster Provisioning
* Juju:

  * Add Kubernetes 1.6 support to Juju charms ([#44500](https://github.com/kubernetes/kubernetes/pull/44500), [@Cynerva](https://github.com/Cynerva))

    * Add metric collection to charms for autoscaling

    * Update kubernetes-e2e charm to fail when test suite fails

    * Update Juju charms to use snaps

    * Add registry action to the kubernetes-worker charm

    * Add support for kube-proxy cluster-cidr option to kubernetes-worker charm

    * Fix kubernetes-master charm starting services before TLS certs are saved

    * Fix kubernetes-worker charm failures in LXD

    * Fix stop hook failure on kubernetes-worker charm

    * Fix handling of juju kubernetes-worker.restart-needed state

    * Fix nagios checks in charms

  * Enable GPU mode if GPU hardware detected ([#43467](https://github.com/kubernetes/kubernetes/pull/43467), [@tvansteenburgh](https://github.com/tvansteenburgh))

  * Fix ceph-secret type to kubernetes.io/rbd in kubernetes-master charm ([#44635](https://github.com/kubernetes/kubernetes/pull/44635), [@Cynerva](https://github.com/Cynerva))

  * Disallows installation of upstream docker from PPA in the Juju kubernetes-worker charm. ([#44681](https://github.com/kubernetes/kubernetes/pull/44681), [@wwwtyro](https://github.com/wwwtyro))

  * Resolves juju vsphere hostname bug showing only a single node in a scaled node-pool. ([#44780](https://github.com/kubernetes/kubernetes/pull/44780), [@chuckbutler](https://github.com/chuckbutler))

  * Fixes a bug in the kubernetes-worker Juju charm code that attempted to give kube-proxy more than one api endpoint. ([#44677](https://github.com/kubernetes/kubernetes/pull/44677), [@wwwtyro](https://github.com/wwwtyro))

  * Added CIFS PV support for Juju Charms ([#45117](https://github.com/kubernetes/kubernetes/pull/45117), [@chuckbutler](https://github.com/chuckbutler))

  * Fixes juju kubernetes master: 1. Get certs from a dead leader. 2. Append tokens. ([#43620](https://github.com/kubernetes/kubernetes/pull/43620), [@ktsakalozos](https://github.com/ktsakalozos))

  * kubernetes-master juju charm properly detects etcd-scale events and reconfigures appropriately. ([#44967](https://github.com/kubernetes/kubernetes/pull/44967), [@chuckbutler](https://github.com/chuckbutler))

 	* Use correct option name in the kubernetes-worker layer registry action ([#44921](https://github.com/kubernetes/kubernetes/pull/44921), [@jacekn](https://github.com/jacekn))

	* Send dns details only after cdk-addons are configured ([#44945](https://github.com/kubernetes/kubernetes/pull/44945), [@ktsakalozos](https://github.com/ktsakalozos))

	* Added support to the pause action in the kubernetes-worker charm for new flag `--delete-local-data` ([#44931](https://github.com/kubernetes/kubernetes/pull/44931), [@chuckbutler](https://github.com/chuckbutler))

	* Add namespace-{list, create, delete} actions to the kubernetes-master layer ([#44277](https://github.com/kubernetes/kubernetes/pull/44277), [@jacekn](https://github.com/jacekn))

	* Using http2 in kubeapi-load-balancer to fix `kubectl exec` uses ([#43625](https://github.com/kubernetes/kubernetes/pull/43625), [@mbruzek](https://github.com/mbruzek))


  * Don't append :443 to registry domain in the kubernetes-worker layer registry action ([#45550](https://github.com/kubernetes/kubernetes/pull/45550), [@jacekn](https://github.com/jacekn))

* kubeadm

  * Enable the Node Authorizer/Admission plugin in v1.7 ([#46879](https://github.com/kubernetes/kubernetes/pull/46879), [@luxas](https://github.com/luxas))

  * Users can now pass extra parameters to etcd in a kubeadm cluster ([#42246](https://github.com/kubernetes/kubernetes/pull/42246), [@jamiehannaford](https://github.com/jamiehannaford))

  * Make kubeadm use the new CSR approver in v1.7 ([#46864](https://github.com/kubernetes/kubernetes/pull/46864), [@luxas](https://github.com/luxas))

  * Allow enabling multiple authorization modes at the same time ([#42557](https://github.com/kubernetes/kubernetes/pull/42557), [@xilabao](https://github.com/xilabao))

  * add proxy client-certs to kube-apiserver to allow it to proxy aggregated api servers ([#43715](https://github.com/kubernetes/kubernetes/pull/43715), [@deads2k](https://github.com/deads2k))* CentOS provider

* hyperkube

  * The hyperkube image has been slimmed down and no longer includes addon manifests and other various scripts. These were introduced for the now removed docker-multinode setup system. ([#44555](https://github.com/kubernetes/kubernetes/pull/44555), [@luxas](https://github.com/luxas))

* Support secure etcd cluster for centos provider. ([#42994](https://github.com/kubernetes/kubernetes/pull/42994), [@Shawyeok](https://github.com/Shawyeok))

* Update to kube-addon-manager:v6.4-beta.2: kubectl v1.6.4 and refreshed base images ([#47389](https://github.com/kubernetes/kubernetes/pull/47389), [@ixdy](https://github.com/ixdy))

* Remove Initializers from admission-control in kubernetes-master charm for pre-1.7 ([#46987](https://github.com/kubernetes/kubernetes/pull/46987), [@Cynerva](https://github.com/Cynerva))

* Added state guards to the idle_status messaging in the kubernetes-master charm to make deployment faster on initial deployment. ([#47183](https://github.com/kubernetes/kubernetes/pull/47183), [@chuckbutler](https://github.com/chuckbutler))

#### Cluster federation
* Features:

  * Adds annotations to all Federation objects created by kubefed. ([#42683](https://github.com/kubernetes/kubernetes/pull/42683), [@perotinus](https://github.com/perotinus))

	* Mechanism of adding `federation domain maps` to kube-dns deployment via `--federations` flag is superseded by adding/updating `federations` key in `kube-system/kube-dns` configmap. If user is using kubefed tool to join cluster federation, adding federation domain maps to kube-dns is already taken care by `kubefed join` and does not need further action.

	* Prints out status updates when running `kubefed init` ([#41849](https://github.com/kubernetes/kubernetes/pull/41849), [@perotinus](https://github.com/perotinus))

	* `kubefed init` now supports overriding the default etcd image name with the `--etcd-image` parameter. ([#46247](https://github.com/kubernetes/kubernetes/pull/46247), [@marun](https://github.com/marun))

	* kubefed will now configure NodeInternalIP as the federation API server endpoint when NodeExternalIP is unavailable for federation API servers exposed as NodePort services ([#46960](https://github.com/kubernetes/kubernetes/pull/46960), [@lukaszo](https://github.com/lukaszo))

	* Automate configuring nameserver in cluster-dns for CoreDNS provider ([#42895](https://github.com/kubernetes/kubernetes/pull/42895), [@shashidharatd](https://github.com/shashidharatd))

	* A new controller for managing DNS records is introduced which can be optionally disabled to enable third party components to manage DNS records for federated services. ([#45034](https://github.com/kubernetes/kubernetes/pull/45034), [@shashidharatd](https://github.com/shashidharatd))

  * Remove the `--secret-name` flag from `kubefed join`, instead generating the secret name arbitrarily. ([#42513](https://github.com/kubernetes/kubernetes/pull/42513), [@perotinus](https://github.com/perotinus))

  *  Use StorageClassName for etcd pvc ([#46323](https://github.com/kubernetes/kubernetes/pull/46323), [@marun](https://github.com/marun))

* Bug fixes:

	* Allow disabling federation controllers through override args ([#44209](https://github.com/kubernetes/kubernetes/pull/44209), [@irfanurrehman](https://github.com/irfanurrehman))

	* Kubefed: Use service accounts instead of the user's credentials when accessing joined clusters' API servers. ([#42042](https://github.com/kubernetes/kubernetes/pull/42042), [@perotinus](https://github.com/perotinus))

	* Avoid panic if route53 fields are nil ([#44380](https://github.com/kubernetes/kubernetes/pull/44380), [@justinsb](https://github.com/justinsb))


#### Credential provider
* add rancher credential provider ([#40160](https://github.com/kubernetes/kubernetes/pull/40160), [@wlan0](https://github.com/wlan0))

#### Information for Kubernetes clients (openapi, swagger, client-go)
* Features:

  * Add Host field to TCPSocketAction ([#42902](https://github.com/kubernetes/kubernetes/pull/42902), [@louyihua](https://github.com/louyihua))

	* Add the ability to lock on ConfigMaps to support HA for self hosted components ([#42666](https://github.com/kubernetes/kubernetes/pull/42666), [@timothysc](https://github.com/timothysc))

	* validateClusterInfo: use clientcmdapi.NewCluster() ([#44221](https://github.com/kubernetes/kubernetes/pull/44221), [@ncdc](https://github.com/ncdc))

	* OpenAPI spec is now available in protobuf binary and gzip format (with ETag support) ([#45836](https://github.com/kubernetes/kubernetes/pull/45836), [@mbohlool](https://github.com/mbohlool))

	* HostAliases is now parsed with hostAliases json keys to be in line with the feature's name. ([#47512](https://github.com/kubernetes/kubernetes/pull/47512), [@rickypai](https://github.com/rickypai))

	* Add redirect support to SpdyRoundTripper ([#44451](https://github.com/kubernetes/kubernetes/pull/44451), [@ncdc](https://github.com/ncdc))

	* Duplicate recurring Events now include the latest event's Message string ([#46034](https://github.com/kubernetes/kubernetes/pull/46034), [@kensimon](https://github.com/kensimon))

* Bug fixes:

  * Fix serialization of EnforceNodeAllocatable ([#44606](https://github.com/kubernetes/kubernetes/pull/44606), [@ivan4th](https://github.com/ivan4th))

	* Use OS-specific libs when computing client User-Agent in kubectl, etc. ([#44423](https://github.com/kubernetes/kubernetes/pull/44423), [@monopole](https://github.com/monopole))


#### Instrumentation
* Bumped Heapster to v1.4.0. More details about the release https://github.com/kubernetes/heapster/releases/tag/v1.4.0

* Fluentd manifest pod is no longer created on non-registered master when creating clusters using kube-up.sh. ([#44721](https://github.com/kubernetes/kubernetes/pull/44721), [@piosz](https://github.com/piosz))

* Stackdriver cluster logging now deploys a new component to export Kubernetes events. ([#46700](https://github.com/kubernetes/kubernetes/pull/46700), [@crassirostris](https://github.com/crassirostris))

* Stackdriver Logging deployment exposes metrics on node port 31337 when enabled. ([#47402](https://github.com/kubernetes/kubernetes/pull/47402), [@crassirostris](https://github.com/crassirostris))

* Upgrade Elasticsearch Addon to v5.4.0 ([#45589](https://github.com/kubernetes/kubernetes/pull/45589), [@it-svit](https://github.com/it-svit))

#### Internal storage layer
* prevent pods/status from touching ownerreferences ([#45826](https://github.com/kubernetes/kubernetes/pull/45826), [@deads2k](https://github.com/deads2k))

* Ensure that autoscaling/v1 is the preferred version for API discovery when autoscaling/v2alpha1 is enabled. ([#45741](https://github.com/kubernetes/kubernetes/pull/45741), [@DirectXMan12](https://github.com/DirectXMan12))

* The proxy subresource APIs for nodes, services, and pods now support the HTTP PATCH method. ([#44929](https://github.com/kubernetes/kubernetes/pull/44929), [@liggitt](https://github.com/liggitt))

* Fluentd now tolerates all NoExecute Taints when run in gcp configuration. ([#45715](https://github.com/kubernetes/kubernetes/pull/45715), [@gmarek](https://github.com/gmarek))


#### Kubernetes Dashboard

* Increase Dashboard's memory requests and limits ([#44712](https://github.com/kubernetes/kubernetes/pull/44712), [@maciaszczykm](https://github.com/maciaszczykm))

* Update Dashboard version to 1.6.1 ([#45953](https://github.com/kubernetes/kubernetes/pull/45953), [@maciaszczykm](https://github.com/maciaszczykm))


#### kube-dns
* Updates kube-dns to 1.14.2 ([#45684](https://github.com/kubernetes/kubernetes/pull/45684), [@bowei](https://github.com/bowei))

   * Support kube-master-url flag without kubeconfig

   * Fix concurrent R/Ws in dns.go

   * Fix confusing logging when initialize server

   * Fix printf in cmd/kube-dns/app/server.go

   * Fix version on startup and `--version` flag

   * Support specifying port number for nameserver in stubDomains

#### kube-proxy
* Features:

  * ratelimit runs of iptables by sync-period flags ([#46266](https://github.com/kubernetes/kubernetes/pull/46266), [@thockin](https://github.com/thockin))

  * Log warning when invalid dir passed to `kubectl proxy --www` ([#44952](https://github.com/kubernetes/kubernetes/pull/44952), [@CaoShuFeng](https://github.com/CaoShuFeng))

  * Add `--write-config-to` flag to kube-proxy to allow users to write the default configuration settings to a file. ([#45908](https://github.com/kubernetes/kubernetes/pull/45908), [@ncdc](https://github.com/ncdc))

	* When switching from the service.beta.kubernetes.io/external-traffic annotation to the new ([#46716](https://github.com/kubernetes/kubernetes/pull/46716), [@thockin](https://github.com/thockin)) externalTrafficPolicy field, the values chnag as follows: * "OnlyLocal" becomes "Local" * "Global" becomes "Cluster".


* Bug fixes:

  * Fix corner-case with OnlyLocal Service healthchecks. ([#44313](https://github.com/kubernetes/kubernetes/pull/44313), [@thockin](https://github.com/thockin))

	* Fix DNS suffix search list support in Windows kube-proxy. ([#45642](https://github.com/kubernetes/kubernetes/pull/45642), [@JiangtianLi](https://github.com/JiangtianLi))

#### kube-scheduler
* Scheduler can receive its policy configuration from a ConfigMap ([#43892](https://github.com/kubernetes/kubernetes/pull/43892), [@bsalamat](https://github.com/bsalamat))

* Aggregated used ports at the NodeInfo level for PodFitsHostPorts predicate. ([#42524](https://github.com/kubernetes/kubernetes/pull/42524), [@k82cn](https://github.com/k82cn))

* leader election lock based on scheduler name ([#42961](https://github.com/kubernetes/kubernetes/pull/42961), [@wanghaoran1988](https://github.com/wanghaoran1988))


#### Storage

* Features

  * The options passed to a Flexvolume plugin's mount command now contains the pod name (kubernetes.io/pod.name), namespace (kubernetes.io/pod.namespace), uid (kubernetes.io/pod.uid), and service account name (kubernetes.io/serviceAccount.name). ([#39488](https://github.com/kubernetes/kubernetes/pull/39488), [@liggitt](https://github.com/liggitt))

  * GCE and AWS dynamic provisioners extension: admins can configure zone(s) in which a persistent volume shall be created. ([#38505](https://github.com/kubernetes/kubernetes/pull/38505), [@pospispa](https://github.com/pospispa))

  * Implement API usage metrics for GCE storage. ([#40338](https://github.com/kubernetes/kubernetes/pull/40338), [@gnufied](https://github.com/gnufied))

  * Add support for emitting metrics from openstack cloudprovider about storage operations. ([#46008](https://github.com/kubernetes/kubernetes/pull/46008), [@NickrenREN](https://github.com/NickrenREN))

  * vSphere cloud provider: vSphere storage policy support for dynamic volume provisioning. ([#46176](https://github.com/kubernetes/kubernetes/pull/46176), [@BaluDontu](https://github.com/BaluDontu))

  * Support StorageClass in Azure file volume ([#42170](https://github.com/kubernetes/kubernetes/pull/42170), [@rootfs](https://github.com/rootfs))

  * Start recording cloud provider metrics for AWS ([#43477](https://github.com/kubernetes/kubernetes/pull/43477), [@gnufied](https://github.com/gnufied))

  * Support iSCSI CHAP authentication ([#43396](https://github.com/kubernetes/kubernetes/pull/43396), [@rootfs](https://github.com/rootfs))

  * Openstack cinder v1/v2/auto API support ([#40423](https://github.com/kubernetes/kubernetes/pull/40423), [@mkutsevol](https://github.com/mkutsevol)](https://github.com/kubernetes/kubernetes/pull/41498), [@mikebryant](https://github.com/mikebryant))

  * Alpha feature: allows users to set storage limit to isolate EmptyDir volumes. It enforces the limit by evicting pods that exceed their storage limits ([#45686](https://github.com/kubernetes/kubernetes/pull/45686), [@jingxu97](https://github.com/jingxu97))

* Bug fixes

  * Fixes issue with Flexvolume, introduced in 1.6.0, where drivers without an attacher would fail (node indefinitely waiting for attach). A driver API addition is introduced: drivers that don't implement attach should return attach: false on init. ([#47503](https://github.com/kubernetes/kubernetes/pull/47503), [@chakri-nelluri](https://github.com/chakri-nelluri))

  * Fix dynamic provisioning of PVs with inaccurate AccessModes by refusing to provision when PVCs ask for AccessModes that can't be satisfied by the PVs' underlying volume plugin. ([#47274](https://github.com/kubernetes/kubernetes/pull/47274), [@wongma7](https://github.com/wongma7))

  * Fix pods failing to start if they specify a file as a volume subPath to mount. ([#45623](https://github.com/kubernetes/kubernetes/pull/45623), [@wongma7](https://github.com/wongma7))

  * Fix erroneous FailedSync and FailedMount events being periodically and indefinitely posted on Pods after kubelet is restarted. ([#44781](https://github.com/kubernetes/kubernetes/pull/44781), [@wongma7](https://github.com/wongma7))

  * Fix AWS EBS volumes not getting detached from node if routine to verify volumes are attached runs while the node is down ([#46463](https://github.com/kubernetes/kubernetes/pull/46463), [@wongma7](https://github.com/wongma7))

  * Improves performance of Cinder volume attach/detach operations. ([#41785](https://github.com/kubernetes/kubernetes/pull/41785), [@jamiehannaford](https://github.com/jamiehannaford))

  * Fix iSCSI iSER mounting. ([#47281](https://github.com/kubernetes/kubernetes/pull/47281), [@mtanino](https://github.com/mtanino))

  * iscsi storage plugin: Fix dangling session when using multiple target portal addresses. ([#46239](https://github.com/kubernetes/kubernetes/pull/46239), [@mtanino](https://github.com/mtanino))


  * Fix log spam due to unnecessary status update when node is deleted. ([#45923](https://github.com/kubernetes/kubernetes/pull/45923), [@verult](https://github.com/verult))

  * Don't try to attach volume to new node if it is already attached to another node and the volume does not support multi-attach. ([#45346](https://github.com/kubernetes/kubernetes/pull/45346), [@codablock](https://github.com/codablock))

  * detach the volume when pod is terminated ([#45286](https://github.com/kubernetes/kubernetes/pull/45286), [@gnufied](https://github.com/gnufied))

  * Roll up volume error messages in the kubelet sync loop. ([#44938](https://github.com/kubernetes/kubernetes/pull/44938), [@jayunit100](https://github.com/jayunit100))

  * Catch error when failed to make directory in NFS volume plugin ([#38801](https://github.com/kubernetes/kubernetes/pull/38801), [@nak3](https://github.com/nak3))



#### Networking

* DNS and name resolution

  * Updates kube-dns to 1.14.2 ([#45684](https://github.com/kubernetes/kubernetes/pull/45684), [@bowei](https://github.com/bowei))

    * Support kube-master-url flag without kubeconfig

    * Fix concurrent R/Ws in dns.go

    * Fix confusing logging when initializing server

    * Support specifying port number for nameserver in stubDomains

  * A new field hostAliases has been added to pod.spec to support adding entries to a Pod's /etc/hosts file. ([#44641](https://github.com/kubernetes/kubernetes/pull/44641), [@rickypai](https://github.com/rickypai))

  * Fix DNS suffix search list support in Windows kube-proxy. ([#45642](https://github.com/kubernetes/kubernetes/pull/45642), [@JiangtianLi](https://github.com/JiangtianLi))

* Kube-proxy

  * ratelimit runs of iptables by sync-period flags ([#46266](https://github.com/kubernetes/kubernetes/pull/46266), [@thockin](https://github.com/thockin))

  * Fix corner-case with OnlyLocal Service healthchecks. ([#44313](https://github.com/kubernetes/kubernetes/pull/44313), [@thockin](https://github.com/thockin))

* Exclude nodes labeled as master from LoadBalancer / NodePort; restores documented behaviour. ([#44745](https://github.com/kubernetes/kubernetes/pull/44745), [@justinsb](https://github.com/justinsb))

* Adds support for CNI ConfigLists, which permit plugin chaining. ([#42202](https://github.com/kubernetes/kubernetes/pull/42202), [@squeed](https://github.com/squeed))

* Fix node selection logic on initial LB creation ([#45773](https://github.com/kubernetes/kubernetes/pull/45773), [@justinsb](https://github.com/justinsb))

* When switching from the service.beta.kubernetes.io/external-traffic annotation to the new externalTrafficPolicy field, the values change as follows: * "OnlyLocal" becomes "Local" * "Global" becomes "Cluster". ([#46716](https://github.com/kubernetes/kubernetes/pull/46716), [@thockin](https://github.com/thockin))

* servicecontroller: Fix node selection logic on initial LB creation ([#45773](https://github.com/kubernetes/kubernetes/pull/45773), [@justinsb](https://github.com/justinsb))

* fixed HostAlias in PodSpec to allow foo.bar hostnames instead of just foo DNS labels. ([#46809](https://github.com/kubernetes/kubernetes/pull/46809), [@rickypai](https://github.com/rickypai))


#### Node controller
* Bug fixes:

  * Fix [transition between NotReady and Unreachable taints](https://github.com/kubernetes/kubernetes/issues/43444). ([#44042](https://github.com/kubernetes/kubernetes/pull/44042), [@gmarek](https://github.com/gmarek))


#### Node Components

* Features

  * Removes the deprecated kubelet flag `--babysit-daemons` ([#44230](https://github.com/kubernetes/kubernetes/pull/44230), [@mtaufen](https://github.com/mtaufen))

  * make dockershim.sock configurable ([#43914](https://github.com/kubernetes/kubernetes/pull/43914), [@ncdc](https://github.com/ncdc))

  * Support running Ubuntu image on GCE node ([#44744](https://github.com/kubernetes/kubernetes/pull/44744), [@yguo0905](https://github.com/yguo0905))

  * Kubernetes now shares a single PID namespace among all containers in a pod when running with docker >= 1.13.1. This means processes can now signal processes in other containers in a pod, but it also means that the `kubectl exec {pod} kill 1` pattern will cause the Pod to be restarted rather than a single container. ([#45236](https://github.com/kubernetes/kubernetes/pull/45236), [@verb](https://github.com/verb))

  * A new field hostAliases has been added to the pod spec to support [adding entries to a Pod's /etc/hosts file](https://kubernetes.io/docs/concepts/services-networking/add-entries-to-pod-etc-hosts-with-host-aliases/). ([#44641](https://github.com/kubernetes/kubernetes/pull/44641), [@rickypai](https://github.com/rickypai))

  * With `--feature-gates=RotateKubeletClientCertificate=true` set, the Kubelet will ([#41912](https://github.com/kubernetes/kubernetes/pull/41912), [@jcbsmpsn](https://github.com/jcbsmpsn))

    * request a client certificate from the API server during the boot cycle and pause

    * waiting for the request to be satisfied. It will continually refresh the certificate

  * Create clusters with GPUs in GCE by specifying `type=<gpu-type>,count=<gpu-count>` to NODE_ACCELERATORS environment variable. ([#45130](https://github.com/kubernetes/kubernetes/pull/45130), [@vishh](https://github.com/vishh))

    * List of available GPUs - [https://cloud.google.com/compute/docs/gpus/#introduction](https://cloud.google.com/compute/docs/gpus/#introduction)

  * Disk Pressure triggers the deletion of terminated containers on the node. ([#45896](https://github.com/kubernetes/kubernetes/pull/45896), [@dashpole](https://github.com/dashpole))

  * Support status.hostIP in downward API ([#42717](https://github.com/kubernetes/kubernetes/pull/42717), [@andrewsykim](https://github.com/andrewsykim))

  * Upgrade Node Problem Detector to v0.4.1. New features added:

    * Add /dev/kmsg support for kernel log parsing. ([#112](https://github.com/kubernetes/node-problem-detector/pull/112), [@euank](https://github.com/euank))

    * Add ABRT support. ([#105](https://github.com/kubernetes/node-problem-detector/pull/105), [@juliusmilan](https://github.com/juliusmilan))

    * Add a docker image corruption problem detection in the default docker monitor config. ([#117](https://github.com/kubernetes/node-problem-detector/pull/117), [@ajitak](https://github.com/ajitak))
    
  * Upgrade CAdvisor to v0.26.1. New features added:

    * Add Docker overlay2 storage driver support.

    * Add ZFS support.

    * Add UDP metrics (collection disabled by default).

  * Roll up volume error messages in the kubelet sync loop. ([#44938](https://github.com/kubernetes/kubernetes/pull/44938), [@jayunit100](https://github.com/jayunit100))
  
  * Allow pods to opt out of PodPreset mutation via an annotation on the pod. ([#44965](https://github.com/kubernetes/kubernetes/pull/44965), [@jpeeler](https://github.com/jpeeler))

  * Add generic Toleration for NoExecute Taints to NodeProblemDetector, so that NPD can be scheduled to nodes with NoExecute taints by default. ([#45883](https://github.com/kubernetes/kubernetes/pull/45883), [@gmarek](https://github.com/gmarek))

  * Prevent kubelet from setting allocatable < 0 for a resource upon initial creation. ([#46516](https://github.com/kubernetes/kubernetes/pull/46516), [@derekwaynecarr](https://github.com/derekwaynecarr))

* Bug fixes

  * Changed Kubelet default image-gc-high-threshold to 85% to resolve a conflict with default settings in docker that prevented image garbage collection from resolving low disk space situations when using devicemapper storage. ([#40432](https://github.com/kubernetes/kubernetes/pull/40432), [@sjenning](https://github.com/sjenning))

  * Mark all static pods on the Master node as critical to prevent preemption ([#47356](https://github.com/kubernetes/kubernetes/pull/47356), [@dashpole](https://github.com/dashpole))

  * Restrict active deadline seconds max allowed value to be maximum uint32 to avoid overflow ([#46640](https://github.com/kubernetes/kubernetes/pull/46640), [@derekwaynecarr](https://github.com/derekwaynecarr))

  * Fix a bug with cAdvisorPort in the KubeletConfiguration that prevented setting it to 0, which is in fact a valid option, as noted in issue [#11710](https://github.com/kubernetes/kubernetes/pull/11710). ([#46876](https://github.com/kubernetes/kubernetes/pull/46876), [@mtaufen](https://github.com/mtaufen))

  * Fix a bug where container cannot run as root when SecurityContext.RunAsNonRoot is false. ([#47009](https://github.com/kubernetes/kubernetes/pull/47009), [@yujuhong](https://github.com/yujuhong))

  * Fix the Kubelet PLEG update timestamp to better reflect the health of the component when the container runtime request hangs. ([#45496](https://github.com/kubernetes/kubernetes/pull/45496), [@andyxning](https://github.com/andyxning))

  * Avoid failing sync loop health check on container runtime errors ([#47124](https://github.com/kubernetes/kubernetes/pull/47124), [@andyxning](https://github.com/andyxning))

  * Fix a bug where Kubelet does not ignore pod manifest files starting with dots ([#45111](https://github.com/kubernetes/kubernetes/pull/45111), [@dwradcliffe](https://github.com/dwradcliffe))

  * Fix kubelet reset liveness probe failure count across pod restart boundaries ([#46371](https://github.com/kubernetes/kubernetes/pull/46371), [@sjenning](https://github.com/sjenning))

  * Fix log spam due to unnecessary status update when node is deleted. ([#45923](https://github.com/kubernetes/kubernetes/pull/45923), [@verult](https://github.com/verult))

  * Fix kubelet event recording for selected events. ([#46246](https://github.com/kubernetes/kubernetes/pull/46246), [@derekwaynecarr](https://github.com/derekwaynecarr))

  * Fix image garbage collector attempting to remove in-use images. ([#46121](https://github.com/kubernetes/kubernetes/pull/46121), [@Random-Liu](https://github.com/Random-Liu))

  * Detach the volume when pod is terminated ([#45286](https://github.com/kubernetes/kubernetes/pull/45286), [@gnufied](https://github.com/gnufied))

  * CRI: Fix StopContainer timeout ([#44970](https://github.com/kubernetes/kubernetes/pull/44970), [@Random-Liu](https://github.com/Random-Liu))

  * CRI: Fix kubelet failing to start when using rkt. ([#44569](https://github.com/kubernetes/kubernetes/pull/44569), [@yujuhong](https://github.com/yujuhong))

  * CRI: `kubectl logs -f` now stops following when container stops, as it did pre-CRI. ([#44406](https://github.com/kubernetes/kubernetes/pull/44406), [@Random-Liu](https://github.com/Random-Liu))

  * Fixes a bug where pods were evicted even after images are successfully deleted. ([#44986](https://github.com/kubernetes/kubernetes/pull/44986), [@dashpole](https://github.com/dashpole))

  * When creating a container using envFrom, ([#42083](https://github.com/kubernetes/kubernetes/pull/42083), [@fraenkel](https://github.com/fraenkel)
    * validate the name of the ConfigMap in a ConfigMapRef
    * validate the name of the Secret in a SecretRef

  * Fix the bug where StartedAt time is not reported for exited containers. ([#45977](https://github.com/kubernetes/kubernetes/pull/45977), [@yujuhong](https://github.com/yujuhong))

* Changes/deprecations

  * Marks the Kubelet's `--master-service-namespace` flag deprecated ([#44250](https://github.com/kubernetes/kubernetes/pull/44250), [@mtaufen](https://github.com/mtaufen))

  * Remove PodSandboxStatus.Linux.Namespaces.Network from CRI since it is not used/needed. ([#45166](https://github.com/kubernetes/kubernetes/pull/45166), [@feiskyer](https://github.com/feiskyer))

  * Remove the `--enable-cri` flag. CRI is now the default, and the only way to integrate with Kubelet for the container runtimes.([#45194](https://github.com/kubernetes/kubernetes/pull/45194), [@yujuhong](https://github.com/yujuhong))

  * CRI has been moved to package pkg/kubelet/apis/cri/v1alpha1/runtime as part of Kubelet API path cleanup. ([#47113](https://github.com/kubernetes/kubernetes/pull/47113), [@feiskyer](https://github.com/feiskyer))


#### Scheduling

* The fix makes scheduling go routine waiting for cache (e.g. Pod) to be synced. ([#45453](https://github.com/kubernetes/kubernetes/pull/45453), [@k82cn](https://github.com/k82cn))

* Move hardPodAffinitySymmetricWeight to scheduler policy config ([#44159](https://github.com/kubernetes/kubernetes/pull/44159), [@wanghaoran1988](https://github.com/wanghaoran1988))

* Align Extender's validation with prioritizers. ([#45091](https://github.com/kubernetes/kubernetes/pull/45091), [@k82cn](https://github.com/k82cn))

* Removed old scheduler constructor. ([#45472](https://github.com/kubernetes/kubernetes/pull/45472), [@k82cn](https://github.com/k82cn))

* Fixes the overflow for priorityconfig- valid range {1, 9223372036854775806}. ([#45122](https://github.com/kubernetes/kubernetes/pull/45122), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))


#### Security
* Features:

  * Permission to use a PodSecurityPolicy can now be granted within a single namespace by allowing the use verb on the podsecuritypolicies resource within the namespace. ([#42360](https://github.com/kubernetes/kubernetes/pull/42360), [@liggitt](https://github.com/liggitt))

  * Break the 'certificatesigningrequests' controller into a 'csrapprover' controller and 'csrsigner' controller. ([#45514](https://github.com/kubernetes/kubernetes/pull/45514), [@mikedanese](https://github.com/mikedanese))

  * `kubectl auth can-i` now supports non-resource URLs ([#46432](https://github.com/kubernetes/kubernetes/pull/46432), [@CaoShuFeng](https://github.com/CaoShuFeng))

  * Promote kubelet tls bootstrap to beta. Add a non-experimental flag to use it and deprecate the old flag. ([#46799](https://github.com/kubernetes/kubernetes/pull/46799), [@mikedanese](https://github.com/mikedanese))

  * Add the alpha.image-policy.k8s.io/failed-open=true annotation when the image policy webhook encounters an error and fails open. ([#46264](https://github.com/kubernetes/kubernetes/pull/46264), [@Q-Lee](https://github.com/Q-Lee))

  * Add an AEAD encrypting transformer for storing secrets encrypted at rest ([#41939](https://github.com/kubernetes/kubernetes/pull/41939), [@smarterclayton](https://github.com/smarterclayton))

  * Add secretbox and AES-CBC encryption modes to at rest encryption. AES-CBC is considered superior to AES-GCM because it is resistant to nonce-reuse attacks, and secretbox uses Poly1305 and XSalsa20. ([#46916](https://github.com/kubernetes/kubernetes/pull/46916), [@smarterclayton](https://github.com/smarterclayton))

* Bug fixes:

  * Make gcp auth provider not to override the Auth header if it's already exits ([#45575](https://github.com/kubernetes/kubernetes/pull/45575), [@wanghaoran1988](https://github.com/wanghaoran1988))

  * The oidc client plugin has reduce round trips and fix scopes requested ([#45317](https://github.com/kubernetes/kubernetes/pull/45317), [@ericchiang](https://github.com/ericchiang))

  * API requests using impersonation now include the system:authenticated group in the impersonated user automatically. ([#44076](https://github.com/kubernetes/kubernetes/pull/44076), [@liggitt](https://github.com/liggitt))

  * RBAC role and rolebinding auto-reconciliation is now performed only when the RBAC authorization mode is enabled. ([#43813](https://github.com/kubernetes/kubernetes/pull/43813), [@liggitt](https://github.com/liggitt))

  * PodSecurityPolicy now recognizes pods that specify runAsNonRoot: false in their security context and does not overwrite the specified value ([#47073](https://github.com/kubernetes/kubernetes/pull/47073), [@Q-Lee](https://github.com/Q-Lee))

  * Tokens retrieved from Google Cloud with application default credentials will not be cached if the client fails authorization ([#46694](https://github.com/kubernetes/kubernetes/pull/46694), [@matt-tyler](https://github.com/matt-tyler))

  * Update kube-dns, metadata-proxy, and fluentd-gcp, event-exporter, prometheus-to-sd, and ip-masq-agent addons with new base images containing fixes for CVE-2016-4448, CVE-2016-9841, CVE-2016-9843, CVE-2017-1000366, CVE-2017-2616, and CVE-2017-9526. ([#47877](https://github.com/kubernetes/kubernetes/pull/47877), [@ixdy](https://github.com/ixdy))

  * Fixed an issue mounting the wrong secret into pods as a service account token. ([#44102](https://github.com/kubernetes/kubernetes/pull/44102), [@ncdc](https://github.com/ncdc))

#### Scalability

* The HorizontalPodAutoscaler controller will now only send updates when it has new status information, reducing the number of writes caused by the controller. ([#47078](https://github.com/kubernetes/kubernetes/pull/47078), [@DirectXMan12](https://github.com/DirectXMan12))


## **External Dependency Version Information**

Continuous integration builds have used the following versions of external dependencies, however, this is not a strong recommendation and users should consult an appropriate installation or upgrade guide before deciding what versions of etcd, docker or rkt to use.

* Docker versions 1.10.3, 1.11.2, 1.12.6 have been validated

    * Docker version 1.12.6 known issues

        * overlay2 driver not fully supported

        * live-restore not fully supported

        * no shared pid namespace support

    * Docker version 1.11.2 known issues

        * Kernel crash with Aufs storage driver on Debian Jessie ([#27885](https://github.com/kubernetes/kubernetes/pull/27885)) which can be identified by the [node problem detector](https://kubernetes.io/docs/tasks/debug-application-cluster/monitor-node-health/)

        * Leaked File descriptors ([#275](https://github.com/docker/containerd/issues/275))

        * Additional memory overhead per container ([#21737](https://github.com/kubernetes/kubernetes/pull/21737))

    * Docker 1.10.3 contains [backports provided by RedHat](https://github.com/docker/docker/compare/v1.10.3...runcom:docker-1.10.3-stable) for known issues

* For issues with Docker 1.13.X please see the [1.13.X tracking issue](https://github.com/kubernetes/kubernetes/issues/42926)

* rkt version 1.23.0+

    * known issues with the rkt runtime are [listed in the Getting Started Guide](https://kubernetes.io/docs/getting-started-guides/rkt/notes/)

* etcd version 3.0.17

* Go version: 1.8.3. [Link to announcement](https://groups.google.com/d/msg/kubernetes-dev/0XRRz6UhhTM/YODWVnuDBQAJ)

    * Kubernetes can only be compiled with Go 1.8. Support for all other versions is dropped.


### Previous Releases Included in v1.7.0
- [v1.7.0-rc.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v170-rc1)
- [v1.7.0-beta.2](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v170-beta2)
- [v1.7.0-beta.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v170-beta1)
- [v1.7.0-alpha.4](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v170-alpha4)
- [v1.7.0-alpha.3](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v170-alpha3)
- [v1.7.0-alpha.2](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v170-alpha2)
- [v1.7.0-alpha.1](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG.md#v170-alpha1)



# v1.7.0-rc.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.0-rc.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes.tar.gz) | `9da0e04de83e14f87540b5b58f415b5cdb78e552e07dc35985ddb1b7f618a2f2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-src.tar.gz) | `f4e6cfd0d859d7880d14d1052919a9eb79c26e1cd4105330dda8b05f073cab40`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-darwin-386.tar.gz) | `5f161559ce91321577c09f03edf6d3416f1964056644c8725394d9c23089b052`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | `c54b07d2b0240e2be57ff6bf95794bf826a082a7b4e8316c9ec45e92539d6252`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-linux-386.tar.gz) | `d61874a51678dee6cb1e5514e703b7070c27fb728e8b18533a5233fcca2e30fd`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | `4004cec39c637fa7a2e3d309d941f3e73e0a16a3511c5e46cbb2fa6bb27d89e5`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | `88c37ea21d7a2c464be6fee29db4f295d738028871127197253923cec00cf179`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-linux-arm.tar.gz) | `0e5e5f52fe93a78003c6cac171a6aae8cb1f2f761e325d509558df84aba57b32`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | `d4586a64f239654a53faf1a6c18fc5d5c99bb95df593bf92b5e9fac0daba71e2`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | `728097218b051df26b90863779588517183fa4e1f55dee414aff188e4a50e7df`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-windows-386.tar.gz) | `d949bd6977a707b46609ee740f3a16592e7676a6dc81ad495d9f511cb4d2cb98`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | `b787198e3320ef4094112f44e0442f062c04ce2137c14bbec10f5df9fbb3f404`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | `e5eaa8951d021621b160d41bc1350dcf64178c46a0e6e656be78a5e5b267dc5d`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | `08b694b46bf7b5906408a331a9ccfb9143114d414d64fcca8a6daf6ec79c282b`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-server-linux-arm.tar.gz) | `ca980d1669e22cc3846fc2bdf77e6bdc1c49820327128db0d0388c4def77bc16`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | `c656106048696bd2c4b66a3f8e348b37634abf48a9dc1f4eb941e01da9597b26`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | `7888ed82b33b0002a488224ffa7a93e865e1d2b01e4ccc44b8d04ff4be5fef71`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | `26c74018b048e2ec0d2df61216bda77bdf29c23f34dac6d7b8a55a56f0f95927`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | `e5c6d38556f840067b0eea4ca862c5c79a89ff47063dccecf1c0fdc2c25a9a9b`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-node-linux-arm.tar.gz) | `4cf1d7843ede557bd629970d1bc21a936b76bf9138fc96224e538c5a61f6e203`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | `e7a870c53af210cc00f0854e2ffad8ee06b20c4028f256d60d04f31a630291d1`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | `78865fe4029a39744865e0acb4dd15f6f22de8264f7c65a65df52891c3b91967`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | `8b632e7c79e750e7102d02120508f0394d3f11a2c36b42d2c5f96ec4f0f1f1ed`

## Changelog since v1.7.0-beta.2

### Action Required

* The following alpha API groups were unintentionally enabled by default in previous releases, and will no longer be enabled by default in v1.8: ([#47690](https://github.com/kubernetes/kubernetes/pull/47690), [@caesarxuchao](https://github.com/caesarxuchao))
    * rbac.authorization.k8s.io/v1alpha1
    * settings.k8s.io/v1alpha1
    * If you wish to continue using them in v1.8, please enable them explicitly using the `--runtime-config` flag of the apiserver (for example, `--runtime-config="rbac.authorization.k8s.io/v1alpha1,settings.k8s.io/v1alpha1"`)
* Paths containing backsteps (for example, "../bar") are no longer allowed in hostPath volume paths, or in volumeMount subpaths ([#47290](https://github.com/kubernetes/kubernetes/pull/47290), [@jhorwit2](https://github.com/jhorwit2))
* Azure: Change container permissions to private for provisioned volumes. If you have existing Azure volumes that were created by Kubernetes v1.6.0-v1.6.5, you should change the permissions on them manually. ([#47605](https://github.com/kubernetes/kubernetes/pull/47605), [@brendandburns](https://github.com/brendandburns))

### Other notable changes

* Update kube-dns, metadata-proxy, and fluentd-gcp, event-exporter, prometheus-to-sd, and ip-masq-agent addons with new base images containing fixes for CVE-2016-4448, CVE-2016-9841, CVE-2016-9843,  CVE-2017-1000366, CVE-2017-2616, and CVE-2017-9526. ([#47877](https://github.com/kubernetes/kubernetes/pull/47877), [@ixdy](https://github.com/ixdy))
* Bump the memory request/limit for ip-masq-daemon. ([#47887](https://github.com/kubernetes/kubernetes/pull/47887), [@dnardo](https://github.com/dnardo))
* HostAliases is now parsed with `hostAliases` json keys to be in line with the feature's name. ([#47512](https://github.com/kubernetes/kubernetes/pull/47512), [@rickypai](https://github.com/rickypai))
* Fixes issue w/Flex volume, introduced in 1.6.0, where drivers without an attacher would fail (node indefinitely waiting for attach). Drivers that don't implement attach should return `attach: false` on `init`. ([#47503](https://github.com/kubernetes/kubernetes/pull/47503), [@chakri-nelluri](https://github.com/chakri-nelluri))
* Tokens retrieved from Google Cloud with application default credentials will not be cached if the client fails authorization ([#46694](https://github.com/kubernetes/kubernetes/pull/46694), [@matt-tyler](https://github.com/matt-tyler))
* ip-masq-agent is now the default for GCE ([#47794](https://github.com/kubernetes/kubernetes/pull/47794), [@dnardo](https://github.com/dnardo))
* Taints support in gce/salt startup scripts.  ([#47632](https://github.com/kubernetes/kubernetes/pull/47632), [@mwielgus](https://github.com/mwielgus))
* Fix VolumeClaims/capacity in "kubectl describe statefulsets" output. ([#47573](https://github.com/kubernetes/kubernetes/pull/47573), [@k82cn](https://github.com/k82cn))
* New 'service.beta.kubernetes.io/aws-load-balancer-extra-security-groups' Service annotation to specify extra Security Groups to be added to ELB created by AWS cloudprovider ([#45268](https://github.com/kubernetes/kubernetes/pull/45268), [@redbaron](https://github.com/redbaron))
* AWS: clean up blackhole routes when using kubenet ([#47572](https://github.com/kubernetes/kubernetes/pull/47572), [@justinsb](https://github.com/justinsb))
* The protobuf serialization of API objects has been updated to store maps in a predictable order to ensure that the representation of that object does not change when saved into etcd. This prevents the same object from being seen as being modified, even when no values have changed. ([#47701](https://github.com/kubernetes/kubernetes/pull/47701), [@smarterclayton](https://github.com/smarterclayton))
* Mark Static pods on the Master as critical ([#47356](https://github.com/kubernetes/kubernetes/pull/47356), [@dashpole](https://github.com/dashpole))
* kubectl logs with label selector supports specifying a container name ([#44282](https://github.com/kubernetes/kubernetes/pull/44282), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Adds an approval work flow to the the certificate approver that will approve certificate signing requests from kubelets that meet all the criteria of kubelet server certificates. ([#46884](https://github.com/kubernetes/kubernetes/pull/46884), [@jcbsmpsn](https://github.com/jcbsmpsn))
* AWS: Maintain a cache of all instances, to fix problem with > 200 nodes with ELBs ([#47410](https://github.com/kubernetes/kubernetes/pull/47410), [@justinsb](https://github.com/justinsb))
* Bump GLBC version to 0.9.5 - fixes [loss of manually modified GCLB health check settings](https://github.com/kubernetes/kubernetes/issues/47559) upon upgrade from pre-1.6.4 to either 1.6.4 or 1.6.5. ([#47567](https://github.com/kubernetes/kubernetes/pull/47567), [@nicksardo](https://github.com/nicksardo))
* Update cluster-proportional-autoscaler, metadata-proxy, and fluentd-gcp addons with fixes for CVE-2016-4448, CVE-2016-8859, CVE-2016-9841, CVE-2016-9843, and CVE-2017-9526. ([#47545](https://github.com/kubernetes/kubernetes/pull/47545), [@ixdy](https://github.com/ixdy))
* AWS: Batch DescribeInstance calls with nodeNames to 150 limit, to stay within AWS filter limits. ([#47516](https://github.com/kubernetes/kubernetes/pull/47516), [@gnufied](https://github.com/gnufied))



# v1.8.0-alpha.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.8.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes.tar.gz) | `47088d4a0b79ce75a90e73b1dd7f864fc17fe5ff5cea553a072c7a277a70a104`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-src.tar.gz) | `ec2cb19b55e24c7b9728437fb9e39a442c07b68eaea636b2f6bb340e4b9696dc`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `c2fb538ce73f0ed74bd343485cd8873efcff580e4d948ea4bf2732f1b059e463`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `01a1cb673fbb764e47edaea07c1d3fdddd99bbd7b025f9b2498f38c99d5be4b2`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `5bebebf12fb39db8be10f9758a92ce385013d07e629741421b09da88bd9fc0f1`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `b02ae110b3694562b195189c3cb8eca21095153d0cb5552360053304dee425f1`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `e6220b9e62856ad8345cb845c1365b3f177ee22d6f9718f11a1f373d7a70fd21`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `e35c62a3781841898c91724af136fbb35fd99cf15ca5ec947c1a4bc2f6e4a73d`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `7b02c25a764bd367e9931006def88d3fc03cf9e846cce2e77cfbc95f0e206433`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `ab6ba1bf43dd28c776a8cc5cae44413c45a7405f2996c277aba5ee3f6f73e305`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `eb1516db15807111ef03547b0104dcb89a310481ef8f867a65f3c57f20f56e30`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `525e599a2846fe166a5f1eb14483edee9d6b866aa096e16896f6544afad31768`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `bb0a37bb1fefa735ec1eb651fec60c22b180c9bca1bd5e0317e1bcdbf4aa0819`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `68fd804bd1f4d944a25112a67ef8b1cbae55051b110134850715b6f51f93f40c`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `822161bee3e8b3b64bb7cea297264729b3cc6d6a008c86f16b4aef16cde5b0de`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `9354336df2694427e3d6bc9b0b1fe286f3f9a7f6ef8f239bd6319b4af1c02162`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `d4a87e3713f190a4cc7db1f43a6105c3c95e1eb8de45ae269b9bd1ecd52296ce`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `dc7c5865041008fcfdad050380fb33c23a361f7a1f4fbce78b164e2906a1b7f9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `d572cec5ec679e5543e9ee5e2529a51bb8d5ca5f3773e4218c5491a0bd77b7a4`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `4b0fae35ed01ca66fb0f82ea2ea7f804378f592d0c15425dc3934f4b7b6f19a8`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `d5684a2d1a640e7b0fdf82a3faa0edef2b20e50a83ff6baea461699b0d74b583`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `bb444cc79035044cfb58cbe3d7bccd7998522dcf6d993441cf29fd03c249897c`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.8.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `9b54e823c504601193b5ae2d37cb1d297ae9b5acfa1497b6f530a835071a7b6d`

## Changelog since v1.7.0-alpha.4

### Action Required

* The following alpha API groups were unintentionally enabled by default in previous releases, and will no longer be enabled by default in v1.8: ([#47690](https://github.com/kubernetes/kubernetes/pull/47690), [@caesarxuchao](https://github.com/caesarxuchao))
    * rbac.authorization.k8s.io/v1alpha1
    * settings.k8s.io/v1alpha1
    * If you wish to continue using them in v1.8, please enable them explicitly using the `--runtime-config` flag of the apiserver (for example, `--runtime-config="rbac.authorization.k8s.io/v1alpha1,settings.k8s.io/v1alpha1"`)
* Paths containing backsteps (for example, "../bar") are no longer allowed in hostPath volume paths, or in volumeMount subpaths ([#47290](https://github.com/kubernetes/kubernetes/pull/47290), [@jhorwit2](https://github.com/jhorwit2))
* Azure: Change container permissions to private for provisioned volumes. If you have existing Azure volumes that were created by Kubernetes v1.6.0-v1.6.5, you should change the permissions on them manually. ([#47605](https://github.com/kubernetes/kubernetes/pull/47605), [@brendandburns](https://github.com/brendandburns))
* New and upgraded 1.7 GCE/GKE clusters no longer have an RBAC ClusterRoleBinding that grants the `cluster-admin` ClusterRole to the `default` service account in the `kube-system` namespace. ([#46750](https://github.com/kubernetes/kubernetes/pull/46750), [@cjcullen](https://github.com/cjcullen))
    * If this permission is still desired, run the following command to explicitly grant it, either before or after upgrading to 1.7:
    *     kubectl create clusterrolebinding kube-system-default --serviceaccount=kube-system:default --clusterrole=cluster-admin
* kube-apiserver: a new authorization mode (`--authorization-mode=Node`) authorizes nodes to access secrets, configmaps, persistent volume claims and persistent volumes related to their pods. ([#46076](https://github.com/kubernetes/kubernetes/pull/46076), [@liggitt](https://github.com/liggitt))
        * Nodes must use client credentials that place them in the `system:nodes` group with a username of `system:node:<nodeName>` in order to be authorized by the node authorizer (the credentials obtained by the kubelet via TLS bootstrapping satisfy these requirements)
        * When used in combination with the `RBAC` authorization mode (`--authorization-mode=Node,RBAC`), the `system:node` role is no longer automatically granted to the `system:nodes` group.
* kube-controller-manager has dropped support for the `--insecure-experimental-approve-all-kubelet-csrs-for-group` flag. Instead, the `csrapproving` controller uses authorization checks to determine whether to approve certificate signing requests: ([#45619](https://github.com/kubernetes/kubernetes/pull/45619), [@mikedanese](https://github.com/mikedanese))
        * requests for a TLS client certificate for any node are approved if the CSR creator has `create` permission on the `certificatesigningrequests` resource and `nodeclient` subresource in the `certificates.k8s.io` API group
        * requests from a node for a TLS client certificate for itself are approved if the CSR creator has `create` permission on the `certificatesigningrequests` resource and the `selfnodeclient` subresource in the `certificates.k8s.io` API group
        * requests from a node for a TLS serving certificate for itself are approved if the CSR creator has `create` permission on the `certificatesigningrequests` resource and the `selfnodeserver` subresource in the `certificates.k8s.io` API group
* Support updating storageclasses in etcd to storage.k8s.io/v1. You must do this prior to upgrading to 1.8. ([#46116](https://github.com/kubernetes/kubernetes/pull/46116), [@ncdc](https://github.com/ncdc))
* The namespace API object no longer supports the deletecollection operation. ([#46407](https://github.com/kubernetes/kubernetes/pull/46407), [@liggitt](https://github.com/liggitt))
* NetworkPolicy has been moved from `extensions/v1beta1` to the new ([#39164](https://github.com/kubernetes/kubernetes/pull/39164), [@danwinship](https://github.com/danwinship))
	`networking.k8s.io/v1` API group. The structure remains unchanged from
	the beta1 API.
	The `net.beta.kubernetes.io/network-policy` annotation on Namespaces
	to opt in to isolation has been removed. Instead, isolation is now
	determined at a per-pod level, with pods being isolated if there is
	any NetworkPolicy whose spec.podSelector targets them. Pods that are
	targeted by NetworkPolicies accept traffic that is accepted by any of
	the NetworkPolicies (and nothing else), and pods that are not targeted
	by any NetworkPolicy accept all traffic by default.
	Action Required:
	When upgrading to Kubernetes 1.7 (and a network plugin that supports
	the new NetworkPolicy v1 semantics), to ensure full behavioral
	compatibility with v1beta1:
	1. In Namespaces that previously had the "DefaultDeny" annotation,
	   you can create equivalent v1 semantics by creating a
	   NetworkPolicy that matches all pods but does not allow any
	   traffic:

	   ```yaml
           kind: NetworkPolicy
           apiVersion: networking.k8s.io/v1
           metadata:
             name: default-deny
           spec:
             podSelector:
	   ```

	   This will ensure that pods that aren't matched by any other
	   NetworkPolicy will continue to be fully-isolated, as they were
	   before.
	2. In Namespaces that previously did not have the "DefaultDeny"
	   annotation, you should delete any existing NetworkPolicy
	   objects. These would have had no effect before, but with v1
	   semantics they might cause some traffic to be blocked that you
	   didn't intend to be blocked.

### Other notable changes

* kubectl logs with label selector supports specifying a container name ([#44282](https://github.com/kubernetes/kubernetes/pull/44282), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Adds an approval work flow to the the certificate approver that will approve certificate signing requests from kubelets that meet all the criteria of kubelet server certificates. ([#46884](https://github.com/kubernetes/kubernetes/pull/46884), [@jcbsmpsn](https://github.com/jcbsmpsn))
* AWS: Maintain a cache of all instances, to fix problem with > 200 nodes with ELBs ([#47410](https://github.com/kubernetes/kubernetes/pull/47410), [@justinsb](https://github.com/justinsb))
* Bump GLBC version to 0.9.5 - fixes [loss of manually modified GCLB health check settings](https://github.com/kubernetes/kubernetes/issues/47559) upon upgrade from pre-1.6.4 to either 1.6.4 or 1.6.5. ([#47567](https://github.com/kubernetes/kubernetes/pull/47567), [@nicksardo](https://github.com/nicksardo))
* Update cluster-proportional-autoscaler, metadata-proxy, and fluentd-gcp addons with fixes for CVE-2016-4448, CVE-2016-8859, CVE-2016-9841, CVE-2016-9843, and CVE-2017-9526. ([#47545](https://github.com/kubernetes/kubernetes/pull/47545), [@ixdy](https://github.com/ixdy))
* AWS: Batch DescribeInstance calls with nodeNames to 150 limit, to stay within AWS filter limits. ([#47516](https://github.com/kubernetes/kubernetes/pull/47516), [@gnufied](https://github.com/gnufied))
* AWS: Process disk attachments even with duplicate NodeNames ([#47406](https://github.com/kubernetes/kubernetes/pull/47406), [@justinsb](https://github.com/justinsb))
* kubefed will now configure NodeInternalIP as the federation API server endpoint when NodeExternalIP is unavailable for federation API servers exposed as NodePort services ([#46960](https://github.com/kubernetes/kubernetes/pull/46960), [@lukaszo](https://github.com/lukaszo))
* PodSecurityPolicy now recognizes pods that specify `runAsNonRoot: false` in their security context and does not overwrite the specified value ([#47073](https://github.com/kubernetes/kubernetes/pull/47073), [@Q-Lee](https://github.com/Q-Lee))
* Bump GLBC version to 0.9.4 ([#47468](https://github.com/kubernetes/kubernetes/pull/47468), [@nicksardo](https://github.com/nicksardo))
* Stackdriver Logging deployment exposes metrics on node port 31337 when enabled. ([#47402](https://github.com/kubernetes/kubernetes/pull/47402), [@crassirostris](https://github.com/crassirostris))
* Update to kube-addon-manager:v6.4-beta.2: kubectl v1.6.4 and refreshed base images ([#47389](https://github.com/kubernetes/kubernetes/pull/47389), [@ixdy](https://github.com/ixdy))
* Enable iptables -w in kubeadm selfhosted ([#46372](https://github.com/kubernetes/kubernetes/pull/46372), [@cmluciano](https://github.com/cmluciano))
* Azure plugin for client auth ([#43987](https://github.com/kubernetes/kubernetes/pull/43987), [@cosmincojocar](https://github.com/cosmincojocar))
* Fix dynamic provisioning of PVs with inaccurate AccessModes by refusing to provision when PVCs ask for AccessModes that can't be satisfied by the PVs' underlying volume plugin ([#47274](https://github.com/kubernetes/kubernetes/pull/47274), [@wongma7](https://github.com/wongma7))
* AWS: Avoid spurious ELB listener recreation - ignore case when matching protocol ([#47391](https://github.com/kubernetes/kubernetes/pull/47391), [@justinsb](https://github.com/justinsb))
* gce kube-up: The `Node` authorization mode and `NodeRestriction` admission controller are now enabled ([#46796](https://github.com/kubernetes/kubernetes/pull/46796), [@mikedanese](https://github.com/mikedanese))
* update gophercloud/gophercloud dependency for reauthentication fixes ([#45545](https://github.com/kubernetes/kubernetes/pull/45545), [@stuart-warren](https://github.com/stuart-warren))
* fix sync loop health check with separating runtime errors ([#47124](https://github.com/kubernetes/kubernetes/pull/47124), [@andyxning](https://github.com/andyxning))
* servicecontroller: Fix node selection logic on initial LB creation ([#45773](https://github.com/kubernetes/kubernetes/pull/45773), [@justinsb](https://github.com/justinsb))
* Fix iSCSI iSER mounting. ([#47281](https://github.com/kubernetes/kubernetes/pull/47281), [@mtanino](https://github.com/mtanino))
* StorageOS Volume Driver ([#42156](https://github.com/kubernetes/kubernetes/pull/42156), [@croomes](https://github.com/croomes))
    * [StorageOS](http://www.storageos.com) can be used as a storage provider for Kubernetes.  With StorageOS, capacity from local or attached storage is pooled across the cluster, providing converged infrastructure for cloud-native applications. 
* CRI has been moved to package `pkg/kubelet/apis/cri/v1alpha1/runtime`. ([#47113](https://github.com/kubernetes/kubernetes/pull/47113), [@feiskyer](https://github.com/feiskyer))
* Make gcp auth provider not to override the Auth header if it's already exits ([#45575](https://github.com/kubernetes/kubernetes/pull/45575), [@wanghaoran1988](https://github.com/wanghaoran1988))
* Allow pods to opt out of PodPreset mutation via an annotation on the pod. ([#44965](https://github.com/kubernetes/kubernetes/pull/44965), [@jpeeler](https://github.com/jpeeler))
* Add Traditional Chinese translation for kubectl ([#46559](https://github.com/kubernetes/kubernetes/pull/46559), [@warmchang](https://github.com/warmchang))
* Remove Initializers from admission-control in kubernetes-master charm for pre-1.7 ([#46987](https://github.com/kubernetes/kubernetes/pull/46987), [@Cynerva](https://github.com/Cynerva))
* Added state guards to the idle_status messaging in the kubernetes-master charm to make deployment faster on initial deployment. ([#47183](https://github.com/kubernetes/kubernetes/pull/47183), [@chuckbutler](https://github.com/chuckbutler))
* Bump up Node Problem Detector version to v0.4.0, which added support of parsing log from /dev/kmsg and ABRT. ([#46743](https://github.com/kubernetes/kubernetes/pull/46743), [@Random-Liu](https://github.com/Random-Liu))
* kubeadm: Enable the Node Authorizer/Admission plugin in v1.7  ([#46879](https://github.com/kubernetes/kubernetes/pull/46879), [@luxas](https://github.com/luxas))
* Deprecated Binding objects in 1.7. ([#47041](https://github.com/kubernetes/kubernetes/pull/47041), [@k82cn](https://github.com/k82cn))
* Add secretbox and AES-CBC encryption modes to at rest encryption.  AES-CBC is considered superior to AES-GCM because it is resistant to nonce-reuse attacks, and secretbox uses Poly1305 and XSalsa20. ([#46916](https://github.com/kubernetes/kubernetes/pull/46916), [@smarterclayton](https://github.com/smarterclayton))
* The HorizontalPodAutoscaler controller will now only send updates when it has new status information, reducing the number of writes caused by the controller. ([#47078](https://github.com/kubernetes/kubernetes/pull/47078), [@DirectXMan12](https://github.com/DirectXMan12))
* gpusInUse info error when kubelet restarts ([#46087](https://github.com/kubernetes/kubernetes/pull/46087), [@tianshapjq](https://github.com/tianshapjq))
* kubeadm: Modifications to cluster-internal resources installed by kubeadm will be overwritten when upgrading from v1.6 to v1.7. ([#47081](https://github.com/kubernetes/kubernetes/pull/47081), [@luxas](https://github.com/luxas))
* Added exponential backoff to Azure cloudprovider ([#46660](https://github.com/kubernetes/kubernetes/pull/46660), [@jackfrancis](https://github.com/jackfrancis))
* fixed HostAlias in PodSpec to allow `foo.bar` hostnames instead of just `foo` DNS labels. ([#46809](https://github.com/kubernetes/kubernetes/pull/46809), [@rickypai](https://github.com/rickypai))
* Implements rolling update for StatefulSets. Updates can be performed using the RollingUpdate, Paritioned, or OnDelete strategies. OnDelete implements the manual behavior from 1.6. status now tracks  ([#46669](https://github.com/kubernetes/kubernetes/pull/46669), [@kow3ns](https://github.com/kow3ns))
    * replicas, readyReplicas, currentReplicas, and updatedReplicas. The semantics of replicas is now consistent with DaemonSet and ReplicaSet, and readyReplicas has the semantics that replicas did prior to this release.
* Add Japanese translation for kubectl ([#46756](https://github.com/kubernetes/kubernetes/pull/46756), [@girikuncoro](https://github.com/girikuncoro))
* federation: Add admission controller for policy-based placement ([#44786](https://github.com/kubernetes/kubernetes/pull/44786), [@tsandall](https://github.com/tsandall))
* Get command uses OpenAPI schema to enhance display for a resource if run with flag 'use-openapi-print-columns'.  ([#46235](https://github.com/kubernetes/kubernetes/pull/46235), [@droot](https://github.com/droot))
    * An example command:
    * kubectl get pods --use-openapi-print-columns 
* add gzip compression to GET and LIST requests ([#45666](https://github.com/kubernetes/kubernetes/pull/45666), [@ilackarms](https://github.com/ilackarms))
* Fix the bug where container cannot run as root when SecurityContext.RunAsNonRoot is false. ([#47009](https://github.com/kubernetes/kubernetes/pull/47009), [@yujuhong](https://github.com/yujuhong))
* Fixes a bug with cAdvisorPort in the KubeletConfiguration that prevented setting it to 0, which is in fact a valid option, as noted in issue [#11710](https://github.com/kubernetes/kubernetes/pull/11710). ([#46876](https://github.com/kubernetes/kubernetes/pull/46876), [@mtaufen](https://github.com/mtaufen))
* Stackdriver cluster logging now deploys a new component to export Kubernetes events. ([#46700](https://github.com/kubernetes/kubernetes/pull/46700), [@crassirostris](https://github.com/crassirostris))
* Alpha feature: allows users to set storage limit to isolate EmptyDir volumes. It enforces the limit by evicting pods that exceed their storage limits   ([#45686](https://github.com/kubernetes/kubernetes/pull/45686), [@jingxu97](https://github.com/jingxu97))
* Adds the `Categories []string` field to API resources, which represents the list of group aliases (e.g. "all") that every resource belongs to.  ([#43338](https://github.com/kubernetes/kubernetes/pull/43338), [@fabianofranz](https://github.com/fabianofranz))
* Promote kubelet tls bootstrap to beta. Add a non-experimental flag to use it and deprecate the old flag. ([#46799](https://github.com/kubernetes/kubernetes/pull/46799), [@mikedanese](https://github.com/mikedanese))
* Fix disk partition discovery for brtfs ([#46816](https://github.com/kubernetes/kubernetes/pull/46816), [@dashpole](https://github.com/dashpole))
    * Add ZFS support
    * Add overlay2 storage driver support
* Support creation of GCP Internal Load Balancers from Service objects ([#46663](https://github.com/kubernetes/kubernetes/pull/46663), [@nicksardo](https://github.com/nicksardo))
* Introduces status conditions to the HorizontalPodAutoscaler in autoscaling/v2alpha1, indicating the current status of a given HorizontalPodAutoscaler, and why it is or is not scaling. ([#46550](https://github.com/kubernetes/kubernetes/pull/46550), [@DirectXMan12](https://github.com/DirectXMan12))
* Support OpenAPI spec aggregation for kube-aggregator ([#46734](https://github.com/kubernetes/kubernetes/pull/46734), [@mbohlool](https://github.com/mbohlool))
* Implement kubectl rollout undo and history for DaemonSet ([#46144](https://github.com/kubernetes/kubernetes/pull/46144), [@janetkuo](https://github.com/janetkuo))
* Respect PDBs during node upgrades and add test coverage to the ServiceTest upgrade test. ([#45748](https://github.com/kubernetes/kubernetes/pull/45748), [@mml](https://github.com/mml))
* Disk Pressure triggers the deletion of terminated containers on the node. ([#45896](https://github.com/kubernetes/kubernetes/pull/45896), [@dashpole](https://github.com/dashpole))
* Add the `alpha.image-policy.k8s.io/failed-open=true` annotation when the image policy webhook encounters an error and fails open. ([#46264](https://github.com/kubernetes/kubernetes/pull/46264), [@Q-Lee](https://github.com/Q-Lee))
* Enable kubelet csr bootstrap in GCE/GKE ([#40760](https://github.com/kubernetes/kubernetes/pull/40760), [@mikedanese](https://github.com/mikedanese))
* Implement Daemonset history ([#45924](https://github.com/kubernetes/kubernetes/pull/45924), [@janetkuo](https://github.com/janetkuo))
* When switching from the `service.beta.kubernetes.io/external-traffic` annotation to the new ([#46716](https://github.com/kubernetes/kubernetes/pull/46716), [@thockin](https://github.com/thockin))
    * `externalTrafficPolicy` field, the values chnag as follows:
          * "OnlyLocal" becomes "Local"
          * "Global" becomes "Cluster".
* Fix kubelet reset liveness probe failure count across pod restart boundaries ([#46371](https://github.com/kubernetes/kubernetes/pull/46371), [@sjenning](https://github.com/sjenning))
* The gce metadata server can be hidden behind a proxy, hiding the kubelet's token. ([#45565](https://github.com/kubernetes/kubernetes/pull/45565), [@Q-Lee](https://github.com/Q-Lee))
* AWS: Allow configuration of a single security group for ELBs ([#45500](https://github.com/kubernetes/kubernetes/pull/45500), [@nbutton23](https://github.com/nbutton23))
* Allow remote admission controllers to be dynamically added and removed by administrators.  External admission controllers make an HTTP POST containing details of the requested action which the service can approve or reject. ([#46388](https://github.com/kubernetes/kubernetes/pull/46388), [@lavalamp](https://github.com/lavalamp))
* iscsi storage plugin: Fix dangling session when using multiple target portal addresses. ([#46239](https://github.com/kubernetes/kubernetes/pull/46239), [@mtanino](https://github.com/mtanino))
* Duplicate recurring Events now include the latest event's Message string ([#46034](https://github.com/kubernetes/kubernetes/pull/46034), [@kensimon](https://github.com/kensimon))
* With --feature-gates=RotateKubeletClientCertificate=true set, the kubelet will ([#41912](https://github.com/kubernetes/kubernetes/pull/41912), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * request a client certificate from the API server during the boot cycle and pause
    * waiting for the request to be satisfied. It will continually refresh the certificate
    * as the certificates expiration approaches.
* The Kubernetes API supports retrieving tabular output for API resources via a new mime-type `application/json;as=Table;v=v1alpha1;g=meta.k8s.io`.  The returned object (if the server supports it) will be of type `meta.k8s.io/v1alpha1` with `Table`, and contain column and row information related to the resource.  Each row will contain information about the resource - by default it will be the object metadata, but callers can add the `?includeObject=Object` query parameter and receive the full object.  In the future kubectl will use this to retrieve the results of `kubectl get`. ([#40848](https://github.com/kubernetes/kubernetes/pull/40848), [@smarterclayton](https://github.com/smarterclayton))
* This change add nonResourceURL to kubectl auth cani ([#46432](https://github.com/kubernetes/kubernetes/pull/46432), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Webhook added to the API server which omits structured audit log events. ([#45919](https://github.com/kubernetes/kubernetes/pull/45919), [@ericchiang](https://github.com/ericchiang))
* By default, --low-diskspace-threshold-mb is not set, and --eviction-hard includes "nodefs.available<10%,nodefs.inodesFree<5%" ([#46448](https://github.com/kubernetes/kubernetes/pull/46448), [@dashpole](https://github.com/dashpole))
* kubectl edit and kubectl apply will keep the ordering of elements in merged lists ([#45980](https://github.com/kubernetes/kubernetes/pull/45980), [@mengqiy](https://github.com/mengqiy))
* [Federation][kubefed]: Use StorageClassName for etcd pvc ([#46323](https://github.com/kubernetes/kubernetes/pull/46323), [@marun](https://github.com/marun))
* Restrict active deadline seconds max allowed value to be maximum uint32 ([#46640](https://github.com/kubernetes/kubernetes/pull/46640), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Implement kubectl get controllerrevisions ([#46655](https://github.com/kubernetes/kubernetes/pull/46655), [@janetkuo](https://github.com/janetkuo))
* Local storage plugin ([#44897](https://github.com/kubernetes/kubernetes/pull/44897), [@msau42](https://github.com/msau42))
* With `--feature-gates=RotateKubeletServerCertificate=true` set, the kubelet will ([#45059](https://github.com/kubernetes/kubernetes/pull/45059), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * request a server certificate from the API server during the boot cycle and pause
    * waiting for the request to be satisfied. It will continually refresh the certificate as
    * the certificates expiration approaches.
* Allow PSP's to specify a whitelist of allowed paths for host volume based on path prefixes ([#43946](https://github.com/kubernetes/kubernetes/pull/43946), [@jhorwit2](https://github.com/jhorwit2))
* Add `kubectl config rename-context` ([#46114](https://github.com/kubernetes/kubernetes/pull/46114), [@arthur0](https://github.com/arthur0))
* Fix AWS EBS volumes not getting detached from node if routine to verify volumes are attached runs while the node is down ([#46463](https://github.com/kubernetes/kubernetes/pull/46463), [@wongma7](https://github.com/wongma7))
* Move hardPodAffinitySymmetricWeight to scheduler policy config ([#44159](https://github.com/kubernetes/kubernetes/pull/44159), [@wanghaoran1988](https://github.com/wanghaoran1988))
* AWS: support node port health check ([#43585](https://github.com/kubernetes/kubernetes/pull/43585), [@foolusion](https://github.com/foolusion))
* Add generic Toleration for NoExecute Taints to NodeProblemDetector ([#45883](https://github.com/kubernetes/kubernetes/pull/45883), [@gmarek](https://github.com/gmarek))
* support replaceKeys patch strategy and directive for strategic merge patch ([#44597](https://github.com/kubernetes/kubernetes/pull/44597), [@mengqiy](https://github.com/mengqiy))
* Augment CRI to support retrieving container stats from the runtime. ([#45614](https://github.com/kubernetes/kubernetes/pull/45614), [@yujuhong](https://github.com/yujuhong))
* Prevent kubelet from setting allocatable < 0 for a resource upon initial creation. ([#46516](https://github.com/kubernetes/kubernetes/pull/46516), [@derekwaynecarr](https://github.com/derekwaynecarr))
* add --non-resource-url to kubectl create clusterrole ([#45809](https://github.com/kubernetes/kubernetes/pull/45809), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Add `kubectl apply edit-last-applied` subcommand ([#42256](https://github.com/kubernetes/kubernetes/pull/42256), [@shiywang](https://github.com/shiywang))
* Adding admissionregistration API group which enables dynamic registration of initializers and external admission webhooks. It is an alpha feature. ([#46294](https://github.com/kubernetes/kubernetes/pull/46294), [@caesarxuchao](https://github.com/caesarxuchao))
* Fix log spam due to unnecessary status update when node is deleted. ([#45923](https://github.com/kubernetes/kubernetes/pull/45923), [@verult](https://github.com/verult))
* GCE installs will now avoid IP masquerade for all RFC-1918 IP blocks, rather than just 10.0.0.0/8.  This means that clusters can ([#46473](https://github.com/kubernetes/kubernetes/pull/46473), [@thockin](https://github.com/thockin))
    * be created in 192.168.0.0./16 and 172.16.0.0/12 while preserving the container IPs (which would be lost before).
* `set selector` and `set subject` no longer print "running in local/dry-run mode..." at the top, so their output can be piped as valid yaml or json ([#46507](https://github.com/kubernetes/kubernetes/pull/46507), [@bboreham](https://github.com/bboreham))
* ControllerRevision type added for StatefulSet and DaemonSet history. ([#45867](https://github.com/kubernetes/kubernetes/pull/45867), [@kow3ns](https://github.com/kow3ns))
* Bump Go version to 1.8.3 ([#46429](https://github.com/kubernetes/kubernetes/pull/46429), [@wojtek-t](https://github.com/wojtek-t))
* Upgrade Elasticsearch Addon to v5.4.0 ([#45589](https://github.com/kubernetes/kubernetes/pull/45589), [@it-svit](https://github.com/it-svit))
* PodDisruptionBudget now uses ControllerRef to decide which controller owns a given Pod, so it doesn't get confused by controllers with overlapping selectors. ([#45003](https://github.com/kubernetes/kubernetes/pull/45003), [@krmayankk](https://github.com/krmayankk))
* aws: Support for ELB tagging by users ([#45932](https://github.com/kubernetes/kubernetes/pull/45932), [@lpabon](https://github.com/lpabon))
* Portworx volume driver no longer has to run on the master. ([#45518](https://github.com/kubernetes/kubernetes/pull/45518), [@harsh-px](https://github.com/harsh-px))
* kube-proxy: ratelimit runs of iptables by sync-period flags ([#46266](https://github.com/kubernetes/kubernetes/pull/46266), [@thockin](https://github.com/thockin))
* Deployments are updated to use (1) a more stable hashing algorithm (fnv) than the previous one (adler) and (2) a hashing collision avoidance mechanism that will ensure new rollouts will not block on hashing collisions anymore. ([#44774](https://github.com/kubernetes/kubernetes/pull/44774), [@kargakis](https://github.com/kargakis))
* The Prometheus metrics for the kube-apiserver for tracking incoming API requests and latencies now return the `subresource` label for correctly attributing the type of API call. ([#46354](https://github.com/kubernetes/kubernetes/pull/46354), [@smarterclayton](https://github.com/smarterclayton))
* Add Simplified Chinese translation for kubectl ([#45573](https://github.com/kubernetes/kubernetes/pull/45573), [@shiywang](https://github.com/shiywang))
* The --namespace flag is now honored for in-cluster clients that have an empty configuration. ([#46299](https://github.com/kubernetes/kubernetes/pull/46299), [@ncdc](https://github.com/ncdc))
* Fix init container status reporting when active deadline is exceeded. ([#46305](https://github.com/kubernetes/kubernetes/pull/46305), [@sjenning](https://github.com/sjenning))
* Improves performance of Cinder volume attach/detach operations ([#41785](https://github.com/kubernetes/kubernetes/pull/41785), [@jamiehannaford](https://github.com/jamiehannaford))
* GCE and AWS dynamic provisioners extension: admins can configure zone(s) in which a persistent volume shall be created. ([#38505](https://github.com/kubernetes/kubernetes/pull/38505), [@pospispa](https://github.com/pospispa))
* Break the 'certificatesigningrequests' controller into a 'csrapprover' controller and 'csrsigner' controller. ([#45514](https://github.com/kubernetes/kubernetes/pull/45514), [@mikedanese](https://github.com/mikedanese))
* Modifies kubefed to create and the federation controller manager to use credentials associated with a service account rather than the user's credentials. ([#42042](https://github.com/kubernetes/kubernetes/pull/42042), [@perotinus](https://github.com/perotinus))
* Adds a MaxUnavailable field to PodDisruptionBudget ([#45587](https://github.com/kubernetes/kubernetes/pull/45587), [@foxish](https://github.com/foxish))
* The behavior of some watch calls to the server when filtering on fields was incorrect.  If watching objects with a filter, when an update was made that no longer matched the filter a DELETE event was correctly sent.  However, the object that was returned by that delete was not the (correct) version before the update, but instead, the newer version.  That meant the new object was not matched by the filter.  This was a regression from behavior between cached watches on the server side and uncached watches, and thus broke downstream API clients. ([#46223](https://github.com/kubernetes/kubernetes/pull/46223), [@smarterclayton](https://github.com/smarterclayton))
* vSphere cloud provider: vSphere Storage policy Support for dynamic volume provisioning ([#46176](https://github.com/kubernetes/kubernetes/pull/46176), [@BaluDontu](https://github.com/BaluDontu))
* Add support for emitting metrics from openstack cloudprovider about storage operations. ([#46008](https://github.com/kubernetes/kubernetes/pull/46008), [@NickrenREN](https://github.com/NickrenREN))
* 'kubefed init' now supports overriding the default etcd image name with the --etcd-image parameter. ([#46247](https://github.com/kubernetes/kubernetes/pull/46247), [@marun](https://github.com/marun))
* remove the elasticsearch template ([#45952](https://github.com/kubernetes/kubernetes/pull/45952), [@harryge00](https://github.com/harryge00))
* Adds the `CustomResourceDefinition` (crd) types to the `kube-apiserver`.  These are the successors to `ThirdPartyResource`.  See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/thirdpartyresources.md for more details. ([#46055](https://github.com/kubernetes/kubernetes/pull/46055), [@deads2k](https://github.com/deads2k))
* StatefulSets now include an alpha scaling feature accessible by setting the `spec.podManagementPolicy` field to `Parallel`.  The controller will not wait for pods to be ready before adding the other pods, and will replace deleted pods as needed.  Since parallel scaling creates pods out of order, you cannot depend on predictable membership changes within your set. ([#44899](https://github.com/kubernetes/kubernetes/pull/44899), [@smarterclayton](https://github.com/smarterclayton))
* fix kubelet event recording for selected events. ([#46246](https://github.com/kubernetes/kubernetes/pull/46246), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Moved qos to api.helpers. ([#44906](https://github.com/kubernetes/kubernetes/pull/44906), [@k82cn](https://github.com/k82cn))
* Kubelet PLEG updates the relist timestamp only after successfully relisting. ([#45496](https://github.com/kubernetes/kubernetes/pull/45496), [@andyxning](https://github.com/andyxning))
* OpenAPI spec is now available in protobuf binary and gzip format (with ETag support) ([#45836](https://github.com/kubernetes/kubernetes/pull/45836), [@mbohlool](https://github.com/mbohlool))
* Added support to a hierarchy of kubectl plugins (a tree of plugins as children of other plugins). ([#45981](https://github.com/kubernetes/kubernetes/pull/45981), [@fabianofranz](https://github.com/fabianofranz))
    * Added exported env vars to kubectl plugins so that plugin developers have access to global flags, namespace, the plugin descriptor and the full path to the caller binary.
* Ignored mirror pods in PodPreset admission plugin. ([#45958](https://github.com/kubernetes/kubernetes/pull/45958), [@k82cn](https://github.com/k82cn))
* Don't try to attach volume to new node if it is already attached to another node and the volume does not support multi-attach. ([#45346](https://github.com/kubernetes/kubernetes/pull/45346), [@codablock](https://github.com/codablock))
* The Calico version included in kube-up for GCE has been updated to v2.2. ([#38169](https://github.com/kubernetes/kubernetes/pull/38169), [@caseydavenport](https://github.com/caseydavenport))
* Kubelet: Fix image garbage collector attempting to remove in-use images. ([#46121](https://github.com/kubernetes/kubernetes/pull/46121), [@Random-Liu](https://github.com/Random-Liu))
* Add ip-masq-agent addon to the addons folder which is used in GCE if  --non-masquerade-cidr is set to 0/0 ([#46038](https://github.com/kubernetes/kubernetes/pull/46038), [@dnardo](https://github.com/dnardo))
* Fix serialization of EnforceNodeAllocatable ([#44606](https://github.com/kubernetes/kubernetes/pull/44606), [@ivan4th](https://github.com/ivan4th))
* Add --write-config-to flag to kube-proxy to allow users to write the default configuration settings to a file. ([#45908](https://github.com/kubernetes/kubernetes/pull/45908), [@ncdc](https://github.com/ncdc))
* The `NodeRestriction` admission plugin limits the `Node` and `Pod` objects a kubelet can modify. In order to be limited by this admission plugin, kubelets must use credentials in the `system:nodes` group, with a username in the form `system:node:<nodeName>`. Such kubelets will only be allowed to modify their own `Node` API object, and only modify `Pod` API objects that are bound to their node. ([#45929](https://github.com/kubernetes/kubernetes/pull/45929), [@liggitt](https://github.com/liggitt))
* vSphere cloud provider: Report same Node IP as both internal and external. ([#45201](https://github.com/kubernetes/kubernetes/pull/45201), [@abrarshivani](https://github.com/abrarshivani))
* The options passed to a flexvolume plugin's mount command now contains the pod name (`kubernetes.io/pod.name`), namespace (`kubernetes.io/pod.namespace`), uid (`kubernetes.io/pod.uid`), and service account name (`kubernetes.io/serviceAccount.name`). ([#39488](https://github.com/kubernetes/kubernetes/pull/39488), [@liggitt](https://github.com/liggitt))



# v1.7.0-beta.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.0-beta.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes.tar.gz) | `40814fcc343ee49df6a999165486714b5e970d90a368332c8e233a5741306a4c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-src.tar.gz) | `864561a13af5869722276eb0f2d7c0c3bb8946c4ea23551b6a8a68027737cf1b`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `f4802f28767b55b0b29251485482e4db06dc15b257d9e9c8917d47a8531ebc20`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `0a9bb88dec66390e428f499046b35a9e3fbb253d1357006821240f3854fd391e`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-linux-386.tar.gz) | `fbf5c1c9b0d9bfa987936539c8635d809becf2ab447187f6e908ad3d5acebdc5`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `6b56b70519093c87a6a86543bcd137d8bea7b8ae172fdaa2914793baf47883eb`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `ff075b68d0dbbfd04788772d39299f16ee4c1a0f8ff175ed697afca206574707`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `81fec317664151ae318eca49436c9273e106ec869267b453c377544446d865e8`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | `91ee08c0209b767a576164eb6b44450f12ef29dedbca78b3daa447c6516b42fb`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | `28868e4bdd72861c87dd6bce4218fe56e578dd5998cab2da56bde0335904a26b`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-windows-386.tar.gz) | `779e7d864d762af4b039e511e14362426d8e60491a02f5ef571092aac9bc2b22`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `d35a306cb041026625335a330b4edffa8babec8e0b2d90b170ab8f318af87ff6`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `27f71259e3a7e819a6f5ffcf8ad63827f09e928173402e85690ec6943ef3a2fe`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `c9e331c452902293ea00e89ea1944d144c9200b97f033b56f469636c8c7b718d`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `bf3e1b45982ef0a25483bd212553570fa3a1cda49f9a097a9796400fbb70e810`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | `90da52c556b0634241d2da84347537c49b16bfcb0d226afb4213f4ea5a9b80ec`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | `0c4243bae5310764508dba649d8440afbbd11fde2cac3ce651872a9f22694d45`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-node-linux-amd64.tar.gz) | `d6c9d9642c31150b68b8da5143384bd4eee0617e16833d9bbafff94f25a76161`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-node-linux-arm64.tar.gz) | `b91b52b5708539710817a9378295ca4c19afbb75016aa2908c00678709d641ec`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-node-linux-arm.tar.gz) | `3b3421abb90985773745a68159df338eb12c47645434a56c3806dd48e92cb023`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-node-linux-ppc64le.tar.gz) | `a6b843af1284252636cf31a9523ff825c23dee5d57da24bf970031c846242ce5`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-node-linux-s390x.tar.gz) | `43830c0509e9477534661292fc3f4a100250adbee316028c5e869644d75aa478`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.2/kubernetes-node-windows-amd64.tar.gz) | `0ea1ee0dfc483248b3d20177bf023375289214ba153a6466a68764cf02931b52`

## Changelog since v1.7.0-beta.1

### Action Required

* New and upgraded 1.7 GCE/GKE clusters no longer have an RBAC ClusterRoleBinding that grants the `cluster-admin` ClusterRole to the `default` service account in the `kube-system` namespace. ([#46750](https://github.com/kubernetes/kubernetes/pull/46750), [@cjcullen](https://github.com/cjcullen))
    * If this permission is still desired, run the following command to explicitly grant it, either before or after upgrading to 1.7:
    *     kubectl create clusterrolebinding kube-system-default --serviceaccount=kube-system:default --clusterrole=cluster-admin

### Other notable changes

* AWS: Process disk attachments even with duplicate NodeNames ([#47406](https://github.com/kubernetes/kubernetes/pull/47406), [@justinsb](https://github.com/justinsb))
* kubefed will now configure NodeInternalIP as the federation API server endpoint when NodeExternalIP is unavailable for federation API servers exposed as NodePort services ([#46960](https://github.com/kubernetes/kubernetes/pull/46960), [@lukaszo](https://github.com/lukaszo))
* PodSecurityPolicy now recognizes pods that specify `runAsNonRoot: false` in their security context and does not overwrite the specified value ([#47073](https://github.com/kubernetes/kubernetes/pull/47073), [@Q-Lee](https://github.com/Q-Lee))
* Bump GLBC version to 0.9.4 ([#47468](https://github.com/kubernetes/kubernetes/pull/47468), [@nicksardo](https://github.com/nicksardo))
* Stackdriver Logging deployment exposes metrics on node port 31337 when enabled. ([#47402](https://github.com/kubernetes/kubernetes/pull/47402), [@crassirostris](https://github.com/crassirostris))
* Update to kube-addon-manager:v6.4-beta.2: kubectl v1.6.4 and refreshed base images ([#47389](https://github.com/kubernetes/kubernetes/pull/47389), [@ixdy](https://github.com/ixdy))
* Enable iptables -w in kubeadm selfhosted ([#46372](https://github.com/kubernetes/kubernetes/pull/46372), [@cmluciano](https://github.com/cmluciano))
* Azure plugin for client auth ([#43987](https://github.com/kubernetes/kubernetes/pull/43987), [@cosmincojocar](https://github.com/cosmincojocar))
* Fix dynamic provisioning of PVs with inaccurate AccessModes by refusing to provision when PVCs ask for AccessModes that can't be satisfied by the PVs' underlying volume plugin ([#47274](https://github.com/kubernetes/kubernetes/pull/47274), [@wongma7](https://github.com/wongma7))
* AWS: Avoid spurious ELB listener recreation - ignore case when matching protocol ([#47391](https://github.com/kubernetes/kubernetes/pull/47391), [@justinsb](https://github.com/justinsb))
* gce kube-up: The `Node` authorization mode and `NodeRestriction` admission controller are now enabled ([#46796](https://github.com/kubernetes/kubernetes/pull/46796), [@mikedanese](https://github.com/mikedanese))
* update gophercloud/gophercloud dependency for reauthentication fixes ([#45545](https://github.com/kubernetes/kubernetes/pull/45545), [@stuart-warren](https://github.com/stuart-warren))
* fix sync loop health check with separating runtime errors ([#47124](https://github.com/kubernetes/kubernetes/pull/47124), [@andyxning](https://github.com/andyxning))
* servicecontroller: Fix node selection logic on initial LB creation ([#45773](https://github.com/kubernetes/kubernetes/pull/45773), [@justinsb](https://github.com/justinsb))
* Fix iSCSI iSER mounting. ([#47281](https://github.com/kubernetes/kubernetes/pull/47281), [@mtanino](https://github.com/mtanino))
* StorageOS Volume Driver ([#42156](https://github.com/kubernetes/kubernetes/pull/42156), [@croomes](https://github.com/croomes))
    * [StorageOS](http://www.storageos.com) can be used as a storage provider for Kubernetes.  With StorageOS, capacity from local or attached storage is pooled across the cluster, providing converged infrastructure for cloud-native applications. 
* CRI has been moved to package `pkg/kubelet/apis/cri/v1alpha1/runtime`. ([#47113](https://github.com/kubernetes/kubernetes/pull/47113), [@feiskyer](https://github.com/feiskyer))
* Make gcp auth provider not to override the Auth header if it's already exits ([#45575](https://github.com/kubernetes/kubernetes/pull/45575), [@wanghaoran1988](https://github.com/wanghaoran1988))
* Allow pods to opt out of PodPreset mutation via an annotation on the pod. ([#44965](https://github.com/kubernetes/kubernetes/pull/44965), [@jpeeler](https://github.com/jpeeler))
* Add Traditional Chinese translation for kubectl ([#46559](https://github.com/kubernetes/kubernetes/pull/46559), [@warmchang](https://github.com/warmchang))
* Remove Initializers from admission-control in kubernetes-master charm for pre-1.7 ([#46987](https://github.com/kubernetes/kubernetes/pull/46987), [@Cynerva](https://github.com/Cynerva))
* Added state guards to the idle_status messaging in the kubernetes-master charm to make deployment faster on initial deployment. ([#47183](https://github.com/kubernetes/kubernetes/pull/47183), [@chuckbutler](https://github.com/chuckbutler))
* Bump up Node Problem Detector version to v0.4.0, which added support of parsing log from /dev/kmsg and ABRT. ([#46743](https://github.com/kubernetes/kubernetes/pull/46743), [@Random-Liu](https://github.com/Random-Liu))
* kubeadm: Enable the Node Authorizer/Admission plugin in v1.7  ([#46879](https://github.com/kubernetes/kubernetes/pull/46879), [@luxas](https://github.com/luxas))
* Deprecated Binding objects in 1.7. ([#47041](https://github.com/kubernetes/kubernetes/pull/47041), [@k82cn](https://github.com/k82cn))
* Add secretbox and AES-CBC encryption modes to at rest encryption.  AES-CBC is considered superior to AES-GCM because it is resistant to nonce-reuse attacks, and secretbox uses Poly1305 and XSalsa20. ([#46916](https://github.com/kubernetes/kubernetes/pull/46916), [@smarterclayton](https://github.com/smarterclayton))
* The HorizontalPodAutoscaler controller will now only send updates when it has new status information, reducing the number of writes caused by the controller. ([#47078](https://github.com/kubernetes/kubernetes/pull/47078), [@DirectXMan12](https://github.com/DirectXMan12))
* gpusInUse info error when kubelet restarts ([#46087](https://github.com/kubernetes/kubernetes/pull/46087), [@tianshapjq](https://github.com/tianshapjq))
* kubeadm: Modifications to cluster-internal resources installed by kubeadm will be overwritten when upgrading from v1.6 to v1.7. ([#47081](https://github.com/kubernetes/kubernetes/pull/47081), [@luxas](https://github.com/luxas))



# v1.7.0-beta.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.7/examples)

## Downloads for v1.7.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes.tar.gz) | `e2fe83b443544dbb17c5ce481b6b3dcc9e62fbc573b5e270939282a31a910543`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-src.tar.gz) | `321df2749cf4687ec62549bc532eb9e17f159c26f4748732746bce1a4d41e77f`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `308cc980ee14aca49235569302e188dac08879f9236ed405884dada3b4984f44`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `791bc498c2bfd858497d7257500954088bec19dbfeb9809e7c09983fba04f2a6`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-linux-386.tar.gz) | `d9ecac5521cedcc6a94d6b07a57f58f15bb25e43bd766911d2f16cf491a985ac`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `33e800a541a1ce7a89e26dcfaa3650c06cf7239ae22272da944fb0d1288380e1`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `8b245f239ebbede700adac1380f63a71025b8e1f7010e97665c77a0af84effaf`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `730aeeda02e500cc9300c7a555d4e0a1221b7cf182e95e6a9fbe16d90bbbc762`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `7c97431547f40e9dece33e602993c19eab53306e64d16bf44c5e881ba52e5ab4`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `8e95fcc59d9741d67789a8e6370a545c273206f7ff07e19154fe8f0126754571`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-windows-386.tar.gz) | `8bcd3ed7b6081e2a68e5a68cca71632104fef57e96ec5c16191028d113d7e54b`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `1b32e418255f0c6b122b7aba5df9798d37c44c594ac36915ef081076d7464d52`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `2df51991734490871a6d6933ad15e785d543ecae2b06563fc92eb97a019f7eea`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `8c97a97249d644fffbdcd87867e516f1029a3609979379ac4c6ea077f5b5b9b7`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `8e98741d19bd4a51ad275ca6bf793e0c305b75f2ac6569fb553b6cb62daa943e`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `71398347d2aae5345431f4e4c2bedcbdf5c3f406952ce254ef0ae9e4f55355a1`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `1f4fcbc1a70692a57accdab420ad2411acd4672f546473e977ef1c09357418bb`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `b84d291bc3e35912b4da067b3bf328dded87f875dc479b994408a161867c80e5`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `2d306f1e757c49f9358791d7b0176e29f1aa32b6e6d70369b0e40c11a18b2df0`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `3957988bd800514a67ee1cf9e21f99f7e0797810ef3c22fd1604f0b6d1d6dad4`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `f7b3c9c01a25e6afd31dafaeed1eb926f6aae741c0f0967cca2c12492e509fd0`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `de7db84acd32cd7d5b3ac0957cded289335e187539e5495899e05b4043974892`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `efbafcae12ee121cf3a507bba8e36ac43d23d8262dc1a575b85e546ff81030fb`

## Changelog since v1.7.0-alpha.4

### Action Required

* kube-apiserver: a new authorization mode (`--authorization-mode=Node`) authorizes nodes to access secrets, configmaps, persistent volume claims and persistent volumes related to their pods. ([#46076](https://github.com/kubernetes/kubernetes/pull/46076), [@liggitt](https://github.com/liggitt))
        * Nodes must use client credentials that place them in the `system:nodes` group with a username of `system:node:<nodeName>` in order to be authorized by the node authorizer (the credentials obtained by the kubelet via TLS bootstrapping satisfy these requirements)
        * When used in combination with the `RBAC` authorization mode (`--authorization-mode=Node,RBAC`), the `system:node` role is no longer automatically granted to the `system:nodes` group.
* kube-controller-manager has dropped support for the `--insecure-experimental-approve-all-kubelet-csrs-for-group` flag. Instead, the `csrapproving` controller uses authorization checks to determine whether to approve certificate signing requests: ([#45619](https://github.com/kubernetes/kubernetes/pull/45619), [@mikedanese](https://github.com/mikedanese))
        * requests for a TLS client certificate for any node are approved if the CSR creator has `create` permission on the `certificatesigningrequests` resource and `nodeclient` subresource in the `certificates.k8s.io` API group
        * requests from a node for a TLS client certificate for itself are approved if the CSR creator has `create` permission on the `certificatesigningrequests` resource and the `selfnodeclient` subresource in the `certificates.k8s.io` API group
        * requests from a node for a TLS serving certificate for itself are approved if the CSR creator has `create` permission on the `certificatesigningrequests` resource and the `selfnodeserver` subresource in the `certificates.k8s.io` API group
* Support updating storageclasses in etcd to storage.k8s.io/v1. You must do this prior to upgrading to 1.8. ([#46116](https://github.com/kubernetes/kubernetes/pull/46116), [@ncdc](https://github.com/ncdc))
* The namespace API object no longer supports the deletecollection operation. ([#46407](https://github.com/kubernetes/kubernetes/pull/46407), [@liggitt](https://github.com/liggitt))
* NetworkPolicy has been moved from `extensions/v1beta1` to the new ([#39164](https://github.com/kubernetes/kubernetes/pull/39164), [@danwinship](https://github.com/danwinship))
	`networking.k8s.io/v1` API group. The structure remains unchanged from
	the beta1 API.
	The `net.beta.kubernetes.io/network-policy` annotation on Namespaces
	to opt in to isolation has been removed. Instead, isolation is now
	determined at a per-pod level, with pods being isolated if there is
	any NetworkPolicy whose spec.podSelector targets them. Pods that are
	targeted by NetworkPolicies accept traffic that is accepted by any of
	the NetworkPolicies (and nothing else), and pods that are not targeted
	by any NetworkPolicy accept all traffic by default.
	Action Required:
	When upgrading to Kubernetes 1.7 (and a network plugin that supports
	the new NetworkPolicy v1 semantics), to ensure full behavioral
	compatibility with v1beta1:
	1. In Namespaces that previously had the "DefaultDeny" annotation,
	   you can create equivalent v1 semantics by creating a
	   NetworkPolicy that matches all pods but does not allow any
	   traffic:

	   ```yaml
           kind: NetworkPolicy
           apiVersion: networking.k8s.io/v1
           metadata:
             name: default-deny
           spec:
             podSelector:
	   ```

	   This will ensure that pods that aren't matched by any other
	   NetworkPolicy will continue to be fully-isolated, as they were
	   before.
	2. In Namespaces that previously did not have the "DefaultDeny"
	   annotation, you should delete any existing NetworkPolicy
	   objects. These would have had no effect before, but with v1
	   semantics they might cause some traffic to be blocked that you
	   didn't intend to be blocked.

### Other notable changes

* Added exponential backoff to Azure cloudprovider ([#46660](https://github.com/kubernetes/kubernetes/pull/46660), [@jackfrancis](https://github.com/jackfrancis))
* fixed HostAlias in PodSpec to allow `foo.bar` hostnames instead of just `foo` DNS labels. ([#46809](https://github.com/kubernetes/kubernetes/pull/46809), [@rickypai](https://github.com/rickypai))
* Implements rolling update for StatefulSets. Updates can be performed using the RollingUpdate, Paritioned, or OnDelete strategies. OnDelete implements the manual behavior from 1.6. status now tracks  ([#46669](https://github.com/kubernetes/kubernetes/pull/46669), [@kow3ns](https://github.com/kow3ns))
    * replicas, readyReplicas, currentReplicas, and updatedReplicas. The semantics of replicas is now consistent with DaemonSet and ReplicaSet, and readyReplicas has the semantics that replicas did prior to this release.
* Add Japanese translation for kubectl ([#46756](https://github.com/kubernetes/kubernetes/pull/46756), [@girikuncoro](https://github.com/girikuncoro))
* federation: Add admission controller for policy-based placement ([#44786](https://github.com/kubernetes/kubernetes/pull/44786), [@tsandall](https://github.com/tsandall))
* Get command uses OpenAPI schema to enhance display for a resource if run with flag 'use-openapi-print-columns'.  ([#46235](https://github.com/kubernetes/kubernetes/pull/46235), [@droot](https://github.com/droot))
    * An example command:
    * kubectl get pods --use-openapi-print-columns 
* add gzip compression to GET and LIST requests ([#45666](https://github.com/kubernetes/kubernetes/pull/45666), [@ilackarms](https://github.com/ilackarms))
* Fix the bug where container cannot run as root when SecurityContext.RunAsNonRoot is false. ([#47009](https://github.com/kubernetes/kubernetes/pull/47009), [@yujuhong](https://github.com/yujuhong))
* Fixes a bug with cAdvisorPort in the KubeletConfiguration that prevented setting it to 0, which is in fact a valid option, as noted in issue [#11710](https://github.com/kubernetes/kubernetes/pull/11710). ([#46876](https://github.com/kubernetes/kubernetes/pull/46876), [@mtaufen](https://github.com/mtaufen))
* Stackdriver cluster logging now deploys a new component to export Kubernetes events. ([#46700](https://github.com/kubernetes/kubernetes/pull/46700), [@crassirostris](https://github.com/crassirostris))
* Alpha feature: allows users to set storage limit to isolate EmptyDir volumes. It enforces the limit by evicting pods that exceed their storage limits   ([#45686](https://github.com/kubernetes/kubernetes/pull/45686), [@jingxu97](https://github.com/jingxu97))
* Adds the `Categories []string` field to API resources, which represents the list of group aliases (e.g. "all") that every resource belongs to.  ([#43338](https://github.com/kubernetes/kubernetes/pull/43338), [@fabianofranz](https://github.com/fabianofranz))
* Promote kubelet tls bootstrap to beta. Add a non-experimental flag to use it and deprecate the old flag. ([#46799](https://github.com/kubernetes/kubernetes/pull/46799), [@mikedanese](https://github.com/mikedanese))
* Fix disk partition discovery for brtfs ([#46816](https://github.com/kubernetes/kubernetes/pull/46816), [@dashpole](https://github.com/dashpole))
    * Add ZFS support
    * Add overlay2 storage driver support
* Support creation of GCP Internal Load Balancers from Service objects ([#46663](https://github.com/kubernetes/kubernetes/pull/46663), [@nicksardo](https://github.com/nicksardo))
* Introduces status conditions to the HorizontalPodAutoscaler in autoscaling/v2alpha1, indicating the current status of a given HorizontalPodAutoscaler, and why it is or is not scaling. ([#46550](https://github.com/kubernetes/kubernetes/pull/46550), [@DirectXMan12](https://github.com/DirectXMan12))
* Support OpenAPI spec aggregation for kube-aggregator ([#46734](https://github.com/kubernetes/kubernetes/pull/46734), [@mbohlool](https://github.com/mbohlool))
* Implement kubectl rollout undo and history for DaemonSet ([#46144](https://github.com/kubernetes/kubernetes/pull/46144), [@janetkuo](https://github.com/janetkuo))
* Respect PDBs during node upgrades and add test coverage to the ServiceTest upgrade test. ([#45748](https://github.com/kubernetes/kubernetes/pull/45748), [@mml](https://github.com/mml))
* Disk Pressure triggers the deletion of terminated containers on the node. ([#45896](https://github.com/kubernetes/kubernetes/pull/45896), [@dashpole](https://github.com/dashpole))
* Add the `alpha.image-policy.k8s.io/failed-open=true` annotation when the image policy webhook encounters an error and fails open. ([#46264](https://github.com/kubernetes/kubernetes/pull/46264), [@Q-Lee](https://github.com/Q-Lee))
* Enable kubelet csr bootstrap in GCE/GKE ([#40760](https://github.com/kubernetes/kubernetes/pull/40760), [@mikedanese](https://github.com/mikedanese))
* Implement Daemonset history ([#45924](https://github.com/kubernetes/kubernetes/pull/45924), [@janetkuo](https://github.com/janetkuo))
* When switching from the `service.beta.kubernetes.io/external-traffic` annotation to the new ([#46716](https://github.com/kubernetes/kubernetes/pull/46716), [@thockin](https://github.com/thockin))
    * `externalTrafficPolicy` field, the values chnag as follows:
          * "OnlyLocal" becomes "Local"
          * "Global" becomes "Cluster".
* Fix kubelet reset liveness probe failure count across pod restart boundaries ([#46371](https://github.com/kubernetes/kubernetes/pull/46371), [@sjenning](https://github.com/sjenning))
* The gce metadata server can be hidden behind a proxy, hiding the kubelet's token. ([#45565](https://github.com/kubernetes/kubernetes/pull/45565), [@Q-Lee](https://github.com/Q-Lee))
* AWS: Allow configuration of a single security group for ELBs ([#45500](https://github.com/kubernetes/kubernetes/pull/45500), [@nbutton23](https://github.com/nbutton23))
* Allow remote admission controllers to be dynamically added and removed by administrators.  External admission controllers make an HTTP POST containing details of the requested action which the service can approve or reject. ([#46388](https://github.com/kubernetes/kubernetes/pull/46388), [@lavalamp](https://github.com/lavalamp))
* iscsi storage plugin: Fix dangling session when using multiple target portal addresses. ([#46239](https://github.com/kubernetes/kubernetes/pull/46239), [@mtanino](https://github.com/mtanino))
* Duplicate recurring Events now include the latest event's Message string ([#46034](https://github.com/kubernetes/kubernetes/pull/46034), [@kensimon](https://github.com/kensimon))
* With --feature-gates=RotateKubeletClientCertificate=true set, the kubelet will ([#41912](https://github.com/kubernetes/kubernetes/pull/41912), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * request a client certificate from the API server during the boot cycle and pause
    * waiting for the request to be satisfied. It will continually refresh the certificate
    * as the certificates expiration approaches.
* The Kubernetes API supports retrieving tabular output for API resources via a new mime-type `application/json;as=Table;v=v1alpha1;g=meta.k8s.io`.  The returned object (if the server supports it) will be of type `meta.k8s.io/v1alpha1` with `Table`, and contain column and row information related to the resource.  Each row will contain information about the resource - by default it will be the object metadata, but callers can add the `?includeObject=Object` query parameter and receive the full object.  In the future kubectl will use this to retrieve the results of `kubectl get`. ([#40848](https://github.com/kubernetes/kubernetes/pull/40848), [@smarterclayton](https://github.com/smarterclayton))
* This change add nonResourceURL to kubectl auth cani ([#46432](https://github.com/kubernetes/kubernetes/pull/46432), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Webhook added to the API server which omits structured audit log events. ([#45919](https://github.com/kubernetes/kubernetes/pull/45919), [@ericchiang](https://github.com/ericchiang))
* By default, --low-diskspace-threshold-mb is not set, and --eviction-hard includes "nodefs.available<10%,nodefs.inodesFree<5%" ([#46448](https://github.com/kubernetes/kubernetes/pull/46448), [@dashpole](https://github.com/dashpole))
* kubectl edit and kubectl apply will keep the ordering of elements in merged lists ([#45980](https://github.com/kubernetes/kubernetes/pull/45980), [@mengqiy](https://github.com/mengqiy))
* [Federation][kubefed]: Use StorageClassName for etcd pvc ([#46323](https://github.com/kubernetes/kubernetes/pull/46323), [@marun](https://github.com/marun))
* Restrict active deadline seconds max allowed value to be maximum uint32 ([#46640](https://github.com/kubernetes/kubernetes/pull/46640), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Implement kubectl get controllerrevisions ([#46655](https://github.com/kubernetes/kubernetes/pull/46655), [@janetkuo](https://github.com/janetkuo))
* Local storage plugin ([#44897](https://github.com/kubernetes/kubernetes/pull/44897), [@msau42](https://github.com/msau42))
* With `--feature-gates=RotateKubeletServerCertificate=true` set, the kubelet will ([#45059](https://github.com/kubernetes/kubernetes/pull/45059), [@jcbsmpsn](https://github.com/jcbsmpsn))
    * request a server certificate from the API server during the boot cycle and pause
    * waiting for the request to be satisfied. It will continually refresh the certificate as
    * the certificates expiration approaches.
* Allow PSP's to specify a whitelist of allowed paths for host volume based on path prefixes ([#43946](https://github.com/kubernetes/kubernetes/pull/43946), [@jhorwit2](https://github.com/jhorwit2))
* Add `kubectl config rename-context` ([#46114](https://github.com/kubernetes/kubernetes/pull/46114), [@arthur0](https://github.com/arthur0))
* Fix AWS EBS volumes not getting detached from node if routine to verify volumes are attached runs while the node is down ([#46463](https://github.com/kubernetes/kubernetes/pull/46463), [@wongma7](https://github.com/wongma7))
* Move hardPodAffinitySymmetricWeight to scheduler policy config ([#44159](https://github.com/kubernetes/kubernetes/pull/44159), [@wanghaoran1988](https://github.com/wanghaoran1988))
* AWS: support node port health check ([#43585](https://github.com/kubernetes/kubernetes/pull/43585), [@foolusion](https://github.com/foolusion))
* Add generic Toleration for NoExecute Taints to NodeProblemDetector ([#45883](https://github.com/kubernetes/kubernetes/pull/45883), [@gmarek](https://github.com/gmarek))
* support replaceKeys patch strategy and directive for strategic merge patch ([#44597](https://github.com/kubernetes/kubernetes/pull/44597), [@mengqiy](https://github.com/mengqiy))
* Augment CRI to support retrieving container stats from the runtime. ([#45614](https://github.com/kubernetes/kubernetes/pull/45614), [@yujuhong](https://github.com/yujuhong))
* Prevent kubelet from setting allocatable < 0 for a resource upon initial creation. ([#46516](https://github.com/kubernetes/kubernetes/pull/46516), [@derekwaynecarr](https://github.com/derekwaynecarr))
* add --non-resource-url to kubectl create clusterrole ([#45809](https://github.com/kubernetes/kubernetes/pull/45809), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Add `kubectl apply edit-last-applied` subcommand ([#42256](https://github.com/kubernetes/kubernetes/pull/42256), [@shiywang](https://github.com/shiywang))
* Adding admissionregistration API group which enables dynamic registration of initializers and external admission webhooks. It is an alpha feature. ([#46294](https://github.com/kubernetes/kubernetes/pull/46294), [@caesarxuchao](https://github.com/caesarxuchao))
* Fix log spam due to unnecessary status update when node is deleted. ([#45923](https://github.com/kubernetes/kubernetes/pull/45923), [@verult](https://github.com/verult))
* GCE installs will now avoid IP masquerade for all RFC-1918 IP blocks, rather than just 10.0.0.0/8.  This means that clusters can ([#46473](https://github.com/kubernetes/kubernetes/pull/46473), [@thockin](https://github.com/thockin))
    * be created in 192.168.0.0./16 and 172.16.0.0/12 while preserving the container IPs (which would be lost before).
* `set selector` and `set subject` no longer print "running in local/dry-run mode..." at the top, so their output can be piped as valid yaml or json ([#46507](https://github.com/kubernetes/kubernetes/pull/46507), [@bboreham](https://github.com/bboreham))
* ControllerRevision type added for StatefulSet and DaemonSet history. ([#45867](https://github.com/kubernetes/kubernetes/pull/45867), [@kow3ns](https://github.com/kow3ns))
* Bump Go version to 1.8.3 ([#46429](https://github.com/kubernetes/kubernetes/pull/46429), [@wojtek-t](https://github.com/wojtek-t))
* Upgrade Elasticsearch Addon to v5.4.0 ([#45589](https://github.com/kubernetes/kubernetes/pull/45589), [@it-svit](https://github.com/it-svit))
* PodDisruptionBudget now uses ControllerRef to decide which controller owns a given Pod, so it doesn't get confused by controllers with overlapping selectors. ([#45003](https://github.com/kubernetes/kubernetes/pull/45003), [@krmayankk](https://github.com/krmayankk))
* aws: Support for ELB tagging by users ([#45932](https://github.com/kubernetes/kubernetes/pull/45932), [@lpabon](https://github.com/lpabon))
* Portworx volume driver no longer has to run on the master. ([#45518](https://github.com/kubernetes/kubernetes/pull/45518), [@harsh-px](https://github.com/harsh-px))
* kube-proxy: ratelimit runs of iptables by sync-period flags ([#46266](https://github.com/kubernetes/kubernetes/pull/46266), [@thockin](https://github.com/thockin))
* Deployments are updated to use (1) a more stable hashing algorithm (fnv) than the previous one (adler) and (2) a hashing collision avoidance mechanism that will ensure new rollouts will not block on hashing collisions anymore. ([#44774](https://github.com/kubernetes/kubernetes/pull/44774), [@kargakis](https://github.com/kargakis))
* The Prometheus metrics for the kube-apiserver for tracking incoming API requests and latencies now return the `subresource` label for correctly attributing the type of API call. ([#46354](https://github.com/kubernetes/kubernetes/pull/46354), [@smarterclayton](https://github.com/smarterclayton))
* Add Simplified Chinese translation for kubectl ([#45573](https://github.com/kubernetes/kubernetes/pull/45573), [@shiywang](https://github.com/shiywang))
* The --namespace flag is now honored for in-cluster clients that have an empty configuration. ([#46299](https://github.com/kubernetes/kubernetes/pull/46299), [@ncdc](https://github.com/ncdc))
* Fix init container status reporting when active deadline is exceeded. ([#46305](https://github.com/kubernetes/kubernetes/pull/46305), [@sjenning](https://github.com/sjenning))
* Improves performance of Cinder volume attach/detach operations ([#41785](https://github.com/kubernetes/kubernetes/pull/41785), [@jamiehannaford](https://github.com/jamiehannaford))
* GCE and AWS dynamic provisioners extension: admins can configure zone(s) in which a persistent volume shall be created. ([#38505](https://github.com/kubernetes/kubernetes/pull/38505), [@pospispa](https://github.com/pospispa))
* Break the 'certificatesigningrequests' controller into a 'csrapprover' controller and 'csrsigner' controller. ([#45514](https://github.com/kubernetes/kubernetes/pull/45514), [@mikedanese](https://github.com/mikedanese))
* Modifies kubefed to create and the federation controller manager to use credentials associated with a service account rather than the user's credentials. ([#42042](https://github.com/kubernetes/kubernetes/pull/42042), [@perotinus](https://github.com/perotinus))
* Adds a MaxUnavailable field to PodDisruptionBudget ([#45587](https://github.com/kubernetes/kubernetes/pull/45587), [@foxish](https://github.com/foxish))
* The behavior of some watch calls to the server when filtering on fields was incorrect.  If watching objects with a filter, when an update was made that no longer matched the filter a DELETE event was correctly sent.  However, the object that was returned by that delete was not the (correct) version before the update, but instead, the newer version.  That meant the new object was not matched by the filter.  This was a regression from behavior between cached watches on the server side and uncached watches, and thus broke downstream API clients. ([#46223](https://github.com/kubernetes/kubernetes/pull/46223), [@smarterclayton](https://github.com/smarterclayton))
* vSphere cloud provider: vSphere Storage policy Support for dynamic volume provisioning ([#46176](https://github.com/kubernetes/kubernetes/pull/46176), [@BaluDontu](https://github.com/BaluDontu))
* Add support for emitting metrics from openstack cloudprovider about storage operations. ([#46008](https://github.com/kubernetes/kubernetes/pull/46008), [@NickrenREN](https://github.com/NickrenREN))
* 'kubefed init' now supports overriding the default etcd image name with the --etcd-image parameter. ([#46247](https://github.com/kubernetes/kubernetes/pull/46247), [@marun](https://github.com/marun))
* remove the elasticsearch template ([#45952](https://github.com/kubernetes/kubernetes/pull/45952), [@harryge00](https://github.com/harryge00))
* Adds the `CustomResourceDefinition` (crd) types to the `kube-apiserver`.  These are the successors to `ThirdPartyResource`.  See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/api-machinery/thirdpartyresources.md for more details. ([#46055](https://github.com/kubernetes/kubernetes/pull/46055), [@deads2k](https://github.com/deads2k))
* StatefulSets now include an alpha scaling feature accessible by setting the `spec.podManagementPolicy` field to `Parallel`.  The controller will not wait for pods to be ready before adding the other pods, and will replace deleted pods as needed.  Since parallel scaling creates pods out of order, you cannot depend on predictable membership changes within your set. ([#44899](https://github.com/kubernetes/kubernetes/pull/44899), [@smarterclayton](https://github.com/smarterclayton))
* fix kubelet event recording for selected events. ([#46246](https://github.com/kubernetes/kubernetes/pull/46246), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Moved qos to api.helpers. ([#44906](https://github.com/kubernetes/kubernetes/pull/44906), [@k82cn](https://github.com/k82cn))
* Kubelet PLEG updates the relist timestamp only after successfully relisting. ([#45496](https://github.com/kubernetes/kubernetes/pull/45496), [@andyxning](https://github.com/andyxning))
* OpenAPI spec is now available in protobuf binary and gzip format (with ETag support) ([#45836](https://github.com/kubernetes/kubernetes/pull/45836), [@mbohlool](https://github.com/mbohlool))
* Added support to a hierarchy of kubectl plugins (a tree of plugins as children of other plugins). ([#45981](https://github.com/kubernetes/kubernetes/pull/45981), [@fabianofranz](https://github.com/fabianofranz))
    * Added exported env vars to kubectl plugins so that plugin developers have access to global flags, namespace, the plugin descriptor and the full path to the caller binary.
* Ignored mirror pods in PodPreset admission plugin. ([#45958](https://github.com/kubernetes/kubernetes/pull/45958), [@k82cn](https://github.com/k82cn))
* Don't try to attach volume to new node if it is already attached to another node and the volume does not support multi-attach. ([#45346](https://github.com/kubernetes/kubernetes/pull/45346), [@codablock](https://github.com/codablock))
* The Calico version included in kube-up for GCE has been updated to v2.2. ([#38169](https://github.com/kubernetes/kubernetes/pull/38169), [@caseydavenport](https://github.com/caseydavenport))
* Kubelet: Fix image garbage collector attempting to remove in-use images. ([#46121](https://github.com/kubernetes/kubernetes/pull/46121), [@Random-Liu](https://github.com/Random-Liu))
* Add ip-masq-agent addon to the addons folder which is used in GCE if  --non-masquerade-cidr is set to 0/0 ([#46038](https://github.com/kubernetes/kubernetes/pull/46038), [@dnardo](https://github.com/dnardo))
* Fix serialization of EnforceNodeAllocatable ([#44606](https://github.com/kubernetes/kubernetes/pull/44606), [@ivan4th](https://github.com/ivan4th))
* Add --write-config-to flag to kube-proxy to allow users to write the default configuration settings to a file. ([#45908](https://github.com/kubernetes/kubernetes/pull/45908), [@ncdc](https://github.com/ncdc))
* The `NodeRestriction` admission plugin limits the `Node` and `Pod` objects a kubelet can modify. In order to be limited by this admission plugin, kubelets must use credentials in the `system:nodes` group, with a username in the form `system:node:<nodeName>`. Such kubelets will only be allowed to modify their own `Node` API object, and only modify `Pod` API objects that are bound to their node. ([#45929](https://github.com/kubernetes/kubernetes/pull/45929), [@liggitt](https://github.com/liggitt))
* vSphere cloud provider: Report same Node IP as both internal and external. ([#45201](https://github.com/kubernetes/kubernetes/pull/45201), [@abrarshivani](https://github.com/abrarshivani))
* The options passed to a flexvolume plugin's mount command now contains the pod name (`kubernetes.io/pod.name`), namespace (`kubernetes.io/pod.namespace`), uid (`kubernetes.io/pod.uid`), and service account name (`kubernetes.io/serviceAccount.name`). ([#39488](https://github.com/kubernetes/kubernetes/pull/39488), [@liggitt](https://github.com/liggitt))



# v1.7.0-alpha.4

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.7.0-alpha.4


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes.tar.gz) | `14ef2ce3c9348dce7e83aeb167be324da93b90dbb8016f2aecb097c982abf790`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-src.tar.gz) | `faef422988e805a3970985eabff03ed88cfb95ad0d2223abe03011145016e5d0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-darwin-386.tar.gz) | `077dc5637f42a35c316a5e1c3a38e09625971894a186dd7b1e60408c9a0ac4b8`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-darwin-amd64.tar.gz) | `8e43eb7d1969e82eeb17973e4f09e9fe44ff3430cd2c35170d72a631c460deeb`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-linux-386.tar.gz) | `6ddfdbcb25243901c965b1e009e26a90b1fd08d6483906e1235ef380f6f93c97`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-linux-amd64.tar.gz) | `3e7cdd8e0e4d67ff2a0ee2548a4c48a433f84a25384ee9d22c06f4eb2e6db6d7`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-linux-arm64.tar.gz) | `3970c88d2c36fcb43a64d4e889a3eb2cc298e893f6084b9a3c902879d777487d`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-linux-arm.tar.gz) | `156909c55feb06036afff72aa180bd20c14758690cd04c7d8867f49c968e6372`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-linux-ppc64le.tar.gz) | `601fe881a131ce7868fdecfb1439da94ab5a1f1d3700efe4b8319617ceb23d4e`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-linux-s390x.tar.gz) | `2ed3e74e6a972d9ed5b2206fa5e811663497082384f488eada9901e9a99929c7`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-windows-386.tar.gz) | `1aba520fe0bf620f0e77f697194dfd5e336e4a97e2af01f8b94b0f03dbb6299c`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-client-windows-amd64.tar.gz) | `aaf4a42549ea1113915649e636612ea738ead383140d92944c80f3c0d5df8161`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-server-linux-amd64.tar.gz) | `1389c798e7805ec26826c0d3b17ab0d4bd51e0db21cf2f5d4bda5e2b530a6bf1`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-server-linux-arm64.tar.gz) | `ccb99da4b069e63695b3b1d8add9a173e21a0bcaf03d031014460092ec726fb4`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-server-linux-arm.tar.gz) | `6eb3fe27e5017ed834a309cba21342a8c1443486a75ec87611fa66649dd5926a`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-server-linux-ppc64le.tar.gz) | `9b5030b0205ccccfd08b832eec917853fee8bcd34b04033ba35f17698be4a32f`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-server-linux-s390x.tar.gz) | `36b692c221005b52c2a243ddfc16e41a7b157e10fee8662bcd8270280b3f0927`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-node-linux-amd64.tar.gz) | `bba76ad441716f938df0fd8c23c48588d1f80603e39dcca1a29c8b3bbe8c1658`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-node-linux-arm64.tar.gz) | `e3e729847a13fd41ee7f969aabb14d3a0f6f8523f6f079f77a618bf5d781fb9c`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-node-linux-arm.tar.gz) | `520f98f244dd35bb0d96072003548f8b3aacc1e7beb31b5bc527416f07af9d32`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-node-linux-ppc64le.tar.gz) | `686490ba55ea8c7569b3b506f898315c8b1b243de23733e0cd537e2db8e067cb`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-node-linux-s390x.tar.gz) | `a36bb76b390007b271868987739c550c8ac4e856f218f67f2fd780309a2a522e`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.4/kubernetes-node-windows-amd64.tar.gz) | `e78c5a32584d96ec177e38b445c053e40c358e0549b925981c118f4c23578261`

## Changelog since v1.7.0-alpha.3

### Action Required

* `kubectl create role` and `kubectl create clusterrole` no longer allow specifying multiple resource names as comma-separated arguments. Use repeated `--resource-name` arguments to specify multiple resource names.  ([#44950](https://github.com/kubernetes/kubernetes/pull/44950), [@xilabao](https://github.com/xilabao))

### Other notable changes

* avoid concrete examples for missingResourceError ([#45582](https://github.com/kubernetes/kubernetes/pull/45582), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Fix DNS suffix search list support in Windows kube-proxy. ([#45642](https://github.com/kubernetes/kubernetes/pull/45642), [@JiangtianLi](https://github.com/JiangtianLi))
* Fix the bug where StartedAt time is not reported for exited containers. ([#45977](https://github.com/kubernetes/kubernetes/pull/45977), [@yujuhong](https://github.com/yujuhong))
* Update Dashboard version to 1.6.1 ([#45953](https://github.com/kubernetes/kubernetes/pull/45953), [@maciaszczykm](https://github.com/maciaszczykm))
* Examples: fixed cassandra mirror detection that assumes an FTP site will always be presented ([#45965](https://github.com/kubernetes/kubernetes/pull/45965), [@pompomJuice](https://github.com/pompomJuice))
* Removes the deprecated kubelet flag --babysit-daemons ([#44230](https://github.com/kubernetes/kubernetes/pull/44230), [@mtaufen](https://github.com/mtaufen))
* [Federation] Automate configuring nameserver in cluster-dns for CoreDNS provider ([#42895](https://github.com/kubernetes/kubernetes/pull/42895), [@shashidharatd](https://github.com/shashidharatd))
* Add an AEAD encrypting transformer for storing secrets encrypted at rest ([#41939](https://github.com/kubernetes/kubernetes/pull/41939), [@smarterclayton](https://github.com/smarterclayton))
* Update Minio example ([#45444](https://github.com/kubernetes/kubernetes/pull/45444), [@NitishT](https://github.com/NitishT))
* [Federation] Segregate DNS related code to separate controller ([#45034](https://github.com/kubernetes/kubernetes/pull/45034), [@shashidharatd](https://github.com/shashidharatd))
* API Registration is now in beta. ([#45247](https://github.com/kubernetes/kubernetes/pull/45247), [@mbohlool](https://github.com/mbohlool))
* Allow kcm and scheduler to lock on ConfigMaps. ([#45739](https://github.com/kubernetes/kubernetes/pull/45739), [@timothysc](https://github.com/timothysc))
* kubelet config should actually ignore files starting with dots ([#45111](https://github.com/kubernetes/kubernetes/pull/45111), [@dwradcliffe](https://github.com/dwradcliffe))
* Fix lint failures on kubernetes-e2e charm ([#45832](https://github.com/kubernetes/kubernetes/pull/45832), [@Cynerva](https://github.com/Cynerva))
* Mirror pods must now indicate the nodeName they are bound to on creation. The mirror pod annotation is now treated as immutable and cannot be added to an existing pod, removed from a pod, or modified. ([#45775](https://github.com/kubernetes/kubernetes/pull/45775), [@liggitt](https://github.com/liggitt))
* Updating apiserver to return UID of the deleted resource. Clients can use this UID to verify that the resource was deleted or waiting for finalizers. ([#45600](https://github.com/kubernetes/kubernetes/pull/45600), [@nikhiljindal](https://github.com/nikhiljindal))
* OwnerReferencesPermissionEnforcement admission plugin ignores pods/status. ([#45747](https://github.com/kubernetes/kubernetes/pull/45747), [@derekwaynecarr](https://github.com/derekwaynecarr))
* prevent pods/status from touching ownerreferences ([#45826](https://github.com/kubernetes/kubernetes/pull/45826), [@deads2k](https://github.com/deads2k))
* Fix lint errors in juju kubernetes master and e2e charms ([#45494](https://github.com/kubernetes/kubernetes/pull/45494), [@ktsakalozos](https://github.com/ktsakalozos))
* Ensure that autoscaling/v1 is the preferred version for API discovery when autoscaling/v2alpha1 is enabled. ([#45741](https://github.com/kubernetes/kubernetes/pull/45741), [@DirectXMan12](https://github.com/DirectXMan12))
* Promotes Source IP preservation for Virtual IPs to GA. ([#41162](https://github.com/kubernetes/kubernetes/pull/41162), [@MrHohn](https://github.com/MrHohn))
    * Two api fields are defined correspondingly:
    * - Service.Spec.ExternalTrafficPolicy <- 'service.beta.kubernetes.io/external-traffic' annotation.
    * - Service.Spec.HealthCheckNodePort <- 'service.beta.kubernetes.io/healthcheck-nodeport' annotation.
* Fix pods failing to start if they specify a file as a volume subPath to mount. ([#45623](https://github.com/kubernetes/kubernetes/pull/45623), [@wongma7](https://github.com/wongma7))
* the resource quota controller was not adding quota to be resynced at proper interval ([#45685](https://github.com/kubernetes/kubernetes/pull/45685), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Marks the Kubelet's --master-service-namespace flag deprecated ([#44250](https://github.com/kubernetes/kubernetes/pull/44250), [@mtaufen](https://github.com/mtaufen))
* fluentd will tolerate all NoExecute Taints when run in gcp configuration. ([#45715](https://github.com/kubernetes/kubernetes/pull/45715), [@gmarek](https://github.com/gmarek))
* Added Group/Version/Kind and Action extension to OpenAPI Operations  ([#44787](https://github.com/kubernetes/kubernetes/pull/44787), [@mbohlool](https://github.com/mbohlool))
* Updates kube-dns to 1.14.2 ([#45684](https://github.com/kubernetes/kubernetes/pull/45684), [@bowei](https://github.com/bowei))
    * - Support kube-master-url flag without kubeconfig
    * - Fix concurrent R/Ws in dns.go
    * - Fix confusing logging when initialize server
    * - Fix printf in cmd/kube-dns/app/server.go
    * - Fix version on startup and --version flag
    * - Support specifying port number for nameserver in stubDomains
* detach the volume when pod is terminated ([#45286](https://github.com/kubernetes/kubernetes/pull/45286), [@gnufied](https://github.com/gnufied))
* Don't append :443 to registry domain in the kubernetes-worker layer registry action ([#45550](https://github.com/kubernetes/kubernetes/pull/45550), [@jacekn](https://github.com/jacekn))
* vSphere cloud provider: Fix volume detach on node failure. ([#45569](https://github.com/kubernetes/kubernetes/pull/45569), [@divyenpatel](https://github.com/divyenpatel))
* Remove the deprecated `--enable-cri` flag. CRI is now the default,  ([#45194](https://github.com/kubernetes/kubernetes/pull/45194), [@yujuhong](https://github.com/yujuhong))
    * and the only way to integrate with kubelet for the container runtimes.
* AWS: Remove check that forces loadBalancerSourceRanges to be 0.0.0.0/0.  ([#38636](https://github.com/kubernetes/kubernetes/pull/38636), [@dhawal55](https://github.com/dhawal55))
* Fix erroneous FailedSync and FailedMount events being periodically and indefinitely posted on Pods after kubelet is restarted ([#44781](https://github.com/kubernetes/kubernetes/pull/44781), [@wongma7](https://github.com/wongma7))
* Kubernetes now shares a single PID namespace among all containers in a pod when running with docker >= 1.13.1. This means processes can now signal processes in other containers in a pod, but it also means that the `kubectl exec {pod} kill 1` pattern will cause the pod to be restarted rather than a single container. ([#45236](https://github.com/kubernetes/kubernetes/pull/45236), [@verb](https://github.com/verb))
* azure: add support for UDP ports ([#45523](https://github.com/kubernetes/kubernetes/pull/45523), [@colemickens](https://github.com/colemickens))
    * azure: fix support for multiple `loadBalancerSourceRanges`
    * azure: support the Service spec's `sessionAffinity`
* The fix makes scheduling go routine waiting for cache (e.g. Pod) to be synced. ([#45453](https://github.com/kubernetes/kubernetes/pull/45453), [@k82cn](https://github.com/k82cn))
* vSphere cloud provider: Filter out IPV6 node addresses. ([#45181](https://github.com/kubernetes/kubernetes/pull/45181), [@BaluDontu](https://github.com/BaluDontu))
* Default behaviour in cinder storageclass is changed. If availability is not specified, the zone is chosen by algorithm. It makes possible to spread stateful pods across many zones. ([#44798](https://github.com/kubernetes/kubernetes/pull/44798), [@zetaab](https://github.com/zetaab))
* A small clean up to remove unnecessary functions. ([#45018](https://github.com/kubernetes/kubernetes/pull/45018), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Removed old scheduler constructor. ([#45472](https://github.com/kubernetes/kubernetes/pull/45472), [@k82cn](https://github.com/k82cn))
* vSphere cloud provider: Fix fetching of VM UUID on Ubuntu 16.04 and Fedora. ([#45311](https://github.com/kubernetes/kubernetes/pull/45311), [@divyenpatel](https://github.com/divyenpatel))
* This fixes the overflow for priorityconfig-  valid range {1, 9223372036854775806}. ([#45122](https://github.com/kubernetes/kubernetes/pull/45122), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Bump cluster autoscaler to v0.5.4, which fixes scale down issues with pods ignoring SIGTERM. ([#45483](https://github.com/kubernetes/kubernetes/pull/45483), [@mwielgus](https://github.com/mwielgus))
* Create clusters with GPUs in GKE by specifying "type=<gpu-type>,count=<gpu-count>" to NODE_ACCELERATORS env var. ([#45130](https://github.com/kubernetes/kubernetes/pull/45130), [@vishh](https://github.com/vishh))
    * List of available GPUs - https://cloud.google.com/compute/docs/gpus/#introduction
* Remove deprecated node address type `NodeLegacyHostIP`. ([#44830](https://github.com/kubernetes/kubernetes/pull/44830), [@NickrenREN](https://github.com/NickrenREN))
* UIDs and GIDs now use apimachinery types ([#44714](https://github.com/kubernetes/kubernetes/pull/44714), [@jamiehannaford](https://github.com/jamiehannaford))
* Enable basic auth username rotation for GCI ([#44590](https://github.com/kubernetes/kubernetes/pull/44590), [@ihmccreery](https://github.com/ihmccreery))
* Kubectl taint node based on label selector ([#44740](https://github.com/kubernetes/kubernetes/pull/44740), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Scheduler perf modular extensions. ([#44770](https://github.com/kubernetes/kubernetes/pull/44770), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))



# v1.7.0-alpha.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.7.0-alpha.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes.tar.gz) | `03437cacddd91bb7dc21960c960d673ceb99b53040860638aa1d1fbde6d59fb5`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-src.tar.gz) | `190441318abddb44cfcbaec2f1b91d1a76167b91165ce5ae0d1a99c1130a2a36`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `1c3dcc57e014b15395a140eeeb285e38cf5510939b4113d053006d57d8e13087`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `c33d893f67d8ac90834c36284ef88c529c43662c7179e2a9e4b17671c057400b`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `5f3e44b8450db4f93a7ea1f366259c6333007a4536cb242212837bb241c3bbef`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `85ac41dd849f3f9e033d4e123f79c4bd5d7b43bdd877d57dfc8fd2cadcef94be`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `f693032dde194de67900fe8cc5252959d70992b89a24ea43e11e9949835df5db`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `22fa2d2a77310acac1b08a7091929b03977afb2e4a246b054d38b3da15b84e33`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `8717e6042a79f6a79f4527370adb1bbc903b0b9930c6aeee0174687b7443f9d4`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `161c1da92b681decfb9800854bf3b9ff0110ba75c11008a784b891f3a57b032d`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `19f5898a1fdef8c4caf27c6c2b79b0e085127b1d209f57361bce52ca8080842d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `ff79c61efa87af3eeb7357740a495997d223d256b2e54c139572154e113dc247`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `13677b0400758f0d74087768be7abf3fd7bd927f0b874b8d6becc11394cdec2c`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `0a2df3a6ebe157aa8a7e89bd8805dbad3623e122cc0f3614bfcb4ad528bd6ab1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `76611e01de80c07ec954c91612a550063b9efc0c223e5dd638d71f4a3f3d9430`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `2fe29a5871afe693f020e9642e6bc664c497e71598b70673d4f2c4523f57e28b`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `33a1eb93a5d7004987de38ef54e888f0593e31cf9250be3e25118a1d1b474c07`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `de369ca9e5207fb67b26788b41cee1c75935baae348fedc1adf9dbae8c066e7d`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `21839fe6c2a3fd3c165dea6ddbacdec008cdd154c9704866d13ac4dfb14ad7ae`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `2326a074f7c9ba205d996f4f42b8f511c33d909aefd3ea329cc579c4c14b5300`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `58a3aeb5d55d040fd3133dbaa26eb966057ed2b35a5e0522ce8c1ebf4e9b2364`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `2c231a8357d891012574b522ee7fa5e25c6b62b6d888d9bbbb195950cbe18536`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `870bb1ab53a3f2bb5a3c068b425cd6330e71c86dc2ab899c79f733b63ddb51c5`

## Changelog since v1.7.0-alpha.2

### Action Required

* Refactor kube-proxy configuration ([#34727](https://github.com/kubernetes/kubernetes/pull/34727), [@ncdc](https://github.com/ncdc))

### Other notable changes

* kubeadm: Fix invalid assign statement so it is possible to register the master kubelet with other initial Taints ([#45376](https://github.com/kubernetes/kubernetes/pull/45376), [@luxas](https://github.com/luxas))
* Use Docker API Version instead of docker version ([#44068](https://github.com/kubernetes/kubernetes/pull/44068), [@mkumatag](https://github.com/mkumatag))
* bump(golang.org/x/oauth2): a6bd8cefa1811bd24b86f8902872e4e8225f74c4 ([#45056](https://github.com/kubernetes/kubernetes/pull/45056), [@ericchiang](https://github.com/ericchiang))
* apimachinery: make explicit that meta.KindToResource is only a guess ([#45272](https://github.com/kubernetes/kubernetes/pull/45272), [@sttts](https://github.com/sttts))
* Remove PodSandboxStatus.Linux.Namespaces.Network from CRI. ([#45166](https://github.com/kubernetes/kubernetes/pull/45166), [@feiskyer](https://github.com/feiskyer))
* Fixed misspelled http URL in the cluster-dns example ([#45246](https://github.com/kubernetes/kubernetes/pull/45246), [@psiwczak](https://github.com/psiwczak))
* separate discovery from the apiserver ([#43003](https://github.com/kubernetes/kubernetes/pull/43003), [@deads2k](https://github.com/deads2k))
* Remove the `--secret-name` flag from `kubefed join`, instead generating the secret name arbitrarily. ([#42513](https://github.com/kubernetes/kubernetes/pull/42513), [@perotinus](https://github.com/perotinus))
* Added InterPodAffinity unit test case with Namespace. ([#45152](https://github.com/kubernetes/kubernetes/pull/45152), [@k82cn](https://github.com/k82cn))
* Use munged semantic version for side-loaded docker tag ([#44981](https://github.com/kubernetes/kubernetes/pull/44981), [@ixdy](https://github.com/ixdy))
* Increase Dashboard's memory requests and limits ([#44712](https://github.com/kubernetes/kubernetes/pull/44712), [@maciaszczykm](https://github.com/maciaszczykm))
* PodSpec's `HostAliases` now write entries into the Kubernetes-managed hosts file. ([#45148](https://github.com/kubernetes/kubernetes/pull/45148), [@rickypai](https://github.com/rickypai))
* Create and push a docker image for the cloud-controller-manager ([#45154](https://github.com/kubernetes/kubernetes/pull/45154), [@luxas](https://github.com/luxas))
* Align Extender's validation with prioritizers. ([#45091](https://github.com/kubernetes/kubernetes/pull/45091), [@k82cn](https://github.com/k82cn))
* Retry calls we report config changes quickly. ([#44959](https://github.com/kubernetes/kubernetes/pull/44959), [@ktsakalozos](https://github.com/ktsakalozos))
* A new field `hostAliases` has been added to `pod.spec` to support adding entries to a Pod's /etc/hosts file. ([#44641](https://github.com/kubernetes/kubernetes/pull/44641), [@rickypai](https://github.com/rickypai))
* Added CIFS PV support for Juju Charms ([#45117](https://github.com/kubernetes/kubernetes/pull/45117), [@chuckbutler](https://github.com/chuckbutler))
* Some container runtimes share a process (PID) namespace for all containers in a pod. This will become the default for Docker in a future release of Kubernetes. You can preview this functionality if running with the CRI and Docker 1.13.1 by enabling the --experimental-docker-enable-shared-pid kubelet flag. ([#41583](https://github.com/kubernetes/kubernetes/pull/41583), [@verb](https://github.com/verb))
* add APIService conditions ([#43301](https://github.com/kubernetes/kubernetes/pull/43301), [@deads2k](https://github.com/deads2k))
* Log warning when invalid dir passed to kubectl proxy --www ([#44952](https://github.com/kubernetes/kubernetes/pull/44952), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Roll up volume error messages in the kubelet sync loop. ([#44938](https://github.com/kubernetes/kubernetes/pull/44938), [@jayunit100](https://github.com/jayunit100))
* Introduces the ability to extend kubectl by adding third-party plugins. Developer preview, please refer to the documentation for instructions about how to use it. ([#37499](https://github.com/kubernetes/kubernetes/pull/37499), [@fabianofranz](https://github.com/fabianofranz))
* Fixes juju kubernetes master: 1. Get certs from a dead leader. 2. Append tokens. ([#43620](https://github.com/kubernetes/kubernetes/pull/43620), [@ktsakalozos](https://github.com/ktsakalozos))
* Use correct option name in the kubernetes-worker layer registry action ([#44921](https://github.com/kubernetes/kubernetes/pull/44921), [@jacekn](https://github.com/jacekn))
* Start recording cloud provider metrics for AWS ([#43477](https://github.com/kubernetes/kubernetes/pull/43477), [@gnufied](https://github.com/gnufied))
* Bump GLBC version to 0.9.3 ([#45055](https://github.com/kubernetes/kubernetes/pull/45055), [@nicksardo](https://github.com/nicksardo))
* Add metrics to all major gce operations {latency, errors} ([#44510](https://github.com/kubernetes/kubernetes/pull/44510), [@bowei](https://github.com/bowei))
    * The new metrics are:
    *   cloudprovider_gce_api_request_duration_seconds{request, region, zone}
    *   cloudprovider_gce_api_request_errors{request, region, zone}
 
    * `request` is the specific function that is used.
    * `region` is the target region (Will be "<n/a>" if not applicable)
    * `zone` is the target zone (Will be "<n/a>" if not applicable)
    * Note: this fixes some issues with the previous implementation of
    * metrics for disks:
    * - Time duration tracked was of the initial API call, not the entire
    *   operation.
    * - Metrics label tuple would have resulted in many independent
    *   histograms stored, one for each disk. (Did not aggregate well).
* Update kubernetes-e2e charm to use snaps ([#45044](https://github.com/kubernetes/kubernetes/pull/45044), [@Cynerva](https://github.com/Cynerva))
* Log the error (if any) in e2e metrics gathering step ([#45039](https://github.com/kubernetes/kubernetes/pull/45039), [@shyamjvs](https://github.com/shyamjvs))
* The proxy subresource APIs for nodes, services, and pods now support the HTTP PATCH method. ([#44929](https://github.com/kubernetes/kubernetes/pull/44929), [@liggitt](https://github.com/liggitt))
* cluster-autoscaler: Fix duplicate writing of logs. ([#45017](https://github.com/kubernetes/kubernetes/pull/45017), [@MaciekPytel](https://github.com/MaciekPytel))
* CRI: Fix StopContainer timeout ([#44970](https://github.com/kubernetes/kubernetes/pull/44970), [@Random-Liu](https://github.com/Random-Liu))
* Fixes a bug where pods were evicted even after images are successfully deleted. ([#44986](https://github.com/kubernetes/kubernetes/pull/44986), [@dashpole](https://github.com/dashpole))
* Fix some false negatives in detection of meaningful conflicts during strategic merge patch with maps and lists. ([#43469](https://github.com/kubernetes/kubernetes/pull/43469), [@enisoc](https://github.com/enisoc))
* kubernetes-master juju charm properly detects etcd-scale events and reconfigures appropriately. ([#44967](https://github.com/kubernetes/kubernetes/pull/44967), [@chuckbutler](https://github.com/chuckbutler))
* Add redirect support to SpdyRoundTripper ([#44451](https://github.com/kubernetes/kubernetes/pull/44451), [@ncdc](https://github.com/ncdc))
* Support running Ubuntu image on GCE node ([#44744](https://github.com/kubernetes/kubernetes/pull/44744), [@yguo0905](https://github.com/yguo0905))
* Send dns details only after cdk-addons are configured ([#44945](https://github.com/kubernetes/kubernetes/pull/44945), [@ktsakalozos](https://github.com/ktsakalozos))
* Added support to the pause action in the kubernetes-worker charm for new flag --delete-local-data ([#44931](https://github.com/kubernetes/kubernetes/pull/44931), [@chuckbutler](https://github.com/chuckbutler))
* Upgrade go version to v1.8 ([#41636](https://github.com/kubernetes/kubernetes/pull/41636), [@luxas](https://github.com/luxas))
* Add namespace-{list, create, delete} actions to the kubernetes-master layer ([#44277](https://github.com/kubernetes/kubernetes/pull/44277), [@jacekn](https://github.com/jacekn))
* Fix problems with scaling up the cluster when unschedulable pods have some persistent volume claims. ([#44860](https://github.com/kubernetes/kubernetes/pull/44860), [@mwielgus](https://github.com/mwielgus))
* Feature/hpa upscale downscale delay configurable ([#42101](https://github.com/kubernetes/kubernetes/pull/42101), [@Dmitry1987](https://github.com/Dmitry1987))
* Add short name "netpol" for networkpolicies ([#42241](https://github.com/kubernetes/kubernetes/pull/42241), [@xiangpengzhao](https://github.com/xiangpengzhao))
* Restored the ability of kubectl running inside a pod to consume resource files specifying a different namespace than the one the pod is running in. ([#44862](https://github.com/kubernetes/kubernetes/pull/44862), [@liggitt](https://github.com/liggitt))
* e2e: handle nil ReplicaSet in checkDeploymentRevision ([#44859](https://github.com/kubernetes/kubernetes/pull/44859), [@sttts](https://github.com/sttts))
* Fix false positive "meaningful conflict" detection for strategic merge patch with integer values. ([#44788](https://github.com/kubernetes/kubernetes/pull/44788), [@enisoc](https://github.com/enisoc))
* Documented NodePort networking for CDK. ([#44863](https://github.com/kubernetes/kubernetes/pull/44863), [@chuckbutler](https://github.com/chuckbutler))
* Deployments and DaemonSets are now considered complete once all of the new pods are up and running - affects `kubectl rollout status` (and ProgressDeadlineSeconds for Deployments) ([#44672](https://github.com/kubernetes/kubernetes/pull/44672), [@kargakis](https://github.com/kargakis))
* Exclude nodes labeled as master from LoadBalancer / NodePort; restores documented behaviour. ([#44745](https://github.com/kubernetes/kubernetes/pull/44745), [@justinsb](https://github.com/justinsb))
* Fixes issue during LB creation where ports where incorrectly assigned to a floating IP ([#44387](https://github.com/kubernetes/kubernetes/pull/44387), [@jamiehannaford](https://github.com/jamiehannaford))
* Remove redis-proxy.yaml sample, as the image is nowhere to be found. ([#44801](https://github.com/kubernetes/kubernetes/pull/44801), [@klausenbusk](https://github.com/klausenbusk))
* Resolves juju vsphere hostname bug showing only a single node in a scaled node-pool. ([#44780](https://github.com/kubernetes/kubernetes/pull/44780), [@chuckbutler](https://github.com/chuckbutler))
* kubectl commands run inside a pod using a kubeconfig file now use the namespace specified in the kubeconfig file, instead of using the pod namespace. If no kubeconfig file is used, or the kubeconfig does not specify a namespace, the pod namespace is still used as a fallback. ([#44570](https://github.com/kubernetes/kubernetes/pull/44570), [@liggitt](https://github.com/liggitt))
* This adds support for CNI ConfigLists, which permit plugin chaining. ([#42202](https://github.com/kubernetes/kubernetes/pull/42202), [@squeed](https://github.com/squeed))
* API requests using impersonation now include the `system:authenticated` group in the impersonated user automatically. ([#44076](https://github.com/kubernetes/kubernetes/pull/44076), [@liggitt](https://github.com/liggitt))
* Print conditions of RC/RS in 'kubectl describe' command. ([#44710](https://github.com/kubernetes/kubernetes/pull/44710), [@xiangpengzhao](https://github.com/xiangpengzhao))
* cinder: Add support for the KVM virtio-scsi driver ([#41498](https://github.com/kubernetes/kubernetes/pull/41498), [@mikebryant](https://github.com/mikebryant))
* Disallows installation of upstream docker from PPA in the Juju kubernetes-worker charm. ([#44681](https://github.com/kubernetes/kubernetes/pull/44681), [@wwwtyro](https://github.com/wwwtyro))
* Fluentd manifest pod is no longer created on non-registered master when creating clusters using kube-up.sh. ([#44721](https://github.com/kubernetes/kubernetes/pull/44721), [@piosz](https://github.com/piosz))
* Job controller now respects ControllerRef to avoid fighting over Pods. ([#42176](https://github.com/kubernetes/kubernetes/pull/42176), [@enisoc](https://github.com/enisoc))
* CronJob controller now respects ControllerRef to avoid fighting with other controllers. ([#42177](https://github.com/kubernetes/kubernetes/pull/42177), [@enisoc](https://github.com/enisoc))
* The hyperkube image has been slimmed down and no longer includes addon manifests and other various scripts. These were introduced for the now removed docker-multinode setup system. ([#44555](https://github.com/kubernetes/kubernetes/pull/44555), [@luxas](https://github.com/luxas))
* Refactoring reorganize taints function in kubectl to expose operations ([#43171](https://github.com/kubernetes/kubernetes/pull/43171), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* The Kubernetes API server now exits if it encounters a networking failure (e.g. the networking interface hosting its address goes away) to allow a process manager (systemd/kubelet/etc) to react to the problem.  Previously the server would log the failure and try again to bind to its configured address:port. ([#42272](https://github.com/kubernetes/kubernetes/pull/42272), [@marun](https://github.com/marun))
* Fixes a bug in the kubernetes-worker Juju charm code that attempted to give kube-proxy more than one api endpoint. ([#44677](https://github.com/kubernetes/kubernetes/pull/44677), [@wwwtyro](https://github.com/wwwtyro))
* Fixes a missing comma in a list of strings. ([#44678](https://github.com/kubernetes/kubernetes/pull/44678), [@wwwtyro](https://github.com/wwwtyro))
* Fix ceph-secret type to kubernetes.io/rbd in kubernetes-master charm ([#44635](https://github.com/kubernetes/kubernetes/pull/44635), [@Cynerva](https://github.com/Cynerva))



# v1.7.0-alpha.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.7.0-alpha.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes.tar.gz) | `d60465c07b8aa4b5bc8e3de98769d72d22985489e5cdfd1a3165e36c755d6c3b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-src.tar.gz) | `b0b388571225e37a5b9bca6624a92e69273af907cdb300a6d0ac6a0d0d364bd4`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `55b04bc43c45bd93cf30174036ad64109ca1070ab3b331882e956f483dac2b6a`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `d61c055ca90aacb6feb10f45feaaf11f188052598cfef79f4930358bb37e09ad`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `e10ce9339ee6158759675bfb002409fa7f70c701aa5a8a5ac97abc56742561b7`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `b9cb60ba71dfa144ed1e6f2116afd078782372d427912838c56f3b77a74afda0`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `bc0446c484dba91d8f1e32c0175b81dca5c6ff0ac9f5dd3f69cff529afb83aff`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `f794765ca98a2c0611fda32756250eff743c25b66cd4d973fc5720a55771c1c6`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `216cb6e96ba6af5ae259c069576fcd873c48a8a4e8918f5e08ac13427fbefd57`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `fb7903d028744fdfe3119ade6b2ee71532e3d69a82bd5834206fe84e50821253`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `6bdfbd12361f814c86f268dcc807314f322efe9390ca2d91087e617814e91684`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `fd26fc5f0e967b9f6ab18bc28893f2037712891179ddb67b035434c94612f7e3`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `e14c0748789f6a1c3840ab05d0ad5b796a0f03722ee923f8208740f702c0bc19`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `270e0a6fcc0a2f38c8c6e8929a4a593535014bde88f69479a52c5b625bca435c`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `0bd58c2f8d8b6e8110354ccd71eb97eb873aca7b074ce9f83dab4f62a696e964`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `57a4a5dcdb573fb6dc08dbd53d0f196c66d245fa2159a92bf8da0d29128e486d`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `404c8dcc300281f5588e6f4dd15e3c41f858c6597e37a817913112d545a7f736`

## Changelog since v1.7.0-alpha.1

### Action Required

* `kubectl create rolebinding` and `kubectl create clusterrolebinding` no longer allow specifying multiple subjects as comma-separated arguments. Use repeated `--user`, `--group`, or `--serviceaccount` arguments to specify multiple subjects.  ([#43903](https://github.com/kubernetes/kubernetes/pull/43903), [@xilabao](https://github.com/xilabao))

### Other notable changes

* Add support for Azure internal load balancer ([#43510](https://github.com/kubernetes/kubernetes/pull/43510), [@karataliu](https://github.com/karataliu))
* Improved output on 'kubectl get' and 'kubectl describe' for generic objects. ([#44222](https://github.com/kubernetes/kubernetes/pull/44222), [@fabianofranz](https://github.com/fabianofranz))
* Add Kubernetes 1.6 support to Juju charms ([#44500](https://github.com/kubernetes/kubernetes/pull/44500), [@Cynerva](https://github.com/Cynerva))
    * Add metric collection to charms for autoscaling
    * Update kubernetes-e2e charm to fail when test suite fails
    * Update Juju charms to use snaps
    * Add registry action to the kubernetes-worker charm
    * Add support for kube-proxy cluster-cidr option to kubernetes-worker charm
    * Fix kubernetes-master charm starting services before TLS certs are saved
    * Fix kubernetes-worker charm failures in LXD
    * Fix stop hook failure on kubernetes-worker charm
    * Fix handling of juju kubernetes-worker.restart-needed state
    * Fix nagios checks in charms
* Users can now specify listen and advertise URLs for etcd in a kubeadm cluster  ([#42246](https://github.com/kubernetes/kubernetes/pull/42246), [@jamiehannaford](https://github.com/jamiehannaford))
* Fixed `kubectl cluster-info dump` to support multi-container pod. ([#44088](https://github.com/kubernetes/kubernetes/pull/44088), [@xingzhou](https://github.com/xingzhou))
* Prints out status updates when running `kubefed init` ([#41849](https://github.com/kubernetes/kubernetes/pull/41849), [@perotinus](https://github.com/perotinus))
* CRI: Fix kubelet failing to start when using rkt. ([#44569](https://github.com/kubernetes/kubernetes/pull/44569), [@yujuhong](https://github.com/yujuhong))
* Remove deprecatedPublicIPs field ([#44519](https://github.com/kubernetes/kubernetes/pull/44519), [@thockin](https://github.com/thockin))
* Remove deprecated ubuntu kube-up deployment. ([#44344](https://github.com/kubernetes/kubernetes/pull/44344), [@mikedanese](https://github.com/mikedanese))
* Use OS-specific libs when computing client User-Agent in kubectl, etc. ([#44423](https://github.com/kubernetes/kubernetes/pull/44423), [@monopole](https://github.com/monopole))
* kube-apiserver now drops unneeded path information if an older version of Windows kubectl sends it. ([#44421](https://github.com/kubernetes/kubernetes/pull/44421), [@mml](https://github.com/mml))
* Extending the gc admission plugin so that a user who doesn't have delete permission of the *owner* cannot modify blockOwnerDeletion field of existing ownerReferences, or add new ownerReference with blockOwnerDeletion=true ([#43876](https://github.com/kubernetes/kubernetes/pull/43876), [@caesarxuchao](https://github.com/caesarxuchao))
* kube-apiserver: --service-account-lookup now defaults to true, requiring the Secret API object containing the token to exist in order for a service account token to be valid. This enables service account tokens to be revoked by deleting the Secret object containing the token. ([#44071](https://github.com/kubernetes/kubernetes/pull/44071), [@liggitt](https://github.com/liggitt))
* CRI: `kubectl logs -f` now stops following when container stops, as it did pre-CRI. ([#44406](https://github.com/kubernetes/kubernetes/pull/44406), [@Random-Liu](https://github.com/Random-Liu))
* Add completion support for --namespace and --cluster to kubectl ([#44251](https://github.com/kubernetes/kubernetes/pull/44251), [@superbrothers](https://github.com/superbrothers))
* dnsprovider: avoid panic if route53 fields are nil ([#44380](https://github.com/kubernetes/kubernetes/pull/44380), [@justinsb](https://github.com/justinsb))
* In 'kubectl describe', find controllers with ControllerRef, instead of showing the original creator. ([#42849](https://github.com/kubernetes/kubernetes/pull/42849), [@janetkuo](https://github.com/janetkuo))
* Heat cluster operations now support environments that have multiple Swift URLs ([#41561](https://github.com/kubernetes/kubernetes/pull/41561), [@jamiehannaford](https://github.com/jamiehannaford))
* Adds support for allocation of pod IPs via IP aliases. ([#42147](https://github.com/kubernetes/kubernetes/pull/42147), [@bowei](https://github.com/bowei))
* alpha volume provisioning is removed and default storage class should be used instead. ([#44090](https://github.com/kubernetes/kubernetes/pull/44090), [@NickrenREN](https://github.com/NickrenREN))
* validateClusterInfo: use clientcmdapi.NewCluster() ([#44221](https://github.com/kubernetes/kubernetes/pull/44221), [@ncdc](https://github.com/ncdc))
* Fix corner-case with OnlyLocal Service healthchecks. ([#44313](https://github.com/kubernetes/kubernetes/pull/44313), [@thockin](https://github.com/thockin))
* Adds annotations to all Federation objects created by kubefed. ([#42683](https://github.com/kubernetes/kubernetes/pull/42683), [@perotinus](https://github.com/perotinus))
* [Federation][Kubefed] Bug fix to enable disabling federation controllers through override args ([#44209](https://github.com/kubernetes/kubernetes/pull/44209), [@irfanurrehman](https://github.com/irfanurrehman))
* [Federation] Remove deprecated federation-apiserver-kubeconfig secret ([#44287](https://github.com/kubernetes/kubernetes/pull/44287), [@shashidharatd](https://github.com/shashidharatd))
* Scheduler can receive its policy configuration from a ConfigMap ([#43892](https://github.com/kubernetes/kubernetes/pull/43892), [@bsalamat](https://github.com/bsalamat))
* AWS cloud provider: fix support running the master with a different AWS account or even on a different cloud provider than the nodes. ([#44235](https://github.com/kubernetes/kubernetes/pull/44235), [@mrIncompetent](https://github.com/mrIncompetent))
* add rancher credential provider ([#40160](https://github.com/kubernetes/kubernetes/pull/40160), [@wlan0](https://github.com/wlan0))
* Support generating Open API extensions for strategic merge patch tags in go struct tags ([#44121](https://github.com/kubernetes/kubernetes/pull/44121), [@mbohlool](https://github.com/mbohlool))
* Use go1.8.1 for arm and ppc64le ([#44216](https://github.com/kubernetes/kubernetes/pull/44216), [@mkumatag](https://github.com/mkumatag))
* Aggregated used ports at the NodeInfo level for `PodFitsHostPorts` predicate. ([#42524](https://github.com/kubernetes/kubernetes/pull/42524), [@k82cn](https://github.com/k82cn))
* Catch error when failed to make directory in NFS volume plugin ([#38801](https://github.com/kubernetes/kubernetes/pull/38801), [@nak3](https://github.com/nak3))
* Support iSCSI CHAP authentication ([#43396](https://github.com/kubernetes/kubernetes/pull/43396), [@rootfs](https://github.com/rootfs))
* Support context completion for kubectl config use-context ([#42336](https://github.com/kubernetes/kubernetes/pull/42336), [@superbrothers](https://github.com/superbrothers))
* print warning when delete current context ([#42538](https://github.com/kubernetes/kubernetes/pull/42538), [@adohe](https://github.com/adohe))
* Add node e2e tests for hostPid ([#44119](https://github.com/kubernetes/kubernetes/pull/44119), [@feiskyer](https://github.com/feiskyer))
* kubeadm: Make `kubeadm reset` tolerant of a disabled docker service. ([#43951](https://github.com/kubernetes/kubernetes/pull/43951), [@luxas](https://github.com/luxas))
* kubelet: make dockershim.sock configurable ([#43914](https://github.com/kubernetes/kubernetes/pull/43914), [@ncdc](https://github.com/ncdc))
* Fix [broken service accounts when using dedicated service account key](https://github.com/kubernetes/kubernetes/issues/44285). ([#44169](https://github.com/kubernetes/kubernetes/pull/44169), [@mikedanese](https://github.com/mikedanese))
* Fix incorrect conflict errors applying strategic merge patches to resources. ([#43871](https://github.com/kubernetes/kubernetes/pull/43871), [@liggitt](https://github.com/liggitt))
* Fix [transition between NotReady and Unreachable taints](https://github.com/kubernetes/kubernetes/issues/43444). ([#44042](https://github.com/kubernetes/kubernetes/pull/44042), [@gmarek](https://github.com/gmarek))
* leader election lock based on scheduler name ([#42961](https://github.com/kubernetes/kubernetes/pull/42961), [@wanghaoran1988](https://github.com/wanghaoran1988))
* [Federation] Remove FEDERATIONS_DOMAIN_MAP references ([#43137](https://github.com/kubernetes/kubernetes/pull/43137), [@shashidharatd](https://github.com/shashidharatd))
* Fix for [federation failing to propagate cascading deletion](https://github.com/kubernetes/kubernetes/issues/44304). ([#44108](https://github.com/kubernetes/kubernetes/pull/44108), [@csbell](https://github.com/csbell))
* Fix bug with service nodeports that have no backends not being rejected, when they should be.  This is not a regression vs v1.5 - it's a fix that didn't quite fix hard enough. ([#43972](https://github.com/kubernetes/kubernetes/pull/43972), [@thockin](https://github.com/thockin))
* Fix for [failure to delete federation controllers with finalizers](https://github.com/kubernetes/kubernetes/issues/43828). ([#44084](https://github.com/kubernetes/kubernetes/pull/44084), [@nikhiljindal](https://github.com/nikhiljindal))
* Fix container hostPid settings. ([#44097](https://github.com/kubernetes/kubernetes/pull/44097), [@feiskyer](https://github.com/feiskyer))
* Fixed an issue mounting the wrong secret into pods as a service account token. ([#44102](https://github.com/kubernetes/kubernetes/pull/44102), [@ncdc](https://github.com/ncdc))



# v1.7.0-alpha.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.7.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes.tar.gz) | `a8430f678ae5abb16909183bb6472d49084b26c2990854dac73f55be69941435`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-src.tar.gz) | `09792d0b31c3c0f085f54a62c0d151029026cee3c57ac8c3456751ef2243967f`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `115543a5ec55f9039136e0ecfd90d6510b146075d13987fad9c03db3761fbac6`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `91b7cc89386041125af2ecafd3c6e73197f0b7af3ec817d9aed4822e1543eee9`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `7a77bfec2873907ad1f955e33414a9afa029d37d90849bf652e7bab1f2c668ed`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `674d1a839869ac308f3a273ab41be42dab8b52e96526effdbd268255ab6ad4c1`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `4b0164b0474987df5829dcd88c0cdf2d16dbcba30a03cd0ad5ca860d6b4a2f3f`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `cb5a941c3e61465eab544c7b23acd4be6969d74ac23bd9370aa3f9dfc24f2b42`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `d583aff4c86de142b5e6e23cd5c8eb9617fea6574acede9fa2420169405429c6`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `ab14c4806b4e9c7a41993924467969886e1288216d80d2d077a2c35f26fc8cc5`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `0af3f9d1193d9ea49bb4e1cb46142b846b70ceb49ab47ad6fc2497a0dc88395d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `12a9dffda6ba8916149b681f49af506790be97275fe6fc16552ac765aef20a99`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `d6b4c285a89172692e4ba82b777cc9df5b2f5061caa0a9cef6add246a848eeb9`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `e73fb04d4ff692f19de09cfc3cfa17014e23df4150b26c20c3329f688c164358`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `98763b72ba6652abfd5b671981506f8c35ab522d34af34636e5095413769eeb5`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `b39dbb0dc96dcdf1ec4cbd5788e00e46c0d11efb42c6dbdec64758aa8aa9d8e5`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.7.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `c0171e2f22c4e51f25185e71387301ad2c0ade90139fe96dec1c2f999de71716`

## Changelog since v1.6.0

### Other notable changes

* Juju: Enable GPU mode if GPU hardware detected ([#43467](https://github.com/kubernetes/kubernetes/pull/43467), [@tvansteenburgh](https://github.com/tvansteenburgh))
* Check the error before parsing the apiversion ([#44047](https://github.com/kubernetes/kubernetes/pull/44047), [@yujuhong](https://github.com/yujuhong))
* get-kube-local.sh checks pods with option "--namespace=kube-system" ([#42518](https://github.com/kubernetes/kubernetes/pull/42518), [@mtanino](https://github.com/mtanino))
* Using http2 in kubeapi-load-balancer to fix kubectl exec uses ([#43625](https://github.com/kubernetes/kubernetes/pull/43625), [@mbruzek](https://github.com/mbruzek))
* Support status.hostIP in downward API ([#42717](https://github.com/kubernetes/kubernetes/pull/42717), [@andrewsykim](https://github.com/andrewsykim))
* AWS cloud provider: allow to set KubernetesClusterID or KubernetesClusterTag in combination with VPC. ([#42512](https://github.com/kubernetes/kubernetes/pull/42512), [@scheeles](https://github.com/scheeles))
* changed kubelet default image-gc-high-threshold to 85% to resolve a conflict with default settings in docker that prevented image garbage collection from resolving low disk space situations when using devicemapper storage. ([#40432](https://github.com/kubernetes/kubernetes/pull/40432), [@sjenning](https://github.com/sjenning))
* When creating a container using envFrom, ([#42083](https://github.com/kubernetes/kubernetes/pull/42083), [@fraenkel](https://github.com/fraenkel))
    * 1. validate the name of the ConfigMap in a ConfigMapRef
    * 2. validate the name of the Secret in a SecretRef
* RBAC role and rolebinding auto-reconciliation is now performed only when the RBAC authorization mode is enabled. ([#43813](https://github.com/kubernetes/kubernetes/pull/43813), [@liggitt](https://github.com/liggitt))
* Permission to use a PodSecurityPolicy can now be granted within a single namespace by allowing the `use` verb on the `podsecuritypolicies` resource within the namespace. ([#42360](https://github.com/kubernetes/kubernetes/pull/42360), [@liggitt](https://github.com/liggitt))
* Enable audit log in local cluster ([#42379](https://github.com/kubernetes/kubernetes/pull/42379), [@xilabao](https://github.com/xilabao))
* Fix a deadlock in kubeadm master initialization. ([#43835](https://github.com/kubernetes/kubernetes/pull/43835), [@mikedanese](https://github.com/mikedanese))
* Implement API usage metrics for gce storage ([#40338](https://github.com/kubernetes/kubernetes/pull/40338), [@gnufied](https://github.com/gnufied))
* kubeadm: clean up exited containers and network checkpoints ([#43836](https://github.com/kubernetes/kubernetes/pull/43836), [@yujuhong](https://github.com/yujuhong))
* ActiveDeadlineSeconds is validated in workload controllers now, make sure it's not set anywhere (it shouldn't be set by default and having it set means your controller will restart the Pods at some point) ([#38741](https://github.com/kubernetes/kubernetes/pull/38741), [@sandflee](https://github.com/sandflee))
* azure: all clients poll duration is now 5 seconds ([#43699](https://github.com/kubernetes/kubernetes/pull/43699), [@colemickens](https://github.com/colemickens))
* addressing issue [#39427](https://github.com/kubernetes/kubernetes/pull/39427) adding a flag --output to 'kubectl version' ([#39858](https://github.com/kubernetes/kubernetes/pull/39858), [@alejandroEsc](https://github.com/alejandroEsc))
* Support secure etcd cluster for centos provider. ([#42994](https://github.com/kubernetes/kubernetes/pull/42994), [@Shawyeok](https://github.com/Shawyeok))
* Use Cluster Autoscaler 0.5.1, which fixes an issue in Cluster Autoscaler 0.5 where the cluster may be scaled up unnecessarily. Also the status of Cluster Autoscaler is now exposed in kube-system/cluster-autoscaler-status config map. ([#43745](https://github.com/kubernetes/kubernetes/pull/43745), [@mwielgus](https://github.com/mwielgus))
* Use ProviderID to address nodes in the cloudprovider ([#42604](https://github.com/kubernetes/kubernetes/pull/42604), [@wlan0](https://github.com/wlan0))
* Openstack cinder v1/v2/auto API support ([#40423](https://github.com/kubernetes/kubernetes/pull/40423), [@mkutsevol](https://github.com/mkutsevol))
* API resource discovery now includes the `singularName` used to refer to the resource. ([#43312](https://github.com/kubernetes/kubernetes/pull/43312), [@deads2k](https://github.com/deads2k))
* Add the ability to lock on ConfigMaps to support HA for self hosted components ([#42666](https://github.com/kubernetes/kubernetes/pull/42666), [@timothysc](https://github.com/timothysc))
* OpenStack clusters can now specify whether worker nodes are assigned a floating IP ([#42638](https://github.com/kubernetes/kubernetes/pull/42638), [@jamiehannaford](https://github.com/jamiehannaford))
* Add Host field to TCPSocketAction ([#42902](https://github.com/kubernetes/kubernetes/pull/42902), [@louyihua](https://github.com/louyihua))
* Support StorageClass in Azure file volume ([#42170](https://github.com/kubernetes/kubernetes/pull/42170), [@rootfs](https://github.com/rootfs))
* Be able to specify the timeout to wait for pod for kubectl logs/attach ([#41813](https://github.com/kubernetes/kubernetes/pull/41813), [@shiywang](https://github.com/shiywang))
* Add support for bring-your-own ip address for Services on Azure ([#42034](https://github.com/kubernetes/kubernetes/pull/42034), [@brendandburns](https://github.com/brendandburns))
* 1. create configmap has a new option --from-env-file that populates a configmap from file which follows a key=val format for each line. ([#38882](https://github.com/kubernetes/kubernetes/pull/38882), [@fraenkel](https://github.com/fraenkel))
    * 2. create secret has a new option --from-env-file that populates a configmap from file which follows a key=val format for each line.
* update the signing key for percona debian and ubuntu packages ([#41186](https://github.com/kubernetes/kubernetes/pull/41186), [@dixudx](https://github.com/dixudx))
* fc: Drop multipath.conf snippet ([#36698](https://github.com/kubernetes/kubernetes/pull/36698), [@fabiand](https://github.com/fabiand))

Please see the [Releases Page](https://github.com/kubernetes/kubernetes/releases) for older releases.

Release notes of older releases can be found in:
- [CHANGELOG-1.2.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.2.md)
- [CHANGELOG-1.3.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.3.md)
- [CHANGELOG-1.4.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.4.md)
- [CHANGELOG-1.5.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.5.md)
- [CHANGELOG-1.6.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.6.md)

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CHANGELOG.md?pixel)]()
