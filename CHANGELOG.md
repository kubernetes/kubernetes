<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.7.4](#v174)
  - [Downloads for v1.7.4](#downloads-for-v174)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.7.3](#changelog-since-v173)
    - [Other notable changes](#other-notable-changes)
- [v1.6.8](#v168)
  - [Downloads for v1.6.8](#downloads-for-v168)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.6.7](#changelog-since-v167)
    - [Other notable changes](#other-notable-changes-1)
- [v1.7.3](#v173)
  - [Downloads for v1.7.3](#downloads-for-v173)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.7.2](#changelog-since-v172)
    - [Other notable changes](#other-notable-changes-2)
- [v1.7.2](#v172)
  - [Downloads for v1.7.2](#downloads-for-v172)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.7.1](#changelog-since-v171)
    - [Other notable changes](#other-notable-changes-3)
- [v1.7.1](#v171)
  - [Downloads for v1.7.1](#downloads-for-v171)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.7.0](#changelog-since-v170)
    - [Other notable changes](#other-notable-changes-4)
- [v1.8.0-alpha.2](#v180-alpha2)
  - [Downloads for v1.8.0-alpha.2](#downloads-for-v180-alpha2)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.7.0](#changelog-since-v170-1)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-5)
- [v1.6.7](#v167)
  - [Downloads for v1.6.7](#downloads-for-v167)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.6.6](#changelog-since-v166)
    - [Other notable changes](#other-notable-changes-6)
- [v1.7.0](#v170)
  - [Downloads for v1.7.0](#downloads-for-v170)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
  - [**Major Themes**](#major-themes)
  - [**Action Required Before Upgrading**](#action-required-before-upgrading)
    - [Network](#network)
    - [Storage](#storage)
    - [API Machinery](#api-machinery)
    - [Controller Manager](#controller-manager)
    - [kubectl (CLI)](#kubectl-cli)
    - [kubeadm](#kubeadm)
    - [Cloud Providers](#cloud-providers)
  - [**Known Issues**](#known-issues)
  - [**Deprecations**](#deprecations)
    - [Cluster provisioning scripts](#cluster-provisioning-scripts)
    - [Client libraries](#client-libraries)
    - [DaemonSet](#daemonset)
    - [kube-proxy](#kube-proxy)
    - [Namespace](#namespace)
    - [Scheduling](#scheduling)
  - [**Notable Features**](#notable-features)
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
    - [**Cluster Lifecycle**](#cluster-lifecycle)
      - [kubeadm](#kubeadm-1)
      - [Cloud Provider Support](#cloud-provider-support)
    - [**Cluster Federation**](#cluster-federation)
      - [Placement Policy](#placement-policy)
      - [Cluster Selection](#cluster-selection)
    - [**Instrumentation**](#instrumentation)
      - [Core Metrics API](#core-metrics-api)
    - [**Internationalization**](#internationalization)
    - [**kubectl (CLI)**](#kubectl-cli-1)
    - [**Networking**](#networking)
      - [Network Policy](#network-policy)
      - [Load Balancing](#load-balancing)
    - [**Node Components**](#node-components)
      - [Container Runtime Interface](#container-runtime-interface)
    - [**Scheduling**](#scheduling-1)
      - [Scheduler Extender](#scheduler-extender)
    - [**Storage**](#storage-1)
      - [Local Storage](#local-storage)
      - [Volume Plugins](#volume-plugins)
      - [Metrics](#metrics)
    - [**Other notable changes**](#other-notable-changes-7)
      - [Admission plugin](#admission-plugin)
      - [API Machinery](#api-machinery-1)
      - [Application autoscaling](#application-autoscaling-1)
      - [Application Deployment](#application-deployment-1)
      - [Cluster Autoscaling](#cluster-autoscaling)
      - [Cloud Provider Enhancement](#cloud-provider-enhancement)
      - [Cluster Provisioning](#cluster-provisioning)
      - [Cluster federation](#cluster-federation-1)
      - [Credential provider](#credential-provider)
      - [Information for Kubernetes clients (openapi, swagger, client-go)](#information-for-kubernetes-clients-openapi-swagger-client-go)
      - [Instrumentation](#instrumentation-1)
      - [Internal storage layer](#internal-storage-layer)
      - [Kubernetes Dashboard](#kubernetes-dashboard)
      - [kube-dns](#kube-dns)
      - [kube-proxy](#kube-proxy-1)
      - [kube-scheduler](#kube-scheduler)
      - [Storage](#storage-2)
      - [Networking](#networking-1)
      - [Node controller](#node-controller)
      - [Node Components](#node-components-1)
      - [Scheduling](#scheduling-2)
      - [Security](#security-1)
      - [Scalability](#scalability)
  - [**External Dependency Version Information**](#external-dependency-version-information)
    - [Previous Releases Included in v1.7.0](#previous-releases-included-in-v170)
- [v1.7.0-rc.1](#v170-rc1)
  - [Downloads for v1.7.0-rc.1](#downloads-for-v170-rc1)
    - [Client Binaries](#client-binaries-8)
    - [Server Binaries](#server-binaries-8)
    - [Node Binaries](#node-binaries-8)
  - [Changelog since v1.7.0-beta.2](#changelog-since-v170-beta2)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-8)
- [v1.8.0-alpha.1](#v180-alpha1)
  - [Downloads for v1.8.0-alpha.1](#downloads-for-v180-alpha1)
    - [Client Binaries](#client-binaries-9)
    - [Server Binaries](#server-binaries-9)
    - [Node Binaries](#node-binaries-9)
  - [Changelog since v1.7.0-alpha.4](#changelog-since-v170-alpha4)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-9)
- [v1.6.6](#v166)
  - [Downloads for v1.6.6](#downloads-for-v166)
    - [Client Binaries](#client-binaries-10)
    - [Server Binaries](#server-binaries-10)
    - [Node Binaries](#node-binaries-10)
  - [Changelog since v1.6.5](#changelog-since-v165)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-10)
- [v1.7.0-beta.2](#v170-beta2)
  - [Downloads for v1.7.0-beta.2](#downloads-for-v170-beta2)
    - [Client Binaries](#client-binaries-11)
    - [Server Binaries](#server-binaries-11)
    - [Node Binaries](#node-binaries-11)
  - [Changelog since v1.7.0-beta.1](#changelog-since-v170-beta1)
    - [Action Required](#action-required-4)
    - [Other notable changes](#other-notable-changes-11)
- [v1.6.5](#v165)
  - [Known Issues for v1.6.5](#known-issues-for-v165)
  - [Downloads for v1.6.5](#downloads-for-v165)
    - [Client Binaries](#client-binaries-12)
    - [Server Binaries](#server-binaries-12)
    - [Node Binaries](#node-binaries-12)
  - [Changelog since v1.6.4](#changelog-since-v164)
    - [Other notable changes](#other-notable-changes-12)
- [v1.7.0-beta.1](#v170-beta1)
  - [Downloads for v1.7.0-beta.1](#downloads-for-v170-beta1)
    - [Client Binaries](#client-binaries-13)
    - [Server Binaries](#server-binaries-13)
    - [Node Binaries](#node-binaries-13)
  - [Changelog since v1.7.0-alpha.4](#changelog-since-v170-alpha4-1)
    - [Action Required](#action-required-5)
    - [Other notable changes](#other-notable-changes-13)
- [v1.6.4](#v164)
  - [Known Issues for v1.6.4](#known-issues-for-v164)
  - [Downloads for v1.6.4](#downloads-for-v164)
    - [Client Binaries](#client-binaries-14)
    - [Server Binaries](#server-binaries-14)
    - [Node Binaries](#node-binaries-14)
  - [Changelog since v1.6.3](#changelog-since-v163)
    - [Other notable changes](#other-notable-changes-14)
- [v1.7.0-alpha.4](#v170-alpha4)
  - [Downloads for v1.7.0-alpha.4](#downloads-for-v170-alpha4)
    - [Client Binaries](#client-binaries-15)
    - [Server Binaries](#server-binaries-15)
    - [Node Binaries](#node-binaries-15)
  - [Changelog since v1.7.0-alpha.3](#changelog-since-v170-alpha3)
    - [Action Required](#action-required-6)
    - [Other notable changes](#other-notable-changes-15)
- [v1.6.3](#v163)
  - [Known Issues for v1.6.3](#known-issues-for-v163)
  - [Downloads for v1.6.3](#downloads-for-v163)
    - [Client Binaries](#client-binaries-16)
    - [Server Binaries](#server-binaries-16)
    - [Node Binaries](#node-binaries-16)
  - [Changelog since v1.6.2](#changelog-since-v162)
    - [Other notable changes](#other-notable-changes-16)
- [v1.7.0-alpha.3](#v170-alpha3)
  - [Downloads for v1.7.0-alpha.3](#downloads-for-v170-alpha3)
    - [Client Binaries](#client-binaries-17)
    - [Server Binaries](#server-binaries-17)
    - [Node Binaries](#node-binaries-17)
  - [Changelog since v1.7.0-alpha.2](#changelog-since-v170-alpha2)
    - [Action Required](#action-required-7)
    - [Other notable changes](#other-notable-changes-17)
- [v1.5.7](#v157)
  - [Downloads for v1.5.7](#downloads-for-v157)
    - [Client Binaries](#client-binaries-18)
    - [Server Binaries](#server-binaries-18)
    - [Node Binaries](#node-binaries-18)
  - [Changelog since v1.5.6](#changelog-since-v156)
    - [Other notable changes](#other-notable-changes-18)
- [v1.4.12](#v1412)
  - [Downloads for v1.4.12](#downloads-for-v1412)
    - [Client Binaries](#client-binaries-19)
    - [Server Binaries](#server-binaries-19)
    - [Node Binaries](#node-binaries-19)
  - [Changelog since v1.4.9](#changelog-since-v149)
    - [Other notable changes](#other-notable-changes-19)
- [v1.7.0-alpha.2](#v170-alpha2)
  - [Downloads for v1.7.0-alpha.2](#downloads-for-v170-alpha2)
    - [Client Binaries](#client-binaries-20)
    - [Server Binaries](#server-binaries-20)
  - [Changelog since v1.7.0-alpha.1](#changelog-since-v170-alpha1)
    - [Action Required](#action-required-8)
    - [Other notable changes](#other-notable-changes-20)
- [v1.6.2](#v162)
  - [Downloads for v1.6.2](#downloads-for-v162)
    - [Client Binaries](#client-binaries-21)
    - [Server Binaries](#server-binaries-21)
  - [Changelog since v1.6.1](#changelog-since-v161)
    - [Other notable changes](#other-notable-changes-21)
- [v1.7.0-alpha.1](#v170-alpha1)
  - [Downloads for v1.7.0-alpha.1](#downloads-for-v170-alpha1)
    - [Client Binaries](#client-binaries-22)
    - [Server Binaries](#server-binaries-22)
  - [Changelog since v1.6.0](#changelog-since-v160)
    - [Other notable changes](#other-notable-changes-22)
- [v1.6.1](#v161)
  - [Downloads for v1.6.1](#downloads-for-v161)
    - [Client Binaries](#client-binaries-23)
    - [Server Binaries](#server-binaries-23)
  - [Changelog since v1.6.0](#changelog-since-v160-1)
    - [Other notable changes](#other-notable-changes-23)
- [v1.6.0](#v160)
  - [Downloads for v1.6.0](#downloads-for-v160)
    - [Client Binaries](#client-binaries-24)
    - [Server Binaries](#server-binaries-24)
  - [WARNING: etcd backup strongly recommended](#warning:-etcd-backup-strongly-recommended)
  - [Major updates and release themes](#major-updates-and-release-themes)
  - [Action Required](#action-required-9)
    - [Certificates API](#certificates-api)
    - [Cluster Autoscaler](#cluster-autoscaler)
    - [Deployment](#deployment)
    - [Federation](#federation)
    - [Internal Storage Layer](#internal-storage-layer-1)
    - [Node Components](#node-components-2)
    - [kubectl](#kubectl)
    - [RBAC](#rbac)
    - [Scheduling](#scheduling-3)
    - [Service](#service)
    - [StatefulSet](#statefulset-1)
    - [Volumes](#volumes)
  - [Notable Features](#notable-features-1)
    - [Autoscaling](#autoscaling)
    - [DaemonSet](#daemonset-2)
    - [Deployment](#deployment-1)
    - [Federation](#federation-1)
    - [Internal Storage Layer](#internal-storage-layer-2)
    - [kubeadm](#kubeadm-2)
    - [Node Components](#node-components-3)
    - [RBAC](#rbac-1)
    - [Scheduling](#scheduling-4)
    - [Service Catalog](#service-catalog)
    - [Volumes](#volumes-1)
  - [Deprecations](#deprecations-1)
    - [Cluster Provisioning Scripts](#cluster-provisioning-scripts-1)
    - [kubeadm](#kubeadm-3)
    - [Other Deprecations](#other-deprecations)
  - [Changes to API Resources](#changes-to-api-resources)
    - [ABAC](#abac)
    - [Admission Control](#admission-control-1)
    - [Authentication](#authentication)
    - [Authorization](#authorization)
    - [Autoscaling](#autoscaling-1)
    - [Certificates](#certificates)
    - [ConfigMap](#configmap)
    - [CronJob](#cronjob)
    - [DaemonSet](#daemonset-3)
    - [Deployment](#deployment-2)
    - [Node](#node)
    - [Pod](#pod)
    - [Pod Security Policy](#pod-security-policy)
    - [RBAC](#rbac-2)
    - [ReplicaSet](#replicaset)
    - [Secrets](#secrets)
    - [Service](#service-1)
    - [StatefulSet](#statefulset-2)
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
      - [kubefed](#kubefed-1)
      - [Other Notable Changes](#other-notable-changes-24)
    - [Garbage Collector](#garbage-collector)
    - [kubeadm](#kubeadm-4)
    - [kubectl](#kubectl-1)
      - [New Commands](#new-commands)
      - [Create subcommands](#create-subcommands)
      - [Updates to existing commands](#updates-to-existing-commands)
      - [Updates to apply](#updates-to-apply)
      - [Updates to edit](#updates-to-edit)
      - [Bug fixes](#bug-fixes)
      - [Other Notable Changes](#other-notable-changes-25)
    - [Node Components](#node-components-4)
      - [Bug fixes](#bug-fixes-1)
    - [kube-controller-manager](#kube-controller-manager)
    - [kube-dns](#kube-dns-1)
    - [kube-proxy](#kube-proxy-2)
    - [Scheduler](#scheduler)
    - [Volume Plugins](#volume-plugins-1)
      - [Azure Disk](#azure-disk)
      - [GlusterFS](#glusterfs)
      - [Photon](#photon)
      - [rbd](#rbd)
      - [vSphere](#vsphere-1)
      - [Other Notable Changes](#other-notable-changes-26)
  - [Changes to Cluster Provisioning Scripts](#changes-to-cluster-provisioning-scripts)
    - [AWS](#aws-1)
    - [Juju](#juju)
    - [libvirt CoreOS](#libvirt-coreos)
    - [GCE](#gce-1)
    - [OpenStack](#openstack)
    - [Container Images](#container-images)
    - [Other Notable Changes](#other-notable-changes-27)
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
  - [External Dependency Version Information](#external-dependency-version-information-1)
  - [Changelog since v1.6.0-rc.1](#changelog-since-v160-rc1)
    - [Previous Releases Included in v1.6.0](#previous-releases-included-in-v160)
- [v1.5.6](#v156)
  - [Downloads for v1.5.6](#downloads-for-v156)
    - [Client Binaries](#client-binaries-25)
    - [Server Binaries](#server-binaries-25)
  - [Changelog since v1.5.5](#changelog-since-v155)
    - [Other notable changes](#other-notable-changes-28)
- [v1.6.0-rc.1](#v160-rc1)
  - [Downloads for v1.6.0-rc.1](#downloads-for-v160-rc1)
    - [Client Binaries](#client-binaries-26)
    - [Server Binaries](#server-binaries-26)
  - [Changelog since v1.6.0-beta.4](#changelog-since-v160-beta4)
    - [Other notable changes](#other-notable-changes-29)
- [v1.5.5](#v155)
  - [Downloads for v1.5.5](#downloads-for-v155)
    - [Client Binaries](#client-binaries-27)
    - [Server Binaries](#server-binaries-27)
  - [Changelog since v1.5.4](#changelog-since-v154)
- [v1.6.0-beta.4](#v160-beta4)
  - [Downloads for v1.6.0-beta.4](#downloads-for-v160-beta4)
    - [Client Binaries](#client-binaries-28)
    - [Server Binaries](#server-binaries-28)
  - [Changelog since v1.6.0-beta.3](#changelog-since-v160-beta3)
    - [Other notable changes](#other-notable-changes-30)
- [v1.6.0-beta.3](#v160-beta3)
  - [Downloads for v1.6.0-beta.3](#downloads-for-v160-beta3)
    - [Client Binaries](#client-binaries-29)
    - [Server Binaries](#server-binaries-29)
  - [Changelog since v1.6.0-beta.2](#changelog-since-v160-beta2)
    - [Other notable changes](#other-notable-changes-31)
- [v1.6.0-beta.2](#v160-beta2)
  - [Downloads for v1.6.0-beta.2](#downloads-for-v160-beta2)
    - [Client Binaries](#client-binaries-30)
    - [Server Binaries](#server-binaries-30)
  - [Changelog since v1.6.0-beta.1](#changelog-since-v160-beta1)
    - [Action Required](#action-required-10)
    - [Other notable changes](#other-notable-changes-32)
- [v1.5.4](#v154)
  - [Downloads for v1.5.4](#downloads-for-v154)
    - [Client Binaries](#client-binaries-31)
    - [Server Binaries](#server-binaries-31)
  - [Changelog since v1.5.3](#changelog-since-v153)
    - [Other notable changes](#other-notable-changes-33)
- [v1.6.0-beta.1](#v160-beta1)
  - [Downloads for v1.6.0-beta.1](#downloads-for-v160-beta1)
    - [Client Binaries](#client-binaries-32)
    - [Server Binaries](#server-binaries-32)
  - [Changelog since v1.6.0-alpha.3](#changelog-since-v160-alpha3)
    - [Action Required](#action-required-11)
    - [Other notable changes](#other-notable-changes-34)
- [v1.6.0-alpha.3](#v160-alpha3)
  - [Downloads for v1.6.0-alpha.3](#downloads-for-v160-alpha3)
    - [Client Binaries](#client-binaries-33)
    - [Server Binaries](#server-binaries-33)
  - [Changelog since v1.6.0-alpha.2](#changelog-since-v160-alpha2)
    - [Other notable changes](#other-notable-changes-35)
- [v1.4.9](#v149)
  - [Downloads for v1.4.9](#downloads-for-v149)
    - [Client Binaries](#client-binaries-34)
    - [Server Binaries](#server-binaries-34)
  - [Changelog since v1.4.8](#changelog-since-v148)
    - [Other notable changes](#other-notable-changes-36)
- [v1.5.3](#v153)
  - [Downloads for v1.5.3](#downloads-for-v153)
    - [Client Binaries](#client-binaries-35)
    - [Server Binaries](#server-binaries-35)
    - [Node Binaries](#node-binaries-20)
  - [Changelog since v1.5.2](#changelog-since-v152)
    - [Other notable changes](#other-notable-changes-37)
- [v1.6.0-alpha.2](#v160-alpha2)
  - [Downloads for v1.6.0-alpha.2](#downloads-for-v160-alpha2)
    - [Client Binaries](#client-binaries-36)
    - [Server Binaries](#server-binaries-36)
  - [Changelog since v1.6.0-alpha.1](#changelog-since-v160-alpha1)
    - [Other notable changes](#other-notable-changes-38)
- [v1.6.0-alpha.1](#v160-alpha1)
  - [Downloads for v1.6.0-alpha.1](#downloads-for-v160-alpha1)
    - [Client Binaries](#client-binaries-37)
    - [Server Binaries](#server-binaries-37)
  - [Changelog since v1.5.0](#changelog-since-v150)
    - [Action Required](#action-required-12)
    - [Other notable changes](#other-notable-changes-39)
- [v1.5.2](#v152)
  - [Downloads for v1.5.2](#downloads-for-v152)
    - [Client Binaries](#client-binaries-38)
    - [Server Binaries](#server-binaries-38)
  - [Changelog since v1.5.1](#changelog-since-v151)
    - [Other notable changes](#other-notable-changes-40)
- [v1.4.8](#v148)
  - [Downloads for v1.4.8](#downloads-for-v148)
    - [Client Binaries](#client-binaries-39)
    - [Server Binaries](#server-binaries-39)
  - [Changelog since v1.4.7](#changelog-since-v147)
    - [Other notable changes](#other-notable-changes-41)
- [v1.5.1](#v151)
  - [Downloads for v1.5.1](#downloads-for-v151)
    - [Client Binaries](#client-binaries-40)
    - [Server Binaries](#server-binaries-40)
  - [Changelog since v1.5.0](#changelog-since-v150-1)
    - [Other notable changes](#other-notable-changes-42)
  - [Known Issues for v1.5.1](#known-issues-for-v151)
- [v1.5.0](#v150)
  - [Downloads for v1.5.0](#downloads-for-v150)
    - [Client Binaries](#client-binaries-41)
    - [Server Binaries](#server-binaries-41)
  - [Major Themes](#major-themes-1)
  - [Features](#features)
  - [Known Issues](#known-issues-1)
  - [Notable Changes to Existing Behavior](#notable-changes-to-existing-behavior)
  - [Deprecations](#deprecations-2)
  - [Action Required Before Upgrading](#action-required-before-upgrading-1)
  - [External Dependency Version Information](#external-dependency-version-information-2)
  - [Changelog since v1.5.0-beta.3](#changelog-since-v150-beta3)
    - [Other notable changes](#other-notable-changes-43)
    - [Previous Releases Included in v1.5.0](#previous-releases-included-in-v150)
- [v1.4.7](#v147)
  - [Downloads for v1.4.7](#downloads-for-v147)
    - [Client Binaries](#client-binaries-42)
    - [Server Binaries](#server-binaries-42)
  - [Changelog since v1.4.6](#changelog-since-v146)
    - [Other notable changes](#other-notable-changes-44)
- [v1.5.0-beta.3](#v150-beta3)
  - [Downloads for v1.5.0-beta.3](#downloads-for-v150-beta3)
    - [Client Binaries](#client-binaries-43)
    - [Server Binaries](#server-binaries-43)
  - [Changelog since v1.5.0-beta.2](#changelog-since-v150-beta2)
    - [Other notable changes](#other-notable-changes-45)
- [v1.5.0-beta.2](#v150-beta2)
  - [Downloads for v1.5.0-beta.2](#downloads-for-v150-beta2)
    - [Client Binaries](#client-binaries-44)
    - [Server Binaries](#server-binaries-44)
  - [Changelog since v1.5.0-beta.1](#changelog-since-v150-beta1)
    - [Other notable changes](#other-notable-changes-46)
- [v1.5.0-beta.1](#v150-beta1)
  - [Downloads for v1.5.0-beta.1](#downloads-for-v150-beta1)
    - [Client Binaries](#client-binaries-45)
    - [Server Binaries](#server-binaries-45)
  - [Changelog since v1.5.0-alpha.2](#changelog-since-v150-alpha2)
    - [Action Required](#action-required-13)
    - [Other notable changes](#other-notable-changes-47)
- [v1.4.6](#v146)
  - [Downloads for v1.4.6](#downloads-for-v146)
    - [Client Binaries](#client-binaries-46)
    - [Server Binaries](#server-binaries-46)
  - [Changelog since v1.4.5](#changelog-since-v145)
    - [Other notable changes](#other-notable-changes-48)
- [v1.3.10](#v1310)
  - [Downloads for v1.3.10](#downloads-for-v1310)
    - [Client Binaries](#client-binaries-47)
    - [Server Binaries](#server-binaries-47)
  - [Changelog since v1.3.9](#changelog-since-v139)
    - [Other notable changes](#other-notable-changes-49)
- [v1.4.5](#v145)
  - [Downloads for v1.4.5](#downloads-for-v145)
    - [Client Binaries](#client-binaries-48)
    - [Server Binaries](#server-binaries-48)
  - [Changelog since v1.4.4](#changelog-since-v144)
    - [Other notable changes](#other-notable-changes-50)
- [v1.5.0-alpha.2](#v150-alpha2)
  - [Downloads for v1.5.0-alpha.2](#downloads-for-v150-alpha2)
    - [Client Binaries](#client-binaries-49)
    - [Server Binaries](#server-binaries-49)
  - [Changelog since v1.5.0-alpha.1](#changelog-since-v150-alpha1)
    - [Action Required](#action-required-14)
    - [Other notable changes](#other-notable-changes-51)
- [v1.2.7](#v127)
  - [Downloads for v1.2.7](#downloads-for-v127)
    - [Client Binaries](#client-binaries-50)
    - [Server Binaries](#server-binaries-50)
  - [Changelog since v1.2.6](#changelog-since-v126)
    - [Other notable changes](#other-notable-changes-52)
- [v1.4.4](#v144)
  - [Downloads for v1.4.4](#downloads-for-v144)
    - [Client Binaries](#client-binaries-51)
    - [Server Binaries](#server-binaries-51)
  - [Changelog since v1.4.3](#changelog-since-v143)
    - [Other notable changes](#other-notable-changes-53)
- [v1.3.9](#v139)
  - [Downloads](#downloads)
  - [Changelog since v1.3.8](#changelog-since-v138)
    - [Other notable changes](#other-notable-changes-54)
- [v1.4.3](#v143)
  - [Downloads](#downloads-1)
  - [Changelog since v1.4.2-beta.1](#changelog-since-v142-beta1)
    - [Other notable changes](#other-notable-changes-55)
- [v1.4.2](#v142)
  - [Downloads](#downloads-2)
  - [Changelog since v1.4.2-beta.1](#changelog-since-v142-beta1-1)
    - [Other notable changes](#other-notable-changes-56)
- [v1.5.0-alpha.1](#v150-alpha1)
  - [Downloads](#downloads-3)
  - [Changelog since v1.4.0-alpha.3](#changelog-since-v140-alpha3)
    - [Experimental Features](#experimental-features)
    - [Action Required](#action-required-15)
    - [Other notable changes](#other-notable-changes-57)
- [v1.4.2-beta.1](#v142-beta1)
  - [Downloads](#downloads-4)
  - [Changelog since v1.4.1](#changelog-since-v141)
    - [Other notable changes](#other-notable-changes-58)
- [v1.4.1](#v141)
  - [Downloads](#downloads-5)
  - [Changelog since v1.4.1-beta.2](#changelog-since-v141-beta2)
- [v1.4.1-beta.2](#v141-beta2)
  - [Downloads](#downloads-6)
  - [Changelog since v1.4.0](#changelog-since-v140)
    - [Other notable changes](#other-notable-changes-59)
- [v1.3.8](#v138)
  - [Downloads](#downloads-7)
  - [Changelog since v1.3.7](#changelog-since-v137)
    - [Other notable changes](#other-notable-changes-60)
- [v1.4.0](#v140)
  - [Downloads](#downloads-8)
  - [Major Themes](#major-themes-2)
  - [Features](#features-1)
  - [Known Issues](#known-issues-2)
  - [Notable Changes to Existing Behavior](#notable-changes-to-existing-behavior-1)
    - [Deployments](#deployments-1)
    - [kubectl rolling-update: < v1.4.0 client vs >=v1.4.0 cluster](#kubectl-rolling-update:-<-v140-client-vs->=v140-cluster)
    - [kubectl delete: < v1.4.0 client vs >=v1.4.0 cluster](#kubectl-delete:-<-v140-client-vs->=v140-cluster)
    - [DELETE operation in REST API](#delete-operation-in-rest-api)
  - [Action Required Before Upgrading](#action-required-before-upgrading-2)
- [optionally, remove the old secret](#optionally-remove-the-old-secret)
  - [Previous Releases Included in v1.4.0](#previous-releases-included-in-v140)
- [v1.4.0-beta.11](#v140-beta11)
  - [Downloads](#downloads-9)
  - [Changelog since v1.4.0-beta.10](#changelog-since-v140-beta10)
- [v1.4.0-beta.10](#v140-beta10)
  - [Downloads](#downloads-10)
  - [Changelog since v1.4.0-beta.8](#changelog-since-v140-beta8)
    - [Other notable changes](#other-notable-changes-61)
- [v1.4.0-beta.8](#v140-beta8)
  - [Downloads](#downloads-11)
  - [Changelog since v1.4.0-beta.7](#changelog-since-v140-beta7)
- [v1.4.0-beta.7](#v140-beta7)
  - [Downloads](#downloads-12)
  - [Changelog since v1.4.0-beta.6](#changelog-since-v140-beta6)
    - [Other notable changes](#other-notable-changes-62)
- [v1.4.0-beta.6](#v140-beta6)
  - [Downloads](#downloads-13)
  - [Changelog since v1.4.0-beta.5](#changelog-since-v140-beta5)
    - [Other notable changes](#other-notable-changes-63)
- [v1.4.0-beta.5](#v140-beta5)
  - [Downloads](#downloads-14)
  - [Changelog since v1.4.0-beta.3](#changelog-since-v140-beta3)
    - [Other notable changes](#other-notable-changes-64)
- [v1.3.7](#v137)
  - [Downloads](#downloads-15)
  - [Changelog since v1.3.6](#changelog-since-v136)
    - [Other notable changes](#other-notable-changes-65)
- [v1.4.0-beta.3](#v140-beta3)
  - [Downloads](#downloads-16)
  - [Changelog since v1.4.0-beta.2](#changelog-since-v140-beta2)
  - [Behavior changes caused by enabling the garbage collector](#behavior-changes-caused-by-enabling-the-garbage-collector)
    - [kubectl rolling-update](#kubectl-rolling-update)
    - [kubectl delete](#kubectl-delete)
    - [DELETE operation in REST API](#delete-operation-in-rest-api-1)
- [v1.4.0-beta.2](#v140-beta2)
  - [Downloads](#downloads-17)
  - [Changelog since v1.4.0-beta.1](#changelog-since-v140-beta1)
    - [Other notable changes](#other-notable-changes-66)
- [v1.4.0-beta.1](#v140-beta1)
  - [Downloads](#downloads-18)
  - [Changelog since v1.4.0-alpha.3](#changelog-since-v140-alpha3-1)
    - [Action Required](#action-required-16)
    - [Other notable changes](#other-notable-changes-67)
- [v1.3.6](#v136)
  - [Downloads](#downloads-19)
  - [Changelog since v1.3.5](#changelog-since-v135)
    - [Other notable changes](#other-notable-changes-68)
- [v1.4.0-alpha.3](#v140-alpha3)
  - [Downloads](#downloads-20)
  - [Changelog since v1.4.0-alpha.2](#changelog-since-v140-alpha2)
    - [Action Required](#action-required-17)
    - [Other notable changes](#other-notable-changes-69)
- [v1.3.5](#v135)
  - [Downloads](#downloads-21)
  - [Changelog since v1.3.4](#changelog-since-v134)
    - [Other notable changes](#other-notable-changes-70)
- [v1.3.4](#v134)
  - [Downloads](#downloads-22)
  - [Changelog since v1.3.3](#changelog-since-v133)
    - [Other notable changes](#other-notable-changes-71)
- [v1.4.0-alpha.2](#v140-alpha2)
  - [Downloads](#downloads-23)
  - [Changelog since v1.4.0-alpha.1](#changelog-since-v140-alpha1)
    - [Action Required](#action-required-18)
    - [Other notable changes](#other-notable-changes-72)
- [v1.3.3](#v133)
  - [Downloads](#downloads-24)
  - [Changelog since v1.3.2](#changelog-since-v132)
    - [Other notable changes](#other-notable-changes-73)
  - [Known Issues](#known-issues-3)
- [v1.3.2](#v132)
  - [Downloads](#downloads-25)
  - [Changelog since v1.3.1](#changelog-since-v131)
    - [Other notable changes](#other-notable-changes-74)
- [v1.3.1](#v131)
  - [Downloads](#downloads-26)
  - [Changelog since v1.3.0](#changelog-since-v130)
    - [Other notable changes](#other-notable-changes-75)
- [v1.2.6](#v126)
  - [Downloads](#downloads-27)
  - [Changelog since v1.2.5](#changelog-since-v125)
    - [Other notable changes](#other-notable-changes-76)
- [v1.4.0-alpha.1](#v140-alpha1)
  - [Downloads](#downloads-28)
  - [Changelog since v1.3.0](#changelog-since-v130-1)
    - [Experimental Features](#experimental-features-1)
    - [Action Required](#action-required-19)
    - [Other notable changes](#other-notable-changes-77)
- [v1.3.0](#v130)
  - [Downloads](#downloads-29)
  - [Highlights](#highlights)
  - [Known Issues and Important Steps before Upgrading](#known-issues-and-important-steps-before-upgrading)
      - [ThirdPartyResource](#thirdpartyresource)
      - [kubectl](#kubectl-2)
      - [kubernetes Core Known Issues](#kubernetes-core-known-issues)
      - [Docker runtime Known Issues](#docker-runtime-known-issues)
      - [Rkt runtime Known Issues](#rkt-runtime-known-issues)
  - [Provider-specific Notes](#provider-specific-notes)
  - [Previous Releases Included in v1.3.0](#previous-releases-included-in-v130)
- [v1.3.0-beta.3](#v130-beta3)
  - [Downloads](#downloads-30)
  - [Changelog since v1.3.0-beta.2](#changelog-since-v130-beta2)
    - [Action Required](#action-required-20)
    - [Other notable changes](#other-notable-changes-78)
- [v1.2.5](#v125)
  - [Downloads](#downloads-31)
  - [Changes since v1.2.4](#changes-since-v124)
    - [Other notable changes](#other-notable-changes-79)
- [v1.3.0-beta.2](#v130-beta2)
  - [Downloads](#downloads-32)
  - [Changes since v1.3.0-beta.1](#changes-since-v130-beta1)
    - [Experimental Features](#experimental-features-2)
    - [Other notable changes](#other-notable-changes-80)
- [v1.3.0-beta.1](#v130-beta1)
  - [Downloads](#downloads-33)
  - [Changes since v1.3.0-alpha.5](#changes-since-v130-alpha5)
    - [Action Required](#action-required-21)
    - [Other notable changes](#other-notable-changes-81)
- [v1.3.0-alpha.5](#v130-alpha5)
  - [Downloads](#downloads-34)
  - [Changes since v1.3.0-alpha.4](#changes-since-v130-alpha4)
    - [Action Required](#action-required-22)
    - [Other notable changes](#other-notable-changes-82)
- [v1.3.0-alpha.4](#v130-alpha4)
  - [Downloads](#downloads-35)
  - [Changes since v1.3.0-alpha.3](#changes-since-v130-alpha3)
    - [Action Required](#action-required-23)
    - [Other notable changes](#other-notable-changes-83)
- [v1.2.4](#v124)
  - [Downloads](#downloads-36)
  - [Changes since v1.2.3](#changes-since-v123)
    - [Other notable changes](#other-notable-changes-84)
- [v1.3.0-alpha.3](#v130-alpha3)
  - [Downloads](#downloads-37)
  - [Changes since v1.3.0-alpha.2](#changes-since-v130-alpha2)
    - [Action Required](#action-required-24)
    - [Other notable changes](#other-notable-changes-85)
- [v1.2.3](#v123)
  - [Downloads](#downloads-38)
  - [Changes since v1.2.2](#changes-since-v122)
    - [Action Required](#action-required-25)
    - [Other notable changes](#other-notable-changes-86)
- [v1.3.0-alpha.2](#v130-alpha2)
  - [Downloads](#downloads-39)
  - [Changes since v1.3.0-alpha.1](#changes-since-v130-alpha1)
    - [Other notable changes](#other-notable-changes-87)
- [v1.2.2](#v122)
  - [Downloads](#downloads-40)
  - [Changes since v1.2.1](#changes-since-v121)
    - [Other notable changes](#other-notable-changes-88)
- [v1.2.1](#v121)
  - [Downloads](#downloads-41)
  - [Changes since v1.2.0](#changes-since-v120)
    - [Other notable changes](#other-notable-changes-89)
- [v1.3.0-alpha.1](#v130-alpha1)
  - [Downloads](#downloads-42)
  - [Changes since v1.2.0](#changes-since-v120-1)
    - [Action Required](#action-required-26)
    - [Other notable changes](#other-notable-changes-90)
- [v1.2.0](#v120)
  - [Downloads](#downloads-43)
  - [Changes since v1.1.1](#changes-since-v111)
    - [Major Themes](#major-themes-3)
    - [Other notable improvements](#other-notable-improvements)
    - [Experimental Features](#experimental-features-3)
    - [Action required](#action-required-27)
    - [Known Issues](#known-issues-4)
      - [Docker Known Issues](#docker-known-issues)
        - [1.9.1](#191)
    - [Provider-specific Notes](#provider-specific-notes-1)
      - [Various](#various)
      - [AWS](#aws-2)
      - [GCE](#gce-2)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


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
* PodDisruptionBudget now uses [ControllerRef](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/controller-ref.md) to make the right decisions about Pod eviction even if the built in application controllers have overlapping selectors.

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
* fix sync loop health check with seperating runtime errors ([#47124](https://github.com/kubernetes/kubernetes/pull/47124), [@andyxning](https://github.com/andyxning))
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
* Adds the `CustomResourceDefinition` (crd) types to the `kube-apiserver`.  These are the successors to `ThirdPartyResource`.  See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/thirdpartyresources.md for more details. ([#46055](https://github.com/kubernetes/kubernetes/pull/46055), [@deads2k](https://github.com/deads2k))
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
* fix sync loop health check with seperating runtime errors ([#47124](https://github.com/kubernetes/kubernetes/pull/47124), [@andyxning](https://github.com/andyxning))
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
* Adds the `CustomResourceDefinition` (crd) types to the `kube-apiserver`.  These are the successors to `ThirdPartyResource`.  See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/thirdpartyresources.md for more details. ([#46055](https://github.com/kubernetes/kubernetes/pull/46055), [@deads2k](https://github.com/deads2k))
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



# v1.5.7

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.7


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes.tar.gz) | `36bc0bcdce4060546f3fef7186f1207d30d5fd340e72113ff592966bd6684827`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-src.tar.gz) | `b329b02e9542049b9b85f8083a466e51799691bcf06fdf172b9c0f1cb61bdb6d`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-darwin-386.tar.gz) | `824ea7e5987e4ac7915b11fcd86658221a5a1e942a3f5210383435953509f96f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-darwin-amd64.tar.gz) | `251a91eff457640066dd395393b16aae81850225db29207c07321b62fd9213ab`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-linux-386.tar.gz) | `84c69d23010304308459ad520375fd017f57562f8a78b6157ef0ea093636a8b6`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-linux-amd64.tar.gz) | `991e1eab65d1817ca3600e3ba3bc63ed86cf139a4669f84899f593ff684fb36c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-linux-arm64.tar.gz) | `afe9c001a41b88da351ddf0cb3d506d3d8da7d9a94ae2d4b05062b2927c81fec`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-linux-arm.tar.gz) | `a936578c04887a2e1fe0a25e05f4d9663cd673d3fbac0c522bf75710d7f39f9b`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-windows-386.tar.gz) | `529ae014f0603868c82ee89601668fac17fa55932535d5925a7b61b1f301e61f`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-client-windows-amd64.tar.gz) | `f1f7e588dca059a4cbe97b4a28a983d346f93fc2bb0d4a1dbbb7d55a3e33caef`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-server-linux-amd64.tar.gz) | `ae18d659811da316d4a8bbdce15c4396fdee0068f9d3247a72c3a23433fee44c`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-server-linux-arm64.tar.gz) | `d56187d19b42848b7ff09e82c0452120c173ae56709cae88f96312ee7c41b0c4`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-server-linux-arm.tar.gz) | `aaa4d9414620bb1834401a17f2b877fe1347a4f8fc37c940092ac7f112e22934`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-node-linux-amd64.tar.gz) | `40c294ef5af4d548d37a599ee7fa07462f116fa5784d2b1501d95eeb04b8d34d`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-node-linux-arm64.tar.gz) | `37482d5933c99fca526d0d47f0cfb2b69101f2e60dd5240b490b9768c8e4171e`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-node-linux-arm.tar.gz) | `786ddb390a9fac6e41caa4bb8054240ddb5255b9937bb37d01d77e23857bb407`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.7/kubernetes-node-windows-amd64.tar.gz) | `c3e89390c8026845fcf72765e84b7e3cd383de705ef46f4e3d422b343d66bd47`

## Changelog since v1.5.6

### Other notable changes

* kube-apiserver now drops unneeded path information if an older version of Windows kubectl sends it. ([#44585](https://github.com/kubernetes/kubernetes/pull/44585), [@mml](https://github.com/mml))
* Fix for kube-proxy healthcheck handling an update that simultaneously removes one port and adds another. ([#44365](https://github.com/kubernetes/kubernetes/pull/44365), [@MrHohn](https://github.com/MrHohn))
* Fix [broken service accounts when using dedicated service account key](https://github.com/kubernetes/kubernetes/issues/44285). ([#44169](https://github.com/kubernetes/kubernetes/pull/44169), [@mikedanese](https://github.com/mikedanese))
* Update to fluentd-gcp:1.28.3, rebased on ubuntu-slim:0.8 ([#43928](https://github.com/kubernetes/kubernetes/pull/43928), [@ixdy](https://github.com/ixdy))
* Fix polarity of NodePort logic to avoid leaked ports ([#43259](https://github.com/kubernetes/kubernetes/pull/43259), [@thockin](https://github.com/thockin))
* Upgrade golang versions to 1.7.5 ([#41771](https://github.com/kubernetes/kubernetes/pull/41771), [@cblecker](https://github.com/cblecker))



# v1.4.12

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.4/examples)

## Downloads for v1.4.12


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes.tar.gz) | `f0d7ca7e1c92174c900d49087347d043b817eb589803eacc7727a84df9280ed2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-src.tar.gz) | `251835f258d79f186d8c715b18f2ccb93312270b35c22434b4ff27bc1de50eda`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-darwin-386.tar.gz) | `e91c76b6281fe7b488f2f30aeaeecde58a6df1a0e23f6c431b6dc9d1adc1ff1a`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-darwin-amd64.tar.gz) | `4504bc965bd1b5bcea91d18c3a879252026796fdd251b72e3541499c65ac20e0`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-linux-386.tar.gz) | `adf1f939db2da0b87bca876d9bee69e0d6bf4ca4a78e64195e9a08960e5ef010`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-linux-amd64.tar.gz) | `5419bdbba8144b55bf7bf2af1aefa531e25279f31a02d692f19b505862d0204f`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-linux-arm64.tar.gz) | `98ae30ac2e447b9e3c2768cac6861de5368d80cbd2db1983697c5436a2a2fe75`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-linux-arm.tar.gz) | `ed8e9901c130aebfd295a6016cccb123ee42d826619815250a6add2d03942c69`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-windows-386.tar.gz) | `bdca3096bed1a4c485942ab1d3f9351f5de00962058adefbb5297d50071461d4`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-client-windows-amd64.tar.gz) | `a74934eca20dd2e753d385ddca912e76dafbfff2a65e3e3a1ec3c5c40fd92bc8`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-server-linux-amd64.tar.gz) | `bf8aa3e2e204c1f782645f7df9338767daab7be3ab47a4670e2df08ee410ee7f`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-server-linux-arm64.tar.gz) | `7c5cfe06fe1fcfe11bd754921e88582d16887aacb6cee0eb82573c88debce65e`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-server-linux-arm.tar.gz) | `551c2bc2e3d1c0b8fa30cc0b0c8fae1acf561b5e303e9ddaf647e49239a97e6e`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node*.tar.gz](https://dl.k8s.io/v1.4.12/kubernetes-node*.tar.gz) | ``

## Changelog since v1.4.9

### Other notable changes

* kube-apiserver now drops unneeded path information if an older version of Windows kubectl sends it. ([#44586](https://github.com/kubernetes/kubernetes/pull/44586), [@mml](https://github.com/mml))
* Bump gcr.io/google_containers/glbc from 0.8.0 to 0.9.2. Release notes: [0.9.0](https://github.com/kubernetes/ingress/releases/tag/0.9.0), [0.9.1](https://github.com/kubernetes/ingress/releases/tag/0.9.1), [0.9.2](https://github.com/kubernetes/ingress/releases/tag/0.9.2) ([#43098](https://github.com/kubernetes/kubernetes/pull/43098), [@timstclair](https://github.com/timstclair))
* Patch CVE-2016-8859 in alpine based images: ([#42937](https://github.com/kubernetes/kubernetes/pull/42937), [@timstclair](https://github.com/timstclair))
    * - gcr.io/google-containers/etcd-empty-dir-cleanup
    * - gcr.io/google-containers/kube-dnsmasq-amd64
* Check if pathExists before performing Unmount ([#39311](https://github.com/kubernetes/kubernetes/pull/39311), [@rkouj](https://github.com/rkouj))
* Unmount operation should not fail if volume is already unmounted ([#38547](https://github.com/kubernetes/kubernetes/pull/38547), [@rkouj](https://github.com/rkouj))
* Updates base image used for `kube-addon-manager` to latest `python:2.7-slim` and embedded `kubectl` to `v1.3.10`. No functionality changes expected. ([#42842](https://github.com/kubernetes/kubernetes/pull/42842), [@ixdy](https://github.com/ixdy))
* list-resources: don't fail if the grep fails to match any resources ([#41933](https://github.com/kubernetes/kubernetes/pull/41933), [@ixdy](https://github.com/ixdy))
* Update gcr.io/google-containers/rescheduler to v0.2.2, which uses busybox as a base image instead of ubuntu. ([#41911](https://github.com/kubernetes/kubernetes/pull/41911), [@ixdy](https://github.com/ixdy))
* Backporting TPR fix to 1.4 ([#42380](https://github.com/kubernetes/kubernetes/pull/42380), [@foxish](https://github.com/foxish))
* Fix AWS device allocator to only use valid device names ([#41455](https://github.com/kubernetes/kubernetes/pull/41455), [@gnufied](https://github.com/gnufied))
* Reverts to looking up the current VM in vSphere using the machine's UUID, either obtained via sysfs or via the `vm-uuid` parameter in the cloud configuration file. ([#40892](https://github.com/kubernetes/kubernetes/pull/40892), [@robdaemon](https://github.com/robdaemon))
* We change the default attach_detach_controller sync period to 1 minute to reduce the query frequency through cloud provider to check whether volumes are attached or not.  ([#41363](https://github.com/kubernetes/kubernetes/pull/41363), [@jingxu97](https://github.com/jingxu97))
* Bump GCI to gci-stable-56-9000-84-2: Fixed google-accounts-daemon breaks on GCI when network is unavailable. Fixed iptables-restore performance regression. ([#41831](https://github.com/kubernetes/kubernetes/pull/41831), [@freehan](https://github.com/freehan))
* Update fluentd-gcp addon to 1.25.2 ([#41863](https://github.com/kubernetes/kubernetes/pull/41863), [@ixdy](https://github.com/ixdy))
* Bump GCE ContainerVM to container-vm-v20170214 to address CVE-2016-9962. ([#41449](https://github.com/kubernetes/kubernetes/pull/41449), [@zmerlynn](https://github.com/zmerlynn))



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
* Scheduler can recieve its policy configuration from a ConfigMap ([#43892](https://github.com/kubernetes/kubernetes/pull/43892), [@bsalamat](https://github.com/bsalamat))
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
* release-note-none ([#41139](https://github.com/kubernetes/kubernetes/pull/41139), [@juanvallejo](https://github.com/juanvallejo))
* fc: Drop multipath.conf snippet ([#36698](https://github.com/kubernetes/kubernetes/pull/36698), [@fabiand](https://github.com/fabiand))



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
* Deployment now fully respects ControllerRef to avoid fighting over Pods and ReplicaSets. At the time of upgrade, **you must not have Deployments with selectors that overlap**, or else [ownership of ReplicaSets may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/controller-ref.md#upgrading). ([#42175](https://github.com/kubernetes/kubernetes/pull/42175), [@enisoc](https://github.com/enisoc))

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
    https://kubernetes.io/docs/concepts/cluster-administration/node-allocatable/
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
* StatefulSet now respects ControllerRef to avoid fighting over Pods. At the time of upgrade, **you must not have StatefulSets with selectors that overlap** with any other controllers (such as ReplicaSets), or else [ownership of Pods may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/controller-ref.md#upgrading). ([#42080](https://github.com/kubernetes/kubernetes/pull/42080), [@enisoc](https://github.com/enisoc))

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
* **[stable]** The internal storage layer for kubernetes cluster state has been updated to use etcd v3 by default. Existing clusters will have to plan for a data migration window. ([docs](https://github.com/kubernetes/kubernetes.github.io/pull/2763))([kubernetes/features#44](https://github.com/kubernetes/features/issues/44))

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
* **[alpha]** Flex volume API and Improved lifecycle (flexvolume).

## Deprecations
* Remove extensions/v1beta1 Jobs resource, and job/v1beta1 generator. ([#38614](https://github.com/kubernetes/kubernetes/pull/38614), [@soltysh](https://github.com/soltysh))
* `federation/deploy/deploy.sh` was an interim solution introduced in Kubernetes v1.4 to simplify the federation control plane deployment experience. Now that we have `kubefed`, we are deprecating `deploy.sh` scripts. ([#38902](https://github.com/kubernetes/kubernetes/pull/38902), [@madhusudancs](https://github.com/madhusudancs))


### Cluster Provisioning Scripts
* The bash AWS deployment via kube-up.sh has been deprecated. See http://kubernetes.io/docs/getting-started-guides/aws/ for alternatives. ([#38772](https://github.com/kubernetes/kubernetes/pull/38772), [@zmerlynn](https://github.com/zmerlynn))
* Remove Azure kube-up as the Azure community has focused efforts elsewhere. ([#41672](https://github.com/kubernetes/kubernetes/pull/41672), [@mikedanese](https://github.com/mikedanese))
* Remove the deprecated vsphere kube-up. ([#39140](https://github.com/kubernetes/kubernetes/pull/39140), [@kerneltime](https://github.com/kerneltime))

### kubeadm
* Quite a few flags been renamed or removed.  Those options that are removed as flags can still be accessed via the config file.  Most noteably this includes external etcd settings and the option for setting the cloud provider on the API server.  The [kubeadm reference documentation](https://kubernetes.io/docs/admin/kubeadm/) is up to date with the new flags.

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



# v1.5.6

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.6


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes.tar.gz) | `14a514bb9ed331eb1854a1d66cfaa53290c382641e154c901bcb14eb2cd683b4`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-src.tar.gz) | `cf3d85bcfd148ed6a54c64b4102a10cc4e54332907fb3d9a6c6e2658a31ca2e9`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-darwin-386.tar.gz) | `30b819eb1427317be38a4f534fc2c369d43e67499e5df79cdd5d4cfac14f8d36`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-darwin-amd64.tar.gz) | `3eedf919b2feff4c21edcadb493247013274a3672f6a3d46f19e13af211cea4e`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-linux-386.tar.gz) | `351bb189f6be835baadda3b87909472c4a9f522ece6e6425250ef227937f2d58`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-linux-amd64.tar.gz) | `d7c3508dc5029c6fefb1bf6f381af92d8626ac5a4b7246009832c03768ae670f`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-linux-arm64.tar.gz) | `2eaf838ab853c94f05c362a8ce089f32acdb6062356399a6f5fe7cdb13a6fa0c`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-linux-arm.tar.gz) | `e5212f6d9577bd090c88a7124edba86f925e08c710865623d9fb914a5b72e67f`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-windows-386.tar.gz) | `5a4fdbf0cb88f0e889d8dca1e6c073c167a8c3c7d7b1caad10dbe0dc2eb46677`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-client-windows-amd64.tar.gz) | `b1170a33c5c6fe2c3f71e820f11cf877f0ee72b60a6546aaf989267c89598656`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-server-linux-amd64.tar.gz) | `995959c43661c22b0f2ede45b62061f37c25a53388bcdd8988f928574070390a`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-server-linux-arm64.tar.gz) | `c6b20af2f0c5e3abe20c18aac734846923c8ff3afda637ef1fbd6d3b3820e3b7`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.6/kubernetes-server-linux-arm.tar.gz) | `7d439b2012a0280d40441a5871b25a07b540f5e84c561d2bf8c67725ebbf115d`

## Changelog since v1.5.5

### Other notable changes

* kube-up (with gce/gci and gce/coreos providers) now ensures the authentication token file contains correct tokens for the control plane components, even if the file already exists (ensures upgrades and downgrades work successfully) ([#43676](https://github.com/kubernetes/kubernetes/pull/43676), [@liggitt](https://github.com/liggitt))
* Patch CVE-2016-8859 in alpine based images: ([#42936](https://github.com/kubernetes/kubernetes/pull/42936), [@timstclair](https://github.com/timstclair))
    * - gcr.io/google-containers/cluster-proportional-autoscaler-amd64
    * - gcr.io/google-containers/dnsmasq-metrics-amd64
    * - gcr.io/google-containers/etcd-empty-dir-cleanup
    * - gcr.io/google-containers/kube-addon-manager
    * - gcr.io/google-containers/kube-dnsmasq-amd64
* - Disable thin_ls due to excessive iops ([#43113](https://github.com/kubernetes/kubernetes/pull/43113), [@dashpole](https://github.com/dashpole))
    * - Ignore .mount cgroups, fixing dissappearing stats
    * - Fix wc goroutine leak
    * - Update aws-sdk-go dependency to 1.6.10
* PodSecurityPolicy authorization is correctly enforced by the PodSecurityPolicy admission plugin. ([#43489](https://github.com/kubernetes/kubernetes/pull/43489), [@liggitt](https://github.com/liggitt))
* Bump gcr.io/google_containers/glbc from 0.9.1 to 0.9.2. Release notes: [0.9.2](https://github.com/kubernetes/ingress/releases/tag/0.9.2) ([#43097](https://github.com/kubernetes/kubernetes/pull/43097), [@timstclair](https://github.com/timstclair))
* Update gcr.io/google-containers/rescheduler to v0.2.2, which uses busybox as a base image instead of ubuntu. ([#41911](https://github.com/kubernetes/kubernetes/pull/41911), [@ixdy](https://github.com/ixdy))
* restored normalization of custom `--etcd-prefix` when `--storage-backend` is set to etcd3 ([#42506](https://github.com/kubernetes/kubernetes/pull/42506), [@liggitt](https://github.com/liggitt))



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



# v1.5.5

This release contains a fix for a PodSecurityPolicy vulnerability which allows users to make use of any existing PodSecurityPolicy object, even ones they are not authorized to use.

Other then that, this release contains no other changes from 1.5.4.

The vulnerability is tracked in http://issue.k8s.io/43459.

**Who is affected?**

Only Kubernetes 1.5.0-1.5.4 installations that do all of the following:
* Enable the PodSecurityPolicy API (which is not enabled by default):
  * `--runtime-config=extensions/v1beta1/podsecuritypolicy=true`
* Enable the PodSecurityPolicy admission plugin (which is not enabled by default):
  * `--admission-control=...,PodSecurityPolicy,...`
* Use authorization to limit users' ability to use specific PodSecurityPolicy objects

**What is the impact?**

A user that is authorized to create pods can make use of any existing PodSecurityPolicy, even ones they are not authorized to use.

**How can I mitigate this prior to installing 1.5.5?**

1. Export existing PodSecurityPolicy objects:
  * `kubectl get podsecuritypolicies -o yaml > psp.yaml`
2. Review and delete any PodSecurityPolicy objects you do not want all pod-creating users to be able to use (NOTE: Privileged users that were making use of those policies will also lose access to those policies). For example:
  * `kubectl delete podsecuritypolicies/my-privileged-policy`
3. After upgrading to 1.5.5, re-create the exported PodSecurityPolicy objects:
  * `kubectl create -f psp.yaml`

## Downloads for v1.5.5


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes.tar.gz) | `ff171d53b6dba2aace899dbfa06044d3a54d798896f7b6dd483f20d2c05374ed`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-src.tar.gz) | `25207344982bcf76172c7d156106357a7113b3909ac851e19b437dbba9402af6`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-darwin-386.tar.gz) | `92eb19b1464674078927263642205498a9b4e496909138626de721f8ff2eb3f1`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-darwin-amd64.tar.gz) | `dd2076d8a3062459b82481bf064d80a198df580f2c34efe7132a091c19d8084c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-linux-386.tar.gz) | `8366a72910c987e4140db42244741752efac8e06f0e13f5d0cbc1cc9bec9733c`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-linux-amd64.tar.gz) | `73536e200fee9f4de19ebfd7d2e063a04f5ccb93073982032e79dc47ae92e89a`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-linux-arm64.tar.gz) | `8f679bd012ecbc58f0a916f393d3fc79de6dc2624320b04edc1b9249213a49f8`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-linux-arm.tar.gz) | `1998d6398aef02898babc5ff20484fe7c538f75f78c650631afea1a555aee8d1`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-windows-386.tar.gz) | `dff6fe02a6090feb949acc5753633891bcbdb7ecfb2bff3fa132d025713cbd55`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-client-windows-amd64.tar.gz) | `bd7c7c39122135b58da89a700580475a3cadbb31aa1b35175ff2f80067bedc0d`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-server-linux-amd64.tar.gz) | `578977b62af58639548d743991cd2f71b0fd58f9caa729131824f8dde85b5c6e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-server-linux-arm64.tar.gz) | `01a1104d8c5a22c26b8b0a402bf0362d749b7d13a636b31c64fb51bb61ea3a01`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.5/kubernetes-server-linux-arm.tar.gz) | `06c5ca1f962f368219835ed6d075ef6e3a72685f2f0988823f44dd2e602e1980`

## Changelog since v1.5.4

**No notable changes for this release**



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

* Deployment now fully respects ControllerRef to avoid fighting over Pods and ReplicaSets. At the time of upgrade, **you must not have Deployments with selectors that overlap**, or else [ownership of ReplicaSets may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/controller-ref.md#upgrading). ([#42175](https://github.com/kubernetes/kubernetes/pull/42175), [@enisoc](https://github.com/enisoc))
* StatefulSet now respects ControllerRef to avoid fighting over Pods. At the time of upgrade, **you must not have StatefulSets with selectors that overlap** with any other controllers (such as ReplicaSets), or else [ownership of Pods may change](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/controller-ref.md#upgrading). ([#42080](https://github.com/kubernetes/kubernetes/pull/42080), [@enisoc](https://github.com/enisoc))

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



# v1.5.4

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.4


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes.tar.gz) | `2ff668c687c1bdf8351dcae102901b1d46cc50e446bde08a244c2e65739de4c3`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-src.tar.gz) | `172d33787ec2d11345d152becdc96982d3057ed16426910302c1b103980b634b`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-darwin-386.tar.gz) | `53e7c4839025ad04c1104b99e1f8b45f4fe639397c623e2e050acb53cb0a8cbd`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-darwin-amd64.tar.gz) | `6fac39282c9599566874d63c57b305798e4096a42ef83a8965f615c1d709559c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-linux-386.tar.gz) | `80719626f7e6db6d2d04e57bb7edad3077b774a11ebccea3fcddadaa48cbf0a6`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-linux-amd64.tar.gz) | `24001bc0c7ddb32cd72ac9bed55543830424fba734587ac23b812d8d047a9091`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-linux-arm64.tar.gz) | `094ff4fe7a10e23a397803869a11a3cc508f3990d9e3b4fbccaefe44be2ad81a`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-linux-arm.tar.gz) | `b12b823d12942d7fccaf791343e9c9854073de3e03cc57a7e4bd7b03fec9806b`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-windows-386.tar.gz) | `e5ae9775cfe695d2d855b29c01f19b0fd0fad008071d8e95f47f70beb16291a8`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-client-windows-amd64.tar.gz) | `40cc26a8216e703217264194b68d6b5af28ffa1b9b48b23232027c5d63d8b28c`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-server-linux-amd64.tar.gz) | `a61cb36d64c8a4111cf04f9d1aac5d8418d07a7c8a682522203b0dfa76f9c806`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-server-linux-arm64.tar.gz) | `abaa5052f9d0daaebf6b7375c9667c9160355b8ea074daac76ba8a79a24cab37`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.4/kubernetes-server-linux-arm.tar.gz) | `ffff55a0f5f5848fdde32a2766dc63cdf26629ca4f91db458381ffb55cf49535`

## Changelog since v1.5.3

### Other notable changes

* Fix AWS device allocator to only use valid device names ([#41455](https://github.com/kubernetes/kubernetes/pull/41455), [@gnufied](https://github.com/gnufied))
* The kube-apiserver [basic audit log](https://kubernetes.io/docs/admin/audit/) can be enabled in GCE by exporting the environment variable `ENABLE_APISERVER_BASIC_AUDIT=true` before running `cluster/kube-up.sh`. This will log to `/var/log/kube-apiserver-audit.log` and use the same `logrotate` settings as `/var/log/kube-apiserver.log`. ([#41211](https://github.com/kubernetes/kubernetes/pull/41211), [@enisoc](https://github.com/enisoc))
* list-resources: don't fail if the grep fails to match any resources ([#41933](https://github.com/kubernetes/kubernetes/pull/41933), [@ixdy](https://github.com/ixdy))
* Bump GCE ContainerVM to container-vm-v20170214 to address CVE-2016-9962. ([#41449](https://github.com/kubernetes/kubernetes/pull/41449), [@zmerlynn](https://github.com/zmerlynn))



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



# v1.4.9

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.4/examples)

## Downloads for v1.4.9


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes.tar.gz) | `9d385d555073c7cf509a92ce3aa96d0414a93c21c51bcf020744c70b4b290aa2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-src.tar.gz) | `6fd7d33775356f0245d06b401ac74d8227a92abd07cc5a0ef362bac16e01f011`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-darwin-386.tar.gz) | `16b362f3cf56dee7b0c291188767222fd65176ed9573a8b87e8acf7eb6b22ed9`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-darwin-amd64.tar.gz) | `537e5c5d8a9148cd464f5d6d0a796e214add04c185b859ea9e39a4cc7264394c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-linux-386.tar.gz) | `e9d2e55b42e002771c32d9f26e8eb0b65c257ea257e8ab19f7fd928f21caace8`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-linux-amd64.tar.gz) | `1ba81d64d1ae165b73375d61d364c642068385d6a1d68196d90e42a8d0fd6c7d`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-linux-arm64.tar.gz) | `d0398d2b11ed591575adde3ce9e1ad877fe37b8b56bd2be5b2aee344a35db330`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-linux-arm.tar.gz) | `714b06319bf047084514803531edab6a0a262c5f38a0d0bfda0a8e59672595b6`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-windows-386.tar.gz) | `16a7224313889d2f98a7d072f328198790531fd0e724eaeeccffe82521ae63b8`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-client-windows-amd64.tar.gz) | `dc19651287701ea6dcbd7b4949db2331468f730e8ebe951de1216f1105761d97`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-server-linux-amd64.tar.gz) | `6a104d143f8568a8ce16c979d1cb2eb357263d96ab43bd399b05d28f8da2b961`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-server-linux-arm64.tar.gz) | `8137ecde19574e6aba0cd9efe127f3b3eb02c312d7691745df3a23e40b7a5d72`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.4.9/kubernetes-server-linux-arm.tar.gz) | `085195abeb9133cb43f0e6198e638ded7f15beca44d19503c2836339a7e604aa`

## Changelog since v1.4.8

### Other notable changes

* Bump GCE ContainerVM to container-vm-v20170201 to address CVE-2016-9962. ([#40828](https://github.com/kubernetes/kubernetes/pull/40828), [@zmerlynn](https://github.com/zmerlynn))
* Bump GCI to gci-beta-56-9000-80-0 ([#41027](https://github.com/kubernetes/kubernetes/pull/41027), [@dchen1107](https://github.com/dchen1107))
* Fix for detach volume when node is not present/ powered off ([#40118](https://github.com/kubernetes/kubernetes/pull/40118), [@BaluDontu](https://github.com/BaluDontu))
* Bump GCI to gci-beta-56-9000-80-0 ([#41027](https://github.com/kubernetes/kubernetes/pull/41027), [@dchen1107](https://github.com/dchen1107))
* Move b.gcr.io/k8s_authenticated_test to gcr.io/k8s-authenticated-test ([#40335](https://github.com/kubernetes/kubernetes/pull/40335), [@zmerlynn](https://github.com/zmerlynn))
* Prep node_e2e for GCI to COS name change ([#41088](https://github.com/kubernetes/kubernetes/pull/41088), [@jessfraz](https://github.com/jessfraz))
* If ExperimentalCriticalPodAnnotation=True flag gate is set, kubelet will ensure that pods with `scheduler.alpha.kubernetes.io/critical-pod` annotation will be admitted even under resource pressure, will not be evicted, and are reasonably protected from system OOMs. ([#41052](https://github.com/kubernetes/kubernetes/pull/41052), [@vishh](https://github.com/vishh))
* Fix resync goroutine leak in ListAndWatch ([#35672](https://github.com/kubernetes/kubernetes/pull/35672), [@tatsuhiro-t](https://github.com/tatsuhiro-t))
* Kubelet will no longer set hairpin mode on every interface on the machine when an error occurs in setting up hairpin for a specific interface. ([#36990](https://github.com/kubernetes/kubernetes/pull/36990), [@bboreham](https://github.com/bboreham))
* Bump GCE ContainerVM to container-vm-v20170201 to address CVE-2016-9962. ([#40828](https://github.com/kubernetes/kubernetes/pull/40828), [@zmerlynn](https://github.com/zmerlynn))
* Adding vmdk file extension for vmDiskPath in vsphere DeleteVolume ([#40538](https://github.com/kubernetes/kubernetes/pull/40538), [@divyenpatel](https://github.com/divyenpatel))
* Prevent hotloops on error conditions, which could fill up the disk faster than log rotation can free space. ([#40497](https://github.com/kubernetes/kubernetes/pull/40497), [@lavalamp](https://github.com/lavalamp))
* Update GCE ContainerVM deployment to container-vm-v20170117 to pick up CVE fixes in base image. ([#40094](https://github.com/kubernetes/kubernetes/pull/40094), [@zmerlynn](https://github.com/zmerlynn))
* Update kube-proxy image to be based off of Debian 8.6 base image. ([#39695](https://github.com/kubernetes/kubernetes/pull/39695), [@ixdy](https://github.com/ixdy))
* Update amd64 kube-proxy base image to debian-iptables-amd64:v5 ([#39725](https://github.com/kubernetes/kubernetes/pull/39725), [@ixdy](https://github.com/ixdy))



# v1.5.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes.tar.gz) | `a4d997be9e3ac0f9838a58fb80d08c2ab02e00afb9d16d3db18d99c85b88b316`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-src.tar.gz) | `a23636ee40a60c1bb3255a03177f522c28133f74c6d09a5437f6b56b7e1d5296`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-darwin-386.tar.gz) | `2f8eeb772c22c7dad5a32d6ee17e8b309503b56fbcb0abdc74e1f94e86b33520`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-darwin-amd64.tar.gz) | `b044240271223aa93f8bdb8054824a48ba5571460d2e6c90688dccd0892e5c7e`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-linux-386.tar.gz) | `d2649a41e4a64c2027e321254e4ef3e690371bd0c7eece12d3395e49d8171617`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-linux-amd64.tar.gz) | `eaf386a46eeee324bb71349bba7d5d3f41d7d19af75537cf9e4e7045d7068f68`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-linux-arm64.tar.gz) | `2f2d45296651e5696f373838ba019e8b8bb11b2a2772a55f0a6e367ec6c18e2d`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-linux-arm.tar.gz) | `56b8b207fd914dc7c16fdb675a3917ab9bff0efbe745ee1675abbff2b5854d32`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-windows-386.tar.gz) | `fe3136e3c6bd983e55396341c451f896e478e8c9d0b3d1418e1d1fccee3d7b75`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-client-windows-amd64.tar.gz) | `8e315cb48135a4ed26585e9d8cf88f550ac51e3658b981bb53cb0952e9b3393a`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-server-linux-amd64.tar.gz) | `ad4d101bec0ef981a7e1efbe11223e502ff644368d70ad54915e15fcb3ad6735`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-server-linux-arm64.tar.gz) | `bfd66c57d1071bdd213d4c6d124d491959ae3509994e5a23cc2720a8ad18526d`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-server-linux-arm.tar.gz) | `12b335637b7a4aa019cee600b0161d51e6317a87bec0500e1f9d85990f6352d5`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-node-linux-amd64.tar.gz) | `3f54e2d101b6351513ce9425a23f9a196e965326c3a7f78a98ef1dad452e5830`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-node-linux-arm.tar.gz) | `6508b64755dc0ff90f23921d2b8bb6c0c321c38edeaf24fd4c22282880a87a11`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-node-linux-arm64.tar.gz) | `578ef8a6958fb4bf2e0438cdef7707d12456186a1b8c4b18aa66f47b9221a713`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.3/kubernetes-node-windows-amd64.tar.gz) | `aa166b275b3d0f80cbf23fbee7f42358b6176f37fd9ef66837f38910d4626079`

## Changelog since v1.5.2

### Other notable changes

* We change the default attach_detach_controller sync period to 1 minute to reduce the query frequency through cloud provider to check whether volumes are attached or not.  ([#41363](https://github.com/kubernetes/kubernetes/pull/41363), [@jingxu97](https://github.com/jingxu97))
* Added configurable etcd initial-cluster-state to kube-up script ([#41320](https://github.com/kubernetes/kubernetes/pull/41320), [@jszczepkowski](https://github.com/jszczepkowski))
* If ExperimentalCriticalPodAnnotation=True flag gate is set, kubelet will ensure that pods with `scheduler.alpha.kubernetes.io/critical-pod` annotation will be admitted even under resource pressure, will not be evicted, and are reasonably protected from system OOMs. ([#41052](https://github.com/kubernetes/kubernetes/pull/41052), [@vishh](https://github.com/vishh))
* Reverts to looking up the current VM in vSphere using the machine's UUID, either obtained via sysfs or via the `vm-uuid` parameter in the cloud configuration file. ([#40892](https://github.com/kubernetes/kubernetes/pull/40892), [@robdaemon](https://github.com/robdaemon))
* Fix for detach volume when node is not present/ powered off ([#40118](https://github.com/kubernetes/kubernetes/pull/40118), [@BaluDontu](https://github.com/BaluDontu))
* Move b.gcr.io/k8s_authenticated_test to gcr.io/k8s-authenticated-test ([#40335](https://github.com/kubernetes/kubernetes/pull/40335), [@zmerlynn](https://github.com/zmerlynn))
* Bump up GLBC version from 0.9.0-beta to 0.9.1 ([#41037](https://github.com/kubernetes/kubernetes/pull/41037), [@bprashanth](https://github.com/bprashanth))
* azure: fix Azure Container Registry integration ([#40142](https://github.com/kubernetes/kubernetes/pull/40142), [@colemickens](https://github.com/colemickens))
* azure disk: restrict name length for Azure specifications ([#40030](https://github.com/kubernetes/kubernetes/pull/40030), [@colemickens](https://github.com/colemickens))
* Bump GCI to gci-beta-56-9000-80-0 ([#41027](https://github.com/kubernetes/kubernetes/pull/41027), [@dchen1107](https://github.com/dchen1107))
* Bump up glbc version to 0.9.0-beta.1 ([#40565](https://github.com/kubernetes/kubernetes/pull/40565), [@bprashanth](https://github.com/bprashanth))
* Enable lazy inode table and journal initialization for ext3 and ext4 ([#38865](https://github.com/kubernetes/kubernetes/pull/38865), [@codablock](https://github.com/codablock))
* Kubelet will no longer set hairpin mode on every interface on the machine when an error occurs in setting up hairpin for a specific interface. ([#36990](https://github.com/kubernetes/kubernetes/pull/36990), [@bboreham](https://github.com/bboreham))
* The SubjectAccessReview API passes subresource and resource name information to the authorizer to answer authorization queries. ([#40935](https://github.com/kubernetes/kubernetes/pull/40935), [@liggitt](https://github.com/liggitt))
* Bump GCE ContainerVM to container-vm-v20170201 to address CVE-2016-9962. ([#40828](https://github.com/kubernetes/kubernetes/pull/40828), [@zmerlynn](https://github.com/zmerlynn))
* Reduce time needed to attach Azure disks ([#40066](https://github.com/kubernetes/kubernetes/pull/40066), [@codablock](https://github.com/codablock))
* Fixes request header authenticator by presenting the request header client CA so that the front proxy will authenticate using its client certificate. ([#40301](https://github.com/kubernetes/kubernetes/pull/40301), [@deads2k](https://github.com/deads2k))
* Fix failing load balancers in Azure ([#40405](https://github.com/kubernetes/kubernetes/pull/40405), [@codablock](https://github.com/codablock))
* Add a KUBERNETES_NODE_* section to build kubelet/kube-proxy for windows ([#38919](https://github.com/kubernetes/kubernetes/pull/38919), [@brendandburns](https://github.com/brendandburns))
* Update GCE ContainerVM deployment to container-vm-v20170117 to pick up CVE fixes in base image. ([#40094](https://github.com/kubernetes/kubernetes/pull/40094), [@zmerlynn](https://github.com/zmerlynn))
* Adding vmdk file extension for vmDiskPath in vsphere DeleteVolume ([#40538](https://github.com/kubernetes/kubernetes/pull/40538), [@divyenpatel](https://github.com/divyenpatel))
* AWS: Remove duplicate calls to DescribeInstance during volume operations ([#39842](https://github.com/kubernetes/kubernetes/pull/39842), [@gnufied](https://github.com/gnufied))
* Caching added to the OIDC client auth plugin to fix races and reduce the time kubectl commands using this plugin take by several seconds. ([#38167](https://github.com/kubernetes/kubernetes/pull/38167), [@ericchiang](https://github.com/ericchiang))
* Actually fix local-cluster-up on 1.5 branch ([#40501](https://github.com/kubernetes/kubernetes/pull/40501), [@lavalamp](https://github.com/lavalamp))
* Prevent hotloops on error conditions, which could fill up the disk faster than log rotation can free space. ([#40497](https://github.com/kubernetes/kubernetes/pull/40497), [@lavalamp](https://github.com/lavalamp))
* Fix issue with PodDisruptionBudgets in which `minAvailable` specified as a percentage did not work with StatefulSet Pods. ([#39454](https://github.com/kubernetes/kubernetes/pull/39454), [@foxish](https://github.com/foxish))
* Fix panic in vSphere cloud provider ([#38423](https://github.com/kubernetes/kubernetes/pull/38423), [@BaluDontu](https://github.com/BaluDontu))
* Allow missing keys in templates by default ([#39486](https://github.com/kubernetes/kubernetes/pull/39486), [@ncdc](https://github.com/ncdc))
* Fix kubectl get -f <file> -o <nondefault printer> so it prints all items in the file ([#39038](https://github.com/kubernetes/kubernetes/pull/39038), [@ncdc](https://github.com/ncdc))
* Endpoints, that tolerate unready Pods, are now listing Pods in state Terminating as well ([#37093](https://github.com/kubernetes/kubernetes/pull/37093), [@simonswine](https://github.com/simonswine))
* Add path exist check in getPodVolumePathListFromDisk ([#38909](https://github.com/kubernetes/kubernetes/pull/38909), [@jingxu97](https://github.com/jingxu97))
* Ensure the GCI metadata files do not have newline at the end ([#38727](https://github.com/kubernetes/kubernetes/pull/38727), [@Amey-D](https://github.com/Amey-D))
* AWS: recognize eu-west-2 region ([#38746](https://github.com/kubernetes/kubernetes/pull/38746), [@justinsb](https://github.com/justinsb))
* Fix space issue in volumePath with vSphere Cloud Provider ([#38338](https://github.com/kubernetes/kubernetes/pull/38338), [@BaluDontu](https://github.com/BaluDontu))
* Fix issue when attempting to unmount a wrong vSphere volume ([#37413](https://github.com/kubernetes/kubernetes/pull/37413), [@BaluDontu](https://github.com/BaluDontu))
* Changed default scsi controller type in vSphere Cloud Provider ([#38426](https://github.com/kubernetes/kubernetes/pull/38426), [@abrarshivani](https://github.com/abrarshivani))
* Fixes API compatibility issue with empty lists incorrectly returning a null `items` field instead of an empty array. ([#39834](https://github.com/kubernetes/kubernetes/pull/39834), [@liggitt](https://github.com/liggitt))
* AWS: Add sequential allocator for device names. ([#38818](https://github.com/kubernetes/kubernetes/pull/38818), [@jsafrane](https://github.com/jsafrane))
* Fix fsGroup to vSphere ([#38655](https://github.com/kubernetes/kubernetes/pull/38655), [@abrarshivani](https://github.com/abrarshivani))



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



# v1.5.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes.tar.gz) | `67344958325a70348db5c4e35e59f9c3552232cdc34defb8a0a799ed91c671a3`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-src.tar.gz) | `93241d0f7b69de71d68384699b225ed8a5439bde03dc154827a2b7a6a343791e`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-darwin-386.tar.gz) | `1e8a3186907fe5e00f8afcd2ca7a207703d5c499d86c80839333cd7cc4eee9ad`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-darwin-amd64.tar.gz) | `64ebd769d96aa5a12f13c4d8c4f6ddce58eae90765c55b7942872dc91447e4d7`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-linux-386.tar.gz) | `a8ecb343a7baf9e01459cd903c09291dbbe72e12431e259e60e11b243b2740f7`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-linux-amd64.tar.gz) | `9d5b6edebb5ee09b20f35d821d3d233ff4d5935880fc8ea8f1fa654d5fd23e51`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-linux-arm64.tar.gz) | `03fd45f96e5d2b66c568b213d0ab6a216aad8c383d5ea4654f7ba8ef5c4d6747`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-linux-arm.tar.gz) | `527fbf42e2e4a2785ad367484a4db619b04484621006fa098cde0ffc3ad3496f`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-windows-386.tar.gz) | `3afe8d3ef470e81a4d793539c2a05fbbca9f0710ced1c132b1105469924e3cea`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-client-windows-amd64.tar.gz) | `dbb63c5211d62512b412efcb52d0a394f19a8417f3e5cd153a7f04c619eb5b41`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-server-linux-amd64.tar.gz) | `8c4be20caa87530fdd17e539abe6f2d3cfccaef9156d262d4d9859ca8b6e3a38`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-server-linux-arm64.tar.gz) | `e0251c3209acebf55e98db521cf29aaa74076a4119b1b19780620faf81d18f44`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.2/kubernetes-server-linux-arm.tar.gz) | `548ad7e061263ff53b80f3ab10a3c7f9289e89a4c56b5a8f49ae513ba88ea93a`

## Changelog since v1.5.1

### Other notable changes

* Fixes NotAuthenticated errors that appear in the kubelet and kube-controller-manager due to never logging in to vSphere ([#36169](https://github.com/kubernetes/kubernetes/pull/36169), [@robdaemon](https://github.com/robdaemon))
* Update amd64 kube-proxy base image to debian-iptables-amd64:v5 ([#39725](https://github.com/kubernetes/kubernetes/pull/39725), [@ixdy](https://github.com/ixdy))
* Update kube-proxy image to be based off of Debian 8.6 base image. ([#39695](https://github.com/kubernetes/kubernetes/pull/39695), [@ixdy](https://github.com/ixdy))
* Fixes an HPA-related panic due to division-by-zero. ([#39694](https://github.com/kubernetes/kubernetes/pull/39694), [@DirectXMan12](https://github.com/DirectXMan12))
* Update fluentd-gcp addon to 1.28.1 ([#39706](https://github.com/kubernetes/kubernetes/pull/39706), [@ixdy](https://github.com/ixdy))
* Provide kubernetes-controller-manager flags to control volume attach/detach reconciler sync.  The duration of the syncs can be controlled, and the syncs can be shut off as well.  ([#39551](https://github.com/kubernetes/kubernetes/pull/39551), [@chrislovecnm](https://github.com/chrislovecnm))
* AWS: Recognize ca-central-1 region ([#38410](https://github.com/kubernetes/kubernetes/pull/38410), [@justinsb](https://github.com/justinsb))
* fix nil dereference when doing a volume type check on persistent volumes ([#39529](https://github.com/kubernetes/kubernetes/pull/39529), [@sjenning](https://github.com/sjenning))
* Generate OpenAPI definition for inlined types ([#39466](https://github.com/kubernetes/kubernetes/pull/39466), [@mbohlool](https://github.com/mbohlool))
* Admit critical pods in the kubelet ([#38836](https://github.com/kubernetes/kubernetes/pull/38836), [@bprashanth](https://github.com/bprashanth))
* assign -998 as the oom_score_adj for critical pods (e.g. kube-proxy) ([#39114](https://github.com/kubernetes/kubernetes/pull/39114), [@dchen1107](https://github.com/dchen1107))
* Don't evict static pods ([#39059](https://github.com/kubernetes/kubernetes/pull/39059), [@bprashanth](https://github.com/bprashanth))
* Fix an issue where AWS tear-down leaks an DHCP Option Set. ([#38645](https://github.com/kubernetes/kubernetes/pull/38645), [@zmerlynn](https://github.com/zmerlynn))
* Give apply the versioned struct that generated from the type defined in the restmapping. ([#38982](https://github.com/kubernetes/kubernetes/pull/38982), [@ymqytw](https://github.com/ymqytw))
* Add support for Azure Container Registry, update Azure dependencies ([#37783](https://github.com/kubernetes/kubernetes/pull/37783), [@brendandburns](https://github.com/brendandburns))
* Fixes an issue where `hack/local-up-cluster.sh` would fail on the API server start with ([#38898](https://github.com/kubernetes/kubernetes/pull/38898), [@deads2k](https://github.com/deads2k))
    * !!! [1215 15:42:56] Timed out waiting for apiserver:  to answer at https://localhost:6443/version; tried 10 waiting 1 between each
* Since `kubernetes.tar.gz` no longer includes client or server binaries, `cluster/kube-{up,down,push}.sh` now automatically download released binaries if they are missing. ([#38730](https://github.com/kubernetes/kubernetes/pull/38730), [@ixdy](https://github.com/ixdy))
* Fixed validation of multizone cluster for GCE ([#38695](https://github.com/kubernetes/kubernetes/pull/38695), [@jszczepkowski](https://github.com/jszczepkowski))
* Fix nil pointer dereference in test framework ([#37583](https://github.com/kubernetes/kubernetes/pull/37583), [@mtaufen](https://github.com/mtaufen))
* Fixed detection of master during creation of multizone nodes cluster by kube-up. ([#38617](https://github.com/kubernetes/kubernetes/pull/38617), [@jszczepkowski](https://github.com/jszczepkowski))
* Kubelet: Add image cache. ([#38375](https://github.com/kubernetes/kubernetes/pull/38375), [@Random-Liu](https://github.com/Random-Liu))



# v1.4.8

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.4/examples)

## Downloads for v1.4.8


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes.tar.gz) | `888d2e6c5136e8805805498729a1da55cf89addfd28f098e0d2cf3f28697ab5c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-src.tar.gz) | `0992c3f4f4cb21011fea32187c909babc1a3806f35cec86aacfe9c3d8bef2485`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-darwin-386.tar.gz) | `8b1c9931544b7b42df64ea98e0d8e1430d09eea3c9f78309834e4e18b091dc18`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-darwin-amd64.tar.gz) | `a306a687979013b8a27acae244d000de9a77f73714ccf96510ecf0398d677051`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-linux-386.tar.gz) | `81fc5e1b5aba4e0aead37c82c7e45891c4493c7df51da5200f83462b6f7ad98f`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-linux-amd64.tar.gz) | `704a5f8424190406821b69283f802ade95e39944efcce10bcaf4bd7b3183abc4`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-linux-arm64.tar.gz) | `7f3e5e8dadb51257afa8650bcd3db3e8f3bc60e767c1a13d946b88fa8625a326`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-linux-arm.tar.gz) | `461d359067cd90542ce2ceb46a4b2ec9d92dd8fd1e7d21a9d9f469c98f446e56`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-windows-386.tar.gz) | `894a9c8667e4c4942cb25ac32d10c4f6de8477c6bbbad94e9e6f47121151f5df`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-client-windows-amd64.tar.gz) | `b2bd4afdd3eaea305c03b94b0864c5622abf19113c6794dedff4ad85327fda01`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-server-linux-amd64.tar.gz) | `c3dc0e26c00bbe40bd19f61d2d7faeaa56384355c58a0efc4227a360b3eb2da2`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-server-linux-arm64.tar.gz) | `745d7ba03bb9c6b57a5a36b389f6467a0707f0a1476d7536ad47417c853eeffd`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.4.8/kubernetes-server-linux-arm.tar.gz) | `dc21f9c659f1d762cad9d0cce0a32146c11cd0d41c58eb2dcbfb0c9f9707349f`

## Changelog since v1.4.7

### Other notable changes

* AWS: recognize eu-west-2 region ([#38746](https://github.com/kubernetes/kubernetes/pull/38746), [@justinsb](https://github.com/justinsb))
* Add path exist check in getPodVolumePathListFromDisk ([#38909](https://github.com/kubernetes/kubernetes/pull/38909), [@jingxu97](https://github.com/jingxu97))
* Update fluentd-gcp addon to 1.21.1/1.25.1. ([#39705](https://github.com/kubernetes/kubernetes/pull/39705), [@ixdy](https://github.com/ixdy))
* Admit critical pods in the kubelet ([#38836](https://github.com/kubernetes/kubernetes/pull/38836), [@bprashanth](https://github.com/bprashanth))
* assign -998 as the oom_score_adj for critical pods (e.g. kube-proxy) ([#39114](https://github.com/kubernetes/kubernetes/pull/39114), [@dchen1107](https://github.com/dchen1107))
* Don't evict static pods ([#39059](https://github.com/kubernetes/kubernetes/pull/39059), [@bprashanth](https://github.com/bprashanth))
* Provide kubernetes-controller-manager flags to control volume attach/detach reconciler sync.  The duration of the syncs can be controlled, and the syncs can be shut off as well.  ([#39551](https://github.com/kubernetes/kubernetes/pull/39551), [@chrislovecnm](https://github.com/chrislovecnm))
* AWS: Recognize ca-central-1 region ([#38410](https://github.com/kubernetes/kubernetes/pull/38410), [@justinsb](https://github.com/justinsb))
* Add TLS conf for Go1.7 ([#38600](https://github.com/kubernetes/kubernetes/pull/38600), [@k82cn](https://github.com/k82cn))
* Fix fsGroup to vSphere ([#38655](https://github.com/kubernetes/kubernetes/pull/38655), [@abrarshivani](https://github.com/abrarshivani))
* Only set sysctls for infra containers ([#32383](https://github.com/kubernetes/kubernetes/pull/32383), [@sttts](https://github.com/sttts))
* fix kubectl taint e2e flake: add retries for removing taint ([#33872](https://github.com/kubernetes/kubernetes/pull/33872), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* portfordwardtester: avoid data loss during send+close+exit ([#37103](https://github.com/kubernetes/kubernetes/pull/37103), [@sttts](https://github.com/sttts))
* Wait for the port to be ready before starting ([#38260](https://github.com/kubernetes/kubernetes/pull/38260), [@fraenkel](https://github.com/fraenkel))
* Ensure the GCI metadata files do not have newline at the end ([#38727](https://github.com/kubernetes/kubernetes/pull/38727), [@Amey-D](https://github.com/Amey-D))
* Fix nil pointer dereference in test framework ([#37583](https://github.com/kubernetes/kubernetes/pull/37583), [@mtaufen](https://github.com/mtaufen))
* Kubelet: Add image cache. ([#38375](https://github.com/kubernetes/kubernetes/pull/38375), [@Random-Liu](https://github.com/Random-Liu))
* Collect logs for dead kubelets too ([#37671](https://github.com/kubernetes/kubernetes/pull/37671), [@mtaufen](https://github.com/mtaufen))



# v1.5.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes.tar.gz) | `adc4f6ec1fc8f97ed19f474ffcc0af2d050f92dc20ecec2799741802019205ec`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-src.tar.gz) | `27e5009b906b9f233a7be1efcf51140be945446d828c006c171d03fe07e43565`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-darwin-386.tar.gz) | `06f8155f0df381bca3b4e27bbd28834f7601e32cbe3d0c1f24be90516c5b8a3b`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-darwin-amd64.tar.gz) | `3ede7d74c5f2f918547bca4d813901e33580c8b8f19828da21a5c2296ff4b8be`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-linux-386.tar.gz) | `b96c3c359146e4fc4d8ff4cf09216bbbb9dbaf3f405488d4aaa45ac741c98f99`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-linux-amd64.tar.gz) | `662fc57057290deb38ec49dd7daf4a4a5b91def2dbdb7ee7a4494dec611379a5`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-linux-arm64.tar.gz) | `c33936b7a27f296c7b85bbfac1fe303573580a948dd1f3174916da9a5a954d49`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-linux-arm.tar.gz) | `31ea3e4cbcc9574a37566a2cc3c809105d56a739e9cbd387bf878acacedf9ec8`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-windows-386.tar.gz) | `95420d0d49e2875703ac09a1b6021252644ba162349c6c506b06f2677852de5d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-client-windows-amd64.tar.gz) | `534a3c5bdde989c7339df05c4e7793c6c50e5ebc0a663b1a9cdd25bce43a5a74`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-server-linux-amd64.tar.gz) | `871a9f35e1c73f571b7113e01a91d7bfc5bfe3501e910c921a18313774b25fd1`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-server-linux-arm64.tar.gz) | `e13b070ef70d2cea512a839095dbf95249d2f7b5dcbfb378539548c888efe196`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.1/kubernetes-server-linux-arm.tar.gz) | `c54cf106e919149731a23da60ad354eadc53b3bf544ab91d4d48ff0c87fdaa7e`

## Changelog since v1.5.0

### Other notable changes

* Changes the default value of the "anonymous-auth" flag to a safer default value of false. This affects kube apiserver and federation apiserver. See https://groups.google.com/forum/#!topic/kubernetes-announce/iclRj-6Nfsg for more details.  ([#38708](https://github.com/kubernetes/kubernetes/pull/38708), [@erictune](https://github.com/erictune))
* Fixes issue where if the audit log is enabled and anonymous authentication is disabled, then an unauthenticated user request will cause a panic and crash the `kube-apiserver`. ([#38717](https://github.com/kubernetes/kubernetes/pull/38717), [@deads2k](https://github.com/deads2k))

## Known Issues for v1.5.1

- `hack/local-up-cluster.sh` script times out waiting for apiserver to answer, see [#38847](https://github.com/kubernetes/kubernetes/issues/38847).
  To workaround this, modify the script to pass `--anonymous-auth=true` to `sudo -E "${GO_OUT}/hyperkube" apiserver ...` when starting `kube-apiserver`.

# v1.5.0

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.0


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes.tar.gz) | `52b7df98ea05fb3ebbababf1ccb7f6d4e6f4cad00b8d09350f270aa7e3ad7e85`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-src.tar.gz) | `fbefb2544667f96045c346cee595b0f315282dfdbd41a8f2d5ccc74054a4078e`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-darwin-386.tar.gz) | `27d71bb6b16a26387ee30272bd4ee5758deccafafdc91b38f3d0dc19a34e129e`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-darwin-amd64.tar.gz) | `5fa8550235919568d7d839b19de00e9bdd72a97cfde21dbdbe07fefd6d6290dc`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-linux-386.tar.gz) | `032a17701c014b8bbbb83c7da1046d8992a41031628cf7e1959a94378f5f195b`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-linux-amd64.tar.gz) | `afae4fadb7bbb1532967f88fef1de6458abda17219f634cc2c41608fd83ae7f6`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-linux-arm64.tar.gz) | `acca7607dae678a0165b7e10685e0eff0d418beebe7c25eaffe18c85717b5cc4`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-linux-arm.tar.gz) | `fbc182b6d9ae476c7c509486d773074fd1007032886a8177735e08010c43f89d`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-windows-386.tar.gz) | `a8ddea329bc8d57267294464c163d8c2f7837f6353f8c685271864ed8b8bc54d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-client-windows-amd64.tar.gz) | `bc3a76f1414fa1f4b2fb92732de2100d346edb7b870ed5414ea062bb401a8ebd`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-server-linux-amd64.tar.gz) | `b9c122d709c0556c1e19d31d98bf26ee530f91c0119f4454fb930cef5a0c1aa7`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-server-linux-arm64.tar.gz) | `3bbba5c8dedc47db8f9ebdfac5468398cce2470617de9d550affef9702b724c9`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.0/kubernetes-server-linux-arm.tar.gz) | `3ff9ccdd641690fd1c8878408cd369beca1f9f8b212198e251862d40cf2dadc0`

## Major Themes

- StatefulSets (ex-PetSets)
  - StatefulSets are beta now (fixes and stabilization)
- Improved Federation Support
  - New command: `kubefed`
  - DaemonSets
  - Deployments
  - ConfigMaps
- Simplified Cluster Deployment
  - Improvements to `kubeadm`
  - HA Setup for Master
- Node Robustness and Extensibility
  - Windows Server Container support
  - CRI for pluggable container runtimes
  - `kubelet` API supports authentication and authorization

## Features

Features for this release were tracked via the use of the [kubernetes/features](https://github.com/kubernetes/features) issues repo.  Each Feature issue is owned by a Special Interest Group from [kubernetes/community](https://github.com/kubernetes/community)

- **API Machinery**
  - [beta] `kube-apiserver` support for the OpenAPI spec is moving from alpha to beta. The first [non-go client](https://github.com/kubernetes-incubator/client-python) is based on it ([kubernetes/features#53](https://github.com/kubernetes/features/issues/53))
- **Apps**
  - [stable] When replica sets cannot create pods, they will now report detail via the API about the underlying reason ([kubernetes/features#120](https://github.com/kubernetes/features/issues/120))
  - [stable] `kubectl apply` is now able to delete resources you no longer need with `--prune` ([kubernetes/features#128](https://github.com/kubernetes/features/issues/128))
  - [beta] Deployments that cannot make progress in rolling out the newest version will now indicate via the API they are blocked ([docs](http://kubernetes.io/docs/user-guide/deployments/#failed-deployment)) ([kubernetes/features#122](https://github.com/kubernetes/features/issues/122))
  - [beta] StatefulSets allow workloads that require persistent identity or per-instance storage to be created and managed on Kubernetes. ([docs](http://kubernetes.io/docs/concepts/abstractions/controllers/statefulsets/)) ([kubernetes/features#137](https://github.com/kubernetes/features/issues/137))
  - [beta] In order to preserve safety guarantees the cluster no longer force deletes pods on un-responsive nodes and users are now warned if they try to force delete pods via the CLI. ([docs](http://kubernetes.io/docs/tasks/manage-stateful-set/scale-stateful-set/)) ([kubernetes/features#119](https://github.com/kubernetes/features/issues/119))
- **Auth**
  - [alpha] Further polishing of the Role-based access control alpha API including a default set of cluster roles. ([docs](http://kubernetes.io/docs/admin/authorization/)) ([kubernetes/features#2](https://github.com/kubernetes/features/issues/2))
  - [beta] Added ability to authenticate/authorize access to the Kubelet API ([docs](http://kubernetes.io/docs/admin/kubelet-authentication-authorization/)) ([kubernetes/features#89](https://github.com/kubernetes/features/issues/89))
- **AWS**
  - [stable] Roles should appear in kubectl get nodes ([kubernetes/features#113](https://github.com/kubernetes/features/issues/113))
- **Cluster Lifecycle**
  - [alpha] Improved UX and usability for the kubeadm binary that makes it easy to get a new cluster running. ([docs](http://kubernetes.io/docs/getting-started-guides/kubeadm/)) ([changelog](https://github.com/kubernetes/kubeadm/blob/master/CHANGELOG.md)) ([kubernetes/features#11](https://github.com/kubernetes/features/issues/11))
- **Cluster Ops**
  - [alpha] Added ability to create/remove clusters w/highly available (replicated) masters on GCE using kube-up/kube-down scripts. ([docs](http://kubernetes.io/docs/admin/ha-master-gce/)) ([kubernetes/features#48](https://github.com/kubernetes/features/issues/48))
- **Federation**
  - [alpha] Support for ConfigMaps in federation. ([docs](http://kubernetes.io/docs/user-guide/federation/configmap/)) ([kubernetes/features#105](https://github.com/kubernetes/features/issues/105))
  - [alpha] Alpha level support for DaemonSets in federation. ([docs](http://kubernetes.io/docs/user-guide/federation/daemonsets/)) ([kubernetes/features#101](https://github.com/kubernetes/features/issues/101))
  - [alpha] Alpha level support for Deployments in federation. ([docs](http://kubernetes.io/docs/user-guide/federation/deployment/)) ([kubernetes/features#100](https://github.com/kubernetes/features/issues/100))
  - [alpha] Cluster federation: Added support for DeleteOptions.OrphanDependents for federation resources. ([docs](http://kubernetes.io/docs/user-guide/federation/#cascading-deletion)) ([kubernetes/features#99](https://github.com/kubernetes/features/issues/99))
  - [alpha] Introducing `kubefed`, a new command line tool to simplify federation control plane. ([docs](http://kubernetes.io/docs/admin/federation/kubefed/)) ([kubernetes/features#97](https://github.com/kubernetes/features/issues/97))
- **Network**
  - [stable] Services can reference another service by DNS name, rather than being hosted in pods ([kubernetes/features#33](https://github.com/kubernetes/features/issues/33))
  - [beta] Opt in source ip preservation for Services with Type NodePort or LoadBalancer ([docs](http://kubernetes.io/docs/tutorials/services/source-ip/)) ([kubernetes/features#27](https://github.com/kubernetes/features/issues/27))
  - [stable] Enable DNS Horizontal Autoscaling with beta ConfigMap parameters support ([docs](http://kubernetes.io/docs/tasks/administer-cluster/dns-horizontal-autoscaling/))
- **Node**
  - [alpha] Added ability to preserve access to host userns when userns remapping is enabled in container runtime ([kubernetes/features#127](https://github.com/kubernetes/features/issues/127))
  - [alpha] Introducing the v1alpha1 CRI API to allow pluggable container runtimes; an experimental docker-CRI integration is ready for testing and feedback. ([docs](https://github.com/kubernetes/community/blob/master/contributors/devel/container-runtime-interface.md)) ([kubernetes/features#54](https://github.com/kubernetes/features/issues/54))
  - [alpha] Kubelet launches container in a per pod cgroup hierarchy based on quality of service tier ([kubernetes/features#126](https://github.com/kubernetes/features/issues/126))
  - [beta] Kubelet integrates with memcg notification API to detect when a hard eviction threshold is crossed ([kubernetes/features#125](https://github.com/kubernetes/features/issues/125))
  - [beta] Introducing the beta version containerized node conformance test gcr.io/google_containers/node-test:0.2 for users to verify node setup. ([docs](http://kubernetes.io/docs/admin/node-conformance/)) ([kubernetes/features#84](https://github.com/kubernetes/features/issues/84))
- **Scheduling**
  - [alpha] Added support for accounting opaque integer resources. ([docs](http://kubernetes.io/docs/user-guide/compute-resources/#opaque-integer-resources-alpha-feature)) ([kubernetes/features#76](https://github.com/kubernetes/features/issues/76))
  - [beta] PodDisruptionBudget has been promoted to beta, can be used to safely drain nodes while respecting application SLO's ([docs](http://kubernetes.io/docs/tasks/administer-cluster/safely-drain-node/)) ([kubernetes/features#85](https://github.com/kubernetes/features/issues/85))
- **UI**
  - [stable] Dashboard UI now shows all user facing objects and their resource usage. ([docs](http://kubernetes.io/docs/user-guide/ui/)) ([kubernetes/features#136](https://github.com/kubernetes/features/issues/136))
- **Windows**
  - [alpha] Added support for Windows Server 2016 nodes and scheduling Windows Server Containers ([docs](http://kubernetes.io/docs/getting-started-guides/windows/)) ([kubernetes/features#116](https://github.com/kubernetes/features/issues/116))

## Known Issues

Populated via [v1.5.0 known issues / FAQ accumulator](https://github.com/kubernetes/kubernetes/issues/37134)

* CRI [known issues and
  limitations](https://github.com/kubernetes/community/blob/master/contributors/devel/container-runtime-interface.md#kubernetes-v15-release-cri-v1alpha1)
* getDeviceNameFromMount() function doesn't return the volume path correctly when the volume path contains spaces [#37712](https://github.com/kubernetes/kubernetes/issues/37712)
* Federation alpha features do not have feature gates defined and
are hence enabled by default. This will be fixed in a future release.
[#38593](https://github.com/kubernetes/kubernetes/issues/38593)
* Federation control plane can be upgraded by updating the image
fields in the `Deployment` specs of the control plane components.
However, federation control plane upgrades were not tested in this
release [38537](https://github.com/kubernetes/kubernetes/issues/38537)

## Notable Changes to Existing Behavior

* Node controller no longer force-deletes pods from the api-server. ([#35235](https://github.com/kubernetes/kubernetes/pull/35235), [@foxish](https://github.com/foxish))
  * For StatefulSet (previously PetSet), this change means creation of
    replacement pods is blocked until old pods are definitely not running
    (indicated either by the kubelet returning from partitioned state,
    deletion of the Node object, deletion of the instance in the cloud provider,
    or force deletion of the pod from the api-server).
    This helps prevent "split brain" scenarios in clustered applications by
    ensuring that unreachable pods will not be presumed dead unless some
    "fencing" operation has provided one of the above indications.
  * For all other existing controllers except StatefulSet, this has no effect on
    the ability of the controller to replace pods because the controllers do not
    reuse pod names (they use generate-name).
  * User-written controllers that reuse names of pod objects should evaluate this change.
  * When deleting an object with `kubectl delete ... --grace-period=0`, the client will
    begin a graceful deletion and wait until the resource is fully deleted.  To force
    deletion immediately, use the `--force` flag. This prevents users from accidentally
    allowing two Stateful Set pods to share the same persistent volume which could lead to data
    corruption [#37263](https://github.com/kubernetes/kubernetes/pull/37263)


* Allow anonymous API server access, decorate authenticated users with system:authenticated group ([#32386](https://github.com/kubernetes/kubernetes/pull/32386), [@liggitt](https://github.com/liggitt))
  * kube-apiserver learned the '--anonymous-auth' flag, which defaults to true. When enabled, requests to the secure port that are not rejected by other configured authentication methods are treated as anonymous requests, and given a username of 'system:anonymous' and a group of 'system:unauthenticated'.
  * Authenticated users are decorated with a 'system:authenticated' group.
  * **IMPORTANT**: See Action Required for important actions related to this change.

* kubectl get -o jsonpath=... will now throw an error if the path is to a field not present in the json, even if the path is for a field valid for the type.  This is a change from the pre-1.5 behavior, which would return the default value for some fields even if they were not present in the json. ([#37991](https://github.com/kubernetes/kubernetes/pull/37991), [@pwittrock](https://github.com/pwittrock))

* The strategicmerge patchMergeKey for VolumeMounts was changed from "name" to "mountPath".  This was necessary because the name field refers to the name of the Volume, and is not a unique key for the VolumeMount.  Multiple VolumeMounts will have the same Volume name if mounting the same volume more than once.  The "mountPath" is verified to be unique and can act as the mergekey.  ([#35071](https://github.com/kubernetes/kubernetes/pull/35071), [@pwittrock](https://github.com/pwittrock))

## Deprecations

* extensions/v1beta1.Jobs is deprecated, use batch/v1.Job instead ([#36355](https://github.com/kubernetes/kubernetes/pull/36355), [@soltysh](https://github.com/soltysh))
* The kubelet --reconcile-cdir flag is deprecated because it has no function anymore. ([#35523](https://github.com/kubernetes/kubernetes/pull/35523), [@luxas](https://github.com/luxas))
* Notice of deprecation for recycler [#36760](https://github.com/kubernetes/kubernetes/pull/36760)
* The init-container (pod.beta.kubernetes.io/init-containers) annotations used to accept capitalized field names that could be accidently generated by the k8s.io/kubernetes/pkg/api package. Using an upper case field name will now return an error and all users should use the versioned API types from `pkg/api/v1` when serializing from Golang.

## Action Required Before Upgrading

* **Important Security-related changes before upgrading
  * You *MUST* set `--anonymous-auth=false` flag on your kube-apiserver unless you are a developer testing this feature and understand it.
  If you do not, you risk allowing unauthorized users to access your apiserver.
  * You *MUST* set `--anonymous-auth=false` flag on your federation apiserver unless you are a developer testing this feature and understand it.
    If you do not, you risk allowing unauthorized users to access your federation apiserver.
  * You do not need to adjust this flag on Kubelet:  there was no authorization for the Kubelet APIs in 1.4.
* batch/v2alpha1.ScheduledJob has been renamed, use batch/v2alpha1.CronJob instead ([#36021](https://github.com/kubernetes/kubernetes/pull/36021), [@soltysh](https://github.com/soltysh))
* PetSet has been renamed to StatefulSet.
  If you have existing PetSets, **you must perform extra migration steps** both
  before and after upgrading to convert them to StatefulSets. ([docs](http://kubernetes.io/docs/tasks/manage-stateful-set/upgrade-pet-set-to-stateful-set/)) ([#35663](https://github.com/kubernetes/kubernetes/pull/35663), [@janetkuo](https://github.com/janetkuo))
* If you are upgrading your Cluster Federation components from v1.4.x, please update your `federation-apiserver` and `federation-controller-manager` manifests to the new version ([#30601](https://github.com/kubernetes/kubernetes/pull/30601), [@madhusudancs](https://github.com/madhusudancs))
* The deprecated kubelet --configure-cbr0 flag has been removed, and with that the "classic" networking mode as well.  If you depend on this mode, please investigate whether the other network plugins `kubenet` or `cni` meet your needs. ([#34906](https://github.com/kubernetes/kubernetes/pull/34906), [@luxas](https://github.com/luxas))
* New client-go structure, refer to kubernetes/client-go for versioning policy ([#34989](https://github.com/kubernetes/kubernetes/pull/34989), [@caesarxuchao](https://github.com/caesarxuchao))
* The deprecated kube-scheduler --bind-pods-qps and --bind-pods burst flags have been removed, use --kube-api-qps and --kube-api-burst instead ([#34471](https://github.com/kubernetes/kubernetes/pull/34471), [@timothysc](https://github.com/timothysc))
* If you used the [PodDisruptionBudget](http://kubernetes.io/docs/admin/disruptions/) feature in 1.4 (i.e. created `PodDisruptionBudget` objects), then **BEFORE**  upgrading from 1.4 to 1.5, you must delete all `PodDisruptionBudget` objects (`policy/v1alpha1/PodDisruptionBudget`) that you have created. It is not possible to delete these objects after you upgrade, and their presence will prevent you from using the beta PodDisruptionBudget feature in 1.5 (which uses `policy/v1beta1/PodDisruptionBudget`). If you have already upgraded, you will need to downgrade the master to 1.4 to delete the `policy/v1alpha1/PodDisruptionBudget` objects.

## External Dependency Version Information

Continuous integration builds have used the following versions of external dependencies, however, this is not a strong recommendation and users should consult an appropriate installation or upgrade guide before deciding what versions of etcd, docker or rkt to use.

* Docker versions 1.10.3 - 1.12.3
  * Docker version 1.11.2 known issues
    - Kernel crash with Aufs storage driver on Debian Jessie ([#27885]((https://github.com/kubernetes/kubernetes/issues/27885))
      which can be identified by the [node problem detector](http://kubernetes.io/docs/admin/node-problem/)
    - Leaked File descriptors ([#275](https://github.com/docker/containerd/issues/275))
    - Additional memory overhead per container ([#21737]((https://github.com/docker/docker/issues/21737))
  * Docker version 1.12.1 [has been validated](https://github.com/kubernetes/kubernetes/issues/28698) through the Kubernetes docker automated validation framework as has Docker version 1.12.3
 *  Docker 1.10.3 contains [backports provided by RedHat](https://github.com/docker/docker/compare/v1.10.3...runcom:docker-1.10.3-stable) for known issues
  * Docker versions as old as may 1.9.1 work with [known issues](CHANGELOG.md#191) but this is not guaranteed
* rkt version 1.21.0
  * known issues with the rkt runtime are [listed here](http://kubernetes.io/docs/getting-started-guides/rkt/notes/)
* etcd version 2.2.1
  * etcd version 3.0.14 [has also been validated](https://k8s-gubernator.appspot.com/builds/kubernetes-jenkins/logs/ci-kubernetes-e2e-gce-etcd3/) but does require [specific configuration steps](https://coreos.com/blog/migrating-applications-etcd-v3.html)

## Changelog since v1.5.0-beta.3

### Other notable changes

* Bump GCE debian image to container-vm-v20161208 ([release notes](https://cloud.google.com/compute/docs/containers/container_vms#changelog)) ([#38444](https://github.com/kubernetes/kubernetes/pull/38444), [@timstclair](https://github.com/timstclair))
* Fix unmountDevice issue caused by shared mount in GCI ([#38411](https://github.com/kubernetes/kubernetes/pull/38411), [@jingxu97](https://github.com/jingxu97))

### Previous Releases Included in v1.5.0

- [v1.5.0-beta.3](CHANGELOG.md#v150-beta3)
- [v1.5.0-beta.2](CHANGELOG.md#v150-beta2)
- [v1.5.0-beta.1](CHANGELOG.md#v150-beta1)
- [v1.5.0-alpha.2](CHANGELOG.md#v150-alpha2)
- [v1.5.0-alpha.1](CHANGELOG.md#v150-alpha1)



# v1.4.7

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.4/examples)

## Downloads for v1.4.7


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes.tar.gz) | `d193f76e70322010b3e86ac61c7a893175f9e62d37bece87cfd14ea068c8d187`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-src.tar.gz) | `7c7ef45e903ed2691c73bb2752805f190b4042ba233a6260f2cdeab7d0ac9bd3`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-darwin-386.tar.gz) | `a5a3ec9f5270156cf507b4c6bf2d08da67062a2ed9cb5f21e8891f2fd83f438a`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-darwin-amd64.tar.gz) | `e5328781640b19e86b59aa8afd665dd21999c6740acbee8332cfa20745d6a5ce`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-linux-386.tar.gz) | `61082afc6aee2dc5bbd35bfda2e5991bd9f9730192f1c9396b6db500fc64e121`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-linux-amd64.tar.gz) | `36232c9e21298f5f53dbf4851520a8cc53a2d6b6d2be8810cf5258a067570314`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-linux-arm64.tar.gz) | `802d0c5e7bb55dacdd19afe73ed71d0726960ec9933c49e77051df7e2594790b`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-linux-arm.tar.gz) | `f42d8d2d918b31564d12d742bce2263df0c93807619bd03194028ff2714f1a17`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-windows-386.tar.gz) | `b45dcdfe0ba0177fad5419b4fd6b5b80bf9bca0e56e7fe19d2bc217c9aae1f9d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-client-windows-amd64.tar.gz) | `ae4666aea8fa74ef1cce746d1d90cbadc972850560b65a8eeff4417fdede6b4e`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-server-linux-amd64.tar.gz) | `56e01e9788d1ef0499b1783768022cb188b5bb840d1499a62e9f0a18c2bd2bd5`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-server-linux-arm64.tar.gz) | `6654ef3c142694a79ec2596929ceec36a399407e1fb74b09be1a67c59b30ca42`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.4.7/kubernetes-server-linux-arm.tar.gz) | `b10e78286dea804d69311e3805c35f5414b0669094edec7a2e0ba99170a5d04a`

## Changelog since v1.4.6

### Other notable changes

* Exit with error if <version number or publication> is not the final parameter. ([#37723](https://github.com/kubernetes/kubernetes/pull/37723), [@mtaufen](https://github.com/mtaufen))
* Fix GCI mounter issue ([#38124](https://github.com/kubernetes/kubernetes/pull/38124), [@jingxu97](https://github.com/jingxu97))
* Fix space issue in volumePath with vSphere Cloud Provider ([#38338](https://github.com/kubernetes/kubernetes/pull/38338), [@BaluDontu](https://github.com/BaluDontu))
* Fix panic in vSphere cloud provider ([#38423](https://github.com/kubernetes/kubernetes/pull/38423), [@BaluDontu](https://github.com/BaluDontu))
* Changed default scsi controller type in vSphere Cloud Provider ([#38426](https://github.com/kubernetes/kubernetes/pull/38426), [@abrarshivani](https://github.com/abrarshivani))
* Fix unmountDevice issue caused by shared mount in GCI ([#38411](https://github.com/kubernetes/kubernetes/pull/38411), [@jingxu97](https://github.com/jingxu97))
* Implement CanMount() for gfsMounter for linux ([#36686](https://github.com/kubernetes/kubernetes/pull/36686), [@rkouj](https://github.com/rkouj))
* Better messaging for missing volume binaries on host ([#36280](https://github.com/kubernetes/kubernetes/pull/36280), [@rkouj](https://github.com/rkouj))
* fix mesos unit tests ([#38196](https://github.com/kubernetes/kubernetes/pull/38196), [@deads2k](https://github.com/deads2k))
* Fix Service Update on LoadBalancerSourceRanges Field ([#37720](https://github.com/kubernetes/kubernetes/pull/37720), [@freehan](https://github.com/freehan))
* Include serial port output in GCP log-dump ([#37248](https://github.com/kubernetes/kubernetes/pull/37248), [@mtaufen](https://github.com/mtaufen))
* Collect installation and configuration service logs for tests ([#37401](https://github.com/kubernetes/kubernetes/pull/37401), [@mtaufen](https://github.com/mtaufen))
* Use shasum if sha1sum doesn't exist in the path ([#37362](https://github.com/kubernetes/kubernetes/pull/37362), [@roberthbailey](https://github.com/roberthbailey))
* Guard the ready replica checking by server version ([#37303](https://github.com/kubernetes/kubernetes/pull/37303), [@krousey](https://github.com/krousey))
* Fix issue when attempting to unmount a wrong vSphere volume ([#37413](https://github.com/kubernetes/kubernetes/pull/37413), [@BaluDontu](https://github.com/BaluDontu))
* Fix issue in converting AWS volume ID from mount paths ([#36840](https://github.com/kubernetes/kubernetes/pull/36840), [@jingxu97](https://github.com/jingxu97))
* Correct env var name in configure-helper ([#33848](https://github.com/kubernetes/kubernetes/pull/33848), [@mtaufen](https://github.com/mtaufen))
* wait until the pods are deleted completely ([#34778](https://github.com/kubernetes/kubernetes/pull/34778), [@ymqytw](https://github.com/ymqytw))
* AWS: recognize us-east-2 region ([#35013](https://github.com/kubernetes/kubernetes/pull/35013), [@justinsb](https://github.com/justinsb))
* Replace controller presence checking logic ([#36924](https://github.com/kubernetes/kubernetes/pull/36924), [@krousey](https://github.com/krousey))
* Fix a bug in scheduler happening after retrying unsuccessful bindings ([#37293](https://github.com/kubernetes/kubernetes/pull/37293), [@wojtek-t](https://github.com/wojtek-t))
* Try self-repair scheduler cache or panic ([#37379](https://github.com/kubernetes/kubernetes/pull/37379), [@wojtek-t](https://github.com/wojtek-t))
* Ignore mirror pods with RestartPolicy == Never in restart tests ([#34462](https://github.com/kubernetes/kubernetes/pull/34462), [@yujuhong](https://github.com/yujuhong))
* Change image-puller restart policy to OnFailure ([#37070](https://github.com/kubernetes/kubernetes/pull/37070), [@gmarek](https://github.com/gmarek))
* Filter out non-RestartAlways mirror pod in restart test. ([#37203](https://github.com/kubernetes/kubernetes/pull/37203), [@Random-Liu](https://github.com/Random-Liu))
* Validate volume spec before returning azure mounter ([#37018](https://github.com/kubernetes/kubernetes/pull/37018), [@rootfs](https://github.com/rootfs))
* Networking test rewrite ([#31559](https://github.com/kubernetes/kubernetes/pull/31559), [@bprashanth](https://github.com/bprashanth))
* Fix the equality checks for numeric values in cluster/gce/util.sh. ([#37638](https://github.com/kubernetes/kubernetes/pull/37638), [@roberthbailey](https://github.com/roberthbailey))
* Use gsed on the Mac ([#37562](https://github.com/kubernetes/kubernetes/pull/37562), [@roberthbailey](https://github.com/roberthbailey))
* Fix TestServiceAlloc flakes ([#37487](https://github.com/kubernetes/kubernetes/pull/37487), [@wojtek-t](https://github.com/wojtek-t))
* Change ScheduledJob POD name suffix from hash to Unix Epoch ([#36883](https://github.com/kubernetes/kubernetes/pull/36883), [@jakub-d](https://github.com/jakub-d))
* Add support for NFSv4 and GlusterFS in GCI base image ([#37336](https://github.com/kubernetes/kubernetes/pull/37336), [@jingxu97](https://github.com/jingxu97))
* Use generous limits in the resource usage tracking tests ([#36623](https://github.com/kubernetes/kubernetes/pull/36623), [@yujuhong](https://github.com/yujuhong))



# v1.5.0-beta.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.0-beta.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes.tar.gz) | `c2b29b38d29829b7b2591559d0d36495d463de0e18a2611bd1d66f2baea6352c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-src.tar.gz) | `0b3327b6f0b024c989aba1e546d50d56fc89ed6df74c09fc55b9f9c4a667b771`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-darwin-386.tar.gz) | `82a7144ae1371c3320019c8e6a76e95242d85aae9dedccc4884b677cda544c0e`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-darwin-amd64.tar.gz) | `3aeea90acfbaf776e2c812e34df4c11a44720e4c5b86c4c0e9a8aaf221149335`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-linux-386.tar.gz) | `d55fb1dfe64e62bffbf03f1a7c8bd666562014ad0d438049f0f801f5fa583914`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-linux-amd64.tar.gz) | `779b2f1c0eb3eca7dd60332972ccfc79e557e34f080c210dfb6aa6e18e71bbf4`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-linux-arm64.tar.gz) | `b5f0a3b23d7082eaefe7090d7a8f9952fd8b00d44a90137200bc5a91001b6e95`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-linux-arm.tar.gz) | `ccadbef7ce7c89fc48988c57585c0ccb7488d2dcc7e96f4e43c5bb64e44b9e29`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-windows-386.tar.gz) | `da1428b6ed138134358c72af570a65565c5188a1c6e50cee42becb1a48441d91`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-client-windows-amd64.tar.gz) | `7b74aeb215b0f0ff86bae262af5bafe7083a44293e1ab2545f5de3ac42deda0b`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-server-linux-amd64.tar.gz) | `c56aa39fd4e732c86a2729aa427ca2fc95130bd788053aa8e8f6a8efd9e1310e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-server-linux-arm64.tar.gz) | `9f55082ca5face2db2d6d54bed2a831622e747e1aa527ee8adc61d0ed3fcfab8`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.5.0-beta.3/kubernetes-server-linux-arm.tar.gz) | `4a7c037ac221531eee4e47b66a2aa12fce4044d2d4acbef0e48b09e0a8fe950b`

## Changelog since v1.5.0-beta.2

### Other notable changes

* Better compat with very old iptables (e.g. CentOS 6) ([#37594](https://github.com/kubernetes/kubernetes/pull/37594), [@thockin](https://github.com/thockin))
* fix permissions when using fsGroup ([#37009](https://github.com/kubernetes/kubernetes/pull/37009), [@sjenning](https://github.com/sjenning))
* Fix GCI mounter issue ([#38124](https://github.com/kubernetes/kubernetes/pull/38124), [@jingxu97](https://github.com/jingxu97))
* fix mesos unit tests ([#38196](https://github.com/kubernetes/kubernetes/pull/38196), [@deads2k](https://github.com/deads2k))
* Fix Service Update on LoadBalancerSourceRanges Field ([#37720](https://github.com/kubernetes/kubernetes/pull/37720), [@freehan](https://github.com/freehan))
* Fix Service Update on LoadBalancerSourceRanges Field ([#37720](https://github.com/kubernetes/kubernetes/pull/37720), [@freehan](https://github.com/freehan))
* Set kernel.softlockup_panic =1 based on the flag. ([#38001](https://github.com/kubernetes/kubernetes/pull/38001), [@dchen1107](https://github.com/dchen1107))
* Fix logic error in graceful deletion ([#37721](https://github.com/kubernetes/kubernetes/pull/37721), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Enable containerized mounter only for nfs and glusterfs types ([#37990](https://github.com/kubernetes/kubernetes/pull/37990), [@jingxu97](https://github.com/jingxu97))
* GCI: Remove /var/lib/docker/network ([#37593](https://github.com/kubernetes/kubernetes/pull/37593), [@yujuhong](https://github.com/yujuhong))
* kubelet: don't reject pods without adding them to the pod manager ([#37661](https://github.com/kubernetes/kubernetes/pull/37661), [@yujuhong](https://github.com/yujuhong))
* Fix photon controller plugin to construct with correct PdID ([#37167](https://github.com/kubernetes/kubernetes/pull/37167), [@luomiao](https://github.com/luomiao))
* Fix the equality checks for numeric values in cluster/gce/util.sh. ([#37638](https://github.com/kubernetes/kubernetes/pull/37638), [@roberthbailey](https://github.com/roberthbailey))
* federation service controller: stop deleting services from underlying clusters when federated service is deleted. ([#37353](https://github.com/kubernetes/kubernetes/pull/37353), [@nikhiljindal](https://github.com/nikhiljindal))
* Set Dashboard UI version to v1.5.0 ([#37684](https://github.com/kubernetes/kubernetes/pull/37684), [@rf232](https://github.com/rf232))
* When deleting an object with `--grace-period=0`, the client will begin a graceful deletion and wait until the resource is fully deleted.  To force deletion, use the `--force` flag. ([#37263](https://github.com/kubernetes/kubernetes/pull/37263), [@smarterclayton](https://github.com/smarterclayton))
* Removes shorthand flag -w from kubectl apply ([#37345](https://github.com/kubernetes/kubernetes/pull/37345), [@MrHohn](https://github.com/MrHohn))
* Update doc for kubectl apply ([#37397](https://github.com/kubernetes/kubernetes/pull/37397), [@ymqytw](https://github.com/ymqytw))
* Try self-repair scheduler cache or panic ([#37379](https://github.com/kubernetes/kubernetes/pull/37379), [@wojtek-t](https://github.com/wojtek-t))
* Use gsed on the Mac ([#37562](https://github.com/kubernetes/kubernetes/pull/37562), [@roberthbailey](https://github.com/roberthbailey))
* Fix TestServiceAlloc flakes ([#37487](https://github.com/kubernetes/kubernetes/pull/37487), [@wojtek-t](https://github.com/wojtek-t))



# v1.5.0-beta.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.0-beta.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes.tar.gz) | `4a6cb512dee2312ffe291f4209759309576ca477cf51fb8447b30a7cb2a887ed`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-src.tar.gz) | `fe71f19b607183da4abf5f537e7ccbe72ac3306b0933ee1f519253c78bf9252f`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `37bcd12754a28ba6b4d030c68526bc6369f1fa3b7b0e405277bb13989ed0f9da`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `760817040ca040dd4ba8929cfb714b8bf6704c6ac2ec9985b56fa77b4da03d2c`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-linux-386.tar.gz) | `87d694445a3e532748d07e0d0da05c1ae8b84b46c54ec1415c9603533747a465`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `b2bcd07a525428fe24da628afca22b019b8f2847d1999da8fce72b7342cf64ed`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `262c4fa70039389aa5d5b73a0def325471bd24b858157d60c0389fbee5ca671e`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `52c9341c1e6aa923aed4497c061121c192f209c90fcf31135edc45241a684bfa`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-windows-386.tar.gz) | `7d8e3bcdfa9dc3d5fde70c60a37e543cc59d23b25e2b0a2274e672d0bae013c2`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `75143c176bc817fc49a79229dfae8c7429d0a3deeaba54a397dddce3e37e8550`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `61c209048da1612796a30b880076b7f9b59038821da63bbecac4c56f24216312`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `2c6952e16c0b0c153ca3d424b3deca9b43a8e421b1a59359bc10260309bf470c`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `cf3e37a89358cae1d2d36aaad10f3e906269bc3df611279dbed9f50e81449fad`

## Changelog since v1.5.0-beta.1

### Other notable changes

* Modify GCI mounter to enable NFSv3 ([#36610](https://github.com/kubernetes/kubernetes/pull/36610), [@jingxu97](https://github.com/jingxu97))
* Third party resources are now deleted when a namespace is deleted. ([#35947](https://github.com/kubernetes/kubernetes/pull/35947), [@brendandburns](https://github.com/brendandburns))
* kube-dns ([#36775](https://github.com/kubernetes/kubernetes/pull/36775), [@bowei](https://github.com/bowei))
    * Added --config-map and --config-map-namespace command line options.
    * If --config-map is set, kube-dns will load dynamic configuration from the config map
    * referenced by --config-map-namespace, --config-map. The config-map supports
    * the following properties: "federations".
    * --federations flag is now deprecated. Prefer to set federations via the config-map.
    * Federations can be configured by settings the "federations" field to the value currently
    * set in the command line.
    * Example:
    *   kind: ConfigMap
    *   apiVersion: v1
    *   metadata:
    *     name: kube-dns
    *     namespace: kube-system
    *   data:
    *     federations: abc=def
* azure: support multiple ipconfigs on a NIC ([#36841](https://github.com/kubernetes/kubernetes/pull/36841), [@colemickens](https://github.com/colemickens))
* Fix issue in converting AWS volume ID from mount paths ([#36840](https://github.com/kubernetes/kubernetes/pull/36840), [@jingxu97](https://github.com/jingxu97))
* fix leaking memory backed volumes of terminated pods ([#36779](https://github.com/kubernetes/kubernetes/pull/36779), [@sjenning](https://github.com/sjenning))
* Default logging subsystem's resiliency was greatly improved, fluentd memory consumption and OOM error probability was reduced. ([#37021](https://github.com/kubernetes/kubernetes/pull/37021), [@Crassirostris](https://github.com/Crassirostris))



# v1.5.0-beta.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.5/examples)

## Downloads for v1.5.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes.tar.gz) | `62c51bcee460794cda30e720c65509b679b51015c62c075e6e735fe29d089e2b`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-src.tar.gz) | `8c950c7377eb40670d0438ccb68bbeaf1100ed2e919e012bc98479ff07ddd393`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `e71af85542837842ff3b0fb8137332f4e1ce4c453d225da292e1fa781f1c74d7`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `033d02c1382553f977057827b6a5b82f1b69aecd44b649c937781d1cccb763d1`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-linux-386.tar.gz) | `1e7a435f2f7d06e3de9bd8c8d0457b6548aa15ad5cdab4241391f290a28b804f`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `3c07a89e8eb785a7b37842d4b0bc0471fcc7b4e3a4bd973e6f8936cbc6030d76`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `680a2786d9782395b613e27509df2d0f671a2471a43533ccdbc6b71cfb332072`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `2a5b10fbd69ce9b1da0403a80d71684ee2cf4d75298a5ec19e069ae826da81ed`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-windows-386.tar.gz) | `10acbf09ffbc04f549d1cffff98a533b456562d5c09a2d0f315523b70072c35d`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `3317f90da242b0fb95a3cbc669fc4941d7b56b5ff90ac528c166e915bee31fdf`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `fdb257c0bbf64304441fd377a5ee330de10696aa0b5c1b6c27fa73a6c00121ae`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `a174cf6c9351da786b8780f5edca158a4e021d4af597bcc66f238601fb37c2b1`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `1dc520b9a4428321225ba6cfa0f79b702965d7f6994357c15e0195c5af1528ff`

## Changelog since v1.5.0-alpha.2

### Action Required

* Deprecate extensions/v1beta1.Jobs ([#36355](https://github.com/kubernetes/kubernetes/pull/36355), [@soltysh](https://github.com/soltysh))
* Rename ScheduledJobs to CronJobs. ([#36021](https://github.com/kubernetes/kubernetes/pull/36021), [@soltysh](https://github.com/soltysh))
* Read the federation controller manager kubeconfig from a filesystem path ([#30601](https://github.com/kubernetes/kubernetes/pull/30601), [@madhusudancs](https://github.com/madhusudancs))
* Node controller to not force delete pods ([#35235](https://github.com/kubernetes/kubernetes/pull/35235), [@foxish](https://github.com/foxish))
* Add perma-failed deployments API ([#19343](https://github.com/kubernetes/kubernetes/pull/19343), [@kargakis](https://github.com/kargakis))

### Other notable changes

* Federation: allow specification of dns zone by ID ([#36336](https://github.com/kubernetes/kubernetes/pull/36336), [@justinsb](https://github.com/justinsb))
* K8s 1.5 keeps container-vm as the default node image on GCE for backwards compatibility reasons. Please beware that container-vm is officially deprecated (supported with security patches only) and you should replace it with GCI if at all possible. You can review the migration guide here for more detail: https://cloud.google.com/container-engine/docs/node-image-migration ([#36822](https://github.com/kubernetes/kubernetes/pull/36822), [@mtaufen](https://github.com/mtaufen))
* Add a flag allowing contention profiling of the API server ([#36756](https://github.com/kubernetes/kubernetes/pull/36756), [@gmarek](https://github.com/gmarek))
* Rename `--cgroups-per-qos` to `--experimental-cgroups-per-qos` in Kubelet ([#36767](https://github.com/kubernetes/kubernetes/pull/36767), [@vishh](https://github.com/vishh))
* Implement CanMount() for gfsMounter for linux ([#36686](https://github.com/kubernetes/kubernetes/pull/36686), [@rkouj](https://github.com/rkouj))
* Default host user namespace via experimental flag ([#31169](https://github.com/kubernetes/kubernetes/pull/31169), [@pweil-](https://github.com/pweil-))
* Use generous limits in the resource usage tracking tests ([#36623](https://github.com/kubernetes/kubernetes/pull/36623), [@yujuhong](https://github.com/yujuhong))
* Update Dashboard UI version to 1.4.2 ([#35895](https://github.com/kubernetes/kubernetes/pull/35895), [@rf232](https://github.com/rf232))
* Add support for service load balancer source ranges to Azure load balancers. ([#36696](https://github.com/kubernetes/kubernetes/pull/36696), [@brendandburns](https://github.com/brendandburns))
* gci-dev-56-8977-0-0: ([#36681](https://github.com/kubernetes/kubernetes/pull/36681), [@mtaufen](https://github.com/mtaufen))
    * Date:           Nov 03, 2016
    * Kernel:         ChromiumOS-4.4
    * Kubernetes:     v1.4.5
    * Docker:         v1.11.2
    * Changelog (vs 55-8872-18-0)
            * Updated kubernetes to v1.4.5
            * Fixed a bug in e2fsprogs that caused mke2fs to take a very long time. Upstream fix: http://git.kernel.org/cgit/fs/ext2/e2fsprogs.git/commit/?h=next&id=d33e690fe7a6cbeb51349d9f2c7fb16a6ebec9c2
* Fix strategic patch for list of primitive type with merge sementic ([#35647](https://github.com/kubernetes/kubernetes/pull/35647), [@ymqytw](https://github.com/ymqytw))
* Fix issue in reconstruct volume data when kubelet restarts ([#36616](https://github.com/kubernetes/kubernetes/pull/36616), [@jingxu97](https://github.com/jingxu97))
* Ensure proper serialization of updates and creates in federation test watcher ([#36613](https://github.com/kubernetes/kubernetes/pull/36613), [@mwielgus](https://github.com/mwielgus))
* Add support for SourceIP preservation in Azure LBs ([#36557](https://github.com/kubernetes/kubernetes/pull/36557), [@brendandburns](https://github.com/brendandburns))
* Fix fetching pids running in a cgroup, which caused problems with OOM score adjustments & setting the /system cgroup ("misc" in the summary API). ([#36551](https://github.com/kubernetes/kubernetes/pull/36551), [@timstclair](https://github.com/timstclair))
* federation: Adding support for DeleteOptions.OrphanDependents for federated replicasets and deployments. Setting it to false while deleting a federated replicaset or deployment also deletes the corresponding resource from all registered clusters. ([#36476](https://github.com/kubernetes/kubernetes/pull/36476), [@nikhiljindal](https://github.com/nikhiljindal))
* kubectl: show node label if defined ([#35901](https://github.com/kubernetes/kubernetes/pull/35901), [@justinsb](https://github.com/justinsb))
* Migrates addons from RCs to Deployments ([#36008](https://github.com/kubernetes/kubernetes/pull/36008), [@MrHohn](https://github.com/MrHohn))
* Avoid setting S_ISGID on files in volumes ([#36386](https://github.com/kubernetes/kubernetes/pull/36386), [@sjenning](https://github.com/sjenning))
* federation: Adding support for DeleteOptions.OrphanDependents for federated daemonsets and ingresses. Setting it to false while deleting a federated daemonset or ingress also deletes the corresponding resource from all registered clusters. ([#36330](https://github.com/kubernetes/kubernetes/pull/36330), [@nikhiljindal](https://github.com/nikhiljindal))
* Add authz to psp admission ([#33080](https://github.com/kubernetes/kubernetes/pull/33080), [@pweil-](https://github.com/pweil-))
* Better messaging for missing volume binaries on host ([#36280](https://github.com/kubernetes/kubernetes/pull/36280), [@rkouj](https://github.com/rkouj))
* Add Windows support to kube-proxy ([#36079](https://github.com/kubernetes/kubernetes/pull/36079), [@jbhurat](https://github.com/jbhurat))
* Support persistent volume usage for kubernetes running on Photon Controller platform ([#36133](https://github.com/kubernetes/kubernetes/pull/36133), [@luomiao](https://github.com/luomiao))
* GCI nodes use an external mounter script to mount NFS & GlusterFS storage volumes ([#36267](https://github.com/kubernetes/kubernetes/pull/36267), [@vishh](https://github.com/vishh))
* Add retry to node scheduability marking. ([#36211](https://github.com/kubernetes/kubernetes/pull/36211), [@brendandburns](https://github.com/brendandburns))
* specify custom ca file to verify the keystone server ([#35488](https://github.com/kubernetes/kubernetes/pull/35488), [@dixudx](https://github.com/dixudx))
* AWS: Support default value for ExternalHost ([#33568](https://github.com/kubernetes/kubernetes/pull/33568), [@justinsb](https://github.com/justinsb))
* HPA: Consider unready pods separately ([#33593](https://github.com/kubernetes/kubernetes/pull/33593), [@DirectXMan12](https://github.com/DirectXMan12))
* Node Conformance Test: Containerize the node e2e test ([#31093](https://github.com/kubernetes/kubernetes/pull/31093), [@Random-Liu](https://github.com/Random-Liu))
* federation: Adding support for DeleteOptions.OrphanDependents for federated secrets. Setting it to false while deleting a federated secret also deletes the corresponding secrets from all registered clusters. ([#36296](https://github.com/kubernetes/kubernetes/pull/36296), [@nikhiljindal](https://github.com/nikhiljindal))
* Deploy kube-dns with cluster-proportional-autoscaler ([#33239](https://github.com/kubernetes/kubernetes/pull/33239), [@MrHohn](https://github.com/MrHohn))
* Adds support for StatefulSets in kubectl drain. ([#35483](https://github.com/kubernetes/kubernetes/pull/35483), [@ymqytw](https://github.com/ymqytw))
    * Switches to use the eviction sub-resource instead of deletion in kubectl drain, if server supports.
* azure: load balancer preserves destination ip address ([#36256](https://github.com/kubernetes/kubernetes/pull/36256), [@colemickens](https://github.com/colemickens))
* LegacyHostIP will be deprecated in 1.7. ([#36095](https://github.com/kubernetes/kubernetes/pull/36095), [@caesarxuchao](https://github.com/caesarxuchao))
* Fix LBaaS version detection in openstack cloudprovider ([#36249](https://github.com/kubernetes/kubernetes/pull/36249), [@sjenning](https://github.com/sjenning))
* Node Conformance Test: Add system verification ([#32427](https://github.com/kubernetes/kubernetes/pull/32427), [@Random-Liu](https://github.com/Random-Liu))
* kubelet bootstrap: start hostNetwork pods before we have PodCIDR ([#35526](https://github.com/kubernetes/kubernetes/pull/35526), [@justinsb](https://github.com/justinsb))
* Enable HPA controller based on autoscaling/v1 api group ([#36215](https://github.com/kubernetes/kubernetes/pull/36215), [@piosz](https://github.com/piosz))
* Remove unused WaitForDetach from Detacher interface and plugins ([#35629](https://github.com/kubernetes/kubernetes/pull/35629), [@kiall](https://github.com/kiall))
* Initial work on running windows containers on Kubernetes ([#31707](https://github.com/kubernetes/kubernetes/pull/31707), [@alexbrand](https://github.com/alexbrand))
* Per Volume Inode Accounting ([#35132](https://github.com/kubernetes/kubernetes/pull/35132), [@dashpole](https://github.com/dashpole))
* [AppArmor] Hold bad AppArmor pods in pending rather than rejecting ([#35342](https://github.com/kubernetes/kubernetes/pull/35342), [@timstclair](https://github.com/timstclair))
* Federation: separate notion of zone-name & dns-suffix ([#35372](https://github.com/kubernetes/kubernetes/pull/35372), [@justinsb](https://github.com/justinsb))
* In order to bypass graceful deletion of pods (to immediately remove the pod from the API) the user must now provide the `--force` flag in addition to `--grace-period=0`.  This prevents users from accidentally force deleting pods without being aware of the consequences of force deletion.  Force deleting pods for resources like StatefulSets can result in multiple pods with the same name having running processes in the cluster, which may lead to data corruption or data inconsistency when using shared storage or common API endpoints. ([#35484](https://github.com/kubernetes/kubernetes/pull/35484), [@smarterclayton](https://github.com/smarterclayton))
* NPD: Add e2e test for NPD v0.2. ([#35740](https://github.com/kubernetes/kubernetes/pull/35740), [@Random-Liu](https://github.com/Random-Liu))
* DELETE requests can now pass in their DeleteOptions as a query parameter or a body parameter, rather than just as a body parameter. ([#35806](https://github.com/kubernetes/kubernetes/pull/35806), [@bdbauer](https://github.com/bdbauer))
* make using service account credentials from controllers optional ([#35970](https://github.com/kubernetes/kubernetes/pull/35970), [@deads2k](https://github.com/deads2k))
* AWS: strong-typing for k8s vs aws volume ids ([#35883](https://github.com/kubernetes/kubernetes/pull/35883), [@justinsb](https://github.com/justinsb))
* Controller changes for perma failed deployments ([#35691](https://github.com/kubernetes/kubernetes/pull/35691), [@kargakis](https://github.com/kargakis))
* Proxy min sync period ([#35334](https://github.com/kubernetes/kubernetes/pull/35334), [@timothysc](https://github.com/timothysc))
* Federated ConfigMap controller ([#35635](https://github.com/kubernetes/kubernetes/pull/35635), [@mwielgus](https://github.com/mwielgus))
* have basic kubectl crud agnostic of registered types ([#36085](https://github.com/kubernetes/kubernetes/pull/36085), [@deads2k](https://github.com/deads2k))
* Fix how we iterate over active jobs when removing them for Replace policy ([#36161](https://github.com/kubernetes/kubernetes/pull/36161), [@soltysh](https://github.com/soltysh))
* Adds TCPCloseWaitTimeout option to kube-proxy for sysctl nf_conntrack_tcp_timeout_time_wait ([#35919](https://github.com/kubernetes/kubernetes/pull/35919), [@bowei](https://github.com/bowei))
* Pods that are terminating due to eviction by the nodecontroller (typically due to unresponsive kubelet, or network partition) now surface in `kubectl get` output  ([#36017](https://github.com/kubernetes/kubernetes/pull/36017), [@foxish](https://github.com/foxish))
    * as being in state "Unknown", along with a longer description in `kubectl describe` output.
* The hostname of the node (as autodetected by the kubelet, specified via --hostname-override, or determined by the cloudprovider) is now recorded as an address of type "Hostname" in the status of the Node API object. The hostname is expected to be resolveable from the apiserver. ([#25532](https://github.com/kubernetes/kubernetes/pull/25532), [@mkulke](https://github.com/mkulke))
* [Kubelet] Add alpha support for `--cgroups-per-qos` using the configured `--cgroup-driver`. Disabled by default. ([#31546](https://github.com/kubernetes/kubernetes/pull/31546), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Move Statefulset (previously PetSet) to v1beta1 ([#35731](https://github.com/kubernetes/kubernetes/pull/35731), [@janetkuo](https://github.com/janetkuo))
* The error handling behavior of `pkg/client/restclient.Result` has changed.  Calls to `Result.Raw()` will no longer parse the body, although they will still return errors that react to `pkg/api/errors.Is*()` as in previous releases.  Callers of `Get()` and `Into()` will continue to receive errors that are parsed from the body if the kind and apiVersion of the body match the `Status` object. ([#36001](https://github.com/kubernetes/kubernetes/pull/36001), [@smarterclayton](https://github.com/smarterclayton))
    * This more closely aligns rest client as a generic RESTful client, while preserving the special Kube API extended error handling for the `Get` and `Into` methods (which most Kube clients use).
* Making the pod.alpha.kubernetes.io/initialized annotation optional in PetSet pods ([#35739](https://github.com/kubernetes/kubernetes/pull/35739), [@foxish](https://github.com/foxish))
* AWS: recognize us-east-2 region ([#35013](https://github.com/kubernetes/kubernetes/pull/35013), [@justinsb](https://github.com/justinsb))
* Eviction manager evicts based on inode consumption ([#35137](https://github.com/kubernetes/kubernetes/pull/35137), [@dashpole](https://github.com/dashpole))
* SELinux Overhaul ([#33663](https://github.com/kubernetes/kubernetes/pull/33663), [@pmorie](https://github.com/pmorie))
* Add SNI support to the apiserver ([#35109](https://github.com/kubernetes/kubernetes/pull/35109), [@sttts](https://github.com/sttts))
* The main kubernetes repository stops hosting archived version of released clients. Please use [client-go](https://github.com/kubernetes/client-go). ([#35928](https://github.com/kubernetes/kubernetes/pull/35928), [@caesarxuchao](https://github.com/caesarxuchao))
* Correct the article in generated documents ([#32557](https://github.com/kubernetes/kubernetes/pull/32557), [@asalkeld](https://github.com/asalkeld))
* Update PodAntiAffinity to ignore calls to subresources ([#35608](https://github.com/kubernetes/kubernetes/pull/35608), [@soltysh](https://github.com/soltysh))
* The apiserver can now select which type of kubelet-reported address to use for apiserver->node communications, using the --kubelet-preferred-address-types flag. ([#35497](https://github.com/kubernetes/kubernetes/pull/35497), [@liggitt](https://github.com/liggitt))
* update list of vailable resources ([#32687](https://github.com/kubernetes/kubernetes/pull/32687), [@jouve](https://github.com/jouve))
* Remove stale volumes if endpoint/svc creation fails. ([#35285](https://github.com/kubernetes/kubernetes/pull/35285), [@humblec](https://github.com/humblec))
* add kubectl cp ([#34914](https://github.com/kubernetes/kubernetes/pull/34914), [@brendandburns](https://github.com/brendandburns))
* Remove Job also from .status.active for Replace strategy ([#35420](https://github.com/kubernetes/kubernetes/pull/35420), [@soltysh](https://github.com/soltysh))
* Let release_1_5 clientset include multiple versions of a group ([#35471](https://github.com/kubernetes/kubernetes/pull/35471), [@caesarxuchao](https://github.com/caesarxuchao))
* support editing before creating resource ([#33250](https://github.com/kubernetes/kubernetes/pull/33250), [@ymqytw](https://github.com/ymqytw))
* allow authentication through a front-proxy ([#35452](https://github.com/kubernetes/kubernetes/pull/35452), [@deads2k](https://github.com/deads2k))
* On GCI, cleanup kubelet startup ([#35319](https://github.com/kubernetes/kubernetes/pull/35319), [@vishh](https://github.com/vishh))
* Add a retry when reading a file content from a container ([#35560](https://github.com/kubernetes/kubernetes/pull/35560), [@jingxu97](https://github.com/jingxu97))
* Fix cadvisor_unsupported and the crossbuild ([#35817](https://github.com/kubernetes/kubernetes/pull/35817), [@luxas](https://github.com/luxas))
* [PHASE 1] Opaque integer resource accounting. ([#31652](https://github.com/kubernetes/kubernetes/pull/31652), [@ConnorDoyle](https://github.com/ConnorDoyle))
* Add sync state loop in master's volume reconciler ([#34859](https://github.com/kubernetes/kubernetes/pull/34859), [@jingxu97](https://github.com/jingxu97))
* Bump GCE debian image to container-vm-v20161025 (CVE-2016-5195 Dirty… ([#35825](https://github.com/kubernetes/kubernetes/pull/35825), [@dchen1107](https://github.com/dchen1107))
* GC pod ips ([#35572](https://github.com/kubernetes/kubernetes/pull/35572), [@bprashanth](https://github.com/bprashanth))
* Stop including arch-specific binaries in kubernetes.tar.gz ([#35737](https://github.com/kubernetes/kubernetes/pull/35737), [@ixdy](https://github.com/ixdy))
* Rename PetSet to StatefulSet ([#35663](https://github.com/kubernetes/kubernetes/pull/35663), [@janetkuo](https://github.com/janetkuo))
* Enable containerized storage plugins mounter on GCI ([#35350](https://github.com/kubernetes/kubernetes/pull/35350), [@vishh](https://github.com/vishh))
* Bump container-vm version in config-test.sh ([#35705](https://github.com/kubernetes/kubernetes/pull/35705), [@mtaufen](https://github.com/mtaufen))
* Cadvisor root path configuration ([#35136](https://github.com/kubernetes/kubernetes/pull/35136), [@dashpole](https://github.com/dashpole))
* ssh pubkey parsing: prevent segfault ([#35323](https://github.com/kubernetes/kubernetes/pull/35323), [@mikkeloscar](https://github.com/mikkeloscar))



# v1.4.6

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads for v1.4.6


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes.tar.gz) | `6f8242aa29493e1f824997748419e4a287c28b06ed13f17b1ba94bf07fdfa3be`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-src.tar.gz) | `a2a2d885d246300b52adb5d7e1471b382c77d90a816618518c2a6e9941208e40`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-darwin-386.tar.gz) | `4db6349c976f893d0000dcb5b2ab09327824d0c38b3beab961711a0951cdfc82`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-darwin-amd64.tar.gz) | `2d31dea858569f518410effb20d3c3b9a6798d706dacbafd85f1f67f9ccbe288`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-linux-386.tar.gz) | `7980cf6132a7a6bf3816b8fd60d7bc1c9cb447d45196c31312b9d73567010909`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-linux-amd64.tar.gz) | `95b3cbd339f7d104d5b69b08d53060bfc78bd4ee7a94ede7ba4c0a76b615f8b1`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-linux-arm64.tar.gz) | `0f03cff262b0f4cc218b0f79294b4cbd8f92146c31137c75a27012d956864c79`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-linux-arm.tar.gz) | `f8c76fe8c41a5084cc1a1ab3e08d7e2d815f7baedfadac0dc6f9157ed2c607c9`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-windows-386.tar.gz) | `c29b3c8c8a72246852db048e922ad2221f35e1c309571f73fd9f3d9b01be5f79`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-client-windows-amd64.tar.gz) | `95bf20bdbe354476bbd3647adf72985698ded53a59819baa8268b5811e19f952`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-server-linux-amd64.tar.gz) | `f0a60c45f3360696431288826e56df3b8c18c1dc6fc3f0ea83409f970395e38f`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-server-linux-arm64.tar.gz) | `8c667d4792fcfee821a2041e5d0356e1abc2b3fa6fe7b69c5479e48c858ba29c`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.6/kubernetes-server-linux-arm.tar.gz) | `c57246d484b5f98d6aa16591f2b4c4c1a01ebbc7be05bce8690a4f3b88582844`

## Changelog since v1.4.5

### Other notable changes

* Fix issue in reconstruct volume data when kubelet restarts ([#36616](https://github.com/kubernetes/kubernetes/pull/36616), [@jingxu97](https://github.com/jingxu97))
* Add sync state loop in master's volume reconciler ([#34859](https://github.com/kubernetes/kubernetes/pull/34859), [@jingxu97](https://github.com/jingxu97))
* AWS: strong-typing for k8s vs aws volume ids ([#35883](https://github.com/kubernetes/kubernetes/pull/35883), [@justinsb](https://github.com/justinsb))
* Bump GCI version to gci-beta-55-8872-47-0 ([#36679](https://github.com/kubernetes/kubernetes/pull/36679), [@mtaufen](https://github.com/mtaufen))

```
  gci-beta-55-8872-47-0:
  Date:           Nov 11, 2016
  Kernel:         ChromiumOS-4.4
  Kubernetes:     v1.4.5
  Docker:         v1.11.2
  Changelog (vs 55-8872-18-0)
    * Cherry-pick runc PR#608: Eliminate redundant parsing of mountinfo
    * Updated kubernetes to v1.4.5
    * Fixed a bug in e2fsprogs that caused mke2fs to take a very long time. Upstream fix: http://git.kernel.org/cgit/fs/ext2/e2fsprogs.git/commit/?h=next&id=d33e690fe7a6cbeb51349d9f2c7fb16a6ebec9c2 
```

* Fix fetching pids running in a cgroup, which caused problems with OOM score adjustments & setting the /system cgroup ("misc" in the summary API). ([#36614](https://github.com/kubernetes/kubernetes/pull/36614), [@timstclair](https://github.com/timstclair))
* DELETE requests can now pass in their DeleteOptions as a query parameter or a body parameter, rather than just as a body parameter. ([#35806](https://github.com/kubernetes/kubernetes/pull/35806), [@bdbauer](https://github.com/bdbauer))
* rkt: Convert image name to be a valid acidentifier ([#34375](https://github.com/kubernetes/kubernetes/pull/34375), [@euank](https://github.com/euank))
* Remove stale volumes if endpoint/svc creation fails. ([#35285](https://github.com/kubernetes/kubernetes/pull/35285), [@humblec](https://github.com/humblec))
* Remove Job also from .status.active for Replace strategy ([#35420](https://github.com/kubernetes/kubernetes/pull/35420), [@soltysh](https://github.com/soltysh))
* Update PodAntiAffinity to ignore calls to subresources ([#35608](https://github.com/kubernetes/kubernetes/pull/35608), [@soltysh](https://github.com/soltysh))
* Adds TCPCloseWaitTimeout option to kube-proxy for sysctl nf_conntrack_tcp_timeout_time_wait ([#35919](https://github.com/kubernetes/kubernetes/pull/35919), [@bowei](https://github.com/bowei))
* Fix how we iterate over active jobs when removing them for Replace policy ([#36161](https://github.com/kubernetes/kubernetes/pull/36161), [@soltysh](https://github.com/soltysh))
* Bump GCI version to latest m55 version in GCE for K8s 1.4 ([#36302](https://github.com/kubernetes/kubernetes/pull/36302), [@mtaufen](https://github.com/mtaufen))
* Add a check for file size if the reading content returns empty ([#33976](https://github.com/kubernetes/kubernetes/pull/33976), [@jingxu97](https://github.com/jingxu97))
* Add a retry when reading a file content from a container ([#35560](https://github.com/kubernetes/kubernetes/pull/35560), [@jingxu97](https://github.com/jingxu97))
* Skip CLOSE_WAIT e2e test if server is 1.4.5 ([#36404](https://github.com/kubernetes/kubernetes/pull/36404), [@bowei](https://github.com/bowei))
* Adds etcd3 changes ([#36232](https://github.com/kubernetes/kubernetes/pull/36232), [@wojtek-t](https://github.com/wojtek-t))
* Adds TCPCloseWaitTimeout option to kube-proxy for sysctl nf_conntrack_tcp_timeout_time_wait ([#36099](https://github.com/kubernetes/kubernetes/pull/36099), [@bowei](https://github.com/bowei))



# v1.3.10

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads for v1.3.10


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes.tar.gz) | `0f61517fbab1feafbe1024da0b88bfe16e61fed7e612285d70e3ecb53ce518cf`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-src.tar.gz) | `7b1be0dcc12ae1b0cb1928b770c1025755fd0858ce7520907bacda19e5bfa53f`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-darwin-386.tar.gz) | `64a7012411a506ff7825e7b9c64b50197917d6f4e1128ea0e7b30a121059da47`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-darwin-amd64.tar.gz) | `5d85843e643eaebe3e34e48810f4786430b5ecce915144e01ba2d8539aa77364`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-linux-386.tar.gz) | `06d478c601b1d4aa1fc539e9120adbcbbd2fb370d062516f84a064e465d8eadc`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-linux-amd64.tar.gz) | `fe571542482b8ba3ff94b9e5e9657f6ab4fc0feb8971930dc80b7ae2548d669b`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-linux-arm64.tar.gz) | `176b52d35150ca9f08a7e90e33e2839b7574afe350edf4fafa46745d77bb5aa4`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-linux-arm.tar.gz) | `1c3bf4ac1e4eb0e02f785db725efd490beaf06c8acd26d694971ba510b60a94d`
[kubernetes-client-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-linux-ppc64le.tar.gz) | `172cd0af71fcba7c51e9476732dbe86ba251c03b1d74f912111e4e755be540ce`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-windows-386.tar.gz) | `f2d2f82d7e285c98d8cc58a8a6e13a1122c9f60bb2c73e4cefe3555f963e56cd`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-client-windows-amd64.tar.gz) | `ac0aa2b09dfeb8001e76f3aefe82c7bd2fda5bd0ef744ac3aed966b99c8dc8e5`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-server-linux-amd64.tar.gz) | `bf0d3924ff84c95c316fcb4b21876cc019bd648ca8ab87fd6b2712ccda30992b`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-server-linux-arm64.tar.gz) | `45e88d1c8edc17d7f1deab8d040a769d8647203c465d76763abb1ce445a98773`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-server-linux-arm.tar.gz) | `40ac46a265021615637f07d532cd563b4256dcf340a27c594bfd3501fe66b84c`
[kubernetes-server-linux-ppc64le.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.10/kubernetes-server-linux-ppc64le.tar.gz) | `faa5075ab3e6688666bbbb274fa55a825513ee082a3b17bcddb5b8f4fd6f9aa0`

## Changelog since v1.3.9

### Other notable changes

* gci: decouple from the built-in kubelet version ([#31367](https://github.com/kubernetes/kubernetes/pull/31367), [@Amey-D](https://github.com/Amey-D))
* Bump GCE debian image to container-vm-v20161025 (CVE-2016-5195 Dirty… ([#35825](https://github.com/kubernetes/kubernetes/pull/35825), [@dchen1107](https://github.com/dchen1107))
* Add RELEASE_INFRA_PUSH related code to support pushes from kubernetes/release. ([#28922](https://github.com/kubernetes/kubernetes/pull/28922), [@david-mcmahon](https://github.com/david-mcmahon))



# v1.4.5

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads for v1.4.5


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes.tar.gz) | `339f4d1c7a374ddb32334268c4af8dae0b86d1567a9c812087d672a7defe233c`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-src.tar.gz) | `69b1b022400794d491200a9365ea9bf735567348d0299920462cf7167c76ba61`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-darwin-386.tar.gz) | `6012dab54687f7eb41ce9cd6b4676e15b774fbfbeadb7e00c806ba3f63fe10ce`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-darwin-amd64.tar.gz) | `981b321f4393fc9892c6558321e1d8ee6d8256b85f09266c8794fdcee9cb1c07`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-linux-386.tar.gz) | `75ce408ef9f4b277718701c025955cd628eeee4180d8e9e7fd8ecf008878429f`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-linux-amd64.tar.gz) | `0c0768d7646cec490ca1e47a4e2f519724fc75d984d411aa92fe17a82356532b`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-linux-arm64.tar.gz) | `910a6465b1ecbf1aae8f6cd16e35ac7ad7b0e598557941937d02d16520e2e37c`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-linux-arm.tar.gz) | `29644cca627cdce6c7aad057d9680eee87d21b1bbd6af02f7277f24eccbc95f7`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-windows-386.tar.gz) | `dc249cc0f6cbb0e0705f7b43929461b6702ae91148218da070bb99e8a8f6f108`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-client-windows-amd64.tar.gz) | `d60d275ad5f45ebe83a458912de96fd8381540d4bcf91023fe2173af6acd535b`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-server-linux-amd64.tar.gz) | `25e12aaf3f93c320f6aa640bb1430d4c0e99e3b0e83bcef660d2a513bdef2c20`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-server-linux-arm64.tar.gz) | `e768146c9476b96f092409030349b4c5bb9682287567fe2732888ad5ed1d3ede`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.5/kubernetes-server-linux-arm.tar.gz) | `26581dc0fc31542c831a588baad9ad391598e5b2ff299a0fc92a2c04990b3edd`

## Changelog since v1.4.4

### Other notable changes

* Fix volume states out of sync problem after kubelet restarts ([#33616](https://github.com/kubernetes/kubernetes/pull/33616), [@jingxu97](https://github.com/jingxu97))
* cross: add IsNotMountPoint() to mount_unsupported.go ([#35566](https://github.com/kubernetes/kubernetes/pull/35566), [@rootfs](https://github.com/rootfs))
* Bump GCE debian image to container-vm-v20161025 (CVE-2016-5195 Dirty COW) ([#35825](https://github.com/kubernetes/kubernetes/pull/35825), [@dchen1107](https://github.com/dchen1107))
* Avoid overriding system and kubelet cgroups on GCI ([#35319](https://github.com/kubernetes/kubernetes/pull/35319), [@vishh](https://github.com/vishh))
        * Make the kubectl from k8s release the default on GCI
* kubelet summary rootfs now refers to the filesystem that contains the Kubelet RootDirectory (var/lib/kubelet) instead of cadvisor's rootfs ( / ), since they may be different filesystems. ([#35136](https://github.com/kubernetes/kubernetes/pull/35136), [@dashpole](https://github.com/dashpole))
* Fix cadvisor_unsupported and the crossbuild ([#35817](https://github.com/kubernetes/kubernetes/pull/35817), [@luxas](https://github.com/luxas))
* kubenet: SyncHostports for both running and ready to run pods. ([#31388](https://github.com/kubernetes/kubernetes/pull/31388), [@yifan-gu](https://github.com/yifan-gu))
* GC pod ips ([#35572](https://github.com/kubernetes/kubernetes/pull/35572), [@bprashanth](https://github.com/bprashanth))
* Fix version string generation for local version different from release and not based on `-alpha.no` or `-beta.no` suffixed tag. ([#34612](https://github.com/kubernetes/kubernetes/pull/34612), [@jellonek](https://github.com/jellonek))
* Node status updater should SetNodeStatusUpdateNeeded if it fails to update status ([#34368](https://github.com/kubernetes/kubernetes/pull/34368), [@jingxu97](https://github.com/jingxu97))
* Fixed flakes caused by petset tests. ([#35158](https://github.com/kubernetes/kubernetes/pull/35158), [@foxish](https://github.com/foxish))
* Added rkt binary to GCI ([#35321](https://github.com/kubernetes/kubernetes/pull/35321), [@vishh](https://github.com/vishh))
* Bump container-vm version in config-test.sh ([#35705](https://github.com/kubernetes/kubernetes/pull/35705), [@mtaufen](https://github.com/mtaufen))
* Delete all firewall rules (and optionally network) on GCE/GKE cluster teardown ([#34577](https://github.com/kubernetes/kubernetes/pull/34577), [@ixdy](https://github.com/ixdy))
* Fixed mutation warning in Attach/Detach controller ([#35273](https://github.com/kubernetes/kubernetes/pull/35273), [@jsafrane](https://github.com/jsafrane))
* Dynamic provisioning for vSphere ([#30836](https://github.com/kubernetes/kubernetes/pull/30836), [@abrarshivani](https://github.com/abrarshivani))
* Update grafana version used by default in kubernetes to 3.1.1 ([#35435](https://github.com/kubernetes/kubernetes/pull/35435), [@Crassirostris](https://github.com/Crassirostris))
* vSphere Kube-up: resolve vm-names on all nodes ([#35365](https://github.com/kubernetes/kubernetes/pull/35365), [@kerneltime](https://github.com/kerneltime))
* Improve source IP preservation test, fail the test instead of panic. ([#34030](https://github.com/kubernetes/kubernetes/pull/34030), [@MrHohn](https://github.com/MrHohn))
* Fix [#31085](https://github.com/kubernetes/kubernetes/pull/31085), include output checking in retry loop ([#34107](https://github.com/kubernetes/kubernetes/pull/34107), [@MrHohn](https://github.com/MrHohn))
* vSphere kube-up: Wait for cbr0 configuration to complete before setting up routes. ([#35232](https://github.com/kubernetes/kubernetes/pull/35232), [@kerneltime](https://github.com/kerneltime))
* Substitute gcloud regex with regexp ([#35346](https://github.com/kubernetes/kubernetes/pull/35346), [@bprashanth](https://github.com/bprashanth))
* Fix PDB e2e test, off-by-one ([#35274](https://github.com/kubernetes/kubernetes/pull/35274), [@soltysh](https://github.com/soltysh))
* etcd3: API storage - decouple decorator from filter ([#31189](https://github.com/kubernetes/kubernetes/pull/31189), [@hongchaodeng](https://github.com/hongchaodeng))
* etcd3: v3client + grpc client leak fix ([#31704](https://github.com/kubernetes/kubernetes/pull/31704), [@timothysc](https://github.com/timothysc))
* etcd3: watcher logging error ([#32831](https://github.com/kubernetes/kubernetes/pull/32831), [@hongchaodeng](https://github.com/hongchaodeng))
* etcd: watcher centralize error handling ([#32907](https://github.com/kubernetes/kubernetes/pull/32907), [@hongchaodeng](https://github.com/hongchaodeng))
* etcd: stop watcher when watch channel is closed ([#33003](https://github.com/kubernetes/kubernetes/pull/33003), [@hongchaodeng](https://github.com/hongchaodeng))
* etcd3: dereference the UID pointer for a readable error message. ([#33349](https://github.com/kubernetes/kubernetes/pull/33349), [@madhusudancs](https://github.com/madhusudancs))
* etcd3: pass SelectionPredicate instead of Filter to storage layer ([#31190](https://github.com/kubernetes/kubernetes/pull/31190), [@hongchaodeng](https://github.com/hongchaodeng))
* etcd3: make gets for previous value in watch serialize-able ([#34089](https://github.com/kubernetes/kubernetes/pull/34089), [@wojtek-t](https://github.com/wojtek-t))
* etcd3: minor cleanups ([#34234](https://github.com/kubernetes/kubernetes/pull/34234), [@wojtek-t](https://github.com/wojtek-t))
* etcd3: update etcd godep to 3.0.9 to address TestWatch issues ([#32822](https://github.com/kubernetes/kubernetes/pull/32822), [@timothysc](https://github.com/timothysc))
* etcd3: update to etcd 3.0.10 ([#33393](https://github.com/kubernetes/kubernetes/pull/33393), [@timothysc](https://github.com/timothysc))
* etcd3: use PrevKV to remove additional get ([#34246](https://github.com/kubernetes/kubernetes/pull/34246), [@hongchaodeng](https://github.com/hongchaodeng))
* etcd3: avoid unnecessary decoding in etcd3 client  ([#34435](https://github.com/kubernetes/kubernetes/pull/34435), [@wojtek-t](https://github.com/wojtek-t))
* etcd3: fix suite ([#32477](https://github.com/kubernetes/kubernetes/pull/32477), [@wojtek-t](https://github.com/wojtek-t))



# v1.5.0-alpha.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads for v1.5.0-alpha.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes.tar.gz) | `77f04c646657b683210a17aeca62e56bf985702b267942b41729406970c40cee`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-src.tar.gz) | `f6090cc853e56159099bf12169f0d84e29fd2c055b0c7dbdac755ee94439a6a6`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `917adbc70156d55371c1aea62279a521e930e7ff130728aa176505f0268182e3`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `9c8084eeab05b6db0508f789cb8a05b4f864ee23ea37b43e17af0026fb67defa`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `3498f9cd73bb947b7cd8c4e5fb3ebe0676fbc98cf346a807f1b7c252aa068d68`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `e9bf2e48212bb275b113d0a1f6091c4692126c8af3c4e0a986e483ec27190e82`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `9c514a482d4dd44d64f3d47eb3d64b434343f10abdecf1b5176ff0078d3b7008`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `c51a8ebc2c3ca2f914042a6017852feb315fd3ceba8b0d5186349b553da11fdb`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `32b006e1f9e6c14fe399806bb82ec4bf8658ab9828753d1b14732bb8dbb72062`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `1e142f1fe76bdd660b4f1be51eef4e51705585fccb94e674a7d891ffe8c3b4e3`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `4a3b550a1ede8bebd14413a37e3fc10c8403a3e3fbbce096de443351d076817a`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `00e58bb04bf150c554f28d8fd2f72fbdd1e7918999aaea9c88c91c8f71946ffe`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `6837ff73249c0f3e7ba2d7c00321274db0f97b5cd0b4dc58d5cc3a2119e1c820`

## Changelog since v1.5.0-alpha.1

### Action Required

* Deprecate the --reconcile-cidr kubelet flag because it has no function anymore ([#35523](https://github.com/kubernetes/kubernetes/pull/35523), [@luxas](https://github.com/luxas))
* Removed the deprecated kubelet --configure-cbr0 flag, and with that the "classic" networking mode as well ([#34906](https://github.com/kubernetes/kubernetes/pull/34906), [@luxas](https://github.com/luxas))
* New client-go structure ([#34989](https://github.com/kubernetes/kubernetes/pull/34989), [@caesarxuchao](https://github.com/caesarxuchao))
* Remove scheduler flags that were marked as deprecated 2+ releases ago. ([#34471](https://github.com/kubernetes/kubernetes/pull/34471), [@timothysc](https://github.com/timothysc))

### Other notable changes

* Make the fake RESTClient usable by all the API groups, not just core. ([#35492](https://github.com/kubernetes/kubernetes/pull/35492), [@madhusudancs](https://github.com/madhusudancs))
* Adding support for DeleteOptions.OrphanDependents for federated namespaces. Setting it to false while deleting a federated namespace also deletes the corresponding namespace from all registered clusters. ([#34648](https://github.com/kubernetes/kubernetes/pull/34648), [@nikhiljindal](https://github.com/nikhiljindal))
* Kubelet flag '--mounter-path' renamed to '--experimental-mounter-path' ([#35646](https://github.com/kubernetes/kubernetes/pull/35646), [@vishh](https://github.com/vishh))
* Node status updater should SetNodeStatusUpdateNeeded if it fails to update status ([#34368](https://github.com/kubernetes/kubernetes/pull/34368), [@jingxu97](https://github.com/jingxu97))
* Deprecate OpenAPI spec for GroupVersion endpoints in favor of single spec /swagger.json ([#35388](https://github.com/kubernetes/kubernetes/pull/35388), [@mbohlool](https://github.com/mbohlool))
* kubelet authn/authz ([#34381](https://github.com/kubernetes/kubernetes/pull/34381), [@liggitt](https://github.com/liggitt))
* Fix volume states out of sync problem after kubelet restarts ([#33616](https://github.com/kubernetes/kubernetes/pull/33616), [@jingxu97](https://github.com/jingxu97))
* Added rkt binary to GCI ([#35321](https://github.com/kubernetes/kubernetes/pull/35321), [@vishh](https://github.com/vishh))
* Fixed mutation warning in Attach/Detach controller ([#35273](https://github.com/kubernetes/kubernetes/pull/35273), [@jsafrane](https://github.com/jsafrane))
* Don't count failed pods as "not-ready" ([#35404](https://github.com/kubernetes/kubernetes/pull/35404), [@brendandburns](https://github.com/brendandburns))
* fixed typo in script which made setting custom cidr in gce using kube-up impossible ([#35267](https://github.com/kubernetes/kubernetes/pull/35267), [@tommywo](https://github.com/tommywo))
* The podGC controller will now always run, irrespective of the value supplied to the "terminated-pod-gc-threshold" flag supplied to the controller manager.  ([#35476](https://github.com/kubernetes/kubernetes/pull/35476), [@foxish](https://github.com/foxish))
    * The specific behavior of the podGC controller to clean up terminated pods is still governed by the flag, but the podGC's responsibilities have evolved beyond just cleaning up terminated pods.
* Update grafana version used by default in kubernetes to 3.1.1 ([#35435](https://github.com/kubernetes/kubernetes/pull/35435), [@Crassirostris](https://github.com/Crassirostris))
* vSphere Kube-up: resolve vm-names on all nodes ([#35365](https://github.com/kubernetes/kubernetes/pull/35365), [@kerneltime](https://github.com/kerneltime))
* bootstrap: Start hostNetwork pods even if network plugin not ready ([#33347](https://github.com/kubernetes/kubernetes/pull/33347), [@justinsb](https://github.com/justinsb))
* Factor out post-init swagger and OpenAPI routes ([#32590](https://github.com/kubernetes/kubernetes/pull/32590), [@sttts](https://github.com/sttts))
* Substitute gcloud regex with regexp ([#35346](https://github.com/kubernetes/kubernetes/pull/35346), [@bprashanth](https://github.com/bprashanth))
* Remove support for multi-architecture code in `kubeadm`, which was released untested. ([#35124](https://github.com/kubernetes/kubernetes/pull/35124), [@errordeveloper](https://github.com/errordeveloper))
* vSphere kube-up: Wait for cbr0 configuration to complete before setting up routes. ([#35232](https://github.com/kubernetes/kubernetes/pull/35232), [@kerneltime](https://github.com/kerneltime))
* Remove last probe time from replica sets ([#35199](https://github.com/kubernetes/kubernetes/pull/35199), [@kargakis](https://github.com/kargakis))
* Update the GCI image to gci-dev-55-8872-18-0 ([#35243](https://github.com/kubernetes/kubernetes/pull/35243), [@maisem](https://github.com/maisem))
* Add `--mounter-path` flag to kubelet that will allow overriding the `mount` command used by kubelet ([#34994](https://github.com/kubernetes/kubernetes/pull/34994), [@jingxu97](https://github.com/jingxu97))
* Fix a bug under the rkt runtime whereby image-registries with ports would not be fetched from ([#34375](https://github.com/kubernetes/kubernetes/pull/34375), [@euank](https://github.com/euank))
* Updated default Elasticsearch and Kibana used for elasticsearch logging destination to versions 2.4.1 and 4.6.1 respectively. ([#34969](https://github.com/kubernetes/kubernetes/pull/34969), [@Crassirostris](https://github.com/Crassirostris))
* Loadbalanced client src ip preservation enters beta ([#33957](https://github.com/kubernetes/kubernetes/pull/33957), [@bprashanth](https://github.com/bprashanth))
* Add NodePort value in kubectl output ([#34922](https://github.com/kubernetes/kubernetes/pull/34922), [@zreigz](https://github.com/zreigz))
* kubectl drain now waits until pods have been delete from the Node before exiting ([#34778](https://github.com/kubernetes/kubernetes/pull/34778), [@ymqytw](https://github.com/ymqytw))
* Don't report FS stats for system containers in the Kubelet Summary API ([#34998](https://github.com/kubernetes/kubernetes/pull/34998), [@timstclair](https://github.com/timstclair))
* Fixed flakes caused by petset tests. ([#35158](https://github.com/kubernetes/kubernetes/pull/35158), [@foxish](https://github.com/foxish))
* Add validation that detects repeated keys in the labels and annotations maps ([#34407](https://github.com/kubernetes/kubernetes/pull/34407), [@brendandburns](https://github.com/brendandburns))
* Change merge key for VolumeMount to mountPath ([#35071](https://github.com/kubernetes/kubernetes/pull/35071), [@thockin](https://github.com/thockin))
* kubelet: storage: don't hang kubelet on unresponsive nfs ([#35038](https://github.com/kubernetes/kubernetes/pull/35038), [@sjenning](https://github.com/sjenning))
* Fix kube vsphere.kerneltime ([#34997](https://github.com/kubernetes/kubernetes/pull/34997), [@kerneltime](https://github.com/kerneltime))
* Add PSP support for seccomp profiles ([#28300](https://github.com/kubernetes/kubernetes/pull/28300), [@pweil-](https://github.com/pweil-))
* Updated Go to 1.7 ([#28742](https://github.com/kubernetes/kubernetes/pull/28742), [@jessfraz](https://github.com/jessfraz))
* HPA: fixed wrong count for target replicas calculations ([#34821](https://github.com/kubernetes/kubernetes/pull/34821)). ([#34955](https://github.com/kubernetes/kubernetes/pull/34955), [@jszczepkowski](https://github.com/jszczepkowski))
* Improves how 'kubectl' uses the terminal size when printing help and usage. ([#34502](https://github.com/kubernetes/kubernetes/pull/34502), [@fabianofranz](https://github.com/fabianofranz))
* Updated Elasticsearch image from version 1.5.1 to version 2.4.1. Updated Kibana image from version 4.0.2 to version 4.6.1. ([#34562](https://github.com/kubernetes/kubernetes/pull/34562), [@Crassirostris](https://github.com/Crassirostris))
* libvirt-coreos: Download the coreos_production_qemu_image over SSL. ([#34646](https://github.com/kubernetes/kubernetes/pull/34646), [@roberthbailey](https://github.com/roberthbailey))
* Add a new global option "--request-timeout" to the `kubectl` client ([#33958](https://github.com/kubernetes/kubernetes/pull/33958), [@juanvallejo](https://github.com/juanvallejo))
* Add support for admission controller based on namespace node selectors. ([#24980](https://github.com/kubernetes/kubernetes/pull/24980), [@aveshagarwal](https://github.com/aveshagarwal))
* Add 'kubectl set resources' ([#27206](https://github.com/kubernetes/kubernetes/pull/27206), [@JacobTanenbaum](https://github.com/JacobTanenbaum))
* Support trust id as a scope in the OpenStack authentication logic ([#32111](https://github.com/kubernetes/kubernetes/pull/32111), [@MatMaul](https://github.com/MatMaul))
* Only wait for cache syncs once in NodeController ([#34851](https://github.com/kubernetes/kubernetes/pull/34851), [@ncdc](https://github.com/ncdc))
* NodeController waits for informer sync before doing anything ([#34809](https://github.com/kubernetes/kubernetes/pull/34809), [@gmarek](https://github.com/gmarek))
* azure: lower log priority for skipped nic update message ([#34730](https://github.com/kubernetes/kubernetes/pull/34730), [@colemickens](https://github.com/colemickens))
* Security Group support for OpenStack Load Balancers ([#31921](https://github.com/kubernetes/kubernetes/pull/31921), [@grahamhayes](https://github.com/grahamhayes))
* Make NodeController recognize deletion tombstones ([#34786](https://github.com/kubernetes/kubernetes/pull/34786), [@davidopp](https://github.com/davidopp))
* Delete all firewall rules (and optionally network) on GCE/GKE cluster teardown ([#34577](https://github.com/kubernetes/kubernetes/pull/34577), [@ixdy](https://github.com/ixdy))
* Fix panic in NodeController caused by receiving DeletedFinalStateUnknown object from the cache. ([#34694](https://github.com/kubernetes/kubernetes/pull/34694), [@gmarek](https://github.com/gmarek))
* azure: add PrimaryAvailabilitySet to config, only use nodes in that set in the loadbalancer pool ([#34526](https://github.com/kubernetes/kubernetes/pull/34526), [@colemickens](https://github.com/colemickens))
* Fix leaking ingress resources in federated ingress e2e test. ([#34652](https://github.com/kubernetes/kubernetes/pull/34652), [@quinton-hoole](https://github.com/quinton-hoole))
* pvc.Spec.Resources.Requests min and max can be enforced with a LimitRange of type "PersistentVolumeClaim" in the namespace ([#30145](https://github.com/kubernetes/kubernetes/pull/30145), [@markturansky](https://github.com/markturansky))
* Federated DaemonSet controller. Supports all the API that regular DaemonSet has. ([#34319](https://github.com/kubernetes/kubernetes/pull/34319), [@mwielgus](https://github.com/mwielgus))
* New federation deployment mechanism now allows non-GCP clusters. ([#34620](https://github.com/kubernetes/kubernetes/pull/34620), [@madhusudancs](https://github.com/madhusudancs))
        * Writes the federation kubeconfig to the local kubeconfig file.
* Update the series and the README to reflect the change. ([#30374](https://github.com/kubernetes/kubernetes/pull/30374), [@mbruzek](https://github.com/mbruzek))
* Replica set conditions API ([#33905](https://github.com/kubernetes/kubernetes/pull/33905), [@kargakis](https://github.com/kargakis))
* etcd3: avoid unnecessary decoding in etcd3 client  ([#34435](https://github.com/kubernetes/kubernetes/pull/34435), [@wojtek-t](https://github.com/wojtek-t))
* Test x509 intermediates correctly ([#34524](https://github.com/kubernetes/kubernetes/pull/34524), [@liggitt](https://github.com/liggitt))
* Add `cifs-utils` to the hyperkube image. ([#34416](https://github.com/kubernetes/kubernetes/pull/34416), [@colemickens](https://github.com/colemickens))
* etcd3: use PrevKV to remove additional get ([#34246](https://github.com/kubernetes/kubernetes/pull/34246), [@hongchaodeng](https://github.com/hongchaodeng))
* Fix upgrade.sh image setup ([#34468](https://github.com/kubernetes/kubernetes/pull/34468), [@mtaufen](https://github.com/mtaufen))



# v1.2.7

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads for v1.2.7


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes.tar.gz) | `53db157923c17fa7a0addb3e4dfe7d1b9194b9266a87d371a251d5bb790a1832`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-src.tar.gz) | `e6e46831706743d8263581d0575507cf5ffc265096d22e5e84cf1c3ae925db5e`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-client-darwin-386.tar.gz) | `8418767e45c62c2ef5f9b4479ed02af64e190ce07dcbafa1920e93e71f419c55`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-client-darwin-amd64.tar.gz) | `41d742c2c55e7686311978eaaddee3844b990a0fe49fa8597158bcb0ee4c05c9`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-client-linux-386.tar.gz) | `619e0a450cddf10ed1d42ed1d6330d41a75b9c1e00eb654cbe4b0422cd6099c5`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-client-linux-amd64.tar.gz) | `9a5fcd87514b88eb25173e574aef5b5343816c07ab5947d06787c9f12c40f54a`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-client-linux-arm.tar.gz) | `fd6e39b4a56e03448382825f27f4f30a2e981a8d20f4a8cedbd084bbb4577d42`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-client-windows-386.tar.gz) | `862625cb3d9445cff1b09e4ebcdb60dd93b5b2dc34bb6022d2eeed7c8d8bc5d8`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-client-windows-amd64.tar.gz) | `054337e41187e39950de93e4670bc78a95b6901cc2f95c50ff437d9825ae94c5`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-server-linux-amd64.tar.gz) | `fef041e9cbe5bcf8fd708f81ee2e2783429af1ab9cfb151d645ef9be96e19b73`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.7/kubernetes-server-linux-arm.tar.gz) | `ce02d7bcd75c31db4f7b9922c19ea2a3312b0ba579b0dcd96b279b661eca18a8`

## Changelog since v1.2.6

### Other notable changes

* Test x509 intermediates correctly ([#34524](https://github.com/kubernetes/kubernetes/pull/34524), [@liggitt](https://github.com/liggitt))



# v1.4.4

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads for v1.4.4


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes.tar.gz) | `2732bfc56ceabc872b6af3f460cbda68c2384c95a1c0c72eb33e5ff0e03dc9da`
[kubernetes-src.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-src.tar.gz) | `29c6cf1567e6b7f6c3ecb71acead083b7535b22ac20bd8166b29074e8a0f6441`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-darwin-386.tar.gz) | `e983b1837e4165e4bc8e361000468421f16dbd5ae90b0c49af6280dbcecf57b1`
[kubernetes-client-darwin-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-darwin-amd64.tar.gz) | `8c58231c8340e546336b70d86b6a76285b9f7a0c13b802b350b68610dfaedb35`
[kubernetes-client-linux-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-linux-386.tar.gz) | `33e5d2da52325367db08bcc80791cef2e21fdae176b496b063b3a37115f3f075`
[kubernetes-client-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-linux-amd64.tar.gz) | `5fd6215ef0673f5a8e385660cf233d67d26dd79568c69e2328b103fbf1bd752a`
[kubernetes-client-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-linux-arm64.tar.gz) | `2d6d0400cd59b042e2da074cbd3b13b9dc61da1dbba04468d67119294cf72435`
[kubernetes-client-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-linux-arm.tar.gz) | `ff99f26082a77e37caa66aa07ec56bfc7963e6ac782550be5090a8b158f7e89a`
[kubernetes-client-windows-386.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-windows-386.tar.gz) | `82e762727a8f607180a1e339e058cc9739ad55960d3517c5170bcd5b64179f13`
[kubernetes-client-windows-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-client-windows-amd64.tar.gz) | `4de735ba72c729589efbcd2b8fc4920786fffd96850173c13cbf469819d00808`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-server-linux-amd64.tar.gz) | `6d5ff37941328df33c0efc5876bb7b82722bc584f1976fe632915db7bf3f316a`
[kubernetes-server-linux-arm64.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-server-linux-arm64.tar.gz) | `6ec40848ea29c0982b89c746d716b0958438a6eb774aea20a5ef7885a7060aed`
[kubernetes-server-linux-arm.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.4/kubernetes-server-linux-arm.tar.gz) | `43d6a3260d73cfe652af2ffa7b7092444fe57429cb45e90eb99f0a70012ee033`

## Changelog since v1.4.3

### Other notable changes

* Update the GCI image to gci-dev-55-8872-18-0 ([#35243](https://github.com/kubernetes/kubernetes/pull/35243), [@maisem](https://github.com/maisem))
* Change merge key for VolumeMount to mountPath ([#35071](https://github.com/kubernetes/kubernetes/pull/35071), [@thockin](https://github.com/thockin))
* Turned-off etcd listening on public ports as potentially insecure. Removed ([#35192](https://github.com/kubernetes/kubernetes/pull/35192), [@jszczepkowski](https://github.com/jszczepkowski))
    * experimental support for master replication.
* Add support for vSphere Cloud Provider when deploying via kubeup on vSphere. ([#31467](https://github.com/kubernetes/kubernetes/pull/31467), [@kerneltime](https://github.com/kerneltime))
* Fix kube vsphere.kerneltime ([#34997](https://github.com/kubernetes/kubernetes/pull/34997), [@kerneltime](https://github.com/kerneltime))
* HPA: fixed wrong count for target replicas calculations ([#34821](https://github.com/kubernetes/kubernetes/pull/34821)). ([#34955](https://github.com/kubernetes/kubernetes/pull/34955), [@jszczepkowski](https://github.com/jszczepkowski))
* Fix leaking ingress resources in federated ingress e2e test. ([#34652](https://github.com/kubernetes/kubernetes/pull/34652), [@quinton-hoole](https://github.com/quinton-hoole))
* azure: add PrimaryAvailabilitySet to config, only use nodes in that set in the loadbalancer pool ([#34526](https://github.com/kubernetes/kubernetes/pull/34526), [@colemickens](https://github.com/colemickens))
* azure: lower log priority for skipped nic update message ([#34730](https://github.com/kubernetes/kubernetes/pull/34730), [@colemickens](https://github.com/colemickens))



# v1.3.9

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.9/kubernetes.tar.gz) | `a994c732d2b852bbee55a78601d50d046323021a99b0801aea07dacf64c2c59a`

## Changelog since v1.3.8

### Other notable changes

* Test x509 intermediates correctly ([#34524](https://github.com/kubernetes/kubernetes/pull/34524), [@liggitt](https://github.com/liggitt))
* Remove headers that are unnecessary for proxy target ([#34076](https://github.com/kubernetes/kubernetes/pull/34076), [@mbohlool](https://github.com/mbohlool))



# v1.4.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.3/kubernetes.tar.gz) | `c3dccccc005bc22eaf814ccb8e72b4f876167ab38ac594bb7e44c98f162a0f1c`

## Changelog since v1.4.2-beta.1

### Other notable changes

* Fix non-starting node controller in 1.4 branch ([#34895](https://github.com/kubernetes/kubernetes/pull/34895), [@wojtek-t](https://github.com/wojtek-t))
* Cherrypick [#34851](https://github.com/kubernetes/kubernetes/pull/34851) "Only wait for cache syncs once in NodeController" ([#34861](https://github.com/kubernetes/kubernetes/pull/34861), [@jessfraz](https://github.com/jessfraz))
* NodeController waits for informer sync before doing anything ([#34809](https://github.com/kubernetes/kubernetes/pull/34809), [@gmarek](https://github.com/gmarek))
* Make NodeController recognize deletion tombstones ([#34786](https://github.com/kubernetes/kubernetes/pull/34786), [@davidopp](https://github.com/davidopp))
* Fix panic in NodeController caused by receiving DeletedFinalStateUnknown object from the cache. ([#34694](https://github.com/kubernetes/kubernetes/pull/34694), [@gmarek](https://github.com/gmarek))
* Update GlusterFS provisioning readme with endpoint/service details ([#31854](https://github.com/kubernetes/kubernetes/pull/31854), [@humblec](https://github.com/humblec))
* Add logging for enabled/disabled API Groups ([#32198](https://github.com/kubernetes/kubernetes/pull/32198), [@deads2k](https://github.com/deads2k))
* New federation deployment mechanism now allows non-GCP clusters. ([#34620](https://github.com/kubernetes/kubernetes/pull/34620), [@madhusudancs](https://github.com/madhusudancs))
        * Writes the federation kubeconfig to the local kubeconfig file.



# v1.4.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.2/kubernetes.tar.gz) | `0730e207944ca96c9d9588a571a5eff0f8fdbb0e1287423513a2b2a4baca9f77`

## Changelog since v1.4.2-beta.1

### Other notable changes

* Cherrypick [#34851](https://github.com/kubernetes/kubernetes/pull/34851) "Only wait for cache syncs once in NodeController" ([#34861](https://github.com/kubernetes/kubernetes/pull/34861), [@jessfraz](https://github.com/jessfraz))
* NodeController waits for informer sync before doing anything ([#34809](https://github.com/kubernetes/kubernetes/pull/34809), [@gmarek](https://github.com/gmarek))
* Make NodeController recognize deletion tombstones ([#34786](https://github.com/kubernetes/kubernetes/pull/34786), [@davidopp](https://github.com/davidopp))
* Fix panic in NodeController caused by receiving DeletedFinalStateUnknown object from the cache. ([#34694](https://github.com/kubernetes/kubernetes/pull/34694), [@gmarek](https://github.com/gmarek))
* Update GlusterFS provisioning readme with endpoint/service details ([#31854](https://github.com/kubernetes/kubernetes/pull/31854), [@humblec](https://github.com/humblec))
* Add logging for enabled/disabled API Groups ([#32198](https://github.com/kubernetes/kubernetes/pull/32198), [@deads2k](https://github.com/deads2k))
* New federation deployment mechanism now allows non-GCP clusters. ([#34620](https://github.com/kubernetes/kubernetes/pull/34620), [@madhusudancs](https://github.com/madhusudancs))
        * Writes the federation kubeconfig to the local kubeconfig file.



# v1.5.0-alpha.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.5.0-alpha.1/kubernetes.tar.gz) | `86bfcfffaa210ddf18983ff066470ef9c06ee00449b2238043e2777aac2c906d`

## Changelog since v1.4.0-alpha.3

### Experimental Features

* `kubeadm` (alpha) provides an easy way to securely bootstrap Kubernetes on Linux, see http://kubernetes.io/docs/kubeadm/ ([#33262](https://github.com/kubernetes/kubernetes/pull/33262), [@errordeveloper](https://github.com/errordeveloper))
* Alpha JWS Discovery API for locating an apiserver securely ([#32203](https://github.com/kubernetes/kubernetes/pull/32203), [@dgoodwin](https://github.com/dgoodwin))

### Action Required

* kube-apiserver learned the '--anonymous-auth' flag, which defaults to true. When enabled, requests to the secure port that are not rejected by other configured authentication methods are treated as anonymous requests, and given a username of 'system:anonymous' and a group of 'system:unauthenticated'.  ([#32386](https://github.com/kubernetes/kubernetes/pull/32386), [@liggitt](https://github.com/liggitt))
    * Authenticated users are decorated with a 'system:authenticated' group.
    * NOTE: anonymous access is enabled by default. If you rely on authentication alone to authorize access, change to use an authorization mode other than AlwaysAllow, or or set '--anonymous-auth=false'.
* The NamespaceExists and NamespaceAutoProvision admission controllers have been removed. ([#31250](https://github.com/kubernetes/kubernetes/pull/31250), [@derekwaynecarr](https://github.com/derekwaynecarr))
    * All cluster operators should use NamespaceLifecycle.
* Federation binaries and their corresponding docker images - `federation-apiserver` and `federation-controller-manager` are now folded in to the `hyperkube` binary. If you were using one of these binaries or docker images, please switch to using the `hyperkube` version. Please refer to the federation manifests - `federation/manifests/federation-apiserver.yaml` and `federation/manifests/federation-controller-manager-deployment.yaml` for examples. ([#29929](https://github.com/kubernetes/kubernetes/pull/29929), [@madhusudancs](https://github.com/madhusudancs))

### Other notable changes

* The kube-apiserver --service-account-key-file option can be specified multiple times, or can point to a file containing multiple keys, to enable rotation of signing keys. ([#34029](https://github.com/kubernetes/kubernetes/pull/34029), [@liggitt](https://github.com/liggitt))
* The apiserver now uses addresses reported by the kubelet in the Node object's status for apiserver->kubelet communications, rather than the name of the Node object. The address type used defaults to `InternalIP`, `ExternalIP`, and `LegacyHostIP` address types, in that order. ([#33718](https://github.com/kubernetes/kubernetes/pull/33718), [@justinsb](https://github.com/justinsb))
* Federated deployment controller that supports the same api as the regular kubernetes deployment controller. ([#34109](https://github.com/kubernetes/kubernetes/pull/34109), [@mwielgus](https://github.com/mwielgus))
* Match GroupVersionKind against specific version ([#34010](https://github.com/kubernetes/kubernetes/pull/34010), [@soltysh](https://github.com/soltysh))
* fix yaml decode issue ([#34297](https://github.com/kubernetes/kubernetes/pull/34297), [@AdoHe](https://github.com/AdoHe))
* kubectl annotate now supports --dry-run ([#34199](https://github.com/kubernetes/kubernetes/pull/34199), [@asalkeld](https://github.com/asalkeld))
* kubectl: Add external ip information to node when '-o wide' is used ([#33552](https://github.com/kubernetes/kubernetes/pull/33552), [@floreks](https://github.com/floreks))
* Update GCI base image: ([#34156](https://github.com/kubernetes/kubernetes/pull/34156), [@adityakali](https://github.com/adityakali))
        * Enabled VXLAN and IP_SET config options in kernel to support some networking tools (ebtools)
        * OpenSSL CVE fixes
* ContainerVm/GCI image: try to use ifdown/ifup if available ([#33595](https://github.com/kubernetes/kubernetes/pull/33595), [@freehan](https://github.com/freehan))
* Use manifest digest (as `docker-pullable://`) as ImageID when available (exposes a canonical, pullable image ID for containers). ([#33014](https://github.com/kubernetes/kubernetes/pull/33014), [@DirectXMan12](https://github.com/DirectXMan12))
* Add kubelet awareness to taint tolerant match caculator. ([#26501](https://github.com/kubernetes/kubernetes/pull/26501), [@resouer](https://github.com/resouer))
* Fix nil pointer issue when getting metrics from volume mounter ([#34251](https://github.com/kubernetes/kubernetes/pull/34251), [@jingxu97](https://github.com/jingxu97))
* Enforce Disk based pod eviction with GCI base image in Kubelet ([#33520](https://github.com/kubernetes/kubernetes/pull/33520), [@vishh](https://github.com/vishh))
* Remove headers that are unnecessary for proxy target ([#34076](https://github.com/kubernetes/kubernetes/pull/34076), [@mbohlool](https://github.com/mbohlool))
* Add missing argument to log message in federated ingress controller. ([#34158](https://github.com/kubernetes/kubernetes/pull/34158), [@quinton-hoole](https://github.com/quinton-hoole))
* The kubelet --eviction-minimum-reclaim option can now take precentages as well as absolute values for resources quantities ([#33392](https://github.com/kubernetes/kubernetes/pull/33392), [@sjenning](https://github.com/sjenning))
* The implicit registration of Prometheus metrics for workqueue has been removed, and a plug-able interface was added. If you were using workqueue in your own binaries and want these metrics, add the following to your imports in the main package: "k8s.io/pkg/util/workqueue/prometheus". ([#33792](https://github.com/kubernetes/kubernetes/pull/33792), [@caesarxuchao](https://github.com/caesarxuchao))
* Add kubectl --node-port option for specifying the service nodeport ([#33319](https://github.com/kubernetes/kubernetes/pull/33319), [@juanvallejo](https://github.com/juanvallejo))
* To reduce memory usage to reasonable levels in smaller clusters, kube-apiserver now sets the deserialization cache size based on the target memory usage. ([#34000](https://github.com/kubernetes/kubernetes/pull/34000), [@wojtek-t](https://github.com/wojtek-t))
* use service accounts as clients for controllers ([#33310](https://github.com/kubernetes/kubernetes/pull/33310), [@deads2k](https://github.com/deads2k))
* Add a new option "--local" to the `kubectl annotate` ([#34074](https://github.com/kubernetes/kubernetes/pull/34074), [@asalkeld](https://github.com/asalkeld))
* Add a new option "--local" to the `kubectl label` ([#33990](https://github.com/kubernetes/kubernetes/pull/33990), [@asalkeld](https://github.com/asalkeld))
* Initialize podsWithAffinity to avoid scheduler panic ([#33967](https://github.com/kubernetes/kubernetes/pull/33967), [@xiang90](https://github.com/xiang90))
* Fix base image pinning during upgrades via cluster/gce/upgrade.sh ([#33147](https://github.com/kubernetes/kubernetes/pull/33147), [@vishh](https://github.com/vishh))
* Remove the flannel experimental overlay ([#33862](https://github.com/kubernetes/kubernetes/pull/33862), [@luxas](https://github.com/luxas))
* CRI: Remove the mount name and port name. ([#33970](https://github.com/kubernetes/kubernetes/pull/33970), [@yifan-gu](https://github.com/yifan-gu))
* Enable kubectl describe rs to work when apiserver does not support pods ([#33794](https://github.com/kubernetes/kubernetes/pull/33794), [@nikhiljindal](https://github.com/nikhiljindal))
* Heal the namespaceless ingresses in federation e2e. ([#33977](https://github.com/kubernetes/kubernetes/pull/33977), [@quinton-hoole](https://github.com/quinton-hoole))
* Fix issue in updating device path when volume is attached multiple times ([#33796](https://github.com/kubernetes/kubernetes/pull/33796), [@jingxu97](https://github.com/jingxu97))
* ECDSA keys can now be used for signing and verifying service account tokens. ([#33565](https://github.com/kubernetes/kubernetes/pull/33565), [@liggitt](https://github.com/liggitt))
* OnlyLocal nodeports ([#33587](https://github.com/kubernetes/kubernetes/pull/33587), [@bprashanth](https://github.com/bprashanth))
* Remove flannel because now everything here is upstreamed ([#33860](https://github.com/kubernetes/kubernetes/pull/33860), [@luxas](https://github.com/luxas))
* Use patched golang1.7.1 for cross-builds targeting darwin ([#33803](https://github.com/kubernetes/kubernetes/pull/33803), [@ixdy](https://github.com/ixdy))
* Bump up addon kube-dns to v20 for graceful termination ([#33774](https://github.com/kubernetes/kubernetes/pull/33774), [@MrHohn](https://github.com/MrHohn))
* Creating LoadBalancer Service with "None" ClusterIP is no longer possible ([#33274](https://github.com/kubernetes/kubernetes/pull/33274), [@nebril](https://github.com/nebril))
* Increase timeout for federated ingress test. ([#33610](https://github.com/kubernetes/kubernetes/pull/33610), [@quinton-hoole](https://github.com/quinton-hoole))
* Use UpdateStatus, not Update, to add LoadBalancerStatus to Federated Ingress.  ([#33605](https://github.com/kubernetes/kubernetes/pull/33605), [@quinton-hoole](https://github.com/quinton-hoole))
* add anytoken authenticator ([#33378](https://github.com/kubernetes/kubernetes/pull/33378), [@deads2k](https://github.com/deads2k))
* Fixes in HPA: consider only running pods; proper denominator in avg request calculations. ([#33735](https://github.com/kubernetes/kubernetes/pull/33735), [@jszczepkowski](https://github.com/jszczepkowski))
* When CORS Handler is enabled, we now add a new HTTP header named "Access-Control-Expose-Headers" with a value of "Date". This allows the "Date" HTTP header to be accessed from XHR/JavaScript. ([#33242](https://github.com/kubernetes/kubernetes/pull/33242), [@dims](https://github.com/dims))
* promote contrib/mesos to incubator ([#33658](https://github.com/kubernetes/kubernetes/pull/33658), [@deads2k](https://github.com/deads2k))
* MinReadySeconds / AvailableReplicas for ReplicaSets ([#32771](https://github.com/kubernetes/kubernetes/pull/32771), [@kargakis](https://github.com/kargakis))
* Kubectl drain will now drain finished Pods ([#31763](https://github.com/kubernetes/kubernetes/pull/31763), [@fraenkel](https://github.com/fraenkel))
* Adds the -deployment option to e2e.go, adds the ability to run e2e.go using a `kops` deployment. ([#33518](https://github.com/kubernetes/kubernetes/pull/33518), [@zmerlynn](https://github.com/zmerlynn))
* Tune down initialDelaySeconds for readinessProbe. ([#33146](https://github.com/kubernetes/kubernetes/pull/33146), [@MrHohn](https://github.com/MrHohn))
* kube-proxy: Add a lower-bound for conntrack (128k default) ([#33051](https://github.com/kubernetes/kubernetes/pull/33051), [@thockin](https://github.com/thockin))
* local-up-cluster.sh: add SERVICE_CLUSTER_IP_RANGE as option ([#32921](https://github.com/kubernetes/kubernetes/pull/32921), [@aanm](https://github.com/aanm))
* Default HTTP2 on, post fixes from [#29001](https://github.com/kubernetes/kubernetes/pull/29001) ([#32231](https://github.com/kubernetes/kubernetes/pull/32231), [@timothysc](https://github.com/timothysc))
* Split dns healthcheck into two different urls ([#32406](https://github.com/kubernetes/kubernetes/pull/32406), [@MrHohn](https://github.com/MrHohn))
* Remove kubectl namespace command ([#33275](https://github.com/kubernetes/kubernetes/pull/33275), [@maciaszczykm](https://github.com/maciaszczykm))
* Automatic generation of man pages ([#33277](https://github.com/kubernetes/kubernetes/pull/33277), [@mkumatag](https://github.com/mkumatag))
* Fixes memory/goroutine leak in Federation Service controller. ([#33359](https://github.com/kubernetes/kubernetes/pull/33359), [@shashidharatd](https://github.com/shashidharatd))
* Switch k8s on GCE to use GCI by default ([#33353](https://github.com/kubernetes/kubernetes/pull/33353), [@vishh](https://github.com/vishh))
* Move HighWaterMark to the top of the struct in order to fix arm, second time ([#33376](https://github.com/kubernetes/kubernetes/pull/33376), [@luxas](https://github.com/luxas))
* Fix race condition in setting node statusUpdateNeeded flag  ([#32807](https://github.com/kubernetes/kubernetes/pull/32807), [@jingxu97](https://github.com/jingxu97))
* Fix the DOCKER_OPTS appending bug. ([#33163](https://github.com/kubernetes/kubernetes/pull/33163), [@DjangoPeng](https://github.com/DjangoPeng))
* Send recycle events from pod to pv. ([#27714](https://github.com/kubernetes/kubernetes/pull/27714), [@jsafrane](https://github.com/jsafrane))
* Add port forwarding for rkt with kvm stage1 ([#32126](https://github.com/kubernetes/kubernetes/pull/32126), [@jjlakis](https://github.com/jjlakis))
* The value of the `versioned.Event` object (returned by watch APIs) in the Swagger 1.2 schemas has been updated from `*versioned.Event` which was not expected by many client tools. The new value is consistent with other structs returned by the API. ([#33007](https://github.com/kubernetes/kubernetes/pull/33007), [@smarterclayton](https://github.com/smarterclayton))
* Remove cpu limits for dns pod to avoid CPU starvation ([#33227](https://github.com/kubernetes/kubernetes/pull/33227), [@vishh](https://github.com/vishh))
* Allow secure access to apiserver from Admission Controllers ([#31491](https://github.com/kubernetes/kubernetes/pull/31491), [@dims](https://github.com/dims))
* Resolves x509 verification issue with masters dialing nodes when started with --kubelet-certificate-authority ([#33141](https://github.com/kubernetes/kubernetes/pull/33141), [@liggitt](https://github.com/liggitt))
* Fix possible panic in PodAffinityChecker ([#33086](https://github.com/kubernetes/kubernetes/pull/33086), [@ivan4th](https://github.com/ivan4th))
* Upgrading Container-VM base image for k8s on GCE. Brief changelog as follows: ([#32738](https://github.com/kubernetes/kubernetes/pull/32738), [@Amey-D](https://github.com/Amey-D))
    *     - Fixed performance regression in veth device driver
    *     - Docker and related binaries are statically linked
    *     - Fixed the issue of systemd being oom-killable
* Move HighWaterMark to the top of the struct in order to fix arm ([#33117](https://github.com/kubernetes/kubernetes/pull/33117), [@luxas](https://github.com/luxas))
* kubenet: SyncHostports for both running and ready to run pods. ([#31388](https://github.com/kubernetes/kubernetes/pull/31388), [@yifan-gu](https://github.com/yifan-gu))
* Limit the number of names per image reported in the node status ([#32914](https://github.com/kubernetes/kubernetes/pull/32914), [@yujuhong](https://github.com/yujuhong))
* Support Quobyte as StorageClass ([#31434](https://github.com/kubernetes/kubernetes/pull/31434), [@johscheuer](https://github.com/johscheuer))
* Use a patched go1.7.1 for building linux/arm ([#32517](https://github.com/kubernetes/kubernetes/pull/32517), [@luxas](https://github.com/luxas))
* Add line break after events in kubectl describe ([#31463](https://github.com/kubernetes/kubernetes/pull/31463), [@fabianofranz](https://github.com/fabianofranz))
* Specific error message on failed rolling update issued by older kubectl against 1.4 master ([#32751](https://github.com/kubernetes/kubernetes/pull/32751), [@caesarxuchao](https://github.com/caesarxuchao))
* Make the informer library available for the go client library. ([#32718](https://github.com/kubernetes/kubernetes/pull/32718), [@mikedanese](https://github.com/mikedanese))
* Added --log-facility flag to enhance dnsmasq logging ([#32422](https://github.com/kubernetes/kubernetes/pull/32422), [@MrHohn](https://github.com/MrHohn))
* Set Dashboard UI to final 1.4 version ([#32666](https://github.com/kubernetes/kubernetes/pull/32666), [@bryk](https://github.com/bryk))
* Fix audit_test regex for iso8601 timestamps ([#32593](https://github.com/kubernetes/kubernetes/pull/32593), [@johnbieren](https://github.com/johnbieren))
* Docker digest validation is too strict ([#32627](https://github.com/kubernetes/kubernetes/pull/32627), [@smarterclayton](https://github.com/smarterclayton))
* Bumped Heapster to v1.2.0. ([#32649](https://github.com/kubernetes/kubernetes/pull/32649), [@piosz](https://github.com/piosz))
    * More details about the release https://github.com/kubernetes/heapster/releases/tag/v1.2.0
* add local subject access review API ([#32407](https://github.com/kubernetes/kubernetes/pull/32407), [@deads2k](https://github.com/deads2k))
* make --runtime-config=api/all=true|false work ([#32582](https://github.com/kubernetes/kubernetes/pull/32582), [@jlowdermilk](https://github.com/jlowdermilk))
* Added new kubelet flags `--cni-bin-dir` and `--cni-conf-dir` to specify where CNI files are located. ([#32151](https://github.com/kubernetes/kubernetes/pull/32151), [@bboreham](https://github.com/bboreham))
    * Fixed CNI configuration on GCI platform when using CNI.
* Move push-ci-build.sh to kubernetes/release repo ([#32444](https://github.com/kubernetes/kubernetes/pull/32444), [@david-mcmahon](https://github.com/david-mcmahon))
* vendor: update github.com/coreos/go-oidc client package ([#31564](https://github.com/kubernetes/kubernetes/pull/31564), [@ericchiang](https://github.com/ericchiang))
* Fixed an issue that caused a credential error when deploying federation control plane onto a GKE cluster. ([#31747](https://github.com/kubernetes/kubernetes/pull/31747), [@madhusudancs](https://github.com/madhusudancs))
* Error if a contextName is provided but not found in the kubeconfig. ([#31767](https://github.com/kubernetes/kubernetes/pull/31767), [@asalkeld](https://github.com/asalkeld))
* Use a Deployment for kube-dns ([#32018](https://github.com/kubernetes/kubernetes/pull/32018), [@MrHohn](https://github.com/MrHohn))
* Support graceful termination in kube-dns ([#31894](https://github.com/kubernetes/kubernetes/pull/31894), [@MrHohn](https://github.com/MrHohn))
* When prompting for passwords, don't echo to the terminal ([#31586](https://github.com/kubernetes/kubernetes/pull/31586), [@brendandburns](https://github.com/brendandburns))
* add group prefix matching for kubectl usage ([#32140](https://github.com/kubernetes/kubernetes/pull/32140), [@deads2k](https://github.com/deads2k))
* Stick to 2.2.1 etcd ([#32404](https://github.com/kubernetes/kubernetes/pull/32404), [@caesarxuchao](https://github.com/caesarxuchao))
* Fix a bug in kubelet hostport logic which flushes KUBE-MARK-MASQ iptables chain ([#32413](https://github.com/kubernetes/kubernetes/pull/32413), [@freehan](https://github.com/freehan))
* Make sure finalizers prevent deletion on storage that supports graceful deletion ([#32351](https://github.com/kubernetes/kubernetes/pull/32351), [@caesarxuchao](https://github.com/caesarxuchao))
* AWS: Change default networking for kube-up to kubenet ([#32239](https://github.com/kubernetes/kubernetes/pull/32239), [@zmerlynn](https://github.com/zmerlynn))
* Use etcd 2.3.7 ([#32359](https://github.com/kubernetes/kubernetes/pull/32359), [@wojtek-t](https://github.com/wojtek-t))
* Allow missing keys in jsonpath ([#31714](https://github.com/kubernetes/kubernetes/pull/31714), [@smarterclayton](https://github.com/smarterclayton))
* Changes 'kubectl rollout status' to wait until all updated replicas are available before finishing. ([#31499](https://github.com/kubernetes/kubernetes/pull/31499), [@areed](https://github.com/areed))
* add selfsubjectaccessreview API ([#31271](https://github.com/kubernetes/kubernetes/pull/31271), [@deads2k](https://github.com/deads2k))
* Add kubectl describe cmd support for vSphere volume ([#31045](https://github.com/kubernetes/kubernetes/pull/31045), [@abrarshivani](https://github.com/abrarshivani))
* Enable kubelet eviction whenever inodes free is < 5% on GCE ([#31545](https://github.com/kubernetes/kubernetes/pull/31545), [@vishh](https://github.com/vishh))
* Use federated namespace instead of the bootstrap cluster's namespace in Ingress e2e tests. ([#32105](https://github.com/kubernetes/kubernetes/pull/32105), [@madhusudancs](https://github.com/madhusudancs))
* Move StorageClass to a storage group ([#31886](https://github.com/kubernetes/kubernetes/pull/31886), [@deads2k](https://github.com/deads2k))
* Some components like kube-dns and kube-proxy could fail to load the service account token when started within a pod. Properly handle empty configurations to try loading the service account config. ([#31947](https://github.com/kubernetes/kubernetes/pull/31947), [@smarterclayton](https://github.com/smarterclayton))
* Removed comments in json config when using kubectl edit with -o json ([#31685](https://github.com/kubernetes/kubernetes/pull/31685), [@jellonek](https://github.com/jellonek))
* fixes invalid null selector issue in sysdig example yaml ([#31393](https://github.com/kubernetes/kubernetes/pull/31393), [@baldwinSPC](https://github.com/baldwinSPC))
* Rescheduler which ensures that critical pods are always scheduled enabled by default in GCE. ([#31974](https://github.com/kubernetes/kubernetes/pull/31974), [@piosz](https://github.com/piosz))
* retry oauth token fetch in gce cloudprovider ([#32021](https://github.com/kubernetes/kubernetes/pull/32021), [@mikedanese](https://github.com/mikedanese))
* Deprecate the old cbr0 and flannel networking modes ([#31197](https://github.com/kubernetes/kubernetes/pull/31197), [@freehan](https://github.com/freehan))
* AWS: fix volume device assignment race condition ([#31090](https://github.com/kubernetes/kubernetes/pull/31090), [@justinsb](https://github.com/justinsb))
* The certificates API group has been renamed to certificates.k8s.io ([#31887](https://github.com/kubernetes/kubernetes/pull/31887), [@liggitt](https://github.com/liggitt))
* Increase Dashboard UI version to v1.4.0-beta2 ([#31518](https://github.com/kubernetes/kubernetes/pull/31518), [@bryk](https://github.com/bryk))
* Fixed incomplete kubectl bash completion. ([#31333](https://github.com/kubernetes/kubernetes/pull/31333), [@xingzhou](https://github.com/xingzhou))
* Added liveness probe to Heapster service. ([#31878](https://github.com/kubernetes/kubernetes/pull/31878), [@mksalawa](https://github.com/mksalawa))
* Adding clusters to the list of valid resources printed by kubectl help ([#31719](https://github.com/kubernetes/kubernetes/pull/31719), [@nikhiljindal](https://github.com/nikhiljindal))
* Kubernetes server components using `kubeconfig` files no longer default to `http://localhost:8080`.  Administrators must specify a server value in their kubeconfig files. ([#30808](https://github.com/kubernetes/kubernetes/pull/30808), [@smarterclayton](https://github.com/smarterclayton))
* Update influxdb to 0.12 ([#31519](https://github.com/kubernetes/kubernetes/pull/31519), [@piosz](https://github.com/piosz))
* Include security options in the container created event ([#31557](https://github.com/kubernetes/kubernetes/pull/31557), [@timstclair](https://github.com/timstclair))
* Federation can now be deployed using the `federation/deploy/deploy.sh` script. This script does not depend on any of the development environment shell library/scripts. This is an alternative to the current `federation-up.sh`/`federation-down.sh` scripts. Both the scripts are going to co-exist in this release, but the `federation-up.sh`/`federation-down.sh` scripts might be removed in a future release in favor of `federation/deploy/deploy.sh` script. ([#30744](https://github.com/kubernetes/kubernetes/pull/30744), [@madhusudancs](https://github.com/madhusudancs))
* Add get/delete cluster, delete context to kubectl config ([#29821](https://github.com/kubernetes/kubernetes/pull/29821), [@alexbrand](https://github.com/alexbrand))
* rkt: Force `rkt fetch` to fetch from remote to conform the image pull policy. ([#31378](https://github.com/kubernetes/kubernetes/pull/31378), [@yifan-gu](https://github.com/yifan-gu))
* Allow services which use same port, different protocol to use the same nodePort for both ([#30253](https://github.com/kubernetes/kubernetes/pull/30253), [@AdoHe](https://github.com/AdoHe))
* Handle overlapping deployments gracefully ([#30730](https://github.com/kubernetes/kubernetes/pull/30730), [@janetkuo](https://github.com/janetkuo))
* Remove environment variables and internal Kubernetes Docker labels from cAdvisor Prometheus metric labels. ([#31064](https://github.com/kubernetes/kubernetes/pull/31064), [@grobie](https://github.com/grobie))
    * Old behavior:
    * - environment variables explicitly whitelisted via --docker-env-metadata-whitelist were exported as `container_env_*=*`. Default is zero so by default non were exported
    * - all docker labels were exported as `container_label_*=*`
    * New behavior:
    * - Only `container_name`, `pod_name`, `namespace`, `id`, `image`, and `name` labels are exposed
    * - no environment variables will be exposed ever via /metrics, even if whitelisted
* Filter duplicate network packets in promiscuous bridge mode (with ebtables) ([#28717](https://github.com/kubernetes/kubernetes/pull/28717), [@freehan](https://github.com/freehan))
* Refactor to simplify the hard-traveled path of the KubeletConfiguration object ([#29216](https://github.com/kubernetes/kubernetes/pull/29216), [@mtaufen](https://github.com/mtaufen))
* Fix overflow issue in controller-manager rate limiter ([#31396](https://github.com/kubernetes/kubernetes/pull/31396), [@foxish](https://github.com/foxish))



# v1.4.2-beta.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.2-beta.1/kubernetes.tar.gz) | `b72986a0adcb7e08feb580c5d72de129ac2ecc128c154fd79785bac2d2e760f7`

## Changelog since v1.4.1

### Other notable changes

* Fix base image pinning during upgrades via cluster/gce/upgrade.sh ([#33147](https://github.com/kubernetes/kubernetes/pull/33147), [@vishh](https://github.com/vishh))
* Fix upgrade.sh image setup ([#34468](https://github.com/kubernetes/kubernetes/pull/34468), [@mtaufen](https://github.com/mtaufen))
* Add `cifs-utils` to the hyperkube image. ([#34416](https://github.com/kubernetes/kubernetes/pull/34416), [@colemickens](https://github.com/colemickens))
* Match GroupVersionKind against specific version ([#34010](https://github.com/kubernetes/kubernetes/pull/34010), [@soltysh](https://github.com/soltysh))
* Fixed an issue that caused a credential error when deploying federation control plane onto a GKE cluster. ([#31747](https://github.com/kubernetes/kubernetes/pull/31747), [@madhusudancs](https://github.com/madhusudancs))
* Test x509 intermediates correctly ([#34524](https://github.com/kubernetes/kubernetes/pull/34524), [@liggitt](https://github.com/liggitt))



# v1.4.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.1/kubernetes.tar.gz) | `b51971d872426ba71bb09b9a9191bb95fc0e48390dc287a9080e3876c8e19a95`

## Changelog since v1.4.1-beta.2

**No notable changes for this release**



# v1.4.1-beta.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.1-beta.2/kubernetes.tar.gz) | `708fbaabf17a69c69c2c9a715e152a29d47334b8c98d217ba17e9b42d6770f25`

## Changelog since v1.4.0

### Other notable changes

* Update GCI base image: ([#34156](https://github.com/kubernetes/kubernetes/pull/34156), [@adityakali](https://github.com/adityakali))
        * Enabled VXLAN and IP_SET config options in kernel to support some networking tools (ebtools)
        * OpenSSL CVE fixes
* ContainerVm/GCI image: try to use ifdown/ifup if available ([#33595](https://github.com/kubernetes/kubernetes/pull/33595), [@freehan](https://github.com/freehan))
* Make the informer library available for the go client library. ([#32718](https://github.com/kubernetes/kubernetes/pull/32718), [@mikedanese](https://github.com/mikedanese))
* Enforce Disk based pod eviction with GCI base image in Kubelet ([#33520](https://github.com/kubernetes/kubernetes/pull/33520), [@vishh](https://github.com/vishh))
* Fix nil pointer issue when getting metrics from volume mounter ([#34251](https://github.com/kubernetes/kubernetes/pull/34251), [@jingxu97](https://github.com/jingxu97))
* Enable kubectl describe rs to work when apiserver does not support pods ([#33794](https://github.com/kubernetes/kubernetes/pull/33794), [@nikhiljindal](https://github.com/nikhiljindal))
* Increase timeout for federated ingress test. ([#33610](https://github.com/kubernetes/kubernetes/pull/33610), [@quinton-hoole](https://github.com/quinton-hoole))
* Remove headers that are unnecessary for proxy target ([#34076](https://github.com/kubernetes/kubernetes/pull/34076), [@mbohlool](https://github.com/mbohlool))
* Support graceful termination in kube-dns ([#31894](https://github.com/kubernetes/kubernetes/pull/31894), [@MrHohn](https://github.com/MrHohn))
* Added --log-facility flag to enhance dnsmasq logging ([#32422](https://github.com/kubernetes/kubernetes/pull/32422), [@MrHohn](https://github.com/MrHohn))
* Split dns healthcheck into two different urls ([#32406](https://github.com/kubernetes/kubernetes/pull/32406), [@MrHohn](https://github.com/MrHohn))
* Tune down initialDelaySeconds for readinessProbe. ([#33146](https://github.com/kubernetes/kubernetes/pull/33146), [@MrHohn](https://github.com/MrHohn))
* Bump up addon kube-dns to v20 for graceful termination ([#33774](https://github.com/kubernetes/kubernetes/pull/33774), [@MrHohn](https://github.com/MrHohn))
* Send recycle events from pod to pv. ([#27714](https://github.com/kubernetes/kubernetes/pull/27714), [@jsafrane](https://github.com/jsafrane))
* Limit the number of names per image reported in the node status ([#32914](https://github.com/kubernetes/kubernetes/pull/32914), [@yujuhong](https://github.com/yujuhong))
* Fixes in HPA: consider only running pods; proper denominator in avg request calculations. ([#33735](https://github.com/kubernetes/kubernetes/pull/33735), [@jszczepkowski](https://github.com/jszczepkowski))
* Fix audit_test regex for iso8601 timestamps ([#32593](https://github.com/kubernetes/kubernetes/pull/32593), [@johnbieren](https://github.com/johnbieren))
* Limit the number of names per image reported in the node status ([#32914](https://github.com/kubernetes/kubernetes/pull/32914), [@yujuhong](https://github.com/yujuhong))
* Fix the DOCKER_OPTS appending bug. ([#33163](https://github.com/kubernetes/kubernetes/pull/33163), [@DjangoPeng](https://github.com/DjangoPeng))
* Remove cpu limits for dns pod to avoid CPU starvation ([#33227](https://github.com/kubernetes/kubernetes/pull/33227), [@vishh](https://github.com/vishh))
* Fixes memory/goroutine leak in Federation Service controller. ([#33359](https://github.com/kubernetes/kubernetes/pull/33359), [@shashidharatd](https://github.com/shashidharatd))
* Use UpdateStatus, not Update, to add LoadBalancerStatus to Federated Ingress.  ([#33605](https://github.com/kubernetes/kubernetes/pull/33605), [@quinton-hoole](https://github.com/quinton-hoole))
* Initialize podsWithAffinity to avoid scheduler panic ([#33967](https://github.com/kubernetes/kubernetes/pull/33967), [@xiang90](https://github.com/xiang90))
* Heal the namespaceless ingresses in federation e2e. ([#33977](https://github.com/kubernetes/kubernetes/pull/33977), [@quinton-hoole](https://github.com/quinton-hoole))
* Add missing argument to log message in federated ingress controller. ([#34158](https://github.com/kubernetes/kubernetes/pull/34158), [@quinton-hoole](https://github.com/quinton-hoole))
* Fix issue in updating device path when volume is attached multiple times ([#33796](https://github.com/kubernetes/kubernetes/pull/33796), [@jingxu97](https://github.com/jingxu97))
* To reduce memory usage to reasonable levels in smaller clusters, kube-apiserver now sets the deserialization cache size based on the target memory usage. ([#34000](https://github.com/kubernetes/kubernetes/pull/34000), [@wojtek-t](https://github.com/wojtek-t))
* Fix possible panic in PodAffinityChecker ([#33086](https://github.com/kubernetes/kubernetes/pull/33086), [@ivan4th](https://github.com/ivan4th))
* Fix race condition in setting node statusUpdateNeeded flag  ([#32807](https://github.com/kubernetes/kubernetes/pull/32807), [@jingxu97](https://github.com/jingxu97))
* kube-proxy: Add a lower-bound for conntrack (128k default) ([#33051](https://github.com/kubernetes/kubernetes/pull/33051), [@thockin](https://github.com/thockin))
* Use patched golang1.7.1 for cross-builds targeting darwin ([#33803](https://github.com/kubernetes/kubernetes/pull/33803), [@ixdy](https://github.com/ixdy))
* Move HighWaterMark to the top of the struct in order to fix arm ([#33117](https://github.com/kubernetes/kubernetes/pull/33117), [@luxas](https://github.com/luxas))
* Move HighWaterMark to the top of the struct in order to fix arm, second time ([#33376](https://github.com/kubernetes/kubernetes/pull/33376), [@luxas](https://github.com/luxas))



# v1.3.8

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.8/kubernetes.tar.gz) | `66cf72d8f07e2f700acfcb11536694e0d904483611ff154f34a8380c63720a8d`

## Changelog since v1.3.7

### Other notable changes

* AWS: fix volume device assignment race condition ([#31090](https://github.com/kubernetes/kubernetes/pull/31090), [@justinsb](https://github.com/justinsb))



# v1.4.0

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0/kubernetes.tar.gz) | `6cf3d78230f7659b87fa399a56a7aaed1fde6a73be9d05e25feedacfbd8d5a16`

## Major Themes

- **Simplified User Experience**
  - Easier to get a cluster up and running (eg: `kubeadm`, intra-cluster bootstrapping)
  - Easier to understand a cluster (eg: API audit logs, server-based API defaults)
- **Stateful Appplication Support**
  - Enhanced persistence capabilities (eg: `StorageClasses`, new volume plugins)
  - New resources and scheduler features (eg: `ScheduledJob` resource, pod/node affinity/anti-affinity)
- **Cluster Federation**
  - Global Multi-cluster HTTP(S) Ingress across GCE and GKE clusters.
  - Expanded support for federated hybrid-cloud resources including ReplicaSets, Secrets, Namespaces and Events.
- **Security**
  - Increased pod-level security granularity (eg: Container Image Policies, AppArmor and `sysctl` support)
  - Increased cluster-level security granularity (eg: Access Review API)

## Features

This is the first release tracked via the use of the [kubernetes/features](https://github.com/kubernetes/features) issues repo.  Each Feature issue is owned by a Special Interest Group from [kubernetes/community](https://github.com/kubernetes/community)

- **API Machinery**
  - [alpha] Generate audit logs for every request user performs against secured API server endpoint. ([docs](http://kubernetes.io/docs/admin/audit/)) ([kubernetes/features#22](https://github.com/kubernetes/features/issues/22))
  - [beta] `kube-apiserver` now publishes a swagger 2.0 spec in addition to a swagger 1.2 spec ([kubernetes/features#53](https://github.com/kubernetes/features/issues/53))
  - [beta] Server-side garbage collection is enabled by default. See [user-guide](http://kubernetes.io/docs/user-guide/garbage-collection/)
- **Apps**
  - [alpha] Introducing 'ScheduledJobs', which allow running time based Jobs, namely once at a specified time or repeatedly at specified point in time. ([docs](http://kubernetes.io/docs/user-guide/scheduled-jobs/)) ([kubernetes/features#19](https://github.com/kubernetes/features/issues/19))
- **Auth**
  - [alpha] Container Image Policy allows an access controller to determine whether a pod may be scheduled based on a policy ([docs](http://kubernetes.io/docs/admin/admission-controllers/#imagepolicywebhook)) ([kubernetes/features#59](https://github.com/kubernetes/features/issues/59))
  - [alpha] Access Review APIs expose authorization engine to external inquiries for delegation, inspection, and debugging ([docs](http://kubernetes.io/docs/admin/authorization/)) ([kubernetes/features#37](https://github.com/kubernetes/features/issues/37))
- **Cluster Lifecycle**
  - [alpha] Ensure critical cluster infrastructure pods (Heapster, DNS, etc.) can schedule by evicting regular pods when necessary to make the critical pods schedule. ([docs](http://kubernetes.io/docs/admin/rescheduler/#guaranteed-scheduling-of-critical-add-on-pods)) ([kubernetes/features#62](https://github.com/kubernetes/features/issues/62))
  - [alpha] Simplifies bootstrapping of TLS secured communication between the API server and kubelet. ([docs](http://kubernetes.io/docs/admin/master-node-communication/#kubelet-tls-bootstrap)) ([kubernetes/features#43](https://github.com/kubernetes/features/issues/43))
  - [alpha] The `kubeadm` tool makes it much easier to bootstrap Kubernetes. ([docs](http://kubernetes.io/docs/getting-started-guides/kubeadm/)) ([kubernetes/features#11](https://github.com/kubernetes/features/issues/11))
- **Federation**
  - [alpha] Creating a `Federated Ingress` is as simple as submitting
    an `Ingress` creation request to the Federation API Server. The
    Federation control system then creates and maintains a single
    global virtual IP to load balance incoming HTTP(S) traffic across
    some or all the registered clusters, across all regions. Google's
    GCE L7 LoadBalancer is the first supported implementation, and
	is available in this release.
	([docs](http://kubernetes.io/docs/user-guide/federation/federated-ingress.md))
	([kubernetes/features#82](https://github.com/kubernetes/features/issues/82))
  - [beta] `Federated Replica Sets` create and maintain matching
    `Replica Set`s in some or all clusters in a federation, with the
    desired replica count distributed equally or according to
    specified per-cluster weights.
	([docs](http://kubernetes.io/docs/user-guide/federation/federated-replicasets.md))
	([kubernetes/features#46](https://github.com/kubernetes/features/issues/46))
  - [beta] `Federated Secrets` are created and kept consistent across all clusters in a federation.
    ([docs](http://kubernetes.io/docs/user-guide/federation/federated-secrets.md))
    ([kubernetes/features#68](https://github.com/kubernetes/features/issues/68))
  - [beta] Federation API server gained support for events and many
    federation controllers now report important events.
    ([docs](http://kubernetes.io/docs/user-guide/federation/events))
    ([kubernetes/features#70](https://github.com/kubernetes/features/issues/70))
  - [alpha] Creating a `Federated Namespace` causes matching
    `Namespace`s to be created and maintained in all the clusters registered with that federation. ([docs](http://kubernetes.io/docs/user-guide/federation/federated-namespaces.md)) ([kubernetes/features#69](https://github.com/kubernetes/features/issues/69))
  - [alpha] ingress has alpha support for a single master multi zone cluster ([docs](http://kubernetes.io/docs/user-guide/ingress.md#failing-across-availability-zones)) ([kubernetes/features#52](https://github.com/kubernetes/features/issues/52))
- **Network**
  - [alpha] Service LB now has alpha support for preserving client source IP ([docs](http://kubernetes.io/docs/user-guide/load-balancer/)) ([kubernetes/features#27](https://github.com/kubernetes/features/issues/27))
- **Node**
  - [alpha] Publish node performance dashboard at http://node-perf-dash.k8s.io/#/builds ([docs](https://github.com/kubernetes/contrib/blob/master/node-perf-dash/README.md)) ([kubernetes/features#83](https://github.com/kubernetes/features/issues/83))
  - [alpha] Pods now have alpha support for setting whitelisted, safe sysctls. Unsafe sysctls can be whitelisted on the kubelet. ([docs](http://kubernetes.io/docs/admin/sysctls/)) ([kubernetes/features#34](https://github.com/kubernetes/features/issues/34))
  - [beta] AppArmor profiles can be specified & applied to pod containers ([docs](http://kubernetes.io/docs/admin/apparmor/)) ([kubernetes/features#24](https://github.com/kubernetes/features/issues/24))
  - [beta] Cluster policy to control access and defaults of security related features ([docs](http://kubernetes.io/docs/user-guide/pod-security-policy/)) ([kubernetes/features#5](https://github.com/kubernetes/features/issues/5))
  - [stable] kubelet is able to evict pods when it observes disk pressure ([docs](http://kubernetes.io/docs/admin/out-of-resource/)) ([kubernetes/features#39](https://github.com/kubernetes/features/issues/39))
  - [stable] Automated docker validation results posted to https://k8s-testgrid.appspot.com/docker [kubernetes/features#57](https://github.com/kubernetes/features/issues/57)
- **Scheduling**
  - [alpha] Allows pods to require or prohibit (or prefer or prefer not) co-scheduling on the same node (or zone or other topology domain) as another set of pods. ([docs](http://kubernetes.io/docs/user-guide/node-selection/) ([kubernetes/features#51](https://github.com/kubernetes/features/issues/51))
- **Storage**
  - [beta] Persistant Volume provisioning now supports multiple provisioners using StorageClass configuration. ([docs](http://kubernetes.io/docs/user-guide/persistent-volumes/)) ([kubernetes/features#36](https://github.com/kubernetes/features/issues/36))
  - [stable] New volume plugin for the Quobyte Distributed File System ([docs](http://kubernetes.io/docs/user-guide/volumes/#quobyte)) ([kubernetes/features#80](https://github.com/kubernetes/features/issues/80))
  - [stable] New volume plugin for Azure Data Disk ([docs](http://kubernetes.io/docs/user-guide/volumes/#azurediskvolume)) ([kubernetes/features#79](https://github.com/kubernetes/features/issues/79))
- **UI**
  - [stable] Kubernetes Dashboard UI - a great looking Kubernetes Dashboard UI with 90% CLI parity for at-a-glance management. [docs](https://github.com/kubernetes/dashboard)
  - [stable] `kubectl` no longer applies defaults before sending objects to the server in create and update requests, allowing the server to apply the defaults. ([kubernetes/features#55](https://github.com/kubernetes/features/issues/55))

## Known Issues

- Completed pods lose logs across node upgrade ([#32324](https://github.com/kubernetes/kubernetes/issues/32324))
- Pods are deleted across node upgrade ([#32323](https://github.com/kubernetes/kubernetes/issues/32323))
- Secure master -> node communication ([#11816](https://github.com/kubernetes/kubernetes/issues/11816))
- upgrading master doesn't upgrade kubectl ([#32538](https://github.com/kubernetes/kubernetes/issues/32538))
- Specific error message on failed rolling update issued by older kubectl against 1.4 master ([#32751](https://github.com/kubernetes/kubernetes/issues/32751))
- bump master cidr range from /30 to /29 ([#32886](https://github.com/kubernetes/kubernetes/issues/32886))
- non-hostNetwork daemonsets will almost always have a pod that fails to schedule ([#32900](https://github.com/kubernetes/kubernetes/issues/32900))
- Service loadBalancerSourceRanges doesn't respect updates ([#33033](https://github.com/kubernetes/kubernetes/issues/33033))
- disallow user to update loadbalancerSourceRanges ([#33346](https://github.com/kubernetes/kubernetes/issues/33346))

## Notable Changes to Existing Behavior

### Deployments

- ReplicaSets of paused Deployments are now scaled while the Deployment is paused. This is retroactive to existing Deployments.
- When scaling a Deployment during a rollout, the ReplicaSets of all Deployments are now scaled proportionally based on the number of replicas they each have instead of only scaling the newest ReplicaSet.

### kubectl rolling-update: < v1.4.0 client vs >=v1.4.0 cluster

Old version kubectl's rolling-update command is compatible with Kubernetes 1.4 and higher only if you specify a new replication controller name. You will need to update to kubectl 1.4 or higher to use the rolling update command against a 1.4 cluster if you want to keep the original name, or you'll have to do two rolling updates.

If you do happen to use old version kubectl's rolling update against a 1.4 cluster, it will fail, usually with an error message that will direct you here. If you saw that error, then don't worry, the operation succeeded except for the part where the new replication controller is renamed back to the old name. You can just do another rolling update using kubectl 1.4 or higher to change the name back: look for a replication controller that has the original name plus a random suffix.

Unfortunately, there is a much rarer second possible failure mode: the replication controller gets renamed to the old name, but there is a duplicated set of pods in the cluster. kubectl will not report an error since it thinks its job is done.

If this happens to you, you can wait at most 10 minutes for the replication controller to start a resync, the extra pods will then be deleted. Or, you can manually trigger a resync by change the replicas in the spec of the replication controller.

### kubectl delete: < v1.4.0 client vs >=v1.4.0 cluster

If you use an old version kubectl to delete a replication controller or replicaset, then after the delete command has returned, the replication controller or the replicaset will continue to exist in the key-value store for a short period of time (<1s). You probably will not notice any difference if you use kubectl manually, but you might notice it if you are using kubectl in a script.

### DELETE operation in REST API

* **Replication controller & Replicaset**: the DELETE request of a replication controller or a replicaset becomes asynchronous by default. The object will continue to exist in the key-value store for some time. The API server will set its metadata.deletionTimestamp, add the "orphan" finalizer to its metadata.finalizers. The object will be deleted from the key-value store after the garbage collector orphans its dependents. Please refer to this [user-guide](http://kubernetes.io/docs/user-guide/garbage-collector/) for more information regarding the garbage collection.

* **Other objects**: no changes unless you explicitly request orphaning.

## Action Required Before Upgrading

- If you are using Kubernetes to manage `docker` containers, please be aware Kubernetes has been validated to work with docker 1.9.1, docker 1.11.2 (#23397), and docker 1.12.0 (#28698)
- If you upgrade your apiserver to 1.4.x but leave your kubelets at 1.3.x, they will not report init container status, but init containers will work properly.  Upgrading kubelets to 1.4.x fixes this.
- The NamespaceExists and NamespaceAutoProvision admission controllers have been removed, use the NamespaceLifecycle admission controller instead (#31250, @derekwaynecarr)
- If upgrading Cluster Federation components from 1.3.x, the `federation-apiserver` and `federation-controller-manager` binaries have been folded into `hyperkube`.  Please switch to using that instead.  (#29929, @madhusudancs)
- If you are using the PodSecurityPolicy feature (eg: `kubectl get podsecuritypolicy` does not error, and returns one or more objects), be aware that init containers have moved from alpha to beta.  If there are any pods with the key `pods.beta.kubernetes.io/init-containers`, then that pod may not have been filtered by the PodSecurityPolicy. You should find such pods and either delete them or audit them to ensure they do not use features that you intend to be blocked by PodSecurityPolicy. (#31026, @erictune)
- If upgrading Cluster Federation components from 1.3.x, please ensure your cluster name is a valid DNS label (#30956, @nikhiljindal)
- kubelet's `--config` flag has been deprecated, use `--pod-manifest-path` instead (#29999, @mtaufen)
- If upgrading Cluster Federation components from 1.3.x, be aware the federation-controller-manager now looks for a different secret name.  Run the following to migrate (#28938, @madhusudancs)

```
kubectl --namespace=federation get secret federation-apiserver-secret -o json | sed 's/federation-apiserver-secret/federation-apiserver-kubeconfig/g' | kubectl create -f -
# optionally, remove the old secret
kubectl delete secret --namespace=federation federation-apiserver-secret
```

- Kubernetes components no longer handle panics, and instead actively crash.  All Kubernetes components should be run by something that actively restarts them. This is true of the default setups, but those with custom environments may need to double-check (#28800, @lavalamp)
- kubelet now defaults to `--cloud-provider=auto-detect`, use `--cloud-provider=''` to preserve previous default of no cloud provider (#28258, @vishh)

## Previous Releases Included in v1.4.0

For a detailed list of all changes that were included in this release, please refer to the following CHANGELOG entries:

- [v1.4.0-beta.10](CHANGELOG.md#v140-beta10)
- [v1.4.0-beta.8](CHANGELOG.md#v140-beta8)
- [v1.4.0-beta.7](CHANGELOG.md#v140-beta7)
- [v1.4.0-beta.6](CHANGELOG.md#v140-beta6)
- [v1.4.0-beta.5](CHANGELOG.md#v140-beta5)
- [v1.4.0-beta.3](CHANGELOG.md#v140-beta3)
- [v1.4.0-beta.2](CHANGELOG.md#v140-beta2)
- [v1.4.0-beta.1](CHANGELOG.md#v140-beta1)
- [v1.4.0-alpha.3](CHANGELOG.md#v140-alpha3)
- [v1.4.0-alpha.2](CHANGELOG.md#v140-alpha2)
- [v1.4.0-alpha.1](CHANGELOG.md#v140-alpha1)



# v1.4.0-beta.11

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.11/kubernetes.tar.gz) | `993e785f501d2fa86c9035b55a875c420059b3541a32b5822acf5fefb9a61916`

## Changelog since v1.4.0-beta.10

**No notable changes for this release**



# v1.4.0-beta.10

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.10/kubernetes.tar.gz) | `f3f1f0e5cf8234d640c8e9444c73343f04be8685f92b6a1ad66190f84de2e3a7`

## Changelog since v1.4.0-beta.8

### Other notable changes

* Remove cpu limits for dns pod to avoid CPU starvation ([#33227](https://github.com/kubernetes/kubernetes/pull/33227), [@vishh](https://github.com/vishh))
* Resolves x509 verification issue with masters dialing nodes when started with --kubelet-certificate-authority ([#33141](https://github.com/kubernetes/kubernetes/pull/33141), [@liggitt](https://github.com/liggitt))
* Upgrading Container-VM base image for k8s on GCE. Brief changelog as follows: ([#32738](https://github.com/kubernetes/kubernetes/pull/32738), [@Amey-D](https://github.com/Amey-D))
    *     - Fixed performance regression in veth device driver
    *     - Docker and related binaries are statically linked
    *     - Fixed the issue of systemd being oom-killable
* Update cAdvisor to v0.24.0 - see the [cAdvisor changelog](https://github.com/google/cadvisor/blob/v0.24.0/CHANGELOG.md) for the full list of changes. ([#33052](https://github.com/kubernetes/kubernetes/pull/33052), [@timstclair](https://github.com/timstclair))



# v1.4.0-beta.8

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.8/kubernetes.tar.gz) | `31701c5c675c137887b58d7914e39b4c8a9c03767c0c3d89198a52f4476278ca`

## Changelog since v1.4.0-beta.7

**No notable changes for this release**



# v1.4.0-beta.7

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.7/kubernetes.tar.gz) | `51e8f3ebe55cfcfbe582dd6e5ea60ae125d89373477571c0faee70eff51bab31`

## Changelog since v1.4.0-beta.6

### Other notable changes

* Use a patched go1.7.1 for building linux/arm ([#32517](https://github.com/kubernetes/kubernetes/pull/32517), [@luxas](https://github.com/luxas))
* Specific error message on failed rolling update issued by older kubectl against 1.4 master ([#32751](https://github.com/kubernetes/kubernetes/pull/32751), [@caesarxuchao](https://github.com/caesarxuchao))



# v1.4.0-beta.6

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.6/kubernetes.tar.gz) | `0b0158e4745663b48c55527247d3e64cc3649f875fa7611fc7b38fa5c3b736bd`

## Changelog since v1.4.0-beta.5

### Other notable changes

* Set Dashboard UI to final 1.4 version ([#32666](https://github.com/kubernetes/kubernetes/pull/32666), [@bryk](https://github.com/bryk))



# v1.4.0-beta.5

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.5/kubernetes.tar.gz) | `ec6b233b0448472e05e6820b8ea1644119ae4f9fe3a1516cf978117c19bad0a9`

## Changelog since v1.4.0-beta.3

### Other notable changes

* Bumped Heapster to v1.2.0. ([#32649](https://github.com/kubernetes/kubernetes/pull/32649), [@piosz](https://github.com/piosz))
    * More details about the release https://github.com/kubernetes/heapster/releases/tag/v1.2.0
* Docker digest validation is too strict ([#32627](https://github.com/kubernetes/kubernetes/pull/32627), [@smarterclayton](https://github.com/smarterclayton))
* Added new kubelet flags `--cni-bin-dir` and `--cni-conf-dir` to specify where CNI files are located. ([#32151](https://github.com/kubernetes/kubernetes/pull/32151), [@bboreham](https://github.com/bboreham))
    * Fixed CNI configuration on GCI platform when using CNI.
* make --runtime-config=api/all=true|false work ([#32582](https://github.com/kubernetes/kubernetes/pull/32582), [@jlowdermilk](https://github.com/jlowdermilk))
* AWS: Change default networking for kube-up to kubenet ([#32239](https://github.com/kubernetes/kubernetes/pull/32239), [@zmerlynn](https://github.com/zmerlynn))



# v1.3.7

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.7/kubernetes.tar.gz) | `ad18566a09ff87b36107c2ea238fa5e20988d7a62c85df9c8598920679fec4a1`

## Changelog since v1.3.6

### Other notable changes

* AWS: Add ap-south-1 to list of known AWS regions ([#28428](https://github.com/kubernetes/kubernetes/pull/28428), [@justinsb](https://github.com/justinsb))
* Back porting critical vSphere bug fixes to release 1.3 ([#31993](https://github.com/kubernetes/kubernetes/pull/31993), [@dagnello](https://github.com/dagnello))
* Back port - Openstack provider allowing more than one service port for lbaas v2 ([#32001](https://github.com/kubernetes/kubernetes/pull/32001), [@dagnello](https://github.com/dagnello))
* Fix a bug in kubelet hostport logic which flushes KUBE-MARK-MASQ iptables chain ([#32413](https://github.com/kubernetes/kubernetes/pull/32413), [@freehan](https://github.com/freehan))
* Fixes the panic that occurs in the federation controller manager when registering a GKE cluster to the federation. Fixes issue [#30790](https://github.com/kubernetes/kubernetes/pull/30790). ([#30940](https://github.com/kubernetes/kubernetes/pull/30940), [@madhusudancs](https://github.com/madhusudancs))



# v1.4.0-beta.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.3/kubernetes.tar.gz) | `5a6802703c6b0b652e72166a4347fee7899c46205463f6797dc78f8086876465`

## Changelog since v1.4.0-beta.2

**No notable changes for this release**

## Behavior changes caused by enabling the garbage collector

### kubectl rolling-update

Old version kubectl's rolling-update command is compatible with Kubernetes 1.4 and higher **only if** you specify a new replication controller name. You will need to update to kubectl 1.4 or higher to use the rolling update command against a 1.4 cluster if you want to keep the original name, or you'll have to do two rolling updates.

If you do happen to use old version kubectl's rolling update against a 1.4 cluster, it will fail, usually with an error message that will direct you here. If you saw that error, then don't worry, the operation succeeded except for the part where the new replication controller is renamed back to the old name. You can just do another rolling update using kubectl 1.4 or higher to change the name back: look for a replication controller that has the original name plus a random suffix.

Unfortunately, there is a much rarer second possible failure mode: the replication controller gets renamed to the old name, but there is a duplicate set of pods in the cluster. kubectl will not report an error since it thinks its job is done.

If this happens to you, you can wait at most 10 minutes for the replication controller to start a resync, the extra pods will then be deleted. Or, you can manually trigger a resync by change the replicas in the spec of the replication controller.

### kubectl delete

If you use an old version kubectl to delete a replication controller or a replicaset, then after the delete command has returned, the replication controller or the replicaset will continue to exist in the key-value store for a short period of time (<1s). You probably will not notice any difference if you use kubectl manually, but you might notice it if you are using kubectl in a script. To fix it, you can poll the API server to confirm the object is deleted.

### DELETE operation in REST API

* **Replication controller & Replicaset**: the DELETE request of a replication controller or a replicaset becomes asynchronous by default. The object will continue to exist in the key-value store for some time. The API server will set its metadata.deletionTimestamp, add the "orphan" finalizer to its metadata.finalizers. The object will be deleted from the key-value store after the garbage collector orphans its dependents. Please refer to this [user-guide](http://kubernetes.io/docs/user-guide/garbage-collector/) for more information regarding the garbage collection.

* **Other objects**: no changes unless you explicitly request orphaning.


# v1.4.0-beta.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.2/kubernetes.tar.gz) | `0c6f54eb9059090c88f10a448ed5bcb6ef663abbd76c79281fd8dcb72faa6315`

## Changelog since v1.4.0-beta.1

### Other notable changes

* Fix a bug in kubelet hostport logic which flushes KUBE-MARK-MASQ iptables chain ([#32413](https://github.com/kubernetes/kubernetes/pull/32413), [@freehan](https://github.com/freehan))
* Stick to 2.2.1 etcd ([#32404](https://github.com/kubernetes/kubernetes/pull/32404), [@caesarxuchao](https://github.com/caesarxuchao))
* Use etcd 2.3.7 ([#32359](https://github.com/kubernetes/kubernetes/pull/32359), [@wojtek-t](https://github.com/wojtek-t))
* AWS: Change default networking for kube-up to kubenet ([#32239](https://github.com/kubernetes/kubernetes/pull/32239), [@zmerlynn](https://github.com/zmerlynn))
* Make sure finalizers prevent deletion on storage that supports graceful deletion ([#32351](https://github.com/kubernetes/kubernetes/pull/32351), [@caesarxuchao](https://github.com/caesarxuchao))
* Some components like kube-dns and kube-proxy could fail to load the service account token when started within a pod. Properly handle empty configurations to try loading the service account config. ([#31947](https://github.com/kubernetes/kubernetes/pull/31947), [@smarterclayton](https://github.com/smarterclayton))
* Use federated namespace instead of the bootstrap cluster's namespace in Ingress e2e tests. ([#32105](https://github.com/kubernetes/kubernetes/pull/32105), [@madhusudancs](https://github.com/madhusudancs))



# v1.4.0-beta.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.4/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-beta.1/kubernetes.tar.gz) | `837296455933629b6792a8954f2c5b17d55c1149c12b644101f2f02549d06d25`

## Changelog since v1.4.0-alpha.3

### Action Required

* The NamespaceExists and NamespaceAutoProvision admission controllers have been removed. ([#31250](https://github.com/kubernetes/kubernetes/pull/31250), [@derekwaynecarr](https://github.com/derekwaynecarr))
    * All cluster operators should use NamespaceLifecycle.
* Federation binaries and their corresponding docker images - `federation-apiserver` and `federation-controller-manager` are now folded in to the `hyperkube` binary. If you were using one of these binaries or docker images, please switch to using the `hyperkube` version. Please refer to the federation manifests - `federation/manifests/federation-apiserver.yaml` and `federation/manifests/federation-controller-manager-deployment.yaml` for examples. ([#29929](https://github.com/kubernetes/kubernetes/pull/29929), [@madhusudancs](https://github.com/madhusudancs))
* Use upgraded container-vm by default on worker nodes for GCE k8s clusters ([#31023](https://github.com/kubernetes/kubernetes/pull/31023), [@vishh](https://github.com/vishh))

### Other notable changes

* Enable kubelet eviction whenever inodes free is < 5% on GCE ([#31545](https://github.com/kubernetes/kubernetes/pull/31545), [@vishh](https://github.com/vishh))
* Move StorageClass to a storage group ([#31886](https://github.com/kubernetes/kubernetes/pull/31886), [@deads2k](https://github.com/deads2k))
* Some components like kube-dns and kube-proxy could fail to load the service account token when started within a pod. Properly handle empty configurations to try loading the service account config. ([#31947](https://github.com/kubernetes/kubernetes/pull/31947), [@smarterclayton](https://github.com/smarterclayton))
* Removed comments in json config when using kubectl edit with -o json ([#31685](https://github.com/kubernetes/kubernetes/pull/31685), [@jellonek](https://github.com/jellonek))
* fixes invalid null selector issue in sysdig example yaml ([#31393](https://github.com/kubernetes/kubernetes/pull/31393), [@baldwinSPC](https://github.com/baldwinSPC))
* Rescheduler which ensures that critical pods are always scheduled enabled by default in GCE. ([#31974](https://github.com/kubernetes/kubernetes/pull/31974), [@piosz](https://github.com/piosz))
* retry oauth token fetch in gce cloudprovider ([#32021](https://github.com/kubernetes/kubernetes/pull/32021), [@mikedanese](https://github.com/mikedanese))
* Deprecate the old cbr0 and flannel networking modes ([#31197](https://github.com/kubernetes/kubernetes/pull/31197), [@freehan](https://github.com/freehan))
* AWS: fix volume device assignment race condition ([#31090](https://github.com/kubernetes/kubernetes/pull/31090), [@justinsb](https://github.com/justinsb))
* The certificates API group has been renamed to certificates.k8s.io ([#31887](https://github.com/kubernetes/kubernetes/pull/31887), [@liggitt](https://github.com/liggitt))
* Increase Dashboard UI version to v1.4.0-beta2 ([#31518](https://github.com/kubernetes/kubernetes/pull/31518), [@bryk](https://github.com/bryk))
* Fixed incomplete kubectl bash completion. ([#31333](https://github.com/kubernetes/kubernetes/pull/31333), [@xingzhou](https://github.com/xingzhou))
* Added liveness probe to Heapster service. ([#31878](https://github.com/kubernetes/kubernetes/pull/31878), [@mksalawa](https://github.com/mksalawa))
* Adding clusters to the list of valid resources printed by kubectl help ([#31719](https://github.com/kubernetes/kubernetes/pull/31719), [@nikhiljindal](https://github.com/nikhiljindal))
* Kubernetes server components using `kubeconfig` files no longer default to `http://localhost:8080`.  Administrators must specify a server value in their kubeconfig files. ([#30808](https://github.com/kubernetes/kubernetes/pull/30808), [@smarterclayton](https://github.com/smarterclayton))
* Update influxdb to 0.12 ([#31519](https://github.com/kubernetes/kubernetes/pull/31519), [@piosz](https://github.com/piosz))
* Include security options in the container created event ([#31557](https://github.com/kubernetes/kubernetes/pull/31557), [@timstclair](https://github.com/timstclair))
* Federation can now be deployed using the `federation/deploy/deploy.sh` script. This script does not depend on any of the development environment shell library/scripts. This is an alternative to the current `federation-up.sh`/`federation-down.sh` scripts. Both the scripts are going to co-exist in this release, but the `federation-up.sh`/`federation-down.sh` scripts might be removed in a future release in favor of `federation/deploy/deploy.sh` script. ([#30744](https://github.com/kubernetes/kubernetes/pull/30744), [@madhusudancs](https://github.com/madhusudancs))
* Add get/delete cluster, delete context to kubectl config ([#29821](https://github.com/kubernetes/kubernetes/pull/29821), [@alexbrand](https://github.com/alexbrand))
* rkt: Force `rkt fetch` to fetch from remote to conform the image pull policy. ([#31378](https://github.com/kubernetes/kubernetes/pull/31378), [@yifan-gu](https://github.com/yifan-gu))
* Allow services which use same port, different protocol to use the same nodePort for both ([#30253](https://github.com/kubernetes/kubernetes/pull/30253), [@AdoHe](https://github.com/AdoHe))
* Handle overlapping deployments gracefully ([#30730](https://github.com/kubernetes/kubernetes/pull/30730), [@janetkuo](https://github.com/janetkuo))
* Remove environment variables and internal Kubernetes Docker labels from cAdvisor Prometheus metric labels. ([#31064](https://github.com/kubernetes/kubernetes/pull/31064), [@grobie](https://github.com/grobie))
    * Old behavior:
    * - environment variables explicitly whitelisted via --docker-env-metadata-whitelist were exported as `container_env_*=*`. Default is zero so by default non were exported
    * - all docker labels were exported as `container_label_*=*`
    * New behavior:
    * - Only `container_name`, `pod_name`, `namespace`, `id`, `image`, and `name` labels are exposed
    * - no environment variables will be exposed ever via /metrics, even if whitelisted
* Filter duplicate network packets in promiscuous bridge mode (with ebtables) ([#28717](https://github.com/kubernetes/kubernetes/pull/28717), [@freehan](https://github.com/freehan))
* Refactor to simplify the hard-traveled path of the KubeletConfiguration object ([#29216](https://github.com/kubernetes/kubernetes/pull/29216), [@mtaufen](https://github.com/mtaufen))
* Fix overflow issue in controller-manager rate limiter ([#31396](https://github.com/kubernetes/kubernetes/pull/31396), [@foxish](https://github.com/foxish))



# v1.3.6

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.6/kubernetes.tar.gz) | `2db7ace2f72a2e162329a6dc969a5a158bb8c5d0f8054c5b1b2b1063aa22020d`

## Changelog since v1.3.5

### Other notable changes

* Addresses vSphere Volume Attach limits ([#29881](https://github.com/kubernetes/kubernetes/pull/29881), [@dagnello](https://github.com/dagnello))
* Increase request timeout based on termination grace period ([#31275](https://github.com/kubernetes/kubernetes/pull/31275), [@dims](https://github.com/dims))
* Skip safe to detach check if node API object no longer exists ([#30737](https://github.com/kubernetes/kubernetes/pull/30737), [@saad-ali](https://github.com/saad-ali))
* Nodecontroller doesn't flip readiness on pods if kubeletVersion < 1.2.0 ([#30828](https://github.com/kubernetes/kubernetes/pull/30828), [@bprashanth](https://github.com/bprashanth))
* Update cadvisor to v0.23.9 to fix a problem where attempting to gather container filesystem usage statistics could result in corrupted devicemapper thin pool storage for Docker. ([#30307](https://github.com/kubernetes/kubernetes/pull/30307), [@sjenning](https://github.com/sjenning))



# v1.4.0-alpha.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-alpha.3/kubernetes.tar.gz) | `8055f0373e3b6bdee865749ef9bcfc765396a40f39ec2fa3cd31b675d1bbf5d9`

## Changelog since v1.4.0-alpha.2

### Action Required

* Moved init-container feature from alpha to beta. ([#31026](https://github.com/kubernetes/kubernetes/pull/31026), [@erictune](https://github.com/erictune))
    * Security Action Required:
    * This only applies to you if you use the PodSecurityPolicy feature.  You are using that feature if `kubectl get podsecuritypolicy` returns one or more objects.  If it returns an error, you are not using it.
    * If there are any pods with the key `pods.beta.kubernetes.io/init-containers`, then that pod may not have been filtered by the PodSecurityPolicy.  You should find such pods and either delete them or audit them to ensure they do not use features that you intend to be blocked by PodSecurityPolicy.
    * Explanation of Feature
    * In 1.3, an init container is specified with this annotation key
    * on the pod or pod template: `pods.alpha.kubernetes.io/init-containers`.
    * In 1.4, either that key or this key: `pods.beta.kubernetes.io/init-containers`,
    * can be used.
    * When you GET an object, you will see both annotation keys with the same values.
    * You can safely roll back from 1.4 to 1.3, and things with init-containers
    * will still work (pods, deployments, etc).
    * If you are running 1.3, only use the alpha annotation, or it may be lost when
    * rolling forward.
    * The status has moved from annotation key
    * `pods.beta.kubernetes.io/init-container-statuses` to
    * `pods.beta.kubernetes.io/init-container-statuses`.
    * Any code that inspects this annotation should be changed to use the new key.
    * State of Initialization will continue to be reported in both pods.alpha.kubernetes.io/initialized
    * and in `podStatus.conditions.{status: "True", type: Initialized}`
* Action required: federation-only: Please update your cluster name to be a valid DNS label. ([#30956](https://github.com/kubernetes/kubernetes/pull/30956), [@nikhiljindal](https://github.com/nikhiljindal))
    * Updating federation.v1beta1.Cluster API to disallow subdomains as valid cluster names. Only DNS labels are allowed as valid cluster names now.
* [Kubelet] Rename `--config` to `--pod-manifest-path`. `--config` is deprecated. ([#29999](https://github.com/kubernetes/kubernetes/pull/29999), [@mtaufen](https://github.com/mtaufen))

### Other notable changes

* rkt: Improve support for privileged pod (pod whose all containers are privileged)  ([#31286](https://github.com/kubernetes/kubernetes/pull/31286), [@yifan-gu](https://github.com/yifan-gu))
* The pod annotation `security.alpha.kubernetes.io/sysctls` now allows customization of namespaced and well isolated kernel parameters (sysctls), starting with `kernel.shm_rmid_forced`, `net.ipv4.ip_local_port_range` and `net.ipv4.tcp_syncookies` for Kubernetes 1.4. ([#27180](https://github.com/kubernetes/kubernetes/pull/27180), [@sttts](https://github.com/sttts))
    * The pod annotation  `security.alpha.kubernetes.io/unsafe-sysctls` allows customization of namespaced sysctls where isolation is unclear. Unsafe sysctls must be enabled at-your-own-risk on the kubelet with the `--experimental-allowed-unsafe-sysctls` flag. Future versions will improve on resource isolation and more sysctls will be considered safe.
* Increase request timeout based on termination grace period ([#31275](https://github.com/kubernetes/kubernetes/pull/31275), [@dims](https://github.com/dims))
* Fixed two issues of kubectl bash completion. ([#31135](https://github.com/kubernetes/kubernetes/pull/31135), [@xingzhou](https://github.com/xingzhou))
* Reduced size of fluentd images. ([#31239](https://github.com/kubernetes/kubernetes/pull/31239), [@aledbf](https://github.com/aledbf))
* support Azure data disk volume ([#29836](https://github.com/kubernetes/kubernetes/pull/29836), [@rootfs](https://github.com/rootfs))
* fix Openstack provider to allow more than one service port for lbaas v2 ([#30649](https://github.com/kubernetes/kubernetes/pull/30649), [@dagnello](https://github.com/dagnello))
* Add kubelet --network-plugin-mtu flag for MTU selection ([#30376](https://github.com/kubernetes/kubernetes/pull/30376), [@justinsb](https://github.com/justinsb))
* Let Services preserve client IPs and not double-hop from external LBs (alpha) ([#29409](https://github.com/kubernetes/kubernetes/pull/29409), [@girishkalele](https://github.com/girishkalele))
* [Kubelet] Optionally consume configuration from <node-name> named config maps ([#30090](https://github.com/kubernetes/kubernetes/pull/30090), [@mtaufen](https://github.com/mtaufen))
* [GarbageCollector] Allow per-resource default garbage collection behavior ([#30838](https://github.com/kubernetes/kubernetes/pull/30838), [@caesarxuchao](https://github.com/caesarxuchao))
* Action required: If you have a running federation control plane, you will have to ensure that for all federation resources, the corresponding namespace exists in federation control plane. ([#31139](https://github.com/kubernetes/kubernetes/pull/31139), [@nikhiljindal](https://github.com/nikhiljindal))
    * federation-apiserver now supports NamespaceLifecycle admission control, which is enabled by default. Set the --admission-control flag on the server to change that.
* Configure webhook ([#30923](https://github.com/kubernetes/kubernetes/pull/30923), [@Q-Lee](https://github.com/Q-Lee))
* Federated Ingress Controller ([#30419](https://github.com/kubernetes/kubernetes/pull/30419), [@quinton-hoole](https://github.com/quinton-hoole))
* Federation replicaset controller ([#29741](https://github.com/kubernetes/kubernetes/pull/29741), [@jianhuiz](https://github.com/jianhuiz))
* AWS: More ELB attributes via service annotations ([#30695](https://github.com/kubernetes/kubernetes/pull/30695), [@krancour](https://github.com/krancour))
* Impersonate user extra ([#30881](https://github.com/kubernetes/kubernetes/pull/30881), [@deads2k](https://github.com/deads2k))
* DNS, Heapster and UI are critical addons ([#30995](https://github.com/kubernetes/kubernetes/pull/30995), [@piosz](https://github.com/piosz))
* AWS: Support HTTP->HTTP mode for ELB ([#30563](https://github.com/kubernetes/kubernetes/pull/30563), [@knarz](https://github.com/knarz))
* kube-up: Allow IP restrictions for SSH and HTTPS API access on AWS. ([#27061](https://github.com/kubernetes/kubernetes/pull/27061), [@Naddiseo](https://github.com/Naddiseo))
* Add readyReplicas to replica sets ([#29481](https://github.com/kubernetes/kubernetes/pull/29481), [@kargakis](https://github.com/kargakis))
* The implicit registration of Prometheus metrics for request count and latency have been removed, and a plug-able interface was added. If you were using our client libraries in your own binaries and want these metrics, add the following to your imports in the main package: "k8s.io/pkg/client/metrics/prometheus".  ([#30638](https://github.com/kubernetes/kubernetes/pull/30638), [@krousey](https://github.com/krousey))
* Add support for --image-pull-policy to 'kubectl run' ([#30614](https://github.com/kubernetes/kubernetes/pull/30614), [@AdoHe](https://github.com/AdoHe))
* x509 authenticator: get groups from subject's organization field ([#30392](https://github.com/kubernetes/kubernetes/pull/30392), [@ericchiang](https://github.com/ericchiang))
* Add initial support for TokenFile to to the client config file. ([#29696](https://github.com/kubernetes/kubernetes/pull/29696), [@brendandburns](https://github.com/brendandburns))
* update kubectl help output for better organization ([#25524](https://github.com/kubernetes/kubernetes/pull/25524), [@AdoHe](https://github.com/AdoHe))
* daemonset controller should respect taints ([#31020](https://github.com/kubernetes/kubernetes/pull/31020), [@mikedanese](https://github.com/mikedanese))
* Implement TLS bootstrap for kubelet using `--experimental-bootstrap-kubeconfig`  (2nd take) ([#30922](https://github.com/kubernetes/kubernetes/pull/30922), [@yifan-gu](https://github.com/yifan-gu))
* rkt: Support subPath volume mounts feature ([#30934](https://github.com/kubernetes/kubernetes/pull/30934), [@yifan-gu](https://github.com/yifan-gu))
* Return container command exit codes in kubectl run/exec ([#26541](https://github.com/kubernetes/kubernetes/pull/26541), [@sttts](https://github.com/sttts))
* Fix kubectl describe to display a container's resource limit env vars as node allocatable when the limits are not set ([#29849](https://github.com/kubernetes/kubernetes/pull/29849), [@aveshagarwal](https://github.com/aveshagarwal))
* The `valueFrom.fieldRef.name` field on environment variables in pods and objects with pod templates now allows two additional fields to be used: ([#27880](https://github.com/kubernetes/kubernetes/pull/27880), [@smarterclayton](https://github.com/smarterclayton))
        * `spec.nodeName` will return the name of the node this pod is running on
        * `spec.serviceAccountName` will return the name of the service account this pod is running under
* Adding ImagePolicyWebhook admission controller. ([#30631](https://github.com/kubernetes/kubernetes/pull/30631), [@ecordell](https://github.com/ecordell))
* Validate involvedObject.Namespace matches event.Namespace ([#30533](https://github.com/kubernetes/kubernetes/pull/30533), [@liggitt](https://github.com/liggitt))
* allow group impersonation ([#30803](https://github.com/kubernetes/kubernetes/pull/30803), [@deads2k](https://github.com/deads2k))
* Always return command output for exec probes and kubelet RunInContainer ([#30731](https://github.com/kubernetes/kubernetes/pull/30731), [@ncdc](https://github.com/ncdc))
* Enable the garbage collector by default ([#30480](https://github.com/kubernetes/kubernetes/pull/30480), [@caesarxuchao](https://github.com/caesarxuchao))
* use valid_resources to replace kubectl.PossibleResourceTypes ([#30955](https://github.com/kubernetes/kubernetes/pull/30955), [@lojies](https://github.com/lojies))
* oidc auth provider: don't trim issuer URL ([#30944](https://github.com/kubernetes/kubernetes/pull/30944), [@ericchiang](https://github.com/ericchiang))
* Add a short `-n` for `kubectl --namespace` ([#30630](https://github.com/kubernetes/kubernetes/pull/30630), [@silasbw](https://github.com/silasbw))
* Federated secret controller ([#30669](https://github.com/kubernetes/kubernetes/pull/30669), [@kshafiee](https://github.com/kshafiee))
* Add Events for operation_executor to show status of mounts, failed/successful to show in describe events ([#27778](https://github.com/kubernetes/kubernetes/pull/27778), [@screeley44](https://github.com/screeley44))
* Alpha support for OpenAPI (aka. Swagger 2.0) specification served on /swagger.json (enabled by default)  ([#30233](https://github.com/kubernetes/kubernetes/pull/30233), [@mbohlool](https://github.com/mbohlool))
* Disable linux/ppc64le compilation by default ([#30659](https://github.com/kubernetes/kubernetes/pull/30659), [@ixdy](https://github.com/ixdy))
* Implement dynamic provisioning (beta) of PersistentVolumes via StorageClass ([#29006](https://github.com/kubernetes/kubernetes/pull/29006), [@jsafrane](https://github.com/jsafrane))
* Allow setting permission mode bits on secrets, configmaps and downwardAPI files ([#28936](https://github.com/kubernetes/kubernetes/pull/28936), [@rata](https://github.com/rata))
* Skip safe to detach check if node API object no longer exists ([#30737](https://github.com/kubernetes/kubernetes/pull/30737), [@saad-ali](https://github.com/saad-ali))
* The Kubelet now supports the `--require-kubeconfig` option which reads all client config from the provided `--kubeconfig` file and will cause the Kubelet to exit with error code 1 on error.  It also forces the Kubelet to use the server URL from the kubeconfig file rather than the  `--api-servers` flag.  Without this flag set, a failure to read the kubeconfig file would only result in a warning message. ([#30798](https://github.com/kubernetes/kubernetes/pull/30798), [@smarterclayton](https://github.com/smarterclayton))
    * In a future release, the value of this flag will be defaulted to `true`.
* Adding container image verification webhook API. ([#30241](https://github.com/kubernetes/kubernetes/pull/30241), [@Q-Lee](https://github.com/Q-Lee))
* Nodecontroller doesn't flip readiness on pods if kubeletVersion < 1.2.0 ([#30828](https://github.com/kubernetes/kubernetes/pull/30828), [@bprashanth](https://github.com/bprashanth))
* AWS: Handle kube-down case where the LaunchConfig is dangling ([#30816](https://github.com/kubernetes/kubernetes/pull/30816), [@zmerlynn](https://github.com/zmerlynn))
* kubectl will no longer do client-side defaulting on create and replace. ([#30250](https://github.com/kubernetes/kubernetes/pull/30250), [@krousey](https://github.com/krousey))
* Added warning msg for `kubectl get` ([#28352](https://github.com/kubernetes/kubernetes/pull/28352), [@vefimova](https://github.com/vefimova))
* Removed support for HPA in extensions client. ([#30504](https://github.com/kubernetes/kubernetes/pull/30504), [@piosz](https://github.com/piosz))
* Implement DisruptionController. ([#25921](https://github.com/kubernetes/kubernetes/pull/25921), [@mml](https://github.com/mml))
* [Kubelet] Check if kubelet is running as uid 0 ([#30466](https://github.com/kubernetes/kubernetes/pull/30466), [@vishh](https://github.com/vishh))
* Fix third party APIResource reporting ([#29724](https://github.com/kubernetes/kubernetes/pull/29724), [@brendandburns](https://github.com/brendandburns))
* speed up RC scaler ([#30383](https://github.com/kubernetes/kubernetes/pull/30383), [@deads2k](https://github.com/deads2k))
* Set pod state as "unknown" when CNI plugin fails ([#30137](https://github.com/kubernetes/kubernetes/pull/30137), [@nhlfr](https://github.com/nhlfr))
* Cluster Federation components can now be built and deployed using the make command. Please see federation/README.md for details. ([#29515](https://github.com/kubernetes/kubernetes/pull/29515), [@madhusudancs](https://github.com/madhusudancs))
* Adding events to federation control plane ([#30421](https://github.com/kubernetes/kubernetes/pull/30421), [@nikhiljindal](https://github.com/nikhiljindal))
* [kubelet] Introduce --protect-kernel-defaults flag to make the tunable behaviour configurable ([#27874](https://github.com/kubernetes/kubernetes/pull/27874), [@ingvagabund](https://github.com/ingvagabund))
* Add support for kube-up.sh to deploy Calico network policy to GCI masters ([#29037](https://github.com/kubernetes/kubernetes/pull/29037), [@matthewdupre](https://github.com/matthewdupre))
* Added 'kubectl top' command showing the resource usage metrics. ([#28844](https://github.com/kubernetes/kubernetes/pull/28844), [@mksalawa](https://github.com/mksalawa))
* Add basic audit logging ([#27087](https://github.com/kubernetes/kubernetes/pull/27087), [@soltysh](https://github.com/soltysh))
* Marked NodePhase deprecated. ([#30005](https://github.com/kubernetes/kubernetes/pull/30005), [@dchen1107](https://github.com/dchen1107))
* Name the job created by scheduledjob (sj) deterministically with sj's name and a hash of job's scheduled time. ([#30420](https://github.com/kubernetes/kubernetes/pull/30420), [@janetkuo](https://github.com/janetkuo))
* add metrics for workqueues ([#30296](https://github.com/kubernetes/kubernetes/pull/30296), [@deads2k](https://github.com/deads2k))
* Adding ingress resource to federation apiserver ([#30112](https://github.com/kubernetes/kubernetes/pull/30112), [@nikhiljindal](https://github.com/nikhiljindal))
* Update Dashboard UI to version v1.1.1 ([#30273](https://github.com/kubernetes/kubernetes/pull/30273), [@bryk](https://github.com/bryk))
* Update etcd 2.2 references to use 3.0.x ([#29399](https://github.com/kubernetes/kubernetes/pull/29399), [@timothysc](https://github.com/timothysc))
* HPA: ignore scale targets whose replica count is 0 ([#29212](https://github.com/kubernetes/kubernetes/pull/29212), [@sjenning](https://github.com/sjenning))
* Add total inodes to kubelet summary api ([#30231](https://github.com/kubernetes/kubernetes/pull/30231), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Updates required for juju kubernetes to use the tls-terminated etcd charm. ([#30104](https://github.com/kubernetes/kubernetes/pull/30104), [@mbruzek](https://github.com/mbruzek))
* Fix PVC.Status.Capacity and AccessModes after binding ([#29982](https://github.com/kubernetes/kubernetes/pull/29982), [@jsafrane](https://github.com/jsafrane))
* allow a read-only rbd image mounted by multiple pods ([#29622](https://github.com/kubernetes/kubernetes/pull/29622), [@rootfs](https://github.com/rootfs))
* [kubelet] Auto-discover node IP if neither cloud provider exists and IP is not explicitly specified ([#29907](https://github.com/kubernetes/kubernetes/pull/29907), [@luxas](https://github.com/luxas))
* kubectl config set-crentials: add arguments for auth providers ([#30007](https://github.com/kubernetes/kubernetes/pull/30007), [@ericchiang](https://github.com/ericchiang))
* Scheduledjob controller ([#29137](https://github.com/kubernetes/kubernetes/pull/29137), [@janetkuo](https://github.com/janetkuo))
* add subjectaccessreviews resource ([#20573](https://github.com/kubernetes/kubernetes/pull/20573), [@deads2k](https://github.com/deads2k))
* AWS/GCE: Rework use of master name ([#30047](https://github.com/kubernetes/kubernetes/pull/30047), [@zmerlynn](https://github.com/zmerlynn))
* Add density (batch pods creation latency and resource) and resource performance tests to `test-e2e-node' built for Linux only ([#30026](https://github.com/kubernetes/kubernetes/pull/30026), [@coufon](https://github.com/coufon))
* Clean up items from moving local cluster setup guides ([#30035](https://github.com/kubernetes/kubernetes/pull/30035), [@pwittrock](https://github.com/pwittrock))
* federation: Adding secret API ([#29138](https://github.com/kubernetes/kubernetes/pull/29138), [@kshafiee](https://github.com/kshafiee))
* Introducing ScheduledJobs as described in [the proposal](docs/proposals/scheduledjob.md) as part of `batch/v2alpha1` version (experimental feature). ([#25816](https://github.com/kubernetes/kubernetes/pull/25816), [@soltysh](https://github.com/soltysh))
* Node disk pressure should induce image gc ([#29880](https://github.com/kubernetes/kubernetes/pull/29880), [@derekwaynecarr](https://github.com/derekwaynecarr))
* oidc authentication plugin: don't trim issuer URLs with trailing slashes ([#29860](https://github.com/kubernetes/kubernetes/pull/29860), [@ericchiang](https://github.com/ericchiang))
* Allow leading * in ingress hostname ([#29204](https://github.com/kubernetes/kubernetes/pull/29204), [@aledbf](https://github.com/aledbf))
* Rewrite service controller to apply best controller pattern ([#25189](https://github.com/kubernetes/kubernetes/pull/25189), [@mfanjie](https://github.com/mfanjie))
* Fix issue with kubectl annotate when --resource-version is provided. ([#29319](https://github.com/kubernetes/kubernetes/pull/29319), [@juanvallejo](https://github.com/juanvallejo))
* Reverted conversion of influx-db to Pet Set, it is now a Replication Controller. ([#30080](https://github.com/kubernetes/kubernetes/pull/30080), [@jszczepkowski](https://github.com/jszczepkowski))
* rbac validation: rules can't combine non-resource URLs and regular resources ([#29930](https://github.com/kubernetes/kubernetes/pull/29930), [@ericchiang](https://github.com/ericchiang))
* VSAN support for VSphere Volume Plugin ([#29172](https://github.com/kubernetes/kubernetes/pull/29172), [@abrarshivani](https://github.com/abrarshivani))
* Addresses vSphere Volume Attach limits ([#29881](https://github.com/kubernetes/kubernetes/pull/29881), [@dagnello](https://github.com/dagnello))
* allow restricting subresource access ([#29988](https://github.com/kubernetes/kubernetes/pull/29988), [@deads2k](https://github.com/deads2k))
* Add density (batch pods creation latency and resource) and resource performance tests to `test-e2e-node' ([#29764](https://github.com/kubernetes/kubernetes/pull/29764), [@coufon](https://github.com/coufon))
* Allow Secret & ConfigMap keys to contain caps, dots, and underscores ([#25458](https://github.com/kubernetes/kubernetes/pull/25458), [@errm](https://github.com/errm))
* allow watching old resources with kubectl ([#27392](https://github.com/kubernetes/kubernetes/pull/27392), [@sjenning](https://github.com/sjenning))
* azure: kube-up respects AZURE_RESOURCE_GROUP ([#28700](https://github.com/kubernetes/kubernetes/pull/28700), [@colemickens](https://github.com/colemickens))
* Modified influxdb petset to provision persistent  volume. ([#28840](https://github.com/kubernetes/kubernetes/pull/28840), [@jszczepkowski](https://github.com/jszczepkowski))
* Allow service names up to 63 characters (RFC 1035) ([#29523](https://github.com/kubernetes/kubernetes/pull/29523), [@fraenkel](https://github.com/fraenkel))
* Change eviction policies in NodeController: ([#28897](https://github.com/kubernetes/kubernetes/pull/28897), [@gmarek](https://github.com/gmarek))
    * - add a "partialDisruption" mode, when more than 33% of Nodes in the zone are not Ready
    * - add "fullDisruption" mode, when all Nodes in the zone are not Ready
    * Eviction behavior depends on the mode in which NodeController is operating:
    * - if the new state is "partialDisruption" or "fullDisruption" we call a user defined function that returns a new QPS to use (default 1/10 of the default rate, and the default rate respectively),
    * - if the new state is "normal" we resume normal operation (go back to default limiter settings),
    * - if all zones in the cluster are in "fullDisruption" state we stop all evictions.
* Add a flag for `kubectl expose`to set ClusterIP and allow headless services ([#28239](https://github.com/kubernetes/kubernetes/pull/28239), [@ApsOps](https://github.com/ApsOps))
* Add support to quota pvc storage requests ([#28636](https://github.com/kubernetes/kubernetes/pull/28636), [@derekwaynecarr](https://github.com/derekwaynecarr))



# v1.3.5

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.5/kubernetes.tar.gz) | `46be88ce927124f7cef7e280720b42c63051086880b7ebdba298b561dbe19f82`

## Changelog since v1.3.4

### Other notable changes

* Update Dashboard UI to version v1.1.1 ([#30273](https://github.com/kubernetes/kubernetes/pull/30273), [@bryk](https://github.com/bryk))
* allow restricting subresource access ([#30001](https://github.com/kubernetes/kubernetes/pull/30001), [@deads2k](https://github.com/deads2k))
* Fix PVC.Status.Capacity and AccessModes after binding ([#29982](https://github.com/kubernetes/kubernetes/pull/29982), [@jsafrane](https://github.com/jsafrane))
* oidc authentication plugin: don't trim issuer URLs with trailing slashes ([#29860](https://github.com/kubernetes/kubernetes/pull/29860), [@ericchiang](https://github.com/ericchiang))
* network/cni: Bring up the `lo` interface for rkt ([#29310](https://github.com/kubernetes/kubernetes/pull/29310), [@euank](https://github.com/euank))
* Fixing kube-up for CVM masters. ([#29140](https://github.com/kubernetes/kubernetes/pull/29140), [@maisem](https://github.com/maisem))



# v1.3.4

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.4/kubernetes.tar.gz) | `818acc1a8ba61cff434d4c0c5aa3d342d06e6907b565cfd8651b8cfcf3f0a1e6`

## Changelog since v1.3.3

### Other notable changes

* NetworkPolicy cherry-pick 1.3 ([#29556](https://github.com/kubernetes/kubernetes/pull/29556), [@caseydavenport](https://github.com/caseydavenport))
* Allow mounts to run in parallel for non-attachable volumes ([#28939](https://github.com/kubernetes/kubernetes/pull/28939), [@saad-ali](https://github.com/saad-ali))
* add enhanced volume and mount logging for block devices ([#24797](https://github.com/kubernetes/kubernetes/pull/24797), [@screeley44](https://github.com/screeley44))
* kube-up: increase download timeout for kubernetes.tar.gz ([#29426](https://github.com/kubernetes/kubernetes/pull/29426), [@justinsb](https://github.com/justinsb))
* Fix RBAC authorizer of ServiceAccount ([#29071](https://github.com/kubernetes/kubernetes/pull/29071), [@albatross0](https://github.com/albatross0))
* Update docker engine-api to dea108d3aa ([#29144](https://github.com/kubernetes/kubernetes/pull/29144), [@ronnielai](https://github.com/ronnielai))
* Assume volume is detached if node doesn't exist ([#29485](https://github.com/kubernetes/kubernetes/pull/29485), [@saad-ali](https://github.com/saad-ali))
* Make PD E2E Tests Wait for Detach to Prevent Kernel Errors ([#29031](https://github.com/kubernetes/kubernetes/pull/29031), [@saad-ali](https://github.com/saad-ali))
* Fix "PVC Volume not detached if pod deleted via namespace deletion" issue ([#29077](https://github.com/kubernetes/kubernetes/pull/29077), [@saad-ali](https://github.com/saad-ali))
* append an abac rule for $KUBE_USER. ([#29164](https://github.com/kubernetes/kubernetes/pull/29164), [@cjcullen](https://github.com/cjcullen))



# v1.4.0-alpha.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-alpha.2/kubernetes.tar.gz) | `787ce63a5149a1cb47d14c55450172e3a045d85349682d2e17ff492de9e415b9`

## Changelog since v1.4.0-alpha.1

### Action Required

* Federation API server kubeconfig secret consumed by federation-controller-manager has a new name. ([#28938](https://github.com/kubernetes/kubernetes/pull/28938), [@madhusudancs](https://github.com/madhusudancs))
    * If you are upgrading your Cluster Federation components from v1.3.x, please run this command to migrate the federation-apiserver-secret to federation-apiserver-kubeconfig serect;
    * $ kubectl --namespace=federation get secret federation-apiserver-secret -o json | sed 's/federation-apiserver-secret/federation-apiserver-kubeconfig/g' | kubectl create -f -
    * You might also want to delete the old secret using this command:
    * $ kubectl delete secret --namespace=federation federation-apiserver-secret
* Stop eating panics ([#28800](https://github.com/kubernetes/kubernetes/pull/28800), [@lavalamp](https://github.com/lavalamp))

### Other notable changes

* Add API for StorageClasses ([#29694](https://github.com/kubernetes/kubernetes/pull/29694), [@childsb](https://github.com/childsb))
* Fix kubectl help command ([#29737](https://github.com/kubernetes/kubernetes/pull/29737), [@andreykurilin](https://github.com/andreykurilin))
* add shorthand cm for configmaps ([#29652](https://github.com/kubernetes/kubernetes/pull/29652), [@lojies](https://github.com/lojies))
* Bump cadvisor dependencies to latest head.  ([#29492](https://github.com/kubernetes/kubernetes/pull/29492), [@Random-Liu](https://github.com/Random-Liu))
* If a service of type node port declares multiple ports, quota on "services.nodeports" will charge for each port in the service. ([#29457](https://github.com/kubernetes/kubernetes/pull/29457), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Add an Azure CloudProvider Implementation ([#28821](https://github.com/kubernetes/kubernetes/pull/28821), [@colemickens](https://github.com/colemickens))
* Add support for kubectl create quota command ([#28351](https://github.com/kubernetes/kubernetes/pull/28351), [@sttts](https://github.com/sttts))
* Assume volume is detached if node doesn't exist ([#29485](https://github.com/kubernetes/kubernetes/pull/29485), [@saad-ali](https://github.com/saad-ali))
* kube-up: increase download timeout for kubernetes.tar.gz ([#29426](https://github.com/kubernetes/kubernetes/pull/29426), [@justinsb](https://github.com/justinsb))
* Allow multiple APIs to register for the same API Group ([#28414](https://github.com/kubernetes/kubernetes/pull/28414), [@brendandburns](https://github.com/brendandburns))
* Fix a problem with multiple APIs clobbering each other in registration. ([#28431](https://github.com/kubernetes/kubernetes/pull/28431), [@brendandburns](https://github.com/brendandburns))
* Removing images with multiple tags ([#29316](https://github.com/kubernetes/kubernetes/pull/29316), [@ronnielai](https://github.com/ronnielai))
* add enhanced volume and mount logging for block devices ([#24797](https://github.com/kubernetes/kubernetes/pull/24797), [@screeley44](https://github.com/screeley44))
* append an abac rule for $KUBE_USER. ([#29164](https://github.com/kubernetes/kubernetes/pull/29164), [@cjcullen](https://github.com/cjcullen))
* add tokenreviews endpoint to implement webhook ([#28788](https://github.com/kubernetes/kubernetes/pull/28788), [@deads2k](https://github.com/deads2k))
* Fix "PVC Volume not detached if pod deleted via namespace deletion" issue ([#29077](https://github.com/kubernetes/kubernetes/pull/29077), [@saad-ali](https://github.com/saad-ali))
* Allow mounts to run in parallel for non-attachable volumes ([#28939](https://github.com/kubernetes/kubernetes/pull/28939), [@saad-ali](https://github.com/saad-ali))
* Fix working_set calculation in kubelet ([#29153](https://github.com/kubernetes/kubernetes/pull/29153), [@vishh](https://github.com/vishh))
* Fix RBAC authorizer of ServiceAccount ([#29071](https://github.com/kubernetes/kubernetes/pull/29071), [@albatross0](https://github.com/albatross0))
* kubectl proxy changed to now allow urls to pods with "attach" or "exec" in the pod name ([#28765](https://github.com/kubernetes/kubernetes/pull/28765), [@nhlfr](https://github.com/nhlfr))
* AWS: Added experimental option to skip zone check ([#28417](https://github.com/kubernetes/kubernetes/pull/28417), [@kevensen](https://github.com/kevensen))
* Ubuntu: Enable ssh compression when downloading binaries during cluster creation ([#26746](https://github.com/kubernetes/kubernetes/pull/26746), [@MHBauer](https://github.com/MHBauer))
* Add extensions/replicaset to federation-apiserver ([#24764](https://github.com/kubernetes/kubernetes/pull/24764), [@jianhuiz](https://github.com/jianhuiz))
* federation: Adding namespaces API ([#26298](https://github.com/kubernetes/kubernetes/pull/26298), [@nikhiljindal](https://github.com/nikhiljindal))
* Improve quota controller performance by eliminating unneeded list calls ([#29134](https://github.com/kubernetes/kubernetes/pull/29134), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Make Daemonset use GeneralPredicates ([#28803](https://github.com/kubernetes/kubernetes/pull/28803), [@lukaszo](https://github.com/lukaszo))
* Update docker engine-api to dea108d3aa ([#29144](https://github.com/kubernetes/kubernetes/pull/29144), [@ronnielai](https://github.com/ronnielai))
* Fixing kube-up for CVM masters. ([#29140](https://github.com/kubernetes/kubernetes/pull/29140), [@maisem](https://github.com/maisem))
* Fix logrotate config on GCI ([#29139](https://github.com/kubernetes/kubernetes/pull/29139), [@adityakali](https://github.com/adityakali))
* GCE bring-up: Differentiate NODE_TAGS from NODE_INSTANCE_PREFIX ([#29141](https://github.com/kubernetes/kubernetes/pull/29141), [@zmerlynn](https://github.com/zmerlynn))
* hyperkube: fix build for 3rd party registry (again) ([#28489](https://github.com/kubernetes/kubernetes/pull/28489), [@liyimeng](https://github.com/liyimeng))
* Detect flakes in PR builder e2e runs ([#27898](https://github.com/kubernetes/kubernetes/pull/27898), [@lavalamp](https://github.com/lavalamp))
* Remove examples moved to docs site ([#23513](https://github.com/kubernetes/kubernetes/pull/23513), [@erictune](https://github.com/erictune))
* Do not query the metadata server to find out if running on GCE.  Retry metadata server query for gcr if running on gce. ([#28871](https://github.com/kubernetes/kubernetes/pull/28871), [@vishh](https://github.com/vishh))
* Change maxsize to size in logrotate. ([#29128](https://github.com/kubernetes/kubernetes/pull/29128), [@bprashanth](https://github.com/bprashanth))
* Change setting "kubectl --record=false" to stop updating the change-cause when a previous change-cause is found. ([#28234](https://github.com/kubernetes/kubernetes/pull/28234), [@damemi](https://github.com/damemi))
* Add "kubectl --overwrite" flag to automatically resolve conflicts between the modified and live configuration using values from the modified configuration. ([#26136](https://github.com/kubernetes/kubernetes/pull/26136), [@AdoHe](https://github.com/AdoHe))
* Make discovery summarizer call servers in parallel ([#26705](https://github.com/kubernetes/kubernetes/pull/26705), [@nebril](https://github.com/nebril))
* Don't recreate lb cloud resources on kcm restart ([#29082](https://github.com/kubernetes/kubernetes/pull/29082), [@bprashanth](https://github.com/bprashanth))
* List all nodes and occupy cidr map before starting allocations ([#29062](https://github.com/kubernetes/kubernetes/pull/29062), [@bprashanth](https://github.com/bprashanth))
* Fix GPU resource validation ([#28743](https://github.com/kubernetes/kubernetes/pull/28743), [@therc](https://github.com/therc))
* Make PD E2E Tests Wait for Detach to Prevent Kernel Errors ([#29031](https://github.com/kubernetes/kubernetes/pull/29031), [@saad-ali](https://github.com/saad-ali))
* Scale kube-proxy conntrack limits by cores (new default behavior) ([#28876](https://github.com/kubernetes/kubernetes/pull/28876), [@thockin](https://github.com/thockin))
* [Kubelet] Improving QOS in kubelet by introducing QoS level Cgroups - `--cgroups-per-qos` ([#27853](https://github.com/kubernetes/kubernetes/pull/27853), [@dubstack](https://github.com/dubstack))
* AWS: Add ap-south-1 to list of known AWS regions ([#28428](https://github.com/kubernetes/kubernetes/pull/28428), [@justinsb](https://github.com/justinsb))
* Add RELEASE_INFRA_PUSH related code to support pushes from kubernetes/release. ([#28922](https://github.com/kubernetes/kubernetes/pull/28922), [@david-mcmahon](https://github.com/david-mcmahon))
* Fix watch cache filtering ([#28966](https://github.com/kubernetes/kubernetes/pull/28966), [@liggitt](https://github.com/liggitt))
* Deprecate deleting-pods-burst ControllerManager flag ([#28882](https://github.com/kubernetes/kubernetes/pull/28882), [@gmarek](https://github.com/gmarek))
* Add support for terminal resizing for exec, attach, and run. Note that for Docker, exec sessions ([#25273](https://github.com/kubernetes/kubernetes/pull/25273), [@ncdc](https://github.com/ncdc))
    * inherit the environment from the primary process, so if the container was created with tty=false,
    * that means the exec session's TERM variable will default to "dumb". Users can override this by
    * setting TERM=xterm (or whatever is appropriate) to get the correct "smart" terminal behavior.
* Implement alpha version of PreferAvoidPods ([#20699](https://github.com/kubernetes/kubernetes/pull/20699), [@jiangyaoguo](https://github.com/jiangyaoguo))
* Retry when apiserver fails to listen on insecure port ([#28797](https://github.com/kubernetes/kubernetes/pull/28797), [@aaronlevy](https://github.com/aaronlevy))
* Add SSH_OPTS to config ssh and scp port ([#28872](https://github.com/kubernetes/kubernetes/pull/28872), [@lojies](https://github.com/lojies))
* kube-up: install new Docker pre-requisite (libltdl7) when not in image ([#28745](https://github.com/kubernetes/kubernetes/pull/28745), [@justinsb](https://github.com/justinsb))
* Separate rate limiters for Pod evictions for different zones in NodeController ([#28843](https://github.com/kubernetes/kubernetes/pull/28843), [@gmarek](https://github.com/gmarek))
* Add --quiet to hide the 'waiting for pods to be running' message in kubectl run ([#28801](https://github.com/kubernetes/kubernetes/pull/28801), [@janetkuo](https://github.com/janetkuo))
* Controllers doesn't take any actions when being deleted. ([#27438](https://github.com/kubernetes/kubernetes/pull/27438), [@gmarek](https://github.com/gmarek))
* Add "deploy" abbrev for deployments to kubectl ([#24087](https://github.com/kubernetes/kubernetes/pull/24087), [@Frostman](https://github.com/Frostman))
* --no-header available now for custom-column ([#26696](https://github.com/kubernetes/kubernetes/pull/26696), [@gitfred](https://github.com/gitfred))



# v1.3.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.3/kubernetes.tar.gz) | `a92a74a0d3f7d02d01ac2c8dfb5ee2e97b0485819e77b2110eb7c6b7c782478c`

## Changelog since v1.3.2

### Other notable changes

* Removing images with multiple tags ([#29316](https://github.com/kubernetes/kubernetes/pull/29316), [@ronnielai](https://github.com/ronnielai))
* kubectl: don't display an empty list when trying to get a single resource that isn't found ([#28294](https://github.com/kubernetes/kubernetes/pull/28294), [@ncdc](https://github.com/ncdc))
* Fix working_set calculation in kubelet ([#29154](https://github.com/kubernetes/kubernetes/pull/29154), [@vishh](https://github.com/vishh))
* Don't delete affinity when endpoints are empty ([#28655](https://github.com/kubernetes/kubernetes/pull/28655), [@freehan](https://github.com/freehan))
* GCE bring-up: Differentiate NODE_TAGS from NODE_INSTANCE_PREFIX ([#29141](https://github.com/kubernetes/kubernetes/pull/29141), [@zmerlynn](https://github.com/zmerlynn))
* Fix logrotate config on GCI ([#29139](https://github.com/kubernetes/kubernetes/pull/29139), [@adityakali](https://github.com/adityakali))
* Do not query the metadata server to find out if running on GCE.  Retry metadata server query for gcr if running on gce. ([#28871](https://github.com/kubernetes/kubernetes/pull/28871), [@vishh](https://github.com/vishh))
* Fix GPU resource validation ([#28743](https://github.com/kubernetes/kubernetes/pull/28743), [@therc](https://github.com/therc))
* Scale kube-proxy conntrack limits by cores (new default behavior) ([#28876](https://github.com/kubernetes/kubernetes/pull/28876), [@thockin](https://github.com/thockin))
* Don't recreate lb cloud resources on kcm restart ([#29082](https://github.com/kubernetes/kubernetes/pull/29082), [@bprashanth](https://github.com/bprashanth))

## Known Issues

There are a number of known issues that have been found and are being worked on.
Please be aware of them as you test your workloads.

* PVC Volume not detached if pod deleted via namespace deletion ([29051](https://github.com/kubernetes/kubernetes/issues/29051))
* Google Compute Engine PD Detach fails if node no longer exists ([29358](https://github.com/kubernetes/kubernetes/issues/29358))
* Mounting (only 'default-token') volume takes a long time when creating a batch of pods  (parallelization issue) ([28616](https://github.com/kubernetes/kubernetes/issues/28616))
* Error while tearing down pod, "device or resource busy" on service account secret ([28750](https://github.com/kubernetes/kubernetes/issues/28750))



# v1.3.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.2/kubernetes.tar.gz) | `f46664d04dc2966c77d8727bba57f57b5f917572` | `1a5b0639941054585d0432dd5ce3abc7`

## Changelog since v1.3.1

### Other notable changes

* List all nodes and occupy cidr map before starting allocations ([#29062](https://github.com/kubernetes/kubernetes/pull/29062), [@bprashanth](https://github.com/bprashanth))
* Fix watch cache filtering ([#28968](https://github.com/kubernetes/kubernetes/pull/28968), [@liggitt](https://github.com/liggitt))
* Lock all possible kubecfg files at the beginning of ModifyConfig. ([#28232](https://github.com/kubernetes/kubernetes/pull/28232), [@cjcullen](https://github.com/cjcullen))



# v1.3.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3.0/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.1/kubernetes.tar.gz) | `5645b12beda22137204439de8260c62c9925f89b` | `ae6e9902ec70c1322d9a0a29ef385190`

## Changelog since v1.3.0

### Other notable changes

* Fix watch cache filtering ([#29046](https://github.com/kubernetes/kubernetes/pull/29046), [@liggitt](https://github.com/liggitt))



# v1.2.6

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.6/kubernetes.tar.gz) | `50023455d00af52c41a7158b4bd117b2dfd4a100` | `cf0411bcb620eb13b08b93578efffc43`

## Changelog since v1.2.5

### Other notable changes

* Fix watch cache filtering ([#28967](https://github.com/kubernetes/kubernetes/pull/28967), [@liggitt](https://github.com/liggitt))
* Fix problems with container restarts and flocker ([#25874](https://github.com/kubernetes/kubernetes/pull/25874), [@simonswine](https://github.com/simonswine))



# v1.4.0-alpha.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.4.0-alpha.1/kubernetes.tar.gz) | `11a199208c5164a291c1767a1b9e64e45fdea747` | `334f349daf9268d8ac091d7fcc8e4626`

## Changelog since v1.3.0

### Experimental Features

* An alpha implementation of the TLS bootstrap API described in docs/proposals/kubelet-tls-bootstrap.md. ([#25562](https://github.com/kubernetes/kubernetes/pull/25562), [@gtank](https://github.com/gtank))

### Action Required

* [kubelet] Allow opting out of automatic cloud provider detection in kubelet. By default kubelet will auto-detect cloud providers ([#28258](https://github.com/kubernetes/kubernetes/pull/28258), [@vishh](https://github.com/vishh))
* If you use one of the kube-dns replication controller manifest in `cluster/saltbase/salt/kube-dns`, i.e. `cluster/saltbase/salt/kube-dns/{skydns-rc.yaml.base,skydns-rc.yaml.in}`, either substitute one of `__PILLAR__FEDERATIONS__DOMAIN__MAP__` or `{{ pillar['federations_domain_map'] }}` with the corresponding federation name to domain name value or remove them if you do not support cluster federation at this time. If you plan to substitute the parameter with its value, here is an example for `{{ pillar['federations_domain_map'] }` ([#28132](https://github.com/kubernetes/kubernetes/pull/28132), [@madhusudancs](https://github.com/madhusudancs))
    * pillar['federations_domain_map'] = "- --federations=myfederation=federation.test"
    * where `myfederation` is the name of the federation and `federation.test` is the domain name registered for the federation.
* Proportionally scale paused and rolling deployments ([#20273](https://github.com/kubernetes/kubernetes/pull/20273), [@kargakis](https://github.com/kargakis))

### Other notable changes

* Support --all-namespaces in kubectl describe ([#26315](https://github.com/kubernetes/kubernetes/pull/26315), [@dims](https://github.com/dims))
* Add checks in Create and Update Cgroup methods ([#28566](https://github.com/kubernetes/kubernetes/pull/28566), [@dubstack](https://github.com/dubstack))
* Update coreos node e2e image to a version that uses cgroupfs ([#28661](https://github.com/kubernetes/kubernetes/pull/28661), [@dubstack](https://github.com/dubstack))
* Don't delete affinity when endpoints are empty ([#28655](https://github.com/kubernetes/kubernetes/pull/28655), [@freehan](https://github.com/freehan))
* Enable memory based pod evictions by default on the kubelet.   ([#28607](https://github.com/kubernetes/kubernetes/pull/28607), [@derekwaynecarr](https://github.com/derekwaynecarr))
    * Trigger pod eviction when available memory falls below 100Mi.
* Enable extensions/v1beta1/NetworkPolicy by default ([#28549](https://github.com/kubernetes/kubernetes/pull/28549), [@caseydavenport](https://github.com/caseydavenport))
* MESOS: Support a pre-installed km binary at a well known, agent-local path ([#28447](https://github.com/kubernetes/kubernetes/pull/28447), [@k82cn](https://github.com/k82cn))
* kubectl should print usage at the bottom ([#25640](https://github.com/kubernetes/kubernetes/pull/25640), [@dims](https://github.com/dims))
* A new command "kubectl config get-contexts" has been added. ([#25463](https://github.com/kubernetes/kubernetes/pull/25463), [@asalkeld](https://github.com/asalkeld))
* kubectl: ignore only update conflicts in the scaler ([#27048](https://github.com/kubernetes/kubernetes/pull/27048), [@kargakis](https://github.com/kargakis))
* Declare out of disk when there is no free inodes ([#28176](https://github.com/kubernetes/kubernetes/pull/28176), [@ronnielai](https://github.com/ronnielai))
* Includes the number of free inodes in stat summary ([#28173](https://github.com/kubernetes/kubernetes/pull/28173), [@ronnielai](https://github.com/ronnielai))
* kubectl: don't display an empty list when trying to get a single resource that isn't found ([#28294](https://github.com/kubernetes/kubernetes/pull/28294), [@ncdc](https://github.com/ncdc))
* Graceful deletion bumps object's generation ([#27269](https://github.com/kubernetes/kubernetes/pull/27269), [@gmarek](https://github.com/gmarek))
* Allow specifying secret data using strings ([#28263](https://github.com/kubernetes/kubernetes/pull/28263), [@liggitt](https://github.com/liggitt))
* kubectl help now provides "Did you mean this?" suggestions for typo/invalid command names. ([#27049](https://github.com/kubernetes/kubernetes/pull/27049), [@andreykurilin](https://github.com/andreykurilin))
* Lock all possible kubecfg files at the beginning of ModifyConfig. ([#28232](https://github.com/kubernetes/kubernetes/pull/28232), [@cjcullen](https://github.com/cjcullen))
* Enable HTTP2 by default ([#28114](https://github.com/kubernetes/kubernetes/pull/28114), [@timothysc](https://github.com/timothysc))
* Influxdb migrated to PetSet and PersistentVolumes. ([#28109](https://github.com/kubernetes/kubernetes/pull/28109), [@jszczepkowski](https://github.com/jszczepkowski))
* Change references to gs://kubernetes-release/ci ([#28193](https://github.com/kubernetes/kubernetes/pull/28193), [@zmerlynn](https://github.com/zmerlynn))
* Build: Add KUBE_GCS_RELEASE_BUCKET_MIRROR option to push-ci-build.sh ([#28172](https://github.com/kubernetes/kubernetes/pull/28172), [@zmerlynn](https://github.com/zmerlynn))
* Skip multi-zone e2e tests unless provider is GCE, GKE or AWS ([#27871](https://github.com/kubernetes/kubernetes/pull/27871), [@lukaszo](https://github.com/lukaszo))
* Making DHCP_OPTION_SET_ID creation optional ([#27278](https://github.com/kubernetes/kubernetes/pull/27278), [@activars](https://github.com/activars))
* Convert service account token controller to use a work queue ([#23858](https://github.com/kubernetes/kubernetes/pull/23858), [@liggitt](https://github.com/liggitt))
* Adding OWNERS for federation ([#28042](https://github.com/kubernetes/kubernetes/pull/28042), [@nikhiljindal](https://github.com/nikhiljindal))
* Modifying the default container GC policy parameters ([#27881](https://github.com/kubernetes/kubernetes/pull/27881), [@ronnielai](https://github.com/ronnielai))
* Kubelet can retrieve host IP even when apiserver has not been contacted ([#27508](https://github.com/kubernetes/kubernetes/pull/27508), [@aaronlevy](https://github.com/aaronlevy))
* Add the Patch method to the generated clientset. ([#27293](https://github.com/kubernetes/kubernetes/pull/27293), [@caesarxuchao](https://github.com/caesarxuchao))
* let patch use --local flag like `kubectl set image` ([#26722](https://github.com/kubernetes/kubernetes/pull/26722), [@deads2k](https://github.com/deads2k))
* enable recursive processing in kubectl edit ([#25085](https://github.com/kubernetes/kubernetes/pull/25085), [@metral](https://github.com/metral))
* Image GC logic should compensate for reserved blocks ([#27996](https://github.com/kubernetes/kubernetes/pull/27996), [@ronnielai](https://github.com/ronnielai))
* Bump minimum API version for docker to 1.21 ([#27208](https://github.com/kubernetes/kubernetes/pull/27208), [@yujuhong](https://github.com/yujuhong))
* Adding lock files for kubeconfig updating ([#28034](https://github.com/kubernetes/kubernetes/pull/28034), [@krousey](https://github.com/krousey))


# v1.3.0

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0/kubernetes.tar.gz) | `88249c443d438666928379aa7fe865b389ed72ea` | `9270f001aef8c03ff5db63456ca9eecc`

## Highlights

* Authorization:
  * **Alpha** RBAC authorization API group
* Federation
  * federation api group is now **beta**
  * Services from all federated clusters are now registered in Cloud DNS (AWS and GCP).
* Stateful Apps:
  * **alpha** PetSets manage stateful apps
  * **alpha** Init containers provide one-time setup for stateful containers
* Updating:
  * Retry Pod/RC updates in kubectl rolling-update.
  * Stop 'kubectl drain' deleting pods with local storage.
  * Add `kubectl rollout status`
* Security/Auth
  * L7 LB controller and disk attach controllers run on master, so nodes do not need those privileges.
  * Setting TLS1.2 minimum
  * `kubectl create secret tls` command
  * Webhook Token Authenticator
  * **beta** PodSecurityPolicy objects limits use of security-sensitive features by pods.
* Kubectl
  * Display line number on JSON errors
  * Add flag -t as shorthand for --tty
* Resources
  * Improved node stability by *optionally* evicting pods upon memory pressure - [Design Doc](https://github.com/kubernetes/kubernetes/blob/release-1.3/docs/proposals/kubelet-eviction.md)
  * **alpha**: NVIDIA GPU support ([#24836](https://github.com/kubernetes/kubernetes/pull/24836), [@therc](https://github.com/therc))
  * Adding loadBalancer services and nodeports services to quota system

## Known Issues and Important Steps before Upgrading

The following versions of Docker Engine are supported - *[v1.10](https://github.com/kubernetes/kubernetes/issues/19720)*, *[v1.11](https://github.com/kubernetes/kubernetes/issues/23397)*
Although *v1.9* is still compatible, we recommend upgrading to one of the supported versions.
All prior versions of docker will not be supported.

#### ThirdPartyResource

If you use ThirdPartyResource objects, they have moved from being namespaced-scoped to be cluster-scoped. Before upgrading to 1.3.0, export and delete any existing ThirdPartyResource objects using a 1.2.x client:

kubectl get thirdpartyresource --all-namespaces -o yaml > tprs.yaml
kubectl delete -f tprs.yaml

After upgrading to 1.3.0, re-register the third party resource objects at the root scope (using a 1.3 server and client):

kubectl create -f tprs.yaml

#### kubectl

Kubectl flag `--container-port` flag is deprecated: it will be removed in the future, please use `--target-port` instead.

#### kubernetes Core Known Issues

- Kube Proxy crashes infrequently due to a docker bug ([#24000](https://github.com/docker/docker/issues/24000))
  - This issue can be resolved by restarting docker daemon
- CORS works only in insecure mode ([#24086](https://github.com/kubernetes/kubernetes/issues/24086))
- Persistent volume claims gets added incorrectly after being deleted under stress. Happens very infrequently. ([#26082](https://github.com/kubernetes/kubernetes/issues/26082))

#### Docker runtime Known Issues

- Kernel crash with Aufs storage driver on Debian Jessie ([#27885](https://github.com/kubernetes/kubernetes/issues/27885))
  - Consider running the *new* [kubernetes node problem detector](https://github.com/kubernetes/node-problem-detector) to identify this (and other) kernel issues automatically.

- File descriptors are leaked in docker v1.11 ([#275](https://github.com/docker/containerd/issues/275))
- Additional memory overhead per container in docker v1.11 ([#21737](https://github.com/docker/docker/issues/21737))
- [List of upstream fixes](https://github.com/docker/docker/compare/v1.10.3...runcom:docker-1.10.3-stable) for docker v1.10 identified by RedHat

#### Rkt runtime Known Issues

- A detailed list of known issues can be found [here](https://github.com/kubernetes/kubernetes.github.io/blob/release-1.3/docs/getting-started-guides/rkt/notes.md)

*More Instructions coming soon*

## Provider-specific Notes

* AWS
  * Support for ap-northeast-2 region (Seoul)
  * Allow cross-region image pulling with ECR
  * More reliable kube-up/kube-down
  * Enable ICMP Type 3 Code 4 for ELBs
  * ARP caching fix
  * Use /dev/xvdXX names
  * ELB:
    * ELB proxy protocol support
    * mixed plaintext/encrypted ports support in ELBs
    * SSL support for ELB listeners
  * Allow VPC CIDR to be specified (experimental)
  * Fix problems with >2 security groups
* GCP:
  * Enable using gcr.io as a Docker registry mirror.
  * Make bigger master root disks in GCE for large clusters.
  * Change default clusterCIDRs from /16 to /14 allowing 1000 Node clusters by default.
  * Allow Debian Jessie on GCE.
  * Node problem detector addon pod detects and reports kernel deadlocks.
* OpenStack
  * Provider added.
* VSphere:
  * Provider updated.

## Previous Releases Included in v1.3.0

- [v1.3.0-beta.3](CHANGELOG.md#v130-beta3)
- [v1.3.0-beta.2](CHANGELOG.md#v130-beta2)
- [v1.3.0-beta.1](CHANGELOG.md#v130-beta1)
- [v1.3.0-alpha.5](CHANGELOG.md#v130-alpha5)
- [v1.3.0-alpha.4](CHANGELOG.md#v130-alpha4)
- [v1.3.0-alpha.3](CHANGELOG.md#v130-alpha3)
- [v1.3.0-alpha.2](CHANGELOG.md#v130-alpha2)
- [v1.3.0-alpha.1](CHANGELOG.md#v130-alpha1)



# v1.3.0-beta.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-beta.3/kubernetes.tar.gz) | `9d18964a294f356bfdc841957dcad8ff35ed909c` | `ee5fcdf86645135ed132663967876dd6`

## Changelog since v1.3.0-beta.2

### Action Required

* [kubelet] Allow opting out of automatic cloud provider detection in kubelet. By default kubelet will auto-detect cloud providers ([#28258](https://github.com/kubernetes/kubernetes/pull/28258), [@vishh](https://github.com/vishh))
* If you use one of the kube-dns replication controller manifest in `cluster/saltbase/salt/kube-dns`, i.e. `cluster/saltbase/salt/kube-dns/{skydns-rc.yaml.base,skydns-rc.yaml.in}`, either substitute one of `__PILLAR__FEDERATIONS__DOMAIN__MAP__` or `{{ pillar['federations_domain_map'] }}` with the corresponding federation name to domain name value or remove them if you do not support cluster federation at this time. If you plan to substitute the parameter with its value, here is an example for `{{ pillar['federations_domain_map'] }` ([#28132](https://github.com/kubernetes/kubernetes/pull/28132), [@madhusudancs](https://github.com/madhusudancs))
    * pillar['federations_domain_map'] = "- --federations=myfederation=federation.test"
    * where `myfederation` is the name of the federation and `federation.test` is the domain name registered for the federation.
* federation: Upgrading the groupversion to v1beta1 ([#28186](https://github.com/kubernetes/kubernetes/pull/28186), [@nikhiljindal](https://github.com/nikhiljindal))
* Set Dashboard UI version to v1.1.0 ([#27869](https://github.com/kubernetes/kubernetes/pull/27869), [@bryk](https://github.com/bryk))

### Other notable changes

* Build: Add KUBE_GCS_RELEASE_BUCKET_MIRROR option to push-ci-build.sh ([#28172](https://github.com/kubernetes/kubernetes/pull/28172), [@zmerlynn](https://github.com/zmerlynn))
* Image GC logic should compensate for reserved blocks ([#27996](https://github.com/kubernetes/kubernetes/pull/27996), [@ronnielai](https://github.com/ronnielai))
* Bump minimum API version for docker to 1.21 ([#27208](https://github.com/kubernetes/kubernetes/pull/27208), [@yujuhong](https://github.com/yujuhong))
* Adding lock files for kubeconfig updating ([#28034](https://github.com/kubernetes/kubernetes/pull/28034), [@krousey](https://github.com/krousey))
* federation service controller: fixing the logic to update DNS records ([#27999](https://github.com/kubernetes/kubernetes/pull/27999), [@quinton-hoole](https://github.com/quinton-hoole))
* federation: Updating KubeDNS to try finding a local service first for federation query ([#27708](https://github.com/kubernetes/kubernetes/pull/27708), [@nikhiljindal](https://github.com/nikhiljindal))
* Support journal logs in fluentd-gcp on GCI ([#27981](https://github.com/kubernetes/kubernetes/pull/27981), [@a-robinson](https://github.com/a-robinson))
* Copy and display source location prominently on Kubernetes instances ([#27985](https://github.com/kubernetes/kubernetes/pull/27985), [@maisem](https://github.com/maisem))
* Federation e2e support for AWS ([#27791](https://github.com/kubernetes/kubernetes/pull/27791), [@colhom](https://github.com/colhom))
* Copy and display source location prominently on Kubernetes instances ([#27840](https://github.com/kubernetes/kubernetes/pull/27840), [@zmerlynn](https://github.com/zmerlynn))
* AWS/GCE: Spread PetSet volume creation across zones, create GCE volumes in non-master zones ([#27553](https://github.com/kubernetes/kubernetes/pull/27553), [@justinsb](https://github.com/justinsb))
* GCE provider: Create TargetPool with 200 instances, then update with rest ([#27829](https://github.com/kubernetes/kubernetes/pull/27829), [@zmerlynn](https://github.com/zmerlynn))
* Add sources to server tarballs. ([#27830](https://github.com/kubernetes/kubernetes/pull/27830), [@david-mcmahon](https://github.com/david-mcmahon))
* Retry Pod/RC updates in kubectl rolling-update ([#27509](https://github.com/kubernetes/kubernetes/pull/27509), [@janetkuo](https://github.com/janetkuo))
* AWS kube-up: Authorize route53 in the IAM policy ([#27794](https://github.com/kubernetes/kubernetes/pull/27794), [@justinsb](https://github.com/justinsb))
* Allow conformance tests to run on non-GCE providers ([#26932](https://github.com/kubernetes/kubernetes/pull/26932), [@aaronlevy](https://github.com/aaronlevy))
* AWS kube-up: move to Docker 1.11.2 ([#27676](https://github.com/kubernetes/kubernetes/pull/27676), [@justinsb](https://github.com/justinsb))
* Fixed an issue that Deployment may be scaled down further than allowed by maxUnavailable when minReadySeconds is set. ([#27728](https://github.com/kubernetes/kubernetes/pull/27728), [@janetkuo](https://github.com/janetkuo))



# v1.2.5

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.5/kubernetes.tar.gz) | `ddf12d7f37dfef25308798d71ad547761d0785ac` | `69d770df8fa4eceb57167e34df3962ca`

## Changes since v1.2.4

### Other notable changes

* Retry Pod/RC updates in kubectl rolling-update ([#27509](https://github.com/kubernetes/kubernetes/pull/27509), [@janetkuo](https://github.com/janetkuo))
* GCE provider: Create TargetPool with 200 instances, then update with rest ([#27865](https://github.com/kubernetes/kubernetes/pull/27865), [@zmerlynn](https://github.com/zmerlynn))
* GCE provider: Limit Filter calls to regexps rather than large blobs ([#27741](https://github.com/kubernetes/kubernetes/pull/27741), [@zmerlynn](https://github.com/zmerlynn))
* Fix strategic merge diff list diff bug ([#26418](https://github.com/kubernetes/kubernetes/pull/26418), [@AdoHe](https://github.com/AdoHe))
* AWS: Fix long-standing bug in stringSetToPointers ([#26331](https://github.com/kubernetes/kubernetes/pull/26331), [@therc](https://github.com/therc))
* AWS kube-up: Increase timeout waiting for docker start ([#25405](https://github.com/kubernetes/kubernetes/pull/25405), [@justinsb](https://github.com/justinsb))
* Fix hyperkube flag parsing ([#25512](https://github.com/kubernetes/kubernetes/pull/25512), [@colhom](https://github.com/colhom))
* kubectl rolling-update support for same image ([#24645](https://github.com/kubernetes/kubernetes/pull/24645), [@jlowdermilk](https://github.com/jlowdermilk))
* Return "410 Gone" errors via watch stream when using watch cache ([#25369](https://github.com/kubernetes/kubernetes/pull/25369), [@liggitt](https://github.com/liggitt))



# v1.3.0-beta.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-beta.2/kubernetes.tar.gz) | `9c95762970b943d6c6547f0841c1e5471148b0e3` | `dc9e8560f24459b2313317b15910bee7`

## Changes since v1.3.0-beta.1

### Experimental Features

* Init containers enable pod authors to perform tasks before their normal containers start. Each init container is started in order, and failing containers will prevent the application from starting. ([#23666](https://github.com/kubernetes/kubernetes/pull/23666), [@smarterclayton](https://github.com/smarterclayton))

### Other notable changes

* GCE provider: Limit Filter calls to regexps rather than large blobs ([#27741](https://github.com/kubernetes/kubernetes/pull/27741), [@zmerlynn](https://github.com/zmerlynn))
* Show LASTSEEN, the sorting key, as the first column in `kubectl get event` output ([#27549](https://github.com/kubernetes/kubernetes/pull/27549), [@therc](https://github.com/therc))
* GCI: fix kubectl permission issue [#27643](https://github.com/kubernetes/kubernetes/pull/27643) ([#27740](https://github.com/kubernetes/kubernetes/pull/27740), [@andyzheng0831](https://github.com/andyzheng0831))
* Add federation api and cm servers to hyperkube ([#27586](https://github.com/kubernetes/kubernetes/pull/27586), [@colhom](https://github.com/colhom))
* federation: Creating kubeconfig files to be used for creating secrets for clusters on aws and gke ([#27332](https://github.com/kubernetes/kubernetes/pull/27332), [@nikhiljindal](https://github.com/nikhiljindal))
* AWS: Enable ICMP Type 3 Code 4 for ELBs ([#27677](https://github.com/kubernetes/kubernetes/pull/27677), [@justinsb](https://github.com/justinsb))
* Bumped Heapster to v1.1.0. ([#27542](https://github.com/kubernetes/kubernetes/pull/27542), [@piosz](https://github.com/piosz))
    * More details about the release https://github.com/kubernetes/heapster/releases/tag/v1.1.0
* Deleting federation-push.sh ([#27400](https://github.com/kubernetes/kubernetes/pull/27400), [@nikhiljindal](https://github.com/nikhiljindal))
* Validate-cluster finishes shortly after at most ALLOWED_NOTREADY_NODE… ([#26778](https://github.com/kubernetes/kubernetes/pull/26778), [@gmarek](https://github.com/gmarek))
* AWS kube-down: Issue warning if VPC not found ([#27518](https://github.com/kubernetes/kubernetes/pull/27518), [@justinsb](https://github.com/justinsb))
* gce/kube-down: Parallelize IGM deletion, batch more ([#27302](https://github.com/kubernetes/kubernetes/pull/27302), [@zmerlynn](https://github.com/zmerlynn))
* Enable dynamic allocation of heapster/eventer cpu request/limit ([#27185](https://github.com/kubernetes/kubernetes/pull/27185), [@gmarek](https://github.com/gmarek))
* 'kubectl describe pv' now shows events ([#27431](https://github.com/kubernetes/kubernetes/pull/27431), [@jsafrane](https://github.com/jsafrane))
* AWS kube-up: set net.ipv4.neigh.default.gc_thresh1=0 to avoid ARP over-caching ([#27682](https://github.com/kubernetes/kubernetes/pull/27682), [@justinsb](https://github.com/justinsb))
* AWS volumes: Use /dev/xvdXX names with EC2 ([#27628](https://github.com/kubernetes/kubernetes/pull/27628), [@justinsb](https://github.com/justinsb))
* Add a test config variable to specify desired Docker version to run on GCI. ([#26813](https://github.com/kubernetes/kubernetes/pull/26813), [@wonderfly](https://github.com/wonderfly))
* Check for thin_is binary in path for devicemapper when using ThinPoolWatcher and fix uint64 overflow issue for CPU stats ([#27591](https://github.com/kubernetes/kubernetes/pull/27591), [@dchen1107](https://github.com/dchen1107))
* Change default value of deleting-pods-burst to 1 ([#27606](https://github.com/kubernetes/kubernetes/pull/27606), [@gmarek](https://github.com/gmarek))
* MESOS: fix race condition in contrib/mesos/pkg/queue/delay ([#24916](https://github.com/kubernetes/kubernetes/pull/24916), [@jdef](https://github.com/jdef))
* including federation binaries in the list of images we push during release ([#27396](https://github.com/kubernetes/kubernetes/pull/27396), [@nikhiljindal](https://github.com/nikhiljindal))
* fix updatePod() of RS and RC controllers ([#27415](https://github.com/kubernetes/kubernetes/pull/27415), [@caesarxuchao](https://github.com/caesarxuchao))
* Change default value of deleting-pods-burst to 1 ([#27422](https://github.com/kubernetes/kubernetes/pull/27422), [@gmarek](https://github.com/gmarek))
* A new volume manager was introduced in kubelet that synchronizes volume mount/unmount (and attach/detach, if attach/detach controller is not enabled). ([#26801](https://github.com/kubernetes/kubernetes/pull/26801), [@saad-ali](https://github.com/saad-ali))
    * This eliminates the race conditions between the pod creation loop and the orphaned volumes loops. It also removes the unmount/detach from the `syncPod()` path so volume clean up never blocks the `syncPod` loop.



# v1.3.0-beta.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-beta.1/kubernetes.tar.gz) | `2b54995ee8f52d78dc31c3d7291e8dfa5c809fe7` | `f1022a84c3441cae4ebe1d295470be8f`

## Changes since v1.3.0-alpha.5

### Action Required

* Fixing logic to generate ExternalHost in genericapiserver ([#26796](https://github.com/kubernetes/kubernetes/pull/26796), [@nikhiljindal](https://github.com/nikhiljindal))
* federation: Updating federation-controller-manager to use secret to get federation-apiserver's kubeconfig ([#26819](https://github.com/kubernetes/kubernetes/pull/26819), [@nikhiljindal](https://github.com/nikhiljindal))

### Other notable changes

* federation: fix dns provider initialization issues ([#27252](https://github.com/kubernetes/kubernetes/pull/27252), [@mfanjie](https://github.com/mfanjie))
* Updating federation up scripts to work in non e2e setup ([#27260](https://github.com/kubernetes/kubernetes/pull/27260), [@nikhiljindal](https://github.com/nikhiljindal))
* version bump for gci to milestone 53 ([#27210](https://github.com/kubernetes/kubernetes/pull/27210), [@adityakali](https://github.com/adityakali))
* kubectl apply: retry applying a patch if a version conflict error is encountered ([#26557](https://github.com/kubernetes/kubernetes/pull/26557), [@AdoHe](https://github.com/AdoHe))
* Revert "Wait for arc.getArchive() to complete before running tests" ([#27130](https://github.com/kubernetes/kubernetes/pull/27130), [@pwittrock](https://github.com/pwittrock))
* ResourceQuota BestEffort scope aligned with Pod level QoS ([#26969](https://github.com/kubernetes/kubernetes/pull/26969), [@derekwaynecarr](https://github.com/derekwaynecarr))
* The AWS cloudprovider will cache results from DescribeInstances() if the set of nodes hasn't changed ([#26900](https://github.com/kubernetes/kubernetes/pull/26900), [@therc](https://github.com/therc))
* GCE provider: Log full contents of long operations ([#26962](https://github.com/kubernetes/kubernetes/pull/26962), [@zmerlynn](https://github.com/zmerlynn))
* Fix system container detection in kubelet on systemd. ([#26586](https://github.com/kubernetes/kubernetes/pull/26586), [@derekwaynecarr](https://github.com/derekwaynecarr))
    * This fixed environments where CPU and Memory Accounting were not enabled on the unit that launched the kubelet or docker from reporting the root cgroup when monitoring usage stats for those components.
* New default horizontalpodautoscaler/v1 generator for kubectl autoscale. ([#26775](https://github.com/kubernetes/kubernetes/pull/26775), [@piosz](https://github.com/piosz))
    * Use autoscaling/v1 in kubectl by default.
* federation: Adding dnsprovider flags to federation-controller-manager ([#27158](https://github.com/kubernetes/kubernetes/pull/27158), [@nikhiljindal](https://github.com/nikhiljindal))
* federation service controller: fixing a bug so that existing services are created in newly registered clusters ([#27028](https://github.com/kubernetes/kubernetes/pull/27028), [@mfanjie](https://github.com/mfanjie))
* Rename environment variables (KUBE_)ENABLE_NODE_AUTOSCALER to (KUBE_)ENABLE_CLUSTER_AUTOSCALER.  ([#27117](https://github.com/kubernetes/kubernetes/pull/27117), [@mwielgus](https://github.com/mwielgus))
* support for mounting local-ssds on GCI ([#27143](https://github.com/kubernetes/kubernetes/pull/27143), [@adityakali](https://github.com/adityakali))
* AWS: support mixed plaintext/encrypted ports in ELBs via service.beta.kubernetes.io/aws-load-balancer-ssl-ports annotation ([#26976](https://github.com/kubernetes/kubernetes/pull/26976), [@therc](https://github.com/therc))
* Updating e2e docs with instructions on running federation tests ([#27072](https://github.com/kubernetes/kubernetes/pull/27072), [@colhom](https://github.com/colhom))
* LBaaS v2 Support for Openstack Cloud Provider Plugin ([#25987](https://github.com/kubernetes/kubernetes/pull/25987), [@dagnello](https://github.com/dagnello))
* GCI: add support for network plugin ([#27027](https://github.com/kubernetes/kubernetes/pull/27027), [@andyzheng0831](https://github.com/andyzheng0831))
* Bump cAdvisor to v0.23.3 ([#27065](https://github.com/kubernetes/kubernetes/pull/27065), [@timstclair](https://github.com/timstclair))
* Stop 'kubectl drain' deleting pods with local storage. ([#26667](https://github.com/kubernetes/kubernetes/pull/26667), [@mml](https://github.com/mml))
* Networking e2es: Wait for all nodes to be schedulable in kubeproxy and networking tests ([#27008](https://github.com/kubernetes/kubernetes/pull/27008), [@zmerlynn](https://github.com/zmerlynn))
* change clientset of service controller to versioned ([#26694](https://github.com/kubernetes/kubernetes/pull/26694), [@mfanjie](https://github.com/mfanjie))
* Use gcr.io as a Docker registry mirror when setting up a cluster in GCE. ([#25841](https://github.com/kubernetes/kubernetes/pull/25841), [@ojarjur](https://github.com/ojarjur))
* correction on rbd volume object and defaults ([#25490](https://github.com/kubernetes/kubernetes/pull/25490), [@rootfs](https://github.com/rootfs))
* Bump GCE debian image to container-v1-3-v20160604 ([#26851](https://github.com/kubernetes/kubernetes/pull/26851), [@zmerlynn](https://github.com/zmerlynn))
* Option to enable http2 on client connections. ([#25280](https://github.com/kubernetes/kubernetes/pull/25280), [@timothysc](https://github.com/timothysc))
* kubectl get ingress output remove rules ([#26684](https://github.com/kubernetes/kubernetes/pull/26684), [@AdoHe](https://github.com/AdoHe))
* AWS kube-up: Remove SecurityContextDeny admission controller (to mirror GCE) ([#25381](https://github.com/kubernetes/kubernetes/pull/25381), [@zquestz](https://github.com/zquestz))
* Fix third party ([#25894](https://github.com/kubernetes/kubernetes/pull/25894), [@brendandburns](https://github.com/brendandburns))
* AWS Route53 dnsprovider ([#26049](https://github.com/kubernetes/kubernetes/pull/26049), [@quinton-hoole](https://github.com/quinton-hoole))
* GCI/Trusty: support the Docker registry mirror ([#26745](https://github.com/kubernetes/kubernetes/pull/26745), [@andyzheng0831](https://github.com/andyzheng0831))
* Kubernetes v1.3 introduces a new Attach/Detach Controller. This controller manages attaching and detaching of volumes on-behalf of nodes.  ([#26351](https://github.com/kubernetes/kubernetes/pull/26351), [@saad-ali](https://github.com/saad-ali))
    * This ensures that attachment and detachment of volumes is independent of any single nodes’ availability. Meaning, if a node or kubelet becomes unavailable for any reason, the volumes attached to that node will be detached so they are free to be attached to other nodes.
    * Specifically the new controller watches the API server for scheduled pods. It processes each pod and ensures that any volumes that implement the volume Attacher interface are attached to the node their pod is scheduled to.
    * When a pod is deleted, the controller waits for the volume to be safely unmounted by kubelet. It does this by waiting for the volume to no longer be present in the nodes Node.Status.VolumesInUse list. If the volume is not safely unmounted by kubelet within a pre-configured duration (3 minutes in Kubernetes v1.3), the controller unilaterally detaches the volume (this prevents volumes from getting stranded on nodes that become unavailable).
    * In order to remain backwards compatible, the new controller only manages attach/detach of volumes that are scheduled to nodes that opt-in to controller management. Nodes running v1.3 or higher of Kubernetes opt-in to controller management by default by setting the "volumes.kubernetes.io/controller-managed-attach-detach" annotation on the Node object on startup. This behavior is gated by a new kubelet flag, "enable-controller-attach-detach,” (default true).
    * In order to safely upgrade an existing Kubernetes cluster without interruption of volume attach/detach logic:
        * First upgrade the master to Kubernetes v1.3.
            * This will start the new attach/detach controller.
            * The new controller will initially ignore volumes for all nodes since they lack the "volumes.kubernetes.io/controller-managed-attach-detach" annotation.
        * Then upgrade nodes to Kubernetes v1.3.
            * As nodes are upgraded, they will automatically, by default, opt-in to attach/detach controller management, which will cause the controller to start managing attaches/detaches for volumes that get scheduled to those nodes.
* Added DNS Reverse Record logic for service IPs ([#26226](https://github.com/kubernetes/kubernetes/pull/26226), [@ArtfulCoder](https://github.com/ArtfulCoder))
* read gluster log to surface glusterfs plugin errors properly in describe events ([#24808](https://github.com/kubernetes/kubernetes/pull/24808), [@screeley44](https://github.com/screeley44))



# v1.3.0-alpha.5

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-alpha.5/kubernetes.tar.gz) | `724bf5a4437ca9dc75d9297382f47a179e8dc5a6` | `2a8b4a5297df3007fce69f1e344fd87e`

## Changes since v1.3.0-alpha.4

### Action Required

* Add direct serializer ([#26251](https://github.com/kubernetes/kubernetes/pull/26251), [@caesarxuchao](https://github.com/caesarxuchao))
* Add a NodeCondition "NetworkUnavailable" to prevent scheduling onto a node until the routes have been created  ([#26415](https://github.com/kubernetes/kubernetes/pull/26415), [@wojtek-t](https://github.com/wojtek-t))
* Add garbage collector into kube-controller-manager ([#26341](https://github.com/kubernetes/kubernetes/pull/26341), [@caesarxuchao](https://github.com/caesarxuchao))
* Add orphaning finalizer logic to GC ([#25599](https://github.com/kubernetes/kubernetes/pull/25599), [@caesarxuchao](https://github.com/caesarxuchao))
* GCI-backed masters mount srv/kubernetes and srv/sshproxy in the right place ([#26238](https://github.com/kubernetes/kubernetes/pull/26238), [@ihmccreery](https://github.com/ihmccreery))
* Updaing QoS policy to be at the pod level ([#14943](https://github.com/kubernetes/kubernetes/pull/14943), [@vishh](https://github.com/vishh))
* add CIDR allocator for NodeController ([#19242](https://github.com/kubernetes/kubernetes/pull/19242), [@mqliang](https://github.com/mqliang))
* Adding garbage collector controller ([#24509](https://github.com/kubernetes/kubernetes/pull/24509), [@caesarxuchao](https://github.com/caesarxuchao))

### Other notable changes

* Fix a bug with pluralization of third party resources ([#25374](https://github.com/kubernetes/kubernetes/pull/25374), [@brendandburns](https://github.com/brendandburns))
* Run l7 controller on master  ([#26048](https://github.com/kubernetes/kubernetes/pull/26048), [@bprashanth](https://github.com/bprashanth))
* AWS: ELB proxy protocol support via annotation service.beta.kubernetes.io/aws-load-balancer-proxy-protocol ([#24569](https://github.com/kubernetes/kubernetes/pull/24569), [@williamsandrew](https://github.com/williamsandrew))
* kubectl run --restart=Never creates pods ([#25253](https://github.com/kubernetes/kubernetes/pull/25253), [@soltysh](https://github.com/soltysh))
* Add LabelSelector to PersistentVolumeClaimSpec ([#25917](https://github.com/kubernetes/kubernetes/pull/25917), [@pmorie](https://github.com/pmorie))
* Removed metrics api group ([#26073](https://github.com/kubernetes/kubernetes/pull/26073), [@piosz](https://github.com/piosz))
* Fixed check in kubectl autoscale: cpu consumption can be higher than 100%. ([#26162](https://github.com/kubernetes/kubernetes/pull/26162), [@jszczepkowski](https://github.com/jszczepkowski))
* Add support for 3rd party objects to kubectl label ([#24882](https://github.com/kubernetes/kubernetes/pull/24882), [@brendandburns](https://github.com/brendandburns))
* Move shell completion generation into 'kubectl completion' command ([#23801](https://github.com/kubernetes/kubernetes/pull/23801), [@sttts](https://github.com/sttts))
* Fix strategic merge diff list diff bug ([#26418](https://github.com/kubernetes/kubernetes/pull/26418), [@AdoHe](https://github.com/AdoHe))
* Setting TLS1.2 minimum because TLS1.0 and TLS1.1 are vulnerable ([#26169](https://github.com/kubernetes/kubernetes/pull/26169), [@victorgp](https://github.com/victorgp))
* Kubelet: Periodically reporting image pulling progress in log ([#26145](https://github.com/kubernetes/kubernetes/pull/26145), [@Random-Liu](https://github.com/Random-Liu))
* Federation service controller is one key component of federation controller manager, it watches federation service, creates/updates services to all registered clusters, and update DNS records to global DNS server. ([#26034](https://github.com/kubernetes/kubernetes/pull/26034), [@mfanjie](https://github.com/mfanjie))
* Stabilize map order in kubectl describe ([#26046](https://github.com/kubernetes/kubernetes/pull/26046), [@timoreimann](https://github.com/timoreimann))
* Google Cloud DNS dnsprovider - replacement for [#25389](https://github.com/kubernetes/kubernetes/pull/25389) ([#26020](https://github.com/kubernetes/kubernetes/pull/26020), [@quinton-hoole](https://github.com/quinton-hoole))
* Fix system container detection in kubelet on systemd. ([#25982](https://github.com/kubernetes/kubernetes/pull/25982), [@derekwaynecarr](https://github.com/derekwaynecarr))
    * This fixed environments where CPU and Memory Accounting were not enabled on the unit
    * that launched the kubelet or docker from reporting the root cgroup when
    * monitoring usage stats for those components.
* Added pods-per-core to kubelet. [#25762](https://github.com/kubernetes/kubernetes/pull/25762) ([#25813](https://github.com/kubernetes/kubernetes/pull/25813), [@rrati](https://github.com/rrati))
* promote sourceRange into service spec ([#25826](https://github.com/kubernetes/kubernetes/pull/25826), [@freehan](https://github.com/freehan))
* kube-controller-manager: Add configure-cloud-routes option ([#25614](https://github.com/kubernetes/kubernetes/pull/25614), [@justinsb](https://github.com/justinsb))
* kubelet: reading cloudinfo from cadvisor ([#21373](https://github.com/kubernetes/kubernetes/pull/21373), [@enoodle](https://github.com/enoodle))
* Disable cAdvisor event storage by default ([#24771](https://github.com/kubernetes/kubernetes/pull/24771), [@timstclair](https://github.com/timstclair))
* Remove docker-multinode ([#26031](https://github.com/kubernetes/kubernetes/pull/26031), [@luxas](https://github.com/luxas))
* nodecontroller: Fix log message on successful update ([#26207](https://github.com/kubernetes/kubernetes/pull/26207), [@zmerlynn](https://github.com/zmerlynn))
* remove deprecated generated typed clients ([#26336](https://github.com/kubernetes/kubernetes/pull/26336), [@caesarxuchao](https://github.com/caesarxuchao))
* Kubenet host-port support through iptables ([#25604](https://github.com/kubernetes/kubernetes/pull/25604), [@freehan](https://github.com/freehan))
* Add metrics support for a GCE PD, EC2 EBS & Azure File volumes ([#25852](https://github.com/kubernetes/kubernetes/pull/25852), [@vishh](https://github.com/vishh))
* Bump cAdvisor to v0.23.2 - See [changelog](https://github.com/google/cadvisor/blob/master/CHANGELOG.md) for details ([#25914](https://github.com/kubernetes/kubernetes/pull/25914), [@timstclair](https://github.com/timstclair))
* Alpha version of "Role Based Access Control" API. ([#25634](https://github.com/kubernetes/kubernetes/pull/25634), [@ericchiang](https://github.com/ericchiang))
* Add Seccomp API ([#25324](https://github.com/kubernetes/kubernetes/pull/25324), [@jfrazelle](https://github.com/jfrazelle))
* AWS: Fix long-standing bug in stringSetToPointers ([#26331](https://github.com/kubernetes/kubernetes/pull/26331), [@therc](https://github.com/therc))
* Add dnsmasq as a DNS cache in kube-dns pod ([#26114](https://github.com/kubernetes/kubernetes/pull/26114), [@ArtfulCoder](https://github.com/ArtfulCoder))
* routecontroller: Add wait.NonSlidingUntil, use it ([#26301](https://github.com/kubernetes/kubernetes/pull/26301), [@zmerlynn](https://github.com/zmerlynn))
* Attempt 2: Bump GCE containerVM to container-v1-3-v20160517 (Docker 1.11.1) again. ([#26001](https://github.com/kubernetes/kubernetes/pull/26001), [@dchen1107](https://github.com/dchen1107))
* Downward API implementation for resources limits and requests ([#24179](https://github.com/kubernetes/kubernetes/pull/24179), [@aveshagarwal](https://github.com/aveshagarwal))
* GCE clusters start using GCI as the default OS image for masters ([#26197](https://github.com/kubernetes/kubernetes/pull/26197), [@wonderfly](https://github.com/wonderfly))
* Add a 'kubectl clusterinfo dump' option ([#20672](https://github.com/kubernetes/kubernetes/pull/20672), [@brendandburns](https://github.com/brendandburns))
* Fixing heapster memory requirements. ([#26109](https://github.com/kubernetes/kubernetes/pull/26109), [@Q-Lee](https://github.com/Q-Lee))
* Handle federated service name lookups in kube-dns. ([#25727](https://github.com/kubernetes/kubernetes/pull/25727), [@madhusudancs](https://github.com/madhusudancs))
* Support sort-by timestamp in kubectl get ([#25600](https://github.com/kubernetes/kubernetes/pull/25600), [@janetkuo](https://github.com/janetkuo))
* vSphere Volume Plugin Implementation ([#24947](https://github.com/kubernetes/kubernetes/pull/24947), [@abithap](https://github.com/abithap))
* ResourceQuota controller uses rate limiter to prevent hot-loops in error situations ([#25748](https://github.com/kubernetes/kubernetes/pull/25748), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Fix hyperkube flag parsing ([#25512](https://github.com/kubernetes/kubernetes/pull/25512), [@colhom](https://github.com/colhom))
* Add a kubectl create secret tls command ([#24719](https://github.com/kubernetes/kubernetes/pull/24719), [@bprashanth](https://github.com/bprashanth))
* Introduce a new add-on pod NodeProblemDetector. ([#25986](https://github.com/kubernetes/kubernetes/pull/25986), [@Random-Liu](https://github.com/Random-Liu))
    * NodeProblemDetector is a DaemonSet running on each node, monitoring node health and reporting
    * node problems as NodeCondition and Event. Currently it already supports kernel log monitoring, and
    * will support more problem detection in the future. It is enabled by default on gce now.
* Handle cAdvisor partial failures ([#25933](https://github.com/kubernetes/kubernetes/pull/25933), [@timstclair](https://github.com/timstclair))
* Use SkyDNS as a library for a more integrated kube DNS ([#23930](https://github.com/kubernetes/kubernetes/pull/23930), [@ArtfulCoder](https://github.com/ArtfulCoder))
* Introduce node memory pressure condition to scheduler ([#25531](https://github.com/kubernetes/kubernetes/pull/25531), [@ingvagabund](https://github.com/ingvagabund))
* Fix detection of docker cgroup on RHEL ([#25907](https://github.com/kubernetes/kubernetes/pull/25907), [@ncdc](https://github.com/ncdc))
* Kubelet evicts pods when available memory falls below configured eviction thresholds ([#25772](https://github.com/kubernetes/kubernetes/pull/25772), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Use protobufs by default to communicate with apiserver (still store JSONs in etcd) ([#25738](https://github.com/kubernetes/kubernetes/pull/25738), [@wojtek-t](https://github.com/wojtek-t))
* Implement NetworkPolicy v1beta1 API object / client support. ([#25638](https://github.com/kubernetes/kubernetes/pull/25638), [@caseydavenport](https://github.com/caseydavenport))
* Only expose top N images in `NodeStatus` ([#25328](https://github.com/kubernetes/kubernetes/pull/25328), [@resouer](https://github.com/resouer))
* Extend secrets volumes with path control ([#25285](https://github.com/kubernetes/kubernetes/pull/25285), [@ingvagabund](https://github.com/ingvagabund))
* With this PR, kubectl and other RestClient's using the AuthProvider framework can make OIDC authenticated requests, and, if there is a refresh token present, the tokens will be refreshed as needed. ([#25270](https://github.com/kubernetes/kubernetes/pull/25270), [@bobbyrullo](https://github.com/bobbyrullo))
* Make addon-manager cross-platform and use it with hyperkube ([#25631](https://github.com/kubernetes/kubernetes/pull/25631), [@luxas](https://github.com/luxas))
* kubelet: Optionally, have kubelet exit if lock file contention is observed, using --exit-on-lock-contention flag ([#25596](https://github.com/kubernetes/kubernetes/pull/25596), [@derekparker](https://github.com/derekparker))
* Bump up glbc version to 0.6.2 ([#25446](https://github.com/kubernetes/kubernetes/pull/25446), [@bprashanth](https://github.com/bprashanth))
* Add "kubectl set image" for easier updating container images (for pods or resources with pod templates).  ([#25509](https://github.com/kubernetes/kubernetes/pull/25509), [@janetkuo](https://github.com/janetkuo))
* NodeController doesn't evict Pods if no Nodes are Ready ([#25571](https://github.com/kubernetes/kubernetes/pull/25571), [@gmarek](https://github.com/gmarek))
* Incompatible change of kube-up.sh:  ([#25734](https://github.com/kubernetes/kubernetes/pull/25734), [@jszczepkowski](https://github.com/jszczepkowski))
    * when turning on cluster autoscaler by setting KUBE_ENABLE_NODE_AUTOSCALER=true,
    * KUBE_AUTOSCALER_MIN_NODES and KUBE_AUTOSCALER_MAX_NODES need to be set.
* systemd node spec proposal ([#17688](https://github.com/kubernetes/kubernetes/pull/17688), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Bump GCE ContainerVM to container-v1-3-v20160517 (Docker 1.11.1) ([#25843](https://github.com/kubernetes/kubernetes/pull/25843), [@zmerlynn](https://github.com/zmerlynn))
* AWS: Move enforcement of attached AWS device limit from kubelet to scheduler ([#23254](https://github.com/kubernetes/kubernetes/pull/23254), [@jsafrane](https://github.com/jsafrane))
* Refactor persistent volume controller ([#24331](https://github.com/kubernetes/kubernetes/pull/24331), [@jsafrane](https://github.com/jsafrane))
* Add support for running GCI on the GCE cloud provider ([#25425](https://github.com/kubernetes/kubernetes/pull/25425), [@andyzheng0831](https://github.com/andyzheng0831))
* Implement taints and tolerations ([#24134](https://github.com/kubernetes/kubernetes/pull/24134), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* Add init containers to pods ([#23567](https://github.com/kubernetes/kubernetes/pull/23567), [@smarterclayton](https://github.com/smarterclayton))



# v1.3.0-alpha.4

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-alpha.4/kubernetes.tar.gz) | `758e97e7e50153840379ecd9f8fda1869543539f` | `4e18ae6a428c99fcc30e2137d7c41854`

## Changes since v1.3.0-alpha.3

### Action Required

* validate third party resources ([#25007](https://github.com/kubernetes/kubernetes/pull/25007), [@liggitt](https://github.com/liggitt))
* Automatically create the kube-system namespace ([#25196](https://github.com/kubernetes/kubernetes/pull/25196), [@luxas](https://github.com/luxas))
* Make ThirdPartyResource a root scoped object ([#25006](https://github.com/kubernetes/kubernetes/pull/25006), [@liggitt](https://github.com/liggitt))
* mark container-port flag as deprecated ([#25072](https://github.com/kubernetes/kubernetes/pull/25072), [@AdoHe](https://github.com/AdoHe))
* Provide flags to use etcd3 backed storage ([#24455](https://github.com/kubernetes/kubernetes/pull/24455), [@hongchaodeng](https://github.com/hongchaodeng))

### Other notable changes

* Fix hyperkube's layer caching, and remove --make-symlinks at build time ([#25693](https://github.com/kubernetes/kubernetes/pull/25693), [@luxas](https://github.com/luxas))
* AWS: More support for ap-northeast-2 region ([#24464](https://github.com/kubernetes/kubernetes/pull/24464), [@matthewrudy](https://github.com/matthewrudy))
* Make bigger master root disks in GCE for large clusters ([#25670](https://github.com/kubernetes/kubernetes/pull/25670), [@gmarek](https://github.com/gmarek))
* AWS kube-down: don't fail if ELB not in VPC - [#23784](https://github.com/kubernetes/kubernetes/pull/23784) ([#23785](https://github.com/kubernetes/kubernetes/pull/23785), [@ajohnstone](https://github.com/ajohnstone))
* Build hyperkube in hack/local-up-cluster instead of separate binaries ([#25627](https://github.com/kubernetes/kubernetes/pull/25627), [@luxas](https://github.com/luxas))
* enable recursive processing in kubectl rollout ([#25110](https://github.com/kubernetes/kubernetes/pull/25110), [@metral](https://github.com/metral))
* Support struct,array,slice types when sorting kubectl output ([#25022](https://github.com/kubernetes/kubernetes/pull/25022), [@zhouhaibing089](https://github.com/zhouhaibing089))
* federated api servers: Adding a discovery summarizer server ([#20358](https://github.com/kubernetes/kubernetes/pull/20358), [@nikhiljindal](https://github.com/nikhiljindal))
* AWS: Allow cross-region image pulling with ECR ([#24369](https://github.com/kubernetes/kubernetes/pull/24369), [@therc](https://github.com/therc))
* Automatically add node labels beta.kubernetes.io/{os,arch} ([#23684](https://github.com/kubernetes/kubernetes/pull/23684), [@luxas](https://github.com/luxas))
* kubectl "rm" will suggest using "delete"; "ps" and "list" will suggest "get". ([#25181](https://github.com/kubernetes/kubernetes/pull/25181), [@janetkuo](https://github.com/janetkuo))
* Add IPv6 address support for pods - does NOT include services ([#23090](https://github.com/kubernetes/kubernetes/pull/23090), [@tgraf](https://github.com/tgraf))
* Use local disk for ConfigMap volume instead of tmpfs ([#25306](https://github.com/kubernetes/kubernetes/pull/25306), [@pmorie](https://github.com/pmorie))
* Alpha support for scheduling pods on machines with NVIDIA GPUs whose kubelets use the `--experimental-nvidia-gpus` flag, using the alpha.kubernetes.io/nvidia-gpu resource  ([#24836](https://github.com/kubernetes/kubernetes/pull/24836), [@therc](https://github.com/therc))
* AWS: SSL support for ELB listeners through annotations ([#23495](https://github.com/kubernetes/kubernetes/pull/23495), [@therc](https://github.com/therc))
* Implement `kubectl rollout status` that can be used to watch a deployment's rollout status ([#19946](https://github.com/kubernetes/kubernetes/pull/19946), [@janetkuo](https://github.com/janetkuo))
* Webhook Token Authenticator ([#24902](https://github.com/kubernetes/kubernetes/pull/24902), [@cjcullen](https://github.com/cjcullen))
* Update PodSecurityPolicy types and add admission controller that could enforce them ([#24600](https://github.com/kubernetes/kubernetes/pull/24600), [@pweil-](https://github.com/pweil-))
* Introducing ScheduledJobs as described in [the proposal](docs/proposals/scheduledjob.md) as part of `batch/v2alpha1` version (experimental feature). ([#24970](https://github.com/kubernetes/kubernetes/pull/24970), [@soltysh](https://github.com/soltysh))
* kubectl now supports validation of nested objects with different ApiGroups (e.g. objects in a List) ([#25172](https://github.com/kubernetes/kubernetes/pull/25172), [@pwittrock](https://github.com/pwittrock))
* Change default clusterCIDRs from /16 to /14 in GCE configs allowing 1000 Node clusters by default. ([#25350](https://github.com/kubernetes/kubernetes/pull/25350), [@gmarek](https://github.com/gmarek))
* Add 'kubectl set' ([#25444](https://github.com/kubernetes/kubernetes/pull/25444), [@janetkuo](https://github.com/janetkuo))
* vSphere Cloud Provider Implementation  ([#24703](https://github.com/kubernetes/kubernetes/pull/24703), [@dagnello](https://github.com/dagnello))
* Added JobTemplate, a preliminary step for ScheduledJob and Workflow ([#21675](https://github.com/kubernetes/kubernetes/pull/21675), [@soltysh](https://github.com/soltysh))
* Openstack provider ([#21737](https://github.com/kubernetes/kubernetes/pull/21737), [@zreigz](https://github.com/zreigz))
* AWS kube-up: Allow VPC CIDR to be specified (experimental) ([#23362](https://github.com/kubernetes/kubernetes/pull/23362), [@miguelfrde](https://github.com/miguelfrde))
* Return "410 Gone" errors via watch stream when using watch cache ([#25369](https://github.com/kubernetes/kubernetes/pull/25369), [@liggitt](https://github.com/liggitt))
* GKE provider: Add cluster-ipv4-cidr and arbitrary flags ([#25437](https://github.com/kubernetes/kubernetes/pull/25437), [@zmerlynn](https://github.com/zmerlynn))
* AWS kube-up: Increase timeout waiting for docker start ([#25405](https://github.com/kubernetes/kubernetes/pull/25405), [@justinsb](https://github.com/justinsb))
* Sort resources in quota errors to avoid duplicate events ([#25161](https://github.com/kubernetes/kubernetes/pull/25161), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Display line number on JSON errors ([#25038](https://github.com/kubernetes/kubernetes/pull/25038), [@mfojtik](https://github.com/mfojtik))
* If the cluster node count exceeds the GCE TargetPool maximum (currently 1000), ([#25178](https://github.com/kubernetes/kubernetes/pull/25178), [@zmerlynn](https://github.com/zmerlynn))
    * randomly select which nodes are members of Kubernetes External Load Balancers.
* Clarify supported version skew between masters, nodes, and clients ([#25087](https://github.com/kubernetes/kubernetes/pull/25087), [@ihmccreery](https://github.com/ihmccreery))
* Move godeps to vendor/ ([#24242](https://github.com/kubernetes/kubernetes/pull/24242), [@thockin](https://github.com/thockin))
* Introduce events flag for describers ([#24554](https://github.com/kubernetes/kubernetes/pull/24554), [@ingvagabund](https://github.com/ingvagabund))
* run kube-addon-manager in a static pod ([#23600](https://github.com/kubernetes/kubernetes/pull/23600), [@mikedanese](https://github.com/mikedanese))
* Reimplement 'pause' in C - smaller footprint all around ([#23009](https://github.com/kubernetes/kubernetes/pull/23009), [@uluyol](https://github.com/uluyol))
* Add subPath to mount a child dir or file of a volumeMount ([#22575](https://github.com/kubernetes/kubernetes/pull/22575), [@MikaelCluseau](https://github.com/MikaelCluseau))
* Handle image digests in node status and image GC ([#25088](https://github.com/kubernetes/kubernetes/pull/25088), [@ncdc](https://github.com/ncdc))
* PLEG: reinspect pods that failed prior inspections ([#25077](https://github.com/kubernetes/kubernetes/pull/25077), [@ncdc](https://github.com/ncdc))
* Fix kubectl create secret/configmap to allow = values ([#24989](https://github.com/kubernetes/kubernetes/pull/24989), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Upgrade installed packages when building hyperkube to improve the security profile ([#25114](https://github.com/kubernetes/kubernetes/pull/25114), [@aaronlevy](https://github.com/aaronlevy))
* GCI/Trusty: Support ABAC authorization ([#24950](https://github.com/kubernetes/kubernetes/pull/24950), [@andyzheng0831](https://github.com/andyzheng0831))
* fix cinder volume dir umount issue [#24717](https://github.com/kubernetes/kubernetes/pull/24717) ([#24718](https://github.com/kubernetes/kubernetes/pull/24718), [@chengyli](https://github.com/chengyli))
* Inter pod topological affinity and anti-affinity implementation ([#22985](https://github.com/kubernetes/kubernetes/pull/22985), [@kevin-wangzefeng](https://github.com/kevin-wangzefeng))
* start etcd compactor in background ([#25010](https://github.com/kubernetes/kubernetes/pull/25010), [@hongchaodeng](https://github.com/hongchaodeng))
* GCI: Add two GCI specific metadata pairs ([#25105](https://github.com/kubernetes/kubernetes/pull/25105), [@andyzheng0831](https://github.com/andyzheng0831))
* Ensure status is not changed during an update of PV, PVC, HPA objects ([#24924](https://github.com/kubernetes/kubernetes/pull/24924), [@mqliang](https://github.com/mqliang))
* GCE: Prefer preconfigured node tags for firewalls, if available ([#25148](https://github.com/kubernetes/kubernetes/pull/25148), [@a-robinson](https://github.com/a-robinson))
* kubectl rolling-update support for same image ([#24645](https://github.com/kubernetes/kubernetes/pull/24645), [@jlowdermilk](https://github.com/jlowdermilk))
* Add an entry to the salt config to allow Debian jessie on GCE. ([#25123](https://github.com/kubernetes/kubernetes/pull/25123), [@jlewi](https://github.com/jlewi))
    * As with the existing Wheezy image on GCE, docker is expected
    * to already be installed in the image.
* Mark kube-push.sh as broken ([#25095](https://github.com/kubernetes/kubernetes/pull/25095), [@ihmccreery](https://github.com/ihmccreery))
* AWS: Add support for ap-northeast-2 region (Seoul) ([#24457](https://github.com/kubernetes/kubernetes/pull/24457), [@leokhoa](https://github.com/leokhoa))
* GCI: Update the command to get the image ([#24987](https://github.com/kubernetes/kubernetes/pull/24987), [@andyzheng0831](https://github.com/andyzheng0831))
* Port-forward: use out and error streams instead of glog ([#17030](https://github.com/kubernetes/kubernetes/pull/17030), [@csrwng](https://github.com/csrwng))
* Promote Pod Hostname & Subdomain to fields (were annotations) ([#24362](https://github.com/kubernetes/kubernetes/pull/24362), [@ArtfulCoder](https://github.com/ArtfulCoder))
* Validate deletion timestamp doesn't change on update ([#24839](https://github.com/kubernetes/kubernetes/pull/24839), [@liggitt](https://github.com/liggitt))
* Add flag -t as shorthand for --tty ([#24365](https://github.com/kubernetes/kubernetes/pull/24365), [@janetkuo](https://github.com/janetkuo))
* Add support for running clusters on GCI ([#24893](https://github.com/kubernetes/kubernetes/pull/24893), [@andyzheng0831](https://github.com/andyzheng0831))
* Switch to ABAC authorization from AllowAll ([#24210](https://github.com/kubernetes/kubernetes/pull/24210), [@cjcullen](https://github.com/cjcullen))
* Fix DeletingLoadBalancer event generation. ([#24833](https://github.com/kubernetes/kubernetes/pull/24833), [@a-robinson](https://github.com/a-robinson))



# v1.2.4

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.4/kubernetes.tar.gz) | `f3aea83f8f0e16b2b41998a2edc09eb42fd8d945` | `ab0aca3a20e8eba43c8ff9d672793618`

## Changes since v1.2.3

### Other notable changes

* Ensure status is not changed during an update of PV, PVC, HPA objects ([#24924](https://github.com/kubernetes/kubernetes/pull/24924), [@mqliang](https://github.com/mqliang))
* GCI: Add two GCI specific metadata pairs ([#25105](https://github.com/kubernetes/kubernetes/pull/25105), [@andyzheng0831](https://github.com/andyzheng0831))
* Add an entry to the salt config to allow Debian jessie on GCE. ([#25123](https://github.com/kubernetes/kubernetes/pull/25123), [@jlewi](https://github.com/jlewi))
    * As with the existing Wheezy image on GCE, docker is expected
    * to already be installed in the image.
* Fix DeletingLoadBalancer event generation. ([#24833](https://github.com/kubernetes/kubernetes/pull/24833), [@a-robinson](https://github.com/a-robinson))
* GCE: Prefer preconfigured node tags for firewalls, if available ([#25148](https://github.com/kubernetes/kubernetes/pull/25148), [@a-robinson](https://github.com/a-robinson))
* Drain pods created from ReplicaSets in 'kubectl drain' ([#23689](https://github.com/kubernetes/kubernetes/pull/23689), [@maclof](https://github.com/maclof))
* GCI: Update the command to get the image ([#24987](https://github.com/kubernetes/kubernetes/pull/24987), [@andyzheng0831](https://github.com/andyzheng0831))
* Validate deletion timestamp doesn't change on update ([#24839](https://github.com/kubernetes/kubernetes/pull/24839), [@liggitt](https://github.com/liggitt))
* Add support for running clusters on GCI ([#24893](https://github.com/kubernetes/kubernetes/pull/24893), [@andyzheng0831](https://github.com/andyzheng0831))
* Trusty: Add retry in curl commands ([#24749](https://github.com/kubernetes/kubernetes/pull/24749), [@andyzheng0831](https://github.com/andyzheng0831))



# v1.3.0-alpha.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-alpha.3/kubernetes.tar.gz) | `01e0dc68653173614dc99f44875173478f837b38` | `ae22c35f3a963743d21daa17683e0288`

## Changes since v1.3.0-alpha.2

### Action Required

* Updating go-restful to generate "type":"object" instead of "type":"any" in swagger-spec (breaks kubectl 1.1) ([#22897](https://github.com/kubernetes/kubernetes/pull/22897), [@nikhiljindal](https://github.com/nikhiljindal))
* Make watch cache treat resourceVersion consistent with uncached watch ([#24008](https://github.com/kubernetes/kubernetes/pull/24008), [@liggitt](https://github.com/liggitt))

### Other notable changes

* Trusty: Add retry in curl commands ([#24749](https://github.com/kubernetes/kubernetes/pull/24749), [@andyzheng0831](https://github.com/andyzheng0831))
* Collect and expose runtime's image storage usage via Kubelet's /stats/summary endpoint ([#23595](https://github.com/kubernetes/kubernetes/pull/23595), [@vishh](https://github.com/vishh))
* Adding loadBalancer services to quota system ([#24247](https://github.com/kubernetes/kubernetes/pull/24247), [@sdminonne](https://github.com/sdminonne))
* Enforce --max-pods in kubelet admission; previously was only enforced in scheduler ([#24674](https://github.com/kubernetes/kubernetes/pull/24674), [@gmarek](https://github.com/gmarek))
* All clients under ClientSet share one RateLimiter. ([#24166](https://github.com/kubernetes/kubernetes/pull/24166), [@gmarek](https://github.com/gmarek))
* Remove requirement that Endpoints IPs be IPv4 ([#23317](https://github.com/kubernetes/kubernetes/pull/23317), [@aanm](https://github.com/aanm))
* Fix unintended change of Service.spec.ports[].nodePort during kubectl apply ([#24180](https://github.com/kubernetes/kubernetes/pull/24180), [@AdoHe](https://github.com/AdoHe))
* Don't log private SSH key ([#24506](https://github.com/kubernetes/kubernetes/pull/24506), [@timstclair](https://github.com/timstclair))
* Incremental improvements to kubelet e2e tests ([#24426](https://github.com/kubernetes/kubernetes/pull/24426), [@pwittrock](https://github.com/pwittrock))
* Bridge off-cluster traffic into services by masquerading. ([#24429](https://github.com/kubernetes/kubernetes/pull/24429), [@cjcullen](https://github.com/cjcullen))
* Flush conntrack state for removed/changed UDP Services ([#22573](https://github.com/kubernetes/kubernetes/pull/22573), [@freehan](https://github.com/freehan))
* Allow setting the Host header in a httpGet probe ([#24292](https://github.com/kubernetes/kubernetes/pull/24292), [@errm](https://github.com/errm))
* Fix goroutine leak in ssh-tunnel healthcheck. ([#24487](https://github.com/kubernetes/kubernetes/pull/24487), [@cjcullen](https://github.com/cjcullen))
* Fix gce.getDiskByNameUnknownZone logic. ([#24452](https://github.com/kubernetes/kubernetes/pull/24452), [@a-robinson](https://github.com/a-robinson))
* Make etcd cache size configurable ([#23914](https://github.com/kubernetes/kubernetes/pull/23914), [@jsravn](https://github.com/jsravn))
* Drain pods created from ReplicaSets in 'kubectl drain' ([#23689](https://github.com/kubernetes/kubernetes/pull/23689), [@maclof](https://github.com/maclof))
* Make kubectl edit not convert GV on edits ([#23437](https://github.com/kubernetes/kubernetes/pull/23437), [@DirectXMan12](https://github.com/DirectXMan12))
* don't ship kube-registry-proxy and pause images in tars. ([#23605](https://github.com/kubernetes/kubernetes/pull/23605), [@mikedanese](https://github.com/mikedanese))
* Do not throw creation errors for containers that fail immediately after being started ([#23894](https://github.com/kubernetes/kubernetes/pull/23894), [@vishh](https://github.com/vishh))
* Add a client flag to delete "--now" for grace period 0 ([#23756](https://github.com/kubernetes/kubernetes/pull/23756), [@smarterclayton](https://github.com/smarterclayton))
* add act-as powers ([#23549](https://github.com/kubernetes/kubernetes/pull/23549), [@deads2k](https://github.com/deads2k))
* Build Kubernetes, etcd and flannel for arm64 and ppc64le ([#23931](https://github.com/kubernetes/kubernetes/pull/23931), [@luxas](https://github.com/luxas))
* Honor starting resourceVersion in watch cache ([#24208](https://github.com/kubernetes/kubernetes/pull/24208), [@ncdc](https://github.com/ncdc))
* Update the pause image to build for arm64 and ppc64le ([#23697](https://github.com/kubernetes/kubernetes/pull/23697), [@luxas](https://github.com/luxas))
* Return more useful error information when a persistent volume fails to mount ([#23122](https://github.com/kubernetes/kubernetes/pull/23122), [@screeley44](https://github.com/screeley44))
* Trusty: Avoid unnecessary in-memory temp files ([#24144](https://github.com/kubernetes/kubernetes/pull/24144), [@andyzheng0831](https://github.com/andyzheng0831))
* e2e: fix error checking in kubelet stats ([#24205](https://github.com/kubernetes/kubernetes/pull/24205), [@yujuhong](https://github.com/yujuhong))
* Fixed mounting with containerized kubelet ([#23435](https://github.com/kubernetes/kubernetes/pull/23435), [@jsafrane](https://github.com/jsafrane))
* Adding nodeports services to quota ([#22154](https://github.com/kubernetes/kubernetes/pull/22154), [@sdminonne](https://github.com/sdminonne))
* e2e: adapt kubelet_perf.go to use the new summary metrics API ([#24003](https://github.com/kubernetes/kubernetes/pull/24003), [@yujuhong](https://github.com/yujuhong))
* kubelet: add RSS memory to the summary API ([#24015](https://github.com/kubernetes/kubernetes/pull/24015), [@yujuhong](https://github.com/yujuhong))



# v1.2.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.3/kubernetes.tar.gz) | `b2ce4e0c72562d09ba06e3c0913f0bd78da0285e` | `69e75650de30d5a52d144799e94a168d`

## Changes since v1.2.2

### Action Required

* Make watch cache treat resourceVersion consistent with uncached watch ([#24008](https://github.com/kubernetes/kubernetes/pull/24008), [@liggitt](https://github.com/liggitt))

### Other notable changes

* Fix unintended change of Service.spec.ports[].nodePort during kubectl apply ([#24180](https://github.com/kubernetes/kubernetes/pull/24180), [@AdoHe](https://github.com/AdoHe))
* Flush conntrack state for removed/changed UDP Services ([#22573](https://github.com/kubernetes/kubernetes/pull/22573), [@freehan](https://github.com/freehan))
* Allow setting the Host header in a httpGet probe ([#24292](https://github.com/kubernetes/kubernetes/pull/24292), [@errm](https://github.com/errm))
* Bridge off-cluster traffic into services by masquerading. ([#24429](https://github.com/kubernetes/kubernetes/pull/24429), [@cjcullen](https://github.com/cjcullen))
* Version-guard Kubectl client Guestbook application test against deployments ([#24478](https://github.com/kubernetes/kubernetes/pull/24478), [@ihmccreery](https://github.com/ihmccreery))
* Fix goroutine leak in ssh-tunnel healthcheck. ([#24487](https://github.com/kubernetes/kubernetes/pull/24487), [@cjcullen](https://github.com/cjcullen))
* Fixed mounting with containerized kubelet ([#23435](https://github.com/kubernetes/kubernetes/pull/23435), [@jsafrane](https://github.com/jsafrane))
* Do not throw creation errors for containers that fail immediately after being started ([#23894](https://github.com/kubernetes/kubernetes/pull/23894), [@vishh](https://github.com/vishh))
* Honor starting resourceVersion in watch cache ([#24208](https://github.com/kubernetes/kubernetes/pull/24208), [@ncdc](https://github.com/ncdc))
* Fix TerminationMessagePath ([#23658](https://github.com/kubernetes/kubernetes/pull/23658), [@Random-Liu](https://github.com/Random-Liu))
* Fix gce.getDiskByNameUnknownZone logic. ([#24452](https://github.com/kubernetes/kubernetes/pull/24452), [@a-robinson](https://github.com/a-robinson))
* kubelet: add RSS memory to the summary API ([#24015](https://github.com/kubernetes/kubernetes/pull/24015), [@yujuhong](https://github.com/yujuhong))
* e2e: adapt kubelet_perf.go to use the new summary metrics API ([#24003](https://github.com/kubernetes/kubernetes/pull/24003), [@yujuhong](https://github.com/yujuhong))
* e2e: fix error checking in kubelet stats ([#24205](https://github.com/kubernetes/kubernetes/pull/24205), [@yujuhong](https://github.com/yujuhong))
* Trusty: Avoid unnecessary in-memory temp files ([#24144](https://github.com/kubernetes/kubernetes/pull/24144), [@andyzheng0831](https://github.com/andyzheng0831))
* Allowing type object in kubectl swagger validation ([#24054](https://github.com/kubernetes/kubernetes/pull/24054), [@nikhiljindal](https://github.com/nikhiljindal))
* Add ClusterUpgrade tests ([#24150](https://github.com/kubernetes/kubernetes/pull/24150), [@ihmccreery](https://github.com/ihmccreery))
* Trusty: Do not create the docker-daemon cgroup ([#23996](https://github.com/kubernetes/kubernetes/pull/23996), [@andyzheng0831](https://github.com/andyzheng0831))



# v1.3.0-alpha.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/master/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-alpha.2/kubernetes.tar.gz) | `305c8c2af7e99d463dbbe4208ecfe2b50585e796` | `aadb8d729d855e69212008f8fda628c0`

## Changes since v1.3.0-alpha.1

### Other notable changes

* Make kube2sky and skydns docker images cross-platform ([#19376](https://github.com/kubernetes/kubernetes/pull/19376), [@luxas](https://github.com/luxas))
* Allowing type object in kubectl swagger validation ([#24054](https://github.com/kubernetes/kubernetes/pull/24054), [@nikhiljindal](https://github.com/nikhiljindal))
* Fix TerminationMessagePath ([#23658](https://github.com/kubernetes/kubernetes/pull/23658), [@Random-Liu](https://github.com/Random-Liu))
* Trusty: Do not create the docker-daemon cgroup ([#23996](https://github.com/kubernetes/kubernetes/pull/23996), [@andyzheng0831](https://github.com/andyzheng0831))
* Make ConfigMap volume readable as non-root ([#23793](https://github.com/kubernetes/kubernetes/pull/23793), [@pmorie](https://github.com/pmorie))
* only include running and pending pods in daemonset should place calculation ([#23929](https://github.com/kubernetes/kubernetes/pull/23929), [@mikedanese](https://github.com/mikedanese))
* Upgrade to golang 1.6 ([#22149](https://github.com/kubernetes/kubernetes/pull/22149), [@luxas](https://github.com/luxas))
* Cross-build hyperkube and debian-iptables for ARM. Also add a flannel image ([#21617](https://github.com/kubernetes/kubernetes/pull/21617), [@luxas](https://github.com/luxas))
* Add a timeout to the sshDialer to prevent indefinite hangs. ([#23843](https://github.com/kubernetes/kubernetes/pull/23843), [@cjcullen](https://github.com/cjcullen))
* Ensure object returned by volume getCloudProvider incorporates cloud config ([#23769](https://github.com/kubernetes/kubernetes/pull/23769), [@saad-ali](https://github.com/saad-ali))
* Update Dashboard UI addon to v1.0.1 ([#23724](https://github.com/kubernetes/kubernetes/pull/23724), [@maciaszczykm](https://github.com/maciaszczykm))
* Add zsh completion for kubectl ([#23797](https://github.com/kubernetes/kubernetes/pull/23797), [@sttts](https://github.com/sttts))
* AWS kube-up: tolerate a lack of ephemeral volumes ([#23776](https://github.com/kubernetes/kubernetes/pull/23776), [@justinsb](https://github.com/justinsb))
* duplicate kube-apiserver to federated-apiserver ([#23509](https://github.com/kubernetes/kubernetes/pull/23509), [@jianhuiz](https://github.com/jianhuiz))
* Kubelet: Start using the official docker engine-api ([#23506](https://github.com/kubernetes/kubernetes/pull/23506), [@Random-Liu](https://github.com/Random-Liu))
* Fix so setup-files don't recreate/invalidate certificates that already exist ([#23550](https://github.com/kubernetes/kubernetes/pull/23550), [@luxas](https://github.com/luxas))
* A pod never terminated if a container image registry was unavailable ([#23746](https://github.com/kubernetes/kubernetes/pull/23746), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Fix jsonpath to handle maps with key of nonstring types ([#23358](https://github.com/kubernetes/kubernetes/pull/23358), [@aveshagarwal](https://github.com/aveshagarwal))
* Trusty: Regional release .tar.gz support ([#23558](https://github.com/kubernetes/kubernetes/pull/23558), [@andyzheng0831](https://github.com/andyzheng0831))
* Add support for 3rd party objects to kubectl ([#18835](https://github.com/kubernetes/kubernetes/pull/18835), [@brendandburns](https://github.com/brendandburns))
* Remove unnecessary override of /etc/init.d/docker on containervm image. ([#23593](https://github.com/kubernetes/kubernetes/pull/23593), [@dchen1107](https://github.com/dchen1107))
* make docker-checker more robust ([#23662](https://github.com/kubernetes/kubernetes/pull/23662), [@ArtfulCoder](https://github.com/ArtfulCoder))
* Change kube-proxy & fluentd CPU request to 20m/80m. ([#23646](https://github.com/kubernetes/kubernetes/pull/23646), [@cjcullen](https://github.com/cjcullen))
* Create a new Deployment in kube-system for every version. ([#23512](https://github.com/kubernetes/kubernetes/pull/23512), [@Q-Lee](https://github.com/Q-Lee))
* IngressTLS: allow secretName to be blank for SNI routing ([#23500](https://github.com/kubernetes/kubernetes/pull/23500), [@tam7t](https://github.com/tam7t))
* don't sync deployment when pod selector is empty ([#23467](https://github.com/kubernetes/kubernetes/pull/23467), [@mikedanese](https://github.com/mikedanese))
* AWS: Fix problems with >2 security groups ([#23340](https://github.com/kubernetes/kubernetes/pull/23340), [@justinsb](https://github.com/justinsb))



# v1.2.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.2/kubernetes.tar.gz) | `8dede5833a1986434adea80749624f81a0db7bb4` | `72a5389f22827fb5133fdc3b7bfb9b3a`

## Changes since v1.2.1

### Other notable changes

* Trusty: Update heapster manifest handling code ([#23434](https://github.com/kubernetes/kubernetes/pull/23434), [@andyzheng0831](https://github.com/andyzheng0831))
* Support addon Deployments, make heapster a deployment with a nanny. ([#22893](https://github.com/kubernetes/kubernetes/pull/22893), [@Q-Lee](https://github.com/Q-Lee))
* Create a new Deployment in kube-system for every version. ([#23512](https://github.com/kubernetes/kubernetes/pull/23512), [@Q-Lee](https://github.com/Q-Lee))
* Use SCP to dump logs and parallelize a bit. ([#22835](https://github.com/kubernetes/kubernetes/pull/22835), [@spxtr](https://github.com/spxtr))
* Trusty: Regional release .tar.gz support ([#23558](https://github.com/kubernetes/kubernetes/pull/23558), [@andyzheng0831](https://github.com/andyzheng0831))
* Make ConfigMap volume readable as non-root ([#23793](https://github.com/kubernetes/kubernetes/pull/23793), [@pmorie](https://github.com/pmorie))
* only include running and pending pods in daemonset should place calculation ([#23929](https://github.com/kubernetes/kubernetes/pull/23929), [@mikedanese](https://github.com/mikedanese))
* A pod never terminated if a container image registry was unavailable ([#23746](https://github.com/kubernetes/kubernetes/pull/23746), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Update Dashboard UI addon to v1.0.1 ([#23724](https://github.com/kubernetes/kubernetes/pull/23724), [@maciaszczykm](https://github.com/maciaszczykm))
* Ensure object returned by volume getCloudProvider incorporates cloud config ([#23769](https://github.com/kubernetes/kubernetes/pull/23769), [@saad-ali](https://github.com/saad-ali))
* Add a timeout to the sshDialer to prevent indefinite hangs. ([#23843](https://github.com/kubernetes/kubernetes/pull/23843), [@cjcullen](https://github.com/cjcullen))
* AWS kube-up: tolerate a lack of ephemeral volumes ([#23776](https://github.com/kubernetes/kubernetes/pull/23776), [@justinsb](https://github.com/justinsb))
* Fix so setup-files don't recreate/invalidate certificates that already exist ([#23550](https://github.com/kubernetes/kubernetes/pull/23550), [@luxas](https://github.com/luxas))

# v1.2.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.1/kubernetes.tar.gz) | `1639807c5788e1c6b1ab51fd30b723fb5debd865` | `235a1da47972c96a560d718d3256ca4f`


## Changes since v1.2.0

### Other notable changes

* AWS: Fix problems with >2 security groups ([#23340](https://github.com/kubernetes/kubernetes/pull/23340), [@justinsb](https://github.com/justinsb))
* IngressTLS: allow secretName to be blank for SNI routing ([#23500](https://github.com/kubernetes/kubernetes/pull/23500), [@tam7t](https://github.com/tam7t))
* Heapster patch release to 1.0.2 ([#23487](https://github.com/kubernetes/kubernetes/pull/23487), [@piosz](https://github.com/piosz))
* Remove unnecessary override of /etc/init.d/docker on containervm image. ([#23593](https://github.com/kubernetes/kubernetes/pull/23593), [@dchen1107](https://github.com/dchen1107))
* Change kube-proxy & fluentd CPU request to 20m/80m. ([#23646](https://github.com/kubernetes/kubernetes/pull/23646), [@cjcullen](https://github.com/cjcullen))
* make docker-checker more robust ([#23662](https://github.com/kubernetes/kubernetes/pull/23662), [@ArtfulCoder](https://github.com/ArtfulCoder))
* validate that daemonsets don't have empty selectors on creation ([#23530](https://github.com/kubernetes/kubernetes/pull/23530), [@mikedanese](https://github.com/mikedanese))
* don't sync deployment when pod selector is empty ([#23467](https://github.com/kubernetes/kubernetes/pull/23467), [@mikedanese](https://github.com/mikedanese))
* Support differentiation of OS distro in e2e tests ([#23466](https://github.com/kubernetes/kubernetes/pull/23466), [@andyzheng0831](https://github.com/andyzheng0831))
* don't sync daemonsets with selectors that match all pods ([#23223](https://github.com/kubernetes/kubernetes/pull/23223), [@mikedanese](https://github.com/mikedanese))
* Trusty: Avoid reaching GCE custom metadata size limit ([#22818](https://github.com/kubernetes/kubernetes/pull/22818), [@andyzheng0831](https://github.com/andyzheng0831))
* Update kubectl help for 1.2 resources ([#23305](https://github.com/kubernetes/kubernetes/pull/23305), [@janetkuo](https://github.com/janetkuo))
* Removing URL query param from swagger UI to fix the XSS issue ([#23234](https://github.com/kubernetes/kubernetes/pull/23234), [@nikhiljindal](https://github.com/nikhiljindal))
* Fix hairpin mode ([#23325](https://github.com/kubernetes/kubernetes/pull/23325), [@MurgaNikolay](https://github.com/MurgaNikolay))
* Bump to container-vm-v20160321 ([#23313](https://github.com/kubernetes/kubernetes/pull/23313), [@zmerlynn](https://github.com/zmerlynn))
* Remove the restart-kube-proxy and restart-apiserver functions ([#23180](https://github.com/kubernetes/kubernetes/pull/23180), [@roberthbailey](https://github.com/roberthbailey))
* Copy annotations back from RS to Deployment on rollback ([#23160](https://github.com/kubernetes/kubernetes/pull/23160), [@janetkuo](https://github.com/janetkuo))
* Trusty: Support hybrid cluster with nodes on ContainerVM ([#23079](https://github.com/kubernetes/kubernetes/pull/23079), [@andyzheng0831](https://github.com/andyzheng0831))
* update expose command description to add deployment ([#23246](https://github.com/kubernetes/kubernetes/pull/23246), [@AdoHe](https://github.com/AdoHe))
* Add a rate limiter to the GCE cloudprovider ([#23019](https://github.com/kubernetes/kubernetes/pull/23019), [@alex-mohr](https://github.com/alex-mohr))
* Add a Deployment example for kubectl expose. ([#23222](https://github.com/kubernetes/kubernetes/pull/23222), [@madhusudancs](https://github.com/madhusudancs))
* Use versioned object when computing patch ([#23145](https://github.com/kubernetes/kubernetes/pull/23145), [@liggitt](https://github.com/liggitt))
* kubelet: send all recevied pods in one update ([#23141](https://github.com/kubernetes/kubernetes/pull/23141), [@yujuhong](https://github.com/yujuhong))
* Add a SSHKey sync check to the master's healthz (when using SSHTunnels). ([#23167](https://github.com/kubernetes/kubernetes/pull/23167), [@cjcullen](https://github.com/cjcullen))
* Validate minimum CPU limits to be >= 10m ([#23143](https://github.com/kubernetes/kubernetes/pull/23143), [@vishh](https://github.com/vishh))
* Fix controller-manager race condition issue which cause endpoints flush during restart ([#23035](https://github.com/kubernetes/kubernetes/pull/23035), [@xinxiaogang](https://github.com/xinxiaogang))
* MESOS: forward globally declared cadvisor housekeeping flags ([#22974](https://github.com/kubernetes/kubernetes/pull/22974), [@jdef](https://github.com/jdef))
* Trusty: support developer workflow on base image ([#22960](https://github.com/kubernetes/kubernetes/pull/22960), [@andyzheng0831](https://github.com/andyzheng0831))



# v1.3.0-alpha.1

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/HEAD/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.0-alpha.1/kubernetes.tar.gz) | `e0041b08e220a4704ea2ad90a6ec7c8f2120c2d3` | `7bb2df32aea94678f72a8d1f43a12098`

## Changes since v1.2.0

### Action Required

* Disabling swagger ui by default on apiserver. Adding a flag that can enable it ([#23025](https://github.com/kubernetes/kubernetes/pull/23025), [@nikhiljindal](https://github.com/nikhiljindal))
* restore ability to run against secured etcd ([#21535](https://github.com/kubernetes/kubernetes/pull/21535), [@AdoHe](https://github.com/AdoHe))

### Other notable changes

* validate that daemonsets don't have empty selectors on creation ([#23530](https://github.com/kubernetes/kubernetes/pull/23530), [@mikedanese](https://github.com/mikedanese))
* Trusty: Update heapster manifest handling code ([#23434](https://github.com/kubernetes/kubernetes/pull/23434), [@andyzheng0831](https://github.com/andyzheng0831))
* Support differentiation of OS distro in e2e tests ([#23466](https://github.com/kubernetes/kubernetes/pull/23466), [@andyzheng0831](https://github.com/andyzheng0831))
* don't sync daemonsets with selectors that match all pods ([#23223](https://github.com/kubernetes/kubernetes/pull/23223), [@mikedanese](https://github.com/mikedanese))
* Trusty: Avoid reaching GCE custom metadata size limit ([#22818](https://github.com/kubernetes/kubernetes/pull/22818), [@andyzheng0831](https://github.com/andyzheng0831))
* Update kubectl help for 1.2 resources ([#23305](https://github.com/kubernetes/kubernetes/pull/23305), [@janetkuo](https://github.com/janetkuo))
* Support addon Deployments, make heapster a deployment with a nanny. ([#22893](https://github.com/kubernetes/kubernetes/pull/22893), [@Q-Lee](https://github.com/Q-Lee))
* Removing URL query param from swagger UI to fix the XSS issue ([#23234](https://github.com/kubernetes/kubernetes/pull/23234), [@nikhiljindal](https://github.com/nikhiljindal))
* Fix hairpin mode ([#23325](https://github.com/kubernetes/kubernetes/pull/23325), [@MurgaNikolay](https://github.com/MurgaNikolay))
* Bump to container-vm-v20160321 ([#23313](https://github.com/kubernetes/kubernetes/pull/23313), [@zmerlynn](https://github.com/zmerlynn))
* Remove the restart-kube-proxy and restart-apiserver functions ([#23180](https://github.com/kubernetes/kubernetes/pull/23180), [@roberthbailey](https://github.com/roberthbailey))
* Copy annotations back from RS to Deployment on rollback ([#23160](https://github.com/kubernetes/kubernetes/pull/23160), [@janetkuo](https://github.com/janetkuo))
* Trusty: Support hybrid cluster with nodes on ContainerVM ([#23079](https://github.com/kubernetes/kubernetes/pull/23079), [@andyzheng0831](https://github.com/andyzheng0831))
* update expose command description to add deployment ([#23246](https://github.com/kubernetes/kubernetes/pull/23246), [@AdoHe](https://github.com/AdoHe))
* Add a rate limiter to the GCE cloudprovider ([#23019](https://github.com/kubernetes/kubernetes/pull/23019), [@alex-mohr](https://github.com/alex-mohr))
* Add a Deployment example for kubectl expose. ([#23222](https://github.com/kubernetes/kubernetes/pull/23222), [@madhusudancs](https://github.com/madhusudancs))
* Use versioned object when computing patch ([#23145](https://github.com/kubernetes/kubernetes/pull/23145), [@liggitt](https://github.com/liggitt))
* kubelet: send all recevied pods in one update ([#23141](https://github.com/kubernetes/kubernetes/pull/23141), [@yujuhong](https://github.com/yujuhong))
* Add a SSHKey sync check to the master's healthz (when using SSHTunnels). ([#23167](https://github.com/kubernetes/kubernetes/pull/23167), [@cjcullen](https://github.com/cjcullen))
* Validate minimum CPU limits to be >= 10m ([#23143](https://github.com/kubernetes/kubernetes/pull/23143), [@vishh](https://github.com/vishh))
* Fix controller-manager race condition issue which cause endpoints flush during restart ([#23035](https://github.com/kubernetes/kubernetes/pull/23035), [@xinxiaogang](https://github.com/xinxiaogang))
* MESOS: forward globally declared cadvisor housekeeping flags ([#22974](https://github.com/kubernetes/kubernetes/pull/22974), [@jdef](https://github.com/jdef))
* Trusty: support developer workflow on base image ([#22960](https://github.com/kubernetes/kubernetes/pull/22960), [@andyzheng0831](https://github.com/andyzheng0831))
* Bumped Heapster to stable version 1.0.0 ([#22993](https://github.com/kubernetes/kubernetes/pull/22993), [@piosz](https://github.com/piosz))
* Deprecating --api-version flag ([#22410](https://github.com/kubernetes/kubernetes/pull/22410), [@nikhiljindal](https://github.com/nikhiljindal))
* allow resource.version.group in kubectl ([#22853](https://github.com/kubernetes/kubernetes/pull/22853), [@deads2k](https://github.com/deads2k))
* Use SCP to dump logs and parallelize a bit. ([#22835](https://github.com/kubernetes/kubernetes/pull/22835), [@spxtr](https://github.com/spxtr))
* update wide option output ([#22772](https://github.com/kubernetes/kubernetes/pull/22772), [@AdoHe](https://github.com/AdoHe))
* Change scheduler logic from random to round-robin ([#22430](https://github.com/kubernetes/kubernetes/pull/22430), [@gmarek](https://github.com/gmarek))



# v1.2.0

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.0/kubernetes.tar.gz) | `52dd998e1191f464f581a9b87017d70ce0b058d9` | `c0ce9e6150e9d7a19455db82f3318b4c`

## Changes since v1.1.1

### Major Themes

  * <strong>Significant scale improvements</strong>. Increased cluster scale by 400% to 1000 nodes with 30,000 pods per cluster.
Kubelet supports 100 pods per node with 4x reduced system overhead.
  * <strong>Simplified application deployment and management. </strong>
     * Dynamic Configuration (ConfigMap API in the core API group) enables application
configuration to be stored as a Kubernetes API object and pulled dynamically on
container startup, as an alternative to baking in command-line flags when a
container is built.
     * Turnkey Deployments (Deployment API (Beta) in the Extensions API group)
automate deployment and rolling updates of applications, specified
declaratively. It handles versioning, multiple simultaneous rollouts,
aggregating status across all pods, maintaining application availability, and
rollback.
  * <strong>Automated cluster management: </strong>
     * Kubernetes clusters can now span zones within a cloud provider. Pods from a
service will be automatically spread across zones, enabling applications to
tolerate zone failure.
     * Simplified way to run a container on every node (DaemonSet API (Beta) in the
Extensions API group): Kubernetes can schedule a service (such as a logging
agent) that runs one, and only one, pod per node.
     * TLS and L7 support (Ingress API (Beta) in the Extensions API group): Kubernetes
is now easier to integrate into custom networking environments by supporting
TLS for secure communication and L7 http-based traffic routing.
     * Graceful Node Shutdown (aka drain) - The new “kubectl drain” command gracefully
evicts pods from nodes in preparation for disruptive operations like kernel
upgrades or maintenance.
     * Custom Metrics for Autoscaling (HorizontalPodAutoscaler API in the Autoscaling
API group): The Horizontal Pod Autoscaling feature now supports custom metrics
(Alpha), allowing you to specify application-level metrics and thresholds to
trigger scaling up and down the number of pods in your application.
  * <strong>New GUI</strong> (dashboard) allows you to get started quickly and enables the same
functionality found in the CLI as a more approachable and discoverable way of
interacting with the system. Note: the GUI is enabled by default in 1.2 clusters.

<img src="docs/images/newgui.png" width="" alt="Dashboard UI screenshot showing cards that represent applications that run inside a cluster" title="Dashboard UI apps screen">

### Other notable improvements

  * Job was Beta in 1.1 and is GA in 1.2 .
     * <code>apiVersion: batch/v1 </code>is now available.  You now do not need to specify the <code>.spec.selector</code> field — a [unique selector is automatically generated ](http://kubernetes.io/docs/user-guide/jobs/#pod-selector)for you.
     * The previous version, <code>apiVersion: extensions/v1beta1</code>, is still supported.  Even if you roll back to 1.1, the objects created using
the new apiVersion will still be accessible, using the old version.   You can
continue to use your existing JSON and YAML files until you are ready to switch
to <code>batch/v1</code>.  We may remove support for Jobs with  <code>apiVersion: extensions/v1beta1 </code>in 1.3 or 1.4.
  *  HorizontalPodAutoscaler was Beta in 1.1 and is GA in 1.2 .
     * <code>apiVersion: autoscaling/v1 </code>is now available.  Changes in this version are:
        * Field CPUUtilization which was a nested structure CPUTargetUtilization in
HorizontalPodAutoscalerSpec was replaced by TargetCPUUtilizationPercentage
which is an integer.
        * ScaleRef of type SubresourceReference in HorizontalPodAutoscalerSpec which
referred to scale subresource of the resource being scaled was replaced by
ScaleTargetRef which points just to the resource being scaled.
        * In extensions/v1beta1 if CPUUtilization in HorizontalPodAutoscalerSpec was not
specified it was set to 80 by default while in autoscaling/v1 HPA object
without TargetCPUUtilizationPercentage specified is a valid object. Pod
autoscaler controller will apply a default scaling policy in this case which is
equivalent to the previous one but may change in the future.
     * The previous version, <code>apiVersion: extensions/v1beta1</code>, is still supported.  Even if you roll back to 1.1, the objects created using
the new apiVersions will still be accessible, using the old version.  You can
continue to use your existing JSON and YAML files until you are ready to switch
to <code>autoscaling/v1</code>.  We may remove support for HorizontalPodAutoscalers with  <code>apiVersion: extensions/v1beta1 </code>in 1.3 or 1.4.
  * Kube-Proxy now defaults to an iptables-based proxy. If the --proxy-mode flag is
specified while starting kube-proxy (‘userspace’ or ‘iptables’), the flag value
will be respected. If the flag value is not specified, the kube-proxy respects
the Node object annotation: ‘net.beta.kubernetes.io/proxy-mode’. If the
annotation is not specified, then ‘iptables’ mode is the default. If kube-proxy
is unable to start in iptables mode because system requirements are not met
(kernel or iptables versions are insufficient), the kube-proxy will fall-back
to userspace mode. Kube-proxy is much more performant and less
resource-intensive in ‘iptables’ mode.
  * Node stability can be improved by reserving [resources](https://github.com/kubernetes/kubernetes/blob/release-1.2/docs/proposals/node-allocatable.md) for the base operating system using --system-reserved and --kube-reserved Kubelet flags
  * Liveness and readiness probes now support more configuration parameters:
periodSeconds, successThreshold, failureThreshold
  * The new ReplicaSet API (Beta) in the Extensions API group is similar to
ReplicationController, but its [selector](http://kubernetes.io/docs/user-guide/labels/#label-selectors) is more general (supports set-based selector; whereas ReplicationController
only supports equality-based selector).
  * Scale subresource support is now expanded to ReplicaSets along with
ReplicationControllers and Deployments. Scale now supports two different types
of selectors to accommodate both [equality-based selectors](http://kubernetes.io/docs/user-guide/labels/#equality-based-requirement) supported by ReplicationControllers and [set-based selectors](http://kubernetes.io/docs/user-guide/labels/#set-based-requirement) supported by Deployments and ReplicaSets.
  * “kubectl run” now produces Deployments (instead of ReplicationControllers) and
Jobs (instead of Pods) by default.
  * Pods can now consume Secret data in environment variables and inject those
environment variables into a container’s command-line args.
  * Stable version of Heapster which scales up to 1000 nodes: more metrics, reduced
latency, reduced cpu/memory consumption (~4mb per monitored node).
  * Pods now have a security context which allows users to specify:
     * attributes which apply to the whole pod:
        * User ID
        * Whether all containers should be non-root
        * Supplemental Groups
        * FSGroup - a special supplemental group
        * SELinux options
     * If a pod defines an FSGroup, that Pod’s system (emptyDir, secret, configMap,
etc) volumes and block-device volumes will be owned by the FSGroup, and each
container in the pod will run with the FSGroup as a supplemental group
  * Volumes that support SELinux labelling are now automatically relabeled with the
Pod’s SELinux context, if specified
  * A stable client library release\_1\_2 is added. The library is [here](pkg/client/clientset_generated/), and detailed doc is [here](docs/devel/generating-clientset.md#released-clientsets). We will keep the interface of this go client stable.
  * New Azure File Service Volume Plugin enables mounting Microsoft Azure File
Volumes (SMB 2.1 and 3.0) into a Pod. See [example](https://github.com/kubernetes/kubernetes/blob/release-1.2/examples/azure_file/README.md) for details.
  * Logs usage and root filesystem usage of a container, volumes usage of a pod and node disk usage are exposed through Kubelet new metrics API.

### Experimental Features

  * Dynamic Provisioning of PersistentVolumes: Kubernetes previously required all
volumes to be manually provisioned by a cluster administrator before use. With
this feature, volume plugins that support it (GCE PD, AWS EBS, and Cinder) can
automatically provision a PersistentVolume to bind to an unfulfilled
PersistentVolumeClaim.
  * Run multiple schedulers in parallel, e.g. one or more custom schedulers
alongside the default Kubernetes scheduler, using pod annotations to select
among the schedulers for each pod. Documentation is [here](http://kubernetes.io/docs/admin/multiple-schedulers.md), design doc is [here](docs/proposals/multiple-schedulers.md).
  * More expressive node affinity syntax, and support for “soft” node affinity.
Node selectors (to constrain pods to schedule on a subset of nodes) now support
the operators {<code>In, NotIn, Exists, DoesNotExist, Gt, Lt</code>}  instead of just conjunction of exact match on node label values. In
addition, we’ve introduced a new “soft” kind of node selector that is just a
hint to the scheduler; the scheduler will try to satisfy these requests but it
does not guarantee they will be satisfied. Both the “hard” and “soft” variants
of node affinity use the new syntax. Documentation is [here](http://kubernetes.io/docs/user-guide/node-selection/) (see section “Alpha feature in Kubernetes v1.2: Node Affinity“). Design doc is [here](https://github.com/kubernetes/kubernetes/blob/release-1.2/docs/design/nodeaffinity.md).
  * A pod can specify its own Hostname and Subdomain via annotations (<code>pod.beta.kubernetes.io/hostname, pod.beta.kubernetes.io/subdomain)</code>. If the Subdomain matches the name of a [headless service](http://kubernetes.io/docs/user-guide/services/#headless-services) in the same namespace, a DNS A record is also created for the pod’s FQDN. More
details can be found in the [DNS README](https://github.com/kubernetes/kubernetes/blob/release-1.2/cluster/saltbase/salt/kube-dns/README.md#a-records-and-hostname-based-on-pod-annotations---a-beta-feature-in-kubernetes-v12). Changes were introduced in PR [#20688](https://github.com/kubernetes/kubernetes/pull/20688).
  * New SchedulerExtender enables users to implement custom
out-of-(the-scheduler)-process scheduling predicates and priority functions,
for example to schedule pods based on resources that are not directly managed
by Kubernetes. Changes were introduced in PR [#13580](https://github.com/kubernetes/kubernetes/pull/13580). Example configuration and documentation is available [here](docs/design/scheduler_extender.md). This is an alpha feature and may not be supported in its current form at beta
or GA.
  * New Flex Volume Plugin enables users to use out-of-process volume plugins that
are installed to “/usr/libexec/kubernetes/kubelet-plugins/volume/exec/” on
every node, instead of being compiled into the Kubernetes binary. See [example](examples/volumes/flexvolume/README.md) for details.
  * vendor volumes into a pod. It expects vendor drivers are installed in the
volume plugin path on each kubelet node. This is an alpha feature and may
change in future.
  * Kubelet exposes a new Alpha metrics API - /stats/summary in a user friendly format with reduced system overhead. The measurement is done in PR [#22542](https://github.com/kubernetes/kubernetes/pull/22542).

### Action required

  * Docker v1.9.1 is officially recommended. Docker v1.8.3 and Docker v1.10 are
supported. If you are using an older release of Docker, please upgrade. Known
issues with Docker 1.9.1 can be found below.
  * CPU hardcapping will be enabled by default for containers with CPU limit set,
if supported by the kernel. You should either adjust your CPU limit, or set CPU
request only, if you want to avoid hardcapping. If the kernel does not support
CPU Quota, NodeStatus will contain a warning indicating that CPU Limits cannot
be enforced.
  * The following applies only if you use the Go language client (<code>/pkg/client/unversioned</code>) to create Job by defining Go variables of type "<code>k8s.io/kubernetes/pkg/apis/extensions".Job</code>).  We think <strong>this is not common</strong>, so if you are not sure what this means, you probably aren't doing this.  If
you do this, then, at the time you re-vendor the "<code>k8s.io/kubernetes/"</code> code, you will need to set <code>job.Spec.ManualSelector = true</code>, or else set <code>job.Spec.Selector = nil.  </code>Otherwise, the jobs you create may be rejected.  See [Specifying your own pod selector](http://kubernetes.io/docs/user-guide/jobs/#specifying-your-own-pod-selector).
  * Deployment was Alpha in 1.1 (though it had apiVersion extensions/v1beta1) and
was disabled by default. Due to some non-backward-compatible API changes, any
Deployment objects you created in 1.1 won’t work with in the 1.2 release.
     * Before upgrading to 1.2, <strong>delete all Deployment alpha-version resources</strong>, including the Replication Controllers and Pods the Deployment manages. Then
create Deployment Beta resources after upgrading to 1.2. Not deleting the
Deployment objects may cause the deployment controller to mistakenly match
other pods and delete them, due to the selector API change.
     * Client (kubectl) and server versions must match (both 1.1 or both 1.2) for any
Deployment-related operations.
     * Behavior change:
        * Deployment creates ReplicaSets instead of ReplicationControllers.
        * Scale subresource now has a new <code>targetSelector</code> field in its status. This field supports the new set-based selectors supported
by Deployments, but in a serialized format.
     * Spec change:
        * Deployment’s [selector](http://kubernetes.io/docs/user-guide/labels/#label-selectors) is now more general (supports set-based selector; it only supported
equality-based selector in 1.1).
        * .spec.uniqueLabelKey is removed -- users can’t customize unique label key --
and its default value is changed from
“deployment.kubernetes.io/podTemplateHash” to “pod-template-hash”.
        * .spec.strategy.rollingUpdate.minReadySeconds is moved to .spec.minReadySeconds
  * DaemonSet was Alpha in 1.1 (though it had apiVersion extensions/v1beta1) and
was disabled by default. Due to some non-backward-compatible API changes, any
DaemonSet objects you created in 1.1 won’t work with in the 1.2 release.
     * Before upgrading to 1.2, <strong>delete all DaemonSet alpha-version resources</strong>. If you do not want to disrupt the pods, use kubectl delete daemonset <name>
--cascade=false. Then create DaemonSet Beta resources after upgrading to 1.2.
     * Client (kubectl) and server versions must match (both 1.1 or both 1.2) for any
DaemonSet-related operations.
     * Behavior change:
        * DaemonSet pods will be created on nodes with .spec.unschedulable=true and will
not be evicted from nodes whose Ready condition is false.
        * Updates to the pod template are now permitted. To perform a rolling update of a
DaemonSet, update the pod template and then delete its pods one by one; they
will be replaced using the updated template.
     * Spec change:
        * DaemonSet’s [selector](http://kubernetes.io/docs/user-guide/labels/#label-selectors) is now more general (supports set-based selector; it only supported
equality-based selector in 1.1).
  * Running against a secured etcd requires these flags to be passed to
kube-apiserver (instead of --etcd-config):
     * --etcd-certfile, --etcd-keyfile (if using client cert auth)
     * --etcd-cafile (if not using system roots)
  * As part of preparation in 1.2 for adding support for protocol buffers (and the
direct YAML support in the API available today), the Content-Type and Accept
headers are now properly handled as per the HTTP spec.  As a consequence, if
you had a client that was sending an invalid Content-Type or Accept header to
the API, in 1.2 you will either receive a 415 or 406 error.
The only client
this is known to affect is curl when you use -d with JSON but don't set a
content type, helpfully sends "application/x-www-urlencoded", which is not
correct.
Other client authors should double check that you are sending proper
accept and content type headers, or set no value (in which case JSON is the
default).
An example using curl:
<code>curl -H "Content-Type: application/json" -XPOST -d
'{"apiVersion":"v1","kind":"Namespace","metadata":{"name":"kube-system"}}' "[http://127.0.0.1:8080/api/v1/namespaces](http://127.0.0.1:8080/api/v1/namespaces)"</code>
  * The version of InfluxDB is bumped from 0.8 to 0.9 which means storage schema
change. More details [here](https://docs.influxdata.com/influxdb/v0.9/administration/upgrading/).
  * We have renamed “minions” to “nodes”.  If you were specifying NUM\_MINIONS or
MINION\_SIZE to kube-up, you should now specify NUM\_NODES or NODE\_SIZE.

### Known Issues

  * Paused deployments can't be resized and don't clean up old ReplicaSets.
  * Minimum memory limit is 4MB. This is a docker limitation
  * Minimum CPU limits is 10m. This is a Linux Kernel limitation
  * “kubectl rollout undo” (i.e. rollback) will hang on paused deployments, because
paused deployments can’t be rolled back (this is expected), and the command
waits for rollback events to return the result. Users should use “kubectl
rollout resume” to resume a deployment before rolling back.
  * “kubectl edit <list>” will open the editor multiple times, once for each
resource in the list.
  * If you create HPA object using autoscaling/v1 API without specifying
targetCPUUtilizationPercentage and read it using kubectl it will print default
value as specified in extensions/v1beta1 (see details in [#23196](https://github.com/kubernetes/kubernetes/issues/23196)).
  * If a node or kubelet crashes with a volume attached, the volume will remain
attached to that node. If that volume can only be attached to one node at a
time (GCE PDs attached in RW mode, for example), then the volume must be
manually detached before Kubernetes can attach it to other nodes.
  * If a volume is already attached to a node any subsequent attempts to attach it
again (due to kubelet restart, for example) will fail. The volume must either
be manually detached first or the pods referencing it deleted (which would
trigger automatic volume detach).
  * In very large clusters it may happen that a few nodes won’t register in API
server in a given timeframe for whatever reasons (networking issue, machine
failure, etc.). Normally when kube-up script will encounter even one NotReady
node it will fail, even though the cluster most likely will be working. We
added an environmental variable to kube-up ALLOWED\_NOTREADY\_NODES that
defines the number of nodes that if not Ready in time won’t cause kube-up
failure.
  * “kubectl rolling-update” only supports Replication Controllers (it doesn’t
support Replica Sets). It’s recommended to use Deployment 1.2 with “kubectl
rollout” commands instead, if you want to rolling update Replica Sets.
  * When live upgrading Kubelet to 1.2 without draining the pods running on the node,
the containers will be restarted by Kubelet (see details in [#23104](https://github.com/kubernetes/kubernetes/issues/23104)).

#### Docker Known Issues

##### 1.9.1

  * Listing containers can be slow at times which will affect kubelet performance.
More information [here](https://github.com/docker/docker/issues/17720)
  * Docker daemon restarts can fail. Docker checkpoints have to deleted between
restarts. More information [here](https://github.com/kubernetes/kubernetes/issues/20995)
  * Pod IP allocation-related issues. Deleting the docker checkpoint prior to
restarting the daemon alleviates this issue, but hasn’t been verified to
completely eliminate the IP allocation issue. More information [here](https://github.com/kubernetes/kubernetes/issues/21523#issuecomment-191498969)
  * Daemon becomes unresponsive (rarely) due to kernel deadlocks. More information [here](https://github.com/kubernetes/kubernetes/issues/21866#issuecomment-189492391)

### Provider-specific Notes

#### Various

   Core changes:

  * Support for load balancers with source ranges

#### AWS

Core changes:

  * Support for ELBs with complex configurations: better subnet selection with
multiple subnets, and internal ELBs
  * Support for VPCs with private dns names
  * Multiple fixes to EBS volume mounting code for robustness, and to support
mounting the full number of AWS recommended volumes.
  * Multiple fixes to avoid hitting AWS rate limits, and to throttle if we do
  * Support for the EC2 Container Registry (currently in us-east-1 only)

With kube-up:

  * Automatically install updates on boot & reboot
  * Use optimized image based on Jessie by default
  * Add support for Ubuntu Wily
  * Master is configured with automatic restart-on-failure, via CloudWatch
  * Bootstrap reworked to be more similar to GCE; better supports reboots/restarts
  * Use an elastic IP for the master by default
  * Experimental support for node spot instances (set NODE\_SPOT\_PRICE=0.05)

#### GCE

  * Ubuntu Trusty support added

Please see the [Releases Page](https://github.com/kubernetes/kubernetes/releases) for older releases.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CHANGELOG.md?pixel)]()
