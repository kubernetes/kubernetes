<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.15.0](#v1150)
  - [Downloads for v1.15.0](#downloads-for-v1150)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
- [Kubernetes v1.15 Release Notes](#kubernetes-v115-release-notes)
  - [1.15 What’s New](#115-whats-new)
      - [Continuous Improvement](#continuous-improvement)
      - [Extensibility](#extensibility)
    - [Extensibility around core Kubernetes APIs](#extensibility-around-core-kubernetes-apis)
      - [CustomResourceDefinitions Pruning](#customresourcedefinitions-pruning)
      - [CustomResourceDefinition Defaulting](#customresourcedefinition-defaulting)
      - [CustomResourceDefinition OpenAPI Publishing](#customresourcedefinition-openapi-publishing)
    - [Cluster Lifecycle Stability and Usability Improvements](#cluster-lifecycle-stability-and-usability-improvements)
    - [Continued improvement of CSI](#continued-improvement-of-csi)
      - [Additional Notable Feature Updates](#additional-notable-feature-updates)
  - [Known Issues](#known-issues)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
      - [API Machinery](#api-machinery)
      - [Apps](#apps)
      - [Auth](#auth)
      - [AWS](#aws)
      - [Azure](#azure)
      - [CLI](#cli)
      - [Lifecycle](#lifecycle)
      - [Network](#network)
      - [Node](#node)
      - [Storage](#storage)
  - [Deprecations and Removals](#deprecations-and-removals)
  - [Metrics Changes](#metrics-changes)
    - [Added metrics](#added-metrics)
    - [Deprecated/changed metrics](#deprecatedchanged-metrics)
  - [Notable Features](#notable-features)
    - [Stable](#stable)
    - [Beta](#beta)
    - [Alpha](#alpha)
    - [Staging Repositories](#staging-repositories)
    - [CLI Improvements](#cli-improvements)
    - [Misc](#misc)
  - [API Changes](#api-changes)
  - [Other notable changes](#other-notable-changes)
    - [API Machinery](#api-machinery-1)
    - [Apps](#apps-1)
    - [Auth](#auth-1)
    - [Autoscaling](#autoscaling)
    - [AWS](#aws-1)
    - [Azure](#azure-1)
    - [CLI](#cli-1)
    - [Cloud Provider](#cloud-provider)
    - [Cluster Lifecycle](#cluster-lifecycle)
    - [GCP](#gcp)
    - [Instrumentation](#instrumentation)
    - [Network](#network-1)
    - [Node](#node-1)
    - [OpenStack](#openstack)
    - [Release](#release)
    - [Scheduling](#scheduling)
    - [Storage](#storage-1)
    - [VMware](#vmware)
    - [Windows](#windows)
  - [Dependencies](#dependencies)
    - [Changed](#changed)
    - [Unchanged](#unchanged)
- [v1.15.0-rc.1](#v1150-rc1)
  - [Downloads for v1.15.0-rc.1](#downloads-for-v1150-rc1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.15.0-beta.2](#changelog-since-v1150-beta2)
    - [Other notable changes](#other-notable-changes-1)
- [v1.15.0-beta.2](#v1150-beta2)
  - [Downloads for v1.15.0-beta.2](#downloads-for-v1150-beta2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.15.0-beta.1](#changelog-since-v1150-beta1)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-2)
- [v1.15.0-beta.1](#v1150-beta1)
  - [Downloads for v1.15.0-beta.1](#downloads-for-v1150-beta1)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.15.0-alpha.3](#changelog-since-v1150-alpha3)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-3)
- [v1.15.0-alpha.3](#v1150-alpha3)
  - [Downloads for v1.15.0-alpha.3](#downloads-for-v1150-alpha3)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.15.0-alpha.2](#changelog-since-v1150-alpha2)
    - [Other notable changes](#other-notable-changes-4)
- [v1.15.0-alpha.2](#v1150-alpha2)
  - [Downloads for v1.15.0-alpha.2](#downloads-for-v1150-alpha2)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.15.0-alpha.1](#changelog-since-v1150-alpha1)
    - [Other notable changes](#other-notable-changes-5)
- [v1.15.0-alpha.1](#v1150-alpha1)
  - [Downloads for v1.15.0-alpha.1](#downloads-for-v1150-alpha1)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.14.0](#changelog-since-v1140)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-6)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.15.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.15.0


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes.tar.gz) | `cb03adc8bee094b93652a19cb77ca4b7b0b2ec201cf9c09958128eb93b4c717514fb423ef60c8fdd2af98ea532ef8d9f3155a684a3a7dc2a20cba0f8d7821a79`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-src.tar.gz) | `a682c88539b46741f6f3b2fa27017d52e88149e0cf0fe49c5a84ff30018cfa18922772a49828091364910570cf5f6b4089a128b400f48a278d6ac7b18ef84635`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-darwin-386.tar.gz) | `bb14d564f5c2e4da964f6dcaf4026ac7371b35ecf5d651d226fb7cc0c3f194c1540860b7cf5ba35c1ebbdf683cefd8011bd35d345cf6707a1584f6a20230db96`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-darwin-amd64.tar.gz) | `8c218437588d960f6782576038bc63af5623e66291d37029653d4bdbba5e19b3e8a8a0225d250d76270ab243aa97fa15ccaf7cae84fefc05a129c05687854c0e`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-linux-386.tar.gz) | `6a17e7215d0eb9ca18d4b55ee179a13f1f111ac995aad12bf2613b9dbee1a6a3a25e8856fdb902955c47d076131c03fc074fad5ad490bc09d6dc53638a358582`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-linux-amd64.tar.gz) | `0906a8f7de1e5c5efd124385fdee376893733f343d3e8113e4f0f02dfae6a1f5b12dca3e2384700ea75ec39985b7c91832a3aeb8fa4f13ffd736c56a86f23594`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-linux-arm.tar.gz) | `1d3418665b4998d6fff1c137424eb60302129098321052d7c5cee5a0e2a5624c9eb2fd19c94b50a598ddf039664e5795e97ba99ae66aabc0ee79f48d23c30a65`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-linux-arm64.tar.gz) | `986d6bec386b3bb427e49cd7e41390c7dc5361da4f2f7fc2a823507f83579ea1402de566651519bf83267bf2a92dc4bc40b72bb587cdc78aa8b9027f629e8436`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-linux-ppc64le.tar.gz) | `81315af33bc21f9f8808b125e1f4c7a1f797c70f01098fe1fe8dba73d05d89074209c70e39b0fd8b42a5e43f2392ece3a070b9e83be5c4978e82ddad3ce09452`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-linux-s390x.tar.gz) | `485978a24ba97a2a2cac162a6984d4b5c32dbe95882cf18d2fd2bf74477f689abc6e9d6b10ec016cd5957b0b71237cd9c01d850ff1c7bd07a561d0c2d6598ee7`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-windows-386.tar.gz) | `9a1b5d0f6fbfc85269e9bd7e08be95eeb9a11f43ea38325b8a736e768f3e855e681eef17508ca0c9da6ab9cbed2875dba5beffc91d1418316b7ca3efa192c768`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-client-windows-amd64.tar.gz) | `f2f0221c7d364e3e71b2d9747628298422441c43b731d58c14d7a0ed292e5f12011780c482bdb8f613ddc966868fd422e4ca01e4b522601d74cdee49c59a1766`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-server-linux-amd64.tar.gz) | `fee0200887c7616e3706394b0540b471ad24d57bb587a3a7154adfcd212c7a2521605839b0e95c23d61c86f6c21ef85c63f0d0a0504ba378b4c28cd110771c31`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-server-linux-arm.tar.gz) | `2d329ec0e231dbd4ec750317fc45fb8a966b9a81b45f1af0dde3ca0d1ae66a5ade39c6b64f6a1a492b55f6fca04057113ec05de61cb0f11caeee2fb7639e7775`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-server-linux-arm64.tar.gz) | `0fb64d934d82c17eee15e1f97fc5eeeb4af6e042c30abe41a4d245cde1d9d81ee4dad7e0b0b3f707a509c84fce42289edd2b18c4e364e99a1c396f666f114dcf`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-server-linux-ppc64le.tar.gz) | `5cac4b5951692921389db280ec587037eb3bb7ec4ccf08599ecee2fa39c2a5980df9aba80fc276c78b203222ad297671c45a9fed690ad7bcd774854bd918012b`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-server-linux-s390x.tar.gz) | `39a33f0bb0e06b34779d741e6758b6f7d385e0b933ab799b233e3d4e317f76b5d1e1a6d196f3c7a30a24916ddb7c3c95c8b1c5f6683bce709b2054e1fc018b77`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-node-linux-amd64.tar.gz) | `73abf50e44319763be3124891a1db36d7f7b38124854a1f223ebd91dce8e848a825716c48c9915596447b16388e5b752ca90d4b9977348221adb8a7e3d2242fd`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-node-linux-arm.tar.gz) | `b7ddb82efa39ba5fce5b4124d83279357397a1eb60be24aa19ccbd8263e5e6146bfaff52d7f5167b14d6d9b919c4dcd34319009701e9461d820dc40b015890a0`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-node-linux-arm64.tar.gz) | `458f20f7e9ca2ebddef8738de6a2baa8b8d958b22a935e4d7ac099b07bed91fe44126342faa8942cf23214855b20d2a52fcb95b1fbb8ae6fe33b601ecdbf0c39`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-node-linux-ppc64le.tar.gz) | `d4d5bfe9b9d56495b00322f62aed0f76029d774bff5004d68e85a0db4fb3b4ceb3cef79a4f56e322b8bb47b4adbf3966cff0b5a24f9678da02122f2024ecc6cd`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-node-linux-s390x.tar.gz) | `b967034c8db871a7f503407d5a096fcd6811771c9a294747b0a028659af582fbc47061c388adfabf1c84cd73b33f7bbf5377eb5b31ab51832ea0b5625a82e799`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0/kubernetes-node-windows-amd64.tar.gz) | `dd021d8f2a3d9ddff6e88bce678c28cc0f38165a5d7a388df952d900dcfd1dcaf45c7e75c6387d061014cba15aaf7453905a46e84ddd8b3f8eff2539d50fce9b`

# Kubernetes v1.15 Release Notes

## 1.15 What’s New

A complete changelog for the release notes is now hosted in a customizable format at [https://relnotes.k8s.io/](https://relnotes.k8s.io/?releaseVersions=1.15.0). Check it out and please give us your feedback!

Kubernetes 1.15 consists of **25 enhancements**: 2 moving to stable, 13 in beta, and 10 in alpha. The main themes of this release are:

#### Continuous Improvement
- Project sustainability is not just about features. Many SIGs have been working on  improving test coverage, ensuring the basics stay reliable, and stability of the core feature set and working on maturing existing features and cleaning up the backlog.

#### Extensibility

- The community has been asking for continuing support of extensibility, so this cycle features more work around CRDs and API Machinery. Most of the enhancements in this cycle were from SIG API Machinery and related areas.

### Extensibility around core Kubernetes APIs

#### CustomResourceDefinitions Pruning
To enforce both data consistency and security, Kubernetes performs pruning, or the automatic removal of unknown fields in objects sent to a Kubernetes API. An "unknown" field is one that is not specified in the OpenAPI validation schema. This behavior is already in place for native resources and ensures only data structures specified by the CRD developer are persisted to etcd. It will be available as a beta feature in Kubernetes 1.15.

Pruning is activated by setting `spec.preserveUnknownFields: false` in the CustomResourceDefinition. A future apiextensions.k8s.io/v1 variant of CRDs will enforce pruning.

Pruning requires that CRD developer provides complete, structural validation schemas, either at the top-level or for all versions of the CRD.

#### CustomResourceDefinition Defaulting

CustomResourceDefinitions also have new support for defaulting, with defaults specified using the `default` keyword in the OpenAPI validation schema. Defaults are set for unspecified fields in an object sent to the API, and when reading from etcd.

Defaulting will be available as alpha in Kubernetes 1.15 and requires structural schemas.

#### CustomResourceDefinition OpenAPI Publishing

OpenAPI specs for native types have long been served at /openapi/v2, and they are consumed by a number of components, notably kubectl client-side validation, kubectl explain and OpenAPI based client generators.

With Kubernetes 1.15 as beta, OpenAPI schemas are also published for CRDs, as long as their schemas are structural.

These changes are reflected in the following Kubernetes enhancements:
([#383](https://github.com/kubernetes/enhancements/issues/383)), ([#575](https://github.com/kubernetes/enhancements/issues/575) ), ([#492](https://github.com/kubernetes/enhancements/issues/492) ), ([#598](https://github.com/kubernetes/enhancements/issues/598) ), ([#692](https://github.com/kubernetes/enhancements/issues/692) ), ([#95](https://github.com/kubernetes/enhancements/issues/95) ), ([#995](https://github.com/kubernetes/enhancements/issues/995) ), ([#956](https://github.com/kubernetes/enhancements/issues/956) )

### Cluster Lifecycle Stability and Usability Improvements
Work on making Kubernetes installation, upgrade and configuration even more robust has been a major focus for this cycle for SIG Cluster Lifecycle (see the May 6, 2019 [Community Update](https://docs.google.com/presentation/d/1QUOsQxfEfHlMq4lPjlK2ewQHsr9peEKymDw5_XwZm8Q/edit?usp=sharing)). Bug fixes across bare metal tooling and production-ready user stories, such as the high availability use cases have been given priority for 1.15.

kubeadm, the cluster lifecycle building block, continues to receive features and stability work required for bootstrapping production clusters efficiently. kubeadm has promoted high availability (HA) capability to beta, allowing users to use the familiar `kubeadm init` and `kubeadm join` commands to [configure and deploy an HA control plane](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/high-availability/). An entire new test suite has been created specifically for ensuring these features will stay stable over time.

Certificate management has become more robust in 1.15, with kubeadm now seamlessly rotating all your certificates (on upgrades) before they expire. Check the [kubeadm documentation](https://github.com/kubernetes/website/blob/dev-1.15/content/en/docs/reference/setup-tools/kubeadm/kubeadm-alpha.md) for information on how to manage your certificates.

The kubeadm configuration file API is moving from v1beta1 to v1beta2 in 1.15.

These changes are reflected in the following Kubernetes enhancements:
([#357](https://github.com/kubernetes/enhancements/issues/357) ), ([#970](https://github.com/kubernetes/enhancements/issues/970) )

### Continued improvement of CSI
In Kubernetes v1.15, SIG Storage continued work to [enable migration of in-tree volume plugins](https://github.com/kubernetes/enhancements/issues/625) to the Container Storage Interface (CSI). SIG Storage worked on bringing CSI to feature parity with in-tree functionality, including functionality like resizing, inline volumes, and more. SIG Storage introduces new alpha functionality in CSI that doesn't exist in the Kubernetes Storage subsystem yet, like volume cloning.

Volume cloning enables users to specify another PVC as a "DataSource" when provisioning a new volume. If the underlying storage system supports this functionality and implements the "CLONE_VOLUME" capability in its CSI driver, then the new volume becomes a clone of the source volume.

These changes are reflected in the following Kubernetes enhancements:
([#625](https://github.com/kubernetes/enhancements/issues/625))

#### Additional Notable Feature Updates
- Support for go modules in Kubernetes Core.
- Continued preparation for cloud provider extraction and code organization.  The cloud provider code has been moved to kubernetes/legacy-cloud-providers for easier removal later and external consumption.
- Kubectl [get and describe](https://github.com/kubernetes/enhancements/issues/515) now works with extensions
- Nodes now support [third party monitoring plugins](https://github.com/kubernetes/enhancements/issues/606).
- A new [Scheduling Framework](https://github.com/kubernetes/enhancements/issues/624) for schedule plugins is now Alpha.
- ExecutionHook API [designed to trigger hook commands](https://github.com/kubernetes/enhancements/issues/962) in containers is now Alpha.
- Continued deprecation of extensions/v1beta1, apps/v1beta1, and apps/v1beta2 APIs; these extensions will be retired in 1.16!

Check the [release notes website](https://relnotes.k8s.io/?releaseVersions=1.15.0) for the complete changelog of notable features and fixes.




## Known Issues

- Concurrently joining control-plane nodes does not work as expected in kubeadm 1.15.0. The feature was planned for release in 1.15.0, but a fix may  come in a follow up patch release.

- Using `--log-file` is known to be problematic in 1.15. This presents as things being logged multiple times to the same file. The behaviour and details of this issue, as well as some preliminary attempts at fixing it are documented [here](https://github.com/kubernetes/kubernetes/issues/78734#issuecomment-501372131)

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

#### API Machinery

- `k8s.io/kubernetes` and published components (such as `k8s.io/client-go` and `k8s.io/api`) now contain go module files including dependency version information. See [go-modules](http://git.k8s.io/client-go/INSTALL.md#go-modules) for details on consuming `k8s.io/client-go` using go modules. ([#74877](https://github.com/kubernetes/kubernetes/pull/74877), [@liggitt](https://github.com/liggitt))

#### Apps

- Hyperkube short aliases have been removed from source code, because hyperkube docker image currently creates these aliases. ([#76953](https://github.com/kubernetes/kubernetes/pull/76953), [@Rand01ph](https://github.com/Rand01ph))

#### Auth

- The Rancher credential provider has now been removed. This only affects you if you are using the downstream Rancher distro. ([#77099](https://github.com/kubernetes/kubernetes/pull/77099), [@dims](https://github.com/dims))


#### AWS

- The `system:aws-cloud-provider` cluster role, deprecated in v1.13, is no longer auto-created. Deployments using the AWS cloud provider should grant required permissions to the `aws-cloud-provider` service account in the `kube-system` namespace as part of deployment.  ([#66635](https://github.com/kubernetes/kubernetes/pull/66635), [@wgliang](https://github.com/wgliang))

#### Azure

- Kubelet can now run without identity on Azure. A sample cloud provider configuration is:  `{"vmType": "vmss", "useInstanceMetadata": true, "subscriptionId": "<subscriptionId>"}` ([#77906](https://github.com/kubernetes/kubernetes/pull/77906), [@feiskyer](https://github.com/feiskyer))
- Multiple Kubernetes clusters can now share the same resource group
    - When upgrading from previous releases, issues will arise with public IPs if multiple clusters share the same resource group. To solve these problems, make the following changes to the cluster:
Recreate the relevant LoadBalancer services, or add a new tag 'kubernetes-cluster-name: <cluster-name>' manually for existing public IPs.
Configure each cluster with a different cluster name using `kube-controller-manager --cluster-name=<cluster-name>` ([#77630](https://github.com/kubernetes/kubernetes/pull/77630), [@feiskyer](https://github.com/feiskyer))
- The cloud config for Azure cloud provider can now be initialized from Kubernetes secret azure-cloud-provider in kube-system namespace
    - the secret is a serialized version of `azure.json` file with key cloud-config. And the secret name is azure-cloud-provider.
    - A new option cloudConfigType has been added to the cloud-config file. Supported values are: `file`, `secret` and `merge` (`merge` is the default value).
    - To allow Azure cloud provider to read secrets, the [RBAC rules](https://github.com/kubernetes/kubernetes/pull/78242) should be configured.

#### CLI

- `kubectl scale job`, deprecated since 1.10, has been removed. ([#78445](https://github.com/kubernetes/kubernetes/pull/78445), [@soltysh](https://github.com/soltysh))
- The deprecated `--pod`/`-p` flag for `kubectl exec` has been removed. The flag has been marked as deprecated since k8s version v1.12. ([#76713](https://github.com/kubernetes/kubernetes/pull/76713), [@prksu](https://github.com/prksu))


#### Lifecycle

- Support for deprecated old kubeadm v1alpha3 config has been totally removed. ([#75179](https://github.com/kubernetes/kubernetes/pull/75179), [@rosti](https://github.com/rosti))
- kube-up.sh no longer supports "centos" and "local" providers. ([#76711](https://github.com/kubernetes/kubernetes/pull/76711), [@dims](https://github.com/dims))

#### Network

- The deprecated flag `--conntrack-max` has been removed from kube-proxy. Users of this flag should switch to `--conntrack-min` and `--conntrack-max-per-core` instead. ([#78399](https://github.com/kubernetes/kubernetes/pull/78399), [@rikatz](https://github.com/rikatz))
- The deprecated kube-proxy flag `--cleanup-iptables` has been removed. ([#78344](https://github.com/kubernetes/kubernetes/pull/78344), [@aramase](https://github.com/aramase))

#### Node

- The deprecated kubelet security controls `AllowPrivileged`, `HostNetworkSources`, `HostPIDSources`, and `HostIPCSources` have been removed. Enforcement of these restrictions should be done through admission control (such as `PodSecurityPolicy`) instead. ([#77820](https://github.com/kubernetes/kubernetes/pull/77820), [@dims](https://github.com/dims))
- The deprecated Kubelet flag `--allow-privileged` has been removed. Remove any use of the flag from your kubelet scripts or manifests. ([#77820](https://github.com/kubernetes/kubernetes/pull/77820), [@dims](https://github.com/dims))
- The kubelet now only collects cgroups metrics for the node, container runtime, kubelet, pods, and containers. ([#72787](https://github.com/kubernetes/kubernetes/pull/72787), [@dashpole](https://github.com/dashpole))

#### Storage

- The `Node.Status.Volumes.Attached.DevicePath` field is now unset for CSI volumes. You must update any external controllers that depend on this field. ([#75799](https://github.com/kubernetes/kubernetes/pull/75799), [@msau42](https://github.com/msau42))
- CSI alpha CRDs have been removed ([#75747](https://github.com/kubernetes/kubernetes/pull/75747), [@msau42](https://github.com/msau42))
- The `StorageObjectInUseProtection` admission plugin is enabled by default, so  the default enabled admission plugins are now `NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota,StorageObjectInUseProtection`. Please note that if you previously had not set the `--admission-control` flag, your cluster behavior may change (to be more standard). ([#74610](https://github.com/kubernetes/kubernetes/pull/74610), [@oomichi](https://github.com/oomichi))



## Deprecations and Removals

- kubectl
  - `kubectl convert`, deprecated since v1.14, will be removed in v1.17.
  - The `--export` flag for the `kubectl get` command, deprecated since v1.14, will be removed in v1.18.
  - The `--pod`/`-p` flag for `kubectl exec`, deprecated since 1.12, has been removed.
  - `kubectl scale job`, deprecated since 1.10, has been removed. ([#78445](https://github.com/kubernetes/kubernetes/pull/78445), [@soltysh](https://github.com/soltysh))


- kubelet
  - The `beta.kubernetes.io/os` and `beta.kubernetes.io/arch` labels, deprecated since v1.14, are targeted for removal in v1.18.
  - The `--containerized` flag, deprecated since v1.14, will be removed in a future release.
  - cAdvisor json endpoints have been deprecated. ([#78504](https://github.com/kubernetes/kubernetes/pull/78504), [@dashpole](https://github.com/dashpole))

- kube-apiserver
  - The `--enable-logs-handler` flag and log-serving functionality is deprecated, and scheduled to be removed in v1.19. ([#77611](https://github.com/kubernetes/kubernetes/pull/77611), [@rohitsardesai83](https://github.com/rohitsardesai83))

- kube-proxy
  - The deprecated `--cleanup-iptables` has been removed,. ([#78344](https://github.com/kubernetes/kubernetes/pull/78344), [@aramase](https://github.com/aramase))


- API
  - Ingress resources will no longer be served from `extensions/v1beta1` in v1.19. Migrate use to the `networking.k8s.io/v1beta1` API, available since v1.14. Existing persisted data can be retrieved via the `networking.k8s.io/v1beta1` API.
  - NetworkPolicy resources will no longer be served from `extensions/v1beta1` in v1.16. Migrate use to the `networking.k8s.io/v1` API, available since v1.8. Existing persisted data can be retrieved via the `networking.k8s.io/v1` API.
  - PodSecurityPolicy resources will no longer be served from `extensions/v1beta1` in v1.16. Migrate to the `policy/v1beta1` API, available since v1.10. Existing persisted data can be retrieved via the `policy/v1beta1` API.
  - DaemonSet, Deployment, and ReplicaSet resources will no longer be served from `extensions/v1beta1`, `apps/v1beta1`, or `apps/v1beta2` in v1.16. Migrate to the `apps/v1` API, available since v1.9. Existing persisted data can be retrieved via the `apps/v1` API.
  - PriorityClass resources will no longer be served from `scheduling.k8s.io/v1beta1` and `scheduling.k8s.io/v1alpha1` in v1.17. Migrate use to the `scheduling.k8s.io/v1` API, available since v1.14. Existing persisted data can be retrieved via the `scheduling.k8s.io/v1` API.
  - The `export` query parameter for list API calls, deprecated since v1.14, will be removed in v1.18.
  - The `series.state` field in the events.k8s.io/v1beta1 Event API is deprecated and will be removed in v1.18 ([#75987](https://github.com/kubernetes/kubernetes/pull/75987), [@yastij](https://github.com/yastij))

- kubeadm
  - The `kubeadm upgrade node config` and `kubeadm upgrade node experimental-control-plane` commands are deprecated in favor of `kubeadm upgrade node`, and will be removed in a future release. ([#78408](https://github.com/kubernetes/kubernetes/pull/78408), [@fabriziopandini](https://github.com/fabriziopandini))
  - The flag `--experimental-control-plane` is now deprecated in favor of  `--control-plane`. The flag `--experimental-upload-certs` is now deprecated in favor of `--upload-certs` ([#78452](https://github.com/kubernetes/kubernetes/pull/78452), [@fabriziopandini](https://github.com/fabriziopandini))
  - `kubeadm config upload` has been deprecated, as its replacement is now graduated. Please use `kubeadm init phase upload-config` instead. ([#77946](https://github.com/kubernetes/kubernetes/pull/77946), [@Klaven](https://github.com/Klaven))

- The following features are now GA, and the associated feature gates are deprecated and will be removed in v1.17:
  - `GCERegionalPersistentDisk`

## Metrics Changes

### Added metrics

- The metric `kube_proxy_sync_proxy_rules_last_timestamp_seconds` is now available, indicating the last time that kube-proxy successfully applied proxying rules. ([#74027](https://github.com/kubernetes/kubernetes/pull/74027), [@squeed](https://github.com/squeed))
- `process_start_time_seconds` has been added to kubelet’s '/metrics/probes' endpoint ([#77975](https://github.com/kubernetes/kubernetes/pull/77975), [@logicalhan](https://github.com/logicalhan))
- Scheduler: added metrics to record the number of pending pods in different queues ([#75501](https://github.com/kubernetes/kubernetes/pull/75501), [@Huang-Wei](https://github.com/Huang-Wei))
- Exposed CSI volume stats via kubelet volume metrics ([#76188](https://github.com/kubernetes/kubernetes/pull/76188), [@humblec](https://github.com/humblec))
- Added a new `storage_operation_status_count` metric for kube-controller-manager and kubelet to count success and error statues. ([#75750](https://github.com/kubernetes/kubernetes/pull/75750), [@msau42](https://github.com/msau42))

### Deprecated/changed metrics

- kubelet probe metrics are now of the counter type rather than the gauge type, and the `prober_probe_result` has been replaced by `prober_probe_total`. ([#76074](https://github.com/kubernetes/kubernetes/pull/76074), [@danielqsj](https://github.com/danielqsj))
- The `transformer_failures_total` metric is deprecated in favor of `transformation_operation_total`. The old metric will continue to be populated but will be removed in a future release. ([#70715](https://github.com/kubernetes/kubernetes/pull/70715), [@immutableT](https://github.com/immutableT))
- Introducing new semantic for metric `volume_operation_total_seconds` to be the end to end latency of volume provisioning/deletion. Existing metric "storage_operation_duration_seconds" will remain untouched, however it is exposed to the following potential issues:
  1. For volumes provisioned/deleted via external provisioner/deleter, `storage_operation_duration_seconds` will NOT wait for the external operation to be done before reporting latency metric (effectively close to 0). This will be fixed by using `volume_operation_total_seconds` instead
  2. if there's a transient error happened during "provisioning/deletion", i.e., a volume is still in-use while a deleteVolume has been called, original `storage_operation_duration_seconds` will NOT wait until a volume has been finally deleted before reporting an inaccurate latency metric. The newly implemented metric `volume_operation_total_seconds`, however, waits until a provisioning/deletion operation has been fully executed.

    Potential impacts:
    If an SLO/alert has been defined based on `volume_operation_total_seconds`, it might get violated because of the more accurate metric might be significantly larger than previously reported. The metric is defined to be a histogram and the new semantic could change the distribution. ([#78061](https://github.com/kubernetes/kubernetes/pull/78061), [@yuxiangqian](https://github.com/yuxiangqian))

- Implement the scheduling framework with `Reserve`, `Prebind`, `Permit`, `Post-bind`, `Queue sort` and `Unreserve` extension points.
([#77567](https://github.com/kubernetes/kubernetes/pull/77567), [@wgliang](https://github.com/wgliang))
([#77559](https://github.com/kubernetes/kubernetes/pull/77559), [@ahg-g](https://github.com/ahg-g))
([#77529](https://github.com/kubernetes/kubernetes/pull/77529), [@draveness](https://github.com/draveness))
([#77598](https://github.com/kubernetes/kubernetes/pull/77598), [@danielqsj](https://github.com/danielqsj))
([#77501](https://github.com/kubernetes/kubernetes/pull/77501), [@JieJhih](https://github.com/JieJhih))
([#77457](https://github.com/kubernetes/kubernetes/pull/77457), [@danielqsj](https://github.com/danielqsj))
- Replaced *_admission_latencies_milliseconds_summary and *_admission_latencies_milliseconds metrics because they were reporting seconds rather than milliseconds. They were also subject to multiple naming guideline violations (units should be in base units and "duration" is the best practice labelling to measure the time a request takes). Please convert to use *_admission_duration_seconds and *_admission_duration_seconds_summary, as these now report the unit as described, and follow the instrumentation best practices. ([#75279](https://github.com/kubernetes/kubernetes/pull/75279), [@danielqsj](https://github.com/danielqsj))
- Fixed admission metrics histogram bucket sizes to cover 25ms to ~2.5 seconds. ([#78608](https://github.com/kubernetes/kubernetes/pull/78608), [@jpbetz](https://github.com/jpbetz))
- Fixed incorrect prometheus azure metrics. ([#77722](https://github.com/kubernetes/kubernetes/pull/77722), [@andyzhangx](https://github.com/andyzhangx))
- `kubectl scale job`, deprecated since 1.10, has been removed. ([#78445](https://github.com/kubernetes/kubernetes/pull/78445), [@soltysh](https://github.com/soltysh))



## Notable Features

### Stable

- You can now create a non-preempting Pod priority. If set on a class, the pod will continue to be prioritized above queued pods of a lesser class, but will not preempt running pods. ([#74614](https://github.com/kubernetes/kubernetes/pull/74614), [@denkensk](https://github.com/denkensk))

- Third party device monitoring is now enabled by default (KubeletPodResources). ([#77274](https://github.com/kubernetes/kubernetes/pull/77274), [@RenaudWasTaken](https://github.com/RenaudWasTaken))
- The kube-apiserver’s `watch` can now be enabled for events using the `--watch-cache-sizes` flag. ([#74321](https://github.com/kubernetes/kubernetes/pull/74321), [@yastij](https://github.com/yastij))

### Beta

- Admission webhooks can now register for a single version of a resource (for example, `apps/v1 deployments`) and be called when any other version of that resource is modified (for example `extensions/v1beta1 deployments`). This allows new versions of a resource to be handled by admission webhooks without needing to update every webhook to understand the new version. See the API documentation for the `matchPolicy: Equivalent` option in MutatingWebhookConfiguration and ValidatingWebhookConfiguration types. ([#78135](https://github.com/kubernetes/kubernetes/pull/78135), [@liggitt](https://github.com/liggitt))
- The CustomResourcePublishOpenAPI feature is now beta and enabled by default. CustomResourceDefinitions with [structural schemas](https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190425-structural-openapi.md) now publish schemas in the OpenAPI document served at `/openapi/v2`. CustomResourceDefinitions with non-structural schemas have a `NonStructuralSchema` condition added with details about what needs to be corrected in the validation schema. ([#77825](https://github.com/kubernetes/kubernetes/pull/77825), [@roycaihw](https://github.com/roycaihw))
- Online volume expansion (ExpandInUsePersistentVolumes) is now a beta feature. As such, it is enabled by default. ([#77755](https://github.com/kubernetes/kubernetes/pull/77755), [@gnufied](https://github.com/gnufied))
- The `SupportNodePidsLimit` feature is now beta, and enabled by default.  It is no longer necessary to set the feature gate `SupportNodePidsLimit=true`. ([#76221](https://github.com/kubernetes/kubernetes/pull/76221), [@RobertKrawitz](https://github.com/RobertKrawitz))
- kubeadm now includes the ability to specify certificate encryption and decryption keys for the upload and download certificate phases as part of the new v1beta2 kubeadm config format. ([#77012](https://github.com/kubernetes/kubernetes/pull/77012), [@rosti](https://github.com/rosti))
- You can now use kubeadm's `InitConfiguration` and `JoinConfiguration` to define which preflight errors will be ignored. ([#75499](https://github.com/kubernetes/kubernetes/pull/75499), [@marccarre](https://github.com/marccarre))
- CustomResourcesDefinition conversion via Web Hooks is promoted to beta. Note that you must set `spec.preserveUnknownFields` to `false`. ([#78426](https://github.com/kubernetes/kubernetes/pull/78426), [@sttts](https://github.com/sttts))
- Group Managed Service Account support has moved to a new API for beta. Special annotations for Windows GMSA support have been deprecated.
([#75459](https://github.com/kubernetes/kubernetes/pull/75459), [@wk8](https://github.com/wk8))
- The `storageVersionHash` feature is now beta. `StorageVersionHash` is a field in the discovery document of each resource. It enables clients to detect whether the storage version of that resource has changed. Its value must be treated as opaque by clients. Only equality comparison on the value is valid. ([#78325](https://github.com/kubernetes/kubernetes/pull/78325), [@caesarxuchao](https://github.com/caesarxuchao))
- Ingress objects are now persisted in etcd using the `networking.k8s.io/v1beta1` version ([#77139](https://github.com/kubernetes/kubernetes/pull/77139), [@cmluciano](https://github.com/cmluciano))
- NodeLocal DNSCache graduating to beta. ([#77887](https://github.com/kubernetes/kubernetes/pull/77887), [@prameshj](https://github.com/prameshj))

### Alpha

- kubelet now allows the use of XFS quotas (on XFS and suitably configured ext4fs filesystems) to monitor storage consumption for ephemeral storage.  This method of monitoring consumption, which is currently available only for `emptyDir` volumes, is faster and more accurate than the old method of walking the filesystem tree. Note that it does not enforce limits, it only monitors consumption. To utilize this functionality, set the feature gate `LocalStorageCapacityIsolationFSQuotaMonitoring=true`. For ext4fs filesystems, create the filesystem with `mkfs.ext4 -O project <block_device>` and run `tune2fs -Q prjquota `block device`; XFS filesystems need no additional preparation. The filesystem must be mounted with option `project` in `/etc/fstab`. If the primary partition is the root filesystem, add `rootflags=pquota` to the GRUB config file. ([#66928](https://github.com/kubernetes/kubernetes/pull/66928), [@RobertKrawitz](https://github.com/RobertKrawitz))
- Finalizer Protection for Service LoadBalancers (ServiceLoadBalancerFinalizer) has been added as an Alpha feature, which is disabled by default. This feature ensures the Service resource is not fully deleted until the correlating load balancer resources are deleted. ([#78262](https://github.com/kubernetes/kubernetes/pull/78262), [@MrHohn](https://github.com/MrHohn))
- Inline CSI ephemeral volumes can now be controlled with PodSecurityPolicy when the CSIInlineVolume alpha feature is enabled. ([#76915](https://github.com/kubernetes/kubernetes/pull/76915), [@vladimirvivien](https://github.com/vladimirvivien))
- Kubernetes now includes an alpha field, `AllowWatchBookmarks`, in ListOptions for requesting the watching of bookmarks from apiserver. The implementation in apiserver is hidden behind the feature gate `WatchBookmark`. ([#74074](https://github.com/kubernetes/kubernetes/pull/74074), [@wojtek-t](https://github.com/wojtek-t))

### Staging Repositories

- The CRI API is now available in the `k8s.io/cri-api` staging repository. ([#75531](https://github.com/kubernetes/kubernetes/pull/75531), [@dims](https://github.com/dims))
- Support for the Azure File plugin has been added to `csi-translation-lib` (CSIMigrationAzureFile). ([#78356](https://github.com/kubernetes/kubernetes/pull/78356), [@andyzhangx](https://github.com/andyzhangx))
- Added support for Azure Disk plugin to csi-translation-lib (CSIMigrationAzureDisk) ([#78330](https://github.com/kubernetes/kubernetes/pull/78330), [@andyzhangx](https://github.com/andyzhangx))

### CLI Improvements

- Added `kubeadm upgrade node`. This command can be used to upgrade both secondary control-plane nodes and worker nodes. The `kubeadm upgrade node config` and `kubeadm upgrade node experimental-control-plane` commands are now deprecated. ([#78408](https://github.com/kubernetes/kubernetes/pull/78408), [@fabriziopandini](https://github.com/fabriziopandini))
- The `kubectl top` command now includes a `--sort-by` option to sort by `memory` or `cpu`. ([#75920](https://github.com/kubernetes/kubernetes/pull/75920), [@artmello](https://github.com/artmello))
- `kubectl rollout restart` now works for DaemonSets and StatefulSets. ([#77423](https://github.com/kubernetes/kubernetes/pull/77423), [@apelisse](https://github.com/apelisse))
- `kubectl get --watch=true` now prints custom resource definitions with custom print columns. ([#76161](https://github.com/kubernetes/kubernetes/pull/76161), [@liggitt](https://github.com/liggitt))
- Added `kubeadm alpha certs certificate-key` command to generate secure random key to use on `kubeadm init --experimental-upload-certs` ([#77848](https://github.com/kubernetes/kubernetes/pull/77848), [@yagonobre](https://github.com/yagonobre))
- Kubernetes now supports printing the `volumeMode` using `kubectl get pv/pvc -o wide` ([#76646](https://github.com/kubernetes/kubernetes/pull/76646), [@cwdsuzhou](https://github.com/cwdsuzhou))
- Created a new `kubectl rollout restart` command that does a rolling restart of a deployment. ([#76062](https://github.com/kubernetes/kubernetes/pull/76062), [@apelisse](https://github.com/apelisse))
- `kubectl exec` now allows using the resource name to select a matching pod and `--pod-running-timeout` flag to wait till at least one pod is running. ([#73664](https://github.com/kubernetes/kubernetes/pull/73664), [@prksu](https://github.com/prksu))
- `kubeadm alpha certs renew` and `kubeadm upgrade` now supports renewal of certificates embedded in KubeConfig files managed by kubeadm; this does not apply to certificates signed by external CAs.  ([#77180](https://github.com/kubernetes/kubernetes/pull/77180), [@fabriziopandini](https://github.com/fabriziopandini))
- Kubeadm: a new command `kubeadm alpha certs check-expiration` was created in order to help users in managing expiration for local PKI certificates ([#77863](https://github.com/kubernetes/kubernetes/pull/77863), [@fabriziopandini](https://github.com/fabriziopandini))

### Misc

- Service account controller clients to now use the TokenRequest API, and tokens are periodically rotated. ([#72179](https://github.com/kubernetes/kubernetes/pull/72179), [@WanLinghao](https://github.com/WanLinghao))
- Added `ListPager.EachListItem` utility function to client-go to enable incremental processing of chunked list responses ([#75849](https://github.com/kubernetes/kubernetes/pull/75849), [@jpbetz](https://github.com/jpbetz))
- Object count quota is now supported for namespaced custom resources using the `count/<resource>.<group>` syntax. ([#72384](https://github.com/kubernetes/kubernetes/pull/72384), [@zhouhaibing089](https://github.com/zhouhaibing089))
- Added completed job status in Cron Job event. ([#75712](https://github.com/kubernetes/kubernetes/pull/75712), [@danielqsj](https://github.com/danielqsj))
- Pod disruption budgets can now be updated and patched. ([#69867](https://github.com/kubernetes/kubernetes/pull/69867), [@davidmccormick](https://github.com/davidmccormick))
- Add CRD spec.preserveUnknownFields boolean, defaulting to true in v1beta1 and to false in v1 CRDs. If false, fields not specified in the validation schema will be removed when sent to the API server or when read from etcd. ([#77333](https://github.com/kubernetes/kubernetes/pull/77333), [@sttts](https://github.com/sttts))
- Added RuntimeClass restrictions and defaulting to PodSecurityPolicy. ([#73795](https://github.com/kubernetes/kubernetes/pull/73795), [@tallclair](https://github.com/tallclair))
- Kubelet plugin registration now has retry and exponential backoff logic for when registration of plugins (such as CSI or device plugin) fail. ([#73891](https://github.com/kubernetes/kubernetes/pull/73891), [@taragu](https://github.com/taragu))
- proxy/transport now supports Content-Encoding: deflate ([#76551](https://github.com/kubernetes/kubernetes/pull/76551), [@JieJhih](https://github.com/JieJhih))
- Admission webhooks are now properly called for `scale` and `deployments/rollback` subresources. ([#76849](https://github.com/kubernetes/kubernetes/pull/76849), [@liggitt](https://github.com/liggitt))

## API Changes

- CRDs get support for x-kubernetes-int-or-string to allow faithful representation of IntOrString types in CustomResources.([#78815](https://github.com/kubernetes/kubernetes/pull/78815), [@sttts](https://github.com/sttts))
- Introduced the [`v1beta2`](https://docs.google.com/document/d/1XnP67oO1i9VcDIpw42IzptnJsc5OQM-HTf8cVcjCR2w/edit) config format to kubeadm. ([#76710](https://github.com/kubernetes/kubernetes/pull/76710), [@rosti](https://github.com/rosti))
- Resource list requests for `PartialObjectMetadata` now correctly return list metadata like the resourceVersion and the continue token. ([#75971](https://github.com/kubernetes/kubernetes/pull/75971), [@smarterclayton](https://github.com/smarterclayton))
- Added a condition `NonStructuralSchema` to `CustomResourceDefinition` listing Structural Schema violations as defined in the [KEP](https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190425-structural-openapi.md). CRD authors should update their validation schemas to be structural in order to participate in future CRD features. ([#77207](https://github.com/kubernetes/kubernetes/pull/77207), [@sttts](https://github.com/sttts))
- Promoted meta.k8s.io/v1beta1 Table and PartialObjectMetadata to v1. ([#77136](https://github.com/kubernetes/kubernetes/pull/77136), [@smarterclayton](https://github.com/smarterclayton))
- Introduced the flag `--ipvs-strict-arp` to configure stricter ARP sysctls, defaulting to false to preserve existing behaviors. This was enabled by default in 1.13.0, which impacted a few CNI plugins. ([#75295](https://github.com/kubernetes/kubernetes/pull/75295), [@lbernail](https://github.com/lbernail))
- CRD validation schemas should not specify `metadata` fields other than `name` and `generateName`. A schema will not be considered structural (and therefore ready for future features) if `metadata` is specified in any other way. ([#77653](https://github.com/kubernetes/kubernetes/pull/77653), [@sttts](https://github.com/sttts))

## Other notable changes

### API Machinery

- Added port configuration to Admission webhook configuration service reference.
- Added port configuration to AuditSink webhook configuration service reference.
- Added port configuration to CRD Conversion webhook configuration service reference.
- Added port configuration to kube-aggregator service reference. ([#74855](https://github.com/kubernetes/kubernetes/pull/74855), [@mbohlool](https://github.com/mbohlool))
- Implemented deduplication logic for v1beta1.Event API ([#65782](https://github.com/kubernetes/kubernetes/pull/65782), [@yastij](https://github.com/yastij))
- Added `objectSelector` to admission webhook configurations. `objectSelector` is evaluated the oldObject and newObject that would be sent to the webhook, and is considered to match if either object matches the selector. A null object (oldObject in the case of create, or newObject in the case of delete) or an object that cannot have labels (like a DeploymentRollback or a PodProxyOptions object) is not considered to match. Use the object selector only if the webhook is opt-in, because end users may skip the admission webhook by setting the labels. ([#78505](https://github.com/kubernetes/kubernetes/pull/78505), [@caesarxuchao](https://github.com/caesarxuchao))
- Watch will now support converting response objects into Table or PartialObjectMetadata forms. ([#71548](https://github.com/kubernetes/kubernetes/pull/71548), [@smarterclayton](https://github.com/smarterclayton))
- In CRD webhook conversion, Kubernetes will now ignore changes to metadata other than for labels and annotations. ([#77743](https://github.com/kubernetes/kubernetes/pull/77743), [@sttts](https://github.com/sttts))
- Added ListMeta.RemainingItemCount. When responding to a LIST request, if the server has more data available, and if the request does not contain label selectors or field selectors, the server sets the ListOptions.RemainingItemCount to the number of remaining objects. ([#75993](https://github.com/kubernetes/kubernetes/pull/75993), [@caesarxuchao](https://github.com/caesarxuchao))
- Clients may now request that API objects are converted to the `v1.Table` and `v1.PartialObjectMetadata` forms for generic access to objects. ([#77448](https://github.com/kubernetes/kubernetes/pull/77448), [@smarterclayton](https://github.com/smarterclayton))

- Fixed a spurious error where update requests to the status subresource of multi-version custom resources would complain about an incorrect API version. ([#78713](https://github.com/kubernetes/kubernetes/pull/78713), [@liggitt](https://github.com/liggitt))
- Fixed a bug in apiserver storage that could cause just-added finalizers to be ignored immediately following a delete request, leading to premature deletion. ([#77619](https://github.com/kubernetes/kubernetes/pull/77619), [@caesarxuchao](https://github.com/caesarxuchao))
- API requests rejected by admission webhooks which specify an http status code < 400 are now assigned a 400 status code. ([#77022](https://github.com/kubernetes/kubernetes/pull/77022), [@liggitt](https://github.com/liggitt))
- Fixed a transient error API requests for custom resources could encounter while changes to the CustomResourceDefinition were being applied. ([#77816](https://github.com/kubernetes/kubernetes/pull/77816), [@liggitt](https://github.com/liggitt))
[@smarterclayton](https://github.com/smarterclayton))
- Added name validation for dynamic client methods in client-go ([#75072](https://github.com/kubernetes/kubernetes/pull/75072), [@lblackstone](https://github.com/lblackstone))
- CustomResourceDefinition with invalid regular expression in the pattern field of OpenAPI v3 validation schemas are no longer considered structural. ([#78453](https://github.com/kubernetes/kubernetes/pull/78453), [@sttts](https://github.com/sttts))
- API paging is now enabled by default in k8s.io/apiserver recommended options, and in k8s.io/sample-apiserver ([#77278](https://github.com/kubernetes/kubernetes/pull/77278), [@liggitt](https://github.com/liggitt))

- Increased verbose level for local openapi aggregation logs to avoid flooding the log during normal operation ([#75781](https://github.com/kubernetes/kubernetes/pull/75781), [@roycaihw](https://github.com/roycaihw))
- k8s.io/client-go/dynamic/dynamicinformer.NewFilteredDynamicSharedInformerFactory now honours the `namespace` argument. ([#77945](https://github.com/kubernetes/kubernetes/pull/77945), [@michaelfig](https://github.com/michaelfig))
- client-go and kubectl no longer write cached discovery files with world-accessible file permissions. ([#77874](https://github.com/kubernetes/kubernetes/pull/77874), [@yuchengwu](https://github.com/yuchengwu))
- Fixed an error with stuck informers when an etcd watch receives update or delete events with missing data. ([#76675](https://github.com/kubernetes/kubernetes/pull/76675), [@ryanmcnamara](https://github.com/ryanmcnamara))
- `DelayingQueue.ShutDown()` can now be invoked multiple times without causing a closed channel panic. ([#77170](https://github.com/kubernetes/kubernetes/pull/77170), [@smarterclayton](https://github.com/smarterclayton))
- When specifying an invalid value for a label, it was not always clear which label the value was specified for. Starting with this release, the label's key is included in such error messages, which makes debugging easier. ([#77144](https://github.com/kubernetes/kubernetes/pull/77144), [@kenegozi](https://github.com/kenegozi))
- Fixed a regression error when proxying responses from aggregated API servers, which could cause watch requests to hang until the first event was received. ([#75887](https://github.com/kubernetes/kubernetes/pull/75887), [@liggitt](https://github.com/liggitt))
- Fixed a bug where dry-run is not honored for pod/eviction sub-resource. ([#76969](https://github.com/kubernetes/kubernetes/pull/76969), [@apelisse](https://github.com/apelisse))

- DeleteOptions parameters for deletecollection endpoints are now published in the OpenAPI spec. ([#77843](https://github.com/kubernetes/kubernetes/pull/77843), [@roycaihw](https://github.com/roycaihw))
- Active watches of custom resources now terminate properly if the CRD is modified. ([#78029](https://github.com/kubernetes/kubernetes/pull/78029), [@liggitt](https://github.com/liggitt))
- Fixed a potential deadlock in the resource quota controller. Enabled recording partial usage info for quota objects specifying multiple resources, when only some of the resources' usage can be determined. ([#74747](https://github.com/kubernetes/kubernetes/pull/74747), [@liggitt](https://github.com/liggitt))
- Updates that remove remaining `metadata.finalizers` from  an object that is pending deletion (non-nil metadata.deletionTimestamp) and has no graceful deletion pending (nil or 0 metadata.deletionGracePeriodSeconds) now results in immediate deletion of the object. ([#77952](https://github.com/kubernetes/kubernetes/pull/77952), [@liggitt](https://github.com/liggitt))
- client-go: The `rest.AnonymousClientConfig(*rest.Config) *rest.Config` helper method no longer copies custom `Transport` and `WrapTransport` fields, because those can be used to inject user credentials. ([#75771](https://github.com/kubernetes/kubernetes/pull/75771), [@liggitt](https://github.com/liggitt))
- Validating admission webhooks are now properly called for CREATE operations on the following resources: pods/binding, pods/eviction, bindings ([#76910](https://github.com/kubernetes/kubernetes/pull/76910), [@liggitt](https://github.com/liggitt))
- Removed the function Parallelize, please convert to use the function ParallelizeUntil. ([#76595](https://github.com/kubernetes/kubernetes/pull/76595), [@danielqsj](https://github.com/danielqsj))

### Apps

- Users can now specify a DataSource/Kind of type `PersistentVolumeClaim` in their PVC spec.  This can then be detected by the external csi-provisioner and plugins if capable. ([#76913](https://github.com/kubernetes/kubernetes/pull/76913), [@j-griffith](https://github.com/j-griffith))
- Fixed bug in DaemonSetController causing it to stop processing some DaemonSets for 5 minutes after node removal. ([#76060](https://github.com/kubernetes/kubernetes/pull/76060), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
- StatefulSet controllers no longer force a resync every 30 seconds when nothing has changed. ([#75622](https://github.com/kubernetes/kubernetes/pull/75622), [@jonsabo](https://github.com/jonsabo))
- Enhanced the daemonset sync logic to avoid a problem where pods are thought to be unavailable when the controller's clock is slower than the node's clock. ([#77208](https://github.com/kubernetes/kubernetes/pull/77208), [@DaiHao](https://github.com/DaiHao))
- Fixed a bug that caused a DaemonSet rolling update to hang when its pod gets stuck at terminating.  ([#77773](https://github.com/kubernetes/kubernetes/pull/77773), [@DaiHao](https://github.com/DaiHao))
- Route controller now respects rate limiting to the cloud provider on deletion; previously it was only for create. ([#78581](https://github.com/kubernetes/kubernetes/pull/78581), [@andrewsykim](https://github.com/andrewsykim))
- Removed extra pod creation expectations when daemonset fails to create pods in batches. ([#74856](https://github.com/kubernetes/kubernetes/pull/74856), [@draveness](https://github.com/draveness))
- Resolved spurious rollouts of workload controllers when upgrading the API server, due to incorrect defaulting of an alpha procMount field in pods. ([#78885](https://github.com/kubernetes/kubernetes/pull/78885), [@liggitt](https://github.com/liggitt))

### Auth

- Fixed OpenID Connect (OIDC) token refresh when the client secret contains a special character. ([#76914](https://github.com/kubernetes/kubernetes/pull/76914), [@tsuna](https://github.com/tsuna))
- Improved `kubectl auth can-i` command by warning users when they try to access a resource out of scope. ([#76014](https://github.com/kubernetes/kubernetes/pull/76014), [@WanLinghao](https://github.com/WanLinghao))
- Validating admission webhooks are now properly called for CREATE operations on the following resources: tokenreviews, subjectaccessreviews, localsubjectaccessreviews, selfsubjectaccessreviews, selfsubjectrulesreviews ([#76959](https://github.com/kubernetes/kubernetes/pull/76959), [@sbezverk](https://github.com/sbezverk))

### Autoscaling

- Horizontal Pod Autoscaling can now scale targets up even when one or more metrics are invalid/unavailable, as long as one metric indicates a scale up should occur. ([#78503](https://github.com/kubernetes/kubernetes/pull/78503), [@gjtempleton](https://github.com/gjtempleton))


### AWS

- Kubernetes will now use the zone from the node for topology aware aws-ebs volume creation to reduce unnecessary cloud provider calls. ([#78276](https://github.com/kubernetes/kubernetes/pull/78276), [@zhan849](https://github.com/zhan849))
- Kubernetes now supports configure accessLogs for AWS NLB. ([#78497](https://github.com/kubernetes/kubernetes/pull/78497), [@M00nF1sh](https://github.com/M00nF1sh))
- Kubernetes now supports update LoadBalancerSourceRanges for AWS NLB([#74692](https://github.com/kubernetes/kubernetes/pull/74692), [@M00nF1sh](https://github.com/M00nF1sh))
- Kubernetes now supports configure TLS termination for AWS NLB([#74910](https://github.com/kubernetes/kubernetes/pull/74910), [@M00nF1sh](https://github.com/M00nF1sh))
- Kubernetes will now consume the AWS region list from the AWS SDK instead of a hard-coded list in the cloud provider. ([#75990](https://github.com/kubernetes/kubernetes/pull/75990), [@mcrute](https://github.com/mcrute))
- Limit use of tags when calling EC2 API to prevent API throttling for very large clusters. ([#76749](https://github.com/kubernetes/kubernetes/pull/76749), [@mcrute](https://github.com/mcrute))
- The AWS credential provider can now obtain ECR credentials even without the AWS cloud provider or being on an EC2 instance. Additionally, AWS credential provider caching has been improved to honor the ECR credential timeout. ([#75587](https://github.com/kubernetes/kubernetes/pull/75587), [@tiffanyfay](https://github.com/tiffanyfay))


### Azure

- Kubernetes now supports specifying the Resource Group of the Route Table when updating the Pod network route on Azure. ([#75580](https://github.com/kubernetes/kubernetes/pull/75580), [@suker200](https://github.com/suker200))
- Kubernetes now uses instance-level update APIs for Azure VMSS loadbalancer operations. ([#76656](https://github.com/kubernetes/kubernetes/pull/76656), [@feiskyer](https://github.com/feiskyer))
- Users can now specify azure file share name in the azure file plugin, making it possible to use existing shares or specify a new share name. ([#76988](https://github.com/kubernetes/kubernetes/pull/76988), [@andyzhangx](https://github.com/andyzhangx))
- You can now run kubelet with no Azure identity. A sample cloud provider configuration is:  `{"vmType": "vmss", "useInstanceMetadata": true, "subscriptionId": "<subscriptionId>"}` ([#77906](https://github.com/kubernetes/kubernetes/pull/77906), [@feiskyer](https://github.com/feiskyer))
- Fixed some service tags not supported issues for Azure LoadBalancer service. ([#77719](https://github.com/kubernetes/kubernetes/pull/77719), [@feiskyer](https://github.com/feiskyer))
- Fixed an issue where `pull image` fails from a cross-subscription Azure Container Registry when using MSI to authenticate. ([#77245](https://github.com/kubernetes/kubernetes/pull/77245), [@norshtein](https://github.com/norshtein))
- Azure cloud provider can now be configured by Kubernetes secrets and a new option `cloudConfigType` has been introduced. Candidate values are `file`, `secret` or `merge` (default is `merge`). Note that the secret is a serialized version of `azure.json` file with key cloud-config. And the secret name is azure-cloud-provider in kube-system namespace. ([#78242](https://github.com/kubernetes/kubernetes/pull/78242), [@feiskyer](https://github.com/feiskyer))

### CLI

- Fixed `kubectl exec` usage string to correctly reflect flag placement. ([#77589](https://github.com/kubernetes/kubernetes/pull/77589), [@soltysh](https://github.com/soltysh))
- Fixed `kubectl describe cronjobs` error of `Successful Job History Limit`. ([#77347](https://github.com/kubernetes/kubernetes/pull/77347), [@danielqsj](https://github.com/danielqsj))
- In the `kubectl describe` output, the fields with names containing special characters are now displayed as-is without any pretty formatting, avoiding awkward outputs.  ([#75483](https://github.com/kubernetes/kubernetes/pull/75483), [@gsadhani](https://github.com/gsadhani))
- Fixed incorrect handling by kubectl of custom resources whose Kind is "Status". ([#77368](https://github.com/kubernetes/kubernetes/pull/77368), [@liggitt](https://github.com/liggitt))
- Report cp errors consistently, providing full message whether copying to or from a pod.  ([#77010](https://github.com/kubernetes/kubernetes/pull/77010), [@soltysh](https://github.com/soltysh))
- Preserved existing namespace information in manifests when running `
set ... --local` commands. ([#77267](https://github.com/kubernetes/kubernetes/pull/77267), [@liggitt](https://github.com/liggitt))
- Support for parsing more v1.Taint forms has been added. For example, `key:effect`, `key=:effect-` are now accepted. ([#74159](https://github.com/kubernetes/kubernetes/pull/74159), [@dlipovetsky](https://github.com/dlipovetsky))

### Cloud Provider

- The GCE-only flag `cloud-provider-gce-lb-src-cidrs` is now optional for external cloud providers. ([#76627](https://github.com/kubernetes/kubernetes/pull/76627), [@timoreimann](https://github.com/timoreimann))
- Fixed a bug where cloud-controller-manager initializes nodes multiple times. ([#75405](https://github.com/kubernetes/kubernetes/pull/75405), [@tghartland](https://github.com/tghartland))

### Cluster Lifecycle

- `kubeadm upgrade` now renews all the certificates used by a component before upgrading the component itself, with the exception of certificates signed by external CAs. User can eventually opt-out of certificate renewal during upgrades by setting the new flag `--certificate-renewal` to false. ([#76862](https://github.com/kubernetes/kubernetes/pull/76862), [@fabriziopandini](https://github.com/fabriziopandini))
- kubeadm still generates RSA keys when deploying a node, but also accepts ECDSA
keys if they already exist in the directory specified in the `--cert-dir` option. ([#76390](https://github.com/kubernetes/kubernetes/pull/76390), [@rojkov](https://github.com/rojkov))
- kubeadm now implements CRI detection for Windows worker nodes ([#78053](https://github.com/kubernetes/kubernetes/pull/78053), [@ksubrmnn](https://github.com/ksubrmnn))
- Added `--image-repository` flag to `kubeadm config images`. ([#75866](https://github.com/kubernetes/kubernetes/pull/75866), [@jmkeyes](https://github.com/jmkeyes))

- kubeadm: The kubeadm reset command has now been exposed as phases. ([#77847](https://github.com/kubernetes/kubernetes/pull/77847), [@yagonobre](https://github.com/yagonobre))
- kubeadm: Improved resiliency when it comes to updating the `kubeadm-config` configmap upon new control plane joins or resets. This allows for safe multiple control plane joins and/or resets. ([#76821](https://github.com/kubernetes/kubernetes/pull/76821), [@ereslibre](https://github.com/ereslibre))
- kubeadm: Bumped the minimum supported Docker version to 1.13.1 ([#77051](https://github.com/kubernetes/kubernetes/pull/77051), [@chenzhiwei](https://github.com/chenzhiwei))
- Reverted the CoreDNS version to 1.3.1 for kubeadm ([#78545](https://github.com/kubernetes/kubernetes/pull/78545), [@neolit123](https://github.com/neolit123))
- kubeadm: Fixed the machine readability of `kubeadm token create --print-join-command` ([#75487](https://github.com/kubernetes/kubernetes/pull/75487), [@displague](https://github.com/displague))
- `kubeadm alpha certs renew --csr-only` now reads the current certificates as the authoritative source for certificates attributes (same as kubeadm alpha certs renew). ([#77780](https://github.com/kubernetes/kubernetes/pull/77780), [@fabriziopandini](https://github.com/fabriziopandini))
- kubeadm: You can now delete multiple bootstrap tokens at once. ([#75646](https://github.com/kubernetes/kubernetes/pull/75646), [@bart0sh](https://github.com/bart0sh))
- util/initsystem: Added support for the OpenRC init system ([#73101](https://github.com/kubernetes/kubernetes/pull/73101), [@oz123](https://github.com/oz123))
- Default TTL for DNS records in kubernetes zone has been changed from 5s to 30s to keep consistent with old dnsmasq based kube-dns. The TTL can be customized with command `kubectl edit -n kube-system configmap/coredns`. ([#76238](https://github.com/kubernetes/kubernetes/pull/76238), [@Dieken](https://github.com/Dieken))
- Communication between the etcd server and kube-apiserver on master is now overridden to use HTTPS instead of HTTP when mTLS is enabled in GCE. ([#74690](https://github.com/kubernetes/kubernetes/pull/74690), [@wenjiaswe](https://github.com/wenjiaswe))

### GCP

- [stackdriver addon] Bumped prometheus-to-sd to v0.5.0 to pick up security fixes.
[fluentd-gcp addon] Bumped fluentd-gcp-scaler to v0.5.1 to pick up security fixes.
[fluentd-gcp addon] Bumped event-exporter to v0.2.4 to pick up security fixes.
[fluentd-gcp addon] Bumped prometheus-to-sd to v0.5.0 to pick up security fixes.
[metatada-proxy addon] Bumped prometheus-to-sd v0.5.0 to pick up security fixes. ([#75362](https://github.com/kubernetes/kubernetes/pull/75362), [@serathius](https://github.com/serathius))
- [fluentd-gcp addon] Bump fluentd-gcp-scaler to v0.5.2 to pick up security fixes. ([#76762](https://github.com/kubernetes/kubernetes/pull/76762), [@serathius](https://github.com/serathius))
- The GCERegionalPersistentDisk feature gate (GA in 1.13) can no longer be disabled. The feature gate will be removed in v1.17. ([#77412](https://github.com/kubernetes/kubernetes/pull/77412), [@liggitt](https://github.com/liggitt))
- GCE/Windows: When the service cannot be stopped Stackdriver logging processes are now force killed ([#77378](https://github.com/kubernetes/kubernetes/pull/77378), [@yujuhong](https://github.com/yujuhong))
- Reduced GCE log rotation check from 1 hour to every 5 minutes.  Rotation policy is unchanged (new day starts, log file size > 100MB). ([#76352](https://github.com/kubernetes/kubernetes/pull/76352), [@jpbetz](https://github.com/jpbetz))
- GCE/Windows: disabled stackdriver logging agent to prevent node startup failures ([#76099](https://github.com/kubernetes/kubernetes/pull/76099), [@yujuhong](https://github.com/yujuhong))
- API servers using the default Google Compute Engine bootstrapping scripts will have their insecure port (`:8080`) disabled by default. To enable the insecure port, set `ENABLE_APISERVER_INSECURE_PORT=true` in kube-env or as an environment variable. ([#77447](https://github.com/kubernetes/kubernetes/pull/77447), [@dekkagaijin](https://github.com/dekkagaijin))
- Fixed a NPD bug on GCI, so that it disables glog writing to files for log-counter. ([#76211](https://github.com/kubernetes/kubernetes/pull/76211), [@wangzhen127](https://github.com/wangzhen127))
- Windows nodes on GCE now have the Windows firewall enabled by default. ([#78507](https://github.com/kubernetes/kubernetes/pull/78507), [@pjh](https://github.com/pjh))
- Added `CNI_VERSION` and `CNI_SHA1` environment variables in `kube-up.sh` to configure CNI versions on GCE. ([#76353](https://github.com/kubernetes/kubernetes/pull/76353), [@Random-Liu](https://github.com/Random-Liu))
- GCE clusters will include some IP ranges that are not used on the public Internet in the list of non-masq IPs. Bumped ip-masq-agent version to v2.3.0 with flag `nomasq-all-reserved-ranges` turned on. ([#77458](https://github.com/kubernetes/kubernetes/pull/77458), [@grayluck](https://github.com/grayluck))
- GCE/Windows: added support for the stackdriver logging agent ([#76850](https://github.com/kubernetes/kubernetes/pull/76850), [@yujuhong](https://github.com/yujuhong))
- GCE Windows nodes will rely solely on kubernetes and kube-proxy (and not the GCE agent) for network address management. ([#75855](https://github.com/kubernetes/kubernetes/pull/75855), [@pjh](https://github.com/pjh))
- Ensured that the `node-role.kubernetes.io/master` taint is applied to the master with NoSchedule on GCE. ([#78183](https://github.com/kubernetes/kubernetes/pull/78183), [@cheftako](https://github.com/cheftako))
- Windows nodes on GCE now use a known-working 1809 image rather than the latest 1809 image. ([#76722](https://github.com/kubernetes/kubernetes/pull/76722), [@pjh](https://github.com/pjh))
- kube-up.sh scripts now disable the KubeletPodResources feature for Windows nodes, due to issue #[78628](https://github.com/kubernetes/kubernetes/pull/78668). ([#78668](https://github.com/kubernetes/kubernetes/pull/78668), [@mtaufen](https://github.com/mtaufen))


### Instrumentation

- [metrics-server addon] Restored the ability to connect to nodes via IP addresses. ([#76819](https://github.com/kubernetes/kubernetes/pull/76819), [@serathius](https://github.com/serathius))
- If a pod has a running instance, the stats of its previously terminated instances will not show up in the kubelet summary stats any more for CRI runtimes such as containerd and cri-o. This keeps the behavior consistent with Docker integration, and fixes an issue that some container Prometheus metrics don't work when there are summary stats for multiple instances of the same pod. ([#77426](https://github.com/kubernetes/kubernetes/pull/77426), [@Random-Liu](https://github.com/Random-Liu))


### Network

- Ingress objects are now persisted in etcd using the networking.k8s.io/v1beta1 version ([#77139](https://github.com/kubernetes/kubernetes/pull/77139), [@cmluciano](https://github.com/cmluciano))
- Transparent kube-proxy restarts when using IPVS are now allowed. ([#75283](https://github.com/kubernetes/kubernetes/pull/75283), [@lbernail](https://github.com/lbernail))
- Packets considered INVALID by conntrack are now dropped. In particular, this fixes
a problem where spurious retransmits in a long-running TCP connection to a service
IP could result in the connection being closed with the error "Connection reset by
peer" ([#74840](https://github.com/kubernetes/kubernetes/pull/74840), [@anfernee](https://github.com/anfernee))
- kube-proxy no longer automatically cleans up network rules created by running kube-proxy in other modes. If you are switching the kube-proxy mode (EG: iptables to IPVS), you will need to run `kube-proxy --cleanup`, or restart the worker node (recommended) before restarting kube-proxy. If you are not switching kube-proxy between different modes, this change should not require any action. ([#76109](https://github.com/kubernetes/kubernetes/pull/76109), [@vllry](https://github.com/vllry))
- kube-proxy: HealthzBindAddress and MetricsBindAddress now support ipv6 addresses. ([#76320](https://github.com/kubernetes/kubernetes/pull/76320), [@JieJhih](https://github.com/JieJhih))
- The userspace proxy now respects the IPTables proxy's minSyncInterval parameter. ([#71735](https://github.com/kubernetes/kubernetes/pull/71735), [@dcbw](https://github.com/dcbw))

- iptables proxier: now routes local traffic to LB IPs to service chain ([#77523](https://github.com/kubernetes/kubernetes/pull/77523), [@andrewsykim](https://github.com/andrewsykim))
- IPVS: Disabled graceful termination for UDP traffic to solve issues with high number of UDP connections (DNS / syslog in particular) ([#77802](https://github.com/kubernetes/kubernetes/pull/77802), [@lbernail](https://github.com/lbernail))
- Fixed a bug where kube-proxy returns error due to existing ipset rules using a different hash type. ([#77371](https://github.com/kubernetes/kubernetes/pull/77371), [@andrewsykim](https://github.com/andrewsykim))
- Fixed spurious error messages about failing to clean up iptables rules when using iptables 1.8. ([#77303](https://github.com/kubernetes/kubernetes/pull/77303), [@danwinship](https://github.com/danwinship))
- Increased log level to 2 for IPVS graceful termination ([#78395](https://github.com/kubernetes/kubernetes/pull/78395), [@andrewsykim](https://github.com/andrewsykim))
- kube-proxy: os exit when CleanupAndExit is set to true ([#76732](https://github.com/kubernetes/kubernetes/pull/76732), [@JieJhih](https://github.com/JieJhih))
- Kubernetes will now allow trailing dots in the externalName of Services of type ExternalName. ([#78385](https://github.com/kubernetes/kubernetes/pull/78385), [@thz](https://github.com/thz))

### Node

- The dockershim container runtime now accepts the `docker` runtime handler from a RuntimeClass. ([#78323](https://github.com/kubernetes/kubernetes/pull/78323), [@tallclair](https://github.com/tallclair))
- The init container can now get its own field value as environment variable values using downwardAPI support. ([#75109](https://github.com/kubernetes/kubernetes/pull/75109), [@yuchengwu](https://github.com/yuchengwu))
- UpdateContainerResources is no longer recorded as a `container_status` operation. It now uses the label `update_container`. ([#75278](https://github.com/kubernetes/kubernetes/pull/75278), [@Nessex](https://github.com/Nessex))
- kubelet: fix fail to close kubelet->API connections on heartbeat failure when bootstrapping or client certificate rotation is disabled ([#78016](https://github.com/kubernetes/kubernetes/pull/78016), [@gaorong](https://github.com/gaorong))
- Set selinux label at plugin socket directory ([#73241](https://github.com/kubernetes/kubernetes/pull/73241), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
- Fixed detection of non-root image user ID.([#78261](https://github.com/kubernetes/kubernetes/pull/78261), [@tallclair](https://github.com/tallclair))
- Signal handling is now initialized within hyperkube commands that require it, such as apiserver and kubelet. ([#76659](https://github.com/kubernetes/kubernetes/pull/76659), [@S-Chan](https://github.com/S-Chan))
- The Kubelet now properly requests protobuf objects where they are supported from the apiserver, reducing load in large clusters. ([#75602](https://github.com/kubernetes/kubernetes/pull/75602), [@smarterclayton](https://github.com/smarterclayton))

### OpenStack

- You can now define a kubeconfig file for the OpenStack cloud provider. ([#77415](https://github.com/kubernetes/kubernetes/pull/77415), [@Fedosin](https://github.com/Fedosin))
- OpenStack user credentials can now be read from a secret instead of a local config file. ([#75062](https://github.com/kubernetes/kubernetes/pull/75062), [@Fedosin](https://github.com/Fedosin))

### Release

- Removed hyperkube short aliases from source code because hyperkube docker image currently create these aliases. ([#76953](https://github.com/kubernetes/kubernetes/pull/76953), [@Rand01ph](https://github.com/Rand01ph))

### Scheduling

- Tolerations with the same key and effect will be merged into one that has the value of the latest toleration for best effort pods. ([#75985](https://github.com/kubernetes/kubernetes/pull/75985), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- Achieved 2X performance improvement on both required and preferred PodAffinity. ([#76243](https://github.com/kubernetes/kubernetes/pull/76243), [@Huang-Wei](https://github.com/Huang-Wei))
- Fixed a scheduler racing issue to ensure low priority pods are unschedulable on the node(s) where high priority pods have `NominatedNodeName` set to the node(s).  ([#77990](https://github.com/kubernetes/kubernetes/pull/77990), [@Huang-Wei](https://github.com/Huang-Wei))

### Storage

- Fixed issue with kubelet waiting on invalid devicepath on AWS ([#78595](https://github.com/kubernetes/kubernetes/pull/78595), [@gnufied](https://github.com/gnufied))
- StorageOS volumes now show correct mount information (node and mount time) in the StorageOS administration CLI and UI. ([#78522](https://github.com/kubernetes/kubernetes/pull/78522), [@croomes](https://github.com/croomes))
- Fixed issue in Portworx volume driver causing controller manager to crash. ([#76341](https://github.com/kubernetes/kubernetes/pull/76341), [@harsh-px](https://github.com/harsh-px))
- For an empty regular file, `stat --printf %F` will now display `regular empty file` instead of `regular file`. ([#62159](https://github.com/kubernetes/kubernetes/pull/62159), [@dixudx](https://github.com/dixudx))
- You can now have different operation names for different storage operations. This still prevents two operations on same volume from happening concurrently but if the operation changes, it resets the exponential backoff.
([#75213](https://github.com/kubernetes/kubernetes/pull/75213), [@gnufied](https://github.com/gnufied))
- Reduced event spam for `AttachVolume` storage operation. ([#75986](https://github.com/kubernetes/kubernetes/pull/75986), [@mucahitkurt](https://github.com/mucahitkurt))
- Until this release, the iscsi plugin was waiting 10 seconds for a path to appear in the device list. However this timeout is not enough, or is less than the default device discovery timeout in most systems, which prevents certain devices from being discovered. This timeout has been raised to 30 seconds, which should help to avoid mount issues due to device discovery. ([#78475](https://github.com/kubernetes/kubernetes/pull/78475), [@humblec](https://github.com/humblec))
- Added a field to store CSI volume expansion secrets ([#77516](https://github.com/kubernetes/kubernetes/pull/77516), [@gnufied](https://github.com/gnufied))
- Fixed a bug in block volume expansion. ([#77317](https://github.com/kubernetes/kubernetes/pull/77317), [@gnufied](https://github.com/gnufied))
- Count PVCs that are unbound towards attach limit. ([#73863](https://github.com/kubernetes/kubernetes/pull/73863), [@gnufied](https://github.com/gnufied))

### VMware

- SAML token delegation (required for Zones support in vSphere) is now supported ([#78876](https://github.com/kubernetes/kubernetes/pull/78876), [@dougm](https://github.com/dougm))
- vSphere SAML token auth is now supported when using Zones ([#75515](https://github.com/kubernetes/kubernetes/pull/75515), [@dougm](https://github.com/dougm))

### Windows

- Kubectl port-forward for Windows containers was added in v1.15. To use it, you’ll need to build a new pause image including WinCAT. ([#75479](https://github.com/kubernetes/kubernetes/pull/75479), [@benmoss](https://github.com/benmoss))
- We’re working to simplify the Windows node join experience with better scripts and kubeadm. Scripts and doc updates are still in the works, but some of the needed improvements are included in 1.15.  These include:
    - Windows kube-proxy will wait for HNS network creation on start ([#78612](https://github.com/kubernetes/kubernetes/pull/78612), [@ksubrmnn](https://github.com/ksubrmnn))
    - kubeadm: implemented CRI detection for Windows worker nodes ([#78053](https://github.com/kubernetes/kubernetes/pull/78053), [@ksubrmnn](https://github.com/ksubrmnn))
- Worked toward support for Windows Server version 1903, including adding Windows support for preserving the destination IP as the VIP when loadbalancing with DSR. ([#74825](https://github.com/kubernetes/kubernetes/pull/74825), [@ksubrmnn](https://github.com/ksubrmnn))
- Bug fix: Windows Kubelet nodes will now correctly search the default location for Docker credentials (`%USERPROFILE%\.docker\config.json`) when pulling images from a private registry. (https://kubernetes.io/docs/concepts/containers/images/#configuring-nodes-to-authenticate-to-a-private-registry) ([#78528](https://github.com/kubernetes/kubernetes/pull/78528), [@bclau](https://github.com/bclau))


## Dependencies

### Changed

- The default Go version was updated to 1.12.5. ([#78528](https://github.com/kubernetes/kubernetes/pull/78528))
- cri-tools has been updated to v1.14.0. ([#75658](https://github.com/kubernetes/kubernetes/pull/75658))
- Cluster Autoscaler has been updated to v1.15.0. ([#78866](https://github.com/kubernetes/kubernetes/pull/78866))
- Kibana has been upgraded to v6.6.1. ([#71251](https://github.com/kubernetes/kubernetes/pull/71251))
- CAdvisor has been updated to v0.33.2. ([#76291](https://github.com/kubernetes/kubernetes/pull/76291))
- Fluentd-gcp-scaler has been upgraded to v0.5.2. ([#76762](https://github.com/kubernetes/kubernetes/pull/76762))
- Fluentd in fluentd-elasticsearch has been upgraded to v1.4.2. ([#76854](https://github.com/kubernetes/kubernetes/pull/76854))
- fluentd-elasticsearch has been updated to v2.5.2. ([#76854](https://github.com/kubernetes/kubernetes/pull/76854))
- event-exporter has been updated to v0.2.5. ([#77815](https://github.com/kubernetes/kubernetes/pull/77815))
- es-image has been updated to Elasticsearch 6.7.2. ([#77765](https://github.com/kubernetes/kubernetes/pull/77765))
- metrics-server has been updated to v0.3.3. ([#77950](https://github.com/kubernetes/kubernetes/pull/77950))
- ip-masq-agent has been updated to v2.4.1. ([#77844](https://github.com/kubernetes/kubernetes/pull/77844))
- addon-manager has been updated to v9.0.1 ([#77282](https://github.com/kubernetes/kubernetes/pull/77282))
- go-autorest has been updated to v11.1.2 ([#77070](https://github.com/kubernetes/kubernetes/pull/77070))
- klog has been updated to 0.3.0 ([#76474](https://github.com/kubernetes/kubernetes/pull/76474))
- k8s-dns-node-cache image has been updated to v1.15.1 ([#76640](https://github.com/kubernetes/kubernetes/pull/76640), [@george-angel](https://github.com/george-angel))

### Unchanged

- Default etcd server version remains unchanged at v3.3.10. The etcd client version was updated to v3.3.10. ([#71615](https://github.com/kubernetes/kubernetes/pull/71615), [#70168](https://github.com/kubernetes/kubernetes/pull/70168), [#76917](https://github.com/kubernetes/kubernetes/pull/76917))
- The list of validated docker versions remains unchanged.
  - The current list is 1.13.1, 17.03, 17.06, 17.09, 18.06, 18.09. ([#72823](https://github.com/kubernetes/kubernetes/pull/72823), [#72831](https://github.com/kubernetes/kubernetes/pull/72831))
- CNI remains unchanged at v0.7.5. ([#75455](https://github.com/kubernetes/kubernetes/pull/75455))
- CSI remains unchanged at to v1.1.0. ([#75391](https://github.com/kubernetes/kubernetes/pull/75391))
- The dashboard add-on remains unchanged at v1.10.1. ([#72495](https://github.com/kubernetes/kubernetes/pull/72495))
- kube-dns is unchanged at v1.14.13 as of Kubernetes 1.12. ([#68900](https://github.com/kubernetes/kubernetes/pull/68900))
- Influxdb is unchanged at v1.3.3 as of Kubernetes 1.10. ([#53319](https://github.com/kubernetes/kubernetes/pull/53319))
- Grafana is unchanged at v4.4.3 as of Kubernetes 1.10. ([#53319](https://github.com/kubernetes/kubernetes/pull/53319))
- The fluent-plugin-kubernetes_metadata_filter plugin in fluentd-elasticsearch is unchanged at v2.1.6. ([#71180](https://github.com/kubernetes/kubernetes/pull/71180))
- fluentd-gcp is unchanged at v3.2.0 as of Kubernetes 1.13. ([#70954](https://github.com/kubernetes/kubernetes/pull/70954))
- OIDC authentication is unchanged at coreos/go-oidc v2 as of Kubernetes 1.10. ([#58544](https://github.com/kubernetes/kubernetes/pull/58544))
- Calico is unchanged at v3.3.1 as of Kubernetes 1.13. ([#70932](https://github.com/kubernetes/kubernetes/pull/70932))
- crictl on GCE was updated to v1.14.0. ([#75658](https://github.com/kubernetes/kubernetes/pull/75658))
- CoreDNS is unchanged at v1.3.1 as of Kubernetes 1.14. ([#78691](https://github.com/kubernetes/kubernetes/pull/78691))
- GLBC remains unchanged at v1.2.3 as of Kubernetes 1.12. ([#66793](https://github.com/kubernetes/kubernetes/pull/66793))
- Ingress-gce remains unchanged at v1.2.3 as of Kubernetes 1.12. ([#66793](https://github.com/kubernetes/kubernetes/pull/66793))
- [v1.15.0-rc.1](#v1150-rc1)
- [v1.15.0-beta.2](#v1150-beta2)
- [v1.15.0-beta.1](#v1150-beta1)
- [v1.15.0-alpha.3](#v1150-alpha3)
- [v1.15.0-alpha.2](#v1150-alpha2)
- [v1.15.0-alpha.1](#v1150-alpha1)



# v1.15.0-rc.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.15.0-rc.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes.tar.gz) | `45733de20d0e46a0937577912d945434fa12604bd507f7a6df9a28b9c60b7699f2f13f2a6b99b6cc2a8cf012391346c961deae76f5902274ea09ba17e1796c4d`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-src.tar.gz) | `63394dee48a5c69cecd26c2a8e54e6ed5c422a239b78a267c47b640f7c6774a68109179ebedd6bdb99bd9526b718831f754f75efed986dd01f8dea20988c498d`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-darwin-386.tar.gz) | `6af05492d75b4e2b510381dd7947afd104bf412cfcfff86ccf5ec1f1071928c6b100ea5baa4ce75641b50ca7f77e5130fb336674879faf69ee1bb036bbe5b2e9`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | `72e4ac298a6fc0b64673243fd0e02fe8d51d534dca6361690f204d43ae87caaf09293ff2074c25422e69312debb16c7f0bc2b285578bd585468fe09d77c829c8`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-linux-386.tar.gz) | `06f96a3b48a92ec45125fbcff64ed13466be9c0aa418dfe64e158b7a122de4e50cf75fbee76830cfb6a9d46612f579c76edb84ab7d242b44ed9bee4b0286defb`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | `ba97ccad5c572e264bccf97c69d93d49f4da02512a7e3fbfa01d5569e15cca0f23bf4dd2fb3f3e89c1f6b3aa92654a51dc3e09334ef66cc2354c91cc1904ddd9`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-linux-arm.tar.gz) | `6155c5775ebe937dabcfeb53983358e269fb43396b15a170214be0b3f682f78b682845ca1d1abbf94139752f812d887914dfff85dcb41626886d85460b8ba1a3`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | `ff6ef9f14be3c01f700546d949cfb2da91400f93bc4c8d0dc82cea442bf20593403956ffbe7934daad42d706949167b28b5bcc89e08488bbc5fa0fdd7369b753`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | `09dbec3378130acd52aee71ba0ac7ad3942ac1b05f17886868bb499c32abd89ff277d2ac28da71962ba741a5ea2cae07b3dd5ace1fc8c4fa9ffc7f7e79dd62e4`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | `8f1c211ef5764c57965d3ca197c93f8dcd768f7eb0ee9d5524f0867a8650ef8da9c21dced739697e879ba131e71311cc7df323ee7664fb35b9ea7f0149a686e3`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-windows-386.tar.gz) | `4bea6bd88eb41c7c1f0d495da6d0c7f39b55f2ccbbc0939ccd97a470aeff637bf2b2a42f94553df5073cb762787622f2467fca8c17fcc7d92619cbc26f4c3c95`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | `235e83e4bcf9535fb41a5d18dae145545ca4a7703ec6f7d6b3d0c3887c6981bb8fd12c367db2ba0cae0297724c16330978d569b2bad131aea7e1efcebef6b6a4`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | `7de5aa86903ae91e97ce3017d815ab944b2ce36b2a64b0d8222e49887013596d953c5e68fa30a3f6e8bc5973c4c247de490e6b3dd38ecdea17aa0d2dc7846841`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-server-linux-arm.tar.gz) | `05d42c2a72c7ec54adc4e61bccae842fbab3e6f4f06ac3123eb6449fe7828698eeff2f2a1bfb883f443bae1b8a97ec0703f1e6243e1a1a74d57bf383fcc007e2`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | `143152305c6b9a99d95da4e6ed479ab33b1c4a58f5386496f9b680bf7d601d87f5a0c4f9dce6aceb4d231bb7054ff5018666851192bd1db86b84bef9dedb1e01`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | `7cf9084939319cf9ab67989151dd3384ffb4eb2c2575c8654c3afac65cabe27f499349c4f48633dc15e0cdadb2bf540ef054b57eb8fbd375b63e4592cf57c5e9`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | `aaca5140e6bfeb67259d47e28da75da9a8f335ed4b61580d9f13061c4010a7739631cbb2aabbe3a9ec47023837ac2f06f7e005789f411d61c8248991a23c0982`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | `ec53dc1eb78be6e80470c5606b515e6859a245136e6b19a6bbb1f18dbc0aa192858dcf77e913138ef09426fc064dd2be8f4252a9914a0a1b358d683888a316ff`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-node-linux-arm.tar.gz) | `369e6a6f1f989af3863bc645019448964f0f1f28ace15680a888bc6e8b9192374ad823602709cb22969574876a700a3ef4c1889a8443b1526d3ccb6c6257da25`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | `c3ffd6c293feec6739881bf932c4fb5d49c01698b16bf950d63185883fcadacc2b7875e9c390423927a3a07d52971923f6f0c4c084fd073585874804e9984ead`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | `edeafe6bf1deeee4dd0174bdd3a09ece5a9a895667fcf60691a8b81ba5f99ec905cf231f9ea08ed25d58ddf692e9d1152484a085f0cfa1226ebf4476e12ccd9e`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | `3d10142101327ee9a6d754488c3e9e4fd0b5f3a43f3ef4a19c5d9da993fbab6306443c8877160de76dfecf32076606861ea4eb44e66e666036196d5f3e0e44ad`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | `514d09f3936af68746fc11e3b83f82c744ddab1c8160b59cb1b42ea8417dc0987d71040f37f6591d4df92da24e438d301932d7ccd93918692672b6176dc4f77b`

## Changelog since v1.15.0-beta.2

### Other notable changes

* Resolves spurious rollouts of workload controllers when upgrading the API server, due to incorrect defaulting of an alpha procMount field in pods ([#78885](https://github.com/kubernetes/kubernetes/pull/78885), [@liggitt](https://github.com/liggitt))
* vSphere: allow SAML token delegation (required for Zones support) ([#78876](https://github.com/kubernetes/kubernetes/pull/78876), [@dougm](https://github.com/dougm))
* Update Cluster Autoscaler to 1.15.0; changelog: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.15.0 ([#78866](https://github.com/kubernetes/kubernetes/pull/78866), [@losipiuk](https://github.com/losipiuk))
* Revert the CoreDNS version to 1.3.1 ([#78691](https://github.com/kubernetes/kubernetes/pull/78691), [@rajansandeep](https://github.com/rajansandeep))
* CRDs get support for x-kuberntes-int-or-string to allow faithful representation of IntOrString types in CustomResources. ([#78815](https://github.com/kubernetes/kubernetes/pull/78815), [@sttts](https://github.com/sttts))
* fix: retry detach azure disk issue ([#78700](https://github.com/kubernetes/kubernetes/pull/78700), [@andyzhangx](https://github.com/andyzhangx))
    * try to only update vm if detach a non-existing disk when got <200, error> after detach disk operation
* Fix issue with kubelet waiting on invalid devicepath on AWS ([#78595](https://github.com/kubernetes/kubernetes/pull/78595), [@gnufied](https://github.com/gnufied))
* Fixed a spurious error where update requests to the status subresource of multi-version custom resources would complain about an incorrect API version. ([#78713](https://github.com/kubernetes/kubernetes/pull/78713), [@liggitt](https://github.com/liggitt))
* Fix admission metrics histogram bucket sizes to cover 25ms to ~2.5 seconds. ([#78608](https://github.com/kubernetes/kubernetes/pull/78608), [@jpbetz](https://github.com/jpbetz))
* Revert Promotion of resource quota scope selector to GA ([#78696](https://github.com/kubernetes/kubernetes/pull/78696), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))



# v1.15.0-beta.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.15.0-beta.2


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes.tar.gz) | `e6c98ae93c710bb655e9b55d5ae60c56001fefb0fce74c624c18a032b94798cdfdc88ecbb1065dc36144147a9e9a77b69fba48a26097d132e708ddedde2f90b5`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-src.tar.gz) | `c9666ddb858631721f15e988bb5c30e222f0db1c38a6d67721b9ddcfac870d5f2dd8fc399736c55117ba94502ffe7ab0bb5a9e390e18a05196b463184c42da56`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `084e37b2d5d06aab37b34aba012eb6c2bb4d33bef433bef0340e306def8fddcbffb487cd150379283d11c3fa35387596780a12e306c39359f9a59106de20e8eb`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `7319108bb6e7b28575d64dadc3f397de30eb6f4f3ae1bef2001a2e84f98cb64577ff1794c41e2a700600045272b4648cd201e434f27f0ec1fb23638b86a7cac1`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-linux-386.tar.gz) | `5c4c8993c3a57f08cf08232ce5f3ecd5a2acffe9f5bc779fd00a4042a2d2099cc5fcf07c40d3524439e2fd79ebaa52c64fa06866ff3146e27b4aafd8233a6c72`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `607cd737c944d186c096d38bc256656b6226534c36ffcaab981df0a755e62fe7967649ff6d2e198348d1640302e799ab4de788bbeb297c1577e0b20f603f93c1`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `9a0aac4210c453311d432fab0925cb9b275efa2d01335443795c35e4d7dde22cbf3a2cee5f74e50c90d80b8f252ad818c4199f6019b87b57c18fa4ea50ff0408`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `6f416001e9fb42e1720302a6a46cee94952a2a825281ac7c5d6cce549f81b36b78585228ecee0fe2de56afbf44605c36a0abf100d59f25c40352c8c2e44d1168`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | `4c0e4451b6bfd08cdb851ef8e68d5206cbd55c60a65bb95e2951ab22f2f2d4a15c653ad8638a64e96b5975102db0aa338c16cea470c5f57bdf43e56db9848351`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | `d5c47fe6e79e73b426881e9ee00291952d70c65bfbdb69216e84b86ddaf2ffe5dc9447ea94d07a91a479ed85850125103d4bd0aa2ecd98c503b57d9c2018a68d`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-windows-386.tar.gz) | `d906d737a90ca0287156e42569479c9918f89f9a02e6fb800ea250a8c2a7a4792372401ecb25a342eebc2a8270ec2ebb714764af99afae83e6fe4b6a71d23f5b`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `7b0c9f14600bdfb77dc2935ba0c3407f7d5720a3a0b7ca9a18fe3fabb87a2279216cc56fa136116b28b4b3ade7f3d2cf6f3c8e31cf1809c0fe575c3b0635bca6`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `636ebe9044f0033e3eff310e781d395f31a871a53e322932f331d2496975148a415053d5f67ba4ecd562bf3c9f6e066518e6dc805e756f552a23ad370f1fb992`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `ff656458f1d19345538a4145b97821403f418a06503ef94f6c0d0662f671b54b37aedbce064dc14f2d293bb997b3c1dc77decdaf979d333bc8ba5beae01592e6`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `a95199a2b2f81c38c6c14791668598986595bedd41c9e9b2e94add0e93c5d0132f975e7a9042ae7abd4aeefd70d6a63f06030f632ecabffa358f73a575c7733f`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | `856d949df9494576e2dbd3b99d8097e97e8c4d2d195404f8307285303ff94ab7de282b55cd01d00bdafce20fa060585c97a065828269e6386abca245e15b2730`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | `7215091725f742977120f2ee4f4bc504dcff75d7258b7e90fcb4e41a2527d6cfd914d621258bd9735c08c86f53100300878eb0bbc89e13990145b77fe55dcbe1`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-node-linux-amd64.tar.gz) | `47b8c18afaa5f81b82a42309e95cf6b3f849db18bc2e8aeaaaa54ee219b5c412ba5c92276d3efe9c8fa4d10b7da1667fd7c8bede8f7a4bef9fe429ccadf910c3`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-node-linux-arm.tar.gz) | `64d5ad334f9448c3444cd90b0a6a7f07d83f4fb307e850686eb14b13f8926f832ef994c93341488dbc67750af9d5b922e0f6b9cc98316813fd1960c38c0a9f77`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-node-linux-arm64.tar.gz) | `62d1e7fb2f1f271ca349d29bc43f683e7025107d893e974131063403746bb58ce203166656985c1ff22a4eef4d6d5a3373a9f49bdf9a55ad883308aedbc33cfb`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-node-linux-ppc64le.tar.gz) | `215a2e3a40c88922427d73af3d38b6a2827c2a699a76fa7acf1a171814d36c0abec406820045ae3f33f88d087dc9ceee3b8d5e6b9c70e77fb8095d1b8aa0cf7d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-node-linux-s390x.tar.gz) | `d75f2a2fb430e7e7368f456590698fe04930c623269ffba88dd546a45ac9dd1f08f007bef28b53d232da3636c44c8f5e8e4135d8fe32ffc1bcdd45a8db883e45`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.2/kubernetes-node-windows-amd64.tar.gz) | `c8eeb1d9ada781a97dc368d308fb040124f644225579f18bb41bff0f354d65ea9e90fa2d4a161826c93c05f689abd4f7971fa80ea533c88b5a828cfc6f5a0801`

## Changelog since v1.15.0-beta.1

### Action Required

* ACTION REQUIRED  The deprecated flag --conntrack-max has been removed from kube-proxy. Users of this flag should switch to --conntrack-min and --conntrack-max-per-core instead. ([#78399](https://github.com/kubernetes/kubernetes/pull/78399), [@rikatz](https://github.com/rikatz))
* ACTION REQUIRED: kubeadm: the mixture of "--config" and "--certificate-key" is no longer allowed. The InitConfiguration and JoinConfiguration objects now support the "certificateKey" field and this field should be used instead of the command line argument in case a configuration file is already passed. ([#78542](https://github.com/kubernetes/kubernetes/pull/78542), [@neolit123](https://github.com/neolit123))
* Azure cloud provider could now be configured by Kubernetes secrets and a new option `cloudConfigType` is introduced, whose candicate values are `file`, `secret` and `merge` (default is `merge`). ([#78242](https://github.com/kubernetes/kubernetes/pull/78242), [@feiskyer](https://github.com/feiskyer))
    * action required:
    * Since Azure cloud provider would read Kubernetes secrets, the following RBAC should be configured:
    *     ---
    *     apiVersion: rbac.authorization.k8s.io/v1beta1
    *     kind: ClusterRole
    *     metadata:
    *     labels:
    *         kubernetes.io/cluster-service: "true"
    *     name: system:azure-cloud-provider-secret-getter
    *     rules:
    *     - apiGroups: [""]
    *     resources: ["secrets"]
    *     verbs:
    *     - get
    *     ---
    *     apiVersion: rbac.authorization.k8s.io/v1beta1
    *     kind: ClusterRoleBinding
    *     metadata:
    *     labels:
    *         kubernetes.io/cluster-service: "true"
    *     name: system:azure-cloud-provider-secret-getter
    *     roleRef:
    *     apiGroup: rbac.authorization.k8s.io
    *     kind: ClusterRole
    *     name: system:azure-cloud-provider-secret-getter
    *     subjects:
    *     - kind: ServiceAccount
    *     name: azure-cloud-provider
    *     namespace: kube-system

### Other notable changes

* kube-up.sh scripts now disable the KubeletPodResources feature for Windows nodes, due to issue [#78628](https://github.com/kubernetes/kubernetes/pull/78628). ([#78668](https://github.com/kubernetes/kubernetes/pull/78668), [@mtaufen](https://github.com/mtaufen))
* StorageOS volumes now show correct mount information (node and mount time) in the StorageOS administration CLI and UI. ([#78522](https://github.com/kubernetes/kubernetes/pull/78522), [@croomes](https://github.com/croomes))
* Horizontal Pod Autoscaling can now scale targets up even when one or more metrics are invalid/unavailable as long as one metric indicates a scale up should occur. ([#78503](https://github.com/kubernetes/kubernetes/pull/78503), [@gjtempleton](https://github.com/gjtempleton))
* kubeadm: revert the CoreDNS version to 1.3.1 ([#78545](https://github.com/kubernetes/kubernetes/pull/78545), [@neolit123](https://github.com/neolit123))
* Move online volume expansion to beta ([#77755](https://github.com/kubernetes/kubernetes/pull/77755), [@gnufied](https://github.com/gnufied))
* Fixes a memory leak in Kubelet on Windows caused by not not closing containers when fetching container metrics ([#78594](https://github.com/kubernetes/kubernetes/pull/78594), [@benmoss](https://github.com/benmoss))
* Windows kube-proxy will wait for HNS network creation on start ([#78612](https://github.com/kubernetes/kubernetes/pull/78612), [@ksubrmnn](https://github.com/ksubrmnn))
* Fix error handling for loading initCfg in kubeadm upgrade and apply ([#78611](https://github.com/kubernetes/kubernetes/pull/78611), [@odinuge](https://github.com/odinuge))
* Route controller now respects rate limiting to the cloud provider on deletion, previously it was only for create. ([#78581](https://github.com/kubernetes/kubernetes/pull/78581), [@andrewsykim](https://github.com/andrewsykim))
* Windows Kubelet nodes will now correctly search the default location for Docker credentials (`%USERPROFILE%\.docker* Windows nodes on GCE now have the Windows firewall enabled by default. ([#78507](https://github.com/kubernetes/kubernetes/pull/78507), [@pjh](https://github.com/pjh))
* Added objectSelector to admission webhook configurations. objectSelector is evaluated the oldObject and newObject that would be sent to the webhook, and is considered to match if either object matches the selector. A null object (oldObject in the case of create, or newObject in the case of delete) or an object that cannot have labels (like a DeploymentRollback or a PodProxyOptions object) is not considered to match. Use the object selector only if the webhook is opt-in, because end users may skip the admission webhook by setting the labels. ([#78505](https://github.com/kubernetes/kubernetes/pull/78505), [@caesarxuchao](https://github.com/caesarxuchao))
* Deprecate kubelet cAdvisor json endpoints ([#78504](https://github.com/kubernetes/kubernetes/pull/78504), [@dashpole](https://github.com/dashpole))
* Supports configure accessLogs for AWS NLB ([#78497](https://github.com/kubernetes/kubernetes/pull/78497), [@M00nF1sh](https://github.com/M00nF1sh))
* Till this release, iscsi plugin was waiting 10 seconds for a path to appear in the device list. However this timeout is not enough or less than default device discovery timeout in most of the systems which cause certain device to be not accounted for the volume. This timeout has been lifted to 30seconds from this release and it should help to avoid mount issues due to device discovery. ([#78475](https://github.com/kubernetes/kubernetes/pull/78475), [@humblec](https://github.com/humblec))
* Remove deprecated --pod/-p flag from kubectl exec. The flag has been marked as deprecated since k8s version v1.12 ([#76713](https://github.com/kubernetes/kubernetes/pull/76713), [@prksu](https://github.com/prksu))
* CustomResourceDefinition with invalid regular expression in the pattern field of OpenAPI v3 validation schemas are not considere structural. ([#78453](https://github.com/kubernetes/kubernetes/pull/78453), [@sttts](https://github.com/sttts))
* Fixed panic in kube-proxy when parsing iptables-save output ([#78428](https://github.com/kubernetes/kubernetes/pull/78428), [@luksa](https://github.com/luksa))
* Remove deprecated flag --cleanup-iptables from kube-proxy ([#78344](https://github.com/kubernetes/kubernetes/pull/78344), [@aramase](https://github.com/aramase))
* The storageVersionHash feature is beta now. "StorageVersionHash" is a field in the discovery document of each resource. It allows clients to detect if the storage version of that resource has changed. Its value must be treated as opaque by clients. Only equality comparison on the value is valid. ([#78325](https://github.com/kubernetes/kubernetes/pull/78325), [@caesarxuchao](https://github.com/caesarxuchao))
* Use zone from node for topology aware aws-ebs volume creation to reduce unnecessary cloud provider calls ([#78276](https://github.com/kubernetes/kubernetes/pull/78276), [@zhan849](https://github.com/zhan849))
* Finalizer Protection for Service LoadBalancers is now added as Alpha (disabled by default). This feature ensures the Service resource is not fully deleted until the correlating load balancer resources are deleted. ([#78262](https://github.com/kubernetes/kubernetes/pull/78262), [@MrHohn](https://github.com/MrHohn))
* Introducing new semantic for metric "volume_operation_total_seconds" to be the end to end latency of volume provisioning/deletion. Existing metric "storage_operation_duration_seconds" will remain untouched however exposed to the following potential issues: ([#78061](https://github.com/kubernetes/kubernetes/pull/78061), [@yuxiangqian](https://github.com/yuxiangqian))
    * 1. for volume's provisioned/deleted via external provisioner/deleter, "storage_operation_duration_seconds" will NOT wait for the external operation to be done before reporting latency metric (effectively close to 0). This will be fixed by using "volume_operation_total_seconds" instead
    * 2. if there's a transient error happened during "provisioning/deletion", i.e., a volume is still in-use while a deleteVolume has been called, original "storage_operation_duration_seconds" will NOT wait until a volume has been finally deleted before reporting a not accurate latency metric. The newly implemented metric "volume_operation_total_seconds", however, wait util a provisioning/deletion operation has been fully executed.
    * Potential impacts:
    * If an SLO/alert has been defined based on "volume_operation_total_seconds", it might get violated because of the more accurate metric might be significantly larger than previously reported. The metric is defined to be a histogram and the new semantic could change the distribution.
* metrics added to kubelet endpoint 'metrics/probes': ([#77975](https://github.com/kubernetes/kubernetes/pull/77975), [@logicalhan](https://github.com/logicalhan))
    *    process_start_time_seconds 
* NodeLocal DNSCache graduating to beta. ([#77887](https://github.com/kubernetes/kubernetes/pull/77887), [@prameshj](https://github.com/prameshj))
* Kubelet will attempt to use wincat.exe in the pause container for port forwarding when running on Windows ([#75479](https://github.com/kubernetes/kubernetes/pull/75479), [@benmoss](https://github.com/benmoss))
* iptables proxier: route local traffic to LB IPs to service chain ([#77523](https://github.com/kubernetes/kubernetes/pull/77523), [@andrewsykim](https://github.com/andrewsykim))
* When the number of jobs exceeds 500, cronjob should schedule without error. ([#77475](https://github.com/kubernetes/kubernetes/pull/77475), [@liucimin](https://github.com/liucimin))
* Enable 3rd party device monitoring by default ([#77274](https://github.com/kubernetes/kubernetes/pull/77274), [@RenaudWasTaken](https://github.com/RenaudWasTaken))
* This change enables a user to specify a DataSource/Kind of type "PersistentVolumeClaim" in their PVC spec.  This can then be detected by the external csi-provisioner and plugins if capable. ([#76913](https://github.com/kubernetes/kubernetes/pull/76913), [@j-griffith](https://github.com/j-griffith))
* proxy/transport: Support Content-Encoding: deflate ([#76551](https://github.com/kubernetes/kubernetes/pull/76551), [@JieJhih](https://github.com/JieJhih))
* Add --sort-by option to kubectl top command ([#75920](https://github.com/kubernetes/kubernetes/pull/75920), [@artmello](https://github.com/artmello))
* Introduce Topolgy into the runtimeClass API ([#75744](https://github.com/kubernetes/kubernetes/pull/75744), [@yastij](https://github.com/yastij))
* Kubelet plugin registration now has retry and exponential backoff logic for when registration of plugins (like CSI or device plugin) fail. ([#73891](https://github.com/kubernetes/kubernetes/pull/73891), [@taragu](https://github.com/taragu))
* Windows support for preserving the destination IP as the VIP when loadbalancing with DSR. ([#74825](https://github.com/kubernetes/kubernetes/pull/74825), [@ksubrmnn](https://github.com/ksubrmnn))
* Add  NonPrempting field to the PriorityClass. ([#74614](https://github.com/kubernetes/kubernetes/pull/74614), [@denkensk](https://github.com/denkensk))
* The kubelet only collects metrics for the node, container runtime, kubelet, pods, and containers. ([#72787](https://github.com/kubernetes/kubernetes/pull/72787), [@dashpole](https://github.com/dashpole))
* Improved README for k8s.io/sample-apiserver ([#73447](https://github.com/kubernetes/kubernetes/pull/73447), [@MikeSpreitzer](https://github.com/MikeSpreitzer))
* kubeadm: flag “--experimental-control-plane” is now deprecated. use “--control-plane” instead ([#78452](https://github.com/kubernetes/kubernetes/pull/78452), [@fabriziopandini](https://github.com/fabriziopandini))
    * kubeadm: flag “--experimental-upload-certs” is now deprecated. use “--upload-certs” instead
* Promote resource quota scope selector to GA ([#78448](https://github.com/kubernetes/kubernetes/pull/78448), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* `kubectl scale job`, deprecated since 1.10, has been removed ([#78445](https://github.com/kubernetes/kubernetes/pull/78445), [@soltysh](https://github.com/soltysh))
* CustomResourcesDefinition conversion via webhooks is promoted to beta. It requires that spec.preserveUnknownFields is set to false. ([#78426](https://github.com/kubernetes/kubernetes/pull/78426), [@sttts](https://github.com/sttts))
* kubeadm: a new command `kubeadm upgrade node` is introduced for upgrading nodes (both secondary control-plane nodes and worker nodes)  ([#78408](https://github.com/kubernetes/kubernetes/pull/78408), [@fabriziopandini](https://github.com/fabriziopandini))
    * The command `kubeadm upgrade node config` is now deprecated; use `kubeadm upgrade node` instead.
    * The command `kubeadm upgrade node experimental-control-plane` is now deprecated; use `kubeadm upgrade node` instead.
* Increase log level to 2 for IPVS graceful termination ([#78395](https://github.com/kubernetes/kubernetes/pull/78395), [@andrewsykim](https://github.com/andrewsykim))
* Add support for Azure File plugin to csi-translation-lib ([#78356](https://github.com/kubernetes/kubernetes/pull/78356), [@andyzhangx](https://github.com/andyzhangx))
* refactor AWS NLB securityGroup handling ([#74692](https://github.com/kubernetes/kubernetes/pull/74692), [@M00nF1sh](https://github.com/M00nF1sh))
* Handle resize operation for volume plugins migrated to CSI ([#77994](https://github.com/kubernetes/kubernetes/pull/77994), [@gnufied](https://github.com/gnufied))
* Inline CSI ephemeral volumes can now be controlled with PodSecurityPolicy when the CSIInlineVolume alpha feature is enabled ([#76915](https://github.com/kubernetes/kubernetes/pull/76915), [@vladimirvivien](https://github.com/vladimirvivien))
* Add support for Azure Disk plugin to csi-translation-lib ([#78330](https://github.com/kubernetes/kubernetes/pull/78330), [@andyzhangx](https://github.com/andyzhangx))
* Ensures that the node-role.kubernetes.io/master taint is applied to the master with NoSchedule on GCE. ([#78183](https://github.com/kubernetes/kubernetes/pull/78183), [@cheftako](https://github.com/cheftako))
* Add Post-bind extension point to the scheduling framework ([#77567](https://github.com/kubernetes/kubernetes/pull/77567), [@wgliang](https://github.com/wgliang))
* Add CRD support for default values in OpenAPI v3 validation schemas. `default` values are set for object fields which are undefined in request payload and in data read from etcd. Defaulting is alpha and disabled by default, if the feature gate CustomResourceDefaulting is not enabled. ([#77558](https://github.com/kubernetes/kubernetes/pull/77558), [@sttts](https://github.com/sttts))
* kubeadm: v1beta2 InitConfiguration no longer embeds ClusterConfiguration it it. ([#77739](https://github.com/kubernetes/kubernetes/pull/77739), [@rosti](https://github.com/rosti))
* kube-apiserver: the `--enable-logs-handler` flag and log-serving functionality is deprecated, and scheduled to be removed in v1.19. ([#77611](https://github.com/kubernetes/kubernetes/pull/77611), [@rohitsardesai83](https://github.com/rohitsardesai83))
* Fix vSphere SAML token auth when using Zones ([#78137](https://github.com/kubernetes/kubernetes/pull/78137), [@dougm](https://github.com/dougm))
* Admission webhooks can now register for a single version of a resource (for example, `apps/v1 deployments`) and be called when any other version of that resource is modified (for example `extensions/v1beta1 deployments`). This allows new versions of a resource to be handled by admission webhooks without needing to update every webhook to understand the new version. See the API documentation for the `matchPolicy: Equivalent` option in MutatingWebhookConfiguration and ValidatingWebhookConfiguration types. ([#78135](https://github.com/kubernetes/kubernetes/pull/78135), [@liggitt](https://github.com/liggitt))
* Add `kubeadm alpha certs certificate-key` command to generate secure random key to use on `kubeadm init --experimental-upload-certs` ([#77848](https://github.com/kubernetes/kubernetes/pull/77848), [@yagonobre](https://github.com/yagonobre))
* IPVS: Disable graceful termination for UDP traffic to solve issues with high number of UDP connections (DNS / syslog in particular) ([#77802](https://github.com/kubernetes/kubernetes/pull/77802), [@lbernail](https://github.com/lbernail))
* In CRD webhook conversion ignore changes to metadata other than for labels and annotations. ([#77743](https://github.com/kubernetes/kubernetes/pull/77743), [@sttts](https://github.com/sttts))
* Allow trailing dots in the externalName of Services of type ExternalName. ([#78385](https://github.com/kubernetes/kubernetes/pull/78385), [@thz](https://github.com/thz))
* Fix a bug where kube-proxy returns error due to existing ipset rules using a different hash type. ([#77371](https://github.com/kubernetes/kubernetes/pull/77371), [@andrewsykim](https://github.com/andrewsykim))
* kubeadm: implement CRI detection for Windows worker nodes ([#78053](https://github.com/kubernetes/kubernetes/pull/78053), [@ksubrmnn](https://github.com/ksubrmnn))



# v1.15.0-beta.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.15.0-beta.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes.tar.gz) | `c0dcbe90feaa665613a6a1ca99c1ab68d9174c5bcd3965ff9b8d9bad345dfa9e5eaa04a544262e3648438c852c5ce2c7ae34caecebefdb06091747a23098571c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-src.tar.gz) | `b79bc690792e0fbc380e47d6708250211a4e742d306fb433a1b6b50d5cea79227d4e836127f33791fb29c9a228171cd48e11bead624c8401818db03c6dc8b310`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `b79ca71cf048515084cffd9459153e6ad4898f123fda1b6aa158e5b59033e97f3b4eb1a5563c0bfe4775d56a5dc58d651d5275710b9b250db18d60cc945ea992`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `699a76b03ad3d1a38bd7e1ffb7765526cc33fb40b0e7dc0a782de3e9473e0e0d8b61a876c0d4e724450c3f2a6c2e91287eefae1c34982c84b5c76a598fbbca2c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-linux-386.tar.gz) | `5fa8bc2cbd6c9f6a8c9fe3fa96cad85f98e2d21132333ab7068b73d2c7cd27a7ebe1384fef22fdfdb755f635554efca850fe154f9f272e505a5f594f86ffadff`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `3dfbd496cd8bf9348fd2532f4c0360fe58ddfaab9d751f81cfbf9d9ddb8a347e004a9af84578aaa69bb8ee1f8cfc7adc5fd1864a32261dff94dd5a59e5f94c00`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `4abcac1fa5c1ca5e9d245e87ca6f601f7013b6a7e9a9d8dae7b322e62c8332e94f0ab63db71c0c2a535eb45bf2da51055ca5311768b8e927a0766ad99f727a72`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `22e2d6fc8eb1f64528215901c7cc8a016dda824557667199b9c9d5478f163962240426ef2a518e3981126be82a1da01cf585b1bf08d9fd2933a370beaef8d766`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `8d6f283020d76382e00b9e96f1c880654196aead67f17285ad1faf7ca7d1d2c2776e30deb9b67cee516f0efa8c260026925924ea7655881f9d75e9e5a4b8a9b7`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `3320edd26be88e9ba60b5fbb326a0e42934255bb8f1c2774eb2d309318e6dbd45d8f7162d741b7b8c056c1c0f2b943dd9939bcdde2ada80c6d9de3843e35aefe`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-windows-386.tar.gz) | `951d1c9b2e68615b6f26b85e27895a6dfea948b7e4c566e27b11fde8f32592f28de569bb9723136d830548f65018b9e9df8bf29823828778796568bff7f38c36`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `2f049941d3902b2915bea5430a29254ac0936e4890c742162993ad13a6e6e3e5b6a40cd3fc4cfd406c55eba5112b55942e6c85e5f6a5aa83d0e85853ccccb130`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `9049dc0680cb96245473422bb2c5c6ca8b1930d7e0256d993001f5de95f4c9980ded018d189b69d90c66a09af93152aa2823182ae0f3cbed72fb66a1e13a9d8c`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `38f08b9e78ea3cbe72b473cda1cd48352ee879ce0cd414c0decf2abce63bab6bdf8dc05639990c84c63faf215c581f580aadd1d73be4be233ff5c87b636184b9`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `6cd0166162fc13c9d47cb441e8dd3ff21fae6d2417d3eb780b24ebcd615ac0841ec0602e746371dc62b8bddebf94989a7e075d96718c3989dc1c12adbe366cf9`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `79570f97383f102be77478a4bc19d0d2c2551717c5f37e8aa159a0889590fc2ac0726d4899a0d9bc33e8c9e701290114222c468a76b755dc2604b113ab992ef3`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `7e1371631373407c3a1b231d09610d1029d1981026f02206a11fd58471287400809523b91de578eb26ca77a7fe4a86dcc32e225c797642733188ad043600f82e`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `819bc76079474791d468a2945c9d0858f066a54b54fcc8a84e3f9827707d6f52f9c2abcf9ea7a2dd3f68852f9bd483b8773b979c46c60e5506dc93baab3bb067`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `1054e793d5a38ac0616cc3e56c85053beda3f39bc3dad965d73397756e3d78ea07d1208b0fdd5f8e9e6a10f75da017100ef6b04fdb650983262eaad682d84c38`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `8357b8ee1ff5b2705fea1f70fdb3a10cb09ed1e48ee0507032dbadfb68b44b3c11c0c796541e6e0bbf010b20040871ca91f8edb4756d6596999092ca4931a540`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `cf62d7a660dd16ee56717a786c04b457478bf51f262fefa2d1500035ccf5bb7cc605f16ef331852f5023671d61b7c3ef348c148288c5c41fb4e309679fa51265`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `60f3eb8bfe3694f5def28661c62b67a56fb5d9efad7cfeb5dc7e76f8a15be625ac123e8ee0ac543a4464a400fca3851731d41418409d385ef8ff99156b816b0c`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `66fb625fd68a9b754e63a3e1369a21e6d2116120b5dc5aae837896f21072ce4c03d96507b66e6a239f720abcf742adef6d06d85e19bebf935d4927cccdc6817d`

## Changelog since v1.15.0-alpha.3

### Action Required

* ACTION REQUIRED: Deprecated Kubelet security controls AllowPrivileged, HostNetworkSources, HostPIDSources, HostIPCSources have been removed. Enforcement of these restrictions should be done through admission control instead (e.g. PodSecurityPolicy). ([#77820](https://github.com/kubernetes/kubernetes/pull/77820), [@dims](https://github.com/dims))
    * ACTION REQUIRED: The deprecated Kubelet flag `--allow-privileged` has been removed. Remove any use of `--allow-privileged` from your kubelet scripts or manifests.
* Fix public IPs issues when multiple clusters are sharing the same resource group. ([#77630](https://github.com/kubernetes/kubernetes/pull/77630), [@feiskyer](https://github.com/feiskyer))
    * action required: 
        * If the cluster is upgraded from old releases and the same resource group would be shared by multiple clusters, please recreate those LoadBalancer services or add a new tag 'kubernetes-cluster-name: <cluster-name>' manually for existing public IPs.
        * For multiple clusters sharing the same resource group, they should be configured with different cluster name by `kube-controller-manager --cluster-name=<cluster-name>`

### Other notable changes

* fix azure retry issue when return 2XX with error ([#78298](https://github.com/kubernetes/kubernetes/pull/78298), [@andyzhangx](https://github.com/andyzhangx))
* The dockershim container runtime now accepts the `docker` runtime handler from a RuntimeClass. ([#78323](https://github.com/kubernetes/kubernetes/pull/78323), [@tallclair](https://github.com/tallclair))
* GCE: Disable the Windows defender to work around a bug that could cause nodes to crash and reboot ([#78272](https://github.com/kubernetes/kubernetes/pull/78272), [@yujuhong](https://github.com/yujuhong))
* The CustomResourcePublishOpenAPI feature is now beta and enabled by default. CustomResourceDefinitions with [structural schemas](https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190425-structural-openapi.md) now publish schemas in the OpenAPI document served at `/openapi/v2`. CustomResourceDefinitions with non-structural schemas have a `NonStructuralSchema` condition added with details about what needs to be corrected in the validation schema. ([#77825](https://github.com/kubernetes/kubernetes/pull/77825), [@roycaihw](https://github.com/roycaihw))
* kubeadm's ignored pre-flight errors can now be configured via InitConfiguration and JoinConfiguration. ([#75499](https://github.com/kubernetes/kubernetes/pull/75499), [@marccarre](https://github.com/marccarre))
* Fix broken detection of non-root image user ID ([#78261](https://github.com/kubernetes/kubernetes/pull/78261), [@tallclair](https://github.com/tallclair))
* kubelet: fix fail to close kubelet->API connections on heartbeat failure when bootstrapping or client certificate rotation is disabled ([#78016](https://github.com/kubernetes/kubernetes/pull/78016), [@gaorong](https://github.com/gaorong))
* remove vmsizelist call in azure disk GetVolumeLimits which happens in kubelet finally ([#77851](https://github.com/kubernetes/kubernetes/pull/77851), [@andyzhangx](https://github.com/andyzhangx))
* reverts an aws-ebs volume provisioner optimization as we need to further discuss a viable optimization ([#78200](https://github.com/kubernetes/kubernetes/pull/78200), [@zhan849](https://github.com/zhan849))
* API changes and deprecating the use of special annotations for Windows GMSA support (version beta) ([#75459](https://github.com/kubernetes/kubernetes/pull/75459), [@wk8](https://github.com/wk8))
* apiextensions: publish (only) structural OpenAPI schemas ([#77554](https://github.com/kubernetes/kubernetes/pull/77554), [@sttts](https://github.com/sttts))
* Set selinux label at plugin socket directory ([#73241](https://github.com/kubernetes/kubernetes/pull/73241), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Fix a bug that causes DaemonSet rolling update to hang when its pod gets stuck at terminating.  ([#77773](https://github.com/kubernetes/kubernetes/pull/77773), [@DaiHao](https://github.com/DaiHao))
* Kubeadm: a new command `kubeadm alpha certs check-expiration` was created in order to help users in managing expiration for local PKI certificates ([#77863](https://github.com/kubernetes/kubernetes/pull/77863), [@fabriziopandini](https://github.com/fabriziopandini))
* kubeadm: fix a bug related to volume unmount if the kubelet run directory is a symbolic link ([#77507](https://github.com/kubernetes/kubernetes/pull/77507), [@cuericlee](https://github.com/cuericlee))
* n/a ([#78059](https://github.com/kubernetes/kubernetes/pull/78059), [@figo](https://github.com/figo))
* Add configuration options for the scheduling framework and its plugins. ([#77501](https://github.com/kubernetes/kubernetes/pull/77501), [@JieJhih](https://github.com/JieJhih))
* Publish DeleteOptions parameters for deletecollection endpoints in OpenAPI spec ([#77843](https://github.com/kubernetes/kubernetes/pull/77843), [@roycaihw](https://github.com/roycaihw))
* CoreDNS is now version 1.5.0 ([#78030](https://github.com/kubernetes/kubernetes/pull/78030), [@rajansandeep](https://github.com/rajansandeep))
    *     - A `ready` plugin has been included to report pod readiness
    *     - The `proxy` plugin has been deprecated. The `forward` plugin is to be used instead.
    *     - CoreDNS fixes the logging now that kubernetes’ client lib switched to klog from glog.
* Upgrade Azure network API version to 2018-07-01, so that EnableTcpReset could be enabled on Azure standard loadbalancer (SLB). ([#78012](https://github.com/kubernetes/kubernetes/pull/78012), [@feiskyer](https://github.com/feiskyer))
* Fixed a scheduler racing issue to ensure low priority pods to be unschedulable on the node(s) where high priority pods have `NominatedNodeName` set to the node(s).  ([#77990](https://github.com/kubernetes/kubernetes/pull/77990), [@Huang-Wei](https://github.com/Huang-Wei))
* Support starting Kubernetes on GCE using containerd in COS and Ubuntu with `KUBE_CONTAINER_RUNTIME=containerd`. ([#77889](https://github.com/kubernetes/kubernetes/pull/77889), [@Random-Liu](https://github.com/Random-Liu))
* DelayingQueue.ShutDown() is now able to be invoked multiple times without causing a closed channel panic. ([#77170](https://github.com/kubernetes/kubernetes/pull/77170), [@smarterclayton](https://github.com/smarterclayton))
* For admission webhooks registered for DELETE operations on k8s built APIs or CRDs, the apiserver now sends the existing object as admissionRequest.Request.OldObject to the webhook.  ([#76346](https://github.com/kubernetes/kubernetes/pull/76346), [@caesarxuchao](https://github.com/caesarxuchao))
    * For custom apiservers they uses the generic registry in the apiserver library, they get this behavior automatically.
* Expose CSI volume stats via kubelet volume metrics ([#76188](https://github.com/kubernetes/kubernetes/pull/76188), [@humblec](https://github.com/humblec))
* Active watches of custom resources now terminate properly if the CRD is modified. ([#78029](https://github.com/kubernetes/kubernetes/pull/78029), [@liggitt](https://github.com/liggitt))
* Add CRD spec.preserveUnknownFields boolean, defaulting to true in v1beta1 and to false in v1 CRDs. If false, fields not specified in the validation schema will be removed when sent to the API server or when read from etcd. ([#77333](https://github.com/kubernetes/kubernetes/pull/77333), [@sttts](https://github.com/sttts))
* Updates that remove remaining `metadata.finalizers` from  an object that is pending deletion (non-nil metadata.deletionTimestamp) and has no graceful deletion pending (nil or 0 metadata.deletionGracePeriodSeconds) now results in immediate deletion of the object. ([#77952](https://github.com/kubernetes/kubernetes/pull/77952), [@liggitt](https://github.com/liggitt))
* Deprecates the kubeadm config upload command as it's replacement is now graduated. Please see `kubeadm init phase upload-config` ([#77946](https://github.com/kubernetes/kubernetes/pull/77946), [@Klaven](https://github.com/Klaven))
* k8s.io/client-go/dynamic/dynamicinformer.NewFilteredDynamicSharedInformerFactory now honours namespace argument ([#77945](https://github.com/kubernetes/kubernetes/pull/77945), [@michaelfig](https://github.com/michaelfig))
* `kubectl rollout restart` now works for daemonsets and statefulsets. ([#77423](https://github.com/kubernetes/kubernetes/pull/77423), [@apelisse](https://github.com/apelisse))
* Fix incorrect azuredisk lun error ([#77912](https://github.com/kubernetes/kubernetes/pull/77912), [@andyzhangx](https://github.com/andyzhangx))
* Kubelet could be run with no Azure identity now. A sample cloud provider configure is:  `{"vmType": "vmss", "useInstanceMetadata": true}` ([#77906](https://github.com/kubernetes/kubernetes/pull/77906), [@feiskyer](https://github.com/feiskyer))
* client-go and kubectl no longer write cached discovery files with world-accessible file permissions ([#77874](https://github.com/kubernetes/kubernetes/pull/77874), [@yuchengwu](https://github.com/yuchengwu))
* kubeadm: expose the kubeadm reset command as phases ([#77847](https://github.com/kubernetes/kubernetes/pull/77847), [@yagonobre](https://github.com/yagonobre))
* kubeadm: kubeadm alpha certs renew  --csr-only now reads the current certificates as the authoritative source for certificates attributes (same as kubeadm alpha certs renew) ([#77780](https://github.com/kubernetes/kubernetes/pull/77780), [@fabriziopandini](https://github.com/fabriziopandini))
* Support "queue-sort" extension point for scheduling framework ([#77529](https://github.com/kubernetes/kubernetes/pull/77529), [@draveness](https://github.com/draveness))
* Allow init container to get its own field value as environment variable values(downwardAPI spport) ([#75109](https://github.com/kubernetes/kubernetes/pull/75109), [@yuchengwu](https://github.com/yuchengwu))
* The metric `kube_proxy_sync_proxy_rules_last_timestamp_seconds` is now available, indicating the last time that kube-proxy successfully applied proxying rules. ([#74027](https://github.com/kubernetes/kubernetes/pull/74027), [@squeed](https://github.com/squeed))
* Fix panic logspam when running kubelet in standalone mode. ([#77888](https://github.com/kubernetes/kubernetes/pull/77888), [@tallclair](https://github.com/tallclair))
* consume the AWS region list from the AWS SDK instead of a hard-coded list in the cloud provider ([#75990](https://github.com/kubernetes/kubernetes/pull/75990), [@mcrute](https://github.com/mcrute))
* Add `Option` field to the admission webhook `AdmissionReview` API that provides the operation options (e.g. `DeleteOption` or `CreateOption`) for the operation being performed. ([#77563](https://github.com/kubernetes/kubernetes/pull/77563), [@jpbetz](https://github.com/jpbetz))
* Fix bug where cloud-controller-manager initializes nodes multiple times ([#75405](https://github.com/kubernetes/kubernetes/pull/75405), [@tghartland](https://github.com/tghartland))
* Fixed a transient error API requests for custom resources could encounter while changes to the CustomResourceDefinition were being applied. ([#77816](https://github.com/kubernetes/kubernetes/pull/77816), [@liggitt](https://github.com/liggitt))
* Fix kubectl exec usage string ([#77589](https://github.com/kubernetes/kubernetes/pull/77589), [@soltysh](https://github.com/soltysh))
* CRD validation schemas should not specify `metadata` fields other than `name` and `generateName`. A schema will not be considered structural (and therefore ready for future features) if `metadata` is specified in any other way. ([#77653](https://github.com/kubernetes/kubernetes/pull/77653), [@sttts](https://github.com/sttts))
* Implement Permit extension point of the scheduling framework. ([#77559](https://github.com/kubernetes/kubernetes/pull/77559), [@ahg-g](https://github.com/ahg-g))
* Fixed a bug in the apiserver storage that could cause just-added finalizers to be ignored on an immediately following delete request, leading to premature deletion. ([#77619](https://github.com/kubernetes/kubernetes/pull/77619), [@caesarxuchao](https://github.com/caesarxuchao))
* add operation name for vm/vmss update operations in prometheus metrics ([#77491](https://github.com/kubernetes/kubernetes/pull/77491), [@andyzhangx](https://github.com/andyzhangx))
* fix incorrect prometheus azure metrics ([#77722](https://github.com/kubernetes/kubernetes/pull/77722), [@andyzhangx](https://github.com/andyzhangx))
* Clients may now request that API objects are converted to the `v1.Table` and `v1.PartialObjectMetadata` forms for generic access to objects. ([#77448](https://github.com/kubernetes/kubernetes/pull/77448), [@smarterclayton](https://github.com/smarterclayton))
* ingress:  Update in-tree Ingress controllers, examples, and clients to target networking.k8s.io/v1beta1 ([#77617](https://github.com/kubernetes/kubernetes/pull/77617), [@cmluciano](https://github.com/cmluciano))
* util/initsystem: add support for the OpenRC init system ([#73101](https://github.com/kubernetes/kubernetes/pull/73101), [@oz123](https://github.com/oz123))
* Signal handling is initialized within hyperkube commands that require it (apiserver, kubelet) ([#76659](https://github.com/kubernetes/kubernetes/pull/76659), [@S-Chan](https://github.com/S-Chan))
* Fix some service tags not supported issues for Azure LoadBalancer service ([#77719](https://github.com/kubernetes/kubernetes/pull/77719), [@feiskyer](https://github.com/feiskyer))
* Add Un-reserve extension point for the scheduling framework. ([#77598](https://github.com/kubernetes/kubernetes/pull/77598), [@danielqsj](https://github.com/danielqsj))
* Once merged, `legacy cloud providers` unit tests will run as part of ci, just as they were before they move from `./pkg/cloudproviders/providers`  ([#77704](https://github.com/kubernetes/kubernetes/pull/77704), [@khenidak](https://github.com/khenidak))
* Check if container memory stats are available before accessing it ([#77656](https://github.com/kubernetes/kubernetes/pull/77656), [@yastij](https://github.com/yastij))
* Add a field to store CSI volume expansion secrets ([#77516](https://github.com/kubernetes/kubernetes/pull/77516), [@gnufied](https://github.com/gnufied))
* Add a condition NonStructuralSchema to CustomResourceDefinition listing Structural Schema violations as defined in KEP https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190425-structural-openapi.md. CRD authors should update their validation schemas to be structural in order to participate in future CRD features. ([#77207](https://github.com/kubernetes/kubernetes/pull/77207), [@sttts](https://github.com/sttts))
* NONE ([#74314](https://github.com/kubernetes/kubernetes/pull/74314), [@oomichi](https://github.com/oomichi))
* Update to use go 1.12.5 ([#77528](https://github.com/kubernetes/kubernetes/pull/77528), [@cblecker](https://github.com/cblecker))
* Fix race conditions for Azure loadbalancer and route updates. ([#77490](https://github.com/kubernetes/kubernetes/pull/77490), [@feiskyer](https://github.com/feiskyer))
* remove VM API call dep in azure disk WaitForAttach ([#77483](https://github.com/kubernetes/kubernetes/pull/77483), [@andyzhangx](https://github.com/andyzhangx))
* N/A ([#77425](https://github.com/kubernetes/kubernetes/pull/77425), [@figo](https://github.com/figo))
* Fix TestEventChannelFull random fail ([#76603](https://github.com/kubernetes/kubernetes/pull/76603), [@changyaowei](https://github.com/changyaowei))
* `aws-cloud-provider` service account in the `kube-system` namespace need to be granted with list node permission with this optimization ([#76976](https://github.com/kubernetes/kubernetes/pull/76976), [@zhan849](https://github.com/zhan849))
* Remove hyperkube short aliases from source code, Because hyperkube docker image currently create these aliases. ([#76953](https://github.com/kubernetes/kubernetes/pull/76953), [@Rand01ph](https://github.com/Rand01ph))
* Allow to define kubeconfig file for OpenStack cloud provider. ([#77415](https://github.com/kubernetes/kubernetes/pull/77415), [@Fedosin](https://github.com/Fedosin))
* API servers using the default Google Compute Engine bootstrapping scripts will have their insecure port (`:8080`) disabled by default. To enable the insecure port, set `ENABLE_APISERVER_INSECURE_PORT=true` in kube-env or as an environment variable. ([#77447](https://github.com/kubernetes/kubernetes/pull/77447), [@dekkagaijin](https://github.com/dekkagaijin))
* GCE clusters will include some IP ranges that are not in used on the public Internet to the list of non-masq IPs. ([#77458](https://github.com/kubernetes/kubernetes/pull/77458), [@grayluck](https://github.com/grayluck))
    * Bump ip-masq-agent version to v2.3.0 with flag `nomasq-all-reserved-ranges` turned on.
* Implement un-reserve extension point for the scheduling framework. ([#77457](https://github.com/kubernetes/kubernetes/pull/77457), [@danielqsj](https://github.com/danielqsj))
* If a pod has a running instance, the stats of its previously terminated instances will not show up in the kubelet summary stats any more for CRI runtimes like containerd and cri-o. ([#77426](https://github.com/kubernetes/kubernetes/pull/77426), [@Random-Liu](https://github.com/Random-Liu))
    * This keeps the behavior consistent with Docker integration, and fixes an issue that some container Prometheus metrics don't work when there are summary stats for multiple instances of the same pod.
* Limit use of tags when calling EC2 API to prevent API throttling for very large clusters ([#76749](https://github.com/kubernetes/kubernetes/pull/76749), [@mcrute](https://github.com/mcrute))
* When specifying an invalid value for a label, it was not always ([#77144](https://github.com/kubernetes/kubernetes/pull/77144), [@kenegozi](https://github.com/kenegozi))
    * clear which label the value was specified for. Starting with this release, the
    * label's key is included in such error messages, which makes debugging easier.



# v1.15.0-alpha.3

[Documentation](https://docs.k8s.io)

## Downloads for v1.15.0-alpha.3


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes.tar.gz) | `88d9ced283324136e9230a0c92ad9ade10d1f52d095d5a3f9827a1ebe0cf87b5edf713cff9093cc5c61311282fe861b7c02d1da62a6ba74e2c19584e5d6084a6`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-src.tar.gz) | `c6cfe656825da66e863cd08887b3ce4374e3dae0448e33c77f960aec168c1cbad46e2485ddb9dc00f0733b4464f1e8c6e20f333097f43848decc07576ffb8d69`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `9df574b99dd03b15c784afa0bf91e826d687c5a2c7279878ddc9489e5542b2b24da5dc876eb01da0182dd4dabfda3b427875dcde16a99478923e9f74233640c1`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `bd8ac74d57e2c5dbfb36a8a3f79802a85393d914c0f513f83395f4b951a41d58ef23081d67edd1dacc039ef29bc761dcd17787b3315954f7460e15a15150dd5e`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `8ffecc41f973564b18ee6ee0cf3d2c553e9f4649b13e99dc92f427a3861b04c599e94b14ecab8b3f6018cc1248dec72cd0318c41a5d51364961cf14c8667b89c`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `8c62df3e8f02d0fe6388f82cf3af32c592783a012744b0595e5ae66097643dc6e28171322d69c1cd7e30c6b411f6f2b727728a503aec8f9d0c7cfdee44f307f5`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `6e411c605778e2a079971bfe6f066bd834dcaa13a6e1369d1a5064cc16a95aee8e6b07197522e4ef83d40692869dbd1b082a784102cad8168375202db773ce80`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `52daf658b97c66bf67b24ad45adf27e70cf8e721e616250bef06c8d4d4b6e0820647b337c38eec2673d440c2578989ba1ca1d24b4babeb7c0e22834700c225d5`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `0f2fe4d16518640a958166bc9e1963d594828e6edfa37c018778ccce79761561d0f9f8db206bd4ed122ce068d74e10cd25655bb6763fb0d53c881f0199db09bf`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `58582b030c95160460f7061000c19da225d175249beff26d4a3f5d415670ff374781b4612e1b8e01e86d31772e4ab86cd41553885d514f013df9c01cbda4b7c2`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `d2898a2e2c6d28c9069479b7dfcf5dc640864e20090441c9bb101e3f6a1cbc28051135b60143dc6b8f1edaa896e8467d3c1b7bbd7b75a3f1fb3657da6eb7385d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `50fa515ba4be8a30739cb811d8750260f2746914b98de9989c58e9b100d07f59a9b701d83a06646ccf3ad53c74b8a7a35c9eb860fb0cff27178145f457921c1b`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `b124b2fa18935bbc15b9a3c0447df931314b41d36d2cd9a65bebd090dafec9bc8f3614bf0fca97504d9d5270580b0e5e3f8564a7c8d87fde57cd593b73a7697d`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `cde20282adb8d43e350c932c5a52176c2e1accb80499631a46c6d6980c1967c324a77e295a14eb0e37702bcd26462980ac5fe5f1ee689386d974ac4c28d7b462`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `657b24b24dddb475a737be8e65669caf3c41102de5feb990b8b0f29066f823130ff759b1579a6ddbb08fef1e75edca3621054934253ef9d636f4bbcc255093ea`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `2373012c73109a38a6a2b64f1db716d62a65a4a64ccf246680f226dba96b598f9757ded4e2d3581ba4f499a28e7d8d89bbc0db98a09c812fdc7e12a014fb70ec`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `c2ce4362766bb08ffccea13893431c5f59d02f996fbb5fad1fe0014a9670440dca9e9ab4037116e19f090eeba9bdbb2ff8d2e80128afe29a86adb043a7c4e674`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `c26b0b2fff310d791c91e610252a86966df271b745a3ded8067328dab04fd3c1600bf1f67d728521472fbba067be2a2a52c927c6af4ae6cbabf237f74843b5dd`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `79e70e550a401435b0f3d06b60312bc0740924ca56607eae9cd0d12dce1a6ea1ade1a850145ba05fccec1f52eb6879767e901b6fe2e7b499cf4c632d9ebae017`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `5f920cf9e169c863760a27022f3f0e1503cedcb6b84089a7e77a05d2d449a9a68f23f1ea48924acc8221e78f151e832e07cbb5586e6e652c56c2fd6ff6009551`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `6037b555f484337e659b347ce0ca725e0a25e2e3034100a9ebc4c18668eb102093e8477cca8022cd99957a4532034ad0b7d1cf356c0bb6582f8acf9895e46423`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `a32a0a22ade7658e5fb924ca8b0ccca40e96f872d136062842c046fd3f17ecc056c22d6cfa3736cbbbac3b648299ef976ad6811ed942e13af3185d83e3440d97`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `005120b6500ee9839a6914a08ec270ccd273b5dea863da17d4da5ab1e47a7dee5b174cf5d923870186d144b954778d26e3e4445dc997411f267b200001e13e03`

## Changelog since v1.15.0-alpha.2

### Other notable changes

* Adding ListMeta.RemainingItemCount. When responding a LIST request, if the server has more data available, and if the request does not contain label selectors or field selectors, the server sets the ListOptions.RemainingItemCount to the number of remaining objects. ([#75993](https://github.com/kubernetes/kubernetes/pull/75993), [@caesarxuchao](https://github.com/caesarxuchao))
* This PR removes unused soak test cauldron ([#77335](https://github.com/kubernetes/kubernetes/pull/77335), [@loqutus](https://github.com/loqutus))
* N/A ([#76966](https://github.com/kubernetes/kubernetes/pull/76966), [@figo](https://github.com/figo))
* kubeadm: kubeadm alpha certs renew and kubeadm upgrade now supports renews of certificates embedded in KubeConfig files managed by kubeadm; this does not apply to certificates signed by external CAs.  ([#77180](https://github.com/kubernetes/kubernetes/pull/77180), [@fabriziopandini](https://github.com/fabriziopandini))
* As of Kubernetes 1.15, the SupportNodePidsLimit feature introduced as alpha in Kubernetes 1.14 is now beta, and the ability to utilize it is enabled by default.  It is no longer necessary to set the feature gate `SupportNodePidsLimit=true`.  In all other respects, this functionality behaves as it did in Kubernetes 1.14. ([#76221](https://github.com/kubernetes/kubernetes/pull/76221), [@RobertKrawitz](https://github.com/RobertKrawitz))
* Bump addon-manager to v9.0.1 ([#77282](https://github.com/kubernetes/kubernetes/pull/77282), [@MrHohn](https://github.com/MrHohn))
    * - Rebase image on debian-base:v1.0.0
* Fix kubectl describe CronJobs error of `Successful Job History Limit`. ([#77347](https://github.com/kubernetes/kubernetes/pull/77347), [@danielqsj](https://github.com/danielqsj))
* Remove extra pod creation expections when daemonset fails to create pods in batches. ([#74856](https://github.com/kubernetes/kubernetes/pull/74856), [@draveness](https://github.com/draveness))
* enhance the daemonset sync logic in clock-skew scenario ([#77208](https://github.com/kubernetes/kubernetes/pull/77208), [@DaiHao](https://github.com/DaiHao))
* GCE-only flag `cloud-provider-gce-lb-src-cidrs` becomes optional for external cloud providers. ([#76627](https://github.com/kubernetes/kubernetes/pull/76627), [@timoreimann](https://github.com/timoreimann))
* The GCERegionalPersistentDisk feature gate (GA in 1.13) can no longer be disabled. The feature gate will be removed in v1.17. ([#77412](https://github.com/kubernetes/kubernetes/pull/77412), [@liggitt](https://github.com/liggitt))
* API requests rejected by admission webhooks which specify an http status code < 400 are now assigned a 400 status code. ([#77022](https://github.com/kubernetes/kubernetes/pull/77022), [@liggitt](https://github.com/liggitt))
* kubeadm: Add ability to specify certificate encryption and decryption key for the upload/download certificates phases as part of the new v1beta2 kubeadm config format. ([#77012](https://github.com/kubernetes/kubernetes/pull/77012), [@rosti](https://github.com/rosti))
* Fixes incorrect handling by kubectl of custom resources whose Kind is "Status" ([#77368](https://github.com/kubernetes/kubernetes/pull/77368), [@liggitt](https://github.com/liggitt))
* kubeadm: disable the kube-proxy DaemonSet on non-Linux nodes. This step is required to support Windows worker nodes. ([#76327](https://github.com/kubernetes/kubernetes/pull/76327), [@neolit123](https://github.com/neolit123))
* Add etag for NSG updates so as to fix nsg race condition ([#77210](https://github.com/kubernetes/kubernetes/pull/77210), [@feiskyer](https://github.com/feiskyer))
* The `series.state` field in the events.k8s.io/v1beta1 Event API is deprecated and will be removed in v1.18 ([#75987](https://github.com/kubernetes/kubernetes/pull/75987), [@yastij](https://github.com/yastij))
* API paging is now enabled by default in k8s.io/apiserver recommended options, and in k8s.io/sample-apiserver ([#77278](https://github.com/kubernetes/kubernetes/pull/77278), [@liggitt](https://github.com/liggitt))
* GCE/Windows: force kill Stackdriver logging processes when the service cannot be stopped ([#77378](https://github.com/kubernetes/kubernetes/pull/77378), [@yujuhong](https://github.com/yujuhong))
* ingress objects are now persisted in etcd using the networking.k8s.io/v1beta1 version ([#77139](https://github.com/kubernetes/kubernetes/pull/77139), [@cmluciano](https://github.com/cmluciano))
* [fluentd-gcp addon] Bump fluentd-gcp-scaler to v0.5.2 to pick up security fixes. ([#76762](https://github.com/kubernetes/kubernetes/pull/76762), [@serathius](https://github.com/serathius))
* Add RuntimeClass restrictions & defaulting to PodSecurityPolicy. ([#73795](https://github.com/kubernetes/kubernetes/pull/73795), [@tallclair](https://github.com/tallclair))
* Promote meta.k8s.io/v1beta1 Table and PartialObjectMetadata to v1. ([#77136](https://github.com/kubernetes/kubernetes/pull/77136), [@smarterclayton](https://github.com/smarterclayton))
* Fix bug with block volume expansion ([#77317](https://github.com/kubernetes/kubernetes/pull/77317), [@gnufied](https://github.com/gnufied))
* Fixes spurious error messages about failing to clean up iptables rules when using iptables 1.8. ([#77303](https://github.com/kubernetes/kubernetes/pull/77303), [@danwinship](https://github.com/danwinship))
* Add TLS termination support for NLB ([#74910](https://github.com/kubernetes/kubernetes/pull/74910), [@M00nF1sh](https://github.com/M00nF1sh))
* Preserves existing namespace information in manifests when running `kubectl set ... --local` commands ([#77267](https://github.com/kubernetes/kubernetes/pull/77267), [@liggitt](https://github.com/liggitt))
* fix issue that pull image failed from a cross-subscription Azure Container Registry when using MSI to authenticate ([#77245](https://github.com/kubernetes/kubernetes/pull/77245), [@norshtein](https://github.com/norshtein))
* Clean links handling in cp's tar code ([#76788](https://github.com/kubernetes/kubernetes/pull/76788), [@soltysh](https://github.com/soltysh))
* Implement and update interfaces and skeleton for the scheduling framework. ([#75848](https://github.com/kubernetes/kubernetes/pull/75848), [@bsalamat](https://github.com/bsalamat))
* Fixes segmentation fault issue with Protobuf library when log entries are deeply nested. ([#77224](https://github.com/kubernetes/kubernetes/pull/77224), [@qingling128](https://github.com/qingling128))
* kubeadm: support sub-domain wildcards in certificate SANs ([#76920](https://github.com/kubernetes/kubernetes/pull/76920), [@sempr](https://github.com/sempr))
* Fixes an error with stuck informers when an etcd watch receives update or delete events with missing data ([#76675](https://github.com/kubernetes/kubernetes/pull/76675), [@ryanmcnamara](https://github.com/ryanmcnamara))



# v1.15.0-alpha.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.15.0-alpha.2


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes.tar.gz) | `88ca590c9bc2a095492310fee73bd191398375bc7f549e66e8978c48be8a9c0f9ad26e3881b84d5f2f2e49273333b3086dd99cc8c52de68e38464729f0d2828f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-src.tar.gz) | `f587073d7b58903a52beeaa911c932047294be54b6f395063c65b46a61113af1aeca37c0edc536525398f0051968708cc9bb17a2173edb8c2e8f3938ad91c0b0`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `1b944693f3813702e64f41fc11102af59beceb5ded52aac3109ebe39eb2e9103d10b26f29519337a36c86dec5c472d2b0dd5bb0264969a587345b6bb89142520`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `233bba8324f7570e527f7ef22a01552c28dbabc6eef658311668ed554923344791c2c9314678f205424a638fefebbbf67dd32be99cb70019cc77a08dbae08f4d`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `1203729b3180328631d4192c5f4cfb09e3fea958be544fe4ee3e86826422a6242d7eae9d3efba055ada4e65dbc7a3020305da97223d24416dd40686271fb3537`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `ad0613c88d4f97b2a8f35fff607bf6168724b28838587218ccece14afb52b531f723ced372de3a4014ee76ae2c738f523790178395a2b59d4b5f53fc3451fd04`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `e9d3905d306504838d417051df43431f724ea689fd3564e575f8235fc80d771b9bc72c98eae4641e9e3c5619fc93550b93634ff33d8db3b0058e348d7258ee3d`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `a426b27d0851d84b76d225b9366668521441539e7582b2439e973c98c84909fc0a236478d505c6cf50598c4ecb4796f3214ee5c80d42653ddb8e30d5ce7732be`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `be717777159b6f0c472754be704d543b80168cc02d76ca936f6559a55752530e061fe311df3906660dcaf7950a7cbea102232fb54bc4056384c11018d1dfff24`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `4a4a08d23be247e1543c85895c211e9fee8e8fa276e5aa31ed012804fa0921eeb0e5828f8ef152742b41dc1db08658dec01c0287b2828c3d3b91f260243c2457`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `8d16d655d7d4213a45a583f81b31056a02dd2100d06d8072a8ec77e255630bd9acfff062d7ab46946f94d667a8d73c611818445464638f3a3ef69c29e9aafda7`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `d4ece03464aaa9c2416d7acf9de7f94f3e01fa17f6f7469a9aedaefa90d4b0af193a1b78fb514fd9de0a55a45244a076e3897e62f9208581523690bbe0353357`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `932557827bfcc329162fcf29510f40951bdd5da4890de62fd5c44d5290349b0942ffe07bb2b518ca0f21b4de4c27ec6cfa338ec2b40e938e3a9f6e3ab5db89c0`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `e1c5349feab83ad458b9a5956026c48c7ce53f3becc09c537eda8984cea56bb254e7972d467e3b3349ad8e35cf70bebcb4b6a0ab98cbe43ab5f1238f0844d151`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `e8cfe09ff625b36b58d97440d82dbc06795d503729b45a8d077de7c73b70f350010747ad2c118ea75946e40cbf5cdfb1fdfa686c8cc714d4ec942f9bf2925664`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `99770fe0abd0ec2d5f7e38d434a82fa323b2e25124e62aadf483dd68e763b07292e9303a2c8d96964bed91cab7050e0f5be02c76919c33dcc18b46d541677022`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `3f0772f3b470d59330dd6b44a43af640a7ec42354d734a1aef491769d20a2dadaebda71cac6ad926082e03e967c6dd16ce9c440183d705c8c7c5a33f6d7b89be`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `9c879a12174a8c69124a649a8e6d51a5d4c174741d743f68f9ccec349aa671ca085e33cf63ba6047e89c9e16c2122758bbcac01eba48864cd834d18ff6c6bd36`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `3ac31c7f6b01896da60028037f30f8b6f331b7cd989dcfabd5623dbfbbed8a60ff5911fc175d976e831075587f2cd79c97f50b5cfa73bac203746bd2f6b75cd1`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `669376d5673534d53d2546bc7768f00a3add74da452061dbc2892f59efba28dc54835e4bc556c84ef54cb761f9e65f2b54e274f39faa0d609976da76fcdd87df`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `b1c7fb9fcafc216fa2bd9551399f11a592922556dfad4c56fa273a7c54426fbb63b786ecf44d71148f5c8bd08212f9915c0b784790661302b9953d6da44934d7`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `b93ae8cebd79d1ce0cb2aed66ded63b3541fcca23a1f879299c422774fb757ad3c30e782ccd7314480d247a5435c434014ed8a4cc3943b3078df0ef5b5a5b8f1`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `e99127789e045972d0c52c61902f00297c208851bb65e01d28766b6f9439f81a56e48f3fc1a20189c59ea76d3ba4ac3dd230ad054c8a2106ae8a19d4232137ba`

## Changelog since v1.15.0-alpha.1

### Other notable changes

* Kubemark scripts have been fixed for IKS clusters. ([#76909](https://github.com/kubernetes/kubernetes/pull/76909), [@Huang-Wei](https://github.com/Huang-Wei))
* fix azure disk list corruption issue ([#77187](https://github.com/kubernetes/kubernetes/pull/77187), [@andyzhangx](https://github.com/andyzhangx))
* kubeadm: kubeadm upgrade now renews all the certificates used by one component before upgrading the component itself, with the exception of certificates signed by external CAs. User can eventually opt-out from certificate renewal during upgrades by setting the new flag --certificate-renewal to false. ([#76862](https://github.com/kubernetes/kubernetes/pull/76862), [@fabriziopandini](https://github.com/fabriziopandini))
* kube-proxy: os exit when CleanupAndExit is set to true ([#76732](https://github.com/kubernetes/kubernetes/pull/76732), [@JieJhih](https://github.com/JieJhih))
* kubectl exec now allows using resource name (e.g., deployment/mydeployment) to select a matching pod. ([#73664](https://github.com/kubernetes/kubernetes/pull/73664), [@prksu](https://github.com/prksu))
    * kubectl exec now allows using --pod-running-timeout flag to wait till at least one pod is running.
* kubeadm: add optional ECDSA support. ([#76390](https://github.com/kubernetes/kubernetes/pull/76390), [@rojkov](https://github.com/rojkov))
    * kubeadm still generates RSA keys when deploying a node, but also accepts ECDSA
    * keys if they exist already in the directory specified in --cert-dir option.
* kube-proxy: HealthzBindAddress and MetricsBindAddress support ipv6 address. ([#76320](https://github.com/kubernetes/kubernetes/pull/76320), [@JieJhih](https://github.com/JieJhih))
* Packets considered INVALID by conntrack are now dropped. In particular, this fixes ([#74840](https://github.com/kubernetes/kubernetes/pull/74840), [@anfernee](https://github.com/anfernee))
    * a problem where spurious retransmits in a long-running TCP connection to a service
    * IP could result in the connection being closed with the error "Connection reset by
    * peer"
* Introduce the v1beta2 config format to kubeadm. ([#76710](https://github.com/kubernetes/kubernetes/pull/76710), [@rosti](https://github.com/rosti))
* kubeadm: bump the minimum supported Docker version to 1.13.1 ([#77051](https://github.com/kubernetes/kubernetes/pull/77051), [@chenzhiwei](https://github.com/chenzhiwei))
* Rancher credential provider has now been removed ([#77099](https://github.com/kubernetes/kubernetes/pull/77099), [@dims](https://github.com/dims))
* Support print volumeMode using `kubectl get pv/pvc -o wide` ([#76646](https://github.com/kubernetes/kubernetes/pull/76646), [@cwdsuzhou](https://github.com/cwdsuzhou))
* Upgrade go-autorest to v11.1.2 ([#77070](https://github.com/kubernetes/kubernetes/pull/77070), [@feiskyer](https://github.com/feiskyer))
* Fixes a bug where dry-run is not honored for pod/eviction sub-resource. ([#76969](https://github.com/kubernetes/kubernetes/pull/76969), [@apelisse](https://github.com/apelisse))
* Reduce event spam for AttachVolume storage operation ([#75986](https://github.com/kubernetes/kubernetes/pull/75986), [@mucahitkurt](https://github.com/mucahitkurt))
* Report cp errors consistently  ([#77010](https://github.com/kubernetes/kubernetes/pull/77010), [@soltysh](https://github.com/soltysh))
* specify azure file share name in azure file plugin ([#76988](https://github.com/kubernetes/kubernetes/pull/76988), [@andyzhangx](https://github.com/andyzhangx))
* Migrate oom watcher not relying on cAdviosr's API any more ([#74942](https://github.com/kubernetes/kubernetes/pull/74942), [@WanLinghao](https://github.com/WanLinghao))
* Validating admission webhooks are now properly called for CREATE operations on the following resources: tokenreviews, subjectaccessreviews, localsubjectaccessreviews, selfsubjectaccessreviews, selfsubjectrulesreviews ([#76959](https://github.com/kubernetes/kubernetes/pull/76959), [@sbezverk](https://github.com/sbezverk))
* Fix OpenID Connect (OIDC) token refresh when the client secret contains a special character. ([#76914](https://github.com/kubernetes/kubernetes/pull/76914), [@tsuna](https://github.com/tsuna))
* kubeadm: Improve resiliency when it comes to updating the `kubeadm-config` config map upon new control plane joins or resets. This allows for safe multiple control plane joins and/or resets. ([#76821](https://github.com/kubernetes/kubernetes/pull/76821), [@ereslibre](https://github.com/ereslibre))
* Validating admission webhooks are now properly called for CREATE operations on the following resources: pods/binding, pods/eviction, bindings ([#76910](https://github.com/kubernetes/kubernetes/pull/76910), [@liggitt](https://github.com/liggitt))
* Default TTL for DNS records in kubernetes zone is changed from 5s to 30s to keep consistent with old dnsmasq based kube-dns. The TTL can be customized with command `kubectl edit -n kube-system configmap/coredns`. ([#76238](https://github.com/kubernetes/kubernetes/pull/76238), [@Dieken](https://github.com/Dieken))
* Fixed a kubemark panic when hollow-node is morphed as proxy. ([#76848](https://github.com/kubernetes/kubernetes/pull/76848), [@Huang-Wei](https://github.com/Huang-Wei))
* k8s-dns-node-cache image version v1.15.1 ([#76640](https://github.com/kubernetes/kubernetes/pull/76640), [@george-angel](https://github.com/george-angel))
* GCE/Windows: add support for stackdriver logging agent ([#76850](https://github.com/kubernetes/kubernetes/pull/76850), [@yujuhong](https://github.com/yujuhong))
* Admission webhooks are now properly called for `scale` and `deployments/rollback` subresources ([#76849](https://github.com/kubernetes/kubernetes/pull/76849), [@liggitt](https://github.com/liggitt))
* Switch to instance-level update APIs for Azure VMSS loadbalancer operations ([#76656](https://github.com/kubernetes/kubernetes/pull/76656), [@feiskyer](https://github.com/feiskyer))
* kubeadm: kubeadm alpha cert renew now ignores certificates signed by external CAs ([#76865](https://github.com/kubernetes/kubernetes/pull/76865), [@fabriziopandini](https://github.com/fabriziopandini))
* Update to use go 1.12.4 ([#76576](https://github.com/kubernetes/kubernetes/pull/76576), [@cblecker](https://github.com/cblecker))
* [metrics-server addon] Restore connecting to nodes via IP addresses ([#76819](https://github.com/kubernetes/kubernetes/pull/76819), [@serathius](https://github.com/serathius))
* fix detach azure disk back off issue which has too big lock in failure retry condition ([#76573](https://github.com/kubernetes/kubernetes/pull/76573), [@andyzhangx](https://github.com/andyzhangx))
* Updated klog to 0.3.0 ([#76474](https://github.com/kubernetes/kubernetes/pull/76474), [@vincepri](https://github.com/vincepri))
* kube-up.sh no longer supports "centos" and "local" providers ([#76711](https://github.com/kubernetes/kubernetes/pull/76711), [@dims](https://github.com/dims))
* Ensure the backend pools are set correctly for Azure SLB with multiple backend pools (e.g. outbound rules) ([#76691](https://github.com/kubernetes/kubernetes/pull/76691), [@feiskyer](https://github.com/feiskyer))
* Windows nodes on GCE use a known-working 1809 image rather than the latest 1809 image. ([#76722](https://github.com/kubernetes/kubernetes/pull/76722), [@pjh](https://github.com/pjh))
* The userspace proxy now respects the IPTables proxy's minSyncInterval parameter. ([#71735](https://github.com/kubernetes/kubernetes/pull/71735), [@dcbw](https://github.com/dcbw))
* Kubeadm will now include the missing certificate key if it is unable to find an expected key during `kubeadm join` when used with the `--experimental-control-plane` flow ([#76636](https://github.com/kubernetes/kubernetes/pull/76636), [@mdaniel](https://github.com/mdaniel))



# v1.15.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.15.0-alpha.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes.tar.gz) | `e07246d1811bfcaf092a3244f94e4bcbfd050756aea1b56e8af54e9c016c16c9211ddeaaa08b8b398e823895dd7a8fc757e5674e11a86f1edc6f718b837cfe0c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-src.tar.gz) | `ebd902a1cfdde0d9a0062f3f21732eed76eb123da04a25f9f5c7cfce8a2926dc8331e6028c3cd27aa84aaa0bf069422a0a0b0a61e6e5f48be7fe4934e1e786fc`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `88ce20f3c1f914aebca3439b3f4b642c9c371970945a25e623730826168ebadc53706ac6f4422ea4295de86c7c6bff14ec96ad3cc8ae52d9920ecbdc9dab1729`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `a5c1a43c7e3dbb27c1a4c7e4111596331887206f768072e3fb7671075c11f2ed7c26873eef291c048415247845e86ff58aa9946a89c4aede5d847677e871ccd5`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `cf7513ab821cd0c979b1421034ce50e9bc0f347c184551cf4a9b6beab06588adda19f1b53b073525c0e73b5961beb5c1fab913c040c911acaa36496e4386a70d`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `964296e9289e12bc02ec05fb5ca9e6766654f81e1885989f8185ee8b47573ae07731e8b3cb69742b58ab1e795df8e47fd110d3226057a4c56a9ebeae162f8b35`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `3480209c2112315d81e9ac22bc2a5961a805621b82ad80dc04c7044b7a8d63b3515f77ebdfad632555468b784bab92d018aeb92c42e8b382d0ce9f358f397514`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `be7d5bb5fddfbbe95d32b354b6ed26831b1afc406dc78e9188eae3d957991ea4ceb04b434d729891d017081816125c61ea67ac10ce82773e25edb9f45b39f2d3`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `bfaeb3b8b0b2e2dde8900cd2910786cb68804ad7d173b6b52c15400041d7e8db30ff601a7de6a789a8788100eda496f0ff6d5cdcabef775d4b09117e002fe758`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `653c99e3171f74e52903ac9101cf8280a5e9d82969c53e9d481a72e0cb5b4a22951f88305545c0916ba958ca609c39c249200780fed3f9bf88fa0b2d2438259c`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `9b2862996eadf4e97d890f21bd4392beca80e356c7f94abaf5968b4ea3c2485f3391c89ce331c1de69ff9380de0c0b7be8635b079c79181e046b854b4c2530e6`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `97d87fcbc0cd821b3ca5ebfbda0b38fdc9c5a5ec58e521936163fead936995c6b26b0f05b711fbc3d61315848b6733778cb025a34de837321cf2bb0a1cca76d0`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `ffa2db2c39676e39535bcee3f41f4d178b239ca834c1aa6aafb75fb58cc5909ab94b712f2be6c0daa27ff249de6e31640fb4e5cdc7bdae82fc5dd2ad9f659518`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `a526cf7009fec5cd43da693127668006d3d6c4ebfb719e8c5b9b78bd5ad34887d337f25b309693bf844eedcc77c972c5981475ed3c00537d638985c6d6af71de`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `4f9c8f85eebbf9f0023c9311560b7576cb5f4d2eac491e38aa4050c82b34f6a09b3702b3d8c1d7737d0f27fd2df82e8b0db5ab4600ca51efd5bd21ac38049062`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `bf95f15c3edd9a7f6c2911eedd55655a60da288c9df3fed4c5b2b7cc11d5e1da063546a44268d6c3cb7d48c48d566a0776b2536f847507bcbcd419dcc8643f49`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `a2588d8b3df5f7599cd84635e5772f9ba2c665287c54a6167784bb284eb09fb0e518e9acb0e295e18a77d48cc354c8918751b63f82504177a0b1838e9e89dfd3`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `b4e9faadd0e03d3d89de496b5248547b159a7fe0c26319d898a448f3da80eb7d7d346494ca52634e89850fbb8b2db1f996bc8e7efca6cff1d26370a77b669967`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `bf6db10d15a97ae39e2fcdf32c11c6cd8afcd254dc2fbc1fc00c5c74d6179f4ed74c973f221b0f41a29ad2e7d03e5fdebf1ab927ca2e2dea010e7519badf39a9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `e89b95a23e36164b10510492841d7d140a9bd1799846f4ee1e8fbd74e8f6c512093a412edfb93bd68da10718ccdbe826f4b6ffa80e868461e7b7880c1cc44346`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `47f47c8b7fafc7d6ed0e55308ccb2a3b289e174d763c4a6415b7f1b7d2b81e4ee090a4c361eadd7cb9dd774638d0f0ad45d271ab21cc230a1b8564f06d9edae8`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `8a0af4be530008bc8f120cd82ec592d08b09a85a2a558c10d712ff44867c4ef3369b3e4e2f5a5d0c2fa375c337472b1b2e67b01ef3615eb174d36fbfd80ec2ff`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.15.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `f48886bf8f965572b78baf9e02417a56fab31870124240cac02809615caa0bc9be214d182e041fc142240f83500fe69c063d807cbe5566e9d8b64854ca39104b`

## Changelog since v1.14.0

### Action Required

* client-go: The `rest.AnonymousClientConfig(*rest.Config) *rest.Config` helper method no longer copies custom `Transport` and `WrapTransport` fields, because those can be used to inject user credentials. ([#75771](https://github.com/kubernetes/kubernetes/pull/75771), [@liggitt](https://github.com/liggitt))
* ACTION REQUIRED: The Node.Status.Volumes.Attached.DevicePath field is now unset for CSI volumes. Update any external controllers that depend on this field. ([#75799](https://github.com/kubernetes/kubernetes/pull/75799), [@msau42](https://github.com/msau42))

### Other notable changes

* Remove the function Parallelize, please convert to use the function ParallelizeUntil. ([#76595](https://github.com/kubernetes/kubernetes/pull/76595), [@danielqsj](https://github.com/danielqsj))
* StorageObjectInUseProtection admission plugin is additionally enabled by default. ([#74610](https://github.com/kubernetes/kubernetes/pull/74610), [@oomichi](https://github.com/oomichi))
    * So default enabled admission plugins are now `NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota,StorageObjectInUseProtection`. Please note that if you previously had not set the `--admission-control` flag, your cluster behavior may change (to be more standard).
* Juju provider source moved to the Charmed Kubernetes org ([#76628](https://github.com/kubernetes/kubernetes/pull/76628), [@kwmonroe](https://github.com/kwmonroe))
* improve `kubectl auth can-i` command by warning users when they try access resource out of scope ([#76014](https://github.com/kubernetes/kubernetes/pull/76014), [@WanLinghao](https://github.com/WanLinghao))
* Introduce API for watch bookmark events. ([#74074](https://github.com/kubernetes/kubernetes/pull/74074), [@wojtek-t](https://github.com/wojtek-t))
    * Introduce Alpha field `AllowWatchBookmarks` in ListOptions for requesting watch bookmarks from apiserver. The implementation in apiserver is hidden behind feature gate `WatchBookmark` (currently in Alpha stage).
* Override protocol between etcd server and kube-apiserver on master with HTTPS instead HTTP when mTLS is enabled in GCE ([#74690](https://github.com/kubernetes/kubernetes/pull/74690), [@wenjiaswe](https://github.com/wenjiaswe))
* Fix issue in Portworx volume driver causing controller manager to crash ([#76341](https://github.com/kubernetes/kubernetes/pull/76341), [@harsh-px](https://github.com/harsh-px))
* kubeadm: Fix a bug where if couple of CRIs are installed a user override of the CRI during join (via kubeadm join --cri-socket ...) is ignored and kubeadm bails out with an error ([#76505](https://github.com/kubernetes/kubernetes/pull/76505), [@rosti](https://github.com/rosti))
* UpdateContainerResources is no longer recorded as a `container_status` operation. It now uses the label `update_container` ([#75278](https://github.com/kubernetes/kubernetes/pull/75278), [@Nessex](https://github.com/Nessex))
* Bump metrics-server to v0.3.2 ([#76437](https://github.com/kubernetes/kubernetes/pull/76437), [@brett-elliott](https://github.com/brett-elliott))
* The kubelet's /spec endpoint no longer provides cloud provider information (cloud_provider, instance_type, instance_id).  ([#76291](https://github.com/kubernetes/kubernetes/pull/76291), [@dims](https://github.com/dims))
* Change kubelet probe metrics to counter type. ([#76074](https://github.com/kubernetes/kubernetes/pull/76074), [@danielqsj](https://github.com/danielqsj))
    * The metrics `prober_probe_result` is replaced by `prober_probe_total`.
* Reduce GCE log rotation check from 1 hour to every 5 minutes.  Rotation policy is unchanged (new day starts, log file size > 100MB). ([#76352](https://github.com/kubernetes/kubernetes/pull/76352), [@jpbetz](https://github.com/jpbetz))
* Add ListPager.EachListItem utility function to client-go to enable incremental processing of chunked list responses ([#75849](https://github.com/kubernetes/kubernetes/pull/75849), [@jpbetz](https://github.com/jpbetz))
* Added `CNI_VERSION` and `CNI_SHA1` environment variables in kube-up.sh to configure CNI versions on GCE. ([#76353](https://github.com/kubernetes/kubernetes/pull/76353), [@Random-Liu](https://github.com/Random-Liu))
* Update cri-tools to v1.14.0 ([#75658](https://github.com/kubernetes/kubernetes/pull/75658), [@feiskyer](https://github.com/feiskyer))
* 2X performance improvement on both required and preferred PodAffinity. ([#76243](https://github.com/kubernetes/kubernetes/pull/76243), [@Huang-Wei](https://github.com/Huang-Wei))
* scheduler: add metrics to record number of pending pods in different queues ([#75501](https://github.com/kubernetes/kubernetes/pull/75501), [@Huang-Wei](https://github.com/Huang-Wei))
* Create a new `kubectl rollout restart` command that does a rolling restart of a deployment. ([#76062](https://github.com/kubernetes/kubernetes/pull/76062), [@apelisse](https://github.com/apelisse))
* - Added port configuration to Admission webhook configuration service reference. ([#74855](https://github.com/kubernetes/kubernetes/pull/74855), [@mbohlool](https://github.com/mbohlool))
    * - Added port configuration to AuditSink webhook configuration service reference.
    * - Added port configuration to CRD Conversion webhook configuration service reference.
    * - Added port configuration to kube-aggregator service reference.
* `kubectl get -w` now prints custom resource definitions with custom print columns ([#76161](https://github.com/kubernetes/kubernetes/pull/76161), [@liggitt](https://github.com/liggitt))
* Fixes bug in DaemonSetController causing it to stop processing some DaemonSets for 5 minutes after node removal. ([#76060](https://github.com/kubernetes/kubernetes/pull/76060), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* no ([#75820](https://github.com/kubernetes/kubernetes/pull/75820), [@YoubingLi](https://github.com/YoubingLi))
* Use stdlib to log stack trace when a panic occurs ([#75853](https://github.com/kubernetes/kubernetes/pull/75853), [@roycaihw](https://github.com/roycaihw))
* Fixes a NPD bug on GCI, so that it disables glog writing to files for log-counter ([#76211](https://github.com/kubernetes/kubernetes/pull/76211), [@wangzhen127](https://github.com/wangzhen127))
* Tolerations with the same key and effect will be merged into one which has the value of the latest toleration for best effort pods. ([#75985](https://github.com/kubernetes/kubernetes/pull/75985), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Fix empty array expansion error in cluster/gce/util.sh ([#76111](https://github.com/kubernetes/kubernetes/pull/76111), [@kewu1992](https://github.com/kewu1992))
* kube-proxy no longer automatically cleans up network rules created by running kube-proxy in other modes. If you are switching the mode that kube-proxy is in running in (EG: iptables to IPVS), you will need to run `kube-proxy --cleanup`, or restart the worker node (recommended) before restarting kube-proxy. ([#76109](https://github.com/kubernetes/kubernetes/pull/76109), [@vllry](https://github.com/vllry))
    * If you are not switching kube-proxy between different modes, this change should not require any action.
* Adds a new "storage_operation_status_count" metric for kube-controller-manager and kubelet to count success and error statues. ([#75750](https://github.com/kubernetes/kubernetes/pull/75750), [@msau42](https://github.com/msau42))
* GCE/Windows: disable stackdriver logging agent to prevent node startup failures ([#76099](https://github.com/kubernetes/kubernetes/pull/76099), [@yujuhong](https://github.com/yujuhong))
* StatefulSet controllers no longer force a resync every 30 seconds when nothing has changed. ([#75622](https://github.com/kubernetes/kubernetes/pull/75622), [@jonsabo](https://github.com/jonsabo))
* Ensures the conformance test image saves results before exiting when ginkgo returns non-zero value. ([#76039](https://github.com/kubernetes/kubernetes/pull/76039), [@johnSchnake](https://github.com/johnSchnake))
* Add --image-repository flag to "kubeadm config images". ([#75866](https://github.com/kubernetes/kubernetes/pull/75866), [@jmkeyes](https://github.com/jmkeyes))
* Paginate requests from the kube-apiserver watch cache to etcd in chunks. ([#75389](https://github.com/kubernetes/kubernetes/pull/75389), [@jpbetz](https://github.com/jpbetz))
    * Paginate reflector init and resync List calls that are not served by watch cache.
* `k8s.io/kubernetes` and published components (like `k8s.io/client-go` and `k8s.io/api`) now publish go module files containing dependency version information. See http://git.k8s.io/client-go/INSTALL.md#go-modules for details on consuming `k8s.io/client-go` using go modules. ([#74877](https://github.com/kubernetes/kubernetes/pull/74877), [@liggitt](https://github.com/liggitt))
* give users the option to suppress detailed output in integration test ([#76063](https://github.com/kubernetes/kubernetes/pull/76063), [@Huang-Wei](https://github.com/Huang-Wei))
* CSI alpha CRDs have been removed ([#75747](https://github.com/kubernetes/kubernetes/pull/75747), [@msau42](https://github.com/msau42))
* Fixes a regression proxying responses from aggregated API servers which could cause watch requests to hang until the first event was received ([#75887](https://github.com/kubernetes/kubernetes/pull/75887), [@liggitt](https://github.com/liggitt))
* Support specify the Resource Group of Route Table when update Pod network route (Azure) ([#75580](https://github.com/kubernetes/kubernetes/pull/75580), [@suker200](https://github.com/suker200))
* Support parsing more v1.Taint forms. `key:effect`, `key=:effect-` are now accepted. ([#74159](https://github.com/kubernetes/kubernetes/pull/74159), [@dlipovetsky](https://github.com/dlipovetsky))
* Resource list requests for PartialObjectMetadata now correctly return list metadata like the resourceVersion and the continue token. ([#75971](https://github.com/kubernetes/kubernetes/pull/75971), [@smarterclayton](https://github.com/smarterclayton))
* `StubDomains` and `Upstreamnameserver` which contains a service name will be omitted while translating to the equivalent CoreDNS config. ([#75969](https://github.com/kubernetes/kubernetes/pull/75969), [@rajansandeep](https://github.com/rajansandeep))
* Count PVCs that are unbound towards attach limit ([#73863](https://github.com/kubernetes/kubernetes/pull/73863), [@gnufied](https://github.com/gnufied))
* Increased verbose level for local openapi aggregation logs to avoid flooding the log during normal operation ([#75781](https://github.com/kubernetes/kubernetes/pull/75781), [@roycaihw](https://github.com/roycaihw))
* In the 'kubectl describe' output, the fields with names containing special characters are displayed as-is without any pretty formatting.  ([#75483](https://github.com/kubernetes/kubernetes/pull/75483), [@gsadhani](https://github.com/gsadhani))
* Support both JSON and YAML for scheduler configuration. ([#75857](https://github.com/kubernetes/kubernetes/pull/75857), [@danielqsj](https://github.com/danielqsj))
* kubeadm: fix "upgrade plan" not defaulting to a "stable" version if no version argument is passed ([#75900](https://github.com/kubernetes/kubernetes/pull/75900), [@neolit123](https://github.com/neolit123))
* clean up func podTimestamp in queue ([#75754](https://github.com/kubernetes/kubernetes/pull/75754), [@denkensk](https://github.com/denkensk))
* The AWS credential provider can now obtain ECR credentials even without the AWS cloud provider or being on an EC2 instance. Additionally, AWS credential provider caching has been improved to honor the ECR credential timeout. ([#75587](https://github.com/kubernetes/kubernetes/pull/75587), [@tiffanyfay](https://github.com/tiffanyfay))
* Add completed job status in Cronjob event. ([#75712](https://github.com/kubernetes/kubernetes/pull/75712), [@danielqsj](https://github.com/danielqsj))
* kubeadm: implement deletion of multiple bootstrap tokens at once ([#75646](https://github.com/kubernetes/kubernetes/pull/75646), [@bart0sh](https://github.com/bart0sh))
* GCE Windows nodes will rely solely on kubernetes and kube-proxy (and not the GCE agent) for network address management. ([#75855](https://github.com/kubernetes/kubernetes/pull/75855), [@pjh](https://github.com/pjh))
* kubeadm: preflight checks on external etcd certificates are now skipped when joining a control-plane node with automatic copy of cluster certificates (--certificate-key) ([#75847](https://github.com/kubernetes/kubernetes/pull/75847), [@fabriziopandini](https://github.com/fabriziopandini))
* [stackdriver addon] Bump prometheus-to-sd to v0.5.0 to pick up security fixes. ([#75362](https://github.com/kubernetes/kubernetes/pull/75362), [@serathius](https://github.com/serathius))
    * [fluentd-gcp addon] Bump fluentd-gcp-scaler to v0.5.1 to pick up security fixes.
    * [fluentd-gcp addon] Bump event-exporter to v0.2.4 to pick up security fixes.
    * [fluentd-gcp addon] Bump prometheus-to-sd to v0.5.0 to pick up security fixes.
    * [metatada-proxy addon] Bump prometheus-to-sd v0.5.0 to pick up security fixes.
* Support describe pod with inline csi volumes ([#75513](https://github.com/kubernetes/kubernetes/pull/75513), [@cwdsuzhou](https://github.com/cwdsuzhou))
* Object count quota is now supported for namespaced custom resources using the count/<resource>.<group> syntax. ([#72384](https://github.com/kubernetes/kubernetes/pull/72384), [@zhouhaibing089](https://github.com/zhouhaibing089))
* In case kubeadm can't access the current Kubernetes version remotely and fails to parse ([#72454](https://github.com/kubernetes/kubernetes/pull/72454), [@rojkov](https://github.com/rojkov))
    * the git-based version it falls back to a static predefined value of
    * k8s.io/kubernetes/cmd/kubeadm/app/constants.CurrentKubernetesVersion.
* Fixed a potential deadlock in resource quota controller ([#74747](https://github.com/kubernetes/kubernetes/pull/74747), [@liggitt](https://github.com/liggitt))
        * Enabled recording partial usage info for quota objects specifying multiple resources, when only some of the resources' usage can be determined.
* CRI API will now be available in the kubernetes/cri-api repository ([#75531](https://github.com/kubernetes/kubernetes/pull/75531), [@dims](https://github.com/dims))
* Support vSphere SAML token auth when using Zones ([#75515](https://github.com/kubernetes/kubernetes/pull/75515), [@dougm](https://github.com/dougm))
* Transition service account controller clients to TokenRequest API ([#72179](https://github.com/kubernetes/kubernetes/pull/72179), [@WanLinghao](https://github.com/WanLinghao))
* kubeadm: reimplemented IPVS Proxy check that produced confusing warning message. ([#75036](https://github.com/kubernetes/kubernetes/pull/75036), [@bart0sh](https://github.com/bart0sh))
* Allow to read OpenStack user credentials from a secret instead of a local config file. ([#75062](https://github.com/kubernetes/kubernetes/pull/75062), [@Fedosin](https://github.com/Fedosin))
* watch can now be enabled  for events using the flag --watch-cache-sizes on kube-apiserver ([#74321](https://github.com/kubernetes/kubernetes/pull/74321), [@yastij](https://github.com/yastij))
* kubeadm: Support for deprecated old kubeadm v1alpha3 config is totally removed. ([#75179](https://github.com/kubernetes/kubernetes/pull/75179), [@rosti](https://github.com/rosti))
* The Kubelet now properly requests protobuf objects where they are ([#75602](https://github.com/kubernetes/kubernetes/pull/75602), [@smarterclayton](https://github.com/smarterclayton))
    * supported from the apiserver, reducing load in large clusters.
* Add name validation for dynamic client methods in client-go ([#75072](https://github.com/kubernetes/kubernetes/pull/75072), [@lblackstone](https://github.com/lblackstone))
* Users may now execute `get-kube-binaries.sh` to request a client for an OS/Arch unlike the one of the host on which the script is invoked. ([#74889](https://github.com/kubernetes/kubernetes/pull/74889), [@akutz](https://github.com/akutz))
* Move config local to controllers in kube-controller-manager ([#72800](https://github.com/kubernetes/kubernetes/pull/72800), [@stewart-yu](https://github.com/stewart-yu))
* Fix some potential deadlocks and file descriptor leaking for inotify watches. ([#75376](https://github.com/kubernetes/kubernetes/pull/75376), [@cpuguy83](https://github.com/cpuguy83))
* [IPVS] Introduces flag ipvs-strict-arp to configure stricter ARP sysctls, defaulting to false to preserve existing behaviors. This was enabled by default in 1.13.0, which impacted a few CNI plugins. ([#75295](https://github.com/kubernetes/kubernetes/pull/75295), [@lbernail](https://github.com/lbernail))
* [IPVS] Allow for transparent kube-proxy restarts ([#75283](https://github.com/kubernetes/kubernetes/pull/75283), [@lbernail](https://github.com/lbernail))
* Replace *_admission_latencies_milliseconds_summary and *_admission_latencies_milliseconds metrics due to reporting wrong unit (was labelled milliseconds, but reported seconds), and multiple naming guideline violations (units should be in base units and "duration" is the best practice labelling to measure the time a request takes). Please convert to use *_admission_duration_seconds and *_admission_duration_seconds_summary, these now report the unit as described, and follow the instrumentation best practices. ([#75279](https://github.com/kubernetes/kubernetes/pull/75279), [@danielqsj](https://github.com/danielqsj))
* Reset exponential backoff when storage operation changes ([#75213](https://github.com/kubernetes/kubernetes/pull/75213), [@gnufied](https://github.com/gnufied))
* Watch will now support converting response objects into Table or PartialObjectMetadata forms. ([#71548](https://github.com/kubernetes/kubernetes/pull/71548), [@smarterclayton](https://github.com/smarterclayton))
* N/A ([#74974](https://github.com/kubernetes/kubernetes/pull/74974), [@goodluckbot](https://github.com/goodluckbot))
* kubeadm: fix the machine readability of "kubeadm token create --print-join-command" ([#75487](https://github.com/kubernetes/kubernetes/pull/75487), [@displague](https://github.com/displague))
* Update Cluster Autoscaler to 1.14.0; changelog: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.14.0 ([#75480](https://github.com/kubernetes/kubernetes/pull/75480), [@losipiuk](https://github.com/losipiuk))

