<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.12.0](#v1120)
  - [Downloads for v1.12.0](#downloads-for-v1120)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Known Issues](#known-issues)
  - [Major Themes](#major-themes)
    - [SIG API Machinery](#sig-api-machinery)
    - [SIG-autoscaling](#sig-autoscaling)
    - [SIG-Azure](#sig-azure)
- [Adding Azure Availability Zones support to cloud provider.](#adding-azure-availability-zones-support-to-cloud-provider)
- [Supporting Cross RG resources (disks, Azure File and node [Experimental]](#supporting-cross-rg-resources-disks-azure-file-and-node-experimental)
    - [SIG-cli](#sig-cli)
    - [SIG-cloud-provider](#sig-cloud-provider)
    - [SIG-cluster-lifecycle](#sig-cluster-lifecycle)
    - [SIG-ibmcloud](#sig-ibmcloud)
    - [SIG-instrumentation](#sig-instrumentation)
    - [SIG-node](#sig-node)
    - [SIG-OpenStack](#sig-openstack)
    - [SIG-scheduling](#sig-scheduling)
    - [SIG-service-catalog](#sig-service-catalog)
    - [SIG-storage](#sig-storage)
    - [SIG-vmware](#sig-vmware)
    - [SIG-windows](#sig-windows)
  - [Action Required](#action-required)
  - [Deprecations and removals](#deprecations-and-removals)
  - [New Features](#new-features)
  - [API Changes](#api-changes)
  - [Other Notable Changes](#other-notable-changes)
    - [SIG API Machinery](#sig-api-machinery-1)
    - [SIG Apps](#sig-apps)
    - [SIG Auth](#sig-auth)
    - [SIG Autoscaling](#sig-autoscaling-1)
    - [SIG AWS](#sig-aws)
    - [SIG Azure](#sig-azure-1)
    - [SIG CLI](#sig-cli-1)
    - [SIG Cloud Provider](#sig-cloud-provider-1)
    - [SIG Cluster Lifecycle](#sig-cluster-lifecycle-1)
    - [SIG GCP](#sig-gcp)
    - [SIG Instrumentation](#sig-instrumentation-1)
    - [SIG Network](#sig-network)
    - [SIG Node](#sig-node-1)
    - [SIG OpenStack](#sig-openstack-1)
    - [SIG Scheduling](#sig-scheduling-1)
    - [SIG Storage](#sig-storage-1)
    - [SIG VMWare](#sig-vmware-1)
    - [SIG Windows](#sig-windows-1)
  - [Other Notable Changes](#other-notable-changes-1)
    - [Bug Fixes](#bug-fixes)
    - [Not Very Notable (that is, non-user-facing)](#not-very-notable-that-is-non-user-facing)
  - [External Dependencies](#external-dependencies)
- [v1.12.0-rc.2](#v1120-rc2)
  - [Downloads for v1.12.0-rc.2](#downloads-for-v1120-rc2)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.12.0-rc.1](#changelog-since-v1120-rc1)
    - [Other notable changes](#other-notable-changes-2)
- [v1.12.0-rc.1](#v1120-rc1)
  - [Downloads for v1.12.0-rc.1](#downloads-for-v1120-rc1)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.12.0-beta.2](#changelog-since-v1120-beta2)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-3)
- [v1.12.0-beta.2](#v1120-beta2)
  - [Downloads for v1.12.0-beta.2](#downloads-for-v1120-beta2)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.12.0-beta.1](#changelog-since-v1120-beta1)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-4)
- [v1.12.0-beta.1](#v1120-beta1)
  - [Downloads for v1.12.0-beta.1](#downloads-for-v1120-beta1)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.12.0-alpha.1](#changelog-since-v1120-alpha1)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-5)
- [v1.12.0-alpha.1](#v1120-alpha1)
  - [Downloads for v1.12.0-alpha.1](#downloads-for-v1120-alpha1)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.11.0](#changelog-since-v1110)
    - [Action Required](#action-required-4)
    - [Other notable changes](#other-notable-changes-6)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.12.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.12.0


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes.tar.gz) | `a3db4289ed722db75e51b50f6070d9ec4237c6da0c15e306846d88f4ac5d23c632e1e91c356f54be8abbaa8826c2e416adcc688612dfcb3dd9b92724e45dbefe`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-src.tar.gz) | `d7c1b837095eb1c0accdbe56020a4f9e64ecc8856fb95f872ff1eacc932948630f62df1d848320cf29f380ce8683c0e150b1a8ac815f1a00e29c5bd33061c1eb`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-darwin-386.tar.gz) | `a78608d8a1a88219425d9c6266acbf3d93bf1541862cef4c84a6b0bf4741d80f34c91eb1997587d370f69df2df07af261b724bb8ab6080528df7a65c73239471`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-darwin-amd64.tar.gz) | `eea9201e28dff246730cf43134584df0f94a3de05d1a88191ed62c20ebdab40ce9eae97852571fbc991e9b26f5e0f7042578a5113a75cec1773233e800408fd6`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-linux-386.tar.gz) | `11c5d6629cd8cbcf9ca241043774ca93085edc642b878afb77b3cef2ef26f8b018af1ade362ed742d3781975ed3b4c227b7364e44e5de4d0d96382ddeac3d764`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-linux-amd64.tar.gz) | `41d976898cd56a2899bfdcac028a54f2ea5b729320908004bdb3ea33576a1d0f25baa61e12a14c9eb011d876db56b4be91221a1f0898b471f0908b38a2fdf280`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-linux-arm.tar.gz) | `c7f363effbbbaddc85d933d4b86f5b56ce6e6472e763ae59ff6888084280a4efda21c4447afba80a479ac6b021094cb31a02c9bd522da866643c084bc03515df`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-linux-arm64.tar.gz) | `8dd0ef808d75e4456aa3fd3d109248280f7436be9c72790d99a8cd7643561160569e9ad466c75240d1b195be33241b8020047f78c83b8671b210e9eff201a644`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-linux-ppc64le.tar.gz) | `eff7b0cab10adad04558a24be283c990466380b0dcd0f71be25ac4421c88fec7291e895503308539058cfe178a7b6d4e7b1974c6cb57e2e59853e04ae626d2c3`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-linux-s390x.tar.gz) | `535fb787c8b26f4dcf9b159a7cd00ea482c4e14d5fc2cd150402ba8ea2ccfb28c2cdae73843b31b689ad8c20ccd18a6caf82935e2bdf0a7778aa2ce6aa94b17c`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-windows-386.tar.gz) | `11036a56d60c5e9ee12f02147ca9f233498a008c901e1e68196444be961440f5d544e1ca180930183f01e2a486a17e4634324e2453a5d0239504680089075aa7`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-client-windows-amd64.tar.gz) | `e560abcb8fbe733ec7d945d9e12f6e7a873dd3c0fd1cbe1ecd369775f9374f289242778deea80c47d46d62a0e392b5b64d8dc3bd1258cec088c20508b3af2c4d`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-server-linux-amd64.tar.gz) | `093d44afc221c9bdf6d5d825726404efbb07b882ca4f69186ec681273f24875f8b8b0065bceba27b1ec1727bf08ba2d0d73649ec48d5e48872b2635c21b5313c`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-server-linux-arm.tar.gz) | `a3178ed50562d24b63e27fa9bd99ccd1b244dea508b537ad08c49ce78bb4ba0fea606216135aea67b89329a0185cc27abfc36513ff186adca8ec39bb72cef9ae`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-server-linux-arm64.tar.gz) | `b8bf707dabd0710fbc4590ce75a63773339e00f32779a4b59c5039b94888acfe96689ef76a1599a870d51bd56db62d60e1c22b08b163717b3581dea7c82ad293`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-server-linux-ppc64le.tar.gz) | `a9d8e1eef7f3a548b44ebb9df3f9f6b5592773d4b89bbe17842242b8c9bb67331a4513255f54169a602933da8a731f6a8820b88c73f2c1e21f5c9d50f6d0ee07`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-server-linux-s390x.tar.gz) | `e584d42d7059ed917dcc66e328e20ef15487ccc2b0ebffa43f0c466633d8ac49d6e0f6cbdf5f9b3824cd8575acbcca02f7815651ea13616ae1043dd7d518de2d`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-node-linux-amd64.tar.gz) | `6e0d16a21bd0f9a84222838cf75532a32df350b08b5073b3dbbc3338720daf6a1c24927ee191175d2d07a5b9d3d8bf6b5aaf3cfef6dfeb1f010c6a5f442e5e5e`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-node-linux-arm.tar.gz) | `8509894b54a6e0d42aef637ef84443688e2f8ee0942b33842651e5760aad6f8283045a2bd55b8e4f43dcf63aa43a743920be524752d520d50f884dff4dd8d441`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-node-linux-arm64.tar.gz) | `f1555af73cf96d12e632b2cf42f2c4ac962d8da25fb41f36d768428a93544bee0fdcc86237e5d15d513e71795a63f39aa0c192127c3835fc1f89edd3248790a1`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-node-linux-ppc64le.tar.gz) | `fb23f3021350d3f60df4ccab113f927f3521fd1f91851e028eb05e246fe6269c25ebe0dc4257b797c61d36accab6772a3bcced0b5208e61b96756890f09aae55`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-node-linux-s390x.tar.gz) | `fbf6cb2273ab4d253693967a5ee111b5177dd23b08a26d33c1e90ec6e5bf2f1d6877858721ecdd7ad583cbfb548020ac025261bf3ebb6184911ce6f0fb1d0b20`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0/kubernetes-node-windows-amd64.tar.gz) | `fdec44561ef0e4d50c6a256aa6eb7255e5da4f6511e91f08d0e579ff13c91faa42e1e07a7992ad2a03b234d636c5f708c9a08437d837bb24e724caaec90dbf69`

> - Start SHA: 91e7b4fd31fcd3d5f436da26c980becec37ceefe
> - End Sha: 337e0e18f1aefa199bd0a1786f8eab42e948064c

## Known Issues

- Feature [#566](https://github.com/kubernetes/kubernetes/issues/566) enabling CoreDNS as the default for kube-up deployments was dropped from the release due to a scalability memory resource consumption issue observed. If a cluster operator is considering using CoreDNS on a cluster greater than 2000 nodes, it may be necessary to give more consideration to CoreDNS pod memory resource limits and experimentally measure that memory usage versus cluster resource availability.
- kube-controller-manager currently needs a writable `--cert-dir` (default is `/var/run/kubernetes`) for generating self-signed certificates, when no `--tls-cert-file` or `--tls-private-key-file` are provided.
- The `system:kube-controller-manager` ClusterRole lacks permission to `get` the `configmap` extension-apiserver-authentication. kube-controller-manager errors if run with a service account bound to the clusterrole.
- Runtime handler and Windows npipe protocol are not supported yet in crictl v1.11.x. Those features will be supported in crictl [v1.12.0](https://github.com/kubernetes-sigs/cri-tools/releases/tag/v1.12.0), together with Kubernetes v1.12.1.

## Major Themes

### SIG API Machinery

SIG API work this cycle involved development of the "dry run" functionality, which enables users to see the results of a particular command without persisting those changes.

### SIG-autoscaling

SIG Autoscaling focused on improving the Horizontal Pod Autoscaling API and algorithm:
- We released autoscaling/v2beta2, which cleans up and unifies the API
- We improved readiness detection and smoothing to work well in a larger variety or use cases

### SIG-Azure

Sig Azure was focused on two primary new alpha features:
# Adding Azure Availability Zones support to cloud provider.
# Supporting Cross RG resources (disks, Azure File and node [Experimental]

Besides the above new features, support for Azure Virtual Machine Scale Sets (VMSS) and Cluster-Autoscaler is now stable and considered GA:

- Azure virtual machine scale sets (VMSS) allow you to create and manage identical load balanced
VMs that automatically increase or decrease based on demand or a set schedule.
- With this new stable feature, Kubernetes supports the scaling of containerized applications
with Azure VMSS, including the ability to integrate it with cluster-autoscaler to automatically
adjust the size of the Kubernetes clusters based on the same conditions. 

### SIG-cli

SIG CLI focused on implementing the new plugin mechanism, providing a library with common CLI tooling for plugin authors and further refactorings of the code.

### SIG-cloud-provider

This is the first Kubernetes release for this SIG! In v1.12, SIG Cloud Provider focused on building the processes and infrastructure to better support existing and new cloud providers. Some of these initiatives (many of which are still in progress) are:

- Reporting E2E conformance test results to TestGrid from every cloud provider (in collaboration with SIG Testing & SIG Release)
- Defining minimum required documentation from each cloud provider which includes (in collaboration with SIG Docs): 
  - example manifests for the kube-apiserver, kube-controller-manager, kube-schedule, kubelet, and the cloud-controller-manager
  - labels/annotations that are consumed by any cloud specific controllers

In addition to the above, SIG Cloud Provider has been focusing on a long running effort to remove cloud provider code from kubernetes/kubernetes. 

### SIG-cluster-lifecycle

In 1.12, SIG Cluster lifecycle has focused on improving the user experience in kubeadm, by fixing a number of bugs and adding some new important features.

Here is a list of some of the changes that have been made to kubeadm:

- Kubeadm internal config has been promoted to `v1alpha3`:
  - `v1alpha1` has been removed.
  - `v1alpha3` has split apart `MasterConfiguration` into separate components; `InitConfiguration`, `ClusterConfiguration`, `JoinConfiguration`, `KubeletConfiguration`, and `KubeProxyConfiguration`
  - Different configuration types can be supplied all in the same file separated by `---`.
- Improved CRI handling
  - crictl is no longer required in docker-only setups.
  - Better detection of installed CRI.
  - Better output for image pull errors.
- Improved air-gapped and offline support
  - kubeadm now handles air-gapped environments by using the local client version as a fallback.
  - Some kubeadm commands are now allowed to work in a completely offline mode.
- Certificate handling improvements:
  - Renew certs as part of upgrade.
  - New `kubeadm alpha phase certs renew` command for renewing certificates.
  - Certificates created with kubeadm now have improved uniqueness of Distinguished Name fields.
- HA improvements:
  - `kubeadm join --experimental-control-plane` can now be used to join control plane instances to an existing cluster.
  - `kubeadm upgrade node experimental-control-plane` can now be used for upgrading secondary control plane instances created with `kubeadm join --experimental-control-plane`.
Multi-arch support (EXPERIMENTAL):
  - kubeadm now adds support for docker “schema 2” manifest lists. This is preliminary part of the process of making kubeadm based k8s deployments to support multiple architectures.
Deprecating features:
  - The Alpha feature-gates HighAvailability, SelfHosting, CertsInSecrets are now deprecated, and will be removed in k8s v1.13.0.

### SIG-ibmcloud

As a newly created SIG, the SIG-ibmcloud has mainly focused on SIG set up, sharing IBM Clouds ongoing Kubernetes work like scalability tests, Kubernetes upgrade strategy etc. with the SIG members and start working on processes to move cloud provider code to a public GitHub repo.

### SIG-instrumentation

No feature work, but a large refactoring of metrics-server as well as a number of bug fixes.

### SIG-node

SIG-node graduated the PodShareProcessNamespace feature from alpha to beta.  This feature allows a pod spec to request that all containers in a pod share a common process namespaces.  

Two alpha features were also added in this release.  

The RuntimeClass alpha feature enables a node to surface multiple runtime options to support a variety of workload types.  Examples include native linux containers, and “sandboxed” containers that isolate the container from the host kernel.  

The CustomCFSQuotaPeriod alpha feature enables node administrators to change the default period used to enforce CFS quota on a node.  This can improve performance for some workloads that experience latency while using CFS quota with the default measurement period.  Finally, the SIG continues to focus on improving reliability by fixing bugs while working out design and implementation of future features.

### SIG-OpenStack

SIG-OpenStack development was primarily focused on fixing bugs and improving feature parity with OpenStack resources. New features were primarily limited to the external provider in an effort to drive adoption of the OpenStack external provider over the in-tree provider.

In-tree bug fixes and improvements included:
- Fix load balancer status without VIP.
- Fix filtering of server status.
- Fix resizing PVC of Cinder volume.
- Disable load balancer configuration if it is not defined in cloud config.
- Add support for node shutdown taint.

The external provider includes all of the above with the additional fixes and features:
- Fix bug to prevent allocation of existing floating IP.
- Fix Cinder authentication bug when OS_DOMAIN_NAME not specified.
- Fix Keystone authentication errors by skipping synchronization for unscoped tokens.
- Fix authentication error for client-auth-plugin
- Fix dependency references from in-tree-provider to point to external provider.
- Add shutdown instance by Provider ID.
- Add annotation to preserve floating IP after service delete.
- Add conformance testing to stable and development branches.
- Add support support to Manilla for trustee authentication and supplying custom CAs.
- Add and update documentation.
- Add support to Manilla for provisioning existing shares.
- Add cluster name to load balancer description
- Add synchronization between Kubernetes and Keystone projects
- Add use internal DNS name for 'hostname' of nodes.
- Add support for CSI spec v0.3.0 for both Cinder and Manilla
- Add 'cascade delete' support for Octavia load balancers to improve performance.
- Add improved load balancer naming.

### SIG-scheduling

SIG Scheduling development efforts have been primarily focused on improving performance and reliability of the scheduler.
- Performance of the inter-pod affinity/anti-affinity feature is improved over 100X via algorithmic optimization.
- DaemonSet pods, which used to be scheduled by the DaemonSet controller, will be scheduled by the default scheduler in 1.12. This change allows DaemonSet pods to enjoy all the scheduling features of the default scheduler.
- The Image Locality priority function of the scheduler has been improved and is now enabled by default. With this feature enabled, nodes that have all or a partial set of images required for running a pod are preferred over other nodes, which improves pod start-up time.
- TaintNodeByCondition has been moved to Beta and is enabled by default.
- Scheduler throughput has been improved by ~50% in large clusters (>2000 nodes).

### SIG-service-catalog
- The Originating Identity feature, which lets the broker know which user that performed an action, is now GA.
- [Namespaced Brokers](https://svc-cat.io/docs/namespaced-broker-resources/), which enable operators to install a broker into a namespace instead of the cluster level, reached GA.
- The [Service Plan Defaults](https://svc-cat.io/docs/service-plan-defaults/) feature is in alpha and is under active development. This feature gives operators the ability to define defaults for when someone provisions a service.
- We now support [filtering which services are exposed by Service Catalog](https://svc-cat.io/docs/catalog-restrictions/).
- We have also Improved the CLI experience both for kubectl and svcat by improving the output formatting, and by adding more commands.

### SIG-storage

SIG Storage promoted the [Kubernetes volume topology feature](https://github.com/kubernetes/features/issues/490) to beta. This enables Kubernetes to understand and act intelligently on volume accessibility information (such as the “zone” a cloud volume is provisioned in, the “rack” that a SAN array is accessible from, and so on).

The [dynamic maximum volume count](https://github.com/kubernetes/features/issues/554) feature was also moved to beta. This enables a volume plugin to specify the maximum number of a given volume type per node as a function of the node characteristics (for example, a larger limit for larger nodes, a smaller limit for smaller nodes).

SIG Storage also worked on a number of [Container Storage Interface (CSI) features](https://github.com/kubernetes/features/issues/178) this quarter in anticipation of moving support for CSI from beta to GA in the next Kubernetes release. This includes graduating the dependent “mount namespace propagation” feature to GA, moving the Kubelet plugin registration mechanism to beta, adding alpha support for a new CSI driver registry as well as for topology, and adding a number of alpha features to support the use of CSI for “local ephemeral volumes” (that is, volumes that exist for the lifecycle of a pod and contain some injected information, like a token or secret).

With Kubernetes v1.12, SIG Storage also introduced alpha support for [volume snapshotting](https://github.com/kubernetes/features/issues/177). This feature introduces the ability to create/delete volume snapshots and create new volumes from a snapshot using the Kubernetes API.

### SIG-vmware

SIG-VMware development was primarily focused on fixing bugs for the in-tree cloud provider, starting the development of the external cloud provider and taking ownership of the cluster-api provider for vSphere.

In-tree cloud provider bug fixes and improvements included:
- Adding initial Zones support to the provider using vSphere Tags
- Improving the testing harness for the cloud provider by introducing vcsim for automated testing
- Fixing a bug that was preventing updates from 1.10 to 1.11

The external cloud provider was established and reached feature parity with in-tree, and we expect to stabilize it and have it as preferred deployment model by 1.13. We are also getting started on externalizing the vSphere volume functionalities in a CSI plugin to fully reproduce the current in-tree storage functionality.

The Cluster API effort is currently undergoing a complete rehaul of the existing codebase, moving off Terraform and into using govmomi directly.

### SIG-windows

SIG Windows focused on stability and reliability of our existing feature set. We primarily fixed bugs as we march towards a near future stable release.

## Action Required

- etcd2 as a backend is deprecated and support will be removed in Kubernetes 1.13.
- The --storage-versions flag of kube-apiserver is now deprecated. This flag should be omitted to ensure the default storage versions are used. Otherwise the cluster is not safe to upgrade to a version newer than 1.12. This flag will be removed in 1.13. ([#68080](https://github.com/kubernetes/kubernetes/pull/68080), [@caesarxuchao](https://github.com/caesarxuchao)) Courtesy of SIG API Machinery
- Volume dynamic provisioning scheduling has been moved to beta, which means that the DynamicProvisioningScheduling alpha feature gate has been removed but the VolumeScheduling beta feature gate is still required for this feature. ([#67432](https://github.com/kubernetes/kubernetes/pull/67432), [@lichuqiang](https://github.com/lichuqiang)) Courtesy of SIG Apps, SIG Architecture, SIG Storage, and SIG Testing
- The API server and client-go libraries have been fixed to support additional non-alpha-numeric characters in UserInfo "extra" data keys. Both should be updated in order to properly support extra data containing "/" characters or other characters disallowed in HTTP headers. ([#65799](https://github.com/kubernetes/kubernetes/pull/65799), [@dekkagaijin](https://github.com/dekkagaijin)) Courtesy of SIG Auth
- The `NodeConfiguration` kind in the kubeadm v1alpha2 API has been renamed `JoinConfiguration` in v1alpha3 ([#65951](https://github.com/kubernetes/kubernetes/pull/65951), [@luxas](https://github.com/luxas)) Courtesy of SIG Cluster Lifecycle
- The `MasterConfiguration` kind in the kubeadm v1alpha2 API has been renamed `InitConfiguration` in v1alpha3 ([#65945](https://github.com/kubernetes/kubernetes/pull/65945), [@luxas](https://github.com/luxas)) Courtesy of SIG Cluster Lifecycle
- The formerly publicly-available cAdvisor web UI that the kubelet started using `--cadvisor-port` has been entirely removed in 1.12. The recommended way to run cAdvisor if you still need it, is via a DaemonSet. ([#65707](https://github.com/kubernetes/kubernetes/pull/65707), [@dims](https://github.com/dims))
- Cluster Autoscaler version has been updated to 1.3.1-beta.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.1-beta.1 ([#65857](https://github.com/kubernetes/kubernetes/pull/65857), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska)) Courtesy of SIG Autoscaling
- kubeadm: The `v1alpha1` config API has been removed. ([#65628](https://github.com/kubernetes/kubernetes/pull/65628), [@luxas](https://github.com/luxas)) Courtesy of SIG Cluster Lifecycle
- kube-apiserver: When using `--enable-admission-plugins` the `Priority` admission plugin is now enabled by default (matching changes in 1.11.1+). If using `--admission-control` to fully specify the set of admission plugins, it is now necessary to add the `Priority` admission plugin for the PodPriority feature to work properly. ([#65739](https://github.com/kubernetes/kubernetes/pull/65739), [@liggitt](https://github.com/liggitt)) Courtesy of SIG Scheduling
- The `system-node-critical` and `system-cluster-critical` priority classes are now limited to the `kube-system` namespace by the `PodPriority` admission plugin (matching changes in 1.11.1+). ([#65593](https://github.com/kubernetes/kubernetes/pull/65593), [@bsalamat](https://github.com/bsalamat)) Courtesy of SIG Scheduling
- kubeadm: Control plane images (etcd, kube-apiserver, kube-proxy, etc.) no longer use arch suffixes. Arch suffixes are kept for kube-dns only. ([#66960](https://github.com/kubernetes/kubernetes/pull/66960), 
[@rosti](https://github.com/rosti)) Courtesy of SIG Cluster Lifecycle, SIG Release, and SIG Testing
- kubeadm - Feature-gates HighAvailability, SelfHosting, CertsInSecrets are now deprecated and can no longer be used for new clusters. Cluster updates using above feature-gates flag is not supported. ([#67786](https://github.com/kubernetes/kubernetes/pull/67786), [@fabriziopandini](https://github.com/fabriziopandini)) Courtesy of SIG Cluster Lifecycle
- 'KubeSchedulerConfiguration' which used to be under GroupVersion 'componentconfig/v1alpha1',
is now under 'kubescheduler.config.k8s.io/v1alpha1'.  ([#66916](https://github.com/kubernetes/kubernetes/pull/66916), [@dixudx](https://github.com/dixudx)) Courtesy of SIG Cluster Lifecycle, SIG Scheduling, and SIG Testing
- The flag `--skip-preflight-checks` of kubeadm has been removed. Please use `--ignore-preflight-errors` instead. ([#62727](https://github.com/kubernetes/kubernetes/pull/62727), [@xiangpengzhao](https://github.com/xiangpengzhao))
- If Openstack LoadBalancer is not defined in cloud config, the loadbalancer will no longer beis not initialized. any more in openstack. All setups must have some setting under that section for the OpenStack provider. ([#65781](https://github.com/kubernetes/kubernetes/pull/65781), [@zetaab](https://github.com/zetaab))

## Deprecations and removals

- Kubeadm: The Alpha feature-gates HighAvailability, SelfHosting, CertsInSecrets are now deprecated, and will be removed in k8s v1.13.0.
- The cloudstack and ovirt controllers have been deprecated and will be removed in a future version. ([#68199](https://github.com/kubernetes/kubernetes/pull/68199), [@dims](https://github.com/dims))
- All kubectl run generators have been deprecated except for run-pod/v1. This is part of a move to make `kubectl run` simpler, enabling it create only pods; if additional resources are needed, you should use `kubectl create` instead. ([#68132](https://github.com/kubernetes/kubernetes/pull/68132), [@soltysh](https://github.com/soltysh))
- The deprecated --interactive flag has been removed from kubectl logs. ([#65420](https://github.com/kubernetes/kubernetes/pull/65420), [@jsoref](https://github.com/jsoref))
-The deprecated shorthand flag `-c` has been removed from `kubectl version (--client)`. ([#66817](https://github.com/kubernetes/kubernetes/pull/66817), [@charrywanganthony](https://github.com/charrywanganthony))
- The `--pod` flag (`-p` shorthand) of the kubectl exec command has been marked as deprecated, and will be removed in a future version. This flag is currently optional. ([#66558](https://github.com/kubernetes/kubernetes/pull/66558), [@quasoft](https://github.com/quasoft))
- kubectl: `--use-openapi-print-columns` has been deprecated in favor of `--server-print`, and will be removed in a future version. ([#65601](https://github.com/kubernetes/kubernetes/pull/65601), [@liggitt](https://github.com/liggitt))
- The watch API endpoints prefixed with `/watch` are deprecated and will be removed in a future release. These standard method for watching resources (supported since v1.0) is to use the list API endpoints with a `?watch=true` parameter. All client-go clients have used the parameter method since v1.6.0. ([#65147](https://github.com/kubernetes/kubernetes/pull/65147), [@liggitt](https://github.com/liggitt))
- Using the Horizontal Pod Autoscaler with metrics from Heapster is now deprecated and will be disabled in a future version. ([#68089](https://github.com/kubernetes/kubernetes/pull/68089), [@DirectXMan12](https://github.com/DirectXMan12))
- The watch API endpoints prefixed with `/watch` are deprecated and will be removed in a future release. These standard method for watching resources (supported since v1.0) is to use the list API endpoints with a `?watch=true` parameter. All client-go clients have used the parameter method since v1.6.0. ([#65147](https://github.com/kubernetes/kubernetes/pull/65147), [@liggitt](https://github.com/liggitt))

## New Features

- Kubernetes now registers volume topology information reported by a node-level Container Storage Interface (CSI) driver. This enables Kubernetes support of CSI topology mechanisms. ([#67684](https://github.com/kubernetes/kubernetes/pull/67684), [@verult](https://github.com/verult)) Courtesy of SIG API Machinery, SIG Node, SIG Storage, and SIG Testing
- Addon-manager has been bumped to v8.7 ([#68299](https://github.com/kubernetes/kubernetes/pull/68299), [@MrHohn](https://github.com/MrHohn)) Courtesy of SIG Cluster Lifecycle, and SIG Testing
- The CSI volume plugin no longer needs an external attacher for non-attachable CSI volumes. ([#67955](https://github.com/kubernetes/kubernetes/pull/67955), [@jsafrane](https://github.com/jsafrane)) Courtesy of SIG API Machinery, SIG Node, SIG Storage, and SIG Testing
- KubeletPluginsWatcher feature graduated to beta. ([#68200](https://github.com/kubernetes/kubernetes/pull/68200), [@RenaudWasTaken](https://github.com/RenaudWasTaken)) Courtesy of SIG Node, SIG Storage, and SIG Testing
- A TTL mechanism has been added to clean up Jobs after they finish. ([#66840](https://github.com/kubernetes/kubernetes/pull/66840), [@janetkuo](https://github.com/janetkuo)) Courtesy of SIG API Machinery, SIG Apps, SIG Architecture, and SIG Testing
- The scheduler is now optimized to throttle computational tasks involved with node selection. ([#67555](https://github.com/kubernetes/kubernetes/pull/67555), [@wgliang](https://github.com/wgliang)) Courtesy of SIG API Machinery, and SIG Scheduling
- The  performance of Pod affinity/anti-affinity in the scheduler has been improved. ([#67788](https://github.com/kubernetes/kubernetes/pull/67788), [@ahmad-diaa](https://github.com/ahmad-diaa)) Courtesy of SIG Scalability, and SIG Scheduling
- A kubelet parameter and config option has been added to change the CFS quota period from the default 100ms to some other value between 1µs and 1s. This was done to improve response latencies for workloads running in clusters with guaranteed and burstable QoS classes. ([#63437](https://github.com/kubernetes/kubernetes/pull/63437), [@szuecs](https://github.com/szuecs)) Courtesy of SIG API Machinery, SIG Apps, SIG Architecture, SIG CLI,, SIG Node, and SIG Scheduling
- Secure serving on port 10258 to cloud-controller-manager (configurable via `--secure-port`) is now enabled. Delegated authentication and authorization are to be configured using the same flags as for aggregated API servers. Without configuration, the secure port will only allow access to `/healthz`. ([#67069](https://github.com/kubernetes/kubernetes/pull/67069), [@sttts](https://github.com/sttts)) Courtesy of SIG Auth, and SIG Cloud Provider
- The commands `kubeadm alpha phases renew <cert-name>` have been added. ([#67910](https://github.com/kubernetes/kubernetes/pull/67910), [@liztio](https://github.com/liztio)) Courtesy of SIG API Machinery, and SIG Cluster Lifecycle
- ProcMount has been added to SecurityContext and AllowedProcMounts has been added to PodSecurityPolicy to allow paths in the container's /proc to not be masked. ([#64283](https://github.com/kubernetes/kubernetes/pull/64283), [@jessfraz](https://github.com/jessfraz)) Courtesy of SIG API Machinery, SIG Apps, SIG Architecture, and SIG Node
- Secure serving on port 10257 to kube-controller-manager (configurable via `--secure-port`) is now enabled. Delegated authentication and authorization are to be configured using the same flags as for aggregated API servers. Without configuration, the secure port will only allow access to `/healthz`. ([#64149](https://github.com/kubernetes/kubernetes/pull/64149), [@sttts](https://github.com/sttts)) Courtesy of SIG API Machinery, SIG Auth, SIG Cloud Provider, SIG Scheduling, and SIG Testing
- Azure cloud provider now supports unmanaged nodes (such as on-prem) that are labeled with `kubernetes.azure.com/managed=false` and `alpha.service-controller.kubernetes.io/exclude-balancer=true` ([#67984](https://github.com/kubernetes/kubernetes/pull/67984), [@feiskyer](https://github.com/feiskyer)) Courtesy of SIG Azure, and SIG Cloud Provider
- SCTP is now supported as an additional protocol (alpha) alongside TCP and UDP in Pod, Service, Endpoint, and NetworkPolicy.  ([#64973](https://github.com/kubernetes/kubernetes/pull/64973), [@janosi](https://github.com/janosi)) Courtesy of SIG API Machinery, SIG Apps, SIG Architecture, SIG CLI, SIG Cloud Provider, SIG Cluster Lifecycle, SIG Network, SIG Node, and SIG Scheduling
- Autoscaling/v2beta2 and custom_metrics/v1beta2 have been introduced, which implement metric selectors for Object and Pods metrics, as well as allowing AverageValue targets on Objects, similar to External metrics. ([#64097](https://github.com/kubernetes/kubernetes/pull/64097), [@damemi](https://github.com/damemi)) Courtesy of SIG API Machinery, SIG Architecture, SIG Autoscaling, SIG CLI, and SIG Testing
- kubelet: Users can now enable the alpha NodeLease feature gate to have the Kubelet create and periodically renew a Lease in the kube-node-lease namespace. The lease duration defaults to 40s, and can be configured via the kubelet.config.k8s.io/v1beta1.KubeletConfiguration's NodeLeaseDurationSeconds field. ([#66257](https://github.com/kubernetes/kubernetes/pull/66257), [@mtaufen](https://github.com/mtaufen)) Courtesy of SIG API Machinery, SIG Apps, SIG Architecture, SIG Cluster Lifecycle, SIG Node, and SIG Testing
- PodReadinessGate is now turned on by default. ([#67406](https://github.com/kubernetes/kubernetes/pull/67406), [@freehan](https://github.com/freehan)) Courtesy of SIG Node
- Azure cloud provider now supports cross resource group nodes that are labeled with `kubernetes.azure.com/resource-group=<rg-name>` and `alpha.service-controller.kubernetes.io/exclude-balancer=true` ([#67604](https://github.com/kubernetes/kubernetes/pull/67604), [@feiskyer](https://github.com/feiskyer)) Courtesy of SIG Azure, SIG Cloud Provider, and SIG Storage
- Annotations are now supported for remote admission webhooks. ([#58679](https://github.com/kubernetes/kubernetes/pull/58679), [@CaoShuFeng](https://github.com/CaoShuFeng)) Courtesy of SIG API Machinery, and SIG Auth
- The scheduler now scores fewer than all nodes in every scheduling cycle. This can improve performance of the scheduler in large clusters. ([#66733](https://github.com/kubernetes/kubernetes/pull/66733), [@bsalamat](https://github.com/bsalamat)) Courtesy of SIG Scheduling
- Node affinity for Azure unzoned managed disks has been added. ([#67229](https://github.com/kubernetes/kubernetes/pull/67229), [@feiskyer](https://github.com/feiskyer)) Courtesy of SIG Azure
- The Attacher/Detacher interfaces for local storage have been refactored  ([#66884](https://github.com/kubernetes/kubernetes/pull/66884), [@NickrenREN](https://github.com/NickrenREN)) Courtesy of SIG Storage
- DynamicProvisioningScheduling and VolumeScheduling is now supported for Azure managed disks. Feature gates DynamicProvisioningScheduling and VolumeScheduling should be enabled before using this feature. ([#67121](https://github.com/kubernetes/kubernetes/pull/67121), [@feiskyer](https://github.com/feiskyer)) Courtesy of SIG Azure, and SIG Storage
- The audit.k8s.io api group has been upgraded from v1beta1 to v1. ([#65891](https://github.com/kubernetes/kubernetes/pull/65891), [@CaoShuFeng](https://github.com/CaoShuFeng)) Courtesy of SIG API Machinery
- The quota admission configuration API graduated to v1beta1. ([#66156](https://github.com/kubernetes/kubernetes/pull/66156), [@vikaschoudhary16](https://github.com/vikaschoudhary16)) Courtesy of SIG Node, and SIG Scheduling
- Kube-apiserver --help flag help is now printed in sections. ([#64517](https://github.com/kubernetes/kubernetes/pull/64517), [@sttts](https://github.com/sttts))
- Azure managed disks now support availability zones and new parameters `zoned`, `zone` and `zones` are added for AzureDisk storage class. ([#66553](https://github.com/kubernetes/kubernetes/pull/66553), [@feiskyer](https://github.com/feiskyer)) Courtesy of SIG Azure
- Kubectl create job command has been added. ([#60316](https://github.com/kubernetes/kubernetes/pull/60316), [@soltysh](https://github.com/soltysh)) Courtesy of SIG CLI
- Kubelet serving certificate bootstrapping and rotation has been promoted to beta status. ([#66726](https://github.com/kubernetes/kubernetes/pull/66726), [@liggitt](https://github.com/liggitt)) Courtesy of SIG Auth, and SIG Node
- Azure nodes with availability zone will now have label `failure-domain.beta.kubernetes.io/zone=<region>-<zoneID>`. ([#66242](https://github.com/kubernetes/kubernetes/pull/66242), [@feiskyer](https://github.com/feiskyer)) Courtesy of SIG Azure
- kubeadm: Default component configs are now printable via kubeadm config print-default ([#66074](https://github.com/kubernetes/kubernetes/pull/66074), [@rosti](https://github.com/rosti)) Courtesy of SIG Cluster Lifecycle
- Mount propagation has been promoted to GA. The `MountPropagation` feature gate is deprecated and will be removed in 1.13. ([#67255](https://github.com/kubernetes/kubernetes/pull/67255), [@bertinatto](https://github.com/bertinatto)) Courtesy of SIG Apps, SIG Architecture, SIG Node, and SIG Storage
- Ubuntu 18.04 (Bionic) series has been added to Juju charms ([#65644](https://github.com/kubernetes/kubernetes/pull/65644), [@tvansteenburgh](https://github.com/tvansteenburgh))
- kubeadm: The kubeadm configuration now supports the definition of more than one control plane instances with their own APIEndpoint. The APIEndpoint for the "bootstrap" control plane instance should be defined using `InitConfiguration.APIEndpoint`, while the APIEndpoints for additional control plane instances should be added using `JoinConfiguration.APIEndpoint`. ([#67832](https://github.com/kubernetes/kubernetes/pull/67832), [@fabriziopandini](https://github.com/fabriziopandini))
- Add new `--server-dry-run` flag to `kubectl apply` so that the request will be sent to the server with the dry-run flag (alpha), which means that changes won't be persisted. ([#68069](https://github.com/kubernetes/kubernetes/pull/68069), [@apelisse](https://github.com/apelisse))
- Introduce CSI Cluster Registration mechanism to ease CSI plugin discovery and allow CSI drivers to customize Kubernetes' interaction with them. ([#67803](https://github.com/kubernetes/kubernetes/pull/67803), [@saad-ali](https://github.com/saad-ali))
- The PodShareProcessNamespace feature to configure PID namespace sharing within a pod has been promoted to beta. ([#66507](https://github.com/kubernetes/kubernetes/pull/66507), [@verb](https://github.com/verb))

## API Changes

- kubeadm now supports the phase command "alpha phase kubelet config annotate-cri". ([#68449](https://github.com/kubernetes/kubernetes/pull/68449), [@fabriziopandini](https://github.com/fabriziopandini))
- kubeadm: --cri-socket now defaults to tcp://localhost:2375 when running on Windows. ([#67447](https://github.com/kubernetes/kubernetes/pull/67447), [@benmoss](https://github.com/benmoss))
- kubeadm now includes a new EXPERIMENTAL `--rootfs`, which (if specified) causes kubeadm to chroot before performing any file operations.  This is expected to be useful when setting up kubernetes on a different filesystem, such as invoking kubeadm from docker. ([#54935](https://github.com/kubernetes/kubernetes/pull/54935), [@anguslees](https://github.com/anguslees))
- The  command line option  --cri-socket-path of the kubeadm subcommand "kubeadm config images pull" has been renamed to --cri-socket to be consistent with the rest of kubeadm subcommands. 
- kubeadm: The ControlPlaneEndpoint was moved from the API config struct to ClusterConfiguration ([#67830](https://github.com/kubernetes/kubernetes/pull/67830), [@fabriziopandini](https://github.com/fabriziopandini))
- kubeadm: InitConfiguration now consists of two structs: InitConfiguration and ClusterConfiguration ([#67441](https://github.com/kubernetes/kubernetes/pull/67441), [@rosti](https://github.com/rosti))
- The RuntimeClass API has been added. This feature is in alpha, and the RuntimeClass feature gate must be enabled in order to use it. The RuntimeClass API resource defines different classes of runtimes that may be used to run containers in the cluster. Pods can select a RuntimeClass to use via the RuntimeClassName field. ([#67737](https://github.com/kubernetes/kubernetes/pull/67737), [@tallclair](https://github.com/tallclair))
- To address the possibility of dry-run requests overwhelming admission webhooks that rely on side effects and a reconciliation mechanism, a new field is being added to `admissionregistration.k8s.io/v1beta1.ValidatingWebhookConfiguration` and `admissionregistration.k8s.io/v1beta1.MutatingWebhookConfiguration` so that webhooks can explicitly register as having dry-run support. If a dry-run request is made on a resource that triggers a non dry-run supporting webhook, the request will be completely rejected, with "400: Bad Request". Additionally, a new field is being added to the `admission.k8s.io/v1beta1.AdmissionReview` API object, exposing to webhooks whether or not the request being reviewed is a dry-run. ([#66936](https://github.com/kubernetes/kubernetes/pull/66936), [@jennybuckley](https://github.com/jennybuckley))
- CRI now supports a "runtime_handler" field for RunPodSandboxRequest, used for selecting the runtime configuration to run the sandbox with (alpha feature). ([#67518](https://github.com/kubernetes/kubernetes/pull/67518), [@tallclair](https://github.com/tallclair))
- More fields are allowed at the root of the CRD validation schema when the status subresource is enabled. ([#65357](https://github.com/kubernetes/kubernetes/pull/65357), [@nikhita](https://github.com/nikhita))
- The --docker-disable-shared-pid kubelet flag has been removed. PID namespace sharing can instead be enable per-pod using the ShareProcessNamespace option. ([#66506](https://github.com/kubernetes/kubernetes/pull/66506), [@verb](https://github.com/verb))
- Added the --dns-loop-detect option to dnsmasq, which is run by kube-dns. ([#67302](https://github.com/kubernetes/kubernetes/pull/67302), [@dixudx](https://github.com/dixudx))
- Kubernetes now supports extra `--prune-whitelist` resources in kube-addon-manager. ([#67743](https://github.com/kubernetes/kubernetes/pull/67743), [@Random-Liu](https://github.com/Random-Liu))
- Graduate Resource Quota ScopeSelectors to beta, and enable it by default. ([#67077](https://github.com/kubernetes/kubernetes/pull/67077), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
- The OpenAPI spec and documentation now reflect the 202 Accepted response path for delete requests. Note that this change in the openapi spec may affect some clients that depend on the error paths.  ([#63418](https://github.com/kubernetes/kubernetes/pull/63418), [@roycaihw](https://github.com/roycaihw))
- The alpha `Initializers` admission plugin is no longer enabled by default. This matches the off-by-default behavior of the alpha API which drives initializer behavior. ([#66039](https://github.com/kubernetes/kubernetes/pull/66039), [@liggitt](https://github.com/liggitt))
- Adding validation to kube-scheduler at the API level ([#66799](https://github.com/kubernetes/kubernetes/pull/66799), [@noqcks](https://github.com/noqcks))
- `DisruptedPods` field in `PodDisruptionBudget` is optional instead of required. ([#63757](https://github.com/kubernetes/kubernetes/pull/63757), [@nak3](https://github.com/nak3))

## Other Notable Changes

### SIG API Machinery

- `kubectl get apiservice` now shows the target service and whether the service is available ([#67747](https://github.com/kubernetes/kubernetes/pull/67747), [@smarterclayton](https://github.com/smarterclayton))
- Apiserver panics will now be returned as 500 errors rather than terminating the apiserver process. ([#68001](https://github.com/kubernetes/kubernetes/pull/68001), [@sttts](https://github.com/sttts))
- API paging is now enabled for custom resource definitions, custom resources and APIService objects. ([#67861](https://github.com/kubernetes/kubernetes/pull/67861), [@liggitt](https://github.com/liggitt))
- To address the possibility dry-run requests overwhelming admission webhooks that rely on side effects and a reconciliation mechanism, a new field is being added to admissionregistration.k8s.io/v1beta1.ValidatingWebhookConfiguration and admissionregistration.k8s.io/v1beta1.MutatingWebhookConfiguration so that webhooks can explicitly register as having dry-run support. If a dry-run request is made on a resource that triggers a non dry-run supporting webhook, the request will be completely rejected, with "400: Bad Request". Additionally, a new field is being added to the admission.k8s.io/v1beta1.AdmissionReview API object, exposing to webhooks whether or not the request being reviewed is a dry-run. ([#66936](https://github.com/kubernetes/kubernetes/pull/66936), [@jennybuckley](https://github.com/jennybuckley))
- kube-apiserver now includes all registered API groups in discovery, including registered extension API group/versions for unavailable extension API servers. ([#66932](https://github.com/kubernetes/kubernetes/pull/66932), [@nilebox](https://github.com/nilebox))
- kube-apiserver: setting a `dryRun` query parameter on a CONNECT request will now cause the request to be rejected, consistent with behavior of other mutating API requests. Examples of CONNECT APIs are the `nodes/proxy`, `services/proxy`, `pods/proxy`, `pods/exec`, and `pods/attach` subresources. Note that this prevents sending a `dryRun` parameter to backends via `{nodes,services,pods}/proxy` subresources. ([#66083](https://github.com/kubernetes/kubernetes/pull/66083), [@jennybuckley](https://github.com/jennybuckley))
- In clusters where the DryRun feature is enabled, dry-run requests will go through the normal admission chain. Because of this, ImagePolicyWebhook authors should especially make sure that their webhooks do not rely on side effects. ([#66391](https://github.com/kubernetes/kubernetes/pull/66391), [@jennybuckley](https://github.com/jennybuckley))
- Added etcd_object_count metrics for CustomResources. ([#65983](https://github.com/kubernetes/kubernetes/pull/65983), [@sttts](https://github.com/sttts))
- The OpenAPI version field will now be properly autopopulated without needing other OpenAPI fields present in generic API server code. ([#66411](https://github.com/kubernetes/kubernetes/pull/66411), [@DirectXMan12](https://github.com/DirectXMan12))
- TLS timeouts have been extended to work around slow arm64 math/big functions. ([#66264](https://github.com/kubernetes/kubernetes/pull/66264), [@joejulian](https://github.com/joejulian))
- Kubernetes now checks CREATE admission for create-on-update requests instead of UPDATE admission. ([#65572](https://github.com/kubernetes/kubernetes/pull/65572), [@yue9944882](https://github.com/yue9944882))
- kube- and cloud-controller-manager can now listen on ports up to 65535 rather than 32768, solving problems with operating systems that request these higher ports.. ([#65860](https://github.com/kubernetes/kubernetes/pull/65860), [@sttts](https://github.com/sttts))
- LimitRange and Endpoints resources can be created via an update API call if the object does not already exist. When this occurs, an authorization check is now made to ensure the user making the API call is authorized to create the object. In previous releases, only an update authorization check was performed. ([#65150](https://github.com/kubernetes/kubernetes/pull/65150), [@jennybuckley](https://github.com/jennybuckley))
- More fields are allowed at the root of the CRD validation schema when the status subresource is enabled. ([#65357](https://github.com/kubernetes/kubernetes/pull/65357), [@nikhita](https://github.com/nikhita))
- api-machinery utility functions `SetTransportDefaults` and `DialerFor` once again respect custom Dial functions set on transports ([#65547](https://github.com/kubernetes/kubernetes/pull/65547), [@liggitt](https://github.com/liggitt))
- AdvancedAuditing has been promoted to GA, replacing the previous (legacy) audit logging mechanisms. ([#65862](https://github.com/kubernetes/kubernetes/pull/65862), [@loburm](https://github.com/loburm))
- Added --authorization-always-allow-paths to components doing delegated authorization to exclude certain HTTP paths like /healthz from authorization. ([#67543](https://github.com/kubernetes/kubernetes/pull/67543), [@sttts](https://github.com/sttts))
- Allow ImageReview backend to return annotations to be added to the created pod. ([#64597](https://github.com/kubernetes/kubernetes/pull/64597), [@wteiken](https://github.com/wteiken))
- Upon receiving a LIST request with an expired continue token, the apiserver now returns a continue token together with the 410 "the from parameter is too old" error. If the client does not care about getting a list from a consistent snapshot, the client can use this token to continue listing from the next key, but the returned chunk will be from the latest snapshot. ([#67284](https://github.com/kubernetes/kubernetes/pull/67284), [@caesarxuchao](https://github.com/caesarxuchao))

### SIG Apps

- The service controller will now retry creating the load balancer when `persistUpdate` fails due to conflict. ([#68087](https://github.com/kubernetes/kubernetes/pull/68087), [@grayluck](https://github.com/grayluck))
- The latent controller caches no longer cause repeating deletion messages for deleted pods. ([#67826](https://github.com/kubernetes/kubernetes/pull/67826), [@deads2k](https://github.com/deads2k))

### SIG Auth

- TokenRequest and TokenRequestProjection are now beta features. To enable these feature, the API server needs to be started with the `--service-account-issuer`, `--service-account-signing-key-file`, and `--service-account-api-audiences` flags. 
([#67349](https://github.com/kubernetes/kubernetes/pull/67349), [@mikedanese](https://github.com/mikedanese))
- The admin RBAC role now aggregates edit and view.  The edit RBAC role now aggregates view.  ([#66684](https://github.com/kubernetes/kubernetes/pull/66684), [@deads2k](https://github.com/deads2k))
- UserInfo derived from service account tokens created from the TokenRequest API now include the pod name and UID in the Extra field. ([#61858](https://github.com/kubernetes/kubernetes/pull/61858), [@mikedanese](https://github.com/mikedanese))
- The extension API server can now dynamically discover the requestheader CA certificate when the core API server doesn't use certificate based authentication for it's clients. ([#66394](https://github.com/kubernetes/kubernetes/pull/66394), [@rtripat](https://github.com/rtripat))

### SIG Autoscaling

- Horizontal Pod Autoscaler default update interval has been increased from 30s to 15s, improving HPA reaction time for metric changes. ([#68021](https://github.com/kubernetes/kubernetes/pull/68021), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
- To avoid soft-deleted pods incorrectly affecting scale up replica count calculations, the HPA controller will stop counting soft-deleted pods for scaling purposes. ([#67067](https://github.com/kubernetes/kubernetes/pull/67067), [@moonek](https://github.com/moonek))
- HPA reaction to metric changes has been spend up by removing the scale up forbidden window. ([#66615](https://github.com/kubernetes/kubernetes/pull/66615), [@jbartosik](https://github.com/jbartosik))

### SIG AWS

- AWS LoadBalancer security group ICMP rules now match the documentation of  spec.loadBalancerSourceRanges ([#63572](https://github.com/kubernetes/kubernetes/pull/63572), [@haz-mat](https://github.com/haz-mat))
- The aws cloud provider now reports a `Hostname` address type for nodes based on the `local-hostname` metadata key. ([#67715](https://github.com/kubernetes/kubernetes/pull/67715), [@liggitt](https://github.com/liggitt))

### SIG Azure

- \API calls for Azure instance metadata have been reduced to help avoid "too many requests" errors.. ([#67478](https://github.com/kubernetes/kubernetes/pull/67478), [@feiskyer](https://github.com/feiskyer))
- Azure Go SDK has been upgraded to v19.0.0 and VirtualMachineScaleSetVM now supports availability zones. ([#66648](https://github.com/kubernetes/kubernetes/pull/66648), [@feiskyer](https://github.com/feiskyer))
- User Assigned MSI (https://docs.microsoft.com/en-us/azure/active-directory/managed-service-identity/overview), which provides for managed identities, is now suppored for Kubernetes clusters on Azure. ([#66180](https://github.com/kubernetes/kubernetes/pull/66180), [@kkmsft](https://github.com/kkmsft))
- The Azure load balancer idle connection timeout for services is now configurable.([#66045](https://github.com/kubernetes/kubernetes/pull/6605), [@cpuguy83](https://github.com/cpuguy83))
- When provisioning workloads, Kubernetes will now skip nodes that have a primary NIC in a 'Failed' provisioningState. ([#65412](https://github.com/kubernetes/kubernetes/pull/65412), [@yastij](https://github.com/yastij))
- The NodeShutdown taint is now supported for Azure. ([#68033](https://github.com/kubernetes/kubernetes/pull/68033), [@yastij](https://github.com/yastij))

### SIG CLI

- Added a sample-cli-plugin staging repository and cli-runtime staging repository to help showcase the new kubectl plugins mechanism. ([#67938](https://github.com/kubernetes/kubernetes/pull/67938), [#67658](https://github.com/kubernetes/kubernetes/pull/67658), [@soltysh](https://github.com/soltysh))
- The plugin mechanism functionality now closely follows the git plugin design ([#66876](https://github.com/kubernetes/kubernetes/pull/66876), [@juanvallejo](https://github.com/juanvallejo))
- kubectl patch now respects --local ([#67399](https://github.com/kubernetes/kubernetes/pull/67399), [@deads2k](https://github.com/deads2k))
- kubectl: When an object can't be updated and must be deleted by force, kubectl will now recreating resources for immutable fields.([#66602](https://github.com/kubernetes/kubernetes/pull/66602), [@dixudx](https://github.com/dixudx))
- `kubectl create {clusterrole,role}`'s `--resources` flag now supports asterisk to specify all resources. ([#62945](https://github.com/kubernetes/kubernetes/pull/62945), [@nak3](https://github.com/nak3))
- kubectl: the wait command now prints an error message and exits with the code 1, if there is no resources matching selectors ([#66692](https://github.com/kubernetes/kubernetes/pull/66692), [@m1kola](https://github.com/m1kola))
- Kubectl now handles newlines for `command`, `args`, `env`, and `annotations` in `kubectl describe` wrapping. ([#66841](https://github.com/kubernetes/kubernetes/pull/66841), [@smarterclayton](https://github.com/smarterclayton))
- The `kubectl patch` command no longer exits with exit code 1 when a redundant patch results in a no-op ([#66725](https://github.com/kubernetes/kubernetes/pull/66725), [@juanvallejo](https://github.com/juanvallejo))
- The output of `kubectl get events` has been improved to prioritize showing the message, and to move some fields to `-o wide`. ([#66643](https://github.com/kubernetes/kubernetes/pull/66643), [@smarterclayton](https://github.com/smarterclayton))
- `kubectl config set-context` can now set attributes of the current context, such as the current namespace, by passing `--current` instead of a specific context name ([#66140](https://github.com/kubernetes/kubernetes/pull/66140), [@liggitt](https://github.com/liggitt))
- "kubectl delete" no longer waits for dependent objects to be deleted when removing parent resources ([#65908](https://github.com/kubernetes/kubernetes/pull/65908), [@juanvallejo](https://github.com/juanvallejo))
- A new flag, `--keepalive`, has been introduced, for kubectl proxy to allow setting keep-alive period for long-running request. ([#63793](https://github.com/kubernetes/kubernetes/pull/63793), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
- kubectl: fixed a regression with --use-openapi-print-columns that would not print object contents ([#65600](https://github.com/kubernetes/kubernetes/pull/65600), [@liggitt](https://github.com/liggitt))
- The display of jobs in `kubectl get` and `kubectl describe` has been improved to emphasize progress and duration. ([#65463](https://github.com/kubernetes/kubernetes/pull/65463), [@smarterclayton](https://github.com/smarterclayton))
- CSI volume attributes have been added to kubectl describe pv`. ([#65074](https://github.com/kubernetes/kubernetes/pull/65074), [@wgliang](https://github.com/wgliang))
- Running `kubectl describe pvc` now shows which pods are mounted to the pvc being described with the `Mounted By` field ([#65837](https://github.com/kubernetes/kubernetes/pull/65837), [@clandry94](https://github.com/clandry94))
- `kubectl create secret tls` can now read certificate and key files from process substitution arguments ([#67713](https://github.com/kubernetes/kubernetes/pull/67713), [@liggitt](https://github.com/liggitt))
- `kubectl rollout status` now works for unlimited timeouts. ([#67817](https://github.com/kubernetes/kubernetes/pull/67817), [@tnozicka](https://github.com/tnozicka))

### SIG Cloud Provider

- The cloudstack cloud provider now reports a `Hostname` address type for nodes based on the `local-hostname` metadata key. ([#67719](https://github.com/kubernetes/kubernetes/pull/67719), [@liggitt](https://github.com/liggitt))
- The OpenStack cloud provider now reports a `Hostname` address type for nodes ([#67748](https://github.com/kubernetes/kubernetes/pull/67748), [@FengyunPan2](https://github.com/FengyunPan2))
- The vSphere cloud provider now suppoerts zones. ([#66795](https://github.com/kubernetes/kubernetes/pull/66795), [@jiatongw](https://github.com/jiatongw))

### SIG Cluster Lifecycle

- External CAs can now be used for kubeadm with only a certificate, as long as all required certificates already exist. ([#68296](https://github.com/kubernetes/kubernetes/pull/68296), [@liztio](https://github.com/liztio))
- kubeadm now works better when not connected to the Internet. In addition,  common kubeadm commands will now work without an available networking interface. ([#67397](https://github.com/kubernetes/kubernetes/pull/67397), [@neolit123](https://github.com/neolit123))
- Scrape frequency of metrics-server has been increased to 30s.([#68127](https://github.com/kubernetes/kubernetes/pull/68127), [@serathius](https://github.com/serathius))
- Kubernetes juju charms will now use CSI for ceph. ([#66523](https://github.com/kubernetes/kubernetes/pull/66523), [@hyperbolic2346](https://github.com/hyperbolic2346))
- kubeadm uses audit policy v1 instead of v1beta1 ([#67176](https://github.com/kubernetes/kubernetes/pull/67176), [@charrywanganthony](https://github.com/charrywanganthony))
- Kubeadm nodes will no longer be able to run with an empty or invalid hostname in /proc/sys/kernel/hostname ([#64815](https://github.com/kubernetes/kubernetes/pull/64815), [@dixudx](https://github.com/dixudx))
- kubeadm now can join the cluster with pre-existing client certificate if provided ([#66482](https://github.com/kubernetes/kubernetes/pull/66482), [@dixudx](https://github.com/dixudx))
([#66382](https://github.com/kubernetes/kubernetes/pull/66382), [@bart0sh](https://github.com/bart0sh)) 
- kubeadm will no longer hang indefinitely if there is no Internet connection and --kubernetes-version is not specified.([#65676](https://github.com/kubernetes/kubernetes/pull/65676), [@dkoshkin](https://github.com/dkoshkin))
- kubeadm: kube-proxy will now run on all nodes, and not just master nodes.([#65931](https://github.com/kubernetes/kubernetes/pull/65931), [@neolit123](https://github.com/neolit123))
- kubeadm now uses separate YAML documents for the kubelet and kube-proxy ComponentConfigs. ([#65787](https://github.com/kubernetes/kubernetes/pull/65787), [@luxas](https://github.com/luxas))
- kubeadm will now print required flags when running `kubeadm upgrade plan`.([#65802](https://github.com/kubernetes/kubernetes/pull/65802), [@xlgao-zju](https://github.com/xlgao-zju))
- Unix support for ZFS as a valid graph driver has been added for Docker, enabling users to use Kubeadm with ZFS. ([#65635](https://github.com/kubernetes/kubernetes/pull/65635), [@neolit123](https://github.com/neolit123))

### SIG GCP

- GCE: decrease cpu requests on master node, to allow more components to fit on one core machine. ([#67504](https://github.com/kubernetes/kubernetes/pull/67504), [@loburm](https://github.com/loburm))
- Kubernetes 1.12 includes a large number of metadata agent improvements, including expanding the metadata agent's access to all API groups and removing metadata agent config maps in favor of command line flags. It also includes improvements to the logging agent, such as multiple fixes and adjustments.
 ([#66485](https://github.com/kubernetes/kubernetes/pull/66485), [@bmoyles0117](https://github.com/bmoyles0117))
- cluster/gce: Kubernetes now generates consistent key sizes in config-default.sh using /dev/urandom instead of /dev/random   ([#67139](https://github.com/kubernetes/kubernetes/pull/67139), [@yogi-sagar](https://github.com/yogi-sagar))

### SIG Instrumentation

 The etcdv3 client can now be monitored by Prometheus. ([#64741](https://github.com/kubernetes/kubernetes/pull/64741), [@wgliang](https://github.com/wgliang))

### SIG Network

- The ip-masq-agent will now be scheduled in all nodes except master due to NoSchedule/NoExecute tolerations. ([#66260](https://github.com/kubernetes/kubernetes/pull/66260), [@tanshanshan](https://github.com/tanshanshan))
- The CoreDNS service can now be monitored by Prometheus. ([#65589](https://github.com/kubernetes/kubernetes/pull/65589), [@rajansandeep](https://github.com/rajansandeep))
- Traffic shaping is now supported for the CNI network driver. ([#63194](https://github.com/kubernetes/kubernetes/pull/63194), [@m1093782566](https://github.com/m1093782566))
- The dockershim now sets the "bandwidth" and "ipRanges" CNI capabilities (dynamic parameters). Plugin authors and administrators can now take advantage of this by updating their CNI configuration file. For more information, see the [CNI docs](https://github.com/containernetworking/cni/blob/master/CONVENTIONS.md#dynamic-plugin-specific-fields-capabilities--runtime-configuration) ([#64445](https://github.com/kubernetes/kubernetes/pull/64445), [@squeed](https://github.com/squeed))

### SIG Node

- RuntimeClass is a new API resource for defining different classes of runtimes that may be used to run containers in the cluster. Pods can select a RunitmeClass to use via the RuntimeClassName field. This feature is in alpha, and the RuntimeClass feature gate must be enabled in order to use it. ([#67737](https://github.com/kubernetes/kubernetes/pull/67737), [@tallclair](https://github.com/tallclair))
- Sped up kubelet start time by executing an immediate runtime and node status update when the Kubelet sees that it has a CIDR. ([#67031](https://github.com/kubernetes/kubernetes/pull/67031), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
- cpumanager will now rollback state if updateContainerCPUSet failed, indicating that the container start failed. This change will prevent CPU leaks. ([#67430](https://github.com/kubernetes/kubernetes/pull/67430), [@choury](https://github.com/choury))
- [CRI] RunPodSandboxRequest now has a runtime_handler field for selecting the runtime configuration to run the sandbox with. This feature is in alpha for 1.12.. ([#67518](https://github.com/kubernetes/kubernetes/pull/67518), [@tallclair](https://github.com/tallclair))
- If a container's requested device plugin resource hasn't registered after Kubelet restart, the container start will now fail.([#67145](https://github.com/kubernetes/kubernetes/pull/67145), [@jiayingz](https://github.com/jiayingz))
- Upgraded TaintNodesByCondition to beta. ([#62111](https://github.com/kubernetes/kubernetes/pull/62111), [@k82cn](https://github.com/k82cn))
- The PodShareProcessNamespace feature to configure PID namespace sharing within a pod has been promoted to beta. ([#66507](https://github.com/kubernetes/kubernetes/pull/66507), [@verb](https://github.com/verb))
- The CPU Manager will now validate the state of the node,  enabling Kubernetes to maintain the CPU topology even if resources change. ([#66718](https://github.com/kubernetes/kubernetes/pull/66718), [@ipuustin](https://github.com/ipuustin))
- Added support kubelet plugin watcher in device manager, as part of the new plugin system. ([#58755](https://github.com/kubernetes/kubernetes/pull/58755), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
- Expose docker registry config for addons used in Juju deployments ([#66092](https://github.com/kubernetes/kubernetes/pull/66092), [@kwmonroe](https://github.com/kwmonroe))
- `RunAsGroup` which has been broken since 1.10, now works. ([#65926](https://github.com/kubernetes/kubernetes/pull/65926), [@Random-Liu](https://github.com/Random-Liu))
- The systemd config files are now reloaded before kubelet starts, so changes can take effect([#65702](https://github.com/kubernetes/kubernetes/pull/65702), [@mborsz](https://github.com/mborsz))
- Hostnames are now converted to lowercase before being used for node lookups in the kubernetes-worker charm. ([#65487](https://github.com/kubernetes/kubernetes/pull/65487), [@dshcherb](https://github.com/dshcherb))
- kubelets that specify `--cloud-provider` now only report addresses in Node status as determined by the cloud provider (unless `--hostname-override` is used to force reporting of the specified hostname) ([#65594](https://github.com/kubernetes/kubernetes/pull/65594), [@liggitt](https://github.com/liggitt))
- Kubelet now exposes `/debug/flags/v` to allow dynamically setting glog logging level.  For example, to change glog level to 3, you only have to send a PUT request like `curl -X PUT http://127.0.0.1:8080/debug/flags/v -d "3"`. ([#64601](https://github.com/kubernetes/kubernetes/pull/64601), [@hzxuzhonghu](https://github.com/hzxuzhonghu))

### SIG OpenStack

- Openstack now supports the node shutdown taint. The taint is added when an instance is shutdown in openstack. ([#67982](https://github.com/kubernetes/kubernetes/pull/67982), [@zetaab](https://github.com/zetaab))

### SIG Scheduling

- The equivalence class cache has been redesigned to be a two level cache, resulting in a significant increase in scheduling throughput and performance. ([#65714](https://github.com/kubernetes/kubernetes/pull/65714), [@resouer](https://github.com/resouer))
- kube-scheduler can now listen on ports up to 65535, correcting a problem with certain operating systems that request ports greater than 32768. ([#65833](https://github.com/kubernetes/kubernetes/pull/65833), [@sttts](https://github.com/sttts))
- Performance of the anti-affinity predicate of the default scheduler has been improved. ([#66948](https://github.com/kubernetes/kubernetes/pull/66948), [@mohamed-mehany](https://github.com/mohamed-mehany))
- The unreachable taint gets applied to a node when it loses its network connection. ([#67734](https://github.com/kubernetes/kubernetes/pull/67734), [@Huang-Wei](https://github.com/Huang-Wei))
- If `TaintNodesByCondition` is enabled, add `node.kubernetes.io/unschedulable` and `node.kubernetes.io/network-unavailable` automatically to DaemonSet pods. ([#64954](https://github.com/kubernetes/kubernetes/pull/64954), [@k82cn](https://github.com/k82cn))

### SIG Storage

- The AllowedTopologies field inside StorageClass is now validated against set and map semantics. Specifically, there cannot be duplicate TopologySelectorTerms, MatchLabelExpressions keys, or TopologySelectorLabelRequirement Values. ([#66843](https://github.com/kubernetes/kubernetes/pull/66843), [@verult](https://github.com/verult))
- A PersistentVolumeClaim may not have been synced to the controller local cache in time if the PersistentVolumeis bound by an external PV binder (such as kube-scheduler), so Kubernetes will now double check if PVC is not found in order to prevent the volume from being incorrectly reclaimed. ([#67062](https://github.com/kubernetes/kubernetes/pull/67062), [@cofyc](https://github.com/cofyc))
- Filesystems will now be properly unmounted when a backend is not reachable and returns EIO. ([#67097](https://github.com/kubernetes/kubernetes/pull/67097), [@chakri-nelluri](https://github.com/chakri-nelluri))
- The logic for attaching volumes has been changed so that attachdetach controller attaches volumes immediately when a Pod's PVCs are bound, preventing a problem that caused pods to have extremely long startup times. ([#66863](https://github.com/kubernetes/kubernetes/pull/66863), [@cofyc](https://github.com/cofyc))
- Dynamic provisions that create iSCSI PVs can now ensure that multipath is used by specifying 2 or more target portals in the PV, which will cause kubelet to wait up to 10 seconds for the multipath device. PVs with just one portal continue to work as before, with kubelet not waiting for the multipath device and just using the first disk it finds. ([#67140](https://github.com/kubernetes/kubernetes/pull/67140), [@bswartz](https://github.com/bswartz))
- ScaleIO volumes can now be provisioned without having to first manually create /dev/disk/by-id path on each kubernetes node (if not already present). ([#66174](https://github.com/kubernetes/kubernetes/pull/66174), [@ddebroy](https://github.com/ddebroy))
- Multi-line annotations injected via downward API files will no longer be sorted, scrambling their information. ([#65992](https://github.com/kubernetes/kubernetes/pull/65992), [@liggitt](https://github.com/liggitt))
- The constructed volume spec for the CSI plugin now includes a volume mode field. ([#65456](https://github.com/kubernetes/kubernetes/pull/65456), [@wenlxie](https://github.com/wenlxie))
- Kubernetes now includes a metric that reports the number of PVCs that are in-use,with plugin and node name as dimensions, making it possible to figure out how many PVCs each node is using when troubleshooting attach/detach issues.
 ([#64527](https://github.com/kubernetes/kubernetes/pull/64527), [@gnufied](https://github.com/gnufied))
- Added support to restore a volume from a volume snapshot data source.  ([#67087](https://github.com/kubernetes/kubernetes/pull/67087), [@xing-yang](https://github.com/xing-yang))
- When attaching iSCSI volumes, kubelet now scans only the specific LUNs being attached, and also deletes them after detaching. This avoids dangling references to LUNs that no longer exist, which used to be the cause of random I/O errors/timeouts in kernel logs, slowdowns during block-device related operations, and very rare cases of data corruption.
([#63176](https://github.com/kubernetes/kubernetes/pull/63176), [@bswartz](https://github.com/bswartz))
- Both directory and block devices are now supported for local volume plugin FileSystem VolumeMode.  ([#63011](https://github.com/kubernetes/kubernetes/pull/63011), [@NickrenREN](https://github.com/NickrenREN))
- CSI NodePublish call can optionally contain information about the pod that requested the CSI volume. ([#67945](https://github.com/kubernetes/kubernetes/pull/67945), [@jsafrane](https://github.com/jsafrane))
- Added support for volume attach limits for CSI volumes. ([#67731](https://github.com/kubernetes/kubernetes/pull/67731), [@gnufied](https://github.com/gnufied))

### SIG VMWare

- The vmUUID is now preserved when renewing nodeinfo in the vSphere cloud provider. ([#66007](https://github.com/kubernetes/kubernetes/pull/66007), [@w-leads](https://github.com/w-leads))
- You can now configure the vsphere cloud provider with a trusted Root-CA, enabling you to take advantage of TLS certificate rotation. ([#64758](https://github.com/kubernetes/kubernetes/pull/64758), [@mariantalla](https://github.com/mariantalla))

### SIG Windows

- Kubelet no longer attempts to sync iptables on non-Linux systems.. ([#67690](https://github.com/kubernetes/kubernetes/pull/67690), [@feiskyer](https://github.com/feiskyer))
- Kubelet no longer applies default hard evictions of nodefs.inodesFree on non-Linux systems. ([#67709](https://github.com/kubernetes/kubernetes/pull/67709), [@feiskyer](https://github.com/feiskyer))
- Windows system container "pods" now support kubelet stats. ([#66427](https://github.com/kubernetes/kubernetes/pull/66427), [@feiskyer](https://github.com/feiskyer))

## Other Notable Changes

### Bug Fixes

- Update debian-iptables and hyperkube-base images to include CVE fixes. ([#67365](https://github.com/kubernetes/kubernetes/pull/67365), [@ixdy](https://github.com/ixdy))
- Fix for resourcepool-path configuration in the vsphere.conf file. ([#66261](https://github.com/kubernetes/kubernetes/pull/66261), [@divyenpatel](https://github.com/divyenpatel))
- This fix prevents a GCE PD volume from being mounted if the udev device link is stale and tries to correct the link. ([#66832](https://github.com/kubernetes/kubernetes/pull/66832), [@msau42](https://github.com/msau42))
- Fix controller-manager crashes when flex plugin is removed from flex plugin directory ([#65536](https://github.com/kubernetes/kubernetes/pull/65536), [@gnufied](https://github.com/gnufied))
- Fix local volume directory can't be deleted because of volumeMode error ([#65310](https://github.com/kubernetes/kubernetes/pull/65310), [@wenlxie](https://github.com/wenlxie))
- bugfix: Do not print feature gates in the generic apiserver code for glog level 0 ([#65584](https://github.com/kubernetes/kubernetes/pull/65584), [@neolit123](https://github.com/neolit123))
- Fix an issue that pods using hostNetwork keep increasing. ([#67456](https://github.com/kubernetes/kubernetes/pull/67456), [@Huang-Wei](https://github.com/Huang-Wei))
- fixes an out of range panic in the NoExecuteTaintManager controller when running a non-64-bit build ([#65596](https://github.com/kubernetes/kubernetes/pull/65596), [@liggitt](https://github.com/liggitt))
- Fix kubelet to not leak goroutines/intofiy watchers on an inactive connection if it's closed ([#67285](https://github.com/kubernetes/kubernetes/pull/67285), [@yujuhong](https://github.com/yujuhong))
- Fix pod launch by kubelet when --cgroups-per-qos=false and --cgroup-driver="systemd" ([#66617](https://github.com/kubernetes/kubernetes/pull/66617), [@pravisankar](https://github.com/pravisankar))
- Fixed a panic in the node status update logic when existing node has nil labels. ([#66307](https://github.com/kubernetes/kubernetes/pull/66307), [@guoshimin](https://github.com/guoshimin))
- Fix the bug where image garbage collection is disabled by mistake. ([#66051](https://github.com/kubernetes/kubernetes/pull/66051), [@jiaxuanzhou](https://github.com/jiaxuanzhou))
- Fix a bug that preempting a pod may block forever. ([#65987](https://github.com/kubernetes/kubernetes/pull/65987), [@Random-Liu](https://github.com/Random-Liu))
- fixes the errors/warnings in fluentd configuration ([#67947](https://github.com/kubernetes/kubernetes/pull/67947), [@saravanan30erd](https://github.com/saravanan30erd))
- Fixed an issue which prevented `gcloud` from working on GCE when metadata concealment was enabled. ([#66630](https://github.com/kubernetes/kubernetes/pull/66630), [@dekkagaijin](https://github.com/dekkagaijin))
- Fix Stackdriver integration based on node annotation container.googleapis.com/instance_id. ([#66676](https://github.com/kubernetes/kubernetes/pull/66676), [@kawych](https://github.com/kawych))
- GCE: Fixes loadbalancer creation and deletion issues appearing in 1.10.5. ([#66400](https://github.com/kubernetes/kubernetes/pull/66400), [@nicksardo](https://github.com/nicksardo))
- Fixed exception detection in fluentd-gcp plugin. ([#65361](https://github.com/kubernetes/kubernetes/pull/65361), [@xperimental](https://github.com/xperimental))
- kubeadm:  Fix panic when node annotation is nil ([#67648](https://github.com/kubernetes/kubernetes/pull/67648), [@xlgao-zju](https://github.com/xlgao-zju))
- kubeadm: stop setting UID in the kubelet ConfigMap ([#66341](https://github.com/kubernetes/kubernetes/pull/66341), [@runiq](https://github.com/runiq))
- bazel deb package bugfix: The kubeadm deb package now reloads the kubelet after installation ([#65554](https://github.com/kubernetes/kubernetes/pull/65554), [@rdodev](https://github.com/rdodev))
- fix cluster-info dump error ([#66652](https://github.com/kubernetes/kubernetes/pull/66652), [@charrywanganthony](https://github.com/charrywanganthony))
- Fix kubelet startup failure when using ExecPlugin in kubeconfig ([#66395](https://github.com/kubernetes/kubernetes/pull/66395), [@awly](https://github.com/awly))
- kubectl: fixes a panic displaying pods with nominatedNodeName set ([#66406](https://github.com/kubernetes/kubernetes/pull/66406), [@liggitt](https://github.com/liggitt))
- prevents infinite CLI wait on delete when item is recreated ([#66136](https://github.com/kubernetes/kubernetes/pull/66136), [@deads2k](https://github.com/deads2k))
- Fix 'kubectl cp' with no arguments causes a panic ([#65482](https://github.com/kubernetes/kubernetes/pull/65482), [@wgliang](https://github.com/wgliang))
- Fixes the wrong elasticsearch node counter ([#65627](https://github.com/kubernetes/kubernetes/pull/65627), [@IvanovOleg](https://github.com/IvanovOleg))
- Fix an issue with dropped audit logs, when truncating and batch backends enabled at the same time. ([#65823](https://github.com/kubernetes/kubernetes/pull/65823), [@loburm](https://github.com/loburm))
- DaemonSet: Fix bug- daemonset didn't create pod after node have enough resource ([#67337](https://github.com/kubernetes/kubernetes/pull/67337), [@linyouchong](https://github.com/linyouchong))
- DaemonSet controller is now using backoff algorithm to avoid hot loops fighting with kubelet on pod recreation when a particular DaemonSet is misconfigured. ([#65309](https://github.com/kubernetes/kubernetes/pull/65309), [@tnozicka](https://github.com/tnozicka))
- Avoid creating new controller revisions for statefulsets when cache is stale ([#67039](https://github.com/kubernetes/kubernetes/pull/67039), [@mortent](https://github.com/mortent))
- Fixes issue when updating a DaemonSet causes a hash collision. ([#66476](https://github.com/kubernetes/kubernetes/pull/66476), [@mortent](https://github.com/mortent))
- fix rollout status for statefulsets ([#62943](https://github.com/kubernetes/kubernetes/pull/62943), [@faraazkhan](https://github.com/faraazkhan))
- fixes a validation error that could prevent updates to StatefulSet objects containing non-normalized resource requests ([#66165](https://github.com/kubernetes/kubernetes/pull/66165), [@liggitt](https://github.com/liggitt))
- Headless Services with no ports defined will now create Endpoints correctly, and appear in DNS. ([#67622](https://github.com/kubernetes/kubernetes/pull/67622), [@thockin](https://github.com/thockin))
- Prevent `resourceVersion` updates for custom resources on no-op writes. ([#67562](https://github.com/kubernetes/kubernetes/pull/67562), [@nikhita](https://github.com/nikhita))
- kube-controller-manager can now start the quota controller when discovery results can only be partially determined. ([#67433](https://github.com/kubernetes/kubernetes/pull/67433), [@deads2k](https://github.com/deads2k))
- Immediately close the other side of the connection when proxying. ([#67288](https://github.com/kubernetes/kubernetes/pull/67288), [@MHBauer](https://github.com/MHBauer))
- kube-apiserver: fixes error creating system priority classes when starting multiple apiservers simultaneously ([#67372](https://github.com/kubernetes/kubernetes/pull/67372), [@tanshanshan](https://github.com/tanshanshan))
-  Forget rate limit when CRD establish controller successfully updated CRD condition ([#67370](https://github.com/kubernetes/kubernetes/pull/67370), [@yue9944882](https://github.com/yue9944882))
- fixes a panic when using a mutating webhook admission plugin with a DELETE operation ([#66425](https://github.com/kubernetes/kubernetes/pull/66425), [@liggitt](https://github.com/liggitt))
- Fix creation of custom resources when the CRD contains non-conventional pluralization and subresources ([#66249](https://github.com/kubernetes/kubernetes/pull/66249), [@deads2k](https://github.com/deads2k))
- Aadjusted http/2 buffer sizes for apiservers to prevent starvation issues between concurrent streams ([#67902](https://github.com/kubernetes/kubernetes/pull/67902), [@liggitt](https://github.com/liggitt))
- Fixed a bug that was blocking extensible error handling when serializing API responses error out. Previously, serialization failures always resulted in the status code of the original response being returned. Now, the following behavior occurs: ([#67041](https://github.com/kubernetes/kubernetes/pull/67041), [@tristanburgess](https://github.com/tristanburgess))
- Fixes issue where pod scheduling may fail when using local PVs and pod affinity and anti-affinity without the default StatefulSet OrderedReady pod management policy ([#67556](https://github.com/kubernetes/kubernetes/pull/67556), [@msau42](https://github.com/msau42))
- Fix panic when processing Azure HTTP response. ([#68210](https://github.com/kubernetes/kubernetes/pull/68210), [@feiskyer](https://github.com/feiskyer))
- Fix volume limit for EBS on m5 and c5 instance types ([#66397](https://github.com/kubernetes/kubernetes/pull/66397), [@gnufied](https://github.com/gnufied))
- Fix a bug on GCE that /etc/crictl.yaml is not generated when crictl is preloaded. ([#66877](https://github.com/kubernetes/kubernetes/pull/66877), [@Random-Liu](https://github.com/Random-Liu))
- Revert #63905: Setup dns servers and search domains for Windows Pods. DNS for Windows containers will be set by CNI plugins. ([#66587](https://github.com/kubernetes/kubernetes/pull/66587), [@feiskyer](https://github.com/feiskyer))
- Fix validation for HealthzBindAddress in kube-proxy when --healthz-port is set to 0 ([#66138](https://github.com/kubernetes/kubernetes/pull/66138), [@wsong](https://github.com/wsong))
- Fixes issue [#68899](https://github.com/kubernetes/kubernetes/issues/68899) where pods might schedule on an unschedulable node. ([#68984](https://github.com/kubernetes/kubernetes/issues/68984), [@k82cn](https://github.com/k82cn))

### Not Very Notable (that is, non-user-facing)

- Unit tests have been added for scopes and scope selectors in the quota spec ([#66351](https://github.com/kubernetes/kubernetes/pull/66351), [@vikaschoudhary16](https://github.com/vikaschoudhary16)) Courtesy of SIG Node, and SIG Scheduling
- kubelet v1beta1 external ComponentConfig types are now available in the `k8s.io/kubelet` repo ([#67263](https://github.com/kubernetes/kubernetes/pull/67263), [@luxas](https://github.com/luxas)) Courtesy of SIG Cluster Lifecycle, SIG Node, SIG Scheduling, and SIG Testing
- Use sync.map to scale ecache better ([#66862](https://github.com/kubernetes/kubernetes/pull/66862), [@resouer](https://github.com/resouer))
- Extender preemption should respect IsInterested() ([#66291](https://github.com/kubernetes/kubernetes/pull/66291), [@resouer](https://github.com/resouer))
- This PR will leverage subtests on the existing table tests for the scheduler units. ([#63665](https://github.com/kubernetes/kubernetes/pull/63665), [@xchapter7x](https://github.com/xchapter7x))
- This PR will leverage subtests on the existing table tests for the scheduler units. ([#63666](https://github.com/kubernetes/kubernetes/pull/63666), [@xchapter7x](https://github.com/xchapter7x))
- Re-adds `pkg/generated/bindata.go` to the repository to allow some parts of k8s.io/kubernetes to be go-vendorable. ([#65985](https://github.com/kubernetes/kubernetes/pull/65985), [@ixdy](https://github.com/ixdy))
- If `TaintNodesByCondition` enabled, taint node with `TaintNodeUnschedulable` when initializing node to avoid race condition.
([#63955](https://github.com/kubernetes/kubernetes/pull/63955), [@k82cn](https://github.com/k82cn))
- Remove rescheduler since scheduling DS pods by default scheduler is moving to beta. ([#67687](https://github.com/kubernetes/kubernetes/pull/67687), [@Lion-Wei](https://github.com/Lion-Wei))
- kubeadm: make sure pre-pulled kube-proxy image and the one specified in its daemon set manifest are the same ([#67131](https://github.com/kubernetes/kubernetes/pull/67131), [@rosti](https://github.com/rosti))
- kubeadm: remove misleading error message regarding image pulling ([#66658](https://github.com/kubernetes/kubernetes/pull/66658), [@dixudx](https://github.com/dixudx))
- kubeadm: Pull sidecar and dnsmasq-nanny images when using kube-dns ([#66499](https://github.com/kubernetes/kubernetes/pull/66499), [@rosti](https://github.com/rosti))
- kubeadm: Fix pause image to not use architecture, as it is a manifest list ([#65920](https://github.com/kubernetes/kubernetes/pull/65920), [@dims](https://github.com/dims))
- kubeadm: Remove usage of `PersistentVolumeLabel` ([#65827](https://github.com/kubernetes/kubernetes/pull/65827), [@xlgao-zju](https://github.com/xlgao-zju))
- kubeadm: Add a `v1alpha3` API. This change creates a v1alpha3 API that is initially a duplicate of v1alpha2. ([#65629](https://github.com/kubernetes/kubernetes/pull/65629), [@luxas](https://github.com/luxas))
- Improved error message when checking the rollout status of StatefulSet with OnDelete strategy type. ([#66983](https://github.com/kubernetes/kubernetes/pull/66983), [@mortent](https://github.com/mortent))
- Defaults for file audit logging backend in batch mode changed: ([#67223](https://github.com/kubernetes/kubernetes/pull/67223), [@tallclair](https://github.com/tallclair))
- Role, ClusterRole and their bindings for cloud-provider is put under system namespace. Their addonmanager mode switches to EnsureExists. ([#67224](https://github.com/kubernetes/kubernetes/pull/67224), [@grayluck](https://github.com/grayluck))
- Don't let aggregated apiservers fail to launch if the external-apiserver-authentication configmap is not found in the cluster. ([#67836](https://github.com/kubernetes/kubernetes/pull/67836), [@sttts](https://github.com/sttts))
- Always create configmaps/extensions-apiserver-authentication from kube-apiserver. ([#67694](https://github.com/kubernetes/kubernetes/pull/67694), [@sttts](https://github.com/sttts))
- Switched certificate data replacement from "REDACTED" to "DATA+OMITTED" ([#66023](https://github.com/kubernetes/kubernetes/pull/66023), [@ibrasho](https://github.com/ibrasho))
- Decrease the amount of time it takes to modify kubeconfig files with large amounts of contexts ([#67093](https://github.com/kubernetes/kubernetes/pull/67093), [@juanvallejo](https://github.com/juanvallejo))
- Make EBS volume expansion faster ([#66728](https://github.com/kubernetes/kubernetes/pull/66728), [@gnufied](https://github.com/gnufied))
- Remove unused binary and container image for kube-aggregator. The functionality is already integrated into the kube-apiserver. ([#67157](https://github.com/kubernetes/kubernetes/pull/67157), [@dims](https://github.com/dims))
- kube-controller-manager now uses the informer cache instead of active pod gets in HPA controller ([#68241](https://github.com/kubernetes/kubernetes/pull/68241), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
- Replace scale down forbidden window with scale down stabilization window. Rather than waiting a fixed period of time between scale downs HPA now scales down to the highest recommendation it during the scale down stabilization window. ([#68122](https://github.com/kubernetes/kubernetes/pull/68122), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
- Improve CPU sample sanitization in HPA by taking metric's freshness into account. ([#68068](https://github.com/kubernetes/kubernetes/pull/68068), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
- Replace scale up forbidden window with disregarding CPU samples collected when pod was initializing. ([#67252](https://github.com/kubernetes/kubernetes/pull/67252), [@jbartosik](https://github.com/jbartosik))
- [e2e] verifying LimitRange update is effective before creating new pod ([#68171](https://github.com/kubernetes/kubernetes/pull/68171), [@dixudx](https://github.com/dixudx))
- Port 31337 will be used by fluentd ([#68051](https://github.com/kubernetes/kubernetes/pull/68051), [@Szetty](https://github.com/Szetty))
- Fix flexvolume in containarized kubelets ([#65549](https://github.com/kubernetes/kubernetes/pull/65549), [@gnufied](https://github.com/gnufied))
- The check for unsupported plugins during volume resize has been moved from the admission controller to the two controllers that handle volume resize. ([#66780](https://github.com/kubernetes/kubernetes/pull/66780), [@kangarlou](https://github.com/kangarlou))
- kubeadm: remove redundant flags settings for kubelet ([#64682](https://github.com/kubernetes/kubernetes/pull/64682), [@dixudx](https://github.com/dixudx))
- Set “priorityClassName: system-node-critical” on kube-proxy manifest by default. ([#60150](https://github.com/kubernetes/kubernetes/pull/60150), [@MrHohn](https://github.com/MrHohn))
- kube-proxy v1beta1 external ComponentConfig types are now available in the `k8s.io/kube-proxy` repo ([#67688](https://github.com/kubernetes/kubernetes/pull/67688), [@Lion-Wei](https://github.com/Lion-Wei))
- add missing LastTransitionTime of ContainerReady condition ([#64867](https://github.com/kubernetes/kubernetes/pull/64867), [@dixudx](https://github.com/dixudx))

##  External Dependencies

- Default etcd server was updated to v3.2.24. ([#68318](https://github.com/kubernetes/kubernetes/pull/68318))
- Rescheduler is unchanged from v1.11: v0.4.0. ([#65454](https://github.com/kubernetes/kubernetes/pull/65454))
- The list of validated docker versions was updated to 1.11.1, 1.12.1, 1.13.1, 17.03, 17.06, 17.09, 18.06. ([#68495](https://github.com/kubernetes/kubernetes/pull/68495))
- The default Go version was updated to 1.10.4. ([68802](https://github.com/kubernetes/kubernetes/pull/68802))
- The minimum supported Go version was updated to 1.10.2 ([#63412](https://github.com/kubernetes/kubernetes/pull/63412))
- CNI is unchanged from v1.10: v0.6.0 ([#51250](https://github.com/kubernetes/kubernetes/pull/51250))
- CSI is unchanged from v1.11:  0.3.0 ([#64719](https://github.com/kubernetes/kubernetes/pull/64719))
- The dashboard add-on unchanged from v1.10: v1.8.3. ([#57326](https://github.com/kubernetes/kubernetes/pull/57326))
- Bump Heapster to v1.6.0-beta as compared to v1.5.2 in v1.11  ([#67074](https://github.com/kubernetes/kubernetes/pull/67074))
- Cluster Autoscaler has been upgraded to v1.12.0 ([#s8739](https://github.com/kubernetes/kubernetes/pull/68739))
- kube-dns was updated to v1.14.13. ([#68900](https://github.com/kubernetes/kubernetes/pull/68900))
- Influxdb is unchanged from v1.10: v1.3.3 ([#53319](https://github.com/kubernetes/kubernetes/pull/53319))
- Grafana is unchanged from v1.10: v4.4.3 ([#53319](https://github.com/kubernetes/kubernetes/pull/53319))
- Kibana is at v6.3.2.  ([#67582](https://github.com/kubernetes/kubernetes/pull/67582))
- CAdvisor is unchanged from v1.11:  v0.30.1 ([#64987](https://github.com/kubernetes/kubernetes/pull/64987))
- fluentd-gcp-scaler has been updated to v0.4.0, up from 0.3.0 in v1.11. ([#67691](https://github.com/kubernetes/kubernetes/pull/67691))
- fluentd in fluentd-es-image is unchanged from 1.10: v1.1.0 ([#58525](https://github.com/kubernetes/kubernetes/pull/58525))
- Fluentd in fluentd-elasticsearch is unchanged from v1.11:  v1.2.4 ([#67434](https://github.com/kubernetes/kubernetes/pull/67434))
- fluentd-elasticsearch is unchanged from 1.10: v2.0.4 ([#58525](https://github.com/kubernetes/kubernetes/pull/58525))
- The fluent-plugin-kubernetes_metadata_filter plugin in fluentd-elasticsearch has been downgraded to version 2.0.0 ([#67544](https://github.com/kubernetes/kubernetes/pull/67544))
- fluentd-gcp is unchanged from 1.10: v3.0.0. ([#60722](https://github.com/kubernetes/kubernetes/pull/60722))
- Ingress glbc is unchanged from 1.10: v1.0.0 ([#61302](https://github.com/kubernetes/kubernetes/pull/61302))
- OIDC authentication is unchanged from 1.10: coreos/go-oidc v2 ([#58544](https://github.com/kubernetes/kubernetes/pull/58544))
- Calico is unchanged from 1.10: v2.6.7 ([#59130](https://github.com/kubernetes/kubernetes/pull/59130))
- hcsshim is unchanged from v1.11, at v0.11 ([#64272](https://github.com/kubernetes/kubernetes/pull/64272))
- gitRepo volumes in pods no longer require git 1.8.5 or newer; older git versions are now supported. ([#62394](https://github.com/kubernetes/kubernetes/pull/62394))
- Upgraded crictl on GCE to v1.11.1, up from 1.11.0 on v1.11.  ([#66152](https://github.com/kubernetes/kubernetes/pull/66152))
- CoreDNS has been updated to v1.2.2, up from v1.1.3 in v1.11 ([#68076](https://github.com/kubernetes/kubernetes/pull/68076))
- Setup dns servers and search domains for Windows Pods in dockershim. Docker EE version >= 17.10.0 is required for propagating DNS to containers. ([#63905](https://github.com/kubernetes/kubernetes/pull/63905))
- Istio addon is unchanged from v1.11, at  0.8.0. See [full Istio release notes](https://istio.io/about/notes/0.6.html) ([#64537](https://github.com/kubernetes/kubernetes/pull/64537))
- cadvisor godeps is unchanged from v1.11, at  v0.30.0 ([#64800](https://github.com/kubernetes/kubernetes/pull/64800))
- event-exporter to version v0.2.2, compared to v0.2.0 in v1.11. ([#66157](https://github.com/kubernetes/kubernetes/pull/66157))
- Rev the Azure SDK for networking to 2017-06-01 ([#61955](https://github.com/kubernetes/kubernetes/pull/61955))
- Es-image has been upgraded to Elasticsearch 6.3.2 ([#67484](https://github.com/kubernetes/kubernetes/pull/67484))
- metrics-server has been upgraded to v0.3.1. ([#68746](https://github.com/kubernetes/kubernetes/pull/68746))
- GLBC has been updated to v1.2.3 ([#66793](https://github.com/kubernetes/kubernetes/pull/66793))
- Ingress-gce has been updated to v 1.2.3 ([#66793](https://github.com/kubernetes/kubernetes/pull/66793))
- ip-masq-agen has been updated to v2.1.1 ([#67916](https://github.com/kubernetes/kubernetes/pull/67916))
- [v1.12.0-rc.2](#v1120-rc2)
- [v1.12.0-rc.1](#v1120-rc1)
- [v1.12.0-beta.2](#v1120-beta2)
- [v1.12.0-beta.1](#v1120-beta1)
- [v1.12.0-alpha.1](#v1120-alpha1)



# v1.12.0-rc.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.12/examples)

## Downloads for v1.12.0-rc.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes.tar.gz) | `184ea437bc72d0e6a4c96b964de53181273e919a1d4785515da3406c7e982bf5`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-src.tar.gz) | `aee82938827ef05ab0ee81bac42f4f79fff126294469868d02efb3426717d71e`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-darwin-386.tar.gz) | `40ed3ef9bbc4fad7787dd14eae952edf06d40e1094604bc6d10209b8778c3121`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-darwin-amd64.tar.gz) | `a317fe3801ea5387ce474b9759a7e28ede8324587f79935a7a945da44c99a4b2`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-linux-386.tar.gz) | `cd61b4b71d6b739582c02b5be1d87d928507bc59f64ee72629a920cc529a0941`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-linux-amd64.tar.gz) | `306af04fc18ca2588e16fd831358df50a2cb02219687b543073836f835de8583`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-linux-arm.tar.gz) | `497584f2686339cce857cff1ebf4ed10dcd63f4684a03c242b0828fcd307be4c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-linux-arm64.tar.gz) | `1dfbb8c299f5af15239ef39135a6c8a52ee4c234764ee0437d8f707e636c9124`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-linux-ppc64le.tar.gz) | `668d6f35c5f6adcd25584d9ef74c549db13ffca9d93b4bc8d25609a8e5837640`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-linux-s390x.tar.gz) | `8a8e205c38858bd9d161115e5e2870c6cfc9c82e189d156e7062e6fa979c3fda`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-windows-386.tar.gz) | `cdef48279c22cc8c764e43a4b9c2a86f02f21c80abbbcd48041fb1e89fb1eb67`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-client-windows-amd64.tar.gz) | `50621a3d2b1550c69325422c6dce78f5690574b35d3778dd3afcf698b57f0f54`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-server-linux-amd64.tar.gz) | `87a8438887a2daa199508aae591b158025860b8381c64cbe9b1d0c06c4eebde9`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-server-linux-arm.tar.gz) | `f65be73870a0e564ef8ce1b6bb2b75ff7021a6807de84b5750e4fa78635051b6`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-server-linux-arm64.tar.gz) | `171f15aa8b7c365f4fee70ce025c882a921d0075bd726a99b5534cadd09273ef`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-server-linux-ppc64le.tar.gz) | `abc2003d58bd1aca517415c582ed1e8bb1ed596bf04197f4fc7c0c51865a9f86`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-server-linux-s390x.tar.gz) | `e2ce834abb4d45d91fd7a8d774e47f0f8092eb4edcf556605c2ef6e2b190b8b1`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-node-linux-amd64.tar.gz) | `6016c3a1e14c42dcc88caed6497de1b2c56a02bb52d836b19e2ff52098302dda`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-node-linux-arm.tar.gz) | `e712e38c8037159ea074ad93c2f2905cf279f3f119e5fdbf9b97391037a8813f`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-node-linux-arm64.tar.gz) | `7f4095f12d8ad9438919fa447360113799f88bb9435369b9307a41dd9c7692a6`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-node-linux-ppc64le.tar.gz) | `4aeb5dbb0c68e54570542eb5a1d7506d73c81b57eba3c2080ee73bb53dbc3be0`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-node-linux-s390x.tar.gz) | `a160599598167208286db6dc73b415952836218d967fa964fc432b213f1b9908`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.2/kubernetes-node-windows-amd64.tar.gz) | `174bedf62b7959d4cb1b1595666f607cd6377c7a2e2208fef5bd554603db5db3`

## Changelog since v1.12.0-rc.1

### Other notable changes

* Update to use manifest list for etcd image ([#68896](https://github.com/kubernetes/kubernetes/pull/68896), [@ixdy](https://github.com/ixdy))
* Fix Azure nodes power state for InstanceShutdownByProviderID() ([#68921](https://github.com/kubernetes/kubernetes/pull/68921), [@feiskyer](https://github.com/feiskyer))
* Bump kube-dns to 1.14.13 ([#68900](https://github.com/kubernetes/kubernetes/pull/68900), [@MrHohn](https://github.com/MrHohn))
    * - Update Alpine base image to 3.8.1.
    * - Build multi-arch images correctly.
* kubelet: fix grpc timeout in the CRI client ([#67793](https://github.com/kubernetes/kubernetes/pull/67793), [@fisherxu](https://github.com/fisherxu))
* Update to golang 1.10.4 ([#68802](https://github.com/kubernetes/kubernetes/pull/68802), [@ixdy](https://github.com/ixdy))
* kubeadm now uses fat manifests for the kube-dns images ([#68830](https://github.com/kubernetes/kubernetes/pull/68830), [@rosti](https://github.com/rosti))
* Update Cluster Autoscaler version to 1.12.0. ([#68739](https://github.com/kubernetes/kubernetes/pull/68739), [@losipiuk](https://github.com/losipiuk))
    * See https://github.com/kubernetes/autoscaler/releases/tag/1.12.0 for CA release notes.
* kube-proxy restores the *filter table when running in ipvs mode. ([#68786](https://github.com/kubernetes/kubernetes/pull/68786), [@alexjx](https://github.com/alexjx))
* New kubeDNS image fixes an issue where SRV records were incorrectly being compressed. Added manifest file for multiple arch images. ([#68430](https://github.com/kubernetes/kubernetes/pull/68430), [@prameshj](https://github.com/prameshj))
* Drain should delete terminal pods. ([#68767](https://github.com/kubernetes/kubernetes/pull/68767), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))



# v1.12.0-rc.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.12/examples)

## Downloads for v1.12.0-rc.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes.tar.gz) | `ac65cf9571c3a03105f373db23c8d7f4d01fe1c9ee09b06615bb02d0b81d572c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-src.tar.gz) | `28518e1d9c7fe5c54aa3b57235ac8d1a7dae02aec04177c38ca157fc2d16edb6`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-darwin-386.tar.gz) | `7b6f6f264464d40b7975baecdd796d4f75c5a305999b4ae1f4513646184cac7c`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | `5feabe3e616125a36ce4c8021d6bdccdec0f3d82f151b80af7cac1453255b4d5`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-linux-386.tar.gz) | `40524a1a09dd24081b3494593a02a461227727f8706077542f2b8603e1cf7e06`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | `ac2c9757d7df761bdf8ffc259fff07448c300dd110c7dbe2ae3830197eb023e9`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-linux-arm.tar.gz) | `02f27ae16e8ebb12b3cb66391fe85f64de08a99450d726e9defd2c5bcd590955`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | `1286af2cad3f8e2ee8e2dc18a738935779631b58e7ef3da8794bbeadca2f332e`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | `9c04419b159fb0fe501d6e0c8122d6a80b5d6961070ebc5e759f4327a1156cf4`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | `104d5c695826971c64cb0cec26cf791d609d3e831edb33574e9af2c4b191f049`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-windows-386.tar.gz) | `0096f8126eb04eafa9decd258f6d09977d24eee91b83781347a34ebb7d2064aa`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | `a641a1a421795279a6213163d7becab9dc6014362e6566f13d660ef1638dc286`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | `202958d3cfb774fd065ad1ec2477dc9c92ce7f0ff355807c9a2a3a61e8dad927`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-server-linux-arm.tar.gz) | `474de8f6a58d51eb01f6cc73b41897351528a839f818d5c4f828a484f8bc988b`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | `dbd5affd244815bf45ac0c7a56265800864db623a6a37e7ce9ebe5e5896453f8`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | `a62fefa8ad7b3fbfeb7702dac7d4d6f37823b6c3e4edae3356bf0781b48e42e1`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | `0f77690f87503c8ee7ccb473c9d2b9d26420292defd82249509cf50d8bb1a16c`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | `2191845147d5aab08f14312867f86078b513b6aff8685bb8ce84a06b78ae9914`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-node-linux-arm.tar.gz) | `54de98d7d2a71b78bc7a45e70a2005144d210401663f5a9daadedd05f89291f0`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | `a765514e0c4865bb20ceb476af83b9d9356c9b565cfe12615ecf7ad3d5a6b4f7`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | `b7ae7d159602d0b933614071f11216ede4df3fc2b28a30d0018e06b3bb22cf6e`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | `7d4f502eda6aa70b7a18420344abfaec740d74a1edffcb9869e4305c22bba260`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | `ed5516b1f66a39592a101bec135022b3905a66ae526b8ed3e2e9dff5ed68eda0`

## Changelog since v1.12.0-beta.2

### Action Required

* Service events are now added in azure-cloud-provider for easily identify the underground errors of Azure API. ([#68212](https://github.com/kubernetes/kubernetes/pull/68212), [@feiskyer](https://github.com/feiskyer))
    * Action required: The following clusterrole and clusterrolebinding should be applied:
    ```
         kind: List
         apiVersion: v1
         items:
         - apiVersion: rbac.authorization.k8s.io/v1
           kind: ClusterRole
           metadata:
             labels:
               kubernetes.io/cluster-service: "true"
             name: system:azure-cloud-provider
           rules:
           - apiGroups: [""]
             resources: ["events"]
             verbs:
             - create
             - patch
             - update
         - apiVersion: rbac.authorization.k8s.io/v1
           kind: ClusterRoleBinding
           metadata:
             labels:
               kubernetes.io/cluster-service: "true"
             name: system:azure-cloud-provider
           roleRef:
             apiGroup: rbac.authorization.k8s.io
             kind: ClusterRole
             name: system:azure-cloud-provider
           subjects:
           - kind: ServiceAccount
             name: azure-cloud-provider
             namespace: kube-system
    ```
    * If the clusterrole with same has already been provisioned (e.g. for accessing azurefile secrets), then the above yaml should be merged togather, e.g.
    ```
         kind: List
         apiVersion: v1
         items:
         - apiVersion: rbac.authorization.k8s.io/v1
           kind: ClusterRole
           metadata:
             labels:
               kubernetes.io/cluster-service: "true"
             name: system:azure-cloud-provider
           rules:
           - apiGroups: [""]
             resources: ["events"]
             verbs:
             - create
             - patch
             - update
           - apiGroups: [""]
             resources: ["secrets"]
             verbs:
             - get
             - create
         - apiVersion: rbac.authorization.k8s.io/v1
           kind: ClusterRoleBinding
           metadata:
             labels:
               kubernetes.io/cluster-service: "true"
             name: system:azure-cloud-provider
           roleRef:
             apiGroup: rbac.authorization.k8s.io
             kind: ClusterRole
             name: system:azure-cloud-provider
           subjects:
           - kind: ServiceAccount
             name: azure-cloud-provider
             namespace: kube-system
           - kind: ServiceAccount
             name: persistent-volume-binder
             namespace: kube-system
    ```

### Other notable changes

* Update metrics-server to v0.3.1 ([#68746](https://github.com/kubernetes/kubernetes/pull/68746), [@DirectXMan12](https://github.com/DirectXMan12))
* Upgrade kubeadm's version of docker support ([#68495](https://github.com/kubernetes/kubernetes/pull/68495), [@yuansisi](https://github.com/yuansisi))
* fix a bug that overwhelming number of prometheus metrics are generated because $NAMESPACE is not replaced by string "{namespace}" ([#68530](https://github.com/kubernetes/kubernetes/pull/68530), [@wenjiaswe](https://github.com/wenjiaswe))
* The feature gates `ReadOnlyAPIDataVolumes` and `ServiceProxyAllowExternalIPs`, deprecated since 1.10, have been removed and any references must be removed from command-line invocations. ([#67951](https://github.com/kubernetes/kubernetes/pull/67951), [@liggitt](https://github.com/liggitt))
* Verify invalid secret/configmap/projected volumes before calling setup ([#68691](https://github.com/kubernetes/kubernetes/pull/68691), [@gnufied](https://github.com/gnufied))
* Fix bug that caused `kubectl` commands to sometimes fail to refresh access token when running against GKE clusters. ([#66314](https://github.com/kubernetes/kubernetes/pull/66314), [@jlowdermilk](https://github.com/jlowdermilk))
* Use KubeDNS by default in GCE setups, as CoreDNS has significantly higher memory usage in large clusters. ([#68629](https://github.com/kubernetes/kubernetes/pull/68629), [@shyamjvs](https://github.com/shyamjvs))
* Fix PodAntiAffinity issues in case of multiple affinityTerms. ([#68173](https://github.com/kubernetes/kubernetes/pull/68173), [@Huang-Wei](https://github.com/Huang-Wei))
* Make APIGroup field in TypedLocalObjectReference optional. ([#68419](https://github.com/kubernetes/kubernetes/pull/68419), [@xing-yang](https://github.com/xing-yang))
* Fix potential panic when getting azure load balancer status ([#68609](https://github.com/kubernetes/kubernetes/pull/68609), [@feiskyer](https://github.com/feiskyer))
* Fix kubelet panics when RuntimeClass is enabled. ([#68521](https://github.com/kubernetes/kubernetes/pull/68521), [@yujuhong](https://github.com/yujuhong))
* - cAdvisor: Fix NVML initialization race condition ([#68431](https://github.com/kubernetes/kubernetes/pull/68431), [@dashpole](https://github.com/dashpole))
    * - cAdvisor: Fix brtfs filesystem discovery
    * - cAdvisor: Fix race condition with AllDockerContainers
    * - cAdvisor: Don't watch .mount cgroups
    * - cAdvisor: Reduce lock contention during list containers
* Promote ScheduleDaemonSetPods by default scheduler to beta ([#67899](https://github.com/kubernetes/kubernetes/pull/67899), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))



# v1.12.0-beta.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.12/examples)

## Downloads for v1.12.0-beta.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes.tar.gz) | `7163d18b9c1bd98ce804b17469ed67b399deb7b574dd12a86609fc647c5c773b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-src.tar.gz) | `6225b71b2dec0f29afb713e64d2b6b82bd0e122274c31310c0de19ef023cb1d0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `f2ec9799e47c28fce336bc90a6e9b4e47def7081fd73b8e2164940f0a6c824c7`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `0e8cfcbe5ec862423ced97da1d9740d4cc4904a0d5cd11a60616aee596bc7622`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-linux-386.tar.gz) | `1cbd6e8dd892cfc2555d37e733b66aaf85df9950466c7295875d312ac254ddfc`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `47337b58a26a4953e5c061d28e3ec89b3d4354bce40f9b51fbe269598caeff03`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `eaaed82f428fb7ddbb10b4e39a2f287817c33ae24ff16008159f437acc653d4a`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `3249d1c7d5d5500793546eb144fe537d1984a01c7a79c1382eb2e26a78e532cd`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | `67afd34f2199deff901b0872a177dc448ba700dc4ced9ede6f3187a0eed2c6fb`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | `e8faa6e45c6e2aeb67ac65737e09be87c190e3c89782ec87a9a205d4f1af9246`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-windows-386.tar.gz) | `2395051c8cbd0a995b5f3689c0f8c0447bcc1c46440d8cdeffd7c7fccf8e8ae1`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `c6a38ee6eda20656b391ecfcc1f24505eb8a3a5a3200d4bddede318291773619`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `795c713a91118218f5952e1bd4cf0933f36476aa3d9d60a9ee43c9bae8400fd3`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `1798d48a37b8f06878e0ecb8d9b67d0fb5c8ee721608412add57725eb5ce5f1e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `da2459b5e811daaa2fc04a072773e81dc220400f3aeb6e29bb9594c306c7b266`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | `7fd1c2ba0c2c9da5db54f8d0aed28261f03e9953ce01fa367e4ce3d84bf01b4f`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | `c9fafb009d7e5da74f588aaa935244c452de52b9488863b90e8b477b1bb16e52`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-node-linux-amd64.tar.gz) | `ab901137b499829b20b868492d04c1f69d738620b96eb349c642d6d773c44448`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-node-linux-arm.tar.gz) | `116dd82721f200f3f37df0e47aebb611fdd7856f94d4c2ebb1d51db21b793a9c`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-node-linux-arm64.tar.gz) | `56d8316eb95f7f54c154625063617b86ffb8e2cc80b8225cce4f5c91d2d3a64f`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-node-linux-ppc64le.tar.gz) | `66535b16ad588ba3bfcb40728a0497c6821360ab7be9c3ced2072bfa107e5c46`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-node-linux-s390x.tar.gz) | `688e09becc9327e50c68b33161eac63a8ba018c02fb298cbd0de82d6ed5dba90`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.2/kubernetes-node-windows-amd64.tar.gz) | `b72582f67d19c06f605ca9b02c08b7227796c15c639e3c09b06a8b667c4569fe`

## Changelog since v1.12.0-beta.1

### Action Required

* Action required: The --storage-versions flag of kube-apiserver is deprecated. Please omit this flag to ensure the default storage versions are used. Otherwise the cluster is not safe to upgrade to a version newer than 1.12. This flag will be removed in 1.13. ([#68080](https://github.com/kubernetes/kubernetes/pull/68080), [@caesarxuchao](https://github.com/caesarxuchao))

### Other notable changes

* kubeadm: add mandatory "--config" flag to "kubeadm alpha phase preflight" ([#68446](https://github.com/kubernetes/kubernetes/pull/68446), [@neolit123](https://github.com/neolit123))
* Apply user configurations for local etcd ([#68334](https://github.com/kubernetes/kubernetes/pull/68334), [@SataQiu](https://github.com/SataQiu))
* kubeadm: added phase command "alpha phase kubelet config annotate-cri" ([#68449](https://github.com/kubernetes/kubernetes/pull/68449), [@fabriziopandini](https://github.com/fabriziopandini))
* If `TaintNodesByCondition` is enabled, add `node.kubernetes.io/unschedulable` and ([#64954](https://github.com/kubernetes/kubernetes/pull/64954), [@k82cn](https://github.com/k82cn))
    *  `node.kubernetes.io/network-unavailable` automatically to DaemonSet pods.
* Deprecate cloudstack and ovirt controllers ([#68199](https://github.com/kubernetes/kubernetes/pull/68199), [@dims](https://github.com/dims))
* add missing LastTransitionTime of ContainerReady condition ([#64867](https://github.com/kubernetes/kubernetes/pull/64867), [@dixudx](https://github.com/dixudx))
* kube-controller-manager: use informer cache instead of active pod gets in HPA controller ([#68241](https://github.com/kubernetes/kubernetes/pull/68241), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* Support NodeShutdown taint for azure ([#68033](https://github.com/kubernetes/kubernetes/pull/68033), [@yastij](https://github.com/yastij))
* Registers volume topology information reported by a node-level Container Storage Interface (CSI) driver. This enables Kubernetes support of CSI topology mechanisms. ([#67684](https://github.com/kubernetes/kubernetes/pull/67684), [@verult](https://github.com/verult))
* Update default etcd server to 3.2.24 for kubernetes 1.12 ([#68318](https://github.com/kubernetes/kubernetes/pull/68318), [@timothysc](https://github.com/timothysc))
* External CAs can now be used for kubeadm with only a certificate, as long as all required certificates already exist. ([#68296](https://github.com/kubernetes/kubernetes/pull/68296), [@liztio](https://github.com/liztio))
* Bump addon-manager to v8.7 ([#68299](https://github.com/kubernetes/kubernetes/pull/68299), [@MrHohn](https://github.com/MrHohn))
    * - Support extra `--prune-whitelist` resources in kube-addon-manager.
    * - Update kubectl to v1.10.7.
* Let service controller retry creating load balancer when persistUpdate failed due to conflict. ([#68087](https://github.com/kubernetes/kubernetes/pull/68087), [@grayluck](https://github.com/grayluck))
* Kubelet now only sync iptables on Linux. ([#67690](https://github.com/kubernetes/kubernetes/pull/67690), [@feiskyer](https://github.com/feiskyer))
* CSI NodePublish call can optionally contain information about the pod that requested the CSI volume. ([#67945](https://github.com/kubernetes/kubernetes/pull/67945), [@jsafrane](https://github.com/jsafrane))
* [e2e] verifying LimitRange update is effective before creating new pod ([#68171](https://github.com/kubernetes/kubernetes/pull/68171), [@dixudx](https://github.com/dixudx))
* cluster/gce: generate consistent key sizes in config-default.sh using /dev/urandom instead of /dev/random   ([#67139](https://github.com/kubernetes/kubernetes/pull/67139), [@yogi-sagar](https://github.com/yogi-sagar))
* Add support for volume attach limits for CSI volumes ([#67731](https://github.com/kubernetes/kubernetes/pull/67731), [@gnufied](https://github.com/gnufied))
* CSI volume plugin does not need external attacher for non-attachable CSI volumes. ([#67955](https://github.com/kubernetes/kubernetes/pull/67955), [@jsafrane](https://github.com/jsafrane))
* KubeletPluginsWatcher feature graduates to beta. ([#68200](https://github.com/kubernetes/kubernetes/pull/68200), [@RenaudWasTaken](https://github.com/RenaudWasTaken))
* Update etcd client to 3.2.24 for latest release ([#68147](https://github.com/kubernetes/kubernetes/pull/68147), [@timothysc](https://github.com/timothysc))
* [fluentd-gcp-scaler addon] Bump fluentd-gcp-scaler to 0.4 to pick up security fixes. ([#67691](https://github.com/kubernetes/kubernetes/pull/67691), [@loburm](https://github.com/loburm))
    * [prometheus-to-sd addon] Bump prometheus-to-sd to 0.3.1 to pick up security fixes, bug fixes and new features.
    * [event-exporter addon] Bump event-exporter to 0.2.3 to pick up security fixes.
* Fixes issue where pod scheduling may fail when using local PVs and pod affinity and anti-affinity without the default StatefulSet OrderedReady pod management policy ([#67556](https://github.com/kubernetes/kubernetes/pull/67556), [@msau42](https://github.com/msau42))
* Kubelet only applies default hard evictions of nodefs.inodesFree on Linux ([#67709](https://github.com/kubernetes/kubernetes/pull/67709), [@feiskyer](https://github.com/feiskyer))
* Add kubelet stats for windows system container "pods" ([#66427](https://github.com/kubernetes/kubernetes/pull/66427), [@feiskyer](https://github.com/feiskyer))
* Add a TTL machenism to clean up Jobs after they finish. ([#66840](https://github.com/kubernetes/kubernetes/pull/66840), [@janetkuo](https://github.com/janetkuo))



# v1.12.0-beta.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.12/examples)

## Downloads for v1.12.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes.tar.gz) | `caa332b14a6ea9d24710e3b015a91b62c04cab14bed14c49077e08bd82b8f4c1`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-src.tar.gz) | `821bdea3a52a348306fa8226bcfffa67b375cf1dd80e4be343ce0b38dd20a9a0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `58323c0a81afe53dd0dda1c6eb513caa4c82514fb6c7f0a327242e573ce80490`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `28e9344ede16890ea7848c261e461ded89c3bb2dd5b08446da04b071b48f0b02`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-386.tar.gz) | `a9eece5e0994d2ad5e07152d88787a8b5e9efcdf78983a5bafe3699e5274a9da`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `9a67750cc4243335f0c2eb89db1c4b54b0a8af08c59e2041636d0a3e946546bf`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `bbd2644f843917a3de517a53c90b327502b577fe533a9ad3da4fe6bc437c4a02`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `630946f49ef18dd43c004d99dccd9ae76390281f54740d7335c042f6f006324b`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `1d4e5cd83faf4cae8e16667576492fcd48a72f69e8fd89d599a8b555a41e90d6`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `9cefdcf21a62075b5238fda8ef2db08f81b0541ebce0e67353af1dded9e53483`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-windows-386.tar.gz) | `8b0085606ff38bded362bbe4826b5c8ee5199a33d5cbbc1b9b58f1336648ad5b`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `f44a3ec55dc7d926e681c33b5f7830c6d1cb165e24e349e426c1089b2d05a1df`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `1bf7364aa168fc251768bc850d66fef1d93f324f0ec85f6dce74080627599b70`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `dadc94fc0564cfa98add5287763bbe9c33bf8ba3eebad95fb2258c33fe8c5df3`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `2e6c8a7810705594f191b33476bf4c8fca8cebb364f0855dfea577b01fca7b7e`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `ced4a0a4e03639378eff0d3b8bfb832f5fb96be8df3e0befbdbd71373a323130`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `7e1a3fac2115c15b5baa0db04c7f319fbaaca92aa4c4588ecf62fb19812465a8`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `81d2e2f4cd3254dd345c1e921b12bff62eb96e7551336c44fb0da5407bf5fe5f`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `b14734a20190aca2b2af9cee59549d285be4f0c38faf89c5308c94534110edc1`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `ad0a81ecf6ef8346b7aa98a8d02a4f3853d0a5439d149a14b1ac2307b763b2ad`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `8e6d72837fe19afd055786c8731bd555fe082e107195c956c6985e56a03d504f`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `0fc7d55fb2750b29c0bbc36da050c8bf14508b1aa40e38e3b7f6cf311b464827`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `09bf133156b9bc474d272bf16e765b143439959a1f007283c477e7999f2b4d6a`

## Changelog since v1.12.0-alpha.1

### Action Required

* Move volume dynamic provisioning scheduling to beta (ACTION REQUIRED: The DynamicProvisioningScheduling alpha feature gate has been removed. The VolumeScheduling beta feature gate is still required for this feature) ([#67432](https://github.com/kubernetes/kubernetes/pull/67432), [@lichuqiang](https://github.com/lichuqiang))

### Other notable changes

* Not split nodes when searching for nodes but doing it all at once. ([#67555](https://github.com/kubernetes/kubernetes/pull/67555), [@wgliang](https://github.com/wgliang))
* Deprecate kubectl run generators, except for run-pod/v1 ([#68132](https://github.com/kubernetes/kubernetes/pull/68132), [@soltysh](https://github.com/soltysh))
* Using the Horizontal Pod Autoscaler with metrics from Heapster is now deprecated. ([#68089](https://github.com/kubernetes/kubernetes/pull/68089), [@DirectXMan12](https://github.com/DirectXMan12))
* Support both directory and block device for local volume plugin FileSystem VolumeMode  ([#63011](https://github.com/kubernetes/kubernetes/pull/63011), [@NickrenREN](https://github.com/NickrenREN))
* Add CSI volume attributes for kubectl describe pv. ([#65074](https://github.com/kubernetes/kubernetes/pull/65074), [@wgliang](https://github.com/wgliang))
* `kubectl rollout status` now works for unlimited timeouts. ([#67817](https://github.com/kubernetes/kubernetes/pull/67817), [@tnozicka](https://github.com/tnozicka))
* Fix panic when processing Azure HTTP response. ([#68210](https://github.com/kubernetes/kubernetes/pull/68210), [@feiskyer](https://github.com/feiskyer))
* add mixed protocol support for azure load balancer ([#67986](https://github.com/kubernetes/kubernetes/pull/67986), [@andyzhangx](https://github.com/andyzhangx))
* Replace scale down forbidden window with scale down stabilization window. Rather than waiting a fixed period of time between scale downs HPA now scales down to the highest recommendation it during the scale down stabilization window. ([#68122](https://github.com/kubernetes/kubernetes/pull/68122), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* Adding validation to kube-scheduler at the API level ([#66799](https://github.com/kubernetes/kubernetes/pull/66799), [@noqcks](https://github.com/noqcks))
* Improve performance of Pod affinity/anti-affinity in the scheduler ([#67788](https://github.com/kubernetes/kubernetes/pull/67788), [@ahmad-diaa](https://github.com/ahmad-diaa))
* kubeadm: fix air-gapped support and also allow some kubeadm commands to work without an available networking interface ([#67397](https://github.com/kubernetes/kubernetes/pull/67397), [@neolit123](https://github.com/neolit123))
* Increase Horizontal Pod Autoscaler default update interval (30s -> 15s). It will improve HPA reaction time for metric changes. ([#68021](https://github.com/kubernetes/kubernetes/pull/68021), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* Increase scrape frequency of metrics-server to 30s ([#68127](https://github.com/kubernetes/kubernetes/pull/68127), [@serathius](https://github.com/serathius))
* Add new `--server-dry-run` flag to `kubectl apply` so that the request will be sent to the server with the dry-run flag (alpha), which means that changes won't be persisted. ([#68069](https://github.com/kubernetes/kubernetes/pull/68069), [@apelisse](https://github.com/apelisse))
* kubelet v1beta1 external ComponentConfig types are now available in the `k8s.io/kubelet` repo ([#67263](https://github.com/kubernetes/kubernetes/pull/67263), [@luxas](https://github.com/luxas))
* Adds a kubelet parameter and config option to change CFS quota period from the default 100ms to some other value between 1µs and 1s. This was done to improve response latencies for workloads running in clusters with guaranteed and burstable QoS classes.   ([#63437](https://github.com/kubernetes/kubernetes/pull/63437), [@szuecs](https://github.com/szuecs))
* Enable secure serving on port 10258 to cloud-controller-manager (configurable via `--secure-port`). Delegated authentication and authorization have to be configured like for aggregated API servers. ([#67069](https://github.com/kubernetes/kubernetes/pull/67069), [@sttts](https://github.com/sttts))
* Support extra `--prune-whitelist` resources in kube-addon-manager. ([#67743](https://github.com/kubernetes/kubernetes/pull/67743), [@Random-Liu](https://github.com/Random-Liu))
* Upon receiving a LIST request with expired continue token, the apiserver now returns a continue token together with the 410 "the from parameter is too old " error. If the client does not care about getting a list from a consistent snapshot, the client can use this token to continue listing from the next key, but the returned chunk will be from the latest snapshot. ([#67284](https://github.com/kubernetes/kubernetes/pull/67284), [@caesarxuchao](https://github.com/caesarxuchao))
* Role, ClusterRole and their bindings for cloud-provider is put under system namespace. Their addonmanager mode switches to EnsureExists. ([#67224](https://github.com/kubernetes/kubernetes/pull/67224), [@grayluck](https://github.com/grayluck))
* Mount propagation has promoted to GA. The `MountPropagation` feature gate is deprecated and will be removed in 1.13. ([#67255](https://github.com/kubernetes/kubernetes/pull/67255), [@bertinatto](https://github.com/bertinatto))
* Introduce CSI Cluster Registration mechanism to ease CSI plugin discovery and allow CSI drivers to customize Kubernetes' interaction with them. ([#67803](https://github.com/kubernetes/kubernetes/pull/67803), [@saad-ali](https://github.com/saad-ali))
* Adds the commands `kubeadm alpha phases renew <cert-name>` ([#67910](https://github.com/kubernetes/kubernetes/pull/67910), [@liztio](https://github.com/liztio))
* ProcMount added to SecurityContext and AllowedProcMounts added to PodSecurityPolicy to allow paths in the container's /proc to not be masked. ([#64283](https://github.com/kubernetes/kubernetes/pull/64283), [@jessfraz](https://github.com/jessfraz))
* support cross resource group for azure file ([#68117](https://github.com/kubernetes/kubernetes/pull/68117), [@andyzhangx](https://github.com/andyzhangx))
* Port 31337 will be used by fluentd ([#68051](https://github.com/kubernetes/kubernetes/pull/68051), [@Szetty](https://github.com/Szetty))
* Improve CPU sample sanitization in HPA by taking metric's freshness into account. ([#68068](https://github.com/kubernetes/kubernetes/pull/68068), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* CoreDNS is now v1.2.2 for Kubernetes 1.12 ([#68076](https://github.com/kubernetes/kubernetes/pull/68076), [@rajansandeep](https://github.com/rajansandeep))
* Enable secure serving on port 10257 to kube-controller-manager (configurable via `--secure-port`). Delegated authentication and authorization have to be configured like for aggregated API servers. ([#64149](https://github.com/kubernetes/kubernetes/pull/64149), [@sttts](https://github.com/sttts))
* Update metrics-server to v0.3.0. ([#68077](https://github.com/kubernetes/kubernetes/pull/68077), [@DirectXMan12](https://github.com/DirectXMan12))
* TokenRequest and TokenRequestProjection are now beta features. To enable these feature, the API server needs to be started with the following flags: ([#67349](https://github.com/kubernetes/kubernetes/pull/67349), [@mikedanese](https://github.com/mikedanese))
        * --service-account-issuer
        * --service-account-signing-key-file
        * --service-account-api-audiences
* Don't let aggregated apiservers fail to launch if the external-apiserver-authentication configmap is not found in the cluster. ([#67836](https://github.com/kubernetes/kubernetes/pull/67836), [@sttts](https://github.com/sttts))
* Promote AdvancedAuditing to GA, replacing the previous (legacy) audit logging mechanisms. ([#65862](https://github.com/kubernetes/kubernetes/pull/65862), [@loburm](https://github.com/loburm))
* Azure cloud provider now supports unmanaged nodes (such as on-prem) that are labeled with `kubernetes.azure.com/managed=false` and `alpha.service-controller.kubernetes.io/exclude-balancer=true` ([#67984](https://github.com/kubernetes/kubernetes/pull/67984), [@feiskyer](https://github.com/feiskyer))
* `kubectl get apiservice` now shows the target service and whether the service is available ([#67747](https://github.com/kubernetes/kubernetes/pull/67747), [@smarterclayton](https://github.com/smarterclayton))
* Openstack supports now node shutdown taint. Taint is added when instance is shutdown in openstack. ([#67982](https://github.com/kubernetes/kubernetes/pull/67982), [@zetaab](https://github.com/zetaab))
* Return apiserver panics as 500 errors instead terminating the apiserver process. ([#68001](https://github.com/kubernetes/kubernetes/pull/68001), [@sttts](https://github.com/sttts))
* Fix VMWare VM freezing bug by reverting [#51066](https://github.com/kubernetes/kubernetes/pull/51066) ([#67825](https://github.com/kubernetes/kubernetes/pull/67825), [@nikopen](https://github.com/nikopen))
* Make CoreDNS be the default DNS server in kube-up (instead of kube-dns formerly).  ([#67569](https://github.com/kubernetes/kubernetes/pull/67569), [@fturib](https://github.com/fturib))
    * It is still possible to deploy kube-dns by setting CLUSTER_DNS_CORE_DNS=false.
* Added support to restore a volume from a volume snapshot data source.  ([#67087](https://github.com/kubernetes/kubernetes/pull/67087), [@xing-yang](https://github.com/xing-yang))
* fixes the errors/warnings in fluentd configuration ([#67947](https://github.com/kubernetes/kubernetes/pull/67947), [@saravanan30erd](https://github.com/saravanan30erd))
* Stop counting soft-deleted pods for scaling purposes in HPA controller to avoid soft-deleted pods incorrectly affecting scale up replica count calculation. ([#67067](https://github.com/kubernetes/kubernetes/pull/67067), [@moonek](https://github.com/moonek))
* delegated authn/z: optionally opt-out of mandatory authn/authz kubeconfig ([#67545](https://github.com/kubernetes/kubernetes/pull/67545), [@sttts](https://github.com/sttts))
* kubeadm: Control plane images (etcd, kube-apiserver, kube-proxy, etc.) don't use arch suffixes. Arch suffixes are kept for kube-dns only. ([#66960](https://github.com/kubernetes/kubernetes/pull/66960), [@rosti](https://github.com/rosti))
* Adds sample-cli-plugin staging repository ([#67938](https://github.com/kubernetes/kubernetes/pull/67938), [@soltysh](https://github.com/soltysh))
* adjusted http/2 buffer sizes for apiservers to prevent starvation issues between concurrent streams ([#67902](https://github.com/kubernetes/kubernetes/pull/67902), [@liggitt](https://github.com/liggitt))
* SCTP is now supported as additional protocol (alpha) alongside TCP and UDP in Pod, Service, Endpoint, and NetworkPolicy.   ([#64973](https://github.com/kubernetes/kubernetes/pull/64973), [@janosi](https://github.com/janosi))
* Always create configmaps/extensions-apiserver-authentication from kube-apiserver. ([#67694](https://github.com/kubernetes/kubernetes/pull/67694), [@sttts](https://github.com/sttts))
* kube-proxy v1beta1 external ComponentConfig types are now available in the `k8s.io/kube-proxy` repo ([#67688](https://github.com/kubernetes/kubernetes/pull/67688), [@Lion-Wei](https://github.com/Lion-Wei))
* Apply unreachable taint to a node when it lost network connection. ([#67734](https://github.com/kubernetes/kubernetes/pull/67734), [@Huang-Wei](https://github.com/Huang-Wei))
* Allow ImageReview backend to return annotations to be added to the created pod. ([#64597](https://github.com/kubernetes/kubernetes/pull/64597), [@wteiken](https://github.com/wteiken))
* Bump ip-masq-agent to v2.1.1 ([#67916](https://github.com/kubernetes/kubernetes/pull/67916), [@MrHohn](https://github.com/MrHohn))
    * - Update debian-iptables image for CVEs.
    * - Change chain name to IP-MASQ to be compatible with the
    * pre-injected masquerade rules.
* AllowedTopologies field inside StorageClass is now validated against set and map semantics. Specifically, there cannot be duplicate TopologySelectorTerms, MatchLabelExpressions keys, and TopologySelectorLabelRequirement Values. ([#66843](https://github.com/kubernetes/kubernetes/pull/66843), [@verult](https://github.com/verult))
* Introduces autoscaling/v2beta2 and custom_metrics/v1beta2, which implement metric selectors for Object and Pods metrics, as well as allowing AverageValue targets on Objects, similar to External metrics. ([#64097](https://github.com/kubernetes/kubernetes/pull/64097), [@damemi](https://github.com/damemi))
* The cloudstack cloud provider now reports a `Hostname` address type for nodes based on the `local-hostname` metadata key. ([#67719](https://github.com/kubernetes/kubernetes/pull/67719), [@liggitt](https://github.com/liggitt))
* kubeadm: --cri-socket now defaults to tcp://localhost:2375 when running on Windows ([#67447](https://github.com/kubernetes/kubernetes/pull/67447), [@benmoss](https://github.com/benmoss))
* kubeadm: The kubeadm configuration now support definition of more than one control plane instances with their own APIEndpoint. The APIEndpoint for the "bootstrap" control plane instance should be defined using `InitConfiguration.APIEndpoint`, while the APIEndpoints for additional control plane instances should be added using `JoinConfiguration.APIEndpoint`.   ([#67832](https://github.com/kubernetes/kubernetes/pull/67832), [@fabriziopandini](https://github.com/fabriziopandini))
* Enable dynamic azure disk volume limits ([#67772](https://github.com/kubernetes/kubernetes/pull/67772), [@andyzhangx](https://github.com/andyzhangx))
* kubelet: Users can now enable the alpha NodeLease feature gate to have the Kubelet create and periodically renew a Lease in the kube-node-lease namespace. The lease duration defaults to 40s, and can be configured via the kubelet.config.k8s.io/v1beta1.KubeletConfiguration's NodeLeaseDurationSeconds field. ([#66257](https://github.com/kubernetes/kubernetes/pull/66257), [@mtaufen](https://github.com/mtaufen))
* latent controller caches no longer cause repeating deletion messages for deleted pods ([#67826](https://github.com/kubernetes/kubernetes/pull/67826), [@deads2k](https://github.com/deads2k))
* API paging is now enabled for custom resource definitions, custom resources and APIService objects ([#67861](https://github.com/kubernetes/kubernetes/pull/67861), [@liggitt](https://github.com/liggitt))
* kubeadm: ControlPlaneEndpoint was moved from the API config struct to ClusterConfiguration ([#67830](https://github.com/kubernetes/kubernetes/pull/67830), [@fabriziopandini](https://github.com/fabriziopandini))
* kubeadm - feature-gates HighAvailability, SelfHosting, CertsInSecrets are now deprecated and can't be used anymore for new clusters. Update of cluster using above feature-gates flag is not supported ([#67786](https://github.com/kubernetes/kubernetes/pull/67786), [@fabriziopandini](https://github.com/fabriziopandini))
* Replace scale up forbidden window with disregarding CPU samples collected when pod was initializing. ([#67252](https://github.com/kubernetes/kubernetes/pull/67252), [@jbartosik](https://github.com/jbartosik))
* Moving KubeSchedulerConfiguration from ComponentConfig API types to staging repos ([#66916](https://github.com/kubernetes/kubernetes/pull/66916), [@dixudx](https://github.com/dixudx))
* Improved error message when checking the rollout status of StatefulSet with OnDelete strategy type ([#66983](https://github.com/kubernetes/kubernetes/pull/66983), [@mortent](https://github.com/mortent))
* RuntimeClass is a new API resource for defining different classes of runtimes that may be used to run containers in the cluster. Pods can select a RunitmeClass to use via the RuntimeClassName field. This feature is in alpha, and the RuntimeClass feature gate must be enabled in order to use it. ([#67737](https://github.com/kubernetes/kubernetes/pull/67737), [@tallclair](https://github.com/tallclair))
* Remove rescheduler since scheduling DS pods by default scheduler is moving to beta. ([#67687](https://github.com/kubernetes/kubernetes/pull/67687), [@Lion-Wei](https://github.com/Lion-Wei))
* Turn on PodReadinessGate by default ([#67406](https://github.com/kubernetes/kubernetes/pull/67406), [@freehan](https://github.com/freehan))
* Speed up kubelet start time by executing an immediate runtime and node status update when the Kubelet sees that it has a CIDR. ([#67031](https://github.com/kubernetes/kubernetes/pull/67031), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* The OpenStack cloud provider now reports a `Hostname` address type for nodes ([#67748](https://github.com/kubernetes/kubernetes/pull/67748), [@FengyunPan2](https://github.com/FengyunPan2))
* The aws cloud provider now reports a `Hostname` address type for nodes based on the `local-hostname` metadata key. ([#67715](https://github.com/kubernetes/kubernetes/pull/67715), [@liggitt](https://github.com/liggitt))
* Azure cloud provider now supports cross resource group nodes that are labeled with `kubernetes.azure.com/resource-group=<rg-name>` and `alpha.service-controller.kubernetes.io/exclude-balancer=true` ([#67604](https://github.com/kubernetes/kubernetes/pull/67604), [@feiskyer](https://github.com/feiskyer))
* Reduce API calls for Azure instance metadata. ([#67478](https://github.com/kubernetes/kubernetes/pull/67478), [@feiskyer](https://github.com/feiskyer))
* `kubectl create secret tls` can now read certificate and key files from process substitution arguments ([#67713](https://github.com/kubernetes/kubernetes/pull/67713), [@liggitt](https://github.com/liggitt))
* change default value of kind for azure disk ([#67483](https://github.com/kubernetes/kubernetes/pull/67483), [@andyzhangx](https://github.com/andyzhangx))
* To address the possibility dry-run requests overwhelming admission webhooks that rely on side effects and a reconciliation mechanism, a new field is being added to admissionregistration.k8s.io/v1beta1.ValidatingWebhookConfiguration and admissionregistration.k8s.io/v1beta1.MutatingWebhookConfiguration so that webhooks can explicitly register as having dry-run support. If a dry-run request is made on a resource that triggers a non dry-run supporting webhook, the request will be completely rejected, with "400: Bad Request". Additionally, a new field is being added to the admission.k8s.io/v1beta1.AdmissionReview API object, exposing to webhooks whether or not the request being reviewed is a dry-run. ([#66936](https://github.com/kubernetes/kubernetes/pull/66936), [@jennybuckley](https://github.com/jennybuckley))
* Kubeadm ha upgrade ([#66973](https://github.com/kubernetes/kubernetes/pull/66973), [@fabriziopandini](https://github.com/fabriziopandini))
* kubeadm: InitConfiguration now consists of two structs: InitConfiguration and ClusterConfiguration ([#67441](https://github.com/kubernetes/kubernetes/pull/67441), [@rosti](https://github.com/rosti))
* Updated Cluster Autoscaler version to 1.3.2-beta.2. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.2-beta.2 ([#67697](https://github.com/kubernetes/kubernetes/pull/67697), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* cpumanager: rollback state if updateContainerCPUSet failed ([#67430](https://github.com/kubernetes/kubernetes/pull/67430), [@choury](https://github.com/choury))
* [CRI] Adds a "runtime_handler" field to RunPodSandboxRequest, for selecting the runtime configuration to run the sandbox with (alpha feature). ([#67518](https://github.com/kubernetes/kubernetes/pull/67518), [@tallclair](https://github.com/tallclair))
* Create cli-runtime staging repository ([#67658](https://github.com/kubernetes/kubernetes/pull/67658), [@soltysh](https://github.com/soltysh))
* Headless Services with no ports defined will now create Endpoints correctly, and appear in DNS. ([#67622](https://github.com/kubernetes/kubernetes/pull/67622), [@thockin](https://github.com/thockin))
* Kubernetes juju charms will now use CSI for ceph. ([#66523](https://github.com/kubernetes/kubernetes/pull/66523), [@hyperbolic2346](https://github.com/hyperbolic2346))
* kubeadm:  Fix panic when node annotation is nil ([#67648](https://github.com/kubernetes/kubernetes/pull/67648), [@xlgao-zju](https://github.com/xlgao-zju))
* Prevent `resourceVersion` updates for custom resources on no-op writes. ([#67562](https://github.com/kubernetes/kubernetes/pull/67562), [@nikhita](https://github.com/nikhita))
* Fail container start if its requested device plugin resource hasn't registered after Kubelet restart. ([#67145](https://github.com/kubernetes/kubernetes/pull/67145), [@jiayingz](https://github.com/jiayingz))
* Use sync.map to scale ecache better ([#66862](https://github.com/kubernetes/kubernetes/pull/66862), [@resouer](https://github.com/resouer))
* DaemonSet: Fix bug- daemonset didn't create pod after node have enough resource ([#67337](https://github.com/kubernetes/kubernetes/pull/67337), [@linyouchong](https://github.com/linyouchong))
* updates kibana to 6.3.2  ([#67582](https://github.com/kubernetes/kubernetes/pull/67582), [@monotek](https://github.com/monotek))
* fixes json logging in fluentd-elasticsearch image by downgrading fluent-plugin-kubernetes_metadata_filter plugin to version 2.0.0 ([#67544](https://github.com/kubernetes/kubernetes/pull/67544), [@monotek](https://github.com/monotek))
* add --dns-loop-detect option to dnsmasq run by kube-dns ([#67302](https://github.com/kubernetes/kubernetes/pull/67302), [@dixudx](https://github.com/dixudx))
* Switched certificate data replacement from "REDACTED" to "DATA+OMITTED" ([#66023](https://github.com/kubernetes/kubernetes/pull/66023), [@ibrasho](https://github.com/ibrasho))
* improve performance of anti-affinity predicate of default scheduler. ([#66948](https://github.com/kubernetes/kubernetes/pull/66948), [@mohamed-mehany](https://github.com/mohamed-mehany))
* Fixed a bug that was blocking extensible error handling when serializing API responses error out. Previously, serialization failures always resulted in the status code of the original response being returned. Now, the following behavior occurs: ([#67041](https://github.com/kubernetes/kubernetes/pull/67041), [@tristanburgess](https://github.com/tristanburgess))
    *    - If the serialization type is application/vnd.kubernetes.protobuf, and protobuf marshaling is not implemented for the requested API resource type, a '406 Not Acceptable is returned'.
    *    - If the serialization type is 'application/json':
    *         - If serialization fails, and the original status code was an failure (e.g. 4xx or 5xx), the original status code will be returned.
    *         - If serialization fails, and the original status code was not a failure (e.g. 2xx), the status code of the serialization failure will be returned. By default, this is '500 Internal Server Error', because JSON serialization is our default, and not supposed to be implemented on a type-by-type basis.
* Add a feature to the scheduler to score fewer than all nodes in every scheduling cycle. This can improve performance of the scheduler in large clusters. ([#66733](https://github.com/kubernetes/kubernetes/pull/66733), [@bsalamat](https://github.com/bsalamat))
* kube-controller-manager can now start the quota controller when discovery results can only be partially determined. ([#67433](https://github.com/kubernetes/kubernetes/pull/67433), [@deads2k](https://github.com/deads2k))
* The plugin mechanism functionality now closely follows the git plugin design ([#66876](https://github.com/kubernetes/kubernetes/pull/66876), [@juanvallejo](https://github.com/juanvallejo))
* GCE: decrease cpu requests on master node, to allow more components to fit on one core machine. ([#67504](https://github.com/kubernetes/kubernetes/pull/67504), [@loburm](https://github.com/loburm))
* PVC may not be synced to controller local cache in time if PV is bound by external PV binder (e.g. kube-scheduler), double check if PVC is not found to prevent reclaiming PV wrongly. ([#67062](https://github.com/kubernetes/kubernetes/pull/67062), [@cofyc](https://github.com/cofyc))
* add more storage account sku support for azure disk ([#67528](https://github.com/kubernetes/kubernetes/pull/67528), [@andyzhangx](https://github.com/andyzhangx))
* updates es-image to elasticsearch 6.3.2 ([#67484](https://github.com/kubernetes/kubernetes/pull/67484), [@monotek](https://github.com/monotek))
* Bump GLBC version to 1.2.3 ([#66793](https://github.com/kubernetes/kubernetes/pull/66793), [@freehan](https://github.com/freehan))
* kube-apiserver: fixes error creating system priority classes when starting multiple apiservers simultaneously ([#67372](https://github.com/kubernetes/kubernetes/pull/67372), [@tanshanshan](https://github.com/tanshanshan))
* kubectl patch now respects --local ([#67399](https://github.com/kubernetes/kubernetes/pull/67399), [@deads2k](https://github.com/deads2k))
* Defaults for file audit logging backend in batch mode changed: ([#67223](https://github.com/kubernetes/kubernetes/pull/67223), [@tallclair](https://github.com/tallclair))
    * - Logs are written 1 at a time (no batching)
    * - Only a single writer process (lock contention)
* Forget rate limit when CRD establish controller successfully updated CRD condition ([#67370](https://github.com/kubernetes/kubernetes/pull/67370), [@yue9944882](https://github.com/yue9944882))
* updates fluentd in fluentd-elasticsearch to version 1.2.4 ([#67434](https://github.com/kubernetes/kubernetes/pull/67434), [@monotek](https://github.com/monotek))
        * also updates activesupport, fluent-plugin-elasticsearch & oj gems
* The dockershim now sets the "bandwidth" and "ipRanges" CNI capabilities (dynamic parameters). Plugin authors and administrators can now take advantage of this by updating their CNI configuration file. For more information, see the [CNI docs](https://github.com/containernetworking/cni/blob/master/CONVENTIONS.md#dynamic-plugin-specific-fields-capabilities--runtime-configuration) ([#64445](https://github.com/kubernetes/kubernetes/pull/64445), [@squeed](https://github.com/squeed))
* Expose `/debug/flags/v` to allow kubelet dynamically set glog logging level.  If want to change glog level to 3, you only have to send a PUT request like `curl -X PUT http://127.0.0.1:8080/debug/flags/v -d "3"`. ([#64601](https://github.com/kubernetes/kubernetes/pull/64601), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Fix an issue that pods using hostNetwork keep increasing. ([#67456](https://github.com/kubernetes/kubernetes/pull/67456), [@Huang-Wei](https://github.com/Huang-Wei))
* DaemonSet controller is now using backoff algorithm to avoid hot loops fighting with kubelet on pod recreation when a particular DaemonSet is misconfigured. ([#65309](https://github.com/kubernetes/kubernetes/pull/65309), [@tnozicka](https://github.com/tnozicka))
* Add node affinity for Azure unzoned managed disks ([#67229](https://github.com/kubernetes/kubernetes/pull/67229), [@feiskyer](https://github.com/feiskyer))
* Attacher/Detacher refactor for local storage ([#66884](https://github.com/kubernetes/kubernetes/pull/66884), [@NickrenREN](https://github.com/NickrenREN))
* Update debian-iptables and hyperkube-base images to include CVE fixes. ([#67365](https://github.com/kubernetes/kubernetes/pull/67365), [@ixdy](https://github.com/ixdy))
* Fix an issue where filesystems are not unmounted when a backend is not reachable and returns EIO. ([#67097](https://github.com/kubernetes/kubernetes/pull/67097), [@chakri-nelluri](https://github.com/chakri-nelluri))
* Update Cluster Autoscaler version to 1.3.2-beta.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.2-beta.1 ([#67396](https://github.com/kubernetes/kubernetes/pull/67396), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* Remove unused binary and container image for kube-aggregator. The functionality is already integrated into the kube-apiserver. ([#67157](https://github.com/kubernetes/kubernetes/pull/67157), [@dims](https://github.com/dims))
* Avoid creating new controller revisions for statefulsets when cache is stale ([#67039](https://github.com/kubernetes/kubernetes/pull/67039), [@mortent](https://github.com/mortent))
* Revert [#63905](https://github.com/kubernetes/kubernetes/pull/63905): Setup dns servers and search domains for Windows Pods. DNS for Windows containers will be set by CNI plugins. ([#66587](https://github.com/kubernetes/kubernetes/pull/66587), [@feiskyer](https://github.com/feiskyer))
* attachdetach controller attaches volumes immediately when Pod's PVCs are bound ([#66863](https://github.com/kubernetes/kubernetes/pull/66863), [@cofyc](https://github.com/cofyc))
* The check for unsupported plugins during volume resize has been moved from the admission controller to the two controllers that handle volume resize. ([#66780](https://github.com/kubernetes/kubernetes/pull/66780), [@kangarlou](https://github.com/kangarlou))
* Fix kubelet to not leak goroutines/intofiy watchers on an inactive connection if it's closed ([#67285](https://github.com/kubernetes/kubernetes/pull/67285), [@yujuhong](https://github.com/yujuhong))
* fix azure disk create failure due to sdk upgrade ([#67236](https://github.com/kubernetes/kubernetes/pull/67236), [@andyzhangx](https://github.com/andyzhangx))
* Kubeadm join --control-plane main workflow ([#66873](https://github.com/kubernetes/kubernetes/pull/66873), [@fabriziopandini](https://github.com/fabriziopandini))
* Dynamic provisions that create iSCSI PVs can ensure that multipath is used by specifying 2 or more target portals in the PV, which will cause kubelet to wait up to 10 seconds for the multipath device. PVs with just one portal continue to work as before, with kubelet not waiting for the multipath device and just using the first disk it finds. ([#67140](https://github.com/kubernetes/kubernetes/pull/67140), [@bswartz](https://github.com/bswartz))
* kubectl: recreating resources for immutable fields when force is applied ([#66602](https://github.com/kubernetes/kubernetes/pull/66602), [@dixudx](https://github.com/dixudx))
* Remove deprecated --interactive flag from kubectl logs. ([#65420](https://github.com/kubernetes/kubernetes/pull/65420), [@jsoref](https://github.com/jsoref))
* kubeadm uses audit policy v1 instead of v1beta1 ([#67176](https://github.com/kubernetes/kubernetes/pull/67176), [@charrywanganthony](https://github.com/charrywanganthony))
* kubeadm: make sure pre-pulled kube-proxy image and the one specified in its daemon set manifest are the same ([#67131](https://github.com/kubernetes/kubernetes/pull/67131), [@rosti](https://github.com/rosti))
* Graduate Resource Quota ScopeSelectors to beta, and enable it by default. ([#67077](https://github.com/kubernetes/kubernetes/pull/67077), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Decrease the amount of time it takes to modify kubeconfig files with large amounts of contexts ([#67093](https://github.com/kubernetes/kubernetes/pull/67093), [@juanvallejo](https://github.com/juanvallejo))
* Fixes issue when updating a DaemonSet causes a hash collision. ([#66476](https://github.com/kubernetes/kubernetes/pull/66476), [@mortent](https://github.com/mortent))
* fix cluster-info dump error ([#66652](https://github.com/kubernetes/kubernetes/pull/66652), [@charrywanganthony](https://github.com/charrywanganthony))
* The PodShareProcessNamespace feature to configure PID namespace sharing within a pod has been promoted to beta. ([#66507](https://github.com/kubernetes/kubernetes/pull/66507), [@verb](https://github.com/verb))
* `kubectl create {clusterrole,role}`'s `--resources` flag supports asterisk to specify all resources. ([#62945](https://github.com/kubernetes/kubernetes/pull/62945), [@nak3](https://github.com/nak3))
* Bump up version number of debian-base, debian-hyperkube-base and debian-iptables.  ([#67026](https://github.com/kubernetes/kubernetes/pull/67026), [@satyasm](https://github.com/satyasm))
    * Also updates dependencies of users of debian-base. 
    * debian-base version 0.3.1 is already available.
* DynamicProvisioningScheduling and VolumeScheduling is now supported for Azure managed disks. Feature gates DynamicProvisioningScheduling and VolumeScheduling should be enabled before using this feature. ([#67121](https://github.com/kubernetes/kubernetes/pull/67121), [@feiskyer](https://github.com/feiskyer))
* kube-apiserver now includes all registered API groups in discovery, including registered extension API group/versions for unavailable extension API servers. ([#66932](https://github.com/kubernetes/kubernetes/pull/66932), [@nilebox](https://github.com/nilebox))
* Allows extension API server to dynamically discover the requestheader CA certificate when the core API server doesn't use certificate based authentication for it's clients ([#66394](https://github.com/kubernetes/kubernetes/pull/66394), [@rtripat](https://github.com/rtripat))
* audit.k8s.io api group is upgraded from v1beta1 to v1. ([#65891](https://github.com/kubernetes/kubernetes/pull/65891), [@CaoShuFeng](https://github.com/CaoShuFeng))
    * Deprecated element metav1.ObjectMeta and Timestamp are removed from audit Events in v1 version.
    * Default value of option --audit-webhook-version and --audit-log-version will be changed from `audit.k8s.io/v1beta1` to `audit.k8s.io/v1` in release 1.13
* scope AWS LoadBalancer security group ICMP rules to spec.loadBalancerSourceRanges ([#63572](https://github.com/kubernetes/kubernetes/pull/63572), [@haz-mat](https://github.com/haz-mat))
* Add NoSchedule/NoExecute tolerations to ip-masq-agent, ensuring it to be scheduled in all nodes except master. ([#66260](https://github.com/kubernetes/kubernetes/pull/66260), [@tanshanshan](https://github.com/tanshanshan))
* The flag `--skip-preflight-checks` of kubeadm has been removed. Please use `--ignore-preflight-errors` instead. ([#62727](https://github.com/kubernetes/kubernetes/pull/62727), [@xiangpengzhao](https://github.com/xiangpengzhao))
* The watch API endpoints prefixed with `/watch` are deprecated and will be removed in a future release. These standard method for watching resources (supported since v1.0) is to use the list API endpoints with a `?watch=true` parameter. All client-go clients have used the parameter method since v1.6.0. ([#65147](https://github.com/kubernetes/kubernetes/pull/65147), [@liggitt](https://github.com/liggitt))
* Bump Heapster to v1.6.0-beta.1 ([#67074](https://github.com/kubernetes/kubernetes/pull/67074), [@kawych](https://github.com/kawych))
* kube-apiserver: setting a `dryRun` query parameter on a CONNECT request will now cause the request to be rejected, consistent with behavior of other mutating API requests. Examples of CONNECT APIs are the `nodes/proxy`, `services/proxy`, `pods/proxy`, `pods/exec`, and `pods/attach` subresources. Note that this prevents sending a `dryRun` parameter to backends via `{nodes,services,pods}/proxy` subresources. ([#66083](https://github.com/kubernetes/kubernetes/pull/66083), [@jennybuckley](https://github.com/jennybuckley))
* In clusters where the DryRun feature is enabled, dry-run requests will go through the normal admission chain. Because of this, ImagePolicyWebhook authors should especially make sure that their webhooks do not rely on side effects. ([#66391](https://github.com/kubernetes/kubernetes/pull/66391), [@jennybuckley](https://github.com/jennybuckley))
* Metadata Agent Improvements ([#66485](https://github.com/kubernetes/kubernetes/pull/66485), [@bmoyles0117](https://github.com/bmoyles0117))
    * Bump metadata agent version to 0.2-0.0.21-1.
    * Expand the metadata agent's access to all API groups.
    * Remove metadata agent config maps in favor of command line flags.
    * Update the metadata agent's liveness probe to a new /healthz handler.
    * Logging Agent Improvements
    * Bump logging agent version to 0.2-1.5.33-1-k8s-1.
    * Appropriately set log severity for k8s_container.
    * Fix detect exceptions plugin to analyze message field instead of log field.
    * Fix detect exceptions plugin to analyze streams based on local resource id.
    * Disable the metadata agent for monitored resource construction in logging.
    * Disable timestamp adjustment in logs to optimize performance.
    * Reduce logging agent buffer chunk limit to 512k to optimize performance.
* kubectl: the wait command now prints an error message and exits with the code 1, if there is no resources matching selectors ([#66692](https://github.com/kubernetes/kubernetes/pull/66692), [@m1kola](https://github.com/m1kola))
* Quota admission configuration api graduated to v1beta1 ([#66156](https://github.com/kubernetes/kubernetes/pull/66156), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Unit tests for scopes and scope selectors in the quota spec ([#66351](https://github.com/kubernetes/kubernetes/pull/66351), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Print kube-apiserver --help flag help in sections. ([#64517](https://github.com/kubernetes/kubernetes/pull/64517), [@sttts](https://github.com/sttts))
* Azure managed disks now support availability zones and new parameters `zoned`, `zone` and `zones` are added for AzureDisk storage class. ([#66553](https://github.com/kubernetes/kubernetes/pull/66553), [@feiskyer](https://github.com/feiskyer))
* nodes: improve handling of erroneous host names ([#64815](https://github.com/kubernetes/kubernetes/pull/64815), [@dixudx](https://github.com/dixudx))
* remove deprecated shorthand flag `-c` from `kubectl version (--client)` ([#66817](https://github.com/kubernetes/kubernetes/pull/66817), [@charrywanganthony](https://github.com/charrywanganthony))
* Added etcd_object_count metrics for CustomResources. ([#65983](https://github.com/kubernetes/kubernetes/pull/65983), [@sttts](https://github.com/sttts))
* Handle newlines for `command`, `args`, `env`, and `annotations` in `kubectl describe` wrapping ([#66841](https://github.com/kubernetes/kubernetes/pull/66841), [@smarterclayton](https://github.com/smarterclayton))
* Fix pod launch by kubelet when --cgroups-per-qos=false and --cgroup-driver="systemd" ([#66617](https://github.com/kubernetes/kubernetes/pull/66617), [@pravisankar](https://github.com/pravisankar))
* kubelet: fix nil pointer dereference while enforce-node-allocatable flag is not config properly ([#66190](https://github.com/kubernetes/kubernetes/pull/66190), [@linyouchong](https://github.com/linyouchong))
* Fix a bug on GCE that /etc/crictl.yaml is not generated when crictl is preloaded. ([#66877](https://github.com/kubernetes/kubernetes/pull/66877), [@Random-Liu](https://github.com/Random-Liu))
* This fix prevents a GCE PD volume from being mounted if the udev device link is stale and tries to correct the link. ([#66832](https://github.com/kubernetes/kubernetes/pull/66832), [@msau42](https://github.com/msau42))



# v1.12.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.12.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes.tar.gz) | `603345769f5e2306e5c22db928aa1cbedc6af63f387ab7a8818cb0111292133f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-src.tar.gz) | `f8fb4610cee20195381e54bfd163fbaeae228d68986817b685948b8957f324d0`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `e081c275601bcaa45d906a976d35902256f836bb60caa738a2fd8719ff3e1048`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `2dd222a267ac247dce4dfc52aff313f20c427b4351f7410aadebe8569ede3139`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `46b16d6b0429163da67b06242772c3c6c5ab9da6deda5306e63d21be04b4811d`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `8b8bf0a8a4568559d3762a72c1095ab37785fc8bbbb290aaff3a34341a24d7eb`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `d71dc60e087746b2832e66170053816dc8ed42e95efe0769ed926a6e044175ef`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `e9091bbfb997d1603dfd17ba9f145ca7dacf304f04d10230e056f8a12ce44445`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `fc6c0985ccbd806add497f2557000f7e90f3176427250e019a40e8acf7c42282`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `b8c64b318d702f6e8be76330fd5da9b87e2e4e31e904ea7e00c0cd6412ab2bcf`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `cb96e353eb5d400756a93c8d16321d0fac87d6a4f8ad89fda42858f8e4d85e9d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `003284f983cafc6fd0ce1205c03d47e638a999def1ef4e1e77bfb9149e5f598b`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `d9c282cd02c8c3fdbeb2f46abd0ddd257a8449e94be3beed2514c6e30a335a87`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `613390ba73f4236feb10bb4f70cbf96e504cf8d598da0180efc887d316b8bc5e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `1dd417f59d17c3583c6b4a3989d24c57e4989eb7b6ab9f2aa10c4cbf9bf5c11b`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `44e9e6424ed3a5a91f5adefa456b2b71c0c5d3b01be9f60f5c8c0f958815ffc1`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `3118d9c955f9a50f86ebba324894f06dbf7c1cb8f9bc5bdf6a95caf2a6678805`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `6b4d363d190e0ce6f4e41d19a0ac350b39cad7859bc442166a1da9124d1a82bb`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `c80ac005c228217b871bf3e9de032044659db3aa048cc95b101820e31d62264c`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `d8b84e7cc6ff5d0e26b045de37bdd40ca8809c303b601d8604902e5957d98621`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `b0a667c5c905e6e724fba95d44797fb52afb564aedd1c25cbd4e632e152843e9`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `78e7dbb82543ea6ac70767ed63c92823726adb6257f6b70b5911843d18288df7`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.12.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `1a3e11cc3f1a0297de2b894a43eb56ede5fbd5cdc43e4da7e61171f5c1f3ef60`

## Changelog since v1.11.0

### Action Required

* action required: the API server and client-go libraries have been fixed to support additional non-alpha-numeric characters in UserInfo "extra" data keys. Both should be updated in order to properly support extra data containing "/" characters or other characters disallowed in HTTP headers. ([#65799](https://github.com/kubernetes/kubernetes/pull/65799), [@dekkagaijin](https://github.com/dekkagaijin))
* [action required] The `NodeConfiguration` kind in the kubeadm v1alpha2 API has been renamed `JoinConfiguration` in v1alpha3 ([#65951](https://github.com/kubernetes/kubernetes/pull/65951), [@luxas](https://github.com/luxas))
* ACTION REQUIRED: Removes defaulting of CSI file system type to ext4. All the production drivers listed under https://kubernetes-csi.github.io/docs/Drivers.html were inspected and should not be impacted after this change. If you are using a driver not in that list, please test the drivers on an updated test cluster first. ``` ([#65499](https://github.com/kubernetes/kubernetes/pull/65499), [@krunaljain](https://github.com/krunaljain))
* [action required] The `MasterConfiguration` kind in the kubeadm v1alpha2 API has been renamed `InitConfiguration` in v1alpha3 ([#65945](https://github.com/kubernetes/kubernetes/pull/65945), [@luxas](https://github.com/luxas))
* [action required] The formerly publicly-available cAdvisor web UI that the kubelet started using `--cadvisor-port` is now entirely removed in 1.12. The recommended way to run cAdvisor if you still need it, is via a DaemonSet. ([#65707](https://github.com/kubernetes/kubernetes/pull/65707), [@dims](https://github.com/dims))
* Cluster Autoscaler version updated to 1.3.1-beta.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.1-beta.1 ([#65857](https://github.com/kubernetes/kubernetes/pull/65857), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
    * Default value for expendable pod priority cutoff in GCP deployment of Cluster Autoscaler changed from 0 to -10.
    * action required: users deploying workloads with priority lower than 0 may want to use priority lower than -10 to avoid triggering scale-up.
* [action required] kubeadm: The `v1alpha1` config API has been removed. ([#65628](https://github.com/kubernetes/kubernetes/pull/65628), [@luxas](https://github.com/luxas))
    * Please convert your `v1alpha1` configuration files to `v1alpha2` using the
    * `kubeadm config migrate` command of kubeadm v1.11.x
* kube-apiserver: the `Priority` admission plugin is now enabled by default when using `--enable-admission-plugins`. If using `--admission-control` to fully specify the set of admission plugins, the `Priority` admission plugin should be added if using the `PodPriority` feature, which is enabled by default in 1.11. ([#65739](https://github.com/kubernetes/kubernetes/pull/65739), [@liggitt](https://github.com/liggitt))
* The `system-node-critical` and `system-cluster-critical` priority classes are now limited to the `kube-system` namespace by the `PodPriority` admission plugin. ([#65593](https://github.com/kubernetes/kubernetes/pull/65593), [@bsalamat](https://github.com/bsalamat))
* kubernetes-worker juju charm: Added support for setting the --enable-ssl-chain-completion option on the ingress proxy.  "action required": if your installation relies on supplying incomplete certificate chains and using OCSP to fill them in, you must set "ingress-ssl-chain-completion" to "true" in your juju configuration. ([#63845](https://github.com/kubernetes/kubernetes/pull/63845), [@paulgear](https://github.com/paulgear))

### Other notable changes

* admin RBAC role now aggregates edit and view.  edit RBAC role now aggregates view.  ([#66684](https://github.com/kubernetes/kubernetes/pull/66684), [@deads2k](https://github.com/deads2k))
* Speed up HPA reaction to metric changes by removing scale up forbidden window. ([#66615](https://github.com/kubernetes/kubernetes/pull/66615), [@jbartosik](https://github.com/jbartosik))
    * Scale up forbidden window was protecting HPA against making decision to scale up based on metrics gathered during pod initialisation (which may be invalid, for example pod may be using a lot of CPU despite not doing any "actual" work).
    * To avoid that negative effect only use per pod metrics from pods that are:
    * - ready (so metrics about them should be valid), or
    * - unready but creation and last readiness change timestamps are apart more than 10s (pods that have formerly been ready and so metrics are in at least some cases (pod becoming unready because of overload) very useful).
* The `kubectl patch` command no longer exits with exit code 1 when a redundant patch results in a no-op ([#66725](https://github.com/kubernetes/kubernetes/pull/66725), [@juanvallejo](https://github.com/juanvallejo))
* Improved the output of `kubectl get events` to prioritize showing the message, and move some fields to `-o wide`. ([#66643](https://github.com/kubernetes/kubernetes/pull/66643), [@smarterclayton](https://github.com/smarterclayton))
* Added CPU Manager state validation in case of changed CPU topology. ([#66718](https://github.com/kubernetes/kubernetes/pull/66718), [@ipuustin](https://github.com/ipuustin))
* Make EBS volume expansion faster ([#66728](https://github.com/kubernetes/kubernetes/pull/66728), [@gnufied](https://github.com/gnufied))
* Kubelet serving certificate bootstrapping and rotation has been promoted to beta status. ([#66726](https://github.com/kubernetes/kubernetes/pull/66726), [@liggitt](https://github.com/liggitt))
* Flag --pod (-p shorthand) of kubectl exec command marked as deprecated ([#66558](https://github.com/kubernetes/kubernetes/pull/66558), [@quasoft](https://github.com/quasoft))
* Fixed an issue which prevented `gcloud` from working on GCE when metadata concealment was enabled. ([#66630](https://github.com/kubernetes/kubernetes/pull/66630), [@dekkagaijin](https://github.com/dekkagaijin))
* Azure Go SDK has been upgraded to v19.0.0 and VirtualMachineScaleSetVM now supports availability zones. ([#66648](https://github.com/kubernetes/kubernetes/pull/66648), [@feiskyer](https://github.com/feiskyer))
* kubeadm now can join the cluster with pre-existing client certificate if provided ([#66482](https://github.com/kubernetes/kubernetes/pull/66482), [@dixudx](https://github.com/dixudx))
* If `TaintNodesByCondition` enabled, taint node with `TaintNodeUnschedulable` when ([#63955](https://github.com/kubernetes/kubernetes/pull/63955), [@k82cn](https://github.com/k82cn))
    * initializing node to avoid race condition.
* kubeadm: remove misleading error message regarding image pulling ([#66658](https://github.com/kubernetes/kubernetes/pull/66658), [@dixudx](https://github.com/dixudx))
* Fix Stackdriver integration based on node annotation container.googleapis.com/instance_id. ([#66676](https://github.com/kubernetes/kubernetes/pull/66676), [@kawych](https://github.com/kawych))
* Fix kubelet startup failure when using ExecPlugin in kubeconfig ([#66395](https://github.com/kubernetes/kubernetes/pull/66395), [@awly](https://github.com/awly))
* When attaching iSCSI volumes, kubelet now scans only the specific ([#63176](https://github.com/kubernetes/kubernetes/pull/63176), [@bswartz](https://github.com/bswartz))
    * LUNs being attached, and also deletes them after detaching. This avoids
    * dangling references to LUNs that no longer exist, which used to be the
    * cause of random I/O errors/timeouts in kernel logs, slowdowns during
    * block-device related operations, and very rare cases of data corruption.
* kubeadm: Pull sidecar and dnsmasq-nanny images when using kube-dns ([#66499](https://github.com/kubernetes/kubernetes/pull/66499), [@rosti](https://github.com/rosti))
* Extender preemption should respect IsInterested() ([#66291](https://github.com/kubernetes/kubernetes/pull/66291), [@resouer](https://github.com/resouer))
* Properly autopopulate OpenAPI version field without needing other OpenAPI fields present in generic API server code. ([#66411](https://github.com/kubernetes/kubernetes/pull/66411), [@DirectXMan12](https://github.com/DirectXMan12))
* renamed command line option  --cri-socket-path of the kubeadm subcommand "kubeadm config images pull" to --cri-socket to be consistent with the rest of kubeadm subcommands. ([#66382](https://github.com/kubernetes/kubernetes/pull/66382), [@bart0sh](https://github.com/bart0sh))
* The --docker-disable-shared-pid kubelet flag has been removed. PID namespace sharing can instead be enable per-pod using the ShareProcessNamespace option. ([#66506](https://github.com/kubernetes/kubernetes/pull/66506), [@verb](https://github.com/verb))
* Add support for using User Assigned MSI (https://docs.microsoft.com/en-us/azure/active-directory/managed-service-identity/overview) with Kubernetes cluster on Azure. ([#66180](https://github.com/kubernetes/kubernetes/pull/66180), [@kkmsft](https://github.com/kkmsft))
* fix acr could not be listed in sp issue ([#66429](https://github.com/kubernetes/kubernetes/pull/66429), [@andyzhangx](https://github.com/andyzhangx))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63665](https://github.com/kubernetes/kubernetes/pull/63665), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* Fix volume limit for EBS on m5 and c5 instance types ([#66397](https://github.com/kubernetes/kubernetes/pull/66397), [@gnufied](https://github.com/gnufied))
* Extend TLS timeouts to work around slow arm64 math/big ([#66264](https://github.com/kubernetes/kubernetes/pull/66264), [@joejulian](https://github.com/joejulian))
* kubeadm: stop setting UID in the kubelet ConfigMap ([#66341](https://github.com/kubernetes/kubernetes/pull/66341), [@runiq](https://github.com/runiq))
* kubectl: fixes a panic displaying pods with nominatedNodeName set ([#66406](https://github.com/kubernetes/kubernetes/pull/66406), [@liggitt](https://github.com/liggitt))
* Update crictl to v1.11.1. ([#66152](https://github.com/kubernetes/kubernetes/pull/66152), [@Random-Liu](https://github.com/Random-Liu))
* fixes a panic when using a mutating webhook admission plugin with a DELETE operation ([#66425](https://github.com/kubernetes/kubernetes/pull/66425), [@liggitt](https://github.com/liggitt))
* GCE: Fixes loadbalancer creation and deletion issues appearing in 1.10.5. ([#66400](https://github.com/kubernetes/kubernetes/pull/66400), [@nicksardo](https://github.com/nicksardo))
* Azure nodes with availability zone now will have label `failure-domain.beta.kubernetes.io/zone=<region>-<zoneID>`. ([#66242](https://github.com/kubernetes/kubernetes/pull/66242), [@feiskyer](https://github.com/feiskyer))
* Re-design equivalence class cache to two level cache ([#65714](https://github.com/kubernetes/kubernetes/pull/65714), [@resouer](https://github.com/resouer))
* Checks CREATE admission for create-on-update requests instead of UPDATE admission ([#65572](https://github.com/kubernetes/kubernetes/pull/65572), [@yue9944882](https://github.com/yue9944882))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63666](https://github.com/kubernetes/kubernetes/pull/63666), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* Fixed a panic in the node status update logic when existing node has nil labels. ([#66307](https://github.com/kubernetes/kubernetes/pull/66307), [@guoshimin](https://github.com/guoshimin))
* Bump Ingress-gce version to 1.2.0 ([#65641](https://github.com/kubernetes/kubernetes/pull/65641), [@freehan](https://github.com/freehan))
* Bump event-exporter to 0.2.2 to pick up security fixes. ([#66157](https://github.com/kubernetes/kubernetes/pull/66157), [@loburm](https://github.com/loburm))
* Allow ScaleIO volumes to be provisioned without having to first manually create /dev/disk/by-id path on each kubernetes node (if not already present) ([#66174](https://github.com/kubernetes/kubernetes/pull/66174), [@ddebroy](https://github.com/ddebroy))
* fix rollout status for statefulsets ([#62943](https://github.com/kubernetes/kubernetes/pull/62943), [@faraazkhan](https://github.com/faraazkhan))
* Fix for resourcepool-path configuration in the vsphere.conf file. ([#66261](https://github.com/kubernetes/kubernetes/pull/66261), [@divyenpatel](https://github.com/divyenpatel))
* OpenAPI spec and documentation reflect 202 Accepted response path for delete request ([#63418](https://github.com/kubernetes/kubernetes/pull/63418), [@roycaihw](https://github.com/roycaihw))
* fixes a validation error that could prevent updates to StatefulSet objects containing non-normalized resource requests ([#66165](https://github.com/kubernetes/kubernetes/pull/66165), [@liggitt](https://github.com/liggitt))
* Fix validation for HealthzBindAddress in kube-proxy when --healthz-port is set to 0 ([#66138](https://github.com/kubernetes/kubernetes/pull/66138), [@wsong](https://github.com/wsong))
* kubeadm: use an HTTP request timeout when fetching the latest version of Kubernetes from dl.k8s.io ([#65676](https://github.com/kubernetes/kubernetes/pull/65676), [@dkoshkin](https://github.com/dkoshkin))
* Support configuring the Azure load balancer idle connection timeout for services ([#66045](https://github.com/kubernetes/kubernetes/pull/66045), [@cpuguy83](https://github.com/cpuguy83))
* `kubectl config set-context` can now set attributes of the current context, like the current namespace, by passing `--current` instead of a specific context name ([#66140](https://github.com/kubernetes/kubernetes/pull/66140), [@liggitt](https://github.com/liggitt))
* The alpha `Initializers` admission plugin is no longer enabled by default. This matches the off-by-default behavior of the alpha API which drives initializer behavior. ([#66039](https://github.com/kubernetes/kubernetes/pull/66039), [@liggitt](https://github.com/liggitt))
* kubeadm: Default component configs are printable via kubeadm config print-default ([#66074](https://github.com/kubernetes/kubernetes/pull/66074), [@rosti](https://github.com/rosti))
* prevents infinite CLI wait on delete when item is recreated ([#66136](https://github.com/kubernetes/kubernetes/pull/66136), [@deads2k](https://github.com/deads2k))
* Preserve vmUUID when renewing nodeinfo in vSphere cloud provider ([#66007](https://github.com/kubernetes/kubernetes/pull/66007), [@w-leads](https://github.com/w-leads))
* Cluster Autoscaler version updated to 1.3.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.1 ([#66122](https://github.com/kubernetes/kubernetes/pull/66122), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* Expose docker registry config for addons used in Juju deployments ([#66092](https://github.com/kubernetes/kubernetes/pull/66092), [@kwmonroe](https://github.com/kwmonroe))
* kubelets that specify `--cloud-provider` now only report addresses in Node status as determined by the cloud provider ([#65594](https://github.com/kubernetes/kubernetes/pull/65594), [@liggitt](https://github.com/liggitt))
        * kubelet serving certificate rotation now reacts to changes in reported node addresses, and will request certificates for addresses set by an external cloud provider
* Fix the bug where image garbage collection is disabled by mistake. ([#66051](https://github.com/kubernetes/kubernetes/pull/66051), [@jiaxuanzhou](https://github.com/jiaxuanzhou))
* fixes an issue with multi-line annotations injected via downward API files getting scrambled ([#65992](https://github.com/kubernetes/kubernetes/pull/65992), [@liggitt](https://github.com/liggitt))
* kubeadm: run kube-proxy on non-master tainted nodes ([#65931](https://github.com/kubernetes/kubernetes/pull/65931), [@neolit123](https://github.com/neolit123))
* "kubectl delete" no longer waits for dependent objects to be deleted when removing parent resources ([#65908](https://github.com/kubernetes/kubernetes/pull/65908), [@juanvallejo](https://github.com/juanvallejo))
* Introduce a new flag `--keepalive` for kubectl proxy to allow setting keep-alive period for long-running request. ([#63793](https://github.com/kubernetes/kubernetes/pull/63793), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* If Openstack LoadBalancer is not defined in cloud config, the loadbalancer is not initialized any more in openstack. All setups must have some setting under that section ([#65781](https://github.com/kubernetes/kubernetes/pull/65781), [@zetaab](https://github.com/zetaab))
* Re-adds `pkg/generated/bindata.go` to the repository to allow some parts of k8s.io/kubernetes to be go-vendorable. ([#65985](https://github.com/kubernetes/kubernetes/pull/65985), [@ixdy](https://github.com/ixdy))
* Fix a bug that preempting a pod may block forever. ([#65987](https://github.com/kubernetes/kubernetes/pull/65987), [@Random-Liu](https://github.com/Random-Liu))
* Fix flexvolume in containarized kubelets ([#65549](https://github.com/kubernetes/kubernetes/pull/65549), [@gnufied](https://github.com/gnufied))
* Add volume mode filed to constructed volume spec for CSI plugin ([#65456](https://github.com/kubernetes/kubernetes/pull/65456), [@wenlxie](https://github.com/wenlxie))
* Fix an issue with dropped audit logs, when truncating and batch backends enabled at the same time. ([#65823](https://github.com/kubernetes/kubernetes/pull/65823), [@loburm](https://github.com/loburm))
* Support traffic shaping for CNI network driver ([#63194](https://github.com/kubernetes/kubernetes/pull/63194), [@m1093782566](https://github.com/m1093782566))
* kubeadm: Use separate YAML documents for the kubelet and kube-proxy ComponentConfigs ([#65787](https://github.com/kubernetes/kubernetes/pull/65787), [@luxas](https://github.com/luxas))
* kubeadm: Fix pause image to not use architecture, as it is a manifest list ([#65920](https://github.com/kubernetes/kubernetes/pull/65920), [@dims](https://github.com/dims))
* kubeadm: print required flags when running kubeadm upgrade plan ([#65802](https://github.com/kubernetes/kubernetes/pull/65802), [@xlgao-zju](https://github.com/xlgao-zju))
* Fix `RunAsGroup` which doesn't work since 1.10. ([#65926](https://github.com/kubernetes/kubernetes/pull/65926), [@Random-Liu](https://github.com/Random-Liu))
* Running `kubectl describe pvc` now shows which pods are mounted to the pvc being described with the `Mounted By` field ([#65837](https://github.com/kubernetes/kubernetes/pull/65837), [@clandry94](https://github.com/clandry94))
* fix azure storage account creation failure ([#65846](https://github.com/kubernetes/kubernetes/pull/65846), [@andyzhangx](https://github.com/andyzhangx))
* Allow kube- and cloud-controller-manager to listen on ports up to 65535. ([#65860](https://github.com/kubernetes/kubernetes/pull/65860), [@sttts](https://github.com/sttts))
* Allow kube-scheduler to listen on ports up to 65535. ([#65833](https://github.com/kubernetes/kubernetes/pull/65833), [@sttts](https://github.com/sttts))
* kubeadm: Remove usage of `PersistentVolumeLabel` ([#65827](https://github.com/kubernetes/kubernetes/pull/65827), [@xlgao-zju](https://github.com/xlgao-zju))
* kubeadm: Add a `v1alpha3` API. ([#65629](https://github.com/kubernetes/kubernetes/pull/65629), [@luxas](https://github.com/luxas))
* Update to use go1.10.3 ([#65726](https://github.com/kubernetes/kubernetes/pull/65726), [@ixdy](https://github.com/ixdy))
* LimitRange and Endpoints resources can be created via an update API call if the object does not already exist. When this occurs, an authorization check is now made to ensure the user making the API call is authorized to create the object. In previous releases, only an update authorization check was performed. ([#65150](https://github.com/kubernetes/kubernetes/pull/65150), [@jennybuckley](https://github.com/jennybuckley))
* Fix 'kubectl cp' with no arguments causes a panic ([#65482](https://github.com/kubernetes/kubernetes/pull/65482), [@wgliang](https://github.com/wgliang))
* bazel deb package bugfix: The kubeadm deb package now reloads the kubelet after installation ([#65554](https://github.com/kubernetes/kubernetes/pull/65554), [@rdodev](https://github.com/rdodev))
* fix smb mount issue ([#65751](https://github.com/kubernetes/kubernetes/pull/65751), [@andyzhangx](https://github.com/andyzhangx))
* More fields are allowed at the root of the CRD validation schema when the status subresource is enabled. ([#65357](https://github.com/kubernetes/kubernetes/pull/65357), [@nikhita](https://github.com/nikhita))
* Reload systemd config files before starting kubelet. ([#65702](https://github.com/kubernetes/kubernetes/pull/65702), [@mborsz](https://github.com/mborsz))
* Unix: support ZFS as a valid graph driver for Docker ([#65635](https://github.com/kubernetes/kubernetes/pull/65635), [@neolit123](https://github.com/neolit123))
* Fix controller-manager crashes when flex plugin is removed from flex plugin directory ([#65536](https://github.com/kubernetes/kubernetes/pull/65536), [@gnufied](https://github.com/gnufied))
* Enable etcdv3 client prometheus metics ([#64741](https://github.com/kubernetes/kubernetes/pull/64741), [@wgliang](https://github.com/wgliang))
* skip nodes that have a primary NIC in a 'Failed' provisioningState ([#65412](https://github.com/kubernetes/kubernetes/pull/65412), [@yastij](https://github.com/yastij))
* kubeadm: remove redundant flags settings for kubelet ([#64682](https://github.com/kubernetes/kubernetes/pull/64682), [@dixudx](https://github.com/dixudx))
* Fixes the wrong elasticsearch node counter ([#65627](https://github.com/kubernetes/kubernetes/pull/65627), [@IvanovOleg](https://github.com/IvanovOleg))
* - Can configure the vsphere cloud provider with a trusted Root-CA ([#64758](https://github.com/kubernetes/kubernetes/pull/64758), [@mariantalla](https://github.com/mariantalla))
* Add Ubuntu 18.04 (Bionic) series to Juju charms ([#65644](https://github.com/kubernetes/kubernetes/pull/65644), [@tvansteenburgh](https://github.com/tvansteenburgh))
* Fix local volume directory can't be deleted because of volumeMode error ([#65310](https://github.com/kubernetes/kubernetes/pull/65310), [@wenlxie](https://github.com/wenlxie))
* kubectl: --use-openapi-print-columns is deprecated in favor of --server-print ([#65601](https://github.com/kubernetes/kubernetes/pull/65601), [@liggitt](https://github.com/liggitt))
* Add prometheus scrape port to CoreDNS service ([#65589](https://github.com/kubernetes/kubernetes/pull/65589), [@rajansandeep](https://github.com/rajansandeep))
* fixes an out of range panic in the NoExecuteTaintManager controller when running a non-64-bit build ([#65596](https://github.com/kubernetes/kubernetes/pull/65596), [@liggitt](https://github.com/liggitt))
* kubectl: fixes a regression with --use-openapi-print-columns that would not print object contents ([#65600](https://github.com/kubernetes/kubernetes/pull/65600), [@liggitt](https://github.com/liggitt))
* Hostnames are now converted to lowercase before being used for node lookups in the kubernetes-worker charm. ([#65487](https://github.com/kubernetes/kubernetes/pull/65487), [@dshcherb](https://github.com/dshcherb))
* N/A ([#64660](https://github.com/kubernetes/kubernetes/pull/64660), [@figo](https://github.com/figo))
* bugfix: Do not print feature gates in the generic apiserver code for glog level 0 ([#65584](https://github.com/kubernetes/kubernetes/pull/65584), [@neolit123](https://github.com/neolit123))
* Add metrics for PVC in-use ([#64527](https://github.com/kubernetes/kubernetes/pull/64527), [@gnufied](https://github.com/gnufied))
* Fixed exception detection in fluentd-gcp plugin. ([#65361](https://github.com/kubernetes/kubernetes/pull/65361), [@xperimental](https://github.com/xperimental))
* api-machinery utility functions `SetTransportDefaults` and `DialerFor` once again respect custom Dial functions set on transports ([#65547](https://github.com/kubernetes/kubernetes/pull/65547), [@liggitt](https://github.com/liggitt))
* Improve the display of jobs in `kubectl get` and `kubectl describe` to emphasize progress and duration. ([#65463](https://github.com/kubernetes/kubernetes/pull/65463), [@smarterclayton](https://github.com/smarterclayton))
* kubectl convert previous created a list inside of a list.  Now it is only wrapped once. ([#65489](https://github.com/kubernetes/kubernetes/pull/65489), [@deads2k](https://github.com/deads2k))
* fix azure disk creation issue when specifying external resource group ([#65516](https://github.com/kubernetes/kubernetes/pull/65516), [@andyzhangx](https://github.com/andyzhangx))
* fixes a regression in kube-scheduler to properly load client connection information from a `--config` file that references a kubeconfig file ([#65507](https://github.com/kubernetes/kubernetes/pull/65507), [@liggitt](https://github.com/liggitt))
* Fixed cleanup of CSI metadata files. ([#65323](https://github.com/kubernetes/kubernetes/pull/65323), [@jsafrane](https://github.com/jsafrane))
* Update Rescheduler's manifest to use version 0.4.0. ([#65454](https://github.com/kubernetes/kubernetes/pull/65454), [@bsalamat](https://github.com/bsalamat))
* On COS, NPD creates a node condition for frequent occurrences of unregister_netdevice ([#65342](https://github.com/kubernetes/kubernetes/pull/65342), [@dashpole](https://github.com/dashpole))
* Properly manage security groups for loadbalancer services on OpenStack. ([#65373](https://github.com/kubernetes/kubernetes/pull/65373), [@multi-io](https://github.com/multi-io))
* Add user-agent to audit-logging. ([#64812](https://github.com/kubernetes/kubernetes/pull/64812), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* kubeadm: notify the user of manifest upgrade timeouts ([#65164](https://github.com/kubernetes/kubernetes/pull/65164), [@xlgao-zju](https://github.com/xlgao-zju))
* Fixes incompatibility with custom scheduler extender configurations specifying `bindVerb` ([#65424](https://github.com/kubernetes/kubernetes/pull/65424), [@liggitt](https://github.com/liggitt))
* Using `kubectl describe` on CRDs that use underscores will be prettier. ([#65391](https://github.com/kubernetes/kubernetes/pull/65391), [@smarterclayton](https://github.com/smarterclayton))
* Improve scheduler's performance by eliminating sorting of nodes by their score. ([#65396](https://github.com/kubernetes/kubernetes/pull/65396), [@bsalamat](https://github.com/bsalamat))
* Add more conditions to the list of predicate failures that won't be resolved by preemption. ([#64995](https://github.com/kubernetes/kubernetes/pull/64995), [@bsalamat](https://github.com/bsalamat))
* Allow access to ClusterIP from the host network namespace when kube-proxy is started in IPVS mode without either masqueradeAll or clusterCIDR flags ([#65388](https://github.com/kubernetes/kubernetes/pull/65388), [@lbernail](https://github.com/lbernail))
* User can now use `sudo crictl` on GCE cluster. ([#65389](https://github.com/kubernetes/kubernetes/pull/65389), [@Random-Liu](https://github.com/Random-Liu))
* Tolerate missing watch permission when deleting a resource ([#65370](https://github.com/kubernetes/kubernetes/pull/65370), [@deads2k](https://github.com/deads2k))
* Prevents a `kubectl delete` hang when deleting controller managed lists ([#65367](https://github.com/kubernetes/kubernetes/pull/65367), [@deads2k](https://github.com/deads2k))
* fixes a memory leak in the kube-controller-manager observed when large numbers of pods with tolerations are created/deleted ([#65339](https://github.com/kubernetes/kubernetes/pull/65339), [@liggitt](https://github.com/liggitt))
* checkLimitsForResolvConf for the  pod create and update events instead of checking period ([#64860](https://github.com/kubernetes/kubernetes/pull/64860), [@wgliang](https://github.com/wgliang))
* Fix concurrent map access panic ([#65334](https://github.com/kubernetes/kubernetes/pull/65334), [@dashpole](https://github.com/dashpole))
    * Don't watch .mount cgroups to reduce number of inotify watches
    * Fix NVML initialization race condition
    * Fix brtfs disk metrics when using a subdirectory of a subvolume
* Change Azure ARM Rate limiting error message. ([#65292](https://github.com/kubernetes/kubernetes/pull/65292), [@wgliang](https://github.com/wgliang))
* AWS now checks for validity of ecryption key when creating encrypted volumes. Dynamic provisioning of encrypted volume may get slower due to these checks. ([#65223](https://github.com/kubernetes/kubernetes/pull/65223), [@jsafrane](https://github.com/jsafrane))
* Report accurate status for kubernetes-master and -worker charms. ([#65187](https://github.com/kubernetes/kubernetes/pull/65187), [@kwmonroe](https://github.com/kwmonroe))
* Fixed issue 63608, which is that under rare circumstances the ResourceQuota admission controller could lose track of an request in progress and time out after waiting 10 seconds for a decision to be made. ([#64598](https://github.com/kubernetes/kubernetes/pull/64598), [@MikeSpreitzer](https://github.com/MikeSpreitzer))
* In the vSphere cloud provider the `Global.vm-uuid` configuration option is not deprecated anymore, it can be used to overwrite the VMUUID on the controller-manager ([#65152](https://github.com/kubernetes/kubernetes/pull/65152), [@alvaroaleman](https://github.com/alvaroaleman))
* fluentd-gcp grace termination period increased to 60s. ([#65084](https://github.com/kubernetes/kubernetes/pull/65084), [@x13n](https://github.com/x13n))
* Pass cluster_location argument to Heapster ([#65176](https://github.com/kubernetes/kubernetes/pull/65176), [@kawych](https://github.com/kawych))
* Fix a scalability issue where high rates of event writes degraded etcd performance. ([#64539](https://github.com/kubernetes/kubernetes/pull/64539), [@ccding](https://github.com/ccding))
* Corrected a mistake in the documentation for wait.PollImmediate(...) ([#65026](https://github.com/kubernetes/kubernetes/pull/65026), [@spew](https://github.com/spew))
* Split 'scheduling_latency_seconds' metric into finer steps (predicate, priority, premption) ([#65306](https://github.com/kubernetes/kubernetes/pull/65306), [@shyamjvs](https://github.com/shyamjvs))
* Etcd health checks by the apiserver now ensure the apiserver can connect to and exercise the etcd API ([#65027](https://github.com/kubernetes/kubernetes/pull/65027), [@liggitt](https://github.com/liggitt))
* Add e2e regression tests for the kubelet being secure ([#64140](https://github.com/kubernetes/kubernetes/pull/64140), [@dixudx](https://github.com/dixudx))
* set EnableHTTPSTrafficOnly in azure storage account creation ([#64957](https://github.com/kubernetes/kubernetes/pull/64957), [@andyzhangx](https://github.com/andyzhangx))
* Fixes an issue where Portworx PVCs remain in pending state when created using a StorageClass with empty parameters ([#64895](https://github.com/kubernetes/kubernetes/pull/64895), [@harsh-px](https://github.com/harsh-px))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63662](https://github.com/kubernetes/kubernetes/pull/63662), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63661](https://github.com/kubernetes/kubernetes/pull/63661), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63660](https://github.com/kubernetes/kubernetes/pull/63660), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* Updated default image for nginx ingress in CDK to match current Kubernetes docs. ([#64285](https://github.com/kubernetes/kubernetes/pull/64285), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Added block volume support to Cinder volume plugin. ([#64879](https://github.com/kubernetes/kubernetes/pull/64879), [@bertinatto](https://github.com/bertinatto))
* fixed incorrect OpenAPI schema for CustomResourceDefinition objects ([#65256](https://github.com/kubernetes/kubernetes/pull/65256), [@liggitt](https://github.com/liggitt))
* ignore not found file error when watching manifests ([#64880](https://github.com/kubernetes/kubernetes/pull/64880), [@dixudx](https://github.com/dixudx))
* add port-forward examples for sevice ([#64773](https://github.com/kubernetes/kubernetes/pull/64773), [@MasayaAoyama](https://github.com/MasayaAoyama))
* Fix issues for block device not mapped to container. ([#64555](https://github.com/kubernetes/kubernetes/pull/64555), [@wenlxie](https://github.com/wenlxie))
* Update crictl on GCE to v1.11.0. ([#65254](https://github.com/kubernetes/kubernetes/pull/65254), [@Random-Liu](https://github.com/Random-Liu))
* Fixes missing nodes lines when kubectl top nodes ([#64389](https://github.com/kubernetes/kubernetes/pull/64389), [@yue9944882](https://github.com/yue9944882))
* keep pod state consistent when scheduler cache UpdatePod ([#64692](https://github.com/kubernetes/kubernetes/pull/64692), [@adohe](https://github.com/adohe))
* add external resource group support for azure disk ([#64427](https://github.com/kubernetes/kubernetes/pull/64427), [@andyzhangx](https://github.com/andyzhangx))
* Increase the gRPC max message size to 16MB in the remote container runtime. ([#64672](https://github.com/kubernetes/kubernetes/pull/64672), [@mcluseau](https://github.com/mcluseau))
* The new default value for the --allow-privileged parameter of the Kubernetes-worker charm has been set to true based on changes which went into the Kubernetes 1.10 release. Before this change the default value was set to false. If you're installing Canonical Kubernetes you should expect this value to now be true by default and you should now look to use PSP (pod security policies).  ([#64104](https://github.com/kubernetes/kubernetes/pull/64104), [@CalvinHartwell](https://github.com/CalvinHartwell))
* The --remove-extra-subjects and --remove-extra-permissions flags have been enabled for kubectl auth reconcile ([#64541](https://github.com/kubernetes/kubernetes/pull/64541), [@mrogers950](https://github.com/mrogers950))
* Fix kubectl drain --timeout option when eviction is used. ([#64378](https://github.com/kubernetes/kubernetes/pull/64378), [@wrdls](https://github.com/wrdls))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63659](https://github.com/kubernetes/kubernetes/pull/63659), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.

