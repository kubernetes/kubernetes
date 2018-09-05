<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.11.3](#v1113)
  - [Downloads for v1.11.3](#downloads-for-v1113)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.11.2](#changelog-since-v1112)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
- [v1.11.2](#v1112)
  - [Downloads for v1.11.2](#downloads-for-v1112)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.11.1](#changelog-since-v1111)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-1)
- [v1.11.1](#v1111)
  - [Downloads for v1.11.1](#downloads-for-v1111)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.11.0](#changelog-since-v1110)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-2)
- [v1.11.0](#v1110)
  - [Downloads for v1.11.0](#downloads-for-v1110)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
- [Kubernetes 1.11 Release Notes](#kubernetes-111-release-notes)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST do this before you upgrade)](#no-really-you-must-do-this-before-you-upgrade)
  - [Major Themes](#major-themes)
    - [SIG API Machinery](#sig-api-machinery)
    - [SIG Auth](#sig-auth)
    - [SIG CLI](#sig-cli)
    - [SIG Cluster Lifecycle](#sig-cluster-lifecycle)
    - [SIG Instrumentation](#sig-instrumentation)
    - [SIG Network](#sig-network)
    - [SIG Node](#sig-node)
    - [SIG OpenStack](#sig-openstack)
    - [SIG Scheduling](#sig-scheduling)
    - [SIG Storage](#sig-storage)
    - [SIG Windows](#sig-windows)
  - [Known Issues](#known-issues)
  - [Before Upgrading](#before-upgrading)
      - [New Deprecations](#new-deprecations)
      - [Removed Deprecations](#removed-deprecations)
      - [Graduated to Stable/GA](#graduated-to-stablega)
      - [Graduated to Beta](#graduated-to-beta)
    - [New alpha features](#new-alpha-features)
  - [Other Notable Changes](#other-notable-changes-3)
    - [SIG API Machinery](#sig-api-machinery-1)
    - [SIG Apps](#sig-apps)
    - [SIG Auth](#sig-auth-1)
    - [SIG Autoscaling](#sig-autoscaling)
    - [SIG Azure](#sig-azure)
    - [SIG CLI](#sig-cli-1)
    - [SIG Cluster Lifecycle](#sig-cluster-lifecycle-1)
    - [SIG GCP](#sig-gcp)
    - [SIG Instrumentation](#sig-instrumentation-1)
    - [SIG Network](#sig-network-1)
    - [SIG Node](#sig-node-1)
    - [SIG OpenStack](#sig-openstack-1)
    - [SIG Scheduling](#sig-scheduling-1)
    - [SIG Storage](#sig-storage-1)
    - [SIG vSphere](#sig-vsphere)
    - [SIG Windows](#sig-windows-1)
    - [Additional changes](#additional-changes)
  - [External Dependencies](#external-dependencies)
  - [Bug Fixes](#bug-fixes)
      - [General Fixes and Reliability](#general-fixes-and-reliability)
  - [Non-user-facing changes](#non-user-facing-changes)
- [v1.11.0-rc.3](#v1110-rc3)
  - [Downloads for v1.11.0-rc.3](#downloads-for-v1110-rc3)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.11.0-rc.2](#changelog-since-v1110-rc2)
    - [Other notable changes](#other-notable-changes-4)
- [v1.11.0-rc.2](#v1110-rc2)
  - [Downloads for v1.11.0-rc.2](#downloads-for-v1110-rc2)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.11.0-rc.1](#changelog-since-v1110-rc1)
    - [Other notable changes](#other-notable-changes-5)
- [v1.11.0-rc.1](#v1110-rc1)
  - [Downloads for v1.11.0-rc.1](#downloads-for-v1110-rc1)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.11.0-beta.2](#changelog-since-v1110-beta2)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-6)
- [v1.11.0-beta.2](#v1110-beta2)
  - [Downloads for v1.11.0-beta.2](#downloads-for-v1110-beta2)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
  - [Changelog since v1.11.0-beta.1](#changelog-since-v1110-beta1)
    - [Action Required](#action-required-4)
    - [Other notable changes](#other-notable-changes-7)
- [v1.11.0-beta.1](#v1110-beta1)
  - [Downloads for v1.11.0-beta.1](#downloads-for-v1110-beta1)
    - [Client Binaries](#client-binaries-8)
    - [Server Binaries](#server-binaries-8)
    - [Node Binaries](#node-binaries-8)
  - [Changelog since v1.11.0-alpha.2](#changelog-since-v1110-alpha2)
    - [Action Required](#action-required-5)
    - [Other notable changes](#other-notable-changes-8)
- [v1.11.0-alpha.2](#v1110-alpha2)
  - [Downloads for v1.11.0-alpha.2](#downloads-for-v1110-alpha2)
    - [Client Binaries](#client-binaries-9)
    - [Server Binaries](#server-binaries-9)
    - [Node Binaries](#node-binaries-9)
  - [Changelog since v1.11.0-alpha.1](#changelog-since-v1110-alpha1)
    - [Other notable changes](#other-notable-changes-9)
- [v1.11.0-alpha.1](#v1110-alpha1)
  - [Downloads for v1.11.0-alpha.1](#downloads-for-v1110-alpha1)
    - [Client Binaries](#client-binaries-10)
    - [Server Binaries](#server-binaries-10)
    - [Node Binaries](#node-binaries-10)
  - [Changelog since v1.10.0](#changelog-since-v1100)
    - [Action Required](#action-required-6)
    - [Other notable changes](#other-notable-changes-10)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.11.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.11/examples)

## Downloads for v1.11.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes.tar.gz) | `032fd2483176aea999cc92a455676cf1dbf70538e916fedaa6b851b50c6009c3`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-src.tar.gz) | `bbb1541985eaa3d2ccb5d627196423298e3a2581adb21f0aa3c8650691780e3f`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-darwin-386.tar.gz) | `983ad8892a392373009fef3aee823aec5c23bec789eb386e889aa5163b887b27`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-darwin-amd64.tar.gz) | `8e4b0f9cc1918e22c44468c667b2a362d933216ae37cc4d9e770a3e5c8d0ea35`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-linux-386.tar.gz) | `48ae3e0c9ce4b6964abe5a93cc0677f39e7e23f2ae09ff965e5fb8a807ab930f`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-linux-amd64.tar.gz) | `14a70ac05c00fcfd7d632fc9e7a5fbc6615ce1b370bb1a0e506a24972d461493`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-linux-arm.tar.gz) | `c62835797d58b50f19706d897a9106b219f5868b0a6c7bb62c6284f809c01473`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-linux-arm64.tar.gz) | `a43510f821d349519ecba27b24788a0e41eae31d79bc7af73b6132190b0dcce2`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-linux-ppc64le.tar.gz) | `11aca4d23d16b676576c7a78b4ad444b2b478ff60f7ab8838885d5e2c28836da`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-linux-s390x.tar.gz) | `1dd9204e91f9865fbf85da90f7afbcca6cf00df145627f7d500d9ec865d67f95`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-windows-386.tar.gz) | `e72c2495f9bf8aff2ee8010125371ed00bb048ebd9d800fe3cd7dabf10c23528`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-client-windows-amd64.tar.gz) | `d1bd2cf1ef21753d572060882c258b30b286bb4376edc7970b0798b40d5c05e8`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-server-linux-amd64.tar.gz) | `e49d0db1791555d73add107d2110d54487df538b35b9dde0c5590ac4c5e9e039`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-server-linux-arm.tar.gz) | `7b0932dc1c48352fa300b5eec66c2222772c5ccadee2d0187558b55d5c780037`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-server-linux-arm64.tar.gz) | `9e89136c2e84f2e5e709b84123166f149babb6c3f17efe5a607f7166c2a5ee79`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-server-linux-ppc64le.tar.gz) | `ffc9b8eb6e421d64c355629c6e1a9bf775ec11ad89a1817983461dc6821dd308`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-server-linux-s390x.tar.gz) | `0dbb55cd80ac62d3c8c9859a1d6cb0b55b04ad4b3a880f0c9ff1f1040db6333b`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-node-linux-amd64.tar.gz) | `14761794210f7b17a0a51ea9746e55eb1d538c7b86601b98e1df562e68f96024`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-node-linux-arm.tar.gz) | `578232ffbd559997b0d8302afbe4ef1f425507c8642f8d72f00f5cb3c07c2655`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-node-linux-arm64.tar.gz) | `d827b87664c5f6fa627fb1294cb816d5f7bc7a959da659319b51e37f0ece2142`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-node-linux-ppc64le.tar.gz) | `a60b6deccaa6a25276d7ce9a08a038df0aad8f4566ce05a71f4e084d609e6803`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-node-linux-s390x.tar.gz) | `5435a7040f0f289619176625084c8ec7c400481931d18b061a7a2697225fd23d`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.3/kubernetes-node-windows-amd64.tar.gz) | `b8c9353388390bf256fb8d3d32cc03cd1828aa5f38f94883d41edaec3de21ae7`

## Changelog since v1.11.2

### Action Required

* action required: the API server and client-go libraries have been fixed to support additional non-alpha-numeric characters in UserInfo "extra" data keys. Both should be updated in order to properly support extra data containing "/" characters or other characters disallowed in HTTP headers. ([#65799](https://github.com/kubernetes/kubernetes/pull/65799), [@dekkagaijin](https://github.com/dekkagaijin))

### Other notable changes

* Orphan DaemonSet when deleting with --cascade option set ([#68007](https://github.com/kubernetes/kubernetes/pull/68007), [@soltysh](https://github.com/soltysh))
* Bump event-exporter to 0.2.2 to pick up security fixes. ([#66157](https://github.com/kubernetes/kubernetes/pull/66157), [@loburm](https://github.com/loburm))
* PVC may not be synced to controller local cache in time if PV is bound by external PV binder (e.g. kube-scheduler), double check if PVC is not found to prevent reclaiming PV wrongly. ([#67062](https://github.com/kubernetes/kubernetes/pull/67062), [@cofyc](https://github.com/cofyc))
* kube-apiserver: fixes error creating system priority classes when starting multiple apiservers simultaneously ([#67372](https://github.com/kubernetes/kubernetes/pull/67372), [@tanshanshan](https://github.com/tanshanshan))
* Add NoSchedule/NoExecute tolerations to ip-masq-agent, ensuring it to be scheduled in all nodes except master. ([#66260](https://github.com/kubernetes/kubernetes/pull/66260), [@tanshanshan](https://github.com/tanshanshan))
* Prevent `resourceVersion` updates for custom resources on no-op writes. ([#67562](https://github.com/kubernetes/kubernetes/pull/67562), [@nikhita](https://github.com/nikhita))
* fix cluster-info dump error ([#66652](https://github.com/kubernetes/kubernetes/pull/66652), [@charrywanganthony](https://github.com/charrywanganthony))
* attachdetach controller attaches volumes immediately when Pod's PVCs are bound ([#66863](https://github.com/kubernetes/kubernetes/pull/66863), [@cofyc](https://github.com/cofyc))
* kube-apiserver now includes all registered API groups in discovery, including registered extension API group/versions for unavailable extension API servers. ([#66932](https://github.com/kubernetes/kubernetes/pull/66932), [@nilebox](https://github.com/nilebox))
* fixes an issue with multi-line annotations injected via downward API files getting scrambled ([#65992](https://github.com/kubernetes/kubernetes/pull/65992), [@liggitt](https://github.com/liggitt))
* Revert [#63905](https://github.com/kubernetes/kubernetes/pull/63905): Setup dns servers and search domains for Windows Pods. DNS for Windows containers will be set by CNI plugins. ([#66587](https://github.com/kubernetes/kubernetes/pull/66587), [@feiskyer](https://github.com/feiskyer))
* Fix an issue that pods using hostNetwork keep increasing. ([#67456](https://github.com/kubernetes/kubernetes/pull/67456), [@Huang-Wei](https://github.com/Huang-Wei))
* Bump Heapster to v1.6.0-beta.1 ([#67074](https://github.com/kubernetes/kubernetes/pull/67074), [@kawych](https://github.com/kawych))
* fix azure disk create failure due to sdk upgrade ([#67236](https://github.com/kubernetes/kubernetes/pull/67236), [@andyzhangx](https://github.com/andyzhangx))
* Fix creation of custom resources when the CRD contains non-conventional pluralization and subresources ([#66249](https://github.com/kubernetes/kubernetes/pull/66249), [@deads2k](https://github.com/deads2k))
* Allows extension API server to dynamically discover the requestheader CA certificate when the core API server doesn't use certificate based authentication for it's clients ([#66394](https://github.com/kubernetes/kubernetes/pull/66394), [@rtripat](https://github.com/rtripat))



# v1.11.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes.tar.gz) | `deaabdab00003ef97309cde188a61db8c2737721efeb75f7b19c6240863d9cbe`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-src.tar.gz) | `7d98defe2eb909bc5ebcbddd8d55ffaa902d5d3c76d9e42fe0a092c341016a6e`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-darwin-386.tar.gz) | `bee271bf7b251428fe5cf21fd586cc43dfd2c842932f4d9e0b1b549546dc7a2a`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-darwin-amd64.tar.gz) | `fc1d506c63b48997100aa6099b5d1b019dcd41bb962805f34273ada9a6b2252c`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-linux-386.tar.gz) | `bc7d637eff280bacf2059a487ca374169e63cf311808eb8e8b7504fc8a28a07d`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-linux-amd64.tar.gz) | `f7444a2200092ca6b4cd92408810ae88506fb7a27f762298fafb009b031250e3`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-linux-arm.tar.gz) | `d976e2ca975d5142364df996e4e9597817b65ab6bd0212bde807ec82f7345879`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-linux-arm64.tar.gz) | `7641599ee2d8743f861ff723cf54d3f01732f354729f4c8c101cad5ceeeb0e62`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-linux-ppc64le.tar.gz) | `ceea9bb4f218b3cd7346b44c56ffc7113540ceb1eb59e34df503b281722516a9`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-linux-s390x.tar.gz) | `e1b3ae15e84c8f911537c5e8af0d79d5187ded344fc3330e9d43f22dff3073bb`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-windows-386.tar.gz) | `5ea3c0aba710df3c95bb59621c8b220989ac45698b5268a75b17144d361d338c`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-client-windows-amd64.tar.gz) | `0499fe3b55fa6252c6c3611c3b22e8f65bf3d2aebcde7b847f7572a539ac9d70`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-server-linux-amd64.tar.gz) | `1323f4d58522e4480641948299f4804e09c20357682bc12d547f78a60c920836`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-server-linux-arm.tar.gz) | `a0d145a1794da4ae1018ad2b14c74e564be3c0c13514a3ed32c6177723d2e41f`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-server-linux-arm64.tar.gz) | `8f4419ec15470e450094b24299ecae21eb59fc8308009b5b726b98730f7e13a2`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-server-linux-ppc64le.tar.gz) | `e40aebbaa17976699125913fd6a7d8b66ab343e02c7da674bd4256d1de029950`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-server-linux-s390x.tar.gz) | `924d2b177e3646be59f6411b574d99e32ca74e5b7419e12e8642bc085d735560`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-node-linux-amd64.tar.gz) | `d254aa910a0dc47e587bec5a5fa8544c743835ba6d9ca5627d9d11a2df90c561`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-node-linux-arm.tar.gz) | `d1104c040efbdca831d5af584ad1482a2ade883af198aefef99ad1884794cefd`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-node-linux-arm64.tar.gz) | `c983f7e35aaee922865ab5d4d7893f168e026c3216dbc87658ce1bf6d8e67e06`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-node-linux-ppc64le.tar.gz) | `b8f784a4b142450bb990ab80a76950c8816e96e4b6e15134d0af5bd48f62c9f3`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-node-linux-s390x.tar.gz) | `b541c476b797f346b5ea68437904995c02d55e089433bdca6fd40531034fc838`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.2/kubernetes-node-windows-amd64.tar.gz) | `c3e91f4b30b801cc942bd0f160dc1afd01bebc3d2f11e69f373b1116409ffb44`

## Changelog since v1.11.1

### Action Required

* Cluster Autoscaler version updated to 1.3.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.1 ([#66123](https://github.com/kubernetes/kubernetes/pull/66123), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
    * Default value for expendable pod priority cutoff in GCP deployment of Cluster Autoscaler changed from 0 to -10.
    * action required: users deploying workloads with priority lower than 0 may want to use priority lower than -10 to avoid triggering scale-up.

### Other notable changes

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
* Fix a bug on GCE that /etc/crictl.yaml is not generated when crictl is preloaded. ([#66877](https://github.com/kubernetes/kubernetes/pull/66877), [@Random-Liu](https://github.com/Random-Liu))
* Fix validation for HealthzBindAddress in kube-proxy when --healthz-port is set to 0 ([#66138](https://github.com/kubernetes/kubernetes/pull/66138), [@wsong](https://github.com/wsong))
* fix acr could not be listed in sp issue ([#66429](https://github.com/kubernetes/kubernetes/pull/66429), [@andyzhangx](https://github.com/andyzhangx))
* Fixed an issue which prevented `gcloud` from working on GCE when metadata concealment was enabled. ([#66630](https://github.com/kubernetes/kubernetes/pull/66630), [@dekkagaijin](https://github.com/dekkagaijin))
* Fix for resourcepool-path configuration in the vsphere.conf file. ([#66261](https://github.com/kubernetes/kubernetes/pull/66261), [@divyenpatel](https://github.com/divyenpatel))
* This fix prevents a GCE PD volume from being mounted if the udev device link is stale and tries to correct the link. ([#66832](https://github.com/kubernetes/kubernetes/pull/66832), [@msau42](https://github.com/msau42))
* Fix kubelet startup failure when using ExecPlugin in kubeconfig ([#66395](https://github.com/kubernetes/kubernetes/pull/66395), [@awly](https://github.com/awly))
* GCE: Fixes loadbalancer creation and deletion issues appearing in 1.10.5. ([#66400](https://github.com/kubernetes/kubernetes/pull/66400), [@nicksardo](https://github.com/nicksardo))
* kubeadm: Pull sidecar and dnsmasq-nanny images when using kube-dns ([#66499](https://github.com/kubernetes/kubernetes/pull/66499), [@rosti](https://github.com/rosti))
* fix smb mount issue ([#65751](https://github.com/kubernetes/kubernetes/pull/65751), [@andyzhangx](https://github.com/andyzhangx))
* Extend TLS timeouts to work around slow arm64 math/big ([#66264](https://github.com/kubernetes/kubernetes/pull/66264), [@joejulian](https://github.com/joejulian))
* Allow ScaleIO volumes to be provisioned without having to first manually create /dev/disk/by-id path on each kubernetes node (if not already present) ([#66174](https://github.com/kubernetes/kubernetes/pull/66174), [@ddebroy](https://github.com/ddebroy))
* kubeadm: stop setting UID in the kubelet ConfigMap ([#66341](https://github.com/kubernetes/kubernetes/pull/66341), [@runiq](https://github.com/runiq))
* fixes a panic when using a mutating webhook admission plugin with a DELETE operation ([#66425](https://github.com/kubernetes/kubernetes/pull/66425), [@liggitt](https://github.com/liggitt))
* Update crictl to v1.11.1. ([#66152](https://github.com/kubernetes/kubernetes/pull/66152), [@Random-Liu](https://github.com/Random-Liu))
* kubectl: fixes a panic displaying pods with nominatedNodeName set ([#66406](https://github.com/kubernetes/kubernetes/pull/66406), [@liggitt](https://github.com/liggitt))
* fixes a validation error that could prevent updates to StatefulSet objects containing non-normalized resource requests ([#66165](https://github.com/kubernetes/kubernetes/pull/66165), [@liggitt](https://github.com/liggitt))
* Tolerate missing watch permission when deleting a resource ([#65370](https://github.com/kubernetes/kubernetes/pull/65370), [@deads2k](https://github.com/deads2k))
* prevents infinite CLI wait on delete when item is recreated ([#66136](https://github.com/kubernetes/kubernetes/pull/66136), [@deads2k](https://github.com/deads2k))
* Preserve vmUUID when renewing nodeinfo in vSphere cloud provider ([#66007](https://github.com/kubernetes/kubernetes/pull/66007), [@w-leads](https://github.com/w-leads))



# v1.11.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes.tar.gz) | `77d93c4ab10b1c4421835ebf3c81dc9c6d2a798949ee9132418e24d500c22d6e`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-src.tar.gz) | `e597a3a73f4c4933e9fb145d398adfc4e245e4465bbea50b0e55c78d2b0e70ef`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-darwin-386.tar.gz) | `d668a91a52ad9c0a95a94172f89b42b42ca8f9eafe4ac479a97fe2e11f5dbd8e`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-darwin-amd64.tar.gz) | `5d6ce0f956b789840baf207b6d2bb252a4f8f0eaf6981207eb7df25e39871452`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-linux-386.tar.gz) | `1e47c66db3b7a194327f1d3082b657140d4cfee09eb03162a658d0604c31028e`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-linux-amd64.tar.gz) | `a6c7537434fedde75fb77c593b2d2978be1aed00896a354120c5b7164e54aa99`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-linux-arm.tar.gz) | `6eed4c3f11eb844947344e283482eeeb38a4b59eb8e24174fb706e997945ce12`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-linux-arm64.tar.gz) | `c260ee179420ce396ab972ab1252a26431c50b5412de2466ede1fb506d5587af`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-linux-ppc64le.tar.gz) | `01ec89ebbeb2b673504bb629e6a20793c31e29fc9b96100796533c391f3b13f2`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-linux-s390x.tar.gz) | `28b171b63d5c49d0d64006d331daba0ef6e9e841d69c3588bb3502eb122ef76a`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-windows-386.tar.gz) | `9ee394cadd909a937aef5c82c3499ae12da226ccbaa21f6d82c4878b7cb31d6c`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-client-windows-amd64.tar.gz) | `ab2c21e627a2fab52193ad7af0aabc001520975aac35660dc5f857320176e6c4`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-server-linux-amd64.tar.gz) | `f120baa4b37323a8d7cd6e8027f7b19a544f528d2cae4028366ffbb28dc68d8a`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-server-linux-arm.tar.gz) | `eac27b81cf2819619fdda54a83f06aecf77aefef1f2f2accd7adcc725cb607ff`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-server-linux-arm64.tar.gz) | `25d87248f0da9ba71a4e6c5d1b7af2259ffd43435715d52db6044ebe85466fad`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-server-linux-ppc64le.tar.gz) | `7eba9021f93b6f99167cd088933aabbf11d5a6f990d796fc1b884ed97e066a3b`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-server-linux-s390x.tar.gz) | `144fa932ab4bea9e810958dd859fdf9b11a9f90918c22b2c9322b6c21b5c82ed`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-node-linux-amd64.tar.gz) | `45fae35f7c3b23ff8557dcf638eb631dabbcc46a804534ca9d1043d846ec4408`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-node-linux-arm.tar.gz) | `19c29a635807979a87dcac610f79373df8ee90de823cf095362dcca086844831`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-node-linux-arm64.tar.gz) | `35b9a5fa8671c46b9c175a4920dce269fccf84b1defdbccb24e76c4eab9fb255`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-node-linux-ppc64le.tar.gz) | `b4a111ee652b42c9d92288d4d86f4897af524537b9409b1f5cedefb4122bb2d6`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-node-linux-s390x.tar.gz) | `4730b9d81cdde078c17c0831b1b20eeda65f4df37e0f595accc63bd2c1635bae`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.1/kubernetes-node-windows-amd64.tar.gz) | `d7fdf0341efe3d6a80a295aae19874a4099644c7ddba5fa34bee3a6924e0840b`

## Changelog since v1.11.0

### Action Required

* ACTION REQUIRED: Removes defaulting of CSI file system type to ext4. All the production drivers listed under https://kubernetes-csi.github.io/docs/Drivers.html were inspected and should not be impacted after this change. If you are using a driver not in that list, please test the drivers on an updated test cluster first. ``` ([#65499](https://github.com/kubernetes/kubernetes/pull/65499), [@krunaljain](https://github.com/krunaljain))
* kube-apiserver: the `Priority` admission plugin is now enabled by default when using `--enable-admission-plugins`. If using `--admission-control` to fully specify the set of admission plugins, the `Priority` admission plugin should be added if using the `PodPriority` feature, which is enabled by default in 1.11. ([#65739](https://github.com/kubernetes/kubernetes/pull/65739), [@liggitt](https://github.com/liggitt))
* The `system-node-critical` and `system-cluster-critical` priority classes are now limited to the `kube-system` namespace by the `PodPriority` admission plugin. ([#65593](https://github.com/kubernetes/kubernetes/pull/65593), [@bsalamat](https://github.com/bsalamat))

### Other notable changes

* kubeadm: run kube-proxy on non-master tainted nodes ([#65931](https://github.com/kubernetes/kubernetes/pull/65931), [@neolit123](https://github.com/neolit123))
* Fix an issue with dropped audit logs, when truncating and batch backends enabled at the same time. ([#65823](https://github.com/kubernetes/kubernetes/pull/65823), [@loburm](https://github.com/loburm))
* set EnableHTTPSTrafficOnly in azure storage account creation ([#64957](https://github.com/kubernetes/kubernetes/pull/64957), [@andyzhangx](https://github.com/andyzhangx))
* Re-adds `pkg/generated/bindata.go` to the repository to allow some parts of k8s.io/kubernetes to be go-vendorable. ([#65985](https://github.com/kubernetes/kubernetes/pull/65985), [@ixdy](https://github.com/ixdy))
* Fix a scalability issue where high rates of event writes degraded etcd performance. ([#64539](https://github.com/kubernetes/kubernetes/pull/64539), [@ccding](https://github.com/ccding))
* "kubectl delete" no longer waits for dependent objects to be deleted when removing parent resources ([#65908](https://github.com/kubernetes/kubernetes/pull/65908), [@juanvallejo](https://github.com/juanvallejo))
* Update to use go1.10.3 ([#65726](https://github.com/kubernetes/kubernetes/pull/65726), [@ixdy](https://github.com/ixdy))
* Fix the bug where image garbage collection is disabled by mistake. ([#66051](https://github.com/kubernetes/kubernetes/pull/66051), [@jiaxuanzhou](https://github.com/jiaxuanzhou))
* Fix `RunAsGroup` which doesn't work since 1.10. ([#65926](https://github.com/kubernetes/kubernetes/pull/65926), [@Random-Liu](https://github.com/Random-Liu))
* Fixed cleanup of CSI metadata files. ([#65323](https://github.com/kubernetes/kubernetes/pull/65323), [@jsafrane](https://github.com/jsafrane))
* Reload systemd config files before starting kubelet. ([#65702](https://github.com/kubernetes/kubernetes/pull/65702), [@mborsz](https://github.com/mborsz))
* Fix a bug that preempting a pod may block forever. ([#65987](https://github.com/kubernetes/kubernetes/pull/65987), [@Random-Liu](https://github.com/Random-Liu))
* fixes a regression in kubectl printing behavior when using go-template or jsonpath output that resulted in a "unable to match a printer" error message ([#65979](https://github.com/kubernetes/kubernetes/pull/65979), [@juanvallejo](https://github.com/juanvallejo))
* add external resource group support for azure disk ([#64427](https://github.com/kubernetes/kubernetes/pull/64427), [@andyzhangx](https://github.com/andyzhangx))
* Properly manage security groups for loadbalancer services on OpenStack. ([#65373](https://github.com/kubernetes/kubernetes/pull/65373), [@multi-io](https://github.com/multi-io))
* Allow access to ClusterIP from the host network namespace when kube-proxy is started in IPVS mode without either masqueradeAll or clusterCIDR flags ([#65388](https://github.com/kubernetes/kubernetes/pull/65388), [@lbernail](https://github.com/lbernail))
* kubeadm: Fix pause image to not use architecture, as it is a manifest list ([#65920](https://github.com/kubernetes/kubernetes/pull/65920), [@dims](https://github.com/dims))
* bazel deb package bugfix: The kubeadm deb package now reloads the kubelet after installation ([#65554](https://github.com/kubernetes/kubernetes/pull/65554), [@rdodev](https://github.com/rdodev))
* The garbage collector now supports CustomResourceDefinitions and APIServices. ([#65915](https://github.com/kubernetes/kubernetes/pull/65915), [@nikhita](https://github.com/nikhita))
* fix azure storage account creation failure ([#65846](https://github.com/kubernetes/kubernetes/pull/65846), [@andyzhangx](https://github.com/andyzhangx))
* skip nodes that have a primary NIC in a 'Failed' provisioningState ([#65412](https://github.com/kubernetes/kubernetes/pull/65412), [@yastij](https://github.com/yastij))
* Fix 'kubectl cp' with no arguments causes a panic ([#65482](https://github.com/kubernetes/kubernetes/pull/65482), [@wgliang](https://github.com/wgliang))
* On COS, NPD creates a node condition for frequent occurrences of unregister_netdevice ([#65342](https://github.com/kubernetes/kubernetes/pull/65342), [@dashpole](https://github.com/dashpole))
* bugfix: Do not print feature gates in the generic apiserver code for glog level 0 ([#65584](https://github.com/kubernetes/kubernetes/pull/65584), [@neolit123](https://github.com/neolit123))
* Add prometheus scrape port to CoreDNS service ([#65589](https://github.com/kubernetes/kubernetes/pull/65589), [@rajansandeep](https://github.com/rajansandeep))
* kubectl: fixes a regression with --use-openapi-print-columns that would not print object contents ([#65600](https://github.com/kubernetes/kubernetes/pull/65600), [@liggitt](https://github.com/liggitt))
* fixes an out of range panic in the NoExecuteTaintManager controller when running a non-64-bit build ([#65596](https://github.com/kubernetes/kubernetes/pull/65596), [@liggitt](https://github.com/liggitt))
* Fixed issue 63608, which is that under rare circumstances the ResourceQuota admission controller could lose track of an request in progress and time out after waiting 10 seconds for a decision to be made. ([#64598](https://github.com/kubernetes/kubernetes/pull/64598), [@MikeSpreitzer](https://github.com/MikeSpreitzer))



# v1.11.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes.tar.gz) | `3c779492574a5d8ce702d89915184f5dd52280da909abf134232e5ab00b4a885`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-src.tar.gz) | `f0b2d8e61860acaf50a9bae0dc36b8bfdb4bb41b8d0a1bb5a9bc3d87aad3b794`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-darwin-386.tar.gz) | `196738ef058510438b3129f0a72544544b7d52a8732948b4f9358781f87dab59`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-darwin-amd64.tar.gz) | `9ec8357b10b79f8fd87f3a836879d0a4bb46fb70adbb82f1e34dc7e91d74999f`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-linux-386.tar.gz) | `e8ee8a965d3ea241d9768b9ac868ecbbee112ef45038ff219e4006fa7f4ab4e2`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-linux-amd64.tar.gz) | `d31377c92b4cc9b3da086bc1974cbf57b0d2c2b22ae789ba84cf1b7554ea7067`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-linux-arm.tar.gz) | `9e9da909293a4682a5d6270a39894b056b3e901532b15eb8fdc0814a8d628d65`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-linux-arm64.tar.gz) | `149df9daac3e596042f5759977f9f9299a397130d9dddc2d4a2b513dd64f1092`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-linux-ppc64le.tar.gz) | `ff3d3e4714406d92e9a2b7ef2887519800b89f6592a756524f7a37dc48057f44`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-linux-s390x.tar.gz) | `e5a39bdc1e474d9d00974a81101e043aaff37c30c1418fb85a0c2561465e14c7`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-windows-386.tar.gz) | `4ba1102a33c6d4df650c4864a118f99a9882021fea6f250a35f4b4f4a2d68eaa`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-client-windows-amd64.tar.gz) | `0bb74af7358f9a2f4139ed1c10716a2f5f0c1c13ab3af71a0621a1983233c8d7`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-server-linux-amd64.tar.gz) | `b8a8a88afd8a40871749b2362dbb21295c6a9c0a85b6fc87e7febea1688eb99e`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-server-linux-arm.tar.gz) | `88b9168013bb07a7e17ddc0638e7d36bcd2984d049a50a96f54cb4218647d8da`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-server-linux-arm64.tar.gz) | `12fab9e9f0e032f278c0e114c72ea01899a0430fc772401f23e26de306e0f59f`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-server-linux-ppc64le.tar.gz) | `6616d726a651e733cfd4cccd78bfdc1d421c4a446edf4b617b8fd8f5e21f073e`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-server-linux-s390x.tar.gz) | `291838980929c8073ac592219d9576c84a9bdf233585966c81a380c3d753316e`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-node-linux-amd64.tar.gz) | `b23e905efb828fdffc4efc208f7343236b22c964e408fe889f529502aed4a335`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-node-linux-arm.tar.gz) | `44bf8973581887a2edd33eb637407e76dc0dc3a5abcc2ff04aec8338b533156d`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-node-linux-arm64.tar.gz) | `51e481c782233b46ee21e9635c7d8c2a84450cbe30d7b1cbe5c5982b33f40b13`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-node-linux-ppc64le.tar.gz) | `d1a3feda31a954d3a83193a51a117873b6ef9f8acc3e10b3f1504fece91f2eb8`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-node-linux-s390x.tar.gz) | `0ad76c6e6aef670c215256803b3b0d19f4730a0843429951c6421564c73d4932`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0/kubernetes-node-windows-amd64.tar.gz) | `8ad26200ed40d40a1b78d7a5dbe56220f0813d31194f40f267b476499fe2c5c3`

# Kubernetes 1.11 Release Notes

## Urgent Upgrade Notes
### (No, really, you MUST do this before you upgrade)

Before upgrading to Kubernetes 1.11, you must keep the following in mind:

* **JSON configuration files that contain fields with incorrect case will no longer be valid. You must correct these files before upgrading.** When specifying keys in JSON resource definitions during direct API server communication, the keys are case-sensitive. A bug introduced in Kubernetes 1.8 caused the API server to accept a request with incorrect case and coerce it to correct case, but this behaviour has been fixed in 1.11 and the API server will once again be enforcing the correct case. It’s worth noting that during this time, the `kubectl` tool continued to enforce case-sensitive keys, so users that strictly manage resources with `kubectl` will be unaffected by this change. ([#65034](https://github.com/kubernetes/kubernetes/pull/65034), [@caesarxuchao](https://github.com/caesarxuchao))
* **[Pod priority and preemption](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/) is now enabled by default.** Note that this means that pods from *any* namespace can now request priority classes that compete with and/or cause preemption of critical system pods that are already running. If that is not desired, disable the PodPriority feature by setting `--feature-gates=PodPriority=false` on the kube-apiserver, kube-scheduler, and kubelet components before upgrading to 1.11. Disabling the PodPriority feature limits [critical pods](https://kubernetes.io/docs/tasks/administer-cluster/guaranteed-scheduling-critical-addon-pods/#marking-pod-as-critical-when-priorites-are-enabled) to the `kube-system` namespace.

## Major Themes

### SIG API Machinery

This release SIG API Machinery focused mainly on CustomResources. For example, subresources for CustomResources are now beta and enabled by default. With this, updates to the `/status` subresource will disallow updates to all fields other than `.status` (not just `.spec` and `.metadata` as before). Also, `required` and `description` can be used at the root of the CRD OpenAPI validation schema when the `/status` subresource is enabled.

In addition, users can now create multiple versions of CustomResourceDefinitions, but without any kind of automatic conversion, and CustomResourceDefinitions now allow specification of additional columns for `kubectl get` output via the `spec.additionalPrinterColumns` field.

### SIG Auth

Work this cycle focused on graduating existing functions, and on making security functions more understandable for users.

RBAC [cluster role aggregation](https://kubernetes.io/docs/reference/access-authn-authz/rbac/#aggregated-clusterroles), introduced in 1.9, graduated to stable status with no changes in 1.11, and [client-go credential plugins](https://kubernetes.io/docs/reference/access-authn-authz/authentication/#client-go-credential-plugins)  graduated to beta status, while also adding support for obtaining TLS credentials from an external plugin.

Kubernetes 1.11 also makes it easier to see what's happening, as audit events can now be annotated with information about how an API request was handled:
  * Authorization sets `authorization.k8s.io/decision` and `authorization.k8s.io/reason` annotations with the authorization decision ("allow" or "forbid") and a human-readable description of why the decision was made (for example, RBAC includes the name of the role/binding/subject which allowed a request).
  * PodSecurityPolicy admission sets `podsecuritypolicy.admission.k8s.io/admit-policy` and `podsecuritypolicy.admission.k8s.io/validate-policy` annotations containing the name of the policy that allowed a pod to be admitted. (PodSecurityPolicy also gained the ability to [limit hostPath volume mounts to be read-only](https://kubernetes.io/docs/concepts/policy/pod-security-policy/#volumes-and-file-systems).)

In addition, the NodeRestriction admission plugin now prevents kubelets from modifying taints on their Node API objects, making it easier to keep track of which nodes should be in use.

### SIG CLI

SIG CLI's main focus this release was on refactoring `kubectl` internals to improve composability, readability and testability of `kubectl` commands. Those refactors will allow the team to extract a mechanism for extensibility of kubectl -- that is, plugins -- in the next releases.

### SIG Cluster Lifecycle

SIG Cluster Lifecycle focused on improving kubeadm’s user experience by including a set of new commands related to maintaining the kubeadm configuration file, the API version of which has now has been incremented to `v1alpha2`. These commands can handle the migration of the configuration to a newer version, printing the default configuration, and listing and pulling the required container images for bootstrapping a cluster.

Other notable changes include:
* CoreDNS replaces kube-dns as the default DNS provider
* Improved user experience for environments without a public internet connection and users using other CRI runtimes than Docker
* Support for structured configuration for the kubelet, which avoids the need to modify the systemd drop-in file
* Many improvements to the upgrade process and other bug fixes

### SIG Instrumentation

As far as Sig Instrumentation, the major change in Kubernetes 1.11 is the deprecation of Heapster as part of ongoing efforts to move to the new Kubernetes monitoring model. Clusters still using Heapster for autoscaling should be migrated over to metrics-server and the custom metrics API. See the deprecation section for more information.

### SIG Network

The main milestones for SIG Network this release are the graduation of IPVS-based load balancing and CoreDNS to general availability.

IPVS is an alternative approach to in-cluster load balancing that uses in-kernel hash tables rather than the previous iptables approach, while CoreDNS is a replacement for kube-dns for service discovery.

### SIG Node

SIG-Node advanced several features and made incremental improvements in a few key topic areas this release.

The dynamic kubelet config feature graduated to beta, so it is enabled by default, simplifying management of the node object itself.  Kubelets that are configured to work with the CRI may take advantage of the log rotation feature, which is graduating to beta this release.

The cri-tools project, which aims to provide consistent tooling for operators to debug and introspect their nodes in production independent of their chosen container runtime, graduated to GA.

As far as platforms, working with SIG-Windows, enhancements were made to the kubelet to improve platform support on Windows operating systems, and improvements to resource management were also made.  In particular, support for sysctls on Linux graduated to beta.

### SIG OpenStack

SIG-OpenStack continued to build out testing, with eleven acceptance tests covering a wide-range of scenarios and use-cases. During the 1.11 cycle our reporting back to test-grid has qualified the OpenStack cloud provider as a gating job for the Kubernetes release.

New features include improved integration between the Keystone service and Kubernetes RBAC, and a number of stability and compatibility improvements across the entire provider code-base.

### SIG Scheduling
[Pod Priority and Preemption](https://kubernetes.io/docs/concepts/configuration/pod-priority-preemption/) has graduated to Beta, so it is enabled by default. Note that this involves [significant and important changes for operators](https://github.com/kubernetes/sig-release/pull/201/files). The team also worked on improved performance and reliability of the scheduler.

### SIG Storage

Sig Storage graduated two features that had been introduced in previous versions and introduced three new features in an alpha state.

The StorageProtection feature, which prevents deletion of PVCs while Pods are still using them and of PVs while still bound to a PVC, is now generally available, and volume resizing, which lets you increase size of a volume after a Pod restarts is now beta, which means it is on by default.

New alpha features include:
* Online volume resizing will increase the filesystem size of a resized volume without requiring a Pod restart.
* AWS EBS and GCE PD volumes support increased limits on the maximum number of attached volumes per node.
* Subpath volume directories can be created using DownwardAPI environment variables.

### SIG Windows

This release supports more of Kubernetes API for pods and containers on Windows, including:

* Metrics for Pod, Container, Log filesystem
* The run_as_user security contexts
* Local persistent volumes and fstype for Azure disk

Improvements in Windows Server version 1803 also bring new storage functionality to Kubernetes v1.11, including:

* Volume mounts for ConfigMap and Secret
* Flexvolume plugins for SMB and iSCSI storage are also available out-of-tree at [Microsoft/K8s-Storage-Plugins](https://github.com/Microsoft/K8s-Storage-Plugins)

## Known Issues

* IPVS based kube-proxy doesn't support graceful close connections for terminating pod. This issue will be fixed in a future release. ([#57841](https://github.com/kubernetes/kubernetes/pull/57841), [@jsravn](https://github.com/jsravn))
* kube-proxy needs to be configured to override hostname in some environments. ([#857](https://github.com/kubernetes/kubeadm/issues/857), [@detiber](https://github.com/detiber))
* There's a known issue where the Vertical Pod Autoscaler will radically change implementation in 1.12, so users of VPA (alpha) in 1.11 are warned that they will not be able to automatically migrate their VPA configs from 1.11 to 1.12.


## Before Upgrading

* When Response is a `metav1.Status`, it is no longer copied into the audit.Event status. Only the "status", "reason" and "code" fields are set.  For example, when we run `kubectl get pods abc`, the API Server returns a status object:
```{"kind":"Status","apiVersion":"v1","metadata":{},"status":"Failure","message":"pods \"abc\" not found","reason":"NotFound","details":{"name":"abc","kind":"pods"},"code":404}```
In previous versions, the whole object was logged in audit events. Starting in 1.11, only `status`, `reason`, and `code` are logged. Code that relies on the older version must be updated to avoid errors.
([#62695](https://github.com/kubernetes/kubernetes/pull/62695), [@CaoShuFeng](https://github.com/CaoShuFeng))
* HTTP transport now uses `context.Context` to cancel dial operations. k8s.io/client-go/transport/Config struct has been updated to accept a function with a `context.Context` parameter. This is a breaking change if you use this field in your code. ([#60012](https://github.com/kubernetes/kubernetes/pull/60012), [@ash2k](https://github.com/ash2k))
* kubectl: This client version requires the `apps/v1` APIs, so it will not work against a cluster version older than v1.9.0. Note that kubectl only guarantees compatibility with clusters that are +/-1 minor version away. ([#61419](https://github.com/kubernetes/kubernetes/pull/61419), [@enisoc](https://github.com/enisoc))
* Pod priority and preemption is now enabled by default. Even if you don't plan to use this feature, you might need to take some action immediately after upgrading. In multi-tenant clusters where not all users are trusted, you are advised to create appropriate quotas for two default priority classes, system-cluster-critical and system-node-critical, which are added to clusters by default. `ResourceQuota` should be created to limit users from creating Pods at these priorities if not all users of your cluster are trusted. We do not advise disabling this feature because critical system Pods rely on the scheduler preemption to be scheduled when cluster is under resource pressure.
* Default mount propagation has changed from `HostToContainer` ("rslave" in Linux terminology), as it was in 1.10, to `None` ("private") to match the behavior in 1.9 and earlier releases; `HostToContainer` as a default caused regressions in some pods. If you are relying on this behavior you will need to set it explicitly. ([#62462](https://github.com/kubernetes/kubernetes/pull/62462), [@jsafrane](https://github.com/jsafrane))
* The kube-apiserver `--storage-version` flag has been removed; you must use `--storage-versions` instead. ([#61453](https://github.com/kubernetes/kubernetes/pull/61453), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Authors of aggregated API servers must not rely on authorization being done by the kube-apiserver, and must do delegated authorization in addition. ([#61349](https://github.com/kubernetes/kubernetes/pull/61349), [@sttts](https://github.com/sttts))
* GC is now bound by QPS so if you need more QPS to avoid ratelimiting GC, you'll have to set it explicitly. ([#63657](https://github.com/kubernetes/kubernetes/pull/63657), [@shyamjvs](https://github.com/shyamjvs))
* `kubeadm join` is now blocking on the kubelet performing the TLS Bootstrap properly. Earlier, `kubeadm join` only did the discovery part and exited successfully without checking that the kubelet actually started properly and performed the TLS bootstrap correctly. Now, as kubeadm runs some post-join steps (for example, annotating the Node API object with the CRISocket), `kubeadm join` is now waiting for the kubelet to perform the TLS Bootstrap, and then uses that credential to perform further actions. This also improves the UX, as `kubeadm` will exit with a non-zero code if the kubelet isn't in a functional state, instead of pretending everything's fine.
 ([#64792](https://github.com/kubernetes/kubernetes/pull/64792), [@luxas](https://github.com/luxas))
* The structure of the kubelet dropin in the kubeadm deb package has changed significantly. Instead of hard-coding the parameters for the kubelet in the dropin, a structured configuration file for the kubelet is used, and is expected to be present in `/var/lib/kubelet/config.yaml`. For runtime-detected, instance-specific configuration values, a environment file with dynamically-generated flags at `kubeadm init` or `kubeadm join` run time is used. Finally, if you want to override something specific for the kubelet that can't be done via the kubeadm Configuration file (which is preferred), you might add flags to the `KUBELET_EXTRA_ARGS` environment variable in either `/etc/default/kubelet`
or `/etc/sysconfig/kubelet`, depending on the system you're running on.
([#64780](https://github.com/kubernetes/kubernetes/pull/64780), [@luxas](https://github.com/luxas))
* The `--node-name` flag for kubeadm now dictates the Node API object name the kubelet uses for registration, in all cases but where you might use an in-tree cloud provider. If you're not using an in-tree cloud provider, `--node-name` will set the Node API object name. If you're using an in-tree cloud provider, you MUST make `--node-name` match the name the in-tree cloud provider decides to use.
([#64706](https://github.com/kubernetes/kubernetes/pull/64706), [@liztio](https://github.com/liztio))
* The `PersistentVolumeLabel` admission controller is now disabled by default. If you depend on this feature (AWS/GCE) then ensure it is added to the `--enable-admission-plugins` flag on the kube-apiserver. ([#64326](https://github.com/kubernetes/kubernetes/pull/64326), [@andrewsykim](https://github.com/andrewsykim))
* kubeadm: kubelets in kubeadm clusters now disable the readonly port (10255). If you're relying on unauthenticated access to the readonly port, please switch to using the secure port (10250). Instead, you can now use ServiceAccount tokens when talking to the secure port, which will make it easier to get access to, for example, the `/metrics` endpoint of the kubelet, securely. ([#64187](https://github.com/kubernetes/kubernetes/pull/64187), [@luxas](https://github.com/luxas))
* The formerly publicly-available cAdvisor web UI that the kubelet ran on port 4194 by default is now turned off by default. The flag configuring what port to run this UI on `--cadvisor-port` was deprecated in v1.10. Now the default is `--cadvisor-port=0`, in other words, to not run the web server. If you still need to run cAdvisor, the recommended way to run it is via a DaemonSet. Note that the `--cadvisor-port` will be removed in v1.12 ([#63881](https://github.com/kubernetes/kubernetes/pull/63881), [@luxas](https://github.com/luxas))

#### New Deprecations

* As a reminder, etcd2 as a backend is deprecated and support will be removed in Kubernetes 1.13. Please ensure that your clusters are upgraded to etcd3 as soon as possible.
* InfluxDB cluster monitoring has been deprecated as part of the deprecation of Heapster. Instead, you may use the [metrics server](https://github.com/kubernetes-incubator/metrics-server). It's a simplified heapster that is able to gather and serve current metrics values. It provides the Metrics API that is used by `kubectl top`, and horizontal pod autoscaler. Note that it doesn't include some features of Heapster, such as short term metrics for graphs in kube-dashboard and dedicated push sinks, which proved hard to maintain and scale. Clusters using Heapster for transferring metrics into long-term storage should consider using their metric solution's native Kubernetes support, if present, or should consider alternative solutions. ([#62328](https://github.com/kubernetes/kubernetes/pull/62328), [@serathius](https://github.com/serathius))
* The kubelet `--rotate-certificates` flag is now deprecated, and will be removed in a future release. The kubelet certificate rotation feature can now be enabled via the `.RotateCertificates` field in the kubelet's config file.  ([#63912](https://github.com/kubernetes/kubernetes/pull/63912), [@luxas](https://github.com/luxas))
* The kubeadm configuration file version has been upgraded from `v1alpha1` to `v1alpha2`. `v1alpha1` read support exists in v1.11, but will be removed in v1.12. ([#63788](https://github.com/kubernetes/kubernetes/pull/63788), [@luxas](https://github.com/luxas))
The following PRs changed the API spec:
  * In the new v1alpha2 kubeadm Configuration API, the `.CloudProvider` and `.PrivilegedPods` fields don't exist anymore. Instead, you should use the out-of-tree cloud provider implementations, which are beta in v1.11.
  * If you have to use the legacy in-tree cloud providers, you can rearrange your config like the example below. If you need the `cloud-config` file (located in `{cloud-config-path}`), you can mount it into the API Server and controller-manager containers using ExtraVolumes, as in:
```
kind: MasterConfiguration
apiVersion: kubeadm.k8s.io/v1alpha2
apiServerExtraArgs:
  cloud-provider: "{cloud}"
  cloud-config: "{cloud-config-path}"
apiServerExtraVolumes:
- name: cloud
  hostPath: "{cloud-config-path}"
  mountPath: "{cloud-config-path}"
controllerManagerExtraArgs:
  cloud-provider: "{cloud}"
  cloud-config: "{cloud-config-path}"
controllerManagerExtraVolumes:
- name: cloud
  hostPath: "{cloud-config-path}"
  mountPath: "{cloud-config-path}"
```
* If you need to use the `.PrivilegedPods` functionality, you can still edit the manifests in `/etc/kubernetes/manifests/`, and set `.SecurityContext.Privileged=true` for the apiserver and controller manager.
 ([#63866](https://github.com/kubernetes/kubernetes/pull/63866), [@luxas](https://github.com/luxas))
 * kubeadm: The Token-related fields in the `MasterConfiguration` object have now been refactored. Instead of the top-level `.Token`, `.TokenTTL`, `.TokenUsages`, `.TokenGroups` fields, there is now a `BootstrapTokens` slice of `BootstrapToken` objects that support the same features under the `.Token`, `.TTL`, `.Usages`, `.Groups` fields. ([#64408](https://github.com/kubernetes/kubernetes/pull/64408), [@luxas](https://github.com/luxas))
  * `.NodeName` and `.CRISocket` in the `MasterConfiguration` and `NodeConfiguration` v1alpha1 API objects are now `.NodeRegistration.Name` and `.NodeRegistration.CRISocket` respectively in the v1alpha2 API. The `.NoTaintMaster` field has been removed in the v1alpha2 API. ([#64210](https://github.com/kubernetes/kubernetes/pull/64210), [@luxas](https://github.com/luxas))
  * kubeadm: Support for `.AuthorizationModes` in the kubeadm v1alpha2 API has been removed. Instead, you can use the `.APIServerExtraArgs` and `.APIServerExtraVolumes` fields to achieve the same effect. Files using the v1alpha1 API and setting this field will be automatically upgraded to this v1alpha2 API and the information will be preserved. ([#64068](https://github.com/kubernetes/kubernetes/pull/64068), [@luxas](https://github.com/luxas))
* The annotation `service.alpha.kubernetes.io/tolerate-unready-endpoints` is deprecated.  Users should use Service.spec.publishNotReadyAddresses instead. ([#63742](https://github.com/kubernetes/kubernetes/pull/63742), [@thockin](https://github.com/thockin))
* `--show-all`, which only affected pods, and even then only for human readable/non-API printers, is inert in v1.11, and will be removed in a future release. ([#60793](https://github.com/kubernetes/kubernetes/pull/60793), [@charrywanganthony](https://github.com/charrywanganthony))
* The `kubectl rolling-update` is now deprecated.  Use `kubectl rollout` instead.  ([#61285](https://github.com/kubernetes/kubernetes/pull/61285), [@soltysh](https://github.com/soltysh))
* kube-apiserver: the default `--endpoint-reconciler-type` is now `lease`. The `master-count` endpoint reconciler type is deprecated and will be removed in 1.13. ([#63383](https://github.com/kubernetes/kubernetes/pull/63383), [@liggitt](https://github.com/liggitt))
* OpenStack built-in cloud provider is now deprecated. Please use the external cloud provider for OpenStack. ([#63524](https://github.com/kubernetes/kubernetes/pull/63524), [@dims](https://github.com/dims))
* The Kubelet's deprecated `--allow-privileged` flag now defaults to true. This enables users to stop setting `--allow-privileged` in order to transition to `PodSecurityPolicy`. Previously, users had to continue setting `--allow-privileged`, because the default was false. ([#63442](https://github.com/kubernetes/kubernetes/pull/63442), [@mtaufen](https://github.com/mtaufen))
* The old dynamic client has been replaced by a new one.  The previous dynamic client will exist for one release in `client-go/deprecated-dynamic`.  Switch as soon as possible. ([#63446](https://github.com/kubernetes/kubernetes/pull/63446), [@deads2k](https://github.com/deads2k))
* In-tree support for openstack credentials is now deprecated. please use the "client-keystone-auth" from the cloud-provider-openstack repository. details on how to use this new capability is documented [here](https://github.com/kubernetes/cloud-provider-openstack/blob/master/docs/using-client-keystone-auth.md) ([#64346](https://github.com/kubernetes/kubernetes/pull/64346), [@dims](https://github.com/dims))
* The GitRepo volume type is deprecated. To provision a container with a git repo, mount an `EmptyDir` into an `InitContainer` that clones the repo using git, then moEmptyDir` into the Pod's container.
([#63445](https://github.com/kubernetes/kubernetes/pull/63445), [@ericchiang](https://github.com/ericchiang))
* Alpha annotation for PersistentVolume node affinity has been removed.  Update your PersistentVolumes to use the beta PersistentVolume.nodeAffinity field before upgrading to this release. ([#61816](https://github.com/kubernetes/kubernetes/pull/61816), [@wackxu
](https://github.com/wackxu))

#### Removed Deprecations

* kubeadm has removed the `.ImagePullPolicy` field in the v1alpha2 API version. Instead it's set statically to `IfNotPresent` for all required images. If you want to always pull the latest images before cluster init (as `Always` would do), run `kubeadm config images pull` before each `kubeadm init`. If you don't want the kubelet to pull any images at `kubeadm init` time, for example if you don't have an internet connection, you can also run `kubeadm config images pull` before `kubeadm init` or side-load the images some other way (such as `docker load -i image.tar`). Having the images locally cached will result in no pull at runtime, which makes it possible to run without any internet connection. ([#64096](https://github.com/kubernetes/kubernetes/pull/64096), [@luxas](https://github.com/luxas))
* kubeadm has removed `.Etcd.SelfHosting` from its configuration API. It was never used in practice ([#63871](https://github.com/kubernetes/kubernetes/pull/63871), [@luxas](https://github.com/luxas))
* The deprecated and inactive option '--enable-custom-metrics'  has been removed in 1.11. ([#60699](https://github.com/kubernetes/kubernetes/pull/60699), [@CaoShuFeng](https://github.com/CaoShuFeng))
* --include-extended-apis, which was deprecated back in [#32894](https://github.com/kubernetes/kubernetes/pull/32894), has been removed. ([#62803](https://github.com/kubernetes/kubernetes/pull/62803), [@deads2k](https://github.com/deads2k))
* Kubelets will no longer set `externalID` in their node spec. This feature has been deprecated since v1.1. ([#61877](https://github.com/kubernetes/kubernetes/pull/61877), [@mikedanese](https://github.com/mikedanese))
* The `initresource` admission plugin has been removed. ([#58784](https://github.com/kubernetes/kubernetes/pull/58784), [@wackxu](https://github.com/wackxu))
* `ObjectMeta `, `ListOptions`, and `DeleteOptions` have been removed from the core api group.  Please reference them in `meta/v1` instead. ([#61809](https://github.com/kubernetes/kubernetes/pull/61809), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* The deprecated `--mode` flag in `check-network-mode` has been removed. ([#60102](https://github.com/kubernetes/kubernetes/pull/60102), [@satyasm](https://github.com/satyasm))
* Support for the `alpha.kubernetes.io/nvidia-gpu` resource, which was deprecated in 1.10, has been removed. Please use the resource exposed by DevicePlugins instead (`nvidia.com/gpu`). ([#61498](https://github.com/kubernetes/kubernetes/pull/61498), [@mindprince](https://github.com/mindprince))
* The `kube-cloud-controller-manager` flag `--service-account-private-key-file` has been removed. Use `--use-service-account-credentials` instead. ([#60875](https://github.com/kubernetes/kubernetes/pull/60875), [@charrywanganthony](https://github.com/charrywanganthony))
* The rknetes code, which was deprecated in 1.10, has been removed. Use rktlet and CRI instead. ([#61432](https://github.com/kubernetes/kubernetes/pull/61432), [@filbranden](https://github.com/filbranden))
* DaemonSet scheduling associated with the alpha ScheduleDaemonSetPods feature flag has been emoved. See https://github.com/kubernetes/features/issues/548 for feature status. ([#61411](https://github.com/kubernetes/kubernetes/pull/61411), [@liggitt](https://github.com/liggitt))
* The `METADATA_AGENT_VERSION` configuration option has been removed to keep metadata agent version consistent across Kubernetes deployments. ([#63000](https://github.com/kubernetes/kubernetes/pull/63000), [@kawych](https://github.com/kawych))
* The deprecated `--service-account-private-key-file` flag has been removed from the cloud-controller-manager. The flag is still present and supported in the kube-controller-manager. ([#65182](https://github.com/kubernetes/kubernetes/pull/65182), [@liggitt](https://github.com/liggitt))
* Removed alpha functionality that allowed the controller manager to approve kubelet server certificates. This functionality should be replaced by automating validation and approval of node server certificate signing requests. ([#62471](https://github.com/kubernetes/kubernetes/pull/62471), [@mikedanese](https://github.com/mikedanese))

#### Graduated to Stable/GA
* IPVS-based in-cluster load balancing is now GA ([ref](https://github.com/kubernetes/features/issues/265))
* Enable CoreDNS as a DNS plugin for Kubernetes ([ref](https://github.com/kubernetes/features/issues/427))
* Azure Go SDK is now GA ([#63063](https://github.com/kubernetes/kubernetes/pull/63063), [@feiskyer](https://github.com/feiskyer))
* ClusterRole aggregation is now GA ([ref](https://github.com/kubernetes/features/issues/502))
* CRI validation test suite is now GA ([ref](https://github.com/kubernetes/features/issues/292))
* StorageObjectInUseProtection is now GA ([ref](https://github.com/kubernetes/features/issues/498)) and ([ref](https://github.com/kubernetes/features/issues/499))

#### Graduated to Beta

* Supporting out-of-tree/external cloud providers is now considered beta ([ref](https://github.com/kubernetes/features/issues/88))
* Resizing PersistentVolumes after pod restart is now considered beta. ([ref](https://github.com/kubernetes/features/issues/284))
* sysctl support is now considered beta ([ref](https://github.com/kubernetes/features/issues/34))
* Support for Azure Virtual Machine Scale Sets is now considered beta. ([ref](https://github.com/kubernetes/features/issues/513))
* Azure support for Cluster Autoscaler is now considered beta. ([ref](https://github.com/kubernetes/features/issues/514))
* The ability to limit a node's access to the API is now considered beta. ([ref](https://github.com/kubernetes/features/issues/279))
* CustomResource versioning is now considered beta. ([ref](https://github.com/kubernetes/features/issues/544))
* Windows container configuration in CRI is now considered beta ([ref](https://github.com/kubernetes/features/issues/547))
* CRI logging and stats are now considered beta ([ref](https://github.com/kubernetes/features/issues/552))
* The dynamic Kubelet config feature is now beta, and the DynamicKubeletConfig feature gate is on by default. In order to use dynamic Kubelet config, ensure that the Kubelet's --dynamic-config-dir option is set.  ([#64275](https://github.com/kubernetes/kubernetes/pull/64275), [@mtaufen](https://github.com/mtaufen))
* The Sysctls experimental feature has been promoted to beta (enabled by default via the `Sysctls` feature flag). PodSecurityPolicy and Pod objects now have fields for specifying and controlling sysctls. Alpha sysctl annotations will be ignored by 1.11+ kubelets. All alpha sysctl annotations in existing deployments must be converted to API fields to be effective. ([#6371](https://github.com/kubernetes/kubernetes/pull/63717), [@ingvagabund](https://github.com/ingvagabund))
* Volume expansion is now considered Beta. ([#64288](https://github.com/kubernetes/kubernetes/pull/64288), [@gnufied](https://github.com/gnufied))
* CRI container log rotation is now considered beta, and is enabled by default. ([#64046](https://github.com/kubernetes/kubernetes/pull/64046), [@yujuhong](https://github.com/yujuhong))
* The `PriorityClass` API has been promoted to `scheduling.k8s.io/v1beta1` ([#63100](https://github.com/kubernetes/kubernetes/pull/63100), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* The priorityClass feature is now considered beta. ([#63724](https://github.com/kubernetes/kubernetes/pull/63724), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* client-go: credential exec plugins is now considered beta. ([#64482](https://github.com/kubernetes/kubernetes/pull/64482), [@ericchiang](https://github.com/ericchiang))
* Subresources for custom resources is now considered beta and enabled by default. With this, updates to the `/status` subresource will disallow updates to all fields other than `.status` (not just `.spec` and `.metadata` as before). Also, `required` can be used at the root of the CRD OpenAPI validation schema when the `/status` subresource is enabled. ([#63598](https://github.com/kubernetes/kubernetes/pull/63598), [@nikhita](https://github.com/nikhita))

### New alpha features

* kube-scheduler can now schedule DaemonSet pods ([ref](https://github.com/kubernetes/features/issues/548))
* You can now resize PersistentVolumes without taking them offline ([ref](https://github.com/kubernetes/features/issues/531))
* You can now set a maximum volume count ([ref](https://github.com/kubernetes/features/issues/554))
* You can now do environment variable expansion in a subpath mount. ([ref](https://github.com/kubernetes/features/issues/559))
* You can now run containers in a pod as a particular group. ([ref](https://github.com/kubernetes/features/issues/213))
You can now bind tokens to service requests. ([ref](https://github.com/kubernetes/features/issues/542))
* The --experimental-qos-reserve kubelet flags has been replaced by the alpha level --qos-reserved flag or the QOSReserved field in the kubeletconfig, and requires the QOSReserved feature gate to be enabled. ([#62509](https://github.com/kubernetes/kubernetes/pull/62509), [@sjenning](https://github.com/sjenning))

## Other Notable Changes

### SIG API Machinery

* Orphan delete is now supported for custom resources. ([#63386](https://github.com/kubernetes/kubernetes/pull/63386), [@roycaihw](https://github.com/roycaihw))
* Metadata of CustomResources is now pruned and schema-checked during deserialization of requests and when read from etcd. In the former case, invalid meta data is rejected, in the later it is dropped from the CustomResource objects. ([#64267](https://github.com/kubernetes/kubernetes/pull/64267), [@sttts](https://github.com/sttts))
* The kube-apiserver openapi doc now includes extensions identifying `APIService` and `CustomResourceDefinition` `kind`s ([#64174](https://github.com/kubernetes/kubernetes/pull/64174), [@liggitt](https://github.com/liggitt))
* CustomResourceDefinitions Status subresource now supports GET and PATCH ([#63619](https://github.com/kubernetes/kubernetes/pull/63619), [@roycaihw](https://github.com/roycaihw))
* When updating `/status` subresource of a custom resource, only the value at the `.status` subpath for the update is considered. ([#63385](https://github.cm/kubernetes/kubernetes/pull/63385), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Added a way to pass extra arguments to etcd. The these extra arguments can be used to adjust runtime configuration like heartbeat interval etc. ([#63961](https://github.com/kubernetes/kubernetes/pull/63961), [@mborsz](https://github.com/mborsz))
* Added Establishing Controller on CRDs to avoid race between Established condition and CRs actually served. In HA setups, the Established condition is delayed by 5 seconds. ([#63068](https://github.com/kubernetes/kubernetes/pull/63068), [@xmudrii](https://github.com/xmudrii))
* Added `spec.additionalPrinterColumns` to CRDs to define server side printing columns. ([#60991](https://github.com/kubernetes/kubernetes/pull/60991), [@sttts](https://github.com/sttts))
* Added CRD Versioning with NOP converter ([#63830](https://github.com/kubernetes/kubernetes/pull/63830), [@mbohlool](https://github.com/mbohlool))
* Allow "required" and "description" to be used at the CRD OpenAPI validation schema root when the `/status` subresource is enabled. ([#63533](https://github.com/kubernetes/kubernetes/pull/63533), [@sttts](https://github.com/sttts))
* Etcd health checks by the apiserver now ensure the apiserver can connect to and exercise the etcd API. ([#65027](https://github.com/kubernetes/kubernetes/pull/65027), [@liggitt](https://github.com/liggitt)) api- machinery
* The deprecated `--service-account-private-key-file` flag has been removed from the `cloud-controller-manager`. The flag is still present and supported in the `kube-controller-manager`. ([#65182](https://github.com/kubernetes/kubernetes/pull/65182), [@liggitt](https://github.com/liggitt))
* Webhooks for the mutating admission controller now support the "remove" operation. ([#64255](https://github.com/kubernetes/kubernetes/pull/64255), [@rojkov](https://github.com/rojkov)) sig-API machinery
* The CRD OpenAPI v3 specification for validation now allows `additionalProperties`, which are mutually exclusive to properties. ([#62333](https://github.com/kubernetes/kubernetes/pull/62333), [@sttts](https://github.com/sttts))
* Added the apiserver configuration option to choose the audit output version. ([#60056](https://github.com/kubernetes/kubernetes/pull/60056), [@crassirostris](https://github.com/crassirostris))
* Created a new `dryRun` query parameter for mutating endpoints. If the parameter is set, then the query will be rejected, as the feature is not implemented yet. This will allow forward compatibility with future clients; otherwise, future clients talking with older apiservers might end up modifying a resource even if they include the `dryRun` query parameter. ([#63557](https://github.com/kubernetes/kubernetes/pull/63557), [@apelisse](https://github.com/apelisse))
* `list`/`watch` API requests with a `fieldSelector` that specifies `metadata.name` can now be authorized as requests for an individual named resource ([#63469](https://github.com/kubernetes/kubernetes/pull/63469), [@wojtek-t](https://github.com/wojtek-t))
* Exposed `/debug/flags/v` to allow dynamically set glog logging level.  For example, to change glog level to 3, send a PUT request such as `curl -X PUT http://127.0.0.1:8080/debug/flags/v -d "3"`. ([#63777](https://github.com/kubernetes/kubernetes/pull/63777), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Exec authenticator plugin supports TLS client certificates. ([#61803](https://github.com/kubernetes/kubernetes/pull/61803), [@awly](https://github.com/awly))
* The `serverAddressByClientCIDRs` property in `metav1.APIGroup`(discovery API) is now optional instead of required. ([#61963](https://github.com/kubernetes/kubernetes/pull/61963), [@roycaihw](https://github.com/roycaihw))
* `apiservices/status` and `certificatesigningrequests/status` now support `GET` and `PATCH` ([#64063](https://github.com/kubernetes/kubernetes/pull/64063), [@roycaihw](https://github.com/roycaihw))
* APIServices with kube-like versions (e.g. `v1`, `v2beta1`, etc.) will be sorted appropriately within each group.  ([#64004](https://github.com/kubernetes/kubernetes/pull/64004), [@mbohlool](https://github.com/mbohlool))
* Event object references with apiversion will now that value. ([#63913](https://github.com/kubernetes/kubernetes/pull/63913), [@deads2k](https://github.com/deads2k))
* Fixes the `kubernetes.default.svc` loopback service resolution to use a loopback configuration. ([#62649](https://github.com/kubernetes/kubernetes/pull/62649), [@liggitt](https://github.com/liggitt))

### SIG Apps

* Added generators for `apps/v1` deployments. ([#61288](https://github.com/kubernetes/kubernetes/pull/61288), [@ayushpateria](https://github.com/ayushpateria))

### SIG Auth

* RBAC information is now included in audit logs via audit.Event annotations:
    * authorization.k8s.io/decision = {allow, forbid}
	* authorization.k8s.io/reason = human-readable reason for the decision ([#58807](https://github.com/kubernetes/kubernetes/pull/58807), [@CaoShuFeng](https://github.com/CaoShuFeng))
* `kubectl certificate approve|deny` will not modify an already approved or denied CSR unless the `--force` flag is provided. ([#61971](https://github.com/kubernetes/kubernetes/pull/61971), [@smarterclayton](https://github.com/smarterclayton))
* The `--bootstrap-kubeconfig` argument to Kubelet previously created the first bootstrap client credentials in the certificates directory as `kubelet-client.key` and `kubelet-client.crt`.  Subsequent certificates created by cert rotation were created in a combined PEM file that was atomically rotated as `kubelet-client-DATE.pem` in that directory, which meant clients relying on the `node.kubeconfig` generated by bootstrapping would never use a rotated cert.  The initial bootstrap certificate is now generated into the cert directory as a PEM file and symlinked to `kubelet-client-current.pem` so that the generated kubeconfig remains valid after rotation. ([#62152](https://github.com/kubernetes/kubernetes/pull/62152), [@smarterclayton](https://github.com/smarterclayton))
* Owner references can now be set during creation, even if the user doesn't have deletion power ([#63403](https://github.com/kubernetes/kubernetes/pull/63403), [@deads2k](https://github.com/deads2k))
* Laid the groundwork for OIDC distributed claims handling in the apiserver authentication token checker. A distributed claim allows the OIDC provider to delegate a claim to a separate URL. ([ref](http://openid.net/specs/openid-connect-core-1_0.html#AggregatedDistributedClaims)). ([#63213](https://github.com/kubernetes/kubernetes/pull/63213), [@filmil](https://github.com/filmil))
* RBAC: all configured authorizers are now checked to determine if an RBAC role or clusterrole escalation (setting permissions the user does not currently have via RBAC) is allowed. ([#56358](https://github.com/kubernetes/kubernetes/pull/56358), [@liggitt](https://github.com/liggitt))
* kube-apiserver: OIDC authentication now supports requiring specific claims with `--oidc-required-claim=<claim>=<value>` Previously, there was no mechanism for a user to specify claims in the OIDC authentication process that were requid to be present in the ID Token with an expected value. This version now makes it possible to require claims support for the OIDC authentication. It allows users to pass in a `--oidc-required-claims` flag, and `key=value` pairs in the API config, which will ensure that the specified required claims are checked against the ID Token claims. ([#62136](https://github.com/kubernetes/kubernetes/pull/62136), [@rithujohn191](https://github.com/rithujohn191))
* Included the list of security groups when failing with the errors that more than one is tagged. ([#58874](https://github.com/kubernetes/kubernetes/pull/58874), [@sorenmat](https://github.com/sorenmat))
* Added proxy for container streaming in kubelet for streaming auth. ([#64006](https://github.com/kubernetes/kubernetes/pull/64006), [@Random-Liu](https://github.com/Random-Liu))
* PodSecurityPolicy admission information has been added to audit logs. ([#58143](https://github.com/kubernetes/kubernetes/pull/58143), [@CaoShuFeng](https://github.com/CaoShuFeng))
* TokenRequests now are required to have an expiration duration between 10 minutes and 2^32 seconds. ([#63999](https://github.com/kubernetes/kubernetes/pull/63999), [@mikedanese](https://github.com/mikedanese))
* The `NodeRestriction` admission plugin now prevents kubelets from modifying/removing taints applied to their Node API object. ([#63167](https://github.com/kubernetes/kubernetes/pull/63167), [@liggitt](https://github.com/liggitt))
* authz: nodes should not be able to delete themselves ([#62818](https://github.com/kubernetes/kubernetes/pull/62818), [@mikedanese](https://github.com/mikedanese))

### SIG Autoscaling

* A cluster-autoscaler ClusterRole is added to cover only the functionality required by Cluster Autoscaler and avoid abusing system:cluster-admin role. Cloud providers other than GCE might want to update their deployments or sample yaml files to reuse the role created via add-on. ([#64503](https://github.com/kubernetes/kubernetes/pull/64503), [@kgolab](https://github.com/kgolab))

### SIG Azure

* The Azure cloud provider now supports standard SKU load balancer and public IP.
`excludeMasterFromStandardLB` defaults to true, which means master nodes are excluded from the standard load balancer. Also note that because all nodes (except master) are added as loadbalancer backends, the standard load balancer doesn't work with the `service.beta.kubernetes.io/azure-load-balancer-mode` annotation.
([#61884](https://github.com/kubernetes/kubernetes/pull/61884), [#62707](https://github.com/kubernetes/kubernetes/pull/62707), [@feiskyer](https://github.com/feiskyer))
* The Azure cloud provider now supports specifying allowed service tags by the `service.beta.kubernetes.io/azure-allowed-service-tags` annotation. ([#61467](https://github.com/kubernetes/kubernetes/pull/61467), [@feiskyer](https://github.com/feiskyer))
* You can now change the size of an azuredisk PVC using `kubectl edit pvc pvc-azuredisk`. Note that this operation will fail if the volume is already attached to a  running VM. ([#64386](https://github.com/kubernetes/kubernetes/pull/64386), [@andyzhangx](https://github.com/andyzhangx))
* Block device support has been added for azure disk. ([#63841](https://github.com/kubernetes/kubernetes/pull/63841), [@andyzhangx](https://github.com/andyzhangx))
* Azure VM names can now contain the underscore (`_`) character ([#63526](https://github.com/kubernetes/kubernetes/pull/63526), [@djsly](https://github.com/djsly))
* Azure disks now support external resource groups.
([#64427](https://github.com/kubernetes/kuernetes/pull/64427), [@andyzhangx](https://github.com/andyzhangx))
* Added reason message logs for non-existant Azure resources.
([#64248](https://github.com/kubernetes/kubernetes/pull/64248), [@feiskyer](https://github.com/feiskyer))

### SIG CLI

* You can now use the `base64decode` function in kubectl go templates to decode base64-encoded data, such as `kubectl get secret SECRET -o go-template='{{ .data.KEY | base64decode }}'`. ([#60755](https://github.com/kubernetes/kubernetes/pull/60755), [@glb](https://github.com/glb))
* `kubectl patch` now supports `--dry-run`. ([#60675](https://github.com/kubernetes/kubernetes/pull/60675), [@timoreimann](https://github.com/timoreimann))
* The global flag `--match-server-version` is now global.  `kubectl version` will respect it. ([#63613](https://github.com/kubernetes/kubernetes/pull/63613), [@deads2k](https://github.com/deads2k))
* kubectl will list all allowed print formats when an invalid format is passed. ([#64371](https://github.com/kubernetes/kubernetes/pull/64371), [@CaoShuFeng](https://github.com/CaoShuFeng))
* The global flag "context" now gets applied to `kubectl config view --minify`. In previous versions, this command was only available for `current-context`. Now it will be easier for users to view other non current contexts when minifying. ([#64608](https://github.com/kubernetes/kubernetes/pull/64608), [@dixudx](https://github.com/dixudx))
* `kubectl apply --prune` supports CronJob resources.  ([#62991](https://github.com/kubernetes/kubernetes/pull/62991), [@tomoe](https://github.com/tomoe))
* The `--dry-run` flag has been enabled for `kubectl auth reconcile` ([#64458](https://github.com/kubernetes/kubernetes/pull/64458), [@mrogers950](https://github.com/mrogers950))
* `kubectl wait` is a new command that allows waiting for one or more resources to be deleted or to reach a specific condition. It adds a `kubectl wait --for=[delete|condition=condition-name] resource/string` command. ([#64034](https://github.com/kubernetes/kubernetes/pull/64034), [@deads2k](https://github.com/deads2k))
* `kubectl auth reconcile` only works with rbac.v1; all the core helpers have been switched over to use the external types. ([#63967](https://github.com/kubernetes/kubernetes/pull/63967), [@deads2k](https://github.com/deads2k))
* kubectl and client-go now detect duplicated names for user, cluster and context when loading kubeconfig and report this condition as an error.  ([#60464](https://github.com/kubernetes/kubernetes/pull/60464), [@roycaihw](https://github.com/roycaihw))
* Added 'UpdateStrategyType' and 'RollingUpdateStrategy' to 'kubectl describe sts' command output. ([#63844](https://github.com/kubernetes/kubernetes/pull/63844), [@tossmilestone](https://github.com/tossmilestone))
* Initial Korean translation for kubectl has been added. ([#62040](https://github.com/kubernetes/kubernetes/pull/62040), [@ianychoi](https://github.com/ianychoi))
* `kubectl cp` now supports completion.
([#60371](https://github.com/kubernetes/kubernetes/pull/60371), [@superbrothers](https://github.com/superbrothers))
* The shortcuts that were moved server-side in at least 1.9 have been removed from being hardcoded in kubectl. This means that the client-based restmappers have been moved to client-go, where everyone who needs them can have access. ([#63507](https://github.com/kubernetes/kubernetes/pull/63507), [@deads2k](https://github.com/deads2k))
* When using `kubectl delete` with selection criteria, the defaults to is now to ignore "not found" errors. Note that this does not apply when deleting a speciic resource. ([#63490](https://github.com/kubernetes/kubernetes/pull/63490), [@deads2k](https://github.com/deads2k))
* `kubectl create [secret | configmap] --from-file` now works on Windows with fully-qualified paths ([#63439](https://github.com/kubernetes/kubernetes/pull/63439), [@liggitt](https://github.com/liggitt))
* Portability across systems has been increased by the use of `/usr/bin/env` in all script shebangs. ([#62657](https://github.com/kubernetes/kubernetes/pull/62657), [@matthyx](https://github.com/matthyx))
* You can now use `kubectl api-resources` to discover resources.
 ([#42873](https://github.com/kubernetes/kubernetes/pull/42873),  [@xilabao](https://github.com/xilabao))
* You can now display requests/limits of extended resources in node allocated resources. ([#46079](https://github.com/kubernetes/kubernetes/pull/46079), [@xiangpengzhao](https://github.com/xiangpengzhao))
* The `--remove-extra-subjects` and `--remove-extra-permissions` flags have been enabled for `kubectl auth reconcile` ([#64541](https://github.com/kubernetes/kubernetes/pull/64541), [@mrogers950](https://github.com/mrogers950))
* kubectl now has improved compatibility with older servers when creating/updating API objects ([#61949](https://github.com/kubernetes/kubernetes/pull/61949), [@liggitt](https://github.com/liggitt))
* `kubectl apply` view/edit-last-applied now supports completion. ([#60499](https://github.com/kubernetes/kubernetes/pull/60499), [@superbrothers](https://github.com/superbrothers))

### SIG Cluster Lifecycle

  * kubeadm: The `:Etcd` struct has been refactored in the v1alpha2 API. All the options now reside under either `.Etcd.Local` or `.Etcd.External`. Automatic conversions from the v1alpha1 API are supported. ([#64066](https://github.com/kubernetes/kubernetes/pull/64066), [@luxas](https://github.com/luxas))
* kubeadm now uses an upgraded API version for the configuration file, `kubeadm.k8s.io/v1alpha2`. kubeadm in v1.11 will still be able to read `v1alpha1` configuration, and will automatically convert the configuration to `v1alpha2`, both internally and when storing the configuration in the ConfigMap in the cluster. ([#63788](https://github.com/kubernetes/kubernetes/pull/63788), [@luxas](https://github.com/luxas))
* Phase `kubeadm alpha phase kubelet` has been added to support dynamic kubelet configuration in kubeadm. ([#57224](https://github.com/kubernetes/kubernetes/pull/57224), [@xiangpengzhao](https://github.com/xiangpengzhao))
* The kubeadm config option `API.ControlPlaneEndpoint` has been extended to take an optional port, which may differ from the apiserver's bind port. ([#62314](https://github.com/kubernetes/kubernetes/pull/62314), [@rjosephwright](https://github.com/rjosephwright))
* The `--cluster-name` parameter has been added to kubeadm init, enabling users to specify the cluster name in kubeconfig. ([#60852](https://github.com/kubernetes/kubernetes/pull/60852), [@karan](https://github.com/karan))
* The logging feature for kubeadm commands now supports a verbosity setting. ([#57661](https://github.com/kubernetes/kubernetes/pull/57661), [@vbmade2000](https://github.com/vbmade2000))
* kubeadm now has a join timeout that can be controlled via the `discoveryTimeout` config option. This option is set to 5 minutes by default. ([#60983](https://github.com/kubernetes/kubernetes/pull/60983), [@rosti](https://github.com/rosti))
* Added the `writable` boolean option to kubeadm config. This option works on a per-volume basis for `ExtraVolumes` config keys. ([#60428](https://github.com/kubernetes/kubernetes/pul60428), [@rosti](https://github.com/rosti))
* Added a new `kubeadm upgrade node config` command. ([#64624](https://github.com/kubernetes/kubernetes/pull/64624), [@luxas](https://github.com/luxas))
* kubeadm now makes the CoreDNS container more secure by dropping (root) capabilities and improves the integrity of the container by running the whole container in read-only.  ([#64473](https://github.com/kubernetes/kubernetes/pull/64473), [@nberlee](https://github.com/nberlee))
* kubeadm now detects the Docker cgroup driver and starts the kubelet with the matching driver. This eliminates a common error experienced by new users in when the Docker cgroup driver is not the same as the one set for the kubelet due to different Linux distributions setting different cgroup drivers for Docker, making it hard to start the kubelet properly.
([#64347](https://github.com/kubernetes/kubernetes/pull/64347), [@neolit123](https://github.com/neolit123))
* Added a 'kubeadm config migrate' command to convert old API types to their newer counterparts in the new, supported API types. This is just a client-side tool; it just executes locally without requiring a cluster to be running, operating in much the same way as a Unix pipe that upgrades config files. ([#64232](https://github.com/kubernetes/kubernetes/pull/64232), [@luxas](https://github.com/luxas))
* kubeadm will now pull required images during preflight checks if it cannot find them on the system. ([#64105](https://github.com/kubernetes/kubernetes/pull/64105), [@chuckha](https://github.com/chuckha))
* "kubeadm init" now writes a structured and versioned kubelet ComponentConfiguration file to `/var/lib/kubelet/config.yaml` and an environment file with runtime flags that you can source in the systemd kubelet dropin to `/var/lib/kubelet/kubeadm-flags.env`. ([#63887](https://github.com/kubernetes/kubernetes/pull/63887), [@luxas](https://github.com/luxas))
* A `kubeadm config print-default` command has now been added. You can use this command to output a starting point when writing your own kubeadm configuration files. ([#63969](https://github.com/kubernetes/kubernetes/pull/63969), [@luxas](https://github.com/luxas))
* Updated kubeadm's minimum supported kubernetes in v1.11.x to 1.10 ([#63920](https://github.com/kubernetes/kubernetes/pull/63920), [@dixudx](https://github.com/dixudx))
* Added the `kubeadm upgrade diff` command to show how static pod manifests will be changed by an upgrade. This command shows the changes that will be made to the static pod manifests before applying them. This is a narrower case than kubeadm upgrade apply --dry-run, which specifically focuses on the static pod manifests. ([#63930](https://github.com/kubernetes/kubernetes/pull/63930), [@liztio](https://github.com/liztio))
* The `kubeadm config images pull` command can now be used to pull container images used by kubeadm. ([#63833](https://github.com/kubernetes/kubernetes/pull/63833), [@chuckha](https://github.com/chuckha))
* kubeadm will now deploy CoreDNS by default instead of KubeDNS ([#63509](https://github.com/kubernetes/kubernetes/pull/63509), [@detiber](https://github.com/detiber))
* Preflight checks for kubeadm no longer validate custom kube-apiserver, kube-controller-manager and kube-scheduler arguments. ([#63673](https://github.com/kubernetes/kubernetes/pull/63673), [@chuckha](https://github.com/chuckha))
* Added a `kubeadm config images list` command that lists required container images for a kubeadm install. ([#63450](https://github.com/kubernetes/kubernetes/pull/63450), [@chuckha](https://github.com/chukha))
* You can now use `kubeadm token` specifying `--kubeconfig`.  In this case, kubeadm searches the current user home path and the environment variable KUBECONFIG for existing files. If provided, the `--kubeconfig` flag will be honored instead. ([#62850](https://github.com/kubernetes/kubernetes/pull/62850), [@neolit123](https://github.com/neolit123))
([#64988](https://github.com/kubernetes/kubernetes/pull/64988), [@detiber](https://github.com/detiber))
* kubeadm now sets peer URLs for the default etcd instance. Previously we left the defaults, which meant the peer URL was unsecured.
* Kubernetes now packages crictl in a cri-tools deb and rpm package. ([#64836](https://github.com/kubernetes/kubernetes/pull/64836), [@chuckha](https://github.com/chuckha))
* kubeadm now prompts the user for confirmation when resetting a master node. ([#59115](https://github.com/kubernetes/kubernetes/pull/59115), [@alexbrand](https://github.com/alexbrand))
* kubead now creates kube-proxy with a toleration to run on all nodes, no matter the taint. ([#62390](https://github.com/kubernetes/kubernetes/pull/62390), [@discordianfish](https://github.com/discordianfish))
* kubeadm now sets the kubelet `--resolv-conf` flag conditionally on init. ([#64665](https://github.com/kubernetes/kubernetes/pull/64665), [@stealthybox](https://github.com/stealthybox))
* Added ipset and udevadm to the hyperkube base image. ([#61357](https://github.com/kubernetes/kubernetes/pull/61357), [@rphillips](https://github.com/rphillips))

### SIG GCP

* Kubernetes clusters on GCE now have crictl installed. Users can use it to help debug their nodes. See the [crictl documentation](https://github.com/kubernetes-incubator/cri-tools/blob/master/docs/crictl.md) for details. ([#63357](https://github.com/kubernetes/kubernetes/pull/63357), [@Random-Liu](https://github.com/Random-Liu))
* `cluster/kube-up.sh` now provisions a Kubelet config file for GCE via the metadata server. This file is installed by the corresponding GCE init scripts. ([#62183](https://github.com/kubernetes/kubernetes/pull/62183), [@mtaufen](https://github.com/mtaufen))
* GCE: Update cloud provider to use TPU v1 API ([#64727](https://github.com/kubernetes/kubernetes/pull/64727), [@yguo0905](https://github.com/yguo0905))
* GCE: Bump GLBC version to 1.1.1 - fixing an issue of handling multiple certs with identical certificates. ([#62751](https://github.com/kubernetes/kubernetes/pull/62751), [@nicksardo](https://github.com/nicksardo))

### SIG Instrumentation

* Added prometheus cluster monitoring addon to kube-up. ([#62195](https://github.com/kubernetes/kubernetes/pull/62195), [@serathius](https://github.com/serathius))
* Kubelet now exposes a new endpoint, `/metrics/probes`, which exposes a Prometheus metric containing the liveness and/or readiness probe results for a container. ([#61369](https://github.com/kubernetes/kubernetes/pull/61369), [@rramkumar1](https://github.com/rramkumar1))

### SIG Network

* The internal IP address of the node is now added as additional information for kubectl. ([#57623](https://github.com/kubernetes/kubernetes/pull/57623), [@dixudx](https://github.com/dixudx))
* NetworkPolicies can now target specific pods in other namespaces by including both a namespaceSelector and a podSelector in the same peer element. ([#60452](https://github.com/kubernetes/kubernetes/pull/60452), [@danwinship](https://github.com/danwinship))
* CoreDNS deployment configuration now uses the k8s.gcr.io imageRepository. ([#64775](https://github.com/kubernetes/kubernetes/pull/64775), [@rajansandeep](https://giub.com/rajansandeep))
* kubelet's `--cni-bin-dir` option now accepts multiple comma-separated CNI binary directory paths, which are searched for CNI plugins in the given order. ([#58714](https://github.com/kubernetes/kubernetes/pull/58714), [@dcbw](https://github.com/dcbw))
* You can now use `--ipvs-exclude-cidrs` to specify a list of CIDR's which the IPVS proxier should not touch when cleaning up IPVS rules. ([#62083](https://github.com/kubernetes/kubernetes/pull/62083), [@rramkumar1](https://github.com/rramkumar1))
* You can now receive node DNS info with the `--node-ip` flag, which adds `ExternalDNS`, `InternalDNS`, and `ExternalIP` to kubelet's output. ([#63170](https://github.com/kubernetes/kubernetes/pull/63170), [@micahhausler](https://github.com/micahhausler))
* You can now have services that listen on the same host ports on different interfaces by specifying `--nodeport-addresses`. ([#62003](https://github.com/kubernetes/kubernetes/pull/62003), [@m1093782566](https://github.com/m1093782566))
* Added port-forward examples for service

### SIG Node

* CRI: The container log path has been changed from containername_attempt#.log to containername/attempt#.log  ([#62015](https://github.com/kubernetes/kubernetes/pull/62015), [@feiskyer](https://github.com/feiskyer))
* Introduced the `ContainersReady` condition in Pod status. ([#64646](https://github.com/kubernetes/kubernetes/pull/64646), [@freehan](https://github.com/freehan))
* Kubelet will now set extended resource capacity to zero after it restarts. If the extended resource is exported by a device plugin, its capacity will change to a valid value after the device plugin re-connects with the Kubelet. If the extended resource is exported by an external component through direct node status capacity patching, the component should repatch the field after kubelet becomes ready again. During the time gap, pods previously assigned with such resources may fail kubelet admission but their controller should create new pods in response to such failures. ([#64784](https://github.com/kubernetes/kubernetes/pull/64784), [@jiayingz](https://github.com/jiayingz)) node
* You can now use a security context with Windows containers
([#64009](https://github.com/kubernetes/kubernetes/pull/64009), [@feiskyer](https://github.com/feiskyer))
* Added e2e regression tests for kubelet security. ([#64140](https://github.com/kubernetes/kubernetes/pull/64140), [@dixudx](https://github.com/dixudx))
* The maximum number of images the Kubelet will report in the Node status can now be controlled via the Kubelet's `--node-status-max-images` flag. The default (50) remains the same. ([#64170](https://github.com/kubernetes/kubernetes/pull/64170), [@mtaufen](https://github.com/mtaufen))
* The Kubelet now exports metrics that report the assigned (`node_config_assigned`), last-known-good (`node_config_last_known_good`), and active (`node_config_active`) config sources, and a metric indicating whether the node is experiencing a config-related error (`node_config_error`). The config source metrics always report the value `1`, and carry the `node_config_name`, `node_config_uid`, `node_config_resource_version`, and `node_config_kubelet_key labels`, which identify the config version. The error metric reports `1` if there is an error, `0` otherwise. ([#57527](https://github.com/kubernetes/kubernetes/pull/57527), [@mtaufen](https://github.com/mtaufen))
* You now have the ability to quota resources by priority. ([#57963](https://github.com/kubernetes/kubernetes/pull/57963), [@vikaschoudhary16](https://github.com/ikaschoudhary16))
* The gRPC max message size in the remote container runtime has been increased to 16MB. ([#64672](https://github.com/kubernetes/kubernetes/pull/64672), [@mcluseau](https://github.com/mcluseau))
* Added a feature gate for the plugin watcher. ([#64605](https://github.com/kubernetes/kubernetes/pull/64605), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* The status of dynamic Kubelet config is now reported via Node.Status.Config, rather than the KubeletConfigOk node condition. ([#63314](https://github.com/kubernetes/kubernetes/pull/63314), [@mtaufen](https://github.com/mtaufen))
* You must now specify `Node.Spec.ConfigSource.ConfigMap.KubeletConfigKey` when using dynamic Kubelet config to tell the Kubelet which key of the `ConfigMap` identifies its config file. ([#59847](https://github.com/kubernetes/kubernetes/pull/59847), [@mtaufen](https://github.com/mtaufen))
* The dynamic Kubelet config feature will now update the config in the event of a ConfigMap mutation, which reduces the chance for silent config skew. Only name, namespace, and kubeletConfigKey may now be set in `Node.Spec.ConfigSource.ConfigMap`. The least disruptive pattern for config management is still to create a new ConfigMap and incrementally roll out a new `Node.Spec.ConfigSource`. ([#63221](https://github.com/kubernetes/kubernetes/pull/63221), [@mtaufen](https://github.com/mtaufen))
* Change seccomp annotation from "docker/default" to "runtime/default" ([#62662](https://github.com/kubernetes/kubernetes/pull/62662), [@wangzhen127](https://github.com/wangzhen127))
* The node authorizer now automatically sets up rules for `Node.Spec.ConfigSource` when the DynamicKubeletConfig feature gate is enabled. ([#60100](https://github.com/kubernetes/kubernetes/pull/60100), [@mtaufen](https://github.com/mtaufen))
* CRI now defines mounting behavior. If the host path doesn't exist, the runtime should return an error. If the host path is a symlink, the runtime should follow the symlink and mount the real destination to the container. ([#61460](https://github.com/kubernetes/kubernetes/pull/61460), [@feiskyer](https://github.com/feiskyer))

### SIG OpenStack

* Provide a meaningful error message in the openstack cloud provider when no valid IP address can be found for a node, rather than just the first address of the node, which leads to a load balancer error if that address is a hostname or DNS name instead of an IP address. ([#64318](https://github.com/kubernetes/kubernetes/pull/64318), [@gonzolino](https://github.com/gonzolino))
* Restored the pre-1.10 behavior of the openstack cloud provider, which uses the instance name as the Kubernetes Node name. This requires instances be named with RFC-1123 compatible names. ([#63903](https://github.com/kubernetes/kubernetes/pull/63903), [@liggitt](https://github.com/liggitt))
* Kubernetes will try to read the openstack auth config from the client config and fall back to read from the environment variables if the auth config is not available. ([#60200](https://github.com/kubernetes/kubernetes/pull/60200), [@dixudx](https://github.com/dixudx))

### SIG Scheduling

* Schedule DaemonSet Pods in scheduler, rather than the Daemonset controller.
([#63223](https://github.com/kubernetes/kubernetes/pull/63223), [@k82cn](https://github.com/k82cn))
* Added `MatchFields` to `NodeSelectorTerm`; in 1.11, it only supports `metadata.name`. ([#62002](https://github.com/kubernetes/kubernetes/pull/62002), [@k82cn](https://github.com/k82cn))
* kube-scheduler now has the `--write-config-to` flag so that Scheduler canwritets default configuration to a file.
([#62515](https://github.com/kubernetes/kubernetes/pull/62515), [@resouer](https://github.com/resouer))
* Performance of the affinity/anti-affinity predicate for the default scheduler has been significantly improved. ([#62211](https://github.com/kubernetes/kubernetes/pull/62211), [@bsalamat](https://github.com/bsalamat))
* The 'scheduling_latency_seconds' metric into has been split into finer steps (predicate, priority, preemption). ([#65306](https://github.com/kubernetes/kubernetes/pull/65306), [@shyamjvs](https://github.com/shyamjvs))
* Scheduler now has a summary-type metric, 'scheduling_latency_seconds'. ([#64838](https://github.com/kubernetes/kubernetes/pull/64838), [@krzysied](https://github.com/krzysied))
* `nodeSelector.matchFields` (node's `metadata.node`) is now supported in scheduler. ([#62453](https://github.com/kubernetes/kubernetes/pull/62453), [@k82cn](https://github.com/k82cn))
* Added a parametrizable priority function mapping requested/capacity ratio to priority. This function is disabled by default and can be enabled via the scheduler policy config file.
([#63929](https://github.com/kubernetes/kubernetes/pull/63929), [@losipiuk](https://github.com/losipiuk))
* System critical priority classes are now automatically added at cluster boostrapping. ([#60519](https://github.com/kubernetes/kubernetes/pull/60519), [@bsalamat](https://github.com/bsalamat))

### SIG Storage

* AWS EBS, Azure Disk, GCE PD and Ceph RBD volume plugins now support dynamic provisioning of raw block volumes. ([#64447](https://github.com/kubernetes/kubernetes/pull/64447), [@jsafrane](https://github.com/jsafrane))
* gitRepo volumes in pods no longer require git 1.8.5 or newer; older git versions are now supported. ([#62394](https://github.com/kubernetes/kubernetes/pull/62394), [@jsafrane](https://github.com/jsafrane))
* Added support for resizing Portworx volumes. ([#62308](https://github.com/kubernetes/kubernetes/pull/62308), [@harsh-px](https://github.com/harsh-px))
* Added block volume support to Cinder volume plugin. ([#64879](https://github.com/kubernetes/kubernetes/pull/64879), [@bertinatto](https://github.com/bertinatto))
* Provided API support for external CSI storage drivers to support block volumes. ([#64723](https://github.com/kubernetes/kubernetes/pull/64723), [@vladimirvivien](https://github.com/vladimirvivien))
* Volume topology aware dynamic provisioning for external provisioners is now supported. ([#63193](https://github.com/kubernetes/kubernetes/pull/63193), [@lichuqiang](https://github.com/lichuqiang))
* Added a volume projection that is able to project service account tokens. ([#62005](https://github.com/kubernetes/kubernetes/pull/62005), [@mikedanese](https://github.com/mikedanese))
* PodSecurityPolicy now supports restricting hostPath volume mounts to be readOnly and under specific path prefixes ([#58647](https://github.com/kubernetes/kubernetes/pull/58647), [@jhorwit2](https://github.com/jhorwit2))
* Added StorageClass API to restrict topologies of dynamically provisioned volumes. ([#63233](https://github.com/kubernetes/kubernetes/pull/63233), [@lichuqiang](https://github.com/lichuqiang))
* Added Alpha support for dynamic volume limits based on node type ([#64154](https://github.com/kubernetes/kubernetes/pull/64154), [@gnufied](https://github.com/gnufied))
* AWS EBS volumes can be now used as ReadOnly in pods. ([#64403](https://github.com/kubernetes/kubernetes/pull/64403), [@jsafrane](https://github.com/jsafrane))
* Basic plumbing for volume topology aware dynamic provisionin has been implemented. ([#63232](https://github.com/kubernetes/kubernetes/pull/63232), [@lichuqiang](https://github.com/lichuqiang))
* Changed ext3/ext4 volume creation to not reserve any portion of the volume for the root user. When creating ext3/ext4 volume, mkfs defaults to reserving 5% of the volume for the super-user (root). This patch changes the mkfs to pass -m0 to disable this setting.
([#64102](https://github.com/kubernetes/kubernetes/pull/64102), [@atombender](https://github.com/atombender))
* Added support for NFS relations on kubernetes-worker charm. ([#63817](https://github.com/kubernetes/kubernetes/pull/63817), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Implemented kubelet side online file system resizing ([#62460](https://github.com/kubernetes/kubernetes/pull/62460), [@mlmhl](https://github.com/mlmhl))
* Generated subpath name from Downward API env ([#49388](https://github.com/kubernetes/kubernetes/pull/49388), [@kevtaylor](https://github.com/kevtaylor))

### SIG vSphere

* Added a mechanism in vSphere Cloud Provider to get credentials from Kubernetes secrets, rather than the plain text `vsphere.conf` file.([#63902](https://github.com/kubernetes/kubernetes/pull/63902), [@abrarshivani](https://github.com/abrarshivani))
* vSphere Cloud Provider: added SAML token authentication support ([#63824](https://github.com/kubernetes/kubernetes/pull/63824), [@dougm](https://github.com/dougm))

### SIG Windows

* Added log and fs stats for Windows containers. ([#62266](https://github.com/kubernetes/kubernetes/pull/62266), [@feiskyer](https://github.com/feiskyer))
* Added security contexts for Windows containers. [#64009](https://github.com/kubernetes/kubernetes/pull/64009), ([@feiskyer](https://github.com/feiskyer))
* Added local persistent volumes for Windows containers. ([#62012](https://github.com/kubernetes/kubernetes/pull/62012), [@andyzhangx](https://github.com/andyzhangx)) and fstype for Azure disk ([#61267](https://github.com/kubernetes/kubernetes/pull/61267), [@andyzhangx](https://github.com/andyzhangx))
* Improvements in Windows Server version 1803 also bring new storage functionality to Kubernetes v1.11, including:
   * Volume mounts for ConfigMap and Secret
   * Flexvolume plugins for SMB and iSCSI storage are also available out-of-tree at [Microsoft/K8s-Storage-Plugins](https://github.com/Microsoft/K8s-Storage-Plugins)
* Setup dns servers and search domains for Windows Pods in dockershim. Docker EE version >= 17.10.0 is required for propagating DNS to containers. ([#63905](https://github.com/kubernetes/kubernetes/pull/63905), [@feiskyer](https://github.com/feiskyer))

### Additional changes

* Extended the Stackdriver Metadata Agent by adding a new Deployment for ingesting unscheduled pods and services.   ([#62043](https://github.com/kubernetes/kubernetes/pull/62043), [@supriyagarg](https://github.com/supriyagarg))
* Added all kinds of resource objects' statuses in HPA description. ([#59609](https://github.com/kubernetes/kubernetes/pull/59609), [@zhangxiaoyu-zidif](https://github.com/zhangxiaoyu-zidif))
* Implemented preemption for extender with a verb and new interface ([#58717](https://github.com/kubernetes/kubernetes/pull/58717), [@resouer](https://github.com/resouer))
* Updated nvidia-gpu-device-plugin DaemonSet config to use RollingUpdate updateStrategy instead of OnDelete. ([#64296](https://github.com/kubernetes/kubernetes/pull/64296), [@mindprince](https://github.com/mindprince))
* increased grpc client default response size. ([#63977](https://github.com/kubernetes/kubernetes/pull/677), [@runcom](https://github.com/runcom))
* Applied pod name and namespace labels to pod cgroup in cAdvisor metrics ([#63406](https://github.com/kubernetes/kubernetes/pull/63406), [@derekwaynecarr](https://github.com/derekwaynecarr))
* [fluentd-gcp addon] Use the logging agent's node name as the metadata agent URL. ([#63353](https://github.com/kubernetes/kubernetes/pull/63353), [@bmoyles0117](https://github.com/bmoyles0117))
* The new default value for the --allow-privileged parameter of the Kubernetes-worker charm has been set to true based on changes which went into the Kubernetes 1.10 release. Before this change the default value was set to false. If you're installing Canonical Kubernetes you should expect this value to now be true by default and you should now look to use PSP (pod security policies).  ([#64104](https://github.com/kubernetes/kubernetes/pull/64104), [@CalvinHartwell](https://github.com/CalvinHartwell))

## External Dependencies

* Default etcd server version is v3.2.18 compared with v3.1.12 in v1.10 ([#61198](https://github.com/kubernetes/kubernetes/pull/61198))
* Rescheduler is v0.4.0, compared with v0.3.1 in v1.10 ([#65454](https://github.com/kubernetes/kubernetes/pull/65454))
* The validated docker versions are the same as for v1.10: 1.11.2 to 1.13.1 and 17.03.x (ref)
* The Go version is go1.10.2, as compared to go1.9.3 in v1.10. ([#63412](https://github.com/kubernetes/kubernetes/pull/63412))
* The minimum supported go is the same as for v1.10: go1.9.1. ([#55301](https://github.com/kubernetes/kubernetes/pull/55301))
* CNI is the same as v1.10: v0.6.0 ([#51250](https://github.com/kubernetes/kubernetes/pull/51250))
* CSI is updated to 0.3.0 as compared to 0.2.0 in v1.10. ([#64719](https://github.com/kubernetes/kubernetes/pull/64719))
* The dashboard add-on is the same as v1.10: v1.8.3. ([#517326](https://github.com/kubernetes/kubernetes/pull/57326))
* Bump Heapster to v1.5.2 as compared to v1.5.0 in v1.10 ([#61396](https://github.com/kubernetes/kubernetes/pull/61396))
* Updates Cluster Autoscaler version to v1.3.0 from v1.2.0 in v1.10. See [release notes](https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.0) for details. ([#65219](https://github.com/kubernetes/kubernetes/pull/65219))
* Kube-dns has been updated to v1.14.10, as compared to v1.14.8 in v1.10 ([#62676](https://github.com/kubernetes/kubernetes/pull/62676))
* Influxdb is unchanged from v1.10: v1.3.3 ([#53319](https://github.com/kubernetes/kubernetes/pull/53319))
* Grafana is unchanged from v1.10: v4.4.3 ([#53319](https://github.com/kubernetes/kubernetes/pull/53319))
* CAdvisor is v0.30.1, as opposed to v0.29.1 in v1.10 ([#64987](https://github.com/kubernetes/kubernetes/pull/64987))
* fluentd-gcp-scaler is unchanged from v1.10: v0.3.0 ([#61269](https://github.com/kubernetes/kubernetes/pull/61269))
* fluentd in fluentd-es-image is unchanged from 1.10: v1.1.0 ([#58525](https://github.com/kubernetes/kubernetes/pull/58525))
* fluentd-elasticsearch is unchanged from 1.10: v2.0.4 ([#58525](https://github.com/kubernetes/kubernetes/pull/58525))
* fluentd-gcp is unchanged from 1.10: v3.0.0. ([#60722](https://github.com/kubernetes/kubernetes/pull/60722))
* Ingress glbc is unchanged from 1.10: v1.0.0 ([#61302](https://github.com/kubernetes/kubernetes/pull/61302))
* OIDC authentication is unchanged from 1.10: coreos/go-oidc v2 ([#58544](https://github.com/kubernetes/kubernetes/pull/58544))
* Calico is unchanged from 1.10: v2.6.7 ([#59130](https://github.com/kubernetes/kubernetes/pull/59130))
* hcsshim has been updated to v0.6.11 ([#64272](https://github.com/kubernetes/kubernetes/pull/64272))
* gitRepo volumes in pods no longer require git 1.8.5 or newer; older git versions are now supported. ([#62394](https://github.com/kubernetes/kubernetes/pull/62394))
* Update crictl on GCE to v1.11.0. ([#65254](https://github.com/kubernetes/kubernetes/pull/65254))
* CoreDNS is now v1.1.3 ([#64258](https://github.com/kubernetes/kubernetes/pull/64258))
* Setup dns servers and search domains for Windows Pods in dockershim. Docker EE version >= 17.10.0 is required for propagating DNS to containers. ([#63905](https://github.com/kubernetes/kubernetes/pull/63905))
* Update version of Istio addon from 0.5.1 to 0.8.0. See [full Istio release notes](https://istio.io/about/notes/0.6.html).([#64537](https://github.com/kubernetes/kubernetes/pull/64537))
* Update cadvisor godeps to v0.30.0 ([#64800](https://github.com/kubernetes/kubernetes/pull/64800))
* Update event-exporter to version v0.2.0  that supports old (gke_container/gce_instance) and new (k8s_container/k8s_node/k8s_pod) stackdriver resources. ([#63918](https://github.com/kubernetes/kubernetes/pull/63918))
* Rev the Azure SDK for networking to 2017-06-01 ([#61955](https://github.com/kubernetes/kubernetes/pull/61955))

## Bug Fixes

* Fixed spurious "unable to find api field" errors patching custom resources ([#63146](https://github.com/kubernetes/kubernetes/pull/63146), [@liggitt](https://github.com/liggitt))
* Nodes are not deleted from kubernetes anymore if node is shutdown in Openstack. ([#59931](https://github.com/kubernetes/kubernetes/pull/59931), [@zetaab](https://github.com/zetaab))
* Re-enabled nodeipam controller for external clouds. Re-enables nodeipam controller for external clouds. Also does a small refactor so that we don't need to pass in allocateNodeCidr into the controller.
 ([#63049](https://github.com/kubernetes/kubernetes/pull/63049), [@andrewsykim](https://github.com/andrewsykim))
* Fixed a configuration error when upgrading kubeadm from 1.9 to 1.10+; Kubernetes must have the same major and minor versions as the kubeadm library. ([#62568](https://github.com/kubernetes/kubernetes/pull/62568), [@liztio](https://github.com/liztio))
* kubectl no longer renders a List as suffix kind name for CRD resources ([#62512](https://github.com/kubernetes/kubernetes/pull/62512), [@dixudx](https://github.com/dixudx))
* Restored old behavior to the `--template` flag in `get.go`. In old releases, providing a `--template` flag value and no `--output` value implicitly assigned a default value ("go-template") to `--output`, printing using the provided template argument.
([#65377](https://github.com/kubernetes/kubernetes/pull/65377),[@juanvallejo](https://github.com/juanvallejo))
* Ensured cloudprovider.InstanceNotFound is reported when the VM is not found on Azure ([#61531](https://github.com/kubernetes/kubernetes/pull/61531), [@feiskyer](https://github.com/feiskyer))
* Kubernetes version command line parameter in kubeadm has been updated to drop an unnecessary redirection from ci/latest.txt to ci-cross/latest.txt. Users should know exactly where the builds are stored on Google Cloud storage buckets from now on. For example for 1.9 and 1.10, users can specify ci/latest-1.9 and ci/latest-1.10 as the CI build jobs what build images correctly updates those. The CI jobs for master update the ci-cross/latest location, so if you are looking for latest master builds, then the correct parameter to use would be ci-cross/latest. ([#63504](https://github.com/kubernetes/kubernetes/pull/63504), [@dims](https://github.cm/dims))
* Fixes incompatibility with custom scheduler extender configurations specifying `bindVerb` ([#65424](https://github.com/kubernetes/kubernetes/pull/65424), [@liggitt](https://github.com/liggitt))
* kubectl built for darwin from darwin now enables cgo to use the system-native C libraries for DNS resolution. Cross-compiled kubectl (e.g. from an official kubernetes release) still uses the go-native netgo DNS implementation.  ([#64219](https://github.com/kubernetes/kubernetes/pull/64219), [@ixdy](https://github.com/ixdy))
* API server properly parses propagationPolicy as a query parameter sent with a delete request ([#63414](https://github.com/kubernetes/kubernetes/pull/63414), [@roycaihw](https://github.com/roycaihw))
* Corrected a race condition in bootstrapping aggregated cluster roles in new HA clusters ([#63761](https://github.com/kubernetes/kubernetes/pull/63761), [@liggitt](https://github.com/liggitt))
* kubelet: fix hangs in updating Node status after network interruptions/changes between the kubelet and API server ([#63492](https://github.com/kubernetes/kubernetes/pull/63492), [@liggitt](https://github.com/liggitt))
* Added log and fs stats for Windows containers ([#62266](https://github.com/kubernetes/kubernetes/pull/62266), [@feiskyer](https://github.com/feiskyer))
* Fail fast if cgroups-per-qos is set on Windows ([#62984](https://github.com/kubernetes/kubernetes/pull/62984), [@feiskyer](https://github.com/feiskyer))
* Minor fix for VolumeZoneChecker predicate, storageclass can be in annotation and spec. ([#63749](https://github.com/kubernetes/kubernetes/pull/63749), [@wenlxie](https://github.com/wenlxie))
* Fixes issue for readOnly subpath mounts for SELinux systems and when the volume mountPath already existed in the container image. ([#64351](https://github.com/kubernetes/kubernetes/pull/64351), [@msau42](https://github.com/msau42))
* Fixed CSI gRPC connection leak during volume operations. ([#64519](https://github.com/kubernetes/kubernetes/pull/64519), [@vladimirvivien](https://github.com/vladimirvivien))
* Fixed error reporting of CSI volumes attachment. ([#63303](https://github.com/kubernetes/kubernetes/pull/63303), [@jsafrane](https://github.com/jsafrane))
* Fixed SELinux relabeling of CSI volumes. ([#64026](https://github.com/kubernetes/kubernetes/pull/64026), [@jsafrane](https://github.com/jsafrane))
* Fixed detach of already detached CSI volumes. ([#63295](https://github.com/kubernetes/kubernetes/pull/63295), [@jsafrane](https://github.com/jsafrane))
* fix rbd device works at block mode not get mapped to container  ([#64555](https://github.com/kubernetes/kubernetes/pull/64555), [@wenlxie](https://github.com/wenlxie))
* Fixed an issue where Portworx PVCs remain in pending state when created using a StorageClass with empty parameters ([#64895](https://github.com/kubernetes/kubernetes/pull/64895), [@harsh-px](https://github.com/harsh-px)) storage
* FIX: The OpenStack cloud providers DeleteRoute method fails to delete routes when it can’t find the corresponding instance in OpenStack. (#62729, databus23)
* [fluentd-gcp addon] Increase CPU limit for fluentd to 1 core to achieve 100kb/s throughput. ([#62430](https://github.com/kubernetes/kubernetes/pull/62430), [@bmoyles0117](https://github.com/bmoyles0117))
* GCE: Fixed operation polling to adhere to the specified interval. Furthermore, operation errors are now returned instead of ignored. ([#64630](https://github.com/kubernetes/kubernetes/pull/64630), [@nicksardo](https://github.com/nicksardo))
* Included kms-plugin-container.manifest to master nifests tarball. ([#65035](https://github.com/kubernetes/kubernetes/pull/65035), [@immutableT](https://github.com/immutableT))
* Fixed missing nodes lines when kubectl top nodes ([#64389](https://github.com/kubernetes/kubernetes/pull/64389), [@yue9944882](https://github.com/yue9944882)) sig-cli
* Fixed kubectl drain --timeout option when eviction is used. ([#64378](https://github.com/kubernetes/kubernetes/pull/64378), [@wrdls](https://github.com/wrdls)) sig-cli
* Fixed kubectl auth can-i exit code. It will return 1 if the user is not allowed and 0 if it's allowed. ([#59579](https://github.com/kubernetes/kubernetes/pull/59579), [@fbac](https://github.com/fbac))
* Fixed data loss issue if using existing azure disk with partitions in disk mount  ([#63270](https://github.com/kubernetes/kubernetes/pull/63270), [@andyzhangx](https://github.com/andyzhangx))
* Fixed azure file size grow issue ([#64383](https://github.com/kubernetes/kubernetes/pull/64383), [@andyzhangx](https://github.com/andyzhangx))
* Fixed SessionAffinity not updated issue for Azure load balancer ([#64180](https://github.com/kubernetes/kubernetes/pull/64180), [@feiskyer](https://github.com/feiskyer))
* Fixed kube-controller-manager panic while provisioning Azure security group rules ([#64739](https://github.com/kubernetes/kubernetes/pull/64739), [@feiskyer](https://github.com/feiskyer))
* Fixed API server panic during concurrent GET or LIST requests with non-empty `resourceVersion`. ([#65092](https://github.com/kubernetes/kubernetes/pull/65092), [@sttts](https://github.com/sttts))
* Fixed incorrect OpenAPI schema for CustomResourceDefinition objects ([#65256](https://github.com/kubernetes/kubernetes/pull/65256), [@liggitt](https://github.com/liggitt))
* Fixed issue where PersistentVolume.NodeAffinity.NodeSelectorTerms were ANDed instead of ORed. ([#62556](https://github.com/kubernetes/kubernetes/pull/62556), [@msau42](https://github.com/msau42))
* Fixed potential infinite loop that can occur when NFS PVs are recycled. ([#62572](https://github.com/kubernetes/kubernetes/pull/62572), [@joelsmith](https://github.com/joelsmith))
* Fixed column alignment when kubectl get is used with custom columns from OpenAPI schema ([#56629](https://github.com/kubernetes/kubernetes/pull/56629), [@luksa](https://github.com/luksa))
* kubectl: restore the ability to show resource kinds when displaying multiple objects ([#61985](https://github.com/kubernetes/kubernetes/pull/61985), [@liggitt](https://github.com/liggitt))
* Fixed a panic in `kubectl run --attach ...` when the api server failed to create the runtime object (due to name conflict, PSP restriction, etc.) ([#61713](https://github.com/kubernetes/kubernetes/pull/61713), [@mountkin](https://github.com/mountkin))
* kube-scheduler has been fixed to use `--leader-elect` option back to true (as it was in previous versions) ([#59732](https://github.com/kubernetes/kubernetes/pull/59732), [@dims](https://github.com/dims))
* kubectl: fixes issue with `-o yaml` and `-o json` omitting kind and apiVersion when used with `--dry-run` ([#61808](https://github.com/kubernetes/kubernetes/pull/61808), [@liggitt](https://github.com/liggitt))
* Ensure reasons end up as comments in `kubectl edit`. ([#60990](https://github.com/kubernetes/kubernetes/pull/60990), [@bmcstdio](https://github.com/bmcstdio))
* Fixes issue where subpath readOnly mounts failed ([#63045](https://github.com/kubernetes/kubernetes/pull/63045), [@msau42](https://github.com/msau42))
* Fix stackdriver metrics for node memory using wrong metric type ([#63535](https://github.co/kubernetes/kubernetes/pull/63535), [@serathius](https://github.com/serathius))
* fix mount unmount failure for a Windows pod ([#63272](https://github.com/kubernetes/kubernetes/pull/63272), [@andyzhangx](https://github.com/andyzhangx))

#### General Fixes and Reliability

* Fixed a regression in kube-scheduler to properly load client connection information from a `--config` file that references a kubeconfig file. ([#65507](https://github.com/kubernetes/kubernetes/pull/65507), [@liggitt](https://github.com/liggitt))
* Fix regression in `v1.JobSpec.backoffLimit` that caused failed Jobs to be restarted indefinitely. ([#63650](https://github.com/kubernetes/kubernetes/pull/63650), [@soltysh](https://github.com/soltysh))
* fixes a potential deadlock in the garbage collection controller ([#64235](https://github.com/kubernetes/kubernetes/pull/64235), [@liggitt](https://github.com/liggitt))
* fix formatAndMount func issue on Windows ([#63248](https://github.com/kubernetes/kubernetes/pull/63248), [@andyzhangx](https://github.com/andyzhangx))
* Fix issue of colliding nodePorts when the cluster has services with externalTrafficPolicy=Local ([#64349](https://github.com/kubernetes/kubernetes/pull/64349), [@nicksardo](https://github.com/nicksardo))
* fixes a panic applying json patches containing out of bounds operations ([#64355](https://github.com/kubernetes/kubernetes/pull/64355), [@liggitt](https://github.com/liggitt))
* Fix incorrectly propagated ResourceVersion in ListRequests returning 0 items. ([#64150](https://github.com/kubernetes/kubernetes/pull/64150), [@wojtek-t](https://github.com/wojtek-t))
* GCE: Fix to make the built-in `kubernetes` service properly point to the master's load balancer address in clusters that use multiple master VMs. ([#63696](https://github.com/kubernetes/kubernetes/pull/63696), [@grosskur](https://github.com/grosskur))
* Fixes fake client generation for non-namespaced subresources ([#60445](https://github.com/kubernetes/kubernetes/pull/60445), [@jhorwit2](https://github.com/jhorwit2))
* Schedule even if extender is not available when using extender  ([#61445](https://github.com/kubernetes/kubernetes/pull/61445), [@resouer](https://github.com/resouer))
* Fix panic create/update CRD when mutating/validating webhook configured. ([#61404](https://github.com/kubernetes/kubernetes/pull/61404), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Pods requesting resources prefixed with `*kubernetes.io` will remain unscheduled if there are no nodes exposing that resource. ([#61860](https://github.com/kubernetes/kubernetes/pull/61860), [@mindprince](https://github.com/mindprince))
* fix scheduling policy on ConfigMap breaks without the --policy-configmap-namespace flag set ([#61388](https://github.com/kubernetes/kubernetes/pull/61388), [@zjj2wry](https://github.com/zjj2wry))
* Bugfix for erroneous upgrade needed messaging in kubernetes worker charm. ([#60873](https://github.com/kubernetes/kubernetes/pull/60873), [@wwwtyro](https://github.com/wwwtyro))
* Fix inter-pod anti-affinity check to consider a pod a match when all the anti-affinity terms match. ([#62715](https://github.com/kubernetes/kubernetes/pull/62715), [@bsalamat](https://github.com/bsalamat))
* Pod affinity `nodeSelectorTerm.matchExpressions` may now be empty, and works as previously documented: nil or empty `matchExpressions` matches no objects in scheduler. ([#62448](https://github.com/kubernetes/kubernetes/pull/62448), [@k82cn](https://github.com/k82cn))
* Fix an issue in inter-pod affinity predicate that cause affinity to self being processed correctly ([#62591](https://github.com/kubernetes/kubernetes/pull/62591), [@bsalamat](https://github.com/bsalamat))
* fix WaitForAttach failure issue for azure disk ([#62612](https://github.com/kubernetes/kubernetes/pull/62612), [@andyzhangx](https://github.com/andyzhangx))
* Fix user visible files creation for windows ([#62375](https://github.com/kubernetes/kubernetes/pull/62375), [@feiskyer](https://github.com/feiskyer))
* Fix machineID getting for vmss nodes when using instance metadata ([#62611](https://github.com/kubernetes/kubernetes/pull/62611), [@feiskyer](https://github.com/feiskyer))
* Fix Forward chain default reject policy for IPVS proxier ([#62007](https://github.com/kubernetes/kubernetes/pull/62007), [@m1093782566](https://github.com/m1093782566))
* fix nsenter GetFileType issue in containerized kubelet ([#62467](https://github.com/kubernetes/kubernetes/pull/62467), [@andyzhangx](https://github.com/andyzhangx))
* Ensure expected load balancer is selected for Azure ([#62450](https://github.com/kubernetes/kubernetes/pull/62450), [@feiskyer](https://github.com/feiskyer))
* Resolves forbidden error when the `daemon-set-controller` cluster role access `controllerrevisions` resources. ([#62146](https://github.com/kubernetes/kubernetes/pull/62146), [@frodenas](https://github.com/frodenas))
* fix incompatible file type checking on Windows ([#62154](https://github.com/kubernetes/kubernetes/pull/62154), [@dixudx](https://github.com/dixudx))
* fix local volume absolute path issue on Windows ([#620s18](https://github.com/kubernetes/kubernetes/pull/62018), [@andyzhangx](https://github.com/andyzhangx))
* fix the issue that default azure disk fsypte(ext4) does not work on Windows ([#62250](https://github.com/kubernetes/kubernetes/pull/62250), [@andyzhangx](https://github.com/andyzhangx))
* Fixed bug in rbd-nbd utility when nbd is used. ([#62168](https://github.com/kubernetes/kubernetes/pull/62168), [@piontec](https://github.com/piontec))
* fix local volume issue on Windows ([#62012](https://github.com/kubernetes/kubernetes/pull/62012), [@andyzhangx](https://github.com/andyzhangx))
* Fix a bug that fluentd doesn't inject container logs for CRI container runtimes (containerd, cri-o etc.) into elasticsearch on GCE. ([#61818](https://github.com/kubernetes/kubernetes/pull/61818), [@Random-Liu](https://github.com/Random-Liu))
* flexvolume: trigger plugin init only for the relevant plugin while probe ([#58519](https://github.com/kubernetes/kubernetes/pull/58519), [@linyouchong](https://github.com/linyouchong))
* Fixed ingress issue with CDK and pre-1.9 versions of kubernetes. ([#61859](https://github.com/kubernetes/kubernetes/pull/61859), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Fixed racy panics when using fake watches with ObjectTracker ([#61195](https://github.com/kubernetes/kubernetes/pull/61195), [@grantr](https://github.com/grantr))
* Fixed mounting of UNIX sockets(and other special files) in subpaths ([#61480](https://github.com/kubernetes/kubernetes/pull/61480), [@gnufscied](https://github.com/gnufied))
* Fixed [#61123](https://github.com/kubernetes/kubernetes/pull/61123) by triggering syncer.Update on all cases including when a syncer is created ([#61124](https://github.com/kubernetes/kubernetes/pull/61124), [@satyasm](https://github.com/satyasm))
* Fixed data race in node lifecycle controller ([#60831](https://github.com/kubernetes/kubernetes/pull/60831), [@resouer](https://github.com/resouer))
* Fixed resultRun by resetting it to 0 on pod restart ([#62853](https://github.com/kubernetes/kubernetes/pull62853), [@tony612](https://github.com/tony612))
* Fixed the liveness probe to use `/bin/bash -c` instead of `/bin/bash c`. ([#63033](https://github.com/kubernetes/kubernetes/pull/63033), [@bmoyles0117](https://github.com/bmoyles0117))
* Fixed scheduler informers to receive events for all the pods in the cluster. ([#63003](https://github.com/kubernetes/kubernetes/pull/63003), [@bsalamat](https://github.com/bsalamat))
* Fixed in vSphere Cloud Provider to handle upgrades from kubernetes version less than v1.9.4 to v1.9.4 and above. ([#62919](https://github.com/kubernetes/kubernetes/pull/62919), [@abrarshivani](https://github.com/abrarshivani))
* Fixed error where config map for Metadata Agent was not created by addon manager. ([#62909](https://github.com/kubernetes/kubernetes/pull/62909), [@kawych](https://github.com/kawych))
* Fixed permissions to allow statefulset scaling for admins, editors, and viewers ([#62336](https://github.com/kubernetes/kubernetes/pull/62336), [@deads2k](https://github.com/deads2k))
* GCE: Fixed for internal load balancer management resulting in backend services with outdated instance group links. ([#62885](https://github.com/kubernetes/kubernetes/pull/62885), [@nicksardo](https://github.com/nicksardo))
* Deployment will stop adding pod-template-hash labels/selector to ReplicaSets and Pods it adopts. Resources created by Deployments are not affected (will still have pod-template-hash labels/selector).  ([#61615](https://github.com/kubernetes/kubernetes/pull/61615), [@janetkuo](https://github.com/janetkuo))
* Used inline func to ensure unlock is executed ([#61644](https://github.com/kubernetes/kubernetes/pull/61644), [@resouer](https://github.com/resouer))
* kubernetes-master charm now properly clears the client-ca-file setting on the apiserver snap ([#61479](https://github.com/kubernetes/kubernetes/pull/61479), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Bound cloud allocator to 10 retries with 100 ms delay between retries. ([#61375](https://github.com/kubernetes/kubernetes/pull/61375), [@satyasm](https://github.com/satyasm))
* Respect fstype in Windows for azure disk ([#61267](https://github.com/kubernetes/kubernetes/pull/61267), [@andyzhangx](https://github.com/andyzhangx))
* Unready pods will no longer impact the number of desired replicas when using horizontal auto-scaling with external metrics or object metrics. ([#60886](https://github.com/kubernetes/kubernetes/pull/60886), [@mattjmcnaughton](https://github.com/mattjmcnaughton))
* Removed unsafe double RLock in cpumanager ([#62464](https://github.com/kubernetes/kubernetes/pull/62464), [@choury](https://github.com/choury))

## Non-user-facing changes

* Remove UID mutation from request.context. ([#63957](https://github.com/kubernetes/kubernetes/pull/63957), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Use Patch instead of Put to sync pod status. ([#62306](https://github.com/kubernetes/kubernetes/pull/62306), [@freehan](https://github.com/freehan))
* Allow env from resource with keys & updated tests ([#60636](https://github.com/kubernetes/kubernetes/pull/60636), [@PhilipGough](https://github.com/PhilipGough))
* set EnableHTTPSTrafficOnly in azure storage account creation ([#64957](https://github.com/kubernetes/kubernetes/pull/64957), [@andyzhangx](https://github.com/andyzhangx))
* New conformance test added for Watch. ([#61424](https://github.com/kubernetes/kubernetes/pull/61424), [@jennybuckley](https://github.com/jennybuckley))
* Use DeleteOptions.PropagationPolicy instead of OrphanDependents in kubectl  ([#59851](https://github.com/kubernetes/kubernetes/pull/59851), [@nilebox](https://github.com/nilebox))
* Add probe based mechanism for kubelet plugin discovery ([#63328](https://github.com/kubernetes/kubernetes/pull/63328), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* keep pod state consistent when scheduler cache UpdatePod ([#64692](https://github.com/kubernetes/kubernetes/pull/64692),  [@adohe](https://github.com/adohe))
* kubectl delete does not use reapers for removing objects anymore, but relies on server-side GC entirely ([#63979](https://github.com/kubernetes/kubernetes/pull/63979), [@soltysh](https://github.com/soltysh))
* Updated default image for nginx ingress in CDK to match current Kubernetes docs. ([#64285](https://github.com/kubernetes/kubernetes/pull/64285), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Increase scheduler cache generation number monotonically in order to avoid collision and use of stale information in scheduler. ([#63264](https://github.com/kubernetes/kubernetes/pull/63264), [@bsalamat](https://github.com/bsalamat))
* Adding CSI driver registration code. ([#64560](https://github.com/kubernetes/kubernetes/pull/64560), [@sbezverk](https://github.com/sbezverk))
* Do not check vmSetName when getting Azure node's IP ([#63541](https://github.com/kubernetes/kubernetes/pull/63541), [@feiskyer](https://github.com/feiskyer))
* [fluentd-gcp addon] Update event-exporter image to have the latest base image. ([#61727](https://github.com/kubernetes/kubernetes/pull/61727), [@crassirostris](https://github.com/crassirostris))
* Make volume usage metrics available for Cinder ([#62668](https://github.com/kubernetes/kubernetes/pull/62668), [@zetaab](https://github.com/zetaab))
* cinder volume plugin : When the cinder volume status is `error`,  controller will not do `attach ` and `detach ` operation ([#61082](https://github.com/kubernetes/kubernetes/pull/61082), [@wenlxie](https://github.com/wenlxie))
* Allow user to scale l7 default backend deployment ([#62685](https://github.com/kubernetes/kubernetes/pull/62685), [@freehan](https://github.com/freehan))
* Add support to ingest log entries to Stackdriver against new "k8s_container" and "k8s_node" resources. ([#62076](https://github.com/kubernetes/kubernetes/pull/62076), [@qingling128](https://github.com/qingling128))
* Disabled CheckNodeMemoryPressure and CheckNodeDiskPressure predicates if TaintNodesByCondition enabled ([#60398](https://github.com/kubernetes/kubernetes/pull/60398), [@k82cn](https://github.com/k82cn))
* Support custom test configuration for IPAM performance integration tests ([#61959](https://github.com/kubernetes/kubernetes/pull/61959), [@satyasm](https://github.com/satyasm))
* OIDC authentication now allows tokens without an "email_verified" claim when using the "email" claim. If an "email_verified" claim is present when using the "email" claim, it must be `true`. ([#61508](https://github.com/kubernetes/kubernetes/pull/61508), [@rithujohn191](https://github.com/rithujohn191))
* Add e2e test for CRD Watch ([#61025](https://github.com/kubernetes/kubernetes/pull/61025), [@ayushpateria](https://github.com/ayushpateria))
* Return error if get NodeStageSecret and NodePublishSecret failed in CSI volume plugin ([#61096](https://github.com/kubernetes/kubernetes/pull/61096), [@mlmhl](https://github.com/mlmhl))
* kubernetes-master charm now supports metrics server for horizontal pod autoscaler. ([#60174](https://github.com/kubernetes/kubernetes/pull/60174), [@hyperbolic2346](https://github.com/hyperbolic2346))
* In a GCE cluster, the default `HIRPIN_MODE` is now "hairpin-veth". ([#60166](https://github.com/kubernetes/kubernetes/pull/60166), [@rramkumar1](https://github.com/rramkumar1))
* Balanced resource allocation priority in scheduler to include volume count on node  ([#60525](https://github.com/kubernetes/kubernetes/pull/60525), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* new dhcp-domain parameter to be used for figuring out the hostname of a node ([#61890](https://github.com/kubernetes/kubernetes/pull/61890), [@dims](https://github.com/dims))
* Disable ipamperf integration tests as part of every PR verification. ([#61863](https://github.com/kubernetes/kubernetes/pull/61863), [@satyasm](https://github.com/satyasm))
* Enable server-side print in kubectl by default, with the ability to turn it off with --server-print=false ([#61477](https://github.com/kubernetes/kubernetes/pull/61477), [@soltysh](https://github.com/soltysh))
* Updated admission controller settings for Juju deployed Kubernetes clusters ([#61427](https://github.com/kubernetes/kubernetes/pull/61427), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Performance test framework and basic tests for the IPAM controller, to simulate behavior of the four supported modes under lightly loaded and loaded conditions, where load is defined as the number of operations to perform as against the configured kubernetes. ([#61143](https://github.com/kubernetes/kubernetes/pull/61143), [@satyasm](https://github.com/satyasm))
* Removed always pull policy from the template for ingress on CDK. ([#61598](https://github.com/kubernetes/kubernetes/pull/61598), [@hyperbolic2346](https://github.com/hyperbolic2346))
* `make test-cmd` now works on OSX. ([#61393](https://github.com/kubernetes/kubernetes/pull/61393), [@totherme](https://github.com/totherme))
* Conformance: ReplicaSet must be supported in the `apps/v1` version. ([#61367](https://github.com/kubernetes/kubernetes/pull/61367), [@enisoc](https://github.com/enisoc))
* Remove 'system' prefix from Metadata Agent rbac configuration ([#61394](https://github.com/kubernetes/kubernetes/pull/61394), [@kawych](https://github.com/kawych))
* Support new NODE_OS_DISTRIBUTION 'custom' on GCE on a new add event. ([#61235](https://github.com/kubernetes/kubernetes/pull/61235), [@yguo0905](https://github.com/yguo0905))
* include file name in the error when visiting files ([#60919](https://github.com/kubernetes/kubernetes/pull/60919), [@dixudx](https://github.com/dixudx))
* Split PodPriority and PodPreemption feature gate ([#62243](https://github.com/kubernetes/kubernetes/pull/62243), [@resouer](https://github.com/resouer))
* Code generated for CRDs now passes `go vet`. ([#62412](https://github.com/kubernetes/kubernetes/pull/62412), [@bhcleek](https://github.com/bhcleek))
* "beginPort+offset" format support for port range which affects kube-proxy only ([#58731](https://github.com/kubernetes/kubernetes/pull/58731), [@yue9944882](https://github.com/yue9944882))
* Added e2e test for watch ([#60331](https://github.com/kubernetes/kubernetes/pull/60331), [@jennybuckley](https://github.com/jennybuckley))
* add warnings on using pod-infra-container-image for remote container runtime ([#62982](https://github.com/kubernetes/kubernetes/pull/62982), [@dixudx](https://github.com/dixudx))
* Mount additional paths required for a working CA root, for setups where /etc/ssl/certs doesn't contains certificates but just symlink. ([#59122](https://github.com/kubernetes/kubernetes/pull/59122), [@klausenbusk](https://github.com/klausenbusk))
* Introduce truncating audit bacnd that can be enabled for existing backend to limit the size of individual audit events and batches of events. ([#61711](https://github.com/kubernetes/kubernetes/pull/61711), [@crassirostris](https://github.com/crassirostris))
* stop kubelet to cloud provider integration potentially wedging kubelet sync loop ([#62543](https://github.com/kubernetes/kubernetes/pull/62543), [@ingvagabund](https://github.com/ingvagabund))
* Set pod status to "Running" if there is at least one container still reporting as "Running" status and others are "Completed". ([#62642](https://github.com/kubernetes/kubernetes/pull/62642), [@ceshihao](https://github.com/ceshihao))
* Fix memory cgroup notifications, and reduce associated log spam. ([#63220](https://github.com/kubernetes/kubernetes/pull/63220), [@dashpole](https://github.com/dashpole))
* Remove never used NewCronJobControllerFromClient method (#59471, dmathieu)


- [v1.11.0-rc.3](#v1110-rc3)
- [v1.11.0-rc.2](#v1110-rc2)
- [v1.11.0-rc.1](#v1110-rc1)
- [v1.11.0-beta.2](#v1110-beta2)
- [v1.11.0-beta.1](#v1110-beta1)
- [v1.11.0-alpha.2](#v1110-alpha2)
- [v1.11.0-alpha.1](#v1110-alpha1)



# v1.11.0-rc.3

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0-rc.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes.tar.gz) | `25879ba96d7baf1eb9002956cef3ee40597ed7507784262881a09c00d35ab4c6`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-src.tar.gz) | `748786c0847e278530c790f82af52797de8b5a9e494e727d0049d4b35e370327`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-darwin-386.tar.gz) | `7a3c1b89d6787e275b4b6b855237da6964145e0234b82243c7c6803f1cbd3b46`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-darwin-amd64.tar.gz) | `0265652c3d7f98e36d1d591e3e6ec5018825b6c0cd37bf65c4d043dc313279e3`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-linux-386.tar.gz) | `600d9c83ba4d2126da1cfcd0c079d97c8ede75fad61bead1135dc9e4f7e325ce`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-linux-amd64.tar.gz) | `143fdaf82480dab68b1c783ae9f21916783335f3e4eaa132d72a2c1f7b4b393f`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-linux-arm.tar.gz) | `1bf4a0823c9c8128b19a2f0a8fbaf81226a313bc35132412a9fa1d251c2af07c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-linux-arm64.tar.gz) | `643b84a227838dd6f1dc6c874f6966e9f098b64fd7947ff940776613fa2addf0`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-linux-ppc64le.tar.gz) | `f46e1952046e977defd1a308ebe6de3ba6a710d562d17de987966a630ea2f7a3`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-linux-s390x.tar.gz) | `7ba61a3d8e6b50b238814eb086c6f9a9354342be9ac1882d0751d6cd2ce9f295`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-windows-386.tar.gz) | `587ca7b09cd45864b8093a8aa10284d473db1f528a6173cd2e58f336673aade0`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-client-windows-amd64.tar.gz) | `a8b1aac95def9f2bf54a5bbd2d83a1dd7778d0a08f1986187063a9a288a9079b`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-server-linux-amd64.tar.gz) | `d19cc5604370eb2fa826420c99dcbdbbb9bf096ea2916549a46ace990c09e20e`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-server-linux-arm.tar.gz) | `47b4ac984a855df2c78443a527705e45909da27405bb7cd8f257a5cde0314518`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-server-linux-arm64.tar.gz) | `09f8c2692f8de291c522fc96a5cbefcd60fe7a1ba9235251be11e6dda8663360`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-server-linux-ppc64le.tar.gz) | `594ff5991206887a70ec0c13624fa940f7ef4ce9cb17f9d8906f7a124a7ae4d1`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-server-linux-s390x.tar.gz) | `43a635f34ce473dcf52870e1d8fad324776d4d958b9829a3dce49eb07f8c4412`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-node-linux-amd64.tar.gz) | `b3259ed3bf2063aca9e6061cc27752adc4d787dfada4498bc4495cbc962826a2`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-node-linux-arm.tar.gz) | `9c71370709c345e4495708d8a2c03c1698f59cc9ca60678f498e895170530f9f`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-node-linux-arm64.tar.gz) | `d3d1cb767da267ebe8c03c7c6176490d5d047e33596704d099597ff50e5ae3b6`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-node-linux-ppc64le.tar.gz) | `d7c623d9ccce9cbb4c8a5d1432ac00222b54f420699d565416e09555e2cc7ff3`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-node-linux-s390x.tar.gz) | `288cd27f2e428a3e805c7fcc2c3945c0c6ee2db4812ad293e2bfd9f85bccf428`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.3/kubernetes-node-windows-amd64.tar.gz) | `991765513e0f778ec5416de456dfd709ed90a2fa97741f50dfdb0d30ee4ccbc0`

## Changelog since v1.11.0-rc.2

### Other notable changes

* Pass cluster_location argument to Heapster ([#65176](https://github.com/kubernetes/kubernetes/pull/65176), [@kawych](https://github.com/kawych))
* Fix concurrent map access panic ([#65331](https://github.com/kubernetes/kubernetes/pull/65331), [@dashpole](https://github.com/dashpole))
    * Don't watch .mount cgroups to reduce number of inotify watches
    * Fix NVML initialization race condition
    * Fix brtfs disk metrics when using a subdirectory of a subvolume
* User can now use `sudo crictl` on GCE cluster. ([#65389](https://github.com/kubernetes/kubernetes/pull/65389), [@Random-Liu](https://github.com/Random-Liu))



# v1.11.0-rc.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0-rc.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes.tar.gz) | `30742ea1e24ade88e148db872eeef58597813bc67d485c0ff6e4b7284d59500a`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-src.tar.gz) | `77e1f018820542088f1e9af453a139ae8ad0691cbde98ab01695a8f499dbe4cf`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-darwin-386.tar.gz) | `2f8777fcb938bbc310fb481a56dca62e14c27f6a85e61ab4650aeb28e5f9f05a`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-darwin-amd64.tar.gz) | `30a5ed844d2b6b6b75e19e1f68f5c18ff8ec4f268c149737a6e715bc0a6e297f`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-linux-386.tar.gz) | `e4c60f463366fdf62e9c10c45c6f6b75d63aa3bd6665a0b56c9c2e2104ea9da6`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-linux-amd64.tar.gz) | `1d62f9ac92f23897d4545ebaf15d78b13b04157d83a839e347f4bd02cc484af4`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-linux-arm.tar.gz) | `8f52c6da9f95c7e127a6945a164e66d5266ebf2f4d02261653c5dd6936ec6b00`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-linux-arm64.tar.gz) | `e6b677601f0d78cf9463a86d6cc33b4861a88d2fbf3728b9c449a216fb84578e`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-linux-ppc64le.tar.gz) | `2cd49eb1d5f6d97f1342ee7f4803e9713a9cf4bfa419c86f4e1f82182d27f535`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-linux-s390x.tar.gz) | `e8134efaea3146336b24e76ae2f6f5cdc63f6aeecc65b52cd0aae92edb8432ac`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-windows-386.tar.gz) | `226b8c687251c877d5876f95f086b131ff3f831fca01dd07caf168269ee2c51d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-client-windows-amd64.tar.gz) | `c590a3a7f2e08f8046752b5bbc0d0b11f174f750fdd7912a68dd5335fcedc03d`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-server-linux-amd64.tar.gz) | `13c518091348c1b4355bf6b1a72514e71f68ad68a51df7d0706666c488e51158`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-server-linux-arm.tar.gz) | `d4b4fa98ece74d2cc240cf43b59629fe0115d3750d5938ae5ece972251a96018`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-server-linux-arm64.tar.gz) | `6b9e9de414619fb28dbbee05537697c2fdce130abe65372b477d3858571bfabd`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-server-linux-ppc64le.tar.gz) | `537f27284ad47d37d9ab8c4f4113b90f55948f88cd5dbab203349a34a9ddeccb`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-server-linux-s390x.tar.gz) | `71299a59bd4b7b38242631b3f441885ca9dcd99934427c8399b4f4598cc47fbb`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-node-linux-amd64.tar.gz) | `792da4aa3c06dee14b10f219591af8e967e466c5d5646d8973abfb1071cb5202`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-node-linux-arm.tar.gz) | `40a276dd0efdd6e87206d9b2a994ba49c336a455bad7076ddb22a4a6aa0a885f`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-node-linux-arm64.tar.gz) | `867504f25a864130c28f18aa5e99be0b2a8e0223ea86d46a4033e76cbe865533`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-node-linux-ppc64le.tar.gz) | `b1ff4471acf84a0d4f43854c778d6e18f8d0358da1323d1812f1d1a922b56662`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-node-linux-s390x.tar.gz) | `b527ab6ad8f7a3220e743780412c2d6c7fdaccc4eaa71ccfe90ad3e4e98d1d80`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.2/kubernetes-node-windows-amd64.tar.gz) | `1643e19c7dd5b139a6ab81768249d62392fcad5f6f2aec7edab279009368898b`

## Changelog since v1.11.0-rc.1

### Other notable changes

* Prevents a `kubectl delete` hang when deleting controller managed lists ([#65367](https://github.com/kubernetes/kubernetes/pull/65367), [@deads2k](https://github.com/deads2k))
* fixes a memory leak in the kube-controller-manager observed when large numbers of pods with tolerations are created/deleted ([#65339](https://github.com/kubernetes/kubernetes/pull/65339), [@liggitt](https://github.com/liggitt))
* The "kubectl cp" command now supports path shortcuts (../) in remote paths. ([#65189](https://github.com/kubernetes/kubernetes/pull/65189), [@juanvallejo](https://github.com/juanvallejo))
* Split 'scheduling_latency_seconds' metric into finer steps (predicate, priority, premption) ([#65306](https://github.com/kubernetes/kubernetes/pull/65306), [@shyamjvs](https://github.com/shyamjvs))
* fixed incorrect OpenAPI schema for CustomResourceDefinition objects ([#65256](https://github.com/kubernetes/kubernetes/pull/65256), [@liggitt](https://github.com/liggitt))



# v1.11.0-rc.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0-rc.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes.tar.gz) | `f4d6126030d76f4340bf36ba02562388ea6984aa3d3f3ece39359c2a0f605b73`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-src.tar.gz) | `6383966a2bc5b252f1938fdfe4a7c35fafaa7642da22f86a017e2b718dedda92`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-darwin-386.tar.gz) | `1582a21d8e7c9ec8719a003cd79a7c51e984f2b7b703f0816af50efa4b838c6f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | `77ae2765fcac147095d2791f42b212a6c150764a311dfb6e7740a70d0c155574`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-linux-386.tar.gz) | `87f6e22ef05bcd468424b02da2a58c0d695bd875e2130cb94adb842988aa532c`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | `978147f7989b5669a74be5af7c6fe9b3039956c958d17dc53f65ae2364f8485c`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-linux-arm.tar.gz) | `e7e13c6f500f86641f62fcaa34715fd8aa40913fe97ac507a73a726fb6d2f3f4`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | `5e35f3c80f0811b252c725c938dc4803034b4925d6fa1c2f0042132fd19d6db2`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | `0cec908e2f85763e9f066661c2f12122b13901004f552729ced66673f12669da`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | `ae6e0d7eb75647531b224d8a873528bb951858bfddc9595771def8a26dd2a709`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-windows-386.tar.gz) | `9eaba9edce7e06c15088612b90c8adc714509cab8ba612019c960dc3fe306b9d`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | `dae41cc0be99bec6b28c8bd96eccd6c41b2d51602bc6a374dff922c34708354f`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | `73510e5be3650bdeb219e93f78b042b4c9b616cbe672c68cab2e713c13f040ca`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-server-linux-arm.tar.gz) | `00475cb20dbabbc7f1a048f0907ef1b2cf34cfacab3ad82d2d86e2afae466eca`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | `00b1a2fa9e7c6b9929e09d7e0ec9aadc3e697d7527dcda9cd7d57e89daf618f5`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | `6c2d303a243ca4452c19b613bc71c92222c33c9322983f9a485231a7d2471681`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | `c93d9021bd00bd1adda521e6952c72e08beebe8d994ad92cc14c741555e429a9`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | `7d84cd7f60186d59e84e4b48bc5cd25ddd0fbcef4ebb2a2a3bd06831433c0135`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-node-linux-arm.tar.gz) | `4fa046b5c0b3d860e741b33f4da722a16d4b7de9674ab6a60da2d5749b3175ef`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | `db80b1916da3262b1e3aeb658b9a9c829a76e85f97e30c5fc1b07a3ef331003a`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | `c693a8b7827f9098e8f407182febc24041dd396fdd66c61f8b666252fbbb342a`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | `ee5becf3f2034157e4c50488278095c3685a01b7f715693a1053fa986d983dcf`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | `65f4f7a96f89c8dcba6c21e79aeac677790c8338c3f8f0e9e27fb16154d7e06f`

## Changelog since v1.11.0-beta.2

### Action Required

* A cluster-autoscaler ClusterRole is added to cover only the functionality required by Cluster Autoscaler and avoid abusing system:cluster-admin role. ([#64503](https://github.com/kubernetes/kubernetes/pull/64503), [@kgolab](https://github.com/kgolab))
    * action required: Cloud providers other than GCE might want to update their deployments or sample yaml files to reuse the role created via add-on.

### Other notable changes

* The "kubectl cp" command now supports path shortcuts (../) in remote paths. ([#65189](https://github.com/kubernetes/kubernetes/pull/65189), [@juanvallejo](https://github.com/juanvallejo))
* Update crictl on GCE to v1.11.0. ([#65254](https://github.com/kubernetes/kubernetes/pull/65254), [@Random-Liu](https://github.com/Random-Liu))
* kubeadm: Use the release-1.11 branch by default ([#65229](https://github.com/kubernetes/kubernetes/pull/65229), [@luxas](https://github.com/luxas))
* Updates Cluster Autoscaler version to 1.3.0. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.0 ([#65219](https://github.com/kubernetes/kubernetes/pull/65219), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* The deprecated `--service-account-private-key-file` flag has been removed from the cloud-controller-manager. The flag is still present and supported in the kube-controller-manager. ([#65182](https://github.com/kubernetes/kubernetes/pull/65182), [@liggitt](https://github.com/liggitt))
* Update Cluster Autoscaler to v1.3.0-beta.2. Release notes for this version: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.0-beta.2 ([#65148](https://github.com/kubernetes/kubernetes/pull/65148), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* Fixed API server panic during concurrent GET or LIST requests with non-empty `resourceVersion`. ([#65092](https://github.com/kubernetes/kubernetes/pull/65092), [@sttts](https://github.com/sttts))
* Kubernetes json deserializer is now case-sensitive to restore compatibility with pre-1.8 servers. ([#65034](https://github.com/kubernetes/kubernetes/pull/65034), [@caesarxuchao](https://github.com/caesarxuchao))
    * If your config files contains fields with wrong case, the config files will be now invalid.
* GCE: Fixes operation polling to adhere to the specified interval. Furthermore, operation errors are now returned instead of ignored. ([#64630](https://github.com/kubernetes/kubernetes/pull/64630), [@nicksardo](https://github.com/nicksardo))
* Updated hcsshim dependency to v0.6.11 ([#64272](https://github.com/kubernetes/kubernetes/pull/64272), [@jessfraz](https://github.com/jessfraz))
* Include kms-plugin-container.manifest to master manifests tarball. ([#65035](https://github.com/kubernetes/kubernetes/pull/65035), [@immutableT](https://github.com/immutableT))
* kubeadm - Ensure the peer port is secured by explicitly setting the peer URLs for the default etcd instance. ([#64988](https://github.com/kubernetes/kubernetes/pull/64988), [@detiber](https://github.com/detiber))
    * kubeadm - Ensure that the etcd certificates are generated using a proper CN
    * kubeadm - Update generated etcd peer certificate to include localhost addresses for the default configuration.
    * kubeadm - Increase the manifest update timeout to make upgrades a bit more reliable.
* Kubernetes depends on v0.30.1 of cAdvisor ([#64987](https://github.com/kubernetes/kubernetes/pull/64987), [@dashpole](https://github.com/dashpole))
* Update Cluster Autoscaler version to 1.3.0-beta.1. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.3.0-beta.1 ([#64977](https://github.com/kubernetes/kubernetes/pull/64977), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* Webhooks for the mutating admission controller now support "remove" operation. ([#64255](https://github.com/kubernetes/kubernetes/pull/64255), [@rojkov](https://github.com/rojkov))
* deprecated and inactive option '--enable-custom-metrics' is removed in 1.11 ([#60699](https://github.com/kubernetes/kubernetes/pull/60699), [@CaoShuFeng](https://github.com/CaoShuFeng))
* kubernetes now packages cri-tools (crictl) in addition to all the other kubeadm tools in a deb and rpm. ([#64836](https://github.com/kubernetes/kubernetes/pull/64836), [@chuckha](https://github.com/chuckha))
* Fix setup of configmap/secret/projected/downwardapi volumes ([#64855](https://github.com/kubernetes/kubernetes/pull/64855), [@gnufied](https://github.com/gnufied))
* Setup dns servers and search domains for Windows Pods in dockershim. Docker EE version >= 17.10.0 is required for propagating DNS to containers. ([#63905](https://github.com/kubernetes/kubernetes/pull/63905), [@feiskyer](https://github.com/feiskyer))



# v1.11.0-beta.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0-beta.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes.tar.gz) | `0addbff3fc61047460da0fca7413f4cc679fac7482c3f09aa4f4a60d8ec8dd5c`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-src.tar.gz) | `943629abc5b046cc5db280417e5cf3a8342c5f67c8deb3d7283b02de67b3a3c3`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-darwin-386.tar.gz) | `9b714bb99e9d8c51c718d9ec719412b2006c921e6a5566acf387797b57014386`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-darwin-amd64.tar.gz) | `11fc9f94c82b2adc860964be8e84ed1e17ae711329cac3c7aff58067caeeffe2`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-linux-386.tar.gz) | `016abd161dc394ab6e1e8f57066ff413b523c71ac2af458bfc8dfa2107530910`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-linux-amd64.tar.gz) | `f98c223c24680aae583ff63fa8e1ef49421ddd660bd748fea493841c24ad6417`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-linux-arm.tar.gz) | `78cf5dca303314023d6f82c7570e92b814304029fb7d3941d7c04855679e120d`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-linux-arm64.tar.gz) | `c35e03687d491d9ca955121912c56d00741c86381370ed5890b0ee8b629a3e01`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-linux-ppc64le.tar.gz) | `4e848a58f822f971dbda607d26128d1b718fc07665d2f65b87936eec40b037b2`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-linux-s390x.tar.gz) | `ead83a70e4782efdaea3645ca2a59e51209041ce41f9d805d5c1d10f029b1cb0`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-windows-386.tar.gz) | `c357b28c83e769517d7b19e357260d62485e861005d98f84c752d109fa48bd20`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-client-windows-amd64.tar.gz) | `2ae78921a35a8a582b226521f904f0840c17e3e097364d6a3fcd10d196bec0dc`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-server-linux-amd64.tar.gz) | `26bd6e05a4bf942534f0578b1cdbd11b8c868aa3331e2681734ecc93d75f6b85`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-server-linux-arm.tar.gz) | `df706ccad0a235613e644eda363c49bfb858860a2ae5219b17b996f36669a7fc`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-server-linux-arm64.tar.gz) | `73f3e7a82d7c78a9f03ce0c84ae4904942f0bf88b3bf045fc9b1707b686cb04e`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-server-linux-ppc64le.tar.gz) | `ebeb67e45e630469d55b442d2c6092065f1c1403d1965c4340d0b6c1fa7f6676`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-server-linux-s390x.tar.gz) | `c82e6a41b8e451600fb5bfdad3addf3c35b5edb518a7bf9ebd03af0574d57975`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-node-linux-amd64.tar.gz) | `e6dbd56c10fee83f400e76ae02325eda0a583347f6b965eeb610c90d664d7990`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-node-linux-arm.tar.gz) | `df9d18c3af4d6ee237a238b3029823f6e90b2ae3f0d25b741d4b3fedb7ea14f8`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-node-linux-arm64.tar.gz) | `d84e98702651615336256d3453516df9ad39f39400f6091d9e2b4c95b4111ede`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-node-linux-ppc64le.tar.gz) | `a62037f00ab29302f72aa23116c304b676cc41a6f47f79a2faf4e4ea18059178`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-node-linux-s390x.tar.gz) | `bef66f2080f7ebf442234d841ec9c994089fa02b400d98e1b01021f1f66c4cd0`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.2/kubernetes-node-windows-amd64.tar.gz) | `2b029715b98c3355a172ed5a6e08e73ad4ef264c74a26ed5a3da67f90764b7dc`

## Changelog since v1.11.0-beta.1

### Action Required

* [action required] `kubeadm join` is now blocking on the kubelet performing the TLS Bootstrap properly. ([#64792](https://github.com/kubernetes/kubernetes/pull/64792), [@luxas](https://github.com/luxas))
    * Earlier, `kubeadm join` only did the discovery part and exited successfully without checking that the
    * kubelet actually started properly and performed the TLS bootstrap correctly. Now, as kubeadm runs
    * some post-join steps (e.g. annotating the Node API object with the CRISocket as in this PR, as a
    * stop-gap until this is discoverable automatically), `kubeadm join` is now waiting for the kubelet to
    * perform the TLS Bootstrap, and then uses that credential to perform further actions. This also
    * improves the UX, as `kubeadm` will exit with a non-zero code if the kubelet isn't in a functional
    * state, instead of pretending like everything's fine.
* [action required] The structure of the kubelet dropin in the kubeadm deb package has changed significantly. ([#64780](https://github.com/kubernetes/kubernetes/pull/64780), [@luxas](https://github.com/luxas))
    * Instead of hard-coding the parameters for the kubelet in the dropin, a structured configuration file
    * for the kubelet is used, and is expected to be present in `/var/lib/kubelet/config.yaml`.
    * For runtime-detected, instance-specific configuration values, a environment file with
    * dynamically-generated flags at `kubeadm init` or `kubeadm join` run time is used.
    * Finally, if the user wants to override something specific for the kubelet that can't be done via
    * the kubeadm Configuration file (which is preferred), they might add flags to the
    * `KUBELET_EXTRA_ARGS` environment variable in either `/etc/default/kubelet`
    * or `/etc/sysconfig/kubelet`, depending on the system you're running on.
* [action required] The `--node-name` flag for kubeadm now dictates the Node API object name the ([#64706](https://github.com/kubernetes/kubernetes/pull/64706), [@liztio](https://github.com/liztio))
    * kubelet uses for registration, in all cases but where you might use an in-tree cloud provider.
    * If you're not using an in-tree cloud provider, `--node-name` will set the Node API object name.
    * If you're using an in-tree cloud provider, you MUST make `--node-name` match the name the
    * in-tree cloud provider decides to use.
* [action required] kubeadm: The Token-related fields in the `MasterConfiguration` object have now been refactored. Instead of the top-level `.Token`, `.TokenTTL`, `.TokenUsages`, `.TokenGroups` fields, there is now a `BootstrapTokens` slice of `BootstrapToken` objects that support the same features under the `.Token`, `.TTL`, `.Usages`, `.Groups` fields. ([#64408](https://github.com/kubernetes/kubernetes/pull/64408), [@luxas](https://github.com/luxas))

### Other notable changes

* Add Vertical Pod Autoscaler to autoscaling/v2beta1 ([#63797](https://github.com/kubernetes/kubernetes/pull/63797), [@kgrygiel](https://github.com/kgrygiel))
* kubeadm: only run kube-proxy on architecture consistent nodes ([#64696](https://github.com/kubernetes/kubernetes/pull/64696), [@dixudx](https://github.com/dixudx))
* kubeadm: Add a new `kubeadm upgrade node config` command ([#64624](https://github.com/kubernetes/kubernetes/pull/64624), [@luxas](https://github.com/luxas))
* Orphan delete is now supported for custom resources ([#63386](https://github.com/kubernetes/kubernetes/pull/63386), [@roycaihw](https://github.com/roycaihw))
* Update version of Istio addon from 0.6.0 to 0.8.0. ([#64537](https://github.com/kubernetes/kubernetes/pull/64537), [@ostromart](https://github.com/ostromart))
    * See https://istio.io/about/notes/0.8.html for full Isto release notes.
* Provides API support for external CSI storage drivers to support block volumes. ([#64723](https://github.com/kubernetes/kubernetes/pull/64723), [@vladimirvivien](https://github.com/vladimirvivien))
* kubectl will list all allowed print formats when an invalid format is passed. ([#64371](https://github.com/kubernetes/kubernetes/pull/64371), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Use IONice to reduce IO priority of du and find ([#64800](https://github.com/kubernetes/kubernetes/pull/64800), [@dashpole](https://github.com/dashpole))
    * cAdvisor ContainerReference no longer contains Labels. Use ContainerSpec instead.
    * Fix a bug where cadvisor failed to discover a sub-cgroup that was created soon after the parent cgroup.
* Kubelet will set extended resource capacity to zero after it restarts. If the extended resource is exported by a device plugin, its capacity will change to a valid value after the device plugin re-connects with the Kubelet. If the extended resource is exported by an external component through direct node status capacity patching, the component should repatch the field after kubelet becomes ready again. During the time gap, pods previously assigned with such resources may fail kubelet admission but their controller should create new pods in response to such failures. ([#64784](https://github.com/kubernetes/kubernetes/pull/64784), [@jiayingz](https://github.com/jiayingz))
* Introduce ContainersReady condition in Pod Status ([#64646](https://github.com/kubernetes/kubernetes/pull/64646), [@freehan](https://github.com/freehan))
* The Sysctls experimental feature has been promoted to beta (enabled by default via the `Sysctls` feature flag). PodSecurityPolicy and Pod objects now have fields for specifying and controlling sysctls. Alpha sysctl annotations will be ignored by 1.11+ kubelets. All alpha sysctl annotations in existing deployments must be converted to API fields to be effective. ([#63717](https://github.com/kubernetes/kubernetes/pull/63717), [@ingvagabund](https://github.com/ingvagabund))
* kubeadm now configures the etcd liveness probe correctly when etcd is listening on all interfaces ([#64670](https://github.com/kubernetes/kubernetes/pull/64670), [@stealthybox](https://github.com/stealthybox))
* Fix regression in `v1.JobSpec.backoffLimit` that caused failed Jobs to be restarted indefinitely. ([#63650](https://github.com/kubernetes/kubernetes/pull/63650), [@soltysh](https://github.com/soltysh))
* GCE: Update cloud provider to use TPU v1 API ([#64727](https://github.com/kubernetes/kubernetes/pull/64727), [@yguo0905](https://github.com/yguo0905))
* Kubelet: Add security context for Windows containers ([#64009](https://github.com/kubernetes/kubernetes/pull/64009), [@feiskyer](https://github.com/feiskyer))
* Volume topology aware dynamic provisioning ([#63193](https://github.com/kubernetes/kubernetes/pull/63193), [@lichuqiang](https://github.com/lichuqiang))
* CoreDNS deployment configuration now uses k8s.gcr.io imageRepository ([#64775](https://github.com/kubernetes/kubernetes/pull/64775), [@rajansandeep](https://github.com/rajansandeep))
* Updated Container Storage Interface specification version to v0.3.0 ([#64719](https://github.com/kubernetes/kubernetes/pull/64719), [@davidz627](https://github.com/davidz627))
* Kubeadm: Make CoreDNS run in read-only mode and drop all unneeded privileges  ([#64473](https://github.com/kubernetes/kubernetes/pull/64473), [@nberlee](https://github.com/nberlee))
* Add a volume projection that is able to project service account tokens. ([#62005](https://github.com/kubernetes/kubernetes/pull/62005), [@mikedanese](https://github.com/mikedanese))
* Fix kubectl auth can-i exit code. It will return 1 if the user is not allowed and 0 if it's allowed. ([#59579](https://github.com/kubernetes/kubernetes/pull/59579), [@fbac](https://github.com/fbac))
* apply global flag "context" for kubectl config view --minify ([#64608](https://github.com/kubernetes/kubernetes/pull/64608), [@dixudx](https://github.com/dixudx))
* Fix kube-controller-manager panic while provisioning Azure security group rules ([#64739](https://github.com/kubernetes/kubernetes/pull/64739), [@feiskyer](https://github.com/feiskyer))
* API change for volume topology aware dynamic provisioning ([#63233](https://github.com/kubernetes/kubernetes/pull/63233), [@lichuqiang](https://github.com/lichuqiang))
* Add azuredisk PV size grow feature ([#64386](https://github.com/kubernetes/kubernetes/pull/64386), [@andyzhangx](https://github.com/andyzhangx))
* Modify e2e tests to use priorityClass beta version & switch priorityClass feature to beta ([#63724](https://github.com/kubernetes/kubernetes/pull/63724), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Adding CSI driver registration code. ([#64560](https://github.com/kubernetes/kubernetes/pull/64560), [@sbezverk](https://github.com/sbezverk))
* fixes a potential deadlock in the garbage collection controller ([#64235](https://github.com/kubernetes/kubernetes/pull/64235), [@liggitt](https://github.com/liggitt))
* Fixes issue for readOnly subpath mounts for SELinux systems and when the volume mountPath already existed in the container image. ([#64351](https://github.com/kubernetes/kubernetes/pull/64351), [@msau42](https://github.com/msau42))
* Add log and fs stats for Windows containers ([#62266](https://github.com/kubernetes/kubernetes/pull/62266), [@feiskyer](https://github.com/feiskyer))
* client-go: credential exec plugins have been promoted to beta ([#64482](https://github.com/kubernetes/kubernetes/pull/64482), [@ericchiang](https://github.com/ericchiang))
* Revert [#64364](https://github.com/kubernetes/kubernetes/pull/64364) to resurrect rescheduler. For More info see[#64725] https://github.com/kubernetes/kubernetes/issues/64725 and ([#64592](https://github.com/kubernetes/kubernetes/pull/64592), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Add RequestedToCapacityRatioPriority priority function. Function is parametrized with set of points mapping node utilization (0-100) to score (0-10). ([#63929](https://github.com/kubernetes/kubernetes/pull/63929), [@losipiuk](https://github.com/losipiuk))
    * Function is linear between points. Resource utilization is defined as one minus ratio of total amount of resource requested by pods on node and node's capacity (scaled to 100).
    * Final utilization used for computation is arithmetic mean of cpu utilization and memory utilization.
    * Function is disabled by default and can be enabled via scheduler policy config file.
    * If no parametrization is specified in config file it defaults to one which gives score 10 to utilization 0 and score 0 to utilization 100.
* `kubeadm init` detects if systemd-resolved is running and configures the kubelet to use a working resolv.conf. ([#64665](https://github.com/kubernetes/kubernetes/pull/64665), [@stealthybox](https://github.com/stealthybox))
* fix data loss issue if using existing azure disk with partitions in disk mount  ([#63270](https://github.com/kubernetes/kubernetes/pull/63270), [@andyzhangx](https://github.com/andyzhangx))
* Meta data of CustomResources is now pruned and schema checked during deserialization of requests and when read from etcd. In the former case, invalid meta data is rejected, in the later it is dropped from the CustomResource objects. ([#64267](https://github.com/kubernetes/kubernetes/pull/64267), [@sttts](https://github.com/sttts))
* Add Alpha support for dynamic volume limits based on node type ([#64154](https://github.com/kubernetes/kubernetes/pull/64154), [@gnufied](https://github.com/gnufied))
* Fixed CSI gRPC connection leak during volume operations. ([#64519](https://github.com/kubernetes/kubernetes/pull/64519), [@vladimirvivien](https://github.com/vladimirvivien))
* in-tree support for openstack credentials is now deprecated. please use the "client-keystone-auth" from the cloud-provider-openstack repository. details on how to use this new capability is documented here - https://github.com/kubernetes/cloud-provider-openstack/blob/master/docs/using-client-keystone-auth.md ([#64346](https://github.com/kubernetes/kubernetes/pull/64346), [@dims](https://github.com/dims))
* `ScheduleDaemonSetPods` is an alpha feature (since v1.11) that causes DaemonSet Pods ([#63223](https://github.com/kubernetes/kubernetes/pull/63223), [@k82cn](https://github.com/k82cn))
    * to be scheduler by default scheduler, instead of Daemonset controller. When it is enabled,
    * the `NodeAffinity` term (instead of `.spec.nodeName`) is added to the DaemonSet Pods;
    * this enables the default scheduler to bind the Pod to the target host. If node affinity
    * of DaemonSet Pod already exists, it will be replaced.
    * DaemonSet controller will only perform these operations when creating DaemonSet Pods;
    * and those operations will only modify the Pods of DaemonSet, no changes are made to the
    * `.spec.template` of DaemonSet.
* fix formatAndMount func issue on Windows ([#63248](https://github.com/kubernetes/kubernetes/pull/63248), [@andyzhangx](https://github.com/andyzhangx))
* AWS EBS, Azure Disk, GCE PD and Ceph RBD volume plugins support dynamic provisioning of raw block volumes. ([#64447](https://github.com/kubernetes/kubernetes/pull/64447), [@jsafrane](https://github.com/jsafrane))
* kubeadm upgrade apply can now ignore version errors with --force ([#64570](https://github.com/kubernetes/kubernetes/pull/64570), [@liztio](https://github.com/liztio))
* Adds feature gate for plugin watcher ([#64605](https://github.com/kubernetes/kubernetes/pull/64605), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Kubelet now proxies container streaming between apiserver and container runtime. The connection between kubelet and apiserver is authenticated. Container runtime should change streaming server to serve on localhost, to make the connection between kubelet and container runtime local. ([#64006](https://github.com/kubernetes/kubernetes/pull/64006), [@Random-Liu](https://github.com/Random-Liu))
    * In this way, the whole container streaming connection is secure. To switch back to the old behavior, set `--redirect-container-streaming=true` flag.
* TokenRequests now are required to have an expiration duration between 10 minutes and 2^32 seconds. ([#63999](https://github.com/kubernetes/kubernetes/pull/63999), [@mikedanese](https://github.com/mikedanese))
* Expose `/debug/flags/v` to allow dynamically set glog logging level, if want to change glog level to 3, you only have to send a PUT request with like `curl -X PUT http://127.0.0.1:8080/debug/flags/v -d "3"`. ([#63777](https://github.com/kubernetes/kubernetes/pull/63777), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* New conformance test added for Watch. ([#61424](https://github.com/kubernetes/kubernetes/pull/61424), [@jennybuckley](https://github.com/jennybuckley))
* The GitRepo volume type is deprecated. To provision a container with a git repo, mount an EmptyDir into an InitContainer that clones the repo using git, then mount the EmptyDir into the Pod's container. ([#63445](https://github.com/kubernetes/kubernetes/pull/63445), [@ericchiang](https://github.com/ericchiang))
* kubeadm now preserves previous manifests after upgrades ([#64337](https://github.com/kubernetes/kubernetes/pull/64337), [@liztio](https://github.com/liztio))
* Implement kubelet side online file system resizing ([#62460](https://github.com/kubernetes/kubernetes/pull/62460), [@mlmhl](https://github.com/mlmhl))



# v1.11.0-beta.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0-beta.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes.tar.gz) | `3209303a10ca8dd311c500ee858b9151b43c1bb5c2b3a9fb9281722e021d6871`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-src.tar.gz) | `c2e4d3b1beb4cd0b2a775394a30da2c2949d380e57f729dc48c541069c103326`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-darwin-386.tar.gz) | `cbded4d58b3d2cbeb2e43c48c9dd359834c9c9aa376751a7f8960be45601fb40`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-darwin-amd64.tar.gz) | `ceccd21fda90b96865801053f1784d4062d69b11e2e911483223860dfe6c3a17`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-linux-386.tar.gz) | `75c9794a7f43f891aa839b2571fa44ffced25197578adc31b4c3cb28d7fbf158`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-linux-amd64.tar.gz) | `184905f6b8b856306483d811d015cf0b28c0703ceb372594622732da2a07989f`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-linux-arm.tar.gz) | `2d985829499588d32483d7c6a36b3b0f2b6d4031eda31c65b066b77bc51bae66`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-linux-arm64.tar.gz) | `268556ede751058162a42d0156f27e42e37b23d60b2485e350cffe6e1b376fa4`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-linux-ppc64le.tar.gz) | `8859bd7a37bf5a659eb17e47d2c54d228950b2ef48243c93f11799c455789983`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-linux-s390x.tar.gz) | `90bbe2fc45ae722a05270820336b9178baaab198401bb6888e817afe6a1a304e`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-windows-386.tar.gz) | `948b01f555abfc30990345004d5ce679d4b9d0a32d699a50b6d8309040b2b2f2`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-client-windows-amd64.tar.gz) | `091e9d4e7fa611cf06d2907d159e0cc36ae8602403ad0819d62df4ddbaba6095`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-server-linux-amd64.tar.gz) | `727a5e8241035d631d90f3d119a27384abe93cde14c242c4d2d1cf948f84a650`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-server-linux-arm.tar.gz) | `6eb7479348e9480d9d1ee31dc991297b93e076dd21b567c595f82d45b66ef949`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-server-linux-arm64.tar.gz) | `9eab5ccdfba2803a743ed12b4323ad0e8e0215779edf5752224103b6667a35c1`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-server-linux-ppc64le.tar.gz) | `d86b07ee28ed3d2c0668a2737fff4b3d025d4cd7b6f1aadc85f8f13b4c12e578`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-server-linux-s390x.tar.gz) | `c2d19acb88684a52a74f469ab26874ab224023f29290865e08c86338d30dd598`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-node-linux-amd64.tar.gz) | `2957bf3e9dc9cd9570597434909e5ef03e996f8443c02f9d95fa6de2cd17126f`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-node-linux-arm.tar.gz) | `5995b8b9628fca9eaa92c283cfb4199ab353efa8953b980eec994f49ac3a0ebd`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-node-linux-arm64.tar.gz) | `996691b3b894ec9769be1ee45c5053ff1560e3ef161de8f8b9ac067c0d3559d3`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-node-linux-ppc64le.tar.gz) | `8bb7fe72ec704afa5ad96356787972144b0f7923fc68678894424f1f62da7041`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-node-linux-s390x.tar.gz) | `4c1f0314ad60537c8a7866b0cabdece21284ee91ae692d1999b3d5273ee7cbaf`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-beta.1/kubernetes-node-windows-amd64.tar.gz) | `158832f41cd452f93482cc8a8f1dd69cc243eb63ce3581e7f2eab2de323f6202`

## Changelog since v1.11.0-alpha.2

### Action Required

* [action required] `.NodeName` and `.CRISocket` in the `MasterConfiguration` and `NodeConfiguration` v1alpha1 API objects are now `.NodeRegistration.Name` and `.NodeRegistration.CRISocket` respectively in the v1alpha2 API. The `.NoTaintMaster` field has been removed in the v1alpha2 API. ([#64210](https://github.com/kubernetes/kubernetes/pull/64210), [@luxas](https://github.com/luxas))
* (ACTION REQUIRED) PersisntVolumeLabel admission controller is now disabled by default. If you depend on this feature (AWS/GCE) then ensure it is added to the `--enable-admission-plugins` flag on the kube-apiserver. ([#64326](https://github.com/kubernetes/kubernetes/pull/64326), [@andrewsykim](https://github.com/andrewsykim))
* [action required] kubeadm: The `:Etcd` struct has been refactored in the v1alpha2 API. All the options now reside under either `.Etcd.Local` or `.Etcd.External`. Automatic conversions from the v1alpha1 API are supported. ([#64066](https://github.com/kubernetes/kubernetes/pull/64066), [@luxas](https://github.com/luxas))
* [action required] kubeadm: kubelets in kubeadm clusters now disable the readonly port (10255). If you're relying on unauthenticated access to the readonly port, please switch to using the secure port (10250). Instead, you can now use ServiceAccount tokens when talking to the secure port, which will make it easier to get access to e.g. the `/metrics` endpoint of the kubelet securely. ([#64187](https://github.com/kubernetes/kubernetes/pull/64187), [@luxas](https://github.com/luxas))
* [action required] kubeadm: Support for `.AuthorizationModes` in the kubeadm v1alpha2 API has been removed. Instead, you can use the `.APIServerExtraArgs` and `.APIServerExtraVolumes` fields to achieve the same effect. Files using the v1alpha1 API and setting this field will be automatically upgraded to this v1alpha2 API and the information will be preserved. ([#64068](https://github.com/kubernetes/kubernetes/pull/64068), [@luxas](https://github.com/luxas))
* [action required] The formerly publicly-available cAdvisor web UI that the kubelet ran on port 4194 by default is now turned off by default. The flag configuring what port to run this UI on `--cadvisor-port` was deprecated in v1.10. Now the default is `--cadvisor-port=0`, in other words, to not run the web server. The recommended way to run cAdvisor if you still need it, is via a DaemonSet. The `--cadvisor-port` will be removed in v1.12 ([#63881](https://github.com/kubernetes/kubernetes/pull/63881), [@luxas](https://github.com/luxas))
* [action required] kubeadm: The `.ImagePullPolicy` field has been removed in the v1alpha2 API version. Instead it's set statically to `IfNotPresent` for all required images. If you want to always pull the latest images before cluster init (like what `Always` would do), run `kubeadm config images pull` before each `kubeadm init`. If you don't want the kubelet to pull any images at `kubeadm init` time, as you for instance don't have an internet connection, you can also run `kubeadm config images pull` before `kubeadm init` or side-load the images some other way (e.g. `docker load -i image.tar`). Having the images locally cached will result in no pull at runtime, which makes it possible to run without any internet connection. ([#64096](https://github.com/kubernetes/kubernetes/pull/64096), [@luxas](https://github.com/luxas))
* [action required] In the new v1alpha2 kubeadm Configuration API, the `.CloudProvider` and `.PrivilegedPods` fields don't exist anymore. ([#63866](https://github.com/kubernetes/kubernetes/pull/63866), [@luxas](https://github.com/luxas))
    * Instead, you should use the out-of-tree cloud provider implementations which are beta in v1.11.
    * If you have to use the legacy in-tree cloud providers, you can rearrange your config like the example below. In case you need the `cloud-config` file (located in `{cloud-config-path}`), you can mount it into the API Server and controller-manager containers using ExtraVolumes like the example below.
    * If you need to use the `.PrivilegedPods` functionality, you can still edit the manifests in `/etc/kubernetes/manifests/`, and set `.SecurityContext.Privileged=true` for the apiserver and controller manager.
```
kind: MasterConfiguration
apiVersion: kubeadm.k8s.io/v1alpha2
apiServerExtraArgs:
  cloud-provider: "{cloud}"
  cloud-config: "{cloud-config-path}"
apiServerExtraVolumes:
- name: cloud
  hostPath: "{cloud-config-path}"
  mountPath: "{cloud-config-path}"
controllerManagerExtraArgs:
  cloud-provider: "{cloud}"
  cloud-config: "{cloud-config-path}"
controllerManagerExtraVolumes:
- name: cloud
  hostPath: "{cloud-config-path}"
  mountPath: "{cloud-config-path}"
```
* [action required] kubeadm now uses an upgraded API version for the configuration file, `kubeadm.k8s.io/v1alpha2`. kubeadm in v1.11 will still be able to read `v1alpha1` configuration, and will automatically convert the configuration to `v1alpha2` internally and when storing the configuration in the ConfigMap in the cluster. ([#63788](https://github.com/kubernetes/kubernetes/pull/63788), [@luxas](https://github.com/luxas))
* The annotation `service.alpha.kubernetes.io/tolerate-unready-endpoints` is deprecated.  Users should use Service.spec.publishNotReadyAddresses instead. ([#63742](https://github.com/kubernetes/kubernetes/pull/63742), [@thockin](https://github.com/thockin))
* avoid duplicate status in audit events ([#62695](https://github.com/kubernetes/kubernetes/pull/62695), [@CaoShuFeng](https://github.com/CaoShuFeng))

### Other notable changes

* Remove rescheduler from master. ([#64364](https://github.com/kubernetes/kubernetes/pull/64364), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Declare IPVS-based kube-proxy GA ([#58442](https://github.com/kubernetes/kubernetes/pull/58442), [@m1093782566](https://github.com/m1093782566))
* kubeadm: conditionally set the kubelet cgroup driver for Docker ([#64347](https://github.com/kubernetes/kubernetes/pull/64347), [@neolit123](https://github.com/neolit123))
* kubectl built for darwin from darwin now enables cgo to use the system-native C libraries for DNS resolution. Cross-compiled kubectl (e.g. from an official kubernetes release) still uses the go-native netgo DNS implementation.  ([#64219](https://github.com/kubernetes/kubernetes/pull/64219), [@ixdy](https://github.com/ixdy))
* AWS EBS volumes can be now used as ReadOnly in pods. ([#64403](https://github.com/kubernetes/kubernetes/pull/64403), [@jsafrane](https://github.com/jsafrane))
* Exec authenticator plugin supports TLS client certificates. ([#61803](https://github.com/kubernetes/kubernetes/pull/61803), [@awly](https://github.com/awly))
* Use Patch instead of Put to sync pod status ([#62306](https://github.com/kubernetes/kubernetes/pull/62306), [@freehan](https://github.com/freehan))
* kubectl apply --prune supports CronJob resource.  ([#62991](https://github.com/kubernetes/kubernetes/pull/62991), [@tomoe](https://github.com/tomoe))
* Label ExternalEtcdClientCertificates can be used for ignoring all preflight check issues related to client certificate files for external etcd. ([#64269](https://github.com/kubernetes/kubernetes/pull/64269), [@kad](https://github.com/kad))
* Provide a meaningful error message in openstack cloud provider when no valid IP address can be found for a node ([#64318](https://github.com/kubernetes/kubernetes/pull/64318), [@gonzolino](https://github.com/gonzolino))
* kubeadm: Add a 'kubeadm config migrate' command to convert old API types to their newer counterparts in the new, supported API types. This is just a client-side tool, it just executes locally without requiring a cluster to be running. You can think about this as an Unix pipe that upgrades config files. ([#64232](https://github.com/kubernetes/kubernetes/pull/64232), [@luxas](https://github.com/luxas))
* The --dry-run flag has been enabled for kubectl auth reconcile ([#64458](https://github.com/kubernetes/kubernetes/pull/64458), [@mrogers950](https://github.com/mrogers950))
* Add probe based mechanism for kubelet plugin discovery ([#63328](https://github.com/kubernetes/kubernetes/pull/63328), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Add Establishing Controller on CRDs to avoid race between Established condition and CRs actually served. In HA setups, the Established condition is delayed by 5 seconds. ([#63068](https://github.com/kubernetes/kubernetes/pull/63068), [@xmudrii](https://github.com/xmudrii))
* CoreDNS is now v1.1.3 ([#64258](https://github.com/kubernetes/kubernetes/pull/64258), [@rajansandeep](https://github.com/rajansandeep))
* kubeadm will pull required images during preflight checks if it cannot find them on the system ([#64105](https://github.com/kubernetes/kubernetes/pull/64105), [@chuckha](https://github.com/chuckha))
* kubeadm: rename the addon parameter `kube-dns` to `coredns` for `kubeadm alpha phases addons` as CoreDNS is now the default DNS server in 1.11. ([#64274](https://github.com/kubernetes/kubernetes/pull/64274), [@neolit123](https://github.com/neolit123))
* kubeadm: when starting the API server use the arguments --enable-admission-plugins and --disable-admission-plugins instead of the deprecated --admission-control. ([#64165](https://github.com/kubernetes/kubernetes/pull/64165), [@neolit123](https://github.com/neolit123))
* Add spec.additionalPrinterColumns to CRDs to define server side printing columns. ([#60991](https://github.com/kubernetes/kubernetes/pull/60991), [@sttts](https://github.com/sttts))
* fix azure file size grow issue ([#64383](https://github.com/kubernetes/kubernetes/pull/64383), [@andyzhangx](https://github.com/andyzhangx))
* Fix issue of colliding nodePorts when the cluster has services with externalTrafficPolicy=Local ([#64349](https://github.com/kubernetes/kubernetes/pull/64349), [@nicksardo](https://github.com/nicksardo))
* fixes a panic applying json patches containing out of bounds operations ([#64355](https://github.com/kubernetes/kubernetes/pull/64355), [@liggitt](https://github.com/liggitt))
* Fail fast if cgroups-per-qos is set on Windows ([#62984](https://github.com/kubernetes/kubernetes/pull/62984), [@feiskyer](https://github.com/feiskyer))
* Move Volume expansion to Beta ([#64288](https://github.com/kubernetes/kubernetes/pull/64288), [@gnufied](https://github.com/gnufied))
* kubectl delete does not use reapers for removing objects anymore, but relies on server-side GC entirely ([#63979](https://github.com/kubernetes/kubernetes/pull/63979), [@soltysh](https://github.com/soltysh))
* Basic plumbing for volume topology aware dynamic provisioning ([#63232](https://github.com/kubernetes/kubernetes/pull/63232), [@lichuqiang](https://github.com/lichuqiang))
* API server properly parses propagationPolicy as a query parameter sent with a delete request ([#63414](https://github.com/kubernetes/kubernetes/pull/63414), [@roycaihw](https://github.com/roycaihw))
* Property `serverAddressByClientCIDRs` in `metav1.APIGroup` (discovery API) now become optional instead of required ([#61963](https://github.com/kubernetes/kubernetes/pull/61963), [@roycaihw](https://github.com/roycaihw))
* The dynamic Kubelet config feature is now beta, and the DynamicKubeletConfig feature gate is on by default. In order to use dynamic Kubelet config, ensure that the Kubelet's --dynamic-config-dir option is set.  ([#64275](https://github.com/kubernetes/kubernetes/pull/64275), [@mtaufen](https://github.com/mtaufen))
* Add reason message logs for non-exist Azure resources ([#64248](https://github.com/kubernetes/kubernetes/pull/64248), [@feiskyer](https://github.com/feiskyer))
* Fix SessionAffinity not updated issue for Azure load balancer ([#64180](https://github.com/kubernetes/kubernetes/pull/64180), [@feiskyer](https://github.com/feiskyer))
* The kube-apiserver openapi doc now includes extensions identifying APIService and CustomResourceDefinition kinds ([#64174](https://github.com/kubernetes/kubernetes/pull/64174), [@liggitt](https://github.com/liggitt))
* apiservices/status and certificatesigningrequests/status now support GET and PATCH ([#64063](https://github.com/kubernetes/kubernetes/pull/64063), [@roycaihw](https://github.com/roycaihw))
* kubectl: This client version requires the `apps/v1` APIs, so it will not work against a cluster version older than v1.9.0. Note that kubectl only guarantees compatibility with clusters that are +/-1 minor version away. ([#61419](https://github.com/kubernetes/kubernetes/pull/61419), [@enisoc](https://github.com/enisoc))
* Correct the way we reset containers and pods in kubeadm via crictl ([#63862](https://github.com/kubernetes/kubernetes/pull/63862), [@runcom](https://github.com/runcom))
* Allow env from resource with keys & updated tests ([#60636](https://github.com/kubernetes/kubernetes/pull/60636), [@PhilipGough](https://github.com/PhilipGough))
* The kubelet certificate rotation feature can now be enabled via the `.RotateCertificates` field in the kubelet's config file. The `--rotate-certificates` flag is now deprecated, and will be removed in a future release. ([#63912](https://github.com/kubernetes/kubernetes/pull/63912), [@luxas](https://github.com/luxas))
* Use DeleteOptions.PropagationPolicy instead of OrphanDependents in kubectl  ([#59851](https://github.com/kubernetes/kubernetes/pull/59851), [@nilebox](https://github.com/nilebox))
* add block device support for azure disk ([#63841](https://github.com/kubernetes/kubernetes/pull/63841), [@andyzhangx](https://github.com/andyzhangx))
* Fix incorrectly propagated ResourceVersion in ListRequests returning 0 items. ([#64150](https://github.com/kubernetes/kubernetes/pull/64150), [@wojtek-t](https://github.com/wojtek-t))
* Changes ext3/ext4 volume creation to not reserve any portion of the volume for the root user. ([#64102](https://github.com/kubernetes/kubernetes/pull/64102), [@atombender](https://github.com/atombender))
* Add CRD Versioning with NOP converter ([#63830](https://github.com/kubernetes/kubernetes/pull/63830), [@mbohlool](https://github.com/mbohlool))
* adds a kubectl wait command ([#64034](https://github.com/kubernetes/kubernetes/pull/64034), [@deads2k](https://github.com/deads2k))
* "kubeadm init" now writes a structured and versioned kubelet ComponentConfiguration file to `/var/lib/kubelet/config.yaml` and an environment file with runtime flags (you can source this file in the systemd kubelet dropin) to `/var/lib/kubelet/kubeadm-flags.env`. ([#63887](https://github.com/kubernetes/kubernetes/pull/63887), [@luxas](https://github.com/luxas))
* `kubectl auth reconcile` only works with rbac.v1 ([#63967](https://github.com/kubernetes/kubernetes/pull/63967), [@deads2k](https://github.com/deads2k))
* The dynamic Kubelet config feature will now update config in the event of a ConfigMap mutation, which reduces the chance for silent config skew. Only name, namespace, and kubeletConfigKey may now be set in Node.Spec.ConfigSource.ConfigMap. The least disruptive pattern for config management is still to create a new ConfigMap and incrementally roll out a new Node.Spec.ConfigSource. ([#63221](https://github.com/kubernetes/kubernetes/pull/63221), [@mtaufen](https://github.com/mtaufen))
* Graduate CRI container log rotation to beta, and enable it by default. ([#64046](https://github.com/kubernetes/kubernetes/pull/64046), [@yujuhong](https://github.com/yujuhong))
* APIServices with kube-like versions (e.g. v1, v2beta1, etc.) will be sorted appropriately within each group.  ([#64004](https://github.com/kubernetes/kubernetes/pull/64004), [@mbohlool](https://github.com/mbohlool))
* kubectl and client-go now detects duplicated name for user, cluster and context when loading kubeconfig and reports error  ([#60464](https://github.com/kubernetes/kubernetes/pull/60464), [@roycaihw](https://github.com/roycaihw))
* event object references with apiversion will now report an apiversion. ([#63913](https://github.com/kubernetes/kubernetes/pull/63913), [@deads2k](https://github.com/deads2k))
* Subresources for custom resources is now beta and enabled by default. With this, updates to the `/status` subresource will disallow updates to all fields other than `.status` (not just `.spec` and `.metadata` as before). Also, `required` can be used at the root of the CRD OpenAPI validation schema when the `/status` subresource is enabled. ([#63598](https://github.com/kubernetes/kubernetes/pull/63598), [@nikhita](https://github.com/nikhita))
* increase grpc client default response size ([#63977](https://github.com/kubernetes/kubernetes/pull/63977), [@runcom](https://github.com/runcom))
* HTTP transport now uses `context.Context` to cancel dial operations. k8s.io/client-go/transport/Config struct has been updated to accept a function with a `context.Context` parameter. This is a breaking change if you use this field in your code. ([#60012](https://github.com/kubernetes/kubernetes/pull/60012), [@ash2k](https://github.com/ash2k))
* Adds a mechanism in vSphere Cloud Provider to get credentials from Kubernetes secrets ([#63902](https://github.com/kubernetes/kubernetes/pull/63902), [@abrarshivani](https://github.com/abrarshivani))
* kubeadm: A `kubeadm config print-default` command has now been added that you can use as a starting point when writing your own kubeadm configuration files ([#63969](https://github.com/kubernetes/kubernetes/pull/63969), [@luxas](https://github.com/luxas))
* Update event-exporter to version v0.2.0  that supports old (gke_container/gce_instance) and new (k8s_container/k8s_node/k8s_pod) stackdriver resources. ([#63918](https://github.com/kubernetes/kubernetes/pull/63918), [@cezarygerard](https://github.com/cezarygerard))
* Cluster Autoscaler 1.2.2 (release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.2.2) ([#63974](https://github.com/kubernetes/kubernetes/pull/63974), [@aleksandra-malinowska](https://github.com/aleksandra-malinowska))
* Update kubeadm's minimum supported kubernetes in v1.11.x to 1.10 ([#63920](https://github.com/kubernetes/kubernetes/pull/63920), [@dixudx](https://github.com/dixudx))
* Add 'UpdateStrategyType' and 'RollingUpdateStrategy' to 'kubectl describe sts' command output. ([#63844](https://github.com/kubernetes/kubernetes/pull/63844), [@tossmilestone](https://github.com/tossmilestone))
* Remove UID mutation from request.context. ([#63957](https://github.com/kubernetes/kubernetes/pull/63957), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* kubeadm has removed `.Etcd.SelfHosting` from its configuration API. It was never used in practice. ([#63871](https://github.com/kubernetes/kubernetes/pull/63871), [@luxas](https://github.com/luxas))
* list/watch API requests with a fieldSelector that specifies `metadata.name` can now be authorized as requests for an individual named resource ([#63469](https://github.com/kubernetes/kubernetes/pull/63469), [@wojtek-t](https://github.com/wojtek-t))
* Add a way to pass extra arguments to etcd. ([#63961](https://github.com/kubernetes/kubernetes/pull/63961), [@mborsz](https://github.com/mborsz))
* minor fix for VolumeZoneChecker predicate, storageclass can be in annotation and spec. ([#63749](https://github.com/kubernetes/kubernetes/pull/63749), [@wenlxie](https://github.com/wenlxie))
* vSphere Cloud Provider: add SAML token authentication support ([#63824](https://github.com/kubernetes/kubernetes/pull/63824), [@dougm](https://github.com/dougm))
* adds the `kubeadm upgrade diff` command to show how static pod manifests will be changed by an upgrade. ([#63930](https://github.com/kubernetes/kubernetes/pull/63930), [@liztio](https://github.com/liztio))
* Fix memory cgroup notifications, and reduce associated log spam. ([#63220](https://github.com/kubernetes/kubernetes/pull/63220), [@dashpole](https://github.com/dashpole))
* Adds a `kubeadm config images pull` command to pull container images used by kubeadm. ([#63833](https://github.com/kubernetes/kubernetes/pull/63833), [@chuckha](https://github.com/chuckha))
* Restores the pre-1.10 behavior of the openstack cloud provider which uses the instance name as the Kubernetes Node name. This requires instances be named with RFC-1123 compatible names. ([#63903](https://github.com/kubernetes/kubernetes/pull/63903), [@liggitt](https://github.com/liggitt))
* Added support for NFS relations on kubernetes-worker charm. ([#63817](https://github.com/kubernetes/kubernetes/pull/63817), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Stop using InfluxDB as default cluster monitoring ([#62328](https://github.com/kubernetes/kubernetes/pull/62328), [@serathius](https://github.com/serathius))
    * InfluxDB cluster monitoring is deprecated and will be removed in v1.12
* GCE: Fix to make the built-in `kubernetes` service properly point to the master's load balancer address in clusters that use multiple master VMs. ([#63696](https://github.com/kubernetes/kubernetes/pull/63696), [@grosskur](https://github.com/grosskur))
* Kubernetes cluster on GCE have crictl installed now. Users can use it to help debug their node. The documentation of crictl can be found https://github.com/kubernetes-incubator/cri-tools/blob/master/docs/crictl.md. ([#63357](https://github.com/kubernetes/kubernetes/pull/63357), [@Random-Liu](https://github.com/Random-Liu))
* The NodeRestriction admission plugin now prevents kubelets from modifying/removing taints applied to their Node API object. ([#63167](https://github.com/kubernetes/kubernetes/pull/63167), [@liggitt](https://github.com/liggitt))
* The status of dynamic Kubelet config is now reported via Node.Status.Config, rather than the KubeletConfigOk node condition. ([#63314](https://github.com/kubernetes/kubernetes/pull/63314), [@mtaufen](https://github.com/mtaufen))
* kubeadm now checks that IPv4/IPv6 forwarding is enabled ([#63872](https://github.com/kubernetes/kubernetes/pull/63872), [@kad](https://github.com/kad))
* kubeadm will now deploy CoreDNS by default instead of KubeDNS ([#63509](https://github.com/kubernetes/kubernetes/pull/63509), [@detiber](https://github.com/detiber))
* This PR will leverage subtests on the existing table tests for the scheduler units. ([#63658](https://github.com/kubernetes/kubernetes/pull/63658), [@xchapter7x](https://github.com/xchapter7x))
    * Some refactoring of error/status messages and functions to align with new approach.
* kubeadm upgrade now supports external etcd setups again ([#63495](https://github.com/kubernetes/kubernetes/pull/63495), [@detiber](https://github.com/detiber))
* fix mount unmount failure for a Windows pod ([#63272](https://github.com/kubernetes/kubernetes/pull/63272), [@andyzhangx](https://github.com/andyzhangx))
* CRI: update documents for container logpath. The container log path has been changed from containername_attempt#.log to containername/attempt#.log  ([#62015](https://github.com/kubernetes/kubernetes/pull/62015), [@feiskyer](https://github.com/feiskyer))
* Create a new `dryRun` query parameter for mutating endpoints. If the parameter is set, then the query will be rejected, as the feature is not implemented yet. This will allow forward compatibility with future clients; otherwise, future clients talking with older apiservers might end up modifying a resource even if they include the `dryRun` query parameter. ([#63557](https://github.com/kubernetes/kubernetes/pull/63557), [@apelisse](https://github.com/apelisse))
* kubelet: fix hangs in updating Node status after network interruptions/changes between the kubelet and API server ([#63492](https://github.com/kubernetes/kubernetes/pull/63492), [@liggitt](https://github.com/liggitt))
* The `PriorityClass` API is promoted to `scheduling.k8s.io/v1beta1` ([#63100](https://github.com/kubernetes/kubernetes/pull/63100), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Services can listen on same host ports on different interfaces with --nodeport-addresses specified ([#62003](https://github.com/kubernetes/kubernetes/pull/62003), [@m1093782566](https://github.com/m1093782566))
* kubeadm will no longer generate an unused etcd CA and certificates when configured to use an external etcd cluster. ([#63806](https://github.com/kubernetes/kubernetes/pull/63806), [@detiber](https://github.com/detiber))
* corrects a race condition in bootstrapping aggregated cluster roles in new HA clusters ([#63761](https://github.com/kubernetes/kubernetes/pull/63761), [@liggitt](https://github.com/liggitt))
* Adding initial Korean translation for kubectl ([#62040](https://github.com/kubernetes/kubernetes/pull/62040), [@ianychoi](https://github.com/ianychoi))
* Report node DNS info with --node-ip flag ([#63170](https://github.com/kubernetes/kubernetes/pull/63170), [@micahhausler](https://github.com/micahhausler))
* The old dynamic client has been replaced by a new one.  The previous dynamic client will exist for one release in `client-go/deprecated-dynamic`.  Switch as soon as possible. ([#63446](https://github.com/kubernetes/kubernetes/pull/63446), [@deads2k](https://github.com/deads2k))
* CustomResourceDefinitions Status subresource now supports GET and PATCH ([#63619](https://github.com/kubernetes/kubernetes/pull/63619), [@roycaihw](https://github.com/roycaihw))
* Re-enable nodeipam controller for external clouds.  ([#63049](https://github.com/kubernetes/kubernetes/pull/63049), [@andrewsykim](https://github.com/andrewsykim))
* Removes a preflight check for kubeadm that validated custom kube-apiserver, kube-controller-manager and kube-scheduler arguments. ([#63673](https://github.com/kubernetes/kubernetes/pull/63673), [@chuckha](https://github.com/chuckha))
* Adds a list-images subcommand to kubeadm that lists required images for a kubeadm install. ([#63450](https://github.com/kubernetes/kubernetes/pull/63450), [@chuckha](https://github.com/chuckha))
* Apply pod name and namespace labels to pod cgroup in cAdvisor metrics ([#63406](https://github.com/kubernetes/kubernetes/pull/63406), [@derekwaynecarr](https://github.com/derekwaynecarr))
* try to read openstack auth config from client config and fall back to read from the environment variables if not available ([#60200](https://github.com/kubernetes/kubernetes/pull/60200), [@dixudx](https://github.com/dixudx))
* GC is now bound by QPS (it wasn't before) and so if you need more QPS to avoid ratelimiting GC, you'll have to set it. ([#63657](https://github.com/kubernetes/kubernetes/pull/63657), [@shyamjvs](https://github.com/shyamjvs))
* The Kubelet's deprecated --allow-privileged flag now defaults to true. This enables users to stop setting --allow-privileged in order to transition to PodSecurityPolicy. Previously, users had to continue setting --allow-privileged, because the default was false. ([#63442](https://github.com/kubernetes/kubernetes/pull/63442), [@mtaufen](https://github.com/mtaufen))
* You must now specify Node.Spec.ConfigSource.ConfigMap.KubeletConfigKey when using dynamic Kubelet config to tell the Kubelet which key of the ConfigMap identifies its config file. ([#59847](https://github.com/kubernetes/kubernetes/pull/59847), [@mtaufen](https://github.com/mtaufen))
* Kubernetes version command line parameter in kubeadm has been updated to drop an unnecessary redirection from ci/latest.txt to ci-cross/latest.txt. Users should know exactly where the builds are stored on Google Cloud storage buckets from now on. For example for 1.9 and 1.10, users can specify ci/latest-1.9 and ci/latest-1.10 as the CI build jobs what build images correctly updates those. The CI jobs for master update the ci-cross/latest location, so if you are looking for latest master builds, then the correct parameter to use would be ci-cross/latest. ([#63504](https://github.com/kubernetes/kubernetes/pull/63504), [@dims](https://github.com/dims))
* Search standard KubeConfig file locations when using `kubeadm token` without `--kubeconfig`. ([#62850](https://github.com/kubernetes/kubernetes/pull/62850), [@neolit123](https://github.com/neolit123))
* Include the list of security groups when failing with the errors that more then one is tagged ([#58874](https://github.com/kubernetes/kubernetes/pull/58874), [@sorenmat](https://github.com/sorenmat))
* Allow "required" to be used at the CRD OpenAPI validation schema when the /status subresource is enabled. ([#63533](https://github.com/kubernetes/kubernetes/pull/63533), [@sttts](https://github.com/sttts))
* When updating /status subresource of a custom resource, only the value at the `.status` subpath for the update is considered. ([#63385](https://github.com/kubernetes/kubernetes/pull/63385), [@CaoShuFeng](https://github.com/CaoShuFeng))
* Supported nodeSelector.matchFields (node's `metadata.node`) in scheduler. ([#62453](https://github.com/kubernetes/kubernetes/pull/62453), [@k82cn](https://github.com/k82cn))
* Do not check vmSetName when getting Azure node's IP ([#63541](https://github.com/kubernetes/kubernetes/pull/63541), [@feiskyer](https://github.com/feiskyer))
* Fix stackdriver metrics for node memory using wrong metric type ([#63535](https://github.com/kubernetes/kubernetes/pull/63535), [@serathius](https://github.com/serathius))
* [fluentd-gcp addon] Use the logging agent's node name as the metadata agent URL. ([#63353](https://github.com/kubernetes/kubernetes/pull/63353), [@bmoyles0117](https://github.com/bmoyles0117))
* `kubectl cp` supports completion. ([#60371](https://github.com/kubernetes/kubernetes/pull/60371), [@superbrothers](https://github.com/superbrothers))
* Azure VMSS: support VM names to contain the `_` character ([#63526](https://github.com/kubernetes/kubernetes/pull/63526), [@djsly](https://github.com/djsly))
* OpenStack built-in cloud provider is now deprecated. Please use the external cloud provider for OpenStack. ([#63524](https://github.com/kubernetes/kubernetes/pull/63524), [@dims](https://github.com/dims))
* the shortcuts which were moved server-side in at least 1.9 have been removed from being hardcoded in kubectl ([#63507](https://github.com/kubernetes/kubernetes/pull/63507), [@deads2k](https://github.com/deads2k))
* Fixes fake client generation for non-namespaced subresources ([#60445](https://github.com/kubernetes/kubernetes/pull/60445), [@jhorwit2](https://github.com/jhorwit2))
* `kubectl delete` with selection criteria defaults to ignoring not found errors ([#63490](https://github.com/kubernetes/kubernetes/pull/63490), [@deads2k](https://github.com/deads2k))
* Increase scheduler cache generation number monotonically in order to avoid collision and use of stale information in scheduler. ([#63264](https://github.com/kubernetes/kubernetes/pull/63264), [@bsalamat](https://github.com/bsalamat))
* Fixes issue where subpath readOnly mounts failed ([#63045](https://github.com/kubernetes/kubernetes/pull/63045), [@msau42](https://github.com/msau42))
* Update to use go1.10.2 ([#63412](https://github.com/kubernetes/kubernetes/pull/63412), [@praseodym](https://github.com/praseodym))
* `kubectl create [secret | configmap] --from-file` now works on Windows with fully-qualified paths ([#63439](https://github.com/kubernetes/kubernetes/pull/63439), [@liggitt](https://github.com/liggitt))
* kube-apiserver: the default `--endpoint-reconciler-type` is now `lease`. The `master-count` endpoint reconciler type is deprecated and will be removed in 1.13. ([#63383](https://github.com/kubernetes/kubernetes/pull/63383), [@liggitt](https://github.com/liggitt))
* owner references can be set during creation without deletion power ([#63403](https://github.com/kubernetes/kubernetes/pull/63403), [@deads2k](https://github.com/deads2k))
* Lays groundwork for OIDC distributed claims handling in the apiserver authentication token checker. ([#63213](https://github.com/kubernetes/kubernetes/pull/63213), [@filmil](https://github.com/filmil))
    * A distributed claim allows the OIDC provider to delegate a claim to a
    * separate URL.  Distributed claims are of the form as seen below, and are
    * defined in the OIDC Connect Core 1.0, section 5.6.2.
    * For details, see:
    * http://openid.net/specs/openid-connect-core-1_0.html#AggregatedDistributedClaims
* Use /usr/bin/env in all script shebangs to increase portability. ([#62657](https://github.com/kubernetes/kubernetes/pull/62657), [@matthyx](https://github.com/matthyx))



# v1.11.0-alpha.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0-alpha.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes.tar.gz) | `8f352d4f44b0c539cfb4fb72a64098c155771916cff31642b131f1eb7879da20`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-src.tar.gz) | `d2de8df039fd3bd997c992abedb0353e37691053bd927627c6438ad654055f80`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `ca70a374de0c3be4897d913f6ad22e426c6336837be6debff3cbf5f3fcf4b3ae`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `d6e0e6f286ef20a54047038b337b8a47f6cbd105b69917137c5c30c8fbee006f`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `6e73e49fa99391e1474d63a102f3cf758ef84b781bc0c0de42f1e5d1cc89132b`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `1c0c7a7aefabcda0d0407dfadd2ee7e379b395ae4ad1671535d99305e72eb2ae`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `e6310653c31114efe32db29aa06c2c1530c285cda4cccc30edf4926d0417a3a6`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `188312f25a53cf30f8375ab5727e64067ede4fba53823c3a4e2e4b768938244e`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `875f77e17c3236dde0d6e5f302c52a5193f1bf1d79d72115ae1c6de5f494b0a3`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `18502d6bd9fb483c3a858d73e2d55e32b946cbb351e09788671aca6010e39ba8`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `f0e83868dd731365b8e3f95fe33622a59d0b67d97907089c2a1c56a8eca8ebf7`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `571898fd6f612d75c9cfb248875cefbe9761155f3e8c7df48fce389606414028`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `1f36c8bb40050d4371f0d8362e8fad9d60c39c5f7f9e5569ec70d0731c9dd438`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `f503c149c1aaef2df9fea146524c4f2cb505a1946062959d1acf8bc399333437`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `660d282c18e2988744d902cb2c9f3b962b3418cbfae3644e3ea854835ca19d32`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `0682060c38c704c710cc42a887b40e26726fad9cb23368ef44236527c2a7858f`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `319337deee4e12e30da57ca484ef435f280a36792c2e2e3cd3515079b911281a`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `8d111b862d4cb3490d5ee2b97acd439e10408cba0c7f04c98a9f0470a4869e20`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `e04a30445bdabc0b895e036497fdebd102c39a53660108e45c870ae7ebc6dced`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `5fea9ce404e76e7d32c06aa2e1fbf2520531901c16a2e5f0047712d0a9422e42`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `fc6e0568f5f72790d14260ff70fe0802490a3772ed9aef2723952d706ef0fa3d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `54f97b09c5adb4657e48fda59a9f4657386b0aa4be787c188eef1ece41bd4eb8`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `72dbc9c474b15cc70e7d806cd0f78f10af1f9a7b4a11f014167f1d47277154cf`

## Changelog since v1.11.0-alpha.1

### Other notable changes

* `kubeadm upgrade plan` now accepts a version which improves the UX nicer in air-gapped environments. ([#63201](https://github.com/kubernetes/kubernetes/pull/63201), [@chuckha](https://github.com/chuckha))
* kubectl now supports --field-selector for `delete`, `label`, and `annotate` ([#60717](https://github.com/kubernetes/kubernetes/pull/60717), [@liggitt](https://github.com/liggitt))
* kube-apiserver: `--endpoint-reconciler-type` now defaults to `lease`. The `master-count` reconciler is deprecated and will be removed in 1.13. ([#58474](https://github.com/kubernetes/kubernetes/pull/58474), [@rphillips](https://github.com/rphillips))
* OpenStack cloudprovider: Fix deletion of orphaned routes ([#62729](https://github.com/kubernetes/kubernetes/pull/62729), [@databus23](https://github.com/databus23))
* Fix a bug that headless service without ports fails to have endpoint created. ([#62497](https://github.com/kubernetes/kubernetes/pull/62497), [@MrHohn](https://github.com/MrHohn))
* Fix panic for attaching AzureDisk to vmss nodes ([#63275](https://github.com/kubernetes/kubernetes/pull/63275), [@feiskyer](https://github.com/feiskyer))
* `kubectl api-resources` now supports filtering to resources supporting specific verbs, and can output fully qualified resource names suitable for combining with commands like `kubectl get` ([#63254](https://github.com/kubernetes/kubernetes/pull/63254), [@liggitt](https://github.com/liggitt))
* fix cephfs fuse mount bug when user is not admin ([#61804](https://github.com/kubernetes/kubernetes/pull/61804), [@zhangxiaoyu-zidif](https://github.com/zhangxiaoyu-zidif))
* StorageObjectInUseProtection feature is GA. ([#62870](https://github.com/kubernetes/kubernetes/pull/62870), [@pospispa](https://github.com/pospispa))
* fixed spurious "unable to find api field" errors patching custom resources ([#63146](https://github.com/kubernetes/kubernetes/pull/63146), [@liggitt](https://github.com/liggitt))
* KUBE_API_VERSIONS is no longer respected.  It was used for testing, but runtime-config is the proper flag to set. ([#63165](https://github.com/kubernetes/kubernetes/pull/63165), [@deads2k](https://github.com/deads2k))
* Added CheckNodePIDPressurePredicate to checks if a pod can be scheduled on ([#60007](https://github.com/kubernetes/kubernetes/pull/60007), [@k82cn](https://github.com/k82cn))
    * a node reporting pid pressure condition.
* Upgrade Azure Go SDK to stable version (v14.6.0) ([#63063](https://github.com/kubernetes/kubernetes/pull/63063), [@feiskyer](https://github.com/feiskyer))
* kubeadm: prompt the user for confirmation when resetting a master node ([#59115](https://github.com/kubernetes/kubernetes/pull/59115), [@alexbrand](https://github.com/alexbrand))
* add warnings on using pod-infra-container-image for remote container runtime ([#62982](https://github.com/kubernetes/kubernetes/pull/62982), [@dixudx](https://github.com/dixudx))
* Deprecate kubectl rolling-update  ([#61285](https://github.com/kubernetes/kubernetes/pull/61285), [@soltysh](https://github.com/soltysh))
* client-go developers: the new dynamic client is easier to use and the old is deprecated, you must switch. ([#62913](https://github.com/kubernetes/kubernetes/pull/62913), [@deads2k](https://github.com/deads2k))
* Fix issue where on re-registration of device plugin, `allocatable` was not getting updated. This issue makes devices invisible to the Kubelet if device plugin restarts. Only work-around, if this fix is not there, is to restart the kubelet and then start device plugin. ([#63118](https://github.com/kubernetes/kubernetes/pull/63118), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Remove METADATA_AGENT_VERSION configuration option. ([#63000](https://github.com/kubernetes/kubernetes/pull/63000), [@kawych](https://github.com/kawych))
* kubelets are no longer allowed to delete their own Node API object. Prior to 1.11, in rare circumstances related to cloudprovider node ID changes, kubelets would attempt to delete/recreate their Node object at startup. If a legacy kubelet encounters this situation, a cluster admin can remove the Node object: ([#62818](https://github.com/kubernetes/kubernetes/pull/62818), [@mikedanese](https://github.com/mikedanese))
        * `kubectl delete node/<nodeName>`
    * or grant self-deletion permission explicitly:
        * `kubectl create clusterrole self-deleting-nodes --verb=delete --resource=nodes`
        * `kubectl create clusterrolebinding self-deleting-nodes --clusterrole=self-deleting-nodes --group=system:nodes`
* kubeadm creates kube-proxy with a toleration to run on all nodes, no matter the taint. ([#62390](https://github.com/kubernetes/kubernetes/pull/62390), [@discordianfish](https://github.com/discordianfish))
* fix resultRun by resetting it to 0 on pod restart ([#62853](https://github.com/kubernetes/kubernetes/pull/62853), [@tony612](https://github.com/tony612))
* Mount additional paths required for a working CA root, for setups where /etc/ssl/certs doesn't contains certificates but just symlink. ([#59122](https://github.com/kubernetes/kubernetes/pull/59122), [@klausenbusk](https://github.com/klausenbusk))
* Introduce truncating audit backend that can be enabled for existing backend to limit the size of individual audit events and batches of events. ([#61711](https://github.com/kubernetes/kubernetes/pull/61711), [@crassirostris](https://github.com/crassirostris))
* kubeadm upgrade no longer races leading to unexpected upgrade behavior on pod restarts ([#62655](https://github.com/kubernetes/kubernetes/pull/62655), [@stealthybox](https://github.com/stealthybox))
    * kubeadm upgrade now successfully upgrades etcd and the controlplane to use TLS
    * kubeadm upgrade now supports external etcd setups
    * kubeadm upgrade can now rollback and restore etcd after an upgrade failure
* Add --ipvs-exclude-cidrs flag to kube-proxy.  ([#62083](https://github.com/kubernetes/kubernetes/pull/62083), [@rramkumar1](https://github.com/rramkumar1))
* Fix the liveness probe to use `/bin/bash -c` instead of `/bin/bash c`. ([#63033](https://github.com/kubernetes/kubernetes/pull/63033), [@bmoyles0117](https://github.com/bmoyles0117))
* Added `MatchFields` to `NodeSelectorTerm`; in 1.11, it only support `metadata.name`. ([#62002](https://github.com/kubernetes/kubernetes/pull/62002), [@k82cn](https://github.com/k82cn))
* Fix scheduler informers to receive events for all the pods in the cluster. ([#63003](https://github.com/kubernetes/kubernetes/pull/63003), [@bsalamat](https://github.com/bsalamat))
* removed unsafe double RLock in cpumanager ([#62464](https://github.com/kubernetes/kubernetes/pull/62464), [@choury](https://github.com/choury))
* Fix in vSphere Cloud Provider to handle upgrades from kubernetes version less than v1.9.4 to v1.9.4 and above. ([#62919](https://github.com/kubernetes/kubernetes/pull/62919), [@abrarshivani](https://github.com/abrarshivani))
* The `--bootstrap-kubeconfig` argument to Kubelet previously created the first bootstrap client credentials in the certificates directory as `kubelet-client.key` and `kubelet-client.crt`.  Subsequent certificates created by cert rotation were created in a combined PEM file that was atomically rotated as `kubelet-client-DATE.pem` in that directory, which meant clients relying on the `node.kubeconfig` generated by bootstrapping would never use a rotated cert.  The initial bootstrap certificate is now generated into the cert directory as a PEM file and symlinked to `kubelet-client-current.pem` so that the generated kubeconfig remains valid after rotation. ([#62152](https://github.com/kubernetes/kubernetes/pull/62152), [@smarterclayton](https://github.com/smarterclayton))
* stop kubelet to cloud provider integration potentially wedging kubelet sync loop ([#62543](https://github.com/kubernetes/kubernetes/pull/62543), [@ingvagabund](https://github.com/ingvagabund))
* Fix error where config map for Metadata Agent was not created by addon manager. ([#62909](https://github.com/kubernetes/kubernetes/pull/62909), [@kawych](https://github.com/kawych))
* Fixes the kubernetes.default.svc loopback service resolution to use a loopback configuration. ([#62649](https://github.com/kubernetes/kubernetes/pull/62649), [@liggitt](https://github.com/liggitt))
* Code generated for CRDs now passes `go vet`. ([#62412](https://github.com/kubernetes/kubernetes/pull/62412), [@bhcleek](https://github.com/bhcleek))
* fix permissions to allow statefulset scaling for admins, editors, and viewers ([#62336](https://github.com/kubernetes/kubernetes/pull/62336), [@deads2k](https://github.com/deads2k))
* Add support of standard LB to Azure vmss ([#62707](https://github.com/kubernetes/kubernetes/pull/62707), [@feiskyer](https://github.com/feiskyer))
* GCE: Fix for internal load balancer management resulting in backend services with outdated instance group links. ([#62885](https://github.com/kubernetes/kubernetes/pull/62885), [@nicksardo](https://github.com/nicksardo))
* The --experimental-qos-reserve kubelet flags is replaced by the alpha level --qos-reserved flag or QOSReserved field in the kubeletconfig and requires the QOSReserved feature gate to be enabled. ([#62509](https://github.com/kubernetes/kubernetes/pull/62509), [@sjenning](https://github.com/sjenning))
* Set pod status to "Running" if there is at least one container still reporting as "Running" status and others are "Completed". ([#62642](https://github.com/kubernetes/kubernetes/pull/62642), [@ceshihao](https://github.com/ceshihao))
* Split PodPriority and PodPreemption feature gate ([#62243](https://github.com/kubernetes/kubernetes/pull/62243), [@resouer](https://github.com/resouer))
* Add support to resize Portworx volumes. ([#62308](https://github.com/kubernetes/kubernetes/pull/62308), [@harsh-px](https://github.com/harsh-px))



# v1.11.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.11.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes.tar.gz) | `8e7f2b4c8f8fb948b4f7882038fd1bb3f2b967ee240d30d58347f40083ed199b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-src.tar.gz) | `62ab39d8fd02309c74c2a978402ef809c0fe4bb576f1366d6bb0cff26d62e2ff`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `332fd9e243c9c37e31fd26d8fa1a7ccffba770a48a9b0ffe57403f028c6ad6f4`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `1703462ad564d2d52257fd59b0c8acab595fd08b41ea73fed9f6ccb4bfa074c7`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `61073b7c5266624e0f7be323481b3111ee01511b6b96cf16468044d8a68068e3`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `9a29117fa44ffc14a7004d55f4de97ad88d94076826cfc0bf9ec73c998c78f64`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `55114364aacd4eb6d080b818c859877dd5ce46b8f1e58e1469dfa9a50ade1cf9`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `276fb16cf4aef7d1444ca754ec83365ff36184e1bc30104853f791a57934ee37`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `8a9096dd1908b8f4004249daff7ae408e390dbc728cd237bc558192744f52116`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `9297755244647b90c2d41ce9e04ee31fb158a69f011c0f4f1ec2310fa57234e7`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `449562a4d6d82b5eb60151e6ff0b301f92b92f957e3a38b741a4c0d8b3c0611f`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `ab97f150723614bcbacdf27c4ced8b45166425522a44e7de693d0e987c425f07`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `4c2db4089271366933d0b63ea7fe8f0d9eb4af06fe91d6aac1b8240e2fbd62e1`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `d5abdfe5aa28b23cf4f4f6be27db031f885f87e2defef680f2d5b92098b2d783`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `bd8a8d7c45108f4b0c2af81411c00e338e410b680abe4463f6b6d88e8adcc817`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `cb5341af600c82d391fc5ca726ff96c48e741f597360a56cc2ada0a0f9e7ec95`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `91009df3801430afde03e888f1f13a83bcb9d00b7cd4194b085684cc11657549`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `22bf846c692545e7c2655e2ebe06ffc61313d7c76e4f75716be4cec457b548ed`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `351095bb0ec177ce1ba950d366516ed6154f6ce920eac39e2a26c48203a94e11`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `947e6e9e362652db435903e9b40f14750a7ab3cc60622e78257797f6ed63b1ab`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `1a0a1d0b96c3e01bc0737245eed76ed3db970c8d80c42450072193f23a0e186b`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `6891b2e8f1f93b4f590981dccc6fd976a50a0aa5c425938fc5ca3a9c0742d16a`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.11.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `70daea86c14fcafbd46f3d1bb252db50148fb9aab3371dffc4a039791caebac5`

## Changelog since v1.10.0

### Action Required

* NONE ([#62643](https://github.com/kubernetes/kubernetes/pull/62643), [@xiangpengzhao](https://github.com/xiangpengzhao))
* ACTION REQUIRED: Alpha annotation for PersistentVolume node affinity has been removed.  Update your PersistentVolumes to use the beta PersistentVolume.nodeAffinity field before upgrading to this release ([#61816](https://github.com/kubernetes/kubernetes/pull/61816), [@wackxu](https://github.com/wackxu))
* ACTION REQUIRED: In-place node upgrades to this release from versions 1.7.14, 1.8.9, and 1.9.4 are not supported if using subpath volumes with PVCs.  Such pods should be drained from the node first. ([#61373](https://github.com/kubernetes/kubernetes/pull/61373), [@msau42](https://github.com/msau42))

### Other notable changes

* Make volume usage metrics available for Cinder ([#62668](https://github.com/kubernetes/kubernetes/pull/62668), [@zetaab](https://github.com/zetaab))
* kubectl stops rendering List as suffix kind name for CRD resources ([#62512](https://github.com/kubernetes/kubernetes/pull/62512), [@dixudx](https://github.com/dixudx))
* Removes --include-extended-apis which was deprecated back in https://github.com/kubernetes/kubernetes/pull/32894 ([#62803](https://github.com/kubernetes/kubernetes/pull/62803), [@deads2k](https://github.com/deads2k))
* Add write-config-to to scheduler ([#62515](https://github.com/kubernetes/kubernetes/pull/62515), [@resouer](https://github.com/resouer))
* Kubelets will no longer set `externalID` in their node spec. ([#61877](https://github.com/kubernetes/kubernetes/pull/61877), [@mikedanese](https://github.com/mikedanese))
* kubeadm preflight: check CRI socket path if defined, otherwise check for Docker ([#62481](https://github.com/kubernetes/kubernetes/pull/62481), [@taharah](https://github.com/taharah))
* fix network setup in hack/local-up-cluster.sh (https://github.com/kubernetes/kubernetes/pull/60431) ([#60633](https://github.com/kubernetes/kubernetes/pull/60633), [@pohly](https://github.com/pohly))
    * better error diagnostics in hack/local-up-cluster.sh output
* Add prometheus cluster monitoring addon to kube-up ([#62195](https://github.com/kubernetes/kubernetes/pull/62195), [@serathius](https://github.com/serathius))
* Fix inter-pod anti-affinity check to consider a pod a match when all the anti-affinity terms match. ([#62715](https://github.com/kubernetes/kubernetes/pull/62715), [@bsalamat](https://github.com/bsalamat))
* GCE: Bump GLBC version to 1.1.1 - fixing an issue of handling multiple certs with identical certificates ([#62751](https://github.com/kubernetes/kubernetes/pull/62751), [@nicksardo](https://github.com/nicksardo))
* fixes configuration error when upgrading kubeadm from 1.9 to 1.10+ ([#62568](https://github.com/kubernetes/kubernetes/pull/62568), [@liztio](https://github.com/liztio))
    * enforces  kubeadm  upgrading kubernetes from the same major and minor versions as the kubeadm binary.
* Allow user to scale l7 default backend deployment ([#62685](https://github.com/kubernetes/kubernetes/pull/62685), [@freehan](https://github.com/freehan))
* Pod affinity `nodeSelectorTerm.matchExpressions` may now be empty, and works as previously documented: nil or empty `matchExpressions` matches no objects in scheduler. ([#62448](https://github.com/kubernetes/kubernetes/pull/62448), [@k82cn](https://github.com/k82cn))
* Add [@andrewsykim](https://github.com/andrewsykim) as an approver for CCM related code. ([#62749](https://github.com/kubernetes/kubernetes/pull/62749), [@andrewsykim](https://github.com/andrewsykim))
* Fix an issue in inter-pod affinity predicate that cause affinity to self being processed incorrectly ([#62591](https://github.com/kubernetes/kubernetes/pull/62591), [@bsalamat](https://github.com/bsalamat))
* fix WaitForAttach failure issue for azure disk ([#62612](https://github.com/kubernetes/kubernetes/pull/62612), [@andyzhangx](https://github.com/andyzhangx))
* Update kube-dns to Version 1.14.10. Major changes: ([#62676](https://github.com/kubernetes/kubernetes/pull/62676), [@MrHohn](https://github.com/MrHohn))
    * Fix a bug in DNS resolution for externalName services
    * and PTR records that need to query from upstream nameserver.
* Update version of Istio addon from 0.5.1 to 0.6.0. ([#61911](https://github.com/kubernetes/kubernetes/pull/61911), [@ostromart](https://github.com/ostromart))
    * See https://istio.io/about/notes/0.6.html for full Isto release notes.
* Phase `kubeadm alpha phase kubelet` is added to support dynamic kubelet configuration in kubeadm. ([#57224](https://github.com/kubernetes/kubernetes/pull/57224), [@xiangpengzhao](https://github.com/xiangpengzhao))
* `kubeadm alpha phase kubeconfig user` supports groups (organizations) to be specified in client cert. ([#62627](https://github.com/kubernetes/kubernetes/pull/62627), [@xiangpengzhao](https://github.com/xiangpengzhao))
* Fix user visible files creation for windows ([#62375](https://github.com/kubernetes/kubernetes/pull/62375), [@feiskyer](https://github.com/feiskyer))
* remove deprecated initresource admission plugin ([#58784](https://github.com/kubernetes/kubernetes/pull/58784), [@wackxu](https://github.com/wackxu))
* Fix machineID getting for vmss nodes when using instance metadata ([#62611](https://github.com/kubernetes/kubernetes/pull/62611), [@feiskyer](https://github.com/feiskyer))
* Fixes issue where PersistentVolume.NodeAffinity.NodeSelectorTerms were ANDed instead of ORed. ([#62556](https://github.com/kubernetes/kubernetes/pull/62556), [@msau42](https://github.com/msau42))
* Fix potential infinite loop that can occur when NFS PVs are recycled. ([#62572](https://github.com/kubernetes/kubernetes/pull/62572), [@joelsmith](https://github.com/joelsmith))
* Fix Forward chain default reject policy for IPVS proxier ([#62007](https://github.com/kubernetes/kubernetes/pull/62007), [@m1093782566](https://github.com/m1093782566))
* The kubeadm config option `API.ControlPlaneEndpoint` has been extended to take an optional port which may differ from the apiserver's bind port. ([#62314](https://github.com/kubernetes/kubernetes/pull/62314), [@rjosephwright](https://github.com/rjosephwright))
* cluster/kube-up.sh now provisions a Kubelet config file for GCE via the metadata server. This file is installed by the corresponding GCE init scripts. ([#62183](https://github.com/kubernetes/kubernetes/pull/62183), [@mtaufen](https://github.com/mtaufen))
* Remove alpha functionality that allowed the controller manager to approve kubelet server certificates. ([#62471](https://github.com/kubernetes/kubernetes/pull/62471), [@mikedanese](https://github.com/mikedanese))
* gitRepo volumes in pods no longer require git 1.8.5 or newer, older git versions are supported too now. ([#62394](https://github.com/kubernetes/kubernetes/pull/62394), [@jsafrane](https://github.com/jsafrane))
* Default mount propagation has changed from "HostToContainer" ("rslave" in Linux terminology) to "None" ("private") to match the behavior in 1.9 and earlier releases. "HostToContainer" as a default caused regressions in some pods. ([#62462](https://github.com/kubernetes/kubernetes/pull/62462), [@jsafrane](https://github.com/jsafrane))
* improve performance of affinity/anti-affinity predicate of default scheduler significantly. ([#62211](https://github.com/kubernetes/kubernetes/pull/62211), [@bsalamat](https://github.com/bsalamat))
* fix nsenter GetFileType issue in containerized kubelet ([#62467](https://github.com/kubernetes/kubernetes/pull/62467), [@andyzhangx](https://github.com/andyzhangx))
* Ensure expected load balancer is selected for Azure ([#62450](https://github.com/kubernetes/kubernetes/pull/62450), [@feiskyer](https://github.com/feiskyer))
* Resolves forbidden error when the `daemon-set-controller` cluster role access `controllerrevisions` resources. ([#62146](https://github.com/kubernetes/kubernetes/pull/62146), [@frodenas](https://github.com/frodenas))
* Adds --cluster-name to kubeadm init for specifying the cluster name in kubeconfig. ([#60852](https://github.com/kubernetes/kubernetes/pull/60852), [@karan](https://github.com/karan))
* Upgrade the default etcd server version to 3.2.18 ([#61198](https://github.com/kubernetes/kubernetes/pull/61198), [@jpbetz](https://github.com/jpbetz))
* [fluentd-gcp addon] Increase CPU limit for fluentd to 1 core to achieve 100kb/s throughput. ([#62430](https://github.com/kubernetes/kubernetes/pull/62430), [@bmoyles0117](https://github.com/bmoyles0117))
* GCE: Bump GLBC version to 1.1.0 - supporting multiple certificates and HTTP2 ([#62427](https://github.com/kubernetes/kubernetes/pull/62427), [@nicksardo](https://github.com/nicksardo))
* Fixed #731 kubeadm upgrade ignores HighAvailability feature gate ([#62455](https://github.com/kubernetes/kubernetes/pull/62455), [@fabriziopandini](https://github.com/fabriziopandini))
* Cluster Autoscaler 1.2.1 (release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.2.1) ([#62457](https://github.com/kubernetes/kubernetes/pull/62457), [@mwielgus](https://github.com/mwielgus))
* Add generators for `apps/v1` deployments. ([#61288](https://github.com/kubernetes/kubernetes/pull/61288), [@ayushpateria](https://github.com/ayushpateria))
* kubeadm: surface external etcd preflight validation errors ([#60585](https://github.com/kubernetes/kubernetes/pull/60585), [@alexbrand](https://github.com/alexbrand))
* kube-apiserver: oidc authentication now supports requiring specific claims with `--oidc-required-claim=<claim>=<value>` ([#62136](https://github.com/kubernetes/kubernetes/pull/62136), [@rithujohn191](https://github.com/rithujohn191))
* Implements verbosity logging feature for kubeadm commands ([#57661](https://github.com/kubernetes/kubernetes/pull/57661), [@vbmade2000](https://github.com/vbmade2000))
* Allow additionalProperties in CRD OpenAPI v3 specification for validation, mutually exclusive to properties. ([#62333](https://github.com/kubernetes/kubernetes/pull/62333), [@sttts](https://github.com/sttts))
* cinder volume plugin :  ([#61082](https://github.com/kubernetes/kubernetes/pull/61082), [@wenlxie](https://github.com/wenlxie))
    * When the cinder volume status is `error`,  controller will not do `attach ` and `detach ` operation
* fix incompatible file type checking on Windows ([#62154](https://github.com/kubernetes/kubernetes/pull/62154), [@dixudx](https://github.com/dixudx))
* fix local volume absolute path issue on Windows ([#62018](https://github.com/kubernetes/kubernetes/pull/62018), [@andyzhangx](https://github.com/andyzhangx))
* Remove `ObjectMeta ` `ListOptions` `DeleteOptions` from core api group.  Please use that in meta/v1 ([#61809](https://github.com/kubernetes/kubernetes/pull/61809), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* fix the issue that default azure disk fsypte(ext4) does not work on Windows ([#62250](https://github.com/kubernetes/kubernetes/pull/62250), [@andyzhangx](https://github.com/andyzhangx))
* RBAC information is included in audit logs via audit.Event annotations: ([#58807](https://github.com/kubernetes/kubernetes/pull/58807), [@CaoShuFeng](https://github.com/CaoShuFeng))
    * authorization.k8s.io/decision = {allow, forbid}
    * authorization.k8s.io/reason = human-readable reason for the decision
* Update kube-dns to Version 1.14.9 in kubeadm. ([#61918](https://github.com/kubernetes/kubernetes/pull/61918), [@MrHohn](https://github.com/MrHohn))
* Add support to ingest log entries to Stackdriver against new "k8s_container" and "k8s_node" resources. ([#62076](https://github.com/kubernetes/kubernetes/pull/62076), [@qingling128](https://github.com/qingling128))
* remove deprecated --mode flag in check-network-mode ([#60102](https://github.com/kubernetes/kubernetes/pull/60102), [@satyasm](https://github.com/satyasm))
* Schedule even if extender is not available when using extender  ([#61445](https://github.com/kubernetes/kubernetes/pull/61445), [@resouer](https://github.com/resouer))
* Fixed column alignment when kubectl get is used with custom columns from OpenAPI schema ([#56629](https://github.com/kubernetes/kubernetes/pull/56629), [@luksa](https://github.com/luksa))
* Fixed bug in rbd-nbd utility when nbd is used. ([#62168](https://github.com/kubernetes/kubernetes/pull/62168), [@piontec](https://github.com/piontec))
* Extend the Stackdriver Metadata Agent by adding a new Deployment for ingesting unscheduled pods, and services.   ([#62043](https://github.com/kubernetes/kubernetes/pull/62043), [@supriyagarg](https://github.com/supriyagarg))
* Disabled CheckNodeMemoryPressure and CheckNodeDiskPressure predicates if TaintNodesByCondition enabled ([#60398](https://github.com/kubernetes/kubernetes/pull/60398), [@k82cn](https://github.com/k82cn))
* kubeadm config can now override the Node CIDR Mask Size passed to kube-controller-manager. ([#61705](https://github.com/kubernetes/kubernetes/pull/61705), [@jstangroome](https://github.com/jstangroome))
* Add warnings that authors of aggregated API servers must not rely on authorization being done by the kube-apiserver. ([#61349](https://github.com/kubernetes/kubernetes/pull/61349), [@sttts](https://github.com/sttts))
* Support custom test configuration for IPAM performance integration tests ([#61959](https://github.com/kubernetes/kubernetes/pull/61959), [@satyasm](https://github.com/satyasm))
* GCE: Updates GLBC version to 1.0.1 which includes a fix which prevents multi-cluster ingress objects from creating full load balancers. ([#62075](https://github.com/kubernetes/kubernetes/pull/62075), [@nicksardo](https://github.com/nicksardo))
* OIDC authentication now allows tokens without an "email_verified" claim when using the "email" claim. If an "email_verified" claim is present when using the "email" claim, it must be `true`. ([#61508](https://github.com/kubernetes/kubernetes/pull/61508), [@rithujohn191](https://github.com/rithujohn191))
* fix local volume issue on Windows ([#62012](https://github.com/kubernetes/kubernetes/pull/62012), [@andyzhangx](https://github.com/andyzhangx))
* kubeadm: Introduce join timeout that can be controlled via the discoveryTimeout config option (set to 5 minutes by default). ([#60983](https://github.com/kubernetes/kubernetes/pull/60983), [@rosti](https://github.com/rosti))
* Add e2e test for CRD Watch ([#61025](https://github.com/kubernetes/kubernetes/pull/61025), [@ayushpateria](https://github.com/ayushpateria))
* Fix panic create/update CRD when mutating/validating webhook configured. ([#61404](https://github.com/kubernetes/kubernetes/pull/61404), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Fix a bug that fluentd doesn't inject container logs for CRI container runtimes (containerd, cri-o etc.) into elasticsearch on GCE. ([#61818](https://github.com/kubernetes/kubernetes/pull/61818), [@Random-Liu](https://github.com/Random-Liu))
* Support for "alpha.kubernetes.io/nvidia-gpu" resource which was deprecated in 1.10 is removed. Please use the resource exposed by DevicePlugins instead ("nvidia.com/gpu"). ([#61498](https://github.com/kubernetes/kubernetes/pull/61498), [@mindprince](https://github.com/mindprince))
* Pods requesting resources prefixed with `*kubernetes.io` will remain unscheduled if there are no nodes exposing that resource. ([#61860](https://github.com/kubernetes/kubernetes/pull/61860), [@mindprince](https://github.com/mindprince))
* flexvolume: trigger plugin init only for the relevant plugin while probe ([#58519](https://github.com/kubernetes/kubernetes/pull/58519), [@linyouchong](https://github.com/linyouchong))
* Update to use go1.10.1 ([#60597](https://github.com/kubernetes/kubernetes/pull/60597), [@cblecker](https://github.com/cblecker))
* Rev the Azure SDK for networking to 2017-06-01 ([#61955](https://github.com/kubernetes/kubernetes/pull/61955), [@brendandburns](https://github.com/brendandburns))
* Return error if get NodeStageSecret and NodePublishSecret failed in CSI volume plugin ([#61096](https://github.com/kubernetes/kubernetes/pull/61096), [@mlmhl](https://github.com/mlmhl))
* kubectl: improves compatibility with older servers when creating/updating API objects ([#61949](https://github.com/kubernetes/kubernetes/pull/61949), [@liggitt](https://github.com/liggitt))
* kubernetes-master charm now supports metrics server for horizontal pod autoscaler. ([#60174](https://github.com/kubernetes/kubernetes/pull/60174), [@hyperbolic2346](https://github.com/hyperbolic2346))
* fix scheduling policy on ConfigMap breaks without the --policy-configmap-namespace flag set ([#61388](https://github.com/kubernetes/kubernetes/pull/61388), [@zjj2wry](https://github.com/zjj2wry))
* kubectl: restore the ability to show resource kinds when displaying multiple objects ([#61985](https://github.com/kubernetes/kubernetes/pull/61985), [@liggitt](https://github.com/liggitt))
* `kubectl certificate approve|deny` will not modify an already approved or denied CSR unless the `--force` flag is provided. ([#61971](https://github.com/kubernetes/kubernetes/pull/61971), [@smarterclayton](https://github.com/smarterclayton))
* Kubelet now exposes a new endpoint /metrics/probes which exposes a Prometheus metric containing the liveness and/or readiness probe results for a container. ([#61369](https://github.com/kubernetes/kubernetes/pull/61369), [@rramkumar1](https://github.com/rramkumar1))
* Balanced resource allocation priority in scheduler to include volume count on node  ([#60525](https://github.com/kubernetes/kubernetes/pull/60525), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* new dhcp-domain parameter to be used for figuring out the hostname of a node ([#61890](https://github.com/kubernetes/kubernetes/pull/61890), [@dims](https://github.com/dims))
* Fixed a panic in `kubectl run --attach ...` when the api server failed to create the runtime object (due to name conflict, PSP restriction, etc.) ([#61713](https://github.com/kubernetes/kubernetes/pull/61713), [@mountkin](https://github.com/mountkin))
* Ensure reasons end up as comments in `kubectl edit`. ([#60990](https://github.com/kubernetes/kubernetes/pull/60990), [@bmcstdio](https://github.com/bmcstdio))
* kube-scheduler has been fixed to use `--leader-elect` option back to true (as it was in previous versions) ([#59732](https://github.com/kubernetes/kubernetes/pull/59732), [@dims](https://github.com/dims))
* Azure cloud provider now supports standard SKU load balancer and public IP. To use it, set cloud provider config with ([#61884](https://github.com/kubernetes/kubernetes/pull/61884), [@feiskyer](https://github.com/feiskyer))
    * {
    *   "loadBalancerSku": "standard",
    *   "excludeMasterFromStandardLB": true,
    * }
    * If excludeMasterFromStandardLB is not set, it will be default to true, which means master nodes are excluded to the backend of standard LB.
    * Also note standard load balancer doesn't work with annotation `service.beta.kubernetes.io/azure-load-balancer-mode`. This is because all nodes (except master) are added as the LB backends.
* The node authorizer now automatically sets up rules for Node.Spec.ConfigSource when the DynamicKubeletConfig feature gate is enabled. ([#60100](https://github.com/kubernetes/kubernetes/pull/60100), [@mtaufen](https://github.com/mtaufen))
* Update kube-dns to Version 1.14.9. Major changes: ([#61908](https://github.com/kubernetes/kubernetes/pull/61908), [@MrHohn](https://github.com/MrHohn))
    * Fix for kube-dns returns NXDOMAIN when not yet synced with apiserver.
    * Don't generate empty record for externalName service.
    * Add validation for upstreamNameserver port.
    * Update go version to 1.9.3.
* CRI: define the mount behavior when host path does not exist: runtime should report error if the host path doesn't exist ([#61460](https://github.com/kubernetes/kubernetes/pull/61460), [@feiskyer](https://github.com/feiskyer))
* Fixed ingress issue with CDK and pre-1.9 versions of kubernetes. ([#61859](https://github.com/kubernetes/kubernetes/pull/61859), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Removed rknetes code, which was deprecated in 1.10. ([#61432](https://github.com/kubernetes/kubernetes/pull/61432), [@filbranden](https://github.com/filbranden))
* Disable ipamperf integration tests as part of every PR verification. ([#61863](https://github.com/kubernetes/kubernetes/pull/61863), [@satyasm](https://github.com/satyasm))
* Enable server-side print in kubectl by default, with the ability to turn it off with --server-print=false ([#61477](https://github.com/kubernetes/kubernetes/pull/61477), [@soltysh](https://github.com/soltysh))
* Add ipset and udevadm to the hyperkube base image. ([#61357](https://github.com/kubernetes/kubernetes/pull/61357), [@rphillips](https://github.com/rphillips))
* In a GCE cluster, the default HAIRPIN_MODE is now "hairpin-veth". ([#60166](https://github.com/kubernetes/kubernetes/pull/60166), [@rramkumar1](https://github.com/rramkumar1))
* Deployment will stop adding pod-template-hash labels/selector to ReplicaSets and Pods it adopts. Resources created by Deployments are not affected (will still have pod-template-hash labels/selector).  ([#61615](https://github.com/kubernetes/kubernetes/pull/61615), [@janetkuo](https://github.com/janetkuo))
* kubectl: fixes issue with `-o yaml` and `-o json` omitting kind and apiVersion when used with `--dry-run` ([#61808](https://github.com/kubernetes/kubernetes/pull/61808), [@liggitt](https://github.com/liggitt))
* Updated admission controller settings for Juju deployed Kubernetes clusters ([#61427](https://github.com/kubernetes/kubernetes/pull/61427), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Performance test framework and basic tests for the IPAM controller, to simulate behavior ([#61143](https://github.com/kubernetes/kubernetes/pull/61143), [@satyasm](https://github.com/satyasm))
    * of the four supported modes under lightly loaded and loaded conditions, where load is
    * defined as the number of operations to perform as against the configured kubernetes
    * API server QPS.
* kubernetes-master charm now properly clears the client-ca-file setting on the apiserver snap ([#61479](https://github.com/kubernetes/kubernetes/pull/61479), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Fix racy panics when using fake watches with ObjectTracker ([#61195](https://github.com/kubernetes/kubernetes/pull/61195), [@grantr](https://github.com/grantr))
* [fluentd-gcp addon] Update event-exporter image to have the latest base image. ([#61727](https://github.com/kubernetes/kubernetes/pull/61727), [@crassirostris](https://github.com/crassirostris))
* Use inline func to ensure unlock is executed ([#61644](https://github.com/kubernetes/kubernetes/pull/61644), [@resouer](https://github.com/resouer))
* `kubectl apply view/edit-last-applied support completion. ([#60499](https://github.com/kubernetes/kubernetes/pull/60499), [@superbrothers](https://github.com/superbrothers))
* Automatically add system critical priority classes at cluster boostrapping. ([#60519](https://github.com/kubernetes/kubernetes/pull/60519), [@bsalamat](https://github.com/bsalamat))
* Ensure cloudprovider.InstanceNotFound is reported when the VM is not found on Azure ([#61531](https://github.com/kubernetes/kubernetes/pull/61531), [@feiskyer](https://github.com/feiskyer))
* Azure cloud provider now supports specifying allowed service tags by annotation `service.beta.kubernetes.io/azure-allowed-service-tags` ([#61467](https://github.com/kubernetes/kubernetes/pull/61467), [@feiskyer](https://github.com/feiskyer))
* Add all kinds of resource objects' statuses in HPA description. ([#59609](https://github.com/kubernetes/kubernetes/pull/59609), [@zhangxiaoyu-zidif](https://github.com/zhangxiaoyu-zidif))
* Bound cloud allocator to 10 retries with 100 ms delay between retries. ([#61375](https://github.com/kubernetes/kubernetes/pull/61375), [@satyasm](https://github.com/satyasm))
* Removed always pull policy from the template for ingress on CDK. ([#61598](https://github.com/kubernetes/kubernetes/pull/61598), [@hyperbolic2346](https://github.com/hyperbolic2346))
* escape literal percent sign when formatting ([#61523](https://github.com/kubernetes/kubernetes/pull/61523), [@dixudx](https://github.com/dixudx))
* Cluster Autoscaler 1.2.0 - release notes available here: https://github.com/kubernetes/autoscaler/releases ([#61561](https://github.com/kubernetes/kubernetes/pull/61561), [@mwielgus](https://github.com/mwielgus))
* Fix mounting of UNIX sockets(and other special files) in subpaths ([#61480](https://github.com/kubernetes/kubernetes/pull/61480), [@gnufied](https://github.com/gnufied))
* `kubectl patch` now supports `--dry-run`. ([#60675](https://github.com/kubernetes/kubernetes/pull/60675), [@timoreimann](https://github.com/timoreimann))
* fix sorting taints in case the sorting keys are equal ([#61255](https://github.com/kubernetes/kubernetes/pull/61255), [@dixudx](https://github.com/dixudx))
* NetworkPolicies can now target specific pods in other namespaces by including both a namespaceSelector and a podSelector in the same peer element. ([#60452](https://github.com/kubernetes/kubernetes/pull/60452), [@danwinship](https://github.com/danwinship))
* include node internal ip as additional information for kubectl ([#57623](https://github.com/kubernetes/kubernetes/pull/57623), [@dixudx](https://github.com/dixudx))
* Add apiserver configuration option to choose audit output version. ([#60056](https://github.com/kubernetes/kubernetes/pull/60056), [@crassirostris](https://github.com/crassirostris))
* `make test-cmd` now works on OSX. ([#61393](https://github.com/kubernetes/kubernetes/pull/61393), [@totherme](https://github.com/totherme))
* Remove kube-apiserver `--storage-version` flag, use `--storage-versions` instead. ([#61453](https://github.com/kubernetes/kubernetes/pull/61453), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Bump Heapster to v1.5.2 ([#61396](https://github.com/kubernetes/kubernetes/pull/61396), [@kawych](https://github.com/kawych))
* Conformance: ReplicaSet must be supported in the `apps/v1` version. ([#61367](https://github.com/kubernetes/kubernetes/pull/61367), [@enisoc](https://github.com/enisoc))
* You can now use the `base64decode` function in kubectl go templates to decode base64-encoded data, for example `kubectl get secret SECRET -o go-template='{{ .data.KEY | base64decode }}'`. ([#60755](https://github.com/kubernetes/kubernetes/pull/60755), [@glb](https://github.com/glb))
* Remove 'system' prefix from Metadata Agent rbac configuration ([#61394](https://github.com/kubernetes/kubernetes/pull/61394), [@kawych](https://github.com/kawych))
* Remove `--tls-ca-file` flag. ([#61386](https://github.com/kubernetes/kubernetes/pull/61386), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* fix sorting tolerations in case the keys are equal ([#61252](https://github.com/kubernetes/kubernetes/pull/61252), [@dixudx](https://github.com/dixudx))
* respect fstype in Windows for azure disk ([#61267](https://github.com/kubernetes/kubernetes/pull/61267), [@andyzhangx](https://github.com/andyzhangx))
* `--show-all` (which only affected pods and only for human readable/non-API printers) is inert in v1.11, and will be removed in a future release. ([#60793](https://github.com/kubernetes/kubernetes/pull/60793), [@charrywanganthony](https://github.com/charrywanganthony))
* Remove never used NewCronJobControllerFromClient method ([#59471](https://github.com/kubernetes/kubernetes/pull/59471), [@dmathieu](https://github.com/dmathieu))
* Support new NODE_OS_DISTRIBUTION 'custom' on GCE ([#61235](https://github.com/kubernetes/kubernetes/pull/61235), [@yguo0905](https://github.com/yguo0905))
* Fixed [#61123](https://github.com/kubernetes/kubernetes/pull/61123) by triggering syncer.Update on all cases including when a syncer is created ([#61124](https://github.com/kubernetes/kubernetes/pull/61124), [@satyasm](https://github.com/satyasm))
    * on a new add event.
* Unready pods will no longer impact the number of desired replicas when using horizontal auto-scaling with external metrics or object metrics. ([#60886](https://github.com/kubernetes/kubernetes/pull/60886), [@mattjmcnaughton](https://github.com/mattjmcnaughton))
* include file name in the error when visiting files ([#60919](https://github.com/kubernetes/kubernetes/pull/60919), [@dixudx](https://github.com/dixudx))
* Implement preemption for extender with a verb and new interface ([#58717](https://github.com/kubernetes/kubernetes/pull/58717), [@resouer](https://github.com/resouer))
* `kube-cloud-controller-manager` flag `--service-account-private-key-file` is removed in v1.11 ([#60875](https://github.com/kubernetes/kubernetes/pull/60875), [@charrywanganthony](https://github.com/charrywanganthony))
* kubeadm: Add the writable boolean option to kubeadm config. The option works on a per-volume basis for *ExtraVolumes config keys. ([#60428](https://github.com/kubernetes/kubernetes/pull/60428), [@rosti](https://github.com/rosti))
* DaemonSet scheduling associated with the alpha ScheduleDaemonSetPods feature flag has been removed from the 1.10 release. See https://github.com/kubernetes/features/issues/548 for feature status. ([#61411](https://github.com/kubernetes/kubernetes/pull/61411), [@liggitt](https://github.com/liggitt))
* Bugfix for erroneous upgrade needed messaging in kubernetes worker charm. ([#60873](https://github.com/kubernetes/kubernetes/pull/60873), [@wwwtyro](https://github.com/wwwtyro))
* Fix data race in node lifecycle controller ([#60831](https://github.com/kubernetes/kubernetes/pull/60831), [@resouer](https://github.com/resouer))
* Nodes are not deleted from kubernetes anymore if node is shutdown in Openstack. ([#59931](https://github.com/kubernetes/kubernetes/pull/59931), [@zetaab](https://github.com/zetaab))
* "beginPort+offset" format support for port range which affects kube-proxy only ([#58731](https://github.com/kubernetes/kubernetes/pull/58731), [@yue9944882](https://github.com/yue9944882))
* Added e2e test for watch ([#60331](https://github.com/kubernetes/kubernetes/pull/60331), [@jennybuckley](https://github.com/jennybuckley))
* kubelet's --cni-bin-dir option now accepts multiple comma-separated CNI binary directory paths, which are search for CNI plugins in the given order. ([#58714](https://github.com/kubernetes/kubernetes/pull/58714), [@dcbw](https://github.com/dcbw))

