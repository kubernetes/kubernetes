<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.3.10](#v1310)
  - [Downloads for v1.3.10](#downloads-for-v1310)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
  - [Changelog since v1.3.9](#changelog-since-v139)
    - [Other notable changes](#other-notable-changes)
- [v1.3.9](#v139)
  - [Downloads](#downloads)
  - [Changelog since v1.3.8](#changelog-since-v138)
    - [Other notable changes](#other-notable-changes-1)
- [v1.3.8](#v138)
  - [Downloads](#downloads-1)
  - [Changelog since v1.3.7](#changelog-since-v137)
    - [Other notable changes](#other-notable-changes-2)
- [v1.3.7](#v137)
  - [Downloads](#downloads-2)
  - [Changelog since v1.3.6](#changelog-since-v136)
    - [Other notable changes](#other-notable-changes-3)
- [v1.3.6](#v136)
  - [Downloads](#downloads-3)
  - [Changelog since v1.3.5](#changelog-since-v135)
    - [Other notable changes](#other-notable-changes-4)
- [v1.3.5](#v135)
  - [Downloads](#downloads-4)
  - [Changelog since v1.3.4](#changelog-since-v134)
    - [Other notable changes](#other-notable-changes-5)
- [v1.3.4](#v134)
  - [Downloads](#downloads-5)
  - [Changelog since v1.3.3](#changelog-since-v133)
    - [Other notable changes](#other-notable-changes-6)
- [v1.3.3](#v133)
  - [Downloads](#downloads-6)
  - [Changelog since v1.3.2](#changelog-since-v132)
    - [Other notable changes](#other-notable-changes-7)
  - [Known Issues](#known-issues)
- [v1.3.2](#v132)
  - [Downloads](#downloads-7)
  - [Changelog since v1.3.1](#changelog-since-v131)
    - [Other notable changes](#other-notable-changes-8)
- [v1.3.1](#v131)
  - [Downloads](#downloads-8)
  - [Changelog since v1.3.0](#changelog-since-v130)
    - [Other notable changes](#other-notable-changes-9)
- [v1.3.0](#v130)
  - [Downloads](#downloads-9)
  - [Highlights](#highlights)
  - [Known Issues and Important Steps before Upgrading](#known-issues-and-important-steps-before-upgrading)
      - [ThirdPartyResource](#thirdpartyresource)
      - [kubectl](#kubectl)
      - [kubernetes Core Known Issues](#kubernetes-core-known-issues)
      - [Docker runtime Known Issues](#docker-runtime-known-issues)
      - [Rkt runtime Known Issues](#rkt-runtime-known-issues)
  - [Provider-specific Notes](#provider-specific-notes)
  - [Previous Releases Included in v1.3.0](#previous-releases-included-in-v130)
- [v1.3.0-beta.3](#v130-beta3)
  - [Downloads](#downloads-10)
  - [Changelog since v1.3.0-beta.2](#changelog-since-v130-beta2)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-10)
- [v1.3.0-beta.2](#v130-beta2)
  - [Downloads](#downloads-11)
  - [Changes since v1.3.0-beta.1](#changes-since-v130-beta1)
    - [Experimental Features](#experimental-features)
    - [Other notable changes](#other-notable-changes-11)
- [v1.3.0-beta.1](#v130-beta1)
  - [Downloads](#downloads-12)
  - [Changes since v1.3.0-alpha.5](#changes-since-v130-alpha5)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-12)
- [v1.3.0-alpha.5](#v130-alpha5)
  - [Downloads](#downloads-13)
  - [Changes since v1.3.0-alpha.4](#changes-since-v130-alpha4)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-13)
- [v1.3.0-alpha.4](#v130-alpha4)
  - [Downloads](#downloads-14)
  - [Changes since v1.3.0-alpha.3](#changes-since-v130-alpha3)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-14)
- [v1.3.0-alpha.3](#v130-alpha3)
  - [Downloads](#downloads-15)
  - [Changes since v1.3.0-alpha.2](#changes-since-v130-alpha2)
    - [Action Required](#action-required-4)
    - [Other notable changes](#other-notable-changes-15)
- [v1.3.0-alpha.2](#v130-alpha2)
  - [Downloads](#downloads-16)
  - [Changes since v1.3.0-alpha.1](#changes-since-v130-alpha1)
    - [Other notable changes](#other-notable-changes-16)
- [v1.3.0-alpha.1](#v130-alpha1)
  - [Downloads](#downloads-17)
  - [Changes since v1.2.0](#changes-since-v120)
    - [Action Required](#action-required-5)
    - [Other notable changes](#other-notable-changes-17)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


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



# v1.3.8

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.3/examples)

## Downloads

binary | sha256 hash
------ | -----------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.3.8/kubernetes.tar.gz) | `66cf72d8f07e2f700acfcb11536694e0d904483611ff154f34a8380c63720a8d`

## Changelog since v1.3.7

### Other notable changes

* AWS: fix volume device assignment race condition ([#31090](https://github.com/kubernetes/kubernetes/pull/31090), [@justinsb](https://github.com/justinsb))



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

Please see the [Releases Page](https://github.com/kubernetes/kubernetes/releases) for older releases.

Release notes of older releases can be found in:
- [CHANGELOG-1.2.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.2.md)

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CHANGELOG.md?pixel)]()
