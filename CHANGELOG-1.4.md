<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.4.12](#v1412)
  - [Downloads for v1.4.12](#downloads-for-v1412)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.4.9](#changelog-since-v149)
    - [Other notable changes](#other-notable-changes)
- [v1.4.9](#v149)
  - [Downloads for v1.4.9](#downloads-for-v149)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
  - [Changelog since v1.4.8](#changelog-since-v148)
    - [Other notable changes](#other-notable-changes-1)
- [v1.4.8](#v148)
  - [Downloads for v1.4.8](#downloads-for-v148)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
  - [Changelog since v1.4.7](#changelog-since-v147)
    - [Other notable changes](#other-notable-changes-2)
- [v1.4.7](#v147)
  - [Downloads for v1.4.7](#downloads-for-v147)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
  - [Changelog since v1.4.6](#changelog-since-v146)
    - [Other notable changes](#other-notable-changes-3)
- [v1.4.6](#v146)
  - [Downloads for v1.4.6](#downloads-for-v146)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
  - [Changelog since v1.4.5](#changelog-since-v145)
    - [Other notable changes](#other-notable-changes-4)
- [v1.4.5](#v145)
  - [Downloads for v1.4.5](#downloads-for-v145)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
  - [Changelog since v1.4.4](#changelog-since-v144)
    - [Other notable changes](#other-notable-changes-5)
- [v1.4.4](#v144)
  - [Downloads for v1.4.4](#downloads-for-v144)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
  - [Changelog since v1.4.3](#changelog-since-v143)
    - [Other notable changes](#other-notable-changes-6)
- [v1.4.3](#v143)
  - [Downloads](#downloads)
  - [Changelog since v1.4.2-beta.1](#changelog-since-v142-beta1)
    - [Other notable changes](#other-notable-changes-7)
- [v1.4.2](#v142)
  - [Downloads](#downloads-1)
  - [Changelog since v1.4.2-beta.1](#changelog-since-v142-beta1-1)
    - [Other notable changes](#other-notable-changes-8)
- [v1.4.2-beta.1](#v142-beta1)
  - [Downloads](#downloads-2)
  - [Changelog since v1.4.1](#changelog-since-v141)
    - [Other notable changes](#other-notable-changes-9)
- [v1.4.1](#v141)
  - [Downloads](#downloads-3)
  - [Changelog since v1.4.1-beta.2](#changelog-since-v141-beta2)
- [v1.4.1-beta.2](#v141-beta2)
  - [Downloads](#downloads-4)
  - [Changelog since v1.4.0](#changelog-since-v140)
    - [Other notable changes](#other-notable-changes-10)
- [v1.4.0](#v140)
  - [Downloads](#downloads-5)
  - [Major Themes](#major-themes)
  - [Features](#features)
  - [Known Issues](#known-issues)
  - [Notable Changes to Existing Behavior](#notable-changes-to-existing-behavior)
    - [Deployments](#deployments)
    - [kubectl rolling-update: < v1.4.0 client vs >=v1.4.0 cluster](#kubectl-rolling-update--v140-client-vs-v140-cluster)
    - [kubectl delete: < v1.4.0 client vs >=v1.4.0 cluster](#kubectl-delete--v140-client-vs-v140-cluster)
    - [DELETE operation in REST API](#delete-operation-in-rest-api)
  - [Action Required Before Upgrading](#action-required-before-upgrading)
- [optionally, remove the old secret](#optionally-remove-the-old-secret)
  - [Previous Releases Included in v1.4.0](#previous-releases-included-in-v140)
- [v1.4.0-beta.11](#v140-beta11)
  - [Downloads](#downloads-6)
  - [Changelog since v1.4.0-beta.10](#changelog-since-v140-beta10)
- [v1.4.0-beta.10](#v140-beta10)
  - [Downloads](#downloads-7)
  - [Changelog since v1.4.0-beta.8](#changelog-since-v140-beta8)
    - [Other notable changes](#other-notable-changes-11)
- [v1.4.0-beta.8](#v140-beta8)
  - [Downloads](#downloads-8)
  - [Changelog since v1.4.0-beta.7](#changelog-since-v140-beta7)
- [v1.4.0-beta.7](#v140-beta7)
  - [Downloads](#downloads-9)
  - [Changelog since v1.4.0-beta.6](#changelog-since-v140-beta6)
    - [Other notable changes](#other-notable-changes-12)
- [v1.4.0-beta.6](#v140-beta6)
  - [Downloads](#downloads-10)
  - [Changelog since v1.4.0-beta.5](#changelog-since-v140-beta5)
    - [Other notable changes](#other-notable-changes-13)
- [v1.4.0-beta.5](#v140-beta5)
  - [Downloads](#downloads-11)
  - [Changelog since v1.4.0-beta.3](#changelog-since-v140-beta3)
    - [Other notable changes](#other-notable-changes-14)
- [v1.4.0-beta.3](#v140-beta3)
  - [Downloads](#downloads-12)
  - [Changelog since v1.4.0-beta.2](#changelog-since-v140-beta2)
  - [Behavior changes caused by enabling the garbage collector](#behavior-changes-caused-by-enabling-the-garbage-collector)
    - [kubectl rolling-update](#kubectl-rolling-update)
    - [kubectl delete](#kubectl-delete)
    - [DELETE operation in REST API](#delete-operation-in-rest-api-1)
- [v1.4.0-beta.2](#v140-beta2)
  - [Downloads](#downloads-13)
  - [Changelog since v1.4.0-beta.1](#changelog-since-v140-beta1)
    - [Other notable changes](#other-notable-changes-15)
- [v1.4.0-beta.1](#v140-beta1)
  - [Downloads](#downloads-14)
  - [Changelog since v1.4.0-alpha.3](#changelog-since-v140-alpha3)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-16)
- [v1.4.0-alpha.3](#v140-alpha3)
  - [Downloads](#downloads-15)
  - [Changelog since v1.4.0-alpha.2](#changelog-since-v140-alpha2)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-17)
- [v1.4.0-alpha.2](#v140-alpha2)
  - [Downloads](#downloads-16)
  - [Changelog since v1.4.0-alpha.1](#changelog-since-v140-alpha1)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-18)
- [v1.4.0-alpha.1](#v140-alpha1)
  - [Downloads](#downloads-17)
  - [Changelog since v1.3.0](#changelog-since-v130)
    - [Experimental Features](#experimental-features)
    - [Action Required](#action-required-3)
    - [Other notable changes](#other-notable-changes-19)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


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
  - [beta] Persistent Volume provisioning now supports multiple provisioners using StorageClass configuration. ([docs](http://kubernetes.io/docs/user-guide/persistent-volumes/)) ([kubernetes/features#36](https://github.com/kubernetes/features/issues/36))
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

Please see the [Releases Page](https://github.com/kubernetes/kubernetes/releases) for older releases.

Release notes of older releases can be found in:
- [CHANGELOG-1.2.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.2.md)
- [CHANGELOG-1.3.md](https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG-1.3.md)

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/CHANGELOG.md?pixel)]()
