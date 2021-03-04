<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.2.7](#v127)
  - [Downloads for v1.2.7](#downloads-for-v127)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
  - [Changelog since v1.2.6](#changelog-since-v126)
    - [Other notable changes](#other-notable-changes)
- [v1.2.6](#v126)
  - [Downloads for v1.2.6](#downloads-for-v126)
  - [Changelog since v1.2.5](#changelog-since-v125)
    - [Other notable changes](#other-notable-changes-1)
- [v1.2.5](#v125)
  - [Downloads for v1.2.5](#downloads-for-v125)
  - [Changes since v1.2.4](#changes-since-v124)
    - [Other notable changes](#other-notable-changes-2)
- [v1.2.4](#v124)
  - [Downloads for v1.2.4](#downloads-for-v124)
  - [Changes since v1.2.3](#changes-since-v123)
    - [Other notable changes](#other-notable-changes-3)
- [v1.2.3](#v123)
  - [Downloads for v1.2.3](#downloads-for-v123)
  - [Changes since v1.2.2](#changes-since-v122)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-4)
- [v1.2.2](#v122)
  - [Downloads for v1.2.2](#downloads-for-v122)
  - [Changes since v1.2.1](#changes-since-v121)
    - [Other notable changes](#other-notable-changes-5)
- [v1.2.1](#v121)
  - [Downloads for v1.2.1](#downloads-for-v121)
  - [Changes since v1.2.0](#changes-since-v120)
    - [Other notable changes](#other-notable-changes-6)
- [v1.2.0](#v120)
  - [Downloads for v1.2.0](#downloads-for-v120)
  - [Changes since v1.1.1](#changes-since-v111)
    - [Major Themes](#major-themes)
    - [Other notable improvements](#other-notable-improvements)
    - [Experimental Features](#experimental-features)
    - [Action required](#action-required-1)
    - [Known Issues](#known-issues)
      - [Docker Known Issues](#docker-known-issues)
        - [1.9.1](#191)
    - [Provider-specific Notes](#provider-specific-notes)
      - [Various](#various)
      - [AWS](#aws)
      - [GCE](#gce)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


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



# v1.2.6

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads for v1.2.6

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.6/kubernetes.tar.gz) | `50023455d00af52c41a7158b4bd117b2dfd4a100` | `cf0411bcb620eb13b08b93578efffc43`

## Changelog since v1.2.5

### Other notable changes

* Fix watch cache filtering ([#28967](https://github.com/kubernetes/kubernetes/pull/28967), [@liggitt](https://github.com/liggitt))
* Fix problems with container restarts and flocker ([#25874](https://github.com/kubernetes/kubernetes/pull/25874), [@simonswine](https://github.com/simonswine))



# v1.2.5

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads for v1.2.5

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



# v1.2.4

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads for v1.2.4

binary | sha1 hash | md5 hash
------ | --------- | --------
[kubernetes.tar.gz](https://storage.googleapis.com/kubernetes-release/release/v1.2.4/kubernetes.tar.gz) | `f3aea83f8f0e16b2b41998a2edc09eb42fd8d945` | `ab0aca3a20e8eba43c8ff9d672793618`

## Changes since v1.2.3

### Other notable changes

* Ensure status is not changed during an update of PV, PVC, HPA objects ([#24924](https://github.com/kubernetes/kubernetes/pull/24924), [@mqliang](https://github.com/mqliang))
* GCI: Add two GCI specific metadata pairs ([#25105](https://github.com/kubernetes/kubernetes/pull/25105), [@andyzheng0831](https://github.com/andyzheng0831))
* Add an entry to the salt config to allow Debian jessie on GCE. ([#25123](https://github.com/kubernetes/kubernetes/pull/25123), [@jlewi](https://github.com/jlewi))
    * As with the existing Wheezy image on GCE, docker is expected to already be installed in the image.
* Fix DeletingLoadBalancer event generation. ([#24833](https://github.com/kubernetes/kubernetes/pull/24833), [@a-robinson](https://github.com/a-robinson))
* GCE: Prefer preconfigured node tags for firewalls, if available ([#25148](https://github.com/kubernetes/kubernetes/pull/25148), [@a-robinson](https://github.com/a-robinson))
* Drain pods created from ReplicaSets in 'kubectl drain' ([#23689](https://github.com/kubernetes/kubernetes/pull/23689), [@maclof](https://github.com/maclof))
* GCI: Update the command to get the image ([#24987](https://github.com/kubernetes/kubernetes/pull/24987), [@andyzheng0831](https://github.com/andyzheng0831))
* Validate deletion timestamp doesn't change on update ([#24839](https://github.com/kubernetes/kubernetes/pull/24839), [@liggitt](https://github.com/liggitt))
* Add support for running clusters on GCI ([#24893](https://github.com/kubernetes/kubernetes/pull/24893), [@andyzheng0831](https://github.com/andyzheng0831))
* Trusty: Add retry in curl commands ([#24749](https://github.com/kubernetes/kubernetes/pull/24749), [@andyzheng0831](https://github.com/andyzheng0831))



# v1.2.3

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads for v1.2.3

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



# v1.2.2

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads for v1.2.2

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

## Downloads for v1.2.1

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



# v1.2.0

[Documentation](http://kubernetes.github.io) & [Examples](http://releases.k8s.io/release-1.2/examples)

## Downloads for v1.2.0

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