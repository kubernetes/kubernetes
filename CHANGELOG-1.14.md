<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.14.0-alpha.1](#v1140-alpha1)
  - [Downloads for v1.14.0-alpha.1](#downloads-for-v1140-alpha1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.13.0](#changelog-since-v1130)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.14.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.14.0-alpha.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes.tar.gz) | `fac80e5674e547d00987516fb2eca6ea9947529307566be6a12932e3c9e430e8ad094afae748f31e9574838d98052423e3634a067f1456f7c13f6b27bfa63bcc`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-src.tar.gz) | `d1b5b2c15cb0daa076606f4ccf887724b0166dee0320f2a61d16ab4689931ab0cf5dac4c499aea3d434eb96d589d2b3effe0037e2244978d4290bd19b9a3edea`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `307c426e4abaf81648af393ddd641c225d87b02d8662d1309fe3528f14ed91b2470f6b46dc8ce0459cf196e2cec906f7eb972bf4c9a96cbd570e206f5a059dca`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `8daa85f3e8feaea0d55f20f850038dd113f0f08b62eef944b08a9109d4e69f323a8fcf20c12790c78386b454148bcc9a0cdf106ba3393620709d185c291887fa`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `28d73c299cb9859fdfeb3e4869a7a9c77f5679309c2613bd2c72d92dafd5faad0653a7377616190edd29cb8fa1aff104daba98f398e72f3447a132f208dde756`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `eb923e13026f80b743a57100d4f94995f322ab6f107c34ffd9aa74b5a6c6a4a410aff8921a4f675ace7db2ff8158a90874b8f56d3142ad2cbe615c11ec2d4535`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `279b0d0c560900021abea4bbfc25aeca7389f0b37d80022dc3335147344663424e7ba6a0abecb2dca1d2facb4163e26080750736a9a1932d67422f88b0940679`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `d69d28361b9c9e16f3e6804ccda92d55ee743e63aba7fded04edf1f7202b1fa96c235e36ab2ca17df99b4aede80b92150790885bdb7f5b4d7956af3c269dd83c`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `ca6ebb87df98bf179c94f54a4e8ae2ef2ea534b1bc5014331f937aa9d4c0442d5423651457871ef5c51f481ba8a3f449d69ef7e42e49c1b313f66cff3d44926f`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `13fa2058ceba66d8da5ba5982aa302cdd1c61d15253183ab97739229584a178f057f7979b49a035cb2355197dbb388d1642939e2c002b10e23263127030022ab`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `42ba4bba477e2958aab674a0fbf888bd5401fa5fbc39466b6cad0fc97e249ac949042c513bf176957bcb336a906e612d9c6790215e78c280225351236ec96993`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `d5f339fe4d37c61babc97208446d1859423b7679f34040f72e9138b72a18d982e66732d1f4b4f3443700f9cbe96bfc0e12eaec0a8a373fb903b49efdafcbae04`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `bcbcbd3ac4419e54e894d1e595f883e61fcf9db0353a30d794a9e5030cde8957abe8124fa5265e8c52fbc93f07cfe79b2493f791dc225468bf927b7ab4694087`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `fda4ea9168555f724659601b06737dea6ec95574569df4ef7e4ab6c2cca3327623ef310bf34f792767f00ee8069b9dd83564835d43daf973087be816be40010b`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `c142857711ec698844cd61188e70b5ab185ba2c8828cf5563a2f42958489e2ae4dbb2c1626271d4f5582167bb363e55ed03afb15e7e86cd414e0dc049fe384c0`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `524a40c5717b24c5a3b2491c4c61cf3038ba5ae7f343797a1b56a5906d6a0a3eb57e9ae78590c28ac3d441d9d1bb480a0c264a07e009a4365503ad2357614aa8`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `ef943fe326b05ece57f2e409ab1cc5fe863f5effa591abae17181c84a5eb4061e9f394ffcc8ee6ebb3f5165b183bab747a8cef540cbb1436343e8180cec037e0`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `396f7588e9131dd1b99d101c8bb94fb7e67ab067327ee58dab5a6e24887d8fbb6fc78fe50804abb0ab2f626034881d4280b3f678a1fd8b34891762bf2172b268`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `b75c1550438da0b66582d6de90436ee3c44e41e67f74947d93ee9a07ed2b7757762f3f2b05bd7b5589d7e1ea2eb3616b2ef4fe59a9fbe9d8e7cb8f0c9d3dd158`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `b6c46f9250b5565fa178ecc99ffedc6724b0bfffb73acc7d3da2c678af71008a264502cc4a48a6e7452bd0a60d77194141bbc2ea9af49176ea66e27d874b77ac`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `8d505c61a59bc9fc53d6f219d6434ddd962ba383654c46e16d413cee0ad6bd26f276a9860ad3680349bcfacb361e75de07fc44f7d14c054c47b6bd0eae63615f`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `83b6cf0fb348faa93fa40ec2a947b202b3a5a2081c3896ae39618f947a57b431bc774fbe3a5437719f50f002de252438dc16bac6f632c11140f55d5051094ae6`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.14.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `43471680533685c534023787cd40431b67041bab43e93dea457283ee0f08a8fa02ee9ade3737d8e64d1d3255a281af9a107cb61f9e4d9c99dee188c82a075580`

## Changelog since v1.13.0

### Action Required

* action required ([#68753](https://github.com/kubernetes/kubernetes/pull/68753), [@johnSchnake](https://github.com/johnSchnake))
    * If you are running E2E tests which require SSH keys and you utilize environment variables to override their location, you may need to modify the environment variable set. On all providers the environment variable override can now be either an absolute path to the key or a relative path (relative to ~/.ssh). Specifically the changes are:
    *  - Created new GCE_SSH_KEY allowing specification of SSH keys for gce, gke, and kubemark.
    *  - AWS_SSH_KEY, previously assumed to be an absolute path can now be either relative or absolute
    *  - LOCAL_SSH_KEY (for local and vsphere providers) was previously assumed to be a filename relative to ~/.ssh but can now also be an absolute path
    *  - KUBE_SSH_KEY (for skeleton provider) was previously assumed to be a filename relative to ~/.ssh but can now also be an absolute path

### Other notable changes

* Connections from Pods to Services with 0 endpoints will now ICMP reject immediately, rather than blackhole and timeout. ([#72534](https://github.com/kubernetes/kubernetes/pull/72534), [@thockin](https://github.com/thockin))
* Improve efficiency of preemption logic in clusters with many pending pods. ([#72895](https://github.com/kubernetes/kubernetes/pull/72895), [@bsalamat](https://github.com/bsalamat))
* Change scheduler metrics to conform metrics guidelines. ([#72332](https://github.com/kubernetes/kubernetes/pull/72332), [@danielqsj](https://github.com/danielqsj))
    * The following metrics are deprecated, and will be removed in a future release:
        * `e2e_scheduling_latency_microseconds`
        * `scheduling_algorithm_latency_microseconds`
        * `scheduling_algorithm_predicate_evaluation`
        * `scheduling_algorithm_priority_evaluation`
        * `scheduling_algorithm_preemption_evaluation`
        * `binding_latency_microseconds`
    * Please convert to the following metrics:
        * `e2e_scheduling_latency_seconds`
        * `scheduling_algorithm_latency_seconds`
        * `scheduling_algorithm_predicate_evaluation_seconds`
        * `scheduling_algorithm_priority_evaluation_seconds`
        * `scheduling_algorithm_preemption_evaluation_seconds`
        * `binding_latency_seconds`
* Fix SelectorSpreadPriority scheduler to match all selectors when distributing pods. ([#72801](https://github.com/kubernetes/kubernetes/pull/72801), [@Ramyak](https://github.com/Ramyak))
* Add bootstrap service account & cluster roles for node-lifecycle-controller, cloud-node-lifecycle-controller, and cloud-node-controller.   ([#72764](https://github.com/kubernetes/kubernetes/pull/72764), [@andrewsykim](https://github.com/andrewsykim))
* Fixes spurious 0-length API responses. ([#72856](https://github.com/kubernetes/kubernetes/pull/72856), [@liggitt](https://github.com/liggitt))
* Updates Fluentd to 1.3.2 & added filter_parser  ([#71180](https://github.com/kubernetes/kubernetes/pull/71180), [@monotek](https://github.com/monotek))
* The leaderelection package allows the lease holder to release its lease when the calling context is cancelled. This allows ([#71490](https://github.com/kubernetes/kubernetes/pull/71490), [@smarterclayton](https://github.com/smarterclayton))
    * faster handoff when a leader-elected process is gracefully terminated. 
* Make volume binder resilient to races between main schedule loop and async binding operation  ([#72045](https://github.com/kubernetes/kubernetes/pull/72045), [@cofyc](https://github.com/cofyc))
* Bump minimum docker API version to 1.26 (1.13.1) ([#72831](https://github.com/kubernetes/kubernetes/pull/72831), [@yujuhong](https://github.com/yujuhong))
* If the `TokenRequestProjection` feature gate is disabled, projected serviceAccountToken volume sources are now dropped at object creation time, or at object update time if the existing object did not have a projected serviceAccountToken volume source. Previously, these would result in validation errors. ([#72714](https://github.com/kubernetes/kubernetes/pull/72714), [@mourya007](https://github.com/mourya007))
* Add `metrics-port` to kube-proxy cmd flags. ([#72682](https://github.com/kubernetes/kubernetes/pull/72682), [@whypro](https://github.com/whypro))
* kubectl: fixed an issue with "too old resource version" errors continuously appearing when calling `kubectl delete` ([#72825](https://github.com/kubernetes/kubernetes/pull/72825), [@liggitt](https://github.com/liggitt))
* [Breaking change, client-go]: The WaitFor function returns, probably an ErrWaitTimeout, when the done channel is closed, even if the `WaitFunc` doesn't handle the done channel. ([#72364](https://github.com/kubernetes/kubernetes/pull/72364), [@kdada](https://github.com/kdada))
* removes newline from json output for windows nodes [#72657](https://github.com/kubernetes/kubernetes/pull/72657) ([#72659](https://github.com/kubernetes/kubernetes/pull/72659), [@jsturtevant](https://github.com/jsturtevant))
* The DenyEscalatingExec and DenyExecOnPrivileged admission plugins are deprecated and will be removed in v1.18. Use of `PodSecurityPolicy` or a custom admission plugin to limit creation of pods is recommended instead. ([#72737](https://github.com/kubernetes/kubernetes/pull/72737), [@liggitt](https://github.com/liggitt))
* Fix `describe statefulset` not printing number of desired replicas correctly ([#72781](https://github.com/kubernetes/kubernetes/pull/72781), [@tghartland](https://github.com/tghartland))
* Fix kube-proxy PodSecurityPolicy binding on GCE & GKE. This was only an issue when running kube-proxy as a DaemonSet, with PodSecurityPolicy enabled. ([#72761](https://github.com/kubernetes/kubernetes/pull/72761), [@tallclair](https://github.com/tallclair))
* Drops `status.Conditions` of new `PersistentVolume` objects if it was not set on the old object during `PrepareForUpdate`. ([#72739](https://github.com/kubernetes/kubernetes/pull/72739), [@rajathagasthya](https://github.com/rajathagasthya))
* kubelet: fixes cadvisor internal error when "--container-runtime-endpoint" is set to "unix:///var/run/crio/crio.sock". ([#72340](https://github.com/kubernetes/kubernetes/pull/72340), [@makocchi-git](https://github.com/makocchi-git))
* The `spec.SecurityContext.Sysctls` field is now dropped during creation of `Pod` objects unless the `Sysctls` feature gate is enabled. ([#72752](https://github.com/kubernetes/kubernetes/pull/72752), [@rajathagasthya](https://github.com/rajathagasthya))
    * The `spec.AllowedUnsafeSysctls` and `spec.ForbiddenSysctls` fields are now dropped during creation of `PodSecurityPolicy` objects unless the `Sysctls` feature gate is enabled.
* kubeadm: fixed storing of front-proxy certificate in secrets required by kube-controller-manager selfhosting pivoting ([#72727](https://github.com/kubernetes/kubernetes/pull/72727), [@bart0sh](https://github.com/bart0sh))
* Administrator is able to configure max pids for a pod on a node. ([#72076](https://github.com/kubernetes/kubernetes/pull/72076), [@derekwaynecarr](https://github.com/derekwaynecarr))
* Move users of `factory.NewConfigFactory` to `scheduler.New`. ([#71875](https://github.com/kubernetes/kubernetes/pull/71875), [@wgliang](https://github.com/wgliang))
* The `spec.SecurityContext.ShareProcessNamespace` field is now dropped during creation of `Pod` objects unless the `PodShareProcessNamespace ` feature gate is enabled. ([#72698](https://github.com/kubernetes/kubernetes/pull/72698), [@rajathagasthya](https://github.com/rajathagasthya))
* kube-apiserver: When configuring integration with external KMS Providers, users  can supply timeout value (i.e. how long should kube-apiserver wait before giving up on a call to KMS).  ([@immutableT](https://github.com/immutableT) ) ([#72540](https://github.com/kubernetes/kubernetes/pull/72540), [@immutableT](https://github.com/immutableT))
* The `spec.readinessGates` field is now dropped during creation of `Pod` objects unless the `PodReadinessGates` feature gate is enabled. ([#72695](https://github.com/kubernetes/kubernetes/pull/72695), [@rajathagasthya](https://github.com/rajathagasthya))
* The `spec.dataSource` field is now dropped during creation of PersistentVolumeClaim objects unless the `VolumeSnapshotDataSource` feature gate is enabled. ([#72666](https://github.com/kubernetes/kubernetes/pull/72666), [@rajathagasthya](https://github.com/rajathagasthya))
* Stop kubelet logging a warning to override hostname if there's no change detected. ([#71560](https://github.com/kubernetes/kubernetes/pull/71560), [@KashifSaadat](https://github.com/KashifSaadat))
* client-go: fake clients now properly return NotFound errors when attempting to patch non-existent objects ([#70886](https://github.com/kubernetes/kubernetes/pull/70886), [@bouk](https://github.com/bouk))
* kubectl: fixes a bug determining the correct namespace while running in a pod when the `--context` flag is explicitly specified, and the referenced context specifies the namespace `default` ([#72529](https://github.com/kubernetes/kubernetes/pull/72529), [@liggitt](https://github.com/liggitt))
* Fix scheduling starvation of pods in cluster with large number of unschedulable pods. ([#72619](https://github.com/kubernetes/kubernetes/pull/72619), [@everpeace](https://github.com/everpeace))
* If the AppArmor feature gate is disabled, AppArmor-specific annotations in pod and pod templates are dropped when the object is created, and during update of objects that do not already contain AppArmor annotations, rather than triggering a validation error. ([#72655](https://github.com/kubernetes/kubernetes/pull/72655), [@liggitt](https://github.com/liggitt))
* client-go: shortens refresh period for token files to 1 minute to ensure auto-rotated projected service account tokens are read frequently enough. ([#72437](https://github.com/kubernetes/kubernetes/pull/72437), [@liggitt](https://github.com/liggitt))
* Multiple tests which previously failed due to lack of external IP addresses defined on the nodes should now be passable. ([#68792](https://github.com/kubernetes/kubernetes/pull/68792), [@johnSchnake](https://github.com/johnSchnake))
* kubeadm: fixed incorrect controller manager pod mutations during selfhosting pivoting ([#72518](https://github.com/kubernetes/kubernetes/pull/72518), [@bart0sh](https://github.com/bart0sh))
* Increase Azure default maximumLoadBalancerRuleCount to 250. ([#72621](https://github.com/kubernetes/kubernetes/pull/72621), [@feiskyer](https://github.com/feiskyer))
* RuntimeClass is now printed with extra `RUNTIME-HANDLER` column. ([#72446](https://github.com/kubernetes/kubernetes/pull/72446), [@Huang-Wei](https://github.com/Huang-Wei))
* Updates the kubernetes dashboard add-on to v1.10.1. Skipping dashboard login is no longer enabled by default. ([#72495](https://github.com/kubernetes/kubernetes/pull/72495), [@liggitt](https://github.com/liggitt))
* [GCP] Remove confusing error log entry form fluentd scalers. ([#72243](https://github.com/kubernetes/kubernetes/pull/72243), [@cezarygerard](https://github.com/cezarygerard))
* change azure disk host cache to ReadOnly by default ([#72229](https://github.com/kubernetes/kubernetes/pull/72229), [@andyzhangx](https://github.com/andyzhangx))
* Nodes deleted in the cloud provider with Ready condition `Unknown` should also be deleted on the API server.  ([#72559](https://github.com/kubernetes/kubernetes/pull/72559), [@andrewsykim](https://github.com/andrewsykim))
* `kubectl apply --prune` now uses the apps/v1 API to prune workload resources ([#72352](https://github.com/kubernetes/kubernetes/pull/72352), [@liggitt](https://github.com/liggitt))
* Fixes a bug in HPA controller so HPAs are always updated every resyncPeriod (15 seconds). ([#72373](https://github.com/kubernetes/kubernetes/pull/72373), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* IPVS: "ExternalTrafficPolicy: Local" now works with LoadBalancer services using loadBalancerIP ([#72432](https://github.com/kubernetes/kubernetes/pull/72432), [@lbernail](https://github.com/lbernail))
* Fixes issue with cleaning up stale NFS subpath mounts ([#71804](https://github.com/kubernetes/kubernetes/pull/71804), [@msau42](https://github.com/msau42))
* Modify the scheduling result struct and improve logging for successful binding. ([#71926](https://github.com/kubernetes/kubernetes/pull/71926), [@wgliang](https://github.com/wgliang))
* Run one etcd storage compaction per default interval of 5min. Do not run one for each resource and each CRD. This fixes the compaction log spam and reduces load on etcd. ([#68557](https://github.com/kubernetes/kubernetes/pull/68557), [@sttts](https://github.com/sttts))
* kube-apiserver: `--runtime-config` can once again be used to enable/disable serving specific resources in the `extensions/v1beta1` API group. Note that specific resource enablement/disablement is only allowed for the `extensions/v1beta1` API group for legacy reasons. Attempts to enable/disable individual resources in other API groups will print a warning, and will return an error in future releases. ([#72249](https://github.com/kubernetes/kubernetes/pull/72249), [@liggitt](https://github.com/liggitt))
* kubeadm: fixed storing of etcd certificates in secrets required by kube-apiserver selfhosting pivoting ([#72478](https://github.com/kubernetes/kubernetes/pull/72478), [@bart0sh](https://github.com/bart0sh))
* kubeadm: remove the deprecated "--address" flag for controller-manager and scheduler. ([#71973](https://github.com/kubernetes/kubernetes/pull/71973), [@MalloZup](https://github.com/MalloZup))
* kube-apiserver: improves performance of requests made with service account token authentication ([#71816](https://github.com/kubernetes/kubernetes/pull/71816), [@liggitt](https://github.com/liggitt))
* Use prometheus conventions for workqueue metrics. ([#71300](https://github.com/kubernetes/kubernetes/pull/71300), [@danielqsj](https://github.com/danielqsj))
    * It is now deprecated to use the following metrics:
        * `{WorkQueueName}_depth`
        * `{WorkQueueName}_adds`
        * `{WorkQueueName}_queue_latency`
        * `{WorkQueueName}_work_duration`
        * `{WorkQueueName}_unfinished_work_seconds`
        * `{WorkQueueName}_longest_running_processor_microseconds`
        * `{WorkQueueName}_retries`
    * Please convert to the following metrics:
        * `workqueue_depth`
        * `workqueue_adds_total`
        * `workqueue_queue_latency_seconds`
        * `workqueue_work_duration_seconds`
        * `workqueue_unfinished_work_seconds`
        * `workqueue_longest_running_processor_seconds`
        * `workqueue_retries_total`
* Fix inability to use k8s with dockerd having default IPC mode set to private. ([#70826](https://github.com/kubernetes/kubernetes/pull/70826), [@kolyshkin](https://github.com/kolyshkin))
* Fix a race condition in the scheduler preemption logic that could cause nominatedNodeName of a pod not to be considered in one or more scheduling cycles. ([#72259](https://github.com/kubernetes/kubernetes/pull/72259), [@bsalamat](https://github.com/bsalamat))
* Fix registration for scheduling framework plugins with the default plugin set ([#72396](https://github.com/kubernetes/kubernetes/pull/72396), [@y-taka-23](https://github.com/y-taka-23))
* The GA VolumeScheduling feature gate can no longer be disabled and will be removed in a future release ([#72382](https://github.com/kubernetes/kubernetes/pull/72382), [@liggitt](https://github.com/liggitt))
* Fix race condition introduced by graceful termination which can lead to a deadlock in kube-proxy ([#72361](https://github.com/kubernetes/kubernetes/pull/72361), [@lbernail](https://github.com/lbernail))
* Fixes issue where subpath volume content was deleted during orphaned pod cleanup for Local volumes that are directories (and not mount points) on the root filesystem. ([#72291](https://github.com/kubernetes/kubernetes/pull/72291), [@msau42](https://github.com/msau42))
* Fixes `kubectl create secret docker-registry` compatibility ([#72344](https://github.com/kubernetes/kubernetes/pull/72344), [@liggitt](https://github.com/liggitt))
* Add-on manifests now use the apps/v1 API for DaemonSets, Deployments, and ReplicaSets ([#72203](https://github.com/kubernetes/kubernetes/pull/72203), [@liggitt](https://github.com/liggitt))
* "kubectl wait" command now supports the "--all" flag to select all resources in the namespace of the specified resource types. ([#70599](https://github.com/kubernetes/kubernetes/pull/70599), [@caesarxuchao](https://github.com/caesarxuchao))
* `deployments/rollback` is now passed through validation/admission controllers ([#72271](https://github.com/kubernetes/kubernetes/pull/72271), [@jhrv](https://github.com/jhrv))
* The `Lease` API type in the `coordination.k8s.io` API group is promoted to `v1` ([#72239](https://github.com/kubernetes/kubernetes/pull/72239), [@wojtek-t](https://github.com/wojtek-t))
* Move compatibility_test.go to pkg/scheduler/api ([#72014](https://github.com/kubernetes/kubernetes/pull/72014), [@huynq0911](https://github.com/huynq0911))
* New Azure cloud provider option 'cloudProviderBackoffMode' has been added to reduce Azure API retries. Candidate values are: ([#70866](https://github.com/kubernetes/kubernetes/pull/70866), [@feiskyer](https://github.com/feiskyer))
        * default (or empty string): keep same with before.
        * v2: only backoff retry with Azure SDK with fixed exponent 2.
* Set percentage of nodes scored in each cycle dynamically based on the cluster size. ([#72140](https://github.com/kubernetes/kubernetes/pull/72140), [@wgliang](https://github.com/wgliang))
* Fix AAD support for Azure sovereign cloud in kubectl ([#72143](https://github.com/kubernetes/kubernetes/pull/72143), [@karataliu](https://github.com/karataliu))
* Make kube-proxy service abstraction optional. ([#71355](https://github.com/kubernetes/kubernetes/pull/71355), [@bradhoekstra](https://github.com/bradhoekstra))
    * Add the 'service.kubernetes.io/service-proxy-name' label to a Service to disable the kube-proxy service proxy implementation.
* kubectl: `-A` can now be used as a shortcut for `--all-namespaces` ([#72006](https://github.com/kubernetes/kubernetes/pull/72006), [@soltysh](https://github.com/soltysh))
* discovery.CachedDiscoveryInterface implementation returned by NewMemCacheClient has changed semantics of Invalidate method -- the cache refresh is now deferred to the first cache lookup. ([#70994](https://github.com/kubernetes/kubernetes/pull/70994), [@mborsz](https://github.com/mborsz))
* Fix device mountable volume names in DSW to prevent races in device mountable plugin, e.g. local. ([#71509](https://github.com/kubernetes/kubernetes/pull/71509), [@cofyc](https://github.com/cofyc))
* Enable customize in kubectl: kubectl will be able to recognize directories with kustomization.YAML ([#70875](https://github.com/kubernetes/kubernetes/pull/70875), [@Liujingfang1](https://github.com/Liujingfang1))
* Stably sort controllerrevisions. This can prevent pods of statefulsets from continually rolling. ([#66882](https://github.com/kubernetes/kubernetes/pull/66882), [@ryanmcnamara](https://github.com/ryanmcnamara))
* Update to use go1.11.4. ([#72084](https://github.com/kubernetes/kubernetes/pull/72084), [@ixdy](https://github.com/ixdy))
* fixes an issue deleting pods containing subpath volume mounts with the VolumeSubpath feature disabled ([#70490](https://github.com/kubernetes/kubernetes/pull/70490), [@liggitt](https://github.com/liggitt))
* Clean up old eclass code ([#71399](https://github.com/kubernetes/kubernetes/pull/71399), [@resouer](https://github.com/resouer))
* Fix a race condition in which kubeadm only waits for the kubelets kubeconfig file when it has performed the TLS bootstrap, but wasn't waiting for certificates to be present in the filesystem ([#72030](https://github.com/kubernetes/kubernetes/pull/72030), [@ereslibre](https://github.com/ereslibre))
* In addition to restricting GCE metadata requests to known APIs, the metadata-proxy now restricts query strings to known parameters. ([#71094](https://github.com/kubernetes/kubernetes/pull/71094), [@dekkagaijin](https://github.com/dekkagaijin))
* kubeadm: fix a possible panic when joining a new control plane node in HA scenarios ([#72123](https://github.com/kubernetes/kubernetes/pull/72123), [@anitgandhi](https://github.com/anitgandhi))
* fix race condition when attach azure disk in vmss ([#71992](https://github.com/kubernetes/kubernetes/pull/71992), [@andyzhangx](https://github.com/andyzhangx))
* Update to use go1.11.3 with fix for CVE-2018-16875 ([#72035](https://github.com/kubernetes/kubernetes/pull/72035), [@seemethere](https://github.com/seemethere))
* kubeadm: fix a bug when syncing etcd endpoints ([#71945](https://github.com/kubernetes/kubernetes/pull/71945), [@pytimer](https://github.com/pytimer))
* fix kubelet log flushing issue in azure disk ([#71990](https://github.com/kubernetes/kubernetes/pull/71990), [@andyzhangx](https://github.com/andyzhangx))
* Disable proxy to loopback and linklocal ([#71980](https://github.com/kubernetes/kubernetes/pull/71980), [@micahhausler](https://github.com/micahhausler))
* Fix overlapping filenames in diff if multiple resources have the same name. ([#71923](https://github.com/kubernetes/kubernetes/pull/71923), [@apelisse](https://github.com/apelisse))
* fix issue: vm sku restriction policy does not work in azure disk attach/detach ([#71941](https://github.com/kubernetes/kubernetes/pull/71941), [@andyzhangx](https://github.com/andyzhangx))
* kubeadm: Create /var/lib/etcd with correct permissions (0700) by default. ([#71885](https://github.com/kubernetes/kubernetes/pull/71885), [@dims](https://github.com/dims))
* Scheduler only activates unschedulable pods if node's scheduling related properties change. ([#71551](https://github.com/kubernetes/kubernetes/pull/71551), [@mlmhl](https://github.com/mlmhl))
* kube-proxy in IPVS mode will stop initiating connections to terminating pods for services with sessionAffinity set. ([#71834](https://github.com/kubernetes/kubernetes/pull/71834), [@lbernail](https://github.com/lbernail))
* kubeadm: improve hostport parsing error messages ([#71258](https://github.com/kubernetes/kubernetes/pull/71258), [@bart0sh](https://github.com/bart0sh))
* Support graceful termination with IPVS when deleting a service ([#71895](https://github.com/kubernetes/kubernetes/pull/71895), [@lbernail](https://github.com/lbernail))
* Include CRD for BGPConfigurations, needed for calico 2.x to 3.x upgrade. ([#71868](https://github.com/kubernetes/kubernetes/pull/71868), [@satyasm](https://github.com/satyasm))
* apply: fix detection of non-dry-run enabled servers ([#71854](https://github.com/kubernetes/kubernetes/pull/71854), [@apelisse](https://github.com/apelisse))
* Clear UDP conntrack entry on endpoint changes when using nodeport  ([#71573](https://github.com/kubernetes/kubernetes/pull/71573), [@JacobTanenbaum](https://github.com/JacobTanenbaum))
* Add successful and failed history limits to cronjob describe ([#71844](https://github.com/kubernetes/kubernetes/pull/71844), [@soltysh](https://github.com/soltysh))
* kube-controller-manager: fixed issue display help for the deprecated insecure --port flag ([#71601](https://github.com/kubernetes/kubernetes/pull/71601), [@liggitt](https://github.com/liggitt))
* kubectl: fixes regression in --sort-by behavior ([#71805](https://github.com/kubernetes/kubernetes/pull/71805), [@liggitt](https://github.com/liggitt))
* Fixes pod deletion when cleaning old cronjobs ([#71801](https://github.com/kubernetes/kubernetes/pull/71801), [@soltysh](https://github.com/soltysh))
* kubeadm: use kubeconfig flag instead of kubeconfig-dir on init phase bootstrap-token ([#71803](https://github.com/kubernetes/kubernetes/pull/71803), [@yagonobre](https://github.com/yagonobre))
* kube-scheduler: restores ability to run without authentication configuration lookup permissions ([#71755](https://github.com/kubernetes/kubernetes/pull/71755), [@liggitt](https://github.com/liggitt))
* Add aggregator_unavailable_apiservice_{count,gauge} metrics in the kube-aggregator. ([#71380](https://github.com/kubernetes/kubernetes/pull/71380), [@sttts](https://github.com/sttts))
* Fixes apiserver nil pointer panics when requesting v2beta1 autoscaling object metrics ([#71744](https://github.com/kubernetes/kubernetes/pull/71744), [@yue9944882](https://github.com/yue9944882))
* Only use the first IP address got from instance metadata. This is because Azure CNI would set up a list of IP addresses in instance metadata, while only the first one is the Node's IP. ([#71736](https://github.com/kubernetes/kubernetes/pull/71736), [@feiskyer](https://github.com/feiskyer))
* client-go: restores behavior of populating the BearerToken field in rest.Config objects constructed from kubeconfig files containing tokenFile config, or from in-cluster configuration. An additional BearerTokenFile field is now populated to enable constructed clients to periodically refresh tokens. ([#71713](https://github.com/kubernetes/kubernetes/pull/71713), [@liggitt](https://github.com/liggitt))
* kubeadm: remove deprecated kubeadm config print-defaults command ([#71467](https://github.com/kubernetes/kubernetes/pull/71467), [@rosti](https://github.com/rosti))
* hack/local-up-cluster.sh now enables kubelet authentication/authorization by default (they can be disabled with KUBELET_AUTHENTICATION_WEBHOOK=false and KUBELET_AUTHORIZATION_WEBHOOK=false ([#71690](https://github.com/kubernetes/kubernetes/pull/71690), [@liggitt](https://github.com/liggitt))
* Fixes an issue where Azure VMSS instances not existing in Azure were not being deleted by the Cloud Controller Manager.  ([#71597](https://github.com/kubernetes/kubernetes/pull/71597), [@marc-sensenich](https://github.com/marc-sensenich))
* kubeadm reset correcty unmounts mount points inside /var/lib/kubelet ([#71663](https://github.com/kubernetes/kubernetes/pull/71663), [@bart0sh](https://github.com/bart0sh))
* Upgrade default etcd server to 3.3.10 ([#71615](https://github.com/kubernetes/kubernetes/pull/71615), [@jpbetz](https://github.com/jpbetz))
* When creating a service with annotation: service.beta.kubernetes.io/load-balancer-source-ranges containing multiple source ranges and service.beta.kubernetes.io/azure-shared-securityrule: "false",  the NSG rules will be collapsed. ([#71484](https://github.com/kubernetes/kubernetes/pull/71484), [@ritazh](https://github.com/ritazh))
* disable node's proxy use of http probe ([#68663](https://github.com/kubernetes/kubernetes/pull/68663), [@WanLinghao](https://github.com/WanLinghao))
* Bumps version of kubernetes-cni to 0.6.0 ([#71629](https://github.com/kubernetes/kubernetes/pull/71629), [@mauilion](https://github.com/mauilion))
* On GCI, NPD starts to monitor kubelet, docker, containerd crashlooping, read-only filesystem and corrupt docker overlay2 issues. ([#71522](https://github.com/kubernetes/kubernetes/pull/71522), [@wangzhen127](https://github.com/wangzhen127))
* When a kubelet is using --bootstrap-kubeconfig and certificate rotation, it no longer waits for bootstrap to succeed before launching static pods. ([#71174](https://github.com/kubernetes/kubernetes/pull/71174), [@smarterclayton](https://github.com/smarterclayton))
* Add an plugin interfaces for "reserve" and "prebind" extension points of the scheduling framework. ([#70227](https://github.com/kubernetes/kubernetes/pull/70227), [@bsalamat](https://github.com/bsalamat))
* Fix scheduling starvation of pods in cluster with large number of unschedulable pods. ([#71488](https://github.com/kubernetes/kubernetes/pull/71488), [@bsalamat](https://github.com/bsalamat))
* Reduce CSI log and event spam. ([#71581](https://github.com/kubernetes/kubernetes/pull/71581), [@saad-ali](https://github.com/saad-ali))
* Add conntrack as a dependency of kubelet and kubeadm when building rpms and debs. Both require conntrack to handle cleanup of connections. ([#71540](https://github.com/kubernetes/kubernetes/pull/71540), [@mauilion](https://github.com/mauilion))
* UDP connections now support graceful termination in IPVS mode ([#71515](https://github.com/kubernetes/kubernetes/pull/71515), [@lbernail](https://github.com/lbernail))
* Log etcd client errors. The verbosity is set with the usual `-v` flag. ([#71318](https://github.com/kubernetes/kubernetes/pull/71318), [@sttts](https://github.com/sttts))
* The `DefaultFeatureGate` package variable now only exposes readonly feature gate methods. Methods for mutating feature gates have moved into a `MutableFeatureGate` interface and are accessible via the `DefaultMutableFeatureGate` package variable. Only top-level commands and options setup should access `DefaultMutableFeatureGate`. ([#71302](https://github.com/kubernetes/kubernetes/pull/71302), [@liggitt](https://github.com/liggitt))
* `node.kubernetes.io/pid-pressure` toleration is added for DaemonSet pods, and `node.kubernetes.io/out-of-disk` isn't added any more even if it's a critical pod. ([#67036](https://github.com/kubernetes/kubernetes/pull/67036), [@Huang-Wei](https://github.com/Huang-Wei))
* Update k8s.io/utils to allow for asynchronous process control ([#71047](https://github.com/kubernetes/kubernetes/pull/71047), [@hoegaarden](https://github.com/hoegaarden))
* Fixes possible panic during volume detach, if corresponding volume plugin became non-attachable ([#71471](https://github.com/kubernetes/kubernetes/pull/71471), [@mshaverdo](https://github.com/mshaverdo))
* Fix cloud-controller-manager crash when using AWS provider and PersistentVolume initializing controller  ([#70432](https://github.com/kubernetes/kubernetes/pull/70432), [@mvladev](https://github.com/mvladev))
* Fixes an issue where Portworx volumes cannot be mounted if 9001 port is already in use on the host and users remap 9001 to another port. ([#70392](https://github.com/kubernetes/kubernetes/pull/70392), [@harsh-px](https://github.com/harsh-px))
* Fix `SubPath` printing of `VolumeMounts`. ([#70127](https://github.com/kubernetes/kubernetes/pull/70127), [@dtaniwaki](https://github.com/dtaniwaki))
* Fixes incorrect paths (missing first letter) when copying files from pods to ([#69885](https://github.com/kubernetes/kubernetes/pull/69885), [@clickyotomy](https://github.com/clickyotomy))
    * local in `kubectl cp'.
* Fix AWS NLB security group updates where valid security group ports were incorrectly removed ([#68422](https://github.com/kubernetes/kubernetes/pull/68422), [@kellycampbell](https://github.com/kellycampbell))
    * when updating a service or when node changes occur.

