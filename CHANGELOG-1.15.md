<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.15.0-beta.1](#v1150-beta1)
  - [Downloads for v1.15.0-beta.1](#downloads-for-v1150-beta1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.15.0-alpha.3](#changelog-since-v1150-alpha3)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
- [v1.15.0-alpha.3](#v1150-alpha3)
  - [Downloads for v1.15.0-alpha.3](#downloads-for-v1150-alpha3)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.15.0-alpha.2](#changelog-since-v1150-alpha2)
    - [Other notable changes](#other-notable-changes-1)
- [v1.15.0-alpha.2](#v1150-alpha2)
  - [Downloads for v1.15.0-alpha.2](#downloads-for-v1150-alpha2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.15.0-alpha.1](#changelog-since-v1150-alpha1)
    - [Other notable changes](#other-notable-changes-2)
- [v1.15.0-alpha.1](#v1150-alpha1)
  - [Downloads for v1.15.0-alpha.1](#downloads-for-v1150-alpha1)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.14.0](#changelog-since-v1140)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-3)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


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

