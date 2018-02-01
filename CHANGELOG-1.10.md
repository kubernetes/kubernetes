<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.10.0-alpha.3](#v1100-alpha3)
  - [Downloads for v1.10.0-alpha.3](#downloads-for-v1100-alpha3)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.10.0-alpha.2](#changelog-since-v1100-alpha2)
    - [Other notable changes](#other-notable-changes)
- [v1.10.0-alpha.2](#v1100-alpha2)
  - [Downloads for v1.10.0-alpha.2](#downloads-for-v1100-alpha2)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.10.0-alpha.1](#changelog-since-v1100-alpha1)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes-1)
- [v1.10.0-alpha.1](#v1100-alpha1)
  - [Downloads for v1.10.0-alpha.1](#downloads-for-v1100-alpha1)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.9.0](#changelog-since-v190)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-2)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.10.0-alpha.3

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.10.0-alpha.3


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes.tar.gz) | `246f0373ccb25a243a387527b32354b69fc2211c422e71479d22bfb3a829c8fb`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-src.tar.gz) | `f9c60bb37fb7b363c9f66d8efd8aa5a36ea2093c61317c950719b3ddc86c5e10`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `ca8dfd7fbd34478e7ba9bba3779fcca08f7efd4f218b0c8a7f52bbeea0f42cd7`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `713c35d99f44bd19d225d2c9f2d7c4f3976b5dd76e9a817b2aaf68ee0cb5a939`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `7601e55e3bb0f0fc11611c68c4bc000c3cbbb7a09652c386e482a1671be7e2d6`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `8a6c498531c1832176e22d622008a98bac6043f05dec96747649651531ed3fd7`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `81561820fb5a000152e9d8d94882e0ed6228025ea7973ee98173b5fc89d62a42`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `6ce8c3ed253a10d78e62e000419653a29c411cd64910325b21ff3370cb0a89eb`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `a46b42c94040767f6bbf2ce10aef36d8dbe94c0069f866a848d69b2274f8f0bc`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `fa3e656b612277fc4c303aef95c60b58ed887e36431db23d26b536f226a23cf6`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `832e12266495ac55cb54a999bc5ae41d42d160387b487d8b4ead577d96686b62`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `7056a3eb5a8f9e8fa0326aa6e0bf97fc5b260447315f8ec7340be5747a16f5fd`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `dc8e2be2fcb6477249621fb5c813c853371a3bf8732c5cb3a6d6cab667cfa324`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `399071ad9042a72bccd6e1aa322405c02b4a807c0b4f987d608c4c9c369979d6`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `7457ad16665e331fa9224a3d61690206723721197ad9760c3b488de9602293f5`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `ffcb728d879c0347bd751c9bccac3520bb057d203ba1acd55f8c727295282049`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `f942f6e15886a1fb0d91d04adf47677068c56070dff060f38c371c3ee3e99648`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `81b22beb30be9d270016c7b35b86ea585f29c0c5f09128da9341f9f67c8865f9`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `d9020b99c145f44c519b1a95b55ed24e69d9c679a02352c7e05e86042daca9d1`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `1d10bee4ed62d70b318f5703b2cd8295a08e199f810d6b361f367907e3f01fb6`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `67cd4dde212abda37e6f9e6dee1bb59db96e0727100ef0aa561c15562df0f3e1`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `362b030e011ea6222b1f2dec62311d3971bcce4dba94997963e2a091efbf967b`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `e609a2b0410acbb64d3ee6d7f134d98723d82d05bdbead1eaafd3584d3e45c39`

## Changelog since v1.10.0-alpha.2

### Other notable changes

* Fixed issue with kubernetes-worker option allow-privileged not properly handling the value True with a capital T. ([#59116](https://github.com/kubernetes/kubernetes/pull/59116), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Added anti-affinity to kube-dns pods ([#57683](https://github.com/kubernetes/kubernetes/pull/57683), [@vainu-arto](https://github.com/vainu-arto))
* cloudprovider/openstack: fix bug the tries to use octavia client to query flip ([#59075](https://github.com/kubernetes/kubernetes/pull/59075), [@jrperritt](https://github.com/jrperritt))
* Windows containers now support experimental Hyper-V isolation by setting annotation `experimental.windows.kubernetes.io/isolation-type=hyperv` and feature gates HyperVContainer. Only one container per pod is supported yet. ([#58751](https://github.com/kubernetes/kubernetes/pull/58751), [@feiskyer](https://github.com/feiskyer))
* `crds` is added as a shortname for CustomResourceDefinition i.e. `kubectl get crds` can now be used. ([#59061](https://github.com/kubernetes/kubernetes/pull/59061), [@nikhita](https://github.com/nikhita))
* Fix an issue where port forwarding doesn't forward local TCP6 ports to the pod ([#57457](https://github.com/kubernetes/kubernetes/pull/57457), [@vfreex](https://github.com/vfreex))
* YAMLDecoder Read now tracks rest of buffer on io.ErrShortBuffer ([#58817](https://github.com/kubernetes/kubernetes/pull/58817), [@karlhungus](https://github.com/karlhungus))
* Prevent kubelet from getting wedged if initialization of modules returns an error. ([#59020](https://github.com/kubernetes/kubernetes/pull/59020), [@brendandburns](https://github.com/brendandburns))
* Fixed a race condition inside kubernetes-worker that would result in a temporary error situation. ([#59005](https://github.com/kubernetes/kubernetes/pull/59005), [@hyperbolic2346](https://github.com/hyperbolic2346))
* [GCE] Apiserver uses `InternalIP` as the most preferred kubelet address type by default. ([#59019](https://github.com/kubernetes/kubernetes/pull/59019), [@MrHohn](https://github.com/MrHohn))
* Deprecate insecure flags `--insecure-bind-address`, `--insecure-port` and remove  `--public-address-override`. ([#59018](https://github.com/kubernetes/kubernetes/pull/59018), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* Support GetLabelsForVolume in OpenStack Provider ([#58871](https://github.com/kubernetes/kubernetes/pull/58871), [@edisonxiang](https://github.com/edisonxiang))
* Build using go1.9.3. ([#59012](https://github.com/kubernetes/kubernetes/pull/59012), [@ixdy](https://github.com/ixdy))
* CRI: Add a call to reopen log file for a container.  ([#58899](https://github.com/kubernetes/kubernetes/pull/58899), [@yujuhong](https://github.com/yujuhong))
* The alpha KubeletConfigFile feature gate has been removed, because it was redundant with the Kubelet's --config flag. It is no longer necessary to set this gate to use the flag. The --config flag is still considered alpha. ([#58978](https://github.com/kubernetes/kubernetes/pull/58978), [@mtaufen](https://github.com/mtaufen))
* `kubectl scale` can now scale any resource (kube, CRD, aggregate) conforming to the standard scale endpoint ([#58298](https://github.com/kubernetes/kubernetes/pull/58298), [@p0lyn0mial](https://github.com/p0lyn0mial))
* kube-apiserver flag --tls-ca-file has had no effect for some time.  It is now deprecated and slated for removal in 1.11.  If you are specifying this flag, you must remove it from your launch config before ugprading to 1.11. ([#58968](https://github.com/kubernetes/kubernetes/pull/58968), [@deads2k](https://github.com/deads2k))
* Fix regression in the CRI: do not add a default hostname on short image names ([#58955](https://github.com/kubernetes/kubernetes/pull/58955), [@runcom](https://github.com/runcom))
* Get windows kernel version directly from registry ([#58498](https://github.com/kubernetes/kubernetes/pull/58498), [@feiskyer](https://github.com/feiskyer))
* Remove deprecated --require-kubeconfig flag, remove default --kubeconfig value ([#58367](https://github.com/kubernetes/kubernetes/pull/58367), [@zhangxiaoyu-zidif](https://github.com/zhangxiaoyu-zidif))
* Google Cloud Service Account email addresses can now be used in RBAC ([#58141](https://github.com/kubernetes/kubernetes/pull/58141), [@ahmetb](https://github.com/ahmetb))
    * Role bindings since the default scopes now include the "userinfo.email"
    * scope. This is a breaking change if the numeric uniqueIDs of the Google
    * service accounts were being used in RBAC role bindings. The behavior
    * can be overridden by explicitly specifying the scope values as
    * comma-separated string in the "users[*].config.scopes" field in the
    * KUBECONFIG file.
* kube-apiserver is changed to use SSH tunnels for webhook iff the webhook is not directly routable from apiserver's network environment. ([#58644](https://github.com/kubernetes/kubernetes/pull/58644), [@yguo0905](https://github.com/yguo0905))
* Updated priority of mirror pod according to PriorityClassName. ([#58485](https://github.com/kubernetes/kubernetes/pull/58485), [@k82cn](https://github.com/k82cn))
* Fixes a bug where kubelet crashes trying to free memory under memory pressure ([#58574](https://github.com/kubernetes/kubernetes/pull/58574), [@yastij](https://github.com/yastij))



# v1.10.0-alpha.2

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.10.0-alpha.2


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes.tar.gz) | `89efeb8b16c40e5074f092f51399995f0fe4a0312367a8f54bd227c3c6fcb629`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-src.tar.gz) | `eefbbf435f1b7a0e416f4e6b2c936c49ce5d692994da8d235c5e25bc408eec57`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `878366200ddfb9128a133d7d377057c6f878b24357062cf5243c0f0aac26b292`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `dc065b9ecfa513607eac6e7dd125b2c25c9a9e7c13d0b2b6e56586e17bbd6ae5`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `93c2462051935d8f6bca6c72d09948963d47cd64426660f63e0cea7d37e24812`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `0eef61285fad1f9ff8392c59986d3a41887abc642bcb5cb451c5a5300927e2c4`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `6cf7913730a57b503beaf37f5c4d0f97789358983ed03654036f8b986b60cc62`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `f03c3ecbf4c08d263f2daa8cbe838e20452d6650b80e9a74762c155c26a579b7`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `25a2f93ebb721901d262adae4c0bdaa4cf1293793e9dff4507e031b85f46aff8`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `3e0b9ef771f36edb61bd61ccb67996ed41793c01f8686509bf93e585ee882c94`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `387e5e6b0535f4f5996c0732f1b591d80691acaec86e35482c7b90e00a1856f7`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `c10a72d40252707b732d33d03beec3c6380802d0a6e3214cbbf4af258fddf28c`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `42c1e016e8b0c5cc36c7bf574abca18c63e16d719d35e19ddbcbcd5aaeabc46c`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `b7774c54344c75bf5c703d4ca271f0af6c230e86cbe40eafd9cbf98a4f4be6e9`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `c11c8554506b64d6fd1a6e79bfc4e1e19f4f826b9ba98de81bc757901e8cdc43`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `196bd957804b2a9049189d225e49bf78e52e9adef12c072128e4e85d35da438e`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `be12fbea28a6cb089734782fe11e6f90a30785b9ad1ec02bc08a59afeb95c173`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `a1feb239dfc473b49adf95d7d94e4a9c6c7d07416d4e935e3fc10175ffaa7163`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `26583c0bd08313bdc0bdfba6745f3ccd0f117431d3a5e2623bb5015675d506b8`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `79c6299a5482467e3e85ee881f21edf5d491bc28c94e547d9297d1e1ad1b7458`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `2732fd288f1eac44c599423ce28cbdb85b54a646970a3714be5ff86d1b14b5e2`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `8d49432f0ff3baf55e71c29fb6ffc1673b2a45b9eae2e1906138b1409da53940`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `15ff74edfa98cd1afadcc4e53dd592b1e2935fbab76ad731309d355ae23bdd09`

## Changelog since v1.10.0-alpha.1

### Action Required

* Bug fix: webhooks now do not skip cluster-scoped resources ([#58185](https://github.com/kubernetes/kubernetes/pull/58185), [@caesarxuchao](https://github.com/caesarxuchao))
    * Action required: Before upgrading your Kubernetes clusters, double check if you had configured webhooks for cluster-scoped objects (e.g., nodes, persistentVolume), these webhooks will start to take effect. Delete/modify the configs if that's not desirable.

### Other notable changes

* Fixing extra_sans option on master and load balancer. ([#58843](https://github.com/kubernetes/kubernetes/pull/58843), [@hyperbolic2346](https://github.com/hyperbolic2346))
* ConfigMap objects now support binary data via a new `binaryData` field. When using `kubectl create configmap --from-file`, files containing non-UTF8 data will be placed in this new field in order to preserve the non-UTF8 data. Use of this feature requires 1.10+ apiserver and kubelets. ([#57938](https://github.com/kubernetes/kubernetes/pull/57938), [@dims](https://github.com/dims))
* New alpha feature to limit the number of processes running in a pod. Cluster administrators will be able to place limits by using the new kubelet command line parameter --pod-max-pids. Note that since this is a alpha feature they will need to enable the "SupportPodPidsLimit" feature. ([#57973](https://github.com/kubernetes/kubernetes/pull/57973), [@dims](https://github.com/dims))
* Add storage-backend configuration option to kubernetes-master charm. ([#58830](https://github.com/kubernetes/kubernetes/pull/58830), [@wwwtyro](https://github.com/wwwtyro))
* use containing API group when resolving shortname from discovery ([#58741](https://github.com/kubernetes/kubernetes/pull/58741), [@dixudx](https://github.com/dixudx))
* Fix kubectl explain for resources not existing in default version of API group ([#58753](https://github.com/kubernetes/kubernetes/pull/58753), [@soltysh](https://github.com/soltysh))
* Ensure config has been created before attempting to launch ingress. ([#58756](https://github.com/kubernetes/kubernetes/pull/58756), [@wwwtyro](https://github.com/wwwtyro))
* Access to externally managed IP addresses via the kube-apiserver service proxy subresource is no longer allowed by default. This can be re-enabled via the `ServiceProxyAllowExternalIPs` feature gate, but will be disallowed completely in 1.11 ([#57265](https://github.com/kubernetes/kubernetes/pull/57265), [@brendandburns](https://github.com/brendandburns))
* Added support for external cloud providers in kubeadm ([#58259](https://github.com/kubernetes/kubernetes/pull/58259), [@dims](https://github.com/dims))
* rktnetes has been deprecated in favor of rktlet. Please see https://github.com/kubernetes-incubator/rktlet for more information. ([#58418](https://github.com/kubernetes/kubernetes/pull/58418), [@yujuhong](https://github.com/yujuhong))
* Fixes bug finding master replicas in GCE when running multiple Kubernetes clusters ([#58561](https://github.com/kubernetes/kubernetes/pull/58561), [@jesseshieh](https://github.com/jesseshieh))
* Update Calico version to v2.6.6 ([#58482](https://github.com/kubernetes/kubernetes/pull/58482), [@tmjd](https://github.com/tmjd))
* Promoting the apiregistration.k8s.io (aggregation) to GA ([#58393](https://github.com/kubernetes/kubernetes/pull/58393), [@deads2k](https://github.com/deads2k))
* Stability: Make Pod delete event handling of scheduler more robust. ([#58712](https://github.com/kubernetes/kubernetes/pull/58712), [@bsalamat](https://github.com/bsalamat))
* Added support for network spaces in the kubeapi-load-balancer charm ([#58708](https://github.com/kubernetes/kubernetes/pull/58708), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Added support for network spaces in the kubernetes-master charm ([#58704](https://github.com/kubernetes/kubernetes/pull/58704), [@hyperbolic2346](https://github.com/hyperbolic2346))
* update etcd unified version to 3.1.10 ([#54242](https://github.com/kubernetes/kubernetes/pull/54242), [@zouyee](https://github.com/zouyee))
* updates fluentd in fluentd-es-image to fluentd 1.1.0 ([#58525](https://github.com/kubernetes/kubernetes/pull/58525), [@monotek](https://github.com/monotek))
* Support metrics API in `kubectl top` commands. ([#56206](https://github.com/kubernetes/kubernetes/pull/56206), [@brancz](https://github.com/brancz))
* Added support for network spaces in the kubernetes-worker charm ([#58523](https://github.com/kubernetes/kubernetes/pull/58523), [@hyperbolic2346](https://github.com/hyperbolic2346))
* CustomResourceDefinitions: OpenAPI v3 validation schemas containing `$ref`references are no longer permitted (valid references could not be constructed previously because property ids were not permitted either). Before upgrading, ensure CRD definitions do not include those `$ref` fields. ([#58438](https://github.com/kubernetes/kubernetes/pull/58438), [@carlory](https://github.com/carlory))
* Openstack: register metadata.hostname as node name ([#58502](https://github.com/kubernetes/kubernetes/pull/58502), [@dixudx](https://github.com/dixudx))
* Added nginx and default backend images to kubernetes-worker config. ([#58542](https://github.com/kubernetes/kubernetes/pull/58542), [@hyperbolic2346](https://github.com/hyperbolic2346))
* --tls-min-version on kubelet and kube-apiserver allow for configuring minimum TLS versions ([#58528](https://github.com/kubernetes/kubernetes/pull/58528), [@deads2k](https://github.com/deads2k))
* Fixes an issue where the resourceVersion of an object in a DELETE watch event was not the resourceVersion of the delete itself, but of the last update to the object. This could disrupt the ability of clients clients to re-establish watches properly. ([#58547](https://github.com/kubernetes/kubernetes/pull/58547), [@liggitt](https://github.com/liggitt))
* Fixed crash in kubectl cp when path has multiple leading slashes ([#58144](https://github.com/kubernetes/kubernetes/pull/58144), [@tomerf](https://github.com/tomerf))
* kube-apiserver: requests to endpoints handled by unavailable extension API servers (as indicated by an `Available` condition of `false` in the registered APIService) now return `503` errors instead of `404` errors. ([#58070](https://github.com/kubernetes/kubernetes/pull/58070), [@weekface](https://github.com/weekface))
* Correctly handle transient connection reset errors on GET requests from client library. ([#58520](https://github.com/kubernetes/kubernetes/pull/58520), [@porridge](https://github.com/porridge))
* Authentication information for OpenStack cloud provider can now be specified as environment variables ([#58300](https://github.com/kubernetes/kubernetes/pull/58300), [@dims](https://github.com/dims))
* Bump GCE metadata proxy to v0.1.9 to pick up security fixes. ([#58221](https://github.com/kubernetes/kubernetes/pull/58221), [@ihmccreery](https://github.com/ihmccreery))
* - kubeadm now supports CIDR notations in NO_PROXY environment variable ([#53895](https://github.com/kubernetes/kubernetes/pull/53895), [@kad](https://github.com/kad))
* kubeadm now accept `--apiserver-extra-args`, `--controller-manager-extra-args` and `--scheduler-extra-args` to override / specify additional flags for control plane components ([#58080](https://github.com/kubernetes/kubernetes/pull/58080), [@simonferquel](https://github.com/simonferquel))
* Add `--enable-admission-plugin` `--disable-admission-plugin` flags and deprecate `--admission-control`. ([#58123](https://github.com/kubernetes/kubernetes/pull/58123), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
    * Afterwards, don't care about the orders specified in the flags.
* "ExternalTrafficLocalOnly" has been removed from feature gate. It has been a GA feature since v1.7. ([#56948](https://github.com/kubernetes/kubernetes/pull/56948), [@MrHohn](https://github.com/MrHohn))
* GCP: allow a master to not include a metadata concealment firewall rule (if it's not running the metadata proxy). ([#58104](https://github.com/kubernetes/kubernetes/pull/58104), [@ihmccreery](https://github.com/ihmccreery))
* kube-apiserver: fixes loading of `--admission-control-config-file` containing AdmissionConfiguration apiserver.k8s.io/v1alpha1 config object ([#58439](https://github.com/kubernetes/kubernetes/pull/58439), [@liggitt](https://github.com/liggitt))
* Fix issue when using OpenStack config drive for node metadata ([#57561](https://github.com/kubernetes/kubernetes/pull/57561), [@dims](https://github.com/dims))
* Add FSType for CSI volume source to specify filesystems ([#58209](https://github.com/kubernetes/kubernetes/pull/58209), [@NickrenREN](https://github.com/NickrenREN))
* OpenStack cloudprovider: Ensure orphaned routes are removed. ([#56258](https://github.com/kubernetes/kubernetes/pull/56258), [@databus23](https://github.com/databus23))
* Reduce Metrics Server memory requirement ([#58391](https://github.com/kubernetes/kubernetes/pull/58391), [@kawych](https://github.com/kawych))
* Fix a bug affecting nested data volumes such as secret, configmap, etc. ([#57422](https://github.com/kubernetes/kubernetes/pull/57422), [@joelsmith](https://github.com/joelsmith))
* kubectl now enforces required flags at a more fundamental level ([#53631](https://github.com/kubernetes/kubernetes/pull/53631), [@dixudx](https://github.com/dixudx))
* Remove alpha Initializers from kubadm admission control ([#58428](https://github.com/kubernetes/kubernetes/pull/58428), [@dixudx](https://github.com/dixudx))
* Enable ValidatingAdmissionWebhook and MutatingAdmissionWebhook in kubeadm from v1.9 ([#58255](https://github.com/kubernetes/kubernetes/pull/58255), [@dixudx](https://github.com/dixudx))
* Fixed encryption key and encryption provider rotation ([#58375](https://github.com/kubernetes/kubernetes/pull/58375), [@liggitt](https://github.com/liggitt))
* set fsGroup by securityContext.fsGroup in azure file ([#58316](https://github.com/kubernetes/kubernetes/pull/58316), [@andyzhangx](https://github.com/andyzhangx))
* Remove deprecated and unmaintained salt support. kubernetes-salt.tar.gz will no longer be published in the release tarball. ([#58248](https://github.com/kubernetes/kubernetes/pull/58248), [@mikedanese](https://github.com/mikedanese))
* Detach and clear bad disk URI ([#58345](https://github.com/kubernetes/kubernetes/pull/58345), [@rootfs](https://github.com/rootfs))
* Allow version arg in kubeadm upgrade apply to be optional if config file already have version info ([#53220](https://github.com/kubernetes/kubernetes/pull/53220), [@medinatiger](https://github.com/medinatiger))
* feat(fakeclient): push event on watched channel on add/update/delete ([#57504](https://github.com/kubernetes/kubernetes/pull/57504), [@yue9944882](https://github.com/yue9944882))
* Custom resources can now be submitted to and received from the API server in application/yaml format, consistent with other API resources. ([#58260](https://github.com/kubernetes/kubernetes/pull/58260), [@liggitt](https://github.com/liggitt))
* remove spaces from kubectl describe hpa ([#56331](https://github.com/kubernetes/kubernetes/pull/56331), [@shiywang](https://github.com/shiywang))
* fluentd-gcp updated to version 2.0.14. ([#58224](https://github.com/kubernetes/kubernetes/pull/58224), [@zombiezen](https://github.com/zombiezen))
* Instrument the Azure cloud provider for Prometheus monitoring. ([#58204](https://github.com/kubernetes/kubernetes/pull/58204), [@cosmincojocar](https://github.com/cosmincojocar))
* -Add scheduler optimization options, short circuit all predicates if … ([#56926](https://github.com/kubernetes/kubernetes/pull/56926), [@wgliang](https://github.com/wgliang))
* Remove deprecated ContainerVM support from GCE kube-up.  ([#58247](https://github.com/kubernetes/kubernetes/pull/58247), [@mikedanese](https://github.com/mikedanese))
* Remove deprecated kube-push.sh functionality.  ([#58246](https://github.com/kubernetes/kubernetes/pull/58246), [@mikedanese](https://github.com/mikedanese))
* The getSubnetIDForLB() should return subnet id rather than net id. ([#58208](https://github.com/kubernetes/kubernetes/pull/58208), [@FengyunPan](https://github.com/FengyunPan))
* Avoid panic when failing to allocate a Cloud CIDR (aka GCE Alias IP Range).  ([#58186](https://github.com/kubernetes/kubernetes/pull/58186), [@negz](https://github.com/negz))
* Handle Unhealthy devices ([#57266](https://github.com/kubernetes/kubernetes/pull/57266), [@vikaschoudhary16](https://github.com/vikaschoudhary16))
* Expose Metrics Server metrics via /metric endpoint. ([#57456](https://github.com/kubernetes/kubernetes/pull/57456), [@kawych](https://github.com/kawych))
* Remove deprecated container-linux support in gce kube-up.sh.  ([#58098](https://github.com/kubernetes/kubernetes/pull/58098), [@mikedanese](https://github.com/mikedanese))
* openstack cinder detach problem is fixed if nova is shutdowned ([#56846](https://github.com/kubernetes/kubernetes/pull/56846), [@zetaab](https://github.com/zetaab))
* Fixes a possible deadlock preventing quota from being recalculated ([#58107](https://github.com/kubernetes/kubernetes/pull/58107), [@ironcladlou](https://github.com/ironcladlou))
* fluentd-es addon: multiline stacktraces are now grouped into one entry automatically ([#58063](https://github.com/kubernetes/kubernetes/pull/58063), [@monotek](https://github.com/monotek))
* GCE: Allows existing internal load balancers to continue using an outdated subnetwork  ([#57861](https://github.com/kubernetes/kubernetes/pull/57861), [@nicksardo](https://github.com/nicksardo))
* ignore images in used by running containers when GC ([#57020](https://github.com/kubernetes/kubernetes/pull/57020), [@dixudx](https://github.com/dixudx))
* Remove deprecated and unmaintained photon-controller kube-up.sh.  ([#58096](https://github.com/kubernetes/kubernetes/pull/58096), [@mikedanese](https://github.com/mikedanese))
* The kubelet flag to run docker containers with a process namespace that is shared between all containers in a pod is now deprecated and will be replaced by a new field in `v1.Pod` that configures this behavior. ([#58093](https://github.com/kubernetes/kubernetes/pull/58093), [@verb](https://github.com/verb))
* fix device name change issue for azure disk: add remount logic ([#57953](https://github.com/kubernetes/kubernetes/pull/57953), [@andyzhangx](https://github.com/andyzhangx))
* The Kubelet now explicitly registers all of its command-line flags with an internal flagset, which prevents flags from third party libraries from unintentionally leaking into the Kubelet's command-line API. Many unintentionally leaked flags are now marked deprecated, so that users have a chance to migrate away from them before they are removed. One previously leaked flag, --cloud-provider-gce-lb-src-cidrs, was entirely removed from the Kubelet's command-line API, because it is irrelevant to Kubelet operation. ([#57613](https://github.com/kubernetes/kubernetes/pull/57613), [@mtaufen](https://github.com/mtaufen))
* Remove deprecated and unmaintained libvirt-coreos kube-up.sh.  ([#58023](https://github.com/kubernetes/kubernetes/pull/58023), [@mikedanese](https://github.com/mikedanese))
* Remove deprecated and unmaintained windows installer.  ([#58020](https://github.com/kubernetes/kubernetes/pull/58020), [@mikedanese](https://github.com/mikedanese))
* Remove deprecated and unmaintained openstack-heat kube-up.sh.  ([#58021](https://github.com/kubernetes/kubernetes/pull/58021), [@mikedanese](https://github.com/mikedanese))
* Fixes authentication problem faced during various vSphere operations. ([#57978](https://github.com/kubernetes/kubernetes/pull/57978), [@prashima](https://github.com/prashima))
* fluentd-gcp updated to version 2.0.13. ([#57789](https://github.com/kubernetes/kubernetes/pull/57789), [@x13n](https://github.com/x13n))
* Add support for cloud-controller-manager in local-up-cluster.sh ([#57757](https://github.com/kubernetes/kubernetes/pull/57757), [@dims](https://github.com/dims))
* Update CSI spec dependency to point to v0.1.0 tag ([#57989](https://github.com/kubernetes/kubernetes/pull/57989), [@NickrenREN](https://github.com/NickrenREN))
* Update kube-dns to Version 1.14.8 that includes only small changes to how Prometheus metrics are collected. ([#57918](https://github.com/kubernetes/kubernetes/pull/57918), [@rramkumar1](https://github.com/rramkumar1))
* Add proxy_read_timeout flag to kubeapi_load_balancer charm. ([#57926](https://github.com/kubernetes/kubernetes/pull/57926), [@wwwtyro](https://github.com/wwwtyro))
* Adding support for Block Volume type to rbd plugin. ([#56651](https://github.com/kubernetes/kubernetes/pull/56651), [@sbezverk](https://github.com/sbezverk))
* Fixes a bug in Heapster deployment for google sink. ([#57902](https://github.com/kubernetes/kubernetes/pull/57902), [@kawych](https://github.com/kawych))
* Forbid unnamed contexts in kubeconfigs. ([#56769](https://github.com/kubernetes/kubernetes/pull/56769), [@dixudx](https://github.com/dixudx))
* Upgrade to etcd client 3.2.13 and grpc 1.7.5 to improve HA etcd cluster stability. ([#57480](https://github.com/kubernetes/kubernetes/pull/57480), [@jpbetz](https://github.com/jpbetz))
* Default scheduler code is moved out of the plugin directory. ([#57852](https://github.com/kubernetes/kubernetes/pull/57852), [@misterikkit](https://github.com/misterikkit))
    * plugin/pkg/scheduler -> pkg/scheduler
    * plugin/cmd/kube-scheduler -> cmd/kube-scheduler
* Bump metadata proxy version to v0.1.7 to pick up security fix. ([#57762](https://github.com/kubernetes/kubernetes/pull/57762), [@ihmccreery](https://github.com/ihmccreery))
* HugePages feature is beta ([#56939](https://github.com/kubernetes/kubernetes/pull/56939), [@derekwaynecarr](https://github.com/derekwaynecarr))
* GCE: support passing kube-scheduler policy config via SCHEDULER_POLICY_CONFIG ([#57425](https://github.com/kubernetes/kubernetes/pull/57425), [@yguo0905](https://github.com/yguo0905))
* Returns an error for non overcommitable resources if they don't have limit field set in container spec. ([#57170](https://github.com/kubernetes/kubernetes/pull/57170), [@jiayingz](https://github.com/jiayingz))
* Update defaultbackend image to 1.4 and deployment apiVersion to apps/v1 ([#57866](https://github.com/kubernetes/kubernetes/pull/57866), [@zouyee](https://github.com/zouyee))
* kubeadm: set kube-apiserver advertise address using downward API ([#56084](https://github.com/kubernetes/kubernetes/pull/56084), [@andrewsykim](https://github.com/andrewsykim))
* CDK nginx ingress is now handled via a daemon set. ([#57530](https://github.com/kubernetes/kubernetes/pull/57530), [@hyperbolic2346](https://github.com/hyperbolic2346))
* The kubelet uses a new release 3.1 of the pause container with the Docker runtime. This version will clean up orphaned zombie processes that it inherits. ([#57517](https://github.com/kubernetes/kubernetes/pull/57517), [@verb](https://github.com/verb))
* Allow kubectl set image|env on a cronjob ([#57742](https://github.com/kubernetes/kubernetes/pull/57742), [@soltysh](https://github.com/soltysh))
* Move local PV negative scheduling tests to integration ([#57570](https://github.com/kubernetes/kubernetes/pull/57570), [@sbezverk](https://github.com/sbezverk))
* fix azure disk not available issue when device name changed ([#57549](https://github.com/kubernetes/kubernetes/pull/57549), [@andyzhangx](https://github.com/andyzhangx))
* Only create Privileged PSP binding during e2e tests if RBAC is enabled. ([#56382](https://github.com/kubernetes/kubernetes/pull/56382), [@mikkeloscar](https://github.com/mikkeloscar))
* RBAC: The system:kubelet-api-admin cluster role can be used to grant full access to the kubelet API ([#57128](https://github.com/kubernetes/kubernetes/pull/57128), [@liggitt](https://github.com/liggitt))
* Allow kubernetes components to react to SIGTERM signal and shutdown gracefully. ([#57756](https://github.com/kubernetes/kubernetes/pull/57756), [@mborsz](https://github.com/mborsz))
* ignore nonexistent ns net file error when deleting container network in case a retry ([#57697](https://github.com/kubernetes/kubernetes/pull/57697), [@dixudx](https://github.com/dixudx))
* check psp HostNetwork in DenyEscalatingExec admission controller. ([#56839](https://github.com/kubernetes/kubernetes/pull/56839), [@hzxuzhonghu](https://github.com/hzxuzhonghu))
* The alpha `--init-config-dir` flag has been removed. Instead, use the `--config` flag to reference a kubelet configuration file directly. ([#57624](https://github.com/kubernetes/kubernetes/pull/57624), [@mtaufen](https://github.com/mtaufen))
* Add cache for VM get operation in azure cloud provider ([#57432](https://github.com/kubernetes/kubernetes/pull/57432), [@karataliu](https://github.com/karataliu))
* Fix garbage collection when the controller-manager uses --leader-elect=false ([#57340](https://github.com/kubernetes/kubernetes/pull/57340), [@jmcmeek](https://github.com/jmcmeek))
* iSCSI sessions managed by kubernetes will now explicitly set startup.mode to 'manual' to ([#57475](https://github.com/kubernetes/kubernetes/pull/57475), [@stmcginnis](https://github.com/stmcginnis))
    * prevent automatic login after node failure recovery. This is the default open-iscsi mode, so
    * this change will only impact users who have changed their startup.mode to be 'automatic'
    * in /etc/iscsi/iscsid.conf.
* Configurable liveness probe initial delays for etcd and kube-apiserver in GCE ([#57749](https://github.com/kubernetes/kubernetes/pull/57749), [@wojtek-t](https://github.com/wojtek-t))
* Fixed garbage collection hang ([#57503](https://github.com/kubernetes/kubernetes/pull/57503), [@liggitt](https://github.com/liggitt))
* Fixes controller manager crash in certain vSphere cloud provider environment. ([#57286](https://github.com/kubernetes/kubernetes/pull/57286), [@rohitjogvmw](https://github.com/rohitjogvmw))
* Remove useInstanceMetadata parameter from Azure cloud provider. ([#57647](https://github.com/kubernetes/kubernetes/pull/57647), [@feiskyer](https://github.com/feiskyer))
* Support multiple scale sets in Azure cloud provider. ([#57543](https://github.com/kubernetes/kubernetes/pull/57543), [@feiskyer](https://github.com/feiskyer))
* GCE: Fixes ILB creation on automatic networks with manually created subnetworks. ([#57351](https://github.com/kubernetes/kubernetes/pull/57351), [@nicksardo](https://github.com/nicksardo))
* Improve scheduler performance of MatchInterPodAffinity predicate. ([#57476](https://github.com/kubernetes/kubernetes/pull/57476), [@misterikkit](https://github.com/misterikkit))
* Improve scheduler performance of MatchInterPodAffinity predicate. ([#57477](https://github.com/kubernetes/kubernetes/pull/57477), [@misterikkit](https://github.com/misterikkit))
* Improve scheduler performance of MatchInterPodAffinity predicate. ([#57478](https://github.com/kubernetes/kubernetes/pull/57478), [@misterikkit](https://github.com/misterikkit))
* Allow use resource ID to specify public IP address in azure_loadbalancer ([#53557](https://github.com/kubernetes/kubernetes/pull/53557), [@yolo3301](https://github.com/yolo3301))
* Fixes a bug where if an error was returned that was not an `autorest.DetailedError` we would return `"not found", nil` which caused nodes to go to `NotReady` state. ([#57484](https://github.com/kubernetes/kubernetes/pull/57484), [@brendandburns](https://github.com/brendandburns))
* Add the path '/version/' to the `system:discovery` cluster role. ([#57368](https://github.com/kubernetes/kubernetes/pull/57368), [@brendandburns](https://github.com/brendandburns))
* Fixes issue creating docker secrets with kubectl 1.9 for accessing docker private registries. ([#57463](https://github.com/kubernetes/kubernetes/pull/57463), [@dims](https://github.com/dims))
* adding predicates ordering for the kubernetes scheduler. ([#57168](https://github.com/kubernetes/kubernetes/pull/57168), [@yastij](https://github.com/yastij))
* Free up CPU and memory requested but unused by Metrics Server Pod Nanny. ([#57252](https://github.com/kubernetes/kubernetes/pull/57252), [@kawych](https://github.com/kawych))
* The alpha Accelerators feature gate is deprecated and will be removed in v1.11. Please use device plugins instead. They can be enabled using the DevicePlugins feature gate. ([#57384](https://github.com/kubernetes/kubernetes/pull/57384), [@mindprince](https://github.com/mindprince))
* Fixed dynamic provisioning of GCE PDs to round to the next GB instead of GiB ([#56600](https://github.com/kubernetes/kubernetes/pull/56600), [@edisonxiang](https://github.com/edisonxiang))
* Separate loop and plugin control ([#52371](https://github.com/kubernetes/kubernetes/pull/52371), [@cheftako](https://github.com/cheftako))
* Use old dns-ip mechanism with older cdk-addons. ([#57403](https://github.com/kubernetes/kubernetes/pull/57403), [@wwwtyro](https://github.com/wwwtyro))
* Retry 'connection refused' errors when setting up clusters on GCE. ([#57394](https://github.com/kubernetes/kubernetes/pull/57394), [@mborsz](https://github.com/mborsz))
* Upgrade to etcd client 3.2.11 and grpc 1.7.5 to improve HA etcd cluster stability. ([#57160](https://github.com/kubernetes/kubernetes/pull/57160), [@jpbetz](https://github.com/jpbetz))
* Added the ability to select pods in a chosen node to be drained, based on given pod label-selector ([#56864](https://github.com/kubernetes/kubernetes/pull/56864), [@juanvallejo](https://github.com/juanvallejo))
* Wait for kubedns to be ready when collecting the cluster IP. ([#57337](https://github.com/kubernetes/kubernetes/pull/57337), [@wwwtyro](https://github.com/wwwtyro))
* Use "k8s.gcr.io" for container images rather than "gcr.io/google_containers".  This is just a redirect, for now, so should not impact anyone materially.   ([#54174](https://github.com/kubernetes/kubernetes/pull/54174), [@thockin](https://github.com/thockin))
    * Documentation and tools should all convert to the new name. Users should take note of this in case they see this new name in the system.
* Fix ipvs proxier nodeport eth* assumption ([#56685](https://github.com/kubernetes/kubernetes/pull/56685), [@m1093782566](https://github.com/m1093782566))



# v1.10.0-alpha.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

## Downloads for v1.10.0-alpha.1


filename | sha256 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes.tar.gz) | `403b90bfa32f7669b326045a629bd15941c533addcaf0c49d3c3c561da0542f2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-src.tar.gz) | `266da065e9eddf19d36df5ad325f2f854101a0e712766148e87d998e789b80cf`

### Client Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `5aaa8e294ae4060d34828239e37f37b45fa5a69508374be668965102848626be`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `40a8e3bab11b88a2bb8e748f0b29da806d89b55775508039abe9c38c5f4ab97d`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `e08dde0b561529f0b2bb39c141f4d7b1c943749ef7c1f9779facf5fb5b385d6a`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `76a05d31acaab932ef45c67e1d6c9273933b8bc06dd5ce9bad3c7345d5267702`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `4b833c9e80f3e4ac4958ea0ffb5ae564b31d2a524f6a14e58802937b2b936d73`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `f1484ab75010a2258ed7717b1284d0c139d17e194ac9e391b8f1c0999eec3c2d`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `da884f09ec753925b2c1f27ea0a1f6c3da2056855fc88f47929bb3d6c2a09312`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `c486f760c6707fc92d1659d3cbe33d68c03190760b73ac215957ee52f9c19195`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `514c550b7ff85ac33e6ed333bcc06461651fe4004d8b7c12ca67f5dc1d2198bf`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `ddad59222f6a8cb4e88c4330c2a967c4126cb22ac5e0d7126f9f65cca0fb9f45`

### Server Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `514efd798ce1d7fe4233127f3334a3238faad6c26372a2d457eff02cbe72d756`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `f71f75fb96221f65891fc3e04fd52ae4e5628da8b7b4fbedece3fab4cb650afa`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `a9d8c2386813fd690e60623a6ee1968fe8f0a1a8e13bc5cc12b2caf8e8a862e1`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `21336a5e40aead4e2ec7e744a99d72bf8cb552341f3141abf8f235beb250cd93`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `257e44d38fef83f08990b6b9b5e985118e867c0c33f0e869f0900397b9d30498`

### Node Binaries

filename | sha256 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `97bf1210f0595ebf496ca7b000c4367f8a459d97ef72459efc6d0e07a072398f`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `eebcd3c14fb4faeb82ab047a2152db528adc2d9f7b20eef6f5dc58202ebe3124`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `3d4428416c775a0a6463f623286bd2ecdf9240ce901e1fbae180dfb564c53ea1`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `5cc96b24fad0ac1779a66f9b136d90e975b07bf619fea905e6c26ac5a4c41168`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `134c13338edf4efcd511f4161742fbaa6dc232965d3d926c3de435e8a080fcbb`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.10.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `ae54bf2bbcb99cdcde959140460d0f83c0ecb187d060b594ae9c5349960ab055`

## Changelog since v1.9.0

### Action Required

* [action required] Remove the kubelet's `--cloud-provider=auto-detect` feature ([#56287](https://github.com/kubernetes/kubernetes/pull/56287), [@stewart-yu](https://github.com/stewart-yu))

### Other notable changes

* Fix Heapster configuration and Metrics Server configuration to enable overriding default resource requirements. ([#56965](https://github.com/kubernetes/kubernetes/pull/56965), [@kawych](https://github.com/kawych))
* YAMLDecoder Read now returns the number of bytes read ([#57000](https://github.com/kubernetes/kubernetes/pull/57000), [@sel](https://github.com/sel))
* Retry 'connection refused' errors when setting up clusters on GCE. ([#57324](https://github.com/kubernetes/kubernetes/pull/57324), [@mborsz](https://github.com/mborsz))
* Update kubeadm's minimum supported Kubernetes version in v1.10.x to v1.9.0 ([#57233](https://github.com/kubernetes/kubernetes/pull/57233), [@xiangpengzhao](https://github.com/xiangpengzhao))
* Graduate CPU Manager feature from alpha to beta. ([#55977](https://github.com/kubernetes/kubernetes/pull/55977), [@ConnorDoyle](https://github.com/ConnorDoyle))
* Drop hacks used for Mesos integration that was already removed from main kubernetes repository ([#56754](https://github.com/kubernetes/kubernetes/pull/56754), [@dims](https://github.com/dims))
* Compare correct file names for volume detach operation ([#57053](https://github.com/kubernetes/kubernetes/pull/57053), [@prashima](https://github.com/prashima))
* Improved event generation in volume mount, attach, and extend operations ([#56872](https://github.com/kubernetes/kubernetes/pull/56872), [@davidz627](https://github.com/davidz627))
* GCE: bump COS image version to cos-stable-63-10032-71-0 ([#57204](https://github.com/kubernetes/kubernetes/pull/57204), [@yujuhong](https://github.com/yujuhong))
* fluentd-gcp updated to version 2.0.11. ([#56927](https://github.com/kubernetes/kubernetes/pull/56927), [@x13n](https://github.com/x13n))
* calico-node addon tolerates all NoExecute and NoSchedule taints by default. ([#57122](https://github.com/kubernetes/kubernetes/pull/57122), [@caseydavenport](https://github.com/caseydavenport))
* Support LoadBalancer for Azure Virtual Machine Scale Sets ([#57131](https://github.com/kubernetes/kubernetes/pull/57131), [@feiskyer](https://github.com/feiskyer))
* Makes the kube-dns addon optional so that users can deploy their own DNS solution. ([#57113](https://github.com/kubernetes/kubernetes/pull/57113), [@wwwtyro](https://github.com/wwwtyro))
* Enabled log rotation for load balancer's api logs to prevent running out of disk space. ([#56979](https://github.com/kubernetes/kubernetes/pull/56979), [@hyperbolic2346](https://github.com/hyperbolic2346))
* Remove ScrubDNS interface from cloudprovider. ([#56955](https://github.com/kubernetes/kubernetes/pull/56955), [@feiskyer](https://github.com/feiskyer))
* Fix `etcd-version-monitor` to backward compatibly support etcd 3.1 [go-grpc-prometheus](https://github.com/grpc-ecosystem/go-grpc-prometheus) metrics format. ([#56871](https://github.com/kubernetes/kubernetes/pull/56871), [@jpbetz](https://github.com/jpbetz))
* enable flexvolume on Windows node ([#56921](https://github.com/kubernetes/kubernetes/pull/56921), [@andyzhangx](https://github.com/andyzhangx))
* When using Role-Based Access Control, the "admin", "edit", and "view" roles now have the expected permissions on NetworkPolicy resources. ([#56650](https://github.com/kubernetes/kubernetes/pull/56650), [@danwinship](https://github.com/danwinship))
* Fix the PersistentVolumeLabel controller from initializing the PV labels when it's not the next pending initializer. ([#56831](https://github.com/kubernetes/kubernetes/pull/56831), [@jhorwit2](https://github.com/jhorwit2))
* kube-apiserver: The external hostname no longer longer use the cloud provider API to select a default. It can be set explicitly using --external-hostname, if needed. ([#56812](https://github.com/kubernetes/kubernetes/pull/56812), [@dims](https://github.com/dims))
* Use GiB unit for creating and resizing volumes for Glusterfs ([#56581](https://github.com/kubernetes/kubernetes/pull/56581), [@gnufied](https://github.com/gnufied))
* PersistentVolume flexVolume sources can now reference secrets in a namespace other than the PersistentVolumeClaim's namespace. ([#56460](https://github.com/kubernetes/kubernetes/pull/56460), [@liggitt](https://github.com/liggitt))
* Scheduler skips pods that use a PVC that either does not exist or is being deleted. ([#55957](https://github.com/kubernetes/kubernetes/pull/55957), [@jsafrane](https://github.com/jsafrane))
* Fixed a garbage collection race condition where objects with ownerRefs pointing to cluster-scoped objects could be deleted incorrectly. ([#57211](https://github.com/kubernetes/kubernetes/pull/57211), [@liggitt](https://github.com/liggitt))
* Kubectl explain now prints out the Kind and API version of the resource being explained ([#55689](https://github.com/kubernetes/kubernetes/pull/55689), [@luksa](https://github.com/luksa))
* api-server provides specific events when unable to repair a service cluster ip or node port ([#54304](https://github.com/kubernetes/kubernetes/pull/54304), [@frodenas](https://github.com/frodenas))
* Added docker-logins config to kubernetes-worker charm ([#56217](https://github.com/kubernetes/kubernetes/pull/56217), [@Cynerva](https://github.com/Cynerva))
* delete useless params containerized ([#56146](https://github.com/kubernetes/kubernetes/pull/56146), [@jiulongzaitian](https://github.com/jiulongzaitian))
* add mount options support for azure disk ([#56147](https://github.com/kubernetes/kubernetes/pull/56147), [@andyzhangx](https://github.com/andyzhangx))
* Use structured generator for kubectl autoscale  ([#55913](https://github.com/kubernetes/kubernetes/pull/55913), [@wackxu](https://github.com/wackxu))
* K8s supports cephfs fuse mount. ([#55866](https://github.com/kubernetes/kubernetes/pull/55866), [@zhangxiaoyu-zidif](https://github.com/zhangxiaoyu-zidif))
* COS: Keep the docker network checkpoint ([#54805](https://github.com/kubernetes/kubernetes/pull/54805), [@yujuhong](https://github.com/yujuhong))
* Fixed documentation typo in IPVS README. ([#56578](https://github.com/kubernetes/kubernetes/pull/56578), [@shift](https://github.com/shift))

