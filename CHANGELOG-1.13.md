<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.13.0-alpha.1](#v1130-alpha1)
  - [Downloads for v1.13.0-alpha.1](#downloads-for-v1130-alpha1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.12.0](#changelog-since-v1120)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.13.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.13.0-alpha.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes.tar.gz) | `9f8a34b54a22ea4d7925c2f8d0e0cb2e2005486b1ed89e594bc0100ec7202fc247b89c5cbde5dc50c1f9d9f27e4f92aa0ca71fdffb9d079f63751bb1859d5bb4`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-src.tar.gz) | `a27a7c254d3677c823bd6fd1d0d5f9b1e78ccf807837173669a0079b0812a23444d646d80c2433c167ae50bf1a0e2a4b1d7cbd7457a505fc666464b069bd1e5f`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `d77d33c6d6357b99089f65e1c9ec3cabdcf526ec56e87bdee6b09a8c1b1f1b8f6f0ed6d32f2d3b352391da848afc945e5bb6bfc4c05d90fb4ba429e2d2c3ec0f`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `5b4a586defa2ba0ea7c8893dedfe48cae52a2cd324bcb311a3877e27493abb6cb76550e8201a9cac488cde9f83e0d30e6569b95641e8098fd9ec5df9c9e027b2`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `d50572fbb716393004ad2984a15043d2dfadedd16ae03a73fc85653266ae389071fd2c993923fbe9ea7fbd6b8cbeb6680ef147245e20f334969184d4b571509b`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `12ab709e574228f170a2ee2686e18dcbfcf59f64599b2ab9047c2ed63f4bd23d6c9fc48104431c9fa616e0ba30041e1c44fb3994ab54c5c98c0c4a94c5ea4b80`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `3a8c75b62cf9e6476417246d4aaeda5a13b74bc073444fc3649198b9d5dc1e7a62aa6b914c7da5a42bcd6164a8f63aa8b256b3579979b6465afe5aa5533ce501`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `0f5b5956850f11a826d59d226b6a22645ca1f63893cd33c17dfe004bd316f2704d800beb0b9c91a204efc125241825243c9e89b86c01e523bd07636ef925772e`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `06c60dd2e4e8d1ab45474a5b85345b4f644d0c1c66e167596c6c91bd607f957b68121fdb7efed362cb6799e7bcf14752b01f8bea0c929deb85311180f11469be`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `4630e9e523beb02d8d3900c71b3306561c2d119d588399c93d578184eb1a53601ceefe15a600740c13d565eaf24a679f17e371e2b19a70f77bbceab84acb58b3`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `0c0fcc9c492aceb00ff7fd3c10ba228c7bb10d6139b75ceecd8f85532797c5dc1162b39d94ebae5fa6b4c26f2b2a81630371617426f7537d9d11456943c7d50c`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `3548a6d8618c6c7c8042ae8c3eb69654314392c46f839de24ab72d9faa79993a6cf989f6ac619e418e817300081742c9928c8c2dd82cffc74f7c0e532e42288d`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `9dbf2343ef9539b7d4d73949bcd9eef6f46ece59e97fa3390a0e695d0cb2eacbbbae17e3ed53432a8018f55e6db2421a58739aacbc163776d8b2fe774ad62c34`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `a985f3c302246df9bff4b927a2596d209c19fb2f245aa5cb5de189b6a9d247d6fd0234edd45968a691f1a2714a0b72ffba2df1aebf361ba1a3461ab7e5fda2ff`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `80d20df07e6a29b7aedccbd4e26c1c0565b2a1c3146e1a5bb2ebd2e8cf9ab063db137389a498fd6a6c3c42da43486186af6f65fba399b332e4cae134badd7ab0`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `7d45ed3aa8b36e9e666b334ff3ed3de238caea34b4a92b5e1a61a6e7223ae8581bafff43a5b72447a43e118ad4b2c5c15aa0d6faab9a6a72a8fcf99abf340a1e`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `30698478fab2fe7daccac97917b0b21b018c194ec39b005728f8cddf77f889aa3e1a520e0d1d681f9d8b7889a887aa6eb98c33cb04a8bf52b9a40bb8589aa34f`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `4497d14ac81677b43f0b75a457890c1f3bb8745a39875f58d53c734bec1947c37388228e0c952ca87f22d74af101a9263546db07bf1a021d59cba3cea1d5e5b9`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `a3b0357db50e0dec7b0474816fec287388adabc76cc309a40dee9bc73771c951e4526a06145a8b332d72e5999dabb5e467d00d7c47035559a79e56f863def2f7`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `43af8ec4c5f2a1e2baa8cd13817e127fb6a3576dd811a30c4cc5f04d8a9a8bb2267eb5c42e0a895cf2ec0e3260b73c818249296c957625f4048e13102606680a`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `840354219b3e59ed05b5b44cbbf4d45ccc4c0d74044e28c8a557ca75d12e509b091eb10d9bed81e300cb484b0a0f735424383c8b266fbb3e2aa7a2d50dceaf9e`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `796ca2e6855bd942a9a63d93f847ae62c5ee74195e041b60b89ee7d0e5a75643a8809bfaa36898daa176bccf140a8e5e858f5cb74e457d8bbb0a650600628ceb`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.13.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `96d666e8446d09088bdcb440559035118dce07a2d9f5718856192fd807b61840d2d6cee2a808eb13f2e947784300edc1ac41ade14dc3536c044e4d08982f0fdd`

## Changelog since v1.12.0

### Action Required

* kube-apiserver: the deprecated `--etcd-quorum-read` flag has been removed, and quorum reads are always enabled when fetching data from etcd. ([#69527](https://github.com/kubernetes/kubernetes/pull/69527), [@liggitt](https://github.com/liggitt))
* Moved staging/src/k8s.io/client-go/tools/bootstrap to staging/src/k8s… ([#67356](https://github.com/kubernetes/kubernetes/pull/67356), [@yliaog](https://github.com/yliaog))
* [action required] kubeadm: The `v1alpha2` config API has been removed. ([#69055](https://github.com/kubernetes/kubernetes/pull/69055), [@fabriziopandini](https://github.com/fabriziopandini))
    * Please convert your `v1alpha2` configuration files to `v1alpha3` using the
    * `kubeadm config migrate` command of kubeadm v1.12.x

### Other notable changes

* Refactor factory_test.go to use a fake k8s client. ([#69412](https://github.com/kubernetes/kubernetes/pull/69412), [@tossmilestone](https://github.com/tossmilestone))
* kubeadm: fix a case where fetching a kubernetesVersion from the internet still happened even if some commands don't need it. ([#69645](https://github.com/kubernetes/kubernetes/pull/69645), [@neolit123](https://github.com/neolit123))
* Add tolerations for Stackdriver Logging and Metadata Agents. ([#69737](https://github.com/kubernetes/kubernetes/pull/69737), [@qingling128](https://github.com/qingling128))
* Fix a bug in the scheduler that could cause the scheduler to go to an infinite loop when all nodes in a zone are removed. ([#69758](https://github.com/kubernetes/kubernetes/pull/69758), [@bsalamat](https://github.com/bsalamat))
* Dry-run is promoted to Beta and will be enabled by default. ([#69644](https://github.com/kubernetes/kubernetes/pull/69644), [@apelisse](https://github.com/apelisse))
* `kubectl get priorityclass` now prints value column by default. ([#69431](https://github.com/kubernetes/kubernetes/pull/69431), [@Huang-Wei](https://github.com/Huang-Wei))
* Added a new container based image for running e2e tests ([#69368](https://github.com/kubernetes/kubernetes/pull/69368), [@dims](https://github.com/dims))
* Remove the deprecated --google-json-key flag from kubelet. ([#69354](https://github.com/kubernetes/kubernetes/pull/69354), [@yujuhong](https://github.com/yujuhong))
* kube-apiserver: fixes `procMount` field incorrectly being marked as required in openapi schema ([#69694](https://github.com/kubernetes/kubernetes/pull/69694), [@jessfraz](https://github.com/jessfraz))
* The `LC_ALL` and `LC_MESSAGES` env vars can now be used to set desired locale for `kubectl` while keeping `LANG` unchanged. ([#69500](https://github.com/kubernetes/kubernetes/pull/69500), [@m1kola](https://github.com/m1kola))
* Add ability to control primary GID of containers through Pod Spec and PodSecurityPolicy ([#67802](https://github.com/kubernetes/kubernetes/pull/67802), [@krmayankk](https://github.com/krmayankk))
* NodeLifecycleController: Now node lease renewal is treated as the heartbeat signal from the node, in addition to NodeStatus Update. ([#69241](https://github.com/kubernetes/kubernetes/pull/69241), [@wangzhen127](https://github.com/wangzhen127))
* [GCE] Enable by default audit logging truncating backend. ([#68288](https://github.com/kubernetes/kubernetes/pull/68288), [@loburm](https://github.com/loburm))
* Enable insertId generation, and update Stackdriver Logging Agent image to 0.5-1.5.36-1-k8s. This help reduce log duplication and guarantee log order. ([#68920](https://github.com/kubernetes/kubernetes/pull/68920), [@qingling128](https://github.com/qingling128))
* Move NodeInfo utils into pkg/scheduler/cache. ([#69495](https://github.com/kubernetes/kubernetes/pull/69495), [@wgliang](https://github.com/wgliang))
* adds dynamic shared informers to write generic, non-generated controllers ([#69308](https://github.com/kubernetes/kubernetes/pull/69308), [@p0lyn0mial](https://github.com/p0lyn0mial))
* Move CacheComparer to pkg/scheduler/internal/cache/comparer. ([#69317](https://github.com/kubernetes/kubernetes/pull/69317), [@wgliang](https://github.com/wgliang))
* Updating OWNERS list for vSphere Cloud Provider. ([#69187](https://github.com/kubernetes/kubernetes/pull/69187), [@SandeepPissay](https://github.com/SandeepPissay))
* The default storage class annotation for the storage addons has been changed to use the GA variant ([#68345](https://github.com/kubernetes/kubernetes/pull/68345), [@smelchior](https://github.com/smelchior))
* Upgrade to etcd 3.3 client ([#69322](https://github.com/kubernetes/kubernetes/pull/69322), [@jpbetz](https://github.com/jpbetz))
* fix GetVolumeLimits log flushing issue ([#69558](https://github.com/kubernetes/kubernetes/pull/69558), [@andyzhangx](https://github.com/andyzhangx))
* It is now possible to use named ports in the `kubectl port-forward` command ([#69477](https://github.com/kubernetes/kubernetes/pull/69477), [@m1kola](https://github.com/m1kola))
* kubeadm: fix a possible scenario where kubeadm can pull much newer control-plane images ([#69301](https://github.com/kubernetes/kubernetes/pull/69301), [@neolit123](https://github.com/neolit123))
* test/e2e/e2e.test: ([#69105](https://github.com/kubernetes/kubernetes/pull/69105), [@pohly](https://github.com/pohly))
        * -viper-config can be used to set also the options defined by command line flags
        * the default config file is "e2e.yaml/toml/json/..." and the test starts when no such config is found (as before) but if -viper-config is used, the config file must exist
        * -viper-config can be used to select a file with full path, with or without file suffix
        * the csiImageVersion/Registry flags were renamed to storage.csi.imageVersion/Registry
* Move FakeCache to pkg/scheduler/internal/cache/fake. ([#69318](https://github.com/kubernetes/kubernetes/pull/69318), [@wgliang](https://github.com/wgliang))
* The "kubectl cp" command now supports path shortcuts (../) in remote paths. ([#65189](https://github.com/kubernetes/kubernetes/pull/65189), [@juanvallejo](https://github.com/juanvallejo))
* Fixed subpath in containerized kubelet. ([#69565](https://github.com/kubernetes/kubernetes/pull/69565), [@jsafrane](https://github.com/jsafrane))
* The runtimeHandler field on the RuntimeClass resource now accepts the empty string. ([#69550](https://github.com/kubernetes/kubernetes/pull/69550), [@tallclair](https://github.com/tallclair))
* Kubelet can now parse PEM file containing both TLS certificate and key in arbitrary order. Previously key was always required to be first. ([#69536](https://github.com/kubernetes/kubernetes/pull/69536), [@awly](https://github.com/awly))
* Scheduling conformance tests related to daemonsets should set the annotation that relaxes node selection restrictions, if any are set. This ensures conformance tests can run on a wider array of clusters. ([#68793](https://github.com/kubernetes/kubernetes/pull/68793), [@aveshagarwal](https://github.com/aveshagarwal))
* Replace Parallelize with function ParallelizeUntil and formally deprecate the Parallelize. ([#68403](https://github.com/kubernetes/kubernetes/pull/68403), [@wgliang](https://github.com/wgliang))
* Move scheduler cache interface and implementation to pkg/scheduler/internal/cache. ([#68968](https://github.com/kubernetes/kubernetes/pull/68968), [@wgliang](https://github.com/wgliang))
* Update to use go1.11.1 ([#69386](https://github.com/kubernetes/kubernetes/pull/69386), [@cblecker](https://github.com/cblecker))
* Any external provider should be aware the cloud-provider interface should be imported from :- ([#68310](https://github.com/kubernetes/kubernetes/pull/68310), [@cheftako](https://github.com/cheftako))
    * cloudprovider "k8s.io/cloud-provider"
* kubeadm: Fix a crash if the etcd local alpha phase is called when the configuration contains an external etcd cluster ([#69420](https://github.com/kubernetes/kubernetes/pull/69420), [@ereslibre](https://github.com/ereslibre))
* kubeadm now allows mixing of init/cluster and join configuration in a single YAML file (although a warning gets printed in this case).  ([#69426](https://github.com/kubernetes/kubernetes/pull/69426), [@rosti](https://github.com/rosti))
* Code-gen: Remove lowercasing for project imports ([#68484](https://github.com/kubernetes/kubernetes/pull/68484), [@jsturtevant](https://github.com/jsturtevant))
* Fix client cert setup in delegating authentication logic ([#69430](https://github.com/kubernetes/kubernetes/pull/69430), [@DirectXMan12](https://github.com/DirectXMan12))
* service.beta.kubernetes.io/aws-load-balancer-internal now supports true and false values, previously it only supported non-empty strings ([#69436](https://github.com/kubernetes/kubernetes/pull/69436), [@mcrute](https://github.com/mcrute))
* OpenAPI spec and API reference now reflect dryRun query parameter for POST/PUT/PATCH operations ([#69359](https://github.com/kubernetes/kubernetes/pull/69359), [@roycaihw](https://github.com/roycaihw))
* kubeadm: Add a `v1beta1` API. ([#69289](https://github.com/kubernetes/kubernetes/pull/69289), [@fabriziopandini](https://github.com/fabriziopandini))
* kube-apiserver has removed support for the `etcd2` storage backend (deprecated since v1.9). Existing clusters must migrate etcd v2 data to etcd v3 storage before upgrading to v1.13. ([#69310](https://github.com/kubernetes/kubernetes/pull/69310), [@liggitt](https://github.com/liggitt))
* List operations against the API now return internal server errors instead of partially complete lists when a value cannot be transformed from storage. The updated behavior is consistent with all other operations that require transforming data from storage such as watch and get. ([#69399](https://github.com/kubernetes/kubernetes/pull/69399), [@mikedanese](https://github.com/mikedanese))
* `kubectl wait` now supports condition value checks other than true using `--for condition=available=false` ([#69295](https://github.com/kubernetes/kubernetes/pull/69295), [@deads2k](https://github.com/deads2k))
* CCM server will not listen insecurely if secure port is specified ([#68982](https://github.com/kubernetes/kubernetes/pull/68982), [@aruneli](https://github.com/aruneli))
* Bump cluster-proportional-autoscaler to 1.3.0 ([#69338](https://github.com/kubernetes/kubernetes/pull/69338), [@MrHohn](https://github.com/MrHohn))
    * - Rebase docker image on scratch.
* fix inconsistency in windows kernel proxy when updating HNS policy. ([#68923](https://github.com/kubernetes/kubernetes/pull/68923), [@delulu](https://github.com/delulu))
* Fixes the sample-apiserver so that its BanFlunder admission plugin can be used. ([#68417](https://github.com/kubernetes/kubernetes/pull/68417), [@MikeSpreitzer](https://github.com/MikeSpreitzer))
* Fixed CSIDriver API object to allow missing fields. ([#69331](https://github.com/kubernetes/kubernetes/pull/69331), [@jsafrane](https://github.com/jsafrane))
* Bump addon-manager to v8.8 ([#69337](https://github.com/kubernetes/kubernetes/pull/69337), [@MrHohn](https://github.com/MrHohn))
    * - Rebase docker image on debian-base:0.3.2.
* Update defaultbackend image to 1.5. Users should concentrate on updating scripts to the new version. ([#69120](https://github.com/kubernetes/kubernetes/pull/69120), [@aledbf](https://github.com/aledbf))
* Bump Dashboard version to v1.10.0 ([#68450](https://github.com/kubernetes/kubernetes/pull/68450), [@jeefy](https://github.com/jeefy))
* Fixed panic on iSCSI volume tear down. ([#69140](https://github.com/kubernetes/kubernetes/pull/69140), [@jsafrane](https://github.com/jsafrane))
* Update defaultbackend to v1.5 ([#69334](https://github.com/kubernetes/kubernetes/pull/69334), [@bowei](https://github.com/bowei))
* Remove unused chaosclient. ([#68409](https://github.com/kubernetes/kubernetes/pull/68409), [@wgliang](https://github.com/wgliang))
* Enable AttachVolumeLimit feature ([#69225](https://github.com/kubernetes/kubernetes/pull/69225), [@gnufied](https://github.com/gnufied))
* Update crictl to v1.12.0 ([#69033](https://github.com/kubernetes/kubernetes/pull/69033), [@feiskyer](https://github.com/feiskyer))
* Wait for pod failed event in subpath test. ([#69300](https://github.com/kubernetes/kubernetes/pull/69300), [@mrunalp](https://github.com/mrunalp))
* [GCP] Added env variables to control CPU requests of kube-controller-manager and kube-scheduler. ([#68823](https://github.com/kubernetes/kubernetes/pull/68823), [@loburm](https://github.com/loburm))
* Bump up pod short start timeout to 2 minutes. ([#69291](https://github.com/kubernetes/kubernetes/pull/69291), [@mrunalp](https://github.com/mrunalp))
* Use the mounted "/var/run/secrets/kubernetes.io/serviceaccount/token" as the token file for running in-cluster based e2e testing. ([#69273](https://github.com/kubernetes/kubernetes/pull/69273), [@dims](https://github.com/dims))
* apiservice availability related to networking glitches are corrected faster ([#68678](https://github.com/kubernetes/kubernetes/pull/68678), [@deads2k](https://github.com/deads2k))
* extract volume attachment status checking operation as a common function when attaching a CSI volume ([#68931](https://github.com/kubernetes/kubernetes/pull/68931), [@mlmhl](https://github.com/mlmhl))
* PodSecurityPolicy objects now support a `MayRunAs` rule for `fsGroup` and `supplementalGroups` options. This allows specifying ranges of allowed GIDs for pods/containers without forcing a default GID the way `MustRunAs` does. This means that a container to which such a policy applies to won't use any fsGroup/supplementalGroup GID if not explicitly specified, yet a specified GID must still fall in the GID range according to the policy. ([#65135](https://github.com/kubernetes/kubernetes/pull/65135), [@stlaz](https://github.com/stlaz))
* Images for cloud-controller-manager, kube-apiserver, kube-controller-manager, and kube-scheduler now contain a minimal /etc/nsswitch.conf and should respect /etc/hosts for lookups ([#69238](https://github.com/kubernetes/kubernetes/pull/69238), [@BenTheElder](https://github.com/BenTheElder))
* add deprecation warning for all cloud providers ([#69171](https://github.com/kubernetes/kubernetes/pull/69171), [@andrewsykim](https://github.com/andrewsykim))
* IPVS proxier mode now support connection based graceful termination. ([#66012](https://github.com/kubernetes/kubernetes/pull/66012), [@Lion-Wei](https://github.com/Lion-Wei))
* Fix panic in kubectl rollout commands ([#69150](https://github.com/kubernetes/kubernetes/pull/69150), [@soltysh](https://github.com/soltysh))
* Add fallbacks to ARM API when getting empty node IP from Azure IMDS ([#69077](https://github.com/kubernetes/kubernetes/pull/69077), [@feiskyer](https://github.com/feiskyer))
* Deduplicate PATH items when reading plugins. ([#69089](https://github.com/kubernetes/kubernetes/pull/69089), [@soltysh](https://github.com/soltysh))
* Adds permissions for startup of an on-cluster kube-controller-manager ([#69062](https://github.com/kubernetes/kubernetes/pull/69062), [@dghubble](https://github.com/dghubble))
* Fixes issue [[#68899](https://github.com/kubernetes/kubernetes/pull/68899)](https://github.com/kubernetes/kubernetes/issues/68899) where pods might schedule on an unschedulable node. ([#68984](https://github.com/kubernetes/kubernetes/pull/68984), [@k82cn](https://github.com/k82cn))
* Returns error if NodeGetInfo fails. ([#68979](https://github.com/kubernetes/kubernetes/pull/68979), [@xing-yang](https://github.com/xing-yang))
* Pod disruption budgets shouldn't be checked for terminal pods while evicting ([#68892](https://github.com/kubernetes/kubernetes/pull/68892), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
* Fix scheduler crashes when Prioritize Map function returns error. ([#68563](https://github.com/kubernetes/kubernetes/pull/68563), [@DylanBLE](https://github.com/DylanBLE))
* kubeadm: create control plane with ClusterFirstWithHostNet DNS policy ([#68890](https://github.com/kubernetes/kubernetes/pull/68890), [@andrewrynhard](https://github.com/andrewrynhard))
* Reduced excessive logging from fluentd-gcp-scaler. ([#68837](https://github.com/kubernetes/kubernetes/pull/68837), [@x13n](https://github.com/x13n))
* adds dynamic lister ([#68748](https://github.com/kubernetes/kubernetes/pull/68748), [@p0lyn0mial](https://github.com/p0lyn0mial))
* kubectl: add the --no-headers flag to `kubectl top ...` ([#67890](https://github.com/kubernetes/kubernetes/pull/67890), [@WanLinghao](https://github.com/WanLinghao))
* Restrict redirect following from the apiserver to same-host redirects, and ignore redirects in some cases. ([#66516](https://github.com/kubernetes/kubernetes/pull/66516), [@tallclair](https://github.com/tallclair))
* Fixed pod cleanup when /var/lib/kubelet is a symlink. ([#68741](https://github.com/kubernetes/kubernetes/pull/68741), [@jsafrane](https://github.com/jsafrane))
* Add "only_cpu_and_memory" GET parameter to /stats/summary http handler in kubelet. If parameter is true then only cpu and memory will be present in response. ([#67829](https://github.com/kubernetes/kubernetes/pull/67829), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* Start synchronizing pods after network is ready.  ([#68752](https://github.com/kubernetes/kubernetes/pull/68752), [@krzysztof-jastrzebski](https://github.com/krzysztof-jastrzebski))
* kubectl has gained new --profile and --profile-output options to output go profiles ([#68681](https://github.com/kubernetes/kubernetes/pull/68681), [@dlespiau](https://github.com/dlespiau))
* Provides FSGroup capability on FlexVolume driver. It allows to disable the VolumeOwnership operation when volume is mounted ([#68680](https://github.com/kubernetes/kubernetes/pull/68680), [@benoitf](https://github.com/benoitf))
* Apply _netdev mount option on bind mount ([#68626](https://github.com/kubernetes/kubernetes/pull/68626), [@gnufied](https://github.com/gnufied))
* fix UnmountDevice failure on Windows ([#68608](https://github.com/kubernetes/kubernetes/pull/68608), [@andyzhangx](https://github.com/andyzhangx))
* Allows changing nodeName in endpoint update. ([#68575](https://github.com/kubernetes/kubernetes/pull/68575), [@prameshj](https://github.com/prameshj))
* kube-apiserver would return 400 Bad Request when it couldn't decode a json patch. ([#68346](https://github.com/kubernetes/kubernetes/pull/68346), [@CaoShuFeng](https://github.com/CaoShuFeng))
    * kube-apiserver would return 422 Unprocessable Entity when a json patch couldn't be applied to one object.
* remove unused ReplicasetControllerOptions ([#68121](https://github.com/kubernetes/kubernetes/pull/68121), [@dixudx](https://github.com/dixudx))
* Pass signals to fluentd process ([#68064](https://github.com/kubernetes/kubernetes/pull/68064), [@gianrubio](https://github.com/gianrubio))
* Flex drivers by default do not produce metrics. Flex plugins can enable metrics collection by setting  the capability 'supportsMetrics' to true. Make sure the file system can support fs stat to produce metrics in this case. ([#67508](https://github.com/kubernetes/kubernetes/pull/67508), [@brahmaroutu](https://github.com/brahmaroutu))
* Use monotonically increasing generation to prevent scheduler equivalence cache race. ([#67308](https://github.com/kubernetes/kubernetes/pull/67308), [@cofyc](https://github.com/cofyc))
* Fix kubelet service file permission warning ([#66669](https://github.com/kubernetes/kubernetes/pull/66669), [@daixiang0](https://github.com/daixiang0))
* Add prometheus metric for scheduling throughput. ([#64526](https://github.com/kubernetes/kubernetes/pull/64526), [@misterikkit](https://github.com/misterikkit))
* Get public IP for Azure vmss nodes. ([#68498](https://github.com/kubernetes/kubernetes/pull/68498), [@feiskyer](https://github.com/feiskyer))
* test/integration: add a basic test for covering CronJobs ([#66937](https://github.com/kubernetes/kubernetes/pull/66937), [@mortent](https://github.com/mortent))
* Make service environment variables optional ([#68754](https://github.com/kubernetes/kubernetes/pull/68754), [@bradhoekstra](https://github.com/bradhoekstra))

