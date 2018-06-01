<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.11.0-beta.1](#v1110-beta1)
  - [Downloads for v1.11.0-beta.1](#downloads-for-v1110-beta1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.11.0-alpha.2](#changelog-since-v1110-alpha2)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
- [v1.11.0-alpha.2](#v1110-alpha2)
  - [Downloads for v1.11.0-alpha.2](#downloads-for-v1110-alpha2)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.11.0-alpha.1](#changelog-since-v1110-alpha1)
    - [Other notable changes](#other-notable-changes-1)
- [v1.11.0-alpha.1](#v1110-alpha1)
  - [Downloads for v1.11.0-alpha.1](#downloads-for-v1110-alpha1)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.10.0](#changelog-since-v1100)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-2)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.11.0-beta.1

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/release-1.11/examples)

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
* [action required] kubeadm: The `:Etcd` struct has been refactored in the v1alpha2 API. All the options now reside under either `.Etcd.Local` or `.Etcd.External`. Automatic conversions from the v1alpha1 API are supported. ([#64066](https://github.com/kubernetes/kubernetes/pull/64066), [@luxas](https://github.com/luxas))
* [action required] kubeadm: kubelets in kubeadm clusters now disable the readonly port (10255). If you're relying on unauthenticated access to the readonly port, please switch to using the secure port (10250). Instead, you can now use ServiceAccount tokens when talking to the secure port, which will make it easier to get access to e.g. the `/metrics` endpoint of the kubelet securely. ([#64187](https://github.com/kubernetes/kubernetes/pull/64187), [@luxas](https://github.com/luxas))
* [action required] kubeadm: Support for `.AuthorizationModes` in the kubeadm v1alpha2 API has been removed. Instead, you can use the `.APIServerExtraArgs` and `.APIServerExtraVolumes` fields to achieve the same effect. Files using the v1alpha1 API and setting this field will be automatically upgraded to this v1alpha2 API and the information will be preserved. ([#64068](https://github.com/kubernetes/kubernetes/pull/64068), [@luxas](https://github.com/luxas))
* [action required] The formerly publicly-available cAdvisor web UI that the kubelet ran on port 4194 by default is now turned off by default. The flag configuring what port to run this UI on `--cadvisor-port` was deprecated in v1.10. Now the default is `--cadvisor-port=0`, in other words, to not run the web server. The recommended way to run cAdvisor if you still need it, is via a DaemonSet. The `--cadvisor-port` will be removed in v1.12 ([#63881](https://github.com/kubernetes/kubernetes/pull/63881), [@luxas](https://github.com/luxas))
* [action required] kubeadm: The `.ImagePullPolicy` field has been removed in the v1alpha2 API version. Instead it's set statically to `IfNotPresent` for all required images. If you want to always pull the latest images before cluster init (like what `Always` would do), run `kubeadm config images pull` before each `kubeadm init`. If you don't want the kubelet to pull any images at `kubeadm init` time, as you for instance don't have an internet connection, you can also run `kubeadm config images pull` before `kubeadm init` or side-load the images some other way (e.g. `docker load -i image.tar`). Having the images locally cached will result in no pull at runtime, which makes it possible to run without any internet connection. ([#64096](https://github.com/kubernetes/kubernetes/pull/64096), [@luxas](https://github.com/luxas))
* [action required] In the new v1alpha2 kubeadm Configuration API, the `.CloudProvider` and `.PrivilegedPods` fields don't exist anymore. ([#63866](https://github.com/kubernetes/kubernetes/pull/63866), [@luxas](https://github.com/luxas))
    * Instead, you should use the out-of-tree cloud provider implementations which are beta in v1.11.
    * If you have to use the legacy in-tree cloud providers, you can rearrange your config like the example below. In case you need the `cloud-config` file (located in `{cloud-config-path}`), you can mount it into the API Server and controller-manager containers using ExtraVolumes like the example below.
    * If you need to use the `.PrivilegedPods` functionality, you can still edit the manifests in
    * `/etc/kubernetes/manifests/`, and set `.SecurityContext.Privileged=true` for the apiserver
    * and controller manager.
    * ---
    * kind: MasterConfiguration
    * apiVersion: kubeadm.k8s.io/v1alpha2
    * apiServerExtraArgs:
    *   cloud-provider: "{cloud}"
    *   cloud-config: "{cloud-config-path}"
    * apiServerExtraVolumes:
    * - name: cloud
    *   hostPath: "{cloud-config-path}"
    *   mountPath: "{cloud-config-path}"
    * controllerManagerExtraArgs:
    *   cloud-provider: "{cloud}"
    *   cloud-config: "{cloud-config-path}"
    * controllerManagerExtraVolumes:
    * - name: cloud
    *   hostPath: "{cloud-config-path}"
    *   mountPath: "{cloud-config-path}"
    * ---
* [action required] kubeadm now uses an upgraded API version for the configuration file, `kubeadm.k8s.io/v1alpha2`. kubeadm in v1.11 will still be able to read `v1alpha1` configuration, and will automatically convert the configuration to `v1alpha2` internally and when storing the configuration in the ConfigMap in the cluster. ([#63788](https://github.com/kubernetes/kubernetes/pull/63788), [@luxas](https://github.com/luxas))
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

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

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

[Documentation](https://docs.k8s.io) & [Examples](https://releases.k8s.io/master/examples)

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
    * - Fix a bug in DNS resolution for externalName services
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
    * - Fix for kube-dns returns NXDOMAIN when not yet synced with apiserver.
    * - Don't generate empty record for externalName service.
    * - Add validation for upstreamNameserver port.
    * - Update go version to 1.9.3.
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

