<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.16.0-alpha.3](#v1160-alpha3)
  - [Downloads for v1.16.0-alpha.3](#downloads-for-v1160-alpha3)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.16.0-alpha.2](#changelog-since-v1160-alpha2)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
- [v1.16.0-alpha.2](#v1160-alpha2)
  - [Downloads for v1.16.0-alpha.2](#downloads-for-v1160-alpha2)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.16.0-alpha.1](#changelog-since-v1160-alpha1)
    - [Action Required](#action-required-1)
    - [Other notable changes](#other-notable-changes-1)
- [v1.16.0-alpha.1](#v1160-alpha1)
  - [Downloads for v1.16.0-alpha.1](#downloads-for-v1160-alpha1)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.15.0](#changelog-since-v1150)
    - [Action Required](#action-required-2)
    - [Other notable changes](#other-notable-changes-2)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


# v1.16.0-alpha.3

[Documentation](https://docs.k8s.io)

## Downloads for v1.16.0-alpha.3


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes.tar.gz) | `82bc119f8d1e44518ab4f4bdefb96158b1a3634c003fe1bc8dcd62410189449fbd6736126409d39a6e2d211a036b4aa98baef3b3c6d9f7505e63430847d127c2`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-src.tar.gz) | `bbf330b887a5839e3d3219f5f4aa38f1c70eab64228077f846da80395193b2b402b60030741de14a9dd4de963662cfe694f6ab04035309e54dc48e6dddd5c05d`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-darwin-386.tar.gz) | `8d509bdc1ca62463cbb25548ec270792630f6a883f3194e5bdbbb3d6f8568b00f695e39950b7b01713f2f05f206c4d1df1959c6ee80f8a3e390eb94759d344b2`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | `1b00b3a478c210e3c3e6c346f5c4f7f43a00d5ef6acb8d9c1feaf26f913b9d4f97eb6db99bbf67953ef6399abe4fbb79324973c1744a6a8cd76067cb2aeed2ca`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-linux-386.tar.gz) | `82424207b4ef52c3722436eaaf86dbe5c93c6670fd09c2b04320251028fd1bb75724b4f490b6e8b443bd8e5f892ab64612cd22206119924dafde424bdee9348a`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | `57ba937e58755d3b7dfd19626fedb95718f9c1d44ac1c5b4c8c46d11ba0f8783f3611c7b946b563cac9a3cf104c35ba5605e5e76b48ba2a707d787a7f50f7027`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | `3a3601026e019b299a6f662b887ebe749f08782d7ed0d37a807c38a01c6ba19f23e2837c9fb886053ad6e236a329f58a11ee3ec4ba96a8729905ae78a7f6c58c`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | `4cdeb2e678c6b817a04f9f5d92c5c6df88e0f954550961813fca63af4501d04c08e3f4353dd8b6dce96e2ee197a4c688245f03c888417a436b3cf70abd4ba53a`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | `0cc7c8f7b48f5affb679352a94e42d8b4003b9ca6f8cbeaf315d2eceddd2e8446a58ba1d4a0df18e8f9c69d0d3b5a46f25b2e6a916e57975381e504d1a4daa1b`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | `9d8fa639f543e707dc65f24ce2f8c73a50c606ec7bc27d17840f45ac150d00b3b3f83de5e3b21f72b598acf08273e4b9a889f199f4ce1b1d239b28659e6cd131`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-windows-386.tar.gz) | `05bf6e696da680bb8feec4f411f342a9661b6165f4f0c72c069871983f199418c4d4fa1e034136bc8be41c5fecc9934a123906f2d5666c09a876db16ae8c11ad`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | `b2097bc851f5d3504e562f68161910098b46c66c726b92b092a040acda965fed01f45e7b9e513a4259c7a5ebd65d7aa3e3b711f4179139a935720d91216ef5c2`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | `721bd09b64e5c8f220332417089a772d9073c0dc5cdfa240984cfeb0d681b4a02620fb3ebf1b9f6a82a4dd3423f5831c259d4bad502dce87f145e0a08cb73ee9`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | `e7638ce4b88b4282f0a157593cfe809fa9cc9139ea7ebae4762ef5ac1dfaa516903a8acb34a45937eb94b2699e5d4c68c639cbe40cbed2a6b97681aeace9948e`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | `395566c4be3c2ca5b07e81221b3370bc7ccbef0879f96a9384650fcaf4f699f3b2744ba1d97ae42cc6c5d9e1a65a41a793a8b0c9e01a0a65f57c56b1420f8141`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | `90fcba066efd76d2f271a0eb26ed4d90483674d04f5e8cc39ec1e5b7f343311f2f1c40de386f35d3c69759628a1c7c075559c09b6c4542e42fbbe0daeb61a5fa`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | `b25014bcf4138722a710451f6e58ee57588b4d47fcceeda8f6866073c1cc08641082ec56e94b0c6d586c0835ce9b55d205d254436fc22a744b24d8c74e8e5cce`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | `6925a71096530f7114a68e755d07cb8ba714bc60b477360c85d76d7b71d3a3c0b78a650877d81aae35b308ded27c8207b5fe72d990abc43db3aa8a7d6d7f94f4`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | `073310e1ccf9a8af998d4c0402ae86bee4f253d2af233b0c45cea55902268c2fe7190a41a990b079e24536e9efa27b94249c3a9236531a166ba3ac06c0f26f92`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | `c55e9aecef906e56a6003f441a7d336846edb269aed1c7a31cf834b0730508706e73ea0ae135c1604b0697c9e2582480fbfba8ba105152698c240e324da0cbd2`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | `e89d72d27bb0a7f9133ef7310f455ba2b4c46e9852c43e0a981b68a413bcdd18de7168eb16d93cf87a5ada6a4958592d3be80c9be1e6895fa48e2f7fa70f188d`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | `6ef8a25f2f80a806672057dc030654345e87d269babe7cf166f7443e04c0b3a9bc1928cbcf5abef1f0f0fcd37f3a727f789887dbbdae62f9d1fd90a71ed26b39`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | `22fd1cea6e0150c06dbdc7249635bbf93c4297565d5a9d13e653f9365cd61a0b8306312efc806d267c47be81621016b114510a269c622cccc916ecff4d10f33c`

## Changelog since v1.16.0-alpha.2

### Action Required

* ACTION REQUIRED: ([#80676](https://github.com/kubernetes/kubernetes/pull/80676), [@fabriziopandini](https://github.com/fabriziopandini))
    * kubeadm now deletes the bootstrap-kubelet.conf file after TLS bootstrap
    * User relying on bootstrap-kubelet.conf should switch to kubelet.conf that contains node credentials

### Other notable changes

* Fixes validation of VolumeAttachment API objects created with inline volume sources. ([#80945](https://github.com/kubernetes/kubernetes/pull/80945), [@tedyu](https://github.com/tedyu))
* Azure disks of shared kind will no longer fail if they do not contain skuname or  ([#80837](https://github.com/kubernetes/kubernetes/pull/80837), [@rmweir](https://github.com/rmweir))
    *     storageaccounttype.
* kubeadm: fix "certificate-authority" files not being pre-loaded when using file discovery ([#80966](https://github.com/kubernetes/kubernetes/pull/80966), [@neolit123](https://github.com/neolit123))
* Errors from pod volume set up are now propagated as pod events. ([#80369](https://github.com/kubernetes/kubernetes/pull/80369), [@jsafrane](https://github.com/jsafrane))
* kubeadm: enable secure serving for the kube-scheduler ([#80951](https://github.com/kubernetes/kubernetes/pull/80951), [@neolit123](https://github.com/neolit123))
* Kubernetes client users may disable automatic compression when invoking Kubernetes APIs by setting the `DisableCompression` field on their rest.Config.  This is recommended when clients communicate primarily over high bandwidth / low latency networks where response compression does not improve end to end latency. ([#80919](https://github.com/kubernetes/kubernetes/pull/80919), [@smarterclayton](https://github.com/smarterclayton))
* kubectl get did not correctly count the number of binaryData keys when listing config maps. ([#80827](https://github.com/kubernetes/kubernetes/pull/80827), [@smarterclayton](https://github.com/smarterclayton))
* Implement "post-filter" extension point for scheduling framework ([#78097](https://github.com/kubernetes/kubernetes/pull/78097), [@draveness](https://github.com/draveness))
* Reduces GCE PD Node Attach Limits by 1 since the node boot disk is considered an attachable disk ([#80923](https://github.com/kubernetes/kubernetes/pull/80923), [@davidz627](https://github.com/davidz627))
* This PR fixes an error when using external etcd but storing etcd certificates in the same folder and with the same name used by kubeadm for local etcd certificates; for an older version of kubeadm, the workaround is to avoid file name used by kubeadm for local etcd. ([#80867](https://github.com/kubernetes/kubernetes/pull/80867), [@fabriziopandini](https://github.com/fabriziopandini))
* When specifying `--(kube|system)-reserved-cgroup`, with `--cgroup-driver=systemd`, it is now possible to use the fully qualified cgroupfs name (i.e. `/test-cgroup.slice`). ([#78793](https://github.com/kubernetes/kubernetes/pull/78793), [@mattjmcnaughton](https://github.com/mattjmcnaughton))
* kubeadm: treat non-fatal errors as warnings when doing reset ([#80862](https://github.com/kubernetes/kubernetes/pull/80862), [@drpaneas](https://github.com/drpaneas))
* kube-addon-manager has been updated to v9.0.2 to fix a bug in leader election (https://github.com/kubernetes/kubernetes/pull/80575) ([#80861](https://github.com/kubernetes/kubernetes/pull/80861), [@mborsz](https://github.com/mborsz))
* Determine system model to get credentials for windows nodes. ([#80764](https://github.com/kubernetes/kubernetes/pull/80764), [@liyanhui1228](https://github.com/liyanhui1228))
* TBD ([#80730](https://github.com/kubernetes/kubernetes/pull/80730), [@jennybuckley](https://github.com/jennybuckley))
* The `AdmissionReview` API sent to and received from admission webhooks has been promoted to `admission.k8s.io/v1`. Webhooks can specify a preference for receiving `v1` AdmissionReview objects with `admissionReviewVersions: ["v1","v1beta1"]`, and must respond with an API object in the same `apiVersion` they are sent. When webhooks use `admission.k8s.io/v1`, the following additional validation is performed on their responses: ([#80231](https://github.com/kubernetes/kubernetes/pull/80231), [@liggitt](https://github.com/liggitt))
        * `response.patch` and `response.patchType` are not permitted from validating admission webhooks
        * `apiVersion: "admission.k8s.io/v1"` is required
        * `kind: "AdmissionReview"` is required
        * `response.uid: "<value of request.uid>"` is required
        * `response.patchType: "JSONPatch"` is required (if `response.patch` is set)
* "kubeadm join" fails if file-based discovery is too long, with a default timeout of 5 minutes. ([#80804](https://github.com/kubernetes/kubernetes/pull/80804), [@olivierlemasle](https://github.com/olivierlemasle))
* enhance Azure cloud provider code to support both AAD and ADFS authentication. ([#80841](https://github.com/kubernetes/kubernetes/pull/80841), [@rjaini](https://github.com/rjaini))
* Attempt to set the kubelet's hostname & internal IP if `--cloud-provider=external` and no node addresses exists ([#75229](https://github.com/kubernetes/kubernetes/pull/75229), [@andrewsykim](https://github.com/andrewsykim))
* kubeadm: avoid double deletion of the upgrade prepull DaemonSet ([#80798](https://github.com/kubernetes/kubernetes/pull/80798), [@xlgao-zju](https://github.com/xlgao-zju))
* Fixes problems with connecting to services on localhost on some systems; in particular, DNS queries to systemd-resolved on Ubuntu. ([#80591](https://github.com/kubernetes/kubernetes/pull/80591), [@danwinship](https://github.com/danwinship))
* Implement normalize plugin extension point for the scheduler framework. ([#80383](https://github.com/kubernetes/kubernetes/pull/80383), [@liu-cong](https://github.com/liu-cong))
* Fixed the bash completion error with override flags. ([#80802](https://github.com/kubernetes/kubernetes/pull/80802), [@dtaniwaki](https://github.com/dtaniwaki))
* Fix CVE-2019-11247: API server allows access to custom resources via wrong scope ([#80750](https://github.com/kubernetes/kubernetes/pull/80750), [@sttts](https://github.com/sttts))
* Failed iscsi logout is now re-tried periodically. ([#78941](https://github.com/kubernetes/kubernetes/pull/78941), [@jsafrane](https://github.com/jsafrane))
* Fix public IP not found issues for VMSS nodes ([#80703](https://github.com/kubernetes/kubernetes/pull/80703), [@feiskyer](https://github.com/feiskyer))
* In order to enable dual-stack support within kubeadm and kubernetes components, as part of the init config file, the user should set feature-gate IPv6DualStack=true in the ClusterConfiguration. Additionally, for each worker node, the user should set the feature-gate for kubelet using either nodeRegistration.kubeletExtraArgs or  KUBELET_EXTRA_ARGS. ([#80531](https://github.com/kubernetes/kubernetes/pull/80531), [@Arvinderpal](https://github.com/Arvinderpal))
* Fix error in `kubeadm join --discovery-file` when using discovery files with embedded credentials ([#80675](https://github.com/kubernetes/kubernetes/pull/80675), [@fabriziopandini](https://github.com/fabriziopandini))



# v1.16.0-alpha.2

[Documentation](https://docs.k8s.io)

## Downloads for v1.16.0-alpha.2


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes.tar.gz) | `7dfa3f8b9e98e528e2b49ed9cca5e95f265b9e102faac636ff0c29e045689145be236b98406a62eb0385154dc0c1233cac049806c99c9e46590cad5aa729183f`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-src.tar.gz) | `7cf14b92c96cab5fcda3115ec66b44562ca26ea6aa46bc7fa614fa66bda1bdf9ac1f3c94ef0dfa0e37c992c7187ecf4205b253f37f280857e88a318f8479c9a9`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-darwin-386.tar.gz) | `4871756de2cd1add0b07ec1e577c500d18a59e2f761595b939e1d4e10fbe0a119479ecaaf53d75cb2138363deae23cc88cba24fe3018cec6a27a3182f37cae92`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | `dbd9ca5fd90652ffc1606f50029d711eb52d34b707b7c04f29201f85aa8a5081923a53585513634f3adb6ace2bc59be9d4ad2abc49fdc3790ef805378c111e68`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-linux-386.tar.gz) | `6b049098b1dc65416c5dcc30346b82e5cf69a1cdd7e7b065429a76d302ef4b2a1c8e2dc621e9d5c1a6395a1fbd97f196d99404810880d118576e7b94e5621e4c`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | `7240a9d49e445e9fb0c9d360a9287933c6c6e7d81d6e11b0d645d3f9b6f3f1372cc343f03d10026518df5d6c95525e84c41b06a034c9ec2c9e306323dbd9325b`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | `947b0d9aeeef08961c0582b4c3c94b7ae1016d20b0c9f50af5fe760b3573f17497059511bcb57ac971a5bdadeb5c77dfd639d5745042ecc67541dd702ee7c657`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | `aff0258a223f5061552d340cda36872e3cd7017368117bbb14dc0f8a3a4db8c715c11743bedd72189cd43082aa9ac1ced64a6337c2f174bdcbeef094b47e76b0`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | `3eabecd62290ae8d876ae45333777b2c9959e39461197dbe90e6ba07d0a4c50328cbdf44e77d2bd626e435ffc69593d0e8b807b36601c19dd1a1ef17e6810b4f`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | `6651b2d95d0a8dd748c33c9e8018ab606b4061956cc2b6775bd0b008b04ea33df27be819ce6c391ceb2191b53acbbc088d602ed2d86bdd7a3a3fc1c8f876798a`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-windows-386.tar.gz) | `4b6c11b7a318e5fcac19144f6ab1638126c299e08c7b908495591674abcf4c7dd16f63c74c7d901beff24006150d2a31e0f75e28a9e14d6d0d88a09dafb014f0`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | `760ae08da6045ae7089fb27a9324e77bed907662659364857e1a8d103d19ba50e80544d8c21a086738b15baebfd9a5fa78d63638eff7bbe725436c054ba649cc`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | `69db41f3d79aa0581c36a3736ab8dc96c92127b82d3cf25c5effc675758fe713ca7aa7e5b414914f1bc73187c6cee5f76d76b74a2ee1c0e7fa61557328f1b8ef`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | `ca302f53ee91ab4feb697bb34d360d0872a7abea59c5f28cceefe9237a914c77d68722b85743998ab12bf8e42005e63a1d1a441859c2426c1a8d745dd33f4276`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | `79ab1f0a542ce576ea6d81cd2a7c068da6674177b72f1b5f5e3ca47edfdb228f533683a073857b6bc53225a230d15d3ba4b0cb9b6d5d78a309aa6e24c2f6c500`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | `fbe5b45326f1d03bcdd9ffd46ab454917d79f629ba23dae9d667d0c7741bc2f5db2960bf3c989bb75c19c9dc1609dacbb8a6dc9a440e5b192648e70db7f68721`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | `eb13ac306793679a3a489136bb7eb6588472688b2bb2aa0e54e61647d8c9da6d3589c19e7ac434c24defa78cb65f7b72593eedec1e7431c7ecae872298efc4de`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | `a4bde88f3e0f6233d04f04d380d5f612cd3c574bd66b9f3ee531fa76e3e0f1c6597edbc9fa61251a377e8230bce0ce6dc1cf57fd19080bb7d13f14a391b27fe8`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | `7d72aa8c1d883b9f047e5b98dbb662bdfd314f9c06af4213068381ffaac116e68d1aad76327ead7a4fd97976ea72277cebcf765c56b265334cb3a02c83972ec1`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | `c9380bb59ba26dcfe1ab52b5cb02e2d920313defda09ec7d19ccbc18f54def4b57cf941ac8a397392beb5836fdc12bc9600d4055f2cfd1319896cfc9631cab10`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | `7bcd79b368a62c24465fce7dcb024bb629eae034e09fb522fb43bb5798478ca2660a3ccc596b424325c6f69e675468900f3b41f3924e7ff453e3db40150b3c16`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | `9bda9dd24ee5ca65aaefece4213b46ef57cde4904542d94e6147542e42766f8b80fe24d99a6b8711bd7dbe00c415169a9f258f433c5f5345c2e17c2bb82f2670`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | `d5906f229d2d8e99bdb37e7d155d54560b82ea28ce881c5a0cde8f8d20bff8fd2e82ea4b289ae3e58616d3ec8c23ac9b473cb714892a377feb87ecbce156147d`

## Changelog since v1.16.0-alpha.1

### Action Required

* Revert "scheduler.alpha.kubernetes.io/critical-pod annotation is removed. Pod priority (spec.priorityClassName) should be used instead to mark pods as critical. Action required!" ([#80277](https://github.com/kubernetes/kubernetes/pull/80277), [@draveness](https://github.com/draveness))
* ACTION REQUIRED: container images tar files for 'amd64' will now contain the architecture in the RepoTags manifest.json section. ([#80266](https://github.com/kubernetes/kubernetes/pull/80266), [@javier-b-perez](https://github.com/javier-b-perez))
    * If you are using docker manifests there are not visible changes.

### Other notable changes

* Use HTTPS as etcd-apiserver protocol when mTLS between etcd and kube-apiserver on master is enabled, change etcd metrics/health port to 2382. ([#77561](https://github.com/kubernetes/kubernetes/pull/77561), [@wenjiaswe](https://github.com/wenjiaswe))
* kubelet: change node-lease-renew-interval to 0.25 of lease-renew-duration ([#80429](https://github.com/kubernetes/kubernetes/pull/80429), [@gaorong](https://github.com/gaorong))
* Fix error handling and potential go null pointer exception in kubeadm upgrade diff ([#80648](https://github.com/kubernetes/kubernetes/pull/80648), [@odinuge](https://github.com/odinuge))
* New flag --endpoint-updates-batch-period in kube-controller-manager can be used to reduce number of endpoints updates generated by pod changes. ([#80509](https://github.com/kubernetes/kubernetes/pull/80509), [@mborsz](https://github.com/mborsz))
* kubeadm: produce errors if they occur when resetting cluster status for a control-plane node ([#80573](https://github.com/kubernetes/kubernetes/pull/80573), [@bart0sh](https://github.com/bart0sh))
* When a load balancer type service is created in a k8s cluster that is backed by Azure Standard Load Balancer, the corresponding load balancer rule added in the Azure Standard Load Balancer would now have the "EnableTcpReset" property set to true.  ([#80624](https://github.com/kubernetes/kubernetes/pull/80624), [@xuto2](https://github.com/xuto2))
* Update portworx plugin dependency on libopenstorage/openstorage to v1.0.0 ([#80495](https://github.com/kubernetes/kubernetes/pull/80495), [@adityadani](https://github.com/adityadani))
* Fixed detachment of deleted volumes on OpenStack / Cinder. ([#80518](https://github.com/kubernetes/kubernetes/pull/80518), [@jsafrane](https://github.com/jsafrane))
* when PodInfoOnMount is enabled for a CSI driver, the new csi.storage.k8s.io/ephemeral parameter in the volume context allows a driver's NodePublishVolume implementation to determine on a case-by-case basis whether the volume is ephemeral or a normal persistent volume ([#79983](https://github.com/kubernetes/kubernetes/pull/79983), [@pohly](https://github.com/pohly))
* Update gogo/protobuf to serialize backward, as to get better performance on deep objects. ([#77355](https://github.com/kubernetes/kubernetes/pull/77355), [@apelisse](https://github.com/apelisse))
* Remove GetReference() and GetPartialReference() function from pkg/api/ref, as the same function exists also in staging/src/k8s.io/client-go/tools/ref ([#80361](https://github.com/kubernetes/kubernetes/pull/80361), [@wojtek-t](https://github.com/wojtek-t))
* Fixed a bug in the CSI metrics that does not return not supported error when a CSI driver does not support metrics. ([#79851](https://github.com/kubernetes/kubernetes/pull/79851), [@jparklab](https://github.com/jparklab))
* Fixed a bug in kube-addon-manager's leader election logic that made all replicas active. ([#80575](https://github.com/kubernetes/kubernetes/pull/80575), [@mborsz](https://github.com/mborsz))
* Kibana has been slightly revamped/improved in the latest version ([#80421](https://github.com/kubernetes/kubernetes/pull/80421), [@lostick](https://github.com/lostick))
* kubeadm: fixed ignoring errors when pulling control plane images ([#80529](https://github.com/kubernetes/kubernetes/pull/80529), [@bart0sh](https://github.com/bart0sh))
* CRDs under k8s.io and kubernetes.io must have the "api-approved.kubernetes.io" set to either `unapproved.*` or a link to the pull request approving the schema.  See https://github.com/kubernetes/enhancements/pull/1111 for more details. ([#79992](https://github.com/kubernetes/kubernetes/pull/79992), [@deads2k](https://github.com/deads2k))
* Reduce kube-proxy cpu usage in IPVS mode when a large number of nodePort services exist. ([#79444](https://github.com/kubernetes/kubernetes/pull/79444), [@cezarsa](https://github.com/cezarsa))
* Add CSI Migration Shim for VerifyVolumesAreAttached and BulkVolumeVerify ([#80443](https://github.com/kubernetes/kubernetes/pull/80443), [@davidz627](https://github.com/davidz627))
* Fix a bug that causes DaemonSet rolling update hang when there exist failed pods.  ([#78170](https://github.com/kubernetes/kubernetes/pull/78170), [@DaiHao](https://github.com/DaiHao))
* Fix retry issues when the nodes are under deleting on Azure ([#80419](https://github.com/kubernetes/kubernetes/pull/80419), [@feiskyer](https://github.com/feiskyer))
* Add support for AWS EBS on windows ([#79552](https://github.com/kubernetes/kubernetes/pull/79552), [@wongma7](https://github.com/wongma7))
* Passing an invalid policy name in the `--cpu-manager-policy` flag will now cause the kubelet to fail instead of simply ignoring the flag and running the `cpumanager`s default policy instead. ([#80294](https://github.com/kubernetes/kubernetes/pull/80294), [@klueska](https://github.com/klueska))
* Add Filter extension point to the scheduling framework. ([#78477](https://github.com/kubernetes/kubernetes/pull/78477), [@YoubingLi](https://github.com/YoubingLi))
* cpuUsageNanoCores is now reported in the Kubelet summary API on Windows nodes ([#80176](https://github.com/kubernetes/kubernetes/pull/80176), [@liyanhui1228](https://github.com/liyanhui1228))
* `[]TopologySpreadConstraint` is introduced into PodSpec to support the "Even Pods Spread" alpha feature. ([#77327](https://github.com/kubernetes/kubernetes/pull/77327), [@Huang-Wei](https://github.com/Huang-Wei))
* kubeadm: fall back to client version in case of certain HTTP errors ([#80024](https://github.com/kubernetes/kubernetes/pull/80024), [@RainbowMango](https://github.com/RainbowMango))
* NFS Drivers are now enabled to collect metrics, StatFS metrics provider is used to collect the metrics. ([#75805](https://github.com/kubernetes/kubernetes/pull/75805) , [@brahmaroutu](https://github.com/brahmaroutu)) ([#75805](https://github.com/kubernetes/kubernetes/pull/75805), [@brahmaroutu](https://github.com/brahmaroutu))
* make node lease renew interval more heuristic based on node-status-update-frequency in kubelet ([#80173](https://github.com/kubernetes/kubernetes/pull/80173), [@gaorong](https://github.com/gaorong))
* Introduction of the pod overhead feature to the scheduler.  This functionality is alpha-level as of  ([#78319](https://github.com/kubernetes/kubernetes/pull/78319), [@egernst](https://github.com/egernst))
    * Kubernetes v1.16, and is only honored by servers that enable the PodOverhead feature.gate. 
* N/A ([#80260](https://github.com/kubernetes/kubernetes/pull/80260), [@khenidak](https://github.com/khenidak))
* Add v1.Container.SecurityContext.WindowsOptions.RunAsUserName to the pod spec  ([#79489](https://github.com/kubernetes/kubernetes/pull/79489), [@bclau](https://github.com/bclau))
* Pass-through volume MountOptions to global mount (NodeStageVolume) on the node for CSI ([#80191](https://github.com/kubernetes/kubernetes/pull/80191), [@davidz627](https://github.com/davidz627))
* Add Score extension point to the scheduling framework. ([#79109](https://github.com/kubernetes/kubernetes/pull/79109), [@ahg-g](https://github.com/ahg-g))



# v1.16.0-alpha.1

[Documentation](https://docs.k8s.io)

## Downloads for v1.16.0-alpha.1


filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes.tar.gz) | `4834c52267414000fa93c0626bded5a969cf65d3d4681c20e5ae2c5f62002a51dfb8ee869484f141b147990915ba57be96108227f86c4e9f571b4b25e7ed0773`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-src.tar.gz) | `9329d51f5c73f830f3c895c2601bc78e51d2d412b928c9dae902e9ba8d46338f246a79329a27e4248ec81410ff103510ba9b605bb03e08a48414b2935d2c164b`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-darwin-386.tar.gz) | `3cedffb92a0fca4f0b2d41f8b09baa59dff58df96446e8eece4e1b81022d9fdda8da41b5f73a3468435474721f03cffc6e7beabb25216b089a991b68366c73bc`
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | `14de6bb296b4d022f50778b160c98db3508c9c7230946e2af4eb2a1d662d45b86690e9e04bf3e592ec094e12bed1f2bb74cd59d769a0eaac3c81d9b80e0a79c8`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-linux-386.tar.gz) | `8b2b9fa55890895239b99fabb866babe50aca599591db1ecf9429e49925ae478b7c813b9d7704a20f41f2d50947c3b3deecb594544f1f3eae6c4e97ae9bb9b70`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | `e927ac7b314777267b95e0871dd70c352ec0fc967ba221cb6cba523fa6f18d9d193e4ce92a1f9fa669f9c961de0e34d69e770ef745199ed3693647dd0d692e57`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | `4a230a6d34e2ffd7df40c5b726fbcbb7ef1373d81733bfb75685b2448ed181eb49ef27668fc33700f30de88e5bbdcc1e52649b9d31c7940760f48c6e6eb2f403`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | `87c8d7185df23b3496ceb74606558d895a64daf0c41185c833a233e29216131baac6e356a57bb78293ed9d0396966ecc3b00789f2b66af352dc286b101bcc69a`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | `16ea5efa2fc29bc7448a609a7118e7994e901ab26462aac52f03b4851d4c9d103ee12d2335360f8aa503ddbb2a71f3000f0fcb33597dd813df4f5ad5f4819fa9`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | `7390ad1682227a70550b20425fa5287fecf6a5d413493b03df3a7795614263e7883f30f3078bbb9fbd389d2a1dab073f8f401be89b82bd5861fa6b0aeda579eb`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-windows-386.tar.gz) | `88251896dfe38e59699b879f643704c0195e7a5af2cb00078886545f49364a2e3b497590781f135b80d60e256bad3a4ea197211f4f061c98dee096f0845e7a9b`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | `766b2a9bf097e45b2549536682cf25129110bd0562ab0df70e841ff8657dd7033119b0929e7a213454f90594b19b90fa57d89918cee33ceadba7d689449fe333`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | `dfd5c2609990c9b9b94249c654931b240dc072f2cc303e1e1d6dec1fddfb0a9e127e3898421ace00ab1947a3ad2f87cfd1266fd0b6193ef00f942269388ef372`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | `7704c2d3c57950f184322263ac2be1649a0d737d176e7fed1897031d0efb8375805b5f12c7cf9ba87ac06ad8a635d6e399382d99f3cbb418961a4f0901465f50`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | `fbbd87cc38cfb6429e3741bfd87ecec4b69b551df6fb7c121900ced4c1cd0bc77a317ca8abd41f71ffd7bc0b1c7144fecb22fa405d0b211b238df24d28599333`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | `cfed5b936eb2fe44df5d0c9c6484bee38ef370fb1258522e8c62fb6a526e9440c1dc768d8bf33403451ae00519cab1450444da854fd6c6a37665ce925c4e7d69`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | `317681141734347260ad9f918fa4b67e48751f5a7df64a848d2a83c79a4e9dba269c51804b09444463ba88a2c0efa1c307795cd8f06ed840964eb2c725a4ecc3`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | `b3b1013453d35251b8fc4759f6ac26bdeb37f14a98697078535f7f902e8ebca581b5629bbb4493188a7e6077eb5afc61cf275f42bf4d9f503b70bfc58b9730b2`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | `0bacc1791d260d2863ab768b48daf66f0f7f89eeee70e68dd515b05fc9d7f14b466382fe16fa84a103e0023324f681767489d9485560baf9eb80fe0e7ffab503`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | `73bd70cb9d27ce424828a95d715c16fd9dd22396dbe1dfe721eb0aea9e186ec46e6978956613b0978a8da3c22df39790739b038991c0192281881fce41d7c9f1`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | `a865f98838143dc7e1e12d1e258e5f5f2855fcf6e88488fb164ad62cf886d8e2a47fdf186ad6b55172f73826ae19da9b2642b9a0df0fa08f9351a66aeef3cf17`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | `d2f9f746ed0fe00be982a847a3ae1b6a698d5c506be1d3171156902140fec64642ec6d99aa68de08bdc7d65c9a35ac2c36bda53c4db873cb8e7edc419a4ab958`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.16.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | `37f48a6d8174f38668bc41c81222615942bfe07e01f319bdfed409f83a3de3773dceb09fd86330018bb05f830e165e7bd85b3d23d26a50227895e4ec07f8ab98`

## Changelog since v1.15.0

### Action Required

* Migrate scheduler to use v1beta1 Event API. action required: any tool targeting scheduler events needs to use v1beta1 Event API ([#78447](https://github.com/kubernetes/kubernetes/pull/78447), [@yastij](https://github.com/yastij))
* scheduler.alpha.kubernetes.io/critical-pod annotation is removed. Pod priority (spec.priorityClassName) should be used instead to mark pods as critical. Action required! ([#79554](https://github.com/kubernetes/kubernetes/pull/79554), [@draveness](https://github.com/draveness))
* hyperkube: the `--make-symlinks` flag, deprecated in v1.14, has been removed. ([#80017](https://github.com/kubernetes/kubernetes/pull/80017), [@Pothulapati](https://github.com/Pothulapati))
* Node labels `beta.kubernetes.io/metadata-proxy-ready`, `beta.kubernetes.io/metadata-proxy-ready` and `beta.kubernetes.io/kube-proxy-ds-ready` are no longer added on new nodes. ([#79305](https://github.com/kubernetes/kubernetes/pull/79305), [@paivagustavo](https://github.com/paivagustavo))
        * ip-mask-agent addon starts to use the label `node.kubernetes.io/masq-agent-ds-ready` instead of `beta.kubernetes.io/masq-agent-ds-ready` as its node selector.
        * kube-proxy addon starts to use the label `node.kubernetes.io/kube-proxy-ds-ready` instead of `beta.kubernetes.io/kube-proxy-ds-ready` as its node selector.
        * metadata-proxy addon starts to use the label `cloud.google.com/metadata-proxy-ready` instead of `beta.kubernetes.io/metadata-proxy-ready` as its node selector.
        * Kubelet removes the ability to set `kubernetes.io` or `k8s.io` labels via --node-labels other than the [specifically allowed labels/prefixes](https://github.com/kubernetes/enhancements/blob/master/keps/sig-auth/0000-20170814-bounding-self-labeling-kubelets.md#proposal).
* The following APIs are no longer served by default: ([#70672](https://github.com/kubernetes/kubernetes/pull/70672), [@liggitt](https://github.com/liggitt))
        * All resources under `apps/v1beta1` and `apps/v1beta2` - use `apps/v1` instead
        * `daemonsets`, `deployments`, `replicasets` resources under `extensions/v1beta1` - use `apps/v1` instead
        * `networkpolicies` resources under `extensions/v1beta1` - use `networking.k8s.io/v1` instead
        * `podsecuritypolicies` resources under `extensions/v1beta1` - use `policy/v1beta1` instead
    * Serving these resources can be temporarily re-enabled using the `--runtime-config` apiserver flag. 
        * `apps/v1beta1=true`
        * `apps/v1beta2=true`
        * `extensions/v1beta1/daemonsets=true,extensions/v1beta1/deployments=true,extensions/v1beta1/replicasets=true,extensions/v1beta1/networkpolicies=true,extensions/v1beta1/podsecuritypolicies=true`
    * The ability to serve these resources will be completely removed in v1.18.
* ACTION REQUIRED: Removed deprecated flag `--resource-container` from kube-proxy. ([#78294](https://github.com/kubernetes/kubernetes/pull/78294), [@vllry](https://github.com/vllry))
    * The deprecated `--resource-container` flag has been removed from kube-proxy, and specifying it will now cause an error.  The behavior is now as if you specified `--resource-container=""`.  If you previously specified a non-empty `--resource-container`, you can no longer do so as of kubernetes 1.16.

### Other notable changes

* When HPAScaleToZero feature gate is enabled HPA supports scaling to zero pods based on object or external metrics. HPA remains active as long as at least one metric value available. ([#74526](https://github.com/kubernetes/kubernetes/pull/74526), [@DXist](https://github.com/DXist))
    * To downgrade the cluster to version that does not support scale-to-zero feature:
    * 1. make sure there are no hpa objects with minReplicas=0. Here is a oneliner to update it to 1:
    *     $ kubectl get hpa --all-namespaces  --no-headers=true | awk  '{if($6==0) printf "kubectl patch hpa/%s --namespace=%s -p \"{\\"spec\\":{\\"minReplicas\\":1}}\"
", $2, $1 }' | sh
    * 2. disable HPAScaleToZero feature gate
* Add support for writing out of tree custom scheduler plugins. ([#78162](https://github.com/kubernetes/kubernetes/pull/78162), [@hex108](https://github.com/hex108))
* Remove deprecated github.com/kardianos/osext dependency ([#80142](https://github.com/kubernetes/kubernetes/pull/80142), [@loqutus](https://github.com/loqutus))
* Add Bind extension point to the scheduling framework. ([#79313](https://github.com/kubernetes/kubernetes/pull/79313), [@chenchun](https://github.com/chenchun))
* On Windows systems, %USERPROFILE% is now preferred over %HOMEDRIVE%\%HOMEPATH% as the home folder if %HOMEDRIVE%\%HOMEPATH% does not contain a .kube* Add --kubernetes-version to "kubeadm init phase certs ca" and "kubeadm init phase kubeconfig" ([#80115](https://github.com/kubernetes/kubernetes/pull/80115), [@gyuho](https://github.com/gyuho))
* kubeadm ClusterConfiguration now supports featureGates: IPv6DualStack: true ([#80145](https://github.com/kubernetes/kubernetes/pull/80145), [@Arvinderpal](https://github.com/Arvinderpal))
* Fix a bug that ListOptions.AllowWatchBookmarks wasn't propagating correctly in kube-apiserver. ([#80157](https://github.com/kubernetes/kubernetes/pull/80157), [@wojtek-t](https://github.com/wojtek-t))
* Bugfix: csi plugin supporting raw block that does not need attach mounted failed ([#79920](https://github.com/kubernetes/kubernetes/pull/79920), [@cwdsuzhou](https://github.com/cwdsuzhou))
* Increase log level for graceful termination to v=5 ([#80100](https://github.com/kubernetes/kubernetes/pull/80100), [@andrewsykim](https://github.com/andrewsykim))
* kubeadm: support fetching configuration from the original cluster for 'upgrade diff' ([#80025](https://github.com/kubernetes/kubernetes/pull/80025), [@SataQiu](https://github.com/SataQiu))
* The sample-apiserver gains support for OpenAPI v2 spec serving at `/openapi/v2`. ([#79843](https://github.com/kubernetes/kubernetes/pull/79843), [@sttts](https://github.com/sttts))
    * The `generate-internal-groups.sh` script in k8s.io/code-generator will generate OpenAPI definitions by default in `pkg/generated/openapi`. Additional API group dependencies can be added via `OPENAPI_EXTRA_PACKAGES=<group>/<version> <group2>/<version2>...`.
* Cinder and ScaleIO volume providers have been deprecated and will be removed in a future release. ([#80099](https://github.com/kubernetes/kubernetes/pull/80099), [@dims](https://github.com/dims))
* kubelet's --containerized flag was deprecated in 1.14. This flag is removed in 1.16. ([#80043](https://github.com/kubernetes/kubernetes/pull/80043), [@dims](https://github.com/dims))
* Optimize EC2 DescribeInstances API calls in aws cloud provider library by querying instance ID instead of EC2 filters when possible ([#78140](https://github.com/kubernetes/kubernetes/pull/78140), [@zhan849](https://github.com/zhan849))
* etcd migration image no longer supports etcd2 version.  ([#80037](https://github.com/kubernetes/kubernetes/pull/80037), [@dims](https://github.com/dims))
* Promote WatchBookmark feature to beta and enable it by default. ([#79786](https://github.com/kubernetes/kubernetes/pull/79786), [@wojtek-t](https://github.com/wojtek-t))
    * With WatchBookmark feature, clients are able to request watch events with BOOKMARK type. Clients should not assume bookmarks are returned at any specific interval, nor may they assume the server will send any BOOKMARK event during a session.
* update to use go 1.12.7 ([#79966](https://github.com/kubernetes/kubernetes/pull/79966), [@tao12345666333](https://github.com/tao12345666333))
* Add --shutdown-delay-duration to kube-apiserver in order to delay a graceful shutdown. `/healthz` will keep returning success during this time and requests are normally served, but `/readyz` will return faillure immediately. This delay can be used to allow the SDN to update iptables on all nodes and stop sending traffic. ([#74416](https://github.com/kubernetes/kubernetes/pull/74416), [@sttts](https://github.com/sttts))
* The `MutatingWebhookConfiguration` and `ValidatingWebhookConfiguration` APIs have been promoted to `admissionregistration.k8s.io/v1`: ([#79549](https://github.com/kubernetes/kubernetes/pull/79549), [@liggitt](https://github.com/liggitt))
        * `failurePolicy` default changed from `Ignore` to `Fail` for v1
        * `matchPolicy` default changed from `Exact` to `Equivalent` for v1
        * `timeout` default changed from `30s` to `10s` for v1
        * `sideEffects` default value is removed and the field made required for v1
        * `admissionReviewVersions` default value is removed and the field made required for v1 (supported versions for AdmissionReview are `v1` and `v1beta1`)
        * The `name` field for specified webhooks must be unique for `MutatingWebhookConfiguration` and `ValidatingWebhookConfiguration` objects created via `admissionregistration.k8s.io/v1`
    * The `admissionregistration.k8s.io/v1beta1` versions of `MutatingWebhookConfiguration` and `ValidatingWebhookConfiguration` are deprecated and will no longer be served in v1.19.
* The garbage collector and generic object quota controller have been updated to use the metadata client which improves memory ([#78742](https://github.com/kubernetes/kubernetes/pull/78742), [@smarterclayton](https://github.com/smarterclayton))
    * and CPU usage of the Kube controller manager.
* SubjectAccessReview requests sent for RBAC escalation, impersonation, and pod security policy authorization checks now populate the version attribute. ([#80007](https://github.com/kubernetes/kubernetes/pull/80007), [@liggitt](https://github.com/liggitt))
* na ([#79892](https://github.com/kubernetes/kubernetes/pull/79892), [@mikebrow](https://github.com/mikebrow))
* Use O_CLOEXEC to ensure file descriptors do not leak to subprocesses. ([#74691](https://github.com/kubernetes/kubernetes/pull/74691), [@cpuguy83](https://github.com/cpuguy83))
* The namespace controller has been updated to use the metadata client which improves memory ([#78744](https://github.com/kubernetes/kubernetes/pull/78744), [@smarterclayton](https://github.com/smarterclayton))
    * and CPU usage of the Kube controller manager.
* NONE ([#79933](https://github.com/kubernetes/kubernetes/pull/79933), [@mm4tt](https://github.com/mm4tt))
* add  `kubectl replace --raw` and `kubectl delete --raw` to have parity with create and get ([#79724](https://github.com/kubernetes/kubernetes/pull/79724), [@deads2k](https://github.com/deads2k))
* E2E tests no longer add command line flags directly to the command line, test suites that want that need to be updated if they don't use HandleFlags ([#75593](https://github.com/kubernetes/kubernetes/pull/75593), [@pohly](https://github.com/pohly))
    * loading a -viper-config=e2e.yaml with suffix (introduced in 1.13) works again and now has a regression test
* Kubernetes now supports transparent compression of API responses. Clients that send `Accept-Encoding: gzip` will now receive a GZIP compressed response body if the API call was larger than 128KB.  Go clients automatically request gzip-encoding by default and should see reduced transfer times for very large API requests.  Clients in other languages may need to make changes to benefit from compression. ([#77449](https://github.com/kubernetes/kubernetes/pull/77449), [@smarterclayton](https://github.com/smarterclayton))
* Resolves an issue serving aggregated APIs backed by services that respond to requests to `/` with non-2xx HTTP responses ([#79895](https://github.com/kubernetes/kubernetes/pull/79895), [@deads2k](https://github.com/deads2k))
* updated fluentd to 1.5.1, elasticsearchs & kibana to 7.1.1 ([#79014](https://github.com/kubernetes/kubernetes/pull/79014), [@monotek](https://github.com/monotek))
* kubeadm: implement support for concurrent add/remove of stacked etcd members ([#79677](https://github.com/kubernetes/kubernetes/pull/79677), [@neolit123](https://github.com/neolit123))
* Added a metric 'apiserver_watch_events_total' that can be used to understand the number of watch events in the system. ([#78732](https://github.com/kubernetes/kubernetes/pull/78732), [@mborsz](https://github.com/mborsz))
* KMS Providers will install a healthz check for the status of kms-pluign in kube-apiservers' encryption config.  ([#78540](https://github.com/kubernetes/kubernetes/pull/78540), [@immutableT](https://github.com/immutableT))
* Fixes a bug in openapi published for custom resources using x-kubernetes-preserve-unknown-fields extensions, so that kubectl will allow sending unknown fields for that portion of the object. ([#79636](https://github.com/kubernetes/kubernetes/pull/79636), [@liggitt](https://github.com/liggitt))
* A new client `k8s.io/client-go/metadata.Client` has been added for accessing objects generically. This client makes it easier to retrieve only the metadata (the `metadata` sub-section) from resources on the cluster in an efficient manner for use cases that deal with objects generically, like the garbage collector, quota, or the namespace controller. The client asks the server to return a `meta.k8s.io/v1 PartialObjectMetadata` object for list, get, delete, watch, and patch operations on both normal APIs and custom resources which can be encoded in protobuf for additional work. If the server does not yet support this API the client will gracefully fall back to JSON and transform the response objects into PartialObjectMetadata. ([#77819](https://github.com/kubernetes/kubernetes/pull/77819), [@smarterclayton](https://github.com/smarterclayton))
* changes timeout value in csi plugin from 15s to 2min which fixes the timeout issue ([#79529](https://github.com/kubernetes/kubernetes/pull/79529), [@andyzhangx](https://github.com/andyzhangx))
* kubeadm: provide "--control-plane-endpoint" flag for `controlPlaneEndpoint` ([#79270](https://github.com/kubernetes/kubernetes/pull/79270), [@SataQiu](https://github.com/SataQiu))
* Fixes invalid "time stamp is the future" error when kubectl cp-ing a file ([#73982](https://github.com/kubernetes/kubernetes/pull/73982), [@tanshanshan](https://github.com/tanshanshan))
* Kubelet should now more reliably report the same primary node IP even if the set of node IPs reported by the CloudProvider changes. ([#79391](https://github.com/kubernetes/kubernetes/pull/79391), [@danwinship](https://github.com/danwinship))
* To configure controller manager to use ipv6dual stack: ([#73977](https://github.com/kubernetes/kubernetes/pull/73977), [@khenidak](https://github.com/khenidak))
    * use --cluster-cidr="<cidr1>,<cidr2>".
    * Notes:
 
    * 1. Only the first two cidrs are used (soft limits for Alpha, might be lifted later on). 
    * 2. Only the "RangeAllocator" (default) is allowed as a value for --cidr-allocator-type . Cloud allocators are not compatible with ipv6dualstack 
* When using the conformance test image, a new environment variable E2E_USE_GO_RUNNER will cause the tests to be run with the new Golang-based test runner rather than the current bash wrapper. ([#79284](https://github.com/kubernetes/kubernetes/pull/79284), [@johnSchnake](https://github.com/johnSchnake))
* kubeadm: prevent PSP blocking of upgrade image prepull by using a non-root user ([#77792](https://github.com/kubernetes/kubernetes/pull/77792), [@neolit123](https://github.com/neolit123))
* kubelet now accepts a --cni-cache-dir option, which defaults to /var/lib/cni/cache, where CNI stores cache files. ([#78908](https://github.com/kubernetes/kubernetes/pull/78908), [@dcbw](https://github.com/dcbw))
* Update Azure API versions (containerregistry --> 2018-09-01, network --> 2018-08-01) ([#79583](https://github.com/kubernetes/kubernetes/pull/79583), [@justaugustus](https://github.com/justaugustus))
* Fix possible fd leak and closing of dirs in doSafeMakeDir  ([#79534](https://github.com/kubernetes/kubernetes/pull/79534), [@odinuge](https://github.com/odinuge))
* kubeadm: fix the bug that "--cri-socket" flag does not work for `kubeadm reset` ([#79498](https://github.com/kubernetes/kubernetes/pull/79498), [@SataQiu](https://github.com/SataQiu))
* kubectl logs --selector will support --tail=-1. ([#74943](https://github.com/kubernetes/kubernetes/pull/74943), [@JishanXing](https://github.com/JishanXing))
* Introduce a new admission controller for RuntimeClass. Initially, RuntimeClass will be used to apply the pod overhead associated with a given RuntimeClass to the Pod.Spec if a corresponding RuntimeClassName is specified. ([#78484](https://github.com/kubernetes/kubernetes/pull/78484), [@egernst](https://github.com/egernst))
    * PodOverhead is an alpha feature as of Kubernetes 1.16.
* Fix kubelet errors in AArch64 with huge page sizes smaller than 1MiB ([#78495](https://github.com/kubernetes/kubernetes/pull/78495), [@odinuge](https://github.com/odinuge))
* The alpha `metadata.initializers` field, deprecated in 1.13, has been removed. ([#79504](https://github.com/kubernetes/kubernetes/pull/79504), [@yue9944882](https://github.com/yue9944882))
* Fix duplicate error messages in cli commands ([#79493](https://github.com/kubernetes/kubernetes/pull/79493), [@odinuge](https://github.com/odinuge))
* Default resourceGroup should be used when the value of annotation azure-load-balancer-resource-group is an empty string. ([#79514](https://github.com/kubernetes/kubernetes/pull/79514), [@feiskyer](https://github.com/feiskyer))
* Fixes output of `kubectl get --watch-only` when watching a single resource ([#79345](https://github.com/kubernetes/kubernetes/pull/79345), [@liggitt](https://github.com/liggitt))
* RateLimiter add a context-aware method, fix client-go request goruntine backlog in async timeout scene. ([#79375](https://github.com/kubernetes/kubernetes/pull/79375), [@answer1991](https://github.com/answer1991))
* Fix a bug where kubelet would not retry pod sandbox creation when the restart policy of the pod is Never ([#79451](https://github.com/kubernetes/kubernetes/pull/79451), [@yujuhong](https://github.com/yujuhong))
* Fix CRD validation error on 'items' field. ([#76124](https://github.com/kubernetes/kubernetes/pull/76124), [@tossmilestone](https://github.com/tossmilestone))
* The CRD handler now properly re-creates stale CR storage to reflect CRD update. ([#79114](https://github.com/kubernetes/kubernetes/pull/79114), [@roycaihw](https://github.com/roycaihw))
* Integrated volume limits for in-tree and CSI volumes into one scheduler predicate. ([#77595](https://github.com/kubernetes/kubernetes/pull/77595), [@bertinatto](https://github.com/bertinatto))
* Fix a bug in server printer that could cause kube-apiserver to panic. ([#79349](https://github.com/kubernetes/kubernetes/pull/79349), [@roycaihw](https://github.com/roycaihw))
* Mounts /home/kubernetes/bin/nvidia/vulkan/icd.d on the host to /etc/vulkan/icd.d inside containers requesting GPU. ([#78868](https://github.com/kubernetes/kubernetes/pull/78868), [@chardch](https://github.com/chardch))
* Remove CSIPersistentVolume feature gates ([#79309](https://github.com/kubernetes/kubernetes/pull/79309), [@draveness](https://github.com/draveness))
* Init container resource requests now impact pod QoS class ([#75223](https://github.com/kubernetes/kubernetes/pull/75223), [@sjenning](https://github.com/sjenning))
* Correct the maximum allowed insecure bind port for the kube-scheduler and kube-apiserver to 65535. ([#79346](https://github.com/kubernetes/kubernetes/pull/79346), [@ncdc](https://github.com/ncdc))
* Fix remove the etcd member from the cluster during a kubeadm reset. ([#79326](https://github.com/kubernetes/kubernetes/pull/79326), [@bradbeam](https://github.com/bradbeam))
* Remove KubeletPluginsWatcher feature gates ([#79310](https://github.com/kubernetes/kubernetes/pull/79310), [@draveness](https://github.com/draveness))
* Remove HugePages, VolumeScheduling, CustomPodDNS and PodReadinessGates feature flags ([#79307](https://github.com/kubernetes/kubernetes/pull/79307), [@draveness](https://github.com/draveness))
* The GA PodPriority feature gate is now on by default and cannot be disabled. The feature gate will be removed in v1.18. ([#79262](https://github.com/kubernetes/kubernetes/pull/79262), [@draveness](https://github.com/draveness))
* Remove pids cgroup controller requirement when related feature gates are disabled ([#79073](https://github.com/kubernetes/kubernetes/pull/79073), [@rafatio](https://github.com/rafatio))
* Add Bind extension point of the scheduling framework ([#78513](https://github.com/kubernetes/kubernetes/pull/78513), [@chenchun](https://github.com/chenchun))
* if targetPort is changed that will process by service controller  ([#77712](https://github.com/kubernetes/kubernetes/pull/77712), [@Sn0rt](https://github.com/Sn0rt))
* update to use go 1.12.6 ([#78958](https://github.com/kubernetes/kubernetes/pull/78958), [@tao12345666333](https://github.com/tao12345666333))
* kubeadm: fix a potential panic if kubeadm discovers an invalid, existing kubeconfig file ([#79165](https://github.com/kubernetes/kubernetes/pull/79165), [@neolit123](https://github.com/neolit123))
* fix kubelet fail to delete orphaned pod directory when the kubelet's pods directory (default is "/var/lib/kubelet/pods") symbolically links to another disk device's directory ([#79094](https://github.com/kubernetes/kubernetes/pull/79094), [@gaorong](https://github.com/gaorong))
* Addition of Overhead field to the PodSpec and RuntimeClass types as part of the Pod Overhead KEP ([#76968](https://github.com/kubernetes/kubernetes/pull/76968), [@egernst](https://github.com/egernst))
* fix pod list return value of framework.WaitForPodsWithLabelRunningReady ([#78687](https://github.com/kubernetes/kubernetes/pull/78687), [@pohly](https://github.com/pohly))
* The behavior of the default handler for 404 requests fro the GCE Ingress load balancer is slightly modified in the sense that it now exports metrics using prometheus. The metrics exported include:  ([#79106](https://github.com/kubernetes/kubernetes/pull/79106), [@vbannai](https://github.com/vbannai))
    * - http_404_request_total  (the number of 404 requests handled)
    * - http_404_request_duration_ms (the amount of time the server took to respond in ms)
    * Also includes percentile groupings. The directory for the default 404 handler includes instructions on how to enable prometheus for monitoring and setting alerts.
* The kube-apiserver has improved behavior for both startup and shutdown sequences and also now exposes `eadyz` for readiness checking. Readyz includes all existing healthz checks but also adds a shutdown check. When a cluster admin initiates a shutdown, the kube-apiserver will try to process existing requests (for the duration of request timeout) before killing the apiserver process.   ([#78458](https://github.com/kubernetes/kubernetes/pull/78458), [@logicalhan](https://github.com/logicalhan))
    * The apiserver also now takes an optional flag "--maximum-startup-sequence-duration". This allows you to explicitly define an upper bound on the apiserver startup sequences before healthz begins to fail. By keeping the kubelet liveness initial delay short, this can enable quick kubelet recovery as soon as we have a boot sequence which has not completed in our expected time frame, despite lack of completion from longer boot sequences (like RBAC). Kube-apiserver behavior when the value of this flag is zero is backwards compatible (this is as the defaulted value of the flag).
* fix: make azure disk URI as case insensitive ([#79020](https://github.com/kubernetes/kubernetes/pull/79020), [@andyzhangx](https://github.com/andyzhangx))
* Enable cadvisor ProcessMetrics collecting. ([#79002](https://github.com/kubernetes/kubernetes/pull/79002), [@jiayingz](https://github.com/jiayingz))
* Fixes a bug where `kubectl set config` hangs and uses 100% CPU on some invalid property names  ([#79000](https://github.com/kubernetes/kubernetes/pull/79000), [@pswica](https://github.com/pswica))
* Fix a string comparison bug in IPVS graceful termination where UDP real servers are not deleted. ([#78999](https://github.com/kubernetes/kubernetes/pull/78999), [@andrewsykim](https://github.com/andrewsykim))
* Reflector watchHandler Warning log 'The resourceVersion for the provided watch is too old.' is now logged as Info.  ([#78991](https://github.com/kubernetes/kubernetes/pull/78991), [@sallyom](https://github.com/sallyom))
* fix a bug that pods not be deleted from unmatched nodes by daemon controller   ([#78974](https://github.com/kubernetes/kubernetes/pull/78974), [@DaiHao](https://github.com/DaiHao))
* NONE ([#78821](https://github.com/kubernetes/kubernetes/pull/78821), [@jhedev](https://github.com/jhedev))
* Volume expansion is enabled in the default GCE storageclass ([#78672](https://github.com/kubernetes/kubernetes/pull/78672), [@msau42](https://github.com/msau42))
* kubeadm: use the service-cidr flag to pass the desired service CIDR to the kube-controller-manager via its service-cluster-ip-range flag. ([#78625](https://github.com/kubernetes/kubernetes/pull/78625), [@Arvinderpal](https://github.com/Arvinderpal))
* kubeadm: introduce deterministic ordering for the certificates generation in the phase command "kubeadm init phase certs" . ([#78556](https://github.com/kubernetes/kubernetes/pull/78556), [@neolit123](https://github.com/neolit123))
* Add Pre-filter extension point to the scheduling framework. ([#78005](https://github.com/kubernetes/kubernetes/pull/78005), [@ahg-g](https://github.com/ahg-g))
* fix pod stuck issue due to corrupt mnt point in flexvol plugin, call Unmount if PathExists returns any error ([#75234](https://github.com/kubernetes/kubernetes/pull/75234), [@andyzhangx](https://github.com/andyzhangx))

