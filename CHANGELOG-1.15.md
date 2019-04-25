<!-- BEGIN MUNGE: GENERATED_TOC -->
- [v1.15.0-alpha.1](#v1150-alpha1)
  - [Downloads for v1.15.0-alpha.1](#downloads-for-v1150-alpha1)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.14.0](#changelog-since-v1140)
    - [Action Required](#action-required)
    - [Other notable changes](#other-notable-changes)
<!-- END MUNGE: GENERATED_TOC -->

<!-- NEW RELEASE NOTES ENTRY -->


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

