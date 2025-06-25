<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.34.0-alpha.1](#v1340-alpha1)
  - [Downloads for v1.34.0-alpha.1](#downloads-for-v1340-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
    - [Container Images](#container-images)
  - [Changelog since v1.33.0](#changelog-since-v1330)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)

<!-- END MUNGE: GENERATED_TOC -->

# v1.34.0-alpha.1


## Downloads for v1.34.0-alpha.1



### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes.tar.gz) | 4125206915e9f0cd7bffd77021f210901bade4747d84855c8210922c82e2085628a05b81cef137e347b16a05828f99ac2a27a8f8f19a14397011031454736ea0
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-src.tar.gz) | c1dfe0a1df556adcad5881a7960da5348feacc23894188b94eb75be0b156912ab8680b94e2579a96d9d71bff74b1c813b8592de6926fba8e5a030a88d8b4b208

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | 22c4d1031297ea1833b3cd3e6805008c34b66f932ead3818db3eb2663a71510a8cdb53a05852991d54e354800ee97a2aad4afc31726d956f38c674929ce10778
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 6be320d2075d8a7835751c019556059ff2fca704d0bbeeff181248492d8ed6fcc2d6d6b68c509e4453431100b06a20268e61b9e434b638a78ebfad68e7c41276
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-386.tar.gz) | e63ac6b7127591068626a3d7caf0e1bae6390106f6c93efae34b18e38af257f1521635eb2adf76c40ad0f0d9a5397947bbb0215087d4d2e87ce6f253b6aec1a4
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 12dc8dc4997b71038c377bfd9869610110cebb20afcb051e85c86832f75bc8e7eabbb08b5caa00423c5f8df68210ad5ca140a61d4a8e9ad8640f648250205752
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | 0a7f8df6abfe9971f778add6771135d7079c245b18dd941eacf1230f75f461e7d8302142584aa4d60062c8cfd4e021f21ae5aa428d82b5fbe3697bda0e5854ff
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | b1442640ac1e45268e9916d0c51e711b7640fd2594ecad05a0d990c19db2e0dcde53cc90fb13588a2b926e25c831f62bf5461fa9c8e6a03a83573cc1c3791903
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | e5a028da7fcb24aee85d010741c864fa4e5a3d6c87223b5c397686107a53dd2801a8c75cf9e1046ab28c97b06a5457aa6b3e4f809cd46cbe4858f78b2cb6a4df
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 4d3fce13d8f29e801c4d7355f83ded4d2e4abcc0b788f09d616ef7f89bd04e9d92d0b32e6e365118e618b32020d8b43e4cbd59a82262cc787b98f42e7df4ddbc
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-windows-386.tar.gz) | 3bbe15f8856cab69c727b02766024e1bb430add8ad18216929a96d7731d255c5d5bb6b678a4d4e7a021f2e976633b69c0516c2260dcc0bee7d2447f64bd52fe8
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | 1833d8b09d5524df91120115667f897df47ad66edb57d2570e022234794c4d0d09212fca9b0b64e21ccc8ce6dcd41080bf9198c81583949cb8001c749f25e8a0
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-client-windows-arm64.tar.gz) | c0819674e11923b38d2df7cb9955929247a5b0752c93fc5215300da3514c592348cbe649a5c6fd6ac63500c6d68cf61a2733c099788164547e3f7738afe78ecf

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | acd0b0b6723789780fd536894a965001056e94e92e2070edacdb53d2d879f56a90cc2c1ad0ff6d634ed74ef4debcefa01eee9f675cc4c70063da6cc52cc140d3
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 31321659424b4847ec456ae507486efe57c8e903c2bc450df65ffc3bc90011ba050e8351ab32133943dfebd9d6e8ad47f2546a7cdc47e424cdaf0dc7247e08c3
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | fe81aa313be46ed5cc91507e58bc165e98722921d33473c29d382dceb948b1ffc0437d74825277a7da487f9390dec64f6a70617b05e0441c106fa87af737b90c
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | 69a54f40e7a8684a6a1606f0463266d83af615f70a55d750031d82601c8070f4f9161048018c78e0859faa631ec9984fc20af3bc17240c8fc9394c6cbffacaf9

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 797a5df349e571330e8090bd78f024d659d0d46e8a7352210b80ac594ef50dc2f3866240b75f7c0d2e08fa526388d0dfdcb91b4686f01b547c860a2d0a9846a7
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 552a114facbd42c655574953186ba15a91c061b3db9ad25e665892c355347bf841e1bf716f8e28a16f1f1b37492911103212ec452bf5e663f8fcf26fae3ccc6a
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 7f08bad1921127fdceba7deb58d305e0b599de7ab588da936ff753ab4c6410b5db0634d71094e97ee1baeaccc491370c88268f6a540eedb556c90fb1ce350eda
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 4d1ac168b4591bf5ed7773d87eb47e64eb322adb6fd22b89f4f79c9849aee70188f0fa04a18775feff6f9baf95277499c56cd471a56240a87f9810c82434ba35
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.34.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 896e508aa1c0bb3249c01554aea0ea25d65c4d9740772f8c053ded411b89a34a1c1e954e62fad10a1366cb0a9534af9b3d4e0a46acd956b47eb801e900dfcbe6

### Container Images

All container images are available as manifest lists and support the described
architectures. It is also possible to pull a specific architecture directly by
adding the "-$ARCH" suffix  to the container image name.

name | architectures
---- | -------------
[registry.k8s.io/conformance:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/conformance-s390x)
[registry.k8s.io/kube-apiserver:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-apiserver-s390x)
[registry.k8s.io/kube-controller-manager:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-controller-manager-s390x)
[registry.k8s.io/kube-proxy:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-proxy-s390x)
[registry.k8s.io/kube-scheduler:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kube-scheduler-s390x)
[registry.k8s.io/kubectl:v1.34.0-alpha.1](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl) | [amd64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-amd64), [arm64](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-arm64), [ppc64le](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-ppc64le), [s390x](https://console.cloud.google.com/artifacts/docker/k8s-artifacts-prod/southamerica-east1/images/kubectl-s390x)

## Changelog since v1.33.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - For metrics `apiserver_cache_list_fetched_objects_total`, `apiserver_cache_list_returned_objects_total`, `apiserver_cache_list_total` replace `resource_prefix` label with API `group` and `resource` labels.
  For metrics `etcd_request_duration_seconds`, `etcd_requests_total` and `etcd_request_errors_total` replace `type` label with API `resource` and `group` label.
  For metric `apiserver_selfrequest_total` add a API `group` label.
  For metrics `apiserver_watch_events_sizes` and `apiserver_watch_events_total` replace API `kind` label with `resource` label.
  For metrics `apiserver_request_body_size_bytes`, `apiserver_storage_events_received_total`, `apiserver_storage_list_evaluated_objects_total`, `apiserver_storage_list_fetched_objects_total`, `apiserver_storage_list_returned_objects_total`, `apiserver_storage_list_total`, `apiserver_watch_cache_events_dispatched_total`, `apiserver_watch_cache_events_received_total`, `apiserver_watch_cache_initializations_total`, `apiserver_watch_cache_resource_version`, `watch_cache_capacity`, `apiserver_init_events_total`, `apiserver_terminated_watchers_total`, `watch_cache_capacity_increase_total`, `watch_cache_capacity_decrease_total`, `apiserver_watch_cache_read_wait_seconds`, `apiserver_watch_cache_consistent_read_total`, `apiserver_storage_consistency_checks_total`, `etcd_bookmark_counts`, `storage_decode_errors_total` extract the API group from `resource` label and put it in new `group` label. ([#131845](https://github.com/kubernetes/kubernetes/pull/131845), [@serathius](https://github.com/serathius)) [SIG API Machinery, Etcd, Instrumentation and Testing]
  - Kubelet:  removed the deprecated flag `--cloud-config` from the command line. ([#130161](https://github.com/kubernetes/kubernetes/pull/130161), [@carlory](https://github.com/carlory)) [SIG Cloud Provider, Node and Scalability]
  - Scheduling Framework exposes NodeInfos to the PreFilterPlugins.
  The PreFilterPlugins need to accept the NodeInfo list from the arguments. ([#130720](https://github.com/kubernetes/kubernetes/pull/130720), [@saintube](https://github.com/saintube)) [SIG Node, Scheduling, Storage and Testing]
 
## Changes by Kind

### Deprecation

- Deprecate preferences field in kubeconfig in favor of kuberc ([#131741](https://github.com/kubernetes/kubernetes/pull/131741), [@soltysh](https://github.com/soltysh)) [SIG API Machinery, CLI, Cluster Lifecycle and Testing]
- Kubeadm: consistently print an 'error: ' prefix before errors. ([#132080](https://github.com/kubernetes/kubernetes/pull/132080), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubeadm: only expose non-deprecated klog flags, 'v' and 'vmodule', to align with KEP https://features.k8s.io/2845 ([#131647](https://github.com/kubernetes/kubernetes/pull/131647), [@carsontham](https://github.com/carsontham)) [SIG Cluster Lifecycle]
- [cloud-provider] respect the "exclude-from-external-load-balancers=false" label ([#131085](https://github.com/kubernetes/kubernetes/pull/131085), [@kayrus](https://github.com/kayrus)) [SIG Cloud Provider and Network]

### API Change

- #### Additional documentation e.g., KEPs (Kubernetes Enhancement Proposals), usage docs, etc.:
  
  <!--
  This section can be blank if this pull request does not require a release note.
  
  When adding links which point to resources within git repositories, like
  KEPs or supporting documentation, please reference a specific commit and avoid
  linking directly to the master branch. This ensures that links reference a
  specific point in time, rather than a document that may change over time.
  
  See here for guidance on getting permanent links to files: https://help.github.com/en/articles/getting-permanent-links-to-files
  
  Please use the following format for linking documentation:
  - [KEP]: <link>
  - [Usage]: <link>
  - [Other doc]: <link>
  --> ([#131996](https://github.com/kubernetes/kubernetes/pull/131996), [@ritazh](https://github.com/ritazh)) [SIG Node and Testing]
- DRA API: resource.k8s.io/v1alpha3 now only contains DeviceTaintRule. All other types got removed because they became obsolete when introducing the v1beta1 API in 1.32.
  before updating a cluster where resourceclaims, resourceclaimtemplates, deviceclasses, or resourceslices might have been stored using Kubernetes < 1.32, delete all of those resources before updating and recreate them as needed while running Kubernetes >= 1.32. ([#132000](https://github.com/kubernetes/kubernetes/pull/132000), [@pohly](https://github.com/pohly)) [SIG Etcd, Node, Scheduling and Testing]
- Extends the nodeports scheduling plugin to consider hostPorts used by restartable init containers. ([#132040](https://github.com/kubernetes/kubernetes/pull/132040), [@avrittrohwer](https://github.com/avrittrohwer)) [SIG Scheduling and Testing]
- Kube-apiserver: Caching of authorization webhook decisions for authorized and unauthorized requests can now be disabled in the `--authorization-config` file by setting the new fields `cacheAuthorizedRequests` or `cacheUnauthorizedRequests` to `false` explicitly. See https://kubernetes.io/docs/reference/access-authn-authz/authorization/#using-configuration-file-for-authorization for more details. ([#129237](https://github.com/kubernetes/kubernetes/pull/129237), [@rfranzke](https://github.com/rfranzke)) [SIG API Machinery and Auth]
- Kube-apiserver: Promoted the `StructuredAuthenticationConfiguration` feature gate to GA. ([#131916](https://github.com/kubernetes/kubernetes/pull/131916), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-apiserver: the AuthenticationConfiguration type accepted in `--authentication-config` files has been promoted to `apiserver.config.k8s.io/v1`. ([#131752](https://github.com/kubernetes/kubernetes/pull/131752), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-log-runner: rotating log output into a new file when reaching a certain file size can be requested via the new `-log-file-size` parameter. `-log-file-age` enables automatical removal of old output files.  Periodic flushing can be requested through ` -flush-interval`. ([#127667](https://github.com/kubernetes/kubernetes/pull/127667), [@zylxjtu](https://github.com/zylxjtu)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Release, Scheduling, Storage, Testing and Windows]
- Kubectl: graduated `kuberc` support to beta. A `kuberc` configuration file provides a mechanism for customizing kubectl behavior (separate from kubeconfig, which configured cluster access across different clients). ([#131818](https://github.com/kubernetes/kubernetes/pull/131818), [@soltysh](https://github.com/soltysh)) [SIG CLI and Testing]
- Promote the RelaxedEnvironmentVariableValidation feature gate to GA and lock it in the default enabled state. ([#132054](https://github.com/kubernetes/kubernetes/pull/132054), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Apps, Architecture, Node and Testing]
- Remove inaccurate statement about requiring ports from pod spec hostNetwork field ([#130994](https://github.com/kubernetes/kubernetes/pull/130994), [@BenTheElder](https://github.com/BenTheElder)) [SIG Network and Node]
- TBD ([#131318](https://github.com/kubernetes/kubernetes/pull/131318), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Architecture, Auth, Etcd, Network and Testing]
- The validation of `replicas` field in the ReplicationController `/scale` subresource has been migrated to declarative validation.
  If the `DeclarativeValidation` feature gate is enabled, mismatches with existing validation are reported via metrics.
  If the `DeclarativeValidationTakeover` feature gate is enabled, declarative validation is the primary source of errors for migrated fields. ([#131664](https://github.com/kubernetes/kubernetes/pull/131664), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery and Apps]
- The validation-gen code generator generates validation code that supports validation ratcheting. ([#132236](https://github.com/kubernetes/kubernetes/pull/132236), [@yongruilin](https://github.com/yongruilin)) [SIG API Machinery, Apps, Auth and Node]
- Update etcd version to v3.6.0 ([#131501](https://github.com/kubernetes/kubernetes/pull/131501), [@joshjms](https://github.com/joshjms)) [SIG API Machinery, Cloud Provider, Cluster Lifecycle, Etcd and Testing]
- When the IsDNS1123SubdomainWithUnderscore function returns an error, it will return the correct regex information dns1123SubdomainFmtWithUnderscore. ([#132034](https://github.com/kubernetes/kubernetes/pull/132034), [@ChosenFoam](https://github.com/ChosenFoam)) [SIG Network]
- Zero-value `metadata.creationTimestamp` values are now omitted and no longer serialize an explicit `null` in JSON, YAML, and CBOR output ([#130989](https://github.com/kubernetes/kubernetes/pull/130989), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Etcd, Instrumentation, Network, Node, Scheduling, Storage and Testing]

### Feature

- Add a flag to `kubectl version` that detects whether a client/server version mismatch is outside the officially supported range. ([#127365](https://github.com/kubernetes/kubernetes/pull/127365), [@omerap12](https://github.com/omerap12)) [SIG CLI]
- Add support for CEL expressions with escaped names in structured authentication config.  Using `[` for accessing claims or user data is preferred when names contain characters that would need to be escaped.  CEL optionals via `?` can be used in places where `has` cannot be used, i.e. `claims[?"kubernetes.io"]` or `user.extra[?"domain.io/foo"]`. ([#131574](https://github.com/kubernetes/kubernetes/pull/131574), [@enj](https://github.com/enj)) [SIG API Machinery and Auth]
- Added Traffic Distribution field to `kubectl describe service` output ([#131491](https://github.com/kubernetes/kubernetes/pull/131491), [@tchap](https://github.com/tchap)) [SIG CLI]
- Added a `--show-swap` option to `kubectl top` subcommands ([#129458](https://github.com/kubernetes/kubernetes/pull/129458), [@iholder101](https://github.com/iholder101)) [SIG CLI]
- Added alpha metrics for compatibility versioning ([#131842](https://github.com/kubernetes/kubernetes/pull/131842), [@michaelasp](https://github.com/michaelasp)) [SIG API Machinery, Architecture, Instrumentation and Scheduling]
- Enabling completion for aliases defined in kuberc ([#131586](https://github.com/kubernetes/kubernetes/pull/131586), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Graduate ResilientWatchCacheInitialization to GA ([#131979](https://github.com/kubernetes/kubernetes/pull/131979), [@serathius](https://github.com/serathius)) [SIG API Machinery]
- Graduate configurable endpoints for anonymous authentication using the authentication configuration file to stable. ([#131654](https://github.com/kubernetes/kubernetes/pull/131654), [@vinayakankugoyal](https://github.com/vinayakankugoyal)) [SIG API Machinery and Testing]
- Graduated relaxed DNS search string validation to GA. For the Pod API, `.spec.dnsConfig.searches`
  now allows an underscore (`_`) where a dash (`-`) would be allowed, and it allows search strings be a single dot `.`. ([#132036](https://github.com/kubernetes/kubernetes/pull/132036), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Network and Testing]
- Graduated scheduler `QueueingHint` support to GA (general availability) ([#131973](https://github.com/kubernetes/kubernetes/pull/131973), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Kube-apiserver: Promoted `ExternalServiceAccountTokenSigner` feature to beta, which enables external signing of service account tokens and fetching of public verifying keys, by enabling the beta `ExternalServiceAccountTokenSigner` feature gate and specifying `--service-account-signing-endpoint`. The flag value can either be the location of a Unix domain socket on a filesystem, or be prefixed with an @ symbol and name a Unix domain socket in the abstract socket namespace. ([#131300](https://github.com/kubernetes/kubernetes/pull/131300), [@HarshalNeelkamal](https://github.com/HarshalNeelkamal)) [SIG API Machinery, Auth and Testing]
- Kube-controller-manager events to support contextual logging. ([#128351](https://github.com/kubernetes/kubernetes/pull/128351), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery]
- Kube-proxy: Check if IPv6 is available on Linux before using it ([#131265](https://github.com/kubernetes/kubernetes/pull/131265), [@rikatz](https://github.com/rikatz)) [SIG Network]
- Kubeadm: add support for ECDSA-P384 as an encryption algorithm type in v1beta4. ([#131677](https://github.com/kubernetes/kubernetes/pull/131677), [@lalitc375](https://github.com/lalitc375)) [SIG Cluster Lifecycle]
- Kubeadm: fixed issue where etcd member promotion fails with an error saying the member was already promoted ([#130782](https://github.com/kubernetes/kubernetes/pull/130782), [@BernardMC](https://github.com/BernardMC)) [SIG Cluster Lifecycle]
- Kubeadm: graduated the `NodeLocalCRISocket` feature gate to beta and enabed it by default. When its enabled, kubeadm will:
    1. Generate a `/var/lib/kubelet/instance-config.yaml` file to customize the `containerRuntimeEndpoint` field in per-node kubelet configurations.
    2. Remove the `kubeadm.alpha.kubernetes.io/cri-socket` annotation from nodes during upgrade operations.
    3. Remove the `--container-runtime-endpoint` flag from the `/var/lib/kubelet/kubeadm-flags.env` file during upgrades. ([#131981](https://github.com/kubernetes/kubernetes/pull/131981), [@HirazawaUi](https://github.com/HirazawaUi)) [SIG Cluster Lifecycle]
- Kubeadm: switched the validation check for Linux kernel version to throw warnings instead of errors. ([#131919](https://github.com/kubernetes/kubernetes/pull/131919), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
- Kubelet: the `--image-credential-provider-config` flag previously only accepted an individual file, but can now specify a directory path as well; when a directory is specified, all .json/.yaml/.yml files in the directory are loaded and merged in lexicographical order. ([#131658](https://github.com/kubernetes/kubernetes/pull/131658), [@dims](https://github.com/dims)) [SIG Auth and Node]
- Kubernetes api-server now merges selectors built from matchLabelKeys into the labelSelector of topologySpreadConstraints, 
  aligning Pod Topology Spread with the approach used by Inter-Pod Affinity.
  
  To avoid breaking existing pods that use matchLabelKeys, the current scheduler behavior will be preserved until it is removed in v1.34. 
  Therefore, do not upgrade your scheduler directly from v1.32 to v1.34. 
  Instead, upgrade step-by-step (from v1.32 to v1.33, then to v1.34), 
  ensuring that any pods created at v1.32 with matchLabelKeys are either removed or already scheduled by the time you reach v1.34.
  
  If you maintain controllers that previously relied on matchLabelKeys (for instance, to simulate scheduling), 
  you likely no longer need to handle matchLabelKeys directly. Instead, you can just rely on the labelSelector field going forward.
  
  Additionally, a new feature gate `MatchLabelKeysInPodTopologySpreadSelectorMerge`, which is enabled by default, has been 
  added to control this behavior. ([#129874](https://github.com/kubernetes/kubernetes/pull/129874), [@mochizuki875](https://github.com/mochizuki875)) [SIG Apps, Node, Scheduling and Testing]
- Kubernetes is now built using Go 1.24.3 ([#131934](https://github.com/kubernetes/kubernetes/pull/131934), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- Kubernetes is now built using Go 1.24.4 ([#132222](https://github.com/kubernetes/kubernetes/pull/132222), [@cpanato](https://github.com/cpanato)) [SIG Release and Testing]
- LeaseLocks can now have custom Labels that different holders will overwrite when they become the holder of the underlying lease. ([#131632](https://github.com/kubernetes/kubernetes/pull/131632), [@DerekFrank](https://github.com/DerekFrank)) [SIG API Machinery]
- Non-scheduling related errors (e.g., network errors) don't lengthen the Pod scheduling backoff time. ([#128748](https://github.com/kubernetes/kubernetes/pull/128748), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling and Testing]
- Promote feature OrderedNamespaceDeletion to GA. ([#131514](https://github.com/kubernetes/kubernetes/pull/131514), [@cici37](https://github.com/cici37)) [SIG API Machinery and Testing]
- Removed "endpoint-controller" and "workload-leader-election" FlowSchemas from the default APF configuration.
  
  migrate the lock type used in the leader election in your workloads from configmapsleases/endpointsleases to leases. ([#131215](https://github.com/kubernetes/kubernetes/pull/131215), [@tosi3k](https://github.com/tosi3k)) [SIG API Machinery, Apps, Network, Scalability and Scheduling]
- The PreferSameTrafficDistribution feature gate is now enabled by default,
  enabling the `PreferSameNode` traffic distribution value for Services. ([#132127](https://github.com/kubernetes/kubernetes/pull/132127), [@danwinship](https://github.com/danwinship)) [SIG Apps and Network]
- Updated the built in `system:monitoring` role with permission to access kubelet metrics endpoints. ([#132178](https://github.com/kubernetes/kubernetes/pull/132178), [@gavinkflam](https://github.com/gavinkflam)) [SIG Auth]

### Failing Test

- Kube-apiserver: The --service-account-signing-endpoint flag now only validates the format of abstract socket names ([#131509](https://github.com/kubernetes/kubernetes/pull/131509), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Auth]

### Bug or Regression

- Check for newer resize fields when deciding recovery feature's status in kubelet ([#131418](https://github.com/kubernetes/kubernetes/pull/131418), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- DRA: ResourceClaims requesting a fixed number of devices with `adminAccess` will no longer be allocated the same device multiple times. ([#131299](https://github.com/kubernetes/kubernetes/pull/131299), [@nojnhuh](https://github.com/nojnhuh)) [SIG Node]
- Disable reading of disk geometry before calling expansion for ext and xfs filesystems ([#131568](https://github.com/kubernetes/kubernetes/pull/131568), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Do not expand PVCs annotated with node-expand-not-required ([#131907](https://github.com/kubernetes/kubernetes/pull/131907), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Etcd, Node, Storage and Testing]
- Do not expand volume on the node, if controller expansion is finished ([#131868](https://github.com/kubernetes/kubernetes/pull/131868), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Do not log error event when waiting for expansion on the kubelet ([#131408](https://github.com/kubernetes/kubernetes/pull/131408), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Do not remove CSI json file if volume is already mounted on subsequent errors ([#131311](https://github.com/kubernetes/kubernetes/pull/131311), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Fix ReplicationController reconciliation when the DeploymentReplicaSetTerminatingReplicas feature gate is enabled ([#131822](https://github.com/kubernetes/kubernetes/pull/131822), [@atiratree](https://github.com/atiratree)) [SIG Apps]
- Fix a bug causing unexpected delay of creating pods for newly created jobs ([#132109](https://github.com/kubernetes/kubernetes/pull/132109), [@linxiulei](https://github.com/linxiulei)) [SIG Apps and Testing]
- Fix a bug in Job controller which could result in creating unnecessary Pods for a Job which is already
  recognized as finished (successful or failed). ([#130333](https://github.com/kubernetes/kubernetes/pull/130333), [@kmala](https://github.com/kmala)) [SIG Apps and Testing]
- Fix the allocatedResourceStatuses Field name mismatch in PVC status validation ([#131213](https://github.com/kubernetes/kubernetes/pull/131213), [@carlory](https://github.com/carlory)) [SIG Apps]
- Fixed a bug in CEL's common.UnstructuredToVal where `==` evaluates to false for identical objects when a field is present but the value is null.  This bug does not impact the Kubernetes API. ([#131559](https://github.com/kubernetes/kubernetes/pull/131559), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery]
- Fixed a bug that caused duplicate validation when updating a ReplicaSet. ([#131873](https://github.com/kubernetes/kubernetes/pull/131873), [@gavinkflam](https://github.com/gavinkflam)) [SIG Apps]
- Fixed a panic issue related to kubectl revision history kubernetes/kubectl#1724 ([#130503](https://github.com/kubernetes/kubernetes/pull/130503), [@tahacodes](https://github.com/tahacodes)) [SIG CLI]
- Fixed a possible deadlock in the watch client that could happen if the watch was not stopped. ([#131266](https://github.com/kubernetes/kubernetes/pull/131266), [@karlkfi](https://github.com/karlkfi)) [SIG API Machinery]
- Fixed an incorrect reference to `JoinConfigurationKind` in the error message when no ResetConfiguration is found during `kubeadm reset` with the `--config` flag. ([#132258](https://github.com/kubernetes/kubernetes/pull/132258), [@J3m3](https://github.com/J3m3)) [SIG Cluster Lifecycle]
- Fixed an issue where `insufficientResources` was logged as a pointer during pod preemption, making logs more readable. ([#132183](https://github.com/kubernetes/kubernetes/pull/132183), [@chrisy-x](https://github.com/chrisy-x)) [SIG Node]
- Fixed incorrect behavior for AllocationMode: All in ResourceClaim when used in subrequests. ([#131660](https://github.com/kubernetes/kubernetes/pull/131660), [@mortent](https://github.com/mortent)) [SIG Node]
- Fixed misleading response codes in admission control metrics. ([#132165](https://github.com/kubernetes/kubernetes/pull/132165), [@gavinkflam](https://github.com/gavinkflam)) [SIG API Machinery, Architecture and Instrumentation]
- Fixes an issue where Windows kube-proxy's ModifyLoadBalancer API updates did not match HNS state in version 15.4. ModifyLoadBalancer policy is supported from Kubernetes 1.31+. ([#131506](https://github.com/kubernetes/kubernetes/pull/131506), [@princepereira](https://github.com/princepereira)) [SIG Windows]
- HPA controller will no longer emit a 'FailedRescale' event if a scale operation initially fails due to a conflict but succeeds after a retry; a 'SuccessfulRescale' event will be emitted instead. A 'FailedRescale' event is still emitted if retries are exhausted. ([#132007](https://github.com/kubernetes/kubernetes/pull/132007), [@AumPatel1](https://github.com/AumPatel1)) [SIG Apps and Autoscaling]
- Improve error message when a pod with user namespaces is created and the runtime doesn't support user namespaces. ([#131623](https://github.com/kubernetes/kubernetes/pull/131623), [@rata](https://github.com/rata)) [SIG Node]
- Kube-apiserver: Fixes OIDC discovery document publishing when external service account token signing is enabled ([#131493](https://github.com/kubernetes/kubernetes/pull/131493), [@hoskeri](https://github.com/hoskeri)) [SIG API Machinery, Auth and Testing]
- Kube-apiserver: cronjob objects now default empty `spec.jobTemplate.spec.podFailurePolicy.rules[*].onPodConditions[*].status` fields as documented, avoiding validation failures during write requests. ([#131525](https://github.com/kubernetes/kubernetes/pull/131525), [@carlory](https://github.com/carlory)) [SIG Apps]
- Kube-proxy:  Remove iptables cli wait interval flag ([#131961](https://github.com/kubernetes/kubernetes/pull/131961), [@cyclinder](https://github.com/cyclinder)) [SIG Network]
- Kube-scheduler: in Kubernetes 1.33, the number of devices that can be allocated per ResourceClaim was accidentally reduced to 16. Now the supported number of devices per ResourceClaim is 32 again. ([#131662](https://github.com/kubernetes/kubernetes/pull/131662), [@mortent](https://github.com/mortent)) [SIG Node]
- Kubelet: close a loophole where static pods could reference arbitrary ResourceClaims. The pods created by the kubelet then don't run due to a sanity check, but such references shouldn't be allowed regardless. ([#131844](https://github.com/kubernetes/kubernetes/pull/131844), [@pohly](https://github.com/pohly)) [SIG Apps, Auth and Node]
- Kubelet: fix a bug where the unexpected NodeResizeError condition was in PVC status when the csi driver does not support node volume expansion and the pvc has the ReadWriteMany access mode. ([#131495](https://github.com/kubernetes/kubernetes/pull/131495), [@carlory](https://github.com/carlory)) [SIG Storage]
- Reduce 5s delay of tainting `node.kubernetes.io/unreachable:NoExecute` when a Node becomes unreachable ([#120816](https://github.com/kubernetes/kubernetes/pull/120816), [@tnqn](https://github.com/tnqn)) [SIG Apps and Node]
- Skip pod backoff completely when PodMaxBackoffDuration kube-scheduler option is set to zero and SchedulerPopFromBackoffQ feature gate is enabled. ([#131965](https://github.com/kubernetes/kubernetes/pull/131965), [@macsko](https://github.com/macsko)) [SIG Scheduling]
- The shorthand for --output flag in kubectl explain was accidentally deleted, but has been added back. ([#131962](https://github.com/kubernetes/kubernetes/pull/131962), [@superbrothers](https://github.com/superbrothers)) [SIG CLI]
- `kubectl create|delete|get|replace --raw` commands now honor server root paths specified in the kubeconfig file. ([#131165](https://github.com/kubernetes/kubernetes/pull/131165), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]

### Other (Cleanup or Flake)

- Added a warning to `kubectl attach`, notifying / reminding users that commands and output are available via the `log` subresource of that Pod. ([#127183](https://github.com/kubernetes/kubernetes/pull/127183), [@mochizuki875](https://github.com/mochizuki875)) [SIG Auth, CLI, Node and Security]
- Bump cel-go dependency to v0.25.0. The changeset is available at: https://github.com/google/cel-go/compare/v0.23.2...v0.25.0 ([#131444](https://github.com/kubernetes/kubernetes/pull/131444), [@erdii](https://github.com/erdii)) [SIG API Machinery, Auth, Cloud Provider and Node]
- Bump kube dns to v1.26.4 ([#132012](https://github.com/kubernetes/kubernetes/pull/132012), [@pacoxu](https://github.com/pacoxu)) [SIG Cloud Provider]
- By default the binaries like kube-apiserver are built with "grpcnotrace" tag enabled. Please use DBG flag if you want to enable golang tracing. ([#132210](https://github.com/kubernetes/kubernetes/pull/132210), [@dims](https://github.com/dims)) [SIG Architecture]
- Changed apiserver to treat failures decoding a mutating webhook patch as failures to call the webhook so they trigger the webhook failurePolicy and count against metrics like `webhook_fail_open_count` ([#131627](https://github.com/kubernetes/kubernetes/pull/131627), [@dims](https://github.com/dims)) [SIG API Machinery]
- DRA kubelet: logging now uses `driverName` like the rest of the Kubernetes components, instead of `pluginName`. ([#132096](https://github.com/kubernetes/kubernetes/pull/132096), [@pohly](https://github.com/pohly)) [SIG Node and Testing]
- DRA kubelet: recovery from mistakes like scheduling a pod onto a node with the required driver not running is a bit simpler now because the kubelet does not block pod deletion unnecessarily. ([#131968](https://github.com/kubernetes/kubernetes/pull/131968), [@pohly](https://github.com/pohly)) [SIG Node and Testing]
- Fixed some missing white spaces in the flag descriptions and logs. ([#131562](https://github.com/kubernetes/kubernetes/pull/131562), [@logica0419](https://github.com/logica0419)) [SIG Network]
- Hack/update-codegen.sh now automatically ensures goimports and protoc ([#131459](https://github.com/kubernetes/kubernetes/pull/131459), [@BenTheElder](https://github.com/BenTheElder)) [SIG API Machinery]
- Kube-apiserver: removed the deprecated `apiserver_encryption_config_controller_automatic_reload_success_total` and `apiserver_encryption_config_controller_automatic_reload_failure_total` metrics in favor of `apiserver_encryption_config_controller_automatic_reloads_total`. ([#132238](https://github.com/kubernetes/kubernetes/pull/132238), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Testing]
- Kube-scheduler: removed the deprecated scheduler_scheduler_cache_size metric  in favor of scheduler_cache_size ([#131425](https://github.com/kubernetes/kubernetes/pull/131425), [@carlory](https://github.com/carlory)) [SIG Scheduling]
- Kubeadm: fixed missing space when printing the warning about pause image mismatch. ([#131563](https://github.com/kubernetes/kubernetes/pull/131563), [@logica0419](https://github.com/logica0419)) [SIG Cluster Lifecycle]
- Kubeadm: made the coredns deployment manifest use named ports consistently for the liveness and readiness probes. ([#131587](https://github.com/kubernetes/kubernetes/pull/131587), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle]
- Kubectl interactive delete: treat empty newline input as N ([#132251](https://github.com/kubernetes/kubernetes/pull/132251), [@ardaguclu](https://github.com/ardaguclu)) [SIG CLI]
- Migrate pkg/kubelet/status to contextual logging ([#130852](https://github.com/kubernetes/kubernetes/pull/130852), [@Chulong-Li](https://github.com/Chulong-Li)) [SIG Node]
- Promote `apiserver_authentication_config_controller_automatic_reloads_total` and `apiserver_authentication_config_controller_automatic_reload_last_timestamp_seconds` metrics to BETA. ([#131798](https://github.com/kubernetes/kubernetes/pull/131798), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Instrumentation]
- Promote `apiserver_authorization_config_controller_automatic_reloads_total` and `apiserver_authorization_config_controller_automatic_reload_last_timestamp_seconds` metrics to BETA. ([#131768](https://github.com/kubernetes/kubernetes/pull/131768), [@aramase](https://github.com/aramase)) [SIG API Machinery, Auth and Instrumentation]
- Promoted the `SeparateTaintEvictionController` feature gate to GA; it is now enabled unconditionally. ([#122634](https://github.com/kubernetes/kubernetes/pull/122634), [@carlory](https://github.com/carlory)) [SIG API Machinery, Apps, Node and Testing]
- Removed generally available feature-gate `PodDisruptionConditions`. ([#129501](https://github.com/kubernetes/kubernetes/pull/129501), [@carlory](https://github.com/carlory)) [SIG Apps]
- Removes support for API streaming from the `List() method` of the dynamic client. ([#132229](https://github.com/kubernetes/kubernetes/pull/132229), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery, CLI and Testing]
- Removes support for API streaming from the `List() method` of the metadata client. ([#132149](https://github.com/kubernetes/kubernetes/pull/132149), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Removes support for API streaming from the `List() method` of the typed client. ([#132257](https://github.com/kubernetes/kubernetes/pull/132257), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery and Testing]
- Removes support for API streaming from the rest client. ([#132285](https://github.com/kubernetes/kubernetes/pull/132285), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- Types: CycleState, StateData, StateKey and ErrNotFound moved from pkg/scheduler/framework to k8s.io/kube-scheduler/framework.
  Type CycleState that is passed to each plugin in scheduler framework is changed to the new interface CycleState (in k8s.io/kube-scheduler/framework) ([#131887](https://github.com/kubernetes/kubernetes/pull/131887), [@ania-borowiec](https://github.com/ania-borowiec)) [SIG Node, Scheduling, Storage and Testing]
- Updated CNI plugins to v1.7.1 ([#131602](https://github.com/kubernetes/kubernetes/pull/131602), [@adrianmoisey](https://github.com/adrianmoisey)) [SIG Cloud Provider, Node and Testing]
- Updated cri-tools to v1.33.0. ([#131406](https://github.com/kubernetes/kubernetes/pull/131406), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider]
- Upgrade CoreDNS to v1.12.1 ([#131151](https://github.com/kubernetes/kubernetes/pull/131151), [@yashsingh74](https://github.com/yashsingh74)) [SIG Cloud Provider and Cluster Lifecycle]

## Dependencies

### Added
- buf.build/gen/go/bufbuild/protovalidate/protocolbuffers/go: 63bb56e
- github.com/GoogleCloudPlatform/opentelemetry-operations-go/detectors/gcp: [v1.26.0](https://github.com/GoogleCloudPlatform/opentelemetry-operations-go/tree/detectors/gcp/v1.26.0)
- github.com/bufbuild/protovalidate-go: [v0.9.1](https://github.com/bufbuild/protovalidate-go/tree/v0.9.1)
- github.com/envoyproxy/go-control-plane/envoy: [v1.32.4](https://github.com/envoyproxy/go-control-plane/tree/envoy/v1.32.4)
- github.com/envoyproxy/go-control-plane/ratelimit: [v0.1.0](https://github.com/envoyproxy/go-control-plane/tree/ratelimit/v0.1.0)
- github.com/go-jose/go-jose/v4: [v4.0.4](https://github.com/go-jose/go-jose/tree/v4.0.4)
- github.com/golang-jwt/jwt/v5: [v5.2.2](https://github.com/golang-jwt/jwt/tree/v5.2.2)
- github.com/grpc-ecosystem/go-grpc-middleware/providers/prometheus: [v1.0.1](https://github.com/grpc-ecosystem/go-grpc-middleware/tree/providers/prometheus/v1.0.1)
- github.com/grpc-ecosystem/go-grpc-middleware/v2: [v2.3.0](https://github.com/grpc-ecosystem/go-grpc-middleware/tree/v2.3.0)
- github.com/spiffe/go-spiffe/v2: [v2.5.0](https://github.com/spiffe/go-spiffe/tree/v2.5.0)
- github.com/zeebo/errs: [v1.4.0](https://github.com/zeebo/errs/tree/v1.4.0)
- go.etcd.io/raft/v3: v3.6.0
- go.opentelemetry.io/contrib/detectors/gcp: v1.34.0
- go.opentelemetry.io/otel/sdk/metric: v1.34.0

### Changed
- cel.dev/expr: v0.19.1 → v0.23.1
- cloud.google.com/go/compute/metadata: v0.5.0 → v0.6.0
- github.com/Microsoft/hnslib: [v0.0.8 → v0.1.1](https://github.com/Microsoft/hnslib/compare/v0.0.8...v0.1.1)
- github.com/cncf/xds/go: [b4127c9 → 2f00578](https://github.com/cncf/xds/compare/b4127c9...2f00578)
- github.com/coredns/corefile-migration: [v1.0.25 → v1.0.26](https://github.com/coredns/corefile-migration/compare/v1.0.25...v1.0.26)
- github.com/cpuguy83/go-md2man/v2: [v2.0.4 → v2.0.6](https://github.com/cpuguy83/go-md2man/compare/v2.0.4...v2.0.6)
- github.com/envoyproxy/go-control-plane: [v0.13.0 → v0.13.4](https://github.com/envoyproxy/go-control-plane/compare/v0.13.0...v0.13.4)
- github.com/envoyproxy/protoc-gen-validate: [v1.1.0 → v1.2.1](https://github.com/envoyproxy/protoc-gen-validate/compare/v1.1.0...v1.2.1)
- github.com/fsnotify/fsnotify: [v1.7.0 → v1.9.0](https://github.com/fsnotify/fsnotify/compare/v1.7.0...v1.9.0)
- github.com/fxamacker/cbor/v2: [v2.7.0 → v2.8.0](https://github.com/fxamacker/cbor/compare/v2.7.0...v2.8.0)
- github.com/golang/glog: [v1.2.2 → v1.2.4](https://github.com/golang/glog/compare/v1.2.2...v1.2.4)
- github.com/google/cel-go: [v0.23.2 → v0.25.0](https://github.com/google/cel-go/compare/v0.23.2...v0.25.0)
- github.com/grpc-ecosystem/grpc-gateway/v2: [v2.24.0 → v2.26.3](https://github.com/grpc-ecosystem/grpc-gateway/compare/v2.24.0...v2.26.3)
- github.com/ishidawataru/sctp: [7ff4192 → ae8eb7f](https://github.com/ishidawataru/sctp/compare/7ff4192...ae8eb7f)
- github.com/jonboulle/clockwork: [v0.4.0 → v0.5.0](https://github.com/jonboulle/clockwork/compare/v0.4.0...v0.5.0)
- github.com/modern-go/reflect2: [v1.0.2 → 35a7c28](https://github.com/modern-go/reflect2/compare/v1.0.2...35a7c28)
- github.com/spf13/cobra: [v1.8.1 → v1.9.1](https://github.com/spf13/cobra/compare/v1.8.1...v1.9.1)
- github.com/spf13/pflag: [v1.0.5 → v1.0.6](https://github.com/spf13/pflag/compare/v1.0.5...v1.0.6)
- github.com/vishvananda/netlink: [62fb240 → v1.3.1](https://github.com/vishvananda/netlink/compare/62fb240...v1.3.1)
- github.com/vishvananda/netns: [v0.0.4 → v0.0.5](https://github.com/vishvananda/netns/compare/v0.0.4...v0.0.5)
- go.etcd.io/bbolt: v1.3.11 → v1.4.0
- go.etcd.io/etcd/api/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/client/pkg/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/client/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/pkg/v3: v3.5.21 → v3.6.1
- go.etcd.io/etcd/server/v3: v3.5.21 → v3.6.1
- go.etcd.io/gofail: v0.1.0 → v0.2.0
- go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful: v0.42.0 → v0.44.0
- go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc: v0.58.0 → v0.60.0
- go.opentelemetry.io/contrib/propagators/b3: v1.17.0 → v1.19.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc: v1.33.0 → v1.34.0
- go.opentelemetry.io/otel/exporters/otlp/otlptrace: v1.33.0 → v1.34.0
- go.opentelemetry.io/otel/metric: v1.33.0 → v1.35.0
- go.opentelemetry.io/otel/sdk: v1.33.0 → v1.34.0
- go.opentelemetry.io/otel/trace: v1.33.0 → v1.35.0
- go.opentelemetry.io/otel: v1.33.0 → v1.35.0
- go.opentelemetry.io/proto/otlp: v1.4.0 → v1.5.0
- google.golang.org/genproto/googleapis/api: e6fa225 → a0af3ef
- google.golang.org/genproto/googleapis/rpc: e6fa225 → a0af3ef
- google.golang.org/grpc: v1.68.1 → v1.72.1
- k8s.io/gengo/v2: 1244d31 → 85fd79d
- k8s.io/system-validators: v1.9.1 → v1.10.1
- k8s.io/utils: 3ea5e8c → 4c0f3b2
- sigs.k8s.io/structured-merge-diff/v4: v4.6.0 → v4.7.0

### Removed
- cloud.google.com/go/accessapproval: v1.7.4
- cloud.google.com/go/accesscontextmanager: v1.8.4
- cloud.google.com/go/aiplatform: v1.58.0
- cloud.google.com/go/analytics: v0.22.0
- cloud.google.com/go/apigateway: v1.6.4
- cloud.google.com/go/apigeeconnect: v1.6.4
- cloud.google.com/go/apigeeregistry: v0.8.2
- cloud.google.com/go/appengine: v1.8.4
- cloud.google.com/go/area120: v0.8.4
- cloud.google.com/go/artifactregistry: v1.14.6
- cloud.google.com/go/asset: v1.17.0
- cloud.google.com/go/assuredworkloads: v1.11.4
- cloud.google.com/go/automl: v1.13.4
- cloud.google.com/go/baremetalsolution: v1.2.3
- cloud.google.com/go/batch: v1.7.0
- cloud.google.com/go/beyondcorp: v1.0.3
- cloud.google.com/go/bigquery: v1.58.0
- cloud.google.com/go/billing: v1.18.0
- cloud.google.com/go/binaryauthorization: v1.8.0
- cloud.google.com/go/certificatemanager: v1.7.4
- cloud.google.com/go/channel: v1.17.4
- cloud.google.com/go/cloudbuild: v1.15.0
- cloud.google.com/go/clouddms: v1.7.3
- cloud.google.com/go/cloudtasks: v1.12.4
- cloud.google.com/go/compute: v1.23.3
- cloud.google.com/go/contactcenterinsights: v1.12.1
- cloud.google.com/go/container: v1.29.0
- cloud.google.com/go/containeranalysis: v0.11.3
- cloud.google.com/go/datacatalog: v1.19.2
- cloud.google.com/go/dataflow: v0.9.4
- cloud.google.com/go/dataform: v0.9.1
- cloud.google.com/go/datafusion: v1.7.4
- cloud.google.com/go/datalabeling: v0.8.4
- cloud.google.com/go/dataplex: v1.14.0
- cloud.google.com/go/dataproc/v2: v2.3.0
- cloud.google.com/go/dataqna: v0.8.4
- cloud.google.com/go/datastore: v1.15.0
- cloud.google.com/go/datastream: v1.10.3
- cloud.google.com/go/deploy: v1.17.0
- cloud.google.com/go/dialogflow: v1.48.1
- cloud.google.com/go/dlp: v1.11.1
- cloud.google.com/go/documentai: v1.23.7
- cloud.google.com/go/domains: v0.9.4
- cloud.google.com/go/edgecontainer: v1.1.4
- cloud.google.com/go/errorreporting: v0.3.0
- cloud.google.com/go/essentialcontacts: v1.6.5
- cloud.google.com/go/eventarc: v1.13.3
- cloud.google.com/go/filestore: v1.8.0
- cloud.google.com/go/firestore: v1.14.0
- cloud.google.com/go/functions: v1.15.4
- cloud.google.com/go/gkebackup: v1.3.4
- cloud.google.com/go/gkeconnect: v0.8.4
- cloud.google.com/go/gkehub: v0.14.4
- cloud.google.com/go/gkemulticloud: v1.1.0
- cloud.google.com/go/gsuiteaddons: v1.6.4
- cloud.google.com/go/iam: v1.1.5
- cloud.google.com/go/iap: v1.9.3
- cloud.google.com/go/ids: v1.4.4
- cloud.google.com/go/iot: v1.7.4
- cloud.google.com/go/kms: v1.15.5
- cloud.google.com/go/language: v1.12.2
- cloud.google.com/go/lifesciences: v0.9.4
- cloud.google.com/go/logging: v1.9.0
- cloud.google.com/go/longrunning: v0.5.4
- cloud.google.com/go/managedidentities: v1.6.4
- cloud.google.com/go/maps: v1.6.3
- cloud.google.com/go/mediatranslation: v0.8.4
- cloud.google.com/go/memcache: v1.10.4
- cloud.google.com/go/metastore: v1.13.3
- cloud.google.com/go/monitoring: v1.17.0
- cloud.google.com/go/networkconnectivity: v1.14.3
- cloud.google.com/go/networkmanagement: v1.9.3
- cloud.google.com/go/networksecurity: v0.9.4
- cloud.google.com/go/notebooks: v1.11.2
- cloud.google.com/go/optimization: v1.6.2
- cloud.google.com/go/orchestration: v1.8.4
- cloud.google.com/go/orgpolicy: v1.12.0
- cloud.google.com/go/osconfig: v1.12.4
- cloud.google.com/go/oslogin: v1.13.0
- cloud.google.com/go/phishingprotection: v0.8.4
- cloud.google.com/go/policytroubleshooter: v1.10.2
- cloud.google.com/go/privatecatalog: v0.9.4
- cloud.google.com/go/pubsub: v1.34.0
- cloud.google.com/go/pubsublite: v1.8.1
- cloud.google.com/go/recaptchaenterprise/v2: v2.9.0
- cloud.google.com/go/recommendationengine: v0.8.4
- cloud.google.com/go/recommender: v1.12.0
- cloud.google.com/go/redis: v1.14.1
- cloud.google.com/go/resourcemanager: v1.9.4
- cloud.google.com/go/resourcesettings: v1.6.4
- cloud.google.com/go/retail: v1.14.4
- cloud.google.com/go/run: v1.3.3
- cloud.google.com/go/scheduler: v1.10.5
- cloud.google.com/go/secretmanager: v1.11.4
- cloud.google.com/go/security: v1.15.4
- cloud.google.com/go/securitycenter: v1.24.3
- cloud.google.com/go/servicedirectory: v1.11.3
- cloud.google.com/go/shell: v1.7.4
- cloud.google.com/go/spanner: v1.55.0
- cloud.google.com/go/speech: v1.21.0
- cloud.google.com/go/storagetransfer: v1.10.3
- cloud.google.com/go/talent: v1.6.5
- cloud.google.com/go/texttospeech: v1.7.4
- cloud.google.com/go/tpu: v1.6.4
- cloud.google.com/go/trace: v1.10.4
- cloud.google.com/go/translate: v1.10.0
- cloud.google.com/go/video: v1.20.3
- cloud.google.com/go/videointelligence: v1.11.4
- cloud.google.com/go/vision/v2: v2.7.5
- cloud.google.com/go/vmmigration: v1.7.4
- cloud.google.com/go/vmwareengine: v1.0.3
- cloud.google.com/go/vpcaccess: v1.7.4
- cloud.google.com/go/webrisk: v1.9.4
- cloud.google.com/go/websecurityscanner: v1.6.4
- cloud.google.com/go/workflows: v1.12.3
- cloud.google.com/go: v0.112.0
- github.com/BurntSushi/toml: [v0.3.1](https://github.com/BurntSushi/toml/tree/v0.3.1)
- github.com/census-instrumentation/opencensus-proto: [v0.4.1](https://github.com/census-instrumentation/opencensus-proto/tree/v0.4.1)
- github.com/client9/misspell: [v0.3.4](https://github.com/client9/misspell/tree/v0.3.4)
- github.com/cncf/udpa/go: [269d4d4](https://github.com/cncf/udpa/tree/269d4d4)
- github.com/ghodss/yaml: [v1.0.0](https://github.com/ghodss/yaml/tree/v1.0.0)
- github.com/go-kit/kit: [v0.9.0](https://github.com/go-kit/kit/tree/v0.9.0)
- github.com/go-logfmt/logfmt: [v0.4.0](https://github.com/go-logfmt/logfmt/tree/v0.4.0)
- github.com/go-stack/stack: [v1.8.0](https://github.com/go-stack/stack/tree/v1.8.0)
- github.com/golang-jwt/jwt/v4: [v4.5.2](https://github.com/golang-jwt/jwt/tree/v4.5.2)
- github.com/golang/mock: [v1.1.1](https://github.com/golang/mock/tree/v1.1.1)
- github.com/grpc-ecosystem/grpc-gateway: [v1.16.0](https://github.com/grpc-ecosystem/grpc-gateway/tree/v1.16.0)
- github.com/konsorten/go-windows-terminal-sequences: [v1.0.1](https://github.com/konsorten/go-windows-terminal-sequences/tree/v1.0.1)
- github.com/kr/logfmt: [b84e30a](https://github.com/kr/logfmt/tree/b84e30a)
- github.com/opentracing/opentracing-go: [v1.1.0](https://github.com/opentracing/opentracing-go/tree/v1.1.0)
- go.etcd.io/etcd/client/v2: v2.305.21
- go.etcd.io/etcd/raft/v3: v3.5.21
- go.uber.org/atomic: v1.7.0
- golang.org/x/lint: d0100b6
- google.golang.org/appengine: v1.4.0
- google.golang.org/genproto: ef43131
- honnef.co/go/tools: ea95bdf