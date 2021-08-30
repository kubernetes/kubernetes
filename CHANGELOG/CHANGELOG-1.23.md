<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.23.0-alpha.1](#v1230-alpha1)
  - [Downloads for v1.23.0-alpha.1](#downloads-for-v1230-alpha1)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.22.0](#changelog-since-v1220)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Documentation](#documentation)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)

<!-- END MUNGE: GENERATED_TOC -->

# v1.23.0-alpha.1


## Downloads for v1.23.0-alpha.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes.tar.gz) | f7c76f1e077b5d98019347b2c9b79eaa0c79d428542b9c15dab23886c276ca16314f200ca37af914c52264c0e1e5d0bde639d6adf37368d5e7b29d230df00d95
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-src.tar.gz) | f267f26eca20cd7018e68abeeed38aed5c10dbbae7c531c4e08e507196a4dd3f511eb8d41ee8b09495544337d8e1940a8ca04e94084f8dd172698a96564fb070

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-darwin-amd64.tar.gz) | deb110839c2c3cf94ca9b29df2f0b07b3fad6937d7bb6e9d2516d01345c8e324f6ab86fe1d34f1443f04c3d1fc328b53b3d756c295f4ed22f1994071fbc8c9cb
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-darwin-arm64.tar.gz) | 1473cb9fc4847b0daff6c9e3189ce55fadc22fb6190161e744e5438066a714cb467fdebfb35f6445a27f5010df94ee602fff492a2382e0f308fda111d53af1f4
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-386.tar.gz) | ed5f5b0777ca51790d185764afc2c812f82ae27c35d897570fc86cabee90dc0a445d9d8c37c981bd3684ba9cd47dc0d75d0094578e79ef7b591d3c1b6564280f
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-amd64.tar.gz) | 39f2a888e7a43c9e4a4018301894786f6babe23d79ab7a143e06444f69bc14aec2e158d355c5b48da4356e7bd72ec9b1268f8b12815c8b709395f36ad9a68a2f
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-arm.tar.gz) | b6b8333d8adb4bc6a943bcd2c6cd1a0aeaf0b926d06aa03b759e3c723c81ccc91804debc64fedcd7d678eefdee9bdacc52b2891bd084a15fd5f7918a70e51a15
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-arm64.tar.gz) | 3cb8217b9a5363cebad4989253e02c8a37259b61eafc2f08681508c11c5f68448cad43282257c3d90ad510cc9a62645b7f1adeb99fedf5e13c181495e3754ee4
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-ppc64le.tar.gz) | e411700fb13b25deca6347983cdafe47199f0df00086ccd7b3e7d52a7b3bee7e96a85c2568dd52c956fd4ea8b4a6991859c57c9b73a13e06440b456c65b11687
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-linux-s390x.tar.gz) | 6c1395792a175de77436352d0893476363497b0f6a616f4415f91aed5e780d1f25b515021939a7563046237c7b651caba0d1fbf7c4c461677d1b9308b227e94c
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-windows-386.tar.gz) | f3aec7136c21d24a99145ce294a859078fcbf11bae132b8b4081555a6656c0d95ccbaca02a86dc257d557ecebc0673d0771b9cdd10593712a643e8cc0f61d681
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-client-windows-amd64.tar.gz) | b29697ba0a25f3d871ffbe5800dcb23ec9fd27c0122a284e17c21f1258f7dd9d341813aeb7826159c7999581a16db19fbb6eeeab48f5c89975df7595d19102c3

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-amd64.tar.gz) | a5b3edca559b84cd9d22b43b23d0607951d434e185dcb313b831604d83dd306cfc017599994d3944ce77360116024eb59a302851325bb2c29c185a80db2e6eac
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-arm.tar.gz) | 2334dbcff3ba22a50f252998eb63991b6c816659dbaa5f749370fc1b1f78f0af7739e50ab64c14a23c4e7dfa8917568e2a3b85bdffdb2cc691ee23ae8f5c8326
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-arm64.tar.gz) | 58674443ce6e359a995dd7c4289bf730e616bcaf336837b77333a206d4e98693d9356a0a670ffbe0b274e2997a8b76a164153cf084f0ff5f91f40f00b5512684
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-ppc64le.tar.gz) | f60ebdd04e2348b1ba51540cad93fa24cb133fd25db97150000bffaff8ccb41e1b6506bcde6b7d913aee7701478f975a97775430a82980105383fdb1cc13d260
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-server-linux-s390x.tar.gz) | ff008aa0ba1bf755f32c7251c6aceb12b6f9de00d2e2729302b51960e70e486bd82da62d21d70ad81c14e01910ab2afe0fd2509ebfdec050d36f88ee1f0330b2

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-amd64.tar.gz) | 352502f10fbc4579bd9556e3f73ca7513184371ea563d12a39d655d39bb14ccf0f485f4f2b54a77d984c91ff0de2acea7225f98532a1247da5b9ecc65081bc1a
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-arm.tar.gz) | af9de95e2b9e4c1f39cb9757d4dca020f7d276b6702302a2d92e7a93e9986528615ce54531e62b96f6e8a0b9863cddbb264f42b1f59374948ac3499af60d9532
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-arm64.tar.gz) | 45a286cb1d469b16d046af02047cf63a8407222e4a39fe696f5652e0587e0c9ffbdbab6505ce85e2726ba10db3189a7fbe70e316bc610caedc8cbb49fed28076
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-ppc64le.tar.gz) | 7a540a3ff0295998a1679b0ccd50cb1825faf1d0afd6ed08138ab3767c83a2743aa43b122c8da89ee00161f57c0af8d76012e890f9fe6d77b4ee8aff4e32e50f
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-linux-s390x.tar.gz) | 3cd7656221ac2fa161abcf237878cff26c1d97cf77d9b784736c97a56841397ff859e43947d81a83f8fe4164701da41a1dad69b551c4e1fee49b3f8196878236
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.1/kubernetes-node-windows-amd64.tar.gz) | 21e63913024e88a48244a598cd400fbae6ce8f8910202f1b635812fbc9281b7c6097eb10a321dd18846484a198845bba58970d83b5119a367862cf8418d4d08c

## Changelog since v1.22.0

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

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
  --> ([#104389](https://github.com/kubernetes/kubernetes/pull/104389), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
  - Kubeadm: remove the deprecated flag --experimental-patches for the init|join|upgrade commands. The flag --patches is no longer allowed in a mixture with the flag --config. Please use the kubeadm configuration for setting patches for a node using {Init|Join}Configuration.patches. ([#104065](https://github.com/kubernetes/kubernetes/pull/104065), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
 
## Changes by Kind

### Deprecation

- Add apiserver_longrunning_requests metric to replace the soon to be deprecated apiserver_longrunning_gauge metric. ([#103799](https://github.com/kubernetes/kubernetes/pull/103799), [@jyz0309](https://github.com/jyz0309)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Kubeadm: remove the --port flag from the manifest for the kube-controller-manager since the flag has been a NO-OP since 1.22 and insecure serving was removed for the component. ([#104157](https://github.com/kubernetes/kubernetes/pull/104157), [@knight42](https://github.com/knight42)) [SIG Cluster Lifecycle]

### API Change

- CSIDriver.Spec.StorageCapacity can now be modified. ([#101789](https://github.com/kubernetes/kubernetes/pull/101789), [@pohly](https://github.com/pohly)) [SIG Storage]
- Kube-apiserver: The `rbac.authorization.k8s.io/v1alpha1` API version is removed; use the `rbac.authorization.k8s.io/v1` API, available since v1.8. The `scheduling.k8s.io/v1alpha1` API version is removed; use the `scheduling.k8s.io/v1` API, available since v1.14. ([#104248](https://github.com/kubernetes/kubernetes/pull/104248), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, Network and Testing]
- Kube-controller-manager supports '--concurrent-ephemeralvolume-syncs' flag to set the number of ephemeral volume controller workers. ([#102981](https://github.com/kubernetes/kubernetes/pull/102981), [@SataQiu](https://github.com/SataQiu)) [SIG API Machinery and Apps]

### Feature

- Adding support for multiple --from-env-file flags ([#101646](https://github.com/kubernetes/kubernetes/pull/101646), [@lauchokyip](https://github.com/lauchokyip)) [SIG CLI]
- All folks to build kubernetes with a custom kube-cross image ([#104185](https://github.com/kubernetes/kubernetes/pull/104185), [@dims](https://github.com/dims)) [SIG Release and Testing]
- Allow node expansion of local volumes ([#102886](https://github.com/kubernetes/kubernetes/pull/102886), [@gnufied](https://github.com/gnufied)) [SIG Storage and Testing]
- Client-go event library allows customizing spam filtering function. 
  It is now possible to override `SpamKeyFunc`, which is used by event filtering to detect spam in the events. ([#103918](https://github.com/kubernetes/kubernetes/pull/103918), [@olagacek](https://github.com/olagacek)) [SIG API Machinery and Instrumentation]
- Constants/variables from k8s.io for STABLE metrics is now supported ([#103654](https://github.com/kubernetes/kubernetes/pull/103654), [@coffeepac](https://github.com/coffeepac)) [SIG Auth, Instrumentation, Node and Testing]
- Display Labels when kubectl describe ingress ([#103894](https://github.com/kubernetes/kubernetes/pull/103894), [@kabab](https://github.com/kabab)) [SIG CLI]
- Expose a `NewUnstructuredExtractor` from apply configurations `meta/v1` package that enables extracting objects into unstructured apply configurations ([#103564](https://github.com/kubernetes/kubernetes/pull/103564), [@kevindelgado](https://github.com/kevindelgado)) [SIG API Machinery, Cluster Lifecycle, Release and Testing]
- Introduce a feature gate DisableKubeletCloudCredentialProviders which allows disabling the in-tree kubelet credential providers.
  
  The DisableKubeletCloudCredentialProviders FeatureGate is currently in Alpha, which means is currently disabled by default. Once the FeatureGate moves to beta, in-tree credential providers will be disabled by default, and users will need to migrate to using external credential providers. ([#102507](https://github.com/kubernetes/kubernetes/pull/102507), [@ostrain](https://github.com/ostrain)) [SIG Cloud Provider]
- Introduces a new metric: admission_webhook_request_total with the following labels: name (string) - the webhook name, type (string) - the admission type, operation (string) - the requested verb, code (int) - the HTTP status code, rejected (bool) - whether the request was rejected, namespace (string) - the namespace of the requested resource. ([#103162](https://github.com/kubernetes/kubernetes/pull/103162), [@rmoriar1](https://github.com/rmoriar1)) [SIG API Machinery and Instrumentation]
- Kube-up.sh installs csi-proxy v1.0.1-gke.0 ([#104426](https://github.com/kubernetes/kubernetes/pull/104426), [@mauriciopoppe](https://github.com/mauriciopoppe)) [SIG Cloud Provider, Storage and Windows]
- Kubeadm: add support for dry running "kubeadm join". The new flag "kubeadm join --dry-run" is similar to the existing flag for "kubeadm init/upgrade" and allows you to see what changes would be applied. ([#103027](https://github.com/kubernetes/kubernetes/pull/103027), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle]
- Kubernetes is now built with Golang 1.16.7 ([#104199](https://github.com/kubernetes/kubernetes/pull/104199), [@cpanato](https://github.com/cpanato)) [SIG Cloud Provider, Instrumentation, Release and Testing]
- The ServiceAccountIssuerDiscovery feature gate is removed. It reached GA in Kubernetes 1.21. ([#103685](https://github.com/kubernetes/kubernetes/pull/103685), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery and Auth]
- Updated Cluster Autosaler to version 1.22.0. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.22.0 ([#104293](https://github.com/kubernetes/kubernetes/pull/104293), [@x13n](https://github.com/x13n)) [SIG Autoscaling and Cloud Provider]
- Updates the following images to pick up CVE fixes:
  - debian to v1.9.0
  - debian-iptables to v1.6.6
  - setcap to v2.0.4 ([#104142](https://github.com/kubernetes/kubernetes/pull/104142), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery, Release and Testing]

### Documentation

- Update description of --audit-log-maxbackup to describe behavior when value = 0 ([#103843](https://github.com/kubernetes/kubernetes/pull/103843), [@Arkessler](https://github.com/Arkessler)) [SIG API Machinery]

### Bug or Regression

- 1. Changes json representation for a conflicted taint to Key=Effect when a conflicted taint occurs in kubectl taint. ([#104011](https://github.com/kubernetes/kubernetes/pull/104011), [@manugupt1](https://github.com/manugupt1)) [SIG CLI]
- A new server run option 'shutdown-send-retry-after'  has been introduced. If true the HTTP Server
  will continue listening until all non longrunning request(s) in flight have been drained, during this window all 
  incoming requests will be rejected with a status code 429 and a 'Retry-After' response header. ([#101257](https://github.com/kubernetes/kubernetes/pull/101257), [@tkashem](https://github.com/tkashem)) [SIG API Machinery]
- Adds Kubernetes Events to the Kubelet Graceful Shutdown feature ([#101081](https://github.com/kubernetes/kubernetes/pull/101081), [@rphillips](https://github.com/rphillips)) [SIG Node]
- CA, certificate and key bundles for the generic-apiserver based servers will be reloaded immediately after the files are changed. ([#104102](https://github.com/kubernetes/kubernetes/pull/104102), [@tnqn](https://github.com/tnqn)) [SIG API Machinery and Testing]
- Fix kube-apiserver metric reporting for the deprecated watch path of /api/<version>/watch/... ([#104161](https://github.com/kubernetes/kubernetes/pull/104161), [@wojtek-t](https://github.com/wojtek-t)) [SIG API Machinery and Instrumentation]
- Fix: skip case sensitivity when checking Azure NSG rules ([#104384](https://github.com/kubernetes/kubernetes/pull/104384), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fixed an issue which didn't append OS's environment variables with the one provided in Credential Provider Config file, which may lead to failed execution of external credential provider binary. 
  See https://github.com/kubernetes/kubernetes/issues/102750 ([#103231](https://github.com/kubernetes/kubernetes/pull/103231), [@n4j](https://github.com/n4j)) [SIG Auth and Node]
- Fixed architecture within manifest for non `amd64` etcd images. ([#104116](https://github.com/kubernetes/kubernetes/pull/104116), [@saschagrunert](https://github.com/saschagrunert)) [SIG API Machinery]
- Fixed bug where kubectl would emit duplicate warning messages for flag names that contain an underscore and recommend using a nonexistent flag in some cases ([#103852](https://github.com/kubernetes/kubernetes/pull/103852), [@brianpursley](https://github.com/brianpursley)) [SIG CLI and Cluster Lifecycle]
- Graceful node shutdown, allow the actual inhibit delay to be greater than the expected inhibit delay ([#103137](https://github.com/kubernetes/kubernetes/pull/103137), [@wzshiming](https://github.com/wzshiming)) [SIG Node]
- Kube-apiserver: Avoids unnecessary repeated calls to admission webhooks that reject an update or delete request. ([#104182](https://github.com/kubernetes/kubernetes/pull/104182), [@liggitt](https://github.com/liggitt)) [SIG API Machinery]
- Kube-proxy: delete stale conntrack UDP entries for loadbalancer ingress IP. ([#104009](https://github.com/kubernetes/kubernetes/pull/104009), [@aojea](https://github.com/aojea)) [SIG Network]
- Kubeadm: When adding an etcd peer to an existing cluster, if an error is returned indicating the peer has already been added, this is accepted and a ListMembers call is used instead to return the existing cluster. This helps diminish the exponential backoff when the first AddMember call times out, while still retaining a similar performance when the peer had already been added from a previous call. ([#104134](https://github.com/kubernetes/kubernetes/pull/104134), [@ihgann](https://github.com/ihgann)) [SIG Cluster Lifecycle]
- Pass additional flags to subpath mount to avoid flakes in certain conditions ([#104253](https://github.com/kubernetes/kubernetes/pull/104253), [@mauriciopoppe](https://github.com/mauriciopoppe)) [SIG Storage]
- Update Go used to build migrate script in etcd image to v1.16.7 ([#104301](https://github.com/kubernetes/kubernetes/pull/104301), [@serathius](https://github.com/serathius)) [SIG API Machinery and Release]

### Other (Cleanup or Flake)

- Deprecate apiserver_longrunning_gauge and apiserver_register_watchers in 1.23.0 ([#103793](https://github.com/kubernetes/kubernetes/pull/103793), [@yan-lgtm](https://github.com/yan-lgtm)) [SIG API Machinery, Cluster Lifecycle and Instrumentation]
- Kube-apiserver: sets an upper-bound on the lifetime of idle keep-alive connections and time to read the headers of incoming requests ([#103958](https://github.com/kubernetes/kubernetes/pull/103958), [@liggitt](https://github.com/liggitt)) [SIG API Machinery and Node]
- Kubeadm: external etcd endpoints passed in the ClusterConfiguration that have Unicode characters are no longer IDNA encoded (converted to Punycode). They are now just URL encoded as per Go's implementation of RFC-3986, have duplicate "/" removed from the URL paths, and passed like that directly to the kube-apiserver --etcd-servers flag. If you have etcd endpoints that have Unicode characters, it is advisable to encode them in advance with tooling that is fully IDNA compliant. If you don't do that, the Go standard library (used in k8s and etcd) would do it for you when making requests to the endpoints. ([#103801](https://github.com/kubernetes/kubernetes/pull/103801), [@gkarthiks](https://github.com/gkarthiks)) [SIG Cluster Lifecycle]
- Kubeadm: update references to legacy artifacts locations, the 'ci-cross' prefix has been removed from the version match as it does not exist in the new 'gs://k8s-release-dev' bucket ([#103813](https://github.com/kubernetes/kubernetes/pull/103813), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- Migratecmd/kube-proxy/app logs to structured logging ([#98913](https://github.com/kubernetes/kubernetes/pull/98913), [@yxxhero](https://github.com/yxxhero)) [SIG Network]
- Surface warning when users don't set propagationPolicy for jobs while deleting ([#104080](https://github.com/kubernetes/kubernetes/pull/104080), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps]
- The AllowInsecureBackendProxy feature gate is removed. It reached GA in Kubernetes 1.21. ([#103796](https://github.com/kubernetes/kubernetes/pull/103796), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG API Machinery]
- The `StartupProbe` feature gate that is GA since v1.20 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104168](https://github.com/kubernetes/kubernetes/pull/104168), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Node]
- The apiserver exposes 4 new metrics that allow to track the status of the Service CIDRs allocations:
      - current number of available IPs per Service CIDR
      - current number of used IPs per Service CIDR
      - total number of allocation per Service CIDR
      - total number of allocation errors per ServiceCIDR ([#104119](https://github.com/kubernetes/kubernetes/pull/104119), [@aojea](https://github.com/aojea)) [SIG Apps, Instrumentation and Network]
- The flag `--deployment-controller-sync-period` has no effect now, deprecate it and will be removed in v1.24. ([#103538](https://github.com/kubernetes/kubernetes/pull/103538), [@Pingan2017](https://github.com/Pingan2017)) [SIG Apps]
- Troubleshooting: informers log handlers that take more than 100 milliseconds to process an object if the DeltaFIFO queue starts to grow beyond 10 elements. ([#103917](https://github.com/kubernetes/kubernetes/pull/103917), [@aojea](https://github.com/aojea)) [SIG API Machinery]
- Update cri-tools dependency to v1.22.0 ([#104430](https://github.com/kubernetes/kubernetes/pull/104430), [@saschagrunert](https://github.com/saschagrunert)) [SIG Cloud Provider and Node]
- ``gcr.io/kubernetes-e2e-test-images`` will no longer be used in E2E / CI testing, ``k8s.gcr.io/e2e-test-images`` will be used instead. ([#103724](https://github.com/kubernetes/kubernetes/pull/103724), [@claudiubelu](https://github.com/claudiubelu)) [SIG API Machinery and Testing]

## Dependencies

### Added
- github.com/google/martian/v3: [v3.1.0](https://github.com/google/martian/v3/tree/v3.1.0)
- github.com/kr/fs: [v0.1.0](https://github.com/kr/fs/tree/v0.1.0)
- github.com/pkg/sftp: [v1.10.1](https://github.com/pkg/sftp/tree/v1.10.1)

### Changed
- cloud.google.com/go/bigquery: v1.4.0 → v1.8.0
- cloud.google.com/go/storage: v1.6.0 → v1.10.0
- cloud.google.com/go: v0.54.0 → v0.81.0
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [7901bc8 → ea6160c](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/7901bc8...ea6160c)
- github.com/bketelsen/crypt: [5cbc8cc → v0.0.4](https://github.com/bketelsen/crypt/compare/5cbc8cc...v0.0.4)
- github.com/golang/mock: [v1.4.4 → v1.5.0](https://github.com/golang/mock/compare/v1.4.4...v1.5.0)
- github.com/google/pprof: [1ebb73c → cbba55b](https://github.com/google/pprof/compare/1ebb73c...cbba55b)
- github.com/hashicorp/golang-lru: [v0.5.1 → v0.5.0](https://github.com/hashicorp/golang-lru/compare/v0.5.1...v0.5.0)
- github.com/ianlancetaylor/demangle: [5e5cf60 → 28f6c0f](https://github.com/ianlancetaylor/demangle/compare/5e5cf60...28f6c0f)
- github.com/magiconair/properties: [v1.8.1 → v1.8.5](https://github.com/magiconair/properties/compare/v1.8.1...v1.8.5)
- github.com/mitchellh/go-homedir: [v1.1.0 → v1.0.0](https://github.com/mitchellh/go-homedir/compare/v1.1.0...v1.0.0)
- github.com/mitchellh/mapstructure: [v1.1.2 → v1.4.1](https://github.com/mitchellh/mapstructure/compare/v1.1.2...v1.4.1)
- github.com/pelletier/go-toml: [v1.2.0 → v1.9.3](https://github.com/pelletier/go-toml/compare/v1.2.0...v1.9.3)
- github.com/prometheus/common: [v0.26.0 → v0.28.0](https://github.com/prometheus/common/compare/v0.26.0...v0.28.0)
- github.com/spf13/afero: [v1.2.2 → v1.6.0](https://github.com/spf13/afero/compare/v1.2.2...v1.6.0)
- github.com/spf13/cast: [v1.3.0 → v1.3.1](https://github.com/spf13/cast/compare/v1.3.0...v1.3.1)
- github.com/spf13/cobra: [v1.1.3 → v1.2.1](https://github.com/spf13/cobra/compare/v1.1.3...v1.2.1)
- github.com/spf13/jwalterweatherman: [v1.0.0 → v1.1.0](https://github.com/spf13/jwalterweatherman/compare/v1.0.0...v1.1.0)
- github.com/spf13/viper: [v1.7.0 → v1.8.1](https://github.com/spf13/viper/compare/v1.7.0...v1.8.1)
- go.opencensus.io: v0.22.3 → v0.23.0
- golang.org/x/net: 37e1c6a → abc4532
- golang.org/x/oauth2: bf48bf1 → f6687ab
- google.golang.org/api: v0.20.0 → v0.46.0
- google.golang.org/appengine: v1.6.5 → v1.6.7
- gopkg.in/ini.v1: v1.51.0 → v1.62.0
- honnef.co/go/tools: v0.0.1-2020.1.3 → v0.0.1-2020.1.4
- k8s.io/gengo: b6c5ce2 → 485abfe
- k8s.io/kube-openapi: 9528897 → 7fbd8d5
- k8s.io/utils: 4b05e18 → efc7438

### Removed
- cloud.google.com/go/datastore: v1.1.0
- cloud.google.com/go/pubsub: v1.2.0
- github.com/alecthomas/units: [f65c72e](https://github.com/alecthomas/units/tree/f65c72e)
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/etcd: [v3.3.13+incompatible](https://github.com/coreos/etcd/tree/v3.3.13)
- github.com/coreos/go-systemd: [95778df](https://github.com/coreos/go-systemd/tree/95778df)
- github.com/coreos/pkg: [399ea9e](https://github.com/coreos/pkg/tree/399ea9e)
- github.com/dgrijalva/jwt-go: [v3.2.0+incompatible](https://github.com/dgrijalva/jwt-go/tree/v3.2.0)
- github.com/google/martian: [v2.1.0+incompatible](https://github.com/google/martian/tree/v2.1.0)
- github.com/jpillora/backoff: [v1.0.0](https://github.com/jpillora/backoff/tree/v1.0.0)