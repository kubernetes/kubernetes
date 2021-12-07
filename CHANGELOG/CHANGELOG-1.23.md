<!-- BEGIN MUNGE: GENERATED_TOC -->

- [v1.23.0](#v1230)
  - [Downloads for v1.23.0](#downloads-for-v1230)
    - [Source Code](#source-code)
    - [Client Binaries](#client-binaries)
    - [Server Binaries](#server-binaries)
    - [Node Binaries](#node-binaries)
  - [Changelog since v1.22.0](#changelog-since-v1220)
  - [What's New (Major Themes)](#whats-new-major-themes)
    - [Deprecation of FlexVolume](#deprecation-of-flexvolume)
    - [Deprecation of klog specific flags](#deprecation-of-klog-specific-flags)
    - [Software Supply Chain SLSA Level 1 Compliance in the Kubernetes Release Process](#software-supply-chain-slsa-level-1-compliance-in-the-kubernetes-release-process)
    - [IPv4/IPv6 Dual-stack Networking graduates to GA](#ipv4ipv6-dual-stack-networking-graduates-to-ga)
    - [HorizontalPodAutoscaler v2 graduates to GA](#horizontalpodautoscaler-v2-graduates-to-ga)
    - [Generic Ephemeral Volume feature graduates to GA](#generic-ephemeral-volume-feature-graduates-to-ga)
    - [Skip Volume Ownership change graduates to GA](#skip-volume-ownership-change-graduates-to-ga)
    - [Allow CSI drivers to opt-in to volume ownership and permission change graduates to GA](#allow-csi-drivers-to-opt-in-to-volume-ownership-and-permission-change-graduates-to-ga)
    - [PodSecurity graduates to Beta](#podsecurity-graduates-to-beta)
    - [Container Runtime Interface (CRI) v1 is default](#container-runtime-interface-cri-v1-is-default)
    - [Structured logging graduate to Beta](#structured-logging-graduate-to-beta)
    - [Simplified Multi-point plugin configuration for scheduler](#simplified-multi-point-plugin-configuration-for-scheduler)
    - [CSI Migration updates](#csi-migration-updates)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade)
  - [Changes by Kind](#changes-by-kind)
    - [Deprecation](#deprecation)
    - [API Change](#api-change)
    - [Feature](#feature)
    - [Documentation](#documentation)
    - [Failing Test](#failing-test)
    - [Bug or Regression](#bug-or-regression)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake)
  - [Dependencies](#dependencies)
    - [Added](#added)
    - [Changed](#changed)
    - [Removed](#removed)
- [v1.23.0-rc.1](#v1230-rc1)
  - [Downloads for v1.23.0-rc.1](#downloads-for-v1230-rc1)
    - [Source Code](#source-code-1)
    - [Client Binaries](#client-binaries-1)
    - [Server Binaries](#server-binaries-1)
    - [Node Binaries](#node-binaries-1)
  - [Changelog since v1.23.0-rc.0](#changelog-since-v1230-rc0)
  - [Changes by Kind](#changes-by-kind-1)
    - [Bug or Regression](#bug-or-regression-1)
  - [Dependencies](#dependencies-1)
    - [Added](#added-1)
    - [Changed](#changed-1)
    - [Removed](#removed-1)
- [v1.23.0-rc.0](#v1230-rc0)
  - [Downloads for v1.23.0-rc.0](#downloads-for-v1230-rc0)
    - [Source Code](#source-code-2)
    - [Client Binaries](#client-binaries-2)
    - [Server Binaries](#server-binaries-2)
    - [Node Binaries](#node-binaries-2)
  - [Changelog since v1.23.0-beta.0](#changelog-since-v1230-beta0)
  - [Changes by Kind](#changes-by-kind-2)
    - [API Change](#api-change-1)
    - [Feature](#feature-1)
    - [Bug or Regression](#bug-or-regression-2)
  - [Dependencies](#dependencies-2)
    - [Added](#added-2)
    - [Changed](#changed-2)
    - [Removed](#removed-2)
- [v1.23.0-beta.0](#v1230-beta0)
  - [Downloads for v1.23.0-beta.0](#downloads-for-v1230-beta0)
    - [Source Code](#source-code-3)
    - [Client Binaries](#client-binaries-3)
    - [Server Binaries](#server-binaries-3)
    - [Node Binaries](#node-binaries-3)
  - [Changelog since v1.23.0-alpha.4](#changelog-since-v1230-alpha4)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-1)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-1)
  - [Changes by Kind](#changes-by-kind-3)
    - [Deprecation](#deprecation-1)
    - [API Change](#api-change-2)
    - [Feature](#feature-2)
    - [Documentation](#documentation-1)
    - [Bug or Regression](#bug-or-regression-3)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-1)
  - [Dependencies](#dependencies-3)
    - [Added](#added-3)
    - [Changed](#changed-3)
    - [Removed](#removed-3)
- [v1.23.0-alpha.4](#v1230-alpha4)
  - [Downloads for v1.23.0-alpha.4](#downloads-for-v1230-alpha4)
    - [Source Code](#source-code-4)
    - [Client Binaries](#client-binaries-4)
    - [Server Binaries](#server-binaries-4)
    - [Node Binaries](#node-binaries-4)
  - [Changelog since v1.23.0-alpha.3](#changelog-since-v1230-alpha3)
  - [Changes by Kind](#changes-by-kind-4)
    - [Deprecation](#deprecation-2)
    - [API Change](#api-change-3)
    - [Feature](#feature-3)
    - [Failing Test](#failing-test-1)
    - [Bug or Regression](#bug-or-regression-4)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-2)
  - [Dependencies](#dependencies-4)
    - [Added](#added-4)
    - [Changed](#changed-4)
    - [Removed](#removed-4)
- [v1.23.0-alpha.3](#v1230-alpha3)
  - [Downloads for v1.23.0-alpha.3](#downloads-for-v1230-alpha3)
    - [Source Code](#source-code-5)
    - [Client Binaries](#client-binaries-5)
    - [Server Binaries](#server-binaries-5)
    - [Node Binaries](#node-binaries-5)
  - [Changelog since v1.23.0-alpha.2](#changelog-since-v1230-alpha2)
  - [Changes by Kind](#changes-by-kind-5)
    - [Deprecation](#deprecation-3)
    - [API Change](#api-change-4)
    - [Feature](#feature-4)
    - [Bug or Regression](#bug-or-regression-5)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-3)
  - [Dependencies](#dependencies-5)
    - [Added](#added-5)
    - [Changed](#changed-5)
    - [Removed](#removed-5)
- [v1.23.0-alpha.2](#v1230-alpha2)
  - [Downloads for v1.23.0-alpha.2](#downloads-for-v1230-alpha2)
    - [Source Code](#source-code-6)
    - [Client Binaries](#client-binaries-6)
    - [Server Binaries](#server-binaries-6)
    - [Node Binaries](#node-binaries-6)
  - [Changelog since v1.23.0-alpha.1](#changelog-since-v1230-alpha1)
  - [Changes by Kind](#changes-by-kind-6)
    - [Deprecation](#deprecation-4)
    - [API Change](#api-change-5)
    - [Feature](#feature-5)
    - [Documentation](#documentation-2)
    - [Bug or Regression](#bug-or-regression-6)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-4)
  - [Dependencies](#dependencies-6)
    - [Added](#added-6)
    - [Changed](#changed-6)
    - [Removed](#removed-6)
- [v1.23.0-alpha.1](#v1230-alpha1)
  - [Downloads for v1.23.0-alpha.1](#downloads-for-v1230-alpha1)
    - [Source Code](#source-code-7)
    - [Client Binaries](#client-binaries-7)
    - [Server Binaries](#server-binaries-7)
    - [Node Binaries](#node-binaries-7)
  - [Changelog since v1.22.0](#changelog-since-v1220-1)
  - [Urgent Upgrade Notes](#urgent-upgrade-notes-2)
    - [(No, really, you MUST read this before you upgrade)](#no-really-you-must-read-this-before-you-upgrade-2)
  - [Changes by Kind](#changes-by-kind-7)
    - [Deprecation](#deprecation-5)
    - [API Change](#api-change-6)
    - [Feature](#feature-6)
    - [Documentation](#documentation-3)
    - [Bug or Regression](#bug-or-regression-7)
    - [Other (Cleanup or Flake)](#other-cleanup-or-flake-5)
  - [Dependencies](#dependencies-7)
    - [Added](#added-7)
    - [Changed](#changed-7)
    - [Removed](#removed-7)

<!-- END MUNGE: GENERATED_TOC -->

# v1.23.0

[Documentation](https://docs.k8s.io)

## Downloads for v1.23.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes.tar.gz) | `850f92f4a4f397773ceabdacdb0513fa3cd2eb8867f7e3697f42bc595c3c710f81a8b9b34679d783ca2e1900dd272e0af209126cf55719e321af8da04a4b1c2b`
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-src.tar.gz) | `ee53eb3b32bc4745a3f58dd0af1a8f4157e74b71b896eb39ae0658f6f3d0497b8a1334205cc19a3fe1cc7056b7606b0c745e89860f73b14ffaf4ed7bf9b6ef11`

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-darwin-amd64.tar.gz) | `ec7acc668cf32ec2ecf9ff515096cfe0d421c096522f4d4f6dd5504046051d6b4a15a130aab67e0d545078b26c1a0d27f2f567b1f4ac68b76324e15a216799f5`
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-darwin-arm64.tar.gz) | `5c8dcc6c847d44bef1d739015627369b8b7f0b27d96bb0264bac1ea029d3e247565639cdfb9041e28d74f27c7650661ec275e452ede448181c454d77916fc432`
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-linux-386.tar.gz) | `a94aa935b348dfeee09b47f3a34e8cb7b2d7213dc28e8df189a4b8438a317cfd575d97ae09651fb6a46a528cd00f0808c6e6c95d9e176837b5be463620593acd`
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-linux-amd64.tar.gz) | `82574d81696693510d8becd2a2319d517dda2b424b63c5525299ed24c9f4abe1b1803fc799b2a75c650175a5a70ef03eab6068e5bf2a286a43fcaabee9681747`
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-linux-arm.tar.gz) | `8dd15ed487883f76ed869458df3b5e859518491ac5b3aedb7ec95fd6c8ba1bde081859e0a7c171d9674a773b655db67975dbab9b842d1c0864a300341e58035a`
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-linux-arm64.tar.gz) | `125f51a712fa4cad241e84a57baa4bc7950b4977bb4f7275ec21e82758ea90137d00d39841061cd9c4665144a2cf8ab87b99e185a64152f9682ef524266c45a3`
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-linux-ppc64le.tar.gz) | `ebf7130485c33a59fc81c2a6b8d19b847470f57f4be49ead06e0405a9b34891de456199a87ef105b0428f3d9767ac39f83b8199171ba60ab7b7c538b0558d1de`
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-linux-s390x.tar.gz) | `dc0a122755d096f18de5d33a7596bae3c8cce058d22999c6754377cd074b8d69f597c62b5468cc80eda4144edea026b9b946b522b144e8d9ad66874de2d0bd9b`
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-windows-386.tar.gz) | `d64f72835dca883e666fc44f80db6b75467b5f952bf40f241b8b1034a8b7565aa101b3d8d9be4b47523cf22a1bb51db3924376405fdf40cd3cab688c5e9db21c`
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-windows-amd64.tar.gz) | `915b7e23517dba67db9aa8b20f18f3451897fe7ab2bae1cd64bc22810e38efa3f2f62483ba377c6d1fed738de24874c2c14ef772f02f201f41a35d80eb2f872b`
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-client-windows-arm64.tar.gz) | `2702663c0bc4316e83573c6c262040e72c13c6ea50b9dd042dd8d375e719e8729db759db3bffa8bd6a838661278e02a3047ce402b0348ae082393b37643047cb`

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-server-linux-amd64.tar.gz) | `633cba8102648735b93d91a9840a39597b3242d2489081e71b0131a9bf246e2f52060bc8f8abcc2154f39c1e2410dec5e8b189d220743b55f861062c86edd2f8`
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-server-linux-arm.tar.gz) | `39eb3f85a46f6c71ba70ffa391e706b6c57c8f9fb7eea95960f944a3fee7883da56e3dca9eec7f3121800911e45681b2bcc78c4585ce7081720a8658e1837f47`
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-server-linux-arm64.tar.gz) | `91236a70b0ff67b54c939215ac71a7e03e4202e71d8b11f687fc6406eb54da6271e8f095a4e812b3aaad23275e0a04b4cfa7bfa8f455a98ccf8c5a22b8b5e59f`
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-server-linux-ppc64le.tar.gz) | `a41f590fcc271861d73cae14032c51d7674efba48f160550da8be7095240be39a59bdd8012886eb94afa8062576e43643a76c2796ce847d6df33d7147a807e9f`
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-server-linux-s390x.tar.gz) | `2bb2d3087e911e5c296ae194697190470606c1ac761fb3e69533492109d012e756428daac93e89665f6bbac7c4b4fd1ab9e554718244234f43c6e7219c396eff`

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-node-linux-amd64.tar.gz) | `4b0d9188914f3b8dbb7cab384b9f2a63b5d0f53b78c7c9d4920613e780d17afffd95809eb828753556048481634b5a5bd0b89b50360e4d3f2a37d21bb88d5a36`
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-node-linux-arm.tar.gz) | `039c77a9a5f7e12d826ded7f8590b9f63c79863fcc463e150848ab6304de171aebbdc948f9398478b96a763f982d02f955f8327f651a04e8caced4c6fd2b8e12`
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-node-linux-arm64.tar.gz) | `d60c740f2f5b2ffe95bc9d1ea0e26ca6af1aef718d5c498339eabe6a9cc1cdff8cb69f9cc0f2389adb1eaa3dacdbcaacae3e19fd27ce035f561813d8f29b5f90`
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-node-linux-ppc64le.tar.gz) | `163b33fcab0226e950c2d2415ece74ac840c592ede2e7276babab466ef3ef1cdbee95102ca4cbefeb836ce1b2897d19b0534967dbd0dcb6706753b453adb27eb`
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-node-linux-s390x.tar.gz) | `c01db605cc3b744a6a13962e318f94ae66eb221d4ed3758a868452bc610cdcf79297332e158cf9e88002e3ac72c9489b7c27b001ba40df1852c6ba098afb9586`
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0/kubernetes-node-windows-amd64.tar.gz) | `d5431cd60990bd56649ce11e3c5a72b92a733e6abafaee6517cec333a3dcabd6469ebd4e3d9b052d250ecb35a0c1dcac16cd841612d73886438731f0f817e2d1`

## Changelog since v1.22.0

## What's New (Major Themes)

### Deprecation of FlexVolume

FlexVolume is deprecated. Out-of-tree CSI driver is the recommended way to write volume drivers in Kubernetes.
See [this doc](https://github.com/kubernetes/community/blob/master/sig-storage/volume-plugin-faq.md#kubernetes-volume-plugin-faq-for-storage-vendors) for more information.
Maintainers of FlexVolume drivers should implement a CSI driver and move users of FlexVolume to CSI.
Users of FlexVolume should move their workloads to CSI driver.

### Deprecation of klog specific flags

To simplify the code base, several [logging flags got marked as deprecated](https://kubernetes.io/docs/concepts/cluster-administration/system-logs/#klog) in Kubernetes 1.23.
The code which implements them will be removed in a future release, so users of those need to start replacing the deprecated flags with some alternative solutions.

### Software Supply Chain SLSA Level 1 Compliance in the Kubernetes Release Process

Kubernetes releases are now generating provenance attestation files describing the staging and release phases of the release process and artifacts are verified as they are handed over from one phase to the next.
This final piece completes the work needed to comply with Level 1 of the [SLSA security framework](https://slsa.dev/) (Supply-chain Levels for Software Artifacts).

### IPv4/IPv6 Dual-stack Networking graduates to GA

[IPv4/IPv6 dual-stack networking](https://github.com/kubernetes/enhancements/tree/master/keps/sig-network/563-dual-stack) graduates to GA.
Since 1.21, Kubernetes clusters are enabled to support dual-stack networking by default.
In 1.23, the `IPv6DualStack` feature gate is removed.
The use of dual-stack networking is not mandatory.
Although clusters are enabled to support dual-stack networking, Pods and Services continue to default to single-stack.
To use dual-stack networking: Kubernetes nodes have routable IPv4/IPv6 network interfaces, a dual-stack capable CNI network plugin is used, Pods are configured to be dual-stack and Services have their `.spec.ipFamilyPolicy` field set to either `PreferDualStack` or `RequireDualStack`.

### HorizontalPodAutoscaler v2 graduates to GA

Version 2 of the HorizontalPodAutoscaler API graduates to stable in the 1.23 release. The HorizontalPodAutoscaler `autoscaling/v2beta2` API is deprecated in favor of the new `autoscaling/v2` API, which the Kubernetes project recommends for all use cases.

This release does *not* deprecate the v1 HorizontalPodAutoscaler API.

### Generic Ephemeral Volume feature graduates to GA

The generic ephemeral volume feature moved to GA in 1.23.
This feature allows any existing storage driver that supports dynamic provisioning to be used as an ephemeral volume with the volume’s lifecycle bound to the Pod.
All StorageClass parameters for volume provisioning and all features supported with PersistentVolumeClaims are supported.

### Skip Volume Ownership change graduates to GA

The feature to configure volume permission and ownership change policy for Pods moved to GA in 1.23.
This allows users to skip recursive permission changes on mount and speeds up the pod start up time.

### Allow CSI drivers to opt-in to volume ownership and permission change graduates to GA

The feature to allow CSI Drivers to declare support for fsGroup based permissions graduates to GA in 1.23.

### PodSecurity graduates to Beta

[PodSecurity](https://kubernetes.io/docs/concepts/security/pod-security-admission/) moves to Beta.
`PodSecurity` replaces the deprecated `PodSecurityPolicy` admission controller.
`PodSecurity` is an admission controller that enforces Pod Security Standards on Pods in a Namespace based on specific namespace labels that set the enforcement level.
In 1.23, the `PodSecurity` feature gate is enabled by default.

### Container Runtime Interface (CRI) v1 is default

The Kubelet now supports the CRI `v1` API, which is now the project-wide default.
If a container runtime does not support the `v1` API, Kubernetes will fall back to the `v1alpha2` implementation.
There is no intermediate action required by end-users, because `v1` and `v1alpha2` do not differ in their implementation.
It is likely that `v1alpha2` will be removed in one of the future Kubernetes releases to be able to develop `v1`.

### Structured logging graduate to Beta

Structured logging reached its Beta milestone. Most log messages from kubelet and kube-scheduler have been converted. Users are encouraged to try out JSON output or parsing of the structured text format and provide feedback on possible solutions for the open issues, such as handling of multi-line strings in log values.

### Simplified Multi-point plugin configuration for scheduler

The kube-scheduler is adding a new, simplified config field for Plugins to allow multiple extension points to be enabled in one spot.
The new `multiPoint` plugin field is intended to simplify most scheduler setups for administrators.
Plugins that are enabled via `multiPoint` will automatically be registered for each individual extension point that they implement.
For example, a plugin that implements Score and Filter extensions can be simultaneously enabled for both.
This means entire plugins can be enabled and disabled without having to manually edit individual extension point settings.
These extension points can now be abstracted away due to their irrelevance for most users.

### CSI Migration updates

CSI Migration enables the replacement of existing in-tree storage plugins such as `kubernetes.io/gce-pd` or `kubernetes.io/aws-ebs` with a corresponding CSI driver.
If CSI Migration is working properly, Kubernetes end users shouldn’t notice a difference.
After migration, Kubernetes users may continue to rely on all the functionality of in-tree storage plugins using the existing interface.
- CSI Migration feature is turned on by default but stays in Beta for GCE PD, AWS EBS, and Azure Disk in 1.23.
- CSI Migration is introduced as an Alpha feature for Ceph RBD and Portworx in 1.23.

## Urgent Upgrade Notes 

### (No, really, you MUST read this before you upgrade)

- Kubeadm: remove the deprecated flag `--experimental-patches` for the `init|join|upgrade` commands. The flag `--patches` is no longer allowed in a mixture with the flag `--config`. Please use the kubeadm configuration for setting patches for a node using `{Init|Join}Configuration.patches`. ([#104065](https://github.com/kubernetes/kubernetes/pull/104065), [@pacoxu](https://github.com/pacoxu))
 - Log messages in JSON format are written to stderr by default now (same as text format) instead of stdout. Users who expected JSON output on stdout must now capture stderr instead or in addition to stdout. ([#106146](https://github.com/kubernetes/kubernetes/pull/106146), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Cluster Lifecycle and Instrumentation]
 - Support for the seccomp annotations `seccomp.security.alpha.kubernetes.io/pod` and `container.seccomp.security.alpha.kubernetes.io/[name]` has been deprecated since 1.19, will be dropped in 1.25. Transition to using the `seccompProfile` API field. ([#104389](https://github.com/kubernetes/kubernetes/pull/104389), [@saschagrunert](https://github.com/saschagrunert))
 - [kube-log-runner](https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/component-base/logs/kube-log-runner) is included in release tar balls. It can be used to replace the deprecated `--log-file` parameter. ([#106123](https://github.com/kubernetes/kubernetes/pull/106123), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Cloud Provider, Cluster Lifecycle and Instrumentation]
 
## Changes by Kind

### Deprecation

- A deprecation notice has been added when using the kube-proxy userspace proxier, which will be removed in v1.25. (#103860) ([#104631](https://github.com/kubernetes/kubernetes/pull/104631), [@perithompson](https://github.com/perithompson))
- Added `apiserver_longrunning_requests` metric to replace the soon to be deprecated `apiserver_longrunning_gauge` metric. ([#103799](https://github.com/kubernetes/kubernetes/pull/103799), [@jyz0309](https://github.com/jyz0309))
- Controller-manager: the following flags have no effect and would be removed in v1.24:
  - `--port`
  - `--address`
  The insecure port flags `--port` may only be set to 0 now.
  Also `metricsBindAddress` and `healthzBindAddress` fields from `kubescheduler.config.k8s.io/v1beta1` are no-op and expected to be empty. Removed in `kubescheduler.config.k8s.io/v1beta2` completely. ([#96345](https://github.com/kubernetes/kubernetes/pull/96345), [@ingvagabund](https://github.com/ingvagabund))
- Feature-gate `VolumeSubpath` has been deprecated and cannot be disabled. It will be completely removed in 1.25 ([#105474](https://github.com/kubernetes/kubernetes/pull/105474), [@mauriciopoppe](https://github.com/mauriciopoppe))
- Kubeadm: add a new `output/v1alpha2` API that is identical to the `output/v1alpha1`, but attempts to resolve some internal dependencies with the `kubeadm/v1beta2` API. The `output/v1alpha1` API is now deprecated and will be removed in a future release. ([#105295](https://github.com/kubernetes/kubernetes/pull/105295), [@neolit123](https://github.com/neolit123))
- Kubeadm: add the kubeadm specific, Alpha (disabled by default) feature gate UnversionedKubeletConfigMap. When this feature is enabled kubeadm will start using a new naming format for the ConfigMap where it stores the KubeletConfiguration structure. The old format included the Kubernetes version - "kube-system/kubelet-config-1.22", while the new format does not - "kube-system/kubelet-config". A similar formatting change is done for the related RBAC rules. The old format is now DEPRECATED and will be removed after the feature graduates to GA. When writing the ConfigMap kubeadm (init, upgrade apply) will respect the value of UnversionedKubeletConfigMap, while when reading it (join, reset, upgrade), it would attempt to use new format first and fallback to the legacy format if needed. ([#105741](https://github.com/kubernetes/kubernetes/pull/105741), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Testing]
- Kubeadm: remove the deprecated / NO-OP phase `update-cluster-status` in `kubeadm reset` ([#105888](https://github.com/kubernetes/kubernetes/pull/105888), [@neolit123](https://github.com/neolit123))
- Remove 'master' as a valid EgressSelection type in the EgressSelectorConfiguration API. ([#102242](https://github.com/kubernetes/kubernetes/pull/102242), [@pacoxu](https://github.com/pacoxu))
- Removed `kubectl --dry-run` empty default value and boolean values. `kubectl --dry-run` usage must be specified with `--dry-run=(server|client|none)`. ([#105327](https://github.com/kubernetes/kubernetes/pull/105327), [@julianvmodesto](https://github.com/julianvmodesto))
- Removed deprecated metric `scheduler_volume_scheduling_duration_seconds`. ([#104518](https://github.com/kubernetes/kubernetes/pull/104518), [@dntosas](https://github.com/dntosas))
- The deprecated `--experimental-bootstrap-kubeconfig` flag has been removed.
  This can be set via `--bootstrap-kubeconfig`. ([#103172](https://github.com/kubernetes/kubernetes/pull/103172), [@niulechuan](https://github.com/niulechuan))

### API Change

- A new field `omitManagedFields` has been added to both `audit.Policy` and `audit.PolicyRule` 
  so cluster operators can opt in to omit managed fields of the request and response bodies from 
  being written to the API audit log. ([#94986](https://github.com/kubernetes/kubernetes/pull/94986), [@tkashem](https://github.com/tkashem)) [SIG API Machinery, Auth, Cloud Provider and Testing]
- A small regression in Service updates was fixed. The circumstances are so unlikely that probably nobody would ever hit it. ([#104601](https://github.com/kubernetes/kubernetes/pull/104601), [@thockin](https://github.com/thockin))
- Added a feature gate `StatefulSetAutoDeletePVC`, which allows PVCs automatically created for StatefulSet pods to be automatically deleted. ([#99728](https://github.com/kubernetes/kubernetes/pull/99728), [@mattcary](https://github.com/mattcary))
- Client-go impersonation config can specify a UID to pass impersonated uid information through in requests. ([#104483](https://github.com/kubernetes/kubernetes/pull/104483), [@margocrawf](https://github.com/margocrawf))
- Create HPA v2 from v2beta2 with some fields changed. ([#102534](https://github.com/kubernetes/kubernetes/pull/102534), [@wangyysde](https://github.com/wangyysde)) [SIG API Machinery, Apps, Auth, Autoscaling and Testing]
- Ephemeral containers graduated to beta and are now available by default. ([#105405](https://github.com/kubernetes/kubernetes/pull/105405), [@verb](https://github.com/verb))
- Fix kube-proxy regression on UDP services because the logic to detect stale connections was not considering if the endpoint was ready. ([#106163](https://github.com/kubernetes/kubernetes/pull/106163), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Contributor Experience, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage, Testing and Windows]
- If a conflict occurs when creating an object with `generateName`, the server now returns an "AlreadyExists" error with a retry option. ([#104699](https://github.com/kubernetes/kubernetes/pull/104699), [@vincepri](https://github.com/vincepri))
- Implement support for recovering from volume expansion failures ([#106154](https://github.com/kubernetes/kubernetes/pull/106154), [@gnufied](https://github.com/gnufied)) [SIG API Machinery, Apps and Storage]
- In kubelet, log verbosity and flush frequency can also be configured via the configuration file and not just via command line flags. In other commands (kube-apiserver, kube-controller-manager), the flags are listed in the "Logs flags" group and not under "Global" or "Misc". The type for `-vmodule` was made a bit more descriptive (`pattern=N,...` instead of `moduleSpec`). ([#106090](https://github.com/kubernetes/kubernetes/pull/106090), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, CLI, Cluster Lifecycle, Instrumentation, Node and Scheduling]
- Introduce `OS` field in the PodSpec ([#104693](https://github.com/kubernetes/kubernetes/pull/104693), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- Introduce `v1beta3` API for scheduler. This version 
  - increases the weight of user specifiable priorities.
  The weights of following priority plugins are increased
    - `TaintTolerations` to 3 - as leveraging node tainting to group nodes in the cluster is becoming a widely-adopted practice
    - `NodeAffinity` to 2
    - `InterPodAffinity` to 2
  
  - Won't have `HealthzBindAddress`, `MetricsBindAddress` fields ([#104251](https://github.com/kubernetes/kubernetes/pull/104251), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- Introduce v1beta2 for Priority and Fairness with no changes in API spec. ([#104399](https://github.com/kubernetes/kubernetes/pull/104399), [@tkashem](https://github.com/tkashem))
- JSON log output is configurable and now supports writing info messages to stdout and error messages to stderr. Info messages can be buffered in memory. The default is to write both to stdout without buffering, as before. ([#104873](https://github.com/kubernetes/kubernetes/pull/104873), [@pohly](https://github.com/pohly))
- JobTrackingWithFinalizers graduates to beta. Feature is enabled by default. ([#105687](https://github.com/kubernetes/kubernetes/pull/105687), [@alculquicondor](https://github.com/alculquicondor))
- Kube-apiserver: Fixes handling of CRD schemas containing literal null values in enums. ([#104969](https://github.com/kubernetes/kubernetes/pull/104969), [@liggitt](https://github.com/liggitt))
- Kube-apiserver: The `rbac.authorization.k8s.io/v1alpha1` API version is removed; use the `rbac.authorization.k8s.io/v1` API, available since v1.8. The `scheduling.k8s.io/v1alpha1` API version is removed; use the `scheduling.k8s.io/v1` API, available since v1.14. ([#104248](https://github.com/kubernetes/kubernetes/pull/104248), [@liggitt](https://github.com/liggitt))
- Kube-scheduler: support for configuration file version `v1beta1` is removed. Update configuration files to v1beta2(xref: https://github.com/kubernetes/enhancements/issues/2901) or v1beta3 before upgrading to 1.23. ([#104782](https://github.com/kubernetes/kubernetes/pull/104782), [@kerthcet](https://github.com/kerthcet))
- KubeSchedulerConfiguration provides a new field `MultiPoint` which will register a plugin for all valid extension points ([#105611](https://github.com/kubernetes/kubernetes/pull/105611), [@damemi](https://github.com/damemi)) [SIG Scheduling and Testing]
- Kubelet should reject pods whose OS doesn't match the node's OS label. ([#105292](https://github.com/kubernetes/kubernetes/pull/105292), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps and Node]
- Kubelet: turn the KubeletConfiguration v1beta1 `ResolverConfig` field from a `string` to `*string`. ([#104624](https://github.com/kubernetes/kubernetes/pull/104624), [@Haleygo](https://github.com/Haleygo))
- Kubernetes is now built using go 1.17. ([#103692](https://github.com/kubernetes/kubernetes/pull/103692), [@justaugustus](https://github.com/justaugustus))
- Performs strict server side schema validation requests via the `fieldValidation=[Strict,Warn,Ignore]`. ([#105916](https://github.com/kubernetes/kubernetes/pull/105916), [@kevindelgado](https://github.com/kevindelgado))
- Promote `IPv6DualStack` feature to stable.
  Controller Manager flags for the node IPAM controller have slightly changed:
  1. When configuring a dual-stack cluster, the user must specify both `--node-cidr-mask-size-ipv4` and `--node-cidr-mask-size-ipv6` to set the per-node IP mask sizes, instead of the previous `--node-cidr-mask-size` flag.
  2. The `--node-cidr-mask-size` flag is mutually exclusive with `--node-cidr-mask-size-ipv4` and `--node-cidr-mask-size-ipv6`.
  3. Single-stack clusters do not need to change, but may choose to use the more specific flags.  Users can use either the older `--node-cidr-mask-size` flag or one of the newer `--node-cidr-mask-size-ipv4` or `--node-cidr-mask-size-ipv6` flags to configure the per-node IP mask size, provided that the flag's IP family matches the cluster's IP family (--cluster-cidr). ([#104691](https://github.com/kubernetes/kubernetes/pull/104691), [@khenidak](https://github.com/khenidak))
- Remove `NodeLease` feature gate that was graduated and locked to stable in 1.17 release. ([#105222](https://github.com/kubernetes/kubernetes/pull/105222), [@cyclinder](https://github.com/cyclinder))
- Removed deprecated `--seccomp-profile-root`/`seccompProfileRoot` config. ([#103941](https://github.com/kubernetes/kubernetes/pull/103941), [@saschagrunert](https://github.com/saschagrunert))
- Since golang 1.17 both net.ParseIP and net.ParseCIDR rejects leading zeros in the dot-decimal notation of IPv4 addresses,
  Kubernetes will keep allowing leading zeros on IPv4 address to not break the compatibility.
  IMPORTANT: Kubernetes interprets leading zeros on IPv4 addresses as decimal, users must not rely on parser alignment to not being impacted by the associated security advisory:
  CVE-2021-29923 golang standard library "net" - Improper Input Validation of octal literals in golang 1.16.2 and below standard library "net" results in indeterminate SSRF & RFI vulnerabilities.
  Reference: https://nvd.nist.gov/vuln/detail/CVE-2021-29923 ([#104368](https://github.com/kubernetes/kubernetes/pull/104368), [@aojea](https://github.com/aojea))
- StatefulSet `minReadySeconds` is promoted to beta. ([#104045](https://github.com/kubernetes/kubernetes/pull/104045), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- Support pod priority based node graceful shutdown. ([#102915](https://github.com/kubernetes/kubernetes/pull/102915), [@wzshiming](https://github.com/wzshiming))
- The "Generic Ephemeral Volume" feature graduates to GA. It is now enabled unconditionally. ([#105609](https://github.com/kubernetes/kubernetes/pull/105609), [@pohly](https://github.com/pohly))
- The Kubelet's `--register-with-taints` option is now available via the Kubelet config file field registerWithTaints ([#105437](https://github.com/kubernetes/kubernetes/pull/105437), [@cmssczy](https://github.com/cmssczy)) [SIG Node and Scalability]
- The `CSIDriver.Spec.StorageCapacity` can now be modified. ([#101789](https://github.com/kubernetes/kubernetes/pull/101789), [@pohly](https://github.com/pohly))
- The `CSIVolumeFSGroupPolicy` feature has moved from beta to GA. ([#105940](https://github.com/kubernetes/kubernetes/pull/105940), [@dobsonj](https://github.com/dobsonj))
- The `IngressClass.Spec.Parameters.Namespace` field is now GA. ([#104636](https://github.com/kubernetes/kubernetes/pull/104636), [@hbagdi](https://github.com/hbagdi))
- The `Service.spec.ipFamilyPolicy` field is now *required* in order to create or update a Service as dual-stack.  This is a breaking change from the beta behavior.  Previously the server would try to infer the value of that field from either `ipFamilies` or `clusterIPs`, but that caused ambiguity on updates.  Users who want a dual-stack Service MUST specify `ipFamilyPolicy` as either "PreferDualStack" or "RequireDualStack". ([#96684](https://github.com/kubernetes/kubernetes/pull/96684), [@thockin](https://github.com/thockin))
- The `TTLAfterFinished` feature gate is now GA and enabled by default. ([#105219](https://github.com/kubernetes/kubernetes/pull/105219), [@sahilvv](https://github.com/sahilvv))
- The `kube-controller-manager` supports `--concurrent-ephemeralvolume-syncs` flag to set the number of ephemeral volume controller workers. ([#102981](https://github.com/kubernetes/kubernetes/pull/102981), [@SataQiu](https://github.com/SataQiu))
- The legacy scheduler policy config is removed in v1.23, the associated flags `policy-config-file`, `policy-configmap`, `policy-configmap-namespace` and `use-legacy-policy-config` are also removed. Migrate to Component Config instead, see https://kubernetes.io/docs/reference/scheduling/config/ for details. ([#105424](https://github.com/kubernetes/kubernetes/pull/105424), [@kerthcet](https://github.com/kerthcet))
- Track the number of Pods with a Ready condition in Job status. The feature is alpha and needs the feature gate JobReadyPods to be enabled. ([#104915](https://github.com/kubernetes/kubernetes/pull/104915), [@alculquicondor](https://github.com/alculquicondor))
- Users of `LogFormatRegistry` in component-base must update their code to use the logr v1.0.0 API. The JSON log output now uses the format from go-logr/zapr (no `v` field for error messages, additional information for invalid calls) and has some fixes (correct source code location for warnings about invalid log calls). ([#104103](https://github.com/kubernetes/kubernetes/pull/104103), [@pohly](https://github.com/pohly))
- Validation rules for Custom Resource Definitions can be written in the [CEL expression language](https://github.com/google/cel-spec) using the `x-kubernetes-validations` extension in OpenAPIv3 schemas (alpha). This is gated by the alpha "CustomResourceValidationExpressions" feature gate. ([#106051](https://github.com/kubernetes/kubernetes/pull/106051), [@jpbetz](https://github.com/jpbetz)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Storage and Testing]

### Feature

- (beta feature) If the CSI driver supports the NodeServiceCapability `VOLUME_MOUNT_GROUP` and the `DelegateFSGroupToCSIDriver` feature gate is enabled, kubelet will delegate applying FSGroup to the driver by passing it to NodeStageVolume and NodePublishVolume, regardless of what other FSGroup policies are set. ([#106330](https://github.com/kubernetes/kubernetes/pull/106330), [@verult](https://github.com/verult)) [SIG Storage]
- Add a new `distribute-cpus-across-numa` option to the static `CPUManager` policy. When enabled, this will trigger the `CPUManager` to evenly distribute CPUs across NUMA nodes in cases where more than one NUMA node is required to satisfy the allocation. ([#105631](https://github.com/kubernetes/kubernetes/pull/105631), [@klueska](https://github.com/klueska))
- Add fish shell completion to kubectl. ([#92989](https://github.com/kubernetes/kubernetes/pull/92989), [@WLun001](https://github.com/WLun001))
- Add mechanism to load simple sniffer class into fluentd-elasticsearch image ([#92853](https://github.com/kubernetes/kubernetes/pull/92853), [@cosmo0920](https://github.com/cosmo0920))
- Add support for Portworx plugin to csi-translation-lib. Alpha release
  
  Portworx CSI driver is required to enable migration.
  This PR adds support of the `CSIMigrationPortworx` feature gate, which can be enabled by:
  
  1. Adding the feature flag to the kube-controller-manager `--feature-gates=CSIMigrationPortworx=true` 
  2. Adding the feature flag to the kubelet config:
  
  featureGates:
    CSIMigrationPortworx: true ([#103447](https://github.com/kubernetes/kubernetes/pull/103447), [@trierra](https://github.com/trierra)) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage, Testing and Windows]
- Add support to generate client-side binaries for windows/arm64 platform ([#104894](https://github.com/kubernetes/kubernetes/pull/104894), [@pacoxu](https://github.com/pacoxu))
- Added PowerShell completion generation by running `kubectl completion powershell`. ([#103758](https://github.com/kubernetes/kubernetes/pull/103758), [@zikhan](https://github.com/zikhan))
- Added a `Processing` condition for the `workqueue` API.
  Changed `Shutdown` for the `workqueue` API to wait until the work queue finishes processing all in-flight items. ([#101928](https://github.com/kubernetes/kubernetes/pull/101928), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu))
- Added a new feature gate `CustomResourceValidationExpressions` to enable expression validation for Custom Resource. ([#105107](https://github.com/kubernetes/kubernetes/pull/105107), [@cici37](https://github.com/cici37))
- Added a new flag `--append-server-path` to `kubectl proxy` that will automatically append the kube context server path to each request. ([#97350](https://github.com/kubernetes/kubernetes/pull/97350), [@FabianKramm](https://github.com/FabianKramm))
- Added ability for `kubectl wait` to wait on arbitary JSON path ([#105776](https://github.com/kubernetes/kubernetes/pull/105776), [@lauchokyip](https://github.com/lauchokyip))
- Added support for `PodAndContainerStatsFromCRI` feature gate, which allows a user to specify their pod stats must also come from the CRI, not `cAdvisor`. ([#103095](https://github.com/kubernetes/kubernetes/pull/103095), [@haircommander](https://github.com/haircommander))
- Added support for setting controller-manager log level online. ([#104571](https://github.com/kubernetes/kubernetes/pull/104571), [@h4ghhh](https://github.com/h4ghhh))
- Added the ability to specify whether to use an RFC7396 JSON Merge Patch, an RFC6902 JSON Patch, or a Strategic Merge Patch to perform an override of the resources created by `kubectl run` and `kubectl expose`. ([#105140](https://github.com/kubernetes/kubernetes/pull/105140), [@brianpursley](https://github.com/brianpursley))
- Adding option for `kubectl cp` to resume on network errors until completion, requires tar in addition to tail inside the container image ([#104792](https://github.com/kubernetes/kubernetes/pull/104792), [@matthyx](https://github.com/matthyx))
- Adding support for multiple `--from-env-file flags`. ([#104232](https://github.com/kubernetes/kubernetes/pull/104232), [@lauchokyip](https://github.com/lauchokyip))
- Adding support for multiple `--from-env-file` flags. ([#101646](https://github.com/kubernetes/kubernetes/pull/101646), [@lauchokyip](https://github.com/lauchokyip))
- Adds `--as-uid` flag to `kubectl` to allow uid impersonation in the same way as user and group impersonation. ([#105794](https://github.com/kubernetes/kubernetes/pull/105794), [@margocrawf](https://github.com/margocrawf))
- Adds new [alpha] command 'kubectl events' ([#99557](https://github.com/kubernetes/kubernetes/pull/99557), [@bboreham](https://github.com/bboreham))
- Allow node expansion of local volumes. ([#102886](https://github.com/kubernetes/kubernetes/pull/102886), [@gnufied](https://github.com/gnufied))
- Allow to build kubernetes with a custom `kube-cross` image. ([#104185](https://github.com/kubernetes/kubernetes/pull/104185), [@dims](https://github.com/dims))
- Allows users to prevent garbage collection on pinned images ([#103299](https://github.com/kubernetes/kubernetes/pull/103299), [@wgahnagl](https://github.com/wgahnagl)) [SIG Node]
- CRI v1 is now the project default. If a container runtime does not support the v1 API, Kubernetes will fall back to the v1alpha2 implementation. ([#106501](https://github.com/kubernetes/kubernetes/pull/106501), [@ehashman](https://github.com/ehashman))
- Changed feature `CSIMigrationAWS` to on by default. This feature requires the AWS EBS CSI driver to be installed. ([#106098](https://github.com/kubernetes/kubernetes/pull/106098), [@wongma7](https://github.com/wongma7))
- Client-go: pass `DeleteOptions` down to the fake client `Reactor` ([#102945](https://github.com/kubernetes/kubernetes/pull/102945), [@chenchun](https://github.com/chenchun))
- Cloud providers can set service account names for cloud controllers. ([#103178](https://github.com/kubernetes/kubernetes/pull/103178), [@nckturner](https://github.com/nckturner))
- Display Labels when kubectl describe ingress. ([#103894](https://github.com/kubernetes/kubernetes/pull/103894), [@kabab](https://github.com/kabab))
- Enhance scheduler `VolumeBinding` plugin to handle Lost PVC as `UnschedulableAndUnresolvable` ([#105245](https://github.com/kubernetes/kubernetes/pull/105245), [@yibozhuang](https://github.com/yibozhuang))
- Ensures that volume is deleted from the storage backend when the user tries to delete the PV object manually and the PV `ReclaimPolicy` is set to `Delete`. ([#105773](https://github.com/kubernetes/kubernetes/pull/105773), [@deepakkinni](https://github.com/deepakkinni))
- Expose a `NewUnstructuredExtractor` from apply configurations `meta/v1` package that enables extracting objects into unstructured apply configurations. ([#103564](https://github.com/kubernetes/kubernetes/pull/103564), [@kevindelgado](https://github.com/kevindelgado))
- Feature gate `StorageObjectInUseProtection` has been deprecated and cannot be disabled. It will be completely removed in 1.25 ([#105495](https://github.com/kubernetes/kubernetes/pull/105495), [@ikeeip](https://github.com/ikeeip))
- Graduating `controller_admission_duration_seconds`, `step_admission_duration_seconds`, `webhook_admission_duration_seconds`, `apiserver_current_inflight_requests` and `apiserver_response_sizes` metrics to stable. ([#106122](https://github.com/kubernetes/kubernetes/pull/106122), [@rezakrimi](https://github.com/rezakrimi)) [SIG API Machinery, Instrumentation and Testing]
- Graduating `pending_pods`, `preemption_attempts_total`, `preemption_victims` and `schedule_attempts_total` metrics to stable. Also `e2e_scheduling_duration_seconds` is renamed to `scheduling_attempt_duration_seconds` and the latter is graduated to stable. ([#105941](https://github.com/kubernetes/kubernetes/pull/105941), [@rezakrimi](https://github.com/rezakrimi)) [SIG Instrumentation, Scheduling and Testing]
- Health check of kube-controller-manager now includes each controller. ([#104667](https://github.com/kubernetes/kubernetes/pull/104667), [@jiahuif](https://github.com/jiahuif))
- Integration testing now takes periodic Prometheus scrapes from the etcd server.
  There is a new script ,`hack/run-prometheus-on-etcd-scrapes.sh`, that runs a containerized Prometheus server against an archive of such scrapes. ([#106190](https://github.com/kubernetes/kubernetes/pull/106190), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery and Testing]
- Introduce a feature gate `DisableKubeletCloudCredentialProviders` which allows disabling the in-tree kubelet credential providers.
  
  The feature gate `DisableKubeletCloudCredentialProviders` is currently in Alpha, which means is currently disabled by default. Once this feature gate moves to beta, in-tree credential providers will be disabled by default, and users will need to migrate to use external credential providers. ([#102507](https://github.com/kubernetes/kubernetes/pull/102507), [@ostrain](https://github.com/ostrain))
- Introduces a new metric: `admission_webhook_request_total` with the following labels: name (string) - the webhook name, type (string) - the admission type, operation (string) - the requested verb, code (int) - the HTTP status code, rejected (bool) - whether the request was rejected, namespace (string) - the namespace of the requested resource. ([#103162](https://github.com/kubernetes/kubernetes/pull/103162), [@rmoriar1](https://github.com/rmoriar1))
- Kubeadm: add support for dry running `kubeadm join`. The new flag `kubeadm join --dry-run` is similar to the existing flag for `kubeadm init/upgrade` and allows you to see what changes would be applied. ([#103027](https://github.com/kubernetes/kubernetes/pull/103027), [@Haleygo](https://github.com/Haleygo))
- Kubeadm: do not check if the `/etc/kubernetes/manifests` folder is empty on joining worker nodes during preflight ([#104942](https://github.com/kubernetes/kubernetes/pull/104942), [@SataQiu](https://github.com/SataQiu))
- Kubectl will now provide shell completion choices for the `--output/-o` flag ([#105851](https://github.com/kubernetes/kubernetes/pull/105851), [@marckhouzam](https://github.com/marckhouzam))
- Kubelet should reconcile `kubernetes.io/os` and `kubernetes.io/arch` labels on the node object. The side-effect of this is kubelet would deny admission to pod which has nodeSelector with label `kubernetes.io/os` or `kubernetes.io/arch` which doesn't match the underlying OS or arch on the host OS. 
  - The label reconciliation happens as part of periodic status update which can be configured via flag `--node-status-update-frequency` ([#104613](https://github.com/kubernetes/kubernetes/pull/104613), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Node, Testing and Windows]
- Kubernetes is now built with Golang 1.16.7. ([#104199](https://github.com/kubernetes/kubernetes/pull/104199), [@cpanato](https://github.com/cpanato))
- Kubernetes is now built with Golang 1.17.1. ([#104904](https://github.com/kubernetes/kubernetes/pull/104904), [@cpanato](https://github.com/cpanato))
- Kubernetes is now built with Golang 1.17.2 ([#105563](https://github.com/kubernetes/kubernetes/pull/105563), [@mengjiao-liu](https://github.com/mengjiao-liu))
- Kubernetes is now built with Golang 1.17.3 ([#106209](https://github.com/kubernetes/kubernetes/pull/106209), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- Move `ConfigurableFSGroupPolicy` to GA and rename metric `volume_fsgroup_recursive_apply` to `volume_apply_access_control` ([#105885](https://github.com/kubernetes/kubernetes/pull/105885), [@gnufied](https://github.com/gnufied))
- Move the `GetAllocatableResources` Endpoint in PodResource API to the beta that will make it enabled by default. ([#105003](https://github.com/kubernetes/kubernetes/pull/105003), [@swatisehgal](https://github.com/swatisehgal))
- Moving `WindowsHostProcessContainers` feature to beta ([#106058](https://github.com/kubernetes/kubernetes/pull/106058), [@marosset](https://github.com/marosset))
- Node affinity, Node selectors, and tolerations are now mutable for Jobs that are suspended and have never been started ([#105479](https://github.com/kubernetes/kubernetes/pull/105479), [@ahg-g](https://github.com/ahg-g))
- Pod template annotations and labels are now mutable for Jobs that are suspended and have never been started ([#105980](https://github.com/kubernetes/kubernetes/pull/105980), [@ahg-g](https://github.com/ahg-g))
- PodSecurity: in 1.23+ restricted policy levels, Pods and containers which set `runAsUser=0` are forbidden at admission-time; previously, they would be rejected at runtime ([#105857](https://github.com/kubernetes/kubernetes/pull/105857), [@liggitt](https://github.com/liggitt))
- Shell completion now knows to continue suggesting resource names when the command supports it.  For example `kubectl get pod pod1 <TAB>` will suggest more Pod names. ([#105711](https://github.com/kubernetes/kubernetes/pull/105711), [@marckhouzam](https://github.com/marckhouzam))
- Support to enable Hyper-V in GCE Windows Nodes created with `kube-up` ([#105999](https://github.com/kubernetes/kubernetes/pull/105999), [@mauriciopoppe](https://github.com/mauriciopoppe))
- The CPUManager policy options are now enabled, and we introduce a graduation path for the new CPU Manager policy options. ([#105012](https://github.com/kubernetes/kubernetes/pull/105012), [@fromanirh](https://github.com/fromanirh))
- The Pods and Pod controllers that are exempted from the PodSecurity admission process are now marked with the `pod-security.kubernetes.io/exempt: user/namespace/runtimeClass` annotation, based on what caused the exemption.
  
  The enforcement level that allowed or denied a Pod during PodSecurity admission is now marked by the `pod-security.kubernetes.io/enforce-policy` annotation.
  
  The annotation that informs about audit policy violations changed from `pod-security.kubernetes.io/audit` to `pod-security.kubernetes.io/audit-violation`. ([#105908](https://github.com/kubernetes/kubernetes/pull/105908), [@stlaz](https://github.com/stlaz))
- The `/openapi/v3` endpoint will be populated with OpenAPI v3 if the feature flag is enabled ([#105945](https://github.com/kubernetes/kubernetes/pull/105945), [@Jefftree](https://github.com/Jefftree))
- The `CSIMigrationGCE` feature flag is turned `ON` by default ([#104722](https://github.com/kubernetes/kubernetes/pull/104722), [@leiyiz](https://github.com/leiyiz))
- The `DownwardAPIHugePages` feature is now enabled by default. ([#106271](https://github.com/kubernetes/kubernetes/pull/106271), [@mysunshine92](https://github.com/mysunshine92))
- The `PodSecurity `admission plugin has graduated to `beta` and is enabled by default. The admission configuration version has been promoted to `pod-security.admission.config.k8s.io/v1beta1`. See https://kubernetes.io/docs/concepts/security/pod-security-admission/ for usage guidelines. ([#106089](https://github.com/kubernetes/kubernetes/pull/106089), [@liggitt](https://github.com/liggitt))
- The `ServiceAccountIssuerDiscovery` feature gate is removed. It reached GA in Kubernetes 1.21. ([#103685](https://github.com/kubernetes/kubernetes/pull/103685), [@mengjiao-liu](https://github.com/mengjiao-liu))
- The `constants/variables` from k8s.io for STABLE metrics is now supported. ([#103654](https://github.com/kubernetes/kubernetes/pull/103654), [@coffeepac](https://github.com/coffeepac))
- The `kubectl describe namespace` now shows Conditions ([#106219](https://github.com/kubernetes/kubernetes/pull/106219), [@dlipovetsky](https://github.com/dlipovetsky))
- The etcd container image now supports Windows. ([#92433](https://github.com/kubernetes/kubernetes/pull/92433), [@claudiubelu](https://github.com/claudiubelu))
- The kube-apiserver's Prometheus metrics have been extended with some that describe the costs of handling LIST requests.  They are as follows.
  - *apiserver_cache_list_total*: Counter of LIST requests served from watch cache, broken down by resource_prefix and index_name
  - *apiserver_cache_list_fetched_objects_total*: Counter of objects read from watch cache in the course of serving a LIST request, broken down by resource_prefix and index_name
  - *apiserver_cache_list_evaluated_objects_total*: Counter of objects tested in the course of serving a LIST request from watch cache, broken down by resource_prefix
  - *apiserver_cache_list_returned_objects_total*: Counter of objects returned for a LIST request from watch cache, broken down by resource_prefix
  - *apiserver_storage_list_total*: Counter of LIST requests served from etcd, broken down by resource
  - *apiserver_storage_list_fetched_objects_total*: Counter of objects read from etcd in the course of serving a LIST request, broken down by resource
  - *apiserver_storage_list_evaluated_objects_total*: Counter of objects tested in the course of serving a LIST request from etcd, broken down by resource
  - *apiserver_storage_list_returned_objects_total*: Counter of objects returned for a LIST request from etcd, broken down by resource ([#104983](https://github.com/kubernetes/kubernetes/pull/104983), [@MikeSpreitzer](https://github.com/MikeSpreitzer))
- The pause image list now contains Windows Server 2022. ([#104438](https://github.com/kubernetes/kubernetes/pull/104438), [@nick5616](https://github.com/nick5616))
- The script `kube-up.sh` installs `csi-proxy v1.0.1-gke.0`. ([#104426](https://github.com/kubernetes/kubernetes/pull/104426), [@mauriciopoppe](https://github.com/mauriciopoppe))
- This PR adds the following metrics for API Priority and Fairness.
  - **apiserver_flowcontrol_priority_level_seat_count_samples**: histograms of seats occupied by executing requests (both regular and final-delay phases included), broken down by priority_level; the observations are taken once per millisecond.
  - **apiserver_flowcontrol_priority_level_seat_count_watermarks**: histograms of high and low watermarks of number of seats occupied by executing requests (both regular and final-delay phases included), broken down by priority_level.
  - **apiserver_flowcontrol_watch_count_samples**: histograms of number of watches relevant to a given mutating request, broken down by that request's priority_level and flow_schema. ([#105873](https://github.com/kubernetes/kubernetes/pull/105873), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery, Instrumentation and Testing]
- Topology Aware Hints have graduated to beta. ([#106433](https://github.com/kubernetes/kubernetes/pull/106433), [@robscott](https://github.com/robscott)) [SIG Network]
- Turn on CSIMigrationAzureDisk by default on 1.23 ([#104670](https://github.com/kubernetes/kubernetes/pull/104670), [@andyzhangx](https://github.com/andyzhangx))
- Update the system-validators library to v1.6.0 ([#106323](https://github.com/kubernetes/kubernetes/pull/106323), [@neolit123](https://github.com/neolit123)) [SIG Cluster Lifecycle and Node]
- Updated Cluster Autosaler to version `1.22.0`. Release notes: https://github.com/kubernetes/autoscaler/releases/tag/cluster-autoscaler-1.22.0. ([#104293](https://github.com/kubernetes/kubernetes/pull/104293), [@x13n](https://github.com/x13n))
- Updates `debian-iptables` to v1.6.7 to pick up CVE fixes. ([#104970](https://github.com/kubernetes/kubernetes/pull/104970), [@PushkarJ](https://github.com/PushkarJ))
- Updates the following images to pick up CVE fixes:
  - `debian` to v1.9.0
  - `debian-iptables` to v1.6.6
  - `setcap` to v2.0.4 ([#104142](https://github.com/kubernetes/kubernetes/pull/104142), [@mengjiao-liu](https://github.com/mengjiao-liu))
- Upgrade etcd to 3.5.1 ([#105706](https://github.com/kubernetes/kubernetes/pull/105706), [@uthark](https://github.com/uthark)) [SIG Cloud Provider, Cluster Lifecycle and Testing]
- When feature gate `JobTrackingWithFinalizers` is enabled:
  - Limit the number of Pods tracked in a single Job sync to avoid starvation of small Jobs.
  - The metric `job_pod_finished_total` counts the number of finished Pods tracked by the Job controller. ([#105197](https://github.com/kubernetes/kubernetes/pull/105197), [@alculquicondor](https://github.com/alculquicondor))
- When using `RequestedToCapacityRatio` ScoringStrategy, empty shape will cause error. ([#106169](https://github.com/kubernetes/kubernetes/pull/106169), [@kerthcet](https://github.com/kerthcet)) [SIG Scheduling]
- `client-go` event library allows customizing spam filtering function. 
  It is now possible to override `SpamKeyFunc`, which is used by event filtering to detect spam in the events. ([#103918](https://github.com/kubernetes/kubernetes/pull/103918), [@olagacek](https://github.com/olagacek))
- `client-go`, using log level 9, traces the following events of a HTTP request:
      - DNS lookup
      - TCP dialing
      - TLS handshake
      - Time to get a connection from the pool
      - Time to process a request ([#105156](https://github.com/kubernetes/kubernetes/pull/105156), [@aojea](https://github.com/aojea))
- `kube-scheduler` now logs node and plugin scoring  even though --v<10
  - scores of the top 3 plugins in the top 3 nodes are dumped if --v=4,5
  - scores of all plugins in the top 6 nodes are dumped if --v=6,7,8,9 ([#103515](https://github.com/kubernetes/kubernetes/pull/103515), [@muma378](https://github.com/muma378))

### Documentation

- Graduating `pod_scheduling_duration_seconds`, `pod_scheduling_attempts`, `framework_extension_point_duration_seconds`, `plugin_execution_duration_seconds` and `queue_incoming_pods_total` metrics to stable. ([#106266](https://github.com/kubernetes/kubernetes/pull/106266), [@ahg-g](https://github.com/ahg-g)) [SIG Instrumentation, Scheduling and Testing]
- The test "[sig-network] EndpointSlice should have Endpoints and EndpointSlices pointing to API Server [Conformance]" only requires that there is an EndpointSlice that references the "kubernetes.default" service, it no longer requires that its named "kubernetes". ([#104664](https://github.com/kubernetes/kubernetes/pull/104664), [@aojea](https://github.com/aojea))
- Update description of `--audit-log-maxbackup` to describe behavior when `value = 0`. ([#103843](https://github.com/kubernetes/kubernetes/pull/103843), [@Arkessler](https://github.com/Arkessler))
- Users should not rely on unsupported CRON_TZ variable when specifying schedule, both the API server and cronjob controller will emit warnings pointing to https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/ containing explanation ([#106455](https://github.com/kubernetes/kubernetes/pull/106455), [@soltysh](https://github.com/soltysh)) [SIG Apps]

### Failing Test

- Fixes hostPath storage E2E tests within SELinux enabled env ([#104551](https://github.com/kubernetes/kubernetes/pull/104551), [@Elbehery](https://github.com/Elbehery))

### Bug or Regression

- (PodSecurity admission) errors validating workload resources (deployment, replicaset, etc.) no longer block admission. ([#106017](https://github.com/kubernetes/kubernetes/pull/106017), [@tallclair](https://github.com/tallclair)) [SIG Auth]
- A pod that the Kubelet rejects was still considered as being accepted for a brief period of time after rejection, which might cause some pods to be rejected briefly that could fit on the node.  A pod that is still terminating (but has status indicating it has failed) may also still be consuming resources and so should also be considered. ([#104817](https://github.com/kubernetes/kubernetes/pull/104817), [@smarterclayton](https://github.com/smarterclayton))
- Add Kubernetes Events to the `Kubelet Graceful Shutdown` feature. ([#101081](https://github.com/kubernetes/kubernetes/pull/101081), [@rphillips](https://github.com/rphillips))
- Add Pod Security admission metrics: `pod_security_evaluations_total`, `pod_security_exemptions_total`, `pod_security_errors_total` ([#105898](https://github.com/kubernetes/kubernetes/pull/105898), [@tallclair](https://github.com/tallclair))
- Add support for Windows Network stats in Containerd ([#105744](https://github.com/kubernetes/kubernetes/pull/105744), [@jsturtevant](https://github.com/jsturtevant)) [SIG Node, Testing and Windows]
- Added show-capacity option to `kubectl top node` to show `Capacity` resource usage ([#102917](https://github.com/kubernetes/kubernetes/pull/102917), [@bysnupy](https://github.com/bysnupy)) [SIG CLI]
- Apimachinery: Pretty printed JSON and YAML output is now indented consistently. ([#105466](https://github.com/kubernetes/kubernetes/pull/105466), [@liggitt](https://github.com/liggitt))
- Be able to create a Pod with Generic Ephemeral Volumes as raw block devices. ([#105682](https://github.com/kubernetes/kubernetes/pull/105682), [@pohly](https://github.com/pohly))
- CA, certificate and key bundles for the `generic-apiserver` based servers will be reloaded immediately after the files are changed. ([#104102](https://github.com/kubernetes/kubernetes/pull/104102), [@tnqn](https://github.com/tnqn))
- Change `kubectl diff --invalid-arg` status code from 1 to 2 to match docs ([#105445](https://github.com/kubernetes/kubernetes/pull/105445), [@ardaguclu](https://github.com/ardaguclu))
- Changed kubectl describe to compute age of an event using the `EventSeries.count` and `EventSeries.lastObservedTime`. ([#104482](https://github.com/kubernetes/kubernetes/pull/104482), [@harjas27](https://github.com/harjas27))
- Changes behaviour of kube-proxy start; does not attempt to set specific `sysctl` values (which does not work in recent Kernel versions anymore in non-init namespaces), when the current sysctl values are already set higher. ([#103174](https://github.com/kubernetes/kubernetes/pull/103174), [@Napsty](https://github.com/Napsty))
- Client-go uses the same HTTP client for all the generated groups and versions, allowing to share customized transports for multiple groups versions. ([#105490](https://github.com/kubernetes/kubernetes/pull/105490), [@aojea](https://github.com/aojea))
- Disable aufs module for gce clusters. ([#103831](https://github.com/kubernetes/kubernetes/pull/103831), [@lizhuqi](https://github.com/lizhuqi))
- Do not unmount and mount subpath bind mounts during container creation unless bind mount changes ([#105512](https://github.com/kubernetes/kubernetes/pull/105512), [@gnufied](https://github.com/gnufied)) [SIG Storage]
- Don't prematurely close reflectors in case of slow initialization in watch based manager to fix issues with inability to properly mount secrets/configmaps. ([#104604](https://github.com/kubernetes/kubernetes/pull/104604), [@wojtek-t](https://github.com/wojtek-t))
- Don't use a custom dialer for the kubelet if is not rotating certificates, so we can reuse TCP connections and have only one between the apiserver and the kubelet.
  If users experiment problems with stale connections  using HTTP1.1, they can force the previous behavior of the kubelet by setting the environment variable DISABLE_HTTP2. ([#104844](https://github.com/kubernetes/kubernetes/pull/104844), [@aojea](https://github.com/aojea)) [SIG API Machinery, Auth and Node]
- EndpointSlice Mirroring controller now cleans up managed EndpointSlices when a Service selector is added ([#105997](https://github.com/kubernetes/kubernetes/pull/105997), [@robscott](https://github.com/robscott)) [SIG Apps, Network and Testing]
- Enhanced event messages for pod failed for exec probe timeout ([#106201](https://github.com/kubernetes/kubernetes/pull/106201), [@yxxhero](https://github.com/yxxhero)) [SIG Node]
- Ensure Pods are removed from the scheduler cache when the scheduler misses deletion events due to transient errors ([#106102](https://github.com/kubernetes/kubernetes/pull/106102), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Ensure `InstanceShutdownByProviderID` return false for creating Azure VMs. ([#104382](https://github.com/kubernetes/kubernetes/pull/104382), [@feiskyer](https://github.com/feiskyer))
- Evicted and other terminated Pods will no longer revert to the `Running` phase. ([#105462](https://github.com/kubernetes/kubernetes/pull/105462), [@ehashman](https://github.com/ehashman))
- Fix `kube-apiserver` metric reporting for the deprecated watch path of `/api/<version>/watch/...`. ([#104161](https://github.com/kubernetes/kubernetes/pull/104161), [@wojtek-t](https://github.com/wojtek-t))
- Fix a regression where the Kubelet failed to exclude already completed pods from calculations about how many resources it was currently using when deciding whether to allow more pods. ([#104577](https://github.com/kubernetes/kubernetes/pull/104577), [@smarterclayton](https://github.com/smarterclayton))
- Fix detach disk issue on deleting vmss node. ([#104572](https://github.com/kubernetes/kubernetes/pull/104572), [@andyzhangx](https://github.com/andyzhangx))
- Fix job controller syncs: In case of conflicts, ensure that the sync happens with the most up to date information. Improves reliability of JobTrackingWithFinalizers. ([#105214](https://github.com/kubernetes/kubernetes/pull/105214), [@alculquicondor](https://github.com/alculquicondor))
- Fix job tracking with finalizers for more than 500 pods, ensuring all finalizers are removed before counting the Pod. ([#104666](https://github.com/kubernetes/kubernetes/pull/104666), [@alculquicondor](https://github.com/alculquicondor))
- Fix pod name of NonIndexed Jobs to not include rogue `-1` substring ([#105676](https://github.com/kubernetes/kubernetes/pull/105676), [@alculquicondor](https://github.com/alculquicondor))
- Fix scoring for `NodeResourcesBalancedAllocation` plugins when nodes have containers with no requests. ([#105845](https://github.com/kubernetes/kubernetes/pull/105845), [@ahmad-diaa](https://github.com/ahmad-diaa))
- Fix system default topology spreading when nodes don't have zone labels. Pods correctly spread by default now. ([#105046](https://github.com/kubernetes/kubernetes/pull/105046), [@alculquicondor](https://github.com/alculquicondor))
- Fix: do not try to delete a LoadBalancer that does not exist ([#105777](https://github.com/kubernetes/kubernetes/pull/105777), [@nilo19](https://github.com/nilo19))
- Fix: ignore non-VMSS error for VMAS nodes in `reconcileBackendPools`. ([#103997](https://github.com/kubernetes/kubernetes/pull/103997), [@nilo19](https://github.com/nilo19))
- Fix: leave the probe path empty for TCP probes ([#105253](https://github.com/kubernetes/kubernetes/pull/105253), [@nilo19](https://github.com/nilo19))
- Fix: remove VMSS and VMSS instances from SLB backend pool only when necessary ([#105839](https://github.com/kubernetes/kubernetes/pull/105839), [@nilo19](https://github.com/nilo19))
- Fix: skip `instance not found` when decoupling VMSSs from LB ([#105666](https://github.com/kubernetes/kubernetes/pull/105666), [@nilo19](https://github.com/nilo19))
- Fix: skip case sensitivity when checking Azure NSG rules. ([#104384](https://github.com/kubernetes/kubernetes/pull/104384), [@feiskyer](https://github.com/feiskyer))
- Fixed a bug that prevents a PersistentVolume that has a PersistentVolumeClaim UID which doesn't exist in local cache but exists in etcd from being updated to the Released phase. ([#105211](https://github.com/kubernetes/kubernetes/pull/105211), [@xiaopingrubyist](https://github.com/xiaopingrubyist))
- Fixed a bug where using `kubectl patch` with `$deleteFromPrimitiveList` on a nonexistent or empty list would add the item to the list ([#105421](https://github.com/kubernetes/kubernetes/pull/105421), [@brianpursley](https://github.com/brianpursley))
- Fixed a bug which could cause webhooks to have an incorrect copy of the old object after an Apply or Update ([#106195](https://github.com/kubernetes/kubernetes/pull/106195), [@alexzielenski](https://github.com/alexzielenski)) [SIG API Machinery]
- Fixed a bug which kubectl would emit duplicate warning messages for flag names that contain an underscore and recommend using a nonexistent flag in some cases. ([#103852](https://github.com/kubernetes/kubernetes/pull/103852), [@brianpursley](https://github.com/brianpursley))
- Fixed a panic in `kubectl` when creating secrets with an improper output type ([#106317](https://github.com/kubernetes/kubernetes/pull/106317), [@lauchokyip](https://github.com/lauchokyip))
- Fixed a regression setting `--audit-log-path=-` to log to stdout in 1.22 pre-release. ([#103875](https://github.com/kubernetes/kubernetes/pull/103875), [@andrewrynhard](https://github.com/andrewrynhard))
- Fixed an issue which didn't append OS's environment variables with the one provided in Credential Provider Config file, which may fail execution of external credential provider binary. 
  See https://github.com/kubernetes/kubernetes/issues/102750. ([#103231](https://github.com/kubernetes/kubernetes/pull/103231), [@n4j](https://github.com/n4j))
- Fixed applying of SELinux labels to CSI volumes on very busy systems (with "error checking for SELinux support: could not get consistent content of /proc/self/mountinfo after 3 attempts") ([#105934](https://github.com/kubernetes/kubernetes/pull/105934), [@jsafrane](https://github.com/jsafrane)) [SIG Storage]
- Fixed architecture within manifest for non `amd64` etcd images. ([#104116](https://github.com/kubernetes/kubernetes/pull/104116), [@saschagrunert](https://github.com/saschagrunert))
- Fixed architecture within manifest for non `amd64` etcd images. ([#105484](https://github.com/kubernetes/kubernetes/pull/105484), [@saschagrunert](https://github.com/saschagrunert))
- Fixed azure disk translation issue due to lower case `managed` kind. ([#103439](https://github.com/kubernetes/kubernetes/pull/103439), [@andyzhangx](https://github.com/andyzhangx))
- Fixed client IP preservation for NodePort service with protocol SCTP in ipvs. ([#104756](https://github.com/kubernetes/kubernetes/pull/104756), [@tnqn](https://github.com/tnqn))
- Fixed concurrent map access causing panics when logging timed-out API calls. ([#105734](https://github.com/kubernetes/kubernetes/pull/105734), [@marseel](https://github.com/marseel))
- Fixed consolidate logs for `instance not found` error
  Fixed skip `not found` nodes when reconciling LB backend address pools ([#105188](https://github.com/kubernetes/kubernetes/pull/105188), [@nilo19](https://github.com/nilo19))
- Fixed occasional pod cgroup freeze when using cgroup v1 and systemd driver. ([#104528](https://github.com/kubernetes/kubernetes/pull/104528), [@kolyshkin](https://github.com/kolyshkin))
- Fixed the issue where logging output of kube-scheduler configuration files included line breaks and escape characters. The output also attempted to output the configuration file in one section without showing the user a more readable format. ([#106228](https://github.com/kubernetes/kubernetes/pull/106228), [@sanchayanghosh](https://github.com/sanchayanghosh)) [SIG Scheduling]
- Fixes a bug that could result in the EndpointSlice controller unnecessarily updating EndpointSlices associated with a Service that had Topology Aware Hints enabled. ([#105267](https://github.com/kubernetes/kubernetes/pull/105267), [@llhuii](https://github.com/llhuii))
- Fixes a regression that could cause panics in LRU caches in controller-manager, kubelet, kube-apiserver, or client-go. ([#104466](https://github.com/kubernetes/kubernetes/pull/104466), [@stbenjam](https://github.com/stbenjam))
- Fixes an issue where an admission webhook can observe a v1 Pod object that does not have the `defaultMode` field set in the injected service account token volume in kube-api-server. ([#104523](https://github.com/kubernetes/kubernetes/pull/104523), [@liggitt](https://github.com/liggitt))
- Fixes the `should support building a client with a CSR` E2E test to work with clusters configured with short certificate lifetimes ([#105396](https://github.com/kubernetes/kubernetes/pull/105396), [@liggitt](https://github.com/liggitt))
- Graceful node shutdown, allow the actual inhibit delay to be greater than the expected inhibit delay. ([#103137](https://github.com/kubernetes/kubernetes/pull/103137), [@wzshiming](https://github.com/wzshiming))
- Handle Generic Ephemeral Volumes properly in the node limits scheduler filter and the kubelet `hostPath` check. ([#100482](https://github.com/kubernetes/kubernetes/pull/100482), [@pohly](https://github.com/pohly))
- Headless Services with no selector which were created without dual-stack enabled will be defaulted to RequireDualStack instead of PreferDualStack.  This is consistent with such Services which are created with dual-stack enabled. ([#104986](https://github.com/kubernetes/kubernetes/pull/104986), [@thockin](https://github.com/thockin))
- Ignore `not a vmss instance` error for VMAS nodes in `EnsureBackendPoolDeleted`. ([#105185](https://github.com/kubernetes/kubernetes/pull/105185), [@ialidzhikov](https://github.com/ialidzhikov))
- Ignore the case when comparing azure tags in service annotation. ([#104705](https://github.com/kubernetes/kubernetes/pull/104705), [@nilo19](https://github.com/nilo19))
- Ignore the case when updating Azure tags. ([#104593](https://github.com/kubernetes/kubernetes/pull/104593), [@nilo19](https://github.com/nilo19))
- Introduce a new server run option 'shutdown-send-retry-after'. If true the HTTP Server will continue listening until all non longrunning request(s) in flight have been drained, during this window all incoming requests will be rejected with a status code `429` and a 'Retry-After' response header. ([#101257](https://github.com/kubernetes/kubernetes/pull/101257), [@tkashem](https://github.com/tkashem))
- Kube-apiserver: Avoid unnecessary repeated calls to `admission webhooks` that reject an update or delete request. ([#104182](https://github.com/kubernetes/kubernetes/pull/104182), [@liggitt](https://github.com/liggitt))
- Kube-apiserver: Server Side Apply merge order is reverted to match v1.22 behavior until `http://issue.k8s.io/104641` is resolved. ([#106661](https://github.com/kubernetes/kubernetes/pull/106661), [@liggitt](https://github.com/liggitt))
- Kube-apiserver: events created via the `events.k8s.io` API group for cluster-scoped objects are now permitted in the default namespace as well for compatibility with events clients and the `v1` API ([#100125](https://github.com/kubernetes/kubernetes/pull/100125), [@h4ghhh](https://github.com/h4ghhh))
- Kube-apiserver: fix a memory leak when deleting multiple objects with a `deletecollection`. ([#105606](https://github.com/kubernetes/kubernetes/pull/105606), [@sxllwx](https://github.com/sxllwx))
- Kube-proxy health check ports used to listen to `:<port>` for each of the services. This is not needed and opens ports in addresses the cluster user may not have intended. The PR limits listening to all node address which are controlled by `--nodeport-addresses` flag. if no addresses are provided then we default to existing behavior by listening to `:<port>` for each service ([#104742](https://github.com/kubernetes/kubernetes/pull/104742), [@khenidak](https://github.com/khenidak))
- Kube-proxy: delete stale conntrack UDP entries for loadbalancer ingress IP. ([#104009](https://github.com/kubernetes/kubernetes/pull/104009), [@aojea](https://github.com/aojea))
- Kube-scheduler now doesn't print any usage message when unknown flag is specified. ([#104503](https://github.com/kubernetes/kubernetes/pull/104503), [@sanposhiho](https://github.com/sanposhiho))
- Kube-up now includes CoreDNS version v1.8.6 ([#106091](https://github.com/kubernetes/kubernetes/pull/106091), [@rajansandeep](https://github.com/rajansandeep)) [SIG Cloud Provider]
- Kubeadm: When adding an etcd peer to an existing cluster, if an error is returned indicating the peer has already been added, this is accepted and a `ListMembers` call is used instead to return the existing cluster. This helps to diminish the exponential backoff when the first AddMember call times out, while still retaining a similar performance when the peer has already been added from a previous call. ([#104134](https://github.com/kubernetes/kubernetes/pull/104134), [@ihgann](https://github.com/ihgann))
- Kubeadm: do not allow empty `--config` paths to be passed to `kubeadm kubeconfig user` ([#105649](https://github.com/kubernetes/kubernetes/pull/105649), [@navist2020](https://github.com/navist2020))
- Kubeadm: fix a bug on Windows worker nodes, where the downloaded KubeletConfiguration from the cluster can contain Linux paths that do not work on Windows and can trip the kubelet binary. ([#105992](https://github.com/kubernetes/kubernetes/pull/105992), [@hwdef](https://github.com/hwdef)) [SIG Cluster Lifecycle and Windows]
- Kubeadm: switch the preflight check (called 'Swap') that verifies if swap is enabled on Linux hosts to report a warning instead of an error. This is related to the graduation of the NodeSwap feature gate in the kubelet to Beta and being enabled by default in 1.23 - allows swap support on Linux hosts. In the next release of kubeadm (1.24) the preflight check will be removed, thus we recommend that you stop using it - e.g. via `--ignore-preflight-errors` or the kubeadm config. ([#104854](https://github.com/kubernetes/kubernetes/pull/104854), [@pacoxu](https://github.com/pacoxu))
- Kubelet did not report `kubelet_volume_stats_*` metrics for Generic Ephemeral Volumes. ([#105569](https://github.com/kubernetes/kubernetes/pull/105569), [@pohly](https://github.com/pohly))
- Kubelet's Node Grace Shutdown will terminate probes when shutting down ([#105215](https://github.com/kubernetes/kubernetes/pull/105215), [@rphillips](https://github.com/rphillips))
- Kubelet: fixes a file descriptor leak in log rotation ([#106382](https://github.com/kubernetes/kubernetes/pull/106382), [@rphillips](https://github.com/rphillips)) [SIG Node]
- Kubelet: the printing of flags at the start of kubelet now uses the final logging configuration. ([#106520](https://github.com/kubernetes/kubernetes/pull/106520), [@pohly](https://github.com/pohly))
- Make the etcd client (used by the API server) retry certain types of errors. The full list of retriable (codes.Unavailable) errors can be found at https://github.com/etcd-io/etcd/blob/main/api/v3rpc/rpctypes/error.go#L72 ([#105069](https://github.com/kubernetes/kubernetes/pull/105069), [@p0lyn0mial](https://github.com/p0lyn0mial))
- Metrics changes: Fix exposed buckets of `scheduler_volume_scheduling_duration_seconds_bucket` metric. ([#100720](https://github.com/kubernetes/kubernetes/pull/100720), [@dntosas](https://github.com/dntosas))
- Migrated kubernetes object references (= name + namespace) to structured logging when using JSON as log output format ([#104877](https://github.com/kubernetes/kubernetes/pull/104877), [@pohly](https://github.com/pohly))
- Pass additional flags to subpath mount to avoid flakes in certain conditions. ([#104253](https://github.com/kubernetes/kubernetes/pull/104253), [@mauriciopoppe](https://github.com/mauriciopoppe))
- Pod SecurityContext sysctls name parameter for update requests where the existing object's sysctl contains slashes and kubelet sysctl whitelist support contains slashes. ([#102393](https://github.com/kubernetes/kubernetes/pull/102393), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Apps, Auth, Node, Storage and Testing]
- Pod will not start when Init container was OOM killed. ([#104650](https://github.com/kubernetes/kubernetes/pull/104650), [@yxxhero](https://github.com/yxxhero)) [SIG Node]
- PodResources interface was changed, now it returns only isolated CPUs ([#97415](https://github.com/kubernetes/kubernetes/pull/97415), [@AlexeyPerevalov](https://github.com/AlexeyPerevalov))
- Provide IPv6 support for internal load balancer. ([#103794](https://github.com/kubernetes/kubernetes/pull/103794), [@nilo19](https://github.com/nilo19))
- Reduce the number of calls to docker for stats via dockershim. For Windows this reduces the latency when calling docker, for Linux this saves cpu cycles. ([#104287](https://github.com/kubernetes/kubernetes/pull/104287), [@jsturtevant](https://github.com/jsturtevant)) [SIG Node and Windows]
- Removed the error message label from the `kubelet_started_pods_errors_total` metric ([#105213](https://github.com/kubernetes/kubernetes/pull/105213), [@yxxhero](https://github.com/yxxhero))
- Resolves a potential issue with GC and NS controllers which may delete objects after getting a 404 response from the server during its startup. This PR ensures that requests to aggregated APIs will get 503, not 404 while the APIServiceRegistrationController hasn't finished its job. ([#104748](https://github.com/kubernetes/kubernetes/pull/104748), [@p0lyn0mial](https://github.com/p0lyn0mial))
- Respect grace period when updating static pods. ([#104743](https://github.com/kubernetes/kubernetes/pull/104743), [@gjkim42](https://github.com/gjkim42)) [SIG Node and Testing]
- Revert building binaries with PIE mode. ([#105352](https://github.com/kubernetes/kubernetes/pull/105352), [@ehashman](https://github.com/ehashman))
- Reverts adding namespace label to admission metrics (and histogram exansion) due to cardinality issues. ([#104033](https://github.com/kubernetes/kubernetes/pull/104033), [@s-urbaniak](https://github.com/s-urbaniak))
- Reverts the CRI API version surfaced by dockershim to `v1alpha2`. ([#106808](https://github.com/kubernetes/kubernetes/pull/106808), [@saschagrunert](https://github.com/saschagrunert))
- Scheduler resource metrics over fractional binary quantities (2.5Gi, 1.1Ki) were incorrectly reported as very small values. ([#103751](https://github.com/kubernetes/kubernetes/pull/103751), [@y-tag](https://github.com/y-tag))
- Support more than 100 disk mounts on Windows ([#105673](https://github.com/kubernetes/kubernetes/pull/105673), [@andyzhangx](https://github.com/andyzhangx))
- Support using negative array index in JSON patch replace operations. ([#105896](https://github.com/kubernetes/kubernetes/pull/105896), [@zqzten](https://github.com/zqzten))
- The `--leader-elect*` CLI args are now honored in scheduler. ([#105915](https://github.com/kubernetes/kubernetes/pull/105915), [@Huang-Wei](https://github.com/Huang-Wei))
- The `--leader-elect*` CLI args are now honored in the scheduler. ([#105712](https://github.com/kubernetes/kubernetes/pull/105712), [@Huang-Wei](https://github.com/Huang-Wei))
- The `client-go` dynamic client sets the header `Content-Type: application/json` by default ([#104327](https://github.com/kubernetes/kubernetes/pull/104327), [@sxllwx](https://github.com/sxllwx))
- The `kube-Proxy` now correctly filters out unready endpoints for Services with Topology. ([#106507](https://github.com/kubernetes/kubernetes/pull/106507), [@robscott](https://github.com/robscott))
- The `pods/binding` subresource now honors `metadata.uid` and `metadata.resourceVersion` ([#105913](https://github.com/kubernetes/kubernetes/pull/105913), [@aholic](https://github.com/aholic))
- The kube-proxy sync_proxy_rules_iptables_total metric now gives
  the correct number of rules, rather than being off by one.
  
  Fixed multiple iptables proxy regressions introduced in 1.22:
  
    - When using Services with SessionAffinity, client affinity for an
      endpoint now gets broken when that endpoint becomes non-ready
      (rather than continuing until the endpoint is fully deleted).
  
    - Traffic to a service IP now starts getting rejected (as opposed to
      merely dropped) as soon as there are no longer any *usable*
      endpoints, rather than waiting until all of the terminating
      endpoints have terminated even when those terminating endpoints
      were not being used.
  
    - Chains for endpoints that won't be used are no longer output to
      iptables, saving a bit of memory/time/cpu. ([#106030](https://github.com/kubernetes/kubernetes/pull/106030), [@danwinship](https://github.com/danwinship)) [SIG Network]
- Topology Aware Hints now ignores unready endpoints when assigning hints. ([#106510](https://github.com/kubernetes/kubernetes/pull/106510), [@robscott](https://github.com/robscott))
- Topology Hints now excludes control plane notes from capacity calculations. ([#104744](https://github.com/kubernetes/kubernetes/pull/104744), [@robscott](https://github.com/robscott))
- Update Go used to build migrate script in etcd image to v1.16.7. ([#104301](https://github.com/kubernetes/kubernetes/pull/104301), [@serathius](https://github.com/serathius))
- Updated json representation for a conflicted taint to `Key=Effect` when a conflicted taint occurs in kubectl taint. ([#104011](https://github.com/kubernetes/kubernetes/pull/104011), [@manugupt1](https://github.com/manugupt1))
- Upgrades functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv4.4.1 ([#106389](https://github.com/kubernetes/kubernetes/pull/106389), [@natasha41575](https://github.com/natasha41575)) [SIG CLI]
- Watch requests that are delegated to aggregated API servers no longer reserve concurrency units (seats) in the API Priority and Fairness dispatcher for their entire duration. ([#105511](https://github.com/kubernetes/kubernetes/pull/105511), [@benluddy](https://github.com/benluddy))
- When a static pod file is deleted and recreated while using a fixed UID, the pod was not properly restarted. ([#104847](https://github.com/kubernetes/kubernetes/pull/104847), [@smarterclayton](https://github.com/smarterclayton))
- XFS-filesystems are now force-formatted (option `-f`) in order to avoid problems being formatted due to detection of magic super-blocks. This aligns with the behaviour of formatting of ext3/4 filesystems. ([#104923](https://github.com/kubernetes/kubernetes/pull/104923), [@davidkarlsen](https://github.com/davidkarlsen))
- `--log-flush-frequency` had no effect in several commands or was missing. Help and warning texts were not always using the right format for a command (`add_dir_header` instead of `add-dir-header`). Fixing this included cleaning up flag handling in component-base/logs: that package no longer adds flags to the global flag sets. Commands which want the klog and `--log-flush-frequency` flags must explicitly call `logs.AddFlags`; the new `cli.Run` does that for commands. That helper function also covers flag normalization and printing of usage and errors in a consistent way (print usage text first if parsing failed, then the error). ([#105076](https://github.com/kubernetes/kubernetes/pull/105076), [@pohly](https://github.com/pohly))

### Other (Cleanup or Flake)

- All `klog` flags except for `-v` and `-vmodule` are deprecated. Support for `-vmodule` is only guaranteed for the text log format. ([#105042](https://github.com/kubernetes/kubernetes/pull/105042), [@pohly](https://github.com/pohly))
- Better pod events ("waiting for ephemeral volume controller to create the persistentvolumeclaim"" instead of "persistentvolumeclaim not found") when using generic ephemeral volumes. ([#104605](https://github.com/kubernetes/kubernetes/pull/104605), [@pohly](https://github.com/pohly))
- Changed buckets in apiserver_request_duration_seconds metric from [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60] to [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60] ([#106306](https://github.com/kubernetes/kubernetes/pull/106306), [@pawbana](https://github.com/pawbana)) [SIG API Machinery, Instrumentation and Testing]
- Deprecate `apiserver_longrunning_gauge` and `apiserver_register_watchers` in 1.23.0. ([#103793](https://github.com/kubernetes/kubernetes/pull/103793), [@yan-lgtm](https://github.com/yan-lgtm))
- Enhanced error message for nodes not selected by scheduler due to pod's PersistentVolumeClaim(s) bound to PersistentVolume(s) that do not exist. ([#105196](https://github.com/kubernetes/kubernetes/pull/105196), [@yibozhuang](https://github.com/yibozhuang))
- Fix an issue in cleaning up `CertificateSigningRequest` objects with an unparseable `status.certificate` field. ([#103823](https://github.com/kubernetes/kubernetes/pull/103823), [@liggitt](https://github.com/liggitt))
- Kube-apiserver: requests to node, Service, and Pod `/proxy` subresources with no additional URL path now only automatically redirect GET and HEAD requests. ([#95128](https://github.com/kubernetes/kubernetes/pull/95128), [@Riaankl](https://github.com/Riaankl))
- Kube-apiserver: sets an upper-bound on the lifetime of idle keep-alive connections and the time to read the headers of incoming requests. ([#103958](https://github.com/kubernetes/kubernetes/pull/103958), [@liggitt](https://github.com/liggitt))
- Kubeadm: external etcd endpoints passed in the `ClusterConfiguration` that have Unicode characters are no longer IDNA encoded (converted to Punycode). They are now just URL encoded as per Go's implementation of RFC-3986, have duplicate "/" removed from the URL paths, and passed like that directly to the `kube-apiserver` `--etcd-servers` flag. If you have etcd endpoints that have Unicode characters, it is advisable to encode them in advance with tooling that is fully IDNA compliant. If you don't do that, the Go standard library (used in k8s and etcd) would do it for you when making requests to the endpoints. ([#103801](https://github.com/kubernetes/kubernetes/pull/103801), [@gkarthiks](https://github.com/gkarthiks))
- Kubeadm: remove the `--port` flag from the manifest for the `kube-controller-manager` since the flag has been a NO-OP since 1.22 and insecure serving was removed for the component. ([#104157](https://github.com/kubernetes/kubernetes/pull/104157), [@knight42](https://github.com/knight42))
- Kubeadm: remove the `--port` flag from the manifest for the kube-scheduler since the flag has been a NO-OP since 1.23 and insecure serving was removed for the component. ([#105034](https://github.com/kubernetes/kubernetes/pull/105034), [@pacoxu](https://github.com/pacoxu))
- Kubeadm: update references to legacy artifacts locations, the `ci-cross` prefix has been removed from the version match as it does not exist in the new `gs://k8s-release-dev` bucket. ([#103813](https://github.com/kubernetes/kubernetes/pull/103813), [@SataQiu](https://github.com/SataQiu))
- Kubectl: deprecated command line flags (like several of the klog flags) now have a `DEPRECATED: <explanation>` comment. ([#106172](https://github.com/kubernetes/kubernetes/pull/106172), [@pohly](https://github.com/pohly)) [SIG CLI]
- Kubemark is now built as a portable, static binary. ([#106150](https://github.com/kubernetes/kubernetes/pull/106150), [@pohly](https://github.com/pohly)) [SIG Scalability and Testing]
- Migrate `cmd/proxy/{config, healthcheck, winkernel}` to structured logging ([#104944](https://github.com/kubernetes/kubernetes/pull/104944), [@jyz0309](https://github.com/jyz0309))
- Migrate `pkg/proxy` to structured logging ([#104908](https://github.com/kubernetes/kubernetes/pull/104908), [@CIPHERTron](https://github.com/CIPHERTron))
- Migrate `pkg/scheduler/framework/plugins/interpodaffinity/filtering.go`,`pkg/scheduler/framework/plugins/podtopologyspread/filtering.go`, `pkg/scheduler/framework/plugins/volumezone/volume_zone.go` to structured logging ([#105931](https://github.com/kubernetes/kubernetes/pull/105931), [@mengjiao-liu](https://github.com/mengjiao-liu))
- Migrate `pkg/scheduler` to structured logging. ([#99273](https://github.com/kubernetes/kubernetes/pull/99273), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085))
- Migrate cmd/proxy/app and pkg/proxy/meta_proxier to structured logging ([#104928](https://github.com/kubernetes/kubernetes/pull/104928), [@jyz0309](https://github.com/jyz0309))
- Migrated `cmd/kube-scheduler/app/server.go`, `pkg/scheduler/framework/plugins/nodelabel/node_label.go`, `pkg/scheduler/framework/plugins/nodevolumelimits/csi.go`, `pkg/scheduler/framework/plugins/nodevolumelimits/non_csi.go` to structured logging ([#105855](https://github.com/kubernetes/kubernetes/pull/105855), [@shivanshu1333](https://github.com/shivanshu1333))
- Migrated `pkg/proxy/ipvs` to structured logging ([#104932](https://github.com/kubernetes/kubernetes/pull/104932), [@shivanshu1333](https://github.com/shivanshu1333))
- Migrated `pkg/proxy/userspace` to structured logging. ([#104931](https://github.com/kubernetes/kubernetes/pull/104931), [@shivanshu1333](https://github.com/shivanshu1333))
- Migrated `pkg/proxy` to structured logging ([#104891](https://github.com/kubernetes/kubernetes/pull/104891), [@shivanshu1333](https://github.com/shivanshu1333))
- Migrated `pkg/scheduler/framework/plugins/volumebinding/assume_cache.go` to structured logging. ([#105904](https://github.com/kubernetes/kubernetes/pull/105904), [@mengjiao-liu](https://github.com/mengjiao-liu)) [SIG Instrumentation, Scheduling and Storage]
- Migrated `pkg/scheduler/framework/preemption/preemption.go`, `pkg/scheduler/framework/plugins/examples/stateful/stateful.go`, and `pkg/scheduler/framework/plugins/noderesources/resource_allocation.go` to structured logging ([#105967](https://github.com/kubernetes/kubernetes/pull/105967), [@shivanshu1333](https://github.com/shivanshu1333)) [SIG Instrumentation, Node and Scheduling]
- Migrated pkg/proxy/winuserspace to structured logging ([#105035](https://github.com/kubernetes/kubernetes/pull/105035), [@shivanshu1333](https://github.com/shivanshu1333))
- Migrated scheduler file `cache.go` to structured logging ([#105969](https://github.com/kubernetes/kubernetes/pull/105969), [@shivanshu1333](https://github.com/shivanshu1333)) [SIG Instrumentation and Scheduling]
- Migrated scheduler files `comparer.go`, `dumper.go`, `node_tree.go` to structured logging ([#105968](https://github.com/kubernetes/kubernetes/pull/105968), [@shivanshu1333](https://github.com/shivanshu1333)) [SIG Instrumentation and Scheduling]
- More detailed logging has been added to the EndpointSlice controller for Topology. ([#104741](https://github.com/kubernetes/kubernetes/pull/104741), [@robscott](https://github.com/robscott))
- Remove deprecated and not supported old cronjob controller. ([#106126](https://github.com/kubernetes/kubernetes/pull/106126), [@soltysh](https://github.com/soltysh)) [SIG Apps]
- Remove ignore error flag for drain, and set this feature as default ([#105571](https://github.com/kubernetes/kubernetes/pull/105571), [@yuzhiquan](https://github.com/yuzhiquan)) [SIG CLI]
- Remove the deprecated flags `--csr-only` and `--csr-dir` from `kubeadm certs renew`. Please use `kubeadm certs generate-csr` instead. ([#104796](https://github.com/kubernetes/kubernetes/pull/104796), [@RA489](https://github.com/RA489))
- Support allocating whole NUMA nodes in the CPUManager when there is not a 1:1 mapping between socket and NUMA node ([#102015](https://github.com/kubernetes/kubernetes/pull/102015), [@klueska](https://github.com/klueska))
- Support for Windows Server 2022 was added to the `k8s.gcr.io/pause:3.6` image. ([#104711](https://github.com/kubernetes/kubernetes/pull/104711), [@claudiubelu](https://github.com/claudiubelu))
- Surface warning when users don't set `propagationPolicy` for jobs while deleting. ([#104080](https://github.com/kubernetes/kubernetes/pull/104080), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla))
- The `AllowInsecureBackendProxy` feature gate is removed. It reached GA in Kubernetes 1.21. ([#103796](https://github.com/kubernetes/kubernetes/pull/103796), [@mengjiao-liu](https://github.com/mengjiao-liu))
- The `BoundServiceAccountTokenVolume` feature gate that is GA since v1.22 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104167](https://github.com/kubernetes/kubernetes/pull/104167), [@ialidzhikov](https://github.com/ialidzhikov))
- The `StartupProbe` feature gate that is GA since v1.20 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104168](https://github.com/kubernetes/kubernetes/pull/104168), [@ialidzhikov](https://github.com/ialidzhikov))
- The `SupportPodPidsLimit` and  `SupportNodePidsLimit` feature gates that are GA since v1.20 are unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104163](https://github.com/kubernetes/kubernetes/pull/104163), [@ialidzhikov](https://github.com/ialidzhikov))
- The `apiserver` exposes 4 new metrics that allow to track the status of the Service CIDRs allocations:
      - current number of available IPs per Service CIDR
      - current number of used IPs per Service CIDR
      - total number of allocation per Service CIDR
      - total number of allocation errors per ServiceCIDR ([#104119](https://github.com/kubernetes/kubernetes/pull/104119), [@aojea](https://github.com/aojea))
- The flag `--deployment-controller-sync-period` has been deprecated and will be removed in v1.24. ([#103538](https://github.com/kubernetes/kubernetes/pull/103538), [@Pingan2017](https://github.com/Pingan2017))
- The image `gcr.io/kubernetes-e2e-test-images` will no longer be used in E2E / CI testing, `k8s.gcr.io/e2e-test-images` will be used instead. ([#103724](https://github.com/kubernetes/kubernetes/pull/103724), [@claudiubelu](https://github.com/claudiubelu))
- The kube-proxy image contains `/go-runner` as a replacement for deprecated klog flags. ([#106301](https://github.com/kubernetes/kubernetes/pull/106301), [@pohly](https://github.com/pohly))
- The maximum length of the `CSINode` id field has increased to 256 bytes to match the CSI spec. ([#104160](https://github.com/kubernetes/kubernetes/pull/104160), [@pacoxu](https://github.com/pacoxu))
- Troubleshooting: informers log handlers that take more than 100 milliseconds to process an object if the `DeltaFIFO` queue starts to grow beyond 10 elements. ([#103917](https://github.com/kubernetes/kubernetes/pull/103917), [@aojea](https://github.com/aojea))
- Update `cri-tools` dependency to v1.22.0. ([#104430](https://github.com/kubernetes/kubernetes/pull/104430), [@saschagrunert](https://github.com/saschagrunert))
- Update `migratecmd/kube-proxy/app` logs to structured logging. ([#98913](https://github.com/kubernetes/kubernetes/pull/98913), [@yxxhero](https://github.com/yxxhero))
- Update build images to Debian 11 (Bullseye)
  - debian-base:bullseye-v1.0.0
  - debian-iptables:bullseye-v1.0.0
  - go-runner:v2.3.1-go1.17.1-bullseye.0
  - kube-cross:v1.23.0-go1.17.1-bullseye.1
  - setcap:bullseye-v1.0.0
  - cluster/images/etcd: Build 3.5.0-2 image
  - test/conformance/image: Update runner image to base-debian11 ([#105158](https://github.com/kubernetes/kubernetes/pull/105158), [@justaugustus](https://github.com/justaugustus))
- Update conformance image to use `debian-base:buster-v1.9.0`. ([#104696](https://github.com/kubernetes/kubernetes/pull/104696), [@PushkarJ](https://github.com/PushkarJ))
- `volume.kubernetes.io/storage-provisioner` annotation will be added to dynamic provisioning required PVC. `volume.beta.kubernetes.io/storage-provisioner` annotation is deprecated. ([#104590](https://github.com/kubernetes/kubernetes/pull/104590), [@Jiawei0227](https://github.com/Jiawei0227))

## Dependencies

### Added
- bazil.org/fuse: 371fbbd
- github.com/OneOfOne/xxhash: [v1.2.2](https://github.com/OneOfOne/xxhash/tree/v1.2.2)
- github.com/antlr/antlr4/runtime/Go/antlr: [b48c857](https://github.com/antlr/antlr4/runtime/Go/antlr/tree/b48c857)
- github.com/cespare/xxhash: [v1.1.0](https://github.com/cespare/xxhash/tree/v1.1.0)
- github.com/cncf/xds/go: [fbca930](https://github.com/cncf/xds/go/tree/fbca930)
- github.com/getkin/kin-openapi: [v0.76.0](https://github.com/getkin/kin-openapi/tree/v0.76.0)
- github.com/go-logr/zapr: [v1.2.0](https://github.com/go-logr/zapr/tree/v1.2.0)
- github.com/google/cel-go: [v0.9.0](https://github.com/google/cel-go/tree/v0.9.0)
- github.com/google/cel-spec: [v0.6.0](https://github.com/google/cel-spec/tree/v0.6.0)
- github.com/google/martian/v3: [v3.1.0](https://github.com/google/martian/v3/tree/v3.1.0)
- github.com/kr/fs: [v0.1.0](https://github.com/kr/fs/tree/v0.1.0)
- github.com/pkg/sftp: [v1.10.1](https://github.com/pkg/sftp/tree/v1.10.1)
- github.com/spaolacci/murmur3: [f09979e](https://github.com/spaolacci/murmur3/tree/f09979e)
- sigs.k8s.io/json: c049b76

### Changed
- cloud.google.com/go/bigquery: v1.4.0 → v1.8.0
- cloud.google.com/go/storage: v1.6.0 → v1.10.0
- cloud.google.com/go: v0.54.0 → v0.81.0
- github.com/GoogleCloudPlatform/k8s-cloud-provider: [7901bc8 → ea6160c](https://github.com/GoogleCloudPlatform/k8s-cloud-provider/compare/7901bc8...ea6160c)
- github.com/Microsoft/go-winio: [v0.4.15 → v0.4.17](https://github.com/Microsoft/go-winio/compare/v0.4.15...v0.4.17)
- github.com/Microsoft/hcsshim: [5eafd15 → v0.8.22](https://github.com/Microsoft/hcsshim/compare/5eafd15...v0.8.22)
- github.com/benbjohnson/clock: [v1.0.3 → v1.1.0](https://github.com/benbjohnson/clock/compare/v1.0.3...v1.1.0)
- github.com/bketelsen/crypt: [5cbc8cc → v0.0.4](https://github.com/bketelsen/crypt/compare/5cbc8cc...v0.0.4)
- github.com/containerd/cgroups: [0dbf7f0 → v1.0.1](https://github.com/containerd/cgroups/compare/0dbf7f0...v1.0.1)
- github.com/containerd/containerd: [v1.4.4 → v1.4.11](https://github.com/containerd/containerd/compare/v1.4.4...v1.4.11)
- github.com/containerd/continuity: [aaeac12 → v0.1.0](https://github.com/containerd/continuity/compare/aaeac12...v0.1.0)
- github.com/containerd/fifo: [a9fb20d → v1.0.0](https://github.com/containerd/fifo/compare/a9fb20d...v1.0.0)
- github.com/containerd/go-runc: [5a6d9f3 → v1.0.0](https://github.com/containerd/go-runc/compare/5a6d9f3...v1.0.0)
- github.com/containerd/typeurl: [v1.0.1 → v1.0.2](https://github.com/containerd/typeurl/compare/v1.0.1...v1.0.2)
- github.com/coredns/corefile-migration: [v1.0.12 → v1.0.14](https://github.com/coredns/corefile-migration/compare/v1.0.12...v1.0.14)
- github.com/docker/docker: [v20.10.2+incompatible → v20.10.7+incompatible](https://github.com/docker/docker/compare/v20.10.2...v20.10.7)
- github.com/envoyproxy/go-control-plane: [668b12f → 63b5d3c](https://github.com/envoyproxy/go-control-plane/compare/668b12f...63b5d3c)
- github.com/evanphx/json-patch: [v4.11.0+incompatible → v4.12.0+incompatible](https://github.com/evanphx/json-patch/compare/v4.11.0...v4.12.0)
- github.com/go-logr/logr: [v0.4.0 → v1.2.0](https://github.com/go-logr/logr/compare/v0.4.0...v1.2.0)
- github.com/golang/glog: [23def4e → v1.0.0](https://github.com/golang/glog/compare/23def4e...v1.0.0)
- github.com/golang/mock: [v1.4.4 → v1.5.0](https://github.com/golang/mock/compare/v1.4.4...v1.5.0)
- github.com/google/cadvisor: [v0.39.2 → v0.43.0](https://github.com/google/cadvisor/compare/v0.39.2...v0.43.0)
- github.com/google/pprof: [1ebb73c → cbba55b](https://github.com/google/pprof/compare/1ebb73c...cbba55b)
- github.com/hashicorp/golang-lru: [v0.5.1 → v0.5.0](https://github.com/hashicorp/golang-lru/compare/v0.5.1...v0.5.0)
- github.com/ianlancetaylor/demangle: [5e5cf60 → 28f6c0f](https://github.com/ianlancetaylor/demangle/compare/5e5cf60...28f6c0f)
- github.com/json-iterator/go: [v1.1.11 → v1.1.12](https://github.com/json-iterator/go/compare/v1.1.11...v1.1.12)
- github.com/magiconair/properties: [v1.8.1 → v1.8.5](https://github.com/magiconair/properties/compare/v1.8.1...v1.8.5)
- github.com/mitchellh/go-homedir: [v1.1.0 → v1.0.0](https://github.com/mitchellh/go-homedir/compare/v1.1.0...v1.0.0)
- github.com/mitchellh/mapstructure: [v1.1.2 → v1.4.1](https://github.com/mitchellh/mapstructure/compare/v1.1.2...v1.4.1)
- github.com/modern-go/reflect2: [v1.0.1 → v1.0.2](https://github.com/modern-go/reflect2/compare/v1.0.1...v1.0.2)
- github.com/opencontainers/runc: [v1.0.1 → v1.0.2](https://github.com/opencontainers/runc/compare/v1.0.1...v1.0.2)
- github.com/pelletier/go-toml: [v1.2.0 → v1.9.3](https://github.com/pelletier/go-toml/compare/v1.2.0...v1.9.3)
- github.com/prometheus/common: [v0.26.0 → v0.28.0](https://github.com/prometheus/common/compare/v0.26.0...v0.28.0)
- github.com/spf13/afero: [v1.2.2 → v1.6.0](https://github.com/spf13/afero/compare/v1.2.2...v1.6.0)
- github.com/spf13/cast: [v1.3.0 → v1.3.1](https://github.com/spf13/cast/compare/v1.3.0...v1.3.1)
- github.com/spf13/cobra: [v1.1.3 → v1.2.1](https://github.com/spf13/cobra/compare/v1.1.3...v1.2.1)
- github.com/spf13/jwalterweatherman: [v1.0.0 → v1.1.0](https://github.com/spf13/jwalterweatherman/compare/v1.0.0...v1.1.0)
- github.com/spf13/viper: [v1.7.0 → v1.8.1](https://github.com/spf13/viper/compare/v1.7.0...v1.8.1)
- github.com/yuin/goldmark: [v1.3.5 → v1.4.0](https://github.com/yuin/goldmark/compare/v1.3.5...v1.4.0)
- go.opencensus.io: v0.22.3 → v0.23.0
- go.uber.org/zap: v1.17.0 → v1.19.0
- golang.org/x/crypto: 5ea612d → 32db794
- golang.org/x/net: 37e1c6a → e898025
- golang.org/x/oauth2: bf48bf1 → 2bc19b1
- golang.org/x/sys: 59db8d7 → f4d4317
- golang.org/x/term: 6a3ed07 → 6886f2d
- golang.org/x/text: v0.3.6 → v0.3.7
- golang.org/x/tools: v0.1.2 → d4cc65f
- google.golang.org/api: v0.20.0 → v0.46.0
- google.golang.org/appengine: v1.6.5 → v1.6.7
- google.golang.org/genproto: f16073e → fe13028
- google.golang.org/grpc: v1.38.0 → v1.40.0
- google.golang.org/protobuf: v1.26.0 → v1.27.1
- gopkg.in/ini.v1: v1.51.0 → v1.62.0
- honnef.co/go/tools: v0.0.1-2020.1.3 → v0.0.1-2020.1.4
- k8s.io/gengo: b6c5ce2 → 485abfe
- k8s.io/klog/v2: v2.9.0 → v2.30.0
- k8s.io/kube-openapi: 9528897 → e816edb
- k8s.io/system-validators: v1.5.0 → v1.6.0
- k8s.io/utils: 4b05e18 → cb0fa31
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.22 → v0.0.25
- sigs.k8s.io/kustomize/api: v0.8.11 → v0.10.1
- sigs.k8s.io/kustomize/cmd/config: v0.9.13 → v0.10.2
- sigs.k8s.io/kustomize/kustomize/v4: v4.2.0 → v4.4.1
- sigs.k8s.io/kustomize/kyaml: v0.11.0 → v0.13.0

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
- gotest.tools: v2.2.0+incompatible



# v1.23.0-rc.1


## Downloads for v1.23.0-rc.1

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes.tar.gz) | 4cf838ebcd3bb756cc604b194dd3b58716b41a34c35f636a7c23af4a501829c7ce23def2fae43547f86bff326bb9268a16b34058e83cada4626930b60a928d97
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-src.tar.gz) | 88af717a64f237de86f287c3715540fcaf6fd9e526a715832b395965ae27187319cc8955434f2511bcace46be455f56d22d569b083bf65f63bd0539fc6ce76a0

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-darwin-amd64.tar.gz) | 2aa8dbe6fe7926dd78083a243323dd0090f19acb22cc6ab6e7fb74ae3efd494f9927469b466a78eda1a210657935237ae235fbcf99dcc765908e590d8b5979ee
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-darwin-arm64.tar.gz) | 8aaaa11ecdefc349a64b12775656c9f1ed9bfdb72e3dac0f651c629f4f113dc701eb01d07e529c9633bbd5d0eca455a2bf0f654d52830b77917c28ec7bb84dd5
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-linux-386.tar.gz) | 03f482611ec7d12b30c03e382e53b1b00d7263882204d146a0fb74185d19eb90487ee87d898ec8b6c2ed388ac7b31455d79bafc3cf490efacea06825c1ff49c4
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-linux-amd64.tar.gz) | bac04a5d242ff32fbed1a7e756107afd68652ed489c405aee026132f8da57107ea9292ad776ec9f68c5b8a4bc5391a56d6a0ea538461d78778a4f09633fe3bba
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-linux-arm.tar.gz) | cc3ac01dd58f01e8d85b4c0ef70914331addc5b4abde05175d5194d3754c11b7f113a2e2ff79d15a17562565cd7572e643054b67457261563e9f316c3e20a473
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-linux-arm64.tar.gz) | 8fd0109f6ff07656dc0a265248c3fba81cf2faeb3a872dfb4423f14a92bff793e9ca8c330d83a1e957cdf9358287e325dd4f2ea7e52a0bf66ff1f2631b900a2c
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-linux-ppc64le.tar.gz) | 0610276ae835a9b151ad5ac9b192ef7d15b385279c9676ac4f2dfcb0600cb66dd4dcd39e07d65d637f8ae07f0f808f9eddf3312c45c6c435c2db0e0716cf3c29
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-linux-s390x.tar.gz) | bbba68df6895eb722ea7f8678e25e7be6e2d34a50e30c806445515e02b00e3d6982af8e2d54ee757c4fb661fea2f9fa61bd46ffe576897f0583598e3fb0be080
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-windows-386.tar.gz) | 1c41e825e83b0363967af94e759d6f8adb6f77f7f176c804922b089262df933d21528734dd8ebffd3d49fa7196a558a61d178d64f45444708688fe571dd2b4be
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-windows-amd64.tar.gz) | 2b39c66f4d69cdaeff9fa8f46331b2eece1714998df7a9300b6be85e9fec7ccc8cccdc02b3900ab7aa989816242b410234a1b9562b2dd51cff89ae19fd9be491
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-client-windows-arm64.tar.gz) | 77d00160870636703c9604613b6829c3727241616723de638653f5044373ed233608d8680357d16a37ee8fe88f0c158d93da644ddcddd27047f1d76da1a98a68

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-server-linux-amd64.tar.gz) | d54cbd2ed123ab30ad8662523a9f28af530bba0fe15b22738dbc2a5c3f6917c7ef4e0fcd03eaea36cee38420e968b91607e8d5022598e55030a21777448c57c0
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-server-linux-arm.tar.gz) | 835d94a3833e7b3b257f2174840e974f957d17e026dc2e1781f2d4590738afc947a27982cd06579a711c31dad5933beaa01c3629950dcb3e36f21b4f8e57aad8
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-server-linux-arm64.tar.gz) | 1fed95a3649d87390cb266918f5ed1b8fb00b4ef027cf004f6bffb3d05d1f6f84497c4ac11f34f3d8334d83cab929f46b62122e107ce6c0c6d628c01bb957fc7
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-server-linux-ppc64le.tar.gz) | cb563f6d5625eaa5db4f92b239508c0f30e2c7d9e6200651f23cf59f1962301c2113830e2718c515bcc05cce1a5c1b34d77281c3dc2253e4429e8425a852c238
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-server-linux-s390x.tar.gz) | a382396ddf43b6a4f5e95a95f58c2618cb63e5d5bbe3d8f8c74bdc1f23855eec14e4aa32e6ad6b2d2bfc0ecd2f9da602b2f410f17b0e680dc8f2e69d7f0f0127

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-node-linux-amd64.tar.gz) | fca9757fe2c67a01841810890fd62fb546c6a7ccb3f475d68a1960a054c0d4db44f71b2669e96d4c7bdb9ac727156448c9eeea64dbdb7442a04e91e5f88fdd76
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-node-linux-arm.tar.gz) | 30a29ba0a71d5cf29a359da78c0c42abc1c478d8cb4736b1e4dfb905770f537e170eba6b1079173867e6973e34b6d51f9d9f86c5573e1e6e7539de0bdd617c7b
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-node-linux-arm64.tar.gz) | 066a437d780b0c871e5635246f9434ba9e7652393546f8b3edba83a56e5a431b4e9aa443a82670d8c5f906bb630426f8a3d8c8b0d146426a20816f588182ea38
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-node-linux-ppc64le.tar.gz) | 8816dc1907a743cf8345acc41f5831d195b8531e313024060a5f322e5ba53061c2878f483e1d7253ed06bdb7edbbcf8784d2f5ef1d6c134e486ee8db9d1852b2
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-node-linux-s390x.tar.gz) | 671e9513d0aeace5352bf22a1522aa64e4f0b041110f6d11663d4fa3c97dc1c60a6197341dd000c8fd35e0cd51b279e748bea04b220dbecd9f589058632e59cc
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.1/kubernetes-node-windows-amd64.tar.gz) | cae457dde9c3bd813b97b2f57912f72cb08668b31700ceb356234ad78183fac45312e9a92e83e094ba7d991152bfe4375d098dfa6db6e3f2675630dd61441039

## Changelog since v1.23.0-rc.0

## Changes by Kind

### Bug or Regression

- Kube-apiserver: Server Side Apply merge order is reverted to match v1.22 behavior until http://issue.k8s.io/104641 is resolved. ([#106661](https://github.com/kubernetes/kubernetes/pull/106661), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- Reverts the CRI API version surfaced by dockershim to v1alpha2 ([#106808](https://github.com/kubernetes/kubernetes/pull/106808), [@saschagrunert](https://github.com/saschagrunert)) [SIG Network and Node]

## Dependencies

### Added
_Nothing has changed._

### Changed
- sigs.k8s.io/structured-merge-diff/v4: v4.2.0 → v4.1.2

### Removed
_Nothing has changed._



# v1.23.0-rc.0


## Downloads for v1.23.0-rc.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes.tar.gz) | ede62f7d1bde6a11e60b8ff119366c42902090c5b005ca73590856645c16ff12c904cdf45528cb5f48d4ece31db62f8a2c6a2fc4f10d29052c660036f5a47b5b
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-src.tar.gz) | 6103bde6ceeb7b6c40e6e7391731acc4228cf799ee8b7cf612baa8327212a183f16fd560f25b1d608e7f629c230310c585e2e1551436f9569a9d7d5a8c3dbb38

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-darwin-amd64.tar.gz) | 0266edfb98cf69c62466c87caa1028510cdb0600dfee9f25ba13b6936935011f6b90e8fe6b008a4c8d060012066adff45b28f01cf1c7f9e24293800576c5dc74
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-darwin-arm64.tar.gz) | 8bebf2537a53670a8487ccb43faec62b00439124c99d67ae88c5f1360bb863f03648c430be836580b56dc4526609fd53ef35cabf7a4622c346c87731d5e94575
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-linux-386.tar.gz) | 7cb542707711b5c4cf1402f07d2102b8633b4b75c43be4424f3e15da047badf484b458f6c3e27ecc82343f58f99437c4ab92a0a17f4976f806636d7b070c3396
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-linux-amd64.tar.gz) | eec960213ccb94b1cf1dc47aaf508eb12c1d04c13474b66960db4e0379d23a59aa484bba06cc04e6e7aba22a6e0eb8afd11a35799829dd540767eaedd66b37d1
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-linux-arm.tar.gz) | 7db536134da64c586058603283d752bc0bd7c2ea63b312513fff95d65bc24f978b24c0145446efea6f0b6b8f87f3c74c9a4dd581fa5a186ac630e6d58d30ee9d
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-linux-arm64.tar.gz) | 93a33630fe6bd89fb06f739f7a4184c151c4ac5a8230798b5c3a9137f553f59495e8cc2231f155d6ba51517f70ee095752872c067a12e7bba69ba2b4d9724ee9
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-linux-ppc64le.tar.gz) | afc9ffb6632b4c85837f87d6764e54a8111d4df3a23320294ac0b942dd089789d018faf91a4d5a22ddc4496d9204470267fba09b2a1d6d6f8ad74353df060675
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-linux-s390x.tar.gz) | d5c28f9e65d6a910cf6478342c3e1bd968b16820b2aae6d7ab51d1a646a3b4e46dd7bc2f255f5a02e516d36c90320f4c7b0f836192aae37f45da283fbd76dc88
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-windows-386.tar.gz) | cccb34fd97fb3f05aaf900569bd07772e4ba95f723f7ea71f191926fe05f01b4e608ab49d876b65405051c504043c74b725d09c92321e17a35587703269a37a8
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-windows-amd64.tar.gz) | 61298763df834a36a1f10c0a45cc7c0b520ad38c2f013fd39e0ab11cb07b7f416b9aa695c4289cf21baf9281c3a9ec4e99e93b4b82b368658b91fa19f6486c69
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-client-windows-arm64.tar.gz) | 99d52ce6c6f620464239e4610711daa8d79e0551bab587d9d2342fe315a07fb94b19ffbfdc8f4a769ae51c848f2896517ac3abd256504492bc24389878749775

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-server-linux-amd64.tar.gz) | fff4bdfce528a16abce7d075570bad7e5fd3b64baa5bd154595b44d25945379e4ea6fb56869681cd2222ad55965214d7e96ea1d32ddd519629a1b943dfbceb2b
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-server-linux-arm.tar.gz) | dc777f74f6d6eef8d56d379cad36e535566993df3abf0be5e00cf22790f01f4336cd757b395aa724b5743db2ad74ce966746755d4970e04a599861f59ea8f12c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-server-linux-arm64.tar.gz) | a4d935e6816e6bd14e037819949644626813885ef308c7e5ab0a680f71b155cd164c46b963dcdbdcd91cadbf1d0870c66934bb62c9f95698699d1ae3dcc25cb5
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-server-linux-ppc64le.tar.gz) | 9a69761c04556e246e046e18bd4b875e813aef1a1e01931ad0aef61ad11d741d4a1f416949dc9f5a8b7103823f6061557d075a35b4766a8e1192872a3fedb637
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-server-linux-s390x.tar.gz) | 6f64559358d05d659faf79476524101420b6b4e83e27bba11c407624723926adfd14045792a8a874100527d21cf334b56293a94ac1c3e8916b7ec3e9926b52e4

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-node-linux-amd64.tar.gz) | ed3972e5bb9550d0999b4f4b9da315607bfdb349fae9395e23bb36e72c6cc30aa6df42084b0faa0f5f890b983ed7132357d49d755cde081fe4c53bf78a935f83
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-node-linux-arm.tar.gz) | c87b42358cc75f362eb5a1b52780ee62fa7c18a8760cf7ee744e1910558ffbe5869fea75e49bf8822795c5255ddd935a9364aa067bffeca500b8fb61d70402c5
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-node-linux-arm64.tar.gz) | 34cf8aea703f279559765fdf5c0a079e4679e407134c666082a4ee56d1352c1b66291c847a983ee547a64209143a83a15d89f73d4f2fbb3473dfefb8f9aa630e
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-node-linux-ppc64le.tar.gz) | 86e8bb17b715aea6df3c4ddfd68e4423414ac9a9b86cd38e6272a2226849a456d86496d6154b468b97c22164dc22f80f1dd456a2011ca22f1b782791cf4d3c84
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-node-linux-s390x.tar.gz) | 61723328d454ceade01b1ad5e71bf96caae9badd58b02e877833eafae8465cbc9d09cfd1f414749c4040419bbc594a9256f02a44ce7e762a9753ae50511e57de
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-rc.0/kubernetes-node-windows-amd64.tar.gz) | 3c22248012c2e301209832ded5ecd20d87eb4350a09592763c7de6b46c821b158abbad816c76e012e692ff21ba5da9fb0e861b914166f339f9a0d676c3d4b60a

## Changelog since v1.23.0-beta.0

## Changes by Kind

### API Change

- Add gRPC probe to Pod.Spec.Container.{Liveness,Readiness,Startup}Probe (#106463, @SergeyKanzhelev) [SIG API Machinery, Apps, CLI, Node and Testing]
- Adds a feature gate StatefulSetAutoDeletePVC, which allows PVCs automatically created for StatefulSet pods to be automatically deleted. (#99728, @mattcary) [SIG API Machinery, Apps, Auth and Testing]
- Performs strict server side schema validation requests via the `fieldValidation=[Strict,Warn,Ignore]` query parameter. (#105916, @kevindelgado) [SIG API Machinery, Apps, Auth, Cloud Provider and Testing]
- Support pod priority based node graceful shutdown (#102915, @wzshiming) [SIG Node and Testing]

### Feature

- CRI v1 is now the project default. If a container runtime does not support the v1 API, Kubernetes will fall back to the v1alpha2 implementation. (#106501, @ehashman) [SIG Network, Node and Testing]

### Bug or Regression

- Kube-Proxy now correctly filters out unready endpoints for Services with Topology Aware Hints enabled. (#106507, @robscott) [SIG Network]
- Kubelet: the printing of flags at the start of kubelet now uses the final logging configuration (#106520, @pohly) [SIG Node]
- Topology Aware Hints now ignores unready endpoints when assigning hints. (#106510, @robscott) [SIG Apps and Network]

## Dependencies

### Added
_Nothing has changed._

### Changed
_Nothing has changed._

### Removed
_Nothing has changed._



# v1.23.0-beta.0


## Downloads for v1.23.0-beta.0

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes.tar.gz) | 048cc297840fd70dc571863bbed9da8176a479ca6b8ff17c9a2cc1b1dbf286377d85eb7fccc5d85e1d652658c393ea1eab7ab518631510e1e7462ea638a56b2b
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-src.tar.gz) | 1d3f6f5bb54b61312934169845417dffc428bed0f51342dc2b0eebf7f16899843b0f66f9fb2dcdb2a6e9f25bbdc930ea9adac552b0b011e656151c8cae2f4f71

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-darwin-amd64.tar.gz) | e22ce7199acf369eacf8422c8ee417041289e927bfc03c238f45faec75c2dabd7f8201c77ed39f20ac311d1ba289766825b7b2f738cfc59b5652a20b98117180
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-darwin-arm64.tar.gz) | 22fa13ca86eb5837db3844b6b7fd134c3ffa3ba5a008635bfa83613a100fa48b3e2331cdf5d368cb267c3cd27e3947fe08ac2540342f1b221192e972695a2cd6
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-linux-386.tar.gz) | 8e239ce934d121b21b534a6d521ca02bf1c6709831e181d103c8d86cdab01b296546be25902162b1060876744f3b579de018b7c2d198e5d5efdd9c849b3ba7ef
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-linux-amd64.tar.gz) | e9355264e3ca91da833fe3c8c1dcc55c287a9b813aad91f26b09e6a75f48be57d12cb235c5f9c6fe2a0aceee09e2b5da84568d81d8002066c8e77d848a03f112
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-linux-arm.tar.gz) | 80e93b6c8cce8221f9a5aba8018fcd95b7ec57728a202fdd158b8df86a733e32d6bb60d8b7ea78da9556058074e9bb88c072b4207a43a4fd2f256cce2593a8df
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-linux-arm64.tar.gz) | 769a1aa41988bbf11a11ef40f42c76740fcbe7fe1fd5d6da948729e1a62bf9c4f28101f47fa9ccd12de50a378b3654e1e4c2d50afad59182c03b8d1e972341e7
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-linux-ppc64le.tar.gz) | 4a9346caef2714f03e65dc3e5e46ade1b311b91ef184b8a47466583e834f44dcdb21c3800793e87c20064b25c3eac2c34637ff6817f1752d52425cdfd5a912fb
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-linux-s390x.tar.gz) | f2129ea05e581a38bdc2771cfdd92ad990620fabf9655f7343c56541a544aa4c6c1e1a2e91a338d06dd0064f35fb5e3027259c317a0909badcbadc9e418c6ced
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-windows-386.tar.gz) | 2dc9459b02f4ed564a7d0e2062e3590c5240debc6a64449d1c714382ded197d5fcf99feecb80ba6483d265ab34126958737cd692783e675b39159be94729c018
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-windows-amd64.tar.gz) | e58cb2f87f619d34afbb2c2c0f2bab484970406216698b79129637cb27c5508b2ca4bd2a3a91847868631bd72947887317692a73fec0f8d67c26aa59868c9d8f
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-client-windows-arm64.tar.gz) | 515bd2e3c95afe613db998ed42ea5456771c488e0963c9fe0328816a6baba09ea4e915d22538e05d478556d17f1678d6a96b75cae25ba742be73da23d04f72ff

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-server-linux-amd64.tar.gz) | adc6c0e5c07c3e1d24ac4399ea725da5d72a043feaea0063f26188e469b4b8cf537df245015631f1efce9d5e457724858327da3c7c9763f6ca4538aaf77a5e67
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-server-linux-arm.tar.gz) | e6e673cb9baecc56ae03d716569769391cd6f8d38d85810f0199e71b20a4d4c3c92efe7b31a67af463fb01029d94cbcb0c6fe7a0918123055f3fa8f373e76c49
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-server-linux-arm64.tar.gz) | f91dc6e948b702784909ca0c4b8758ad9dbfbcd202ec4e329666b07d42488df00ad64de6a68405668ed881e62e0515271c8168e8316519cd95802239abde4951
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-server-linux-ppc64le.tar.gz) | fbbf3daff8caa89f8249122ba19d67a0d9298fb47d327c0bebd7a54adad4fe6e809164d8bf8e563c79b1f9c8b646f29d18789ec938cbc5746e30649b392c7121
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-server-linux-s390x.tar.gz) | a4ccda542f1b86667e6bf29afd091a2ce6f3a30165ff8b918585fc7794be26d00bd846acaa5b805b270a60df69fbe9827bab6ee472129996e28052bbbe1b0593

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-node-linux-amd64.tar.gz) | 4d7dd2e50fe65fd1140c51deeb90d8d9f89bbba59502becf626757e2e9eb59fb781bbf3ecb899f1b8e391746329c5c017177287004195387151799e73887f05b
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-node-linux-arm.tar.gz) | d38cd4a06b983a7253d99a6d927c40cbacc636bd73d33172ee03cda502f806638d3cc6f096bc13a55a2faf11ab3e85d77dfd20559e2c880cf54f45ba0875c75c
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-node-linux-arm64.tar.gz) | fa1fa35f30ca589e031485affd2a1016ba5ca0efdf64b35d49c7738342acb55c40733e53fb3b477734bab68d97b00f9adcfb5954ab365169d8f00ac804cc60fb
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-node-linux-ppc64le.tar.gz) | 412b3a133a7711e32455e49d1aac4ce9ee0e44df89afca40dfa8ac52a8aa98649bd4dd7eff85addd8a525bb16b65966dbde1df0c62a994213b4cfa1a7a3b8128
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-node-linux-s390x.tar.gz) | 7e0e217893665a56406b6f1404d616da8578396890b04474fed12ea6b48f5fbf52432efd43c13f66a643284fd54c0fd3441940c777eb1cd0796443fd72d69b6f
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-beta.0/kubernetes-node-windows-amd64.tar.gz) | 768dfe871a028ff7d972d9b59935c1ebdcc8ea0ccf990ee84060ef3bb995ddecb48a49d9fb2ff12dc44ed404d6d9362ee78af3492a4206bb23eb8a0ac8d63ca2

## Changelog since v1.23.0-alpha.4

## Urgent Upgrade Notes

### (No, really, you MUST read this before you upgrade)

 - Log messages in JSON format are written to stderr by default now (same as text format) instead of stdout. Users who expected JSON output on stdout must now capture stderr instead or in addition to stdout. (#106146, @pohly) [SIG API Machinery, Architecture, Cluster Lifecycle and Instrumentation]
  - [kube-log-runner](https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/component-base/logs/kube-log-runner) is included in release tar balls. It can be used to replace the deprecated `--log-file` parameter. (#106123, @pohly) [SIG API Machinery, Architecture, Cloud Provider, Cluster Lifecycle and Instrumentation]
 
## Changes by Kind

### Deprecation

- Kubeadm: add a new output/v1alpha2 API that is identical to the output/v1alpha1, but attempts to resolve some internal dependencies with the kubeadm/v1beta2 API. The output/v1alpha1 API is now deprecated and will be removed in a future release. (#105295, @neolit123) [SIG Cluster Lifecycle]
- Kubeadm: add the kubeadm specific, Alpha (disabled by default) feature gate UnversionedKubeletConfigMap. When this feature is enabled kubeadm will start using a new naming format for the ConfigMap where it stores the KubeletConfiguration structure. The old format included the Kubernetes version - "kube-system/kubelet-config-1.22", while the new format does not - "kube-system/kubelet-config". A similar formatting change is done for the related RBAC rules. The old format is now DEPRECATED and will be removed after the feature graduates to GA. When writing the ConfigMap kubeadm (init, upgrade apply) will respect the value of UnversionedKubeletConfigMap, while when reading it (join, reset, upgrade), it would attempt to use new format first and fallback to the legacy format if needed. (#105741, @neolit123) [SIG Cluster Lifecycle and Testing]

### API Change

- A new field `omitManagedFields` has been added to both `audit.Policy` and `audit.PolicyRule` 
  so cluster operators can opt in to omit managed fields of the request and response bodies from 
  being written to the API audit log. (#94986, @tkashem) [SIG API Machinery, Auth, Cloud Provider and Testing]
- Create HPA v2 from v2beta2 with some fields changed. (#102534, @wangyysde) [SIG API Machinery, Apps, Auth, Autoscaling and Testing]
- Fix kube-proxy regression on UDP services because the logic to detect stale connections was not considering if the endpoint was ready. (#106163, @aojea) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Contributor Experience, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage, Testing and Windows]
- Implement support for recovering from volume expansion failures (#106154, @gnufied) [SIG API Machinery, Apps and Storage]
- In kubelet, log verbosity and flush frequency can also be configured via the configuration file and not just via command line flags. In other commands (kube-apiserver, kube-controller-manager), the flags are listed in the "Logs flags" group and not under "Global" or "Misc". The type for `-vmodule` was made a bit more descriptive (`pattern=N,...` instead of `moduleSpec`). (#106090, @pohly) [SIG API Machinery, Architecture, CLI, Cluster Lifecycle, Instrumentation, Node and Scheduling]
- IngressClass.Spec.Parameters.Namespace field is now GA. (#104636, @hbagdi) [SIG Network and Testing]
- KubeSchedulerConfiguration provides a new field `MultiPoint` which will register a plugin for all valid extension points (#105611, @damemi) [SIG Scheduling and Testing]
- Kubelet should reject pods whose OS doesn't match the node's OS label. (#105292, @ravisantoshgudimetla) [SIG Apps and Node]
- The CSIVolumeFSGroupPolicy feature has moved from beta to GA. (#105940, @dobsonj) [SIG Storage]
- The Kubelet's `--register-with-taints` option is now available via the Kubelet config file field registerWithTaints (#105437, @cmssczy) [SIG Node and Scalability]
- Validation rules for Custom Resource Definitions can be written in the [CEL expression language](https://github.com/google/cel-spec) using the `x-kubernetes-validations` extension in OpenAPIv3 schemas (alpha). This is gated by the alpha "CustomResourceValidationExpressions" feature gate. (#106051, @jpbetz) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Storage and Testing]

### Feature

- (beta feature) If the CSI driver supports the NodeServiceCapability `VOLUME_MOUNT_GROUP` and the `DelegateFSGroupToCSIDriver` feature gate is enabled, kubelet will delegate applying FSGroup to the driver by passing it to NodeStageVolume and NodePublishVolume, regardless of what other FSGroup policies are set. (#106330, @verult) [SIG Storage]
- /openapi/v3 endpoint will be populated with OpenAPI v3 if the feature flag is enabled (#105945, @Jefftree) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Node, Storage and Testing]
- Add support for PodAndContainerStatsFromCRI featuregate, which allows a user to specify their pod stats must also come from the CRI, not cAdvisor. (#103095, @haircommander) [SIG Node]
- Add support for Portworx plugin to csi-translation-lib. Alpha release
  
  Portworx CSI driver is required to enable migration.
  This PR adds support of the `CSIMigrationPortworx` feature gate, which can be enabled by:
  
  1. Adding the feature flag to the kube-controller-manager `--feature-gates=CSIMigrationPortworx=true` 
  2. Adding the feature flag to the kubelet config:
  
  featureGates:
    CSIMigrationPortworx: true (#103447, @trierra) [SIG API Machinery, Apps, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage, Testing and Windows]
- Added ability for kubectl wait to wait on arbitary JSON path (#105776, @lauchokyip) [SIG CLI]
- Added the ability to specify whether to use an RFC7396 JSON Merge Patch, an RFC6902 JSON Patch, or a Strategic Merge Patch to perform an override of the resources created by kubectl run and kubectl expose. (#105140, @brianpursley) [SIG CLI]
- Adding option for kubectl cp to resume on network errors until completion, requires tar in addition to tail inside the container image (#104792, @matthyx) [SIG CLI]
- Adds --as-uid flag to kubectl to allow uid impersonation in the same way as user and group impersonation. (#105794, @margocrawf) [SIG API Machinery, Auth, CLI and Testing]
- Allows users to prevent garbage collection on pinned images (#103299, @wgahnagl) [SIG Node]
- CSIMigrationGCE feature flag is turned ON by default (#104722, @leiyiz) [SIG Apps, Cloud Provider, Node, Storage and Testing]
- Changed feature CSIMigrationAWS to on by default. This feature requires the AWS EBS CSI driver to be installed. (#106098, @wongma7) [SIG Storage]
- Ensures that volume is deleted from the storage backend when the user tries to delete the PV object manually and the PV ReclaimPolicy is Delete. (#105773, @deepakkinni) [SIG Apps and Storage]
- Graduating `controller_admission_duration_seconds`, `step_admission_duration_seconds`, `webhook_admission_duration_seconds`, `apiserver_current_inflight_requests` and `apiserver_response_sizes` metrics to stable. (#106122, @rezakrimi) [SIG API Machinery, Instrumentation and Testing]
- Graduating `pending_pods`, `preemption_attempts_total`, `preemption_victims` and `schedule_attempts_total` metrics to stable. Also `e2e_scheduling_duration_seconds` is renamed to `scheduling_attempt_duration_seconds` and the latter is graduated to stable. (#105941, @rezakrimi) [SIG Instrumentation, Scheduling and Testing]
- Integration testing now takes periodic Prometheus scrapes from the etcd server.
  There is a new script ,`hack/run-prometheus-on-etcd-scrapes.sh`, that runs a containerized Prometheus server against an archive of such scrapes. (#106190, @MikeSpreitzer) [SIG API Machinery and Testing]
- Kube-apiserver: when merging lists, Server Side Apply now prefers the order of the submitted request instead of the existing persisted object (#105983, @jiahuif) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Storage and Testing]
- Kubectl describe namespace now shows Conditions (#106219, @dlipovetsky) [SIG CLI]
- Kubelet should reconcile `kubernetes.io/os` and `kubernetes.io/arch` labels on the node object. The side-effect of this is kubelet would deny admission to pod which has nodeSelector with label `kubernetes.io/os` or `kubernetes.io/arch` which doesn't match the underlying OS or arch on the host OS. 
  - The label reconciliation happens as part of periodic status update which can be configured via flag `--node-status-update-frequency` (#104613, @ravisantoshgudimetla) [SIG Node, Testing and Windows]
- Kubernetes is now built with Golang 1.17.3 (#106209, @cpanato) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- Move ConfigurableFSGroupPolicy to GA
  Rename metric volume_fsgroup_recursive_apply to volume_apply_access_control (#105885, @gnufied) [SIG Instrumentation and Storage]
- Moving WindowsHostProcessContainers feature to beta (#106058, @marosset) [SIG Windows]
- The DownwardAPIHugePages feature is now enabled by default. (#106271, @mysunshine92) [SIG Node]
- The PodSecurity admission plugin has graduated to beta and is enabled by default. The admission configuration version has been promoted to pod-security.admission.config.k8s.io/v1beta1. See https://kubernetes.io/docs/concepts/security/pod-security-admission/ for usage guidelines. (#106089, @liggitt) [SIG Auth and Testing]
- This PR adds the following metrics for API Priority and Fairness.
  - **apiserver_flowcontrol_priority_level_seat_count_samples**: histograms of seats occupied by executing requests (both regular and final-delay phases included), broken down by priority_level; the observations are taken once per millisecond.
  - **apiserver_flowcontrol_priority_level_seat_count_watermarks**: histograms of high and low watermarks of number of seats occupied by executing requests (both regular and final-delay phases included), broken down by priority_level.
  - **apiserver_flowcontrol_watch_count_samples**: histograms of number of watches relevant to a given mutating request, broken down by that request's priority_level and flow_schema. (#105873, @MikeSpreitzer) [SIG API Machinery, Instrumentation and Testing]
- Topology Aware Hints have graduated to beta. (#106433, @robscott) [SIG Network]
- Update the system-validators library to v1.6.0 (#106323, @neolit123) [SIG Cluster Lifecycle and Node]
- Upgrade etcd to 3.5.1 (#105706, @uthark) [SIG Cloud Provider, Cluster Lifecycle and Testing]
- When using `RequestedToCapacityRatio` ScoringStrategy, empty shape will cause error. (#106169, @kerthcet) [SIG Scheduling]

### Documentation

- Graduating `pod_scheduling_duration_seconds`, `pod_scheduling_attempts`, `framework_extension_point_duration_seconds`, `plugin_execution_duration_seconds` and `queue_incoming_pods_total` metrics to stable. (#106266, @ahg-g) [SIG Instrumentation, Scheduling and Testing]
- Users should not rely on unsupported CRON_TZ variable when specifying schedule, both the API server and cronjob controller will emit warnings pointing to https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/ containing explanation (#106455, @soltysh) [SIG Apps]

### Bug or Regression

- (PodSecurity admission) errors validating workload resources (deployment, replicaset, etc.) no longer block admission. (#106017, @tallclair) [SIG Auth]
- Add support for Windows Network stats in Containerd (#105744, @jsturtevant) [SIG Node, Testing and Windows]
- Added show-capacity option to `kubectl top node` to show `Capacity` resource usage (#102917, @bysnupy) [SIG CLI]
- Do not unmount and mount subpath bind mounts during container creation unless bind mount changes (#105512, @gnufied) [SIG Storage]
- Don't use a custom dialer for the kubelet if is not rotating certificates, so we can reuse TCP connections and have only one between the apiserver and the kubelet.
  If users experiment problems with stale connections  using HTTP1.1, they can force the previous behavior of the kubelet by setting the environment variable DISABLE_HTTP2. (#104844, @aojea) [SIG API Machinery, Auth and Node]
- EndpointSlice Mirroring controller now cleans up managed EndpointSlices when a Service selector is added (#105997, @robscott) [SIG Apps, Network and Testing]
- Enhanced event messages for pod failed for exec probe timeout (#106201, @yxxhero) [SIG Node]
- Ensure Pods are removed from the scheduler cache when the scheduler misses deletion events due to transient errors (#106102, @alculquicondor) [SIG Scheduling]
- Fix a panic in kubectl when creating secrets with an improper output type (#106317, @lauchokyip) [SIG CLI]
- Fixed a bug which could cause webhooks to have an incorrect copy of the old object after an Apply or Update (#106195, @alexzielenski) [SIG API Machinery]
- Fixed applying of SELinux labels to CSI volumes on very busy systems (with "error checking for SELinux support: could not get consistent content of /proc/self/mountinfo after 3 attempts") (#105934, @jsafrane) [SIG Storage]
- Fixed bug where using kubectl patch with $deleteFromPrimitiveList on a nonexistent or empty list would add the item to the list (#105421, @brianpursley) [SIG API Machinery]
- Fixed the issue where logging output of kube-scheduler configuration files included line breaks and escape characters. The output also attempted to output the configuration file in one section without showing the user a more readable format. (#106228, @sanchayanghosh) [SIG Scheduling]
- Kube-up now includes CoreDNS version v1.8.6 (#106091, @rajansandeep) [SIG Cloud Provider]
- Kubeadm: fix a bug on Windows worker nodes, where the downloaded KubeletConfiguration from the cluster can contain Linux paths that do not work on Windows and can trip the kubelet binary. (#105992, @hwdef) [SIG Cluster Lifecycle and Windows]
- Kubectl port-forward service will now properly exit when the attached pod dies (#103526, @brianpursley) [SIG API Machinery]
- Kubelet: fixes a file descriptor leak in log rotation (#106382, @rphillips) [SIG Node]
- Pod SecurityContext sysctls name parameter for update requests where the existing object's sysctl contains slashes and kubelet sysctl whitelist support contains slashes. (#102393, @mengjiao-liu) [SIG Apps, Auth, Node, Storage and Testing]
- Pod will not start when Init container was OOM killed. (#104650, @yxxhero) [SIG Node]
- Reduce the number of calls to docker for stats via dockershim. For Windows this reduces the latency when calling docker, for Linux this saves cpu cycles. (#104287, @jsturtevant) [SIG Node and Windows]
- Respect grace period when updating static pods. (#104743, @gjkim42) [SIG Node and Testing]
- The kube-proxy sync_proxy_rules_iptables_total metric now gives
  the correct number of rules, rather than being off by one.
  
  Fixed multiple iptables proxy regressions introduced in 1.22:
  
    - When using Services with SessionAffinity, client affinity for an
      endpoint now gets broken when that endpoint becomes non-ready
      (rather than continuing until the endpoint is fully deleted).
  
    - Traffic to a service IP now starts getting rejected (as opposed to
      merely dropped) as soon as there are no longer any *usable*
      endpoints, rather than waiting until all of the terminating
      endpoints have terminated even when those terminating endpoints
      were not being used.
  
    - Chains for endpoints that won't be used are no longer output to
      iptables, saving a bit of memory/time/cpu. (#106030, @danwinship) [SIG Network]
- Upgrades functionality of `kubectl kustomize` as described at
  https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv4.4.1 (#106389, @natasha41575) [SIG CLI]

### Other (Cleanup or Flake)

- Changed buckets in apiserver_request_duration_seconds metric from [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60] to [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60] (#106306, @pawbana) [SIG API Machinery, Instrumentation and Testing]
- Kubectl: deprecated command line flags (like several of the klog flags) now have a `DEPRECATED: <explanation>` comment. (#106172, @pohly) [SIG CLI]
- Kubemark is now built as a portable, static binary. (#106150, @pohly) [SIG Scalability and Testing]
- Migrated `pkg/scheduler/framework/plugins/volumebinding/assume_cache.go` to structured logging. (#105904, @mengjiao-liu) [SIG Instrumentation, Scheduling and Storage]
- Migrated `pkg/scheduler/framework/preemption/preemption.go`, `pkg/scheduler/framework/plugins/examples/stateful/stateful.go`, and `pkg/scheduler/framework/plugins/noderesources/resource_allocation.go` to structured logging (#105967, @shivanshu1333) [SIG Instrumentation, Node and Scheduling]
- Migrated scheduler file `cache.go` to structured logging (#105969, @shivanshu1333) [SIG Instrumentation and Scheduling]
- Migrated scheduler files `comparer.go`, `dumper.go`, `node_tree.go` to structured logging (#105968, @shivanshu1333) [SIG Instrumentation and Scheduling]
- Remove deprecated and not supported old cronjob controller. (#106126, @soltysh) [SIG Apps]
- Remove ignore error flag for drain, and set this feature as default (#105571, @yuzhiquan) [SIG CLI]
- The kube-proxy image contains /go-runner as a replacement for deprecated klog flags. (#106301, @pohly) [SIG Testing]

## Dependencies

### Added
- github.com/OneOfOne/xxhash: [v1.2.2](https://github.com/OneOfOne/xxhash/tree/v1.2.2)
- github.com/antlr/antlr4/runtime/Go/antlr: [b48c857](https://github.com/antlr/antlr4/runtime/Go/antlr/tree/b48c857)
- github.com/cespare/xxhash: [v1.1.0](https://github.com/cespare/xxhash/tree/v1.1.0)
- github.com/cncf/xds/go: [fbca930](https://github.com/cncf/xds/go/tree/fbca930)
- github.com/getkin/kin-openapi: [v0.76.0](https://github.com/getkin/kin-openapi/tree/v0.76.0)
- github.com/google/cel-go: [v0.9.0](https://github.com/google/cel-go/tree/v0.9.0)
- github.com/google/cel-spec: [v0.6.0](https://github.com/google/cel-spec/tree/v0.6.0)
- github.com/spaolacci/murmur3: [f09979e](https://github.com/spaolacci/murmur3/tree/f09979e)

### Changed
- github.com/containerd/containerd: [v1.4.9 → v1.4.11](https://github.com/containerd/containerd/compare/v1.4.9...v1.4.11)
- github.com/coredns/corefile-migration: [v1.0.12 → v1.0.14](https://github.com/coredns/corefile-migration/compare/v1.0.12...v1.0.14)
- github.com/docker/docker: [v20.10.2+incompatible → v20.10.7+incompatible](https://github.com/docker/docker/compare/v20.10.2...v20.10.7)
- github.com/envoyproxy/go-control-plane: [668b12f → 63b5d3c](https://github.com/envoyproxy/go-control-plane/compare/668b12f...63b5d3c)
- github.com/golang/glog: [23def4e → v1.0.0](https://github.com/golang/glog/compare/23def4e...v1.0.0)
- github.com/google/cadvisor: [v0.39.2 → v0.43.0](https://github.com/google/cadvisor/compare/v0.39.2...v0.43.0)
- golang.org/x/net: 60bc85c → e898025
- golang.org/x/sys: 41cdb87 → f4d4317
- golang.org/x/text: v0.3.6 → v0.3.7
- google.golang.org/genproto: f16073e → fe13028
- google.golang.org/grpc: v1.38.0 → v1.40.0
- google.golang.org/protobuf: v1.26.0 → v1.27.1
- k8s.io/kube-openapi: 7fbd8d5 → e816edb
- k8s.io/system-validators: v1.5.0 → v1.6.0
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.23 → v0.0.25
- sigs.k8s.io/kustomize/api: v0.8.11 → v0.10.1
- sigs.k8s.io/kustomize/cmd/config: v0.9.13 → v0.10.2
- sigs.k8s.io/kustomize/kustomize/v4: v4.2.0 → v4.4.1
- sigs.k8s.io/kustomize/kyaml: v0.11.0 → v0.13.0
- sigs.k8s.io/structured-merge-diff/v4: v4.1.2 → v4.2.0

### Removed
_Nothing has changed._



# v1.23.0-alpha.4


## Downloads for v1.23.0-alpha.4

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes.tar.gz) | aeb10a3fbb89694c52d47203cc958d3543b21426938a9664348163aacd41e20ea7670617a28d8ce6d8d51492980facd5fab062e8ad664dafd7b8dbff1c2bb54f
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-src.tar.gz) | b7a8999335ce15b68360478b22af4daaed10e9db50d597e077d731de194208355d1b2134f5635331d9049dc638d05f1f792d52c5890e521f0af3dc2f3e64fbb8

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-darwin-amd64.tar.gz) | 5654879ac03f4c7193a8df49cfd4b7253add031c197f50bada40942738bf5720d1c06e31a1d1a7bd1b1a540aa46897e4b34ad8a7e087bd206a7b69b9ffaf5edb
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-darwin-arm64.tar.gz) | 5dce9fee32436c971ef17595f88f3c74f5644ab3af0e3f854a79fb42f3c8d6d8f507fbb0d7b5bcba52ddf1a49ada2559c477278037cf2dadccba72f0398a1093
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-linux-386.tar.gz) | e9eb7dab22801c043da2833fde89d2cd721b9dd622df0ff42b25a6742cfab5cff8bfe3ebbda6cc584cf92db3940b95e25aff935863ed999374ee8923ca0b1215
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-linux-amd64.tar.gz) | 570eeaed029bb05235c58138a777cfd6a4b17d4d91aba346b1fc9a0e573781947599d31a8997e889165db561a18a7ab4d613c2b40a8b2dc0d0225f2411b0fd73
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-linux-arm.tar.gz) | 298923762745cc064a4489aa01d55f57076b84538aef3a6a3554b60257d9959b4eebbb8aeeecdaf14246fa4f1c17750e1b69c63d4940ae71f87010692e41675c
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-linux-arm64.tar.gz) | 498527f1cf2d16af576a6b6d27b5ddbb876e24bd85e34e2c91cf39ef467d366b2059e580fdcccb91e0b61a5f52795273b77ed94a1073b5c0bd574b8661afbe0e
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-linux-ppc64le.tar.gz) | 2632b0fb69565819ef1b6797a834e65f96629df4fd8bec01fce7370672a39afa181854d6ab44afc1c4a6b8143158cf170f5a8e61b75a48071ade2d5ab89d1b2c
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-linux-s390x.tar.gz) | b793a5a8fce9109343ada86f29cf356c6973cd80d81ca47af5c7e4fa11ffccc273f77aba52b1db42ee12abb94ee23677c21910f57c9385646e35742a1c60e17e
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-windows-386.tar.gz) | b92e34ee58e1247c1c444134dd9fa78033d0fda1f51509b43016543596cb211128f8aff730d9a3a9118dfeba139186db2a5dd45455427c7521776e63ee77218c
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-windows-amd64.tar.gz) | 0b5ea6a2de0ff6f71647f428fdbee67c7eb2b918d725cf236ce60daa02e94bd998d15ea0ebb20c4106453e220d11d31506161d5dee3cde6c616dfb5efd11c25e
[kubernetes-client-windows-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-client-windows-arm64.tar.gz) | a4e570be453d1df779bb85c62efb41e98209bb93b57b7655a94a737d552c90f9d3061df9088204c7787344dc6a3eb3f843c58651394c0436d2c90b55e499bfab

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-server-linux-amd64.tar.gz) | cf215ee7372edd7d5dbf07faee8ccf83de477c8cd431c0fac58357bb8e027349d8edf87364e7db5cec0936f991388f7b183e81e5f92cb6cdf6303efd8cd65d83
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-server-linux-arm.tar.gz) | d6281a6727fcdab956170dca7563fc5099ef79b06c96b2f6bc87fcd0b74f1dea0e14aca344cb41b5b6811919cf4a6d6f60cb08b7fb7034690fd0c4ead82e55ca
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-server-linux-arm64.tar.gz) | 4ec30cdfd8128ca405201c0c40750e10bac016e1e53a7662265328564b09e4feb831a259125bcdd64169d221145cbb166a463216e884dd76f4bb9a72a00e64e0
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-server-linux-ppc64le.tar.gz) | 42b31174a95d0999c78750a1d2c866918c91d11d6406df4e984913f64806708add35c27c0daf255b5d28e98eb815355d1913911f921d34e618dea4d2ebf91949
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-server-linux-s390x.tar.gz) | c4e2b38681c0858d560adc8a330f27e95a035cb0e426c6ff332dcd435cefe88441ea866badab5514c4055191324c48aac108d5d6934a9fd4697da179168b6632

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-node-linux-amd64.tar.gz) | 708b40c9c0d2cfcb6f9874aa3f1b5a27796cbe2bfe7a2345f381e0d9062df8a6769b2bf29b8b641929ef1f5952897c5739e876e8315eb51cced460b13994c247
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-node-linux-arm.tar.gz) | f89448106af23d6658b9c2e7b43240fb82051d2f89a302ee61fc1cc78e593535993ba12f412c3df907f415e55c38f1783ce9141075198bb9f197b6fa26328d49
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-node-linux-arm64.tar.gz) | bcd8d9fbb244048a3ef3f79f1d4e8f2645bbd69caf353e67ee5c5a4ffd4443da420e5984422933cd4c622c58017a942e20af076f26bfa22f5f38f73a831370ca
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-node-linux-ppc64le.tar.gz) | c0724d053c601d4e80ea19957bd32005aeba0cf8f5e03e8e36412aed0777e860ae680302eef632c8e7d4ef1a8e789e48dc58489ad1a7bf7fd20cb0f755e797af
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-node-linux-s390x.tar.gz) | 0245f592b92d79ccd102961e5b23a9f5b275829e627254fe8ce5f0a7df53ec2c4a9436942686b9d31b696635ab88131cf92e1002869369fa1cf6f080f8073b5f
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.4/kubernetes-node-windows-amd64.tar.gz) | 2a5c6c79ea65f47a42d25b236709a00eafb793e5d87b5f56516da16b85b06e03020679f7cabfb7dc4bc252ff57da0afa52e357915cb1d3801dc3d5c32f096edf

## Changelog since v1.23.0-alpha.3

## Changes by Kind

### Deprecation

- A deprecation notice has been added when using the kube-proxy Userspace proxier, which will be removed in v1.25. (#103860) (#104631, @perithompson) [SIG Network]
- Feature-gate VolumeSubpath has been deprecated and cannot be disabled. It will be completely removed in 1.25 (#105474, @mauriciopoppe) [SIG Storage]
- Kubeadm: remove the deprecated / NO-OP phase "update-cluster-status" in "kubeadm reset" (#105888, @neolit123) [SIG Cluster Lifecycle]
- Removed kubectl --dry-run empty default value and boolean values. kubectl --dry-run usage must be specified with --dry-run=(server|client|none). (#105327, @julianvmodesto) [SIG CLI and Testing]

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
  --> (#104782, @kerthcet) [SIG Scheduling and Testing]
- Ephemeral containers have reached beta maturity and are now available by default. (#105405, @verb) [SIG API Machinery, Apps, Node and Testing]
- Introduce OS field in the Pod Spec (#104693, @ravisantoshgudimetla) [SIG API Machinery and Apps]
- Introduce v1beta3 api for scheduler. This version 
  - increases the weight of user specifiable priorities.
  The weights of following priority plugins are increased
    - TaintTolerations to 3 - as leveraging node tainting to group nodes in the cluster is becoming a widely-adopted practice
    - NodeAffinity to 2
    - InterPodAffinity to 2
  
  - Won't have HealthzBindAddress, MetricsBindAddress fields (#104251, @ravisantoshgudimetla) [SIG Scheduling and Testing]
- JSON log output is configurable and now supports writing info messages to stdout and error messages to stderr. Info messages can be buffered in memory. The default is to write both to stdout without buffering, as before. (#104873, @pohly) [SIG API Machinery, Architecture, CLI, Cluster Lifecycle, Instrumentation, Node and Scheduling]
- JobTrackingWithFinalizers graduates to beta. Feature is enabled by default. (#105687, @alculquicondor) [SIG Apps and Testing]
- Remove NodeLease feature gate that was graduated and locked to stable in 1.17 release. (#105222, @cyclinder) [SIG Apps, Node and Testing]
- TTLAfterFinished is now GA and enabled by default (#105219, @sahilvv) [SIG API Machinery, Apps, Auth and Testing]
- The "Generic Ephemeral Volume" feature graduates to GA. It is now enabled unconditionally. (#105609, @pohly) [SIG API Machinery, Apps, Auth, Node, Scheduling, Storage and Testing]
- The legacy scheduler policy config is removed in v1.23, the associated flags policy-config-file, policy-configmap, policy-configmap-namespace and use-legacy-policy-config are also removed. Migrate to Component Config instead, see https://kubernetes.io/docs/reference/scheduling/config/ for details. (#105424, @kerthcet) [SIG Scheduling and Testing]
- Track the number of Pods with a Ready condition in Job status. The feature is alpha and needs the feature gate JobReadyPods to be enabled. (#104915, @alculquicondor) [SIG API Machinery, Apps, CLI and Testing]

### Feature

- Add a new `distribute-cpus-across-numa` option to the static `CPUManager` policy. When enabled, this will trigger the `CPUManager` to evenly distribute CPUs across NUMA nodes in cases where more than one NUMA node is required to satisfy the allocation. (#105631, @klueska) [SIG Node]
- Add support to generate client-side binaries for windows/arm64 platform (#104894, @pacoxu) [SIG CLI, Testing and Windows]
- Added a new feature gate `CustomResourceValidationExpressions` to enable expression validation for Custom Resource. (#105107, @cici37) [SIG API Machinery]
- Adds new [alpha] command 'kubectl events' (#99557, @bboreham) [SIG CLI]
- Client-go, using log level 9, trace the following events of an http request:
      - dns lookup
      - tcp dialing
      - tls handshake
      - time to get a connection from the pool
      - time to process a request (#105156, @aojea) [SIG API Machinery]
- Client-go: pass `DeleteOptions` down to the fake client `Reactor` (#102945, @chenchun) [SIG API Machinery, Apps and Auth]
- Enhance scheduler volumebinding plugin to handle Lost PVC as UnschedulableAndUnresolvable during PreFilter stage (#105245, @yibozhuang) [SIG Scheduling and Storage]
- Feature-gate `StorageObjectInUseProtection` has been deprecated and cannot be disabled. It will be completely removed in 1.25 (#105495, @ikeeip) [SIG Apps]
- Kubectl will now provide shell completion choices for the --output/-o flag (#105851, @marckhouzam) [SIG CLI]
- Kubernetes is now built with Golang 1.17.2 (#105563, @mengjiao-liu) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- Move the getAllocatableResources endpoint in podresource-api to the beta that will make it enabled by default. (#105003, @swatisehgal) [SIG Node and Testing]
- Node affinity, node selector and tolerations are now mutable for jobs that are suspended and have never been started (#105479, @ahg-g) [SIG Apps, Scheduling and Testing]
- Pod template annotations and labels are now mutable for jobs that are suspended and have never been started (#105980, @ahg-g) [SIG Apps]
- PodSecurity: add a container image and manifests for the PodSecurity validating admission webhook (#105923, @liggitt) [SIG Auth]
- PodSecurity: in 1.23+ restricted policy levels, pods and containers which set runAsUser=0 are forbidden at admission-time; previously, they would be rejected at runtime (#105857, @liggitt) [SIG Auth]
- Shell completion now knows to continue suggesting resource names when the command supports it.  For example "kubectl get pod pod1 <TAB>" will suggest more pod names. (#105711, @marckhouzam) [SIG CLI]
- Support to enable Hyper-V in GCE Windows Nodes created with kube-up (#105999, @mauriciopoppe) [SIG Cloud Provider and Windows]
- The CPUManager policy options are now enabled, and we introduce a graduation path for the new CPU Manager policy options. (#105012, @fromanirh) [SIG Node and Testing]
- The etcd container image now supports Windows. (#92433, @claudiubelu) [SIG API Machinery and Windows]
- The pods and pod controllers that are exempted from the PodSecurity admission process are now marked with the "pod-security.kubernetes.io/exempt: user/namespace/runtimeClass" annotation, based on what caused the exemption.
  
  The enforcement level that allowed or denied pod during PodSecurity admission is now marked by the "pod-security.kubernetes.io/enforce-policy" annotation.
  
  The annotation that informs about audit policy violations changed from ""pod-security.kubernetes.io/audit" to ""pod-security.kubernetes.io/audit-violation". (#105908, @stlaz) [SIG Auth]
- When feature gate JobTrackingWithFinalizers is enabled:
  - Limit the number of pods tracked in a single job sync to avoid starvation of small jobs.
  - The metric job_pod_finished_total counts the number of finished pods tracked by the job controller (#105197, @alculquicondor) [SIG Apps, Instrumentation and Testing]

### Failing Test

- Fixes hostpath storage e2e tests within SELinux enabled env (#104551, @Elbehery) [SIG Testing]

### Bug or Regression

- (PodSecurity admission) errors validating workload resources (deployment, replicaset, etc.) no longer block admission. (#106017, @tallclair) [SIG Auth]
- Add Pod Security admission metrics: pod_security_evaluations_total, pod_security_exemptions_total, pod_security_errors_total (#105898, @tallclair) [SIG Auth, Instrumentation and Testing]
- Apimachinery: pretty-printed json and yaml output is now indented consistently (#105466, @liggitt) [SIG API Machinery]
- Change `kubectl diff --invalid-arg` status code from 1 to 2 to match docs (#105445, @ardaguclu) [SIG CLI]
- Client-go uses the same http client for all the generated groups and versions, allowing to share customized transports for multiple groups versions. (#105490, @aojea) [SIG API Machinery, Auth, Instrumentation and Testing]
- Evicted and other terminated pods will no longer revert to Running phase (#105462, @ehashman) [SIG Node and Testing]
- Fix pod name of NonIndexed jobs to not include rogue -1 substring (#105676, @alculquicondor) [SIG Apps]
- Fix scoring for NodeResourcesBalancedAllocation plugins when nodes have containers with no requests. (#105845, @ahmad-diaa) [SIG Scheduling]
- Fix: consolidate logs for instance not found error
  fix: skip not found nodes when reconciling LB backend address pools (#105188, @nilo19) [SIG Cloud Provider]
- Fix: do not delete the lb that does not exist (#105777, @nilo19) [SIG Cloud Provider]
- Fix: ignore not a VMSS error for VMAS nodes in EnsureBackendPoolDeleted. (#105185, @ialidzhikov) [SIG Cloud Provider]
- Fix: leave the probe path empty for TCP probes (#105253, @nilo19) [SIG Cloud Provider]
- Fix: remove VMSS and VMSS instances from SLB backend pool only when necessary (#105839, @nilo19) [SIG Cloud Provider]
- Fix: skip instance not found when decoupling vmss from lb (#105666, @nilo19) [SIG Cloud Provider]
- Fixed a bug that prevents PersistentVolume that has a Claim UID which doesn't exist in local cache but exists in ETCD from being updated to Released phase. (#105211, @xiaopingrubyist) [SIG Apps]
- Fixed architecture within manifest for non `amd64` etcd images. (#105484, @saschagrunert) [SIG API Machinery]
- Fixes a bug that could result in the EndpointSlice controller unnecessarily updating EndpointSlices associated with a Service that had Topology Aware Hints enabled. (#105267, @llhuii) [SIG Apps and Network]
- Fixes the `should support building a client with a CSR` e2e test to work with clusters configured with short certificate lifetimes (#105396, @liggitt) [SIG Auth and Testing]
- Generic ephemeral volumes can be used also as raw block devices, but the Pod validation was refusing to create pods with that combination. (#105682, @pohly) [SIG Apps, Storage and Testing]
- Generic ephemeral volumes were not considered properly by the the node limits scheduler filter and the kubelet hostpath check. (#100482, @pohly) [SIG Node, Scheduling, Storage and Testing]
- Kube-apiserver: fix a memory leak when deleting multiple objects with a deletecollection. (#105606, @sxllwx) [SIG API Machinery]
- Kubeadm: do not allow empty "--config" paths to be passed to "kubeadm kubeconfig user" (#105649, @navist2020) [SIG Cluster Lifecycle]
- Kubelet did not report `kubelet_volume_stats_*` metrics for generic ephemeral voiumes. (#105569, @pohly) [SIG Node]
- Kubelet's Node Grace Shutdown will terminate probes when shutting down. (#105215, @rphillips) [SIG Node]
- Kubernetes object references (= name + namespace) were not logged as struct when using JSON as log output format. (#104877, @pohly) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Podresources interface was changed, now it returns only isolated cpus (#97415, @AlexeyPerevalov) [SIG Node and Testing]
- Release-note Removed error message label from kubelet_started_pods_errors_total metric (#105213, @yxxhero) [SIG Instrumentation and Node]
- Resolves a potential issue with GC and NS controllers which may delete objects after getting a 404 response from the server during its startup. This PR ensures that requests to aggregated APIs will get 503, not 404 while the APIServiceRegistrationController hasn't finished its job. (#104748, @p0lyn0mial) [SIG API Machinery]
- Revert building binaries with PIE mode. (#105352, @ehashman) [SIG Node, Release and Security]
- Support more than 100 disk mounts on Windows (#105673, @andyzhangx) [SIG Storage and Windows]
- Support using negative array index in json patch replace operations. (#105896, @zqzten) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- The --leader-elect* CLI args are now honored in scheduler. (#105712, @Huang-Wei) [SIG Scheduling]
- The client-go dynamic client sets the header 'Content-Type: application/json' by default (#104327, @sxllwx) [SIG API Machinery]
- The pods/binding subresource now honors `metadata.uid` and `metadata.resourceVersion` preconditions (#105913, @aholic) [SIG Scheduling]
- Topology Hints now excludes control plane notes from capacity calculations. (#104744, @robscott) [SIG Apps and Network]
- Watch requests that are delegated to aggregated apiservers no longer reserve concurrency units (seats) in the API Priority and Fairness dispatcher for their entire duration. (#105511, @benluddy) [SIG API Machinery]
- `--log-flush-frequency` had no effect in several commands or was missing. Help and warning texts were not always using the right format for a command (`add_dir_header` instead of `add-dir-header`). Fixing this included cleaning up flag handling in component-base/logs: that package no longer adds flags to the global flag sets. Commands which want the klog and --log-flush-frequency flags must explicitly call logs.AddFlags; the new cli.Run does that for commands. That helper function also covers flag normalization and printing of usage and errors in a consistent way (print usage text first if parsing failed, then the error). (#105076, @pohly) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling and Testing]

### Other (Cleanup or Flake)

- All klog flags except for `-v` and `-vmodule` are deprecated. Support for `-vmodule` is only guaranteed for the text log format. (#105042, @pohly) [SIG API Machinery, Architecture, CLI, Cluster Lifecycle and Instrumentation]
- Kube-apiserver: requests to node, service, and pod `/proxy` subresources with no additional URL path now only automatically redirect GET and HEAD requests. (#95128, @Riaankl) [SIG API Machinery, Architecture and Testing]
- Migrate `pkg/scheduler/framework/plugins/interpodaffinity/filtering.go`,`pkg/scheduler/framework/plugins/podtopologyspread/filtering.go`, `pkg/scheduler/framework/plugins/volumezone/volume_zone.go` to structured logging (#105931, @mengjiao-liu) [SIG Instrumentation and Scheduling]
- Migrated `cmd/kube-scheduler/app/server.go`, `pkg/scheduler/framework/plugins/nodelabel/node_label.go`, `pkg/scheduler/framework/plugins/nodevolumelimits/csi.go`, `pkg/scheduler/framework/plugins/nodevolumelimits/non_csi.go` to structured logging (#105855, @shivanshu1333) [SIG Instrumentation and Scheduling]
- Migrated pkg/proxy to structured logging (#104891, @shivanshu1333) [SIG Network]
- Migrated pkg/proxy/ipvs to structured logging (#104932, @shivanshu1333) [SIG Network]
- Support allocating whole NUMA nodes in the CPUManager when there is not a 1:1 mapping between socket and NUMA node (#102015, @klueska) [SIG Node]

## Dependencies

### Added
- sigs.k8s.io/json: c049b76

### Changed
- github.com/evanphx/json-patch: [v4.11.0+incompatible → v4.12.0+incompatible](https://github.com/evanphx/json-patch/compare/v4.11.0...v4.12.0)
- github.com/go-logr/logr: [v1.1.0 → v1.2.0](https://github.com/go-logr/logr/compare/v1.1.0...v1.2.0)
- github.com/go-logr/zapr: [v1.1.0 → v1.2.0](https://github.com/go-logr/zapr/compare/v1.1.0...v1.2.0)
- k8s.io/klog/v2: v2.20.0 → v2.30.0
- k8s.io/utils: bdf08cb → cb0fa31

### Removed
_Nothing has changed._



# v1.23.0-alpha.3


## Downloads for v1.23.0-alpha.3

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes.tar.gz) | 083e6ca03c9d701768b1b5666f354223a3f7dca9fc6410ce45bbf5947152620e300b46df9b6019134e7d736ba44916537eb3bea8fa57e5f7bc3cc34898b4a5dd
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-src.tar.gz) | c3fc74d52e1b7e808c03b9caa30e3e73be30eb8330ce676000b93d5324bbdba93bd005d125b999ba937b79d4751af99b37986911365416f7175d223345f95914

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-darwin-amd64.tar.gz) | 31d8adc657afbd305df18bfec397a825536357e23b241a19aa538b6ddefefc59743f737db98756e04deea89cc6f260d40a80f02b4d1dc34af1d19e8d796dcd8a
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-darwin-arm64.tar.gz) | b69c4d6cde1c476bafa2ca9916ce3e5bf7286be0ff6a08193bdd1a954ba89b64b1b14193d1acec17ccc141024ee3097971448017b5c9f1327e0961b1e92b2224
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-386.tar.gz) | 059f25ee48aa4b0d1621d6ba87af8fb7e765634d723d98a4e9739f50d3703e7dd3973f4d1ed886c0f3ad6eba165ed81d4e63ecde3b39e66fcbec7d3aa2dfed2e
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-amd64.tar.gz) | 291dba14160803065895799adcde39bdad7a5b0372403f283d6d5e9a094fe1fc79c70e7546f93ee692b9fd297e2667cb558e4209161ecb4bf89965df5746ed4d
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-arm.tar.gz) | 988e12cd7466033578acc487447df376c409e4f79726a4721af1aedbe931e927b22a93d6224891b61b55c7a0ec12e42d8cfcd40e15a9a0cbbc1dbf0e59ab0341
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-arm64.tar.gz) | b3f21dac41b38e671fa7a95892468e2c27fab51abf9c77b336550e5ec213af204e16cac11dd76262fedb0087cf5ad1950af7e36599a38d50cc270cf831cd4f0b
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-ppc64le.tar.gz) | beebf01e2e4ff09bb711284bb9a5c7cc519e4ac8a826dc829394fa28bd9a3149ba73088eaf6712d39a8cab96b0a1c2859e9d5955fee892b759eaddcdeaa8b93c
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-linux-s390x.tar.gz) | 87e5d3d8ba01f9fefb2300e9f06146a254d39d72eaa10cad8c444428b738b3763483ee9eb82f0a13d2ff5aba35fdcb4320598fd5a6a2a07ea3fd00b4ac682d3c
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-windows-386.tar.gz) | 71bfc5a1df9c47735476af10225830212f68c83357ff7d443e18f9b7881524db910781a95d11ff6697cb587352059b5841f7b24fda40b5302ad252bfb6da7e51
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-client-windows-amd64.tar.gz) | 078b0c698f9535f3eee41ecf162d57e2ace67243da36067b78b30cfbb7b27cfcf97af4c5db48cdd592953e26b42b31794002eb96317476849e89e2126c6df99d

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-amd64.tar.gz) | 951b790158dadf46c32e1a1e9c12f2cc8f41e1645602ebff6b4130a08a377bc6d92549186b420332d620d67191123d98a5d717ac0f5ee9643bebe88947ead8fa
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-arm.tar.gz) | 0e7a5b9f39b4f45c45bdb5a19dd3695d28f53e1039d76bc572421c707917944d28b1dbfc36e59214b5bc2b93a787900d8e6eb0b587aa801ea8a8faacdb814a4e
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-arm64.tar.gz) | 921e060120b8651a0f80977360faca9f207189cee10bc61f669ceba4e540ef48c0ceff1a877ee4c7d31b01b88096bce93c577f68f93b2341c8542dfd89972b60
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-ppc64le.tar.gz) | 292cde446b754a87f4ef5384fadbd30017e53ed2744d45a724be467c86ccd9837bfb490db6396642a869937f2f0d080d9655e89ca3345f8365d109a9bcdd18d9
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-server-linux-s390x.tar.gz) | e0ea667f828ce3b36ca4b2a05fb286da5eb321852c50caf0957694553caf2908b27bcc37a5a82277a2606cf6ff4d9e33617ad61628845d9c21f5cf68c960ca92

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-amd64.tar.gz) | e13cd3f75628d354bd1544a5495600fb905741431eb4af4da3d980cc0b7565e3f9c1585d9686cc4e967e54fb854f05bbedfe0c60bb7b855fa027ac8ac45b26e0
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-arm.tar.gz) | 6c91b42350528692ff558b667bffd41c5b967c7aa6101471274e4b16b0ac6f84afe01722881328fd4f6f8fe71c7852620fa000186c6f7e56e498fcc2c67ad793
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-arm64.tar.gz) | 81728e1388e9cdb436d6847c868f28ab2771331e5e40cd5a7af13cb8dc80a7e4e66a215c12f8183b4884807a3962f913ef5343b889e3c4ecd0e410e8d53aaea9
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-ppc64le.tar.gz) | 299649f1b25cc38f3a7543ef4d3ee6d42c85e24ac41b4eb61927bc5c5f0c533a39f9ddd4d5ad1df54c625d77aeb41f6c31b1ca7fd8983262f84fefdf1cb2cfd0
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-linux-s390x.tar.gz) | fd6cbc93f98abff9803b43215af6e75a4f7b91ca06969220a779468f34b5ec5ec69f20b529e0cd7b10ba8769bbe2507d46f84ce1d8cd0760380ab9264dd94672
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.3/kubernetes-node-windows-amd64.tar.gz) | a5bfaf2e3ad8d3d2127c3e3e0f131c615a03563253da6bf0e1fd793f6ef71287f341ce1bd0d35eb9a81e0721a5baf03e7c72863b5ed8eb45e8fe70573904ed54

## Changelog since v1.23.0-alpha.2

## Changes by Kind

### Deprecation

- Remove 'master' as a valid EgressSelection type in the EgressSelectorConfiguration API. ([#102242](https://github.com/kubernetes/kubernetes/pull/102242), [@pacoxu](https://github.com/pacoxu)) [SIG API Machinery and Cloud Provider]
- Remove VolumeSubpath feature gate ([#105090](https://github.com/kubernetes/kubernetes/pull/105090), [@saad-ali](https://github.com/saad-ali)) [SIG Apps, Node and Storage]
- The deprecated --experimental-bootstrap-kubeconfig flag has been removed.
  This can be set via --bootstrap-kubeconfig. ([#103172](https://github.com/kubernetes/kubernetes/pull/103172), [@niulechuan](https://github.com/niulechuan)) [SIG Node]

### API Change

- Client-go impersonation config can specify a UID to pass impersonated uid information through in requests. ([#104483](https://github.com/kubernetes/kubernetes/pull/104483), [@margocrawf](https://github.com/margocrawf)) [SIG API Machinery, Auth and Testing]
- IPv6DualStack feature moved to stable.
  Controller Manager flags for the node IPAM controller have slightly changed:
  1. When configuring a dual-stack cluster, the user must specify both --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6 to set the per-node IP mask sizes, instead of the previous --node-cidr-mask-size flag.
  2. The --node-cidr-mask-size flag is mutually exclusive with --node-cidr-mask-size-ipv4 and --node-cidr-mask-size-ipv6.
  3. Single-stack clusters do not need to change, but may choose to use the more specific flags.  Users can use either the older --node-cidr-mask-size flag or one of the newer --node-cidr-mask-size-ipv4 or --node-cidr-mask-size-ipv6 flags to configure the per-node IP mask size, provided that the flag's IP family matches the cluster's IP family (--cluster-cidr). ([#104691](https://github.com/kubernetes/kubernetes/pull/104691), [@khenidak](https://github.com/khenidak)) [SIG API Machinery, Apps, Auth, Cloud Provider, Cluster Lifecycle, Network, Node and Testing]
- Kubelet: turn the KubeletConfiguration v1beta1 `ResolverConfig` field from a `string` to `*string`. ([#104624](https://github.com/kubernetes/kubernetes/pull/104624), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle and Node]

### Feature

- Add mechanism to load simple sniffer class into fluentd-elasticsearch image ([#92853](https://github.com/kubernetes/kubernetes/pull/92853), [@cosmo0920](https://github.com/cosmo0920)) [SIG Cloud Provider and Instrumentation]
- Kubeadm: do not check if the '/etc/kubernetes/manifests' folder is empty on joining worker nodes during preflight ([#104942](https://github.com/kubernetes/kubernetes/pull/104942), [@SataQiu](https://github.com/SataQiu)) [SIG Cluster Lifecycle]
- The kube-apiserver's Prometheus metrics have been extended with some that describe the costs of handling LIST requests.  They are as follows.
  - *apiserver_cache_list_total*: Counter of LIST requests served from watch cache, broken down by resource_prefix and index_name
  - *apiserver_cache_list_fetched_objects_total*: Counter of objects read from watch cache in the course of serving a LIST request, broken down by resource_prefix and index_name
  - *apiserver_cache_list_evaluated_objects_total*: Counter of objects tested in the course of serving a LIST request from watch cache, broken down by resource_prefix
  - *apiserver_cache_list_returned_objects_total*: Counter of objects returned for a LIST request from watch cache, broken down by resource_prefix
  - *apiserver_storage_list_total*: Counter of LIST requests served from etcd, broken down by resource
  - *apiserver_storage_list_fetched_objects_total*: Counter of objects read from etcd in the course of serving a LIST request, broken down by resource
  - *apiserver_storage_list_evaluated_objects_total*: Counter of objects tested in the course of serving a LIST request from etcd, broken down by resource
  - *apiserver_storage_list_returned_objects_total*: Counter of objects returned for a LIST request from etcd, broken down by resource ([#104983](https://github.com/kubernetes/kubernetes/pull/104983), [@MikeSpreitzer](https://github.com/MikeSpreitzer)) [SIG API Machinery and Instrumentation]
- Turn on CSIMigrationAzureDisk by default on 1.23 ([#104670](https://github.com/kubernetes/kubernetes/pull/104670), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]

### Bug or Regression

- Changes behaviour of kube-proxy start; does not attempt to set specific sysctl values (which does not work in recent Kernel versions anymore in non-init namespaces), when the current sysctl values are already set higher. ([#103174](https://github.com/kubernetes/kubernetes/pull/103174), [@Napsty](https://github.com/Napsty)) [SIG Network]
- Fix job controller syncs: In case of conflicts, ensure that the sync happens with the most up to date information. Improves reliability of JobTrackingWithFinalizers. ([#105214](https://github.com/kubernetes/kubernetes/pull/105214), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps]
- Fix system default topology spreading when nodes don't have zone labels. Pods correctly spread by default now. ([#105046](https://github.com/kubernetes/kubernetes/pull/105046), [@alculquicondor](https://github.com/alculquicondor)) [SIG Scheduling]
- Headless Services with no selector which were created without dual-stack enabled will be defaulted to RequireDualStack instead of PreferDualStack.  This is consistent with such Services which are created with dual-stack enabled. ([#104986](https://github.com/kubernetes/kubernetes/pull/104986), [@thockin](https://github.com/thockin)) [SIG Network]
- Kube-apiserver: events created via the `events.k8s.io` API group for cluster-scoped objects are now permitted in the default namespace as well for compatibility with events clients and the `v1` API ([#100125](https://github.com/kubernetes/kubernetes/pull/100125), [@h4ghhh](https://github.com/h4ghhh)) [SIG API Machinery, Apps and Testing]
- Kube-controller incorrectly enabled support for generic ephemeral inline volumes if the storage object in use protection feature was enabled. ([#104913](https://github.com/kubernetes/kubernetes/pull/104913), [@pohly](https://github.com/pohly)) [SIG API Machinery]
- Kubeadm: switch the preflight check (called 'Swap') that verifies if swap is enabled on Linux hosts to report a warning instead of an error. This is related to the graduation of the NodeSwap feature gate in the kubelet to Beta and being enabled by default in 1.23 - allows swap support on Linux hosts. In the next release of kubeadm (1.24) the preflight check will be removed, thus we recommend that you stop using it - e.g. via --ignore-preflight-errors or the kubeadm config. ([#104854](https://github.com/kubernetes/kubernetes/pull/104854), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Makes the etcd client (used by the API server) retry certain types of errors. The full list of retriable (codes.Unavailable) errors can be found at https://github.com/etcd-io/etcd/blob/main/api/v3rpc/rpctypes/error.go#L72 ([#105069](https://github.com/kubernetes/kubernetes/pull/105069), [@p0lyn0mial](https://github.com/p0lyn0mial)) [SIG API Machinery]
- When a static pod file is deleted and recreated while using a fixed UID, the pod was not properly restarted. ([#104847](https://github.com/kubernetes/kubernetes/pull/104847), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node and Testing]
- XFS-filesystems are now force-formatted (option `-f`) in order to avoid problems being formatted due to detection of magic super-blocks. This aligns with the behaviour of formatting of ext3/4 filesystems. ([#104923](https://github.com/kubernetes/kubernetes/pull/104923), [@davidkarlsen](https://github.com/davidkarlsen)) [SIG Storage]

### Other (Cleanup or Flake)

- Enhanced error message for nodes not selected by scheduler due to pod's PersistentVolumeClaim(s) bound to PersistentVolume(s) that do not exist. ([#105196](https://github.com/kubernetes/kubernetes/pull/105196), [@yibozhuang](https://github.com/yibozhuang)) [SIG Scheduling and Storage]
- Kubeadm: remove the --port flag from the manifest for the kube-scheduler since the flag has been a NO-OP since 1.23 and insecure serving was removed for the component. ([#105034](https://github.com/kubernetes/kubernetes/pull/105034), [@pacoxu](https://github.com/pacoxu)) [SIG Cluster Lifecycle]
- Migrate `cmd/proxy/{config, healthcheck, winkernel}` to structured logging ([#104944](https://github.com/kubernetes/kubernetes/pull/104944), [@jyz0309](https://github.com/jyz0309)) [SIG Network]
- Migrate cmd/proxy/app and pkg/proxy/meta_proxier to structured logging ([#104928](https://github.com/kubernetes/kubernetes/pull/104928), [@jyz0309](https://github.com/jyz0309)) [SIG Apps, Cluster Lifecycle, Network, Node and Testing]
- Migrate pkg/proxy to structured logs ([#104908](https://github.com/kubernetes/kubernetes/pull/104908), [@CIPHERTron](https://github.com/CIPHERTron)) [SIG Network]
- Migrated pkg/proxy/winuserspace to structured logging ([#105035](https://github.com/kubernetes/kubernetes/pull/105035), [@shivanshu1333](https://github.com/shivanshu1333)) [SIG Network]
- The `BoundServiceAccountTokenVolume` feature gate that is GA since v1.22 is unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104167](https://github.com/kubernetes/kubernetes/pull/104167), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Auth]
- The `SupportPodPidsLimit` and  `SupportNodePidsLimit` feature gates that are GA since v1.20 are unconditionally enabled, and can no longer be specified via the `--feature-gates` argument. ([#104163](https://github.com/kubernetes/kubernetes/pull/104163), [@ialidzhikov](https://github.com/ialidzhikov)) [SIG Node]
- Update build images to Debian 11 (Bullseye)
  - debian-base:bullseye-v1.0.0
  - debian-iptables:bullseye-v1.0.0
  - go-runner:v2.3.1-go1.17.1-bullseye.0
  - kube-cross:v1.23.0-go1.17.1-bullseye.1
  - setcap:bullseye-v1.0.0
  - cluster/images/etcd: Build 3.5.0-2 image
  - test/conformance/image: Update runner image to base-debian11 ([#105158](https://github.com/kubernetes/kubernetes/pull/105158), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Architecture, Release and Testing]

## Dependencies

### Added
_Nothing has changed._

### Changed
- github.com/json-iterator/go: [v1.1.11 → v1.1.12](https://github.com/json-iterator/go/compare/v1.1.11...v1.1.12)
- github.com/modern-go/reflect2: [v1.0.1 → v1.0.2](https://github.com/modern-go/reflect2/compare/v1.0.1...v1.0.2)

### Removed
_Nothing has changed._



# v1.23.0-alpha.2


## Downloads for v1.23.0-alpha.2

### Source Code

filename | sha512 hash
-------- | -----------
[kubernetes.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes.tar.gz) | 121d51f42a52b28e27a4b2f914a4f80fa3fba6328e6a4a5c96dec39c5b28c05461fcc290ef35a49058e237091532b24db3cd8c61801bcb6736aee1dd7dbcffc3
[kubernetes-src.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-src.tar.gz) | 641d47241acfadb3b13bccec57795749d2c9e3e07ffa7aa4b30df3a488643631eb8e5cd581bcfb764dff4ac5ed755f72d94e80746142123b09e1675e81421a91

### Client Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-client-darwin-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-darwin-amd64.tar.gz) | f734cb514ee56adcb2d991a6f0550df907c72f8a61cc2a13117e61b8d5826ff942a582a2e9383deb1a61d5df2243362f1327942a3b4883490eb3296647ce3737
[kubernetes-client-darwin-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-darwin-arm64.tar.gz) | 24d1f851cd5782f8f39054e37beda1554dadd8a28cb3272b00d50fc095d1fc3018768c1ea72a44eda61ff0f58f71b33dd28cbdc54467d620e87c3694ecf14cc2
[kubernetes-client-linux-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-386.tar.gz) | 082ad4abea58de3b629fc2ed4560a836cdbeb1adefb0c4cf47044bf33c750d8fcd8a06e2c4ce365853e83a58d52e0129d510a698dd894bd1261f8184dd1cab42
[kubernetes-client-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-amd64.tar.gz) | b3b0b23479c05b57ca574cf17cdcde7e716033bc4f6a80532d1175d8e533e3202bece0dcf503731d5a60319c526ce1ce4a0bc900bf87536321208a59cf890e35
[kubernetes-client-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-arm.tar.gz) | f5dac2976ce04310f74bba6102080554309b851fbd966ff1220d3eb23089db8eb8da519a6bd8865c94f2f24346a4d27eb40fd0a3ff06ca9c6874e1fc6f356b67
[kubernetes-client-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-arm64.tar.gz) | 057b372150749b13a38e04802c7cf566765e0fbb27f1b5f7bf6d3cc3f71eb3020916ea7f8579ecc7fcc10e2db1b5c8caa31a1e8a3aac80da86e4e777f515d42f
[kubernetes-client-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-ppc64le.tar.gz) | 9a090d22aeba011c6d039bff59dbdc23ac4a112828db3cbba588d8b0ee1cd14d16e0eacefbb000e5a3ff26bcce4730824819f86a99b7a9826f35fa9964f9f27a
[kubernetes-client-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-linux-s390x.tar.gz) | 435e20055badb619289dc7c572af300bd2f86068d0b8f326e8d9abfda5347f2449e316158c412e9b946a2541208c3e8cc6e5c823946e74ac4fc2d594d410179a
[kubernetes-client-windows-386.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-windows-386.tar.gz) | 55f192a4d095d494bb53af1b7133124b762a677eb46247b9dba71d10ea6830b37c30d603908e7a9c63f371baff508b19406e89b231ed5ece0497627f09753f68
[kubernetes-client-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-client-windows-amd64.tar.gz) | 944059d1f1918a793490b95be8130d06189508ba8e79e79ca8cfd2ab98bf396ac551786514b093cc6afe4b3fd15736d728cfcdce18bb32fbee41bc0a97f5c4be

### Server Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-server-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-amd64.tar.gz) | a76a4b86ee151ba027f7cf4a2072451ae4c829182bb14e00ce1967421744bfc1e58f141b6eaf2ab27ece67054ae307f8e0768477ab9c3c4749eaad397d495182
[kubernetes-server-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-arm.tar.gz) | 95aeb4eb473ab4920d81904bc89c6126732b9c6888f9e57493ee99d692042ca44f6844ac1dade1409565f4d9fbec59445402e1f7deac6cbf5b6df16ac814b58c
[kubernetes-server-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-arm64.tar.gz) | 3c56e906aafc2a1ac72300352a334662bec5d59e3e523c19b9d65bc52ad9075dc2631f259513efd0f654e220fe0e7d54dfa5028d7eaad81d5d87ca251653f75d
[kubernetes-server-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-ppc64le.tar.gz) | b74bacafe9bb6a7cf407747b03e78ae3873e50deec4eaa08758d5e1d5287ac23af59b3ef26f888fe4cd44ccb1455beafcd1384e700230eb445720e3acae5f2e3
[kubernetes-server-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-server-linux-s390x.tar.gz) | d3f8f8d9c233b114129f615252d42782cd366978a49506393a40af3f8b5b1250ce99e9806881675e112a69270a0411fb2f00ea19b99ad7415b9e0074beb2726d

### Node Binaries

filename | sha512 hash
-------- | -----------
[kubernetes-node-linux-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-amd64.tar.gz) | 146e2f762c179178a57a8c7af7c26470c5d580b8ff8400615162ad1056625f87ce2b32598538d82652f88639e54afb782810529b074c36eb52cc6374414a6181
[kubernetes-node-linux-arm.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-arm.tar.gz) | 9357d1b387e1b049fb6cec06a7081afc2ce7e906484c9b061fb0449d147a6c4f9c9dc7a9219cdca5ed71df6c73784f360018d9e48d4fa2aa7eeabef60649d7a4
[kubernetes-node-linux-arm64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-arm64.tar.gz) | 8394f8f9d6ee823cb9a470ea67e15d4d0c6aca7065fe826788f50955905373fc3cdddd6db43901c07736588d8d6a3d3e2916bc8d45fd6bd06307583686137a0a
[kubernetes-node-linux-ppc64le.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-ppc64le.tar.gz) | 7211cb426834484bff39f1ab3c9541203429039f8f5e522ca9e28c43da749e197128a3cae28db0467fc339305d2f23f85e8b4ed9ec116506c3d8076744a88d5e
[kubernetes-node-linux-s390x.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-linux-s390x.tar.gz) | a7c1a38250398171d3df5865749e9928867c4f44106ae66d44cf9f948ce4f4eed9d1f273a5d369996425b1e12482fceccde4c7652770a8c9fb3f161811323b69
[kubernetes-node-windows-amd64.tar.gz](https://dl.k8s.io/v1.23.0-alpha.2/kubernetes-node-windows-amd64.tar.gz) | 2007b3b16597cc06b486f87f35b6c637404f07c11d88b8c8e1c2c9bbea97f762bd7d4f9a31f42f78a917c595af5cb89e6885dd88f3766836dc6e4ec79cf084f2

## Changelog since v1.23.0-alpha.1

## Changes by Kind

### Deprecation

- Controller-manager: the following flags have no effect and would be removed in v1.24:
  - `--port`
  - `--address`
  The insecure port flags `--port` may only be set to 0 now.
  Also `metricsBindAddress` and `healthzBindAddress` fields from `kubescheduler.config.k8s.io/v1beta1` are no-op and expected to be empty. Removed in `kubescheduler.config.k8s.io/v1beta2` completely.
  
  In addition, please be careful that:
  - kube-scheduler MUST start with `--authorization-kubeconfig` and `--authentication-kubeconfig` correctly set to get authentication/authorization working.
  - liveness/readiness probes to kube-scheduler MUST use HTTPS now, and the default port has been changed to 10259.
  - Applications that fetch metrics from kube-scheduler should use a dedicated service account which is allowed to access nonResourceURLs `/metrics`. ([#96345](https://github.com/kubernetes/kubernetes/pull/96345), [@ingvagabund](https://github.com/ingvagabund)) [SIG Cloud Provider, Scheduling and Testing]
- Removed deprecated metric `scheduler_volume_scheduling_duration_seconds` ([#104518](https://github.com/kubernetes/kubernetes/pull/104518), [@dntosas](https://github.com/dntosas)) [SIG Instrumentation, Scheduling and Storage]

### API Change

- A small regression in Service updates was fixed.  The circumstances are so unlikely that probably nobody would ever hit it. ([#104601](https://github.com/kubernetes/kubernetes/pull/104601), [@thockin](https://github.com/thockin)) [SIG Network]
- Introduce v1beta2 for Priority and Fairness with no changes in API spec ([#104399](https://github.com/kubernetes/kubernetes/pull/104399), [@tkashem](https://github.com/tkashem)) [SIG API Machinery and Testing]
- Kube-apiserver: Fixes handling of CRD schemas containing literal null values in enums. ([#104969](https://github.com/kubernetes/kubernetes/pull/104969), [@liggitt](https://github.com/liggitt)) [SIG API Machinery, Apps and Network]
- Kubelet: turn the KubeletConfiguration v1beta1 `ResolverConfig` field from a `string` to `*string`. ([#104624](https://github.com/kubernetes/kubernetes/pull/104624), [@Haleygo](https://github.com/Haleygo)) [SIG Cluster Lifecycle and Node]
- Kubernetes is now built using go1.17 ([#103692](https://github.com/kubernetes/kubernetes/pull/103692), [@justaugustus](https://github.com/justaugustus)) [SIG API Machinery, Apps, Architecture, Auth, Autoscaling, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scheduling, Storage and Testing]
- Removed deprecated `--seccomp-profile-root`/`seccompProfileRoot` config ([#103941](https://github.com/kubernetes/kubernetes/pull/103941), [@saschagrunert](https://github.com/saschagrunert)) [SIG Node]
- Since golang 1.17 both net.ParseIP and net.ParseCIDR rejects leading zeros in the dot-decimal notation of IPv4 addresses.
  Kubernetes will keep allowing leading zeros on IPv4 address to not break the compatibility.
  IMPORTANT: Kubernetes interprets leading zeros on IPv4 addresses as decimal, users must not rely on parser alignment to not being impacted by the associated security advisory:
  CVE-2021-29923 golang standard library "net" - Improper Input Validation of octal literals in golang 1.16.2 and below standard library "net" results in indeterminate SSRF & RFI vulnerabilities.
  Reference: https://nvd.nist.gov/vuln/detail/CVE-2021-29923 ([#104368](https://github.com/kubernetes/kubernetes/pull/104368), [@aojea](https://github.com/aojea)) [SIG API Machinery, Apps, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation, Network, Node, Release, Scalability, Scheduling, Storage and Testing]
- StatefulSet minReadySeconds is promoted to beta ([#104045](https://github.com/kubernetes/kubernetes/pull/104045), [@ravisantoshgudimetla](https://github.com/ravisantoshgudimetla)) [SIG Apps and Testing]
- The `Service.spec.ipFamilyPolicy` field is now *required* in order to create or update a Service as dual-stack.  This is a breaking change from the beta behavior.  Previously the server would try to infer the value of that field from either `ipFamilies` or `clusterIPs`, but that caused ambiguity on updates.  Users who want a dual-stack Service MUST specify `ipFamilyPolicy` as either "PreferDualStack" or "RequireDualStack". ([#96684](https://github.com/kubernetes/kubernetes/pull/96684), [@thockin](https://github.com/thockin)) [SIG API Machinery, Apps, Network and Testing]
- Users of LogFormatRegistry in component-base must update their code to use the logr v1.0.0 API. The JSON log output now uses the format from go-logr/zapr (no `v` field for error messages, additional information for invalid calls) and has some fixes (correct source code location for warnings about invalid log calls). ([#104103](https://github.com/kubernetes/kubernetes/pull/104103), [@pohly](https://github.com/pohly)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- When creating an object with generateName, if a conflict occurs the server now returns an AlreadyExists error with a retry option. ([#104699](https://github.com/kubernetes/kubernetes/pull/104699), [@vincepri](https://github.com/vincepri)) [SIG API Machinery]

### Feature

- Add fish shell completion to kubectl ([#92989](https://github.com/kubernetes/kubernetes/pull/92989), [@WLun001](https://github.com/WLun001)) [SIG CLI]
- Added PowerShell completion generation by running `kubectl completion powershell` ([#103758](https://github.com/kubernetes/kubernetes/pull/103758), [@zikhan](https://github.com/zikhan)) [SIG CLI]
- Added a `Processing` condition for the workqueue API
  Changed `Shutdown` for the workqueue API to wait until the work queue finishes processing all in-flight items. ([#101928](https://github.com/kubernetes/kubernetes/pull/101928), [@alexanderConstantinescu](https://github.com/alexanderConstantinescu)) [SIG API Machinery and Apps]
- Added a new flag `--append-server-path` to `kubectl proxy` that will automatically append the kube context server path to each request. ([#97350](https://github.com/kubernetes/kubernetes/pull/97350), [@FabianKramm](https://github.com/FabianKramm)) [SIG API Machinery, CLI and Testing]
- Added support for setting controller-manager log level online ([#104571](https://github.com/kubernetes/kubernetes/pull/104571), [@h4ghhh](https://github.com/h4ghhh)) [SIG API Machinery, Apps and Cloud Provider]
- Adding support for multiple --from-env-file flags ([#104232](https://github.com/kubernetes/kubernetes/pull/104232), [@lauchokyip](https://github.com/lauchokyip)) [SIG CLI]
- Cloud providers can set service account names for cloud controllers. ([#103178](https://github.com/kubernetes/kubernetes/pull/103178), [@nckturner](https://github.com/nckturner)) [SIG API Machinery and Cloud Provider]
- Health check of kube-controller-manager now includes each controller. ([#104667](https://github.com/kubernetes/kubernetes/pull/104667), [@jiahuif](https://github.com/jiahuif)) [SIG API Machinery and Cloud Provider]
- Kube-scheduler now logs node and plugin scoring  even though --v<10
  - socres of the top 3 plugins in the top 3 nodes are dumped if --v=4,5
  - socres of all plugins in the top 6 nodes are dumped if --v=6,7,8,9 ([#103515](https://github.com/kubernetes/kubernetes/pull/103515), [@muma378](https://github.com/muma378)) [SIG Scheduling]
- Kubernetes is now built with Golang 1.17.1 ([#104904](https://github.com/kubernetes/kubernetes/pull/104904), [@cpanato](https://github.com/cpanato)) [SIG API Machinery, Cloud Provider, Instrumentation, Release and Testing]
- The pause image list now contains Windows Server 2022 ([#104438](https://github.com/kubernetes/kubernetes/pull/104438), [@nick5616](https://github.com/nick5616)) [SIG Windows]
- Updates  debian-iptables to v1.6.7 to pick up CVE fixes ([#104970](https://github.com/kubernetes/kubernetes/pull/104970), [@PushkarJ](https://github.com/PushkarJ)) [SIG API Machinery, Network, Release, Security and Testing]

### Documentation

- Conformance: the test "[sig-network] EndpointSlice should have Endpoints and EndpointSlices pointing to API Server [Conformance]" only requires that there is an EndpointSlice that references the "kubernetes.default" service, it no longer requires that its named "kubernetes". ([#104664](https://github.com/kubernetes/kubernetes/pull/104664), [@aojea](https://github.com/aojea)) [SIG Architecture, Network and Testing]

### Bug or Regression

- A pod that the Kubelet rejects was still considered as being accepted for a brief period of time after rejection, which might cause some pods to be rejected briefly that could fit on the node.  A pod that is still terminating (but has status indicating it has failed) may also still be consuming resources and so should also be considered. ([#104817](https://github.com/kubernetes/kubernetes/pull/104817), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Changed kubectl describe to compute Age of an event using the count and lastObservedTime fields available in the event series ([#104482](https://github.com/kubernetes/kubernetes/pull/104482), [@harjas27](https://github.com/harjas27)) [SIG CLI]
- Don't prematurely close reflectors in case of slow initialization in watch based manager to fix issues with inability to properly mount secrets/configmaps. ([#104604](https://github.com/kubernetes/kubernetes/pull/104604), [@wojtek-t](https://github.com/wojtek-t)) [SIG Node]
- Fix Job tracking with finalizers for more than 500 pods, ensuring all finalizers are removed before counting the Pod. ([#104666](https://github.com/kubernetes/kubernetes/pull/104666), [@alculquicondor](https://github.com/alculquicondor)) [SIG Apps and Instrumentation]
- Fix a regression where the Kubelet failed to exclude already completed pods from calculations about how many resources it was currently using when deciding whether to allow more pods. ([#104577](https://github.com/kubernetes/kubernetes/pull/104577), [@smarterclayton](https://github.com/smarterclayton)) [SIG Node]
- Fix detach disk issue on deleting vmss node ([#104572](https://github.com/kubernetes/kubernetes/pull/104572), [@andyzhangx](https://github.com/andyzhangx)) [SIG Cloud Provider]
- Fix: ensure InstanceShutdownByProviderID return false for creating Azure VMs ([#104382](https://github.com/kubernetes/kubernetes/pull/104382), [@feiskyer](https://github.com/feiskyer)) [SIG Cloud Provider]
- Fix: ignore the case when comparing azure tags in service annotation ([#104705](https://github.com/kubernetes/kubernetes/pull/104705), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fix: ignore the case when updating Azure tags ([#104593](https://github.com/kubernetes/kubernetes/pull/104593), [@nilo19](https://github.com/nilo19)) [SIG Cloud Provider]
- Fixed bug where kubectl would emit duplicate warning messages for flag names that contain an underscore and recommend using a nonexistent flag in some cases ([#103852](https://github.com/kubernetes/kubernetes/pull/103852), [@brianpursley](https://github.com/brianpursley)) [SIG CLI and Cluster Lifecycle]
- Fixed client IP preservation for NodePort service with protocol SCTP in ipvs mode ([#104756](https://github.com/kubernetes/kubernetes/pull/104756), [@tnqn](https://github.com/tnqn)) [SIG Network]
- Fixed occasional pod cgroup freeze when using cgroup v1 and systemd driver. ([#104528](https://github.com/kubernetes/kubernetes/pull/104528), [@kolyshkin](https://github.com/kolyshkin)) [SIG Node]
- Fixes a regression that could cause panics in LRU caches in controller-manager, kubelet, kube-apiserver, or client-go ([#104466](https://github.com/kubernetes/kubernetes/pull/104466), [@stbenjam](https://github.com/stbenjam)) [SIG API Machinery, Architecture, Auth, CLI, Cloud Provider, Cluster Lifecycle, Instrumentation and Storage]
- Kube-apiserver: fixes an issue where an admission webhook can observe a v1 Pod object that does not have the `defaultMode` field set in the injected service account token volume ([#104523](https://github.com/kubernetes/kubernetes/pull/104523), [@liggitt](https://github.com/liggitt)) [SIG Auth]
- Kube-proxy health check ports used to listen to :<port> for each of the services. This is not needed and opens ports in addresses the cluster user may not have intended. The PR limits listening to all node address which are controlled by `--nodeport-addresses` flag. if no addresses are provided then we default to existing behavior by listening to :<port> for each service ([#104742](https://github.com/kubernetes/kubernetes/pull/104742), [@khenidak](https://github.com/khenidak)) [SIG Network]
- Kube-scheduler now doesn't print any usage message when unknown flag is specified ([#104503](https://github.com/kubernetes/kubernetes/pull/104503), [@sanposhiho](https://github.com/sanposhiho)) [SIG Scheduling]
- Metrics changes: Fix exposed buckets of `scheduler_volume_scheduling_duration_seconds_bucket` metric ([#100720](https://github.com/kubernetes/kubernetes/pull/100720), [@dntosas](https://github.com/dntosas)) [SIG Apps, Instrumentation, Scheduling and Storage]
- Scheduler resource metrics over fractional binary quantities (2.5Gi, 1.1Ki) were incorrectly reported as very small values. ([#103751](https://github.com/kubernetes/kubernetes/pull/103751), [@y-tag](https://github.com/y-tag)) [SIG API Machinery and Scheduling]

### Other (Cleanup or Flake)

- Generic ephemeral volumes: better pod events ("waiting for ephemeral volume controller to create the persistentvolumeclaim"" instead of "persistentvolumeclaim not found") ([#104605](https://github.com/kubernetes/kubernetes/pull/104605), [@pohly](https://github.com/pohly)) [SIG Scheduling and Storage]
- Kubeadm: remove the deprecated flags "--csr-only" and "--csr-dir" from "kubeadm certs renew". Please use "kubeadm certs generate-csr" instead. ([#104796](https://github.com/kubernetes/kubernetes/pull/104796), [@RA489](https://github.com/RA489)) [SIG Cluster Lifecycle]
- Migrate `pkg/scheduler` to structured logging ([#99273](https://github.com/kubernetes/kubernetes/pull/99273), [@yangjunmyfm192085](https://github.com/yangjunmyfm192085)) [SIG Scheduling]
- Migrated pkg/proxy/userspace to structured logging ([#104931](https://github.com/kubernetes/kubernetes/pull/104931), [@shivanshu1333](https://github.com/shivanshu1333)) [SIG Network]
- More detailed logging has been added to the EndpointSlice controller for Topology Aware Hints. ([#104741](https://github.com/kubernetes/kubernetes/pull/104741), [@robscott](https://github.com/robscott)) [SIG Apps and Network]
- Support for Windows Server 2022 was added to the k8s.gcr.io/pause:3.6 image. ([#104711](https://github.com/kubernetes/kubernetes/pull/104711), [@claudiubelu](https://github.com/claudiubelu)) [SIG CLI, Cloud Provider, Cluster Lifecycle, Node, Release and Testing]
- The maximum length of the CSINode id field has increased to 256 bytes to match the CSI spec ([#104160](https://github.com/kubernetes/kubernetes/pull/104160), [@pacoxu](https://github.com/pacoxu)) [SIG Storage]
- Update conformance image to use debian-base:buster-v1.9.0 ([#104696](https://github.com/kubernetes/kubernetes/pull/104696), [@PushkarJ](https://github.com/PushkarJ)) [SIG Architecture, Release, Security and Testing]
- `volume.kubernetes.io/storage-provisioner` annotation will be added to dynamic provisioning required PVC. `volume.beta.kubernetes.io/storage-provisioner` annotation is deprecated. ([#104590](https://github.com/kubernetes/kubernetes/pull/104590), [@Jiawei0227](https://github.com/Jiawei0227)) [SIG Apps and Storage]

## Dependencies

### Added
- bazil.org/fuse: 371fbbd
- github.com/go-logr/zapr: [v1.1.0](https://github.com/go-logr/zapr/tree/v1.1.0)
- github.com/kr/fs: [v0.1.0](https://github.com/kr/fs/tree/v0.1.0)
- github.com/pkg/sftp: [v1.10.1](https://github.com/pkg/sftp/tree/v1.10.1)

### Changed
- github.com/Microsoft/go-winio: [v0.4.15 → v0.4.17](https://github.com/Microsoft/go-winio/compare/v0.4.15...v0.4.17)
- github.com/Microsoft/hcsshim: [5eafd15 → v0.8.22](https://github.com/Microsoft/hcsshim/compare/5eafd15...v0.8.22)
- github.com/benbjohnson/clock: [v1.0.3 → v1.1.0](https://github.com/benbjohnson/clock/compare/v1.0.3...v1.1.0)
- github.com/bketelsen/crypt: [5cbc8cc → v0.0.4](https://github.com/bketelsen/crypt/compare/5cbc8cc...v0.0.4)
- github.com/containerd/cgroups: [0dbf7f0 → v1.0.1](https://github.com/containerd/cgroups/compare/0dbf7f0...v1.0.1)
- github.com/containerd/containerd: [v1.4.4 → v1.4.9](https://github.com/containerd/containerd/compare/v1.4.4...v1.4.9)
- github.com/containerd/continuity: [aaeac12 → v0.1.0](https://github.com/containerd/continuity/compare/aaeac12...v0.1.0)
- github.com/containerd/fifo: [a9fb20d → v1.0.0](https://github.com/containerd/fifo/compare/a9fb20d...v1.0.0)
- github.com/containerd/go-runc: [5a6d9f3 → v1.0.0](https://github.com/containerd/go-runc/compare/5a6d9f3...v1.0.0)
- github.com/containerd/typeurl: [v1.0.1 → v1.0.2](https://github.com/containerd/typeurl/compare/v1.0.1...v1.0.2)
- github.com/go-logr/logr: [v0.4.0 → v1.1.0](https://github.com/go-logr/logr/compare/v0.4.0...v1.1.0)
- github.com/magiconair/properties: [v1.8.1 → v1.8.5](https://github.com/magiconair/properties/compare/v1.8.1...v1.8.5)
- github.com/mitchellh/go-homedir: [v1.1.0 → v1.0.0](https://github.com/mitchellh/go-homedir/compare/v1.1.0...v1.0.0)
- github.com/mitchellh/mapstructure: [v1.1.2 → v1.4.1](https://github.com/mitchellh/mapstructure/compare/v1.1.2...v1.4.1)
- github.com/opencontainers/runc: [v1.0.1 → v1.0.2](https://github.com/opencontainers/runc/compare/v1.0.1...v1.0.2)
- github.com/pelletier/go-toml: [v1.2.0 → v1.9.3](https://github.com/pelletier/go-toml/compare/v1.2.0...v1.9.3)
- github.com/spf13/afero: [v1.2.2 → v1.6.0](https://github.com/spf13/afero/compare/v1.2.2...v1.6.0)
- github.com/spf13/cast: [v1.3.0 → v1.3.1](https://github.com/spf13/cast/compare/v1.3.0...v1.3.1)
- github.com/spf13/cobra: [v1.1.3 → v1.2.1](https://github.com/spf13/cobra/compare/v1.1.3...v1.2.1)
- github.com/spf13/jwalterweatherman: [v1.0.0 → v1.1.0](https://github.com/spf13/jwalterweatherman/compare/v1.0.0...v1.1.0)
- github.com/spf13/viper: [v1.7.0 → v1.8.1](https://github.com/spf13/viper/compare/v1.7.0...v1.8.1)
- github.com/yuin/goldmark: [v1.3.5 → v1.4.0](https://github.com/yuin/goldmark/compare/v1.3.5...v1.4.0)
- go.uber.org/zap: v1.17.0 → v1.19.0
- golang.org/x/crypto: 5ea612d → 32db794
- golang.org/x/net: abc4532 → 60bc85c
- golang.org/x/oauth2: f6687ab → 2bc19b1
- golang.org/x/sys: 59db8d7 → 41cdb87
- golang.org/x/term: 6a3ed07 → 6886f2d
- golang.org/x/tools: v0.1.2 → d4cc65f
- gopkg.in/ini.v1: v1.51.0 → v1.62.0
- k8s.io/klog/v2: v2.9.0 → v2.20.0
- k8s.io/utils: efc7438 → bdf08cb
- sigs.k8s.io/apiserver-network-proxy/konnectivity-client: v0.0.22 → v0.0.23

### Removed
- github.com/coreos/bbolt: [v1.3.2](https://github.com/coreos/bbolt/tree/v1.3.2)
- github.com/coreos/etcd: [v3.3.13+incompatible](https://github.com/coreos/etcd/tree/v3.3.13)
- github.com/coreos/go-systemd: [95778df](https://github.com/coreos/go-systemd/tree/95778df)
- github.com/coreos/pkg: [399ea9e](https://github.com/coreos/pkg/tree/399ea9e)
- github.com/dgrijalva/jwt-go: [v3.2.0+incompatible](https://github.com/dgrijalva/jwt-go/tree/v3.2.0)
- gotest.tools: v2.2.0+incompatible



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