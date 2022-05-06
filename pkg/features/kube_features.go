/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package features

import (
	"k8s.io/apimachinery/pkg/util/runtime"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

// Every feature gate should add method here following this template:
//
// // owner: @username
// // kep: http://kep.k8s.io/NNN
// // alpha: v1.X
// MyFeature featuregate.Feature = "MyFeature"
//
// Feature gates should be listed in alphabetical, case-sensitive
// (upper before any lower case character) order. This reduces the risk
// of code conflicts because changes are more likely to be scattered
// across the file.

// owner: @bswartz
// alpha: v1.18
// beta: v1.24
//
// Enables usage of any object for volume data source in PVCs
const AnyVolumeDataSource featuregate.Feature = "AnyVolumeDataSource"

var _ = add(AnyVolumeDataSource, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta}) // on by default in 1.24

// owner: @tallclair
// beta: v1.4
const AppArmor featuregate.Feature = "AppArmor"

var _ = add(AppArmor, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @szuecs
// alpha: v1.12
//
// Enable nodes to change CPUCFSQuotaPeriod
const CPUCFSQuotaPeriod featuregate.Feature = "CustomCPUCFSQuotaPeriod"

var _ = add(CPUCFSQuotaPeriod, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @ConnorDoyle
// alpha: v1.8
// beta: v1.10
//
// Alternative container-level CPU affinity policies.
const CPUManager featuregate.Feature = "CPUManager"

var _ = add(CPUManager, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @fromanirh
// alpha: v1.23
// beta: see below.
//
// Allow fine-tuning of cpumanager policies, experimental, alpha-quality options
// Per https://groups.google.com/g/kubernetes-sig-architecture/c/Nxsc7pfe5rw/m/vF2djJh0BAAJ
// We want to avoid a proliferation of feature gates. This feature gate:
// - will guard *a group* of cpumanager options whose quality level is alpha.
// - will never graduate to beta or stable.
// See https://groups.google.com/g/kubernetes-sig-architecture/c/Nxsc7pfe5rw/m/vF2djJh0BAAJ
// for details about the removal of this feature gate.
const CPUManagerPolicyAlphaOptions featuregate.Feature = "CPUManagerPolicyAlphaOptions"

var _ = add(CPUManagerPolicyAlphaOptions, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @fromanirh
// beta: v1.23
// beta: see below.
//
// Allow fine-tuning of cpumanager policies, experimental, beta-quality options
// Per https://groups.google.com/g/kubernetes-sig-architecture/c/Nxsc7pfe5rw/m/vF2djJh0BAAJ
// We want to avoid a proliferation of feature gates. This feature gate:
// - will guard *a group* of cpumanager options whose quality level is beta.
// - is thus *introduced* as beta
// - will never graduate to stable.
// See https://groups.google.com/g/kubernetes-sig-architecture/c/Nxsc7pfe5rw/m/vF2djJh0BAAJ
// for details about the removal of this feature gate.
const CPUManagerPolicyBetaOptions featuregate.Feature = "CPUManagerPolicyBetaOptions"

var _ = add(CPUManagerPolicyBetaOptions, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @fromanirh
// alpha: v1.22
// beta: v1.23
//
// Allow the usage of options to fine-tune the cpumanager policies.
const CPUManagerPolicyOptions featuregate.Feature = "CPUManagerPolicyOptions"

var _ = add(CPUManagerPolicyOptions, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @pohly
// alpha: v1.14
// beta: v1.16
//
// Enables CSI Inline volumes support for pods
const CSIInlineVolume featuregate.Feature = "CSIInlineVolume"

var _ = add(CSIInlineVolume, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @davidz627
// alpha: v1.14
// beta: v1.17
//
// Enables the in-tree storage to CSI Plugin migration feature.
const CSIMigration featuregate.Feature = "CSIMigration"

var _ = add(CSIMigration, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @leakingtapan
// alpha: v1.14
// beta: v1.17
//
// Enables the AWS EBS in-tree driver to AWS EBS CSI Driver migration feature.
const CSIMigrationAWS featuregate.Feature = "CSIMigrationAWS"

var _ = add(CSIMigrationAWS, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @andyzhangx
// alpha: v1.15
// beta: v1.19
// GA: v1.24
//
// Enables the Azure Disk in-tree driver to Azure Disk Driver migration feature.
const CSIMigrationAzureDisk featuregate.Feature = "CSIMigrationAzureDisk"

var _ = add(CSIMigrationAzureDisk, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA}) // On by default in 1.23 (requires Azure Disk CSI driver)

// owner: @andyzhangx
// alpha: v1.15
// beta: v1.21
//
// Enables the Azure File in-tree driver to Azure File Driver migration feature.
const CSIMigrationAzureFile featuregate.Feature = "CSIMigrationAzureFile"

var _ = add(CSIMigrationAzureFile, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta}) // On by default in 1.24 (requires Azure File CSI driver)

// owner: @davidz627
// alpha: v1.14
// beta: v1.17
//
// Enables the GCE PD in-tree driver to GCE CSI Driver migration feature.
const CSIMigrationGCE featuregate.Feature = "CSIMigrationGCE"

var _ = add(CSIMigrationGCE, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta}) // On by default in 1.23 (requires GCE PD CSI Driver)

// owner: @adisky
// alpha: v1.14
// beta: v1.18
//
// Enables the OpenStack Cinder in-tree driver to OpenStack Cinder CSI Driver migration feature.
const CSIMigrationOpenStack featuregate.Feature = "CSIMigrationOpenStack"

var _ = add(CSIMigrationOpenStack, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @trierra
// alpha: v1.23
//
// Enables the Portworx in-tree driver to Portworx migration feature.
const CSIMigrationPortworx featuregate.Feature = "CSIMigrationPortworx"

var _ = add(CSIMigrationPortworx, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha}) // Off by default (requires Portworx CSI driver)

// owner: @humblec
// alpha: v1.23
//
// Enables the RBD in-tree driver to RBD CSI Driver  migration feature.
const CSIMigrationRBD featuregate.Feature = "CSIMigrationRBD"

var _ = add(CSIMigrationRBD, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha}) // Off by default (requires RBD CSI driver)

// owner: @divyenpatel
// beta: v1.19 (requires: vSphere vCenter/ESXi Version: 7.0u2, HW Version: VM version 15)
//
// Enables the vSphere in-tree driver to vSphere CSI Driver migration feature.
const CSIMigrationvSphere featuregate.Feature = "CSIMigrationvSphere"

var _ = add(CSIMigrationvSphere, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta}) // Off by default (requires vSphere CSI driver)

// owner: @humblec, @zhucan
// kep: http://kep.k8s.io/3171
// alpha: v1.24
//
// Enables SecretRef field in CSI NodeExpandVolume request.
const CSINodeExpandSecret featuregate.Feature = "CSINodeExpandSecret"

var _ = add(CSINodeExpandSecret, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @pohly
// alpha: v1.19
// beta: v1.21
// GA: v1.24
//
// Enables tracking of available storage capacity that CSI drivers provide.
const CSIStorageCapacity featuregate.Feature = "CSIStorageCapacity"

var _ = add(CSIStorageCapacity, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @fengzixu
// alpha: v1.21
//
// Enables kubelet to detect CSI volume condition and send the event of the abnormal volume to the corresponding pod that is using it.
const CSIVolumeHealth featuregate.Feature = "CSIVolumeHealth"

var _ = add(CSIVolumeHealth, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @enj
// beta: v1.22
// ga: v1.24
//
// Allows clients to request a duration for certificates issued via the Kubernetes CSR API.
const CSRDuration featuregate.Feature = "CSRDuration"

var _ = add(CSRDuration, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @jiahuif
// alpha: v1.21
// beta:  v1.22
// GA:    v1.24
//
// Enables Leader Migration for kube-controller-manager and cloud-controller-manager
const ControllerManagerLeaderMigration featuregate.Feature = "ControllerManagerLeaderMigration"

var _ = add(ControllerManagerLeaderMigration, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @deejross
// kep: http://kep.k8s.io/3140
// alpha: v1.24
//
// Enables support for time zones in CronJobs.
const CronJobTimeZone featuregate.Feature = "CronJobTimeZone"

var _ = add(CronJobTimeZone, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @smarterclayton
// alpha: v1.21
// beta: v1.22
// DaemonSets allow workloads to maintain availability during update per node
const DaemonSetUpdateSurge featuregate.Feature = "DaemonSetUpdateSurge"

var _ = add(DaemonSetUpdateSurge, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta}) // on by default in 1.22

// owner: @alculquicondor
// alpha: v1.19
// beta: v1.20
// GA: v1.24
//
// Enables the use of PodTopologySpread scheduling plugin to do default
// spreading and disables legacy SelectorSpread plugin.
const DefaultPodTopologySpread featuregate.Feature = "DefaultPodTopologySpread"

var _ = add(DefaultPodTopologySpread, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @gnufied, @verult
// alpha: v1.22
// beta: v1.23
// If supported by the CSI driver, delegates the role of applying FSGroup to
// the driver by passing FSGroup through the NodeStageVolume and
// NodePublishVolume calls.
const DelegateFSGroupToCSIDriver featuregate.Feature = "DelegateFSGroupToCSIDriver"

var _ = add(DelegateFSGroupToCSIDriver, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @jiayingz
// beta: v1.10
//
// Enables support for Device Plugins
const DevicePlugins featuregate.Feature = "DevicePlugins"

var _ = add(DevicePlugins, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @RenaudWasTaken @dashpole
// alpha: v1.19
// beta: v1.20
//
// Disables Accelerator Metrics Collected by Kubelet
const DisableAcceleratorUsageMetrics featuregate.Feature = "DisableAcceleratorUsageMetrics"

var _ = add(DisableAcceleratorUsageMetrics, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @andrewsykim
// alpha: v1.22
//
// Disable any functionality in kube-apiserver, kube-controller-manager and kubelet related to the `--cloud-provider` component flag.
const DisableCloudProviders featuregate.Feature = "DisableCloudProviders"

var _ = add(DisableCloudProviders, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @andrewsykim
// alpha: v1.23
//
// Disable in-tree functionality in kubelet to authenticate to cloud provider container registries for image pull credentials.
const DisableKubeletCloudCredentialProviders featuregate.Feature = "DisableKubeletCloudCredentialProviders"

var _ = add(DisableKubeletCloudCredentialProviders, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @derekwaynecarr
// alpha: v1.20
// beta: v1.21 (off by default until 1.22)
//
// Enables usage of hugepages-<size> in downward API.
const DownwardAPIHugePages featuregate.Feature = "DownwardAPIHugePages"

var _ = add(DownwardAPIHugePages, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta}) // on by default in 1.22

// owner: @mtaufen
// alpha: v1.4
// beta: v1.11
// deprecated: 1.22
const DynamicKubeletConfig featuregate.Feature = "DynamicKubeletConfig"

var _ = add(DynamicKubeletConfig, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Deprecated}) // feature gate is deprecated in 1.22, kubelet logic is removed in 1.24, api server logic can be removed in 1.26

// owner: @andrewsykim
// kep: http://kep.k8s.io/1672
// alpha: v1.20
// beta: v1.22
//
// Enable Terminating condition in Endpoint Slices.
const EndpointSliceTerminatingCondition featuregate.Feature = "EndpointSliceTerminatingCondition"

var _ = add(EndpointSliceTerminatingCondition, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @verb
// alpha: v1.16
// beta: v1.23
//
// Allows running an ephemeral container in pod namespaces to troubleshoot a running pod.
const EphemeralContainers featuregate.Feature = "EphemeralContainers"

var _ = add(EphemeralContainers, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @andrewsykim @SergeyKanzhelev
// GA: v1.20
//
// Ensure kubelet respects exec probe timeouts. Feature gate exists in-case existing workloads
// may depend on old behavior where exec probe timeouts were ignored.
// Lock to default and remove after v1.22 based on user feedback that should be reflected in KEP #1972 update
const ExecProbeTimeout featuregate.Feature = "ExecProbeTimeout"

var _ = add(ExecProbeTimeout, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA}) // lock to default and remove after v1.22 based on KEP #1972 update

// owner: @gnufied
// alpha: v1.14
// beta: v1.16
// GA: 1.24
// Ability to expand CSI volumes
const ExpandCSIVolumes featuregate.Feature = "ExpandCSIVolumes"

var _ = add(ExpandCSIVolumes, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA}) // remove in 1.26

// owner: @mlmhl @gnufied
// beta: v1.15
// GA: 1.24
// Ability to expand persistent volumes' file system without unmounting volumes.
const ExpandInUsePersistentVolumes featuregate.Feature = "ExpandInUsePersistentVolumes"

var _ = add(ExpandInUsePersistentVolumes, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA}) // remove in 1.26

// owner: @gnufied
// beta: v1.11
// GA: 1.24
// Ability to Expand persistent volumes
const ExpandPersistentVolumes featuregate.Feature = "ExpandPersistentVolumes"

var _ = add(ExpandPersistentVolumes, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA}) // remove in 1.26

// owner: @gjkim42
// kep: http://kep.k8s.io/2595
// alpha: v1.22
//
// Enables apiserver and kubelet to allow up to 32 DNSSearchPaths and up to 2048 DNSSearchListChars.
const ExpandedDNSConfig featuregate.Feature = "ExpandedDNSConfig"

var _ = add(ExpandedDNSConfig, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @pweil-
// alpha: v1.5
//
// Default userns=host for containers that are using other host namespaces, host mounts, the pod
// contains a privileged container, or specific non-namespaced capabilities (MKNOD, SYS_MODULE,
// SYS_TIME). This should only be enabled if user namespace remapping is enabled in the docker daemon.
const ExperimentalHostUserNamespaceDefaultingGate featuregate.Feature = "ExperimentalHostUserNamespaceDefaulting"

// owner: @yuzhiquan, @bowei, @PxyUp, @SergeyKanzhelev
// kep: http://kep.k8s.io/2727
// alpha: v1.23
// beta: v1.24
//
// Enables GRPC probe method for {Liveness,Readiness,Startup}Probe.
const GRPCContainerProbe featuregate.Feature = "GRPCContainerProbe"

var _ = add(ExperimentalHostUserNamespaceDefaultingGate, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta})

// owner: @bobbypage
// alpha: v1.20
// beta:  v1.21
// Adds support for kubelet to detect node shutdown and gracefully terminate pods prior to the node being shutdown.
const GracefulNodeShutdown featuregate.Feature = "GracefulNodeShutdown"

var _ = add(GRPCContainerProbe, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @wzshiming
// alpha: v1.23
// beta:  v1.24
// Make the kubelet use shutdown configuration based on pod priority values for graceful shutdown.
const GracefulNodeShutdownBasedOnPodPriority featuregate.Feature = "GracefulNodeShutdownBasedOnPodPriority"

var _ = add(GracefulNodeShutdown, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @arjunrn @mwielgus @josephburnett
// alpha: v1.20
//
// Add support for the HPA to scale based on metrics from individual containers
// in target pods
const HPAContainerMetrics featuregate.Feature = "HPAContainerMetrics"

var _ = add(GracefulNodeShutdownBasedOnPodPriority, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @dxist
// alpha: v1.16
//
// Enables support of HPA scaling to zero pods when an object or custom metric is configured.
const HPAScaleToZero featuregate.Feature = "HPAScaleToZero"

var _ = add(HPAContainerMetrics, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @deepakkinni @xing-yang
// kep: http://kep.k8s.io/2680
// alpha: v1.23
//
// Honor Persistent Volume Reclaim Policy when it is "Delete" irrespective of PV-PVC
// deletion ordering.
const HonorPVReclaimPolicy featuregate.Feature = "HonorPVReclaimPolicy"

var _ = add(HonorPVReclaimPolicy, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @ravig
// alpha: v1.23
// beta: v1.24
// IdentifyPodOS allows user to specify OS on which they'd like the Pod run. The user should still set the nodeSelector
// with appropriate `kubernetes.io/os` label for scheduler to identify appropriate node for the pod to run.
const IdentifyPodOS featuregate.Feature = "IdentifyPodOS"

var _ = add(IdentifyPodOS, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @leakingtapan
// alpha: v1.21
//
// Disables the AWS EBS in-tree driver.
const InTreePluginAWSUnregister featuregate.Feature = "InTreePluginAWSUnregister"

var _ = add(InTreePluginAWSUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @andyzhangx
// alpha: v1.21
//
// Disables the Azure Disk in-tree driver.
const InTreePluginAzureDiskUnregister featuregate.Feature = "InTreePluginAzureDiskUnregister"

var _ = add(InTreePluginAzureDiskUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @andyzhangx
// alpha: v1.21
//
// Disables the Azure File in-tree driver.
const InTreePluginAzureFileUnregister featuregate.Feature = "InTreePluginAzureFileUnregister"

var _ = add(InTreePluginAzureFileUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @Jiawei0227
// alpha: v1.21
//
// Disables the GCE PD in-tree driver.
const InTreePluginGCEUnregister featuregate.Feature = "InTreePluginGCEUnregister"

var _ = add(InTreePluginGCEUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @adisky
// alpha: v1.21
//
// Disables the OpenStack Cinder in-tree driver.
const InTreePluginOpenStackUnregister featuregate.Feature = "InTreePluginOpenStackUnregister"

var _ = add(InTreePluginOpenStackUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @trierra
// alpha: v1.23
//
// Disables the Portworx in-tree driver.
const InTreePluginPortworxUnregister featuregate.Feature = "InTreePluginPortworxUnregister"

var _ = add(InTreePluginPortworxUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @humblec
// alpha: v1.23
//
// Disables the RBD in-tree driver.
const InTreePluginRBDUnregister featuregate.Feature = "InTreePluginRBDUnregister"

var _ = add(InTreePluginRBDUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @divyenpatel
// alpha: v1.21
//
// Disables the vSphere in-tree driver.
const InTreePluginvSphereUnregister featuregate.Feature = "InTreePluginvSphereUnregister"

var _ = add(InTreePluginvSphereUnregister, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @alculquicondor
// alpha: v1.21
// beta: v1.22
// stable: v1.24
//
// Allows Job controller to manage Pod completions per completion index.
const IndexedJob featuregate.Feature = "IndexedJob"

var _ = add(IndexedJob, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @ahg
// beta: v1.23
//
// Allow updating node scheduling directives in the pod template of jobs. Specifically,
// node affinity, selector and tolerations. This is allowed only for suspended jobs
// that have never been unsuspended before.
const JobMutableNodeSchedulingDirectives featuregate.Feature = "JobMutableNodeSchedulingDirectives"

var _ = add(JobMutableNodeSchedulingDirectives, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @alculquicondor
// alpha: v1.23
// beta: v1.24
//
// Track the number of pods with Ready condition in the Job status.
const JobReadyPods featuregate.Feature = "JobReadyPods"

var _ = add(JobReadyPods, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @alculquicondor
// alpha: v1.22
// beta: v1.23
//
// Track Job completion without relying on Pod remaining in the cluster
// indefinitely. Pod finalizers, in addition to a field in the Job status
// allow the Job controller to keep track of Pods that it didn't account for
// yet.
const JobTrackingWithFinalizers featuregate.Feature = "JobTrackingWithFinalizers"

var _ = add(JobTrackingWithFinalizers, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta}) // Disabled due to #109485

// owner: @andrewsykim @adisky
// alpha: v1.20
// beta: v1.24
//
// Enable kubelet exec plugins for image pull credentials.
const KubeletCredentialProviders featuregate.Feature = "KubeletCredentialProviders"

var _ = add(KubeletCredentialProviders, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @AkihiroSuda
// alpha: v1.22
//
// Enables support for running kubelet in a user namespace.
// The user namespace has to be created before running kubelet.
// All the node components such as CRI need to be running in the same user namespace.
const KubeletInUserNamespace featuregate.Feature = "KubeletInUserNamespace"

var _ = add(KubeletInUserNamespace, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @dashpole
// alpha: v1.13
// beta: v1.15
//
// Enables the kubelet's pod resources grpc endpoint
const KubeletPodResources featuregate.Feature = "KubeletPodResources"

var _ = add(KubeletPodResources, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @fromanirh
// alpha: v1.21
// beta: v1.23
// Enable POD resources API to return allocatable resources
const KubeletPodResourcesGetAllocatable featuregate.Feature = "KubeletPodResourcesGetAllocatable"

var _ = add(KubeletPodResourcesGetAllocatable, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @zshihang
// kep: http://kep.k8s.io/2800
// beta: v1.24
//
// Stop auto-generation of secret-based service account tokens.
const LegacyServiceAccountTokenNoAutoGeneration featuregate.Feature = "LegacyServiceAccountTokenNoAutoGeneration"

var _ = add(LegacyServiceAccountTokenNoAutoGeneration, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @jinxu
// beta: v1.10
//
// New local storage types to support local storage capacity isolation
const LocalStorageCapacityIsolation featuregate.Feature = "LocalStorageCapacityIsolation"

var _ = add(LocalStorageCapacityIsolation, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @RobertKrawitz
// alpha: v1.15
//
// Allow use of filesystems for ephemeral storage monitoring.
// Only applies if LocalStorageCapacityIsolation is set.
const LocalStorageCapacityIsolationFSQuotaMonitoring featuregate.Feature = "LocalStorageCapacityIsolationFSQuotaMonitoring"

var _ = add(LocalStorageCapacityIsolationFSQuotaMonitoring, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @damemi
// alpha: v1.21
// beta: v1.22
//
// Enables scaling down replicas via logarithmic comparison of creation/ready timestamps
const LogarithmicScaleDown featuregate.Feature = "LogarithmicScaleDown"

var _ = add(LogarithmicScaleDown, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @krmayankk
// alpha: v1.24
//
// Enables maxUnavailable for StatefulSet
const MaxUnavailableStatefulSet featuregate.Feature = "MaxUnavailableStatefulSet"

var _ = add(MaxUnavailableStatefulSet, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @cynepco3hahue(alukiano) @cezaryzukowski @k-wiatrzyk
// alpha: v1.21
// beta: v1.22
// Allows setting memory affinity for a container based on NUMA topology
const MemoryManager featuregate.Feature = "MemoryManager"

var _ = add(MemoryManager, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @xiaoxubeii
// kep: http://kep.k8s.io/2570
// alpha: v1.22
//
// Enables kubelet to support memory QoS with cgroups v2.
const MemoryQoS featuregate.Feature = "MemoryQoS"

var _ = add(MemoryQoS, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @sanposhiho
// kep: http://kep.k8s.io/3022
// alpha: v1.24
//
// Enable MinDomains in Pod Topology Spread.
const MinDomainsInPodTopologySpread featuregate.Feature = "MinDomainsInPodTopologySpread"

var _ = add(MinDomainsInPodTopologySpread, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @janosi @bridgetkromhout
// kep: http://kep.k8s.io/1435
// alpha: v1.20
// beta: v1.24
//
// Enables the usage of different protocols in the same Service with type=LoadBalancer
const MixedProtocolLBService featuregate.Feature = "MixedProtocolLBService"

var _ = add(MixedProtocolLBService, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @rikatz
// kep: http://kep.k8s.io/2079
// alpha: v1.21
// beta:  v1.22
//
// Enables the endPort field in NetworkPolicy to enable a Port Range behavior in Network Policies.
const NetworkPolicyEndPort featuregate.Feature = "NetworkPolicyEndPort"

var _ = add(NetworkPolicyEndPort, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @rikatz
// kep: http://kep.k8s.io/2943
// alpha: v1.24
//
// Enables NetworkPolicy status subresource
const NetworkPolicyStatus featuregate.Feature = "NetworkPolicyStatus"

var _ = add(NetworkPolicyStatus, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @kerthcet
// kep: http://kep.k8s.io/3094
// alpha: v1.25
//
// Allow users to specify whether to take nodeAffinity/nodeTaint into consideration when
// calculating pod topology spread skew.
const NodeInclusionPolicyInPodTopologySpread featuregate.Feature = "NodeInclusionPolicyInPodTopologySpread"

var _ = add(NodeInclusionPolicyInPodTopologySpread, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @xing-yang @sonasingh46
// kep: http://kep.k8s.io/2268
// alpha: v1.24
//
// Allow pods to failover to a different node in case of non graceful node shutdown
const NodeOutOfServiceVolumeDetach featuregate.Feature = "NodeOutOfServiceVolumeDetach"

var _ = add(NodeOutOfServiceVolumeDetach, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @ehashman
// alpha: v1.22
//
// Permits kubelet to run with swap enabled
const NodeSwap featuregate.Feature = "NodeSwap"

var _ = add(NodeSwap, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @denkensk
// alpha: v1.15
// beta: v1.19
// ga: v1.24
//
// Enables NonPreempting option for priorityClass and pod.
const NonPreemptingPriority featuregate.Feature = "NonPreemptingPriority"

var _ = add(NonPreemptingPriority, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @ahg-g
// alpha: v1.21
// beta: v1.22
// GA: v1.24
//
// Allow specifying NamespaceSelector in PodAffinityTerm.
const PodAffinityNamespaceSelector featuregate.Feature = "PodAffinityNamespaceSelector"

var _ = add(PodAffinityNamespaceSelector, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @haircommander
// kep: http://kep.k8s.io/2364
// alpha: v1.23
//
// Configures the Kubelet to use the CRI to populate pod and container stats, instead of supplimenting with stats from cAdvisor.
// Requires the CRI implementation supports supplying the required stats.
const PodAndContainerStatsFromCRI featuregate.Feature = "PodAndContainerStatsFromCRI"

var _ = add(PodAndContainerStatsFromCRI, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @ahg-g
// alpha: v1.21
// beta: v1.22
//
// Enables controlling pod ranking on replicaset scale-down.
const PodDeletionCost featuregate.Feature = "PodDeletionCost"

var _ = add(PodDeletionCost, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @egernst
// alpha: v1.16
// beta: v1.18
// ga: v1.24
//
// Enables PodOverhead, for accounting pod overheads which are specific to a given RuntimeClass
const PodOverhead featuregate.Feature = "PodOverhead"

var _ = add(PodOverhead, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @liggitt, @tallclair, sig-auth
// alpha: v1.22
// beta: v1.23
//
// Enables the PodSecurity admission plugin
const PodSecurity featuregate.Feature = "PodSecurity"

var _ = add(PodSecurity, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @chendave
// alpha: v1.21
// beta: v1.22
// GA: v1.24
//
// PreferNominatedNode tells scheduler whether the nominated node will be checked first before looping
// all the rest of nodes in the cluster.
// Enabling this feature also implies the preemptor pod might not be dispatched to the best candidate in
// some corner case, e.g. another node releases enough resources after the nominated node has been set
// and hence is the best candidate instead.
const PreferNominatedNode featuregate.Feature = "PreferNominatedNode"

var _ = add(PreferNominatedNode, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @ehashman
// alpha: v1.21
//
// Allows user to override pod-level terminationGracePeriod for probes
const ProbeTerminationGracePeriod featuregate.Feature = "ProbeTerminationGracePeriod"

var _ = add(ProbeTerminationGracePeriod, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta}) // Default to false in beta 1.22, set to true in 1.24

// owner: @jessfraz
// alpha: v1.12
//
// Enables control over ProcMountType for containers.
const ProcMountType featuregate.Feature = "ProcMountType"

var _ = add(ProcMountType, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @andrewsykim
// kep: http://kep.k8s.io/1669
// alpha: v1.22
//
// Enable kube-proxy to handle terminating ednpoints when externalTrafficPolicy=Local
const ProxyTerminatingEndpoints featuregate.Feature = "ProxyTerminatingEndpoints"

var _ = add(ProxyTerminatingEndpoints, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @sjenning
// alpha: v1.11
//
// Allows resource reservations at the QoS level preventing pods at lower QoS levels from
// bursting into resources requested at higher QoS levels (memory only for now)
const QOSReserved featuregate.Feature = "QOSReserved"

var _ = add(QOSReserved, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @chrishenzie
// alpha: v1.22
//
// Enables usage of the ReadWriteOncePod PersistentVolume access mode.
const ReadWriteOncePod featuregate.Feature = "ReadWriteOncePod"

var _ = add(ReadWriteOncePod, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @gnufied
// kep: http://kep.k8s.io/1790
// alpha: v1.23
//
// Allow users to recover from volume expansion failure
const RecoverVolumeExpansionFailure featuregate.Feature = "RecoverVolumeExpansionFailure"

var _ = add(RecoverVolumeExpansionFailure, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @mikedanese
// alpha: v1.7
// beta: v1.12
//
// Gets a server certificate for the kubelet from the Certificate Signing
// Request API instead of generating one self signed and auto rotates the
// certificate as expiration approaches.
const RotateKubeletServerCertificate featuregate.Feature = "RotateKubeletServerCertificate"

var _ = add(RotateKubeletServerCertificate, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @saschagrunert
// alpha: v1.22
//
// Enables the use of `RuntimeDefault` as the default seccomp profile for all workloads.
const SeccompDefault featuregate.Feature = "SeccompDefault"

var _ = add(SeccompDefault, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @maplain @andrewsykim
// kep: http://kep.k8s.io/2086
// alpha: v1.21
// beta: v1.22
//
// Enables node-local routing for Service internal traffic
const ServiceInternalTrafficPolicy featuregate.Feature = "ServiceInternalTrafficPolicy"

var _ = add(ServiceIPStaticSubrange, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @aojea
// kep: http://kep.k8s.io/3070
// alpha: v1.24
//
// Subdivide the ClusterIP range for dynamic and static IP allocation.
const ServiceIPStaticSubrange featuregate.Feature = "ServiceIPStaticSubrange"

var _ = add(ServiceInternalTrafficPolicy, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @andrewsykim @uablrek
// kep: http://kep.k8s.io/1864
// alpha: v1.20
// beta: v1.22
// ga: v1.24
//
// Allows control if NodePorts shall be created for services with "type: LoadBalancer" by defining the spec.AllocateLoadBalancerNodePorts field (bool)
const ServiceLBNodePortControl featuregate.Feature = "ServiceLBNodePortControl"

var _ = add(ServiceLBNodePortControl, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @andrewsykim @XudongLiuHarold
// kep: http://kep.k8s.io/1959
// alpha: v1.21
// beta: v1.22
// GA: v1.24
//
// Enable support multiple Service "type: LoadBalancer" implementations in a cluster by specifying LoadBalancerClass
const ServiceLoadBalancerClass featuregate.Feature = "ServiceLoadBalancerClass"

var _ = add(ServiceLoadBalancerClass, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @derekwaynecarr
// alpha: v1.20
// beta: v1.22
//
// Enables kubelet support to size memory backed volumes
const SizeMemoryBackedVolumes featuregate.Feature = "SizeMemoryBackedVolumes"

var _ = add(SizeMemoryBackedVolumes, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @mattcary
// alpha: v1.22
//
// Enables policies controlling deletion of PVCs created by a StatefulSet.
const StatefulSetAutoDeletePVC featuregate.Feature = "StatefulSetAutoDeletePVC"

var _ = add(StatefulSetAutoDeletePVC, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @ravig
// kep: https://kep.k8s.io/2607
// alpha: v1.22
// beta: v1.23
// StatefulSetMinReadySeconds allows minReadySeconds to be respected by StatefulSet controller
const StatefulSetMinReadySeconds featuregate.Feature = "StatefulSetMinReadySeconds"

var _ = add(StatefulSetMinReadySeconds, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @adtac
// alpha: v1.21
// beta: v1.22
// GA: v1.24
//
// Allows jobs to be created in the suspended state.
const SuspendJob featuregate.Feature = "SuspendJob"

var _ = add(SuspendJob, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA, LockToDefault: true}) // remove in 1.26

// owner: @robscott
// kep: http://kep.k8s.io/2433
// alpha: v1.21
// beta: v1.23
//
// Enables topology aware hints for EndpointSlices
const TopologyAwareHints featuregate.Feature = "TopologyAwareHints"

var _ = add(TopologyAwareHints, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @lmdaly
// alpha: v1.16
// beta: v1.18
//
// Enable resource managers to make NUMA aligned decisions
const TopologyManager featuregate.Feature = "TopologyManager"

var _ = add(TopologyManager, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @cofyc
// alpha: v1.21
const VolumeCapacityPriority featuregate.Feature = "VolumeCapacityPriority"

var _ = add(VolumeCapacityPriority, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @ksubrmnn
// alpha: v1.14
//
// Allows kube-proxy to create DSR loadbalancers for Windows
const WinDSR featuregate.Feature = "WinDSR"

var _ = add(WinDSR, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

// owner: @ksubrmnn
// alpha: v1.14
// beta: v1.20
//
// Allows kube-proxy to run in Overlay mode for Windows
const WinOverlay featuregate.Feature = "WinOverlay"

var _ = add(WinOverlay, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// owner: @marosset
// alpha: v1.22
// beta: v1.23
//
// Enables support for 'HostProcess' containers on Windows nodes.
const WindowsHostProcessContainers featuregate.Feature = "WindowsHostProcessContainers"

var _ = add(WindowsHostProcessContainers, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})

// Do not add new features at the end unless that is where it belongs due to
// alphabetical sorting. See comment at the top for details.

func init() {
	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	add(genericfeatures.APIListChunking, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})
	add(genericfeatures.APIPriorityAndFairness, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})
	add(genericfeatures.APIResponseCompression, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})
	add(genericfeatures.AdvancedAuditing, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA})
	add(genericfeatures.CustomResourceValidationExpressions, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})
	add(genericfeatures.DryRun, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA})
	add(genericfeatures.OpenAPIEnums, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})
	add(genericfeatures.OpenAPIV3, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta})
	add(genericfeatures.ServerSideApply, featuregate.FeatureSpec{Default: true, PreRelease: featuregate.GA})
	add(genericfeatures.ServerSideFieldValidation, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
	add(HPAScaleToZero, featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Alpha})

	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
//
// Entries are separated from each other with blank lines to avoid sweeping gofmt changes
// when adding or removing one entry.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{}

func add(feature featuregate.Feature, spec featuregate.FeatureSpec) bool {
	defaultKubernetesFeatureGates[feature] = spec
	return true
}
