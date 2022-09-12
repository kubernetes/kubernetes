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

const (
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
	AnyVolumeDataSource featuregate.Feature = "AnyVolumeDataSource"

	// owner: @tallclair
	// beta: v1.4
	AppArmor featuregate.Feature = "AppArmor"

	// owner: @szuecs
	// alpha: v1.12
	//
	// Enable nodes to change CPUCFSQuotaPeriod
	CPUCFSQuotaPeriod featuregate.Feature = "CustomCPUCFSQuotaPeriod"

	// owner: @ConnorDoyle
	// alpha: v1.8
	// beta: v1.10
	//
	// Alternative container-level CPU affinity policies.
	CPUManager featuregate.Feature = "CPUManager"

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
	CPUManagerPolicyAlphaOptions featuregate.Feature = "CPUManagerPolicyAlphaOptions"

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
	CPUManagerPolicyBetaOptions featuregate.Feature = "CPUManagerPolicyBetaOptions"

	// owner: @fromanirh
	// alpha: v1.22
	// beta: v1.23
	//
	// Allow the usage of options to fine-tune the cpumanager policies.
	CPUManagerPolicyOptions featuregate.Feature = "CPUManagerPolicyOptions"

	// owner: @pohly
	// alpha: v1.14
	// beta: v1.16
	// GA: v1.25
	//
	// Enables CSI Inline volumes support for pods
	CSIInlineVolume featuregate.Feature = "CSIInlineVolume"

	// owner: @davidz627
	// alpha: v1.14
	// beta: v1.17
	//
	// Enables the in-tree storage to CSI Plugin migration feature.
	CSIMigration featuregate.Feature = "CSIMigration"

	// owner: @leakingtapan
	// alpha: v1.14
	// beta: v1.17
	// GA: v1.25
	//
	// Enables the AWS EBS in-tree driver to AWS EBS CSI Driver migration feature.
	CSIMigrationAWS featuregate.Feature = "CSIMigrationAWS"

	// owner: @andyzhangx
	// alpha: v1.15
	// beta: v1.19
	// GA: v1.24
	//
	// Enables the Azure Disk in-tree driver to Azure Disk Driver migration feature.
	CSIMigrationAzureDisk featuregate.Feature = "CSIMigrationAzureDisk"

	// owner: @andyzhangx
	// alpha: v1.15
	// beta: v1.21
	//
	// Enables the Azure File in-tree driver to Azure File Driver migration feature.
	CSIMigrationAzureFile featuregate.Feature = "CSIMigrationAzureFile"

	// owner: @davidz627
	// alpha: v1.14
	// beta: v1.17
	// GA: 1.25
	//
	// Enables the GCE PD in-tree driver to GCE CSI Driver migration feature.
	CSIMigrationGCE featuregate.Feature = "CSIMigrationGCE"

	// owner: @trierra
	// alpha: v1.23
	//
	// Enables the Portworx in-tree driver to Portworx migration feature.
	CSIMigrationPortworx featuregate.Feature = "CSIMigrationPortworx"

	// owner: @humblec
	// alpha: v1.23
	//
	// Enables the RBD in-tree driver to RBD CSI Driver  migration feature.
	CSIMigrationRBD featuregate.Feature = "CSIMigrationRBD"

	// owner: @divyenpatel
	// beta: v1.19 (requires: vSphere vCenter/ESXi Version: 7.0u2, HW Version: VM version 15)
	//
	// Enables the vSphere in-tree driver to vSphere CSI Driver migration feature.
	CSIMigrationvSphere featuregate.Feature = "CSIMigrationvSphere"

	// owner: @humblec, @zhucan
	// kep: http://kep.k8s.io/3171
	// alpha: v1.25
	//
	// Enables SecretRef field in CSI NodeExpandVolume request.
	CSINodeExpandSecret featuregate.Feature = "CSINodeExpandSecret"

	// owner: @pohly
	// alpha: v1.19
	// beta: v1.21
	// GA: v1.24
	//
	// Enables tracking of available storage capacity that CSI drivers provide.
	CSIStorageCapacity featuregate.Feature = "CSIStorageCapacity"

	// owner: @fengzixu
	// alpha: v1.21
	//
	// Enables kubelet to detect CSI volume condition and send the event of the abnormal volume to the corresponding pod that is using it.
	CSIVolumeHealth featuregate.Feature = "CSIVolumeHealth"

	// owner: @adrianreber
	// kep: http://kep.k8s.io/2008
	// alpha: v1.25
	//
	// Enables container Checkpoint support in the kubelet
	ContainerCheckpoint featuregate.Feature = "ContainerCheckpoint"

	// owner: @jiahuif
	// alpha: v1.21
	// beta:  v1.22
	// GA:    v1.24
	//
	// Enables Leader Migration for kube-controller-manager and cloud-controller-manager
	ControllerManagerLeaderMigration featuregate.Feature = "ControllerManagerLeaderMigration"

	// owner: @deejross, @soltysh
	// kep: http://kep.k8s.io/3140
	// alpha: v1.24
	// beta: v1.25
	//
	// Enables support for time zones in CronJobs.
	CronJobTimeZone featuregate.Feature = "CronJobTimeZone"

	// owner: @smarterclayton
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.25
	// DaemonSets allow workloads to maintain availability during update per node
	DaemonSetUpdateSurge featuregate.Feature = "DaemonSetUpdateSurge"

	// owner: @alculquicondor
	// alpha: v1.19
	// beta: v1.20
	// GA: v1.24
	//
	// Enables the use of PodTopologySpread scheduling plugin to do default
	// spreading and disables legacy SelectorSpread plugin.
	DefaultPodTopologySpread featuregate.Feature = "DefaultPodTopologySpread"

	// owner: @gnufied, @verult
	// alpha: v1.22
	// beta: v1.23
	// If supported by the CSI driver, delegates the role of applying FSGroup to
	// the driver by passing FSGroup through the NodeStageVolume and
	// NodePublishVolume calls.
	DelegateFSGroupToCSIDriver featuregate.Feature = "DelegateFSGroupToCSIDriver"

	// owner: @jiayingz
	// beta: v1.10
	//
	// Enables support for Device Plugins
	DevicePlugins featuregate.Feature = "DevicePlugins"

	// owner: @RenaudWasTaken @dashpole
	// alpha: v1.19
	// beta: v1.20
	// ga: v1.25
	//
	// Disables Accelerator Metrics Collected by Kubelet
	DisableAcceleratorUsageMetrics featuregate.Feature = "DisableAcceleratorUsageMetrics"

	// owner: @andrewsykim
	// alpha: v1.22
	//
	// Disable any functionality in kube-apiserver, kube-controller-manager and kubelet related to the `--cloud-provider` component flag.
	DisableCloudProviders featuregate.Feature = "DisableCloudProviders"

	// owner: @andrewsykim
	// alpha: v1.23
	//
	// Disable in-tree functionality in kubelet to authenticate to cloud provider container registries for image pull credentials.
	DisableKubeletCloudCredentialProviders featuregate.Feature = "DisableKubeletCloudCredentialProviders"

	// owner: @derekwaynecarr
	// alpha: v1.20
	// beta: v1.21 (off by default until 1.22)
	//
	// Enables usage of hugepages-<size> in downward API.
	DownwardAPIHugePages featuregate.Feature = "DownwardAPIHugePages"

	// owner: @mtaufen
	// alpha: v1.4
	// beta: v1.11
	// deprecated: 1.22
	DynamicKubeletConfig featuregate.Feature = "DynamicKubeletConfig"

	// owner: @andrewsykim
	// kep: http://kep.k8s.io/1672
	// alpha: v1.20
	// beta: v1.22
	//
	// Enable Terminating condition in Endpoint Slices.
	EndpointSliceTerminatingCondition featuregate.Feature = "EndpointSliceTerminatingCondition"

	// owner: @verb
	// alpha: v1.16
	// beta: v1.23
	// GA: v1.25
	//
	// Allows running an ephemeral container in pod namespaces to troubleshoot a running pod.
	EphemeralContainers featuregate.Feature = "EphemeralContainers"

	// owner: @andrewsykim @SergeyKanzhelev
	// GA: v1.20
	//
	// Ensure kubelet respects exec probe timeouts. Feature gate exists in-case existing workloads
	// may depend on old behavior where exec probe timeouts were ignored.
	// Lock to default and remove after v1.22 based on user feedback that should be reflected in KEP #1972 update
	ExecProbeTimeout featuregate.Feature = "ExecProbeTimeout"

	// owner: @gnufied
	// alpha: v1.14
	// beta: v1.16
	// GA: 1.24
	// Ability to expand CSI volumes
	ExpandCSIVolumes featuregate.Feature = "ExpandCSIVolumes"

	// owner: @mlmhl @gnufied
	// beta: v1.15
	// GA: 1.24
	// Ability to expand persistent volumes' file system without unmounting volumes.
	ExpandInUsePersistentVolumes featuregate.Feature = "ExpandInUsePersistentVolumes"

	// owner: @gnufied
	// beta: v1.11
	// GA: 1.24
	// Ability to Expand persistent volumes
	ExpandPersistentVolumes featuregate.Feature = "ExpandPersistentVolumes"

	// owner: @gjkim42
	// kep: http://kep.k8s.io/2595
	// alpha: v1.22
	//
	// Enables apiserver and kubelet to allow up to 32 DNSSearchPaths and up to 2048 DNSSearchListChars.
	ExpandedDNSConfig featuregate.Feature = "ExpandedDNSConfig"

	// owner: @pweil-
	// alpha: v1.5
	//
	// Default userns=host for containers that are using other host namespaces, host mounts, the pod
	// contains a privileged container, or specific non-namespaced capabilities (MKNOD, SYS_MODULE,
	// SYS_TIME). This should only be enabled if user namespace remapping is enabled in the docker daemon.
	ExperimentalHostUserNamespaceDefaultingGate featuregate.Feature = "ExperimentalHostUserNamespaceDefaulting"

	// owner: @yuzhiquan, @bowei, @PxyUp, @SergeyKanzhelev
	// kep: http://kep.k8s.io/2727
	// alpha: v1.23
	// beta: v1.24
	//
	// Enables GRPC probe method for {Liveness,Readiness,Startup}Probe.
	GRPCContainerProbe featuregate.Feature = "GRPCContainerProbe"

	// owner: @bobbypage
	// alpha: v1.20
	// beta:  v1.21
	// Adds support for kubelet to detect node shutdown and gracefully terminate pods prior to the node being shutdown.
	GracefulNodeShutdown featuregate.Feature = "GracefulNodeShutdown"

	// owner: @wzshiming
	// alpha: v1.23
	// beta:  v1.24
	// Make the kubelet use shutdown configuration based on pod priority values for graceful shutdown.
	GracefulNodeShutdownBasedOnPodPriority featuregate.Feature = "GracefulNodeShutdownBasedOnPodPriority"

	// owner: @arjunrn @mwielgus @josephburnett
	// alpha: v1.20
	//
	// Add support for the HPA to scale based on metrics from individual containers
	// in target pods
	HPAContainerMetrics featuregate.Feature = "HPAContainerMetrics"

	// owner: @dxist
	// alpha: v1.16
	//
	// Enables support of HPA scaling to zero pods when an object or custom metric is configured.
	HPAScaleToZero featuregate.Feature = "HPAScaleToZero"

	// owner: @deepakkinni @xing-yang
	// kep: http://kep.k8s.io/2680
	// alpha: v1.23
	//
	// Honor Persistent Volume Reclaim Policy when it is "Delete" irrespective of PV-PVC
	// deletion ordering.
	HonorPVReclaimPolicy featuregate.Feature = "HonorPVReclaimPolicy"

	// owner: @ravig
	// alpha: v1.23
	// beta: v1.24
	// GA: v1.25
	// IdentifyPodOS allows user to specify OS on which they'd like the Pod run. The user should still set the nodeSelector
	// with appropriate `kubernetes.io/os` label for scheduler to identify appropriate node for the pod to run.
	IdentifyPodOS featuregate.Feature = "IdentifyPodOS"

	// owner: @leakingtapan
	// alpha: v1.21
	//
	// Disables the AWS EBS in-tree driver.
	InTreePluginAWSUnregister featuregate.Feature = "InTreePluginAWSUnregister"

	// owner: @andyzhangx
	// alpha: v1.21
	//
	// Disables the Azure Disk in-tree driver.
	InTreePluginAzureDiskUnregister featuregate.Feature = "InTreePluginAzureDiskUnregister"

	// owner: @andyzhangx
	// alpha: v1.21
	//
	// Disables the Azure File in-tree driver.
	InTreePluginAzureFileUnregister featuregate.Feature = "InTreePluginAzureFileUnregister"

	// owner: @Jiawei0227
	// alpha: v1.21
	//
	// Disables the GCE PD in-tree driver.
	InTreePluginGCEUnregister featuregate.Feature = "InTreePluginGCEUnregister"

	// owner: @trierra
	// alpha: v1.23
	//
	// Disables the Portworx in-tree driver.
	InTreePluginPortworxUnregister featuregate.Feature = "InTreePluginPortworxUnregister"

	// owner: @humblec
	// alpha: v1.23
	//
	// Disables the RBD in-tree driver.
	InTreePluginRBDUnregister featuregate.Feature = "InTreePluginRBDUnregister"

	// owner: @divyenpatel
	// alpha: v1.21
	//
	// Disables the vSphere in-tree driver.
	InTreePluginvSphereUnregister featuregate.Feature = "InTreePluginvSphereUnregister"

	// owner: @alculquicondor
	// alpha: v1.21
	// beta: v1.22
	// stable: v1.24
	//
	// Allows Job controller to manage Pod completions per completion index.
	IndexedJob featuregate.Feature = "IndexedJob"

	// owner: @danwinship
	// kep: http://kep.k8s.io/3178
	// alpha: v1.25
	//
	// Causes kubelet to no longer create legacy IPTables rules
	IPTablesOwnershipCleanup featuregate.Feature = "IPTablesOwnershipCleanup"

	// owner: @mimowo
	// kep: http://kep.k8s.io/3329
	// alpha: v1.25
	//
	// Allow users to specify handling of pod failures based on container exit codes
	// and pod conditions.
	JobPodFailurePolicy featuregate.Feature = "JobPodFailurePolicy"

	// owner: @ahg
	// beta: v1.23
	//
	// Allow updating node scheduling directives in the pod template of jobs. Specifically,
	// node affinity, selector and tolerations. This is allowed only for suspended jobs
	// that have never been unsuspended before.
	JobMutableNodeSchedulingDirectives featuregate.Feature = "JobMutableNodeSchedulingDirectives"

	// owner: @alculquicondor
	// alpha: v1.23
	// beta: v1.24
	//
	// Track the number of pods with Ready condition in the Job status.
	JobReadyPods featuregate.Feature = "JobReadyPods"

	// owner: @alculquicondor
	// alpha: v1.22
	// beta: v1.23
	//
	// Track Job completion without relying on Pod remaining in the cluster
	// indefinitely. Pod finalizers, in addition to a field in the Job status
	// allow the Job controller to keep track of Pods that it didn't account for
	// yet.
	JobTrackingWithFinalizers featuregate.Feature = "JobTrackingWithFinalizers"

	// owner: @andrewsykim @adisky
	// alpha: v1.20
	// beta: v1.24
	//
	// Enable kubelet exec plugins for image pull credentials.
	KubeletCredentialProviders featuregate.Feature = "KubeletCredentialProviders"

	// owner: @AkihiroSuda
	// alpha: v1.22
	//
	// Enables support for running kubelet in a user namespace.
	// The user namespace has to be created before running kubelet.
	// All the node components such as CRI need to be running in the same user namespace.
	KubeletInUserNamespace featuregate.Feature = "KubeletInUserNamespace"

	// owner: @dashpole
	// alpha: v1.13
	// beta: v1.15
	//
	// Enables the kubelet's pod resources grpc endpoint
	KubeletPodResources featuregate.Feature = "KubeletPodResources"

	// owner: @fromanirh
	// alpha: v1.21
	// beta: v1.23
	// Enable POD resources API to return allocatable resources
	KubeletPodResourcesGetAllocatable featuregate.Feature = "KubeletPodResourcesGetAllocatable"

	// owner: @sallyom
	// kep: http://kep.k8s.io/2832
	// alpha: v1.25
	//
	// Add support for distributed tracing in the kubelet
	KubeletTracing featuregate.Feature = "KubeletTracing"

	// owner: @zshihang
	// kep: http://kep.k8s.io/2800
	// beta: v1.24
	//
	// Stop auto-generation of secret-based service account tokens.
	LegacyServiceAccountTokenNoAutoGeneration featuregate.Feature = "LegacyServiceAccountTokenNoAutoGeneration"

	// owner: @jinxu
	// beta: v1.10
	// stable: v1.25
	//
	// Support local ephemeral storage types for local storage capacity isolation feature.
	LocalStorageCapacityIsolation featuregate.Feature = "LocalStorageCapacityIsolation"

	// owner: @RobertKrawitz
	// alpha: v1.15
	//
	// Allow use of filesystems for ephemeral storage monitoring.
	// Only applies if LocalStorageCapacityIsolation is set.
	LocalStorageCapacityIsolationFSQuotaMonitoring featuregate.Feature = "LocalStorageCapacityIsolationFSQuotaMonitoring"

	// owner: @damemi
	// alpha: v1.21
	// beta: v1.22
	//
	// Enables scaling down replicas via logarithmic comparison of creation/ready timestamps
	LogarithmicScaleDown featuregate.Feature = "LogarithmicScaleDown"

	// owner: @denkensk
	// kep: http://kep.k8s.io/3243
	// alpha: v1.25
	//
	// Enable MatchLabelKeys in PodTopologySpread.
	MatchLabelKeysInPodTopologySpread featuregate.Feature = "MatchLabelKeysInPodTopologySpread"

	// owner: @krmayankk
	// alpha: v1.24
	//
	// Enables maxUnavailable for StatefulSet
	MaxUnavailableStatefulSet featuregate.Feature = "MaxUnavailableStatefulSet"

	// owner: @cynepco3hahue(alukiano) @cezaryzukowski @k-wiatrzyk
	// alpha: v1.21
	// beta: v1.22
	// Allows setting memory affinity for a container based on NUMA topology
	MemoryManager featuregate.Feature = "MemoryManager"

	// owner: @xiaoxubeii
	// kep: http://kep.k8s.io/2570
	// alpha: v1.22
	//
	// Enables kubelet to support memory QoS with cgroups v2.
	MemoryQoS featuregate.Feature = "MemoryQoS"

	// owner: @sanposhiho
	// kep: http://kep.k8s.io/3022
	// alpha: v1.24
	// beta: v1.25
	//
	// Enable MinDomains in Pod Topology Spread.
	MinDomainsInPodTopologySpread featuregate.Feature = "MinDomainsInPodTopologySpread"

	// owner: @janosi @bridgetkromhout
	// kep: http://kep.k8s.io/1435
	// alpha: v1.20
	// beta: v1.24
	//
	// Enables the usage of different protocols in the same Service with type=LoadBalancer
	MixedProtocolLBService featuregate.Feature = "MixedProtocolLBService"

	// owner: @sarveshr7
	// kep: http://kep.k8s.io/2593
	// alpha: v1.25
	//
	// Enables the MultiCIDR Range allocator.
	MultiCIDRRangeAllocator featuregate.Feature = "MultiCIDRRangeAllocator"

	// owner: @rikatz
	// kep: http://kep.k8s.io/2079
	// alpha: v1.21
	// beta:  v1.22
	// ga: v1.25
	//
	// Enables the endPort field in NetworkPolicy to enable a Port Range behavior in Network Policies.
	NetworkPolicyEndPort featuregate.Feature = "NetworkPolicyEndPort"

	// owner: @rikatz
	// kep: http://kep.k8s.io/2943
	// alpha: v1.24
	//
	// Enables NetworkPolicy status subresource
	NetworkPolicyStatus featuregate.Feature = "NetworkPolicyStatus"

	// owner: @xing-yang @sonasingh46
	// kep: http://kep.k8s.io/2268
	// alpha: v1.24
	//
	// Allow pods to failover to a different node in case of non graceful node shutdown
	NodeOutOfServiceVolumeDetach featuregate.Feature = "NodeOutOfServiceVolumeDetach"

	// owner: @ehashman
	// alpha: v1.22
	//
	// Permits kubelet to run with swap enabled
	NodeSwap featuregate.Feature = "NodeSwap"

	// owner: @denkensk
	// alpha: v1.15
	// beta: v1.19
	// ga: v1.24
	//
	// Enables NonPreempting option for priorityClass and pod.
	NonPreemptingPriority featuregate.Feature = "NonPreemptingPriority"

	// owner: @ahg-g
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.24
	//
	// Allow specifying NamespaceSelector in PodAffinityTerm.
	PodAffinityNamespaceSelector featuregate.Feature = "PodAffinityNamespaceSelector"

	// owner: @haircommander
	// kep: http://kep.k8s.io/2364
	// alpha: v1.23
	//
	// Configures the Kubelet to use the CRI to populate pod and container stats, instead of supplimenting with stats from cAdvisor.
	// Requires the CRI implementation supports supplying the required stats.
	PodAndContainerStatsFromCRI featuregate.Feature = "PodAndContainerStatsFromCRI"

	// owner: @ahg-g
	// alpha: v1.21
	// beta: v1.22
	//
	// Enables controlling pod ranking on replicaset scale-down.
	PodDeletionCost featuregate.Feature = "PodDeletionCost"

	// owner: @mimowo
	// kep: http://kep.k8s.io/3329
	// alpha: v1.25
	//
	// Enables support for appending a dedicated pod condition indicating that
	// the pod is being deleted due to a disruption.
	PodDisruptionConditions featuregate.Feature = "PodDisruptionConditions"

	// owner: @ddebroy
	// alpha: v1.25
	//
	// Enables reporting of PodHasNetwork condition in pod status after pod
	// sandbox creation and network configuration completes successfully
	PodHasNetworkCondition featuregate.Feature = "PodHasNetworkCondition"

	// owner: @egernst
	// alpha: v1.16
	// beta: v1.18
	// ga: v1.24
	//
	// Enables PodOverhead, for accounting pod overheads which are specific to a given RuntimeClass
	PodOverhead featuregate.Feature = "PodOverhead"

	// owner: @liggitt, @tallclair, sig-auth
	// alpha: v1.22
	// beta: v1.23
	// ga: v1.25
	//
	// Enables the PodSecurity admission plugin
	PodSecurity featuregate.Feature = "PodSecurity"

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
	PreferNominatedNode featuregate.Feature = "PreferNominatedNode"

	// owner: @ehashman
	// alpha: v1.21
	// beta: v1.22
	//
	// Allows user to override pod-level terminationGracePeriod for probes
	ProbeTerminationGracePeriod featuregate.Feature = "ProbeTerminationGracePeriod"

	// owner: @jessfraz
	// alpha: v1.12
	//
	// Enables control over ProcMountType for containers.
	ProcMountType featuregate.Feature = "ProcMountType"

	// owner: @andrewsykim
	// kep: http://kep.k8s.io/1669
	// alpha: v1.22
	//
	// Enable kube-proxy to handle terminating ednpoints when externalTrafficPolicy=Local
	ProxyTerminatingEndpoints featuregate.Feature = "ProxyTerminatingEndpoints"

	// owner: @sjenning
	// alpha: v1.11
	//
	// Allows resource reservations at the QoS level preventing pods at lower QoS levels from
	// bursting into resources requested at higher QoS levels (memory only for now)
	QOSReserved featuregate.Feature = "QOSReserved"

	// owner: @chrishenzie
	// alpha: v1.22
	//
	// Enables usage of the ReadWriteOncePod PersistentVolume access mode.
	ReadWriteOncePod featuregate.Feature = "ReadWriteOncePod"

	// owner: @gnufied
	// kep: http://kep.k8s.io/1790
	// alpha: v1.23
	//
	// Allow users to recover from volume expansion failure
	RecoverVolumeExpansionFailure featuregate.Feature = "RecoverVolumeExpansionFailure"

	// owner: @RomanBednar
	// kep: http://kep.k8s.io/3333
	// alpha: v1.25
	//
	// Allow assigning StorageClass to unbound PVCs retroactively
	RetroactiveDefaultStorageClass featuregate.Feature = "RetroactiveDefaultStorageClass"

	// owner: @mikedanese
	// alpha: v1.7
	// beta: v1.12
	//
	// Gets a server certificate for the kubelet from the Certificate Signing
	// Request API instead of generating one self signed and auto rotates the
	// certificate as expiration approaches.
	RotateKubeletServerCertificate featuregate.Feature = "RotateKubeletServerCertificate"

	// owner: @saschagrunert
	// alpha: v1.22
	//
	// Enables the use of `RuntimeDefault` as the default seccomp profile for all workloads.
	SeccompDefault featuregate.Feature = "SeccompDefault"

	// owner: @maplain @andrewsykim
	// kep: http://kep.k8s.io/2086
	// alpha: v1.21
	// beta: v1.22
	//
	// Enables node-local routing for Service internal traffic
	ServiceInternalTrafficPolicy featuregate.Feature = "ServiceInternalTrafficPolicy"

	// owner: @aojea
	// kep: http://kep.k8s.io/3070
	// alpha: v1.24
	// beta: v1.25
	//
	// Subdivide the ClusterIP range for dynamic and static IP allocation.
	ServiceIPStaticSubrange featuregate.Feature = "ServiceIPStaticSubrange"

	// owner: @andrewsykim @uablrek
	// kep: http://kep.k8s.io/1864
	// alpha: v1.20
	// beta: v1.22
	// ga: v1.24
	//
	// Allows control if NodePorts shall be created for services with "type: LoadBalancer" by defining the spec.AllocateLoadBalancerNodePorts field (bool)
	ServiceLBNodePortControl featuregate.Feature = "ServiceLBNodePortControl"

	// owner: @andrewsykim @XudongLiuHarold
	// kep: http://kep.k8s.io/1959
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.24
	//
	// Enable support multiple Service "type: LoadBalancer" implementations in a cluster by specifying LoadBalancerClass
	ServiceLoadBalancerClass featuregate.Feature = "ServiceLoadBalancerClass"

	// owner: @derekwaynecarr
	// alpha: v1.20
	// beta: v1.22
	//
	// Enables kubelet support to size memory backed volumes
	SizeMemoryBackedVolumes featuregate.Feature = "SizeMemoryBackedVolumes"

	// owner: @mattcary
	// alpha: v1.22
	//
	// Enables policies controlling deletion of PVCs created by a StatefulSet.
	StatefulSetAutoDeletePVC featuregate.Feature = "StatefulSetAutoDeletePVC"

	// owner: @ravig
	// kep: https://kep.k8s.io/2607
	// alpha: v1.22
	// beta: v1.23
	// GA: v1.25
	// StatefulSetMinReadySeconds allows minReadySeconds to be respected by StatefulSet controller
	StatefulSetMinReadySeconds featuregate.Feature = "StatefulSetMinReadySeconds"

	// owner: @adtac
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.24
	//
	// Allows jobs to be created in the suspended state.
	SuspendJob featuregate.Feature = "SuspendJob"

	// owner: @robscott
	// kep: http://kep.k8s.io/2433
	// alpha: v1.21
	// beta: v1.23
	//
	// Enables topology aware hints for EndpointSlices
	TopologyAwareHints featuregate.Feature = "TopologyAwareHints"

	// owner: @lmdaly
	// alpha: v1.16
	// beta: v1.18
	//
	// Enable resource managers to make NUMA aligned decisions
	TopologyManager featuregate.Feature = "TopologyManager"

	// owner: @rata, @giuseppe
	// kep: http://kep.k8s.io/127
	// alpha: v1.25
	//
	// Enables user namespace support for stateless pods.
	UserNamespacesStatelessPodsSupport featuregate.Feature = "UserNamespacesStatelessPodsSupport"

	// owner: @cofyc
	// alpha: v1.21
	VolumeCapacityPriority featuregate.Feature = "VolumeCapacityPriority"

	// owner: @ksubrmnn
	// alpha: v1.14
	//
	// Allows kube-proxy to create DSR loadbalancers for Windows
	WinDSR featuregate.Feature = "WinDSR"

	// owner: @ksubrmnn
	// alpha: v1.14
	// beta: v1.20
	//
	// Allows kube-proxy to run in Overlay mode for Windows
	WinOverlay featuregate.Feature = "WinOverlay"

	// owner: @marosset
	// alpha: v1.22
	// beta: v1.23
	//
	// Enables support for 'HostProcess' containers on Windows nodes.
	WindowsHostProcessContainers featuregate.Feature = "WindowsHostProcessContainers"

	// owner: @kerthcet
	// kep: http://kep.k8s.io/3094
	// alpha: v1.25
	//
	// Allow users to specify whether to take nodeAffinity/nodeTaint into consideration when
	// calculating pod topology spread skew.
	NodeInclusionPolicyInPodTopologySpread featuregate.Feature = "NodeInclusionPolicyInPodTopologySpread"

	// owner: @jsafrane
	// kep: http://kep.k8s.io/1710
	// alpha: v1.25
	// Speed up container startup by mounting volumes with the correct SELinux label
	// instead of changing each file on the volumes recursively.
	// Initial implementation focused on ReadWriteOncePod volumes.
	SELinuxMountReadWriteOncePod featuregate.Feature = "SELinuxMountReadWriteOncePod"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
//
// Entries are separated from each other with blank lines to avoid sweeping gofmt changes
// when adding or removing one entry.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AnyVolumeDataSource: {Default: true, PreRelease: featuregate.Beta}, // on by default in 1.24

	AppArmor: {Default: true, PreRelease: featuregate.Beta},

	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},

	CPUManager: {Default: true, PreRelease: featuregate.Beta},

	CPUManagerPolicyAlphaOptions: {Default: false, PreRelease: featuregate.Alpha},

	CPUManagerPolicyBetaOptions: {Default: true, PreRelease: featuregate.Beta},

	CPUManagerPolicyOptions: {Default: true, PreRelease: featuregate.Beta},

	CSIInlineVolume: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	CSIMigration: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	CSIMigrationAWS: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	CSIMigrationAzureDisk: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // On by default in 1.23 (requires Azure Disk CSI driver)

	CSIMigrationAzureFile: {Default: true, PreRelease: featuregate.Beta}, // On by default in 1.24 (requires Azure File CSI driver)

	CSIMigrationGCE: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.25 (requires GCE PD CSI Driver)

	CSIMigrationPortworx: {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires Portworx CSI driver)

	CSIMigrationRBD: {Default: false, PreRelease: featuregate.Alpha}, // Off by default (requires RBD CSI driver)

	CSIMigrationvSphere: {Default: true, PreRelease: featuregate.Beta},

	CSINodeExpandSecret: {Default: false, PreRelease: featuregate.Alpha},

	CSIStorageCapacity: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	CSIVolumeHealth: {Default: false, PreRelease: featuregate.Alpha},

	ContainerCheckpoint: {Default: false, PreRelease: featuregate.Alpha},

	ControllerManagerLeaderMigration: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	CronJobTimeZone: {Default: true, PreRelease: featuregate.Beta},

	DaemonSetUpdateSurge: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	DefaultPodTopologySpread: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	DelegateFSGroupToCSIDriver: {Default: true, PreRelease: featuregate.Beta},

	DevicePlugins: {Default: true, PreRelease: featuregate.Beta},

	DisableAcceleratorUsageMetrics: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	DisableCloudProviders: {Default: false, PreRelease: featuregate.Alpha},

	DisableKubeletCloudCredentialProviders: {Default: false, PreRelease: featuregate.Alpha},

	DownwardAPIHugePages: {Default: true, PreRelease: featuregate.Beta}, // on by default in 1.22

	DynamicKubeletConfig: {Default: false, PreRelease: featuregate.Deprecated}, // feature gate is deprecated in 1.22, kubelet logic is removed in 1.24, api server logic can be removed in 1.26

	EndpointSliceTerminatingCondition: {Default: true, PreRelease: featuregate.Beta},

	EphemeralContainers: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	ExecProbeTimeout: {Default: true, PreRelease: featuregate.GA}, // lock to default and remove after v1.22 based on KEP #1972 update

	ExpandCSIVolumes: {Default: true, PreRelease: featuregate.GA}, // remove in 1.26

	ExpandInUsePersistentVolumes: {Default: true, PreRelease: featuregate.GA}, // remove in 1.26

	ExpandPersistentVolumes: {Default: true, PreRelease: featuregate.GA}, // remove in 1.26

	ExpandedDNSConfig: {Default: false, PreRelease: featuregate.Alpha},

	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: featuregate.Beta},

	GRPCContainerProbe: {Default: true, PreRelease: featuregate.Beta},

	GracefulNodeShutdown: {Default: true, PreRelease: featuregate.Beta},

	GracefulNodeShutdownBasedOnPodPriority: {Default: true, PreRelease: featuregate.Beta},

	HPAContainerMetrics: {Default: false, PreRelease: featuregate.Alpha},

	HonorPVReclaimPolicy: {Default: false, PreRelease: featuregate.Alpha},

	IdentifyPodOS: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	InTreePluginAWSUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginAzureDiskUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginAzureFileUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginGCEUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginPortworxUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginRBDUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginvSphereUnregister: {Default: false, PreRelease: featuregate.Alpha},

	IndexedJob: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	IPTablesOwnershipCleanup: {Default: false, PreRelease: featuregate.Alpha},

	JobPodFailurePolicy: {Default: false, PreRelease: featuregate.Alpha},

	JobMutableNodeSchedulingDirectives: {Default: true, PreRelease: featuregate.Beta},

	JobReadyPods: {Default: true, PreRelease: featuregate.Beta},

	JobTrackingWithFinalizers: {Default: true, PreRelease: featuregate.Beta},

	KubeletCredentialProviders: {Default: true, PreRelease: featuregate.Beta},

	KubeletInUserNamespace: {Default: false, PreRelease: featuregate.Alpha},

	KubeletPodResources: {Default: true, PreRelease: featuregate.Beta},

	KubeletPodResourcesGetAllocatable: {Default: true, PreRelease: featuregate.Beta},

	KubeletTracing: {Default: false, PreRelease: featuregate.Alpha},

	LegacyServiceAccountTokenNoAutoGeneration: {Default: true, PreRelease: featuregate.Beta},

	LocalStorageCapacityIsolation: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	LocalStorageCapacityIsolationFSQuotaMonitoring: {Default: false, PreRelease: featuregate.Alpha},

	LogarithmicScaleDown: {Default: true, PreRelease: featuregate.Beta},

	MatchLabelKeysInPodTopologySpread: {Default: false, PreRelease: featuregate.Alpha},

	MaxUnavailableStatefulSet: {Default: false, PreRelease: featuregate.Alpha},

	MemoryManager: {Default: true, PreRelease: featuregate.Beta},

	MemoryQoS: {Default: false, PreRelease: featuregate.Alpha},

	MinDomainsInPodTopologySpread: {Default: false, PreRelease: featuregate.Beta},

	MixedProtocolLBService: {Default: true, PreRelease: featuregate.Beta},

	MultiCIDRRangeAllocator: {Default: false, PreRelease: featuregate.Alpha},

	NetworkPolicyEndPort: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	NetworkPolicyStatus: {Default: false, PreRelease: featuregate.Alpha},

	NodeOutOfServiceVolumeDetach: {Default: false, PreRelease: featuregate.Alpha},

	NodeSwap: {Default: false, PreRelease: featuregate.Alpha},

	NonPreemptingPriority: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	PodAffinityNamespaceSelector: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	PodAndContainerStatsFromCRI: {Default: false, PreRelease: featuregate.Alpha},

	PodDeletionCost: {Default: true, PreRelease: featuregate.Beta},

	PodDisruptionConditions: {Default: false, PreRelease: featuregate.Alpha},

	PodHasNetworkCondition: {Default: false, PreRelease: featuregate.Alpha},

	PodOverhead: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	PodSecurity: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	PreferNominatedNode: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	ProbeTerminationGracePeriod: {Default: true, PreRelease: featuregate.Beta}, // Default to true in beta 1.25

	ProcMountType: {Default: false, PreRelease: featuregate.Alpha},

	ProxyTerminatingEndpoints: {Default: false, PreRelease: featuregate.Alpha},

	QOSReserved: {Default: false, PreRelease: featuregate.Alpha},

	ReadWriteOncePod: {Default: false, PreRelease: featuregate.Alpha},

	RecoverVolumeExpansionFailure: {Default: false, PreRelease: featuregate.Alpha},

	RetroactiveDefaultStorageClass: {Default: false, PreRelease: featuregate.Alpha},

	RotateKubeletServerCertificate: {Default: true, PreRelease: featuregate.Beta},

	SeccompDefault: {Default: true, PreRelease: featuregate.Beta},

	ServiceIPStaticSubrange: {Default: true, PreRelease: featuregate.Beta},

	ServiceInternalTrafficPolicy: {Default: true, PreRelease: featuregate.Beta},

	ServiceLBNodePortControl: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	ServiceLoadBalancerClass: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	SizeMemoryBackedVolumes: {Default: true, PreRelease: featuregate.Beta},

	StatefulSetAutoDeletePVC: {Default: false, PreRelease: featuregate.Alpha},

	StatefulSetMinReadySeconds: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.27

	SuspendJob: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26

	TopologyAwareHints: {Default: true, PreRelease: featuregate.Beta},

	TopologyManager: {Default: true, PreRelease: featuregate.Beta},

	VolumeCapacityPriority: {Default: false, PreRelease: featuregate.Alpha},

	UserNamespacesStatelessPodsSupport: {Default: false, PreRelease: featuregate.Alpha},

	WinDSR: {Default: false, PreRelease: featuregate.Alpha},

	WinOverlay: {Default: true, PreRelease: featuregate.Beta},

	WindowsHostProcessContainers: {Default: true, PreRelease: featuregate.Beta},

	NodeInclusionPolicyInPodTopologySpread: {Default: false, PreRelease: featuregate.Alpha},

	SELinuxMountReadWriteOncePod: {Default: false, PreRelease: featuregate.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:

	genericfeatures.AggregatedDiscoveryEndpoint: {Default: false, PreRelease: featuregate.Alpha},

	genericfeatures.APIListChunking: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIPriorityAndFairness: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIResponseCompression: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.AdvancedAuditing: {Default: true, PreRelease: featuregate.GA},

	genericfeatures.CustomResourceValidationExpressions: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.DryRun: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.28

	genericfeatures.OpenAPIEnums: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.OpenAPIV3: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.ServerSideApply: {Default: true, PreRelease: featuregate.GA},

	genericfeatures.ServerSideFieldValidation: {Default: true, PreRelease: featuregate.Beta},

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
	HPAScaleToZero: {Default: false, PreRelease: featuregate.Alpha},
}
