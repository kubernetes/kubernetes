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
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/util/runtime"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // kep: https://kep.k8s.io/NNN
	// // alpha: v1.X
	// MyFeature featuregate.Feature = "MyFeature"
	//
	// Feature gates should be listed in alphabetical, case-sensitive
	// (upper before any lower case character) order. This reduces the risk
	// of code conflicts because changes are more likely to be scattered
	// across the file.

	// owner: @ttakahashi21 @mkimuram
	// kep: https://kep.k8s.io/3294
	// alpha: v1.26
	//
	// Enable usage of Provision of PVCs from snapshots in other namespaces
	CrossNamespaceVolumeDataSource featuregate.Feature = "CrossNamespaceVolumeDataSource"

	// owner: @bswartz
	// alpha: v1.18
	// beta: v1.24
	//
	// Enables usage of any object for volume data source in PVCs
	AnyVolumeDataSource featuregate.Feature = "AnyVolumeDataSource"

	// owner: @nabokihms
	// alpha: v1.26
	// beta: v1.27
	// GA: v1.28
	//
	// Enables API to get self subject attributes after authentication.
	APISelfSubjectReview featuregate.Feature = "APISelfSubjectReview"

	// owner: @tallclair
	// beta: v1.4
	AppArmor featuregate.Feature = "AppArmor"

	// owner: @danwinship
	// alpha: v1.27
	//
	// Enables dual-stack --node-ip in kubelet with external cloud providers
	CloudDualStackNodeIPs featuregate.Feature = "CloudDualStackNodeIPs"

	// owner: @ahmedtd
	// alpha: v1.26
	//
	// Enable ClusterTrustBundle objects and Kubelet integration.
	ClusterTrustBundle featuregate.Feature = "ClusterTrustBundle"

	// owner: @szuecs
	// alpha: v1.12
	//
	// Enable nodes to change CPUCFSQuotaPeriod
	CPUCFSQuotaPeriod featuregate.Feature = "CustomCPUCFSQuotaPeriod"

	// owner: @ConnorDoyle, @fromanirh (only for GA graduation)
	// alpha: v1.8
	// beta: v1.10
	// GA: v1.26
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

	// owner: @andyzhangx
	// alpha: v1.15
	// beta: v1.21
	// GA: v1.26
	//
	// Enables the Azure File in-tree driver to Azure File Driver migration feature.
	CSIMigrationAzureFile featuregate.Feature = "CSIMigrationAzureFile"

	// owner: @mfordjody
	// alpha: v1.26
	//
	// Skip validation Enable in next version
	SkipReadOnlyValidationGCE featuregate.Feature = "SkipReadOnlyValidationGCE"

	// owner: @trierra
	// alpha: v1.23
	//
	// Enables the Portworx in-tree driver to Portworx migration feature.
	CSIMigrationPortworx featuregate.Feature = "CSIMigrationPortworx"

	// owner: @humblec
	// alpha: v1.23
	// deprecated: v1.28
	//
	// Enables the RBD in-tree driver to RBD CSI Driver  migration feature.
	CSIMigrationRBD featuregate.Feature = "CSIMigrationRBD"

	// owner: @divyenpatel
	// beta: v1.19 (requires: vSphere vCenter/ESXi Version: 7.0u2, HW Version: VM version 15)
	// GA: 1.26
	// Enables the vSphere in-tree driver to vSphere CSI Driver migration feature.
	CSIMigrationvSphere featuregate.Feature = "CSIMigrationvSphere"

	// owner: @humblec, @zhucan
	// kep: https://kep.k8s.io/3171
	// alpha: v1.25
	// beta: v1.27
	//
	// Enables SecretRef field in CSI NodeExpandVolume request.
	CSINodeExpandSecret featuregate.Feature = "CSINodeExpandSecret"

	// owner: @fengzixu
	// alpha: v1.21
	//
	// Enables kubelet to detect CSI volume condition and send the event of the abnormal volume to the corresponding pod that is using it.
	CSIVolumeHealth featuregate.Feature = "CSIVolumeHealth"

	// owner: @nckturner
	// kep:  http://kep.k8s.io/2699
	// alpha: v1.27
	// Enable webhooks in cloud controller manager
	CloudControllerManagerWebhook featuregate.Feature = "CloudControllerManagerWebhook"

	// owner: @adrianreber
	// kep: https://kep.k8s.io/2008
	// alpha: v1.25
	//
	// Enables container Checkpoint support in the kubelet
	ContainerCheckpoint featuregate.Feature = "ContainerCheckpoint"

	// owner: @bhcleek @wzshiming
	// GA: v1.25
	//
	// Normalize HttpGet URL and Header passing for lifecycle handlers with probers.
	ConsistentHTTPGetHandlers featuregate.Feature = "ConsistentHTTPGetHandlers"

	// owner: @helayoty
	// beta: v1.28
	// Set the scheduled time as an annotation in the job.
	CronJobsScheduledAnnotation featuregate.Feature = "CronJobsScheduledAnnotation"

	// owner: @deejross, @soltysh
	// kep: https://kep.k8s.io/3140
	// alpha: v1.24
	// beta: v1.25
	// GA: 1.27
	//
	// Enables support for time zones in CronJobs.
	CronJobTimeZone featuregate.Feature = "CronJobTimeZone"

	// owner: @thockin
	// deprecated: v1.28
	//
	// Changes when the default value of PodSpec.containers[].ports[].hostPort
	// is assigned.  The default is to only set a default value in Pods.
	// Enabling this means a default will be assigned even to embeddedPodSpecs
	// (e.g. in a Deployment), which is the historical default.
	DefaultHostNetworkHostPortsInPodTemplates featuregate.Feature = "DefaultHostNetworkHostPortsInPodTemplates"

	// owner: @elezar
	// kep: http://kep.k8s.io/4009
	// alpha: v1.28
	//
	// Add support for CDI Device IDs in the Device Plugin API.
	DevicePluginCDIDevices featuregate.Feature = "DevicePluginCDIDevices"

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
	// ga: v1.27
	//
	// Enables usage of hugepages-<size> in downward API.
	DownwardAPIHugePages featuregate.Feature = "DownwardAPIHugePages"

	// owner: @pohly
	// kep: http://kep.k8s.io/3063
	// alpha: v1.26
	//
	// Enables support for resources with custom parameters and a lifecycle
	// that is independent of a Pod.
	DynamicResourceAllocation featuregate.Feature = "DynamicResourceAllocation"

	// owner: @harche
	// kep: http://kep.k8s.io/3386
	// alpha: v1.25
	//
	// Allows using event-driven PLEG (pod lifecycle event generator) through kubelet
	// which avoids frequent relisting of containers which helps optimize performance.
	EventedPLEG featuregate.Feature = "EventedPLEG"

	// owner: @andrewsykim @SergeyKanzhelev
	// GA: v1.20
	//
	// Ensure kubelet respects exec probe timeouts. Feature gate exists in-case existing workloads
	// may depend on old behavior where exec probe timeouts were ignored.
	// Lock to default and remove after v1.22 based on user feedback that should be reflected in KEP #1972 update
	ExecProbeTimeout featuregate.Feature = "ExecProbeTimeout"

	// owner: @gjkim42
	// kep: https://kep.k8s.io/2595
	// alpha: v1.22
	// beta: v1.26
	// GA: v1.28
	//
	// Enables apiserver and kubelet to allow up to 32 DNSSearchPaths and up to 2048 DNSSearchListChars.
	ExpandedDNSConfig featuregate.Feature = "ExpandedDNSConfig"

	// owner: @pweil-
	// alpha: v1.5
	// deprecated: v1.28
	//
	// This flag used to be needed for dockershim CRI and currently does nothing.
	ExperimentalHostUserNamespaceDefaultingGate featuregate.Feature = "ExperimentalHostUserNamespaceDefaulting"

	// owner: @yuzhiquan, @bowei, @PxyUp, @SergeyKanzhelev
	// kep: https://kep.k8s.io/2727
	// alpha: v1.23
	// beta: v1.24
	// stable: v1.27
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

	// owner: @arjunrn @mwielgus @josephburnett @sanposhiho
	// kep: https://kep.k8s.io/1610
	// alpha: v1.20
	// beta:  v1.27
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
	// kep: https://kep.k8s.io/2680
	// alpha: v1.23
	//
	// Honor Persistent Volume Reclaim Policy when it is "Delete" irrespective of PV-PVC
	// deletion ordering.
	HonorPVReclaimPolicy featuregate.Feature = "HonorPVReclaimPolicy"

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

	// owner: @adisky
	// alpha: v1.21
	//
	// Disables the OpenStack Cinder in-tree driver.
	InTreePluginOpenStackUnregister featuregate.Feature = "InTreePluginOpenStackUnregister"

	// owner: @trierra
	// alpha: v1.23
	//
	// Disables the Portworx in-tree driver.
	InTreePluginPortworxUnregister featuregate.Feature = "InTreePluginPortworxUnregister"

	// owner: @humblec
	// alpha: v1.23
	// deprecated: v1.28
	//
	// Disables the RBD in-tree driver.
	InTreePluginRBDUnregister featuregate.Feature = "InTreePluginRBDUnregister"

	// owner: @divyenpatel
	// alpha: v1.21
	//
	// Disables the vSphere in-tree driver.
	InTreePluginvSphereUnregister featuregate.Feature = "InTreePluginvSphereUnregister"

	// owner: @danwinship
	// kep: https://kep.k8s.io/3178
	// alpha: v1.25
	// beta: v1.27
	// stable: v1.28
	//
	// Causes kubelet to no longer create legacy IPTables rules
	IPTablesOwnershipCleanup featuregate.Feature = "IPTablesOwnershipCleanup"

	// owner: @mimowo
	// kep: https://kep.k8s.io/3850
	// alpha: v1.28
	//
	// Allows users to specify counting of failed pods per index.
	JobBackoffLimitPerIndex featuregate.Feature = "JobBackoffLimitPerIndex"

	// owner: @ahg
	// beta: v1.23
	// stable: v1.27
	//
	// Allow updating node scheduling directives in the pod template of jobs. Specifically,
	// node affinity, selector and tolerations. This is allowed only for suspended jobs
	// that have never been unsuspended before.
	JobMutableNodeSchedulingDirectives featuregate.Feature = "JobMutableNodeSchedulingDirectives"

	// owner: @mimowo
	// kep: https://kep.k8s.io/3329
	// alpha: v1.25
	// beta: v1.26
	//
	// Allow users to specify handling of pod failures based on container exit codes
	// and pod conditions.
	JobPodFailurePolicy featuregate.Feature = "JobPodFailurePolicy"

	// owner: @kannon92
	// kep : https://kep.k8s.io/3939
	// alpha: v1.28
	//
	// Allow users to specify recreating pods of a job only when
	// pods have fully terminated.
	JobPodReplacementPolicy featuregate.Feature = "JobPodReplacementPolicy"
	// owner: @alculquicondor
	// alpha: v1.23
	// beta: v1.24
	//
	// Track the number of pods with Ready condition in the Job status.
	JobReadyPods featuregate.Feature = "JobReadyPods"

	// owner: @alculquicondor
	// alpha: v1.22
	// beta: v1.23
	// stable: v1.26
	//
	// Track Job completion without relying on Pod remaining in the cluster
	// indefinitely. Pod finalizers, in addition to a field in the Job status
	// allow the Job controller to keep track of Pods that it didn't account for
	// yet.
	JobTrackingWithFinalizers featuregate.Feature = "JobTrackingWithFinalizers"

	// owner: @marquiz
	// kep: http://kep.k8s.io/4033
	// alpha: v1.28
	//
	// Enable detection of the kubelet cgroup driver configuration option from
	// the CRI.  The CRI runtime also needs to support this feature in which
	// case the kubelet will ignore the cgroupDriver (--cgroup-driver)
	// configuration option. If runtime doesn't support it, the kubelet will
	// fallback to using it's cgroupDriver option.
	KubeletCgroupDriverFromCRI featuregate.Feature = "KubeletCgroupDriverFromCRI"

	// owner: @AkihiroSuda
	// alpha: v1.22
	//
	// Enables support for running kubelet in a user namespace.
	// The user namespace has to be created before running kubelet.
	// All the node components such as CRI need to be running in the same user namespace.
	KubeletInUserNamespace featuregate.Feature = "KubeletInUserNamespace"

	// owner: @dashpole, @ffromani (only for GA graduation)
	// alpha: v1.13
	// beta: v1.15
	// GA: v1.28
	//
	// Enables the kubelet's pod resources grpc endpoint
	KubeletPodResources featuregate.Feature = "KubeletPodResources"

	// owner: @moshe010
	// alpha: v1.27
	//
	// Enable POD resources API to return resources allocated by Dynamic Resource Allocation
	KubeletPodResourcesDynamicResources featuregate.Feature = "KubeletPodResourcesDynamicResources"

	// owner: @moshe010
	// alpha: v1.27
	//
	// Enable POD resources API with Get method
	KubeletPodResourcesGet featuregate.Feature = "KubeletPodResourcesGet"

	// owner: @ffromani
	// alpha: v1.21
	// beta: v1.23
	// GA: v1.28
	// Enable POD resources API to return allocatable resources
	KubeletPodResourcesGetAllocatable featuregate.Feature = "KubeletPodResourcesGetAllocatable"

	// owner: @sallyom
	// kep: https://kep.k8s.io/2832
	// alpha: v1.25
	// beta: v1.27
	//
	// Add support for distributed tracing in the kubelet
	KubeletTracing featuregate.Feature = "KubeletTracing"

	// owner: @alexanderConstantinescu
	// kep: http://kep.k8s.io/3836
	// alpha: v1.28
	//
	// Implement connection draining for terminating nodes for
	// `externalTrafficPolicy: Cluster` services.
	KubeProxyDrainingTerminatingNodes featuregate.Feature = "KubeProxyDrainingTerminatingNodes"

	// owner: @zshihang
	// kep: https://kep.k8s.io/2800
	// beta: v1.24
	// ga: v1.26
	//
	// Stop auto-generation of secret-based service account tokens.
	LegacyServiceAccountTokenNoAutoGeneration featuregate.Feature = "LegacyServiceAccountTokenNoAutoGeneration"

	// owner: @zshihang
	// kep: http://kep.k8s.io/2800
	// alpha: v1.26
	// beta: v1.27
	//
	// Enables tracking of secret-based service account tokens usage.
	LegacyServiceAccountTokenTracking featuregate.Feature = "LegacyServiceAccountTokenTracking"

	// owner: @yt2985
	// kep: http://kep.k8s.io/2800
	// alpha: v1.28
	//
	// Enables cleaning up of secret-based service account tokens.
	LegacyServiceAccountTokenCleanUp featuregate.Feature = "LegacyServiceAccountTokenCleanUp"

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
	// kep: https://kep.k8s.io/3243
	// alpha: v1.25
	// beta: v1.27
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
	// kep: https://kep.k8s.io/2570
	// alpha: v1.22
	//
	// Enables kubelet to support memory QoS with cgroups v2.
	MemoryQoS featuregate.Feature = "MemoryQoS"

	// owner: @sanposhiho
	// kep: https://kep.k8s.io/3022
	// alpha: v1.24
	// beta: v1.25
	//
	// Enable MinDomains in Pod Topology Spread.
	MinDomainsInPodTopologySpread featuregate.Feature = "MinDomainsInPodTopologySpread"

	// owner: @danwinship
	// kep: http://kep.k8s.io/3453
	// alpha: v1.26
	// beta: v1.27
	//
	// Enables new performance-improving code in kube-proxy iptables mode
	MinimizeIPTablesRestore featuregate.Feature = "MinimizeIPTablesRestore"

	// owner: @sarveshr7
	// kep: https://kep.k8s.io/2593
	// alpha: v1.25
	//
	// Enables the MultiCIDR Range allocator.
	MultiCIDRRangeAllocator featuregate.Feature = "MultiCIDRRangeAllocator"

	// owner: @aojea
	// kep: https://kep.k8s.io/1880
	// alpha: v1.27
	//
	// Enables the dynamic configuration of Service IP ranges
	MultiCIDRServiceAllocator featuregate.Feature = "MultiCIDRServiceAllocator"

	// owner: @jsafrane
	// kep: https://kep.k8s.io/3756
	// alpha: v1.25 (as part of SELinuxMountReadWriteOncePod)
	// beta: v1.27
	// Robust VolumeManager reconstruction after kubelet restart.
	NewVolumeManagerReconstruction featuregate.Feature = "NewVolumeManagerReconstruction"

	// owner: @aravindhp @LorbusChris
	// kep: http://kep.k8s.io/2271
	// alpha: v1.27
	//
	// Enables querying logs of node services using the /logs endpoint
	NodeLogQuery featuregate.Feature = "NodeLogQuery"

	// owner: @xing-yang @sonasingh46
	// kep: https://kep.k8s.io/2268
	// alpha: v1.24
	// beta: v1.26
	// GA: v1.28
	//
	// Allow pods to failover to a different node in case of non graceful node shutdown
	NodeOutOfServiceVolumeDetach featuregate.Feature = "NodeOutOfServiceVolumeDetach"

	// owner: @iholder101
	// alpha: v1.22
	// beta1: v1.28. For more info, please look at the KEP: https://kep.k8s.io/2400.
	//
	// Permits kubelet to run with swap enabled
	NodeSwap featuregate.Feature = "NodeSwap"

	// owner: @mortent, @atiratree, @ravig
	// kep: http://kep.k8s.io/3018
	// alpha: v1.26
	// beta: v1.27
	//
	// Enables PDBUnhealthyPodEvictionPolicy for PodDisruptionBudgets
	PDBUnhealthyPodEvictionPolicy featuregate.Feature = "PDBUnhealthyPodEvictionPolicy"

	// owner: @RomanBednar
	// kep: https://kep.k8s.io/3762
	// alpha: v1.28
	//
	// Adds a new field to persistent volumes which holds a timestamp of when the volume last transitioned its phase.
	PersistentVolumeLastPhaseTransitionTime featuregate.Feature = "PersistentVolumeLastPhaseTransitionTime"

	// owner: @haircommander
	// kep: https://kep.k8s.io/2364
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
	// kep: https://kep.k8s.io/3329
	// alpha: v1.25
	// beta: v1.26
	//
	// Enables support for appending a dedicated pod condition indicating that
	// the pod is being deleted due to a disruption.
	PodDisruptionConditions featuregate.Feature = "PodDisruptionConditions"

	// owner: @danielvegamyhre
	// kep: https://kep.k8s.io/4017
	// beta: v1.28
	//
	// Set pod completion index as a pod label for Indexed Jobs.
	PodIndexLabel featuregate.Feature = "PodIndexLabel"

	// owner: @ddebroy
	// alpha: v1.25
	//
	// Enables reporting of PodReadyToStartContainersCondition condition in pod status after pod
	// sandbox creation and network configuration completes successfully
	PodReadyToStartContainersCondition featuregate.Feature = "PodReadyToStartContainersCondition"

	// owner: @wzshiming
	// kep: http://kep.k8s.io/2681
	// alpha: v1.28
	//
	// Adds pod.status.hostIPs and downward API
	PodHostIPs featuregate.Feature = "PodHostIPs"

	// owner: @Huang-Wei
	// kep: https://kep.k8s.io/3521
	// alpha: v1.26
	// beta: v1.27
	//
	// Enable users to specify when a Pod is ready for scheduling.
	PodSchedulingReadiness featuregate.Feature = "PodSchedulingReadiness"

	// owner: @rphillips
	// alpha: v1.21
	// beta: v1.22
	// ga: v1.28
	//
	// Allows user to override pod-level terminationGracePeriod for probes
	ProbeTerminationGracePeriod featuregate.Feature = "ProbeTerminationGracePeriod"

	// owner: @jessfraz
	// alpha: v1.12
	//
	// Enables control over ProcMountType for containers.
	ProcMountType featuregate.Feature = "ProcMountType"

	// owner: @andrewsykim
	// kep: https://kep.k8s.io/1669
	// alpha: v1.22
	// beta: v1.26
	// GA: v1.28
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
	// kep: https://kep.k8s.io/2485
	// alpha: v1.22
	// beta: v1.27
	//
	// Enables usage of the ReadWriteOncePod PersistentVolume access mode.
	ReadWriteOncePod featuregate.Feature = "ReadWriteOncePod"

	// owner: @gnufied
	// kep: https://kep.k8s.io/1790
	// alpha: v1.23
	//
	// Allow users to recover from volume expansion failure
	RecoverVolumeExpansionFailure featuregate.Feature = "RecoverVolumeExpansionFailure"

	// owner: @RomanBednar
	// kep: https://kep.k8s.io/3333
	// alpha: v1.25
	// beta: 1.26
	// stable: v1.28
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

	// owner: @danielvegamyhre
	// kep: https://kep.k8s.io/2413
	// beta: v1.27
	//
	// Allows mutating spec.completions for Indexed job when done in tandem with
	// spec.parallelism. Specifically, spec.completions is mutable iff spec.completions
	// equals to spec.parallelism before and after the update.
	ElasticIndexedJob featuregate.Feature = "ElasticIndexedJob"

	// owner: @sanposhiho
	// kep: http://kep.k8s.io/3063
	// beta: v1.28
	//
	// Enables the scheduler's enhancement called QueueingHints,
	// which benefits to reduce the useless requeueing.
	SchedulerQueueingHints featuregate.Feature = "SchedulerQueueingHints"

	// owner: @saschagrunert
	// kep: https://kep.k8s.io/2413
	// alpha: v1.22
	// beta: v1.25
	// ga: v1.27
	//
	// Enables the use of `RuntimeDefault` as the default seccomp profile for all workloads.
	SeccompDefault featuregate.Feature = "SeccompDefault"

	// owner: @mtardy
	// alpha: v1.0
	//
	// Putting this admission plugin behind a feature gate is part of the
	// deprecation process. For details about the removal see:
	// https://github.com/kubernetes/kubernetes/issues/111516
	SecurityContextDeny featuregate.Feature = "SecurityContextDeny"

	// owner: @xuzhenglun
	// kep: http://kep.k8s.io/3682
	// alpha: v1.27
	// beta: v1.28
	//
	// Subdivide the NodePort range for dynamic and static port allocation.
	ServiceNodePortStaticSubrange featuregate.Feature = "ServiceNodePortStaticSubrange"

	// owner: @gjkim42 @SergeyKanzhelev @matthyx @tzneal
	// kep: http://kep.k8s.io/753
	// alpha: v1.28
	//
	// Introduces sidecar containers, a new type of init container that starts
	// before other containers but remains running for the full duration of the
	// pod's lifecycle and will not block pod termination.
	SidecarContainers featuregate.Feature = "SidecarContainers"

	// owner: @derekwaynecarr
	// alpha: v1.20
	// beta: v1.22
	//
	// Enables kubelet support to size memory backed volumes
	SizeMemoryBackedVolumes featuregate.Feature = "SizeMemoryBackedVolumes"

	// owner: @alexanderConstantinescu
	// kep: http://kep.k8s.io/3458
	// beta: v1.27
	//
	// Enables less load balancer re-configurations by the service controller
	// (KCCM) as an effect of changing node state.
	StableLoadBalancerNodeSet featuregate.Feature = "StableLoadBalancerNodeSet"

	// owner: @mattcary
	// alpha: v1.22
	// beta: v1.27
	//
	// Enables policies controlling deletion of PVCs created by a StatefulSet.
	StatefulSetAutoDeletePVC featuregate.Feature = "StatefulSetAutoDeletePVC"

	// owner: @psch
	// alpha: v1.26
	// beta: v1.27
	//
	// Enables a StatefulSet to start from an arbitrary non zero ordinal
	StatefulSetStartOrdinal featuregate.Feature = "StatefulSetStartOrdinal"

	// owner: @robscott
	// kep: https://kep.k8s.io/2433
	// alpha: v1.21
	// beta: v1.23
	//
	// Enables topology aware hints for EndpointSlices
	TopologyAwareHints featuregate.Feature = "TopologyAwareHints"

	// owner: @lmdaly, @swatisehgal (for GA graduation)
	// alpha: v1.16
	// beta: v1.18
	// GA: v1.27
	//
	// Enable resource managers to make NUMA aligned decisions
	TopologyManager featuregate.Feature = "TopologyManager"

	// owner: @PiotrProkop
	// kep: https://kep.k8s.io/3545
	// alpha: v1.26
	//
	// Allow fine-tuning of topology manager policies with alpha options.
	// This feature gate:
	// - will guard *a group* of topology manager options whose quality level is alpha.
	// - will never graduate to beta or stable.
	TopologyManagerPolicyAlphaOptions featuregate.Feature = "TopologyManagerPolicyAlphaOptions"

	// owner: @PiotrProkop
	// kep: https://kep.k8s.io/3545
	// alpha: v1.26
	//
	// Allow fine-tuning of topology manager policies with beta options.
	// This feature gate:
	// - will guard *a group* of topology manager options whose quality level is beta.
	// - is thus *introduced* as beta
	// - will never graduate to stable.
	TopologyManagerPolicyBetaOptions featuregate.Feature = "TopologyManagerPolicyBetaOptions"

	// owner: @PiotrProkop
	// kep: https://kep.k8s.io/3545
	// alpha: v1.26
	//
	// Allow the usage of options to fine-tune the topology manager policies.
	TopologyManagerPolicyOptions featuregate.Feature = "TopologyManagerPolicyOptions"

	// owner: @richabanker
	// alpha: v1.28
	//
	// Proxies client to an apiserver capable of serving the request in the event of version skew.
	UnknownVersionInteroperabilityProxy featuregate.Feature = "UnknownVersionInteroperabilityProxy"

	// owner: @rata, @giuseppe
	// kep: https://kep.k8s.io/127
	// alpha: v1.25
	//
	// Enables user namespace support for stateless pods.
	UserNamespacesSupport featuregate.Feature = "UserNamespacesSupport"

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
	// kep: https://kep.k8s.io/3503
	// alpha: v1.26
	//
	// Enables support for joining Windows containers to a hosts' network namespace.
	WindowsHostNetwork featuregate.Feature = "WindowsHostNetwork"

	// owner: @kerthcet
	// kep: https://kep.k8s.io/3094
	// alpha: v1.25
	// beta: v1.26
	//
	// Allow users to specify whether to take nodeAffinity/nodeTaint into consideration when
	// calculating pod topology spread skew.
	NodeInclusionPolicyInPodTopologySpread featuregate.Feature = "NodeInclusionPolicyInPodTopologySpread"

	// owner: @jsafrane
	// kep: https://kep.k8s.io/1710
	// alpha: v1.25
	// beta: v1.27
	// Speed up container startup by mounting volumes with the correct SELinux label
	// instead of changing each file on the volumes recursively.
	// Initial implementation focused on ReadWriteOncePod volumes.
	SELinuxMountReadWriteOncePod featuregate.Feature = "SELinuxMountReadWriteOncePod"

	// owner: @vinaykul
	// kep: http://kep.k8s.io/1287
	// alpha: v1.27
	//
	// Enables In-Place Pod Vertical Scaling
	InPlacePodVerticalScaling featuregate.Feature = "InPlacePodVerticalScaling"
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
	CrossNamespaceVolumeDataSource: {Default: false, PreRelease: featuregate.Alpha},

	AnyVolumeDataSource: {Default: true, PreRelease: featuregate.Beta}, // on by default in 1.24

	APISelfSubjectReview: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.28; remove in 1.30

	AppArmor: {Default: true, PreRelease: featuregate.Beta},

	CloudDualStackNodeIPs: {Default: false, PreRelease: featuregate.Alpha},

	ClusterTrustBundle: {Default: false, PreRelease: featuregate.Alpha},

	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},

	CPUManager: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.26

	CPUManagerPolicyAlphaOptions: {Default: false, PreRelease: featuregate.Alpha},

	CPUManagerPolicyBetaOptions: {Default: true, PreRelease: featuregate.Beta},

	CPUManagerPolicyOptions: {Default: true, PreRelease: featuregate.Beta},

	CSIMigrationAzureFile: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.28

	CSIMigrationPortworx: {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires Portworx CSI driver)

	CSIMigrationRBD: {Default: false, PreRelease: featuregate.Deprecated}, //  deprecated in 1.28, remove in 1.31

	CSIMigrationvSphere: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	CSINodeExpandSecret: {Default: true, PreRelease: featuregate.Beta},

	CSIVolumeHealth: {Default: false, PreRelease: featuregate.Alpha},

	SkipReadOnlyValidationGCE: {Default: false, PreRelease: featuregate.Alpha},

	CloudControllerManagerWebhook: {Default: false, PreRelease: featuregate.Alpha},

	ContainerCheckpoint: {Default: false, PreRelease: featuregate.Alpha},

	ConsistentHTTPGetHandlers: {Default: true, PreRelease: featuregate.GA},

	CronJobsScheduledAnnotation: {Default: true, PreRelease: featuregate.Beta},

	CronJobTimeZone: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	DefaultHostNetworkHostPortsInPodTemplates: {Default: false, PreRelease: featuregate.Deprecated},

	DisableCloudProviders: {Default: false, PreRelease: featuregate.Alpha},

	DisableKubeletCloudCredentialProviders: {Default: false, PreRelease: featuregate.Alpha},

	DevicePluginCDIDevices: {Default: false, PreRelease: featuregate.Alpha},

	DownwardAPIHugePages: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in v1.29

	DynamicResourceAllocation: {Default: false, PreRelease: featuregate.Alpha},

	EventedPLEG: {Default: false, PreRelease: featuregate.Alpha},

	ExecProbeTimeout: {Default: true, PreRelease: featuregate.GA}, // lock to default and remove after v1.22 based on KEP #1972 update

	ExpandedDNSConfig: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.30

	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: featuregate.Deprecated, LockToDefault: true}, // remove in 1.30

	GRPCContainerProbe: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, //remove in 1.29

	GracefulNodeShutdown: {Default: true, PreRelease: featuregate.Beta},

	GracefulNodeShutdownBasedOnPodPriority: {Default: true, PreRelease: featuregate.Beta},

	HPAContainerMetrics: {Default: true, PreRelease: featuregate.Beta},

	HonorPVReclaimPolicy: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginAWSUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginAzureDiskUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginAzureFileUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginGCEUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginOpenStackUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginPortworxUnregister: {Default: false, PreRelease: featuregate.Alpha},

	InTreePluginRBDUnregister: {Default: false, PreRelease: featuregate.Deprecated}, // deprecated in 1.28, remove in 1.31

	InTreePluginvSphereUnregister: {Default: false, PreRelease: featuregate.Alpha},

	IPTablesOwnershipCleanup: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.30

	JobBackoffLimitPerIndex: {Default: false, PreRelease: featuregate.Alpha},

	JobMutableNodeSchedulingDirectives: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	JobPodFailurePolicy: {Default: true, PreRelease: featuregate.Beta},

	JobPodReplacementPolicy: {Default: false, PreRelease: featuregate.Alpha},

	JobReadyPods: {Default: true, PreRelease: featuregate.Beta},

	JobTrackingWithFinalizers: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.28

	KubeletCgroupDriverFromCRI: {Default: false, PreRelease: featuregate.Alpha},

	KubeletInUserNamespace: {Default: false, PreRelease: featuregate.Alpha},

	KubeletPodResources: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.28, remove in 1.30

	KubeletPodResourcesDynamicResources: {Default: false, PreRelease: featuregate.Alpha},

	KubeletPodResourcesGet: {Default: false, PreRelease: featuregate.Alpha},

	KubeletPodResourcesGetAllocatable: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.28, remove in 1.30

	KubeletTracing: {Default: true, PreRelease: featuregate.Beta},

	KubeProxyDrainingTerminatingNodes: {Default: false, PreRelease: featuregate.Alpha},

	LegacyServiceAccountTokenNoAutoGeneration: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	LegacyServiceAccountTokenTracking: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.30

	LegacyServiceAccountTokenCleanUp: {Default: false, PreRelease: featuregate.Alpha},

	LocalStorageCapacityIsolationFSQuotaMonitoring: {Default: false, PreRelease: featuregate.Alpha},

	LogarithmicScaleDown: {Default: true, PreRelease: featuregate.Beta},

	MatchLabelKeysInPodTopologySpread: {Default: true, PreRelease: featuregate.Beta},

	MaxUnavailableStatefulSet: {Default: false, PreRelease: featuregate.Alpha},

	MemoryManager: {Default: true, PreRelease: featuregate.Beta},

	MemoryQoS: {Default: false, PreRelease: featuregate.Alpha},

	MinDomainsInPodTopologySpread: {Default: true, PreRelease: featuregate.Beta},

	MinimizeIPTablesRestore: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.30

	MultiCIDRRangeAllocator: {Default: false, PreRelease: featuregate.Alpha},

	MultiCIDRServiceAllocator: {Default: false, PreRelease: featuregate.Alpha},

	NewVolumeManagerReconstruction: {Default: true, PreRelease: featuregate.Beta},

	NodeLogQuery: {Default: false, PreRelease: featuregate.Alpha},

	NodeOutOfServiceVolumeDetach: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	NodeSwap: {Default: false, PreRelease: featuregate.Beta},

	PDBUnhealthyPodEvictionPolicy: {Default: true, PreRelease: featuregate.Beta},

	PersistentVolumeLastPhaseTransitionTime: {Default: false, PreRelease: featuregate.Alpha},

	PodAndContainerStatsFromCRI: {Default: false, PreRelease: featuregate.Alpha},

	PodDeletionCost: {Default: true, PreRelease: featuregate.Beta},

	PodDisruptionConditions: {Default: true, PreRelease: featuregate.Beta},

	PodReadyToStartContainersCondition: {Default: false, PreRelease: featuregate.Alpha},

	PodHostIPs: {Default: false, PreRelease: featuregate.Alpha},

	PodSchedulingReadiness: {Default: true, PreRelease: featuregate.Beta},

	ProbeTerminationGracePeriod: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	ProcMountType: {Default: false, PreRelease: featuregate.Alpha},

	ProxyTerminatingEndpoints: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.30

	QOSReserved: {Default: false, PreRelease: featuregate.Alpha},

	ReadWriteOncePod: {Default: true, PreRelease: featuregate.Beta},

	RecoverVolumeExpansionFailure: {Default: false, PreRelease: featuregate.Alpha},

	RetroactiveDefaultStorageClass: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	RotateKubeletServerCertificate: {Default: true, PreRelease: featuregate.Beta},

	ElasticIndexedJob: {Default: true, PreRelease: featuregate.Beta},

	SchedulerQueueingHints: {Default: false, PreRelease: featuregate.Beta},

	SeccompDefault: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	SecurityContextDeny: {Default: false, PreRelease: featuregate.Alpha},

	ServiceNodePortStaticSubrange: {Default: true, PreRelease: featuregate.Beta},

	SidecarContainers: {Default: false, PreRelease: featuregate.Alpha},

	SizeMemoryBackedVolumes: {Default: true, PreRelease: featuregate.Beta},

	StableLoadBalancerNodeSet: {Default: true, PreRelease: featuregate.Beta},

	StatefulSetAutoDeletePVC: {Default: true, PreRelease: featuregate.Beta},

	StatefulSetStartOrdinal: {Default: true, PreRelease: featuregate.Beta},

	TopologyAwareHints: {Default: true, PreRelease: featuregate.Beta},

	TopologyManager: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.27; remove in 1.29

	TopologyManagerPolicyAlphaOptions: {Default: false, PreRelease: featuregate.Alpha},

	TopologyManagerPolicyBetaOptions: {Default: true, PreRelease: featuregate.Beta},

	TopologyManagerPolicyOptions: {Default: true, PreRelease: featuregate.Beta},

	UnknownVersionInteroperabilityProxy: {Default: false, PreRelease: featuregate.Alpha},

	VolumeCapacityPriority: {Default: false, PreRelease: featuregate.Alpha},

	UserNamespacesSupport: {Default: false, PreRelease: featuregate.Alpha},

	WinDSR: {Default: false, PreRelease: featuregate.Alpha},

	WinOverlay: {Default: true, PreRelease: featuregate.Beta},

	WindowsHostNetwork: {Default: true, PreRelease: featuregate.Alpha},

	NodeInclusionPolicyInPodTopologySpread: {Default: true, PreRelease: featuregate.Beta},

	SELinuxMountReadWriteOncePod: {Default: true, PreRelease: featuregate.Beta},

	InPlacePodVerticalScaling: {Default: false, PreRelease: featuregate.Alpha},

	PodIndexLabel: {Default: true, PreRelease: featuregate.Beta},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:

	genericfeatures.AdmissionWebhookMatchConditions: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.AggregatedDiscoveryEndpoint: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIListChunking: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIPriorityAndFairness: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIResponseCompression: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.ValidatingAdmissionPolicy: {Default: false, PreRelease: featuregate.Beta},

	genericfeatures.CustomResourceValidationExpressions: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.OpenAPIEnums: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.OpenAPIV3: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	genericfeatures.ServerSideApply: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	genericfeatures.ServerSideFieldValidation: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	genericfeatures.UnauthenticatedHTTP2DOSMitigation: {Default: false, PreRelease: featuregate.Beta},

	// inherited features from apiextensions-apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:

	apiextensionsfeatures.CRDValidationRatcheting: {Default: false, PreRelease: featuregate.Alpha},

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
	HPAScaleToZero: {Default: false, PreRelease: featuregate.Alpha},
}
