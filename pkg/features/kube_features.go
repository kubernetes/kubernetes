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
	clientfeatures "k8s.io/client-go/features"
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

	// owner: @thockin
	// deprecated: v1.29
	//
	// Enables Service.status.ingress.loadBanace to be set on
	// services of types other than LoadBalancer.
	AllowServiceLBStatusOnNonLB featuregate.Feature = "AllowServiceLBStatusOnNonLB"

	// owner: @bswartz
	// alpha: v1.18
	// beta: v1.24
	//
	// Enables usage of any object for volume data source in PVCs
	AnyVolumeDataSource featuregate.Feature = "AnyVolumeDataSource"

	// owner: @tallclair
	// beta: v1.4
	AppArmor featuregate.Feature = "AppArmor"

	// owner: @danwinship
	// alpha: v1.27
	// beta: v1.29
	// GA: v1.30
	//
	// Enables dual-stack --node-ip in kubelet with external cloud providers
	CloudDualStackNodeIPs featuregate.Feature = "CloudDualStackNodeIPs"

	// owner: @ahmedtd
	// alpha: v1.26
	//
	// Enable ClusterTrustBundle objects and Kubelet integration.
	ClusterTrustBundle featuregate.Feature = "ClusterTrustBundle"

	// owner: @ahmedtd
	// alpha: v1.28
	//
	// Enable ClusterTrustBundle Kubelet projected volumes.  Depends on ClusterTrustBundle.
	ClusterTrustBundleProjection featuregate.Feature = "ClusterTrustBundleProjection"

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

	// owner: @mfordjody
	// alpha: v1.26
	//
	// Bypasses obsolete validation that GCP volumes are read-only when used in
	// Deployments.
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

	// owner: @humblec, @zhucan
	// kep: https://kep.k8s.io/3171
	// alpha: v1.25
	// beta: v1.27
	// GA: v1.29
	// Enables SecretRef field in CSI NodeExpandVolume request.
	CSINodeExpandSecret featuregate.Feature = "CSINodeExpandSecret"

	// owner: @fengzixu
	// alpha: v1.21
	//
	// Enables kubelet to detect CSI volume condition and send the event of the abnormal volume to the corresponding pod that is using it.
	CSIVolumeHealth featuregate.Feature = "CSIVolumeHealth"

	// owner: @seans3
	// kep: http://kep.k8s.io/4006
	// alpha: v1.29
	//
	// Enables StreamTranslator proxy to handle WebSockets upgrade requests for the
	// version of the RemoteCommand subprotocol that supports the "close" signal.
	TranslateStreamCloseWebsocketRequests featuregate.Feature = "TranslateStreamCloseWebsocketRequests"

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
	// beta: v1.29
	//
	// Add support for CDI Device IDs in the Device Plugin API.
	DevicePluginCDIDevices featuregate.Feature = "DevicePluginCDIDevices"

	// owner: @andrewsykim
	// alpha: v1.22
	// beta: v1.29
	//
	// Disable any functionality in kube-apiserver, kube-controller-manager and kubelet related to the `--cloud-provider` component flag.
	DisableCloudProviders featuregate.Feature = "DisableCloudProviders"

	// owner: @andrewsykim
	// alpha: v1.23
	// beta: v1.29
	//
	// Disable in-tree functionality in kubelet to authenticate to cloud provider container registries for image pull credentials.
	DisableKubeletCloudCredentialProviders featuregate.Feature = "DisableKubeletCloudCredentialProviders"

	// owner: @HirazawaUi
	// kep: http://kep.k8s.io/4004
	// alpha: v1.29
	// DisableNodeKubeProxyVersion disable the status.nodeInfo.kubeProxyVersion field of v1.Node
	DisableNodeKubeProxyVersion featuregate.Feature = "DisableNodeKubeProxyVersion"

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

	// owner: @mimowo
	// kep: https://kep.k8s.io/3850
	// alpha: v1.28
	// beta: v1.29
	//
	// Allows users to specify counting of failed pods per index.
	JobBackoffLimitPerIndex featuregate.Feature = "JobBackoffLimitPerIndex"

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
	// beta: v1.29
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

	// KubeletSeparateDiskGC enables Kubelet to garbage collection images/containers on different filesystems
	// owner: @kannon92
	// kep: https://kep.k8s.io/4191
	// alpha: v1.29
	KubeletSeparateDiskGC featuregate.Feature = "KubeletSeparateDiskGC"

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
	// beta: v1.30
	//
	// Implement connection draining for terminating nodes for
	// `externalTrafficPolicy: Cluster` services.
	KubeProxyDrainingTerminatingNodes featuregate.Feature = "KubeProxyDrainingTerminatingNodes"

	// owner: @yt2985
	// kep: http://kep.k8s.io/2799
	// alpha: v1.28
	// beta: v1.29
	// GA: v1.30
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

	// owner: @sanposhiho
	// kep: https://kep.k8s.io/3633
	// alpha: v1.29
	//
	// Enables the MatchLabelKeys and MismatchLabelKeys in PodAffinity and PodAntiAffinity.
	MatchLabelKeysInPodAffinity featuregate.Feature = "MatchLabelKeysInPodAffinity"

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

	// owner: @danwinship
	// kep: https://kep.k8s.io/3866
	// alpha: v1.29
	//
	// Allows running kube-proxy with `--mode nftables`.
	NFTablesProxyMode featuregate.Feature = "NFTablesProxyMode"

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
	// beta: v1.29
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

	// owner: @ddebroy, @kannon92
	// alpha: v1.25
	// beta: v1.29
	//
	// Enables reporting of PodReadyToStartContainersCondition condition in pod status after pod
	// sandbox creation and network configuration completes successfully
	PodReadyToStartContainersCondition featuregate.Feature = "PodReadyToStartContainersCondition"

	// owner: @wzshiming
	// kep: http://kep.k8s.io/2681
	// alpha: v1.28
	// beta: v1.29
	//
	// Adds pod.status.hostIPs and downward API
	PodHostIPs featuregate.Feature = "PodHostIPs"

	// owner: @AxeZhan
	// kep: http://kep.k8s.io/3960
	// alpha: v1.29
	// beta: v1.30
	//
	// Enables SleepAction in container lifecycle hooks
	PodLifecycleSleepAction featuregate.Feature = "PodLifecycleSleepAction"

	// owner: @Huang-Wei
	// kep: https://kep.k8s.io/3521
	// alpha: v1.26
	// beta: v1.27
	//
	// Enable users to specify when a Pod is ready for scheduling.
	PodSchedulingReadiness featuregate.Feature = "PodSchedulingReadiness"

	// owner: @jessfraz
	// alpha: v1.12
	//
	// Enables control over ProcMountType for containers.
	ProcMountType featuregate.Feature = "ProcMountType"

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
	// GA: v1.29
	//
	// Enables usage of the ReadWriteOncePod PersistentVolume access mode.
	ReadWriteOncePod featuregate.Feature = "ReadWriteOncePod"

	// owner: @gnufied
	// kep: https://kep.k8s.io/1790
	// alpha: v1.23
	//
	// Allow users to recover from volume expansion failure
	RecoverVolumeExpansionFailure featuregate.Feature = "RecoverVolumeExpansionFailure"

	// owner: @mikedanese
	// alpha: v1.7
	// beta: v1.12
	//
	// Gets a server certificate for the kubelet from the Certificate Signing
	// Request API instead of generating one self signed and auto rotates the
	// certificate as expiration approaches.
	RotateKubeletServerCertificate featuregate.Feature = "RotateKubeletServerCertificate"

	// owner: @kiashok
	// kep: https://kep.k8s.io/4216
	// alpha: v1.29
	//
	// Adds support to pull images based on the runtime class specified.
	RuntimeClassInImageCriAPI featuregate.Feature = "RuntimeClassInImageCriApi"

	// owner: @danielvegamyhre
	// kep: https://kep.k8s.io/2413
	// beta: v1.27
	//
	// Allows mutating spec.completions for Indexed job when done in tandem with
	// spec.parallelism. Specifically, spec.completions is mutable iff spec.completions
	// equals to spec.parallelism before and after the update.
	ElasticIndexedJob featuregate.Feature = "ElasticIndexedJob"

	// owner: @sanposhiho
	// kep: http://kep.k8s.io/4247
	// beta: v1.28
	//
	// Enables the scheduler's enhancement called QueueingHints,
	// which benefits to reduce the useless requeueing.
	SchedulerQueueingHints featuregate.Feature = "SchedulerQueueingHints"

	// owner: @atosatto @yuanchen8911
	// kep: http://kep.k8s.io/3902
	// beta: v1.29
	//
	// Decouples Taint Eviction Controller, performing taint-based Pod eviction, from Node Lifecycle Controller.
	SeparateTaintEvictionController featuregate.Feature = "SeparateTaintEvictionController"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	// alpha: v1.29
	//
	// Controls whether JTIs (UUIDs) are embedded into generated service account tokens, and whether these JTIs are
	// recorded into the audit log for future requests made by these tokens.
	ServiceAccountTokenJTI featuregate.Feature = "ServiceAccountTokenJTI"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	// alpha: v1.29
	//
	// Controls whether the apiserver supports binding service account tokens to Node objects.
	ServiceAccountTokenNodeBinding featuregate.Feature = "ServiceAccountTokenNodeBinding"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	// alpha: v1.29
	//
	// Controls whether the apiserver will validate Node claims in service account tokens.
	ServiceAccountTokenNodeBindingValidation featuregate.Feature = "ServiceAccountTokenNodeBindingValidation"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	// alpha: v1.29
	//
	// Controls whether the apiserver embeds the node name and uid for the associated node when issuing
	// service account tokens bound to Pod objects.
	ServiceAccountTokenPodNodeInfo featuregate.Feature = "ServiceAccountTokenPodNodeInfo"

	// owner: @xuzhenglun
	// kep: http://kep.k8s.io/3682
	// alpha: v1.27
	// beta: v1.28
	// stable: v1.29
	//
	// Subdivide the NodePort range for dynamic and static port allocation.
	ServiceNodePortStaticSubrange featuregate.Feature = "ServiceNodePortStaticSubrange"

	// owner: @gjkim42 @SergeyKanzhelev @matthyx @tzneal
	// kep: http://kep.k8s.io/753
	// alpha: v1.28
	// beta: v1.29
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
	// GA: v1.30
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

	// owner: @mattcarry, @sunnylovestiramisu
	// kep: https://kep.k8s.io/3751
	// alpha: v1.29
	//
	// Enables user specified volume attributes for persistent volumes, like iops and throughput.
	VolumeAttributesClass featuregate.Feature = "VolumeAttributesClass"

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

	// owner: @Sh4d1,@RyanAoh
	// kep: http://kep.k8s.io/1860
	// alpha: v1.29
	// LoadBalancerIPMode enables the IPMode field in the LoadBalancerIngress status of a Service
	LoadBalancerIPMode featuregate.Feature = "LoadBalancerIPMode"

	// owner: @haircommander
	// kep: http://kep.k8s.io/4210
	// alpha: v1.29
	// ImageMaximumGCAge enables the Kubelet configuration field of the same name, allowing an admin
	// to specify the age after which an image will be garbage collected.
	ImageMaximumGCAge featuregate.Feature = "ImageMaximumGCAge"

	// owner: @saschagrunert
	// alpha: v1.28
	//
	// Enables user namespace support for Pod Security Standards. Enabling this
	// feature will modify all Pod Security Standard rules to allow setting:
	// spec[.*].securityContext.[runAsNonRoot,runAsUser]
	// This feature gate should only be enabled if all nodes in the cluster
	// support the user namespace feature and have it enabled. The feature gate
	// will not graduate or be enabled by default in future Kubernetes
	// releases.
	UserNamespacesPodSecurityStandards featuregate.Feature = "UserNamespacesPodSecurityStandards"

	// owner: @ahutsunshine
	// beta: v1.30
	//
	// Allows namespace indexer for namespace scope resources in apiserver cache to accelerate list operations.
	StorageNamespaceIndex featuregate.Feature = "StorageNamespaceIndex"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))

	// Register all client-go features with kube's feature gate instance and make all client-go
	// feature checks use kube's instance. The effect is that for kube binaries, client-go
	// features are wired to the existing --feature-gates flag just as all other features
	// are. Further, client-go features automatically support the existing mechanisms for
	// feature enablement metrics and test overrides.
	ca := &clientAdapter{utilfeature.DefaultMutableFeatureGate}
	runtime.Must(clientfeatures.AddFeaturesToExistingFeatureGates(ca))
	clientfeatures.ReplaceFeatureGates(ca)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
//
// Entries are separated from each other with blank lines to avoid sweeping gofmt changes
// when adding or removing one entry.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	CrossNamespaceVolumeDataSource: {Default: false, PreRelease: featuregate.Alpha},

	AllowServiceLBStatusOnNonLB: {Default: false, PreRelease: featuregate.Deprecated}, // remove after 1.29

	AnyVolumeDataSource: {Default: true, PreRelease: featuregate.Beta}, // on by default in 1.24

	AppArmor: {Default: true, PreRelease: featuregate.Beta},

	CloudDualStackNodeIPs: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32

	ClusterTrustBundle: {Default: false, PreRelease: featuregate.Alpha},

	ClusterTrustBundleProjection: {Default: false, PreRelease: featuregate.Alpha},

	CPUCFSQuotaPeriod: {Default: false, PreRelease: featuregate.Alpha},

	CPUManager: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.26

	CPUManagerPolicyAlphaOptions: {Default: false, PreRelease: featuregate.Alpha},

	CPUManagerPolicyBetaOptions: {Default: true, PreRelease: featuregate.Beta},

	CPUManagerPolicyOptions: {Default: true, PreRelease: featuregate.Beta},

	CSIMigrationPortworx: {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires Portworx CSI driver)

	CSIMigrationRBD: {Default: false, PreRelease: featuregate.Deprecated}, //  deprecated in 1.28, remove in 1.31

	CSINodeExpandSecret: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	CSIVolumeHealth: {Default: false, PreRelease: featuregate.Alpha},

	SkipReadOnlyValidationGCE: {Default: true, PreRelease: featuregate.Deprecated}, // remove in 1.31

	TranslateStreamCloseWebsocketRequests: {Default: false, PreRelease: featuregate.Alpha},

	CloudControllerManagerWebhook: {Default: false, PreRelease: featuregate.Alpha},

	ContainerCheckpoint: {Default: false, PreRelease: featuregate.Alpha},

	ConsistentHTTPGetHandlers: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	CronJobsScheduledAnnotation: {Default: true, PreRelease: featuregate.Beta},

	DefaultHostNetworkHostPortsInPodTemplates: {Default: false, PreRelease: featuregate.Deprecated},

	DisableCloudProviders: {Default: true, PreRelease: featuregate.Beta},

	DisableKubeletCloudCredentialProviders: {Default: true, PreRelease: featuregate.Beta},

	DisableNodeKubeProxyVersion: {Default: false, PreRelease: featuregate.Alpha},

	DevicePluginCDIDevices: {Default: true, PreRelease: featuregate.Beta},

	DynamicResourceAllocation: {Default: false, PreRelease: featuregate.Alpha},

	EventedPLEG: {Default: false, PreRelease: featuregate.Alpha},

	ExecProbeTimeout: {Default: true, PreRelease: featuregate.GA}, // lock to default and remove after v1.22 based on KEP #1972 update

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

	JobBackoffLimitPerIndex: {Default: true, PreRelease: featuregate.Beta},

	JobPodFailurePolicy: {Default: true, PreRelease: featuregate.Beta},

	JobPodReplacementPolicy: {Default: true, PreRelease: featuregate.Beta},

	JobReadyPods: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	KubeletCgroupDriverFromCRI: {Default: false, PreRelease: featuregate.Alpha},

	KubeletInUserNamespace: {Default: false, PreRelease: featuregate.Alpha},

	KubeletPodResourcesDynamicResources: {Default: false, PreRelease: featuregate.Alpha},

	KubeletPodResourcesGet: {Default: false, PreRelease: featuregate.Alpha},

	KubeletSeparateDiskGC: {Default: false, PreRelease: featuregate.Alpha},

	KubeletTracing: {Default: true, PreRelease: featuregate.Beta},

	KubeProxyDrainingTerminatingNodes: {Default: true, PreRelease: featuregate.Beta},

	LegacyServiceAccountTokenCleanUp: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.30; remove in 1.32

	LocalStorageCapacityIsolationFSQuotaMonitoring: {Default: false, PreRelease: featuregate.Alpha},

	LogarithmicScaleDown: {Default: true, PreRelease: featuregate.Beta},

	MatchLabelKeysInPodAffinity: {Default: false, PreRelease: featuregate.Alpha},

	MatchLabelKeysInPodTopologySpread: {Default: true, PreRelease: featuregate.Beta},

	MaxUnavailableStatefulSet: {Default: false, PreRelease: featuregate.Alpha},

	MemoryManager: {Default: true, PreRelease: featuregate.Beta},

	MemoryQoS: {Default: false, PreRelease: featuregate.Alpha},

	MinDomainsInPodTopologySpread: {Default: true, PreRelease: featuregate.Beta},

	MultiCIDRServiceAllocator: {Default: false, PreRelease: featuregate.Alpha},

	NewVolumeManagerReconstruction: {Default: true, PreRelease: featuregate.Beta},

	NFTablesProxyMode: {Default: false, PreRelease: featuregate.Alpha},

	NodeLogQuery: {Default: false, PreRelease: featuregate.Alpha},

	NodeOutOfServiceVolumeDetach: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	NodeSwap: {Default: false, PreRelease: featuregate.Beta},

	PDBUnhealthyPodEvictionPolicy: {Default: true, PreRelease: featuregate.Beta},

	PersistentVolumeLastPhaseTransitionTime: {Default: true, PreRelease: featuregate.Beta},

	PodAndContainerStatsFromCRI: {Default: false, PreRelease: featuregate.Alpha},

	PodDeletionCost: {Default: true, PreRelease: featuregate.Beta},

	PodDisruptionConditions: {Default: true, PreRelease: featuregate.Beta},

	PodReadyToStartContainersCondition: {Default: true, PreRelease: featuregate.Beta},

	PodHostIPs: {Default: true, PreRelease: featuregate.Beta},

	PodLifecycleSleepAction: {Default: true, PreRelease: featuregate.Beta},

	PodSchedulingReadiness: {Default: true, PreRelease: featuregate.Beta},

	ProcMountType: {Default: false, PreRelease: featuregate.Alpha},

	QOSReserved: {Default: false, PreRelease: featuregate.Alpha},

	ReadWriteOncePod: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	RecoverVolumeExpansionFailure: {Default: false, PreRelease: featuregate.Alpha},

	RotateKubeletServerCertificate: {Default: true, PreRelease: featuregate.Beta},

	RuntimeClassInImageCriAPI: {Default: false, PreRelease: featuregate.Alpha},

	ElasticIndexedJob: {Default: true, PreRelease: featuregate.Beta},

	SchedulerQueueingHints: {Default: false, PreRelease: featuregate.Beta},

	SeparateTaintEvictionController: {Default: true, PreRelease: featuregate.Beta},

	ServiceAccountTokenJTI: {Default: false, PreRelease: featuregate.Alpha},

	ServiceAccountTokenPodNodeInfo: {Default: false, PreRelease: featuregate.Alpha},

	ServiceAccountTokenNodeBinding: {Default: false, PreRelease: featuregate.Alpha},

	ServiceAccountTokenNodeBindingValidation: {Default: false, PreRelease: featuregate.Alpha},

	ServiceNodePortStaticSubrange: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.29; remove in 1.31

	SidecarContainers: {Default: true, PreRelease: featuregate.Beta},

	SizeMemoryBackedVolumes: {Default: true, PreRelease: featuregate.Beta},

	StableLoadBalancerNodeSet: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.30, remove in 1.31

	StatefulSetAutoDeletePVC: {Default: true, PreRelease: featuregate.Beta},

	StatefulSetStartOrdinal: {Default: true, PreRelease: featuregate.Beta},

	TopologyAwareHints: {Default: true, PreRelease: featuregate.Beta},

	TopologyManagerPolicyAlphaOptions: {Default: false, PreRelease: featuregate.Alpha},

	TopologyManagerPolicyBetaOptions: {Default: true, PreRelease: featuregate.Beta},

	TopologyManagerPolicyOptions: {Default: true, PreRelease: featuregate.Beta},

	UnknownVersionInteroperabilityProxy: {Default: false, PreRelease: featuregate.Alpha},

	VolumeAttributesClass: {Default: false, PreRelease: featuregate.Alpha},

	VolumeCapacityPriority: {Default: false, PreRelease: featuregate.Alpha},

	UserNamespacesSupport: {Default: false, PreRelease: featuregate.Alpha},

	WinDSR: {Default: false, PreRelease: featuregate.Alpha},

	WinOverlay: {Default: true, PreRelease: featuregate.Beta},

	WindowsHostNetwork: {Default: true, PreRelease: featuregate.Alpha},

	NodeInclusionPolicyInPodTopologySpread: {Default: true, PreRelease: featuregate.Beta},

	SELinuxMountReadWriteOncePod: {Default: true, PreRelease: featuregate.Beta},

	InPlacePodVerticalScaling: {Default: false, PreRelease: featuregate.Alpha},

	PodIndexLabel: {Default: true, PreRelease: featuregate.Beta},

	LoadBalancerIPMode: {Default: false, PreRelease: featuregate.Alpha},

	ImageMaximumGCAge: {Default: false, PreRelease: featuregate.Alpha},

	UserNamespacesPodSecurityStandards: {Default: false, PreRelease: featuregate.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:

	genericfeatures.AdmissionWebhookMatchConditions: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.AggregatedDiscoveryEndpoint: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIListChunking: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32

	genericfeatures.APIPriorityAndFairness: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	genericfeatures.APIResponseCompression: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIServerIdentity: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.APIServerTracing: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.ConsistentListFromCache: {Default: false, PreRelease: featuregate.Alpha},

	genericfeatures.CustomResourceValidationExpressions: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	genericfeatures.EfficientWatchResumption: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	genericfeatures.KMSv1: {Default: false, PreRelease: featuregate.Deprecated},

	genericfeatures.KMSv2: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	genericfeatures.KMSv2KDF: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	genericfeatures.OpenAPIEnums: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.RemainingItemCount: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	genericfeatures.ServerSideApply: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	genericfeatures.ServerSideFieldValidation: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	genericfeatures.StorageVersionAPI: {Default: false, PreRelease: featuregate.Alpha},

	genericfeatures.StorageVersionHash: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.StructuredAuthenticationConfiguration: {Default: false, PreRelease: featuregate.Alpha},

	genericfeatures.StructuredAuthorizationConfiguration: {Default: false, PreRelease: featuregate.Alpha},

	genericfeatures.UnauthenticatedHTTP2DOSMitigation: {Default: true, PreRelease: featuregate.Beta},

	genericfeatures.ValidatingAdmissionPolicy: {Default: false, PreRelease: featuregate.Beta},

	genericfeatures.WatchBookmark: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	genericfeatures.WatchList: {Default: false, PreRelease: featuregate.Alpha},

	genericfeatures.ZeroLimitedNominalConcurrencyShares: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32

	// inherited features from apiextensions-apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:

	apiextensionsfeatures.CRDValidationRatcheting: {Default: false, PreRelease: featuregate.Alpha},

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
	HPAScaleToZero: {Default: false, PreRelease: featuregate.Alpha},

	StorageNamespaceIndex: {Default: true, PreRelease: featuregate.Beta},
}
