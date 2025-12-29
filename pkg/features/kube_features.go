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
	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	"k8s.io/component-base/featuregate"
	zpagesfeatures "k8s.io/component-base/zpages/features"
	kcmfeatures "k8s.io/controller-manager/pkg/features"
)

// Every feature gate should have an entry here following this template:
//
// // owner: @username
// // kep: https://kep.k8s.io/NNN
// MyFeature featuregate.Feature = "MyFeature"
//
// Feature gates should be listed in alphabetical, case-sensitive
// (upper before any lower case character) order. This reduces the risk
// of code conflicts because changes are more likely to be scattered
// across the file.
const (
	// owner: @aojea
	//
	// Allow kubelet to request a certificate without any Node IP available, only
	// with DNS names.
	AllowDNSOnlyNodeCSR featuregate.Feature = "AllowDNSOnlyNodeCSR"

	// owner: @micahhausler
	//
	// Setting AllowInsecureKubeletCertificateSigningRequests to true disables node admission validation of CSRs
	// for kubelet signers where CN=system:node:$nodeName.
	AllowInsecureKubeletCertificateSigningRequests featuregate.Feature = "AllowInsecureKubeletCertificateSigningRequests"

	// owner: @HirazawaUi
	//
	// Allow spec.terminationGracePeriodSeconds to be overridden by MaxPodGracePeriodSeconds in soft evictions.
	AllowOverwriteTerminationGracePeriodSeconds featuregate.Feature = "AllowOverwriteTerminationGracePeriodSeconds"

	// owner: @bswartz
	//
	// Enables usage of any object for volume data source in PVCs
	AnyVolumeDataSource featuregate.Feature = "AnyVolumeDataSource"

	// owner: @liggitt
	// kep: https://kep.k8s.io/4601
	//
	// Make the Node authorizer use fine-grained selector authorization.
	// Requires AuthorizeWithSelectors to be enabled.
	AuthorizeNodeWithSelectors featuregate.Feature = "AuthorizeNodeWithSelectors"

	// owner: @seans3
	// kep: http://kep.k8s.io/4006
	//
	// Forces authorization of the "create" verb for pod subresources like exec, attach, and portforward.
	// See: https://github.com/kubernetes/kubernetes/issues/133515
	AuthorizePodWebsocketUpgradeCreatePermission featuregate.Feature = "AuthorizePodWebsocketUpgradeCreatePermission"

	// owner: @szuecs
	//
	// Enable nodes to change CPUCFSQuotaPeriod
	CPUCFSQuotaPeriod featuregate.Feature = "CustomCPUCFSQuotaPeriod"

	// owner: @fromanirh
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

	// owner: @ffromani
	//
	// Allow the usage of options to fine-tune the cpumanager policies.
	CPUManagerPolicyOptions featuregate.Feature = "CPUManagerPolicyOptions"

	// owner: @trierra
	// kep:  http://kep.k8s.io/2589
	//
	// Enables the Portworx in-tree driver to Portworx migration feature.
	CSIMigrationPortworx featuregate.Feature = "CSIMigrationPortworx"

	// owner: @aramase
	// kep:  http://kep.k8s.io/5538
	//
	// Enables CSI drivers to opt-in for receiving service account tokens from kubelet
	// through the dedicated secrets field in NodePublishVolumeRequest instead of the volume_context field.
	CSIServiceAccountTokenSecrets featuregate.Feature = "CSIServiceAccountTokenSecrets"

	// owner: @fengzixu
	//
	// Enables kubelet to detect CSI volume condition and send the event of the abnormal volume to the corresponding pod that is using it.
	CSIVolumeHealth featuregate.Feature = "CSIVolumeHealth"

	// owner: @HirazawaUi
	//
	// Enabling this feature gate will cause the pod's status to change due to a kubelet restart.
	ChangeContainerStatusOnKubeletRestart featuregate.Feature = "ChangeContainerStatusOnKubeletRestart"

	// owner: @sanposhiho @wojtek-t
	// kep: https://kep.k8s.io/5278
	//
	// Clear pod.Status.NominatedNodeName when pod is bound to a node.
	// This prevents stale information from affecting external scheduling components.
	ClearingNominatedNodeNameAfterBinding featuregate.Feature = "ClearingNominatedNodeNameAfterBinding"

	// owner: @ahmedtd
	//
	// Enable ClusterTrustBundle objects and Kubelet integration.
	ClusterTrustBundle featuregate.Feature = "ClusterTrustBundle"

	// owner: @ahmedtd
	//
	// Enable ClusterTrustBundle Kubelet projected volumes.  Depends on ClusterTrustBundle.
	ClusterTrustBundleProjection featuregate.Feature = "ClusterTrustBundleProjection"

	// owner: @adrianreber
	// kep: https://kep.k8s.io/2008
	//
	// Enables container Checkpoint support in the kubelet
	ContainerCheckpoint featuregate.Feature = "ContainerCheckpoint"

	// onwer: @yuanwang04
	// kep: https://kep.k8s.io/5307
	//
	// Supports container restart policy and container restart policy rules to override the pod restart policy.
	// Enable a single container to restart even if the pod has restart policy "Never".
	ContainerRestartRules featuregate.Feature = "ContainerRestartRules"

	// owner: @sreeram-venkitesh
	//
	// Enables configuring custom stop signals for containers from container lifecycle
	ContainerStopSignals featuregate.Feature = "ContainerStopSignals"

	// owner: @jefftree
	// kep: https://kep.k8s.io/4355
	//
	// Enables coordinated leader election in the API server
	CoordinatedLeaderElection featuregate.Feature = "CoordinatedLeaderElection"

	// owner: @ttakahashi21 @mkimuram
	// kep: https://kep.k8s.io/3294
	//
	// Enable usage of Provision of PVCs from snapshots in other namespaces
	CrossNamespaceVolumeDataSource featuregate.Feature = "CrossNamespaceVolumeDataSource"

	// owner: @ritazh
	// kep: http://kep.k8s.io/5018
	//
	// Enables support for requesting admin access in a ResourceClaim.
	// Admin access is granted even if a device is already in use and,
	// depending on the DRA driver, may enable additional permissions
	// when a container uses the allocated device.
	DRAAdminAccess featuregate.Feature = "DRAAdminAccess"
	// owner: @sunya-ch
	// kep: https://kep.k8s.io/5075
	//
	// DRAConsumableCapacity
	DRAConsumableCapacity featuregate.Feature = "DRAConsumableCapacity"

	// owner: @KobayashiD27
	// kep: http://kep.k8s.io/5007
	// alpha: v1.34
	//
	// Enables support for delaying the binding of pods
	// which depend on devices with binding conditions.
	//
	// DRAResourceClaimDeviceStatus also needs to be
	// enabled.
	DRADeviceBindingConditions featuregate.Feature = "DRADeviceBindingConditions"

	// owner: @pohly
	// kep: http://kep.k8s.io/5055
	//
	// DeviceTaintRules allow administrators to add taints to devices.
	DRADeviceTaintRules featuregate.Feature = "DRADeviceTaintRules"

	// owner: @pohly
	// kep: http://kep.k8s.io/5055
	//
	// Marking devices as tainted can prevent using them for new pods and/or
	// cause pods using them to stop. Users can decide to tolerate taints.
	DRADeviceTaints featuregate.Feature = "DRADeviceTaints"

	// owner: @yliaog
	// kep: http://kep.k8s.io/5004
	//
	// Enables support for providing extended resource requests backed by DRA.
	DRAExtendedResource featuregate.Feature = "DRAExtendedResource"

	// owner: @mortent, @cici37
	// kep: http://kep.k8s.io/4815
	//
	// Enables support for dynamically partitioning devices based on
	// which parts of them were allocated during scheduling.
	//
	DRAPartitionableDevices featuregate.Feature = "DRAPartitionableDevices"

	// owner: @mortent
	// kep: http://kep.k8s.io/4816
	//
	// Enables support for providing a prioritized list of requests
	// for resources. The first entry that can be satisfied will
	// be selected.
	DRAPrioritizedList featuregate.Feature = "DRAPrioritizedList"

	// owner: @LionelJouin
	// kep: http://kep.k8s.io/4817
	//
	// Enables support the ResourceClaim.status.devices field and for setting this
	// status from DRA drivers.
	DRAResourceClaimDeviceStatus featuregate.Feature = "DRAResourceClaimDeviceStatus"

	// owner: @pohly
	// kep: http://kep.k8s.io/4381
	//
	// Enables aborting the per-node Filter operation in the scheduler after
	// a certain time (10 seconds by default, configurable in the DynamicResources
	// scheduler plugin configuration).
	DRASchedulerFilterTimeout featuregate.Feature = "DRASchedulerFilterTimeout"

	// owner: @atiratree
	// kep: http://kep.k8s.io/3973
	//
	// Deployments and replica sets can now also track terminating pods via .status.terminatingReplicas.
	DeploymentReplicaSetTerminatingReplicas featuregate.Feature = "DeploymentReplicaSetTerminatingReplicas"

	// owner: @aojea
	//
	// The apiservers with the MultiCIDRServiceAllocator feature enable, in order to support live migration from the old bitmap ClusterIP
	// allocators to the new IPAddress allocators introduced by the MultiCIDRServiceAllocator feature, performs a dual-write on
	// both allocators. This feature gate disables the dual write on the new Cluster IP allocators.
	DisableAllocatorDualWrite featuregate.Feature = "DisableAllocatorDualWrite"

	// owner: @ffromani
	// beta: v1.33
	//
	// Disables CPU Quota for containers which have exclusive CPUs allocated.
	// Disables pod-Level CPU Quota for pods containing at least one container with exclusive CPUs allocated
	// Exclusive CPUs for a container (init, application, sidecar) are allocated when:
	// (1) cpumanager policy is static,
	// (2) the pod has QoS Guaranteed,
	// (3) the container has integer cpu request.
	// The expected behavior is that CPU Quota for containers having exclusive CPUs allocated is disabled.
	// Because this fix changes a long-established (but incorrect) behavior, users observing
	// any regressions can use the DisableCPUQuotaWithExclusiveCPUs feature gate (default on) to
	// restore the old behavior. Please file issues if you hit issues and have to use this Feature Gate.
	// The Feature Gate will be locked to true and then removed in +2 releases (1.35) if there are no bug reported
	DisableCPUQuotaWithExclusiveCPUs featuregate.Feature = "DisableCPUQuotaWithExclusiveCPUs"

	// owner: @HirazawaUi
	// kep: http://kep.k8s.io/4004
	//
	// DisableNodeKubeProxyVersion disable the status.nodeInfo.kubeProxyVersion field of v1.Node
	DisableNodeKubeProxyVersion featuregate.Feature = "DisableNodeKubeProxyVersion"

	// owner: @pohly
	// kep: http://kep.k8s.io/4381
	//
	// Enables support for resources with custom parameters and a lifecycle
	// that is independent of a Pod. Resource allocation is done by the scheduler
	// based on "structured parameters".
	DynamicResourceAllocation featuregate.Feature = "DynamicResourceAllocation"

	// owner: @HirazawaUi
	// kep: http://kep.k8s.io/3721
	//
	// Allow containers to read environment variables from a file.
	// Environment variables file must be produced by an initContainer and located within an emptyDir volume.
	// The kubelet will populate the environment variables in the container
	// from the specified file in the emptyDir volume, without mounting the file.
	EnvFiles featuregate.Feature = "EnvFiles"

	// owner: @harche
	// kep: http://kep.k8s.io/3386
	//
	// Allows using event-driven PLEG (pod lifecycle event generator) through kubelet
	// which avoids frequent relisting of containers which helps optimize performance.
	EventedPLEG featuregate.Feature = "EventedPLEG"

	// owner: @andrewsykim @SergeyKanzhelev
	//
	// Ensure kubelet respects exec probe timeouts. Feature gate exists in-case existing workloads
	// may depend on old behavior where exec probe timeouts were ignored.
	// Locked to default, will remove in v1.38. Progress is reflected in KEP #1972 update
	ExecProbeTimeout featuregate.Feature = "ExecProbeTimeout"

	// owner: @HarshalNeelkamal
	//
	// Enables external service account JWT signing and key management.
	// If enabled, it allows passing --service-account-signing-endpoint flag to configure external signer.
	ExternalServiceAccountTokenSigner featuregate.Feature = "ExternalServiceAccountTokenSigner"

	// owner: @erictune @wojtek-t
	//
	// Enables support for gang scheduling in kube-scheduler.
	GangScheduling featuregate.Feature = "GangScheduling"

	// owner: @erictune @wojtek-t
	//
	// Enables support for generic Workload API.
	GenericWorkload featuregate.Feature = "GenericWorkload"

	// owner: @vinayakankugoyal @thockin
	//
	// Controls if the gitRepo volume plugin is supported or not.
	// KEP #5040 disables the gitRepo volume plugin by default starting v1.33,
	// this provides a way for users to override it.
	GitRepoVolumeDriver featuregate.Feature = "GitRepoVolumeDriver"

	// owner: @bobbypage
	// Adds support for kubelet to detect node shutdown and gracefully terminate pods prior to the node being shutdown.
	GracefulNodeShutdown featuregate.Feature = "GracefulNodeShutdown"

	// owner: @wzshiming
	// Make the kubelet use shutdown configuration based on pod priority values for graceful shutdown.
	GracefulNodeShutdownBasedOnPodPriority featuregate.Feature = "GracefulNodeShutdownBasedOnPodPriority"

	// owner: @jm-franc
	// kep: https://kep.k8s.io/4951
	//
	// Enables support of configurable HPA scale-up and scale-down tolerances.
	HPAConfigurableTolerance featuregate.Feature = "HPAConfigurableTolerance"

	// owner: @dxist
	//
	// Enables support of HPA scaling to zero pods when an object or custom metric is configured.
	HPAScaleToZero featuregate.Feature = "HPAScaleToZero"

	// owner: @deepakkinni @xing-yang
	// kep: https://kep.k8s.io/2644
	//
	// Honor Persistent Volume Reclaim Policy when it is "Delete" irrespective of PV-PVC
	// deletion ordering.
	HonorPVReclaimPolicy featuregate.Feature = "HonorPVReclaimPolicy"

	// owner: @HirazawaUi
	// kep: https://kep.k8s.io/4762
	//
	// Allows setting any FQDN as the pod's hostname
	HostnameOverride featuregate.Feature = "HostnameOverride"

	// owner: @haircommander
	// kep: http://kep.k8s.io/4210
	// ImageMaximumGCAge enables the Kubelet configuration field of the same name, allowing an admin
	// to specify the age after which an image will be garbage collected.
	ImageMaximumGCAge featuregate.Feature = "ImageMaximumGCAge"

	// owner: @saschagrunert
	// kep: https://kep.k8s.io/4639
	//
	// Enables the image volume source.
	ImageVolume featuregate.Feature = "ImageVolume"

	// owner: @iholder101
	// kep: https://kep.k8s.io/4639
	//
	// Enables adding the ImageVolume's digest to the pod's status.
	ImageVolumeWithDigest featuregate.Feature = "ImageVolumeWithDigest"

	// owner: @ndixita
	// kep: https://kep.k8s.io/5419
	//
	// Enables specifying resources at pod-level.
	InPlacePodLevelResourcesVerticalScaling featuregate.Feature = "InPlacePodLevelResourcesVerticalScaling"

	// owner: @vinaykul,@tallclair
	// kep: http://kep.k8s.io/1287
	//
	// Enables In-Place Pod Vertical Scaling
	InPlacePodVerticalScaling featuregate.Feature = "InPlacePodVerticalScaling"

	// owner: @tallclair
	// kep: http://kep.k8s.io/1287
	//
	// Deprecated: This feature gate is no longer used.
	// Was: Enables the AllocatedResources field in container status. This feature requires
	// InPlacePodVerticalScaling also be enabled.
	InPlacePodVerticalScalingAllocatedStatus featuregate.Feature = "InPlacePodVerticalScalingAllocatedStatus"

	// owner: @tallclair @esotsal
	//
	// Allow resource resize for containers in Guaranteed pods with integer CPU requests ( default false ).
	// Applies only in nodes with InPlacePodVerticalScaling and CPU Manager features enabled, and
	// CPU Manager Static Policy option set.
	InPlacePodVerticalScalingExclusiveCPUs featuregate.Feature = "InPlacePodVerticalScalingExclusiveCPUs"

	// owner: @tallclair @pkrishn
	//
	// Allow memory resize for containers in Guaranteed pods (default false) when Memory Manager Policy is set to Static.
	// Applies only in nodes with InPlacePodVerticalScaling and Memory Manager features enabled.
	InPlacePodVerticalScalingExclusiveMemory featuregate.Feature = "InPlacePodVerticalScalingExclusiveMemory"

	// owner: @trierra
	//
	// Disables the Portworx in-tree driver.
	InTreePluginPortworxUnregister featuregate.Feature = "InTreePluginPortworxUnregister"

	// owner: @mimowo
	// kep: https://kep.k8s.io/3850
	//
	// Allows users to specify counting of failed pods per index.
	JobBackoffLimitPerIndex featuregate.Feature = "JobBackoffLimitPerIndex"

	// owner: @mimowo
	// kep: https://kep.k8s.io/4368
	//
	// Allows to delegate reconciliation of a Job object to an external controller.
	JobManagedBy featuregate.Feature = "JobManagedBy"

	// owner: @kannon92
	// kep : https://kep.k8s.io/3939
	//
	// Allow users to specify recreating pods of a job only when
	// pods have fully terminated.
	JobPodReplacementPolicy featuregate.Feature = "JobPodReplacementPolicy"

	// owner: @tenzen-y
	// kep: https://kep.k8s.io/3998
	//
	// Allow users to specify when a Job can be declared as succeeded
	// based on the set of succeeded pods.
	JobSuccessPolicy featuregate.Feature = "JobSuccessPolicy"

	// owner: @marquiz
	// kep: http://kep.k8s.io/4033
	//
	// Enable detection of the kubelet cgroup driver configuration option from
	// the CRI.  The CRI runtime also needs to support this feature in which
	// case the kubelet will ignore the cgroupDriver (--cgroup-driver)
	// configuration option. If runtime doesn't support it, the kubelet will
	// fallback to using it's cgroupDriver option.
	KubeletCgroupDriverFromCRI featuregate.Feature = "KubeletCgroupDriverFromCRI"

	// owner: @lauralorenz @hankfreund
	// kep: https://kep.k8s.io/5593
	//
	// Enables support for configurable per-node backoff maximums for restarting
	// containers (aka containers in CrashLoopBackOff)
	KubeletCrashLoopBackOffMax featuregate.Feature = "KubeletCrashLoopBackOffMax"

	// owner: @stlaz
	// kep: https://kep.k8s.io/2535
	//
	// Enables tracking credentials for image pulls in order to authorize image
	// access for different tenants.
	KubeletEnsureSecretPulledImages featuregate.Feature = "KubeletEnsureSecretPulledImages"

	// owner: @vinayakankugoyal
	// kep: http://kep.k8s.io/2862
	//
	// Enable fine-grained kubelet API authorization for webhook based
	// authorization.
	KubeletFineGrainedAuthz featuregate.Feature = "KubeletFineGrainedAuthz"

	// owner: @AkihiroSuda
	//
	// Enables support for running kubelet in a user namespace.
	// The user namespace has to be created before running kubelet.
	// All the node components such as CRI need to be running in the same user namespace.
	KubeletInUserNamespace featuregate.Feature = "KubeletInUserNamespace"

	// KubeletPSI enables Kubelet to surface PSI metrics
	// owner: @roycaihw
	// kep: https://kep.k8s.io/4205
	KubeletPSI featuregate.Feature = "KubeletPSI"

	// owner: @moshe010
	//
	// Enable POD resources API to return resources allocated by Dynamic Resource Allocation
	KubeletPodResourcesDynamicResources featuregate.Feature = "KubeletPodResourcesDynamicResources"

	// owner: @moshe010
	//
	// Enable POD resources API with Get method
	KubeletPodResourcesGet featuregate.Feature = "KubeletPodResourcesGet"

	// owner: @ffromani
	// Deprecated: v1.34
	//
	// issue: https://github.com/kubernetes/kubernetes/issues/119423
	// Disables restricted output for the podresources API list endpoint.
	// "Restricted" output only includes the pods which are actually running and thus they
	// hold resources. Turns out this was originally the intended behavior, see:
	// https://github.com/kubernetes/kubernetes/pull/79409#issuecomment-505975671
	// This behavior was lost over time and interaction with memory manager creates
	// an unfixable bug because the endpoint returns spurious stale information the clients
	// cannot filter out, because the API doesn't provide enough context. See:
	// https://github.com/kubernetes/kubernetes/issues/132020
	// The endpoint has returning extra information for long time, but that information
	// is also useless for the purpose of this API. Nevertheless, we are changing a long-established
	// albeit buggy behavior, so users observing any regressions can use the
	// KubeletPodResourcesListUseActivePods/ feature gate (default on) to restore the old behavior.
	// Please file issues if you hit issues and have to use this Feature Gate.
	// The Feature Gate will be locked to true in +4 releases (1.38) and then removed (1.39)
	// if there are no bug reported.
	KubeletPodResourcesListUseActivePods featuregate.Feature = "KubeletPodResourcesListUseActivePods"

	// owner: @hoskeri
	//
	// Restores previous behavior where Kubelet fails self registration if node create returns 403 Forbidden.
	KubeletRegistrationGetOnExistsOnly featuregate.Feature = "KubeletRegistrationGetOnExistsOnly"

	// owner: @kannon92
	// kep: https://kep.k8s.io/4191
	//
	// The split image filesystem feature enables kubelet to perform garbage collection
	// of images (read-only layers) and/or containers (writeable layers) deployed on
	// separate filesystems.
	KubeletSeparateDiskGC featuregate.Feature = "KubeletSeparateDiskGC"

	// owner: @aramase
	// kep: http://kep.k8s.io/4412
	//
	// Enable kubelet to send the service account token bound to the pod for which the image
	// is being pulled to the credential provider plugin.
	KubeletServiceAccountTokenForCredentialProviders featuregate.Feature = "KubeletServiceAccountTokenForCredentialProviders"

	// owner: @sallyom
	// kep: https://kep.k8s.io/2832
	//
	// Add support for distributed tracing in the kubelet
	KubeletTracing featuregate.Feature = "KubeletTracing"

	// owner: @RobertKrawitz
	//
	// Allow use of filesystems for ephemeral storage monitoring.
	// Only applies if LocalStorageCapacityIsolation is set.
	// Relies on UserNamespacesSupport feature, and thus should follow it when setting defaults.
	LocalStorageCapacityIsolationFSQuotaMonitoring featuregate.Feature = "LocalStorageCapacityIsolationFSQuotaMonitoring"

	// owner: @damemi
	//
	// Enables scaling down replicas via logarithmic comparison of creation/ready timestamps
	LogarithmicScaleDown featuregate.Feature = "LogarithmicScaleDown"

	// owner: @sanposhiho
	// kep: https://kep.k8s.io/3633
	//
	// Enables the MatchLabelKeys and MismatchLabelKeys in PodAffinity and PodAntiAffinity.
	MatchLabelKeysInPodAffinity featuregate.Feature = "MatchLabelKeysInPodAffinity"

	// owner: @denkensk
	// kep: https://kep.k8s.io/3243
	//
	// Enable MatchLabelKeys in PodTopologySpread.
	MatchLabelKeysInPodTopologySpread featuregate.Feature = "MatchLabelKeysInPodTopologySpread"

	// owner: @mochizuki875
	// kep: https://kep.k8s.io/3243
	//
	// Enable merging key-value labels into LabelSelector corresponding to MatchLabelKeys in PodTopologySpread.
	MatchLabelKeysInPodTopologySpreadSelectorMerge featuregate.Feature = "MatchLabelKeysInPodTopologySpreadSelectorMerge"

	// owner: @krmayankk
	// kep: https://kep.k8s.io/961
	//
	// Enables maxUnavailable for StatefulSet
	MaxUnavailableStatefulSet featuregate.Feature = "MaxUnavailableStatefulSet"

	// owner: @cynepco3hahue(alukiano) @cezaryzukowski @k-wiatrzyk, @Tal-or (only for GA graduation)
	//
	// Allows setting memory affinity for a container based on NUMA topology
	MemoryManager featuregate.Feature = "MemoryManager"

	// owner: @xiaoxubeii
	// kep: https://kep.k8s.io/2570
	//
	// Enables kubelet to support memory QoS with cgroups v2.
	MemoryQoS featuregate.Feature = "MemoryQoS"

	// owner: @aojea
	// kep: https://kep.k8s.io/1880
	//
	// Enables the dynamic configuration of Service IP ranges
	MultiCIDRServiceAllocator featuregate.Feature = "MultiCIDRServiceAllocator"

	// owner: torredil
	// kep: https://kep.k8s.io/4876
	//
	// Makes CSINode.Spec.Drivers[*].Allocatable.Count mutable, allowing CSI drivers to
	// update the number of volumes that can be allocated on a node
	MutableCSINodeAllocatableCount featuregate.Feature = "MutableCSINodeAllocatableCount"

	// owner: huww98
	// kep: https://kep.k8s.io/5381
	//
	// Makes PersistentVolume.Spec.NodeAffinity mutable, allowing CSI drivers to
	// update the topology info when the data is migrated
	MutablePVNodeAffinity featuregate.Feature = "MutablePVNodeAffinity"

	// owner: @kannon92
	// kep: https://kep.k8s.io/5440
	//
	// Enables mutable pod resources for suspended Jobs, regardless of whether they have started before.
	MutablePodResourcesForSuspendedJobs featuregate.Feature = "MutablePodResourcesForSuspendedJobs"

	// owner: @mimowo
	// kep: https://kep.k8s.io/5440
	//
	// Enables mutable scheduling directives for suspended Jobs, regardless of whether they have started before.
	MutableSchedulingDirectivesForSuspendedJobs featuregate.Feature = "MutableSchedulingDirectivesForSuspendedJobs"

	// owner: @danwinship
	// kep: https://kep.k8s.io/3866
	//
	// Allows running kube-proxy with `--mode nftables`.
	NFTablesProxyMode featuregate.Feature = "NFTablesProxyMode"

	// owner: @pravk03, @tallclair
	// kep: https://kep.k8s.io/5328
	//
	// Enables the DeclaredFeatures API in the NodeStatus, populated by the Kubelet. Also enables the scheduler filter using DeclaredFeatures.
	NodeDeclaredFeatures featuregate.Feature = "NodeDeclaredFeatures"

	// owner: @kerthcet
	// kep: https://kep.k8s.io/3094
	//
	// Allow users to specify whether to take nodeAffinity/nodeTaint into consideration when
	// calculating pod topology spread skew.
	NodeInclusionPolicyInPodTopologySpread featuregate.Feature = "NodeInclusionPolicyInPodTopologySpread"

	// owner: @aravindhp @LorbusChris
	// kep: http://kep.k8s.io/2271
	//
	// Enables querying logs of node services using the /logs endpoint. Enabling this feature has security implications.
	// The recommendation is to enable it on a need basis for debugging purposes and disabling otherwise.
	NodeLogQuery featuregate.Feature = "NodeLogQuery"

	// owner: @iholder101 @kannon92
	// kep: https://kep.k8s.io/2400
	//
	// Permits kubelet to run with swap enabled.
	NodeSwap featuregate.Feature = "NodeSwap"

	// owner: @sanposhiho, @wojtek-t
	// kep: https://kep.k8s.io/5278
	//
	// Extends NominatedNodeName field to express expected pod placement, allowing
	// both the scheduler and external components (e.g., Cluster Autoscaler, Karpenter, Kueue)
	// to share pod placement intentions. This enables better coordination between
	// components, prevents inappropriate node scale-downs, and helps the scheduler
	// resume work after restarts.
	NominatedNodeNameForExpectation featuregate.Feature = "NominatedNodeNameForExpectation"

	// owner: @bwsalmon
	// kep: https://kep.k8s.io/5598
	//
	// Enables opportunistic batching in the scheduler.
	OpportunisticBatching featuregate.Feature = "OpportunisticBatching"

	// owner: @cici37
	// kep: https://kep.k8s.io/5080
	//
	// Enables ordered namespace deletion.
	OrderedNamespaceDeletion featuregate.Feature = "OrderedNamespaceDeletion"

	// owner: @haircommander
	// kep: https://kep.k8s.io/2364
	//
	// Configures the Kubelet to use the CRI to populate pod and container stats, instead of supplimenting with stats from cAdvisor.
	// Requires the CRI implementation supports supplying the required stats.
	PodAndContainerStatsFromCRI featuregate.Feature = "PodAndContainerStatsFromCRI"

	// owner: @ahmedtd
	// kep: https://kep.k8s.io/4317
	//
	// Enable PodCertificateRequest objects and podCertificate projected volume sources.
	PodCertificateRequest featuregate.Feature = "PodCertificateRequest"

	// owner: @ahg-g
	//
	// Enables controlling pod ranking on replicaset scale-down.
	PodDeletionCost featuregate.Feature = "PodDeletionCost"

	// owner: @ndixita
	// key: https://kep.k8s.io/2837
	//
	// Enables specifying resources at pod-level.
	PodLevelResources featuregate.Feature = "PodLevelResources"

	// owner: @AxeZhan
	// kep: http://kep.k8s.io/3960
	//
	// Enables SleepAction in container lifecycle hooks
	PodLifecycleSleepAction featuregate.Feature = "PodLifecycleSleepAction"

	// owner: @sreeram-venkitesh
	// kep: http://kep.k8s.io/4818
	//
	// Allows zero value for sleep duration in SleepAction in container lifecycle hooks
	PodLifecycleSleepActionAllowZero featuregate.Feature = "PodLifecycleSleepActionAllowZero"

	// owner: @knight42
	// kep: https://kep.k8s.io/3288
	//
	// Enables only stdout or stderr of the container to be retrievd.
	PodLogsQuerySplitStreams featuregate.Feature = "PodLogsQuerySplitStreams"

	// owner: @natasha41575
	// kep: http://kep.k8s.io/5067
	//
	// Enables the pod to report status.ObservedGeneration to reflect the generation of the last observed podspec.
	PodObservedGenerationTracking featuregate.Feature = "PodObservedGenerationTracking"

	// owner: @ddebroy, @kannon92
	//
	// Enables reporting of PodReadyToStartContainersCondition condition in pod status after pod
	// sandbox creation and network configuration completes successfully
	PodReadyToStartContainersCondition featuregate.Feature = "PodReadyToStartContainersCondition"

	// owner: @Huang-Wei
	// kep: https://kep.k8s.io/3521
	//
	// Enable users to specify when a Pod is ready for scheduling.
	PodSchedulingReadiness featuregate.Feature = "PodSchedulingReadiness"

	// owner: @munnerz
	// kep: https://kep.k8s.io/4742
	// alpha: v1.33
	// beta: v1.35
	//
	// Enables the PodTopologyLabelsAdmission admission plugin that mutates `pod/binding`
	// requests by copying the `topology.kubernetes.io/{zone,region}` labels from the assigned
	// Node object (in the Binding being admitted) onto the Binding
	// so that it can be persisted onto the Pod object when the Pod is being scheduled.
	// This allows workloads running in pods to understand the topology information of their assigned node.
	// Enabling this feature also permits external schedulers to set labels on pods in an atomic
	// operation when scheduling a Pod by setting the `metadata.labels` field on the submitted Binding,
	// similar to how `metadata.annotations` behaves.
	PodTopologyLabelsAdmission featuregate.Feature = "PodTopologyLabelsAdmission"

	// owner: @seans3
	// kep: http://kep.k8s.io/4006
	//
	// Enables PortForward to be proxied with a websocket client
	PortForwardWebsockets featuregate.Feature = "PortForwardWebsockets"

	// owner: @danwinship
	// kep: https://kep.k8s.io/3015
	//
	// Enables PreferSameZone and PreferSameNode values for trafficDistribution
	PreferSameTrafficDistribution featuregate.Feature = "PreferSameTrafficDistribution"

	// owner: @sreeram-venkitesh
	//
	// Denies pod admission if static pods reference other API objects.
	PreventStaticPodAPIReferences featuregate.Feature = "PreventStaticPodAPIReferences"

	// owner: @jessfraz
	//
	// Enables control over ProcMountType for containers.
	ProcMountType featuregate.Feature = "ProcMountType"

	// owner: @sjenning
	//
	// Allows resource reservations at the QoS level preventing pods at lower QoS levels from
	// bursting into resources requested at higher QoS levels (memory only for now)
	QOSReserved featuregate.Feature = "QOSReserved"

	// owner: @gnufied
	// kep: https://kep.k8s.io/1790
	//
	// Allow users to recover from volume expansion failure
	RecoverVolumeExpansionFailure featuregate.Feature = "RecoverVolumeExpansionFailure"

	// owner: @AkihiroSuda
	// kep: https://kep.k8s.io/3857
	//
	// Allows recursive read-only mounts.
	RecursiveReadOnlyMounts featuregate.Feature = "RecursiveReadOnlyMounts"

	// owner: @lauralorenz
	// kep: https://kep.k8s.io/4603
	//
	// Enables support for a lower internal cluster-wide backoff maximum for restarting
	// containers (aka containers in CrashLoopBackOff)
	ReduceDefaultCrashLoopBackOffDecay featuregate.Feature = "ReduceDefaultCrashLoopBackOffDecay"

	// owner: @adrianmoisey
	// kep: https://kep.k8s.io/4427
	//
	// Relaxed DNS search string validation.
	RelaxedDNSSearchValidation featuregate.Feature = "RelaxedDNSSearchValidation"

	// owner: @HirazawaUi
	// kep: https://kep.k8s.io/4369
	//
	// Allow almost all printable ASCII characters in environment variables
	RelaxedEnvironmentVariableValidation featuregate.Feature = "RelaxedEnvironmentVariableValidation"

	// owner: @adrianmoisey
	// kep: https://kep.k8s.io/5311
	//
	// Relaxed DNS search string validation.
	RelaxedServiceNameValidation featuregate.Feature = "RelaxedServiceNameValidation"

	// owner: @zhangweikop
	//
	// Enable kubelet tls server to update certificate if the specified certificate files are changed.
	// This feature is useful when specifying tlsCertFile & tlsPrivateKeyFile in kubelet Configuration.
	// No effect for other cases such as using serverTLSbootstap.
	ReloadKubeletServerCertificateFile featuregate.Feature = "ReloadKubeletServerCertificateFile"

	// owner: @SergeyKanzhelev
	// kep: https://kep.k8s.io/4680
	//
	// Adds the AllocatedResourcesStatus to the container status.
	ResourceHealthStatus featuregate.Feature = "ResourceHealthStatus"

	// owner: @yuanwang04
	// kep: https://kep.k8s.io/5532
	//
	// Restart the pod in-place on the same node.
	RestartAllContainersOnContainerExits featuregate.Feature = "RestartAllContainersOnContainerExits"

	// owner: @mikedanese
	//
	// Gets a server certificate for the kubelet from the Certificate Signing
	// Request API instead of generating one self signed and auto rotates the
	// certificate as expiration approaches.
	RotateKubeletServerCertificate featuregate.Feature = "RotateKubeletServerCertificate"

	// owner: @kiashok
	// kep: https://kep.k8s.io/4216
	//
	// Adds support to pull images based on the runtime class specified.
	RuntimeClassInImageCriAPI featuregate.Feature = "RuntimeClassInImageCriApi"

	// owner: @jsafrane
	// kep: https://kep.k8s.io/1710
	//
	// Speed up container startup by mounting volumes with the correct SELinux label
	// instead of changing each file on the volumes recursively.
	// Enables the SELinuxChangePolicy field in PodSecurityContext before SELinuxMount featgure gate is enabled.
	SELinuxChangePolicy featuregate.Feature = "SELinuxChangePolicy"

	// owner: @jsafrane
	// kep: https://kep.k8s.io/1710
	// Speed up container startup by mounting volumes with the correct SELinux label
	// instead of changing each file on the volumes recursively.
	SELinuxMount featuregate.Feature = "SELinuxMount"

	// owner: @jsafrane
	// kep: https://kep.k8s.io/1710
	// Speed up container startup by mounting volumes with the correct SELinux label
	// instead of changing each file on the volumes recursively.
	// Initial implementation focused on ReadWriteOncePod volumes.
	SELinuxMountReadWriteOncePod featuregate.Feature = "SELinuxMountReadWriteOncePod"

	// owner: @macsko
	// kep: http://kep.k8s.io/5229
	//
	// Makes all API calls during scheduling asynchronous, by introducing a new kube-scheduler-wide way of handling such calls.
	SchedulerAsyncAPICalls featuregate.Feature = "SchedulerAsyncAPICalls"

	// owner: @sanposhiho
	// kep: http://kep.k8s.io/4832
	//
	// Running some expensive operation within the scheduler's preemption asynchronously,
	// which improves the scheduling latency when the preemption involves in.
	SchedulerAsyncPreemption featuregate.Feature = "SchedulerAsyncPreemption"

	// owner: @macsko
	// kep: http://kep.k8s.io/5142
	//
	// Improves scheduling queue behavior by popping pods from the backoffQ when the activeQ is empty.
	// This allows to process potentially schedulable pods ASAP, eliminating a penalty effect of the backoff queue.
	SchedulerPopFromBackoffQ featuregate.Feature = "SchedulerPopFromBackoffQ"

	// owner: @sanposhiho
	// kep: http://kep.k8s.io/4247
	//
	// Enables the scheduler's enhancement called QueueingHints,
	// which benefits to reduce the useless requeueing.
	SchedulerQueueingHints featuregate.Feature = "SchedulerQueueingHints"

	// owner: @atosatto @yuanchen8911
	// kep: http://kep.k8s.io/3902
	//
	// Decouples Taint Eviction Controller, performing taint-based Pod eviction, from Node Lifecycle Controller.
	SeparateTaintEvictionController featuregate.Feature = "SeparateTaintEvictionController"

	// owner: @aramase
	// kep: https://kep.k8s.io/4412
	//
	// ServiceAccountNodeAudienceRestriction is used to restrict the audience for which the
	// kubelet can request a service account token for.
	ServiceAccountNodeAudienceRestriction featuregate.Feature = "ServiceAccountNodeAudienceRestriction"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	//
	// Controls whether JTIs (UUIDs) are embedded into generated service account tokens, and whether these JTIs are
	// recorded into the audit log for future requests made by these tokens.
	ServiceAccountTokenJTI featuregate.Feature = "ServiceAccountTokenJTI"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	//
	// Controls whether the apiserver supports binding service account tokens to Node objects.
	ServiceAccountTokenNodeBinding featuregate.Feature = "ServiceAccountTokenNodeBinding"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	//
	// Controls whether the apiserver will validate Node claims in service account tokens.
	ServiceAccountTokenNodeBindingValidation featuregate.Feature = "ServiceAccountTokenNodeBindingValidation"

	// owner: @munnerz
	// kep: http://kep.k8s.io/4193
	//
	// Controls whether the apiserver embeds the node name and uid for the associated node when issuing
	// service account tokens bound to Pod objects.
	ServiceAccountTokenPodNodeInfo featuregate.Feature = "ServiceAccountTokenPodNodeInfo"

	// owner: @gauravkghildiyal @robscott
	// kep: https://kep.k8s.io/4444
	//
	// Enables trafficDistribution field on Services.
	ServiceTrafficDistribution featuregate.Feature = "ServiceTrafficDistribution"

	// owner: @gjkim42 @SergeyKanzhelev @matthyx @tzneal
	// kep: http://kep.k8s.io/753
	//
	// Introduces sidecar containers, a new type of init container that starts
	// before other containers but remains running for the full duration of the
	// pod's lifecycle and will not block pod termination.
	SidecarContainers featuregate.Feature = "SidecarContainers"

	// owner: @liggitt
	//
	// Mitigates spurious statefulset rollouts due to controller revision comparison mismatches
	// which are not semantically significant (e.g. serialization differences or missing defaulted fields).
	StatefulSetSemanticRevisionComparison featuregate.Feature = "StatefulSetSemanticRevisionComparison"

	// owner: @cupnes
	// kep: https://kep.k8s.io/4049
	//
	// Enables scoring nodes by available storage capacity with
	// StorageCapacityScoring feature gate.
	StorageCapacityScoring featuregate.Feature = "StorageCapacityScoring"

	// owner: @ahutsunshine
	//
	// Allows namespace indexer for namespace scope resources in apiserver cache to accelerate list operations.
	// Superseded by BtreeWatchCache.
	StorageNamespaceIndex featuregate.Feature = "StorageNamespaceIndex"

	// owner: @enj, @michaelasp
	// kep: https://kep.k8s.io/4192
	//
	// Enables support for the StorageVersionMigrator controller.
	StorageVersionMigrator featuregate.Feature = "StorageVersionMigrator"

	// owner: @serathius
	// Allow API server JSON encoder to encode collections item by item, instead of all at once.
	StreamingCollectionEncodingToJSON featuregate.Feature = "StreamingCollectionEncodingToJSON"

	// owner: serathius
	// Allow API server Protobuf encoder to encode collections item by item, instead of all at once.
	StreamingCollectionEncodingToProtobuf featuregate.Feature = "StreamingCollectionEncodingToProtobuf"

	// owner: @danwinship
	// kep: https://kep.k8s.io/4858
	//
	// Requires stricter validation of IP addresses and CIDR values in API objects.
	StrictIPCIDRValidation featuregate.Feature = "StrictIPCIDRValidation"

	// owner: @everpeace
	// kep: https://kep.k8s.io/3619
	//
	// Enable SupplementalGroupsPolicy feature in PodSecurityContext
	SupplementalGroupsPolicy featuregate.Feature = "SupplementalGroupsPolicy"

	// owner: @zhifei92
	//
	// Enables the systemd watchdog for the kubelet. When enabled, the kubelet will
	// periodically notify the systemd watchdog to indicate that it is still alive.
	// This can help prevent the system from restarting the kubelet if it becomes
	// unresponsive. The feature gate is enabled by default, but should only be used
	// if the system supports the systemd watchdog feature and has it configured properly.
	SystemdWatchdog = featuregate.Feature("SystemdWatchdog")

	// owner: @helayoty
	// kep: https://kep.k8s.io/5471
	//
	// Enables numeric comparison operators (Lt, Gt) for tolerations to match taints with threshold-based values.
	TaintTolerationComparisonOperators featuregate.Feature = "TaintTolerationComparisonOperators"

	// owner: @robscott
	// kep: https://kep.k8s.io/2433
	//
	// Enables topology aware hints for EndpointSlices
	TopologyAwareHints featuregate.Feature = "TopologyAwareHints"

	// owner: @PiotrProkop
	// kep: https://kep.k8s.io/3545
	//
	// Allow fine-tuning of topology manager policies with alpha options.
	// This feature gate:
	// - will guard *a group* of topology manager options whose quality level is alpha.
	// - will never graduate to beta or stable.
	TopologyManagerPolicyAlphaOptions featuregate.Feature = "TopologyManagerPolicyAlphaOptions"

	// owner: @PiotrProkop
	// kep: https://kep.k8s.io/3545
	//
	// Allow fine-tuning of topology manager policies with beta options.
	// This feature gate:
	// - will guard *a group* of topology manager options whose quality level is beta.
	// - is thus *introduced* as beta
	// - will never graduate to stable.
	TopologyManagerPolicyBetaOptions featuregate.Feature = "TopologyManagerPolicyBetaOptions"

	// owner: @PiotrProkop
	// kep: https://kep.k8s.io/3545
	//
	// Allow the usage of options to fine-tune the topology manager policies.
	TopologyManagerPolicyOptions featuregate.Feature = "TopologyManagerPolicyOptions"

	// owner: @seans3
	// kep: http://kep.k8s.io/4006
	//
	// Enables StreamTranslator proxy to handle WebSockets upgrade requests for the
	// version of the RemoteCommand subprotocol that supports the "close" signal.
	TranslateStreamCloseWebsocketRequests featuregate.Feature = "TranslateStreamCloseWebsocketRequests"

	// owner: @HirazawaUi
	// kep: https://kep.k8s.io/5607
	//
	// Allow hostNetwork pods to use user namespaces
	UserNamespacesHostNetworkSupport featuregate.Feature = "UserNamespacesHostNetworkSupport"

	// owner: @rata, @giuseppe
	// kep: https://kep.k8s.io/127
	//
	// Enables user namespace support for stateless pods.
	UserNamespacesSupport featuregate.Feature = "UserNamespacesSupport"

	// owner: @mattcarry, @sunnylovestiramisu
	// kep: https://kep.k8s.io/3751
	//
	// Enables user specified volume attributes for persistent volumes, like iops and throughput.
	VolumeAttributesClass featuregate.Feature = "VolumeAttributesClass"

	// owner: @gnufied
	// kep: https://kep.k8s.io/5030
	//
	// Enables volume limit scaling for CSI drivers. This allows scheduler to
	// co-ordinate better with cluster-autoscaler for storage limits.
	VolumeLimitScaling featuregate.Feature = "VolumeLimitScaling"

	// owner: @ksubrmnn
	//
	// Allows kube-proxy to create DSR loadbalancers for Windows
	WinDSR featuregate.Feature = "WinDSR"

	// owner: @ksubrmnn
	//
	// Allows kube-proxy to run in Overlay mode for Windows
	WinOverlay featuregate.Feature = "WinOverlay"

	// owner: @jsturtevant
	// kep: https://kep.k8s.io/4888
	//
	// Add CPU and Memory Affinity support to Windows nodes with CPUManager, MemoryManager and Topology manager
	WindowsCPUAndMemoryAffinity featuregate.Feature = "WindowsCPUAndMemoryAffinity"

	// owner: @zylxjtu
	// kep: https://kep.k8s.io/4802
	//
	// Enables support for graceful shutdown windows node.
	WindowsGracefulNodeShutdown featuregate.Feature = "WindowsGracefulNodeShutdown"

	// owner: @marosset
	// kep: https://kep.k8s.io/3503
	//
	// Enables support for joining Windows containers to a hosts' network namespace.
	WindowsHostNetwork featuregate.Feature = "WindowsHostNetwork"
)

// defaultVersionedKubernetesFeatureGates consists of all known Kubernetes-specific feature keys with VersionedSpecs.
// To add a new feature, define a key for it in pkg/features/kube_features.go and add it here. The features will be
// available throughout Kubernetes binaries.
// For features available via specific kubernetes components like apiserver,
// cloud-controller-manager, etc find the respective kube_features.go file
// (eg:staging/src/apiserver/pkg/features/kube_features.go), define the versioned
// feature gate there, and reference it in this file.
// To support n-3 compatibility version, features may only be removed 3 releases after graduation.
//
// Entries are alphabetized.
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	AllowDNSOnlyNodeCSR: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Deprecated},
	},

	AllowInsecureKubeletCertificateSigningRequests: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Deprecated},
	},

	AllowOverwriteTerminationGracePeriodSeconds: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Deprecated},
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Deprecated, LockToDefault: true}, // remove in 1.38
	},

	AnyVolumeDataSource: {
		{Version: version.MustParse("1.18"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.33 -> remove in 1.36
	},

	AuthorizeNodeWithSelectors: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	AuthorizePodWebsocketUpgradeCreatePermission: {
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	CPUCFSQuotaPeriod: {
		{Version: version.MustParse("1.12"), Default: false, PreRelease: featuregate.Alpha},
	},

	CPUManagerPolicyAlphaOptions: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
	},

	CPUManagerPolicyBetaOptions: {
		{Version: version.MustParse("1.23"), Default: true, PreRelease: featuregate.Beta},
	},

	CPUManagerPolicyOptions: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.23"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.36
	},

	CSIMigrationPortworx: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},                    // On by default (requires Portworx CSI driver)
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.36
	},

	CSIServiceAccountTokenSecrets: {
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	CSIVolumeHealth: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
	},

	ChangeContainerStatusOnKubeletRestart: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Deprecated},
	},

	ClearingNominatedNodeNameAfterBinding: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	ClusterTrustBundle: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
	},

	ClusterTrustBundleProjection: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
	},

	ContainerCheckpoint: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},

	ContainerRestartRules: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	ContainerStopSignals: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
	},

	CrossNamespaceVolumeDataSource: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRAAdminAccess: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	DRAConsumableCapacity: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRADeviceBindingConditions: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRADeviceTaintRules: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRADeviceTaints: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRAExtendedResource: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRAPartitionableDevices: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRAPrioritizedList: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	DRAResourceClaimDeviceStatus: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	DRASchedulerFilterTimeout: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	DeploymentReplicaSetTerminatingReplicas: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	DisableAllocatorDualWrite: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove after MultiCIDRServiceAllocator is GA
	},

	DisableCPUQuotaWithExclusiveCPUs: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	DisableNodeKubeProxyVersion: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Deprecated},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Deprecated}, // lock to default in 1.34 and remove in v1.37
	},

	DynamicResourceAllocation: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
		// TODO (https://github.com/kubernetes/kubernetes/issues/134459): remove completely in 1.38
	},
	EnvFiles: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},
	EventedPLEG: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
	},

	ExecProbeTimeout: {
		{Version: version.MustParse("1.20"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in v1.38
	},

	ExternalServiceAccountTokenSigner: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	GangScheduling: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	GenericWorkload: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	GitRepoVolumeDriver: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Deprecated},
	},

	GracefulNodeShutdown: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.21"), Default: true, PreRelease: featuregate.Beta},
	},

	GracefulNodeShutdownBasedOnPodPriority: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
	},

	HPAConfigurableTolerance: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	HPAScaleToZero: {
		{Version: version.MustParse("1.16"), Default: false, PreRelease: featuregate.Alpha},
	},

	HonorPVReclaimPolicy: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.36
	},

	HostnameOverride: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	ImageMaximumGCAge: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.38
	},

	ImageVolume: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	ImageVolumeWithDigest: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	InPlacePodLevelResourcesVerticalScaling: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	InPlacePodVerticalScaling: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.38
	},

	InPlacePodVerticalScalingAllocatedStatus: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Deprecated}, // remove in 1.36
	},

	InPlacePodVerticalScalingExclusiveCPUs: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	InPlacePodVerticalScalingExclusiveMemory: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
	},

	InTreePluginPortworxUnregister: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha}, // remove it along with CSIMigrationPortworx in 1.36
	},

	JobBackoffLimitPerIndex: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.36
	},

	JobManagedBy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.38
	},

	JobPodReplacementPolicy: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	JobSuccessPolicy: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.36
	},

	KubeletCgroupDriverFromCRI: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	KubeletCrashLoopBackOffMax: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletEnsureSecretPulledImages: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletFineGrainedAuthz: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletInUserNamespace: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
	},

	KubeletPSI: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletPodResourcesDynamicResources: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletPodResourcesGet: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletPodResourcesListUseActivePods: {
		{Version: version.MustParse("1.0"), Default: false, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Deprecated}, // lock to default in 1.38, remove in 1.39
	},

	KubeletRegistrationGetOnExistsOnly: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Deprecated},
	},

	KubeletSeparateDiskGC: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletServiceAccountTokenForCredentialProviders: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	KubeletTracing: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	LocalStorageCapacityIsolationFSQuotaMonitoring: {
		{Version: version.MustParse("1.15"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},

	LogarithmicScaleDown: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.22"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	MatchLabelKeysInPodAffinity: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	MatchLabelKeysInPodTopologySpread: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
	},

	MatchLabelKeysInPodTopologySpreadSelectorMerge: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	MaxUnavailableStatefulSet: {
		{Version: version.MustParse("1.24"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	MemoryManager: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.22"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	MemoryQoS: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
	},

	MultiCIDRServiceAllocator: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37 (locked to default in 1.34)
	},

	MutableCSINodeAllocatableCount: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	MutablePVNodeAffinity: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	MutablePodResourcesForSuspendedJobs: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	MutableSchedulingDirectivesForSuspendedJobs: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	NFTablesProxyMode: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	NodeDeclaredFeatures: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	NodeInclusionPolicyInPodTopologySpread: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	NodeLogQuery: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Beta},
	},

	NodeSwap: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	NominatedNodeNameForExpectation: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	OpportunisticBatching: {
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	OrderedNamespaceDeletion: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	PodAndContainerStatsFromCRI: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
	},

	PodCertificateRequest: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Beta},
	},

	PodDeletionCost: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.22"), Default: true, PreRelease: featuregate.Beta},
	},

	PodLevelResources: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	PodLifecycleSleepAction: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.34; remove in 1.37
	},

	PodLifecycleSleepActionAllowZero: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.34, remove in 1.37
	},

	PodLogsQuerySplitStreams: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	PodObservedGenerationTracking: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.35, remove in 1.38
	},

	PodReadyToStartContainersCondition: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},

	PodSchedulingReadiness: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.30; remove in 1.32
	},

	PodTopologyLabelsAdmission: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	PortForwardWebsockets: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},

	PreferSameTrafficDistribution: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	PreventStaticPodAPIReferences: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	ProcMountType: {
		{Version: version.MustParse("1.12"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	QOSReserved: {
		{Version: version.MustParse("1.11"), Default: false, PreRelease: featuregate.Alpha},
	},

	RecoverVolumeExpansionFailure: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.34; remove in 1.37
	},

	RecursiveReadOnlyMounts: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.36
	},

	ReduceDefaultCrashLoopBackOffDecay: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
	},

	RelaxedDNSSearchValidation: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	RelaxedEnvironmentVariableValidation: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	RelaxedServiceNameValidation: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
	},

	ReloadKubeletServerCertificateFile: {
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
	},

	ResourceHealthStatus: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
	},

	RestartAllContainersOnContainerExits: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	RotateKubeletServerCertificate: {
		{Version: version.MustParse("1.7"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.12"), Default: true, PreRelease: featuregate.Beta},
	},

	RuntimeClassInImageCriAPI: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
	},

	SELinuxChangePolicy: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	SELinuxMount: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
	},

	SELinuxMountReadWriteOncePod: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
	},

	SchedulerAsyncAPICalls: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Beta},
	},

	SchedulerAsyncPreemption: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	SchedulerPopFromBackoffQ: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	SchedulerQueueingHints: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	SeparateTaintEvictionController: {
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37 (locked to default in 1.34)
	},

	ServiceAccountNodeAudienceRestriction: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	ServiceAccountTokenJTI: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	ServiceAccountTokenNodeBinding: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	ServiceAccountTokenNodeBindingValidation: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	ServiceAccountTokenPodNodeInfo: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	ServiceTrafficDistribution: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA and LockToDefault in 1.33, remove in 1.36
	},

	SidecarContainers: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, LockToDefault: true, PreRelease: featuregate.GA}, // GA in 1.33 remove in 1.36
	},

	StatefulSetSemanticRevisionComparison: {
		// This is a mitigation for a 1.34 regression due to serialization differences that cannot be feature-gated,
		// so this mitigation should not auto-disable even if emulating versions prior to 1.34 with --emulation-version.
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.Beta},
	},

	StorageCapacityScoring: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
	},

	StorageNamespaceIndex: {
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Deprecated},
	},

	StorageVersionMigrator: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Beta},
	},

	StreamingCollectionEncodingToJSON: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	StreamingCollectionEncodingToProtobuf: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	StrictIPCIDRValidation: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
	},

	SupplementalGroupsPolicy: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.38
	},

	SystemdWatchdog: {
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remov in 1.37
	},

	TaintTolerationComparisonOperators: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	TopologyAwareHints: {
		{Version: version.MustParse("1.21"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	TopologyManagerPolicyAlphaOptions: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
	},

	TopologyManagerPolicyBetaOptions: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
	},

	TopologyManagerPolicyOptions: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA},
	},

	TranslateStreamCloseWebsocketRequests: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
	},

	UserNamespacesHostNetworkSupport: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	UserNamespacesSupport: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	VolumeAttributesClass: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	VolumeLimitScaling: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	WinDSR: {
		{Version: version.MustParse("1.14"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	WinOverlay: {
		{Version: version.MustParse("1.14"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.20"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	WindowsCPUAndMemoryAffinity: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	WindowsGracefulNodeShutdown: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	WindowsHostNetwork: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Deprecated},
	},

	apiextensionsfeatures.CRDObservedGenerationTracking: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Beta},
	},

	apiextensionsfeatures.CRDValidationRatcheting: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	apiextensionsfeatures.CustomResourceFieldSelectors: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, LockToDefault: true, PreRelease: featuregate.GA},
	},

	genericfeatures.APIResponseCompression: {
		{Version: version.MustParse("1.8"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.16"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.APIServerIdentity: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.APIServerTracing: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	genericfeatures.APIServingWithRoutine: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},

	genericfeatures.AggregatedDiscoveryRemoveBetaType: {
		{Version: version.MustParse("1.0"), Default: false, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Deprecated},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Deprecated, LockToDefault: true},
	},

	genericfeatures.AllowParsingUserUIDFromCertAuth: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.AllowUnsafeMalformedObjectDeletion: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	genericfeatures.AnonymousAuthConfigurableEndpoints: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	genericfeatures.AuthorizeWithSelectors: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	genericfeatures.BtreeWatchCache: {
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	genericfeatures.CBORServingAndStorage: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	genericfeatures.ConcurrentWatchObjectDecode: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},

	genericfeatures.ConsistentListFromCache: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	genericfeatures.ConstrainedImpersonation: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	genericfeatures.CoordinatedLeaderElection: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
	},

	genericfeatures.DeclarativeValidation: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.DeclarativeValidationTakeover: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
	},

	genericfeatures.DetectCacheInconsistency: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.KMSv1: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Deprecated},
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Deprecated},
	},

	genericfeatures.ListFromCacheSnapshot: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.MutatingAdmissionPolicy: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Beta},
	},

	genericfeatures.OpenAPIEnums: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.RemoteRequestHeaderUID: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.ResilientWatchCacheInitialization: {
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	genericfeatures.RetryGenerateName: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, LockToDefault: true, PreRelease: featuregate.GA},
	},

	genericfeatures.SeparateCacheWatchRPC: {
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Deprecated},
	},

	genericfeatures.SizeBasedListCostEstimate: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.StorageVersionAPI: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
	},

	genericfeatures.StorageVersionHash: {
		{Version: version.MustParse("1.14"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.15"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.StructuredAuthenticationConfiguration: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA and LockToDefault in 1.34, remove in 1.37
	},

	genericfeatures.StructuredAuthenticationConfigurationEgressSelector: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.StructuredAuthenticationConfigurationJWKSMetrics: {
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.StructuredAuthorizationConfiguration: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	genericfeatures.TokenRequestServiceAccountUIDValidation: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.UnauthenticatedHTTP2DOSMitigation: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},

	genericfeatures.UnknownVersionInteroperabilityProxy: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
	},

	genericfeatures.WatchCacheInitializationPostStartHook: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},

	genericfeatures.WatchFromStorageWithoutResourceVersion: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Deprecated, LockToDefault: true},
	},

	genericfeatures.WatchList: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		// switch this back to false because the json and proto streaming encoders appear to work better.
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	kcmfeatures.CloudControllerManagerWatchBasedRoutesReconciliation: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	kcmfeatures.CloudControllerManagerWebhook: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
	},

	zpagesfeatures.ComponentFlagz: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	zpagesfeatures.ComponentStatusz: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},
}

// defaultKubernetesFeatureGateDependencies enumerates the dependencies of any feature gate that
// depends on another. Dependencies ensure that a dependent feature gate can only be enabled if all
// of its dependencies are also enabled, and ensures a feature at a higher stability level cannot
// depend on a less stable feature.
//
// Entries are alphabetized.
var defaultKubernetesFeatureGateDependencies = map[featuregate.Feature][]featuregate.Feature{
	AllowDNSOnlyNodeCSR: {},

	AllowInsecureKubeletCertificateSigningRequests: {},

	AllowOverwriteTerminationGracePeriodSeconds: {},

	AnyVolumeDataSource: {},

	AuthorizeNodeWithSelectors: {genericfeatures.AuthorizeWithSelectors},

	AuthorizePodWebsocketUpgradeCreatePermission: {},

	CPUCFSQuotaPeriod: {},

	CPUManagerPolicyAlphaOptions: {},

	CPUManagerPolicyBetaOptions: {},

	CPUManagerPolicyOptions: {},

	CSIMigrationPortworx: {},

	CSIServiceAccountTokenSecrets: {},

	CSIVolumeHealth: {},

	ChangeContainerStatusOnKubeletRestart: {},

	ClearingNominatedNodeNameAfterBinding: {},

	ClusterTrustBundle: {},

	ClusterTrustBundleProjection: {ClusterTrustBundle},

	ContainerCheckpoint: {},

	ContainerRestartRules: {},

	ContainerStopSignals: {},

	CoordinatedLeaderElection: {},

	CrossNamespaceVolumeDataSource: {},

	DRAAdminAccess: {DynamicResourceAllocation},

	DRAConsumableCapacity: {DynamicResourceAllocation},

	DRADeviceBindingConditions: {DynamicResourceAllocation, DRAResourceClaimDeviceStatus},

	DRADeviceTaintRules: {DRADeviceTaints}, // DynamicResourceAllocation is indirect.

	DRADeviceTaints: {DynamicResourceAllocation},

	DRAExtendedResource: {DynamicResourceAllocation},

	DRAPartitionableDevices: {DynamicResourceAllocation},

	DRAPrioritizedList: {DynamicResourceAllocation},

	DRAResourceClaimDeviceStatus: {}, // Soft dependency on DynamicResourceAllocation due to on/off-by-default conflict.

	DRASchedulerFilterTimeout: {DynamicResourceAllocation},

	DeploymentReplicaSetTerminatingReplicas: {},

	DisableAllocatorDualWrite: {MultiCIDRServiceAllocator},

	DisableCPUQuotaWithExclusiveCPUs: {},

	DisableNodeKubeProxyVersion: {},

	DynamicResourceAllocation: {},

	EnvFiles: {},

	EventedPLEG: {},

	ExecProbeTimeout: {},

	ExternalServiceAccountTokenSigner: {},

	GangScheduling: {GenericWorkload},

	GenericWorkload: {},

	GitRepoVolumeDriver: {},

	GracefulNodeShutdown: {},

	GracefulNodeShutdownBasedOnPodPriority: {GracefulNodeShutdown},

	HPAConfigurableTolerance: {},

	HPAScaleToZero: {},

	HonorPVReclaimPolicy: {},

	HostnameOverride: {},

	ImageMaximumGCAge: {},

	ImageVolume: {},

	ImageVolumeWithDigest: {ImageVolume},

	InPlacePodLevelResourcesVerticalScaling: {InPlacePodVerticalScaling, PodLevelResources, NodeDeclaredFeatures},

	InPlacePodVerticalScaling: {},

	InPlacePodVerticalScalingAllocatedStatus: {InPlacePodVerticalScaling},

	InPlacePodVerticalScalingExclusiveCPUs: {InPlacePodVerticalScaling},

	InPlacePodVerticalScalingExclusiveMemory: {InPlacePodVerticalScaling, MemoryManager},

	InTreePluginPortworxUnregister: {},

	JobBackoffLimitPerIndex: {},

	JobManagedBy: {},

	JobPodReplacementPolicy: {},

	JobSuccessPolicy: {},

	KubeletCgroupDriverFromCRI: {},

	KubeletCrashLoopBackOffMax: {},

	KubeletEnsureSecretPulledImages: {},

	KubeletFineGrainedAuthz: {},

	KubeletInUserNamespace: {},

	KubeletPSI: {},

	KubeletPodResourcesDynamicResources: {},

	KubeletPodResourcesGet: {},

	KubeletPodResourcesListUseActivePods: {},

	KubeletRegistrationGetOnExistsOnly: {},

	KubeletSeparateDiskGC: {},

	KubeletServiceAccountTokenForCredentialProviders: {},

	KubeletTracing: {},

	LocalStorageCapacityIsolationFSQuotaMonitoring: {},

	LogarithmicScaleDown: {},

	MatchLabelKeysInPodAffinity: {},

	MatchLabelKeysInPodTopologySpread: {},

	MatchLabelKeysInPodTopologySpreadSelectorMerge: {MatchLabelKeysInPodTopologySpread},

	MaxUnavailableStatefulSet: {},

	MemoryManager: {},

	MemoryQoS: {},

	MultiCIDRServiceAllocator: {},

	MutableCSINodeAllocatableCount: {},

	MutablePVNodeAffinity: {},

	MutablePodResourcesForSuspendedJobs: {},

	MutableSchedulingDirectivesForSuspendedJobs: {},

	NFTablesProxyMode: {},

	NodeDeclaredFeatures: {},

	NodeInclusionPolicyInPodTopologySpread: {},

	NodeLogQuery: {},

	NodeSwap: {},

	NominatedNodeNameForExpectation: {},

	OpportunisticBatching: {},

	OrderedNamespaceDeletion: {},

	PodAndContainerStatsFromCRI: {},

	PodCertificateRequest: {AuthorizeNodeWithSelectors},

	PodDeletionCost: {},

	PodLevelResources: {},

	PodLifecycleSleepAction: {},

	PodLifecycleSleepActionAllowZero: {PodLifecycleSleepAction},

	PodLogsQuerySplitStreams: {},

	PodObservedGenerationTracking: {},

	PodReadyToStartContainersCondition: {},

	PodSchedulingReadiness: {},

	PodTopologyLabelsAdmission: {},

	PortForwardWebsockets: {},

	PreferSameTrafficDistribution: {},

	PreventStaticPodAPIReferences: {},

	ProcMountType: {UserNamespacesSupport},

	QOSReserved: {},

	RecoverVolumeExpansionFailure: {},

	RecursiveReadOnlyMounts: {},

	ReduceDefaultCrashLoopBackOffDecay: {},

	RelaxedDNSSearchValidation: {},

	RelaxedEnvironmentVariableValidation: {},

	RelaxedServiceNameValidation: {},

	ReloadKubeletServerCertificateFile: {},

	ResourceHealthStatus: {DynamicResourceAllocation},

	// RestartAllContainersOnContainerExits introduces a new container restart rule action.
	// All restart rules will be dropped by API if ContainerRestartRules feature is not enabled.
	RestartAllContainersOnContainerExits: {ContainerRestartRules, NodeDeclaredFeatures},

	RotateKubeletServerCertificate: {},

	RuntimeClassInImageCriAPI: {},

	SELinuxChangePolicy: {},

	SELinuxMount: {},

	SELinuxMountReadWriteOncePod: {},

	SchedulerAsyncAPICalls: {},

	SchedulerAsyncPreemption: {},

	SchedulerPopFromBackoffQ: {},

	SchedulerQueueingHints: {},

	SeparateTaintEvictionController: {},

	ServiceAccountNodeAudienceRestriction: {},

	ServiceAccountTokenJTI: {},

	ServiceAccountTokenNodeBinding: {ServiceAccountTokenNodeBindingValidation},

	ServiceAccountTokenNodeBindingValidation: {},

	ServiceAccountTokenPodNodeInfo: {},

	ServiceTrafficDistribution: {},

	SidecarContainers: {},

	StatefulSetSemanticRevisionComparison: {},

	StorageCapacityScoring: {},

	StorageNamespaceIndex: {},

	StorageVersionMigrator: {},

	StreamingCollectionEncodingToJSON: {},

	StreamingCollectionEncodingToProtobuf: {},

	StrictIPCIDRValidation: {},

	SupplementalGroupsPolicy: {},

	SystemdWatchdog: {},

	TaintTolerationComparisonOperators: {},

	TopologyAwareHints: {},

	TopologyManagerPolicyAlphaOptions: {},

	TopologyManagerPolicyBetaOptions: {},

	TopologyManagerPolicyOptions: {},

	TranslateStreamCloseWebsocketRequests: {},

	UserNamespacesHostNetworkSupport: {UserNamespacesSupport},

	UserNamespacesSupport: {},

	VolumeAttributesClass: {},

	VolumeLimitScaling: {},

	WinDSR: {},

	WinOverlay: {},

	WindowsCPUAndMemoryAffinity: {MemoryManager},

	WindowsGracefulNodeShutdown: {GracefulNodeShutdown},

	WindowsHostNetwork: {},

	apiextensionsfeatures.CRDObservedGenerationTracking: {},

	apiextensionsfeatures.CRDValidationRatcheting: {},

	apiextensionsfeatures.CustomResourceFieldSelectors: {},

	genericfeatures.APIResponseCompression: {},

	genericfeatures.APIServerIdentity: {},

	genericfeatures.APIServerTracing: {},

	genericfeatures.APIServingWithRoutine: {},

	genericfeatures.AggregatedDiscoveryRemoveBetaType: {},

	genericfeatures.AllowParsingUserUIDFromCertAuth: {},

	genericfeatures.AllowUnsafeMalformedObjectDeletion: {},

	genericfeatures.AnonymousAuthConfigurableEndpoints: {},

	genericfeatures.AuthorizeWithSelectors: {},

	genericfeatures.BtreeWatchCache: {},

	genericfeatures.CBORServingAndStorage: {},

	genericfeatures.ConcurrentWatchObjectDecode: {},

	genericfeatures.ConsistentListFromCache: {},

	genericfeatures.ConstrainedImpersonation: {},

	genericfeatures.DeclarativeValidation: {},

	genericfeatures.DeclarativeValidationTakeover: {genericfeatures.DeclarativeValidation},

	genericfeatures.DetectCacheInconsistency: {},

	genericfeatures.KMSv1: {},

	genericfeatures.ListFromCacheSnapshot: {},

	genericfeatures.MutatingAdmissionPolicy: {},

	genericfeatures.OpenAPIEnums: {},

	genericfeatures.RemoteRequestHeaderUID: {},

	genericfeatures.ResilientWatchCacheInitialization: {},

	genericfeatures.RetryGenerateName: {},

	genericfeatures.SeparateCacheWatchRPC: {},

	genericfeatures.SizeBasedListCostEstimate: {},

	genericfeatures.StorageVersionAPI: {genericfeatures.APIServerIdentity},

	genericfeatures.StorageVersionHash: {},

	genericfeatures.StructuredAuthenticationConfiguration: {},

	genericfeatures.StructuredAuthenticationConfigurationEgressSelector: {genericfeatures.StructuredAuthenticationConfiguration},

	genericfeatures.StructuredAuthenticationConfigurationJWKSMetrics: {genericfeatures.StructuredAuthenticationConfiguration},

	genericfeatures.StructuredAuthorizationConfiguration: {},

	genericfeatures.TokenRequestServiceAccountUIDValidation: {},

	genericfeatures.UnauthenticatedHTTP2DOSMitigation: {},

	genericfeatures.UnknownVersionInteroperabilityProxy: {genericfeatures.APIServerIdentity},

	genericfeatures.WatchCacheInitializationPostStartHook: {},

	genericfeatures.WatchFromStorageWithoutResourceVersion: {},

	genericfeatures.WatchList: {},

	kcmfeatures.CloudControllerManagerWatchBasedRoutesReconciliation: {},

	kcmfeatures.CloudControllerManagerWebhook: {},

	zpagesfeatures.ComponentFlagz: {},

	zpagesfeatures.ComponentStatusz: {},
}

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.AddVersioned(defaultVersionedKubernetesFeatureGates))
	runtime.Must(utilfeature.DefaultMutableFeatureGate.AddDependencies(defaultKubernetesFeatureGateDependencies))
	runtime.Must(zpagesfeatures.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))

	// Register all client-go features with kube's feature gate instance and make all client-go
	// feature checks use kube's instance. The effect is that for kube binaries, client-go
	// features are wired to the existing --feature-gates flag just as all other features
	// are. Further, client-go features automatically support the existing mechanisms for
	// feature enablement metrics and test overrides.
	ca := &clientAdapter{utilfeature.DefaultMutableFeatureGate}
	runtime.Must(clientfeatures.AddVersionedFeaturesToExistingFeatureGates(ca))
	clientfeatures.ReplaceFeatureGates(ca)
}
