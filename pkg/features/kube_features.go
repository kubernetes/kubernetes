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
	// // alpha: v1.X
	// MyFeature featuregate.Feature = "MyFeature"

	// owner: @tallclair
	// beta: v1.4
	AppArmor featuregate.Feature = "AppArmor"

	// owner: @mtaufen
	// alpha: v1.4
	// beta: v1.11
	DynamicKubeletConfig featuregate.Feature = "DynamicKubeletConfig"

	// owner: @pweil-
	// alpha: v1.5
	//
	// Default userns=host for containers that are using other host namespaces, host mounts, the pod
	// contains a privileged container, or specific non-namespaced capabilities (MKNOD, SYS_MODULE,
	// SYS_TIME). This should only be enabled if user namespace remapping is enabled in the docker daemon.
	ExperimentalHostUserNamespaceDefaultingGate featuregate.Feature = "ExperimentalHostUserNamespaceDefaulting"

	// owner: @jiayingz
	// beta: v1.10
	//
	// Enables support for Device Plugins
	DevicePlugins featuregate.Feature = "DevicePlugins"

	// owner: @dxist
	// alpha: v1.16
	//
	// Enables support of HPA scaling to zero pods when an object or custom metric is configured.
	HPAScaleToZero featuregate.Feature = "HPAScaleToZero"

	// owner: @mikedanese
	// alpha: v1.7
	// beta: v1.12
	//
	// Gets a server certificate for the kubelet from the Certificate Signing
	// Request API instead of generating one self signed and auto rotates the
	// certificate as expiration approaches.
	RotateKubeletServerCertificate featuregate.Feature = "RotateKubeletServerCertificate"

	// owner: @jinxu
	// beta: v1.10
	//
	// New local storage types to support local storage capacity isolation
	LocalStorageCapacityIsolation featuregate.Feature = "LocalStorageCapacityIsolation"

	// owner: @gnufied
	// beta: v1.11
	// Ability to Expand persistent volumes
	ExpandPersistentVolumes featuregate.Feature = "ExpandPersistentVolumes"

	// owner: @mlmhl
	// beta: v1.15
	// Ability to expand persistent volumes' file system without unmounting volumes.
	ExpandInUsePersistentVolumes featuregate.Feature = "ExpandInUsePersistentVolumes"

	// owner: @gnufied
	// alpha: v1.14
	// beta: v1.16
	// Ability to expand CSI volumes
	ExpandCSIVolumes featuregate.Feature = "ExpandCSIVolumes"

	// owner: @verb
	// alpha: v1.16
	//
	// Allows running an ephemeral container in pod namespaces to troubleshoot a running pod.
	EphemeralContainers featuregate.Feature = "EphemeralContainers"

	// owner: @sjenning
	// alpha: v1.11
	//
	// Allows resource reservations at the QoS level preventing pods at lower QoS levels from
	// bursting into resources requested at higher QoS levels (memory only for now)
	QOSReserved featuregate.Feature = "QOSReserved"

	// owner: @ConnorDoyle
	// alpha: v1.8
	// beta: v1.10
	//
	// Alternative container-level CPU affinity policies.
	CPUManager featuregate.Feature = "CPUManager"

	// owner: @szuecs
	// alpha: v1.12
	//
	// Enable nodes to change CPUCFSQuotaPeriod
	CPUCFSQuotaPeriod featuregate.Feature = "CustomCPUCFSQuotaPeriod"

	// owner: @lmdaly
	// alpha: v1.16
	// beta: v1.18
	//
	// Enable resource managers to make NUMA aligned decisions
	TopologyManager featuregate.Feature = "TopologyManager"

	// owner: @cynepco3hahue(alukiano) @cezaryzukowski @k-wiatrzyk
	// alpha:: v1.20

	// Allows setting memory affinity for a container based on NUMA topology
	MemoryManager featuregate.Feature = "MemoryManager"

	// owner: @sjenning
	// alpha: v1.4
	// beta: v1.11
	// ga: v1.21
	//
	// Enable pods to set sysctls on a pod
	Sysctls featuregate.Feature = "Sysctls"

	// owner @smarterclayton
	// alpha: v1.16
	// beta:  v1.19
	// ga:  v1.21
	//
	// Enable legacy behavior to vary cluster functionality on the node-role.kubernetes.io labels. On by default (legacy), will be turned off in 1.18.
	// Lock to false in v1.21 and remove in v1.22.
	LegacyNodeRoleBehavior featuregate.Feature = "LegacyNodeRoleBehavior"

	// owner @brendandburns
	// alpha: v1.9
	// beta:  v1.19
	// ga:  v1.21
	//
	// Enable nodes to exclude themselves from service load balancers
	ServiceNodeExclusion featuregate.Feature = "ServiceNodeExclusion"

	// owner @smarterclayton
	// alpha: v1.16
	// beta:  v1.19
	// ga:  v1.21
	//
	// Enable nodes to exclude themselves from network disruption checks
	NodeDisruptionExclusion featuregate.Feature = "NodeDisruptionExclusion"

	// owner: @pospispa
	// GA: v1.11
	//
	// Postpone deletion of a PV or a PVC when they are being used
	StorageObjectInUseProtection featuregate.Feature = "StorageObjectInUseProtection"

	// owner: @dims, @derekwaynecarr
	// alpha: v1.10
	// beta: v1.14
	// GA: v1.20
	//
	// Implement support for limiting pids in pods
	SupportPodPidsLimit featuregate.Feature = "SupportPodPidsLimit"

	// owner: @mikedanese
	// alpha: v1.13
	// beta: v1.21
	//
	// Migrate ServiceAccount volumes to use a projected volume consisting of a
	// ServiceAccountTokenVolumeProjection. This feature adds new required flags
	// to the API server.
	BoundServiceAccountTokenVolume featuregate.Feature = "BoundServiceAccountTokenVolume"

	// owner: @mtaufen
	// alpha: v1.18
	// beta: v1.20
	// stable: v1.21
	//
	// Enable OIDC discovery endpoints (issuer and JWKS URLs) for the service
	// account issuer in the API server.
	// Note these endpoints serve minimally-compliant discovery docs that are
	// intended to be used for service account token verification.
	ServiceAccountIssuerDiscovery featuregate.Feature = "ServiceAccountIssuerDiscovery"

	// owner: @Random-Liu
	// beta: v1.11
	// ga:   v1.21
	//
	// Enable container log rotation for cri container runtime
	CRIContainerLogRotation featuregate.Feature = "CRIContainerLogRotation"

	// owner: @krmayankk
	// beta:  v1.14
	// ga:    v1.21
	//
	// Enables control over the primary group ID of containers' init processes.
	RunAsGroup featuregate.Feature = "RunAsGroup"

	// owner: @saad-ali
	// ga: 	  v1.10
	//
	// Allow mounting a subpath of a volume in a container
	// Do not remove this feature gate even though it's GA
	VolumeSubpath featuregate.Feature = "VolumeSubpath"

	// owner: @ravig
	// alpha: v1.11
	//
	// Include volume count on node to be considered for balanced resource allocation while scheduling.
	// A node which has closer cpu,memory utilization and volume count is favoured by scheduler
	// while making decisions.
	BalanceAttachedNodeVolumes featuregate.Feature = "BalanceAttachedNodeVolumes"

	// owner: @pohly
	// alpha: v1.14
	// beta: v1.16
	//
	// Enables CSI Inline volumes support for pods
	CSIInlineVolume featuregate.Feature = "CSIInlineVolume"

	// owner: @pohly
	// alpha: v1.19
	// beta: v1.21
	//
	// Enables tracking of available storage capacity that CSI drivers provide.
	CSIStorageCapacity featuregate.Feature = "CSIStorageCapacity"

	// owner: @alculquicondor
	// beta: v1.20
	//
	// Enables the use of PodTopologySpread scheduling plugin to do default
	// spreading and disables legacy SelectorSpread plugin.
	DefaultPodTopologySpread featuregate.Feature = "DefaultPodTopologySpread"

	// owner: @pohly
	// alpha: v1.19
	// beta: v1.21
	//
	// Enables generic ephemeral inline volume support for pods
	GenericEphemeralVolume featuregate.Feature = "GenericEphemeralVolume"

	// owner: @chendave
	// alpha: v1.21
	//
	// PreferNominatedNode tells scheduler whether the nominated node will be checked first before looping
	// all the rest of nodes in the cluster.
	// Enabling this feature also implies the preemptor pod might not be dispatched to the best candidate in
	// some corner case, e.g. another node releases enough resources after the nominated node has been set
	// and hence is the best candidate instead.
	PreferNominatedNode featuregate.Feature = "PreferNominatedNode"

	// owner: @tallclair
	// alpha: v1.12
	// beta:  v1.14
	// GA: v1.20
	//
	// Enables RuntimeClass, for selecting between multiple runtimes to run a pod.
	RuntimeClass featuregate.Feature = "RuntimeClass"

	// owner: @mtaufen
	// alpha: v1.12
	// beta:  v1.14
	// GA: v1.17
	//
	// Kubelet uses the new Lease API to report node heartbeats,
	// (Kube) Node Lifecycle Controller uses these heartbeats as a node health signal.
	NodeLease featuregate.Feature = "NodeLease"

	// owner: @janosi
	// alpha: v1.12
	// beta:  v1.19
	// GA:    v1.20
	//
	// Enables SCTP as new protocol for Service ports, NetworkPolicy, and ContainerPort in Pod/Containers definition
	SCTPSupport featuregate.Feature = "SCTPSupport"

	// owner: @rikatz
	// alpha: v1.21
	//
	// Enables the endPort field in NetworkPolicy to enable a Port Range behavior in Network Policies.
	NetworkPolicyEndPort featuregate.Feature = "NetworkPolicyEndPort"

	// owner: @xing-yang
	// alpha: v1.12
	// beta: v1.17
	// GA: v1.20
	//
	// Enable volume snapshot data source support.
	VolumeSnapshotDataSource featuregate.Feature = "VolumeSnapshotDataSource"

	// owner: @jessfraz
	// alpha: v1.12
	//
	// Enables control over ProcMountType for containers.
	ProcMountType featuregate.Feature = "ProcMountType"

	// owner: @janetkuo
	// alpha: v1.12
	//
	// Allow TTL controller to clean up Pods and Jobs after they finish.
	TTLAfterFinished featuregate.Feature = "TTLAfterFinished"

	// owner: @alculquicondor
	// alpha: v1.21
	//
	// Allows Job controller to manage Pod completions per completion index.
	IndexedJob featuregate.Feature = "IndexedJob"

	// owner: @dashpole
	// alpha: v1.13
	// beta: v1.15
	//
	// Enables the kubelet's pod resources grpc endpoint
	KubeletPodResources featuregate.Feature = "KubeletPodResources"

	// owner: @davidz627
	// alpha: v1.14
	// beta: v1.17
	//
	// Enables the in-tree storage to CSI Plugin migration feature.
	CSIMigration featuregate.Feature = "CSIMigration"

	// owner: @davidz627
	// alpha: v1.14
	// beta: v1.17
	//
	// Enables the GCE PD in-tree driver to GCE CSI Driver migration feature.
	CSIMigrationGCE featuregate.Feature = "CSIMigrationGCE"

	// owner: @Jiawei0227
	// alpha: v1.21
	//
	// Disables the GCE PD in-tree driver.
	InTreePluginGCEUnregister featuregate.Feature = "InTreePluginGCEUnregister"

	// owner: @leakingtapan
	// alpha: v1.14
	// beta: v1.17
	//
	// Enables the AWS EBS in-tree driver to AWS EBS CSI Driver migration feature.
	CSIMigrationAWS featuregate.Feature = "CSIMigrationAWS"

	// owner: @leakingtapan
	// alpha: v1.21
	//
	// Disables the AWS EBS in-tree driver.
	InTreePluginAWSUnregister featuregate.Feature = "InTreePluginAWSUnregister"

	// owner: @andyzhangx
	// alpha: v1.15
	// beta: v1.19
	//
	// Enables the Azure Disk in-tree driver to Azure Disk Driver migration feature.
	CSIMigrationAzureDisk featuregate.Feature = "CSIMigrationAzureDisk"

	// owner: @andyzhangx
	// alpha: v1.21
	//
	// Disables the Azure Disk in-tree driver.
	InTreePluginAzureDiskUnregister featuregate.Feature = "InTreePluginAzureDiskUnregister"

	// owner: @andyzhangx
	// alpha: v1.15
	// beta: v1.21
	//
	// Enables the Azure File in-tree driver to Azure File Driver migration feature.
	CSIMigrationAzureFile featuregate.Feature = "CSIMigrationAzureFile"

	// owner: @andyzhangx
	// alpha: v1.21
	//
	// Disables the Azure File in-tree driver.
	InTreePluginAzureFileUnregister featuregate.Feature = "InTreePluginAzureFileUnregister"

	// owner: @divyenpatel
	// beta: v1.19 (requires: vSphere vCenter/ESXi Version: 7.0u1, HW Version: VM version 15)
	//
	// Enables the vSphere in-tree driver to vSphere CSI Driver migration feature.
	CSIMigrationvSphere featuregate.Feature = "CSIMigrationvSphere"

	// owner: @divyenpatel
	// beta: v1.19 (requires: vSphere vCenter/ESXi Version: 7.0u1, HW Version: VM version 15)
	//
	// Disables the vSphere in-tree driver.
	// Expects vSphere CSI Driver to be installed and configured on all nodes.
	CSIMigrationvSphereComplete featuregate.Feature = "CSIMigrationvSphereComplete"

	// owner: @divyenpatel
	// alpha: v1.21
	//
	// Disables the vSphere in-tree driver.
	InTreePluginvSphereUnregister featuregate.Feature = "InTreePluginvSphereUnregister"

	// owner: @adisky
	// alpha: v1.14
	// beta: v1.18
	//
	// Enables the OpenStack Cinder in-tree driver to OpenStack Cinder CSI Driver migration feature.
	CSIMigrationOpenStack featuregate.Feature = "CSIMigrationOpenStack"

	// owner: @adisky
	// alpha: v1.21
	//
	// Disables the OpenStack Cinder in-tree driver.
	InTreePluginOpenStackUnregister featuregate.Feature = "InTreePluginOpenStackUnregister"

	// owner: @huffmanca
	// alpha: v1.19
	// beta: v1.20
	//
	// Determines if a CSI Driver supports applying fsGroup.
	CSIVolumeFSGroupPolicy featuregate.Feature = "CSIVolumeFSGroupPolicy"

	// owner: @gnufied
	// alpha: v1.18
	// beta: v1.20
	// Allows user to configure volume permission change policy for fsGroups when mounting
	// a volume in a Pod.
	ConfigurableFSGroupPolicy featuregate.Feature = "ConfigurableFSGroupPolicy"

	// owner: @RobertKrawitz, @derekwaynecarr
	// beta: v1.15
	// GA: v1.20
	//
	// Implement support for limiting pids in nodes
	SupportNodePidsLimit featuregate.Feature = "SupportNodePidsLimit"

	// owner: @RobertKrawitz
	// alpha: v1.15
	//
	// Allow use of filesystems for ephemeral storage monitoring.
	// Only applies if LocalStorageCapacityIsolation is set.
	LocalStorageCapacityIsolationFSQuotaMonitoring featuregate.Feature = "LocalStorageCapacityIsolationFSQuotaMonitoring"

	// owner: @denkensk
	// alpha: v1.15
	// beta: v1.19
	//
	// Enables NonPreempting option for priorityClass and pod.
	NonPreemptingPriority featuregate.Feature = "NonPreemptingPriority"

	// owner: @egernst
	// alpha: v1.16
	// beta: v1.18
	//
	// Enables PodOverhead, for accounting pod overheads which are specific to a given RuntimeClass
	PodOverhead featuregate.Feature = "PodOverhead"

	// owner: @khenidak
	// alpha: v1.15
	// beta: v1.21
	//
	// Enables ipv6 dual stack
	IPv6DualStack featuregate.Feature = "IPv6DualStack"

	// owner: @robscott @freehan
	// alpha: v1.16
	// beta: v1.18
	// ga: v1.21
	//
	// Enable Endpoint Slices for more scalable Service endpoints.
	EndpointSlice featuregate.Feature = "EndpointSlice"

	// owner: @robscott @freehan
	// alpha: v1.18
	// beta: v1.19
	//
	// Enable Endpoint Slice consumption by kube-proxy for improved scalability.
	EndpointSliceProxying featuregate.Feature = "EndpointSliceProxying"

	// owner: @robscott @kumarvin123
	// alpha: v1.19
	// beta: v1.21
	//
	// Enable Endpoint Slice consumption by kube-proxy in Windows for improved scalability.
	WindowsEndpointSliceProxying featuregate.Feature = "WindowsEndpointSliceProxying"

	// owner: @matthyx
	// alpha: v1.16
	// beta: v1.18
	// GA: v1.20
	//
	// Enables the startupProbe in kubelet worker.
	StartupProbe featuregate.Feature = "StartupProbe"

	// owner: @deads2k
	// beta: v1.17
	// GA: v1.21
	//
	// Enables the users to skip TLS verification of kubelets on pod logs requests
	AllowInsecureBackendProxy featuregate.Feature = "AllowInsecureBackendProxy"

	// owner: @mortent
	// alpha: v1.3
	// beta:  v1.5
	//
	// Enable all logic related to the PodDisruptionBudget API object in policy
	PodDisruptionBudget featuregate.Feature = "PodDisruptionBudget"

	// owner: @alaypatel07, @soltysh
	// alpha: v1.20
	// beta: v1.21
	//
	// CronJobControllerV2 controls whether the controller manager starts old cronjob
	// controller or new one which is implemented with informers and delaying queue
	//
	// This feature is deprecated, and will be removed in v1.22.
	CronJobControllerV2 featuregate.Feature = "CronJobControllerV2"

	// owner: @smarterclayton
	// alpha: v1.21
	//
	// DaemonSets allow workloads to maintain availability during update per node
	DaemonSetUpdateSurge featuregate.Feature = "DaemonSetUpdateSurge"

	// owner: @m1093782566
	// alpha: v1.17
	//
	// Enables topology aware service routing
	ServiceTopology featuregate.Feature = "ServiceTopology"

	// owner: @robscott
	// alpha: v1.18
	// beta:  v1.19
	// ga:    v1.20
	//
	// Enables AppProtocol field for Services and Endpoints.
	ServiceAppProtocol featuregate.Feature = "ServiceAppProtocol"

	// owner: @wojtek-t
	// alpha: v1.18
	// beta:  v1.19
	// ga:    v1.21
	//
	// Enables a feature to make secrets and configmaps data immutable.
	ImmutableEphemeralVolumes featuregate.Feature = "ImmutableEphemeralVolumes"

	// owner: @bart0sh
	// alpha: v1.18
	// beta: v1.19
	//
	// Enables usage of HugePages-<size> in a volume medium,
	// e.g. emptyDir:
	//        medium: HugePages-1Gi
	HugePageStorageMediumSize featuregate.Feature = "HugePageStorageMediumSize"

	// owner: @derekwaynecarr
	// alpha: v1.20
	// beta: v1.21 (off by default until 1.22)
	//
	// Enables usage of hugepages-<size> in downward API.
	DownwardAPIHugePages featuregate.Feature = "DownwardAPIHugePages"

	// owner: @freehan
	// GA: v1.18
	//
	// Enable ExternalTrafficPolicy for Service ExternalIPs.
	// This is for bug fix #69811
	ExternalPolicyForExternalIP featuregate.Feature = "ExternalPolicyForExternalIP"

	// owner: @bswartz
	// alpha: v1.18
	//
	// Enables usage of any object for volume data source in PVCs
	AnyVolumeDataSource featuregate.Feature = "AnyVolumeDataSource"

	// owner: @javidiaz
	// alpha: v1.19
	// beta: v1.20
	//
	// Allow setting the Fully Qualified Domain Name (FQDN) in the hostname of a Pod. If a Pod does not
	// have FQDN, this feature has no effect.
	SetHostnameAsFQDN featuregate.Feature = "SetHostnameAsFQDN"

	// owner: @ksubrmnn
	// alpha: v1.14
	// beta: v1.20
	//
	// Allows kube-proxy to run in Overlay mode for Windows
	WinOverlay featuregate.Feature = "WinOverlay"

	// owner: @ksubrmnn
	// alpha: v1.14
	//
	// Allows kube-proxy to create DSR loadbalancers for Windows
	WinDSR featuregate.Feature = "WinDSR"

	// owner: @RenaudWasTaken @dashpole
	// alpha: v1.19
	// beta: v1.20
	//
	// Disables Accelerator Metrics Collected by Kubelet
	DisableAcceleratorUsageMetrics featuregate.Feature = "DisableAcceleratorUsageMetrics"

	// owner: @arjunrn @mwielgus @josephburnett
	// alpha: v1.20
	//
	// Add support for the HPA to scale based on metrics from individual containers
	// in target pods
	HPAContainerMetrics featuregate.Feature = "HPAContainerMetrics"

	// owner: @zshihang
	// alpha: v1.13
	// beta: v1.20
	// ga: v1.21
	//
	// Allows kube-controller-manager to publish kube-root-ca.crt configmap to
	// every namespace. This feature is a prerequisite of BoundServiceAccountTokenVolume.
	RootCAConfigMap featuregate.Feature = "RootCAConfigMap"

	// owner: @andrewsykim
	// alpha: v1.20
	//
	// Enable Terminating condition in Endpoint Slices.
	EndpointSliceTerminatingCondition featuregate.Feature = "EndpointSliceTerminatingCondition"

	// owner: @robscott
	// alpha: v1.20
	//
	// Enable NodeName field on Endpoint Slices.
	EndpointSliceNodeName featuregate.Feature = "EndpointSliceNodeName"

	// owner: @derekwaynecarr
	// alpha: v1.20
	//
	// Enables kubelet support to size memory backed volumes
	SizeMemoryBackedVolumes featuregate.Feature = "SizeMemoryBackedVolumes"

	// owner: @andrewsykim @SergeyKanzhelev
	// GA: v1.20
	//
	// Ensure kubelet respects exec probe timeouts. Feature gate exists in-case existing workloads
	// may depend on old behavior where exec probe timeouts were ignored.
	// Lock to default and remove after v1.22 based on user feedback that should be reflected in KEP #1972 update
	ExecProbeTimeout featuregate.Feature = "ExecProbeTimeout"

	// owner: @andrewsykim
	// alpha: v1.20
	//
	// Enable kubelet exec plugins for image pull credentials.
	KubeletCredentialProviders featuregate.Feature = "KubeletCredentialProviders"

	// owner: @zshihang
	// alpha: v1.20
	// beta: v1.21
	//
	// Enable kubelet to pass pod's service account token to NodePublishVolume
	// call of CSI driver which is mounting volumes for that pod.
	CSIServiceAccountToken featuregate.Feature = "CSIServiceAccountToken"

	// owner: @bobbypage
	// alpha: v1.20
	// beta:  v1.21
	// Adds support for kubelet to detect node shutdown and gracefully terminate pods prior to the node being shutdown.
	GracefulNodeShutdown featuregate.Feature = "GracefulNodeShutdown"

	// owner: @andrewsykim @uablrek
	// alpha: v1.20
	//
	// Allows control if NodePorts shall be created for services with "type: LoadBalancer" by defining the spec.AllocateLoadBalancerNodePorts field (bool)
	ServiceLBNodePortControl featuregate.Feature = "ServiceLBNodePortControl"

	// owner: @janosi
	// alpha: v1.20
	//
	// Enables the usage of different protocols in the same Service with type=LoadBalancer
	MixedProtocolLBService featuregate.Feature = "MixedProtocolLBService"

	// owner: @cofyc
	// alpha: v1.21
	VolumeCapacityPriority featuregate.Feature = "VolumeCapacityPriority"

	// owner: @ahg-g
	// alpha: v1.21
	//
	// Enables controlling pod ranking on replicaset scale-down.
	PodDeletionCost featuregate.Feature = "PodDeletionCost"

	// owner: @robscott
	// alpha: v1.21
	//
	// Enables topology aware hints for EndpointSlices
	TopologyAwareHints featuregate.Feature = "TopologyAwareHints"

	// owner: @ehashman
	// alpha: v1.21
	//
	// Allows user to override pod-level terminationGracePeriod for probes
	ProbeTerminationGracePeriod featuregate.Feature = "ProbeTerminationGracePeriod"

	// owner: @ahg-g
	// alpha: v1.21
	//
	// Allow specifying NamespaceSelector in PodAffinityTerm.
	PodAffinityNamespaceSelector featuregate.Feature = "PodAffinityNamespaceSelector"

	// owner: @andrewsykim @xudongliuharold
	// alpha: v1.21
	//
	// Enable support multiple Service "type: LoadBalancer" implementations in a cluster by specifying LoadBalancerClass
	ServiceLoadBalancerClass featuregate.Feature = "ServiceLoadBalancerClass"

	// owner: @damemi
	// aplpha: v1.21
	//
	// Enables scaling down replicas via logarithmic comparison of creation/ready timestamps
	LogarithmicScaleDown featuregate.Feature = "LogarithmicScaleDown"

	// owner: @hbagdi
	// alpha: v1.21
	//
	// Enable Scope and Namespace fields on IngressClassParametersReference.
	IngressClassNamespacedParams featuregate.Feature = "IngressClassNamespacedParams"

	// owner: @maplain @andrewsykim
	// alpha: v1.21
	//
	// Enables node-local routing for Service internal traffic
	ServiceInternalTrafficPolicy featuregate.Feature = "ServiceInternalTrafficPolicy"

	// owner: @adtac
	// alpha: v1.21
	//
	// Allows jobs to be created in the suspended state.
	SuspendJob featuregate.Feature = "SuspendJob"

	// owner: @fromanirh
	// alpha: v1.21
	//
	// Enable POD resources API to return allocatable resources
	KubeletPodResourcesGetAllocatable featuregate.Feature = "KubeletPodResourcesGetAllocatable"

	// owner: @jayunit100 @abhiraut @rikatz
	// beta: v1.21
	//
	// Labels all namespaces with a default label "kubernetes.io/metadata.name: <namespaceName>"
	NamespaceDefaultLabelName featuregate.Feature = "NamespaceDefaultLabelName"

	// owner: @fengzixu
	// alpha: v1.21
	//
	// Enables kubelet to detect CSI volume condition and send the event of the abnormal volume to the corresponding pod that is using it.
	CSIVolumeHealth featuregate.Feature = "CSIVolumeHealth"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmor:             {Default: true, PreRelease: featuregate.Beta},
	DynamicKubeletConfig: {Default: true, PreRelease: featuregate.Beta},
	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: featuregate.Beta},
	DevicePlugins:                                  {Default: true, PreRelease: featuregate.Beta},
	RotateKubeletServerCertificate:                 {Default: true, PreRelease: featuregate.Beta},
	LocalStorageCapacityIsolation:                  {Default: true, PreRelease: featuregate.Beta},
	Sysctls:                                        {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.23
	EphemeralContainers:                            {Default: false, PreRelease: featuregate.Alpha},
	QOSReserved:                                    {Default: false, PreRelease: featuregate.Alpha},
	ExpandPersistentVolumes:                        {Default: true, PreRelease: featuregate.Beta},
	ExpandInUsePersistentVolumes:                   {Default: true, PreRelease: featuregate.Beta},
	ExpandCSIVolumes:                               {Default: true, PreRelease: featuregate.Beta},
	CPUManager:                                     {Default: true, PreRelease: featuregate.Beta},
	MemoryManager:                                  {Default: false, PreRelease: featuregate.Alpha},
	CPUCFSQuotaPeriod:                              {Default: false, PreRelease: featuregate.Alpha},
	TopologyManager:                                {Default: true, PreRelease: featuregate.Beta},
	ServiceNodeExclusion:                           {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	NodeDisruptionExclusion:                        {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	StorageObjectInUseProtection:                   {Default: true, PreRelease: featuregate.GA},
	SupportPodPidsLimit:                            {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.23
	SupportNodePidsLimit:                           {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.23
	BoundServiceAccountTokenVolume:                 {Default: true, PreRelease: featuregate.Beta},
	ServiceAccountIssuerDiscovery:                  {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	CRIContainerLogRotation:                        {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	CSIMigration:                                   {Default: true, PreRelease: featuregate.Beta},
	CSIMigrationGCE:                                {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires GCE PD CSI Driver)
	InTreePluginGCEUnregister:                      {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAWS:                                {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires AWS EBS CSI driver)
	InTreePluginAWSUnregister:                      {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAzureDisk:                          {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires Azure Disk CSI driver)
	InTreePluginAzureDiskUnregister:                {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAzureFile:                          {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires Azure File CSI driver)
	InTreePluginAzureFileUnregister:                {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationvSphere:                            {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires vSphere CSI driver)
	CSIMigrationvSphereComplete:                    {Default: false, PreRelease: featuregate.Beta}, // remove in 1.22
	InTreePluginvSphereUnregister:                  {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationOpenStack:                          {Default: true, PreRelease: featuregate.Beta},
	InTreePluginOpenStackUnregister:                {Default: false, PreRelease: featuregate.Alpha},
	VolumeSubpath:                                  {Default: true, PreRelease: featuregate.GA},
	ConfigurableFSGroupPolicy:                      {Default: true, PreRelease: featuregate.Beta},
	BalanceAttachedNodeVolumes:                     {Default: false, PreRelease: featuregate.Alpha},
	CSIInlineVolume:                                {Default: true, PreRelease: featuregate.Beta},
	CSIStorageCapacity:                             {Default: true, PreRelease: featuregate.Beta},
	CSIServiceAccountToken:                         {Default: true, PreRelease: featuregate.Beta},
	GenericEphemeralVolume:                         {Default: true, PreRelease: featuregate.Beta},
	CSIVolumeFSGroupPolicy:                         {Default: true, PreRelease: featuregate.Beta},
	RuntimeClass:                                   {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.23
	NodeLease:                                      {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	SCTPSupport:                                    {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	NetworkPolicyEndPort:                           {Default: false, PreRelease: featuregate.Alpha},
	VolumeSnapshotDataSource:                       {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.21
	ProcMountType:                                  {Default: false, PreRelease: featuregate.Alpha},
	TTLAfterFinished:                               {Default: true, PreRelease: featuregate.Beta},
	IndexedJob:                                     {Default: false, PreRelease: featuregate.Alpha},
	KubeletPodResources:                            {Default: true, PreRelease: featuregate.Beta},
	LocalStorageCapacityIsolationFSQuotaMonitoring: {Default: false, PreRelease: featuregate.Alpha},
	NonPreemptingPriority:                          {Default: true, PreRelease: featuregate.Beta},
	PodOverhead:                                    {Default: true, PreRelease: featuregate.Beta},
	IPv6DualStack:                                  {Default: true, PreRelease: featuregate.Beta},
	EndpointSlice:                                  {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	EndpointSliceProxying:                          {Default: true, PreRelease: featuregate.Beta},
	EndpointSliceTerminatingCondition:              {Default: false, PreRelease: featuregate.Alpha},
	EndpointSliceNodeName:                          {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, //remove in 1.25
	WindowsEndpointSliceProxying:                   {Default: true, PreRelease: featuregate.Beta},
	StartupProbe:                                   {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.23
	AllowInsecureBackendProxy:                      {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.23
	PodDisruptionBudget:                            {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	CronJobControllerV2:                            {Default: true, PreRelease: featuregate.Beta},
	DaemonSetUpdateSurge:                           {Default: false, PreRelease: featuregate.Alpha},
	ServiceTopology:                                {Default: false, PreRelease: featuregate.Alpha},
	ServiceAppProtocol:                             {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	ImmutableEphemeralVolumes:                      {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.24
	HugePageStorageMediumSize:                      {Default: true, PreRelease: featuregate.Beta},
	DownwardAPIHugePages:                           {Default: false, PreRelease: featuregate.Beta},                   // on by default in 1.22
	ExternalPolicyForExternalIP:                    {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	AnyVolumeDataSource:                            {Default: false, PreRelease: featuregate.Alpha},
	DefaultPodTopologySpread:                       {Default: true, PreRelease: featuregate.Beta},
	SetHostnameAsFQDN:                              {Default: true, PreRelease: featuregate.Beta},
	WinOverlay:                                     {Default: true, PreRelease: featuregate.Beta},
	WinDSR:                                         {Default: false, PreRelease: featuregate.Alpha},
	DisableAcceleratorUsageMetrics:                 {Default: true, PreRelease: featuregate.Beta},
	HPAContainerMetrics:                            {Default: false, PreRelease: featuregate.Alpha},
	RootCAConfigMap:                                {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	SizeMemoryBackedVolumes:                        {Default: false, PreRelease: featuregate.Alpha},
	ExecProbeTimeout:                               {Default: true, PreRelease: featuregate.GA}, // lock to default and remove after v1.22 based on KEP #1972 update
	KubeletCredentialProviders:                     {Default: false, PreRelease: featuregate.Alpha},
	GracefulNodeShutdown:                           {Default: true, PreRelease: featuregate.Beta},
	ServiceLBNodePortControl:                       {Default: false, PreRelease: featuregate.Alpha},
	MixedProtocolLBService:                         {Default: false, PreRelease: featuregate.Alpha},
	VolumeCapacityPriority:                         {Default: false, PreRelease: featuregate.Alpha},
	PreferNominatedNode:                            {Default: false, PreRelease: featuregate.Alpha},
	ProbeTerminationGracePeriod:                    {Default: false, PreRelease: featuregate.Alpha},
	RunAsGroup:                                     {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
	PodDeletionCost:                                {Default: false, PreRelease: featuregate.Alpha},
	TopologyAwareHints:                             {Default: false, PreRelease: featuregate.Alpha},
	PodAffinityNamespaceSelector:                   {Default: false, PreRelease: featuregate.Alpha},
	ServiceLoadBalancerClass:                       {Default: false, PreRelease: featuregate.Alpha},
	LogarithmicScaleDown:                           {Default: false, PreRelease: featuregate.Alpha},
	IngressClassNamespacedParams:                   {Default: false, PreRelease: featuregate.Alpha},
	ServiceInternalTrafficPolicy:                   {Default: false, PreRelease: featuregate.Alpha},
	SuspendJob:                                     {Default: false, PreRelease: featuregate.Alpha},
	KubeletPodResourcesGetAllocatable:              {Default: false, PreRelease: featuregate.Alpha},
	NamespaceDefaultLabelName:                      {Default: true, PreRelease: featuregate.Beta}, // graduate to GA and lock to default in 1.22, remove in 1.24
	CSIVolumeHealth:                                {Default: false, PreRelease: featuregate.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.StreamingProxyRedirects: {Default: true, PreRelease: featuregate.Deprecated},
	genericfeatures.ValidateProxyRedirects:  {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.AdvancedAuditing:        {Default: true, PreRelease: featuregate.GA},
	genericfeatures.APIResponseCompression:  {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.APIListChunking:         {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.DryRun:                  {Default: true, PreRelease: featuregate.GA},
	genericfeatures.ServerSideApply:         {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.APIPriorityAndFairness:  {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.WarningHeaders:          {Default: true, PreRelease: featuregate.Beta},

	// features that enable backwards compatibility but are scheduled to be removed
	// ...
	HPAScaleToZero:         {Default: false, PreRelease: featuregate.Alpha},
	LegacyNodeRoleBehavior: {Default: false, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.22
}
