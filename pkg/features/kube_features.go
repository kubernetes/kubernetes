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

	// owner: @tallclair
	// beta: v1.4
	AppArmor featuregate.Feature = "AppArmor"

	// owner: @mtaufen
	// alpha: v1.4
	// beta: v1.11
	// deprecated: 1.22
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
	// GA: 1.24
	// Ability to Expand persistent volumes
	ExpandPersistentVolumes featuregate.Feature = "ExpandPersistentVolumes"

	// owner: @mlmhl @gnufied
	// beta: v1.15
	// GA: 1.24
	// Ability to expand persistent volumes' file system without unmounting volumes.
	ExpandInUsePersistentVolumes featuregate.Feature = "ExpandInUsePersistentVolumes"

	// owner: @gnufied
	// alpha: v1.14
	// beta: v1.16
	// GA: 1.24
	// Ability to expand CSI volumes
	ExpandCSIVolumes featuregate.Feature = "ExpandCSIVolumes"

	// owner: @verb
	// alpha: v1.16
	// beta: v1.23
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
	// alpha: v1.21
	// beta: v1.22

	// Allows setting memory affinity for a container based on NUMA topology
	MemoryManager featuregate.Feature = "MemoryManager"

	// owner: @pospispa
	// GA: v1.11
	//
	// Postpone deletion of a PV or a PVC when they are being used
	StorageObjectInUseProtection featuregate.Feature = "StorageObjectInUseProtection"

	// owner: @saad-ali
	// ga: 	  v1.10
	//
	// Allow mounting a subpath of a volume in a container
	// NOTE: This feature gate has been deprecated and is no longer enforced.
	// It will be completely removed in 1.25. Until then, it's still visible in `kubelet --help`
	VolumeSubpath featuregate.Feature = "VolumeSubpath"

	// owner: @pohly
	// alpha: v1.14
	// beta: v1.16
	//
	// Enables CSI Inline volumes support for pods
	CSIInlineVolume featuregate.Feature = "CSIInlineVolume"

	// owner: @pohly
	// alpha: v1.19
	// beta: v1.21
	// GA: v1.24
	//
	// Enables tracking of available storage capacity that CSI drivers provide.
	CSIStorageCapacity featuregate.Feature = "CSIStorageCapacity"

	// owner: @alculquicondor
	// alpha: v1.19
	// beta: v1.20
	// GA: v1.24
	//
	// Enables the use of PodTopologySpread scheduling plugin to do default
	// spreading and disables legacy SelectorSpread plugin.
	DefaultPodTopologySpread featuregate.Feature = "DefaultPodTopologySpread"

	// owner: @pohly
	// alpha: v1.19
	// beta: v1.21
	// GA: v1.23
	//
	// Enables generic ephemeral inline volume support for pods
	GenericEphemeralVolume featuregate.Feature = "GenericEphemeralVolume"

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

	// owner: @rikatz
	// kep: http://kep.k8s.io/2079
	// alpha: v1.21
	// beta:  v1.22
	//
	// Enables the endPort field in NetworkPolicy to enable a Port Range behavior in Network Policies.
	NetworkPolicyEndPort featuregate.Feature = "NetworkPolicyEndPort"

	// owner: @jessfraz
	// alpha: v1.12
	//
	// Enables control over ProcMountType for containers.
	ProcMountType featuregate.Feature = "ProcMountType"

	// owner: @janetkuo
	// alpha: v1.12
	// beta:  v1.21
	//
	// Allow TTL controller to clean up Pods and Jobs after they finish.
	TTLAfterFinished featuregate.Feature = "TTLAfterFinished"

	// owner: @alculquicondor
	// alpha: v1.21
	// beta: v1.22
	// stable: v1.24
	//
	// Allows Job controller to manage Pod completions per completion index.
	IndexedJob featuregate.Feature = "IndexedJob"

	// owner: @alculquicondor
	// alpha: v1.22
	// beta: v1.23
	//
	// Track Job completion without relying on Pod remaining in the cluster
	// indefinitely. Pod finalizers, in addition to a field in the Job status
	// allow the Job controller to keep track of Pods that it didn't account for
	// yet.
	JobTrackingWithFinalizers featuregate.Feature = "JobTrackingWithFinalizers"

	// owner: @alculquicondor
	// alpha: v1.23
	// beta: v1.24
	//
	// Track the number of pods with Ready condition in the Job status.
	JobReadyPods featuregate.Feature = "JobReadyPods"

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
	// GA: v1.24
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
	// beta: v1.19 (requires: vSphere vCenter/ESXi Version: 7.0u2, HW Version: VM version 15)
	//
	// Enables the vSphere in-tree driver to vSphere CSI Driver migration feature.
	CSIMigrationvSphere featuregate.Feature = "CSIMigrationvSphere"

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

	// owner: @trierra
	// alpha: v1.23
	//
	// Enables the Portworx in-tree driver to Portworx migration feature.
	CSIMigrationPortworx featuregate.Feature = "CSIMigrationPortworx"

	// owner: @trierra
	// alpha: v1.23
	//
	// Disables the Portworx in-tree driver.
	InTreePluginPortworxUnregister featuregate.Feature = "InTreePluginPortworxUnregister"

	// owner: @humblec
	// alpha: v1.23
	//
	// Enables the RBD in-tree driver to RBD CSI Driver  migration feature.
	CSIMigrationRBD featuregate.Feature = "CSIMigrationRBD"

	// owner: @humblec
	// alpha: v1.23
	//
	// Disables the RBD in-tree driver.
	InTreePluginRBDUnregister featuregate.Feature = "InTreePluginRBDUnregister"

	// owner: @huffmanca, @dobsonj
	// alpha: v1.19
	// beta: v1.20
	// GA: v1.23
	//
	// Determines if a CSI Driver supports applying fsGroup.
	CSIVolumeFSGroupPolicy featuregate.Feature = "CSIVolumeFSGroupPolicy"

	// owner: @gnufied
	// alpha: v1.18
	// beta: v1.20
	// GA: v1.23
	// Allows user to configure volume permission change policy for fsGroups when mounting
	// a volume in a Pod.
	ConfigurableFSGroupPolicy featuregate.Feature = "ConfigurableFSGroupPolicy"

	// owner: @gnufied, @verult
	// alpha: v1.22
	// beta: v1.23
	// If supported by the CSI driver, delegates the role of applying FSGroup to
	// the driver by passing FSGroup through the NodeStageVolume and
	// NodePublishVolume calls.
	DelegateFSGroupToCSIDriver featuregate.Feature = "DelegateFSGroupToCSIDriver"

	// owner: @RobertKrawitz
	// alpha: v1.15
	//
	// Allow use of filesystems for ephemeral storage monitoring.
	// Only applies if LocalStorageCapacityIsolation is set.
	LocalStorageCapacityIsolationFSQuotaMonitoring featuregate.Feature = "LocalStorageCapacityIsolationFSQuotaMonitoring"

	// owner: @denkensk
	// alpha: v1.15
	// beta: v1.19
	// ga: v1.24
	//
	// Enables NonPreempting option for priorityClass and pod.
	NonPreemptingPriority featuregate.Feature = "NonPreemptingPriority"

	// owner: @egernst
	// alpha: v1.16
	// beta: v1.18
	// ga: v1.24
	//
	// Enables PodOverhead, for accounting pod overheads which are specific to a given RuntimeClass
	PodOverhead featuregate.Feature = "PodOverhead"

	// owner: @khenidak
	// kep: http://kep.k8s.io/563
	// alpha: v1.15
	// beta: v1.21
	// ga: v1.23
	//
	// Enables ipv6 dual stack
	IPv6DualStack featuregate.Feature = "IPv6DualStack"

	// owner: @robscott @freehan
	// kep: http://kep.k8s.io/752
	// alpha: v1.16
	// beta: v1.18
	// ga: v1.21
	//
	// Enable Endpoint Slices for more scalable Service endpoints.
	EndpointSlice featuregate.Feature = "EndpointSlice"

	// owner: @robscott @freehan
	// kep: http://kep.k8s.io/752
	// alpha: v1.18
	// beta: v1.19
	// ga: v1.22
	//
	// Enable Endpoint Slice consumption by kube-proxy for improved scalability.
	EndpointSliceProxying featuregate.Feature = "EndpointSliceProxying"

	// owner: @robscott @kumarvin123
	// kep: http://kep.k8s.io/752
	// alpha: v1.19
	// beta: v1.21
	// ga: v1.22
	//
	// Enable Endpoint Slice consumption by kube-proxy in Windows for improved scalability.
	WindowsEndpointSliceProxying featuregate.Feature = "WindowsEndpointSliceProxying"

	// owner: @mortent
	// alpha: v1.3
	// beta:  v1.5
	//
	// Enable all logic related to the PodDisruptionBudget API object in policy
	PodDisruptionBudget featuregate.Feature = "PodDisruptionBudget"

	// owner: @smarterclayton
	// alpha: v1.21
	// beta: v1.22
	// DaemonSets allow workloads to maintain availability during update per node
	DaemonSetUpdateSurge featuregate.Feature = "DaemonSetUpdateSurge"

	// owner: @derekwaynecarr
	// alpha: v1.20
	// beta: v1.21 (off by default until 1.22)
	//
	// Enables usage of hugepages-<size> in downward API.
	DownwardAPIHugePages featuregate.Feature = "DownwardAPIHugePages"

	// owner: @bswartz
	// alpha: v1.18
	// beta: v1.24
	//
	// Enables usage of any object for volume data source in PVCs
	AnyVolumeDataSource featuregate.Feature = "AnyVolumeDataSource"

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

	// owner: @andrewsykim
	// kep: http://kep.k8s.io/1672
	// alpha: v1.20
	// beta: v1.22
	//
	// Enable Terminating condition in Endpoint Slices.
	EndpointSliceTerminatingCondition featuregate.Feature = "EndpointSliceTerminatingCondition"

	// owner: @andrewsykim
	// kep: http://kep.k8s.io/1669
	// alpha: v1.22
	//
	// Enable kube-proxy to handle terminating ednpoints when externalTrafficPolicy=Local
	ProxyTerminatingEndpoints featuregate.Feature = "ProxyTerminatingEndpoints"

	// owner: @robscott
	// kep: http://kep.k8s.io/752
	// alpha: v1.20
	//
	// Enable NodeName field on Endpoint Slices.
	EndpointSliceNodeName featuregate.Feature = "EndpointSliceNodeName"

	// owner: @derekwaynecarr
	// alpha: v1.20
	// beta: v1.22
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

	// owner: @andrewsykim @adisky
	// alpha: v1.20
	// beta: v1.24
	//
	// Enable kubelet exec plugins for image pull credentials.
	KubeletCredentialProviders featuregate.Feature = "KubeletCredentialProviders"

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

	// owner: @zshihang
	// alpha: v1.20
	// beta: v1.21
	// ga: v1.22
	//
	// Enable kubelet to pass pod's service account token to NodePublishVolume
	// call of CSI driver which is mounting volumes for that pod.
	CSIServiceAccountToken featuregate.Feature = "CSIServiceAccountToken"

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

	// owner: @andrewsykim @uablrek
	// kep: http://kep.k8s.io/1864
	// alpha: v1.20
	// beta: v1.22
	// ga: v1.24
	//
	// Allows control if NodePorts shall be created for services with "type: LoadBalancer" by defining the spec.AllocateLoadBalancerNodePorts field (bool)
	ServiceLBNodePortControl featuregate.Feature = "ServiceLBNodePortControl"

	// owner: @janosi @bridgetkromhout
	// kep: http://kep.k8s.io/1435
	// alpha: v1.20
	// beta: v1.24
	//
	// Enables the usage of different protocols in the same Service with type=LoadBalancer
	MixedProtocolLBService featuregate.Feature = "MixedProtocolLBService"

	// owner: @cofyc
	// alpha: v1.21
	VolumeCapacityPriority featuregate.Feature = "VolumeCapacityPriority"

	// owner: @mattcary
	// alpha: v1.22
	//
	// Enables policies controlling deletion of PVCs created by a StatefulSet.
	StatefulSetAutoDeletePVC featuregate.Feature = "StatefulSetAutoDeletePVC"

	// owner: @ahg-g
	// alpha: v1.21
	// beta: v1.22
	//
	// Enables controlling pod ranking on replicaset scale-down.
	PodDeletionCost featuregate.Feature = "PodDeletionCost"

	// owner: @robscott
	// kep: http://kep.k8s.io/2433
	// alpha: v1.21
	// beta: v1.23
	//
	// Enables topology aware hints for EndpointSlices
	TopologyAwareHints featuregate.Feature = "TopologyAwareHints"

	// owner: @ehashman
	// alpha: v1.21
	//
	// Allows user to override pod-level terminationGracePeriod for probes
	ProbeTerminationGracePeriod featuregate.Feature = "ProbeTerminationGracePeriod"

	// owner: @ehashman
	// alpha: v1.22
	//
	// Permits kubelet to run with swap enabled
	NodeSwap featuregate.Feature = "NodeSwap"

	// owner: @ahg-g
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.24
	//
	// Allow specifying NamespaceSelector in PodAffinityTerm.
	PodAffinityNamespaceSelector featuregate.Feature = "PodAffinityNamespaceSelector"

	// owner: @andrewsykim @XudongLiuHarold
	// kep: http://kep.k8s.io/1959
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.24
	//
	// Enable support multiple Service "type: LoadBalancer" implementations in a cluster by specifying LoadBalancerClass
	ServiceLoadBalancerClass featuregate.Feature = "ServiceLoadBalancerClass"

	// owner: @damemi
	// alpha: v1.21
	// beta: v1.22
	//
	// Enables scaling down replicas via logarithmic comparison of creation/ready timestamps
	LogarithmicScaleDown featuregate.Feature = "LogarithmicScaleDown"

	// owner: @hbagdi
	// kep: http://kep.k8s.io/2365
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.23
	//
	// Enable Scope and Namespace fields on IngressClassParametersReference.
	IngressClassNamespacedParams featuregate.Feature = "IngressClassNamespacedParams"

	// owner: @maplain @andrewsykim
	// kep: http://kep.k8s.io/2086
	// alpha: v1.21
	// beta: v1.22
	//
	// Enables node-local routing for Service internal traffic
	ServiceInternalTrafficPolicy featuregate.Feature = "ServiceInternalTrafficPolicy"

	// owner: @adtac
	// alpha: v1.21
	// beta: v1.22
	// GA: v1.24
	//
	// Allows jobs to be created in the suspended state.
	SuspendJob featuregate.Feature = "SuspendJob"

	// owner: @fromanirh
	// alpha: v1.21
	// beta: v1.23
	// Enable POD resources API to return allocatable resources
	KubeletPodResourcesGetAllocatable featuregate.Feature = "KubeletPodResourcesGetAllocatable"

	// owner: @fengzixu
	// alpha: v1.21
	//
	// Enables kubelet to detect CSI volume condition and send the event of the abnormal volume to the corresponding pod that is using it.
	CSIVolumeHealth featuregate.Feature = "CSIVolumeHealth"

	// owner: @marosset
	// alpha: v1.22
	// beta: v1.23
	//
	// Enables support for 'HostProcess' containers on Windows nodes.
	WindowsHostProcessContainers featuregate.Feature = "WindowsHostProcessContainers"

	// owner: @ravig
	// kep: https://kep.k8s.io/2607
	// alpha: v1.22
	// beta: v1.23
	// StatefulSetMinReadySeconds allows minReadySeconds to be respected by StatefulSet controller
	StatefulSetMinReadySeconds featuregate.Feature = "StatefulSetMinReadySeconds"

	// owner: @ravig
	// alpha: v1.23
	// beta: v1.24
	// IdentifyPodOS allows user to specify OS on which they'd like the Pod run. The user should still set the nodeSelector
	// with appropriate `kubernetes.io/os` label for scheduler to identify appropriate node for the pod to run.
	IdentifyPodOS featuregate.Feature = "IdentifyPodOS"

	// owner: @gjkim42
	// kep: http://kep.k8s.io/2595
	// alpha: v1.22
	//
	// Enables apiserver and kubelet to allow up to 32 DNSSearchPaths and up to 2048 DNSSearchListChars.
	ExpandedDNSConfig featuregate.Feature = "ExpandedDNSConfig"

	// owner: @saschagrunert
	// alpha: v1.22
	//
	// Enables the use of `RuntimeDefault` as the default seccomp profile for all workloads.
	SeccompDefault featuregate.Feature = "SeccompDefault"

	// owner: @liggitt, @tallclair, sig-auth
	// alpha: v1.22
	// beta: v1.23
	//
	// Enables the PodSecurity admission plugin
	PodSecurity featuregate.Feature = "PodSecurity"

	// owner: @chrishenzie
	// alpha: v1.22
	//
	// Enables usage of the ReadWriteOncePod PersistentVolume access mode.
	ReadWriteOncePod featuregate.Feature = "ReadWriteOncePod"

	// owner: @enj
	// beta: v1.22
	// ga: v1.24
	//
	// Allows clients to request a duration for certificates issued via the Kubernetes CSR API.
	CSRDuration featuregate.Feature = "CSRDuration"

	// owner: @AkihiroSuda
	// alpha: v1.22
	//
	// Enables support for running kubelet in a user namespace.
	// The user namespace has to be created before running kubelet.
	// All the node components such as CRI need to be running in the same user namespace.
	KubeletInUserNamespace featuregate.Feature = "KubeletInUserNamespace"

	// owner: @xiaoxubeii
	// kep: http://kep.k8s.io/2570
	// alpha: v1.22
	//
	// Enables kubelet to support memory QoS with cgroups v2.
	MemoryQoS featuregate.Feature = "MemoryQoS"

	// owner: @fromanirh
	// alpha: v1.22
	// beta: v1.23
	//
	// Allow the usage of options to fine-tune the cpumanager policies.
	CPUManagerPolicyOptions featuregate.Feature = "CPUManagerPolicyOptions"

	// owner: @jiahuif
	// alpha: v1.21
	// beta:  v1.22
	// GA:    v1.24
	//
	// Enables Leader Migration for kube-controller-manager and cloud-controller-manager
	ControllerManagerLeaderMigration featuregate.Feature = "ControllerManagerLeaderMigration"

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

	// owner: @ahg
	// beta: v1.23
	//
	// Allow updating node scheduling directives in the pod template of jobs. Specifically,
	// node affinity, selector and tolerations. This is allowed only for suspended jobs
	// that have never been unsuspended before.
	JobMutableNodeSchedulingDirectives featuregate.Feature = "JobMutableNodeSchedulingDirectives"

	// owner: @haircommander
	// kep: http://kep.k8s.io/2364
	// alpha: v1.23
	//
	// Configures the Kubelet to use the CRI to populate pod and container stats, instead of supplimenting with stats from cAdvisor.
	// Requires the CRI implementation supports supplying the required stats.
	PodAndContainerStatsFromCRI featuregate.Feature = "PodAndContainerStatsFromCRI"

	// owner: @deepakkinni @xing-yang
	// kep: http://kep.k8s.io/2680
	// alpha: v1.23
	//
	// Honor Persistent Volume Reclaim Policy when it is "Delete" irrespective of PV-PVC
	// deletion ordering.
	HonorPVReclaimPolicy featuregate.Feature = "HonorPVReclaimPolicy"

	// owner: @gnufied
	// kep: http://kep.k8s.io/1790
	// alpha: v1.23
	//
	// Allow users to recover from volume expansion failure
	RecoverVolumeExpansionFailure featuregate.Feature = "RecoverVolumeExpansionFailure"

	// owner: @yuzhiquan, @bowei, @PxyUp, @SergeyKanzhelev
	// kep: http://kep.k8s.io/2727
	// alpha: v1.23
	// beta: v1.24
	//
	// Enables GRPC probe method for {Liveness,Readiness,Startup}Probe.
	GRPCContainerProbe featuregate.Feature = "GRPCContainerProbe"

	// owner: @zshihang
	// kep: http://kep.k8s.io/2800
	// beta: v1.24
	//
	// Stop auto-generation of secret-based service account tokens.
	LegacyServiceAccountTokenNoAutoGeneration featuregate.Feature = "LegacyServiceAccountTokenNoAutoGeneration"

	// owner: @sanposhiho
	// kep: http://kep.k8s.io/3022
	// alpha: v1.24
	//
	// Enable MinDomains in Pod Topology Spread.
	MinDomainsInPodTopologySpread featuregate.Feature = "MinDomainsInPodTopologySpread"

	// owner: @aojea
	// kep: http://kep.k8s.io/3070
	// alpha: v1.24
	//
	// Subdivide the ClusterIP range for dynamic and static IP allocation.
	ServiceIPStaticSubrange featuregate.Feature = "ServiceIPStaticSubrange"

	// owner: @xing-yang @sonasingh46
	// kep: http://kep.k8s.io/2268
	// alpha: v1.24
	//
	// Allow pods to failover to a different node in case of non graceful node shutdown
	NodeOutOfServiceVolumeDetach featuregate.Feature = "NodeOutOfServiceVolumeDetach"

	// owner: @krmayankk
	// alpha: v1.24
	//
	// Enables maxUnavailable for StatefulSet
	MaxUnavailableStatefulSet featuregate.Feature = "MaxUnavailableStatefulSet"

	// owner: @rikatz
	// kep: http://kep.k8s.io/2943
	// alpha: v1.24
	//
	// Enables NetworkPolicy status subresource
	NetworkPolicyStatus featuregate.Feature = "NetworkPolicyStatus"

	// owner: @deejross
	// kep: http://kep.k8s.io/3140
	// alpha: v1.24
	//
	// Enables support for time zones in CronJobs.
	CronJobTimeZone featuregate.Feature = "CronJobTimeZone"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AppArmor:             {Default: true, PreRelease: featuregate.Beta},
	DynamicKubeletConfig: {Default: false, PreRelease: featuregate.Deprecated}, // feature gate is deprecated in 1.22, kubelet logic is removed in 1.24, api server logic can be removed in 1.26
	ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: featuregate.Beta},
	DevicePlugins:                                  {Default: true, PreRelease: featuregate.Beta},
	RotateKubeletServerCertificate:                 {Default: true, PreRelease: featuregate.Beta},
	LocalStorageCapacityIsolation:                  {Default: true, PreRelease: featuregate.Beta},
	EphemeralContainers:                            {Default: true, PreRelease: featuregate.Beta},
	QOSReserved:                                    {Default: false, PreRelease: featuregate.Alpha},
	ExpandPersistentVolumes:                        {Default: true, PreRelease: featuregate.GA}, // remove in 1.25
	ExpandInUsePersistentVolumes:                   {Default: true, PreRelease: featuregate.GA}, // remove in 1.25
	ExpandCSIVolumes:                               {Default: true, PreRelease: featuregate.GA}, // remove in 1.25
	CPUManager:                                     {Default: true, PreRelease: featuregate.Beta},
	MemoryManager:                                  {Default: true, PreRelease: featuregate.Beta},
	CPUCFSQuotaPeriod:                              {Default: false, PreRelease: featuregate.Alpha},
	TopologyManager:                                {Default: true, PreRelease: featuregate.Beta},
	StorageObjectInUseProtection:                   {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	CSIMigration:                                   {Default: true, PreRelease: featuregate.Beta},
	CSIMigrationGCE:                                {Default: true, PreRelease: featuregate.Beta}, // On by default in 1.23 (requires GCE PD CSI Driver)
	InTreePluginGCEUnregister:                      {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAWS:                                {Default: true, PreRelease: featuregate.Beta},
	InTreePluginAWSUnregister:                      {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAzureDisk:                          {Default: true, PreRelease: featuregate.GA}, // On by default in 1.23 (requires Azure Disk CSI driver)
	InTreePluginAzureDiskUnregister:                {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationAzureFile:                          {Default: true, PreRelease: featuregate.Beta}, // On by default in 1.24 (requires Azure File CSI driver)
	InTreePluginAzureFileUnregister:                {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationvSphere:                            {Default: false, PreRelease: featuregate.Beta}, // Off by default (requires vSphere CSI driver)
	InTreePluginvSphereUnregister:                  {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationOpenStack:                          {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	InTreePluginOpenStackUnregister:                {Default: false, PreRelease: featuregate.Alpha},
	CSIMigrationRBD:                                {Default: false, PreRelease: featuregate.Alpha}, // Off by default (requires RBD CSI driver)
	InTreePluginRBDUnregister:                      {Default: false, PreRelease: featuregate.Alpha},
	ConfigurableFSGroupPolicy:                      {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	CSIMigrationPortworx:                           {Default: false, PreRelease: featuregate.Alpha},                  // Off by default (requires Portworx CSI driver)
	InTreePluginPortworxUnregister:                 {Default: false, PreRelease: featuregate.Alpha},
	CSIInlineVolume:                                {Default: true, PreRelease: featuregate.Beta},
	CSIStorageCapacity:                             {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	CSIServiceAccountToken:                         {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.23
	GenericEphemeralVolume:                         {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	CSIVolumeFSGroupPolicy:                         {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	VolumeSubpath:                                  {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	NetworkPolicyEndPort:                           {Default: true, PreRelease: featuregate.Beta},
	ProcMountType:                                  {Default: false, PreRelease: featuregate.Alpha},
	TTLAfterFinished:                               {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	IndexedJob:                                     {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	JobTrackingWithFinalizers:                      {Default: false, PreRelease: featuregate.Beta},                   // Disabled due to #109485
	JobReadyPods:                                   {Default: true, PreRelease: featuregate.Beta},
	KubeletPodResources:                            {Default: true, PreRelease: featuregate.Beta},
	LocalStorageCapacityIsolationFSQuotaMonitoring: {Default: false, PreRelease: featuregate.Alpha},
	NonPreemptingPriority:                          {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	PodOverhead:                                    {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	IPv6DualStack:                                  {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	EndpointSlice:                                  {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	EndpointSliceProxying:                          {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	EndpointSliceTerminatingCondition:              {Default: true, PreRelease: featuregate.Beta},
	ProxyTerminatingEndpoints:                      {Default: false, PreRelease: featuregate.Alpha},
	EndpointSliceNodeName:                          {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	WindowsEndpointSliceProxying:                   {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	PodDisruptionBudget:                            {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	DaemonSetUpdateSurge:                           {Default: true, PreRelease: featuregate.Beta},                    // on by default in 1.22
	DownwardAPIHugePages:                           {Default: true, PreRelease: featuregate.Beta},                    // on by default in 1.22
	AnyVolumeDataSource:                            {Default: true, PreRelease: featuregate.Beta},                    // on by default in 1.24
	DefaultPodTopologySpread:                       {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	WinOverlay:                                     {Default: true, PreRelease: featuregate.Beta},
	WinDSR:                                         {Default: false, PreRelease: featuregate.Alpha},
	DisableAcceleratorUsageMetrics:                 {Default: true, PreRelease: featuregate.Beta},
	HPAContainerMetrics:                            {Default: false, PreRelease: featuregate.Alpha},
	SizeMemoryBackedVolumes:                        {Default: true, PreRelease: featuregate.Beta},
	ExecProbeTimeout:                               {Default: true, PreRelease: featuregate.GA}, // lock to default and remove after v1.22 based on KEP #1972 update
	KubeletCredentialProviders:                     {Default: true, PreRelease: featuregate.Beta},
	GracefulNodeShutdown:                           {Default: true, PreRelease: featuregate.Beta},
	GracefulNodeShutdownBasedOnPodPriority:         {Default: true, PreRelease: featuregate.Beta},
	ServiceLBNodePortControl:                       {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	MixedProtocolLBService:                         {Default: true, PreRelease: featuregate.Beta},
	VolumeCapacityPriority:                         {Default: false, PreRelease: featuregate.Alpha},
	PreferNominatedNode:                            {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	ProbeTerminationGracePeriod:                    {Default: false, PreRelease: featuregate.Beta},                   // Default to false in beta 1.22, set to true in 1.24
	NodeSwap:                                       {Default: false, PreRelease: featuregate.Alpha},
	PodDeletionCost:                                {Default: true, PreRelease: featuregate.Beta},
	StatefulSetAutoDeletePVC:                       {Default: false, PreRelease: featuregate.Alpha},
	TopologyAwareHints:                             {Default: true, PreRelease: featuregate.Beta},
	PodAffinityNamespaceSelector:                   {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	ServiceLoadBalancerClass:                       {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	IngressClassNamespacedParams:                   {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.25
	ServiceInternalTrafficPolicy:                   {Default: true, PreRelease: featuregate.Beta},
	LogarithmicScaleDown:                           {Default: true, PreRelease: featuregate.Beta},
	SuspendJob:                                     {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	KubeletPodResourcesGetAllocatable:              {Default: true, PreRelease: featuregate.Beta},
	CSIVolumeHealth:                                {Default: false, PreRelease: featuregate.Alpha},
	WindowsHostProcessContainers:                   {Default: true, PreRelease: featuregate.Beta},
	DisableCloudProviders:                          {Default: false, PreRelease: featuregate.Alpha},
	DisableKubeletCloudCredentialProviders:         {Default: false, PreRelease: featuregate.Alpha},
	StatefulSetMinReadySeconds:                     {Default: true, PreRelease: featuregate.Beta},
	ExpandedDNSConfig:                              {Default: false, PreRelease: featuregate.Alpha},
	SeccompDefault:                                 {Default: false, PreRelease: featuregate.Alpha},
	PodSecurity:                                    {Default: true, PreRelease: featuregate.Beta},
	ReadWriteOncePod:                               {Default: false, PreRelease: featuregate.Alpha},
	CSRDuration:                                    {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	DelegateFSGroupToCSIDriver:                     {Default: true, PreRelease: featuregate.Beta},
	KubeletInUserNamespace:                         {Default: false, PreRelease: featuregate.Alpha},
	MemoryQoS:                                      {Default: false, PreRelease: featuregate.Alpha},
	CPUManagerPolicyOptions:                        {Default: true, PreRelease: featuregate.Beta},
	ControllerManagerLeaderMigration:               {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.26
	CPUManagerPolicyAlphaOptions:                   {Default: false, PreRelease: featuregate.Alpha},
	CPUManagerPolicyBetaOptions:                    {Default: true, PreRelease: featuregate.Beta},
	JobMutableNodeSchedulingDirectives:             {Default: true, PreRelease: featuregate.Beta},
	IdentifyPodOS:                                  {Default: true, PreRelease: featuregate.Beta},
	PodAndContainerStatsFromCRI:                    {Default: false, PreRelease: featuregate.Alpha},
	HonorPVReclaimPolicy:                           {Default: false, PreRelease: featuregate.Alpha},
	RecoverVolumeExpansionFailure:                  {Default: false, PreRelease: featuregate.Alpha},
	GRPCContainerProbe:                             {Default: true, PreRelease: featuregate.Beta},
	LegacyServiceAccountTokenNoAutoGeneration:      {Default: true, PreRelease: featuregate.Beta},
	MinDomainsInPodTopologySpread:                  {Default: false, PreRelease: featuregate.Alpha},
	ServiceIPStaticSubrange:                        {Default: false, PreRelease: featuregate.Alpha},
	NodeOutOfServiceVolumeDetach:                   {Default: false, PreRelease: featuregate.Alpha},
	MaxUnavailableStatefulSet:                      {Default: false, PreRelease: featuregate.Alpha},
	NetworkPolicyStatus:                            {Default: false, PreRelease: featuregate.Alpha},
	CronJobTimeZone:                                {Default: false, PreRelease: featuregate.Alpha},

	// inherited features from generic apiserver, relisted here to get a conflict if it is changed
	// unintentionally on either side:
	genericfeatures.AdvancedAuditing:                    {Default: true, PreRelease: featuregate.GA},
	genericfeatures.APIResponseCompression:              {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.APIListChunking:                     {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.DryRun:                              {Default: true, PreRelease: featuregate.GA},
	genericfeatures.ServerSideApply:                     {Default: true, PreRelease: featuregate.GA},
	genericfeatures.APIPriorityAndFairness:              {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.OpenAPIEnums:                        {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.CustomResourceValidationExpressions: {Default: false, PreRelease: featuregate.Alpha},
	genericfeatures.OpenAPIV3:                           {Default: true, PreRelease: featuregate.Beta},
	genericfeatures.ServerSideFieldValidation:           {Default: false, PreRelease: featuregate.Alpha},
	// features that enable backwards compatibility but are scheduled to be removed
	// ...
	HPAScaleToZero: {Default: false, PreRelease: featuregate.Alpha},
}
