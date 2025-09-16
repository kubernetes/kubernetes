/*
Copyright 2023 The Kubernetes Authors.

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

// Package feature contains pre-defined features used by test/e2e and/or
// test/e2e_node.
package feature

import (
	"k8s.io/kubernetes/test/e2e/framework"
)

// Please keep the list in alphabetical, case-sensitive
// (upper before any lower case character) order.
var (
	// TODO: document the feature (owning SIG, when to use this feature for a test)
	APIServerIdentity = framework.WithFeature(framework.ValidFeatures.Add("APIServerIdentity"))

	// Owner: sig-lifecycle
	// This label is used for tests which need the following controllers to be enabled:
	// - bootstrap-signer-controller
	// - token-cleaner-controller
	BootstrapTokens = framework.WithFeature(framework.ValidFeatures.Add("BootstrapTokens"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	BoundServiceAccountTokenVolume = framework.WithFeature(framework.ValidFeatures.Add("BoundServiceAccountTokenVolume"))

	// Owner: sig-api-machinery
	// Marks tests that exercise the CBOR data format for serving or storage.
	CBOR = framework.WithFeature(framework.ValidFeatures.Add("CBOR"))

	// owning-sig: sig-node
	// kep: https://kep.k8s.io/3570
	// test-infra jobs:
	// - https://testgrid.k8s.io/sig-node-kubelet#kubelet-serial-gce-e2e-cpu-manager
	//
	// This label is used for tests which need:
	// - run only CPU Manager tests on specific jobs, i.e., ci-kubernetes-node-kubelet-serial-cpu-manager and pull-kubernetes-node-kubelet-serial-cpu-manager
	CPUManager = framework.WithFeature(framework.ValidFeatures.Add("CPUManager"))

	// Owner: sig-node
	// Marks test that exercise checkpointing of containers
	CheckpointContainer = framework.WithFeature(framework.ValidFeatures.Add("CheckpointContainer"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	CloudProvider = framework.WithFeature(framework.ValidFeatures.Add("CloudProvider"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	ClusterDowngrade = framework.WithFeature(framework.ValidFeatures.Add("ClusterDowngrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	ClusterScaleUpBypassScheduler = framework.WithFeature(framework.ValidFeatures.Add("ClusterScaleUpBypassScheduler"))

	// Owner: sig-autoscaling
	ClusterSizeAutoscalingScaleDown = framework.WithFeature(framework.ValidFeatures.Add("ClusterSizeAutoscalingScaleDown"))

	// Owner: sig-autoscaling
	ClusterSizeAutoscalingScaleUp = framework.WithFeature(framework.ValidFeatures.Add("ClusterSizeAutoscalingScaleUp"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	ClusterUpgrade = framework.WithFeature(framework.ValidFeatures.Add("ClusterUpgrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	ComprehensiveNamespaceDraining = framework.WithFeature(framework.ValidFeatures.Add("ComprehensiveNamespaceDraining"))

	// Onwer: sig-node
	// Enables configuring per-container restart policy and restart policy rules.
	ContainerRestartRules = framework.WithFeature(framework.ValidFeatures.Add("ContainerRestartRules"))

	// Owner: sig-node
	// Enables configuring custom stop signals for containers from container lifecycle
	ContainerStopSignals = framework.WithFeature(framework.ValidFeatures.Add("ContainerStopSignals"))

	// Owner: sig-api-machinery
	// Marks tests that require coordinated leader election
	CoordinatedLeaderElection = framework.WithFeature(framework.ValidFeatures.Add("CoordinatedLeaderElection"))

	// Owner: sig-node
	// Tests marked with this feature MUST run with the CRI Proxy configured so errors can be injected into the kubelet's CRI calls.
	// This is useful for testing how the kubelet handles various error conditions in its CRI interactions.
	// test-infra jobs:
	// - pull-kubernetes-node-e2e-cri-proxy-serial (need manual trigger)
	// - ci-kubernetes-node-e2e-cri-proxy-serial
	CriProxy = framework.WithFeature(framework.ValidFeatures.Add("CriProxy"))

	// OWNER: sig-node
	// Testing critical pod admission
	CriticalPod = framework.WithFeature(framework.ValidFeatures.Add("CriticalPod"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	CustomMetricsAutoscaling = framework.WithFeature(framework.ValidFeatures.Add("CustomMetricsAutoscaling"))

	// OWNER: sig-node
	// Testing device managers
	DeviceManager = framework.WithFeature(framework.ValidFeatures.Add("DeviceManager"))

	// OWNER: sig-node
	// Testing device plugins
	DevicePlugin = framework.WithFeature(framework.ValidFeatures.Add("DevicePlugin"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Downgrade = framework.WithFeature(framework.ValidFeatures.Add("Downgrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	// OWNER: sig-node
	// Testing downward API huge pages
	DownwardAPIHugePages = framework.WithFeature(framework.ValidFeatures.Add("DownwardAPIHugePages"))

	// owning-sig: sig-node
	// kep: https://kep.k8s.io/4381
	// test-infra jobs:
	// - https://testgrid.k8s.io/sig-node-dynamic-resource-allocation
	//
	// This feature label is used for tests which need:
	// - kubelet support for plugins with the "usual" paths under /var/lib/kubelet
	// - a container runtime where support for CDI (https://github.com/cncf-tags/container-device-interface)
	//   is enabled such that passing CDI device IDs through CRI fields is supported
	//
	// In addition, tests must be labeled with framework.WithFeatureGate to document
	// their dependency on specific feature gates and the corresponding API groups.
	DynamicResourceAllocation = framework.WithFeature(framework.ValidFeatures.Add("DynamicResourceAllocation"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	EphemeralStorage = framework.WithFeature(framework.ValidFeatures.Add("EphemeralStorage"))

	// OWNER: sig-node
	// Testing eviction manager
	Eviction = framework.WithFeature(framework.ValidFeatures.Add("Eviction"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Example = framework.WithFeature(framework.ValidFeatures.Add("Example"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	ExperimentalResourceUsageTracking = framework.WithFeature(framework.ValidFeatures.Add("ExperimentalResourceUsageTracking"))

	// OWNER: sig-storage
	// These tests need kube-controller-manager that can execute a shell (bash). Most Kubernetes e2e
	// tests run with kube-controller-manager as a distroless container without such a shell.
	// If you need to run these tests,  please build your own image with required packages (like bash).
	// See https://github.com/kubernetes/kubernetes/issues/78737 for more details.
	Flexvolumes = framework.WithFeature(framework.ValidFeatures.Add("Flexvolumes"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	GKENodePool = framework.WithFeature(framework.ValidFeatures.Add("GKENodePool"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	GPUClusterDowngrade = framework.WithFeature(framework.ValidFeatures.Add("GPUClusterDowngrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	GPUClusterUpgrade = framework.WithFeature(framework.ValidFeatures.Add("GPUClusterUpgrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	GPUDevicePlugin = framework.WithFeature(framework.ValidFeatures.Add("GPUDevicePlugin"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	GPUMasterUpgrade = framework.WithFeature(framework.ValidFeatures.Add("GPUMasterUpgrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	GPUUpgrade = framework.WithFeature(framework.ValidFeatures.Add("GPUUpgrade"))

	// OWNER: sig-node
	// Testing garbage collection of images/containers
	GarbageCollect = framework.WithFeature(framework.ValidFeatures.Add("GarbageCollect"))

	// OWNER: sig-node
	// Testing graceful node shutdown
	GracefulNodeShutdown = framework.WithFeature(framework.ValidFeatures.Add("GracefulNodeShutdown"))

	// OWNER: sig-node
	// GracefulNodeShutdown based on pod priority
	GracefulNodeShutdownBasedOnPodPriority = framework.WithFeature(framework.ValidFeatures.Add("GracefulNodeShutdownBasedOnPodPriority"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	HAMaster = framework.WithFeature(framework.ValidFeatures.Add("HAMaster"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	HPA = framework.WithFeature(framework.ValidFeatures.Add("HPA"))

	// OWNER: sig-autoscaling
	// Marks tests that require HPA configurable tolerance (https://kep.k8s.io/4951).
	HPAConfigurableTolerance = framework.WithFeature(framework.ValidFeatures.Add("HPAConfigurableTolerance"))

	// owner: sig-node
	HostAccess = framework.WithFeature(framework.ValidFeatures.Add("HostAccess"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	HugePages = framework.WithFeature(framework.ValidFeatures.Add("HugePages"))

	// Owner: sig-network
	// Marks tests that require a cluster with dual-stack pod and service networks.
	IPv6DualStack = framework.WithFeature(framework.ValidFeatures.Add("IPv6DualStack"))

	// Owner: sig-node
	ImageID = framework.WithFeature(framework.ValidFeatures.Add("ImageID"))

	// Owner: sig-node
	// ImageVolume is used for testing the image volume source feature (https://kep.k8s.io/4639).
	ImageVolume = framework.WithFeature(framework.ValidFeatures.Add("ImageVolume"))

	// Owner: sig-network
	// Marks tests that require a conforming implementation of
	// Ingress.networking.k8s.io to be present.
	Ingress = framework.WithFeature(framework.ValidFeatures.Add("Ingress"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Kind = framework.WithFeature(framework.ValidFeatures.Add("Kind"))

	// Owner: sig-network
	// Marks tests that require kube-dns-autoscaler
	KubeDNSAutoscaler = framework.WithFeature(framework.ValidFeatures.Add("KubeDNSAutoscaler"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	KubeProxyDaemonSetDowngrade = framework.WithFeature(framework.ValidFeatures.Add("KubeProxyDaemonSetDowngrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	KubeProxyDaemonSetMigration = framework.WithFeature(framework.ValidFeatures.Add("KubeProxyDaemonSetMigration"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	KubeProxyDaemonSetUpgrade = framework.WithFeature(framework.ValidFeatures.Add("KubeProxyDaemonSetUpgrade"))

	// Owner: sig-network
	// Marks tests that require the kernel to have support for the nfacct subsystem.
	// (Some distros don't include this in the kernel.)
	KubeProxyNFAcct = framework.WithFeature(framework.ValidFeatures.Add("KubeProxyNFAcct"))

	// Owner: sig-node
	// Testing kubelet drop in KEP
	KubeletConfigDropInDir = framework.WithFeature(framework.ValidFeatures.Add("KubeletConfigDropInDir"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	KubeletCredentialProviders = framework.WithFeature(framework.ValidFeatures.Add("KubeletCredentialProviders"))

	// Owner: sig-node
	// Testing kubelet PSI metrics KEP
	KubeletPSI = framework.WithFeature(framework.ValidFeatures.Add("KubeletPSI"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	KubeletSecurity = framework.WithFeature(framework.ValidFeatures.Add("KubeletSecurity"))

	// KubeletSeparateDiskGC (SIG-node, used for testing separate image filesystem <https://kep.k8s.io/4191>)
	// The tests need separate disk settings on nodes and separate filesystems in storage.conf
	KubeletSeparateDiskGC = framework.WithFeature(framework.ValidFeatures.Add("KubeletSeparateDiskGC"))

	// Owner: sig-storage
	LSCIQuotaMonitoring = framework.WithFeature(framework.ValidFeatures.Add("LSCIQuotaMonitoring"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	LabelSelector = framework.WithFeature(framework.ValidFeatures.Add("LabelSelector"))

	// Owner: sig-network
	// Marks tests that require a cloud provider that implements LoadBalancer Services
	LoadBalancer = framework.WithFeature(framework.ValidFeatures.Add("LoadBalancer"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	LocalStorageCapacityIsolationQuota = framework.WithFeature(framework.ValidFeatures.Add("LocalStorageCapacityIsolationQuota"))

	// owning-sig: sig-node
	// Marks a disruptive test for lock contention
	LockContention = framework.WithFeature(framework.ValidFeatures.Add("LockContention"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	MasterUpgrade = framework.WithFeature(framework.ValidFeatures.Add("MasterUpgrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	MemoryManager = framework.WithFeature(framework.ValidFeatures.Add("MemoryManager"))

	// Owner: sig-api-machinery
	// Marks tests that enforce ordered namespace deletion.
	MutatingAdmissionPolicy = framework.WithFeature(framework.ValidFeatures.Add("MutatingAdmissionPolicy"))

	// Owner: sig-network
	// Marks tests that require a conforming implementation of
	// NetworkPolicy.networking.k8s.io to be present.
	NetworkPolicy = framework.WithFeature(framework.ValidFeatures.Add("NetworkPolicy"))

	// Owner: sig-network
	// Marks tests that require working external DNS.
	NetworkingDNS = framework.WithFeature(framework.ValidFeatures.Add("Networking-DNS"))

	// Owner: sig-network
	// Marks tests that require connectivity to the Internet via IPv4
	NetworkingIPv4 = framework.WithFeature(framework.ValidFeatures.Add("Networking-IPv4"))

	// Owner: sig-network
	// Marks tests that require connectivity to the Internet via IPv6
	NetworkingIPv6 = framework.WithFeature(framework.ValidFeatures.Add("Networking-IPv6"))

	// Owner: sig-network
	// Marks a single test that creates potentially-disruptive amounts of network
	// traffic between nodes.
	NetworkingPerformance = framework.WithFeature(framework.ValidFeatures.Add("Networking-Performance"))

	// Owner: sig-node
	// Testing node allocatable validations
	NodeAllocatable = framework.WithFeature(framework.ValidFeatures.Add("NodeAllocatable"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	NodeAuthorizer = framework.WithFeature(framework.ValidFeatures.Add("NodeAuthorizer"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	NodeLogQuery = framework.WithFeature(framework.ValidFeatures.Add("NodeLogQuery"))

	// Owner: sig-node
	// Node Problem Detect e2e tests in tree.
	NodeProblemDetector = framework.WithFeature(framework.ValidFeatures.Add("NodeProblemDetector"))

	// Owner: sig-node
	// Tests aiming to verify oom_score functionality
	OOMScoreAdj = framework.WithFeature(framework.ValidFeatures.Add("OOMScoreAdj"))

	// Owner: sig-network
	// Marks a single test that tests cluster DNS performance with many services.
	PerformanceDNS = framework.WithFeature(framework.ValidFeatures.Add("PerformanceDNS"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	PodGarbageCollector = framework.WithFeature(framework.ValidFeatures.Add("PodGarbageCollector"))

	// owner: sig-node
	// Marks a test for for pod-level resources feature that requires
	// PodLevelResources feature gate to be enabled.
	PodLevelResources = framework.WithFeature(framework.ValidFeatures.Add("PodLevelResources"))

	// Owner: sig-node
	// Marks tests that require a cluster with PodLogsQuerySplitStreams
	// (used for testing specific log stream <https://kep.k8s.io/3288>)
	PodLogsQuerySplitStreams = framework.WithFeature(framework.ValidFeatures.Add("PodLogsQuerySplitStreams"))

	// Owner: sig-node
	// Marks tests that require a cluster with PodObservedGenerationTracking
	PodObservedGenerationTracking = framework.WithFeature(framework.ValidFeatures.Add("PodObservedGenerationTracking"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	PodPriority = framework.WithFeature(framework.ValidFeatures.Add("PodPriority"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	PodReadyToStartContainersCondition = framework.WithFeature(framework.ValidFeatures.Add("PodReadyToStartContainersCondition"))

	// Owner: sig-node
	// Marks tests which exercise or consume the kubelet-local Pod Resources API
	// see: KEPs 606, 2043; see: pkg/kubelet/apis/podresources/
	PodResourcesAPI = framework.WithFeature(framework.ValidFeatures.Add("PodResourcesAPI"))

	// Owner: sig-node
	// Verify ProcMount feature.
	// Used in combination with user namespaces
	ProcMountType = framework.WithFeature(framework.ValidFeatures.Add("ProcMountType"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Reboot = framework.WithFeature(framework.ValidFeatures.Add("Reboot"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	ReclaimPolicy = framework.WithFeature(framework.ValidFeatures.Add("ReclaimPolicy"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	RegularResourceUsageTracking = framework.WithFeature(framework.ValidFeatures.Add("RegularResourceUsageTracking"))

	// Owner: sig-network
	// Marks tests of KEP-4427 that require the `RelaxedDNSSearchValidation` feature gate
	RelaxedDNSSearchValidation = framework.WithFeature(framework.ValidFeatures.Add("RelaxedDNSSearchValidation"))

	// Owner: sig-node
	// resource health Status for device plugins and DRA <https://kep.k8s.io/4680>
	ResourceHealthStatus = framework.WithFeature(framework.ValidFeatures.Add("ResourceHealthStatus"))

	// Owner: sig-node
	// Device Management metrics
	ResourceMetrics = framework.WithFeature(framework.ValidFeatures.Add("ResourceMetrics"))

	// Owner: sig-node
	// Runtime Handler
	RuntimeHandler = framework.WithFeature(framework.ValidFeatures.Add("RuntimeHandler"))

	// Owner: sig-network
	// Marks tests that require a pod networking implementation that supports SCTP
	// traffic between pods.
	SCTPConnectivity = framework.WithFeature(framework.ValidFeatures.Add("SCTPConnectivity"))

	// Owner: sig-storage
	// This feature marks tests that need all schedulable Linux nodes in the cluster to have SELinux enabled.
	SELinux = framework.WithFeature(framework.ValidFeatures.Add("SELinux"))

	// Owner: sig-storage
	// This feature marks tests that need SELinuxMountReadWriteOncePod feature gate enabled and SELinuxMount **disabled**.
	// This is a temporary feature to allow testing of metrics when SELinuxMount is disabled.
	// TODO: remove when SELinuxMount feature gate is enabled by default.
	SELinuxMountReadWriteOncePodOnly = framework.WithFeature(framework.ValidFeatures.Add("SELinuxMountReadWriteOncePodOnly"))

	// Owner: sig-scheduling
	// Marks tests of the asynchronous preemption (KEP-4832) that require the `SchedulerAsyncPreemption` feature gate.
	SchedulerAsyncPreemption = framework.WithFeature(framework.ValidFeatures.Add("SchedulerAsyncPreemption"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	SeccompDefault = framework.WithFeature(framework.ValidFeatures.Add("SeccompDefault"))

	// Owner: sig-auth
	// Marks tests that require a conforming implementation of
	// Node claims for serviceaccounts. Typically this means that the
	// ServiceAccountTokenNodeBindingValidation feature must be enabled.
	ServiceAccountTokenNodeBindingValidation = framework.WithFeature(framework.ValidFeatures.Add("ServiceAccountTokenNodeBindingValidation"))

	// Owner: sig-network
	// Marks tests of KEP-1880 that require the `MultiCIDRServiceAllocator` feature gate
	// and the networking.k8s.io/v1alpha1 API.
	ServiceCIDRs = framework.WithFeature(framework.ValidFeatures.Add("ServiceCIDRs"))

	// Owner: sig-node
	// Sidecar KEP-753
	SidecarContainers = framework.WithFeature(framework.ValidFeatures.Add("SidecarContainers"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StackdriverAcceleratorMonitoring = framework.WithFeature(framework.ValidFeatures.Add("StackdriverAcceleratorMonitoring"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StackdriverCustomMetrics = framework.WithFeature(framework.ValidFeatures.Add("StackdriverCustomMetrics"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StackdriverExternalMetrics = framework.WithFeature(framework.ValidFeatures.Add("StackdriverExternalMetrics"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StackdriverMetadataAgent = framework.WithFeature(framework.ValidFeatures.Add("StackdriverMetadataAgent"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StackdriverMonitoring = framework.WithFeature(framework.ValidFeatures.Add("StackdriverMonitoring"))

	// Tests marked with this feature require the kubelet to be running in standalone mode (--standalone-mode=true) like this:
	// make test-e2e-node PARALLELISM=1 FOCUS="StandaloneMode" TEST_ARGS='--kubelet-flags="--fail-swap-on=false" --standalone-mode=true'
	// Tests validating the behavior of kubelet when running without the API server.
	StandaloneMode = framework.WithFeature(framework.ValidFeatures.Add("StandaloneMode"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StatefulSet = framework.WithFeature(framework.ValidFeatures.Add("StatefulSet"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StatefulUpgrade = framework.WithFeature(framework.ValidFeatures.Add("StatefulUpgrade"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StorageProvider = framework.WithFeature(framework.ValidFeatures.Add("StorageProvider"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	StorageVersionAPI = framework.WithFeature(framework.ValidFeatures.Add("StorageVersionAPI"))

	// Owner: sig-node
	// Marks tests that require a cluster with SupplementalGroupsPolicy
	// (used for testing fine-grained SupplementalGroups control <https://kep.k8s.io/3619>)
	SupplementalGroupsPolicy = framework.WithFeature(framework.ValidFeatures.Add("SupplementalGroupsPolicy"))

	// Added to test Swap Feature
	// This label should be used when testing KEP-2400 (Node Swap Support)
	Swap = framework.WithFeature(framework.ValidFeatures.Add("NodeSwap"))

	SystemNodeCriticalPod = framework.WithFeature(framework.ValidFeatures.Add("SystemNodeCriticalPod"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	TopologyManager = framework.WithFeature(framework.ValidFeatures.Add("TopologyManager"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Upgrade = framework.WithFeature(framework.ValidFeatures.Add("Upgrade"))

	// Owned by SIG Node
	// Can be used when the UserNamespacesPodSecurityStandards kubelet feature
	// gate is enabled to relax the application of Pod Security Standards in a
	// controlled way.
	UserNamespacesPodSecurityStandards = framework.WithFeature(framework.ValidFeatures.Add("UserNamespacesPodSecurityStandards"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	UserNamespacesSupport = framework.WithFeature(framework.ValidFeatures.Add("UserNamespacesSupport"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	ValidatingAdmissionPolicy = framework.WithFeature(framework.ValidFeatures.Add("ValidatingAdmissionPolicy"))

	// Owner: sig-storage
	// TODO: Remove it once the csi driver is promoted to GA and the manifest is updated.
	VolumeAttributesClass = framework.WithFeature(framework.ValidFeatures.Add("VolumeAttributesClass"))

	// Owner: sig-storage
	// Volume group snapshot tests
	VolumeGroupSnapshotDataSource = framework.WithFeature(framework.ValidFeatures.Add("volumegroupsnapshot"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	VolumeSnapshotDataSource = framework.WithFeature(framework.ValidFeatures.Add("VolumeSnapshotDataSource"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	VolumeSourceXFS = framework.WithFeature(framework.ValidFeatures.Add("VolumeSourceXFS"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Volumes = framework.WithFeature(framework.ValidFeatures.Add("Volumes"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Vsphere = framework.WithFeature(framework.ValidFeatures.Add("vsphere"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	Windows = framework.WithFeature(framework.ValidFeatures.Add("Windows"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	WindowsHostProcessContainers = framework.WithFeature(framework.ValidFeatures.Add("WindowsHostProcessContainers"))

	// TODO: document the feature (owning SIG, when to use this feature for a test)
	WindowsHyperVContainers = framework.WithFeature(framework.ValidFeatures.Add("WindowsHyperVContainers"))
)

func init() {
	// This prevents adding additional ad-hoc features in tests.
	framework.ValidFeatures.Freeze()
}
