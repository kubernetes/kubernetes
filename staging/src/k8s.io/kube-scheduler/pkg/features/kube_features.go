/*
Copyright The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
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
	// owner: @tosi3k
	// kep: https://kep.k8s.io/6012
	//
	// Enables support for CompositePodGroups.
	CompositePodGroup featuregate.Feature = "CompositePodGroup"

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

	// owner: @gauravkghildiyal
	// kep: http://kep.k8s.io/6080
	//
	// Enables support for derived attributes in Dynamic Resource Allocation (DRA).
	DRADerivedAttributes featuregate.Feature = "DRADerivedAttributes"

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

	// owner: @omeryahud
	// kep: https://kep.k8s.io/5963
	//
	// Enables drivers to declare opaque compatibility groups on each
	// device.consumesCounters[] entry of a ResourceSlice. The scheduler then
	// only co-allocates devices drawing from the same counter set when their
	// declared groups intersect, moving detection of incompatible co-allocation
	// (e.g. GPU MIG vs vGPU on one physical device) from preparation-time
	// failure to scheduling-time rejection.
	//
	// DRAPartitionableDevices also needs to be enabled, since the field lives
	// on consumesCounters[] entries which only exist for partitionable devices.
	DRADeviceCompatibilityGroups featuregate.Feature = "DRADeviceCompatibilityGroups"

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

	// owner: @sunya-ch
	// kep: https://kep.k8s.io/5075
	//
	// Enables fractional (milli-unit) values in CapacityRequestPolicyRange
	// min, max, and step fields.
	DRAFractionalCapacityRange featuregate.Feature = "DRAFractionalCapacityRange"

	// owner: @everpeace
	// kep: http://kep.k8s.io/5491
	//
	// Enable list type attributes for DRA devices in ResourceSlice
	// and extends ResourceClaim's matchAttribute/distinctAttribute
	// semantics so that they can work with list type attributes.
	DRAListTypeAttributes featuregate.Feature = "DRAListTypeAttributes"

	// owner: @pravk03
	// kep: https://kep.k8s.io/5517
	//
	// Enables support for node allocatable resources backed by DRA.
	DRANodeAllocatableResources featuregate.Feature = "DRANodeAllocatableResources"

	// owner: @troychiu
	// kep: http://kep.k8s.io/5945
	//
	// Enables support for declaring that node-local operations (preparation and
	// clean-up) are optional for devices.
	DRAOptionalNodeOperations featuregate.Feature = "DRAOptionalNodeOperations"

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

	// owner: @nojnhuh
	// kep: https://kep.k8s.io/5729
	//
	// Enables support for reserving and replicating templated ResourceClaims for an entire PodGroup.
	DRAWorkloadResourceClaims featuregate.Feature = "DRAWorkloadResourceClaims"

	// owner: @pohly
	// kep: http://kep.k8s.io/4381
	//
	// Enables support for resources with custom parameters and a lifecycle
	// that is independent of a Pod. Resource allocation is done by the scheduler
	// based on "structured parameters".
	DynamicResourceAllocation featuregate.Feature = "DynamicResourceAllocation"

	// owner: @erictune @wojtek-t
	//
	// Enables support for generic Workload API.
	GenericWorkload featuregate.Feature = "GenericWorkload"

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

	// owner: @natasha41575
	// kep: https://kep.k8s.io/5836
	//
	// Enables scheduler-triggered preemption for deferred in-place pod vertical scaling pods.
	InPlacePodVerticalScalingSchedulerPreemption featuregate.Feature = "InPlacePodVerticalScalingSchedulerPreemption"

	// owner: @tetianakh
	//
	// Enables the fast path for inter-pod affinity calculations when the topology key is kubernetes.io/hostname.
	InterPodAffinityHostnameFastPath featuregate.Feature = "InterPodAffinityHostnameFastPath"

	// owner: @denkensk
	// kep: https://kep.k8s.io/3243
	//
	// Enable MatchLabelKeys in PodTopologySpread.
	MatchLabelKeysInPodTopologySpread featuregate.Feature = "MatchLabelKeysInPodTopologySpread"

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

	// owner: @wojtek-t @argh4k
	// kep: https://kep.k8s.io/5710
	//
	// Enables specifying PreemptionPolicy at podgroup level.
	PodGroupPreemptionPolicy featuregate.Feature = "PodGroupPreemptionPolicy"

	// owner: @ndixita
	// key: https://kep.k8s.io/2837
	//
	// Enables specifying resources at pod-level.
	PodLevelResources featuregate.Feature = "PodLevelResources"

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

	// owner: @geetasg
	// kep: https://kep.k8s.io/6132
	//
	// Enables PreQueueingHint extension point to narrow pod evaluation on events.
	SchedulerPreQueueingHints featuregate.Feature = "SchedulerPreQueueingHints"

	// owner: @cupnes
	// kep: https://kep.k8s.io/4049
	//
	// Enables scoring nodes by available storage capacity with
	// StorageCapacityScoring feature gate.
	StorageCapacityScoring featuregate.Feature = "StorageCapacityScoring"

	// owner: @helayoty
	// kep: https://kep.k8s.io/5471
	//
	// Enables numeric comparison operators (Lt, Gt) for tolerations to match taints with threshold-based values.
	TaintTolerationComparisonOperators featuregate.Feature = "TaintTolerationComparisonOperators"

	// owner: @44past4
	// kep: https://kep.k8s.io/5732
	//
	// Enables topology-aware workload scheduling feature in kube-scheduler and related PodGroup API fields.
	// When enabled, scheduler will try various placements for a pod group and pick the best one.
	TopologyAwareWorkloadScheduling featuregate.Feature = "TopologyAwareWorkloadScheduling"

	// owner: @mattcarry, @sunnylovestiramisu
	// kep: https://kep.k8s.io/3751
	//
	// Enables user specified volume attributes for persistent volumes, like iops and throughput.
	VolumeAttributesClass featuregate.Feature = "VolumeAttributesClass"

	// owner: @gnufied
	// kep: https://kep.k8s.io/5030
	// alpha: v1.35
	// beta: v1.37
	//
	// Enables volume limit scaling for CSI drivers. This allows scheduler to
	// co-ordinate better with cluster-autoscaler for storage limits.
	VolumeLimitScaling featuregate.Feature = "VolumeLimitScaling"
)

// SetupCurrentKubernetesSpecificFeatureGates adds the scheduler feature gates to the provided feature gate.
func SetupCurrentKubernetesSpecificFeatureGates(featureGates featuregate.MutableVersionedFeatureGate) error {
	return featureGates.AddVersioned(defaultVersionedKubernetesFeatureGates)
}

// defaultVersionedKubernetesFeatureGates consists of all known scheduler-specific feature keys with VersionedSpecs.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
//
// Entries are alphabetized and separated from each other with blank lines to avoid sweeping gofmt changes
// when adding or removing one entry.
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	CompositePodGroup: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRAAdminAccess: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.36; remove in 1.39
	},

	DRAConsumableCapacity: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.Beta},
	},

	DRADerivedAttributes: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRADeviceBindingConditions: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.Beta},
	},

	DRADeviceCompatibilityGroups: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRADeviceTaintRules: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: false, PreRelease: featuregate.Beta},                    // Depends on an off-by-default beta API.
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.GA, LockToDefault: false}, // LockToDefault: true in 1.38; remove in 1.41
	},

	DRADeviceTaints: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.GA, LockToDefault: false}, // LockToDefault: true in 1.38; remove in 1.41
	},

	DRAExtendedResource: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA in 1.37; remove in 1.40
	},

	DRAFractionalCapacityRange: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.Beta, MinCompatibilityVersion: version.MustParse("1.37")},
	},

	DRAListTypeAttributes: {
		{Version: version.MustParse("1.36"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRANodeAllocatableResources: {
		{Version: version.MustParse("1.36"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRAOptionalNodeOperations: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	DRAPartitionableDevices: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.Beta},
	},

	DRAPrioritizedList: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
		// Remove completely in 1.40.
	},

	DRAResourceClaimDeviceStatus: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.40
	},

	DRASchedulerFilterTimeout: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	DRAWorkloadResourceClaims: {
		{Version: version.MustParse("1.36"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Beta},
	},

	DynamicResourceAllocation: {
		{Version: version.MustParse("1.26"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
		// TODO (https://github.com/kubernetes/kubernetes/issues/134459): remove completely in 1.38
	},

	GenericWorkload: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Beta},
	},

	InPlacePodLevelResourcesVerticalScaling: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.Beta},
	},

	InPlacePodVerticalScaling: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.38
	},

	InPlacePodVerticalScalingSchedulerPreemption: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	InterPodAffinityHostnameFastPath: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	MatchLabelKeysInPodTopologySpread: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
	},

	NodeDeclaredFeatures: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	NodeInclusionPolicyInPodTopologySpread: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	NominatedNodeNameForExpectation: {
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	OpportunisticBatching: {
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	PodGroupPreemptionPolicy: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	PodLevelResources: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
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

	SchedulerPreQueueingHints: {
		{Version: version.MustParse("1.37"), Default: false, PreRelease: featuregate.Alpha},
	},

	StorageCapacityScoring: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.Beta},
	},

	TaintTolerationComparisonOperators: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	TopologyAwareWorkloadScheduling: {
		{Version: version.MustParse("1.36"), Default: false, PreRelease: featuregate.Alpha},
	},

	VolumeAttributesClass: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.36"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	VolumeLimitScaling: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.37"), Default: true, PreRelease: featuregate.Beta},
	},
}
