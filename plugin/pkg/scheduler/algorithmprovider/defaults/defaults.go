/*
Copyright 2014 The Kubernetes Authors.

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

package defaults

import (
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"

	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/plugin/pkg/scheduler/core"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"

	"github.com/golang/glog"
)

const (

	// ClusterAutoscalerProvider defines the default autoscaler provider
	ClusterAutoscalerProvider = "ClusterAutoscalerProvider"
	// StatefulSetKind defines the name of 'StatefulSet' kind
	StatefulSetKind = "StatefulSet"
)

func init() {
	// Register functions that extract metadata used by predicates and priorities computations.
	factory.RegisterPredicateMetadataProducerFactory(
		func(args factory.PluginFactoryArgs) algorithm.PredicateMetadataProducer {
			return predicates.NewPredicateMetadataFactory(args.PodLister)
		})
	factory.RegisterPriorityMetadataProducerFactory(
		func(args factory.PluginFactoryArgs) algorithm.MetadataProducer {
			return priorities.PriorityMetadata
		})

	registerAlgorithmProvider(defaultPredicates(), defaultPriorities())

	// IMPORTANT NOTES for predicate developers:
	// We are using cached predicate result for pods belonging to the same equivalence class.
	// So when implementing a new predicate, you are expected to check whether the result
	// of your predicate function can be affected by related API object change (ADD/DELETE/UPDATE).
	// If yes, you are expected to invalidate the cached predicate result for related API object change.
	// For example:
	// https://github.com/kubernetes/kubernetes/blob/36a218e/plugin/pkg/scheduler/factory/factory.go#L422

	// Registers predicates and priorities that are not enabled by default, but user can pick when creating his
	// own set of priorities/predicates.

	// PodFitsPorts has been replaced by PodFitsHostPorts for better user understanding.
	// For backwards compatibility with 1.0, PodFitsPorts is registered as well.
	factory.RegisterFitPredicate("PodFitsPorts", predicates.PodFitsHostPorts)
	// Fit is defined based on the absence of port conflicts.
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	factory.RegisterFitPredicate("PodFitsHostPorts", predicates.PodFitsHostPorts)
	// Fit is determined by resource availability.
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	factory.RegisterFitPredicate("PodFitsResources", predicates.PodFitsResources)
	// Fit is determined by the presence of the Host parameter and a string match
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	factory.RegisterFitPredicate("HostName", predicates.PodFitsHost)
	// Fit is determined by node selector query.
	factory.RegisterFitPredicate("MatchNodeSelector", predicates.PodMatchNodeSelector)

	// Use equivalence class to speed up predicates & priorities
	factory.RegisterGetEquivalencePodFunction(predicates.GetEquivalencePod)

	// ServiceSpreadingPriority is a priority config factory that spreads pods by minimizing
	// the number of pods (belonging to the same service) on the same node.
	// Register the factory so that it's available, but do not include it as part of the default priorities
	// Largely replaced by "SelectorSpreadPriority", but registered for backward compatibility with 1.0
	factory.RegisterPriorityConfigFactory(
		"ServiceSpreadingPriority",
		factory.PriorityConfigFactory{
			Function: func(args factory.PluginFactoryArgs) algorithm.PriorityFunction {
				return priorities.NewSelectorSpreadPriority(args.ServiceLister, algorithm.EmptyControllerLister{}, algorithm.EmptyReplicaSetLister{}, algorithm.EmptyStatefulSetLister{})
			},
			Weight: 1,
		},
	)

	// EqualPriority is a prioritizer function that gives an equal weight of one to all nodes
	// Register the priority function so that its available
	// but do not include it as part of the default priorities
	factory.RegisterPriorityFunction2("EqualPriority", core.EqualPriorityMap, nil, 1)
	// ImageLocalityPriority prioritizes nodes based on locality of images requested by a pod. Nodes with larger size
	// of already-installed packages required by the pod will be preferred over nodes with no already-installed
	// packages required by the pod or a small total size of already-installed packages required by the pod.
	factory.RegisterPriorityFunction2("ImageLocalityPriority", priorities.ImageLocalityPriorityMap, nil, 1)
	// Optional, cluster-autoscaler friendly priority function - give used nodes higher priority.
	factory.RegisterPriorityFunction2("MostRequestedPriority", priorities.MostRequestedPriorityMap, nil, 1)
}

func defaultPredicates() sets.String {
	return sets.NewString(
		// Fit is determined by volume zone requirements.
		factory.RegisterFitPredicateFactory(
			"NoVolumeZoneConflict",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewVolumeZonePredicate(args.PVInfo, args.PVCInfo)
			},
		),
		// Fit is determined by whether or not there would be too many AWS EBS volumes attached to the node
		factory.RegisterFitPredicateFactory(
			"MaxEBSVolumeCount",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewMaxPDVolumeCountPredicate(predicates.EBSVolumeFilterType, args.PVInfo, args.PVCInfo)
			},
		),
		// Fit is determined by whether or not there would be too many GCE PD volumes attached to the node
		factory.RegisterFitPredicateFactory(
			"MaxGCEPDVolumeCount",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewMaxPDVolumeCountPredicate(predicates.GCEPDVolumeFilterType, args.PVInfo, args.PVCInfo)
			},
		),
		// Fit is determined by whether or not there would be too many Azure Disk volumes attached to the node
		factory.RegisterFitPredicateFactory(
			"MaxAzureDiskVolumeCount",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewMaxPDVolumeCountPredicate(predicates.AzureDiskVolumeFilterType, args.PVInfo, args.PVCInfo)
			},
		),
		// Fit is determined by inter-pod affinity.
		factory.RegisterFitPredicateFactory(
			predicates.MatchInterPodAffinity,
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewPodAffinityPredicate(args.NodeInfo, args.PodLister)
			},
		),

		// Fit is determined by non-conflicting disk volumes.
		factory.RegisterFitPredicate("NoDiskConflict", predicates.NoDiskConflict),

		// GeneralPredicates are the predicates that are enforced by all Kubernetes components
		// (e.g. kubelet and all schedulers)
		factory.RegisterFitPredicate("GeneralPredicates", predicates.GeneralPredicates),

		// Fit is determined by node memory pressure condition.
		factory.RegisterFitPredicate("CheckNodeMemoryPressure", predicates.CheckNodeMemoryPressurePredicate),

		// Fit is determined by node disk pressure condition.
		factory.RegisterFitPredicate("CheckNodeDiskPressure", predicates.CheckNodeDiskPressurePredicate),

		// Fit is determied by node condtions: not ready, network unavailable and out of disk.
		factory.RegisterMandatoryFitPredicate("CheckNodeCondition", predicates.CheckNodeConditionPredicate),

		// Fit is determined based on whether a pod can tolerate all of the node's taints
		factory.RegisterFitPredicate("PodToleratesNodeTaints", predicates.PodToleratesNodeTaints),

		// Fit is determined by volume zone requirements.
		factory.RegisterFitPredicateFactory(
			"NoVolumeNodeConflict",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewVolumeNodePredicate(args.PVInfo, args.PVCInfo, nil)
			},
		),
	)
}

// ApplyFeatureGates applies algorithm by feature gates.
func ApplyFeatureGates() {
	predSet := defaultPredicates()

	if utilfeature.DefaultFeatureGate.Enabled(features.TaintNodesByCondition) {
		// Remove "CheckNodeCondition" predicate
		factory.RemoveFitPredicate("CheckNodeCondition")
		predSet.Delete("CheckNodeCondition")

		// Fit is determined based on whether a pod can tolerate all of the node's taints
		predSet.Insert(factory.RegisterMandatoryFitPredicate("PodToleratesNodeTaints", predicates.PodToleratesNodeTaints))

		glog.Warningf("TaintNodesByCondition is enabled, PodToleratesNodeTaints predicate is mandatory")
	}

	registerAlgorithmProvider(predSet, defaultPriorities())
}

func registerAlgorithmProvider(predSet, priSet sets.String) {
	// Registers algorithm providers. By default we use 'DefaultProvider', but user can specify one to be used
	// by specifying flag.
	factory.RegisterAlgorithmProvider(factory.DefaultProvider, predSet, priSet)
	// Cluster autoscaler friendly scheduling algorithm.
	factory.RegisterAlgorithmProvider(ClusterAutoscalerProvider, predSet,
		copyAndReplace(priSet, "LeastRequestedPriority", "MostRequestedPriority"))
}

func defaultPriorities() sets.String {
	return sets.NewString(
		// spreads pods by minimizing the number of pods (belonging to the same service or replication controller) on the same node.
		factory.RegisterPriorityConfigFactory(
			"SelectorSpreadPriority",
			factory.PriorityConfigFactory{
				Function: func(args factory.PluginFactoryArgs) algorithm.PriorityFunction {
					return priorities.NewSelectorSpreadPriority(args.ServiceLister, args.ControllerLister, args.ReplicaSetLister, args.StatefulSetLister)
				},
				Weight: 1,
			},
		),
		// pods should be placed in the same topological domain (e.g. same node, same rack, same zone, same power domain, etc.)
		// as some other pods, or, conversely, should not be placed in the same topological domain as some other pods.
		factory.RegisterPriorityConfigFactory(
			"InterPodAffinityPriority",
			factory.PriorityConfigFactory{
				Function: func(args factory.PluginFactoryArgs) algorithm.PriorityFunction {
					return priorities.NewInterPodAffinityPriority(args.NodeInfo, args.NodeLister, args.PodLister, args.HardPodAffinitySymmetricWeight)
				},
				Weight: 1,
			},
		),

		// Prioritize nodes by least requested utilization.
		factory.RegisterPriorityFunction2("LeastRequestedPriority", priorities.LeastRequestedPriorityMap, nil, 1),

		// Prioritizes nodes to help achieve balanced resource usage
		factory.RegisterPriorityFunction2("BalancedResourceAllocation", priorities.BalancedResourceAllocationMap, nil, 1),

		// Set this weight large enough to override all other priority functions.
		// TODO: Figure out a better way to do this, maybe at same time as fixing #24720.
		factory.RegisterPriorityFunction2("NodePreferAvoidPodsPriority", priorities.CalculateNodePreferAvoidPodsPriorityMap, nil, 10000),

		// Prioritizes nodes that have labels matching NodeAffinity
		factory.RegisterPriorityFunction2("NodeAffinityPriority", priorities.CalculateNodeAffinityPriorityMap, priorities.CalculateNodeAffinityPriorityReduce, 1),

		// Prioritizes nodes that marked with taint which pod can tolerate.
		factory.RegisterPriorityFunction2("TaintTolerationPriority", priorities.ComputeTaintTolerationPriorityMap, priorities.ComputeTaintTolerationPriorityReduce, 1),
	)
}

func copyAndReplace(set sets.String, replaceWhat, replaceWith string) sets.String {
	result := sets.NewString(set.List()...)
	if result.Has(replaceWhat) {
		result.Delete(replaceWhat)
		result.Insert(replaceWith)
	}
	return result
}
