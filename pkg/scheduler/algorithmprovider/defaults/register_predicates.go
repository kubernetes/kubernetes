/*
Copyright 2018 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/factory"
)

func init() {
	// Register functions that extract metadata used by predicates computations.
	factory.RegisterPredicateMetadataProducerFactory(
		func(args factory.PluginFactoryArgs) predicates.PredicateMetadataProducer {
			return predicates.NewPredicateMetadataFactory(args.PodLister)
		})

	// IMPORTANT NOTES for predicate developers:
	// Registers predicates and priorities that are not enabled by default, but user can pick when creating their
	// own set of priorities/predicates.

	// PodFitsPorts has been replaced by PodFitsHostPorts for better user understanding.
	// For backwards compatibility with 1.0, PodFitsPorts is registered as well.
	factory.RegisterFitPredicate("PodFitsPorts", predicates.PodFitsHostPorts)
	// Fit is defined based on the absence of port conflicts.
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	factory.RegisterFitPredicate(predicates.PodFitsHostPortsPred, predicates.PodFitsHostPorts)
	// Fit is determined by resource availability.
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	factory.RegisterFitPredicate(predicates.PodFitsResourcesPred, predicates.PodFitsResources)
	// Fit is determined by the presence of the Host parameter and a string match
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	factory.RegisterFitPredicate(predicates.HostNamePred, predicates.PodFitsHost)
	// Fit is determined by node selector query.
	factory.RegisterFitPredicate(predicates.MatchNodeSelectorPred, predicates.PodMatchNodeSelector)

	// Fit is determined by volume zone requirements.
	factory.RegisterFitPredicateFactory(
		predicates.NoVolumeZoneConflictPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewVolumeZonePredicate(args.PVInfo, args.PVCInfo, args.StorageClassInfo)
		},
	)
	// Fit is determined by whether or not there would be too many AWS EBS volumes attached to the node
	factory.RegisterFitPredicateFactory(
		predicates.MaxEBSVolumeCountPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.EBSVolumeFilterType, args.PVInfo, args.PVCInfo)
		},
	)
	// Fit is determined by whether or not there would be too many GCE PD volumes attached to the node
	factory.RegisterFitPredicateFactory(
		predicates.MaxGCEPDVolumeCountPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.GCEPDVolumeFilterType, args.PVInfo, args.PVCInfo)
		},
	)
	// Fit is determined by whether or not there would be too many Azure Disk volumes attached to the node
	factory.RegisterFitPredicateFactory(
		predicates.MaxAzureDiskVolumeCountPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.AzureDiskVolumeFilterType, args.PVInfo, args.PVCInfo)
		},
	)
	factory.RegisterFitPredicateFactory(
		predicates.MaxCSIVolumeCountPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewCSIMaxVolumeLimitPredicate(args.PVInfo, args.PVCInfo, args.StorageClassInfo)
		},
	)
	factory.RegisterFitPredicateFactory(
		predicates.MaxCinderVolumeCountPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.CinderVolumeFilterType, args.PVInfo, args.PVCInfo)
		},
	)

	// Fit is determined by inter-pod affinity.
	factory.RegisterFitPredicateFactory(
		predicates.MatchInterPodAffinityPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewPodAffinityPredicate(args.NodeInfo, args.PodLister)
		},
	)

	// Fit is determined by non-conflicting disk volumes.
	factory.RegisterFitPredicate(predicates.NoDiskConflictPred, predicates.NoDiskConflict)

	// GeneralPredicates are the predicates that are enforced by all Kubernetes components
	// (e.g. kubelet and all schedulers)
	factory.RegisterFitPredicate(predicates.GeneralPred, predicates.GeneralPredicates)

	// Fit is determined by node memory pressure condition.
	factory.RegisterFitPredicate(predicates.CheckNodeMemoryPressurePred, predicates.CheckNodeMemoryPressurePredicate)

	// Fit is determined by node disk pressure condition.
	factory.RegisterFitPredicate(predicates.CheckNodeDiskPressurePred, predicates.CheckNodeDiskPressurePredicate)

	// Fit is determined by node pid pressure condition.
	factory.RegisterFitPredicate(predicates.CheckNodePIDPressurePred, predicates.CheckNodePIDPressurePredicate)

	// Fit is determined by node conditions: not ready, network unavailable or out of disk.
	factory.RegisterMandatoryFitPredicate(predicates.CheckNodeConditionPred, predicates.CheckNodeConditionPredicate)

	// Fit is determined based on whether a pod can tolerate all of the node's taints
	factory.RegisterFitPredicate(predicates.PodToleratesNodeTaintsPred, predicates.PodToleratesNodeTaints)

	// Fit is determined by volume topology requirements.
	factory.RegisterFitPredicateFactory(
		predicates.CheckVolumeBindingPred,
		func(args factory.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewVolumeBindingPredicate(args.VolumeBinder)
		},
	)
}
