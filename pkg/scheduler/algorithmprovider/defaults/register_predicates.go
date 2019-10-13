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
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
)

func init() {
	// Register functions that extract metadata used by predicates computations.
	scheduler.RegisterPredicateMetadataProducer(predicates.GetPredicateMetadata)

	// IMPORTANT NOTES for predicate developers:
	// Registers predicates and priorities that are not enabled by default, but user can pick when creating their
	// own set of priorities/predicates.

	// PodFitsPorts has been replaced by PodFitsHostPorts for better user understanding.
	// For backwards compatibility with 1.0, PodFitsPorts is registered as well.
	scheduler.RegisterFitPredicate("PodFitsPorts", predicates.PodFitsHostPorts)
	// Fit is defined based on the absence of port conflicts.
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	scheduler.RegisterFitPredicate(predicates.PodFitsHostPortsPred, predicates.PodFitsHostPorts)
	// Fit is determined by resource availability.
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	scheduler.RegisterFitPredicate(predicates.PodFitsResourcesPred, predicates.PodFitsResources)
	// Fit is determined by the presence of the Host parameter and a string match
	// This predicate is actually a default predicate, because it is invoked from
	// predicates.GeneralPredicates()
	scheduler.RegisterFitPredicate(predicates.HostNamePred, predicates.PodFitsHost)
	// Fit is determined by node selector query.
	scheduler.RegisterFitPredicate(predicates.MatchNodeSelectorPred, predicates.PodMatchNodeSelector)

	// Fit is determined by volume zone requirements.
	scheduler.RegisterFitPredicateFactory(
		predicates.NoVolumeZoneConflictPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewVolumeZonePredicate(args.PVInfo, args.PVCInfo, args.StorageClassInfo)
		},
	)
	// Fit is determined by whether or not there would be too many AWS EBS volumes attached to the node
	scheduler.RegisterFitPredicateFactory(
		predicates.MaxEBSVolumeCountPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.EBSVolumeFilterType, args.CSINodeInfo, args.StorageClassInfo, args.PVInfo, args.PVCInfo)
		},
	)
	// Fit is determined by whether or not there would be too many GCE PD volumes attached to the node
	scheduler.RegisterFitPredicateFactory(
		predicates.MaxGCEPDVolumeCountPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.GCEPDVolumeFilterType, args.CSINodeInfo, args.StorageClassInfo, args.PVInfo, args.PVCInfo)
		},
	)
	// Fit is determined by whether or not there would be too many Azure Disk volumes attached to the node
	scheduler.RegisterFitPredicateFactory(
		predicates.MaxAzureDiskVolumeCountPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.AzureDiskVolumeFilterType, args.CSINodeInfo, args.StorageClassInfo, args.PVInfo, args.PVCInfo)
		},
	)
	scheduler.RegisterFitPredicateFactory(
		predicates.MaxCSIVolumeCountPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewCSIMaxVolumeLimitPredicate(args.CSINodeInfo, args.PVInfo, args.PVCInfo, args.StorageClassInfo)
		},
	)
	scheduler.RegisterFitPredicateFactory(
		predicates.MaxCinderVolumeCountPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewMaxPDVolumeCountPredicate(predicates.CinderVolumeFilterType, args.CSINodeInfo, args.StorageClassInfo, args.PVInfo, args.PVCInfo)
		},
	)

	// Fit is determined by inter-pod affinity.
	scheduler.RegisterFitPredicateFactory(
		predicates.MatchInterPodAffinityPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewPodAffinityPredicate(args.NodeInfo, args.PodLister)
		},
	)

	// Fit is determined by non-conflicting disk volumes.
	scheduler.RegisterFitPredicate(predicates.NoDiskConflictPred, predicates.NoDiskConflict)

	// GeneralPredicates are the predicates that are enforced by all Kubernetes components
	// (e.g. kubelet and all schedulers)
	scheduler.RegisterFitPredicate(predicates.GeneralPred, predicates.GeneralPredicates)

	// Fit is determined by node memory pressure condition.
	scheduler.RegisterFitPredicate(predicates.CheckNodeMemoryPressurePred, predicates.CheckNodeMemoryPressurePredicate)

	// Fit is determined by node disk pressure condition.
	scheduler.RegisterFitPredicate(predicates.CheckNodeDiskPressurePred, predicates.CheckNodeDiskPressurePredicate)

	// Fit is determined by node pid pressure condition.
	scheduler.RegisterFitPredicate(predicates.CheckNodePIDPressurePred, predicates.CheckNodePIDPressurePredicate)

	// Fit is determined by node conditions: not ready, network unavailable or out of disk.
	scheduler.RegisterMandatoryFitPredicate(predicates.CheckNodeConditionPred, predicates.CheckNodeConditionPredicate)

	// Fit is determined based on whether a pod can tolerate all of the node's taints
	scheduler.RegisterFitPredicate(predicates.PodToleratesNodeTaintsPred, predicates.PodToleratesNodeTaints)

	// Fit is determined by volume topology requirements.
	scheduler.RegisterFitPredicateFactory(
		predicates.CheckVolumeBindingPred,
		func(args scheduler.PluginFactoryArgs) predicates.FitPredicate {
			return predicates.NewVolumeBindingPredicate(args.VolumeBinder)
		},
	)
}
