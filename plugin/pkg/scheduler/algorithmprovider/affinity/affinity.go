/*
Copyright 2014 Google Inc. All rights reserved.

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

// This algorithm provider has predicates and priorities related to affinity/anti-affinity for the scheduler.
package affinity

import (
	algorithm "github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"
)

const AffinityProvider string = "AffinityProvider"

func init() {
	factory.RegisterAlgorithmProvider(AffinityProvider, affinityPredicates(), affinityPriorities())
}

func affinityPredicates() util.StringSet {
	return util.NewStringSet(
		"HostName",
		"MatchNodeSelector",
		"PodFitsPorts",
		"PodFitsResources",
		"NoDiskConflict",
		// Ensures that all pods within the same service are hosted on minions within the same region as defined by the "region" label
		factory.RegisterFitPredicate("RegionAffinity", algorithm.NewServiceAffinityPredicate(factory.PodLister, factory.ServiceLister, factory.MinionLister, []string{"region"})),
		// Fit is defined based on the presence of the "region" label on a minion, regardless of value.
		factory.RegisterFitPredicate("RegionRequired", algorithm.NewNodeLabelPredicate(factory.MinionLister, []string{"region"}, true)),
	)
}

func affinityPriorities() util.StringSet {
	return util.NewStringSet(
		"LeastRequestedPriority",
		"ServiceSpreadingPriority",
		// spreads pods belonging to the same service across minions in different zones
		factory.RegisterPriorityFunction("ZoneSpread", algorithm.NewServiceAntiAffinityPriority(factory.ServiceLister, "zone"), 2),
		// Prioritize nodes based on the presence of the "zone" label on a minion, regardless of value.
		factory.RegisterPriorityFunction("ZonePreferred", algorithm.NewNodeLabelPriority("zone", true), 1),
	)
}
