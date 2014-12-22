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

// This is the default algorithm provider for the scheduler.
package defaults

import (
	algorithm "github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"
)

func init() {
	factory.RegisterAlgorithmProvider(factory.DefaultProvider, defaultPredicates(), defaultPriorities())
}

func defaultPredicates() util.StringSet {
	return util.NewStringSet(
		// Fit is defined based on the absence of port conflicts.
		factory.RegisterFitPredicate("PodFitsPorts", algorithm.PodFitsPorts),
		// Fit is determined by resource availability.
		factory.RegisterFitPredicate("PodFitsResources", algorithm.NewResourceFitPredicate(factory.MinionLister)),
		// Fit is determined by non-conflicting disk volumes.
		factory.RegisterFitPredicate("NoDiskConflict", algorithm.NoDiskConflict),
		// Fit is determined by node selector query.
		factory.RegisterFitPredicate("MatchNodeSelector", algorithm.NewSelectorMatchPredicate(factory.MinionLister)),
		// Fit is determined by the presence of the Host parameter and a string match
		factory.RegisterFitPredicate("HostName", algorithm.PodFitsHost),
	)
}

func defaultPriorities() util.StringSet {
	return util.NewStringSet(
		// Prioritize nodes by least requested utilization.
		factory.RegisterPriorityFunction("LeastRequestedPriority", algorithm.LeastRequestedPriority, 1),
		// spreads pods by minimizing the number of pods (belonging to the same service) on the same minion.
		factory.RegisterPriorityFunction("ServiceSpreadingPriority", algorithm.NewServiceSpreadPriority(factory.ServiceLister), 1),
		// spreads pods belonging to the same service across minions in different zones
		// TODO: remove the hardcoding of the "zone" label and move it to a constant
		factory.RegisterPriorityFunction("ZoneSpreadingPriority", algorithm.NewServiceAntiAffinityPriority(factory.ServiceLister, "zone"), 1),
		// EqualPriority is a prioritizer function that gives an equal weight of one to all minions
		factory.RegisterPriorityFunction("EqualPriority", algorithm.EqualPriority, 0),
	)
}
