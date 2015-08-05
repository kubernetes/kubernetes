/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/plugin/pkg/scheduler/factory"
)

func init() {
	factory.RegisterAlgorithmProvider(factory.DefaultProvider, defaultPredicates(), defaultPriorities())
	// EqualPriority is a prioritizer function that gives an equal weight of one to all minions
	// Register the priority function so that its available
	// but do not include it as part of the default priorities
	factory.RegisterPriorityFunction("EqualPriority", scheduler.EqualPriority, 1)
}

func defaultPredicates() util.StringSet {
	return util.NewStringSet(
		// Fit is defined based on the absence of port conflicts.
		factory.RegisterFitPredicate("PodFitsPorts", predicates.PodFitsPorts),
		// Fit is determined by resource availability.
		factory.RegisterFitPredicateFactory(
			"PodFitsResources",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewResourceFitPredicate(args.NodeInfo)
			},
		),
		// Fit is determined by non-conflicting disk volumes.
		factory.RegisterFitPredicate("NoDiskConflict", predicates.NoDiskConflict),
		// Fit is determined by node selector query.
		factory.RegisterFitPredicateFactory(
			"MatchNodeSelector",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewSelectorMatchPredicate(args.NodeInfo)
			},
		),
		// Fit is determined by the presence of the Host parameter and a string match
		factory.RegisterFitPredicate("HostName", predicates.PodFitsHost),
	)
}

func defaultPriorities() util.StringSet {
	return util.NewStringSet(
		// Prioritize nodes by least requested utilization.
		factory.RegisterPriorityFunction("LeastRequestedPriority", priorities.LeastRequestedPriority, 1),
		// Prioritizes nodes to help achieve balanced resource usage
		factory.RegisterPriorityFunction("BalancedResourceAllocation", priorities.BalancedResourceAllocation, 1),
		// spreads pods by minimizing the number of pods (belonging to the same service) on the same minion.
		factory.RegisterPriorityConfigFactory(
			"SelectorSpreadPriority",
			factory.PriorityConfigFactory{
				Function: func(args factory.PluginFactoryArgs) algorithm.PriorityFunction {
					return priorities.NewSelectorSpreadPriority(args.ServiceLister, args.ControllerLister)
				},
				Weight: 1,
			},
		),
	)
}
