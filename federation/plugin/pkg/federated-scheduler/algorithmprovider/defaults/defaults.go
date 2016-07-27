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

// This is the default algorithm provider for the federated-scheduler.
package defaults

import (
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm/predicates"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm/priorities"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/factory"
	"k8s.io/kubernetes/pkg/util/sets"
)

func init() {
	factory.RegisterAlgorithmProvider(factory.DefaultProvider, defaultPredicates(), defaultPriorities())
}

func defaultPredicates() sets.String {
	return sets.NewString(
		// Fit is determined by cluster selector query.
		factory.RegisterFitPredicateFactory(
			"MatchClusterSelector",
			func(args factory.PluginFactoryArgs) algorithm.FitPredicate {
				return predicates.NewSelectorMatchPredicate(args.ClusterInfo)
			},
		),
	)
}

func defaultPriorities() sets.String {
	return sets.NewString(
		factory.RegisterPriorityConfigFactory(
			"RandomChoosePriority",
			factory.PriorityConfigFactory{
				Function: func(args factory.PluginFactoryArgs) algorithm.PriorityFunction {
					return priorities.NewRandomChoosePriority(args.ClusterLister)
				},
				Weight: 1,
			},
		),
	)
}
