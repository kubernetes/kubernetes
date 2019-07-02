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
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/factory"
)

func init() {
	// Register functions that extract metadata used by priorities computations.
	factory.RegisterPriorityMetadataProducerFactory(
		func(args factory.PluginFactoryArgs) priorities.PriorityMetadataProducer {
			return priorities.NewPriorityMetadataFactory(args.ServiceLister, args.ControllerLister, args.ReplicaSetLister, args.StatefulSetLister)
		})

	// ServiceSpreadingPriority is a priority config factory that spreads pods by minimizing
	// the number of pods (belonging to the same service) on the same node.
	// Register the factory so that it's available, but do not include it as part of the default priorities
	// Largely replaced by "SelectorSpreadPriority", but registered for backward compatibility with 1.0
	factory.RegisterPriorityConfigFactory(
		priorities.ServiceSpreadingPriority,
		factory.PriorityConfigFactory{
			MapReduceFunction: func(args factory.PluginFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
				return priorities.NewSelectorSpreadPriority(args.ServiceLister, algorithm.EmptyControllerLister{}, algorithm.EmptyReplicaSetLister{}, algorithm.EmptyStatefulSetLister{})
			},
			Weight: 1,
		},
	)
	// EqualPriority is a prioritizer function that gives an equal weight of one to all nodes
	// Register the priority function so that its available
	// but do not include it as part of the default priorities
	factory.RegisterPriorityFunction2(priorities.EqualPriority, core.EqualPriorityMap, nil, 1)
	// Optional, cluster-autoscaler friendly priority function - give used nodes higher priority.
	factory.RegisterPriorityFunction2(priorities.MostRequestedPriority, priorities.MostRequestedPriorityMap, nil, 1)
	factory.RegisterPriorityFunction2(
		priorities.RequestedToCapacityRatioPriority,
		priorities.RequestedToCapacityRatioResourceAllocationPriorityDefault().PriorityMap,
		nil,
		1)
	// spreads pods by minimizing the number of pods (belonging to the same service or replication controller) on the same node.
	factory.RegisterPriorityConfigFactory(
		priorities.SelectorSpreadPriority,
		factory.PriorityConfigFactory{
			MapReduceFunction: func(args factory.PluginFactoryArgs) (priorities.PriorityMapFunction, priorities.PriorityReduceFunction) {
				return priorities.NewSelectorSpreadPriority(args.ServiceLister, args.ControllerLister, args.ReplicaSetLister, args.StatefulSetLister)
			},
			Weight: 1,
		},
	)
	// pods should be placed in the same topological domain (e.g. same node, same rack, same zone, same power domain, etc.)
	// as some other pods, or, conversely, should not be placed in the same topological domain as some other pods.
	factory.RegisterPriorityConfigFactory(
		priorities.InterPodAffinityPriority,
		factory.PriorityConfigFactory{
			Function: func(args factory.PluginFactoryArgs) priorities.PriorityFunction {
				return priorities.NewInterPodAffinityPriority(args.NodeInfo, args.NodeLister, args.PodLister, args.HardPodAffinitySymmetricWeight)
			},
			Weight: 1,
		},
	)

	// Prioritize nodes by least requested utilization.
	factory.RegisterPriorityFunction2(priorities.LeastRequestedPriority, priorities.LeastRequestedPriorityMap, nil, 1)

	// Prioritizes nodes to help achieve balanced resource usage
	factory.RegisterPriorityFunction2(priorities.BalancedResourceAllocation, priorities.BalancedResourceAllocationMap, nil, 1)

	// Set this weight large enough to override all other priority functions.
	// TODO: Figure out a better way to do this, maybe at same time as fixing #24720.
	factory.RegisterPriorityFunction2(priorities.NodePreferAvoidPodsPriority, priorities.CalculateNodePreferAvoidPodsPriorityMap, nil, 10000)

	// Prioritizes nodes that have labels matching NodeAffinity
	factory.RegisterPriorityFunction2(priorities.NodeAffinityPriority, priorities.CalculateNodeAffinityPriorityMap, priorities.CalculateNodeAffinityPriorityReduce, 1)

	// Prioritizes nodes that marked with taint which pod can tolerate.
	factory.RegisterPriorityFunction2(priorities.TaintTolerationPriority, priorities.ComputeTaintTolerationPriorityMap, priorities.ComputeTaintTolerationPriorityReduce, 1)

	// ImageLocalityPriority prioritizes nodes that have images requested by the pod present.
	factory.RegisterPriorityFunction2(priorities.ImageLocalityPriority, priorities.ImageLocalityPriorityMap, nil, 1)
}
