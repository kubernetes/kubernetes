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

const Provider string = "AffinityProvider"

func init() {
	factory.RegisterAlgorithmProvider(Provider, defaultPredicates(), defaultPriorities())
}

func defaultPredicates() util.StringSet {
	return util.NewStringSet(
		// Fit is defined based on whether the minion has the specified label values as the pod being scheduled
		// Alternately, if the pod does not specify any/all labels, the other pods in the service are looked at
		factory.RegisterFitPredicate("ServiceAffinity", algorithm.NewServiceAffinityPredicate(factory.PodLister, factory.ServiceLister, factory.MinionLister, []string{"region"})),
	)
}

func defaultPriorities() util.StringSet {
	return util.NewStringSet(
		// spreads pods belonging to the same service across minions in different zones
		// region and zone can be nested infrastructure topology levels and defined by labels on minions
		factory.RegisterPriorityFunction("ZoneSpreadingPriority", algorithm.NewServiceAntiAffinityPriority(factory.ServiceLister, "zone"), 1),
	)
}
