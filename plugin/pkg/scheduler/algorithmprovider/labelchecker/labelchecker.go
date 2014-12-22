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
package labelchecker

import (
	algorithm "github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/factory"
)

const Provider string = "LabelCheckerProvider"

func init() {
	factory.RegisterAlgorithmProvider(Provider, defaultPredicates(), defaultPriorities())
}

func defaultPredicates() util.StringSet {
	return util.NewStringSet(
		// Fit is defined based on the presence/absence of a label on a minion, regardless of value.
		factory.RegisterFitPredicate("NodeLabelPredicate", algorithm.NewNodeLabelPredicate(factory.MinionLister, []string{"region"}, true)),
	)
}

func defaultPriorities() util.StringSet {
	return util.NewStringSet(
		// Prioritize nodes based on the presence/absence of a label on a minion, regardless of value.
		factory.RegisterPriorityFunction("NodeLabelPriority", algorithm.NewNodeLabelPriority("", true), 1),
	)
}
