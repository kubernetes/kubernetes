/*
Copyright 2016 The Kubernetes Authors.

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

package core

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/generic"
)

// legacyObjectCountAliases are what we used to do simple object counting quota with mapped to alias
var legacyObjectCountAliases = map[schema.GroupVersionResource]api.ResourceName{
	v1.SchemeGroupVersion.WithResource("configmaps"):             api.ResourceConfigMaps,
	v1.SchemeGroupVersion.WithResource("resourcequotas"):         api.ResourceQuotas,
	v1.SchemeGroupVersion.WithResource("replicationcontrollers"): api.ResourceReplicationControllers,
	v1.SchemeGroupVersion.WithResource("secrets"):                api.ResourceSecrets,
}

// NewEvaluators returns the list of static evaluators that manage more than counts
func NewEvaluators(f quota.ListerForResourceFunc) []quota.Evaluator {
	// these evaluators have special logic
	result := []quota.Evaluator{
		NewPodEvaluator(f, clock.RealClock{}),
		NewServiceEvaluator(f),
		NewPersistentVolumeClaimEvaluator(f),
	}
	// these evaluators require an alias for backwards compatibility
	for gvr, alias := range legacyObjectCountAliases {
		result = append(result,
			generic.NewObjectCountEvaluator(false, gvr.GroupResource(), generic.ListResourceUsingListerFunc(f, gvr), alias))
	}
	return result
}
