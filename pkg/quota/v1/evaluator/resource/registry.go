/*
Copyright 2025 The Kubernetes Authors.

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

package resource

import (
	quota "k8s.io/apiserver/pkg/quota/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// NewEvaluators returns the list of static evaluators that manage more than counts
func NewEvaluators(f quota.ListerForResourceFunc) []quota.Evaluator {
	// these evaluators have special logic
	result := []quota.Evaluator{}
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
		result = append(result, NewResourceClaimEvaluator(f))
	}
	return result
}
