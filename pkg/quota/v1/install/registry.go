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

package install

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	quota "k8s.io/kubernetes/pkg/quota/v1"
	core "k8s.io/kubernetes/pkg/quota/v1/evaluator/core"
	generic "k8s.io/kubernetes/pkg/quota/v1/generic"
)

// NewQuotaConfigurationForAdmission returns a quota configuration for admission control.
func NewQuotaConfigurationForAdmission() quota.Configuration {
	evaluators := core.NewEvaluators(nil)
	return generic.NewConfiguration(evaluators, DefaultIgnoredResources())
}

// NewQuotaConfigurationForControllers returns a quota configuration for controllers.
func NewQuotaConfigurationForControllers(f quota.ListerForResourceFunc) quota.Configuration {
	evaluators := core.NewEvaluators(f)
	return generic.NewConfiguration(evaluators, DefaultIgnoredResources())
}

// ignoredResources are ignored by quota by default
var ignoredResources = map[schema.GroupResource]struct{}{
	// virtual resources that aren't stored and shouldn't be quota-ed
	{Group: "", Resource: "bindings"}:                                      {},
	{Group: "", Resource: "componentstatuses"}:                             {},
	{Group: "authentication.k8s.io", Resource: "tokenreviews"}:             {},
	{Group: "authorization.k8s.io", Resource: "subjectaccessreviews"}:      {},
	{Group: "authorization.k8s.io", Resource: "selfsubjectaccessreviews"}:  {},
	{Group: "authorization.k8s.io", Resource: "localsubjectaccessreviews"}: {},
	{Group: "authorization.k8s.io", Resource: "selfsubjectrulesreviews"}:   {},

	// events haven't been quota-ed before
	{Group: "", Resource: "events"}: {},
}

// DefaultIgnoredResources returns the default set of resources that quota system
// should ignore. This is exposed so downstream integrators can have access to them.
func DefaultIgnoredResources() map[schema.GroupResource]struct{} {
	return ignoredResources
}
