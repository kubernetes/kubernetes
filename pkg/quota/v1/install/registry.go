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
	eventv1 "k8s.io/api/events/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	"k8s.io/client-go/informers"
	"k8s.io/kubernetes/pkg/apis/authentication"
	"k8s.io/kubernetes/pkg/apis/authorization"
	"k8s.io/kubernetes/pkg/quota/v1/evaluator/core"
)

// NewQuotaConfigurationForAdmission returns a quota configuration for admission control.
func NewQuotaConfigurationForAdmission(i informers.SharedInformerFactory) quota.Configuration {
	evaluators := core.NewEvaluators(nil, i)
	return generic.NewConfiguration(evaluators, DefaultIgnoredResources())
}

// NewQuotaConfigurationForControllers returns a quota configuration for controllers.
func NewQuotaConfigurationForControllers(f quota.ListerForResourceFunc, i informers.SharedInformerFactory) quota.Configuration {
	evaluators := core.NewEvaluators(f, i)
	return generic.NewConfiguration(evaluators, DefaultIgnoredResources())
}

// ignoredResources are ignored by quota by default
var ignoredResources = map[schema.GroupResource]struct{}{
	// virtual resources that aren't stored and shouldn't be quota-ed
	{Group: "", Resource: "bindings"}:                                       {},
	{Group: "", Resource: "componentstatuses"}:                              {},
	{Group: authentication.GroupName, Resource: "tokenreviews"}:             {},
	{Group: authentication.GroupName, Resource: "selfsubjectreviews"}:       {},
	{Group: authorization.GroupName, Resource: "subjectaccessreviews"}:      {},
	{Group: authorization.GroupName, Resource: "selfsubjectaccessreviews"}:  {},
	{Group: authorization.GroupName, Resource: "localsubjectaccessreviews"}: {},
	{Group: authorization.GroupName, Resource: "selfsubjectrulesreviews"}:   {},

	// events haven't been quota-ed before
	{Group: "", Resource: "events"}:                {},
	{Group: eventv1.GroupName, Resource: "events"}: {},
}

// DefaultIgnoredResources returns the default set of resources that quota system
// should ignore. This is exposed so downstream integrators can have access to them.
func DefaultIgnoredResources() map[schema.GroupResource]struct{} {
	return ignoredResources
}
