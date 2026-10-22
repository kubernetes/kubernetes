/*
Copyright The Kubernetes Authors.

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

package pod

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/pod"
)

func TestPodRestoredConditionValidation(t *testing.T) {
	path := field.NewPath("status", "conditions").Index(0)
	condition := func(state api.ConditionStatus) api.PodCondition {
		return api.PodCondition{Type: api.PodRestored, Status: state, Reason: "RestoreInProgress", LastTransitionTime: metav1.Now(), ObservedGeneration: 1}
	}
	type testCase struct {
		old        *api.PodCondition
		mutate     func(*api.PodCondition)
		errors     field.ErrorList
		conditions func([]api.PodCondition) []api.PodCondition
	}
	tests := map[string]testCase{
		"start":          {},
		"completed":      {mutate: func(c *api.PodCondition) { c.Status = api.ConditionTrue }},
		"failed":         {mutate: func(c *api.PodCondition) { c.Status = api.ConditionFalse }},
		"invalid status": {mutate: func(c *api.PodCondition) { c.Status = "invalid" }, errors: field.ErrorList{field.NotSupported(path.Child("status"), nil, []string{})}},
		"empty status":   {mutate: func(c *api.PodCondition) { c.Status = "" }, errors: field.ErrorList{field.Required(path.Child("status"), "")}},
		"empty reason":   {mutate: func(c *api.PodCondition) { c.Reason = "" }, errors: field.ErrorList{field.Required(path.Child("reason"), "")}},
		"invalid reason": {mutate: func(c *api.PodCondition) { c.Reason = "not a reason" }, errors: field.ErrorList{field.Invalid(path.Child("reason"), nil, "")}},
		"long reason":    {mutate: func(c *api.PodCondition) { c.Reason = strings.Repeat("a", 1025) }, errors: field.ErrorList{field.TooLong(path.Child("reason"), "", 1024).WithOrigin("maxBytes")}},
		"long message":   {mutate: func(c *api.PodCondition) { c.Message = strings.Repeat("a", 32769) }, errors: field.ErrorList{field.TooLong(path.Child("message"), "", 32768).WithOrigin("maxBytes")}},
		"boundary lengths": {mutate: func(c *api.PodCondition) {
			c.Reason = strings.Repeat("a", 1024)
			c.Message = strings.Repeat("a", 32768)
		}},
		"missing transition time": {mutate: func(c *api.PodCondition) { c.LastTransitionTime = metav1.Time{} }, errors: field.ErrorList{field.Required(path.Child("lastTransitionTime"), "")}},
		"negative generation":     {mutate: func(c *api.PodCondition) { c.ObservedGeneration = -1 }, errors: field.ErrorList{field.Invalid(path.Child("observedGeneration"), nil, "")}},
	}
	for _, terminal := range []api.ConditionStatus{api.ConditionTrue, api.ConditionFalse} {
		old := condition(terminal)
		tests[string(terminal)+" remains terminal"] = testCase{old: &old}
		for _, next := range []api.ConditionStatus{api.ConditionUnknown, api.ConditionTrue, api.ConditionFalse} {
			if next == terminal {
				continue
			}
			tests[string(terminal)+" to "+string(next)] = testCase{old: &old, mutate: func(c *api.PodCondition) { c.Status = next }, errors: field.ErrorList{field.Invalid(path.Child("status"), nil, "")}}
		}
	}
	old := condition(api.ConditionUnknown)
	tests["generation cannot advance"] = testCase{old: &old, mutate: func(c *api.PodCondition) { c.ObservedGeneration++ }, errors: field.ErrorList{field.Invalid(path.Child("observedGeneration"), nil, "")}}
	tests["cannot remove recorded condition"] = testCase{
		old:        &old,
		conditions: func([]api.PodCondition) []api.PodCondition { return nil },
		errors:     field.ErrorList{field.Forbidden(field.NewPath("status", "conditions"), "")},
	}
	tests["duplicate restore condition"] = testCase{
		conditions: func(c []api.PodCondition) []api.PodCondition { return append(c, c[0]) },
		errors:     field.ErrorList{field.Duplicate(field.NewPath("status", "conditions").Index(1).Child("type"), nil)},
	}
	tests["condition order may change"] = testCase{
		old: &old,
		conditions: func(c []api.PodCondition) []api.PodCondition {
			return append([]api.PodCondition{{Type: api.PodReady, Status: api.ConditionFalse}}, c...)
		},
	}
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{APIGroup: "", APIVersion: apiVersion, Resource: "pods", Subresource: "status", Verb: "update", IsResourceRequest: true})
		for name, tc := range tests {
			t.Run(apiVersion+"/"+name, func(t *testing.T) {
				oldPod := podtest.MakePod("pod", podtest.SetRestoreFrom("checkpoint"), podtest.SetResourceVersion("1"))
				c := condition(api.ConditionUnknown)
				if tc.old != nil {
					oldPod.Status.Conditions = []api.PodCondition{*tc.old}
					c = *tc.old
				}
				newPod := oldPod.DeepCopy()
				if tc.mutate != nil {
					tc.mutate(&c)
				}
				newPod.Status.Conditions = []api.PodCondition{c}
				if tc.conditions != nil {
					newPod.Status.Conditions = tc.conditions(newPod.Status.Conditions)
				}
				for _, err := range tc.errors {
					err.MarkFromImperative()
				}
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, newPod, oldPod, registry.StatusStrategy, tc.errors, apitesting.WithSubResources("status"))
			})
		}
	}
}
