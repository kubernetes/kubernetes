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

package pod

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/pod"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			IsResourceRequest: true,
			Verb:              "create",
		})
		testCases := map[string]struct {
			input        *api.Pod
			expectedErrs field.ErrorList
		}{
			"valid": {
				input: podtest.MakePod("foo"),
			},
			"valid toleration key": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Key: "example.com/valid-key", Operator: api.TolerationOpExists},
				)),
			},
			"valid toleration key without prefix": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Key: "simple-key", Operator: api.TolerationOpExists},
				)),
			},
			"empty toleration key (match all, skipped)": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Operator: api.TolerationOpExists},
				)),
			},
			"invalid toleration key format": {
				input: podtest.MakePod("foo", podtest.SetTolerations(
					api.Toleration{Key: "invalid key", Operator: api.TolerationOpExists},
				)),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkAlpha(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				apitesting.VerifyValidationEquivalence(t, ctx, tc.input, registry.Strategy, tc.expectedErrs)
			})
		}
		obj := *podtest.MakePod("foo")
		meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
		obj2 := podtest.MakePod("foo")
		RunDeclarativeValidateEvictionRespondersTestCases(t, ctx, registry.Strategy, field.NewPath("spec"), obj2, func(baseObj *api.Pod, responders []api.EvictionResponder, schedulingGroup *api.PodSchedulingGroup) {
			baseObj.Spec.EvictionResponders = responders
			baseObj.Spec.SchedulingGroup = schedulingGroup
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			Name:              "valid-obj",
			IsResourceRequest: true,
			Verb:              "update",
		})
		updateObj := *podtest.MakePod("foo")
		meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
			APIPrefix:         "api",
			APIGroup:          "",
			APIVersion:        apiVersion,
			Resource:          "pods",
			Subresource:       "status",
			Name:              "foo",
			IsResourceRequest: true,
			Verb:              "update",
		})
		conditionPath := field.NewPath("status", "volumeHealth").Index(0).Child("healthConditions")
		tests := map[string]struct {
			volumeHealth []api.PodVolumeHealth
			expectedErrs field.ErrorList
		}{
			"valid": {
				volumeHealth: []api.PodVolumeHealth{{
					Name:             "vol",
					HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded, Reason: "DiskSlow"}},
				}},
			},
			"name required": {
				volumeHealth: []api.PodVolumeHealth{{}},
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "volumeHealth").Index(0).Child("name"), ""),
				},
			},
			"duplicate volumes": {
				volumeHealth: []api.PodVolumeHealth{{Name: "vol"}, {Name: "vol"}},
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "volumeHealth").Index(1), nil),
				},
			},
			"status required": {
				volumeHealth: []api.PodVolumeHealth{{Name: "vol", HealthConditions: []api.VolumeHealthCondition{{Reason: "DiskSlow"}}}},
				expectedErrs: field.ErrorList{
					field.Required(conditionPath.Index(0).Child("status"), ""),
				},
			},
			"status enum": {
				volumeHealth: []api.PodVolumeHealth{{Name: "vol", HealthConditions: []api.VolumeHealthCondition{{Status: "Invalid", Reason: "DiskSlow"}}}},
				expectedErrs: field.ErrorList{
					field.NotSupported(conditionPath.Index(0).Child("status"), api.VolumeHealthStatusType("Invalid"), []api.VolumeHealthStatusType(nil)),
				},
			},
			"reason required": {
				volumeHealth: []api.PodVolumeHealth{{Name: "vol", HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded}}}},
				expectedErrs: field.ErrorList{
					field.Required(conditionPath.Index(0).Child("reason"), ""),
				},
			},
			"reason max bytes": {
				volumeHealth: []api.PodVolumeHealth{{Name: "vol", HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded, Reason: strings.Repeat("a", 257)}}}},
				expectedErrs: field.ErrorList{
					field.TooLong(conditionPath.Index(0).Child("reason"), "", 256).WithOrigin("maxBytes"),
				},
			},
			"message max bytes": {
				volumeHealth: []api.PodVolumeHealth{{Name: "vol", HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded, Reason: "DiskSlow", Message: strings.Repeat("𝄞", 257)}}}},
				expectedErrs: field.ErrorList{
					field.TooLong(conditionPath.Index(0).Child("message"), "", 1024).WithOrigin("maxBytes"),
				},
			},
			"duplicate conditions": {
				volumeHealth: []api.PodVolumeHealth{{
					Name: "vol",
					HealthConditions: []api.VolumeHealthCondition{
						{Status: api.VolumeHealthDegraded, Reason: "DiskSlow"},
						{Status: api.VolumeHealthDegraded, Reason: "DiskSlow"},
					},
				}},
				expectedErrs: field.ErrorList{
					field.Duplicate(conditionPath.Index(1), nil),
				},
			},
			"too many conditions": {
				volumeHealth: []api.PodVolumeHealth{{Name: "vol", HealthConditions: makePodVolumeHealthConditions(17)}},
				expectedErrs: field.ErrorList{
					field.TooMany(conditionPath, 17, 16).WithOrigin("maxItems"),
				},
			},
		}

		for name, tc := range tests {
			t.Run(apiVersion+"/"+name, func(t *testing.T) {
				oldObj := podtest.MakePod("foo", podtest.SetVolumes(api.Volume{Name: "vol"}))
				oldObj.ResourceVersion = "1"
				updateObj := oldObj.DeepCopy()
				updateObj.Status.VolumeHealth = tc.volumeHealth
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, updateObj, oldObj, registry.StatusStrategy, tc.expectedErrs, apitesting.WithSubResources("status"))
			})
		}
	}
}

func makePodVolumeHealthConditions(count int) []api.VolumeHealthCondition {
	conditions := make([]api.VolumeHealthCondition, count)
	for i := range conditions {
		conditions[i] = api.VolumeHealthCondition{
			Status: api.VolumeHealthDegraded,
			Reason: "Reason" + string(rune('A'+i)),
		}
	}
	return conditions
}
