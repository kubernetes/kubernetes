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

	apiresource "k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	rest "k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	features "k8s.io/kubernetes/pkg/features"
	registry "k8s.io/kubernetes/pkg/registry/core/pod"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRANodeAllocatableResources: true,
	})
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

func TestDeclarativeValidateNodeAllocatableStatus(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRANodeAllocatableResources: true,
	})
	for _, apiVersion := range apiVersions {
		testCases := map[string]struct {
			old          *api.Pod
			update       *api.Pod
			subresource  string
			expectedErrs field.ErrorList
		}{
			"valid no changes": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "cpu", Quantity: new(apiresource.MustParse("1"))},
						},
					},
				}),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "cpu", Quantity: new(apiresource.MustParse("1"))},
						},
					},
				}),
				subresource: "/status",
			},

			"invalid: status overhead both perPod and perContainer nil": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Overhead: []api.NodeAllocatableOverheadResources{
							{Name: "cpu"},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("overhead").Index(0), "", "at least one of perPod or perContainer must be set").MarkFromImperative(),
				},
			},
			"invalid: status overhead perPod negative": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Overhead: []api.NodeAllocatableOverheadResources{
							{Name: "cpu", PerPod: new(apiresource.MustParse("-100m"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("overhead").Index(0).Child("perPod"), "-100m", "must be non-negative").MarkFromImperative(),
				},
			},
			"invalid: status mapping duplicate resource name": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "cpu", Quantity: new(apiresource.MustParse("1"))},
							{Name: "cpu", Quantity: new(apiresource.MustParse("2"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("mapping").Index(1), "cpu"),
				},
			},
			"invalid status: duplicate claim status name": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "cpu", Quantity: new(apiresource.MustParse("1"))},
						},
					},
					{
						ResourceClaimName: "claim-1",
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "memory", Quantity: new(apiresource.MustParse("1G"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(1), "claim-1"),
				},
			},
			"invalid status: duplicate container name": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Containers:        []string{"c1", "c1"},
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "cpu", Quantity: new(apiresource.MustParse("1"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("containers").Index(1), "c1"),
				},
			},
			"invalid status: mapping name required": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Mapping: []api.NodeAllocatableMappedResources{
							{Quantity: new(apiresource.MustParse("1"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("mapping").Index(0).Child("name"), ""),
				},
			},
			"invalid status: mapping quantity required": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "cpu"},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("mapping").Index(0).Child("quantity"), ""),
				},
			},
			"invalid status: duplicate overhead resource name": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Overhead: []api.NodeAllocatableOverheadResources{
							{Name: "cpu", PerPod: new(apiresource.MustParse("100m"))},
							{Name: "cpu", PerPod: new(apiresource.MustParse("200m"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("overhead").Index(1), "cpu"),
				},
			},
			"invalid status: overhead name required": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("claim-1", []api.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "claim-1",
						Overhead: []api.NodeAllocatableOverheadResources{
							{PerPod: new(apiresource.MustParse("100m"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("overhead").Index(0).Child("name"), ""),
				},
			},
			"invalid status: resourceClaimName required": {
				old: makePodWithNodeAllocatableResourceClaimStatuses("", nil),
				update: makePodWithNodeAllocatableResourceClaimStatuses("", []api.NodeAllocatableResourceClaimStatus{
					{
						Mapping: []api.NodeAllocatableMappedResources{
							{Name: "cpu", Quantity: new(apiresource.MustParse("1"))},
						},
					},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "nodeAllocatableResourceClaimStatuses").Index(0).Child("resourceClaimName"), ""),
				},
			},
		}

		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
					APIPrefix:         "api",
					APIGroup:          "",
					APIVersion:        apiVersion,
					Name:              "valid-obj",
					IsResourceRequest: true,
					Subresource:       tc.subresource[1:],
					Verb:              "update",
				})
				var strategy rest.RESTUpdateStrategy
				switch tc.subresource {
				case "/":
					strategy = registry.Strategy
				case "/status":
					strategy = registry.StatusStrategy
				}
				tc.old.ResourceVersion = "1"
				tc.update.ResourceVersion = "1"
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, tc.update, tc.old, strategy, tc.expectedErrs)
			})
		}

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

func makePodWithNodeAllocatableResourceClaimStatuses(claimName string, statuses []api.NodeAllocatableResourceClaimStatus) *api.Pod {
	pod := podtest.MakePod("foo")
	if claimName != "" {
		pod.Spec.ResourceClaims = []api.PodResourceClaim{
			{
				Name:              "claim-ref",
				ResourceClaimName: new(claimName),
			},
		}
	}
	pod.Status.NodeAllocatableResourceClaimStatuses = statuses
	return pod
}
