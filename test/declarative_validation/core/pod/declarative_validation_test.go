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
		type testCase struct {
			input        *api.Pod
			expectedErrs field.ErrorList
		}
		testCases := map[string]testCase{
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
					field.Invalid(field.NewPath("spec", "tolerations").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key").MarkBeta(),
				},
			},
			"activeDeadlineSeconds valid": {
				input: func() *api.Pod {
					p := podtest.MakePod("foo")
					deadline := int64(30)
					p.Spec.ActiveDeadlineSeconds = &deadline
					return p
				}(),
			},
			"activeDeadlineSeconds minimum boundary violation": {
				input: func() *api.Pod {
					p := podtest.MakePod("foo")
					deadline := int64(0)
					p.Spec.ActiveDeadlineSeconds = &deadline
					return p
				}(),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "activeDeadlineSeconds"), int64(0), "").WithOrigin("minimum").MarkAlpha(),
				},
			},
			"activeDeadlineSeconds maximum boundary violation": {
				input: func() *api.Pod {
					p := podtest.MakePod("foo")
					deadline := int64(2147483648)
					p.Spec.ActiveDeadlineSeconds = &deadline
					return p
				}(),
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("spec", "activeDeadlineSeconds"), int64(2147483648), "").WithOrigin("maximum").MarkAlpha(),
				},
			},
		}

		for _, tc := range InvalidPodSpecTestcases(field.NewPath("spec")) {
			testCases["podspec:"+tc.ID] = testCase{
				input:        podtest.MakePod("foo", tc.Tweaks...),
				expectedErrs: tc.ExpectedErrs,
			}
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
			Resource:          "pods",
			Name:              "foo",
			IsResourceRequest: true,
			Verb:              "update",
		})
		deadlinePath := field.NewPath("spec", "activeDeadlineSeconds")
		testCases := map[string]struct {
			oldDeadline  int64
			newDeadline  int64
			expectedErrs field.ErrorList
		}{
			"activeDeadlineSeconds decreased": {
				oldDeadline: 5,
				newDeadline: 3,
			},
			"activeDeadlineSeconds minimum boundary violation": {
				oldDeadline: 5,
				newDeadline: 0,
				expectedErrs: field.ErrorList{
					field.Invalid(deadlinePath, int64(0), "").WithOrigin("minimum").MarkAlpha(),
				},
			},
			"activeDeadlineSeconds maximum boundary violation": {
				oldDeadline: 5,
				newDeadline: 2147483648,
				expectedErrs: field.ErrorList{
					field.Invalid(deadlinePath, int64(2147483648), "").WithOrigin("maximum").MarkAlpha(),
					// ValidatePodUpdate range checks this field itself, and that
					// check is not covered by declarative validation.
					field.Invalid(deadlinePath, int64(2147483648), "").MarkFromImperative(),
				},
			},
		}
		for k, tc := range testCases {
			t.Run(k, func(t *testing.T) {
				oldObj := podtest.MakePod("foo")
				oldObj.ResourceVersion = "1"
				oldDeadline := tc.oldDeadline
				oldObj.Spec.ActiveDeadlineSeconds = &oldDeadline
				updateObj := oldObj.DeepCopy()
				newDeadline := tc.newDeadline
				updateObj.Spec.ActiveDeadlineSeconds = &newDeadline
				apitesting.VerifyUpdateValidationEquivalence(t, ctx, updateObj, oldObj, registry.Strategy, tc.expectedErrs)
			})
		}
	}
}

func TestDeclarativeValidateExtraNodeAllocatableResources(t *testing.T) {
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
				old: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
				}),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
				}),
				subresource: "/status",
			},
			"valid status: pod level and container entries": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "8", "memory", "2Gi"), Limits: resources("cpu", "8", "memory", "2Gi")},
					{ContainerName: "ctr", Requests: resources("memory", "512Mi"), Limits: resources("cpu", "8", "memory", "1536Mi")},
				}),
				subresource: "/status",
			},
			"valid status: ephemeral-storage and hugepages": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("ephemeral-storage", "1Gi", "hugepages-2Mi", "2Mi"), Limits: resources("ephemeral-storage", "1Gi", "hugepages-2Mi", "2Mi")},
				}),
				subresource: "/status",
			},
			"invalid status: duplicate containerName": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "2"), Limits: resources("cpu", "2")},
					{ContainerName: "ctr", Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
					{ContainerName: "ctr", Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "extraNodeAllocatableResources").Index(2), nil),
				},
			},
			"invalid status: two pod level entries": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
					{Requests: resources("cpu", "2"), Limits: resources("cpu", "2")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "extraNodeAllocatableResources").Index(1), nil),
				},
			},
			"invalid status: container entry without a pod level entry": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{ContainerName: "ctr", Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "extraNodeAllocatableResources"), "must have an entry with an empty containerName").MarkFromImperative(),
				},
			},
			"invalid status: unknown container": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
					{ContainerName: "unknown", Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("status", "extraNodeAllocatableResources").Index(1).Child("containerName"), "unknown", "must be empty or match the name of an entry in `spec.initContainers` or `spec.containers`").MarkFromImperative(),
				},
			},
			"invalid status: resource that is not node allocatable": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("kubernetes.io/foo", "1"), Limits: resources("kubernetes.io/foo", "1")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("status", "extraNodeAllocatableResources").Index(0).Child("requests").Key("kubernetes.io/foo"), api.ResourceName("kubernetes.io/foo"), "must be a node allocatable resource name").MarkFromImperative(),
					field.Invalid(field.NewPath("status", "extraNodeAllocatableResources").Index(0).Child("limits").Key("kubernetes.io/foo"), api.ResourceName("kubernetes.io/foo"), "must be a node allocatable resource name").MarkFromImperative(),
				},
			},
			"invalid status: entries without limits": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1")},
					{ContainerName: "ctr"},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "extraNodeAllocatableResources").Index(0).Child("limits"), ""),
					field.Required(field.NewPath("status", "extraNodeAllocatableResources").Index(1).Child("limits"), ""),
				},
			},
			"invalid status: negative quantity": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1"), Limits: resources("cpu", "-1")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("status", "extraNodeAllocatableResources").Index(0).Child("limits").Key("cpu"), "-1", "must be greater than or equal to 0").MarkFromImperative(),
					field.Invalid(field.NewPath("status", "extraNodeAllocatableResources").Index(0).Child("requests").Key("cpu"), "1", "must be less than or equal to cpu limit of -1").MarkFromImperative(),
				},
			},
			"invalid status: aggregate container requests exceed the pod level requests": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1"), Limits: resources("cpu", "2")},
					{ContainerName: "ctr", Requests: resources("cpu", "2"), Limits: resources("cpu", "2")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("status", "extraNodeAllocatableResources").Index(0).Child("requests").Key("cpu"), "1", "must be greater than or equal to the aggregate requests of the container entries (2)").MarkFromImperative(),
				},
			},
			"invalid status: container limit exceeds the pod level limit": {
				old: makePodWithExtraNodeAllocatableResources(nil),
				update: makePodWithExtraNodeAllocatableResources([]api.ExtraNodeAllocatableResource{
					{Requests: resources("cpu", "1"), Limits: resources("cpu", "1")},
					{ContainerName: "ctr", Limits: resources("cpu", "2")},
				}),
				subresource: "/status",
				expectedErrs: field.ErrorList{
					field.Invalid(field.NewPath("status", "extraNodeAllocatableResources").Index(1).Child("limits").Key("cpu"), "2", "must be less than or equal to the limit of the entry with an empty containerName (1)").MarkFromImperative(),
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
		conditionPath := field.NewPath("status", "volumeStatuses").Index(0).Child("healthStatus", "healthConditions")
		tests := map[string]struct {
			volumeHealth []api.PodVolumeStatus
			expectedErrs field.ErrorList
		}{
			"empty volume status": {},
			"valid": {
				volumeHealth: []api.PodVolumeStatus{{
					Name:         "vol",
					HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded, Reason: "DiskSlow"}}},
				}},
			},
			"healthy status accepted": {
				volumeHealth: []api.PodVolumeStatus{{
					Name:         "vol",
					HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthHealthy, Reason: "VolumeHealthy"}}},
				}},
			},
			"unknown status accepted": {
				volumeHealth: []api.PodVolumeStatus{{
					Name:         "vol",
					HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthUnknown, Reason: "FutureHealth"}}},
				}},
			},
			"name required": {
				volumeHealth: []api.PodVolumeStatus{{}},
				expectedErrs: field.ErrorList{
					field.Required(field.NewPath("status", "volumeStatuses").Index(0).Child("name"), ""),
				},
			},
			"duplicate volumes": {
				volumeHealth: []api.PodVolumeStatus{{Name: "vol"}, {Name: "vol"}},
				expectedErrs: field.ErrorList{
					field.Duplicate(field.NewPath("status", "volumeStatuses").Index(1), nil),
				},
			},
			"status required": {
				volumeHealth: []api.PodVolumeStatus{{Name: "vol", HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Reason: "DiskSlow"}}}}},
				expectedErrs: field.ErrorList{
					field.Required(conditionPath.Index(0).Child("status"), ""),
				},
			},
			"status enum": {
				volumeHealth: []api.PodVolumeStatus{{Name: "vol", HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Status: "Invalid", Reason: "DiskSlow"}}}}},
				expectedErrs: field.ErrorList{
					field.NotSupported(conditionPath.Index(0).Child("status"), api.VolumeHealthStatusType("Invalid"), []api.VolumeHealthStatusType(nil)),
				},
			},
			"reason required": {
				volumeHealth: []api.PodVolumeStatus{{Name: "vol", HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded}}}}},
				expectedErrs: field.ErrorList{
					field.Required(conditionPath.Index(0).Child("reason"), ""),
				},
			},
			"reason max bytes": {
				volumeHealth: []api.PodVolumeStatus{{Name: "vol", HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded, Reason: strings.Repeat("a", 257)}}}}},
				expectedErrs: field.ErrorList{
					field.TooLong(conditionPath.Index(0).Child("reason"), "", 256).WithOrigin("maxBytes"),
				},
			},
			"message max bytes": {
				volumeHealth: []api.PodVolumeStatus{{Name: "vol", HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{{Status: api.VolumeHealthDegraded, Reason: "DiskSlow", Message: strings.Repeat("𝄞", 257)}}}}},
				expectedErrs: field.ErrorList{
					field.TooLong(conditionPath.Index(0).Child("message"), "", 1024).WithOrigin("maxBytes"),
				},
			},
			"duplicate conditions": {
				volumeHealth: []api.PodVolumeStatus{{
					Name: "vol",
					HealthStatus: &api.VolumeHealthStatus{HealthConditions: []api.VolumeHealthCondition{
						{Status: api.VolumeHealthDegraded, Reason: "DiskSlow"},
						{Status: api.VolumeHealthDegraded, Reason: "DiskSlow"},
					}},
				}},
				expectedErrs: field.ErrorList{
					field.Duplicate(conditionPath.Index(1), nil),
				},
			},
			"too many conditions": {
				volumeHealth: []api.PodVolumeStatus{{Name: "vol", HealthStatus: &api.VolumeHealthStatus{HealthConditions: makePodVolumeHealthConditions(17)}}},
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
				updateObj.Status.VolumeStatuses = tc.volumeHealth
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

func makePodWithExtraNodeAllocatableResources(entries []api.ExtraNodeAllocatableResource) *api.Pod {
	pod := podtest.MakePod("foo")
	pod.Status.ExtraNodeAllocatableResources = entries
	return pod
}

// resources returns a ResourceList from pairs of resource names and quantities.
func resources(namesAndQuantities ...string) api.ResourceList {
	list := api.ResourceList{}
	for i := 0; i < len(namesAndQuantities); i += 2 {
		list[api.ResourceName(namesAndQuantities[i])] = apiresource.MustParse(namesAndQuantities[i+1])
	}
	return list
}
