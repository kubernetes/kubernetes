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

package workload

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"

	// Ensure all API groups are registered with the scheme
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

var allowedDisruptionModes = sets.New(
	scheduling.DisruptionModePod,
	scheduling.DisruptionModePodGroup,
)

func TestDeclarativeValidate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // Workload is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "workloads",
		IsResourceRequest: true,
		Verb:              "create",
	})

	testCases := map[string]struct {
		input                           scheduling.Workload
		enableTopologyAwareScheduling   bool
		enableDRAWorkloadResourceClaims bool
		enableWorkloadAwarePreemption   bool
		expectedErrs                    field.ErrorList
	}{
		"valid": {
			input: mkValidWorkload(),
		},
		"empty podGroupTemplates": {
			input:        mkValidWorkload(clearPodGroupTemplates()),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates"), "must have at least one item")},
		},
		"too many podGroupTemplates": {
			input:        mkValidWorkload(setManyPodGroupTemplates(scheduling.WorkloadMaxPodGroupTemplates + 1)),
			expectedErrs: field.ErrorList{field.TooMany(field.NewPath("spec", "podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems")},
		},
		"empty podGroupTemplate name": {
			input:        mkValidWorkload(setPodGroupName(0, "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("name"), "")},
		},
		"invalid podGroupTemplate name": {
			input:        mkValidWorkload(setPodGroupName(0, "Invalid_Name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"duplicate podGroupTemplate names": {
			input:        mkValidWorkload(addPodGroupTemplate("main")),
			expectedErrs: field.ErrorList{field.Duplicate(field.NewPath("spec", "podGroupTemplates").Index(1), nil)},
		},
		// Declarative validation treats 0 as "missing" and returns Required error
		// instead of checking minimum constraint and returning Invalid error.
		"gang minCount zero": {
			input:        mkValidWorkload(setPodGroupMinCount(0, 0)),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), "")},
		},
		"gang minCount negative": {
			input:        mkValidWorkload(setPodGroupMinCount(0, -1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy", "gang", "minCount"), nil, "").WithOrigin("minimum")},
		},
		"valid with controllerRef": {
			input: mkValidWorkload(setControllerRef("apps", "Deployment", "my-deployment")),
		},
		"controllerRef with empty APIGroup": {
			input: mkValidWorkload(setControllerRef("", "Pod", "my-pod")),
		},
		"controllerRef invalid APIGroup": {
			input:        mkValidWorkload(setControllerRef("invalid_api_group", "Deployment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"controllerRef too long APIGroup": {
			input:        mkValidWorkload(setControllerRef(strings.Repeat("g", 254), "Deployment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "apiGroup"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"controllerRef missing kind": {
			input:        mkValidWorkload(setControllerRef("apps", "", "my-deployment")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "controllerRef", "kind"), "")},
		},
		"controllerRef invalid kind with slash": {
			input:        mkValidWorkload(setControllerRef("apps", "Deploy/ment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "kind"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"controllerRef missing name": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "controllerRef", "name"), "")},
		},
		"controllerRef invalid name": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "/invalid-name")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "name"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"controllerRef invalid kind with percent": {
			input:        mkValidWorkload(setControllerRef("apps", "Deploy%ment", "my-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "kind"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"controllerRef invalid name with percent": {
			input:        mkValidWorkload(setControllerRef("apps", "Deployment", "my%deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef", "name"), nil, "").WithOrigin("format=k8s-path-segment-name")},
		},
		"policy with neither basic nor gang": {
			input:        mkValidWorkload(clearPodGroupPolicy(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"policy with both basic and gang": {
			input:        mkValidWorkload(setBothPolicies(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"valid with basic policy": {
			input: mkValidWorkload(setBasicPolicy(0)),
		},
		"valid with schedulingConstraints": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "foo")),
			enableTopologyAwareScheduling: true,
		},
		"valid with empty schedulingConstraints": {
			input:                         mkValidWorkload(setSchedulingConstraints(0)),
			enableTopologyAwareScheduling: true,
		},
		"with multiple topology constraints": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "foo"), addTopologyConstraint(0, "bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.TooMany(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology"), 2, 1).WithOrigin("maxItems")},
		},
		"with empty topology key": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), "")},
		},
		"valid with topology key with DNS prefix": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "example.com/Foo")),
			enableTopologyAwareScheduling: true,
		},
		"valid with topology key with prefix with max length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, strings.Repeat("a", 253)+"/"+strings.Repeat("b", 63))),
			enableTopologyAwareScheduling: true,
		},
		"with topology key with prefix exceending max prefix length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, strings.Repeat("a", 254)+"/foo")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key with prefix exceending max name length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "foo/"+strings.Repeat("b", 64))),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key without prefix exceeding max length": {
			input:                         mkValidWorkload(addTopologyConstraint(0, strings.Repeat("b", 64))),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key with invalid characters": {
			input:                         mkValidWorkload(addTopologyConstraint(0, "Example.com/Foo")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"pod disruption mode, workload aware preemption enabled": {
			input:                         mkValidWorkload(setDisruptionMode(0, podDisruptionMode)),
			enableWorkloadAwarePreemption: true,
		},
		"pod disruption mode, workload aware preemption disabled": {
			input:        mkValidWorkload(setDisruptionMode(0, podDisruptionMode)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("disruptionMode"), "")},
		},
		"pod group disruption mode, workload aware preemption enabled": {
			input:                         mkValidWorkload(setDisruptionMode(0, podGroupDisruptionMode)),
			enableWorkloadAwarePreemption: true,
		},
		"pod group disruption mode, workload aware preemption disabled": {
			input:        mkValidWorkload(setDisruptionMode(0, podGroupDisruptionMode)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("disruptionMode"), "")},
		},
		"invalid disruption mode, workload aware preemption enabled": {
			input:                         mkValidWorkload(setDisruptionMode(0, invalidDisruptionMode)),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.NotSupported(field.NewPath("spec", "podGroupTemplates").Index(0).Child("disruptionMode"), invalidDisruptionMode, sets.List(allowedDisruptionModes))},
		},
		"invalid disruption mode, workload aware preemption disabled": {
			input:        mkValidWorkload(setDisruptionMode(0, invalidDisruptionMode)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("disruptionMode"), "")},
		},
		"valid priorityClassName, workload aware preemption enabled": {
			input:                         mkValidWorkload(setPriorityClassName(0, "high-priority")),
			enableWorkloadAwarePreemption: true,
		},
		"valid priorityClassName, workload aware preemption disabled": {
			input:        mkValidWorkload(setPriorityClassName(0, "high-priority")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priorityClassName"), "")},
		},
		"invalid priorityClassName, workload aware preemption enabled": {
			input:                         mkValidWorkload(setPriorityClassName(0, "high/priority")),
			enableWorkloadAwarePreemption: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priorityClassName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"invalid priorityClassName, workload aware preemption disabled": {
			input:        mkValidWorkload(setPriorityClassName(0, "high/priority")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priorityClassName"), "")},
		},

		"valid priority, workload aware preemption enabled": {
			input:                         mkValidWorkload(setPriority(0, 1000)),
			enableWorkloadAwarePreemption: true,
		},
		"valid priority, workload aware preemption disabled": {
			input:        mkValidWorkload(setPriority(0, 1000)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priority"), "")},
		},
		"valid negative priority, workload aware preemption enabled": {
			input:                         mkValidWorkload(setPriority(0, -2147483648)),
			enableWorkloadAwarePreemption: true,
		},
		"valid negative priority, workload aware preemption disabled": {
			input:        mkValidWorkload(setPriority(0, -2147483648)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priority"), "")},
		},
		"too high priority, workload aware preemption enabled": {
			input:                         mkValidWorkload(setPriority(0, scheduling.HighestUserDefinablePriority+1)),
			enableWorkloadAwarePreemption: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priority"), nil, "").WithOrigin("maximum"),
			},
		},
		"too high priority, workload aware preemption disabled": {
			input:        mkValidWorkload(setPriority(0, scheduling.HighestUserDefinablePriority+1)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "podGroupTemplates").Index(0).Child("priority"), "")},
		},
		"ok resourceClaimName reference": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimName: new("resource-claim")})),
		},
		"ok resourceClaimTemplateName reference": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimTemplateName: new("resource-claim-template")})),
		},
		"ok multiple claims": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "claim-1", ResourceClaimName: new("resource-claim-1")},
				scheduling.PodGroupResourceClaim{Name: "claim-2", ResourceClaimName: new("resource-claim-2")},
			)),
		},
		"claim name with prefix": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "../my-claim", ResourceClaimName: new("resource-claim")})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"claim name with path": {
			input: mkValidWorkload(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "my/claim", ResourceClaimName: new("resource-claim")})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"duplicate claim entries": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new("resource-claim-1")},
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new("resource-claim-2")},
			)),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(1), nil),
			},
		},
		"resource claim source empty": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim"},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"resource claim reference and template": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{
					Name:                      "my-claim",
					ResourceClaimName:         new("resource-claim"),
					ResourceClaimTemplateName: new("resource-claim-template"),
				},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"invalid claim reference name": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new(".foo_bar")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"invalid claim template name": {
			input: mkValidWorkload(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new(".foo_bar")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"too many claims": {
			input: mkValidWorkload(func(pg *scheduling.Workload) {
				for i := range scheduling.MaxPodGroupResourceClaims + 1 {
					pg.Spec.PodGroupTemplates[0].ResourceClaims = append(pg.Spec.PodGroupTemplates[0].ResourceClaims, scheduling.PodGroupResourceClaim{
						Name:              "my-claim-" + strconv.Itoa(i),
						ResourceClaimName: new("resource-claim"),
					})
				}
			}),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroupTemplates").Index(0).Child("resourceClaims"), scheduling.MaxPodGroupResourceClaims+1, scheduling.MaxPodGroupResourceClaims).WithOrigin("maxItems"),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: tc.enableTopologyAwareScheduling,
				features.DRAWorkloadResourceClaims:       tc.enableDRAWorkloadResourceClaims,
				features.GangScheduling:                  tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption:         tc.enableWorkloadAwarePreemption,
			})
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs, apitesting.WithMinEmulationVersion(version.MustParse("1.36")))
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // Workload is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	testCases := map[string]struct {
		oldObj                          scheduling.Workload
		updateObj                       scheduling.Workload
		enableTopologyAwareScheduling   bool
		enableDRAWorkloadResourceClaims bool
		enableWorkloadAwarePreemption   bool
		expectedErrs                    field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1")),
		},
		"valid update with unchanged controllerRef": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
		},
		"set controllerRef": {
			oldObj:       mkValidWorkload(setResourceVersion("1")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "different-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef"), nil, "").WithOrigin("immutable")},
		},
		"invalid update controllerRef": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "my-deployment")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "different-deployment")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef"), nil, "").WithOrigin("immutable")},
		},
		"unset controllerRef": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setControllerRef("apps", "Deployment", "different-deployment")),
			updateObj:    mkValidWorkload(setResourceVersion("1")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "controllerRef"), nil, "").WithOrigin("immutable")},
		},
		"invalid update empty podGroupTemplates": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setEmptyPodGroupTemplates()),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "podGroupTemplates"), "must have at least one item"),
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable"),
			},
		},
		"change podGroupTemplate name": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker2")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update too many podGroupTemplates": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setManyPodGroupTemplates(scheduling.WorkloadMaxPodGroupTemplates+1)),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "podGroupTemplates"), scheduling.WorkloadMaxPodGroupTemplates+1, scheduling.WorkloadMaxPodGroupTemplates).WithOrigin("maxItems"),
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable"),
			},
		},
		"add podGroupTemplate": {
			oldObj:       mkValidWorkload(setResourceVersion("1")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker1")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"remove podGroupTemplate": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), addPodGroupTemplate("worker1")),
			updateObj:    mkValidWorkload(setResourceVersion("1")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update with neither basic nor gang": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), clearPodGroupPolicy(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update with both basic and gang": {
			oldObj:    mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), setBothPolicies(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update of gang minCount": {
			oldObj:       mkValidWorkload(setResourceVersion("1")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setPodGroupMinCount(0, 10)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"valid update from gang to basic policy": {
			oldObj:       mkValidWorkload(setResourceVersion("1")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setBasicPolicy(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"valid update with unchanged scheduling constraints": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			enableTopologyAwareScheduling: true,
		},
		"invalid update to scheduling constraints": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setSchedulingConstraints(0)),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid update to topology constraints": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo"), addTopologyConstraint(0, "bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid update to topology key": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"valid update with unchanged scheduling constraints with TAS disabled": {
			oldObj:    mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
		},
		"invalid update to scheduling constraints with TAS disabled": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setSchedulingConstraints(0)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid update to topology constraints with TAS disabled": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo"), addTopologyConstraint(0, "bar")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid update to topology key with TAS disabled": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "foo")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), addTopologyConstraint(0, "bar")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid add of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid add of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidWorkload(setResourceVersion("1")),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid remove of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj:    mkValidWorkload(setResourceVersion("1")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid remove of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidWorkload(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj:                       mkValidWorkload(setResourceVersion("1")),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of disruption mode, workload aware preemption enabled": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setDisruptionMode(0, podDisruptionMode)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setDisruptionMode(0, podGroupDisruptionMode)),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of disruption mode, workload aware preemption disabled": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setDisruptionMode(0, podDisruptionMode)),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setDisruptionMode(0, podGroupDisruptionMode)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of priority class name, workload aware preemption enabled": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setPriorityClassName(0, "low-priority")),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setPriorityClassName(0, "high-priority")),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of priority class name, workload aware preemption disabled": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setPriorityClassName(0, "low-priority")),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setPriorityClassName(0, "high-priority")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of priority, workload aware preemption enabled": {
			oldObj:                        mkValidWorkload(setResourceVersion("1"), setPriority(0, 1000)),
			updateObj:                     mkValidWorkload(setResourceVersion("1"), setPriority(0, 2000)),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of priority, workload aware preemption disabled": {
			oldObj:       mkValidWorkload(setResourceVersion("1"), setPriority(0, 1000)),
			updateObj:    mkValidWorkload(setResourceVersion("1"), setPriority(0, 2000)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplates"), nil, "").WithOrigin("immutable")},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: tc.enableTopologyAwareScheduling,
				features.DRAWorkloadResourceClaims:       tc.enableDRAWorkloadResourceClaims,
				features.GangScheduling:                  tc.enableWorkloadAwarePreemption,
				features.WorkloadAwarePreemption:         tc.enableWorkloadAwarePreemption,
			})
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIPrefix:         "apis",
				APIGroup:          "scheduling.k8s.io",
				APIVersion:        apiVersion,
				Resource:          "workloads",
				Name:              "valid-workload",
				IsResourceRequest: true,
				Verb:              "update",
			})
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, Strategy.ValidateUpdate, tc.expectedErrs, apitesting.WithMinEmulationVersion(version.MustParse("1.36")))
		})
	}
}

func mkValidWorkload(tweaks ...func(obj *scheduling.Workload)) scheduling.Workload {
	obj := scheduling.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-workload",
			Namespace: "default",
		},
		Spec: scheduling.WorkloadSpec{
			PodGroupTemplates: []scheduling.PodGroupTemplate{
				{
					Name: "main",
					SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
						Gang: &scheduling.GangSchedulingPolicy{
							MinCount: 1,
						},
					},
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func setResourceVersion(v string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.ResourceVersion = v
	}
}

func clearPodGroupTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = nil
	}
}

func setEmptyPodGroupTemplates() func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = []scheduling.PodGroupTemplate{}
	}
}

func setManyPodGroupTemplates(n int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = make([]scheduling.PodGroupTemplate, n)
		for i := range obj.Spec.PodGroupTemplates {
			obj.Spec.PodGroupTemplates[i] = scheduling.PodGroupTemplate{
				Name: fmt.Sprintf("group-%d", i),
				SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
					Gang: &scheduling.GangSchedulingPolicy{
						MinCount: 1,
					},
				},
			}
		}
	}
}

func addPodGroupTemplate(name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates = append(obj.Spec.PodGroupTemplates, scheduling.PodGroupTemplate{
			Name: name,
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 1,
				},
			},
		})
	}
}

func setPodGroupName(pgIdx int, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].Name = name
	}
}

func setPodGroupMinCount(pgIdx, min int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy.Gang.MinCount = int32(min)
	}
}

func clearPodGroupPolicy(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
	}
}

func setBasicPolicy(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
		}
	}
}

func setBothPolicies(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
			Gang:  &scheduling.GangSchedulingPolicy{MinCount: 1},
		}
	}
}

func setControllerRef(apiGroup, kind, name string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.ControllerRef = &scheduling.TypedLocalObjectReference{
			APIGroup: apiGroup,
			Kind:     kind,
			Name:     name,
		}
	}
}

func setSchedulingConstraints(pgIdx int) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{}
	}
}

func addTopologyConstraint(pgIdx int, topologyKey string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		if obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints == nil {
			setSchedulingConstraints(pgIdx)(obj)
		}
		obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints.Topology = append(obj.Spec.PodGroupTemplates[pgIdx].SchedulingConstraints.Topology,
			scheduling.TopologyConstraint{Key: topologyKey})
	}
}

func addResourceClaims(claims ...scheduling.PodGroupResourceClaim) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[0].ResourceClaims = append(obj.Spec.PodGroupTemplates[0].ResourceClaims, claims...)
	}
}

func setDisruptionMode(pgIdx int, mode scheduling.DisruptionMode) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].DisruptionMode = &mode
	}
}

func setPriorityClassName(pgIdx int, priorityClassName string) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].PriorityClassName = priorityClassName
	}
}

func setPriority(pgIdx int, priority int32) func(obj *scheduling.Workload) {
	return func(obj *scheduling.Workload) {
		obj.Spec.PodGroupTemplates[pgIdx].Priority = new(priority)
	}
}
