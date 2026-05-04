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

package podgroup

import (
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
	apiVersions := []string{"v1alpha2"} // PodGroup is currently only in v1alpha2
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
		Resource:          "podgroups",
		IsResourceRequest: true,
		Verb:              "create",
	})
	strategy := NewStrategy()

	testCases := map[string]struct {
		input                           scheduling.PodGroup
		enableTopologyAwareScheduling   bool
		enableDRAWorkloadResourceClaims bool
		enableWorkloadAwarePreemption   bool
		expectedErrs                    field.ErrorList
	}{
		"valid": {
			input: mkValidPodGroup(),
		},
		"valid with basic policy": {
			input: mkValidPodGroup(setBasicPolicy()),
		},
		// Declarative validation treats 0 as "missing" and returns Required error
		// instead of checking minimum constraint and returning Invalid error.
		"gang minCount zero": {
			input:        mkValidPodGroup(setPodGroupMinCount(0)),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "schedulingPolicy", "gang", "minCount"), "")},
		},
		"gang minCount negative": {
			input:        mkValidPodGroup(setPodGroupMinCount(-1)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy", "gang", "minCount"), nil, "").WithOrigin("minimum")},
		},
		"no podGroupTemplateRef": {
			input: mkValidPodGroup(unsetPodGroupTemplateRef()),
		},
		"empty podGroupTemplateRef": {
			input:        mkValidPodGroup(setEmptyPodGroupTemplateRef()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "").WithOrigin("union")},
		},
		"podGroupTemplateRef with empty template name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("", "workload")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplateRef", "workload", "podGroupTemplateName"), "")},
		},
		"podGroupTemplateRef invalid template name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("temp/late", "workload")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workload", "podGroupTemplateName"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"podGroupTemplateRef with empty workload name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("template", "")),
			expectedErrs: field.ErrorList{field.Required(field.NewPath("spec", "podGroupTemplateRef", "workload", "workloadName"), "")},
		},
		"podGroupTemplateRef invalid workload name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("template", "work/load")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workload", "workloadName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"podGroupTemplateRef too long workload name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef("template", strings.Repeat("g", 254))),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workload", "workloadName"), nil, "").WithOrigin("format=k8s-long-name")},
		},
		"podGroupTemplateRef too long template name": {
			input:        mkValidPodGroup(setPodGroupTemplateRef(strings.Repeat("g", 254), "workload")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "podGroupTemplateRef", "workload", "podGroupTemplateName"), nil, "").WithOrigin("format=k8s-short-name")},
		},
		"policy with neither basic nor gang": {
			input:        mkValidPodGroup(clearPodGroupPolicy()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"policy with both basic and gang": {
			input:        mkValidPodGroup(setBothPolicies()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("union")},
		},
		"with schedulingConstraints and TAS disabled": {
			input:        mkValidPodGroup(addTopologyConstraint("foo")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "schedulingConstraints"), "")},
		},
		"valid with schedulingConstraints": {
			input:                         mkValidPodGroup(addTopologyConstraint("foo")),
			enableTopologyAwareScheduling: true,
		},
		"schedulingConstraints with multiple topology constraints": {
			input:                         mkValidPodGroup(addTopologyConstraint("foo"), addTopologyConstraint("bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.TooMany(field.NewPath("spec", "schedulingConstraints", "topology"), 2, 1).WithOrigin("maxItems")},
		},
		"valid with empty schedulingConstraints": {
			input:                         mkValidPodGroup(setSchedulingConstraints()),
			enableTopologyAwareScheduling: true,
		},
		"topologyConstraint with empty topology key": {
			input:                         mkValidPodGroup(addTopologyConstraint("")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "schedulingConstraints", "topology").Index(0).Child("key"), "")},
		},
		"valid with topology key with DNS prefix": {
			input:                         mkValidPodGroup(addTopologyConstraint("example.com/Foo")),
			enableTopologyAwareScheduling: true,
		},
		"valid with topology key with prefix with max length": {
			input:                         mkValidPodGroup(addTopologyConstraint(strings.Repeat("a", 253) + "/" + strings.Repeat("b", 63))),
			enableTopologyAwareScheduling: true,
		},
		"with topology key with prefix exceending max prefix length": {
			input:                         mkValidPodGroup(addTopologyConstraint(strings.Repeat("a", 254) + "/foo")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key with prefix exceending max name length": {
			input:                         mkValidPodGroup(addTopologyConstraint("foo/" + strings.Repeat("b", 64))),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key without prefix exceeding max length": {
			input:                         mkValidPodGroup(addTopologyConstraint(strings.Repeat("b", 64))),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"with topology key with invalid characters": {
			input:                         mkValidPodGroup(addTopologyConstraint("Example.com/Foo")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingConstraints", "topology").Index(0).Child("key"), nil, "").WithOrigin("format=k8s-label-key")},
		},
		"pod disruption mode, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePod)),
			enableWorkloadAwarePreemption: true,
		},
		"pod disruption mode, workload aware preemption disabled": {
			input:        mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePod)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "disruptionMode"), "")},
		},
		"pod group disruption mode, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePodGroup)),
			enableWorkloadAwarePreemption: true,
		},
		"pod group disruption mode, workload aware preemption disabled": {
			input:        mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePodGroup)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "disruptionMode"), "")},
		},
		"invalid disruption mode, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode("Invalid")),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.NotSupported(field.NewPath("spec", "disruptionMode"), "Invalid", sets.List(allowedDisruptionModes))},
		},
		"invalid disruption mode, workload aware preemption disabled": {
			input:        mkValidPodGroup(setDisruptionMode("Invalid")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "disruptionMode"), "")},
		},
		"valid pod group without disruption mode, workload aware preemption enabled": {
			input:                         mkValidPodGroup(),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.Required(field.NewPath("spec", "disruptionMode"), "")},
		},
		"valid priority class name, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePod), setPriorityClassName("high-priority")),
			enableWorkloadAwarePreemption: true,
		},
		"valid priority class name, workload aware preemption disabled": {
			input:        mkValidPodGroup(setPriorityClassName("high-priority")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "priorityClassName"), "")},
		},
		"invalid priority class name, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePod), setPriorityClassName("high/priority")),
			enableWorkloadAwarePreemption: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "priorityClassName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"invalid priority class name, workload aware preemption disabled": {
			input:        mkValidPodGroup(setPriorityClassName("high/priority")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "priorityClassName"), "")},
		},
		"valid priority, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePod), setPriority(1000)),
			enableWorkloadAwarePreemption: true,
		},
		"valid priority, workload aware preemption disabled": {
			input:        mkValidPodGroup(setPriority(1000)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "priority"), "")},
		},
		"valid negative priority, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePod), setPriority(-2147483648)),
			enableWorkloadAwarePreemption: true,
		},
		"valid negative priority, workload aware preemption disabled": {
			input:        mkValidPodGroup(setPriority(-2147483648)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "priority"), "")},
		},
		"too high priority, workload aware preemption enabled": {
			input:                         mkValidPodGroup(setDisruptionMode(scheduling.DisruptionModePod), setPriority(scheduling.HighestUserDefinablePriority+1)),
			enableWorkloadAwarePreemption: true,
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "priority"), nil, "").WithOrigin("maximum"),
			},
		},
		"too high priority, workload aware preemption disabled": {
			input:        mkValidPodGroup(setPriority(scheduling.HighestUserDefinablePriority + 1)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "priority"), "")},
		},
		"ok resourceClaimName reference": {
			input: mkValidPodGroup(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimName: new("resource-claim")})),
		},
		"ok resourceClaimTemplateName reference": {
			input: mkValidPodGroup(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "claim", ResourceClaimTemplateName: new("resource-claim-template")})),
		},
		"ok multiple claims": {
			input: mkValidPodGroup(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "claim-1", ResourceClaimName: new("resource-claim-1")},
				scheduling.PodGroupResourceClaim{Name: "claim-2", ResourceClaimName: new("resource-claim-2")},
			)),
		},
		"claim name with prefix": {
			input: mkValidPodGroup(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "../my-claim", ResourceClaimName: new("resource-claim")})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"claim name with path": {
			input: mkValidPodGroup(addResourceClaims(scheduling.PodGroupResourceClaim{Name: "my/claim", ResourceClaimName: new("resource-claim")})),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "resourceClaims").Index(0).Child("name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"duplicate claim entries": {
			input: mkValidPodGroup(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new("resource-claim-1")},
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new("resource-claim-2")},
			)),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec", "resourceClaims").Index(1), nil),
			},
		},
		"resource claim source empty": {
			input: mkValidPodGroup(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim"},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"resource claim reference and template": {
			input: mkValidPodGroup(addResourceClaims(
				scheduling.PodGroupResourceClaim{
					Name:                      "my-claim",
					ResourceClaimName:         new("resource-claim"),
					ResourceClaimTemplateName: new("resource-claim-template"),
				},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "resourceClaims").Index(0), nil, "").WithOrigin("union"),
			},
		},
		"invalid claim reference name": {
			input: mkValidPodGroup(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimName: new(".foo_bar")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "resourceClaims").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"invalid claim template name": {
			input: mkValidPodGroup(addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new(".foo_bar")},
			)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "resourceClaims").Index(0).Child("resourceClaimTemplateName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"too many claims": {
			input: mkValidPodGroup(func(pg *scheduling.PodGroup) {
				pg.Spec.ResourceClaims = make([]scheduling.PodGroupResourceClaim, scheduling.MaxPodGroupResourceClaims+1)
				for i := range pg.Spec.ResourceClaims {
					pg.Spec.ResourceClaims[i] = scheduling.PodGroupResourceClaim{
						Name:              "my-claim-" + strconv.Itoa(i),
						ResourceClaimName: new("resource-claim"),
					}
				}
			}),
			expectedErrs: field.ErrorList{
				field.TooMany(field.NewPath("spec", "resourceClaims"), scheduling.MaxPodGroupResourceClaims+1, scheduling.MaxPodGroupResourceClaims).WithOrigin("maxItems"),
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
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, strategy, tc.expectedErrs, apitesting.WithMinEmulationVersion(version.MustParse("1.36")))
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // PodGroup is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "podgroups",
		Name:              "valid-podgroup",
		IsResourceRequest: true,
		Verb:              "update",
	})
	testCases := map[string]struct {
		oldObj                          scheduling.PodGroup
		updateObj                       scheduling.PodGroup
		enableTopologyAwareScheduling   bool
		enableDRAWorkloadResourceClaims bool
		enableWorkloadAwarePreemption   bool
		expectedErrs                    field.ErrorList
	}{
		"valid update": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1")),
		},
		"invalid update empty podGroupTemplateRef": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setEmptyPodGroupTemplateRef()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update unset podGroupTemplateRef": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), unsetPodGroupTemplateRef()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update change podGroupTemplateRef template name": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setPodGroupTemplateRef("other-template", "workload")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update change podGroupTemplateRef workload name": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setPodGroupTemplateRef("template", "other-workload")),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update with neither basic nor gang": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), clearPodGroupPolicy()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update with both basic and gang": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setBothPolicies()),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update of gang minCount": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), setPodGroupMinCount(10)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("immutable"),
			},
		},
		"invalid update from gang to basic policy": {
			oldObj:       mkValidPodGroup(setResourceVersion("1")),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), setBasicPolicy()),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingPolicy"), nil, "").WithOrigin("immutable")},
		},
		"valid update with unchanged scheduling constraints and TAS disabled": {
			oldObj:    mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
		},
		"valid update with unchanged scheduling constraints": {
			oldObj:                        mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj:                     mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			enableTopologyAwareScheduling: true,
		},
		"invalid update to scheduling constraints": {
			oldObj:                        mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj:                     mkValidPodGroup(setResourceVersion("1"), setSchedulingConstraints()),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid update to topology constraints": {
			oldObj:                        mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj:                     mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo"), addTopologyConstraint("bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid update to topology key": {
			oldObj:                        mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj:                     mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("bar")),
			enableTopologyAwareScheduling: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "schedulingConstraints"), nil, "field is immutable").WithOrigin("immutable")},
		},
		"invalid update to scheduling constraints with TAS disabled": {
			oldObj:       mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), setSchedulingConstraints()),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "schedulingConstraints"), "")},
		},
		"invalid update to topology constraints with TAS disabled": {
			oldObj:       mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo"), addTopologyConstraint("bar")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "schedulingConstraints"), "")},
		},
		"invalid update to topology key with TAS disabled": {
			oldObj:       mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("foo")),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), addTopologyConstraint("bar")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "schedulingConstraints"), "")},
		},
		"invalid add of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "resourceClaims"), nil, "").WithOrigin("immutable")},
		},
		"invalid add of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "resourceClaims"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "resourceClaims"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "resourceClaims"), nil, "").WithOrigin("immutable")},
		},
		"invalid remove of resource claims, DRA workload resource claims disabled": {
			oldObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj:    mkValidPodGroup(setResourceVersion("1")),
			expectedErrs: field.ErrorList{field.Invalid(field.NewPath("spec", "resourceClaims"), nil, "").WithOrigin("immutable")},
		},
		"invalid remove of resource claims, DRA workload resource claims enabled": {
			oldObj: mkValidPodGroup(setResourceVersion("1"), addResourceClaims(
				scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
			)),
			updateObj:                       mkValidPodGroup(setResourceVersion("1")),
			enableDRAWorkloadResourceClaims: true,
			expectedErrs:                    field.ErrorList{field.Invalid(field.NewPath("spec", "resourceClaims"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of disruption mode, workload aware preemption enabled": {
			oldObj:                        mkValidPodGroup(setResourceVersion("1"), setDisruptionMode(scheduling.DisruptionModePod)),
			updateObj:                     mkValidPodGroup(setResourceVersion("1"), setDisruptionMode(scheduling.DisruptionModePodGroup)),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "disruptionMode"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of disruption mode, workload aware preemption disabled": {
			oldObj:       mkValidPodGroup(setResourceVersion("1"), setDisruptionMode(scheduling.DisruptionModePod)),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), setDisruptionMode(scheduling.DisruptionModePodGroup)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "disruptionMode"), "")},
		},
		"invalid update of priority class name, workload aware preemption enabled": {
			oldObj:                        mkValidPodGroup(setResourceVersion("1"), setPriorityClassName("low-priority")),
			updateObj:                     mkValidPodGroup(setResourceVersion("1"), setPriorityClassName("high-priority")),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "priorityClassName"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of priority class name, workload aware preemption disabled": {
			oldObj:       mkValidPodGroup(setResourceVersion("1"), setPriorityClassName("low-priority")),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), setPriorityClassName("high-priority")),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "priorityClassName"), "")},
		},
		"invalid update of priority, workload aware preemption enabled": {
			oldObj:                        mkValidPodGroup(setResourceVersion("1"), setPriority(1000)),
			updateObj:                     mkValidPodGroup(setResourceVersion("1"), setPriority(2000)),
			enableWorkloadAwarePreemption: true,
			expectedErrs:                  field.ErrorList{field.Invalid(field.NewPath("spec", "priority"), nil, "").WithOrigin("immutable")},
		},
		"invalid update of priority, workload aware preemption disabled": {
			oldObj:       mkValidPodGroup(setResourceVersion("1"), setPriority(1000)),
			updateObj:    mkValidPodGroup(setResourceVersion("1"), setPriority(2000)),
			expectedErrs: field.ErrorList{field.Forbidden(field.NewPath("spec", "priority"), "")},
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
			strategy := NewStrategy()
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, strategy, tc.expectedErrs, apitesting.WithMinEmulationVersion(version.MustParse("1.36")))
		})
	}
}

func TestDeclarativeValidateStatusUpdate(t *testing.T) {
	apiVersions := []string{"v1alpha2"} // PodGroup is currently only in v1alpha2
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateStatusUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidateStatusUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:   "apis",
		APIGroup:    "scheduling.k8s.io",
		APIVersion:  apiVersion,
		Resource:    "podgroups",
		Subresource: "status",
		Verb:        "update",
	})
	testCases := map[string]struct {
		oldObj       scheduling.PodGroup
		updateObj    scheduling.PodGroup
		expectedErrs field.ErrorList
	}{
		"valid noop update": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1")),
		},
		"valid status update": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addCondition(scheduling.PodGroupScheduled)),
		},
		"duplicate condition types": {
			oldObj:    mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(setResourceVersion("1"), addCondition(scheduling.PodGroupScheduled), addCondition(scheduling.PodGroupScheduled)),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("status", "conditions").Index(1).Child("type"), scheduling.PodGroupScheduled).MarkFromImperative(),
			},
		},
		"valid resource claim status update": {
			oldObj: mkValidPodGroup(
				setResourceVersion("1"),
				addResourceClaims(
					scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
					scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
				),
			),
			updateObj: mkValidPodGroup(
				setResourceVersion("1"),
				addResourceClaims(
					scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
					scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
				),
				addResourceClaimStatuses(
					scheduling.PodGroupResourceClaimStatus{Name: "my-claim", ResourceClaimName: new("foo-my-claim-12345")},
					scheduling.PodGroupResourceClaimStatus{Name: "my-other-claim", ResourceClaimName: nil},
				),
			),
		},
		"non-existent resource claim in status": {
			oldObj: mkValidPodGroup(setResourceVersion("1")),
			updateObj: mkValidPodGroup(
				setResourceVersion("1"),
				addResourceClaimStatuses(
					scheduling.PodGroupResourceClaimStatus{Name: "no-such-claim", ResourceClaimName: new("my-template")},
				),
			),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "resourceClaimStatuses").Index(0).Child("name"), nil, "").MarkFromImperative(),
			},
		},
		"invalid resource claim name": {
			oldObj: mkValidPodGroup(
				setResourceVersion("1"),
				addResourceClaims(
					scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
				),
			),
			updateObj: mkValidPodGroup(
				setResourceVersion("1"),
				addResourceClaims(
					scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
				),
				addResourceClaimStatuses(
					scheduling.PodGroupResourceClaimStatus{Name: "my-claim", ResourceClaimName: new("%$!#5")},
				),
			),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("status", "resourceClaimStatuses").Index(0).Child("resourceClaimName"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"duplicate claim name": {
			oldObj: mkValidPodGroup(
				setResourceVersion("1"),
				addResourceClaims(
					scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
					scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
				),
			),
			updateObj: mkValidPodGroup(
				setResourceVersion("1"),
				addResourceClaims(
					scheduling.PodGroupResourceClaim{Name: "my-claim", ResourceClaimTemplateName: new("my-template")},
					scheduling.PodGroupResourceClaim{Name: "my-other-claim", ResourceClaimTemplateName: new("my-template")},
				),
				addResourceClaimStatuses(
					scheduling.PodGroupResourceClaimStatus{Name: "my-claim", ResourceClaimName: new("foo-my-claim-12345")},
					scheduling.PodGroupResourceClaimStatus{Name: "my-other-claim", ResourceClaimName: nil},
					scheduling.PodGroupResourceClaimStatus{Name: "my-other-claim", ResourceClaimName: nil},
				),
			),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("status", "resourceClaimStatuses").Index(2), nil),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			strategy := NewStatusStrategy(NewStrategy())
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, strategy, tc.expectedErrs)
		})
	}
}

// mkValidPodGroup produces a PodGroup which passes validation with no tweaks.
func mkValidPodGroup(tweaks ...func(pg *scheduling.PodGroup)) scheduling.PodGroup {
	obj := scheduling.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "podgroup", Namespace: "ns"},
		Spec: scheduling.PodGroupSpec{
			PodGroupTemplateRef: &scheduling.PodGroupTemplateReference{
				Workload: &scheduling.WorkloadPodGroupTemplateReference{
					WorkloadName:         "workload",
					PodGroupTemplateName: "template",
				},
			},
			SchedulingPolicy: scheduling.PodGroupSchedulingPolicy{
				Gang: &scheduling.GangSchedulingPolicy{
					MinCount: 5,
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func setResourceVersion(v string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.ResourceVersion = v
	}
}

func unsetPodGroupTemplateRef() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.PodGroupTemplateRef = nil
	}
}

func setEmptyPodGroupTemplateRef() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{}
	}
}

func setPodGroupMinCount(min int) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy.Gang.MinCount = int32(min)
	}
}

func clearPodGroupPolicy() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{}
	}
}

func setBasicPolicy() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
		}
	}
}

func setBothPolicies() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingPolicy = scheduling.PodGroupSchedulingPolicy{
			Basic: &scheduling.BasicSchedulingPolicy{},
			Gang:  &scheduling.GangSchedulingPolicy{MinCount: 1},
		}
	}
}

func setPodGroupTemplateRef(templateName, workloadName string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.PodGroupTemplateRef = &scheduling.PodGroupTemplateReference{
			Workload: &scheduling.WorkloadPodGroupTemplateReference{
				PodGroupTemplateName: templateName,
				WorkloadName:         workloadName,
			},
		}
	}
}

func addResourceClaims(claims ...scheduling.PodGroupResourceClaim) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.ResourceClaims = append(obj.Spec.ResourceClaims, claims...)
	}
}

func addResourceClaimStatuses(statuses ...scheduling.PodGroupResourceClaimStatus) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Status.ResourceClaimStatuses = append(obj.Status.ResourceClaimStatuses, statuses...)
	}
}

func addCondition(conditionType string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Status.Conditions = append(obj.Status.Conditions, metav1.Condition{
			Type:               conditionType,
			Status:             metav1.ConditionFalse,
			Reason:             scheduling.PodGroupScheduled,
			Message:            "Test status condition message",
			LastTransitionTime: metav1.Now(),
		})
	}
}

func setSchedulingConstraints() func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.SchedulingConstraints = &scheduling.PodGroupSchedulingConstraints{}
	}
}

func addTopologyConstraint(value string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		if obj.Spec.SchedulingConstraints == nil {
			setSchedulingConstraints()(obj)
		}
		obj.Spec.SchedulingConstraints.Topology = append(obj.Spec.SchedulingConstraints.Topology, scheduling.TopologyConstraint{Key: value})
	}
}

func setDisruptionMode(mode scheduling.DisruptionMode) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.DisruptionMode = &mode
	}
}

func setPriorityClassName(priorityClassName string) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.PriorityClassName = priorityClassName
	}
}

func setPriority(priority int32) func(obj *scheduling.PodGroup) {
	return func(obj *scheduling.PodGroup) {
		obj.Spec.Priority = new(priority)
	}
}
