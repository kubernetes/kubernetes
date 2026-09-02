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

package priorityclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	core "k8s.io/kubernetes/pkg/apis/core"
	scheduling "k8s.io/kubernetes/pkg/apis/scheduling"
	registry "k8s.io/kubernetes/pkg/registry/scheduling/priorityclass"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "priorityclasses",
		IsResourceRequest: true,
		Verb:              "create",
	})

	obj := mkPriorityClass()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "scheduling.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "priorityclasses",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	})

	testCases := map[string]struct {
		oldObj       scheduling.PriorityClass
		updateObj    scheduling.PriorityClass
		expectedErrs field.ErrorList
	}{
		"valid update": {
			oldObj:    mkPriorityClass(TweakValue(100)),
			updateObj: mkPriorityClass(TweakValue(100)),
		},
		"invalid update value changed": {
			oldObj:    mkPriorityClass(TweakValue(100)),
			updateObj: mkPriorityClass(TweakValue(101)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update value set from unset": {
			oldObj:    mkPriorityClass(),
			updateObj: mkPriorityClass(TweakValue(100)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update value unset from set": {
			oldObj:    mkPriorityClass(TweakValue(100)),
			updateObj: mkPriorityClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("value"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update preemptionPolicy changed": {
			oldObj:    mkPriorityClass(TweakPreemptionPolicy(core.PreemptLowerPriority)),
			updateObj: mkPriorityClass(TweakPreemptionPolicy(core.PreemptNever)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("preemptionPolicy"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update preemptionPolicy set from unset": {
			oldObj:    mkPriorityClass(),
			updateObj: mkPriorityClass(TweakPreemptionPolicy(core.PreemptLowerPriority)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("preemptionPolicy"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
		"invalid update preemptionPolicy unset from set": {
			oldObj:    mkPriorityClass(TweakPreemptionPolicy(core.PreemptLowerPriority)),
			updateObj: mkPriorityClass(),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("preemptionPolicy"), nil, "").WithOrigin("immutable").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.oldObj.ResourceVersion = "1"
			tc.updateObj.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.updateObj, &tc.oldObj, registry.Strategy, tc.expectedErrs)
		})
	}

	updateObj := mkPriorityClass()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func mkPriorityClass(tweaks ...func(pc *scheduling.PriorityClass)) scheduling.PriorityClass {
	pc := scheduling.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-obj",
		},
	}
	for _, tweak := range tweaks {
		tweak(&pc)
	}
	return pc
}

func TweakValue(value int32) func(pc *scheduling.PriorityClass) {
	return func(pc *scheduling.PriorityClass) {
		pc.Value = value
	}
}

func TweakPreemptionPolicy(policy core.PreemptionPolicy) func(pc *scheduling.PriorityClass) {
	return func(pc *scheduling.PriorityClass) {
		pc.PreemptionPolicy = &policy
	}
}
