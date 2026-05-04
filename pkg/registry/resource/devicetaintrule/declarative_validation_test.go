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

package devicetaintrule

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/resource"
)

var apiVersions = []string{"v1alpha3", "v1beta2"}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "resource.k8s.io",
		APIVersion: apiVersion,
		Resource:   "devicetaintrules",
	})

	testCases := map[string]struct {
		input        resource.DeviceTaintRule
		expectedErrs field.ErrorList
	}{
		"valid": {
			input: mkValidDeviceTaintRule(),
		},
		"missing taint.effect": {
			input: mkValidDeviceTaintRule(tweakTaintEffect("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "taint", "effect"), "").MarkAlpha(),
			},
		},
		"invalid taint.effect": {
			input: mkValidDeviceTaintRule(tweakTaintEffect("BadEffect")),
			expectedErrs: field.ErrorList{
				field.NotSupported(field.NewPath("spec", "taint", "effect"), resource.DeviceTaintEffect("BadEffect"), []resource.DeviceTaintEffect{}).MarkAlpha(),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy, tc.expectedErrs)
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

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "resource.k8s.io",
		APIVersion: apiVersion,
		Resource:   "devicetaintrules",
	})

	testCases := map[string]struct {
		old          resource.DeviceTaintRule
		update       resource.DeviceTaintRule
		expectedErrs field.ErrorList
	}{
		"valid update (no spec change)": {
			old:    mkValidDeviceTaintRule(),
			update: mkValidDeviceTaintRule(),
		},
		"valid update changing effect": {
			old:    mkValidDeviceTaintRule(),
			update: mkValidDeviceTaintRule(tweakTaintEffect(resource.DeviceTaintEffectNoSchedule)),
		},
		"invalid update clearing effect": {
			old:    mkValidDeviceTaintRule(),
			update: mkValidDeviceTaintRule(tweakTaintEffect("")),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec", "taint", "effect"), "").MarkAlpha(),
			},
		},
		// TODO: Add more test cases
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tc.old.ResourceVersion = "1"
			tc.update.ResourceVersion = "2"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy, tc.expectedErrs)
		})
	}
}

func mkValidDeviceTaintRule(tweaks ...func(*resource.DeviceTaintRule)) resource.DeviceTaintRule {
	rule := resource.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-rule",
		},
		Spec: resource.DeviceTaintRuleSpec{
			Taint: resource.DeviceTaint{
				Key:    "example.com/tainted",
				Effect: resource.DeviceTaintEffectNoExecute,
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&rule)
	}
	return rule
}

func tweakTaintEffect(effect resource.DeviceTaintEffect) func(*resource.DeviceTaintRule) {
	return func(r *resource.DeviceTaintRule) {
		r.Spec.Taint.Effect = effect
	}
}
