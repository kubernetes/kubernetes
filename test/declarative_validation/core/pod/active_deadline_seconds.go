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
	"context"
	"math"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/utils/ptr"
)

// RunDeclarativeValidateActiveDeadlineSecondsTestCases exercises the shared
// spec.activeDeadlineSeconds minimum/maximum declarative validation rules
// across any type that embeds a PodSpec. This gives each embedding Kind
// (e.g. ReplicationController, DaemonSet, Deployment, ...) coverage for the
// rule without duplicating the same test cases in every package.
func RunDeclarativeValidateActiveDeadlineSecondsTestCases[T runtime.Object](t *testing.T, ctx context.Context, strategy rest.RESTCreateStrategy, specPath *field.Path, baseObj T, setActiveDeadlineSeconds func(baseObj T, val *int64)) {
	testCases := map[string]struct {
		input        *int64
		expectedErrs field.ErrorList
	}{
		"unset": {
			input: nil,
		},
		"positive": {
			input: ptr.To(int64(100)),
		},
		"max": {
			input: ptr.To(int64(math.MaxInt32)),
		},
		"zero": {
			input: ptr.To(int64(0)),
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("activeDeadlineSeconds"), nil, "").WithOrigin("minimum").MarkAlpha(),
			},
		},
		"negative": {
			input: ptr.To(int64(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("activeDeadlineSeconds"), nil, "").WithOrigin("minimum").MarkAlpha(),
			},
		},
		"over maximum": {
			input: ptr.To(int64(math.MaxInt32 + 1)),
			expectedErrs: field.ErrorList{
				field.Invalid(specPath.Child("activeDeadlineSeconds"), nil, "").WithOrigin("maximum").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run("activeDeadlineSeconds: "+k, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			setActiveDeadlineSeconds(obj, tc.input)
			apitesting.VerifyValidationEquivalence(t, ctx, obj, strategy, tc.expectedErrs)
		})
	}
}

// RunDeclarativeValidateActiveDeadlineSecondsForbiddenTestCases exercises the
// hand-written (non-declarative) rule that forbids spec.activeDeadlineSeconds
// on PodSpecs embedded in controllers with an always-restarting pod template
// (e.g. ReplicationController, DaemonSet, ReplicaSet, StatefulSet). This rule
// depends on a sibling field's value and isn't expressible as a declarative
// tag today, so it stays purely imperative. The declarative minimum/maximum
// shadow-validation still runs independently, so boundary-violating values
// produce both the declarative error and the imperative Forbidden error.
func RunDeclarativeValidateActiveDeadlineSecondsForbiddenTestCases[T runtime.Object](t *testing.T, ctx context.Context, strategy rest.RESTCreateStrategy, specPath *field.Path, baseObj T, setActiveDeadlineSeconds func(baseObj T, val *int64)) {
	forbidden := field.Forbidden(specPath.Child("activeDeadlineSeconds"), "").MarkFromImperative()
	testCases := map[string]struct {
		input        *int64
		expectedErrs field.ErrorList
	}{
		"unset": {
			input: nil,
		},
		"positive": {
			input: ptr.To(int64(100)),
			expectedErrs: field.ErrorList{
				forbidden,
			},
		},
		"max": {
			input: ptr.To(int64(math.MaxInt32)),
			expectedErrs: field.ErrorList{
				forbidden,
			},
		},
		"zero": {
			input: ptr.To(int64(0)),
			expectedErrs: field.ErrorList{
				forbidden,
				field.Invalid(specPath.Child("activeDeadlineSeconds"), nil, "").WithOrigin("minimum").MarkAlpha(),
			},
		},
		"negative": {
			input: ptr.To(int64(-1)),
			expectedErrs: field.ErrorList{
				forbidden,
				field.Invalid(specPath.Child("activeDeadlineSeconds"), nil, "").WithOrigin("minimum").MarkAlpha(),
			},
		},
		"over maximum": {
			input: ptr.To(int64(math.MaxInt32 + 1)),
			expectedErrs: field.ErrorList{
				forbidden,
				field.Invalid(specPath.Child("activeDeadlineSeconds"), nil, "").WithOrigin("maximum").MarkAlpha(),
			},
		},
	}
	for k, tc := range testCases {
		t.Run("activeDeadlineSeconds: "+k, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			setActiveDeadlineSeconds(obj, tc.input)
			apitesting.VerifyValidationEquivalence(t, ctx, obj, strategy, tc.expectedErrs)
		})
	}
}
