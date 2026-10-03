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

package validation

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

// TestMapV1ToV1beta1ErrorLists covers the filter that lets the cross-version validation
// equivalence sweep compare authorization.k8s.io/v1 against v1beta1 even though v1 has
// fields v1beta1 cannot represent.
//
// The filter drops errors, so an over-broad match would silently hide a real divergence.
// These cases pin both directions: what must be excused, and what must survive.
func TestMapV1ToV1beta1ErrorLists(t *testing.T) {
	// Paths that only v1 has, so v1beta1 can never report anything under them.
	v1OnlyErrs := field.ErrorList{
		field.Forbidden(field.NewPath("spec", "authorizationOptions"), "gate disabled"),
		field.Required(field.NewPath("spec", "authorizationOptions", "handledDecisionTypes"), ""),
		field.Duplicate(field.NewPath("spec", "authorizationOptions", "handledDecisionTypes").Index(1), "Allow"),
		field.Forbidden(field.NewPath("status", "conditionalDecision"), "gate disabled"),
		field.Required(field.NewPath("status", "conditionalDecision", "type"), ""),
		field.Invalid(field.NewPath("status", "conditionalDecision", "conditionsMap", "denyConditions").Index(0).Child("id"), "x", "bad"),
	}
	// Paths both versions have, which must always be compared.
	sharedErrs := field.ErrorList{
		field.Invalid(field.NewPath("spec", "user"), "", "at least one of user or group must be specified"),
		field.Invalid(field.NewPath("status"), true, "allowed and denied are mutually exclusive"),
		field.Invalid(field.NewPath("metadata"), "", "must be empty"),
		// Siblings that merely share a prefix with a v1-only path must not be swallowed.
		field.Invalid(field.NewPath("status", "conditionalDecisions"), "", "not the v1-only field"),
		field.Invalid(field.NewPath("spec", "authorizationOptionsExtra"), "", "not the v1-only field"),
	}

	tests := []struct {
		name            string
		gvLeft, gvRight string
		left, right     field.ErrorList
		wantLeft        field.ErrorList
		wantRight       field.ErrorList
	}{
		{
			name:    "v1 on the left has its v1-only errors dropped",
			gvLeft:  "authorization.k8s.io/v1",
			gvRight: "authorization.k8s.io/v1beta1",
			left:    append(append(field.ErrorList{}, sharedErrs...), v1OnlyErrs...),
			right:   sharedErrs,
			// Only the shared errors remain, in their original order.
			wantLeft:  sharedErrs,
			wantRight: sharedErrs,
		},
		{
			name:      "v1 on the right has its v1-only errors dropped",
			gvLeft:    "authorization.k8s.io/v1beta1",
			gvRight:   "authorization.k8s.io/v1",
			left:      sharedErrs,
			right:     append(append(field.ErrorList{}, sharedErrs...), v1OnlyErrs...),
			wantLeft:  sharedErrs,
			wantRight: sharedErrs,
		},
		{
			// A v1beta1 error under a v1-only path would mean the conversion started
			// carrying the field. It must survive so the sweep reports the mismatch.
			name:      "a v1beta1 error under a v1-only path is not dropped",
			gvLeft:    "authorization.k8s.io/v1beta1",
			gvRight:   "authorization.k8s.io/v1",
			left:      v1OnlyErrs,
			right:     field.ErrorList{},
			wantLeft:  v1OnlyErrs,
			wantRight: field.ErrorList{},
		},
		{
			name:      "a pair that does not involve v1beta1 is untouched",
			gvLeft:    "authorization.k8s.io/v1",
			gvRight:   "authorization.k8s.io/v1alpha1",
			left:      append(append(field.ErrorList{}, sharedErrs...), v1OnlyErrs...),
			right:     sharedErrs,
			wantLeft:  append(append(field.ErrorList{}, sharedErrs...), v1OnlyErrs...),
			wantRight: sharedErrs,
		},
		{
			name:      "a pair from another group is untouched",
			gvLeft:    "apps/v1",
			gvRight:   "apps/v1beta1",
			left:      v1OnlyErrs,
			right:     sharedErrs,
			wantLeft:  v1OnlyErrs,
			wantRight: sharedErrs,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotLeft, gotRight := MapV1ToV1beta1ErrorLists(tc.gvLeft, tc.gvRight, tc.left, tc.right)

			if diff := cmp.Diff(errStrings(tc.wantLeft), errStrings(gotLeft)); diff != "" {
				t.Errorf("unexpected left list (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(errStrings(tc.wantRight), errStrings(gotRight)); diff != "" {
				t.Errorf("unexpected right list (-want +got):\n%s", diff)
			}
		})
	}
}

func errStrings(errs field.ErrorList) []string {
	var out []string
	for _, e := range errs {
		out = append(out, e.Error())
	}
	return out
}
