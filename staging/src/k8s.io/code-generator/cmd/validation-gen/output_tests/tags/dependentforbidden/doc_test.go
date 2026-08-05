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

package dependentforbidden

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// One-directional: dependent alone is fine.
	st.Value(&Struct{Dependent: new("d")}).ExpectValid()

	// Trigger alone is fine.
	st.Value(&Struct{Trigger: new("t")}).ExpectValid()

	// Both set → forbidden.
	st.Value(&Struct{Trigger: new("t"), Dependent: new("d")}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Forbidden(field.NewPath("dependent"), "").WithOrigin("dependentForbidden"),
		},
	)

	// Ratchet: unrelated field changed, trigger and dependent set-ness unchanged → skip.
	st.Value(&Struct{Trigger: new("t"), Dependent: new("d"), OtherField: new("new")}).
		OldValue(&Struct{Trigger: new("t"), Dependent: new("d"), OtherField: new("old")}).
		ExpectValid()

	// Ratchet: trigger value changed but set-ness unchanged → skip.
	st.Value(&Struct{Trigger: new("t"), Dependent: new("d")}).
		OldValue(&Struct{Trigger: new("old"), Dependent: new("d")}).
		ExpectValid()

	// Newly set dependent → fire.
	st.Value(&Struct{Trigger: new("t"), Dependent: new("d")}).
		OldValue(&Struct{Trigger: new("t")}).
		ExpectMatches(
			field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
			field.ErrorList{
				field.Forbidden(field.NewPath("dependent"), "").WithOrigin("dependentForbidden"),
			},
		)
}

func TestMultiDependent(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Repeated tags → independent implications, each at its own dependent path.
	st.Value(&MultiDependent{Trigger: new("t"), DependentA: new("a"), DependentB: new("b")}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Forbidden(field.NewPath("dependentA"), "").WithOrigin("dependentForbidden"),
			field.Forbidden(field.NewPath("dependentB"), "").WithOrigin("dependentForbidden"),
		},
	)
}

func TestMultiTrigger(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Distinct triggers → independent implications at the same path.
	// ByOrigin absorbs both actuals.
	st.Value(&MultiTrigger{TriggerA: new("a"), TriggerB: new("b"), Dependent: new("d")}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Forbidden(field.NewPath("dependent"), "").WithOrigin("dependentForbidden"),
		},
	)
}

func TestAllKinds(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Each kind's extractor fires when both trigger and dependent are set.
	st.Value(&AllKinds{
		PtrTrigger:   new("t"),
		PtrDep:       new("d"),
		SliceTrigger: []string{"x"},
		SliceDep:     []string{"y"},
		MapTrigger:   map[string]string{"k": "v"},
		MapDep:       map[string]string{"k": "v"},
		IntTrigger:   1,
		IntDep:       1,
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Forbidden(field.NewPath("ptrDep"), "").WithOrigin("dependentForbidden"),
			field.Forbidden(field.NewPath("sliceDep"), "").WithOrigin("dependentForbidden"),
			field.Forbidden(field.NewPath("mapDep"), "").WithOrigin("dependentForbidden"),
			field.Forbidden(field.NewPath("intDep"), "").WithOrigin("dependentForbidden"),
		},
	)

	// Triggers set but dependents "not set" (empty slice/map, zero int, nil ptr) → valid.
	st.Value(&AllKinds{
		PtrTrigger:   new("t"),
		SliceTrigger: []string{"x"},
		MapTrigger:   map[string]string{"k": "v"},
		IntTrigger:   1,
		SliceDep:     []string{},
		MapDep:       map[string]string{},
		IntDep:       0,
	}).ExpectValid()
}
