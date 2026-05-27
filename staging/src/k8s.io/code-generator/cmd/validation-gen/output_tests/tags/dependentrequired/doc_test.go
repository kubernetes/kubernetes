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

package dependentrequired

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestStruct(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// One-directional: dependent alone is fine.
	st.Value(&Struct{Dependent: ptr.To("d")}).ExpectValid()

	st.Value(&Struct{Trigger: ptr.To("t"), Dependent: ptr.To("d")}).ExpectValid()

	st.Value(&Struct{Trigger: ptr.To("t")}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Required(field.NewPath("dependent"), "").WithOrigin("dependentRequired"),
		},
	)

	// Ratchet: unrelated field changed, trigger and dependent set-ness unchanged → skip.
	st.Value(&Struct{Trigger: ptr.To("t"), OtherField: ptr.To("new")}).
		OldValue(&Struct{Trigger: ptr.To("t"), OtherField: ptr.To("old")}).
		ExpectValid()

	// Ratchet: trigger value changed but set-ness unchanged → skip (same rationale as union).
	st.Value(&Struct{Trigger: ptr.To("t")}).
		OldValue(&Struct{Trigger: ptr.To("old")}).
		ExpectValid()

	// Newly cleared dependent → fire.
	st.Value(&Struct{Trigger: ptr.To("t")}).
		OldValue(&Struct{Trigger: ptr.To("t"), Dependent: ptr.To("d")}).
		ExpectMatches(
			field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
			field.ErrorList{
				field.Required(field.NewPath("dependent"), "").WithOrigin("dependentRequired"),
			},
		)
}

func TestMultiDependent(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Repeated tags → independent implications, each at its own dependent path.
	st.Value(&MultiDependent{Trigger: ptr.To("t")}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Required(field.NewPath("dependentA"), "").WithOrigin("dependentRequired"),
			field.Required(field.NewPath("dependentB"), "").WithOrigin("dependentRequired"),
		},
	)
}

func TestMultiTrigger(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Distinct triggers → independent implications at the same path.
	// ByOrigin absorbs both actuals.
	st.Value(&MultiTrigger{TriggerA: ptr.To("a"), TriggerB: ptr.To("b")}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Required(field.NewPath("dependent"), "").WithOrigin("dependentRequired"),
		},
	)
}

func TestAllKinds(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Each kind's extractor fires.
	st.Value(&AllKinds{
		PtrTrigger:   ptr.To("t"),
		SliceTrigger: []string{"x"},
		MapTrigger:   map[string]string{"k": "v"},
		IntTrigger:   1,
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Required(field.NewPath("ptrDep"), "").WithOrigin("dependentRequired"),
			field.Required(field.NewPath("sliceDep"), "").WithOrigin("dependentRequired"),
			field.Required(field.NewPath("mapDep"), "").WithOrigin("dependentRequired"),
			field.Required(field.NewPath("intDep"), "").WithOrigin("dependentRequired"),
		},
	)

	// Empty slice/map and zero int = "not set".
	st.Value(&AllKinds{
		SliceTrigger: []string{},
		MapTrigger:   map[string]string{},
		IntTrigger:   0,
	}).ExpectValid()
}
